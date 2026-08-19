"""Walk-forward validation of the early-signal ENTRY THRESHOLD.

The threshold question - what should EARLY_SIGNAL_MIN_SCORE be - cannot be
answered by picking the value that looked best over all the data. That is
the same value the noise looked best at, and there is exactly one way to
find out whether the choice survives contact with data it was not chosen
on: choose it on one period and grade it on the next.

    |-- train --|-- test --|
                |-- train --|-- test --|
                            |-- train --|-- test --|

The threshold is refitted inside every train window and then applied,
untouched, to the test window that follows it. The reported number is the
average over the TEST windows only. Nothing in a test window ever
influences the threshold used on it.

WHY THIS WALKS OVER STORED ROWS AND NOT CANDLES

The rest of app/research/ walks a CandleSeries because the technical
strategy can be replayed from candles. The early engine cannot: half its
information comes from differencing market snapshots. What it CAN be
replayed from is its own recorded output - each ForwardReturn row holds
the early score as it stood at signal time and the return that followed.
Sorting those rows by observation time gives a genuine chronological
walk, which is what the method actually requires.

WHAT A RESULT HERE IS AND IS NOT

A stable threshold across test windows is evidence the level is not an
artifact of one period. It is NOT evidence the early score works - if the
score has no edge, walk-forward will faithfully report a stable absence
of edge, and that is the finding. Read `verdict()`, not the number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis.calibration import round_trip_cost_pct

# Candidate thresholds. Spaced to match the buckets the calibration table
# uses, so a walk-forward result can be read against a calibration table
# without mentally re-bucketing anything.
CANDIDATE_THRESHOLDS: tuple[float, ...] = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0)

# Below this, a window's result is a rounding artifact of a handful of
# rows rather than a measurement.
MIN_WINDOW_ROWS = 20
MIN_SELECTED_TRADES = 5


@dataclass
class Window:
    index: int
    train_rows: int
    test_rows: int
    chosen_threshold: float | None
    train_expectancy_pct: float | None
    test_expectancy_pct: float | None
    test_trades: int

    @property
    def graded(self) -> bool:
        return self.test_expectancy_pct is not None

    @property
    def gap(self) -> float | None:
        if self.train_expectancy_pct is None or self.test_expectancy_pct is None:
            return None
        return self.train_expectancy_pct - self.test_expectancy_pct

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 3) if v is not None else None
        return {
            "index": self.index,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "chosen_threshold": self.chosen_threshold,
            "train_expectancy_pct": r(self.train_expectancy_pct),
            "test_expectancy_pct": r(self.test_expectancy_pct),
            "test_trades": self.test_trades,
            "overfit_gap_pct": r(self.gap),
        }


@dataclass
class EarlyWalkForward:
    horizon_minutes: int
    total_rows: int = 0
    windows: list[Window] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def graded(self) -> list[Window]:
        return [w for w in self.windows if w.graded]

    @property
    def conclusive(self) -> bool:
        return len(self.graded) >= 3

    @property
    def mean_test_expectancy_pct(self) -> float | None:
        graded = self.graded
        if not graded:
            return None
        return sum(w.test_expectancy_pct for w in graded) / len(graded)

    @property
    def mean_gap_pct(self) -> float | None:
        gaps = [w.gap for w in self.graded if w.gap is not None]
        return sum(gaps) / len(gaps) if gaps else None

    @property
    def chosen_thresholds(self) -> list[float]:
        return [w.chosen_threshold for w in self.graded if w.chosen_threshold is not None]

    @property
    def stable_threshold(self) -> bool:
        """Did the refit keep landing in the same place?

        A threshold that jumps between 50 and 80 from window to window is
        being fitted to noise even if each individual fit looked good.
        """
        chosen = self.chosen_thresholds
        return bool(chosen) and (max(chosen) - min(chosen)) <= 10.0

    def verdict(self) -> str:
        if not self.conclusive:
            return (
                f"INSUFFICIENT DATA at {self.horizon_minutes}m: {self.total_rows} scored rows "
                f"produced {len(self.graded)} gradable windows (need 3). The early threshold "
                "has NOT been walk-forward validated, and no value for it is supported yet."
            )
        mean = self.mean_test_expectancy_pct
        chosen = self.chosen_thresholds
        spread = max(chosen) - min(chosen)
        head = (
            f"Over {len(self.graded)} out-of-sample windows the refitted threshold averaged "
            f"{sum(chosen) / len(chosen):.0f} (range {min(chosen):.0f}-{max(chosen):.0f}) and "
            f"delivered {mean:+.2f}% after-cost expectancy."
        )
        if mean <= 0:
            return (
                head + " That is not an edge. No threshold made the early score profitable "
                "out-of-sample on this data, which is a result about the score, not about "
                "the threshold."
            )
        if not self.stable_threshold:
            return (
                head + f" The chosen threshold moved {spread:.0f} points between windows, which "
                "is the signature of fitting noise. Treat the positive expectancy as unproven."
            )
        gap = self.mean_gap_pct
        tail = f" Train-to-test gap averaged {gap:+.2f}%." if gap is not None else ""
        return head + " The threshold held steady across windows." + tail

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 3) if v is not None else None
        return {
            "horizon_minutes": self.horizon_minutes,
            "total_rows": self.total_rows,
            "conclusive": self.conclusive,
            "verdict": self.verdict(),
            "mean_test_expectancy_pct": r(self.mean_test_expectancy_pct),
            "mean_overfit_gap_pct": r(self.mean_gap_pct),
            "stable_threshold": self.stable_threshold,
            "windows": [w.as_dict() for w in self.windows],
            "warnings": list(self.warnings),
        }

    def table(self) -> str:
        lines = [self.verdict(), "",
                 f"  {'window':<8}{'train n':>9}{'test n':>8}{'chose':>8}"
                 f"{'train %':>10}{'test %':>9}{'taken':>7}"]
        for w in self.windows:
            chose = f"{w.chosen_threshold:.0f}" if w.chosen_threshold is not None else "-"
            tr = f"{w.train_expectancy_pct:+.2f}" if w.train_expectancy_pct is not None else "n/a"
            te = f"{w.test_expectancy_pct:+.2f}" if w.test_expectancy_pct is not None else "n/a"
            lines.append(f"  {w.index:<8}{w.train_rows:>9}{w.test_rows:>8}{chose:>8}"
                         f"{tr:>10}{te:>9}{w.test_trades:>7}")
        for warning in self.warnings:
            lines.append(f"\n  WARNING: {warning}")
        return "\n".join(lines)


def _expectancy(rows: list[tuple[float, float]], threshold: float, cost_pct: float) -> tuple[float | None, int]:
    """After-cost expectancy of everything at or above `threshold`."""
    taken = [ret - cost_pct for score, ret in rows if score >= threshold]
    if len(taken) < MIN_SELECTED_TRADES:
        return None, len(taken)
    return sum(taken) / len(taken), len(taken)


def _fit(rows: list[tuple[float, float]], cost_pct: float) -> tuple[float | None, float | None]:
    """The threshold with the best in-sample after-cost expectancy.

    Ties break toward the LOWER threshold. A higher threshold that scores
    identically is trading less for the same result, and preferring it
    would quietly select for small samples - which is how a walk-forward
    ends up recommending a threshold that fires twice a month.
    """
    best_threshold: float | None = None
    best_value: float | None = None
    for threshold in CANDIDATE_THRESHOLDS:
        value, _ = _expectancy(rows, threshold, cost_pct)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_threshold, best_value = threshold, value
    return best_threshold, best_value


def walk_forward_early_threshold(
    db: Session,
    *,
    horizon_minutes: int = 60,
    windows: int = 4,
    train_fraction: float = 0.6,
) -> EarlyWalkForward:
    """Refit the early threshold on each train window, grade on the next."""
    cost_pct = round_trip_cost_pct() * 100
    rows = (
        db.query(models.ForwardReturn)
        .filter(
            models.ForwardReturn.horizon_minutes == horizon_minutes,
            models.ForwardReturn.early_score.isnot(None),
            models.ForwardReturn.return_pct.isnot(None),
        )
        .order_by(models.ForwardReturn.observed_at.asc())
        .all()
    )
    data = [(r.early_score, r.return_pct) for r in rows]
    report = EarlyWalkForward(horizon_minutes=horizon_minutes, total_rows=len(data))

    if len(data) < MIN_WINDOW_ROWS * (windows + 1):
        report.warnings.append(
            f"{len(data)} rows is below the {MIN_WINDOW_ROWS * (windows + 1)} needed for "
            f"{windows} windows of at least {MIN_WINDOW_ROWS} rows each"
        )

    if not data:
        return report

    # Anchored-forward split: each fold trains on a chronological block and
    # is graded on the block immediately after it. Blocks never overlap and
    # a test block is never used for fitting.
    block = len(data) // (windows + 1)
    if block < MIN_WINDOW_ROWS:
        return report

    train_len = max(int(block * (1 + train_fraction)), MIN_WINDOW_ROWS)
    for i in range(windows):
        train_start = i * block
        train_end = min(train_start + train_len, len(data))
        test_end = min(train_end + block, len(data))
        train, test = data[train_start:train_end], data[train_end:test_end]
        if len(test) < MIN_WINDOW_ROWS:
            break

        threshold, train_value = _fit(train, cost_pct)
        if threshold is None:
            report.windows.append(
                Window(i + 1, len(train), len(test), None, None, None, 0)
            )
            continue
        test_value, taken = _expectancy(test, threshold, cost_pct)
        report.windows.append(
            Window(i + 1, len(train), len(test), threshold, train_value, test_value, taken)
        )

    return report
