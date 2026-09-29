"""Walk-forward evaluation and a sealed holdout.

Two different protections, often confused:

**Walk-forward** splits the history into consecutive folds and reports each
fold's out-of-sample result separately. A strategy whose edge lives in one fold
and nowhere else is a strategy fitted to that fold, and a single aggregate
number hides exactly that.

**The holdout** is the tail of the data that you agree never to look at while
developing. This module keeps it sealed: `develop()` returns the data minus the
holdout, and the holdout only runs when you pass `--holdout` explicitly. Every
such run is recorded in a small ledger with the config fingerprint at the time.
That ledger is the honest part - nothing can stop you rerunning the holdout,
but after the third run against three different configs it is no longer a
holdout, and the report will say so to your face.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..models import Candle, utcnow
from .metrics import Metrics, compute
from .runner import BacktestResult


@dataclass(slots=True)
class Fold:
    index: int
    train: list[Candle]
    test: list[Candle]

    @property
    def label(self) -> str:
        return f"fold {self.index}"

    def span(self, candles: Sequence[Candle]) -> tuple[str, str]:
        return (candles[0].ts.isoformat(), candles[-1].ts.isoformat()) if candles else ("", "")


def split_folds(
    candles: list[Candle], folds: int = 4, test_fraction: float = 0.25,
    anchored: bool = True, min_train: int = 300,
) -> list[Fold]:
    """Consecutive train/test folds over the series.

    `anchored` keeps every training window starting at the first bar (an
    expanding window, the usual choice when you have limited history); set it
    False for a rolling window of constant length.
    """
    n = len(candles)
    if folds < 1:
        raise ValueError("folds must be at least 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    test_size = max(1, int(n * test_fraction / folds))
    first_train = n - test_size * folds
    if first_train < min_train:
        raise ValueError(
            f"not enough history: {n} bars leaves {first_train} for the first training "
            f"window, below the {min_train}-bar minimum. Use fewer folds, a smaller "
            f"test_fraction, or more data."
        )

    out: list[Fold] = []
    for i in range(folds):
        train_end = first_train + test_size * i
        train_start = 0 if anchored else max(0, train_end - first_train)
        out.append(Fold(
            index=i + 1,
            train=candles[train_start:train_end],
            test=candles[train_end:train_end + test_size],
        ))
    return out


def develop(candles: list[Candle], holdout_fraction: float = 0.2
            ) -> tuple[list[Candle], list[Candle]]:
    """Split into (development, holdout). The holdout is the most recent tail."""
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    cut = len(candles) - int(len(candles) * holdout_fraction)
    return candles[:cut], candles[cut:]


@dataclass
class HoldoutLedger:
    """A record of every time the sealed data has been looked at."""

    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "HoldoutLedger":
        p = Path(path)
        entries: list[dict[str, Any]] = []
        if p.exists():
            try:
                entries = json.loads(p.read_text()).get("runs", [])
            except (OSError, json.JSONDecodeError):
                entries = []
        return cls(p, entries)

    def record(self, fingerprint: str, dataset: str, bars: int) -> None:
        self.entries.append({
            "at": utcnow().isoformat(),
            "config_fingerprint": fingerprint,
            "dataset": dataset,
            "bars": bars,
        })
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"runs": self.entries}, indent=2))
        except OSError:
            pass

    @property
    def runs(self) -> int:
        return len(self.entries)

    @property
    def distinct_configs(self) -> int:
        return len({e.get("config_fingerprint") for e in self.entries})

    def warning(self) -> str:
        """The honest caveat to print above a holdout result."""
        if self.runs <= 1:
            return ""
        if self.distinct_configs <= 1:
            return (f"This holdout has been run {self.runs} times, all with the same "
                    f"config. The result is still out-of-sample.")
        return (
            f"This holdout has now been run {self.runs} times across "
            f"{self.distinct_configs} different configurations. It is no longer a "
            f"clean out-of-sample test - each look leaks information into your "
            f"choices. Treat the number below as optimistic and get fresh data."
        )


@dataclass
class WalkForwardReport:
    folds: list[BacktestResult] = field(default_factory=list)
    holdout: Optional[BacktestResult] = None
    holdout_warning: str = ""
    combined: Optional[Metrics] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [f.to_dict() for f in self.folds],
            "holdout": self.holdout.to_dict() if self.holdout else None,
            "holdout_warning": self.holdout_warning,
            "combined_out_of_sample": self.combined.to_dict() if self.combined else None,
            "consistency": self.consistency(),
        }

    def consistency(self) -> dict[str, Any]:
        """Does the edge survive across folds, or live in one of them?"""
        expectancies = [f.metrics.expectancy_r for f in self.folds if f.metrics.trades]
        if not expectancies:
            return {"folds_with_trades": 0}
        positive = len([e for e in expectancies if e > 0])
        mean = sum(expectancies) / len(expectancies)
        variance = sum((e - mean) ** 2 for e in expectancies) / len(expectancies)
        return {
            "folds_with_trades": len(expectancies),
            "folds_profitable": positive,
            "mean_expectancy_r": round(mean, 4),
            "stdev_expectancy_r": round(variance ** 0.5, 4),
            "worst_fold_r": round(min(expectancies), 4),
            "best_fold_r": round(max(expectancies), 4),
        }

    def render(self) -> str:
        lines = ["  Out-of-sample folds", "  " + "-" * 62,
                 f"  {'fold':<8}{'bars':>7}{'trades':>8}{'win%':>7}"
                 f"{'expectancy':>13}{'net':>12}{'maxDD%':>9}"]
        for fold in self.folds:
            m = fold.metrics
            lines.append(
                f"  {fold.label:<8}{fold.bars:>7}{m.trades:>8}"
                f"{(m.win_rate * 100 if m.trades else 0):>6.0f}%"
                f"{m.expectancy_r:>+12.3f}R{m.net_pnl:>12,.0f}{m.max_drawdown_pct:>9.2f}"
            )
        c = self.consistency()
        if c.get("folds_with_trades"):
            lines += [
                "  " + "-" * 62,
                f"  {c['folds_profitable']}/{c['folds_with_trades']} folds profitable · "
                f"mean {c['mean_expectancy_r']:+.3f}R · sd {c['stdev_expectancy_r']:.3f} · "
                f"worst {c['worst_fold_r']:+.3f}R",
            ]
        if self.combined and self.combined.trades:
            lines += ["", "  Combined out-of-sample", self.combined.render()]
        if self.holdout:
            lines += ["", "  Holdout (sealed)"]
            if self.holdout_warning:
                lines += ["  ! " + self.holdout_warning]
            lines += [self.holdout.metrics.render()]
        return "\n".join(lines)


def combine(results: Sequence[BacktestResult], starting_equity: float) -> Metrics:
    """Pool every fold's trades into one equity curve, in time order."""
    trades = sorted((t for r in results for t in r.trades), key=lambda t: t.closed_at)
    return compute(trades, starting_equity)
