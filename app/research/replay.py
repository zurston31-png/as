"""Re-scoring the bot's own recorded history at different thresholds.

app/research/thresholds.py already sweeps MIN_SIGNAL_SCORE_TO_ENTER, but
it does so by re-running the strategy over a CandleSeries. That answers
"what would this threshold have done on that price history". It does not
answer the question you actually have, which is "what would a different
threshold have done to the candidates MY bot actually saw".

Those differ in a way that matters. The live pipeline rejects tokens for
reasons a candle backtest never models - security, market quality, stale
data, exposure caps - so the population reaching the technical gate in
production is not the population a backtest generates. A threshold tuned
on synthetic candles is tuned on the wrong sample.

HOW THIS WORKS

Every scored candidate leaves a TECHNICAL_SCORE pipeline event carrying
its score, and forward_returns records what happened to it afterwards
WHETHER OR NOT IT WAS TRADED. That pairing is a complete natural
experiment: for any threshold, the set of candidates that would have been
taken is just the rows at or above it, and their realised returns are
already measured.

No re-simulation, no assumptions about fills beyond the recorded cost, and
no look-ahead: each row's return was measured after its own signal.

WHAT IT STILL CANNOT TELL YOU

Position sizing, exposure caps and concurrency are not replayed - this
compares per-candidate expectancy, not portfolio equity curves. A lower
threshold that doubles the trade count might have been blocked by the
concurrent-position limit half the time, and this will not show that.
It answers "were the extra trades any good", which is the first question,
not the only one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis.calibration import round_trip_cost_pct

# Below this many taken candidates a bucket's expectancy is a property of
# a handful of rows rather than of the threshold.
MIN_TAKEN = 20

DEFAULT_THRESHOLDS: tuple[float, ...] = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0)


@dataclass
class ThresholdOutcome:
    threshold: float
    taken: int
    skipped: int
    mean_return_pct: float | None = None
    median_return_pct: float | None = None
    win_rate_pct: float | None = None
    expectancy_net_pct: float | None = None
    total_return_pct: float | None = None      # sum, i.e. cumulative if sized equally
    worst_pct: float | None = None
    best_pct: float | None = None

    @property
    def meaningful(self) -> bool:
        return self.taken >= MIN_TAKEN

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 3) if v is not None else None
        return {
            "threshold": self.threshold,
            "taken": self.taken,
            "skipped": self.skipped,
            "meaningful": self.meaningful,
            "mean_return_pct": r(self.mean_return_pct),
            "median_return_pct": r(self.median_return_pct),
            "win_rate_pct": r(self.win_rate_pct),
            "expectancy_net_pct": r(self.expectancy_net_pct),
            "total_return_pct": r(self.total_return_pct),
            "worst_pct": r(self.worst_pct),
            "best_pct": r(self.best_pct),
        }


@dataclass
class Replay:
    horizon_minutes: int
    population: int = 0
    cost_pct: float = 0.0
    outcomes: list[ThresholdOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> list[ThresholdOutcome]:
        return [o for o in self.outcomes if o.meaningful]

    @property
    def best(self) -> ThresholdOutcome | None:
        candidates = [o for o in self.usable if o.expectancy_net_pct is not None]
        return max(candidates, key=lambda o: o.expectancy_net_pct) if candidates else None

    def verdict(self) -> str:
        if not self.usable:
            return (
                f"INSUFFICIENT DATA at {self.horizon_minutes}m: {self.population} scored "
                f"candidates with measured returns, and no threshold reaches {MIN_TAKEN} "
                "taken trades. Nothing here supports changing the threshold."
            )
        best = self.best
        if best is None:
            return "INSUFFICIENT DATA: no threshold produced a measurable expectancy."
        if best.expectancy_net_pct is not None and best.expectancy_net_pct <= 0:
            return (
                f"NO THRESHOLD WAS PROFITABLE at {self.horizon_minutes}m. The best of them "
                f"({best.threshold:.0f}) still returns {best.expectancy_net_pct:+.2f}% per trade "
                f"after costs over {best.taken} trades. Raising the bar does not fix a score "
                "that is not ranking outcomes - it just trades less of the same thing."
            )
        return (
            f"At {self.horizon_minutes}m, {best.threshold:.0f} gives the best after-cost "
            f"expectancy: {best.expectancy_net_pct:+.2f}% over {best.taken} trades. "
            "Check the neighbours before adopting it - a peak that its neighbours do not "
            "share is a fit to noise."
        )

    def plateau_note(self) -> str | None:
        """Is the best threshold sitting on a stable region or a spike?

        A value that only looks good because the two next to it look bad is
        the classic overfit, and it is invisible if you only read the
        winning row.
        """
        best = self.best
        if best is None or len(self.usable) < 3:
            return None
        ordered = sorted(self.usable, key=lambda o: o.threshold)
        index = next(i for i, o in enumerate(ordered) if o.threshold == best.threshold)
        neighbours = [
            o for i, o in enumerate(ordered)
            if abs(i - index) == 1 and o.expectancy_net_pct is not None
        ]
        if not neighbours:
            return None
        drop = best.expectancy_net_pct - max(n.expectancy_net_pct for n in neighbours)
        if drop > abs(best.expectancy_net_pct) * 0.5:
            return (
                f"WARNING: {best.threshold:.0f} beats its neighbours by {drop:+.2f}% - that is "
                "a spike, not a plateau. Prefer the centre of a stable region."
            )
        return f"{best.threshold:.0f} sits on a plateau; its neighbours are within {drop:.2f}%."

    def as_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "population": self.population,
            "cost_pct": round(self.cost_pct, 4),
            "verdict": self.verdict(),
            "plateau": self.plateau_note(),
            "best": self.best.threshold if self.best else None,
            "outcomes": [o.as_dict() for o in self.outcomes],
            "warnings": list(self.warnings),
        }

    def table(self) -> str:
        lines = [
            self.verdict(), "",
            f"  population: {self.population} scored candidates with a measured "
            f"{self.horizon_minutes}m return, round-trip cost {self.cost_pct * 100:.2f}%",
            "",
            f"  {'thresh':>7}{'taken':>8}{'skipped':>9}{'win %':>8}"
            f"{'mean %':>9}{'net %':>9}{'worst %':>10}{'best %':>10}",
        ]
        for o in self.outcomes:
            mark = " " if o.meaningful else "*"
            def f(v, w=9):
                return f"{v:>+{w}.2f}" if v is not None else f"{'n/a':>{w}}"
            win = f"{o.win_rate_pct:>8.0f}" if o.win_rate_pct is not None else f"{'n/a':>8}"
            lines.append(
                f"{mark} {o.threshold:>6.0f}{o.taken:>8}{o.skipped:>9}{win}"
                f"{f(o.mean_return_pct)}{f(o.expectancy_net_pct)}"
                f"{f(o.worst_pct, 10)}{f(o.best_pct, 10)}"
            )
        lines.append(f"\n  * fewer than {MIN_TAKEN} taken trades - shown, but not evidence")
        note = self.plateau_note()
        if note:
            lines.append(f"  {note}")
        lines.append(
            "\n  Replays THIS BOT'S recorded candidates, not a synthetic backtest, so the\n"
            "  population is the one the live pipeline actually produced. Sizing, exposure\n"
            "  caps and concurrency are NOT replayed: this compares per-trade expectancy,\n"
            "  not an equity curve."
        )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def replay_thresholds(
    db: Session,
    *,
    horizon_minutes: int = 60,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    strategy_version: str | None = None,
) -> Replay:
    """Score every recorded candidate against each candidate threshold."""
    query = db.query(models.ForwardReturn).filter(
        models.ForwardReturn.horizon_minutes == horizon_minutes,
        models.ForwardReturn.score.isnot(None),
        models.ForwardReturn.return_pct.isnot(None),
    )
    if strategy_version:
        query = query.filter(models.ForwardReturn.strategy_version == strategy_version)
    rows = [(r.score, r.return_pct) for r in query.all()]

    cost = round_trip_cost_pct()
    report = Replay(
        horizon_minutes=horizon_minutes, population=len(rows), cost_pct=cost
    )
    cost_pct_points = cost * 100

    if rows:
        distinct = len({round(score) for score, _ in rows})
        if distinct < 5:
            report.warnings.append(
                f"only {distinct} distinct score values in the sample - the thresholds are "
                "mostly slicing the same group, so differences between them mean little"
            )

    for threshold in thresholds:
        taken = [ret for score, ret in rows if score >= threshold]
        skipped = len(rows) - len(taken)
        outcome = ThresholdOutcome(threshold=threshold, taken=len(taken), skipped=skipped)
        if taken:
            net = [r - cost_pct_points for r in taken]
            outcome.mean_return_pct = sum(taken) / len(taken)
            outcome.median_return_pct = _median(taken)
            outcome.win_rate_pct = sum(1 for r in net if r > 0) / len(net) * 100
            outcome.expectancy_net_pct = sum(net) / len(net)
            outcome.total_return_pct = sum(net)
            outcome.worst_pct = min(taken)
            outcome.best_pct = max(taken)
        report.outcomes.append(outcome)

    return report
