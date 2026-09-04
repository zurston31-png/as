"""Comparing the three ways the two engines can be combined.

    A   existing technical strategy alone
    B   Early Signal Engine alone
    C   early signal finds the candidate, technical confirms the entry

The comparison exists because "the early engine is good" is not a
decision. The decision is which of these three configurations to run, and
they differ on axes that trade against each other: C trades least often
and should have the highest per-trade quality, B trades most often and
earliest, A is the status quo.

WHAT DECIDES IT

Out-of-sample expectancy after costs, penalised for the train-to-OOS gap
and for drawdown - the same `robust_score` the rest of app/research/ uses,
for the same reason. Not trade count, not lead time, not the headline
return. A mode that trades four times as often and loses a little on each
is worse than one that rarely trades, however much more alive it looks.

Lead time is reported ALONGSIDE rather than folded into the ranking. It is
the thing the early engine is supposed to buy, so it belongs in the table -
but a mode that gets in earlier and still loses money has not won
anything, and letting lead time influence the rank would let it hide that.

WHY THIS CANNOT RUN YET

The backtest engine walks a CandleSeries. The early engine's flow features
- transaction rate, buy-pressure change, liquidity growth - come from
successive market snapshots, which historical candles do not carry. So
mode B and mode C can only be evaluated on their candle-derived features
here, and `flow_available` records that limitation on every result rather
than letting a partial comparison read as a complete one.

A full comparison needs the bot to have RUN and stored observations. That
is what app/analysis/early_calibration.py is for, and it is the honest
route: measure the early engine live, in paper, before deciding it beats
anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.backtesting.types import BacktestConfig
from app.data.candles import CandleSeries
from app.research.harness import Experiment, VariantResult, run_variants


class Mode:
    TECHNICAL_ONLY = "A: technical only"
    EARLY_ONLY = "B: early signal only"
    COMBINED = "C: early finds, technical confirms"


@dataclass
class ModeResult:
    mode: str
    variant: VariantResult | None
    flow_available: bool
    note: str = ""

    @property
    def tested(self) -> bool:
        return bool(self.variant and self.variant.tested)

    @property
    def oos_expectancy_r(self) -> float | None:
        return self.variant.oos_expectancy_r if self.variant else None

    @property
    def robust_score(self) -> float | None:
        return self.variant.robust_score if self.variant else None

    @property
    def trades(self) -> int:
        return self.variant.oos_trades if self.variant else 0

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "tested": self.tested,
            "flow_available": self.flow_available,
            "note": self.note,
            "oos_trades": self.trades,
            "oos_expectancy_r": (
                round(self.oos_expectancy_r, 4) if self.oos_expectancy_r is not None else None
            ),
            "robust_score": round(self.robust_score, 4) if self.robust_score is not None else None,
            "detail": self.variant.as_dict() if self.variant else None,
        }


@dataclass
class ModeComparison:
    results: list[ModeResult] = field(default_factory=list)
    experiment: Experiment | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def tested(self) -> list[ModeResult]:
        return [r for r in self.results if r.tested]

    @property
    def conclusive(self) -> bool:
        return len(self.tested) >= 2

    @property
    def best(self) -> ModeResult | None:
        candidates = [r for r in self.tested if r.robust_score is not None]
        return max(candidates, key=lambda r: r.robust_score) if candidates else None

    def verdict(self) -> str:
        if not self.conclusive:
            return (
                f"INSUFFICIENT DATA: only {len(self.tested)} of {len(self.results)} modes produced "
                "enough out-of-sample trades to compare. No configuration should be chosen from "
                "this."
            )
        best = self.best
        others = [r for r in self.tested if r is not best]
        spread = best.robust_score - min(r.robust_score for r in others)
        if spread < 0.05:
            return (
                f"The three modes did not separate (robust scores span {spread:.3f}R). Treat them "
                "as equivalent on this data - which is itself a finding: the early engine is not "
                "adding measurable value here."
            )
        return (
            f"'{best.mode}' ranks highest out-of-sample ({best.robust_score:+.3f}R adjusted over "
            f"{best.trades} trades)."
        )

    def as_dict(self) -> dict:
        return {
            "conclusive": self.conclusive,
            "verdict": self.verdict(),
            "best": self.best.mode if self.best else None,
            "results": [r.as_dict() for r in self.results],
            "warnings": list(self.warnings),
        }

    def table(self) -> str:
        lines = [
            self.verdict(), "",
            f"  {'mode':<36}{'OOS n':>7}{'OOS R':>9}{'score':>9}  flow",
        ]
        for r in self.results:
            mark = " " if r.tested else "*"
            oos = f"{r.oos_expectancy_r:+.3f}" if r.oos_expectancy_r is not None else "n/a"
            score = f"{r.robust_score:+.3f}" if r.robust_score is not None else "n/a"
            flow = "yes" if r.flow_available else "NO"
            lines.append(f"{mark} {r.mode:<36}{r.trades:>7}{oos:>9}{score:>9}  {flow}")
        lines.append("\n  * too few out-of-sample trades to count")
        if not all(r.flow_available for r in self.results):
            lines.append(
                "  flow=NO: this mode's transaction-rate, buy-pressure and liquidity-growth\n"
                "  features could not be computed from historical candles, so it was evaluated\n"
                "  on its candle-derived features only. The comparison is partial."
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def compare_modes(
    series: CandleSeries,
    *,
    base_config: BacktestConfig | None = None,
    early_threshold: float = 70.0,
    train_fraction: float = 0.6,
) -> ModeComparison:
    """Run A, B and C over one chronological split.

    Mode B and C are approximated by tightening the technical threshold,
    because the backtest engine has no way to replay stored market
    snapshots and therefore cannot compute the early engine's flow
    features. That approximation is recorded on every result rather than
    hidden: it makes the comparison indicative, not decisive, and the real
    answer comes from running the bot in paper and reading
    /research early-calibration.
    """
    base_config = base_config or BacktestConfig()
    comparison = ModeComparison()

    comparison.warnings.append(
        "Modes B and C are APPROXIMATED here. The early engine's flow features "
        "(transaction rate, buy-pressure change, liquidity growth) come from successive market "
        "snapshots, which historical candles do not carry - so this compares candle-derived "
        "behaviour only. A real comparison needs the bot to have run in paper and stored "
        "observations."
    )

    variants = {
        Mode.TECHNICAL_ONLY: base_config,
        Mode.EARLY_ONLY: BacktestConfig(
            **{**base_config.__dict__, "min_score_to_enter": max(early_threshold - 15, 0)}
        ),
        Mode.COMBINED: BacktestConfig(
            **{**base_config.__dict__, "min_score_to_enter": early_threshold}
        ),
    }

    experiment = run_variants("strategy modes", series, variants, train_fraction=train_fraction)
    by_label = {v.label: v for v in experiment.variants}

    for mode in (Mode.TECHNICAL_ONLY, Mode.EARLY_ONLY, Mode.COMBINED):
        comparison.results.append(
            ModeResult(
                mode=mode,
                variant=by_label.get(mode),
                flow_available=(mode == Mode.TECHNICAL_ONLY),
                note=(
                    "runs the real strategy unchanged"
                    if mode == Mode.TECHNICAL_ONLY
                    else "approximated by threshold - flow features unavailable in a candle backtest"
                ),
            )
        )

    comparison.experiment = experiment
    return comparison
