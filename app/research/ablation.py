"""Feature ablation: does each scoring factor earn its weight?

The signal score is a weighted average of fourteen factors. Fourteen is a
number that arrived by accumulation, not by evidence - each one was added
because trading systems commonly use it. This module tests them.

The method is leave-one-out. For each factor, rebuild the weight map with
that factor removed and the remaining weights renormalised, then run the
strategy over the same chronological split as the full model and compare
OUT-OF-SAMPLE performance.

Reading the result:

    removing it HURTS      the factor was contributing - keep it
    removing it does NOTHING   the factor is redundant with the others -
                           it costs a data dependency and buys nothing
    removing it HELPS      the factor was adding noise - cut it, or cut
                           its weight

The third case is the one people refuse to believe and the reason this
module exists. An indicator that is standard, widely used, and actively
harmful to this strategy on this data is a completely ordinary finding.

RENORMALISATION MATTERS. Dropping a factor and leaving the others summing
to 0.88 would not test "without RSI" - it would test "without RSI, and with
every score depressed by 12%", which changes what the threshold means and
confounds the whole comparison. Weights are always rescaled to sum to 1.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.backtesting.types import BacktestConfig
from app.data.candles import CandleSeries
from app.research.harness import Experiment, VariantResult, run_variants
from app.signals.scoring import DEFAULT_WEIGHTS

FULL_MODEL = "full model"

# A factor whose removal moves out-of-sample expectancy by less than this
# has not been shown to do anything either way.
NEGLIGIBLE_R = 0.02


def weights_without(factor: str, base: dict[str, float] | None = None) -> dict[str, float]:
    """The weight map with `factor` neutralised and the rest renormalised.

    The factor's weight is set to ZERO rather than the key being deleted.
    score_signal() indexes every factor name directly, so a deleted key
    raises KeyError instead of running the ablated model - the ablation
    would crash rather than measure anything. A zero weight is exactly
    equivalent for the arithmetic: the factor contributes nothing to the
    numerator, nothing to the total weight the score divides by, and
    nothing to the missing-data budget that decides reliability.

    Renormalising the rest is not cosmetic either. Leaving the remaining
    weights summing to 0.88 would not test "without RSI"; it would test
    "without RSI, and with every score depressed by 12%", which changes
    what any threshold means and confounds the comparison with a pure
    scale change.
    """
    base = dict(base or DEFAULT_WEIGHTS)
    if factor not in base:
        raise KeyError(f"{factor!r} is not a scoring factor - cannot ablate it")
    remaining_total = sum(v for k, v in base.items() if k != factor)
    if remaining_total <= 0:
        raise ValueError(f"removing {factor!r} leaves no weight behind")
    return {
        k: (0.0 if k == factor else v / remaining_total) for k, v in base.items()
    }


@dataclass
class FactorVerdict:
    factor: str
    weight: float
    full_oos_r: float | None
    ablated_oos_r: float | None
    tested: bool

    @property
    def delta(self) -> float | None:
        """Out-of-sample expectancy WITHOUT the factor minus WITH it.

        Positive means the strategy did better without it.
        """
        if self.full_oos_r is None or self.ablated_oos_r is None:
            return None
        return self.ablated_oos_r - self.full_oos_r

    @property
    def verdict(self) -> str:
        if not self.tested or self.delta is None:
            return "not tested"
        if self.delta > NEGLIGIBLE_R:
            return "HURTS - removing it improved out-of-sample results"
        if self.delta < -NEGLIGIBLE_R:
            return "helps - removing it made results worse"
        return "no measurable effect - redundant with the other factors"

    @property
    def recommendation(self) -> str:
        if not self.tested or self.delta is None:
            return "collect more out-of-sample trades before deciding"
        if self.delta > NEGLIGIBLE_R:
            return f"cut it, or cut its weight from {self.weight:.2f}"
        if self.delta < -NEGLIGIBLE_R:
            return "keep"
        return "candidate for removal - it costs a data dependency and buys nothing measurable"

    def as_dict(self) -> dict:
        return {
            "factor": self.factor,
            "weight": self.weight,
            "full_oos_expectancy_r": round(self.full_oos_r, 4) if self.full_oos_r is not None else None,
            "ablated_oos_expectancy_r": (
                round(self.ablated_oos_r, 4) if self.ablated_oos_r is not None else None
            ),
            "delta_r": round(self.delta, 4) if self.delta is not None else None,
            "tested": self.tested,
            "verdict": self.verdict,
            "recommendation": self.recommendation,
        }


@dataclass
class AblationReport:
    experiment: Experiment
    factors: list[FactorVerdict]

    @property
    def conclusive(self) -> bool:
        return self.experiment.conclusive

    @property
    def harmful(self) -> list[FactorVerdict]:
        return [f for f in self.factors if f.tested and f.delta is not None and f.delta > NEGLIGIBLE_R]

    @property
    def helpful(self) -> list[FactorVerdict]:
        return [f for f in self.factors if f.tested and f.delta is not None and f.delta < -NEGLIGIBLE_R]

    @property
    def inert(self) -> list[FactorVerdict]:
        return [
            f for f in self.factors
            if f.tested and f.delta is not None and abs(f.delta) <= NEGLIGIBLE_R
        ]

    def summary(self) -> str:
        if not self.conclusive:
            return (
                "Feature ablation: INSUFFICIENT DATA. Not enough out-of-sample trades to "
                "compare variants. No factor should be added, removed or reweighted on this."
            )
        lines = [
            f"Feature ablation over {self.experiment.oos_bars} out-of-sample bars:",
            f"  {len(self.helpful)} factor(s) contribute, {len(self.inert)} show no measurable "
            f"effect, {len(self.harmful)} appear to HURT.",
            "",
            f"  {'factor':<22}{'weight':>8}{'delta R':>10}  verdict",
        ]
        for f in sorted(self.factors, key=lambda f: (f.delta is None, -(f.delta or 0))):
            delta = f"{f.delta:+.4f}" if f.delta is not None else "n/a"
            lines.append(f"  {f.factor:<22}{f.weight:>8.2f}{delta:>10}  {f.verdict}")
        if self.harmful:
            lines.append("")
            lines.append(
                "  An indicator being standard is not evidence it helps THIS strategy on THIS "
                "data. Confirm on a second dataset before cutting."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "conclusive": self.conclusive,
            "summary": self.summary(),
            "experiment": self.experiment.as_dict(),
            "factors": [f.as_dict() for f in self.factors],
        }


def run_ablation(
    series: CandleSeries,
    *,
    base_config: BacktestConfig | None = None,
    factors: tuple[str, ...] | None = None,
    train_fraction: float = 0.6,
) -> AblationReport:
    """Leave-one-out over every scoring factor.

    Runs the full model plus one variant per factor, all over the same
    chronological split, and reports the out-of-sample difference each
    factor's removal made.
    """
    base_config = base_config or BacktestConfig()
    base_weights = dict(base_config.weights or DEFAULT_WEIGHTS)
    factors = factors or tuple(base_weights)

    variants: dict[str, BacktestConfig] = {FULL_MODEL: base_config}
    for factor in factors:
        if factor not in base_weights:
            continue
        variant = BacktestConfig(**{**base_config.__dict__, "weights": weights_without(factor, base_weights)})
        variants[f"minus {factor}"] = variant

    experiment = run_variants("feature ablation", series, variants, train_fraction=train_fraction)

    by_label = {v.label: v for v in experiment.variants}
    full: VariantResult | None = by_label.get(FULL_MODEL)
    full_r = full.oos_expectancy_r if full else None

    verdicts = []
    for factor in factors:
        if factor not in base_weights:
            continue
        variant = by_label.get(f"minus {factor}")
        verdicts.append(
            FactorVerdict(
                factor=factor,
                weight=base_weights[factor],
                full_oos_r=full_r,
                ablated_oos_r=variant.oos_expectancy_r if variant else None,
                tested=bool(full and full.tested and variant and variant.tested),
            )
        )

    return AblationReport(experiment=experiment, factors=verdicts)
