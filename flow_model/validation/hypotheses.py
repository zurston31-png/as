"""Pre-registered falsifiable hypotheses.

Why this module exists
----------------------
The brief states targets -- 88-92% at 1R in favourable regimes, ~72%
combined over ten years -- and then states the constraint that matters more:
do not optimize to force those numbers, and report the real results. A target
that lives only in prose is not a constraint on anything. The failure mode it
invites is specific and extremely common: run the backtest, see the result,
then decide which comparison counts as success. By the time a criterion is
chosen after the data is seen, it is no longer a test.

So every criterion the project will be judged by is written down **here**,
as data, **before any backtest exists**. ARCHITECTURE.md section 14.7 records
the five criteria that would falsify the support/resistance construction;
section 6 records the regime expectation as a hypothesis rather than a goal.
This module is those statements made machine-readable, so Phase 8 evaluates
them mechanically instead of by narration, and so the experiment log carries
a timestamped, commit-stamped record that they predate the evidence.

What a registered hypothesis is NOT
-----------------------------------
It is not an objective. Nothing in `signals/`, `risk/` or `backtest/` may
read this module -- the same firewall that applies to `ResearchTargets`
applies here, for the same reason, and `tests/unit/test_no_target_leakage.py`
enforces it. A decision rule that can see the number it is being scored
against will, through enough iterations, reproduce that number and tell you
nothing.

The expected outcome is refutation. A symmetric 1R system at 88-92% implies
an annualized Sharpe far outside anything documented in the literature, and
`H-REGIME-WINRATE` is registered because it is testable, not because it is
plausible. A refuted hypothesis recorded in advance is a real result; a
confirmed hypothesis chosen in arrears is not.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from flow_model.core.model import FrozenModel

__all__ = [
    "Direction",
    "Hypothesis",
    "HYPOTHESES",
    "by_id",
    "registration_rows",
]


class Direction(str, Enum):
    """The shape of the prediction, which decides how it is refuted.

    Spelling this out matters because "the measurement disagreed with the
    prediction" is ambiguous until the predicted shape is fixed. A monotone
    claim is refuted by a non-monotone profile, not by a low level; a
    threshold claim is refuted by a value on the wrong side of the bound.
    """

    MONOTONE_INCREASING = "monotone_increasing"
    """A statistic must rise across ordered buckets of a driver."""

    DISTINCT_SIGN = "distinct_sign"
    """Two drivers must associate with outcome in OPPOSITE directions."""

    AT_LEAST = "at_least"
    """A statistic must reach a stated floor."""

    AT_MOST = "at_most"
    """A statistic must stay under a stated ceiling."""

    ORDERED_GROUPS = "ordered_groups"
    """One named group's statistic must exceed another's."""


class Hypothesis(FrozenModel):
    """One prediction, with its refutation condition fixed in advance."""

    hypothesis_id: str
    source: str
    """The ARCHITECTURE.md section this is the machine-readable form of."""

    claim: str
    """The prediction in plain words."""

    metric: str
    """The measured quantity. Must match an analytics breakdown key."""

    driver: str = ""
    """The variable the claim is conditioned on. Empty for an unconditional claim."""

    direction: Direction
    threshold: float | None = None
    """The bound, for AT_LEAST / AT_MOST. None where the shape is the claim."""

    groups: tuple[str, ...] = ()
    """For ORDERED_GROUPS, highest expected first."""

    refuted_when: str
    """The refutation condition, stated so it can be checked without judgment."""

    consequence: str
    """What is DONE if refuted. A hypothesis with no consequence is decoration."""

    expected_to_hold: bool
    """My prior, recorded so it cannot be revised after the fact."""

    @field_validator("consequence", "refuted_when", "claim")
    @classmethod
    def _must_be_substantive(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError(
                "claim, refuted_when and consequence must be stated in full. A "
                "pre-registration that is too vague to check in advance is "
                "indistinguishable from choosing the criterion afterwards."
            )
        return value

    @field_validator("threshold")
    @classmethod
    def _threshold_in_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be a rate in [0, 1], got {value}")
        return value

    def requires_threshold(self) -> bool:
        return self.direction in (Direction.AT_LEAST, Direction.AT_MOST)


#: The complete set. Appending after a result is seen defeats the purpose, so
#: `tests/unit/test_hypotheses.py` pins the ids and the count.
HYPOTHESES: tuple[Hypothesis, ...] = (
    # --- section 14.7: the five criteria that falsify the S/R construction ---
    Hypothesis(
        hypothesis_id="H-SR-SIGNIFICANCE-MONOTONE",
        source="ARCHITECTURE.md 14.7 (1)",
        claim=(
            "Win rate increases monotonically across ordered buckets of the "
            "significance score S, because S is supposed to measure how much a "
            "level matters."
        ),
        metric="win_rate",
        driver="significance_bucket",
        direction=Direction.MONOTONE_INCREASING,
        refuted_when=(
            "The win rate profile across S buckets is not monotone increasing, "
            "with each bucket holding at least min_bucket_trades trades."
        ),
        consequence=(
            "'Major' is not measuring significance. Report S as non-predictive "
            "and remove the gate rather than reweighting its six terms, since "
            "reweighting against the same data is the overfitting loop."
        ),
        expected_to_hold=False,
    ),
    Hypothesis(
        hypothesis_id="H-SR-CLEANLINESS-MONOTONE",
        source="ARCHITECTURE.md 14.7 (2)",
        claim=(
            "Win rate increases across ordered buckets of the cleanliness score "
            "C, because C is supposed to measure approach quality."
        ),
        metric="win_rate",
        driver="cleanliness_bucket",
        direction=Direction.MONOTONE_INCREASING,
        refuted_when=(
            "The win rate profile across C buckets is not increasing, with each "
            "bucket holding at least min_bucket_trades trades."
        ),
        consequence=(
            "The approach-quality terms are noise. REMOVE them rather than "
            "reweighting, as section 14.7 (2) specifies explicitly."
        ),
        expected_to_hold=False,
    ),
    Hypothesis(
        hypothesis_id="H-SR-SPLIT-JUSTIFIED",
        source="ARCHITECTURE.md 14.7 (3)",
        claim=(
            "s_density and s_touch associate with outcome in OPPOSITE "
            "directions, which is the whole reason significance and cleanliness "
            "are two scores rather than one: historical touches establish a "
            "level, recent touches mean it is under active attack."
        ),
        metric="win_rate",
        driver="s_density_vs_s_touch",
        direction=Direction.DISTINCT_SIGN,
        refuted_when=(
            "The two terms' associations with outcome carry the same sign, at or "
            "beyond the configured significance level."
        ),
        consequence=(
            "The S/C split is unjustified and collapses to a single score. "
            "Report the collapse; do not keep two scores for presentation."
        ),
        expected_to_hold=True,
    ),
    Hypothesis(
        hypothesis_id="H-SR-POPULATION-TRADABLE",
        source="ARCHITECTURE.md 14.7 (4)",
        claim=(
            "Acceptable performance does not require pushing the s_major gate "
            "above 0.75, which would leave too few levels to trade."
        ),
        metric="s_major_required_for_edge",
        direction=Direction.AT_MOST,
        threshold=0.75,
        refuted_when=(
            "Positive expectancy appears only at s_major above 0.75, or the "
            "qualifying-zone population at the chosen gate is below "
            "min_bucket_trades per fold."
        ),
        consequence=(
            "The result is a small-sample artifact, not an edge. Report it as "
            "such and do not present the high-gate numbers as the strategy."
        ),
        expected_to_hold=True,
    ),
    Hypothesis(
        hypothesis_id="H-SR-CBAND-ROBUST",
        source="ARCHITECTURE.md 14.7 (5)",
        claim=(
            "Expectancy survives a one-step change in the zone-width constant "
            "c_band in either direction, within fragility_max_relative_drop."
        ),
        metric="expectancy_relative_drop",
        driver="c_band",
        direction=Direction.AT_MOST,
        threshold=0.30,
        refuted_when=(
            "Either one-step change in c_band drops expectancy by more than "
            "fragility_max_relative_drop relative to the default."
        ),
        consequence=(
            "The zone construction is fragile. Report the strategy AS FRAGILE "
            "in the research report; do not search for a stabler constant, "
            "which would be the same overfitting loop in another costume."
        ),
        expected_to_hold=False,
    ),
    # --- section 6: the brief's own numbers, as hypotheses rather than goals ---
    Hypothesis(
        hypothesis_id="H-REGIME-WINRATE",
        source="ARCHITECTURE.md 6 / the brief",
        claim=(
            "The 1R scalp wins 88-92% of the time in favourable low-volatility "
            "and high-volatility regimes. Registered because it is testable; it "
            "implies an annualized Sharpe far outside anything documented, and "
            "refutation is the expected outcome."
        ),
        metric="win_rate",
        driver="regime",
        direction=Direction.AT_LEAST,
        threshold=0.88,
        groups=("LOW_VOL", "HIGH_VOL"),
        refuted_when=(
            "The measured 1R win rate in LOW_VOL or HIGH_VOL falls below 0.88 "
            "on out-of-sample folds with adequate sample size."
        ),
        consequence=(
            "Report the measured rate. Do not adjust gates, thresholds or costs "
            "to approach 0.88 -- the brief forbids it and it would invalidate "
            "every subsequent number."
        ),
        expected_to_hold=False,
    ),
    Hypothesis(
        hypothesis_id="H-REGIME-CHOP-WORSE",
        source="ARCHITECTURE.md 6 / the brief",
        claim=(
            "Win rate in CHOP is materially below the favourable regimes, which "
            "is the directional part of the brief's expectation and is far more "
            "plausible than its levels."
        ),
        metric="win_rate",
        driver="regime",
        direction=Direction.ORDERED_GROUPS,
        groups=("LOW_VOL", "HIGH_VOL", "CHOP"),
        refuted_when=(
            "CHOP's win rate is not below both LOW_VOL's and HIGH_VOL's by at "
            "least expected_chop_win_rate_reduction."
        ),
        consequence=(
            "The regime split does not separate outcomes, so regime-conditional "
            "sizing is unsupported. Report it and drop the conditioning."
        ),
        expected_to_hold=True,
    ),
    Hypothesis(
        hypothesis_id="H-COMBINED-10Y",
        source="ARCHITECTURE.md 6 / the brief",
        claim=(
            "The combined win rate across ten years of in-sample and "
            "out-of-sample data reaches 72%."
        ),
        metric="win_rate",
        driver="",
        direction=Direction.AT_LEAST,
        threshold=0.72,
        refuted_when=(
            "The pooled ten-year win rate, costs included, falls below 0.72."
        ),
        consequence=(
            "Report the measured value as the headline result. Per the brief, a "
            "stable 72% across regimes would be preferable to 90% in one "
            "period, so a lower stable number is reported as-is, not reframed."
        ),
        expected_to_hold=False,
    ),
    # --- the Kronos overlay, registered when it was added -----------------
    Hypothesis(
        hypothesis_id="H-KRONOS-ADDS-EDGE",
        source="ARCHITECTURE.md 16",
        claim=(
            "Adding the Kronos forecast overlay to the rules-based signal raises "
            "out-of-sample expectancy on bars the checkpoint provably never trained "
            "on. Registered when the integration was added, before any forecast was "
            "scored, so the comparison is not chosen after seeing it."
        ),
        metric="expectancy",
        driver="kronos_enabled",
        direction=Direction.ORDERED_GROUPS,
        groups=("kronos_on", "kronos_off"),
        refuted_when=(
            "Expectancy with the overlay does not exceed expectancy without it, on "
            "bars strictly after a declared pretrain_cutoff, with adequate sample "
            "size in both arms."
        ),
        consequence=(
            "Report that the overlay adds nothing and REMOVE it from the candidate "
            "configuration. Do not retune its horizon, path count or dead band "
            "against the same data -- that is the overfitting loop, and it is worse "
            "here than elsewhere because a learned component has far more capacity "
            "to absorb a tuning signal than a bounded rule does."
        ),
        expected_to_hold=False,
    ),
    Hypothesis(
        hypothesis_id="H-KRONOS-CONTAMINATION-MATTERS",
        source="ARCHITECTURE.md 16",
        claim=(
            "The contaminated signal outperforms the clean one. If a checkpoint that "
            "may have trained on the test bars scores no better than one evaluated "
            "only after its cutoff, then contamination was not doing any work and the "
            "quarantine costs nothing. If it scores much better, the gap IS the "
            "leakage, measured."
        ),
        metric="expectancy",
        driver="kronos_contaminated",
        direction=Direction.ORDERED_GROUPS,
        groups=("contaminated", "clean"),
        refuted_when=(
            "Expectancy under contamination_policy=FLAG over the research window does "
            "not exceed expectancy on post-cutoff bars by more than sampling error."
        ),
        consequence=(
            "Report the gap as the measured value of the leakage, whichever way it "
            "goes. A large gap is the strongest possible argument for the quarantine; "
            "a null gap is evidence the overlay is weak rather than evidence it is "
            "safe, and must never be reported as the latter."
        ),
        expected_to_hold=True,
    ),
    Hypothesis(
        hypothesis_id="H-CONSISTENCY-OVER-PEAK",
        source="ARCHITECTURE.md 10 / the brief",
        claim=(
            "Performance is stable across walk-forward folds rather than "
            "concentrated in a minority of them. The brief ranks consistency "
            "above peak performance, so this is checked directly."
        ),
        metric="fold_profit_concentration",
        driver="walk_forward_fold",
        direction=Direction.AT_MOST,
        threshold=0.50,
        refuted_when=(
            "More than half of total profit comes from a single walk-forward "
            "fold, or the sign of expectancy differs across folds."
        ),
        consequence=(
            "The edge is period-specific. Report every fold separately with the "
            "concentration stated; do not present the pooled number alone."
        ),
        expected_to_hold=False,
    ),
)


def by_id(hypothesis_id: str) -> Hypothesis:
    for item in HYPOTHESES:
        if item.hypothesis_id == hypothesis_id:
            return item
    raise KeyError(
        f"no registered hypothesis {hypothesis_id!r}. Registered: "
        f"{[h.hypothesis_id for h in HYPOTHESES]}"
    )


def registration_rows() -> list[dict[str, Any]]:
    """The payload for the experiment log, one row per hypothesis.

    Logged under TRAIN so the registration itself never spends an evaluation
    budget: writing down a prediction consumes no out-of-sample information.
    """
    return [
        {
            "name": f"register:{item.hypothesis_id}",
            "results": item.model_dump(mode="json"),
            "notes": (
                f"Pre-registered from {item.source} before any backtest result "
                f"exists. Refuted when: {item.refuted_when} Consequence: "
                f"{item.consequence}"
            ),
        }
        for item in HYPOTHESES
    ]
