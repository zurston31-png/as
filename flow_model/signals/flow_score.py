"""Flow Score aggregation: five `ComponentScore`s -> one `FlowScore`, or a refusal.

ARCHITECTURE.md section 7:

    ComponentScores: options c_o in [0, w_o], orderflow c_f in [0, w_f],
                     structure c_s, liquidity c_l, volmom c_v
    FlowScore = c_o + c_f + c_s + c_l + c_v      in [0, 100]

This module does that sum and nothing else. It does not score anything --
`signals/scoring_bars.py` owns STRUCTURE, LIQUIDITY and VOL_MOMENTUM and
`signals/scoring_flow.py` owns ORDER_FLOW and OPTIONS_FLOW. What is owned
here is the one decision neither of them can make alone, because it needs
all five components in view at once:

    whether the five components in hand constitute a Flow Score at all.

## The refusal is the point of the module

`strict_component_availability` defaults TRUE. On bars-only data ORDER_FLOW
(25 points) and OPTIONS_FLOW (20 points) have no feed, so 45 of 100 points
were never measured. Summing the other three gives 55 possible points, and a
number on a 55-point scale compared against a threshold calibrated on a
100-point scale is not a conservative reading of the same quantity -- it is a
different quantity wearing the same name. Section 5 is explicit that the
weight is **not** redistributed, and `AvailabilityReport.scoring_is_valid`
already says the engine "must refuse to emit a Flow Score in that case".

So `aggregate` returns a `FlowScoreOutcome`, which is either a `FlowScore` or
a `FlowScoreRefusal` -- never both, and never an exception. A refusal is a
FIRST-CLASS OUTCOME because the signal engine has to record it as a WAIT
carrying a `WaitReason`, and a raised exception cannot be recorded as a bar's
outcome without the caller catching it and inventing the reason itself.

`WaitReason.COMPONENT_DISABLED` is the reason a refusal carries. The
alternative, `DATA_QUALITY`, is the reason the *data-quality gate* emits at
pipeline stage 2 for a feed that is missing or stale at this instant. The two
facts are different and section 12 counts which gate fired: "this bar's feed
went quiet" and "this dataset structurally cannot produce 45 of the 100
points" would otherwise land in one bucket, and the second condition holds on
every bar of the run while the first holds on a few.

## Both numbers are always reported

`available_points` and `max_points` are on the outcome whether it refused or
not, so a report can say "55.0 of 100.0" without reaching into the component
dict, and can say it about a refusal too. `FlowScore.available_points` is the
total weight of ENABLED components and `max_points` is the configured total;
the gap is the shortfall, and nothing here ever closes it by moving weight.

## Redistribution, when it is explicitly asked for

`redistribute_disabled_weight` is consulted only when
`strict_component_availability` is false. It is off by default. When both are
set the enabled components' weights are scaled up to sum to `total_points`,
every rescaled `ComponentScore` records its original weight in
`detail["weight_before_redistribution"]`, and the outcome carries
`redistributed=True` plus a reason stating in words that the resulting score
is NOT comparable to a score from a complete dataset. It is implemented
because the config field exists and a field that silently does nothing is
worse than one that does something loud; it is not recommended anywhere, and
no default turns it on.

## What this module refuses to guess

* A missing component. All five `Component` members must be present in the
  mapping handed in. Four components is not a degraded Flow Score, it is a
  caller that forgot a scorer, and that raises `FlowScoreError`.
* A component whose `weight` disagrees with `config.flow_score.weights`. The
  weights are the Phase 8 hypothesis surface and live in exactly one place; a
  scorer that returned a different weight would rescale the whole score
  invisibly.
* Weights that do not sum to `total_points`. `FlowScoreConfig` validates this
  at construction, so reaching the check here means the config was built
  through a bypass; it is re-checked because the [0, 100] bound of the sum
  rests on it and an invariant asserted only where it cannot fail is
  decoration.

Nothing here reads `config.research_targets` or the pre-registered
hypotheses, and nothing here claims the score predicts anything.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Mapping, Sequence

from pydantic import Field, computed_field, model_validator

from flow_model.config.schema import FlowModelConfig, FlowScoreConfig
from flow_model.core.contracts import ComponentScore, FeatureVector, FlowScore
from flow_model.core.enums import Component, DataQuality, WaitReason
from flow_model.core.model import FrozenModel
from flow_model.data.quality import AvailabilityReport, ComponentAvailability
from flow_model.signals.scoring_bars import BarsComponentScorer, bars_scorers
from flow_model.signals.scoring_flow import (
    ComponentScorer,
    ComponentScoring,
    feed_dependent_scorers,
)

__all__ = [
    "AnyComponentScorer",
    "FlowScoreError",
    "FlowScoreOutcome",
    "FlowScoreRefusal",
    "REFUSAL_WAIT_REASON",
    "ScoredComponents",
    "WEIGHT_SUM_TOLERANCE",
    "WEIGHT_TOLERANCE",
    "aggregate",
    "all_scorers",
    "score_components",
    "summary_lines",
]


#: Tolerance when comparing a `ComponentScore.weight` against the CONFIGURED
#: weight for that component. Tight, because the two numbers are the same
#: number: the scorer is handed its weight out of `config.flow_score.weights`
#: and echoes it back, so anything beyond float-identity error is a scorer
#: that chose its own weight.
WEIGHT_TOLERANCE = 1e-9

#: Tolerance when checking that the five weights sum to `total_points`.
#:
#: **Deliberately the same tolerance `FlowScoreConfig` validates with, and
#: not tighter.** That invariant is OWNED by `FlowScoreConfig._check`, which
#: accepts `abs(total - total_points) <= 1e-6`. Re-checking it here more
#: strictly than its owner enforces it made this function raise
#: `FlowScoreError` on every bar for a config the config layer had already
#: declared valid -- weights summing to 100.0000005 construct fine and then
#: killed the run, for a discrepancy five thousand times smaller than the
#: smallest threshold comparison in `config.setups`. The check stays (the
#: `[0, total]` bound rests on it) but it now agrees with the layer that
#: decides what a legal config is, so the only sums that reach the raise are
#: sums `FlowScoreConfig` would also have refused.
WEIGHT_SUM_TOLERANCE = 1e-6

#: The `WaitReason` a refusal carries. See the module docstring on why this
#: is not `DATA_QUALITY`.
REFUSAL_WAIT_REASON = WaitReason.COMPONENT_DISABLED

#: Either scorer family. The two have different return types -- the bars
#: scorers return a bare `ComponentScore` and carry their words in
#: `explain()`, the feed-dependent ones return a `ComponentScoring` that
#: carries its words inline -- and `score_components` is the single place
#: that reconciles them, exactly as `scoring_flow.score_component`'s
#: docstring asks.
AnyComponentScorer = BarsComponentScorer | ComponentScorer


class FlowScoreError(RuntimeError):
    """Raised when the aggregate's declared contract is violated.

    Every condition that reaches here is a programming or configuration
    error: a missing component, a weight that disagrees with config, weights
    that do not sum to the configured total. Missing *data* never raises --
    it produces a `FlowScoreRefusal`, which is an outcome the engine records.
    """


# ---------------------------------------------------------------------------
# scoring the five components
# ---------------------------------------------------------------------------


class ScoredComponents(FrozenModel):
    """The five `ComponentScore`s plus the words for why they look that way.

    `scores` goes straight into `FlowScore(components=...)` and into
    `aggregate`. `reasons` is the union of every component's own explanation,
    in `Component` declaration order, for `Signal.reasons`; it is empty
    exactly when all five components were measured and graded GOOD.
    """

    symbol: str
    ts: datetime
    scores: dict[Component, ComponentScore]
    reasons: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unavailable(self) -> tuple[Component, ...]:
        """Components with no measurement, in `Component` declaration order."""
        return tuple(c for c in Component if not self.scores[c].enabled)

    @model_validator(mode="after")
    def _check(self) -> "ScoredComponents":
        missing = [c.value for c in Component if c not in self.scores]
        if missing:
            raise FlowScoreError(
                f"no score for component(s) {sorted(missing)}. A Flow Score is "
                "the sum of all five components; a subset is not a smaller Flow "
                "Score, it is a caller that did not run every scorer."
            )
        for component, score in self.scores.items():
            if score.component is not component:
                raise FlowScoreError(
                    f"score keyed under {component.value} reports component "
                    f"{score.component.value}"
                )
        return self


def all_scorers(config: FlowModelConfig) -> tuple[AnyComponentScorer, ...]:
    """One scorer per component, in `Component` declaration order.

    Takes the root config because the two feed-dependent scorers read
    `config.order_flow` and `config.options_flow`; the three bars-only ones
    accept the root object too. The order is `Component`'s own declaration
    order rather than either sub-module's, so anything iterating the five --
    a report, a hash, a loop -- sees one deterministic sequence.
    """
    if not isinstance(config, FlowModelConfig):
        raise FlowScoreError(
            f"all_scorers takes a FlowModelConfig; got {type(config).__name__}. "
            "The two feed-dependent scorers read config.order_flow and "
            "config.options_flow, which a FlowScoreConfig does not carry."
        )
    by_component: dict[Component, AnyComponentScorer] = {}
    for scorer in (*bars_scorers(config), *feed_dependent_scorers(config)):
        if scorer.component in by_component:  # pragma: no cover - both factories are fixed
            raise FlowScoreError(
                f"two scorers claim component {scorer.component.value}"
            )
        by_component[scorer.component] = scorer
    missing = [c.value for c in Component if c not in by_component]
    if missing:  # pragma: no cover - both factories are fixed
        raise FlowScoreError(f"no scorer for component(s) {missing}")
    return tuple(by_component[c] for c in Component)


def score_components(
    config: FlowModelConfig,
    features: FeatureVector,
    availability: AvailabilityReport | Mapping[Component, ComponentAvailability] | None = None,
    *,
    scorers: Sequence[AnyComponentScorer] | None = None,
) -> ScoredComponents:
    """Score all five components at `features.ts`.

    `availability` is `data/quality.py`'s DATASET-level verdict, as an
    `AvailabilityReport` or as the `{component: ComponentAvailability}`
    mapping inside one. It is passed through to each scorer, which CONSUMES
    it: a refusal propagates, and a `computable=True` verdict never overrides
    a vector whose own keys are MISSING. Nothing is re-derived here.

    **`availability()` reads the whole series and is deliberately forward
    looking.** It must be computed once before a run and handed in, never
    recomputed per bar inside a backtest. This function does not call it.

    `scorers` lets a caller reuse one set of scorer objects across bars
    instead of rebuilding five per bar. They are configuration-only objects
    with no market-data accessor, so sharing them cannot carry a value from
    one bar to the next.
    """
    verdicts = _availability_map(availability)
    chosen = tuple(scorers) if scorers is not None else all_scorers(config)
    scores: dict[Component, ComponentScore] = {}
    reasons: list[str] = []
    for scorer in chosen:
        verdict = verdicts.get(scorer.component)
        result = scorer.score(features, verdict)
        if isinstance(result, ComponentScoring):
            scores[result.component] = result.score
            if result.reason:
                reasons.append(result.reason)
            reasons.extend(result.notes)
        else:
            scores[result.component] = result
            reasons.extend(scorer.explain(features, verdict))
    ordered = {c: scores[c] for c in Component if c in scores}
    return ScoredComponents(
        symbol=features.symbol,
        ts=features.ts,
        scores=ordered,
        reasons=tuple(reasons),
    )


def _availability_map(
    availability: AvailabilityReport | Mapping[Component, ComponentAvailability] | None,
) -> Mapping[Component, ComponentAvailability]:
    if availability is None:
        return {}
    if isinstance(availability, AvailabilityReport):
        return availability.components
    return availability


# ---------------------------------------------------------------------------
# the outcome types
# ---------------------------------------------------------------------------


class FlowScoreRefusal(FrozenModel):
    """The aggregate declined to produce a Flow Score, and why.

    An outcome, not an error. The engine turns this into a WAIT carrying
    `wait_reason`, and section 12's rejection analysis counts it like any
    other gate.
    """

    symbol: str
    ts: datetime
    wait_reason: WaitReason = REFUSAL_WAIT_REASON
    unavailable_components: tuple[Component, ...] = ()
    available_points: float = Field(ge=0.0)
    max_points: float = Field(gt=0.0)
    detail: str = ""
    reasons: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def missing_points(self) -> float:
        """Weight that was never measured. The number the refusal is about."""
        return round(self.max_points - self.available_points, 6)


class FlowScoreOutcome(FrozenModel):
    """Exactly one of: a `FlowScore`, or a `FlowScoreRefusal`.

    `available_points` and `max_points` are present either way, so "55.0 of
    100.0" is reportable without unpacking the union.
    """

    symbol: str
    ts: datetime
    flow_score: FlowScore | None = None
    refusal: FlowScoreRefusal | None = None
    available_points: float = Field(ge=0.0)
    max_points: float = Field(gt=0.0)
    unavailable_components: tuple[Component, ...] = ()
    redistributed: bool = False
    reasons: tuple[str, ...] = ()

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    @property
    def score(self) -> float | None:
        """The 0..100 aggregate, or None when the aggregate refused."""
        return None if self.flow_score is None else self.flow_score.score

    @property
    def quality(self) -> DataQuality:
        """Worst grade among measured components; MISSING on a refusal."""
        if self.flow_score is None:
            return DataQuality.MISSING
        return self.flow_score.quality

    @property
    def wait_reason(self) -> WaitReason | None:
        return None if self.refusal is None else self.refusal.wait_reason

    @computed_field  # type: ignore[prop-decorator]
    @property
    def missing_points(self) -> float:
        return round(self.max_points - self.available_points, 6)

    def require(self) -> FlowScore:
        """The `FlowScore`, or raise. For call sites that treat a refusal as a bug."""
        if self.flow_score is None:
            detail = self.refusal.detail if self.refusal else "no detail"
            raise FlowScoreError(f"the aggregate refused to produce a Flow Score: {detail}")
        return self.flow_score

    @model_validator(mode="after")
    def _check(self) -> "FlowScoreOutcome":
        if (self.flow_score is None) == (self.refusal is None):
            raise FlowScoreError(
                "a FlowScoreOutcome carries exactly one of flow_score and "
                "refusal: both set would let a caller read a score the "
                "aggregate refused, and neither set is not an outcome"
            )
        if self.flow_score is not None:
            if self.flow_score.symbol != self.symbol or self.flow_score.ts != self.ts:
                raise FlowScoreError("outcome and FlowScore disagree on symbol/ts")
        if self.refusal is not None:
            if self.refusal.symbol != self.symbol or self.refusal.ts != self.ts:
                raise FlowScoreError("outcome and refusal disagree on symbol/ts")
        return self


# ---------------------------------------------------------------------------
# the aggregate
# ---------------------------------------------------------------------------


def aggregate(
    config: FlowModelConfig | FlowScoreConfig,
    components: ScoredComponents | Mapping[Component, ComponentScore],
    *,
    symbol: str | None = None,
    ts: datetime | None = None,
) -> FlowScoreOutcome:
    """`FlowScore = c_o + c_f + c_s + c_l + c_v`, or a refusal.

    `components` is a `ScoredComponents` (which carries symbol, ts and the
    components' own reasons) or a bare mapping, in which case `symbol` and
    `ts` are required.

    The order of decisions, which matters because only the first produces an
    outcome:

    1. Contract checks. A missing component, a weight that disagrees with
       `config.flow_score.weights`, or weights that do not sum to
       `total_points` raise `FlowScoreError` -- all three are caller or
       config errors, not data conditions.
    2. Nothing measured at all. Every component unavailable means there is
       no scale to express anything on, so this refuses whatever the flags
       say: `strict=false, redistribute=true` would otherwise divide by zero
       available weight.
    3. `strict_component_availability` and any component unavailable ->
       refuse.
    4. Not strict, `redistribute_disabled_weight` -> scale the enabled
       weights up to `total_points` and say so in words.
    5. Otherwise -> the plain sum, with `available_points` below
       `max_points` and the shortfall reported.
    """
    flow_config = _flow_config(config)
    scored = _as_scored(components, symbol=symbol, ts=ts)
    scores = scored.scores
    _check_weights(flow_config, scores)

    max_points = float(flow_config.total_points)
    unavailable = scored.unavailable
    available_points = round(
        sum(s.weight for s in scores.values() if s.enabled), 6
    )
    shortfall = f"{available_points:.1f} of {max_points:.1f} points are computable"

    if not unavailable:
        return FlowScoreOutcome(
            symbol=scored.symbol,
            ts=scored.ts,
            flow_score=FlowScore(
                symbol=scored.symbol, ts=scored.ts, components=dict(scores),
                max_points=max_points,
            ),
            available_points=available_points,
            max_points=max_points,
            reasons=scored.reasons + (f"flow score: all five components measured, {shortfall}",),
        )

    named = ", ".join(c.value for c in unavailable)
    if available_points <= 0.0:
        return _refuse(
            scored,
            available_points,
            max_points,
            unavailable,
            detail=(
                f"no component was measured ({named}); there is no scale on "
                "which to express a Flow Score"
            ),
        )

    if flow_config.strict_component_availability:
        return _refuse(
            scored,
            available_points,
            max_points,
            unavailable,
            detail=(
                f"strict_component_availability is on and {named} "
                f"{'has' if len(unavailable) == 1 else 'have'} no measurement, "
                f"so only {shortfall}. A score summed from the rest would be on "
                f"a {available_points:.0f}-point scale while every threshold in "
                f"config.setups is written against {max_points:.0f}; refusing is "
                "the only answer that does not quietly change the units"
            ),
        )

    if flow_config.redistribute_disabled_weight:
        scale = max_points / available_points
        rescaled = {
            component: (
                score.replace(
                    weight=round(score.weight * scale, 10),
                    detail={
                        **score.detail,
                        "weight_before_redistribution": score.weight,
                        "redistribution_scale": round(scale, 10),
                    },
                )
                if score.enabled
                else score
            )
            for component, score in scores.items()
        }
        reasons = scored.reasons + (
            f"flow score: {named} unavailable and redistribute_disabled_weight "
            f"is ON, so the remaining components' weights were scaled by "
            f"{scale:.4f} to sum to {max_points:.1f}",
            "flow score: THIS SCORE IS NOT COMPARABLE to a score from a "
            "complete dataset. Redistribution inflates every component that "
            "did have data, and thresholds calibrated on one scale do not "
            "transfer to the other (ARCHITECTURE.md section 5).",
        )
        return FlowScoreOutcome(
            symbol=scored.symbol,
            ts=scored.ts,
            flow_score=FlowScore(
                symbol=scored.symbol, ts=scored.ts, components=rescaled,
                max_points=max_points,
            ),
            available_points=round(
                sum(s.weight for s in rescaled.values() if s.enabled), 6
            ),
            max_points=max_points,
            unavailable_components=unavailable,
            redistributed=True,
            reasons=reasons,
        )

    return FlowScoreOutcome(
        symbol=scored.symbol,
        ts=scored.ts,
        flow_score=FlowScore(
            symbol=scored.symbol, ts=scored.ts, components=dict(scores),
            max_points=max_points,
        ),
        available_points=available_points,
        max_points=max_points,
        unavailable_components=unavailable,
        reasons=scored.reasons
        + (
            f"flow score: {named} unavailable, so {shortfall}. "
            "strict_component_availability is OFF, so the sum of the measured "
            "components is reported as-is; its weight was NOT redistributed, "
            f"and it is a number out of {available_points:.1f}, not out of "
            f"{max_points:.1f}.",
        ),
    )


def _refuse(
    scored: ScoredComponents,
    available_points: float,
    max_points: float,
    unavailable: tuple[Component, ...],
    *,
    detail: str,
) -> FlowScoreOutcome:
    refusal = FlowScoreRefusal(
        symbol=scored.symbol,
        ts=scored.ts,
        unavailable_components=unavailable,
        available_points=available_points,
        max_points=max_points,
        detail=detail,
        reasons=scored.reasons,
    )
    return FlowScoreOutcome(
        symbol=scored.symbol,
        ts=scored.ts,
        refusal=refusal,
        available_points=available_points,
        max_points=max_points,
        unavailable_components=unavailable,
        reasons=scored.reasons + (f"flow score REFUSED: {detail}",),
    )


def _flow_config(config: FlowModelConfig | FlowScoreConfig) -> FlowScoreConfig:
    if isinstance(config, FlowModelConfig):
        return config.flow_score
    if isinstance(config, FlowScoreConfig):
        return config
    raise FlowScoreError(
        f"aggregate takes a FlowModelConfig or a FlowScoreConfig; got "
        f"{type(config).__name__}"
    )


def _as_scored(
    components: ScoredComponents | Mapping[Component, ComponentScore],
    *,
    symbol: str | None,
    ts: datetime | None,
) -> ScoredComponents:
    if isinstance(components, ScoredComponents):
        return components
    if symbol is None or ts is None:
        raise FlowScoreError(
            "aggregate needs symbol and ts when given a bare component mapping; "
            "pass a ScoredComponents, which carries both, or supply them"
        )
    return ScoredComponents(symbol=symbol, ts=ts, scores=dict(components))


def _check_weights(
    flow_config: FlowScoreConfig, scores: Mapping[Component, ComponentScore]
) -> None:
    """Each component's weight is the configured one, and the five sum to the total.

    Checked rather than trusted. The [0, max_points] bound on the sum rests
    entirely on the weights summing to `total_points`, and a scorer that
    returned its own weight -- a cap applied to a weight rather than to a
    magnitude, say -- would rescale the whole score with nothing saying so.

    The two checks use two tolerances, for two different reasons: see
    `WEIGHT_TOLERANCE` (a scorer echoing a configured number, so float
    identity) and `WEIGHT_SUM_TOLERANCE` (an invariant `FlowScoreConfig`
    owns, so exactly as loose as `FlowScoreConfig` enforces it and no
    tighter).
    """
    for component, score in scores.items():
        configured = flow_config.weights.get(component)
        if configured is None:  # pragma: no cover - FlowScoreConfig validates this
            raise FlowScoreError(
                f"config.flow_score.weights carries no weight for {component.value}"
            )
        if not math.isclose(
            score.weight, float(configured), rel_tol=0.0, abs_tol=WEIGHT_TOLERANCE
        ):
            raise FlowScoreError(
                f"{component.value} reports weight {score.weight} but "
                f"config.flow_score.weights says {configured}. A scorer may not "
                "choose its own weight: the weights are one Phase 8 hypothesis "
                "surface and live in exactly one place. An UNAVAILABLE component "
                "keeps its configured weight and sets enabled=False, which is "
                "what makes the shortfall reportable."
            )
    total = sum(float(w) for w in flow_config.weights.values())
    if not math.isclose(
        total, float(flow_config.total_points), rel_tol=0.0, abs_tol=WEIGHT_SUM_TOLERANCE
    ):
        raise FlowScoreError(
            f"component weights sum to {total}, not total_points "
            f"{flow_config.total_points}; section 7 requires them to sum to the "
            "total, and the [0, total] bound on the Flow Score rests on it"
        )


def summary_lines(outcome: FlowScoreOutcome) -> tuple[str, ...]:
    """Human-readable breakdown, for a report or a CLI.

    Separate from the outcome types so that the record stays data and the
    presentation stays here.
    """
    head = f"Flow Score for {outcome.symbol} @ {outcome.ts.isoformat()}"
    lines = [head, "-" * len(head)]
    if outcome.flow_score is None:
        lines.append("  REFUSED -- no Flow Score was produced.")
        if outcome.refusal is not None:
            lines.append(f"  reason: {outcome.refusal.wait_reason.value}")
            lines.append(f"  {outcome.refusal.detail}")
    else:
        flow = outcome.flow_score
        lines.append(f"  {'component':<16}{'weight':>8}{'magnitude':>11}{'dir':>5}{'points':>9}")
        lines.append(f"  {'-' * 49}")
        for component in Component:
            score = flow.components.get(component)
            if score is None:  # pragma: no cover - validated present
                continue
            magnitude = f"{score.magnitude:.3f}" if score.enabled else "n/a"
            lines.append(
                f"  {component.value:<16}{score.weight:>8.1f}{magnitude:>11}"
                f"{score.direction:>5}{score.points:>9.2f}"
            )
        lines.append(f"  {'-' * 49}")
        lines.append(f"  score {flow.score:.2f} of {outcome.available_points:.1f} available")
    lines.append(
        f"  available points: {outcome.available_points:.1f} of {outcome.max_points:.1f}"
    )
    if outcome.unavailable_components:
        named = ", ".join(c.value for c in outcome.unavailable_components)
        if outcome.redistributed:
            # `missing_points` is 0.0 here BY CONSTRUCTION -- redistribution
            # scales the measured components up until the enabled weight sums
            # to `max_points` again -- so the shortfall has to be stated from
            # the CONFIGURED weights instead. Printing `missing_points` with
            # the words "weight not redistributed", which is what this line
            # used to do, told the reader "0.0 points, weight not
            # redistributed" one line above "WEIGHT WAS REDISTRIBUTED": two
            # adjacent lines contradicting each other, with the false one
            # first and the 45 points that have no feed reported as none.
            withheld = round(
                sum(
                    float(s.detail.get("weight_before_redistribution", s.weight))
                    for component, s in (
                        outcome.flow_score.components.items()
                        if outcome.flow_score is not None
                        else ()
                    )
                    if component in outcome.unavailable_components
                ),
                6,
            )
            lines.append(
                f"  UNAVAILABLE: {named} ({withheld:.1f} configured points, "
                "WEIGHT REDISTRIBUTED onto the measured components)"
            )
        else:
            lines.append(
                f"  UNAVAILABLE: {named} ({outcome.missing_points:.1f} points, "
                "weight not redistributed)"
            )
    if outcome.redistributed:
        lines.append("  WEIGHT WAS REDISTRIBUTED: not comparable to a complete dataset.")
    return tuple(lines)
