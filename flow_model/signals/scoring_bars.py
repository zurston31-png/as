"""Component scorers for the three bars-only components.

ARCHITECTURE.md section 5 splits the five Flow Score components by what feed
they need. Three of them need nothing but OHLCV bars:

    STRUCTURE     20 pts   "Full function on bars alone."
    LIQUIDITY     15 pts   "Degrades to volume-percentile only, flagged DEGRADED."
    VOL_MOMENTUM  20 pts   "Full function on bars alone."

They live in one module because that property is what they share, and
because three files of near-identical shape would be churn. The other two
components (ORDER_FLOW 25, OPTIONS_FLOW 20) need feeds bars cannot supply
and are scored elsewhere.

## What a scorer is, and what it is not

A scorer maps an **already-computed** `FeatureVector` to a `ComponentScore`.
It does not compute features and it does not touch market data: the feature
layer did that, and `validation/lookahead.py` has already audited it for
lookahead. Every constructor here takes configuration only -- there is no
accessor on a scorer through which a bar could arrive -- so "every signal
uses only information available at the exact timestamp of the trade" is
inherited from the vector it is handed rather than re-argued here.

A scorer also does **not** aggregate, gate, or select a setup. It produces
one component's `(magnitude, direction, weight, quality)` and stops.

## The rule this module is organized around (section 7)

    "Direction is handled separately from magnitude: each component emits
     (magnitude in [0,1], direction in {-1,0,+1}). The Flow Score is the
     magnitude aggregate; direction agreement is enforced by the gates. This
     avoids the common bug where a strong bearish component inflates a
     bullish score."

So `magnitude` here is strictly "how much evidence", with the sign stripped,
and `direction` is strictly "which way", with the strength stripped. The two
are computed by separate code paths from separate feature keys -- see
`magnitude_keys` and `direction_keys` on each scorer -- and no magnitude is
ever multiplied by, conditioned on, or signed by a direction. A strongly
bearish VOL_MOMENTUM therefore emits a LARGE magnitude and `direction = -1`,
and it is the gates' job to refuse a long. Nothing in this module takes a
majority vote across components, because a majority vote is precisely the
bug section 7 is warning about.

The consequence is worth stating plainly: `FlowScore.score` built from these
scorers is a magnitude aggregate and carries no directional opinion. A high
Flow Score means "a lot is happening", not "go long".

## Boundedness: enforced, not asserted

Every key a magnitude may be built from is listed in
`BOUNDED_MAGNITUDE_KEYS`, and every key a direction may be read from is
listed in `UNIT_DIRECTION_KEYS`. A scorer whose `magnitude_keys` names
anything else raises at construction. That is deliberate: the feature
modules emit unbounded natural-unit diagnostics next to their bounded
scores -- `achievable_rr`, `nearest_zone_price`, `spread_ticks`,
`participation_cost_ticks`, `atr`, `roc`, `relative_volume` -- and a weighted
sum that read one of those would let a single extreme observation dominate a
component. Those keys appear in `ComponentScore.detail` and nowhere else.

No transform is applied to a magnitude input, because each one is already
bounded in [0, 1] *by its own construction* in the feature layer (a convex
combination of bounded terms, a percentile rank, or a geometric mean of two
clipped ratios). Composing them is a convex combination, which is bounded by
the same argument. No z-score appears anywhere in this module.

An out-of-range magnitude input raises rather than being clamped. Clamping a
1.4 into 1.0 would turn a feature-layer regression into a quietly inflated
20-point component; `features/base.py` takes the same position on a
non-finite feature, for the same reason.

## UNAVAILABLE is not the same as zero

These two states produce the same `points` (0.0) and must never be confused:

* **Unavailable.** `enabled=False`, `quality=MISSING`, `magnitude=0.0`,
  `direction=0`. The number does not exist: the component is switched off in
  config, a key it reads is absent from the vector, or the key is graded
  MISSING (which is what `FeatureComputer._not_ready` stamps on the
  placeholder zeros it emits before warmup or without its feed). Because
  `enabled` is False, the weight drops out of `FlowScore.available_points`
  while `ComponentScore.weight` keeps its configured value -- so
  `max_points - available_points` reports the shortfall instead of hiding
  it, and `strict_component_availability` can refuse.
* **Measured as zero.** `enabled=True`, `quality` GOOD or DEGRADED,
  `magnitude=0.0`. The measurement was made and found no evidence. The full
  weight stays in `available_points`.

`detail["available"]` is 1.0 or 0.0 and `detail["quality_reason"]` carries a
code from `REASON_TEXT`, so the distinction survives into the trade record
and into rejection analysis. `explain()` renders the same thing as prose for
`Signal.reasons`.

Quality grades map to availability like this, and the line is drawn at
"was a measurement made at all":

    GOOD      -> available, nothing to report
    DEGRADED  -> available, flagged      (a real measurement, lower grade)
    STALE     -> available, flagged      (a real measurement, late)
    MISSING   -> UNAVAILABLE             (no measurement exists)

`score()` takes an optional `ComponentAvailability` -- `data/quality.py`'s
dataset-level verdict -- and consumes it rather than re-deriving which
components are computable. A refusal there makes the component unavailable
whatever the vector says; a `computable=True` verdict never overrides a
vector whose own keys are MISSING. `signals/scoring_flow.py` applies the
same rule to ORDER_FLOW and OPTIONS_FLOW, so all five components answer to
one availability report.

STALE is deliberately not disabled here. `DataConfig.min_quality_to_trade`
is the `DataQualityGate`'s to apply, at pipeline stage 2, before any scorer
runs; a scorer that applied it too would be a second, divergent source of
truth -- the same argument `features/liquidity.py` makes for not grading
staleness itself.

## Weights

`weight` comes from `config.flow_score.weights[component]`, which the schema
validates to sum to `total_points`. Nothing here hardcodes 20, 15 or 20, and
nothing here changes a default. Those weights are HYPOTHESES for the Phase 8
sweep (ARCHITECTURE.md principle 7); this module reads them and reports what
comes out.

The one weight this module does introduce is the VOL_MOMENTUM split between
its volatility half and its momentum half, which no config field carries.
See `VOL_MOMENTUM_WEIGHT_VOLATILITY`.

## Determinism

Pure functions over floats, in a fixed term order, with no mutable state, no
clock, no RNG and no iteration over an unordered container. Two calls on the
same `FeatureVector` return bit-identical `ComponentScore`s.

## Measured, at the last bar of a synthetic NQ series

Recorded because two of these numbers are surprising and the brief asks for
a surprise to be reported rather than smoothed away. Synthetic NQ, seed 11,
300s bars, 2024-01-02 to 2024-06-01, defaults everywhere, the five bars-only
computers in one `FeatureBundle`:

                       with quotes                 bars only
    STRUCTURE     mag 0.615  dir -1  DEGRADED   mag 0.615  dir -1  DEGRADED
    LIQUIDITY     mag 0.806  dir  0  GOOD       mag 0.500  dir  0  DEGRADED
    VOL_MOMENTUM  mag 0.705  dir -1  GOOD       mag 0.705  dir -1  GOOD

Three things worth stating:

1. **These three components sum to 55 of 100 available points.** A
   `FlowScore` built from them alone reports `available_points = 55.0`, and
   it is NOT a Flow Score -- every threshold in the config was calibrated
   against a 100-point scale. The aggregate, not this module, owns the
   refusal; `bars_scorers()` says so in its own docstring.
2. **STRUCTURE is DEGRADED on every bar, even with a quote feed.**
   `features/levels.py` grades its whole vector DEGRADED when `s_htf` has no
   higher interval to confirm against and `s_flow` has no tick feed, and a
   single-interval dataset has neither. Since `FlowScore.quality` is the
   worst grade over enabled components, the aggregate is DEGRADED on every
   bar of such a dataset. That is the honest reading of the data, not a
   defect to be relaxed, and `DataConfig.min_quality_to_trade` defaults to
   DEGRADED so it does not by itself stop trading -- but anyone expecting a
   GOOD Flow Score from bars should know it will never arrive.
3. **The liquidity magnitude is exactly 0.500 without quotes.** That is
   `NO_QUOTE_SCORE_CAP` binding, carried through unchanged: 7.5 of the
   component's 15 points, which is the stated consequence of half its
   evidence being absent.

Those magnitudes are measurements of the synthetic generator, not findings
about markets, and nothing above was tuned to produce them.

## This module makes no claim about profitability

It converts audited measurements into bounded, weighted, signed-separately
component scores. Whether any of it carries information is what Phases 8-11
are for, and none of these weights or blends is a finding.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from flow_model.config.schema import FlowModelConfig, FlowScoreConfig
from flow_model.core.contracts import ComponentScore, FeatureVector
from flow_model.core.enums import Component, DataQuality
from flow_model.data.quality import ComponentAvailability

__all__ = [
    "BOUNDED_MAGNITUDE_KEYS",
    "BarsComponentScorer",
    "EXPLAIN_KEY_LIMIT",
    "LiquidityScorer",
    "REASON_COMPONENT_DISABLED",
    "REASON_FEED_UNAVAILABLE",
    "REASON_KEYS_ABSENT",
    "REASON_KEY_DEGRADED",
    "REASON_KEY_MISSING",
    "REASON_KEY_STALE",
    "REASON_NONE",
    "REASON_TEXT",
    "ScorerError",
    "StructureScorer",
    "UNIT_DIRECTION_KEYS",
    "VOL_MOMENTUM_WEIGHT_MOMENTUM",
    "VOL_MOMENTUM_WEIGHT_VOLATILITY",
    "VolMomentumScorer",
    "bars_scorers",
    "unit_direction",
    "worst_quality",
]


class ScorerError(RuntimeError):
    """Raised when a scorer's declared contract is violated.

    Two kinds of violation reach here, and both are programming errors
    rather than data conditions: a scorer declared a magnitude key that is
    not bounded, or a feature module handed back a value outside the range
    its own docstring promises. Missing and degraded *data* is not an error
    and never raises -- it produces an unavailable or flagged
    `ComponentScore`.
    """


# ---------------------------------------------------------------------------
# What may enter a magnitude, and what may enter a direction
# ---------------------------------------------------------------------------

#: Feature keys a component magnitude may be built from.
#:
#: Each is bounded in [0, 1] by construction in the module that emits it, and
#: each is the key that module's own docstring nominates as "the one a
#: component score may read":
#:
#:   ``structure_magnitude``  features/levels.py     -- 14.6's w_S*S + w_C*C + w_R*R
#:   ``liquidity_score``      features/liquidity.py  -- sqrt(tightness * adequacy)
#:   ``vol_regime_score``     features/volatility.py -- 1 inside the tradable band
#:   ``momentum_score``       features/momentum.py   -- convex sum of three [0,1] terms
#:
#: The allowlist is checked at construction so that adding an unbounded key
#: to a magnitude is a build error, not a 20-point component dominated by one
#: outlier. Everything else a scorer reads is a diagnostic and goes to
#: `ComponentScore.detail`.
BOUNDED_MAGNITUDE_KEYS = frozenset(
    {
        "structure_magnitude",
        "liquidity_score",
        "vol_regime_score",
        "momentum_score",
    }
)

#: Feature keys a component direction may be read from.
#:
#: Each emits exactly {-1.0, 0.0, +1.0} with a documented meaning:
#:
#:   ``level_direction``  features/levels.py    -- +1 long at support, -1 short
#:                                                 at resistance, 0 no zone
#:   ``direction``        features/momentum.py  -- sign of the net move, 0 inside
#:                                                 the ATR dead band
#:
#: Signed-but-continuous keys are deliberately absent. `vol_expansion` is
#: signed, but its sign is expansion versus compression, not up versus down;
#: `depth_imbalance` is signed, but it is a fallback constant without a quote
#: feed. Turning either into a directional vote would manufacture a
#: direction, which is the failure this allowlist exists to prevent.
UNIT_DIRECTION_KEYS = frozenset({"level_direction", "direction"})

#: Tolerance on a magnitude input's declared [0, 1] bound, and on a
#: direction input's declared {-1, 0, +1}. Wide enough for float error in the
#: feature layer's convex sums, far too narrow to absorb a real regression.
BOUND_TOLERANCE = 1e-9

#: Below this absolute value a direction key reads as 0 rather than as a
#: sign. The three direction keys emit exact zeros, so this only guards
#: against a hand-built vector carrying a denormal; it is not a dead band,
#: and the dead band that matters lives in `features/momentum.py` where the
#: move it suppresses can be measured in ATR.
#:
#: It is `BOUND_TOLERANCE` and not a tighter number, because the two have to
#: agree about which values count as zero. They did not: at 1e-12 the window
#: `1e-12 <= |value| <= 1e-9` was accepted by the bound check as the member
#: 0 -- `round(value)` is 0 and the deviation is inside tolerance -- and then
#: returned as a FULL +-1 directional vote. `unit_direction` now reads the
#: member the bound check accepted, so this constant and that check cannot
#: draw the line in two different places.
DIRECTION_EPSILON = BOUND_TOLERANCE


# ---------------------------------------------------------------------------
# Quality reason codes
# ---------------------------------------------------------------------------
#
# `ComponentScore.detail` is `dict[str, float]`, so a categorical reason has
# to be encoded numerically -- the same constraint `features/levels.py` meets
# with `level_gate_reason`, and the same answer. `REASON_TEXT` renders the
# code, so the reason survives into the trade record as a code and into
# `Signal.reasons` as a sentence, and the two cannot drift.

REASON_NONE = 0.0
REASON_COMPONENT_DISABLED = 1.0
REASON_KEYS_ABSENT = 2.0
REASON_KEY_MISSING = 3.0
REASON_KEY_STALE = 4.0
REASON_KEY_DEGRADED = 5.0
REASON_FEED_UNAVAILABLE = 6.0

REASON_TEXT: Mapping[float, str] = {
    REASON_NONE: "all inputs measured and graded GOOD",
    REASON_COMPONENT_DISABLED: "component switched off in config.flow_score.enabled_components",
    REASON_KEYS_ABSENT: "a feature key this scorer reads is absent from the vector",
    REASON_KEY_MISSING: "a feature key this scorer reads is graded MISSING",
    REASON_KEY_STALE: "a feature key this scorer reads is graded STALE",
    REASON_KEY_DEGRADED: "a feature key this scorer reads is graded DEGRADED",
    REASON_FEED_UNAVAILABLE: (
        "data/quality.py reports this component is not computable from the dataset"
    ),
}

#: Quality grade -> reason code, for the grades that are reportable.
_QUALITY_REASON: Mapping[DataQuality, float] = {
    DataQuality.GOOD: REASON_NONE,
    DataQuality.DEGRADED: REASON_KEY_DEGRADED,
    DataQuality.STALE: REASON_KEY_STALE,
    DataQuality.MISSING: REASON_KEY_MISSING,
}


# ---------------------------------------------------------------------------
# VOL_MOMENTUM's internal split
# ---------------------------------------------------------------------------

#: Share of the VOL_MOMENTUM magnitude carried by `vol_regime_score`.
#:
#: A JUDGEMENT CALL, and the only weight this module introduces. Section 5
#: names the component "Volatility / momentum" and lists three features for
#: each half (ATR percentile, realized vol, vol-of-vol | ROC, efficiency
#: ratio, momentum acceleration) without ranking them, and no config field
#: carries the split -- `FlowScoreConfig.weights` is per *component*, not per
#: half. An equal split is therefore the neutral reading of section 5, and it
#: is also the first thing section 7's weight-testing plan asks for ("equal-
#: weight baseline"). It is a declared prior in the sense of principle 7, not
#: a finding, and it belongs in `config/schema.py` as a sweepable field;
#: adding one means editing a file this phase does not own, so it is reported
#: as a deviation instead.
VOL_MOMENTUM_WEIGHT_VOLATILITY: float = 0.5

#: Share of the VOL_MOMENTUM magnitude carried by `momentum_score`.
VOL_MOMENTUM_WEIGHT_MOMENTUM: float = 0.5


def _check_blend_weights() -> None:
    """The VOL_MOMENTUM split must be convex, checked at import.

    The magnitude's [0, 1] bound comes from the weights summing to 1 and
    nothing else, so a future edit that broke the sum would silently unbound
    a 20-point component. `features/momentum.py` guards its own score
    weights the same way.
    """
    total = VOL_MOMENTUM_WEIGHT_VOLATILITY + VOL_MOMENTUM_WEIGHT_MOMENTUM
    if abs(total - 1.0) > 1e-9:
        raise ScorerError(
            f"VOL_MOMENTUM blend weights sum to {total}, expected 1.0; the "
            "magnitude's [0, 1] bound comes from that sum"
        )
    if VOL_MOMENTUM_WEIGHT_VOLATILITY < 0.0 or VOL_MOMENTUM_WEIGHT_MOMENTUM < 0.0:
        raise ScorerError("VOL_MOMENTUM blend weights must be non-negative")


_check_blend_weights()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def unit_direction(value: float) -> int:
    """Sign of a unit-valued direction feature, as {-1, 0, +1}.

    Sign extraction, not a transform: the keys in `UNIT_DIRECTION_KEYS`
    already emit exactly -1.0, 0.0 or +1.0, so nothing is lost. A value that
    is not one of the three means the emitting module broke its own contract
    and raises rather than being silently reduced to a sign.

    The in-range case raises too, which it did not originally. A fractional
    direction -- 0.3, say -- is a magnitude wearing a direction's name,
    exactly the conflation section 7 separates, and reducing it to a sign
    would promote "weakly bullish" to a FULL directional vote carrying the
    component's whole weight into `FlowScore.opposing_points` and into
    section 3's order-flow gate, which blocks on any opposing direction
    regardless of magnitude. The argument for sign extraction ("the keys
    already emit exactly -1, 0 or +1") is an assumption about the producing
    module, and this is the one place it can be checked;
    `signals/scoring_flow.py` checks it for its own two keys.
    """
    if not math.isfinite(value):
        raise ScorerError(f"direction feature is not finite: {value}")
    nearest = round(value)
    if abs(value - nearest) > BOUND_TOLERANCE or nearest not in (-1, 0, 1):
        raise ScorerError(
            f"direction feature {value} is outside {{-1, 0, +1}}; the feature "
            "module that emitted it has broken its declared contract. A "
            "fractional direction is a magnitude wearing a direction's name, "
            "and reducing it to a sign would cast a full-weight vote the "
            "measurement does not support."
        )
    # The member of {-1, 0, +1} the check above already accepted this value
    # AS, and not an independent sign reading of it. Taking the sign instead
    # disagreed with the check in the window `DIRECTION_EPSILON <= |value| <=
    # BOUND_TOLERANCE`: a 1e-10 was validated as the member 0 and then
    # returned as a full +1 vote, which `signals/scoring_flow.py` reads as 0
    # on the identical input. See `DIRECTION_EPSILON`.
    return int(nearest)


def _bounded_magnitude_input(key: str, value: float) -> float:
    """One magnitude input, validated against its declared [0, 1] bound."""
    if not math.isfinite(value):
        raise ScorerError(f"magnitude input {key!r} is not finite: {value}")
    if value < -BOUND_TOLERANCE or value > 1.0 + BOUND_TOLERANCE:
        raise ScorerError(
            f"magnitude input {key!r} is {value}, outside its declared [0, 1] "
            "bound. Clamping would turn a feature-layer regression into a "
            "quietly inflated component score, so this raises instead."
        )
    return 0.0 if value <= 0.0 else (1.0 if value >= 1.0 else float(value))


def worst_quality(features: FeatureVector, keys: Iterable[str]) -> DataQuality:
    """Worst grade among `keys`, which is the component's grade.

    Deliberately NOT `features.quality`. The bundle's aggregate is the worst
    grade over *every* computer in it, including ones whose feed is absent,
    so on a bars-only dataset it is MISSING for reasons that have nothing to
    do with these three components. `FeatureVector` carries a grade per key
    precisely so a consumer can ask about the keys it actually reads, and
    that is the only honest granularity for a per-component score.

    A key absent from `quality_by_key` grades MISSING via
    `FeatureVector.quality_of`, which is correct: a key nobody declared a
    grade for has no measurement behind it.
    """
    keys = tuple(keys)
    if not keys:
        return DataQuality.MISSING
    return DataQuality.worst(*(features.quality_of(key) for key in keys))


def _absent_keys(features: FeatureVector, keys: Iterable[str]) -> tuple[str, ...]:
    """Keys with no value in the vector at all, in declaration order."""
    return tuple(key for key in keys if key not in features.values)


def _keys_at_grade(
    features: FeatureVector, keys: Iterable[str], grade: DataQuality
) -> tuple[str, ...]:
    return tuple(key for key in keys if features.quality_of(key) is grade)


#: Keys named individually in one `explain()` line before it summarizes.
#:
#: `features/levels.py` grades its whole 21-key vector uniformly, so a
#: single-interval dataset degrades every one of them at once and an
#: exhaustive list is a wall of text that hides the one sentence that
#: matters. The first few keys identify the module; the count carries the
#: rest.
EXPLAIN_KEY_LIMIT = 6


def _named(keys: tuple[str, ...]) -> str:
    """`keys` as a readable list, truncated past `EXPLAIN_KEY_LIMIT`."""
    if len(keys) <= EXPLAIN_KEY_LIMIT:
        return str(list(keys))
    shown = list(keys[:EXPLAIN_KEY_LIMIT])
    return f"{shown} and {len(keys) - EXPLAIN_KEY_LIMIT} more"


# ---------------------------------------------------------------------------
# The scorer base
# ---------------------------------------------------------------------------


class BarsComponentScorer(ABC):
    """One component's `(magnitude, direction, weight, quality)`.

    Constructed from `config.flow_score` and nothing else. There is no
    market-data parameter and no accessor through which one could arrive,
    which is what makes the point-in-time guarantee inherited rather than
    re-argued: a scorer can only see what the audited feature layer put in
    the `FeatureVector` it is handed.

    Subclasses declare four tuples of feature keys -- `magnitude_keys`,
    `direction_keys`, `detail_keys` and (derived) `reads` -- so that a change
    to a feature module which breaks a scorer is traceable from either end.
    `magnitude_keys` and `direction_keys` are disjoint by construction and
    are consumed by separate methods, which is how section 7's separation is
    enforced structurally rather than by convention.

    ## The scope of `quality`

    `ComponentScore.quality` is the worst grade over `required_keys` -- the
    magnitude and direction inputs -- and not over `reads`. The reason is
    that `quality` travels into `FlowScore.quality` and from there into
    `DataConfig.min_quality_to_trade`, so it has to answer "how good is the
    number this component contributed", not "is every diagnostic beside it
    pristine". A DEGRADED `depth_imbalance` on a bar whose `liquidity_score`
    is a clean measurement would otherwise block a trade for a reason that
    changed nothing about the score.

    A diagnostic below GOOD is still reported, in
    `detail["degraded_diagnostic_count"]` and in `explain()`, so the
    information is available to anything that wants to be stricter. It is
    reported rather than enforced.
    """

    #: The component this scorer produces.
    component: Component = Component.STRUCTURE

    #: Keys combined into the magnitude. Must be a subset of
    #: `BOUNDED_MAGNITUDE_KEYS`.
    magnitude_keys: tuple[str, ...] = ()

    #: Keys the direction is read from. Must be a subset of
    #: `UNIT_DIRECTION_KEYS`. Empty means the component has no directional
    #: opinion and always emits 0.
    direction_keys: tuple[str, ...] = ()

    #: Keys reported in `ComponentScore.detail` but never summed into the
    #: magnitude. Natural-unit and signed-continuous features live here.
    detail_keys: tuple[str, ...] = ()

    def __init__(self, config: FlowModelConfig | FlowScoreConfig) -> None:
        """Configuration only: the root config, or just its `flow_score`.

        Both are accepted so that all five component scorers construct the
        same way. `signals/scoring_flow.py` -- the ORDER_FLOW and
        OPTIONS_FLOW scorers, owned separately -- takes a `FlowModelConfig`,
        because those components read `config.order_flow` and
        `config.options_flow`; these three need nothing but
        `config.flow_score`, so the narrower object is also allowed and is
        what the class actually keeps. An aggregator can therefore write one
        construction loop over all five.

        There is no market-data parameter in either form. That is the ban
        `features/base.py` documents for computers and
        `tests/unit/test_feature_contracts.py` enforces: a full-sample
        statistic captured at construction is the one leak the lookahead
        audit cannot see through when it is handed an instance.
        """
        if isinstance(config, FlowModelConfig):
            flow_score = config.flow_score
        elif isinstance(config, FlowScoreConfig):
            flow_score = config
        else:
            raise ScorerError(
                f"{type(self).__name__} takes a FlowModelConfig or a "
                f"FlowScoreConfig and nothing else; got {type(config).__name__}. "
                "A scorer that accepted market data could carry a full-sample "
                "statistic past the lookahead audit."
            )
        self._flow_score = flow_score
        self._validate_declaration()

    # --- declaration checks -------------------------------------------

    def _validate_declaration(self) -> None:
        if not self.magnitude_keys:
            raise ScorerError(f"{self.name}: declares no magnitude keys")
        unbounded = sorted(set(self.magnitude_keys) - BOUNDED_MAGNITUDE_KEYS)
        if unbounded:
            raise ScorerError(
                f"{self.name}: magnitude keys {unbounded} are not in "
                "BOUNDED_MAGNITUDE_KEYS. An unbounded term lets one extreme "
                "observation dominate a weighted sum; report it in detail instead."
            )
        not_unit = sorted(set(self.direction_keys) - UNIT_DIRECTION_KEYS)
        if not_unit:
            raise ScorerError(
                f"{self.name}: direction keys {not_unit} are not in "
                "UNIT_DIRECTION_KEYS. A direction must be {-1, 0, +1}; a signed "
                "continuous feature is not a direction."
            )
        overlap = sorted(set(self.magnitude_keys) & set(self.direction_keys))
        if overlap:
            raise ScorerError(
                f"{self.name}: {overlap} is read as both magnitude and direction. "
                "Section 7 keeps the two separate so a bearish component cannot "
                "inflate a bullish score."
            )
        if self.component not in self._flow_score.weights:
            raise ScorerError(
                f"{self.name}: config.flow_score.weights carries no weight for "
                f"{self.component.value}"
            )

    # --- declared surface ----------------------------------------------

    @property
    def name(self) -> str:
        return f"{type(self).__name__}({self.component.value})"

    @property
    def reads(self) -> tuple[str, ...]:
        """Every feature key this scorer touches, magnitude first.

        Order is declaration order with duplicates removed, so it is stable
        and readable rather than sorted into meaninglessness.
        """
        seen: dict[str, None] = {}
        for key in self.magnitude_keys + self.direction_keys + self.detail_keys:
            seen.setdefault(key, None)
        return tuple(seen)

    @property
    def required_keys(self) -> tuple[str, ...]:
        """Keys without which there is no score: the magnitude and direction
        inputs. A missing `detail_keys` entry costs a diagnostic, not the
        component, so it is reported rather than fatal."""
        return self.magnitude_keys + self.direction_keys

    @property
    def weight(self) -> float:
        """Points this component contributes at magnitude 1.0.

        Straight from `config.flow_score.weights`. Never hardcoded, never
        adjusted here.
        """
        return float(self._flow_score.weights[self.component])

    @property
    def configured_enabled(self) -> bool:
        return bool(self._flow_score.enabled_components.get(self.component, True))

    # --- subclass hooks ------------------------------------------------

    @abstractmethod
    def _magnitude(self, values: Mapping[str, float]) -> float:
        """Magnitude in [0, 1] from the validated magnitude inputs."""

    def _direction(self, values: Mapping[str, float]) -> int:
        """Direction in {-1, 0, +1}. Default: no directional opinion."""
        return 0

    def _derived_detail(self, features: FeatureVector) -> dict[str, float]:
        """Extra diagnostics computed from keys already in `reads`."""
        return {}

    # --- the public call ------------------------------------------------

    def score(
        self,
        features: FeatureVector,
        availability: ComponentAvailability | None = None,
    ) -> ComponentScore:
        """This component's score at `features.ts`.

        Short-circuits to an unavailable score -- `enabled=False`,
        `quality=MISSING`, magnitude 0, direction 0 -- when the component is
        switched off in config, when `availability` refuses it, when a
        required key is absent, or when a required key is graded MISSING. In
        every other case the measurement exists, the component is enabled,
        and `quality` carries its grade.

        `availability`, when supplied, is `data/quality.py`'s DATASET-level
        verdict for this component (`QualityGrader.availability(...)
        .components[component]`). It is CONSUMED, never re-derived from feeds
        and feed requirements a second time. Only a refusal propagates: a
        `computable=True` verdict is a statement about the whole dataset and
        must not talk a point-in-time vector out of its own MISSING keys,
        because `availability()` reads the series while the vector reads one
        instant. Its `quality` is ANDed in through `DataQuality.worst`, so a
        dataset graded DEGRADED cannot be reported GOOD on a bar that
        happens to look clean. `signals/scoring_flow.py` applies the same
        rule to the other two components.
        """
        if not self.configured_enabled:
            return self._unavailable(REASON_COMPONENT_DISABLED)

        if availability is not None:
            self._check_availability_subject(availability)
            if not availability.computable:
                return self._unavailable(
                    REASON_FEED_UNAVAILABLE,
                    missing_feed_count=float(len(availability.missing_required_feeds)),
                )

        absent = _absent_keys(features, self.required_keys)
        if absent:
            return self._unavailable(
                REASON_KEYS_ABSENT, absent_count=float(len(absent))
            )

        key_grade = worst_quality(features, self.required_keys)
        if key_grade is DataQuality.MISSING:
            return self._unavailable(
                REASON_KEY_MISSING,
                missing_count=float(
                    len(_keys_at_grade(features, self.required_keys, DataQuality.MISSING))
                ),
            )
        # The dataset's grade and this bar's grade are both true, so the
        # component's grade is the worse of them.
        grade = (
            key_grade
            if availability is None
            else DataQuality.worst(key_grade, availability.quality)
        )
        if grade is DataQuality.MISSING:
            return self._unavailable(REASON_FEED_UNAVAILABLE)

        magnitude_values = {
            key: _bounded_magnitude_input(key, features.require(key))
            for key in self.magnitude_keys
        }
        direction_values = {key: features.require(key) for key in self.direction_keys}

        magnitude = self._magnitude(magnitude_values)
        if (
            not math.isfinite(magnitude)
            or magnitude < -BOUND_TOLERANCE
            or magnitude > 1.0 + BOUND_TOLERANCE
        ):
            raise ScorerError(
                f"{self.name}: produced magnitude {magnitude}, outside [0, 1]"
            )
        magnitude = 0.0 if magnitude <= 0.0 else (1.0 if magnitude >= 1.0 else magnitude)

        detail = self._detail(features, grade)
        return ComponentScore(
            component=self.component,
            magnitude=magnitude,
            direction=self._direction(direction_values),
            weight=self.weight,
            quality=grade,
            enabled=True,
            features_used=self.reads,
            detail=detail,
        )

    def _check_availability_subject(self, availability: ComponentAvailability) -> None:
        """A verdict about another component is not evidence about this one."""
        if availability.component is not self.component:
            raise ScorerError(
                f"availability verdict is for {availability.component.value} but "
                f"this scorer produces {self.component.value}; scoring one "
                "component against another's feed verdict would report the "
                "wrong gap."
            )

    def explain(
        self,
        features: FeatureVector,
        availability: ComponentAvailability | None = None,
    ) -> tuple[str, ...]:
        """Prose reasons for this component's quality, for `Signal.reasons`.

        Says the same thing as `detail["quality_reason"]` and names the
        specific keys, which a float-valued code cannot. Cheap: it reads
        grades and does no arithmetic, so the engine can call it on the bars
        it decides to record without paying for it on the rest.

        Empty means every required key is GOOD and there is nothing to
        report, which is the only state with nothing to say.

        A diagnostic is mentioned only when its grade is strictly WORSE than
        the component's own, because that is the only case the component
        grade does not already cover. `detail["degraded_diagnostic_count"]`
        counts every diagnostic below GOOD, so the broader number is still
        available to anything that wants it.
        """
        label = self.component.value
        if not self.configured_enabled:
            return (f"{label}: {REASON_TEXT[REASON_COMPONENT_DISABLED]}",)

        if availability is not None:
            self._check_availability_subject(availability)
            if not availability.computable:
                note = (
                    availability.note.strip()
                    or "the dataset does not support this component"
                )
                feeds = ", ".join(
                    f.value for f in availability.missing_required_feeds
                )
                tail = f" Missing required feed(s): {feeds}." if feeds else ""
                return (
                    f"{label}: UNAVAILABLE, {REASON_TEXT[REASON_FEED_UNAVAILABLE]} "
                    f"-- {note}.{tail} Reported unavailable rather than as a zero "
                    "magnitude, and its weight is not redistributed.",
                )

        absent = _absent_keys(features, self.required_keys)
        if absent:
            return (
                f"{label}: UNAVAILABLE, feature key(s) {_named(absent)} absent "
                "from the vector",
            )

        missing = _keys_at_grade(features, self.required_keys, DataQuality.MISSING)
        if missing:
            return (
                f"{label}: UNAVAILABLE, feature key(s) {_named(missing)} graded "
                "MISSING -- no measurement exists, so the component is not "
                "scored and its weight is reported as unavailable",
            )

        grade = worst_quality(features, self.required_keys)
        reasons: list[str] = []
        if availability is not None and availability.quality.rank < grade.rank:
            if availability.quality is DataQuality.MISSING:
                # `score()` returns an UNAVAILABLE ComponentScore here (the
                # combined grade is MISSING), so saying anything else would
                # make `explain()` and `detail["quality_reason"]` tell two
                # different stories about one bar. Falling through would
                # report "magnitude/direction key(s) [] graded MISSING" --
                # an empty key list, because the MISSING grade came from the
                # dataset verdict and not from any key.
                return (
                    f"{label}: UNAVAILABLE, {REASON_TEXT[REASON_FEED_UNAVAILABLE]} "
                    f"-- data/quality.py grades the dataset MISSING for this "
                    f"component, so no measurement of it exists for this run "
                    f"even though this bar's own key(s) are {grade.value}. "
                    "Reported unavailable rather than as a zero magnitude, and "
                    "its weight is not redistributed.",
                )
            reasons.append(
                f"{label}: {availability.quality.value}, data/quality.py grades "
                f"the dataset {availability.quality.value} for this component "
                f"even though this bar's keys are {grade.value}"
            )
            grade = availability.quality
        if grade is not DataQuality.GOOD:
            hit = _keys_at_grade(features, self.required_keys, grade)
            # Only when a key is actually AT this grade. When `grade` came
            # from the dataset verdict a line above, no key is, and this
            # reported "magnitude/direction key(s) [] graded DEGRADED" -- the
            # same empty key list the MISSING branch was written to prevent,
            # for the two grades that branch does not cover, contradicting
            # the line that just said the keys are GOOD. In the ordinary path
            # `grade` IS some key's grade, so `hit` is non-empty and this
            # guard never fires.
            if hit:
                reasons.append(
                    f"{label}: {grade.value}, magnitude/direction key(s) "
                    f"{_named(hit)} graded {grade.value}"
                )
        for worse in (DataQuality.MISSING, DataQuality.STALE, DataQuality.DEGRADED):
            if worse.rank >= grade.rank:
                continue
            hit = tuple(
                key
                for key in _keys_at_grade(features, self.detail_keys, worse)
                if key in features.values
            )
            if hit:
                reasons.append(
                    f"{label}: diagnostic key(s) {_named(hit)} graded "
                    f"{worse.value}; they are reported in detail and do not "
                    "enter the magnitude, so the component's grade is unchanged"
                )
        return tuple(reasons)

    # --- construction helpers -------------------------------------------

    def _unavailable(self, reason: float, **extra: float) -> ComponentScore:
        """An unavailable component: no measurement, weight kept, enabled off.

        The weight is kept at its configured value so that
        `FlowScore.max_points - FlowScore.available_points` reports the
        shortfall. `enabled=False` is what takes it out of
        `available_points`; zeroing the weight instead would make a 100-point
        scale out of 80 points and hide exactly what
        `strict_component_availability` exists to surface.

        `detail` carries no feature values at all. The vector's placeholders
        in this state are `FeatureComputer._not_ready`'s zeros, and copying a
        placeholder into a diagnostic would read as a measurement of zero.
        """
        detail = {
            "available": 0.0,
            "quality_reason": reason,
            "quality_rank": float(DataQuality.MISSING.rank),
        }
        detail.update(extra)
        return ComponentScore(
            component=self.component,
            magnitude=0.0,
            direction=0,
            weight=self.weight,
            quality=DataQuality.MISSING,
            enabled=False,
            # Names what the scorer needed and did not have, which is the
            # whole point of the field being here.
            features_used=self.reads,
            detail=detail,
        )

    def _detail(self, features: FeatureVector, grade: DataQuality) -> dict[str, float]:
        """Diagnostics for an available score.

        Every `detail_keys` entry that is present, plus whatever
        `_derived_detail` adds, plus the availability triple. A `detail_keys`
        entry that is absent is simply omitted -- its absence is itself the
        report, and substituting a zero would read as a measurement.
        """
        detail: dict[str, float] = {}
        for key in self.detail_keys:
            value = features.get(key)
            if value is not None and math.isfinite(value):
                detail[key] = float(value)
        detail.update(self._derived_detail(features))
        detail["available"] = 1.0
        detail["quality_reason"] = _QUALITY_REASON[grade]
        detail["quality_rank"] = float(grade.rank)
        # A diagnostic below GOOD does not change the component's grade --
        # it does not enter the magnitude -- but it must not vanish either,
        # or `depth_imbalance` being a fallback would be invisible on a
        # component whose own score is a clean measurement.
        detail["degraded_diagnostic_count"] = float(
            sum(
                1
                for key in self.detail_keys
                if key in features.values
                and features.quality_of(key) is not DataQuality.GOOD
            )
        )
        return detail


# ---------------------------------------------------------------------------
# STRUCTURE -- 20 points on the brief's prior
# ---------------------------------------------------------------------------


class StructureScorer(BarsComponentScorer):
    """The STRUCTURE component, read off the section 14 S/R engine.

    ## Feature keys read

    Magnitude (`features/levels.py`):
        ``structure_magnitude``

    Direction (`features/levels.py`):
        ``level_direction``

    Diagnostics, reported in `detail`, never summed into the magnitude
    (`features/levels.py`):
        ``level_significance`` ``level_cleanliness`` ``level_rejection``
        ``rejection_confirmed`` ``structure_gate_passed`` ``level_gate_reason``
        ``level_setup_class`` ``achievable_rr`` ``zone_count``
        ``major_zone_count`` ``nearest_zone_distance_atr``
        ``nearest_zone_gap_atr`` ``level_touch_count`` ``level_recent_touches``
        ``level_flow_available``

    Diagnostics, reported in `detail` (`features/structure.py`):
        ``structure_score`` ``structure_direction`` ``bos_direction`` ``choch``
        ``vwap_deviation_atr`` ``opening_range_position``

    Derived in `detail`: ``geometry_agreement`` ``bos_agreement``

    ## Why the magnitude is one key

    Section 14.6 is explicit: "The STRUCTURE component's 20 points then come
    from `magnitude = w_S*S + w_C*C + w_R*R`, with `direction = +1` at
    support and `-1` at resistance." `features/levels.py` already computes
    exactly that expression -- with `w_S`, `w_C`, `w_R` from
    `config.levels.magnitude_weights` -- and emits it as
    `structure_magnitude`. So the magnitude IS that key, passed through. S, C
    and R are not re-derived here, and no second formula is composed on top
    of 14.6's, because a scorer that blended in another term would be
    scoring something section 14.6 does not define while using its name.

    ## What `features/structure.py` contributes, and what it does not

    The pivot/VWAP/session geometry is consumed as *corroboration and
    diagnostics*, not as magnitude or direction. `structure_score` is
    `features/structure.py`'s own bounded geometry-agreement magnitude, and
    `structure_direction`/`bos_direction` are its HH-HL sequence and
    break-of-structure readings; all three are reported in `detail`, together
    with two products that state the corroboration directly:

        geometry_agreement = structure_direction * level_direction
        bos_agreement      = bos_direction      * level_direction

    +1 means the pivot sequence (or the last break) points the same way as
    the trade the level implies -- a long at support inside an HH/HL
    sequence. -1 means it points the other way: a long at support while the
    sequence is LH/LL. 0 means one of them is flat.

    These are deliberately *reported and not folded in*. Section 14.6 fixes
    both the magnitude formula and the direction rule, and a scorer that
    averaged `structure_score` into the magnitude, or let
    `structure_direction` override `level_direction`, would be making a
    different component than the one the binding document specifies. Whether
    geometry disagreement should block a trade is a gate question -- gates
    express "must have", scores express "how good" (14.6) -- so the number is
    put where a gate can read it and no further.

    ## Direction

    `level_direction` verbatim: +1 is a long at support, -1 a short at
    resistance, 0 no zone or price inside one. It is read by a separate code
    path from the magnitude and never multiplied into it, so a strong
    resistance setup contributes a large magnitude with `direction = -1` and
    the gates decide what that forbids.
    """

    component = Component.STRUCTURE
    magnitude_keys = ("structure_magnitude",)
    direction_keys = ("level_direction",)
    detail_keys = (
        # the three section 14 terms, as emitted -- reported so 14.7's
        # falsification tests can bucket outcomes by S and by C
        "level_significance",
        "level_cleanliness",
        "level_rejection",
        "rejection_confirmed",
        # 14.6's gate outcome and the setup it measures off the structure
        "structure_gate_passed",
        "level_gate_reason",
        "level_setup_class",
        "achievable_rr",
        # zone population and geometry, in natural units
        "zone_count",
        "major_zone_count",
        "nearest_zone_distance_atr",
        "nearest_zone_gap_atr",
        "level_touch_count",
        "level_recent_touches",
        "level_flow_available",
        # features/structure.py's independent geometry reading
        "structure_score",
        "structure_direction",
        "bos_direction",
        "choch",
        "vwap_deviation_atr",
        "opening_range_position",
    )

    def _magnitude(self, values: Mapping[str, float]) -> float:
        """14.6's `w_S*S + w_C*C + w_R*R`, already computed by `levels.py`."""
        return values["structure_magnitude"]

    def _direction(self, values: Mapping[str, float]) -> int:
        return unit_direction(values["level_direction"])

    def _derived_detail(self, features: FeatureVector) -> dict[str, float]:
        """Agreement between the level direction and the bar geometry.

        A product of two already-emitted unit directions, so it introduces no
        new formula and cannot disagree with its inputs.
        """
        level = features.get("level_direction")
        if level is None:
            return {}
        derived: dict[str, float] = {}
        for source, name in (
            ("structure_direction", "geometry_agreement"),
            ("bos_direction", "bos_agreement"),
        ):
            other = features.get(source)
            if other is not None and math.isfinite(other):
                derived[name] = float(other) * float(level)
        return derived


# ---------------------------------------------------------------------------
# LIQUIDITY -- 15 points on the brief's prior
# ---------------------------------------------------------------------------


class LiquidityScorer(BarsComponentScorer):
    """The LIQUIDITY component, including its flagged DEGRADED path.

    ## Feature keys read

    Magnitude (`features/liquidity.py`):
        ``liquidity_score``

    Direction: none. See below.

    Diagnostics, reported in `detail`, never summed into the magnitude
    (`features/liquidity.py`):
        ``spread_ticks`` ``spread_percentile`` ``depth_imbalance``
        ``volume_percentile`` ``relative_volume`` ``volume_trend``
        ``dollar_volume`` ``participation_cost_ticks``

    Derived in `detail`: ``spread_measured`` ``profile_measured``

    ## Carrying the degradation through

    Section 5: liquidity "degrades to volume-percentile only, flagged
    DEGRADED". `features/liquidity.py` honours the first half -- without a
    quote feed `spread_ticks` falls back to `spec.typical_spread_ticks`, the
    tightness term is pinned at the no-information value, `liquidity_score`
    is capped at `NO_QUOTE_SCORE_CAP`, and each fallback is named in a note
    -- and it honours the second half by grading `liquidity_score` DEGRADED
    whenever the spread percentile or the time-of-day profile is a fallback.

    This scorer's job is the last link in that chain. `quality` is the worst
    grade over the keys the magnitude and direction are built from, so a
    DEGRADED `liquidity_score` produces a DEGRADED `ComponentScore`, which
    `FlowScore.quality` (the worst over enabled components) then surfaces to
    the decision. Dropping the flag here would end the degradation report at
    the feature layer, where nothing acts on it.

    DEGRADED is reported, not disabled. The component still functions on
    bars -- section 5 says "degrades", not "unavailable" -- so its 15 points
    stay in `available_points` and the grade travels with them. Marking it
    unavailable instead would shrink the scale by 15 points and trigger a
    strict-availability refusal that section 5 does not ask for. Which half
    of the evidence is missing is recoverable from
    `detail["spread_measured"]` and `detail["profile_measured"]`, and from
    `explain()`.

    ## Why this component has no direction

    `direction` is always 0, and that is a JUDGEMENT CALL worth defending.

    Liquidity answers "can this trade be executed at an acceptable cost",
    which is a permission, not an opinion about which way price goes. The one
    signed key in the module, `depth_imbalance`, is top-of-book size skew:
    without a quote feed it is a fallback constant, so a direction read off
    it would be manufactured from configuration on exactly the dataset this
    component is supposed to degrade on. Emitting 0 also keeps 15 points out
    of `FlowScore.opposing_points`, which is correct -- thin liquidity should
    block a trade through the `LiquidityGate` and `WaitReason.LIQUIDITY`,
    where the reason is recorded, and not as a silent directional veto.

    `depth_imbalance` is still reported in `detail` so a gate or a later
    analysis can use it; it is excluded from the magnitude, not discarded.
    """

    component = Component.LIQUIDITY
    magnitude_keys = ("liquidity_score",)
    direction_keys = ()
    detail_keys = (
        "spread_ticks",
        "spread_percentile",
        "depth_imbalance",
        "volume_percentile",
        "relative_volume",
        "volume_trend",
        "dollar_volume",
        "participation_cost_ticks",
    )

    def _magnitude(self, values: Mapping[str, float]) -> float:
        """`liquidity_score`: `sqrt(tightness * adequacy)`, capped on the
        degraded path by `features/liquidity.py` before it reaches here."""
        return values["liquidity_score"]

    def _derived_detail(self, features: FeatureVector) -> dict[str, float]:
        """Which half of the evidence was measured.

        Read off the per-key grades rather than recomputed: `liquidity.py`
        grades `spread_ticks` DEGRADED exactly when the spread is a fallback,
        and `relative_volume` DEGRADED exactly when the time-of-day bucket
        could not supply a median. Re-deriving either from the values would
        be a second, divergent test.
        """
        return {
            "spread_measured": (
                1.0 if features.quality_of("spread_ticks") is DataQuality.GOOD else 0.0
            ),
            "profile_measured": (
                1.0
                if features.quality_of("relative_volume") is DataQuality.GOOD
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# VOL_MOMENTUM -- 20 points on the brief's prior
# ---------------------------------------------------------------------------


class VolMomentumScorer(BarsComponentScorer):
    """The VOL_MOMENTUM component, combining volatility and momentum.

    ## Feature keys read

    Magnitude:
        ``vol_regime_score``   (`features/volatility.py`)
        ``momentum_score``     (`features/momentum.py`)

    Direction:
        ``direction``          (`features/momentum.py`)

    Diagnostics, reported in `detail`, never summed into the magnitude
    (`features/volatility.py`):
        ``atr`` ``atr_pct`` ``atr_percentile`` ``realized_vol``
        ``parkinson_vol`` ``garman_klass_vol`` ``vol_of_vol``
        ``vol_expansion``

    Diagnostics, reported in `detail` (`features/momentum.py`):
        ``roc`` ``roc_atr`` ``efficiency_ratio`` ``acceleration``
        ``momentum_persistence`` ``up_bar_fraction`` ``close_position``

    Derived in `detail`: ``volatility_term`` ``momentum_term``

    ## The magnitude is a sum, not a product

        magnitude = w_v * vol_regime_score + w_m * momentum_score

    with `w_v = VOL_MOMENTUM_WEIGHT_VOLATILITY` and
    `w_m = VOL_MOMENTUM_WEIGHT_MOMENTUM`, which sum to 1 and are checked at
    import. Both inputs are in [0, 1], so the result is too, by the same
    convex-combination argument -- no clip is doing load-bearing work.

    Additive rather than multiplicative, and the reason is in section 14.6:
    "Using a product instead would let one weak term zero an otherwise strong
    setup, which is a gate's job, not a score's." `vol_regime_score` is 0 at
    both volatility extremes, so a product would zero the whole 20-point
    component on a dead or a panicking tape. Section 3 already has a
    `VolatilityGate` that rejects a bar outside the setup's ATR-percentile
    band and records `WaitReason.VOLATILITY_BAND`; the exclusion belongs
    there, where it is a stated rejection, not here, where it would be an
    invisible zero.

    `vol_regime_score` rather than `atr_percentile` is the volatility input
    because the component's magnitude should say "the range is in the band
    where a stop and a target are both payable", not "the range is large". A
    high ATR percentile is not more evidence of anything; it is one end of a
    band whose other end is equally unhelpful, which is what
    `tradable_band_score` encodes and a raw percentile does not.

    ## The three volatility estimators are not cross-checked

    `features/volatility.py` deliberately does not assert that `realized_vol`,
    `parkinson_vol` and `garman_klass_vol` agree -- they run roughly 2x apart
    on the synthetic generator by construction -- and this module adds no
    such assumption. All three are reported in `detail` as separate
    diagnostics; none is averaged with, divided by, or validated against
    another, and none enters the magnitude. An agreement test here would bake
    in an assumption the feature layer explicitly declined to make.

    ## Direction comes from momentum alone

    `direction` is `features/momentum.py`'s sign of the net move, already
    suppressed to 0 inside an ATR-scaled dead band. Volatility contributes no
    direction and must not: `vol_expansion` is signed, but its sign is
    expansion versus compression, and reading it as up versus down would
    invent a directional vote out of a range measurement.

    Magnitude and direction are built by separate methods from disjoint keys,
    so a hard bearish impulse emits a LARGE magnitude with `direction = -1`.
    The magnitude does not shrink because the move is down, and the direction
    does not grow because the move is large. What happens when that disagrees
    with STRUCTURE is the gates' decision, not this scorer's.
    """

    component = Component.VOL_MOMENTUM
    magnitude_keys = ("vol_regime_score", "momentum_score")
    direction_keys = ("direction",)
    detail_keys = (
        # volatility, in natural units and as the bounded signed ratio
        "atr",
        "atr_pct",
        "atr_percentile",
        "realized_vol",
        "parkinson_vol",
        "garman_klass_vol",
        "vol_of_vol",
        "vol_expansion",
        # momentum
        "roc",
        "roc_atr",
        "efficiency_ratio",
        "acceleration",
        "momentum_persistence",
        "up_bar_fraction",
        "close_position",
    )

    def _magnitude(self, values: Mapping[str, float]) -> float:
        """Convex combination of the two halves, in a fixed term order."""
        return (
            VOL_MOMENTUM_WEIGHT_VOLATILITY * values["vol_regime_score"]
            + VOL_MOMENTUM_WEIGHT_MOMENTUM * values["momentum_score"]
        )

    def _direction(self, values: Mapping[str, float]) -> int:
        return unit_direction(values["direction"])

    def _derived_detail(self, features: FeatureVector) -> dict[str, float]:
        """The two weighted halves, so the blend is visible in the record.

        Without these, a 0.5 magnitude could be a strong trend in a dead tape
        or a flat tape in a perfect volatility band, and the trade record
        could not tell the two apart.
        """
        derived: dict[str, float] = {}
        vol = features.get("vol_regime_score")
        mom = features.get("momentum_score")
        if vol is not None and math.isfinite(vol):
            derived["volatility_term"] = VOL_MOMENTUM_WEIGHT_VOLATILITY * float(vol)
        if mom is not None and math.isfinite(mom):
            derived["momentum_term"] = VOL_MOMENTUM_WEIGHT_MOMENTUM * float(mom)
        return derived


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def bars_scorers(
    config: FlowModelConfig | FlowScoreConfig,
) -> tuple[BarsComponentScorer, ...]:
    """The three bars-only scorers, in a fixed order.

    `config` is the root `FlowModelConfig` or just its `flow_score`, so this
    and `signals/scoring_flow.py`'s `feed_dependent_scorers(config)` can be
    called with the same object. The order is fixed so that
    anything iterating these -- an aggregate, a report, a hash -- is
    deterministic; `FlowScore` keys its components by `Component` anyway, so
    the order is for reproducibility rather than semantics.

    Note what this does NOT return: the ORDER_FLOW and OPTIONS_FLOW scorers.
    Those components need feeds bars cannot supply, and with bars-only data
    their 45 points are UNAVAILABLE rather than zero. A caller that built a
    `FlowScore` from these three alone would be publishing a 55-point total
    on a 100-point scale, which is the specific dishonesty
    `strict_component_availability` exists to block; the aggregate owns that
    decision and must see all five components to make it.
    """
    return (
        StructureScorer(config),
        LiquidityScorer(config),
        VolMomentumScorer(config),
    )
