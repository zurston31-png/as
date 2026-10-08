"""Component scorers for the two FEED-DEPENDENT Flow Score components:
`ORDER_FLOW` (25 points on the prior) and `OPTIONS_FLOW` (20 points).

These two are grouped in one module because they share one property that the
other three components do not have: **they are the 45 of 100 points that are
UNAVAILABLE or DEGRADED on bars-only data** (ARCHITECTURE.md section 5,
"Honest consequence"). Everything specific to this file follows from that.

The weights quoted above are the brief's PRIOR, read from
`config.flow_score.weights` at runtime and never hardcoded here. They are
hypotheses for the Phase 8 sweep (section 0.7, section 7); nothing in this
module reads a target or a success criterion, and nothing here was chosen by
looking at a result.

Nothing in this module claims that either component predicts anything. A
scorer reports a measurement and its provenance; whether the measurement has
content is a Phase 8 and Phase 10 question, and no module may answer it.


## What a scorer is, and what it deliberately is not

A scorer maps an ALREADY-COMPUTED `FeatureVector` to a `ComponentScore`. It
reads no market data, takes no `MarketView`, holds no state and has no clock.
The feature layer computed these numbers from a point-in-time view that is
physically unable to return future data (section 4), and
`validation/lookahead.py` has audited each computer under `factory=`. A
scorer that recomputed any part of that would be doing audited work in an
unaudited place.

So this module performs **no transform at all** on the two magnitudes:

* `features/orderflow.py` already composes its five terms into
  `order_flow_magnitude` in [0, 1] with `order_flow_direction` carried
  separately, under the rule in section 7.
* `features/optionsflow.py` already composes its five terms into
  `options_flow_magnitude` in [0, 1] -- capped on the end-of-day path -- with
  `options_flow_direction` carried separately.

Section 7's "explicit, bounded transforms (percentile ranks or `tanh`
squashes -- never unbounded z-scores)" is therefore satisfied upstream, by
construction: every term in both components is a percentile rank, a share or
a `tanh` squash, and the composition is a weighted sum of terms in [-1, 1].
Applying a SECOND transform here would be worse than redundant. Squashing an
already-bounded magnitude would compress it non-linearly and silently move
the scale that every score threshold in `config.setups` is expressed
against; re-ranking it would need history, which a scorer does not have and
must not acquire. This module contains no z-score, no clamp applied to an
unbounded quantity, and no rescaling of a magnitude.

A consequence worth stating for whoever runs Phase 8: because the
composition lives in the feature modules, a sweep over `OrderFlowWeights` or
`OptionsFlowWeights` moves code in `features/`, while a sweep over
`config.flow_score.weights` moves only the `weight` field this module
attaches. Those are two different sweeps over two different layers.

What the scorers DO do is the part that is theirs:

1. decide whether the component was measured at all, and report UNAVAILABLE
   rather than zero when it was not (below);
2. attach the configured weight, so points are `magnitude * weight`;
3. propagate `DataQuality` with the reason, per key rather than per vector;
4. carry the options EOD cap AND its flag into the `ComponentScore`;
5. verify the feature layer's own stated invariants and raise a
   `ScoringError` naming the producing module when one is broken, so a
   renamed or re-meaning'd feature key is a loud failure instead of a quietly
   different score.


## THE distinction this module exists to preserve

`features/orderflow.py` reports the component UNAVAILABLE -- quality MISSING,
every key 0.0 -- when there is no usable tick feed, and section 5 states that
its 25 points are **not redistributed**. Those zeros are placeholders. A
scorer that mapped them to `magnitude=0.0` on an ENABLED `ComponentScore`
would turn

    "never measured: there is no tick feed"

into

    "measured, and the tape is balanced"

and a bars-only Flow Score would then look like a 100-point score that
happened to find no order flow. That is precisely the statement Phase 4's
disablement path exists to prevent, and the reason bars-only data must report
55.0 of 100.0 computable rather than a low score out of 100.

The encoding, which every consumer of this module must read:

| `ComponentScore` state | meaning |
|---|---|
| `enabled=True`, `quality=GOOD` | measured; every term had real inputs |
| `enabled=True`, `quality=DEGRADED` | measured, but a named term was dropped (order flow) or the chain is end-of-day (options flow). The lost weight is NOT redistributed, so `detail["attainable_points"]` is below `weight` |
| `enabled=False`, `quality=MISSING` | **UNAVAILABLE: never measured.** `magnitude` is 0.0 because the contract requires a float in [0, 1] and there is no third value; it is not a measurement and must not be read as one |

`ComponentScore.points` already returns 0.0 when `enabled` is False, and
`FlowScore.available_points` already excludes a disabled component's weight
while `max_points` keeps it -- which is how the gap stays visible. The
`enabled` flag is the load-bearing field, not the magnitude.

Two different situations both produce `enabled=False`, and
`detail["measured"]` / `detail["component_enabled_in_config"]` separate them:
a component ablated in `config.flow_score.enabled_components` (a deliberate
Phase 8 experiment) and a component with no usable feed (a property of the
dataset). `ComponentScoring.reason` says which in words.

This module never redistributes weight, under any configuration. It does not
read `config.flow_score.strict_component_availability` at all:
`strict` decides whether the ENGINE may emit a Flow Score from an incomplete
set of components, which is `AvailabilityReport.scoring_is_valid`'s and the
aggregator's question. A scorer's answer to "was this component measured" is
the same either way, because it is a fact about the data.


## Direction is carried separately, and is never capped or scaled

Section 7, quoted because it is the single most important rule in Phase 6:

    "Direction is handled separately from magnitude: each component emits
     (magnitude in [0,1], direction in {-1,0,+1}). The Flow Score is the
     magnitude aggregate; direction agreement is enforced by the gates. This
     avoids the common bug where a strong bearish component inflates a
     bullish score."

So a strongly BEARISH order-flow reading produces a LARGE magnitude here,
exactly as a strongly bullish one does, and `direction=-1`. This module
contains no code path in which the sign of a direction changes a magnitude,
and none in which a magnitude changes a direction. Refusing the long is the
gates' job (`ConfirmationGate`, `ContradictionGate`, section 3), and
`FlowScore.opposing_points` is the quantity they read.

The options EOD cap is a cap on MAGNITUDE only. An end-of-day chain's
direction is reported uncapped, unscaled and unsuppressed, because a cap
expresses "this measurement carries less weight than an intraday one" and
direction is not a weight. What the cap does to the direction's INFLUENCE is
already correct without touching it: `opposing_points` and `net_direction`
weight a direction by `points`, which the cap has already reduced.

The adversarial case the brief names -- one component strongly bearish, the
rest strongly bullish, high aggregate magnitude, and no trade taken in
either direction -- spans the aggregator and the gates and cannot be
asserted from this file alone. The half that IS this file's: a strongly
bearish component emits a large magnitude and `direction=-1`, with no
attenuation and no sign leaking into the magnitude.


## Exactly which feature keys each scorer reads

Declared as code (`OrderFlowScorer.reads`, `OptionsFlowScorer.reads`) and
restated in each scorer's own docstring, so that a change to a feature module
is traceable to the scorers it breaks. Every read key is checked for presence
on every call; an absent key raises `ScoringError` naming the module that
owes it.

Two classes of feature key are deliberately NOT read:

* **Natural-unit diagnostics.** `signed_delta_contracts`,
  `cvd_slope_normalized`, `options_net_premium`, `options_skew_25d`,
  `options_total_volume`. Both feature modules state that a weighted sum must
  not read these, because one extreme observation in an unbounded feature
  dominates the sum. They are not read and they are not passed through into
  `detail` either, since `detail` is where a later consumer would find them.
* **Feed age.** `options_snapshot_age_seconds`. Staleness is graded by
  `data/quality.py`'s `DataQualityGate` at the top of the pipeline (section
  3), and a scorer that re-judged it would be a second, unsynchronized
  staleness rule.


## `detail` conventions

`ComponentScore.detail` is `dict[str, float]`, and these scorers populate it
with two kinds of entry, distinguished by name:

* Keys spelled EXACTLY as the feature key they came from
  (`options_eod_capped`, `aggression_ratio`, ...) are verbatim pass-throughs,
  greppable straight back to the producing module.
* Keys that name a derivation (`measured`, `attainable_points`,
  `points_withheld_by_cap`, ...) are computed here, from the configured
  weight and the pass-throughs, and are documented at `_DETAIL_*` below.


## Determinism

Pure functions over a frozen input. No clock, no RNG, no mutable state, no
I/O, no logging, no float accumulation order that depends on dict iteration.
`__init__` accepts CONFIGURATION ONLY -- a `FlowModelConfig`, never a vector,
view, series or dataset -- so two independently constructed scorers on one
config agree, and two calls on the same `FeatureVector` agree bit-for-bit.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pydantic import Field

from flow_model.config.schema import FlowModelConfig
from flow_model.core.contracts import ComponentScore, FeatureVector
from flow_model.core.enums import Component, DataQuality
from flow_model.core.model import FrozenModel
from flow_model.data.quality import ComponentAvailability

__all__ = [
    "ScoringError",
    "ComponentScoring",
    "ComponentScorer",
    "OrderFlowScorer",
    "OptionsFlowScorer",
    "feed_dependent_scorers",
    "as_components",
    "ORDER_FLOW_READS",
    "OPTIONS_FLOW_READS",
]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Width of the band inside which a bound violation is treated as a
#: representation error and clamped rather than raised. One part in 1e-9 is
#: the scale at which `0.9999999999999999` and `1.0` differ after a weighted
#: sum of five terms; anything further outside [0, 1] or {-1, 0, +1} is a
#: broken contract in the producing module and raises instead. The clamp is
#: floating-point hygiene, NOT a transform: it cannot move a value by an
#: amount any threshold in this system can resolve.
BOUND_TOLERANCE = 1e-9

#: Prefix `features/orderflow.py` and `features/optionsflow.py` put on every
#: note they emit (`f"{self.name}: {reason}"`). `FeatureVector.notes` is
#: merged across the whole bundle, so a scorer filters by prefix to quote the
#: notes that belong to ITS component rather than another computer's. These
#: are the two computers' `name` attributes; `tests/unit/test_scoring_flow.py`
#: asserts the linkage so a rename there does not silently mute the reasons
#: here. The two spellings differ upstream and are reproduced as they are.
ORDER_FLOW_NOTE_PREFIX = "orderflow:"
OPTIONS_FLOW_NOTE_PREFIX = "options_flow:"

#: `detail` entries computed here rather than passed through. Named so the
#: docstring above and the code cannot drift apart.
_DETAIL_MEASURED = "measured"
_DETAIL_CONFIG_ENABLED = "component_enabled_in_config"
_DETAIL_ATTAINABLE_POINTS = "attainable_points"
_DETAIL_POINTS_WITHHELD = "points_withheld_by_cap"
_DETAIL_CAP_FRACTION = "eod_cap_fraction"

#: Flag encodings, matching the feature modules' own 1.0 / 0.0 convention.
_FLAG_SET = 1.0
_FLAG_CLEAR = 0.0


class ScoringError(RuntimeError):
    """A feature module broke a contract this scorer depends on.

    Raised, never swallowed and never worked around with a fallback value.
    Every condition that raises is one where the honest alternatives are a
    crash or a number whose meaning the scorer cannot establish -- an absent
    key it declared it reads, a magnitude outside [0, 1], a direction outside
    {-1, 0, +1}, or an options cap flag that contradicts the magnitudes it is
    supposed to describe. Degraded and missing DATA are not errors: those
    have their own reported paths and are the normal case on this project's
    only real-data shape.
    """


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------


class ComponentScoring(FrozenModel):
    """A `ComponentScore` plus the words for why it looks the way it does.

    `ComponentScore` is a frozen Phase 1 contract with `detail: dict[str,
    float]`, so a textual reason has nowhere to live inside it, and widening
    an already-specified contract to carry one is not this phase's business.
    The reason matters: "UNAVAILABLE because there is no tick feed" and
    "UNAVAILABLE because the component is ablated for this experiment" are
    the same `ComponentScore` and different facts, and section 12's rejection
    analysis needs the difference in words.

    `score` is the contract object: pass it straight into
    `FlowScore(components=...)`, or use `as_components()` for a set of them.
    """

    score: ComponentScore
    reason: str = Field(
        default="",
        description=(
            "Why the component is UNAVAILABLE or DEGRADED, in words. Empty "
            "exactly when the component was measured with every term "
            "available -- i.e. when quality is GOOD."
        ),
    )
    notes: tuple[str, ...] = Field(
        default=(),
        description=(
            "The producing feature computer's own notes for this component, "
            "verbatim, filtered out of the bundle-merged note tuple."
        ),
    )

    # --- pass-throughs, so a caller need not reach through `.score` -----

    @property
    def component(self) -> Component:
        return self.score.component

    @property
    def magnitude(self) -> float:
        """Bounded magnitude in [0, 1]. **Meaningless unless `measured`.**"""
        return self.score.magnitude

    @property
    def direction(self) -> int:
        return self.score.direction

    @property
    def weight(self) -> float:
        return self.score.weight

    @property
    def quality(self) -> DataQuality:
        return self.score.quality

    @property
    def points(self) -> float:
        return self.score.points

    @property
    def measured(self) -> bool:
        """False for UNAVAILABLE: the component was never measured.

        The question a consumer should ask instead of reading `magnitude`.
        """
        return self.score.enabled


def as_components(
    results: tuple["ComponentScoring", ...] | list["ComponentScoring"],
) -> dict[Component, ComponentScore]:
    """`{component: ComponentScore}`, ready for `FlowScore(components=...)`.

    Raises on a duplicate component rather than letting one scoring silently
    replace another in the dict.
    """
    out: dict[Component, ComponentScore] = {}
    for result in results:
        if result.component in out:
            raise ScoringError(
                f"two scorings for {result.component.value}; a Flow Score holds "
                "one score per component and the second would overwrite the first"
            )
        out[result.component] = result.score
    return out


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


class ComponentScorer(ABC):
    """Shared shape: configuration in, one `ComponentScoring` out.

    The constructor takes a `FlowModelConfig` and nothing else. The ban is
    the same one `features/base.py` documents for computers and
    `tests/unit/test_feature_contracts.py` enforces package-wide: a
    full-sample statistic captured at construction is the one leak the
    lookahead audit cannot see through when it is handed an instance. A
    scorer reads no market data at all, so there is nothing for it to
    capture, and the ban is kept anyway because it is cheaper to keep than to
    re-establish.
    """

    #: Which of the five components this scorer produces.
    component: Component = Component.ORDER_FLOW

    #: Exactly the feature keys `score` reads. Declared, checked on every
    #: call, and restated in the subclass docstring.
    reads: tuple[str, ...] = ()

    #: Prefix of the producing computer's notes in the merged note tuple.
    note_prefix: str = ""

    def __init__(self, config: FlowModelConfig) -> None:
        if not isinstance(config, FlowModelConfig):
            raise ScoringError(
                f"{type(self).__name__} takes a FlowModelConfig and nothing else; "
                f"got {type(config).__name__}. A scorer that accepted market data "
                "could carry a full-sample statistic past the lookahead audit."
            )
        self.config = config

    # --- the configured weight -----------------------------------------

    @property
    def weight(self) -> float:
        """This component's weight, from `config.flow_score.weights`.

        Read, never hardcoded. `FlowScoreConfig._check` guarantees an entry
        for every `Component` and that the five sum to `total_points`, so
        there is no default to fall back on here and a `KeyError` would mean
        the config validator was bypassed.
        """
        return float(self.config.flow_score.weights[self.component])

    @property
    def enabled_in_config(self) -> bool:
        """Whether configuration ablates this component (Phase 8 experiment)."""
        return bool(self.config.flow_score.enabled_components.get(self.component, True))

    @abstractmethod
    def score(
        self,
        features: FeatureVector,
        availability: ComponentAvailability | None = None,
    ) -> ComponentScoring:
        """Score this component at `features.ts`.

        `availability`, when supplied, is `data/quality.py`'s DATASET-level
        verdict for this component (`QualityGrader.availability(...)
        .components[component]`). It is consumed, never re-derived: a
        `computable=False` verdict makes the result UNAVAILABLE whatever the
        vector says, and a `computable=True` verdict never overrides a vector
        that reports its own keys MISSING. Availability is an AND of the two,
        which is the only combination in which neither source can be talked
        out of a refusal.
        """

    def score_component(
        self,
        features: FeatureVector,
        availability: ComponentAvailability | None = None,
    ) -> ComponentScore:
        """`score(...).score`: the bare contract object, for a uniform caller.

        An adapter, not a second implementation. `signals/scoring_bars.py`
        (the three bars-only components, owned separately) returns a
        `ComponentScore` straight out of its own `score`, while these two
        return a `ComponentScoring` because UNAVAILABLE and DEGRADED need a
        reason in words that `ComponentScore` has nowhere to put. An
        aggregator that wants all five components in one shape can call this
        and lose only the reason; one that wants the reason calls `score`.
        Reconciling the two signatures is the aggregator's decision to make
        in one place, and is flagged rather than pre-empted here.
        """
        return self.score(features, availability).score

    # --- shared construction -------------------------------------------

    def _check_keys(self, features: FeatureVector, owner: str) -> None:
        """Every declared read key must be present.

        Presence, not quality: the feature modules emit their full key set
        even on the not-ready path (`FeatureComputer._not_ready` writes a 0.0
        placeholder for each), and `FeatureBundle.compute` does the same for a
        computer whose feed is absent. So an ABSENT key is never a data
        condition -- it means the key was renamed, dropped, or the vector did
        not come from the expected bundle. Each of those changes what a score
        means, so it raises here rather than scoring whatever is left.
        """
        absent = tuple(key for key in self.reads if key not in features.values)
        if absent:
            raise ScoringError(
                f"{type(self).__name__} reads {len(self.reads)} feature keys and "
                f"{list(absent)} are not in the vector at "
                f"{features.ts.isoformat()}. These keys are owned by {owner}; if "
                "one was renamed there, this scorer's declared read set must be "
                "updated with it, because a score computed from the remainder "
                "would be on a different scale than the keys it kept."
            )

    def _notes_for(self, features: FeatureVector) -> tuple[str, ...]:
        """This component's own notes out of the bundle-merged tuple."""
        if not self.note_prefix:
            return ()
        return tuple(n for n in features.notes if n.startswith(self.note_prefix))

    def _unavailable(
        self,
        reason: str,
        *,
        notes: tuple[str, ...] = (),
        detail: dict[str, float] | None = None,
    ) -> ComponentScoring:
        """The UNAVAILABLE result: never measured.

        `magnitude=0.0` and `direction=0` because `ComponentScore` requires a
        float in [0, 1] and an int in {-1, 0, +1} and there is no third value
        in either domain. They are NOT measurements and the fields that say
        so are `enabled=False`, `quality=MISSING` and
        `detail["measured"]=0.0`. `ComponentScore.points` is 0.0 for a
        disabled component, so this casts no directional vote and contributes
        no points; `weight` is kept at its CONFIGURED value, which is what
        makes `FlowScore.available_points` fall below `max_points` instead of
        the gap disappearing.

        There is deliberately no variant of this method that substitutes an
        estimate, flags it DEGRADED and carries on. For order flow section 5
        permits no proxy at all; for options flow it permits a capped
        end-of-day path, which `features/optionsflow.py` already provides and
        which arrives here as DEGRADED, not as something this method builds.
        """
        full = {
            _DETAIL_MEASURED: _FLAG_CLEAR,
            _DETAIL_CONFIG_ENABLED: (
                _FLAG_SET if self.enabled_in_config else _FLAG_CLEAR
            ),
            _DETAIL_ATTAINABLE_POINTS: 0.0,
        }
        if detail:
            full.update(detail)
        return ComponentScoring(
            score=ComponentScore(
                component=self.component,
                magnitude=0.0,
                direction=0,
                weight=self.weight,
                quality=DataQuality.MISSING,
                enabled=False,
                features_used=self.reads,
                detail=full,
            ),
            reason=reason,
            notes=notes,
        )

    # --- shared validation ---------------------------------------------

    def _bounded_magnitude(self, features: FeatureVector, key: str, owner: str) -> float:
        """A magnitude the producing module promised is in [0, 1]."""
        value = features.require(key)
        if value < -BOUND_TOLERANCE or value > 1.0 + BOUND_TOLERANCE:
            raise ScoringError(
                f"{owner} emitted {key}={value!r}, outside [0, 1]. {owner} "
                "documents this key as bounded by construction, so this is a "
                "broken invariant there and not a value to clamp: a magnitude "
                "above 1.0 would let one component exceed its own weight, and a "
                "negative one would subtract points from the Flow Score."
            )
        return min(1.0, max(0.0, value))

    def _direction(self, features: FeatureVector, key: str, owner: str) -> int:
        """A direction the producing module promised is in {-1, 0, +1}."""
        value = features.require(key)
        nearest = round(value)
        if abs(value - nearest) > BOUND_TOLERANCE or nearest not in (-1, 0, 1):
            raise ScoringError(
                f"{owner} emitted {key}={value!r}, which is not one of "
                "{-1.0, 0.0, +1.0}. Section 7 defines direction as a three-valued "
                "vote, and a fractional direction would be a magnitude wearing a "
                "direction's name -- exactly the conflation section 7 separates."
            )
        return int(nearest)

    def _flag(self, features: FeatureVector, key: str, owner: str) -> bool:
        """A 0.0 / 1.0 flag the producing module promised is categorical."""
        value = features.require(key)
        if not (
            math.isclose(value, _FLAG_SET, abs_tol=BOUND_TOLERANCE)
            or math.isclose(value, _FLAG_CLEAR, abs_tol=BOUND_TOLERANCE)
        ):
            raise ScoringError(
                f"{owner} emitted {key}={value!r}, which is neither 0.0 nor 1.0. "
                "This key is a categorical fact encoded as a float, and a value "
                "between the two has no meaning a scorer may guess at."
            )
        return math.isclose(value, _FLAG_SET, abs_tol=BOUND_TOLERANCE)


# ---------------------------------------------------------------------------
# ORDER_FLOW
# ---------------------------------------------------------------------------

#: Exactly the keys `OrderFlowScorer` reads, all owned by
#: `features/orderflow.py`. Kept in one place so the docstring, the contract
#: check and `ComponentScore.features_used` cannot drift apart.
ORDER_FLOW_READS: tuple[str, ...] = (
    # availability and the non-redistribution contract
    "order_flow_available",
    "order_flow_available_weight",
    # the composed component, magnitude and direction carried separately
    "order_flow_magnitude",
    "order_flow_direction",
    # bounded per-term diagnostics, passed into `detail` verbatim
    "signed_delta",
    "cvd_slope",
    "aggression_ratio",
    "absorption",
    "absorption_direction",
    "avg_trade_size_percentile",
    "max_trade_size_percentile",
    "large_trade_event",
    "classification_coverage",
)

#: The subset copied into `detail` under its own feature-key name.
_ORDER_FLOW_PASSTHROUGH: tuple[str, ...] = (
    "order_flow_available_weight",
    "signed_delta",
    "cvd_slope",
    "aggression_ratio",
    "absorption",
    "absorption_direction",
    "avg_trade_size_percentile",
    "max_trade_size_percentile",
    "large_trade_event",
    "classification_coverage",
)

_ORDER_FLOW_OWNER = "features/orderflow.py"


class OrderFlowScorer(ComponentScorer):
    """ORDER_FLOW: the component with no degraded path and no proxy.

    ## Feature keys read (all owned by `features/orderflow.py`)

    Availability and the non-redistribution contract:

    * `order_flow_available` -- 1.0 usable, 0.0 UNAVAILABLE. The producing
      module's own verdict, and the first thing this scorer consults.
    * `order_flow_available_weight` -- in [0, 1]: the share of term weight
      that had real inputs. Below 1.0, dropped terms have lowered the
      ATTAINABLE magnitude and their weight was not reallocated, so
      `detail["attainable_points"]` falls below `weight`.

    The composed component:

    * `order_flow_magnitude` -- in [0, 1].
    * `order_flow_direction` -- -1.0 / 0.0 / +1.0, carried separately.

    Bounded per-term diagnostics, copied verbatim into `detail`:

    * `signed_delta`, `cvd_slope` (both in [-1, 1]),
      `aggression_ratio`, `absorption`, `avg_trade_size_percentile`,
      `max_trade_size_percentile`, `classification_coverage` (all in [0, 1]),
      `absorption_direction` and `large_trade_event` (flags).

    NOT read: `signed_delta_contracts` and `cvd_slope_normalized`. Both are
    unbounded natural-unit diagnostics, and `features/orderflow.py` states
    that a weighted sum must not read them. They are not passed into `detail`
    either, so nothing downstream can find them there and sum them.

    ## Why UNAVAILABLE, not zero

    `features/orderflow.py` disables the component -- quality MISSING, every
    key 0.0, `order_flow_available` 0.0 -- for any of eight named conditions:
    no tick feed, too little history, a tick feed that will not say how it
    classified the aggressor side or that declares a rejected method,
    measured classification coverage below the floor, dropped terms carrying
    too much of the weight, an unreadable or arithmetically unusable window,
    and a tick/bar right-edge misalignment.

    In every one of those the honest statement is "never measured", and this
    scorer propagates it as `enabled=False, quality=MISSING`. The alternative
    -- `magnitude=0.0` on an enabled score -- reads downstream as "the tape
    is balanced", which is a measurement nobody made. On bars-only data that
    single substitution is the difference between reporting 55.0 of 100.0
    points computable and reporting a 100-point Flow Score that found no
    order flow.

    The 25 points stay on the score as `weight` with `enabled=False`, so
    `FlowScore.available_points` drops by exactly 25 and `max_points` does
    not move. Nothing here redistributes them, and there is no configuration
    under which this scorer will.

    ## DEGRADED

    `order_flow_magnitude` is graded DEGRADED by the producing module when at
    least one of its five terms was dropped for want of inputs. The component
    WAS measured -- the surviving terms are real -- but its attainable
    magnitude is `order_flow_available_weight`, not 1.0, and the missing
    weight was not reallocated. That survives into `quality` with the
    producing module's own notes as the reason.

    A caveat carried from `features/orderflow.py` rather than restated as if
    it were fine: with the default constants that module MEASURED its
    `absorption` term to be identically zero on five-minute data, so 0.15 of
    the component's weight is unreachable while the term still reports as
    available. `order_flow_available_weight` is 1.0 in that state and
    `detail["attainable_points"]` therefore overstates the reachable ceiling
    by that share. The constant is a Phase 8 hypothesis and the fix belongs
    in a config validator; moving it here would be choosing a value by
    looking at a result.
    """

    component = Component.ORDER_FLOW
    reads = ORDER_FLOW_READS
    note_prefix = ORDER_FLOW_NOTE_PREFIX

    def score(
        self,
        features: FeatureVector,
        availability: ComponentAvailability | None = None,
    ) -> ComponentScoring:
        notes = self._notes_for(features)

        if not self.enabled_in_config:
            return self._unavailable(
                "ORDER FLOW UNAVAILABLE: disabled in configuration "
                "(flow_score.enabled_components). An ablated component is not a "
                "measured zero, so its weight is reported unavailable rather "
                "than spread over the components that remain.",
                notes=notes,
            )

        refusal = _availability_refusal(availability, self.component)
        if refusal is not None:
            return self._unavailable(refusal, notes=notes)

        self._check_keys(features, _ORDER_FLOW_OWNER)

        if not self._flag(features, "order_flow_available", _ORDER_FLOW_OWNER):
            return self._unavailable(
                "ORDER FLOW UNAVAILABLE: features/orderflow.py reported "
                "order_flow_available=0.0, so the component was never measured. "
                "ARCHITECTURE.md section 5 accepts no proxy for order flow -- no "
                "bar-volume delta, tick rule on bars, close-location or "
                "quote-imbalance estimate is substituted -- and the component's "
                "weight is reported unavailable rather than redistributed.",
                notes=notes,
            )

        key_quality = features.quality_of("order_flow_magnitude")
        if key_quality is DataQuality.MISSING or key_quality is DataQuality.STALE:
            return self._unavailable(
                f"ORDER FLOW UNAVAILABLE: order_flow_magnitude is graded "
                f"{key_quality.value}, so the value present on the key is a "
                "placeholder rather than a measurement. Reported unavailable, "
                "not as a zero magnitude.",
                notes=notes,
            )

        magnitude = self._bounded_magnitude(
            features, "order_flow_magnitude", _ORDER_FLOW_OWNER
        )
        direction = self._direction(
            features, "order_flow_direction", _ORDER_FLOW_OWNER
        )
        available_weight = self._bounded_magnitude(
            features, "order_flow_available_weight", _ORDER_FLOW_OWNER
        )

        detail: dict[str, float] = {
            _DETAIL_MEASURED: _FLAG_SET,
            _DETAIL_CONFIG_ENABLED: _FLAG_SET,
            _DETAIL_ATTAINABLE_POINTS: round(self.weight * available_weight, 10),
        }
        for key in _ORDER_FLOW_PASSTHROUGH:
            detail[key] = float(features.require(key))

        reason = ""
        if key_quality is DataQuality.DEGRADED:
            reason = (
                "ORDER FLOW DEGRADED: at least one of the five order-flow terms "
                "was dropped for want of inputs. The surviving terms are real "
                f"measurements, the attainable magnitude is {available_weight:.4f} "
                "rather than 1.0, and the dropped weight was NOT redistributed."
            )

        return ComponentScoring(
            score=ComponentScore(
                component=self.component,
                magnitude=magnitude,
                direction=direction,
                weight=self.weight,
                quality=key_quality,
                enabled=True,
                features_used=self.reads,
                detail=detail,
            ),
            reason=reason,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# OPTIONS_FLOW
# ---------------------------------------------------------------------------

#: Exactly the keys `OptionsFlowScorer` reads, all owned by
#: `features/optionsflow.py`.
OPTIONS_FLOW_READS: tuple[str, ...] = (
    # the composed component, magnitude and direction carried separately
    "options_flow_magnitude",
    "options_flow_uncapped_magnitude",
    "options_flow_direction",
    # the degradation contract: the cap and its flag
    "options_eod_capped",
    "options_flow_timing_available",
    "options_available_weight",
    "options_terms_available",
    # bounded combination and per-term diagnostics
    "options_sign_agreement",
    "options_imbalance_consensus",
    "options_net_premium_imbalance",
    "options_delta_volume_imbalance",
    "options_oi_change_imbalance",
    "options_skew_25d_pressure",
    "options_gamma_exposure_pressure",
)

#: The subset copied into `detail` under its own feature-key name.
_OPTIONS_FLOW_PASSTHROUGH: tuple[str, ...] = (
    "options_flow_uncapped_magnitude",
    "options_eod_capped",
    "options_flow_timing_available",
    "options_available_weight",
    "options_terms_available",
    "options_sign_agreement",
    "options_imbalance_consensus",
    "options_net_premium_imbalance",
    "options_delta_volume_imbalance",
    "options_oi_change_imbalance",
    "options_skew_25d_pressure",
    "options_gamma_exposure_pressure",
)

_OPTIONS_FLOW_OWNER = "features/optionsflow.py"


class OptionsFlowScorer(ComponentScorer):
    """OPTIONS_FLOW: the component with a capped, flagged degraded path.

    ## Feature keys read (all owned by `features/optionsflow.py`)

    The composed component:

    * `options_flow_magnitude` -- in [0, 1], ALREADY CAPPED on the
      end-of-day path.
    * `options_flow_uncapped_magnitude` -- in [0, 1]: what the magnitude
      would have been without the cap.
    * `options_flow_direction` -- -1.0 / 0.0 / +1.0, carried separately and
      never capped.

    The degradation contract:

    * `options_eod_capped` -- 1.0 exactly when the cap CHANGED the magnitude
      on this bar.
    * `options_flow_timing_available` -- 1.0 an intraday chain (the terms
      describe flow at `features.ts`), 0.0 an end-of-day chain (positioning
      without flow timing).
    * `options_available_weight` in [0, 1] and `options_terms_available` in
      0..5 -- how much of the five-term weight had usable inputs.

    Bounded diagnostics copied verbatim into `detail`:
    `options_sign_agreement`, `options_imbalance_consensus`, and the five
    bounded terms `options_net_premium_imbalance`,
    `options_delta_volume_imbalance`, `options_oi_change_imbalance`,
    `options_skew_25d_pressure`, `options_gamma_exposure_pressure`.

    NOT read: `options_net_premium`, `options_skew_25d`,
    `options_total_volume` (unbounded natural-unit diagnostics a weighted sum
    must not read) and `options_snapshot_age_seconds` (staleness is graded
    once, by `data/quality.py`, at the top of the pipeline).

    ## The cap, and why both halves are carried

    Section 5 requires the EOD-degraded options sub-score to be "capped AND
    flagged". `features/optionsflow.py` applies

        magnitude = min(uncapped, config.options_flow.eod_degraded_cap_fraction)

    whenever `OptionsSnapshot.is_intraday` is False, and raises
    `options_eod_capped` only when the cap actually bound. Verified binding on
    an EOD chain: uncapped 0.514693 -> capped 0.500000.

    This scorer carries the capped magnitude, the uncapped magnitude, the flag
    and the configured cap fraction into `ComponentScore.detail`, so the score
    itself states that it was capped -- a consumer never has to go back to the
    `FeatureVector` to find out. `detail["points_withheld_by_cap"]` is the
    weighted points the cap removed, which is the quantity a Phase 8
    comparison of the two data shapes actually wants.

    It also checks the four invariants the cap implies, because a cap that
    stopped binding while its flag kept firing would be a flag describing a
    restriction that was never applied -- which `OptionsFlowConfig`'s own
    validator calls worse than having neither:

    1. `magnitude <= uncapped` always: a cap can only reduce.
    2. flag set => `magnitude < uncapped`: it bound.
    3. flag clear => `magnitude == uncapped`: it did not.
    4. flag set => timing unavailable: an intraday chain is never capped.

    Plus one cross-check against configuration: flag set => the magnitude
    equals the configured `eod_degraded_cap_fraction`. A mismatch means the
    scorer and the computer were built from different configs, which changes
    what the score means.

    ## Quality, and what is NOT capped

    `features/optionsflow.py` grades `options_flow_magnitude` DEGRADED
    whenever the chain is end-of-day or any term lost its inputs, and GOOD
    only on an intraday chain with every term available. That survives into
    `quality` unchanged. An end-of-day options reading is a real measurement
    of positioning with no flow timing -- DEGRADED, not MISSING -- and
    reporting it unavailable would throw away data the system has.

    UNAVAILABLE here means the producing module took its own not-ready path:
    no options feed, too few snapshots, a chain with a non-finite required
    field, available term weight below the floor, or
    `require_intraday_prints` set against an end-of-day chain. In all of those
    every key is MISSING and this scorer reports `enabled=False`.

    DIRECTION IS NOT CAPPED. The cap says an end-of-day measurement carries
    less weight than an intraday one; it says nothing about which way the
    imbalance leans, and scaling or suppressing the direction would be the
    magnitude/direction conflation section 7 forbids. The cap already reduces
    the direction's influence correctly and without being touched, because
    `FlowScore.net_direction` and `opposing_points` weight each direction by
    `points`, and `points` is the capped magnitude times the weight.
    """

    component = Component.OPTIONS_FLOW
    reads = OPTIONS_FLOW_READS
    note_prefix = OPTIONS_FLOW_NOTE_PREFIX

    @property
    def cap_fraction(self) -> float:
        """`config.options_flow.eod_degraded_cap_fraction`, read not hardcoded.

        A FRACTION of the component's weight rather than absolute points, so
        the cap survives the Phase 8 weight sweep. Its default is a judgement
        call recorded in `OptionsFlowConfig`; this module does not change it
        and does not get to.
        """
        return float(self.config.options_flow.eod_degraded_cap_fraction)

    def score(
        self,
        features: FeatureVector,
        availability: ComponentAvailability | None = None,
    ) -> ComponentScoring:
        notes = self._notes_for(features)

        if not self.enabled_in_config:
            return self._unavailable(
                "OPTIONS FLOW UNAVAILABLE: disabled in configuration "
                "(flow_score.enabled_components). An ablated component is not a "
                "measured zero, so its weight is reported unavailable rather "
                "than spread over the components that remain.",
                notes=notes,
                detail={_DETAIL_CAP_FRACTION: self.cap_fraction},
            )

        refusal = _availability_refusal(availability, self.component)
        if refusal is not None:
            return self._unavailable(
                refusal, notes=notes, detail={_DETAIL_CAP_FRACTION: self.cap_fraction}
            )

        self._check_keys(features, _OPTIONS_FLOW_OWNER)

        key_quality = features.quality_of("options_flow_magnitude")
        if key_quality is DataQuality.MISSING or key_quality is DataQuality.STALE:
            return self._unavailable(
                f"OPTIONS FLOW UNAVAILABLE: options_flow_magnitude is graded "
                f"{key_quality.value}, so features/optionsflow.py took its "
                "not-ready path -- no chain, too few snapshots, a non-finite "
                "required field, available term weight below the floor, or "
                "require_intraday_prints set against an end-of-day chain. "
                "Nothing is synthesized to fill the gap and the weight is not "
                "redistributed.",
                notes=notes,
                detail={_DETAIL_CAP_FRACTION: self.cap_fraction},
            )

        magnitude = self._bounded_magnitude(
            features, "options_flow_magnitude", _OPTIONS_FLOW_OWNER
        )
        uncapped = self._bounded_magnitude(
            features, "options_flow_uncapped_magnitude", _OPTIONS_FLOW_OWNER
        )
        direction = self._direction(
            features, "options_flow_direction", _OPTIONS_FLOW_OWNER
        )
        capped = self._flag(features, "options_eod_capped", _OPTIONS_FLOW_OWNER)
        timing = self._flag(
            features, "options_flow_timing_available", _OPTIONS_FLOW_OWNER
        )
        available_weight = self._bounded_magnitude(
            features, "options_available_weight", _OPTIONS_FLOW_OWNER
        )

        self._check_cap_invariants(
            features=features,
            magnitude=magnitude,
            uncapped=uncapped,
            capped=capped,
            timing=timing,
        )

        # The ceiling the component could actually reach on this bar: the
        # weight that had inputs, further limited by the cap when the chain
        # carries no flow timing. Reported because a magnitude read against a
        # ceiling of 1.0 overstates what was reachable.
        ceiling = available_weight if timing else min(available_weight, self.cap_fraction)

        detail: dict[str, float] = {
            _DETAIL_MEASURED: _FLAG_SET,
            _DETAIL_CONFIG_ENABLED: _FLAG_SET,
            _DETAIL_ATTAINABLE_POINTS: round(self.weight * ceiling, 10),
            _DETAIL_CAP_FRACTION: self.cap_fraction,
            _DETAIL_POINTS_WITHHELD: round(self.weight * (uncapped - magnitude), 10),
        }
        for key in _OPTIONS_FLOW_PASSTHROUGH:
            detail[key] = float(features.require(key))

        reason = ""
        if key_quality is DataQuality.DEGRADED:
            parts = []
            if not timing:
                parts.append(
                    "the chain is end-of-day only, so the terms measure "
                    "positioning without flow timing"
                )
            if available_weight < 1.0 - BOUND_TOLERANCE:
                parts.append(
                    f"terms carrying {1.0 - available_weight:.4f} of the term "
                    "weight had no usable inputs and that weight was NOT "
                    "redistributed"
                )
            if capped:
                parts.append(
                    f"the EOD cap bound: magnitude {magnitude:.6f} from an "
                    f"uncapped {uncapped:.6f}, withholding "
                    f"{self.weight * (uncapped - magnitude):.6f} points"
                )
            elif not timing:
                parts.append(
                    f"the EOD cap applied but did not bind this bar "
                    f"(uncapped {uncapped:.6f} is at or below the cap "
                    f"{self.cap_fraction:.4f})"
                )
            if not parts:  # pragma: no cover - DEGRADED implies one of the above
                parts.append(
                    "features/optionsflow.py graded the magnitude DEGRADED; see "
                    "the notes"
                )
            reason = "OPTIONS FLOW DEGRADED: " + "; ".join(parts) + "."

        return ComponentScoring(
            score=ComponentScore(
                component=self.component,
                magnitude=magnitude,
                direction=direction,
                weight=self.weight,
                quality=key_quality,
                enabled=True,
                features_used=self.reads,
                detail=detail,
            ),
            reason=reason,
            notes=notes,
        )

    # --- the cap's invariants ------------------------------------------

    def _check_cap_invariants(
        self,
        *,
        features: FeatureVector,
        magnitude: float,
        uncapped: float,
        capped: bool,
        timing: bool,
    ) -> None:
        """Verify that the cap and its flag describe each other.

        Section 5 requires the degraded options sub-score to be capped AND
        flagged, and `OptionsFlowConfig._check_cap_can_bind` states why a flag
        that outlives the restriction it describes is worse than having
        neither. These five checks are what make that statement testable at
        runtime instead of only at config-validation time.

        Raises rather than repairing. Every repair available here -- trusting
        the flag, trusting the magnitudes, recomputing the cap -- picks one of
        two contradictory statements about the same bar, and a scorer has no
        grounds for the choice.
        """
        where = f"at {features.ts.isoformat()}"

        if magnitude > uncapped + BOUND_TOLERANCE:
            raise ScoringError(
                f"{_OPTIONS_FLOW_OWNER} emitted options_flow_magnitude "
                f"{magnitude!r} above options_flow_uncapped_magnitude "
                f"{uncapped!r} {where}. A cap can only reduce a magnitude, so "
                "one of the two keys does not mean what its name says."
            )

        if capped:
            if uncapped <= magnitude + BOUND_TOLERANCE:
                raise ScoringError(
                    f"{_OPTIONS_FLOW_OWNER} set options_eod_capped=1.0 {where} "
                    f"while options_flow_magnitude {magnitude!r} equals the "
                    f"uncapped {uncapped!r}. The flag claims the cap changed the "
                    "magnitude and the magnitudes say it did not; a flag "
                    "describing a restriction that was never applied is the "
                    "failure OptionsFlowConfig._check_cap_can_bind refuses."
                )
            if timing:
                raise ScoringError(
                    f"{_OPTIONS_FLOW_OWNER} set options_eod_capped=1.0 {where} "
                    "with options_flow_timing_available=1.0. The EOD cap exists "
                    "because an end-of-day chain carries no flow timing, so an "
                    "intraday chain is never capped and these two flags cannot "
                    "both be set."
                )
            if abs(magnitude - self.cap_fraction) > BOUND_TOLERANCE:
                raise ScoringError(
                    f"{_OPTIONS_FLOW_OWNER} reported a capped magnitude "
                    f"{magnitude!r} {where} that is not this scorer's configured "
                    f"eod_degraded_cap_fraction {self.cap_fraction!r}. The cap is "
                    "applied as min(uncapped, cap) and the flag is raised only "
                    "when it bound, so a bound cap leaves the magnitude AT the "
                    "fraction exactly -- above it and below it are both "
                    "mismatches. The two were built from different "
                    "configurations, and a cap read from one config cannot "
                    "describe a magnitude produced under another."
                )
        elif magnitude < uncapped - BOUND_TOLERANCE:
            raise ScoringError(
                f"{_OPTIONS_FLOW_OWNER} emitted options_flow_magnitude "
                f"{magnitude!r} below the uncapped {uncapped!r} {where} with "
                "options_eod_capped=0.0. Something reduced the magnitude and the "
                "flag does not say what; an unflagged reduction is exactly the "
                "silent degradation section 5 requires to be visible."
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _availability_refusal(
    availability: ComponentAvailability | None, component: Component
) -> str | None:
    """The dataset-level refusal to propagate, or None.

    `data/quality.py`'s `availability()` already decides which components are
    computable from a dataset, and this consumes that decision rather than
    re-deriving it from feeds, requirements and intraday flags a second time.
    Only a refusal is propagated: a `computable=True` verdict is a statement
    about the DATASET and never overrides a `FeatureVector` whose own keys are
    MISSING at this instant, because `availability()` is deliberately
    forward-looking (it reads the whole series) while the vector is
    point-in-time.
    """
    if availability is None:
        return None
    if availability.component is not component:
        raise ScoringError(
            f"availability report is for {availability.component.value} but this "
            f"scorer produces {component.value}; scoring one component against "
            "another's feed verdict would report the wrong gap."
        )
    if availability.computable:
        return None
    note = availability.note.strip() or "the dataset does not support this component"
    missing = ", ".join(f.value for f in availability.missing_required_feeds)
    detail = f" Missing required feed(s): {missing}." if missing else ""
    return (
        f"{component.value.upper().replace('_', ' ')} UNAVAILABLE: "
        f"data/quality.py reports the component is not computable from this "
        f"dataset -- {note}.{detail} Reported unavailable, not as a zero "
        "magnitude, and its weight is not redistributed."
    )


def feed_dependent_scorers(
    config: FlowModelConfig,
) -> tuple[OrderFlowScorer, OptionsFlowScorer]:
    """The two feed-dependent scorers, in `Component` declaration order.

    Together they own the points that are UNAVAILABLE or DEGRADED on
    bars-only data. Grouped in one constructor so that a caller assembling
    the pipeline cannot pick up one and miss the other.
    """
    return (OrderFlowScorer(config), OptionsFlowScorer(config))
