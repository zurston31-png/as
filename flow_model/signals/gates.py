"""The gate sequence, in section 3's order.

ARCHITECTURE.md section 3 fixes the pipeline, and the researcher's brief
restates it as:

    market data -> regime -> liquidity -> structure -> options flow ->
    order flow -> momentum -> Flow Score -> entry filter ->
    risk calculation -> trade

`GateStage` is that order, declared once, and `Signal.wait_reason` on a
rejected bar is always the FIRST stage that failed. Order is load-bearing for
the same reason it is in `risk/limits.py`: section 12 counts which gate
fired, so a bar blocked by several must report the most fundamental one, and
the count is only meaningful if the sequence is identical on every run.

## Where this differs from section 3's code block, and why

Section 3's code block puts `FlowScore.combine` and its
`score < setup.min_score` check BEFORE `ConfirmationGate` and
`ContradictionGate`; the prose order above puts options flow and order flow
first. The two cannot both be implemented, so the prose order is implemented
and the difference is recorded here rather than resolved silently:

* The directional gates need the component DIRECTIONS, which exist as soon
  as the five components are scored. They do not need the aggregate or its
  threshold. So nothing is computed out of order by running them first.
* Section 3's own stated rationale is that "cheap/structural rejections run
  before expensive scoring". A direction that disagrees with the trade is
  more fundamental than a magnitude that falls short of a threshold, so
  running the disagreement first is the behaviour that rationale asks for.
* No `WaitReason` changes. The same enum members are emitted by the same
  comparisons; only which of two already-specified reasons wins on a bar
  that fails both is affected.

Two further deviations, both deliberate and both reported:

* **`SESSION` is a stage that section 3 does not list.** `WaitReason`
  carries `SESSION_CLOSED`, `SessionFilterConfig` exists, and
  `TradingCalendar.is_tradable` already implements it. Left out, the config
  block and the enum member would be unreachable. It runs FIRST, because a
  bar outside the tradable session is not a data-quality problem -- there is
  no market -- and it is skipped (passing, with a reason saying so) when no
  calendar is supplied.
* **The pre-selection gates ask the weaker question.** `REGIME`,
  `FLOW_SCORE` and `MOMENTUM` are each written in section 3 in terms of "the
  setup", which setup selection has not yet determined. Each therefore tests
  the UNION over enabled setups -- could any setup accept this bar -- using
  `signals/setups.py`'s `tradable_regimes`, `min_flow_score_required` and
  `vol_percentile_band`, and the selected setup's own thresholds are applied
  in `signals/setups.py`. A bar that clears the union and fails the
  determined setup is declined there, with the reason section 3 names for
  that comparison.

## What a gate may not do

A gate reads reports and scores. It never touches market data: the only
market-data type in this module's imports is `MarketView`, and the only
thing read off it is bar counts and feed presence, for the warmup
assessment. There is no accessor on a `MarketView` that returns future data
(section 4), so the point-in-time guarantee is inherited rather than
re-argued.

No gate reads `config.research_targets` or the pre-registered hypotheses,
and no gate claims anything about profitability.

## Direction agreement, which is what section 7 asks the gates to enforce

Section 7: "Direction is handled separately from magnitude ... The Flow
Score is the magnitude aggregate; direction agreement is enforced by the
gates. This avoids the common bug where a strong bearish component inflates
a bullish score."

Two things make that true here:

1. **The candidate side is the STRUCTURE's direction and nothing else.**
   `StructureGate` reads `level_direction` -- 14.6's "+1 at support, -1 at
   resistance" -- and the side it returns is the only side the rest of the
   pipeline ever considers. `FlowScore.net_direction()` is never consulted
   to pick a side, anywhere. That is the structural defence against "quietly
   taking the majority direction": there is no code path in which a majority
   of components can outvote the structure, because no code path asks them.
2. **Opposition blocks the only candidate.** `OrderFlowGate` refuses a side
   its measured order flow opposes, `OptionsFlowGate` refuses one its
   measured options flow opposes by more than
   `options_contradiction_points`, and `EntryFilterGate` refuses one whose
   total opposing weight exceeds `max_opposing_points`. A bar where one
   component is strongly bearish and the rest strongly bullish therefore
   produces a HIGH magnitude aggregate and NO trade: the long is blocked by
   the opposition, and the short was never a candidate.

An UNAVAILABLE component never counts as opposition. Its direction is 0
because no measurement was made, and treating that as disagreement would
report "order flow disagreed" about a tape nobody observed. A setup that
cannot proceed without that confirmation says so through
`require_orderflow_confirmation` in `signals/setups.py`, with
`WaitReason.COMPONENT_DISABLED`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from flow_model.config.schema import DataConfig, FlowModelConfig
from flow_model.core.contracts import ComponentScore, FeatureVector, FlowScore, RegimeState
from flow_model.core.enums import (
    Component,
    DataQuality,
    Feed,
    Regime,
    Side,
    StrEnum,
    WaitReason,
)
from flow_model.core.instruments import InstrumentSpec
from flow_model.data.base import SessionCalendarProtocol
from flow_model.data.market_view import MarketView
from flow_model.data.quality import QualityReport
from flow_model.features.base import FeatureComputer
from flow_model.features.levels import (
    DIRECTION_NONE,
    DIRECTION_RESISTANCE,
    DIRECTION_SUPPORT,
    GATE_CLEANLINESS,
    GATE_NO_TARGET,
    GATE_NO_ZONE,
    GATE_PASSED,
    GATE_PROXIMITY,
    GATE_REJECTION,
    GATE_REWARD_RISK,
    GATE_SIGNIFICANCE,
)
from flow_model.signals.flow_score import FlowScoreOutcome
from flow_model.signals.setups import (
    min_flow_score_required,
    tradable_regimes,
    vol_percentile_band,
)

__all__ = [
    "ComputerReadiness",
    "DataQualityGate",
    "EntryFilterGate",
    "FlowScoreGate",
    "GateResult",
    "GateStage",
    "Gate",
    "LIQUIDITY_GATE_MIN_SCORE",
    "LiquidityGate",
    "MomentumGate",
    "OPPOSITION_WAIT_REASON",
    "OptionsFlowGate",
    "OrderFlowGate",
    "RegimeGate",
    "SessionGate",
    "StructureGate",
    "STRUCTURE_WAIT_REASONS",
    "WAIT_REASONS_BY_STAGE",
    "WarmupGate",
    "WarmupReport",
    "assess_warmup",
    "pipeline_gates",
]


class GateStage(StrEnum):
    """The pipeline stages, in the order they run.

    This tuple IS the order. `signals/engine.py` records a `GateResult` per
    stage it reaches, in sequence, so the implemented order is checkable
    against this enum rather than asserted in prose.
    """

    SESSION = "session"
    MARKET_DATA = "market_data"
    WARMUP = "warmup"
    REGIME = "regime"
    LIQUIDITY = "liquidity"
    STRUCTURE = "structure"
    OPTIONS_FLOW = "options_flow"
    ORDER_FLOW = "order_flow"
    MOMENTUM = "momentum"
    FLOW_SCORE = "flow_score"
    ENTRY_FILTER = "entry_filter"
    SETUP = "setup"
    RISK = "risk"


#: Every `WaitReason` each stage can emit. Declared so that a test can assert
#: the union is a subset of the existing enum -- no stage invents a reason --
#: and so that section 12's rejection analysis can label its buckets by stage
#: without re-deriving the mapping from the code.
WAIT_REASONS_BY_STAGE: Mapping[GateStage, tuple[WaitReason, ...]] = {
    GateStage.SESSION: (WaitReason.SESSION_CLOSED,),
    GateStage.MARKET_DATA: (WaitReason.DATA_QUALITY,),
    GateStage.WARMUP: (WaitReason.WARMUP,),
    GateStage.REGIME: (WaitReason.REGIME_BLOCKED,),
    GateStage.LIQUIDITY: (WaitReason.LIQUIDITY,),
    GateStage.STRUCTURE: (WaitReason.NO_STRUCTURE, WaitReason.RR_TOO_LOW),
    GateStage.OPTIONS_FLOW: (WaitReason.OPTIONS_CONTRADICTION,),
    GateStage.ORDER_FLOW: (WaitReason.ORDERFLOW_CONFLICT,),
    GateStage.MOMENTUM: (WaitReason.VOLATILITY_BAND,),
    GateStage.FLOW_SCORE: (
        WaitReason.COMPONENT_DISABLED,
        WaitReason.SCORE_BELOW_THRESHOLD,
    ),
    GateStage.ENTRY_FILTER: (
        WaitReason.ORDERFLOW_CONFLICT,
        WaitReason.OPTIONS_CONTRADICTION,
        WaitReason.NO_STRUCTURE,
        WaitReason.LIQUIDITY,
        WaitReason.VOLATILITY_BAND,
    ),
    GateStage.SETUP: (
        WaitReason.NO_SETUP_MATCH,
        WaitReason.RR_TOO_LOW,
        WaitReason.SCORE_BELOW_THRESHOLD,
        WaitReason.VOLATILITY_BAND,
        WaitReason.ORDERFLOW_CONFLICT,
        WaitReason.NO_STRUCTURE,
        WaitReason.COMPONENT_DISABLED,
    ),
    GateStage.RISK: (WaitReason.RR_TOO_LOW, WaitReason.RISK_LIMIT),
}

#: `features/levels.py`'s `level_gate_reason` -> this pipeline's reason.
#: The mapping is levels.py's own, quoted from its module docstring: "Reasons
#: 1-5 map to `WaitReason.NO_STRUCTURE` and 6-7 to `WaitReason.RR_TOO_LOW`;
#: the finer code is kept so that rejection analysis can say *which* gate did
#: the work." Built from the module's exported constants rather than from the
#: literals, so renumbering there cannot silently remap here.
STRUCTURE_WAIT_REASONS: Mapping[float, WaitReason] = {
    GATE_NO_TARGET: WaitReason.RR_TOO_LOW,
    GATE_REWARD_RISK: WaitReason.RR_TOO_LOW,
}

#: Which component's opposition is reported by which reason at the entry
#: filter. Four of the five are exact: `ORDER_FLOW` and `OPTIONS_FLOW` have
#: their own reasons, `STRUCTURE` opposing the side means there is no
#: structural bias for it, and `LIQUIDITY` has one too (it emits direction 0
#: always, so it can never appear here -- the entry is kept for completeness
#: rather than reachability).
#:
#: `VOL_MOMENTUM` is the one component for which the existing enum has NO
#: exact reason, and no new reason string may be invented. `VOLATILITY_BAND`
#: is the nearest -- it is the reason section 3 assigns to the volatility and
#: momentum stage -- and the precise fact ("VOL_MOMENTUM holds N opposing
#: points against this side") is carried verbatim in `GateResult.detail` and
#: from there into `Signal.wait_detail`, so nothing is lost beyond the
#: coarseness of the bucket. Reported as a deviation rather than papered
#: over.
OPPOSITION_WAIT_REASON: Mapping[Component, WaitReason] = {
    Component.ORDER_FLOW: WaitReason.ORDERFLOW_CONFLICT,
    Component.OPTIONS_FLOW: WaitReason.OPTIONS_CONTRADICTION,
    Component.STRUCTURE: WaitReason.NO_STRUCTURE,
    Component.LIQUIDITY: WaitReason.LIQUIDITY,
    Component.VOL_MOMENTUM: WaitReason.VOLATILITY_BAND,
}

#: Floor on `liquidity_score` at the liquidity gate. **Declared OFF.**
#:
#: Section 3's liquidity gate is "spread/depth/volume outside band -> WAIT",
#: and no document in this project supplies the band. `config.setups` carries
#: per-setup ATR-percentile bands and `config.flow_score` carries the two
#: contradiction thresholds, but there is no liquidity threshold anywhere, and
#: inventing one here would add a tuned constant to the largest overfitting
#: surface in the project while claiming it came from the specification.
#:
#: So the gate enforces the two conditions that need NO constant -- an
#: unusable quote (a measured spread below the instrument's own
#: `min_spread_ticks`) and a bar in which nothing traded -- and exposes this
#: threshold at 0.0, which never fires. It is a declared hypothesis for the
#: Phase 8 sweep, in the sense of principle 7, and it belongs in
#: `config/schema.py` as a sweepable field; adding one means editing a file
#: this phase does not own, so it is reported as a deviation instead, exactly
#: as `signals/scoring_bars.py` did with the VOL_MOMENTUM split.
#:
#: `LiquidityGate(config, min_liquidity_score=x)` overrides it without an
#: edit, so a sweep needs no code change.
LIQUIDITY_GATE_MIN_SCORE: float = 0.0


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's verdict.

    A hot-path record -- one per gate per bar -- so a slots dataclass rather
    than a Pydantic model, matching `core/contracts.py`'s split between
    market-data types and record types.

    `reasons` carries the words even when the gate PASSED, because section 12
    wants the narrative of a taken trade as well as a rejected one, and
    because "options flow could not contradict this trade: the component was
    never measured" is a fact about an accepted bar.
    """

    stage: GateStage
    passed: bool
    wait_reason: WaitReason | None = None
    detail: str = ""
    reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    side: Side | None = None

    def __post_init__(self) -> None:
        if self.passed and self.wait_reason is not None:
            raise ValueError(
                f"{self.stage.value}: a passing gate must not carry a wait_reason"
            )
        if not self.passed and self.wait_reason is None:
            raise ValueError(
                f"{self.stage.value}: a failing gate must state a wait_reason -- "
                "every WAIT is recorded with its blocking gate so that rejection "
                "analysis is possible"
            )
        allowed = WAIT_REASONS_BY_STAGE.get(self.stage, ())
        if self.wait_reason is not None and self.wait_reason not in allowed:
            raise ValueError(
                f"{self.stage.value} emitted {self.wait_reason.value}, which is "
                f"not among its declared reasons "
                f"({', '.join(r.value for r in allowed)}). WAIT_REASONS_BY_STAGE "
                "is what section 12's rejection analysis labels its buckets by; "
                "a gate that emits an undeclared reason makes the labels wrong."
            )


class Gate:
    """Marker base: a stage, and a `check` that returns a `GateResult`.

    Deliberately not an ABC with an abstract `check`: the gates take
    different arguments, because each reads exactly what it needs and no
    gate is handed a context object it could rummage through. The stage
    attribute is the shared part, and `signals/engine.py` owns the sequence.
    """

    stage: GateStage

    def _pass(self, **kwargs: object) -> GateResult:
        return GateResult(stage=self.stage, passed=True, **kwargs)  # type: ignore[arg-type]

    def _fail(self, wait_reason: WaitReason, detail: str, **kwargs: object) -> GateResult:
        return GateResult(
            stage=self.stage,
            passed=False,
            wait_reason=wait_reason,
            detail=detail,
            **kwargs,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 1. SESSION
# ---------------------------------------------------------------------------


class SessionGate(Gate):
    """Is this timestamp inside the tradable session?

    Consumes `TradingCalendar.is_tradable`, which already applies every field
    of `SessionFilterConfig` and returns one of a fixed set of reason
    strings. Nothing about holidays, half-days, the maintenance break or the
    open/close skips is re-derived here; the reason string goes into
    `GateResult.detail` so a calendar bug and a correctly-configured open
    skip stay distinguishable.

    With no calendar the gate PASSES and says so. A session filter that
    silently defaulted to "closed" would make every bar a WAIT on a dataset
    with no calendar; one that silently defaulted to "open" would hide that
    the filter was never applied. Saying so in `reasons` is the third option.
    """

    stage = GateStage.SESSION

    def __init__(
        self,
        config: FlowModelConfig,
        spec: InstrumentSpec | None = None,
        calendar: SessionCalendarProtocol | None = None,
    ) -> None:
        self._session = config.session
        self._spec = spec
        self._calendar = calendar

    def check(self, ts: datetime) -> GateResult:
        if self._calendar is None or self._spec is None:
            return self._pass(
                reasons=(
                    "session: not evaluated -- no trading calendar was supplied to "
                    "the engine, so config.session was not applied to this bar",
                )
            )
        tradable, reason = self._calendar.is_tradable(ts, self._spec, self._session)
        if tradable:
            return self._pass(reasons=("session: inside the tradable session",))
        return self._fail(
            WaitReason.SESSION_CLOSED,
            f"the session filter rejects {ts.isoformat()}: {reason}",
            reasons=(f"session: closed or filtered ({reason})",),
        )


# ---------------------------------------------------------------------------
# 2. MARKET DATA
# ---------------------------------------------------------------------------


class DataQualityGate(Gate):
    """Is the data good enough to act on at this instant?

    Consumes `data/quality.py`'s point-in-time `QualityGrader.grade(view)`
    report. Staleness, coverage, cadence and the per-component feed floors
    are all decided there and none of it is re-derived: this gate asks
    `QualityReport.is_tradable(config.data.min_quality_to_trade)` and reports
    the grader's own note.

    The brief: "Data quality status per signal (GOOD/DEGRADED/MISSING/STALE);
    return WAIT if required data is missing or stale." A non-empty
    `blocking_feeds` is a no independently of `overall`, which is why
    `is_tradable` is asked rather than `overall.is_tradable` -- a feed graded
    DEGRADED can clear a dataset-wide DEGRADED minimum while failing the GOOD
    floor its own component declared.

    **This is the gate that fires on bars-only data.** `order_flow` requires
    `tick_aggregate` and `options_flow` requires `options_snapshot` in
    `config.flow_score.feed_requirements`, so both feeds are blocking and
    every bar is a `DATA_QUALITY` WAIT. The Flow Score aggregate's strict
    refusal (`COMPONENT_DISABLED`) is a later stage and therefore not the
    reported reason there -- it is what fires when a configuration passes
    this gate with a component still unmeasured.
    """

    stage = GateStage.MARKET_DATA

    def __init__(self, config: FlowModelConfig | DataConfig) -> None:
        self._data = config.data if isinstance(config, FlowModelConfig) else config

    def check(self, report: QualityReport | None) -> GateResult:
        minimum = self._data.min_quality_to_trade
        if report is None:
            return self._pass(
                reasons=(
                    "market data: not graded -- no QualityGrader was supplied to "
                    "the engine, so feed quality was not checked on this bar",
                )
            )
        diagnostics = {
            "overall_quality_rank": float(report.overall.rank),
            "blocking_feed_count": float(len(report.blocking_feeds)),
        }
        if report.is_tradable(minimum):
            return self._pass(
                reasons=(
                    f"market data: {report.overall.value}, at or above the "
                    f"{minimum.value} floor",
                ),
                diagnostics=diagnostics,
            )
        blocking = ", ".join(f.value for f in report.blocking_feeds)
        note = report.note or f"overall {report.overall.value} below {minimum.value}"
        return self._fail(
            WaitReason.DATA_QUALITY,
            f"data quality {report.overall.value} is not tradable at the "
            f"{minimum.value} floor"
            + (f"; blocking feed(s): {blocking}" if blocking else "")
            + f". {note}",
            reasons=(
                f"market data: WAIT -- {report.overall.value}"
                + (f", blocking {blocking}" if blocking else "")
                + ". Required data is missing or stale, so no signal is produced "
                "and nothing is estimated in its place.",
            ),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 3. WARMUP
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComputerReadiness:
    """One feature computer's readiness at one bar."""

    name: str
    feeds_present: bool
    bars_required: int
    bars_available: int
    absent_feeds: tuple[Feed, ...] = ()

    @property
    def warm(self) -> bool:
        return self.bars_available >= self.bars_required

    @property
    def blocks_warmup(self) -> bool:
        """True when this computer's window is still filling.

        A computer whose FEED is absent does not block warmup. Its window
        will never fill, and reporting that as `WARMUP` would mean a
        bars-only dataset waited forever for a tick feed it does not have.
        That is an availability fact, and it is reported by
        `data/quality.py` and by the Flow Score aggregate's refusal.
        """
        return self.feeds_present and not self.warm


@dataclass(frozen=True, slots=True)
class WarmupReport:
    """Per-computer readiness, plus the bar count the view could serve."""

    bars_available: int
    computers: tuple[ComputerReadiness, ...]

    @property
    def blocking(self) -> tuple[ComputerReadiness, ...]:
        return tuple(c for c in self.computers if c.blocks_warmup)

    @property
    def bars_required(self) -> int:
        """The longest window among computers whose feeds are present."""
        return max(
            (c.bars_required for c in self.computers if c.feeds_present), default=0
        )


def assess_warmup(
    computers: Sequence[FeatureComputer], view: MarketView
) -> WarmupReport:
    """Readiness of each computer, judged per computer rather than in bulk.

    `FeatureVector.warmup_complete` on a merged bundle is the AND over every
    computer, including ones whose feed is absent. On bars-only data the
    order-flow and options-flow computers are permanently not-ready, so the
    merged flag is permanently False, and a warmup gate reading it would
    report `WARMUP` on every bar of a ten-year run for a reason that has
    nothing to do with warmup. `signals/scoring_bars.py` documents the same
    trap for component availability and reaches the same answer: judge per
    computer, not on the bundle's aggregate.

    Reads only `bar_count` and `has_feed` off the view.
    """
    available = int(view.bar_count())
    readiness = tuple(
        ComputerReadiness(
            name=computer.name,
            feeds_present=computer.feeds_available(view),
            bars_required=int(computer.warmup_bars),
            bars_available=available,
            absent_feeds=tuple(
                feed for feed in sorted(computer.required_feeds, key=lambda f: f.value)
                if not view.has_feed(feed)
            ),
        )
        for computer in computers
    )
    return WarmupReport(bars_available=available, computers=readiness)


class WarmupGate(Gate):
    """Has every computer whose feeds exist filled its window?

    The classic bug this prevents is trading on a half-filled rolling
    window: a 252-bar ATR percentile computed from 30 bars is a different
    statistic with the same name.
    """

    stage = GateStage.WARMUP

    def check(self, report: WarmupReport) -> GateResult:
        diagnostics = {
            "bars_available": float(report.bars_available),
            "bars_required": float(report.bars_required),
            "blocking_computer_count": float(len(report.blocking)),
        }
        if report.bars_available <= 0:
            return self._fail(
                WaitReason.WARMUP,
                "the view serves no bars at all",
                reasons=("warmup: no bars are visible at this timestamp",),
                diagnostics=diagnostics,
            )
        blocking = report.blocking
        if not blocking:
            return self._pass(
                reasons=(
                    f"warmup: complete -- {report.bars_available} bars visible, "
                    f"{report.bars_required} required by the slowest computer "
                    "whose feeds are present",
                ),
                diagnostics=diagnostics,
            )
        worst = max(blocking, key=lambda c: c.bars_required)
        named = ", ".join(f"{c.name}({c.bars_required})" for c in blocking)
        return self._fail(
            WaitReason.WARMUP,
            f"{report.bars_available} bars visible; still filling: {named}",
            reasons=(
                f"warmup: incomplete -- {worst.name} needs "
                f"{worst.bars_required} bars and {report.bars_available} are "
                "visible. No feature is read from a partly-filled window.",
            ),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 4. REGIME
# ---------------------------------------------------------------------------


class RegimeGate(Gate):
    """Is the measured regime one some enabled setup will trade?

    `Regime.UNKNOWN` is never tradable. By this stage the warmup gate has
    already passed, so UNKNOWN here means the detector's hysteresis found no
    run of `min_regime_bars` identical raw labels in its search window -- a
    genuine "no regime is established", not a short history.

    The union over enabled setups is used rather than a single
    `config.allowed_regimes`, which does not exist: regimes are allowed PER
    SETUP. The determined setup's own `allowed_regimes` is re-applied in
    `signals/setups.py`.
    """

    stage = GateStage.REGIME

    def __init__(self, config: FlowModelConfig) -> None:
        self._allowed = tradable_regimes(config)

    def check(self, regime: RegimeState) -> GateResult:
        diagnostics = {
            "regime_confidence": float(regime.confidence),
            "bars_in_regime": float(regime.bars_in_regime),
        }
        if regime.regime is Regime.UNKNOWN:
            return self._fail(
                WaitReason.REGIME_BLOCKED,
                "the regime is UNKNOWN: no confirmed run of identical raw labels "
                "inside the detector's hysteresis window, so no regime is "
                "established at this bar",
                reasons=(
                    "regime: UNKNOWN -- never tradable. The label was not carried "
                    "forward from beyond the hysteresis horizon.",
                ),
                diagnostics=diagnostics,
            )
        if regime.regime not in self._allowed:
            allowed = ", ".join(sorted(r.value for r in self._allowed))
            return self._fail(
                WaitReason.REGIME_BLOCKED,
                f"regime {regime.regime.value} is allowed by no enabled setup "
                f"(enabled setups allow: {allowed})",
                reasons=(
                    f"regime: {regime.regime.value} (confidence "
                    f"{regime.confidence:.2f}) -- no enabled setup trades it",
                ),
                diagnostics=diagnostics,
            )
        return self._pass(
            reasons=(
                f"regime: {regime.regime.value} (confidence "
                f"{regime.confidence:.2f}, {regime.bars_in_regime} bars in regime)",
            ),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 5. LIQUIDITY
# ---------------------------------------------------------------------------


class LiquidityGate(Gate):
    """Can this bar be executed at all?

    Three checks, in order:

    1. A liquidity measurement exists. `liquidity_score` absent or graded
       MISSING means the component was never computed, and a trade admitted
       on an unmeasured execution cost is the thing section 0.4 forbids.
    2. The quote, when it was MEASURED, is usable: a spread below the
       instrument's own `min_spread_ticks` is a broken or crossed book, not a
       tight one. Skipped when `spread_ticks` is the configured fallback
       rather than a measurement -- `features/liquidity.py` grades the key
       DEGRADED in exactly that case -- because a fallback cannot be
       narrower than the instrument's minimum by accident, and judging it
       would be judging configuration.
    3. Something traded. `relative_volume <= 0` is a bar in which no volume
       printed; a position cannot be filled out of it at any participation
       rate.

    Then the declared-off `min_liquidity_score` floor. See
    `LIQUIDITY_GATE_MIN_SCORE` on why section 3's "band" is not implemented
    as a tuned constant.
    """

    stage = GateStage.LIQUIDITY

    def __init__(
        self,
        config: FlowModelConfig,
        spec: InstrumentSpec | None = None,
        *,
        min_liquidity_score: float | None = None,
    ) -> None:
        self._spec = spec
        self._min_score = (
            LIQUIDITY_GATE_MIN_SCORE
            if min_liquidity_score is None
            else float(min_liquidity_score)
        )
        if not 0.0 <= self._min_score <= 1.0:
            raise ValueError(
                f"min_liquidity_score must lie in [0, 1] (liquidity_score is a "
                f"bounded magnitude); got {self._min_score}"
            )

    def check(self, features: FeatureVector) -> GateResult:
        score = _measured(features, "liquidity_score")
        diagnostics: dict[str, float] = {}
        if score is None:
            return self._fail(
                WaitReason.LIQUIDITY,
                "liquidity_score is absent or graded MISSING: no execution-cost "
                "measurement exists for this bar",
                reasons=(
                    "liquidity: WAIT -- no liquidity measurement. Nothing is "
                    "estimated in its place.",
                ),
            )
        diagnostics["liquidity_score"] = score

        spread = _measured(features, "spread_ticks")
        spread_is_measurement = features.quality_of("spread_ticks") is DataQuality.GOOD
        if spread is not None:
            diagnostics["spread_ticks"] = spread
        if (
            spread is not None
            and spread_is_measurement
            and self._spec is not None
            and spread < float(self._spec.min_spread_ticks)
        ):
            return self._fail(
                WaitReason.LIQUIDITY,
                f"measured spread {spread:.4f} ticks is below "
                f"{self._spec.symbol}'s min_spread_ticks "
                f"{self._spec.min_spread_ticks}: the quote is unusable, not tight",
                reasons=(
                    f"liquidity: WAIT -- measured spread {spread:.4f} ticks is "
                    f"narrower than the instrument's minimum "
                    f"{self._spec.min_spread_ticks}, which is a broken book",
                ),
                diagnostics=diagnostics,
            )

        relative_volume = _measured(features, "relative_volume")
        if relative_volume is not None:
            diagnostics["relative_volume"] = relative_volume
            if relative_volume <= 0.0:
                return self._fail(
                    WaitReason.LIQUIDITY,
                    "relative_volume is 0: no volume printed in this bar, so there "
                    "is nothing to be filled out of",
                    reasons=("liquidity: WAIT -- no volume traded in this bar",),
                    diagnostics=diagnostics,
                )

        if score < self._min_score:
            return self._fail(
                WaitReason.LIQUIDITY,
                f"liquidity_score {score:.4f} is below the configured floor "
                f"{self._min_score:.4f}",
                reasons=(
                    f"liquidity: WAIT -- liquidity_score {score:.4f} below the "
                    f"floor {self._min_score:.4f}",
                ),
                diagnostics=diagnostics,
            )
        reasons = [
            f"liquidity: liquidity_score {score:.3f}"
            + (f", spread {spread:.2f} ticks" if spread is not None else "")
        ]
        if spread is not None and not spread_is_measurement:
            reasons.append(
                "liquidity: the spread is the instrument's configured fallback, "
                "not a measurement; features/liquidity.py has already capped "
                "liquidity_score for it"
            )
        return self._pass(reasons=tuple(reasons), diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# 6. STRUCTURE
# ---------------------------------------------------------------------------


class StructureGate(Gate):
    """Does section 14's S/R engine offer a directional bias here?

    Entirely a consumer of `features/levels.py`. Section 14.6 fixes the four
    conditions -- a zone within `entry_atr_window * ATR`, `S >= s_major`,
    `C >= c_min`, and the binary rejection requirement -- and levels.py
    already evaluates them in that order, reporting the first failure in
    `level_gate_reason` and the overall verdict in `structure_gate_passed`.
    None of it is re-derived here, which matters because a second
    implementation of "is this level major" would be a second answer.

    The gate's own contribution is the SIDE: `level_direction` is +1 at
    support and -1 at resistance, and the `Side` it returns is the only side
    the rest of the pipeline considers. See the module docstring on why that
    is the structural defence against taking the majority direction.
    """

    stage = GateStage.STRUCTURE

    def check(self, features: FeatureVector) -> GateResult:
        for key in ("structure_gate_passed", "level_direction", "level_gate_reason"):
            if key not in features.values:
                return self._fail(
                    WaitReason.NO_STRUCTURE,
                    f"feature key {key!r} is absent: the section 14 level engine "
                    "produced no verdict for this bar",
                    reasons=(f"structure: WAIT -- {key} absent from the vector",),
                )
        if features.quality_of("structure_gate_passed") is DataQuality.MISSING:
            return self._fail(
                WaitReason.NO_STRUCTURE,
                "structure_gate_passed is graded MISSING: the level engine made no "
                "measurement at this bar",
                reasons=("structure: WAIT -- no level measurement",),
            )

        passed = float(features.require("structure_gate_passed"))
        reason_code = float(features.require("level_gate_reason"))
        direction = float(features.require("level_direction"))
        diagnostics = {
            "level_gate_reason": reason_code,
            "level_direction": direction,
            "level_significance": float(features.get("level_significance", 0.0) or 0.0),
            "level_cleanliness": float(features.get("level_cleanliness", 0.0) or 0.0),
            "level_rejection": float(features.get("level_rejection", 0.0) or 0.0),
        }

        if not math.isclose(passed, 1.0, abs_tol=1e-9):
            wait_reason = STRUCTURE_WAIT_REASONS.get(reason_code, WaitReason.NO_STRUCTURE)
            return self._fail(
                wait_reason,
                f"section 14.6's structure gate failed with level_gate_reason="
                f"{reason_code:.0f} ({_structure_reason_text(reason_code)})",
                reasons=(
                    f"structure: WAIT -- {_structure_reason_text(reason_code)} "
                    f"(level_gate_reason={reason_code:.0f})",
                ),
                diagnostics=diagnostics,
            )
        if not math.isclose(reason_code, GATE_PASSED, abs_tol=1e-9):
            return self._fail(
                WaitReason.NO_STRUCTURE,
                f"structure_gate_passed is 1 while level_gate_reason is "
                f"{reason_code:.0f}; the level engine's two outputs disagree and "
                "the bar is declined rather than resolved by preferring one",
                reasons=("structure: WAIT -- inconsistent level-engine verdict",),
                diagnostics=diagnostics,
            )
        side = _side_of(direction)
        if side is None:
            return self._fail(
                WaitReason.NO_STRUCTURE,
                f"the structure gate passed but level_direction is {direction:+.0f}: "
                "price is inside the zone, so there is no side to trade",
                reasons=(
                    "structure: WAIT -- the level offers no direction "
                    "(price inside the zone)",
                ),
                diagnostics=diagnostics,
            )
        at = "support" if side is Side.LONG else "resistance"
        return self._pass(
            side=side,
            reasons=(
                f"structure: {side.value} at {at} -- S="
                f"{diagnostics['level_significance']:.3f}, C="
                f"{diagnostics['level_cleanliness']:.3f}, R="
                f"{diagnostics['level_rejection']:.3f}, all four of 14.6's gates "
                "passed",
            ),
            diagnostics=diagnostics,
        )


def _structure_reason_text(code: float) -> str:
    """14.6's gate codes in words. The numbers live in `features/levels.py`."""
    return {
        GATE_NO_ZONE: "no zone was detected at all",
        GATE_PROXIMITY: "price is further than entry_atr_window * ATR from the zone",
        GATE_SIGNIFICANCE: "significance S is below min_significance ('major')",
        GATE_CLEANLINESS: "cleanliness C is below min_cleanliness ('clean')",
        GATE_REJECTION: "14.5's binary rejection requirement was not met",
        GATE_NO_TARGET: "no opposing major zone exists to target",
        GATE_REWARD_RISK: "achievable R:R is below min_reward_risk",
    }.get(code, f"unknown level_gate_reason {code}")


def _side_of(direction: float) -> Side | None:
    if math.isclose(direction, DIRECTION_SUPPORT, abs_tol=1e-9):
        return Side.LONG
    if math.isclose(direction, DIRECTION_RESISTANCE, abs_tol=1e-9):
        return Side.SHORT
    if math.isclose(direction, DIRECTION_NONE, abs_tol=1e-9):
        return None
    raise ValueError(
        f"level_direction={direction} is not one of "
        f"{{{DIRECTION_RESISTANCE}, {DIRECTION_NONE}, {DIRECTION_SUPPORT}}}; a "
        "fractional direction is a magnitude wearing a direction's name"
    )


# ---------------------------------------------------------------------------
# 7. OPTIONS FLOW -- section 3's ContradictionGate
# ---------------------------------------------------------------------------


class OptionsFlowGate(Gate):
    """Does measured options flow strongly oppose the trade?

    Section 3: "options_opposition > limit -> WAIT('options_contradiction')".
    The limit is `config.flow_score.options_contradiction_points`, in
    WEIGHTED POINTS, so a capped EOD magnitude opposes with less force than
    an intraday one without the cap being applied twice -- the cap already
    reduced the magnitude, and points are magnitude times weight.

    An UNAVAILABLE component does not contradict anything. "Never invent
    missing options data": direction 0 on an unmeasured component means no
    observation, not agreement and not disagreement. A setup that needs the
    component says so through its own flags.
    """

    stage = GateStage.OPTIONS_FLOW

    def __init__(self, config: FlowModelConfig) -> None:
        self._limit = float(config.flow_score.options_contradiction_points)

    def check(
        self, components: Mapping[Component, ComponentScore], side: Side
    ) -> GateResult:
        score = components.get(Component.OPTIONS_FLOW)
        if score is None or not score.enabled:
            return self._pass(
                reasons=(
                    "options flow: not measured, so it neither confirms nor "
                    "contradicts. Its weight is reported unavailable rather than "
                    "scored as neutral.",
                )
            )
        opposing = score.points if score.opposes(side) else 0.0
        diagnostics = {
            "options_opposing_points": round(opposing, 10),
            "options_contradiction_limit": self._limit,
            "options_direction": float(score.direction),
        }
        if opposing > self._limit:
            return self._fail(
                WaitReason.OPTIONS_CONTRADICTION,
                f"options flow opposes a {side.value} with {opposing:.2f} weighted "
                f"points, above the {self._limit:.2f}-point contradiction limit",
                reasons=(
                    f"options flow: WAIT -- direction {score.direction:+d} opposes "
                    f"a {side.value} with {opposing:.2f} points (limit "
                    f"{self._limit:.2f})",
                ),
                diagnostics=diagnostics,
            )
        return self._pass(
            reasons=(
                f"options flow: direction {score.direction:+d}, "
                f"{score.points:.2f} points, {opposing:.2f} of them opposing a "
                f"{side.value} (limit {self._limit:.2f})",
            ),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 8. ORDER FLOW -- section 3's ConfirmationGate
# ---------------------------------------------------------------------------


class OrderFlowGate(Gate):
    """Does measured order flow disagree with the structure's direction?

    Section 3: "ConfirmationGate (orderflow must agree with structure
    direction) |-- disagree -> WAIT('orderflow_conflict')". Disagreement is
    read off the DIRECTION, not off a magnitude threshold: direction is
    {-1, 0, +1} by construction, so any opposing sign is a disagreement
    however small the magnitude behind it. The magnitude-weighted version of
    the same question is `max_opposing_points` at the entry filter.

    A measured NEUTRAL (direction 0) is not a disagreement and passes here.
    Whether neutral is good enough is a per-setup question --
    `require_orderflow_confirmation` -- and is answered in
    `signals/setups.py`.

    An UNAVAILABLE component is not a disagreement either. Section 5 accepts
    no proxy for a tick feed, and a bars-only dataset would otherwise report
    "order flow conflicted" on every bar about a tape nobody observed.
    """

    stage = GateStage.ORDER_FLOW

    def check(
        self, components: Mapping[Component, ComponentScore], side: Side
    ) -> GateResult:
        score = components.get(Component.ORDER_FLOW)
        if score is None or not score.enabled:
            return self._pass(
                reasons=(
                    "order flow: not measured, so it cannot confirm or conflict. "
                    "No proxy is substituted (section 5); a setup that requires "
                    "confirmation declines the bar for that reason instead.",
                )
            )
        diagnostics = {
            "order_flow_direction": float(score.direction),
            "order_flow_points": round(score.points, 10),
        }
        if score.opposes(side):
            return self._fail(
                WaitReason.ORDERFLOW_CONFLICT,
                f"order flow direction {score.direction:+d} disagrees with a "
                f"{side.value} ({side.sign:+d}) read off the structure",
                reasons=(
                    f"order flow: WAIT -- direction {score.direction:+d} conflicts "
                    f"with the {side.value} the structure names",
                ),
                diagnostics=diagnostics,
            )
        verdict = "confirms" if score.agrees_with(side) else "is neutral on"
        return self._pass(
            reasons=(
                f"order flow: direction {score.direction:+d} {verdict} the "
                f"{side.value} ({score.points:.2f} points)",
            ),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 9. MOMENTUM -- section 3's VolatilityGate
# ---------------------------------------------------------------------------


class MomentumGate(Gate):
    """Is the ATR percentile inside a band some enabled setup will trade?

    Section 3: "VolatilityGate (ATR percentile within setup band) |-- outside
    -> WAIT('volatility_band')". The setup is not known yet, so the band is
    the union over enabled setups and the determined setup's own band is
    re-applied in `signals/setups.py`.

    `atr_percentile` is the quantity section 3 names. `vol_regime_score` --
    the bounded VOL_MOMENTUM magnitude input -- is a different thing ("the
    range is in the payable band") and is not substituted for it here.

    No directional check. Section 3's confirmation gates are order flow and
    options flow; requiring momentum to agree with the structure would be a
    new trading rule, not an implementation of an existing one. Momentum
    opposition is instead weighed, with every other component's, at the
    entry filter.
    """

    stage = GateStage.MOMENTUM

    def __init__(self, config: FlowModelConfig) -> None:
        self._low, self._high = vol_percentile_band(config)

    def check(self, features: FeatureVector) -> GateResult:
        value = _measured(features, "atr_percentile")
        if value is None:
            return self._fail(
                WaitReason.VOLATILITY_BAND,
                "atr_percentile is absent or graded MISSING, so the setup "
                "volatility band cannot be evaluated",
                reasons=(
                    "momentum: WAIT -- no ATR percentile measurement, so the "
                    "volatility band is unevaluated and the bar is declined "
                    "rather than admitted on an unmeasured quantity",
                ),
            )
        diagnostics = {
            "atr_percentile": value,
            "vol_band_low": self._low,
            "vol_band_high": self._high,
        }
        if not (self._low <= value <= self._high):
            return self._fail(
                WaitReason.VOLATILITY_BAND,
                f"ATR percentile {value:.3f} is outside the union band "
                f"[{self._low:.2f}, {self._high:.2f}] of every enabled setup",
                reasons=(
                    f"momentum: WAIT -- ATR percentile {value:.3f} outside "
                    f"[{self._low:.2f}, {self._high:.2f}], the widest band any "
                    "enabled setup allows",
                ),
                diagnostics=diagnostics,
            )
        return self._pass(
            reasons=(
                f"momentum: ATR percentile {value:.3f} inside the union band "
                f"[{self._low:.2f}, {self._high:.2f}]",
            ),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 10. FLOW SCORE
# ---------------------------------------------------------------------------


class FlowScoreGate(Gate):
    """Did the aggregate produce a Flow Score, and does it clear a threshold?

    Two failures, in this order:

    1. The aggregate REFUSED -- `strict_component_availability` with an
       unmeasured component. `WaitReason.COMPONENT_DISABLED`, carried from
       the refusal rather than chosen here.
    2. The score is below `min_flow_score_required`, the lowest threshold any
       enabled setup declares. The determined setup's own `min_flow_score` is
       re-applied in `signals/setups.py`, which reports the same
       `SCORE_BELOW_THRESHOLD` section 3 assigns to that comparison.

    The refusal is checked first because a refused aggregate has no score to
    compare: reporting `SCORE_BELOW_THRESHOLD` for a bar that produced no
    score would be a threshold comparison against a number that does not
    exist.
    """

    stage = GateStage.FLOW_SCORE

    def __init__(self, config: FlowModelConfig) -> None:
        self._minimum = min_flow_score_required(config)

    def check(self, outcome: FlowScoreOutcome) -> GateResult:
        diagnostics = {
            "available_points": outcome.available_points,
            "max_points": outcome.max_points,
            "missing_points": outcome.missing_points,
            "min_flow_score": self._minimum,
        }
        if outcome.refusal is not None:
            return self._fail(
                outcome.refusal.wait_reason,
                outcome.refusal.detail,
                reasons=(
                    f"flow score: REFUSED -- only {outcome.available_points:.1f} of "
                    f"{outcome.max_points:.1f} points are computable "
                    f"({outcome.missing_points:.1f} have no feed) and the weight "
                    "was not redistributed",
                ),
                diagnostics=diagnostics,
            )
        flow = outcome.require()
        diagnostics["flow_score"] = flow.score
        if flow.score < self._minimum:
            return self._fail(
                WaitReason.SCORE_BELOW_THRESHOLD,
                f"flow score {flow.score:.2f} is below {self._minimum:.2f}, the "
                "lowest min_flow_score any enabled setup declares",
                reasons=(
                    f"flow score: WAIT -- {flow.score:.2f} of "
                    f"{outcome.available_points:.1f} available points, below the "
                    f"lowest setup threshold {self._minimum:.2f}",
                ),
                diagnostics=diagnostics,
            )
        return self._pass(
            reasons=(
                f"flow score: {flow.score:.2f} of {outcome.available_points:.1f} "
                f"available points (max {outcome.max_points:.1f}), at or above the "
                f"lowest setup threshold {self._minimum:.2f}",
            ),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 11. ENTRY FILTER
# ---------------------------------------------------------------------------


class EntryFilterGate(Gate):
    """Does the weight opposing this side exceed `max_opposing_points`?

    This is the magnitude-weighted form of section 7's direction-agreement
    rule, and it is the gate that catches the adversarial case the brief
    names: one component strongly bearish, the rest strongly bullish. The
    aggregate magnitude stays high -- a disagreement is evidence, not
    cancellation -- and the trade is refused because the bearish component
    holds more than `max_opposing_points` against the only side on offer.

    The side is the structure's. There is no branch here that reads
    `FlowScore.net_direction()` and trades the majority; the only question
    asked is whether the candidate side is opposed too strongly.

    The reported reason is the one belonging to the component holding the
    most opposing points. See `OPPOSITION_WAIT_REASON` on the single case --
    `VOL_MOMENTUM` -- for which the existing enum has no exact reason.
    """

    stage = GateStage.ENTRY_FILTER

    def __init__(self, config: FlowModelConfig) -> None:
        self._limit = float(config.flow_score.max_opposing_points)

    def check(self, flow: FlowScore, side: Side) -> GateResult:
        opposing = flow.opposing_points(side)
        agreeing = round(
            sum(c.points for c in flow.components.values() if c.agrees_with(side)), 6
        )
        diagnostics = {
            "opposing_points": opposing,
            "agreeing_points": agreeing,
            "max_opposing_points": self._limit,
        }
        if opposing > self._limit:
            dominant = max(
                (c for c in flow.components.values() if c.opposes(side)),
                key=lambda c: c.points,
            )
            reason = OPPOSITION_WAIT_REASON[dominant.component]
            named = ", ".join(
                f"{c.component.value} {c.points:.2f}"
                for c in flow.components.values()
                if c.opposes(side)
            )
            return self._fail(
                reason,
                f"components opposing a {side.value} hold {opposing:.2f} weighted "
                f"points, above the {self._limit:.2f}-point limit ({named}); the "
                f"largest opposer is {dominant.component.value}",
                reasons=(
                    f"entry filter: WAIT -- {opposing:.2f} points oppose a "
                    f"{side.value} against a {self._limit:.2f}-point limit "
                    f"({named}). The aggregate magnitude is unaffected: a "
                    "disagreement is evidence, and section 7 has the gates refuse "
                    "the trade rather than the score absorb the conflict.",
                ),
                diagnostics=diagnostics,
            )
        return self._pass(
            side=side,
            reasons=(
                f"entry filter: {agreeing:.2f} points agree with the "
                f"{side.value}, {opposing:.2f} oppose it (limit {self._limit:.2f})",
            ),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def pipeline_gates(
    config: FlowModelConfig,
    spec: InstrumentSpec | None = None,
    calendar: SessionCalendarProtocol | None = None,
    *,
    min_liquidity_score: float | None = None,
) -> dict[GateStage, Gate]:
    """Every gate this module owns, keyed by stage.

    `GateStage.SETUP` and `GateStage.RISK` are absent: setup selection lives
    in `signals/setups.py` and the risk stage is `risk/sizing.py` plus
    `risk/limits.py`, both of which already existed and are consumed rather
    than wrapped. `signals/engine.py` runs the stages in `GateStage` order
    and produces a `GateResult` for those two from the modules that own them.
    """
    return {
        GateStage.SESSION: SessionGate(config, spec, calendar),
        GateStage.MARKET_DATA: DataQualityGate(config),
        GateStage.WARMUP: WarmupGate(),
        GateStage.REGIME: RegimeGate(config),
        GateStage.LIQUIDITY: LiquidityGate(
            config, spec, min_liquidity_score=min_liquidity_score
        ),
        GateStage.STRUCTURE: StructureGate(),
        GateStage.OPTIONS_FLOW: OptionsFlowGate(config),
        GateStage.ORDER_FLOW: OrderFlowGate(),
        GateStage.MOMENTUM: MomentumGate(config),
        GateStage.FLOW_SCORE: FlowScoreGate(config),
        GateStage.ENTRY_FILTER: EntryFilterGate(config),
    }


def _measured(features: FeatureVector, key: str) -> float | None:
    """A feature value, or None when absent, graded MISSING, or non-finite.

    MISSING is the grade `FeatureComputer._not_ready` writes alongside its
    placeholder zeros, so reading a value at that grade would read a
    placeholder as a measurement of zero.
    """
    if key not in features.values:
        return None
    if features.quality_of(key) is DataQuality.MISSING:
        return None
    value = float(features.values[key])
    return value if math.isfinite(value) else None
