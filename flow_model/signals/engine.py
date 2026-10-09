"""The signal engine: one `MarketView` in, one `Signal` out.

This is the object section 12 requires live trading and the backtest to
share: "`live/paper.py` -- reuses the *identical* `signal_engine` object as
the backtest (no parallel implementation; this is enforced by a test that
runs both paths over the same synthetic data and asserts identical
signals)."

## Why there is no state

`SignalEngine.evaluate(view)` is a pure function of `(config, view)` plus the
two arguments a caller supplies explicitly (`equity` and a
`RiskLimitManager`). The engine holds configuration, the feature computers,
the regime detector, the five component scorers and the gates -- every one of
them constructed from config alone, with no market-data accessor -- and it
holds no accumulator, no previous bar, no cache keyed by time.

That is not stylistic. `regime/detector.py` records the reason: "a stateful
detector there would have made backtest results depend on evaluation order,
and that was caught as a HIGH finding". The same hazard applies here and the
same answer is taken: everything is recomputed from the visible prefix, so a
sequential backtest walk and a single live call at the same timestamp produce
the identical `Signal`. `tests/unit/test_signal_engine.py` asserts exactly
that by evaluating a bar cold and then again after walking the prefix.

The two stateful inputs are passed in rather than held:

* `equity` -- a number the ledger owns.
* `limits` -- the `RiskLimitManager` from `risk/limits.py`, which is stateful
  by design and takes its clock as an argument for the same reason. The
  engine never constructs one, so two engines cannot disagree about how many
  consecutive losses have occurred.

`availability` is a constructor argument and is NOT recomputed per bar.
`QualityGrader.availability()` reads the whole series and is deliberately
forward-looking; its own docstring says it "must NEVER be called per bar
inside a backtest". It is a property of the dataset, computed once before a
run, so holding it cannot make one bar's answer depend on another's.

## The engine is per symbol

`LiquidityFeatures`, `StructureFeatures` and `LevelFeatures` need an
`InstrumentSpec` at construction, so the computer set is per instrument.
Rather than cache a bundle per symbol -- state, however benign -- the engine
takes its symbol at construction and refuses a view for any other. A
multi-symbol run holds one engine per symbol.

## Short-circuiting

The stages run in `GateStage` order and the first failure ends the
evaluation, exactly as section 3 describes ("cheap/structural rejections run
before expensive scoring"). Each stage computes only what it needs: a bar
rejected on session or data quality never pays for the feature bundle, and a
bar rejected on warmup never pays for scoring. `EngineDecision.gate_results`
holds one result per stage REACHED, in order, so the implemented order is
checkable against `GateStage` rather than asserted in prose, and
`Signal.reasons` carries the narrative of every stage that ran.

Every outcome carries a reason, including every WAIT. That is what makes
section 12's rejection analysis -- how often each gate rejects a trade --
answerable at all.

## What the engine does not do

* It does not build a `TradeIntent`. That is the risk manager's output in
  Phase 7. `EntryPlan.to_trade_intent()` is offered as the conversion, so
  the geometry is validated by the Phase 1 contract rather than re-checked
  here.
* It does not read `config.research_targets` or the pre-registered
  hypotheses, and `Signal.confidence` is NOT a probability of profit. It is
  the Flow Score as a fraction of the points that were actually measured --
  a normalized score and nothing more. No module in this package claims an
  edge; whether any of this works is a Phase 7+ question that has not been
  asked yet.
* It does not re-derive data quality, feed availability, level
  significance, setup bands or risk limits. Each of those already has an
  owner and is consumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping, Sequence

from pydantic import Field, computed_field, model_validator

from flow_model.config.schema import FlowModelConfig
from flow_model.core.contracts import (
    ComponentScore,
    FeatureVector,
    FlowScore,
    RegimeState,
    Signal,
    TradeIntent,
)
from flow_model.core.enums import (
    Component,
    DataQuality,
    Regime,
    SetupType,
    Side,
    SignalAction,
    WaitReason,
)
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel
from flow_model.data.base import SessionCalendarProtocol
from flow_model.data.market_view import MarketView
from flow_model.data.quality import AvailabilityReport, QualityGrader, QualityReport
from flow_model.features.base import FeatureBundle, FeatureComputer
from flow_model.features.levels import LevelFeatures
from flow_model.features.liquidity import LiquidityFeatures
from flow_model.features.momentum import MomentumFeatures
from flow_model.features.optionsflow import OptionsFlowFeatures
from flow_model.features.orderflow import OrderFlowFeatures
from flow_model.features.structure import StructureFeatures
from flow_model.features.volatility import VolatilityFeatures
from flow_model.regime.detector import RegimeDetector
from flow_model.risk.limits import LimitDecision, RiskLimitManager
from flow_model.risk.sizing import SizingDecision, SizingRejection, size_position
from flow_model.signals import setups as setup_module
from flow_model.signals.flow_score import (
    AnyComponentScorer,
    FlowScoreOutcome,
    ScoredComponents,
    aggregate,
    all_scorers,
    score_components,
)
from flow_model.signals.gates import (
    Gate,
    GateResult,
    GateStage,
    WarmupReport,
    assess_warmup,
    pipeline_gates,
)
from flow_model.signals.setups import SetupSelection

__all__ = [
    "EngineDecision",
    "EngineError",
    "EntryPlan",
    "PrecomputedInputs",
    "SIZING_WAIT_REASONS",
    "SignalEngine",
    "StageInputs",
    "ViewInputs",
    "feature_computers",
]


#: `SizingRejection` code -> the pipeline's `WaitReason`.
#:
#: `REWARD_RISK_BELOW_MINIMUM` is `RR_TOO_LOW` and the rest are `RISK_LIMIT`.
#: `risk/sizing.py` refuses a trade whose grid-realizable R falls below the
#: setup's `min_reward_risk` "rather than relabelled: a trade that cannot
#: reach its stated R is not that setup", and that refusal is surfaced as a
#: WAIT here rather than worked around by widening the stop, moving the
#: target or picking a smaller setup -- any of which would relabel the trade
#: the refusal exists to prevent.
SIZING_WAIT_REASONS: Mapping[str, WaitReason] = {
    SizingRejection.REWARD_RISK_BELOW_MINIMUM: WaitReason.RR_TOO_LOW,
    SizingRejection.STOP_TOO_TIGHT: WaitReason.RISK_LIMIT,
    SizingRejection.STOP_TOO_WIDE: WaitReason.RISK_LIMIT,
    SizingRejection.SIZE_ZERO: WaitReason.RISK_LIMIT,
    SizingRejection.RISK_EXCEEDS_CAP: WaitReason.RISK_LIMIT,
    SizingRejection.DEGENERATE_GEOMETRY: WaitReason.RISK_LIMIT,
    SizingRejection.NON_FINITE_INPUT: WaitReason.RISK_LIMIT,
    SizingRejection.EQUITY_NON_POSITIVE: WaitReason.RISK_LIMIT,
}


def _measured_atr(features: FeatureVector) -> float | None:
    """ATR as a MEASUREMENT, or None.

    None for an absent key, a key graded MISSING (the not-ready path writes a
    0.0 placeholder at that grade), a non-finite value, or a non-positive one.
    Mirrors `signals/gates.py`'s `_measured`, which the gates use for the same
    reason: reading a placeholder at MISSING would read a value nobody
    measured.
    """
    if "atr" not in features.values:
        return None
    if features.quality_of("atr") is DataQuality.MISSING:
        return None
    value = float(features.values["atr"])
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


class EngineError(RuntimeError):
    """Raised when the engine's declared contract is violated.

    A view for the wrong symbol, a configured `total_points` the `Signal`
    contract cannot carry, a stage reached out of order. Missing or declining
    DATA never raises -- it produces a WAIT with a reason.
    """


# ---------------------------------------------------------------------------
# the feature layer the engine runs
# ---------------------------------------------------------------------------


def feature_computers(
    config: FlowModelConfig,
    spec: InstrumentSpec,
    calendar: SessionCalendarProtocol | None = None,
) -> tuple[FeatureComputer, ...]:
    """The seven computers whose keys the five component scorers read.

    `RegimeDetector` is deliberately NOT among them. It is a
    `FeatureComputer` so that the lookahead audit can see it, but four of its
    diagnostic keys (`realized_vol`, `vol_of_vol`, `vol_pct`,
    `efficiency`-adjacent) duplicate `VolatilityFeatures`' and
    `MomentumFeatures`' by construction, and `FeatureBundle` refuses
    colliding keys. The engine calls `RegimeDetector.classify(view)`
    separately, which is also how section 3 draws it: a stage of its own
    between the feature bundle and the gates.

    `calendar` is optional only so the engine can be constructed without one;
    `StructureFeatures` and `LevelFeatures` need a session calendar to anchor
    prior-session and opening-range levels, and without one their anchors
    degrade. The engine warns through `Signal.reasons`, not silently.
    """
    return (
        VolatilityFeatures(config.features),
        MomentumFeatures(config.features),
        LiquidityFeatures(config.features, spec),
        StructureFeatures(config.features, config.levels, spec, calendar),
        LevelFeatures(config.features, config.levels, spec, calendar),
        OrderFlowFeatures(config.order_flow, config.features),
        OptionsFlowFeatures(config.options_flow),
    )


# ---------------------------------------------------------------------------
# the plan a directional signal carries
# ---------------------------------------------------------------------------


class EntryPlan(FrozenModel):
    """Entry, stop, target, size and R:R for a directional signal.

    `Signal` is a frozen Phase 1 contract and carries no price geometry
    beyond `reference_price` and `atr`, so widening it to hold a stop and a
    target is not this phase's business. The plan therefore travels beside
    the signal on `EngineDecision`, and `Signal.reasons` states the geometry
    in words so a bare signal is still self-describing.

    Two reward/risk numbers are carried on purpose:

    * `achievable_rr` -- 14.6's MEASUREMENT: the distance to the next
      opposing major zone over the stop distance. This is what selected the
      setup.
    * `planned_r_multiple` -- what the trade actually plans, after
      `risk/sizing.py` put entry, stop and target on the tick grid and
      rounded the target TOWARD entry so reward is never overstated. This is
      the "R" in "2R setup" and the number `TradeRecord.r_label_is_honest()`
      judges.

    They differ whenever 14.6's band floor sits below the setup's own
    `reward_risk` (see `signals/setups.py`). Both are recorded so the gap is
    auditable rather than reconciled away.
    """

    symbol: str
    signal_ts: datetime
    side: Side
    setup: SetupType
    entry_price: float = Field(gt=0.0)
    stop_price: float = Field(gt=0.0)
    target_price: float = Field(gt=0.0)
    size: float = Field(gt=0.0)
    risk_dollars: float = Field(gt=0.0)
    point_value: float = Field(gt=0.0)
    equity_at_signal: float = Field(gt=0.0)
    stop_distance: float = Field(gt=0.0)
    stop_ticks: float = Field(ge=0.0)
    planned_r_multiple: float = Field(gt=0.0)
    achievable_rr: float = Field(ge=0.0)
    structural_stop_price: float | None = None
    structural_target_price: float | None = None
    atr: float | None = None
    max_hold_bars: int = Field(gt=0)
    flow_score: float = Field(ge=0.0, le=100.0)
    regime: Regime
    reasons: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def plans_beyond_measured_target(self) -> bool:
        """True when the planned target lies past the measured opposing zone."""
        return self.planned_r_multiple > self.achievable_rr + 1e-9

    @model_validator(mode="after")
    def _check(self) -> "EntryPlan":
        if self.side is Side.LONG:
            if not (self.stop_price < self.entry_price < self.target_price):
                raise EngineError(
                    f"LONG plan geometry is wrong: stop {self.stop_price}, entry "
                    f"{self.entry_price}, target {self.target_price}"
                )
        elif not (self.target_price < self.entry_price < self.stop_price):
            raise EngineError(
                f"SHORT plan geometry is wrong: target {self.target_price}, entry "
                f"{self.entry_price}, stop {self.stop_price}"
            )
        return self

    def to_trade_intent(self) -> TradeIntent:
        """The Phase 1 `TradeIntent` for this plan.

        Offered rather than built by the engine: section 3 makes the intent
        the risk manager's output, and Phase 7 owns that step. Going through
        the contract means the R definition
        (`risk_dollars == size * |entry - stop| * point_value`) is checked by
        the validator that owns it instead of by a second copy here.
        """
        return TradeIntent(
            symbol=self.symbol,
            signal_ts=self.signal_ts,
            side=self.side,
            setup=self.setup,
            entry_price=self.entry_price,
            stop_price=self.stop_price,
            target_price=self.target_price,
            size=self.size,
            risk_dollars=self.risk_dollars,
            point_value=self.point_value,
            equity_at_signal=self.equity_at_signal,
            flow_score=self.flow_score,
            regime=self.regime,
            max_hold_bars=self.max_hold_bars,
            reasons=self.reasons,
        )


# ---------------------------------------------------------------------------
# stage inputs
# ---------------------------------------------------------------------------


class StageInputs:
    """What each pipeline stage needs, computed when the stage is reached.

    The engine short-circuits, so the feature bundle must not run for a bar
    rejected on session or data quality. Wrapping the inputs lets the
    sequence be written once and still be lazy, and lets a test drive the
    sequence from hand-built features without a `MarketView`.

    Memoized per instance, and an instance lives for exactly one evaluation,
    so nothing survives into the next bar.
    """

    symbol: str
    ts: datetime

    def quality(self) -> QualityReport | None:
        raise NotImplementedError

    def warmup(self) -> WarmupReport | None:
        raise NotImplementedError

    def features(self) -> FeatureVector:
        raise NotImplementedError

    def regime(self) -> RegimeState:
        raise NotImplementedError

    def reference_price(self) -> float | None:
        raise NotImplementedError


class ViewInputs(StageInputs):
    """Stage inputs computed from a `MarketView`, on demand."""

    __slots__ = (
        "symbol",
        "ts",
        "_view",
        "_engine",
        "_quality",
        "_warmup",
        "_features",
        "_regime",
        "_done",
    )

    def __init__(self, view: MarketView, engine: "SignalEngine") -> None:
        self.symbol = view.symbol
        self.ts = view.now
        self._view = view
        self._engine = engine
        self._done: set[str] = set()
        self._quality: QualityReport | None = None
        self._warmup: WarmupReport | None = None
        self._features: FeatureVector | None = None
        self._regime: RegimeState | None = None

    def quality(self) -> QualityReport | None:
        if "quality" not in self._done:
            grader = self._engine.grader
            self._quality = None if grader is None else grader.grade(self._view)
            self._done.add("quality")
        return self._quality

    def warmup(self) -> WarmupReport:
        if self._warmup is None:
            self._warmup = assess_warmup(self._engine.warmup_computers, self._view)
        return self._warmup

    def features(self) -> FeatureVector:
        if self._features is None:
            self._features = self._engine.bundle.compute(self._view)
        return self._features

    def regime(self) -> RegimeState:
        if self._regime is None:
            self._regime = self._engine.detector.classify(self._view)
        return self._regime

    def reference_price(self) -> float | None:
        return self._view.last_price()


class PrecomputedInputs(StageInputs):
    """Stage inputs supplied directly, for a caller that already has them.

    Used by tests to drive the gate sequence from a hand-built
    `FeatureVector`, and available to any caller that computed the feature
    layer itself. Supplying `quality=None` or `warmup=None` makes the
    corresponding gate pass with a reason saying it was not evaluated, which
    is how a caller says "I am not grading feeds here" without the gate
    silently defaulting either way.
    """

    __slots__ = (
        "symbol",
        "ts",
        "_quality",
        "_warmup",
        "_features",
        "_regime",
        "_reference_price",
    )

    def __init__(
        self,
        *,
        features: FeatureVector,
        regime: RegimeState,
        reference_price: float | None = None,
        quality: QualityReport | None = None,
        warmup: WarmupReport | None = None,
    ) -> None:
        self.symbol = features.symbol
        self.ts = features.ts
        self._features = features
        self._regime = regime
        self._reference_price = reference_price
        self._quality = quality
        self._warmup = warmup

    def quality(self) -> QualityReport | None:
        return self._quality

    def warmup(self) -> WarmupReport | None:
        return self._warmup

    def features(self) -> FeatureVector:
        return self._features

    def regime(self) -> RegimeState:
        return self._regime

    def reference_price(self) -> float | None:
        return self._reference_price


# ---------------------------------------------------------------------------
# the decision record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineDecision:
    """Everything the engine computed at one bar.

    `signal` is the contract output and the unit section 12's backtest/live
    parity test compares. The rest is the audit trail: which stages ran, what
    each decided, and the intermediate records a report or a rejection
    analysis wants. A stage that was never reached is absent from
    `gate_results` and its field is None -- not zero, and not a default,
    because "not evaluated" and "evaluated to nothing" are different facts.
    """

    signal: Signal
    gate_results: tuple[GateResult, ...] = ()
    plan: EntryPlan | None = None
    quality: QualityReport | None = None
    warmup: WarmupReport | None = None
    features: FeatureVector | None = None
    regime: RegimeState | None = None
    components: Mapping[Component, ComponentScore] = field(default_factory=dict)
    flow: FlowScoreOutcome | None = None
    selection: SetupSelection | None = None
    sizing: SizingDecision | None = None
    limits: LimitDecision | None = None
    side: Side | None = None

    @property
    def action(self) -> SignalAction:
        return self.signal.action

    @property
    def stages(self) -> tuple[GateStage, ...]:
        """The stages that ran, in the order they ran."""
        return tuple(result.stage for result in self.gate_results)

    @property
    def blocking(self) -> GateResult | None:
        """The first failing gate, or None when the engine emitted a trade."""
        return next((r for r in self.gate_results if not r.passed), None)


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


class SignalEngine:
    """Runs section 3's pipeline for one symbol and emits a `Signal`."""

    def __init__(
        self,
        config: FlowModelConfig,
        symbol: str,
        *,
        calendar: SessionCalendarProtocol | None = None,
        availability: AvailabilityReport | None = None,
        grader: QualityGrader | None = None,
        computers: Sequence[FeatureComputer] | None = None,
        scorers: Sequence[AnyComponentScorer] | None = None,
        gate_set: Mapping[GateStage, Gate] | None = None,
        min_liquidity_score: float | None = None,
    ) -> None:
        if not isinstance(config, FlowModelConfig):
            raise EngineError(
                f"SignalEngine takes a FlowModelConfig; got {type(config).__name__}"
            )
        self.config = config
        self.symbol = symbol
        self.spec = config.spec(symbol)
        self.calendar = calendar
        self.availability = availability
        if availability is not None and availability.symbol != symbol:
            raise EngineError(
                f"availability report is for {availability.symbol!r} but this "
                f"engine is for {symbol!r}; scoring one symbol against another's "
                "feed verdict would report the wrong gap"
            )
        if config.flow_score.total_points > 100.0 + 1e-9:
            raise EngineError(
                f"config.flow_score.total_points is {config.flow_score.total_points}, "
                "but Signal.flow_score is bounded [0, 100] by the Phase 1 contract. "
                "A score above 100 cannot be carried on a Signal, and truncating it "
                "would misreport the number every threshold is compared against."
            )
        self.grader = (
            grader
            if grader is not None
            else QualityGrader(config.data, config.flow_score)
        )
        self.computers = (
            tuple(computers)
            if computers is not None
            else feature_computers(config, self.spec, calendar)
        )
        self.bundle = FeatureBundle(self.computers)
        self.detector = RegimeDetector(config.regime, config.features)
        #: Everything whose window must be full before a signal is produced.
        #: The detector is not in the bundle -- four of its diagnostic keys
        #: collide with the volatility and momentum computers' by
        #: construction -- but its own warmup is as binding as any
        #: computer's, so the warmup gate must see it too.
        self.warmup_computers = (*self.computers, self.detector)
        self.scorers = tuple(scorers) if scorers is not None else all_scorers(config)
        self.gates = dict(
            gate_set
            if gate_set is not None
            else pipeline_gates(
                config,
                self.spec,
                calendar,
                min_liquidity_score=min_liquidity_score,
            )
        )

    # --- the public API -------------------------------------------------

    def evaluate(
        self,
        view: MarketView,
        *,
        equity: float | None = None,
        limits: RiskLimitManager | None = None,
    ) -> Signal:
        """The `Signal` at `view.now`. LONG, SHORT or WAIT, always with a reason.

        This is the method the backtester and paper trading both call, and
        the one whose output the parity test compares. It is a pure function
        of the config, the view and the two explicit arguments.
        """
        return self.decide(view, equity=equity, limits=limits).signal

    def decide(
        self,
        view: MarketView,
        *,
        equity: float | None = None,
        limits: RiskLimitManager | None = None,
    ) -> EngineDecision:
        """`evaluate`, plus every intermediate record, for reports and audits."""
        if view.symbol != self.symbol:
            raise EngineError(
                f"this engine is built for {self.symbol!r} and was handed a view "
                f"for {view.symbol!r}. The feature computers carry that symbol's "
                "InstrumentSpec, so scoring another symbol through them would use "
                "the wrong tick size, point value and session."
            )
        return self.decide_from(
            ViewInputs(view, self), equity=equity, limits=limits
        )

    def decide_from(
        self,
        inputs: StageInputs,
        *,
        equity: float | None = None,
        limits: RiskLimitManager | None = None,
    ) -> EngineDecision:
        """Run the stages over any `StageInputs`.

        `decide` wraps a `MarketView`; a caller that already holds the
        feature layer passes a `PrecomputedInputs`. One sequence, two
        front doors, so the backtest and a hand-built test exercise the same
        gate order.

        The per-symbol contract is enforced HERE as well as in `decide`,
        because this is the other public door and both lead to the same
        `InstrumentSpec`. Without the check, stage inputs for another symbol
        produced a `Signal` carrying that symbol priced with this engine's
        tick size and point value -- a plan whose risk_dollars was wrong by
        the ratio of the two contracts' point values, which is exactly the
        error `decide`'s message says the check exists to prevent.
        """
        if inputs.symbol != self.symbol:
            raise EngineError(
                f"this engine is built for {self.symbol!r} and was handed stage "
                f"inputs for {inputs.symbol!r}. The feature computers, the gates "
                "and the risk layer all carry that symbol's InstrumentSpec, so "
                "the trade would be sized with the wrong tick size and point "
                "value. A multi-symbol run holds one engine per symbol."
            )
        account_equity = (
            float(self.config.risk.starting_equity) if equity is None else float(equity)
        )
        state = _Run(self, inputs, account_equity, limits)
        return state.run()

    # --- helpers a stage needs ------------------------------------------

    def session_date(self, ts: datetime) -> date:
        """The session date for `ts`, from the calendar when one exists.

        A date is needed by `RiskLimitManager.check` for its daily and weekly
        loss limits. Without a calendar the UTC date is used, which is the
        honest fallback rather than a guessed exchange day, and the engine
        says so in `Signal.reasons` through the session gate.
        """
        if self.calendar is None:
            return ts.date()
        return self.calendar.session_date(ts, self.spec)

    def score(
        self, features: FeatureVector
    ) -> ScoredComponents:
        """The five `ComponentScore`s at this bar, with their own reasons."""
        return score_components(
            self.config, features, self.availability, scorers=self.scorers
        )

    def aggregate(self, scored: ScoredComponents) -> FlowScoreOutcome:
        """The Flow Score, or the refusal, from five scored components."""
        return aggregate(self.config, scored)


# ---------------------------------------------------------------------------
# one evaluation
# ---------------------------------------------------------------------------


class _Run:
    """One pass through the stages. Lives for one bar and is then discarded.

    A separate object rather than a long method so that the accumulating
    reason list and gate-result list are obviously local: there is no
    instance of this class reachable from the engine, so nothing it collects
    can leak into the next bar.
    """

    __slots__ = (
        "engine",
        "inputs",
        "equity",
        "limits",
        "results",
        "reasons",
        "_quality",
        "_warmup",
        "_features",
        "_regime",
        "_components",
        "_flow",
        "_selection",
        "_sizing",
        "_limit_decision",
        "_side",
        "_plan",
    )

    def __init__(
        self,
        engine: SignalEngine,
        inputs: StageInputs,
        equity: float,
        limits: RiskLimitManager | None,
    ) -> None:
        self.engine = engine
        self.inputs = inputs
        self.equity = equity
        self.limits = limits
        self.results: list[GateResult] = []
        self.reasons: list[str] = []
        self._quality: QualityReport | None = None
        self._warmup: WarmupReport | None = None
        self._features: FeatureVector | None = None
        self._regime: RegimeState | None = None
        self._components: dict[Component, ComponentScore] = {}
        self._flow: FlowScoreOutcome | None = None
        self._selection: SetupSelection | None = None
        self._sizing: SizingDecision | None = None
        self._limit_decision: LimitDecision | None = None
        self._side: Side | None = None
        self._plan: EntryPlan | None = None

    # --- sequencing -----------------------------------------------------

    def run(self) -> EngineDecision:
        for stage in GateStage:
            result = self._stage(stage)
            if result is None:  # pragma: no cover - every stage is implemented
                raise EngineError(f"stage {stage.value} has no implementation")
            self.results.append(result)
            self.reasons.extend(result.reasons)
            if not result.passed:
                return self._decision(self._wait(result))
            if result.side is not None:
                self._side = result.side
        return self._decision(self._trade())

    def _stage(self, stage: GateStage) -> GateResult | None:
        gates = self.engine.gates
        if stage is GateStage.SESSION:
            return gates[stage].check(self.inputs.ts)  # type: ignore[attr-defined]
        if stage is GateStage.MARKET_DATA:
            self._quality = self.inputs.quality()
            return gates[stage].check(self._quality)  # type: ignore[attr-defined]
        if stage is GateStage.WARMUP:
            self._warmup = self.inputs.warmup()
            if self._warmup is None:
                return GateResult(
                    stage=stage,
                    passed=True,
                    reasons=(
                        "warmup: not evaluated -- the caller supplied the feature "
                        "vector and owns its warmup",
                    ),
                )
            return gates[stage].check(self._warmup)  # type: ignore[attr-defined]
        if stage is GateStage.REGIME:
            self._regime = self.inputs.regime()
            return gates[stage].check(self._regime)  # type: ignore[attr-defined]
        if stage is GateStage.LIQUIDITY:
            self._features = self.inputs.features()
            self.reasons.extend(
                f"features: {note}" for note in self._features.notes
            )
            return gates[stage].check(self._features)  # type: ignore[attr-defined]
        if stage is GateStage.STRUCTURE:
            return gates[stage].check(self._require_features())  # type: ignore[attr-defined]
        if stage is GateStage.OPTIONS_FLOW:
            self._score()
            return gates[stage].check(self._components, self._require_side())  # type: ignore[attr-defined]
        if stage is GateStage.ORDER_FLOW:
            return gates[stage].check(self._components, self._require_side())  # type: ignore[attr-defined]
        if stage is GateStage.MOMENTUM:
            return gates[stage].check(self._require_features())  # type: ignore[attr-defined]
        if stage is GateStage.FLOW_SCORE:
            # The components' own reasons were recorded when they were scored,
            # so the bundle handed to `aggregate` carries none and
            # `outcome.reasons` is exactly the aggregate's own commentary --
            # the availability line, and the redistribution warning when that
            # is on. Added here rather than left on the outcome, so a loud
            # warning cannot be lost between the aggregate and `Signal.reasons`.
            self._flow = self.engine.aggregate(
                ScoredComponents(
                    symbol=self.inputs.symbol,
                    ts=self.inputs.ts,
                    scores=self._components,
                    reasons=(),
                )
            )
            self.reasons.extend(self._flow.reasons)
            return gates[stage].check(self._flow)  # type: ignore[attr-defined]
        if stage is GateStage.ENTRY_FILTER:
            flow = self._require_flow()
            return gates[stage].check(flow, self._require_side())  # type: ignore[attr-defined]
        if stage is GateStage.SETUP:
            return self._setup_stage()
        if stage is GateStage.RISK:
            return self._risk_stage()
        return None

    # --- the two stages whose work lives in other modules ---------------

    def _setup_stage(self) -> GateResult:
        features = self._require_features()
        flow = self._require_flow()
        regime = self._regime
        assert regime is not None  # the regime stage ran and passed
        selection = setup_module.select(
            self.engine.config,
            features,
            side=self._require_side(),
            regime=regime,
            flow_score=flow.score,
            components=self._components,
        )
        self._selection = selection
        diagnostics = {
            "achievable_rr": selection.achievable_rr,
            "planned_reward_risk": selection.planned_reward_risk,
        }
        if not selection.matched:
            assert selection.wait_reason is not None
            return GateResult(
                stage=GateStage.SETUP,
                passed=False,
                wait_reason=selection.wait_reason,
                detail=selection.detail,
                reasons=(f"setup: WAIT -- {selection.detail}",),
                diagnostics=diagnostics,
            )
        return GateResult(
            stage=GateStage.SETUP,
            passed=True,
            reasons=selection.reasons,
            diagnostics=diagnostics,
        )

    def _risk_stage(self) -> GateResult:
        engine = self.engine
        selection = self._selection
        assert selection is not None and selection.setup_config is not None
        setup_config = selection.setup_config
        side = self._require_side()
        features = self._require_features()
        flow = self._require_flow()
        regime = self._regime
        assert regime is not None

        entry_price = self.inputs.reference_price()
        if entry_price is None or not math.isfinite(entry_price) or entry_price <= 0:
            return GateResult(
                stage=GateStage.RISK,
                passed=False,
                wait_reason=WaitReason.RISK_LIMIT,
                detail=(
                    "no reference price is available at this bar, so the entry "
                    "cannot be priced and nothing is sized"
                ),
                reasons=("risk: WAIT -- no reference price to enter at",),
            )
        stop_price = selection.structural_stop_price
        if stop_price is None:
            return GateResult(
                stage=GateStage.RISK,
                passed=False,
                wait_reason=WaitReason.RISK_LIMIT,
                detail=(
                    "features/levels.py supplied no level_stop_price, so 14.6's "
                    "determined stop does not exist and there is no R to define. "
                    "No ATR-multiple stop is substituted: the stop is where the "
                    "thesis is falsified, not a width chosen to make a size work"
                ),
                reasons=("risk: WAIT -- no structural stop, so R is undefined",),
            )

        atr_value = _measured_atr(features)
        if atr_value is None:
            # Section 0.4: "Missing data produces WAIT, never a guess." The
            # setup's `max_stop_atr_multiple` is the ONLY bound on how wide
            # 14.6's determined stop may be, and without an ATR it cannot be
            # evaluated. Sizing with `max_stop_distance=None` does not degrade
            # the cap, it REMOVES it: a 20x-ATR stop was accepted by a setup
            # that caps the stop at 2x, with nothing in the record saying the
            # check had not run. `MomentumGate` and `signals/setups.py` both
            # decline a bar whose `atr_percentile` was never measured "rather
            # than admitted on an unmeasured quantity"; this is the same
            # quantity and gets the same answer.
            return GateResult(
                stage=GateStage.RISK,
                passed=False,
                wait_reason=WaitReason.RISK_LIMIT,
                detail=(
                    "atr is absent or graded MISSING, so this setup's "
                    f"max_stop_atr_multiple ({setup_config.max_stop_atr_multiple:g}) "
                    "cannot be turned into a stop-width cap. The bar is declined "
                    "rather than sized with the only bound on the stop width "
                    "silently not applied"
                ),
                reasons=(
                    "risk: WAIT -- no ATR measurement, so the setup's stop-width "
                    "cap is unevaluable and the stop is not admitted uncapped",
                ),
            )
        max_stop_distance = float(setup_config.max_stop_atr_multiple) * atr_value
        sizing = size_position(
            spec=engine.spec,
            side=side,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            reward_risk=float(setup_config.reward_risk),
            equity=self.equity,
            risk_pct=float(engine.config.risk.risk_per_trade_pct),
            max_risk_pct=float(engine.config.risk.max_risk_per_trade_pct),
            min_stop_ticks=int(setup_config.min_stop_ticks),
            max_stop_distance=max_stop_distance,
            min_reward_risk=float(setup_config.min_reward_risk),
            allow_fractional=bool(engine.spec.allow_fractional_size),
        )
        self._sizing = sizing
        diagnostics = {
            "stop_distance": sizing.stop_distance,
            "stop_ticks": sizing.stop_ticks,
            "planned_r_multiple": sizing.planned_r_multiple,
            "size": sizing.size,
            "risk_dollars": sizing.risk_dollars,
            "equity": self.equity,
        }
        if atr_value is not None:
            diagnostics["atr"] = atr_value
        if not sizing.accepted:
            reason = SIZING_WAIT_REASONS.get(sizing.reason, WaitReason.RISK_LIMIT)
            notes = "; ".join(sizing.notes) or "no further detail"
            return GateResult(
                stage=GateStage.RISK,
                passed=False,
                wait_reason=reason,
                detail=f"position sizing refused the trade ({sizing.reason}): {notes}",
                reasons=(
                    f"risk: WAIT -- sizing refused ({sizing.reason}). {notes}",
                ),
                diagnostics=diagnostics,
            )

        if self.limits is not None:
            decision = self.limits.check(
                ts=self.inputs.ts,
                session_date=engine.session_date(self.inputs.ts),
                equity=self.equity,
                symbol=self.inputs.symbol,
                prospective_risk=sizing.risk_dollars,
            )
            self._limit_decision = decision
            if not decision.allowed:
                return GateResult(
                    stage=GateStage.RISK,
                    passed=False,
                    wait_reason=decision.wait_reason or WaitReason.RISK_LIMIT,
                    detail=f"risk limit {decision.breach}: {decision.detail}",
                    reasons=(
                        f"risk: WAIT -- limit {decision.breach} "
                        f"({decision.detail})"
                        + (
                            ". The account is HALTED, not merely paused."
                            if decision.terminal
                            else ""
                        ),
                    ),
                    diagnostics=diagnostics,
                )

        plan_reasons = tuple(
            f"risk: {note}" for note in sizing.notes
        ) + (
            f"risk: {side.value} {sizing.size:g} at {sizing.entry_price:g}, stop "
            f"{sizing.stop_price:g}, target {sizing.target_price:g} -- "
            f"{sizing.planned_r_multiple:.3f}R planned, "
            f"{sizing.risk_dollars:.2f} at risk "
            f"({sizing.risk_pct_of_equity * 100:.3f}% of equity)",
        )
        if self.limits is None:
            plan_reasons = plan_reasons + (
                "risk: no RiskLimitManager was supplied, so section 8's six "
                "limits were not checked on this bar",
            )
        self._plan = EntryPlan(
            symbol=self.inputs.symbol,
            signal_ts=self.inputs.ts,
            side=side,
            setup=setup_config.setup,
            entry_price=sizing.entry_price,
            stop_price=sizing.stop_price,
            target_price=sizing.target_price,
            size=sizing.size,
            risk_dollars=sizing.risk_dollars,
            point_value=engine.spec.point_value,
            equity_at_signal=self.equity,
            stop_distance=sizing.stop_distance,
            stop_ticks=sizing.stop_ticks,
            planned_r_multiple=sizing.planned_r_multiple,
            achievable_rr=selection.achievable_rr,
            structural_stop_price=selection.structural_stop_price,
            structural_target_price=selection.structural_target_price,
            atr=atr_value,
            max_hold_bars=int(setup_config.max_hold_bars),
            flow_score=flow.score,
            regime=regime.regime,
            reasons=tuple(self.reasons) + plan_reasons,
        )
        return GateResult(
            stage=GateStage.RISK,
            passed=True,
            reasons=plan_reasons,
            diagnostics=diagnostics,
        )

    # --- scoring --------------------------------------------------------

    def _score(self) -> None:
        if self._components:
            return
        scored = self.engine.score(self._require_features())
        self._components = dict(scored.scores)
        self.reasons.extend(scored.reasons)

    # --- signal construction -------------------------------------------

    def _wait(self, result: GateResult) -> Signal:
        assert result.wait_reason is not None
        flow = self._flow
        return Signal(
            symbol=self.inputs.symbol,
            ts=self.inputs.ts,
            action=SignalAction.WAIT,
            setup=self._selection.setup if self._selection is not None else None,
            flow_score=None if flow is None else flow.score,
            confidence=self._confidence(),
            regime=self._regime.regime if self._regime is not None else Regime.UNKNOWN,
            data_quality=self._data_quality(),
            wait_reason=result.wait_reason,
            wait_detail=result.detail,
            reasons=tuple(self.reasons),
            component_points=self._component_points(),
            reference_price=self._reference_price(),
            atr=self._atr(),
        )

    def _trade(self) -> Signal:
        plan = self._plan
        assert plan is not None
        flow = self._require_flow()
        regime = self._regime
        assert regime is not None
        action = (
            SignalAction.LONG if plan.side is Side.LONG else SignalAction.SHORT
        )
        return Signal(
            symbol=self.inputs.symbol,
            ts=self.inputs.ts,
            action=action,
            setup=plan.setup,
            flow_score=flow.score,
            confidence=self._confidence(),
            regime=regime.regime,
            data_quality=self._data_quality(),
            reasons=tuple(self.reasons),
            component_points=self._component_points(),
            reference_price=plan.entry_price,
            atr=plan.atr,
        )

    def _decision(self, signal: Signal) -> EngineDecision:
        return EngineDecision(
            signal=signal,
            gate_results=tuple(self.results),
            plan=self._plan,
            quality=self._quality,
            warmup=self._warmup,
            features=self._features,
            regime=self._regime,
            components=dict(self._components),
            flow=self._flow,
            selection=self._selection,
            sizing=self._sizing,
            limits=self._limit_decision,
            side=self._side,
        )

    # --- derived signal fields -----------------------------------------

    def _confidence(self) -> float:
        """The Flow Score as a fraction of the points that were measured.

        **Not a probability of anything.** It is `score / available_points`,
        which is the only normalization that does not change meaning when a
        component is unavailable: dividing by `max_points` would report a
        complete 55-point score as 0.55 confident and an incomplete one the
        same way, conflating "weak evidence" with "missing evidence".

        0.0 when no score exists, which is every WAIT before the Flow Score
        stage. The regime's own confidence is deliberately NOT folded in:
        multiplying two unrelated numbers would produce a third that measures
        neither.
        """
        flow = self._flow
        if flow is None or flow.flow_score is None or flow.available_points <= 0:
            return 0.0
        return max(0.0, min(1.0, flow.flow_score.score / flow.available_points))

    def _data_quality(self) -> DataQuality:
        """The signal's data-quality status: the worst grade in evidence.

        The feed-level grade from `data/quality.py` and the component-level
        grade from the Flow Score, whichever is worse. The merged
        `FeatureVector.quality` is deliberately NOT used: it is the worst
        grade over all 96 keys of every computer, including ones whose feed
        is absent, so on bars-only data it is MISSING for reasons that have
        nothing to do with the components a signal was built from.
        `FlowScore.quality` is the worst over ENABLED components' required
        keys, which is the grade of the number the decision actually used.
        """
        grades: list[DataQuality] = []
        if self._quality is not None:
            grades.append(self._quality.overall)
        flow = self._flow
        if flow is not None and flow.flow_score is not None:
            grades.append(flow.flow_score.quality)
        if not grades:
            return DataQuality.MISSING
        return DataQuality.worst(*grades)

    def _component_points(self) -> dict[Component, float]:
        """The points that produced `Signal.flow_score`, so the record adds up.

        The aggregate may hand back components that are not the ones it was
        given: with `redistribute_disabled_weight` on, the enabled components'
        weights are scaled up to `total_points`, so their points are larger
        than the scorers' own. Reporting the scorers' points beside the
        redistributed score put two numbers on one record that did not add up
        -- 22.31 of component points beside a reported 40.57 -- and a record
        whose parts do not sum to its total cannot audit the total.

        So the aggregate's components win whenever a Flow Score exists. On
        every other path -- including both non-redistributing ones -- the
        aggregate passes the same `ComponentScore` objects through unchanged,
        so this is identical to reading them here. When the aggregate REFUSED
        there is no score to be consistent with, and the scorers' own points
        are the right thing to report: they are what was measured, and the
        shortfall is what the refusal is about.
        """
        flow = self._flow
        if flow is not None and flow.flow_score is not None:
            return {
                component: score.points
                for component, score in flow.flow_score.components.items()
            }
        return {
            component: score.points for component, score in self._components.items()
        }

    def _reference_price(self) -> float | None:
        if self._plan is not None:
            return self._plan.entry_price
        price = self.inputs.reference_price()
        if price is None or not math.isfinite(price):
            return None
        return float(price)

    def _atr(self) -> float | None:
        """ATR at this bar, or None when it was not measured.

        The grade is consulted, not just the value: `FeatureComputer`'s
        not-ready path writes a 0.0 placeholder alongside a MISSING grade, and
        reporting that as `atr=0.0` would read downstream as a measurement of
        zero volatility. None is "not evaluated", which is the distinction
        `EngineDecision`'s docstring insists on everywhere else.
        """
        if self._features is None:
            return None
        return _measured_atr(self._features)

    # --- invariants -----------------------------------------------------

    def _require_features(self) -> FeatureVector:
        if self._features is None:  # pragma: no cover - the liquidity stage sets it
            raise EngineError(
                "a stage asked for features before the stage that computes them; "
                "the pipeline order in GateStage is what guarantees they exist"
            )
        return self._features

    def _require_side(self) -> Side:
        if self._side is None:  # pragma: no cover - the structure stage sets it
            raise EngineError(
                "a stage asked for the trade side before the structure gate "
                "established it. The side comes from level_direction and from "
                "nowhere else -- never from FlowScore.net_direction()."
            )
        return self._side

    def _require_flow(self) -> FlowScore:
        flow = self._flow
        if flow is None or flow.flow_score is None:  # pragma: no cover
            raise EngineError(
                "a stage asked for the Flow Score before the Flow Score stage "
                "produced one, or after it refused"
            )
        return flow.flow_score
