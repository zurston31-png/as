"""Adversarial tests for the Flow Score aggregate, the gate sequence and setups.

`signals/flow_score.py`, `signals/gates.py` and `signals/setups.py` are the three
modules where a number that looks like a Flow Score, a reason that looks like a
rejection and a label that looks like a setup can all be wrong without anything
downstream noticing. Nothing above them checks their arithmetic: Phase 7 reads
`Signal.flow_score` as a float in [0, 100], `Signal.wait_reason` as a histogram
bucket and `SetupType` as a trade archetype, and all three are plausible whatever
the terms underneath did. Six things can be silently wrong, and the tests are
organized around them.

**The sum (section 7).** `FlowScore = c_o + c_f + c_s + c_l + c_v in [0, 100]`.
A weighted sum hides a dead term: if one component's points never reached the
total, the score would still move with the other four and still look like a
score. `test_the_score_is_the_hand_computed_sum_of_all_five_components` pins all
five at once on magnitudes chosen so the arithmetic is exact in binary
(`0.5 * 20 = 10.00`, `0.75 * 25 = 18.75`, `0.25 * 20 = 5.00`, `1.0 * 15 = 15.00`,
`0.125 * 20 = 2.50`, summing to `51.25`), so the expected value is a number
written here rather than one read back from the implementation. The premise --
that each component's own `points` is the hand number -- is asserted before the
total is.

**The scale the sum is on.** Weights "live in `config.flow_score.weights` and
must sum to 100", and the `[0, total]` bound rests entirely on that. Three tests
cover it: the config refuses weights that do not sum to the total, a scorer that
returns its own weight raises rather than rescaling the whole score invisibly,
and -- the case the module marked unreachable -- weights that clear
`FlowScoreConfig`'s 1e-6 tolerance but not the aggregate's 1e-9 one still raise.

**The refusal is the point of the module.** Under
`strict_component_availability` a missing component must produce a refusal, not
a partial sum, and the refusal has to stay distinguishable from a genuine score.
`test_a_refusal_is_not_a_genuine_score_of_fifty_five` builds both on the same
weights: a complete five-component bar whose hand-computed score is exactly
55.0, and a bars-only bar where 45 of the 100 points have no feed. The first has
`score == 55.0` with `available_points == 100.0`; the second has `score is None`
with `available_points == 55.0`. No single scalar reading can confuse them, which
is the property that matters, because a 55-point scale and a 100-point scale are
different units wearing one name.

**The gate order (section 3).** `Signal.wait_reason` on a rejected bar must be
the FIRST stage that failed, and a test that only breaks one thing asserts
nothing about order. `test_the_first_failing_gate_is_reported_as_the_scene_is_peeled`
therefore starts from a bar that fails THIRTEEN stages at once and repairs
exactly one blocker at a time, asserting the reported stage and reason after
each repair: session -> market data -> warmup -> regime -> liquidity ->
structure -> options flow -> order flow -> momentum -> flow score (refused) ->
flow score (below threshold) -> entry filter -> setup -> risk -> a trade. Every
member of `GateStage` appears as the first failure exactly once, in the enum's
own order, so no stage is dead and none is merely first.

**Setup selection is a measurement (14.6).** The band is read off
`achievable_rr` and the setup's own `reward_risk` is what gets planned. The four
bands are walked over one fixed geometry, including both edges of every band,
and the two failure modes the brief names get their own tests: a 1.79R geometry
plans 1.0R (not promoted) and a 9.0R geometry plans 3.0R (not truncated), and a
geometry whose band names a disabled setup is DECLINED rather than handed to the
setup above or below it.

**Direction never touches magnitude (section 7).** A strongly bearish component
contributes its full magnitude and the gates refuse the trade.
`test_one_bearish_component_keeps_the_aggregate_high_and_blocks_both_sides` holds
four bullish components and one bearish one, asserts the aggregate stays at 95
of 100 points, and asserts that NEITHER side clears the entry filter -- the long
because the bearish component holds 19 points against it, the short because the
other four hold 76. `FlowScore.net_direction()` is `+1` on that aggregate and no
trade is taken, which is the "quietly takes the majority direction" bug the
section warns about.

Two findings this file REPORTS rather than repairs, each at its test:

* `test_three_of_the_entry_filter_s_five_declared_reasons_are_unreachable`.
  `WAIT_REASONS_BY_STAGE[ENTRY_FILTER]` declares five reasons. `ORDER_FLOW`
  cannot be the opposer, because the order-flow gate at stage 8 already refused
  ANY opposing direction; `LIQUIDITY` cannot, because `LiquidityScorer` declares
  no direction keys; `STRUCTURE` cannot, because `StructureScorer` reads its
  direction from `level_direction`, the same key the structure gate reads the
  side from. `gates.py` documents only the `LIQUIDITY` case.
* `test_the_aggregate_s_weight_sum_check_is_reachable_despite_its_pragma`. The
  branch is marked `# pragma: no cover - FlowScoreConfig validates this`, but
  the two checks use tolerances three orders of magnitude apart.

Builders are local on purpose, and none of them reads
`flow_model/config/defaults.yaml`: every weight, threshold and band edge under
test is written out here, so an unrelated edit to the shipped YAML cannot change
an expected value silently. The two numbers the shipped config does own and
these tests do assert against -- `max_opposing_points` 12.0 and
`options_contradiction_points` 14.0 -- are declared as module constants below
and used by name.

**The scorers are fixed by the test.** Section 7's separation is enforced in
`signals/scoring_bars.py` and `signals/scoring_flow.py`, which have their own
files. Here the five `ComponentScore`s are handed in directly through
`SignalEngine(scorers=...)`, so a magnitude is a number chosen here and the
aggregate's total is arithmetic rather than a measurement. The real gates, the
real aggregate, the real selector and the real risk layer run unchanged.
"""

from __future__ import annotations

import ast
import math
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from flow_model.config.schema import (
    BacktestConfig,
    FlowModelConfig,
    FlowScoreConfig,
    RiskConfig,
    SessionFilterConfig,
    SetupConfig,
    StructureLevelConfig,
)
from flow_model.core.contracts import ComponentScore, FeatureVector, FlowScore, RegimeState
from flow_model.core.enums import (
    Component,
    DataQuality,
    Feed,
    InstrumentType,
    Regime,
    SetupType,
    Side,
    SignalAction,
    WaitReason,
)
from flow_model.core.instruments import InstrumentSpec
from flow_model.data.base import FeedStatus
from flow_model.data.quality import QualityReport
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
    SETUP_RR_BANDS,
    setup_class_for,
)
from flow_model.signals.engine import PrecomputedInputs, SignalEngine
from flow_model.signals.flow_score import (
    REFUSAL_WAIT_REASON,
    FlowScoreError,
    FlowScoreOutcome,
    FlowScoreRefusal,
    ScoredComponents,
    aggregate,
    summary_lines,
)
from flow_model.signals.gates import (
    OPPOSITION_WAIT_REASON,
    WAIT_REASONS_BY_STAGE,
    ComputerReadiness,
    EntryFilterGate,
    GateResult,
    GateStage,
    LiquidityGate,
    OptionsFlowGate,
    OrderFlowGate,
    StructureGate,
    WarmupReport,
    assess_warmup,
)
from flow_model.signals.scoring_bars import LiquidityScorer, StructureScorer, VolMomentumScorer
from flow_model.signals.setups import (
    SETUP_BY_RR_CLASS,
    SetupError,
    min_flow_score_required,
    min_reward_risk_floor,
    select,
    tradable_regimes,
    vol_percentile_band,
)

SYMBOL = "NQ"
TS = datetime(2021, 6, 15, 15, 30, tzinfo=timezone.utc)

#: The entry price every scene in this file prices its geometry from. Round so
#: the stop distance, the R multiple and the position size are all exact.
REFERENCE_PRICE = 100.00

#: The weights under test, written out here rather than loaded. 20 + 25 + 20 +
#: 15 + 20 = 100, which is section 7's "must sum to 100".
WEIGHTS: dict[Component, float] = {
    Component.OPTIONS_FLOW: 20.0,
    Component.ORDER_FLOW: 25.0,
    Component.STRUCTURE: 20.0,
    Component.LIQUIDITY: 15.0,
    Component.VOL_MOMENTUM: 20.0,
}
TOTAL_POINTS = 100.0

#: The two contradiction limits the tests compare against, declared locally so a
#: change to the shipped YAML cannot move an expected value here.
MAX_OPPOSING_POINTS = 12.0
OPTIONS_CONTRADICTION_POINTS = 14.0

#: Per-setup thresholds for the local config. The score floors are deliberately
#: NOT the shipped 70/75/80: these tests ask what the gates do at a threshold,
#: not what the shipped threshold is, and 30/40/80 keeps the union floor (30),
#: a mid setup (40) and a floor no test score reaches (80) all distinguishable.
SETUP_REWARD_RISK = {SetupType.SCALP_1R: 1.0, SetupType.SETUP_2R: 2.0, SetupType.DIRECTIONAL_3R: 3.0}
SETUP_MIN_FLOW_SCORE = {SetupType.SCALP_1R: 30.0, SetupType.SETUP_2R: 40.0, SetupType.DIRECTIONAL_3R: 80.0}
SETUP_MIN_REWARD_RISK = {SetupType.SCALP_1R: 1.0, SetupType.SETUP_2R: 1.8, SetupType.DIRECTIONAL_3R: 2.7}
#: DIRECTIONAL_3R's band stops at 0.50 while the other two reach 0.80, so the
#: union band is [0.20, 0.80] and a 0.70 percentile clears the union while
#: failing the setup the structure determined. That is the one asymmetry the
#: pre-selection gates' "weaker question" deviation needs to be testable.
SETUP_VOL_BAND = {
    SetupType.SCALP_1R: (0.20, 0.80),
    SetupType.SETUP_2R: (0.20, 0.80),
    SetupType.DIRECTIONAL_3R: (0.20, 0.50),
}
TRADABLE_REGIMES = (Regime.TRENDING_UP, Regime.TRENDING_DOWN)

#: Reached by no enabled setup, so it is the regime gate's trigger.
BLOCKED_REGIME = Regime.CHOP


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def spec(symbol: str = SYMBOL) -> InstrumentSpec:
    """A future whose point value is a round 20.0 (tick 0.25 worth $5)."""
    return InstrumentSpec(
        symbol=symbol,
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        typical_spread_ticks=2.0,
        min_spread_ticks=1.0,
    )


def flow_score_config(**overrides) -> FlowScoreConfig:
    base = dict(
        weights=dict(WEIGHTS),
        total_points=TOTAL_POINTS,
        strict_component_availability=True,
        redistribute_disabled_weight=False,
        max_opposing_points=MAX_OPPOSING_POINTS,
        options_contradiction_points=OPTIONS_CONTRADICTION_POINTS,
    )
    base.update(overrides)
    return FlowScoreConfig(**base)


def setup_config(setup: SetupType, **overrides) -> SetupConfig:
    low, high = SETUP_VOL_BAND[setup]
    base = dict(
        setup=setup,
        enabled=True,
        reward_risk=SETUP_REWARD_RISK[setup],
        min_flow_score=SETUP_MIN_FLOW_SCORE[setup],
        allowed_regimes=TRADABLE_REGIMES,
        min_vol_percentile=low,
        max_vol_percentile=high,
        min_reward_risk=SETUP_MIN_REWARD_RISK[setup],
        min_stop_ticks=4,
        max_stop_atr_multiple=3.0,
        max_hold_bars=12,
        require_orderflow_confirmation=True,
        require_structure_confirmation=True,
    )
    base.update(overrides)
    return SetupConfig(**base)


def setup_table(**per_setup) -> dict[SetupType, SetupConfig]:
    """All three setups, with per-setup keyword overrides by enum name."""
    return {
        setup: setup_config(setup, **per_setup.get(setup.value, {}))
        for setup in SetupType
    }


def config(**overrides) -> FlowModelConfig:
    """A root config built here, not loaded from defaults.yaml."""
    base = dict(
        name="test_flow_score",
        instruments={SYMBOL: spec()},
        backtest=BacktestConfig(symbols=(SYMBOL,)),
        flow_score=flow_score_config(),
        levels=StructureLevelConfig(min_reward_risk=1.0),
        setups=setup_table(),
        session=SessionFilterConfig(),
        risk=RiskConfig(
            starting_equity=100_000.0,
            risk_per_trade_pct=0.005,
            max_risk_per_trade_pct=0.01,
        ),
    )
    base.update(overrides)
    return FlowModelConfig(**base)


def component(
    which: Component,
    magnitude: float,
    direction: int = 0,
    *,
    enabled: bool = True,
    weight: float | None = None,
    quality: DataQuality = DataQuality.GOOD,
) -> ComponentScore:
    """One component, with its CONFIGURED weight unless a test overrides it."""
    return ComponentScore(
        component=which,
        magnitude=magnitude,
        direction=direction,
        weight=WEIGHTS[which] if weight is None else weight,
        enabled=enabled,
        quality=quality,
    )


def unavailable(which: Component) -> ComponentScore:
    """A component with no measurement, as the scorers emit one.

    Magnitude 0, direction 0, quality MISSING, `enabled=False` -- and its
    CONFIGURED weight, which is what makes `max_points - available_points`
    report the shortfall instead of hiding it.
    """
    return component(
        which, 0.0, 0, enabled=False, quality=DataQuality.MISSING
    )


def components(**by_name: ComponentScore) -> dict[Component, ComponentScore]:
    """The five components keyed by enum, from keyword arguments by value."""
    table = {Component(name): score for name, score in by_name.items()}
    missing = [c.value for c in Component if c not in table]
    assert not missing, f"the test must supply every component; missing {missing}"
    return {c: table[c] for c in Component}


def scored(table: dict[Component, ComponentScore], **overrides) -> ScoredComponents:
    base = dict(symbol=SYMBOL, ts=TS, scores=table)
    base.update(overrides)
    return ScoredComponents(**base)


def all_bullish(magnitude: float = 0.5) -> dict[Component, ComponentScore]:
    """Five measured components agreeing on a LONG, at one magnitude.

    LIQUIDITY is direction 0 because `LiquidityScorer` declares no direction
    keys; this builder does not invent one for it.
    """
    return components(
        options_flow=component(Component.OPTIONS_FLOW, magnitude, 1),
        order_flow=component(Component.ORDER_FLOW, magnitude, 1),
        structure=component(Component.STRUCTURE, magnitude, 1),
        liquidity=component(Component.LIQUIDITY, magnitude, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, magnitude, 1),
    )


def features(
    *,
    symbol: str = SYMBOL,
    ts: datetime = TS,
    liquidity_score: float = 0.90,
    spread_ticks: float = 2.0,
    relative_volume: float = 1.20,
    structure_gate_passed: float = 1.0,
    level_direction: float = DIRECTION_SUPPORT,
    level_gate_reason: float = GATE_PASSED,
    level_significance: float = 0.80,
    level_cleanliness: float = 0.70,
    level_rejection: float = 0.65,
    achievable_rr: float = 1.50,
    level_setup_class: float | None = None,
    level_stop_price: float = 99.00,
    level_target_price: float | None = None,
    atr_percentile: float = 0.50,
    atr: float = 1.00,
    structure_min_reward_risk: float = 1.0,
    drop: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
) -> FeatureVector:
    """The keys the gates, the selector and the risk stage read, and no others.

    `level_setup_class` defaults to `levels.setup_class_for(achievable_rr,
    structure_min_reward_risk)` -- the same pure function `features/levels.py`
    calls -- so the builder cannot accidentally hand `select` an inconsistent
    pair. `test_a_setup_class_that_disagrees_with_the_measurement_raises`
    fabricates one deliberately.

    `level_target_price` defaults to the price 14.6's own definition puts it at
    for a LONG from `REFERENCE_PRICE`:

        achievable_rr = |target - entry| / |entry - stop|
        => target = entry + achievable_rr * (entry - stop)

    so every geometry in this file is internally consistent and the target is a
    number a reader can check rather than a constant that happens to sit there.
    A SHORT scene passes it explicitly.
    """
    if level_target_price is None:
        level_target_price = REFERENCE_PRICE + achievable_rr * (
            REFERENCE_PRICE - level_stop_price
        )
    values = {
        "liquidity_score": liquidity_score,
        "spread_ticks": spread_ticks,
        "relative_volume": relative_volume,
        "structure_gate_passed": structure_gate_passed,
        "level_direction": level_direction,
        "level_gate_reason": level_gate_reason,
        "level_significance": level_significance,
        "level_cleanliness": level_cleanliness,
        "level_rejection": level_rejection,
        "achievable_rr": achievable_rr,
        "level_setup_class": (
            setup_class_for(achievable_rr, structure_min_reward_risk)
            if level_setup_class is None
            else level_setup_class
        ),
        "level_stop_price": level_stop_price,
        "level_target_price": level_target_price,
        "atr_percentile": atr_percentile,
        "atr": atr,
    }
    for key in drop:
        values.pop(key)
    return FeatureVector(
        symbol=symbol,
        ts=ts,
        values=values,
        quality_by_key={
            key: (DataQuality.MISSING if key in missing else DataQuality.GOOD)
            for key in values
        },
        warmup_complete=True,
    )


def regime_state(
    regime: Regime = Regime.TRENDING_UP,
    *,
    confidence: float = 0.80,
    bars_in_regime: int = 24,
) -> RegimeState:
    return RegimeState(
        symbol=SYMBOL,
        ts=TS,
        regime=regime,
        confidence=confidence,
        bars_in_regime=bars_in_regime,
    )


def quality_report(
    *, overall: DataQuality = DataQuality.GOOD, blocking: tuple[Feed, ...] = ()
) -> QualityReport:
    return QualityReport(
        symbol=SYMBOL,
        ts=TS,
        statuses={
            Feed.BARS: FeedStatus(feed=Feed.BARS, quality=overall, rows=600, coverage=1.0)
        },
        overall=overall,
        blocking_feeds=blocking,
        note="hand-built by the test",
    )


def warmup_report(*, bars_available: int = 600, bars_required: int = 514) -> WarmupReport:
    return WarmupReport(
        bars_available=bars_available,
        computers=(
            ComputerReadiness(
                name="levels",
                feeds_present=True,
                bars_required=bars_required,
                bars_available=bars_available,
            ),
        ),
    )


class FixedScorer:
    """Returns a `ComponentScore` the test wrote down.

    Satisfies the `score(features, verdict)` / `explain(features, verdict)`
    shape `signals/flow_score.score_components` reconciles, so the engine's
    real pipeline runs over components whose magnitudes are exact by
    construction.
    """

    def __init__(self, score: ComponentScore) -> None:
        self.component = score.component
        self._score = score

    def score(self, features: FeatureVector, verdict=None) -> ComponentScore:
        return self._score

    def explain(self, features: FeatureVector, verdict=None) -> tuple[str, ...]:
        return (f"{self.component.value}: magnitude fixed by the test",)


class ClosedCalendar:
    """A calendar that rejects every timestamp, with a reason string."""

    def __init__(self, reason: str = "not_rth") -> None:
        self.reason = reason

    def is_tradable(self, ts, instrument, session_cfg) -> tuple[bool, str]:
        return False, self.reason

    def session_date(self, ts, instrument) -> date:
        return ts.date()


def engine_for(cfg: FlowModelConfig, table, *, calendar=None) -> SignalEngine:
    """An engine whose five scorers are fixed and whose computers are unused.

    `computers=()` because `PrecomputedInputs` supplies the feature vector and
    the warmup report; the feature layer has its own eight test files.
    """
    return SignalEngine(
        cfg,
        SYMBOL,
        calendar=calendar,
        computers=(),
        scorers=tuple(FixedScorer(table[c]) for c in Component),
    )


def decide(
    cfg: FlowModelConfig,
    table,
    *,
    vector: FeatureVector | None = None,
    regime: Regime | RegimeState = Regime.TRENDING_UP,
    reference_price: float | None = 100.00,
    quality: QualityReport | None = None,
    warmup: WarmupReport | None = None,
    calendar=None,
    equity: float = 100_000.0,
):
    state = regime if isinstance(regime, RegimeState) else regime_state(regime)
    return engine_for(cfg, table, calendar=calendar).decide_from(
        PrecomputedInputs(
            features=vector if vector is not None else features(),
            regime=state,
            reference_price=reference_price,
            quality=quality,
            warmup=warmup,
        ),
        equity=equity,
    )


# ---------------------------------------------------------------------------
# section 7: the sum, and the scale it is on
# ---------------------------------------------------------------------------


def test_the_score_is_the_hand_computed_sum_of_all_five_components():
    """A weighted sum hides a dead term, so all five points are pinned at once.

    Magnitudes are chosen to be exact in binary floating point, so the total is
    a number written here:

        OPTIONS_FLOW  0.500 * 20 = 10.00
        ORDER_FLOW    0.750 * 25 = 18.75
        STRUCTURE     0.250 * 20 =  5.00
        LIQUIDITY     1.000 * 15 = 15.00
        VOL_MOMENTUM  0.125 * 20 =  2.50
                                   -----
                                   51.25
    """
    table = components(
        options_flow=component(Component.OPTIONS_FLOW, 0.500, 1),
        order_flow=component(Component.ORDER_FLOW, 0.750, 1),
        structure=component(Component.STRUCTURE, 0.250, 1),
        liquidity=component(Component.LIQUIDITY, 1.000, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, 0.125, 1),
    )
    # the premise, before anything rests on it
    assert table[Component.OPTIONS_FLOW].points == 10.00
    assert table[Component.ORDER_FLOW].points == 18.75
    assert table[Component.STRUCTURE].points == 5.00
    assert table[Component.LIQUIDITY].points == 15.00
    assert table[Component.VOL_MOMENTUM].points == 2.50

    outcome = aggregate(flow_score_config(), scored(table))

    assert not outcome.refused
    assert outcome.score == 51.25
    assert outcome.available_points == 100.0
    assert outcome.max_points == 100.0
    assert outcome.missing_points == 0.0
    flow = outcome.require()
    assert flow.score == pytest.approx(
        sum(c.points for c in flow.components.values()), abs=1e-9
    )


def test_a_full_magnitude_bar_reaches_the_configured_total_exactly():
    """The upper bound is the weights' sum, and it is attained, not approached.

    If the aggregate lost a component or applied a weight twice, full magnitude
    on all five would land somewhere other than 100.
    """
    outcome = aggregate(flow_score_config(), scored(all_bullish(1.0)))
    assert outcome.score == 100.0
    assert outcome.require().max_points == 100.0


def test_the_score_stays_inside_zero_and_the_total_across_a_magnitude_grid():
    """Section 7's `in [0, 100]`, over every combination of a coarse grid.

    Bounded by construction rather than by luck: `ComponentScore.magnitude` is
    `[0, 1]` and the weights sum to the total, so no combination can leave the
    range. A grid walk is what catches a sign error or a stray multiplication
    that a single hand-computed case would miss.
    """
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    cfg = flow_score_config()
    for m_o in grid:
        for m_f in grid:
            for m_v in grid:
                table = components(
                    options_flow=component(Component.OPTIONS_FLOW, m_o, 1),
                    order_flow=component(Component.ORDER_FLOW, m_f, -1),
                    structure=component(Component.STRUCTURE, 1.0, 1),
                    liquidity=component(Component.LIQUIDITY, 0.5, 0),
                    vol_momentum=component(Component.VOL_MOMENTUM, m_v, -1),
                )
                score = aggregate(cfg, scored(table)).score
                assert score is not None
                assert 0.0 <= score <= TOTAL_POINTS + 1e-9
                # the sum does not net the two bearish components away
                assert score == pytest.approx(
                    20 * m_o + 25 * m_f + 20 * 1.0 + 15 * 0.5 + 20 * m_v, abs=1e-9
                )


def test_a_magnitude_above_one_is_refused_by_the_contract():
    """The `[0, 100]` bound rests on `magnitude in [0, 1]`; that is enforced."""
    with pytest.raises(ValueError):
        ComponentScore(
            component=Component.STRUCTURE, magnitude=1.5, direction=1, weight=20.0
        )
    with pytest.raises(ValueError):
        ComponentScore(
            component=Component.STRUCTURE, magnitude=0.5, direction=2, weight=20.0
        )


def test_weights_that_do_not_sum_to_the_total_are_refused_by_the_config():
    """Section 7: weights "must sum to 100". The config is where that is held."""
    bad = dict(WEIGHTS)
    bad[Component.VOL_MOMENTUM] = 10.0  # 90, not 100
    with pytest.raises(ValueError, match="sum to"):
        FlowScoreConfig(weights=bad, total_points=100.0)


def test_a_scorer_may_not_choose_its_own_weight():
    """A scorer returning its own weight would rescale the whole score silently.

    STRUCTURE is handed weight 40 against a configured 20: a partial sum of
    `0.5*40 = 20` instead of `0.5*20 = 10` would read as a stronger bar with
    nothing saying the scale moved.
    """
    table = all_bullish(0.5)
    table[Component.STRUCTURE] = component(
        Component.STRUCTURE, 0.5, 1, weight=40.0
    )
    with pytest.raises(FlowScoreError, match="may not choose its own weight"):
        aggregate(flow_score_config(), scored(table))


def test_an_unavailable_component_may_not_have_its_weight_zeroed():
    """Zeroing the weight would make the shortfall vanish from the report.

    `max_points - available_points` is the whole content of "55.0 of 100.0". A
    scorer that reported an unmeasured component at weight 0 would leave
    `available_points == 100.0` and nothing to report.
    """
    table = all_bullish(0.5)
    table[Component.ORDER_FLOW] = component(
        Component.ORDER_FLOW, 0.0, 0, enabled=False, weight=0.0,
        quality=DataQuality.MISSING,
    )
    with pytest.raises(FlowScoreError, match="keeps its configured weight"):
        aggregate(flow_score_config(), scored(table))


def test_the_aggregate_s_weight_sum_check_is_reachable_despite_its_pragma():
    """REPORTED FINDING: the branch marked unreachable is reachable.

    `_check_weights`'s total check carries `# pragma: no cover - FlowScoreConfig
    validates this`, but the two checks use different tolerances:
    `FlowScoreConfig` allows `abs(total - total_points) <= 1e-6` and
    `WEIGHT_TOLERANCE` is 1e-9. Weights summing to 100.0000005 therefore
    construct and then raise in the aggregate. Severity is low -- the aggregate
    is the stricter of the two, which is the direction a safety check should
    err -- but the comment is wrong, and a `pragma: no cover` on a reachable
    branch hides whatever else walks through it.
    """
    drifted = dict(WEIGHTS)
    drifted[Component.VOL_MOMENTUM] = 20.0000005
    cfg = FlowScoreConfig(weights=drifted, total_points=100.0)  # 1e-6 tolerance: fine
    assert sum(cfg.weights.values()) != pytest.approx(100.0, abs=1e-9)

    table = all_bullish(0.5)
    table[Component.VOL_MOMENTUM] = component(
        Component.VOL_MOMENTUM, 0.5, 1, weight=20.0000005
    )
    with pytest.raises(FlowScoreError, match="not total_points"):
        aggregate(cfg, scored(table))


def test_a_total_below_one_hundred_bounds_the_score_at_that_total():
    """The bound is the configured total, not a hard-coded 100.

    Weights of 10/12/10/8/10 sum to 50, so a full-magnitude bar scores 50 and
    `FlowScore.max_points` is 50 -- which is what makes `FlowScore`'s own
    `score <= max_points` validator a check rather than a decoration.
    """
    half = {
        Component.OPTIONS_FLOW: 10.0,
        Component.ORDER_FLOW: 12.0,
        Component.STRUCTURE: 10.0,
        Component.LIQUIDITY: 8.0,
        Component.VOL_MOMENTUM: 10.0,
    }
    cfg = FlowScoreConfig(weights=half, total_points=50.0)
    table = {
        c: ComponentScore(component=c, magnitude=1.0, direction=1, weight=w)
        for c, w in half.items()
    }
    outcome = aggregate(cfg, scored(table))
    assert outcome.score == 50.0
    assert outcome.max_points == 50.0


def test_a_missing_component_is_a_caller_bug_rather_than_a_smaller_score():
    """Four components is not a degraded Flow Score; it is a missing scorer."""
    four = {c: s for c, s in all_bullish(0.5).items() if c is not Component.LIQUIDITY}
    with pytest.raises(FlowScoreError, match="no score for component"):
        ScoredComponents(symbol=SYMBOL, ts=TS, scores=four)


def test_a_score_filed_under_the_wrong_component_is_refused():
    """A mis-keyed score would apply one component's weight to another's magnitude."""
    table = all_bullish(0.5)
    table[Component.LIQUIDITY] = component(Component.VOL_MOMENTUM, 1.0, 1)
    with pytest.raises(FlowScoreError, match="reports component"):
        ScoredComponents(symbol=SYMBOL, ts=TS, scores=table)


def test_a_bare_mapping_needs_a_symbol_and_a_timestamp():
    """A score with no timestamp cannot be attributed to a bar."""
    with pytest.raises(FlowScoreError, match="needs symbol and ts"):
        aggregate(flow_score_config(), all_bullish(0.5))


# ---------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------


def bars_only_components(magnitude: float = 0.5) -> dict[Component, ComponentScore]:
    """The bars-only shape: 55 points measured, 45 with no feed.

    ORDER_FLOW (25) needs tick aggregates and OPTIONS_FLOW (20) needs an
    options snapshot; neither exists on a bars-only dataset, so both are
    UNAVAILABLE and STRUCTURE + LIQUIDITY + VOL_MOMENTUM = 55 points remain.
    """
    return components(
        options_flow=unavailable(Component.OPTIONS_FLOW),
        order_flow=unavailable(Component.ORDER_FLOW),
        structure=component(Component.STRUCTURE, magnitude, 1),
        liquidity=component(Component.LIQUIDITY, magnitude, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, magnitude, 1),
    )


def test_strict_availability_refuses_rather_than_summing_what_is_left():
    """The headline rule: a 55-point sum is not a Flow Score.

    STRUCTURE 20 + LIQUIDITY 15 + VOL_MOMENTUM 20 = 55 points are computable
    and 45 have no feed. A partial sum (0.5 * 55 = 27.5) compared against a
    threshold calibrated on 100 points is a different quantity wearing the same
    name, so the aggregate must refuse.
    """
    outcome = aggregate(flow_score_config(), scored(bars_only_components(0.5)))

    assert outcome.refused is True
    assert outcome.flow_score is None
    assert outcome.score is None
    assert outcome.available_points == 55.0
    assert outcome.max_points == 100.0
    assert outcome.missing_points == 45.0
    assert outcome.wait_reason is WaitReason.COMPONENT_DISABLED
    assert outcome.wait_reason is REFUSAL_WAIT_REASON
    assert outcome.unavailable_components == (
        Component.OPTIONS_FLOW,
        Component.ORDER_FLOW,
    )
    assert outcome.quality is DataQuality.MISSING
    # 27.5 is the number a partial sum would have produced; it appears nowhere
    assert "27.5" not in " ".join(outcome.reasons)


def test_a_refusal_is_not_a_genuine_score_of_fifty_five():
    """A refusal at 55 computable points and a real score of 55 must not collide.

    The complete bar below scores exactly 55.0 out of 100:

        OPTIONS_FLOW  0.50 * 20 = 10.0
        ORDER_FLOW    0.60 * 25 = 15.0
        STRUCTURE     0.50 * 20 = 10.0
        LIQUIDITY     1.00 * 15 = 15.0
        VOL_MOMENTUM  0.25 * 20 =  5.0
                                  ----
                                  55.0

    The bars-only bar has 55 points AVAILABLE and no score at all. Every
    accessor separates them: `score`, `refused`, `available_points`,
    `wait_reason` and `require()`.
    """
    genuine = aggregate(
        flow_score_config(),
        scored(
            components(
                options_flow=component(Component.OPTIONS_FLOW, 0.50, 1),
                order_flow=component(Component.ORDER_FLOW, 0.60, 1),
                structure=component(Component.STRUCTURE, 0.50, 1),
                liquidity=component(Component.LIQUIDITY, 1.00, 0),
                vol_momentum=component(Component.VOL_MOMENTUM, 0.25, 1),
            )
        ),
    )
    refusal = aggregate(flow_score_config(), scored(bars_only_components(0.5)))

    assert genuine.score == pytest.approx(55.0, abs=1e-9)
    assert genuine.available_points == 100.0
    assert genuine.refused is False
    assert genuine.wait_reason is None

    assert refusal.score is None
    assert refusal.available_points == 55.0
    assert refusal.refused is True
    assert refusal.wait_reason is WaitReason.COMPONENT_DISABLED

    # The trap, stated explicitly: the two 55.0s ARE the same number. What
    # keeps them apart is that they are different FIELDS -- one is a score out
    # of 100, the other is how much of the scale existed -- and the refusal has
    # no score at all, so a caller reading `score` cannot reach the wrong one.
    assert genuine.score == refusal.available_points
    assert refusal.score is None
    assert genuine.available_points != refusal.available_points
    with pytest.raises(FlowScoreError):
        refusal.require()
    assert genuine.require().score == pytest.approx(55.0, abs=1e-9)

    assert any("REFUSED" in line for line in summary_lines(refusal))
    assert not any("REFUSED" in line for line in summary_lines(genuine))
    assert any("55.0 of 100.0" in line for line in summary_lines(refusal))


def test_the_refusal_is_an_outcome_and_never_an_exception():
    """The engine has to record a refusal as a bar's WAIT, which an exception
    cannot be without the caller inventing the reason itself."""
    outcome = aggregate(flow_score_config(), scored(bars_only_components(0.9)))
    assert isinstance(outcome, FlowScoreOutcome)
    assert isinstance(outcome.refusal, FlowScoreRefusal)
    assert outcome.refusal.missing_points == 45.0
    assert outcome.refusal.available_points == 55.0


def test_an_outcome_carries_exactly_one_of_a_score_and_a_refusal():
    """Both set would let a caller read a score the aggregate refused."""
    flow = FlowScore(symbol=SYMBOL, ts=TS, components=all_bullish(0.5))
    refusal = FlowScoreRefusal(
        symbol=SYMBOL, ts=TS, available_points=55.0, max_points=100.0
    )
    with pytest.raises(FlowScoreError, match="exactly one"):
        FlowScoreOutcome(
            symbol=SYMBOL, ts=TS, flow_score=flow, refusal=refusal,
            available_points=55.0, max_points=100.0,
        )
    with pytest.raises(FlowScoreError, match="exactly one"):
        FlowScoreOutcome(
            symbol=SYMBOL, ts=TS, available_points=55.0, max_points=100.0
        )


def test_everything_unavailable_refuses_whatever_the_flags_say():
    """With no component measured there is no scale, and redistribution would
    divide by zero available weight."""
    nothing = {c: unavailable(c) for c in Component}
    cfg = flow_score_config(
        strict_component_availability=False, redistribute_disabled_weight=True
    )
    outcome = aggregate(cfg, scored(nothing))
    assert outcome.refused is True
    assert outcome.available_points == 0.0
    assert "no scale" in outcome.refusal.detail


def test_non_strict_without_redistribution_says_the_number_is_out_of_fifty_five():
    """Switching strict off must still report the scale it answered on.

    0.5 * (20 + 15 + 20) = 27.5, and the reason has to say that is out of 55
    and not out of 100, because the only thing that makes the number usable is
    knowing which denominator it belongs to.
    """
    cfg = flow_score_config(strict_component_availability=False)
    outcome = aggregate(cfg, scored(bars_only_components(0.5)))

    assert outcome.refused is False
    assert outcome.score == 27.5
    assert outcome.available_points == 55.0
    assert outcome.max_points == 100.0
    assert outcome.redistributed is False
    joined = " ".join(outcome.reasons)
    assert "NOT redistributed" in joined
    assert "out of 55.0" in joined and "not out of 100.0" in joined


def test_redistribution_inflates_the_measured_components_and_says_so():
    """The one path that changes the scale, implemented loudly rather than quietly.

    scale = 100 / 55 = 1.8181..., so the plain sum of 27.5 becomes
    27.5 * 100/55 = 50.0 exactly. That factor is the whole reason
    redistribution is off by default, and the outcome must state that the
    number is not comparable to one from a complete dataset.
    """
    cfg = flow_score_config(
        strict_component_availability=False, redistribute_disabled_weight=True
    )
    outcome = aggregate(cfg, scored(bars_only_components(0.5)))

    assert outcome.refused is False
    assert outcome.redistributed is True
    assert outcome.score == pytest.approx(50.0, abs=1e-6)
    assert outcome.available_points == pytest.approx(100.0, abs=1e-6)
    flow = outcome.require()
    structure = flow.components[Component.STRUCTURE]
    assert structure.weight == pytest.approx(20.0 * 100.0 / 55.0, abs=1e-6)
    assert structure.detail["weight_before_redistribution"] == 20.0
    # the unmeasured components keep their configured weight and stay off
    assert flow.components[Component.ORDER_FLOW].weight == 25.0
    assert flow.components[Component.ORDER_FLOW].enabled is False
    assert "NOT COMPARABLE" in " ".join(outcome.reasons)


def test_a_redistributed_outcome_reports_no_shortfall_but_names_the_components():
    """REPORTED, not a defect: after redistribution the two point totals agree.

    `available_points` and `max_points` are both 100 once the weights have been
    scaled, so `missing_points` is 0 and a report that reads only those two
    numbers cannot see that 45 points had no feed. What keeps that honest is
    `unavailable_components`, `redistributed` and the capitalised reason, all of
    which are set -- so the fact is reported in three places and absent from
    one. Pinned so a future reader does not take `missing_points == 0` as
    evidence the dataset was complete.
    """
    cfg = flow_score_config(
        strict_component_availability=False, redistribute_disabled_weight=True
    )
    outcome = aggregate(cfg, scored(bars_only_components(0.5)))
    assert outcome.missing_points == pytest.approx(0.0, abs=1e-6)
    assert outcome.unavailable_components == (
        Component.OPTIONS_FLOW,
        Component.ORDER_FLOW,
    )
    assert outcome.redistributed is True
    assert any("WEIGHT WAS REDISTRIBUTED" in line for line in summary_lines(outcome))


def test_no_default_in_this_project_turns_redistribution_on():
    """Both flags ship in the safe position; the test states which that is."""
    shipped = FlowScoreConfig(weights=dict(WEIGHTS), total_points=100.0)
    assert shipped.strict_component_availability is True
    assert shipped.redistribute_disabled_weight is False


def test_redistribution_puts_the_two_point_based_gates_on_different_scales():
    """REPORTED FINDING: redistribution rescales one point gate and not the other.

    Two gates compare WEIGHTED POINTS against a configured limit: the
    options-flow gate (`options_contradiction_points`, 14.00) at stage 7 and the
    entry filter (`max_opposing_points`, 12.00) at stage 11. The engine hands
    stage 7 the UNSCALED `ComponentScore`s from the scorers and stage 11 the
    `FlowScore` the aggregate built -- which, with
    `redistribute_disabled_weight` on, carries weights scaled by
    `total_points / available_points`.

    Demonstrated here with ORDER_FLOW unavailable, so the scale is 100/75 =
    1.3333: OPTIONS_FLOW at magnitude 0.6 holds 12.00 unscaled points, which
    clears the 14.00 contradiction limit at stage 7, and 16.00 scaled points,
    which would not have. The same component's opposition is therefore measured
    on two scales inside one bar.

    Consequence: turning redistribution on makes the entry filter stricter by
    the redistribution factor while leaving the options gate's threshold where
    it was, and shifts which gate a rejection is attributed to -- so section
    12's histogram changes for a reason that has nothing to do with the market.
    The outcome in this scene is still a WAIT, so no trade is admitted that
    should not be; what moves is the reported stage.

    NOT FIXED, and deliberately: the correct resolution is a design decision.
    Section 3 writes the gate limits in weighted points on a 100-point scale,
    which argues the entry filter is right and the options gate should see the
    aggregated components; but redistribution's own reason text says the
    resulting numbers are "NOT COMPARABLE to a score from a complete dataset",
    which argues neither threshold means anything there. Either answer is a
    change to `signals/engine.py`'s data flow. The flag is off by default and
    no default turns it on, which is what keeps the severity at medium.
    """
    cfg = config(
        flow_score=flow_score_config(
            strict_component_availability=False, redistribute_disabled_weight=True
        )
    )
    table = components(
        options_flow=component(Component.OPTIONS_FLOW, 0.60, -1),
        order_flow=unavailable(Component.ORDER_FLOW),
        structure=component(Component.STRUCTURE, 0.90, 1),
        liquidity=component(Component.LIQUIDITY, 0.90, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, 0.90, 1),
    )
    # the unscaled number the options gate sees: 0.60 * 20 = 12.00, under 14.00
    assert table[Component.OPTIONS_FLOW].points == pytest.approx(12.0, abs=1e-9)
    assert OptionsFlowGate(cfg).check(table, Side.LONG).passed is True

    decision = decide(cfg, table, vector=features(achievable_rr=1.50))
    rescaled = decision.flow.require().components[Component.OPTIONS_FLOW]
    # the scaled number the entry filter sees: 0.60 * 20 * (100/75) = 16.00
    assert rescaled.points == pytest.approx(16.0, abs=1e-6)
    assert rescaled.points > cfg.flow_score.options_contradiction_points
    assert decision.gate_results[-1].stage is GateStage.ENTRY_FILTER
    assert decision.signal.wait_reason is WaitReason.OPTIONS_CONTRADICTION
    assert "16.00" in decision.signal.wait_detail


def test_nothing_validates_the_options_limit_against_the_options_weight():
    """REPORTED FINDING: a weight sweep can switch the contradiction gate off.

    `options_contradiction_points` is in WEIGHTED POINTS, so the most the
    OPTIONS_FLOW component can ever hold against a trade is its own weight. A
    config whose options weight is below the limit therefore has a gate that
    cannot fire, and nothing says so: `FlowScoreConfig` validates
    `max_opposing_points <= total_points` but makes no comparable check for
    `options_contradiction_points`, which is accepted at any non-negative value
    including 999.0.

    Section 7 names the weights as the Phase 8 sweep surface, so this is
    reachable by the sweep the project plans rather than by a typo: an
    equal-weight baseline at 20 points apiece keeps the gate alive, and any
    sweep that drops the options weight below 14 silently retires it.

    NOT FIXED: the check belongs in `config/schema.py`, which this phase does
    not own and which an agent is editing, and a limit deliberately set above
    the weight is a legitimate way to say "never block on options flow". The
    defect is that it is silent, not that it is possible.
    """
    thin_options = {
        Component.OPTIONS_FLOW: 5.0,
        Component.ORDER_FLOW: 30.0,
        Component.STRUCTURE: 25.0,
        Component.LIQUIDITY: 20.0,
        Component.VOL_MOMENTUM: 20.0,
    }
    cfg = FlowScoreConfig(
        weights=thin_options,
        total_points=100.0,
        options_contradiction_points=OPTIONS_CONTRADICTION_POINTS,
    )
    assert cfg.weights[Component.OPTIONS_FLOW] < cfg.options_contradiction_points

    root = config(flow_score=cfg)
    gate = OptionsFlowGate(root)
    # full magnitude, full opposition: 1.0 * 5.0 = 5.00 points, under 14.00
    maximal = components(
        options_flow=ComponentScore(
            component=Component.OPTIONS_FLOW, magnitude=1.0, direction=-1, weight=5.0
        ),
        order_flow=ComponentScore(
            component=Component.ORDER_FLOW, magnitude=0.5, direction=1, weight=30.0
        ),
        structure=ComponentScore(
            component=Component.STRUCTURE, magnitude=0.5, direction=1, weight=25.0
        ),
        liquidity=ComponentScore(
            component=Component.LIQUIDITY, magnitude=0.5, direction=0, weight=20.0
        ),
        vol_momentum=ComponentScore(
            component=Component.VOL_MOMENTUM, magnitude=0.5, direction=1, weight=20.0
        ),
    )
    assert gate.check(maximal, Side.LONG).passed is True

    # and the limit itself is unbounded above
    permissive = FlowScoreConfig(
        weights=dict(WEIGHTS), total_points=100.0, options_contradiction_points=999.0
    )
    assert permissive.options_contradiction_points == 999.0
    # by contrast, max_opposing_points IS checked against the total
    with pytest.raises(ValueError, match="max_opposing_points"):
        FlowScoreConfig(
            weights=dict(WEIGHTS), total_points=100.0, max_opposing_points=101.0
        )


def test_a_partial_score_compared_against_full_scale_thresholds_says_so():
    """Switching strict off compares 55-point scores to 100-point thresholds.

    That is what the flag is for, and the hazard the brief names. What makes it
    honest rather than silent is that the aggregate's own reason travels onto
    the `Signal`: the engine extends the reason list with `outcome.reasons`
    before the gate runs, so the bar that was rejected at a 30.00 threshold also
    records that its 27.50 was a number out of 55.0 and not out of 100.0.
    """
    cfg = config(
        flow_score=flow_score_config(strict_component_availability=False),
        setups=setup_table(
            SCALP_1R={"min_flow_score": 30.0},
            SETUP_2R={"min_flow_score": 30.0},
            DIRECTIONAL_3R={"min_flow_score": 30.0},
        ),
    )
    decision = decide(cfg, bars_only_components(0.5))
    assert decision.signal.action is SignalAction.WAIT
    assert decision.gate_results[-1].stage is GateStage.FLOW_SCORE
    assert decision.signal.wait_reason is WaitReason.SCORE_BELOW_THRESHOLD
    assert decision.signal.flow_score == 27.5
    joined = " ".join(decision.signal.reasons)
    assert "out of 55.0" in joined and "not out of 100.0" in joined
    assert "NOT redistributed" in joined


# ---------------------------------------------------------------------------
# section 7: direction never touches magnitude
# ---------------------------------------------------------------------------


def test_one_bearish_component_keeps_the_aggregate_high_and_blocks_both_sides():
    """The brief's adversarial case, in one test.

    Four components strongly bullish and VOL_MOMENTUM strongly bearish, all at
    magnitude 0.95:

        agreeing with a LONG   0.95 * (20 + 25 + 20)      = 61.75
        neutral (LIQUIDITY)    0.95 * 15                  = 14.25
        opposing a LONG        0.95 * 20                  = 19.00
                                                            -----
        aggregate magnitude                                 95.00

    (a) the aggregate is 95 of 100 available points -- a disagreement is
    evidence, not cancellation; and (b) NO trade is taken in either direction:
    the LONG is refused because VOL_MOMENTUM's 19.00 opposing points exceed the
    12.00 limit, and the SHORT is refused because the other three directional
    components hold 61.75 against it. LIQUIDITY's 14.25 points oppose NEITHER
    side -- its direction is 0 -- which is why the two opposing totals sum to
    80.75 and not to the 95.00 aggregate. A system that quietly took the
    majority direction would take the LONG here, since `net_direction()` is +1.
    """
    table = components(
        options_flow=component(Component.OPTIONS_FLOW, 0.95, 1),
        order_flow=component(Component.ORDER_FLOW, 0.95, 1),
        structure=component(Component.STRUCTURE, 0.95, 1),
        liquidity=component(Component.LIQUIDITY, 0.95, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, 0.95, -1),
    )
    cfg = config()
    outcome = aggregate(cfg, scored(table))
    flow = outcome.require()

    # (a) the magnitude aggregate does not net the disagreement away
    assert flow.score == pytest.approx(95.0, abs=1e-9)
    assert flow.score > 0.85 * outcome.available_points
    assert flow.opposing_points(Side.LONG) == pytest.approx(19.0, abs=1e-9)
    assert flow.opposing_points(Side.SHORT) == pytest.approx(61.75, abs=1e-9)
    # the neutral component opposes neither side, so the two totals do not
    # partition the aggregate
    assert flow.points_for(Component.LIQUIDITY) == pytest.approx(14.25, abs=1e-9)
    # the majority is bullish, and that is exactly what must not be traded
    assert flow.net_direction() == 1

    # (b) neither side clears the entry filter
    gate = EntryFilterGate(cfg)
    for side in (Side.LONG, Side.SHORT):
        result = gate.check(flow, side)
        assert result.passed is False, f"{side.value} was admitted"
        assert result.wait_reason is not None


def test_a_mirrored_pair_scores_identically_and_points_oppositely():
    """Magnitude is the same quantity whichever way the components point.

    Flipping every direction must change no magnitude and no total. If any
    magnitude were signed by its direction, the mirrored total would differ.
    """
    bullish = all_bullish(0.6)
    bearish = components(
        **{
            c.value: component(c, s.magnitude, -s.direction)
            for c, s in bullish.items()
        }
    )
    up = aggregate(flow_score_config(), scored(bullish)).require()
    down = aggregate(flow_score_config(), scored(bearish)).require()

    assert up.score == down.score
    for c in Component:
        assert up.components[c].magnitude == down.components[c].magnitude
        assert up.components[c].direction == -down.components[c].direction
    assert up.net_direction() == 1
    assert down.net_direction() == -1


def test_an_unavailable_component_is_never_counted_as_opposition():
    """"Never invent missing options or order-flow data."

    Direction 0 on an unmeasured component means no observation. Reporting
    "order flow disagreed" about a tape nobody observed would be inventing the
    disagreement, so both directional gates pass an UNAVAILABLE component.

    The bars-only bar carries STRUCTURE and VOL_MOMENTUM at direction +1, so
    40 * 0.9 = 36.00 points legitimately oppose a SHORT. What must be zero is
    the contribution of the two UNMEASURED components, in either direction --
    and it is zero because `ComponentScore.points` is 0 when `enabled` is
    False, so the 45 unmeasured points enter neither opposing total.
    """
    table = bars_only_components(0.9)
    flow = aggregate(
        flow_score_config(strict_component_availability=False), scored(table)
    ).require()
    for which in (Component.OPTIONS_FLOW, Component.ORDER_FLOW):
        assert table[which].enabled is False
        assert table[which].points == 0.0
        assert table[which].direction == 0
        assert table[which].opposes(Side.LONG) is False
        assert table[which].opposes(Side.SHORT) is False
    assert flow.opposing_points(Side.LONG) == 0.0
    assert flow.opposing_points(Side.SHORT) == pytest.approx(36.0, abs=1e-9)

    cfg = config()
    assert OrderFlowGate().check(table, Side.LONG).passed is True
    assert OrderFlowGate().check(table, Side.SHORT).passed is True
    assert OptionsFlowGate(cfg).check(table, Side.LONG).passed is True
    assert OptionsFlowGate(cfg).check(table, Side.SHORT).passed is True


def test_net_direction_is_never_called_in_the_three_modules_under_test():
    """The structural defence against trading the majority direction.

    The side comes from `level_direction` and nowhere else. This walks the AST
    of each module rather than grepping its text, so a mention inside a
    docstring -- which is where `net_direction` legitimately appears, explaining
    that it is not called -- cannot be mistaken for a call.
    """
    root = Path(__file__).resolve().parents[2] / "flow_model"
    for relative in (
        "signals/flow_score.py",
        "signals/gates.py",
        "signals/setups.py",
        "risk/sizing.py",
        "risk/limits.py",
    ):
        tree = ast.parse((root / relative).read_text())
        attributes = [
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        ]
        assert "net_direction" not in attributes, (
            f"{relative} reads FlowScore.net_direction(); the side comes from "
            "the structure and no code path may ask the components to vote"
        )


# ---------------------------------------------------------------------------
# the gate sequence
# ---------------------------------------------------------------------------


def test_the_declared_reasons_are_a_subset_of_the_enum_and_cover_every_stage():
    """No stage invents a reason, and none is left without one."""
    assert set(WAIT_REASONS_BY_STAGE) == set(GateStage)
    for stage, reasons in WAIT_REASONS_BY_STAGE.items():
        assert reasons, f"{stage.value} declares no reason"
        for reason in reasons:
            assert isinstance(reason, WaitReason)


def test_every_wait_reason_in_the_enum_is_declared_by_some_stage():
    """A reason no stage can emit is a bucket in section 12's histogram that
    never fills, which reads as "this never happens" rather than "nothing can
    report this"."""
    declared = {r for reasons in WAIT_REASONS_BY_STAGE.values() for r in reasons}
    assert declared == set(WaitReason)


def test_a_failing_gate_without_a_reason_is_refused():
    """Every WAIT is recorded with its blocking gate; a reasonless failure
    would be an unattributable rejection."""
    with pytest.raises(ValueError, match="must state a wait_reason"):
        GateResult(stage=GateStage.LIQUIDITY, passed=False)
    with pytest.raises(ValueError, match="must not carry a wait_reason"):
        GateResult(
            stage=GateStage.LIQUIDITY, passed=True, wait_reason=WaitReason.LIQUIDITY
        )


def test_a_gate_cannot_emit_a_reason_its_stage_did_not_declare():
    """Otherwise the labels on the rejection histogram are wrong."""
    with pytest.raises(ValueError, match="not among its declared reasons"):
        GateResult(
            stage=GateStage.LIQUIDITY,
            passed=False,
            wait_reason=WaitReason.RISK_LIMIT,
            detail="x",
        )


#: The peel. Each row repairs exactly one blocker in the scene above it and
#: names the stage and reason that must then be reported. The scene starts
#: failing thirteen stages at once, so each assertion is about ORDER and not
#: about a single broken thing.
PEEL: tuple[tuple[str, dict, GateStage | None, WaitReason | None], ...] = (
    ("the market is closed", {}, GateStage.SESSION, WaitReason.SESSION_CLOSED),
    (
        "the session opens",
        {"calendar": None},
        GateStage.MARKET_DATA,
        WaitReason.DATA_QUALITY,
    ),
    (
        "the feeds grade GOOD",
        {"quality": "good"},
        GateStage.WARMUP,
        WaitReason.WARMUP,
    ),
    (
        "the windows fill",
        {"warmup": "warm"},
        GateStage.REGIME,
        WaitReason.REGIME_BLOCKED,
    ),
    (
        "the regime becomes one a setup trades",
        {"regime": Regime.TRENDING_UP},
        GateStage.LIQUIDITY,
        WaitReason.LIQUIDITY,
    ),
    (
        "liquidity is measured again",
        {"feature": {"drop": ()}},
        GateStage.STRUCTURE,
        WaitReason.NO_STRUCTURE,
    ),
    (
        "14.6's four structure gates pass",
        {"feature": {"structure_gate_passed": 1.0, "level_gate_reason": GATE_PASSED}},
        GateStage.OPTIONS_FLOW,
        WaitReason.OPTIONS_CONTRADICTION,
    ),
    (
        "options flow turns neutral",
        {"component": {Component.OPTIONS_FLOW: (1.0, 0, True)}},
        GateStage.ORDER_FLOW,
        WaitReason.ORDERFLOW_CONFLICT,
    ),
    (
        "order flow agrees",
        {"component": {Component.ORDER_FLOW: (1.0, 1, True)}},
        GateStage.MOMENTUM,
        WaitReason.VOLATILITY_BAND,
    ),
    (
        "the ATR percentile enters the union band",
        {"feature": {"atr_percentile": 0.50}},
        GateStage.FLOW_SCORE,
        WaitReason.COMPONENT_DISABLED,
    ),
    (
        "the ablated LIQUIDITY component is measured again",
        {"component": {Component.LIQUIDITY: (0.0, 0, True)}},
        GateStage.FLOW_SCORE,
        WaitReason.SCORE_BELOW_THRESHOLD,
    ),
    (
        "the structure's magnitude clears the score floor",
        {"component": {Component.STRUCTURE: (0.50, 1, True)}},
        GateStage.ENTRY_FILTER,
        WaitReason.VOLATILITY_BAND,
    ),
    (
        "vol/momentum stops opposing",
        {"component": {Component.VOL_MOMENTUM: (1.0, 1, True)}},
        GateStage.SETUP,
        WaitReason.RR_TOO_LOW,
    ),
    (
        "the structure offers 1.5R",
        {"feature": {"achievable_rr": 1.50}},
        GateStage.RISK,
        WaitReason.RISK_LIMIT,
    ),
    (
        "the structural stop comes inside max_stop_atr_multiple * ATR",
        {"feature": {"level_stop_price": 99.00}},
        None,
        None,
    ),
)


def test_the_first_failing_gate_is_reported_as_the_scene_is_peeled():
    """Section 3's order, asserted by repairing one layer at a time.

    The opening scene fails every stage: the market is closed, the feeds are
    MISSING with a blocking feed, 10 of 514 bars are visible, the regime is
    CHOP (no enabled setup trades it), `liquidity_score` is absent, 14.6's
    significance gate failed, options flow opposes with 20.00 points (limit
    14.00), order flow opposes, the ATR percentile is 0.95 (union band
    [0.20, 0.80]), LIQUIDITY is ablated so the aggregate refuses, the measured
    magnitudes put the score below the 70.00 union floor, vol/momentum holds
    20.00 points against the long (limit 12.00), achievable R:R is 0.50 (floor
    1.00), and the structural stop is 10.00 points away against a 3.00-point
    maximum.

    Repairing one of those at a time walks the reported stage through every
    member of `GateStage` in the enum's own order and then produces a trade. A
    test that broke one thing at a time would assert only that each gate works;
    this asserts which one WINS, which is the property section 12's histogram
    rests on.

    The score floor here is 70.00 on all three setups rather than this file's
    usual 30/40/80, so that both of the FLOW_SCORE stage's reasons are reachable
    without changing anything else in the scene.
    """
    cfg = config(
        setups=setup_table(
            SCALP_1R={"min_flow_score": 70.0},
            SETUP_2R={"min_flow_score": 70.0},
            DIRECTIONAL_3R={"min_flow_score": 70.0},
        )
    )
    assert min_flow_score_required(cfg) == 70.0
    assert cfg.flow_score.max_opposing_points == MAX_OPPOSING_POINTS
    assert cfg.flow_score.options_contradiction_points == OPTIONS_CONTRADICTION_POINTS

    feature_kwargs = dict(
        drop=("liquidity_score",),
        structure_gate_passed=0.0,
        level_gate_reason=GATE_SIGNIFICANCE,
        level_direction=DIRECTION_SUPPORT,
        atr_percentile=0.95,
        achievable_rr=0.50,
        level_stop_price=90.00,
        atr=1.00,
    )
    component_kwargs: dict[Component, tuple[float, int, bool]] = {
        Component.OPTIONS_FLOW: (1.0, -1, True),
        Component.ORDER_FLOW: (1.0, -1, True),
        Component.STRUCTURE: (0.10, 1, True),
        Component.LIQUIDITY: (0.0, 0, False),
        Component.VOL_MOMENTUM: (1.0, -1, True),
    }
    scene: dict = {
        "calendar": ClosedCalendar(),
        "quality": quality_report(
            overall=DataQuality.MISSING, blocking=(Feed.TICK_AGGREGATE,)
        ),
        "warmup": warmup_report(bars_available=10, bars_required=514),
        "regime": BLOCKED_REGIME,
    }

    observed: list[GateStage] = []
    for label, repair, stage, reason in PEEL:
        feature_kwargs.update(repair.pop("feature", {}))
        for which, (magnitude, direction, enabled) in repair.pop(
            "component", {}
        ).items():
            component_kwargs[which] = (magnitude, direction, enabled)
        for key, value in repair.items():
            scene[key] = (
                quality_report()
                if value == "good"
                else warmup_report()
                if value == "warm"
                else value
            )

        table = components(
            **{
                which.value: (
                    component(which, magnitude, direction)
                    if enabled
                    else unavailable(which)
                )
                for which, (magnitude, direction, enabled) in component_kwargs.items()
            }
        )
        decision = decide(
            cfg,
            table,
            vector=features(**feature_kwargs),
            regime=scene["regime"],
            quality=scene["quality"],
            warmup=scene["warmup"],
            calendar=scene["calendar"],
        )
        signal = decision.signal
        last = decision.gate_results[-1]

        if stage is None:
            assert signal.action is SignalAction.LONG, (
                f"after '{label}' the bar should trade: {signal.wait_reason} "
                f"{signal.wait_detail}"
            )
            assert decision.plan is not None
            assert decision.plan.setup is SetupType.SCALP_1R
            assert tuple(r.stage for r in decision.gate_results) == tuple(GateStage)
            continue

        assert signal.action is SignalAction.WAIT, f"after '{label}' no WAIT"
        assert last.stage is stage, (
            f"after '{label}' the first failing stage was {last.stage.value}, "
            f"expected {stage.value}"
        )
        assert signal.wait_reason is reason, (
            f"after '{label}' the reason was {signal.wait_reason}, expected {reason}"
        )
        assert signal.wait_detail, f"after '{label}' the WAIT carried no detail"
        # the stages that ran are a prefix of GateStage, ending at the failure
        ran = tuple(r.stage for r in decision.gate_results)
        assert ran == tuple(GateStage)[: len(ran)]
        assert ran[-1] is stage
        observed.append(stage)

    # FLOW_SCORE appears twice, once for each of its two declared reasons, so
    # `observed` is longer than GateStage. Two properties hold: every stage is
    # the first failure of some scene, and the reported stage only ever moves
    # FORWARD through GateStage as the scene is repaired.
    order = list(GateStage)
    assert sorted(set(observed), key=order.index) == order, (
        "every stage must be the first failure of some scene"
    )
    positions = [order.index(s) for s in observed]
    assert positions == sorted(positions), (
        "repairing a blocker must never move the reported stage backwards"
    )
    assert observed.count(GateStage.FLOW_SCORE) == 2


def test_the_peel_reaches_every_wait_reason_the_gates_can_emit():
    """A reason no constructed state reaches is dead code or a silent gate.

    The peel covers thirteen of the fourteen `WaitReason` members. The
    fourteenth, `NO_SETUP_MATCH`, needs a band naming a setup the configuration
    does not offer, which is a configuration change rather than a scene change
    and has its own test below.
    """
    reached = {reason for _, _, _, reason in PEEL if reason is not None}
    assert reached == set(WaitReason) - {WaitReason.NO_SETUP_MATCH}


def test_no_setup_match_is_reached_when_the_band_names_a_disabled_setup():
    """The fourteenth reason, and the brief's "NO setup matches"."""
    cfg = config(setups=setup_table(SETUP_2R={"enabled": False}))
    decision = decide(cfg, all_bullish(0.9), vector=features(achievable_rr=2.00))
    assert decision.signal.action is SignalAction.WAIT
    assert decision.signal.wait_reason is WaitReason.NO_SETUP_MATCH
    assert decision.gate_results[-1].stage is GateStage.SETUP


def test_three_of_the_entry_filter_s_five_declared_reasons_are_unreachable():
    """REPORTED FINDING: the entry filter cannot emit three of its five reasons.

    `WAIT_REASONS_BY_STAGE[ENTRY_FILTER]` declares ORDERFLOW_CONFLICT,
    OPTIONS_CONTRADICTION, NO_STRUCTURE, LIQUIDITY and VOLATILITY_BAND. Three
    of them cannot be produced by the pipeline:

    * **ORDER_FLOW.** The order-flow gate at stage 8 refuses ANY opposing
      direction, with no magnitude threshold, so a bar that reaches stage 11
      has order flow agreeing or neutral. Asserted below with a magnitude of
      1e-6, which still blocks.
    * **LIQUIDITY.** `LiquidityScorer.direction_keys` is empty, so the
      component's direction is always 0 and it can never oppose. `gates.py`
      documents this one.
    * **STRUCTURE.** `StructureScorer.direction_keys` is `("level_direction",)`
      -- the same key `StructureGate` reads the side from -- so the component's
      direction equals the side by construction. `gates.py` does NOT document
      this one; its comment says only LIQUIDITY "can never appear here".

    So under any configuration the entry filter reports VOL_MOMENTUM's
    VOLATILITY_BAND, and OPTIONS_CONTRADICTION only in the window where options
    opposition lies in `(max_opposing_points, options_contradiction_points]` --
    `(12.00, 14.00]` as shipped. If those two limits were ever set equal, that
    window would close and the entry filter would become a VOL_MOMENTUM gate
    with a four-reason vocabulary it cannot use. Severity: low on correctness,
    medium on section 12, which labels its buckets from this mapping. The
    mapping itself is correct, which the last block asserts by feeding the gate
    a hand-built aggregate no scorer set can produce.
    """
    # ORDER_FLOW: any opposing direction blocks, however small the magnitude
    tiny = components(
        options_flow=component(Component.OPTIONS_FLOW, 0.0, 0),
        order_flow=component(Component.ORDER_FLOW, 1e-6, -1),
        structure=component(Component.STRUCTURE, 0.5, 1),
        liquidity=component(Component.LIQUIDITY, 0.5, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, 0.5, 1),
    )
    blocked = OrderFlowGate().check(tiny, Side.LONG)
    assert blocked.passed is False
    assert blocked.wait_reason is WaitReason.ORDERFLOW_CONFLICT

    # LIQUIDITY and STRUCTURE: the scorers' declared keys make opposition
    # impossible rather than merely unlikely
    assert LiquidityScorer.direction_keys == ()
    assert StructureScorer.direction_keys == ("level_direction",)
    assert VolMomentumScorer.direction_keys == ("direction",)
    # the structure gate reads the SIDE from the same key the STRUCTURE scorer
    # reads its direction from, so the two agree by construction
    assert StructureGate().check(features()).side is Side.LONG
    assert (
        StructureGate().check(features(level_direction=DIRECTION_RESISTANCE)).side
        is Side.SHORT
    )

    # the mapping is right; only the pipeline cannot reach it
    hand_built = FlowScore(
        symbol=SYMBOL,
        ts=TS,
        components=components(
            options_flow=component(Component.OPTIONS_FLOW, 0.0, 0),
            order_flow=component(Component.ORDER_FLOW, 0.0, 0),
            structure=component(Component.STRUCTURE, 1.0, -1),
            liquidity=component(Component.LIQUIDITY, 0.0, 0),
            vol_momentum=component(Component.VOL_MOMENTUM, 0.0, 0),
        ),
    )
    result = EntryFilterGate(config()).check(hand_built, Side.LONG)
    assert result.passed is False
    assert result.wait_reason is OPPOSITION_WAIT_REASON[Component.STRUCTURE]
    assert result.wait_reason is WaitReason.NO_STRUCTURE


def test_two_of_the_setup_stage_s_declared_reasons_are_unreachable_as_shipped():
    """REPORTED FINDING: NO_STRUCTURE and COMPONENT_DISABLED cannot fire there.

    `WAIT_REASONS_BY_STAGE[SETUP]` declares seven reasons. Two of them cannot
    be reached with `strict_component_availability` on, which is the shipped
    default:

    * **NO_STRUCTURE**, from `require_structure_confirmation` when the STRUCTURE
      component does not agree with the side. It always agrees: both the side
      and the component's direction are read from `level_direction`. Asserted
      below over every value that key can take. (It is reachable by calling
      `select` directly with a side the structure did not name -- see
      `test_select_does_not_check_the_side_against_the_structure_it_came_from`
      -- which the engine never does.)
    * **COMPONENT_DISABLED**, from either confirmation flag when its component
      was never measured. Under strict mode an unmeasured component makes the
      aggregate REFUSE at stage 10, three stages earlier, so the setup stage is
      never reached. Asserted below by showing the same unmeasured ORDER_FLOW
      reports at FLOW_SCORE with strict on and at SETUP with it off.

    The REASONS themselves are reachable elsewhere -- NO_STRUCTURE at the
    structure gate, COMPONENT_DISABLED at the flow-score refusal -- so no enum
    member is dead. What is dead is the (stage, reason) pair, and section 12
    labels its buckets by stage, so two of the setup stage's seven labels can
    never appear. Severity: low on correctness, and worth knowing before
    someone reads an empty bucket as "this never happens in the market".
    """
    cfg = config()
    assert cfg.flow_score.strict_component_availability is True
    for setup in cfg.setups.values():
        assert setup.require_structure_confirmation is True
        assert setup.require_orderflow_confirmation is True

    # NO_STRUCTURE: the side and the component's direction share one source
    for direction, side in (
        (DIRECTION_SUPPORT, Side.LONG),
        (DIRECTION_RESISTANCE, Side.SHORT),
    ):
        gate_side = StructureGate().check(features(level_direction=direction)).side
        assert gate_side is side
        structure_component = component(Component.STRUCTURE, 0.9, int(direction))
        assert structure_component.agrees_with(side) is True
        assert structure_component.opposes(side) is False

    # COMPONENT_DISABLED: strict mode refuses three stages earlier
    table = dict(all_bullish(0.9))
    table[Component.ORDER_FLOW] = unavailable(Component.ORDER_FLOW)
    strict = decide(cfg, table, vector=features(achievable_rr=1.50))
    assert strict.gate_results[-1].stage is GateStage.FLOW_SCORE
    assert strict.signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert strict.selection is None  # the setup stage was never reached

    relaxed = decide(
        config(flow_score=flow_score_config(strict_component_availability=False)),
        table,
        vector=features(achievable_rr=1.50),
    )
    assert relaxed.gate_results[-1].stage is GateStage.SETUP
    assert relaxed.signal.wait_reason is WaitReason.COMPONENT_DISABLED


def test_the_entry_filter_breaks_a_tie_the_same_way_every_run():
    """Section 12 counts gates, so a tie must not be resolved by chance.

    Two components holding exactly 18.00 opposing points each: the reported
    reason has to be identical across calls, because a histogram built from a
    nondeterministic tie-break is not a measurement. The determinism comes from
    `score_components` ordering the mapping by `Component` declaration order
    before the aggregate sees it; the detail names BOTH opposers either way, so
    nothing is lost beyond the bucket.
    """
    cfg = config()
    tied = FlowScore(
        symbol=SYMBOL,
        ts=TS,
        components=components(
            options_flow=component(Component.OPTIONS_FLOW, 0.90, -1),   # 18.00
            order_flow=component(Component.ORDER_FLOW, 0.00, 0),
            structure=component(Component.STRUCTURE, 0.50, 1),
            liquidity=component(Component.LIQUIDITY, 0.50, 0),
            vol_momentum=component(Component.VOL_MOMENTUM, 0.90, -1),   # 18.00
        ),
    )
    assert tied.components[Component.OPTIONS_FLOW].points == pytest.approx(18.0, abs=1e-9)
    assert tied.components[Component.VOL_MOMENTUM].points == pytest.approx(18.0, abs=1e-9)
    first = EntryFilterGate(cfg).check(tied, Side.LONG)
    second = EntryFilterGate(cfg).check(tied, Side.LONG)
    assert first.wait_reason is second.wait_reason
    assert first.detail == second.detail
    assert "options_flow 18.00" in first.detail
    assert "vol_momentum 18.00" in first.detail


def test_a_fractional_level_direction_raises_rather_than_waiting():
    """The one data-shaped input in the gate layer that raises.

    Every other missing or declining measurement produces a WAIT. A
    `level_direction` of 0.5 is not a declining measurement -- it is a
    contract violation, since the key is defined as {-1, 0, +1} -- and a
    fractional direction silently rounded to a side would be a magnitude
    wearing a direction's name. Pinned so the asymmetry stays deliberate: a
    corrupt feature vector halts the run instead of producing signals from it.
    """
    with pytest.raises(ValueError, match="not one of"):
        StructureGate().check(features(level_direction=0.5))


def test_options_opposition_between_the_two_limits_blocks_at_the_entry_filter():
    """The one window in which the entry filter reports OPTIONS_CONTRADICTION.

    13.00 points of options opposition (0.65 * 20) clears the 14.00-point
    contradiction limit at stage 7 and then exceeds the 12.00-point
    `max_opposing_points` at stage 11. Both numbers are hand-computed, and the
    stage distinguishes the two reports of the same reason.
    """
    cfg = config()
    table = components(
        options_flow=component(Component.OPTIONS_FLOW, 0.65, -1),
        order_flow=component(Component.ORDER_FLOW, 0.90, 1),
        structure=component(Component.STRUCTURE, 0.90, 1),
        liquidity=component(Component.LIQUIDITY, 0.90, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, 0.90, 1),
    )
    assert table[Component.OPTIONS_FLOW].points == pytest.approx(13.0, abs=1e-9)
    assert OptionsFlowGate(cfg).check(table, Side.LONG).passed is True

    decision = decide(cfg, table)
    assert decision.signal.action is SignalAction.WAIT
    assert decision.gate_results[-1].stage is GateStage.ENTRY_FILTER
    assert decision.signal.wait_reason is WaitReason.OPTIONS_CONTRADICTION
    assert "13.00" in decision.signal.wait_detail


def test_the_side_comes_from_the_structure_and_the_majority_cannot_outvote_it():
    """A bearish level against three bullish components takes no trade.

    `level_direction` is -1 (resistance), so the only side considered is SHORT.
    Options flow, order flow and vol/momentum all point +1, which makes
    `net_direction()` +1 -- the majority -- and the pipeline never asks them:
    the SHORT is the only candidate, and it is refused.

    MY FIRST EXPECTATION HERE WAS WRONG. I expected ORDERFLOW_CONFLICT, because
    order flow is the confirmation gate section 3 names. The options-flow gate
    runs one stage EARLIER (section 3's prose order is options flow then order
    flow) and its 20.00 opposing points already exceed the 14.00-point
    contradiction limit, so OPTIONS_CONTRADICTION is the correct first failure.
    The implementation is right and my reading of the order was not. The second
    scene below turns options flow neutral and then order flow reports.
    """
    cfg = config()
    table = components(
        options_flow=component(Component.OPTIONS_FLOW, 1.0, 1),
        order_flow=component(Component.ORDER_FLOW, 1.0, 1),
        structure=component(Component.STRUCTURE, 1.0, -1),
        liquidity=component(Component.LIQUIDITY, 1.0, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, 1.0, 1),
    )
    flow = aggregate(cfg, scored(table)).require()
    assert flow.net_direction() == 1
    # three components hold 20 + 25 + 20 = 65 points against the only candidate
    assert flow.opposing_points(Side.SHORT) == pytest.approx(65.0, abs=1e-9)

    decision = decide(
        cfg, table, vector=features(level_direction=DIRECTION_RESISTANCE)
    )
    assert decision.signal.action is SignalAction.WAIT
    # the structure named SHORT, and SHORT is the only side any stage saw
    sides = {r.side for r in decision.gate_results if r.side is not None}
    assert sides == {Side.SHORT}
    assert decision.gate_results[-1].stage is GateStage.OPTIONS_FLOW
    assert decision.signal.wait_reason is WaitReason.OPTIONS_CONTRADICTION

    # with options flow neutral, order flow is the gate that reports
    quiet_options = dict(table)
    quiet_options[Component.OPTIONS_FLOW] = component(Component.OPTIONS_FLOW, 1.0, 0)
    second = decide(
        cfg,
        quiet_options,
        vector=features(level_direction=DIRECTION_RESISTANCE),
    )
    assert second.gate_results[-1].stage is GateStage.ORDER_FLOW
    assert second.signal.wait_reason is WaitReason.ORDERFLOW_CONFLICT
    assert {r.side for r in second.gate_results if r.side is not None} == {Side.SHORT}


def test_a_level_with_no_direction_is_not_resolved_into_a_side():
    """Price inside the zone offers no side, and none is invented."""
    result = StructureGate().check(features(level_direction=DIRECTION_NONE))
    assert result.passed is False
    assert result.wait_reason is WaitReason.NO_STRUCTURE
    assert result.side is None


@pytest.mark.parametrize(
    "code,expected",
    [
        (GATE_NO_ZONE, WaitReason.NO_STRUCTURE),
        (GATE_PROXIMITY, WaitReason.NO_STRUCTURE),
        (GATE_SIGNIFICANCE, WaitReason.NO_STRUCTURE),
        (GATE_CLEANLINESS, WaitReason.NO_STRUCTURE),
        (GATE_REJECTION, WaitReason.NO_STRUCTURE),
        (GATE_NO_TARGET, WaitReason.RR_TOO_LOW),
        (GATE_REWARD_RISK, WaitReason.RR_TOO_LOW),
    ],
)
def test_levels_own_gate_codes_map_to_the_two_reasons_it_published(code, expected):
    """levels.py's mapping, quoted from its docstring: 1-5 -> NO_STRUCTURE,
    6-7 -> RR_TOO_LOW. Asserted against its exported constants rather than the
    literals, so renumbering there cannot silently remap here."""
    result = StructureGate().check(
        features(structure_gate_passed=0.0, level_gate_reason=code)
    )
    assert result.passed is False
    assert result.wait_reason is expected
    assert result.diagnostics["level_gate_reason"] == code


def test_a_level_engine_that_contradicts_itself_declines_the_bar():
    """`structure_gate_passed=1` with a non-passing reason code is a
    feature-layer disagreement, and the bar is declined rather than resolved by
    preferring one of the two outputs."""
    result = StructureGate().check(
        features(structure_gate_passed=1.0, level_gate_reason=GATE_CLEANLINESS)
    )
    assert result.passed is False
    assert result.wait_reason is WaitReason.NO_STRUCTURE
    assert "disagree" in result.detail


def test_the_liquidity_gate_refuses_an_unmeasured_bar_rather_than_guessing():
    """No execution-cost measurement means no trade; nothing is estimated."""
    absent = LiquidityGate(config(), spec()).check(features(drop=("liquidity_score",)))
    assert absent.passed is False
    assert absent.wait_reason is WaitReason.LIQUIDITY

    graded_missing = LiquidityGate(config(), spec()).check(
        features(missing=("liquidity_score",))
    )
    assert graded_missing.passed is False


def test_the_liquidity_gate_s_magnitude_floor_ships_switched_off():
    """REPORTED: section 3's liquidity "band" has no constant in any document.

    The gate enforces the two conditions that need none -- a measured spread
    below the instrument's own minimum, and a bar in which nothing traded -- and
    the `liquidity_score` floor is 0.0, which never fires. The consequence,
    stated plainly: on well-formed data the magnitude half of this gate is
    vacuous until someone sets the floor.
    """
    from flow_model.signals.gates import LIQUIDITY_GATE_MIN_SCORE

    assert LIQUIDITY_GATE_MIN_SCORE == 0.0
    # a floor of 0.0 admits even a zero liquidity_score
    at_zero = LiquidityGate(config(), spec()).check(features(liquidity_score=0.0))
    assert at_zero.passed is True
    # and it can be raised without a code edit
    raised = LiquidityGate(config(), spec(), min_liquidity_score=0.50).check(
        features(liquidity_score=0.40)
    )
    assert raised.passed is False
    assert raised.wait_reason is WaitReason.LIQUIDITY


def test_a_broken_book_and_an_empty_bar_are_the_two_constant_free_checks():
    """A measured spread below `min_spread_ticks` is unusable, not tight; a bar
    with no volume cannot be filled out of at any participation rate."""
    crossed = LiquidityGate(config(), spec()).check(features(spread_ticks=0.50))
    assert crossed.passed is False
    assert "unusable" in crossed.detail

    empty = LiquidityGate(config(), spec()).check(features(relative_volume=0.0))
    assert empty.passed is False
    assert "no volume" in " ".join(empty.reasons)

    # a spread that is the instrument's configured fallback is NOT judged:
    # features/liquidity.py grades that key DEGRADED and has already capped the
    # magnitude, so judging it would be judging configuration
    fallback = LiquidityGate(config(), spec()).check(
        FeatureVector(
            symbol=SYMBOL,
            ts=TS,
            values=dict(features().values, spread_ticks=0.50),
            quality_by_key={
                **features().quality_by_key,
                "spread_ticks": DataQuality.DEGRADED,
            },
        )
    )
    assert fallback.passed is True


def test_warmup_is_judged_per_computer_so_an_absent_feed_never_reports_warmup():
    """The trap: `FeatureVector.warmup_complete` is the AND over every computer.

    On bars-only data the order-flow and options-flow computers are permanently
    not-ready, so the merged flag is permanently False and a gate reading it
    would report WARMUP on every bar of a ten-year run. A computer whose FEED is
    absent does not block warmup.
    """
    absent_feed = ComputerReadiness(
        name="orderflow",
        feeds_present=False,
        bars_required=130,
        bars_available=10,
        absent_feeds=(Feed.TICK_AGGREGATE,),
    )
    warm = ComputerReadiness(
        name="levels", feeds_present=True, bars_required=514, bars_available=600
    )
    cold = ComputerReadiness(
        name="liquidity", feeds_present=True, bars_required=1560, bars_available=600
    )
    assert absent_feed.blocks_warmup is False
    assert warm.blocks_warmup is False
    assert cold.blocks_warmup is True

    report = WarmupReport(bars_available=600, computers=(absent_feed, warm))
    assert report.blocking == ()
    # bars_required is the longest window among computers whose feeds are present
    assert report.bars_required == 514


def test_the_momentum_gate_reads_the_atr_percentile_and_not_the_vol_score():
    """Section 3's band is written against `atr_percentile`. `vol_regime_score`
    is a different quantity ("the range is in the payable band") and is not
    substituted for it, which a bar carrying only the latter proves."""
    cfg = config()
    assert vol_percentile_band(cfg) == (0.20, 0.80)
    decision = decide(cfg, all_bullish(0.9), vector=features(atr_percentile=0.95))
    assert decision.gate_results[-1].stage is GateStage.MOMENTUM
    assert decision.signal.wait_reason is WaitReason.VOLATILITY_BAND

    unmeasured = decide(
        cfg, all_bullish(0.9), vector=features(missing=("atr_percentile",))
    )
    assert unmeasured.gate_results[-1].stage is GateStage.MOMENTUM
    assert unmeasured.signal.wait_reason is WaitReason.VOLATILITY_BAND


def test_a_refused_aggregate_is_reported_before_any_threshold_comparison():
    """A refused aggregate has no score, so SCORE_BELOW_THRESHOLD would be a
    comparison against a number that does not exist."""
    cfg = config()
    decision = decide(cfg, bars_only_components(0.9))
    assert decision.signal.action is SignalAction.WAIT
    assert decision.gate_results[-1].stage is GateStage.FLOW_SCORE
    assert decision.signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert decision.signal.flow_score is None
    assert decision.flow is not None and decision.flow.refused is True
    assert decision.flow.available_points == 55.0
    assert decision.flow.max_points == 100.0
    assert "55.0 of 100.0" in " ".join(decision.signal.reasons)


def test_an_ablated_component_makes_every_bar_refuse_under_strict_mode():
    """REPORTED consequence: a Phase 8 ablation under strict mode trades nothing.

    That is literally what `strict_component_availability` says it does, but it
    means an ablation sweep has to turn strict mode off deliberately rather than
    discovering that every bar came back COMPONENT_DISABLED.
    """
    cfg = config()
    table = all_bullish(0.9)
    table[Component.VOL_MOMENTUM] = unavailable(Component.VOL_MOMENTUM)
    decision = decide(cfg, table)
    assert decision.signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert decision.flow.available_points == 80.0
    assert decision.flow.missing_points == 20.0


# ---------------------------------------------------------------------------
# 14.6: setup selection is a measurement
# ---------------------------------------------------------------------------


def selection_for(
    cfg: FlowModelConfig,
    *,
    achievable_rr: float,
    flow_score: float = 90.0,
    atr_percentile: float = 0.40,
    regime: Regime = Regime.TRENDING_UP,
    side: Side = Side.LONG,
    table: dict[Component, ComponentScore] | None = None,
    structure_min_reward_risk: float = 1.0,
    **feature_kwargs,
):
    return select(
        cfg,
        features(
            achievable_rr=achievable_rr,
            atr_percentile=atr_percentile,
            structure_min_reward_risk=structure_min_reward_risk,
            **feature_kwargs,
        ),
        side=side,
        regime=regime_state(regime),
        flow_score=flow_score,
        components=table if table is not None else all_bullish(0.9),
    )


@pytest.mark.parametrize(
    "achievable_rr,setup,planned",
    [
        (1.00, SetupType.SCALP_1R, 1.0),        # the lowest band's floor, inclusive
        (1.50, SetupType.SCALP_1R, 1.0),
        (1.79, SetupType.SCALP_1R, 1.0),        # NOT promoted to 2R
        (1.80, SetupType.SETUP_2R, 2.0),        # 14.6's second edge, inclusive
        (2.00, SetupType.SETUP_2R, 2.0),
        (2.69, SetupType.SETUP_2R, 2.0),        # NOT promoted to 3R
        (2.70, SetupType.DIRECTIONAL_3R, 3.0),  # 14.6's third edge, inclusive
        (9.00, SetupType.DIRECTIONAL_3R, 3.0),  # NOT truncated, and not 9R either
    ],
)
def test_the_band_read_off_achievable_rr_determines_exactly_one_setup(
    achievable_rr, setup, planned
):
    """14.6 verbatim: `1.0 <= rr < 1.8 -> SCALP_1R`, `1.8 <= rr < 2.7 ->
    SETUP_2R`, `rr >= 2.7 -> DIRECTIONAL_3R`.

    Both edges of every band are walked. The planned R is the SETUP's
    `reward_risk` and not the measured `achievable_rr`, which is what keeps
    `TradeRecord.r_label_is_honest()` passing: a 2.8R measurement recorded as a
    DIRECTIONAL_3R must plan 3.0R, not 2.8R.
    """
    selection = selection_for(config(), achievable_rr=achievable_rr)
    assert selection.matched is True
    assert selection.setup is setup
    assert selection.wait_reason is None
    assert selection.planned_reward_risk == planned
    assert selection.achievable_rr == achievable_rr
    # the band table is levels.py's, and the two readings agree
    assert SETUP_BY_RR_CLASS[setup_class_for(achievable_rr, 1.0)] is setup


def test_a_reward_risk_below_the_floor_is_declined_rather_than_retargeted():
    """The mechanism that stops a 1R scalp becoming a 0.4R target.

    The floor is the HIGHER of `config.levels.min_reward_risk` (1.0) and the
    lowest enabled setup's `min_reward_risk` (SCALP_1R's 1.0), so 1.0 -- and
    0.90 is below it. The measured target is still reported; nothing shrinks it.
    """
    cfg = config()
    assert min_reward_risk_floor(cfg) == 1.0
    selection = selection_for(cfg, achievable_rr=0.90)
    assert selection.matched is False
    assert selection.setup is None
    assert selection.wait_reason is WaitReason.RR_TOO_LOW
    assert selection.achievable_rr == 0.90
    # The measured target for THIS geometry, by the builder's own documented
    # formula: entry + achievable_rr * (entry - stop) = 100 + 0.90 * 1 = 100.90.
    # The first version of this assertion said 101.50, which is the target for
    # the builder's DEFAULT achievable_rr of 1.50 -- asserted against the
    # default instead of the 0.90 this test passes in. The expectation was
    # wrong, not the selector: 100.90 is what keeps |target - entry| /
    # |entry - stop| equal to the 0.90 that was measured.
    assert selection.structural_target_price == pytest.approx(100.90)
    assert "rather than moving the target closer" in selection.detail


def test_a_reward_risk_that_clears_the_floor_but_matches_no_band_is_declined():
    """The gap 14.6 leaves between a configurable floor and its lowest band.

    With `levels.min_reward_risk` and every setup's `min_reward_risk` at 0.50,
    the floor is 0.50 and a 0.70R geometry clears it -- and still belongs to no
    band, because the lowest band starts at 1.0. Declined, not promoted to the
    nearest one. This is why `StructureLevelConfig.min_reward_risk` defaults to
    1.0 rather than `SetupConfig`'s 0.9.
    """
    cfg = config(
        levels=StructureLevelConfig(min_reward_risk=0.50),
        setups=setup_table(
            SCALP_1R={"min_reward_risk": 0.50},
            SETUP_2R={"min_reward_risk": 0.50},
            DIRECTIONAL_3R={"min_reward_risk": 0.50},
        ),
    )
    assert min_reward_risk_floor(cfg) == 0.50
    assert min(floor for floor, _ in SETUP_RR_BANDS) == 1.0
    selection = selection_for(
        cfg, achievable_rr=0.70, structure_min_reward_risk=0.50
    )
    assert selection.matched is False
    assert selection.wait_reason is WaitReason.RR_TOO_LOW
    assert "falls in no band" in selection.detail


def test_a_two_r_geometry_is_declined_rather_than_handed_to_the_setup_below():
    """"Do not force every trade into the same target."

    With SETUP_2R disabled but SCALP_1R enabled (so the floor stays at 1.0), a
    2.0R measurement names SETUP_2R, which is unavailable. The bar is declined:
    not demoted to SCALP_1R, not promoted to DIRECTIONAL_3R.
    """
    cfg = config(setups=setup_table(SETUP_2R={"enabled": False}))
    assert min_reward_risk_floor(cfg) == 1.0
    selection = selection_for(cfg, achievable_rr=2.00)
    assert selection.matched is False
    assert selection.setup is None
    assert selection.wait_reason is WaitReason.NO_SETUP_MATCH
    assert "not retargeted to a setup the structure did not name" in selection.detail


def test_a_three_r_geometry_is_declined_rather_than_truncated_to_two_r():
    """The mirror case: a 3.5R structure with DIRECTIONAL_3R disabled."""
    cfg = config(setups=setup_table(DIRECTIONAL_3R={"enabled": False}))
    selection = selection_for(cfg, achievable_rr=3.50)
    assert selection.matched is False
    assert selection.setup is None
    assert selection.wait_reason is WaitReason.NO_SETUP_MATCH


def test_both_reward_risk_numbers_are_recorded_where_the_band_floor_is_lower():
    """14.6's band floors sit below the setups' own `reward_risk`, and the gap
    is reported rather than reconciled away.

    A 1.85R measurement selects SETUP_2R, which plans 2.0R -- so the planned
    target sits BEYOND the opposing major zone the measurement came from.
    `plans_beyond_measured_target` says so and both numbers are carried.
    """
    selection = selection_for(config(), achievable_rr=1.85)
    assert selection.setup is SetupType.SETUP_2R
    assert selection.achievable_rr == 1.85
    assert selection.planned_reward_risk == 2.0
    assert selection.plans_beyond_measured_target is True
    assert any("optimistic about reachability" in r for r in selection.reasons)

    # and it is False once the measurement exceeds the plan
    comfortable = selection_for(config(), achievable_rr=2.50)
    assert comfortable.setup is SetupType.SETUP_2R
    assert comfortable.plans_beyond_measured_target is False


def test_a_setup_class_that_disagrees_with_the_measurement_raises():
    """A feature-layer regression must be loud, not a mislabelled trade.

    `level_setup_class = 3.0` against `achievable_rr = 1.50` would produce a
    DIRECTIONAL_3R planned at 3R off a structure that supports 1.5R.
    """
    with pytest.raises(SetupError, match="band table"):
        selection_for(config(), achievable_rr=1.50, level_setup_class=3.0)


def test_an_absent_measurement_is_no_setup_match_rather_than_a_default():
    """A setup is a measurement here, not a default."""
    absent = select(
        config(),
        features(drop=("achievable_rr",)),
        side=Side.LONG,
        regime=regime_state(),
        flow_score=90.0,
        components=all_bullish(0.9),
    )
    assert absent.wait_reason is WaitReason.NO_SETUP_MATCH
    assert "never made" in absent.detail

    graded_missing = select(
        config(),
        features(missing=("achievable_rr",)),
        side=Side.LONG,
        regime=regime_state(),
        flow_score=90.0,
        components=all_bullish(0.9),
    )
    assert graded_missing.wait_reason is WaitReason.NO_SETUP_MATCH
    assert "not a default" in graded_missing.detail


def test_the_determined_setup_s_own_thresholds_are_reapplied_after_the_union():
    """The pre-selection gates ask the weaker question; the setup re-asks it.

    A score of 50.0 clears the 30.0 union floor and fails DIRECTIONAL_3R's own
    80.0, and section 3's reason for that comparison is SCORE_BELOW_THRESHOLD --
    not NO_SETUP_MATCH, which would make SCORE_BELOW_THRESHOLD unreachable at
    the setup stage and the histogram useless.
    """
    cfg = config()
    assert min_flow_score_required(cfg) == 30.0
    selection = selection_for(cfg, achievable_rr=3.00, flow_score=50.0)
    assert selection.matched is False
    assert selection.wait_reason is WaitReason.SCORE_BELOW_THRESHOLD
    assert "below this setup's minimum 80.00" in selection.detail


def test_a_setup_band_narrower_than_the_union_declines_at_the_setup_stage():
    """0.70 clears the union band [0.20, 0.80] and fails DIRECTIONAL_3R's
    [0.20, 0.50]. Section 3's reason for that comparison is VOLATILITY_BAND."""
    cfg = config()
    low, high = vol_percentile_band(cfg)
    assert (low, high) == (0.20, 0.80)
    selection = selection_for(cfg, achievable_rr=3.00, atr_percentile=0.70)
    assert selection.wait_reason is WaitReason.VOLATILITY_BAND
    assert "outside this setup's band [0.20, 0.50]" in selection.detail


def test_a_regime_barred_by_the_determined_setup_is_no_setup_match():
    """The brief's "NO setup matches": the structure named a setup this
    configuration does not offer for this bar."""
    cfg = config(
        setups=setup_table(
            DIRECTIONAL_3R={"allowed_regimes": (Regime.TRENDING_DOWN,)}
        )
    )
    selection = selection_for(cfg, achievable_rr=3.00, regime=Regime.TRENDING_UP)
    assert selection.wait_reason is WaitReason.NO_SETUP_MATCH
    assert "allowed_regimes" in selection.detail


def test_required_order_flow_confirmation_distinguishes_absent_from_opposed():
    """Two different facts, two different reasons.

    An UNAVAILABLE tape is COMPONENT_DISABLED -- nothing to confirm with, and
    section 5 accepts no proxy. A MEASURED tape that disagrees is
    ORDERFLOW_CONFLICT. Collapsing them would report "order flow disagreed"
    about a tape nobody observed.
    """
    cfg = config()
    absent = dict(all_bullish(0.9))
    absent[Component.ORDER_FLOW] = unavailable(Component.ORDER_FLOW)
    unmeasured = selection_for(cfg, achievable_rr=1.50, table=absent)
    assert unmeasured.wait_reason is WaitReason.COMPONENT_DISABLED
    assert "no proxy" in unmeasured.detail

    neutral = dict(all_bullish(0.9))
    neutral[Component.ORDER_FLOW] = component(Component.ORDER_FLOW, 0.9, 0)
    not_agreeing = selection_for(cfg, achievable_rr=1.50, table=neutral)
    assert not_agreeing.wait_reason is WaitReason.ORDERFLOW_CONFLICT
    assert "does not agree" in not_agreeing.detail


def test_every_setup_records_its_first_failing_gate_for_rejection_analysis():
    """So a report can answer "how close was this bar to a 2R" without the
    selector ever acting on a setup the structure did not name.

    NOTE on the module docstring: it says `rejections` records "every ENABLED
    setup's" first-failing gate, and a disabled or unconfigured setup is
    recorded too (with gate "enabled" / "configured"). The behaviour is the more
    informative of the two and the docstring is what is inaccurate; recorded
    here rather than changed.
    """
    cfg = config()
    selection = selection_for(cfg, achievable_rr=1.50, flow_score=50.0)
    assert selection.setup is SetupType.SCALP_1R
    by_setup = {r.setup: r for r in selection.rejections}
    # SCALP_1R accepted the bar, so it has no rejection
    assert SetupType.SCALP_1R not in by_setup
    assert by_setup[SetupType.SETUP_2R].gate == "min_reward_risk"
    assert by_setup[SetupType.SETUP_2R].wait_reason is WaitReason.RR_TOO_LOW
    assert by_setup[SetupType.DIRECTIONAL_3R].gate == "min_reward_risk"
    assert all(r.selected_by_band is False for r in selection.rejections)

    # the band's own setup is flagged when it is the one that failed
    declined = selection_for(cfg, achievable_rr=3.00, flow_score=50.0)
    flagged = [r for r in declined.rejections if r.selected_by_band]
    assert [r.setup for r in flagged] == [SetupType.DIRECTIONAL_3R]
    assert flagged[0].gate == "min_flow_score"

    # a setup absent from config.setups is recorded as such, not silently skipped
    partial = config(setups={SetupType.SCALP_1R: setup_config(SetupType.SCALP_1R)})
    thin = selection_for(partial, achievable_rr=1.50)
    assert {r.setup: r.gate for r in thin.rejections} == {
        SetupType.SETUP_2R: "configured",
        SetupType.DIRECTIONAL_3R: "configured",
    }


def test_the_union_helpers_ask_the_weakest_honest_question():
    """Each pre-selection gate's question is "could ANY enabled setup take this".

    Hand-computed from this file's local setups: regimes are the union
    {TRENDING_UP, TRENDING_DOWN}; the band is (min of the mins, max of the
    maxes) = (0.20, 0.80); the score floor is min(30, 40, 80) = 30; the
    reward/risk floor is max(levels 1.0, min(1.0, 1.8, 2.7)) = 1.0.
    """
    cfg = config()
    assert tradable_regimes(cfg) == frozenset(TRADABLE_REGIMES)
    assert Regime.UNKNOWN not in tradable_regimes(cfg)
    assert vol_percentile_band(cfg) == (0.20, 0.80)
    assert min_flow_score_required(cfg) == 30.0
    assert min_reward_risk_floor(cfg) == 1.0

    # disabling the lowest setup raises three of the four
    without_scalp = config(setups=setup_table(SCALP_1R={"enabled": False}))
    assert min_flow_score_required(without_scalp) == 40.0
    assert min_reward_risk_floor(without_scalp) == 1.8
    assert vol_percentile_band(without_scalp) == (0.20, 0.80)


def test_the_union_band_is_a_convex_hull_and_over_admits_disjoint_bands():
    """REPORTED: `vol_percentile_band` is the hull, not the set union.

    With SCALP_1R banded [0.00, 0.30] and DIRECTIONAL_3R banded [0.70, 1.00],
    the set of percentiles some setup accepts is two intervals and the helper
    returns the single interval [0.00, 1.00] that spans them. A bar at 0.50 is
    therefore admitted by the momentum gate although NO setup would take it,
    and is declined one stage later by the setup it was measured into.

    This is harmless on the reason -- the setup stage reports the same
    VOLATILITY_BAND section 3 assigns to that comparison -- and it costs only
    the stage attribution in section 12's histogram. It is reported because the
    docstring says "UNION" and the implementation is `(min of the mins, max of
    the maxes)`, and the two differ exactly when the bands do not overlap.
    """
    cfg = config(
        setups=setup_table(
            SCALP_1R={"min_vol_percentile": 0.00, "max_vol_percentile": 0.30},
            SETUP_2R={"enabled": False},
            DIRECTIONAL_3R={"min_vol_percentile": 0.70, "max_vol_percentile": 1.00},
        )
    )
    assert vol_percentile_band(cfg) == (0.00, 1.00)
    # 0.50 is in no enabled setup's band
    for setup in cfg.enabled_setups():
        assert not (setup.min_vol_percentile <= 0.50 <= setup.max_vol_percentile)

    decision = decide(
        cfg, all_bullish(0.9), vector=features(achievable_rr=1.50, atr_percentile=0.50)
    )
    assert decision.signal.action is SignalAction.WAIT
    # admitted by the momentum gate, declined by the setup
    momentum = [r for r in decision.gate_results if r.stage is GateStage.MOMENTUM]
    assert momentum and momentum[0].passed is True
    assert decision.gate_results[-1].stage is GateStage.SETUP
    assert decision.signal.wait_reason is WaitReason.VOLATILITY_BAND


def test_select_does_not_check_the_side_against_the_structure_it_came_from():
    """REPORTED: `SetupError`'s docstring claims a check that does not exist.

    It says the error covers "a caller that passed a side the structure did not
    name". No such comparison is made: `select` takes `side` as an argument and
    never reads `level_direction`. A caller handing it the wrong side gets a
    DECLINE, not an error -- here NO_STRUCTURE, because
    `require_structure_confirmation` finds the STRUCTURE component pointing the
    other way.

    Not reachable from `signals/engine.py`, where the side comes from the
    structure gate and nowhere else, so the consequence today is a docstring
    that overstates what is enforced. Worth closing before Phase 7 gives
    `select` a second caller.

    MY FIRST EXPECTATION HERE WAS WRONG. I expected NO_STRUCTURE, from
    `require_structure_confirmation`. `_evaluate`'s gate order runs
    `require_orderflow_confirmation` first, and ORDER_FLOW disagrees with the
    wrong side too, so ORDERFLOW_CONFLICT is the correct first failure. The
    second block turns that flag off and reaches the structure one -- which is
    also the only route by which the (SETUP, NO_STRUCTURE) pair is reachable at
    all, and it is a route the engine never takes.
    """
    cfg = config()
    wrong_side = select(
        cfg,
        features(level_direction=DIRECTION_SUPPORT, achievable_rr=1.50),
        side=Side.SHORT,  # the structure named LONG
        regime=regime_state(),
        flow_score=90.0,
        components=all_bullish(0.9),
    )
    assert wrong_side.matched is False
    assert wrong_side.wait_reason is WaitReason.ORDERFLOW_CONFLICT
    assert "does not agree with a SHORT" in wrong_side.detail

    without_orderflow = config(
        setups=setup_table(
            SCALP_1R={"require_orderflow_confirmation": False},
            SETUP_2R={"require_orderflow_confirmation": False},
            DIRECTIONAL_3R={"require_orderflow_confirmation": False},
        )
    )
    structure_flag = select(
        without_orderflow,
        features(level_direction=DIRECTION_SUPPORT, achievable_rr=1.50),
        side=Side.SHORT,
        regime=regime_state(),
        flow_score=90.0,
        components=all_bullish(0.9),
    )
    assert structure_flag.wait_reason is WaitReason.NO_STRUCTURE
    assert "STRUCTURE direction +1 does not agree with a SHORT" in structure_flag.detail


def test_the_band_table_and_the_setup_map_agree_at_import():
    """One copy of 14.6's numbers in the codebase.

    `SETUP_BY_RR_CLASS`'s keys are checked against `levels.SETUP_RR_BANDS` at
    import time, so a band added to 14.6 without a `SetupType` is an
    import-time error rather than a bar declined with NO_SETUP_MATCH.
    """
    assert {cls for _, cls in SETUP_RR_BANDS} == set(SETUP_BY_RR_CLASS)
    assert SETUP_BY_RR_CLASS == {
        1.0: SetupType.SCALP_1R,
        2.0: SetupType.SETUP_2R,
        3.0: SetupType.DIRECTIONAL_3R,
    }
    assert SETUP_RR_BANDS == ((2.7, 3.0), (1.8, 2.0), (1.0, 1.0))


# ---------------------------------------------------------------------------
# the trade the three modules produce together
# ---------------------------------------------------------------------------


def test_the_plan_is_the_setup_s_r_against_the_structural_stop():
    """End to end, with every number hand-computed.

    entry 100.00, structural stop 99.00 -> stop distance 1.00 = 4 ticks at 0.25
    achievable_rr 1.50 -> SCALP_1R, which plans 1.0R -> target 101.00
    risk 0.5% of 100,000 = $500; $20 per point per contract * 1.00 = $20/unit
    -> size 25, risk_dollars 25 * 1.00 * 20 = $500
    """
    cfg = config()
    decision = decide(cfg, all_bullish(0.9), vector=features(achievable_rr=1.50))
    signal, plan = decision.signal, decision.plan

    assert signal.action is SignalAction.LONG
    assert signal.setup is SetupType.SCALP_1R
    assert signal.flow_score == pytest.approx(90.0, abs=1e-9)
    assert plan is not None
    assert plan.entry_price == 100.00
    assert plan.stop_price == 99.00
    assert plan.stop_distance == pytest.approx(1.00, abs=1e-9)
    assert plan.stop_ticks == pytest.approx(4.0, abs=1e-9)
    assert plan.target_price == pytest.approx(101.00, abs=1e-9)
    assert plan.planned_r_multiple == pytest.approx(1.0, abs=1e-9)
    assert plan.achievable_rr == 1.50
    assert plan.size == 25.0
    assert plan.risk_dollars == pytest.approx(500.0, abs=1e-6)
    assert plan.max_hold_bars == 12

    intent = plan.to_trade_intent()
    assert intent.planned_r_multiple == pytest.approx(1.0, abs=1e-9)
    assert intent.risk_dollars == pytest.approx(500.0, abs=1e-6)


def test_a_three_r_structure_plans_three_r_and_not_the_measured_nine():
    """"Do not force every trade into the same target" cuts both ways: the plan
    is the setup's R, and a 9R measurement is not planned as 9R."""
    cfg = config()
    decision = decide(
        cfg,
        all_bullish(0.9),
        vector=features(achievable_rr=9.00, level_target_price=109.00),
    )
    plan = decision.plan
    assert decision.signal.setup is SetupType.DIRECTIONAL_3R
    assert plan is not None
    assert plan.planned_r_multiple == pytest.approx(3.0, abs=1e-9)
    assert plan.target_price == pytest.approx(103.00, abs=1e-9)
    assert plan.achievable_rr == 9.00
    assert plan.plans_beyond_measured_target is False


def test_the_short_side_mirrors_the_geometry():
    """A resistance level gives a SHORT whose stop is above entry."""
    cfg = config()
    table = components(
        options_flow=component(Component.OPTIONS_FLOW, 0.9, -1),
        order_flow=component(Component.ORDER_FLOW, 0.9, -1),
        structure=component(Component.STRUCTURE, 0.9, -1),
        liquidity=component(Component.LIQUIDITY, 0.9, 0),
        vol_momentum=component(Component.VOL_MOMENTUM, 0.9, -1),
    )
    decision = decide(
        cfg,
        table,
        vector=features(
            level_direction=DIRECTION_RESISTANCE,
            level_stop_price=101.00,
            level_target_price=98.50,
            achievable_rr=1.50,
        ),
    )
    plan = decision.plan
    assert decision.signal.action is SignalAction.SHORT
    assert plan is not None
    assert plan.side is Side.SHORT
    assert plan.stop_price == 101.00
    assert plan.target_price == pytest.approx(99.00, abs=1e-9)
    assert plan.planned_r_multiple == pytest.approx(1.0, abs=1e-9)


def test_a_missing_structural_stop_is_a_wait_and_not_an_atr_multiple():
    """The stop is where the thesis is falsified, never a width chosen to make
    a size work. `stop_atr_multiple` is never consulted."""
    cfg = config()
    decision = decide(
        cfg, all_bullish(0.9), vector=features(missing=("level_stop_price",))
    )
    assert decision.signal.action is SignalAction.WAIT
    assert decision.gate_results[-1].stage is GateStage.RISK
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert "No ATR-multiple stop is substituted" in decision.signal.wait_detail


def test_every_wait_in_this_file_s_scenes_carries_a_reason_and_a_narrative():
    """Section 12 needs the reason; a human reading one bar needs the words."""
    cfg = config()
    scenes = (
        dict(table=bars_only_components(0.9), vector=features()),
        dict(table=all_bullish(0.9), vector=features(atr_percentile=0.95)),
        dict(table=all_bullish(0.9), vector=features(achievable_rr=0.50)),
        dict(table=all_bullish(0.1), vector=features()),
        dict(
            table=all_bullish(0.9),
            vector=features(structure_gate_passed=0.0, level_gate_reason=GATE_NO_ZONE),
        ),
    )
    for scene in scenes:
        decision = decide(cfg, scene["table"], vector=scene["vector"])
        signal = decision.signal
        assert signal.action is SignalAction.WAIT
        assert signal.wait_reason is not None
        assert signal.wait_detail
        assert signal.reasons
        assert signal.wait_reason in WAIT_REASONS_BY_STAGE[
            decision.gate_results[-1].stage
        ]


def test_no_module_under_test_reads_the_research_targets_or_the_hypotheses():
    """"Do not optimize for a desired win rate." Checked on the AST here too,
    not only by tests/unit/test_no_target_leakage.py, because these three
    modules are where a threshold would be tempting."""
    root = Path(__file__).resolve().parents[2] / "flow_model"
    for relative in ("signals/flow_score.py", "signals/gates.py", "signals/setups.py"):
        tree = ast.parse((root / relative).read_text())
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert "research_targets" not in attributes, relative
        assert "hypotheses" not in attributes, relative


def test_assess_warmup_reads_only_bar_counts_and_feed_presence():
    """A gate may not touch market data. `assess_warmup` is the one function in
    gates.py handed a view, and the only members it may use are `bar_count` and
    `has_feed` -- so a stub exposing exactly those is enough to drive it."""

    class CountingView:
        symbol = SYMBOL
        now = TS

        def __init__(self) -> None:
            self.calls: list[str] = []

        def bar_count(self) -> int:
            self.calls.append("bar_count")
            return 42

        def has_feed(self, feed) -> bool:
            self.calls.append("has_feed")
            return feed is Feed.BARS

    class Computer:
        name = "fake"
        warmup_bars = 100
        required_feeds = frozenset({Feed.BARS, Feed.TICK_AGGREGATE})

        def feeds_available(self, view) -> bool:
            return all(view.has_feed(f) for f in self.required_feeds)

    view = CountingView()
    report = assess_warmup((Computer(),), view)
    assert report.bars_available == 42
    assert report.computers[0].feeds_present is False
    assert report.computers[0].absent_feeds == (Feed.TICK_AGGREGATE,)
    assert report.blocking == ()
    assert set(view.calls) == {"bar_count", "has_feed"}
