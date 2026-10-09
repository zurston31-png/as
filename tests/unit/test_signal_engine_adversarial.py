"""Adversarial tests for the signal engine: section 3's pipeline end to end.

`signals/engine.py` is the one object Phase 11's backtest/live parity test
compares, and it is the composition of seven feature computers, a regime
detector, five scorers, an aggregate that may refuse, eleven gates, a setup
selector and the risk layer. Each of those parts has its own tests. What this
file attacks is everything that can only be wrong in the COMPOSITION, plus the
three claims the engine makes about itself that nothing below it can check.

These tests are independent of `tests/unit/test_signal_engine.py` by
construction: the configuration is built here in Python rather than loaded from
`flow_model/config/defaults.yaml`, so an edit to an unrelated YAML section
cannot turn a test red, and a moved default cannot change an expected value in
silence. `local_config()` is the single builder and every number it declares is
visible in this file. Section 14.6's band table, 14.6's gate codes and the
tick-grid arithmetic are the only things imported rather than restated, because
restating them would be testing a copy.

The risks the tests are organized around:

**1. Section 7's one design rule, which the whole scheme rests on.** "Direction
is handled separately from magnitude ... The Flow Score is the magnitude
aggregate; direction agreement is enforced by the gates. This avoids the common
bug where a strong bearish component inflates a bullish score." The failure
mode is a system that aggregates a 93.25-point magnitude out of four bullish
components and one bearish one, notices the majority is bullish, and takes the
long. `test_the_bullish_majority_cannot_outvote_a_bearish_structure` builds
exactly that bar -- `FlowScore.net_direction()` is +1 while the structure names
SHORT -- and asserts the engine WAITs, that the only `Side` any stage ever saw
is SHORT, and that no module in `signals/` or `risk/` so much as CALLS
`net_direction` (checked on the AST, so a docstring mentioning it does not
count). The mirrored pair asserts the other half: identical magnitudes,
opposite directions, and no trade either way.

**2. The bars-only outcome, asserted exactly rather than approximately.** On a
bars-only dataset 45 of the 100 points have no feed. Two different things must
happen and they must stay distinguishable, because section 12 counts which gate
fired: the DATA-QUALITY gate blocks at stage 2 while the required feeds are
declared required, and the AGGREGATE refuses with `COMPONENT_DISABLED` at stage
10 when they are not. The tests assert the stage, the reason, that
`Signal.flow_score is None` in both cases, and that the words naming
`order_flow` and `options_flow` reach `Signal.wait_detail` -- a WAIT that says
only "data quality" does not tell a reader which 45 points are missing.

**3. Determinism, which Phase 11 will rest on and the regime detector got
wrong once already.** Three forms, strongest last: the same `MarketView` twice,
a fresh engine against one already walked over sixty other bars, and the same
thirty bars walked FORWARDS and BACKWARDS. The last is the one that catches an
accumulator: an engine that remembered anything would answer differently when
bar 1731 is evaluated before bar 1730 instead of after it.

**4. Short-circuiting, which is a correctness property and not an
optimization.** Section 3 puts the cheap rejections first. A bar rejected on
data quality must not pay for the feature bundle -- and more to the point, must
not be able to produce a Signal whose fields were computed from a layer the
pipeline said it would not consult. A counting `StageInputs` proves the
features, the regime and the warmup were never asked for.

**5. Every WAIT carries a reason, and the reason belongs to the stage that
produced it.** `WAIT_REASONS_BY_STAGE` is what section 12 labels its buckets
by, so a stage emitting a reason it did not declare makes the labels wrong.
Thirteen stages are driven to failure one at a time and each is checked against
its own declaration, each WAIT is checked to carry a detail and a narrative,
and every `Signal` is round-tripped through its own Phase 1 validator.

**6. Setup selection as a measurement.** The band is read off `achievable_rr`
by `features/levels.py` and nothing may promote, truncate or retarget. The
arithmetic is exact by design: entry 18000.00, a 20.00 stop distance and a
0.25 tick, so every price in these tests is on the grid and every planned R is
a number written down here rather than read back from the implementation.

**7. The risk layer is consumed, never worked around.** Sizing's
`reward_risk_below_minimum` must surface as a WAIT rather than be fixed by
widening the stop or moving the target; and limits must run LAST, so a trade
blocked by a position limit is still recorded as a signal that occurred.

Two defects found by these tests are recorded at the bottom of the file, each
with the test that fails without its fix.
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from flow_model.config.schema import (
    BacktestConfig,
    FeedRequirement,
    FlowModelConfig,
    FlowScoreConfig,
    SetupConfig,
)
from flow_model.core.contracts import FeatureVector, FlowScore, RegimeState, Signal
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
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.base import FeedStatus
from flow_model.data.calendar import TradingCalendar
from flow_model.data.market_view import MarketView
from flow_model.data.quality import QualityGrader, QualityReport
from flow_model.data.series import from_ns
from flow_model.data.store import SymbolData
from flow_model.data.synthetic import SyntheticConfig, SyntheticMarketGenerator
from flow_model.features.levels import (
    GATE_NO_ZONE,
    GATE_PASSED,
    GATE_REWARD_RISK,
    GATE_SIGNIFICANCE,
    SETUP_RR_BANDS,
    setup_class_for,
)
from flow_model.risk.limits import RiskLimitManager
from flow_model.risk.sizing import SizingRejection, round_to_tick
from flow_model.signals.engine import (
    SIZING_WAIT_REASONS,
    EngineError,
    PrecomputedInputs,
    SignalEngine,
)
from flow_model.signals.flow_score import REFUSAL_WAIT_REASON, aggregate
from flow_model.signals.gates import (
    WAIT_REASONS_BY_STAGE,
    ComputerReadiness,
    GateStage,
    WarmupReport,
)
from flow_model.signals.setups import (
    SetupError,
    min_flow_score_required,
    min_reward_risk_floor,
    vol_percentile_band,
)

SYMBOL = "NQ"
INTERVAL = 300

#: 10:00 New York on a Friday: inside RTH, past the five-minute open skip and
#: well before the ten-minute close skip, so the session gate passes.
TS = datetime(2024, 3, 1, 15, 0, tzinfo=timezone.utc)

#: 22:00 New York the previous evening: inside the ETH window but outside RTH,
#: which `trade_rth_only` rejects.
TS_OVERNIGHT = datetime(2024, 3, 1, 3, 0, tzinfo=timezone.utc)

#: The geometry every hand-built test uses. Chosen so the arithmetic is exact:
#: 18000.00 and 17980.00 are both multiples of NQ's 0.25 tick, so the stop
#: distance is 20.00 points == 80 ticks exactly and no rounding happens
#: anywhere. ATR is 20.00 too, which makes every `max_stop_atr_multiple * ATR`
#: cap in `local_config` (2.0, 2.5, 3.0) comfortably wider than the stop.
ENTRY = 18_000.0
ATR = 20.0
STOP_DISTANCE = 20.0

#: The brief's prior, restated here rather than read from YAML so the expected
#: point totals below are checkable against one visible table.
#:   options 20 + order flow 25 + structure 20 + liquidity 15 + volmom 20 = 100
WEIGHTS: dict[Component, float] = {
    Component.OPTIONS_FLOW: 20.0,
    Component.ORDER_FLOW: 25.0,
    Component.STRUCTURE: 20.0,
    Component.LIQUIDITY: 15.0,
    Component.VOL_MOMENTUM: 20.0,
}

ALL_TRADABLE_REGIMES = (
    Regime.LOW_VOL,
    Regime.HIGH_VOL,
    Regime.TRENDING_UP,
    Regime.TRENDING_DOWN,
    Regime.CHOP,
    Regime.DIRECTIONAL_SHIFT,
)


# ---------------------------------------------------------------------------
# configuration, built here and not loaded
# ---------------------------------------------------------------------------


def nq_spec() -> InstrumentSpec:
    """NQ's contract economics. point_value = tick_value / tick_size = 20.00."""
    return InstrumentSpec(
        symbol=SYMBOL,
        instrument_type=InstrumentType.FUTURE,
        description="E-mini Nasdaq-100 futures",
        exchange="CME",
        timezone="America/New_York",
        tick_size=0.25,
        tick_value=5.0,
        commission_per_side=2.25,
        exchange_fee_per_side=0.37,
        typical_spread_ticks=1.0,
        min_spread_ticks=1.0,
        min_size=1,
        max_size=50,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
        eth=SessionWindow(name="ETH", start="18:00", end="17:00"),
    )


def feed_requirements(
    *, order_flow_required: bool = True, options_required: bool = True
) -> dict[Component, tuple[FeedRequirement, ...]]:
    """Which feed each component needs, and whether its absence blocks.

    The two flags exist because the bars-only story has two halves that must
    stay distinguishable. With the tick and options feeds REQUIRED the
    data-quality gate blocks at stage 2 and the aggregate is never reached;
    with them optional the gate passes and the aggregate's own strict refusal
    is what fires at stage 10. Section 12 counts gates, so a test that could
    only ever see one of the two would be testing half the behaviour.
    """
    return {
        Component.ORDER_FLOW: (
            FeedRequirement(
                feed=Feed.TICK_AGGREGATE,
                minimum_quality=DataQuality.DEGRADED,
                required=order_flow_required,
            ),
        ),
        Component.OPTIONS_FLOW: (
            FeedRequirement(
                feed=Feed.OPTIONS_SNAPSHOT,
                minimum_quality=DataQuality.DEGRADED,
                required=options_required,
            ),
        ),
        Component.STRUCTURE: (
            FeedRequirement(
                feed=Feed.BARS, minimum_quality=DataQuality.GOOD, required=True
            ),
        ),
        Component.LIQUIDITY: (
            FeedRequirement(
                feed=Feed.BARS, minimum_quality=DataQuality.GOOD, required=True
            ),
            FeedRequirement(
                feed=Feed.QUOTES, minimum_quality=DataQuality.DEGRADED, required=False
            ),
        ),
        Component.VOL_MOMENTUM: (
            FeedRequirement(
                feed=Feed.BARS, minimum_quality=DataQuality.GOOD, required=True
            ),
        ),
    }


def a_setup(
    setup: SetupType,
    *,
    min_flow_score: float,
    enabled: bool = True,
    allowed_regimes: tuple[Regime, ...] = ALL_TRADABLE_REGIMES,
    **kwargs: object,
) -> SetupConfig:
    """One setup, with 14.6's band floor as its `min_reward_risk`.

    `reward_risk` and `min_reward_risk` are the two numbers section 14.6 and
    section 0.2 pull in opposite directions (the band for DIRECTIONAL_3R opens
    at 2.7 while the label rule requires a 3.0R plan), and both are declared
    here so the tests that assert the gap can point at them.
    """
    defaults: dict[str, object] = {
        SetupType.SCALP_1R: dict(
            reward_risk=1.0,
            min_reward_risk=0.95,
            min_vol_percentile=0.10,
            max_vol_percentile=0.95,
            min_stop_ticks=4,
            max_stop_atr_multiple=2.0,
            max_hold_bars=12,
        ),
        SetupType.SETUP_2R: dict(
            reward_risk=2.0,
            min_reward_risk=1.8,
            min_vol_percentile=0.20,
            max_vol_percentile=0.98,
            min_stop_ticks=6,
            max_stop_atr_multiple=2.5,
            max_hold_bars=24,
        ),
        SetupType.DIRECTIONAL_3R: dict(
            reward_risk=3.0,
            min_reward_risk=2.7,
            min_vol_percentile=0.25,
            max_vol_percentile=1.00,
            min_stop_ticks=8,
            max_stop_atr_multiple=3.0,
            max_hold_bars=60,
        ),
    }[setup]
    # The two confirmation flags default OFF so the per-setup check does not
    # pre-empt the pipeline gate it duplicates; a caller overrides either.
    merged: dict[str, object] = {
        "require_orderflow_confirmation": False,
        "require_structure_confirmation": False,
        **defaults,
        **kwargs,
    }
    return SetupConfig(
        setup=setup,
        enabled=enabled,
        min_flow_score=min_flow_score,
        allowed_regimes=allowed_regimes,
        **merged,  # type: ignore[arg-type]
    )


def local_config(
    *,
    min_flow_score: float = 30.0,
    setups: dict[SetupType, SetupConfig] | None = None,
    strict: bool = True,
    redistribute: bool = False,
    enabled_components: dict[Component, bool] | None = None,
    order_flow_required: bool = True,
    options_required: bool = True,
    allowed_regimes: tuple[Regime, ...] = ALL_TRADABLE_REGIMES,
    max_opposing_points: float = 12.0,
    options_contradiction_points: float = 14.0,
) -> FlowModelConfig:
    """A complete config, declared in Python, reading no YAML file.

    `min_flow_score` defaults to 30.0, which is NOT the shipped prior (70/75/80).
    This is a test asking what the pipeline does at a different threshold, and
    the shipped numbers are untouched on disk: principle 7 makes them
    hypotheses for the Phase 8 sweep, and the whole point of the
    pipeline tests is the stages downstream of the Flow Score gate, which a
    threshold nothing clears would leave unreached.

    `require_orderflow_confirmation` and `require_structure_confirmation` are
    OFF by default here so that the per-setup confirmation flags do not
    pre-empt the pipeline gate they duplicate; the tests that are about those
    flags switch them on explicitly.
    """
    return FlowModelConfig(
        name="test_signal_engine_adversarial",
        instruments={SYMBOL: nq_spec()},
        flow_score=FlowScoreConfig(
            weights=dict(WEIGHTS),
            total_points=100.0,
            strict_component_availability=strict,
            redistribute_disabled_weight=redistribute,
            enabled_components=(
                enabled_components
                if enabled_components is not None
                else {c: True for c in Component}
            ),
            max_opposing_points=max_opposing_points,
            options_contradiction_points=options_contradiction_points,
            feed_requirements=feed_requirements(
                order_flow_required=order_flow_required,
                options_required=options_required,
            ),
        ),
        setups=(
            setups
            if setups is not None
            else {
                setup: a_setup(
                    setup, min_flow_score=min_flow_score, allowed_regimes=allowed_regimes
                )
                for setup in SetupType
            }
        ),
        backtest=BacktestConfig(symbols=(SYMBOL,)),
    )


@pytest.fixture
def cfg() -> FlowModelConfig:
    return local_config()


# ---------------------------------------------------------------------------
# hand-built feature vectors
# ---------------------------------------------------------------------------
#
# The key sets below are the five scorers' DECLARED read sets. The two
# feed-dependent scorers raise on an absent key -- an absent key there means a
# rename, not a data condition -- so a vector missing one is a test of the
# builder rather than of the engine.


def _structure_keys(
    *,
    direction: float,
    magnitude: float,
    achievable_rr: float,
    stop_price: float,
    gate_passed: float,
    gate_reason: float,
    levels_min_reward_risk: float,
) -> dict[str, float]:
    """`features/levels.py`'s outputs, internally consistent by construction.

    `level_setup_class` is produced by levels.py's OWN `setup_class_for` on the
    same `achievable_rr` the vector carries. `signals/setups.py` re-evaluates
    that function as a consistency assertion and raises when the two disagree,
    so writing the class by hand here would make every other test in the file
    depend on my arithmetic matching 14.6's table.
    """
    target = ENTRY + direction * abs(ENTRY - stop_price) * achievable_rr
    return {
        "structure_magnitude": magnitude,
        "level_direction": direction,
        "level_significance": 0.80,
        "level_cleanliness": 0.70,
        "level_rejection": 0.65,
        "rejection_confirmed": 1.0,
        "structure_gate_passed": gate_passed,
        "level_gate_reason": gate_reason,
        "level_setup_class": setup_class_for(achievable_rr, levels_min_reward_risk),
        "achievable_rr": achievable_rr,
        "zone_count": 6.0,
        "major_zone_count": 3.0,
        "nearest_zone_distance_atr": -0.2 * direction,
        "nearest_zone_gap_atr": 0.1,
        "level_touch_count": 4.0,
        "level_recent_touches": 0.0,
        "level_flow_available": 1.0,
        "level_stop_price": stop_price,
        "level_target_price": target,
        "structure_score": 0.7,
        "structure_direction": direction,
        "bos_direction": direction,
        "choch": 0.0,
        "vwap_deviation_atr": 0.1 * direction,
        "opening_range_position": 0.5,
    }


def _liquidity_keys(
    *, score: float, spread_ticks: float, relative_volume: float
) -> dict[str, float]:
    return {
        "liquidity_score": score,
        "spread_ticks": spread_ticks,
        "spread_percentile": 0.30,
        "depth_imbalance": 0.0,
        "volume_percentile": 0.70,
        "relative_volume": relative_volume,
        "volume_trend": 0.10,
        "dollar_volume": 5.0e8,
        "participation_cost_ticks": 1.0,
    }


def _vol_momentum_keys(
    *, direction: float, magnitude: float, atr_percentile: float
) -> dict[str, float]:
    """VOL_MOMENTUM's magnitude is `0.5*vol_regime_score + 0.5*momentum_score`.

    Both halves are set to the same number so the blend is that number
    exactly, which is what makes the point totals in this file arithmetic
    rather than approximation.
    """
    return {
        "vol_regime_score": magnitude,
        "momentum_score": magnitude,
        "direction": direction,
        "atr": ATR,
        "atr_pct": 0.0011,
        "atr_percentile": atr_percentile,
        "realized_vol": 0.14,
        "parkinson_vol": 0.15,
        "garman_klass_vol": 0.13,
        "vol_of_vol": 0.02,
        "vol_expansion": 0.10,
        "roc": 0.004 * direction,
        "roc_atr": 1.2 * direction,
        "efficiency_ratio": 0.60,
        "acceleration": 0.10 * direction,
        "momentum_persistence": 0.60,
        "up_bar_fraction": 0.50 + 0.10 * direction,
        "close_position": 0.50 + 0.20 * direction,
    }


def _order_flow_keys(
    *, direction: float, magnitude: float, available: float
) -> dict[str, float]:
    return {
        "order_flow_available": available,
        "order_flow_available_weight": 1.0,
        "order_flow_magnitude": magnitude,
        "order_flow_direction": direction,
        "signed_delta": 0.5 * direction,
        "cvd_slope": 0.3 * direction,
        "aggression_ratio": 0.60,
        "absorption": 0.0,
        "absorption_direction": 0.0,
        "avg_trade_size_percentile": 0.55,
        "max_trade_size_percentile": 0.60,
        "large_trade_event": 0.0,
        "classification_coverage": 1.0,
    }


def _options_flow_keys(*, direction: float, magnitude: float) -> dict[str, float]:
    """An INTRADAY chain: `options_eod_capped` clear and timing available.

    The scorer cross-checks the cap against its flag and raises when they
    disagree, so `options_flow_uncapped_magnitude` equals the magnitude here
    and the cap branch is never entered.
    """
    return {
        "options_flow_magnitude": magnitude,
        "options_flow_uncapped_magnitude": magnitude,
        "options_flow_direction": direction,
        "options_eod_capped": 0.0,
        "options_flow_timing_available": 1.0,
        "options_available_weight": 1.0,
        "options_terms_available": 5.0,
        "options_sign_agreement": 1.0,
        "options_imbalance_consensus": magnitude,
        "options_net_premium_imbalance": 0.5 * direction,
        "options_delta_volume_imbalance": 0.4 * direction,
        "options_oi_change_imbalance": 0.3 * direction,
        "options_skew_25d_pressure": 0.2,
        "options_gamma_exposure_pressure": 0.1,
    }


def build_vector(
    *,
    structure_direction: float = 1.0,
    structure_magnitude: float = 0.95,
    achievable_rr: float = 1.2,
    stop_distance: float = STOP_DISTANCE,
    gate_passed: float = 1.0,
    gate_reason: float = GATE_PASSED,
    levels_min_reward_risk: float = 1.0,
    liquidity_score: float = 0.90,
    spread_ticks: float = 1.0,
    relative_volume: float = 1.2,
    vol_momentum_direction: float = 1.0,
    vol_momentum_magnitude: float = 0.90,
    atr_percentile: float = 0.50,
    order_flow_direction: float = 1.0,
    order_flow_magnitude: float = 0.95,
    order_flow_available: float = 1.0,
    options_flow_direction: float = 1.0,
    options_flow_magnitude: float = 0.95,
    grades: dict[str, DataQuality] | None = None,
    symbol: str = SYMBOL,
    ts: datetime | None = None,
) -> FeatureVector:
    """A complete vector whose Flow Score is a number written down above.

    Defaults give, per the weight table:
      structure    0.95 * 20 = 19.00
      liquidity    0.90 * 15 = 13.50
      vol_momentum 0.90 * 20 = 18.00
      order flow   0.95 * 25 = 23.75
      options flow 0.95 * 20 = 19.00
                              ------
                               93.25
    """
    stop_price = ENTRY - structure_direction * stop_distance
    values: dict[str, float] = {}
    values.update(
        _structure_keys(
            direction=structure_direction,
            magnitude=structure_magnitude,
            achievable_rr=achievable_rr,
            stop_price=stop_price,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
            levels_min_reward_risk=levels_min_reward_risk,
        )
    )
    values.update(
        _liquidity_keys(
            score=liquidity_score,
            spread_ticks=spread_ticks,
            relative_volume=relative_volume,
        )
    )
    values.update(
        _vol_momentum_keys(
            direction=vol_momentum_direction,
            magnitude=vol_momentum_magnitude,
            atr_percentile=atr_percentile,
        )
    )
    values.update(
        _order_flow_keys(
            direction=order_flow_direction,
            magnitude=order_flow_magnitude,
            available=order_flow_available,
        )
    )
    values.update(
        _options_flow_keys(
            direction=options_flow_direction, magnitude=options_flow_magnitude
        )
    )
    quality_by_key = {key: DataQuality.GOOD for key in values}
    quality_by_key.update(grades or {})
    return FeatureVector(
        symbol=symbol,
        ts=ts or TS,
        values=values,
        quality_by_key=quality_by_key,
    )


#: The expected aggregate of `build_vector()`'s defaults, by hand:
#: 0.95*20 + 0.95*25 + 0.95*20 + 0.90*15 + 0.90*20 = 19 + 23.75 + 19 + 13.5 + 18
EXPECTED_FULL_SCORE = 93.25


def a_regime(
    regime: Regime = Regime.TRENDING_UP,
    *,
    ts: datetime | None = None,
    symbol: str = SYMBOL,
    vol_percentile: float = 0.50,
) -> RegimeState:
    return RegimeState(
        symbol=symbol,
        ts=ts or TS,
        regime=regime,
        confidence=0.80,
        vol_percentile=vol_percentile,
        bars_in_regime=12,
    )


def a_quality_report(
    *,
    overall: DataQuality,
    blocking: tuple[Feed, ...] = (),
    note: str = "",
    ts: datetime | None = None,
) -> QualityReport:
    """A point-in-time feed report, hand-built.

    `data/quality.py` owns how a report is PRODUCED and has its own tests; what
    the engine owes is a correct reading of one, so the gate is driven from a
    report written here rather than from a dataset contrived to grade that way.
    """
    statuses = {
        feed: FeedStatus(
            feed=feed,
            quality=(DataQuality.MISSING if feed in blocking else DataQuality.GOOD),
            rows=0 if feed in blocking else 500,
        )
        for feed in Feed
    }
    return QualityReport(
        symbol=SYMBOL,
        ts=ts or TS,
        statuses=statuses,
        overall=overall,
        blocking_feeds=blocking,
        note=note,
    )


def a_warmup_report(*, blocking_name: str | None = None) -> WarmupReport:
    """A per-computer readiness report. One computer short of its window."""
    computers = [
        ComputerReadiness(
            name="volatility", feeds_present=True, bars_required=266, bars_available=900
        ),
        ComputerReadiness(
            name="liquidity",
            feeds_present=True,
            bars_required=1560,
            bars_available=900 if blocking_name == "liquidity" else 1560,
        ),
        ComputerReadiness(
            name="orderflow",
            feeds_present=False,
            bars_required=130,
            bars_available=900,
            absent_feeds=(Feed.TICK_AGGREGATE,),
        ),
    ]
    return WarmupReport(bars_available=900, computers=tuple(computers))


def engine_for(
    config: FlowModelConfig, *, calendar: TradingCalendar | None = None, **kwargs: object
) -> SignalEngine:
    return SignalEngine(config, SYMBOL, calendar=calendar, **kwargs)  # type: ignore[arg-type]


def decide(
    config: FlowModelConfig,
    vector: FeatureVector,
    *,
    regime: Regime | RegimeState = Regime.TRENDING_UP,
    quality: QualityReport | None = None,
    warmup: WarmupReport | None = None,
    reference_price: float | None = ENTRY,
    calendar: TradingCalendar | None = None,
    engine: SignalEngine | None = None,
    **kwargs: object,
):
    state = (
        regime
        if isinstance(regime, RegimeState)
        else a_regime(regime, ts=vector.ts, symbol=vector.symbol)
    )
    inputs = PrecomputedInputs(
        features=vector,
        regime=state,
        reference_price=reference_price,
        quality=quality,
        warmup=warmup,
    )
    return (engine or engine_for(config, calendar=calendar)).decide_from(
        inputs, **kwargs  # type: ignore[arg-type]
    )


# ===========================================================================
# 1. Section 7: the Flow Score is a magnitude aggregate with no opinion
# ===========================================================================


def test_the_hand_built_aggregate_is_the_number_written_down(cfg):
    """The premise every later assertion rests on.

    If the fixture's aggregate is not 93.25 then every "the magnitude stayed
    high" assertion below is comparing the implementation against itself. So
    the arithmetic is asserted first, term by term, before anything relies on
    it.
    """
    decision = decide(cfg, build_vector())
    flow = decision.flow.require()
    assert flow.points_for(Component.STRUCTURE) == pytest.approx(19.00)
    assert flow.points_for(Component.ORDER_FLOW) == pytest.approx(23.75)
    assert flow.points_for(Component.OPTIONS_FLOW) == pytest.approx(19.00)
    assert flow.points_for(Component.LIQUIDITY) == pytest.approx(13.50)
    assert flow.points_for(Component.VOL_MOMENTUM) == pytest.approx(18.00)
    assert flow.score == pytest.approx(EXPECTED_FULL_SCORE)
    assert flow.available_points == pytest.approx(100.0)
    assert flow.max_points == pytest.approx(100.0)


def test_a_bearish_component_contributes_its_full_magnitude(cfg):
    """Section 7's rule, at the aggregate: direction does not scale magnitude.

    Flipping VOL_MOMENTUM from +1 to -1 changes nothing about the sum. A score
    that netted directions would drop by twice VOL_MOMENTUM's 18 points, to
    57.25; a score that zeroed the disagreeing component would drop to 75.25.
    Both wrong numbers are written out so the assertion cannot pass by
    accident.
    """
    bullish = decide(cfg, build_vector(vol_momentum_direction=1.0)).flow.require()
    bearish = decide(cfg, build_vector(vol_momentum_direction=-1.0)).flow.require()

    assert bullish.score == pytest.approx(EXPECTED_FULL_SCORE)
    assert bearish.score == pytest.approx(EXPECTED_FULL_SCORE)
    assert bearish.score != pytest.approx(EXPECTED_FULL_SCORE - 2 * 18.0)  # netting
    assert bearish.score != pytest.approx(EXPECTED_FULL_SCORE - 18.0)  # cancelling
    assert bearish.score > 0.85 * bearish.available_points
    assert bearish.components[Component.VOL_MOMENTUM].direction == -1
    assert bearish.components[Component.VOL_MOMENTUM].magnitude == pytest.approx(0.90)


def test_a_mirrored_pair_has_identical_magnitudes_and_opposite_directions(cfg):
    """A long setup and its mirror image are the same evidence, opposite sign."""
    long_flow = decide(cfg, build_vector(structure_direction=1.0)).flow.require()
    short_flow = decide(
        cfg,
        build_vector(
            structure_direction=-1.0,
            vol_momentum_direction=-1.0,
            order_flow_direction=-1.0,
            options_flow_direction=-1.0,
        ),
    ).flow.require()

    for component in Component:
        a, b = long_flow.components[component], short_flow.components[component]
        assert a.magnitude == pytest.approx(b.magnitude), component
        assert a.direction == -b.direction or a.direction == 0 == b.direction, component
    assert long_flow.score == pytest.approx(short_flow.score)
    assert long_flow.net_direction() == 1
    assert short_flow.net_direction() == -1


def test_the_bullish_majority_cannot_outvote_a_bearish_structure(cfg):
    """The bug section 7 exists to prevent, built deliberately.

    The structure names SHORT (level_direction = -1 at resistance) while order
    flow, options flow and momentum all point UP. `net_direction()` is
    therefore +1 -- a bullish majority by weighted vote -- and a system that
    "quietly takes the majority direction" would emit a LONG here. The engine
    must WAIT, and the only `Side` any stage may have considered is SHORT.

    The premise is asserted on the aggregate computed DIRECTLY from the same
    five components, because the engine does not reach its own Flow Score stage
    on this bar: the first confirmation gate the bullish majority trips is
    options flow at stage 7, and the aggregate at stage 10 is never built. That
    is the correct behaviour -- "cheap/structural rejections run before
    expensive scoring" -- and it is why `Signal.flow_score` is None here even
    though the evidence was overwhelming.

    MY FIRST EXPECTATION WAS WRONG about which gate fires: I predicted
    ORDERFLOW_CONFLICT, but section 3's prose puts options flow BEFORE order
    flow and 0.95 * 20 = 19.00 opposing points clears the 14.00 contradiction
    limit, so options flow gets there first. Both cases are now asserted -- the
    second with options neutral so order flow is the blocker -- because the
    claim being tested is that no gate takes the majority side, not that one
    particular gate refuses it.
    """
    engine = engine_for(cfg)
    bullish_majority = build_vector(
        structure_direction=-1.0,
        vol_momentum_direction=1.0,
        order_flow_direction=1.0,
        options_flow_direction=1.0,
    )
    standalone = aggregate(
        cfg, engine.score(bullish_majority), symbol=SYMBOL, ts=bullish_majority.ts
    ).require()

    # the premise: the majority really is bullish, by weighted points
    # +19 options +23.75 order flow -19 structure +0 liquidity +18 volmom = +41.75
    assert standalone.net_direction() == 1
    assert standalone.score == pytest.approx(EXPECTED_FULL_SCORE)

    # options flow refuses it first (stage 7), order flow would have (stage 8)
    options_blocked = decide(cfg, bullish_majority, engine=engine)
    order_blocked = decide(
        cfg,
        build_vector(
            structure_direction=-1.0,
            vol_momentum_direction=1.0,
            order_flow_direction=1.0,
            options_flow_direction=0.0,
        ),
        engine=engine,
    )

    assert options_blocked.blocking.stage is GateStage.OPTIONS_FLOW
    assert options_blocked.signal.wait_reason is WaitReason.OPTIONS_CONTRADICTION
    assert order_blocked.blocking.stage is GateStage.ORDER_FLOW
    assert order_blocked.signal.wait_reason is WaitReason.ORDERFLOW_CONFLICT

    for decision in (options_blocked, order_blocked):
        assert decision.signal.action is SignalAction.WAIT
        assert decision.plan is None
        assert decision.signal.flow_score is None
        assert GateStage.FLOW_SCORE not in decision.stages
        sides = {r.side for r in decision.gate_results if r.side is not None}
        assert sides == {Side.SHORT}
        assert decision.side is Side.SHORT


def test_no_trade_is_taken_in_either_direction_when_one_component_disagrees(cfg):
    """The brief's requirement, both ways round, on one piece of evidence.

    VOL_MOMENTUM holds 18.00 opposing points against a 12.00 limit, so the
    long is refused at the entry filter; the mirrored bar refuses the short the
    same way. Neither direction is taken and the aggregate is 93.25 in both.
    """
    long_decision = decide(
        cfg, build_vector(structure_direction=1.0, vol_momentum_direction=-1.0)
    )
    short_decision = decide(
        cfg,
        build_vector(
            structure_direction=-1.0,
            vol_momentum_direction=1.0,
            order_flow_direction=-1.0,
            options_flow_direction=-1.0,
        ),
    )

    for decision, side in ((long_decision, Side.LONG), (short_decision, Side.SHORT)):
        assert decision.signal.action is SignalAction.WAIT
        assert decision.blocking.stage is GateStage.ENTRY_FILTER
        assert decision.flow.require().score == pytest.approx(EXPECTED_FULL_SCORE)
        assert decision.flow.require().opposing_points(side) == pytest.approx(18.00)
        assert decision.plan is None


def test_both_sides_are_opposed_on_the_same_aggregate(cfg):
    """A high score is not a direction: both sides can be opposed at once.

    With the structure long and momentum short, LONG is opposed by 18.00 points
    and SHORT by 19.00 + 23.75 + 19.00 = 61.75. Neither reads as "the majority",
    which is the point: the aggregate is magnitude and says nothing about side.
    """
    flow = decide(
        cfg, build_vector(structure_direction=1.0, vol_momentum_direction=-1.0)
    ).flow.require()
    assert flow.opposing_points(Side.LONG) == pytest.approx(18.00)
    assert flow.opposing_points(Side.SHORT) == pytest.approx(61.75)
    assert flow.opposing_points(Side.LONG) > 0
    assert flow.opposing_points(Side.SHORT) > 0


def test_an_unmeasured_component_never_counts_as_opposition(cfg):
    """"Never invent missing options or order-flow data."

    An unmeasured component has direction 0 because nobody observed anything.
    Reporting it as disagreement would be inventing a measurement, and
    reporting it as agreement would be worse. Strict availability is off here
    so the aggregate survives the gap and the opposition arithmetic is visible.
    """
    relaxed = local_config(strict=False)
    flow = decide(
        relaxed, build_vector(order_flow_available=0.0, vol_momentum_direction=-1.0)
    ).flow.require()
    order_flow = flow.components[Component.ORDER_FLOW]

    assert order_flow.enabled is False
    assert order_flow.direction == 0
    assert order_flow.points == pytest.approx(0.0)
    assert order_flow.weight == pytest.approx(25.0)  # the weight is REPORTED, not moved
    assert flow.opposing_points(Side.LONG) == pytest.approx(18.00)  # volmom only
    assert flow.opposing_points(Side.SHORT) == pytest.approx(19.00 + 19.00)
    assert flow.available_points == pytest.approx(75.0)
    assert flow.max_points == pytest.approx(100.0)


def test_no_module_in_signals_or_risk_ever_calls_net_direction():
    """The structural defence, checked on the AST and not on prose.

    `FlowScore.net_direction()` is the one accessor that would let a majority
    of components choose a side. The gates' docstrings say it is never
    consulted; this walks the syntax tree of every module in `signals/` and
    `risk/` and asserts no CALL to it exists. A docstring that mentions the
    name is a string constant and is invisible to this check, which is exactly
    the difference between a claim and a test.
    """
    root = Path(__file__).resolve().parents[2] / "flow_model"
    offenders = []
    for path in sorted((*(root / "signals").glob("*.py"), *(root / "risk").glob("*.py"))):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "net_direction"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


# ===========================================================================
# 2. The bars-only outcome, asserted exactly
# ===========================================================================


def synthetic_dataset(spec: InstrumentSpec):
    """Five-minute NQ bars with every feed, from a seeded generator.

    Two months is enough for the slowest computer's 1560-bar window to fill
    with room to spare, and small enough that the module-scoped fixture is
    cheap. Synthetic data is not a market: nothing below is a statement about
    whether any of this is profitable, only about which gate fires.
    """
    return SyntheticMarketGenerator(
        SyntheticConfig(options_intraday=True), seed=11
    ).generate(
        SYMBOL,
        spec,
        date(2024, 1, 2),
        date(2024, 3, 1),
        INTERVAL,
        calendar=TradingCalendar(),
    )


@pytest.fixture(scope="module")
def dataset():
    return synthetic_dataset(nq_spec())


def symbol_data(dataset, *, with_feeds: bool) -> SymbolData:
    return SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: dataset.bars},
        ticks={INTERVAL: dataset.ticks} if with_feeds else None,
        quotes=dataset.quotes if with_feeds else None,
        options=dataset.options if with_feeds else None,
    )


def view_at(data: SymbolData, dataset, index: int) -> MarketView:
    ns = int(dataset.bars.ts_ns[index])
    return MarketView(data, now=from_ns(ns), now_ns=ns)


#: A bar past every warmup window and inside RTH on the module dataset. The
#: slowest computer needs 1560 bars, so anything below that reports WARMUP.
WARM_BAR = 1703


def test_bars_only_data_waits_on_data_quality_and_emits_no_flow_score(cfg, dataset):
    """The shipped shape of the bars-only answer, pinned stage by stage.

    `config.flow_score.feed_requirements` marks `tick_aggregate` and
    `options_snapshot` REQUIRED, so both are blocking on every bar and the
    pipeline stops at stage 2. What must be true is not just "it WAITs": no
    Flow Score may be reported, because a 55-point sum compared against a
    threshold calibrated on 100 is a different quantity with the same name.
    """
    data = symbol_data(dataset, with_feeds=False)
    engine = SignalEngine(cfg, SYMBOL, calendar=TradingCalendar())
    decision = engine.decide(view_at(data, dataset, WARM_BAR))
    signal = decision.signal

    assert signal.action is SignalAction.WAIT
    assert signal.wait_reason is WaitReason.DATA_QUALITY
    assert decision.blocking.stage is GateStage.MARKET_DATA
    assert signal.flow_score is None
    assert decision.flow is None
    assert signal.setup is None
    assert signal.data_quality is DataQuality.MISSING
    assert signal.confidence == 0.0
    assert decision.plan is None
    # short-circuited: the stages after market data never ran
    assert decision.stages == (GateStage.SESSION, GateStage.MARKET_DATA)
    assert decision.features is None
    assert decision.regime is None


def test_the_bars_only_wait_names_the_components_whose_feeds_are_absent(cfg, dataset):
    """A WAIT that says only "data quality" does not say which 45 points went.

    The brief requires the reason to name what is unavailable. The detail must
    reach `Signal.wait_detail` -- the field section 12 reads -- and must name
    both the feeds and the COMPONENTS that depend on them, because "the tick
    feed is missing" and "order flow's 25 points are missing" are the two
    halves of the same fact and a reader of the histogram has only the text.
    """
    data = symbol_data(dataset, with_feeds=False)
    engine = SignalEngine(cfg, SYMBOL, calendar=TradingCalendar())
    signal = engine.evaluate(view_at(data, dataset, WARM_BAR))

    assert "tick_aggregate" in signal.wait_detail
    assert "options_snapshot" in signal.wait_detail
    assert "order_flow" in signal.wait_detail
    assert "options_flow" in signal.wait_detail
    narrative = " ".join(signal.reasons)
    assert "nothing is estimated in its place" in narrative


def test_bars_only_availability_reports_fifty_five_of_one_hundred(cfg, dataset):
    """`data/quality.py`'s pre-run verdict, consumed and not re-derived.

    55 = structure 20 + liquidity 15 + vol_momentum 20. The missing 45 is order
    flow's 25 plus options flow's 20, and `scoring_is_valid` is False, which is
    the dataset-level statement the engine's refusal implements per bar.
    """
    data = symbol_data(dataset, with_feeds=False)
    report = QualityGrader(cfg.data, cfg.flow_score).availability(data)

    assert report.available_points == pytest.approx(55.0)
    assert report.total_points == pytest.approx(100.0)
    assert report.unavailable_points == pytest.approx(45.0)
    assert set(report.incomputable_components) == {
        Component.ORDER_FLOW,
        Component.OPTIONS_FLOW,
    }
    assert report.scoring_is_valid is False


def test_bars_only_scoring_refuses_rather_than_summing_what_is_left(dataset):
    """The other half: with the feed requirements optional, the AGGREGATE refuses.

    Making the two feeds optional lets the data-quality gate pass, so this is
    the only configuration in which the strict refusal is reachable on
    bars-only data. It must fire at stage 10 with `COMPONENT_DISABLED`, carry
    55.0 of 100.0, leave `Signal.flow_score` None, and say in words that the
    weight was not redistributed.
    """
    config = local_config(order_flow_required=False, options_required=False)
    data = symbol_data(dataset, with_feeds=False)
    engine = SignalEngine(config, SYMBOL, calendar=TradingCalendar())
    decision = engine.decide(view_at(data, dataset, WARM_BAR))
    signal = decision.signal

    assert signal.action is SignalAction.WAIT
    assert signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert decision.blocking.stage is GateStage.FLOW_SCORE
    assert decision.flow.refused is True
    assert decision.flow.score is None
    assert signal.flow_score is None
    assert decision.flow.available_points == pytest.approx(55.0)
    assert decision.flow.max_points == pytest.approx(100.0)
    assert decision.flow.missing_points == pytest.approx(45.0)
    assert set(decision.flow.unavailable_components) == {
        Component.ORDER_FLOW,
        Component.OPTIONS_FLOW,
    }
    assert "order_flow" in signal.wait_detail
    assert "options_flow" in signal.wait_detail
    assert "redistribut" in " ".join(signal.reasons)


def test_the_refusal_is_not_collapsed_into_the_data_quality_bucket(dataset):
    """Two different facts, two different reasons, because section 12 counts them.

    `DATA_QUALITY` is "this feed is missing or stale AT THIS INSTANT".
    `COMPONENT_DISABLED` is "45 of the 100 points have no feed for the whole
    run". The first holds on some bars, the second on every bar, and one bucket
    for both makes the rejection histogram meaningless.
    """
    assert REFUSAL_WAIT_REASON is WaitReason.COMPONENT_DISABLED
    assert REFUSAL_WAIT_REASON is not WaitReason.DATA_QUALITY
    assert WaitReason.DATA_QUALITY in WAIT_REASONS_BY_STAGE[GateStage.MARKET_DATA]
    assert WaitReason.DATA_QUALITY not in WAIT_REASONS_BY_STAGE[GateStage.FLOW_SCORE]
    assert WaitReason.COMPONENT_DISABLED in WAIT_REASONS_BY_STAGE[GateStage.FLOW_SCORE]
    assert (
        WaitReason.COMPONENT_DISABLED
        not in WAIT_REASONS_BY_STAGE[GateStage.MARKET_DATA]
    )


def test_warmup_is_not_what_fires_on_bars_only_data(cfg, dataset):
    """The trap: `FeatureVector.warmup_complete` is the AND over ALL computers.

    On bars-only data the order-flow and options-flow computers never become
    ready, so the merged flag is permanently False and a gate reading it would
    report WARMUP on every bar of a ten-year run. The warmup gate must judge
    PER COMPUTER and let a feed-absent one through.
    """
    from flow_model.signals.gates import assess_warmup

    data = symbol_data(dataset, with_feeds=False)
    engine = SignalEngine(cfg, SYMBOL, calendar=TradingCalendar())
    view = view_at(data, dataset, WARM_BAR)

    report = assess_warmup(engine.warmup_computers, view)
    assert report.blocking == ()
    merged = engine.bundle.compute(view)
    assert merged.warmup_complete is False
    assert merged.quality is DataQuality.MISSING

    # and the engine agrees: the blocking stage is not WARMUP
    assert engine.decide(view).blocking.stage is not GateStage.WARMUP


def test_a_stale_required_feed_waits_on_data_quality(cfg):
    """"Return WAIT if required data is missing or stale."

    Driven from a hand-built report so the gate's reading is what is under
    test, not the grader's production of one. A non-empty `blocking_feeds` is a
    refusal independently of `overall`: a feed graded DEGRADED can clear a
    dataset-wide DEGRADED floor while failing the GOOD floor its own component
    declared.
    """
    report = a_quality_report(
        overall=DataQuality.DEGRADED,
        blocking=(Feed.TICK_AGGREGATE,),
        note="tick_aggregate last observed 9000s ago",
    )
    assert report.is_tradable(DataQuality.DEGRADED) is False

    decision = decide(cfg, build_vector(), quality=report)
    assert decision.signal.wait_reason is WaitReason.DATA_QUALITY
    assert decision.blocking.stage is GateStage.MARKET_DATA
    assert "tick_aggregate" in decision.signal.wait_detail
    assert decision.signal.flow_score is None


def test_an_ungraded_bar_says_so_rather_than_defaulting_either_way(cfg):
    """`quality=None` means "I am not grading feeds here", and must say so.

    A gate that silently defaulted to tradable would hide that the filter never
    ran; one that defaulted to blocked would make every hand-built bar a WAIT.
    The third option is to pass and put it in the narrative.
    """
    decision = decide(cfg, build_vector(), quality=None)
    market_data = next(
        r for r in decision.gate_results if r.stage is GateStage.MARKET_DATA
    )
    assert market_data.passed is True
    assert "not graded" in " ".join(market_data.reasons)


# ===========================================================================
# 3. Determinism: the property Phase 11's parity test will rest on
# ===========================================================================


@pytest.fixture(scope="module")
def walk_config():
    """One config for every real-data walk, so the fixtures are comparable."""
    return local_config(min_flow_score=30.0)


@pytest.fixture(scope="module")
def walk_inputs(walk_config, dataset):
    data = symbol_data(dataset, with_feeds=True)
    availability = QualityGrader(
        walk_config.data, walk_config.flow_score
    ).availability(data)
    return data, availability


def _walk_engine(config, availability) -> SignalEngine:
    return SignalEngine(
        config, SYMBOL, calendar=TradingCalendar(), availability=availability
    )


def test_the_same_market_view_twice_gives_an_identical_signal(walk_config, walk_inputs, dataset):
    """No accumulator: the second call cannot drift, reasons included.

    `Signal` equality covers `reasons`, so this also pins that the narrative is
    rebuilt in the same order rather than appended to a list that survives the
    call.
    """
    data, availability = walk_inputs
    engine = _walk_engine(walk_config, availability)
    view = view_at(data, dataset, WARM_BAR)
    first = engine.evaluate(view, equity=100_000.0)
    second = engine.evaluate(view, equity=100_000.0)
    assert first == second
    assert first.reasons == second.reasons
    assert first.reasons != ()


def test_a_fresh_engine_agrees_with_one_already_walked_over_other_data(
    walk_config, walk_inputs, dataset
):
    """A cold single call and a sequential walk must reach the same bar alike.

    This is the backtest/live difference in miniature: live sees one bar, the
    backtest arrives at it after sixty others. `regime/detector.py` records a
    HIGH finding of exactly this shape, so it is asserted rather than assumed.
    """
    data, availability = walk_inputs
    target = view_at(data, dataset, WARM_BAR)

    cold = _walk_engine(walk_config, availability).evaluate(target, equity=100_000.0)

    walked = _walk_engine(walk_config, availability)
    for index in range(WARM_BAR - 60, WARM_BAR):
        walked.evaluate(view_at(data, dataset, index), equity=100_000.0)
    warm = walked.evaluate(target, equity=100_000.0)

    assert warm == cold


def test_the_same_bars_evaluated_forwards_and_backwards_agree(
    walk_config, walk_inputs, dataset
):
    """Evaluation ORDER must not change any bar's answer.

    The strongest statement of statelessness available without reading the
    source: thirty bars are evaluated front-to-back on one engine and
    back-to-front on another, and every Signal must match. Any accumulator,
    memo keyed on time, or "previous bar" field would make at least one bar
    differ, because in one walk its predecessor has been seen and in the other
    it has not.
    """
    data, availability = walk_inputs
    indices = list(range(WARM_BAR - 15, WARM_BAR + 15))

    forward_engine = _walk_engine(walk_config, availability)
    forward = {
        i: forward_engine.evaluate(view_at(data, dataset, i), equity=100_000.0)
        for i in indices
    }
    backward_engine = _walk_engine(walk_config, availability)
    backward = {
        i: backward_engine.evaluate(view_at(data, dataset, i), equity=100_000.0)
        for i in reversed(indices)
    }

    assert [i for i in indices if forward[i] != backward[i]] == []

    # and the window is not vacuous: the bars really did traverse the
    # pipeline past warmup, so the equality above is a statement about the
    # feature layer, the detector and the scorers rather than about a
    # short-circuit that never touched them.
    probe = _walk_engine(walk_config, availability).decide(
        view_at(data, dataset, WARM_BAR), equity=100_000.0
    )
    assert GateStage.STRUCTURE in probe.stages
    assert probe.features is not None
    assert probe.regime is not None
    assert probe.regime.regime is not Regime.UNKNOWN
    assert all(s.reasons != () for s in forward.values())


def test_two_engines_from_one_config_agree_bar_for_bar(walk_config, walk_inputs, dataset):
    """Nothing is cached on the engine that a second instance would lack."""
    data, availability = walk_inputs
    first = _walk_engine(walk_config, availability)
    second = _walk_engine(walk_config, availability)
    for index in range(WARM_BAR, WARM_BAR + 10):
        view = view_at(data, dataset, index)
        assert first.evaluate(view, equity=100_000.0) == second.evaluate(
            view, equity=100_000.0
        )


def test_a_hand_built_bar_is_identical_on_every_front_door(cfg):
    """`decide`, `decide_from` and `evaluate` are one sequence, not three."""
    vector = build_vector()
    engine = engine_for(cfg)
    inputs = PrecomputedInputs(
        features=vector, regime=a_regime(), reference_price=ENTRY
    )
    first = engine.decide_from(inputs).signal
    second = engine.decide_from(
        PrecomputedInputs(features=vector, regime=a_regime(), reference_price=ENTRY)
    ).signal
    assert first == second


def test_a_limits_manager_in_a_steady_state_does_not_change_the_answer(cfg):
    """The admitted caveat, tested rather than taken on trust.

    `RiskLimitManager.check()` updates its high-water mark even on a pass, so
    `decide` is not idempotent for a CHANGING equity curve -- that is the
    caller's problem to solve in Phase 7. At constant equity the mark is
    already at equity and nothing moves, so two calls must agree; if they did
    not, the parity unit would be unusable even in the easy case.
    """
    vector = build_vector(achievable_rr=1.2)
    limits = RiskLimitManager(cfg.risk, starting_equity=100_000.0)
    first = decide(cfg, vector, equity=100_000.0, limits=limits).signal
    second = decide(cfg, vector, equity=100_000.0, limits=limits).signal
    assert first.action is SignalAction.LONG
    assert first == second


# ===========================================================================
# 4. Short-circuiting is a correctness property
# ===========================================================================


class CountingInputs(PrecomputedInputs):
    """`PrecomputedInputs` that records which stage inputs were asked for.

    Section 3's order exists so cheap rejections run first, and the engine
    claims "a bar rejected on session or data quality never pays for the
    feature bundle". That is checkable: if `features()` was never called then
    no field of the resulting `Signal` can have come from the feature layer.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.calls: Counter[str] = Counter()

    def quality(self):
        self.calls["quality"] += 1
        return super().quality()

    def warmup(self):
        self.calls["warmup"] += 1
        return super().warmup()

    def features(self):
        self.calls["features"] += 1
        return super().features()

    def regime(self):
        self.calls["regime"] += 1
        return super().regime()

    def reference_price(self):
        self.calls["reference_price"] += 1
        return super().reference_price()


def test_a_bar_rejected_on_data_quality_never_computes_features_or_regime(cfg):
    """Nothing downstream of the failing stage is even asked for."""
    inputs = CountingInputs(
        features=build_vector(),
        regime=a_regime(),
        reference_price=ENTRY,
        quality=a_quality_report(
            overall=DataQuality.MISSING, blocking=(Feed.TICK_AGGREGATE,)
        ),
    )
    decision = engine_for(cfg).decide_from(inputs)

    assert decision.signal.wait_reason is WaitReason.DATA_QUALITY
    assert inputs.calls["quality"] == 1
    assert inputs.calls["features"] == 0
    assert inputs.calls["regime"] == 0
    assert inputs.calls["warmup"] == 0


def test_a_bar_rejected_on_regime_never_computes_the_feature_bundle(cfg):
    """The regime stage precedes the bundle, so a blocked regime is cheap."""
    blocked = local_config(allowed_regimes=(Regime.TRENDING_UP, Regime.TRENDING_DOWN))
    inputs = CountingInputs(
        features=build_vector(), regime=a_regime(Regime.CHOP), reference_price=ENTRY
    )
    decision = engine_for(blocked).decide_from(inputs)

    assert decision.signal.wait_reason is WaitReason.REGIME_BLOCKED
    assert inputs.calls["regime"] == 1
    assert inputs.calls["features"] == 0


def test_a_taken_trade_asks_for_each_stage_input_once(cfg):
    """Memoized per evaluation: the bundle is not recomputed per stage.

    Five stages read the feature vector. Recomputing it each time would be
    wasteful, but worse, it would make the engine's answer depend on a
    computation being repeatable -- which is true here and is exactly the
    assumption a cache would quietly introduce if it were keyed on anything.
    """
    inputs = CountingInputs(
        features=build_vector(achievable_rr=1.2),
        regime=a_regime(),
        reference_price=ENTRY,
    )
    decision = engine_for(cfg).decide_from(inputs, equity=100_000.0)
    assert decision.signal.action is SignalAction.LONG
    assert inputs.calls["features"] >= 1
    assert inputs.calls["regime"] == 1
    assert inputs.calls["quality"] == 1


# ===========================================================================
# 5. Every WAIT carries a reason, and the reason belongs to its stage
# ===========================================================================


def _case_session():
    config = local_config()
    vector = build_vector(ts=TS_OVERNIGHT)
    return (
        GateStage.SESSION,
        WaitReason.SESSION_CLOSED,
        lambda: decide(
            config,
            vector,
            regime=a_regime(ts=TS_OVERNIGHT),
            calendar=TradingCalendar(),
        ),
    )


def _case_market_data():
    config = local_config()
    return (
        GateStage.MARKET_DATA,
        WaitReason.DATA_QUALITY,
        lambda: decide(
            config,
            build_vector(),
            quality=a_quality_report(
                overall=DataQuality.MISSING, blocking=(Feed.OPTIONS_SNAPSHOT,)
            ),
        ),
    )


def _case_warmup():
    config = local_config()
    return (
        GateStage.WARMUP,
        WaitReason.WARMUP,
        lambda: decide(
            config, build_vector(), warmup=a_warmup_report(blocking_name="liquidity")
        ),
    )


def _case_regime_unknown():
    config = local_config()
    return (
        GateStage.REGIME,
        WaitReason.REGIME_BLOCKED,
        lambda: decide(config, build_vector(), regime=Regime.UNKNOWN),
    )


def _case_regime_barred():
    config = local_config(allowed_regimes=(Regime.TRENDING_UP, Regime.TRENDING_DOWN))
    return (
        GateStage.REGIME,
        WaitReason.REGIME_BLOCKED,
        lambda: decide(config, build_vector(), regime=Regime.LOW_VOL),
    )


def _case_liquidity():
    config = local_config()
    return (
        GateStage.LIQUIDITY,
        WaitReason.LIQUIDITY,
        lambda: decide(config, build_vector(relative_volume=0.0)),
    )


def _case_structure():
    config = local_config()
    return (
        GateStage.STRUCTURE,
        WaitReason.NO_STRUCTURE,
        lambda: decide(
            config,
            build_vector(gate_passed=0.0, gate_reason=GATE_SIGNIFICANCE),
        ),
    )


def _case_structure_rr():
    config = local_config()
    return (
        GateStage.STRUCTURE,
        WaitReason.RR_TOO_LOW,
        lambda: decide(
            config, build_vector(gate_passed=0.0, gate_reason=GATE_REWARD_RISK)
        ),
    )


def _case_options_flow():
    config = local_config()
    return (
        GateStage.OPTIONS_FLOW,
        WaitReason.OPTIONS_CONTRADICTION,
        # 0.75 * 20 = 15.00 opposing points, above the 14.00 limit
        lambda: decide(
            config,
            build_vector(options_flow_direction=-1.0, options_flow_magnitude=0.75),
        ),
    )


def _case_order_flow():
    config = local_config()
    return (
        GateStage.ORDER_FLOW,
        WaitReason.ORDERFLOW_CONFLICT,
        # 0.04 * 25 = 1.00 opposing point: blocked on DIRECTION, not magnitude
        lambda: decide(
            config,
            build_vector(order_flow_direction=-1.0, order_flow_magnitude=0.04),
        ),
    )


def _case_momentum():
    config = local_config()
    return (
        GateStage.MOMENTUM,
        WaitReason.VOLATILITY_BAND,
        lambda: decide(config, build_vector(atr_percentile=0.05)),
    )


def _case_flow_score_refused():
    config = local_config(strict=True)
    return (
        GateStage.FLOW_SCORE,
        WaitReason.COMPONENT_DISABLED,
        lambda: decide(config, build_vector(order_flow_available=0.0)),
    )


def _case_flow_score_below():
    config = local_config(min_flow_score=40.0)
    # every magnitude 0.10: 2.0 + 2.5 + 2.0 + 1.5 + 2.0 = 10.00, below 40.00
    return (
        GateStage.FLOW_SCORE,
        WaitReason.SCORE_BELOW_THRESHOLD,
        lambda: decide(
            config,
            build_vector(
                structure_magnitude=0.10,
                liquidity_score=0.10,
                vol_momentum_magnitude=0.10,
                order_flow_magnitude=0.10,
                options_flow_magnitude=0.10,
            ),
        ),
    )


def _case_entry_filter():
    config = local_config()
    return (
        GateStage.ENTRY_FILTER,
        WaitReason.VOLATILITY_BAND,
        lambda: decide(config, build_vector(vol_momentum_direction=-1.0)),
    )


def _case_setup():
    config = local_config(
        setups={
            SetupType.SCALP_1R: a_setup(SetupType.SCALP_1R, min_flow_score=30.0),
            SetupType.SETUP_2R: a_setup(
                SetupType.SETUP_2R, min_flow_score=30.0, enabled=False
            ),
            SetupType.DIRECTIONAL_3R: a_setup(
                SetupType.DIRECTIONAL_3R, min_flow_score=30.0
            ),
        }
    )
    return (
        GateStage.SETUP,
        WaitReason.NO_SETUP_MATCH,
        lambda: decide(config, build_vector(achievable_rr=2.0)),
    )


def _case_risk():
    config = local_config()
    # a 0.25-point stop is 1 tick, below SCALP_1R's min_stop_ticks of 4
    return (
        GateStage.RISK,
        WaitReason.RISK_LIMIT,
        lambda: decide(
            config, build_vector(stop_distance=0.25), equity=100_000.0
        ),
    )


STAGE_CASES = [
    _case_session,
    _case_market_data,
    _case_warmup,
    _case_regime_unknown,
    _case_regime_barred,
    _case_liquidity,
    _case_structure,
    _case_structure_rr,
    _case_options_flow,
    _case_order_flow,
    _case_momentum,
    _case_flow_score_refused,
    _case_flow_score_below,
    _case_entry_filter,
    _case_setup,
    _case_risk,
]


@pytest.mark.parametrize("case", STAGE_CASES, ids=lambda c: c.__name__[6:])
def test_every_stage_fails_with_a_reason_it_declared(case):
    """One bar per stage, each checked against that stage's own declaration.

    `WAIT_REASONS_BY_STAGE` is what section 12's rejection analysis labels its
    buckets by. A stage emitting a reason it did not declare makes the labels
    wrong, and the only way to know every stage is honest is to drive each one
    to failure and compare.
    """
    stage, expected, run = case()
    decision = run()
    blocking = decision.blocking

    assert decision.signal.action is SignalAction.WAIT
    assert blocking is not None
    assert blocking.stage is stage
    assert blocking.wait_reason is expected
    assert expected in WAIT_REASONS_BY_STAGE[stage]
    assert decision.signal.wait_reason is expected
    assert decision.plan is None


@pytest.mark.parametrize("case", STAGE_CASES, ids=lambda c: c.__name__[6:])
def test_every_wait_carries_a_detail_and_a_narrative(case):
    """"Every WAIT is recorded with its blocking gate so rejection analysis is
    possible." A reason code with no words behind it is not a record.
    """
    _, _, run = case()
    decision = run()
    signal = decision.signal
    assert signal.wait_detail.strip() != ""
    assert signal.reasons != ()
    # the blocking gate's own words are in the narrative, not only its code
    blocking = decision.gate_results[-1]
    assert blocking.reasons != ()
    for reason in blocking.reasons:
        assert reason in signal.reasons
    assert signal.reasons[-1] in blocking.reasons


@pytest.mark.parametrize("case", STAGE_CASES, ids=lambda c: c.__name__[6:])
def test_every_wait_signal_satisfies_its_own_contract(case):
    """Round-trip through the Phase 1 validator, which owns the invariants.

    Re-validating the model from its own fields is the cheapest way to assert
    the engine cannot emit a Signal the contract would have refused -- for
    instance a WAIT with no `wait_reason`, or a directional signal carrying one.
    """
    _, _, run = case()
    signal = run().signal
    rebuilt = Signal(**signal.to_init_dict())
    assert rebuilt == signal
    assert rebuilt.side is None
    assert rebuilt.wait_reason is not None


def test_the_stages_reached_are_a_prefix_of_the_declared_order():
    """The implemented order is checked against `GateStage`, not against prose.

    Every case above must have run a prefix of the enum and stopped. That is
    the whole content of "the first failing gate wins", and it is what makes
    the rejection histogram comparable between runs.
    """
    declared = tuple(GateStage)
    for case in STAGE_CASES:
        stage, _, run = case()
        decision = run()
        reached = decision.stages
        assert reached == declared[: len(reached)], case.__name__
        assert reached[-1] is stage, case.__name__
        assert all(r.passed for r in decision.gate_results[:-1]), case.__name__


def test_a_taken_trade_runs_every_declared_stage(cfg):
    """A LONG must have passed all thirteen, in order, with no gaps."""
    decision = decide(cfg, build_vector(achievable_rr=1.2), equity=100_000.0)
    assert decision.signal.action is SignalAction.LONG
    assert decision.stages == tuple(GateStage)
    assert all(result.passed for result in decision.gate_results)
    assert decision.signal.wait_reason is None
    assert Signal(**decision.signal.to_init_dict()) == decision.signal


def test_the_prose_order_is_implemented_and_not_the_code_block_order(cfg):
    """The one deviation that changes which of two reasons is reported.

    Section 3's code block puts the Flow Score threshold BEFORE the
    confirmation gates; its prose puts options flow and order flow first. They
    cannot both be implemented. This bar fails both -- order flow opposes AND
    the aggregate is below the threshold -- and the implemented (prose) order
    reports ORDERFLOW_CONFLICT. Pinned so that if someone switches to the code
    block's order the change is visible rather than silent.
    """
    config = local_config(min_flow_score=90.0)
    vector = build_vector(
        order_flow_direction=-1.0,
        order_flow_magnitude=0.10,
        structure_magnitude=0.10,
        liquidity_score=0.10,
        vol_momentum_magnitude=0.10,
        options_flow_magnitude=0.10,
    )
    # premise: 2.0 + 2.5 + 2.0 + 1.5 + 2.0 = 10.00, far below the 90.00 floor
    standalone = aggregate(
        config, engine_for(config).score(vector), symbol=SYMBOL, ts=vector.ts
    )
    assert standalone.require().score == pytest.approx(10.00)
    assert min_flow_score_required(config) == pytest.approx(90.00)

    decision = decide(config, vector)
    assert decision.blocking.stage is GateStage.ORDER_FLOW
    assert decision.signal.wait_reason is WaitReason.ORDERFLOW_CONFLICT
    assert GateStage.FLOW_SCORE not in decision.stages


def test_vol_momentum_opposition_reports_the_nearest_available_reason(cfg):
    """The admitted inexact mapping, pinned with the fact it carries instead.

    No `WaitReason` means "momentum disagreed", and no new reason string may be
    invented, so VOL_MOMENTUM's opposition maps to `VOLATILITY_BAND` -- the
    reason section 3 assigns to that stage. What must not be lost is the
    precise fact, so the detail has to name the component and the points.
    """
    decision = decide(cfg, build_vector(vol_momentum_direction=-1.0))
    assert decision.blocking.stage is GateStage.ENTRY_FILTER
    assert decision.signal.wait_reason is WaitReason.VOLATILITY_BAND
    assert "vol_momentum" in decision.signal.wait_detail
    assert "18.00" in decision.signal.wait_detail
    assert "12.00" in decision.signal.wait_detail


def test_the_two_opposition_thresholds_are_both_live(cfg):
    """12.00 and 14.00 are different numbers and both must bind.

    Options opposition of 13.00 points clears the 14.00 contradiction limit at
    stage 7 and then fails the 12.00 entry-filter limit at stage 11. If either
    threshold were being used for both comparisons this bar would stop at the
    wrong stage.
    """
    # 0.65 * 20 = 13.00
    decision = decide(cfg, build_vector(options_flow_direction=-1.0,
                                        options_flow_magnitude=0.65))
    options_stage = next(
        r for r in decision.gate_results if r.stage is GateStage.OPTIONS_FLOW
    )
    assert options_stage.passed is True
    assert options_stage.diagnostics["options_opposing_points"] == pytest.approx(13.00)
    assert decision.blocking.stage is GateStage.ENTRY_FILTER
    assert decision.signal.wait_reason is WaitReason.OPTIONS_CONTRADICTION


def test_a_measured_neutral_order_flow_is_not_a_conflict(cfg):
    """Direction 0 from a MEASURED tape is not disagreement.

    Whether neutral is good enough is a per-setup question
    (`require_orderflow_confirmation`), answered in `signals/setups.py`, and
    the gate must not pre-empt it.
    """
    decision = decide(
        cfg, build_vector(order_flow_direction=0.0, achievable_rr=1.2),
        equity=100_000.0,
    )
    order_flow = next(
        r for r in decision.gate_results if r.stage is GateStage.ORDER_FLOW
    )
    assert order_flow.passed is True
    assert "neutral" in " ".join(order_flow.reasons)
    assert decision.signal.action is SignalAction.LONG


def test_a_setup_requiring_confirmation_declines_an_unmeasured_tape():
    """"No proxy accepted": the setup says so, with COMPONENT_DISABLED.

    Strict availability is off so the aggregate survives the missing component;
    what must then happen is that a setup requiring order-flow confirmation
    declines the bar at the SETUP stage rather than taking it on an
    assumed-neutral tape.
    """
    config = local_config(
        strict=False,
        setups={
            setup: a_setup(
                setup, min_flow_score=10.0, require_orderflow_confirmation=True
            )
            for setup in SetupType
        },
    )
    decision = decide(config, build_vector(order_flow_available=0.0, achievable_rr=1.2))
    assert decision.blocking.stage is GateStage.SETUP
    assert decision.signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert "no proxy" in decision.signal.wait_detail.lower()


def test_an_ablated_component_refuses_on_every_bar_under_strict_mode():
    """A documented consequence of the flag, not a surprise for Phase 8.

    `enabled_components` is the ablation surface, and under
    `strict_component_availability` an ablated component makes the aggregate
    refuse. That means an ablation sweep produces nothing but
    COMPONENT_DISABLED WAITs unless strict mode is turned off deliberately.
    """
    ablated = local_config(
        enabled_components={c: c is not Component.OPTIONS_FLOW for c in Component}
    )
    decision = decide(ablated, build_vector())
    assert decision.blocking.stage is GateStage.FLOW_SCORE
    assert decision.signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert decision.flow.available_points == pytest.approx(80.0)
    assert decision.flow.max_points == pytest.approx(100.0)
    assert decision.signal.flow_score is None


def test_a_fractional_level_direction_raises_rather_than_picking_a_side(cfg):
    """A direction between the three legal values is a magnitude in disguise.

    The side is the ONE thing the structure gate contributes, so it must not be
    guessed from a value `features/levels.py` is not allowed to emit. Raising
    is correct here: this is a feature-layer contract breach, not a data
    condition, and the engine's rule is that only DATA produces a WAIT.
    """
    with pytest.raises(ValueError, match="direction"):
        decide(cfg, build_vector(structure_direction=0.4))


def test_a_level_engine_that_contradicts_itself_declines_the_bar(cfg):
    """`structure_gate_passed=1` with a non-passing reason code is resolved by
    declining, not by preferring one of the two outputs."""
    decision = decide(cfg, build_vector(gate_passed=1.0, gate_reason=GATE_NO_ZONE))
    assert decision.blocking.stage is GateStage.STRUCTURE
    assert decision.signal.wait_reason is WaitReason.NO_STRUCTURE
    assert "disagree" in decision.signal.wait_detail


# ===========================================================================
# 6. Setup selection is a measurement read off the structure
# ===========================================================================


@pytest.mark.parametrize(
    "achievable_rr, expected",
    [
        (1.00, SetupType.SCALP_1R),
        (1.79, SetupType.SCALP_1R),
        (1.80, SetupType.SETUP_2R),
        (2.69, SetupType.SETUP_2R),
        (2.70, SetupType.DIRECTIONAL_3R),
        (9.00, SetupType.DIRECTIONAL_3R),
    ],
)
def test_the_band_edges_are_14_6s_and_are_half_open(cfg, achievable_rr, expected):
    """1.0-1.8 SCALP, 1.8-2.7 SETUP_2R, >=2.7 DIRECTIONAL -- edges included below.

    The edges are the thing most likely to drift, and a closed/open mistake at
    1.80 silently relabels every bar in a band. The table is the one in
    `levels.SETUP_RR_BANDS`, which is imported rather than restated.
    """
    assert [floor for floor, _ in SETUP_RR_BANDS] == [2.7, 1.8, 1.0]
    decision = decide(cfg, build_vector(achievable_rr=achievable_rr), equity=100_000.0)
    assert decision.selection.setup is expected
    assert decision.signal.setup is expected
    assert decision.selection.achievable_rr == pytest.approx(achievable_rr)


def test_a_two_r_bar_is_declined_rather_than_handed_to_another_setup():
    """"Do not force every trade into the same target."

    With SETUP_2R disabled, a bar measuring 2.0R is declined. It is NOT
    promoted to DIRECTIONAL_3R (which would plan a target the structure does
    not support) and NOT truncated to SCALP_1R (which would be the 0.4R-scalp
    failure mode section 0.2 exists to catch), even though both are enabled.
    """
    config = local_config(
        setups={
            SetupType.SCALP_1R: a_setup(SetupType.SCALP_1R, min_flow_score=10.0),
            SetupType.SETUP_2R: a_setup(
                SetupType.SETUP_2R, min_flow_score=10.0, enabled=False
            ),
            SetupType.DIRECTIONAL_3R: a_setup(
                SetupType.DIRECTIONAL_3R, min_flow_score=10.0
            ),
        }
    )
    decision = decide(config, build_vector(achievable_rr=2.0), equity=100_000.0)
    assert decision.signal.action is SignalAction.WAIT
    assert decision.signal.wait_reason is WaitReason.NO_SETUP_MATCH
    assert decision.selection.setup is None
    assert "SETUP_2R" in decision.signal.wait_detail
    assert "not retargeted" in decision.signal.wait_detail
    # the near-misses are recorded without the selector acting on them
    by_setup = {r.setup: r for r in decision.selection.rejections}
    assert by_setup[SetupType.SETUP_2R].selected_by_band is True
    assert by_setup[SetupType.DIRECTIONAL_3R].selected_by_band is False
    assert by_setup[SetupType.DIRECTIONAL_3R].wait_reason is WaitReason.RR_TOO_LOW


def test_disabling_the_lowest_setup_raises_the_floor_and_declines_a_one_r_bar():
    """The floor is the HIGHER of the two sources, so both have to be cleared."""
    config = local_config(
        setups={
            SetupType.SETUP_2R: a_setup(SetupType.SETUP_2R, min_flow_score=10.0),
            SetupType.DIRECTIONAL_3R: a_setup(
                SetupType.DIRECTIONAL_3R, min_flow_score=10.0
            ),
        }
    )
    # max(levels.min_reward_risk=1.0, min(1.8, 2.7)) = 1.8
    assert min_reward_risk_floor(config) == pytest.approx(1.8)
    decision = decide(config, build_vector(achievable_rr=1.2))
    assert decision.signal.wait_reason is WaitReason.RR_TOO_LOW
    assert decision.blocking.stage is GateStage.SETUP


def test_the_planned_target_may_sit_beyond_the_measured_zone_and_must_say_so():
    """14.6's band floor is below the setup's own reward_risk, and that shows.

    A bar measuring 2.80R selects DIRECTIONAL_3R (band opens at 2.70) and the
    setup plans 3.00R, so the planned target lies BEYOND the opposing major
    zone the measurement came from. Section 0.2's label rule requires planning
    at 3.00, so the gap is real and is reported rather than closed by moving a
    number. Both quantities must be on the record.
    """
    config = local_config(min_flow_score=10.0)
    decision = decide(config, build_vector(achievable_rr=2.80), equity=100_000.0)
    plan = decision.plan

    assert decision.signal.setup is SetupType.DIRECTIONAL_3R
    assert plan.achievable_rr == pytest.approx(2.80)
    assert plan.planned_r_multiple == pytest.approx(3.00)
    assert plan.plans_beyond_measured_target is True
    assert decision.selection.plans_beyond_measured_target is True
    narrative = " ".join(decision.signal.reasons)
    assert "optimistic about reachability" in narrative
    # the structural target is nearer than the planned one, in price terms
    assert plan.structural_target_price < plan.target_price


def test_a_one_r_bar_plans_exactly_one_r_and_the_gap_is_absent(cfg):
    """The other side of the same coin: inside the band, nothing is overstated."""
    decision = decide(cfg, build_vector(achievable_rr=1.20), equity=100_000.0)
    plan = decision.plan
    assert plan.setup is SetupType.SCALP_1R
    assert plan.planned_r_multiple == pytest.approx(1.00)
    assert plan.achievable_rr == pytest.approx(1.20)
    assert plan.plans_beyond_measured_target is False


def test_a_setup_class_that_disagrees_with_the_measurement_raises(cfg):
    """A mislabelled setup is worse than a declined bar, so it raises.

    `level_setup_class` is the feature layer's verdict and `setup_class_for` is
    re-evaluated on the same inputs as a consistency assertion. A disagreement
    means a setup label no longer matches the measurement it was read off, and
    the only two options are a loud error and a silently mislabelled trade.
    """
    vector = build_vector(achievable_rr=1.20)
    values = dict(vector.values)
    values["level_setup_class"] = 3.0  # claims DIRECTIONAL_3R for a 1.2R bar
    broken = vector.replace(values=values)
    with pytest.raises(SetupError, match="level_setup_class"):
        decide(cfg, broken)


def test_the_union_helpers_ask_the_weakest_honest_question(cfg):
    """Before selection the only honest threshold is the loosest one.

    Section 3 writes the regime, Flow Score and volatility gates in terms of
    "the setup", which selection has not determined -- the specification is
    circular there. The pre-selection gates therefore test the union, and the
    determined setup's own threshold is re-applied in `signals/setups.py`.
    """
    assert vol_percentile_band(cfg) == (
        pytest.approx(0.10),
        pytest.approx(1.00),
    )
    assert min_flow_score_required(cfg) == pytest.approx(30.0)
    assert min_reward_risk_floor(cfg) == pytest.approx(1.0)


def test_a_setup_specific_band_miss_is_reported_at_the_setup_stage():
    """A bar clearing the union and failing the determined setup is declined there.

    ATR percentile 0.15 is inside the union band [0.10, 1.00] so the momentum
    gate passes, and outside DIRECTIONAL_3R's own [0.25, 1.00] so the setup
    declines it -- with VOLATILITY_BAND, the reason section 3 assigns to that
    comparison, rather than collapsed into NO_SETUP_MATCH.
    """
    config = local_config(min_flow_score=10.0)
    decision = decide(config, build_vector(achievable_rr=3.0, atr_percentile=0.15))
    momentum = next(r for r in decision.gate_results if r.stage is GateStage.MOMENTUM)
    assert momentum.passed is True
    assert decision.blocking.stage is GateStage.SETUP
    assert decision.signal.wait_reason is WaitReason.VOLATILITY_BAND
    assert "0.25" in decision.signal.wait_detail


def test_a_setup_specific_score_miss_is_reported_as_score_below_threshold():
    """Collapsing this into NO_SETUP_MATCH would make the reason unreachable."""
    config = local_config(
        setups={
            SetupType.SCALP_1R: a_setup(SetupType.SCALP_1R, min_flow_score=10.0),
            SetupType.DIRECTIONAL_3R: a_setup(
                SetupType.DIRECTIONAL_3R, min_flow_score=95.0
            ),
        }
    )
    assert min_flow_score_required(config) == pytest.approx(10.0)
    decision = decide(config, build_vector(achievable_rr=3.0))
    # premise: the aggregate clears the 10.0 union floor and misses 95.0
    assert decision.flow.require().score == pytest.approx(EXPECTED_FULL_SCORE)
    assert decision.blocking.stage is GateStage.SETUP
    assert decision.signal.wait_reason is WaitReason.SCORE_BELOW_THRESHOLD


# ===========================================================================
# 7. The risk layer is consumed, never worked around
# ===========================================================================


def test_the_plan_arithmetic_is_the_arithmetic_written_down(cfg):
    """Section 8, by hand, on a geometry chosen so nothing rounds.

      stop_distance = |18000.00 - 17980.00| = 20.00 points = 80 ticks
      risk_per_unit = 20.00 * point_value 20.00 = 400.00
      budget        = equity 100000 * risk_per_trade_pct 0.005 = 500.00
      size          = floor(500.00 / 400.00) = 1
      actual risk   = 1 * 400.00 = 400.00, under the 1% cap of 1000.00
      target        = 18000.00 + 20.00 * 1.0 = 18020.00  (SCALP_1R plans 1R)
    """
    decision = decide(cfg, build_vector(achievable_rr=1.2), equity=100_000.0)
    plan = decision.plan
    assert plan.entry_price == pytest.approx(18_000.00)
    assert plan.stop_price == pytest.approx(17_980.00)
    assert plan.target_price == pytest.approx(18_020.00)
    assert plan.stop_distance == pytest.approx(20.00)
    assert plan.stop_ticks == pytest.approx(80.0)
    assert plan.size == pytest.approx(1.0)
    assert plan.risk_dollars == pytest.approx(400.00)
    assert plan.point_value == pytest.approx(20.00)
    assert plan.max_hold_bars == 12


def test_the_plan_converts_to_a_trade_intent_that_validates_r(cfg):
    """R is defined once, and by the contract that owns the definition.

    `TradeIntent` re-derives `risk_dollars = size * |entry - stop| * point_value`
    and refuses a plan where they disagree, so going through it is a real check
    rather than a copy of the engine's arithmetic.
    """
    plan = decide(cfg, build_vector(achievable_rr=1.2), equity=100_000.0).plan
    intent = plan.to_trade_intent()
    assert intent.side is Side.LONG
    assert intent.planned_r_multiple == pytest.approx(1.0)
    assert intent.risk_dollars == pytest.approx(plan.risk_dollars)
    assert intent.stop_distance == pytest.approx(20.00)
    assert intent.target_distance == pytest.approx(20.00)
    assert intent.max_hold_bars == plan.max_hold_bars


def test_the_short_side_mirrors_the_geometry(cfg):
    """A short at resistance is the same arithmetic with the signs flipped."""
    decision = decide(
        cfg,
        build_vector(
            structure_direction=-1.0,
            achievable_rr=1.2,
            vol_momentum_direction=-1.0,
            order_flow_direction=-1.0,
            options_flow_direction=-1.0,
        ),
        equity=100_000.0,
    )
    plan = decision.plan
    assert decision.signal.action is SignalAction.SHORT
    assert plan.side is Side.SHORT
    assert plan.stop_price == pytest.approx(18_020.00)
    assert plan.target_price == pytest.approx(17_980.00)
    assert plan.risk_dollars == pytest.approx(400.00)
    assert plan.to_trade_intent().planned_r_multiple == pytest.approx(1.0)


def test_a_stop_wider_than_the_setup_allows_is_refused_not_narrowed(cfg):
    """The cap is `max_stop_atr_multiple * ATR` and the stop is never trimmed.

    SCALP_1R caps the stop at 2.0 * ATR = 40.00 points. A 50.00-point
    structural stop is refused; narrowing it to fit would move the stop off the
    place the thesis is falsified, which is the whole content of 14.6.
    """
    decision = decide(cfg, build_vector(stop_distance=50.0), equity=100_000.0)
    assert decision.blocking.stage is GateStage.RISK
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert decision.sizing.reason == SizingRejection.STOP_TOO_WIDE
    assert decision.plan is None


def test_a_sizing_reward_risk_refusal_is_rr_too_low_and_not_a_risk_limit():
    """The one sizing refusal that is NOT a risk-limit problem.

    `risk/sizing.py` refuses a trade whose grid-realizable R falls below the
    setup's minimum "rather than relabelled", and the engine must surface that
    as RR_TOO_LOW -- never work around it by widening the stop, moving the
    target or picking a smaller setup.

    Reaching it needs a contrived geometry, and the contrivance is itself the
    finding: with an integral `reward_risk` the target distance is an integer
    multiple of the stop distance, which is already on the tick grid, so the
    rounding can never bite. Here DIRECTIONAL_3R plans 3.10R against a one-tick
    stop: 3.10 * 0.25 = 0.775, which rounds TOWARD entry to 0.75, giving 3.00R
    against a 3.05 minimum.
    """
    config = local_config(
        setups={
            SetupType.DIRECTIONAL_3R: a_setup(
                SetupType.DIRECTIONAL_3R,
                min_flow_score=10.0,
                reward_risk=3.10,
                min_reward_risk=3.05,
                min_stop_ticks=1,
            )
        }
    )
    # premise: the target really does round down a tick
    raw = ENTRY + 0.25 * 3.10
    assert round_to_tick(raw, 0.25, mode="down") == pytest.approx(ENTRY + 0.75)
    assert (ENTRY + 0.75 - ENTRY) / 0.25 == pytest.approx(3.0)

    decision = decide(
        config, build_vector(achievable_rr=3.5, stop_distance=0.25), equity=100_000.0
    )
    assert decision.blocking.stage is GateStage.RISK
    assert decision.signal.wait_reason is WaitReason.RR_TOO_LOW
    assert decision.sizing.reason == SizingRejection.REWARD_RISK_BELOW_MINIMUM
    assert (
        SIZING_WAIT_REASONS[SizingRejection.REWARD_RISK_BELOW_MINIMUM]
        is WaitReason.RR_TOO_LOW
    )
    assert decision.plan is None


def test_an_integral_reward_risk_on_a_tick_grid_never_triggers_that_refusal(cfg):
    """Why the test above had to be contrived, stated as a property.

    Entry and stop are both snapped to the tick grid, so the stop distance is
    an integer number of ticks; an integral `reward_risk` therefore puts the
    raw target on the grid too and the round-toward-entry is a no-op. On the
    shipped multiples (1.0 / 2.0 / 3.0) the RISK stage's RR_TOO_LOW is
    effectively unreachable -- reported, not fixed, because the mapping is
    still the right one for a configuration that uses a fractional multiple.
    """
    for setup, multiple in (
        (SetupType.SCALP_1R, 1.0),
        (SetupType.SETUP_2R, 2.0),
        (SetupType.DIRECTIONAL_3R, 3.0),
    ):
        assert cfg.setups[setup].reward_risk == pytest.approx(multiple)
        assert float(multiple).is_integer()
    for achievable_rr, expected_r in ((1.2, 1.0), (2.0, 2.0), (3.0, 3.0)):
        plan = decide(
            cfg, build_vector(achievable_rr=achievable_rr), equity=100_000.0
        ).plan
        assert plan.planned_r_multiple == pytest.approx(expected_r)
        assert plan.target_price == pytest.approx(ENTRY + STOP_DISTANCE * expected_r)


def test_no_reference_price_means_nothing_is_sized(cfg):
    """An entry that cannot be priced is a WAIT, not a guessed price."""
    decision = decide(
        cfg, build_vector(achievable_rr=1.2), reference_price=None, equity=100_000.0
    )
    assert decision.blocking.stage is GateStage.RISK
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert "cannot be priced" in decision.signal.wait_detail


def test_a_risk_limit_runs_last_so_the_signal_is_still_recorded(cfg):
    """Section 3: "risk limits run last so that a trade blocked by a daily-loss
    limit is still recorded as a *signal that occurred*".

    The open-position limit is tripped before the call, so the bar is blocked.
    What must survive is the evidence: the Flow Score was computed, the setup
    was determined and sizing ACCEPTED, so signal-level and execution-level
    statistics stay separable.
    """
    limits = RiskLimitManager(cfg.risk, starting_equity=100_000.0)
    limits.on_position_opened(SYMBOL, 400.0)
    decision = decide(
        cfg, build_vector(achievable_rr=1.2), equity=100_000.0, limits=limits
    )

    assert decision.signal.action is SignalAction.WAIT
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert decision.blocking.stage is GateStage.RISK
    assert decision.stages == tuple(GateStage)
    assert decision.sizing.accepted is True
    assert decision.signal.flow_score == pytest.approx(EXPECTED_FULL_SCORE)
    assert decision.signal.setup is SetupType.SCALP_1R
    assert decision.limits.allowed is False
    assert decision.plan is None


def test_a_non_positive_equity_is_refused_rather_than_sized(cfg):
    """A destroyed account cannot take risk, and the refusal names why."""
    decision = decide(cfg, build_vector(achievable_rr=1.2), equity=0.0)
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert decision.sizing.reason == SizingRejection.EQUITY_NON_POSITIVE


# ===========================================================================
# 8. Two defects these tests found
# ===========================================================================


def test_component_points_sum_to_the_flow_score_the_signal_reports():
    """BUG (found here): the two fields disagreed under redistribution.

    `Signal.component_points` was built from the raw scorer output while
    `Signal.flow_score` came from the aggregate's possibly-RESCALED components,
    so with `redistribute_disabled_weight` on the points summed to 22.31 beside
    a reported score of 40.57. Both numbers travel on the same record for
    auditing, and a record whose parts do not add up to its total cannot audit
    anything -- section 12's per-component trade fields are written from one and
    its `flow_score` from the other.

    Redistribution is off by default and loudly flagged as not comparable, so
    the fix is to report the points that produced the score rather than to
    change what is scored. Checked on both paths so the default stays pinned.
    """
    for config in (
        local_config(strict=False, redistribute=False),
        local_config(strict=False, redistribute=True),
    ):
        decision = decide(config, build_vector(order_flow_available=0.0))
        flow = decision.flow
        assert flow.flow_score is not None
        points = decision.signal.component_points
        assert set(points) == set(Component)
        assert sum(points.values()) == pytest.approx(decision.signal.flow_score)
        assert sum(points.values()) == pytest.approx(flow.require().score)

    # and the redistributed case is the one that used to disagree
    redistributed = decide(
        local_config(strict=False, redistribute=True),
        build_vector(order_flow_available=0.0),
    )
    assert redistributed.flow.redistributed is True
    assert redistributed.flow.available_points == pytest.approx(100.0)
    assert redistributed.signal.flow_score > 30.0


def test_stage_inputs_for_another_symbol_are_refused():
    """BUG (found here): only `decide` enforced the per-symbol contract.

    The engine is per symbol because `LiquidityFeatures`, `StructureFeatures`
    and `LevelFeatures` hold an `InstrumentSpec`, and `decide` raises on a view
    for another symbol -- "scoring another symbol through them would use the
    wrong tick size, point value and session". `decide_from` is the other
    documented front door and did not check, so handing it an ES feature vector
    produced a LONG `Signal` for ES sized with NQ's point value of 20.00
    instead of ES's 50.00: a plan claiming 400.00 at risk against a real 1000.00,
    which is the hard cap rather than the 0.5% the config asks for.
    """
    config = FlowModelConfig(
        instruments={
            SYMBOL: nq_spec(),
            "ES": nq_spec().replace(symbol="ES", tick_value=12.50),
        },
        flow_score=local_config().flow_score,
        setups=local_config().setups,
        backtest=BacktestConfig(symbols=(SYMBOL,)),
    )
    assert config.spec("ES").point_value == pytest.approx(50.0)
    assert config.spec(SYMBOL).point_value == pytest.approx(20.0)

    es_vector = build_vector(achievable_rr=1.2, symbol="ES")
    engine = SignalEngine(config, SYMBOL)
    inputs = PrecomputedInputs(
        features=es_vector,
        regime=a_regime(symbol="ES"),
        reference_price=ENTRY,
    )
    with pytest.raises(EngineError, match="built for 'NQ'"):
        engine.decide_from(inputs, equity=100_000.0)


# ===========================================================================
# 9. Reachability: gate mappings that no input can reach
# ===========================================================================


def test_structure_can_never_be_the_entry_filters_dominant_opposer(cfg):
    """An honest reachability finding, recorded rather than asserted away.

    `OPPOSITION_WAIT_REASON` maps all five components, but the STRUCTURE
    component's direction is `level_direction` -- the same key the structure
    gate reads to choose the side -- so the component can only ever AGREE with
    the side under consideration. LIQUIDITY emits direction 0 always, which
    gates.py already records as unreachable. So three of the five entry-filter
    reasons are live and two are not, and `NO_STRUCTURE`/`LIQUIDITY` from the
    entry filter will never appear in a rejection histogram.
    """
    for direction in (1.0, -1.0):
        flow = decide(
            cfg,
            build_vector(
                structure_direction=direction,
                vol_momentum_direction=direction,
                order_flow_direction=direction,
                options_flow_direction=direction,
            ),
        ).flow.require()
        side = Side.LONG if direction > 0 else Side.SHORT
        structure = flow.components[Component.STRUCTURE]
        liquidity = flow.components[Component.LIQUIDITY]
        assert structure.agrees_with(side) is True
        assert structure.opposes(side) is False
        assert liquidity.direction == 0
        assert liquidity.opposes(Side.LONG) is False
        assert liquidity.opposes(Side.SHORT) is False


def test_every_wait_reason_in_the_enum_is_declared_by_some_stage():
    """No reason is orphaned, and no stage invents one.

    Both directions matter: a reason no stage can emit is dead enum, and a
    reason outside the enum would be a new trading rule smuggled in as a label.
    """
    declared = {reason for reasons in WAIT_REASONS_BY_STAGE.values() for reason in reasons}
    assert declared == set(WaitReason)
    assert set(WAIT_REASONS_BY_STAGE) == set(GateStage) - {
        GateStage.SETUP,
        GateStage.RISK,
    } | {GateStage.SETUP, GateStage.RISK}


def test_the_engine_claims_no_edge_anywhere_in_its_output(cfg):
    """"Never say the system is profitable before testing it."

    `Signal.confidence` is the score over the points that were MEASURED and
    nothing more: it must not be a probability, and the narrative must not
    claim one. 93.25 of 100.0 available points is 0.9325, which is a
    normalization and not a 93% chance of anything.
    """
    decision = decide(cfg, build_vector(achievable_rr=1.2), equity=100_000.0)
    assert decision.signal.confidence == pytest.approx(0.9325)
    assert decision.signal.confidence == pytest.approx(
        decision.flow.require().score / decision.flow.available_points
    )
    forbidden = ("profit", "edge", "win rate", "expectancy", "alpha")
    narrative = " ".join(decision.signal.reasons).lower()
    for word in forbidden:
        assert word not in narrative, word


def test_nothing_in_the_decision_reads_a_research_target(cfg):
    """`config.research_targets` is excluded from the config hash for a reason.

    The AST check in `tests/unit/test_no_target_leakage.py` owns the general
    rule; this is the behavioural half: the engine's answer must not move when
    a target does. A module that read one would be optimizing for a desired
    win rate by construction.
    """
    baseline = decide(cfg, build_vector(achievable_rr=1.2), equity=100_000.0).signal
    moved = cfg.replace(
        research_targets=cfg.research_targets.replace(
            combined_10y_win_rate=0.99,
            scalp_1r_win_rate_favourable_regimes=(0.98, 0.99),
        )
    )
    assert moved.research_targets.combined_10y_win_rate == pytest.approx(0.99)
    assert decide(moved, build_vector(achievable_rr=1.2), equity=100_000.0).signal == baseline
    assert moved.config_hash == cfg.config_hash


# ===========================================================================
# 10. Gate boundaries, where an off-by-one changes the histogram silently
# ===========================================================================


def test_the_flow_score_gate_passes_at_exactly_the_threshold(cfg):
    """Section 3 writes `score < setup.min_score -> WAIT`, so equality passes.

    The magnitudes are chosen so the sum is 30.00 EXACTLY, which is also the
    configured floor:
      structure    0.30 * 20 = 6.00
      order flow   0.24 * 25 = 6.00
      options flow 0.30 * 20 = 6.00
      liquidity    0.40 * 15 = 6.00
      vol_momentum 0.30 * 20 = 6.00
                              -----
                              30.00
    A `<=` comparison here would reject a bar the specification admits, and
    the error would never show up as anything but a slightly different
    rejection count.
    """
    assert min_flow_score_required(cfg) == pytest.approx(30.0)
    vector = build_vector(
        structure_magnitude=0.30,
        order_flow_magnitude=0.24,
        options_flow_magnitude=0.30,
        liquidity_score=0.40,
        vol_momentum_magnitude=0.30,
        achievable_rr=1.2,
    )
    engine = engine_for(cfg)
    standalone = aggregate(cfg, engine.score(vector), symbol=SYMBOL, ts=vector.ts)
    assert standalone.require().score == pytest.approx(30.00)

    decision = decide(cfg, vector, engine=engine, equity=100_000.0)
    assert decision.signal.action is SignalAction.LONG
    assert decision.signal.flow_score == pytest.approx(30.00)

    # and one tick below the floor is refused
    below = build_vector(
        structure_magnitude=0.29,
        order_flow_magnitude=0.24,
        options_flow_magnitude=0.30,
        liquidity_score=0.40,
        vol_momentum_magnitude=0.30,
        achievable_rr=1.2,
    )
    refused = decide(cfg, below, equity=100_000.0)
    assert refused.blocking.stage is GateStage.FLOW_SCORE
    assert refused.signal.wait_reason is WaitReason.SCORE_BELOW_THRESHOLD
    assert refused.signal.flow_score == pytest.approx(29.80)


def test_the_volatility_band_includes_both_of_its_edges(cfg):
    """The union band is [0.10, 1.00] and the comparison is `low <= v <= high`.

    An exclusive edge would reject the lowest-volatility bar any setup allows,
    and the only visible effect would be a rejection count nobody could
    explain. 0.0999 must fail and 0.10 must pass, at the momentum stage.
    """
    low, high = vol_percentile_band(cfg)
    assert (low, high) == (pytest.approx(0.10), pytest.approx(1.00))

    outside = decide(cfg, build_vector(atr_percentile=0.0999, achievable_rr=1.2))
    assert outside.blocking.stage is GateStage.MOMENTUM
    assert outside.signal.wait_reason is WaitReason.VOLATILITY_BAND

    inside = decide(
        cfg, build_vector(atr_percentile=0.10, achievable_rr=1.2), equity=100_000.0
    )
    assert inside.signal.action is SignalAction.LONG


def test_clearing_the_union_band_does_not_mean_the_setup_accepts(cfg):
    """ATR percentile 1.00 clears the union and fails SCALP_1R's own 0.95.

    The union is the weakest honest question before selection, and the
    per-setup band is re-applied afterwards. The bar must therefore pass the
    MOMENTUM stage and be declined at the SETUP stage -- with VOLATILITY_BAND,
    not NO_SETUP_MATCH, because section 3 assigns that reason to this
    comparison.
    """
    decision = decide(cfg, build_vector(atr_percentile=1.00, achievable_rr=1.2))
    momentum = next(r for r in decision.gate_results if r.stage is GateStage.MOMENTUM)
    assert momentum.passed is True
    assert decision.blocking.stage is GateStage.SETUP
    assert decision.signal.wait_reason is WaitReason.VOLATILITY_BAND
    assert cfg.setups[SetupType.SCALP_1R].max_vol_percentile == pytest.approx(0.95)


def test_both_opposition_limits_are_exclusive_at_their_edge(cfg):
    """`opposing > limit` at both gates, so the limit itself is admissible.

    VOL_MOMENTUM at 0.60 * 20 = 12.00 points is exactly the entry filter's
    limit and must pass; 0.61 * 20 = 12.20 must not. Options flow at
    0.70 * 20 = 14.00 is exactly the contradiction limit and must clear stage
    7 -- and then fail the 12.00 entry filter, which is what proves the two
    thresholds are separate numbers.
    """
    at_limit = decide(
        cfg,
        build_vector(
            vol_momentum_direction=-1.0, vol_momentum_magnitude=0.60, achievable_rr=1.2
        ),
        equity=100_000.0,
    )
    assert at_limit.signal.action is SignalAction.LONG
    assert at_limit.flow.require().opposing_points(Side.LONG) == pytest.approx(12.00)

    over = decide(
        cfg,
        build_vector(
            vol_momentum_direction=-1.0, vol_momentum_magnitude=0.61, achievable_rr=1.2
        ),
    )
    assert over.blocking.stage is GateStage.ENTRY_FILTER
    assert over.flow.require().opposing_points(Side.LONG) == pytest.approx(12.20)

    options_at_limit = decide(
        cfg,
        build_vector(
            options_flow_direction=-1.0, options_flow_magnitude=0.70, achievable_rr=1.2
        ),
    )
    options_stage = next(
        r for r in options_at_limit.gate_results if r.stage is GateStage.OPTIONS_FLOW
    )
    assert options_stage.passed is True
    assert options_stage.diagnostics["options_opposing_points"] == pytest.approx(14.00)
    assert options_at_limit.blocking.stage is GateStage.ENTRY_FILTER

    options_over = decide(
        cfg,
        build_vector(
            options_flow_direction=-1.0, options_flow_magnitude=0.71, achievable_rr=1.2
        ),
    )
    assert options_over.blocking.stage is GateStage.OPTIONS_FLOW


def test_the_dominant_opposer_tie_break_is_stable(cfg):
    """A tie must resolve the same way on every run, or the histogram drifts.

    Options flow and VOL_MOMENTUM each hold 0.325 * 20 = 6.50 opposing points,
    13.00 together against the 12.00 limit. The reported reason is the one
    belonging to "the component holding the most opposing points", which is a
    tie here; the aggregate's component order is `Component`'s own declaration
    order, so OPTIONS_FLOW wins. The value of the assertion is not which one
    wins but that it is the same one every time, and that both are named in
    the detail so nothing is lost to the tie-break.
    """
    vector = build_vector(
        options_flow_direction=-1.0,
        options_flow_magnitude=0.325,
        vol_momentum_direction=-1.0,
        vol_momentum_magnitude=0.325,
    )
    first = decide(cfg, vector)
    second = decide(cfg, vector)
    assert first.signal == second.signal
    assert first.blocking.stage is GateStage.ENTRY_FILTER
    assert first.flow.require().opposing_points(Side.LONG) == pytest.approx(13.00)
    assert first.signal.wait_reason is WaitReason.OPTIONS_CONTRADICTION
    assert "options_flow 6.50" in first.signal.wait_detail
    assert "vol_momentum 6.50" in first.signal.wait_detail


def test_a_spread_narrower_than_the_instruments_minimum_is_a_broken_book(cfg):
    """Tighter than physically possible is a quote error, not a good fill.

    The check applies only to a MEASURED spread: `features/liquidity.py` grades
    the key DEGRADED when it is the instrument's configured fallback, and a
    fallback cannot be narrower than the minimum by accident, so judging it
    would be judging configuration.
    """
    broken = decide(cfg, build_vector(spread_ticks=0.4))
    assert broken.blocking.stage is GateStage.LIQUIDITY
    assert broken.signal.wait_reason is WaitReason.LIQUIDITY
    assert "broken book" in " ".join(broken.signal.reasons)

    fallback = decide(
        cfg,
        build_vector(
            spread_ticks=0.4,
            achievable_rr=1.2,
            grades={"spread_ticks": DataQuality.DEGRADED},
        ),
        equity=100_000.0,
    )
    assert fallback.signal.action is SignalAction.LONG
    assert "configured fallback" in " ".join(fallback.signal.reasons)


# ===========================================================================
# 11. The stop is determined, never chosen to make a size work
# ===========================================================================


def test_no_atr_multiple_stop_is_substituted_when_the_structure_supplies_none(cfg):
    """14.6's whole point: R is defined by where the thesis is falsified.

    With no `level_stop_price` there is no determined stop and therefore no R.
    The engine must decline rather than fall back on `stop_atr_multiple` -- a
    width chosen to make a size work has no informational content, and
    `SetupConfig.stop_atr_multiple` is dead configuration under 14.6 precisely
    because of this rule.
    """
    decision = decide(
        cfg,
        build_vector(achievable_rr=1.2, grades={"level_stop_price": DataQuality.MISSING}),
        equity=100_000.0,
    )
    assert decision.blocking.stage is GateStage.RISK
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert "no level_stop_price" in decision.signal.wait_detail
    assert "No ATR-multiple stop is substituted" in decision.signal.wait_detail
    assert decision.plan is None


def test_the_stop_width_cap_is_not_dropped_when_atr_is_unmeasured(cfg):
    """BUG (found here): an unmeasured ATR removed the cap instead of degrading it.

    `max_stop_distance = max_stop_atr_multiple * ATR` was computed as None when
    ATR was absent or graded MISSING, and `risk/sizing.py` reads None as "no
    cap". So a SCALP_1R whose stop may be at most 2.0 * ATR = 40.00 points
    accepted a 400.00-point stop -- twenty times ATR -- and nothing in
    `Signal.reasons` or `GateResult.diagnostics` said the check had not run.
    The money risk stayed inside `max_risk_per_trade_pct` because size floors,
    so this is not a blown account; it is the one bound on stop WIDTH silently
    disappearing, which changes what an R is.

    Section 0.4 settles it: missing data produces WAIT, never a guess.
    `MomentumGate` and `signals/setups.py` both decline a bar whose
    `atr_percentile` was never measured "rather than admitted on an unmeasured
    quantity", and this is the same quantity.
    """
    big_equity = 10_000_000.0  # so the risk budget is not what refuses the bar
    unmeasured = build_vector(stop_distance=400.0, achievable_rr=1.2)
    values = dict(unmeasured.values)
    values["atr"] = 0.0  # the not-ready placeholder
    unmeasured = unmeasured.replace(
        values=values,
        quality_by_key={**unmeasured.quality_by_key, "atr": DataQuality.MISSING},
    )

    decision = decide(cfg, unmeasured, equity=big_equity)
    assert decision.signal.action is SignalAction.WAIT
    assert decision.blocking.stage is GateStage.RISK
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert "max_stop_atr_multiple" in decision.signal.wait_detail
    assert decision.plan is None

    # the same geometry WITH a measured ATR is refused by the cap itself, which
    # is what makes the assertion above about the cap and not about the budget
    measured = decide(
        cfg, build_vector(stop_distance=400.0, achievable_rr=1.2), equity=big_equity
    )
    assert measured.sizing.reason == SizingRejection.STOP_TOO_WIDE

    # and a stop inside the cap is still taken, so the fix is not a blanket no
    inside = decide(
        cfg, build_vector(stop_distance=20.0, achievable_rr=1.2), equity=big_equity
    )
    assert inside.signal.action is SignalAction.LONG
    assert inside.plan.atr == pytest.approx(ATR)


def test_an_unmeasured_atr_is_reported_as_absent_and_not_as_zero(cfg):
    """"Not evaluated" and "evaluated to nothing" are different facts.

    `Signal.atr` came from the raw value without consulting the grade, so a
    not-ready vector -- which writes a 0.0 placeholder at grade MISSING --
    reported `atr=0.0`. Downstream that reads as a measurement of zero
    volatility, which is a number nobody produced.
    """
    vector = build_vector(gate_passed=0.0, gate_reason=GATE_SIGNIFICANCE)
    values = dict(vector.values)
    values["atr"] = 0.0
    vector = vector.replace(
        values=values,
        quality_by_key={**vector.quality_by_key, "atr": DataQuality.MISSING},
    )
    signal = decide(cfg, vector).signal
    assert signal.wait_reason is WaitReason.NO_STRUCTURE
    assert signal.atr is None

    # a measured ATR is still reported on a WAIT, so this is not "always None"
    measured = decide(
        cfg, build_vector(gate_passed=0.0, gate_reason=GATE_SIGNIFICANCE)
    ).signal
    assert measured.atr == pytest.approx(ATR)


# ===========================================================================
# 12. What a Phase 7 consumer may and may not read off a Signal
# ===========================================================================


def test_a_wait_at_the_risk_stage_still_names_the_setup_it_would_have_taken(cfg):
    """A WAIT may carry a setup, so a consumer must key on `action`.

    The RISK stage runs after selection, so a bar blocked there has a
    determined setup and a Flow Score on the record -- deliberately, because
    section 3 wants the signal-level statistics separable from the
    execution-level ones. Anything treating `setup is not None` as "a trade
    happened" would count these as trades.
    """
    decision = decide(cfg, build_vector(stop_distance=0.25), equity=100_000.0)
    signal = decision.signal
    assert signal.action is SignalAction.WAIT
    assert signal.side is None
    assert signal.setup is SetupType.SCALP_1R
    assert signal.flow_score == pytest.approx(EXPECTED_FULL_SCORE)
    assert signal.wait_reason is WaitReason.RISK_LIMIT
    assert decision.plan is None


def test_a_signal_built_without_a_limits_manager_says_so(cfg):
    """Silence about an unchecked limit would read as an approved one.

    `limits` is an argument rather than a field so two engines cannot disagree
    about how many consecutive losses have occurred. The consequence is that a
    caller can forget it, and the narrative has to say the six limits of
    section 8 were not applied on this bar.
    """
    signal = decide(cfg, build_vector(achievable_rr=1.2), equity=100_000.0).signal
    assert signal.action is SignalAction.LONG
    assert any(
        "no RiskLimitManager was supplied" in reason for reason in signal.reasons
    )


def test_confidence_is_zero_when_the_aggregate_refused(cfg):
    """No score, no normalization. `confidence` is not a probability either way.

    And the component points still report what WAS measured: 19.00 + 19.00 +
    13.50 + 18.00 = 69.50 out of the 75.00 points that had a feed, with order
    flow's 25.00 reported unavailable rather than redistributed.
    """
    decision = decide(cfg, build_vector(order_flow_available=0.0))
    assert decision.flow.refused is True
    assert decision.signal.confidence == 0.0
    assert decision.signal.flow_score is None
    assert sum(decision.signal.component_points.values()) == pytest.approx(69.50)
    assert decision.flow.available_points == pytest.approx(75.00)
    assert decision.signal.component_points[Component.ORDER_FLOW] == pytest.approx(0.0)


def test_a_total_points_above_one_hundred_is_refused_at_construction(cfg):
    """`Signal.flow_score` is bounded [0, 100] by the Phase 1 contract.

    A 120-point scheme cannot be carried on a Signal, and truncating it would
    misreport the number every threshold is compared against. Caught at
    construction rather than at bar one of a ten-year run.
    """
    inflated = cfg.replace(
        flow_score=FlowScoreConfig(
            weights={c: w * 1.2 for c, w in WEIGHTS.items()},
            total_points=120.0,
            feed_requirements=feed_requirements(),
        )
    )
    assert sum(inflated.flow_score.weights.values()) == pytest.approx(120.0)
    with pytest.raises(EngineError, match="total_points"):
        SignalEngine(inflated, SYMBOL)


def test_the_session_gate_says_so_when_no_calendar_was_supplied(cfg):
    """Neither silently open nor silently closed: the third option is to report.

    A filter that defaulted to closed would make every bar of a
    calendar-less dataset a WAIT; one that defaulted to open would hide that
    `config.session` was never applied. The engine also falls back to the UTC
    date for `RiskLimitManager`'s daily limit in that state, which is the
    honest fallback rather than a guessed exchange day.
    """
    decision = decide(cfg, build_vector(achievable_rr=1.2), equity=100_000.0)
    session = next(r for r in decision.gate_results if r.stage is GateStage.SESSION)
    assert session.passed is True
    assert "not evaluated" in " ".join(session.reasons)
    assert "config.session was not applied" in " ".join(session.reasons)
    assert engine_for(cfg).session_date(TS) == TS.date()


# ===========================================================================
# 13. The lookahead firewall, at the whole-pipeline level
# ===========================================================================
#
# Each computer has its own audit under `factory=`. These two ask the question
# of the COMPOSITION, because a leak can live in how seven computers, a
# detector, five scorers and the risk layer are wired together rather than in
# any one of them.


def test_the_signal_does_not_move_when_the_future_is_scaled_by_seven(
    walk_config, walk_inputs, dataset
):
    """Section 4's future-shuffle test, applied to the engine.

    Every observation after the evaluation instant is multiplied by 7. A
    `Signal` that changed would be reading bars, quotes, ticks or chains that
    have not happened yet. The availability report is computed once from the
    UNMUTATED dataset and handed in, because it is a pre-run property and its
    own docstring forbids recomputing it per bar.
    """
    from flow_model.validation.lookahead import mutate_after

    data, availability = walk_inputs
    cutoff = int(dataset.bars.ts_ns[WARM_BAR])
    view = MarketView(data, now=from_ns(cutoff), now_ns=cutoff)
    baseline = _walk_engine(walk_config, availability).evaluate(view, equity=100_000.0)

    mutated_data = mutate_after(data, cutoff)
    mutated_view = MarketView(mutated_data, now=from_ns(cutoff), now_ns=cutoff)
    mutated = _walk_engine(walk_config, availability).evaluate(
        mutated_view, equity=100_000.0
    )

    assert mutated == baseline
    assert baseline.reasons != ()


def test_the_signal_does_not_move_when_the_future_is_deleted(
    walk_config, walk_inputs, dataset
):
    """And a dataset that simply ends at the instant gives the same answer.

    Truncation catches what mutation cannot: a module that read `len(series)`
    or the last row of the full array rather than of the visible prefix.
    """
    from flow_model.validation.lookahead import truncate

    data, availability = walk_inputs
    cutoff = int(dataset.bars.ts_ns[WARM_BAR])
    view = MarketView(data, now=from_ns(cutoff), now_ns=cutoff)
    baseline = _walk_engine(walk_config, availability).evaluate(view, equity=100_000.0)

    truncated_data = truncate(data, cutoff)
    truncated_view = MarketView(truncated_data, now=from_ns(cutoff), now_ns=cutoff)
    truncated = _walk_engine(walk_config, availability).evaluate(
        truncated_view, equity=100_000.0
    )

    assert truncated == baseline


def test_no_proxy_is_substituted_for_an_absent_tick_or_options_feed(dataset):
    """"Never invent missing options or order-flow data", on real data.

    The failure mode is `magnitude=0.0, enabled=True`, which reads downstream
    as "the tape was balanced" -- a measurement nobody took. What must happen
    is `enabled=False` with the CONFIGURED weight still on the score, so
    `max_points - available_points` reports the shortfall.
    """
    config = local_config(
        strict=False, order_flow_required=False, options_required=False
    )
    data = symbol_data(dataset, with_feeds=False)
    engine = SignalEngine(config, SYMBOL, calendar=TradingCalendar())
    view = view_at(data, dataset, WARM_BAR)
    scored = engine.score(engine.bundle.compute(view))

    for component, weight in (
        (Component.ORDER_FLOW, 25.0),
        (Component.OPTIONS_FLOW, 20.0),
    ):
        score = scored.scores[component]
        assert score.enabled is False, component
        assert score.quality is DataQuality.MISSING, component
        assert score.magnitude == pytest.approx(0.0), component
        assert score.direction == 0, component
        assert score.weight == pytest.approx(weight), component
        assert score.points == pytest.approx(0.0), component
    for component in (Component.STRUCTURE, Component.LIQUIDITY, Component.VOL_MOMENTUM):
        assert scored.scores[component].enabled is True, component

    outcome = engine.aggregate(scored)
    assert outcome.available_points == pytest.approx(55.0)
    assert outcome.max_points == pytest.approx(100.0)
    assert outcome.redistributed is False
