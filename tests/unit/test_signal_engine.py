"""Phase 6: the Flow Score aggregate, the gate sequence, setups, the engine.

What this file establishes, in order of how much it matters:

1. **Section 7's rule, adversarially.** A strongly bearish component keeps a
   LARGE magnitude, the aggregate does not cancel, and NO TRADE is taken in
   either direction -- including the case where the structure names one side
   and every other component points the other way, which is where a system
   that "quietly takes the majority direction" would reveal itself.
2. **The strict-availability refusal.** With a component unmeasured, the
   aggregate refuses rather than returning a partial sum, and the refusal is
   an outcome the engine records as a WAIT, not an exception.
3. **Setup selection is a measurement.** The band is read off
   `achievable_rr`; a bar that supports 1R is not promoted to 2R and one that
   supports 3R is not truncated.
4. **The gate order is section 3's**, checked against `GateStage` rather
   than asserted in prose, and every WAIT carries a reason.
5. **The engine is stateless**, so a sequential walk and a cold single call
   produce identical signals -- the property section 12's backtest/live
   parity test will rest on.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from flow_model.config.loader import load_config
from flow_model.core.contracts import ComponentScore, FeatureVector, RegimeState
from flow_model.core.enums import (
    Component,
    DataQuality,
    Feed,
    Regime,
    SetupType,
    Side,
    SignalAction,
    WaitReason,
)
from flow_model.data.calendar import TradingCalendar
from flow_model.data.market_view import MarketView
from flow_model.data.quality import QualityGrader
from flow_model.data.series import from_ns
from flow_model.data.store import SymbolData
from flow_model.data.synthetic import SyntheticConfig, SyntheticMarketGenerator
from flow_model.features.levels import GATE_PASSED, setup_class_for
from flow_model.risk.limits import RiskLimitManager
from flow_model.signals import flow_score as flow_score_module
from flow_model.signals import gates as gate_module
from flow_model.signals import setups as setups_api
from flow_model.signals.engine import (
    EngineError,
    PrecomputedInputs,
    SIZING_WAIT_REASONS,
    SignalEngine,
    feature_computers,
)
from flow_model.signals.flow_score import (
    FlowScoreError,
    REFUSAL_WAIT_REASON,
    ScoredComponents,
    aggregate,
    all_scorers,
    score_components,
)
from flow_model.signals.gates import (
    GateResult,
    GateStage,
    WAIT_REASONS_BY_STAGE,
    assess_warmup,
)
from flow_model.validation.lookahead import mutate_after, truncate

SYMBOL = "NQ"
INTERVAL = 300
TS = datetime(2024, 3, 1, 15, 0, tzinfo=timezone.utc)
ENTRY = 18_000.0
ATR = 20.0

#: The LONG bar the real-data walk finds on the shared synthetic dataset,
#: with the relaxed thresholds `relaxed_config` supplies. Pinned rather than
#: searched for, because walking six thousand bars inside a unit test is not
#: worth the seconds; the walk that found it is reproducible from the same
#: seed and the same dataset fixture.
REAL_LONG_BAR = 2094

#: A bar inside the tradable RTH session, well past every warmup window. The
#: dataset's LAST bar is deliberately not used for this: it closes at the RTH
#: close, so the session gate rejects it first and the later stages never
#: run.
IN_SESSION_BAR = REAL_LONG_BAR


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def relaxed_config():
    """The shipped config with the setups' score floors lowered.

    The shipped `min_flow_score` values (70/75/80) are the brief's prior and
    are NOT changed on disk -- principle 7 makes them hypotheses for the
    Phase 8 sweep, and lowering a default to make an output look better is
    exactly what the brief forbids. A TEST may ask what happens at a
    different threshold, which is what this fixture does, in memory, for the
    end-to-end trade path only.
    """
    return load_config(
        overrides=[
            "setups.SCALP_1R.min_flow_score=30.0",
            "setups.SETUP_2R.min_flow_score=30.0",
            "setups.DIRECTIONAL_3R.min_flow_score=30.0",
            "setups.SCALP_1R.allowed_regimes="
            "[LOW_VOL,HIGH_VOL,TRENDING_UP,TRENDING_DOWN,CHOP,DIRECTIONAL_SHIFT]",
        ]
    )


# ---------------------------------------------------------------------------
# hand-built inputs
# ---------------------------------------------------------------------------


def _structure_keys(
    *,
    direction: float,
    magnitude: float,
    achievable_rr: float,
    stop_price: float,
    gate_passed: float = 1.0,
    gate_reason: float = GATE_PASSED,
    min_reward_risk: float = 1.0,
) -> dict[str, float]:
    """Every key the STRUCTURE scorer, the structure gate and `select` read.

    `level_setup_class` is computed with `features/levels.py`'s own
    `setup_class_for`, so the builder can never hand the selector a class
    inconsistent with its `achievable_rr` -- the selector raises on that, and
    a test that triggered the raise would be testing the builder.
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
        "level_setup_class": setup_class_for(achievable_rr, min_reward_risk),
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


def _liquidity_keys(score: float = 0.90) -> dict[str, float]:
    return {
        "liquidity_score": score,
        "spread_ticks": 1.0,
        "spread_percentile": 0.3,
        "depth_imbalance": 0.0,
        "volume_percentile": 0.7,
        "relative_volume": 1.2,
        "volume_trend": 0.1,
        "dollar_volume": 5.0e8,
        "participation_cost_ticks": 1.0,
    }


def _vol_momentum_keys(
    *, direction: float, magnitude: float, atr_percentile: float = 0.50
) -> dict[str, float]:
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
        "vol_expansion": 0.1,
        "roc": 0.004 * direction,
        "roc_atr": 1.2 * direction,
        "efficiency_ratio": 0.6,
        "acceleration": 0.1 * direction,
        "momentum_persistence": 0.6,
        "up_bar_fraction": 0.5 + 0.1 * direction,
        "close_position": 0.5 + 0.2 * direction,
    }


def _order_flow_keys(
    *, direction: float, magnitude: float, available: float = 1.0
) -> dict[str, float]:
    return {
        "order_flow_available": available,
        "order_flow_available_weight": 1.0,
        "order_flow_magnitude": magnitude,
        "order_flow_direction": direction,
        "signed_delta": 0.5 * direction,
        "cvd_slope": 0.3 * direction,
        "aggression_ratio": 0.6,
        "absorption": 0.0,
        "absorption_direction": 0.0,
        "avg_trade_size_percentile": 0.55,
        "max_trade_size_percentile": 0.6,
        "large_trade_event": 0.0,
        "classification_coverage": 1.0,
    }


def _options_flow_keys(*, direction: float, magnitude: float) -> dict[str, float]:
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
    vol_momentum_direction: float = 1.0,
    vol_momentum_magnitude: float = 0.95,
    order_flow_direction: float = 1.0,
    order_flow_magnitude: float = 0.95,
    options_flow_direction: float = 1.0,
    options_flow_magnitude: float = 0.95,
    liquidity_score: float = 0.90,
    atr_percentile: float = 0.50,
    stop_distance: float = ATR,
    gate_passed: float = 1.0,
    gate_reason: float = GATE_PASSED,
    order_flow_available: float = 1.0,
    quality: DataQuality = DataQuality.GOOD,
    grades: dict[str, DataQuality] | None = None,
    ts: datetime | None = None,
) -> FeatureVector:
    """A complete, internally consistent `FeatureVector`.

    Complete in the sense the scorers require: the two feed-dependent
    scorers refuse a vector missing any declared read key, because an absent
    key there means a rename rather than a data condition.
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
        )
    )
    values.update(_liquidity_keys(liquidity_score))
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
    quality_by_key = {key: quality for key in values}
    quality_by_key.update(grades or {})
    return FeatureVector(
        symbol=SYMBOL,
        ts=ts or TS,
        values=values,
        quality_by_key=quality_by_key,
    )


def regime_state(
    regime: Regime = Regime.TRENDING_UP,
    *,
    vol_percentile: float = 0.5,
    ts: datetime | None = None,
) -> RegimeState:
    return RegimeState(
        symbol=SYMBOL,
        ts=ts or TS,
        regime=regime,
        confidence=0.8,
        vol_percentile=vol_percentile,
        bars_in_regime=12,
    )


def engine_for(config, **kwargs) -> SignalEngine:
    """An engine with no calendar and no grader-driven gates.

    `PrecomputedInputs` carries `quality=None` and `warmup=None`, so the
    market-data and warmup gates pass with a reason saying they were not
    evaluated. That is deliberate: a hand-built vector has no feeds to grade,
    and the gates say so rather than defaulting either way.
    """
    return SignalEngine(config, SYMBOL, **kwargs)


def decide(config, vector: FeatureVector, *, regime: Regime | RegimeState = Regime.TRENDING_UP, **kwargs):
    state = regime if isinstance(regime, RegimeState) else regime_state(regime, ts=vector.ts)
    inputs = PrecomputedInputs(
        features=vector, regime=state, reference_price=ENTRY
    )
    return engine_for(config).decide_from(inputs, **kwargs)


# ---------------------------------------------------------------------------
# 1. section 7: a bearish component is evidence, not a vote to be outvoted
# ---------------------------------------------------------------------------


def test_the_aggregate_does_not_cancel_a_disagreement(cfg):
    """One bearish component against four bullish: magnitude stays HIGH.

    Section 7's whole point. The Flow Score is a MAGNITUDE aggregate, so a
    component pointing the other way contributes its full magnitude; it does
    not net against the rest. A score that fell here would mean the
    disagreement had been absorbed into the number instead of being left for
    the gates.
    """
    vector = build_vector(structure_direction=1.0, vol_momentum_direction=-1.0)
    scored = score_components(cfg, vector)
    outcome = aggregate(cfg, scored)
    flow = outcome.require()

    assert outcome.available_points == pytest.approx(100.0)
    assert flow.score > 0.85 * outcome.available_points
    assert flow.components[Component.VOL_MOMENTUM].magnitude > 0.9
    assert flow.components[Component.VOL_MOMENTUM].direction == -1


def test_both_sides_are_opposed_so_neither_reads_as_the_majority(cfg):
    """With the components in conflict, each side has weight against it."""
    vector = build_vector(structure_direction=1.0, vol_momentum_direction=-1.0)
    flow = aggregate(cfg, score_components(cfg, vector)).require()

    assert flow.opposing_points(Side.LONG) > 0.0
    assert flow.opposing_points(Side.SHORT) > 0.0


def test_no_trade_is_taken_in_either_direction_when_components_conflict(cfg):
    """The brief's adversarial requirement, end to end.

    A strongly bearish VOL_MOMENTUM against a bullish structure blocks the
    long; the mirror image blocks the short. There is no configuration of
    these five components under which the conflict produces a trade, because
    the only candidate side is the structure's and the opposition exceeds
    `max_opposing_points` against it.
    """
    long_side = decide(
        cfg,
        build_vector(structure_direction=1.0, vol_momentum_direction=-1.0),
    )
    short_side = decide(
        cfg,
        build_vector(
            structure_direction=-1.0,
            vol_momentum_direction=1.0,
            order_flow_direction=-1.0,
            options_flow_direction=-1.0,
        ),
    )

    for decision in (long_side, short_side):
        assert decision.signal.action is SignalAction.WAIT
        assert decision.signal.wait_reason is not None
        assert decision.signal.reasons
        assert decision.blocking is not None
        assert decision.blocking.stage is GateStage.ENTRY_FILTER
        assert decision.blocking.diagnostics["opposing_points"] > 12.0


def test_the_majority_direction_is_never_traded(cfg):
    """Four bullish components against a bearish structure: no LONG appears.

    This is the specific failure section 7 names. The structure says short
    at resistance; order flow, options flow and momentum all say up, so the
    aggregate's `net_direction()` is +1. A system that read the majority off
    the score would emit a LONG here. The engine emits a WAIT, because the
    side comes from `level_direction` and from nowhere else.
    """
    vector = build_vector(
        structure_direction=-1.0,
        vol_momentum_direction=1.0,
        order_flow_direction=1.0,
        options_flow_direction=1.0,
    )
    flow = aggregate(cfg, score_components(cfg, vector)).require()
    assert flow.net_direction() == 1  # the majority points up

    decision = decide(cfg, vector)
    assert decision.signal.action is SignalAction.WAIT
    assert decision.side is Side.SHORT  # the only side ever considered
    assert decision.signal.action is not SignalAction.LONG


def test_a_mirrored_pair_has_identical_magnitudes_and_opposite_directions(cfg):
    """Direction does not scale magnitude, in either direction."""
    bullish = aggregate(
        cfg, score_components(cfg, build_vector(vol_momentum_direction=1.0))
    ).require()
    bearish = aggregate(
        cfg, score_components(cfg, build_vector(vol_momentum_direction=-1.0))
    ).require()

    for component in Component:
        assert bullish.components[component].magnitude == pytest.approx(
            bearish.components[component].magnitude
        )
    assert bullish.components[Component.VOL_MOMENTUM].direction == 1
    assert bearish.components[Component.VOL_MOMENTUM].direction == -1
    assert bullish.score == pytest.approx(bearish.score)


# ---------------------------------------------------------------------------
# 2. the aggregate: bounds, refusal, and the 55-of-100 report
# ---------------------------------------------------------------------------


def test_the_score_is_the_sum_of_the_five_components(cfg):
    vector = build_vector()
    flow = aggregate(cfg, score_components(cfg, vector)).require()
    expected = sum(
        flow.components[c].magnitude * cfg.flow_score.weights[c] for c in Component
    )
    assert flow.score == pytest.approx(expected)
    assert 0.0 <= flow.score <= 100.0


def test_a_maximal_vector_reaches_exactly_the_configured_total(cfg):
    vector = build_vector(
        structure_magnitude=1.0,
        vol_momentum_magnitude=1.0,
        order_flow_magnitude=1.0,
        options_flow_magnitude=1.0,
        liquidity_score=1.0,
    )
    flow = aggregate(cfg, score_components(cfg, vector)).require()
    assert flow.score == pytest.approx(cfg.flow_score.total_points)


def test_strict_availability_refuses_rather_than_summing_what_is_left(cfg):
    """45 of 100 points unmeasured -> a refusal, not a 55-point 'Flow Score'."""
    assert cfg.flow_score.strict_component_availability is True
    vector = build_vector(order_flow_available=0.0)
    scored = score_components(cfg, vector)
    # Also strip the options component, the other feed-dependent one.
    scored = ScoredComponents(
        symbol=scored.symbol,
        ts=scored.ts,
        scores={
            **scored.scores,
            Component.OPTIONS_FLOW: scored.scores[Component.OPTIONS_FLOW].replace(
                enabled=False, magnitude=0.0, direction=0, quality=DataQuality.MISSING
            ),
        },
    )
    outcome = aggregate(cfg, scored)

    assert outcome.refused
    assert outcome.flow_score is None
    assert outcome.wait_reason is REFUSAL_WAIT_REASON is WaitReason.COMPONENT_DISABLED
    assert outcome.available_points == pytest.approx(55.0)
    assert outcome.max_points == pytest.approx(100.0)
    assert outcome.missing_points == pytest.approx(45.0)
    assert set(outcome.unavailable_components) == {
        Component.ORDER_FLOW,
        Component.OPTIONS_FLOW,
    }
    with pytest.raises(FlowScoreError):
        outcome.require()


def test_the_refusal_is_an_outcome_and_not_an_exception(cfg):
    """The engine has to record it as a WAIT, which a raise cannot be."""
    vector = build_vector(order_flow_available=0.0)
    outcome = aggregate(cfg, score_components(cfg, vector))
    assert outcome.refused
    assert isinstance(outcome, flow_score_module.FlowScoreOutcome)
    assert outcome.refusal is not None
    assert "strict_component_availability" in outcome.refusal.detail


def test_non_strict_reports_a_partial_sum_on_a_partial_scale(cfg):
    """Strict off, redistribution off: the sum of what was measured, labelled."""
    config = load_config(
        overrides=["flow_score.strict_component_availability=false"]
    )
    vector = build_vector(order_flow_available=0.0)
    outcome = aggregate(config, score_components(config, vector))

    assert not outcome.refused
    assert not outcome.redistributed
    assert outcome.available_points == pytest.approx(75.0)
    assert outcome.max_points == pytest.approx(100.0)
    assert outcome.require().score <= outcome.available_points
    assert any("was NOT redistributed" in r for r in outcome.reasons)


def test_redistribution_is_implemented_and_says_it_is_not_comparable(cfg):
    config = load_config(
        overrides=[
            "flow_score.strict_component_availability=false",
            "flow_score.redistribute_disabled_weight=true",
        ]
    )
    vector = build_vector(order_flow_available=0.0)
    outcome = aggregate(config, score_components(config, vector))

    assert outcome.redistributed
    assert outcome.available_points == pytest.approx(100.0)
    structure = outcome.require().components[Component.STRUCTURE]
    assert structure.weight > config.flow_score.weights[Component.STRUCTURE]
    assert structure.detail["weight_before_redistribution"] == pytest.approx(20.0)
    assert any("NOT COMPARABLE" in r for r in outcome.reasons)


def test_everything_unavailable_refuses_even_with_redistribution_on():
    """There is no scale to express a score on, so no flag produces one."""
    config = load_config(
        overrides=[
            "flow_score.strict_component_availability=false",
            "flow_score.redistribute_disabled_weight=true",
        ]
    )
    scores = {
        component: ComponentScore(
            component=component,
            magnitude=0.0,
            direction=0,
            weight=config.flow_score.weights[component],
            quality=DataQuality.MISSING,
            enabled=False,
        )
        for component in Component
    }
    outcome = aggregate(config, scores, symbol=SYMBOL, ts=TS)
    assert outcome.refused
    assert outcome.available_points == pytest.approx(0.0)


def test_a_missing_component_is_a_caller_bug_not_a_smaller_score(cfg):
    scores = {
        component: ComponentScore(
            component=component,
            magnitude=0.5,
            direction=0,
            weight=cfg.flow_score.weights[component],
        )
        for component in Component
        if component is not Component.LIQUIDITY
    }
    with pytest.raises(FlowScoreError, match="liquidity"):
        aggregate(cfg, scores, symbol=SYMBOL, ts=TS)


def test_a_scorer_may_not_choose_its_own_weight(cfg):
    scores = {
        component: ComponentScore(
            component=component,
            magnitude=0.5,
            direction=0,
            weight=cfg.flow_score.weights[component],
        )
        for component in Component
    }
    scores[Component.STRUCTURE] = scores[Component.STRUCTURE].replace(weight=40.0)
    with pytest.raises(FlowScoreError, match="weight"):
        aggregate(cfg, scores, symbol=SYMBOL, ts=TS)


def test_an_unavailable_component_keeps_its_configured_weight(cfg):
    """`enabled=False` is what removes it; a zeroed weight would hide the gap."""
    vector = build_vector(order_flow_available=0.0)
    scored = score_components(cfg, vector)
    order_flow = scored.scores[Component.ORDER_FLOW]
    assert order_flow.enabled is False
    assert order_flow.weight == pytest.approx(25.0)
    assert order_flow.points == 0.0


def test_summary_lines_report_both_numbers_on_a_refusal(cfg):
    vector = build_vector(order_flow_available=0.0)
    outcome = aggregate(cfg, score_components(cfg, vector))
    text = "\n".join(flow_score_module.summary_lines(outcome))
    assert "REFUSED" in text
    assert "75.0 of 100.0" in text


# ---------------------------------------------------------------------------
# 3. setup selection is a measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "achievable_rr, expected",
    [
        (1.0, SetupType.SCALP_1R),
        (1.5, SetupType.SCALP_1R),
        (1.79, SetupType.SCALP_1R),
        (1.8, SetupType.SETUP_2R),
        (2.4, SetupType.SETUP_2R),
        (2.69, SetupType.SETUP_2R),
        (2.7, SetupType.DIRECTIONAL_3R),
        (5.0, SetupType.DIRECTIONAL_3R),
    ],
)
def test_the_band_determines_the_setup(cfg, achievable_rr, expected):
    """14.6's bands, read off the measurement and nothing else."""
    vector = build_vector(achievable_rr=achievable_rr)
    selection = setups_api.select(
        cfg,
        vector,
        side=Side.LONG,
        regime=regime_state(Regime.TRENDING_UP),
        flow_score=95.0,
        components=score_components(cfg, vector).scores,
    )
    assert selection.setup is expected


def test_reward_risk_below_the_floor_declines_rather_than_retargets(cfg):
    vector = build_vector(achievable_rr=0.8)
    selection = setups_api.select(
        cfg,
        vector,
        side=Side.LONG,
        regime=regime_state(Regime.TRENDING_UP),
        flow_score=95.0,
        components=score_components(cfg, vector).scores,
    )
    assert selection.setup is None
    assert selection.wait_reason is WaitReason.RR_TOO_LOW


def test_a_bar_is_neither_promoted_nor_demoted_when_its_setup_is_disabled(cfg):
    """The structure measured 2.0R; with SETUP_2R off, the bar is declined.

    It is not handed up to DIRECTIONAL_3R (which would truncate nothing but
    would plan a target the structure does not support) and not handed down
    to SCALP_1R (which would be the "1R scalp that is really a 0.4R target"
    failure in another costume).
    """
    config = load_config(overrides=["setups.SETUP_2R.enabled=false"])
    vector = build_vector(achievable_rr=2.0)
    selection = setups_api.select(
        config,
        vector,
        side=Side.LONG,
        regime=regime_state(Regime.TRENDING_UP),
        flow_score=95.0,
        components=score_components(config, vector).scores,
    )
    assert selection.setup is None
    assert selection.wait_reason is WaitReason.NO_SETUP_MATCH
    assert "SETUP_2R" in selection.detail
    assert "did not name" in selection.detail


def test_disabling_the_lowest_setup_raises_the_reward_risk_floor(cfg):
    """With SCALP_1R off, 1.2R clears no floor and the reason says so.

    `min_reward_risk_floor` is the higher of the structure layer's own floor
    and the lowest enabled setup's, so disabling the 1R setup moves the floor
    to 1.8. A 1.2R bar is then declined as RR_TOO_LOW rather than as
    NO_SETUP_MATCH -- which is the more precise of the two reasons, and still
    a decline rather than a promotion.
    """
    config = load_config(overrides=["setups.SCALP_1R.enabled=false"])
    assert setups_api.min_reward_risk_floor(config) == pytest.approx(1.8)
    vector = build_vector(achievable_rr=1.2)
    selection = setups_api.select(
        config,
        vector,
        side=Side.LONG,
        regime=regime_state(Regime.TRENDING_UP),
        flow_score=95.0,
        components=score_components(config, vector).scores,
    )
    assert selection.setup is None
    assert selection.wait_reason is WaitReason.RR_TOO_LOW


def test_a_three_r_bar_is_not_truncated_to_two_r(cfg):
    vector = build_vector(achievable_rr=4.0)
    decision = decide(cfg, vector)
    assert decision.signal.action is SignalAction.LONG
    assert decision.signal.setup is SetupType.DIRECTIONAL_3R
    assert decision.plan is not None
    assert decision.plan.planned_r_multiple == pytest.approx(3.0, abs=0.01)
    assert decision.plan.target_price > ENTRY + 2.5 * ATR


def test_a_one_r_bar_plans_one_r(cfg):
    decision = decide(cfg, build_vector(achievable_rr=1.2), regime=Regime.TRENDING_UP)
    assert decision.signal.action is SignalAction.LONG
    assert decision.signal.setup is SetupType.SCALP_1R
    assert decision.plan is not None
    assert decision.plan.planned_r_multiple == pytest.approx(1.0, abs=0.01)


def test_the_two_reward_risk_numbers_are_both_recorded(cfg):
    """14.6's band floor sits below the setup's own reward_risk at 2.8R."""
    decision = decide(cfg, build_vector(achievable_rr=2.8))
    assert decision.signal.action is SignalAction.LONG
    assert decision.selection is not None
    assert decision.selection.setup is SetupType.DIRECTIONAL_3R
    assert decision.selection.achievable_rr == pytest.approx(2.8)
    assert decision.selection.planned_reward_risk == pytest.approx(3.0)
    assert decision.selection.plans_beyond_measured_target is True
    assert decision.plan is not None
    assert decision.plan.plans_beyond_measured_target is True
    assert any("optimistic about reachability" in r for r in decision.signal.reasons)


def test_a_disagreeing_setup_class_raises_rather_than_mislabelling(cfg):
    """A feature-layer regression must not become a mislabelled trade."""
    vector = build_vector(achievable_rr=1.2)
    broken = FeatureVector(
        symbol=vector.symbol,
        ts=vector.ts,
        values={**vector.values, "level_setup_class": 3.0},
        quality_by_key=vector.quality_by_key,
    )
    with pytest.raises(setups_api.SetupError, match="level_setup_class"):
        setups_api.select(
            cfg,
            broken,
            side=Side.LONG,
            regime=regime_state(Regime.TRENDING_UP),
            flow_score=95.0,
            components=score_components(cfg, broken).scores,
        )


def test_every_rejected_setup_records_its_first_failing_gate(cfg):
    """Rejection analysis can see the near-misses without acting on them."""
    vector = build_vector(achievable_rr=1.2)
    selection = setups_api.select(
        cfg,
        vector,
        side=Side.LONG,
        regime=regime_state(Regime.LOW_VOL),
        flow_score=95.0,
        components=score_components(cfg, vector).scores,
    )
    assert selection.setup is SetupType.SCALP_1R
    rejected = {r.setup: r for r in selection.rejections}
    assert SetupType.SETUP_2R in rejected
    assert rejected[SetupType.SETUP_2R].gate == "min_reward_risk"
    assert all(not r.selected_by_band for r in selection.rejections)


def test_the_union_helpers_are_the_weakest_honest_question(cfg):
    assert setups_api.min_flow_score_required(cfg) == pytest.approx(70.0)
    assert setups_api.min_reward_risk_floor(cfg) == pytest.approx(1.0)
    low, high = setups_api.vol_percentile_band(cfg)
    assert (low, high) == pytest.approx((0.10, 1.0))
    assert Regime.UNKNOWN not in setups_api.tradable_regimes(cfg)
    assert Regime.CHOP not in setups_api.tradable_regimes(cfg)


# ---------------------------------------------------------------------------
# 4. the gate sequence
# ---------------------------------------------------------------------------


def test_every_stage_declares_its_reasons_and_invents_none():
    assert set(WAIT_REASONS_BY_STAGE) == set(GateStage)
    for stage, reasons in WAIT_REASONS_BY_STAGE.items():
        assert reasons, f"{stage.value} declares no wait reasons"
        for reason in reasons:
            assert isinstance(reason, WaitReason)


def test_every_wait_reason_in_the_enum_is_reachable_from_some_stage():
    """A reason nothing can emit is dead, and one that is emitted but not
    declared would mislabel section 12's rejection histogram."""
    declared = {r for reasons in WAIT_REASONS_BY_STAGE.values() for r in reasons}
    assert declared == set(WaitReason)


def test_a_failing_gate_must_state_a_reason():
    with pytest.raises(ValueError, match="must state a wait_reason"):
        GateResult(stage=GateStage.LIQUIDITY, passed=False)
    with pytest.raises(ValueError, match="must not carry a wait_reason"):
        GateResult(
            stage=GateStage.LIQUIDITY, passed=True, wait_reason=WaitReason.LIQUIDITY
        )


def test_a_gate_cannot_emit_an_undeclared_reason():
    with pytest.raises(ValueError, match="not among its declared reasons"):
        GateResult(
            stage=GateStage.LIQUIDITY,
            passed=False,
            wait_reason=WaitReason.RISK_LIMIT,
        )


def test_the_engine_runs_the_stages_in_gate_stage_order(cfg):
    decision = decide(cfg, build_vector(achievable_rr=1.2))
    assert decision.signal.action is SignalAction.LONG
    assert decision.stages == tuple(GateStage)


def test_the_first_failing_gate_wins(cfg):
    """A bar blocked by several reports the most fundamental one.

    No structure AND a flow score that would also fail: the structure gate is
    earlier, so `NO_STRUCTURE` is what rejection analysis counts.
    """
    vector = build_vector(
        gate_passed=0.0,
        gate_reason=3.0,  # GATE_SIGNIFICANCE
        structure_magnitude=0.05,
        vol_momentum_magnitude=0.05,
        order_flow_magnitude=0.05,
        options_flow_magnitude=0.05,
        liquidity_score=0.05,
    )
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is WaitReason.NO_STRUCTURE
    assert decision.stages[-1] is GateStage.STRUCTURE


def test_a_structural_reward_risk_failure_reports_rr_too_low(cfg):
    """levels.py's gate reasons 6 and 7 map to RR_TOO_LOW, 1-5 to NO_STRUCTURE."""
    vector = build_vector(gate_passed=0.0, gate_reason=7.0)
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is WaitReason.RR_TOO_LOW
    assert decision.stages[-1] is GateStage.STRUCTURE


def test_order_flow_opposition_blocks_before_the_flow_score(cfg):
    vector = build_vector(order_flow_direction=-1.0)
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is WaitReason.ORDERFLOW_CONFLICT
    assert decision.stages[-1] is GateStage.ORDER_FLOW


def test_options_opposition_above_the_limit_blocks(cfg):
    vector = build_vector(options_flow_direction=-1.0, options_flow_magnitude=0.95)
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is WaitReason.OPTIONS_CONTRADICTION
    assert decision.stages[-1] is GateStage.OPTIONS_FLOW


def test_weak_options_opposition_does_not_block(cfg):
    """19 points of opposition blocks; 2 does not. The limit is in points."""
    vector = build_vector(options_flow_direction=-1.0, options_flow_magnitude=0.1)
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is not WaitReason.OPTIONS_CONTRADICTION


def test_an_unmeasured_component_never_counts_as_opposition(cfg):
    """"Never invent missing order-flow data": absent is not disagreement."""
    config = load_config(
        overrides=["flow_score.strict_component_availability=false"]
    )
    vector = build_vector(order_flow_available=0.0)
    decision = decide(config, vector)
    assert decision.signal.wait_reason is not WaitReason.ORDERFLOW_CONFLICT
    order_flow_stage = next(
        r for r in decision.gate_results if r.stage is GateStage.ORDER_FLOW
    )
    assert order_flow_stage.passed
    assert any("not measured" in r for r in order_flow_stage.reasons)


def test_the_volatility_band_is_the_union_before_selection(cfg):
    vector = build_vector(atr_percentile=0.02)
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is WaitReason.VOLATILITY_BAND
    assert decision.stages[-1] is GateStage.MOMENTUM


def test_a_blocked_regime_is_reported_before_anything_is_scored(cfg):
    decision = decide(cfg, build_vector(), regime=Regime.CHOP)
    assert decision.signal.wait_reason is WaitReason.REGIME_BLOCKED
    assert decision.stages[-1] is GateStage.REGIME
    assert decision.features is None
    assert not decision.components


def test_an_unknown_regime_is_never_tradable(cfg):
    decision = decide(cfg, build_vector(), regime=Regime.UNKNOWN)
    assert decision.signal.wait_reason is WaitReason.REGIME_BLOCKED
    assert decision.signal.regime is Regime.UNKNOWN


def test_no_liquidity_measurement_is_a_wait_not_a_guess(cfg):
    vector = build_vector(grades={"liquidity_score": DataQuality.MISSING})
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is WaitReason.LIQUIDITY
    assert decision.stages[-1] is GateStage.LIQUIDITY


def test_a_bar_with_no_volume_cannot_be_traded(cfg):
    vector = build_vector()
    zero_volume = FeatureVector(
        symbol=vector.symbol,
        ts=vector.ts,
        values={**vector.values, "relative_volume": 0.0},
        quality_by_key=vector.quality_by_key,
    )
    decision = decide(cfg, zero_volume)
    assert decision.signal.wait_reason is WaitReason.LIQUIDITY
    assert "no volume" in decision.signal.wait_detail


def test_an_unusable_quote_is_a_liquidity_wait(cfg):
    """A measured spread narrower than the instrument's own minimum."""
    vector = build_vector()
    crossed = FeatureVector(
        symbol=vector.symbol,
        ts=vector.ts,
        values={**vector.values, "spread_ticks": 0.25},
        quality_by_key=vector.quality_by_key,
    )
    decision = decide(cfg, crossed)
    assert decision.signal.wait_reason is WaitReason.LIQUIDITY
    assert "min_spread_ticks" in decision.signal.wait_detail


def test_the_liquidity_score_floor_ships_switched_off():
    assert gate_module.LIQUIDITY_GATE_MIN_SCORE == 0.0


def test_the_liquidity_floor_can_be_raised_without_a_code_edit(cfg):
    engine = SignalEngine(cfg, SYMBOL, min_liquidity_score=0.95)
    inputs = PrecomputedInputs(
        features=build_vector(liquidity_score=0.5),
        regime=regime_state(),
        reference_price=ENTRY,
    )
    decision = engine.decide_from(inputs)
    assert decision.signal.wait_reason is WaitReason.LIQUIDITY


def test_a_refused_flow_score_is_reported_as_component_disabled(cfg):
    """Strict availability, with the data-quality gate out of the way."""
    vector = build_vector(order_flow_available=0.0)
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert decision.stages[-1] is GateStage.FLOW_SCORE
    assert decision.signal.flow_score is None
    assert decision.flow is not None
    assert decision.flow.available_points == pytest.approx(75.0)
    assert any("not redistributed" in r for r in decision.signal.reasons)


def test_a_score_below_every_setup_threshold_is_reported_once(cfg):
    vector = build_vector(
        structure_magnitude=0.2,
        vol_momentum_magnitude=0.2,
        order_flow_magnitude=0.2,
        options_flow_magnitude=0.2,
        liquidity_score=0.2,
    )
    decision = decide(cfg, vector)
    assert decision.signal.wait_reason is WaitReason.SCORE_BELOW_THRESHOLD
    assert decision.stages[-1] is GateStage.FLOW_SCORE
    assert decision.signal.flow_score is not None


def test_a_setup_specific_score_miss_is_reported_at_the_setup_stage(cfg):
    """Above the union floor (70) and below DIRECTIONAL_3R's own (80)."""
    vector = build_vector(
        achievable_rr=3.5,
        structure_magnitude=0.76,
        vol_momentum_magnitude=0.76,
        order_flow_magnitude=0.76,
        options_flow_magnitude=0.76,
        liquidity_score=0.76,
    )
    decision = decide(cfg, vector)
    assert decision.signal.flow_score is not None
    assert 70.0 <= decision.signal.flow_score < 80.0
    assert decision.signal.wait_reason is WaitReason.SCORE_BELOW_THRESHOLD
    assert decision.stages[-1] is GateStage.SETUP


# ---------------------------------------------------------------------------
# 5. the risk stage
# ---------------------------------------------------------------------------


def test_a_grid_realizable_r_below_the_minimum_is_surfaced_as_a_wait(cfg):
    """`risk/sizing.py` refuses rather than relabelling; the engine reports it.

    A two-tick stop on a 0.25-tick grid cannot carry a 3R target that
    survives rounding toward entry, so sizing refuses with
    `reward_risk_below_minimum` and the engine maps it to `RR_TOO_LOW`.
    """
    assert (
        SIZING_WAIT_REASONS["reward_risk_below_minimum"] is WaitReason.RR_TOO_LOW
    )
    config = load_config(overrides=["setups.SCALP_1R.min_reward_risk=1.0"])
    decision = decide(
        config, build_vector(achievable_rr=1.2, stop_distance=0.3)
    )
    assert decision.signal.action is SignalAction.WAIT
    assert decision.stages[-1] is GateStage.RISK
    assert decision.sizing is not None
    assert decision.sizing.accepted is False


def test_a_stop_wider_than_the_setup_allows_is_a_risk_wait(cfg):
    """`max_stop_atr_multiple * ATR` is the cap, resolved by the engine."""
    decision = decide(cfg, build_vector(achievable_rr=1.2, stop_distance=10 * ATR))
    assert decision.signal.action is SignalAction.WAIT
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert decision.stages[-1] is GateStage.RISK
    assert decision.sizing is not None
    assert decision.sizing.reason == "stop_too_wide"


def test_a_risk_limit_breach_is_a_wait_after_the_signal_was_computed(cfg):
    """Section 3 runs limits last so the signal is still recorded as occurring."""
    limits = RiskLimitManager(cfg.risk)
    for _ in range(cfg.risk.max_consecutive_losses):
        limits.on_trade_closed(
            ts=TS - timedelta(days=1),
            session_date=date(2024, 2, 1),
            symbol=SYMBOL,
            pnl=-100.0,
        )
    decision = decide(cfg, build_vector(achievable_rr=1.2), limits=limits)
    assert decision.signal.action is SignalAction.WAIT
    assert decision.signal.wait_reason is WaitReason.RISK_LIMIT
    assert decision.stages[-1] is GateStage.RISK
    # The score and the setup were computed and are on the record.
    assert decision.signal.flow_score is not None
    assert decision.signal.setup is SetupType.SCALP_1R
    assert decision.limits is not None
    assert decision.limits.breach == "max_consecutive_losses"


def test_the_plan_converts_to_a_trade_intent(cfg):
    decision = decide(cfg, build_vector(achievable_rr=1.2))
    assert decision.plan is not None
    intent = decision.plan.to_trade_intent()
    assert intent.side is Side.LONG
    assert intent.planned_r_multiple == pytest.approx(1.0, abs=0.01)
    assert intent.risk_dollars == pytest.approx(
        intent.size * intent.stop_distance * intent.point_value
    )


def test_the_short_side_plans_a_mirrored_geometry(cfg):
    decision = decide(
        cfg,
        build_vector(
            structure_direction=-1.0,
            achievable_rr=1.2,
            vol_momentum_direction=-1.0,
            order_flow_direction=-1.0,
            options_flow_direction=-1.0,
        ),
    )
    assert decision.signal.action is SignalAction.SHORT
    plan = decision.plan
    assert plan is not None
    assert plan.target_price < plan.entry_price < plan.stop_price


# ---------------------------------------------------------------------------
# 6. every outcome carries a reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, regime",
    [
        ({}, Regime.CHOP),
        ({}, Regime.UNKNOWN),
        ({"gate_passed": 0.0, "gate_reason": 1.0}, Regime.TRENDING_UP),
        ({"order_flow_direction": -1.0}, Regime.TRENDING_UP),
        ({"options_flow_direction": -1.0}, Regime.TRENDING_UP),
        ({"atr_percentile": 0.01}, Regime.TRENDING_UP),
        ({"order_flow_available": 0.0}, Regime.TRENDING_UP),
        ({"vol_momentum_direction": -1.0}, Regime.TRENDING_UP),
        ({"achievable_rr": 0.5}, Regime.TRENDING_UP),
        ({"grades": {"liquidity_score": DataQuality.MISSING}}, Regime.TRENDING_UP),
    ],
)
def test_every_wait_carries_a_reason_and_a_narrative(cfg, kwargs, regime):
    decision = decide(cfg, build_vector(**kwargs), regime=regime)
    signal = decision.signal
    assert signal.action is SignalAction.WAIT
    assert signal.wait_reason is not None
    assert signal.wait_detail
    assert signal.reasons
    assert decision.blocking is not None
    assert decision.blocking.wait_reason is signal.wait_reason


def test_confidence_is_the_score_over_the_points_that_were_measured(cfg):
    decision = decide(cfg, build_vector(achievable_rr=1.2))
    signal = decision.signal
    assert decision.flow is not None
    assert signal.confidence == pytest.approx(
        signal.flow_score / decision.flow.available_points
    )
    assert 0.0 <= signal.confidence <= 1.0


def test_component_points_are_recorded_for_all_five(cfg):
    decision = decide(cfg, build_vector(achievable_rr=1.2))
    assert set(decision.signal.component_points) == set(Component)
    assert sum(decision.signal.component_points.values()) == pytest.approx(
        decision.signal.flow_score
    )


# ---------------------------------------------------------------------------
# 7. statelessness and parity
# ---------------------------------------------------------------------------


def test_the_engine_refuses_a_view_for_another_symbol(cfg, dataset):
    engine = SignalEngine(cfg, "ES")
    view = _view_at(dataset, len(dataset.bars) - 1)
    with pytest.raises(EngineError, match="built for 'ES'"):
        engine.decide(view)


def test_availability_for_the_wrong_symbol_is_refused(cfg, dataset):
    data = _symbol_data(dataset, with_feeds=True)
    report = QualityGrader(cfg.data, cfg.flow_score).availability(data)
    wrong = report.replace(symbol="ES")
    with pytest.raises(EngineError, match="availability report is for"):
        SignalEngine(cfg, SYMBOL, availability=wrong)


def test_two_engines_from_one_config_agree(cfg):
    vector = build_vector(achievable_rr=1.2)
    first = engine_for(cfg).decide_from(
        PrecomputedInputs(
            features=vector, regime=regime_state(), reference_price=ENTRY
        )
    )
    second = engine_for(cfg).decide_from(
        PrecomputedInputs(
            features=vector, regime=regime_state(), reference_price=ENTRY
        )
    )
    assert first.signal == second.signal


def test_the_same_engine_gives_the_same_answer_twice(cfg):
    """No accumulator: a second call on the same inputs cannot drift."""
    engine = engine_for(cfg)
    vector = build_vector(achievable_rr=1.2)

    def once():
        return engine.decide_from(
            PrecomputedInputs(
                features=vector, regime=regime_state(), reference_price=ENTRY
            )
        ).signal

    assert once() == once() == once()


# ---------------------------------------------------------------------------
# 8. against the real feature layer
# ---------------------------------------------------------------------------


def _synthetic(config, *, intraday_options: bool = True, seed: int = 11):
    spec = config.spec(SYMBOL)
    return SyntheticMarketGenerator(
        SyntheticConfig(options_intraday=intraday_options), seed=seed
    ).generate(
        SYMBOL,
        spec,
        date(2024, 1, 2),
        date(2024, 6, 1),
        INTERVAL,
        calendar=TradingCalendar(),
    )


def _symbol_data(dataset, *, with_feeds: bool) -> SymbolData:
    return SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: dataset.bars},
        ticks={INTERVAL: dataset.ticks} if with_feeds else None,
        quotes=dataset.quotes if with_feeds else None,
        options=dataset.options if with_feeds else None,
    )


def _view_at(dataset, index: int, *, with_feeds: bool = True) -> MarketView:
    data = _symbol_data(dataset, with_feeds=with_feeds)
    ns = int(dataset.bars.ts_ns[index])
    return MarketView(data, now=from_ns(ns), now_ns=ns)


@pytest.fixture(scope="module")
def dataset():
    return _synthetic(load_config())


def test_the_computer_set_excludes_the_regime_detector(cfg):
    """Four of the detector's diagnostic keys collide with the bundle's."""
    names = {c.name for c in feature_computers(cfg, cfg.spec(SYMBOL))}
    assert len(names) == 7
    engine = SignalEngine(cfg, SYMBOL)
    assert engine.detector not in engine.computers
    assert engine.detector in engine.warmup_computers


def test_bars_only_data_waits_on_data_quality(cfg, dataset):
    """45 of 100 points have no feed; the required feeds are blocking."""
    engine = SignalEngine(cfg, SYMBOL, calendar=TradingCalendar())
    view = _view_at(dataset, IN_SESSION_BAR, with_feeds=False)
    decision = engine.decide(view)

    assert decision.signal.action is SignalAction.WAIT
    assert decision.signal.wait_reason is WaitReason.DATA_QUALITY
    assert decision.stages[-1] is GateStage.MARKET_DATA
    assert decision.quality is not None
    assert Feed.TICK_AGGREGATE in decision.quality.blocking_feeds
    assert Feed.OPTIONS_SNAPSHOT in decision.quality.blocking_feeds


def test_bars_only_availability_reports_fifty_five_of_one_hundred(cfg, dataset):
    """The number the brief names, straight from `data/quality.py`."""
    data = _symbol_data(dataset, with_feeds=False)
    report = QualityGrader(cfg.data, cfg.flow_score).availability(data)
    assert report.available_points == pytest.approx(55.0)
    assert report.total_points == pytest.approx(100.0)
    assert set(report.incomputable_components) == {
        Component.ORDER_FLOW,
        Component.OPTIONS_FLOW,
    }
    assert report.scoring_is_valid is False


def test_bars_only_scoring_refuses_when_the_quality_gate_is_relaxed(cfg, dataset):
    """With the feed floors made optional, the aggregate is what refuses.

    The data-quality gate fires first on the shipped config, so this makes
    the required feeds optional to reach the Flow Score stage, and shows the
    refusal there: 55.0 of 100.0, `COMPONENT_DISABLED`, nothing
    redistributed.
    """
    config = load_config(
        overrides=[
            "flow_score.feed_requirements.order_flow="
            "[{feed: tick_aggregate, minimum_quality: MISSING, required: false}]",
            "flow_score.feed_requirements.options_flow="
            "[{feed: options_snapshot, minimum_quality: MISSING, required: false}]",
        ]
    )
    data = _symbol_data(dataset, with_feeds=False)
    report = QualityGrader(config.data, config.flow_score).availability(data)
    engine = SignalEngine(
        config, SYMBOL, calendar=TradingCalendar(), availability=report
    )
    view = _view_at(dataset, IN_SESSION_BAR, with_feeds=False)
    decision = engine.decide(view)

    assert decision.signal.action is SignalAction.WAIT
    assert decision.signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert decision.stages[-1] is GateStage.FLOW_SCORE
    assert decision.flow is not None
    assert decision.flow.refused
    assert decision.flow.available_points == pytest.approx(55.0)
    assert decision.flow.max_points == pytest.approx(100.0)
    assert decision.signal.flow_score is None
    assert any("not redistributed" in r for r in decision.signal.reasons)


def test_the_full_feed_dataset_reports_all_one_hundred_points(cfg, dataset):
    data = _symbol_data(dataset, with_feeds=True)
    report = QualityGrader(cfg.data, cfg.flow_score).availability(data)
    assert report.available_points == pytest.approx(100.0)
    assert report.incomputable_components == ()


def test_the_engine_runs_against_the_real_feature_layer(cfg, dataset):
    engine = SignalEngine(cfg, SYMBOL, calendar=TradingCalendar())
    view = _view_at(dataset, len(dataset.bars) - 1)
    decision = engine.decide(view, equity=100_000.0)

    assert decision.signal.symbol == SYMBOL
    assert decision.signal.reasons
    assert decision.signal.wait_reason is not None or decision.plan is not None
    assert decision.stages[0] is GateStage.SESSION


def test_warmup_is_judged_per_computer_not_on_the_merged_vector(cfg, dataset):
    """Bars-only: the feed-dependent computers must not block warmup forever."""
    engine = SignalEngine(cfg, SYMBOL)
    view = _view_at(dataset, len(dataset.bars) - 1, with_feeds=False)
    report = assess_warmup(engine.warmup_computers, view)

    assert report.blocking == ()
    absent = {c.name for c in report.computers if not c.feeds_present}
    assert absent == {"orderflow", "options_flow"}
    # And the merged vector really does say otherwise, which is the trap.
    features = engine.bundle.compute(view)
    assert features.warmup_complete is False
    assert features.quality is DataQuality.MISSING


def test_an_early_bar_waits_on_warmup(cfg, dataset):
    engine = SignalEngine(cfg, SYMBOL)
    view = _view_at(dataset, 50)
    decision = engine.decide(view)
    assert decision.signal.wait_reason is WaitReason.WARMUP
    assert decision.stages[-1] is GateStage.WARMUP
    assert decision.features is None


def test_the_real_path_produces_a_trade_and_a_consistent_plan(
    relaxed_config, dataset
):
    """One LONG, end to end, through every stage in order.

    The bar index is pinned because walking six thousand bars in a unit test
    is not worth the seconds; the walk that found it is reproducible from the
    same seed and the same dataset fixture.
    """
    data = _symbol_data(dataset, with_feeds=True)
    report = QualityGrader(
        relaxed_config.data, relaxed_config.flow_score
    ).availability(data)
    engine = SignalEngine(
        relaxed_config, SYMBOL, calendar=TradingCalendar(), availability=report
    )
    view = _view_at(dataset, REAL_LONG_BAR)
    decision = engine.decide(view, equity=100_000.0)

    assert decision.signal.action is SignalAction.LONG
    assert decision.stages == tuple(GateStage)
    plan = decision.plan
    assert plan is not None
    assert plan.stop_price < plan.entry_price < plan.target_price
    assert plan.planned_r_multiple == pytest.approx(1.0, abs=0.05)
    assert plan.achievable_rr >= 1.0
    assert plan.size >= 1.0
    intent = plan.to_trade_intent()
    assert intent.flow_score == decision.signal.flow_score
    assert decision.signal.data_quality.rank >= DataQuality.DEGRADED.rank


def test_a_cold_call_matches_a_sequential_walk(relaxed_config, dataset):
    """The property section 12's backtest/live parity test will rest on.

    Evaluating one bar with no history of prior calls must equal evaluating
    it after walking the bars before it. Anything the engine accumulated
    would show up here as a difference.
    """
    data = _symbol_data(dataset, with_feeds=True)
    report = QualityGrader(
        relaxed_config.data, relaxed_config.flow_score
    ).availability(data)

    def fresh_engine():
        return SignalEngine(
            relaxed_config,
            SYMBOL,
            calendar=TradingCalendar(),
            availability=report,
        )

    cold = fresh_engine().evaluate(
        _view_at(dataset, REAL_LONG_BAR), equity=100_000.0
    )

    walker = fresh_engine()
    walked = None
    for index in range(REAL_LONG_BAR - 4, REAL_LONG_BAR + 1):
        walked = walker.evaluate(_view_at(dataset, index), equity=100_000.0)

    assert walked == cold
    assert cold.action is SignalAction.LONG


def test_all_five_scorers_are_reused_rather_than_rebuilt(cfg):
    engine = SignalEngine(cfg, SYMBOL)
    assert {s.component for s in engine.scorers} == set(Component)
    assert len(engine.scorers) == 5
    assert tuple(s.component for s in all_scorers(cfg)) == tuple(Component)


# ---------------------------------------------------------------------------
# 9. the lookahead firewall, at the engine level
# ---------------------------------------------------------------------------
#
# The feature layer already has a per-computer audit
# (`validation/lookahead.py`, run under `factory=`). These two tests ask the
# same question of the whole pipeline, because the engine composes seven
# computers, a detector, five scorers, an aggregate, eleven gates, a setup
# selector and the risk layer, and a leak could live in the composition
# rather than in any one part.


def _signal_at(config, data, index_ts_ns: int, availability):
    engine = SignalEngine(
        config, SYMBOL, calendar=TradingCalendar(), availability=availability
    )
    view = MarketView(data, now=from_ns(index_ts_ns), now_ns=index_ts_ns)
    return engine.evaluate(view, equity=100_000.0)


def test_the_signal_does_not_change_when_the_future_is_mutated(
    relaxed_config, dataset
):
    """The future-shuffle test of section 4, applied to the whole engine.

    Every observation after the evaluation instant is scaled by 7. A signal
    that moved would be reading bars, quotes, ticks or chains that had not
    happened yet.
    """
    data = _symbol_data(dataset, with_feeds=True)
    availability = QualityGrader(
        relaxed_config.data, relaxed_config.flow_score
    ).availability(data)
    cutoff = int(dataset.bars.ts_ns[REAL_LONG_BAR])

    baseline = _signal_at(relaxed_config, data, cutoff, availability)
    mutated = _signal_at(
        relaxed_config, mutate_after(data, cutoff), cutoff, availability
    )

    assert baseline.action is SignalAction.LONG
    assert mutated == baseline


def test_the_signal_does_not_change_when_the_future_is_removed(
    relaxed_config, dataset
):
    """And the dataset ending at the evaluation instant gives the same answer."""
    data = _symbol_data(dataset, with_feeds=True)
    availability = QualityGrader(
        relaxed_config.data, relaxed_config.flow_score
    ).availability(data)
    cutoff = int(dataset.bars.ts_ns[REAL_LONG_BAR])

    baseline = _signal_at(relaxed_config, data, cutoff, availability)
    truncated = _signal_at(
        relaxed_config, truncate(data, cutoff), cutoff, availability
    )

    assert truncated == baseline


# ---------------------------------------------------------------------------
# 10. the two deviations this phase had to make, pinned so they stay visible
# ---------------------------------------------------------------------------


def test_an_ablated_component_makes_the_engine_refuse_under_strict_mode(cfg):
    """A Phase 8 ablation on the shipped config refuses on every bar.

    `enabled_components` switches a component off for an experiment, and
    `strict_component_availability` then refuses to score the rest -- which
    is exactly what the flag says it does, and means an ablation sweep has
    to turn strict mode off deliberately rather than discovering later that
    its scores were on a 75-point scale. Pinned here so the interaction is a
    documented consequence rather than a surprise.
    """
    config = load_config(overrides=["flow_score.enabled_components.order_flow=false"])
    assert config.flow_score.strict_component_availability is True
    decision = decide(config, build_vector(achievable_rr=1.2))

    assert decision.signal.action is SignalAction.WAIT
    assert decision.signal.wait_reason is WaitReason.COMPONENT_DISABLED
    assert decision.flow is not None
    assert decision.flow.refused
    assert decision.flow.available_points == pytest.approx(75.0)


def test_vol_momentum_opposition_reports_the_nearest_available_reason(cfg):
    """The one component the existing `WaitReason` enum cannot name exactly.

    `OPPOSITION_WAIT_REASON` maps VOL_MOMENTUM opposition to
    `VOLATILITY_BAND` -- the reason section 3 assigns to the volatility and
    momentum stage -- because no reason in the enum means "momentum
    disagreed" and no new reason string may be invented. The precise fact is
    in `wait_detail`, so rejection analysis loses only the coarseness of the
    bucket. Asserted rather than left in a comment, so the compromise is
    visible to the next reader.
    """
    assert (
        gate_module.OPPOSITION_WAIT_REASON[Component.VOL_MOMENTUM]
        is WaitReason.VOLATILITY_BAND
    )
    decision = decide(
        cfg, build_vector(achievable_rr=1.2, vol_momentum_direction=-1.0)
    )
    assert decision.blocking is not None
    assert decision.blocking.stage is GateStage.ENTRY_FILTER
    assert decision.signal.wait_reason is WaitReason.VOLATILITY_BAND
    assert "vol_momentum" in decision.signal.wait_detail
    assert "largest opposer is vol_momentum" in decision.signal.wait_detail
