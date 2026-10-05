"""Data contracts, with emphasis on the R definition.

The brief's headline targets (88-92% at 1R) are only meaningful if "1R" is
pinned down. These tests pin it down.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from flow_model.core.contracts import (
    Bar,
    ComponentScore,
    CostBreakdown,
    EquityPoint,
    FeatureVector,
    FlowScore,
    OptionsSnapshot,
    QuoteSnapshot,
    RegimeState,
    Signal,
    TickAggregate,
    TradeIntent,
)
from flow_model.core.enums import (
    Component,
    DataQuality,
    ExitReason,
    Regime,
    SetupType,
    Side,
    SignalAction,
    WaitReason,
)

TS = datetime(2020, 6, 1, 10, 0, tzinfo=timezone.utc)


# --- Bar -------------------------------------------------------------------


def test_bar_open_ts_derived_from_close():
    bar = Bar(symbol="NQ", close_ts=TS, open=1, high=2, low=0.5, close=1.5,
              volume=10, interval_seconds=300)
    assert bar.open_ts == TS - timedelta(seconds=300)


@pytest.mark.parametrize(
    "o,h,l,c,v,valid",
    [
        (1.0, 2.0, 0.5, 1.5, 10, True),
        (1.0, 1.0, 1.0, 1.0, 0, True),      # flat bar, zero volume
        (5.0, 2.0, 0.5, 1.5, 10, False),    # open above high
        (1.0, 2.0, 0.5, 9.0, 10, False),    # close above high
        (1.0, 0.5, 2.0, 1.5, 10, False),    # high below low
        (1.0, 2.0, 0.5, 1.5, -1, False),    # negative volume
        (1.0, float("nan"), 0.5, 1.5, 10, False),
        (1.0, float("inf"), 0.5, 1.5, 10, False),
    ],
)
def test_bar_validity_checks(o, h, l, c, v, valid):
    bar = Bar(symbol="NQ", close_ts=TS, open=o, high=h, low=l, close=c,
              volume=v, interval_seconds=60)
    assert bar.is_valid is valid


def test_bar_range_and_typical_price():
    bar = Bar(symbol="NQ", close_ts=TS, open=10, high=12, low=8, close=11,
              volume=1, interval_seconds=60)
    assert bar.range == pytest.approx(4.0)
    assert bar.typical_price == pytest.approx((12 + 8 + 11) / 3)


# --- quotes / ticks / options ---------------------------------------------


def test_quote_derived_values():
    q = QuoteSnapshot(symbol="NQ", ts=TS, bid=100.0, ask=100.5, bid_size=30, ask_size=10)
    assert q.mid == pytest.approx(100.25)
    assert q.spread == pytest.approx(0.5)
    assert q.depth_imbalance == pytest.approx(0.5)
    assert not q.is_crossed


def test_quote_handles_empty_book_and_crossed_market():
    empty = QuoteSnapshot(symbol="NQ", ts=TS, bid=100, ask=101, bid_size=0, ask_size=0)
    assert empty.depth_imbalance == 0.0          # no division by zero
    crossed = QuoteSnapshot(symbol="NQ", ts=TS, bid=101, ask=100, bid_size=1, ask_size=1)
    assert crossed.is_crossed


def test_tick_aggregate_does_not_invent_a_side():
    """Unclassified volume stays unclassified -- that is the whole point."""
    ta = TickAggregate(symbol="NQ", close_ts=TS, buy_volume=600, sell_volume=400,
                       unclassified_volume=1000)
    assert ta.delta == 200
    assert ta.classified_volume == 1000
    assert ta.total_volume == 2000
    assert ta.delta_ratio == pytest.approx(0.2)       # normalized by classified only
    assert ta.classification_coverage == pytest.approx(0.5)


def test_tick_aggregate_zero_volume_is_neutral():
    ta = TickAggregate(symbol="NQ", close_ts=TS, buy_volume=0, sell_volume=0)
    assert ta.delta_ratio == 0.0
    assert ta.classification_coverage == 0.0


def test_options_snapshot_imbalances():
    snap = OptionsSnapshot(
        symbol="SPX", ts=TS, call_volume=1000, put_volume=500,
        call_premium=3_000_000, put_premium=1_000_000,
        call_oi=50_000, put_oi=60_000, iv_25d_call=0.18, iv_25d_put=0.24,
    )
    assert snap.net_premium == pytest.approx(2_000_000)
    assert snap.premium_imbalance == pytest.approx(0.5)
    assert snap.put_call_volume_ratio == pytest.approx(0.5)
    assert snap.skew_25d == pytest.approx(0.06)


def test_options_snapshot_missing_iv_yields_none_not_zero():
    snap = OptionsSnapshot(symbol="SPX", ts=TS, call_volume=1, put_volume=1,
                           call_premium=1, put_premium=1, call_oi=1, put_oi=1)
    assert snap.skew_25d is None          # absent, not 0.0
    assert snap.premium_imbalance == 0.0


def test_options_snapshot_zero_call_volume_ratio_is_none():
    snap = OptionsSnapshot(symbol="SPX", ts=TS, call_volume=0, put_volume=10,
                           call_premium=0, put_premium=1, call_oi=0, put_oi=1)
    assert snap.put_call_volume_ratio is None


def test_eod_options_data_is_flagged_not_disguised():
    snap = OptionsSnapshot(symbol="SPX", ts=TS, call_volume=1, put_volume=1,
                           call_premium=1, put_premium=1, call_oi=1, put_oi=1,
                           is_intraday=False, source="eod_chain")
    assert snap.is_intraday is False
    assert snap.source == "eod_chain"


# --- features --------------------------------------------------------------


def test_feature_vector_quality_is_worst_of_parts():
    fv = FeatureVector(
        symbol="NQ", ts=TS, values={"atr": 20.0, "cvd": -3.0},
        quality_by_key={"atr": DataQuality.GOOD, "cvd": DataQuality.STALE},
    )
    assert fv.quality is DataQuality.STALE


def test_empty_feature_vector_is_missing():
    assert FeatureVector(symbol="NQ", ts=TS).quality is DataQuality.MISSING


def test_feature_vector_require_rejects_absent_and_nonfinite():
    fv = FeatureVector(symbol="NQ", ts=TS, values={"ok": 1.0, "bad": float("nan")},
                       quality_by_key={"ok": DataQuality.GOOD, "bad": DataQuality.GOOD})
    assert fv.require("ok") == 1.0
    with pytest.raises(KeyError):
        fv.require("absent")
    with pytest.raises(ValueError, match="not finite"):
        fv.require("bad")


def test_feature_vector_quality_of_unknown_key_is_missing():
    fv = FeatureVector(symbol="NQ", ts=TS, values={"a": 1.0},
                       quality_by_key={"a": DataQuality.GOOD})
    assert fv.quality_of("nope") is DataQuality.MISSING


def test_feature_vector_merge_rejects_key_collision():
    a = FeatureVector(symbol="NQ", ts=TS, values={"x": 1.0},
                      quality_by_key={"x": DataQuality.GOOD})
    b = FeatureVector(symbol="NQ", ts=TS, values={"y": 2.0},
                      quality_by_key={"y": DataQuality.DEGRADED})
    merged = a.merge(b)
    assert merged.values == {"x": 1.0, "y": 2.0}
    assert merged.quality is DataQuality.DEGRADED
    with pytest.raises(ValueError, match="duplicate feature keys"):
        a.merge(a)


def test_feature_vector_merge_rejects_timestamp_mismatch():
    a = FeatureVector(symbol="NQ", ts=TS, values={"x": 1.0})
    b = FeatureVector(symbol="NQ", ts=TS + timedelta(minutes=5), values={"y": 1.0})
    with pytest.raises(ValueError, match="different symbol/ts"):
        a.merge(b)


def test_regime_unknown_is_not_tradable():
    assert not RegimeState(symbol="NQ", ts=TS, regime=Regime.UNKNOWN).is_tradable
    assert RegimeState(symbol="NQ", ts=TS, regime=Regime.CHOP).is_tradable


# --- scores ----------------------------------------------------------------


def _cs(component, magnitude, direction, weight, **kw):
    return ComponentScore(component=component, magnitude=magnitude,
                          direction=direction, weight=weight, **kw)


def test_component_points_are_weight_times_magnitude():
    assert _cs(Component.ORDER_FLOW, 0.8, 1, 25.0).points == pytest.approx(20.0)


def test_disabled_component_contributes_nothing():
    cs = _cs(Component.OPTIONS_FLOW, 1.0, 1, 20.0, enabled=False)
    assert cs.points == 0.0


def test_component_magnitude_is_bounded():
    with pytest.raises(ValidationError):
        _cs(Component.STRUCTURE, 1.5, 1, 20.0)
    with pytest.raises(ValidationError):
        _cs(Component.STRUCTURE, -0.1, 1, 20.0)


def test_component_direction_is_trinary():
    with pytest.raises(ValidationError):
        _cs(Component.STRUCTURE, 0.5, 2, 20.0)


def test_component_agreement_helpers():
    bull = _cs(Component.ORDER_FLOW, 0.5, 1, 25.0)
    assert bull.agrees_with(Side.LONG) and bull.opposes(Side.SHORT)
    neutral = _cs(Component.ORDER_FLOW, 0.5, 0, 25.0)
    assert not neutral.agrees_with(Side.LONG) and not neutral.opposes(Side.LONG)


def test_bearish_component_cannot_inflate_a_bullish_score():
    """Magnitude and direction are separate; this is the bug being prevented."""
    comps = {
        Component.ORDER_FLOW: _cs(Component.ORDER_FLOW, 1.0, 1, 25.0),
        Component.STRUCTURE: _cs(Component.STRUCTURE, 1.0, -1, 20.0),
    }
    fs = FlowScore(symbol="NQ", ts=TS, components=comps)
    assert fs.score == pytest.approx(45.0)              # magnitude aggregate
    assert fs.opposing_points(Side.LONG) == pytest.approx(20.0)  # but opposition is visible
    assert fs.net_direction() == 1


def test_flow_score_net_direction_balanced_is_zero():
    comps = {
        Component.ORDER_FLOW: _cs(Component.ORDER_FLOW, 0.8, 1, 25.0),
        Component.STRUCTURE: _cs(Component.STRUCTURE, 1.0, -1, 20.0),
    }
    assert FlowScore(symbol="NQ", ts=TS, components=comps).net_direction() == 0


def test_flow_score_available_points_reports_disabled_weight():
    comps = {
        Component.ORDER_FLOW: _cs(Component.ORDER_FLOW, 0.8, 1, 25.0),
        Component.OPTIONS_FLOW: _cs(Component.OPTIONS_FLOW, 0.0, 0, 20.0,
                                    enabled=False, quality=DataQuality.MISSING),
    }
    fs = FlowScore(symbol="NQ", ts=TS, components=comps)
    assert fs.available_points == pytest.approx(25.0)
    assert fs.quality is DataQuality.GOOD       # disabled components excluded from quality


def test_flow_score_cannot_exceed_its_maximum():
    comps = {c: _cs(c, 1.0, 1, 30.0) for c in Component}   # 150 points
    with pytest.raises(ValidationError, match="exceeds max_points"):
        FlowScore(symbol="NQ", ts=TS, components=comps)


def test_flow_score_all_disabled_is_missing_quality():
    comps = {Component.ORDER_FLOW: _cs(Component.ORDER_FLOW, 0.5, 1, 25.0, enabled=False)}
    assert FlowScore(symbol="NQ", ts=TS, components=comps).quality is DataQuality.MISSING


# --- signals ---------------------------------------------------------------


def test_wait_signal_must_name_its_gate():
    with pytest.raises(ValidationError, match="must state a wait_reason"):
        Signal(symbol="NQ", ts=TS, action=SignalAction.WAIT)
    ok = Signal(symbol="NQ", ts=TS, action=SignalAction.WAIT,
                wait_reason=WaitReason.LIQUIDITY)
    assert ok.side is None


def test_directional_signal_requires_setup_and_score():
    with pytest.raises(ValidationError, match="must name a setup"):
        Signal(symbol="NQ", ts=TS, action=SignalAction.LONG, flow_score=80.0)
    with pytest.raises(ValidationError, match="must carry a flow_score"):
        Signal(symbol="NQ", ts=TS, action=SignalAction.LONG, setup=SetupType.SCALP_1R)
    with pytest.raises(ValidationError, match="must not set wait_reason"):
        Signal(symbol="NQ", ts=TS, action=SignalAction.LONG, setup=SetupType.SCALP_1R,
               flow_score=80.0, wait_reason=WaitReason.LIQUIDITY)


def test_signal_side_mapping():
    long = Signal(symbol="NQ", ts=TS, action=SignalAction.LONG,
                  setup=SetupType.SCALP_1R, flow_score=80.0)
    short = long.replace(action=SignalAction.SHORT)
    assert long.side is Side.LONG and short.side is Side.SHORT


def test_flow_score_field_is_bounded_on_signal():
    with pytest.raises(ValidationError):
        Signal(symbol="NQ", ts=TS, action=SignalAction.LONG,
               setup=SetupType.SCALP_1R, flow_score=101.0)


# --- intents: the R definition --------------------------------------------


def _intent(**kw):
    base = dict(
        symbol="NQ", signal_ts=TS, side=Side.LONG, setup=SetupType.SCALP_1R,
        entry_price=18000.0, stop_price=17980.0, target_price=18020.0,
        size=1.0, risk_dollars=400.0, point_value=20.0,
        equity_at_signal=100_000.0, flow_score=75.0, regime=Regime.LOW_VOL,
    )
    base.update(kw)
    return TradeIntent(**base)


def test_planned_r_is_target_over_stop_distance():
    assert _intent().planned_r_multiple == pytest.approx(1.0)
    assert _intent(target_price=18040.0).planned_r_multiple == pytest.approx(2.0)
    assert _intent(target_price=18008.0).planned_r_multiple == pytest.approx(0.4)


def test_potential_reward_dollars():
    assert _intent(target_price=18040.0).potential_reward_dollars == pytest.approx(800.0)


def test_long_geometry_enforced():
    with pytest.raises(ValidationError, match="LONG stop must be below entry"):
        _intent(stop_price=18010.0)
    with pytest.raises(ValidationError, match="LONG target must be above entry"):
        _intent(target_price=17990.0)


def test_short_geometry_enforced():
    short = dict(side=Side.SHORT, entry_price=18000.0, stop_price=18020.0,
                 target_price=17980.0)
    assert _intent(**short).planned_r_multiple == pytest.approx(1.0)
    with pytest.raises(ValidationError, match="SHORT stop must be above entry"):
        _intent(side=Side.SHORT, stop_price=17980.0, target_price=17960.0)
    with pytest.raises(ValidationError, match="SHORT target must be below entry"):
        _intent(side=Side.SHORT, stop_price=18020.0, target_price=18050.0)


def test_risk_dollars_must_match_the_geometry():
    """Prevents the silent corruption of every R figure downstream."""
    with pytest.raises(ValidationError, match="inconsistent with"):
        _intent(risk_dollars=100.0)
    assert _intent(size=3.0, risk_dollars=1200.0).risk_dollars == pytest.approx(1200.0)


def test_zero_width_stop_rejected():
    with pytest.raises(ValidationError):
        _intent(stop_price=18000.0)


# --- trade records ---------------------------------------------------------


def test_realized_r_is_net_pnl_over_risk(make_trade):
    trade = make_trade()
    assert trade.r_multiple == pytest.approx(1.0)
    assert trade.pnl == pytest.approx(400.0)


def test_costs_are_charged_in_r_terms(make_trade):
    trade = make_trade(costs=CostBreakdown(commission=5.24, entry_slippage=5.0,
                                           exit_slippage=5.0, spread_cost=5.0))
    assert trade.costs.total == pytest.approx(20.24)
    assert trade.cost_drag_r == pytest.approx(20.24 / 400.0)
    assert trade.r_multiple == pytest.approx((400.0 - 20.24) / 400.0)


def test_losing_trade_is_minus_one_r(make_trade):
    trade = make_trade(exit_price=17980.0, exit_reason=ExitReason.STOP, mfe_r=0.2, mae_r=-1.0)
    assert trade.r_multiple == pytest.approx(-1.0)
    assert trade.is_loss and not trade.is_win


def test_scratch_trade_classified_separately(make_trade):
    trade = make_trade(exit_price=18000.0, exit_reason=ExitReason.TIME_STOP, mfe_r=0.1)
    assert trade.is_scratch and not trade.is_win and not trade.is_loss


def test_short_trade_pnl_sign(make_trade):
    trade = make_trade(side=Side.SHORT, entry=18000.0, stop=18020.0)
    assert trade.target_price == pytest.approx(17980.0)
    assert trade.r_multiple == pytest.approx(1.0)


def test_r_label_honesty_catches_a_mislabelled_scalp(make_trade):
    """A 'SCALP_1R' that really targets 0.4R is the main way a 90% win rate
    gets manufactured. The record knows."""
    honest = make_trade(rr=1.0)
    assert honest.r_label_is_honest()
    dishonest = make_trade(rr=0.4)
    assert dishonest.planned_r_multiple == pytest.approx(0.4)
    assert not dishonest.r_label_is_honest()


def test_r_label_honesty_for_2r_and_3r(make_trade):
    assert make_trade(setup=SetupType.SETUP_2R, rr=2.0).r_label_is_honest()
    assert not make_trade(setup=SetupType.SETUP_2R, rr=1.2).r_label_is_honest()
    # 3R+ is a floor, not an equality
    assert make_trade(setup=SetupType.DIRECTIONAL_3R, rr=3.0).r_label_is_honest()
    assert make_trade(setup=SetupType.DIRECTIONAL_3R, rr=4.5).r_label_is_honest()
    assert not make_trade(setup=SetupType.DIRECTIONAL_3R, rr=2.0).r_label_is_honest()


def test_holding_time(make_trade):
    assert make_trade(hold_minutes=30).holding_time_seconds == pytest.approx(1800.0)


def test_timestamp_ordering_enforced(make_trade):
    trade = make_trade()
    with pytest.raises(ValidationError, match="cannot happen before its signal"):
        trade.replace(entry_ts=trade.signal_ts - timedelta(minutes=1))
    with pytest.raises(ValidationError, match="exit_ts precedes entry_ts"):
        trade.replace(exit_ts=trade.entry_ts - timedelta(seconds=1))


def test_pnl_accounting_cannot_be_falsified(make_trade):
    trade = make_trade()
    with pytest.raises(ValidationError, match="inconsistent with price path"):
        trade.replace(gross_pnl=10_000.0, pnl=10_000.0)
    with pytest.raises(ValidationError, match=r"!= gross_pnl - costs"):
        trade.replace(pnl=10_000.0)


def test_filled_size_cannot_exceed_requested(make_trade):
    with pytest.raises(ValidationError, match="filled_size exceeds"):
        make_trade().replace(filled_size=2.0)


def test_excursion_signs_enforced(make_trade):
    trade = make_trade()
    with pytest.raises(ValidationError):
        trade.replace(mae_r=0.5)        # MAE is adverse: must be <= 0
    with pytest.raises(ValidationError):
        trade.replace(mfe_r=-0.5)       # MFE is favourable: must be >= 0


def test_partial_fill_scales_pnl(make_trade):
    """A partial fill must not be paid as if fully filled."""
    trade = make_trade(size=4.0)
    partial = trade.replace(
        filled_size=2.0,
        gross_pnl=(18020.0 - 18000.0) * 20.0 * 2.0,
        pnl=(18020.0 - 18000.0) * 20.0 * 2.0,
    )
    assert partial.gross_pnl == pytest.approx(800.0)
    # risk_dollars still reflects the committed risk of the requested size
    assert partial.r_multiple == pytest.approx(800.0 / 1600.0)


# --- equity ----------------------------------------------------------------


def test_equity_drawdown_pct():
    point = EquityPoint(ts=TS, equity=90_000.0, realized_equity=90_000.0,
                        high_water_mark=100_000.0)
    assert point.drawdown_pct == pytest.approx(-0.10)


def test_equity_drawdown_with_no_high_water_mark_is_zero():
    point = EquityPoint(ts=TS, equity=100.0, realized_equity=100.0, high_water_mark=0.0)
    assert point.drawdown_pct == 0.0


def test_cost_breakdown_total_and_non_negativity():
    assert CostBreakdown(commission=1, exchange_fees=2, entry_slippage=3,
                         exit_slippage=4, spread_cost=5).total == pytest.approx(15.0)
    with pytest.raises(ValidationError):
        CostBreakdown(commission=-1)
