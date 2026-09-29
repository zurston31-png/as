from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from tradebot.execution.book import PositionState
from tradebot.models import ClosedTrade, Side, TradeSignal
from tradebot.risk.engine import RejectCode, RiskEngine

NOW = datetime(2026, 3, 10, 14, 0, tzinfo=timezone.utc)
FLAT = PositionState()


def long_signal(entry=100.0, stop=98.0, tp1=104.0, tp2=108.0) -> TradeSignal:
    return TradeSignal("TEST", "5m", Side.LONG, entry, stop, tp1, tp2, ts=NOW)


def losing_trade(pnl=-500.0, at=NOW) -> ClosedTrade:
    return ClosedTrade("x", "TEST", Side.LONG, 100, 98, 1, pnl, -1.0, at, at, "stop")


# ----------------------------------------------------------------- sizing

def test_the_sizing_formula_is_explicit_and_reproducible(config):
    """risk_dollars / (|entry-stop| x point_value), floored."""
    engine = RiskEngine(config)
    sizing = engine.position_size(long_signal())

    assert sizing["equity"] == 100_000.0
    assert sizing["risk_percent"] == pytest.approx(0.005)
    assert sizing["risk_dollars"] == pytest.approx(500.0)
    assert sizing["stop_distance"] == pytest.approx(2.0)
    assert sizing["point_value"] == 1.0
    assert sizing["stop_risk_per_contract"] == pytest.approx(2.0)
    assert sizing["raw_contracts"] == pytest.approx(250.0)
    assert sizing["contracts"] == 250.0
    assert sizing["risk_amount"] == pytest.approx(500.0)

    # Every number above re-derives from the others.
    assert sizing["risk_dollars"] == pytest.approx(sizing["equity"] * sizing["risk_percent"])
    assert sizing["stop_risk_per_contract"] == pytest.approx(
        sizing["stop_distance"] * sizing["point_value"])
    assert sizing["contracts"] == math.floor(
        sizing["risk_dollars"] / sizing["stop_risk_per_contract"])


def test_size_always_rounds_down(config):
    config.market.qty_step = 1.0
    engine = RiskEngine(config)
    sizing = engine.position_size(long_signal(entry=100.0, stop=97.0))   # 500/3 = 166.67
    assert sizing["raw_contracts"] == pytest.approx(166.667, abs=0.01)
    assert sizing["contracts"] == 166.0


def test_zero_contracts_is_a_rejection_not_an_upsize(config):
    """The exact case the MNQ point-value problem produces."""
    config.market.point_value = 20.0        # full-size NQ
    config.risk.starting_equity = 25_000.0
    engine = RiskEngine(config)
    signal = long_signal(entry=25_420, stop=25_360, tp1=25_540, tp2=25_680)

    decision = engine.evaluate(signal, FLAT, NOW)
    assert not decision.allowed
    assert decision.reason_code == RejectCode.SIZE_BELOW_MIN
    assert decision.qty == 0.0
    # 0.5% of 25k = $125; one contract risks 60 x 20 = $1200.
    assert decision.sizing["stop_risk_per_contract"] == pytest.approx(1200.0)
    assert decision.sizing["raw_contracts"] < 1.0
    # Risk was not raised to make it fit.
    assert decision.sizing["risk_dollars"] == pytest.approx(125.0)


def test_fractional_instruments_size_fractionally(config):
    config.market.qty_step = 0.0001
    config.market.min_qty = 0.0001
    config.market.point_value = 1.0
    engine = RiskEngine(config)
    sizing = engine.position_size(long_signal(entry=50_000, stop=49_000,
                                              tp1=52_000, tp2=54_000))
    assert 0 < sizing["contracts"] < 1


# ------------------------------------------------------------------ gates

def test_a_valid_signal_is_allowed(config):
    decision = RiskEngine(config).evaluate(long_signal(), FLAT, NOW)
    assert decision.allowed and decision.reason_code == RejectCode.OK
    assert decision.qty == pytest.approx(250.0)


def test_kill_switch_blocks_everything(config):
    engine = RiskEngine(config)
    engine.trip_kill_switch("manual")
    assert engine.evaluate(long_signal(), FLAT, NOW).reason_code == RejectCode.KILL_SWITCH
    engine.release_kill_switch()
    assert engine.evaluate(long_signal(), FLAT, NOW).allowed


def test_unreconciled_positions_block_before_anything_else(config):
    engine = RiskEngine(config)
    decision = engine.evaluate(long_signal(), FLAT, NOW, reconciled=False,
                               reconciliation_message="broker has a position we don't")
    assert not decision.allowed
    assert decision.reason_code == RejectCode.RECONCILIATION
    assert "broker has a position" in decision.reason


def test_a_missing_stop_is_refused(config):
    decision = RiskEngine(config).evaluate(long_signal(stop=0.0), FLAT, NOW)
    assert decision.reason_code == RejectCode.NO_STOP


def test_poor_reward_to_risk_is_refused(config):
    config.risk.min_rr = 2.0
    decision = RiskEngine(config).evaluate(long_signal(tp1=101.0), FLAT, NOW)
    assert decision.reason_code == RejectCode.MIN_RR


def test_same_direction_stacking_is_refused(config):
    held = PositionState("long", Side.LONG, 10, 1)
    decision = RiskEngine(config).evaluate(long_signal(), held, NOW)
    assert decision.reason_code == RejectCode.SAME_DIRECTION


def test_an_opposite_position_is_refused(config):
    held = PositionState("short", Side.SHORT, 10, 1)
    decision = RiskEngine(config).evaluate(long_signal(), held, NOW)
    assert decision.reason_code == RejectCode.OPPOSITE_DIRECTION
    assert "flatten before reversing" in decision.reason


def test_daily_trade_cap(config):
    config.risk.max_trades_per_day = 2
    engine = RiskEngine(config)
    for _ in range(2):
        engine.register_entry(long_signal())
    assert engine.evaluate(long_signal(), FLAT, NOW).reason_code == RejectCode.MAX_TRADES


def test_cooldown_after_consecutive_losses(config):
    config.risk.max_consecutive_losses = 2
    config.risk.cooldown_minutes_after_loss = 30
    engine = RiskEngine(config)
    engine.register_close(losing_trade(-100))
    assert engine.evaluate(long_signal(), FLAT, NOW).allowed        # one loss is fine
    engine.register_close(losing_trade(-100))
    blocked = engine.evaluate(long_signal(), FLAT, NOW + timedelta(minutes=5))
    assert blocked.reason_code == RejectCode.COOLDOWN
    assert engine.evaluate(long_signal(), FLAT, NOW + timedelta(minutes=35)).allowed


def test_a_win_resets_the_loss_streak(config):
    engine = RiskEngine(config)
    engine.register_close(losing_trade(-100))
    engine.register_close(ClosedTrade("y", "TEST", Side.LONG, 100, 104, 1, 300, 2.0,
                                      NOW, NOW, "tp1"))
    assert engine.state.consecutive_losses == 0
    assert engine.state.cooldown_until is None


def test_daily_loss_limit_trips_the_kill_switch(config):
    config.risk.max_daily_loss_pct = 2.0
    engine = RiskEngine(config)
    engine.register_close(losing_trade(-2500))      # 2.5% of 100k
    assert engine.state.kill_switch
    assert engine.evaluate(long_signal(), FLAT, NOW).reason_code == RejectCode.KILL_SWITCH


def test_risk_ceiling_is_enforced(config):
    config.risk.risk_per_trade_pct = 5.0
    config.risk.max_risk_per_trade_pct = 2.0
    decision = RiskEngine(config).evaluate(long_signal(), FLAT, NOW)
    assert decision.reason_code == RejectCode.RISK_CEILING


# ------------------------------------------------------------------ state

def test_state_survives_a_restart(config):
    engine = RiskEngine(config)
    engine.register_entry(long_signal())
    engine.trip_kill_switch("manual: testing")
    reloaded = RiskEngine(config)
    assert reloaded.state.trades_today == 1
    assert reloaded.state.kill_switch


def test_rolling_the_day_resets_counters_but_keeps_a_manual_halt(config):
    engine = RiskEngine(config)
    engine.register_entry(long_signal())
    engine.trip_kill_switch("manual: overnight halt")
    engine.store.roll_day(NOW.date() + timedelta(days=1))
    assert engine.state.trades_today == 0
    assert engine.state.kill_switch              # a human flipped it, so it stays
    assert engine.state.history and engine.state.history[-1]["trades"] == 1


def test_rolling_the_day_clears_an_automatic_halt(config):
    engine = RiskEngine(config)
    engine.trip_kill_switch("auto: daily loss limit -2.10%")
    engine.store.roll_day(NOW.date() + timedelta(days=1))
    assert not engine.state.kill_switch


def test_the_day_never_rolls_backwards(config):
    """An out-of-order bar must not hand you a fresh set of daily limits."""
    engine = RiskEngine(config)
    engine.tick(NOW)
    engine.register_entry(long_signal())
    engine.tick(NOW - timedelta(days=3))
    assert engine.state.trades_today == 1
