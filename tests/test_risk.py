from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.models import ClosedTrade, Side, TradeSignal
from tradebot.risk.engine import RiskEngine

NOW = datetime(2026, 3, 10, 14, 0, tzinfo=timezone.utc)


def long_signal(entry=100.0, stop=98.0, tp1=104.0, tp2=108.0) -> TradeSignal:
    return TradeSignal("TEST", "5m", Side.LONG, entry, stop, tp1, tp2, ts=NOW)


def losing_trade(pnl=-500.0, at=NOW) -> ClosedTrade:
    return ClosedTrade("x", "TEST", Side.LONG, 100, 98, 1, pnl, -1.0, at, at, "stop")


def test_a_valid_signal_is_sized_and_allowed(config):
    engine = RiskEngine(config)
    decision = engine.evaluate(long_signal(), open_positions=0, now=NOW)
    assert decision.allowed
    # 0.5% of 100k = 500, stop is 2.0 wide at $1/point -> 250 units.
    assert decision.qty == pytest.approx(250.0)
    assert decision.risk_pct == pytest.approx(0.005)


def test_kill_switch_blocks_everything(config):
    engine = RiskEngine(config)
    engine.trip_kill_switch("manual")
    assert not engine.evaluate(long_signal(), 0, NOW).allowed
    engine.release_kill_switch()
    assert engine.evaluate(long_signal(), 0, NOW).allowed


def test_a_missing_stop_is_refused(config):
    engine = RiskEngine(config)
    decision = engine.evaluate(long_signal(stop=0.0), 0, NOW)
    assert not decision.allowed and "missing_stop" in decision.violations


def test_poor_reward_to_risk_is_refused(config):
    config.risk.min_rr = 2.0
    engine = RiskEngine(config)
    decision = engine.evaluate(long_signal(tp1=101.0), 0, NOW)   # 0.5R
    assert not decision.allowed and "min_rr" in decision.violations


def test_daily_trade_cap(config):
    config.risk.max_trades_per_day = 2
    engine = RiskEngine(config)
    for _ in range(2):
        engine.register_entry(long_signal())
    decision = engine.evaluate(long_signal(), 0, NOW)
    assert not decision.allowed and "max_trades_per_day" in decision.violations


def test_max_open_positions(config):
    engine = RiskEngine(config)
    decision = engine.evaluate(long_signal(), open_positions=1, now=NOW)
    assert not decision.allowed and "max_open_positions" in decision.violations


def test_cooldown_after_consecutive_losses(config):
    config.risk.max_consecutive_losses = 2
    config.risk.cooldown_minutes_after_loss = 30
    engine = RiskEngine(config)
    engine.register_close(losing_trade(-100))
    assert engine.evaluate(long_signal(), 0, NOW).allowed        # one loss is fine
    engine.register_close(losing_trade(-100))
    blocked = engine.evaluate(long_signal(), 0, NOW + timedelta(minutes=5))
    assert not blocked.allowed and "cooldown" in blocked.violations
    # ...and it expires on its own.
    assert engine.evaluate(long_signal(), 0, NOW + timedelta(minutes=35)).allowed


def test_a_win_resets_the_loss_streak(config):
    engine = RiskEngine(config)
    engine.register_close(losing_trade(-100))
    engine.register_close(ClosedTrade("y", "TEST", Side.LONG, 100, 104, 1, 300, 2.0, NOW, NOW, "tp1"))
    assert engine.state.consecutive_losses == 0
    assert engine.state.cooldown_until is None


def test_daily_loss_limit_trips_the_kill_switch(config):
    config.risk.max_daily_loss_pct = 2.0
    engine = RiskEngine(config)
    engine.register_close(losing_trade(-2500))      # 2.5% of 100k
    assert engine.state.kill_switch
    assert not engine.evaluate(long_signal(), 0, NOW).allowed


def test_size_rounds_down_and_refuses_sub_minimum(config):
    config.market.min_qty = 1.0
    config.market.qty_step = 1.0
    engine = RiskEngine(config)
    wide = long_signal(entry=100.0, stop=0.5, tp1=300.0, tp2=500.0)   # 99.5 wide
    qty, _, _ = engine.position_size(wide)
    assert qty == 5.0                                # floor(500 / 99.5)
    config.risk.risk_per_trade_pct = 0.0001          # $0.10 of budget
    assert engine.position_size(wide)[0] == 0.0


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
