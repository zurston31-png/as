from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import BASE

from tradebot.execution.base import LiveBroker, LiveBrokerNotConfigured
from tradebot.execution.paper import PaperBroker
from tradebot.models import Candle, Side, TradeSignal


def signal(side=Side.LONG, entry=100.0, stop=98.0, tp1=104.0, tp2=108.0) -> TradeSignal:
    return TradeSignal("TEST", "5m", side, entry, stop, tp1, tp2, ts=BASE)


def bar(high, low, close, offset=1):
    """One bar with the given range; open is the low, which the tests don't rely on."""
    return Candle(BASE + timedelta(minutes=5 * offset), low, high, low, close, 100)


def test_entry_takes_slippage_against_you(config):
    config.execution.slippage_ticks = 2.0          # 2 x 0.25 = 0.5
    broker = PaperBroker(config)
    pos = broker.open(signal(), qty=10)
    assert pos.entry_price == pytest.approx(100.5)

    short = broker.open(signal(side=Side.SHORT, stop=102, tp1=96, tp2=92), qty=10)
    assert short.entry_price == pytest.approx(99.5)


def test_stop_closes_the_position_at_a_loss(config):
    broker = PaperBroker(config)
    broker.open(signal(), qty=10)
    closed = broker.on_candle(bar(high=101, low=97, close=97.5))
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop"
    assert closed[0].pnl == pytest.approx(-20.0)    # (98 - 100) * 10
    assert closed[0].r_multiple == pytest.approx(-1.0)
    assert not broker.positions


def test_a_bar_that_spans_stop_and_target_is_treated_as_a_stop(config):
    broker = PaperBroker(config)
    broker.open(signal(), qty=10)
    closed = broker.on_candle(bar(high=105, low=97, close=104))
    assert closed[0].exit_reason == "stop"


def test_tp1_scales_out_and_moves_the_stop_to_breakeven(config):
    config.execution.partial_at_tp1 = 0.5
    config.execution.move_stop_to_breakeven_after_tp1 = True
    broker = PaperBroker(config)
    pos = broker.open(signal(), qty=10)
    assert broker.on_candle(bar(high=104.5, low=100.5, close=104)) == []
    assert pos.tp1_hit and pos.remaining == pytest.approx(5.0)
    assert pos.stop == pytest.approx(pos.entry_price)
    assert pos.realized == pytest.approx(20.0)      # 5 units x 4 points


def test_runner_reaches_tp2_for_the_full_result(config):
    config.execution.partial_at_tp1 = 0.5
    broker = PaperBroker(config)
    broker.open(signal(), qty=10)
    broker.on_candle(bar(high=104.5, low=100.5, close=104, offset=1))
    closed = broker.on_candle(bar(high=108.5, low=104, close=108.2, offset=2))
    assert len(closed) == 1 and closed[0].exit_reason == "tp2"
    assert closed[0].pnl == pytest.approx(20.0 + 40.0)   # 5 @ +4, then 5 @ +8
    assert closed[0].r_multiple == pytest.approx(3.0)


def test_shorts_are_the_mirror_image(config):
    broker = PaperBroker(config)
    broker.open(signal(side=Side.SHORT, entry=100, stop=102, tp1=96, tp2=92), qty=10)
    closed = broker.on_candle(bar(high=99, low=91.5, close=92))
    assert closed and closed[0].pnl > 0


def test_no_partial_when_scaling_is_disabled(config):
    config.execution.partial_at_tp1 = 0.0
    broker = PaperBroker(config)
    broker.open(signal(), qty=10)
    closed = broker.on_candle(bar(high=104.5, low=100.5, close=104))
    assert closed and closed[0].exit_reason == "tp1"


def test_closed_trades_are_appended_to_the_log(config, tmp_path):
    broker = PaperBroker(config)
    broker.open(signal(), qty=10)
    broker.on_candle(bar(high=101, low=97, close=97.5))
    lines = open(config.execution.trades_path).read().strip().splitlines()
    assert len(lines) == 1 and '"exit_reason": "stop"' in lines[0]


def test_flatten_closes_everything(config):
    broker = PaperBroker(config)
    broker.open(signal(), qty=10)
    closed = broker.close_all(101.0, bar(high=101, low=100, close=101), "flatten")
    assert closed and closed[0].exit_reason == "flatten" and not broker.positions


def test_live_broker_refuses_to_pretend(config):
    with pytest.raises(LiveBrokerNotConfigured):
        LiveBroker()
