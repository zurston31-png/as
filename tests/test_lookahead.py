"""No lookahead.

A backtest that lets a decision see data from after its own timestamp measures
something you will never trade. These tests pin the guarantee down from four
directions: forming bars are invisible, later bars are invisible, an indicator
at bar N doesn't change when bar N+1 arrives, and the broker never fills a
position against the bar that created it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta

import pytest
from conftest import BASE, make_candles

from tradebot.audit import NullAuditLog
from tradebot.models import Candle, Series, Side, TradeSignal
from tradebot.orchestrator import Orchestrator
from tradebot.strategy.context import ContextBuilder, LookaheadError


def series_of(candles) -> Series:
    series = Series("TEST", "5m")
    series.extend(candles)
    return series


def test_a_forming_bar_is_invisible_to_the_context(config):
    candles = make_candles([100.0] * 30)
    forming = Candle(candles[-1].ts + timedelta(minutes=5),
                     open=100, high=999, low=1, close=500, volume=99, closed=False)
    ctx = ContextBuilder(config).build(series_of([*candles, forming]))

    assert ctx.candle.ts == candles[-1].ts
    assert ctx.price == 100.0                      # not 500
    assert all(c.closed for c in ctx.candles)


def test_the_context_stops_at_the_decision_timestamp(config):
    candles = make_candles([100 + i for i in range(30)])
    cutoff = candles[10].ts
    ctx = ContextBuilder(config).build(series_of(candles), decision_ts=cutoff)

    assert ctx.bars == 11
    assert ctx.candle.ts == cutoff
    assert max(c.ts for c in ctx.candles) <= cutoff


def test_indicators_match_a_truncated_series(config):
    """An indicator at bar N must not change when later bars arrive."""
    candles = make_candles([100 + (i % 7) * 1.5 for i in range(80)])
    cutoff = candles[50].ts

    a = ContextBuilder(config).build(series_of(candles[:51]))
    b = ContextBuilder(config).build(series_of(candles), decision_ts=cutoff)

    for period in (9, 21):
        assert a.ema(period) == pytest.approx(b.ema(period))
    assert a.rsi(14) == pytest.approx(b.rsi(14))
    assert a.atr(14) == pytest.approx(b.atr(14))
    assert a.vwap() == pytest.approx(b.vwap())
    assert a.trend() == b.trend()
    assert a.swing("low") == b.swing("low")


@dataclass
class RawSeries:
    """A series-shaped object that skips `Series.upsert`'s ordering guarantees."""

    symbol: str
    timeframe: str
    candles: list


def test_out_of_order_bars_are_rejected(config):
    """Indicators assume time order; a late-replayed bar must not slip through."""
    candles = make_candles([100.0] * 30)
    scrambled = [*candles[:10], candles[20], *candles[10:20], *candles[21:]]

    with pytest.raises(LookaheadError, match="out of order"):
        ContextBuilder(config).build(RawSeries("TEST", "5m", scrambled))


def test_a_correctly_ordered_series_passes_the_check(config):
    candles = make_candles([100.0] * 30)
    assert ContextBuilder(config).build(RawSeries("TEST", "5m", candles)) is not None


def test_the_series_itself_rejects_an_out_of_order_bar(config):
    """`Series.upsert` is the first line of defence; the context check is the second."""
    candles = make_candles([100.0] * 10)
    series = series_of(candles)
    assert not series.upsert(candles[2])      # older than the tail - ignored
    assert len(series) == 10


def test_a_position_never_fills_against_its_own_entry_bar(config):
    """The bar that triggers an entry must not also be the bar that exits it."""
    bot = Orchestrator(config, audit=NullAuditLog())
    # A bar whose range spans both the stop and TP2 of the signal below.
    wide = Candle(BASE, open=100, high=120, low=80, close=100, volume=10)
    signal = TradeSignal("TEST", "5m", Side.LONG, 100, 98, 104, 108, ts=BASE)

    asyncio.run(bot.on_candle(wide))               # existing positions marked first
    bot.broker.open(signal, 1)                     # then the entry is opened
    assert bot.broker.positions                    # it survived its own bar
    assert not bot.broker.closed


def test_the_next_bar_does_fill_it(config):
    bot = Orchestrator(config, audit=NullAuditLog())
    first = Candle(BASE, open=100, high=101, low=99, close=100, volume=10)
    signal = TradeSignal("TEST", "5m", Side.LONG, 100, 98, 104, 108, ts=BASE)

    asyncio.run(bot.on_candle(first))
    bot.broker.open(signal, 1)
    asyncio.run(bot.on_candle(
        Candle(BASE + timedelta(minutes=5), 100, 120, 80, 100, 10)))
    assert bot.broker.closed and not bot.broker.positions
