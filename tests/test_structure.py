from __future__ import annotations

from conftest import make_candles

from tradebot.models import Candle
from tradebot.structure import (
    classify_trend,
    detect_break,
    detect_liquidity_sweep,
    find_swings,
)
from datetime import timedelta


def test_find_swings_marks_the_peak():
    candles = make_candles([10, 11, 12, 15, 12, 11, 10], spread=0.0)
    highs = [s.index for s in find_swings(candles, 2) if s.kind == "high"]
    assert 3 in highs


def test_trend_classification():
    up = make_candles([10, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22], spread=0.0)
    down = make_candles(list(reversed([10, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22])), spread=0.0)
    assert classify_trend(up, 1) == "bullish"
    assert classify_trend(down, 1) == "bearish"


def test_trend_is_range_without_enough_swings():
    assert classify_trend(make_candles([10, 10, 10, 10]), 2) == "range"


def test_structure_break_detects_a_close_above_a_prior_swing_high():
    candles = make_candles([10, 11, 14, 11, 10, 11, 16], spread=0.0)
    result = detect_break(candles, width=1, lookback=3)
    assert result.direction == "up"
    assert result.kind in ("BOS", "CHoCH")


def test_liquidity_sweep_needs_a_rejection_wick():
    base = make_candles([100, 99, 98, 99, 100, 101], spread=0.5)
    # A bar that pokes under the swing low at index 2 and closes back above it.
    sweeper = Candle(
        base[-1].ts + timedelta(minutes=5),
        open=100.0, high=100.5, low=96.0, close=100.0, volume=2000,
    )
    swept = detect_liquidity_sweep([*base, sweeper], width=1, lookback=2)
    assert swept.happened and swept.bullish

    # Same poke, but it closes on the low - no rejection, so no sweep.
    no_reject = Candle(
        base[-1].ts + timedelta(minutes=5),
        open=100.0, high=100.5, low=96.0, close=96.1, volume=2000,
    )
    assert not detect_liquidity_sweep([*base, no_reject], width=1, lookback=2).happened


def test_no_sweep_on_a_short_series():
    assert not detect_liquidity_sweep(make_candles([1, 2, 3])).happened
