from __future__ import annotations

import pytest

from tradebot.indicators import (
    atr,
    crossed_above,
    crossed_below,
    ema,
    rsi,
    session_vwap,
    sma,
)


def test_sma_pads_until_warm():
    out = sma([1, 2, 3, 4], 3)
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)


def test_ema_seeds_from_sma_and_tracks_price():
    values = [10.0] * 20
    out = ema(values, 5)
    assert out[3] is None
    assert out[4] == pytest.approx(10.0)
    assert out[-1] == pytest.approx(10.0)      # flat input -> flat EMA


def test_ema_reacts_faster_than_a_slower_ema():
    values = [10.0] * 30 + [20.0] * 10
    fast, slow = ema(values, 5), ema(values, 20)
    assert fast[-1] > slow[-1]


def test_rsi_bounds():
    rising = [float(i) for i in range(40)]
    falling = list(reversed(rising))
    assert rsi(rising)[-1] == pytest.approx(100.0)
    assert rsi(falling)[-1] == pytest.approx(0.0, abs=1e-6)
    assert all(0 <= v <= 100 for v in rsi([10, 11, 10.5, 12, 11.5] * 10) if v is not None)


def test_rsi_needs_enough_history():
    assert all(v is None for v in rsi([1, 2, 3], 14))


def test_atr_of_constant_range_is_that_range():
    highs = [11.0] * 30
    lows = [9.0] * 30
    closes = [10.0] * 30
    assert atr(highs, lows, closes, 14)[-1] == pytest.approx(2.0)


def test_session_vwap_resets_on_the_session_flag():
    highs = lows = closes = [10.0] * 3 + [20.0] * 3
    volumes = [1.0] * 6
    flags = [True, False, False, True, False, False]
    out = session_vwap(highs, lows, closes, volumes, flags)
    assert out[2] == pytest.approx(10.0)
    assert out[3] == pytest.approx(20.0)      # reset, not blended with the 10s


def test_session_vwap_falls_back_to_typical_price_without_volume():
    prices = [10.0, 20.0]
    out = session_vwap(prices, prices, prices, [0.0, 0.0], [True, False])
    assert out[-1] == pytest.approx(15.0)


def test_cross_detection_reports_bars_ago():
    fast = [1, 1, 1, 5, 6]
    slow = [3, 3, 3, 3, 3]
    assert crossed_above(fast, slow, 5) == 1
    assert crossed_below(fast, slow, 5) is None
    assert crossed_below(list(reversed(fast)), slow, 5) is not None


def test_zero_period_is_rejected():
    with pytest.raises(ValueError):
        ema([1, 2, 3], 0)
