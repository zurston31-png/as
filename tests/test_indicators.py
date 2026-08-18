"""Indicator correctness.

These check against hand-computable values and mathematical invariants, not
just "it returned a number". A silently wrong RSI would poison every signal
score downstream while every other test stayed green.
"""
import pytest

from app.signals.indicators import (
    atr, atr_percent, bollinger_bands, bollinger_width, crossed_above, crossed_below,
    ema, macd, momentum, n_period_high, n_period_low, price_structure, relative_volume,
    rsi, sma, support_resistance, swing_highs, swing_lows, trend_direction, true_range,
    volume_spike, vwap, wilder_smooth,
)


# ---------------------------------------------------------------------------
# moving averages
# ---------------------------------------------------------------------------

def test_sma_matches_hand_calculation():
    values = [1, 2, 3, 4, 5]
    result = sma(values, 3)
    assert result[:2] == [None, None]          # warm-up
    assert result[2] == pytest.approx(2.0)     # (1+2+3)/3
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_sma_output_aligns_with_input_length():
    assert len(sma(list(range(50)), 10)) == 50


def test_sma_of_a_constant_series_is_that_constant():
    assert sma([7.0] * 20, 5)[-1] == pytest.approx(7.0)


def test_ema_is_seeded_from_the_first_sma():
    values = [1, 2, 3, 4, 5, 6]
    result = ema(values, 3)
    assert result[2] == pytest.approx(2.0)  # seed == SMA of first 3
    # then: (4 - 2) * (2/4) + 2 = 3.0
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_ema_reacts_faster_than_sma_to_a_step_change():
    """On a straight line EMA and SMA lag identically — both by (n-1)/2 — so
    a ramp proves nothing. A step change is where the difference shows."""
    values = [100.0] * 20 + [120.0] * 5
    assert ema(values, 10)[-1] > sma(values, 10)[-1]


def test_ema_and_sma_agree_on_a_linear_ramp():
    values = [float(v) for v in range(1, 41)]
    assert ema(values, 10)[-1] == pytest.approx(sma(values, 10)[-1])


def test_ema_of_a_constant_series_is_that_constant():
    assert ema([5.0] * 30, 10)[-1] == pytest.approx(5.0)


def test_short_series_returns_all_none():
    assert sma([1, 2], 10) == [None, None]
    assert ema([1, 2], 10) == [None, None]


def test_period_must_be_positive():
    with pytest.raises(ValueError):
        sma([1, 2, 3], 0)
    with pytest.raises(ValueError):
        ema([1, 2, 3], -1)


def test_wilder_smoothing_differs_from_ema():
    """Wilder uses 1/n, not 2/(n+1) — mixing them up shifts RSI and ATR."""
    values = [float(v) for v in range(1, 40)]
    assert wilder_smooth(values, 14)[-1] != pytest.approx(ema(values, 14)[-1])


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def test_rsi_is_100_when_price_only_rises():
    assert rsi([float(v) for v in range(1, 40)], 14)[-1] == pytest.approx(100.0)


def test_rsi_is_0_when_price_only_falls():
    assert rsi([float(v) for v in range(40, 1, -1)], 14)[-1] == pytest.approx(0.0)


def test_rsi_is_midrange_on_alternating_equal_moves():
    closes = [100.0]
    for i in range(60):
        closes.append(closes[-1] + (1 if i % 2 == 0 else -1))
    assert 40 < rsi(closes, 14)[-1] < 60


def test_rsi_stays_within_bounds():
    import random
    rng = random.Random(1)
    closes = [100.0]
    for _ in range(300):
        closes.append(max(closes[-1] * (1 + rng.gauss(0, 0.02)), 0.01))
    for value in rsi(closes, 14):
        if value is not None:
            assert 0 <= value <= 100


def test_rsi_warm_up_is_none():
    result = rsi([float(v) for v in range(1, 30)], 14)
    assert all(v is None for v in result[:14])
    assert result[14] is not None


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def test_macd_line_is_fast_ema_minus_slow_ema():
    closes = [float(v) for v in range(1, 80)]
    macd_line, signal_line, histogram = macd(closes)
    fast, slow = ema(closes, 12), ema(closes, 26)
    assert macd_line[-1] == pytest.approx(fast[-1] - slow[-1])


def test_macd_histogram_is_line_minus_signal():
    closes = [float(v) for v in range(1, 80)]
    macd_line, signal_line, histogram = macd(closes)
    assert histogram[-1] == pytest.approx(macd_line[-1] - signal_line[-1])


def test_macd_is_positive_in_an_uptrend_and_negative_in_a_downtrend():
    up = [float(v) for v in range(1, 100)]
    down = [float(v) for v in range(100, 1, -1)]
    assert macd(up)[0][-1] > 0
    assert macd(down)[0][-1] < 0


def test_macd_signal_is_none_during_warm_up():
    closes = [float(v) for v in range(1, 30)]
    _, signal_line, _ = macd(closes)
    assert signal_line[0] is None


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------

def test_true_range_uses_the_previous_close_on_a_gap():
    highs = [10, 20]
    lows = [9, 19]
    closes = [9.5, 19.5]
    # Bar 2 gapped up: |high - prev_close| = 20 - 9.5 = 10.5 beats its own range of 1
    assert true_range(highs, lows, closes)[1] == pytest.approx(10.5)


def test_atr_is_positive_and_scales_with_volatility():
    calm_h = [100 + i * 0.1 for i in range(60)]
    calm_l = [99.9 + i * 0.1 for i in range(60)]
    calm_c = [100 + i * 0.1 for i in range(60)]

    wild_h = [100 + i * 0.1 + (5 if i % 2 else 0) for i in range(60)]
    wild_l = [99.9 + i * 0.1 - (5 if i % 2 else 0) for i in range(60)]
    wild_c = [100 + i * 0.1 for i in range(60)]

    calm_atr = atr(calm_h, calm_l, calm_c)[-1]
    wild_atr = atr(wild_h, wild_l, wild_c)[-1]
    assert 0 < calm_atr < wild_atr


def test_atr_percent_is_scale_invariant():
    """A token at $0.0001 and one at $10,000 with the same shape must report
    the same ATR%, which is why sizing uses the percentage form."""
    def build(scale):
        h = [(100 + i * 0.5) * scale for i in range(60)]
        l = [(99 + i * 0.5) * scale for i in range(60)]
        c = [(99.5 + i * 0.5) * scale for i in range(60)]
        return atr_percent(h, l, c)[-1]

    assert build(1.0) == pytest.approx(build(0.00001), rel=1e-6)


def test_bollinger_bands_bracket_the_middle():
    import random
    rng = random.Random(7)
    closes = [100 + rng.gauss(0, 2) for _ in range(60)]
    upper, middle, lower = bollinger_bands(closes, 20, 2.0)
    assert lower[-1] < middle[-1] < upper[-1]


def test_bollinger_bands_collapse_on_a_flat_series():
    closes = [50.0] * 40
    upper, middle, lower = bollinger_bands(closes, 20, 2.0)
    assert upper[-1] == pytest.approx(50.0)
    assert lower[-1] == pytest.approx(50.0)
    assert bollinger_width(closes)[-1] == pytest.approx(0.0)


def test_bollinger_width_grows_with_dispersion():
    tight = [100 + (i % 2) * 0.1 for i in range(60)]
    loose = [100 + (i % 2) * 10 for i in range(60)]
    assert bollinger_width(loose)[-1] > bollinger_width(tight)[-1]


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------

def test_vwap_equals_price_when_price_is_constant():
    n = 30
    closes = [10.0] * n
    assert vwap(closes, closes, closes, [5.0] * n)[-1] == pytest.approx(10.0)


def test_vwap_is_pulled_toward_the_high_volume_price():
    highs = lows = closes = [10.0] * 9 + [20.0]
    volumes = [1.0] * 9 + [1000.0]   # nearly all volume traded at 20
    assert vwap(highs, lows, closes, volumes)[-1] > 19


def test_rolling_vwap_ignores_older_bars():
    highs = lows = closes = [10.0] * 20 + [20.0] * 10
    volumes = [1.0] * 30
    assert vwap(highs, lows, closes, volumes, period=5)[-1] == pytest.approx(20.0)


def test_relative_volume_is_one_for_steady_volume():
    assert relative_volume([100.0] * 40, 20)[-1] == pytest.approx(1.0)


def test_relative_volume_detects_a_doubling():
    volumes = [100.0] * 39 + [200.0]
    assert relative_volume(volumes, 20)[-1] > 1.9


def test_volume_spike_flags_only_the_spike():
    volumes = [100.0] * 39 + [500.0]
    flags = volume_spike(volumes, 20, multiple=2.0)
    assert flags[-1] is True
    assert not any(flags[20:39])


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def test_swing_detection_finds_the_pivot():
    highs = [1, 2, 3, 10, 3, 2, 1]
    lows = [10, 9, 8, 1, 8, 9, 10]
    assert 3 in swing_highs(highs, lookback=3)
    assert 3 in swing_lows(lows, lookback=3)


# Explicit fixtures: each pivot must be a strict local extreme over +/-3 bars,
# which a naive trending zigzag does not satisfy — the trend outruns the swing.
#            idx:  0  1  2   3  4  5   6   7  8  9  10   11 12 13
UPTREND_HIGHS = [  1, 2, 3, 10, 3, 2,  1,  2, 3, 4, 12,   4, 3, 2]
UPTREND_LOWS = [  10, 9, 8,  1, 8, 9, 10,  9, 8, 7,  2,   7, 8, 9]
DOWNTREND_HIGHS = [1, 2, 3, 12, 3, 2,  1,  2, 3, 4, 10,   4, 3, 2]
DOWNTREND_LOWS = [10, 9, 8,  2, 8, 9, 10,  9, 8, 7,  1,   7, 8, 9]


def test_price_structure_reads_an_uptrend():
    """Higher pivot high (10 -> 12) and higher pivot low (1 -> 2)."""
    assert price_structure(UPTREND_HIGHS, UPTREND_LOWS) == "uptrend"


def test_price_structure_reads_a_downtrend():
    """Lower pivot high (12 -> 10) and lower pivot low (2 -> 1)."""
    assert price_structure(DOWNTREND_HIGHS, DOWNTREND_LOWS) == "downtrend"


def test_price_structure_is_ranging_when_highs_and_lows_disagree():
    """Higher high but lower low is an expanding range, not a trend."""
    assert price_structure(UPTREND_HIGHS, DOWNTREND_LOWS) == "ranging"


def test_price_structure_needs_pivots_before_claiming_a_trend():
    assert price_structure([1, 2, 3], [1, 2, 3]) == "ranging"


def test_n_period_high_excludes_the_current_bar():
    """A bar must not help define the level it is tested against."""
    highs = [1, 2, 3, 4, 100]
    assert n_period_high(highs, 4)[4] == pytest.approx(4.0)


def test_n_period_low_excludes_the_current_bar():
    lows = [10, 9, 8, 7, 1]
    assert n_period_low(lows, 4)[4] == pytest.approx(7.0)


def test_trend_direction_on_stacked_emas():
    assert trend_direction([float(v) for v in range(1, 400)]) == "bullish"
    assert trend_direction([float(v) for v in range(400, 1, -1)]) == "bearish"


def test_trend_direction_is_neutral_without_enough_history():
    assert trend_direction([1.0, 2.0, 3.0]) == "neutral"


def test_support_and_resistance_return_recent_pivots():
    highs = [1, 2, 9, 2, 1, 2, 8, 2, 1]
    lows = [9, 8, 1, 8, 9, 8, 2, 8, 9]
    support, resistance = support_resistance(highs, lows, lookback=2)
    assert resistance and support
    assert max(resistance) <= max(highs)


# ---------------------------------------------------------------------------
# crossovers
# ---------------------------------------------------------------------------

def test_crossed_above_fires_only_on_the_crossing_bar():
    fast = [1.0, 2.0, 4.0, 5.0]
    slow = [3.0, 3.0, 3.0, 3.0]
    assert crossed_above(fast, slow, index=2)      # 2 -> 4 crosses 3
    assert not crossed_above(fast, slow, index=3)  # already above


def test_crossed_below_fires_only_on_the_crossing_bar():
    fast = [5.0, 4.0, 2.0, 1.0]
    slow = [3.0, 3.0, 3.0, 3.0]
    assert crossed_below(fast, slow, index=2)
    assert not crossed_below(fast, slow, index=3)


def test_crossovers_are_false_when_data_is_missing():
    assert not crossed_above([None, None], [None, None])
    assert not crossed_below([None, 1.0], [None, 2.0])


def test_crossovers_do_not_raise_on_short_series():
    assert not crossed_above([1.0], [2.0])
    assert not crossed_below([1.0], [2.0])
