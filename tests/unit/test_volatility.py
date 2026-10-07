"""Adversarial tests for the volatility features.

`atr_percentile` is read by the regime detector and by every setup's
volatility band, so an error here does not stay local: it relabels regimes
and re-gates every trade, and the backtest still runs. The tests are
organized around that risk:

* **Every estimator is checked against a hand-written formula on a
  5-bar tail.** The expectations are spelled out with `math.log` and
  `statistics.stdev` in the test body rather than by calling the module back
  -- a test that re-runs the implementation to produce its own expectation
  verifies only that the code is deterministic.
* **The two degenerate tapes are constructed explicitly.** A flat series and
  a collapsed series are where `0/0` and `log(0)` live, and `_vector` raises
  on a non-finite value, so these are the cases that decide whether the
  module can run at all.
* **The warmup boundary is checked at `warmup_bars - 1` and `warmup_bars`.**
  Off by one here produces a number from a half-filled window that looks
  exactly like a feature.
* **The lookahead audit runs with `factory=`**, which is the only form that
  catches a full-sample constant captured in `__init__`.

Builders are local on purpose (file ownership, and so a failure localizes
here rather than to a shared fixture). `bars_with_ranges` is the important
one: with `open == close == base` on every bar, the previous close is always
`base`, so `TR_i = max(r_i, r_i/2, r_i/2) = r_i` exactly. That makes the
whole ATR chain a function of a list of numbers the test chose, which is
what lets the percentile tests assert exact values instead of inequalities.
"""

from __future__ import annotations

import inspect
import math
import statistics
from datetime import date, datetime, timezone

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.config.schema import FeatureConfig
from flow_model.core.enums import DataQuality, Feed
from flow_model.data.market_view import MarketView
from flow_model.data.series import BarSeries, from_ns, to_ns
from flow_model.data.store import SymbolData
from flow_model.features.base import FeatureError
from flow_model.features.volatility import (
    FOUR_LN2,
    GK_BODY_COEFFICIENT,
    LOG_RATIO_LIMIT,
    RTH_SECONDS_PER_SESSION,
    TRADABLE_BAND_HIGH,
    TRADABLE_BAND_LOW,
    TRADING_DAYS_PER_YEAR,
    VOL_EXPANSION_SCALE,
    VolatilityFeatures,
    bars_per_year,
    signed_expansion,
    tradable_band_score,
    true_range,
    wilder_atr_series,
)
from flow_model.validation.lookahead import assert_no_lookahead, audit_computer

INTERVAL = 300
SYMBOL = "NQ"
BASE = 100.0

#: 09:35 New York on a Tuesday, as UTC. The wall-clock value is irrelevant --
#: every comparison in the firewall is on int64 ns -- but a plausible instant
#: keeps a failure message readable.
T0 = datetime(2024, 1, 2, 14, 35, tzinfo=timezone.utc)

#: sqrt(bars per year) at a 300s interval: 252 sessions x 23_400 RTH seconds
#: / 300 = 19_656 bars. Written out rather than imported so the annualization
#: assertions do not depend on the function they are checking.
ANNUALIZE_300 = math.sqrt(252.0 * 6.5 * 3600.0 / 300.0)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def ts_grid(n: int, interval: int = INTERVAL) -> np.ndarray:
    """`n` strictly ascending bar-close stamps, int64 UTC ns."""
    step = int(interval) * 1_000_000_000
    return np.array([to_ns(T0) + step * i for i in range(n)], dtype=np.int64)


def series_from_columns(
    opens, highs, lows, closes, interval: int = INTERVAL, symbol: str = SYMBOL
) -> BarSeries:
    o = np.asarray(opens, dtype=np.float64)
    return BarSeries(
        symbol=symbol,
        ts_ns=ts_grid(o.size, interval),
        interval_seconds=interval,
        columns={
            "open": o,
            "high": np.asarray(highs, dtype=np.float64),
            "low": np.asarray(lows, dtype=np.float64),
            "close": np.asarray(closes, dtype=np.float64),
            "volume": np.full(o.size, 1000.0),
        },
    )


def bars_with_ranges(ranges, base: float = BASE) -> BarSeries:
    """Bars whose true range is exactly `ranges[i]`.

    `open == close == base` on every bar, so the previous close is always
    `base` and `TR = max(r, r/2, r/2) = r`. The ATR chain is then a pure
    function of the numbers the caller passed.
    """
    r = np.asarray(ranges, dtype=np.float64)
    flat = np.full(r.size, float(base))
    return series_from_columns(flat, base + r / 2.0, base - r / 2.0, flat.copy())


def bars_from_log_returns(returns, base: float = BASE, wick: float = 0.0) -> BarSeries:
    """Bars whose close-to-close log returns are exactly `returns`.

    One more bar than returns. `wick` widens high/low beyond the body.
    """
    rets = np.asarray(returns, dtype=np.float64)
    closes = float(base) * np.exp(np.concatenate(([0.0], np.cumsum(rets))))
    opens = np.concatenate((closes[:1], closes[:-1]))
    return series_from_columns(
        opens,
        np.maximum(opens, closes) + wick,
        np.minimum(opens, closes) - wick,
        closes,
    )


def bars_with_tail(tail, filler: int, base: float = BASE) -> BarSeries:
    """`filler` flat bars at `base`, then the explicit `(o, h, l, c)` rows."""
    rows = [(base, base, base, base)] * int(filler) + [tuple(map(float, r)) for r in tail]
    cols = list(zip(*rows))
    return series_from_columns(cols[0], cols[1], cols[2], cols[3])


def symbol_data(series: BarSeries) -> SymbolData:
    return SymbolData(
        symbol=series.symbol,
        primary_interval=series.interval_seconds,
        bars={series.interval_seconds: series},
    )


def view_at(series: BarSeries, index: int = -1) -> MarketView:
    """A view whose cutoff is bar `index`'s close, so that bar is visible."""
    position = index if index >= 0 else len(series) + index
    ts = int(series.ts_ns[position])
    return MarketView(symbol_data(series), now=from_ns(ts), now_ns=ts)


def empty_view(interval: int = INTERVAL) -> MarketView:
    series = BarSeries.empty(SYMBOL, interval)
    return MarketView(symbol_data(series), now=T0, now_ns=to_ns(T0))


def small_config(**overrides) -> FeatureConfig:
    """Short windows so a hand-computed tail fits in a 13-bar series.

    `atr_percentile_lookback` has a `gt=10` bound in the schema, so 11 is the
    smallest legal value and the ATR chain is 2 + 11 = 13 bars.
    """
    fields = dict(
        atr_period=2,
        atr_percentile_lookback=11,
        realized_vol_period=5,
        vol_of_vol_period=2,
    )
    fields.update(overrides)
    return FeatureConfig(**fields)


def dispersed_ranges(n: int, seed: int = 11, low: float = 0.4, high: float = 2.0):
    """Deterministic, genuinely dispersed true ranges.

    A constant-range series makes every ATR identical, which turns a
    percentile into a degenerate 1.0 and lets a broken percentile pass.
    """
    return np.random.default_rng(seed).uniform(low, high, n)


@pytest.fixture(scope="module")
def synthetic_data() -> SymbolData:
    """~3 months of 5-minute NQ bars: comfortably over 2x the 266-bar warmup."""
    from flow_model.data import SyntheticConfig, SyntheticMarketGenerator, TradingCalendar

    config = load_config()
    dataset = SyntheticMarketGenerator(SyntheticConfig(), seed=7).generate(
        SYMBOL,
        config.spec(SYMBOL),
        date(2017, 1, 1),
        date(2017, 4, 1),
        INTERVAL,
        calendar=TradingCalendar(),
    )
    return symbol_data(dataset.bars)


# ---------------------------------------------------------------------------
# true range
# ---------------------------------------------------------------------------


def test_true_range_matches_the_three_way_max_computed_by_hand():
    highs = np.array([10.0, 12.0, 11.0, 20.0, 15.0])
    lows = np.array([9.0, 11.0, 8.0, 19.0, 6.0])
    closes = np.array([9.5, 11.5, 10.0, 19.5, 14.0])
    # bar 1: h-l=1.0, |12-9.5|=2.5, |11-9.5|=1.5          -> 2.5
    # bar 2: h-l=3.0, |11-11.5|=0.5, |8-11.5|=3.5         -> 3.5
    # bar 3: h-l=1.0, |20-10|=10.0, |19-10|=9.0           -> 10.0
    # bar 4: h-l=9.0, |15-19.5|=4.5, |6-19.5|=13.5        -> 13.5
    assert true_range(highs, lows, closes).tolist() == [2.5, 3.5, 10.0, 13.5]


def test_true_range_returns_one_fewer_value_than_bars():
    """The first bar has no previous close inside the window, so it is dropped."""
    series = bars_with_ranges(dispersed_ranges(40))
    ranges = true_range(
        np.asarray(series.col("high")),
        np.asarray(series.col("low")),
        np.asarray(series.col("close")),
    )
    assert ranges.size == len(series) - 1


def test_true_range_uses_the_previous_close_not_the_previous_low():
    """A gap down makes |low - prev_close| the binding term, not high - low."""
    highs = np.array([100.0, 90.0])
    lows = np.array([99.0, 89.0])
    closes = np.array([100.0, 89.5])
    assert true_range(highs, lows, closes).tolist() == [11.0]  # |89 - 100|


def test_true_range_of_a_single_bar_is_empty():
    one = np.array([1.0])
    assert true_range(one, one, one).size == 0


def test_true_range_rejects_arrays_of_different_lengths():
    with pytest.raises(FeatureError, match="equal-length"):
        true_range(np.zeros(5), np.zeros(4), np.zeros(5))


# ---------------------------------------------------------------------------
# Wilder's ATR
# ---------------------------------------------------------------------------


def test_wilder_seed_is_the_simple_mean_of_the_first_period_ranges():
    ranges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert wilder_atr_series(ranges, 3)[0] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)


def test_wilder_atr_matches_an_exact_worked_example():
    """Period 3 over TR = 1..5, worked out as exact fractions.

    seed   = (1 + 2 + 3) / 3            = 2
    next   = (2 * 2 + 4) / 3            = 8/3
    next   = (8/3 * 2 + 5) / 3          = 31/9
    """
    out = wilder_atr_series(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert out.tolist() == pytest.approx([2.0, 8.0 / 3.0, 31.0 / 9.0])


def test_wilder_atr_matches_the_increment_form_at_period_14():
    """Wilder's own phrasing, `ATR += (TR - ATR) / p`, is algebraically the
    same recursion and arithmetically a different one -- so it is a genuine
    independent check at the default period."""
    ranges = np.arange(1.0, 41.0)
    period = 14
    level = sum(ranges[:period]) / period
    expected = [level]
    for value in ranges[period:]:
        level = level + (value - level) / period
        expected.append(level)
    assert wilder_atr_series(ranges, period).tolist() == pytest.approx(expected)


def test_wilder_atr_series_length_is_ranges_minus_period_plus_one():
    for size, period in ((40, 14), (13, 2), (252 + 14 - 1, 14)):
        assert wilder_atr_series(np.ones(size), period).size == size - period + 1


def test_wilder_atr_of_constant_ranges_is_that_constant():
    out = wilder_atr_series(np.full(50, 3.25), 14)
    assert out.tolist() == pytest.approx([3.25] * 37)


def test_wilder_atr_rejects_a_window_shorter_than_its_period():
    with pytest.raises(FeatureError, match="needs at least"):
        wilder_atr_series(np.ones(13), 14)


def test_wilder_atr_rejects_a_non_positive_period():
    with pytest.raises(FeatureError, match="at least 1"):
        wilder_atr_series(np.ones(13), 0)


def test_atr_from_a_view_matches_an_independent_recursion_over_the_same_window():
    """End-to-end on the real accessor path, against the increment form."""
    config = FeatureConfig()
    computer = VolatilityFeatures(config)
    ranges = dispersed_ranges(computer.warmup_bars + 40, seed=5)
    series = bars_with_ranges(ranges)
    window = ranges[-computer.warmup_bars :]

    # TR_i == ranges[i] by construction; the first bar of the window is dropped.
    level = float(np.mean(window[1 : 1 + config.atr_period]))
    for value in window[1 + config.atr_period :]:
        level += (value - level) / config.atr_period

    assert computer.compute(view_at(series)).values["atr"] == pytest.approx(level)


# ---------------------------------------------------------------------------
# annualization
# ---------------------------------------------------------------------------


def test_bars_per_year_for_five_minute_bars_is_seventy_eight_sessions():
    assert bars_per_year(300) == pytest.approx(252.0 * 78.0)


def test_bars_per_year_for_hourly_bars_is_six_and_a_half_per_session():
    assert bars_per_year(3600) == pytest.approx(252.0 * 6.5)


def test_bars_per_year_for_daily_bars_is_exactly_the_session_count():
    assert bars_per_year(86_400) == pytest.approx(TRADING_DAYS_PER_YEAR)


def test_bars_per_year_for_weekly_bars_is_fifty_two_point_something():
    assert bars_per_year(7 * 86_400) == pytest.approx(36.0)


def test_bars_per_year_rejects_a_non_positive_interval():
    for bad in (0, -300):
        with pytest.raises(FeatureError, match="positive number of seconds"):
            bars_per_year(bad)


def test_the_two_annualization_branches_agree_at_the_daily_boundary():
    """A continuity check on the branch: the intraday formula at one session
    and the daily formula at one day must both describe 252 observations."""
    assert bars_per_year(int(RTH_SECONDS_PER_SESSION)) == pytest.approx(
        TRADING_DAYS_PER_YEAR
    )


# ---------------------------------------------------------------------------
# the estimators, against hand-computed tails
# ---------------------------------------------------------------------------


def test_realized_vol_matches_a_hand_computed_stdev_of_log_returns():
    returns = [0.01, -0.02, 0.015, -0.005, 0.0, 0.02, -0.01, 0.005, 0.03, -0.015, 0.01, -0.02]
    config = small_config()
    computer = VolatilityFeatures(config)
    series = bars_from_log_returns(returns)
    assert len(series) == computer.warmup_bars

    expected = statistics.stdev(returns[-config.realized_vol_period :]) * ANNUALIZE_300
    got = computer.compute(view_at(series)).values["realized_vol"]
    assert got == pytest.approx(expected, rel=1e-9)


def test_parkinson_matches_a_hand_computed_value_on_a_five_bar_tail():
    tail_ranges = [1.0, 2.0, 0.5, 4.0, 1.5]
    computer = VolatilityFeatures(small_config())
    series = bars_with_ranges([0.8] * 8 + tail_ranges)
    assert len(series) == computer.warmup_bars

    squares = [
        math.log((BASE + r / 2.0) / (BASE - r / 2.0)) ** 2 for r in tail_ranges
    ]
    expected = math.sqrt((sum(squares) / 5.0) / (4.0 * math.log(2.0))) * ANNUALIZE_300
    assert computer.compute(view_at(series)).values["parkinson_vol"] == pytest.approx(
        expected, rel=1e-12
    )


def test_garman_klass_matches_a_hand_computed_value_with_a_real_body():
    """The body term must actually be subtracted, so `open != close` here."""
    tail = [
        (100.0, 101.0, 99.5, 100.8),
        (100.8, 102.0, 100.0, 100.2),
        (100.2, 100.4, 99.0, 99.1),
        (99.1, 99.3, 97.5, 99.0),
        (99.0, 101.5, 98.9, 101.2),
    ]
    computer = VolatilityFeatures(small_config())
    series = bars_with_tail(tail, filler=8)
    assert len(series) == computer.warmup_bars

    terms = [
        0.5 * math.log(h / l) ** 2
        - (2.0 * math.log(2.0) - 1.0) * math.log(c / o) ** 2
        for o, h, l, c in tail
    ]
    expected = math.sqrt(sum(terms) / 5.0) * ANNUALIZE_300
    assert computer.compute(view_at(series)).values[
        "garman_klass_vol"
    ] == pytest.approx(expected, rel=1e-12)


def test_the_garman_klass_coefficients_are_the_published_ones():
    assert FOUR_LN2 == pytest.approx(2.772588722239781)
    assert GK_BODY_COEFFICIENT == pytest.approx(0.3862943611198906)


def test_with_no_body_garman_klass_is_parkinson_times_sqrt_two_ln_two():
    """`open == close` kills the body term, leaving `sqrt(0.5 * mean(hl^2))`
    against Parkinson's `sqrt(mean(hl^2) / 4 ln 2)`. The ratio is then the
    exact constant `sqrt(2 ln 2)`, independent of the data."""
    computer = VolatilityFeatures(small_config())
    values = computer.compute(view_at(bars_with_ranges(dispersed_ranges(13, seed=3)))).values
    assert values["garman_klass_vol"] / values["parkinson_vol"] == pytest.approx(
        math.sqrt(2.0 * math.log(2.0))
    )


def test_vol_of_vol_matches_a_hand_computed_coefficient_of_variation():
    returns = [0.01, -0.02, 0.015, -0.005, 0.0, 0.02, -0.01, 0.005, 0.03, -0.015, 0.01, -0.02]
    config = small_config()
    computer = VolatilityFeatures(config)

    r, v = config.realized_vol_period, config.vol_of_vol_period
    tail = returns[-(r + v - 1) :]
    levels = [statistics.stdev(tail[i : i + r]) for i in range(v)]
    expected = statistics.stdev(levels) / statistics.mean(levels)

    series = bars_from_log_returns(returns)
    assert len(series) == computer.warmup_bars
    vector = computer.compute(view_at(series))
    assert vector.warmup_complete
    assert vector.values["vol_of_vol"] == pytest.approx(expected, rel=1e-9)


def test_vol_of_vol_is_zero_when_the_realized_vol_windows_coincide():
    """With `vol_of_vol_period = 2` the two windows are `r[0:5]` and `r[1:6]`;
    making `r[5] == r[0]` makes them the same multiset, so the dispersion of
    the realized-vol series is exactly zero and the CV with it."""
    returns = [0.03, -0.01, 0.02, 0.004, -0.02, 0.012] + [
        0.01, -0.02, 0.015, -0.005, 0.02, 0.01
    ]
    computer = VolatilityFeatures(small_config())
    series = bars_from_log_returns(returns)
    assert len(series) == computer.warmup_bars
    vector = computer.compute(view_at(series))
    assert vector.warmup_complete
    assert vector.values["vol_of_vol"] == pytest.approx(0.0, abs=1e-12)
    assert vector.values["realized_vol"] > 0.0


def test_vol_of_vol_is_unchanged_by_the_annualization_factor():
    """It is a coefficient of variation, so the common sqrt(bars_per_year)
    cancels -- the same bars at a different declared interval must give the
    same number."""
    returns = list(dispersed_ranges(12, seed=21, low=-0.02, high=0.03))
    computer = VolatilityFeatures(small_config())
    five_minute = bars_from_log_returns(returns)
    hourly = series_from_columns(
        five_minute.col("open"),
        five_minute.col("high"),
        five_minute.col("low"),
        five_minute.col("close"),
        interval=3600,
    )
    assert len(five_minute) == computer.warmup_bars
    a = computer.compute(view_at(five_minute)).values
    b = computer.compute(view_at(hourly)).values
    assert b["vol_of_vol"] == pytest.approx(a["vol_of_vol"])
    assert b["realized_vol"] != pytest.approx(a["realized_vol"])


def test_atr_pct_is_the_atr_divided_by_the_last_close():
    computer = VolatilityFeatures(small_config())
    series = bars_with_ranges(dispersed_ranges(13, seed=4))
    values = computer.compute(view_at(series)).values
    assert values["atr_pct"] == pytest.approx(values["atr"] / BASE)


# ---------------------------------------------------------------------------
# the ATR percentile
# ---------------------------------------------------------------------------


def test_atr_percentile_is_in_the_unit_interval_with_a_floor_of_one_over_lookback():
    """The current ATR is a member of its own comparison window, so the
    lowest reachable rank is 1 / lookback, not 0."""
    config = FeatureConfig()
    computer = VolatilityFeatures(config)
    ranges = np.concatenate((dispersed_ranges(computer.warmup_bars, seed=6), [1e-6] * 60))
    percentile = computer.compute(view_at(bars_with_ranges(ranges))).values[
        "atr_percentile"
    ]
    assert percentile == pytest.approx(1.0 / config.atr_percentile_lookback)
    assert 0.0 <= percentile <= 1.0


def test_a_new_atr_high_gives_a_percentile_of_exactly_one():
    computer = VolatilityFeatures(FeatureConfig())
    ranges = np.concatenate((dispersed_ranges(computer.warmup_bars, seed=6), [40.0] * 60))
    values = computer.compute(view_at(bars_with_ranges(ranges))).values
    assert values["atr_percentile"] == 1.0


def test_an_inflated_recent_range_raises_the_percentile():
    """The same history, with only the last ten bars' range quadrupled."""
    computer = VolatilityFeatures(FeatureConfig())
    ranges = dispersed_ranges(computer.warmup_bars + 50, seed=8)
    inflated = ranges.copy()
    inflated[-10:] *= 4.0

    calm = computer.compute(view_at(bars_with_ranges(ranges))).values
    loud = computer.compute(view_at(bars_with_ranges(inflated))).values
    assert loud["atr"] > calm["atr"]
    assert loud["atr_percentile"] > calm["atr_percentile"]


def test_the_percentile_rises_monotonically_through_a_range_expansion():
    computer = VolatilityFeatures(FeatureConfig())
    n = computer.warmup_bars
    ranges = np.concatenate(
        (dispersed_ranges(n, seed=9), np.linspace(2.0, 12.0, 30))
    )
    series = bars_with_ranges(ranges)
    percentiles = [
        computer.compute(view_at(series, i)).values["atr_percentile"]
        for i in range(n - 1, len(series))
    ]
    assert all(b >= a for a, b in zip(percentiles, percentiles[1:])), percentiles
    assert percentiles[0] < percentiles[-1] == 1.0


def test_the_percentile_spans_its_range_on_real_synthetic_data(synthetic_data):
    """A feature pinned at one value would pass every bound check above."""
    computer = VolatilityFeatures(FeatureConfig())
    bars = synthetic_data.primary_bars
    sampled = []
    for i in range(computer.warmup_bars - 1, len(bars), 23):
        ts = int(bars.ts_ns[i])
        sampled.append(
            computer.compute(
                MarketView(synthetic_data, now=from_ns(ts), now_ns=ts)
            ).values["atr_percentile"]
        )
    assert min(sampled) < 0.15 and max(sampled) > 0.85
    assert len(set(sampled)) > 20


# ---------------------------------------------------------------------------
# vol_expansion
# ---------------------------------------------------------------------------


def test_vol_expansion_is_positive_when_the_atr_exceeds_its_trailing_median():
    computer = VolatilityFeatures(FeatureConfig())
    ranges = np.concatenate(
        (dispersed_ranges(computer.warmup_bars, seed=12), [8.0] * 40)
    )
    assert computer.compute(view_at(bars_with_ranges(ranges))).values[
        "vol_expansion"
    ] > 0.5


def test_vol_expansion_is_negative_when_the_atr_falls_below_its_median():
    computer = VolatilityFeatures(FeatureConfig())
    ranges = np.concatenate(
        (dispersed_ranges(computer.warmup_bars, seed=12), [0.02] * 40)
    )
    assert computer.compute(view_at(bars_with_ranges(ranges))).values[
        "vol_expansion"
    ] < -0.5


def test_vol_expansion_is_zero_when_every_atr_in_the_window_is_identical():
    """A constant true range makes the ATR equal to its own median, so the log
    ratio is exactly zero -- and the sign reapplied to `squash` must not turn
    that into a spurious positive."""
    computer = VolatilityFeatures(FeatureConfig())
    values = computer.compute(
        view_at(bars_with_ranges(np.full(computer.warmup_bars, 1.75)))
    ).values
    assert values["vol_expansion"] == 0.0


def test_vol_expansion_stays_inside_minus_one_to_one_under_an_extreme_jump():
    computer = VolatilityFeatures(FeatureConfig())
    ranges = np.concatenate(
        (np.full(computer.warmup_bars, 0.01), np.full(40, 150.0))
    )
    value = computer.compute(view_at(bars_with_ranges(ranges, base=1000.0))).values[
        "vol_expansion"
    ]
    assert 0.0 < value < 1.0


def test_the_vol_expansion_scale_maps_one_doubling_to_tanh_of_one():
    """The documented meaning of `VOL_EXPANSION_SCALE = ln 2`."""
    assert VOL_EXPANSION_SCALE == pytest.approx(math.log(2.0))
    assert signed_expansion(2.0, 1.0) == pytest.approx(math.tanh(1.0))
    assert signed_expansion(1.0, 2.0) == pytest.approx(-math.tanh(1.0))


def test_signed_expansion_is_exactly_zero_at_parity():
    assert signed_expansion(3.5, 3.5) == 0.0


def test_signed_expansion_takes_the_sign_of_the_log_ratio():
    """`squash` takes `abs`, so the sign is reapplied by hand -- and if that
    step were dropped a collapse in range would read as an expansion."""
    assert signed_expansion(4.0, 1.0) > 0.0
    assert signed_expansion(1.0, 4.0) < 0.0
    assert signed_expansion(4.0, 1.0) == pytest.approx(-signed_expansion(1.0, 4.0))


def test_signed_expansion_returns_zero_when_there_is_no_trailing_reference():
    """An all-flat trailing window has no median to expand from. Through
    `compute` this coincides with `atr == 0`, so 0/0, and neither expansion
    nor compression is the honest reading."""
    assert signed_expansion(0.0, 0.0) == 0.0
    assert signed_expansion(1.0, 0.0) == 0.0
    assert signed_expansion(1.0, -1.0) == 0.0


def test_signed_expansion_saturates_negative_on_a_total_collapse():
    """`log(0)` is -inf and `squash` answers 0.0 for a non-finite argument --
    "no change in volatility" for a tape that stopped moving. The clamp
    upstream is what makes this -1.0 instead."""
    assert signed_expansion(0.0, 2.0) == pytest.approx(-1.0)


def test_signed_expansion_rejects_non_finite_inputs_without_propagating_them():
    for bad in (math.nan, math.inf, -math.inf):
        assert signed_expansion(bad, 1.0) == 0.0
        assert signed_expansion(1.0, bad) == 0.0


def test_signed_expansion_is_bounded_across_an_extreme_sweep():
    for exponent in range(-30, 31):
        value = signed_expansion(10.0 ** exponent, 1.0)
        assert -1.0 <= value <= 1.0
        assert math.isfinite(value)


def test_the_log_ratio_limit_is_invisible_at_double_precision():
    """The clamp must not be a tunable: at this bound `tanh` is already 1.0
    in float, so clamping cannot change a value the formula could reach."""
    assert LOG_RATIO_LIMIT == pytest.approx(20.0 * VOL_EXPANSION_SCALE)
    assert math.tanh(LOG_RATIO_LIMIT / VOL_EXPANSION_SCALE) == 1.0


# ---------------------------------------------------------------------------
# vol_regime_score
# ---------------------------------------------------------------------------


def test_vol_regime_score_is_one_across_the_whole_tradable_band():
    for p in np.linspace(TRADABLE_BAND_LOW, TRADABLE_BAND_HIGH, 25):
        assert tradable_band_score(float(p)) == pytest.approx(1.0)


def test_vol_regime_score_is_zero_at_both_extremes():
    assert tradable_band_score(0.0) == 0.0
    assert tradable_band_score(1.0) == 0.0


def test_vol_regime_score_is_bounded_and_symmetric_about_the_midpoint():
    for p in np.linspace(0.0, 1.0, 201):
        score = tradable_band_score(float(p))
        assert 0.0 <= score <= 1.0
        assert tradable_band_score(1.0 - float(p)) == pytest.approx(score)


def test_vol_regime_score_falls_off_monotonically_into_each_tail():
    lower = [tradable_band_score(p) for p in np.linspace(0.0, TRADABLE_BAND_LOW, 40)]
    upper = [tradable_band_score(p) for p in np.linspace(TRADABLE_BAND_HIGH, 1.0, 40)]
    assert all(b >= a for a, b in zip(lower, lower[1:]))
    assert all(b <= a for a, b in zip(upper, upper[1:]))


def test_vol_regime_score_clamps_an_out_of_range_percentile():
    assert tradable_band_score(-0.5) == 0.0
    assert tradable_band_score(1.5) == 0.0


def test_vol_regime_score_is_low_on_a_dead_tape():
    """Normal ranges, then a 60-bar collapse: the ATR decays to the bottom of
    its own trailing distribution and the score must follow it down."""
    computer = VolatilityFeatures(FeatureConfig())
    ranges = np.concatenate(
        (dispersed_ranges(computer.warmup_bars, seed=14), np.full(60, 1e-4))
    )
    values = computer.compute(view_at(bars_with_ranges(ranges))).values
    assert values["atr_percentile"] < 0.05
    assert values["vol_regime_score"] < 0.05


def test_vol_regime_score_is_low_in_a_panic():
    computer = VolatilityFeatures(FeatureConfig())
    ranges = np.concatenate(
        (dispersed_ranges(computer.warmup_bars, seed=14), np.full(60, 25.0))
    )
    values = computer.compute(view_at(bars_with_ranges(ranges))).values
    assert values["atr_percentile"] > 0.95
    assert values["vol_regime_score"] < 0.05


def test_vol_regime_score_is_high_in_the_middle_of_a_normal_tape(synthetic_data):
    """The band must actually be reachable on real data, or the gate is a veto."""
    computer = VolatilityFeatures(FeatureConfig())
    bars = synthetic_data.primary_bars
    scores = []
    for i in range(computer.warmup_bars - 1, len(bars), 23):
        ts = int(bars.ts_ns[i])
        scores.append(
            computer.compute(
                MarketView(synthetic_data, now=from_ns(ts), now_ns=ts)
            ).values["vol_regime_score"]
        )
    assert max(scores) == pytest.approx(1.0)
    assert float(np.mean([s > 0.5 for s in scores])) > 0.4


def test_the_band_constants_mirror_the_regime_classifier_defaults():
    """If these drift apart, `vol_regime_score` penalizes a band the regime
    detector does not call extreme, and neither side reports the mismatch."""
    regime = load_config().regime
    assert TRADABLE_BAND_LOW == regime.low_vol_percentile
    assert TRADABLE_BAND_HIGH == regime.high_vol_percentile


# ---------------------------------------------------------------------------
# degenerate tapes
# ---------------------------------------------------------------------------


def test_a_perfectly_flat_series_gives_zero_vol_and_every_value_finite():
    computer = VolatilityFeatures(FeatureConfig())
    flat = np.full(computer.warmup_bars + 20, BASE)
    series = series_from_columns(flat, flat.copy(), flat.copy(), flat.copy())
    vector = computer.compute(view_at(series))

    assert vector.warmup_complete
    assert set(vector.values) == set(computer.keys)
    assert all(math.isfinite(v) for v in vector.values.values())
    for key in (
        "atr", "atr_pct", "realized_vol", "parkinson_vol",
        "garman_klass_vol", "vol_of_vol", "vol_expansion",
    ):
        assert vector.values[key] == 0.0, key


def test_a_perfectly_flat_series_pins_the_percentile_to_the_floor_not_the_ceiling():
    """`fraction at or below` would rank a zero ATR at 1.0 among other zeros,
    and the regime detector would read a halted tape as HIGH_VOL."""
    computer = VolatilityFeatures(FeatureConfig())
    flat = np.full(computer.warmup_bars, BASE)
    vector = computer.compute(
        view_at(series_from_columns(flat, flat.copy(), flat.copy(), flat.copy()))
    )
    assert vector.values["atr_percentile"] == 0.0
    assert vector.values["vol_regime_score"] == 0.0
    assert any("pinned to 0.0" in note for note in vector.notes), vector.notes


def test_a_series_with_one_flat_bar_is_otherwise_unaffected():
    """A single `high == low` bar is a legitimate print, not a divide by zero."""
    computer = VolatilityFeatures(small_config())
    ranges = dispersed_ranges(13, seed=16)
    ranges[-2] = 0.0
    vector = computer.compute(view_at(bars_with_ranges(ranges)))
    assert vector.warmup_complete
    assert all(math.isfinite(v) for v in vector.values.values())
    assert vector.values["parkinson_vol"] > 0.0


def test_garman_klass_is_non_negative_on_every_valid_bar(synthetic_data):
    """OHLC ordering gives ln(h/l) >= |ln(c/o)| and 0.5 > 2 ln 2 - 1, so the
    per-bar term cannot be negative -- asserted on real bars, not argued."""
    computer = VolatilityFeatures(FeatureConfig())
    bars = synthetic_data.primary_bars
    o, h, l, c = (np.asarray(bars.col(k)) for k in ("open", "high", "low", "close"))
    terms = 0.5 * np.log(h / l) ** 2 - GK_BODY_COEFFICIENT * np.log(c / o) ** 2
    assert float(terms.min()) >= 0.0
    for i in range(computer.warmup_bars - 1, len(bars), 97):
        ts = int(bars.ts_ns[i])
        vector = computer.compute(MarketView(synthetic_data, now=from_ns(ts), now_ns=ts))
        assert vector.values["garman_klass_vol"] > 0.0
        assert not any("Garman-Klass" in note for note in vector.notes)


def test_a_non_positive_low_is_reported_not_ready_rather_than_logged():
    """`BarSeries` validates OHLC ordering and finiteness but not positivity,
    so a zero price can reach this computer. Every estimator here is a log of
    a price ratio; emitting a fabricated zero would read as 'no volatility'."""
    computer = VolatilityFeatures(small_config())
    ranges = dispersed_ranges(13, seed=17)
    series = bars_with_ranges(ranges)
    lows = np.asarray(series.col("low")).copy()
    lows[-3] = 0.0
    broken = series_from_columns(
        series.col("open"), series.col("high"), lows, series.col("close")
    )
    vector = computer.compute(view_at(broken))
    assert not vector.warmup_complete
    assert any("non-positive" in note for note in vector.notes), vector.notes


# ---------------------------------------------------------------------------
# warmup, readiness and the declared contract
# ---------------------------------------------------------------------------


def test_warmup_bars_covers_the_whole_atr_chain():
    config = FeatureConfig()
    computer = VolatilityFeatures(config)
    assert computer.warmup_bars == config.atr_period + config.atr_percentile_lookback
    assert computer.warmup_bars == 266


def test_warmup_bars_covers_the_vol_of_vol_chain_when_that_one_is_longer():
    config = FeatureConfig(
        atr_period=2,
        atr_percentile_lookback=11,
        realized_vol_period=100,
        vol_of_vol_period=80,
    )
    assert VolatilityFeatures(config).warmup_bars == 180


def test_warmup_bars_exceeds_the_figure_the_config_computes_for_itself():
    """`FeatureConfig.warmup_bars` is `atr_percentile_lookback + 1` = 253 and
    omits the `atr_period` the ATR chain consumes before its first value
    exists. Declaring the true number here is what keeps the bundle honest;
    the discrepancy is recorded rather than matched."""
    config = FeatureConfig()
    assert VolatilityFeatures(config).warmup_bars > config.warmup_bars
    assert VolatilityFeatures(config).warmup_bars - config.warmup_bars == 13


def test_not_ready_one_bar_before_warmup_and_ready_exactly_at_warmup():
    computer = VolatilityFeatures(small_config())
    n = computer.warmup_bars
    series = bars_with_ranges(dispersed_ranges(n + 5, seed=18))

    short = computer.compute(view_at(series, n - 2))
    assert not short.warmup_complete
    assert "warmup incomplete" in short.notes[0]
    assert f"{n - 1} of {n}" in short.notes[0]

    exact = computer.compute(view_at(series, n - 1))
    assert exact.warmup_complete
    assert exact.values["atr"] > 0.0


def test_an_empty_view_is_not_ready_and_reports_the_absent_feed():
    computer = VolatilityFeatures(small_config())
    vector = computer.compute(empty_view())
    assert not vector.warmup_complete
    assert "bars feed absent" in vector.notes[0]


def test_a_single_bar_view_is_not_ready():
    computer = VolatilityFeatures(small_config())
    vector = computer.compute(view_at(bars_with_ranges([1.0]), 0))
    assert not vector.warmup_complete
    assert "1 of 13" in vector.notes[0]


def test_a_not_ready_vector_carries_zeros_and_missing_quality_for_every_key():
    computer = VolatilityFeatures(small_config())
    vector = computer.compute(empty_view())
    assert set(vector.values) == set(computer.keys)
    assert set(vector.values.values()) == {0.0}
    assert vector.quality == DataQuality.MISSING


def test_the_emitted_keys_are_exactly_the_declared_keys():
    computer = VolatilityFeatures(small_config())
    vector = computer.compute(view_at(bars_with_ranges(dispersed_ranges(13, seed=19))))
    assert tuple(sorted(vector.values)) == tuple(sorted(computer.keys))
    assert len(computer.keys) == len(set(computer.keys)) == 9


def test_every_emitted_value_carries_a_quality_grade():
    computer = VolatilityFeatures(small_config())
    vector = computer.compute(view_at(bars_with_ranges(dispersed_ranges(13, seed=20))))
    assert set(vector.quality_by_key) == set(computer.keys)
    assert vector.quality == DataQuality.GOOD


def test_required_feeds_is_bars_only():
    computer = VolatilityFeatures(small_config())
    assert computer.required_feeds == frozenset({Feed.BARS})
    assert computer.optional_feeds == frozenset()


def test_the_constructor_accepts_config_and_nothing_else():
    """The one leak the audit harness cannot see through is a full-sample
    constant captured at construction, so market data is banned at the
    constructor rather than audited for."""
    parameters = [
        (name, param)
        for name, param in inspect.signature(VolatilityFeatures.__init__).parameters.items()
        if name != "self"
    ]
    assert [name for name, _ in parameters] == ["config"]
    annotation = parameters[0][1].annotation
    assert annotation in (FeatureConfig, "FeatureConfig"), annotation

    banned = (
        "SymbolData", "BarSeries", "ColumnSeries", "DataStore", "MarketView",
        "SyntheticDataset", "ndarray", "DataFrame", "Series",
    )
    rendered = str(inspect.signature(VolatilityFeatures.__init__))
    for name in banned:
        assert name not in rendered, f"{name} must not appear in the constructor"


def test_the_computer_reads_a_fixed_window_not_all_available_history():
    """A window that grew with the dataset would make the same bar produce
    different features in bar 300 of a backtest and in bar 30_000."""
    computer = VolatilityFeatures(FeatureConfig())
    n = computer.warmup_bars
    tail = dispersed_ranges(n, seed=22)
    short = bars_with_ranges(tail)
    long = bars_with_ranges(np.concatenate((dispersed_ranges(400, seed=23), tail)))
    assert computer.compute(view_at(short)).values == pytest.approx(
        computer.compute(view_at(long)).values
    )


# ---------------------------------------------------------------------------
# lookahead, determinism, and a long run
# ---------------------------------------------------------------------------


def test_the_lookahead_audit_passes_with_a_factory(synthetic_data):
    """`factory=` rather than an instance: that is the form that rebuilds the
    computer against each altered dataset and so catches a constant captured
    in `__init__`."""
    config = load_config().features
    computer = VolatilityFeatures(config)
    assert len(synthetic_data.primary_bars) >= 2 * computer.warmup_bars

    result = audit_computer(
        data=synthetic_data,
        factory=lambda d: VolatilityFeatures(config),
        sample=40,
    )
    assert_no_lookahead([result])
    assert result.bars_checked >= 20
    assert set(result.keys_checked) == set(computer.keys)


def test_the_lookahead_audit_also_passes_with_short_windows(synthetic_data):
    """A second configuration, because the window lengths are what decide
    which bars are pre-warmup and therefore which checks actually run."""
    config = small_config()
    result = audit_computer(
        data=synthetic_data, factory=lambda d: VolatilityFeatures(config), sample=25
    )
    assert_no_lookahead([result])


def test_two_computations_on_the_identical_view_agree_exactly():
    computer = VolatilityFeatures(FeatureConfig())
    view = view_at(bars_with_ranges(dispersed_ranges(computer.warmup_bars + 30, seed=24)))
    first = computer.compute(view)
    second = computer.compute(view)
    assert first.values == second.values
    assert first.notes == second.notes


def test_two_independently_constructed_computers_agree_bit_for_bit():
    """No state survives a call, so a fresh computer must give the same answer."""
    config = FeatureConfig()
    series = bars_with_ranges(dispersed_ranges(config.atr_period + 320, seed=25))
    a = VolatilityFeatures(config).compute(view_at(series)).values
    b = VolatilityFeatures(config).compute(view_at(series)).values
    assert a == b


def test_every_value_is_finite_across_a_long_synthetic_run(synthetic_data):
    computer = VolatilityFeatures(FeatureConfig())
    bars = synthetic_data.primary_bars
    scored = 0
    for i in range(computer.warmup_bars - 1, len(bars), 11):
        ts = int(bars.ts_ns[i])
        vector = computer.compute(MarketView(synthetic_data, now=from_ns(ts), now_ns=ts))
        assert vector.warmup_complete
        scored += 1
        for key, value in vector.values.items():
            assert math.isfinite(value), f"{key} is {value} at bar {i}"
        assert 0.0 <= vector.values["atr_percentile"] <= 1.0
        assert 0.0 <= vector.values["vol_regime_score"] <= 1.0
        assert -1.0 <= vector.values["vol_expansion"] <= 1.0
        assert vector.values["atr"] > 0.0
        assert vector.values["realized_vol"] > 0.0
    assert scored > 150


def test_the_three_estimators_stay_within_one_order_of_magnitude(synthetic_data):
    """A unit or annualization error in one of the three shows up as a ratio
    in the hundreds. They are deliberately NOT asserted equal: the generator
    extends each bar's high/low past the body by `bar_range_factor` x the
    per-bar sigma, which inflates the range estimators relative to
    close-to-close by construction -- measured ratio ~1.9x for Parkinson."""
    computer = VolatilityFeatures(FeatureConfig())
    bars = synthetic_data.primary_bars
    ratios = []
    for i in range(computer.warmup_bars - 1, len(bars), 29):
        ts = int(bars.ts_ns[i])
        values = computer.compute(
            MarketView(synthetic_data, now=from_ns(ts), now_ns=ts)
        ).values
        ratios.append(values["parkinson_vol"] / values["realized_vol"])
        ratios.append(values["garman_klass_vol"] / values["realized_vol"])
    assert 0.1 < min(ratios) and max(ratios) < 10.0
