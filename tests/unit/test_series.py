"""Columnar series storage: the invariants everything above it relies on."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.data.base import MonotonicityError, SchemaError
from flow_model.data.series import (
    NS_PER_SECOND,
    BarSeries,
    ColumnSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    from_ns,
    to_ns,
    to_ns_array,
)

TS0 = datetime(2020, 1, 2, 14, 30, tzinfo=timezone.utc)


def _bar_cols(n, base=18000.0):
    close = np.linspace(base, base + 10 * (n - 1), n)
    return {
        "open": close - 5.0,
        "high": close + 10.0,
        "low": close - 10.0,
        "close": close,
        "volume": np.full(n, 1000.0),
    }


def make_bars(n=10, interval=300, symbol="NQ", start=TS0):
    ts = to_ns_array(start + timedelta(seconds=interval * (i + 1)) for i in range(n))
    return BarSeries(symbol=symbol, ts_ns=ts, interval_seconds=interval, columns=_bar_cols(n))


# --- timestamp handling ----------------------------------------------------


def test_naive_datetime_is_rejected():
    """Assuming UTC for a naive exchange timestamp is how a six-hour
    session-boundary error enters a backtest unnoticed."""
    with pytest.raises(SchemaError, match="naive datetime"):
        to_ns(datetime(2020, 1, 1, 9, 30))


def test_non_datetime_is_rejected():
    with pytest.raises(SchemaError, match="expected datetime"):
        to_ns("2020-01-01")


def test_ns_round_trip():
    assert from_ns(to_ns(TS0)) == TS0


def test_ns_round_trip_with_microseconds():
    ts = TS0.replace(microsecond=123456)
    assert from_ns(to_ns(ts)) == ts


def test_non_utc_aware_timestamps_normalize():
    from zoneinfo import ZoneInfo

    eastern = TS0.astimezone(ZoneInfo("America/New_York"))
    assert to_ns(eastern) == to_ns(TS0)


def test_to_ns_array():
    arr = to_ns_array([TS0, TS0 + timedelta(seconds=300)])
    assert arr.dtype == np.int64
    assert arr[1] - arr[0] == 300 * NS_PER_SECOND


# --- construction invariants ----------------------------------------------


def test_strictly_ascending_required():
    with pytest.raises(MonotonicityError, match="strictly ascending"):
        BarSeries(symbol="NQ", ts_ns=np.array([3, 1, 2], dtype=np.int64),
                  interval_seconds=300, columns=_bar_cols(3))


def test_duplicate_timestamps_rejected():
    """Duplicates make searchsorted ambiguous and hide corrected revisions."""
    with pytest.raises(MonotonicityError):
        BarSeries(symbol="NQ", ts_ns=np.array([1, 1, 2], dtype=np.int64),
                  interval_seconds=300, columns=_bar_cols(3))


def test_short_column_rejected():
    cols = _bar_cols(3)
    cols["open"] = cols["open"][:2]
    with pytest.raises(SchemaError, match="would read as zeros"):
        BarSeries(symbol="NQ", ts_ns=np.array([1, 2, 3], dtype=np.int64),
                  interval_seconds=300, columns=cols)


def test_missing_required_column_rejected():
    cols = _bar_cols(3)
    del cols["volume"]
    with pytest.raises(SchemaError, match="missing required columns"):
        BarSeries(symbol="NQ", ts_ns=np.array([1, 2, 3], dtype=np.int64),
                  interval_seconds=300, columns=cols)


def test_unknown_column_rejected():
    """A typo'd column name must not be silently accepted and then ignored."""
    cols = _bar_cols(2) | {"clsoe": np.ones(2)}
    with pytest.raises(SchemaError, match="unknown column"):
        BarSeries(symbol="NQ", ts_ns=np.array([1, 2], dtype=np.int64),
                  interval_seconds=300, columns=cols)


@pytest.mark.parametrize(
    "o,h,l,c",
    [
        (5.0, 2.0, 0.0, 1.0),    # open above high
        (1.0, 2.0, 0.0, 9.0),    # close above high
        (1.0, 0.5, 2.0, 1.0),    # high below low
        (1.0, 2.0, 1.5, 1.8),    # open below low
    ],
)
def test_ohlc_ordering_enforced(o, h, l, c):
    with pytest.raises(SchemaError, match="OHLC ordering violated"):
        BarSeries(symbol="NQ", ts_ns=np.array([1, 2], dtype=np.int64), interval_seconds=300,
                  columns={"open": [o, 1], "high": [h, 2], "low": [l, 0],
                           "close": [c, 1], "volume": [1, 1]})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_prices_rejected(bad):
    with pytest.raises(SchemaError, match="non-finite"):
        BarSeries(symbol="NQ", ts_ns=np.array([1, 2], dtype=np.int64), interval_seconds=300,
                  columns={"open": [1, 1], "high": [bad, 2], "low": [0, 0],
                           "close": [1, 1], "volume": [1, 1]})


def test_negative_volume_rejected():
    with pytest.raises(SchemaError, match="negative values"):
        BarSeries(symbol="NQ", ts_ns=np.array([1, 2], dtype=np.int64), interval_seconds=300,
                  columns={"open": [1, 1], "high": [2, 2], "low": [0, 0],
                           "close": [1, 1], "volume": [-1, 1]})


def test_interval_seconds_required():
    with pytest.raises(SchemaError, match="interval_seconds"):
        BarSeries(symbol="NQ", ts_ns=np.array([1], dtype=np.int64), columns=_bar_cols(1))


def test_single_row_and_empty_series_are_valid():
    assert len(make_bars(1)) == 1
    empty = BarSeries.empty("NQ", 300)
    assert len(empty) == 0 and not empty
    assert empty.first_ts is None and empty.last_ts is None


# --- immutability ----------------------------------------------------------


def test_columns_are_read_only():
    """A feature must not be able to corrupt the store in place."""
    series = make_bars(5)
    with pytest.raises(ValueError):
        series.col("close")[0] = 0.0
    with pytest.raises(ValueError):
        series.window("close", 0, 3)[0] = 0.0
    with pytest.raises(ValueError):
        series.ts_window(0, 3)[0] = 0


def test_copy_of_a_window_is_writable():
    assert make_bars(5).window("close", 0, 3).copy().flags.writeable


def test_constructor_copies_are_independent_of_caller_arrays():
    cols = _bar_cols(4)
    series = BarSeries(symbol="NQ", ts_ns=to_ns_array(
        TS0 + timedelta(seconds=300 * (i + 1)) for i in range(4)),
        interval_seconds=300, columns=cols)
    before = series.col("close")[0]
    cols["close"][0] = 99999.0
    assert series.col("close")[0] == before


# --- cutoff resolution ----------------------------------------------------


def test_observation_at_the_cutoff_is_visible():
    """At the instant a bar closes, it is known."""
    series = make_bars(10)
    fifth_close = TS0 + timedelta(seconds=300 * 5)
    assert series.visible_count(to_ns(fifth_close)) == 5


def test_observation_one_nanosecond_after_the_cutoff_is_hidden():
    series = make_bars(10)
    fifth_close_ns = to_ns(TS0 + timedelta(seconds=300 * 5))
    assert series.visible_count(fifth_close_ns - 1) == 4


def test_visible_count_before_all_data_is_zero():
    assert make_bars(10).visible_count(to_ns(TS0 - timedelta(days=1))) == 0


def test_visible_count_after_all_data_is_full_length():
    series = make_bars(10)
    assert series.visible_count(to_ns(TS0 + timedelta(days=1))) == len(series)


def test_index_at_or_before():
    series = make_bars(10)
    assert series.index_at_or_before(to_ns(TS0 + timedelta(seconds=1500))) == 4
    assert series.index_at_or_before(to_ns(TS0)) == -1


# --- window clamping: the wraparound guard --------------------------------


def test_window_clamps_negative_start_instead_of_wrapping():
    """NumPy would read arr[-5:3] as the TAIL of the array -- i.e. future
    data. Clamping is a correctness requirement, not defensiveness."""
    series = make_bars(100)
    window = series.window("close", -5, 3)
    assert window.tolist() == series.col("close")[:3].tolist()
    assert len(window) == 3


def test_window_clamps_end_beyond_length():
    series = make_bars(10)
    assert len(series.window("close", 5, 999)) == 5


def test_window_with_inverted_bounds_is_empty():
    assert len(make_bars(10).window("close", 7, 3)) == 0


def test_ts_window_clamps_identically():
    series = make_bars(100)
    assert series.ts_window(-5, 3).tolist() == series.ts_ns[:3].tolist()


# --- slicing --------------------------------------------------------------


def test_slice_rows_preserves_type_and_meta():
    series = make_bars(20)
    sliced = series.slice_rows(5, 10)
    assert isinstance(sliced, BarSeries)
    assert len(sliced) == 5
    assert sliced.interval_seconds == series.interval_seconds
    assert sliced.col("close")[0] == series.col("close")[5]


def test_slice_range_is_half_open():
    series = make_bars(10)
    lo = to_ns(TS0 + timedelta(seconds=300 * 3))
    hi = to_ns(TS0 + timedelta(seconds=300 * 6))
    sliced = series.slice_range(lo, hi)
    assert len(sliced) == 3
    assert sliced.first_ts == from_ns(lo)
    assert sliced.last_ts < from_ns(hi)


def test_slice_range_open_ended():
    series = make_bars(10)
    assert len(series.slice_range(None, None)) == 10


# --- bar materialization --------------------------------------------------


def test_bar_at_matches_columns():
    series = make_bars(10)
    bar = series.bar_at(3)
    assert bar.close == series.col("close")[3]
    assert bar.close_ts == from_ns(series.ts_ns[3])
    assert bar.interval_seconds == 300
    assert bar.is_valid


def test_bar_at_rejects_out_of_range():
    series = make_bars(3)
    for index in (-1, 3, 99):
        with pytest.raises(IndexError):
            series.bar_at(index)


def _plain_bars(n=50, symbol="NQ", interval=300):
    """Local bar builder so this file does not depend on shared fixtures."""
    from flow_model.core.contracts import Bar

    out = []
    price = 18000.0
    for i in range(n):
        price += 1.5 if i % 2 else -1.0
        out.append(Bar(symbol=symbol, close_ts=TS0 + timedelta(seconds=interval * (i + 1)),
                       open=price - 2.0, high=price + 3.0, low=price - 3.0, close=price,
                       volume=1000.0 + i, interval_seconds=interval))
    return out


def test_from_bars_round_trip():
    bars = _plain_bars(n=50)
    series = BarSeries.from_bars(bars)
    assert len(series) == 50
    restored = series.to_bars()
    assert [b.close for b in restored] == pytest.approx([b.close for b in bars])
    assert [b.close_ts for b in restored] == [b.close_ts for b in bars]


def test_from_bars_rejects_mixed_symbols_and_intervals():
    a = _plain_bars(n=3, symbol="NQ")
    with pytest.raises(SchemaError, match="multiple symbols"):
        BarSeries.from_bars(a + _plain_bars(n=3, symbol="ES"))
    with pytest.raises(SchemaError, match="multiple intervals"):
        BarSeries.from_bars(a + _plain_bars(n=3, interval=60))


def test_from_bars_rejects_empty():
    with pytest.raises(SchemaError, match="zero bars"):
        BarSeries.from_bars([])


def test_optional_columns_round_trip():
    n = 4
    ts = to_ns_array(TS0 + timedelta(seconds=300 * (i + 1)) for i in range(n))
    series = BarSeries(symbol="NQ", ts_ns=ts, interval_seconds=300,
                       columns=_bar_cols(n) | {"trades": np.full(n, 7.0),
                                               "vwap": np.full(n, 18001.0)})
    bar = series.bar_at(0)
    assert bar.trades == 7 and bar.vwap == pytest.approx(18001.0)
    assert series.has("trades") and not make_bars(2).has("trades")


# --- quotes ---------------------------------------------------------------


def _quotes(n=5):
    ts = to_ns_array(TS0 + timedelta(seconds=300 * (i + 1)) for i in range(n))
    return QuoteSeries(symbol="NQ", ts_ns=ts, columns={
        "bid": np.full(n, 18000.0), "ask": np.full(n, 18000.25),
        "bid_size": np.full(n, 10.0), "ask_size": np.full(n, 20.0)})


def test_quote_at():
    quote = _quotes().quote_at(0)
    assert quote.spread == pytest.approx(0.25)
    assert quote.depth_imbalance == pytest.approx(-1 / 3)


def test_crossed_quotes_are_counted_not_rejected():
    """Crossed quotes occur in real feeds; a high rate means a broken feed,
    so they are counted rather than raised on."""
    n = 4
    ts = to_ns_array(TS0 + timedelta(seconds=300 * (i + 1)) for i in range(n))
    series = QuoteSeries(symbol="NQ", ts_ns=ts, columns={
        "bid": np.array([1.0, 2.0, 1.0, 1.0]), "ask": np.array([2.0, 1.0, 2.0, 2.0]),
        "bid_size": np.ones(n), "ask_size": np.ones(n)})
    assert series.crossed_count() == 1


def test_negative_quote_sizes_rejected():
    n = 2
    with pytest.raises(SchemaError):
        QuoteSeries(symbol="NQ", ts_ns=np.array([1, 2], dtype=np.int64), columns={
            "bid": np.ones(n), "ask": np.full(n, 2.0),
            "bid_size": np.array([-1.0, 1.0]), "ask_size": np.ones(n)})


def test_empty_quote_series_crossed_count_is_zero():
    assert QuoteSeries(symbol="NQ", ts_ns=np.empty(0, dtype=np.int64), columns={
        k: np.empty(0) for k in QuoteSeries.REQUIRED}).crossed_count() == 0


# --- ticks ----------------------------------------------------------------


def _ticks(n=5, unclassified=0.0, method="bid_ask"):
    ts = to_ns_array(TS0 + timedelta(seconds=300 * (i + 1)) for i in range(n))
    return TickSeries(symbol="NQ", ts_ns=ts, meta={"classification_method": method}, columns={
        "buy_volume": np.full(n, 600.0), "sell_volume": np.full(n, 400.0),
        "unclassified_volume": np.full(n, unclassified)})


def test_tick_at_and_delta():
    tick = _ticks().tick_at(0)
    assert tick.delta == pytest.approx(200.0)
    assert tick.delta_ratio == pytest.approx(0.2)
    assert tick.classification_method == "bid_ask"


def test_tick_coverage_reflects_unclassified_volume():
    assert _ticks(unclassified=0.0).coverage() == pytest.approx(1.0)
    assert _ticks(unclassified=1000.0).coverage() == pytest.approx(0.5)


def test_tick_series_without_unclassified_column_is_full_coverage():
    n = 3
    series = TickSeries(symbol="NQ", ts_ns=np.array([1, 2, 3], dtype=np.int64),
                        columns={"buy_volume": np.full(n, 1.0), "sell_volume": np.full(n, 1.0)})
    assert series.coverage() == pytest.approx(1.0)
    assert series.tick_at(0).unclassified_volume == 0.0


def test_unknown_classification_method_defaults_visibly():
    n = 1
    bare = TickSeries(symbol="NQ", ts_ns=np.array([1], dtype=np.int64),
                      columns={"buy_volume": np.ones(n), "sell_volume": np.ones(n)})
    assert bare.classification_method == "unknown"


# --- options --------------------------------------------------------------


def _options(n=3, intraday=False, with_iv=True):
    ts = to_ns_array(TS0 + timedelta(days=i) for i in range(n))
    cols = {
        "call_volume": np.full(n, 1000.0), "put_volume": np.full(n, 500.0),
        "call_premium": np.full(n, 3e6), "put_premium": np.full(n, 1e6),
        "call_oi": np.full(n, 5e4), "put_oi": np.full(n, 6e4),
    }
    if with_iv:
        cols |= {"iv_25d_call": np.full(n, 0.18), "iv_25d_put": np.full(n, 0.24)}
    return OptionsSeries(symbol="SPX", ts_ns=ts, columns=cols,
                         meta={"is_intraday": intraday, "source": "synthetic_eod"})


def test_options_snapshot_derived_values():
    snap = _options().snapshot_at(0)
    assert snap.net_premium == pytest.approx(2e6)
    assert snap.premium_imbalance == pytest.approx(0.5)
    assert snap.skew_25d == pytest.approx(0.06)


def test_eod_options_flag_propagates():
    """EOD chains express positioning but not flow timing, so the flag must
    survive into the snapshot for the quality layer to cap the sub-score."""
    assert _options(intraday=False).snapshot_at(0).is_intraday is False
    assert _options(intraday=True).snapshot_at(0).is_intraday is True
    assert _options().source == "synthetic_eod"


def test_missing_iv_is_none_not_zero():
    """0.0 is a meaningful and very wrong implied volatility."""
    snap = _options(with_iv=False).snapshot_at(0)
    assert snap.iv_25d_call is None
    assert snap.skew_25d is None


def test_nan_optional_value_reads_as_none():
    n = 1
    series = OptionsSeries(symbol="SPX", ts_ns=np.array([1], dtype=np.int64), columns={
        "call_volume": np.ones(n), "put_volume": np.ones(n),
        "call_premium": np.ones(n), "put_premium": np.ones(n),
        "call_oi": np.ones(n), "put_oi": np.ones(n),
        "atm_iv": np.array([np.nan])})
    assert series.snapshot_at(0).atm_iv is None


def test_negative_options_volume_rejected():
    n = 1
    with pytest.raises(SchemaError):
        OptionsSeries(symbol="SPX", ts_ns=np.array([1], dtype=np.int64), columns={
            "call_volume": np.array([-1.0]), "put_volume": np.ones(n),
            "call_premium": np.ones(n), "put_premium": np.ones(n),
            "call_oi": np.ones(n), "put_oi": np.ones(n)})


# --- misc -----------------------------------------------------------------


def test_describe_is_json_friendly():
    described = make_bars(5).describe()
    assert described["rows"] == 5
    assert described["symbol"] == "NQ"
    assert described["meta_interval_seconds"] == 300
    assert isinstance(described["first_ts"], str)


def test_col_error_lists_present_columns():
    with pytest.raises(SchemaError, match="present:"):
        make_bars(3).col("nope")


def test_columns_property_is_sorted():
    assert make_bars(3).columns == ("close", "high", "low", "open", "volume")


def test_base_class_declares_no_columns():
    assert ColumnSeries.REQUIRED == ()


# --- timestamp precision --------------------------------------------------


@pytest.mark.parametrize("microsecond", [0, 1, 123, 123456, 999999])
def test_timestamp_conversion_is_microsecond_exact(microsecond):
    """A float64 holds ~16 significant digits and a modern epoch in seconds
    already uses 10, so routing through value.timestamp() silently loses
    sub-millisecond precision -- which would mangle quote and trade
    timestamps. Conversion is exact integer arithmetic instead."""
    ts = TS0.replace(microsecond=microsecond)
    assert from_ns(to_ns(ts)) == ts


def test_float_route_would_have_been_wrong():
    """Pins the specific defect: the float path is off by tens of
    nanoseconds at a microsecond-precision modern timestamp."""
    ts = TS0.replace(microsecond=999999)
    float_route = int(ts.timestamp() * NS_PER_SECOND)
    assert to_ns(ts) != float_route
    assert to_ns(ts) % 1000 == 0          # exactly microsecond-aligned
    assert abs(to_ns(ts) - float_route) < 1000


def test_microsecond_ordering_is_strict():
    assert to_ns(TS0 - timedelta(microseconds=1)) < to_ns(TS0)
    assert to_ns(TS0) - to_ns(TS0 - timedelta(microseconds=1)) == 1000


def test_from_ns_truncates_and_never_moves_forward():
    """Truncating keeps a derived datetime at or before the instant it
    represents, so converting a cutoff to a datetime cannot advance it."""
    base = to_ns(TS0)
    for offset in (0, 1, 500, 999, 1000, 1001):
        assert from_ns(base + offset) <= TS0 + timedelta(microseconds=offset / 1000)
    assert from_ns(base + 999) == TS0      # sub-microsecond truncated away


def test_nanosecond_resolution_survives_on_the_integer_path():
    """datetime caps at microseconds, but ts_ns arrays and every visibility
    comparison are int64 nanoseconds, so a ns-resolution feed keeps it."""
    ts = np.array([1_000_000_000_000_000_000, 1_000_000_000_000_000_001], dtype=np.int64)
    series = BarSeries(symbol="NQ", ts_ns=ts, interval_seconds=300, columns=_bar_cols(2))
    assert series.visible_count(ts[0]) == 1
    assert series.visible_count(ts[1]) == 2
    assert series.visible_count(ts[1] - 1) == 1


def test_epoch_constant_is_utc():
    from flow_model.data.series import EPOCH

    assert to_ns(EPOCH) == 0
    assert EPOCH.tzinfo is timezone.utc
