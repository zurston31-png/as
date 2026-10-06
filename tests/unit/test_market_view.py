"""Adversarial audit of the point-in-time firewall.

`MarketView` is the one object in the system whose failure is invisible from
above: a lookahead defect does not produce a wrong number, it produces a
*better* number, and nothing downstream has any way to flag it. So this file
attacks the firewall rather than exercising it.

It deliberately imports nothing from `calendar.py`, `synthetic.py`,
`adapters.py`, `clean.py` or `quality.py`. Series are built from raw NumPy
here so that a failure localizes to the firewall instead of to whichever
module happened to produce the data.

The attacks, in order of what they would cost if they succeeded:

1. **Cutoff arithmetic.** A bar closing exactly at the cutoff must be
   visible (you know a bar when it closes); one nanosecond later must not.
   Checked on the int64 path, because `datetime` only carries microseconds
   and the array path carries nanoseconds.
2. **Negative-index wraparound.** `ColumnSeries.window()` clamps `start` to
   0 precisely because a negative `start` would wrap to the END of the array
   under NumPy slicing and hand the caller tomorrow's bars. Every accessor
   that takes `n` is checked, including `spreads` and `deltas`, which do
   arithmetic across two separate windows, and including `n = 10**9`, which
   is what `quality._rows` actually passes.
3. **Future-mutation invariance.** Two stores identical up to bar `i` and
   wildly different after it must give byte-identical answers at bar `i`.
   The past is held constant with a fresh seeded generator per build: a
   version of this test that reused one generator passed for the wrong
   reason and proved nothing.
4. **API shape.** The firewall's guarantee is structural -- a caller can only
   ask for "the last n". An accessor taking a timestamp would quietly break
   it, so the signatures are asserted by introspection and a future accessor
   that breaks the rule fails the build.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.core.contracts import Bar, OptionsSnapshot, QuoteSnapshot, TickAggregate
from flow_model.core.enums import Feed
from flow_model.data.base import SchemaError
from flow_model.data.market_view import LookaheadError, MarketView
from flow_model.data.series import (
    NS_PER_SECOND,
    BarSeries,
    ColumnSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    from_ns,
    to_ns,
)
from flow_model.data.store import DataStore, SymbolData

pytestmark = pytest.mark.lookahead

INTERVAL = 300
HOURLY = 3600
BARS_PER_HOUR = HOURLY // INTERVAL
SYMBOL = "NQ"

#: 09:35 New York on a Tuesday, as UTC. The wall-clock value is irrelevant to
#: the firewall (every comparison is on int64 ns) but a plausible instant
#: makes a failure message readable.
T0 = datetime(2024, 1, 2, 14, 35, tzinfo=timezone.utc)

NS_PER_MICROSECOND = 1_000


# ---------------------------------------------------------------------------
# builders -- local on purpose, so this file depends on no other data module
# ---------------------------------------------------------------------------


def ts_grid(n: int, interval: int = INTERVAL, start: datetime = T0) -> np.ndarray:
    """`n` strictly ascending bar-close stamps, int64 UTC ns."""
    return np.array(
        [to_ns(start) + int(interval) * NS_PER_SECOND * i for i in range(n)], dtype=np.int64
    )


def price_path(n: int, seed: int, base: float = 18000.0) -> np.ndarray:
    """A deterministic close path.

    A *fresh* generator per call is load-bearing for the future-mutation
    test: two builds must agree bit-for-bit on the prefix they share, and a
    generator threaded through both builds would make them differ for a
    reason that has nothing to do with lookahead.
    """
    rng = np.random.default_rng(seed)
    return base * np.exp(np.cumsum(rng.normal(0.0, 0.0008, n)))


def make_bars(
    n: int = 60,
    interval: int = INTERVAL,
    symbol: str = SYMBOL,
    start: datetime = T0,
    seed: int = 11,
    mutate_after: int | None = None,
) -> BarSeries:
    """OHLCV bars. `mutate_after=i` replaces every bar after `i` wholesale."""
    close = price_path(n, seed)
    if mutate_after is not None and mutate_after + 1 < n:
        tail = n - mutate_after - 1
        close[mutate_after + 1 :] = price_path(tail, seed + 99_991, base=1.0e5)
    opens = np.concatenate((close[:1], close[:-1])) if n else close
    volume = np.full(n, 1000.0)
    if mutate_after is not None and mutate_after + 1 < n:
        volume[mutate_after + 1 :] = 7.7e6
    return BarSeries(
        symbol=symbol,
        ts_ns=ts_grid(n, interval, start),
        interval_seconds=interval,
        columns={
            "open": opens,
            "high": np.maximum(opens, close) + 3.0,
            "low": np.minimum(opens, close) - 3.0,
            "close": close,
            "volume": volume,
        },
    )


def make_quotes(
    n: int = 60, interval: int = INTERVAL, symbol: str = SYMBOL, seed: int = 11,
    mutate_after: int | None = None,
) -> QuoteSeries:
    close = price_path(n, seed)
    if mutate_after is not None and mutate_after + 1 < n:
        close[mutate_after + 1 :] = price_path(n - mutate_after - 1, seed + 5, base=1.0e5)
    return QuoteSeries(
        symbol=symbol,
        ts_ns=ts_grid(n, interval),
        columns={
            "bid": close - 0.25,
            "ask": close + 0.25,
            "bid_size": np.full(n, 12.0),
            "ask_size": np.full(n, 8.0),
        },
    )


def make_ticks(
    n: int = 60, interval: int = INTERVAL, symbol: str = SYMBOL, seed: int = 11,
    mutate_after: int | None = None,
) -> TickSeries:
    # One generator per column, not one generator for both: a shared stream
    # would make `sell` depend on `n`, and the truncation test below would then
    # fail for a fixture reason rather than a firewall reason.
    buy = np.round(np.random.default_rng(seed + 1).uniform(100.0, 900.0, n))
    sell = np.round(np.random.default_rng(seed + 2).uniform(100.0, 900.0, n))
    if mutate_after is not None and mutate_after + 1 < n:
        buy[mutate_after + 1 :] = 1.0e6
        sell[mutate_after + 1 :] = 2.0
    return TickSeries(
        symbol=symbol,
        ts_ns=ts_grid(n, interval),
        meta={"classification_method": "bid_ask"},
        columns={
            "buy_volume": buy,
            "sell_volume": sell,
            "unclassified_volume": np.zeros(n),
        },
    )


def make_options(
    n: int = 60, interval: int = INTERVAL, symbol: str = SYMBOL, intraday: bool = False,
    mutate_after: int | None = None,
) -> OptionsSeries:
    """One chain snapshot every `BARS_PER_HOUR` bars, stamped on a bar close."""
    anchors = np.arange(BARS_PER_HOUR - 1, n, BARS_PER_HOUR)
    m = len(anchors)
    values = 1.0 + np.arange(m, dtype=np.float64)
    if mutate_after is not None:
        values = values.copy()
        values[anchors > mutate_after] = 9.9e9
    return OptionsSeries(
        symbol=symbol,
        ts_ns=ts_grid(n, interval)[anchors],
        meta={"is_intraday": intraday, "source": "stub"},
        columns={
            "call_volume": values,
            "put_volume": values + 1.0,
            "call_premium": values * 1.0e6,
            "put_premium": values * 9.0e5,
            "call_oi": values * 1.0e4,
            "put_oi": values * 1.1e4,
        },
    )


def make_symbol_data(
    n: int = 60,
    interval: int = INTERVAL,
    symbol: str = SYMBOL,
    seed: int = 11,
    quotes: bool = True,
    ticks: bool = True,
    options: bool = True,
    hourly: bool = False,
    mutate_after: int | None = None,
) -> SymbolData:
    bars = {interval: make_bars(n, interval, symbol, seed=seed, mutate_after=mutate_after)}
    if hourly:
        # An hourly bar closes on the same instant as every 12th five-minute
        # bar, which is what makes "invisible until its hour closes" testable.
        hn = n // BARS_PER_HOUR
        hourly_bars = make_bars(
            hn,
            HOURLY,
            symbol,
            start=T0 + timedelta(seconds=INTERVAL * (BARS_PER_HOUR - 1)),
            seed=seed + 3,
            mutate_after=None if mutate_after is None else mutate_after // BARS_PER_HOUR,
        )
        bars[HOURLY] = hourly_bars
    return SymbolData(
        symbol=symbol,
        primary_interval=interval,
        bars=bars,
        ticks=(
            {interval: make_ticks(n, interval, symbol, seed=seed, mutate_after=mutate_after)}
            if ticks
            else {}
        ),
        quotes=(
            make_quotes(n, interval, symbol, seed=seed, mutate_after=mutate_after)
            if quotes
            else None
        ),
        options=(make_options(n, interval, symbol, mutate_after=mutate_after) if options else None),
    )


def view_at_bar(data: SymbolData, index: int, latency_seconds: float = 0.0) -> MarketView:
    """A view at the close of bar `index` of the primary series."""
    ts = data.primary_bars.ts_ns[index]
    return MarketView(data, now=from_ns(int(ts)), latency_seconds=latency_seconds)


# --- the accessors that take an `n` ----------------------------------------
#
# Every one of these is a potential negative-index wraparound: each computes
# `visible - n` and hands it to a slice. Kept as one table so a new accessor
# is one line away from being covered by every hazard test below.

ARRAY_ACCESSORS: dict[str, object] = {
    "opens": lambda v, n: v.opens(n),
    "highs": lambda v, n: v.highs(n),
    "lows": lambda v, n: v.lows(n),
    "closes": lambda v, n: v.closes(n),
    "volumes": lambda v, n: v.volumes(n),
    "column_close": lambda v, n: v.column("close", n),
    "bar_timestamps": lambda v, n: v.bar_timestamps(n),
    "quote_column_bid": lambda v, n: v.quote_column("bid", n),
    "quote_column_ask": lambda v, n: v.quote_column("ask", n),
    "spreads": lambda v, n: v.spreads(n),
    "tick_column_buy": lambda v, n: v.tick_column("buy_volume", n),
    "tick_column_sell": lambda v, n: v.tick_column("sell_volume", n),
    "deltas": lambda v, n: v.deltas(n),
    "options_column_call_volume": lambda v, n: v.options_column("call_volume", n),
}

TUPLE_ACCESSORS: dict[str, object] = {
    "bars": lambda v, n: v.bars(n),
    "tick_aggregates": lambda v, n: v.tick_aggregates(n),
}

#: Visible-row ceiling for each accessor, so "never more rows than visible"
#: can be asserted per feed rather than against the bar count alone.
def visible_ceiling(view: MarketView, name: str) -> int:
    if name.startswith("quote_column") or name == "spreads":
        return view.quote_count()
    if name.startswith("tick_column") or name in ("deltas", "tick_aggregates"):
        return view.tick_count()
    if name.startswith("options_column"):
        return view.options_count()
    return view.bar_count()


def stamp_ceiling(view: MarketView, name: str, data=None) -> np.ndarray:
    """The timestamps the rows of `name` were drawn from, newest last.

    Takes the parent `SymbolData` explicitly: the view deliberately holds no
    reference to it, so a test that needs the full history must carry it.
    """
    if data is None:
        raise AssertionError("stamp_ceiling needs the parent SymbolData")
    if name.startswith("quote_column") or name == "spreads":
        return np.asarray(data.quotes.ts_ns)
    if name.startswith("tick_column") or name in ("deltas", "tick_aggregates"):
        return np.asarray(data.ticks[view.primary_interval].ts_ns)
    if name.startswith("options_column"):
        return np.asarray(data.options.ts_ns)
    return np.asarray(data.primary_bars.ts_ns)


# ---------------------------------------------------------------------------
# 1. the exact cutoff boundary
# ---------------------------------------------------------------------------


def test_bar_closing_exactly_at_the_cutoff_is_visible():
    """A bar is known at the instant it closes; `side="right"` says so."""
    data = make_symbol_data(n=20)
    view = view_at_bar(data, 9)
    assert view.bar_count() == 10
    assert int(view.bar_timestamps(1)[0]) == int(data.primary_bars.ts_ns[9])


def test_bar_one_microsecond_after_the_cutoff_is_hidden():
    data = make_symbol_data(n=20)
    cutoff_ns = int(data.primary_bars.ts_ns[9])
    view = MarketView(data, now=from_ns(cutoff_ns - NS_PER_MICROSECOND))
    assert view.bar_count() == 9


def test_bar_one_nanosecond_after_the_cutoff_is_hidden_on_the_int64_path():
    """`datetime` carries microseconds; `ts_ns` carries nanoseconds.

    The cutoff comparison happens on the integers, so a bar one nanosecond
    past `now` must be hidden even though no datetime can express the gap.
    """
    ts = ts_grid(20)
    ts[9] += 1  # one nanosecond after the instant we will evaluate at
    bars = BarSeries(
        symbol=SYMBOL,
        ts_ns=ts,
        interval_seconds=INTERVAL,
        columns={
            "open": np.full(20, 10.0), "high": np.full(20, 11.0), "low": np.full(20, 9.0),
            "close": np.full(20, 10.5), "volume": np.full(20, 1.0),
        },
    )
    data = SymbolData(symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bars})
    view = MarketView(data, now=from_ns(int(ts_grid(20)[9])))
    assert view.bar_count() == 9
    view.assert_no_future_leak()


def test_cutoff_equals_now_when_latency_is_zero():
    data = make_symbol_data(n=10)
    now = from_ns(int(data.primary_bars.ts_ns[4]))
    view = MarketView(data, now=now, latency_seconds=0.0)
    assert view.cutoff == now


def test_naive_now_is_rejected():
    """A naive `now` would be read as UTC and shift the cutoff by the
    exchange's offset -- in New York, five hours of free lookahead."""
    data = make_symbol_data(n=10)
    with pytest.raises(SchemaError):
        MarketView(data, now=datetime(2024, 1, 2, 14, 35))


def test_no_accessor_returns_a_row_after_the_cutoff():
    data = make_symbol_data(n=48, hourly=True)
    view = view_at_bar(data, 25)
    cutoff = view._cutoff_ns
    for name in ARRAY_ACCESSORS:
        rows = int(np.asarray(ARRAY_ACCESSORS[name](view, 10_000)).size)
        stamps = stamp_ceiling(view, name, data)
        if rows:
            assert int(stamps[rows - 1]) <= cutoff, name
    assert int(view.bar_timestamps(10_000, HOURLY).max()) <= cutoff


# ---------------------------------------------------------------------------
# 2. negative-index wraparound
# ---------------------------------------------------------------------------


def test_asking_for_more_bars_than_exist_does_not_return_the_tail():
    """The headline hazard: `visible - n` is negative, and a negative slice
    start wraps to the END of the array under NumPy semantics."""
    data = make_symbol_data(n=1000, quotes=False, ticks=False, options=False)
    view = view_at_bar(data, 10)
    closes = view.closes(500)
    stamps = view.bar_timestamps(500)
    assert closes.size == 11
    assert stamps.size == 11
    assert int(stamps.max()) <= view._cutoff_ns
    # The tail of the full series is nowhere in the answer.
    full = np.asarray(data.primary_bars.col("close"))
    assert np.array_equal(closes, full[:11])
    assert full[-1] not in set(closes.tolist())


@pytest.mark.parametrize("name", sorted(ARRAY_ACCESSORS))
@pytest.mark.parametrize("n", [11, 500, 10**6, 10**9])
def test_array_accessors_never_wrap_for_oversized_n(name, n):
    data = make_symbol_data(n=1000)
    view = view_at_bar(data, 10)
    out = np.asarray(ARRAY_ACCESSORS[name](view, n))
    ceiling = visible_ceiling(view, name)
    assert out.size <= ceiling, f"{name} returned {out.size} rows, visible is {ceiling}"
    stamps = stamp_ceiling(view, name, data)
    if out.size:
        assert int(stamps[out.size - 1]) <= view._cutoff_ns, name


@pytest.mark.parametrize("name", sorted(TUPLE_ACCESSORS))
@pytest.mark.parametrize("n", [11, 500, 10**6, 10**9])
def test_contract_accessors_never_wrap_for_oversized_n(name, n):
    data = make_symbol_data(n=1000)
    view = view_at_bar(data, 10)
    rows = TUPLE_ACCESSORS[name](view, n)
    assert len(rows) <= visible_ceiling(view, name)
    cutoff = view.cutoff
    for row in rows:
        assert row.close_ts <= cutoff


@pytest.mark.parametrize("name", sorted(ARRAY_ACCESSORS))
@pytest.mark.parametrize("n", [0, -1, -1000])
def test_array_accessors_return_nothing_for_non_positive_n(name, n):
    data = make_symbol_data(n=40)
    view = view_at_bar(data, 20)
    assert np.asarray(ARRAY_ACCESSORS[name](view, n)).size == 0


def test_spreads_and_deltas_agree_in_length_with_their_two_windows():
    """Both do arithmetic across two independent windows. If the windows
    could come back different lengths the subtraction would broadcast, or
    raise, instead of returning the right number of rows."""
    data = make_symbol_data(n=200)
    for index in (0, 1, 7, 60, 199):
        view = view_at_bar(data, index)
        for n in (1, 3, 50, 10**9):
            bids = view.quote_column("bid", n)
            asks = view.quote_column("ask", n)
            assert bids.size == asks.size
            assert view.spreads(n).size == bids.size
            buys = view.tick_column("buy_volume", n)
            sells = view.tick_column("sell_volume", n)
            assert buys.size == sells.size
            assert view.deltas(n).size == buys.size


def test_quality_rows_probe_with_one_billion_is_bounded():
    """`quality._rows` really does pass n=10**9 to three accessors."""
    data = make_symbol_data(n=500)
    view = view_at_bar(data, 123)
    assert view.quote_column("bid", 10**9).size == view.quote_count()
    assert view.tick_column("buy_volume", 10**9).size == view.tick_count()
    assert view.options_column("call_volume", 10**9).size == view.options_count()
    assert view.quote_count() == 124


def test_an_absent_optional_tick_column_returns_empty_not_a_wrapped_window():
    data = make_symbol_data(n=40)
    view = view_at_bar(data, 20)
    assert view.tick_column("max_trade_size", 10).size == 0
    assert view.options_column("atm_iv", 10).size == 0


# ---------------------------------------------------------------------------
# 3. full-series sweep
# ---------------------------------------------------------------------------


def test_full_series_sweep_every_view_is_bounded_and_self_consistent():
    data = make_symbol_data(n=300, hourly=True)
    stamps = np.asarray(data.primary_bars.ts_ns)
    for index in range(len(stamps)):
        view = view_at_bar(data, index)
        cutoff = to_ns(view.cutoff)
        bar_stamps = view.bar_timestamps(10_000)
        assert bar_stamps.size == index + 1
        assert int(bar_stamps.max()) <= cutoff
        view.assert_no_future_leak()


def test_full_series_sweep_last_price_tracks_the_visible_close():
    data = make_symbol_data(n=200, quotes=False, ticks=False, options=False)
    closes = np.asarray(data.primary_bars.col("close"))
    for index in range(len(closes)):
        view = view_at_bar(data, index)
        assert view.last_price() == pytest.approx(float(closes[index]))
        assert view.last_bar().close_ts == from_ns(int(data.primary_bars.ts_ns[index]))


def test_sweep_with_latency_never_shows_a_bar_after_the_cutoff():
    data = make_symbol_data(n=200, quotes=False, ticks=False, options=False)
    for index in range(len(data.primary_bars)):
        view = view_at_bar(data, index, latency_seconds=137.0)
        stamps = view.bar_timestamps(10_000)
        if stamps.size:
            assert int(stamps.max()) <= view._cutoff_ns
        view.assert_no_future_leak()


# ---------------------------------------------------------------------------
# 4. future-mutation invariance
# ---------------------------------------------------------------------------


def _snapshot(view: MarketView) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, fn in ARRAY_ACCESSORS.items():
        out[name] = np.asarray(fn(view, 10_000)).copy()
    for name, fn in TUPLE_ACCESSORS.items():
        out[name] = tuple(fn(view, 10_000))
    out["bar_count"] = view.bar_count()
    out["bar_count_hourly"] = view.bar_count(HOURLY)
    out["hourly_closes"] = np.asarray(view.closes(10_000, HOURLY)).copy()
    out["last_price"] = view.last_price()
    out["last_bar"] = view.last_bar()
    out["quote"] = view.quote()
    out["options_snapshot"] = view.options_snapshot()
    out["last_tick_aggregate"] = view.last_tick_aggregate()
    out["available_feeds"] = view.available_feeds()
    out["ages"] = {feed: view.feed_age_seconds(feed) for feed in Feed}
    out["options_are_intraday"] = view.options_are_intraday()
    return out


def _differences(left: dict[str, object], right: dict[str, object]) -> list[str]:
    bad = []
    for key, value in left.items():
        other = right[key]
        same = (
            np.array_equal(value, other)
            if isinstance(value, np.ndarray)
            else value == other
        )
        if not same:
            bad.append(key)
    return bad


@pytest.mark.parametrize("index", [0, 1, 11, 97, 150])
def test_replacing_every_bar_after_i_changes_nothing_visible_at_i(index):
    """The strongest statement the firewall can make.

    Both stores are built from scratch with the same seed, so the prefix is
    bit-identical; only the suffix differs. If any accessor consulted a row
    after the cutoff the two snapshots would diverge.
    """
    clean = make_symbol_data(n=200, hourly=True)
    mutated = make_symbol_data(n=200, hourly=True, mutate_after=index)
    assert _differences(_snapshot(view_at_bar(clean, index)),
                        _snapshot(view_at_bar(mutated, index))) == []


def test_the_mutation_fixture_actually_mutates():
    """Guard on the guard: a fixture that silently produced identical stores
    would make the invariance test pass for the wrong reason, which is how
    the first attempt at this test went."""
    index = 97
    clean = make_symbol_data(n=200, hourly=True)
    mutated = make_symbol_data(n=200, hourly=True, mutate_after=index)
    for name in ("close", "volume"):
        a = np.asarray(clean.primary_bars.col(name))
        b = np.asarray(mutated.primary_bars.col(name))
        assert np.array_equal(a[: index + 1], b[: index + 1]), name
        assert not np.allclose(a[index + 1 :], b[index + 1 :]), name
    assert not np.allclose(
        np.asarray(clean.quotes.col("bid"))[index + 1 :],
        np.asarray(mutated.quotes.col("bid"))[index + 1 :],
    )
    assert not np.allclose(
        np.asarray(clean.ticks[INTERVAL].col("buy_volume"))[index + 1 :],
        np.asarray(mutated.ticks[INTERVAL].col("buy_volume"))[index + 1 :],
    )


def test_truncating_the_store_after_i_changes_nothing_visible_at_i():
    """A view at bar `i` must be indistinguishable from a view on a dataset
    that simply ends at bar `i`. Anything else means a row after `i` is
    being read."""
    index = 60
    full = make_symbol_data(n=200, hourly=True)
    short = make_symbol_data(n=index + 1, hourly=True)
    assert _differences(_snapshot(view_at_bar(full, index)),
                        _snapshot(view_at_bar(short, index))) == []


# ---------------------------------------------------------------------------
# 5. multi-interval visibility
# ---------------------------------------------------------------------------


def test_an_hourly_bar_is_invisible_until_its_hour_closes():
    data = make_symbol_data(n=48, hourly=True)
    assert view_at_bar(data, BARS_PER_HOUR - 2).bar_count(HOURLY) == 0
    assert view_at_bar(data, BARS_PER_HOUR - 1).bar_count(HOURLY) == 1
    assert view_at_bar(data, BARS_PER_HOUR).bar_count(HOURLY) == 1
    assert view_at_bar(data, 2 * BARS_PER_HOUR - 1).bar_count(HOURLY) == 2


def test_hourly_window_never_outruns_the_cutoff_across_a_sweep():
    data = make_symbol_data(n=72, hourly=True)
    for index in range(len(data.primary_bars)):
        view = view_at_bar(data, index)
        stamps = view.bar_timestamps(10_000, HOURLY)
        assert stamps.size == (index + 1) // BARS_PER_HOUR
        if stamps.size:
            assert int(stamps.max()) <= view._cutoff_ns


def test_an_unknown_interval_is_empty_rather_than_falling_back_to_primary():
    """Falling back would answer an hourly question with five-minute bars,
    which reads as a successful computation."""
    data = make_symbol_data(n=48, hourly=True)
    view = view_at_bar(data, 30)
    assert view.bar_count(900) == 0
    assert view.closes(10, 900).size == 0
    assert view.bar_timestamps(10, 900).size == 0
    assert view.bars(10, 900) == ()
    assert view.last_bar(900) is None
    assert view.last_price(900) is None


def test_warmup_ok_counts_only_visible_bars_per_interval():
    data = make_symbol_data(n=48, hourly=True)
    view = view_at_bar(data, 23)
    assert view.bar_count() == 24
    assert view.warmup_ok(24) and not view.warmup_ok(25)
    assert view.warmup_ok(2, HOURLY) and not view.warmup_ok(3, HOURLY)


# ---------------------------------------------------------------------------
# 6. latency
# ---------------------------------------------------------------------------


def test_latency_equal_to_one_interval_hides_exactly_one_more_bar():
    data = make_symbol_data(n=60, quotes=False, ticks=False, options=False)
    for index in (1, 5, 30, 59):
        plain = view_at_bar(data, index, latency_seconds=0.0)
        delayed = view_at_bar(data, index, latency_seconds=float(INTERVAL))
        assert delayed.bar_count() == plain.bar_count() - 1


def test_latency_moves_the_cutoff_back_by_exactly_latency():
    data = make_symbol_data(n=20)
    now = from_ns(int(data.primary_bars.ts_ns[10]))
    view = MarketView(data, now=now, latency_seconds=2.5)
    assert view.cutoff == now - timedelta(seconds=2.5)
    assert view._cutoff_ns == view._now_ns - int(2.5 * NS_PER_SECOND)


def test_sub_interval_latency_hides_a_bar_closing_at_now():
    data = make_symbol_data(n=20, quotes=False, ticks=False, options=False)
    plain = view_at_bar(data, 10, latency_seconds=0.0)
    delayed = view_at_bar(data, 10, latency_seconds=0.000001)
    assert plain.bar_count() == 11
    assert delayed.bar_count() == 10


def test_negative_latency_is_rejected():
    """A negative feed latency is literally a request to see the future, and
    it would otherwise work -- the cutoff would simply be after `now`."""
    data = make_symbol_data(n=10)
    now = from_ns(int(data.primary_bars.ts_ns[4]))
    for latency in (-1e-9, -0.5, -300.0):
        with pytest.raises(ValueError, match="negative"):
            MarketView(data, now=now, latency_seconds=latency)


def test_datastore_rejects_negative_latency_too():
    with pytest.raises(ValueError):
        DataStore(latency_seconds=-1.0)


def test_latency_is_applied_to_every_feed_not_just_bars():
    data = make_symbol_data(n=60)
    plain = view_at_bar(data, 30, latency_seconds=0.0)
    delayed = view_at_bar(data, 30, latency_seconds=float(INTERVAL))
    assert delayed.quote_count() == plain.quote_count() - 1
    assert delayed.tick_count() == plain.tick_count() - 1
    assert delayed.options_count() <= plain.options_count()
    delayed.assert_no_future_leak()


# ---------------------------------------------------------------------------
# 7. read-only arrays and non-aliasing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["opens", "highs", "lows", "closes", "volumes", "column_close", "bar_timestamps",
             "quote_column_bid", "tick_column_buy", "options_column_call_volume"]
)
def test_sliced_windows_are_read_only(name):
    """A consumer that could write through a window would be rewriting
    stored market data for every later view in the run."""
    data = make_symbol_data(n=40)
    view = view_at_bar(data, 30)
    out = ARRAY_ACCESSORS[name](view, 5)
    assert out.size
    assert out.flags.writeable is False
    with pytest.raises(ValueError):
        out[0] = out[0] + 1


@pytest.mark.parametrize("name", ["spreads", "deltas"])
def test_computed_windows_do_not_alias_the_store(name):
    data = make_symbol_data(n=40)
    view = view_at_bar(data, 30)
    before_quotes = np.asarray(data.quotes.col("bid")).copy()
    before_ticks = np.asarray(data.ticks[INTERVAL].col("buy_volume")).copy()
    out = ARRAY_ACCESSORS[name](view, 5)
    out[0] = -12345.0  # a fresh array: writing is allowed, and must not propagate
    assert np.array_equal(np.asarray(data.quotes.col("bid")), before_quotes)
    assert np.array_equal(np.asarray(data.ticks[INTERVAL].col("buy_volume")), before_ticks)


def test_mutating_the_caller_buffer_after_construction_cannot_change_a_view():
    """`BarSeries` copies at construction; if it aliased, a caller holding the
    source array could rewrite history under a running backtest."""
    close = price_path(30, seed=4)
    source = {
        "open": close.copy(), "high": close + 5.0, "low": close - 5.0,
        "close": close, "volume": np.full(30, 100.0),
    }
    bars = BarSeries(symbol=SYMBOL, ts_ns=ts_grid(30), interval_seconds=INTERVAL, columns=source)
    data = SymbolData(symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bars})
    before = np.asarray(view_at_bar(data, 20).closes(10_000)).copy()
    close[:] = 1.0e9
    source["volume"][:] = 0.0
    assert np.array_equal(np.asarray(view_at_bar(data, 20).closes(10_000)), before)


def test_no_accessor_hands_back_a_series_object():
    """Rule 2 of the firewall: only bounded windows leave the view."""
    data = make_symbol_data(n=40, hourly=True)
    view = view_at_bar(data, 20)
    returned = [fn(view, 10_000) for fn in ARRAY_ACCESSORS.values()]
    returned += [fn(view, 10_000) for fn in TUPLE_ACCESSORS.values()]
    returned += [view.quote(), view.options_snapshot(), view.last_bar(),
                 view.last_tick_aggregate()]
    for item in returned:
        assert not isinstance(item, ColumnSeries)
        assert not isinstance(item, SymbolData)


# ---------------------------------------------------------------------------
# 8. API shape -- the structural guarantee, asserted by introspection
# ---------------------------------------------------------------------------

#: Parameter names that would let a caller aim an accessor at an instant.
#: The firewall's guarantee is that you can only ask for "the last n".
TIMESTAMP_FLAVOURED = {
    "ts", "timestamp", "when", "at", "start", "end", "until", "before", "after", "date", "time",
}


def _public_methods() -> list[tuple[str, object]]:
    return [
        (name, member)
        for name, member in inspect.getmembers(MarketView, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]


def test_no_public_accessor_takes_a_timestamp_flavoured_parameter():
    offenders = {}
    for name, member in _public_methods():
        params = set(inspect.signature(member).parameters) - {"self"}
        bad = params & TIMESTAMP_FLAVOURED
        if bad:
            offenders[name] = sorted(bad)
    assert offenders == {}, (
        f"accessors taking a timestamp-flavoured parameter: {offenders}. A caller that "
        "can name an instant can name a future one; the firewall only permits "
        "'the last n'."
    )


def test_no_public_accessor_is_named_like_a_timestamp_query():
    offenders = [
        name
        for name, _ in _public_methods()
        if "between" in name or name.endswith("_at") or name.startswith("value_")
    ]
    assert offenders == []


def test_every_window_accessor_requires_an_explicit_n():
    """A default `n` would let a caller omit the bound and get whatever the
    accessor felt like returning."""
    for name, member in _public_methods():
        params = inspect.signature(member).parameters
        if "n" in params:
            assert params["n"].default is inspect.Parameter.empty, name


def test_slots_match_the_attributes_actually_assigned():
    """A typo in `__slots__` would either raise at construction or, worse,
    silently create a class-level attribute shared across views."""
    import ast
    from pathlib import Path

    source = Path(inspect.getsourcefile(MarketView)).read_text(encoding="utf-8")
    assigned = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Store)
    }
    declared = set(MarketView.__slots__)
    assert assigned - declared == set(), f"assigned but not in __slots__: {assigned - declared}"
    assert declared - assigned == set(), f"declared but never assigned: {declared - assigned}"


def test_a_view_has_no_instance_dict():
    data = make_symbol_data(n=5)
    view = view_at_bar(data, 2)
    assert not hasattr(view, "__dict__")
    with pytest.raises(AttributeError):
        view.sneaky_cached_future = 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 9. degenerate views
# ---------------------------------------------------------------------------


def test_empty_bar_series_view_answers_everything_emptily():
    bars = BarSeries.empty(SYMBOL, INTERVAL)
    data = SymbolData(symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bars})
    view = MarketView(data, now=T0)
    assert view.bar_count() == 0
    assert view.last_bar() is None
    assert view.last_price() is None
    assert view.bars(10) == ()
    assert view.closes(10).size == 0
    assert view.bar_timestamps(10).size == 0
    assert view.has_feed(Feed.BARS) is False
    assert view.available_feeds() == frozenset()
    assert view.feed_age_seconds(Feed.BARS) is None
    view.assert_no_future_leak()


def test_single_bar_view():
    data = make_symbol_data(n=1, quotes=False, ticks=False, options=False)
    view = view_at_bar(data, 0)
    assert view.bar_count() == 1
    assert view.closes(10_000).size == 1
    assert view.bars(10_000)[0].close_ts == view.cutoff
    view.assert_no_future_leak()


def test_all_feeds_absent_except_bars():
    data = make_symbol_data(n=30, quotes=False, ticks=False, options=False)
    view = view_at_bar(data, 20)
    assert view.available_feeds() == frozenset({Feed.BARS})
    assert view.quote() is None
    assert view.options_snapshot() is None
    assert view.last_tick_aggregate() is None
    assert view.quote_column("bid", 10).size == 0
    assert view.tick_column("buy_volume", 10).size == 0
    assert view.options_column("call_volume", 10).size == 0
    assert view.spreads(10).size == 0
    assert view.deltas(10).size == 0
    assert view.options_are_intraday() is False
    for feed in (Feed.QUOTES, Feed.TICK_AGGREGATE, Feed.OPTIONS_SNAPSHOT):
        assert view.has_feed(feed) is False
        assert view.feed_age_seconds(feed) is None
    view.assert_no_future_leak()


def test_a_view_before_all_data_sees_nothing_although_the_data_exists():
    data = make_symbol_data(n=60)
    view = MarketView(data, now=T0 - timedelta(days=1))
    assert view.bar_count() == 0
    assert view.available_feeds() == frozenset()
    assert view.last_price() is None
    view.assert_no_future_leak()


def test_a_view_after_all_data_sees_everything_and_still_respects_the_cutoff():
    data = make_symbol_data(n=60)
    view = MarketView(data, now=T0 + timedelta(days=30))
    assert view.bar_count() == 60
    assert view.closes(10_000).size == 60
    assert int(view.bar_timestamps(10_000).max()) <= view._cutoff_ns
    assert view.feed_age_seconds(Feed.BARS) > 0
    view.assert_no_future_leak()


def test_a_feed_that_starts_late_is_absent_until_its_first_observation():
    """A dataset-level "quotes exist" claim is not a point-in-time one."""
    bars = make_bars(n=60)
    quotes = QuoteSeries(
        symbol=SYMBOL,
        ts_ns=ts_grid(60)[50:],
        columns={
            "bid": np.full(10, 1.0), "ask": np.full(10, 1.5),
            "bid_size": np.full(10, 2.0), "ask_size": np.full(10, 3.0),
        },
    )
    data = SymbolData(symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bars},
                      quotes=quotes)
    assert Feed.QUOTES in data.available_feeds()
    assert view_at_bar(data, 10).has_feed(Feed.QUOTES) is False
    assert view_at_bar(data, 49).has_feed(Feed.QUOTES) is False
    assert view_at_bar(data, 50).has_feed(Feed.QUOTES) is True


def test_feed_age_is_never_negative_across_a_sweep():
    """A negative age would mean an observation dated after `now`."""
    data = make_symbol_data(n=120, hourly=True)
    for index in range(len(data.primary_bars)):
        view = view_at_bar(data, index)
        for feed in Feed:
            age = view.feed_age_seconds(feed)
            assert age is None or age >= 0.0, (index, feed)


def test_newest_visible_contract_objects_are_not_the_newest_overall():
    data = make_symbol_data(n=120)
    view = view_at_bar(data, 40)
    bar = view.last_bar()
    quote = view.quote()
    snapshot = view.options_snapshot()
    tick = view.last_tick_aggregate()
    assert isinstance(bar, Bar) and bar.close_ts <= view.cutoff
    assert isinstance(quote, QuoteSnapshot) and quote.ts <= view.cutoff
    assert isinstance(snapshot, OptionsSnapshot) and snapshot.ts <= view.cutoff
    assert isinstance(tick, TickAggregate) and tick.close_ts <= view.cutoff
    assert bar.close_ts < from_ns(int(data.primary_bars.ts_ns[-1]))


# ---------------------------------------------------------------------------
# 10. the self-check itself
# ---------------------------------------------------------------------------


def _overcut(view, key, series, count):
    """Install a deliberately wrong prefix, simulating a bad cutoff slice.

    The view no longer stores an integer count that can simply be bumped --
    it stores the prefix itself -- so corrupting it means installing a
    prefix of the wrong length and the matching hidden boundary. That is the
    only remaining way to express this class of bug, which is the point of
    the prefix design.
    """
    view._bars[INTERVAL] = series.prefix(count) if key.startswith("bars") else view._bars[INTERVAL]
    view._hidden[key] = series.hidden_boundary_ns(count)
    return view


def test_assert_no_future_leak_catches_a_prefix_that_is_too_long():
    data = make_symbol_data(n=40, quotes=False, ticks=False, options=False)
    view = view_at_bar(data, 20)
    _overcut(view, f"bars[{INTERVAL}]", data.primary_bars, 25)
    with pytest.raises(LookaheadError, match="after the cutoff"):
        view.assert_no_future_leak()


def test_a_prefix_cannot_exceed_the_series_length():
    """Structurally unreachable now: `prefix()` clamps, so the old
    "count past the end of the series" bug cannot be expressed. Asserted so
    the clamp is not quietly removed."""
    data = make_symbol_data(n=40, quotes=False, ticks=False, options=False)
    series = data.primary_bars
    assert len(series.prefix(999)) == len(series) == 40
    assert len(series.prefix(-5)) == 0
    assert series.hidden_boundary_ns(999) is None


def test_assert_no_future_leak_catches_a_prefix_that_is_too_short():
    """An off-by-one that HID the newest bar is not a lookahead, but it is a
    silent accuracy loss, so the self-check covers that direction too."""
    data = make_symbol_data(n=40, quotes=False, ticks=False, options=False)
    view = view_at_bar(data, 20)
    _overcut(view, f"bars[{INTERVAL}]", data.primary_bars, 15)
    with pytest.raises(LookaheadError, match="but is hidden"):
        view.assert_no_future_leak()


def test_assert_no_future_leak_checks_quotes_ticks_and_options_too():
    data = make_symbol_data(n=60)

    view = view_at_bar(data, 30)
    view._quotes = data.quotes.prefix(45)
    with pytest.raises(LookaheadError, match="quotes"):
        view.assert_no_future_leak()

    view = view_at_bar(data, 30)
    view._ticks[INTERVAL] = data.ticks[INTERVAL].prefix(45)
    with pytest.raises(LookaheadError, match=r"ticks\[300\]"):
        view.assert_no_future_leak()

    # Isolate the options feed: lowering the cutoff instead would also
    # violate the bars prefix, which is checked first, so the assertion
    # would pass for the wrong reason.
    view = view_at_bar(data, 30)
    view._options = data.options.prefix(len(data.options))
    with pytest.raises(LookaheadError, match="options"):
        view.assert_no_future_leak()


def test_assert_no_future_leak_rejects_a_non_monotonic_visible_prefix():
    """A non-ascending series makes searchsorted span a future row. The
    check now reports that directly instead of mis-slicing silently."""
    ts = np.array(
        [to_ns(T0), to_ns(T0 + timedelta(seconds=2 * INTERVAL)),
         to_ns(T0 + timedelta(seconds=INTERVAL))],
        dtype=np.int64,
    )
    bars = BarSeries(
        symbol=SYMBOL, ts_ns=ts, interval_seconds=INTERVAL, validate=False,
        columns={
            "open": np.full(3, 10.0), "high": np.full(3, 11.0), "low": np.full(3, 9.0),
            "close": np.full(3, 10.5), "volume": np.full(3, 1.0),
        },
    )
    data = SymbolData(symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bars})
    view = MarketView(data, now=T0 + timedelta(seconds=2 * INTERVAL))
    with pytest.raises(LookaheadError, match="strictly ascending|after the cutoff"):
        view.assert_no_future_leak()


def test_assert_no_future_leak_rejects_a_future_row_that_is_not_the_last_visible_one():
    """Regression for an audit finding: the check used to inspect only
    `ts_ns[count - 1]`, so it certified a view whose visible prefix held a
    future row as long as the LAST visible row was in the past. It now
    bounds the whole prefix."""
    ts = np.array(
        [to_ns(T0 + timedelta(hours=6)), to_ns(T0), to_ns(T0 + timedelta(seconds=INTERVAL))],
        dtype=np.int64,
    )
    bars = BarSeries(
        symbol=SYMBOL, ts_ns=ts, interval_seconds=INTERVAL, validate=False,
        columns={
            "open": np.array([999.0, 100.0, 101.0]), "high": np.array([1000.0, 101.0, 102.0]),
            "low": np.array([998.0, 99.0, 100.0]), "close": np.array([999.0, 100.0, 101.0]),
            "volume": np.full(3, 1.0),
        },
    )
    data = SymbolData(symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bars})
    view = MarketView(data, now=T0 + timedelta(seconds=INTERVAL))
    leaked = [int(x) for x in view.bar_timestamps(10_000) if int(x) > view._cutoff_ns]
    assert leaked, "fixture did not produce a visible future row"
    with pytest.raises(LookaheadError):
        view.assert_no_future_leak()


def test_the_view_holds_only_its_visible_prefix():
    """Regression for an audit finding: the module docstring claims future
    data is absent from the object rather than merely un-requested. The
    first implementation stored the whole SymbolData plus integer counts, so
    the guarantee was procedural and a new accessor that forgot to re-apply
    a count would have returned the full history. It now stores prefixes.
    """
    data = make_symbol_data(n=200, quotes=False, ticks=False, options=False)
    view = view_at_bar(data, 20)
    assert not hasattr(view, "_data")
    assert "_data" not in MarketView.__slots__
    assert len(view._bars[INTERVAL]) == view.bar_count() == 21
    # every stored prefix is bounded, for every feed
    full = make_symbol_data(n=200)
    v2 = view_at_bar(full, 20)
    for store in (v2._bars, v2._ticks):
        for series in store.values():
            assert int(series.ts_ns.max()) <= v2._cutoff_ns
    for series in (v2._quotes, v2._options):
        if series is not None and len(series):
            assert int(series.ts_ns.max()) <= v2._cutoff_ns


def test_iter_views_sees_each_bar_at_its_own_close_with_nanosecond_stamps():
    """Regression for an audit finding: the clock round-tripped through
    datetime, and `from_ns` truncates below the microsecond, so a
    nanosecond-stamped bar was not visible at its own close. `DataStore` now
    passes the exact int64 instant alongside the datetime."""
    ts = ts_grid(20) + 500  # 500 ns past the microsecond grid
    bars = BarSeries(
        symbol=SYMBOL, ts_ns=ts, interval_seconds=INTERVAL,
        columns={
            "open": np.full(20, 10.0), "high": np.full(20, 11.0), "low": np.full(20, 9.0),
            "close": np.full(20, 10.5), "volume": np.full(20, 1.0),
        },
    )
    store = DataStore().add(
        SymbolData(symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bars})
    )
    assert [v.bar_count() for v in store.iter_views(SYMBOL)] == list(range(1, 21))


def test_an_unknown_bar_column_raises_a_data_layer_error():
    """Regression for an audit finding: these used to raise a bare KeyError."""
    data = make_symbol_data(n=30)
    view = view_at_bar(data, 20)
    with pytest.raises(SchemaError):
        view.column("vwap", 5)


def test_unknown_feed_is_rejected_rather_than_silently_absent():
    data = make_symbol_data(n=10)
    view = view_at_bar(data, 5)
    with pytest.raises(ValueError, match="unknown feed"):
        view.has_feed("bars")  # type: ignore[arg-type]


def test_repr_does_not_disclose_anything_past_the_cutoff():
    data = make_symbol_data(n=60)
    view = view_at_bar(data, 10)
    text = repr(view)
    assert view.cutoff.isoformat() in text
    assert from_ns(int(data.primary_bars.ts_ns[-1])).isoformat() not in text
