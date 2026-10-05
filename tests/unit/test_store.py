"""DataStore: ownership of loaded series, and the seal wired into loading."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.data.base import DataLayerError, DataSourceAdapter, Feed
from flow_model.data.market_view import MarketView
from flow_model.data.series import (
    BarSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    from_ns,
    to_ns,
    to_ns_array,
)
from flow_model.data.store import DataStore, SymbolData
from flow_model.validation.splits import SealedDataAccessError, guard_from_config

TS0 = datetime(2020, 1, 2, 14, 30, tzinfo=timezone.utc)


def make_bars(n=20, interval=300, symbol="NQ", start=TS0, base=18000.0):
    ts = to_ns_array(start + timedelta(seconds=interval * (i + 1)) for i in range(n))
    close = base + np.arange(n, dtype=np.float64) * 2.0
    return BarSeries(symbol=symbol, ts_ns=ts, interval_seconds=interval, columns={
        "open": close - 1.0, "high": close + 4.0, "low": close - 4.0,
        "close": close, "volume": np.full(n, 1000.0)})


def make_quotes(n=20, interval=300, symbol="NQ"):
    ts = to_ns_array(TS0 + timedelta(seconds=interval * (i + 1)) for i in range(n))
    return QuoteSeries(symbol=symbol, ts_ns=ts, columns={
        "bid": np.full(n, 18000.0), "ask": np.full(n, 18000.25),
        "bid_size": np.full(n, 12.0), "ask_size": np.full(n, 8.0)})


def make_ticks(n=20, interval=300, symbol="NQ"):
    ts = to_ns_array(TS0 + timedelta(seconds=interval * (i + 1)) for i in range(n))
    return TickSeries(symbol=symbol, ts_ns=ts, meta={"classification_method": "bid_ask"},
                      columns={"buy_volume": np.full(n, 600.0),
                               "sell_volume": np.full(n, 400.0),
                               "unclassified_volume": np.zeros(n)})


def make_options(n=3, symbol="NQ", intraday=False):
    ts = to_ns_array(TS0 + timedelta(days=i) for i in range(n))
    return OptionsSeries(symbol=symbol, ts_ns=ts, meta={"is_intraday": intraday, "source": "stub"},
                         columns={"call_volume": np.full(n, 10.0), "put_volume": np.full(n, 8.0),
                                  "call_premium": np.full(n, 1e6), "put_premium": np.full(n, 9e5),
                                  "call_oi": np.full(n, 1e4), "put_oi": np.full(n, 1.1e4)})


def symbol_data(symbol="NQ", n=20, interval=300, with_extras=True, higher=None):
    bars = {interval: make_bars(n=n, interval=interval, symbol=symbol)}
    for hi in higher or ():
        bars[hi] = make_bars(n=max(2, n // (hi // interval)), interval=hi, symbol=symbol)
    return SymbolData(
        symbol=symbol, primary_interval=interval, bars=bars,
        ticks={interval: make_ticks(n=n, interval=interval, symbol=symbol)} if with_extras else {},
        quotes=make_quotes(n=n, interval=interval, symbol=symbol) if with_extras else None,
        options=make_options(symbol=symbol) if with_extras else None,
    )


class StubAdapter(DataSourceAdapter):
    """Minimal adapter so store tests do not depend on adapters.py."""

    name = "stub"

    def __init__(self, feeds=None, symbol="NQ", n=20):
        self._feeds = feeds if feeds is not None else frozenset(
            {Feed.BARS, Feed.QUOTES, Feed.TICK_AGGREGATE, Feed.OPTIONS_SNAPSHOT})
        self.symbol = symbol
        self.n = n
        self.bar_calls: list[tuple] = []

    def available_feeds(self, symbol):
        return self._feeds

    def load_bars(self, symbol, interval_seconds, start, end):
        self.bar_calls.append((symbol, interval_seconds, start, end))
        if interval_seconds not in (60, 300, 900, 3600):
            raise DataLayerError(f"stub has no {interval_seconds}s bars")
        return make_bars(n=self.n, interval=interval_seconds, symbol=symbol)

    def load_quotes(self, symbol, start, end):
        return make_quotes(n=self.n, symbol=symbol) if Feed.QUOTES in self._feeds else None

    def load_tick_aggregates(self, symbol, interval_seconds, start, end):
        return make_ticks(n=self.n, interval=interval_seconds, symbol=symbol) \
            if Feed.TICK_AGGREGATE in self._feeds else None

    def load_options(self, symbol, start, end):
        return make_options(symbol=symbol) if Feed.OPTIONS_SNAPSHOT in self._feeds else None


# --- SymbolData -----------------------------------------------------------


def test_primary_interval_must_be_loaded():
    with pytest.raises(DataLayerError, match="no bars at the primary interval"):
        SymbolData(symbol="NQ", primary_interval=60, bars={300: make_bars()})


def test_symbol_mismatch_rejected():
    with pytest.raises(DataLayerError, match="does not match"):
        SymbolData(symbol="ES", primary_interval=300, bars={300: make_bars(symbol="NQ")})


def test_registered_interval_must_match_series_interval():
    """Registering a 5-minute series under key 900 would make every
    higher-timeframe feature read the wrong cadence."""
    with pytest.raises(DataLayerError, match="declares"):
        SymbolData(symbol="NQ", primary_interval=300,
                   bars={300: make_bars(interval=300), 900: make_bars(interval=300)})


def test_quote_and_tick_symbol_mismatch_rejected():
    with pytest.raises(DataLayerError, match="does not match"):
        SymbolData(symbol="NQ", primary_interval=300, bars={300: make_bars()},
                   quotes=make_quotes(symbol="ES"))
    with pytest.raises(DataLayerError, match="does not match"):
        SymbolData(symbol="NQ", primary_interval=300, bars={300: make_bars()},
                   ticks={300: make_ticks(symbol="ES")})


def test_available_feeds_reflects_what_is_present():
    assert symbol_data().available_feeds() == frozenset(
        {Feed.BARS, Feed.QUOTES, Feed.TICK_AGGREGATE, Feed.OPTIONS_SNAPSHOT})
    assert symbol_data(with_extras=False).available_feeds() == frozenset({Feed.BARS})


def test_empty_series_do_not_count_as_available_feeds():
    """A present-but-empty series is not a feed."""
    data = SymbolData(symbol="NQ", primary_interval=300, bars={300: make_bars()},
                      quotes=QuoteSeries(symbol="NQ", ts_ns=np.empty(0, dtype=np.int64),
                                         columns={k: np.empty(0) for k in QuoteSeries.REQUIRED}))
    assert Feed.QUOTES not in data.available_feeds()


def test_describe_is_flat_and_informative():
    described = symbol_data(higher=(900,)).describe()
    assert described["bars"] == {300: 20, 900: 6}
    assert described["quotes"] == 20
    assert set(described["feeds"]) == {"bars", "quotes", "tick_aggregate", "options_snapshot"}
    assert isinstance(described["first_ts"], str)


# --- DataStore ------------------------------------------------------------


def test_add_get_and_contains():
    store = DataStore().add(symbol_data("NQ")).add(symbol_data("ES"))
    assert store.symbols == ("ES", "NQ")
    assert "NQ" in store and "CL" not in store
    assert len(store) == 2
    assert store.get("NQ").symbol == "NQ"


def test_missing_symbol_error_lists_loaded_symbols():
    store = DataStore().add(symbol_data("NQ"))
    with pytest.raises(DataLayerError, match=r"loaded symbols: \['NQ'\]"):
        store.get("CL")


def test_negative_store_latency_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        DataStore(latency_seconds=-1.0)


def test_view_is_a_market_view_at_the_requested_instant():
    store = DataStore().add(symbol_data("NQ", n=20))
    now = TS0 + timedelta(seconds=300 * 10)
    view = store.view("NQ", now)
    assert isinstance(view, MarketView)
    assert view.now == now
    assert view.bar_count() == 10
    view.assert_no_future_leak()


def test_store_latency_propagates_into_the_view():
    store = DataStore(latency_seconds=300.0).add(symbol_data("NQ", n=20))
    now = TS0 + timedelta(seconds=300 * 10)
    assert store.view("NQ", now).bar_count() == 9
    assert DataStore().add(symbol_data("NQ", n=20)).view("NQ", now).bar_count() == 10


def test_timeline_is_the_primary_bar_closes():
    store = DataStore().add(symbol_data("NQ", n=20))
    timeline = store.timeline("NQ")
    assert len(timeline) == 20
    assert from_ns(int(timeline[0])) == TS0 + timedelta(seconds=300)
    assert np.all(np.diff(timeline) > 0)


def test_timeline_is_read_only():
    """The clock must not be able to rewrite its own schedule."""
    store = DataStore().add(symbol_data("NQ"))
    with pytest.raises(ValueError):
        store.timeline("NQ")[0] = 0


def test_timeline_for_an_unloaded_interval_raises():
    store = DataStore().add(symbol_data("NQ"))
    with pytest.raises(DataLayerError, match="no bars at 3600s"):
        store.timeline("NQ", interval=3600)


def test_merged_timeline_is_the_sorted_union():
    store = DataStore()
    store.add(symbol_data("NQ", n=10))
    store.add(SymbolData(symbol="ES", primary_interval=300, bars={
        300: make_bars(n=10, start=TS0 + timedelta(seconds=150), symbol="ES")}))
    merged = store.merged_timeline()
    assert len(merged) == 20                     # offset grids do not collide
    assert np.all(np.diff(merged) > 0)


def test_merged_timeline_deduplicates_shared_timestamps():
    store = DataStore().add(symbol_data("NQ", n=10))
    store.add(SymbolData(symbol="ES", primary_interval=300,
                         bars={300: make_bars(n=10, symbol="ES")}))
    assert len(store.merged_timeline()) == 10


def test_merged_timeline_of_an_empty_store_is_empty():
    assert len(DataStore().merged_timeline()) == 0


def test_iter_views_walks_chronologically_and_never_leaks():
    store = DataStore().add(symbol_data("NQ", n=30))
    counts = []
    for index, view in enumerate(store.iter_views("NQ")):
        view.assert_no_future_leak()
        counts.append(view.bar_count())
        assert view.last_bar().close_ts == view.now
        assert index + 1 == view.bar_count()
    assert counts == list(range(1, 31))


# --- loading through the seal guard ---------------------------------------


@pytest.fixture
def guarded(tmp_path):
    config = load_config(overrides=[f"seal.audit_path={tmp_path}/seal_audit.jsonl"])
    return config, guard_from_config(config)


def test_load_populates_every_feed(guarded):
    _, guard = guarded
    store = DataStore(guard=guard)
    adapter = StubAdapter()
    data = store.load(adapter, "NQ", date(2016, 1, 1), date(2017, 1, 1),
                      primary_interval=300, higher_intervals=(900, 3600))
    assert sorted(data.bars) == [300, 900, 3600]
    assert data.quotes is not None and data.options is not None
    assert 300 in data.ticks
    assert "NQ" in store


def test_load_skips_a_higher_interval_the_adapter_lacks(guarded):
    _, guard = guarded
    store = DataStore(guard=guard)
    data = store.load(StubAdapter(), "NQ", date(2016, 1, 1), date(2017, 1, 1),
                      primary_interval=300, higher_intervals=(86400,))
    assert sorted(data.bars) == [300]            # missing interval is skipped, not fatal


def test_load_requires_a_bar_feed(guarded):
    _, guard = guarded
    store = DataStore(guard=guard)
    with pytest.raises(DataLayerError, match="no bar feed"):
        store.load(StubAdapter(feeds=frozenset({Feed.QUOTES})), "NQ",
                   date(2016, 1, 1), date(2017, 1, 1), primary_interval=300)


def test_bar_only_adapter_yields_no_tick_or_options_series(guarded):
    """The honest degradation path: absent feeds are absent, not estimated."""
    _, guard = guarded
    store = DataStore(guard=guard)
    data = store.load(StubAdapter(feeds=frozenset({Feed.BARS})), "NQ",
                      date(2016, 1, 1), date(2017, 1, 1), primary_interval=300)
    assert data.ticks == {}
    assert data.quotes is None and data.options is None
    assert data.available_feeds() == frozenset({Feed.BARS})
    view = store.view("NQ", TS0 + timedelta(seconds=3000))
    assert view.deltas(10).size == 0             # not fabricated from bar volume
    assert view.quote() is None


def test_load_spanning_the_sealed_window_is_blocked(guarded):
    config, guard = guarded
    store = DataStore(guard=guard)
    with pytest.raises(SealedDataAccessError, match="sealed out-of-sample window"):
        store.load(StubAdapter(), "NQ", config.backtest.start, config.backtest.end,
                   primary_interval=300)


def test_clip_to_allowed_trims_the_request_instead_of_raising(guarded):
    config, guard = guarded
    store = DataStore(guard=guard)
    adapter = StubAdapter()
    store.load(adapter, "NQ", config.backtest.start, config.backtest.end,
               primary_interval=300, clip_to_allowed=True)
    _, _, start, end = adapter.bar_calls[0]
    assert start == config.backtest.start
    assert end == config.seal.sealed_start       # stopped at the seal


def test_clip_to_allowed_raises_when_nothing_is_outside_the_seal(guarded):
    config, guard = guarded
    store = DataStore(guard=guard)
    with pytest.raises(DataLayerError, match="entirely\ninside|entirely inside"):
        store.load(StubAdapter(), "NQ", date(2023, 6, 1), date(2024, 6, 1),
                   primary_interval=300, clip_to_allowed=True)


def test_loading_without_a_guard_is_unrestricted():
    store = DataStore()
    data = store.load(StubAdapter(), "NQ", date(2015, 1, 1), date(2025, 1, 1),
                      primary_interval=300)
    assert len(data.primary_bars) == 20


# --- provenance -----------------------------------------------------------


def test_fingerprint_records_source_and_feeds(guarded):
    _, guard = guarded
    store = DataStore(guard=guard)
    data = store.load(StubAdapter(), "NQ", date(2016, 1, 1), date(2017, 1, 1),
                      primary_interval=300)
    fp = data.fingerprint
    assert fp.source == "stub"
    assert fp.rows == 20
    assert Feed.BARS in fp.feeds
    assert len(fp.data_hash) == 16


def test_combined_data_hash_is_stable_and_slice_sensitive(guarded):
    _, guard = guarded

    def build(end):
        store = DataStore(guard=guard)
        store.load(StubAdapter(), "NQ", date(2016, 1, 1), end, primary_interval=300)
        return store.combined_data_hash()

    assert build(date(2017, 1, 1)) == build(date(2017, 1, 1))
    assert build(date(2017, 1, 1)) != build(date(2018, 1, 1))


def test_combined_data_hash_changes_with_symbol_set(guarded):
    _, guard = guarded
    one = DataStore(guard=guard)
    one.load(StubAdapter(), "NQ", date(2016, 1, 1), date(2017, 1, 1), primary_interval=300)
    both = DataStore(guard=guard)
    both.load(StubAdapter(), "NQ", date(2016, 1, 1), date(2017, 1, 1), primary_interval=300)
    both.load(StubAdapter(symbol="ES"), "ES", date(2016, 1, 1), date(2017, 1, 1),
              primary_interval=300)
    assert one.combined_data_hash() != both.combined_data_hash()


def test_store_describe_round_trips_as_json():
    import json

    from flow_model.utils.serialization import dumps

    store = DataStore().add(symbol_data("NQ"))
    assert json.loads(dumps(store.describe()))["symbols"]["NQ"]["primary_interval"] == 300


def test_fingerprints_mapping_covers_every_symbol():
    store = DataStore().add(symbol_data("NQ")).add(symbol_data("ES"))
    assert set(store.fingerprints()) == {"ES", "NQ"}
