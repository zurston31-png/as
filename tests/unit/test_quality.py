"""Quality grading and the feed availability report.

The availability report is the system's honest statement about how much of
the Flow Score is real. These tests exist mostly to ensure it cannot
quietly start saying 100 when the answer is 55.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.core.enums import Component, DataQuality, Feed
from flow_model.data.quality import (
    AvailabilityReport,
    QualityGrader,
    QualityThresholds,
)
from flow_model.data.series import (
    BarSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    from_ns,
    to_ns_array,
)
from flow_model.data.store import DataStore, SymbolData

TS0 = datetime(2020, 1, 2, 14, 35, tzinfo=timezone.utc)
N = 100
INTERVAL = 300


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def grader(config):
    return QualityGrader(config.data, config.flow_score)


def _ts(n=N, interval=INTERVAL):
    return to_ns_array(TS0 + timedelta(seconds=interval * (i + 1)) for i in range(n))


def _bars(n=N, mask=None, interval=INTERVAL):
    ts = _ts(n, interval)
    close = 18000.0 + np.arange(n) * 2.0
    cols = {"open": close - 1, "high": close + 4, "low": close - 4,
            "close": close, "volume": np.full(n, 1000.0)}
    if mask is not None:
        ts = ts[mask]
        cols = {k: v[mask] for k, v in cols.items()}
    return BarSeries(symbol="NQ", ts_ns=ts, interval_seconds=interval, columns=cols)


def _quotes(crossed_every=None):
    ts = _ts()
    bid = np.full(N, 18000.0)
    if crossed_every:
        bid = np.where(np.arange(N) % crossed_every == 0, 18001.0, 18000.0)
    return QuoteSeries(symbol="NQ", ts_ns=ts, columns={
        "bid": bid, "ask": np.full(N, 18000.25),
        "bid_size": np.full(N, 10.0), "ask_size": np.full(N, 10.0)})


def _ticks(classified=1.0):
    ts = _ts()
    total = 1000.0
    classified_volume = total * classified
    return TickSeries(symbol="NQ", ts_ns=ts, meta={"classification_method": "bid_ask"},
                      columns={
                          "buy_volume": np.full(N, classified_volume * 0.6),
                          "sell_volume": np.full(N, classified_volume * 0.4),
                          "unclassified_volume": np.full(N, total - classified_volume)})


def _options(intraday=True):
    ts = _ts()
    return OptionsSeries(symbol="NQ", ts_ns=ts,
                         meta={"is_intraday": intraday, "source": "test"},
                         columns={"call_volume": np.full(N, 10.0), "put_volume": np.full(N, 8.0),
                                  "call_premium": np.full(N, 1e6), "put_premium": np.full(N, 9e5),
                                  "call_oi": np.full(N, 1e4), "put_oi": np.full(N, 1.1e4)})


def _data(bars=None, ticks=True, quotes=True, options=True, intraday=True, mask=None):
    return SymbolData(
        symbol="NQ", primary_interval=INTERVAL,
        bars={INTERVAL: bars if bars is not None else _bars(mask=mask)},
        ticks={INTERVAL: _ticks()} if ticks else {},
        quotes=_quotes() if quotes else None,
        options=_options(intraday) if options else None,
    )


def _view(data, index=-1):
    store = DataStore().add(data)
    stamps = data.primary_bars.ts_ns
    return store.view("NQ", from_ns(int(stamps[index])))


# --- the headline -------------------------------------------------------


def test_bar_only_dataset_reports_55_of_100_points(grader):
    """The single most important number this module produces."""
    report = grader.availability(_data(ticks=False, quotes=False, options=False))
    assert report.available_points == pytest.approx(55.0)
    assert report.unavailable_points == pytest.approx(45.0)
    assert report.fraction_available == pytest.approx(0.55)
    assert set(report.incomputable_components) == {Component.ORDER_FLOW, Component.OPTIONS_FLOW}


def test_bar_only_names_the_missing_feeds(grader):
    report = grader.availability(_data(ticks=False, quotes=False, options=False))
    assert report.components[Component.ORDER_FLOW].missing_required_feeds == (Feed.TICK_AGGREGATE,)
    assert report.components[Component.OPTIONS_FLOW].missing_required_feeds == (
        Feed.OPTIONS_SNAPSHOT,
    )


def test_bar_only_is_not_valid_for_scoring_under_strict_mode(grader):
    report = grader.availability(_data(ticks=False, quotes=False, options=False))
    assert report.strict is True
    assert report.scoring_is_valid is False


def test_summary_states_the_shortfall_and_the_consequence(grader):
    text = "\n".join(grader.availability(_data(ticks=False, quotes=False, options=False)).summary_lines())
    assert "45.0 points have no data feed" in text
    assert "REFUSE" in text
    assert "Do not trust" in text


def test_non_strict_mode_says_scores_are_not_comparable():
    config = load_config(overrides=["flow_score.strict_component_availability=false"])
    grader = QualityGrader(config.data, config.flow_score)
    report = grader.availability(_data(ticks=False, quotes=False, options=False))
    assert report.scoring_is_valid is True            # permitted...
    text = "\n".join(report.summary_lines())
    assert "NOT comparable" in text                    # ...but flagged
    assert report.available_points == pytest.approx(55.0)   # and never rescaled to 100


def test_weight_is_never_redistributed(grader):
    """Redistribution would inflate every score and hide the gap."""
    for data in (_data(ticks=False, quotes=False, options=False), _data(ticks=False)):
        report = grader.availability(data)
        assert report.available_points < report.total_points
        assert sum(a.weight for a in report.components.values()) == pytest.approx(100.0)


def test_full_dataset_reports_all_points(grader):
    report = grader.availability(_data())
    assert report.available_points == pytest.approx(100.0)
    assert report.scoring_is_valid
    assert report.incomputable_components == ()


# --- degradation that is still computable -------------------------------


def test_eod_options_are_degraded_but_computable(grader):
    availability = grader.availability(_data(intraday=False)).components[Component.OPTIONS_FLOW]
    assert availability.computable
    assert availability.quality is DataQuality.DEGRADED
    assert Feed.OPTIONS_SNAPSHOT in availability.degraded_feeds
    assert "flow timing" in availability.note


def test_missing_quotes_degrade_liquidity_without_blocking_it(grader):
    """Quotes are declared required=false for liquidity, so their absence
    must not make the component incomputable."""
    availability = grader.availability(_data(quotes=False)).components[Component.LIQUIDITY]
    assert availability.computable
    assert availability.quality is DataQuality.DEGRADED
    assert Feed.QUOTES in availability.degraded_feeds


def test_disabled_component_is_reported_as_such(grader_config=None):
    config = load_config(overrides=["flow_score.enabled_components.options_flow=false"])
    grader = QualityGrader(config.data, config.flow_score)
    availability = grader.availability(_data()).components[Component.OPTIONS_FLOW]
    assert not availability.computable
    assert "disabled in configuration" in availability.note


def test_as_rows_is_flat_and_serializable(grader):
    import json

    from flow_model.utils.serialization import dumps

    rows = grader.availability(_data(ticks=False)).as_rows()
    assert len(rows) == 5
    restored = json.loads(dumps(rows))
    assert all(not isinstance(v, (dict, list)) for row in restored for v in row.values())


def test_availability_from_store(grader):
    store = DataStore().add(_data())
    reports = grader.availability_from_store(store)
    assert set(reports) == {"NQ"}
    assert isinstance(reports["NQ"], AvailabilityReport)


# --- per-feed grading ---------------------------------------------------


def test_healthy_feeds_grade_good(grader):
    report = grader.grade(_view(_data()))
    assert report.overall is DataQuality.GOOD
    assert report.blocking_feeds == ()
    for feed in Feed:
        assert report.status_of(feed).quality is DataQuality.GOOD


def test_absent_feed_grades_missing(grader):
    report = grader.grade(_view(_data(ticks=False, quotes=False, options=False)))
    assert report.status_of(Feed.TICK_AGGREGATE).quality is DataQuality.MISSING
    assert report.status_of(Feed.BARS).quality is DataQuality.GOOD


def test_staleness_beats_coverage(grader):
    """A complete but frozen feed is not usable."""
    data = _data()
    store = DataStore().add(data)
    late = from_ns(int(data.primary_bars.ts_ns[-1])) + timedelta(hours=2)
    report = grader.grade(store.view("NQ", late))
    assert report.status_of(Feed.BARS).quality is DataQuality.STALE
    assert report.overall is DataQuality.STALE
    assert "frozen feed" in report.status_of(Feed.BARS).note


def test_staleness_threshold_is_configurable():
    config = load_config(overrides=["data.max_staleness_seconds=86400"])
    grader = QualityGrader(config.data, config.flow_score)
    data = _data()
    late = from_ns(int(data.primary_bars.ts_ns[-1])) + timedelta(hours=2)
    report = grader.grade(DataStore().add(data).view("NQ", late))
    assert report.status_of(Feed.BARS).quality is DataQuality.GOOD


def test_gappy_bars_below_the_degraded_floor_grade_missing(grader):
    """A feed 30% absent is not a degraded feed, it is effectively no feed."""
    mask = np.ones(N, dtype=bool)
    mask[::4] = False                      # a quarter of bars removed
    report = grader.grade(_view(_data(mask=mask)))
    status = report.status_of(Feed.BARS)
    assert status.quality is DataQuality.MISSING
    assert status.coverage < 0.80
    assert "effectively no feed" in status.note


def test_mildly_gappy_bars_grade_degraded(grader):
    """Three gaps in the 60-bar window gives ~0.95 contiguity: below the
    0.98 GOOD cutoff but above the 0.80 floor. (One gap gives 0.983, which
    is correctly still GOOD.)"""
    mask = np.ones(N, dtype=bool)
    mask[[50, 60, 70]] = False
    report = grader.grade(_view(_data(mask=mask)))
    status = report.status_of(Feed.BARS)
    assert status.quality is DataQuality.DEGRADED
    assert 0.80 <= status.coverage < 0.98


def test_a_single_gap_remains_good(grader):
    """Pins the boundary so the thresholds cannot drift unnoticed."""
    mask = np.ones(N, dtype=bool)
    mask[50] = False
    status = grader.grade(_view(_data(mask=mask))).status_of(Feed.BARS)
    assert status.quality is DataQuality.GOOD
    assert status.coverage >= 0.98


def test_poor_tick_classification_grades_missing(grader):
    data = SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: _bars()},
                      ticks={INTERVAL: _ticks(classified=0.50)})
    status = grader.grade(_view(data)).status_of(Feed.TICK_AGGREGATE)
    assert status.quality is DataQuality.MISSING
    assert status.coverage == pytest.approx(0.50)


def test_partial_tick_classification_grades_degraded(grader):
    data = SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: _bars()},
                      ticks={INTERVAL: _ticks(classified=0.75)})
    assert grader.grade(_view(data)).status_of(Feed.TICK_AGGREGATE).quality is DataQuality.DEGRADED


def test_crossed_quotes_degrade_the_quote_feed(grader):
    data = SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: _bars()},
                      quotes=_quotes(crossed_every=10))
    status = grader.grade(_view(data)).status_of(Feed.QUOTES)
    assert status.quality is DataQuality.DEGRADED
    assert "crossed-quote rate" in status.note


def test_eod_options_grade_degraded_at_a_point_in_time(grader):
    status = grader.grade(_view(_data(intraday=False))).status_of(Feed.OPTIONS_SNAPSHOT)
    assert status.quality is DataQuality.DEGRADED
    assert "flow timing" in status.note


# --- overall and blocking ----------------------------------------------


def test_overall_is_the_worst_required_feed(grader):
    data = SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: _bars()},
                      ticks={INTERVAL: _ticks(classified=0.75)}, quotes=_quotes(),
                      options=_options(True))
    report = grader.grade(_view(data))
    assert report.overall is DataQuality.DEGRADED


def test_an_optional_feed_does_not_block(grader):
    """Quotes are optional for liquidity, so degrading them must not stop
    trading."""
    data = SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: _bars()},
                      ticks={INTERVAL: _ticks()}, quotes=_quotes(crossed_every=2),
                      options=_options(True))
    report = grader.grade(_view(data))
    assert Feed.QUOTES not in report.blocking_feeds
    assert report.is_tradable(DataQuality.DEGRADED)


def test_a_required_feed_blocks(grader):
    data = SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: _bars()},
                      ticks={INTERVAL: _ticks(classified=0.3)}, quotes=_quotes(),
                      options=_options(True))
    report = grader.grade(_view(data))
    assert Feed.TICK_AGGREGATE in report.blocking_feeds
    assert not report.is_tradable(DataQuality.DEGRADED)
    assert "tick_aggregate" in report.note


def test_required_feeds_excludes_disabled_components():
    config = load_config(overrides=["flow_score.enabled_components.order_flow=false"])
    grader = QualityGrader(config.data, config.flow_score)
    assert Feed.TICK_AGGREGATE not in grader.required_feeds()
    assert Feed.BARS in grader.required_feeds()


def test_report_summary_lines_and_unknown_feed(grader):
    report = grader.grade(_view(_data()))
    text = "\n".join(report.summary_lines())
    assert "overall=GOOD" in text
    assert "bars" in text


# --- thresholds ---------------------------------------------------------


def test_thresholds_reject_inverted_ordering():
    with pytest.raises(ValueError, match="min_coverage_degraded exceeds"):
        QualityThresholds(min_coverage_good=0.5, min_coverage_degraded=0.9)
    with pytest.raises(ValueError, match="min_tick_classification_degraded exceeds"):
        QualityThresholds(min_tick_classification_good=0.5,
                          min_tick_classification_degraded=0.9)


def test_session_breaks_are_excluded_from_coverage(grader):
    """Without this, every overnight break would read as missing data and
    coverage would mean nothing."""
    day_one = [TS0 + timedelta(seconds=INTERVAL * (i + 1)) for i in range(40)]
    day_two = [TS0 + timedelta(days=1, seconds=INTERVAL * (i + 1)) for i in range(40)]
    ts = to_ns_array(day_one + day_two)
    n = len(ts)
    close = 18000.0 + np.arange(n) * 2.0
    bars = BarSeries(symbol="NQ", ts_ns=ts, interval_seconds=INTERVAL, columns={
        "open": close - 1, "high": close + 4, "low": close - 4,
        "close": close, "volume": np.full(n, 1000.0)})
    data = SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: bars})
    status = grader.grade(_view(data)).status_of(Feed.BARS)
    assert status.quality is DataQuality.GOOD
    assert status.coverage == pytest.approx(1.0)


def test_a_grader_that_always_says_good_would_be_useless(grader):
    """Sanity check on the whole module: a deliberately bad dataset must
    produce a non-GOOD overall grade. A grader supplying false assurance is
    worse than no grader."""
    mask = np.ones(N, dtype=bool)
    mask[::4] = False
    data = SymbolData(symbol="NQ", primary_interval=INTERVAL,
                      bars={INTERVAL: _bars(mask=mask)},
                      ticks={INTERVAL: _ticks(classified=0.5)},
                      quotes=_quotes(crossed_every=10), options=_options(False))
    report = grader.grade(_view(data))
    assert report.overall is not DataQuality.GOOD
    assert report.blocking_feeds
