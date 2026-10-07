"""Bar cleaning, with emphasis on causality and the no-fabrication rule."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.core.enums import InstrumentType
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.base import SchemaError
from flow_model.data.clean import (
    DUPLICATE_POLICY_KEEP_LAST,
    MAD_TO_SIGMA,
    BarCleaner,
    detect_gaps,
)
from flow_model.data.series import NS_PER_SECOND, BarSeries, to_ns, to_ns_array

TS0 = datetime(2020, 1, 2, 14, 35, tzinfo=timezone.utc)   # 09:35 ET, on the 5-min grid
INTERVAL = 300


@pytest.fixture
def cleaner():
    return BarCleaner(load_config().data)


@pytest.fixture
def nq():
    return InstrumentSpec(
        symbol="NQ", instrument_type=InstrumentType.FUTURE, tick_size=0.25, tick_value=5.0,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
        eth=SessionWindow(name="ETH", start="18:00", end="17:00"),
    )


def raw(n=120, start=TS0, interval=INTERVAL, seed=3, drift=0.0):
    """Well-formed raw arrays on the interval grid."""
    rng = np.random.default_rng(seed)
    ts = np.array([to_ns(start + timedelta(seconds=interval * (i + 1))) for i in range(n)],
                  dtype=np.int64)
    close = 18000.0 * np.exp(np.cumsum(rng.normal(drift, 0.0008, n)))
    wiggle = np.abs(rng.normal(0, 0.0004, n)) * close
    prev = np.concatenate(([18000.0], close[:-1]))
    return ts, {
        "open": prev,
        "high": np.maximum(prev, close) + wiggle,
        "low": np.minimum(prev, close) - wiggle,
        "close": close,
        "volume": rng.integers(500, 5000, n).astype(np.float64),
    }


def clean_raw(cleaner, ts, cols, **kw):
    return cleaner.clean_raw(symbol="NQ", interval_seconds=INTERVAL, ts_ns=ts, columns=cols, **kw)


# --- pass-through ---------------------------------------------------------


def test_clean_series_passes_through_unchanged(cleaner):
    ts, cols = raw()
    series, report = clean_raw(cleaner, ts, cols)
    assert report.rows_in == report.rows_out == len(ts)
    assert report.retention == pytest.approx(1.0)
    assert report.rows_removed == 0
    assert np.array_equal(series.col("close"), cols["close"])
    assert np.array_equal(series.ts_ns, ts)


def test_clean_accepts_a_validated_series(cleaner):
    ts, cols = raw()
    original = BarSeries(symbol="NQ", ts_ns=ts, interval_seconds=INTERVAL, columns=cols)
    series, report = cleaner.clean(original)
    # BarSeries rejects these at construction, so they cannot be non-zero here.
    assert report.duplicates_dropped == 0
    assert report.out_of_order_dropped == 0
    assert report.malformed_dropped == 0
    assert len(series) == len(original)


# --- malformed ------------------------------------------------------------


@pytest.mark.parametrize(
    "column,value",
    [("high", 0.0), ("low", 1e9), ("close", float("nan")), ("open", float("inf")),
     ("volume", -5.0), ("volume", float("nan"))],
)
def test_malformed_rows_are_dropped_and_counted(cleaner, column, value):
    ts, cols = raw(n=60)
    cols[column] = cols[column].copy()
    cols[column][30] = value
    series, report = clean_raw(cleaner, ts, cols)
    assert report.malformed_dropped == 1
    assert report.rows_out == 59
    assert to_ns(TS0 + timedelta(seconds=INTERVAL * 31)) not in series.ts_ns.tolist()


def test_series_that_is_entirely_malformed_yields_an_empty_report_not_an_exception(cleaner):
    ts, cols = raw(n=30)
    cols["high"] = np.zeros(30)          # every bar violates OHLC ordering
    series, report = clean_raw(cleaner, ts, cols)
    assert len(series) == 0
    assert report.rows_out == 0
    assert report.malformed_dropped == 30
    assert report.retention == 0.0


# --- duplicates and ordering ---------------------------------------------


def test_duplicate_timestamps_keep_the_last_row(cleaner):
    """The policy is on the VALUES, not just the count: a repeated timestamp
    in vendor data is normally a corrected revision of the first."""
    assert DUPLICATE_POLICY_KEEP_LAST
    ts, cols = raw(n=40)
    ts = np.insert(ts, 20, ts[20])
    for name in cols:
        cols[name] = np.insert(cols[name], 20, cols[name][20])
    cols["close"][20] = 17000.0          # the stale first copy
    cols["high"][20] = 17100.0
    cols["low"][20] = 16900.0
    cols["open"][20] = 17000.0
    series, report = clean_raw(cleaner, ts, cols)
    assert report.duplicates_dropped == 1
    assert report.rows_out == 40
    position = series.ts_ns.tolist().index(int(ts[20]))
    assert series.col("close")[position] != pytest.approx(17000.0)


def test_interleaved_row_is_dropped_and_the_in_order_copy_kept(cleaner):
    """A non-adjacent repeat means the export is interleaved. The in-order
    copy is the good one; only the late row is dropped. A global keep-last
    policy would drop both and lose the bar entirely."""
    ts, cols = raw(n=40)
    ts = ts.copy()
    ts[25] = ts[10]                       # a row from earlier, appearing late
    series, report = clean_raw(cleaner, ts, cols)
    assert report.out_of_order_dropped == 1
    assert report.duplicates_dropped == 0
    assert report.rows_out == 39
    assert int(ts[10]) in series.ts_ns.tolist()     # in-order copy survived
    assert np.all(np.diff(series.ts_ns) > 0)
    # dropped, not relocated: no sorting happened
    assert series.col("close")[10] == pytest.approx(cols["close"][10])


def test_output_is_always_strictly_ascending(cleaner):
    ts, cols = raw(n=50)
    ts = ts.copy()
    ts[10], ts[11] = ts[11], ts[10]
    series, _ = clean_raw(cleaner, ts, cols)
    assert np.all(np.diff(series.ts_ns) > 0)


# --- partial bars ---------------------------------------------------------


def test_trailing_off_grid_bar_is_dropped(cleaner):
    ts, cols = raw(n=40)
    ts = ts.copy()
    ts[-1] += 97 * NS_PER_SECOND          # 97 seconds into the next bar
    series, report = clean_raw(cleaner, ts, cols)
    assert report.partial_bars_dropped == 1
    assert report.rows_out == 39
    assert any("off the interval grid" in note for note in report.notes)


def test_partial_bar_dropping_can_be_disabled():
    cleaner = BarCleaner(load_config(overrides=["data.drop_partial_bars=false"]).data)
    ts, cols = raw(n=40)
    ts = ts.copy()
    ts[-1] += 97 * NS_PER_SECOND
    _, report = clean_raw(cleaner, ts, cols)
    assert report.partial_bars_dropped == 0


def test_grid_phase_is_derived_from_the_series_not_assumed(cleaner):
    """A session opening at 08:20 puts bars on a different phase than 09:30.
    Deriving the grid from the data makes partial detection session-agnostic."""
    gold_open = datetime(2020, 1, 2, 13, 20, tzinfo=timezone.utc)   # 08:20 ET
    ts, cols = raw(n=40, start=gold_open)
    _, report = clean_raw(cleaner, ts, cols)
    assert report.partial_bars_dropped == 0          # on-grid for ITS own grid


def test_a_mid_series_off_grid_bar_is_not_treated_as_partial(cleaner):
    ts, cols = raw(n=40)
    ts = ts.copy()
    ts[20] += 11 * NS_PER_SECOND
    _, report = clean_raw(cleaner, ts, cols)
    assert report.partial_bars_dropped == 0          # only the FINAL bar qualifies


# --- outliers: the causality guarantee -----------------------------------


def _spike(ts, cols, index, factor=1.25):
    cols = {k: v.copy() for k, v in cols.items()}
    for name in ("open", "high", "low", "close"):
        cols[name][index] *= factor
    cols["high"][index] = max(cols["high"][index], cols["open"][index], cols["close"][index])
    cols["low"][index] = min(cols["low"][index], cols["open"][index], cols["close"][index])
    return ts, cols


def test_a_price_spike_is_quarantined_and_its_timestamp_recorded(cleaner):
    ts, cols = _spike(*raw(n=120), index=80)
    series, report = clean_raw(cleaner, ts, cols)
    assert report.outliers_quarantined >= 1
    assert report.quarantined_timestamps
    assert len(series) == report.rows_out < 120


def test_outlier_detection_is_causal(cleaner):
    """THE test for this module.

    Append a huge spike to the END of a series. The survival of every
    EARLIER bar must be identical to cleaning the series without it. A
    full-sample or centred dispersion estimate would fail this, and the
    surviving dataset would encode future information.
    """
    ts, cols = raw(n=160)
    without, _ = clean_raw(cleaner, ts, cols)

    ts2, cols2 = _spike(ts, cols, index=159, factor=1.5)
    with_spike, _ = clean_raw(cleaner, ts2, cols2)

    boundary = int(ts[158])
    kept_before = [t for t in without.ts_ns.tolist() if t <= boundary]
    kept_before_with = [t for t in with_spike.ts_ns.tolist() if t <= boundary]
    assert kept_before == kept_before_with


def test_adding_future_bars_never_changes_earlier_survival(cleaner):
    """Stronger form: extending the series must not retroactively
    reclassify any bar already seen."""
    ts, cols = raw(n=200)
    short, _ = clean_raw(cleaner, ts[:120], {k: v[:120] for k, v in cols.items()})
    full, _ = clean_raw(cleaner, ts, cols)
    boundary = int(ts[119])
    assert short.ts_ns.tolist() == [t for t in full.ts_ns.tolist() if t <= boundary]


def test_early_bars_are_never_quarantined_for_lack_of_history(cleaner):
    """There is no trailing window to judge them against, and judging them
    on later data is the lookahead this module avoids."""
    ts, cols = _spike(*raw(n=120), index=3)
    series, report = clean_raw(cleaner, ts, cols)
    assert int(ts[3]) in series.ts_ns.tolist()
    assert from_ts_list(report.quarantined_timestamps, ts[3]) is False


def from_ts_list(stamps, target_ns) -> bool:
    from flow_model.data.series import to_ns as _to_ns

    return any(_to_ns(s) == int(target_ns) for s in stamps)


def test_a_flat_series_is_not_quarantined(cleaner):
    """Zero dispersion offers nothing to judge against, so the cleaner does
    not guess rather than dividing by zero."""
    n = 120
    ts = to_ns_array(TS0 + timedelta(seconds=INTERVAL * (i + 1)) for i in range(n))
    flat = np.full(n, 18000.0)
    series, report = clean_raw(cleaner, ts, {
        "open": flat, "high": flat, "low": flat, "close": flat,
        "volume": np.full(n, 1000.0)})
    assert report.outliers_quarantined == 0
    assert len(series) == n


def test_outlier_sigma_is_configurable():
    ts, cols = _spike(*raw(n=160), index=100, factor=1.08)
    strict = BarCleaner(load_config(overrides=["data.outlier_sigma=3.0"]).data)
    loose = BarCleaner(load_config(overrides=["data.outlier_sigma=50.0"]).data)
    _, strict_report = clean_raw(strict, ts, cols)
    _, loose_report = clean_raw(loose, ts, cols)
    assert strict_report.outliers_quarantined >= loose_report.outliers_quarantined


def test_mad_scale_constant():
    assert MAD_TO_SIGMA == pytest.approx(1.4826)


# --- zero volume ----------------------------------------------------------


def test_zero_volume_bars_are_flagged_but_retained(cleaner):
    """A legitimately quiet bar is information; deleting it distorts the
    time axis."""
    ts, cols = raw(n=60)
    cols["volume"] = cols["volume"].copy()
    cols["volume"][10] = 0.0
    cols["volume"][11] = 0.0
    series, report = clean_raw(cleaner, ts, cols)
    assert report.zero_volume_flagged == 2
    assert report.rows_out == 60
    assert 0.0 in series.col("volume").tolist()


# --- gaps: detected, never filled ----------------------------------------


def test_gaps_are_detected_and_counted(cleaner):
    ts, cols = raw(n=60)
    mask = np.ones(60, dtype=bool)
    mask[20:23] = False                   # three missing bars
    ts, cols = ts[mask], {k: v[mask] for k, v in cols.items()}
    _, report = clean_raw(cleaner, ts, cols)
    assert report.gaps_detected == 1
    assert report.largest_gap_bars == 3


def test_no_bar_is_ever_fabricated(cleaner):
    """The output may only contain timestamps present in the input."""
    ts, cols = raw(n=80)
    mask = np.ones(80, dtype=bool)
    mask[30:36] = False
    ts, cols = ts[mask], {k: v[mask] for k, v in cols.items()}
    series, report = clean_raw(cleaner, ts, cols)
    assert set(series.ts_ns.tolist()) <= set(ts.tolist())
    assert report.gaps_detected == 1
    assert len(series) <= len(ts)


def test_detect_gaps_reports_run_lengths():
    ts = to_ns_array(TS0 + timedelta(seconds=INTERVAL * i) for i in [1, 2, 3, 8, 9, 15])
    gaps = detect_gaps(ts, INTERVAL)
    assert [count for _, _, count in gaps] == [4, 5]


def test_detect_gaps_on_short_series_is_empty():
    assert detect_gaps(np.zeros(0, dtype=np.int64), INTERVAL) == []
    assert detect_gaps(np.array([to_ns(TS0)], dtype=np.int64), INTERVAL) == []


def test_overnight_break_is_not_a_data_gap(cleaner, nq):
    """Without a calendar every session boundary would be reported, and the
    gap count would mean nothing."""
    from flow_model.data.calendar import TradingCalendar

    calendar = TradingCalendar()
    day_one = [TS0 + timedelta(seconds=INTERVAL * (i + 1)) for i in range(20)]
    day_two = [TS0 + timedelta(days=1, seconds=INTERVAL * (i + 1)) for i in range(20)]
    stamps = day_one + day_two
    n = len(stamps)
    rng = np.random.default_rng(5)
    close = 18000.0 + np.cumsum(rng.normal(0, 3.0, n))
    prev = np.concatenate(([18000.0], close[:-1]))
    cols = {
        "open": prev, "high": np.maximum(prev, close) + 2.0,
        "low": np.minimum(prev, close) - 2.0, "close": close,
        "volume": np.full(n, 1000.0),
    }
    ts = to_ns_array(stamps)

    _, with_calendar = clean_raw(cleaner, ts, cols, spec=nq, calendar=calendar)
    _, without_calendar = clean_raw(cleaner, ts, cols)
    assert with_calendar.gaps_detected == 0
    assert without_calendar.gaps_detected == 1
    assert any("upper bound" in note for note in without_calendar.notes)


# --- edges ----------------------------------------------------------------


def test_empty_input(cleaner):
    series, report = clean_raw(cleaner, np.zeros(0, dtype=np.int64),
                               {k: np.zeros(0) for k in BarSeries.REQUIRED})
    assert len(series) == 0
    assert report.rows_in == 0 and report.rows_out == 0
    assert report.retention == 0.0


def test_single_bar_input(cleaner):
    ts, cols = raw(n=1)
    series, report = clean_raw(cleaner, ts, cols)
    assert len(series) == 1
    assert report.partial_bars_dropped == 0      # cannot infer a grid from one bar


def test_missing_required_column_raises(cleaner):
    ts, cols = raw(n=10)
    del cols["volume"]
    with pytest.raises(SchemaError, match="missing required columns"):
        clean_raw(cleaner, ts, cols)


def test_column_length_mismatch_raises(cleaner):
    ts, cols = raw(n=10)
    cols["close"] = cols["close"][:5]
    with pytest.raises(SchemaError, match="rows but ts has"):
        clean_raw(cleaner, ts, cols)


def test_report_is_json_serializable(cleaner):
    import json

    from flow_model.utils.serialization import dumps

    ts, cols = _spike(*raw(n=120), index=80)
    _, report = clean_raw(cleaner, ts, cols)
    restored = json.loads(dumps(report))
    assert restored["rows_in"] == 120
    assert "rows_removed" in restored and "retention" in restored


# ---------------------------------------------------------------------------
# Regressions for the Phase 2 adversarial audit
# ---------------------------------------------------------------------------


def test_gaps_are_measured_on_the_input_not_the_output(cleaner):
    """HIGH finding: gaps were detected on the post-quarantine output, so the
    cleaner reported its own removals as defects in the data. A pristine
    generated dataset came back claiming 60 gaps, every one a hole the
    outlier quarantine had just made."""
    ts, cols = _spike(*raw(n=200), index=120, factor=1.4)
    _, report = clean_raw(cleaner, ts, cols)
    assert report.outliers_quarantined >= 1
    assert report.gaps_detected == 0                       # the INPUT had none
    assert report.gaps_introduced_by_cleaning >= 1         # the cleaner made one


def test_the_two_gap_causes_are_counted_separately(cleaner):
    ts, cols = raw(n=200)
    mask = np.ones(200, dtype=bool)
    mask[50:53] = False                                    # a real hole in the feed
    ts, cols = ts[mask], {k: v[mask] for k, v in cols.items()}
    ts, cols = _spike(ts, cols, index=150, factor=1.4)      # plus a spike to quarantine
    _, report = clean_raw(cleaner, ts, cols)
    assert report.gaps_detected == 1                        # the feed's fault
    assert report.gaps_introduced_by_cleaning >= 1          # the cleaner's
    assert report.largest_gap_bars == 3


def test_disabling_the_quarantine_leaves_both_gap_counts_at_zero():
    """Establishes causation for the finding above."""
    loose = BarCleaner(load_config(overrides=["data.outlier_sigma=1e9"]).data)
    ts, cols = raw(n=200)
    _, report = clean_raw(loose, ts, cols)
    assert report.outliers_quarantined == 0
    assert report.gaps_detected == 0
    assert report.gaps_introduced_by_cleaning == 0


def test_provenance_meta_survives_cleaning(cleaner):
    """HIGH finding: the cleaner discarded the input series' meta, destroying
    the `source` tag and the adapter's `timestamp_is_bar_open` convention --
    the one irreversible provenance fact about a dataset."""
    ts, cols = raw(n=60)
    original = BarSeries(
        symbol="NQ", ts_ns=ts, interval_seconds=INTERVAL, columns=cols,
        meta={"source": "csv", "path": "/data/NQ_300s.csv", "timestamp_is_bar_open": True},
    )
    cleaned, _ = cleaner.clean(original)
    assert cleaned.meta["source"] == "csv"
    assert cleaned.meta["path"] == "/data/NQ_300s.csv"
    assert cleaned.meta["timestamp_is_bar_open"] is True
    assert cleaned.meta["cleaned"] is True
    assert cleaned.interval_seconds == INTERVAL


def test_detect_gaps_requires_a_calendar_that_implements_the_protocol(cleaner, nq):
    """HIGH finding: detect_gaps called session_date, which the published
    protocol did not declare, and swallowed the AttributeError -- so a
    conforming calendar silently produced the no-calendar answer while the
    report claimed otherwise. session_date is now part of the protocol."""
    from flow_model.data.base import SessionCalendarProtocol

    class WithoutSessionDate:
        def is_trading_day(self, day, spec): return True
        def is_half_day(self, day, spec): return False
        def session_of(self, ts, spec): return None
        def is_rth(self, ts, spec): return True

    assert not isinstance(WithoutSessionDate(), SessionCalendarProtocol)
    from flow_model.data.calendar import TradingCalendar

    assert isinstance(TradingCalendar(), SessionCalendarProtocol)

    ts = to_ns_array(TS0 + timedelta(seconds=INTERVAL * i) for i in (1, 2, 3, 8))
    with pytest.raises(AttributeError):
        detect_gaps(ts, INTERVAL, spec=nq, calendar=WithoutSessionDate())


def test_outlier_detection_scored_against_synthetic_ground_truth():
    """Scores the detector against the generator's injected outliers.

    The synthetic generator records which bars it corrupted, which is what
    those labels exist for. Pinning precision and recall here turns two
    numbers that were previously unexamined into a stated operating point:
    a cascade fix and a window widening moved precision from 0.022 (a
    runaway rejection bug) through 0.52 to 0.80.

    Recall is deliberately below 1.0: a 12-sigma print injected during a
    HIGH_VOL regime is genuinely within 10 robust sigma of the local
    dispersion, and flagging it would require either a tighter threshold
    (which costs far more false positives) or knowledge of the regime the
    detector does not have at that bar.
    """
    from flow_model.data import SyntheticConfig, SyntheticMarketGenerator, TradingCalendar
    from flow_model.data.series import to_ns as _to_ns

    config = load_config()
    calendar = TradingCalendar()
    spec = config.spec("NQ")
    generator = SyntheticMarketGenerator(
        SyntheticConfig(outlier_probability=0.002, outlier_sigma_multiple=12.0), seed=7
    )
    dataset = generator.generate(
        "NQ", spec, date(2016, 1, 1), date(2016, 7, 1), 300, calendar=calendar
    )
    truth = {int(dataset.bars.ts_ns[i]) for i in dataset.outlier_indices}
    assert len(truth) >= 10, "fixture must inject enough outliers to score"

    _, report = BarCleaner(config.data).clean(dataset.bars, spec=spec, calendar=calendar)
    flagged = {_to_ns(t) for t in report.quarantined_timestamps}

    true_positives = len(truth & flagged)
    precision = true_positives / max(1, len(flagged))
    recall = true_positives / len(truth)

    assert recall >= 0.70, f"recall {recall:.3f} regressed"
    assert precision >= 0.70, f"precision {precision:.3f} regressed"


def test_a_clean_dataset_produces_few_false_positives():
    """The companion to the above: the detector must not invent outliers.

    A 50-bar window produced 11 false positives here, all at volatility
    regime transitions, because the generator's LOW_VOL to HIGH_VOL ratio is
    5.5x and the dispersion estimate had not caught up.
    """
    from flow_model.data import SyntheticConfig, SyntheticMarketGenerator, TradingCalendar

    config = load_config()
    calendar = TradingCalendar()
    spec = config.spec("NQ")
    dataset = SyntheticMarketGenerator(
        SyntheticConfig(outlier_probability=0.0), seed=7
    ).generate("NQ", spec, date(2016, 1, 1), date(2016, 7, 1), 300, calendar=calendar)

    _, report = BarCleaner(config.data).clean(dataset.bars, spec=spec, calendar=calendar)
    rate = report.outliers_quarantined / len(dataset.bars)
    assert report.outliers_quarantined <= 5, (
        f"{report.outliers_quarantined} false positives on data with no injected "
        f"outliers (rate {rate:.5f})"
    )


def test_a_rejected_bar_does_not_cascade():
    """A stale anchor across an unbounded run of rejections rejects the rest
    of a trending series: 536 false positives on 16 injected spikes, measured.
    Consecutive rejections are capped at one."""
    ts, cols = raw(n=400, drift=0.0015)       # a persistent uptrend
    cols = {k: v.copy() for k, v in cols.items()}
    for name in ("open", "high", "low", "close"):
        cols[name][200] *= 1.35               # one bad print mid-trend
    cols["high"][200] = max(cols["high"][200], cols["open"][200], cols["close"][200])
    cols["low"][200] = min(cols["low"][200], cols["open"][200], cols["close"][200])
    _, report = clean_raw(cleaner_for(load_config().data), ts, cols)
    assert report.outliers_quarantined <= 3, (
        f"{report.outliers_quarantined} bars quarantined from one bad print: the "
        "anchor cascaded"
    )


def cleaner_for(data_config):
    return BarCleaner(data_config)
