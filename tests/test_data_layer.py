"""Candles, data quality gates, and providers.

The look-ahead guard in `up_to` is the most important thing here: if it is
wrong, every backtest result is worthless in a way that looks like skill.
"""
import datetime as dt

import pytest

from app.data.candles import Candle, CandleSeries, Timeframe
from app.data.providers import (
    CsvCandleProvider, SyntheticCandleProvider, _parse_timestamp, resample,
)
from app.data.quality import assess_quality

UTC = dt.timezone.utc
START = dt.datetime(2026, 1, 1, tzinfo=UTC)


def make_series(n=100, timeframe=Timeframe.H1, start_price=100.0, **kw):
    step = kw.get("step", 0.0)
    volume = kw.get("volume", 1000.0)
    candles = []
    price = start_price
    for i in range(n):
        candles.append(Candle(
            timestamp=START + dt.timedelta(seconds=timeframe.seconds * i),
            open=price, high=price * 1.01, low=price * 0.99,
            close=price * (1 + step), volume=volume,
        ))
        price *= (1 + step) if step else 1.0
    return CandleSeries("TEST", timeframe, candles)


def now_after(series):
    """A 'now' one interval past the last candle, so nothing looks stale."""
    return series.last.timestamp + dt.timedelta(seconds=series.timeframe.seconds)


# ---------------------------------------------------------------------------
# candles
# ---------------------------------------------------------------------------

def test_candle_requires_a_timezone():
    with pytest.raises(ValueError, match="timezone-aware"):
        Candle(dt.datetime(2026, 1, 1), 1, 2, 0.5, 1.5, 100)


def test_structural_validation_catches_impossible_bars():
    ok = Candle(START, open=10, high=12, low=9, close=11, volume=5)
    assert ok.is_structurally_valid()

    high_below_low = Candle(START, open=10, high=8, low=9, close=8.5, volume=5)
    assert not high_below_low.is_structurally_valid()

    close_outside_range = Candle(START, open=10, high=11, low=9, close=15, volume=5)
    assert not close_outside_range.is_structurally_valid()

    negative_volume = Candle(START, open=10, high=11, low=9, close=10, volume=-1)
    assert not negative_volume.is_structurally_valid()


def test_series_sorts_oldest_first_regardless_of_input_order():
    late = Candle(START + dt.timedelta(hours=1), 1, 2, 0.5, 1.5, 10)
    early = Candle(START, 1, 2, 0.5, 1.5, 10)
    series = CandleSeries("T", Timeframe.H1, [late, early])
    assert series[0].timestamp < series[1].timestamp


def test_column_views_align_with_the_candles():
    series = make_series(5)
    assert len(series.closes) == len(series) == 5
    assert series.closes[-1] == series[-1].close
    assert series.last_price == series[-1].close


# ---------------------------------------------------------------------------
# look-ahead guard
# ---------------------------------------------------------------------------

def test_up_to_excludes_the_bar_still_forming():
    """A bar stamped with its open time has not closed until one interval
    later. Including it would let a backtest see the future."""
    series = make_series(10, Timeframe.H1)
    third_open = series[2].timestamp

    visible = series.up_to(third_open)
    # Only bars 0 and 1 have finished by the moment bar 2 opens.
    assert len(visible) == 2
    assert visible.last.timestamp == series[1].timestamp


def test_up_to_never_leaks_a_future_candle():
    series = make_series(50, Timeframe.H1)
    for i in range(5, 50):
        cutoff = series[i].timestamp
        for candle in series.up_to(cutoff):
            assert candle.timestamp < cutoff


def test_up_to_on_the_earliest_timestamp_is_empty():
    series = make_series(10)
    assert len(series.up_to(series[0].timestamp)) == 0


def test_split_is_chronological_and_lossless():
    series = make_series(100)
    train, validate, test = series.split(0.5, 0.25, 0.25)

    assert len(train) + len(validate) + len(test) == 100
    assert train.last.timestamp < validate[0].timestamp
    assert validate.last.timestamp < test[0].timestamp


def test_split_rejects_nonsense_fractions():
    with pytest.raises(ValueError):
        make_series(10).split(0, 0)


# ---------------------------------------------------------------------------
# quality gates
# ---------------------------------------------------------------------------

def test_clean_series_is_tradeable():
    series = make_series(100, step=0.001)
    report = assess_quality(series, now=now_after(series))
    assert report.tradeable, report.summary
    assert report.gaps == 0 and report.duplicates == 0


def test_empty_series_is_rejected():
    report = assess_quality(CandleSeries("T", Timeframe.H1, []))
    assert not report.tradeable
    assert "no candles" in report.summary


def test_too_few_candles_is_rejected():
    series = make_series(10, step=0.001)
    report = assess_quality(series, min_candles=50, now=now_after(series))
    assert not report.tradeable
    assert "need 50" in report.summary


def test_duplicate_timestamps_are_rejected():
    series = make_series(60, step=0.001)
    duped = CandleSeries("T", Timeframe.H1, series.candles + [series[10]])
    report = assess_quality(duped, now=now_after(series))
    assert not report.tradeable
    assert report.duplicates == 1
    assert "duplicate" in report.summary


def test_a_small_gap_warns_but_still_trades():
    series = make_series(100, step=0.001)
    kept = [c for i, c in enumerate(series.candles) if i != 50]
    report = assess_quality(CandleSeries("T", Timeframe.H1, kept), now=now_after(series))
    assert report.tradeable, report.summary
    assert report.gaps == 1
    assert any("gap" in w for w in report.warnings)


def test_a_large_gap_is_rejected():
    series = make_series(100, step=0.001)
    kept = series.candles[:40] + series.candles[80:]
    report = assess_quality(CandleSeries("T", Timeframe.H1, kept), now=now_after(series))
    assert not report.tradeable
    assert "absent" in report.summary


def test_stale_feed_is_rejected():
    series = make_series(100, step=0.001)
    late = series.last.timestamp + dt.timedelta(days=3)
    report = assess_quality(series, now=late)
    assert not report.tradeable
    assert "stale" in report.summary


def test_malformed_candles_are_rejected():
    series = make_series(60, step=0.001)
    bad = Candle(
        series.last.timestamp + dt.timedelta(hours=1),
        open=10, high=5, low=9, close=7, volume=100,   # high below low
    )
    broken = CandleSeries("T", Timeframe.H1, series.candles + [bad])
    report = assess_quality(broken, now=now_after(broken))
    assert not report.tradeable
    assert report.malformed == 1
    assert "malformed" in report.summary


def test_frozen_price_is_rejected():
    """A feed repeating the same close is broken, not a calm market."""
    series = make_series(100, step=0.0)
    report = assess_quality(series, now=now_after(series))
    assert not report.tradeable
    assert "frozen" in report.summary


def test_zero_volume_is_rejected():
    series = make_series(100, step=0.001, volume=0.0)
    report = assess_quality(series, now=now_after(series))
    assert not report.tradeable
    assert "zero volume" in report.summary


def test_extreme_bar_raises_a_warning():
    series = make_series(100, step=0.001)
    spike = Candle(
        series.last.timestamp + dt.timedelta(hours=1),
        open=100, high=100_000, low=99, close=101, volume=500,
    )
    widened = CandleSeries("T", Timeframe.H1, series.candles + [spike])
    report = assess_quality(widened, now=now_after(widened))
    assert any("implausible range" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

def test_synthetic_provider_is_deterministic():
    a = SyntheticCandleProvider("bull", seed=1).fetch("X", Timeframe.H1, 100)
    b = SyntheticCandleProvider("bull", seed=1).fetch("X", Timeframe.H1, 100)
    assert a.closes == b.closes


def test_different_seeds_give_different_markets():
    a = SyntheticCandleProvider("bull", seed=1).fetch("X", Timeframe.H1, 100)
    b = SyntheticCandleProvider("bull", seed=2).fetch("X", Timeframe.H1, 100)
    assert a.closes != b.closes


@pytest.mark.parametrize("regime", SyntheticCandleProvider.REGIMES)
def test_every_regime_produces_structurally_valid_candles(regime):
    series = SyntheticCandleProvider(regime, seed=3).fetch("X", Timeframe.H1, 200)
    assert len(series) == 200
    assert all(c.is_structurally_valid() for c in series)


def test_bull_and_bear_regimes_actually_trend():
    bull = SyntheticCandleProvider("bull", seed=5).fetch("X", Timeframe.H1, 300)
    bear = SyntheticCandleProvider("bear", seed=5).fetch("X", Timeframe.H1, 300)
    assert bull.closes[-1] > bull.closes[0]
    assert bear.closes[-1] < bear.closes[0]


def test_volatility_regimes_differ_in_range():
    def mean_range(regime):
        s = SyntheticCandleProvider(regime, seed=9).fetch("X", Timeframe.H1, 300)
        return sum(c.range / c.close for c in s) / len(s)

    assert mean_range("high_volatility") > mean_range("low_volatility") * 5


def test_unknown_regime_is_rejected():
    with pytest.raises(ValueError, match="unknown regime"):
        SyntheticCandleProvider("moon")


def test_timestamp_parsing_handles_every_supported_format():
    expected = dt.datetime(2026, 1, 1, tzinfo=UTC)
    assert _parse_timestamp("1767225600") == expected            # seconds
    assert _parse_timestamp("1767225600000") == expected         # milliseconds
    assert _parse_timestamp("2026-01-01T00:00:00Z") == expected  # ISO
    assert _parse_timestamp("2026-01-01T00:00:00+00:00") == expected


def test_csv_provider_round_trip(tmp_path):
    path = tmp_path / "ABC_1h.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1767225600,10,11,9,10.5,1000\n"
        "1767229200,10.5,12,10,11.5,2000\n",
        encoding="utf-8",
    )
    series = CsvCandleProvider(tmp_path).fetch("ABC", Timeframe.H1)
    assert len(series) == 2
    assert series.closes == [10.5, 11.5]
    assert series[0].timestamp == dt.datetime(2026, 1, 1, tzinfo=UTC)


def test_csv_provider_reports_the_offending_row(tmp_path):
    path = tmp_path / "ABC_1h.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1767225600,10,11,9,10.5,1000\n"
        "1767229200,not-a-number,12,10,11.5,2000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=":3"):
        CsvCandleProvider(tmp_path).fetch("ABC", Timeframe.H1)


def test_csv_provider_missing_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="no candle file"):
        CsvCandleProvider(tmp_path).fetch("NOPE", Timeframe.H1)


# ---------------------------------------------------------------------------
# resampling (multi-timeframe)
# ---------------------------------------------------------------------------

def test_resample_aggregates_ohlcv_correctly():
    series = SyntheticCandleProvider("bull", seed=11).fetch("X", Timeframe.H1, 240)
    h4 = resample(series, Timeframe.H4)

    assert h4.timeframe == Timeframe.H4
    assert len(h4) <= len(series) // 4

    first = h4[0]
    members = [c for c in series if
               int(c.timestamp.timestamp()) // Timeframe.H4.seconds
               == int(first.timestamp.timestamp()) // Timeframe.H4.seconds]
    assert first.open == members[0].open
    assert first.close == members[-1].close
    assert first.high == max(c.high for c in members)
    assert first.low == min(c.low for c in members)
    assert first.volume == pytest.approx(sum(c.volume for c in members))


def test_resample_drops_an_unfinished_bucket():
    """A partly formed higher-timeframe bar must not be emitted — using one
    would be look-ahead on the timeframe you are confirming against."""
    series = SyntheticCandleProvider("bull", seed=12).fetch("X", Timeframe.H1, 240)
    truncated = series.head(len(series) - 2)   # leaves a partial 4h bucket
    assert len(resample(truncated, Timeframe.H4)) < len(resample(series, Timeframe.H4))


def test_resample_refuses_to_go_faster():
    series = SyntheticCandleProvider("bull", seed=13).fetch("X", Timeframe.H4, 50)
    with pytest.raises(ValueError, match="faster"):
        resample(series, Timeframe.H1)


def test_resample_to_the_same_timeframe_is_a_no_op():
    series = SyntheticCandleProvider("bull", seed=14).fetch("X", Timeframe.H1, 50)
    assert resample(series, Timeframe.H1) is series
