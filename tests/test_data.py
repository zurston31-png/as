from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.data.base import (
    CandleAggregator,
    bucket_start,
    parse_candle,
    timeframe_seconds,
)
from tradebot.data.replay import read_csv
from tradebot.data.synthetic import SyntheticSource
from tradebot.data.webhook import WebhookSource

T = datetime(2026, 3, 10, 14, 2, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("text,expected", [
    ("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14400),
    ("1d", 86400), ("30s", 30), ("1W", 604800),
])
def test_timeframe_parsing(text, expected):
    assert timeframe_seconds(text) == expected


@pytest.mark.parametrize("bad", ["", "5", "m", "0m", "-5m", "5y", "abc"])
def test_bad_timeframes_are_rejected(bad):
    with pytest.raises(ValueError):
        timeframe_seconds(bad)


def test_bucket_start_floors_to_the_bar_open():
    assert bucket_start(T, 300).minute == 0
    assert bucket_start(T, 60).minute == 2


def test_aggregator_builds_and_closes_bars():
    agg = CandleAggregator("5m")
    closed, forming = agg.add(100.0, 10, T)
    assert closed is None and forming.open == 100.0

    agg.add(105.0, 5, T + timedelta(seconds=30))
    closed, forming = agg.add(95.0, 5, T + timedelta(seconds=60))
    assert closed is None
    assert forming.high == 105.0 and forming.low == 95.0 and forming.volume == 20

    closed, new_bar = agg.add(99.0, 1, T + timedelta(minutes=6))
    assert closed is not None and closed.closed and closed.close == 95.0
    assert new_bar.ts > closed.ts


@pytest.mark.parametrize("payload,close", [
    ({"close": 25400.5}, 25400.5),
    ({"c": "25400.5"}, 25400.5),
    ({"price": 25400.5}, 25400.5),
])
def test_parse_candle_accepts_the_usual_field_names(payload, close):
    candle = parse_candle(payload)
    assert candle is not None and candle.close == close


def test_parse_candle_handles_epoch_seconds_and_millis():
    a = parse_candle({"close": 1, "time": 1772000000})
    b = parse_candle({"close": 1, "time": 1772000000000})
    assert a.ts == b.ts


def test_parse_candle_rejects_a_payload_with_no_price():
    assert parse_candle({"symbol": "NQ", "note": "hello"}) is None


def test_webhook_accepts_a_closed_bar(config):
    source = WebhookSource("5m")
    result = source.submit({
        "symbol": "TEST", "open": 1, "high": 2, "low": 0.5, "close": 1.5,
        "volume": 10, "time": T.isoformat(), "closed": True,
    })
    assert result["accepted"] and result["closed_bar"]
    assert source.queue.qsize() == 1


def test_webhook_aggregates_tick_shaped_alerts():
    source = WebhookSource("5m")
    source.submit({"close": 100, "time": T.isoformat(), "closed": False})
    source.submit({"close": 101, "time": (T + timedelta(minutes=6)).isoformat(), "closed": False})
    # one forming bar, then a closed bar plus the next forming bar
    assert source.queue.qsize() == 3
    assert source.received == 2


def test_webhook_reports_a_useless_payload():
    source = WebhookSource("5m")
    result = source.submit({"message": "buy now"})
    assert not result["accepted"] and source.rejected == 1


def test_csv_round_trip(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "Time,Open,High,Low,Close,Volume\n"
        "2026-03-10T14:00:00Z,100,102,99,101,500\n"
        "2026-03-10T14:05:00Z,101,103,100,102,600\n"
    )
    candles = read_csv(path)
    assert len(candles) == 2
    assert candles[0].close == 101 and candles[1].volume == 600
    assert candles[0].ts < candles[1].ts


def test_csv_skips_unusable_rows(tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2026-03-10T14:00:00Z,100,102,99,101,500\n"
        ",,,,,\n"
        "time,open,high,low,close,volume\n"
        "2026-03-10T14:05:00Z,101,103,100,102,600\n"
    )
    assert len(read_csv(path)) == 2


def test_synthetic_history_is_deterministic_and_well_formed():
    a = SyntheticSource("TEST", "5m", seed=1).history(200)
    b = SyntheticSource("TEST", "5m", seed=1).history(200)
    assert [c.close for c in a] == [c.close for c in b]
    assert all(c.high >= max(c.open, c.close) and c.low <= min(c.open, c.close) for c in a)
    assert all(c.volume > 0 for c in a)
    assert all(a[i].ts < a[i + 1].ts for i in range(len(a) - 1))
