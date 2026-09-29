"""Continuous feed freshness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tradebot.monitor import FeedMonitor

T = datetime(2026, 3, 10, 14, 0, tzinfo=timezone.utc)


def monitor(**kwargs) -> FeedMonitor:
    return FeedMonitor(bar_seconds=300, max_stale_bars=2.5, **kwargs)


def test_no_data_at_all_is_stale():
    health = monitor().health(now=T)
    assert health.stale and "no market data" in health.reason


def test_a_fresh_bar_is_not_stale():
    m = monitor()
    m.record(T, True, now=T + timedelta(seconds=300))
    assert not m.health(now=T + timedelta(seconds=310)).stale


def test_a_bar_older_than_the_threshold_is_stale():
    """Isolate the bar-age path by keeping messages flowing the whole time."""
    m = monitor()
    m.record(T, True, now=T)
    # Threshold is max(300 * 2.5, 330) = 750s, measured from the bar's close.
    # Forming-bar updates keep arriving, so only bar age can trip this.
    m.record(T, False, now=T + timedelta(seconds=900))
    assert not m.health(now=T + timedelta(seconds=900)).stale       # age 600s

    m.record(T, False, now=T + timedelta(seconds=1200))
    health = m.health(now=T + timedelta(seconds=1200))              # age 900s
    assert health.stale and "over the" in health.reason


def test_silence_is_caught_even_when_the_bar_timestamp_looks_recent():
    """A socket that is open but dead keeps replaying the same bar."""
    m = monitor()
    m.record(T, True, now=T)
    later = T + timedelta(seconds=800)
    # The bar itself is only 500s past its close - under the limit - but nothing
    # has arrived for 800s.
    health = m.health(now=later)
    assert health.stale
    assert "nothing received" in health.reason or "over the" in health.reason


def test_freshness_is_not_enforced_outside_the_session():
    m = monitor()
    m.record(T, True, now=T)
    health = m.health(in_session=False, now=T + timedelta(hours=14))
    assert not health.stale
    assert "not enforced" in health.reason


def test_an_absolute_override_wins():
    m = monitor(max_stale_seconds=60)
    assert m.threshold_seconds == 60
    m.record(T, True, now=T)
    assert m.health(now=T + timedelta(seconds=400)).stale


def test_the_threshold_never_undercuts_one_bar():
    """A tight max_stale_bars must not flag a feed that is simply on time."""
    m = FeedMonitor(bar_seconds=300, max_stale_bars=0.1)
    assert m.threshold_seconds >= 330


def test_counters_track_bars_and_messages():
    m = monitor()
    m.record(T, True, now=T)
    m.record(T, False, now=T + timedelta(seconds=10))          # forming-bar update
    m.record(T + timedelta(seconds=300), True, now=T + timedelta(seconds=300))
    assert m.bars_seen == 2 and m.messages_seen == 3


def test_stale_transitions_are_counted_once_per_episode():
    m = monitor()
    m.record(T, True, now=T)
    for offset in (1200, 1300, 1400):
        m.health(now=T + timedelta(seconds=offset))
    assert m.stale_events == 1

    m.record(T + timedelta(seconds=1500), True, now=T + timedelta(seconds=1500))
    m.health(now=T + timedelta(seconds=1510))                  # recovered
    m.health(now=T + timedelta(seconds=3000))                  # stale again
    assert m.stale_events == 2
