"""Tests for the short-window flow profile.

The whole value of this module is that it refuses to invent windows the
data source does not publish, so most of these tests are about what it
declines to produce.
"""
import datetime as dt

import pytest

from app.analysis.flow import (
    DERIVED, MEASURED, UNAVAILABLE, profile_from_snapshot,
)
from tests.conftest import make_market_snapshot


def _snapshot(**over):
    return make_market_snapshot(**over)


def test_the_four_published_windows_are_measured():
    profile = profile_from_snapshot(_snapshot())
    for label in ("5m", "1h", "6h", "24h"):
        window = profile.get(label)
        assert window is not None, f"{label} missing"
        assert window.source == MEASURED


def test_one_minute_and_fifteen_minute_are_reported_unavailable_not_estimated():
    """DexScreener publishes m5 then jumps to h1. Interpolating would
    invent a number that looks precise and means nothing - the published
    windows are rolling aggregates, so differencing them measures nothing
    well-defined."""
    profile = profile_from_snapshot(_snapshot())

    for label in ("1m", "15m"):
        window = profile.get(label)
        assert window is not None, f"{label} should be listed, not omitted"
        assert window.source == UNAVAILABLE
        assert window.volume_usd is None
        assert window.buys is None
        assert window.detail, "an unavailable window must say why"

    assert "rolling aggregates" in profile.get("1m").detail


def test_buy_pressure_is_none_rather_than_half_when_nothing_traded():
    """A silent market is not a balanced one, and 0.5 would be read as
    balanced by everything downstream."""
    profile = profile_from_snapshot(_snapshot(buys_5m=0, sells_5m=0))
    assert profile.get("5m").buy_pressure is None


def test_buy_pressure_is_computed_from_the_matching_window():
    profile = profile_from_snapshot(_snapshot(buys_5m=30, sells_5m=10))
    assert profile.get("5m").buy_pressure == pytest.approx(0.75)


def test_acceleration_compares_the_short_window_to_the_hourly_rate():
    hot = profile_from_snapshot(_snapshot(volume_5m_usd=10_000.0, volume_1h_usd=60_000.0))
    cold = profile_from_snapshot(_snapshot(volume_5m_usd=1_000.0, volume_1h_usd=60_000.0))

    assert hot.accelerating is True      # 10k x 12 = 120k > 60k
    assert cold.accelerating is False    # 1k x 12 = 12k < 60k


def test_acceleration_is_unknown_rather_than_false_when_data_is_missing():
    assert profile_from_snapshot(_snapshot(volume_5m_usd=None)).accelerating is None


def test_pressure_shift_reports_short_versus_hourly_buying():
    profile = profile_from_snapshot(
        _snapshot(buys_5m=80, sells_5m=20, buys_1h=50, sells_1h=50)
    )
    assert profile.pressure_shift == pytest.approx(0.30)


class _Observation:
    def __init__(self, volume_5m_usd, minutes_ago):
        self.volume_5m_usd = volume_5m_usd
        self.observed_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes_ago)


def test_a_derived_window_is_labelled_derived_and_carries_its_spacing():
    """It is a rate of change across our own polling, not a published
    window total, and must never be mistaken for one."""
    market = _snapshot(volume_5m_usd=9_000.0)
    market.observed_at = dt.datetime.now(dt.timezone.utc)

    profile = profile_from_snapshot(market, observations=[_Observation(4_000.0, 3)])
    derived = next((w for w in profile.windows if w.source == DERIVED), None)

    assert derived is not None
    assert derived.volume_usd == pytest.approx(5_000.0)
    assert "rate of change, not a window total" in derived.detail


def test_two_observations_a_few_seconds_apart_produce_nothing():
    """They describe almost the same rolling window, so the difference is
    noise wearing a number."""
    market = _snapshot(volume_5m_usd=9_000.0)
    market.observed_at = dt.datetime.now(dt.timezone.utc)

    profile = profile_from_snapshot(market, observations=[_Observation(8_900.0, 0.2)])
    assert not [w for w in profile.windows if w.source == DERIVED]


def test_no_observations_means_no_derived_window():
    profile = profile_from_snapshot(_snapshot(), observations=[])
    assert not [w for w in profile.windows if w.source == DERIVED]


def test_the_summary_names_the_unavailable_windows():
    text = profile_from_snapshot(_snapshot()).summary()
    assert "UNAVAILABLE" in text
    assert "1m" in text and "15m" in text
