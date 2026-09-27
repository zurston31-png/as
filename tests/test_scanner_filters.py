"""Tests for app/scanner/filters.py - the free pre-screen that keeps the
expensive stages (rug check, candle fetch) from running on obvious junk.
"""
import datetime as dt

import pytest

from app.config import settings
from app.scanner.discovery import DiscoveredToken
from app.scanner.filters import prescreen


def _token(**overrides) -> DiscoveredToken:
    defaults = dict(
        token_address="TokenAddr111",
        symbol="GOODCOIN",
        chain="solana",
        source="dexscreener",
        liquidity_usd=100_000.0,
        volume_24h_usd=200_000.0,
        buys_24h=300,
        sells_24h=250,
        price_usd=0.01,
        price_change_1h_pct=5.0,
        price_change_24h_pct=20.0,
        pair_created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3),
    )
    defaults.update(overrides)
    return DiscoveredToken(**defaults)


def test_a_healthy_token_passes():
    assert prescreen(_token()).passed


# ---------------------------------------------------------------------------
# missing data must never pass — fail-closed, same rule as the rug filter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["liquidity_usd", "volume_24h_usd", "pair_created_at", "buys_24h", "sells_24h"])
def test_missing_data_is_rejected_not_waved_through(field):
    verdict = prescreen(_token(**{field: None}))
    assert not verdict.passed
    assert verdict.reason


# ---------------------------------------------------------------------------
# individual thresholds
# ---------------------------------------------------------------------------

def test_thin_liquidity_rejected():
    verdict = prescreen(_token(liquidity_usd=settings.SCANNER_MIN_LIQUIDITY_USD - 1))
    assert not verdict.passed
    assert "liquidity" in verdict.reason


def test_low_volume_rejected():
    verdict = prescreen(_token(volume_24h_usd=settings.SCANNER_MIN_VOLUME_24H_USD - 1))
    assert not verdict.passed
    assert "volume" in verdict.reason


def test_brand_new_pool_rejected():
    """The first hours of a pool's life are the highest-risk rug window and
    have too little history for the signal engine to read anyway."""
    fresh = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)
    verdict = prescreen(_token(pair_created_at=fresh))
    assert not verdict.passed
    assert "old" in verdict.reason


def test_very_old_pool_rejected_by_the_ceiling():
    ancient = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=400)
    verdict = prescreen(_token(pair_created_at=ancient))
    assert not verdict.passed


def test_age_ceiling_can_be_disabled_with_zero(monkeypatch):
    monkeypatch.setattr(settings, "SCANNER_MAX_TOKEN_AGE_HOURS", 0)
    ancient = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=400)
    assert prescreen(_token(pair_created_at=ancient)).passed


def test_too_few_transactions_rejected():
    verdict = prescreen(_token(buys_24h=1, sells_24h=1))
    assert not verdict.passed
    assert "trades in 24h" in verdict.reason


def test_overwhelming_sell_pressure_rejected():
    verdict = prescreen(_token(buys_24h=10, sells_24h=990))
    assert not verdict.passed
    assert "sells" in verdict.reason


def test_balanced_buy_sell_passes():
    assert prescreen(_token(buys_24h=500, sells_24h=500)).passed


def test_age_hours_property_is_none_without_a_creation_time():
    assert _token(pair_created_at=None).age_hours is None
