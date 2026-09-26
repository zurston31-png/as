"""Unit tests for app/rugcheck/risk_score.py - the composite 0-100 Rug Risk
Score layered on top of the binary checks in app/rugcheck/filters.py.
"""
import datetime as dt

import pytest

from app.rugcheck.filters import TokenSnapshot
from app.rugcheck.risk_score import score_rug_risk
from app.services.price_feed import MarketSnapshot


def _snap(**overrides) -> TokenSnapshot:
    defaults = dict(
        source="test",
        chain="solana",
        mint_authority_active=False,
        freeze_authority_active=False,
        honeypot=False,
        lp_secured=True,
        lp_secured_pct=1.0,
        top10_pct=0.15,
        liquidity_usd=100_000.0,
        dev_pct=0.02,
        rugged=False,
        danger_flags=[],
    )
    defaults.update(overrides)
    return TokenSnapshot(**defaults)


def _market(**overrides) -> MarketSnapshot:
    defaults = dict(
        price_usd=1.0,
        liquidity_usd=100_000.0,
        volume_24h_usd=50_000.0,
        buys_24h=200,
        sells_24h=180,
        price_change_1h_pct=5.0,
        price_change_24h_pct=10.0,
        pair_created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30),
        fdv_usd=1_000_000.0,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


# ---------------------------------------------------------------------------
# overall shape / level buckets
# ---------------------------------------------------------------------------

def test_clean_token_with_full_data_scores_low_and_safe():
    result = score_rug_risk(_snap(), _market())
    assert result.score < 20
    assert result.level == "safe"
    assert result.reliable


def test_every_worst_case_signal_scores_high_and_critical():
    snap = _snap(
        mint_authority_active=True,
        freeze_authority_active=True,
        honeypot=True,
        lp_secured=False,
        lp_secured_pct=0.0,
        top10_pct=0.90,
        liquidity_usd=500.0,
        dev_pct=0.40,
        rugged=True,
        danger_flags=["large amount of LP unlocked", "suspicious mint"],
    )
    market = _market(
        liquidity_usd=500.0,
        volume_24h_usd=50_000.0,   # huge multiple of the (tiny) liquidity
        buys_24h=5, sells_24h=95,
        price_change_1h_pct=500.0,
        pair_created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5),
    )
    result = score_rug_risk(snap, market)
    assert result.score >= 70
    assert result.level == "critical"


def test_missing_market_data_marks_market_factors_unavailable_not_safe():
    result = score_rug_risk(_snap(), market=None)
    unavailable_names = {f.name for f in result.unavailable}
    assert "token_age" in unavailable_names
    assert "volume_liquidity_ratio" in unavailable_names
    assert "buy_sell_imbalance" in unavailable_names
    assert "price_manipulation" in unavailable_names
    # missing data must never score as if it were safe (0.0) - it's neutral
    for f in result.unavailable:
        assert f.score == 0.5


def test_missing_scanner_data_does_not_default_to_safe():
    snap = _snap(
        mint_authority_active=None, freeze_authority_active=None,
        honeypot=None, lp_secured=None, lp_secured_pct=None,
        top10_pct=None, liquidity_usd=None, dev_pct=None,
    )
    result = score_rug_risk(snap, market=None)
    # With almost everything unavailable, this must be flagged unreliable
    # rather than quietly reported as a low, trustworthy-looking score.
    assert not result.reliable
    assert result.warnings


# ---------------------------------------------------------------------------
# always-honest stub factors
# ---------------------------------------------------------------------------

def test_liquidity_change_and_suspicious_transfers_are_always_unavailable():
    """These two factors have no data source wired up at all today - they
    must say so honestly on every call, never silently score as safe."""
    result = score_rug_risk(_snap(), _market())
    names = {f.name for f in result.unavailable}
    assert "liquidity_change" in names
    assert "suspicious_transfers" in names


# ---------------------------------------------------------------------------
# individual factor behavior
# ---------------------------------------------------------------------------

def test_active_mint_authority_scores_maximum_risk_on_that_factor():
    result = score_rug_risk(_snap(mint_authority_active=True), _market())
    factor = next(f for f in result.factors if f.name == "mint_authority")
    assert factor.score == 1.0
    assert factor.available


def test_freeze_authority_unavailable_on_evm_chain():
    result = score_rug_risk(_snap(chain="ethereum"), _market())
    factor = next(f for f in result.factors if f.name == "freeze_authority")
    assert not factor.available


def test_thin_liquidity_scores_higher_risk_than_deep_liquidity():
    thin = score_rug_risk(_snap(liquidity_usd=1_000.0), _market(liquidity_usd=1_000.0), min_liquidity_usd=15_000)
    deep = score_rug_risk(_snap(liquidity_usd=200_000.0), _market(liquidity_usd=200_000.0), min_liquidity_usd=15_000)
    thin_factor = next(f for f in thin.factors if f.name == "liquidity_depth")
    deep_factor = next(f for f in deep.factors if f.name == "liquidity_depth")
    assert thin_factor.score > deep_factor.score


def test_very_young_pool_scores_higher_risk_than_an_established_one():
    young = score_rug_risk(_snap(), _market(pair_created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)))
    old = score_rug_risk(_snap(), _market(pair_created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=60)))
    young_factor = next(f for f in young.factors if f.name == "token_age")
    old_factor = next(f for f in old.factors if f.name == "token_age")
    assert young_factor.score > old_factor.score


def test_heavy_sell_pressure_scores_higher_than_balanced_trading():
    heavy_sells = score_rug_risk(_snap(), _market(buys_24h=20, sells_24h=180))
    balanced = score_rug_risk(_snap(), _market(buys_24h=100, sells_24h=100))
    sell_factor = next(f for f in heavy_sells.factors if f.name == "buy_sell_imbalance")
    balanced_factor = next(f for f in balanced.factors if f.name == "buy_sell_imbalance")
    assert sell_factor.score > balanced_factor.score


def test_extreme_price_swing_on_thin_liquidity_flags_manipulation():
    result = score_rug_risk(
        _snap(liquidity_usd=5_000.0),
        _market(liquidity_usd=5_000.0, price_change_1h_pct=400.0),
    )
    factor = next(f for f in result.factors if f.name == "price_manipulation")
    assert factor.score == 1.0


def test_rugged_flag_dominates_the_scanner_danger_factor():
    result = score_rug_risk(_snap(rugged=True), _market())
    factor = next(f for f in result.factors if f.name == "scanner_danger_flags")
    assert factor.score == 1.0
    assert "rugged" in factor.reason


# ---------------------------------------------------------------------------
# as_dict / breakdown don't crash and carry the essentials
# ---------------------------------------------------------------------------

def test_as_dict_round_trips_the_essentials():
    result = score_rug_risk(_snap(), _market())
    data = result.as_dict()
    assert data["score"] == pytest.approx(result.score, abs=0.01)
    assert data["level"] == result.level
    assert len(data["factors"]) == len(result.factors)


def test_breakdown_does_not_raise_and_mentions_the_score():
    result = score_rug_risk(_snap(), _market())
    text = result.breakdown()
    assert f"{result.score:.1f}" in text
