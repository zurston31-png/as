"""Tests for app/execution/fill_model.py.

The headline property is test_impact_scales_with_trade_size_against_pool_depth:
the previous simulator charged a $20 trade and a $20,000 trade against the
same thin pool an identical 0.5%, which is the main way a memecoin paper
simulator flatters itself. Impact must now scale with how much of the pool
the trade is actually eating.
"""
import random

import pytest

from app.config import settings
from app.execution.fill_model import (
    ASSUMED_LIQUIDITY_USD,
    MAX_IMPACT_PCT,
    confirmation_delay_seconds,
    delay_drift_pct,
    price_impact_pct,
    simulate_fill,
)


def _rng(seed=1):
    return random.Random(seed)


# ---------------------------------------------------------------------------
# price impact — the constant-product derivation
# ---------------------------------------------------------------------------

def test_impact_matches_the_constant_product_derivation():
    """effective/spot = 1 + d/R, and R (quote side) is half the reported
    total pool value. So a $5,000 trade into a $100,000 pool eats
    5000/50000 = 10%."""
    assert price_impact_pct(5_000, 100_000) == pytest.approx(0.10)
    assert price_impact_pct(1_000, 100_000) == pytest.approx(0.02)


def test_impact_scales_with_trade_size_against_pool_depth():
    """The flat-percentage model this replaces charged these identically."""
    small = price_impact_pct(20, 100_000)
    large = price_impact_pct(20_000, 100_000)
    assert large > small * 100
    assert small < 0.001
    assert large > 0.30


def test_impact_scales_inversely_with_liquidity():
    thin = price_impact_pct(1_000, 20_000)
    deep = price_impact_pct(1_000, 2_000_000)
    assert thin > deep
    assert deep < 0.002


def test_impact_is_capped():
    assert price_impact_pct(10_000_000, 1_000) == MAX_IMPACT_PCT


def test_unknown_liquidity_uses_a_conservative_assumption():
    """An unknown pool must not be assumed forgiving."""
    unknown = price_impact_pct(1_000, None)
    assert unknown == pytest.approx(1_000 / (ASSUMED_LIQUIDITY_USD / 2))
    assert unknown > 0


@pytest.mark.parametrize("bad", [0, -5])
def test_nonsense_liquidity_falls_back_rather_than_dividing_by_zero(bad):
    assert 0 < price_impact_pct(100, bad) <= MAX_IMPACT_PCT


# ---------------------------------------------------------------------------
# confirmation delay and drift
# ---------------------------------------------------------------------------

def test_confirmation_delay_is_within_the_configured_window():
    rng = _rng()
    for _ in range(50):
        d = confirmation_delay_seconds(rng)
        assert settings.PAPER_MIN_CONFIRM_SECONDS <= d <= settings.PAPER_MAX_CONFIRM_SECONDS


def test_a_more_volatile_token_drifts_further_during_confirmation():
    calm = [abs(delay_drift_pct(2.0, 2.0, _rng(s))) for s in range(40)]
    wild = [abs(delay_drift_pct(200.0, 2.0, _rng(s))) for s in range(40)]
    assert sum(wild) / len(wild) > sum(calm) / len(calm) * 10


def test_drift_grows_with_the_delay_by_sqrt_of_time():
    short = [abs(delay_drift_pct(50.0, 1.0, _rng(s))) for s in range(40)]
    long = [abs(delay_drift_pct(50.0, 100.0, _rng(s))) for s in range(40)]
    assert sum(long) / len(long) > sum(short) / len(short)


def test_drift_is_signed_not_always_adverse():
    """Price can move either way while a swap confirms - modeling it as
    always-against would overstate costs as badly as ignoring it
    understates them."""
    samples = [delay_drift_pct(50.0, 2.0, _rng(s)) for s in range(60)]
    assert any(x > 0 for x in samples)
    assert any(x < 0 for x in samples)


def test_missing_volatility_still_produces_drift():
    samples = [delay_drift_pct(None, 2.0, _rng(s)) for s in range(30)]
    assert any(abs(x) > 0 for x in samples)


# ---------------------------------------------------------------------------
# whole-fill behavior
# ---------------------------------------------------------------------------

def test_a_small_trade_into_a_deep_pool_fills_cheaply():
    outcome = simulate_fill(
        side="buy", reference_price=1.0, trade_usd=100, liquidity_usd=5_000_000,
        volatility_1h_pct=5.0, slippage_bps=150, rng=_rng(),
    )
    assert outcome.filled
    assert outcome.fill_price > 1.0           # still worse than mid
    assert outcome.total_cost_pct < 0.01


def test_a_buy_fills_above_mid_and_a_sell_below():
    buy = simulate_fill(side="buy", reference_price=1.0, trade_usd=100,
                        liquidity_usd=1_000_000, volatility_1h_pct=1.0,
                        slippage_bps=500, rng=_rng(7))
    sell = simulate_fill(side="sell", reference_price=1.0, trade_usd=100,
                         liquidity_usd=1_000_000, volatility_1h_pct=1.0,
                         slippage_bps=500, rng=_rng(7))
    assert buy.filled and sell.filled
    assert buy.fill_price > 1.0
    assert sell.fill_price < 1.0


def test_an_oversized_trade_fails_on_slippage_tolerance():
    """This is the outcome the old simulator could never produce: a real
    swap reverts when the market moves past its tolerance."""
    outcome = simulate_fill(
        side="buy", reference_price=1.0, trade_usd=50_000, liquidity_usd=100_000,
        volatility_1h_pct=5.0, slippage_bps=150, rng=_rng(),
    )
    assert not outcome.filled
    assert "slippage tolerance exceeded" in outcome.failure_reason
    assert outcome.impact_pct > 0.15


def test_a_wider_slippage_tolerance_lets_the_same_trade_through():
    kwargs = dict(side="buy", reference_price=1.0, trade_usd=6_000,
                  liquidity_usd=200_000, volatility_1h_pct=3.0)
    tight = simulate_fill(**kwargs, slippage_bps=100, rng=_rng(3))
    loose = simulate_fill(**kwargs, slippage_bps=5_000, rng=_rng(3))
    assert not tight.filled
    assert loose.filled


def test_slippage_bps_is_actually_honored():
    """It used to be accepted and silently ignored."""
    kwargs = dict(side="buy", reference_price=1.0, trade_usd=10_000,
                  liquidity_usd=100_000, volatility_1h_pct=1.0)
    assert not simulate_fill(**kwargs, slippage_bps=50, rng=_rng(2)).filled
    assert simulate_fill(**kwargs, slippage_bps=9_000, rng=_rng(2)).filled


def test_a_failed_fill_reports_why():
    outcome = simulate_fill(
        side="buy", reference_price=1.0, trade_usd=80_000, liquidity_usd=100_000,
        volatility_1h_pct=10.0, slippage_bps=100, rng=_rng(),
    )
    assert not outcome.filled
    assert "price impact" in outcome.failure_reason
    assert "drift" in outcome.failure_reason


def test_fill_price_never_goes_negative_under_extreme_conditions():
    outcome = simulate_fill(
        side="sell", reference_price=0.000001, trade_usd=1_000_000, liquidity_usd=500,
        volatility_1h_pct=500.0, slippage_bps=100_000, rng=_rng(),
    )
    assert outcome.fill_price >= 0.0


def test_the_same_seed_reproduces_the_same_fill():
    kwargs = dict(side="buy", reference_price=1.0, trade_usd=500,
                  liquidity_usd=250_000, volatility_1h_pct=20.0, slippage_bps=300)
    a = simulate_fill(**kwargs, rng=_rng(99))
    b = simulate_fill(**kwargs, rng=_rng(99))
    assert a.fill_price == b.fill_price
    assert a.delay_seconds == b.delay_seconds


def test_cost_components_are_reported_for_the_journal():
    outcome = simulate_fill(
        side="buy", reference_price=1.0, trade_usd=1_000, liquidity_usd=500_000,
        volatility_1h_pct=10.0, slippage_bps=1_000, rng=_rng(),
    )
    assert outcome.filled
    assert outcome.impact_pct > 0
    assert outcome.spread_pct == settings.PAPER_SPREAD_PCT
    assert outcome.fee_pct == settings.PAPER_FEE_PCT
    assert outcome.delay_seconds > 0
