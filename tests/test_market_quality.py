"""Tests for app/signals/market_quality.py.

The score exists to answer one question the other two scores cannot: can
this token actually be traded? Most of these tests therefore pin down the
anti-wash-trading behaviour, because that is the part that is easy to get
subtly wrong in a way nobody notices until the paper results look great and
the real ones don't.
"""
import datetime as dt

import pytest

from app.signals.market_quality import (
    DEFAULT_WEIGHTS,
    MAX_UNAVAILABLE_WEIGHT,
    score_buy_sell_balance,
    score_market_quality,
    score_volume_concentration,
    score_volume_consistency,
    score_volume_to_liquidity,
)
from tests.conftest import make_market_snapshot


def _score(**overrides) -> float:
    return score_market_quality(make_market_snapshot(**overrides), min_liquidity_usd=35_000.0).score


# ---------------------------------------------------------------------------
# the baseline
# ---------------------------------------------------------------------------

def test_a_healthy_market_scores_high():
    result = score_market_quality(make_market_snapshot(), min_liquidity_usd=35_000.0)
    assert result.reliable
    assert result.score > 90
    assert result.concerns == []


def test_weights_sum_to_one():
    # Not cosmetic: the aggregate divides by total weight, so weights that
    # don't sum to 1.0 silently rescale every factor's contribution.
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# missing data is not average data
# ---------------------------------------------------------------------------

def test_no_market_data_scores_zero_and_unreliable():
    """None must not become a neutral 50.

    A neutral score would sail through a `>= 50` gate, which is exactly the
    fail-open behaviour this bot must never have. No data is an
    unanswerable question, not a middling market.
    """
    result = score_market_quality(None)
    assert result.score == 0.0
    assert result.reliable is False
    assert result.factors == []
    assert "no market data" in result.warnings[0]


def test_enough_missing_fields_flags_the_whole_score_unreliable():
    result = score_market_quality(
        make_market_snapshot(
            volume_24h_usd=None, volume_1h_usd=None, buys_24h=None, sells_24h=None,
        ),
        min_liquidity_usd=35_000.0,
    )
    missing_weight = sum(f.weight for f in result.unavailable)
    assert missing_weight > MAX_UNAVAILABLE_WEIGHT
    assert result.reliable is False
    assert result.unavailable


def test_a_single_missing_field_does_not_make_the_score_unreliable():
    # The reliability flag has to tolerate normal API gaps, or every token
    # gets rejected and the bot never trades.
    result = score_market_quality(
        make_market_snapshot(price_change_1h_pct=None), min_liquidity_usd=35_000.0
    )
    assert result.reliable is True
    assert [f.name for f in result.unavailable] == ["price_stability"]


def test_missing_fields_are_flagged_not_scored_as_zero():
    factor = score_volume_to_liquidity(make_market_snapshot(volume_24h_usd=None), 0.16)
    assert factor.available is False
    assert factor.score == 0.5  # explicit "no opinion", not a fabricated 0


# ---------------------------------------------------------------------------
# high volume is not high quality - the point of the module
# ---------------------------------------------------------------------------

def test_a_one_hour_volume_burst_scores_worse_than_steady_volume():
    """Same 24h volume, different shape. The burst must score lower.

    This is the case a naive "more volume = better" score gets backwards:
    a token doing 20x its average hour right now looks fantastic on the 24h
    number and has nothing behind it to exit into.
    """
    steady = _score(volume_24h_usd=400_000.0, volume_1h_usd=20_000.0)
    burst = _score(volume_24h_usd=400_000.0, volume_1h_usd=340_000.0)
    assert burst < steady
    assert burst < 90


def test_more_volume_can_score_worse_when_the_pool_cannot_support_it():
    """A 100x bigger volume number scores WORSE against the same pool."""
    plausible = _score(volume_24h_usd=400_000.0, volume_1h_usd=20_000.0)
    implausible = _score(volume_24h_usd=40_000_000.0, volume_1h_usd=1_700_000.0)
    assert implausible < plausible


def test_extreme_turnover_is_treated_as_wash_trading():
    market = make_market_snapshot(liquidity_usd=50_000.0, volume_24h_usd=5_000_000.0)  # 100x
    factor = score_volume_to_liquidity(market, 0.16)
    assert factor.score < 0.1
    assert "wash" in factor.reason


def test_a_handful_of_whale_prints_scores_worse_than_many_small_ones():
    """Same volume, same pool, different number of participants."""
    whales = score_volume_consistency(
        make_market_snapshot(volume_24h_usd=400_000.0, buys_24h=8, sells_24h=6), 0.12
    )
    crowd = score_volume_consistency(
        make_market_snapshot(volume_24h_usd=400_000.0, buys_24h=1_200, sells_24h=900), 0.12
    )
    assert whales.score < crowd.score
    assert whales.score <= 0.1


def test_fading_interest_scores_below_steady_participation():
    fading = score_volume_concentration(
        make_market_snapshot(volume_24h_usd=400_000.0, volume_1h_usd=500.0), 0.14
    )
    steady = score_volume_concentration(
        make_market_snapshot(volume_24h_usd=400_000.0, volume_1h_usd=20_000.0), 0.14
    )
    assert fading.score < steady.score


def test_zero_transactions_scores_badly_rather_than_dividing_by_zero():
    factor = score_volume_consistency(
        make_market_snapshot(volume_24h_usd=0.0, buys_24h=0, sells_24h=0), 0.12
    )
    assert factor.available is True
    assert factor.score <= 0.1


# ---------------------------------------------------------------------------
# tradeability
# ---------------------------------------------------------------------------

def test_a_thin_pool_drags_the_score_down_hardest():
    """Liquidity carries the most weight: everything else is academic if
    you can't get out."""
    thin = _score(liquidity_usd=10_000.0)
    deep = _score(liquidity_usd=500_000.0)
    assert deep - thin > 15


def test_one_sided_flow_is_penalised_in_both_directions():
    dumping = score_buy_sell_balance(make_market_snapshot(buys_24h=100, sells_24h=900), 0.10)
    only_buys = score_buy_sell_balance(make_market_snapshot(buys_24h=900, sells_24h=100), 0.10)
    balanced = score_buy_sell_balance(make_market_snapshot(buys_24h=500, sells_24h=500), 0.10)
    assert dumping.score < balanced.score
    assert only_buys.score < balanced.score


def test_too_few_trades_makes_the_ratio_unreadable_rather_than_extreme():
    factor = score_buy_sell_balance(make_market_snapshot(buys_24h=5, sells_24h=1), 0.10)
    assert factor.available is False


def test_a_violent_intraday_swing_lowers_the_score():
    calm = _score(price_change_1h_pct=8.0)
    violent = _score(price_change_1h_pct=250.0)
    assert violent < calm


def test_a_brand_new_pool_scores_worse_than_an_established_one():
    now = dt.datetime.now(dt.timezone.utc)
    fresh = _score(pair_created_at=now - dt.timedelta(minutes=30))
    old = _score(pair_created_at=now - dt.timedelta(days=30))
    assert fresh < old


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_concerns_lists_the_weak_available_factors_worst_first():
    result = score_market_quality(
        make_market_snapshot(liquidity_usd=5_000.0, price_change_1h_pct=300.0),
        min_liquidity_usd=35_000.0,
    )
    names = [f.name for f in result.concerns]
    assert "liquidity_depth" in names
    assert "price_stability" in names
    assert result.concerns == sorted(result.concerns, key=lambda f: f.points)


def test_as_dict_is_json_safe_and_keeps_every_factor():
    import json

    payload = score_market_quality(make_market_snapshot(), min_liquidity_usd=35_000.0).as_dict()
    json.dumps(payload)  # must not raise - this is persisted on Signal
    assert len(payload["factors"]) == len(DEFAULT_WEIGHTS)
    assert payload["reliable"] is True
