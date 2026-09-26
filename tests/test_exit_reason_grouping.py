"""Exit-reason buckets must group by the RULE, not the rendered message.

THE BUG THIS PINS

Every exit reason renders the numbers that fired it:

    trend reversal: lower highs after peak $0.00024290
    trend reversal: lower highs after peak $0.00929600

Grouping on the raw string therefore gives every trade its own bucket,
because no two exits fire at the same price. A real deployment showed it
exactly: 24 closed trades produced 24 buckets of one trade each, every
one flagged "too few trades to mean anything" - when 21 of them were the
same rule firing 21 times. The most useful line in the report was the
one line it could not print.

These tests are written against the real strings that deployment
emitted, not invented ones, so a future change to an exit message that
reintroduces an ungroupable field fails here.
"""
import datetime as dt

from app import models
from app.analysis import trade_analytics as ta

NOW = dt.datetime.now(dt.timezone.utc)

# Verbatim from a live paper deployment's report at strategy v-83c77cda.
REAL_TREND = "trend reversal: lower highs after peak $0.00024290"
REAL_TREND_2 = "trend reversal: lower highs after peak $0.03047000"
REAL_MOMENTUM = "momentum loss: price fell 14.1% from recent peak $0.00015270"
REAL_PARTIAL = "partial profit-take at +20.5% ($0.00058220)"
REAL_DEV = "dev/top wallet sold ~96.9% of supply (103.3% -> 6.4%)"


def _exit(trade_id, reason, pnl=1.0):
    return models.Trade(
        id=trade_id, position_id=trade_id, signal_id=None,
        symbol="EXITCOIN", side="sell", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=100.0, qty=100.0, exit_price=1.0,
        pnl_usd=pnl, pnl_pct=pnl / 100.0,
        close_reason=reason,
        closed_at=NOW, created_at=NOW,
    )


def test_the_same_rule_at_different_prices_is_one_bucket():
    """Two trend reversals at different peaks are one rule, not two.

    This is the whole defect in miniature: identical rule, different
    embedded price, and the old code counted them separately.
    """
    trades = [_exit(1, REAL_TREND), _exit(2, REAL_TREND_2)]
    b = ta.breakdown_by_exit_reason(trades)
    assert len(b.buckets) == 1
    assert b.buckets[0].trade_count == 2


def test_the_deployment_that_produced_24_buckets_now_produces_four():
    """The exact shape of the live report: 21 + 1 + 1 + 1."""
    trades = [
        _exit(i, f"trend reversal: lower highs after peak ${i / 100000:.8f}")
        for i in range(1, 22)
    ]
    trades += [
        _exit(90, REAL_MOMENTUM),
        _exit(91, REAL_PARTIAL),
        _exit(92, REAL_DEV),
    ]
    b = ta.breakdown_by_exit_reason(trades)
    assert len(b.buckets) == 4
    # sorted by trade_count descending, so the dominant rule leads
    assert b.buckets[0].trade_count == 21
    assert b.buckets[0].label == "trend reversal: lower highs after peak $N"
    assert sorted(x.trade_count for x in b.buckets) == [1, 1, 1, 21]


def test_distinct_rules_stay_distinct():
    """Normalising must not over-collapse: momentum loss is not a trend
    reversal, and a partial take is not a dev-wallet exit."""
    trades = [
        _exit(1, REAL_TREND), _exit(2, REAL_MOMENTUM),
        _exit(3, REAL_PARTIAL), _exit(4, REAL_DEV),
    ]
    b = ta.breakdown_by_exit_reason(trades)
    assert len(b.buckets) == 4


def test_pnl_aggregates_across_the_collapsed_bucket():
    """Grouping is only useful if the money follows it."""
    trades = [
        _exit(1, REAL_TREND, pnl=2.0),
        _exit(2, REAL_TREND_2, pnl=-5.0),
        _exit(3, "trend reversal: lower highs after peak $0.00000001", pnl=1.0),
    ]
    b = ta.breakdown_by_exit_reason(trades)
    assert len(b.buckets) == 1
    only = b.buckets[0]
    assert only.trade_count == 3
    assert only.win_count == 2
    assert only.total_pnl_usd == -2.0
    assert only.expectancy_usd == -2.0 / 3


def test_liquidity_reasons_with_thousands_separators_collapse():
    """Money in these carries commas; the pattern has to span them or the
    bucket splits on the digits after the comma."""
    a = "liquidity fell 62% since entry ($18,000 to $6,840) - the pool is being drained"
    b_ = "liquidity fell 71% since entry ($92,500 to $26,825) - the pool is being drained"
    b = ta.breakdown_by_exit_reason([_exit(1, a), _exit(2, b_)])
    assert len(b.buckets) == 1
    assert b.buckets[0].label == (
        "liquidity fell N% since entry ($N to $N) - the pool is being drained"
    )


def test_a_missing_or_blank_reason_is_unknown_not_a_bucket():
    """An unrecorded reason is unmeasurable, never its own category."""
    b = ta.breakdown_by_exit_reason(
        [_exit(1, REAL_TREND), _exit(2, None), _exit(3, "   ")]
    )
    assert b.unknown_count == 2
    assert len(b.buckets) == 1


def test_rule_extraction_is_idempotent():
    """Normalising an already-normalised label must not change it, or
    re-reporting a stored bucket key would drift."""
    once = ta.exit_reason_rule(REAL_MOMENTUM)
    assert ta.exit_reason_rule(once) == once


def test_a_reason_carrying_no_numbers_survives_intact():
    """Not every rule renders a measurement; those must pass through."""
    plain = "manual close requested from the dashboard"
    assert ta.exit_reason_rule(plain) == plain
