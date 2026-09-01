"""Tests for app/analysis/trade_analytics.py.

Two rules run through the module and through most of these tests: a
missing value is never turned into a zero, and results from different
strategy versions are never pooled by accident.
"""
import datetime as dt

import pytest

from app import models
from app.analysis import trade_analytics as ta

NOW = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.timezone.utc)


def _trade(
    pnl=None, *, symbol="TESTCOIN", opened_hours_ago=6.0, closed_hours_ago=0.0,
    fee=None, cost_pct=None, delay=None, size=100.0, side="sell",
    status=models.TradeStatus.FILLED.value, signal_id=None, close_reason=None,
    version="v-aaaa1111", pnl_pct=None,
) -> models.Trade:
    return models.Trade(
        symbol=symbol, side=side, status=status, size_usd=size,
        pnl_usd=pnl, pnl_pct=pnl_pct,
        opened_at=NOW - dt.timedelta(hours=opened_hours_ago),
        closed_at=None if pnl is None else NOW - dt.timedelta(hours=closed_hours_ago),
        fee_usd=fee, execution_cost_pct=cost_pct, fill_delay_seconds=delay,
        signal_id=signal_id, close_reason=close_reason, strategy_version=version,
    )


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------

def test_costs_are_summed_across_every_filled_leg():
    """The average cost rate is weighted by notional.

    This asserted 0.007 - the flat mean of 0.006 and 0.008 - which
    contradicted the line above it: $2.20 of cost on $300 of notional is
    0.7333%, not 0.7%. The two legs are different sizes, so a flat mean
    lets the $100 leg count as much as the $200 one and the headline rate
    disagrees with the dollar total printed beside it.

    The test was wrong, not the code. Same defect as the one fixed in
    app/analysis/postmortem.py; corrected here and in
    app/analysis/fill_audit.py at the same time.
    """
    trades = [
        _trade(pnl=10.0, fee=0.25, cost_pct=0.006, delay=1.2, size=100.0),
        _trade(pnl=-5.0, fee=0.50, cost_pct=0.008, delay=0.8, size=200.0),
    ]
    costs = ta.summarize_costs(trades)
    assert costs.total_fees_usd == pytest.approx(0.75)
    assert costs.total_execution_cost_usd == pytest.approx(0.006 * 100 + 0.008 * 200)
    assert costs.total_slippage_usd == pytest.approx(0.006 * 100 + 0.008 * 200 - 0.75)
    # The rate and the dollar total must be the same measurement.
    assert costs.avg_execution_cost_pct == pytest.approx(
        costs.total_execution_cost_usd / 300.0
    )
    assert costs.avg_execution_cost_pct == pytest.approx(0.0073333, abs=1e-6)
    assert costs.avg_fill_delay_seconds == pytest.approx(1.0)
    assert costs.legs_counted == 2
    assert costs.cost_data_complete


def test_a_leg_with_no_recorded_cost_is_not_counted_as_free():
    """Summing an unrecorded fee as zero understates costs, which is the
    direction an error in a trading simulator must never point."""
    trades = [_trade(pnl=10.0, fee=1.0, cost_pct=0.01), _trade(pnl=5.0)]
    costs = ta.summarize_costs(trades)
    assert costs.total_fees_usd == pytest.approx(1.0)
    assert costs.legs_counted == 1
    assert costs.legs_missing_cost_data == 1
    assert costs.cost_data_complete is False
    assert costs.coverage_pct == pytest.approx(50.0)


def test_unfilled_legs_are_ignored_entirely():
    trades = [
        _trade(pnl=10.0, fee=1.0, cost_pct=0.01),
        _trade(status=models.TradeStatus.FAILED.value, fee=99.0),
    ]
    costs = ta.summarize_costs(trades)
    assert costs.total_fees_usd == pytest.approx(1.0)
    assert costs.legs_counted == 1
    assert costs.legs_missing_cost_data == 0


def test_slippage_never_reports_as_negative():
    """A favourable drift can make one leg's total cost less than its fee.
    Reporting a negative portfolio-level 'slippage' invites misreading it
    as a gain."""
    costs = ta.summarize_costs([_trade(pnl=1.0, fee=5.0, cost_pct=0.001, size=100.0)])
    assert costs.total_slippage_usd == 0.0


def test_no_trades_gives_zeros_and_no_averages():
    costs = ta.summarize_costs([])
    assert costs.total_fees_usd == 0.0
    assert costs.avg_execution_cost_pct is None
    assert costs.coverage_pct == 0.0


# ---------------------------------------------------------------------------
# holding time
# ---------------------------------------------------------------------------

def test_holding_time_summary():
    trades = [
        _trade(pnl=20.0, opened_hours_ago=10, closed_hours_ago=0),   # 10h winner
        _trade(pnl=30.0, opened_hours_ago=8, closed_hours_ago=0),    # 8h winner
        _trade(pnl=-5.0, opened_hours_ago=3, closed_hours_ago=0),    # 3h loser
    ]
    summary = ta.summarize_holding_time(trades)
    assert summary.trades_counted == 3
    assert summary.avg_hours == pytest.approx(7.0)
    assert summary.median_hours == pytest.approx(8.0)
    assert summary.shortest_hours == pytest.approx(3.0)
    assert summary.longest_hours == pytest.approx(10.0)
    assert summary.avg_winner_hours == pytest.approx(9.0)
    assert summary.avg_loser_hours == pytest.approx(3.0)


def test_cutting_winners_short_is_detectable():
    """Riding losers and cutting winners is the classic failure mode, and
    it is completely invisible in a win rate."""
    good = ta.summarize_holding_time([
        _trade(pnl=20.0, opened_hours_ago=12), _trade(pnl=-5.0, opened_hours_ago=2),
    ])
    bad = ta.summarize_holding_time([
        _trade(pnl=20.0, opened_hours_ago=2), _trade(pnl=-5.0, opened_hours_ago=12),
    ])
    assert good.winners_held_longer is True
    assert bad.winners_held_longer is False


def test_holding_time_is_unknown_without_both_timestamps():
    trade = _trade(pnl=10.0)
    trade.opened_at = None
    assert ta.holding_time_hours(trade) is None
    assert ta.summarize_holding_time([trade]).trades_counted == 0


def test_naive_timestamps_do_not_crash_the_subtraction():
    """SQLite hands back naive datetimes; mixing them with an aware one
    raises, which would take out the whole analytics page."""
    trade = _trade(pnl=10.0, opened_hours_ago=4)
    trade.opened_at = trade.opened_at.replace(tzinfo=None)
    assert ta.holding_time_hours(trade) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# extremes and concentration
# ---------------------------------------------------------------------------

def test_largest_win_and_loss_are_identified_with_their_symbols():
    trades = [
        _trade(pnl=12.0, symbol="SMALLWIN", pnl_pct=4.0),
        _trade(pnl=250.0, symbol="BIGWIN", pnl_pct=180.0),
        _trade(pnl=-80.0, symbol="BIGLOSS", pnl_pct=-40.0),
    ]
    ex = ta.find_extremes(trades)
    assert (ex.largest_win_usd, ex.largest_win_symbol) == (250.0, "BIGWIN")
    assert (ex.largest_loss_usd, ex.largest_loss_symbol) == (-80.0, "BIGLOSS")
    assert ex.largest_win_pct == 180.0
    assert ex.largest_loss_pct == -40.0


def test_a_strategy_carried_by_one_trade_is_flagged():
    """One lucky trade producing the entire edge is not a working strategy;
    it is a strategy that got lucky once."""
    lucky = ta.find_extremes([_trade(pnl=500.0)] + [_trade(pnl=-5.0) for _ in range(10)])
    assert lucky.best_trade_share_of_profit == pytest.approx(1.0)
    assert lucky.profit_depends_on_one_trade is True

    spread = ta.find_extremes([_trade(pnl=20.0) for _ in range(10)])
    assert spread.best_trade_share_of_profit == pytest.approx(0.1)
    assert spread.profit_depends_on_one_trade is False


def test_an_all_losing_record_has_no_largest_win():
    ex = ta.find_extremes([_trade(pnl=-10.0), _trade(pnl=-20.0)])
    assert ex.largest_win_usd is None
    assert ex.best_trade_share_of_profit is None
    assert ex.profit_depends_on_one_trade is False


def test_no_trades_yields_all_none():
    ex = ta.find_extremes([])
    assert ex.largest_win_usd is None and ex.largest_loss_usd is None


# ---------------------------------------------------------------------------
# breakdowns
# ---------------------------------------------------------------------------

def _signal(sid: int, score=None, quality=None) -> models.Signal:
    signal = models.Signal(
        symbol="X", chain="solana", signal_type="buy", price=1.0, raw_payload={},
        signal_score=score, market_quality_score=quality,
    )
    signal.id = sid
    return signal


def test_signal_score_breakdown_separates_good_setups_from_marginal_ones():
    signals = {1: _signal(1, score=62.0), 2: _signal(2, score=91.0)}
    trades = [
        _trade(pnl=-10.0, signal_id=1), _trade(pnl=-8.0, signal_id=1),
        _trade(pnl=40.0, signal_id=2),
    ]
    breakdown = ta.breakdown_by_signal_score(trades, signals)
    by_label = {b.label: b for b in breakdown.buckets}
    assert by_label["60-70"].trade_count == 2
    assert by_label["60-70"].total_pnl_usd == pytest.approx(-18.0)
    assert by_label["90+"].expectancy_usd == pytest.approx(40.0)


def test_a_trade_whose_score_was_not_recorded_is_counted_as_unknown():
    """Not folded into a bucket, and not silently dropped either - an
    unknown-score trade would otherwise distort whichever bucket absorbed
    it."""
    breakdown = ta.breakdown_by_signal_score([_trade(pnl=5.0, signal_id=None)], {})
    assert breakdown.unknown_count == 1
    assert breakdown.buckets == []


def test_small_buckets_are_shown_but_flagged_as_meaningless():
    """Hiding a thin bucket would make the table look more decisive than
    the data is; treating it as evidence would be worse."""
    signals = {1: _signal(1, score=85.0)}
    breakdown = ta.breakdown_by_signal_score([_trade(pnl=100.0, signal_id=1)], signals)
    bucket = breakdown.buckets[0]
    assert bucket.trade_count == 1
    assert bucket.meaningful is False
    assert breakdown.has_any_meaningful_bucket is False


def test_a_bucket_becomes_meaningful_at_the_threshold():
    signals = {1: _signal(1, score=85.0)}
    trades = [_trade(pnl=1.0, signal_id=1) for _ in range(ta.MIN_TRADES_FOR_A_MEANINGFUL_BUCKET)]
    breakdown = ta.breakdown_by_signal_score(trades, signals)
    assert breakdown.buckets[0].meaningful is True
    assert breakdown.has_any_meaningful_bucket is True


def test_market_quality_breakdown_uses_its_own_edges():
    signals = {1: _signal(1, quality=35.0), 2: _signal(2, quality=92.0)}
    breakdown = ta.breakdown_by_market_quality(
        [_trade(pnl=-20.0, signal_id=1), _trade(pnl=30.0, signal_id=2)], signals
    )
    labels = [b.label for b in breakdown.buckets]
    assert labels == ["<40", "85+"]


def test_liquidity_breakdown_reads_the_rug_check_liquidity():
    breakdown = ta.breakdown_by_liquidity(
        [_trade(pnl=5.0, signal_id=1), _trade(pnl=-5.0, signal_id=2)],
        {1: 18_000.0, 2: 400_000.0},
    )
    assert [b.label for b in breakdown.buckets] == ["<25000", "250000+"]


def test_holding_time_breakdown_labels_carry_their_unit():
    """Labels still carry a unit; the units are now per-edge.

    UPDATED, not loosened. This asserted ["<1h", "24-72h"] against the
    old edges [1, 4, 12, 24, 72]. Those edges were replaced because the
    champion closes in minutes and every trade it makes landed in the
    single "<1h" bucket - see tests/test_duration_resolution.py. The
    assertion is exactly as strict as before: a full, exact label list.
    What it now pins is that each edge renders in its own natural unit
    (30m, 1h, 1d) rather than all of them in hours.
    """
    breakdown = ta.breakdown_by_holding_time([
        _trade(pnl=5.0, opened_hours_ago=0.5), _trade(pnl=5.0, opened_hours_ago=30),
    ])
    assert [b.label for b in breakdown.buckets] == ["30m-1h", "1d-3d"]


def test_exit_reason_breakdown_shows_which_mechanism_earns_its_place():
    trades = [
        _trade(pnl=-15.0, close_reason="stop-loss hit"),
        _trade(pnl=-14.0, close_reason="stop-loss hit"),
        _trade(pnl=45.0, close_reason="take-profit hit"),
        _trade(pnl=8.0, close_reason=None),
    ]
    breakdown = ta.breakdown_by_exit_reason(trades)
    by_label = {b.label: b for b in breakdown.buckets}
    assert by_label["stop-loss hit"].trade_count == 2
    assert by_label["take-profit hit"].total_pnl_usd == pytest.approx(45.0)
    assert breakdown.unknown_count == 1


def test_open_trades_are_excluded_from_every_breakdown():
    """A position that hasn't closed has no realized P&L, and counting it
    as a zero-P&L trade would drag every expectancy toward zero."""
    trades = [_trade(pnl=None, close_reason=None), _trade(pnl=10.0, close_reason="take-profit hit")]
    assert ta.breakdown_by_exit_reason(trades).buckets[0].trade_count == 1
    assert len(ta.closed_trades(trades)) == 1


def test_breakdown_serialises():
    import json

    signals = {1: _signal(1, score=72.0)}
    payload = ta.breakdown_by_signal_score([_trade(pnl=5.0, signal_id=1)], signals).as_dict()
    json.dumps(payload)
    assert payload["dimension"] == "signal score"
    assert payload["buckets"][0]["meaningful"] is False


# ---------------------------------------------------------------------------
# rejections
# ---------------------------------------------------------------------------

def test_rejections_are_grouped_by_type_not_by_free_text():
    """Grouping on the detail string would produce one 'category' per
    rejection, since details embed symbols and numbers."""
    events = [
        models.RiskEvent(event_type="rug_check_rejected", details="TOKENA: honeypot"),
        models.RiskEvent(event_type="rug_check_rejected", details="TOKENB: dev owns 40%"),
        models.RiskEvent(event_type="signal_score_rejected", details="TOKENC: 58.2/100"),
        models.RiskEvent(event_type="trading_resumed", details="manually resumed"),
    ]
    summary = ta.summarize_rejections(events)
    assert summary.total == 3          # the resume event is not a rejection
    assert summary.by_reason[0] == ("rug_check_rejected", 2)
    assert summary.as_dict()["by_reason"][0]["share_pct"] == pytest.approx(66.7)


def test_no_rejections_does_not_divide_by_zero():
    assert ta.summarize_rejections([]).as_dict() == {"total": 0, "by_reason": []}


# ---------------------------------------------------------------------------
# strategy versions
# ---------------------------------------------------------------------------

def test_trades_are_grouped_by_the_configuration_that_produced_them():
    trades = [
        _trade(pnl=10.0, version="v-aaaa1111"),
        _trade(pnl=-5.0, version="v-aaaa1111"),
        _trade(pnl=30.0, version="v-bbbb2222"),
    ]
    grouped = ta.split_by_strategy_version(trades)
    assert set(grouped) == {"v-aaaa1111", "v-bbbb2222"}
    assert len(grouped["v-aaaa1111"]) == 2


def test_unversioned_trades_are_not_folded_into_the_current_version():
    """A trade from before versioning existed belongs to an unknown
    configuration, not to whichever one happens to be running now."""
    grouped = ta.split_by_strategy_version([_trade(pnl=10.0, version=None)])
    assert list(grouped) == ["unversioned"]


def test_the_open_ended_first_bucket_sorts_first():
    """The open-ended first bucket leads the table.

    UPDATED, not loosened. The original docstring described the reason as
    a parsing tiebreak: "<1h" and "1-4h" both parsed to 1, so ordering
    rendered labels needed a rule to keep the open-ended one first. That
    mechanism is gone - buckets are now grouped and sorted by interval
    INDEX, so nothing is recovered from the text and mixed-unit labels
    like "30m-1h" cannot confuse it.

    The PROPERTY is unchanged and still worth pinning, so the test stays
    with the same exact-list assertion against the current edges.
    """
    breakdown = ta.breakdown_by_holding_time([
        _trade(pnl=1.0, opened_hours_ago=0.5),
        _trade(pnl=1.0, opened_hours_ago=2),
        _trade(pnl=1.0, opened_hours_ago=30),
    ])
    assert [b.label for b in breakdown.buckets] == ["30m-1h", "1h-4h", "1d-3d"]


def test_the_open_ended_last_bucket_sorts_last():
    breakdown = ta.breakdown_by_liquidity(
        [_trade(pnl=1.0, signal_id=1), _trade(pnl=1.0, signal_id=2), _trade(pnl=1.0, signal_id=3)],
        {1: 900_000.0, 2: 10_000.0, 3: 60_000.0},
    )
    assert [b.label for b in breakdown.buckets] == ["<25000", "50000-100000", "250000+"]


def test_a_cost_percentage_with_no_notional_is_unmeasured_not_free():
    """execution_cost_pct is a fraction; without a notional it converts to
    exactly $0. Counting that leg as covered would understate the total and
    report 100% coverage while doing it."""
    trades = [
        _trade(pnl=10.0, fee=0.25, cost_pct=0.006, size=100.0),
        _trade(pnl=10.0, fee=None, cost_pct=0.006, size=0.0),   # size never recorded
    ]
    costs = ta.summarize_costs(trades)
    assert costs.legs_counted == 1
    assert costs.legs_missing_cost_data == 1
    assert costs.cost_data_complete is False
    assert costs.total_execution_cost_usd == pytest.approx(0.6)


def test_a_leg_priced_from_qty_is_still_counted():
    """size_usd can be absent on an exit leg; qty x price is the fallback,
    and it must not be mistaken for a missing notional."""
    trade = _trade(pnl=10.0, fee=0.25, cost_pct=0.01, size=0.0)
    trade.qty = 500.0
    trade.exit_price = 0.02          # $10 notional
    costs = ta.summarize_costs([trade])
    assert costs.legs_counted == 1
    assert costs.total_execution_cost_usd == pytest.approx(0.1)
