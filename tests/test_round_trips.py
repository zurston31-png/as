"""Exit legs are not trades.

A position that takes a partial profit and is later closed writes two
filled sell rows. Counting those as two trades inflates the sample size,
distorts win rate and expectancy, breaks streak counting, and - worst -
feeds two correlated observations to machinery that assumes independent
ones. These tests pin the position as the unit of measurement.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.round_trips import group_round_trips

NOW = dt.datetime(2026, 9, 13, 12, 0, tzinfo=dt.timezone.utc)


def _leg(pnl, *, position_id, minutes_ago=0, symbol="RTCOIN"):
    return models.Trade(
        symbol=symbol, side="sell", status=models.TradeStatus.FILLED.value,
        position_id=position_id, pnl_usd=pnl, size_usd=100.0,
        qty=1.0, exit_price=100.0,
        closed_at=NOW - dt.timedelta(minutes=minutes_ago),
        strategy_version="v-test0001",
    )


# ---------------------------------------------------------------------------
# the core defect
# ---------------------------------------------------------------------------

def test_a_partial_and_its_final_close_are_one_round_trip():
    """The whole point. Two legs, one position, one observation."""
    trades = [
        _leg(20.0, position_id=1, minutes_ago=60),   # partial profit
        _leg(-30.0, position_id=1, minutes_ago=10),  # final close
    ]
    summary = group_round_trips(trades)

    assert summary.exit_leg_count == 2
    assert summary.count == 1
    assert summary.counts_agree is False
    assert summary.round_trips[0].pnl_usd == pytest.approx(-10.0)


def test_a_winning_partial_does_not_turn_a_losing_position_into_a_win():
    """Leg-level counting scores this as 1 win and 1 loss - a 50% win rate
    on a position that lost money. The position lost; that is one loss."""
    trades = [
        _leg(20.0, position_id=1, minutes_ago=60),
        _leg(-30.0, position_id=1, minutes_ago=10),
    ]
    summary = group_round_trips(trades)

    assert summary.wins == 0
    assert summary.win_rate_pct == pytest.approx(0.0)
    assert summary.round_trips[0].is_win is False


def test_expectancy_divides_by_positions_not_legs():
    """The denominator was inflated by however many partials fired.

    Three positions, one of which took a partial: four legs. Expectancy
    over legs divides the same P&L by 4 and understates it by 25%.
    """
    trades = [
        _leg(30.0, position_id=1, minutes_ago=90),
        _leg(10.0, position_id=2, minutes_ago=60),   # partial
        _leg(20.0, position_id=2, minutes_ago=50),   # final
        _leg(-30.0, position_id=3, minutes_ago=10),
    ]
    summary = group_round_trips(trades)

    assert summary.exit_leg_count == 4
    assert summary.count == 3
    assert summary.total_pnl_usd == pytest.approx(30.0)
    assert summary.expectancy_usd == pytest.approx(10.0)     # 30 / 3
    assert summary.expectancy_usd != pytest.approx(7.5)      # not 30 / 4


def test_a_winning_partial_cannot_break_a_losing_streak_of_positions():
    """Directly relevant to the consecutive-loss halt.

    Three losing positions in a row, the middle one having banked a small
    partial first. At leg level the streak reads as 1, because the winning
    partial interrupts it. At position level it is 3 - which is what a
    consecutive-loss rule is actually trying to detect.

    NOTE: this measures the streak; it does not change the risk manager,
    which still counts legs (app/risk/manager.py). That rule decides when
    the bot halts, so changing it is a live risk-behaviour change and is
    deliberately not part of this measurement fix.
    """
    trades = [
        _leg(-10.0, position_id=1, minutes_ago=100),
        _leg(5.0, position_id=2, minutes_ago=80),    # partial win...
        _leg(-25.0, position_id=2, minutes_ago=70),  # ...but the position lost
        _leg(-10.0, position_id=3, minutes_ago=10),
    ]
    summary = group_round_trips(trades)

    assert summary.count == 3
    assert summary.longest_losing_streak == 3


# ---------------------------------------------------------------------------
# not over-correcting
# ---------------------------------------------------------------------------

def test_a_book_with_no_partials_is_unchanged():
    """The fix must be a no-op on the common case.

    If every position closes in one go, legs and round trips are the same
    number and nothing about the reported record moves.
    """
    trades = [
        _leg(10.0, position_id=1, minutes_ago=30),
        _leg(-5.0, position_id=2, minutes_ago=20),
        _leg(7.0, position_id=3, minutes_ago=10),
    ]
    summary = group_round_trips(trades)

    assert summary.count == 3
    assert summary.exit_leg_count == 3
    assert summary.counts_agree is True
    assert summary.discrepancy_note() is None
    assert summary.win_rate_pct == pytest.approx(2 / 3 * 100)


def test_unrealized_and_unfilled_legs_are_not_observations():
    """Only realized exits count, matching closed_trades()."""
    pending = _leg(None, position_id=1, minutes_ago=10)
    pending.pnl_usd = None
    open_leg = _leg(5.0, position_id=2, minutes_ago=5)
    open_leg.closed_at = None

    summary = group_round_trips([pending, open_leg, _leg(10.0, position_id=3)])
    assert summary.count == 1
    assert summary.exit_leg_count == 1


def test_a_leg_with_no_position_id_is_counted_but_flagged():
    """Neither fabricate nor discard.

    A leg that cannot be attributed to a position might be a whole trade
    or might be a partial. Merging such legs would invent positions;
    dropping them would hide realized P&L. It is counted as one round trip
    and reported as unverified, so the resulting number is never read as
    exact.
    """
    orphan = _leg(12.0, position_id=None, minutes_ago=20)
    summary = group_round_trips([orphan, _leg(-4.0, position_id=1, minutes_ago=10)])

    assert summary.count == 2
    assert summary.unattributed_leg_count == 1
    assert summary.total_pnl_usd == pytest.approx(8.0)
    note = summary.discrepancy_note()
    assert note is not None and "no position_id" in note


def test_two_orphan_legs_are_never_merged_into_one_position():
    """Grouping them by anything but position_id would invent a position."""
    summary = group_round_trips([
        _leg(5.0, position_id=None, minutes_ago=20),
        _leg(6.0, position_id=None, minutes_ago=10),
    ])
    assert summary.count == 2
    assert summary.unattributed_leg_count == 2


def test_round_trips_are_ordered_oldest_close_first():
    """Streaks are chronological or they are not streaks."""
    summary = group_round_trips([
        _leg(1.0, position_id=2, minutes_ago=10),
        _leg(2.0, position_id=1, minutes_ago=90),
    ])
    assert [rt.position_id for rt in summary.round_trips] == [1, 2]


def test_a_positions_legs_are_ordered_within_the_round_trip():
    summary = group_round_trips([
        _leg(-30.0, position_id=1, minutes_ago=10),
        _leg(20.0, position_id=1, minutes_ago=60),
    ])
    legs = summary.round_trips[0].legs
    assert [t.pnl_usd for t in legs] == [20.0, -30.0]
    assert summary.round_trips[0].partial_count == 1


def test_an_empty_book_reports_nothing_rather_than_dividing_by_zero():
    summary = group_round_trips([])
    assert summary.count == 0
    assert summary.win_rate_pct is None
    assert summary.expectancy_usd is None
    assert summary.profit_factor is None
    assert summary.longest_losing_streak == 0


def test_an_all_winners_book_has_no_profit_factor_rather_than_infinity():
    """Same guard compute_stats uses: infinity is a symptom of a small
    sample, not a strength, and must not serialise as a number."""
    summary = group_round_trips([
        _leg(5.0, position_id=1, minutes_ago=20),
        _leg(6.0, position_id=2, minutes_ago=10),
    ])
    assert summary.profit_factor == float("inf")
    assert summary.win_rate_pct == pytest.approx(100.0)
