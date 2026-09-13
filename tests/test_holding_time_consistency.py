"""The holding-time summary and its breakdown must agree.

THE BUG THIS PINS

`opened_at` lives on the buy leg, `closed_at` on the sell. Resolving the
span needs both, which needs the entry index. `breakdown_by_holding_time`
passed one; `summarize_holding_time` did not.

So one report printed, of the same 24 closed trades:

    HOLDING TIME
      average / median     n/ah / n/ah        <- summary, no entry index
    by holding time
      <1h    24 trades                        <- breakdown, entry index

Both cannot be right. The summary was wrong, and it was wrong in the
quiet direction - "not measured" reads as missing data rather than as a
defect, so it survived the fix that repaired every other panel.

The invariant these tests hold is not "the number is 1.5 hours" but
"the two paths count the same trades". A future caller that forgets the
index again fails here rather than shipping two answers.
"""
import datetime as dt

from app import models
from app.analysis import trade_analytics as ta

NOW = dt.datetime.now(dt.timezone.utc)


def _pair(position_id, held_minutes, pnl):
    """A real position: entry carries opened_at, exit carries closed_at."""
    closed = NOW - dt.timedelta(minutes=5)
    entry = models.Trade(
        id=position_id * 10, position_id=position_id, signal_id=None,
        symbol="HOLDCOIN", side="buy", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=100.0, qty=100.0, entry_price=1.0,
        opened_at=closed - dt.timedelta(minutes=held_minutes),
        created_at=closed - dt.timedelta(minutes=held_minutes),
    )
    exit_ = models.Trade(
        id=position_id * 10 + 1, position_id=position_id, signal_id=None,
        symbol="HOLDCOIN", side="sell", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=100.0, qty=100.0, exit_price=1.0,
        pnl_usd=pnl, pnl_pct=pnl / 100.0,
        closed_at=closed, created_at=closed,
    )
    return [entry, exit_]


def test_the_summary_measures_what_the_breakdown_buckets():
    """The live report's exact contradiction: summary blank, breakdown full."""
    trades = _pair(1, 30, 2.0) + _pair(2, 45, -1.0) + _pair(3, 20, 3.0)

    summary = ta.summarize_holding_time(trades)
    breakdown = ta.breakdown_by_holding_time(trades)

    bucketed = sum(b.trade_count for b in breakdown.buckets)
    assert summary.trades_counted == bucketed == 3
    assert summary.avg_hours is not None


def test_the_summary_no_longer_reports_none_for_measurable_trades():
    """Pins the defect directly: this returned None for every field."""
    summary = ta.summarize_holding_time(_pair(1, 90, 1.0))
    assert summary.avg_hours == 1.5
    assert summary.median_hours == 1.5
    assert summary.shortest_hours == 1.5
    assert summary.longest_hours == 1.5


def test_winner_and_loser_spans_resolve_separately():
    """'Do winners run longer?' is unanswerable if neither side resolves."""
    trades = _pair(1, 120, 5.0) + _pair(2, 30, -5.0)
    s = ta.summarize_holding_time(trades)
    assert s.avg_winner_hours == 2.0
    assert s.avg_loser_hours == 0.5
    assert s.winners_held_longer is True


def test_a_position_with_no_entry_leg_stays_unmeasurable():
    """Not fabricated as zero - an orphan exit genuinely cannot be timed."""
    orphan = _pair(1, 30, 1.0)[1]          # sell leg only
    s = ta.summarize_holding_time([orphan])
    assert s.trades_counted == 0
    assert s.avg_hours is None


def test_measurable_and_unmeasurable_trades_coexist():
    """One orphan must not blank the trades that CAN be measured."""
    trades = _pair(1, 60, 1.0) + [_pair(9, 30, 1.0)[1]]
    s = ta.summarize_holding_time(trades)
    assert s.trades_counted == 1
    assert s.avg_hours == 1.0
