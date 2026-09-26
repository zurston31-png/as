"""Holding time must be reported at the resolution the strategy trades at.

THE DEFECT THIS PINS

The champion closes positions in MINUTES. A live book of 24 trades
reported:

    average / median     0.1h / 0.1h
    shortest / longest   0.0h / 0.4h
    by holding time
      <1h    24 trades    62%    -1.38

Every number is technically correct and none of them is usable. At one
decimal in hours, "0.0h" spans ten seconds to three minutes, and a
breakdown whose first edge is 1h cannot separate a book where the
LONGEST hold is 24 minutes. A breakdown that puts every trade in one
bucket is not a breakdown.

Both are read-side presentation: no scoring, threshold, weight, exit
policy, fee, slippage or classifier is involved, and app/analysis is not
covered by the strategy version hash.
"""
from app import models
from app.analysis import trade_analytics as ta

import datetime as dt

NOW = dt.datetime.now(dt.timezone.utc)


def _pair(position_id, held_minutes, pnl=1.0):
    closed = NOW - dt.timedelta(minutes=1)
    entry = models.Trade(
        id=position_id * 10, position_id=position_id, side="buy",
        symbol="DURCOIN", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=100.0, qty=100.0, entry_price=1.0,
        opened_at=closed - dt.timedelta(minutes=held_minutes),
        created_at=closed - dt.timedelta(minutes=held_minutes),
    )
    exit_ = models.Trade(
        id=position_id * 10 + 1, position_id=position_id, side="sell",
        symbol="DURCOIN", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=100.0, qty=100.0, exit_price=1.0,
        pnl_usd=pnl, pnl_pct=pnl / 100.0,
        closed_at=closed, created_at=closed,
    )
    return [entry, exit_]


def test_sub_minute_holds_read_in_seconds():
    assert ta.format_duration_hours(7 / 3600) == "7s"


def test_sub_hour_holds_read_in_minutes():
    """0.1h and 0.4h were indistinguishable to a reader; 6m and 24m are not."""
    assert ta.format_duration_hours(0.1) == "6m"
    assert ta.format_duration_hours(0.4) == "24m"


def test_hour_and_day_scales_still_read_naturally():
    assert ta.format_duration_hours(3.7) == "3.7h"
    assert ta.format_duration_hours(26.0) == "1.1d"


def test_unmeasurable_stays_none_rather_than_becoming_zero():
    """A duration that could not be measured must not render as '0s'."""
    assert ta.format_duration_hours(None) is None


def test_a_minutes_scale_book_no_longer_collapses_into_one_bucket():
    """The live shape: 24 trades, longest 24 minutes, previously all '<1h'."""
    trades = []
    for i, minutes in enumerate([2, 3, 4, 7, 9, 12, 18, 22, 24], start=1):
        trades += _pair(i, minutes)
    b = ta.breakdown_by_holding_time(trades)
    assert len(b.buckets) > 1, "still one bucket - no resolution gained"
    assert [x.label for x in b.buckets] == ["<5m", "5m-15m", "15m-30m"]
    assert [x.trade_count for x in b.buckets] == [3, 3, 3]


def test_bucket_labels_render_each_edge_in_its_own_unit():
    labels = [ta._holding_edge_label(e) for e in ta.HOLDING_TIME_EDGES_HOURS]
    assert labels == ["5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d"]


def test_slow_holds_still_have_somewhere_to_go():
    """Resolution at the short end must not truncate the long end - a
    slower challenger has to remain expressible in the same report."""
    trades = _pair(1, 2) + _pair(2, 60 * 6) + _pair(3, 60 * 24 * 5)
    b = ta.breakdown_by_holding_time(trades)
    assert [x.label for x in b.buckets] == ["<5m", "4h-12h", "3d+"]


def test_other_breakdowns_keep_their_existing_labels():
    """The bucketing refactor changed how labels are ORDERED, not how the
    numeric dimensions render. A silent relabelling would split every
    historical comparison."""
    pairs = [(65.0, None), (75.0, None), (95.0, None)]
    assert ta._bucket_label(65.0, [20, 40, 60, 80], "") == "60-80"
    assert ta._bucket_label(10.0, [20, 40, 60, 80], "") == "<20"
    assert ta._bucket_label(95.0, [20, 40, 60, 80], "") == "80+"
    assert len(pairs) == 3
