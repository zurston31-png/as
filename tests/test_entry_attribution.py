"""Breakdowns must resolve entry context through the position.

THE BUG THIS PINS

Every per-attribute breakdown asks a question about the ENTRY - "did a
higher signal score trade better?", "did thin liquidity hurt?" - but is
computed over the EXIT legs, because only an exit carries realized P&L.

Entry context lives on the buy leg. An exit leg carries a `signal_id`
ONLY when a TradingView alert asked for the sell; every stop-loss,
take-profit and smart exit is raised by the position monitor and has
none. So `signals.get(exit.signal_id)` was `signals.get(None)` for the
majority of real trades, and the dashboard reported

    BY SIGNAL SCORE - N trade(s) with this value not recorded

for a book where every score HAD been recorded, on the buy leg, all
along. Holding time had the same shape: `opened_at` is stamped on the
buy, so measuring it off the sell gave None and the panel read "-h".

This is a reporting defect, not a strategy one. It changes no trading
decision - but while it stood, no amount of accumulated trades could
answer why the strategy was losing, because every attribute the answer
depends on read as missing.
"""
import datetime as dt

import pytest

from app import models
from app.analysis import trade_analytics as ta

NOW = dt.datetime.now(dt.timezone.utc)


def _entry(position_id, signal_id, *, minutes_ago=90, size=100.0):
    """The buy leg: carries the signal and the open timestamp."""
    return models.Trade(
        id=position_id * 10, position_id=position_id, signal_id=signal_id,
        symbol="ATTRCOIN", side="buy", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=size, qty=size, entry_price=1.0,
        fee_usd=0.25, execution_cost_pct=0.01,
        opened_at=NOW - dt.timedelta(minutes=minutes_ago),
        created_at=NOW - dt.timedelta(minutes=minutes_ago),
    )


def _monitor_exit(position_id, pnl, *, size=100.0):
    """The sell leg as the position monitor writes it.

    signal_id is None and opened_at is unset - that is the real shape of
    a stop-loss or take-profit exit, not a contrived one.
    """
    return models.Trade(
        id=position_id * 10 + 1, position_id=position_id, signal_id=None,
        symbol="ATTRCOIN", side="sell", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=size, qty=size, exit_price=1.0 + pnl / size,
        pnl_usd=pnl, pnl_pct=pnl / size,
        fee_usd=0.25, execution_cost_pct=0.01,
        closed_at=NOW,
        created_at=NOW,
    )


def _signal(signal_id, score, quality=70.0):
    return models.Signal(
        id=signal_id, symbol="ATTRCOIN", token_address="MintATTR",
        signal_type="buy", price=1.0,
        signal_score=score, market_quality_score=quality,
    )


def test_a_monitor_exit_still_resolves_its_entry_signal():
    """The core join. Without it every stop-loss exit is attributeless."""
    trades = [_entry(1, 501), _monitor_exit(1, -5.0)]
    entries = ta.entry_leg_by_position(trades)

    exit_leg = trades[1]
    assert exit_leg.signal_id is None, "fixture must use a real monitor exit"
    assert ta.entry_signal_id(exit_leg, entries) == 501


def test_signal_score_breakdown_is_populated_for_monitor_exits():
    """The symptom the dashboard showed: 'N trade(s) with this value not
    recorded' for a book where every score was recorded on the buy leg."""
    trades = [
        _entry(1, 501), _monitor_exit(1, -5.0),
        _entry(2, 502), _monitor_exit(2, 3.0),
    ]
    signals = {501: _signal(501, 55.0), 502: _signal(502, 85.0)}

    breakdown = ta.breakdown_by_signal_score(trades, signals)
    assert breakdown.unknown_count == 0, (
        f"{breakdown.unknown_count} closed trade(s) still have no signal score"
    )
    assert sum(b.trade_count for b in breakdown.buckets) == 2


def test_market_quality_breakdown_is_populated_for_monitor_exits():
    trades = [_entry(1, 501), _monitor_exit(1, -5.0)]
    breakdown = ta.breakdown_by_market_quality(trades, {501: _signal(501, 55.0, 42.0)})
    assert breakdown.unknown_count == 0


def test_liquidity_breakdown_is_populated_for_monitor_exits():
    """Keyed on signal_id via the rug-check row, so it broke identically."""
    trades = [_entry(1, 501), _monitor_exit(1, -5.0)]
    breakdown = ta.breakdown_by_liquidity(trades, {501: 30_000.0})
    assert breakdown.unknown_count == 0


def test_holding_time_is_measurable_from_the_entry_leg():
    """opened_at is on the buy, closed_at on the sell. Reading both off
    the sell gave None, which is why the panel showed '-h'."""
    trades = [_entry(1, 501, minutes_ago=90), _monitor_exit(1, -5.0)]
    entries = ta.entry_leg_by_position(trades)

    assert ta.holding_time_hours(trades[1]) is None, (
        "the exit leg alone genuinely cannot answer this"
    )
    assert ta.holding_time_hours(trades[1], entries) == pytest.approx(1.5, abs=0.05)

    breakdown = ta.breakdown_by_holding_time(trades)
    assert breakdown.unknown_count == 0


def test_a_tradingview_sell_keeps_its_own_signal():
    """The exit leg's own signal_id still wins when it has one - an alert
    -driven sell is attributable to the alert that asked for it."""
    trades = [_entry(1, 501), _monitor_exit(1, -5.0)]
    trades[1].signal_id = 999
    entries = ta.entry_leg_by_position(trades)
    assert ta.entry_signal_id(trades[1], entries) == 999


def test_an_exit_with_no_entry_leg_stays_unmeasurable():
    """A sell whose buy is absent from the queried window has genuinely
    unknown entry context. Unmeasurable, not guessed."""
    orphan = _monitor_exit(7, -2.0)
    entries = ta.entry_leg_by_position([orphan])
    assert ta.entry_signal_id(orphan, entries) is None
    assert ta.holding_time_hours(orphan, entries) is None
