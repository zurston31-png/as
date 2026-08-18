"""Portfolio-level statistics computed from closed Trade rows, for the
dashboard.

Deliberately separate from app/backtesting/stats.py even though the
formulas overlap: that module works over BacktestTrade (a simulated,
in-memory dataclass produced by a single backtest run); this one works
over models.Trade (a live/paper SQLAlchemy row spanning the bot's whole
history). Keeping them apart avoids coupling the dashboard to backtesting
internals for what is otherwise the same simple math applied to a
differently-shaped input.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app import models


@dataclass
class PortfolioStats:
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    profit_factor: float | None
    expectancy_usd: float
    avg_win_usd: float | None
    avg_loss_usd: float | None
    max_drawdown_pct: float
    current_streak: int          # positive = winning streak, negative = losing streak, 0 = no closed trades yet
    longest_winning_streak: int
    longest_losing_streak: int


def _closed_sorted(trades: list[models.Trade]) -> list[models.Trade]:
    closed = [t for t in trades if t.pnl_usd is not None and t.closed_at is not None]
    return sorted(closed, key=lambda t: t.closed_at)


def compute_equity_curve(
    trades: list[models.Trade], starting_balance: float
) -> list[tuple[dt.datetime, float]]:
    """Cumulative realized equity over time, one point per closed trade
    (plus a starting point at the first trade's close, so a single trade
    still draws a visible line rather than one dot)."""
    closed = _closed_sorted(trades)
    if not closed:
        return []
    curve: list[tuple[dt.datetime, float]] = [(closed[0].closed_at, starting_balance)]
    running = starting_balance
    for t in closed:
        running += t.pnl_usd or 0.0
        curve.append((t.closed_at, running))
    return curve


def compute_portfolio_stats(trades: list[models.Trade], starting_balance: float) -> PortfolioStats:
    closed = _closed_sorted(trades)
    if not closed:
        return PortfolioStats(
            trade_count=0, win_count=0, loss_count=0, win_rate=0.0, profit_factor=None,
            expectancy_usd=0.0, avg_win_usd=None, avg_loss_usd=None, max_drawdown_pct=0.0,
            current_streak=0, longest_winning_streak=0, longest_losing_streak=0,
        )

    wins = [t for t in closed if (t.pnl_usd or 0) > 0]
    losses = [t for t in closed if (t.pnl_usd or 0) <= 0]
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = sum(t.pnl_usd for t in losses)  # <= 0

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    else:
        profit_factor = float("inf") if wins else None

    avg_win = (gross_profit / len(wins)) if wins else None
    avg_loss = (gross_loss / len(losses)) if losses else None
    expectancy = sum(t.pnl_usd for t in closed) / len(closed)

    equity_curve = compute_equity_curve(trades, starting_balance)
    peak = equity_curve[0][1] if equity_curve else starting_balance
    max_dd = 0.0
    for _, value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)

    longest_win = longest_loss = current_win = current_loss = 0
    for t in closed:
        if (t.pnl_usd or 0) > 0:
            current_win += 1
            current_loss = 0
        else:
            current_loss += 1
            current_win = 0
        longest_win = max(longest_win, current_win)
        longest_loss = max(longest_loss, current_loss)
    current_streak = current_win if current_win else -current_loss

    return PortfolioStats(
        trade_count=len(closed),
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=(len(wins) / len(closed) * 100),
        profit_factor=profit_factor,
        expectancy_usd=expectancy,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        max_drawdown_pct=max_dd * 100,
        current_streak=current_streak,
        longest_winning_streak=longest_win,
        longest_losing_streak=longest_loss,
    )
