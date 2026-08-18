"""Backtest performance statistics.

Every number here is computed from the trade list and equity curve alone,
nothing is assumed - an empty trade list produces an all-zero/None stats
object rather than a divide-by-zero crash, since "the strategy never
entered a trade in this window" is itself a meaningful, common result.
"""
from __future__ import annotations

import datetime as dt
import math

from app.backtesting.types import BacktestStats, BacktestTrade


def _streaks(trades: list[BacktestTrade]) -> tuple[int, int]:
    """Longest winning streak, longest losing streak, in trade sequence order."""
    longest_win = longest_loss = 0
    current_win = current_loss = 0
    for t in trades:
        if t.is_win:
            current_win += 1
            current_loss = 0
        else:
            current_loss += 1
            current_win = 0
        longest_win = max(longest_win, current_win)
        longest_loss = max(longest_loss, current_loss)
    return longest_win, longest_loss


def _max_drawdown_pct(equity_curve: list[tuple[dt.datetime, float]]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0][1]
    max_dd = 0.0
    for _, value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
    return max_dd * 100


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _sharpe_and_sortino(trades: list[BacktestTrade]) -> tuple[float | None, float | None]:
    """Per-trade Sharpe/Sortino, annualized by observed trade frequency.

    This is a trade-return series, not a fixed-interval (daily/monthly) one,
    so there is no single "correct" annualization - trades don't happen at
    a fixed cadence. Annualizing by how many trades this backtest actually
    produced per year is a common, defensible approximation, but it is an
    approximation: treat these as directionally useful for comparing
    strategies against EACH OTHER on the same data, not as a literal
    finance-textbook Sharpe ratio.
    """
    if len(trades) < 2:
        return None, None

    returns = [t.pnl_pct for t in trades]
    mean_return = _mean(returns)
    std_return = _stdev(returns)

    span_days = (trades[-1].exit_time - trades[0].entry_time).total_seconds() / 86400
    trades_per_year = (len(trades) / span_days * 365) if span_days > 0 else len(trades)
    annualization = math.sqrt(max(trades_per_year, 1e-9))

    # A near-zero standard deviation from float rounding (e.g. three
    # identical 0.1 returns don't sum to an exact 0.3) must read as "no
    # variance" the same as a true zero would, not blow the ratio up toward
    # infinity - an epsilon guard, not a bare `> 0`.
    epsilon = 1e-9
    sharpe = (mean_return / std_return * annualization) if std_return > epsilon else None

    downside = [r for r in returns if r < 0]
    downside_std = _stdev(downside) if len(downside) >= 2 else (abs(downside[0]) if downside else 0.0)
    sortino = (mean_return / downside_std * annualization) if downside_std > epsilon else None

    return sharpe, sortino


def compute_stats(
    trades: list[BacktestTrade],
    equity_curve: list[tuple[dt.datetime, float]],
    starting_balance_usd: float,
) -> BacktestStats:
    trade_count = len(trades)
    final_balance = equity_curve[-1][1] if equity_curve else starting_balance_usd

    if trade_count == 0:
        return BacktestStats(
            trade_count=0, win_count=0, loss_count=0, win_rate=0.0,
            total_return_pct=0.0, total_return_usd=0.0, final_balance_usd=final_balance,
            avg_win_usd=None, avg_loss_usd=None, profit_factor=None,
            expectancy_usd=0.0, expectancy_r=None, avg_r_multiple=None,
            max_drawdown_pct=_max_drawdown_pct(equity_curve),
            sharpe_ratio=None, sortino_ratio=None,
            longest_winning_streak=0, longest_losing_streak=0,
        )

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]

    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = sum(t.pnl_usd for t in losses)  # <= 0

    avg_win_usd = (gross_profit / len(wins)) if wins else None
    avg_loss_usd = (gross_loss / len(losses)) if losses else None
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else (None if not wins else float("inf"))

    total_return_usd = final_balance - starting_balance_usd
    total_return_pct = (total_return_usd / starting_balance_usd * 100) if starting_balance_usd else 0.0

    r_multiples = [t.r_multiple for t in trades]
    avg_r = _mean(r_multiples) if r_multiples else None

    longest_win_streak, longest_loss_streak = _streaks(trades)
    sharpe, sortino = _sharpe_and_sortino(trades)

    return BacktestStats(
        trade_count=trade_count,
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=(len(wins) / trade_count * 100),
        total_return_pct=total_return_pct,
        total_return_usd=total_return_usd,
        final_balance_usd=final_balance,
        avg_win_usd=avg_win_usd,
        avg_loss_usd=avg_loss_usd,
        profit_factor=profit_factor,
        expectancy_usd=_mean([t.pnl_usd for t in trades]),
        expectancy_r=avg_r,
        avg_r_multiple=avg_r,
        max_drawdown_pct=_max_drawdown_pct(equity_curve),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        longest_winning_streak=longest_win_streak,
        longest_losing_streak=longest_loss_streak,
    )
