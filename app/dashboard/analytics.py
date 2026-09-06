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
from app.analysis import trade_analytics as ta
from app.dashboard.charts import curve_positions


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


@dataclass(frozen=True)
class EquityPoint:
    """One vertex of the equity curve, with the trade that produced it.

    The curve was previously just (time, value) pairs, which is all a
    polyline needs but leaves the chart unreadable: a step down says the
    book lost money and nothing about which token, which exit rule, or how
    long the position was held. Carrying the trade through means the
    dashboard can answer "what happened here?" from the same pass that
    draws the line, with no second query keyed on a timestamp.
    """

    at: dt.datetime
    equity_usd: float
    equity_before_usd: float
    peak_usd: float
    drawdown_pct: float
    trade_number: int                # 1-based; 0 is the opening balance
    trade: models.Trade | None       # None only on the opening point


def compute_equity_points(
    trades: list[models.Trade], starting_balance: float
) -> list[EquityPoint]:
    """Cumulative realized equity over time, one point per closed trade
    (plus a starting point at the first trade's close, so a single trade
    still draws a visible line rather than one dot)."""
    closed = _closed_sorted(trades)
    if not closed:
        return []

    points = [
        EquityPoint(
            at=closed[0].closed_at,
            equity_usd=starting_balance,
            equity_before_usd=starting_balance,
            peak_usd=starting_balance,
            drawdown_pct=0.0,
            trade_number=0,
            trade=None,
        )
    ]
    running = peak = starting_balance
    for number, t in enumerate(closed, start=1):
        before = running
        running += t.pnl_usd or 0.0
        peak = max(peak, running)
        points.append(
            EquityPoint(
                at=t.closed_at,
                equity_usd=running,
                equity_before_usd=before,
                peak_usd=peak,
                drawdown_pct=((peak - running) / peak * 100) if peak > 0 else 0.0,
                trade_number=number,
                trade=t,
            )
        )
    return points


def compute_equity_curve(
    trades: list[models.Trade], starting_balance: float
) -> list[tuple[dt.datetime, float]]:
    """The plain (time, value) curve, for callers that only draw the line.

    Derived from compute_equity_points rather than computed alongside it,
    so the line and the per-point detail can never describe different
    numbers.
    """
    return [(p.at, p.equity_usd) for p in compute_equity_points(trades, starting_balance)]


@dataclass(frozen=True)
class EquityMarker:
    """One hoverable/tappable point on the rendered curve.

    Display-ready on purpose: the template positions it and prints it, and
    does no arithmetic. In particular execution cost arrives here already
    multiplied by 100 - `Trade.execution_cost_pct` is a FRACTION despite
    the name (see docs/GLOSSARY.md), and that unit trap has produced a
    wrong number on this dashboard before.
    """

    x_pct: float
    y_pct: float
    trade_number: int
    total_trades: int
    is_start: bool
    won: bool | None
    symbol: str | None
    token_address: str | None
    closed_at: dt.datetime | None
    pnl_usd: float | None
    pnl_pct: float | None
    equity_before_usd: float
    equity_usd: float
    peak_usd: float
    drawdown_pct: float
    exit_reason: str | None
    holding_time: str | None
    size_usd: float | None
    fee_usd: float | None
    execution_cost_pct: float | None     # percent, not the stored fraction
    entry_price: float | None
    exit_price: float | None
    strategy_version: str | None
    mode: str | None


def build_equity_markers(
    points: list[EquityPoint], trades: list[models.Trade]
) -> list[EquityMarker]:
    """Attach a position on the chart, and the trade's story, to each point.

    `trades` is the whole book rather than just the closed legs because the
    entry context - when the position opened, what it was bought at - lives
    on the BUY leg, and an exit leg carries neither. That join is the same
    one app/analysis/trade_analytics.py makes for every breakdown, so it
    reuses those helpers instead of restating the rule.
    """
    positions = curve_positions([p.equity_usd for p in points])
    if not positions:
        return []

    entries = ta.entry_leg_by_position(trades)
    total = max((p.trade_number for p in points), default=0)

    markers: list[EquityMarker] = []
    for (x_pct, y_pct), point in zip(positions, points):
        t = point.trade
        entry = entries.get(t.position_id) if t is not None else None
        markers.append(
            EquityMarker(
                x_pct=x_pct,
                y_pct=y_pct,
                trade_number=point.trade_number,
                total_trades=total,
                is_start=t is None,
                won=None if t is None else (t.pnl_usd or 0.0) > 0,
                symbol=t.symbol if t is not None else None,
                token_address=t.token_address if t is not None else None,
                closed_at=point.at,
                pnl_usd=t.pnl_usd if t is not None else None,
                pnl_pct=t.pnl_pct if t is not None else None,
                equity_before_usd=point.equity_before_usd,
                equity_usd=point.equity_usd,
                peak_usd=point.peak_usd,
                drawdown_pct=point.drawdown_pct,
                exit_reason=t.close_reason if t is not None else None,
                holding_time=(
                    ta.format_duration_hours(ta.holding_time_hours(t, entries))
                    if t is not None
                    else None
                ),
                size_usd=t.size_usd if t is not None else None,
                fee_usd=t.fee_usd if t is not None else None,
                execution_cost_pct=(
                    t.execution_cost_pct * 100
                    if t is not None and t.execution_cost_pct is not None
                    else None
                ),
                entry_price=entry.entry_price if entry is not None else None,
                exit_price=t.exit_price if t is not None else None,
                strategy_version=t.strategy_version if t is not None else None,
                mode=t.mode if t is not None else None,
            )
        )
    return markers


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
