"""Performance statistics for a set of closed trades.

Everything is reported in R (risk units) as well as currency: a strategy whose
edge only exists at one position size isn't an edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..models import ClosedTrade


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    longest_loss_streak: int = 0
    final_equity: float = 0.0
    return_pct: float = 0.0
    # Share of trades that reached each target. TP1 is `entry.tp1_r` (2R by
    # default) and TP2 is `entry.tp2_r` (4R), so these are the "2R hit rate" and
    # "4R hit rate" - the numbers that say whether the targets are reachable at
    # all, which expectancy alone hides.
    tp1_hit_rate: float = 0.0
    tp2_hit_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}

    def render(self) -> str:
        if not self.trades:
            return "No trades taken."
        rows = [
            ("Trades", f"{self.trades}  ({self.wins}W / {self.losses}L)"),
            ("Win rate", f"{self.win_rate * 100:.1f}%"),
            ("Net P&L", f"{self.net_pnl:+,.2f}"),
            ("Return", f"{self.return_pct:+.2f}%"),
            ("Profit factor", f"{self.profit_factor:.2f}" if self.profit_factor else "inf"),
            ("Expectancy", f"{self.expectancy_r:+.3f}R per trade"),
            ("Avg win / loss", f"{self.avg_win_r:+.2f}R / {self.avg_loss_r:+.2f}R"),
            ("Max drawdown", f"{self.max_drawdown:,.2f}  ({self.max_drawdown_pct:.2f}%)"),
            ("Worst streak", f"{self.longest_loss_streak} losses in a row"),
            ("Target hit rate", f"TP1 {self.tp1_hit_rate * 100:.0f}% · "
                                f"TP2 {self.tp2_hit_rate * 100:.0f}%"),
            ("Final equity", f"{self.final_equity:,.2f}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k.ljust(width)}  {v}" for k, v in rows)


def compute(trades: Sequence[ClosedTrade], starting_equity: float) -> Metrics:
    m = Metrics(final_equity=starting_equity)
    if not trades:
        return m

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    m.trades = len(trades)
    m.wins, m.losses = len(wins), len(losses)
    m.win_rate = m.wins / m.trades
    m.net_pnl = sum(t.pnl for t in trades)
    m.gross_profit = sum(t.pnl for t in wins)
    m.gross_loss = abs(sum(t.pnl for t in losses))
    m.profit_factor = m.gross_profit / m.gross_loss if m.gross_loss else 0.0
    m.expectancy_r = sum(t.r_multiple for t in trades) / m.trades
    m.avg_win_r = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
    m.avg_loss_r = sum(t.r_multiple for t in losses) / len(losses) if losses else 0.0

    # A trade "reached TP1" if it exited there, ran on to TP2, or was stopped
    # out only after taking its partial - all three printed the first target.
    reached_tp1 = [t for t in trades if t.exit_reason in ("tp1", "tp2", "stop_after_tp1")]
    reached_tp2 = [t for t in trades if t.exit_reason == "tp2"]
    m.tp1_hit_rate = len(reached_tp1) / m.trades
    m.tp2_hit_rate = len(reached_tp2) / m.trades

    equity = starting_equity
    peak = starting_equity
    streak = 0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        drawdown = peak - equity
        if drawdown > m.max_drawdown:
            m.max_drawdown = drawdown
            m.max_drawdown_pct = drawdown / peak * 100 if peak else 0.0
        streak = streak + 1 if t.pnl <= 0 else 0
        m.longest_loss_streak = max(m.longest_loss_streak, streak)

    m.final_equity = equity
    m.return_pct = (equity - starting_equity) / starting_equity * 100 if starting_equity else 0.0
    return m
