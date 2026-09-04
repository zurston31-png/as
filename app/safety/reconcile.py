"""Portfolio accounting reconciliation.

The cash ledger is maintained incrementally: every buy subtracts, every
sell adds. Incremental ledgers drift. A lost write, a double-applied
adjustment, a rolled-back transaction that had already moved cash - each
leaves the balance quietly wrong, and every subsequent position size is
computed from that wrong number.

So the balance is checked against the thing it is supposed to summarise:

    expected cash = starting balance
                  + sum(proceeds of every FILLED sell)
                  - sum(cost of every FILLED buy)

If the recorded balance and the expected balance disagree by more than a
rounding tolerance, the books are wrong and the correct response is to
stop opening positions until someone looks - not to carry on sizing trades
off a number known to be false.

WHY IT DOES NOT AUTO-CORRECT. Silently overwriting the ledger with the
computed value would hide the bug that caused the drift and destroy the
evidence needed to find it. A discrepancy is reported, not patched.

Floating-point addition over thousands of trades accumulates real error,
so the tolerance is proportional to the number of trades rather than a
flat cent - a fixed threshold would either false-positive on a long
history or miss a genuine loss on a short one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.services import portfolio

logger = logging.getLogger(__name__)

# Per-trade allowance for accumulated float error, in dollars.
FLOAT_TOLERANCE_PER_TRADE = 1e-6

# Floor on the tolerance, so a fresh database with two trades still has
# room for ordinary representation error.
MIN_TOLERANCE_USD = 0.01


@dataclass
class Reconciliation:
    recorded_cash: float
    expected_cash: float
    buys_usd: float
    sells_usd: float
    filled_trades: int
    tolerance_usd: float
    problems: list[str] = field(default_factory=list)

    @property
    def discrepancy(self) -> float:
        return self.recorded_cash - self.expected_cash

    @property
    def balanced(self) -> bool:
        return abs(self.discrepancy) <= self.tolerance_usd and not self.problems

    def as_dict(self) -> dict:
        return {
            "balanced": self.balanced,
            "recorded_cash": round(self.recorded_cash, 4),
            "expected_cash": round(self.expected_cash, 4),
            "discrepancy": round(self.discrepancy, 4),
            "tolerance_usd": round(self.tolerance_usd, 6),
            "buys_usd": round(self.buys_usd, 2),
            "sells_usd": round(self.sells_usd, 2),
            "filled_trades": self.filled_trades,
            "problems": list(self.problems),
        }

    def summary(self) -> str:
        if self.balanced:
            return (
                f"Books balance: ${self.recorded_cash:,.2f} cash across {self.filled_trades} "
                f"filled trades (discrepancy ${self.discrepancy:+.6f}, within tolerance)."
            )
        lines = [
            f"ACCOUNTING DISCREPANCY: ledger says ${self.recorded_cash:,.2f}, the trade "
            f"record implies ${self.expected_cash:,.2f} - a gap of ${self.discrepancy:+,.4f} "
            f"(tolerance ${self.tolerance_usd:.6f})."
        ]
        for problem in self.problems:
            lines.append(f"  - {problem}")
        lines.append(
            "  Not auto-corrected: overwriting the ledger would hide whatever caused the drift."
        )
        return "\n".join(lines)


def reconcile(db: Session) -> Reconciliation:
    """Check the cash ledger against the trade record."""
    trades = (
        db.query(models.Trade)
        .filter(models.Trade.status == models.TradeStatus.FILLED.value)
        .all()
    )

    buys = 0.0
    sells = 0.0
    problems: list[str] = []

    for trade in trades:
        if trade.side == "buy":
            if trade.size_usd is None:
                problems.append(f"filled buy trade {trade.id} has no size_usd - cannot reconcile it")
                continue
            buys += trade.size_usd
        elif trade.side == "sell":
            if trade.qty is None or trade.exit_price is None:
                problems.append(
                    f"filled sell trade {trade.id} is missing qty or exit_price - "
                    "its proceeds cannot be reconstructed"
                )
                continue
            sells += trade.qty * trade.exit_price
        else:
            problems.append(f"trade {trade.id} has unrecognised side {trade.side!r}")

    expected = settings.PORTFOLIO_STARTING_BALANCE_USD + sells - buys
    recorded = portfolio.get_cash_balance_usd(db)
    tolerance = max(MIN_TOLERANCE_USD, len(trades) * FLOAT_TOLERANCE_PER_TRADE)

    if recorded < 0:
        problems.append(
            f"cash balance is NEGATIVE (${recorded:,.2f}) - the bot has spent money it "
            "did not have, which no sequence of valid trades can produce"
        )

    return Reconciliation(
        recorded_cash=recorded,
        expected_cash=expected,
        buys_usd=buys,
        sells_usd=sells,
        filled_trades=len(trades),
        tolerance_usd=tolerance,
        problems=problems,
    )


def check_position_integrity(db: Session) -> list[str]:
    """Structural problems in the open book that arithmetic would not catch.

    Each of these is impossible under correct operation, so finding one
    means a write went wrong or a process died mid-transaction - and each
    would corrupt sizing or exits if trading continued on top of it.
    """
    problems: list[str] = []
    open_positions = (
        db.query(models.Position)
        .filter(models.Position.status == models.PositionStatus.OPEN.value)
        .all()
    )

    for pos in open_positions:
        if pos.qty is None or pos.qty <= 0:
            problems.append(
                f"position {pos.id} ({pos.symbol}) is OPEN with qty={pos.qty} - "
                "an open position holding nothing cannot be exited"
            )
        if pos.entry_price is None or pos.entry_price <= 0:
            problems.append(
                f"position {pos.id} ({pos.symbol}) has entry_price={pos.entry_price} - "
                "every P&L and stop calculation divides by it"
            )
        if pos.stop_loss is not None and pos.entry_price and pos.stop_loss >= pos.entry_price:
            problems.append(
                f"position {pos.id} ({pos.symbol}) has a stop at or above its entry "
                f"(${pos.stop_loss:.8f} vs ${pos.entry_price:.8f}) - it would exit immediately"
            )
        if pos.initial_qty is not None and pos.qty and pos.qty > pos.initial_qty * 1.000001:
            problems.append(
                f"position {pos.id} ({pos.symbol}) holds more than it was opened with "
                f"({pos.qty} > {pos.initial_qty}) - a partial exit went the wrong way"
            )

    # A closed position with no exit trade means the exit leg was lost.
    closed_without_exit = (
        db.query(models.Position)
        .filter(models.Position.status == models.PositionStatus.CLOSED.value)
        .all()
    )
    for pos in closed_without_exit:
        exits = (
            db.query(models.Trade)
            .filter(
                models.Trade.position_id == pos.id,
                models.Trade.side == "sell",
                models.Trade.status == models.TradeStatus.FILLED.value,
            )
            .count()
        )
        if exits == 0:
            problems.append(
                f"position {pos.id} ({pos.symbol}) is CLOSED but has no filled sell trade - "
                "its proceeds were never credited to the ledger"
            )

    return problems
