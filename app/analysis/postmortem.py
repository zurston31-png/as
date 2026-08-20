"""One complete account of a single closed trade.

The journal already lists trades and the breakdowns already aggregate
them. This is the third view neither provides: everything about ONE trade
on one row, including the parts that only exist because the monitor was
watching while the position was open.

WHY THE PATH MATTERS MORE THAN THE OUTCOME

A closing price is a single number and it hides the trade. +5% that first
went -30% is a stop-loss that happened not to fire, not a winner - the
next one like it loses. -8% that first went +40% is a trailing stop set
too wide, not a bad entry. Both read identically in a P&L column, and
both call for a different fix.

So every post-mortem carries max favourable and max adverse excursion,
measured from the high and low water marks the position monitor records
on every tick. `capture` records how much of the peak the exit actually
kept, which is the one number that grades the exit logic rather than the
entry.

WHAT IS ESTIMATED AND WHAT IS MEASURED

MFE/MAE come from polled prices, not from tick data. A spike between two
polls is invisible, so both are LOWER BOUNDS on the true excursion, and
`samples` reports how many observations they were drawn from. Fees and
execution cost are exact - the paper fill model records them per leg.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app import models


def _aware(moment: dt.datetime | None) -> dt.datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


@dataclass
class PostMortem:
    position_id: int
    symbol: str
    token_address: str | None
    opened_at: dt.datetime | None
    closed_at: dt.datetime | None
    entry_price: float | None
    exit_price: float | None
    qty: float | None
    exit_reason: str | None
    realized_pnl_usd: float
    return_pct: float | None
    hold_minutes: float | None
    max_gain_pct: float | None          # MFE
    max_loss_pct: float | None          # MAE
    fees_usd: float
    execution_cost_pct: float | None
    slippage_pct: float | None
    signal_score: float | None
    market_quality_score: float | None
    rug_risk_score: float | None
    liquidity_at_entry_usd: float | None
    lowest_liquidity_usd: float | None
    samples: int
    strategy_version: str | None

    @property
    def capture(self) -> float | None:
        """Share of the peak the exit actually kept.

        1.0 means it exited at the high. Near 0 means the trade was in
        profit and gave it all back - which is a finding about the exit,
        not the entry, and is invisible in the return column.
        """
        if self.max_gain_pct is None or self.max_gain_pct <= 0 or self.return_pct is None:
            return None
        return self.return_pct / self.max_gain_pct

    @property
    def gave_back_a_winner(self) -> bool:
        """Was up 20%+ and closed flat or worse."""
        return bool(
            self.max_gain_pct is not None
            and self.max_gain_pct >= 20
            and (self.return_pct or 0) <= 0
        )

    @property
    def survived_a_drawdown(self) -> bool:
        """Closed green after being down 20%+.

        Worth flagging separately: these are the trades a tighter stop
        would have turned into losses, so they are the cost side of any
        argument for tightening one.
        """
        return bool(
            self.max_loss_pct is not None
            and self.max_loss_pct <= -20
            and (self.return_pct or 0) > 0
        )

    @property
    def liquidity_drop_pct(self) -> float | None:
        if not self.liquidity_at_entry_usd or self.lowest_liquidity_usd is None:
            return None
        return (1 - self.lowest_liquidity_usd / self.liquidity_at_entry_usd) * 100

    def headline(self) -> str:
        parts = [
            f"{self.symbol}: {self.return_pct:+.1f}%" if self.return_pct is not None
            else f"{self.symbol}: return unknown",
        ]
        if self.hold_minutes is not None:
            parts.append(
                f"held {self.hold_minutes:.0f}m" if self.hold_minutes < 120
                else f"held {self.hold_minutes / 60:.1f}h"
            )
        if self.max_gain_pct is not None and self.max_loss_pct is not None:
            parts.append(f"path {self.max_loss_pct:+.1f}% to {self.max_gain_pct:+.1f}%")
        if self.exit_reason:
            parts.append(self.exit_reason)
        line = " | ".join(parts)

        if self.gave_back_a_winner:
            line += f"  [gave back a {self.max_gain_pct:.0f}% winner]"
        elif self.survived_a_drawdown:
            line += f"  [survived a {abs(self.max_loss_pct):.0f}% drawdown]"
        return line

    def as_dict(self) -> dict:
        def r(v, n=4):
            return round(v, n) if v is not None else None

        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "token_address": self.token_address,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "exit_reason": self.exit_reason,
            "realized_pnl_usd": r(self.realized_pnl_usd, 2),
            "return_pct": r(self.return_pct, 3),
            "hold_minutes": r(self.hold_minutes, 1),
            "max_gain_pct": r(self.max_gain_pct, 3),
            "max_loss_pct": r(self.max_loss_pct, 3),
            "capture": r(self.capture, 3),
            "fees_usd": r(self.fees_usd, 4),
            "execution_cost_pct": r(self.execution_cost_pct, 4),
            "slippage_pct": r(self.slippage_pct, 4),
            "signal_score": self.signal_score,
            "market_quality_score": self.market_quality_score,
            "rug_risk_score": self.rug_risk_score,
            "liquidity_at_entry_usd": self.liquidity_at_entry_usd,
            "liquidity_drop_pct": r(self.liquidity_drop_pct, 1),
            "samples": self.samples,
            "strategy_version": self.strategy_version,
            "gave_back_a_winner": self.gave_back_a_winner,
            "survived_a_drawdown": self.survived_a_drawdown,
            "headline": self.headline(),
        }


def _pct(price: float | None, entry: float | None) -> float | None:
    if price is None or not entry or entry <= 0:
        return None
    return (price / entry - 1) * 100


def build_postmortem(db: Session, position: models.Position) -> PostMortem:
    """Assemble everything known about one closed position."""
    legs = (
        db.query(models.Trade)
        .filter(models.Trade.position_id == position.id)
        .order_by(models.Trade.created_at.asc())
        .all()
    )
    entry_leg = next((t for t in legs if t.side == "buy"), None)
    exit_legs = [t for t in legs if t.side == "sell"]

    # Fees are summed across every leg, entry and all partial exits. A
    # round trip taken in three pieces pays three times, and reporting only
    # the entry fee would understate the cost of the exit logic that split it.
    fees = sum(t.fee_usd or 0.0 for t in legs)
    costs = [t.execution_cost_pct for t in legs if t.execution_cost_pct is not None]

    exit_price = None
    if exit_legs:
        # Size-weighted, so a small scalp out followed by a large exit is
        # not averaged as if the two were equal.
        weighted = sum((t.exit_price or 0) * (t.qty or 0) for t in exit_legs)
        volume = sum(t.qty or 0 for t in exit_legs)
        exit_price = (weighted / volume) if volume else None

    signal = (
        db.query(models.Signal).filter(models.Signal.id == entry_leg.signal_id).first()
        if entry_leg and entry_leg.signal_id else None
    )
    rug = (
        db.query(models.RugCheckResult)
        .filter(models.RugCheckResult.signal_id == entry_leg.signal_id)
        .first()
        if entry_leg and entry_leg.signal_id else None
    )

    opened, closed = _aware(position.opened_at), _aware(position.closed_at)
    hold = (closed - opened).total_seconds() / 60 if opened and closed else None

    return PostMortem(
        position_id=position.id,
        symbol=position.symbol,
        token_address=position.token_address,
        opened_at=opened,
        closed_at=closed,
        entry_price=position.entry_price,
        exit_price=exit_price,
        qty=position.initial_qty or position.qty,
        exit_reason=position.close_reason,
        realized_pnl_usd=position.realized_pnl_usd or 0.0,
        return_pct=_pct(exit_price, position.entry_price),
        hold_minutes=hold,
        max_gain_pct=_pct(position.highest_price_since_entry, position.entry_price),
        max_loss_pct=_pct(position.lowest_price_since_entry, position.entry_price),
        fees_usd=fees,
        execution_cost_pct=(sum(costs) / len(costs)) if costs else None,
        # Execution cost minus the fee component is what actually moved the
        # fill away from mid. Reported separately because a fee is a known
        # constant and slippage is not - conflating them hides which one is
        # eating the edge.
        slippage_pct=(
            (sum(costs) / len(costs)) * 100 - (fees / (position.entry_price * (position.initial_qty or position.qty)) * 100)
            if costs and position.entry_price and (position.initial_qty or position.qty)
            else None
        ),
        signal_score=signal.signal_score if signal else None,
        market_quality_score=signal.market_quality_score if signal else None,
        rug_risk_score=rug.rug_risk_score if rug else None,
        liquidity_at_entry_usd=position.liquidity_at_entry_usd,
        lowest_liquidity_usd=position.lowest_liquidity_usd,
        samples=len(position.recent_prices or []),
        strategy_version=getattr(position, "strategy_version", None),
    )


def recent_postmortems(db: Session, *, limit: int = 50) -> list[PostMortem]:
    """Closed positions, newest first."""
    positions = (
        db.query(models.Position)
        .filter(models.Position.status == models.PositionStatus.CLOSED.value)
        .order_by(models.Position.closed_at.desc())
        .limit(limit)
        .all()
    )
    return [build_postmortem(db, p) for p in positions]
