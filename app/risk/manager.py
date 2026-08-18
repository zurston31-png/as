"""Risk management. Every buy and every close goes through this module —
there is no other code path that opens or closes a position.

Config values are read from Settings, but are additionally clamped to
absolute hard ceilings defined here so a misconfigured .env can never
produce a catastrophically oversized trade, an unbounded daily loss, or an
unbounded number of concurrent positions.
"""
import datetime as dt
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.state import get_state, set_state

# Absolute ceilings - config is clamped to these no matter what .env says.
HARD_MAX_PORTFOLIO_PCT_PER_TRADE = 0.10
HARD_MAX_DAILY_LOSS_PCT = 0.25
HARD_MIN_STOP_LOSS_PCT = 0.03
HARD_MAX_CONCURRENT_POSITIONS = 20

HALT_KEY = "trading_halted"
HALT_REASON_KEY = "trading_halt_reason"


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""


def is_trading_halted(db: Session) -> bool:
    return bool(get_state(db, HALT_KEY, False))


def halt_trading(db: Session, reason: str) -> None:
    set_state(db, HALT_KEY, True)
    set_state(db, HALT_REASON_KEY, reason)
    db.add(models.RiskEvent(event_type="trading_halted", details=reason))


def resume_trading(db: Session) -> None:
    set_state(db, HALT_KEY, False)
    set_state(db, HALT_REASON_KEY, None)
    db.add(models.RiskEvent(event_type="trading_resumed", details="manually resumed"))


class RiskManager:
    def __init__(self):
        self.max_pct_per_trade = min(settings.MAX_PORTFOLIO_PCT_PER_TRADE, HARD_MAX_PORTFOLIO_PCT_PER_TRADE)
        self.daily_loss_limit_pct = min(settings.DAILY_LOSS_LIMIT_PCT, HARD_MAX_DAILY_LOSS_PCT)
        self.stop_loss_pct = max(settings.STOP_LOSS_PCT, HARD_MIN_STOP_LOSS_PCT)
        self.take_profit_pct = settings.TAKE_PROFIT_PCT
        self.max_concurrent_positions = min(settings.MAX_CONCURRENT_POSITIONS, HARD_MAX_CONCURRENT_POSITIONS)

    # ---- sizing ----
    def position_size_usd(self, portfolio_value_usd: float) -> float:
        size = portfolio_value_usd * self.max_pct_per_trade
        return min(size, settings.MAX_TRADE_SIZE_USD)

    def stop_loss_take_profit(self, entry_price: float) -> tuple[float, float]:
        sl = entry_price * (1 - self.stop_loss_pct)
        tp = entry_price * (1 + self.take_profit_pct)
        return sl, tp

    # ---- gating checks that must pass before a buy is allowed ----
    def check_can_open_position(self, db: Session) -> RiskDecision:
        if is_trading_halted(db):
            reason = get_state(db, HALT_REASON_KEY, "") or "trading halted"
            return RiskDecision(False, f"trading halted: {reason}")

        open_count = db.query(models.Position).filter_by(status=models.PositionStatus.OPEN.value).count()
        if open_count >= self.max_concurrent_positions:
            return RiskDecision(
                False, f"max concurrent positions reached ({open_count}/{self.max_concurrent_positions})"
            )

        return RiskDecision(True)

    def evaluate_daily_loss(self, db: Session) -> RiskDecision:
        today = dt.datetime.now(dt.timezone.utc).date()
        starting_balance = settings.PORTFOLIO_STARTING_BALANCE_USD
        closed_today = [
            t
            for t in db.query(models.Trade).filter(models.Trade.closed_at.isnot(None)).all()
            if t.closed_at.date() == today
        ]
        realized_today = sum(t.pnl_usd or 0.0 for t in closed_today)
        loss_limit = starting_balance * self.daily_loss_limit_pct
        if realized_today <= -loss_limit:
            return RiskDecision(
                False, f"daily realized loss ${realized_today:,.2f} breached limit -${loss_limit:,.2f}"
            )
        return RiskDecision(True)
