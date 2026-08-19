"""Risk management. Every buy and every close goes through this module —
there is no other code path that opens or closes a position.

Config values are read from Settings, but are additionally clamped to
absolute hard ceilings defined here so a misconfigured .env can never
produce a catastrophically oversized trade, an unbounded daily loss, or an
unbounded number of concurrent positions.

Position sizing is risk-based, not notional-based: `position_size_usd`
sizes the trade so that a stop-loss hit loses exactly `risk_pct_per_trade`
of the portfolio, whatever the stop distance happens to be. The earlier
version sized every trade at a fixed percent of the portfolio regardless of
where the stop was set, so "2% risk per trade" was never actually the
dollar amount at risk — it was the notional size, and the real risk moved
around with the stop distance with no cap on it at all. Because sizing
reads the portfolio value at call time, it also can never scale a position
up to chase a previous loss: a smaller portfolio after a loss produces a
smaller next trade, by construction, not by a special case.
"""
import datetime as dt
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.identity import instrument_key
from app.state import get_state, set_state

# Absolute ceilings - config is clamped to these no matter what .env says.
HARD_MAX_PORTFOLIO_PCT_PER_TRADE = 0.10   # never risk more than 10% of the portfolio on one trade
HARD_MAX_DAILY_LOSS_PCT = 0.25
HARD_MIN_STOP_LOSS_PCT = 0.03
HARD_MAX_CONCURRENT_POSITIONS = 20
HARD_MAX_EXPOSURE_PER_TOKEN_PCT = 0.25
HARD_MAX_TOTAL_EXPOSURE_PCT = 1.0          # can never size past "fully invested"
HARD_MIN_CONSECUTIVE_LOSSES = 2            # a shutdown that fires on 1 loss isn't a strategy filter
HARD_MAX_CONSECUTIVE_LOSSES = 15
HARD_MAX_DAILY_TRADES = 50
HARD_MAX_COOLDOWN_SECONDS = 86_400         # 1 day - a config typo must not lock the bot out forever

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


def _day_bounds(now: dt.datetime | None = None) -> tuple[dt.datetime, dt.datetime]:
    now = now or dt.datetime.now(dt.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + dt.timedelta(days=1)


class RiskManager:
    def __init__(
        self,
        *,
        max_pct_per_trade: float | None = None,
        daily_loss_limit_pct: float | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        max_concurrent_positions: int | None = None,
        max_exposure_per_token_pct: float | None = None,
        max_total_exposure_pct: float | None = None,
        max_consecutive_losses: int | None = None,
        max_daily_trades: int | None = None,
        cooldown_seconds: int | None = None,
        max_trade_size_usd: float | None = None,
    ):
        """Every parameter defaults to the matching `settings.*` value when
        omitted, so `RiskManager()` behaves exactly as before - this is only
        for callers (the backtester) that need deterministic, config-driven
        limits independent of the live environment's `.env`, while running
        the IDENTICAL sizing/gating formulas live trading uses. A backtest
        result is only meaningful if it reflects the same risk logic that
        would actually run the trade, not a re-implementation that could
        silently drift from it.
        """
        # Fraction of the portfolio risked (lost, if the stop is hit) per trade.
        self.max_pct_per_trade = min(
            max_pct_per_trade if max_pct_per_trade is not None else settings.MAX_PORTFOLIO_PCT_PER_TRADE,
            HARD_MAX_PORTFOLIO_PCT_PER_TRADE,
        )
        self.daily_loss_limit_pct = min(
            daily_loss_limit_pct if daily_loss_limit_pct is not None else settings.DAILY_LOSS_LIMIT_PCT,
            HARD_MAX_DAILY_LOSS_PCT,
        )
        self.stop_loss_pct = max(
            stop_loss_pct if stop_loss_pct is not None else settings.STOP_LOSS_PCT, HARD_MIN_STOP_LOSS_PCT
        )
        self.take_profit_pct = take_profit_pct if take_profit_pct is not None else settings.TAKE_PROFIT_PCT
        self.max_concurrent_positions = min(
            max_concurrent_positions if max_concurrent_positions is not None else settings.MAX_CONCURRENT_POSITIONS,
            HARD_MAX_CONCURRENT_POSITIONS,
        )
        self.max_exposure_per_token_pct = min(
            max_exposure_per_token_pct if max_exposure_per_token_pct is not None
            else settings.MAX_EXPOSURE_PER_TOKEN_PCT,
            HARD_MAX_EXPOSURE_PER_TOKEN_PCT,
        )
        self.max_total_exposure_pct = min(
            max_total_exposure_pct if max_total_exposure_pct is not None else settings.MAX_TOTAL_EXPOSURE_PCT,
            HARD_MAX_TOTAL_EXPOSURE_PCT,
        )
        self.max_consecutive_losses = min(
            max(
                max_consecutive_losses if max_consecutive_losses is not None else settings.MAX_CONSECUTIVE_LOSSES,
                HARD_MIN_CONSECUTIVE_LOSSES,
            ),
            HARD_MAX_CONSECUTIVE_LOSSES,
        )
        self.max_daily_trades = min(
            max_daily_trades if max_daily_trades is not None else settings.MAX_DAILY_TRADES,
            HARD_MAX_DAILY_TRADES,
        )
        self.cooldown_seconds = min(
            cooldown_seconds if cooldown_seconds is not None else settings.TRADE_COOLDOWN_SECONDS,
            HARD_MAX_COOLDOWN_SECONDS,
        )
        self.max_trade_size_usd = max_trade_size_usd if max_trade_size_usd is not None else settings.MAX_TRADE_SIZE_USD

    # ---- sizing ----
    def position_size_usd(
        self,
        portfolio_value_usd: float,
        *,
        stop_loss_pct: float | None = None,
        current_total_exposure_usd: float = 0.0,
        current_symbol_exposure_usd: float = 0.0,
    ) -> float:
        """USD notional for a new position, sized off the stop distance.

        risk_amount = portfolio_value * risk_pct_per_trade   (dollars you're
            willing to lose if the stop is hit)
        notional    = risk_amount / stop_loss_pct             (position size
            that makes that true, whatever the stop distance is)

        Then capped by the absolute per-trade ceiling and by whatever
        exposure room is left, both per-token and portfolio-wide. Returns 0
        when there is no room rather than a negative number, so callers can
        treat "size <= 0" as a plain rejection.
        """
        stop_pct = max(stop_loss_pct if stop_loss_pct is not None else self.stop_loss_pct, HARD_MIN_STOP_LOSS_PCT)
        risk_amount = portfolio_value_usd * self.max_pct_per_trade
        notional = risk_amount / stop_pct
        notional = min(notional, self.max_trade_size_usd)

        total_room = max(portfolio_value_usd * self.max_total_exposure_pct - current_total_exposure_usd, 0.0)
        symbol_room = max(portfolio_value_usd * self.max_exposure_per_token_pct - current_symbol_exposure_usd, 0.0)

        return max(min(notional, total_room, symbol_room), 0.0)

    def stop_loss_take_profit(self, entry_price: float) -> tuple[float, float]:
        sl = entry_price * (1 - self.stop_loss_pct)
        tp = entry_price * (1 + self.take_profit_pct)
        return sl, tp

    # ---- gating checks that must pass before a buy is allowed ----
    def check_can_open_position(
        self, db: Session, symbol: str | None = None, token_address: str | None = None
    ) -> RiskDecision:
        if is_trading_halted(db):
            reason = get_state(db, HALT_REASON_KEY, "") or "trading halted"
            return RiskDecision(False, f"trading halted: {reason}")

        open_count = db.query(models.Position).filter_by(status=models.PositionStatus.OPEN.value).count()
        if open_count >= self.max_concurrent_positions:
            return RiskDecision(
                False, f"max concurrent positions reached ({open_count}/{self.max_concurrent_positions})"
            )

        trades_today = self._trades_opened_today(db)
        if trades_today >= self.max_daily_trades:
            return RiskDecision(
                False, f"daily trade limit reached ({trades_today}/{self.max_daily_trades})"
            )

        if symbol is not None and self.cooldown_seconds > 0:
            remaining = self._cooldown_remaining_seconds(db, symbol, token_address)
            if remaining is not None and remaining > 0:
                return RiskDecision(
                    False,
                    f"cooldown active for {symbol}: {remaining:.0f}s remaining "
                    f"(last trade less than {self.cooldown_seconds}s ago)",
                )

        return RiskDecision(True)

    def evaluate_daily_loss(self, db: Session) -> RiskDecision:
        start, end = _day_bounds()
        starting_balance = settings.PORTFOLIO_STARTING_BALANCE_USD
        realized_today = (
            db.query(models.Trade)
            .filter(models.Trade.closed_at >= start, models.Trade.closed_at < end)
            .with_entities(models.Trade.pnl_usd)
            .all()
        )
        total = sum(pnl for (pnl,) in realized_today if pnl is not None)
        loss_limit = starting_balance * self.daily_loss_limit_pct
        if total <= -loss_limit:
            return RiskDecision(
                False, f"daily realized loss ${total:,.2f} breached limit -${loss_limit:,.2f}"
            )
        return RiskDecision(True)

    def evaluate_consecutive_losses(self, db: Session) -> RiskDecision:
        """Halt after N losing trades in a row, regardless of daily P&L.

        A losing streak can stay under the daily-loss dollar limit while
        still being a clear sign the strategy or the market has stopped
        working for now — this catches that independently of the size of
        each loss.
        """
        recent = (
            db.query(models.Trade.pnl_usd)
            .filter(models.Trade.pnl_usd.isnot(None))
            .order_by(models.Trade.closed_at.desc())
            .limit(self.max_consecutive_losses)
            .all()
        )
        if len(recent) < self.max_consecutive_losses:
            return RiskDecision(True)
        if all(pnl is not None and pnl < 0 for (pnl,) in recent):
            return RiskDecision(
                False,
                f"{self.max_consecutive_losses} consecutive losing trades - "
                "strategy or market conditions may have changed",
            )
        return RiskDecision(True)

    # ---- internal queries ----
    def _trades_opened_today(self, db: Session) -> int:
        start, end = _day_bounds()
        return (
            db.query(models.Trade)
            .filter(
                models.Trade.side == "buy",
                models.Trade.status == models.TradeStatus.FILLED.value,
                models.Trade.opened_at >= start,
                models.Trade.opened_at < end,
            )
            .count()
        )

    def _cooldown_remaining_seconds(
        self, db: Session, symbol: str, token_address: str | None = None
    ) -> float | None:
        """Time left on the re-entry cooldown FOR THIS MINT.

        Matched on the canonical identity rather than the symbol: a
        cooldown started by one PEPE must not silence an unrelated mint
        that happens to share the ticker. See app/identity.py.
        """
        key = instrument_key(symbol, token_address)
        recent = (
            db.query(models.Trade.created_at, models.Trade.symbol, models.Trade.token_address)
            .filter(models.Trade.status == models.TradeStatus.FILLED.value)
            .order_by(models.Trade.created_at.desc())
            .limit(200)
            .all()
        )
        last = next(
            (row for row in recent if instrument_key(row[1], row[2]) == key), None
        )
        if last is None:
            return None
        last_at = last[0]
        if last_at.tzinfo is None:
            last_at = last_at.replace(tzinfo=dt.timezone.utc)
        elapsed = (dt.datetime.now(dt.timezone.utc) - last_at).total_seconds()
        return self.cooldown_seconds - elapsed
