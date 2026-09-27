"""Daily P&L summary job, scheduled via APScheduler in app/main.py."""
import datetime as dt
import logging

from app import models
from app.database import SessionLocal
from app.notifications.notifier import notifier
from app.services import portfolio

logger = logging.getLogger(__name__)


async def send_daily_summary() -> None:
    db = SessionLocal()
    try:
        today = dt.datetime.now(dt.timezone.utc).date()
        trades_today = [
            t
            for t in db.query(models.Trade).filter(models.Trade.closed_at.isnot(None)).all()
            if t.closed_at.date() == today
        ]
        realized = sum(t.pnl_usd or 0.0 for t in trades_today)
        portfolio_value = await portfolio.get_portfolio_value_usd(db)

        existing = db.query(models.DailyPnL).filter_by(date=today.isoformat()).first()
        if existing:
            existing.realized_pnl_usd = realized
            existing.portfolio_value_usd = portfolio_value
            existing.trades_count = len(trades_today)
        else:
            db.add(
                models.DailyPnL(
                    date=today.isoformat(),
                    realized_pnl_usd=realized,
                    portfolio_value_usd=portfolio_value,
                    trades_count=len(trades_today),
                )
            )
        db.commit()

        await notifier.notify_daily_summary(
            {
                "trades_count": len(trades_today),
                "realized_pnl_usd": realized,
                "portfolio_value_usd": portfolio_value,
            }
        )
    except Exception:
        logger.exception("failed to generate daily summary")
        db.rollback()
    finally:
        db.close()
