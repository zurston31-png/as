"""FastAPI application entrypoint.

Wires together: the TradingView webhook, the dashboard, the background
position monitor (stop-loss/take-profit/dev-wallet exits), and the daily
P&L summary scheduler.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.dashboard.routes import router as dashboard_router
from app.database import SessionLocal, init_db
from app.monitor import position_monitor
from app.scanner import loop as scanner_loop
from app.schemas import TradingViewAlert
from app.security import verify_webhook_secret
from app.services.reporting import send_daily_summary
from app.services.trading_service import handle_alert

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()
_monitor_task: asyncio.Task | None = None
_scanner_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _monitor_task, _scanner_task
    init_db()

    mode = "LIVE" if settings.LIVE_TRADING else "PAPER"
    logger.info(
        "starting up | mode=%s chain=%s execution_backend=%s watchlist=%s",
        mode, settings.CHAIN, settings.EXECUTION_BACKEND, settings.SYMBOLS_WATCHLIST,
    )
    if settings.LIVE_TRADING:
        logger.warning("LIVE_TRADING=true — the bot WILL submit real on-chain/exchange orders.")

    _monitor_task = asyncio.create_task(position_monitor.run_forever())

    scanner_blocked = scanner_loop.scanner_blocked_reason()
    if scanner_blocked:
        logger.info("automatic token scanner disabled: %s", scanner_blocked)
    else:
        logger.info("automatic token scanner enabled (every %ss)", settings.SCANNER_INTERVAL_SECONDS)
        _scanner_task = asyncio.create_task(scanner_loop.run_forever())

    scheduler.add_job(
        send_daily_summary,
        CronTrigger(hour=settings.DAILY_SUMMARY_HOUR_UTC, minute=0),
        id="daily_summary",
        replace_existing=True,
    )
    scheduler.start()

    yield

    position_monitor.stop()
    scanner_loop.stop()
    scheduler.shutdown(wait=False)
    if _monitor_task:
        _monitor_task.cancel()
    if _scanner_task:
        _scanner_task.cancel()


app = FastAPI(title="Memecoin Trading Bot", lifespan=lifespan)
app.include_router(dashboard_router)


@app.get("/health")
async def health():
    return {"status": "ok", "live_trading": settings.LIVE_TRADING}


@app.post(settings.WEBHOOK_PATH)
async def tradingview_webhook(alert: TradingViewAlert):
    if not verify_webhook_secret(alert.secret):
        raise HTTPException(status_code=401, detail="invalid webhook secret")

    db = SessionLocal()
    try:
        signal = await handle_alert(db, alert)
        db.commit()
        return JSONResponse({"status": "accepted", "signal_id": signal.id})
    except Exception:
        db.rollback()
        logger.exception("failed to process webhook alert for %s", alert.symbol)
        raise HTTPException(status_code=500, detail="internal error processing alert")
    finally:
        db.close()
