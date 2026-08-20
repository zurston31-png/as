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
from app import backup
from app.autopilot import loop as autopilot_loop
from app.database import SessionLocal, init_db
from app.early import loop as early_loop
from app.monitor import forward_return_worker, position_monitor, shadow_resolver_worker
from app.scanner import loop as scanner_loop
from app.schemas import TradingViewAlert
from app.security import verify_webhook_secret
from app.services import api_health
from app.startup_checks import log_config_coherence
from app.strategy.version import register_current_version
from app.services.reporting import send_daily_summary
from app.services.trading_service import handle_alert

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()
_monitor_task: asyncio.Task | None = None
_scanner_task: asyncio.Task | None = None
_forward_task: asyncio.Task | None = None
_shadow_task: asyncio.Task | None = None
_early_task: asyncio.Task | None = None
_backup_task: asyncio.Task | None = None
_autopilot_task: asyncio.Task | None = None


async def _snapshot_forever() -> None:
    """Take a verified snapshot every BACKUP_INTERVAL_MINUTES.

    Sleeps first. Startup has just restored or opened the database and a
    snapshot taken one second later would only duplicate the shutdown
    snapshot from the previous run, spending a rotation slot on it.
    """
    interval = max(settings.BACKUP_INTERVAL_MINUTES, 1) * 60
    while True:
        try:
            await asyncio.sleep(interval)
            # In a worker thread: the page copy plus integrity_check is
            # seconds of blocking IO on a database of any size, and on the
            # event loop it freezes the position monitor - stop-loss and
            # take-profit checks - along with every HTTP request.
            await asyncio.to_thread(backup.take_snapshot, reason="scheduled")
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never let a failed backup stop the next one from being tried.
            logger.exception("scheduled snapshot failed - will retry next interval")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _monitor_task, _scanner_task, _forward_task, _early_task, _backup_task
    global _shadow_task
    global _autopilot_task

    # BEFORE init_db(). A wiped disk leaves no database at all, and
    # init_db() would create an empty one - after which the restore has
    # nothing obviously-empty to recognise and a schema it must not
    # overwrite. Restoring first means the migration then runs over the
    # recovered data, which is exactly the right order.
    if backup.restore_if_empty():
        logger.warning("recovered the database from a snapshot after an empty start")

    init_db()

    pointless = backup.warn_if_backups_are_pointless()
    if pointless:
        logger.warning("BACKUPS MAY NOT SURVIVE A RESET: %s", pointless)

    mode = "LIVE" if settings.LIVE_TRADING else "PAPER"
    logger.info(
        "starting up | mode=%s chain=%s execution_backend=%s watchlist=%s",
        mode, settings.CHAIN, settings.EXECUTION_BACKEND, settings.SYMBOLS_WATCHLIST,
    )
    if settings.LIVE_TRADING:
        logger.warning("LIVE_TRADING=true — the bot WILL submit real on-chain/exchange orders.")

    # Catch config combinations that make trading impossible (e.g. a score
    # threshold above what the engine produces, or a minimum token age below
    # the history the score needs). Silent at runtime otherwise - the bot
    # just never trades and the logs look fine.
    log_config_coherence()

    # Restore upstream health and register the running strategy version.
    # Without the reload, a restart makes every data source look "unused"
    # and the dashboard loses the fact that one has been down for an hour.
    db = SessionLocal()
    try:
        loaded = api_health.load(db)
        if loaded:
            logger.info("restored health records for %d upstream service(s)", loaded)
        version = register_current_version(db)
        db.commit()
        logger.info("running strategy version %s", version.label)
    except Exception:
        logger.exception("startup bookkeeping failed - continuing without it")
        db.rollback()
    finally:
        db.close()

    _monitor_task = asyncio.create_task(position_monitor.run_forever())
    _forward_task = asyncio.create_task(forward_return_worker.run_forever())
    _early_task = asyncio.create_task(early_loop.run_forever())
    _shadow_task = asyncio.create_task(shadow_resolver_worker.run_forever())
    if settings.BACKUP_ENABLED:
        _backup_task = asyncio.create_task(_snapshot_forever())

    autopilot_blocked = autopilot_loop.blocked_reason()
    if autopilot_blocked:
        logger.info("autopilot disabled: %s", autopilot_blocked)
    else:
        _autopilot_task = asyncio.create_task(autopilot_loop.run_forever())

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
    forward_return_worker.stop()
    shadow_resolver_worker.stop()
    early_loop.stop()
    scheduler.shutdown(wait=False)
    if _monitor_task:
        _monitor_task.cancel()
    if _scanner_task:
        _scanner_task.cancel()
    if _forward_task:
        _forward_task.cancel()
    if _shadow_task:
        _shadow_task.cancel()
    if _early_task:
        _early_task.cancel()
    if _backup_task:
        _backup_task.cancel()
    if _autopilot_task:
        _autopilot_task.cancel()

    # cancel() only schedules the cancellation; the loops are still between
    # await points until control returns here. Wait for them before the
    # shutdown snapshot, otherwise it races the writes it exists to capture.
    pending = [t for t in (_monitor_task, _scanner_task, _forward_task,
                           _early_task, _backup_task, _autopilot_task,
                           _shadow_task) if t is not None]
    if pending:
        await asyncio.wait(pending, timeout=10)

    # Snapshot on the way out, now that the writers really are quiet. On a
    # redeploy this is the last chance to capture everything since the
    # previous scheduled snapshot.
    if settings.BACKUP_ENABLED:
        try:
            await asyncio.to_thread(backup.take_snapshot, reason="shutdown")
        except Exception:
            logger.exception("shutdown snapshot failed")


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
