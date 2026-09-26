"""Background worker that fills in forward returns as horizons come due.

Separate from the position monitor on purpose. The position monitor guards
money and must never be delayed by research bookkeeping; this loop is pure
data collection and is allowed to fall behind, fail, and catch up later.

Runs on its own interval and resolves at most FORWARD_RETURN_BATCH_LIMIT
rows per pass, so a long backlog drains steadily instead of producing one
enormous burst of API calls that trips a rate limit and starves the parts
of the bot that actually trade.
"""
from __future__ import annotations

import asyncio
import logging

from app.analysis import forward_returns
from app.config import settings
from app.database import SessionLocal
from app.monitor.supervisor import run_supervised

logger = logging.getLogger(__name__)

_stop_event = asyncio.Event()


def stop() -> None:
    _stop_event.set()


async def resolve_once() -> dict:
    """One resolution pass. Returns a summary dict for logging and tests."""
    db = SessionLocal()
    try:
        summary = await forward_returns.resolve_due(
            db, limit=settings.FORWARD_RETURN_BATCH_LIMIT
        )
        db.commit()
        return summary
    except Exception:
        db.rollback()
        # Re-raised rather than swallowed: the supervisor owns failure
        # accounting, the throttled notification and the backoff, and a
        # pass that reports its own error looks like a success to it.
        raise
    finally:
        db.close()


async def run_forever() -> None:
    if not forward_returns.enabled():
        logger.info(
            "forward-return collection disabled (FORWARD_RETURNS_ENABLED=false) - "
            "score calibration will have no data to work from"
        )
        return

    interval = settings.FORWARD_RETURN_RESOLVE_INTERVAL_SECONDS
    logger.info("forward-return worker started (every %ss)", interval)
    _stop_event.clear()

    def report(summary) -> None:
        if summary and summary.get("due"):
            logger.info(
                "forward returns: %d due, %d resolved, %d still unavailable, %d abandoned",
                summary["due"], summary["resolved"],
                summary.get("unavailable", 0), summary.get("abandoned", 0),
            )

    await run_supervised(
        "forward-return resolver", resolve_once,
        interval_seconds=interval, stop_event=_stop_event, on_result=report,
    )
    logger.info("forward-return worker stopped")
