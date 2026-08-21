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
from app.notifications.notifier import notifier

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
    except Exception as exc:
        logger.exception("forward-return resolution pass failed")
        db.rollback()
        await notifier.notify_worker_failure("forward-return resolver", exc)
        return {"due": 0, "resolved": 0, "abandoned": 0, "unavailable": 0, "error": True}
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

    while not _stop_event.is_set():
        summary = await resolve_once()
        if summary.get("due"):
            logger.info(
                "forward returns: %d due, %d resolved, %d still unavailable, %d abandoned",
                summary["due"], summary["resolved"],
                summary.get("unavailable", 0), summary.get("abandoned", 0),
            )
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue

    logger.info("forward-return worker stopped")
