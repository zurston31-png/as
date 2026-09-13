"""Background loop that turns hypothetical entries into recorded outcomes.

Kept apart from the position monitor deliberately. That loop guards the
paper account's money and must never be slowed down by research
bookkeeping; this one is pure data collection and is allowed to fall
behind, fail and catch up later without anything depending on it.

It resolves at most SHADOW_RESOLVE_BATCH positions per pass, so a backlog
drains steadily rather than firing one enormous burst of candle requests
that trips a rate limit and starves the parts of the bot that trade.
"""
from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.database import SessionLocal
from app.monitor.supervisor import run_supervised
from app.shadow import resolver

logger = logging.getLogger(__name__)

_stop_event = asyncio.Event()


def stop() -> None:
    _stop_event.set()


async def resolve_once() -> dict:
    """One pass, in its own session. Returns a summary for logs and tests."""
    db = SessionLocal()
    try:
        summary = await resolver.resolve_once(db)
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
    if not settings.SHADOW_ENABLED or not settings.SHADOW_RESOLVER_ENABLED:
        logger.info(
            "shadow resolver disabled - hypothetical positions will stay open and the "
            "paired comparison will have no outcomes to compare"
        )
        return

    interval = settings.SHADOW_RESOLVE_INTERVAL_SECONDS
    logger.info("shadow resolver started (every %ss)", interval)
    _stop_event.clear()

    def report(summary) -> None:
        if summary and summary.get("considered"):
            logger.info(
                "shadow: %d considered, %d closed, %d still open, %d abandoned, "
                "%d horizons recorded",
                summary["considered"], summary.get("closed", 0),
                summary.get("still_open", 0), summary.get("abandoned", 0),
                summary.get("horizons_recorded", 0),
            )

    await run_supervised(
        "shadow resolver", resolve_once,
        interval_seconds=interval, stop_event=_stop_event, on_result=report,
    )
    logger.info("shadow resolver stopped")
