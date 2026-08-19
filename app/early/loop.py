"""The watchlist re-evaluation loop.

WATCH is only useful if something keeps looking. This loop re-scores every
watched token on WATCHLIST_INTERVAL_SECONDS and drives the state machine:
entries that confirm are handed to the normal buy path, entries that
deteriorate are failed with a category, and entries that sit too long
expire.

WHY IT RUNS FASTER THAN THE SCANNER

The scanner sweeps for new candidates on a slower cadence because
discovery is expensive and the universe changes slowly. The watchlist runs
faster because it is racing something specific: the window between "this
looks like it is starting" and "this has already moved". A two-minute
re-evaluation on a token that goes 40% in ten minutes is the difference
between an entry and a chase.

WHAT IT COSTS

One market snapshot per watched token per pass, plus one candle fetch for
any token close enough to confirming that the technical score matters.
WATCHLIST_MAX_SIZE bounds that; without it a busy market would put the bot
into a rate limit and take the price feed down with it, which the kill
switch would then correctly read as a reason to stop trading entirely.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging

from sqlalchemy.orm import Session

from app import models, pipeline
from app.config import settings
from app.database import SessionLocal
from app.early import watchlist as wl
from app.early.engine import Decision, evaluate
from app.services import price_feed

logger = logging.getLogger(__name__)

_stop_event = asyncio.Event()


def stop() -> None:
    _stop_event.set()


def blocked_reason() -> str | None:
    if not settings.EARLY_SIGNAL_ENABLED:
        return "EARLY_SIGNAL_ENABLED=false"
    return None


async def _score_one(db: Session, entry: models.WatchlistEntry) -> None:
    """Re-evaluate one watched token."""
    market = await price_feed.get_market_snapshot(entry.token_address)
    if market is None:
        # No snapshot is not a low score. Leave the entry alone and try
        # again next pass rather than failing it on a missing fetch.
        logger.debug("no snapshot for watched token %s - skipping this pass", entry.symbol)
        return

    wl.store_observation(db, entry.symbol, entry.token_address, market)
    observations = wl.recent_observations(db, entry.token_address)

    series = None
    technical = None
    # The candle fetch and technical score are only worth their cost once a
    # token is close to confirming. Fetching them for every watched token
    # every two minutes would multiply the bot's API load by the size of
    # the watchlist for information that changes nothing while the early
    # score is still well below the confirm threshold.
    close_to_confirming = (entry.early_score or 0) >= settings.EARLY_SIGNAL_WATCH_THRESHOLD
    if close_to_confirming:
        try:
            from app.data.candles import Timeframe
            from app.data.live_provider import fetch_candles
            from app.signals.live_gate import evaluate_live_entry_signal

            series = await fetch_candles(
                entry.chain, entry.token_address, entry.symbol, Timeframe.M5, 300
            )
            score = await evaluate_live_entry_signal(entry.chain, entry.token_address, entry.symbol)
            technical = score.score if score else None
        except Exception:
            logger.warning("technical re-score failed for %s", entry.symbol, exc_info=True)

    verdict = evaluate(
        series=series,
        market=market,
        observations=observations,
        security_passed=True,   # re-checked properly on the buy path below
        technical_score=technical,
    )

    wl.record(
        db,
        token_address=entry.token_address,
        symbol=entry.symbol,
        chain=entry.chain,
        verdict=verdict,
        price=market.price_usd,
    )

    pipeline.record(
        db, stage=pipeline.TECHNICAL_SCORE if technical is not None else pipeline.HISTORY,
        symbol=entry.symbol, token_address=entry.token_address, chain=entry.chain,
        passed=verdict.decision is not Decision.SKIP,
        reason=f"watchlist re-evaluation: {verdict.reason}",
        score=verdict.early_score,
        detail={"early": True, "stage": verdict.stage.value if verdict.stage else None,
                "reliable": verdict.early.reliable if verdict.early else False},
    )

    if verdict.decision is Decision.PAPER_BUY:
        await _hand_to_buy_path(db, entry, market)


async def _hand_to_buy_path(db: Session, entry: models.WatchlistEntry, market) -> None:
    """A confirmed watch goes through the NORMAL buy path.

    Not a parallel trading path. Every protection the bot has - the kill
    switch, the risk gate, the rug check, market quality, exposure caps,
    the realistic fill model - lives in trading_service, and a second entry
    route would eventually drift from it and lose one of them silently.
    The early engine's job ends at "this candidate is worth evaluating".
    """
    from app.services.trading_service import handle_discovered_token

    logger.info("watchlist CONFIRMED %s - handing to the standard buy path", entry.symbol)
    try:
        signal = await handle_discovered_token(
            db,
            symbol=entry.symbol,
            token_address=entry.token_address,
            chain=entry.chain,
            price=market.price_usd or 0.0,
            discovery_source="early_signal",
            extra={
                "early_score": entry.early_score,
                "late_entry_risk": entry.late_entry_risk,
                "momentum_class": entry.momentum_class,
                "stage": entry.stage,
            },
        )
        db.flush()
        traded = (
            db.query(models.Trade)
            .filter_by(signal_id=signal.id, side="buy", status=models.TradeStatus.FILLED.value)
            .first()
            is not None
        )
        if traded:
            wl.mark_traded(db, entry.token_address)
    except Exception:
        logger.exception("handing %s to the buy path failed", entry.symbol)


async def evaluate_once(db: Session | None = None) -> dict:
    """One full watchlist pass. Returns a summary for logging and tests."""
    blocked = blocked_reason()
    if blocked:
        return {"skipped": blocked, "evaluated": 0}

    owns_session = db is None
    db = db or SessionLocal()
    summary = {"evaluated": 0, "confirmed": 0, "failed": 0, "expired": 0, "pruned": 0}

    try:
        summary["expired"] = wl.expire_stale(db)

        for entry in wl.active(db):
            try:
                await _score_one(db, entry)
                summary["evaluated"] += 1
                if entry.state == wl.CONFIRMED:
                    summary["confirmed"] += 1
                elif entry.state == wl.FAILED:
                    summary["failed"] += 1
            except Exception:
                # One bad token must never take down the pass.
                logger.exception("watchlist evaluation failed for %s", entry.symbol)

        summary["pruned"] = wl.prune_observations(db)
        db.commit()
    except Exception:
        logger.exception("watchlist pass failed")
        db.rollback()
    finally:
        if owns_session:
            db.close()

    return summary


async def run_forever() -> None:
    blocked = blocked_reason()
    if blocked:
        logger.info("early signal engine disabled: %s", blocked)
        return

    interval = settings.WATCHLIST_INTERVAL_SECONDS
    logger.info(
        "early signal watchlist started (every %ss, max %d tokens, may_trade=%s)",
        interval, settings.WATCHLIST_MAX_SIZE, settings.EARLY_SIGNAL_MAY_TRADE,
    )
    if not settings.EARLY_SIGNAL_MAY_TRADE:
        logger.info(
            "EARLY_SIGNAL_MAY_TRADE=false - the early engine will record WATCH candidates and "
            "will NOT open positions. The early weights are unvalidated priors until "
            "/research shows the score separates outcomes."
        )
    _stop_event.clear()

    while not _stop_event.is_set():
        summary = await evaluate_once()
        if summary.get("evaluated"):
            logger.info(
                "watchlist: %d evaluated, %d confirmed, %d failed, %d expired",
                summary["evaluated"], summary["confirmed"], summary["failed"], summary["expired"],
            )
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue

    logger.info("early signal watchlist stopped")
