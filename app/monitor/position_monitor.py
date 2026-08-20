"""Background loop that polls every open position for:
  - stop-loss / take-profit triggers
  - dev-wallet-sell risk triggers

Runs as an asyncio task started from app/main.py's startup hook, independent
of the webhook — this is what makes exits happen even if no TradingView
sell alert ever fires.
"""
import asyncio
import logging

from app import models
from app.config import settings
from app.database import SessionLocal
from app.exits.manager import ExitManager, evaluate_liquidity, record_liquidity_tick
from app.monitor.devwallet import check_dev_wallet_exit
from app.early import watchlist
from app.services import price_feed
from app.services.trading_service import close_position, partial_close_position

logger = logging.getLogger(__name__)

_stop_event = asyncio.Event()
exit_manager = ExitManager()


async def _evaluate_position(db, pos: models.Position) -> None:
    if not pos.token_address:
        return

    # One snapshot serves three purposes - price, the stored observation the
    # correlation model reads, and pool depth for the liquidity exit - so it
    # replaces the bare price lookup rather than adding a second request.
    market = await price_feed.get_market_snapshot(pos.token_address)
    price = market.price_usd if market else await price_feed.get_price_usd(pos.token_address)

    if market is not None:
        record_liquidity_tick(pos, market.liquidity_usd)
        action = evaluate_liquidity(pos, market.liquidity_usd)
        if action.kind == "full":
            await close_position(db, pos, reason=action.reason)
            return
        if action.kind == "partial" and not pos.partial_exit_taken:
            await partial_close_position(db, pos, action.fraction, reason=action.reason)
            if pos.status != models.PositionStatus.OPEN.value:
                return

    if price is not None:
        # Keep the reading. This loop is the only place the bot observes a
        # token it HOLDS - the early engine watches candidates, and a
        # candidate stops being watched the moment it becomes a position.
        # Correlation risk (app/risk/correlation.py) has no other source of
        # return history for the open book.
        watchlist.store_price_point(db, pos.symbol, pos.token_address, price)

        action = exit_manager.evaluate(pos, price)
        if action.kind == "full":
            await close_position(db, pos, reason=action.reason)
            return
        if action.kind == "partial":
            await partial_close_position(db, pos, action.fraction, reason=action.reason)
            if pos.status != models.PositionStatus.OPEN.value:
                return

    exit_reason = await check_dev_wallet_exit(pos)
    if exit_reason:
        await close_position(db, pos, reason=exit_reason)


async def _check_positions_once() -> None:
    db = SessionLocal()
    try:
        positions = db.query(models.Position).filter_by(status=models.PositionStatus.OPEN.value).all()
        for pos in positions:
            try:
                await _evaluate_position(db, pos)
            except Exception:
                logger.exception("error evaluating position id=%s (%s)", pos.id, pos.symbol)
        db.commit()
    except Exception:
        logger.exception("position monitor tick failed")
        db.rollback()
    finally:
        db.close()


async def run_forever() -> None:
    logger.info("position monitor loop starting (interval=%ss)", settings.PRICE_POLL_INTERVAL_SECONDS)
    while not _stop_event.is_set():
        await _check_positions_once()
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=settings.PRICE_POLL_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


def stop() -> None:
    _stop_event.set()
