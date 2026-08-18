"""Portfolio ledger.

Cash balance is tracked as an explicit ledger in `bot_state` rather than by
polling a wallet/exchange balance. In paper mode this is exactly the
simulated account. In live mode it is only as accurate as
PORTFOLIO_STARTING_BALANCE_USD matches what you actually funded the
trading wallet/exchange account with — the bot does not (yet) reconcile
against an on-chain/exchange balance automatically. Reconcile periodically
if you want tighter accuracy.
"""
import logging

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.services import price_feed
from app.state import get_state, set_state

logger = logging.getLogger(__name__)

CASH_KEY = "cash_balance_usd"


def get_cash_balance_usd(db: Session) -> float:
    return get_state(db, CASH_KEY, settings.PORTFOLIO_STARTING_BALANCE_USD)


def adjust_cash_balance(db: Session, delta_usd: float) -> float:
    balance = get_cash_balance_usd(db) + delta_usd
    set_state(db, CASH_KEY, balance)
    return balance


async def get_open_positions_value_usd(db: Session) -> float:
    total = 0.0
    positions = db.query(models.Position).filter_by(status=models.PositionStatus.OPEN.value).all()
    for pos in positions:
        price = None
        if pos.token_address:
            try:
                price = await price_feed.get_price_usd(pos.token_address)
            except Exception:
                logger.warning("price lookup failed for %s during valuation", pos.symbol, exc_info=True)
        price = price or pos.entry_price
        total += pos.qty * price
    return total


async def get_portfolio_value_usd(db: Session) -> float:
    cash = get_cash_balance_usd(db)
    positions_value = await get_open_positions_value_usd(db)
    return cash + positions_value
