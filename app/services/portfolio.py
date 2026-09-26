"""Portfolio ledger.

Cash balance is tracked as an explicit ledger in `bot_state` rather than by
polling a wallet/exchange balance. In paper mode this is exactly the
simulated account. In live mode it is only as accurate as
PORTFOLIO_STARTING_BALANCE_USD matches what you actually funded the
trading wallet/exchange account with — the bot does not (yet) reconcile
against an on-chain/exchange balance automatically. Reconcile periodically
if you want tighter accuracy.
"""
import datetime as dt
import json
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.identity import instrument_key
from app.services import price_feed
from app.state import get_state, set_state

logger = logging.getLogger(__name__)

CASH_KEY = "cash_balance_usd"

# What the ledger was actually seeded (or last reset) to, and from when.
#
# The reconciliation check reconstructs the cash balance as
# `starting + sells - buys`, and until this existed the `starting` term
# came from PORTFOLIO_STARTING_BALANCE_USD - the CURRENT value of a
# setting, used to explain a ledger that was seeded from whatever the
# value happened to be when the database was created. Those are the same
# number only until someone edits the setting, at which point the check
# reports a discrepancy the size of the edit, the kill switch fails
# closed on it, and the bot stops opening positions for a reason that has
# nothing to do with its books being wrong.
#
# Recording the baseline makes the check answer the question it is
# actually asking: does the ledger match the trades that have moved it
# since it was set?
BASELINE_KEY = "cash_ledger_baseline"


@dataclass(frozen=True)
class LedgerBaseline:
    """The cash the ledger started from, and the point it started from.

    `since` is None for a ledger that has never been reset - the baseline
    covers the whole trade record, which is what every existing database
    means.
    """

    balance_usd: float
    since: dt.datetime | None


def get_ledger_baseline(db: Session) -> LedgerBaseline:
    """The recorded baseline, or the configured starting balance.

    The fallback is what databases created before this existed need, and
    it reproduces the old behaviour exactly: baseline = the setting,
    covering every trade.
    """
    raw = get_state(db, BASELINE_KEY, None)
    if not isinstance(raw, dict) or "balance_usd" not in raw:
        return LedgerBaseline(settings.PORTFOLIO_STARTING_BALANCE_USD, None)

    since = None
    if raw.get("since"):
        try:
            since = dt.datetime.fromisoformat(raw["since"])
        except (TypeError, ValueError):
            logger.error(
                "ledger baseline has an unreadable 'since' (%r) - treating the "
                "baseline as covering the whole record", raw.get("since"),
            )
    try:
        balance = float(raw["balance_usd"])
    except (TypeError, ValueError):
        logger.error(
            "ledger baseline has an unreadable balance (%r) - falling back to "
            "the configured starting balance", raw.get("balance_usd"),
        )
        return LedgerBaseline(settings.PORTFOLIO_STARTING_BALANCE_USD, None)

    return LedgerBaseline(balance, since)


def set_ledger_baseline(
    db: Session, balance_usd: float, *, since: dt.datetime | None = None
) -> None:
    """Record what the ledger was set to, and from when."""
    set_state(
        db,
        BASELINE_KEY,
        {"balance_usd": float(balance_usd), "since": since.isoformat() if since else None},
    )


def get_cash_balance_usd(db: Session) -> float:
    return get_state(db, CASH_KEY, settings.PORTFOLIO_STARTING_BALANCE_USD)


def adjust_cash_balance(db: Session, delta_usd: float) -> float:
    """Move the cash ledger by `delta_usd`, atomically within this session.

    Read-modify-write on a shared row is the classic way to lose money in
    an accounting ledger: two sessions both read $1,000, both subtract
    $100, both write $900, and $100 of spending vanishes from the books.

    Two things prevent that here. The buy path is serialised per token by
    the entry reservation (app/concurrency.py), and the row is locked for
    the duration of the read and the write so the window between them
    cannot be interleaved by another session on the same database.

    SQLite ignores FOR UPDATE (its writes are serialised by a database-level
    lock anyway, which gives the same guarantee), so this is a no-op there
    and a genuine row lock on Postgres.
    """
    row = (
        db.query(models.BotState)
        .filter(models.BotState.key == CASH_KEY)
        .with_for_update()
        .first()
    )
    current = settings.PORTFOLIO_STARTING_BALANCE_USD
    if row is not None:
        try:
            current = float(json.loads(row.value))
        except (TypeError, ValueError):
            logger.error(
                "cash ledger value %r is unreadable - refusing to guess. "
                "Falling back to the configured starting balance.", row.value,
            )

    balance = current + delta_usd
    set_state(db, CASH_KEY, balance)
    return balance


async def _position_value_usd(pos: models.Position) -> float:
    """Mark one position to market, falling back to cost when the price is
    unavailable.

    The fallback is the honest-but-dangerous option and is deliberately
    reported: a rugged token whose feed has gone quiet still values at what
    was paid for it, which INFLATES portfolio value and therefore the size
    of the next trade. `value_open_positions` returns the count of positions
    priced this way so callers can refuse to size against a stale book
    rather than silently trusting it.
    """
    if pos.token_address:
        try:
            price = await price_feed.get_price_usd(pos.token_address)
        except Exception:
            logger.warning("price lookup failed for %s during valuation", pos.symbol, exc_info=True)
            price = None
        if price and price > 0:
            return pos.qty * price
    return pos.qty * pos.entry_price


@dataclass
class Valuation:
    """Marked-to-market book, plus how much of it could not be priced."""

    total_usd: float
    positions: int
    stale_positions: int          # valued at cost because no live price came back
    stale_usd: float

    @property
    def fully_priced(self) -> bool:
        return self.stale_positions == 0

    @property
    def stale_share(self) -> float:
        return (self.stale_usd / self.total_usd) if self.total_usd > 0 else 0.0


async def value_open_positions(db: Session) -> Valuation:
    """Value the open book AND report what could not be priced.

    Separated from the plain total because "the book is worth $840" and
    "the book is worth $840, but $600 of that is a token whose price feed
    stopped responding an hour ago" are very different facts, and only the
    second one should stop the bot from sizing a new trade off it.
    """
    positions = db.query(models.Position).filter_by(status=models.PositionStatus.OPEN.value).all()
    total = 0.0
    stale_count = 0
    stale_usd = 0.0

    for pos in positions:
        live = None
        if pos.token_address:
            try:
                live = await price_feed.get_price_usd(pos.token_address)
            except Exception:
                logger.warning("price lookup failed for %s during valuation", pos.symbol, exc_info=True)
        if live and live > 0:
            total += pos.qty * live
        else:
            value = pos.qty * pos.entry_price
            total += value
            stale_count += 1
            stale_usd += value

    return Valuation(
        total_usd=total, positions=len(positions),
        stale_positions=stale_count, stale_usd=stale_usd,
    )


async def get_open_positions_value_usd(db: Session) -> float:
    return (await value_open_positions(db)).total_usd


async def get_portfolio_value_usd(db: Session) -> float:
    cash = get_cash_balance_usd(db)
    positions_value = await get_open_positions_value_usd(db)
    return cash + positions_value


async def get_token_exposure_usd(db: Session, symbol: str, token_address: str | None) -> float:
    """Current open notional in ONE TOKEN - feeds the per-token exposure cap.

    Keyed on the mint, not the symbol. Summing every position whose symbol
    happens to read PEPE would let two unrelated mints share (and therefore
    double) a cap that exists to bound single-asset risk. See app/identity.py.
    """
    key = instrument_key(symbol, token_address)
    total = 0.0
    positions = db.query(models.Position).filter_by(status=models.PositionStatus.OPEN.value).all()
    for pos in positions:
        if instrument_key(pos.symbol, pos.token_address) != key:
            continue
        total += await _position_value_usd(pos)
    return total
