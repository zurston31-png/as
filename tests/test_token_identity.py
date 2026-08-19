"""Tests for app/identity.py and the paths that depend on it.

The bug these lock down: dedup, the per-token exposure cap and the
re-entry cooldown all used to key on `symbol`. Symbols on Solana are not
unique and copycat mints share them deliberately, so two unrelated tokens
called PEPE were treated as one - silently, with no error anywhere.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.identity import describe, instrument_key, is_weak
from app.risk.manager import RiskManager
from app.services import portfolio

# Two different mints, same ticker. This is the whole point.
MINT_A = "AaaaPepeMint1111111111111111111111111111111"
MINT_B = "BbbbPepeMint2222222222222222222222222222222"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _clean_positions():
    from app.database import SessionLocal

    def wipe():
        db = SessionLocal()
        try:
            for row in db.query(models.Position).filter(
                models.Position.token_address.in_([MINT_A, MINT_B])
            ).all():
                db.delete(row)
            for row in db.query(models.Trade).filter(
                models.Trade.token_address.in_([MINT_A, MINT_B])
            ).all():
                db.delete(row)
            db.commit()
        finally:
            db.close()

    wipe()
    yield
    wipe()


def _position(mint, qty=1000.0, price=0.01) -> models.Position:
    return models.Position(
        symbol="PEPE", token_address=mint, chain="solana",
        qty=qty, initial_qty=qty, entry_price=price,
        stop_loss=price * 0.85, take_profit=price * 1.3,
        status=models.PositionStatus.OPEN.value, mode="paper",
        opened_at=dt.datetime.now(dt.timezone.utc),
    )


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

def test_the_mint_is_the_identity_not_the_symbol():
    assert instrument_key("PEPE", MINT_A) == MINT_A
    assert instrument_key("PEPE", MINT_A) != instrument_key("PEPE", MINT_B)


def test_the_same_mint_under_a_renamed_symbol_is_still_the_same_token():
    """A token's on-chain metadata can be changed. The mint cannot."""
    assert instrument_key("PEPE", MINT_A) == instrument_key("PEPE2.0", MINT_A)


def test_a_cex_ticker_is_canonical_in_its_own_namespace(monkeypatch):
    """There is no mint on an exchange, so the ticker genuinely is the
    identifier there - a different namespace, not a fallback."""
    monkeypatch.setattr(settings, "EXECUTION_BACKEND", "cex")
    assert instrument_key("BTCUSDT", None) == "BTCUSDT"
    assert is_weak(None) is False


def test_a_missing_mint_on_chain_is_flagged_as_weak(monkeypatch):
    """Falling back to the symbol keeps the bot running, but the caller has
    to be able to see that identity is no longer trustworthy."""
    monkeypatch.setattr(settings, "EXECUTION_BACKEND", "paper")
    assert instrument_key("PEPE", None) == "PEPE"
    assert is_weak(None) is True
    assert is_weak(MINT_A) is False


def test_describe_distinguishes_two_same_symbol_tokens():
    """'already holding PEPE' is exactly the log line that hid this bug."""
    a, b = describe("PEPE", MINT_A), describe("PEPE", MINT_B)
    assert a != b
    assert "PEPE" in a and "no mint" not in a
    assert "no mint address" in describe("PEPE", None)


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.anyio


async def test_exposure_does_not_pool_two_mints_that_share_a_symbol(db_session, monkeypatch):
    """The cap exists to bound single-asset risk. Summing two unrelated
    assets into it would let 'max 10% per token' permit 20% in one."""
    async def no_live_price(_addr):
        return None
    monkeypatch.setattr(portfolio.price_feed, "get_price_usd", no_live_price)

    db_session.add(_position(MINT_A, qty=1000.0, price=0.01))   # $10
    db_session.add(_position(MINT_B, qty=5000.0, price=0.01))   # $50
    db_session.commit()

    assert await portfolio.get_token_exposure_usd(db_session, "PEPE", MINT_A) == pytest.approx(10.0)
    assert await portfolio.get_token_exposure_usd(db_session, "PEPE", MINT_B) == pytest.approx(50.0)
    # ...while the whole-book total naturally still counts both.
    assert await portfolio.get_open_positions_value_usd(db_session) >= 60.0


# ---------------------------------------------------------------------------
# cooldown
# ---------------------------------------------------------------------------

def test_a_cooldown_on_one_mint_does_not_silence_another(db_session):
    rm = RiskManager(cooldown_seconds=900)
    db_session.add(models.Trade(
        symbol="PEPE", token_address=MINT_A, side="buy",
        status=models.TradeStatus.FILLED.value, size_usd=50.0,
        created_at=dt.datetime.now(dt.timezone.utc),
    ))
    db_session.commit()

    blocked = rm.check_can_open_position(db_session, symbol="PEPE", token_address=MINT_A)
    assert not blocked.allowed and "cooldown" in blocked.reason

    allowed = rm.check_can_open_position(db_session, symbol="PEPE", token_address=MINT_B)
    assert allowed.allowed, "an unrelated mint must not inherit another token's cooldown"


# ---------------------------------------------------------------------------
# stale valuation
# ---------------------------------------------------------------------------

async def test_a_position_with_no_live_price_is_reported_as_stale(db_session, monkeypatch):
    """Valuing a dead token at cost inflates portfolio value and therefore
    the size of the next trade. The fallback stays, but it is now visible."""
    async def no_live_price(_addr):
        return None
    monkeypatch.setattr(portfolio.price_feed, "get_price_usd", no_live_price)

    db_session.add(_position(MINT_A, qty=1000.0, price=0.01))
    db_session.commit()

    valuation = await portfolio.value_open_positions(db_session)
    assert valuation.stale_positions >= 1
    assert valuation.stale_usd >= 10.0
    assert valuation.fully_priced is False
    assert valuation.stale_share > 0


async def test_a_priced_position_is_marked_to_market_not_to_cost(db_session, monkeypatch):
    """Scoped to this one position rather than the whole book: the test
    database is shared across the suite, so asserting on a portfolio total
    would make this test depend on which other tests ran first."""
    async def live_price(_addr):
        return 0.02
    monkeypatch.setattr(portfolio.price_feed, "get_price_usd", live_price)

    db_session.add(_position(MINT_A, qty=1000.0, price=0.01))
    db_session.commit()

    # Entry cost was $10; the live price has doubled it.
    exposure = await portfolio.get_token_exposure_usd(db_session, "PEPE", MINT_A)
    assert exposure == pytest.approx(20.0)

    valuation = await portfolio.value_open_positions(db_session)
    assert valuation.positions >= 1
    assert valuation.total_usd >= 20.0
