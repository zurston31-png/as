"""Tests for app/safety/ - accounting reconciliation and the entry gate.

The gate exists for the two failure modes that never announce themselves:
trading on data that stopped updating, and sizing positions off a cash
balance that is wrong. Both keep producing trades and both look normal in
the logs, so these tests are mostly about proving the bot NOTICES.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.safety import killswitch
from app.safety.reconcile import check_position_integrity, reconcile
from app.services import api_health, portfolio

NOW = dt.datetime.now(dt.timezone.utc)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def clean_db():
    """An empty book with the ledger at its starting balance."""
    def wipe(session):
        for model in (models.Trade, models.Position, models.RiskEvent):
            for row in session.query(model).all():
                session.delete(row)
        portfolio.set_state(session, portfolio.CASH_KEY, settings.PORTFOLIO_STARTING_BALANCE_USD)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _buy(db, size_usd, *, symbol="RECCOIN", qty=1000.0, price=0.01, status="filled"):
    db.add(models.Trade(
        symbol=symbol, token_address=f"Mint{symbol}", side="buy", status=status,
        size_usd=size_usd, qty=qty, entry_price=price, opened_at=NOW,
    ))


def _sell(db, qty, exit_price, *, symbol="RECCOIN", status="filled"):
    db.add(models.Trade(
        symbol=symbol, token_address=f"Mint{symbol}", side="sell", status=status,
        size_usd=qty * exit_price, qty=qty, exit_price=exit_price, closed_at=NOW,
    ))


# ===========================================================================
# reconciliation
# ===========================================================================

def test_a_consistent_ledger_reconciles(clean_db):
    _buy(clean_db, 100.0, qty=10_000, price=0.01)
    _sell(clean_db, 10_000, 0.012)                       # $120 back
    portfolio.set_state(
        clean_db, portfolio.CASH_KEY,
        settings.PORTFOLIO_STARTING_BALANCE_USD - 100.0 + 120.0,
    )
    clean_db.commit()

    result = reconcile(clean_db)
    assert result.balanced
    assert result.buys_usd == pytest.approx(100.0)
    assert result.sells_usd == pytest.approx(120.0)
    assert "Books balance" in result.summary()


def test_a_lost_cash_write_is_detected(clean_db):
    """The exact drift an interrupted transaction leaves behind: the trade
    is recorded but the cash never moved."""
    _buy(clean_db, 250.0)
    clean_db.commit()   # ledger deliberately NOT adjusted

    result = reconcile(clean_db)
    assert not result.balanced
    assert result.discrepancy == pytest.approx(250.0)
    assert "ACCOUNTING DISCREPANCY" in result.summary()


def test_a_double_applied_adjustment_is_detected(clean_db):
    _buy(clean_db, 100.0)
    portfolio.adjust_cash_balance(clean_db, -100.0)
    portfolio.adjust_cash_balance(clean_db, -100.0)      # applied twice
    clean_db.commit()

    assert not reconcile(clean_db).balanced


def test_the_discrepancy_is_reported_not_silently_corrected(clean_db):
    """Overwriting the ledger with the computed value would hide whatever
    caused the drift and destroy the evidence needed to find it."""
    _buy(clean_db, 250.0)
    clean_db.commit()
    before = portfolio.get_cash_balance_usd(clean_db)

    result = reconcile(clean_db)

    assert portfolio.get_cash_balance_usd(clean_db) == before
    assert "Not auto-corrected" in result.summary()


def test_unfilled_trades_are_not_counted(clean_db):
    """A reverted swap moved no money and must not appear in the books."""
    _buy(clean_db, 500.0, status="failed")
    clean_db.commit()
    assert reconcile(clean_db).balanced


def test_a_negative_cash_balance_is_flagged_as_impossible(clean_db):
    portfolio.set_state(clean_db, portfolio.CASH_KEY, -50.0)
    clean_db.commit()

    result = reconcile(clean_db)
    assert not result.balanced
    assert any("NEGATIVE" in p for p in result.problems)


def test_a_sell_missing_its_exit_price_is_reported_rather_than_assumed(clean_db):
    """Guessing the proceeds would silently invent money."""
    db_trade = models.Trade(
        symbol="X", side="sell", status=models.TradeStatus.FILLED.value,
        size_usd=100.0, qty=1000.0, exit_price=None,
    )
    clean_db.add(db_trade)
    clean_db.commit()

    result = reconcile(clean_db)
    assert any("proceeds cannot be reconstructed" in p for p in result.problems)
    assert not result.balanced


def test_the_tolerance_scales_with_trade_count(clean_db):
    """A fixed threshold would either false-positive over a long history or
    miss a genuine loss on a short one."""
    small = reconcile(clean_db).tolerance_usd
    for _ in range(500):
        _buy(clean_db, 1.0)
    clean_db.commit()
    assert reconcile(clean_db).tolerance_usd >= small


# ===========================================================================
# position integrity
# ===========================================================================

def _position(db, **overrides):
    defaults = dict(
        symbol="POSCOIN", token_address="MintPos", chain="solana",
        qty=1000.0, initial_qty=1000.0, entry_price=0.01,
        stop_loss=0.0085, take_profit=0.013,
        status=models.PositionStatus.OPEN.value, mode="paper", opened_at=NOW,
    )
    defaults.update(overrides)
    pos = models.Position(**defaults)
    db.add(pos)
    return pos


def test_a_healthy_book_reports_no_problems(clean_db):
    _position(clean_db)
    clean_db.commit()
    assert check_position_integrity(clean_db) == []


def test_an_open_position_holding_nothing_is_caught(clean_db):
    _position(clean_db, qty=0.0)
    clean_db.commit()
    assert any("cannot be exited" in p for p in check_position_integrity(clean_db))


def test_a_zero_entry_price_is_caught(clean_db):
    """Every P&L and stop calculation divides by it."""
    _position(clean_db, entry_price=0.0)
    clean_db.commit()
    assert any("entry_price" in p for p in check_position_integrity(clean_db))


def test_a_stop_above_the_entry_is_caught(clean_db):
    _position(clean_db, stop_loss=0.02)
    clean_db.commit()
    assert any("exit immediately" in p for p in check_position_integrity(clean_db))


def test_holding_more_than_was_bought_is_caught(clean_db):
    _position(clean_db, qty=2000.0, initial_qty=1000.0)
    clean_db.commit()
    assert any("partial exit went the wrong way" in p for p in check_position_integrity(clean_db))


def test_a_closed_position_with_no_exit_trade_is_caught(clean_db):
    """Its proceeds were never credited - the exit leg was lost."""
    _position(clean_db, status=models.PositionStatus.CLOSED.value, closed_at=NOW)
    clean_db.commit()
    assert any("never credited" in p for p in check_position_integrity(clean_db))


# ===========================================================================
# the gate
# ===========================================================================

async def test_a_clean_bot_may_trade(clean_db):
    verdict = await killswitch.may_open_position(clean_db)
    assert verdict.may_trade
    assert verdict.failures == []
    assert "ENTRIES ALLOWED" in verdict.summary()


async def test_broken_accounting_blocks_new_entries(clean_db):
    _buy(clean_db, 250.0)
    clean_db.commit()

    verdict = await killswitch.may_open_position(clean_db)
    assert not verdict.may_trade
    assert "known to be wrong" in verdict.reason
    assert "ENTRIES BLOCKED" in verdict.summary()


async def test_a_corrupt_position_blocks_new_entries(clean_db):
    _position(clean_db, qty=0.0)
    clean_db.commit()
    verdict = await killswitch.may_open_position(clean_db)
    assert not verdict.may_trade


async def test_a_stale_price_feed_blocks_new_entries(clean_db, monkeypatch):
    """The failure the bot is least able to notice on its own: the last
    known price keeps being used and nothing looks wrong."""
    api_health.reset()
    api_health.record_success("dexscreener")
    record = api_health.get("dexscreener")
    record.last_success_at = NOW - dt.timedelta(hours=3)

    verdict = await killswitch.may_open_position(clean_db)
    assert not verdict.may_trade
    assert "dexscreener last succeeded" in verdict.reason
    api_health.reset()


async def test_repeated_failures_block_even_if_a_success_was_recent(clean_db):
    api_health.reset()
    api_health.record_success("dexscreener")
    for _ in range(settings.KILL_SWITCH_MAX_CONSECUTIVE_FAILURES):
        api_health.record_failure("dexscreener", "timeout")
    # Put a recent success timestamp back so only the streak can block.
    api_health.get("dexscreener").last_success_at = NOW

    verdict = await killswitch.may_open_position(clean_db)
    assert not verdict.may_trade
    assert "times in a row" in verdict.reason
    api_health.reset()


async def test_health_can_only_block_never_permit(clean_db):
    """The directional invariant promised by
    tests/test_api_health.py::test_health_tracking_can_only_ever_tighten_a_gate.

    A perfectly healthy API must not rescue a bot whose books are broken."""
    _buy(clean_db, 250.0)                    # accounting is now wrong
    clean_db.commit()
    api_health.reset()
    for _ in range(50):
        api_health.record_success("dexscreener")   # maximally healthy

    verdict = await killswitch.may_open_position(clean_db)
    assert not verdict.may_trade, "good health must not unblock a broken ledger"
    api_health.reset()


async def test_a_mostly_unpriceable_book_blocks_new_entries(clean_db, monkeypatch):
    """Position size is a fraction of portfolio value. If most of the book
    is valued at cost because prices stopped arriving, that fraction is
    computed from a number that no longer means anything."""
    async def no_price(_addr):
        return None
    monkeypatch.setattr(portfolio.price_feed, "get_price_usd", no_price)

    _position(clean_db)
    clean_db.commit()

    verdict = await killswitch.may_open_position(clean_db)
    assert not verdict.may_trade
    assert "valued at cost" in verdict.reason


async def test_a_check_that_raises_counts_as_a_failure(clean_db, monkeypatch):
    """Fail closed: a check that cannot run is not an absence of evidence."""
    def boom(_db):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(killswitch.reconcile_mod, "reconcile", boom)
    verdict = await killswitch.may_open_position(clean_db)
    assert not verdict.may_trade
    assert "treating as a failure" in verdict.reason


async def test_a_manual_halt_blocks_entries(clean_db):
    from app.risk.manager import halt_trading, resume_trading

    halt_trading(clean_db, "daily loss limit")
    clean_db.commit()
    assert not (await killswitch.may_open_position(clean_db)).may_trade

    resume_trading(clean_db)
    clean_db.commit()
    assert (await killswitch.may_open_position(clean_db)).may_trade


async def test_the_switch_can_be_disabled(clean_db, monkeypatch):
    _buy(clean_db, 250.0)
    clean_db.commit()
    monkeypatch.setattr(settings, "KILL_SWITCH_ENABLED", False)

    verdict = await killswitch.may_open_position(clean_db)
    assert verdict.may_trade
    assert verdict.disabled is True
    # "all integrity checks passed" and "nobody checked" must not read the
    # same on a dashboard - they mean opposite things.
    assert "NOT checked" in verdict.reason
    assert "UNCHECKED" in verdict.summary()


async def test_the_switch_never_closes_positions(clean_db):
    """Halting entries is safe. Force-liquidating a book because a price
    feed hiccuped turns a data problem into a realised loss."""
    _position(clean_db)
    _buy(clean_db, 250.0)                    # break accounting so it blocks
    clean_db.commit()
    open_before = clean_db.query(models.Position).filter_by(
        status=models.PositionStatus.OPEN.value
    ).count()

    verdict = await killswitch.may_open_position(clean_db)
    assert not verdict.may_trade

    open_after = clean_db.query(models.Position).filter_by(
        status=models.PositionStatus.OPEN.value
    ).count()
    assert open_after == open_before
    assert "Open positions are unaffected" in verdict.summary()


async def test_the_verdict_serialises(clean_db):
    import json

    json.dumps((await killswitch.may_open_position(clean_db)).as_dict(), allow_nan=False)
