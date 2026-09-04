"""A failed execution has to leave a record and raise the alarm.

Review round 5 asked for this and it was genuinely missing: three code
paths set `TradeStatus.FAILED` and call `notify_error` - the buy, the full
exit and the partial exit - and not one of them had a test. The only
appearances of `TradeStatus.FAILED` anywhere in the suite were fixtures
setting it up, never an assertion that the write path produces it.

WHY IT MATTERS MORE THAN A FILLED-PATH TEST

A fill that does not get recorded is caught within minutes: the position
is missing and the dashboard shows it. A FAILURE that does not get
recorded is invisible by construction. There is no position to be absent,
so the only trace it was ever attempted is the row this code writes, and
the funnel's gap between "signals" and "positions" quietly becomes
unexplainable - the exact thing app/pipeline.py exists to prevent.

On the exit side it is worse than a lost record. A sell that fails and
says nothing leaves a position open that the operator believes was
closed, still carrying risk, with no alert.

These run entirely in paper mode. The execution client is stubbed to
return a failed SwapResult, which is what any backend returns on a
reverted swap - no live flags, no network.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.execution.base import SwapResult
from app.services import portfolio, trading_service

pytestmark = pytest.mark.anyio

MINT = "FailMint111111111111111111111111111111111111"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def clean_db():
    def wipe(session):
        for model in (models.Trade, models.Position, models.Signal, models.RiskEvent):
            for row in session.query(model).all():
                if (getattr(row, "symbol", "") or "").startswith("FAILCOIN") or model is models.RiskEvent:
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


class _RefusingClient:
    """Every order comes back failed, the way a reverted swap does.

    The real SwapResult rather than a stub - a look-alike drifts the
    moment a field is added, which has already happened once in this
    suite (see tests/test_entry_exit_races.py).
    """

    REASON = "swap reverted: insufficient liquidity"

    async def buy(self, _instrument, _usd_amount, _slippage_bps=None):
        return SwapResult(success=False, error=self.REASON)

    async def sell(self, _instrument, _qty, _slippage_bps=None):
        return SwapResult(success=False, error=self.REASON)


@pytest.fixture()
def refusing_client(monkeypatch):
    client = _RefusingClient()
    monkeypatch.setattr(trading_service, "get_execution_client", lambda: client)
    return client


@pytest.fixture()
def captured_errors(monkeypatch):
    """Collect notify_error calls instead of sending them."""
    sent: list[str] = []

    async def capture(message):
        sent.append(message)

    async def quiet(*_a, **_k):
        return None

    monkeypatch.setattr(trading_service.notifier, "notify_error", capture)
    monkeypatch.setattr(trading_service.notifier, "notify_trade_executed", quiet)
    monkeypatch.setattr(trading_service.notifier, "notify_risk_halt", quiet)
    return sent


def _open_position(db, *, qty=100.0, entry_price=1.0):
    pos = models.Position(
        symbol="FAILCOIN", token_address=MINT, chain="solana",
        qty=qty, initial_qty=qty, entry_price=entry_price,
        stop_loss=entry_price * 0.85, take_profit=entry_price * 1.3,
        status=models.PositionStatus.OPEN.value,
        opened_at=dt.datetime.now(dt.timezone.utc),
    )
    db.add(pos)
    db.commit()
    return pos


# ---------------------------------------------------------------------------
# the exit side - a failed sell must not look like a closed position
# ---------------------------------------------------------------------------

async def test_a_failed_sell_records_the_trade_and_alerts(clean_db, refusing_client, captured_errors):
    """The row is the only evidence the attempt happened."""
    position = _open_position(clean_db)

    await trading_service._close_position(
        clean_db, position, "stop loss", signal_id=None
    )
    clean_db.commit()

    trade = (
        clean_db.query(models.Trade)
        .filter(models.Trade.position_id == position.id)
        .one()
    )
    assert trade.status == models.TradeStatus.FAILED.value
    assert _RefusingClient.REASON in (trade.error or ""), (
        "the failure reason must be persisted, not just logged"
    )

    assert captured_errors, "a failed sell sent no alert"
    assert "FAILCOIN" in captured_errors[0]


async def test_a_failed_sell_leaves_the_position_open(clean_db, refusing_client, captured_errors):
    """The dangerous half. Marking it closed on a sell that never
    happened would drop a live position out of the monitor's view while
    it still carries risk, and out of every exposure total."""
    position = _open_position(clean_db)

    await trading_service._close_position(
        clean_db, position, "stop loss", signal_id=None
    )
    clean_db.commit()

    refreshed = clean_db.get(models.Position, position.id)
    assert refreshed.status == models.PositionStatus.OPEN.value
    assert refreshed.qty == pytest.approx(100.0), "quantity was reduced by a sell that failed"
    assert refreshed.closed_at is None


async def test_a_failed_sell_moves_no_cash(clean_db, refusing_client, captured_errors):
    """No fill, no proceeds. Crediting a failed sell would inflate the
    paper account without any position leaving the book."""
    before = portfolio.get_cash_balance_usd(clean_db)
    position = _open_position(clean_db)

    await trading_service._close_position(
        clean_db, position, "stop loss", signal_id=None
    )
    clean_db.commit()

    assert portfolio.get_cash_balance_usd(clean_db) == pytest.approx(before)


async def test_a_failed_partial_exit_records_and_keeps_the_whole_position(
    clean_db, refusing_client, captured_errors
):
    """Same contract on the partial path, which has its own copy of the
    branch and so needs its own test."""
    position = _open_position(clean_db)

    await trading_service._partial_close_position(
        clean_db, position, 0.5, "take partial profit", signal_id=None
    )
    clean_db.commit()

    trade = (
        clean_db.query(models.Trade)
        .filter(models.Trade.position_id == position.id)
        .one()
    )
    assert trade.status == models.TradeStatus.FAILED.value

    refreshed = clean_db.get(models.Position, position.id)
    assert refreshed.qty == pytest.approx(100.0), "a failed partial still reduced the position"
    assert refreshed.status == models.PositionStatus.OPEN.value
    assert captured_errors, "a failed partial exit sent no alert"
