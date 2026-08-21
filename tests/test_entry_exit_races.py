"""Concurrent entry and exit races, driven through the real trading paths.

tests/test_concurrency.py exercises the reservation primitive on its own.
These tests are the other half: they run the actual functions the scanner,
the webhook and the position monitor call, so a future change that keeps
the primitive but stops using it still fails.

The invariants:

  C1  Two concurrent entries for the same mint open exactly one position.
  C2  Two concurrent entries for DIFFERENT mints are not serialised - the
      guard must not cost the scanner its throughput.
  C3  Two concurrent exits of one position sell it once and credit the
      proceeds once.
  C4  A partial exit and a full exit of one position are mutually
      exclusive, not merely each unique.
  C5  An exit that arrives after the position is already closed - by
      another session, so the caller's own object still reads OPEN - does
      nothing.
  C6  A refused exit leaves no trace: no trade row, no cash movement.
  C7  The cash ledger's read-modify-write is not interleavable.
"""
import asyncio

import pytest

from app import models
from app.concurrency import AlreadyReserved, clear_reservations, reserve_entry, reserve_exit
from app.config import settings
from app.database import SessionLocal
from app.services import portfolio, trading_service

pytestmark = pytest.mark.anyio

MINT = "RaceMint1111111111111111111111111111111111"
OTHER = "RaceMint2222222222222222222222222222222222"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _clean_reservations():
    clear_reservations()
    yield
    clear_reservations()


@pytest.fixture()
def clean_db():
    def wipe(session):
        for model in (models.Trade, models.Position, models.Signal, models.RiskEvent):
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


def _position(db, *, symbol="RACECOIN", token_address=MINT, qty=100.0, entry_price=1.0):
    pos = models.Position(
        symbol=symbol, token_address=token_address, qty=qty, initial_qty=qty,
        entry_price=entry_price, stop_loss=entry_price * 0.85,
        take_profit=entry_price * 1.3, status=models.PositionStatus.OPEN.value,
    )
    db.add(pos)
    db.commit()
    return pos


class _SlowSeller:
    """An execution client whose sell awaits, so two exits can interleave.

    The await is the whole point: without one, close_position would run to
    completion before the second caller ever got the event loop and the
    race could not be reproduced at all.
    """

    def __init__(self):
        self.sells = 0

    async def sell(self, _instrument, qty, _slippage_bps):
        self.sells += 1
        await asyncio.sleep(0.01)
        return _Result(qty)


class _Result:
    def __init__(self, qty):
        self.success = True
        self.filled_qty = qty
        self.avg_price = 1.10
        self.tx_hash = "0xrace"
        self.fee_usd = 0.0
        self.execution_cost_pct = 0.0
        self.fill_delay_seconds = 0.0
        self.error = None


@pytest.fixture()
def slow_seller(monkeypatch):
    client = _SlowSeller()
    monkeypatch.setattr(trading_service, "get_execution_client", lambda: client)

    async def quiet(*_a, **_k):
        return None
    monkeypatch.setattr(trading_service.notifier, "notify_trade_executed", quiet)
    monkeypatch.setattr(trading_service.notifier, "notify_error", quiet)
    monkeypatch.setattr(trading_service.notifier, "notify_risk_halt", quiet)
    return client


# ---------------------------------------------------------------------------
# C1/C2 - the entry side
# ---------------------------------------------------------------------------

async def test_two_concurrent_entries_for_one_mint_yield_one_position(clean_db, monkeypatch):
    """C1, through _handle_buy_signal rather than the primitive.

    The inner evaluation is stubbed to create the position directly: the
    real one needs the whole market-data stack, and what is under test
    here is whether the second caller is allowed to reach it at all.
    """
    entered = []

    async def fake_enter(db, signal):
        await asyncio.sleep(0.01)                  # the network round-trips
        entered.append(signal.id)
        _position(db, token_address=signal.token_address)

    monkeypatch.setattr(trading_service, "_evaluate_and_enter", fake_enter)

    signals = []
    for _ in range(2):
        s = models.Signal(symbol="RACECOIN", token_address=MINT, signal_type="buy", price=1.0)
        clean_db.add(s)
        signals.append(s)
    clean_db.commit()

    await asyncio.gather(*(trading_service._handle_buy_signal(clean_db, s) for s in signals))
    clean_db.commit()

    assert len(entered) == 1
    assert clean_db.query(models.Position).count() == 1


async def test_the_refused_entry_is_recorded_rather_than_silently_dropped(clean_db, monkeypatch):
    """A blocked duplicate is a fact about the funnel. Dropping it silently
    would make the gap between signals and positions unexplainable."""
    async def fake_enter(_db, _signal):
        await asyncio.sleep(0.01)

    monkeypatch.setattr(trading_service, "_evaluate_and_enter", fake_enter)

    signals = []
    for _ in range(2):
        s = models.Signal(symbol="RACECOIN", token_address=MINT, signal_type="buy", price=1.0)
        clean_db.add(s)
        signals.append(s)
    clean_db.commit()

    await asyncio.gather(*(trading_service._handle_buy_signal(clean_db, s) for s in signals))
    clean_db.commit()

    blocked = clean_db.query(models.RiskEvent).filter_by(
        event_type="duplicate_entry_blocked"
    ).all()
    assert len(blocked) == 1


async def test_two_different_mints_are_not_serialised(clean_db, monkeypatch):
    """C2. A guard that made the scanner evaluate candidates one at a time
    would cost most of its throughput to prevent a collision that cannot
    happen between different tokens."""
    running = 0
    peak = 0

    async def fake_enter(_db, _signal):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1

    monkeypatch.setattr(trading_service, "_evaluate_and_enter", fake_enter)

    a = models.Signal(symbol="ACOIN", token_address=MINT, signal_type="buy", price=1.0)
    b = models.Signal(symbol="BCOIN", token_address=OTHER, signal_type="buy", price=1.0)
    clean_db.add_all([a, b])
    clean_db.commit()

    await asyncio.gather(
        trading_service._handle_buy_signal(clean_db, a),
        trading_service._handle_buy_signal(clean_db, b),
    )
    assert peak == 2, "the two entries should have overlapped"


# ---------------------------------------------------------------------------
# C3-C6 - the exit side
# ---------------------------------------------------------------------------

async def test_two_concurrent_exits_sell_the_position_once(clean_db, slow_seller):
    """C3, the bug this reservation was added for.

    The position monitor's stop-loss and a TradingView sell alert are two
    callers of close_position on one event loop. Before the reservation,
    both read the same qty, both sold it, and both credited the proceeds -
    the paper account paid twice for a position it held once.
    """
    pos = _position(clean_db)
    cash_before = portfolio.get_cash_balance_usd(clean_db)

    await asyncio.gather(
        trading_service.close_position(clean_db, pos, reason="stop loss"),
        trading_service.close_position(clean_db, pos, reason="tradingview sell"),
    )
    clean_db.commit()

    assert slow_seller.sells == 1, "the position was sold twice"
    sells = clean_db.query(models.Trade).filter_by(side="sell").all()
    assert len(sells) == 1
    assert portfolio.get_cash_balance_usd(clean_db) == pytest.approx(cash_before + 110.0)
    assert pos.status == models.PositionStatus.CLOSED.value


async def test_a_partial_and_a_full_exit_are_mutually_exclusive(clean_db, slow_seller):
    """C4. Both read the same position.qty and both await; the second
    subtraction would be applied to a qty the first already reduced. One
    reservation per POSITION - rather than per exit kind - is what makes
    them exclude each other."""
    pos = _position(clean_db)

    await asyncio.gather(
        trading_service.close_position(clean_db, pos, reason="stop loss"),
        trading_service.partial_close_position(clean_db, pos, 0.5, reason="profit take"),
    )
    clean_db.commit()

    assert slow_seller.sells == 1
    assert len(clean_db.query(models.Trade).filter_by(side="sell").all()) == 1


async def test_an_exit_of_an_already_closed_position_does_nothing(clean_db, slow_seller):
    """C5, the sequential case the reservation cannot catch.

    The monitor loads a batch of open positions; a sell alert closes one
    and commits; the monitor reaches that row still holding the object it
    loaded, which reads OPEN as far as its own session is concerned.
    """
    pos = _position(clean_db)

    # Another session closes it and commits, exactly as the webhook would.
    other = SessionLocal()
    try:
        mirror = other.get(models.Position, pos.id)
        mirror.status = models.PositionStatus.CLOSED.value
        other.commit()
    finally:
        other.close()

    assert pos.status == models.PositionStatus.OPEN.value, "the stale read this test needs"

    await trading_service.close_position(clean_db, pos, reason="stop loss")
    clean_db.commit()

    assert slow_seller.sells == 0
    assert clean_db.query(models.Trade).filter_by(side="sell").count() == 0


async def test_a_refused_exit_leaves_no_trade_and_no_cash_movement(clean_db, slow_seller):
    """C6. A skipped exit must be a true no-op. A half-recorded one - a
    FAILED trade row, or a cash credit without a sale - would corrupt the
    ledger reconciliation that app/safety/killswitch.py depends on."""
    pos = _position(clean_db)
    cash_before = portfolio.get_cash_balance_usd(clean_db)

    async with reserve_exit(pos.id):
        await trading_service.close_position(clean_db, pos, reason="stop loss")
    clean_db.commit()

    assert slow_seller.sells == 0
    assert clean_db.query(models.Trade).count() == 0
    assert portfolio.get_cash_balance_usd(clean_db) == pytest.approx(cash_before)
    assert pos.status == models.PositionStatus.OPEN.value


async def test_exits_of_different_positions_do_not_block_each_other(clean_db, slow_seller):
    """Keyed on the position id, so closing one holding never delays
    closing another - which matters most in exactly the falling market
    where several stops fire at once."""
    a = _position(clean_db, symbol="ACOIN", token_address=MINT)
    b = _position(clean_db, symbol="BCOIN", token_address=OTHER)

    await asyncio.gather(
        trading_service.close_position(clean_db, a, reason="stop loss"),
        trading_service.close_position(clean_db, b, reason="stop loss"),
    )
    clean_db.commit()

    assert slow_seller.sells == 2
    assert a.status == models.PositionStatus.CLOSED.value
    assert b.status == models.PositionStatus.CLOSED.value


async def test_the_exit_reservation_is_released_after_a_failure(clean_db, monkeypatch):
    """A leaked exit reservation is the worst possible leak: the bot could
    no longer close that position at all, so the stop-loss would stop
    working on a holding that is already going wrong."""
    pos = _position(clean_db)

    class _Exploding:
        async def sell(self, *_a, **_k):
            raise RuntimeError("rpc died mid-sell")

    monkeypatch.setattr(trading_service, "get_execution_client", lambda: _Exploding())

    with pytest.raises(RuntimeError):
        await trading_service.close_position(clean_db, pos, reason="stop loss")

    # ...and the position is immediately closable again.
    async with reserve_exit(pos.id):
        pass


# ---------------------------------------------------------------------------
# C7 - the cash ledger
# ---------------------------------------------------------------------------

async def test_the_cash_ledger_read_modify_write_cannot_interleave(clean_db):
    """C7. Two sessions both reading $1,000, both subtracting $100, both
    writing $900 loses $100 from the books.

    adjust_cash_balance is synchronous with no await inside it, so on this
    bot's single event loop the read and the write cannot be separated -
    and that is a property of the code, not luck. This test would fail the
    moment someone made it async or put an await in the middle.
    """
    async def spend():
        portfolio.adjust_cash_balance(clean_db, -100.0)

    start = portfolio.get_cash_balance_usd(clean_db)
    await asyncio.gather(*(spend() for _ in range(10)))
    clean_db.commit()

    assert portfolio.get_cash_balance_usd(clean_db) == pytest.approx(start - 1000.0)


async def test_entry_and_exit_reservations_are_independent(clean_db):
    """Holding a mint for entry must not block exiting a position in it -
    they are different operations on different subjects, and conflating
    them would let an in-flight buy delay a stop-loss."""
    async with reserve_entry(MINT):
        async with reserve_exit(1):
            pass

    async with reserve_exit(1):
        async with reserve_entry(MINT):
            pass

    # ...while each still excludes its own kind.
    async with reserve_exit(1):
        with pytest.raises(AlreadyReserved):
            async with reserve_exit(1):
                pytest.fail("the second exit reservation must not be granted")
