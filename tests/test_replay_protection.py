"""Tests for duplicate/replay protection (app/idempotency.py).

The invariants:

  R1  The key is deterministic: the same alert always produces the same
      key, across processes and restarts. A key that varied would make
      every replay look new.
  R2  Genuinely different alerts get different keys - different bar,
      different direction, different mint, different price.
  R3  A replayed alert produces no second signal row and never reaches the
      buy or sell path.
  R4  The uniqueness is enforced by the DATABASE, not by the application's
      look-then-insert. A duplicate inserted behind the application's back
      is still rejected.
  R5  An alert that cannot be keyed is never silently treated as a
      duplicate - suppressing a real signal is worse than processing one
      twice - and the reason it is unprotected is stated.
  R6  An existing database gets the unique index on upgrade.
"""
import datetime as dt

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from app import idempotency, models
from app.database import SessionLocal, engine
from app.schemas import TradingViewAlert
from app.services import trading_service

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


BAR_MS = 1766248800000
MINT = "So11111111111111111111111111111111111111112"
OTHER_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@pytest.fixture()
def clean_db():
    def wipe(session):
        for model in (models.Trade, models.Position, models.Signal, models.RiskEvent):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _alert(**overrides) -> TradingViewAlert:
    payload = {
        "secret": "test-webhook-secret-please-ignore",
        "symbol": "WSOL",
        "token_address": MINT,
        "chain": "solana",
        "signal": "buy",
        "price": 1.25,
        "time": str(BAR_MS),
    }
    payload.update(overrides)
    return TradingViewAlert.model_validate(payload)


def _key(alert: TradingViewAlert) -> str | None:
    return idempotency.alert_key(
        source="tradingview",
        symbol=alert.symbol,
        token_address=alert.token_address,
        signal_type=alert.signal,
        event_time=alert.parsed_time(),
        price=alert.price,
    )


@pytest.fixture()
def no_trading(monkeypatch):
    """Stub the buy and sell paths and count how often they run.

    These tests are about whether a duplicate REACHES the trading logic,
    not about what the trading logic then does - and driving the real buy
    path would need the whole market-data stack standing up.
    """
    calls = {"buy": 0, "sell": 0}

    async def buy(_db, _signal):
        calls["buy"] += 1

    async def sell(_db, _signal):
        calls["sell"] += 1

    monkeypatch.setattr(trading_service, "_handle_buy_signal", buy)
    monkeypatch.setattr(trading_service, "_handle_sell_signal", sell)
    return calls


# ---------------------------------------------------------------------------
# R1/R2 - the key itself
# ---------------------------------------------------------------------------

def test_the_key_is_deterministic():
    """R1. Two separate computations of the same alert agree. This is what
    makes the key survive a restart - a process-local counter or a
    random nonce would leave a replay window open on every deploy."""
    assert _key(_alert()) == _key(_alert())


def test_the_key_is_stable_across_equivalent_timezone_spellings():
    """The same instant written two ways is one event, not two."""
    utc = idempotency.alert_key(
        source="tradingview", symbol="WSOL", token_address=MINT, signal_type="buy",
        event_time=dt.datetime(2025, 12, 20, 18, 0, tzinfo=dt.timezone.utc), price=1.0,
    )
    offset = idempotency.alert_key(
        source="tradingview", symbol="WSOL", token_address=MINT, signal_type="buy",
        event_time=dt.datetime(2025, 12, 20, 13, 0,
                               tzinfo=dt.timezone(dt.timedelta(hours=-5))), price=1.0,
    )
    assert utc == offset


def test_a_naive_timestamp_is_read_as_utc():
    """Everything else in this bot stores UTC; a naive datetime arriving
    here is UTC that lost its tzinfo, not local time."""
    naive = idempotency.alert_key(
        source="tradingview", symbol="WSOL", token_address=MINT, signal_type="buy",
        event_time=dt.datetime(2025, 12, 20, 18, 0), price=1.0,
    )
    aware = idempotency.alert_key(
        source="tradingview", symbol="WSOL", token_address=MINT, signal_type="buy",
        event_time=dt.datetime(2025, 12, 20, 18, 0, tzinfo=dt.timezone.utc), price=1.0,
    )
    assert naive == aware


@pytest.mark.parametrize("difference", [
    {"time": str(BAR_MS + 300_000)},      # the next bar
    {"signal": "sell"},                   # the opposite direction
    {"token_address": OTHER_MINT},        # a different mint
    {"price": 1.26},                      # an intrabar re-fire
])
def test_genuinely_different_alerts_get_different_keys(difference):
    """R2. Every one of these is a real, distinct event. Collapsing any of
    them into the first alert's key would discard a signal the bot was
    supposed to act on, which is the expensive direction of this error."""
    assert _key(_alert()) != _key(_alert(**difference))


def test_two_mints_sharing_a_ticker_are_not_the_same_alert():
    """Keyed on the canonical mint, like everything else in this bot - see
    app/identity.py. Two unrelated tokens both called PEPE must not
    deduplicate against each other."""
    a = _key(_alert(symbol="PEPE", token_address=MINT))
    b = _key(_alert(symbol="PEPE", token_address=OTHER_MINT))
    assert a != b


def test_the_key_version_is_part_of_the_key():
    """Changing what goes into the key without changing the version would
    make every previously-seen alert look new - a silent replay window on
    exactly the deploy that redefines the rule."""
    baseline = _key(_alert())
    original = idempotency.KEY_VERSION
    try:
        idempotency.KEY_VERSION = original + 1
        assert _key(_alert()) != baseline
    finally:
        idempotency.KEY_VERSION = original


# ---------------------------------------------------------------------------
# R3 - a replay does not reach the trading path
# ---------------------------------------------------------------------------

async def test_a_replayed_buy_creates_one_signal_and_trades_once(clean_db, no_trading):
    """R3, the whole point: a retried delivery must not open a second
    position the risk limits never sized for."""
    first = await trading_service.handle_alert(clean_db, _alert())
    clean_db.commit()
    second = await trading_service.handle_alert(clean_db, _alert())
    clean_db.commit()

    assert not first.duplicate
    assert second.duplicate
    assert second.signal.id == first.signal.id
    assert clean_db.query(models.Signal).count() == 1
    assert no_trading["buy"] == 1


async def test_a_replayed_sell_does_not_close_the_position_twice(clean_db, no_trading):
    """The dangerous direction. A duplicate buy is caught downstream by
    the open-position check; a duplicate sell has no such backstop - the
    position is there, and the close path would happily sell it again."""
    sell = _alert(signal="sell")
    await trading_service.handle_alert(clean_db, sell)
    clean_db.commit()
    await trading_service.handle_alert(clean_db, sell)
    clean_db.commit()

    assert no_trading["sell"] == 1


async def test_the_next_bar_is_not_a_replay(clean_db, no_trading):
    """R2 end to end. Protection that also swallowed the next genuine
    signal would be worse than none."""
    await trading_service.handle_alert(clean_db, _alert())
    clean_db.commit()
    outcome = await trading_service.handle_alert(
        clean_db, _alert(time=str(BAR_MS + 300_000))
    )
    clean_db.commit()

    assert not outcome.duplicate
    assert clean_db.query(models.Signal).count() == 2
    assert no_trading["buy"] == 2


# ---------------------------------------------------------------------------
# R4 - the database is the guarantee
# ---------------------------------------------------------------------------

def test_the_unique_index_exists_on_the_signals_table(clean_db):
    """R4. The application's SELECT-then-INSERT has a gap in it; the index
    is what closes the gap. If this index is missing, every other test in
    this file is checking an optimisation rather than a guarantee."""
    indexes = inspect(engine).get_indexes("signals")
    match = next((ix for ix in indexes if ix["name"] == "ix_signals_idempotency_key"), None)
    assert match is not None, "the idempotency index is not in the database"
    assert match["unique"], "the index exists but does not enforce uniqueness"
    assert match["column_names"] == ["idempotency_key"]


def test_the_database_rejects_a_duplicate_key_inserted_behind_the_app(clean_db):
    """R4 again, from the other side: bypass handle_alert entirely and the
    constraint still holds. This is the test that would fail if someone
    replaced the index with an application-level check."""
    for _ in range(2):
        clean_db.add(models.Signal(
            idempotency_key="a" * 64, symbol="WSOL", token_address=MINT,
            signal_type="buy", price=1.0,
        ))
    with pytest.raises(IntegrityError):
        clean_db.commit()
    clean_db.rollback()


def test_many_unkeyable_signals_can_coexist(clean_db):
    """A unique index treats NULLs as distinct in both SQLite and
    Postgres. Without that, the second unkeyable alert ever received would
    be rejected as a duplicate of the first - which is precisely the false
    suppression the NULL exists to avoid."""
    for _ in range(3):
        clean_db.add(models.Signal(
            idempotency_key=None, symbol="WSOL", token_address=MINT,
            signal_type="buy", price=1.0,
        ))
    clean_db.commit()
    assert clean_db.query(models.Signal).filter(
        models.Signal.idempotency_key.is_(None)
    ).count() == 3


# ---------------------------------------------------------------------------
# R5 - an alert that cannot be keyed
# ---------------------------------------------------------------------------

def test_an_alert_with_no_time_has_no_key():
    """R5. Two genuine alerts on consecutive bars are byte-identical
    without `time`, so any key built from what remains would reject the
    second real signal."""
    assert _key(_alert(time=None)) is None


def test_the_reason_names_the_missing_field_and_the_fix():
    """A NULL key is a gap in protection, and an operator can only close
    it if the message says which field is missing from their alert."""
    reason = idempotency.unprotected_reason(None)
    assert reason and "time" in reason
    assert idempotency.unprotected_reason(dt.datetime.now(dt.timezone.utc)) is None


async def test_two_unkeyable_alerts_are_both_processed(clean_db, no_trading):
    """R5 end to end. Failing closed here would mean discarding real
    signals from anyone whose alert body predates the `time` field."""
    await trading_service.handle_alert(clean_db, _alert(time=None))
    clean_db.commit()
    outcome = await trading_service.handle_alert(clean_db, _alert(time=None))
    clean_db.commit()

    assert not outcome.duplicate
    assert clean_db.query(models.Signal).count() == 2
    assert no_trading["buy"] == 2


async def test_an_unkeyable_alert_reports_why(clean_db, no_trading):
    outcome = await trading_service.handle_alert(clean_db, _alert(time=None))
    clean_db.commit()
    assert outcome.unprotected_reason is not None
    assert "time" in outcome.unprotected_reason


async def test_a_keyed_alert_reports_no_gap(clean_db, no_trading):
    outcome = await trading_service.handle_alert(clean_db, _alert())
    clean_db.commit()
    assert outcome.unprotected_reason is None


# ---------------------------------------------------------------------------
# R6 - upgrading an existing database
# ---------------------------------------------------------------------------

def test_an_existing_database_gains_the_index_on_upgrade(tmp_path):
    """R6. The bot's databases are migrated additively rather than by
    Alembic (app/migrations.py). A constraint that only ever appears on a
    freshly created database would leave every real deployment - all of
    which predate this change - completely unprotected.

    Declared as an Index rather than `unique=True` on the column for
    exactly this reason: the migration's second pass creates missing
    indexes, and would not have created a bare UniqueConstraint.
    """
    from app.migrations import apply_additive_migrations

    old_db = tmp_path / "old.db"
    old_engine = create_engine(f"sqlite:///{old_db}")
    with old_engine.begin() as conn:
        # A signals table as it looked before this change: no key column,
        # so no index either.
        conn.execute(text(
            "CREATE TABLE signals ("
            "id INTEGER PRIMARY KEY, symbol VARCHAR(64), token_address VARCHAR(128), "
            "chain VARCHAR(32), signal_type VARCHAR(16), price FLOAT)"
        ))

    assert "idempotency_key" not in {
        c["name"] for c in inspect(old_engine).get_columns("signals")
    }

    applied = apply_additive_migrations(old_engine)

    assert "signals.idempotency_key" in applied
    assert "index:ix_signals_idempotency_key" in applied

    index = next(
        ix for ix in inspect(old_engine).get_indexes("signals")
        if ix["name"] == "ix_signals_idempotency_key"
    )
    assert index["unique"]

    # And it really constrains the upgraded table, rather than just
    # appearing in the catalogue.
    with old_engine.begin() as conn:
        conn.execute(text("INSERT INTO signals (idempotency_key) VALUES ('dup')"))
    with pytest.raises(IntegrityError):
        with old_engine.begin() as conn:
            conn.execute(text("INSERT INTO signals (idempotency_key) VALUES ('dup')"))
    old_engine.dispose()
