"""Announcing the closed-trade count as it crosses the gate's minimum.

The validation gate will not judge the strategy below 100 closed trades,
which makes 100 the number the whole paper run waits on - and the count
lives in a database on a VPS behind an authenticated dashboard, so
noticing it meant remembering to look. These tests pin the behaviour that
matters for a message arriving on someone's phone: it fires once, it fires
on the right number, and it never takes a trade down with it.
"""
import asyncio
import datetime as dt

import pytest

from app import models
from app.database import SessionLocal
from app.notifications import milestones
from app.state import get_state, set_state

START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(autouse=True)
def clean_state():
    """Each test owns the milestone marker and the trades table.

    The marker is process-wide state in a shared test database, so a test
    that left 100 behind would silence every test after it.
    """
    session = SessionLocal()
    try:
        session.query(models.Trade).delete()
        session.query(models.BotState).filter(
            models.BotState.key == milestones.STATE_KEY
        ).delete()
        session.commit()
    finally:
        session.close()
    yield
    session = SessionLocal()
    try:
        session.query(models.Trade).delete()
        session.query(models.BotState).filter(
            models.BotState.key == milestones.STATE_KEY
        ).delete()
        session.commit()
    finally:
        session.close()


def _add_closed(session, n: int) -> None:
    for i in range(n):
        session.add(
            models.Trade(
                symbol="COIN", side="sell", status=models.TradeStatus.FILLED.value,
                pnl_usd=1.0, closed_at=START + dt.timedelta(minutes=i),
            )
        )
    session.flush()


@pytest.fixture()
def sent(monkeypatch):
    """Capture what would have gone to Telegram/Discord."""
    messages: list[str] = []

    async def fake(text):
        messages.append(text)

    from app.notifications.notifier import notifier

    monkeypatch.setattr(notifier, "notify_milestone", fake)
    return messages


# --- counting ---------------------------------------------------------------

def test_only_closed_legs_are_counted(db):
    """An open position is not a data point. The count has to match the
    one the performance report judges, or the alert announces a sample
    the gate cannot see."""
    _add_closed(db, 3)
    db.add(models.Trade(symbol="COIN", side="buy", status="filled", opened_at=START))
    db.add(models.Trade(symbol="COIN", side="sell", status="filled", pnl_usd=None))
    db.flush()
    assert milestones.count_closed_trades(db) == 3


# --- first run --------------------------------------------------------------

def test_history_already_past_a_milestone_is_not_announced(db, sent):
    """The record was 33 trades deep when this shipped. Greeting the
    operator with "25 closed trades" would be announcing the past."""
    _add_closed(db, 34)
    assert asyncio.run(milestones.announce_if_crossed(db)) is None
    assert sent == []
    assert get_state(db, milestones.STATE_KEY) == 25


def test_a_fresh_database_seeds_at_zero_and_stays_quiet(db, sent):
    assert asyncio.run(milestones.announce_if_crossed(db)) is None
    assert sent == []
    assert get_state(db, milestones.STATE_KEY) == 0


# --- crossing ---------------------------------------------------------------

def test_the_gate_minimum_is_announced_with_what_it_means(db, sent):
    set_state(db, milestones.STATE_KEY, 50)
    _add_closed(db, 100)
    assert asyncio.run(milestones.announce_if_crossed(db)) == 100
    assert len(sent) == 1
    assert "100 closed trades" in sent[0]
    assert "/performance" in sent[0]


def test_below_the_gate_the_message_says_how_many_are_left(db, sent):
    set_state(db, milestones.STATE_KEY, 25)
    _add_closed(db, 50)
    assert asyncio.run(milestones.announce_if_crossed(db)) == 50
    assert "50 more" in sent[0]


def test_it_announces_once_and_then_stays_quiet(db, sent):
    """The check runs after every closed leg. Without the persisted
    marker the operator would get a message per trade forever."""
    set_state(db, milestones.STATE_KEY, 50)
    _add_closed(db, 100)
    assert asyncio.run(milestones.announce_if_crossed(db)) == 100

    _add_closed(db, 5)
    assert asyncio.run(milestones.announce_if_crossed(db)) is None
    assert len(sent) == 1


def test_a_jump_past_several_milestones_announces_the_highest(db, sent):
    """A batch of partial exits closing at once, or a database restored
    mid-run, must not leave the marker stuck below the real count - the
    next milestone would then never fire."""
    set_state(db, milestones.STATE_KEY, 25)
    _add_closed(db, 250)
    assert asyncio.run(milestones.announce_if_crossed(db)) == 200
    assert get_state(db, milestones.STATE_KEY) == 200
    assert len(sent) == 1


def test_a_shrinking_count_never_re_announces(db, sent):
    """A database restored from an older backup goes backwards. The
    marker only ever moves up, so the operator is not told about 100
    twice."""
    set_state(db, milestones.STATE_KEY, 200)
    _add_closed(db, 100)
    assert asyncio.run(milestones.announce_if_crossed(db)) is None
    assert sent == []


# --- it must never break a trade -------------------------------------------

def test_a_failing_notification_does_not_propagate(db, monkeypatch):
    """This is called on the exit path, after the position has already
    been closed and the cash ledger adjusted. A notification that raised
    here would surface as a failed exit."""
    from app.notifications.notifier import notifier

    async def boom(text):
        raise RuntimeError("telegram is down")

    monkeypatch.setattr(notifier, "notify_milestone", boom)
    set_state(db, milestones.STATE_KEY, 50)
    _add_closed(db, 100)
    assert asyncio.run(milestones.announce_if_crossed(db)) is None


def test_the_pending_sell_leg_is_visible_to_the_count(db, sent):
    """`SessionLocal` is autoflush=False. Counting without flushing first
    would miss the leg the caller has only added, and the hundredth trade
    would announce on the hundred-and-first."""
    set_state(db, milestones.STATE_KEY, 50)
    for i in range(99):
        db.add(models.Trade(symbol="COIN", side="sell", status="filled",
                            pnl_usd=1.0, closed_at=START + dt.timedelta(minutes=i)))
    db.flush()
    # The hundredth is added but deliberately NOT flushed.
    db.add(models.Trade(symbol="COIN", side="sell", status="filled",
                        pnl_usd=1.0, closed_at=START + dt.timedelta(minutes=99)))
    assert asyncio.run(milestones.announce_if_crossed(db)) == 100
