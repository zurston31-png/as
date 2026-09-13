"""Auditing what the recorded paper fills were actually charged.

The fill model itself is covered by tests/test_fill_model.py. This is the
different question: whether the trades a RUN produced were costed at all. A
misconfiguration or a market snapshot failing on every call leaves the model
intact and the recorded fills free, and every performance statistic
inherits that without being able to detect it.
"""
import pytest

from app import models
from app.analysis.fill_audit import (
    MAX_FAVOURABLE_SHARE_PCT,
    MIN_FILLS_FOR_VERDICT,
    build_fill_audit,
)
from app.config import settings
from app.database import SessionLocal


@pytest.fixture()
def db():
    def wipe(session):
        for row in session.query(models.Trade).all():
            session.delete(row)
        session.commit()

    session = SessionLocal()
    wipe(session)
    try:
        yield session
    finally:
        wipe(session)
        session.close()


def _fill(db, *, cost_pct=0.004, fee=0.05, delay=1.5, side="buy", symbol="TESTCOIN"):
    db.add(models.Trade(
        symbol=symbol, side=side, qty=100.0, size_usd=100.0, entry_price=1.0,
        status="filled", mode="paper",
        execution_cost_pct=cost_pct, fee_usd=fee, fill_delay_seconds=delay,
    ))


def test_an_uncosted_fill_is_reported_as_broken_not_as_cheap(db):
    """The failure that matters. A trade with no execution cost was free,
    and every number computed from it is overstated - but nothing else in
    the system notices, because a missing cost reads as a good fill."""
    _fill(db, cost_pct=None)
    _fill(db)
    db.commit()

    audit = build_fill_audit(db)
    assert len(audit.uncosted) == 1
    assert "BROKEN" in audit.verdict()
    assert "were free" in audit.verdict()


def test_a_missing_fee_is_named_even_when_the_cost_was_recorded(db):
    _fill(db, fee=None)
    db.commit()
    problems = build_fill_audit(db).fills[0].problems()
    assert any("no fee recorded" in p for p in problems)


def test_a_zero_fee_on_a_real_notional_is_flagged(db):
    """Paper trading charging no fees was a real bug in this repo's history
    (see the commit that fixed it). It must not be able to come back
    silently."""
    _fill(db, fee=0.0)
    db.commit()
    assert any("zero fee" in p for p in build_fill_audit(db).fills[0].problems())


def test_a_missing_confirmation_delay_is_flagged(db):
    _fill(db, delay=None)
    db.commit()
    assert any("confirmation delay" in p for p in build_fill_audit(db).fills[0].problems())


def test_two_good_fills_are_insufficient_to_declare_the_model_healthy(db):
    """Directly the situation this bot is in: two profitable paper trades.
    Both were costed, which is worth saying, and it is not evidence that the
    fill distribution is realistic."""
    _fill(db)
    _fill(db, side="sell")
    db.commit()

    audit = build_fill_audit(db)
    assert audit.n == 2
    assert not audit.uncosted
    verdict = audit.verdict()
    assert "INSUFFICIENT DATA" in verdict
    assert "PLAUSIBLE" not in verdict


def test_mostly_favourable_fills_are_called_suspicious(db):
    """Impact, spread and fees are all non-negative and only drift can be
    negative, so a run where most fills beat the reference price is a run
    where the costs are not being applied."""
    for _ in range(MIN_FILLS_FOR_VERDICT):
        _fill(db, cost_pct=-0.002)
    db.commit()

    audit = build_fill_audit(db)
    assert audit.favourable_share_pct == pytest.approx(100.0)
    assert "SUSPICIOUS" in audit.verdict()


def test_an_occasional_favourable_fill_is_accepted(db):
    """Favourable drift is real. The check is about bulk, not about
    forbidding a fill that happened to land well."""
    favourable = int(MIN_FILLS_FOR_VERDICT * MAX_FAVOURABLE_SHARE_PCT / 100) - 1
    for i in range(MIN_FILLS_FOR_VERDICT):
        _fill(db, cost_pct=-0.001 if i < favourable else 0.004)
    db.commit()

    audit = build_fill_audit(db)
    assert audit.favourable_share_pct < MAX_FAVOURABLE_SHARE_PCT
    assert "PLAUSIBLE" in audit.verdict()


def test_the_reported_floor_is_the_spread_plus_fee_from_settings(db):
    """The number a reader compares the mean cost against has to come from
    the same settings the fill model uses, or the comparison is decorative."""
    _fill(db)
    db.commit()
    expected = (settings.PAPER_SPREAD_PCT + settings.PAPER_FEE_PCT) * 100
    assert build_fill_audit(db).floor_pct == pytest.approx(expected)


def test_live_trades_are_not_mixed_into_the_paper_audit(db):
    """Whatever a live row's costs look like, they are not evidence about
    the simulator."""
    _fill(db)
    db.add(models.Trade(
        symbol="LIVECOIN", side="buy", qty=1.0, size_usd=1.0, entry_price=1.0,
        status="filled", mode="live",
        execution_cost_pct=None, fee_usd=None, fill_delay_seconds=None,
    ))
    db.commit()

    audit = build_fill_audit(db)
    assert audit.n == 1
    assert not audit.uncosted


def test_no_fills_says_so_rather_than_passing_vacuously(db):
    assert "NO FILLS" in build_fill_audit(db).verdict()
