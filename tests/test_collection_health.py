"""Tests for app/analysis/collection.py.

The point of these checks is to catch a pipeline that is quietly writing
unusable observations, so what matters most is that the FAILING paths
actually fire. A health board that only ever goes green is worse than no
board at all - it converts an unnoticed problem into a reassured one.
"""
import datetime as dt

import pytest

from app import models
from app.analysis import collection
from app.analysis.collection import FAIL, INSUFFICIENT, PASS, WARN, check_collection
from app.config import settings
from app.database import SessionLocal

NOW = dt.datetime(2026, 5, 1, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def db():
    session = SessionLocal()

    def wipe():
        for model in (models.ShadowHorizonReturn, models.ShadowPosition, models.ShadowDecision):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


@pytest.fixture(autouse=True)
def one_challenger(monkeypatch):
    monkeypatch.setattr(
        settings, "SHADOW_CHALLENGERS",
        '[{"strategy_id": "strict-70", "min_score_to_enter": 70}]',
    )
    monkeypatch.setattr(settings, "MIN_SIGNAL_SCORE_TO_ENTER", 65.0)


def named(report, name):
    return next(c for c in report.checks if c.name == name)


def decide(db, *, oid, strategy_id, regime="bull/normal/deep_liquidity",
           version="v-test", entered=True):
    row = models.ShadowDecision(
        opportunity_id=oid, strategy_id=strategy_id, strategy_version=version,
        is_champion=strategy_id == "champion", token_address=f"Mint{oid}", symbol="T",
        chain="solana", decision="BUY" if entered else "REJECT", reason="x",
        signal_score=70.0, market_regime=regime,
        liquidity_regime=regime.split("/")[-1] if regime else None,
        entry_price=1.0 if entered else None, fill_succeeded=entered or None,
        fee_pct=0.0025, slippage_pct=0.0, size_usd=100.0, decided_at=NOW,
    )
    db.add(row)
    db.flush()
    return row


def position(db, decision, *, resolved=True, opened_at=NOW, mfe=12.0, mae=-4.0,
             gross=8.0, closed=True):
    row = models.ShadowPosition(
        decision_id=decision.id, opportunity_id=decision.opportunity_id,
        strategy_id=decision.strategy_id, token_address=decision.token_address,
        symbol="T", opened_at=opened_at, entry_price=1.0, size_usd=100.0,
        fees_pct=0.0025, slippage_pct=0.0,
    )
    if resolved:
        row.closed_at = opened_at + dt.timedelta(hours=1)
        row.exit_price = 1.08
        row.gross_return_pct = gross
        row.return_pct = gross - 0.25
        row.max_favorable_pct = mfe
        row.max_adverse_pct = mae
        row.hold_minutes = 60.0
    elif closed:
        row.closed_at = opened_at + dt.timedelta(hours=1)
    db.add(row)
    db.flush()
    return row


def populate(db, *, n=60, resolved=True, regime="bull/normal/deep_liquidity"):
    """A healthy run: both arms on every opportunity, all outcomes resolved."""
    for i in range(n):
        # Vary the regime across the set so the context check has contrast.
        label = regime if i % 2 else "bear/high_volatility/thin_liquidity"
        for strategy_id in ("champion", "strict-70"):
            decision = decide(db, oid=f"opp{i}", strategy_id=strategy_id, regime=label)
            position(db, decision, resolved=resolved)
    db.commit()


# ---------------------------------------------------------------------------
# an empty dataset is not a passing dataset
# ---------------------------------------------------------------------------

def test_an_empty_dataset_never_reports_pass(db):
    """A green board on zero rows is the single most expensive thing this
    file could produce: it says "collection is working" to someone who
    would otherwise go and look."""
    report = check_collection(db)
    graded = [c for c in report.checks if c.status == PASS]
    assert [c.name for c in graded] == ["no duplicate observations"]
    assert report.failures == []
    assert len(report.blocked) >= 6


def test_a_healthy_run_passes_every_check(db):
    populate(db)
    report = check_collection(db)
    assert report.failures == []
    assert named(report, "champion/challenger pairing").status == PASS
    assert named(report, "positions resolve").status == PASS
    assert named(report, "MFE / MAE populate").status == PASS
    assert named(report, "horizon returns populate").status == INSUFFICIENT  # none written


# ---------------------------------------------------------------------------
# the failures worth catching early
# ---------------------------------------------------------------------------

def test_a_challenger_that_never_loaded_is_a_failure(db):
    """The symptom of a malformed SHADOW_CHALLENGERS entry is an arm with no
    data, which otherwise only becomes visible weeks in."""
    for i in range(30):
        decide(db, oid=f"opp{i}", strategy_id="champion")
    db.commit()

    report = check_collection(db)
    assert named(report, "decisions recorded").status == FAIL
    assert "strict-70" in named(report, "decisions recorded").detail


def test_arms_measured_on_different_flow_is_a_failure(db):
    """Pairing is what makes the comparison controlled. An arm that shares
    no opportunities with the champion is being measured on different
    flow, and sample size cannot fix that."""
    for i in range(30):
        decide(db, oid=f"champ{i}", strategy_id="champion")
        decide(db, oid=f"other{i}", strategy_id="strict-70")
    db.commit()

    check = named(check_collection(db), "champion/challenger pairing")
    assert check.status == FAIL
    assert "different flow" in check.detail


def test_positions_that_never_resolve_are_a_failure(db):
    """The exact fault the resolver was built to fix - so it is worth
    asserting that the fix is still running."""
    stale = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)
    for i in range(5):
        decision = decide(db, oid=f"opp{i}", strategy_id="champion")
        position(db, decision, resolved=False, closed=False, opened_at=stale)
    db.commit()

    check = named(check_collection(db), "positions resolve")
    assert check.status == FAIL
    assert check.counts["stalled"] == 5


def test_a_missing_envelope_on_a_resolved_position_is_a_failure(db):
    populate(db, n=40)
    victim = db.query(models.ShadowPosition).first()
    victim.max_favorable_pct = None
    db.commit()

    check = named(check_collection(db), "MFE / MAE populate")
    assert check.status == FAIL
    assert check.counts["missing"] == 1


def test_an_impossible_envelope_is_a_failure(db):
    """MFE below MAE cannot happen, and a return outside the recorded path
    means the exit price came from somewhere the candles never went. Both
    are corruption, not a surprising market."""
    populate(db, n=40)
    victim = db.query(models.ShadowPosition).first()
    victim.max_favorable_pct, victim.max_adverse_pct = -20.0, 5.0
    db.commit()

    check = named(check_collection(db), "MFE / MAE populate")
    assert check.status == FAIL
    assert check.counts["inverted"] == 1


def test_a_return_outside_the_recorded_path_is_a_failure(db):
    populate(db, n=40)
    victim = db.query(models.ShadowPosition).first()
    victim.gross_return_pct = 400.0          # far above its own MFE of 12%
    db.commit()

    assert named(check_collection(db), "MFE / MAE populate").counts["off_path"] == 1


def test_a_missing_regime_column_is_a_failure(db):
    """The consistency bar groups by regime; rows without one cannot be
    grouped at all."""
    for i in range(60):
        for strategy_id in ("champion", "strict-70"):
            decide(db, oid=f"opp{i}", strategy_id=strategy_id, regime=None)
    db.commit()

    check = named(check_collection(db), "regime / liquidity context")
    assert check.status == FAIL


def test_a_single_regime_is_a_warning_not_a_pass(db):
    """One condition seen for a month is a real limitation on what the run
    can ever conclude - worth knowing on day three, not in week six."""
    populate(db, n=40, regime="bull/normal/deep_liquidity")
    for row in db.query(models.ShadowDecision).all():
        row.market_regime = "bull/normal/deep_liquidity"
        row.liquidity_regime = "deep_liquidity"
    db.commit()

    check = named(check_collection(db), "regime / liquidity context")
    assert check.status == WARN
    assert "trend" in check.detail


def test_a_zero_price_horizon_is_a_failure(db):
    """A dead feed written as a real quote. Every return computed from it
    is -100%, which looks like a rug rather than like missing data."""
    populate(db, n=40)
    p = db.query(models.ShadowPosition).first()
    db.add(models.ShadowHorizonReturn(
        position_id=p.id, opportunity_id=p.opportunity_id, strategy_id=p.strategy_id,
        token_address=p.token_address, horizon_minutes=60, due_at=NOW,
        price_at_horizon=0.0,
    ))
    db.commit()

    check = named(check_collection(db), "horizon returns populate")
    assert check.status == FAIL
    assert check.counts["zero_priced"] == 1


def test_mixed_strategy_versions_are_flagged(db):
    """Pooling across versions describes a strategy that never existed. A
    WARN, not a FAIL - the older rows are still valid evidence about the
    older version, they just must not be averaged in."""
    populate(db, n=30)
    for row in db.query(models.ShadowDecision).limit(10).all():
        row.strategy_version = "v-older"
    db.commit()

    check = named(check_collection(db), "single strategy version")
    assert check.status == WARN
    assert "Do not pool" in check.detail


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------

def test_progress_tracks_the_slowest_arm(db, monkeypatch):
    """A comparison is limited by whichever challenger has the least data,
    so averaging the arms would report progress the experiment does not
    have."""
    monkeypatch.setattr(
        settings, "SHADOW_CHALLENGERS",
        '[{"strategy_id": "strict-70", "min_score_to_enter": 70},'
        ' {"strategy_id": "loose-60", "min_score_to_enter": 60}]',
    )
    monkeypatch.setattr(collection, "TARGET_PAIRS", 100)
    for i in range(40):
        decide(db, oid=f"opp{i}", strategy_id="champion")
        decide(db, oid=f"opp{i}", strategy_id="strict-70")
        if i < 10:
            decide(db, oid=f"opp{i}", strategy_id="loose-60")
    db.commit()

    report = check_collection(db)
    assert report.paired == {"strict-70": 40, "loose-60": 10}
    assert report.progress_pct == pytest.approx(10.0)
