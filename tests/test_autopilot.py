"""Tests for the self-improvement loop.

The loop's job is mostly to NOT act, so most of these assert restraint:
it refuses to touch source, refuses to reach the live gates, refuses to
keep retrying a rejected idea, and counts its own past attempts against
itself.
"""
import datetime as dt

import pytest

from app import models
from app.autopilot import changelog, diagnose as triage
from app.autopilot.loop import apply_remedies, blocked_reason, judge_challenger, run_once
from app.autopilot.promote import Arm
from app.config import settings
from app.database import SessionLocal

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def db():
    session = SessionLocal()
    def wipe():
        for row in session.query(models.AutopilotChange).all():
            session.delete(row)
        session.commit()
    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


# ---------------------------------------------------------------------------
# the hard limits
# ---------------------------------------------------------------------------

def test_autopilot_refuses_to_run_when_live_trading_is_on(monkeypatch):
    """An autonomous strategy-changer must not be running while real money
    is at stake, even though it cannot reach the live gates by design."""
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", True)
    monkeypatch.setattr(settings, "LIVE_TRADING", True)

    reason = blocked_reason()
    assert reason is not None
    assert "refuses to run against real funds" in reason


def test_autopilot_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", False)
    assert "AUTOPILOT_ENABLED=false" in blocked_reason()


def test_the_package_cannot_reach_the_live_trading_gates():
    """A grep-level guard. If a future edit gives the loop a path to
    LIVE_TRADING or LIVE_EXECUTION_ACKNOWLEDGED, this fails - which is the
    point, because that edit would otherwise look harmless in review."""
    import pathlib

    offenders = []
    for path in pathlib.Path("app/autopilot").rglob("*.py"):
        text = path.read_text()
        for gate in ("LIVE_EXECUTION_ACKNOWLEDGED", "SCANNER_ALLOW_LIVE_TRADING",
                     "EARLY_SIGNAL_MAY_TRADE"):
            if gate in text:
                offenders.append(f"{path}:{gate}")
        # LIVE_TRADING may only be READ, to refuse to run.
        if "LIVE_TRADING = " in text or "LIVE_TRADING=True" in text:
            offenders.append(f"{path}: assigns LIVE_TRADING")
    assert offenders == [], f"autopilot can reach a live gate: {offenders}"


def test_autopilot_does_not_write_source_files():
    """The loop diagnoses code-level problems and logs them. It must never
    patch: a fault in the fixer lands in the trading path, and the tests
    that would catch it are also files it could edit."""
    import pathlib

    dangerous = ("open(", "Path(", "subprocess", "os.system", "exec(", "eval(")
    offenders = []
    for path in pathlib.Path("app/autopilot").rglob("*.py"):
        body = "\n".join(
            line for line in path.read_text().splitlines()
            if not line.strip().startswith("#")
        )
        for token in dangerous:
            if token in body:
                offenders.append(f"{path.name}:{token}")
    assert offenders == [], f"autopilot touches the filesystem or shell: {offenders}"


# ---------------------------------------------------------------------------
# restraint
# ---------------------------------------------------------------------------

def test_structural_findings_are_logged_not_applied(db):
    """A finding about the strategy is a proposal for a human. Acting on
    it from a few hundred rows is how a loop fits itself to a fortnight."""
    diagnosis = triage.Diagnosis(findings=[
        triage.Finding("t", triage.STRUCTURAL, triage.WARNING, "threshold looks wrong", "d"),
    ])
    applied = apply_remedies(db, diagnosis)
    db.commit()

    assert applied == 0
    entry = changelog.history(db, limit=1)[0]
    assert entry.kind == changelog.PROPOSAL
    assert entry.applied is False


def test_operational_findings_are_applied_and_recorded(db):
    diagnosis = triage.Diagnosis(findings=[
        triage.Finding("p", triage.OPERATIONAL, triage.CRITICAL, "provider down", "d"),
    ])
    assert apply_remedies(db, diagnosis) == 1
    db.commit()
    assert changelog.history(db, limit=1)[0].kind == changelog.REMEDY


def test_a_rejected_challenger_is_recorded_so_it_cannot_be_quietly_retried(db):
    """A search that logs only its wins will retry a rejected idea until
    variance lets it through, and the changelog would show one clean
    promotion with no trace of the attempts."""
    champion = Arm("champ", [0.1] * 40, 10.0, {"bull": [0.1] * 40})
    challenger = Arm("chal", [0.1] * 5, 10.0, {"bull": [0.1] * 5})

    promoted = judge_challenger(
        db, champion=champion, challenger=challenger,
        target="min_score", before={"v": 65}, after={"v": 60},
    )
    db.commit()

    assert promoted is False
    entry = changelog.history(db, limit=1)[0]
    assert entry.kind == changelog.REJECTION
    assert entry.applied is False
    assert entry.after == {}, "a rejected change must not record a new value"


def test_repeated_attempts_tighten_the_bar_across_cycles(db):
    """A nightly search accumulates tries. Counting only within a cycle
    would understate the real number by however many nights it has run."""
    for _ in range(4):
        changelog.record(db, kind=changelog.REJECTION, target="min_score", summary="tried")
    db.commit()

    assert changelog.attempts_against(db, "min_score") == 4


def test_an_oscillating_parameter_is_left_alone(db):
    """Changed and reverted repeatedly is not optimisation converging, it
    is a value being fitted to whichever weeks are in the window."""
    for _ in range(3):
        entry = changelog.record(db, kind=changelog.PROMOTION, target="stop_pct", summary="up")
        db.flush()
        rollback = changelog.record(db, kind=changelog.ROLLBACK, target="stop_pct", summary="back")
        db.flush()
        changelog.mark_reverted(db, entry.id, rollback.id)
    db.commit()

    assert changelog.is_oscillating(db, "stop_pct") is True

    champion = Arm("champ", [0.1] * 60, 10.0, {"bull": [0.1] * 30, "chop": [0.1] * 30})
    challenger = Arm("chal", [0.9] * 60, 10.0, {"bull": [0.9] * 30, "chop": [0.9] * 30})
    assert judge_challenger(
        db, champion=champion, challenger=challenger,
        target="stop_pct", before={}, after={},
    ) is False, "an oscillating knob was changed again"
    db.commit()
    assert "oscillating" in changelog.history(db, limit=1)[0].summary


def test_a_rollback_is_linked_to_what_it_undid(db):
    original = changelog.record(db, kind=changelog.PROMOTION, target="x", summary="a")
    db.flush()
    rollback = changelog.record(db, kind=changelog.ROLLBACK, target="x", summary="b")
    db.flush()
    changelog.mark_reverted(db, original.id, rollback.id)
    db.commit()

    assert db.get(models.AutopilotChange, original.id).reverted_by_id == rollback.id


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------

def test_a_stalled_scanner_is_detected_from_recorded_data(db):
    """A stopped loop looks exactly like a market with no candidates, and
    neither writes an error."""
    db.add(models.PipelineEvent(
        stage="TECHNICAL_SCORE", symbol="T", chain="solana", passed=True,
        occurred_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=9),
    ))
    db.commit()
    try:
        codes = {f.code for f in triage.diagnose(db).findings}
        assert "scanner_stalled" in codes
    finally:
        for row in db.query(models.PipelineEvent).all():
            db.delete(row)
        db.commit()


def test_one_failing_check_does_not_blind_the_whole_triage(db, monkeypatch):
    def boom(_db):
        raise RuntimeError("check exploded")
    monkeypatch.setattr(triage, "_check_providers", boom)

    report = triage.diagnose(db)
    assert any(f.code == "diagnostic_failed" for f in report.findings)


async def test_a_cycle_is_inert_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", False)
    assert "skipped" in await run_once()


async def test_a_cycle_runs_and_reports(db, monkeypatch):
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", True)
    monkeypatch.setattr(settings, "LIVE_TRADING", False)

    summary = await run_once(db)
    assert "headline" in summary
    assert "error" not in summary
