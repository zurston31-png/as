"""Tests for rejection analytics (app/analysis/rejections.py) and its page.

The invariants:

  J1  Attribution is per MINT, not per event. One token re-evaluated forty
      times counts once, or the busiest token masquerades as the biggest
      bottleneck.
  J2  A mint that failed a gate earlier and passed it later was not
      stopped by that gate.
  J3  A mint's terminal stage is the DEEPEST one it reached, and a mint
      that opened a position has no terminal rejection at all.
  J4  Reasons are collapsed to categories, so prices and scores inside the
      text do not produce one "category" per event.
  J5  A near miss is measured against the threshold actually configured,
      and the SECURITY score is inverted - it rejects for being too HIGH.
  J6  Rates are withheld below a sample floor rather than reported
      confidently from three observations.
  J7  An empty window says "nothing was recorded", never "nothing was
      rejected".
  J8  Every figure traces to a recorded event: no estimates, no backfill.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.rejections import (
    MIN_MINTS_FOR_SHARE,
    NEAR_MISS_POINTS,
    build_rejection_report,
)
from app.config import settings
from app.database import SessionLocal

NOW = dt.datetime.now(dt.timezone.utc)
MINT_A = "RejectMintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
MINT_B = "RejectMintBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


@pytest.fixture()
def clean_db():
    def wipe(session):
        for row in session.query(models.PipelineEvent).all():
            session.delete(row)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _event(db, *, stage, passed, mint=MINT_A, symbol="RJC", reason=None,
           score=None, minutes_ago=1.0):
    db.add(models.PipelineEvent(
        occurred_at=NOW - dt.timedelta(minutes=minutes_ago),
        token_address=mint, symbol=symbol, chain="solana",
        stage=stage, passed=passed, reason=reason, score=score, detail={},
    ))


def _stage(report, name):
    return next((s for s in report.stages if s.stage == name), None)


# ---------------------------------------------------------------------------
# J1/J2 - per-mint attribution
# ---------------------------------------------------------------------------

def test_one_mint_rejected_repeatedly_counts_once(clean_db):
    """J1. The scanner re-evaluates the same tokens every pass. Counting
    events would let one token that will never pass the liquidity floor
    look like forty separate candidates lost to it."""
    for i in range(40):
        _event(clean_db, stage="PRESCREEN", passed=False,
               reason="liquidity $812 below scanner minimum", minutes_ago=i)
    clean_db.commit()

    report = build_rejection_report(clean_db, window_hours=24)
    prescreen = _stage(report, "PRESCREEN")

    assert prescreen.mints_stopped == 1
    assert prescreen.events_failed == 40
    assert report.mints_seen == 1


def test_the_repeat_rate_makes_a_busy_token_visible(clean_db):
    """The event count is still reported - it is real - but next to a
    per-mint repeat rate, so a reader can tell forty tokens failing once
    from one token failing forty times."""
    for i in range(40):
        _event(clean_db, stage="PRESCREEN", passed=False,
               reason="liquidity below scanner minimum", minutes_ago=i)
    clean_db.commit()

    reason = _stage(build_rejection_report(clean_db, window_hours=24), "PRESCREEN").reasons[0]
    assert reason.mints == 1
    assert reason.events == 40
    assert reason.repeats_per_mint == pytest.approx(40.0)


def test_a_mint_that_later_passed_was_not_stopped_by_that_gate(clean_db):
    """J2. A cooldown rejection at 09:00 followed by a buy at 11:00 is not
    the risk gate costing a candidate - it is the risk gate working."""
    _event(clean_db, stage="RISK", passed=False, reason="cooldown active", minutes_ago=120)
    _event(clean_db, stage="RISK", passed=True, reason="within every portfolio limit", minutes_ago=1)
    clean_db.commit()

    risk = _stage(build_rejection_report(clean_db, window_hours=24), "RISK")
    assert risk.mints_reaching == 1
    assert risk.mints_passed == 1
    assert risk.mints_stopped == 0
    # ...and the rejection itself is still on the record.
    assert risk.events_failed == 1


# ---------------------------------------------------------------------------
# J3 - terminal stage
# ---------------------------------------------------------------------------

def test_a_mint_is_attributed_to_the_deepest_stage_it_reached(clean_db):
    """J3. Passing PRESCREEN and dying at SECURITY is a security
    rejection, not a prescreen one, even though both stages saw it."""
    _event(clean_db, stage="DISCOVERED", passed=True, minutes_ago=5)
    _event(clean_db, stage="PRESCREEN", passed=True, minutes_ago=4)
    _event(clean_db, stage="RISK", passed=True, minutes_ago=3)
    _event(clean_db, stage="SECURITY", passed=False, reason="top 10 holders hold 71%", minutes_ago=2)
    clean_db.commit()

    report = build_rejection_report(clean_db, window_hours=24)
    assert report.terminal_stage_counts == {"SECURITY": 1}


def test_a_mint_that_opened_a_position_is_not_a_rejection(clean_db):
    """Reaching OPEN_POSITION is the success case. Filing it under
    "stopped at OPEN_POSITION" would read as a failure."""
    for stage in ("DISCOVERED", "PRESCREEN", "RISK", "SECURITY", "OPEN_POSITION"):
        _event(clean_db, stage=stage, passed=True, minutes_ago=5)
    clean_db.commit()

    report = build_rejection_report(clean_db, window_hours=24)
    assert report.terminal_stage_counts == {}
    assert report.mints_reaching_a_position == 1


def test_two_mints_dying_at_different_stages_are_counted_separately(clean_db):
    _event(clean_db, stage="PRESCREEN", passed=False, mint=MINT_A, reason="liquidity too low")
    _event(clean_db, stage="PRESCREEN", passed=True, mint=MINT_B)
    _event(clean_db, stage="SECURITY", passed=False, mint=MINT_B, reason="mint authority not renounced")
    clean_db.commit()

    report = build_rejection_report(clean_db, window_hours=24)
    assert report.terminal_stage_counts == {"PRESCREEN": 1, "SECURITY": 1}
    assert report.mints_seen == 2


def test_the_biggest_blocker_is_the_stage_stopping_the_most_mints(clean_db):
    for i in range(6):
        _event(clean_db, stage="PRESCREEN", passed=False, mint=f"Mint{i}",
               reason="liquidity too low")
    _event(clean_db, stage="PRESCREEN", passed=True, mint=MINT_B)
    _event(clean_db, stage="SECURITY", passed=False, mint=MINT_B, reason="honeypot")
    clean_db.commit()

    report = build_rejection_report(clean_db, window_hours=24)
    assert report.biggest_blocker.stage == "PRESCREEN"


# ---------------------------------------------------------------------------
# J4 - reason categories
# ---------------------------------------------------------------------------

def test_reasons_are_grouped_by_category_not_by_exact_text(clean_db):
    """J4. The raw reasons embed dollar amounts, so grouping on the exact
    string would produce one 'category' per event and a table of 300 rows
    each with a count of one."""
    for i, amount in enumerate([812, 1_204, 66, 9_990]):
        _event(clean_db, stage="PRESCREEN", passed=False, mint=f"Mint{i}",
               reason=f"liquidity ${amount:,} below scanner minimum", minutes_ago=i)
    clean_db.commit()

    reasons = _stage(build_rejection_report(clean_db, window_hours=24), "PRESCREEN").reasons
    assert len(reasons) == 1
    assert reasons[0].mints == 4


def test_reasons_are_ranked_by_mints_before_events(clean_db):
    """A reason affecting many tokens outranks one affecting a single
    token many times - which is the opposite of what an event-count
    ranking would say."""
    for i in range(10):
        _event(clean_db, stage="RISK", passed=False, mint=MINT_A,
               reason="cooldown active", minutes_ago=i)
    for i in range(4):
        _event(clean_db, stage="RISK", passed=False, mint=f"Mint{i}",
               reason="max concurrent positions reached", minutes_ago=i)
    clean_db.commit()

    reasons = _stage(build_rejection_report(clean_db, window_hours=24), "RISK").reasons
    assert reasons[0].reason.startswith("max concurrent positions")
    assert reasons[0].mints == 4
    assert reasons[1].mints == 1
    assert reasons[1].events == 10


# ---------------------------------------------------------------------------
# J5 - near misses
# ---------------------------------------------------------------------------

def test_a_near_miss_is_measured_against_the_configured_threshold(clean_db, monkeypatch):
    """J5. Hard-coding the pass mark would silently mis-measure the moment
    someone changed the setting."""
    monkeypatch.setattr(settings, "MIN_SIGNAL_SCORE_TO_ENTER", 65.0)
    _event(clean_db, stage="TECHNICAL_SCORE", passed=False, mint=MINT_A,
           reason="score below entry threshold", score=63.0)          # near
    _event(clean_db, stage="TECHNICAL_SCORE", passed=False, mint=MINT_B,
           reason="score below entry threshold", score=20.0)          # not near
    clean_db.commit()

    stage = _stage(build_rejection_report(clean_db, window_hours=24), "TECHNICAL_SCORE")
    assert stage.scored_rejections == 2
    assert stage.near_misses == 1
    assert stage.median_rejected_score == pytest.approx(41.5)


def test_the_security_score_is_inverted(clean_db, monkeypatch):
    """J5. Rug risk rejects for being too HIGH, so a near miss is just
    ABOVE the line. Treating it like the others would count the safest
    rejected tokens as near misses and the riskiest as comfortable."""
    monkeypatch.setattr(settings, "REJECT_RUG_SCORE_ABOVE", 65.0)
    _event(clean_db, stage="SECURITY", passed=False, mint=MINT_A,
           reason="rug risk above threshold", score=67.0)             # near
    _event(clean_db, stage="SECURITY", passed=False, mint=MINT_B,
           reason="rug risk above threshold", score=95.0)             # not near
    clean_db.commit()

    stage = _stage(build_rejection_report(clean_db, window_hours=24), "SECURITY")
    assert stage.near_misses == 1


def test_a_stage_that_does_not_score_reports_no_near_misses(clean_db):
    """PRESCREEN has no 0-100 score. Reporting zero near misses for it
    would imply it was measured and found clean."""
    _event(clean_db, stage="PRESCREEN", passed=False, reason="liquidity too low")
    clean_db.commit()

    stage = _stage(build_rejection_report(clean_db, window_hours=24), "PRESCREEN")
    assert stage.scored_rejections == 0
    assert stage.near_miss_share is None
    assert stage.median_rejected_score is None


def test_the_near_miss_margin_is_inclusive_of_the_boundary(clean_db, monkeypatch):
    monkeypatch.setattr(settings, "MIN_SIGNAL_SCORE_TO_ENTER", 65.0)
    _event(clean_db, stage="TECHNICAL_SCORE", passed=False, mint=MINT_A,
           reason="below threshold", score=65.0 - NEAR_MISS_POINTS)
    clean_db.commit()

    assert _stage(build_rejection_report(clean_db, window_hours=24), "TECHNICAL_SCORE").near_misses == 1


# ---------------------------------------------------------------------------
# J6/J7/J8 - honesty about the sample
# ---------------------------------------------------------------------------

def test_a_rate_is_withheld_below_the_sample_floor(clean_db):
    """J6. "100% of candidates rejected" computed from two mints invites
    exactly the over-reading this module exists to discourage."""
    for i in range(MIN_MINTS_FOR_SHARE - 1):
        _event(clean_db, stage="PRESCREEN", passed=False, mint=f"Mint{i}", reason="thin")
    clean_db.commit()

    assert _stage(build_rejection_report(clean_db, window_hours=24), "PRESCREEN").stop_rate is None


def test_a_rate_appears_once_there_is_enough_to_report(clean_db):
    for i in range(MIN_MINTS_FOR_SHARE):
        _event(clean_db, stage="PRESCREEN", passed=False, mint=f"Mint{i}", reason="thin")
    clean_db.commit()

    stage = _stage(build_rejection_report(clean_db, window_hours=24), "PRESCREEN")
    assert stage.stop_rate == pytest.approx(1.0)


def test_an_empty_window_says_nothing_was_recorded(clean_db):
    """J7. "No rejections" and "the scanner never ran" look identical on a
    chart and mean opposite things."""
    report = build_rejection_report(clean_db, window_hours=24)
    assert not report.has_data
    assert report.stages == []
    assert "not the same as" in report.note


def test_events_outside_the_window_are_excluded(clean_db):
    _event(clean_db, stage="PRESCREEN", passed=False, reason="thin", minutes_ago=60 * 48)
    clean_db.commit()

    assert not build_rejection_report(clean_db, window_hours=24).has_data
    assert build_rejection_report(clean_db, window_hours=None).has_data


def test_events_without_a_mint_are_counted_but_not_attributed(clean_db):
    """J8. They are real events and belong in the event totals, but they
    cannot be attributed to a token - and merging them under a shared
    placeholder would make them look like one very busy mint."""
    clean_db.add(models.PipelineEvent(
        occurred_at=NOW, token_address=None, symbol="NOADDR", stage="PRESCREEN",
        passed=False, reason="no contract address supplied", detail={},
    ))
    clean_db.commit()

    report = build_rejection_report(clean_db, window_hours=24)
    stage = _stage(report, "PRESCREEN")
    assert stage.events_failed == 1
    assert stage.mints_reaching == 0
    assert report.mints_seen == 0
    assert "no mint address" in report.note


# ---------------------------------------------------------------------------
# the page itself
# ---------------------------------------------------------------------------

def test_the_rejections_page_renders(clean_db):
    from fastapi.testclient import TestClient
    from app.main import app

    _event(clean_db, stage="PRESCREEN", passed=False, reason="liquidity $812 below minimum")
    clean_db.commit()

    client = TestClient(app)
    auth = (settings.DASHBOARD_USERNAME, settings.DASHBOARD_PASSWORD)
    page = client.get("/rejections", auth=auth)
    assert page.status_code == 200
    assert "Rejection Analytics" in page.text
    assert "PRESCREEN" in page.text


def test_the_rejections_page_renders_with_no_data_at_all(clean_db):
    """The empty state is the one an operator sees first, on a fresh
    install - it has to be right."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    page = client.get("/rejections", auth=(settings.DASHBOARD_USERNAME, settings.DASHBOARD_PASSWORD))
    assert page.status_code == 200
    assert "No pipeline events" in page.text or "not the same as" in page.text


def test_the_rejections_api_returns_the_same_numbers(clean_db):
    from fastapi.testclient import TestClient
    from app.main import app

    _event(clean_db, stage="PRESCREEN", passed=False, reason="liquidity too low")
    clean_db.commit()

    client = TestClient(app)
    body = client.get(
        "/api/rejections?hours=24",
        auth=(settings.DASHBOARD_USERNAME, settings.DASHBOARD_PASSWORD),
    ).json()
    assert body["mints_seen"] == 1
    assert body["terminal_stage_counts"] == {"PRESCREEN": 1}


def test_the_page_requires_authentication():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    assert client.get("/rejections").status_code == 401
    assert client.get("/api/rejections").status_code == 401
