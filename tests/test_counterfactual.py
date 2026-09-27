"""Tests for app/analysis/counterfactual.py.

The failure this analysis exists to catch is a filter that throws away the
best setups while producing a perfectly clean trade log. The tests that
matter are therefore the ones proving it FIRES when that happens - and the
one proving it stays silent about the safety gates no matter how good
their rejects looked.
"""
import datetime as dt

import pytest

from app import models, pipeline
from app.analysis import counterfactual
from app.analysis.counterfactual import MIN_COHORT, build_counterfactual
from app.config import settings
from app.database import SessionLocal

NOW = dt.datetime(2026, 5, 1, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def db():
    session = SessionLocal()

    def wipe():
        for model in (models.ForwardReturn, models.PipelineEvent):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


def candidate(db, *, i, return_pct, rejected_at=None, observed_at=None):
    """One scored candidate: a pipeline trail plus its measured outcome."""
    token = f"Mint{i}"
    at = observed_at or NOW
    stages = [pipeline.TECHNICAL_SCORE, pipeline.MARKET_QUALITY,
              pipeline.SECURITY, pipeline.DATA_QUALITY]
    for offset, stage in enumerate(stages):
        passed = stage != rejected_at
        db.add(models.PipelineEvent(
            occurred_at=at + dt.timedelta(seconds=offset),
            token_address=token, symbol="T", chain="solana",
            stage=stage, passed=passed, reason="x",
        ))
        if not passed:
            break
    db.add(models.ForwardReturn(
        pipeline_event_id=0, token_address=token, symbol="T", observed_at=at,
        score=70.0, price_at_signal=1.0, horizon_minutes=60,
        due_at=at + dt.timedelta(minutes=60), return_pct=return_pct,
        strategy_version="v-test",
    ))


def gate(report, stage):
    return next(g for g in report.gates if g.stage == stage)


# ---------------------------------------------------------------------------
# the finding it exists to produce
# ---------------------------------------------------------------------------

def test_a_filter_rejecting_the_better_opportunities_is_flagged(db):
    """The whole point. A filter that throws away the best setups produces
    exactly the same clean trade log as one that throws away the worst,
    and only the rejects can tell the two apart."""
    for i in range(MIN_COHORT + 10):
        candidate(db, i=i, return_pct=2.0)                                     # accepted
        candidate(db, i=1000 + i, return_pct=25.0, rejected_at=pipeline.MARKET_QUALITY)
    db.commit()

    report = build_counterfactual(db, horizon_minutes=60)
    verdict = gate(report, pipeline.MARKET_QUALITY)
    assert verdict.grade() == "FAIL"
    assert verdict.difference > 0
    assert verdict.p_value is not None and verdict.p_value <= counterfactual.ALPHA
    assert [g.stage for g in report.flagged] == [pipeline.MARKET_QUALITY]
    assert "OUTPERFORMED" in verdict.note()


def test_a_filter_keeping_the_better_half_passes(db):
    for i in range(MIN_COHORT + 10):
        candidate(db, i=i, return_pct=12.0)
        candidate(db, i=1000 + i, return_pct=-30.0, rejected_at=pipeline.MARKET_QUALITY)
    db.commit()

    report = build_counterfactual(db, horizon_minutes=60)
    assert gate(report, pipeline.MARKET_QUALITY).grade() == "PASS"
    assert report.flagged == []
    assert "keeping the better half" in report.verdict()


# ---------------------------------------------------------------------------
# the safety rail
# ---------------------------------------------------------------------------

def test_a_safety_gate_is_never_flagged_however_good_its_rejects_looked(db):
    """A rug that would have paid for eleven minutes is not evidence the
    rug filter is too strict. The one thing an automated search must never
    be able to do is talk itself into switching off the safety rail."""
    for i in range(MIN_COHORT + 10):
        candidate(db, i=i, return_pct=1.0)
        candidate(db, i=1000 + i, return_pct=80.0, rejected_at=pipeline.SECURITY)
    db.commit()

    report = build_counterfactual(db, horizon_minutes=60)
    verdict = gate(report, pipeline.SECURITY)
    assert verdict.protected is True
    assert verdict.difference > 0                 # the cost is real and reported
    assert report.flagged == []                   # and it is not a tuning target
    assert "SAFETY gate" in verdict.note()
    assert "nothing here recommends loosening it" in verdict.note()


def test_every_loss_preventing_gate_is_protected():
    """Security, data quality and risk exist to prevent loss, not to select
    for return. Judging them on the return of what they rejected is the
    wrong question by construction."""
    assert counterfactual.PROTECTED_STAGES == {
        pipeline.SECURITY, pipeline.DATA_QUALITY, pipeline.RISK,
    }


def test_the_module_never_recommends_a_change_to_anything():
    """It is a screening report. Any actual change goes through the
    promotion gate as a challenger, on its own paired sample - so this file
    must not contain the machinery to apply one."""
    import pathlib

    body = pathlib.Path("app/analysis/counterfactual.py").read_text()
    for forbidden in ("settings.MIN_", "db.add(", "db.commit(", "changelog", "promote("):
        assert forbidden not in body


# ---------------------------------------------------------------------------
# refusing to conclude
# ---------------------------------------------------------------------------

def test_a_thin_cohort_is_not_compared(db):
    for i in range(MIN_COHORT + 5):
        candidate(db, i=i, return_pct=2.0)
    for i in range(5):
        candidate(db, i=1000 + i, return_pct=99.0, rejected_at=pipeline.MARKET_QUALITY)
    db.commit()

    verdict = gate(build_counterfactual(db, horizon_minutes=60), pipeline.MARKET_QUALITY)
    assert verdict.grade() == "INSUFFICIENT_DATA"
    assert "statement about the sample" in verdict.note()


def test_stages_before_tracking_starts_are_named_not_rendered_empty(db):
    """An empty row reads as "this filter rejects nothing worth having",
    which is the opposite of what an absent measurement means."""
    report = build_counterfactual(db, horizon_minutes=60)
    assert pipeline.RISK in report.invisible_stages
    assert pipeline.HISTORY in report.invisible_stages
    assert [g.stage for g in report.gates] == []


def test_the_prescreen_is_visible_only_while_its_sampling_is_on(db, monkeypatch):
    """It rejects more candidates than every other gate combined, so
    leaving it unmeasurable left the biggest filter permanently
    unexaminable. Reporting it as unmeasurable while its rows sit in the
    table would send someone looking for a hole that is no longer there -
    and reporting it as measurable when the sampling is off would hide a
    real one."""
    monkeypatch.setattr(settings, "SCANNER_TRACK_PRESCREEN_REJECTS", True)
    assert pipeline.PRESCREEN not in build_counterfactual(db).invisible_stages

    monkeypatch.setattr(settings, "SCANNER_TRACK_PRESCREEN_REJECTS", False)
    assert pipeline.PRESCREEN in build_counterfactual(db).invisible_stages


def test_a_candidate_with_no_pipeline_trail_is_counted_not_assumed_accepted(db):
    """Guessing "accepted" would credit the filters with outcomes they
    never saw."""
    db.add(models.ForwardReturn(
        pipeline_event_id=0, token_address="Orphan", symbol="T", observed_at=NOW,
        score=70.0, price_at_signal=1.0, horizon_minutes=60,
        due_at=NOW + dt.timedelta(minutes=60), return_pct=50.0,
        strategy_version="v-test",
    ))
    db.commit()

    report = build_counterfactual(db, horizon_minutes=60)
    assert report.unmatched == 1
    assert report.accepted.n == 0


def test_the_earliest_gate_in_pipeline_order_owns_the_rejection(db):
    """Two gates can record within the same second. Attributing the
    rejection to whichever row was written first would scatter one
    filter's cost across its neighbours."""
    at = NOW
    for stage in (pipeline.SECURITY, pipeline.MARKET_QUALITY):
        db.add(models.PipelineEvent(
            occurred_at=at, token_address="Both", symbol="T", chain="solana",
            stage=stage, passed=False, reason="x",
        ))
    db.add(models.ForwardReturn(
        pipeline_event_id=0, token_address="Both", symbol="T", observed_at=at,
        score=70.0, price_at_signal=1.0, horizon_minutes=60,
        due_at=at + dt.timedelta(minutes=60), return_pct=5.0, strategy_version="v-test",
    ))
    db.commit()

    report = build_counterfactual(db, horizon_minutes=60)
    # MARKET_QUALITY runs before SECURITY in STAGE_ORDER.
    assert [g.stage for g in report.gates] == [pipeline.MARKET_QUALITY]


def test_a_later_scan_of_the_same_token_is_not_pulled_into_the_window(db):
    """Matching by time window means the window has to be tight enough not
    to capture the same token's next cycle, or one candidate's fate would
    be decided by a different candidate's gates."""
    candidate(db, i=1, return_pct=5.0)                                   # accepted at NOW
    db.add(models.PipelineEvent(
        occurred_at=NOW + dt.timedelta(hours=6), token_address="Mint1", symbol="T",
        chain="solana", stage=pipeline.MARKET_QUALITY, passed=False, reason="later cycle",
    ))
    db.commit()

    report = build_counterfactual(db, horizon_minutes=60)
    assert report.accepted.n == 1
    assert report.gates == []


def test_versions_are_not_pooled(db):
    """A filter that changed mid-run is two filters, and pooling them
    describes neither."""
    for i in range(MIN_COHORT + 5):
        candidate(db, i=i, return_pct=2.0)
    # Committed BEFORE re-reading: the session is autoflush=False, so a
    # query issued here would not see the pending rows and would quietly
    # relabel nothing - the test would then pass for the wrong reason.
    db.commit()
    for row in db.query(models.ForwardReturn).limit(10).all():
        row.strategy_version = "v-older"
    db.commit()

    report = build_counterfactual(db, horizon_minutes=60, strategy_version="v-test")
    assert report.accepted.n == MIN_COHORT + 5 - 10
