"""Tests for app/pipeline.py and app/analysis/stage_funnel.py.

What matters here is that the log answers the question it exists for -
"why did 500 discovered tokens produce only 3 trades?" - and that it never
turns "we don't know" into a number.
"""
import datetime as dt

import pytest

from app import models, pipeline
from app.analysis.stage_funnel import build_stage_funnel
from app.database import SessionLocal


@pytest.fixture()
def clean_events():
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


def _emit(db, stage, passed, *, symbol="TESTCOIN", mint="Mint1", reason="", score=None, detail=None):
    return pipeline.record(
        db, stage=stage, symbol=symbol, token_address=mint,
        passed=passed, reason=reason, score=score, detail=detail,
    )


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------

def test_an_event_records_identity_stage_and_outcome(clean_events):
    _emit(clean_events, pipeline.DISCOVERED, True, reason="listed by dexscreener")
    clean_events.commit()

    row = clean_events.query(models.PipelineEvent).one()
    assert row.stage == "DISCOVERED"
    assert row.token_address == "Mint1"
    assert row.passed is True
    assert row.strategy_version, "every event must name the strategy that produced it"
    assert row.occurred_at is not None


def test_an_unknown_stage_is_refused_rather_than_recorded(clean_events):
    """A typo'd stage name would create a phantom funnel step that never
    converts, which reads as a broken pipeline rather than a broken call."""
    assert _emit(clean_events, "TYPO_STAGE", True) is None
    assert clean_events.query(models.PipelineEvent).count() == 0


def test_recording_never_raises_into_the_caller(clean_events, monkeypatch):
    """An audit gap costs research. An exception here would cost a trade."""
    def boom(*_a, **_k):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(clean_events, "add", boom)
    assert _emit(clean_events, pipeline.SECURITY, False) is None


def test_the_log_is_append_only(clean_events):
    """Two evaluations of one token produce two rows, not an overwrite -
    that is the whole difference from ScannedToken."""
    _emit(clean_events, pipeline.PRESCREEN, False, reason="liquidity too low")
    _emit(clean_events, pipeline.PRESCREEN, True, reason="passed")
    clean_events.commit()
    assert clean_events.query(models.PipelineEvent).count() == 2


def test_a_rejected_token_still_records_its_score(clean_events):
    """The point of the whole exercise: a dataset of only the setups that
    cleared the threshold cannot say whether the threshold is right."""
    _emit(clean_events, pipeline.TECHNICAL_SCORE, False, score=41.2, reason="below threshold")
    clean_events.commit()
    row = clean_events.query(models.PipelineEvent).one()
    assert row.passed is False
    assert row.score == pytest.approx(41.2)


# ---------------------------------------------------------------------------
# the funnel
# ---------------------------------------------------------------------------

def test_conversion_is_measured_against_what_reached_each_stage(clean_events):
    """'2% of discovered tokens got scored' and '80% of scored tokens
    passed' describe different systems. Only the second says whether the
    score threshold is the bottleneck."""
    for i in range(10):
        _emit(clean_events, pipeline.DISCOVERED, True, mint=f"M{i}")
    for i in range(5):
        _emit(clean_events, pipeline.PRESCREEN, i < 2, mint=f"M{i}")
    clean_events.commit()

    funnel = build_stage_funnel(clean_events, window_hours=None)
    discovered = funnel.stage("DISCOVERED")
    prescreen = funnel.stage("PRESCREEN")

    assert discovered.entered == 10 and discovered.pass_rate == 1.0
    assert prescreen.entered == 5, "denominator is what reached the stage"
    assert prescreen.passed == 2
    assert prescreen.pass_rate == pytest.approx(0.4)


def test_a_stage_nobody_reached_reports_no_data_not_zero_percent(clean_events):
    """0% would mean 'it rejected everything', which is a completely
    different instruction to whoever is tuning thresholds."""
    _emit(clean_events, pipeline.DISCOVERED, True)
    clean_events.commit()

    security = build_stage_funnel(clean_events, window_hours=None).stage("SECURITY")
    assert security.entered == 0
    assert security.reached is False
    assert security.pass_rate is None
    assert security.as_dict()["pass_rate_pct"] is None


def test_the_bottleneck_is_the_biggest_absolute_loss_not_the_worst_rate(clean_events):
    """A stage that rejects 100% of the four tokens that reached it is not
    the reason 500 became 3."""
    for i in range(100):
        _emit(clean_events, pipeline.PRESCREEN, i < 6, mint=f"M{i}")     # 94 rejected
    for i in range(4):
        _emit(clean_events, pipeline.SECURITY, False, mint=f"S{i}")      # 4 rejected, 100%
    clean_events.commit()

    funnel = build_stage_funnel(clean_events, window_hours=None)
    assert funnel.bottleneck.stage == "PRESCREEN"


def test_the_prescreen_breakdown_names_the_threshold_doing_the_work(clean_events):
    """'497 failed pre-screen' is useless. '480 failed liquidity' is a
    number you can act on."""
    for i in range(10):
        failed_liquidity = i < 8
        _emit(
            clean_events, pipeline.PRESCREEN, not failed_liquidity, mint=f"M{i}",
            detail={"checks": [
                {"name": "liquidity", "passed": not failed_liquidity},
                {"name": "volume", "passed": True},
                {"name": "age", "passed": i != 9},
            ]},
        )
    clean_events.commit()

    breakdown = build_stage_funnel(clean_events, window_hours=None).prescreen
    assert breakdown.evaluated == 10
    assert breakdown.failed_by_check["liquidity"] == 8
    assert breakdown.failed_by_check["age"] == 1
    assert breakdown.passed_by_check["volume"] == 10
    assert breakdown.pass_rate("liquidity") == pytest.approx(0.2)


def test_overlapping_failures_are_both_counted(clean_events):
    """A token failing three checks at once must show in all three, or
    every threshold looks more decisive than it is."""
    _emit(clean_events, pipeline.PRESCREEN, False, detail={"checks": [
        {"name": "liquidity", "passed": False},
        {"name": "volume", "passed": False},
        {"name": "age", "passed": False},
    ]})
    clean_events.commit()

    breakdown = build_stage_funnel(clean_events, window_hours=None).prescreen
    assert breakdown.failed_by_check == {"liquidity": 1, "volume": 1, "age": 1}


def test_explain_answers_the_question_directly(clean_events):
    for i in range(50):
        _emit(clean_events, pipeline.DISCOVERED, True, mint=f"M{i}")
    for i in range(50):
        _emit(clean_events, pipeline.PRESCREEN, i < 1, mint=f"M{i}")
    _emit(clean_events, pipeline.OPEN_POSITION, True, mint="M0")
    clean_events.commit()

    funnel = build_stage_funnel(clean_events, window_hours=None)
    text = funnel.explain()
    assert "50 tokens discovered" in text
    assert "PRESCREEN" in text
    assert funnel.overall_conversion == pytest.approx(1 / 50)


def test_an_empty_log_says_the_scanner_is_not_seeing_anything(clean_events):
    """A different diagnosis from 'the filters are too tight', and it
    points at a different fix."""
    funnel = build_stage_funnel(clean_events, window_hours=None)
    assert funnel.discovered == 0
    assert funnel.overall_conversion is None
    assert "not seeing candidates" in funnel.explain()


def test_rejection_reasons_group_by_category_not_by_exact_string(clean_events):
    """Reasons embed dollar amounts, so grouping raw strings would produce
    one category per event."""
    for amount in (12_431, 801, 9_004):
        _emit(clean_events, pipeline.PRESCREEN, False,
              reason=f"liquidity ${amount:,} below scanner minimum $35,000")
    clean_events.commit()

    reasons = build_stage_funnel(clean_events, window_hours=None).rejection_reasons
    assert len(reasons) == 1
    stage, _reason, count = reasons[0]
    assert stage == "PRESCREEN" and count == 3


def test_the_window_excludes_older_events(clean_events):
    old = pipeline.record(clean_events, stage=pipeline.DISCOVERED, symbol="OLD",
                          token_address="MintOld", passed=True)
    old.occurred_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)
    _emit(clean_events, pipeline.DISCOVERED, True, symbol="NEW", mint="MintNew")
    clean_events.commit()

    assert build_stage_funnel(clean_events, window_hours=24).discovered == 1
    assert build_stage_funnel(clean_events, window_hours=None).discovered == 2


def test_the_funnel_serialises(clean_events):
    import json

    _emit(clean_events, pipeline.DISCOVERED, True)
    clean_events.commit()
    json.dumps(build_stage_funnel(clean_events, window_hours=None).as_dict(), allow_nan=False)
