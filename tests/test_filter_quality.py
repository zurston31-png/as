"""Per-check pre-screen quality (app/analysis/filter_quality.py).

The pre-screen is five thresholds under one label. A stage-level verdict
saying "the pre-screen rejects worse tokens on average" is compatible with
liquidity doing all the work while the age floor discards winners, so these
tests are mostly about the split staying honest - and about the report
refusing to answer when the sample cannot support an answer.
"""
import datetime as dt

import pytest

from app import models, pipeline
from app.analysis.filter_quality import MIN_ARM_SAMPLE, build_filter_quality
from app.database import SessionLocal

NOW = dt.datetime.now(dt.timezone.utc)
HORIZON = 240


@pytest.fixture()
def db():
    def wipe(session):
        for model in (models.ForwardReturn, models.PipelineEvent):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    session = SessionLocal()
    wipe(session)
    try:
        yield session
    finally:
        wipe(session)
        session.close()


def _candidate(db, mint, *, liquidity_passed, return_pct, volume_passed=True, minutes_ago=300):
    """One prescreen event plus the outcome the mint went on to have."""
    occurred = NOW - dt.timedelta(minutes=minutes_ago)
    event = models.PipelineEvent(
        occurred_at=occurred, token_address=mint, symbol=mint[:6], chain="solana",
        stage=pipeline.PRESCREEN, passed=liquidity_passed and volume_passed,
        reason="test", detail={"checks": [
            {"name": "liquidity", "passed": liquidity_passed,
             "reason": "x", "value": 50_000.0 if liquidity_passed else 900.0,
             "threshold": 35_000.0},
            {"name": "volume", "passed": volume_passed,
             "reason": "x", "value": 80_000.0 if volume_passed else 100.0,
             "threshold": 50_000.0},
        ]},
    )
    db.add(event)
    db.flush()
    db.add(models.ForwardReturn(
        pipeline_event_id=event.id, token_address=mint, symbol=mint[:6],
        observed_at=occurred, score=None, price_at_signal=1.0,
        horizon_minutes=HORIZON, due_at=occurred + dt.timedelta(minutes=HORIZON),
        return_pct=return_pct, filled_at=NOW,
    ))


# ---------------------------------------------------------------------------
# sample-size protection - the report must refuse to answer on thin data
# ---------------------------------------------------------------------------

def test_a_thin_sample_reports_insufficient_data_rather_than_a_verdict(db):
    """The whole failure mode this guards against: a confident verdict from
    nine observations, whose answer is always "you could be trading more"."""
    for i in range(5):
        _candidate(db, f"ThinPass{i}", liquidity_passed=True, return_pct=5.0)
        _candidate(db, f"ThinFail{i}", liquidity_passed=False, return_pct=50.0)
    db.commit()

    report = build_filter_quality(db, horizon_minutes=HORIZON)
    liquidity = next(c for c in report.checks if c.check_name == "liquidity")

    assert not liquidity.measured
    assert "INSUFFICIENT DATA" in liquidity.verdict()
    # Even though the failed arm looks dramatically better, it is not flagged.
    assert liquidity not in report.flagged
    assert "INSUFFICIENT DATA" in report.summary()


def test_one_arm_over_the_floor_is_still_insufficient(db):
    """Both sides need the sample. A well-populated kept arm against four
    rejects cannot tell you what the rejects were worth."""
    for i in range(MIN_ARM_SAMPLE + 5):
        _candidate(db, f"Pass{i}", liquidity_passed=True, return_pct=2.0)
    for i in range(4):
        _candidate(db, f"Fail{i}", liquidity_passed=False, return_pct=80.0)
    db.commit()

    liquidity = next(
        c for c in build_filter_quality(db, horizon_minutes=HORIZON).checks
        if c.check_name == "liquidity"
    )
    assert liquidity.passed_arm.usable and not liquidity.failed_arm.usable
    assert not liquidity.measured


# ---------------------------------------------------------------------------
# the split itself
# ---------------------------------------------------------------------------

def test_each_check_is_judged_separately(db):
    """The reason this module exists. Liquidity keeps the better half while
    volume discards it; a stage-level verdict would average the two into
    'the pre-screen is fine' and hide the volume problem."""
    for i in range(MIN_ARM_SAMPLE):
        # liquidity: kept tokens do better
        _candidate(db, f"LiqGood{i}", liquidity_passed=True, volume_passed=True, return_pct=20.0)
        _candidate(db, f"LiqBad{i}", liquidity_passed=False, volume_passed=True, return_pct=-10.0)
        # volume: rejected tokens do better
        _candidate(db, f"VolBad{i}", liquidity_passed=True, volume_passed=False, return_pct=60.0)
    db.commit()

    report = build_filter_quality(db, horizon_minutes=HORIZON)
    liquidity = next(c for c in report.checks if c.check_name == "liquidity")
    volume = next(c for c in report.checks if c.check_name == "volume")

    assert "keeping the better half" in liquidity.verdict()
    assert "REJECTING THE BETTER HALF" in volume.verdict()
    assert [c.check_name for c in report.flagged] == ["volume"]


def test_returns_are_charged_the_round_trip_the_entry_would_have_paid(db):
    """A gross comparison flatters every rejected token, because a rejected
    token never paid the cost that would have eaten the difference."""
    from app.analysis.calibration import round_trip_cost_pct

    for i in range(MIN_ARM_SAMPLE):
        _candidate(db, f"Net{i}", liquidity_passed=True, return_pct=10.0)
    db.commit()

    liquidity = next(
        c for c in build_filter_quality(db, horizon_minutes=HORIZON).checks
        if c.check_name == "liquidity"
    )
    assert liquidity.passed_arm.mean == pytest.approx(10.0 - round_trip_cost_pct())


def test_the_gain_and_loss_bands_count_what_they_say(db):
    for i in range(MIN_ARM_SAMPLE):
        # Half moon-shots, half stopped out.
        _candidate(db, f"Band{i}", liquidity_passed=True,
                   return_pct=60.0 if i % 2 == 0 else -40.0)
    db.commit()

    arm = next(
        c for c in build_filter_quality(db, horizon_minutes=HORIZON).checks
        if c.check_name == "liquidity"
    ).passed_arm
    assert arm.share_gaining(50.0) == pytest.approx(50.0, abs=2)
    assert arm.share_losing() == pytest.approx(50.0, abs=2)
    assert arm.share_gaining(10.0) >= arm.share_gaining(50.0)


def test_a_prescreen_with_no_measured_outcome_is_counted_but_not_scored(db):
    """Most rejects are never sampled, so most prescreen events have no
    outcome. They must show in the denominator of "how many did we see"
    without inventing a return."""
    occurred = NOW - dt.timedelta(minutes=300)
    db.add(models.PipelineEvent(
        occurred_at=occurred, token_address="NoOutcomeMint", symbol="NOM", chain="solana",
        stage=pipeline.PRESCREEN, passed=False, reason="thin",
        detail={"checks": [{"name": "liquidity", "passed": False, "reason": "x",
                            "value": 100.0, "threshold": 35_000.0}]},
    ))
    db.commit()

    report = build_filter_quality(db, horizon_minutes=HORIZON)
    assert report.events_seen == 1
    assert report.events_with_outcome == 0
    assert not report.checks, "no outcome means no arm to put it in"


def test_an_outcome_for_a_different_mint_is_not_borrowed(db):
    """Matching is by mint. Borrowing a neighbour's outcome would be
    fabricating the measurement this whole report rests on."""
    occurred = NOW - dt.timedelta(minutes=300)
    event = models.PipelineEvent(
        occurred_at=occurred, token_address="MintA", symbol="A", chain="solana",
        stage=pipeline.PRESCREEN, passed=False, reason="thin",
        detail={"checks": [{"name": "liquidity", "passed": False, "reason": "x",
                            "value": 100.0, "threshold": 35_000.0}]},
    )
    db.add(event)
    db.flush()
    db.add(models.ForwardReturn(
        pipeline_event_id=event.id, token_address="MintB", symbol="B",
        observed_at=occurred, score=None, price_at_signal=1.0,
        horizon_minutes=HORIZON, due_at=occurred + dt.timedelta(minutes=HORIZON),
        return_pct=99.0, filled_at=NOW,
    ))
    db.commit()

    assert build_filter_quality(db, horizon_minutes=HORIZON).events_with_outcome == 0


def test_an_outcome_from_a_later_scan_cycle_is_not_matched(db):
    """The same mint is re-examined every SCANNER_RECHECK_MINUTES. An
    outcome recorded hours later belongs to a different evaluation."""
    occurred = NOW - dt.timedelta(hours=6)
    event = models.PipelineEvent(
        occurred_at=occurred, token_address="MintC", symbol="C", chain="solana",
        stage=pipeline.PRESCREEN, passed=False, reason="thin",
        detail={"checks": [{"name": "liquidity", "passed": False, "reason": "x",
                            "value": 100.0, "threshold": 35_000.0}]},
    )
    db.add(event)
    db.flush()
    db.add(models.ForwardReturn(
        pipeline_event_id=event.id, token_address="MintC", symbol="C",
        observed_at=occurred + dt.timedelta(hours=3),   # a later cycle
        score=None, price_at_signal=1.0,
        horizon_minutes=HORIZON, due_at=NOW, return_pct=99.0, filled_at=NOW,
    ))
    db.commit()

    assert build_filter_quality(db, horizon_minutes=HORIZON).events_with_outcome == 0


def test_the_report_never_recommends_a_threshold(db):
    """It reports evidence. Turning that into "set liquidity to 12000" is a
    human decision, and a tool that made it would be optimising the one
    metric that always says 'trade more'."""
    for i in range(MIN_ARM_SAMPLE):
        _candidate(db, f"Rec{i}", liquidity_passed=True, return_pct=-5.0)
        _candidate(db, f"RecF{i}", liquidity_passed=False, return_pct=40.0)
    db.commit()

    liquidity = next(
        c for c in build_filter_quality(db, horizon_minutes=HORIZON).checks
        if c.check_name == "liquidity"
    )
    verdict = liquidity.verdict()
    assert "REJECTING THE BETTER HALF" in verdict
    assert "NOT permission to move it" in verdict
