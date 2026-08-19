"""Tests for the data-readiness report.

The failure mode worth guarding against is optimism: a progress readout
that says "ready" before the underlying tool will actually answer, or that
invents an ETA out of no rate. Either one converts an honest "not yet"
back into a number someone acts on.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.readiness import Requirement, build_readiness
from app.analysis.early_calibration import MIN_BUCKET_SAMPLE


def _row(db, *, score=None, early=None, ret=None, features=None,
         horizon=60, observed=None):
    db.add(models.ForwardReturn(
        pipeline_event_id=1, token_address="M", symbol="M",
        observed_at=observed or dt.datetime.now(dt.timezone.utc),
        score=score, price_at_signal=0.01, horizon_minutes=horizon,
        due_at=dt.datetime.now(dt.timezone.utc),
        return_pct=ret, early_score=early, early_features=features,
    ))


def _req(name):
    def pick(report):
        return next(r for r in report.requirements if r.question == name)
    return pick


def test_an_empty_database_says_start_the_bot_not_something_is_broken(clean_db):
    report = build_readiness(clean_db)
    assert report.scored_candidates == 0
    assert "Start the bot in paper mode" in report.headline()
    assert all(not r.ready for r in report.requirements)


def test_no_eta_is_invented_before_a_rate_can_be_measured(clean_db):
    """An ETA extrapolated from nothing gets remembered as a date."""
    _row(clean_db, score=70.0, ret=5.0)
    clean_db.commit()

    report = build_readiness(clean_db)
    assert report.candidates_per_hour is None
    assert "unknown" in _req("technical calibration")(report).eta(None)


def test_the_rate_comes_from_the_observed_span(clean_db):
    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i in range(20):
        db_time = base + dt.timedelta(hours=i)
        clean_db.add(models.ForwardReturn(
            pipeline_event_id=i, token_address="M", symbol="M",
            observed_at=db_time, score=70.0, price_at_signal=0.01,
            horizon_minutes=60, due_at=db_time, return_pct=1.0,
        ))
    clean_db.commit()

    report = build_readiness(clean_db)
    # 20 distinct candidates over a 19-hour span
    assert report.hours_running == pytest.approx(19.0)
    assert report.candidates_per_hour == pytest.approx(20 / 19)


def test_one_full_bucket_is_not_readiness(clean_db):
    """Calibration compares a top bucket against a bottom one.

    With every row in a single bucket the tool still says INSUFFICIENT
    DATA, so a readiness report that said READY would be claiming an
    answer exists that the tool refuses to give. Counting the total across
    buckets fails the same way one step earlier.
    """
    for _ in range(MIN_BUCKET_SAMPLE + 10):
        _row(clean_db, early=42.0, ret=1.0)     # all in one bucket
    clean_db.commit()

    early = _req("early calibration")(build_readiness(clean_db))
    assert early.have == 0
    assert not early.ready

    from app.analysis.early_calibration import build_early_calibration
    assert "INSUFFICIENT DATA" in build_early_calibration(
        clean_db, horizon_minutes=60
    ).verdict(), "readiness and the tool it reports on must agree"


def test_readiness_arrives_exactly_when_the_tool_starts_answering(clean_db):
    """The number that decides it is the second-fullest bucket, because
    that is the one whose arrival at the floor unblocks the comparison."""
    from app.analysis.early_calibration import build_early_calibration

    for _ in range(MIN_BUCKET_SAMPLE + 10):
        _row(clean_db, early=42.0, ret=-2.0)
    for _ in range(MIN_BUCKET_SAMPLE - 1):
        _row(clean_db, early=85.0, ret=9.0)
    clean_db.commit()

    early = _req("early calibration")(build_readiness(clean_db))
    assert early.have == MIN_BUCKET_SAMPLE - 1
    assert not early.ready
    assert "INSUFFICIENT DATA" in build_early_calibration(clean_db, horizon_minutes=60).verdict()

    _row(clean_db, early=85.0, ret=9.0)     # the row that tips it
    clean_db.commit()

    early = _req("early calibration")(build_readiness(clean_db))
    assert early.ready
    assert "INSUFFICIENT DATA" not in build_early_calibration(
        clean_db, horizon_minutes=60
    ).verdict(), "readiness said ready while the tool was still silent"


def test_a_pending_row_does_not_count_as_measured(clean_db):
    """An unresolved horizon is not an outcome. Counting it would show
    readiness that arrives hours before the data does."""
    for _ in range(MIN_BUCKET_SAMPLE + 5):
        _row(clean_db, early=72.0, ret=None)    # scheduled, not resolved
    clean_db.commit()

    assert _req("early calibration")(build_readiness(clean_db)).have == 0


def test_ablation_readiness_counts_only_rows_that_stored_features(clean_db):
    for _ in range(30):
        _row(clean_db, early=72.0, ret=4.0, features={"relative_volume": {
            "name": "relative_volume", "value": 2.0, "available": True,
            "detail": "", "source": "candles"}})
    for _ in range(30):
        _row(clean_db, early=72.0, ret=4.0, features=None)
    clean_db.commit()

    assert _req("early ablation")(build_readiness(clean_db)).have == 30


def test_the_ablation_count_matches_what_the_ablation_will_actually_load(clean_db):
    """Readiness counts rows in SQL; the ablation parses them in Python.

    They have to agree, or the report promises the ablation a sample it
    will not get. This drifted once already: a Python `None` written to a
    JSON column becomes the JSON value `null`, not SQL NULL, so
    `early_features IS NOT NULL` matched every row and the count was
    silently the row count.
    """
    from app.research.early_ablation import load_samples

    payload = {"relative_volume": {
        "name": "relative_volume", "value": 2.0, "available": True,
        "detail": "", "source": "candles"}}
    for _ in range(30):
        _row(clean_db, early=72.0, ret=4.0, features=payload)
    for _ in range(30):
        _row(clean_db, early=72.0, ret=4.0, features=None)
    for _ in range(5):
        _row(clean_db, early=72.0, ret=4.0, features={})
    clean_db.commit()

    reported = _req("early ablation")(build_readiness(clean_db)).have
    actual = len(load_samples(clean_db, horizon_minutes=60))
    assert reported == actual == 30


def test_a_met_requirement_reads_ready_rather_than_an_eta():
    req = Requirement("x", "cmd", have=50, need=30, unit="rows")
    assert req.ready
    assert req.eta(hours_running=10.0) == "ready"
    assert req.remaining == 0
    assert req.progress == 1.0


def test_each_row_extrapolates_its_own_rate_not_a_shared_one():
    """Quantities accrue at speeds that differ by orders of magnitude.

    A shared candidate rate would put "100 closed positions" a few hours
    out when it is really weeks - the kind of optimism that gets someone
    to stop waiting and start tuning.
    """
    fast = Requirement("fast", "c", have=90, need=100, unit="r")     # 9/hour
    slow = Requirement("slow", "c", have=2, need=100, unit="r")      # 0.2/hour

    assert "1h at the current rate" in fast.eta(hours_running=10.0)   # 9/hour, 10 to go
    assert "days" in slow.eta(hours_running=10.0)                     # 0.2/hour, 98 to go


def test_a_row_still_at_zero_gets_no_eta():
    """There is no rate to extrapolate from nothing, however long the bot
    has been up."""
    req = Requirement("x", "c", have=0, need=30, unit="r")
    assert "unknown" in req.eta(hours_running=500.0)


def test_progress_never_exceeds_one_so_the_bar_cannot_overflow():
    assert Requirement("x", "c", have=1000, need=30, unit="r").progress == 1.0


def test_an_eta_is_reported_in_days_once_it_passes_two():
    # 100 rows in 10h = 10/hour; 1000 more is 100 hours
    assert "days" in Requirement("x", "c", have=100, need=1100, unit="r").eta(10.0)
    # 100 rows in 10h; 10 more is one hour
    assert "h at the current rate" in Requirement("x", "c", have=100, need=110, unit="r").eta(10.0)


def test_the_report_never_suggests_lowering_a_threshold(clean_db):
    """The floors exist because below them the answers are noise. A tool
    that offered a shortcut past them would be the most damaging feature
    in the repository."""
    text = build_readiness(clean_db).table().lower()
    for phrase in ("lower the", "reduce the threshold", "min_bucket_sample=", "instead of 30"):
        assert phrase not in text
    assert "not negotiable" in text
