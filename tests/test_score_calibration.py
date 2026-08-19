"""Tests for score distribution (Phase 3) and score calibration (Phase 4).

The calibration tests matter most. They pin down that the table can return
a NEGATIVE verdict - that it is capable of reporting "the score does not
predict anything" rather than only ever confirming the engine.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.calibration import (
    ALL_BUCKETS,
    MIN_BUCKET_SAMPLE,
    bucket_label,
    build_calibration,
    round_trip_cost_pct,
)
from app.analysis.forward_returns import GIVE_UP_AFTER_HOURS, coverage, resolve_due, schedule
from app.analysis.score_distribution import (
    MIN_SAMPLE_FOR_A_DISTRIBUTION,
    build_score_distribution,
    describe_scores,
)
from app.database import SessionLocal
from app import pipeline

NOW = dt.datetime.now(dt.timezone.utc)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def clean_db():
    def wipe(session):
        for model in (models.ForwardReturn, models.PipelineEvent):
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


# ===========================================================================
# PHASE 3: score distribution
# ===========================================================================

def test_the_distribution_describes_the_scores_it_is_given():
    scores = [float(x) for x in range(40, 90)]     # 40..89, uniform
    d = describe_scores(scores)
    assert d.sample_size == 50
    assert d.minimum == 40 and d.maximum == 89
    assert d.mean == pytest.approx(64.5)
    assert d.median == pytest.approx(64.5)
    assert d.reliable is True


def test_survival_rates_say_what_a_threshold_would_cost_in_candidates():
    """The concrete number behind 'about 3% reach 75, about 26% reach 65'."""
    scores = [float(x) for x in range(40, 90)]
    d = describe_scores(scores)
    # 65..89 is 25 of 50.
    assert d.share_reaching(65) == pytest.approx(0.5)
    # 75..89 is 15 of 50.
    assert d.share_reaching(75) == pytest.approx(0.3)
    assert d.share_reaching(75) < d.share_reaching(65)


def test_a_small_sample_is_flagged_rather_than_presented_as_the_engine():
    d = describe_scores([61.0, 72.0, 55.0])
    assert d.reliable is False
    assert any("arithmetic, not a description" in w for w in d.warnings)


def test_no_scores_yields_an_empty_distribution_not_a_crash():
    d = describe_scores([])
    assert d.sample_size == 0
    assert d.mean is None
    assert d.reliable is False


def test_the_histogram_buckets_every_score():
    scores = [12.0, 47.0, 47.9, 66.0, 99.9]
    d = describe_scores(scores)
    assert sum(count for _label, count in d.histogram) == len(scores)


def test_rejected_scores_are_included_in_the_distribution(clean_db):
    """A distribution over survivors only would put every observation above
    the threshold, which is exactly the measurement that cannot judge it."""
    for i, score in enumerate([41.0, 52.0, 63.0, 78.0]):
        pipeline.record(
            clean_db, stage=pipeline.TECHNICAL_SCORE, symbol=f"T{i}",
            token_address=f"Mint{i}", passed=score >= 65, score=score,
            detail={"reliable": True},
        )
    clean_db.commit()

    d = build_score_distribution(clean_db)
    assert d.sample_size == 4
    assert d.minimum == 41.0, "the rejected 41 must be in the distribution"


def test_unreliable_scores_are_excluded_but_counted(clean_db):
    """A score the engine flagged as built on missing data sits near a
    neutral 50 and would pull the whole distribution toward the middle."""
    pipeline.record(clean_db, stage=pipeline.TECHNICAL_SCORE, symbol="A",
                    token_address="M1", passed=False, score=50.0, detail={"reliable": False})
    pipeline.record(clean_db, stage=pipeline.TECHNICAL_SCORE, symbol="B",
                    token_address="M2", passed=True, score=88.0, detail={"reliable": True})
    clean_db.commit()

    d = build_score_distribution(clean_db)
    assert d.sample_size == 1
    assert d.unreliable_count == 1
    assert any("excluded as unreliable" in w for w in d.warnings)

    both = build_score_distribution(clean_db, include_unreliable=True)
    assert both.sample_size == 2


# ===========================================================================
# PHASE 4: calibration
# ===========================================================================

def _forward(db, score, return_pct, *, horizon=60, measured=True, due_hours_ago=1.0):
    row = models.ForwardReturn(
        pipeline_event_id=0, token_address=f"Mint{score}-{return_pct}", symbol="X",
        observed_at=NOW - dt.timedelta(hours=due_hours_ago + horizon / 60),
        score=score, price_at_signal=1.0, horizon_minutes=horizon,
        due_at=NOW - dt.timedelta(hours=due_hours_ago),
        price_at_horizon=(1.0 * (1 + return_pct / 100)) if measured else None,
        return_pct=return_pct if measured else None,
        filled_at=NOW if measured else None,
        strategy_version="v-test",
    )
    db.add(row)
    return row


def test_bucket_labels_cover_the_thresholds_under_discussion():
    assert bucket_label(41) == "<55"
    assert bucket_label(63) == "60-65"
    assert bucket_label(66) == "65-70"
    assert bucket_label(74.9) == "70-75"
    assert bucket_label(91) == "80+"
    assert set(bucket_label(s) for s in (10, 57, 62, 67, 72, 77, 95)) <= set(ALL_BUCKETS)


def test_a_predictive_score_reads_as_monotonic(clean_db):
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 58.0, -3.0)
        _forward(clean_db, 68.0, 1.0)
        _forward(clean_db, 82.0, 6.0)
    clean_db.commit()

    table = build_calibration(clean_db, horizon_minutes=60)
    assert table.monotonic is True
    assert "separates outcomes" in table.verdict()


def test_a_useless_score_is_reported_as_useless(clean_db):
    """The verdict this module exists to be capable of producing. If a
    higher score does not precede a better outcome, saying so is the
    correct output - not a bug to be tuned away."""
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 58.0, 5.0)
        _forward(clean_db, 82.0, -4.0)
    clean_db.commit()

    table = build_calibration(clean_db, horizon_minutes=60)
    assert table.monotonic is False
    verdict = table.verdict()
    assert "does NOT predict" in verdict
    assert "without improving trade quality" in verdict


def test_a_non_monotonic_but_positive_gradient_is_called_weak(clean_db):
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 58.0, -2.0)
        _forward(clean_db, 68.0, 6.0)     # middle bucket out of order
        _forward(clean_db, 82.0, 3.0)
    clean_db.commit()

    table = build_calibration(clean_db, horizon_minutes=60)
    assert table.monotonic is False
    assert "weak ranking" in table.verdict()


def test_thin_buckets_cannot_produce_a_verdict(clean_db):
    """Two data points are not a calibration curve, and the module must say
    so rather than report a direction."""
    _forward(clean_db, 58.0, -5.0)
    _forward(clean_db, 82.0, 12.0)
    clean_db.commit()

    table = build_calibration(clean_db, horizon_minutes=60)
    assert table.monotonic is None
    assert "INSUFFICIENT DATA" in table.verdict()
    assert table.measurable_buckets == []


def test_costs_are_subtracted_and_are_not_free(clean_db):
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 68.0, 1.0)
    clean_db.commit()

    bucket = next(
        b for b in build_calibration(clean_db, horizon_minutes=60).buckets if b.bucket == "65-70"
    )
    assert bucket.mean_return_pct == pytest.approx(1.0)
    assert bucket.mean_net_of_costs_pct < bucket.mean_return_pct
    assert round_trip_cost_pct() > 0


def test_a_small_edge_can_be_negative_after_costs(clean_db):
    """The number that decides whether a strategy is real."""
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 68.0, 0.2)
    clean_db.commit()

    bucket = next(
        b for b in build_calibration(clean_db, horizon_minutes=60).buckets if b.bucket == "65-70"
    )
    assert bucket.mean_return_pct > 0
    assert bucket.mean_net_of_costs_pct < 0


def test_an_unmeasured_horizon_is_not_counted_as_a_zero_return(clean_db):
    """A token that stopped trading did not return 0%. Conflating the two
    is how a survivorship-biased dataset gets built by accident."""
    for _ in range(10):
        _forward(clean_db, 68.0, 4.0)
    for _ in range(10):
        _forward(clean_db, 68.0, 0.0, measured=False)
    clean_db.commit()

    bucket = next(
        b for b in build_calibration(clean_db, horizon_minutes=60).buckets if b.bucket == "65-70"
    )
    assert bucket.sample_size == 10
    assert bucket.unmeasured == 10
    assert bucket.mean_return_pct == pytest.approx(4.0), "unmeasured rows must not drag the mean down"
    assert bucket.coverage_pct == pytest.approx(50.0)


def test_a_horizon_still_in_the_future_is_pending_not_missing(clean_db):
    _forward(clean_db, 68.0, 0.0, measured=False, due_hours_ago=-5.0)   # due in 5h
    clean_db.commit()

    bucket = next(
        b for b in build_calibration(clean_db, horizon_minutes=60).buckets if b.bucket == "65-70"
    )
    assert bucket.unmeasured == 0, "a horizon that has not elapsed is not a gap"


# ===========================================================================
# forward-return collection
# ===========================================================================

def test_scheduling_creates_one_row_per_horizon(clean_db):
    created = schedule(
        clean_db, pipeline_event_id=1, token_address="Mint1", symbol="X",
        score=71.0, price_at_signal=0.004,
    )
    clean_db.commit()
    assert created == 7
    assert clean_db.query(models.ForwardReturn).count() == 7
    assert {r.horizon_minutes for r in clean_db.query(models.ForwardReturn).all()} == {
        15, 30, 60, 120, 240, 480, 1440
    }


def test_scheduling_refuses_a_corrupt_signal_price(clean_db):
    """Every return divides by it. A zero price is bad data, not a free
    option."""
    assert schedule(clean_db, pipeline_event_id=1, token_address="M", symbol="X",
                    score=71.0, price_at_signal=0.0) == 0
    assert schedule(clean_db, pipeline_event_id=1, token_address="", symbol="X",
                    score=71.0, price_at_signal=1.0) == 0


pytestmark = pytest.mark.anyio


async def test_resolving_computes_the_return_from_the_live_price(clean_db, monkeypatch):
    from app.analysis import forward_returns as fr

    async def price(_addr):
        return 0.006
    monkeypatch.setattr(fr.price_feed, "get_price_usd", price)

    row = models.ForwardReturn(
        pipeline_event_id=1, token_address="Mint1", symbol="X",
        observed_at=NOW - dt.timedelta(hours=2), score=70.0, price_at_signal=0.004,
        horizon_minutes=60, due_at=NOW - dt.timedelta(minutes=30),
    )
    clean_db.add(row)
    clean_db.commit()

    summary = await resolve_due(clean_db, now=NOW)
    clean_db.commit()

    assert summary["resolved"] == 1
    assert row.return_pct == pytest.approx(50.0)   # 0.004 -> 0.006
    assert row.filled_at is not None


async def test_one_price_lookup_serves_every_horizon_for_a_mint(clean_db, monkeypatch):
    """Seven horizons coming due together must not cost seven API calls."""
    from app.analysis import forward_returns as fr

    calls = []

    async def price(addr):
        calls.append(addr)
        return 0.005
    monkeypatch.setattr(fr.price_feed, "get_price_usd", price)

    for minutes in (15, 30, 60):
        clean_db.add(models.ForwardReturn(
            pipeline_event_id=1, token_address="Mint1", symbol="X",
            observed_at=NOW - dt.timedelta(hours=3), score=70.0, price_at_signal=0.004,
            horizon_minutes=minutes, due_at=NOW - dt.timedelta(minutes=10),
        ))
    clean_db.commit()

    await resolve_due(clean_db, now=NOW)
    assert len(calls) == 1


async def test_a_missing_price_leaves_the_row_pending_rather_than_guessing(clean_db, monkeypatch):
    from app.analysis import forward_returns as fr

    async def no_price(_addr):
        return None
    monkeypatch.setattr(fr.price_feed, "get_price_usd", no_price)

    row = models.ForwardReturn(
        pipeline_event_id=1, token_address="Mint1", symbol="X",
        observed_at=NOW - dt.timedelta(hours=2), score=70.0, price_at_signal=0.004,
        horizon_minutes=60, due_at=NOW - dt.timedelta(minutes=30),
    )
    clean_db.add(row)
    clean_db.commit()

    summary = await resolve_due(clean_db, now=NOW)
    assert summary["unavailable"] == 1
    assert row.return_pct is None
    assert row.filled_at is None, "still pending - it may resolve on a later pass"


async def test_a_long_dead_token_is_closed_out_with_a_reason(clean_db, monkeypatch):
    """Kept in the table as explicitly unmeasurable. Dropping it would bias
    the dataset toward tokens that stayed alive."""
    from app.analysis import forward_returns as fr

    async def no_price(_addr):
        return None
    monkeypatch.setattr(fr.price_feed, "get_price_usd", no_price)

    row = models.ForwardReturn(
        pipeline_event_id=1, token_address="DeadMint", symbol="RIP",
        observed_at=NOW - dt.timedelta(days=3), score=70.0, price_at_signal=0.004,
        horizon_minutes=60, due_at=NOW - dt.timedelta(hours=GIVE_UP_AFTER_HOURS + 5),
    )
    clean_db.add(row)
    clean_db.commit()

    summary = await resolve_due(clean_db, now=NOW)
    clean_db.commit()

    assert summary["abandoned"] == 1
    assert row.filled_at is not None
    assert row.return_pct is None
    assert "unmeasurable rather than dropped or zero-filled" in row.failure_reason


def test_coverage_reports_how_complete_the_dataset_is(clean_db):
    """A calibration table built on 20% coverage is describing whichever
    tokens happened to stay liquid."""
    clean_db.add(models.ForwardReturn(
        pipeline_event_id=1, token_address="M", symbol="X", observed_at=NOW,
        score=70.0, price_at_signal=1.0, horizon_minutes=60, due_at=NOW,
        return_pct=5.0, filled_at=NOW,
    ))
    clean_db.add(models.ForwardReturn(
        pipeline_event_id=1, token_address="M", symbol="X", observed_at=NOW,
        score=70.0, price_at_signal=1.0, horizon_minutes=120,
        due_at=NOW + dt.timedelta(hours=1),
    ))
    clean_db.commit()

    stats = coverage(clean_db)
    assert stats["total"] == 2
    assert stats["resolved"] == 1
    assert stats["pending"] == 1
    assert stats["coverage_pct"] == pytest.approx(50.0)
