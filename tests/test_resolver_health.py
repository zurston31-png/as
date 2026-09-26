"""Tests for forward-return resolver health (app/analysis/resolver_health.py).

The invariants:

  H1  An empty dataset is IDLE, and says so - never HEALTHY. "Nothing has
      gone wrong" and "nothing has happened" are different facts.
  H2  Rows overdue but inside their lateness tolerance are BEHIND, not a
      failure: a horizon coming due between passes is how the worker
      normally operates.
  H3  Rows past tolerance are LOSING_DATA - those observations cannot be
      filled honestly any more and will be sealed.
  H4  Past tolerance with nothing ever resolved is STALLED, which needs a
      different fix from a worker that is merely slow.
  H5  "Late" is defined by the same function the resolver uses, so the
      health check and the resolver cannot disagree about which rows are
      lost.
  H6  Sealed rows are split by reason: a dead price feed is a fact about
      the market, lateness is a fact about this worker, and only one is
      fixable.
  H7  Lateness statistics are withheld below a sample floor.
  H8  The unmeasurable rate is computed over CLOSED rows, so a young
      dataset full of pending rows is not reported as mostly lost.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.forward_returns import lateness_tolerance_minutes
from app.analysis.resolver_health import (
    MIN_RESOLUTIONS_FOR_LATENESS,
    ResolverStatus,
    check_resolver_health,
)
from app.database import SessionLocal

NOW = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture()
def clean_db():
    def wipe(session):
        for row in session.query(models.ForwardReturn).all():
            session.delete(row)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _row(db, *, horizon=60, due_minutes_ago=None, resolved=None, sealed=None,
         elapsed=None, event=1):
    """One scheduled forward return.

    `due_minutes_ago` positive means already due. `resolved` sets a return,
    `sealed` sets a failure reason with no return.
    """
    observed = NOW - dt.timedelta(minutes=horizon + (due_minutes_ago or 0))
    due = observed + dt.timedelta(minutes=horizon)
    row = models.ForwardReturn(
        pipeline_event_id=event, token_address=f"Mint{event}", symbol="FRC",
        observed_at=observed, score=70.0, price_at_signal=0.01,
        horizon_minutes=horizon, due_at=due,
    )
    if resolved is not None:
        row.return_pct = resolved
        row.price_at_horizon = 0.011
        row.filled_at = NOW
        row.measured_at = NOW
        row.actual_elapsed_minutes = elapsed if elapsed is not None else horizon
    elif sealed is not None:
        row.filled_at = NOW
        row.failure_reason = sealed
    db.add(row)
    return row


# ---------------------------------------------------------------------------
# H1 - empty is not healthy
# ---------------------------------------------------------------------------

def test_an_empty_dataset_is_idle_not_healthy(clean_db):
    """H1. A green light over an empty table is the single most
    misleading thing a health panel can show."""
    health = check_resolver_health(clean_db, now=NOW)
    assert health.status is ResolverStatus.IDLE
    assert health.scheduled == 0
    assert "not a healthy one" in health.detail


def test_scheduled_but_nothing_due_yet_is_idle(clean_db):
    """Nothing to judge is not the same as judged and passed."""
    _row(clean_db, horizon=60, due_minutes_ago=-30)   # due in 30 minutes
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.status is ResolverStatus.IDLE
    assert health.pending == 1
    assert health.overdue == 0


# ---------------------------------------------------------------------------
# H2/H3/H4/H5 - the backlog ladder
# ---------------------------------------------------------------------------

def test_everything_resolved_and_nothing_due_is_healthy(clean_db):
    for i in range(3):
        _row(clean_db, resolved=4.0, event=i)
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.status is ResolverStatus.HEALTHY
    assert health.resolved == 3


def test_overdue_inside_tolerance_is_behind_not_broken(clean_db):
    """H2. A 60-minute horizon tolerates 15 minutes of lateness, so a row
    five minutes overdue is the batch about to run, not a fault."""
    _row(clean_db, horizon=60, due_minutes_ago=5)
    _row(clean_db, resolved=3.0, event=2)
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.status is ResolverStatus.BEHIND
    assert health.overdue == 1
    assert health.overdue_past_tolerance == 0
    assert health.oldest_overdue_minutes == pytest.approx(5.0)


def test_overdue_past_tolerance_is_losing_data(clean_db):
    """H3. Past this point the row cannot be filled as the horizon it
    claims, so it will be sealed - the observation is gone."""
    _row(clean_db, horizon=60, due_minutes_ago=30)   # tolerance is 15
    _row(clean_db, resolved=3.0, event=2)
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.status is ResolverStatus.LOSING_DATA
    assert health.overdue_past_tolerance == 1
    assert "permanently" in health.detail or "sealed unmeasurable" in health.detail


def test_past_tolerance_with_nothing_ever_resolved_is_stalled(clean_db):
    """H4. Running-but-slow and not-running-at-all need opposite fixes -
    one is a batch-size problem, the other is a dead worker."""
    _row(clean_db, horizon=60, due_minutes_ago=600)
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.status is ResolverStatus.STALLED
    assert "not running" in health.detail


def test_the_tolerance_comes_from_the_resolver_not_a_local_constant(clean_db):
    """H5. Restating the rule would let the health check and the resolver
    disagree about which rows were lost. A short horizon has a
    proportionally short tolerance, and this proves the check honours
    that rather than using one flat window."""
    assert lateness_tolerance_minutes(5) == pytest.approx(1.25)

    # 3 minutes overdue: inside tolerance for a 60m horizon, well outside
    # it for a 5m one.
    _row(clean_db, horizon=5, due_minutes_ago=3, event=1)
    clean_db.commit()
    assert check_resolver_health(clean_db, now=NOW).overdue_past_tolerance == 1

    for row in clean_db.query(models.ForwardReturn).all():
        clean_db.delete(row)
    _row(clean_db, horizon=60, due_minutes_ago=3, event=2)
    clean_db.commit()
    assert check_resolver_health(clean_db, now=NOW).overdue_past_tolerance == 0


def test_the_oldest_overdue_row_is_the_one_reported(clean_db):
    """A hundred rows one minute overdue is a batch about to run; one row
    six hours overdue is a fault. The maximum distinguishes them where a
    count cannot."""
    for i in range(5):
        _row(clean_db, horizon=60, due_minutes_ago=1, event=i)
    _row(clean_db, horizon=60, due_minutes_ago=360, event=99)
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.overdue == 6
    assert health.oldest_overdue_minutes == pytest.approx(360.0)


# ---------------------------------------------------------------------------
# H6 - why rows were sealed
# ---------------------------------------------------------------------------

def test_sealed_rows_are_split_by_reason(clean_db):
    """H6. Pooling them would hide the only one that can be fixed."""
    _row(clean_db, sealed="no price available at the horizon", event=1)
    _row(clean_db, sealed="no price available at the horizon", event=2)
    _row(clean_db, sealed="resolved too late to be this horizon", event=3)
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.sealed_unmeasurable == 3
    assert health.sealed_by_reason["no price available at the horizon"] == 2
    assert health.sealed_by_reason["resolved too late to be this horizon"] == 1


def test_a_sealed_row_with_no_reason_says_the_reason_is_missing(clean_db):
    """Filing it under a neighbouring reason would invent an explanation
    that was never recorded."""
    row = _row(clean_db, sealed="placeholder", event=1)
    row.failure_reason = None
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.sealed_by_reason == {"reason not recorded": 1}


# ---------------------------------------------------------------------------
# H7 - lateness statistics
# ---------------------------------------------------------------------------

def test_lateness_is_withheld_below_the_sample_floor(clean_db):
    """H7. Three slow rows and a slow worker look identical at n=3."""
    for i in range(MIN_RESOLUTIONS_FOR_LATENESS - 1):
        _row(clean_db, resolved=2.0, elapsed=70.0, event=i)
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.median_lateness_minutes is None
    assert health.lateness_samples == MIN_RESOLUTIONS_FOR_LATENESS - 1


def test_lateness_is_reported_once_there_is_enough_of_it(clean_db):
    """A worker resolving 60-minute horizons at 70 minutes is inside
    tolerance and heading out of it - visible here long before coverage
    starts dropping."""
    for i in range(MIN_RESOLUTIONS_FOR_LATENESS):
        _row(clean_db, resolved=2.0, elapsed=70.0, event=i)
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.median_lateness_minutes == pytest.approx(10.0)
    assert health.worst_lateness_minutes == pytest.approx(10.0)


def test_a_row_resolved_early_reports_negative_lateness(clean_db):
    """Not clamped at zero: a row measured before its horizon is a
    different defect from one measured after, and hiding the sign would
    make it invisible."""
    for i in range(MIN_RESOLUTIONS_FOR_LATENESS):
        _row(clean_db, resolved=2.0, elapsed=55.0, event=i)
    clean_db.commit()

    assert check_resolver_health(clean_db, now=NOW).median_lateness_minutes == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# H8 - the unmeasurable rate
# ---------------------------------------------------------------------------

def test_the_unmeasurable_rate_is_over_closed_rows_only(clean_db):
    """H8. Nine pending rows and one sealed row is not "90% lost" - the
    pending ones have not failed at anything yet."""
    _row(clean_db, sealed="no price", event=1)
    _row(clean_db, resolved=5.0, event=2)
    for i in range(3, 11):
        _row(clean_db, horizon=60, due_minutes_ago=-30, event=i)   # pending
    clean_db.commit()

    health = check_resolver_health(clean_db, now=NOW)
    assert health.pending == 8
    assert health.unmeasurable_rate_pct == pytest.approx(50.0)


def test_the_unmeasurable_rate_is_none_when_nothing_has_closed(clean_db):
    """Zero would read as "nothing has been lost", which is true only in
    the vacuous sense that nothing has been attempted."""
    _row(clean_db, horizon=60, due_minutes_ago=-30)
    clean_db.commit()

    assert check_resolver_health(clean_db, now=NOW).unmeasurable_rate_pct is None


def test_the_health_dict_is_json_safe(clean_db):
    """It is served through the dashboard API, so a datetime left in the
    payload is a 500 at render time rather than a test failure here."""
    import json

    _row(clean_db, resolved=2.0, event=1)
    _row(clean_db, sealed="no price", event=2)
    clean_db.commit()

    payload = check_resolver_health(clean_db, now=NOW).as_dict()
    json.dumps(payload)
    assert payload["status"] == "HEALTHY"
    assert payload["healthy"] is True
