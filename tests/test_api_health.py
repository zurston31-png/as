"""Tests for app/services/api_health.py.

The module exists because every gate in this bot fails closed, which makes
a dead upstream and a quiet market produce identical output. These tests
pin down the states that distinction depends on — especially "unused",
which must never render as healthy.
"""
import datetime as dt

import pytest

from app import models
from app.database import SessionLocal
from app.services import api_health


@pytest.fixture(autouse=True)
def _clean():
    api_health.reset()
    yield
    api_health.reset()


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------

def test_a_service_nobody_has_called_is_unused_not_ok():
    """Showing an uncalled service green would be a claim the bot cannot
    support - it has no idea whether that API works."""
    health = api_health.ServiceHealth("nevercalled")
    assert health.status() == "unused"
    assert health.success_rate is None


def test_a_working_service_is_ok():
    api_health.record_success("dexscreener")
    assert api_health.get("dexscreener").status() == "ok"


def test_one_failure_after_a_success_is_degraded_not_down():
    """A single miss on a free public tier is normal and must not light up
    the dashboard as an outage."""
    api_health.record_success("dexscreener")
    api_health.record_failure("dexscreener", "timeout")
    assert api_health.get("dexscreener").status() == "degraded"


def test_repeated_failures_are_down():
    api_health.record_success("goplus")
    for _ in range(api_health.DEGRADED_AFTER_CONSECUTIVE_FAILURES):
        api_health.record_failure("goplus", "HTTP 429")
    assert api_health.get("goplus").status() == "down"


def test_a_service_that_has_never_once_succeeded_is_down():
    api_health.record_failure("birdeye", "401 - bad API key")
    health = api_health.get("birdeye")
    assert health.last_success_at is None
    assert health.status() == "down"


def test_a_success_clears_the_consecutive_failure_count():
    for _ in range(5):
        api_health.record_failure("geckoterminal", "429")
    api_health.record_success("geckoterminal")
    health = api_health.get("geckoterminal")
    assert health.consecutive_failures == 0
    assert health.status() == "ok"
    # ...but the lifetime failure count is not erased.
    assert health.failure_count == 5


def test_a_long_silence_since_the_last_success_is_degraded():
    api_health.record_success("dexscreener")
    health = api_health.get("dexscreener")
    health.last_success_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        seconds=api_health.STALE_AFTER_SECONDS + 60
    )
    assert health.status() == "degraded"


def test_a_naive_timestamp_does_not_crash_the_age_calculation():
    api_health.record_success("dexscreener")
    health = api_health.get("dexscreener")
    health.last_success_at = dt.datetime.now().replace(tzinfo=None)
    assert health.seconds_since_success() < 5


# ---------------------------------------------------------------------------
# counters and reporting
# ---------------------------------------------------------------------------

def test_success_rate_is_computed_from_the_call_counts():
    for _ in range(3):
        api_health.record_success("dexscreener")
    api_health.record_failure("dexscreener", "timeout")
    assert api_health.get("dexscreener").success_rate == pytest.approx(75.0)


def test_a_huge_error_body_is_truncated():
    """Some upstreams return an entire HTML error page, and a dashboard
    cell is not the place for it."""
    api_health.record_failure("goplus", "x" * 5_000)
    assert len(api_health.get("goplus").last_error) == 500


def test_the_snapshot_sorts_problems_to_the_top():
    api_health.record_success("healthy")
    api_health.record_success("flaky")
    api_health.record_failure("flaky", "timeout")
    for _ in range(3):
        api_health.record_failure("broken", "429")

    order = [h.service for h in api_health.snapshot()]
    assert order.index("broken") < order.index("flaky") < order.index("healthy")


def test_as_dict_is_json_safe():
    import json

    api_health.record_success("dexscreener")
    api_health.record_failure("dexscreener", "boom")
    payload = api_health.get("dexscreener").as_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["status"] == "degraded"
    assert payload["last_error"] == "boom"


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

@pytest.fixture()
def db():
    session = SessionLocal()
    for row in session.query(models.ApiHealth).all():
        session.delete(row)
    session.commit()
    try:
        yield session
    finally:
        for row in session.query(models.ApiHealth).all():
            session.delete(row)
        session.commit()
        session.close()


def test_health_is_persisted_and_can_be_reloaded(db):
    api_health.record_success("dexscreener")
    api_health.record_failure("dexscreener", "timeout")
    assert api_health.persist(db) == 1
    db.commit()

    row = db.query(models.ApiHealth).filter_by(service="dexscreener").one()
    assert row.success_count == 1
    assert row.failure_count == 1
    assert row.last_error == "timeout"

    # A restart must not make a service that has been down for an hour
    # suddenly look "unused".
    api_health.reset()
    assert api_health.get("dexscreener") is None
    api_health.load(db)
    assert api_health.get("dexscreener").failure_count == 1


def test_persisting_is_an_upsert_not_an_append(db):
    """Two persists before a commit must update one row, not insert two.

    The session runs with autoflush=False, so a pending INSERT is invisible
    to the next lookup unless persist() flushes it - and the unique
    constraint on `service` turns the duplicate into an IntegrityError that
    poisons the whole transaction.
    """
    api_health.record_success("dexscreener")
    api_health.persist(db, force=True)
    api_health.record_success("dexscreener")
    api_health.persist(db, force=True)
    db.commit()
    assert db.query(models.ApiHealth).filter_by(service="dexscreener").count() == 1
    assert db.query(models.ApiHealth).filter_by(service="dexscreener").one().success_count == 2


def test_persistence_is_debounced(db):
    """The counters are exact in memory; the persisted copy is allowed to
    lag. Writing per HTTP call would mean dozens of writes a minute from
    the scanner alone."""
    api_health.record_success("dexscreener")
    assert api_health.persist(db) == 1        # first write goes through
    api_health.record_success("dexscreener")
    assert api_health.persist(db) == 0        # too soon
    assert api_health.persist(db, force=True) == 1


def test_reloading_an_empty_table_is_a_no_op(db):
    assert api_health.load(db) == 0
    assert api_health.snapshot() == []


# ---------------------------------------------------------------------------
# it must never change what the bot does
# ---------------------------------------------------------------------------

def test_health_tracking_never_influences_trading():
    """A degraded API must not relax a gate. Health that fed back into
    trading decisions would be a way for a broken upstream to lower the
    bar, which is exactly backwards - so nothing outside the dashboard,
    scripts and the HTTP helper may read this module."""
    import subprocess

    out = subprocess.run(
        ["grep", "-rln", "--include=*.py", "api_health", "app/"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    allowed = {
        "app/services/api_health.py",     # itself
        "app/services/http.py",           # records
        "app/dashboard/routes.py",        # reports
        "app/main.py",                    # loads at startup
        "app/models.py",                  # defines the table
    }
    assert set(out) <= allowed, f"api_health reached somewhere it must not: {set(out) - allowed}"
