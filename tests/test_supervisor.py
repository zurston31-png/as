"""Tests for the background-worker supervisor (app/monitor/supervisor.py).

The invariants:

  S1  A pass that raises does not kill the loop - including one that
      raises BEFORE its own try block, which is the case that used to
      silently stop the position monitor and with it every stop-loss.
  S2  A repeatedly failing loop backs off exponentially instead of
      retrying at full speed and making the outage worse.
  S3  Backoff resets the moment a pass succeeds.
  S4  Cancellation is not failure: it propagates immediately, is not
      counted, and is not backed off, or shutdown would hang.
  S5  A failure inside the failure notifier does not become a second
      failure.
  S6  The heartbeat records liveness, and a worker that has never
      succeeded is measured from startup so a loop that died on its first
      pass is still caught.
  S7  Every worker loop in the bot is actually supervised - a helper
      nothing uses is worse than no helper.
"""
import asyncio

import pytest

from app.monitor import supervisor
from app.monitor.supervisor import (
    Heartbeat,
    heartbeat,
    heartbeats,
    reset_heartbeats,
    run_supervised,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _clean():
    reset_heartbeats()
    yield
    reset_heartbeats()


@pytest.fixture(autouse=True)
def _quiet_notifier(monkeypatch):
    """Stop tests reaching for the network. Recorded so the notification
    behaviour itself can be asserted."""
    sent = []

    async def record(worker, exc):
        sent.append((worker, exc))

    monkeypatch.setattr(supervisor.notifier, "notify_worker_failure", record)
    return sent


@pytest.fixture(autouse=True)
def _instant_sleep(monkeypatch):
    """Replace the inter-pass wait with a recorder.

    The supervisor waits by `asyncio.wait_for(stop_event.wait(), timeout=delay)`,
    so intercepting wait_for both makes the tests instant and captures the
    delay actually chosen - which is the thing under test for backoff.
    """
    delays = []

    async def fake_wait_for(awaitable, timeout=None):
        delays.append(timeout)
        # Close the coroutine we are not awaiting, or Python warns.
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(supervisor.asyncio, "wait_for", fake_wait_for)
    return delays


def _stops_after(n: int):
    """A tick that runs `n` times and then sets the stop event."""
    stop = asyncio.Event()
    calls = []

    async def tick():
        calls.append(len(calls))
        if len(calls) >= n:
            stop.set()

    return stop, calls, tick


# ---------------------------------------------------------------------------
# S1 - a failing pass does not kill the loop
# ---------------------------------------------------------------------------

async def test_a_raising_pass_does_not_stop_the_loop():
    """S1. Before this, a pass that raised outside its own handler killed
    the task, and asyncio does not report a task nobody awaits - so the
    position monitor could stop firing stop-losses in total silence."""
    stop = asyncio.Event()
    calls = []

    async def tick():
        calls.append(1)
        if len(calls) >= 3:
            stop.set()
        raise RuntimeError("SessionLocal() failed - disk full")

    await run_supervised("test", tick, interval_seconds=1.0, stop_event=stop)
    assert len(calls) == 3


async def test_even_a_base_exception_is_contained():
    """A worker must not be able to die. MemoryError and friends do not
    inherit from Exception, and one escaping would take the loop with
    it."""
    stop = asyncio.Event()
    calls = []

    async def tick():
        calls.append(1)
        if len(calls) >= 2:
            stop.set()
        raise MemoryError("out of memory")

    await run_supervised("test", tick, interval_seconds=1.0, stop_event=stop)
    assert len(calls) == 2


async def test_the_failure_is_reported(_quiet_notifier):
    stop = asyncio.Event()

    async def tick():
        stop.set()
        raise ValueError("boom")

    await run_supervised("scanner", tick, interval_seconds=1.0, stop_event=stop)
    assert _quiet_notifier[0][0] == "scanner"
    assert isinstance(_quiet_notifier[0][1], ValueError)


# ---------------------------------------------------------------------------
# S2/S3 - backoff
# ---------------------------------------------------------------------------

async def test_a_repeatedly_failing_loop_backs_off(_instant_sleep):
    """S2. Retrying a locked database or a rate-limited upstream at full
    speed makes the thing you are waiting for worse."""
    stop = asyncio.Event()
    calls = []

    async def tick():
        calls.append(1)
        if len(calls) >= 4:
            stop.set()
        raise RuntimeError("still broken")

    await run_supervised("test", tick, interval_seconds=10.0, stop_event=stop)
    assert _instant_sleep == [20.0, 40.0, 80.0, 160.0]


async def test_the_backoff_is_capped(_instant_sleep):
    """Unbounded doubling would eventually mean a worker that recovers
    but does not notice for a day."""
    stop = asyncio.Event()
    calls = []

    async def tick():
        calls.append(1)
        if len(calls) >= 6:
            stop.set()
        raise RuntimeError("still broken")

    await run_supervised(
        "test", tick, interval_seconds=10.0, stop_event=stop, max_backoff_seconds=50.0,
    )
    assert max(_instant_sleep) == 50.0
    assert _instant_sleep[-1] == 50.0


async def test_a_success_resets_the_backoff(_instant_sleep):
    """S3. A worker that recovers must go straight back to its normal
    cadence, not crawl out of the backoff it earned while broken."""
    stop = asyncio.Event()
    calls = []

    async def tick():
        calls.append(1)
        if len(calls) >= 4:
            stop.set()
        if len(calls) < 3:
            raise RuntimeError("broken")

    await run_supervised("test", tick, interval_seconds=10.0, stop_event=stop)
    assert _instant_sleep == [20.0, 40.0, 10.0, 10.0]


async def test_a_healthy_loop_waits_exactly_its_interval(_instant_sleep):
    stop, calls, tick = _stops_after(3)
    await run_supervised("test", tick, interval_seconds=7.5, stop_event=stop)
    assert _instant_sleep == [7.5, 7.5, 7.5]


# ---------------------------------------------------------------------------
# S4 - cancellation
# ---------------------------------------------------------------------------

async def test_cancellation_propagates_and_is_not_counted():
    """S4. Shutdown cancels every task; a supervisor that swallowed it
    would hang the shutdown it was asked to perform."""
    stop = asyncio.Event()

    async def tick():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run_supervised("test", tick, interval_seconds=1.0, stop_event=stop)

    beat = heartbeat("test")
    assert beat.failures == 0
    assert beat.stopped is True, "the heartbeat must still be marked stopped"


# ---------------------------------------------------------------------------
# S5 - reporting a failure must not fail
# ---------------------------------------------------------------------------

async def test_a_broken_notifier_does_not_break_the_loop(monkeypatch):
    """S5. The notification exists to report a bug; it must not become
    one inside the handler that is reporting it."""
    async def explode(*_a, **_k):
        raise RuntimeError("telegram is down")

    monkeypatch.setattr(supervisor.notifier, "notify_worker_failure", explode)

    stop = asyncio.Event()
    calls = []

    async def tick():
        calls.append(1)
        if len(calls) >= 2:
            stop.set()
        raise ValueError("the original problem")

    await run_supervised("test", tick, interval_seconds=1.0, stop_event=stop)
    assert len(calls) == 2


async def test_a_broken_result_handler_does_not_break_the_loop(monkeypatch):
    """The per-worker logging callback is the least important thing in
    the loop and must not be able to stop the most important."""
    stop, calls, tick = _stops_after(2)

    def explode(_result):
        raise RuntimeError("logging blew up")

    await run_supervised(
        "test", tick, interval_seconds=1.0, stop_event=stop, on_result=explode,
    )
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# S6 - the heartbeat
# ---------------------------------------------------------------------------

async def test_the_heartbeat_records_successes_and_failures():
    stop = asyncio.Event()
    calls = []

    async def tick():
        calls.append(1)
        if len(calls) >= 3:
            stop.set()
        if len(calls) == 2:
            raise ValueError("one bad pass")

    await run_supervised("test", tick, interval_seconds=1.0, stop_event=stop)

    beat = heartbeat("test")
    assert beat.passes == 2
    assert beat.failures == 1
    assert beat.consecutive_failures == 0
    assert "ValueError: one bad pass" == beat.last_error
    assert beat.last_success_at is not None


async def test_a_worker_that_never_succeeded_is_measured_from_startup():
    """S6. Otherwise a loop that died on its very first pass has no
    last_success_at, and "overdue" computed from a null reads as fine."""
    import datetime as dt

    beat = Heartbeat(
        worker="test",
        started_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1),
        interval_seconds=60.0,
    )
    assert beat.last_success_at is None
    assert beat.overdue()


async def test_a_recent_success_is_not_overdue():
    import datetime as dt

    beat = Heartbeat(
        worker="test", started_at=dt.datetime.now(dt.timezone.utc),
        interval_seconds=60.0,
        last_success_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=30),
    )
    assert not beat.overdue()


async def test_the_overdue_window_is_three_intervals_not_one():
    """One interval would fire constantly - a pass running slightly long
    is normal - and an alert that cries wolf is worth less than the ten
    minutes of latency this costs on a real stall."""
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    beat = Heartbeat(worker="test", started_at=now, interval_seconds=60.0)

    beat.last_success_at = now - dt.timedelta(seconds=120)
    assert not beat.overdue()
    beat.last_success_at = now - dt.timedelta(seconds=200)
    assert beat.overdue()


async def test_a_stopped_worker_is_not_reported_as_overdue():
    """A deliberately disabled worker is not a fault, and flagging one
    would put a permanent red light on the health panel."""
    stop, calls, tick = _stops_after(1)
    await run_supervised("test", tick, interval_seconds=1.0, stop_event=stop)

    beat = heartbeat("test")
    assert beat.stopped
    assert not beat.overdue()


async def test_the_heartbeat_is_listed_and_json_safe():
    import json

    stop, calls, tick = _stops_after(1)
    await run_supervised("test", tick, interval_seconds=1.0, stop_event=stop)

    assert [b.worker for b in heartbeats()] == ["test"]
    json.dumps(heartbeat("test").as_dict())


# ---------------------------------------------------------------------------
# S7 - every loop actually uses it
# ---------------------------------------------------------------------------

def test_every_background_loop_is_supervised():
    """S7. A helper nothing uses is worse than no helper - it reads as
    protection that is not there. If a new worker is added with a
    hand-rolled `while not stopped` loop, this fails."""
    import inspect

    from app.autopilot import loop as autopilot_loop
    from app.early import loop as early_loop
    from app.monitor import forward_return_worker, position_monitor, shadow_resolver_worker
    from app.scanner import loop as scanner_loop

    for module in (
        position_monitor, forward_return_worker, shadow_resolver_worker,
        early_loop, scanner_loop, autopilot_loop,
    ):
        source = inspect.getsource(module.run_forever)
        assert "run_supervised" in source, f"{module.__name__}.run_forever is not supervised"


def test_no_worker_pass_swallows_its_own_failure():
    """A pass that catches, reports and returns normally looks like a
    SUCCESS to the supervisor - no backoff, no failure count, and a
    heartbeat that says all is well. The rollback stays local; the
    reporting belongs to the supervisor, so each pass must re-raise."""
    import inspect

    from app.autopilot import loop as autopilot_loop
    from app.early import loop as early_loop
    from app.monitor import forward_return_worker, position_monitor, shadow_resolver_worker
    from app.scanner import loop as scanner_loop

    passes = [
        position_monitor._check_positions_once,
        forward_return_worker.resolve_once,
        shadow_resolver_worker.resolve_once,
        early_loop.evaluate_once,
        scanner_loop.scan_once,
        autopilot_loop.run_once,
    ]
    for fn in passes:
        source = inspect.getsource(fn)
        assert "db.rollback()" in source, f"{fn.__qualname__} does not roll back"
        assert "raise" in source, f"{fn.__qualname__} swallows its own failure"
