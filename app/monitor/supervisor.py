"""One loop harness for every background worker.

THE FAILURE THIS EXISTS TO PREVENT

Each worker used to run its own `while not stopped: await one_pass()`
loop, and each `one_pass` caught its own exceptions. That covers the
common case and misses two that matter.

  A PASS CAN FAIL BEFORE ITS OWN try BLOCK. Every worker opened its
  session with `db = SessionLocal()` on the line ABOVE the try. If that
  raised - a full disk, a database file that vanished, a connection pool
  exhausted - the exception went straight past the handler, out of the
  loop, and killed the task. asyncio does not report a task that dies
  unawaited until it is garbage collected, and app/main.py holds a
  reference to every task until shutdown, so nothing was printed at all.

  For the position monitor that is the worst failure in the bot: it is
  the only thing that fires stop-losses, and it would stop silently while
  the dashboard carried on showing open positions and a green health
  panel.

  A LOOP THAT FAILS EVERY TICK RETRIES AT FULL SPEED. When the cause is
  "database is locked" or an upstream rate limit, retrying on the normal
  interval makes the thing it is waiting for worse.

So every worker's loop body now runs through `run_supervised`, which
catches everything the pass can throw, backs off exponentially while it
keeps failing, and records a heartbeat when it succeeds.

WHY A HEARTBEAT AND NOT JUST A LOG

"The scanner has not completed a pass in four hours" is answerable from
process memory and is not answerable from a log nobody is tailing. The
heartbeats are deliberately in memory rather than the database: they
describe THIS process, and a restart genuinely resets them. Persisting
them would make a fresh process inherit the liveness of a dead one,
which is the opposite of what the panel is for.

CANCELLATION IS NOT FAILURE

asyncio.CancelledError is re-raised immediately and never counted as an
error or backed off. Shutdown cancels every task, and a supervisor that
swallowed that would hang the shutdown it was asked to perform.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.notifications.notifier import notifier

logger = logging.getLogger(__name__)

# A failing loop doubles its wait each time, up to this. Ten minutes is
# long enough to stop a retry storm making an outage worse, and short
# enough that a worker recovers on its own within one coffee break rather
# than needing a restart.
MAX_BACKOFF_SECONDS = 600.0


@dataclass
class Heartbeat:
    """Liveness for one worker, as seen from this process."""

    worker: str
    started_at: dt.datetime
    last_success_at: dt.datetime | None = None
    last_failure_at: dt.datetime | None = None
    last_error: str | None = None
    passes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    interval_seconds: float = 0.0
    stopped: bool = False

    def seconds_since_success(self, now: dt.datetime | None = None) -> float | None:
        if self.last_success_at is None:
            return None
        now = now or dt.datetime.now(dt.timezone.utc)
        return (now - self.last_success_at).total_seconds()

    def overdue(self, now: dt.datetime | None = None) -> bool:
        """Has this worker missed enough passes to be worth reporting?

        Three intervals, not one: a pass that runs slightly long, or an
        interval that lands just after a check, is normal and an alert on
        it would fire constantly and be ignored - which costs more than
        the ten minutes of extra latency before a real stall is noticed.

        A worker that has never succeeded is measured from startup, so a
        loop that died on its very first pass is still caught.
        """
        if self.stopped or self.interval_seconds <= 0:
            return False
        now = now or dt.datetime.now(dt.timezone.utc)
        reference = self.last_success_at or self.started_at
        return (now - reference).total_seconds() > self.interval_seconds * 3

    def as_dict(self) -> dict:
        return {
            "worker": self.worker,
            "started_at": self.started_at.isoformat(),
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "last_error": self.last_error,
            "passes": self.passes,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "interval_seconds": self.interval_seconds,
            "seconds_since_success": self.seconds_since_success(),
            "overdue": self.overdue(),
            "stopped": self.stopped,
        }


_heartbeats: dict[str, Heartbeat] = {}


def heartbeats() -> list[Heartbeat]:
    """Every worker this process has started, for the health panel."""
    return list(_heartbeats.values())


def heartbeat(worker: str) -> Heartbeat | None:
    return _heartbeats.get(worker)


def reset_heartbeats() -> None:
    """Tests only. A fresh process starts with an empty dict anyway."""
    _heartbeats.clear()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


async def run_supervised(
    worker: str,
    tick: Callable[[], Awaitable[object]],
    *,
    interval_seconds: float,
    stop_event: asyncio.Event,
    max_backoff_seconds: float = MAX_BACKOFF_SECONDS,
    on_result: Callable[[object], None] | None = None,
) -> None:
    """Run `tick` on an interval until `stop_event` is set.

    `tick` may raise anything: it is caught, reported once per throttle
    window, and retried with a growing delay. `on_result` receives each
    successful pass's return value, for the per-worker logging that used
    to live in the loop.
    """
    beat = Heartbeat(
        worker=worker, started_at=_now(), interval_seconds=interval_seconds
    )
    _heartbeats[worker] = beat

    try:
        while not stop_event.is_set():
            delay = interval_seconds
            try:
                result = await tick()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - a worker must not be able to die
                beat.failures += 1
                beat.consecutive_failures += 1
                beat.last_failure_at = _now()
                beat.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "%s pass failed (%d in a row) - backing off", worker,
                    beat.consecutive_failures,
                )
                # Doubling from the normal interval, capped. The notifier
                # throttles its own repeats, so a loop failing for an hour
                # produces a handful of messages rather than hundreds.
                delay = min(
                    interval_seconds * (2 ** min(beat.consecutive_failures, 10)),
                    max_backoff_seconds,
                )
                try:
                    await notifier.notify_worker_failure(worker, exc)
                except Exception:
                    # Reporting a failure must never become a second one.
                    logger.exception("could not report %s failure", worker)
            else:
                beat.passes += 1
                beat.consecutive_failures = 0
                beat.last_success_at = _now()
                if on_result is not None:
                    try:
                        on_result(result)
                    except Exception:
                        logger.exception("%s result handler failed", worker)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
    finally:
        beat.stopped = True
