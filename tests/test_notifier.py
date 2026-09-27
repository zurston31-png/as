"""Tests for app/notifications/notifier.py's worker-failure notifications.

The background loops (scanner, position monitor, forward-return resolver,
shadow resolver, autopilot, backup, early watchlist) each swallow their own
whole-pass exceptions so one bad tick never takes the process down - but
until now that meant the only record was a server log nobody not already
SSH'd in would read. notify_worker_failure closes that gap; these tests are
about the one thing that would make it worse than the silence it replaces:
a loop failing every tick flooding the channel until the operator mutes it.
"""
import pytest

from app.notifications.notifier import Notifier, WORKER_FAILURE_THROTTLE_SECONDS


class _Recording(Notifier):
    """Captures broadcasts instead of hitting the network."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []

    async def _broadcast(self, text: str) -> None:
        self.sent.append(text)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_a_worker_failure_sends_the_first_time():
    n = _Recording()
    await n.notify_worker_failure("scanner", ValueError("boom"))
    assert len(n.sent) == 1
    assert "scanner" in n.sent[0]
    assert "ValueError" in n.sent[0]
    assert "boom" in n.sent[0]


@pytest.mark.anyio
async def test_the_traceback_is_not_included_only_type_and_message():
    """The full traceback already went to the server log via
    logger.exception right before this is called. A Telegram message is
    not where a stack trace belongs - it names the failure and points at
    where the detail actually lives."""
    n = _Recording()
    await n.notify_worker_failure("scanner", ValueError("boom"))
    assert "docker compose logs" in n.sent[0]
    assert "Traceback" not in n.sent[0]


@pytest.mark.anyio
async def test_a_second_failure_within_the_window_is_suppressed():
    """The failure mode this whole feature exists to avoid: a loop that
    fails every tick sending a message every tick forever, which trains the
    operator to ignore the channel that exists to be trusted."""
    n = _Recording()
    await n.notify_worker_failure("scanner", ValueError("first"))
    await n.notify_worker_failure("scanner", ValueError("second"))
    await n.notify_worker_failure("scanner", RuntimeError("third"))
    assert len(n.sent) == 1


@pytest.mark.anyio
async def test_a_failure_after_the_window_sends_again(monkeypatch):
    """Still broken later must not stay silent forever - it is a reminder,
    not a one-time notice."""
    import app.notifications.notifier as notifier_module

    clock = [1000.0]
    monkeypatch.setattr(notifier_module.time, "monotonic", lambda: clock[0])

    n = _Recording()
    await n.notify_worker_failure("scanner", ValueError("first"))
    clock[0] += WORKER_FAILURE_THROTTLE_SECONDS + 1
    await n.notify_worker_failure("scanner", ValueError("still broken"))

    assert len(n.sent) == 2
    assert "still broken" in n.sent[1]


@pytest.mark.anyio
async def test_different_workers_are_throttled_independently():
    """The scanner failing must not silence the position monitor - they are
    unrelated bugs and both deserve to be seen."""
    n = _Recording()
    await n.notify_worker_failure("scanner", ValueError("scanner broke"))
    await n.notify_worker_failure("position monitor", ValueError("monitor broke"))
    assert len(n.sent) == 2


@pytest.mark.anyio
async def test_a_notification_failure_never_raises():
    """_broadcast already catches its own send errors (see _send_telegram/
    _send_discord); notify_worker_failure must not add a new way for a
    notification about a bug to itself become an unhandled exception inside
    the except block that is reporting it."""
    n = Notifier()  # the real one - no network configured, so both sends no-op

    await n.notify_worker_failure("scanner", ValueError("boom"))  # must not raise
