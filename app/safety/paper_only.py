"""The paper-only guarantee, in one place that anything can call.

CLAUDE.md's first non-negotiable is that this repository runs paper only:
`LIVE_TRADING` and `LIVE_EXECUTION_ACKNOWLEDGED` stay false, and there are
no wallet keys, no real funds and no live-order execution.

Until now that was enforced in three unrelated places - the two execution
backends refuse to submit without the acknowledgement flag, and
app/analysis/preflight.py reports on it - while the operator-facing entry
points (the setup launcher, the scanner one-shot, the test-signal sender)
would happily run against a live-configured .env and say "PAPER mode"
because they only read `LIVE_TRADING` and never the acknowledgement.

One function, so the answer cannot differ depending on which door you came
in through. It reads BOTH flags: either one being true means the
configuration is no longer paper-only, and a deployment with
`LIVE_TRADING=false` but the acknowledgement left on is one restart away
from trading real money.
"""
from __future__ import annotations

from app.config import settings


class LiveExecutionRefused(RuntimeError):
    """Raised when a paper-only entry point is asked to run live."""


def live_flags() -> dict[str, bool]:
    return {
        "LIVE_TRADING": bool(settings.LIVE_TRADING),
        "LIVE_EXECUTION_ACKNOWLEDGED": bool(
            getattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", False)
        ),
    }


def is_paper_only() -> bool:
    return not any(live_flags().values())


def violation_reason() -> str | None:
    """Why this configuration is not paper-only, or None if it is."""
    enabled = [name for name, value in live_flags().items() if value]
    if not enabled:
        return None
    return (
        f"{' and '.join(sorted(enabled))} is enabled. This repository is paper-only: "
        "set both LIVE_TRADING and LIVE_EXECUTION_ACKNOWLEDGED to false in .env and "
        "try again. No wallet keys, no real funds, no live-order execution."
    )


def require_paper_only() -> None:
    """Refuse to continue unless both live flags are off.

    Raises rather than returning a verdict: every caller here is an entry
    point where continuing is the wrong answer, and a boolean invites a
    caller that forgets to check it.
    """
    reason = violation_reason()
    if reason is not None:
        raise LiveExecutionRefused(reason)
