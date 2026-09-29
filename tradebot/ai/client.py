"""Shared Anthropic client plumbing.

The AI layers must never be able to take the trading loop down, so this module
centralises two things: lazy client construction (so the bot starts fine with
no API key) and a single place to ask "is the model layer usable right now?".
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

try:  # the SDK is optional - the rules engine runs without it
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore
    AsyncAnthropic = None  # type: ignore

_client: Optional["AsyncAnthropic"] = None


def sdk_installed() -> bool:
    return anthropic is not None


def credentials_present() -> bool:
    """True when the SDK can plausibly authenticate.

    An unset ANTHROPIC_API_KEY does not mean "no credentials": the SDK also
    reads ANTHROPIC_AUTH_TOKEN and the profile written by `ant auth login`.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    for candidate in ("~/.config/anthropic", "~/.anthropic"):
        if os.path.isdir(os.path.expanduser(candidate)):
            return True
    return False


def get_client(timeout: float = 45.0) -> Optional["AsyncAnthropic"]:
    global _client
    if anthropic is None:
        return None
    if _client is None:
        try:
            _client = AsyncAnthropic(timeout=timeout, max_retries=1)
        except Exception as exc:  # pragma: no cover - bad env, not our problem
            log.warning("could not construct Anthropic client: %s", exc)
            return None
    return _client


def availability() -> dict[str, object]:
    """Reported by `tradebot doctor` and shown in the dashboard header."""
    return {
        "sdk_installed": sdk_installed(),
        "credentials_present": credentials_present(),
        "ready": sdk_installed() and credentials_present(),
    }


NO_CREDENTIALS = (
    "no API credentials - set ANTHROPIC_API_KEY or run `ant auth login`"
)


def describe_error(exc: BaseException) -> str:
    """Short, non-leaky description of an SDK failure for the UI."""
    # The SDK reports a missing key as a bare TypeError from header building,
    # which would otherwise reach the user as the word "TypeError".
    if isinstance(exc, TypeError) and "authentication method" in str(exc):
        return NO_CREDENTIALS
    if anthropic is not None:
        if isinstance(exc, anthropic.RateLimitError):
            return "rate limited"
        if isinstance(exc, anthropic.APITimeoutError):
            return "timed out"
        if isinstance(exc, anthropic.AuthenticationError):
            return "authentication failed - check ANTHROPIC_API_KEY"
        if isinstance(exc, anthropic.APIStatusError):
            return f"API error {exc.status_code}"
        if isinstance(exc, anthropic.APIConnectionError):
            return "could not reach the API"
    return type(exc).__name__
