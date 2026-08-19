"""Shared HTTP helper with rate-limit awareness.

Every external data source this bot uses is a free or cheap public tier
with a rate limit: DexScreener, GeckoTerminal, GoPlus, RugCheck, Birdeye.
Three subsystems hit them concurrently - the scanner (a batch of listing
and hydration calls every 60s), the position monitor (a price fetch per
open position every 30s), and the trade path (rug-check lookups plus a
candle fetch per candidate). Together that is easily enough to earn a 429.

Before this existed there was no retry anywhere: a 429 raised, the caller
logged it and returned None, and the bot's fail-closed design turned that
into "reject the trade". Safe, but it means a burst of rate limiting
silently stops the bot trading with nothing in the logs saying "you are
being throttled" rather than "no good setups". That is a bad failure mode
precisely because it looks like normal quiet behavior.

What this does:

  * retries 429 and 5xx with exponential backoff plus jitter
  * honors a `Retry-After` header when the server sends one, since a
    server's own number beats a guess
  * does NOT retry 4xx other than 429 - a 404 or a bad address will fail
    identically no matter how many times it is asked
  * gives up after MAX_ATTEMPTS and returns None, so every existing caller
    keeps its fail-closed behavior unchanged

Jitter matters here: the scanner fires a batch of requests at once on a
fixed 60s tick, so a fixed backoff would have them all retry in lockstep
and re-trigger the same limit.
"""
from __future__ import annotations

import asyncio
import logging
import random

import httpx

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Seconds to wait per the server's Retry-After header, if usable.

    Only the delta-seconds form is honored. The HTTP-date form is legal but
    rare on these APIs, and mis-parsing a date into a huge sleep would stall
    the bot far worse than falling back to our own backoff.
    """
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        seconds = float(raw.strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_BACKOFF_SECONDS)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter. `attempt` is 1-based."""
    capped = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
    return capped * (0.5 + random.random() / 2)  # 50-100% of the cap


async def get_json(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float = 15.0,
    label: str | None = None,
):
    """GET a JSON document, retrying transient failures. None on give-up.

    Returning None rather than raising keeps every caller's existing
    fail-closed handling intact: no data means the trade is rejected, never
    that a check is skipped.
    """
    label = label or url
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers, params=params)

            if response.status_code in RETRYABLE_STATUS:
                if attempt == MAX_ATTEMPTS:
                    logger.warning(
                        "%s: giving up after %d attempts (last status %d)%s",
                        label, MAX_ATTEMPTS, response.status_code,
                        " - you are being rate limited" if response.status_code == 429 else "",
                    )
                    return None
                delay = _parse_retry_after(response) or _backoff_seconds(attempt)
                logger.info(
                    "%s: status %d, retrying in %.1fs (attempt %d/%d)",
                    label, response.status_code, delay, attempt, MAX_ATTEMPTS,
                )
                await asyncio.sleep(delay)
                continue

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as exc:
            # A non-retryable 4xx - asking again will fail the same way.
            logger.warning("%s: %s", label, exc)
            return None
        except Exception as exc:  # noqa: BLE001 - network/parse errors are all "no data"
            if attempt == MAX_ATTEMPTS:
                logger.warning("%s: giving up after %d attempts (%s)", label, MAX_ATTEMPTS, exc)
                return None
            delay = _backoff_seconds(attempt)
            logger.info("%s: %s, retrying in %.1fs (attempt %d/%d)", label, exc, delay, attempt, MAX_ATTEMPTS)
            await asyncio.sleep(delay)

    return None
