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

from app.services import api_health

logger = logging.getLogger(__name__)

# Indirection so the suite can make backoff instant WITHOUT patching
# asyncio.sleep itself. Patching the attribute on the shared asyncio module
# reaches every module in the process: it silently stops coroutines ever
# suspending, so asyncio.gather stops interleaving and any test written to
# expose a race quietly passes for the wrong reason. Patch this name.
_sleep = asyncio.sleep

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


class LookupFailed(RuntimeError):
    """The upstream could not be reached or would not answer.

    Distinct from an upstream that answered "I have no record of this
    token". Both end up failing a trade closed, but only one of them is a
    reason to distrust the provider, and the audit trail has to be able to
    tell them apart - "we checked and found nothing" and "we could not
    check" are different facts about a security screen.
    """


async def get_json(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float = 15.0,
    label: str | None = None,
    service: str | None = None,
    on_status: dict[int, object] | None = None,
):
    """GET a JSON document, retrying transient failures. None on give-up.

    Returning None rather than raising keeps every caller's existing
    fail-closed handling intact: no data means the trade is rejected, never
    that a check is skipped.

    `service` names the upstream for health tracking (app/services/
    api_health.py). It exists because fail-closed makes a dead API and a
    quiet market look the same from the outside, and only the health
    record can tell them apart. Recording is best-effort and never changes
    the outcome of the call.
    """
    return await request_json(
        "GET", url, headers=headers, params=params,
        timeout=timeout, label=label, service=service, on_status=on_status,
    )


async def post_json(
    url: str,
    *,
    json: dict | None = None,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float = 15.0,
    label: str | None = None,
    service: str | None = None,
    idempotent: bool = False,
):
    """POST a JSON document, retrying only what is safe to retry.

    A GET can always be repeated. A POST cannot: when a request times out
    or the connection drops, there is no way to know whether the server
    processed it, and repeating it can submit the same thing twice.

    So the retry policy splits on `idempotent`:

        idempotent=True   the endpoint builds or reads something and
                          running it twice changes nothing (a quote, an
                          unsigned transaction). Retries like a GET.
        idempotent=False  default. Retries ONLY on 429, because a rate
                          limit is the one response that says the server
                          explicitly declined to process the request. A
                          timeout or a 503 is ambiguous and is not retried.

    Defaulting to False is the fail-safe direction: a caller that has not
    thought about it gets the cautious policy, and rate limiting - the
    thing these callers actually hit - is still handled.
    """
    return await request_json(
        "POST", url, json=json, headers=headers, params=params,
        timeout=timeout, label=label, service=service,
        retry_network_errors=idempotent,
        retryable_status=RETRYABLE_STATUS if idempotent else {429},
    )


async def request_json(
    method: str,
    url: str,
    *,
    json: dict | None = None,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float = 15.0,
    label: str | None = None,
    service: str | None = None,
    retry_network_errors: bool = True,
    retryable_status: set[int] | frozenset[int] = RETRYABLE_STATUS,
    on_status: dict[int, object] | None = None,
):
    """The shared retry/backoff/health loop behind get_json and post_json.

    `on_status` maps a status code to the value to return for it, for
    endpoints where a specific code is a legitimate answer rather than a
    failure - a 404 from a token-report API means "not indexed", which is
    information, not an outage. Those count as the provider having
    answered, so they record as healthy.
    """
    label = label or url
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method, url, headers=headers, params=params, json=json
                )

            if on_status and response.status_code in on_status:
                _note_success(service)
                return on_status[response.status_code]

            if response.status_code in retryable_status:
                if attempt == MAX_ATTEMPTS:
                    logger.warning(
                        "%s: giving up after %d attempts (last status %d)%s",
                        label, MAX_ATTEMPTS, response.status_code,
                        " - you are being rate limited" if response.status_code == 429 else "",
                    )
                    _note_failure(service, f"HTTP {response.status_code} after {MAX_ATTEMPTS} attempts")
                    return None
                delay = _parse_retry_after(response) or _backoff_seconds(attempt)
                logger.info(
                    "%s: status %d, retrying in %.1fs (attempt %d/%d)",
                    label, response.status_code, delay, attempt, MAX_ATTEMPTS,
                )
                await _sleep(delay)
                continue

            response.raise_for_status()
            payload = response.json()
            _note_success(service)
            return payload

        except httpx.HTTPStatusError as exc:
            # A non-retryable 4xx - asking again will fail the same way.
            logger.warning("%s: %s", label, exc)
            _note_failure(service, str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - network/parse errors are all "no data"
            if not retry_network_errors:
                # A non-idempotent request that failed mid-flight. There is
                # no way to know whether the server processed it, so asking
                # again risks doing it twice.
                logger.warning("%s: %s (not retried - request is not idempotent)", label, exc)
                _note_failure(service, str(exc))
                return None
            if attempt == MAX_ATTEMPTS:
                logger.warning("%s: giving up after %d attempts (%s)", label, MAX_ATTEMPTS, exc)
                _note_failure(service, str(exc))
                return None
            delay = _backoff_seconds(attempt)
            logger.info("%s: %s, retrying in %.1fs (attempt %d/%d)", label, exc, delay, attempt, MAX_ATTEMPTS)
            await _sleep(delay)

    return None


def _note_success(service: str | None) -> None:
    """Health recording must never be able to break a working request."""
    if not service:
        return
    try:
        api_health.record_success(service)
    except Exception:  # noqa: BLE001
        logger.debug("could not record API health for %s", service, exc_info=True)


def _note_failure(service: str | None, error: str) -> None:
    if not service:
        return
    try:
        api_health.record_failure(service, error)
    except Exception:  # noqa: BLE001
        logger.debug("could not record API health for %s", service, exc_info=True)
