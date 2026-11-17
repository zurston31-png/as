"""HTTP client for Alpaca's official historical stock bars endpoint.

    GET https://data.alpaca.markets/v2/stocks/bars

What this client guarantees:

* The configured feed and adjustment are sent explicitly on every request. If
  the provider refuses them the request fails with a clear error; a different
  feed is never substituted.
* Pagination follows ``next_page_token`` until the provider has no more data.
* Requests time out, are retried with exponential backoff on transient
  failures, and respect the provider's rate-limit headers.
* Requests are logged without ever including the API credentials.

The client returns bars exactly as the provider sent them. Checking the bars
themselves is the job of :mod:`app.data.validation`.
"""

import logging
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import (
    ALPACA_BARS_PATH,
    ALPACA_MAX_PAGE_LIMIT,
    Settings,
    is_valid_symbol,
    is_valid_timeframe,
)

logger = logging.getLogger(__name__)

#: Server-side failures that are worth retrying.
_RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})

#: Safety limit so a misbehaving server can never cause an endless loop.
_MAX_PAGES = 100_000

#: Longest single pause accepted from rate-limit headers.
_MAX_RATE_LIMIT_WAIT_SECONDS = 120.0

_ERROR_BODY_PREVIEW_CHARS = 300


class AlpacaAPIError(Exception):
    """Base class for every error raised by :class:`AlpacaClient`."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class AlpacaAuthenticationError(AlpacaAPIError):
    """HTTP 401: the API credentials were rejected."""


class AlpacaPermissionError(AlpacaAPIError):
    """HTTP 403: the credentials or subscription do not allow the request."""


class AlpacaRequestError(AlpacaAPIError):
    """The provider rejected the request itself (for example HTTP 400 or 422)."""


class AlpacaRateLimitError(AlpacaAPIError):
    """HTTP 429 persisted after all retries."""


class AlpacaServerError(AlpacaAPIError):
    """HTTP 5xx persisted after all retries."""


class AlpacaNetworkError(AlpacaAPIError):
    """Timeouts or connection failures persisted after all retries."""


class AlpacaResponseError(AlpacaAPIError):
    """The provider answered 200 but the body was not in the documented shape."""


@dataclass(frozen=True)
class BarsPage:
    """One page of the bars endpoint, exactly as returned by the provider."""

    page_number: int
    bars: dict[str, list[Any]]
    next_page_token: str | None

    @property
    def bar_count(self) -> int:
        return sum(len(items) for items in self.bars.values())


def _to_rfc3339(value: datetime) -> str:
    """Format an aware datetime as an RFC 3339 UTC timestamp for the API."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("start and end must be timezone-aware datetimes")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _error_detail(response: httpx.Response) -> str:
    """Extract the provider's error message without assuming a body format."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    text = response.text.strip()
    return text[:_ERROR_BODY_PREVIEW_CHARS] if text else "no error details returned"


class AlpacaClient:
    """Client for the historical stock bars endpoint.

    ``transport``, ``sleep`` and ``clock`` exist so tests can replace the
    network, waiting and the wall clock. Production code uses the defaults.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        backoff_jitter: float = 0.1,
    ) -> None:
        key_id, secret_key = settings.require_alpaca_credentials()

        self._feed = settings.alpaca_feed
        self._adjustment = settings.alpaca_adjustment
        self._currency = settings.alpaca_currency
        self._page_limit = settings.alpaca_page_limit
        self._timeout_seconds = settings.http_timeout_seconds
        self._max_retries = settings.http_max_retries
        self._backoff_base = settings.http_backoff_base_seconds
        self._backoff_max = settings.http_backoff_max_seconds
        self._backoff_jitter = backoff_jitter
        self._sleep = sleep
        self._clock = clock
        self._throttle_until: float | None = None

        # Credentials travel only in these two headers. They are never placed
        # in a URL and never passed to a logger.
        self._http = httpx.Client(
            base_url=settings.alpaca_data_base_url,
            headers={
                "APCA-API-KEY-ID": key_id,
                "APCA-API-SECRET-KEY": secret_key,
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            transport=transport,
        )

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "AlpacaClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    # -- what every request is sent with --------------------------------------

    @property
    def feed(self) -> str:
        return self._feed

    @property
    def adjustment(self) -> str:
        return self._adjustment

    @property
    def currency(self) -> str:
        return self._currency

    # -- public API -------------------------------------------------------------

    def iter_bar_pages(
        self,
        symbols: Sequence[str],
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> Iterator[BarsPage]:
        """Yield every page of bars for ``symbols`` between ``start`` and ``end``.

        ``start`` and ``end`` must be timezone-aware; both ends are inclusive.
        Pages are requested one after another with ``next_page_token`` until
        the provider reports that no further page exists.
        """
        symbols = list(symbols)
        if not symbols:
            raise ValueError("At least one symbol is required.")
        invalid = [symbol for symbol in symbols if not is_valid_symbol(symbol)]
        if invalid:
            raise ValueError(f"Invalid symbol(s): {invalid}")
        if not is_valid_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe!r}")
        page_limit = self._page_limit if limit is None else limit
        if not 1 <= page_limit <= ALPACA_MAX_PAGE_LIMIT:
            raise ValueError(f"limit must be between 1 and {ALPACA_MAX_PAGE_LIMIT}")
        start_text, end_text = _to_rfc3339(start), _to_rfc3339(end)
        if start > end:
            raise ValueError("start must not be after end")

        params: dict[str, str | int] = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start_text,
            "end": end_text,
            "limit": page_limit,
            "adjustment": self._adjustment,
            "feed": self._feed,
            "currency": self._currency,
            "sort": "asc",
        }

        seen_tokens: set[str] = set()
        page_token: str | None = None
        for page_number in range(1, _MAX_PAGES + 1):
            page_params = dict(params)
            if page_token is not None:
                page_params["page_token"] = page_token

            response = self._send(page_params, page_number)
            page = self._parse_page(response, page_number)
            logger.info(
                "Alpaca page %d for %s: %d bars, more pages: %s",
                page_number,
                params["symbols"],
                page.bar_count,
                "yes" if page.next_page_token else "no",
            )
            yield page

            if page.next_page_token is None:
                return
            if page.next_page_token in seen_tokens:
                raise AlpacaResponseError(
                    "Alpaca returned the same next_page_token twice; stopping to avoid an "
                    "endless pagination loop."
                )
            seen_tokens.add(page.next_page_token)
            page_token = page.next_page_token

        raise AlpacaResponseError(f"Pagination did not finish within {_MAX_PAGES} pages.")

    def fetch_bars(
        self,
        symbols: Sequence[str],
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> dict[str, list[Any]]:
        """Return all bars for ``symbols``, following pagination to the end."""
        collected: dict[str, list[Any]] = {}
        for page in self.iter_bar_pages(
            symbols, timeframe=timeframe, start=start, end=end, limit=limit
        ):
            for symbol, items in page.bars.items():
                collected.setdefault(symbol, []).extend(items)
        return collected

    # -- request handling -------------------------------------------------------

    def _send(self, params: dict[str, str | int], page_number: int) -> httpx.Response:
        """Send one page request, retrying transient failures."""
        max_attempts = self._max_retries + 1
        for attempt in range(1, max_attempts + 1):
            self._wait_for_rate_limit_window()
            logger.info(
                "Alpaca request: GET %s symbols=%s timeframe=%s start=%s end=%s feed=%s "
                "adjustment=%s limit=%s page=%d attempt=%d/%d",
                ALPACA_BARS_PATH,
                params["symbols"],
                params["timeframe"],
                params["start"],
                params["end"],
                params["feed"],
                params["adjustment"],
                params["limit"],
                page_number,
                attempt,
                max_attempts,
            )
            started = time.monotonic()
            failure: AlpacaAPIError
            try:
                response = self._http.get(ALPACA_BARS_PATH, params=params)
            except httpx.TimeoutException as exc:
                failure = AlpacaNetworkError(
                    f"The request to Alpaca timed out after {self._timeout_seconds:g} seconds "
                    f"({type(exc).__name__})."
                )
                delay = self._backoff_delay(attempt)
            except httpx.ProxyError as exc:
                # A proxy refusing the connection is a network-policy problem,
                # not a transient fault, so retrying would only waste time.
                raise AlpacaNetworkError(
                    f"A proxy refused the connection to Alpaca ({exc}). Check that "
                    "data.alpaca.markets is reachable from this network."
                ) from exc
            except httpx.TransportError as exc:
                failure = AlpacaNetworkError(
                    f"Network error while contacting Alpaca: {type(exc).__name__}: {exc}"
                )
                delay = self._backoff_delay(attempt)
            except httpx.HTTPError as exc:
                raise AlpacaNetworkError(
                    f"Could not complete the request to Alpaca: {type(exc).__name__}: {exc}"
                ) from exc
            else:
                elapsed_ms = (time.monotonic() - started) * 1000
                request_id = response.headers.get("X-Request-ID")
                logger.info(
                    "Alpaca response: status=%d elapsed_ms=%.0f rate_limit_remaining=%s "
                    "request_id=%s",
                    response.status_code,
                    elapsed_ms,
                    response.headers.get("X-RateLimit-Remaining", "n/a"),
                    request_id or "n/a",
                )
                if response.status_code == 200:
                    self._note_rate_limit(response)
                    return response
                if response.status_code == 429:
                    failure = AlpacaRateLimitError(
                        f"Alpaca rate limit still exceeded after {max_attempts} attempts "
                        f"(HTTP 429: {_error_detail(response)}). Wait a minute and try again.",
                        status_code=429,
                        request_id=request_id,
                    )
                    delay = self._rate_limit_delay(response, attempt)
                elif response.status_code in _RETRYABLE_STATUS_CODES:
                    failure = AlpacaServerError(
                        f"Alpaca server error persisted after {max_attempts} attempts "
                        f"(HTTP {response.status_code}: {_error_detail(response)}).",
                        status_code=response.status_code,
                        request_id=request_id,
                    )
                    delay = self._backoff_delay(attempt)
                else:
                    raise self._client_error(response)

            if attempt == max_attempts:
                raise failure
            logger.warning(
                "Alpaca request attempt %d/%d failed (%s); retrying in %.1f seconds",
                attempt,
                max_attempts,
                type(failure).__name__,
                delay,
            )
            self._sleep(delay)

        raise AssertionError("unreachable")  # pragma: no cover

    def _client_error(self, response: httpx.Response) -> AlpacaAPIError:
        """Translate a non-retryable HTTP error into a clear exception."""
        status = response.status_code
        detail = _error_detail(response)
        request_id = response.headers.get("X-Request-ID")
        suffix = f" [request id: {request_id}]" if request_id else ""

        if status == 401:
            return AlpacaAuthenticationError(
                f"Alpaca rejected the API credentials (HTTP 401: {detail}). Check "
                f"APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env file.{suffix}",
                status_code=status,
                request_id=request_id,
            )
        if status == 403:
            return AlpacaPermissionError(
                f"Alpaca refused the request (HTTP 403: {detail}). Either the API keys are "
                f"not valid, or the account's market data subscription does not allow feed "
                f"'{self._feed}' for this request. SignalIQ never substitutes a different "
                f"feed.{suffix}",
                status_code=status,
                request_id=request_id,
            )
        if status in (400, 422):
            return AlpacaRequestError(
                f"Alpaca rejected the request parameters (HTTP {status}: {detail}).{suffix}",
                status_code=status,
                request_id=request_id,
            )
        if status == 404:
            return AlpacaRequestError(
                f"Alpaca endpoint not found (HTTP 404: {detail}). Check "
                f"SIGNALIQ_ALPACA_DATA_BASE_URL.{suffix}",
                status_code=status,
                request_id=request_id,
            )
        return AlpacaRequestError(
            f"Unexpected response from Alpaca (HTTP {status}: {detail}).{suffix}",
            status_code=status,
            request_id=request_id,
        )

    def _parse_page(self, response: httpx.Response, page_number: int) -> BarsPage:
        """Check the documented top-level shape of a successful response."""
        try:
            payload = response.json()
        except ValueError:
            raise AlpacaResponseError(
                "Alpaca returned a response that is not valid JSON.", status_code=200
            ) from None
        if not isinstance(payload, dict) or "bars" not in payload:
            raise AlpacaResponseError(
                "Alpaca response does not contain the expected 'bars' field.", status_code=200
            )

        raw_bars = payload["bars"]
        if raw_bars is None:  # treated the same as "no bars in this window"
            raw_bars = {}
        if not isinstance(raw_bars, dict):
            raise AlpacaResponseError(
                "Alpaca response field 'bars' is not an object keyed by symbol.", status_code=200
            )
        bars: dict[str, list[Any]] = {}
        for symbol, items in raw_bars.items():
            if items is None:
                items = []
            if not isinstance(items, list):
                raise AlpacaResponseError(
                    f"Alpaca response field 'bars.{symbol}' is not a list.", status_code=200
                )
            bars[str(symbol)] = items

        currency = payload.get("currency")
        if currency is not None and str(currency).upper() != self._currency:
            raise AlpacaResponseError(
                f"Alpaca answered in currency {currency!r} but {self._currency!r} was "
                "requested; refusing to store prices in an unexpected currency.",
                status_code=200,
            )

        token = payload.get("next_page_token")
        if token is not None and not isinstance(token, str):
            raise AlpacaResponseError(
                "Alpaca response field 'next_page_token' is not a string.", status_code=200
            )
        return BarsPage(page_number=page_number, bars=bars, next_page_token=token or None)

    # -- retry timing -------------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff: base, 2*base, 4*base ... capped at the maximum."""
        delay = min(self._backoff_base * (2 ** (attempt - 1)), self._backoff_max)
        if self._backoff_jitter > 0:
            delay += random.uniform(0, delay * self._backoff_jitter)
        return delay

    def _rate_limit_delay(self, response: httpx.Response, attempt: int) -> float:
        """How long to wait after HTTP 429, preferring the provider's own hints."""
        retry_after = _parse_number(response.headers.get("Retry-After"))
        if retry_after is not None and retry_after >= 0:
            delay = retry_after
        else:
            reset_at = _parse_number(response.headers.get("X-RateLimit-Reset"))
            if reset_at is not None:
                delay = max(0.0, reset_at - self._clock()) + 1.0
            else:
                delay = self._backoff_delay(attempt)
        return min(delay, _MAX_RATE_LIMIT_WAIT_SECONDS)

    def _note_rate_limit(self, response: httpx.Response) -> None:
        """Remember when the quota resets if this response used the last request."""
        remaining = _parse_number(response.headers.get("X-RateLimit-Remaining"))
        reset_at = _parse_number(response.headers.get("X-RateLimit-Reset"))
        if remaining is not None and remaining <= 0 and reset_at is not None:
            self._throttle_until = min(reset_at + 1.0, self._clock() + _MAX_RATE_LIMIT_WAIT_SECONDS)
        else:
            self._throttle_until = None

    def _wait_for_rate_limit_window(self) -> None:
        """Pause before a request when the previous response exhausted the quota."""
        if self._throttle_until is None:
            return
        wait = self._throttle_until - self._clock()
        self._throttle_until = None
        if wait > 0:
            logger.warning("Alpaca rate limit reached; waiting %.1f seconds before continuing", wait)
            self._sleep(wait)
