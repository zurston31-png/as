"""Tests for app/services/http.py — rate-limit-aware fetching.

The failure this guards against is subtle: without retries a burst of 429s
made the bot silently stop trading, because every caller is fail-closed and
"no data" reads as "reject the trade". That looks exactly like normal quiet
behavior in the logs, which is the worst kind of outage.
"""
import httpx
import pytest

import app.services.http as http_helper
from app.services.http import (
    MAX_ATTEMPTS, _backoff_seconds, _parse_retry_after, get_json, post_json,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Keep the backoff logic intact but make the waits instant, so these
    tests exercise real retry behavior without actually sleeping."""
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(http_helper, "_sleep", fake_sleep)
    return slept


class _Response:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"ok": True}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self):
        return self._payload


def _client_returning(responses):
    """A fake httpx.AsyncClient yielding the given responses in order."""
    calls = {"n": 0}

    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, headers=None, params=None, json=None):
            # app/services/http.py issues every call through request(), so one
            # retry/backoff/health loop covers GET and POST alike. Delegates to
            # whichever verb method this double defines, passing only the
            # arguments that method actually accepts.
            import inspect

            fn = self.post if method == "POST" else self.get
            accepted = inspect.signature(fn).parameters
            kwargs = {
                name: value
                for name, value in (("headers", headers), ("params", params), ("json", json))
                if name in accepted and value is not None
            }
            return await fn(url, **kwargs)

        async def get(self, url, headers=None, params=None):
            index = min(calls["n"], len(responses) - 1)
            calls["n"] += 1
            result = responses[index]
            if isinstance(result, Exception):
                raise result
            return result

        async def post(self, url, headers=None, params=None, json=None):
            return await self.get(url, headers=headers, params=params)

    return FakeAsyncClient, calls


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

async def test_a_successful_response_is_returned_without_retrying(monkeypatch):
    client, calls = _client_returning([_Response(200, {"data": "yes"})])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await get_json("https://example.test") == {"data": "yes"}
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# retryable failures
# ---------------------------------------------------------------------------

async def test_a_429_is_retried_and_can_then_succeed(monkeypatch):
    client, calls = _client_returning([_Response(429), _Response(200, {"data": "recovered"})])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await get_json("https://example.test") == {"data": "recovered"}
    assert calls["n"] == 2, "should have retried exactly once before succeeding"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_retryable_statuses_are_retried_to_the_attempt_limit(monkeypatch, status):
    client, calls = _client_returning([_Response(status)])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await get_json("https://example.test") is None
    assert calls["n"] == MAX_ATTEMPTS


async def test_a_transient_network_error_is_retried(monkeypatch):
    client, calls = _client_returning([httpx.ConnectError("boom"), _Response(200, {"data": "ok"})])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await get_json("https://example.test") == {"data": "ok"}
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# non-retryable failures
# ---------------------------------------------------------------------------

async def test_a_404_is_not_retried(monkeypatch):
    """A bad address fails identically however many times it's asked -
    retrying just wastes the rate-limit budget that a real 429 needs."""
    client, calls = _client_returning([_Response(404)])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await get_json("https://example.test") is None
    assert calls["n"] == 1


async def test_a_401_is_not_retried(monkeypatch):
    client, calls = _client_returning([_Response(401)])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await get_json("https://example.test") is None
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Retry-After handling
# ---------------------------------------------------------------------------

async def test_retry_after_header_is_honored_over_our_own_backoff(monkeypatch, _no_real_sleeping):
    client, _ = _client_returning([_Response(429, headers={"Retry-After": "7"}), _Response(200)])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    await get_json("https://example.test")
    assert _no_real_sleeping == [7.0], "the server's own number should win over a guess"


def test_parse_retry_after_accepts_delta_seconds():
    assert _parse_retry_after(_Response(429, headers={"Retry-After": "12"})) == 12.0


def test_parse_retry_after_ignores_the_http_date_form():
    """The date form is legal but rare here, and mis-parsing one into a huge
    sleep would stall the bot far worse than our own backoff would."""
    resp = _Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert _parse_retry_after(resp) is None


def test_parse_retry_after_is_capped():
    from app.services.http import MAX_BACKOFF_SECONDS

    assert _parse_retry_after(_Response(429, headers={"Retry-After": "99999"})) == MAX_BACKOFF_SECONDS


def test_parse_retry_after_ignores_a_negative_value():
    assert _parse_retry_after(_Response(429, headers={"Retry-After": "-5"})) is None


def test_parse_retry_after_returns_none_without_the_header():
    assert _parse_retry_after(_Response(429)) is None


# ---------------------------------------------------------------------------
# backoff shape
# ---------------------------------------------------------------------------

def test_backoff_grows_with_each_attempt():
    # Jittered, so compare generously across many samples rather than once.
    early = max(_backoff_seconds(1) for _ in range(50))
    late = min(_backoff_seconds(3) for _ in range(50))
    assert late > early


def test_backoff_is_jittered_not_fixed():
    """The scanner fires a batch of requests on one tick; a fixed backoff
    would have them all retry in lockstep and re-trigger the same limit."""
    samples = {_backoff_seconds(2) for _ in range(25)}
    assert len(samples) > 1


def test_backoff_never_exceeds_the_cap():
    from app.services.http import MAX_BACKOFF_SECONDS

    assert all(_backoff_seconds(n) <= MAX_BACKOFF_SECONDS for n in range(1, 12))


# ---------------------------------------------------------------------------
# POST: retrying one is not automatically safe
# ---------------------------------------------------------------------------

async def test_a_non_idempotent_post_is_not_retried_after_a_network_error(monkeypatch):
    """A POST that failed mid-flight may already have been processed.

    Repeating it can submit the same thing twice, and nothing in the
    failure tells you which happened. Not retrying is the fail-safe
    direction, and it is the default so a caller that has not thought
    about it gets the cautious policy.
    """
    client, calls = _client_returning([httpx.ConnectError("dropped")])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await post_json("https://example.test", json={"x": 1}) is None
    assert calls["n"] == 1, "a non-idempotent POST was retried after a network error"


async def test_a_non_idempotent_post_still_retries_a_rate_limit(monkeypatch):
    """429 is the one response that says the server explicitly declined to
    process the request, so repeating it cannot duplicate anything - and
    rate limiting is what these callers actually hit."""
    client, calls = _client_returning([_Response(429), _Response(200, {"ok": True})])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await post_json("https://example.test", json={"x": 1}) == {"ok": True}
    assert calls["n"] == 2


async def test_a_non_idempotent_post_does_not_retry_a_503(monkeypatch):
    """Ambiguous: the server may have processed it and then failed to
    reply. A GET would retry this; a POST must not."""
    client, calls = _client_returning([_Response(503)])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await post_json("https://example.test", json={"x": 1}) is None
    assert calls["n"] == 1


async def test_an_idempotent_post_retries_like_a_get(monkeypatch):
    """A quote, an eth_call, an unsigned-transaction build: running it
    twice changes nothing, so it gets the full retry policy."""
    client, calls = _client_returning([_Response(503), httpx.ConnectError("x"), _Response(200, {"ok": 1})])
    monkeypatch.setattr(httpx, "AsyncClient", client)

    assert await post_json("https://example.test", json={"x": 1}, idempotent=True) == {"ok": 1}
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# a mapped status is an answer, not an outage
# ---------------------------------------------------------------------------

async def test_a_mapped_status_returns_its_value_and_counts_as_healthy(monkeypatch):
    """A 404 from a token-report API means "not indexed", which is
    information. Treating it as a failure would mark a working provider
    unhealthy every time it was asked about an unknown token."""
    from app.services import api_health

    client, calls = _client_returning([_Response(404)])
    monkeypatch.setattr(httpx, "AsyncClient", client)
    recorded = []
    monkeypatch.setattr(api_health, "record_success", lambda service: recorded.append(service))

    sentinel = object()
    result = await get_json(
        "https://example.test", service="rugcheck_xyz", on_status={404: sentinel}
    )

    assert result is sentinel
    assert calls["n"] == 1, "a mapped status must not be retried"
    assert recorded == ["rugcheck_xyz"]


# ---------------------------------------------------------------------------
# nothing bypasses the wrapper
# ---------------------------------------------------------------------------

def test_no_module_opens_its_own_http_client():
    """Six modules used to, and every one of them was a gap.

    A raw client gets no backoff, so a rate limit on a SECURITY scanner
    became an immediate failure; and it never registers with
    app/services/api_health.py, so the kill switch could not tell a dead
    provider from a quiet market. Both are invisible until the day they
    matter, which is why this is a test and not a convention.
    """
    import pathlib

    offenders = []
    for path in pathlib.Path("app").rglob("*.py"):
        if path == pathlib.Path("app/services/http.py"):
            continue     # the wrapper itself
        if "httpx.AsyncClient" in path.read_text():
            offenders.append(str(path))

    assert offenders == [], (
        "these modules bypass the shared retry/health wrapper: " + ", ".join(offenders)
    )
