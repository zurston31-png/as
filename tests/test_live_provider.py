"""Tests for app/data/live_provider.py - GeckoTerminal pool resolution and
OHLCV parsing. Built against documented/trained knowledge of GeckoTerminal's
public API v2 shape (see the module's own honesty note); these tests prove
the PARSING logic is correct against that assumed shape, not that the shape
itself is right - see the module docstring for why that distinction matters
and what to do about it (scripts/diagnose_token.py against a real token).
"""
import httpx
import pytest

from app.data.candles import Timeframe
from app.data.live_provider import (
    CHAIN_TO_GECKOTERMINAL_NETWORK,
    _find_primary_pool,
    _parse_ohlcv_response,
    fetch_candles,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeResponse:
    """Stands in for httpx.Response. Needs status_code and headers as well
    as json(), because requests now go through app/services/http.py's
    rate-limit-aware wrapper, which inspects both to decide on a retry."""

    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok
        self.status_code = 200 if status_ok else 404
        self.headers = {}

    def raise_for_status(self):
        if not self._status_ok:
            raise httpx.HTTPStatusError("bad status", request=None, response=None)

    def json(self):
        return self._payload


def _fake_client(get_impl):
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

        async def get(self, url, params=None, headers=None):
            return get_impl(url, params, headers)

    return FakeAsyncClient


# ---------------------------------------------------------------------------
# pool resolution
# ---------------------------------------------------------------------------

async def test_find_primary_pool_picks_the_highest_liquidity_pool(monkeypatch):
    def get_impl(url, params, headers):
        return _FakeResponse({
            "data": [
                {"attributes": {"address": "PoolA", "reserve_in_usd": "1000"}},
                {"attributes": {"address": "PoolB", "reserve_in_usd": "50000"}},
                {"attributes": {"address": "PoolC", "reserve_in_usd": "200"}},
            ]
        })

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    pool = await _find_primary_pool("solana", "TokenMint111")
    assert pool == "PoolB"


async def test_find_primary_pool_returns_none_on_empty_data(monkeypatch):
    def get_impl(url, params, headers):
        return _FakeResponse({"data": []})

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    assert await _find_primary_pool("solana", "TokenMint111") is None


async def test_find_primary_pool_returns_none_on_request_failure(monkeypatch):
    def get_impl(url, params, headers):
        return _FakeResponse({}, status_ok=False)

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    assert await _find_primary_pool("solana", "TokenMint111") is None


async def test_find_primary_pool_handles_missing_reserve_field_gracefully(monkeypatch):
    """A pool with no reserve_in_usd field at all must not crash the max()
    comparison - it should just lose to any pool that does report one."""
    def get_impl(url, params, headers):
        return _FakeResponse({
            "data": [
                {"attributes": {"address": "PoolNoReserve"}},
                {"attributes": {"address": "PoolWithReserve", "reserve_in_usd": "500"}},
            ]
        })

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    pool = await _find_primary_pool("solana", "TokenMint111")
    assert pool == "PoolWithReserve"


# ---------------------------------------------------------------------------
# OHLCV parsing
# ---------------------------------------------------------------------------

def test_parse_ohlcv_response_builds_a_candle_series():
    data = {
        "data": {
            "attributes": {
                "ohlcv_list": [
                    [1700000900, 1.02, 1.05, 1.01, 1.04, 5000],
                    [1700000000, 1.00, 1.03, 0.99, 1.02, 4000],
                ]
            }
        }
    }
    series = _parse_ohlcv_response(data, "TESTCOIN", Timeframe.M15)
    assert series is not None
    assert len(series) == 2
    # sorted oldest-first regardless of the input order (newest-first here)
    assert series.candles[0].timestamp < series.candles[1].timestamp
    assert series.candles[0].close == pytest.approx(1.02)
    assert series.candles[1].close == pytest.approx(1.04)


def test_parse_ohlcv_response_skips_malformed_rows_without_failing_the_whole_series():
    data = {
        "data": {
            "attributes": {
                "ohlcv_list": [
                    [1700000000, 1.0, 1.1, 0.9, 1.05, 1000],
                    ["not-a-timestamp", 1, 1, 1, 1, 1],
                    [1700000900],  # too few fields
                ]
            }
        }
    }
    series = _parse_ohlcv_response(data, "TESTCOIN", Timeframe.M15)
    assert series is not None
    assert len(series) == 1


def test_parse_ohlcv_response_returns_none_on_unrecognised_shape():
    assert _parse_ohlcv_response({"unexpected": "shape"}, "TESTCOIN", Timeframe.M15) is None


def test_parse_ohlcv_response_returns_none_on_empty_list():
    data = {"data": {"attributes": {"ohlcv_list": []}}}
    assert _parse_ohlcv_response(data, "TESTCOIN", Timeframe.M15) is None


# ---------------------------------------------------------------------------
# fetch_candles orchestration
# ---------------------------------------------------------------------------

async def test_fetch_candles_returns_none_for_an_unmapped_chain():
    result = await fetch_candles("some_unmapped_chain", "Addr111", "TESTCOIN", Timeframe.M15, 100)
    assert result is None


async def test_fetch_candles_returns_none_when_no_pool_is_found(monkeypatch):
    def get_impl(url, params, headers):
        return _FakeResponse({"data": []})

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    result = await fetch_candles("solana", "Addr111", "TESTCOIN", Timeframe.M15, 100)
    assert result is None


async def test_fetch_candles_full_round_trip(monkeypatch):
    calls = []

    def get_impl(url, params, headers):
        calls.append(url)
        if "/pools/" not in url:
            return _FakeResponse({"data": [{"attributes": {"address": "BestPool", "reserve_in_usd": "9999"}}]})
        return _FakeResponse({
            "data": {"attributes": {"ohlcv_list": [[1700000000, 1.0, 1.1, 0.9, 1.05, 1000]]}}
        })

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    series = await fetch_candles("solana", "Addr111", "TESTCOIN", Timeframe.M15, 100)
    assert series is not None
    assert len(series) == 1
    assert any("/tokens/Addr111/pools" in u for u in calls)
    assert any("/pools/BestPool/ohlcv/minute" in u for u in calls)


def test_every_mapped_chain_has_a_non_empty_network_slug():
    for chain, network in CHAIN_TO_GECKOTERMINAL_NETWORK.items():
        assert network, chain
