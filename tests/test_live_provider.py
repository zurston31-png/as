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
    _token_side,
    fetch_candles,
)
from app.config import settings

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
    pool, side = await _find_primary_pool("solana", "TokenMint111")
    assert pool == "PoolB"
    # UPDATED, not loosened: _find_primary_pool now also reports which side
    # of the pool the token is on, because fetching a pool's candles
    # without knowing that returns the OTHER token's chart. This fixture
    # carries no relationships, so the side is honestly unknown.
    assert side is None


async def test_find_primary_pool_returns_none_on_empty_data(monkeypatch):
    def get_impl(url, params, headers):
        return _FakeResponse({"data": []})

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    assert await _find_primary_pool("solana", "TokenMint111") == (None, None)


async def test_find_primary_pool_returns_none_on_request_failure(monkeypatch):
    def get_impl(url, params, headers):
        return _FakeResponse({}, status_ok=False)

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    assert await _find_primary_pool("solana", "TokenMint111") == (None, None)


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
    pool, _side = await _find_primary_pool("solana", "TokenMint111")
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


# ---------------------------------------------------------------------------
# which token's candles are these? (app/data/live_provider.py:_token_side)
# ---------------------------------------------------------------------------
#
# These exist because of a real defect, not a hypothetical one. The OHLCV
# endpoint is keyed by POOL and defaults to the pool's BASE token, so for
# every token that sits on the QUOTE side of its own deepest pool the bot
# was scoring, and walking exits over, some other asset's chart - a
# well-formed series of the right length over the right window, for the
# wrong token. It surfaced as shadow exit prices clustering at major-asset
# price levels against memecoin entries of a fraction of a cent.

def _pool(address, reserve, *, base=None, quote=None):
    relationships = {}
    if base is not None:
        relationships["base_token"] = {"data": {"id": f"solana_{base}"}}
    if quote is not None:
        relationships["quote_token"] = {"data": {"id": f"solana_{quote}"}}
    return {
        "attributes": {"address": address, "reserve_in_usd": str(reserve)},
        "relationships": relationships,
    }


def test_token_side_identifies_the_base_side():
    pool = _pool("PoolA", 1000, base="MemeMint", quote="SolMint")
    assert _token_side(pool, "MemeMint") == "base"


def test_token_side_identifies_the_quote_side():
    """The case that was silently wrong: a SOL/MEME pool, where asking for
    the pool's candles returns SOL's price, not the memecoin's."""
    pool = _pool("PoolA", 1000, base="SolMint", quote="MemeMint")
    assert _token_side(pool, "MemeMint") == "quote"


def test_token_side_is_case_insensitive_about_the_address():
    """EVM addresses are routinely checksummed on one side of a system and
    lowercased on the other; a case mismatch must not read as 'not ours',
    which would downgrade a known side to an unverifiable one."""
    pool = _pool("PoolA", 1000, base="0xABCdef", quote="SolMint")
    assert _token_side(pool, "0xabcdef") == "base"


def test_token_side_is_none_when_the_pool_names_neither_token():
    pool = _pool("PoolA", 1000, base="SolMint", quote="UsdcMint")
    assert _token_side(pool, "MemeMint") is None


def test_token_side_is_none_when_there_are_no_relationships():
    assert _token_side({"attributes": {"address": "PoolA"}}, "MemeMint") is None


async def test_find_primary_pool_reports_the_side_it_found(monkeypatch):
    def get_impl(url, params, headers):
        return _FakeResponse({"data": [_pool("PoolB", 50000, base="SolMint", quote="MemeMint")]})

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    assert await _find_primary_pool("solana", "MemeMint") == ("PoolB", "quote")


# ---------------------------------------------------------------------------
# the flag that corrects it (GECKOTERMINAL_PIN_TOKEN_SIDE)
# ---------------------------------------------------------------------------

def _two_stage(seen):
    """A fake that answers the pools call, then the OHLCV call, recording
    the params each was given."""
    def get_impl(url, params, headers):
        seen.append((url, dict(params or {})))
        if "/pools/" in url and "/ohlcv/" in url:
            return _FakeResponse({
                "data": {"attributes": {"ohlcv_list": [[1_700_000_000, 1, 2, 0.5, 1.5, 10]]}}
            })
        return _FakeResponse({"data": [_pool("PoolB", 50000, base="SolMint", quote="MemeMint")]})
    return get_impl


async def test_fetch_candles_does_not_pin_the_token_by_default(monkeypatch):
    """Off by default on purpose: switching it on changes which series the
    live entry gate scores for every quote-side token, and therefore which
    trades get taken, without moving the strategy version hash. That is a
    collection boundary for an operator to draw deliberately."""
    seen = []
    monkeypatch.setattr(settings, "GECKOTERMINAL_PIN_TOKEN_SIDE", False)
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(_two_stage(seen)))

    series = await fetch_candles("solana", "MemeMint", "MEME", Timeframe.M15, 300)
    assert series is not None
    ohlcv = [params for url, params in seen if "/ohlcv/" in url][0]
    assert "token" not in ohlcv


async def test_fetch_candles_pins_our_token_when_the_flag_is_on(monkeypatch):
    seen = []
    monkeypatch.setattr(settings, "GECKOTERMINAL_PIN_TOKEN_SIDE", True)
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(_two_stage(seen)))

    series = await fetch_candles("solana", "MemeMint", "MEME", Timeframe.M15, 300)
    assert series is not None
    ohlcv = [params for url, params in seen if "/ohlcv/" in url][0]
    assert ohlcv["token"] == "MemeMint"


async def test_quote_side_token_is_logged_even_when_the_fix_is_off(caplog):
    """The flag controls the BEHAVIOUR, never the evidence. With it off the
    bot keeps using the wrong series - but it must say so, every time, or
    the defect stays invisible exactly as it did before."""
    import logging

    seen = []
    from app.data import live_provider

    original = settings.GECKOTERMINAL_PIN_TOKEN_SIDE
    settings.GECKOTERMINAL_PIN_TOKEN_SIDE = False
    saved = httpx.AsyncClient
    httpx.AsyncClient = _fake_client(_two_stage(seen))
    try:
        with caplog.at_level(logging.ERROR, logger=live_provider.__name__):
            await fetch_candles("solana", "MemeMint", "MEME", Timeframe.M15, 300)
    finally:
        httpx.AsyncClient = saved
        settings.GECKOTERMINAL_PIN_TOKEN_SIDE = original

    assert any("QUOTE token" in record.getMessage() for record in caplog.records)
