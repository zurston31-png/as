"""Tests for app/scanner/discovery.py parsing.

Same standing caveat as tests/test_live_provider.py: these prove the
PARSING is correct against the API shape the module assumes, not that the
shape itself is right (that needs scripts/scan_once.py against a real
server). The value here is that a shape mismatch degrades to "no tokens"
rather than to malformed ones that reach the trading path.
"""
import datetime as dt

import httpx
import pytest

from app.config import settings
from app.scanner.discovery import (
    _addresses_from_profile_payload,
    _best_pair,
    _token_from_pair,
    discover_birdeye,
    discover_tokens,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _pair(address="Addr1", symbol="COIN", liquidity=50_000, chain="solana"):
    return {
        "chainId": chain,
        "baseToken": {"address": address, "symbol": symbol},
        "liquidity": {"usd": liquidity},
        "volume": {"h24": 120_000},
        "txns": {"h24": {"buys": 200, "sells": 150}},
        "priceUsd": "0.0123",
        "priceChange": {"h1": 4.2, "h24": 33.0},
        "pairCreatedAt": int(
            (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).timestamp() * 1000
        ),
    }


def _fake_client(get_impl):
    class FakeResponse:
        """Stands in for httpx.Response. Needs status_code and headers as
        well as json(), because requests now go through
        app/services/http.py's rate-limit-aware wrapper, which inspects
        both to decide whether to retry."""

        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200
            self.headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None, params=None):
            return FakeResponse(get_impl(url, headers, params))

    return FakeAsyncClient


# ---------------------------------------------------------------------------
# payload parsing
# ---------------------------------------------------------------------------

def test_addresses_from_profile_payload_filters_by_chain():
    payload = [
        {"chainId": "solana", "tokenAddress": "SolAddr1"},
        {"chainId": "ethereum", "tokenAddress": "EthAddr1"},
        {"chainId": "solana", "tokenAddress": "SolAddr2"},
    ]
    assert _addresses_from_profile_payload(payload, "solana") == ["SolAddr1", "SolAddr2"]


def test_addresses_from_profile_payload_handles_a_non_list():
    assert _addresses_from_profile_payload({"unexpected": "shape"}, "solana") == []


def test_addresses_from_profile_payload_skips_entries_without_an_address():
    payload = [{"chainId": "solana"}, {"chainId": "solana", "tokenAddress": "Good1"}]
    assert _addresses_from_profile_payload(payload, "solana") == ["Good1"]


def test_best_pair_picks_deepest_liquidity():
    pairs = [_pair("A", liquidity=100), _pair("B", liquidity=90_000), _pair("C", liquidity=5_000)]
    assert _best_pair(pairs)["baseToken"]["address"] == "B"


def test_token_from_pair_maps_every_market_field():
    token = _token_from_pair(_pair(), source="dexscreener")
    assert token is not None
    assert token.symbol == "COIN"
    assert token.liquidity_usd == pytest.approx(50_000)
    assert token.volume_24h_usd == pytest.approx(120_000)
    assert token.buys_24h == 200
    assert token.sells_24h == 150
    assert token.price_usd == pytest.approx(0.0123)
    assert token.age_hours == pytest.approx(48, abs=1)


def test_token_from_pair_returns_none_without_an_address():
    assert _token_from_pair({"baseToken": {"symbol": "NOADDR"}}, source="dexscreener") is None


def test_token_from_pair_leaves_missing_fields_as_none_not_zero():
    """A token whose volume simply wasn't reported must be distinguishable
    from one with genuinely zero volume - the pre-screen treats those very
    differently (reject-as-unknown vs reject-as-dead)."""
    bare = {"chainId": "solana", "baseToken": {"address": "A", "symbol": "X"}}
    token = _token_from_pair(bare, source="dexscreener")
    assert token.volume_24h_usd is None
    assert token.liquidity_usd is None
    assert token.pair_created_at is None


# ---------------------------------------------------------------------------
# source orchestration
# ---------------------------------------------------------------------------

async def test_birdeye_is_skipped_without_an_api_key(monkeypatch):
    monkeypatch.setattr(settings, "BIRDEYE_API_KEY", None)
    assert await discover_birdeye("solana") == []


async def test_birdeye_is_skipped_on_a_non_solana_chain(monkeypatch):
    monkeypatch.setattr(settings, "BIRDEYE_API_KEY", "fake-key")
    assert await discover_birdeye("ethereum") == []


async def test_discover_tokens_deduplicates_across_sources(monkeypatch):
    monkeypatch.setattr(settings, "BIRDEYE_API_KEY", "fake-key")

    def get_impl(url, headers, params):
        if "token-profiles" in url or "token-boosts" in url:
            return [{"chainId": "solana", "tokenAddress": "Shared1"}]
        if "new_listing" in url:
            return {"data": {"items": [{"address": "Shared1"}]}}
        # the hydration endpoint
        return {"pairs": [_pair(address="Shared1", symbol="SHARED")]}

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    tokens = await discover_tokens("solana")
    assert len(tokens) == 1
    assert tokens[0].token_address == "Shared1"


async def test_discover_tokens_survives_one_source_failing(monkeypatch):
    """A Birdeye outage must not take DexScreener's results down with it."""
    monkeypatch.setattr(settings, "BIRDEYE_API_KEY", "fake-key")

    def get_impl(url, headers, params):
        if "new_listing" in url:
            raise RuntimeError("birdeye is down")
        if "token-profiles" in url or "token-boosts" in url:
            return [{"chainId": "solana", "tokenAddress": "Alive1"}]
        return {"pairs": [_pair(address="Alive1", symbol="ALIVE")]}

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    tokens = await discover_tokens("solana")
    assert [t.token_address for t in tokens] == ["Alive1"]


async def test_discover_tokens_returns_empty_on_unrecognised_shapes(monkeypatch):
    def get_impl(url, headers, params):
        return {"totally": "unexpected"}

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(get_impl))
    assert await discover_tokens("solana") == []
