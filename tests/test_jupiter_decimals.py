"""Tests for the decimals-aware unit conversion in app/execution/jupiter.py -
the fix for the documented bug where entry price was computed from raw
mixed-decimal base units instead of whole tokens/USD.
"""
import httpx
import pytest

from app.config import settings
from app.execution.jupiter import (
    USDC_DECIMALS,
    _decimals_cache,
    from_base_units,
    get_mint_decimals,
    to_base_units,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _clear_cache():
    _decimals_cache.clear()
    yield
    _decimals_cache.clear()


def test_to_base_units_and_back_round_trips_for_a_9_decimal_token():
    raw = to_base_units(1234.5, decimals=9)
    assert raw == 1234500000000
    assert from_base_units(raw, decimals=9) == pytest.approx(1234.5)


def test_to_base_units_and_back_round_trips_for_usdc_6_decimals():
    raw = to_base_units(20.0, decimals=6)
    assert raw == 20_000_000
    assert from_base_units(raw, decimals=6) == pytest.approx(20.0)


def test_from_base_units_on_a_9_decimal_token_is_not_the_same_as_6():
    """This is the actual bug being fixed: the same raw amount means a
    wildly different whole-token quantity depending on decimals - treating
    a 9-decimal token's raw units as if they were 6-decimal (or vice versa)
    is off by exactly 1000x, which is precisely the corruption the old code
    produced."""
    raw = 5_000_000_000  # 5 billion raw units
    as_9_decimals = from_base_units(raw, decimals=9)
    as_6_decimals = from_base_units(raw, decimals=6)
    assert as_9_decimals == pytest.approx(5.0)
    assert as_6_decimals == pytest.approx(5000.0)
    assert as_6_decimals == pytest.approx(as_9_decimals * 1000)


async def test_get_mint_decimals_parses_a_realistic_rpc_response(monkeypatch):
    calls = []

    class FakeResponse:
        # app/services/http.py inspects status_code before calling
        # raise_for_status, so a double without one never gets that far.
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "jsonrpc": "2.0",
                "result": {
                    "context": {"slot": 123},
                    "value": {
                        "data": {
                            "parsed": {
                                "info": {"decimals": 9, "isInitialized": True, "supply": "1000000000000"},
                                "type": "mint",
                            },
                            "program": "spl-token",
                        }
                    },
                },
                "id": 1,
            }

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

        async def post(self, url, json):
            calls.append((url, json))
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    decimals = await get_mint_decimals("SomeMintAddress111", rpc_url="https://fake-rpc")
    assert decimals == 9
    assert calls[0][1]["method"] == "getAccountInfo"
    assert calls[0][1]["params"][0] == "SomeMintAddress111"


async def test_get_mint_decimals_caches_after_first_lookup(monkeypatch):
    call_count = 0

    class FakeResponse:
        # app/services/http.py inspects status_code before calling
        # raise_for_status, so a double without one never gets that far.
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"value": {"data": {"parsed": {"info": {"decimals": 6}}}}}}

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

        async def post(self, url, json):
            nonlocal call_count
            call_count += 1
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    await get_mint_decimals("CachedMint111", rpc_url="https://fake-rpc")
    await get_mint_decimals("CachedMint111", rpc_url="https://fake-rpc")
    assert call_count == 1


async def test_get_mint_decimals_short_circuits_for_the_quote_mint():
    """USDC's own decimals are a known constant - no RPC round trip needed."""
    decimals = await get_mint_decimals(settings.QUOTE_MINT)
    assert decimals == USDC_DECIMALS


async def test_get_mint_decimals_raises_a_clear_error_on_an_unrecognised_shape(monkeypatch):
    class FakeResponse:
        # app/services/http.py inspects status_code before calling
        # raise_for_status, so a double without one never gets that far.
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"value": None}}  # e.g. account not found

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

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(ValueError, match="could not read decimals"):
        await get_mint_decimals("BadMint111", rpc_url="https://fake-rpc")


async def test_buy_fails_closed_when_decimals_lookup_fails(monkeypatch):
    """A buy must not fall back to guessing decimals - failing the decimals
    lookup must fail the whole order, the same fail-closed rule every other
    piece of this bot follows for missing data."""
    from app.execution.jupiter import JupiterExecutionClient

    async def broken_decimals(mint, rpc_url=None):
        raise ValueError("mint not found")

    monkeypatch.setattr("app.execution.jupiter.get_mint_decimals", broken_decimals)

    result = await JupiterExecutionClient().buy("SomeMint111", 20.0, 150)
    assert not result.success
    assert "decimals" in result.error


async def test_sell_converts_whole_token_qty_to_base_units_before_quoting(monkeypatch):
    """sell() receives qty in WHOLE tokens (matching every other backend's
    contract) and must convert to raw base units using the token's own
    decimals before it ever reaches Jupiter's quote API - passing the whole
    quantity straight through as if it were already raw units was the other
    half of the original bug."""
    from app.execution.jupiter import JupiterExecutionClient

    async def fake_decimals(mint, rpc_url=None):
        return 9

    seen_amount = {}

    async def fake_quote(self, input_mint, output_mint, amount, slippage_bps):
        seen_amount["amount"] = amount
        return {"inAmount": str(amount), "outAmount": "20000000"}

    monkeypatch.setattr("app.execution.jupiter.get_mint_decimals", fake_decimals)
    monkeypatch.setattr(JupiterExecutionClient, "_get_quote", fake_quote)
    settings.LIVE_TRADING = False  # stop short of the sign/submit path

    await JupiterExecutionClient().sell("SomeMint111", qty=1234.5, slippage_bps=150)
    assert seen_amount["amount"] == to_base_units(1234.5, decimals=9)
