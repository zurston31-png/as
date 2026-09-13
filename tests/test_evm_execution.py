"""Tests for app/execution/evm.py's unit-conversion and fee math - the part
of the EVM backend that's fully testable without the `web3` package or a
real RPC/wallet (the sign-and-submit path itself needs both and can't be
exercised here, same limitation app/execution/jupiter.py has).
"""
import httpx
import pytest

from app.config import settings
from app.execution.evm import (
    _decimals_cache,
    compute_eip1559_fees,
    from_base_units,
    get_erc20_decimals,
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


# ---------------------------------------------------------------------------
# base-unit conversion (same contract as app/execution/jupiter.py)
# ---------------------------------------------------------------------------

def test_to_base_units_and_back_round_trips_for_an_18_decimal_token():
    raw = to_base_units(2.5, decimals=18)
    assert raw == 2_500_000_000_000_000_000
    assert from_base_units(raw, decimals=18) == pytest.approx(2.5)


def test_to_base_units_and_back_round_trips_for_a_6_decimal_stablecoin():
    raw = to_base_units(20.0, decimals=6)
    assert raw == 20_000_000
    assert from_base_units(raw, decimals=6) == pytest.approx(20.0)


def test_mismatched_decimals_would_be_off_by_a_huge_factor():
    """18 vs 6 decimals is a much larger gap than Solana's 9-vs-6 case -
    proof the same class of bug (treating raw units from one decimals
    convention as if they used another) would be even more catastrophic
    here if the conversion were ever skipped."""
    raw = 5_000_000_000_000_000_000  # 5 tokens at 18 decimals
    as_18 = from_base_units(raw, decimals=18)
    as_6 = from_base_units(raw, decimals=6)
    assert as_18 == pytest.approx(5.0)
    assert as_6 == pytest.approx(5_000_000_000_000.0)


# ---------------------------------------------------------------------------
# EIP-1559 fee computation
# ---------------------------------------------------------------------------

def test_compute_eip1559_fees_applies_the_base_fee_buffer():
    max_fee, max_priority = compute_eip1559_fees(base_fee_per_gas=10_000_000_000, max_priority_fee_per_gas=1_000_000_000)
    assert max_priority == 1_000_000_000
    assert max_fee == 10_000_000_000 * 2 + 1_000_000_000


def test_compute_eip1559_fees_custom_buffer_multiplier():
    max_fee, _ = compute_eip1559_fees(base_fee_per_gas=1_000, max_priority_fee_per_gas=100, buffer_multiplier=3.0)
    assert max_fee == 1_000 * 3 + 100


def test_compute_eip1559_fees_max_fee_always_covers_base_plus_priority():
    """maxFeePerGas must never be less than base_fee + priority_fee, or the
    network would reject the transaction outright."""
    max_fee, max_priority = compute_eip1559_fees(base_fee_per_gas=50, max_priority_fee_per_gas=5)
    assert max_fee >= 50 + max_priority


# ---------------------------------------------------------------------------
# get_erc20_decimals - plain eth_call over httpx, no web3 needed
# ---------------------------------------------------------------------------

async def test_get_erc20_decimals_parses_a_realistic_eth_call_response(monkeypatch):
    calls = []

    class FakeResponse:
        # app/services/http.py inspects status_code before calling
        # raise_for_status, so a double without one never gets that far.
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            # decimals()=18 -> uint8 18 left-padded to 32 bytes
            return {"jsonrpc": "2.0", "id": 1, "result": "0x" + "0" * 62 + "12"}

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

    decimals = await get_erc20_decimals("0x" + "1" * 40, rpc_url="https://fake-evm-rpc")
    assert decimals == 18
    assert calls[0][1]["method"] == "eth_call"
    assert calls[0][1]["params"][0]["to"] == "0x" + "1" * 40
    assert calls[0][1]["params"][0]["data"] == "0x313ce567"


async def test_get_erc20_decimals_caches_after_first_lookup(monkeypatch):
    call_count = 0

    class FakeResponse:
        # app/services/http.py inspects status_code before calling
        # raise_for_status, so a double without one never gets that far.
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "0x" + "0" * 62 + "06"}

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

    await get_erc20_decimals("0x" + "2" * 40, rpc_url="https://fake-evm-rpc")
    await get_erc20_decimals("0x" + "2" * 40, rpc_url="https://fake-evm-rpc")
    assert call_count == 1


async def test_get_erc20_decimals_raises_on_an_empty_result(monkeypatch):
    class FakeResponse:
        # app/services/http.py inspects status_code before calling
        # raise_for_status, so a double without one never gets that far.
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "0x"}  # e.g. call to a non-contract address

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
        await get_erc20_decimals("0x" + "3" * 40, rpc_url="https://fake-evm-rpc")


async def test_get_erc20_decimals_requires_an_rpc_url(monkeypatch):
    monkeypatch.setattr(settings, "EVM_RPC_URL", None)
    with pytest.raises(ValueError, match="EVM_RPC_URL"):
        await get_erc20_decimals("0x" + "4" * 40)


# ---------------------------------------------------------------------------
# buy()/sell() wiring - amount conversion happens before the live-execution
# gate, mirroring app/execution/jupiter.py's structure
# ---------------------------------------------------------------------------

async def test_buy_converts_usd_to_quote_token_base_units(monkeypatch):
    from app.execution.evm import EvmExecutionClient

    monkeypatch.setattr(settings, "EVM_QUOTE_TOKEN_ADDRESS", "0x" + "9" * 40)
    settings.LIVE_TRADING = False  # stop short of the sign/submit path

    async def fake_decimals(token, rpc_url=None):
        return 6

    seen = {}

    async def fake_execute(self, src, dst, amount, slippage_bps):
        seen.update(src=src, dst=dst, amount=amount)
        from app.execution.base import SwapResult
        return SwapResult(success=False, error="stopped before live gate")

    monkeypatch.setattr("app.execution.evm.get_erc20_decimals", fake_decimals)
    monkeypatch.setattr(EvmExecutionClient, "_execute_swap", fake_execute)

    await EvmExecutionClient().buy("0x" + "1" * 40, 25.0, 150)
    assert seen["src"] == "0x" + "9" * 40
    assert seen["dst"] == "0x" + "1" * 40
    assert seen["amount"] == to_base_units(25.0, decimals=6)


async def test_sell_converts_whole_token_qty_to_base_units(monkeypatch):
    from app.execution.evm import EvmExecutionClient

    monkeypatch.setattr(settings, "EVM_QUOTE_TOKEN_ADDRESS", "0x" + "9" * 40)
    settings.LIVE_TRADING = False

    async def fake_decimals(token, rpc_url=None):
        return 18

    seen = {}

    async def fake_execute(self, src, dst, amount, slippage_bps):
        seen.update(src=src, dst=dst, amount=amount)
        from app.execution.base import SwapResult
        return SwapResult(success=False, error="stopped before live gate")

    monkeypatch.setattr("app.execution.evm.get_erc20_decimals", fake_decimals)
    monkeypatch.setattr(EvmExecutionClient, "_execute_swap", fake_execute)

    await EvmExecutionClient().sell("0x" + "1" * 40, 42.5, 150)
    assert seen["src"] == "0x" + "1" * 40
    assert seen["dst"] == "0x" + "9" * 40
    assert seen["amount"] == to_base_units(42.5, decimals=18)


async def test_buy_fails_closed_when_decimals_lookup_fails(monkeypatch):
    from app.execution.evm import EvmExecutionClient

    monkeypatch.setattr(settings, "EVM_QUOTE_TOKEN_ADDRESS", "0x" + "9" * 40)

    async def broken_decimals(token, rpc_url=None):
        raise ValueError("no code at that address")

    monkeypatch.setattr("app.execution.evm.get_erc20_decimals", broken_decimals)

    result = await EvmExecutionClient().buy("0x" + "1" * 40, 25.0, 150)
    assert not result.success
    assert "decimals" in result.error
