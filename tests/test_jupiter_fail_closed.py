"""The Jupiter live path must fail CLOSED, and report the right units.

SCOPE, stated plainly: this backend cannot run here. Both live flags are
false, there are no wallet keys, and nothing in this suite reaches a
Solana RPC. These tests drive `_execute_swap` with stubs, so they prove
the decision logic - which failures are caught, what a confirmed-but-
reverted transaction is treated as, and how the two legs map onto
`SwapResult` - and they prove nothing about behaviour against a real
chain. That distinction is the point: the defects below were all in the
decision logic, and all of them were reachable without any network at all.

The invariants:

  L1  Live execution stays refused while either live flag is off.
  L2  Every step that can throw is caught. A backend that raises out of
      `buy`/`sell` is a crash in the trading path, not a failed trade.
  L3  A signature obtained before a failure is always reported - without
      it nobody can find out whether the swap landed anyway.
  L4  CONFIRMED IS NOT SUCCEEDED. A transaction that reaches commitment
      with a non-None `err` reverted, and must not be recorded as a fill.
  L5  `filled_qty` is TOKENS and `avg_price` is USD per token on BOTH
      legs, matching what the paper engine returns and what
      trading_service computes with.
  L6  A fill derived from the quote is labelled as an estimate.
"""
import base64

import pytest

from app.config import settings
from app.execution import jupiter
from app.execution.base import SwapResult

pytestmark = pytest.mark.anyio

USDC = jupiter.USDC_DECIMALS      # 6
TOKEN_DECIMALS = 9


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def live(monkeypatch):
    """Both live flags on, so the refusal guards are out of the way and the
    execution logic itself is under test. Nothing here touches a network:
    every RPC and HTTP call is stubbed below."""
    monkeypatch.setattr(settings, "LIVE_TRADING", True)
    monkeypatch.setattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", True)
    monkeypatch.setattr(settings, "SOLANA_PRIVATE_KEY", "stub-key")


class _Status:
    def __init__(self, err=None):
        self.err = err


class _Confirmation:
    def __init__(self, err=None):
        self.value = [_Status(err)]


class _Send:
    def __init__(self, signature="SIG123"):
        self.value = signature


class _Rpc:
    """A stand-in AsyncClient. Each hook can be told to raise."""

    # A sentinel, because None is a value under test here - "no
    # confirmation response at all" is one of the shapes that must fail
    # closed, and `confirm or default` would quietly turn it into a clean
    # confirmation and pass the test for the wrong reason.
    _UNSET = object()

    def __init__(self, *, send=None, confirm=_UNSET, send_raises=None, confirm_raises=None):
        self._send = send or _Send()
        self._confirm = _Confirmation() if confirm is _Rpc._UNSET else confirm
        self._send_raises = send_raises
        self._confirm_raises = confirm_raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def send_raw_transaction(self, *_a, **_k):
        if self._send_raises:
            raise self._send_raises
        return self._send

    async def confirm_transaction(self, *_a, **_k):
        if self._confirm_raises:
            raise self._confirm_raises
        return self._confirm


def _install(monkeypatch, rpc, *, swap_data=None, sign_raises=None):
    """Stub the solana imports, the /swap build call and the signing."""
    import sys
    import types

    async def fake_post_json(*_a, **_k):
        return swap_data if swap_data is not None else {
            "swapTransaction": base64.b64encode(b"unsigned").decode()
        }

    monkeypatch.setattr(jupiter.http, "post_json", fake_post_json)

    class _Keypair:
        @staticmethod
        def from_base58_string(_s):
            kp = _Keypair()
            return kp

        def pubkey(self):
            return "PUBKEY"

    class _VersionedTransaction:
        def __init__(self, *_a):
            self.message = "msg"

        @staticmethod
        def from_bytes(_b):
            if sign_raises:
                raise sign_raises
            return _VersionedTransaction()

        def __bytes__(self):
            return b"signed"

    mods = {
        "solana": types.ModuleType("solana"),
        "solana.rpc": types.ModuleType("solana.rpc"),
        "solana.rpc.async_api": types.ModuleType("solana.rpc.async_api"),
        "solana.rpc.types": types.ModuleType("solana.rpc.types"),
        "solders": types.ModuleType("solders"),
        "solders.keypair": types.ModuleType("solders.keypair"),
        "solders.transaction": types.ModuleType("solders.transaction"),
    }
    mods["solana.rpc.async_api"].AsyncClient = lambda *_a, **_k: rpc
    mods["solana.rpc.types"].TxOpts = lambda **_k: None
    mods["solders.keypair"].Keypair = _Keypair
    mods["solders.transaction"].VersionedTransaction = _VersionedTransaction
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _quote(in_amount: int, out_amount: int) -> dict:
    return {"inAmount": str(in_amount), "outAmount": str(out_amount)}


async def _buy(client, quote):
    return await client._execute_swap(quote, input_decimals=USDC, output_decimals=TOKEN_DECIMALS)


async def _sell(client, quote):
    return await client._execute_swap(quote, input_decimals=TOKEN_DECIMALS, output_decimals=USDC)


@pytest.fixture
def client():
    return jupiter.JupiterExecutionClient()


# ---------------------------------------------------------------------------
# L1 - the flags still gate everything
# ---------------------------------------------------------------------------

async def test_execution_is_refused_while_live_trading_is_off(client, monkeypatch):
    monkeypatch.setattr(settings, "LIVE_TRADING", False)
    result = await _buy(client, _quote(10**6, 10**9))
    assert not result.success
    assert "LIVE_TRADING" in result.error


async def test_execution_is_refused_without_the_acknowledgement(client, monkeypatch):
    """The second flag exists so LIVE_TRADING alone cannot arm real orders."""
    monkeypatch.setattr(settings, "LIVE_TRADING", True)
    monkeypatch.setattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", False)
    result = await _buy(client, _quote(10**6, 10**9))
    assert not result.success


# ---------------------------------------------------------------------------
# L2/L3 - every step fails closed, and a signature is never lost
# ---------------------------------------------------------------------------

async def test_a_missing_swap_transaction_is_an_error_not_a_crash(client, live, monkeypatch):
    """This was a bare `swap_data["swapTransaction"]`: a KeyError out of the
    execution client, which the buy path does not catch."""
    _install(monkeypatch, _Rpc(), swap_data={"error": "no route"})
    result = await _buy(client, _quote(10**6, 10**9))
    assert not result.success
    assert "swapTransaction" in result.error


async def test_a_signing_failure_is_an_error_not_a_crash(client, live, monkeypatch):
    _install(monkeypatch, _Rpc(), sign_raises=ValueError("malformed transaction"))
    result = await _buy(client, _quote(10**6, 10**9))
    assert not result.success
    assert "decode or sign" in result.error
    assert result.tx_hash is None, "nothing was submitted, so there is no signature to report"


async def test_a_submission_failure_says_the_swap_may_still_have_landed(client, live, monkeypatch):
    """L3. A send that raises does not prove the transaction was not
    broadcast, and reconciling it as 'never happened' is how a real
    position becomes invisible."""
    _install(monkeypatch, _Rpc(send_raises=RuntimeError("rpc timeout")))
    result = await _buy(client, _quote(10**6, 10**9))
    assert not result.success
    assert "may still have been broadcast" in result.error


async def test_a_confirmation_failure_preserves_the_signature(client, live, monkeypatch):
    """L3, the case that matters most: the transaction IS out there and the
    signature is the only way to find out what happened to it."""
    _install(monkeypatch, _Rpc(confirm_raises=TimeoutError("unconfirmed after 90s")))
    result = await _buy(client, _quote(10**6, 10**9))
    assert not result.success
    assert result.tx_hash == "SIG123"
    assert "could not confirm" in result.error


# ---------------------------------------------------------------------------
# L4 - confirmed is not succeeded
# ---------------------------------------------------------------------------

async def test_a_confirmed_but_reverted_swap_is_a_failure(client, live, monkeypatch):
    """The fail-open this whole change is about. A slippage revert confirms
    at the requested commitment and carries a non-None err; the old code
    read 'confirmed' as 'filled' and booked a position the wallet never
    held."""
    _install(monkeypatch, _Rpc(confirm=_Confirmation(err={"InstructionError": [0, "Custom"]})))
    result = await _buy(client, _quote(10**6, 10**9))
    assert not result.success
    assert "reverted on-chain" in result.error
    assert result.tx_hash == "SIG123"


@pytest.mark.parametrize("confirmation", [None, object(), type("R", (), {"value": []})()])
async def test_an_unreadable_confirmation_fails_closed(client, live, monkeypatch, confirmation):
    """The response shape differs across client versions. On the live path,
    a shape we cannot read must be an error - treating 'I could not tell'
    as success is the same fail-open in a different costume."""
    _install(monkeypatch, _Rpc(confirm=confirmation))
    result = await _buy(client, _quote(10**6, 10**9))
    assert not result.success


async def test_a_clean_confirmation_succeeds(client, live, monkeypatch):
    """The gate has to be passable."""
    _install(monkeypatch, _Rpc(confirm=_Confirmation(err=None)))
    result = await _buy(client, _quote(10**6, 10**9))
    assert result.success
    assert result.tx_hash == "SIG123"


# ---------------------------------------------------------------------------
# L5 - units, on both legs
# ---------------------------------------------------------------------------

async def test_a_buy_reports_tokens_bought_and_usd_per_token(client, live, monkeypatch):
    """$100 USDC in, 2,000 tokens out => 2,000 @ $0.05."""
    _install(monkeypatch, _Rpc())
    result = await _buy(client, _quote(100 * 10**USDC, 2_000 * 10**TOKEN_DECIMALS))
    assert result.success
    assert result.filled_qty == pytest.approx(2_000.0)
    assert result.avg_price == pytest.approx(0.05)


async def test_a_sell_reports_tokens_sold_not_usdc_received(client, live, monkeypatch):
    """L5, the accounting corruption. 2,000 tokens in, $100 USDC out is
    still 2,000 @ $0.05 - the same shape the paper engine returns.

    The old code returned filled_qty=100 (the USDC leg) and avg_price=20
    (tokens per USDC, the inverse). trading_service computes proceeds as
    filled_qty * avg_price, so a $100 sale booked $2,000 of proceeds, and
    a partial exit subtracted 100 from a position holding 2,000 tokens.
    """
    _install(monkeypatch, _Rpc())
    result = await _sell(client, _quote(2_000 * 10**TOKEN_DECIMALS, 100 * 10**USDC))
    assert result.success
    assert result.filled_qty == pytest.approx(2_000.0), "filled_qty must be the TOKEN quantity"
    assert result.avg_price == pytest.approx(0.05), "avg_price must be USD per token"
    # ...so the proceeds arithmetic downstream comes out right.
    assert result.filled_qty * result.avg_price == pytest.approx(100.0)


async def test_the_two_legs_round_trip_consistently(client, live, monkeypatch):
    """Buying then selling the same quantity at the same price must produce
    the same qty and price on both sides, or P&L is nonsense."""
    _install(monkeypatch, _Rpc())
    bought = await _buy(client, _quote(100 * 10**USDC, 2_000 * 10**TOKEN_DECIMALS))
    sold = await _sell(client, _quote(2_000 * 10**TOKEN_DECIMALS, 100 * 10**USDC))
    assert bought.filled_qty == pytest.approx(sold.filled_qty)
    assert bought.avg_price == pytest.approx(sold.avg_price)


async def test_a_zero_quantity_quote_does_not_divide_by_zero(client, live, monkeypatch):
    _install(monkeypatch, _Rpc())
    result = await _buy(client, _quote(100 * 10**USDC, 0))
    assert result.success
    assert result.avg_price == 0.0


# ---------------------------------------------------------------------------
# L6 - an estimate is labelled as one
# ---------------------------------------------------------------------------

async def test_a_quote_derived_fill_is_marked_as_an_estimate(client, live, monkeypatch):
    """CLAUDE.md: never present something unmeasured as measured. These
    amounts are what the router EXPECTED before slippage, not what landed -
    reading the executed amounts needs the transaction's balance deltas,
    which this backend does not parse."""
    _install(monkeypatch, _Rpc())
    result = await _buy(client, _quote(100 * 10**USDC, 2_000 * 10**TOKEN_DECIMALS))
    assert result.fill_estimated_from_quote is True


def test_the_paper_engine_does_not_claim_its_fills_are_estimates():
    """The paper fill IS the simulated execution, so the flag stays off and
    the marker keeps meaning something."""
    assert SwapResult(success=True).fill_estimated_from_quote is False
