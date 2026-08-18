"""EVM execution via the 1inch aggregator (https://portal.1inch.dev/documentation/swap/swagger).

Quote -> build calldata -> sign -> submit, using web3.py only for the parts
that actually need it (signing and submission). Like
app/execution/jupiter.py, SwapResult is always in whole-token/USD units -
every conversion crosses the raw-base-units boundary through each token's
actual on-chain decimals (get_erc20_decimals), never assumed. Reading
decimals is a single ERC20 `decimals()` eth_call and needs nothing beyond
plain JSON-RPC over httpx (function selector 0x313ce567, no arguments,
returns one uint8) - the same reason app/execution/jupiter.py's mint-decimals
lookup doesn't need solana-py either. That keeps the live-execution gates
(LIVE_TRADING, LIVE_EXECUTION_ACKNOWLEDGED) as the ONLY place web3 itself
gets imported, instead of duplicating "are we actually armed" checks across
every function that happens to need a decimals read.

Requires EVM_QUOTE_TOKEN_ADDRESS to be set to an ERC20 stablecoin on your
target chain (e.g. USDC). Native-currency-denominated buying (paying in raw
ETH/BNB/etc rather than a stablecoin) is deliberately NOT implemented:
pricing it correctly needs to distinguish the native-currency sentinel
address 1inch uses in swap params from the WRAPPED native token address
price feeds actually index, and that distinction isn't worth the added
complexity for what this bot needs - a stablecoin quote leg sidesteps it
entirely, the same way app/execution/jupiter.py always trades against USDC.

Deliberately does NOT implement mempool resubmission/replacement (a stuck
transaction with too-low gas just times out - see
WAIT_FOR_RECEIPT_TIMEOUT_SECONDS) or MEV-aware submission (no private
mempool / Flashbots-style relay). Both are real production concerns and a
project of their own - a bare-minimum signer is not the same claim as a
production-grade one, and this file says so rather than pretending
otherwise. What IS implemented (EIP-1559 gas pricing, nonce management via
eth_getTransactionCount, decimals-correct accounting) is unit-tested
against a fake/mocked provider in tests/test_evm_execution.py; the actual
sign-and-submit path has never been exercised against a funded wallet or
mainnet from this project - same caveat app/execution/jupiter.py carries,
and for the same reason: it cannot be, without moving real money.

Written against the documented web3.py v7 API surface (AsyncWeb3,
snake_case SignedTransaction.raw_transaction) since requirements-live.txt
pins web3==7.2.0, but not exercised in this sandbox - web3 is not
installed here, and there is no funded wallet/RPC to test against even if
it were. If web3.py raises an AttributeError on the signed-transaction
field, check that library's changelog for the installed version; attribute
naming there has changed between major versions before (rawTransaction in
v5/v6 vs raw_transaction in v7).
"""
import logging

import httpx

from app.config import settings
from app.execution.base import ExecutionClient, SwapResult
from app.services import price_feed

logger = logging.getLogger(__name__)

WAIT_FOR_RECEIPT_TIMEOUT_SECONDS = 120
# maxFeePerGas headroom over the current base fee, so the transaction
# doesn't stall if the base fee rises across a block or two before
# inclusion - a common, conservative default (some wallets use 2x as well).
BASE_FEE_BUFFER_MULTIPLIER = 2.0

ERC20_DECIMALS_SELECTOR = "0x313ce567"  # decimals() function selector

LIVE_EXECUTION_UNACKNOWLEDGED_MSG = (
    "EVM live execution requires LIVE_EXECUTION_ACKNOWLEDGED=true in addition to "
    "LIVE_TRADING=true. This code path signs and submits real on-chain swaps, "
    "implements standard EIP-1559 gas pricing and nonce management but NOT mempool "
    "resubmission or MEV-aware submission, and has only been tested against a mocked "
    "provider, never a funded wallet - review app/execution/evm.py yourself before "
    "setting the flag, and start with a small trade size. LIVE_TRADING=false keeps "
    "paper trading unaffected either way."
)

_decimals_cache: dict[str, int] = {}


def compute_eip1559_fees(
    base_fee_per_gas: int, max_priority_fee_per_gas: int, *, buffer_multiplier: float = BASE_FEE_BUFFER_MULTIPLIER
) -> tuple[int, int]:
    """(maxFeePerGas, maxPriorityFeePerGas) from the chain's current base fee
    and a priority-fee suggestion. maxFeePerGas = base_fee * buffer + tip,
    the standard headroom formula (see BASE_FEE_BUFFER_MULTIPLIER)."""
    max_fee = int(base_fee_per_gas * buffer_multiplier) + int(max_priority_fee_per_gas)
    return max_fee, int(max_priority_fee_per_gas)


def to_base_units(whole_amount: float, decimals: int) -> int:
    return int(round(whole_amount * (10 ** decimals)))


def from_base_units(raw_amount: int, decimals: int) -> float:
    return raw_amount / (10 ** decimals)


def _quote_token_address() -> str:
    if not settings.EVM_QUOTE_TOKEN_ADDRESS:
        raise ValueError(
            "EVM_QUOTE_TOKEN_ADDRESS is not set - live EVM execution needs an ERC20 "
            "stablecoin address (e.g. USDC) on your target chain to trade against"
        )
    return settings.EVM_QUOTE_TOKEN_ADDRESS


async def get_erc20_decimals(token_address: str, rpc_url: str | None = None) -> int:
    """ERC20 `decimals()` via a plain eth_call - cached per address for the
    life of the process, since a token's decimals are immutable once
    deployed. Needs nothing beyond JSON-RPC over httpx, so it works even
    when the `web3` package isn't installed."""
    if token_address in _decimals_cache:
        return _decimals_cache[token_address]

    rpc_url = rpc_url or settings.EVM_RPC_URL
    if not rpc_url:
        raise ValueError("EVM_RPC_URL not configured")

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            rpc_url,
            json={
                "jsonrpc": "2.0", "id": 1, "method": "eth_call",
                "params": [{"to": token_address, "data": ERC20_DECIMALS_SELECTOR}, "latest"],
            },
        )
        resp.raise_for_status()
        data = resp.json()

    result_hex = data.get("result")
    if not result_hex or result_hex == "0x":
        raise ValueError(f"could not read decimals for {token_address} from RPC response: {data}")

    decimals = int(result_hex, 16)
    _decimals_cache[token_address] = decimals
    return decimals


async def _get_1inch_swap_tx(src: str, dst: str, amount: int, from_address: str, slippage_bps: int) -> dict:
    if not settings.ONEINCH_API_KEY:
        raise ValueError("ONEINCH_API_KEY is not set - api.1inch.dev requires an API key")

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{settings.ONEINCH_API_BASE}/{settings.EVM_CHAIN_ID}/swap",
            params={
                "src": src,
                "dst": dst,
                "amount": str(amount),
                "from": from_address,
                "slippage": slippage_bps / 100,  # 1inch wants a percent, not bps
                "disableEstimate": "true",
            },
            headers={"Authorization": f"Bearer {settings.ONEINCH_API_KEY}"},
        )
        resp.raise_for_status()
        return resp.json()


class EvmExecutionClient(ExecutionClient):
    async def get_price(self, instrument: str) -> float | None:
        return await price_feed.get_price_usd(instrument)

    async def _execute_swap(self, src: str, dst: str, amount: int, slippage_bps: int) -> SwapResult:
        if not settings.LIVE_TRADING:
            return SwapResult(success=False, error="LIVE_TRADING is false; refusing to submit an on-chain swap")
        if not settings.LIVE_EXECUTION_ACKNOWLEDGED:
            return SwapResult(success=False, error=LIVE_EXECUTION_UNACKNOWLEDGED_MSG)

        try:
            from web3 import AsyncWeb3
            from web3.providers import AsyncHTTPProvider
        except ImportError as exc:
            return SwapResult(
                success=False,
                error=f"live trading deps missing - run `pip install -r requirements-live.txt`: {exc}",
            )

        if not settings.EVM_RPC_URL:
            return SwapResult(success=False, error="EVM_RPC_URL not configured")
        if not settings.EVM_PRIVATE_KEY:
            return SwapResult(success=False, error="EVM_PRIVATE_KEY not configured")

        try:
            src_decimals = await get_erc20_decimals(src)
            dst_decimals = await get_erc20_decimals(dst)
        except (httpx.HTTPError, ValueError) as exc:
            return SwapResult(success=False, error=f"could not read token decimals: {exc}")

        w3 = AsyncWeb3(AsyncHTTPProvider(settings.EVM_RPC_URL))
        account = w3.eth.account.from_key(settings.EVM_PRIVATE_KEY)

        try:
            swap_data = await _get_1inch_swap_tx(src, dst, amount, account.address, slippage_bps)
        except Exception as exc:  # noqa: BLE001
            return SwapResult(success=False, error=f"1inch swap request failed: {exc}")

        tx = swap_data.get("tx")
        if not tx:
            return SwapResult(success=False, error=f"1inch response had no tx: {swap_data}")

        nonce = await w3.eth.get_transaction_count(account.address, "pending")
        latest_block = await w3.eth.get_block("latest")
        base_fee = latest_block.get("baseFeePerGas")
        if base_fee is None:
            return SwapResult(success=False, error="chain does not report baseFeePerGas (pre-EIP-1559?)")
        priority_fee = await w3.eth.max_priority_fee
        max_fee, max_priority = compute_eip1559_fees(base_fee, priority_fee)

        full_tx = {
            "from": w3.to_checksum_address(tx["from"]),
            "to": w3.to_checksum_address(tx["to"]),
            "data": tx["data"],
            "value": int(tx.get("value", 0)),
            "gas": int(tx.get("gas", 300_000)),
            "nonce": nonce,
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": max_priority,
            "chainId": settings.EVM_CHAIN_ID,
            "type": 2,
        }
        signed = account.sign_transaction(full_tx)

        try:
            tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=WAIT_FOR_RECEIPT_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return SwapResult(success=False, error=f"transaction submission failed: {exc}")

        if receipt.get("status") != 1:
            return SwapResult(success=False, error=f"transaction reverted: {tx_hash.hex()}")

        dst_amount_raw = int(swap_data.get("dstAmount", 0))
        out_whole = from_base_units(dst_amount_raw, dst_decimals)
        in_whole = from_base_units(amount, src_decimals)
        avg_price = (in_whole / out_whole) if out_whole else 0.0
        return SwapResult(success=True, filled_qty=out_whole, avg_price=avg_price, tx_hash=tx_hash.hex())

    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult:
        try:
            quote_token = _quote_token_address()
        except ValueError as exc:
            return SwapResult(success=False, error=str(exc))
        try:
            quote_decimals = await get_erc20_decimals(quote_token)
        except (httpx.HTTPError, ValueError) as exc:
            return SwapResult(success=False, error=f"could not read decimals for quote token: {exc}")

        amount_units = to_base_units(usd_amount, quote_decimals)
        return await self._execute_swap(quote_token, instrument, amount_units, slippage_bps)

    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult:
        try:
            quote_token = _quote_token_address()
        except ValueError as exc:
            return SwapResult(success=False, error=str(exc))
        try:
            instrument_decimals = await get_erc20_decimals(instrument)
        except (httpx.HTTPError, ValueError) as exc:
            return SwapResult(success=False, error=f"could not read decimals for {instrument}: {exc}")

        amount_units = to_base_units(qty, instrument_decimals)
        return await self._execute_swap(instrument, quote_token, amount_units, slippage_bps)
