"""Solana execution via the Jupiter aggregator (https://station.jup.ag/docs/apis/swap-api).

Quantities and prices on SwapResult are in the same units every other
execution backend uses: filled_qty is WHOLE tokens, avg_price is USD per
whole token — matching app/execution/paper.py's contract, which the rest
of the system (RiskManager, the position monitor, the dashboard) assumes
everywhere it does `qty * price`. Jupiter itself speaks only in each
mint's raw base units, so every conversion crosses that boundary here via
the mint's on-chain decimals (fetched once per mint via Solana RPC
`getAccountInfo`, cached in-process since a mint's decimals never change).

This used to be a documented, deliberate bug: avg_price was computed as
raw inAmount/outAmount, which only equals USD-per-whole-token when the
output token happens to have exactly 6 decimals (USDC's own). Solana
memecoins commonly use 9, which made the stored entry price 1000x too low
and closed every live position on the first monitor tick at a fabricated
profit. Fixed by converting through decimals at both boundaries -
to_base_units / from_base_units are the only two functions that do that
math, so there is exactly one place to audit it.

Endpoints/params match the Jupiter v6 Quote/Swap API as of this build.
Jupiter has changed API hosts/auth requirements before (e.g. adding paid
tiers) — if requests start failing, check https://station.jup.ag/docs and
update JUPITER_API_BASE / JUPITER_PRICE_API_BASE in .env accordingly.
"""
import base64
import logging

import httpx

from app.config import settings
from app.execution.base import ExecutionClient, SwapResult

logger = logging.getLogger(__name__)

USDC_DECIMALS = 6

# Solana execution has never been exercised against a funded wallet or
# mainnet from this codebase's own test suite - it can't be, without moving
# real money. The unit-conversion math below (decimals fetch, base-unit
# round-tripping, avg_price computation) is unit-tested against mocked RPC
# responses; the sign-and-submit path itself (solders/solana-py) is not,
# and never can be from an automated test. Requiring this ADDITIONAL
# explicit flag - separate from LIVE_TRADING - means enabling it is a
# second deliberate decision, not a side effect of flipping one setting,
# and the deployer has seen this warning before their first real swap.
LIVE_EXECUTION_UNACKNOWLEDGED_MSG = (
    "Jupiter live execution requires LIVE_EXECUTION_ACKNOWLEDGED=true in addition "
    "to LIVE_TRADING=true. This code path signs and submits real on-chain swaps "
    "and has only been tested against mocked responses, never a funded wallet - "
    "review app/execution/jupiter.py yourself before setting the flag, and start "
    "with a small trade size. EXECUTION_BACKEND=cex uses a mature, widely-used "
    "client library (ccxt) instead if you'd rather avoid that. LIVE_TRADING=false "
    "keeps paper trading unaffected either way."
)

_decimals_cache: dict[str, int] = {}


async def get_mint_decimals(mint: str, rpc_url: str | None = None) -> int:
    """Look up an SPL mint's decimals via Solana JSON-RPC `getAccountInfo`.

    Cached per mint for the life of the process - decimals are immutable
    once a mint is created, so there is never a reason to refetch one.
    """
    if mint in _decimals_cache:
        return _decimals_cache[mint]
    if mint == settings.QUOTE_MINT:
        _decimals_cache[mint] = USDC_DECIMALS
        return USDC_DECIMALS

    rpc_url = rpc_url or settings.SOLANA_RPC_URL
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            rpc_url,
            json={
                "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                "params": [mint, {"encoding": "jsonParsed"}],
            },
        )
        resp.raise_for_status()
        data = resp.json()

    try:
        decimals = data["result"]["value"]["data"]["parsed"]["info"]["decimals"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"could not read decimals for mint {mint} from RPC response: {data}") from exc

    decimals = int(decimals)
    _decimals_cache[mint] = decimals
    return decimals


def to_base_units(whole_amount: float, decimals: int) -> int:
    """Whole-token amount -> raw base units, the integer Jupiter's API expects."""
    return int(round(whole_amount * (10 ** decimals)))


def from_base_units(raw_amount: int, decimals: int) -> float:
    """Raw base units -> whole-token amount, the unit every other execution
    backend's SwapResult uses."""
    return raw_amount / (10 ** decimals)


class JupiterExecutionClient(ExecutionClient):
    def __init__(self):
        self.base = settings.JUPITER_API_BASE
        self.price_base = settings.JUPITER_PRICE_API_BASE
        self.quote_mint = settings.QUOTE_MINT

    async def get_price(self, instrument: str) -> float | None:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.price_base}/price", params={"ids": instrument})
            resp.raise_for_status()
            data = resp.json()
        entry = (data.get("data") or {}).get(instrument)
        return float(entry["price"]) if entry else None

    async def _get_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> dict:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base}/quote",
                params={
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": int(amount),
                    "slippageBps": slippage_bps,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def _execute_swap(self, quote: dict, input_decimals: int, output_decimals: int) -> SwapResult:
        if not settings.LIVE_TRADING:
            return SwapResult(success=False, error="LIVE_TRADING is false; refusing to submit an on-chain swap")
        if not settings.LIVE_EXECUTION_ACKNOWLEDGED:
            return SwapResult(success=False, error=LIVE_EXECUTION_UNACKNOWLEDGED_MSG)

        try:
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TxOpts
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction
        except ImportError as exc:
            return SwapResult(
                success=False,
                error=f"live trading deps missing - run `pip install -r requirements-live.txt`: {exc}",
            )

        if not settings.SOLANA_PRIVATE_KEY:
            return SwapResult(success=False, error="SOLANA_PRIVATE_KEY not configured")

        keypair = Keypair.from_base58_string(settings.SOLANA_PRIVATE_KEY)

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{self.base}/swap",
                json={
                    "quoteResponse": quote,
                    "userPublicKey": str(keypair.pubkey()),
                    "wrapAndUnwrapSol": True,
                    "dynamicComputeUnitLimit": True,
                    "prioritizationFeeLamports": "auto",
                },
            )
            resp.raise_for_status()
            swap_data = resp.json()

        tx_bytes = base64.b64decode(swap_data["swapTransaction"])
        unsigned_tx = VersionedTransaction.from_bytes(tx_bytes)
        signed_tx = VersionedTransaction(unsigned_tx.message, [keypair])

        async with AsyncClient(settings.SOLANA_RPC_URL) as rpc:
            send_result = await rpc.send_raw_transaction(bytes(signed_tx), opts=TxOpts(skip_preflight=False))
            signature = send_result.value
            await rpc.confirm_transaction(signature, commitment="confirmed")

        out_whole = from_base_units(int(quote["outAmount"]), output_decimals)
        in_whole = from_base_units(int(quote["inAmount"]), input_decimals)
        avg_price = (in_whole / out_whole) if out_whole else 0.0
        return SwapResult(success=True, filled_qty=out_whole, avg_price=avg_price, tx_hash=str(signature))

    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult:
        try:
            output_decimals = await get_mint_decimals(instrument)
        except (httpx.HTTPError, ValueError) as exc:
            return SwapResult(success=False, error=f"could not read decimals for {instrument}: {exc}")

        amount_units = to_base_units(usd_amount, USDC_DECIMALS)
        try:
            quote = await self._get_quote(self.quote_mint, instrument, amount_units, slippage_bps)
        except httpx.HTTPError as exc:
            return SwapResult(success=False, error=f"Jupiter quote failed: {exc}")
        if "error" in quote:
            return SwapResult(success=False, error=f"Jupiter quote error: {quote['error']}")
        return await self._execute_swap(quote, input_decimals=USDC_DECIMALS, output_decimals=output_decimals)

    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult:
        try:
            input_decimals = await get_mint_decimals(instrument)
        except (httpx.HTTPError, ValueError) as exc:
            return SwapResult(success=False, error=f"could not read decimals for {instrument}: {exc}")

        amount_units = to_base_units(qty, input_decimals)
        try:
            quote = await self._get_quote(instrument, self.quote_mint, amount_units, slippage_bps)
        except httpx.HTTPError as exc:
            return SwapResult(success=False, error=f"Jupiter quote failed: {exc}")
        if "error" in quote:
            return SwapResult(success=False, error=f"Jupiter quote error: {quote['error']}")
        return await self._execute_swap(quote, input_decimals=input_decimals, output_decimals=USDC_DECIMALS)
