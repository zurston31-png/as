"""Solana execution via the Jupiter aggregator (https://station.jup.ag/docs/apis/swap-api).

Position quantities are tracked in the token's raw base units (exactly what
Jupiter's `outAmount`/`inAmount` report), which sidesteps needing to know
each token's decimals separately: a buy's `outAmount` becomes the qty
stored on the Position, and selling passes that same raw qty back in as
the swap `amount`.

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

    async def _execute_swap(self, quote: dict) -> SwapResult:
        if not settings.LIVE_TRADING:
            return SwapResult(success=False, error="LIVE_TRADING is false; refusing to submit an on-chain swap")

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

        out_amount = int(quote["outAmount"])
        in_amount = int(quote["inAmount"])
        avg_price = (in_amount / out_amount) if out_amount else 0.0
        return SwapResult(success=True, filled_qty=out_amount, avg_price=avg_price, tx_hash=str(signature))

    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult:
        amount_units = int(usd_amount * (10 ** USDC_DECIMALS))
        try:
            quote = await self._get_quote(self.quote_mint, instrument, amount_units, slippage_bps)
        except httpx.HTTPError as exc:
            return SwapResult(success=False, error=f"Jupiter quote failed: {exc}")
        if "error" in quote:
            return SwapResult(success=False, error=f"Jupiter quote error: {quote['error']}")
        return await self._execute_swap(quote)

    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult:
        try:
            quote = await self._get_quote(instrument, self.quote_mint, int(qty), slippage_bps)
        except httpx.HTTPError as exc:
            return SwapResult(success=False, error=f"Jupiter quote failed: {exc}")
        if "error" in quote:
            return SwapResult(success=False, error=f"Jupiter quote error: {quote['error']}")
        return await self._execute_swap(quote)
