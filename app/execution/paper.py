"""Paper trading engine — the default execution backend.

Simulates fills using the live market price (via DexScreener) plus a small
synthetic slippage buffer. No private key is ever touched here, and no
network write happens against any chain or exchange. This is what runs
whenever LIVE_TRADING=false, regardless of EXECUTION_BACKEND.
"""
import uuid

from app.execution.base import ExecutionClient, SwapResult
from app.services import price_feed

# Assumed simulated slippage/impact on top of the observed mid price. This
# is deliberately a bit pessimistic so paper results don't look better than
# a real fill would.
PAPER_SLIPPAGE_BUFFER = 0.005


class PaperExecutionClient(ExecutionClient):
    async def get_price(self, instrument: str) -> float | None:
        return await price_feed.get_price_usd(instrument)

    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult:
        price = await self.get_price(instrument)
        if not price or price <= 0:
            return SwapResult(success=False, error="no price available for token (paper mode)")
        fill_price = price * (1 + PAPER_SLIPPAGE_BUFFER)
        qty = usd_amount / fill_price
        return SwapResult(success=True, filled_qty=qty, avg_price=fill_price, tx_hash=f"PAPER-{uuid.uuid4().hex[:16]}")

    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult:
        price = await self.get_price(instrument)
        if not price or price <= 0:
            return SwapResult(success=False, error="no price available for token (paper mode)")
        fill_price = price * (1 - PAPER_SLIPPAGE_BUFFER)
        return SwapResult(success=True, filled_qty=qty, avg_price=fill_price, tx_hash=f"PAPER-{uuid.uuid4().hex[:16]}")
