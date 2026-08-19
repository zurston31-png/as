"""Paper trading engine — the default execution backend.

Simulates fills using the live market price (via DexScreener) plus a
synthetic slippage buffer AND a trading fee. No private key is ever
touched here, and no network write happens against any chain or exchange.
This is what runs whenever LIVE_TRADING=false, regardless of
EXECUTION_BACKEND.

Both costs are deliberately modeled, and both are applied per side rather
than netted round-trip:

  slippage/impact  the fill is worse than the observed mid price - you pay
                   up on a buy, receive less on a sell
  fee              a DEX/exchange cut, charged on the notional of each leg

The fee used to be missing entirely, which quietly broke the premise the
whole project rests on: the backtester (app/backtesting/fills.py) charges
`fee_pct` per side, so paper trading was systematically CHEAPER than the
backtest meant to validate it. A strategy could look profitable on paper
and unprofitable in the backtest purely from that mismatch - the opposite
of the direction an error should point. PAPER_FEE_PCT defaults to the same
0.25% as BacktestConfig.fee_pct so the two agree unless deliberately
diverged.

The fee is expressed by widening the effective fill price rather than
tracked as a separate ledger entry, because that keeps
`qty * avg_price == the USD the risk manager sized` - an invariant the cash
ledger and position valuation both depend on (see
tests/test_execution_safety.py).
"""
import uuid

from app.config import settings
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
        # Buying: both costs push the effective price UP, so the same USD
        # buys fewer tokens.
        fill_price = price * (1 + PAPER_SLIPPAGE_BUFFER + settings.PAPER_FEE_PCT)
        qty = usd_amount / fill_price
        return SwapResult(success=True, filled_qty=qty, avg_price=fill_price, tx_hash=f"PAPER-{uuid.uuid4().hex[:16]}")

    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult:
        price = await self.get_price(instrument)
        if not price or price <= 0:
            return SwapResult(success=False, error="no price available for token (paper mode)")
        # Selling: both costs push the effective price DOWN, so the same
        # quantity returns fewer dollars.
        fill_price = price * (1 - PAPER_SLIPPAGE_BUFFER - settings.PAPER_FEE_PCT)
        return SwapResult(
            success=True, filled_qty=qty, avg_price=max(fill_price, 0.0),
            tx_hash=f"PAPER-{uuid.uuid4().hex[:16]}",
        )
