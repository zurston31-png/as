"""Paper trading engine — the default execution backend.

Simulates fills against the live market price using the realistic fill
model in app/execution/fill_model.py. No private key is ever touched here,
and no network write happens against any chain or exchange. This is what
runs whenever LIVE_TRADING=false, regardless of EXECUTION_BACKEND.

What a fill costs is driven by the ACTUAL pool it would trade against, not
a flat assumption. The engine pulls the token's market snapshot
(DexScreener liquidity + recent volatility, already fetched elsewhere in
the bot) and hands it to the fill model, which charges:

  price impact   derived from pool depth via constant-product AMM math, so
                 a $20 trade and a $20,000 trade against the same thin pool
                 cost very different amounts - the flat-percentage version
                 this replaces charged them identically, which is the main
                 way a memecoin paper simulator flatters itself
  spread         the bid/ask gap crossed to transact
  delay drift    the price keeps moving between signing and inclusion
  fees           per side
  failed fills   the swap REVERTS when impact + drift exceed SLIPPAGE_BPS,
                 exactly as an on-chain swap with slippage protection does

If the market snapshot can't be fetched, the model falls back to a
deliberately modest assumed pool depth rather than a generous one - an
unknown pool should not be assumed forgiving.
"""
import logging
import random
import uuid

from app.config import settings
from app.execution.base import ExecutionClient, SwapResult
from app.execution.fill_model import simulate_fill
from app.services import price_feed

logger = logging.getLogger(__name__)


class PaperExecutionClient(ExecutionClient):
    def __init__(self, rng: random.Random | None = None):
        """`rng` seeds the fill model's confirmation delay and price drift.

        Left as None in production, where the whole point is that fills are
        not deterministic. Passing a seeded Random makes a paper run
        reproducible, which is what lets a test assert on a specific fill
        instead of on a distribution - and lets an operator replay a session
        exactly.
        """
        self._rng = rng

    async def get_price(self, instrument: str) -> float | None:
        return await price_feed.get_price_usd(instrument)

    async def _market_context(self, instrument: str) -> tuple[float | None, float | None, float | None]:
        """(price, liquidity_usd, 1h volatility %) for the fill model.

        One snapshot call covers all three. On failure the caller still gets
        a price via the plain lookup, and the fill model applies its
        conservative fallbacks for the missing depth/volatility.
        """
        try:
            snapshot = await price_feed.get_market_snapshot(instrument)
        except Exception:
            logger.warning("market snapshot failed for %s - using fallback fill assumptions", instrument)
            snapshot = None

        if snapshot is not None:
            return snapshot.price_usd, snapshot.liquidity_usd, snapshot.price_change_1h_pct
        return await self.get_price(instrument), None, None

    def _result_from(self, outcome, qty: float, instrument: str, side: str) -> SwapResult:
        if not outcome.filled:
            logger.info("paper %s of %s did not fill: %s", side, instrument, outcome.failure_reason)
            return SwapResult(success=False, error=f"paper fill failed: {outcome.failure_reason}")
        return SwapResult(
            success=True, filled_qty=qty, avg_price=outcome.fill_price,
            tx_hash=f"PAPER-{uuid.uuid4().hex[:16]}",
            fee_usd=qty * outcome.fill_price * outcome.fee_pct,
            execution_cost_pct=outcome.total_cost_pct,
            fill_delay_seconds=outcome.delay_seconds,
        )

    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult:
        price, liquidity, volatility = await self._market_context(instrument)
        if not price or price <= 0:
            return SwapResult(success=False, error="no price available for token (paper mode)")

        outcome = simulate_fill(
            side="buy", reference_price=price, trade_usd=usd_amount,
            liquidity_usd=liquidity, volatility_1h_pct=volatility, slippage_bps=slippage_bps,
            rng=self._rng,
        )
        if not outcome.filled and not settings.PAPER_ALLOW_FAILED_FILLS:
            # Failed fills disabled: charge the costs but let it through, so
            # the operator can compare against the always-fills baseline.
            outcome.filled = True
            outcome.total_cost_pct = outcome.impact_pct + outcome.spread_pct + outcome.fee_pct
            outcome.fill_price = price * (1 + outcome.total_cost_pct)

        if not outcome.filled:
            return self._result_from(outcome, 0.0, instrument, "buy")

        qty = usd_amount / outcome.fill_price
        return self._result_from(outcome, qty, instrument, "buy")

    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult:
        price, liquidity, volatility = await self._market_context(instrument)
        if not price or price <= 0:
            return SwapResult(success=False, error="no price available for token (paper mode)")

        # Impact scales with the USD notional being pushed through the pool,
        # so size the sell in dollars at the current price.
        outcome = simulate_fill(
            side="sell", reference_price=price, trade_usd=qty * price,
            liquidity_usd=liquidity, volatility_1h_pct=volatility, slippage_bps=slippage_bps,
            rng=self._rng,
        )
        if not outcome.filled and not settings.PAPER_ALLOW_FAILED_FILLS:
            outcome.filled = True
            outcome.total_cost_pct = outcome.impact_pct + outcome.spread_pct + outcome.fee_pct
            outcome.fill_price = max(price * (1 - outcome.total_cost_pct), 0.0)

        return self._result_from(outcome, qty, instrument, "sell")
