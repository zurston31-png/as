"""CEX execution backend (Binance/Coinbase/Kraken/... via ccxt) for
memecoins that trade on a centralized order book instead of an on-chain
DEX pool.

Requires `ccxt` (see requirements-live.txt) and API credentials with
TRADE permission only — never grant withdrawal permission to the API key
used here.

When EXECUTION_BACKEND=cex, the `instrument` argument is the base asset
symbol (e.g. "DOGE"), not an on-chain token address — trading_service.py
picks the right identifier per backend.
"""
import logging

from app.config import settings
from app.execution.base import ExecutionClient, SwapResult

logger = logging.getLogger(__name__)


class CexExecutionClient(ExecutionClient):
    def __init__(self):
        try:
            import ccxt.async_support as ccxt
        except ImportError as exc:
            raise RuntimeError("ccxt not installed - run `pip install -r requirements-live.txt`") from exc

        if not hasattr(ccxt, settings.CEX_EXCHANGE):
            raise ValueError(f"ccxt has no exchange named '{settings.CEX_EXCHANGE}'")

        exchange_cls = getattr(ccxt, settings.CEX_EXCHANGE)
        self.exchange = exchange_cls(
            {
                "apiKey": settings.CEX_API_KEY,
                "secret": settings.CEX_API_SECRET,
                "enableRateLimit": True,
            }
        )

    @staticmethod
    def _pair(symbol: str) -> str:
        return f"{symbol}/USDT"

    async def get_price(self, instrument: str) -> float | None:
        ticker = await self.exchange.fetch_ticker(self._pair(instrument))
        return ticker.get("last")

    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult:
        if not settings.LIVE_TRADING:
            return SwapResult(success=False, error="LIVE_TRADING is false; refusing to submit a live order")
        pair = self._pair(instrument)
        order = await self.exchange.create_market_buy_order(pair, None, {"quoteOrderQty": usd_amount})
        filled = float(order.get("filled") or 0)
        cost = float(order.get("cost") or usd_amount)
        avg = (cost / filled) if filled else 0.0
        return SwapResult(success=filled > 0, filled_qty=filled, avg_price=avg, tx_hash=str(order.get("id")))

    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult:
        if not settings.LIVE_TRADING:
            return SwapResult(success=False, error="LIVE_TRADING is false; refusing to submit a live order")
        pair = self._pair(instrument)
        order = await self.exchange.create_market_sell_order(pair, qty)
        filled = float(order.get("filled") or qty)
        cost = float(order.get("cost") or 0)
        avg = (cost / filled) if filled else 0.0
        return SwapResult(success=filled > 0, filled_qty=filled, avg_price=avg, tx_hash=str(order.get("id")))
