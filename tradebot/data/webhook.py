"""TradingView (or any other) webhook ingestion.

This is the recommended live path. TradingView already computes your indicators
precisely; an alert carrying the bar's OHLCV is strictly better information
than a screenshot of the same bar, and it arrives on bar close for free.

Set an alert's message body to JSON, e.g.:

    {
      "secret": "your-shared-secret",
      "symbol": "{{ticker}}",
      "timeframe": "{{interval}}",
      "time": "{{timenow}}",
      "open": {{open}}, "high": {{high}}, "low": {{low}},
      "close": {{close}}, "volume": {{volume}},
      "closed": true
    }

and point the alert's webhook URL at POST /webhook/tradingview.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from ..models import Candle, utcnow
from .base import CandleAggregator, parse_candle, timeframe_seconds

log = logging.getLogger(__name__)


class WebhookSource:
    """A push source: the HTTP handler calls `submit`, the loop consumes it."""

    name = "webhook"

    def __init__(self, timeframe: str, maxsize: int = 512) -> None:
        self.timeframe = timeframe
        self.seconds = timeframe_seconds(timeframe)
        self.queue: asyncio.Queue[Candle] = asyncio.Queue(maxsize=maxsize)
        self.aggregator = CandleAggregator(timeframe)
        self.last_payload: Optional[dict[str, Any]] = None
        self.last_received_at = None
        self.received = 0
        self.rejected = 0

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Convert one alert into candles. Returns a small result for the HTTP reply."""
        self.last_payload = payload
        self.last_received_at = utcnow()

        candle = parse_candle(payload, self.seconds)
        if candle is None:
            self.rejected += 1
            return {"accepted": False, "reason": "no usable price in payload"}

        emitted = []
        if candle.closed:
            emitted.append(candle)
        else:
            # A forming-bar or tick-shaped alert: aggregate it ourselves.
            closed, forming = self.aggregator.add(candle.close, candle.volume, candle.ts)
            if closed:
                emitted.append(closed)
            emitted.append(forming)

        for c in emitted:
            try:
                self.queue.put_nowait(c)
            except asyncio.QueueFull:
                self.rejected += 1
                log.warning("webhook queue full - dropping a bar")
                return {"accepted": False, "reason": "queue full"}

        self.received += 1
        return {
            "accepted": True,
            "bars": len(emitted),
            "closed_bar": bool(candle.closed),
            "symbol": payload.get("symbol"),
        }

    def status(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "rejected": self.rejected,
            "queued": self.queue.qsize(),
            "last_received_at": self.last_received_at.isoformat() if self.last_received_at else None,
        }

    async def stream(self) -> AsyncIterator[Candle]:
        while True:
            yield await self.queue.get()
