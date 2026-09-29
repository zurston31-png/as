"""Generic REST polling source.

For any endpoint that returns a current price or a list of OHLCV rows. Point
`feed.poll_url` at it; the response is parsed leniently, so most quote APIs
work without writing an adapter.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional
from urllib.request import Request, urlopen

from ..models import Candle
from .base import CandleAggregator, parse_candle, timeframe_seconds

log = logging.getLogger(__name__)


class PollSource:
    name = "poll"

    def __init__(self, url: str, timeframe: str, interval_seconds: float = 15.0,
                 headers: Optional[dict[str, str]] = None) -> None:
        if not url:
            raise ValueError("feed.poll_url must be set to use the 'poll' source")
        self.url = url
        self.timeframe = timeframe
        self.seconds = timeframe_seconds(timeframe)
        self.interval = max(1.0, interval_seconds)
        self.headers = headers or {}
        self.aggregator = CandleAggregator(timeframe)
        self.errors = 0

    def _fetch(self) -> Any:
        req = Request(self.url, headers={"Accept": "application/json", **self.headers})
        with urlopen(req, timeout=10) as resp:  # noqa: S310 - user-supplied URL by design
            return json.loads(resp.read().decode())

    def _to_candles(self, data: Any) -> list[Candle]:
        if isinstance(data, dict):
            for key in ("candles", "bars", "data", "result", "results"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, list):
            out = []
            for row in data:
                if isinstance(row, dict):
                    c = parse_candle(row, self.seconds)
                    if c:
                        out.append(c)
                elif isinstance(row, (list, tuple)) and len(row) >= 5:
                    # The [time, o, h, l, c, v] convention most exchanges use.
                    c = parse_candle(
                        {"time": row[0], "open": row[1], "high": row[2],
                         "low": row[3], "close": row[4],
                         "volume": row[5] if len(row) > 5 else 0},
                        self.seconds,
                    )
                    if c:
                        out.append(c)
            return sorted(out, key=lambda c: c.ts)
        if isinstance(data, dict):
            c = parse_candle(data, self.seconds)
            return [c] if c else []
        return []

    async def stream(self) -> AsyncIterator[Candle]:
        loop = asyncio.get_running_loop()
        while True:
            try:
                data = await loop.run_in_executor(None, self._fetch)
                candles = self._to_candles(data)
            except Exception as exc:  # noqa: BLE001 - a flaky endpoint must not kill the loop
                self.errors += 1
                log.warning("poll failed (%s): %s", self.errors, exc)
                await asyncio.sleep(self.interval)
                continue

            if len(candles) > 1:
                for c in candles:
                    yield c
            elif candles:
                closed, forming = self.aggregator.add(
                    candles[0].close, candles[0].volume, candles[0].ts
                )
                if closed:
                    yield closed
                yield forming
            await asyncio.sleep(self.interval)
