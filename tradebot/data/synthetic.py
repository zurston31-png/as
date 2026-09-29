"""A synthetic feed so the whole stack runs with zero setup.

It is not a market simulator and is not meant to be traded off. It produces a
trending random walk with volume that clusters on impulse bars, which is enough
to exercise every rule, every risk limit, and the UI. `tradebot serve` uses it
by default precisely so a first run does something visible.
"""

from __future__ import annotations

import asyncio
import random
from datetime import timedelta
from typing import AsyncIterator

from ..models import Candle, utcnow
from .base import bucket_start, timeframe_seconds


class SyntheticSource:
    name = "synthetic"

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        start_price: float = 25_400.0,
        bars_per_second: float = 4.0,
        seed: int | None = None,
        warmup_bars: int = 250,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.seconds = timeframe_seconds(timeframe)
        self.price = start_price
        self.anchor = start_price
        self.interval = 1.0 / bars_per_second if bars_per_second > 0 else 0.25
        self.rng = random.Random(seed)
        self.warmup_bars = warmup_bars
        self._drift = 0.0

    def _next(self, ts) -> Candle:
        # Slowly-varying drift produces runs and reversals rather than noise; a
        # weak pull toward the anchor keeps a long run from wandering off by 3x,
        # which would make ATR-based stops meaningless.
        self._drift = self._drift * 0.92 + self.rng.gauss(0, 0.6)
        pull = (self.anchor - self.price) / self.anchor * 12.0
        vol_scale = self.price * 0.0005
        move = (self._drift * 0.4 + pull + self.rng.gauss(0, 1.0)) * vol_scale
        open_ = self.price
        close = max(1.0, open_ + move)
        wick = abs(self.rng.gauss(0, 1)) * vol_scale * 0.7
        high = max(open_, close) + wick
        low = min(open_, close) - abs(self.rng.gauss(0, 1)) * vol_scale * 0.7
        impulse = abs(close - open_) / vol_scale
        volume = round(max(50.0, self.rng.gauss(1000, 200) * (0.7 + impulse * 0.6)))
        self.price = close
        return Candle(ts, round(open_, 2), round(high, 2), round(low, 2), round(close, 2), volume)

    def history(self, bars: int) -> list[Candle]:
        """Backfill so indicators are warm before the first live bar."""
        end = bucket_start(utcnow(), self.seconds)
        out: list[Candle] = []
        for i in range(bars, 0, -1):
            out.append(self._next(end - timedelta(seconds=self.seconds * i)))
        return out

    async def stream(self) -> AsyncIterator[Candle]:
        for candle in self.history(self.warmup_bars):
            yield candle
        ts = bucket_start(utcnow(), self.seconds)
        while True:
            await asyncio.sleep(self.interval)
            ts = ts + timedelta(seconds=self.seconds)
            yield self._next(ts)
