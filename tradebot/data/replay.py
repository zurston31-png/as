"""CSV replay - the historical-backtest and dry-run feed.

Accepts the column names the common exports use (TradingView, most broker
exports, ccxt dumps) without asking you to rename anything.
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional

from ..models import Candle
from .base import _parse_time

_ALIASES = {
    "time": ("time", "timestamp", "date", "datetime", "open_time", "t"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "price", "last"),
    "volume": ("volume", "vol", "v", "volume btc", "basevolume"),
}


def _pick(row: dict[str, str], key: str) -> Optional[str]:
    lowered = {(k or "").strip().lower(): v for k, v in row.items()}
    for alias in _ALIASES[key]:
        if alias in lowered and lowered[alias] not in (None, ""):
            return lowered[alias]
    return None


def read_csv(path: str | Path) -> list[Candle]:
    """Load a whole OHLCV file into memory, oldest first."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no CSV at {path}")
    candles: list[Candle] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            close = _pick(row, "close")
            if close is None:
                continue
            try:
                c = float(close)
                o = float(_pick(row, "open") or c)
                h = float(_pick(row, "high") or max(o, c))
                lo = float(_pick(row, "low") or min(o, c))
                v = float(_pick(row, "volume") or 0.0)
            except ValueError:
                continue  # header repeated mid-file, or a blank row
            candles.append(Candle(_parse_time(_pick(row, "time")), o, h, lo, c, v, closed=True))
    candles.sort(key=lambda c: c.ts)
    return candles


class ReplaySource:
    name = "replay"

    def __init__(self, path: str | Path, bars_per_second: float = 60.0) -> None:
        self.path = Path(path)
        self.candles = read_csv(path)
        self.interval = 1.0 / bars_per_second if bars_per_second > 0 else 0.0

    def __len__(self) -> int:
        return len(self.candles)

    def iter_candles(self) -> Iterator[Candle]:
        return iter(self.candles)

    async def stream(self) -> AsyncIterator[Candle]:
        for candle in self.candles:
            if self.interval:
                await asyncio.sleep(self.interval)
            yield candle
