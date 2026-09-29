"""Market data sources.

Everything upstream of the strategy engine normalises to one shape: a stream of
`Candle` objects with `closed` set correctly. Rules only ever fire on closed
bars; forming bars are forwarded so the dashboard and the paper broker can see
live price.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import AsyncIterator, Optional, Protocol

from ..models import Candle

_TF_RE = re.compile(r"^(\d+)\s*([smhdwSMHDW])$")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def timeframe_seconds(timeframe: str) -> int:
    """'5m' -> 300. Accepts s/m/h/d/w; raises on anything else."""
    tf = timeframe.strip()
    match = _TF_RE.match(tf)
    if not match:
        raise ValueError(f"unrecognised timeframe {timeframe!r} (expected e.g. '1m', '5m', '1h')")
    amount, unit = int(match.group(1)), match.group(2).lower()
    if amount <= 0:
        raise ValueError(f"timeframe {timeframe!r} must be positive")
    return amount * _UNITS[unit]


def bucket_start(ts: datetime, seconds: int) -> datetime:
    """Floor a timestamp to its bar-open time."""
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % seconds), tz=timezone.utc)


class DataSource(Protocol):
    name: str

    def stream(self) -> AsyncIterator[Candle]: ...


class CandleAggregator:
    """Builds fixed-interval candles out of a stream of trades/prices.

    Used by the tick-shaped feeds (a TradingView alert that only carries a
    price, a REST quote poller). Emits the closed bar as soon as a price
    arrives in a later bucket.
    """

    def __init__(self, timeframe: str) -> None:
        self.seconds = timeframe_seconds(timeframe)
        self.current: Optional[Candle] = None

    def add(self, price: float, volume: float = 0.0, ts: Optional[datetime] = None
            ) -> tuple[Optional[Candle], Candle]:
        """Returns (closed_candle_or_None, forming_candle)."""
        ts = (ts or datetime.now(timezone.utc)).astimezone(timezone.utc)
        start = bucket_start(ts, self.seconds)
        closed: Optional[Candle] = None

        if self.current is None:
            self.current = Candle(start, price, price, price, price, volume, closed=False)
            return None, self.current

        if start > self.current.ts:
            self.current.closed = True
            closed = self.current
            self.current = Candle(start, price, price, price, price, volume, closed=False)
            return closed, self.current

        c = self.current
        c.high = max(c.high, price)
        c.low = min(c.low, price)
        c.close = price
        c.volume += volume
        return None, c

    def force_close(self) -> Optional[Candle]:
        if self.current and not self.current.closed:
            self.current.closed = True
            return self.current
        return None


def parse_candle(payload: dict, default_tf_seconds: int = 300) -> Optional[Candle]:
    """Best-effort parse of an external OHLCV payload (webhooks, REST, CSV rows).

    Tolerates the several timestamp shapes these feeds use - ISO strings, epoch
    seconds, epoch milliseconds - and falls back to now() when there is none.
    """
    def num(*keys: str) -> Optional[float]:
        for k in keys:
            if k in payload and payload[k] not in (None, ""):
                try:
                    return float(payload[k])
                except (TypeError, ValueError):
                    return None
        return None

    close = num("close", "c", "price", "last")
    if close is None:
        return None
    open_ = num("open", "o") or close
    high = num("high", "h") or max(open_, close)
    low = num("low", "l") or min(open_, close)
    volume = num("volume", "v", "vol") or 0.0

    ts = _parse_time(payload.get("time") or payload.get("timestamp") or payload.get("t"))
    ts = bucket_start(ts, default_tf_seconds)
    closed = payload.get("closed", payload.get("bar_closed", True))
    if isinstance(closed, str):
        closed = closed.strip().lower() in ("1", "true", "yes")

    return Candle(ts, open_, high, low, close, volume, closed=bool(closed))


def _parse_time(value) -> datetime:
    if value in (None, ""):
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e11:            # milliseconds
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
