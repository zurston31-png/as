"""OHLCV candles and the series wrapper the rest of the system reads.

Deliberately plain Python — no numpy or pandas. The series here are hundreds
to low thousands of candles, where the speed difference is irrelevant, and
both libraries ship compiled wheels that are a recurring source of install
failures on new Python versions. Keeping the dependency surface small is
worth more than microseconds.

`CandleSeries` is immutable-by-convention and every accessor is
history-only: `series.closes` never includes a candle later than the one you
asked about. That is what keeps look-ahead bias out of the backtester.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return {
            "1m": 60, "5m": 300, "15m": 900,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }[self.value]


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar. `timestamp` is the bar's OPEN time, in UTC."""

    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("candle timestamp must be timezone-aware (UTC)")

    @property
    def typical_price(self) -> float:
        """(H + L + C) / 3 — the price VWAP is built from."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_green(self) -> bool:
        return self.close >= self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    def is_structurally_valid(self) -> bool:
        """Whether the bar is internally coherent.

        A high below the low, or a close outside the high/low range, means
        the feed is corrupt — worth catching before it silently skews an
        indicator.
        """
        return (
            self.high >= self.low
            and self.high >= self.open
            and self.high >= self.close
            and self.low <= self.open
            and self.low <= self.close
            and self.volume >= 0
            and self.open > 0
            and self.close > 0
        )


class CandleSeries:
    """An ordered, oldest-first run of candles for one symbol/timeframe."""

    def __init__(self, symbol: str, timeframe: Timeframe, candles: list[Candle]):
        self.symbol = symbol
        self.timeframe = timeframe
        self._candles = sorted(candles, key=lambda c: c.timestamp)

    # ---- basics ----

    def __len__(self) -> int:
        return len(self._candles)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return CandleSeries(self.symbol, self.timeframe, self._candles[index])
        return self._candles[index]

    def __iter__(self):
        return iter(self._candles)

    def __repr__(self) -> str:
        return f"<CandleSeries {self.symbol} {self.timeframe.value} n={len(self)}>"

    @property
    def candles(self) -> list[Candle]:
        return list(self._candles)

    # ---- column views, oldest first ----

    @property
    def opens(self) -> list[float]:
        return [c.open for c in self._candles]

    @property
    def highs(self) -> list[float]:
        return [c.high for c in self._candles]

    @property
    def lows(self) -> list[float]:
        return [c.low for c in self._candles]

    @property
    def closes(self) -> list[float]:
        return [c.close for c in self._candles]

    @property
    def volumes(self) -> list[float]:
        return [c.volume for c in self._candles]

    @property
    def typical_prices(self) -> list[float]:
        return [c.typical_price for c in self._candles]

    @property
    def last(self) -> Candle | None:
        return self._candles[-1] if self._candles else None

    @property
    def last_price(self) -> float | None:
        return self._candles[-1].close if self._candles else None

    # ---- history-only slicing, the look-ahead guard ----

    def up_to(self, timestamp: dt.datetime) -> CandleSeries:
        """Every candle that had CLOSED at `timestamp`.

        A bar stamped with its open time has not finished until one interval
        later, so a bar opening exactly at `timestamp` is excluded. This is
        the single most important guard against look-ahead bias: a backtest
        stepping through history must only ever see this.
        """
        cutoff = timestamp - dt.timedelta(seconds=self.timeframe.seconds)
        return CandleSeries(
            self.symbol,
            self.timeframe,
            [c for c in self._candles if c.timestamp <= cutoff],
        )

    def head(self, n: int) -> CandleSeries:
        return CandleSeries(self.symbol, self.timeframe, self._candles[:n])

    def tail(self, n: int) -> CandleSeries:
        return CandleSeries(self.symbol, self.timeframe, self._candles[-n:] if n else [])

    def split(self, *fractions: float) -> list[CandleSeries]:
        """Split chronologically into consecutive chunks.

        Used for walk-forward testing: `split(0.5, 0.25, 0.25)` yields
        train / validation / out-of-sample runs that never overlap and are
        always in time order.
        """
        if not fractions:
            return [self]
        total = sum(fractions)
        if total <= 0:
            raise ValueError("split fractions must sum to a positive number")

        out: list[CandleSeries] = []
        start = 0
        for i, frac in enumerate(fractions):
            if i == len(fractions) - 1:
                end = len(self._candles)
            else:
                end = start + int(len(self._candles) * (frac / total))
            out.append(CandleSeries(self.symbol, self.timeframe, self._candles[start:end]))
            start = end
        return out
