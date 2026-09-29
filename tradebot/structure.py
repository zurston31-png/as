"""Market structure and liquidity primitives.

These are the "price action" inputs that a rules engine needs and that a naive
indicator stack misses: where the swings are, whether structure is trending,
and whether price just ran a pool of stops and snapped back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from .models import Candle

Trend = Literal["bullish", "bearish", "range"]


@dataclass(slots=True)
class Swing:
    index: int
    price: float
    kind: Literal["high", "low"]


def find_swings(candles: Sequence[Candle], width: int = 2) -> list[Swing]:
    """Fractal swing points: a high with `width` lower highs on each side.

    `width` bars at each end can never qualify, which is the usual and correct
    behaviour - a swing is only confirmed once price has moved away from it.
    """
    swings: list[Swing] = []
    n = len(candles)
    for i in range(width, n - width):
        window = candles[i - width : i + width + 1]
        c = candles[i]
        if all(c.high >= o.high for o in window) and any(c.high > o.high for o in window):
            swings.append(Swing(i, c.high, "high"))
        if all(c.low <= o.low for o in window) and any(c.low < o.low for o in window):
            swings.append(Swing(i, c.low, "low"))
    return swings


def recent_swings(
    candles: Sequence[Candle], kind: Literal["high", "low"], count: int = 2, width: int = 2
) -> list[Swing]:
    return [s for s in find_swings(candles, width) if s.kind == kind][-count:]


def classify_trend(candles: Sequence[Candle], width: int = 2, lookback: int = 60) -> Trend:
    """Higher-highs/higher-lows -> bullish, the mirror -> bearish, else range."""
    window = list(candles)[-lookback:]
    swings = find_swings(window, width)
    highs = [s.price for s in swings if s.kind == "high"][-2:]
    lows = [s.price for s in swings if s.kind == "low"][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return "range"
    hh, hl = highs[-1] > highs[-2], lows[-1] > lows[-2]
    lh, ll = highs[-1] < highs[-2], lows[-1] < lows[-2]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "range"


@dataclass(slots=True)
class StructureBreak:
    kind: Literal["BOS", "CHoCH", "none"]
    direction: Literal["up", "down", "none"]
    level: Optional[float] = None
    bars_ago: int = 0


def detect_break(
    candles: Sequence[Candle], width: int = 2, lookback: int = 5
) -> StructureBreak:
    """Break of structure: a close beyond the most recent confirmed swing.

    Labelled CHoCH when the break runs against the prevailing trend (the first
    crack in a trend) and BOS when it runs with it.
    """
    window = list(candles)
    if len(window) < width * 2 + 3:
        return StructureBreak("none", "none")
    trend = classify_trend(window, width)
    swings = find_swings(window, width)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    n = len(window)

    for back in range(min(lookback, n)):
        i = n - 1 - back
        c = window[i]
        prior_high = next((s for s in reversed(highs) if s.index < i), None)
        prior_low = next((s for s in reversed(lows) if s.index < i), None)
        if prior_high and c.close > prior_high.price:
            kind = "CHoCH" if trend == "bearish" else "BOS"
            return StructureBreak(kind, "up", prior_high.price, back)
        if prior_low and c.close < prior_low.price:
            kind = "CHoCH" if trend == "bullish" else "BOS"
            return StructureBreak(kind, "down", prior_low.price, back)
    return StructureBreak("none", "none")


@dataclass(slots=True)
class LiquiditySweep:
    happened: bool
    direction: Literal["up", "down", "none"] = "none"
    level: Optional[float] = None
    bars_ago: int = 0

    @property
    def bullish(self) -> bool:
        """A sweep *below* a low is a bullish (buy-side entry) signal."""
        return self.happened and self.direction == "down"

    @property
    def bearish(self) -> bool:
        return self.happened and self.direction == "up"


def detect_liquidity_sweep(
    candles: Sequence[Candle],
    width: int = 2,
    lookback: int = 5,
    min_wick_ratio: float = 0.5,
) -> LiquiditySweep:
    """Wick through a prior swing, close back inside it.

    ``min_wick_ratio`` is the share of the candle's range that must be wick on
    the swept side - that rejection is the whole point of the pattern.
    """
    window = list(candles)
    if len(window) < width * 2 + 3:
        return LiquiditySweep(False)
    swings = find_swings(window, width)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    n = len(window)

    for back in range(min(lookback, n)):
        i = n - 1 - back
        c = window[i]
        if c.range <= 0:
            continue
        prior_low = next((s for s in reversed(lows) if s.index < i), None)
        if (
            prior_low
            and c.low < prior_low.price
            and c.close > prior_low.price
            and c.lower_wick / c.range >= min_wick_ratio
        ):
            return LiquiditySweep(True, "down", prior_low.price, back)
        prior_high = next((s for s in reversed(highs) if s.index < i), None)
        if (
            prior_high
            and c.high > prior_high.price
            and c.close < prior_high.price
            and c.upper_wick / c.range >= min_wick_ratio
        ):
            return LiquiditySweep(True, "up", prior_high.price, back)
    return LiquiditySweep(False)
