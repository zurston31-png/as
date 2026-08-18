"""Market condition classification.

A setup that works in a trend is often exactly wrong in chop, so strategies
need to know which market they are in before deciding whether to act at all.

Two axes are reported independently, because they are independent facts: a
market can be a bull trend AND highly volatile. Collapsing them into one
label would lose that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.data.candles import CandleSeries
from app.signals.indicators import atr_percent, ema, price_structure, sma


class TrendRegime(str, Enum):
    BULL = "bull_trend"
    BEAR = "bear_trend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class VolatilityRegime(str, Enum):
    HIGH = "high_volatility"
    NORMAL = "normal_volatility"
    LOW = "low_volatility"
    UNKNOWN = "unknown"


# ATR as a fraction of price. Memecoins are volatile by nature, so these are
# set well above equity-market norms — 3%+ per candle is genuinely wild even
# here, and under 0.8% is unusually quiet.
HIGH_VOLATILITY_ATR = 0.030
LOW_VOLATILITY_ATR = 0.008

# How far the fast and slow averages must separate before the market counts
# as trending rather than drifting.
TREND_SEPARATION = 0.02


@dataclass
class MarketCondition:
    trend: TrendRegime
    volatility: VolatilityRegime
    atr_pct: float | None = None
    trend_strength: float | None = None   # signed EMA separation
    structure: str = "ranging"
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.trend.value} / {self.volatility.value}"

    @property
    def is_trending(self) -> bool:
        return self.trend in (TrendRegime.BULL, TrendRegime.BEAR)

    @property
    def is_tradeable(self) -> bool:
        """Whether the regime is known well enough to act on at all."""
        return self.trend is not TrendRegime.UNKNOWN and self.volatility is not VolatilityRegime.UNKNOWN

    def describe(self) -> str:
        parts = [self.label]
        if self.atr_pct is not None:
            parts.append(f"ATR {self.atr_pct * 100:.2f}%")
        if self.trend_strength is not None:
            parts.append(f"trend {self.trend_strength * 100:+.2f}%")
        parts.append(f"structure {self.structure}")
        return " | ".join(parts)


def classify(series: CandleSeries, *, min_candles: int = 50) -> MarketCondition:
    """Classify the market from a candle series.

    Returns UNKNOWN rather than guessing when there is not enough history —
    an unknown regime should block a regime-dependent strategy, not be
    silently treated as sideways.
    """
    if len(series) < min_candles:
        return MarketCondition(
            trend=TrendRegime.UNKNOWN,
            volatility=VolatilityRegime.UNKNOWN,
            notes=[f"only {len(series)} candles, need {min_candles} to classify"],
        )

    closes = series.closes
    highs = series.highs
    lows = series.lows
    notes: list[str] = []

    # --- volatility ---
    atr_series = atr_percent(highs, lows, closes, period=14)
    current_atr = atr_series[-1]
    if current_atr is None:
        volatility = VolatilityRegime.UNKNOWN
        notes.append("ATR unavailable")
    elif current_atr >= HIGH_VOLATILITY_ATR:
        volatility = VolatilityRegime.HIGH
    elif current_atr <= LOW_VOLATILITY_ATR:
        volatility = VolatilityRegime.LOW
    else:
        volatility = VolatilityRegime.NORMAL

    # --- trend ---
    # Long averages when there is history for them, shorter ones otherwise,
    # with a note so the caller knows the classification is less robust.
    if len(closes) >= 200:
        fast, slow = ema(closes, 50)[-1], ema(closes, 200)[-1]
    else:
        fast, slow = ema(closes, 9)[-1], sma(closes, min(50, len(closes) - 1))[-1]
        notes.append("fewer than 200 candles - trend read from shorter averages")

    trend_strength = None
    if fast is None or slow is None or not slow:
        trend = TrendRegime.UNKNOWN
        notes.append("moving averages unavailable")
    else:
        trend_strength = (fast - slow) / slow
        if trend_strength >= TREND_SEPARATION:
            trend = TrendRegime.BULL
        elif trend_strength <= -TREND_SEPARATION:
            trend = TrendRegime.BEAR
        else:
            trend = TrendRegime.SIDEWAYS

    return MarketCondition(
        trend=trend,
        volatility=volatility,
        atr_pct=current_atr,
        trend_strength=trend_strength,
        structure=price_structure(highs, lows),
        notes=notes,
    )


def suits_strategy(condition: MarketCondition, allowed: set[str] | None) -> tuple[bool, str]:
    """Whether a strategy should trade in this regime.

    `allowed` is a set of regime values a strategy declares it works in
    (either trend or volatility labels). None means the strategy is
    regime-agnostic and always eligible.
    """
    if allowed is None:
        return True, "strategy is regime-agnostic"

    if not condition.is_tradeable:
        return False, f"market regime could not be determined ({'; '.join(condition.notes)})"

    present = {condition.trend.value, condition.volatility.value}
    matched = present & allowed
    if matched:
        return True, f"regime {sorted(matched)} matches the strategy"
    return False, (
        f"market is {condition.label}, strategy only trades {sorted(allowed)}"
    )
