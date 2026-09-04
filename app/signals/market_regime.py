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


class LiquidityRegime(str, Enum):
    """A third axis, and the one candles cannot see.

    Trend and volatility are properties of the price path. Depth is a
    property of the book, and in memecoins it is often the fact that
    decides whether a correct call is tradeable at all - a perfect signal
    in a $4k pool is not a trade. It comes from the market snapshot, never
    from candles, so it is UNKNOWN whenever no snapshot was available
    rather than being inferred from volume.
    """
    THIN = "low_liquidity"
    MODERATE = "moderate_liquidity"
    DEEP = "deep_liquidity"
    UNKNOWN = "unknown"


# Pool depth in USD. A memecoin under $25k is thin enough that a normal
# position moves the price against itself; over $250k it behaves like a
# real market for the sizes this bot trades.
THIN_LIQUIDITY_USD = 25_000.0
DEEP_LIQUIDITY_USD = 250_000.0


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
    liquidity: LiquidityRegime = LiquidityRegime.UNKNOWN
    atr_pct: float | None = None
    trend_strength: float | None = None   # signed EMA separation
    structure: str = "ranging"
    liquidity_usd: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.trend.value} / {self.volatility.value}"

    @property
    def full_label(self) -> str:
        """All three axes. This is what gets persisted and grouped on."""
        return f"{self.trend.value}/{self.volatility.value}/{self.liquidity.value}"

    @property
    def buckets(self) -> list[str]:
        """The axes as separate labels.

        Performance is grouped one axis at a time. The full cross product
        is 36 cells and a memecoin bot will never have enough trades to
        fill them - slicing that finely produces cells of three trades and
        a confident-looking table built on nothing.
        """
        return [self.trend.value, self.volatility.value, self.liquidity.value]

    def features(self) -> dict:
        """The inputs the classification was made from.

        Stored alongside the label so a past regime call can be re-judged.
        A bare label cannot be audited: "sideways" six weeks ago is not
        reviewable unless the ATR and separation that produced it are
        recorded next to it.
        """
        return {
            "trend": self.trend.value,
            "volatility": self.volatility.value,
            "liquidity": self.liquidity.value,
            "atr_pct": round(self.atr_pct, 6) if self.atr_pct is not None else None,
            "trend_strength": (
                round(self.trend_strength, 6) if self.trend_strength is not None else None
            ),
            "structure": self.structure,
            "liquidity_usd": self.liquidity_usd,
            "thin_below_usd": THIN_LIQUIDITY_USD,
            "deep_above_usd": DEEP_LIQUIDITY_USD,
            "high_volatility_atr": HIGH_VOLATILITY_ATR,
            "low_volatility_atr": LOW_VOLATILITY_ATR,
            "trend_separation": TREND_SEPARATION,
            "notes": list(self.notes),
        }

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


def classify_liquidity(liquidity_usd: float | None) -> LiquidityRegime:
    """Depth band from the market snapshot.

    UNKNOWN when no reading was available. Not inferred from volume: a
    token can trade heavily in a pool that is about to be drained, and
    treating turnover as depth is how a thin pool passes for a deep one.
    """
    if liquidity_usd is None or liquidity_usd <= 0:
        return LiquidityRegime.UNKNOWN
    if liquidity_usd < THIN_LIQUIDITY_USD:
        return LiquidityRegime.THIN
    if liquidity_usd >= DEEP_LIQUIDITY_USD:
        return LiquidityRegime.DEEP
    return LiquidityRegime.MODERATE


def classify_full(
    series: CandleSeries | None,
    *,
    liquidity_usd: float | None = None,
    min_candles: int = 50,
) -> MarketCondition:
    """All three axes at once.

    `series` carries only candles up to the decision point - the caller is
    responsible for never handing this future data, and the backtester
    slices with CandleSeries.head() for exactly that reason. Nothing here
    reaches forward on its own: it reads the series it is given and no
    more.
    """
    if series is None or not len(series):
        condition = MarketCondition(
            trend=TrendRegime.UNKNOWN,
            volatility=VolatilityRegime.UNKNOWN,
            notes=["no candle history - trend and volatility unassessable"],
        )
    else:
        condition = classify(series, min_candles=min_candles)

    condition.liquidity = classify_liquidity(liquidity_usd)
    condition.liquidity_usd = liquidity_usd
    if condition.liquidity is LiquidityRegime.UNKNOWN:
        condition.notes.append("no liquidity reading - depth regime unassessable")
    return condition
