"""Signal scoring: 0-100 from many factors, never one indicator.

Each factor scores the setup from 0.0 (maximally against) through 0.5
(neutral / no opinion) to 1.0 (maximally for), and carries a plain-English
reason. The weighted average of those becomes the 0-100 score.

Two properties matter more than the exact weights:

  Every score is explainable. `SignalScore.factors` holds each factor's
  raw score, weight, points contributed, and why — so a losing trade can be
  read back rather than guessed at.

  A factor with no data scores 0.5 and is flagged, never 0 or 1. Missing
  information must not masquerade as either a reason to buy or a reason to
  sell; it just fails to add conviction. When too much is missing the whole
  score is marked unreliable regardless of the number it produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.data.candles import CandleSeries, Timeframe
from app.data.providers import resample
from app.signals import indicators as ind

NEUTRAL = 0.5

# How much of the total weight may come from factors with no data before the
# score stops being trustworthy.
MAX_UNAVAILABLE_WEIGHT = 0.35


@dataclass
class Factor:
    name: str
    score: float          # 0.0 - 1.0
    weight: float
    reason: str
    available: bool = True

    @property
    def points(self) -> float:
        """Points this factor contributes to the final 0-100 score."""
        return self.score * self.weight * 100

    @property
    def max_points(self) -> float:
        return self.weight * 100


@dataclass
class SignalScore:
    score: float                       # 0-100
    direction: str                     # "long" | "short" | "neutral"
    factors: list[Factor] = field(default_factory=list)
    reliable: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def supporting(self) -> list[Factor]:
        return sorted(
            [f for f in self.factors if f.available and f.score > 0.55],
            key=lambda f: f.points, reverse=True,
        )

    @property
    def opposing(self) -> list[Factor]:
        return sorted(
            [f for f in self.factors if f.available and f.score < 0.45],
            key=lambda f: f.points,
        )

    @property
    def unavailable(self) -> list[Factor]:
        return [f for f in self.factors if not f.available]

    def breakdown(self) -> str:
        """Human-readable account of how the score was reached."""
        lines = [f"Signal score {self.score:.1f}/100 ({self.direction})"]
        if not self.reliable:
            lines.append("  UNRELIABLE: " + "; ".join(self.warnings))
        for f in sorted(self.factors, key=lambda f: f.points, reverse=True):
            marker = " " if f.available else "?"
            lines.append(
                f"  {marker} {f.name:<24} {f.points:5.1f}/{f.max_points:4.1f}  {f.reason}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """For persisting into the trade journal."""
        return {
            "score": round(self.score, 2),
            "direction": self.direction,
            "reliable": self.reliable,
            "warnings": list(self.warnings),
            "factors": [
                {
                    "name": f.name,
                    "score": round(f.score, 3),
                    "weight": f.weight,
                    "points": round(f.points, 2),
                    "reason": f.reason,
                    "available": f.available,
                }
                for f in self.factors
            ],
        }


# Weights sum to 1.0. Trend, volume and structure carry more than oscillators
# because a memecoin breakout is driven by participation, not by RSI.
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_direction": 0.12,
    "multi_timeframe": 0.12,
    "ema_stack": 0.10,
    "macd": 0.09,
    "relative_volume": 0.09,
    "breakout": 0.09,
    "momentum": 0.08,
    "price_structure": 0.07,
    "rsi": 0.06,
    "vwap": 0.06,
    "support_resistance": 0.05,
    "bollinger": 0.04,
    "volume_spike": 0.02,
    "atr_sanity": 0.01,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unavailable(name: str, weight: float, reason: str) -> Factor:
    return Factor(name=name, score=NEUTRAL, weight=weight, reason=reason, available=False)


# ---------------------------------------------------------------------------
# individual factors
# ---------------------------------------------------------------------------

def score_rsi(closes: list[float], weight: float) -> Factor:
    values = ind.rsi(closes, 14)
    value = values[-1] if values else None
    if value is None:
        return _unavailable("rsi", weight, "not enough candles for RSI(14)")

    # Best just above oversold and recovering; worst when already overbought.
    if value >= 75:
        score, note = 0.10, "overbought - poor entry, likely chasing"
    elif value >= 65:
        score, note = 0.35, "extended"
    elif value >= 55:
        score, note = 0.75, "healthy bullish momentum"
    elif value >= 45:
        score, note = 0.60, "neutral, room to move"
    elif value >= 30:
        score, note = 0.70, "recovering from oversold"
    else:
        score, note = 0.40, "deeply oversold - may keep falling"
    return Factor("rsi", score, weight, f"RSI {value:.1f} - {note}")


def score_macd(closes: list[float], weight: float) -> Factor:
    macd_line, signal_line, histogram = ind.macd(closes)
    if not macd_line or macd_line[-1] is None or signal_line[-1] is None:
        return _unavailable("macd", weight, "not enough candles for MACD")

    line, sig, hist = macd_line[-1], signal_line[-1], histogram[-1]
    crossed = ind.crossed_above(macd_line, signal_line)
    rising = (
        len(histogram) > 1 and histogram[-2] is not None and hist > histogram[-2]
    )

    if crossed:
        score, note = 0.95, "just crossed above signal"
    elif line > sig and rising:
        score, note = 0.85, "above signal and expanding"
    elif line > sig:
        score, note = 0.65, "above signal but fading"
    elif rising:
        score, note = 0.45, "below signal but contracting"
    else:
        score, note = 0.15, "below signal and widening"
    return Factor("macd", score, weight, f"MACD {line:+.6g} vs signal {sig:+.6g} - {note}")


def score_ema_stack(closes: list[float], weight: float) -> Factor:
    """EMA 9 / 21 / 50 / 200 alignment."""
    if not closes:
        return _unavailable("ema_stack", weight, "no candles")
    e9, e21 = ind.ema(closes, 9)[-1], ind.ema(closes, 21)[-1]
    if e9 is None or e21 is None:
        return _unavailable("ema_stack", weight, "not enough candles for EMA(21)")

    e50 = ind.ema(closes, 50)[-1] if len(closes) >= 50 else None
    e200 = ind.ema(closes, 200)[-1] if len(closes) >= 200 else None
    price = closes[-1]

    if e50 is not None and e200 is not None:
        if e9 > e21 > e50 > e200 and price > e9:
            return Factor("ema_stack", 1.00, weight, "fully stacked bullish (9>21>50>200)")
        if e9 < e21 < e50 < e200:
            return Factor("ema_stack", 0.00, weight, "fully stacked bearish")
        if e50 > e200:
            score = 0.70 if e9 > e21 else 0.55
            return Factor("ema_stack", score, weight, "golden-cross regime, partial alignment")
        score = 0.30 if e9 < e21 else 0.45
        return Factor("ema_stack", score, weight, "death-cross regime, partial alignment")

    # Short history: judge on 9/21 only and say so.
    if e9 > e21 and price > e9:
        return Factor("ema_stack", 0.75, weight, "9>21 and price above (no 50/200 yet)")
    if e9 > e21:
        return Factor("ema_stack", 0.60, weight, "9>21 but price below (no 50/200 yet)")
    return Factor("ema_stack", 0.25, weight, "9<21 (no 50/200 yet)")


def score_vwap(series: CandleSeries, weight: float) -> Factor:
    if not series.closes:
        return _unavailable("vwap", weight, "no candles")
    values = ind.vwap(series.highs, series.lows, series.closes, series.volumes, period=20)
    value = values[-1] if values else None
    if value is None or not value:
        return _unavailable("vwap", weight, "VWAP unavailable (no volume?)")

    price = series.closes[-1]
    distance = (price - value) / value
    if distance > 0.05:
        score, note = 0.45, "far above VWAP - extended"
    elif distance > 0:
        score, note = 0.80, "above VWAP - buyers in control"
    elif distance > -0.03:
        score, note = 0.55, "just below VWAP"
    else:
        score, note = 0.25, "well below VWAP - sellers in control"
    return Factor("vwap", score, weight, f"price {distance * 100:+.2f}% vs VWAP - {note}")


def score_bollinger(closes: list[float], weight: float) -> Factor:
    upper, middle, lower = ind.bollinger_bands(closes, 20, 2.0)
    if not upper or upper[-1] is None or middle[-1] is None or lower[-1] is None:
        return _unavailable("bollinger", weight, "not enough candles for Bollinger(20)")

    price = closes[-1]
    band = upper[-1] - lower[-1]
    if band <= 0:
        return _unavailable("bollinger", weight, "bands collapsed - no dispersion")

    position = (price - lower[-1]) / band       # 0 at lower band, 1 at upper
    widths = ind.bollinger_width(closes, 20, 2.0)
    recent = [w for w in widths[-40:] if w is not None]
    squeezing = bool(recent) and widths[-1] is not None and widths[-1] <= min(recent) * 1.15

    if position > 1.0:
        score, note = 0.30, "closed above the upper band - overextended"
    elif position > 0.75:
        score, note = 0.70, "upper third - strength"
    elif position > 0.4:
        score, note = 0.60, "mid band"
    else:
        score, note = 0.40, "lower third"

    if squeezing:
        score = min(1.0, score + 0.15)
        note += ", bands squeezing (expansion often follows)"
    return Factor("bollinger", score, weight, f"{position * 100:.0f}% of band - {note}")


def score_atr_sanity(series: CandleSeries, weight: float) -> Factor:
    """Volatility sanity: enough movement to be worth trading, not so much
    that a stop is meaningless."""
    values = ind.atr_percent(series.highs, series.lows, series.closes, 14)
    value = values[-1] if values else None
    if value is None:
        return _unavailable("atr_sanity", weight, "ATR unavailable")

    if value < 0.002:
        score, note = 0.20, "almost no movement - nothing to capture"
    elif value < 0.05:
        score, note = 0.85, "workable volatility"
    elif value < 0.12:
        score, note = 0.50, "high volatility - widen stops"
    else:
        score, note = 0.15, "extreme volatility - stops likely to be run"
    return Factor("atr_sanity", score, weight, f"ATR {value * 100:.2f}% - {note}")


def score_relative_volume(volumes: list[float], weight: float) -> Factor:
    values = ind.relative_volume(volumes, 20)
    value = values[-1] if values else None
    if value is None:
        return _unavailable("relative_volume", weight, "not enough candles for volume average")

    if value >= 3.0:
        score, note = 1.00, "very heavy participation"
    elif value >= 2.0:
        score, note = 0.90, "heavy participation"
    elif value >= 1.3:
        score, note = 0.70, "above average"
    elif value >= 0.8:
        score, note = 0.45, "average"
    else:
        score, note = 0.15, "thin - moves on low volume rarely hold"
    return Factor("relative_volume", score, weight, f"{value:.2f}x average volume - {note}")


def score_volume_spike(volumes: list[float], weight: float) -> Factor:
    flags = ind.volume_spike(volumes, 20, multiple=2.0)
    if not flags or all(f is False for f in flags[:20]) and len(flags) < 21:
        return _unavailable("volume_spike", weight, "not enough candles to detect spikes")
    recent = flags[-3:]
    if recent and recent[-1]:
        return Factor("volume_spike", 1.0, weight, "volume spike on the current candle")
    if any(recent):
        return Factor("volume_spike", 0.7, weight, "volume spike within the last 3 candles")
    return Factor("volume_spike", 0.4, weight, "no recent volume spike")


def score_momentum(closes: list[float], weight: float) -> Factor:
    values = ind.momentum(closes, 10)
    value = values[-1] if values else None
    if value is None:
        return _unavailable("momentum", weight, "not enough candles for 10-period momentum")

    if value > 0.20:
        score, note = 0.55, "parabolic - late to the move"
    elif value > 0.05:
        score, note = 0.90, "strong upward momentum"
    elif value > 0.005:
        score, note = 0.70, "mild upward momentum"
    elif value > -0.02:
        score, note = 0.45, "flat"
    else:
        score, note = 0.15, "falling"
    return Factor("momentum", score, weight, f"{value * 100:+.2f}% over 10 candles - {note}")


def score_price_structure(series: CandleSeries, weight: float) -> Factor:
    if not series.closes:
        return _unavailable("price_structure", weight, "no candles")
    structure = ind.price_structure(series.highs, series.lows)
    mapping = {
        "uptrend": (0.90, "higher highs and higher lows"),
        "downtrend": (0.10, "lower highs and lower lows"),
        "ranging": (0.45, "no clear structure"),
    }
    score, note = mapping[structure]
    return Factor("price_structure", score, weight, f"{structure} - {note}")


def score_support_resistance(series: CandleSeries, weight: float) -> Factor:
    if not series.closes:
        return _unavailable("support_resistance", weight, "no candles")
    support, resistance = ind.support_resistance(series.highs, series.lows)
    if not support and not resistance:
        return _unavailable("support_resistance", weight, "no swing pivots found yet")
    price = series.closes[-1]

    nearest_resistance = min((r for r in resistance if r > price), default=None)
    nearest_support = max((s for s in support if s < price), default=None)

    if nearest_resistance is None:
        return Factor("support_resistance", 0.85, weight,
                      "no overhead resistance in recent range")
    headroom = (nearest_resistance - price) / price
    if nearest_support is None:
        return Factor("support_resistance", 0.45, weight,
                      f"resistance {headroom * 100:.1f}% above, no support beneath")

    risk = (price - nearest_support) / price
    ratio = headroom / risk if risk > 0 else 0
    if ratio >= 2:
        score, note = 0.90, f"{ratio:.1f}:1 headroom to nearest levels"
    elif ratio >= 1:
        score, note = 0.60, f"{ratio:.1f}:1 headroom"
    else:
        score, note = 0.25, f"only {ratio:.1f}:1 - resistance close overhead"
    return Factor("support_resistance", score, weight, note)


def score_breakout(series: CandleSeries, weight: float, period: int = 20) -> Factor:
    if not series.closes:
        return _unavailable("breakout", weight, "no candles")
    highs = ind.n_period_high(series.highs, period)
    level = highs[-1] if highs else None
    if level is None:
        return _unavailable("breakout", weight, f"not enough candles for a {period}-period high")

    close = series.closes[-1]
    if close <= level:
        distance = (level - close) / level
        return Factor("breakout", 0.30, weight,
                      f"below the {period}-candle high by {distance * 100:.2f}%")

    margin = (close - level) / level
    if margin > 0.15:
        return Factor("breakout", 0.55, weight,
                      f"{margin * 100:.1f}% above breakout - already extended")
    if margin > 0.005:
        return Factor("breakout", 0.95, weight,
                      f"closed {margin * 100:.2f}% above the {period}-candle high")
    return Factor("breakout", 0.65, weight, "only marginally above the breakout level")


def score_trend_direction(closes: list[float], weight: float) -> Factor:
    direction = ind.trend_direction(closes)
    mapping = {
        "bullish": (0.95, "EMAs aligned bullish"),
        "bearish": (0.05, "EMAs aligned bearish"),
        "neutral": (0.45, "no clear trend"),
    }
    score, note = mapping[direction]
    if direction == "neutral" and len(closes) < 200:
        return Factor("trend_direction", score, weight,
                      f"{note} (fewer than 200 candles - read from 9/21)")
    return Factor("trend_direction", score, weight, note)


def score_multi_timeframe(series: CandleSeries, weight: float,
                          higher: Timeframe | None = None) -> Factor:
    """Whether a slower timeframe agrees with the entry."""
    higher = higher or _next_timeframe_up(series.timeframe)
    if higher is None:
        return _unavailable("multi_timeframe", weight,
                            f"no higher timeframe defined above {series.timeframe.value}")
    try:
        htf = resample(series, higher)
    except ValueError as exc:
        return _unavailable("multi_timeframe", weight, str(exc))

    if len(htf) < 25:
        return _unavailable("multi_timeframe", weight,
                            f"only {len(htf)} {higher.value} candles after resampling, need 25")

    direction = ind.trend_direction(htf.closes)
    structure = ind.price_structure(htf.highs, htf.lows)

    if direction == "bullish" and structure == "uptrend":
        return Factor("multi_timeframe", 1.00, weight,
                      f"{higher.value} trend and structure both bullish")
    if direction == "bullish":
        return Factor("multi_timeframe", 0.75, weight, f"{higher.value} trend bullish")
    if direction == "bearish":
        return Factor("multi_timeframe", 0.05, weight,
                      f"{higher.value} trend bearish - entry fights the higher timeframe")
    return Factor("multi_timeframe", 0.45, weight, f"{higher.value} trend neutral")


def _next_timeframe_up(timeframe: Timeframe) -> Timeframe | None:
    ladder = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
    try:
        index = ladder.index(timeframe)
    except ValueError:
        return None
    # One step up is a 3-6x ratio at every rung (H1->H4, M15->H1, H4->D1),
    # which is the conventional confirmation timeframe. Two steps reaches
    # 15-24x, where a 300-candle fetch leaves too few higher-timeframe bars
    # to read a trend from at all.
    return ladder[index + 1] if index + 1 < len(ladder) else None


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def score_signal(
    series: CandleSeries,
    *,
    weights: dict[str, float] | None = None,
    breakout_period: int = 20,
    higher_timeframe: Timeframe | None = None,
) -> SignalScore:
    """Score a long setup from 0 to 100 across every factor.

    The score is the weighted average of the factors, rescaled so that a
    fully neutral reading lands at 50.
    """
    weights = weights or DEFAULT_WEIGHTS
    closes, volumes = series.closes, series.volumes

    factors = [
        score_trend_direction(closes, weights["trend_direction"]),
        score_multi_timeframe(series, weights["multi_timeframe"], higher_timeframe),
        score_ema_stack(closes, weights["ema_stack"]),
        score_macd(closes, weights["macd"]),
        score_relative_volume(volumes, weights["relative_volume"]),
        score_breakout(series, weights["breakout"], breakout_period),
        score_momentum(closes, weights["momentum"]),
        score_price_structure(series, weights["price_structure"]),
        score_rsi(closes, weights["rsi"]),
        score_vwap(series, weights["vwap"]),
        score_support_resistance(series, weights["support_resistance"]),
        score_bollinger(closes, weights["bollinger"]),
        score_volume_spike(volumes, weights["volume_spike"]),
        score_atr_sanity(series, weights["atr_sanity"]),
    ]

    total_weight = sum(f.weight for f in factors)
    if total_weight <= 0:
        raise ValueError("factor weights must sum to a positive number")

    raw = sum(f.score * f.weight for f in factors) / total_weight
    score = _clamp(raw) * 100

    unavailable_weight = sum(f.weight for f in factors if not f.available) / total_weight
    warnings: list[str] = []
    reliable = True
    if unavailable_weight > MAX_UNAVAILABLE_WEIGHT:
        reliable = False
        missing = ", ".join(f.name for f in factors if not f.available)
        warnings.append(
            f"{unavailable_weight:.0%} of the score has no data ({missing}) - "
            "treat this score as uninformative rather than neutral"
        )

    if score >= 60:
        direction = "long"
    elif score <= 40:
        direction = "short"
    else:
        direction = "neutral"

    return SignalScore(
        score=score, direction=direction, factors=factors,
        reliable=reliable, warnings=warnings,
    )
