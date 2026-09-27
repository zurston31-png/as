"""The Early Opportunity Score, 0-100.

A FOURTH score, deliberately separate from the other three, because they
answer different questions and averaging them would destroy all four:

    security score   "will this rug?"
    market quality   "can I actually trade it?"
    technical score  "is this a good setup right now?"
    early score      "is demand ARRIVING?"                 <- here

The distinction from the technical score is the one that matters. A
technical score reads the state of the chart: trend, structure, momentum
levels. This reads the DERIVATIVE - is volume accelerating, is transaction
rate rising, is buy pressure building, is the range coiling. A token can
have a mediocre technical score precisely because the move has not happened
yet, which is the entire situation this score exists to find.

    THE WEIGHTS BELOW ARE UNVALIDATED PRIORS.

They come from reasoning about microstructure, not from measured outcomes,
because no outcome data exists yet. That makes this score a hypothesis. The
engine treats it as one: EARLY_SIGNAL_MAY_TRADE defaults to false, so a
high early score can raise a token to WATCH and can never open a position
on its own.

app/analysis/early_calibration.py is what turns the hypothesis into a
finding - or refutes it. If higher scores do not precede better outcomes,
the correct response is to simplify or delete this module, not to reweight
it until the backtest looks good.

Scoring convention matches the rest of the bot: each factor scores 0.0 to
1.0 with 0.5 meaning "no opinion", missing inputs are marked unavailable
rather than scored as fine, and too much missing data marks the whole score
unreliable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.early.features import EarlyFeatures

NEUTRAL = 0.5
MAX_UNAVAILABLE_WEIGHT = 0.40

# --- UNVALIDATED PRIORS -----------------------------------------------------
# Ordered by how directly each one measures "demand is arriving now" rather
# than "demand arrived earlier". The flow group (volume + transaction +
# buy pressure) carries roughly half the weight because it is the part that
# leads price; the structure group is context.
DEFAULT_WEIGHTS: dict[str, float] = {
    "volume_acceleration": 0.18,
    "transaction_acceleration": 0.14,
    "buy_pressure": 0.14,
    "volume_quality": 0.10,
    "liquidity_quality": 0.10,
    "momentum_acceleration": 0.10,
    "price_structure": 0.09,
    "breakout_position": 0.08,
    "relative_volume": 0.07,
}


@dataclass
class EarlyFactor:
    name: str
    score: float
    weight: float
    reason: str
    available: bool = True

    @property
    def points(self) -> float:
        return self.score * self.weight * 100

    @property
    def max_points(self) -> float:
        return self.weight * 100

    def as_dict(self) -> dict:
        return {
            "name": self.name, "score": round(self.score, 3), "weight": self.weight,
            "points": round(self.points, 2), "reason": self.reason, "available": self.available,
        }


@dataclass
class EarlyScore:
    score: float
    factors: list[EarlyFactor] = field(default_factory=list)
    reliable: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def unavailable(self) -> list[EarlyFactor]:
        return [f for f in self.factors if not f.available]

    @property
    def strengths(self) -> list[EarlyFactor]:
        return sorted([f for f in self.factors if f.available and f.score >= 0.7],
                      key=lambda f: f.points, reverse=True)

    @property
    def concerns(self) -> list[EarlyFactor]:
        return sorted([f for f in self.factors if f.available and f.score < 0.4],
                      key=lambda f: f.points)

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "reliable": self.reliable,
            "warnings": list(self.warnings),
            "factors": [f.as_dict() for f in self.factors],
        }

    def breakdown(self) -> str:
        lines = [f"Early opportunity {self.score:.1f}/100"]
        if not self.reliable:
            lines.append("  UNRELIABLE: " + "; ".join(self.warnings))
        for f in sorted(self.factors, key=lambda f: f.points, reverse=True):
            marker = " " if f.available else "?"
            lines.append(f"  {marker} {f.name:<26} {f.points:5.1f}/{f.max_points:4.1f}  {f.reason}")
        return "\n".join(lines)


def _unavailable(name: str, weight: float, reason: str) -> EarlyFactor:
    return EarlyFactor(name, NEUTRAL, weight, reason, available=False)


def _band(value: float, bands: list[tuple[float, float]], above: float) -> float:
    """Map a value through ascending (upper_bound, score) bands."""
    for upper, score in bands:
        if value < upper:
            return score
    return above


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------

def score_volume_acceleration(f: EarlyFeatures, weight: float) -> EarlyFactor:
    """Rising volume, with an explicit ceiling.

    A 3x pickup is interest arriving. A 40x pickup is not 13 times better -
    it is a token whose move is already underway and probably a single
    actor, which is the opposite of early.
    """
    short = f.value("volume_accel_short")
    medium = f.value("volume_accel_medium")
    if short is None and medium is None:
        return _unavailable("volume_acceleration", weight, "no volume history")

    parts = [v for v in (short, medium) if v is not None]
    ratio = sum(parts) / len(parts)
    score = _band(ratio, [(0.8, 0.15), (1.0, 0.35), (1.5, 0.65), (3.0, 1.0), (8.0, 0.55)], above=0.15)
    if ratio >= 8.0:
        detail = f"{ratio:.1f}x - already exploding, not early"
    elif ratio >= 3.0:
        detail = f"{ratio:.1f}x - strong pickup, verging on late"
    elif ratio >= 1.5:
        detail = f"{ratio:.1f}x - clear acceleration"
    elif ratio >= 1.0:
        detail = f"{ratio:.1f}x - mild pickup"
    else:
        detail = f"{ratio:.2f}x - volume is fading"
    return EarlyFactor("volume_acceleration", score, weight, detail)


def score_transaction_acceleration(f: EarlyFeatures, weight: float) -> EarlyFactor:
    change = f.value("txn_rate_change")
    if change is None:
        return _unavailable(
            "transaction_acceleration", weight,
            "needs two stored observations far enough apart to difference",
        )
    score = _band(change, [(0.9, 0.1), (1.0, 0.35), (1.15, 0.7), (1.6, 1.0), (4.0, 0.5)], above=0.2)
    return EarlyFactor(
        "transaction_acceleration", score, weight,
        f"transaction count is {change:.2f}x its previous reading",
    )


def score_buy_pressure(f: EarlyFeatures, weight: float) -> EarlyFactor:
    """Buy dominance, weighted by whether it PERSISTS.

    One observation of 70% buys is a moment. Four in a row is a condition,
    and only the second is evidence of demand rather than of one order.
    """
    pressure = f.value("buy_pressure")
    if pressure is None:
        return _unavailable("buy_pressure", weight, "too few transactions to read a ratio")

    base = _band(pressure, [(0.40, 0.05), (0.48, 0.3), (0.55, 0.6), (0.70, 1.0), (0.85, 0.7)], above=0.35)
    detail = f"{pressure:.0%} buys"

    persistence = f.value("buy_pressure_persistence")
    if persistence is not None:
        # Persistence scales the reading rather than adding to it: buy
        # pressure that appeared once and vanished should not score as
        # two-thirds of sustained pressure.
        base *= 0.55 + 0.45 * persistence
        detail += f", buy-dominant in {persistence:.0%} of recent readings"

    change = f.value("buy_pressure_change")
    if change is not None and change < -0.05:
        base *= 0.7
        detail += f" but falling ({change:+.1%})"

    return EarlyFactor("buy_pressure", min(base, 1.0), weight, detail)


def score_volume_quality(f: EarlyFeatures, weight: float) -> EarlyFactor:
    """Is the volume spread across bars, or one print?"""
    steadiness = f.value("volume_steadiness")
    if steadiness is None:
        return _unavailable("volume_quality", weight, "no volume in the recent window")
    score = _band(steadiness, [(0.2, 0.05), (0.4, 0.3), (0.6, 0.7)], above=1.0)
    concentration = 1 - steadiness
    return EarlyFactor(
        "volume_quality", score, weight,
        f"largest recent bar is {concentration:.0%} of the window's volume"
        + (" - one print, not a market" if concentration > 0.8 else ""),
    )


def score_liquidity_quality(f: EarlyFeatures, weight: float) -> EarlyFactor:
    """Growing depth is good; violently swinging depth is not growth."""
    growth = f.value("liquidity_growth")
    stability = f.value("liquidity_stability")
    if growth is None and stability is None:
        return _unavailable("liquidity_quality", weight, "no liquidity history")

    if growth is None:
        return EarlyFactor("liquidity_quality", 0.5 * (stability or 0.5), weight,
                           "liquidity stable but no growth reading")

    score = _band(growth, [(0.90, 0.05), (0.99, 0.35), (1.02, 0.6), (1.25, 1.0), (2.0, 0.5)], above=0.2)
    detail = f"pool depth {growth:.2f}x its previous reading"
    if stability is not None:
        score *= 0.5 + 0.5 * stability
        if stability < 0.6:
            detail += f", but swinging ({(1 - stability):.0%} range) - unstable, not growing"
    return EarlyFactor("liquidity_quality", min(score, 1.0), weight, detail)


def score_momentum_acceleration(f: EarlyFeatures, weight: float) -> EarlyFactor:
    """EMA slope, RSI crossing up, MACD expanding - all read as turns
    beginning rather than as levels reached."""
    parts: list[tuple[float, str]] = []

    slope = f.value("ema_slope")
    if slope is not None:
        parts.append((_band(slope, [(-0.5, 0.1), (0.0, 0.35), (1.0, 0.75), (5.0, 1.0)], above=0.5),
                      f"EMA9 slope {slope:+.2f}%"))

    crossing = f.value("rsi_crossing_up")
    rsi_level = f.value("rsi_level")
    if crossing is not None and rsi_level is not None:
        if crossing >= 1.0:
            parts.append((1.0, f"RSI crossed up through 50 (now {rsi_level:.0f})"))
        else:
            # A level reading, and high is NOT good here.
            parts.append((_band(rsi_level, [(35, 0.4), (45, 0.6), (60, 0.8), (72, 0.5)], above=0.15),
                          f"RSI {rsi_level:.0f}"))

    macd = f.value("macd_histogram_expanding")
    if macd is not None:
        parts.append((0.9 if macd >= 1.0 else 0.35,
                      "MACD histogram expanding" if macd >= 1.0 else "MACD histogram flat or shrinking"))

    if not parts:
        return _unavailable("momentum_acceleration", weight, "no momentum indicators available")

    score = sum(p[0] for p in parts) / len(parts)
    return EarlyFactor("momentum_acceleration", score, weight, "; ".join(p[1] for p in parts))


def score_price_structure(f: EarlyFeatures, weight: float) -> EarlyFactor:
    """Compression and higher lows - a coil, rather than a chase."""
    parts: list[tuple[float, str]] = []

    comp = f.value("range_compression")
    if comp is not None:
        parts.append((comp, f"range compression {comp:.2f}"))

    highs = f.value("higher_lows")
    if highs is not None:
        parts.append((highs, f"higher lows {highs:.0%}"))

    smooth = f.value("acceleration_smoothness")
    if smooth is not None:
        parts.append((_band(smooth, [(0.35, 0.2), (0.5, 0.6), (0.8, 1.0)], above=0.5),
                      f"{smooth:.0%} of recent bars closed up"))

    if not parts:
        return _unavailable("price_structure", weight, "not enough bars to read structure")

    score = sum(p[0] for p in parts) / len(parts)
    return EarlyFactor("price_structure", score, weight, "; ".join(p[1] for p in parts))


def score_breakout_position(f: EarlyFeatures, weight: float) -> EarlyFactor:
    """Approaching the high scores best. Far ABOVE it scores worst.

    This is the single most direct anti-chase factor in the score: the
    naive version of "breakout detection" rewards being far above the
    range, which is precisely the state of a move that already happened.
    """
    distance = f.value("breakout_proximity")
    if distance is None:
        return _unavailable("breakout_position", weight, "no prior range to measure against")

    if distance < -25:
        return EarlyFactor("breakout_position", 0.2, weight,
                           f"{distance:.1f}% below the range high - nothing is breaking out")
    if distance < -5:
        return EarlyFactor("breakout_position", 0.75, weight,
                           f"{distance:.1f}% below the high - approaching")
    if distance <= 3:
        return EarlyFactor("breakout_position", 1.0, weight,
                           f"{distance:+.1f}% versus the high - right at the breakout")
    if distance <= 15:
        return EarlyFactor("breakout_position", 0.6, weight,
                           f"{distance:+.1f}% above the high - breakout confirmed, partly late")
    return EarlyFactor("breakout_position", 0.1, weight,
                       f"{distance:+.1f}% above the high - the move already happened")


def score_relative_volume(f: EarlyFeatures, weight: float) -> EarlyFactor:
    rvol = f.value("relative_volume")
    if rvol is None:
        return _unavailable("relative_volume", weight, "no volume baseline")
    score = _band(rvol, [(0.8, 0.15), (1.2, 0.4), (2.0, 0.8), (6.0, 1.0), (15.0, 0.5)], above=0.15)
    return EarlyFactor("relative_volume", score, weight,
                       f"{rvol:.1f}x its own average volume")


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

_SCORERS = {
    "volume_acceleration": score_volume_acceleration,
    "transaction_acceleration": score_transaction_acceleration,
    "buy_pressure": score_buy_pressure,
    "volume_quality": score_volume_quality,
    "liquidity_quality": score_liquidity_quality,
    "momentum_acceleration": score_momentum_acceleration,
    "price_structure": score_price_structure,
    "breakout_position": score_breakout_position,
    "relative_volume": score_relative_volume,
}


def score_early_opportunity(
    features: EarlyFeatures, *, weights: dict[str, float] | None = None
) -> EarlyScore:
    """Combine the early factors into one 0-100 score.

    `weights` is injectable so app/research/ can ablate factors without
    editing this module - a factor's weight set to 0 removes it from both
    the numerator and the denominator, which is what makes leave-one-out
    testing meaningful rather than a scale change.
    """
    weights = weights or DEFAULT_WEIGHTS

    factors = [
        _SCORERS[name](features, weight)
        for name, weight in weights.items()
        if name in _SCORERS
    ]

    total_weight = sum(f.weight for f in factors)
    if total_weight <= 0:
        raise ValueError("early-score weights must sum to a positive number")

    raw = sum(f.score * f.weight for f in factors) / total_weight
    score = max(0.0, min(1.0, raw)) * 100

    unavailable_weight = sum(f.weight for f in factors if not f.available) / total_weight
    warnings: list[str] = []
    reliable = True
    if unavailable_weight > MAX_UNAVAILABLE_WEIGHT:
        reliable = False
        missing = ", ".join(f.name for f in factors if not f.available)
        warnings.append(
            f"{unavailable_weight:.0%} of the early score has no data ({missing}) - "
            "unassessable, not average"
        )

    return EarlyScore(score=score, factors=factors, reliable=reliable, warnings=warnings)
