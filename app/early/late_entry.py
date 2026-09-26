"""Late Entry Risk, and the stage ladder.

The anti-chase system. It exists because every momentum signal ever built
has the same failure mode: the conditions that make a token look strongest
are the conditions that appear AFTER most of the move. Volume peaked, RSI
is stretched, price is far above its averages, the breakout is obvious -
and every one of those reads as "confirmation" to a naive score.

So this is a SEPARATE score that runs against the Early Opportunity Score
rather than inside it. Keeping them apart matters: a token can be
genuinely strong (high early score) AND already too far gone to enter
(high late risk), and averaging the two into one number would hide exactly
that case, which is the most common and most expensive one.

    early score  -> "is demand arriving?"
    late risk    -> "did I already miss it?"

The stage ladder turns the pair into one word:

    EARLY          demand is building, price has barely moved
    DEVELOPING     the move is starting, still enterable
    CONFIRMED      breakout underway, entry still defensible
    LATE           most of the move is behind; entry is a chase
    OVEREXTENDED   do not enter at any score

Late risk is deliberately allowed to VETO. A 90 early score with 80 late
risk is not a trade, and no amount of strength on the other side changes
that - which is the whole point of scoring them separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.early.features import EarlyFeatures


class Stage(str, Enum):
    EARLY = "EARLY"
    DEVELOPING = "DEVELOPING"
    CONFIRMED = "CONFIRMED"
    LATE = "LATE"
    OVEREXTENDED = "OVEREXTENDED"

    @property
    def enterable(self) -> bool:
        """Stages where an entry is still defensible."""
        return self in (Stage.EARLY, Stage.DEVELOPING, Stage.CONFIRMED)


@dataclass
class RiskFlag:
    name: str
    points: float          # 0-100 contribution
    reason: str

    def as_dict(self) -> dict:
        return {"name": self.name, "points": round(self.points, 1), "reason": self.reason}


@dataclass
class LateEntryRisk:
    risk: float            # 0 = early, 100 = the move is over
    flags: list[RiskFlag] = field(default_factory=list)
    stage: Stage = Stage.EARLY
    assessable: bool = True

    @property
    def blocking(self) -> bool:
        return not self.stage.enterable

    def as_dict(self) -> dict:
        return {
            "risk": round(self.risk, 1),
            "stage": self.stage.value,
            "assessable": self.assessable,
            "blocking": self.blocking,
            "flags": [f.as_dict() for f in self.flags],
        }

    def summary(self) -> str:
        if not self.flags:
            return f"{self.stage.value} - no late-entry warnings ({self.risk:.0f}/100 risk)"
        top = sorted(self.flags, key=lambda f: f.points, reverse=True)[:3]
        return (
            f"{self.stage.value} - late-entry risk {self.risk:.0f}/100: "
            + "; ".join(f.reason for f in top)
        )


# How much each warning contributes. Extension and stretch dominate,
# because they are the two that most reliably mark a finished move.
FLAG_WEIGHTS = {
    "price_extended": 30.0,
    "above_breakout": 22.0,
    "rsi_stretched": 16.0,
    "far_above_vwap": 12.0,
    "ema_separated": 10.0,
    "volume_peaked": 14.0,
    "buy_pressure_falling": 12.0,
    "liquidity_deteriorating": 14.0,
    "momentum_slowing": 10.0,
}


def assess(features: EarlyFeatures) -> LateEntryRisk:
    """How much of the move has already happened?"""
    flags: list[RiskFlag] = []

    # --- price already ran ---------------------------------------------
    long_return = features.value("return_long")
    if long_return is not None:
        if long_return > 120:
            flags.append(RiskFlag("price_extended", FLAG_WEIGHTS["price_extended"],
                                  f"already +{long_return:.0f}% over the lookback - the move happened"))
        elif long_return > 50:
            flags.append(RiskFlag("price_extended", FLAG_WEIGHTS["price_extended"] * 0.6,
                                  f"already +{long_return:.0f}% - much of the move is behind"))
        elif long_return > 25:
            flags.append(RiskFlag("price_extended", FLAG_WEIGHTS["price_extended"] * 0.3,
                                  f"+{long_return:.0f}% so far - partly extended"))

    breakout = features.value("breakout_proximity")
    if breakout is not None and breakout > 8:
        share = min((breakout - 8) / 40, 1.0)
        flags.append(RiskFlag("above_breakout", FLAG_WEIGHTS["above_breakout"] * (0.4 + 0.6 * share),
                              f"{breakout:+.0f}% above the range high"))

    # --- stretched indicators -------------------------------------------
    rsi = features.value("rsi_level")
    if rsi is not None and rsi > 72:
        share = min((rsi - 72) / 18, 1.0)
        flags.append(RiskFlag("rsi_stretched", FLAG_WEIGHTS["rsi_stretched"] * (0.4 + 0.6 * share),
                              f"RSI {rsi:.0f} - stretched"))

    vwap = features.value("vwap_position")
    if vwap is not None and vwap > 12:
        share = min((vwap - 12) / 40, 1.0)
        flags.append(RiskFlag("far_above_vwap", FLAG_WEIGHTS["far_above_vwap"] * (0.4 + 0.6 * share),
                              f"{vwap:+.0f}% above VWAP"))

    separation = features.value("ema_separation")
    if separation is not None and separation > 6:
        share = min((separation - 6) / 20, 1.0)
        flags.append(RiskFlag("ema_separated", FLAG_WEIGHTS["ema_separated"] * (0.4 + 0.6 * share),
                              f"EMA9 is {separation:+.1f}% above EMA21 - overextended"))

    # --- the move is losing its fuel -------------------------------------
    volume_short = features.value("volume_accel_short")
    if volume_short is not None and volume_short < 0.7:
        flags.append(RiskFlag("volume_peaked", FLAG_WEIGHTS["volume_peaked"],
                              f"volume is {volume_short:.2f}x its previous window - already peaked"))

    pressure_change = features.value("buy_pressure_change")
    if pressure_change is not None and pressure_change < -0.04:
        flags.append(RiskFlag("buy_pressure_falling", FLAG_WEIGHTS["buy_pressure_falling"],
                              f"buy share fell {pressure_change:+.1%} - demand is leaving"))

    liquidity = features.value("liquidity_growth")
    if liquidity is not None and liquidity < 0.92:
        flags.append(RiskFlag("liquidity_deteriorating", FLAG_WEIGHTS["liquidity_deteriorating"],
                              f"pool depth fell to {liquidity:.2f}x - liquidity is being pulled"))

    smoothness = features.value("acceleration_smoothness")
    if smoothness is not None and smoothness < 0.35 and (long_return or 0) > 20:
        flags.append(RiskFlag("momentum_slowing", FLAG_WEIGHTS["momentum_slowing"],
                              f"only {smoothness:.0%} of recent bars closed up after a large move"))

    risk = min(sum(f.points for f in flags), 100.0)

    # Whether the assessment can be trusted at all. Without price history
    # the two dominant flags cannot fire, so a low risk score would mean
    # "we couldn't look" rather than "it's early".
    assessable = features.get("return_long").available and features.get("breakout_proximity").available

    return LateEntryRisk(risk=risk, flags=flags, stage=classify_stage(features, risk, assessable),
                         assessable=assessable)


def classify_stage(features: EarlyFeatures, risk: float, assessable: bool) -> Stage:
    """Where in the move this token is.

    Reads risk together with how far price has actually travelled, because
    the two disagree in an informative way: a token up 5% with several
    warnings is a weak setup, not a late one, and calling it OVEREXTENDED
    would discard a perfectly ordinary early candidate.
    """
    if not assessable:
        # No price history to place it. LATE rather than EARLY, because
        # "we cannot tell" must not become an invitation to enter.
        return Stage.LATE

    travelled = features.value("return_long") or 0.0
    breakout = features.value("breakout_proximity")

    if risk >= 70 or travelled > 150:
        return Stage.OVEREXTENDED
    if risk >= 45 or travelled > 60:
        return Stage.LATE
    if breakout is not None and breakout > 3:
        return Stage.CONFIRMED
    if travelled > 8 or (features.value("volume_accel_short") or 0) > 1.5:
        return Stage.DEVELOPING
    return Stage.EARLY
