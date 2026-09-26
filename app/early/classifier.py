"""Separating healthy momentum from a pump.

Transparent rules, not a model. Two reasons, and both are about this
specific problem rather than a general preference for simplicity:

  There is no labelled training data. Fitting a classifier would mean
  inventing the labels it learns from, and it would then reproduce those
  inventions with a confidence score attached.

  When the engine declines a token, the reason has to be legible. "The
  model said 0.31" cannot be argued with, checked, or improved. "78% of
  the window's volume is in one bar" can be.

If enough labelled outcomes eventually accumulate - which
app/analysis/early_calibration.py is what would produce - a fitted model
becomes worth trying, and it should be judged against these rules rather
than assumed to beat them.

The five classes:

    ACCUMULATION   volume and participation rising, price still quiet.
                   The best case, and the rarest.
    BREAKOUT       controlled expansion out of a range on broad volume.
    LATE_MOMENTUM  real move, but most of it has happened.
    PARABOLIC      violent, near-vertical. Sometimes real, never enterable
                   at this point on this bot's exit logic.
    DISTRIBUTION   price falling. Whatever volume exists is exit volume,
                   and it is the mirror image of accumulation rather than
                   a quiet version of it.
    SUSPICIOUS     the shape does not add up - volume in one print,
                   one-sided flow, or depth moving against price.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.early.features import EarlyFeatures


class MomentumClass(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    BREAKOUT = "BREAKOUT"
    LATE_MOMENTUM = "LATE_MOMENTUM"
    PARABOLIC = "PARABOLIC"
    DISTRIBUTION = "DISTRIBUTION"
    SUSPICIOUS = "SUSPICIOUS"
    UNKNOWN = "UNKNOWN"

    @property
    def preferred(self) -> bool:
        """The two the engine is trying to find."""
        return self in (MomentumClass.ACCUMULATION, MomentumClass.BREAKOUT)


@dataclass
class Classification:
    label: MomentumClass
    reason: str
    confidence: str = "rule-based"

    def as_dict(self) -> dict:
        return {"label": self.label.value, "reason": self.reason, "confidence": self.confidence}


def classify(features: EarlyFeatures) -> Classification:
    """Which shape this token's activity has.

    Checked in order of how strongly each pattern should override the
    others: suspicious first (it disqualifies regardless of everything
    else), then parabolic, then the rest.
    """
    volume_short = features.value("volume_accel_short")
    steadiness = features.value("volume_steadiness")
    travelled = features.value("return_long")
    recent = features.value("return_short")
    pressure = features.value("buy_pressure")
    liquidity = features.value("liquidity_growth")
    breakout = features.value("breakout_proximity")
    compression = features.value("range_compression")
    smoothness = features.value("acceleration_smoothness")

    if travelled is None and volume_short is None:
        return Classification(MomentumClass.UNKNOWN,
                              "neither price history nor volume history is available")

    # --- suspicious: the shape contradicts itself ------------------------
    if steadiness is not None and steadiness < 0.15 and (volume_short or 0) > 2:
        return Classification(
            MomentumClass.SUSPICIOUS,
            f"{(1 - steadiness):.0%} of the volume surge is a single bar - one actor, not a market",
        )
    if pressure is not None and pressure > 0.95 and (volume_short or 0) > 1.5:
        return Classification(
            MomentumClass.SUSPICIOUS,
            f"{pressure:.0%} of trades are buys with almost no sells - one-sided flow of this "
            "purity is usually wash trading or a bot ladder, not demand",
        )
    if liquidity is not None and liquidity < 0.85 and (recent or 0) > 15:
        return Classification(
            MomentumClass.SUSPICIOUS,
            f"price is up {recent:.0f}% while pool depth fell to {liquidity:.2f}x - "
            "liquidity leaving into a rising price is the shape of an exit",
        )

    # --- distribution: falling price is not a quiet one -------------------
    # This check has to come before accumulation. "Quiet" was originally a
    # one-sided test (travelled < 25), which a token down 60% satisfies -
    # so a collapsing chart with any volume at all was being classified as
    # ACCUMULATION at stage EARLY with zero late-entry risk, and would have
    # gone straight onto the watchlist as a promising candidate. Volume
    # during a decline is people leaving.
    if travelled is not None and travelled < -15:
        return Classification(
            MomentumClass.DISTRIBUTION,
            f"{travelled:+.0f}% over the lookback - falling, so any volume here is exit volume",
        )
    if recent is not None and recent < -12:
        return Classification(
            MomentumClass.DISTRIBUTION,
            f"{recent:+.0f}% in the last few bars - actively breaking down",
        )

    # --- parabolic --------------------------------------------------------
    if (recent is not None and recent > 40) or (travelled is not None and travelled > 200):
        return Classification(
            MomentumClass.PARABOLIC,
            f"{recent:+.0f}% in the last few bars" if recent and recent > 40
            else f"{travelled:+.0f}% over the lookback - near-vertical",
        )

    # --- late momentum ----------------------------------------------------
    if travelled is not None and travelled > 60:
        return Classification(
            MomentumClass.LATE_MOMENTUM,
            f"already +{travelled:.0f}% - a real move, but most of it has happened",
        )
    if breakout is not None and breakout > 15:
        return Classification(
            MomentumClass.LATE_MOMENTUM,
            f"{breakout:+.0f}% above the range high - the breakout is well underway",
        )

    # --- breakout ---------------------------------------------------------
    # Anything reaching this point is NOT late: the travelled > 60% and
    # breakout > 15% checks above have already claimed those. So the test
    # here is only "is price at its range high with volume behind it?",
    # and volume merely has to be holding up rather than exploding.
    # Requiring a surge here was a hole in the taxonomy: a healthy trend
    # sitting at its high on steady volume matched nothing at all, fell
    # through to UNKNOWN, and could therefore never confirm.
    if (
        breakout is not None and -8 <= breakout <= 15
        and (volume_short is None or volume_short >= 1.0)
        and (steadiness is None or steadiness > 0.4)
    ):
        return Classification(
            MomentumClass.BREAKOUT,
            f"at the range high ({breakout:+.1f}%)"
            + (f" on {volume_short:.1f}x volume" if volume_short else "")
            + " spread across multiple bars",
        )

    # --- accumulation ------------------------------------------------------
    # Below the range high, price still quiet, volume building. The ceiling
    # is 25% rather than 15% because the gap between "quiet" and the 60%
    # late-momentum floor was leaving ordinary developing moves unclassified.
    quiet = travelled is not None and -10 < travelled < 25   # near flat, in BOTH directions
    building = (volume_short or 0) > 1.05
    coiled = compression is not None and compression > 0.5
    if quiet and building and (coiled or (smoothness or 0) > 0.5):
        return Classification(
            MomentumClass.ACCUMULATION,
            f"volume {volume_short:.1f}x while price is still quiet ({travelled:+.1f}%)"
            + (" and the range is compressing" if coiled else ""),
        )

    if quiet and not building:
        return Classification(MomentumClass.UNKNOWN, "quiet on every measure - nothing is happening")

    return Classification(
        MomentumClass.UNKNOWN,
        "activity does not match a recognised pattern"
        + (f" ({travelled:+.0f}% travelled)" if travelled is not None else ""),
    )
