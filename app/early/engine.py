"""One candidate in, a decision plus its reasoning out.

The decision is one of three:

    PAPER_BUY   every gate cleared and the token is still enterable
    WATCH       promising, but not confirmed - keep looking at it
    SKIP        disqualified, with the disqualifying reason attached

WATCH is the point of the whole design. Without it the engine faces a
false choice on every promising-but-unconfirmed token: buy now and chase,
or discard and miss the move. WATCH keeps it under observation and enters
only if confirmation arrives BEFORE the token becomes overextended -
which is a race the bot can only run if it is looking.

ORDER OF EVALUATION IS LOAD-BEARING

    security -> data quality -> early score -> late-entry -> technical

Security first and absolutely: an outstanding early signal must never
override a critical security failure, so the security verdict is checked
before anything else is even computed. Data quality second, because a
score built on missing inputs is not a low score, it is no score.
Late-entry before technical, because a token that is already gone should
be rejected for being late rather than being given a second chance to pass
on a technical reading that is strong precisely BECAUSE it is late.

DEFAULT POSTURE: the engine cannot open positions.

EARLY_SIGNAL_MAY_TRADE defaults to false, so the best possible outcome
here is WATCH and the existing technical strategy remains the only thing
that trades. The weights in score.py are unvalidated priors, and trading
on them before app/analysis/early_calibration.py shows the score separates
outcomes would be acting on a guess.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.data.candles import CandleSeries
from app.early import features as feature_mod
from app.early.classifier import Classification, MomentumClass, classify
from app.early.late_entry import LateEntryRisk, Stage, assess
from app.early.score import EarlyScore, score_early_opportunity
from app.services.price_feed import MarketSnapshot

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    PAPER_BUY = "PAPER_BUY"
    WATCH = "WATCH"
    SKIP = "SKIP"


@dataclass
class EarlyVerdict:
    decision: Decision
    reason: str
    early: EarlyScore | None = None
    late: LateEntryRisk | None = None
    momentum: Classification | None = None
    features: feature_mod.EarlyFeatures | None = None
    technical_score: float | None = None
    security_score: float | None = None
    market_quality_score: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def stage(self) -> Stage | None:
        return self.late.stage if self.late else None

    @property
    def early_score(self) -> float | None:
        return self.early.score if self.early else None

    @property
    def late_risk(self) -> float | None:
        return self.late.risk if self.late else None

    def as_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "stage": self.stage.value if self.stage else None,
            "early_score": round(self.early_score, 1) if self.early_score is not None else None,
            "technical_score": self.technical_score,
            "security_score": self.security_score,
            "market_quality_score": self.market_quality_score,
            "late_entry_risk": round(self.late_risk, 1) if self.late_risk is not None else None,
            "momentum_class": self.momentum.label.value if self.momentum else None,
            "momentum_reason": self.momentum.reason if self.momentum else None,
            "early_detail": self.early.as_dict() if self.early else None,
            "late_detail": self.late.as_dict() if self.late else None,
            "notes": list(self.notes),
        }

    def explain(self) -> str:
        """The full chain of reasoning, for the token page and the log."""
        lines = [f"{self.decision.value}: {self.reason}"]
        if self.early:
            lines.append("")
            lines.append(self.early.breakdown())
        if self.late:
            lines.append("")
            lines.append(self.late.summary())
        if self.momentum:
            lines.append(f"  classified {self.momentum.label.value}: {self.momentum.reason}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def evaluate(
    *,
    series: CandleSeries | None,
    market: MarketSnapshot | None,
    observations: list | None = None,
    security_passed: bool = True,
    security_reason: str = "",
    security_score: float | None = None,
    technical_score: float | None = None,
    market_quality_score: float | None = None,
    weights: dict[str, float] | None = None,
) -> EarlyVerdict:
    """Judge one candidate.

    `security_passed` is the rug engine's binary verdict and it is checked
    FIRST, before any early feature is computed - an excellent early signal
    must never be able to override it, and the cheapest way to guarantee
    that is to never let the two meet.
    """
    # --- 1. security. Absolute, and checked before anything else. --------
    if not security_passed:
        return EarlyVerdict(
            Decision.SKIP,
            f"security failure - no early signal overrides this: {security_reason or 'rug check failed'}",
            security_score=security_score,
        )

    extracted = feature_mod.extract(series=series, market=market, observations=observations)
    early = score_early_opportunity(extracted, weights=weights)
    late = assess(extracted)
    momentum = classify(extracted)

    verdict = EarlyVerdict(
        Decision.SKIP, "", early=early, late=late, momentum=momentum, features=extracted,
        technical_score=technical_score, security_score=security_score,
        market_quality_score=market_quality_score,
    )

    # --- 2. data quality. Missing inputs are not a low score. ------------
    if not early.reliable:
        verdict.decision = Decision.SKIP
        verdict.reason = (
            f"early score {early.score:.0f} is unreliable - "
            + (early.warnings[0] if early.warnings else "too much missing input data")
        )
        return verdict

    # --- 3. shape. A suspicious pattern disqualifies outright. -----------
    if momentum.label is MomentumClass.SUSPICIOUS:
        verdict.decision = Decision.SKIP
        verdict.reason = f"suspicious activity pattern: {momentum.reason}"
        return verdict

    # --- 4. late entry. Vetoes regardless of how strong the score is. ----
    if late.blocking:
        verdict.decision = Decision.SKIP
        verdict.reason = (
            f"{late.stage.value} - too late to enter even at early score {early.score:.0f}. "
            f"{late.summary()}"
        )
        return verdict

    # --- 5. is it interesting at all? ------------------------------------
    if early.score < settings.EARLY_SIGNAL_WATCH_THRESHOLD:
        verdict.decision = Decision.SKIP
        verdict.reason = (
            f"early score {early.score:.0f} below the watch threshold "
            f"{settings.EARLY_SIGNAL_WATCH_THRESHOLD:.0f}"
        )
        return verdict

    # --- 6. confirmation --------------------------------------------------
    confirmed = (
        early.score >= settings.EARLY_SIGNAL_CONFIRM_THRESHOLD
        and momentum.label.preferred
        and late.stage in (Stage.DEVELOPING, Stage.CONFIRMED)
    )

    if not confirmed:
        verdict.decision = Decision.WATCH
        missing = []
        if early.score < settings.EARLY_SIGNAL_CONFIRM_THRESHOLD:
            missing.append(
                f"score {early.score:.0f} below confirm threshold "
                f"{settings.EARLY_SIGNAL_CONFIRM_THRESHOLD:.0f}"
            )
        if not momentum.label.preferred:
            missing.append(f"pattern is {momentum.label.value}, not accumulation or breakout")
        if late.stage is Stage.EARLY:
            missing.append("nothing has started moving yet")
        verdict.reason = (
            f"WATCH at {late.stage.value} - promising but unconfirmed: " + "; ".join(missing)
        )
        return verdict

    # --- 7. technical confirmation ---------------------------------------
    if settings.EARLY_SIGNAL_REQUIRE_TECHNICAL:
        if technical_score is None:
            verdict.decision = Decision.WATCH
            verdict.reason = (
                f"early score {early.score:.0f} at {late.stage.value}, but no technical score "
                "is available to confirm it"
            )
            return verdict
        if technical_score < settings.MIN_SIGNAL_SCORE_TO_ENTER:
            verdict.decision = Decision.WATCH
            verdict.reason = (
                f"early score {early.score:.0f} at {late.stage.value}, but technical confirmation "
                f"is not there yet ({technical_score:.0f} < {settings.MIN_SIGNAL_SCORE_TO_ENTER:.0f}). "
                "Watching for it to arrive before the token becomes overextended."
            )
            return verdict

    # --- 8. may the engine actually trade? --------------------------------
    if not settings.EARLY_SIGNAL_MAY_TRADE:
        verdict.decision = Decision.WATCH
        verdict.reason = (
            f"CONFIRMED at early score {early.score:.0f} ({late.stage.value}) - but "
            "EARLY_SIGNAL_MAY_TRADE is false, so this is recorded as a watch rather than traded. "
            "The early weights are unvalidated priors until calibration shows the score "
            "separates outcomes."
        )
        verdict.notes.append(
            "This is the switch that turns the early engine from research into a trading signal. "
            "Do not enable it before /research shows a monotonic calibration curve."
        )
        return verdict

    verdict.decision = Decision.PAPER_BUY
    verdict.reason = (
        f"CONFIRMED: early score {early.score:.0f}, {momentum.label.value.lower()} pattern, "
        f"{late.stage.value} stage, late risk {late.risk:.0f}"
        + (f", technical {technical_score:.0f}" if technical_score is not None else "")
    )
    return verdict
