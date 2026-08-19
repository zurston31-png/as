"""Pipeline stage instrumentation.

Every token that enters the bot passes through an explicit, named sequence
of stages, and every stage records what it decided and why:

    DISCOVERED       a listing source returned this mint
    PRESCREEN        liquidity / volume / age / transactions / sell pressure
    RISK             kill switch, portfolio limits, exposure, cooldowns
    HISTORY          enough trustworthy candles exist to score it
    TECHNICAL_SCORE  the 0-100 signal score  (recorded even when rejected)
    MARKET_QUALITY   the 0-100 tradeability score  (likewise)
    SECURITY         the rug engine's verdict and 0-100 risk score
    PAPER_EXECUTION  the simulated fill - which can fail
    OPEN_POSITION    a position now exists
    EXIT             the position closed, and why

The order above is the order the buy path actually runs them in, and the
funnel relies on that: RISK comes before scoring because rejecting a
candidate on a portfolio limit costs nothing, while scoring it costs a pool
resolution and a candle fetch. A funnel listing the stages in any other
order would misattribute where candidates died.

Two design decisions carry the weight here.

APPEND ONLY. Each stage writes a new row rather than updating the token's
current state. "Where is this token now?" and "how many tokens died at
each stage last week, and to which threshold?" are different questions,
and only the second one tells you what to change. The existing
ScannedToken row still answers the first.

SCORES ARE RECORDED FOR REJECTED TOKENS. A dataset containing only the
setups that cleared the threshold cannot tell you whether the threshold
sits in the right place - every survivor scored above it, by construction.
Recording the score of everything the engine scored, and only then
filtering, is what makes the question answerable at all. It is also what
prevents the milder version of survivorship bias in which the research set
quietly becomes "tokens we liked".

Recording must never be able to break trading. Every helper here swallows
its own failures: losing an audit row is a bad day for research, while an
exception escaping into the buy path is a bad day for the portfolio.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from app import models
from app.strategy.version import current_label

logger = logging.getLogger(__name__)

# Stage names, in pipeline order. The order is used by the funnel to
# compute stage-to-stage conversion, so it is data, not decoration.
DISCOVERED = "DISCOVERED"
PRESCREEN = "PRESCREEN"
HISTORY = "HISTORY"
TECHNICAL_SCORE = "TECHNICAL_SCORE"
MARKET_QUALITY = "MARKET_QUALITY"
SECURITY = "SECURITY"
RISK = "RISK"
PAPER_EXECUTION = "PAPER_EXECUTION"
OPEN_POSITION = "OPEN_POSITION"
EXIT = "EXIT"

# Pipeline order, and it must match the order app/services/trading_service.py
# actually evaluates them in - the funnel computes stage-to-stage conversion
# from this sequence, so a wrong order silently misreports which gate is
# costing candidates.
STAGE_ORDER: tuple[str, ...] = (
    DISCOVERED,
    PRESCREEN,
    RISK,
    HISTORY,
    TECHNICAL_SCORE,
    MARKET_QUALITY,
    SECURITY,
    PAPER_EXECUTION,
    OPEN_POSITION,
    EXIT,
)

# EXIT is an OUTCOME, not a filter. Its `passed` flag means "this trade was
# profitable", so counting it as a funnel step would report the loss rate as
# a rejection rate and name it the pipeline's bottleneck - which it can
# never be, since every position that reaches it has already been opened.
TERMINAL_STAGES: tuple[str, ...] = (EXIT,)

# Stages that actually filter candidates. Conversion and bottleneck analysis
# use these.
FILTER_STAGES: tuple[str, ...] = tuple(s for s in STAGE_ORDER if s not in TERMINAL_STAGES)

# Stages that produce a 0-100 score worth analysing a distribution over.
SCORED_STAGES: tuple[str, ...] = (TECHNICAL_SCORE, MARKET_QUALITY, SECURITY)


def record(
    db: Session,
    *,
    stage: str,
    symbol: str,
    token_address: str | None,
    passed: bool,
    reason: str = "",
    chain: str = "solana",
    score: float | None = None,
    detail: dict | None = None,
    signal_id: int | None = None,
) -> models.PipelineEvent | None:
    """Append one stage event. Never raises into the caller.

    Returns the row (not yet committed - the caller's transaction owns it)
    or None if recording failed, which is logged and otherwise ignored:
    an audit gap costs research, an exception here would cost a trade.
    """
    if stage not in STAGE_ORDER:
        logger.error("refusing to record unknown pipeline stage %r", stage)
        return None

    try:
        event = models.PipelineEvent(
            occurred_at=dt.datetime.now(dt.timezone.utc),
            token_address=token_address,
            symbol=symbol,
            chain=chain,
            stage=stage,
            passed=passed,
            reason=reason or "",
            score=score,
            detail=detail or {},
            strategy_version=current_label(),
            signal_id=signal_id,
        )
        db.add(event)
        return event
    except Exception:
        logger.exception("failed to record pipeline event %s for %s", stage, symbol)
        return None


def record_many(db: Session, events: list[dict]) -> int:
    """Record a batch of events, e.g. one DISCOVERED row per scan result.

    Returns how many were recorded. Used by the scanner, where a cycle can
    surface hundreds of mints and one row each is still cheaper than the
    network call that produced them.
    """
    written = 0
    for payload in events:
        if record(db, **payload) is not None:
            written += 1
    return written
