"""Evaluating every strategy on one opportunity and recording the result.

Called from the buy path AFTER the champion has finished deciding, so
nothing here can influence that decision - by the time this runs, the
champion's outcome is already fixed.

WHY EVERY DECISION IS RECORDED, INCLUDING THE REFUSALS

A dataset of entries only is useless for comparison. Two strategies that
both declined the same token agree, and that agreement is information: it
says the disagreement measured elsewhere is not just noise. More
importantly, recording only entries introduces survivorship bias in the
worst place - a challenger that enters rarely would be judged on its
handful of picks while its restraint went uncounted.

So a row is written for BUY, REJECT and NO_SIGNAL alike, and for a BUY
whose simulated fill failed. A strategy that would have tried and missed
is a different observation from one that never tried.

WHAT THIS CANNOT REACH

No execution client, no risk manager, no cash ledger, no `positions`
table. Sizing is a fixed notional from configuration rather than the risk
manager's answer, because that answer depends on live exposure and reading
it would couple a hypothetical to real state - one refactor away from
moving it.
"""
from __future__ import annotations

import datetime as dt
import logging
import random

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.execution.fill_model import simulate_fill
from app.shadow.challengers import CHAMPION_ID, Challenger, enabled
from app.shadow.opportunity import opportunity_id
from app.signals.scoring import score_signal
from app.strategy.version import current_label

logger = logging.getLogger(__name__)

BUY = "BUY"
REJECT = "REJECT"
NO_SIGNAL = "NO_SIGNAL"
EXIT = "EXIT"
HOLD = "HOLD"


def _regime_parts(market_regime: str | None) -> tuple[str | None, str | None]:
    """Split the stored label into the full regime and its liquidity axis."""
    if not market_regime:
        return None, None
    parts = market_regime.split("/")
    return market_regime, (parts[2] if len(parts) > 2 else None)


def _record(
    db: Session,
    *,
    oid: str,
    strategy_id: str,
    is_champion: bool,
    token_address: str,
    symbol: str,
    chain: str,
    decision: str,
    reason: str,
    score: float | None,
    factors: list,
    market_regime: str | None,
    liquidity_usd: float | None,
    reference_price: float | None,
    fill=None,
) -> models.ShadowDecision | None:
    """Write one observation, or skip it if this pair is already recorded.

    The unique constraint does the deduplication rather than a prior read:
    a check-then-insert races itself when the scanner overlaps its own
    cycle, and the loser of that race would create the duplicate sample
    this is meant to prevent.
    """
    regime, liquidity_regime = _regime_parts(market_regime)
    row = models.ShadowDecision(
        opportunity_id=oid,
        strategy_id=strategy_id,
        strategy_version=current_label(),
        is_champion=is_champion,
        token_address=token_address,
        symbol=symbol,
        chain=chain,
        decision=decision,
        reason=reason,
        signal_score=score,
        score_factors=factors or [],
        market_regime=regime,
        liquidity_regime=liquidity_regime,
        liquidity_usd=liquidity_usd,
        reference_price=reference_price,
        entry_price=fill.fill_price if (fill and fill.filled) else None,
        fill_succeeded=(fill.filled if fill else None),
        fill_failure_reason=(fill.failure_reason or None) if fill else None,
        fee_pct=(fill.fee_pct if fill else None),
        slippage_pct=(
            (fill.total_cost_pct - fill.fee_pct) if fill else None
        ),
        size_usd=settings.SHADOW_POSITION_USD if decision == BUY else None,
    )
    # A SAVEPOINT, not a bare flush. A duplicate must roll back THIS ROW
    # and nothing else: the caller has uncommitted champion work in the
    # same session - a RiskEvent, a pipeline event, the signal itself - and
    # a session-wide rollback would silently destroy it. That would make
    # the shadow system mutate champion state, which is the one thing it
    # must never do, and the damage would show up as a missing audit row
    # nobody connected to this code.
    try:
        with db.begin_nested():
            db.add(row)
            db.flush()
    except IntegrityError:
        # Already recorded for this (opportunity, strategy). A retry, a
        # duplicated webhook or an overlapping scan - not a new sample.
        logger.debug("shadow decision already recorded for %s/%s", oid, strategy_id)
        return None
    return row


def _open_hypothetical(
    db: Session, decision: models.ShadowDecision, market_regime: str | None
) -> models.ShadowPosition | None:
    """Open a hypothetical position for a filled BUY.

    Separate table from `positions` on purpose - see app/shadow/__init__.py.
    """
    if decision.decision != BUY or not decision.fill_succeeded or not decision.entry_price:
        return None
    position = models.ShadowPosition(
        decision_id=decision.id,
        opportunity_id=decision.opportunity_id,
        strategy_id=decision.strategy_id,
        token_address=decision.token_address,
        symbol=decision.symbol,
        market_regime=market_regime,
        opened_at=decision.decided_at or dt.datetime.now(dt.timezone.utc),
        entry_price=decision.entry_price,
        size_usd=decision.size_usd,
        fees_pct=decision.fee_pct,
        slippage_pct=decision.slippage_pct,
    )
    db.add(position)
    return position


def _evaluate_challenger(
    challenger: Challenger, series, liquidity_usd: float | None
) -> tuple[str, str, float | None, list]:
    """Score one opportunity with a challenger's parameters.

    Uses the same scorer the champion uses, with a different weight map.
    A challenger that needed different code could not be compared on equal
    terms, because it would not have been through the same gates.
    """
    if series is None or not len(series):
        return NO_SIGNAL, "no candle history to score", None, []

    result = score_signal(series, weights=challenger.weights())
    factors = result.as_dict().get("factors", [])
    threshold = challenger.threshold()

    if not result.reliable:
        return REJECT, f"score {result.score:.1f} unreliable - too much missing data", result.score, factors
    if result.score < threshold:
        return REJECT, f"score {result.score:.1f} below its threshold {threshold:.1f}", result.score, factors
    return BUY, f"score {result.score:.1f} at or above {threshold:.1f}", result.score, factors


def record_opportunity(
    db: Session,
    *,
    token_address: str,
    symbol: str,
    chain: str,
    reference_price: float | None,
    observed_at: dt.datetime | None = None,
    market_regime: str | None = None,
    liquidity_usd: float | None = None,
    champion_decision: str = NO_SIGNAL,
    champion_reason: str = "",
    champion_score: float | None = None,
    champion_factors: list | None = None,
    series=None,
    rng: random.Random | None = None,
    event_id: str | None = None,
) -> dict:
    """Record what every strategy would have done about one opportunity.

    `event_id` is the caller's own canonical id for this look, when it has
    one - a scanner run id, a webhook delivery id. Passing it makes the
    identity exact instead of derived; leaving it out falls back to the
    mint, the time bucket and a coarse market snapshot.

    Returns a summary for logging and tests. Never raises into the caller:
    a fault in the shadow system must not be able to cost a real paper
    entry the champion already decided on.
    """
    summary = {"opportunity_id": None, "recorded": 0, "skipped_duplicate": 0}
    if not settings.SHADOW_ENABLED or not token_address:
        return summary

    observed_at = observed_at or dt.datetime.now(dt.timezone.utc)
    # Depth is part of the identity, not just price. The same quote against
    # a pool that has been drained is not the same chance to trade, and the
    # fill model prices it completely differently.
    oid = opportunity_id(
        token_address, observed_at, event_id=event_id,
        snapshot={"price": reference_price, "liquidity": liquidity_usd},
    )
    summary["opportunity_id"] = oid
    rng = rng or random.Random()

    def fill_for(decision: str):
        if decision != BUY or not reference_price:
            return None
        return simulate_fill(
            side="buy",
            reference_price=reference_price,
            trade_usd=settings.SHADOW_POSITION_USD,
            liquidity_usd=liquidity_usd,
            rng=rng,
        )

    arms: list[tuple[str, bool, str, str, float | None, list]] = [
        (CHAMPION_ID, True, champion_decision, champion_reason,
         champion_score, champion_factors or []),
    ]
    for challenger in enabled():
        decision, reason, score, factors = _evaluate_challenger(
            challenger, series, liquidity_usd
        )
        arms.append((challenger.strategy_id, False, decision, reason, score, factors))

    for strategy_id, is_champion, decision, reason, score, factors in arms:
        row = _record(
            db, oid=oid, strategy_id=strategy_id, is_champion=is_champion,
            token_address=token_address, symbol=symbol, chain=chain,
            decision=decision, reason=reason, score=score, factors=factors,
            market_regime=market_regime, liquidity_usd=liquidity_usd,
            reference_price=reference_price, fill=fill_for(decision),
        )
        if row is None:
            summary["skipped_duplicate"] += 1
            continue
        summary["recorded"] += 1
        _open_hypothetical(db, row, market_regime)

    return summary
