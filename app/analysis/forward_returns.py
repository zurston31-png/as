"""Recording what happened next, for candidates the bot did and did not take.

The calibration table in app/analysis/calibration.py needs an answer key:
for every scored candidate, what the price did over the following minutes
and hours. This module creates those rows when a candidate is scored, and
fills them in later when each horizon comes due.

Why it tracks REJECTED candidates too - the entire point:

    The bot only trades what it already scored highly. Judging the score
    from trades alone therefore asks "did the setups we liked do well?",
    which cannot detect a score that is pure noise: the 55s were never
    given the chance to disagree. Following the rejects is what turns the
    score from an assertion into a measurement.

It is also the cheapest available defence against survivorship bias. The
research set is defined at scoring time, before the outcome is known, so
tokens that later died are in it by construction rather than by whether
anyone remembered to keep them.

WHAT IT NEVER DOES

  * back-fill a missing price with the last known one
  * treat "the token stopped trading" as a 0% return
  * record a horizon before it has elapsed

Each of those would quietly turn a dead token into a flat one, and a flat
token into evidence. A horizon that cannot be measured stays NULL with a
reason attached.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from app import models
from app.analysis.calibration import HORIZONS_MINUTES
from app.config import settings
from app.services import price_feed
from app.strategy.version import current_label

logger = logging.getLogger(__name__)

# Stop chasing a horizon this long after it came due. A token whose feed
# went quiet three days ago is not going to answer, and retrying forever
# turns a research table into a permanent background load.
#
# This is an upper bound only. The binding rule is `lateness_tolerance`
# below, which is always tighter - see why there.
GIVE_UP_AFTER_HOURS = 12.0

# How far past its due time a horizon may be measured and still be called
# that horizon.
#
# This used to be GIVE_UP_AFTER_HOURS alone: a flat twelve hours for every
# horizon. That is defensible for the 1440-minute row and nonsense for the
# 5-minute one. The resolver polls every FORWARD_RETURN_RESOLVE_INTERVAL_
# SECONDS, so any interruption - a laptop sleeping, a restart, a backlog
# past FORWARD_RETURN_BATCH_LIMIT - meant pending 5-minute rows were later
# resolved against a price hours old and stored as the 5-minute return.
# Calibration buckets by horizon, so the short columns silently filled with
# returns that were not those horizons, and nothing in the table recorded
# that it had happened.
#
# A quarter of the horizon's own length keeps the label honest: a "60m"
# return is measured between 60 and 75 minutes out, never at 12 hours. Rows
# outside it are sealed unmeasurable rather than filled, in keeping with
# the rule that a measurement which cannot be taken is never invented.
MAX_LATENESS_FRACTION = 0.25


def lateness_tolerance_minutes(horizon_minutes: int) -> float:
    """Minutes past due that still count as measuring `horizon_minutes`."""
    return horizon_minutes * MAX_LATENESS_FRACTION

# Coarse outcome labels, by close-to-close return. Bucketed rather than
# left continuous so "how often does a high early score produce a strong
# winner?" is a countable question.
OUTCOME_BANDS: tuple[tuple[float, str], ...] = (
    (-30.0, "large_loser"),
    (-8.0, "loser"),
    (8.0, "flat"),
    (30.0, "moderate_winner"),
)
STRONG_WINNER = "strong_winner"


def label_outcome(return_pct: float | None) -> str | None:
    """Bucket a realised return. None stays None - an unmeasured horizon is
    not a flat one."""
    if return_pct is None:
        return None
    for upper, label in OUTCOME_BANDS:
        if return_pct < upper:
            return label
    return STRONG_WINNER


def schedule(
    db: Session,
    *,
    pipeline_event_id: int,
    token_address: str,
    symbol: str,
    score: float | None,
    price_at_signal: float,
    observed_at: dt.datetime | None = None,
    horizons: tuple[int, ...] = HORIZONS_MINUTES,
    early_score: float | None = None,
    late_entry_risk: float | None = None,
    momentum_class: str | None = None,
    early_features: dict | None = None,
    market_regime: str | None = None,
) -> int:
    """Create one pending row per horizon for a freshly scored candidate.

    Returns how many were created. Refuses a non-positive signal price -
    every return would divide by it, and a zero price is corrupt data
    rather than a free option.
    """
    if not price_at_signal or price_at_signal <= 0:
        logger.debug("not scheduling forward returns for %s: no usable signal price", symbol)
        return 0
    if not token_address:
        logger.debug("not scheduling forward returns for %s: no mint address", symbol)
        return 0

    observed_at = observed_at or dt.datetime.now(dt.timezone.utc)
    version = current_label()
    created = 0

    for minutes in horizons:
        db.add(
            models.ForwardReturn(
                pipeline_event_id=pipeline_event_id,
                token_address=token_address,
                symbol=symbol,
                observed_at=observed_at,
                score=score,
                price_at_signal=price_at_signal,
                horizon_minutes=minutes,
                due_at=observed_at + dt.timedelta(minutes=minutes),
                strategy_version=version,
                early_score=early_score,
                late_entry_risk=late_entry_risk,
                momentum_class=momentum_class,
                early_features=early_features,
                market_regime=market_regime,
            )
        )
        created += 1
    return created


def attach_regime(db: Session, pipeline_event_id: int, market_regime: str | None) -> int:
    """Complete the regime on rows scheduled before depth was known.

    Forward returns are scheduled inside the scoring block, where only the
    candle-derived axes exist. The liquidity axis arrives one gate later.
    Without this back-fill every stored regime would read
    ".../unknown" and the promotion gate's consistency bar - which needs
    depth most of all in a memecoin book - would have nothing to group on.
    """
    if not market_regime:
        return 0
    rows = (
        db.query(models.ForwardReturn)
        .filter(models.ForwardReturn.pipeline_event_id == pipeline_event_id)
        .all()
    )
    for row in rows:
        row.market_regime = market_regime
    return len(rows)


def attach_early(db: Session, pipeline_event_id: int, verdict) -> int:
    """Record the early verdict on rows already scheduled for this event.

    Forward returns are scheduled the moment a candidate is scored, before
    the early engine has run - deliberately, because a candidate the
    TECHNICAL gate rejects still has to be followed forward or the
    threshold can never be shown to be wrong. The consequence is that the
    early score is not known at scheduling time, and without this
    back-fill every ForwardReturn.early_score stays NULL and the early
    calibration reads an empty dataset while looking like it is working.

    A verdict with no early score (a security failure, which short-circuits
    before any feature is computed) writes nothing at all rather than
    writing zeros. "The early engine never got to look at this" and "the
    early engine scored it zero" are different facts.
    """
    if getattr(verdict, "early_score", None) is None:
        return 0

    rows = (
        db.query(models.ForwardReturn)
        .filter(models.ForwardReturn.pipeline_event_id == pipeline_event_id)
        .all()
    )
    features = verdict.features.as_dict() if getattr(verdict, "features", None) else None
    for row in rows:
        row.early_score = verdict.early_score
        row.late_entry_risk = verdict.late_risk
        row.momentum_class = verdict.momentum.label.value if verdict.momentum else None
        row.early_features = features
    return len(rows)


def _aware(moment: dt.datetime) -> dt.datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def due_rows(db: Session, *, now: dt.datetime | None = None, limit: int = 200) -> list[models.ForwardReturn]:
    """Rows whose horizon has elapsed and which have not been resolved.

    Ordered oldest first so a backlog drains in the order it accumulated
    rather than starving the earliest observations.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    return (
        db.query(models.ForwardReturn)
        .filter(
            models.ForwardReturn.filled_at.is_(None),
            models.ForwardReturn.due_at <= now,
        )
        .order_by(models.ForwardReturn.due_at.asc())
        .limit(limit)
        .all()
    )


async def _sample_envelope(row: models.ForwardReturn, price: float) -> None:
    """Widen a row's MFE/MAE envelope with the price observed right now.

    Sampled at the resolution interval rather than tick by tick, so both
    extremes are UNDERSTATED - the real drawdown was at least this bad and
    the real peak at least this good. Understating MAE is the safe
    direction; a tick-level envelope would need a trade-level feed the bot
    does not have, and pretending otherwise would make stop-loss analysis
    look more favourable than reality.
    """
    move = (price / row.price_at_signal - 1) * 100
    row.max_favorable_pct = move if row.max_favorable_pct is None else max(row.max_favorable_pct, move)
    row.max_adverse_pct = move if row.max_adverse_pct is None else min(row.max_adverse_pct, move)


async def resolve_due(
    db: Session, *, now: dt.datetime | None = None, limit: int = 200
) -> dict:
    """Resolve elapsed horizons, and sample the path of the pending ones.

    Both halves share ONE price lookup per distinct mint. The eight
    horizons for a token all reference the same current price, and the
    pending ones need that same price to widen their MFE/MAE envelope, so
    fetching per row would multiply the bot's largest source of API load by
    eight for no information gain.

    Sampling the pending rows is what makes MFE/MAE mean anything. Reading
    the price only when a horizon elapses would record the endpoint twice
    over and miss the entire path - and the path is what a stop would have
    hit.

    A row past GIVE_UP_AFTER_HOURS is closed out with a reason rather than
    retried forever. It stays in the table as explicitly unmeasurable,
    which is honest: dropping it would bias the dataset toward tokens that
    stayed alive.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    due = due_rows(db, now=now, limit=limit)

    # Pending rows for the SAME tokens - free to update, since their price
    # is already being fetched.
    mints = {row.token_address for row in due}
    pending: list[models.ForwardReturn] = []
    if mints:
        pending = (
            db.query(models.ForwardReturn)
            .filter(
                models.ForwardReturn.filled_at.is_(None),
                models.ForwardReturn.due_at > now,
                models.ForwardReturn.token_address.in_(mints),
            )
            .all()
        )

    summary = {
        "due": len(due), "resolved": 0, "abandoned": 0,
        "unavailable": 0, "envelope_sampled": 0,
    }
    if not due:
        return summary

    prices: dict[str, float | None] = {}

    async def price_for(mint: str) -> float | None:
        if mint not in prices:
            try:
                prices[mint] = await price_feed.get_price_usd(mint)
            except Exception:
                logger.warning("forward-return price lookup failed for %s", mint, exc_info=True)
                prices[mint] = None
        return prices[mint]

    for row in due:
        overdue_minutes = (now - _aware(row.due_at)).total_seconds() / 60
        tolerance = lateness_tolerance_minutes(row.horizon_minutes)

        if overdue_minutes > tolerance:
            # Past its own window. A price fetched now would describe a
            # different holding period than the one this row is labelled
            # with, and storing it would corrupt the horizon it feeds.
            row.filled_at = now
            row.measured_at = now
            row.actual_elapsed_minutes = (now - _aware(row.observed_at)).total_seconds() / 60
            row.failure_reason = (
                f"came due {overdue_minutes:.0f}min ago, past the {tolerance:.0f}min tolerance "
                f"for a {row.horizon_minutes}min horizon - sealed as unmeasurable rather than "
                f"filled with a price that would describe a "
                f"{row.actual_elapsed_minutes:.0f}min holding period"
            )
            summary["abandoned"] += 1
            continue

        price = await price_for(row.token_address)
        if not price or price <= 0:
            # Left pending deliberately: it may resolve on a later pass,
            # and guessing now would fabricate an outcome.
            summary["unavailable"] += 1
            continue

        await _sample_envelope(row, price)
        row.price_at_horizon = price
        row.return_pct = (price / row.price_at_signal - 1) * 100
        row.outcome = label_outcome(row.return_pct)
        row.filled_at = now
        # Recorded even on a clean resolve. The horizon says what was
        # intended; this says what was actually held, so an analysis can
        # tighten past MAX_LATENESS_FRACTION without re-deriving it.
        row.measured_at = now
        row.actual_elapsed_minutes = (now - _aware(row.observed_at)).total_seconds() / 60
        summary["resolved"] += 1

    for row in pending:
        price = prices.get(row.token_address)
        if price and price > 0:
            await _sample_envelope(row, price)
            summary["envelope_sampled"] += 1

    return summary


def coverage(db: Session) -> dict:
    """How complete the calibration dataset is.

    Reported next to every calibration table, because a table built on 20%
    coverage is describing whichever tokens happened to stay liquid - the
    exact bias the dataset exists to avoid.
    """
    total = db.query(models.ForwardReturn).count()
    if not total:
        return {"total": 0, "resolved": 0, "pending": 0, "unmeasurable": 0, "coverage_pct": 0.0}

    resolved = db.query(models.ForwardReturn).filter(
        models.ForwardReturn.return_pct.isnot(None)
    ).count()
    unmeasurable = db.query(models.ForwardReturn).filter(
        models.ForwardReturn.filled_at.isnot(None),
        models.ForwardReturn.return_pct.is_(None),
    ).count()

    return {
        "total": total,
        "resolved": resolved,
        "pending": total - resolved - unmeasurable,
        "unmeasurable": unmeasurable,
        "coverage_pct": round(resolved / total * 100, 1),
    }


def enabled() -> bool:
    return bool(getattr(settings, "FORWARD_RETURNS_ENABLED", True))
