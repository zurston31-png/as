"""The WATCH state machine.

    DISCOVERED -> WATCH -> CONFIRMED -> PAPER_BUY -> EXIT
    DISCOVERED -> WATCH -> FAILED -> SKIP

WATCH exists because "promising" and "ready" are different facts about a
token, and a bot without a third state is forced to collapse them: buy
every promising candidate immediately, which is chasing, or discard it,
which misses the move entirely. Keeping it under observation is what lets
the bot enter on confirmation that arrives BEFORE the token becomes
overextended - a race it can only run if it is still looking.

Every re-evaluation appends to `score_history`. That is the point of
storing anything at all: it makes it possible to ask later whether an
IMPROVING score predicts better outcomes than a high static one, which is
a question no single snapshot per token can answer.

Entries leave the list in one of three ways, and each is recorded rather
than deleted:

    CONFIRMED    conditions came together in time
    FAILED       they deteriorated, with a failure category attached
    expired      the token sat on the list past WATCHLIST_MAX_AGE_HOURS
                 without resolving either way

Recording failures is not bookkeeping. The false-positive analysis in
app/analysis/early_calibration.py is built entirely from these rows, and a
watchlist that quietly dropped its disappointments would make the engine
permanently unfalsifiable.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.early.engine import Decision, EarlyVerdict
from app.strategy.version import current_label

logger = logging.getLogger(__name__)

# States
DISCOVERED = "DISCOVERED"
WATCH = "WATCH"
CONFIRMED = "CONFIRMED"
PAPER_BUY = "PAPER_BUY"
FAILED = "FAILED"
SKIPPED = "SKIPPED"
EXPIRED = "EXPIRED"

TERMINAL_STATES = (PAPER_BUY, FAILED, SKIPPED, EXPIRED)

# How far the early score must fall from its best before the entry is
# treated as having deteriorated rather than merely wobbled. Scores move a
# few points on noise; a sustained give-back is a different event.
DETERIORATION_DROP = 12.0

# Failure taxonomy. Every FAILED entry gets exactly one of these, chosen by
# what the features said at the moment it failed - so "why do high early
# scores fail?" is answerable by grouping rather than by reading logs.
FAILURE_CATEGORIES = {
    "volume_disappeared": "volume fell away before confirmation arrived",
    "buy_pressure_reversed": "buy pressure flipped to selling",
    "liquidity_fell": "pool depth was withdrawn",
    "security_deteriorated": "the security verdict changed for the worse",
    "failed_breakout": "price approached the range high and was rejected",
    "became_late": "the token ran away before technical confirmation arrived",
    "score_decayed": "the early score faded without a single identifiable cause",
    "expired": "sat on the watchlist without resolving either way",
}


def _aware(moment: dt.datetime | None) -> dt.datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def get(db: Session, token_address: str) -> models.WatchlistEntry | None:
    return db.query(models.WatchlistEntry).filter_by(token_address=token_address).first()


def active(db: Session) -> list[models.WatchlistEntry]:
    """Entries still being watched, most promising first."""
    return (
        db.query(models.WatchlistEntry)
        .filter(models.WatchlistEntry.state.in_([WATCH, CONFIRMED]))
        .order_by(models.WatchlistEntry.early_score.desc())
        .all()
    )


def _append_history(entry: models.WatchlistEntry, verdict: EarlyVerdict, now: dt.datetime) -> None:
    """Append one point to the score history.

    Reassigns the list rather than mutating it in place: SQLAlchemy's JSON
    column does not track in-place mutation, so `history.append(...)` alone
    would be silently discarded on commit and the whole score-history
    feature would store nothing.
    """
    point = {
        "at": now.isoformat(),
        "early": round(verdict.early_score, 2) if verdict.early_score is not None else None,
        "technical": verdict.technical_score,
        "late_risk": round(verdict.late_risk, 2) if verdict.late_risk is not None else None,
        "stage": verdict.stage.value if verdict.stage else None,
        "momentum": verdict.momentum.label.value if verdict.momentum else None,
        "decision": verdict.decision.value,
    }
    history = list(entry.score_history or [])
    history.append(point)
    # Cap the stored history. Long-lived entries would otherwise grow a JSON
    # blob without bound, and the early points stop being interesting once
    # the trend they describe is established.
    entry.score_history = history[-120:]


def record(
    db: Session,
    *,
    token_address: str,
    symbol: str,
    chain: str,
    verdict: EarlyVerdict,
    price: float | None = None,
    now: dt.datetime | None = None,
) -> models.WatchlistEntry | None:
    """Create or update the watchlist entry for one evaluated candidate.

    Returns the entry, or None when the candidate is not worth tracking and
    has no existing row (a SKIP on a token nobody was watching is not an
    event, it is the normal case for almost every token).
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    entry = get(db, token_address)

    if entry is None:
        if verdict.decision is Decision.SKIP:
            return None
        if db.query(models.WatchlistEntry).filter(
            models.WatchlistEntry.state.in_([WATCH, CONFIRMED])
        ).count() >= settings.WATCHLIST_MAX_SIZE:
            logger.info(
                "watchlist is full (%d) - not adding %s", settings.WATCHLIST_MAX_SIZE, symbol
            )
            return None
        entry = models.WatchlistEntry(
            token_address=token_address, symbol=symbol, chain=chain,
            state=WATCH, first_seen_at=now,
            price_at_first_signal=price, first_signal_at=now,
            score_history=[], features={},
        )
        db.add(entry)
        # Flush immediately. The session runs with autoflush=False, so
        # without this a second evaluation of the same token in the same
        # session would not find this row and would insert a duplicate,
        # failing the UNIQUE constraint at commit and taking the whole
        # transaction - including unrelated work - down with it.
        db.flush()

    entry.symbol = symbol
    entry.last_evaluated_at = now
    entry.evaluations = (entry.evaluations or 0) + 1
    entry.early_score = verdict.early_score
    entry.technical_score = verdict.technical_score
    entry.security_score = verdict.security_score
    entry.market_quality_score = verdict.market_quality_score
    entry.late_entry_risk = verdict.late_risk
    entry.stage = verdict.stage.value if verdict.stage else None
    entry.momentum_class = verdict.momentum.label.value if verdict.momentum else None
    entry.reason = verdict.reason
    entry.strategy_version = current_label()
    if verdict.features is not None:
        entry.features = verdict.features.as_dict()

    if verdict.early_score is not None:
        entry.best_early_score = max(entry.best_early_score or 0.0, verdict.early_score)

    _append_history(entry, verdict, now)

    # --- state transition ------------------------------------------------
    if entry.state in TERMINAL_STATES:
        return entry     # already resolved; history still records the tick

    if verdict.decision is Decision.PAPER_BUY:
        entry.state = CONFIRMED
    elif verdict.decision is Decision.WATCH:
        entry.state = WATCH
    else:
        category = classify_failure(entry, verdict)
        entry.state = FAILED
        entry.failure_category = category
        entry.reason = f"{FAILURE_CATEGORIES.get(category, category)}: {verdict.reason}"

    return entry


def classify_failure(entry: models.WatchlistEntry, verdict: EarlyVerdict) -> str:
    """Why this candidate stopped being interesting.

    Ordered so the most specific cause wins. Everything that reaches the
    end is `score_decayed`, which is deliberately the residual bucket - if
    it dominates the false-positive table, the taxonomy is missing a
    category rather than the answer being "it just faded".
    """
    features = verdict.features

    if verdict.security_score is not None and verdict.reason.startswith("security failure"):
        return "security_deteriorated"

    if verdict.stage is not None and not verdict.stage.enterable:
        return "became_late"

    if features is not None:
        liquidity = features.value("liquidity_growth")
        if liquidity is not None and liquidity < 0.9:
            return "liquidity_fell"

        pressure_change = features.value("buy_pressure_change")
        pressure = features.value("buy_pressure")
        if (pressure_change is not None and pressure_change < -0.05) or (
            pressure is not None and pressure < 0.42
        ):
            return "buy_pressure_reversed"

        volume = features.value("volume_accel_short")
        if volume is not None and volume < 0.75:
            return "volume_disappeared"

        breakout = features.value("breakout_proximity")
        best = entry.best_early_score or 0.0
        if breakout is not None and -12 < breakout < 0 and best >= 60:
            return "failed_breakout"

    return "score_decayed"


def expire_stale(db: Session, *, now: dt.datetime | None = None) -> int:
    """Retire entries that sat on the list without resolving.

    Marked EXPIRED rather than deleted. An entry that was promising and
    then simply went nowhere is a false positive, and it belongs in the
    false-positive analysis - deleting it would quietly improve the
    engine's apparent hit rate.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=settings.WATCHLIST_MAX_AGE_HOURS)
    expired = 0

    for entry in active(db):
        first_seen = _aware(entry.first_seen_at)
        if first_seen and first_seen < cutoff:
            entry.state = EXPIRED
            entry.failure_category = "expired"
            entry.reason = (
                f"{FAILURE_CATEGORIES['expired']} after "
                f"{settings.WATCHLIST_MAX_AGE_HOURS:.0f}h "
                f"(best early score {entry.best_early_score or 0:.0f})"
            )
            expired += 1

    return expired


def mark_traded(db: Session, token_address: str) -> models.WatchlistEntry | None:
    """Record that a watched token actually opened a position."""
    entry = get(db, token_address)
    if entry is None:
        return None
    entry.state = PAPER_BUY
    return entry


def deteriorating(entry: models.WatchlistEntry) -> bool:
    """Has the score given back a meaningful amount of its best?

    Scores move a few points on noise. A sustained give-back is a
    different event, and the threshold is what separates the two.
    """
    if entry.best_early_score is None or entry.early_score is None:
        return False
    return (entry.best_early_score - entry.early_score) >= DETERIORATION_DROP


def prune_observations(db: Session, *, now: dt.datetime | None = None) -> int:
    """Drop stored snapshots past their retention window.

    Their entire value is recency - a transaction count from two days ago
    tells you nothing about whether flow is accelerating now.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=settings.OBSERVATION_RETENTION_HOURS)
    rows = (
        db.query(models.TokenObservation)
        .filter(models.TokenObservation.observed_at < cutoff)
        .all()
    )
    for row in rows:
        db.delete(row)
    return len(rows)


def store_observation(db: Session, symbol: str, token_address: str, market) -> models.TokenObservation | None:
    """Persist one market snapshot so flow can be differenced later."""
    if market is None:
        return None
    observation = models.TokenObservation(
        token_address=token_address,
        symbol=symbol,
        observed_at=market.observed_at or dt.datetime.now(dt.timezone.utc),
        price_usd=market.price_usd,
        liquidity_usd=market.liquidity_usd,
        market_cap_usd=market.market_cap_usd,
        volume_5m_usd=market.volume_5m_usd,
        volume_1h_usd=market.volume_1h_usd,
        volume_24h_usd=market.volume_24h_usd,
        buys_1h=market.buys_1h,
        sells_1h=market.sells_1h,
        buys_24h=market.buys_24h,
        sells_24h=market.sells_24h,
        price_change_5m_pct=market.price_change_5m_pct,
        price_change_1h_pct=market.price_change_1h_pct,
    )
    db.add(observation)
    return observation


def store_price_point(
    db: Session, symbol: str, token_address: str, price: float | None
) -> models.TokenObservation | None:
    """Persist one price reading with no surrounding snapshot.

    The position monitor already fetches a price for every open position
    on every pass and then throws it away. Keeping it costs nothing and is
    the only price history the bot has for tokens it HOLDS - the early
    engine only observes tokens it is scanning, and a position stops being
    scanned the moment it is opened. Correlation risk needs the held ones.

    Reuses TokenObservation rather than adding a table, so the existing
    OBSERVATION_RETENTION_HOURS pruning applies unchanged. Every field but
    the price is left NULL, which is accurate: nothing else was measured.
    """
    if price is None or price <= 0:
        return None
    observation = models.TokenObservation(
        token_address=token_address,
        symbol=symbol,
        observed_at=dt.datetime.now(dt.timezone.utc),
        price_usd=price,
    )
    db.add(observation)
    return observation


def price_series(
    db: Session, token_address: str, *, limit: int = 500
) -> list[float]:
    """Stored prices for one token, oldest first.

    Returns the raw prices; the caller differences them into returns.
    Correlating price LEVELS would find that two tokens both drifting
    upward are correlated almost by definition, which is not the question
    - what matters for risk is whether they fall together.
    """
    rows = (
        db.query(models.TokenObservation.price_usd, models.TokenObservation.observed_at)
        .filter(
            models.TokenObservation.token_address == token_address,
            models.TokenObservation.price_usd.isnot(None),
        )
        .order_by(models.TokenObservation.observed_at.desc())
        .limit(limit)
        .all()
    )
    return [price for price, _ in reversed(rows)]


def recent_observations(db: Session, token_address: str, *, limit: int = 12) -> list:
    """The last few stored snapshots for a token, oldest first."""
    rows = (
        db.query(models.TokenObservation)
        .filter(models.TokenObservation.token_address == token_address)
        .order_by(models.TokenObservation.observed_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))
