"""Closing out hypothetical positions, so the shadow system produces evidence.

Until this ran, the shadow tables recorded decisions and nothing else:
every hypothetical entry stayed open forever, `return_pct` stayed NULL,
and the paired comparison had two arms of zeros to compare. Recording what
a strategy WOULD have done is only half a measurement - the other half is
what happened next.

HOW AN OUTCOME IS PRODUCED

Candles after the entry, walked bar by bar through one shared exit rule
(app/shadow/exit_policy.py). Not a live price poll: a poll samples
whichever moments the loop happened to wake up for, and a stop that was
breached between two samples never registers. Bar highs and lows contain
the breach whether anyone was watching or not.

FOUR PROPERTIES THIS HAS TO HAVE

  idempotent      Every field is recomputed from the candles, never
                  accumulated. Running the resolver twice writes the same
                  numbers; running it a hundred times still does.

  restart-safe    All state lives in the database. There is no in-memory
                  cursor to lose, and an interrupted pass resumes by
                  simply looking again at what is still open.

  no look-ahead   A bar is only visible once it has CLOSED as of the
                  resolution instant. A bar still forming knows the future
                  relative to the moment being simulated.

  inert           It reads and writes two shadow tables and nothing else.
                  No live position, no cash ledger, no cooldown, no risk
                  event, no execution client. A hypothetical outcome must
                  not be able to move a single number the paper account
                  reports.

WHAT IT REFUSES TO DO

  * fill a missing price with the last known one
  * treat a token that stopped trading as a 0% return
  * resolve a horizon that has not elapsed
  * quietly reuse a stale quote from hours before the horizon

Each of those turns a dead token into a flat one, and a flat token into
evidence. Anything unmeasurable is stored as unmeasurable, with a reason.
"""
from __future__ import annotations

import datetime as dt
import logging
import math

from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.data.candles import Timeframe
from app.shadow.exit_policy import ExitPolicy, walk

logger = logging.getLogger(__name__)

# Never ask a provider for more than it will serve in one call.
MAX_CANDLES = 1000

# How far past a horizon a quote may sit and still be called that
# horizon's price. Two bars of tolerance: a single missing bar is a feed
# hiccup, a longer gap means the token went quiet and the "price" would be
# a stale number wearing a fresh timestamp.
STALENESS_BARS = 2


def horizons() -> tuple[int, ...]:
    """Fixed horizons, from configuration, always sorted and de-duplicated."""
    raw = getattr(settings, "SHADOW_HORIZONS_MINUTES", "") or ""
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            minutes = int(part)
        except ValueError:
            logger.error("ignoring unparseable shadow horizon %r", part)
            continue
        if minutes > 0:
            out.add(minutes)
    return tuple(sorted(out))


def timeframe() -> Timeframe:
    try:
        return Timeframe(settings.SHADOW_RESOLUTION_TIMEFRAME)
    except ValueError:
        logger.error(
            "SHADOW_RESOLUTION_TIMEFRAME=%r is not a known timeframe - falling back to 5m",
            settings.SHADOW_RESOLUTION_TIMEFRAME,
        )
        return Timeframe.M5


def _aware(moment: dt.datetime | None) -> dt.datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def _round_trip_cost(row: models.ShadowPosition) -> float:
    """The exit-side cost, as a fraction, using the entry's own assumptions.

    Deliberately the SAME fee and slippage the fill model charged on the
    way in, rather than a fresh simulation: the entry price already embeds
    one leg, and a second leg priced off a different draw would make the
    round trip depend on when the resolver happened to run.

    A row with no recorded cost gets zero, which flatters it. That case
    only arises for a position opened before these fields existed, and the
    alternative - inventing a plausible cost - would be fabrication.
    """
    return (row.fees_pct or 0.0) + (row.slippage_pct or 0.0)


def open_positions(db: Session, *, limit: int) -> list[models.ShadowPosition]:
    """Hypothetical positions still waiting for an outcome, oldest first."""
    return (
        db.query(models.ShadowPosition)
        .filter(models.ShadowPosition.closed_at.is_(None))
        .order_by(models.ShadowPosition.opened_at.asc())
        .limit(limit)
        .all()
    )


def positions_awaiting_horizons(
    db: Session, *, now: dt.datetime, limit: int
) -> list[models.ShadowPosition]:
    """Positions - open or closed - with a horizon that has come due.

    Closed ones are included on purpose. A horizon return is a fact about
    the token, not about the trade, so it stays worth recording after the
    stop took the position out. That is precisely the comparison that
    shows whether an exit rule is cutting winners.
    """
    windows = horizons()
    if not windows:
        return []
    earliest_due = now - dt.timedelta(minutes=max(windows))
    give_up = dt.timedelta(hours=settings.SHADOW_UNMEASURABLE_AFTER_HOURS)
    return (
        db.query(models.ShadowPosition)
        .filter(
            models.ShadowPosition.opened_at <= now - dt.timedelta(minutes=min(windows)),
            models.ShadowPosition.opened_at >= earliest_due - give_up,
        )
        .order_by(models.ShadowPosition.opened_at.asc())
        .limit(limit)
        .all()
    )


def positions_with_sealed_horizons(
    db: Session, *, now: dt.datetime, limit: int
) -> list[models.ShadowPosition]:
    """Positions whose horizons are permanently past and still unrecorded.

    The hole this closes: `positions_awaiting_horizons` stops looking at a
    position once its longest horizon is far enough in the past, so a
    horizon that never found a quote simply stayed absent - and absence is
    ambiguous. "Never came due", "came due and could not be measured" and
    "nobody looked" are three different facts that all render as no row.

    Selection is by a NOT-ALL-RECORDED count rather than a time window, so
    a position drops out of this query the moment it is sealed and the
    scan does not grow without bound as history accumulates.
    """
    windows = horizons()
    if not windows:
        return []
    cutoff = (
        now
        - dt.timedelta(minutes=max(windows))
        - dt.timedelta(hours=settings.SHADOW_UNMEASURABLE_AFTER_HOURS)
    )
    recorded = (
        db.query(
            models.ShadowHorizonReturn.position_id.label("position_id"),
            func.count().label("n"),
        )
        .group_by(models.ShadowHorizonReturn.position_id)
        .subquery()
    )
    return (
        db.query(models.ShadowPosition)
        .outerjoin(recorded, recorded.c.position_id == models.ShadowPosition.id)
        .filter(
            models.ShadowPosition.opened_at < cutoff,
            or_(recorded.c.n.is_(None), recorded.c.n < len(windows)),
        )
        .order_by(models.ShadowPosition.opened_at.asc())
        .limit(limit)
        .all()
    )


def _chains(db: Session, tokens: set[str]) -> dict[str, str]:
    """Which chain each token trades on, from the decisions that recorded it.

    Looked up rather than stored on the position: the decision row already
    holds it, and a second copy is a second thing to keep in step.
    """
    if not tokens:
        return {}
    rows = (
        db.query(models.ShadowDecision.token_address, models.ShadowDecision.chain)
        .filter(models.ShadowDecision.token_address.in_(tokens))
        .distinct()
        .all()
    )
    return {token: chain for token, chain in rows if chain}


async def _load_candles(fetch, chain, token, symbol, tf, since, now):
    """One candle request per token, sized to cover every open position."""
    span_seconds = max((now - since).total_seconds(), 0)
    needed = int(math.ceil(span_seconds / tf.seconds)) + 2
    try:
        return await fetch(chain, token, symbol, tf, min(needed, MAX_CANDLES))
    except Exception:
        logger.warning("shadow candle fetch failed for %s", token, exc_info=True)
        return None


def _visible(series, *, opened_at: dt.datetime, now: dt.datetime, tf: Timeframe) -> list:
    """Bars that began at or after entry and had closed by `now`.

    The bar containing the entry is excluded, not truncated: its low may
    have happened before the position existed, and letting it into the
    envelope would report a drawdown the trade never took.
    """
    interval = dt.timedelta(seconds=tf.seconds)
    return [
        candle for candle in series
        if candle.timestamp >= opened_at and candle.timestamp + interval <= now
    ]


def _close_out(
    row: models.ShadowPosition,
    *,
    result,
    policy: ExitPolicy,
    now: dt.datetime,
) -> None:
    """Write a finished outcome. Every field is derived, none accumulated."""
    cost = _round_trip_cost(row)
    gross = (result.exit_price / row.entry_price - 1) * 100

    row.exit_price = result.exit_price
    row.closed_at = result.exit_at
    row.exit_reason = result.exit_reason
    row.gross_return_pct = gross
    row.return_pct = gross - cost * 100
    row.max_favorable_pct = result.max_favorable_pct
    row.max_adverse_pct = result.max_adverse_pct
    row.hold_minutes = max(
        (_aware(result.exit_at) - _aware(row.opened_at)).total_seconds() / 60, 0.0
    )
    row.exit_policy = policy.fingerprint()
    row.bars_observed = result.bars
    row.resolved_at = now


def _abandon(row: models.ShadowPosition, *, now: dt.datetime, reason: str) -> None:
    """Stop chasing an outcome that is not coming.

    `closed_at` is set so the row leaves the working set, but `return_pct`
    stays NULL: this is a position that was entered and whose result is
    unknown, which is a different fact from a position that broke even.
    Every consumer already treats a NULL return as unresolved, so an
    abandoned row can never be counted as evidence.
    """
    row.closed_at = now
    row.resolved_at = now
    row.exit_reason = reason


def _record_horizon(
    db: Session,
    row: models.ShadowPosition,
    *,
    horizon_minutes: int,
    due_at: dt.datetime,
    price: float | None,
    failure_reason: str | None,
) -> bool:
    """Insert one horizon observation, or leave the existing one alone.

    The unique constraint deduplicates rather than a prior read, and a
    SAVEPOINT contains the collision: the caller may hold uncommitted work
    in this session, and a session-wide rollback would destroy it.
    """
    gross = net = None
    if price is not None and price > 0 and row.entry_price:
        gross = (price / row.entry_price - 1) * 100
        net = gross - _round_trip_cost(row) * 100

    record = models.ShadowHorizonReturn(
        position_id=row.id,
        opportunity_id=row.opportunity_id,
        strategy_id=row.strategy_id,
        token_address=row.token_address,
        horizon_minutes=horizon_minutes,
        due_at=due_at,
        measured_at=dt.datetime.now(dt.timezone.utc) if price or failure_reason else None,
        price_at_horizon=price,
        gross_return_pct=gross,
        return_pct=net,
        failure_reason=failure_reason,
    )
    try:
        with db.begin_nested():
            db.add(record)
            db.flush()
    except IntegrityError:
        return False
    return True


def _recorded_horizons(db: Session, position_ids: list[int]) -> set[tuple[int, int]]:
    if not position_ids:
        return set()
    rows = (
        db.query(
            models.ShadowHorizonReturn.position_id,
            models.ShadowHorizonReturn.horizon_minutes,
        )
        .filter(models.ShadowHorizonReturn.position_id.in_(position_ids))
        .all()
    )
    return {(pid, minutes) for pid, minutes in rows}


def _price_at(series, instant: dt.datetime, tf: Timeframe) -> float | None:
    """The close of the last bar that had finished by `instant`.

    Refuses a bar that closed more than `STALENESS_BARS` intervals early -
    that is not this horizon's price, it is the price from before the feed
    went quiet, and passing it off as the former is how a dead token
    becomes a flat one.
    """
    interval = dt.timedelta(seconds=tf.seconds)
    best = None
    for candle in series:
        closed = candle.timestamp + interval
        if closed <= instant and (best is None or closed > best[0]):
            best = (closed, candle.close)
    if best is None:
        return None
    if instant - best[0] > interval * STALENESS_BARS:
        return None
    return best[1]


async def resolve_once(
    db: Session,
    *,
    now: dt.datetime | None = None,
    limit: int | None = None,
    fetch=None,
) -> dict:
    """One resolution pass. Safe to call again immediately.

    `fetch` is injectable so tests can drive the walk with fixed candles
    instead of the network - the exit logic is the part worth testing, and
    a test that needs a live feed does not get written.
    """
    summary = {
        "considered": 0, "closed": 0, "still_open": 0, "abandoned": 0,
        "no_candles": 0, "horizons_recorded": 0, "horizons_unmeasurable": 0,
        "horizons_sealed": 0,
    }
    if not getattr(settings, "SHADOW_RESOLVER_ENABLED", True):
        return summary

    now = _aware(now) or dt.datetime.now(dt.timezone.utc)
    limit = limit or settings.SHADOW_RESOLVE_BATCH
    tf = timeframe()
    policy = ExitPolicy.from_settings()

    still_open = open_positions(db, limit=limit)
    pending_horizons = positions_awaiting_horizons(db, now=now, limit=limit)

    work: dict[int, models.ShadowPosition] = {}
    for row in [*still_open, *pending_horizons]:
        if row.id is not None:
            work[row.id] = row
    if not work:
        # Still seal: a backlog of permanently-unmeasurable horizons can
        # outlive every position the measuring pass still cares about.
        summary["horizons_sealed"] = _seal_horizons(db, now=now, limit=limit)
        return summary
    summary["considered"] = len(work)

    rows = sorted(work.values(), key=lambda r: _aware(r.opened_at))
    chains = _chains(db, {r.token_address for r in rows})
    already = _recorded_horizons(db, [r.id for r in rows])
    windows = horizons()
    give_up = dt.timedelta(hours=settings.SHADOW_UNMEASURABLE_AFTER_HOURS)

    if fetch is None:
        from app.data.live_provider import fetch_candles as fetch

    by_token: dict[str, list[models.ShadowPosition]] = {}
    for row in rows:
        by_token.setdefault(row.token_address, []).append(row)

    for token, group in by_token.items():
        chain = chains.get(token) or settings.CHAIN
        earliest = _aware(min(r.opened_at for r in group))
        series = await _load_candles(
            fetch, chain, token, group[0].symbol, tf, earliest, now
        )
        candles = list(series) if series else []

        for row in group:
            opened_at = _aware(row.opened_at)
            visible = _visible(candles, opened_at=opened_at, now=now, tf=tf)

            if row.closed_at is None:
                age = now - opened_at
                if not visible:
                    summary["no_candles"] += 1
                    if age > dt.timedelta(hours=policy.max_hold_hours) + give_up:
                        _abandon(
                            row, now=now,
                            reason=(
                                f"no candles covering the {age.total_seconds() / 3600:.1f}h since "
                                "entry - outcome recorded as unmeasurable rather than "
                                "assumed flat"
                            ),
                        )
                        summary["abandoned"] += 1
                    else:
                        summary["still_open"] += 1
                else:
                    result = walk(
                        policy,
                        entry_price=row.entry_price,
                        opened_at=opened_at,
                        candles=visible,
                        timeframe=tf,
                    )
                    if result.closed:
                        _close_out(row, result=result, policy=policy, now=now)
                        summary["closed"] += 1
                    else:
                        # Still running. The envelope is refreshed anyway,
                        # so an open position always reports the drawdown
                        # it has actually survived so far.
                        row.max_favorable_pct = result.max_favorable_pct
                        row.max_adverse_pct = result.max_adverse_pct
                        row.bars_observed = result.bars
                        row.exit_policy = policy.fingerprint()
                        summary["still_open"] += 1

            for minutes in windows:
                if (row.id, minutes) in already:
                    continue
                due_at = opened_at + dt.timedelta(minutes=minutes)
                if due_at > now:
                    continue
                price = _price_at(candles, due_at, tf)
                if price is None:
                    if now - due_at <= give_up:
                        # Might still arrive. Not written, so the retry
                        # stays possible and nothing is guessed meanwhile.
                        continue
                    _record_horizon(
                        db, row, horizon_minutes=minutes, due_at=due_at, price=None,
                        failure_reason=(
                            f"no quote within {STALENESS_BARS} bars of the {minutes}m mark, "
                            f"and it came due {(now - due_at).total_seconds() / 3600:.1f}h ago"
                        ),
                    )
                    summary["horizons_unmeasurable"] += 1
                    continue
                if _record_horizon(
                    db, row, horizon_minutes=minutes, due_at=due_at,
                    price=price, failure_reason=None,
                ):
                    summary["horizons_recorded"] += 1

    summary["horizons_sealed"] = _seal_horizons(db, now=now, limit=limit)
    return summary


def _seal_horizons(db: Session, *, now: dt.datetime, limit: int) -> int:
    """Write an explicit unmeasurable row for every horizon nobody will fill.

    Runs after the measuring pass, over positions the measuring pass has
    permanently stopped looking at. No candle fetch: by construction these
    horizons are past the give-up window, so the point is not to try once
    more, it is to replace an ambiguous absence with a stated fact.

    Without this, a gap in the table could mean the horizon never came
    due, or that it came due and no quote existed, or that the resolver
    was never running. Reading a dataset means knowing which.
    """
    sealed = 0
    windows = horizons()
    rows = positions_with_sealed_horizons(db, now=now, limit=limit)
    already = _recorded_horizons(db, [r.id for r in rows])

    for row in rows:
        opened_at = _aware(row.opened_at)
        for minutes in windows:
            if (row.id, minutes) in already:
                continue
            due_at = opened_at + dt.timedelta(minutes=minutes)
            if _record_horizon(
                db, row, horizon_minutes=minutes, due_at=due_at, price=None,
                failure_reason=(
                    f"never measured: the {minutes}m mark passed "
                    f"{(now - due_at).total_seconds() / 3600:.1f}h ago and is now beyond the "
                    f"{settings.SHADOW_UNMEASURABLE_AFTER_HOURS:g}h give-up window - recorded "
                    "as unmeasurable so its absence is not mistaken for a flat outcome"
                ),
            ):
                sealed += 1
    return sealed


def coverage(db: Session) -> dict:
    """How much of the shadow dataset has an outcome attached.

    Printed next to every comparison. A comparison drawn from 20% resolved
    positions is describing whichever tokens kept trading, which is the
    bias the whole apparatus exists to avoid.
    """
    total = db.query(models.ShadowPosition).count()
    if not total:
        return {"positions": 0, "resolved": 0, "open": 0, "unmeasurable": 0, "resolved_pct": 0.0}

    resolved = db.query(models.ShadowPosition).filter(
        models.ShadowPosition.return_pct.isnot(None)
    ).count()
    unmeasurable = db.query(models.ShadowPosition).filter(
        models.ShadowPosition.closed_at.isnot(None),
        models.ShadowPosition.return_pct.is_(None),
    ).count()
    return {
        "positions": total,
        "resolved": resolved,
        "open": total - resolved - unmeasurable,
        "unmeasurable": unmeasurable,
        "resolved_pct": round(resolved / total * 100, 1),
    }
