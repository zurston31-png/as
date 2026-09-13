"""The automatic token scanner loop.

Every SCANNER_INTERVAL_SECONDS:

    discover  (DexScreener + optional Birdeye new listings)
      -> dedupe   (skip anything evaluated within SCANNER_RECHECK_MINUTES)
      -> prescreen (liquidity / volume / age / txns / sell pressure, free)
      -> hand to trading_service.handle_discovered_token, which runs the
         SAME risk gate -> signal score -> rug check -> sizing -> execution
         path a TradingView alert takes
      -> record the outcome on ScannedToken for the audit trail

Cost ordering is deliberate and load-bearing. Discovery can surface
hundreds of brand-new mints a minute; the prescreen rejects most of them
using data that already arrived in the listing payload, so the expensive
stages (several rug-check lookups, a pool resolution plus candle fetch)
only ever run on the few that could plausibly be traded. Reversing that
order would work identically and hammer four APIs to do it.

Positions opened here are monitored by the existing position monitor
(app/monitor/position_monitor.py) exactly like any other position - stop
loss, take profit, trailing stop, and every other Stage 4 exit apply
unchanged, because the scanner never created a special kind of position in
the first place.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import random

from sqlalchemy.orm import Session

from app import models, pipeline
from app.config import settings
from app.database import SessionLocal
from app.monitor.supervisor import run_supervised
from app.scanner.discovery import DiscoveredToken, discover_tokens
from app.analysis import forward_returns
from app.scanner.filters import prescreen
from app.services.trading_service import handle_discovered_token

logger = logging.getLogger(__name__)

_stop_event = asyncio.Event()

# ScannedToken.last_stage values, in pipeline order.
STAGE_PRESCREEN = "prescreen"

# Horizons followed for sampled prescreen rejects.
#
# This was (60, 240, 1440) on the reasoning that the counterfactual reads
# one horizon at a time. That was true of the counterfactual and wrong for
# the question underneath it: whether the pre-screen throws away winners is
# a question about the shape of the move, and three points cannot show
# whether a rejected token spiked at fifteen minutes and gave it all back
# by four hours. Matching the main set makes the two comparable.
#
# 1-minute is deliberately absent despite being asked for. The resolver
# polls every FORWARD_RETURN_RESOLVE_INTERVAL_SECONDS and a horizon is only
# honoured within a quarter of its own length
# (app/analysis/forward_returns.MAX_LATENESS_FRACTION), so a 1-minute row
# needs a ~15-second poll to ever resolve. Scheduling one at the default
# 300s would manufacture rows that are sealed unmeasurable by construction.
PRESCREEN_HORIZONS: tuple[int, ...] = forward_returns.HORIZONS_MINUTES
STAGE_EVALUATED = "evaluated"
STAGE_TRADED = "traded"


def scanner_blocked_reason() -> str | None:
    """Why the scanner must not run right now, or None if it may."""
    if not settings.SCANNER_ENABLED:
        return "SCANNER_ENABLED=false"
    if settings.LIVE_TRADING and not settings.SCANNER_ALLOW_LIVE_TRADING:
        return (
            "LIVE_TRADING=true but SCANNER_ALLOW_LIVE_TRADING=false - auto-discovering brand-new "
            "tokens and buying them unattended with real money is a deliberate extra step; set "
            "SCANNER_ALLOW_LIVE_TRADING=true only if that is genuinely what you want"
        )
    return None


def _track_prescreen_reject(
    db: Session, token: DiscoveredToken, *, rng: random.Random,
    event: models.PipelineEvent | None,
) -> bool:
    """Follow a SAMPLE of prescreen rejects forward, so the gate has a cost.

    The prescreen is where most candidates die, and until now none of them
    were followed - so app/analysis/counterfactual.py could say nothing at
    all about the filter that rejects the most. "No data" is not the same
    finding as "rejects nothing worth having", and the two were
    indistinguishable.

    SAMPLED, because tracking every reject would multiply the bot's largest
    source of API load by the rejection rate, which is most of the flow.
    A random draw keeps the sample unbiased: the mean of a random tenth is
    an unbiased estimate of the mean of the whole, while the first tenth a
    provider happens to list is not.

    NO SCORE is recorded, because none was computed - these tokens never
    reached the scorer. That is deliberate: every calibration query filters
    on a non-null score, so these rows serve the counterfactual without
    entering a table that would then be describing a different population.
    """
    if not settings.SCANNER_TRACK_PRESCREEN_REJECTS:
        return False
    if not forward_returns.enabled():
        return False
    price = token.price_usd or 0.0
    if price <= 0 or not token.token_address:
        return False
    if rng.random() >= settings.SCANNER_PRESCREEN_TRACKING_RATE:
        return False

    # Anchor the forward returns to the prescreen event the caller ALREADY
    # recorded for this token, rather than writing a second one.
    #
    # This used to create its own PRESCREEN row, which meant a sampled
    # reject produced two prescreen events for one evaluation. The funnel
    # counts events, so PRESCREEN read higher than DISCOVERED - 208 against
    # 197 in the field - and the pass rate was computed against an inflated
    # denominator. The sampling is meant to observe the population, not to
    # change the count of it.
    if event is None:
        return False
    db.flush()
    created = forward_returns.schedule(
        db,
        pipeline_event_id=event.id,
        token_address=token.token_address,
        symbol=token.symbol,
        score=None,
        price_at_signal=price,
        horizons=PRESCREEN_HORIZONS,
    )
    return bool(created)


def _record(
    db: Session, token: DiscoveredToken, *, stage: str, reason: str, traded: bool = False
) -> models.ScannedToken:
    row = db.query(models.ScannedToken).filter_by(token_address=token.token_address).first()
    now = dt.datetime.now(dt.timezone.utc)
    if row is None:
        row = models.ScannedToken(
            token_address=token.token_address,
            symbol=token.symbol,
            chain=token.chain,
            discovery_source=token.source,
            first_seen_at=now,
            evaluation_count=0,
            times_traded=0,
        )
        db.add(row)

    row.symbol = token.symbol
    row.last_evaluated_at = now
    row.evaluation_count = (row.evaluation_count or 0) + 1
    row.last_stage = stage
    row.last_reason = reason
    row.liquidity_usd = token.liquidity_usd
    row.volume_24h_usd = token.volume_24h_usd
    if traded:
        row.times_traded = (row.times_traded or 0) + 1
    return row


def _recently_evaluated(db: Session, token_address: str) -> bool:
    row = db.query(models.ScannedToken).filter_by(token_address=token_address).first()
    if row is None or row.last_evaluated_at is None:
        return False
    last = row.last_evaluated_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    elapsed_minutes = (dt.datetime.now(dt.timezone.utc) - last).total_seconds() / 60
    return elapsed_minutes < settings.SCANNER_RECHECK_MINUTES


async def scan_once(db: Session | None = None, rng: random.Random | None = None) -> dict:
    """Run one full scan cycle. Returns a summary dict for logging/tests.

    Takes an optional session so scripts/scan_once.py and the tests can
    drive a single cycle against their own session without going near the
    background loop.
    """
    blocked = scanner_blocked_reason()
    if blocked:
        logger.warning("scanner cycle skipped: %s", blocked)
        return {"skipped": blocked, "discovered": 0, "evaluated": 0, "traded": 0}

    owns_session = db is None
    db = db or SessionLocal()
    summary = {"discovered": 0, "prescreen_rejected": 0, "prescreen_tracked": 0,
               "skipped_recent": 0, "evaluated": 0, "traded": 0}
    rng = rng or random.Random()

    try:
        tokens = await discover_tokens()
        summary["discovered"] = len(tokens)

        considered = 0
        for token in tokens:
            if considered >= settings.SCANNER_MAX_TOKENS_PER_CYCLE:
                logger.info(
                    "scanner hit SCANNER_MAX_TOKENS_PER_CYCLE=%d, deferring the rest to the next cycle",
                    settings.SCANNER_MAX_TOKENS_PER_CYCLE,
                )
                break

            if _recently_evaluated(db, token.token_address):
                summary["skipped_recent"] += 1
                continue

            # Every discovered mint gets a DISCOVERED row, including ones
            # about to be rejected - the denominator of every conversion
            # rate in the funnel is "how many did we actually see?".
            pipeline.record(
                db, stage=pipeline.DISCOVERED, symbol=token.symbol,
                token_address=token.token_address, chain=token.chain, passed=True,
                reason=f"listed by {token.source}",
                detail={
                    "source": token.source,
                    "liquidity_usd": token.liquidity_usd,
                    "volume_24h_usd": token.volume_24h_usd,
                    "age_hours": token.age_hours,
                    "buys_24h": token.buys_24h,
                    "sells_24h": token.sells_24h,
                    "price_usd": token.price_usd,
                },
            )

            verdict = prescreen(token)
            # The full per-check breakdown, not just the first failure, so
            # the funnel can say WHICH threshold rejected the token. Held
            # onto because the reject sampler anchors its forward returns to
            # THIS event rather than writing a second one.
            prescreen_event = pipeline.record(
                db, stage=pipeline.PRESCREEN, symbol=token.symbol,
                token_address=token.token_address, chain=token.chain,
                passed=verdict.passed, reason=verdict.reason,
                detail=verdict.as_dict(),
            )
            if not verdict.passed:
                _record(db, token, stage=STAGE_PRESCREEN, reason=verdict.reason)
                summary["prescreen_rejected"] += 1
                if _track_prescreen_reject(db, token, rng=rng, event=prescreen_event):
                    summary["prescreen_tracked"] += 1
                considered += 1
                continue

            # Past the free filters - this one is worth spending network on.
            considered += 1
            summary["evaluated"] += 1
            try:
                signal = await handle_discovered_token(
                    db,
                    symbol=token.symbol,
                    token_address=token.token_address,
                    chain=token.chain,
                    price=token.price_usd or 0.0,
                    discovery_source=token.source,
                    extra={
                        "liquidity_usd": token.liquidity_usd,
                        "volume_24h_usd": token.volume_24h_usd,
                        "age_hours": token.age_hours,
                    },
                )
                db.flush()
                traded = (
                    db.query(models.Trade)
                    .filter_by(signal_id=signal.id, side="buy", status=models.TradeStatus.FILLED.value)
                    .first()
                    is not None
                )
                if traded:
                    summary["traded"] += 1
                    _record(db, token, stage=STAGE_TRADED, reason="opened a position", traded=True)
                else:
                    _record(
                        db, token, stage=STAGE_EVALUATED,
                        reason="passed pre-screen but was rejected downstream "
                               "(see risk events for this signal)",
                    )
            except Exception:
                # One bad candidate must never take down the whole cycle.
                logger.exception("scanner failed evaluating %s (%s)", token.symbol, token.token_address)
                _record(db, token, stage=STAGE_EVALUATED, reason="evaluation raised an error - see server logs")

        db.commit()
    except Exception:
        db.rollback()
        # Re-raised rather than swallowed: the supervisor owns failure
        # accounting, the throttled notification and the backoff, and a
        # pass that reports its own error looks like a success to it.
        raise
    finally:
        if owns_session:
            db.close()

    logger.info(
        "scan cycle: %d discovered, %d skipped (recent), %d rejected on pre-screen, "
        "%d fully evaluated, %d traded",
        summary["discovered"], summary["skipped_recent"], summary["prescreen_rejected"],
        summary["evaluated"], summary["traded"],
    )
    return summary


async def run_forever() -> None:
    blocked = scanner_blocked_reason()
    if blocked:
        logger.warning("automatic token scanner is NOT running: %s", blocked)
        return

    logger.info(
        "automatic token scanner starting (interval=%ss, min liquidity $%s, min 24h volume $%s)",
        settings.SCANNER_INTERVAL_SECONDS,
        f"{settings.SCANNER_MIN_LIQUIDITY_USD:,.0f}",
        f"{settings.SCANNER_MIN_VOLUME_24H_USD:,.0f}",
    )
    await run_supervised(
        "scanner", scan_once,
        interval_seconds=settings.SCANNER_INTERVAL_SECONDS, stop_event=_stop_event,
    )


def stop() -> None:
    _stop_event.set()
