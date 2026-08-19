"""Orchestrates a single TradingView alert end to end:

    signal -> (buy) risk gate -> rug check -> position sizing -> execution -> persist -> notify
    signal -> (sell) find open position -> execution -> persist -> notify -> daily-loss check

This is the only place that opens or closes a position — the monitor loop
(app/monitor/position_monitor.py) calls back into `close_position` here for
stop-loss/take-profit/dev-wallet exits so the accounting logic isn't
duplicated.
"""
import datetime as dt
import logging

from sqlalchemy.orm import Session

from app import models, pipeline
from app.analysis import forward_returns
from app.config import settings
from app.concurrency import AlreadyReserved, reserve_entry
from app.execution import get_execution_client
from app.identity import describe, instrument_key
from app.notifications.notifier import notifier
from app.risk.manager import RiskManager, halt_trading
from app.safety import killswitch
from app.rugcheck.filters import run_rug_checks
from app.schemas import TradingViewAlert
from app.data.staleness import check_snapshot_freshness
from app.services import portfolio, price_feed
from app.signals.market_quality import score_market_quality
from app.signals.live_gate import evaluate_live_entry_signal
from app.strategy.version import current_label, register_current_version

logger = logging.getLogger(__name__)
risk_manager = RiskManager()


def _instrument_id(symbol: str, token_address: str | None) -> str:
    """What to hand the execution backend. Same rule as identity, and it
    delegates so the two can never drift apart."""
    return instrument_key(symbol, token_address)


def _open_position_for(db: Session, symbol: str, token_address: str | None):
    """The open position in THIS TOKEN, or None.

    Matched on the mint. Matching on the symbol meant a position in one
    PEPE blocked entry into an unrelated PEPE - and, worse, that a sell
    signal could close the wrong one. See app/identity.py.
    """
    key = instrument_key(symbol, token_address)
    for pos in db.query(models.Position).filter_by(status=models.PositionStatus.OPEN.value).all():
        if instrument_key(pos.symbol, pos.token_address) == key:
            return pos
    return None


async def handle_alert(db: Session, alert: TradingViewAlert) -> models.Signal:
    signal = models.Signal(
        symbol=alert.symbol,
        token_address=alert.token_address,
        chain=alert.chain,
        signal_type=alert.signal,
        price=alert.price,
        tv_timestamp=alert.parsed_time(),
        rsi=alert.rsi,
        ema9=alert.ema9,
        ema21=alert.ema21,
        volume=alert.volume,
        volume_sma=alert.volume_sma,
        breakout_level=alert.breakout_level,
        raw_payload=alert.model_dump(exclude={"secret"}),
        strategy_version=current_label(),
    )
    register_current_version(db)
    db.add(signal)
    db.flush()

    try:
        if alert.signal == "buy":
            await _handle_buy_signal(db, signal)
        elif alert.signal == "sell":
            await _handle_sell_signal(db, signal)
        else:
            logger.warning("ignoring signal id=%s with unknown signal type %r", signal.id, alert.signal)
    except Exception:
        logger.exception("unhandled error processing signal id=%s (%s)", signal.id, signal.symbol)
        await notifier.notify_error(f"Unhandled error processing signal {signal.id} ({signal.symbol}) - see server logs")
        raise

    return signal


async def handle_discovered_token(
    db: Session,
    *,
    symbol: str,
    token_address: str,
    chain: str,
    price: float,
    discovery_source: str,
    extra: dict | None = None,
) -> models.Signal:
    """Entry point for the automatic scanner (app/scanner/).

    Records the candidate as a Signal with source="scanner" and then runs it
    through the IDENTICAL buy path a TradingView webhook alert takes - the
    risk gate, the live signal score, the rug check, exposure-aware sizing,
    execution and the position monitor are all the same code. The scanner is
    a new *source of signals*, deliberately not a second trading path: a
    parallel implementation could drift from this one and quietly lose a
    protection, which is exactly the failure this avoids.
    """
    signal = models.Signal(
        symbol=symbol,
        token_address=token_address,
        chain=chain,
        signal_type="buy",
        price=price,
        source="scanner",
        raw_payload={"discovery_source": discovery_source, **(extra or {})},
        strategy_version=current_label(),
    )
    register_current_version(db)
    db.add(signal)
    db.flush()

    try:
        await _handle_buy_signal(db, signal)
    except Exception:
        logger.exception("unhandled error processing scanner candidate %s (%s)", symbol, token_address)
        await notifier.notify_error(f"Unhandled error processing scanner candidate {symbol} - see server logs")
        raise

    return signal


async def _handle_buy_signal(db: Session, signal: models.Signal) -> None:
    """Evaluate one buy candidate and, if every gate clears, open a position.

    Wrapped in an entry reservation because the gates and the position
    creation are separated by several network round-trips, and both the
    scanner loop and the webhook run on the same event loop. Without it two
    candidates for the same mint could interleave, both see an empty book,
    and both open a position - double the intended size, split across two
    rows so the exposure cap never sees it. See app/concurrency.py.
    """
    key = instrument_key(signal.symbol, signal.token_address)
    try:
        async with reserve_entry(key):
            await _evaluate_and_enter(db, signal)
    except AlreadyReserved:
        reason = (
            f"an entry for {describe(signal.symbol, signal.token_address)} is already in "
            "flight - skipping the duplicate rather than opening a second position"
        )
        logger.info(reason)
        db.add(models.RiskEvent(event_type="duplicate_entry_blocked", details=reason, signal_id=signal.id))


def _stage(db: Session, signal: models.Signal, stage: str, passed: bool,
           reason: str = "", score: float | None = None, detail: dict | None = None) -> None:
    """Record one pipeline stage for this signal. See app/pipeline.py."""
    pipeline.record(
        db, stage=stage, symbol=signal.symbol, token_address=signal.token_address,
        chain=signal.chain, passed=passed, reason=reason, score=score,
        detail=detail, signal_id=signal.id,
    )


async def _evaluate_and_enter(db: Session, signal: models.Signal) -> None:
    # The global gate first: before asking whether THIS trade is within
    # limits, ask whether the bot's own state can be trusted to answer
    # that. Sizing a position off a wrong cash balance, or off prices that
    # stopped updating an hour ago, produces trades that look fine and a
    # record nobody can interpret. See app/safety/killswitch.py.
    integrity = await killswitch.may_open_position(db)
    if not integrity.may_trade:
        _stage(db, signal, pipeline.RISK, False, f"kill switch: {integrity.reason}")
        db.add(models.RiskEvent(
            event_type="kill_switch_blocked", details=integrity.reason, signal_id=signal.id
        ))
        logger.error("entry blocked by the kill switch: %s", integrity.reason)
        await notifier.notify_rejection(signal, f"kill switch: {integrity.reason}")
        return

    gate = risk_manager.check_can_open_position(
        db, symbol=signal.symbol, token_address=signal.token_address
    )
    if not gate.allowed:
        _stage(db, signal, pipeline.RISK, False, gate.reason)
        db.add(models.RiskEvent(event_type="buy_blocked", details=gate.reason, signal_id=signal.id))
        await notifier.notify_rejection(signal, gate.reason)
        return

    existing = _open_position_for(db, signal.symbol, signal.token_address)
    if existing:
        reason = f"position already open (id={existing.id})"
        _stage(db, signal, pipeline.RISK, False, reason)
        logger.info(
            "ignoring buy signal for %s: %s",
            describe(signal.symbol, signal.token_address), reason,
        )
        return

    _stage(db, signal, pipeline.RISK, True, "within every portfolio limit")

    # Set when the technical gate says "do not trade this". Everything after
    # that point still runs so the early engine can watch the candidate, but
    # nothing may open a position once it is True.
    technical_rejected = False

    if settings.LIVE_SIGNAL_SCORE_ENABLED:
        score = await evaluate_live_entry_signal(signal.chain, signal.token_address, signal.symbol)
        if score is None:
            reason = (
                f"live signal score unavailable for {signal.symbol} - no trustworthy candle data "
                f"(need >={settings.SIGNAL_SCORE_MIN_CANDLES} live {settings.SIGNAL_SCORE_TIMEFRAME} candles)"
            )
            _stage(db, signal, pipeline.HISTORY, False, reason)
            db.add(models.RiskEvent(event_type="signal_score_unavailable", details=reason, signal_id=signal.id))
            await notifier.notify_rejection(signal, reason)
            return

        _stage(db, signal, pipeline.HISTORY, True, "enough trustworthy candles to score")

        signal.signal_score = score.score
        signal.signal_score_reliable = score.reliable
        signal.signal_score_factors = score.as_dict()["factors"]

        # Recorded BEFORE the threshold test, and recorded whether or not
        # it passes. A dataset of only the setups that cleared the bar
        # cannot say whether the bar is in the right place.
        scored_event = pipeline.record(
            db, stage=pipeline.TECHNICAL_SCORE, symbol=signal.symbol,
            token_address=signal.token_address, chain=signal.chain,
            passed=score.score >= settings.MIN_SIGNAL_SCORE_TO_ENTER and score.reliable,
            reason=f"scored {score.score:.1f}/100 (threshold {settings.MIN_SIGNAL_SCORE_TO_ENTER:.1f})",
            score=score.score, signal_id=signal.id,
            detail={
                "reliable": score.reliable,
                "threshold": settings.MIN_SIGNAL_SCORE_TO_ENTER,
                "direction": score.direction,
                "factors": score.as_dict()["factors"],
            },
        )

        # Follow this candidate's price forward whether or not it is about
        # to be rejected. Tracking only the winners of the threshold test
        # would make it impossible to ever learn that the threshold is
        # wrong - see app/analysis/forward_returns.py.
        if scored_event is not None and forward_returns.enabled():
            db.flush()
            forward_returns.schedule(
                db,
                pipeline_event_id=scored_event.id,
                token_address=signal.token_address,
                symbol=signal.symbol,
                score=score.score,
                price_at_signal=signal.price or 0.0,
            )

        if score.score < settings.MIN_SIGNAL_SCORE_TO_ENTER or not score.reliable:
            reason = (
                f"signal score {score.score:.1f}/100 "
                f"{'(unreliable - too much missing data) ' if not score.reliable else ''}"
                f"below minimum {settings.MIN_SIGNAL_SCORE_TO_ENTER:.1f}"
            )
            db.add(models.RiskEvent(event_type="signal_score_rejected", details=reason, signal_id=signal.id))
            await notifier.notify_rejection(signal, reason)
            # Rejected for TRADING - but not necessarily discarded. The whole
            # point of the Early Signal Engine is that a chart which does not
            # look good YET can be one whose demand is arriving, and returning
            # here would mean the engine only ever sees candidates the bot was
            # already about to buy. `technical_rejected` carries that decision
            # forward: the security and market-quality gates still run (the
            # engine must never see a token that failed security), the engine
            # gets its look, and the function returns before sizing.
            technical_rejected = True
            if not _worth_an_early_look(score):
                return
    else:
        logger.warning("LIVE_SIGNAL_SCORE_ENABLED=false - buy signals are NOT being scored before entry")

    # --- market quality: can this actually be traded, regardless of setup? ---
    # Separate from the signal score on purpose. A token can print a
    # textbook breakout on volume that is entirely wash-traded; the signal
    # engine reads price/volume shape and has no way to tell. This does.
    if settings.MARKET_QUALITY_ENABLED:
        market = await price_feed.get_market_snapshot(signal.token_address) if signal.token_address else None

        freshness = check_snapshot_freshness(market)
        if not freshness.fresh:
            reason = f"market data not usable: {freshness.reason}"
            db.add(models.RiskEvent(event_type="stale_data_rejected", details=reason, signal_id=signal.id))
            await notifier.notify_rejection(signal, reason)
            return

        quality = score_market_quality(market, min_liquidity_usd=settings.MIN_LIQUIDITY_USD)
        signal.market_quality_score = quality.score
        signal.market_quality_factors = quality.as_dict()["factors"]

        _stage(
            db, signal, pipeline.MARKET_QUALITY,
            passed=quality.reliable and quality.score >= settings.MIN_MARKET_QUALITY_SCORE,
            reason=f"scored {quality.score:.1f}/100 (threshold {settings.MIN_MARKET_QUALITY_SCORE:.1f})",
            score=quality.score,
            detail={"reliable": quality.reliable, "factors": quality.as_dict()["factors"]},
        )

        if not quality.reliable or quality.score < settings.MIN_MARKET_QUALITY_SCORE:
            concerns = "; ".join(f.reason for f in quality.concerns[:3]) or "; ".join(quality.warnings)
            reason = (
                f"market quality {quality.score:.1f}/100 "
                f"{'(unreliable - too much missing data) ' if not quality.reliable else ''}"
                f"below minimum {settings.MIN_MARKET_QUALITY_SCORE:.1f}"
                + (f" - {concerns}" if concerns else "")
            )
            db.add(models.RiskEvent(event_type="market_quality_rejected", details=reason, signal_id=signal.id))
            await notifier.notify_rejection(signal, reason)
            return

    report = await run_rug_checks(signal.chain, signal.token_address)
    db.add(
        models.RugCheckResult(
            signal_id=signal.id,
            passed=report.passed,
            reasons=report.reasons,
            ownership_renounced=report.ownership_renounced,
            mint_disabled=report.mint_disabled,
            liquidity_locked=report.liquidity_locked,
            is_honeypot=report.is_honeypot,
            top10_holder_pct=report.top10_holder_pct,
            liquidity_usd=report.liquidity_usd,
            dev_wallet_pct=report.dev_wallet_pct,
            scanner_source=report.source,
            chain_screened=report.chain,
            lookup_outcomes=report.lookup_outcomes,
            rug_risk_score=report.rug_risk_score,
            rug_risk_level=report.rug_risk_level,
            rug_risk_factors=report.rug_risk_factors,
        )
    )

    _stage(
        db, signal, pipeline.SECURITY, report.passed,
        "; ".join(report.reasons) or ("passed every security check" if report.passed else "rug check failed"),
        score=report.rug_risk_score,
        detail={
            "source": report.source,
            "rug_risk_level": report.rug_risk_level,
            "liquidity_usd": report.liquidity_usd,
            "top10_holder_pct": report.top10_holder_pct,
            "dev_wallet_pct": report.dev_wallet_pct,
            "lookup_outcomes": report.lookup_outcomes,
        },
    )

    if not report.passed:
        reason = "; ".join(report.reasons) or "rug check failed"
        db.add(models.RiskEvent(event_type="rug_check_rejected", details=reason, signal_id=signal.id))
        await notifier.notify_rejection(signal, reason)
        return

    # --- early signal: a technical rejection is not always a discard ----
    # Runs after security and market quality have both passed, so nothing
    # here can rescue a token that failed either. Its only power is to put
    # a candidate the technical gate turned down onto the WATCH list
    # instead of throwing it away - which is the whole point of having a
    # third state.
    if settings.EARLY_SIGNAL_ENABLED and signal.token_address:
        try:
            await _consider_for_watchlist(db, signal, report, scored_event)
        except Exception:
            logger.exception("early-signal evaluation failed for %s - continuing", signal.symbol)

    # The technical gate already refused this one. Everything above ran so
    # the early engine could see it; nothing below may act on it.
    if technical_rejected:
        return

    portfolio_value = await portfolio.get_portfolio_value_usd(db)
    total_exposure = await portfolio.get_open_positions_value_usd(db)
    symbol_exposure = await portfolio.get_token_exposure_usd(
        db, signal.symbol, signal.token_address
    )
    size_usd = risk_manager.position_size_usd(
        portfolio_value,
        current_total_exposure_usd=total_exposure,
        current_symbol_exposure_usd=symbol_exposure,
    )

    if size_usd <= 0:
        reason = (
            f"no exposure room left (portfolio ${portfolio_value:,.0f}, "
            f"total exposure ${total_exposure:,.0f}, {signal.symbol} exposure ${symbol_exposure:,.0f})"
        )
        db.add(models.RiskEvent(event_type="exposure_cap_rejected", details=reason, signal_id=signal.id))
        await notifier.notify_rejection(signal, reason)
        return

    if report.liquidity_usd and size_usd > report.liquidity_usd * settings.MAX_PRICE_IMPACT_PCT:
        reason = (
            f"position size ${size_usd:,.0f} would exceed max price-impact budget "
            f"vs ${report.liquidity_usd:,.0f} liquidity"
        )
        db.add(models.RiskEvent(event_type="liquidity_depth_rejected", details=reason, signal_id=signal.id))
        await notifier.notify_rejection(signal, reason)
        return

    instrument = _instrument_id(signal.symbol, signal.token_address)
    client = get_execution_client()
    result = await client.buy(instrument, size_usd, settings.SLIPPAGE_BPS)

    trade = models.Trade(
        signal_id=signal.id,
        symbol=signal.symbol,
        token_address=signal.token_address,
        chain=signal.chain,
        side="buy",
        mode=models.TradeMode.LIVE.value if settings.LIVE_TRADING else models.TradeMode.PAPER.value,
        size_usd=size_usd,
        strategy_version=current_label(),
    )

    if not result.success:
        trade.status = models.TradeStatus.FAILED.value
        trade.error = result.error
        db.add(trade)
        # A reverted swap is a real, common outcome and a real cost. It is
        # recorded as a pipeline stage so the funnel shows how much of the
        # gap between "signals" and "positions" is execution rather than
        # filtering.
        _stage(db, signal, pipeline.PAPER_EXECUTION, False, result.error or "fill failed",
               detail={"size_usd": size_usd})
        await notifier.notify_error(f"Buy failed for {signal.symbol}: {result.error}")
        return

    sl, tp = risk_manager.stop_loss_take_profit(result.avg_price)
    now = dt.datetime.now(dt.timezone.utc)
    trade.status = models.TradeStatus.FILLED.value
    trade.qty = result.filled_qty
    trade.entry_price = result.avg_price
    trade.stop_loss = sl
    trade.take_profit = tp
    trade.tx_hash = result.tx_hash
    trade.opened_at = now
    trade.fee_usd = result.fee_usd
    trade.execution_cost_pct = result.execution_cost_pct
    trade.fill_delay_seconds = result.fill_delay_seconds
    db.add(trade)
    db.flush()

    position = models.Position(
        symbol=signal.symbol,
        token_address=signal.token_address,
        chain=signal.chain,
        qty=result.filled_qty,
        initial_qty=result.filled_qty,
        entry_price=result.avg_price,
        stop_loss=sl,
        take_profit=tp,
        status=models.PositionStatus.OPEN.value,
        mode=trade.mode,
        dev_wallet_pct_at_entry=report.dev_wallet_pct,
        entry_trade_id=trade.id,
        opened_at=now,
        highest_price_since_entry=result.avg_price,
    )
    db.add(position)
    db.flush()
    trade.position_id = position.id

    _stage(
        db, signal, pipeline.PAPER_EXECUTION, True,
        f"filled {result.filled_qty:,.4f} @ ${result.avg_price:.8f}",
        detail={
            "size_usd": size_usd,
            "signal_price": signal.price,
            "fill_price": result.avg_price,
            "slippage_vs_signal_pct": (
                (result.avg_price / signal.price - 1) * 100 if signal.price else None
            ),
            "fee_usd": result.fee_usd,
            "execution_cost_pct": result.execution_cost_pct,
            "fill_delay_seconds": result.fill_delay_seconds,
        },
    )
    _stage(
        db, signal, pipeline.OPEN_POSITION, True,
        f"position {position.id} open",
        detail={"position_id": position.id, "stop_loss": sl, "take_profit": tp},
    )

    portfolio.adjust_cash_balance(db, -size_usd)

    await notifier.notify_trade_executed(trade, extra=f"entry ${result.avg_price:.8f} | SL ${sl:.8f} / TP ${tp:.8f}")


def _worth_an_early_look(score) -> bool:
    """Should a technically-rejected candidate still be shown to the early engine?

    Continuing past the technical gate costs a security lookup and a
    candle fetch per candidate, and the scanner sees a lot of candidates.
    So the extra work is bounded: only scores within
    EARLY_SIGNAL_TECHNICAL_MARGIN of the entry threshold continue. A token
    scoring 12/100 is not an early opportunity the chart has not caught up
    with, it is a bad token, and paying an API call to confirm that on
    every scan would starve the rate budget the real candidates need.

    An UNRELIABLE score does not continue either. Its number is not a
    measurement, so "within 25 points of the threshold" would be reading
    meaning into a value the scorer already disclaimed.
    """
    if not settings.EARLY_SIGNAL_ENABLED:
        return False
    if not score.reliable:
        return False
    floor = settings.MIN_SIGNAL_SCORE_TO_ENTER - settings.EARLY_SIGNAL_TECHNICAL_MARGIN
    return score.score >= floor


async def _consider_for_watchlist(
    db: Session, signal: models.Signal, report, scored_event=None
) -> None:
    """Score a candidate on the early engine and record the verdict.

    Never opens a position and never blocks one. It records a WATCH entry
    so a token that is promising but unconfirmed stays under observation,
    and it stores the observation that makes flow features measurable next
    time. The actual entry decision stays entirely with the code below
    this call.
    """
    from app.early import watchlist as wl
    from app.early.engine import evaluate as evaluate_early

    market = await price_feed.get_market_snapshot(signal.token_address)
    if market is None:
        return

    wl.store_observation(db, signal.symbol, signal.token_address, market)
    observations = wl.recent_observations(db, signal.token_address)

    series = None
    try:
        from app.data.candles import Timeframe
        from app.data.live_provider import fetch_candles

        series = await fetch_candles(
            signal.chain, signal.token_address, signal.symbol, Timeframe.M5, 300
        )
    except Exception:
        # Network and parse failures are expected for a brand-new pool and
        # must not take down the scan. Logged at warning rather than debug:
        # a persistent failure here silently disables the whole early
        # engine, because a candidate with no candles always fails the
        # data-quality gate and is skipped.
        logger.warning(
            "no candle history for early scoring of %s - it cannot be scored",
            signal.symbol, exc_info=True,
        )

    verdict = evaluate_early(
        series=series,
        market=market,
        observations=observations,
        security_passed=report.passed,
        security_reason="; ".join(report.reasons),
        security_score=report.rug_risk_score,
        technical_score=signal.signal_score,
        market_quality_score=signal.market_quality_score,
    )

    entry = wl.record(
        db,
        token_address=signal.token_address,
        symbol=signal.symbol,
        chain=signal.chain,
        verdict=verdict,
        price=signal.price,
    )
    # Back-fill the early verdict onto the forward-return rows that were
    # scheduled earlier in this same pass. They are scheduled before the
    # early engine runs, because a candidate rejected by the TECHNICAL gate
    # must still be followed forward - but that ordering means the early
    # score is not known yet at scheduling time. Without this attach, every
    # ForwardReturn.early_score would be NULL and the entire early
    # calibration would silently have nothing to read.
    if scored_event is not None:
        forward_returns.attach_early(db, scored_event.id, verdict)

    if entry is not None:
        logger.info(
            "early signal %s: %s (early %.0f, late risk %.0f, %s)",
            signal.symbol, verdict.decision.value,
            verdict.early_score or 0, verdict.late_risk or 0,
            verdict.stage.value if verdict.stage else "unstaged",
        )


async def _handle_sell_signal(db: Session, signal: models.Signal) -> None:
    position = _open_position_for(db, signal.symbol, signal.token_address)
    if not position:
        logger.info(
            "sell signal for %s ignored: no open position",
            describe(signal.symbol, signal.token_address),
        )
        return
    await close_position(db, position, reason="sell signal from TradingView", signal_id=signal.id)


async def close_position(db: Session, position: models.Position, reason: str, signal_id: int | None = None) -> None:
    """Exit an open position. Called for TradingView sell signals AND for
    stop-loss / take-profit / dev-wallet-sell triggers from the monitor loop.
    """
    instrument = _instrument_id(position.symbol, position.token_address)
    client = get_execution_client()
    result = await client.sell(instrument, position.qty, settings.SLIPPAGE_BPS)

    trade = models.Trade(
        signal_id=signal_id,
        position_id=position.id,
        symbol=position.symbol,
        token_address=position.token_address,
        chain=position.chain,
        side="sell",
        mode=position.mode,
        size_usd=position.qty * position.entry_price,
        strategy_version=current_label(),
    )

    if not result.success:
        trade.status = models.TradeStatus.FAILED.value
        trade.error = result.error
        db.add(trade)
        await notifier.notify_error(f"Sell failed for {position.symbol} ({reason}): {result.error}")
        return

    proceeds = result.filled_qty * result.avg_price
    cost_basis = position.qty * position.entry_price
    pnl_usd = proceeds - cost_basis
    pnl_pct = (result.avg_price / position.entry_price - 1) if position.entry_price else 0.0
    now = dt.datetime.now(dt.timezone.utc)

    trade.status = models.TradeStatus.FILLED.value
    trade.qty = result.filled_qty
    trade.exit_price = result.avg_price
    trade.pnl_usd = pnl_usd
    trade.pnl_pct = pnl_pct
    trade.tx_hash = result.tx_hash
    trade.closed_at = now
    trade.close_reason = reason
    trade.fee_usd = result.fee_usd
    trade.execution_cost_pct = result.execution_cost_pct
    trade.fill_delay_seconds = result.fill_delay_seconds
    db.add(trade)

    position.status = models.PositionStatus.CLOSED.value
    position.closed_at = now
    position.close_reason = reason

    pipeline.record(
        db, stage=pipeline.EXIT, symbol=position.symbol,
        token_address=position.token_address, chain=position.chain,
        passed=pnl_usd > 0, reason=reason, signal_id=signal_id,
        detail={
            "position_id": position.id,
            "entry_price": position.entry_price,
            "exit_price": result.avg_price,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct * 100,
            "fee_usd": result.fee_usd,
            "held_hours": (
                (now - position.opened_at.replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600
                if position.opened_at and position.opened_at.tzinfo is None
                else (now - position.opened_at).total_seconds() / 3600 if position.opened_at else None
            ),
        },
    )

    portfolio.adjust_cash_balance(db, proceeds)

    await notifier.notify_trade_executed(
        trade, extra=f"closed ({reason}) | exit ${result.avg_price:.8f} | P&L ${pnl_usd:,.2f} ({pnl_pct * 100:+.1f}%)"
    )

    await _check_halt_conditions(db)


async def partial_close_position(
    db: Session, position: models.Position, fraction: float, reason: str, signal_id: int | None = None
) -> None:
    """Sell part of an open position and leave the rest open.

    `fraction` is applied to the position's CURRENT qty. Today only one
    partial exit ever fires per position (app/exits/manager.py sets
    `partial_exit_taken` after the first), so current qty and initial qty
    are the same at the moment this runs - but reading current qty rather
    than `initial_qty` keeps this correct if that ever changes to allow more
    than one partial exit.
    """
    fraction = max(0.0, min(fraction, 1.0))
    qty_to_sell = position.qty * fraction
    if qty_to_sell <= 0:
        return

    instrument = _instrument_id(position.symbol, position.token_address)
    client = get_execution_client()
    result = await client.sell(instrument, qty_to_sell, settings.SLIPPAGE_BPS)

    trade = models.Trade(
        signal_id=signal_id,
        position_id=position.id,
        symbol=position.symbol,
        token_address=position.token_address,
        chain=position.chain,
        side="sell",
        mode=position.mode,
        size_usd=qty_to_sell * position.entry_price,
        strategy_version=current_label(),
    )

    if not result.success:
        trade.status = models.TradeStatus.FAILED.value
        trade.error = result.error
        db.add(trade)
        await notifier.notify_error(f"Partial sell failed for {position.symbol} ({reason}): {result.error}")
        return

    proceeds = result.filled_qty * result.avg_price
    cost_basis = result.filled_qty * position.entry_price
    pnl_usd = proceeds - cost_basis
    pnl_pct = (result.avg_price / position.entry_price - 1) if position.entry_price else 0.0
    now = dt.datetime.now(dt.timezone.utc)

    trade.status = models.TradeStatus.FILLED.value
    trade.qty = result.filled_qty
    trade.exit_price = result.avg_price
    trade.pnl_usd = pnl_usd
    trade.pnl_pct = pnl_pct
    trade.tx_hash = result.tx_hash
    trade.closed_at = now
    trade.close_reason = reason
    trade.fee_usd = result.fee_usd
    trade.execution_cost_pct = result.execution_cost_pct
    trade.fill_delay_seconds = result.fill_delay_seconds
    db.add(trade)

    position.qty -= result.filled_qty
    position.realized_pnl_usd = (position.realized_pnl_usd or 0.0) + pnl_usd

    portfolio.adjust_cash_balance(db, proceeds)

    await notifier.notify_trade_executed(
        trade,
        extra=(
            f"partial exit ({reason}) | {fraction * 100:.0f}% sold | "
            f"exit ${result.avg_price:.8f} | P&L ${pnl_usd:,.2f} ({pnl_pct * 100:+.1f}%)"
        ),
    )

    if position.qty <= 1e-9:
        position.status = models.PositionStatus.CLOSED.value
        position.closed_at = now
        position.close_reason = reason

    await _check_halt_conditions(db)


async def _check_halt_conditions(db: Session) -> None:
    """Run the post-trade halt checks shared by full and partial exits."""
    daily = risk_manager.evaluate_daily_loss(db)
    if not daily.allowed:
        halt_trading(db, daily.reason)
        await notifier.notify_risk_halt(daily.reason)
        return

    streak = risk_manager.evaluate_consecutive_losses(db)
    if not streak.allowed:
        halt_trading(db, streak.reason)
        await notifier.notify_risk_halt(streak.reason)
