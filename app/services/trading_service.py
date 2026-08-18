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

from app import models
from app.config import settings
from app.execution import get_execution_client
from app.notifications.notifier import notifier
from app.risk.manager import RiskManager, halt_trading
from app.rugcheck.filters import run_rug_checks
from app.schemas import TradingViewAlert
from app.services import portfolio

logger = logging.getLogger(__name__)
risk_manager = RiskManager()


def _instrument_id(symbol: str, token_address: str | None) -> str:
    if settings.EXECUTION_BACKEND == "cex":
        return symbol
    return token_address or symbol


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
    )
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


async def _handle_buy_signal(db: Session, signal: models.Signal) -> None:
    gate = risk_manager.check_can_open_position(db, symbol=signal.symbol)
    if not gate.allowed:
        db.add(models.RiskEvent(event_type="buy_blocked", details=gate.reason, signal_id=signal.id))
        await notifier.notify_rejection(signal, gate.reason)
        return

    existing = (
        db.query(models.Position)
        .filter_by(symbol=signal.symbol, status=models.PositionStatus.OPEN.value)
        .first()
    )
    if existing:
        logger.info("ignoring buy signal for %s: position already open (id=%s)", signal.symbol, existing.id)
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

    if not report.passed:
        reason = "; ".join(report.reasons) or "rug check failed"
        db.add(models.RiskEvent(event_type="rug_check_rejected", details=reason, signal_id=signal.id))
        await notifier.notify_rejection(signal, reason)
        return

    portfolio_value = await portfolio.get_portfolio_value_usd(db)
    total_exposure = await portfolio.get_open_positions_value_usd(db)
    symbol_exposure = await portfolio.get_symbol_exposure_usd(db, signal.symbol)
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
    )

    if not result.success:
        trade.status = models.TradeStatus.FAILED.value
        trade.error = result.error
        db.add(trade)
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

    portfolio.adjust_cash_balance(db, -size_usd)

    await notifier.notify_trade_executed(trade, extra=f"entry ${result.avg_price:.8f} | SL ${sl:.8f} / TP ${tp:.8f}")


async def _handle_sell_signal(db: Session, signal: models.Signal) -> None:
    position = (
        db.query(models.Position)
        .filter_by(symbol=signal.symbol, status=models.PositionStatus.OPEN.value)
        .first()
    )
    if not position:
        logger.info("sell signal for %s ignored: no open position", signal.symbol)
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
        symbol=position.symbol,
        token_address=position.token_address,
        chain=position.chain,
        side="sell",
        mode=position.mode,
        size_usd=position.qty * position.entry_price,
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
    db.add(trade)

    position.status = models.PositionStatus.CLOSED.value
    position.closed_at = now
    position.close_reason = reason

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
        symbol=position.symbol,
        token_address=position.token_address,
        chain=position.chain,
        side="sell",
        mode=position.mode,
        size_usd=qty_to_sell * position.entry_price,
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
