"""Walk-forward backtesting engine.

Steps through a CandleSeries one bar at a time, using ONLY `series.head(i+1)`
(candles 0..i) to make every decision at bar i - the same look-ahead
guarantee `CandleSeries.up_to()` gives, just index-based since a backtest
already knows its own bar index rather than a raw timestamp. A decision
made at bar i is filled using a LATER bar's open (see `execution_delay_bars`
in BacktestConfig), never the price that produced the signal, and every
fill goes through app/backtesting/fills.py's slippage/spread/fee model.

Position sizing and stop/target math intentionally reuse
app/risk/manager.py's RiskManager and app/exits/manager.py's ExitManager -
both are now dependency-injectable (see their constructors) specifically so
the backtester runs the SAME formulas live trading does, not a
re-implementation that could silently drift from it. A backtest only
proves something about the strategy if the risk and exit logic it ran is
the logic that would actually run the trade.

Two things are intentionally NOT simulated, and both should be read as
rejections, not gaps papered over:
  - the rug-pull filter (app/rugcheck/) needs live scanner data with no
    historical record to replay against; every backtested "trade" implicitly
    assumes it would have passed screening
  - "min R:R met" is a REAL rejection path, not a formality: when resistance
    exists above price (app/signals/indicators.support_resistance), the
    reward target comes from the nearest level, and a level sitting too
    close to be worth the risk gets rejected. When there's no resistance
    overhead at all - a clean breakout, which score_support_resistance
    already scores bullishly - the target is projected via the configured
    R:R multiple instead, so a strong trend isn't punished for having
    already cleared every recent high.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from app.backtesting.fills import buy_fill, sell_fill
from app.backtesting.stats import compute_stats
from app.backtesting.types import BacktestConfig, BacktestResult, BacktestTrade
from app.data.candles import CandleSeries
from app.exits.manager import ExitManager
from app.risk.manager import RiskManager
from app.signals import indicators as ind
from app.signals.market_regime import classify
from app.signals.scoring import SignalScore, score_signal


@dataclass
class _OpenPosition:
    """Mirrors the attribute names app.exits.manager.ExitManager reads and
    mutates on a real `models.Position` - a plain dataclass works exactly
    the same way there, no SQLAlchemy needed for a backtest to reuse the
    identical exit logic.
    """

    entry_price: float
    stop_loss: float
    take_profit: float
    qty: float
    initial_qty: float
    opened_at: dt.datetime
    entry_bar_index: int
    signal_score: float
    market_regime: str
    risk_amount_usd: float
    highest_price_since_entry: float | None = None
    trailing_stop_active: bool = False
    break_even_applied: bool = False
    partial_exit_taken: bool = False
    recent_prices: list = field(default_factory=list)
    realized_pnl_usd: float = 0.0
    realized_fees_usd: float = 0.0


def _compute_stop_pct(history: CandleSeries, config: BacktestConfig) -> float:
    if config.use_atr_stop:
        atr_values = ind.atr_percent(history.highs, history.lows, history.closes, config.atr_period)
        atr_pct = atr_values[-1] if atr_values else None
        if atr_pct is not None and atr_pct > 0:
            return max(atr_pct * config.atr_multiple, config.min_stop_loss_pct)
    return max(config.fallback_stop_loss_pct, config.min_stop_loss_pct)


def _compute_reward_pct(history: CandleSeries, current_close: float, stop_pct: float,
                         config: BacktestConfig) -> tuple[float, str]:
    """Reward target from the nearest actual resistance level above price.

    No resistance overhead is a CLEAN BREAKOUT, not a missing target -
    app.signals.scoring.score_support_resistance already scores that
    bullishly (0.85, "no overhead resistance in recent range"), so treating
    it as a rejection here would contradict the score and specifically kill
    entries during exactly the strong trends a breakout strategy should be
    able to trade. In that case the target is projected via the configured
    R:R multiple instead. When resistance DOES exist, min_reward_risk is
    still a real gate below: a level sitting too close overhead genuinely
    isn't worth the risk and gets rejected by the caller.
    """
    _, resistance = ind.support_resistance(history.highs, history.lows)
    candidates = [r for r in resistance if r > current_close]
    if not candidates:
        return stop_pct * config.reward_risk_multiple, "no overhead resistance (clean breakout) - projected via R:R multiple"
    nearest = min(candidates)
    reward_pct = (nearest - current_close) / current_close
    return reward_pct, f"nearest resistance ${nearest:.8f}"


def _check_entry_confirmations(
    score: SignalScore, regime, config: BacktestConfig
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if score.score < config.min_score_to_enter:
        reasons.append(f"score {score.score:.1f} below minimum {config.min_score_to_enter:.1f}")
    if score.direction != "long":
        reasons.append(f"signal direction is {score.direction}, not long")
    if not score.reliable:
        reasons.append("signal score marked unreliable (too much missing data)")

    if config.require_regime_tradeable and not regime.is_tradeable:
        reasons.append(f"market regime not determinable: {'; '.join(regime.notes) or 'unknown'}")
    if config.allowed_trend_regimes is not None and regime.trend.value not in config.allowed_trend_regimes:
        reasons.append(
            f"market regime {regime.trend.value!r} not in allowed set {config.allowed_trend_regimes}"
        )

    # Explicit confirmations on top of the weighted score, per the spec's
    # entry-confirmation requirement: HTF trend, volume, and momentum must
    # each individually not be opposing the trade, not just be outweighed by
    # other factors in the composite. A factor with no data is skipped here,
    # not treated as a failed confirmation - warmup/short-history bars
    # already carry their own unreliable-score check above.
    required_factors = {
        "multi_timeframe": "higher-timeframe trend",
        "relative_volume": "volume",
        "momentum": "momentum",
    }
    by_name = {f.name: f for f in score.factors}
    for name, label in required_factors.items():
        factor = by_name.get(name)
        if factor is not None and factor.available and factor.score < 0.55:
            reasons.append(f"{label} does not confirm ({factor.reason})")

    return (len(reasons) == 0), reasons


def _make_trade(position: _OpenPosition, exit_time: dt.datetime, exit_fill_price: float,
                 exit_fee_usd: float, reason: str, bar_index: int) -> BacktestTrade:
    proceeds = position.qty * exit_fill_price
    cost_basis = position.qty * position.entry_price
    leg_pnl = proceeds - cost_basis - exit_fee_usd
    total_pnl = position.realized_pnl_usd + leg_pnl
    total_fees = position.realized_fees_usd + exit_fee_usd
    entry_notional = position.initial_qty * position.entry_price

    return BacktestTrade(
        entry_time=position.opened_at,
        exit_time=exit_time,
        entry_price=position.entry_price,
        exit_price=exit_fill_price,
        qty=position.initial_qty,
        size_usd=entry_notional,
        fees_usd=total_fees,
        pnl_usd=total_pnl,
        pnl_pct=(total_pnl / entry_notional) if entry_notional else 0.0,
        r_multiple=(total_pnl / position.risk_amount_usd) if position.risk_amount_usd else 0.0,
        exit_reason=reason,
        signal_score=position.signal_score,
        market_regime=position.market_regime,
        bars_held=bar_index - position.entry_bar_index,
    )


def run_backtest(
    series: CandleSeries, config: BacktestConfig | None = None, symbol: str | None = None
) -> BacktestResult:
    config = config or BacktestConfig()
    symbol = symbol or series.symbol

    result = BacktestResult(symbol=symbol)
    if len(series) <= config.warmup_bars:
        result.warnings.append(
            f"only {len(series)} candles, need more than warmup_bars={config.warmup_bars} - no bars walked"
        )
        result.stats = compute_stats([], [], config.starting_balance_usd)
        return result

    exit_overrides = config.exit_overrides or {}
    exit_manager = ExitManager(**exit_overrides)
    risk_manager = RiskManager(
        max_pct_per_trade=config.risk_pct_per_trade,
        max_trade_size_usd=config.max_trade_size_usd,
        max_exposure_per_token_pct=config.max_position_pct_of_portfolio,
        max_total_exposure_pct=config.max_position_pct_of_portfolio,
        max_concurrent_positions=1,
    )

    cash = config.starting_balance_usd
    position: _OpenPosition | None = None
    pending_entry: dict | None = None

    trades: list[BacktestTrade] = []
    equity_curve: list[tuple[dt.datetime, float]] = []
    rejections: list[dict] = []

    halted = False
    halt_reason = ""
    daily_pnl: dict[dt.date, float] = {}
    trades_opened_today: dict[dt.date, int] = {}
    recent_outcomes: list[bool] = []  # True = win, most recent last
    last_exit_bar_index: int | None = None

    for i in range(config.warmup_bars, len(series)):
        candle = series[i]
        history = series.head(i + 1)

        # --- resolve a pending entry whose fill bar has arrived ---
        if pending_entry is not None and i == pending_entry["fill_bar_index"]:
            reference_price = candle.open
            notional = risk_manager.position_size_usd(cash, stop_loss_pct=pending_entry["stop_pct"])
            if notional > 0:
                fill = buy_fill(reference_price, notional, config)
                qty = notional / fill.price
                cash -= notional + fill.fee_usd
                position = _OpenPosition(
                    entry_price=fill.price,
                    stop_loss=fill.price * (1 - pending_entry["stop_pct"]),
                    take_profit=fill.price * (1 + pending_entry["reward_pct"]),
                    qty=qty,
                    initial_qty=qty,
                    opened_at=candle.timestamp,
                    entry_bar_index=i,
                    signal_score=pending_entry["score"],
                    market_regime=pending_entry["regime"],
                    risk_amount_usd=notional * pending_entry["stop_pct"],
                    highest_price_since_entry=fill.price,
                    realized_fees_usd=fill.fee_usd,
                )
                trades_opened_today[candle.timestamp.date()] = (
                    trades_opened_today.get(candle.timestamp.date(), 0) + 1
                )
            else:
                rejections.append({
                    "bar": i, "time": candle.timestamp, "reason": "no portfolio room left to size the position",
                })
            pending_entry = None

        # --- manage an open position ---
        elif position is not None:
            action = exit_manager.evaluate(position, candle.close, now=candle.timestamp)
            if action.kind == "full":
                fill = sell_fill(candle.close, position.qty, config)
                trade = _make_trade(position, candle.timestamp, fill.price, fill.fee_usd, action.reason, i)
                trades.append(trade)
                cash += position.qty * fill.price - fill.fee_usd
                position = None
                last_exit_bar_index = i

                day = trade.exit_time.date()
                daily_pnl[day] = daily_pnl.get(day, 0.0) + trade.pnl_usd
                recent_outcomes.append(trade.is_win)
                if len(recent_outcomes) > risk_manager.max_consecutive_losses:
                    recent_outcomes.pop(0)

                if not halted:
                    loss_limit = config.starting_balance_usd * config.daily_loss_limit_pct
                    if daily_pnl[day] <= -loss_limit:
                        halted, halt_reason = True, f"daily realized loss breached {config.daily_loss_limit_pct:.0%} limit"
                        result.warnings.append(f"bar {i}: trading halted - {halt_reason}")
                    elif (
                        len(recent_outcomes) >= risk_manager.max_consecutive_losses
                        and not any(recent_outcomes[-risk_manager.max_consecutive_losses:])
                    ):
                        halted = True
                        halt_reason = f"{risk_manager.max_consecutive_losses} consecutive losing trades"
                        result.warnings.append(f"bar {i}: trading halted - {halt_reason}")

            elif action.kind == "partial":
                fill = sell_fill(candle.close, position.qty * action.fraction, config)
                qty_sold = position.qty * action.fraction
                cash += qty_sold * fill.price - fill.fee_usd
                position.realized_pnl_usd += qty_sold * fill.price - qty_sold * position.entry_price - fill.fee_usd
                position.realized_fees_usd += fill.fee_usd
                position.qty -= qty_sold
                if position.qty <= 1e-9:
                    trade = _make_trade(position, candle.timestamp, fill.price, 0.0, action.reason, i)
                    trades.append(trade)
                    position = None
                    last_exit_bar_index = i

        # --- flat: consider a new entry ---
        elif not halted and pending_entry is None:
            cooldown_ok = (
                last_exit_bar_index is None or (i - last_exit_bar_index) >= config.cooldown_bars
            )
            daily_limit_ok = trades_opened_today.get(candle.timestamp.date(), 0) < config.max_trades_per_day

            if cooldown_ok and daily_limit_ok:
                score = score_signal(
                    history, weights=config.weights, breakout_period=config.breakout_period,
                    higher_timeframe=config.higher_timeframe,
                )
                regime = classify(history)
                ok, reasons = _check_entry_confirmations(score, regime, config)

                if ok:
                    stop_pct = _compute_stop_pct(history, config)
                    reward_pct, reward_note = _compute_reward_pct(history, candle.close, stop_pct, config)
                    r_r = reward_pct / stop_pct if stop_pct > 0 else 0.0
                    if r_r < config.min_reward_risk:
                        rejections.append({
                            "bar": i, "time": candle.timestamp,
                            "reason": f"R:R {r_r:.2f} below minimum {config.min_reward_risk:.2f} ({reward_note})",
                        })
                    else:
                        fill_bar_index = max(i + config.execution_delay_bars, i + 1)
                        if fill_bar_index >= len(series):
                            rejections.append({
                                "bar": i, "time": candle.timestamp,
                                "reason": "not enough future bars left to fill this entry (end of data)",
                            })
                        else:
                            pending_entry = {
                                "fill_bar_index": fill_bar_index,
                                "stop_pct": stop_pct,
                                "reward_pct": reward_pct,
                                "score": score.score,
                                "regime": regime.label,
                            }
                else:
                    rejections.append({"bar": i, "time": candle.timestamp, "reason": "; ".join(reasons)})
            elif not cooldown_ok:
                pass  # quietly skip - cooldown is routine, not worth logging every bar
            # daily limit reached: also skip quietly for the same reason

        equity = cash + (position.qty * candle.close if position is not None else 0.0)
        equity_curve.append((candle.timestamp, equity))

    if position is not None:
        result.warnings.append(
            f"1 position still open at end of backtest data (opened bar {position.entry_bar_index}) - "
            "excluded from trade stats, but its paper value is included in the final equity/return"
        )
    if halted:
        result.warnings.append(f"backtest ended halted: {halt_reason}")

    result.trades = trades
    result.equity_curve = equity_curve
    result.rejections = rejections
    result.stats = compute_stats(trades, equity_curve, config.starting_balance_usd)
    return result
