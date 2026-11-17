"""Read-only performance audit of stored strategy runs.

Nothing here simulates, changes or re-runs anything. Every number is computed
from rows a run already stored (``strategy_runs``, ``strategy_signals``,
``strategy_trades``, ``strategy_equity``) and, for the buy-and-hold benchmark,
from the stored daily bars of the same period.

Definitions used throughout (stated here so they can be checked):

* A trade is a **win** when its net result (after assumed slippage) is above
  zero, a **loss** when below zero, and **breakeven** when exactly zero.
* **Profit factor** = sum of winning net results / |sum of losing net results|.
* **Expectancy** = mean net result per trade, in dollars and in R (multiples of
  the risk planned at entry).
* **Maximum drawdown** = largest fall of the closing equity from its highest
  earlier value, the starting equity included. **Recovery** is the first later
  session whose equity is back at or above that earlier high.
* **Holding period** = sessions from the entry session to the exit session
  (a trade opened and closed on the same day holds 0 sessions).
* A symbol's **strategy return** is its net profit divided by the run's
  starting equity: its contribution inside the shared account. It is NOT the
  result of trading that symbol alone, which would need a new simulation.
* **Buy-and-hold** buys at the open of the first session of the period and
  sells at the close of the last one, on the same split-adjusted bars. Dividends
  are not included, which understates it slightly for dividend payers.
"""

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import StrategyEquityPoint, StrategyRun, StrategySignal, StrategyTrade
from app.strategies.assumptions import buy_fill, sell_fill
from app.strategies.data import DailyData

#: Trading sessions per year, used only to annualise volatility and Sharpe.
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class TradeView:
    """The fields of a stored trade the audit uses."""

    trade_id: int
    symbol: str
    entry_session: date
    exit_session: date
    shares: int
    entry_price: float
    exit_price: float
    exit_reason: str
    sessions_held: int
    gross_pnl: float
    slippage_cost: float
    net_pnl: float
    risk_amount: float
    r_multiple: float


@dataclass(frozen=True)
class EquityView:
    session: date
    equity: float
    open_positions: int


@dataclass(frozen=True)
class TradeStatistics:
    trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float | None
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float | None
    average_win: float | None
    average_loss: float | None
    payoff_ratio: float | None
    expectancy: float | None
    expectancy_r: float | None
    average_win_r: float | None
    average_loss_r: float | None
    total_slippage: float
    average_sessions_held: float | None
    median_sessions_held: float | None
    average_sessions_held_wins: float | None
    average_sessions_held_losses: float | None
    largest_wins: list[TradeView]
    largest_losses: list[TradeView]
    exits_by_reason: dict[str, int]
    max_consecutive_losses: int


@dataclass(frozen=True)
class Drawdown:
    max_drawdown: float  # fraction, e.g. 0.12 = 12 %
    max_drawdown_amount: float
    peak_session: date | None  # None: the peak was the starting equity, before the first session
    peak_equity: float
    trough_session: date | None
    trough_equity: float
    recovery_session: date | None
    sessions_peak_to_trough: int | None
    sessions_trough_to_recovery: int | None
    sessions_underwater: int | None  # peak to recovery, or to the end when not recovered
    recovered: bool


@dataclass(frozen=True)
class SymbolBreakdown:
    symbol: str
    trades: int
    wins: int
    net_profit: float
    gross_profit: float
    gross_loss: float
    strategy_return: float  # net profit / starting equity
    share_of_net_profit: float | None
    share_of_gross_profit: float | None


@dataclass(frozen=True)
class Concentration:
    best_symbol: str | None
    best_symbol_share_of_net_profit: float | None
    net_profit_without_best_symbol: float | None
    best_trade_share_of_net_profit: float | None
    net_profit_without_best_trade: float | None
    top_five_trades_share_of_net_profit: float | None
    symbols_with_net_profit: int
    symbols_with_net_loss: int
    heavily_dependent: bool | None
    reason: str


@dataclass(frozen=True)
class Integrity:
    signals_total: int
    signals_taken: int
    signals_skipped: int
    skipped_by_reason: dict[str, int]
    trades_stored: int
    trades_in_run_header: int
    losing_trades_stored: int
    every_taken_signal_has_one_trade: bool
    no_trade_without_a_taken_signal: bool
    ending_equity_matches_trades: bool
    equity_curve_complete: bool
    trade_identity_holds: bool
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class RunAudit:
    run_id: int
    strategy: str
    strategy_version: str
    period: str
    period_start: date
    period_end: date
    cost_scenario: str
    slippage_bps: float
    parameters_hash: str
    data_fingerprint: str
    result_fingerprint: str | None
    starting_equity: float
    ending_equity: float
    total_return: float
    annualised_return: float | None
    annualised_volatility: float | None
    sharpe_ratio: float | None
    sessions: int
    exposure: float | None  # share of sessions ending with at least one position open
    average_open_positions: float | None
    statistics: TradeStatistics
    drawdown: Drawdown
    symbols: list[SymbolBreakdown]
    concentration: Concentration
    integrity: Integrity
    trades: list[TradeView] = field(repr=False)


@dataclass(frozen=True)
class BuyAndHold:
    symbol: str
    first_session: date
    last_session: date
    entry_price: float  # open of the first session
    exit_price: float  # close of the last session
    total_return: float
    total_return_after_costs: float
    max_drawdown: float


# =============================================================================
# Reading
# =============================================================================


def load_run(session: Session, run_id: int):
    run = session.get(StrategyRun, run_id)
    if run is None:
        raise LookupError(f"No strategy run with id {run_id}.")
    signals = list(
        session.execute(
            select(StrategySignal)
            .where(StrategySignal.run_id == run_id)
            .order_by(StrategySignal.id)
        ).scalars()
    )
    trades = list(
        session.execute(
            select(StrategyTrade).where(StrategyTrade.run_id == run_id).order_by(StrategyTrade.id)
        ).scalars()
    )
    equity = list(
        session.execute(
            select(StrategyEquityPoint)
            .where(StrategyEquityPoint.run_id == run_id)
            .order_by(StrategyEquityPoint.session)
        ).scalars()
    )
    return run, signals, trades, equity


def latest_runs(session: Session, strategy: str, period: str) -> list[StrategyRun]:
    """The newest completed run of each cost scenario, oldest scenario name order kept."""
    runs = session.execute(
        select(StrategyRun)
        .where(
            StrategyRun.strategy == strategy,
            StrategyRun.period == period,
            StrategyRun.status == "completed",
        )
        .order_by(StrategyRun.id.desc())
    ).scalars()
    newest: dict[str, StrategyRun] = {}
    for run in runs:
        newest.setdefault(run.cost_scenario, run)
    return sorted(newest.values(), key=lambda run: run.slippage_bps)


# =============================================================================
# Computations (pure functions over plain values)
# =============================================================================


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def trade_statistics(trades: Sequence[TradeView], *, extremes: int = 5) -> TradeStatistics:
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl < 0]
    gross_profit = math.fsum(trade.net_pnl for trade in wins)
    gross_loss = math.fsum(trade.net_pnl for trade in losses)  # negative or zero
    exits: dict[str, int] = defaultdict(int)
    for trade in trades:
        exits[trade.exit_reason] += 1

    streak = longest = 0
    for trade in sorted(trades, key=lambda t: (t.exit_session, t.trade_id)):
        streak = streak + 1 if trade.net_pnl < 0 else 0
        longest = max(longest, streak)

    def mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    average_win = mean([trade.net_pnl for trade in wins])
    average_loss = mean([trade.net_pnl for trade in losses])
    held = [trade.sessions_held for trade in trades]
    return TradeStatistics(
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        breakeven=len(trades) - len(wins) - len(losses),
        win_rate=_ratio(len(wins), len(trades)),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=math.fsum(trade.net_pnl for trade in trades),
        profit_factor=gross_profit / -gross_loss if gross_loss < 0 else None,
        average_win=average_win,
        average_loss=average_loss,
        payoff_ratio=(
            average_win / -average_loss if average_win is not None and average_loss else None
        ),
        expectancy=mean([trade.net_pnl for trade in trades]),
        expectancy_r=mean([trade.r_multiple for trade in trades]),
        average_win_r=mean([trade.r_multiple for trade in wins]),
        average_loss_r=mean([trade.r_multiple for trade in losses]),
        total_slippage=math.fsum(trade.slippage_cost for trade in trades),
        average_sessions_held=mean([float(value) for value in held]),
        median_sessions_held=float(statistics.median(held)) if held else None,
        average_sessions_held_wins=mean([float(trade.sessions_held) for trade in wins]),
        average_sessions_held_losses=mean([float(trade.sessions_held) for trade in losses]),
        largest_wins=sorted(wins, key=lambda t: (-t.net_pnl, t.trade_id))[:extremes],
        largest_losses=sorted(losses, key=lambda t: (t.net_pnl, t.trade_id))[:extremes],
        exits_by_reason=dict(sorted(exits.items())),
        max_consecutive_losses=longest,
    )


def drawdown(starting_equity: float, curve: Sequence[EquityView]) -> Drawdown:
    peak, peak_index = starting_equity, -1  # -1: the starting equity, before the first session
    worst = 0.0
    worst_peak_index, worst_peak, worst_trough_index = -1, starting_equity, None
    for index, point in enumerate(curve):
        if point.equity > peak:
            peak, peak_index = point.equity, index
        fall = (peak - point.equity) / peak
        if fall > worst:
            worst, worst_peak_index, worst_peak, worst_trough_index = fall, peak_index, peak, index

    if worst_trough_index is None:
        return Drawdown(
            0.0, 0.0, None, starting_equity, None, starting_equity, None, None, None, None, True
        )

    recovery_index = next(
        (i for i in range(worst_trough_index + 1, len(curve)) if curve[i].equity >= worst_peak),
        None,
    )
    end = recovery_index if recovery_index is not None else len(curve) - 1
    return Drawdown(
        max_drawdown=worst,
        max_drawdown_amount=worst_peak - curve[worst_trough_index].equity,
        peak_session=curve[worst_peak_index].session if worst_peak_index >= 0 else None,
        peak_equity=worst_peak,
        trough_session=curve[worst_trough_index].session,
        trough_equity=curve[worst_trough_index].equity,
        recovery_session=curve[recovery_index].session if recovery_index is not None else None,
        sessions_peak_to_trough=worst_trough_index - worst_peak_index,
        sessions_trough_to_recovery=(
            recovery_index - worst_trough_index if recovery_index is not None else None
        ),
        sessions_underwater=end - worst_peak_index,
        recovered=recovery_index is not None,
    )


def symbol_breakdown(
    trades: Sequence[TradeView], starting_equity: float, symbols: Sequence[str]
) -> list[SymbolBreakdown]:
    by_symbol: dict[str, list[TradeView]] = defaultdict(list)
    for trade in trades:
        by_symbol[trade.symbol].append(trade)
    total_net = math.fsum(trade.net_pnl for trade in trades)
    total_gross = math.fsum(trade.net_pnl for trade in trades if trade.net_pnl > 0)
    rows = []
    for symbol in sorted(set(symbols) | set(by_symbol)):
        items = by_symbol.get(symbol, [])
        net = math.fsum(trade.net_pnl for trade in items)
        gross = math.fsum(trade.net_pnl for trade in items if trade.net_pnl > 0)
        rows.append(
            SymbolBreakdown(
                symbol=symbol,
                trades=len(items),
                wins=sum(1 for trade in items if trade.net_pnl > 0),
                net_profit=net,
                gross_profit=gross,
                gross_loss=math.fsum(trade.net_pnl for trade in items if trade.net_pnl < 0),
                strategy_return=net / starting_equity,
                # Shares of a net total are only meaningful when the total is positive.
                share_of_net_profit=net / total_net if total_net > 0 else None,
                share_of_gross_profit=gross / total_gross if total_gross > 0 else None,
            )
        )
    return sorted(rows, key=lambda row: (-row.net_profit, row.symbol))


def concentration(trades: Sequence[TradeView], rows: Sequence[SymbolBreakdown]) -> Concentration:
    """How much of the result rests on one symbol or a few trades.

    "Heavily dependent" is this audit's own convention, stated so it can be
    disputed: the result is flagged when removing the best symbol, or the best
    single trade, would leave a total net profit of zero or less, or when one
    symbol supplies more than half of the net profit.

    Removing a symbol or trade here only subtracts its recorded result. A real
    run without it would have used the freed cash and slots differently, so
    this is an indication, not a substitute for a simulation.
    """
    total = math.fsum(trade.net_pnl for trade in trades)
    winners = sum(1 for row in rows if row.net_profit > 0)
    losers = sum(1 for row in rows if row.net_profit < 0)
    if not trades:
        return Concentration(None, None, None, None, None, None, 0, 0, None, "no trades")
    if total <= 0:
        return Concentration(
            rows[0].symbol if rows else None,
            None,
            None,
            None,
            None,
            None,
            winners,
            losers,
            None,
            "the run made no net profit, so there is no profit to depend on",
        )

    best = rows[0]
    ranked = sorted(trades, key=lambda t: (-t.net_pnl, t.trade_id))
    best_trade = ranked[0].net_pnl
    top_five = math.fsum(trade.net_pnl for trade in ranked[:5])
    without_symbol, without_trade = total - best.net_profit, total - best_trade
    reasons = []
    if without_symbol <= 0:
        reasons.append(f"without {best.symbol} the net profit would be {without_symbol:,.2f}")
    if without_trade <= 0:
        reasons.append(
            f"without the single best trade the net profit would be {without_trade:,.2f}"
        )
    if best.net_profit / total > 0.5:
        reasons.append(f"{best.symbol} supplies {best.net_profit / total:.0%} of the net profit")
    return Concentration(
        best_symbol=best.symbol,
        best_symbol_share_of_net_profit=best.net_profit / total,
        net_profit_without_best_symbol=without_symbol,
        best_trade_share_of_net_profit=best_trade / total,
        net_profit_without_best_trade=without_trade,
        top_five_trades_share_of_net_profit=top_five / total,
        symbols_with_net_profit=winners,
        symbols_with_net_loss=losers,
        heavily_dependent=bool(reasons),
        reason="; ".join(reasons) if reasons else "no single symbol or trade carries the result",
    )


def risk_adjusted(starting_equity: float, curve: Sequence[EquityView], first: date, last: date):
    """Annualised return, volatility and Sharpe ratio (risk-free rate taken as zero)."""
    if not curve:
        return None, None, None
    ending = curve[-1].equity
    days = (last - first).days + 1
    annual = (ending / starting_equity) ** (365.25 / days) - 1.0 if days > 0 else None
    previous = [starting_equity] + [point.equity for point in curve[:-1]]
    returns = [point.equity / before - 1.0 for point, before in zip(curve, previous, strict=True)]
    if len(returns) < 2:
        return annual, None, None
    deviation = statistics.stdev(returns)
    volatility = deviation * math.sqrt(SESSIONS_PER_YEAR)
    sharpe = (
        statistics.fmean(returns) / deviation * math.sqrt(SESSIONS_PER_YEAR) if deviation else None
    )
    return annual, volatility, sharpe


def check_integrity(
    run, signals, trades, curve: Sequence[EquityView], expected_sessions: int | None
):
    problems: list[str] = []
    taken = {signal.id for signal in signals if signal.decision == "taken"}
    trade_signals = [trade.signal_id for trade in trades]
    skipped: dict[str, int] = defaultdict(int)
    for signal in signals:
        if signal.decision != "taken":
            skipped[signal.reason or "no reason recorded"] += 1
            if not signal.reason:
                problems.append(f"skipped signal {signal.id} has no reason")

    every_taken = sorted(trade_signals) == sorted(taken) and len(set(trade_signals)) == len(
        trade_signals
    )
    none_extra = set(trade_signals) <= taken
    if not every_taken:
        problems.append("taken signals and trades do not match one to one")
    if not none_extra:
        problems.append("a trade refers to a signal that was not taken")
    if len(trades) != run.trades_total:
        problems.append(f"{len(trades)} trades stored but the run header says {run.trades_total}")

    expected_end = run.starting_equity + math.fsum(trade.net_pnl for trade in trades)
    matches = run.ending_equity is not None and math.isclose(
        run.ending_equity, expected_end, abs_tol=0.01
    )
    if not matches:
        problems.append(
            f"ending equity {run.ending_equity} differs from starting equity plus all net results "
            f"({expected_end:.2f})"
        )
    complete = bool(curve) and (expected_sessions is None or len(curve) == expected_sessions)
    if curve and not math.isclose(curve[-1].equity, run.ending_equity or 0.0, abs_tol=0.01):
        complete = False
        problems.append("the last equity point does not equal the run's ending equity")
    if expected_sessions is not None and len(curve) != expected_sessions:
        problems.append(f"equity curve has {len(curve)} points, expected {expected_sessions}")

    identity = all(
        math.isclose(trade.net_pnl, trade.gross_pnl - trade.slippage_cost, abs_tol=1e-6)
        and trade.entry_session <= trade.exit_session
        and run.period_start <= trade.entry_session
        and trade.exit_session <= run.period_end
        for trade in trades
    )
    if not identity:
        problems.append("a trade's net result, dates or period bounds do not add up")

    return Integrity(
        signals_total=len(signals),
        signals_taken=len(taken),
        signals_skipped=len(signals) - len(taken),
        skipped_by_reason=dict(sorted(skipped.items())),
        trades_stored=len(trades),
        trades_in_run_header=run.trades_total,
        losing_trades_stored=sum(1 for trade in trades if trade.net_pnl < 0),
        every_taken_signal_has_one_trade=every_taken,
        no_trade_without_a_taken_signal=none_extra,
        ending_equity_matches_trades=matches,
        equity_curve_complete=complete,
        trade_identity_holds=identity,
        problems=problems,
    )


def audit_run(session: Session, run_id: int, *, expected_sessions: int | None = None) -> RunAudit:
    run, signals, stored_trades, stored_equity = load_run(session, run_id)
    trades = [
        TradeView(
            trade_id=t.id,
            symbol=t.symbol,
            entry_session=t.entry_session,
            exit_session=t.exit_session,
            shares=t.shares,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            exit_reason=t.exit_reason,
            sessions_held=t.sessions_held,
            gross_pnl=t.gross_pnl,
            slippage_cost=t.slippage_cost,
            net_pnl=t.net_pnl,
            risk_amount=t.risk_amount,
            r_multiple=t.r_multiple,
        )
        for t in stored_trades
    ]
    curve = [EquityView(p.session, p.equity, p.open_positions) for p in stored_equity]
    ending = run.ending_equity if run.ending_equity is not None else run.starting_equity
    annual, volatility, sharpe = risk_adjusted(
        run.starting_equity, curve, run.period_start, run.period_end
    )
    rows = symbol_breakdown(trades, run.starting_equity, run.symbols)
    return RunAudit(
        run_id=run.id,
        strategy=run.strategy,
        strategy_version=run.strategy_version,
        period=run.period,
        period_start=run.period_start,
        period_end=run.period_end,
        cost_scenario=run.cost_scenario,
        slippage_bps=run.slippage_bps,
        parameters_hash=run.parameters_hash,
        data_fingerprint=run.data_fingerprint,
        result_fingerprint=run.result_fingerprint,
        starting_equity=run.starting_equity,
        ending_equity=ending,
        total_return=ending / run.starting_equity - 1.0,
        annualised_return=annual,
        annualised_volatility=volatility,
        sharpe_ratio=sharpe,
        sessions=len(curve),
        exposure=_ratio(sum(1 for p in curve if p.open_positions > 0), len(curve)),
        average_open_positions=statistics.fmean(p.open_positions for p in curve) if curve else None,
        statistics=trade_statistics(trades),
        drawdown=drawdown(run.starting_equity, curve),
        symbols=rows,
        concentration=concentration(trades, rows),
        integrity=check_integrity(run, signals, stored_trades, curve, expected_sessions),
        trades=trades,
    )


def buy_and_hold(data: DailyData, first: date, last: date, slippage_bps: float) -> list[BuyAndHold]:
    """Buy at the first session's open, sell at the last session's close, per symbol."""
    results = []
    for symbol, bars in sorted(data.bars.items()):
        window = [bar for bar in bars if first <= bar.session <= last]
        if not window:
            continue
        entry, exit_ = window[0].open, window[-1].close
        peak, worst = entry, 0.0
        for bar in window:
            peak = max(peak, bar.close)
            worst = max(worst, (peak - bar.close) / peak)
        results.append(
            BuyAndHold(
                symbol=symbol,
                first_session=window[0].session,
                last_session=window[-1].session,
                entry_price=entry,
                exit_price=exit_,
                total_return=exit_ / entry - 1.0,
                total_return_after_costs=(
                    sell_fill(exit_, slippage_bps) / buy_fill(entry, slippage_bps) - 1.0
                ),
                max_drawdown=worst,
            )
        )
    return results


def trade_overlap(audits: Sequence[RunAudit]) -> dict[str, Any]:
    """How many trades the runs share (same symbol, entry session and exit session)."""
    keys = [{(t.symbol, t.entry_session, t.exit_session) for t in audit.trades} for audit in audits]
    common = set.intersection(*keys) if keys else set()
    return {
        "common_to_all": len(common),
        "per_run": [len(k) for k in keys],
        "only_in": [len(k - common) for k in keys],
    }
