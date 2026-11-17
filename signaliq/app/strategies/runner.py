"""Run one strategy over one period and record everything in the database.

A run is a simulation. It reads stored bars, writes ``strategy_*`` rows and
never changes market data. One run holds one strategy: results of different
strategies are never added together here.

Protecting the out-of-sample period
-----------------------------------
Bars from ``OUT_OF_SAMPLE_START`` on are reserved for a single, honest check.
Two rules enforce that in code:

* A strategy's parameters must be frozen (``freeze_parameters``) before any run
  may touch that period, and such a run must use exactly the frozen parameters
  and the same version of the rules.
* Frozen parameters cannot be replaced. Changing them would mean tuning
  against data that has already been looked at.

Runs on the in-sample period are unrestricted and read no later bar.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.database import session_scope
from app.models import (
    StrategyEquityPoint,
    StrategyParameterFreeze,
    StrategyRun,
    StrategyRunStatus,
    StrategySignal,
    StrategyTrade,
    utc_now,
)
from app.strategies.assumptions import (
    DAILY_COST_SCENARIOS,
    OUT_OF_SAMPLE_START,
    STARTING_EQUITY,
    Period,
)
from app.strategies.data import DAILY_TIMEFRAME, DailyData, load_daily_bars
from app.strategies.momentum_breakout import (
    STRATEGY_NAME as MOMENTUM_BREAKOUT,
)
from app.strategies.momentum_breakout import (
    MomentumBreakout,
    MomentumBreakoutParameters,
)
from app.strategies.simulator import DailyStrategy, SimulationResult, simulate

#: Strategies that can be run. Each is simulated on its own; there is no
#: command, function or table that combines two of them.
STRATEGIES = {
    MOMENTUM_BREAKOUT: (MomentumBreakout, MomentumBreakoutParameters),
}


class StrategyRunError(ValueError):
    """The run was refused before anything was simulated."""


@dataclass(frozen=True)
class RunSummary:
    run_id: int
    strategy: str
    period: str
    period_start: date
    period_end: date
    cost_scenario: str
    slippage_bps: float
    starting_equity: float
    ending_equity: float
    signals_total: int
    signals_taken: int
    signals_skipped_by_reason: dict[str, int]
    trades_total: int
    trades_by_exit_reason: dict[str, int]
    bars_used: int
    excluded_sessions: dict[str, list[date]]
    data_fingerprint: str
    result_fingerprint: str


def parameters_hash(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def build_strategy(name: str, overrides: dict[str, str] | None = None) -> DailyStrategy:
    """Create a strategy by name, optionally changing parameters from ``name=value`` text."""
    if name not in STRATEGIES:
        raise StrategyRunError(
            f"Unknown strategy {name!r}. Available: {', '.join(sorted(STRATEGIES))}."
        )
    strategy_class, parameters_class = STRATEGIES[name]
    try:
        return strategy_class(parameters_class.from_overrides(overrides or {}))
    except ValueError as exc:
        raise StrategyRunError(f"Invalid parameters for {name}: {exc}") from None


def get_freeze(session: Session, strategy: str) -> StrategyParameterFreeze | None:
    return session.get(StrategyParameterFreeze, strategy)


def freeze_parameters(session: Session, strategy: DailyStrategy) -> StrategyParameterFreeze:
    """Fix a strategy's parameters ahead of its out-of-sample check. Can be done once.

    Freezing the same parameters again changes nothing. Freezing different
    ones is refused.
    """
    parameters = strategy.parameters.to_dict()
    digest = parameters_hash(parameters)
    existing = get_freeze(session, strategy.name)
    if existing is not None:
        if (existing.parameters_hash, existing.strategy_version) == (digest, strategy.version):
            return existing
        raise StrategyRunError(
            f"The parameters of {strategy.name} were already frozen on "
            f"{existing.frozen_at.date().isoformat()} and cannot be replaced: changing them now "
            "would be tuning against the out-of-sample period."
        )
    freeze = StrategyParameterFreeze(
        strategy=strategy.name,
        strategy_version=strategy.version,
        parameters=parameters,
        parameters_hash=digest,
    )
    session.add(freeze)
    session.flush()
    return freeze


def _check_out_of_sample_access(session: Session, strategy: DailyStrategy, period: Period) -> None:
    if not period.touches_out_of_sample():
        return
    freeze = get_freeze(session, strategy.name)
    reserved = f"The period from {OUT_OF_SAMPLE_START.isoformat()} on is reserved"
    if freeze is None:
        raise StrategyRunError(
            f"{reserved}: freeze the parameters of {strategy.name} first, then run it once "
            "with exactly those parameters."
        )
    used = parameters_hash(strategy.parameters.to_dict())
    if (freeze.parameters_hash, freeze.strategy_version) != (used, strategy.version):
        raise StrategyRunError(
            f"{reserved} and may only be run with the frozen parameters of {strategy.name} "
            f"(frozen on {freeze.frozen_at.date().isoformat()}). This run uses different "
            "parameters or a different version of the rules."
        )


def run_strategy(
    session_factory: sessionmaker[Session],
    settings: Settings,
    strategy: DailyStrategy,
    period: Period,
    *,
    cost_scenario: str = "base",
    symbols: Sequence[str] | None = None,
    starting_equity: float = STARTING_EQUITY,
) -> RunSummary:
    """Simulate ``strategy`` alone over ``period`` and store the run."""
    if cost_scenario not in DAILY_COST_SCENARIOS:
        raise StrategyRunError(
            f"Unknown cost scenario {cost_scenario!r}. "
            f"Available: {', '.join(DAILY_COST_SCENARIOS)}."
        )
    slippage_bps = DAILY_COST_SCENARIOS[cost_scenario]
    symbols = list(symbols or settings.symbols)
    unknown = [symbol for symbol in symbols if symbol not in settings.symbols]
    if unknown:
        raise StrategyRunError(f"Not in the configured universe: {', '.join(unknown)}.")

    with session_scope(session_factory) as session:
        _check_out_of_sample_access(session, strategy, period)
        data = load_daily_bars(session, settings, symbols, end=period.end)

    last_session = data.last_session
    if last_session is None or last_session < period.start:
        raise StrategyRunError(
            f"No stored daily bars on or after {period.start.isoformat()} for the requested "
            "symbols. Download them with scripts/ingest.py first."
        )
    period_end = min(period.end, last_session) if period.end is not None else last_session
    parameters = strategy.parameters.to_dict()

    result = simulate(
        strategy,
        data.bars,
        period_start=period.start,
        period_end=period_end,
        starting_equity=starting_equity,
        slippage_bps=slippage_bps,
    )
    fingerprint = result.fingerprint()

    with session_scope(session_factory) as session:
        run = StrategyRun(
            status=StrategyRunStatus.COMPLETED,
            finished_at=utc_now(),
            strategy=strategy.name,
            strategy_version=strategy.version,
            period=period.name,
            period_start=period.start,
            period_end=period_end,
            symbols=symbols,
            parameters=parameters,
            parameters_hash=parameters_hash(parameters),
            starting_equity=starting_equity,
            cost_scenario=cost_scenario,
            slippage_bps=slippage_bps,
            timeframe=DAILY_TIMEFRAME,
            source=data.source,
            feed=data.feed,
            adjustment=data.adjustment,
            bars_used=data.bar_count,
            data_first_session=data.first_session,
            data_last_session=last_session,
            data_fingerprint=data.fingerprint,
            result_fingerprint=fingerprint,
            signals_total=len(result.signals),
            signals_taken=sum(1 for signal in result.signals if signal.decision == "taken"),
            trades_total=len(result.trades),
            ending_equity=result.ending_equity,
        )
        session.add(run)
        session.flush()
        run_id = run.id
        _store_result(session, run_id, strategy.name, result)

    return _summary(run_id, strategy, period.name, period.start, period_end, cost_scenario,
                    slippage_bps, starting_equity, data, result, fingerprint)  # fmt: skip


def _store_result(session: Session, run_id: int, strategy: str, result: SimulationResult) -> None:
    signal_ids: list[int] = []
    for signal in result.signals:
        row = StrategySignal(
            run_id=run_id,
            strategy=strategy,
            symbol=signal.symbol,
            signal_session=signal.signal_session,
            signal_type=signal.signal_type,
            close=signal.close,
            indicators=signal.indicators,
            rank_value=signal.rank_value,
            rank=signal.rank,
            decision=signal.decision,
            reason=signal.reason,
            planned_shares=signal.planned_shares,
        )
        session.add(row)
        session.flush()
        signal_ids.append(row.id)

    session.add_all(
        StrategyTrade(
            run_id=run_id,
            signal_id=signal_ids[trade.signal_index],
            strategy=strategy,
            symbol=trade.symbol,
            shares=trade.shares,
            entry_session=trade.entry_session,
            entry_reference_price=trade.entry_reference_price,
            entry_price=trade.entry_price,
            equity_at_signal=trade.equity_at_signal,
            initial_stop=trade.initial_stop,
            risk_amount=trade.risk_amount,
            exit_session=trade.exit_session,
            exit_reference_price=trade.exit_reference_price,
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
            sessions_held=trade.sessions_held,
            gross_pnl=trade.gross_pnl,
            slippage_cost=trade.slippage_cost,
            net_pnl=trade.net_pnl,
            r_multiple=trade.r_multiple,
        )
        for trade in result.trades
    )
    if result.equity:
        session.execute(
            insert(StrategyEquityPoint),
            [
                {
                    "run_id": run_id,
                    "session": point.session,
                    "cash": point.cash,
                    "positions_value": point.positions_value,
                    "equity": point.equity,
                    "open_positions": point.open_positions,
                }
                for point in result.equity
            ],
        )


def _summary(
    run_id: int,
    strategy: DailyStrategy,
    period: str,
    period_start: date,
    period_end: date,
    cost_scenario: str,
    slippage_bps: float,
    starting_equity: float,
    data: DailyData,
    result: SimulationResult,
    fingerprint: str,
) -> RunSummary:
    skipped: dict[str, int] = {}
    for signal in result.signals:
        if signal.decision != "taken":
            skipped[signal.reason or "unknown"] = skipped.get(signal.reason or "unknown", 0) + 1
    exits: dict[str, int] = {}
    for trade in result.trades:
        exits[trade.exit_reason] = exits.get(trade.exit_reason, 0) + 1
    return RunSummary(
        run_id=run_id,
        strategy=strategy.name,
        period=period,
        period_start=period_start,
        period_end=period_end,
        cost_scenario=cost_scenario,
        slippage_bps=slippage_bps,
        starting_equity=starting_equity,
        ending_equity=result.ending_equity,
        signals_total=len(result.signals),
        signals_taken=sum(1 for signal in result.signals if signal.decision == "taken"),
        signals_skipped_by_reason=dict(sorted(skipped.items())),
        trades_total=len(result.trades),
        trades_by_exit_reason=dict(sorted(exits.items())),
        bars_used=data.bar_count,
        excluded_sessions={symbol: days for symbol, days in data.excluded.items() if days},
        data_fingerprint=data.fingerprint,
        result_fingerprint=fingerprint,
    )


def stored_run_rows(session: Session, run_id: int) -> dict[str, list[Any]]:
    """Read back everything a run stored (used to check reproducibility)."""
    return {
        "signals": list(
            session.execute(
                select(StrategySignal)
                .where(StrategySignal.run_id == run_id)
                .order_by(StrategySignal.id)
            ).scalars()
        ),
        "trades": list(
            session.execute(
                select(StrategyTrade)
                .where(StrategyTrade.run_id == run_id)
                .order_by(StrategyTrade.id)
            ).scalars()
        ),
        "equity": list(
            session.execute(
                select(StrategyEquityPoint)
                .where(StrategyEquityPoint.run_id == run_id)
                .order_by(StrategyEquityPoint.session)
            ).scalars()
        ),
    }
