"""Daily, long-only simulation of ONE strategy with its own virtual account.

This is bookkeeping over historical bars. It places no orders and talks to no
broker. Each call simulates a single strategy; two strategies never share cash,
positions or slots.

What happens on each session ``s`` of the period, in this order:

1. At the open
   a. Positions flagged for exit at the previous close are sold at the open.
   b. A position whose stop is at or above the open (a gap through the stop)
      is sold at the open.
   c. Entries planned at the previous close are bought at the open, strongest
      signal first, unless the open is above the plan's limit price.
2. During the session
   A position whose stop lies at or above the session's low is sold at the
   stop price. This includes a position opened at this session's open.
3. At the close
   The account is valued. The strategy's plans for session ``s`` are read:
   exit signals flag positions to be sold at the next open, and entry signals
   are recorded, ranked, sized and planned for the next open.

Because step 3 uses only the plans of session ``s`` (built from bars up to and
including ``s``), and step 1 of the next session uses only its open, no
decision ever depends on a price that was not yet known.

Conservative conventions:

* Sizes are fixed at the close, from the equity and prices known then. If the
  cash at the open does not cover the planned shares, fewer are bought.
* A daily bar does not say whether its low or its high came first. A stop
  inside the bar's range is always assumed to have been hit.
* Slippage always works against the account: buys fill higher and sells fill
  lower than the reference price.
* On the last session of the period every open position is sold at the close
  and marked ``period_end``, so a run never carries a position out of its
  period. These exits are forced by the test design, not by the rules.

Every signal is recorded, including those that were not traded, each with the
reason. Every trade is recorded, winning or losing.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Protocol

from app.strategies.assumptions import buy_fill, sell_fill
from app.strategies.data import DailyBar

SIGNAL_ENTRY = "entry"

DECISION_TAKEN = "taken"
DECISION_SKIPPED = "skipped"

# Why a signal was not traded.
REASON_ALREADY_IN_POSITION = "already_in_position"
REASON_NO_FREE_SLOT = "no_free_slot"
REASON_POSITION_TOO_SMALL = "position_too_small"
REASON_PERIOD_ENDS = "period_ends"
REASON_NO_BAR = "no_bar_on_entry_session"
REASON_GAP_ABOVE_LIMIT = "open_above_entry_limit"
REASON_INSUFFICIENT_CASH = "insufficient_cash"

# Why a position was closed.
EXIT_STOP = "stop"
EXIT_STOP_GAP = "stop_gap"
EXIT_SIGNAL = "exit_signal"
EXIT_PERIOD_END = "period_end"


@dataclass(frozen=True)
class DayPlan:
    """What a strategy concluded from one symbol's bars up to and including ``session``."""

    session: date
    entry_signal: bool
    exit_signal: bool
    #: Entry fill minus this is the stop. Set when ``entry_signal`` is true.
    stop_distance: float | None = None
    #: The entry is not filled when the next open is above this price.
    max_entry_price: float | None = None
    #: Orders competing signals of one session: larger goes first.
    rank_value: float | None = None
    #: The values the rule compared, kept with the signal.
    indicators: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SizingRules:
    #: Share of equity lost if a position is stopped out at its initial stop.
    risk_fraction: float
    #: Largest position, as a share of equity.
    max_position_fraction: float
    max_positions: int


class DailyStrategy(Protocol):
    name: str
    version: str

    @property
    def sizing(self) -> SizingRules: ...

    def build_plans(self, bars: list[DailyBar]) -> list[DayPlan]: ...


@dataclass
class SignalRecord:
    symbol: str
    signal_session: date
    signal_type: str
    close: float
    indicators: dict[str, Any]
    rank_value: float | None
    rank: int | None = None
    decision: str = DECISION_SKIPPED
    reason: str | None = None
    planned_shares: int | None = None


@dataclass
class TradeRecord:
    symbol: str
    #: Position of the opening signal in ``SimulationResult.signals``.
    signal_index: int
    shares: int
    entry_session: date
    entry_reference_price: float
    entry_price: float
    equity_at_signal: float
    initial_stop: float
    risk_amount: float
    exit_session: date
    exit_reference_price: float
    exit_price: float
    exit_reason: str
    sessions_held: int
    gross_pnl: float
    slippage_cost: float
    net_pnl: float
    r_multiple: float


@dataclass(frozen=True)
class EquityRecord:
    session: date
    cash: float
    positions_value: float
    equity: float
    open_positions: int


@dataclass(frozen=True)
class SimulationResult:
    signals: list[SignalRecord]
    trades: list[TradeRecord]
    equity: list[EquityRecord]
    starting_equity: float
    ending_equity: float

    def fingerprint(self) -> str:
        """SHA-256 of every signal, trade and equity point, in order."""
        payload = {
            "signals": [asdict(signal) for signal in self.signals],
            "trades": [asdict(trade) for trade in self.trades],
            "equity": [asdict(point) for point in self.equity],
        }
        text = json.dumps(payload, sort_keys=True, default=lambda value: value.isoformat())
        return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class _Position:
    symbol: str
    signal_index: int
    shares: int
    entry_session: date
    entry_session_number: int
    entry_reference_price: float
    entry_price: float
    equity_at_signal: float
    stop: float
    last_close: float
    exit_pending: bool = False


@dataclass(frozen=True)
class _PlannedEntry:
    symbol: str
    signal_index: int
    shares: int
    stop_distance: float
    max_entry_price: float | None
    equity_at_signal: float


def simulate(
    strategy: DailyStrategy,
    bars: Mapping[str, list[DailyBar]],
    *,
    period_start: date,
    period_end: date,
    starting_equity: float,
    slippage_bps: float,
) -> SimulationResult:
    """Simulate ``strategy`` alone over ``bars`` between two sessions (inclusive).

    Bars before ``period_start`` are used only as history for the indicators.
    Bars after ``period_end`` are never read.
    """
    if period_start > period_end:
        raise ValueError("period_start must not be after period_end")
    if not starting_equity > 0:
        raise ValueError("starting_equity must be greater than 0")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must not be negative")

    sizing = strategy.sizing
    usable = {
        symbol: [bar for bar in items if bar.session <= period_end]
        for symbol, items in sorted(bars.items())
    }
    bar_on: dict[str, dict[date, DailyBar]] = {}
    plan_on: dict[str, dict[date, DayPlan]] = {}
    for symbol, items in usable.items():
        plans = strategy.build_plans(items)
        if len(plans) != len(items) or any(
            plan.session != bar.session for plan, bar in zip(plans, items, strict=True)
        ):
            raise ValueError(
                f"{strategy.name} returned plans that do not match the bars of {symbol}"
            )
        bar_on[symbol] = {bar.session: bar for bar in items}
        plan_on[symbol] = {plan.session: plan for plan in plans}

    sessions = sorted(
        {bar.session for items in usable.values() for bar in items if bar.session >= period_start}
    )

    cash = starting_equity
    positions: dict[str, _Position] = {}
    planned: list[_PlannedEntry] = []
    signals: list[SignalRecord] = []
    trades: list[TradeRecord] = []
    equity_curve: list[EquityRecord] = []

    def close_position(
        position: _Position, session: date, number: int, reference: float, reason: str
    ) -> None:
        nonlocal cash
        fill = sell_fill(reference, slippage_bps)
        cash += position.shares * fill
        gross = position.shares * (reference - position.entry_reference_price)
        net = position.shares * (fill - position.entry_price)
        risk = position.shares * (position.entry_price - position.stop)
        trades.append(
            TradeRecord(
                symbol=position.symbol,
                signal_index=position.signal_index,
                shares=position.shares,
                entry_session=position.entry_session,
                entry_reference_price=position.entry_reference_price,
                entry_price=position.entry_price,
                equity_at_signal=position.equity_at_signal,
                initial_stop=position.stop,
                risk_amount=risk,
                exit_session=session,
                exit_reference_price=reference,
                exit_price=fill,
                exit_reason=reason,
                sessions_held=number - position.entry_session_number,
                gross_pnl=gross,
                slippage_cost=gross - net,
                net_pnl=net,
                r_multiple=net / risk,
            )
        )
        del positions[position.symbol]

    for number, session in enumerate(sessions):
        is_last = number == len(sessions) - 1

        # 1a and 1b. Exits at the open.
        for symbol in sorted(positions):
            position, bar = positions[symbol], bar_on[symbol].get(session)
            if bar is None:
                continue  # no bar today: nothing can be traded, the position waits
            if position.exit_pending:
                close_position(position, session, number, bar.open, EXIT_SIGNAL)
            elif bar.open <= position.stop:
                close_position(position, session, number, bar.open, EXIT_STOP_GAP)

        # 1c. Entries planned at the previous close.
        for entry in planned:
            signal, bar = signals[entry.signal_index], bar_on[entry.symbol].get(session)
            if bar is None:
                signal.reason = REASON_NO_BAR
                continue
            if len(positions) >= sizing.max_positions:
                # A position that should have left at this open could not (no bar for it).
                signal.reason = REASON_NO_FREE_SLOT
                continue
            if entry.max_entry_price is not None and bar.open > entry.max_entry_price:
                signal.reason = REASON_GAP_ABOVE_LIMIT
                continue
            fill = buy_fill(bar.open, slippage_bps)
            shares = min(entry.shares, math.floor(cash / fill))
            if shares < 1:
                signal.reason = REASON_INSUFFICIENT_CASH
                continue
            cash -= shares * fill
            signal.decision, signal.reason = DECISION_TAKEN, None
            positions[entry.symbol] = _Position(
                symbol=entry.symbol,
                signal_index=entry.signal_index,
                shares=shares,
                entry_session=session,
                entry_session_number=number,
                entry_reference_price=bar.open,
                entry_price=fill,
                equity_at_signal=entry.equity_at_signal,
                stop=fill - entry.stop_distance,
                last_close=bar.open,
            )
        planned = []

        # 2. Stops inside the session.
        for symbol in sorted(positions):
            position, bar = positions[symbol], bar_on[symbol].get(session)
            if bar is not None and bar.low <= position.stop:
                close_position(position, session, number, position.stop, EXIT_STOP)

        # The period ends here: nothing is carried out of it.
        if is_last:
            for symbol in sorted(positions):
                position, bar = positions[symbol], bar_on[symbol].get(session)
                reference = bar.close if bar is not None else position.last_close
                close_position(position, session, number, reference, EXIT_PERIOD_END)

        # 3. Value the account at the close.
        for symbol, position in positions.items():
            bar = bar_on[symbol].get(session)
            if bar is not None:
                position.last_close = bar.close
        positions_value = math.fsum(
            position.shares * position.last_close for position in positions.values()
        )
        equity = cash + positions_value
        equity_curve.append(EquityRecord(session, cash, positions_value, equity, len(positions)))

        # 3. Decisions from the plans of this session, for the next open.
        for symbol, position in positions.items():
            plan = plan_on[symbol].get(session)
            if plan is not None and plan.exit_signal:
                position.exit_pending = True

        candidates: list[int] = []
        for symbol in usable:
            plan = plan_on[symbol].get(session)
            if plan is None or not plan.entry_signal:
                continue
            signals.append(
                SignalRecord(
                    symbol=symbol,
                    signal_session=session,
                    signal_type=SIGNAL_ENTRY,
                    close=bar_on[symbol][session].close,
                    indicators=dict(plan.indicators),
                    rank_value=plan.rank_value,
                )
            )
            index = len(signals) - 1
            if symbol in positions:
                signals[index].reason = REASON_ALREADY_IN_POSITION
            elif is_last:
                signals[index].reason = REASON_PERIOD_ENDS
            else:
                candidates.append(index)

        # Strongest first; the symbol breaks ties so the order is always the same.
        candidates.sort(
            key=lambda index: (
                -(
                    signals[index].rank_value
                    if signals[index].rank_value is not None
                    else -math.inf
                ),
                signals[index].symbol,
            )
        )
        free_slots = sizing.max_positions - sum(
            1 for position in positions.values() if not position.exit_pending
        )
        for rank, index in enumerate(candidates, start=1):
            signal = signals[index]
            signal.rank = rank
            plan = plan_on[signal.symbol][session]
            if plan.stop_distance is None or not plan.stop_distance > 0:
                raise ValueError(
                    f"{strategy.name} gave an entry signal for {signal.symbol} on "
                    f"{session.isoformat()} without a positive stop distance"
                )
            if free_slots < 1:
                signal.reason = REASON_NO_FREE_SLOT
                continue
            shares = min(
                math.floor(sizing.risk_fraction * equity / plan.stop_distance),
                math.floor(sizing.max_position_fraction * equity / signal.close),
            )
            signal.planned_shares = shares
            if shares < 1:
                signal.reason = REASON_POSITION_TOO_SMALL
                continue
            free_slots -= 1
            planned.append(
                _PlannedEntry(
                    symbol=signal.symbol,
                    signal_index=index,
                    shares=shares,
                    stop_distance=plan.stop_distance,
                    max_entry_price=plan.max_entry_price,
                    equity_at_signal=equity,
                )
            )

    return SimulationResult(
        signals=signals,
        trades=trades,
        equity=equity_curve,
        starting_equity=starting_equity,
        ending_equity=equity_curve[-1].equity if equity_curve else starting_equity,
    )
