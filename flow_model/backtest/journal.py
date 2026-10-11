"""The journal: every signal, every order and every trade, recorded.

Section 9 records a signal for EVERY bar, "including WAITs + reasons", and
section 12 asks how often each gate fires. That question is unanswerable after
the fact unless the WAITs were written down at the time, so this module treats
a WAIT as a first-class record rather than as the absence of one.

## Volume, and what is never dropped

`config.backtest.record_wait_signals` governs the ROW volume: a ten-year
five-minute run evaluates on the order of 200k bars, almost all of them WAITs,
and a caller may not want 200k rows in SQLite. What it does NOT govern is the
COUNTS. The per-reason and per-stage histograms are accumulated on every bar
whatever the setting, because a rejection analysis that silently loses its
denominator when a flag is flipped is worse than no rejection analysis: the
remaining numbers still look like answers.

## Cost drag is never one number

`cost_drag_by_exit_reason()` is the only cost summary this module offers, and
that is deliberate. A target is a resting limit -- no spread crossed, no
slippage -- while a stop is a market order that additionally pays
`stop_order_extra_slippage_ticks`, so the two exits do not cost the same and
their average is a figure that describes neither. A reason with no trades is
ABSENT from the mapping rather than present with 0.0, because "no trade has
ever exited this way" and "trades exit this way for free" are different facts.

## n = 0

Phase 6 committed the finding that the system takes zero trades (commit
f952d3e: the highest Flow Score observed is 65.23 against a lowest setup
threshold of 70.0). An empty journal is therefore the EXPECTED result, not an
edge case, and everything here is built so that an empty run reports counts of
zero and sums of 0.0 -- both defined over an empty set -- and no ratio at all.
Win rate, profit factor and expectancy are undefined at n=0 and are not
computed here; they belong to `analytics/`, which must say "undefined, n=0"
rather than "0.0".
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from pydantic import Field

from flow_model.backtest.fill_timing import PendingOrder
from flow_model.backtest.fills import EntryFill, FillStatus
from flow_model.core.contracts import CostBreakdown, EquityPoint, Signal, TradeRecord
from flow_model.core.enums import (
    Component,
    DataQuality,
    Regime,
    SetupType,
    Side,
    SignalAction,
    SplitPhase,
    WaitReason,
)
from flow_model.core.model import FrozenModel
from flow_model.utils.logging import get_logger

__all__ = [
    "Journal",
    "JournalError",
    "OrderRecord",
    "SignalRecord",
]

logger = get_logger("backtest.journal")


class JournalError(RuntimeError):
    """Raised when the journal is handed a record it cannot account for."""


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


class SignalRecord(FrozenModel):
    """One bar's decision, as it is persisted.

    A superset of `Signal` by exactly the fields that make the row useful
    without the run it came from: the run id, the blocking gate's STAGE (the
    signal carries the reason, not the stage), and whether the decision
    actually turned into an order. A WAIT whose reason is `risk_limit` and a
    LONG whose order was never submitted are different outcomes, and section
    12's rejection analysis needs to tell them apart.
    """

    run_id: str
    symbol: str
    ts: datetime
    action: SignalAction
    setup: SetupType | None = None
    flow_score: float | None = None
    confidence: float = 0.0
    regime: Regime = Regime.UNKNOWN
    data_quality: DataQuality = DataQuality.MISSING
    wait_reason: WaitReason | None = None
    wait_detail: str = ""
    blocking_stage: str = ""
    stages_run: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    component_points: dict[Component, float] = Field(default_factory=dict)
    reference_price: float | None = None
    atr: float | None = None
    side: Side | None = None
    intent_built: bool = False
    order_submitted: bool = False
    equity_at_signal: float | None = None
    sizing_reason: str = ""
    limit_breach: str = ""
    split_label: SplitPhase | None = None

    @classmethod
    def from_signal(
        cls,
        signal: Signal,
        *,
        run_id: str,
        blocking_stage: str = "",
        stages_run: tuple[str, ...] = (),
        intent_built: bool = False,
        order_submitted: bool = False,
        equity_at_signal: float | None = None,
        sizing_reason: str = "",
        limit_breach: str = "",
        split_label: SplitPhase | None = None,
    ) -> "SignalRecord":
        return cls(
            run_id=run_id,
            symbol=signal.symbol,
            ts=signal.ts,
            action=signal.action,
            setup=signal.setup,
            flow_score=signal.flow_score,
            confidence=signal.confidence,
            regime=signal.regime,
            data_quality=signal.data_quality,
            wait_reason=signal.wait_reason,
            wait_detail=signal.wait_detail,
            blocking_stage=blocking_stage,
            stages_run=stages_run,
            reasons=signal.reasons,
            component_points=dict(signal.component_points),
            reference_price=signal.reference_price,
            atr=signal.atr,
            side=signal.side,
            intent_built=intent_built,
            order_submitted=order_submitted,
            equity_at_signal=equity_at_signal,
            sizing_reason=sizing_reason,
            limit_breach=limit_breach,
            split_label=split_label,
        )


class OrderRecord(FrozenModel):
    """One entry attempt: filled, partially filled, missed, or still pending.

    Missed entries are recorded for the same reason WAITs are. "The signal
    fired and the limit never traded through" and "no signal fired" produce
    the same empty trade list and are not the same finding, and only one of
    them is fixed by changing a threshold.
    """

    run_id: str
    symbol: str
    signal_ts: datetime
    submitted_ts: datetime
    eligible_ts: datetime
    order_type: str
    side: Side
    status: str
    requested_size: float
    filled_size: float = 0.0
    fill_ts: datetime | None = None
    fill_bar_offset: int = 0
    reference_price: float | None = None
    planned_price: float = 0.0
    participation_cap: float | None = None
    bar_volume: float | None = None
    eligibility_lag_seconds: float = 0.0
    ambiguous_fill: bool = False
    miss_reason: str = ""
    costs: CostBreakdown = Field(default_factory=CostBreakdown)
    bars_short_of_horizon: int = 0
    notes: tuple[str, ...] = ()

    @property
    def filled(self) -> bool:
        return self.status in (FillStatus.FILLED.value, FillStatus.PARTIAL.value)

    @classmethod
    def from_fill(cls, fill: EntryFill, *, run_id: str) -> "OrderRecord":
        return cls(
            run_id=run_id,
            symbol=fill.symbol,
            signal_ts=fill.signal_ts,
            submitted_ts=fill.submitted_ts,
            eligible_ts=fill.eligible_ts,
            order_type=fill.order_type,
            side=fill.side,
            status=fill.status.value,
            requested_size=fill.requested_size,
            filled_size=fill.filled_size,
            fill_ts=fill.fill_ts,
            fill_bar_offset=fill.fill_bar_offset,
            reference_price=fill.reference_price,
            planned_price=fill.planned_price,
            participation_cap=fill.participation_cap,
            bar_volume=fill.bar_volume,
            eligibility_lag_seconds=fill.eligibility_lag_seconds,
            ambiguous_fill=fill.ambiguous_fill,
            miss_reason=fill.miss_reason,
            costs=fill.costs,
            notes=fill.notes,
        )

    @classmethod
    def unresolved(
        cls,
        order: PendingOrder,
        *,
        run_id: str,
        bars_short: int,
        detail: str = "",
    ) -> "OrderRecord":
        """An order the data ran out from under.

        Not a miss: a miss is a statement about the market, and this is a
        statement about the dataset. Resolving it against the bars that DO
        exist would declare a resting entry expired on a horizon that never
        elapsed, which is a fabricated outcome.
        """
        return cls(
            run_id=run_id,
            symbol=order.symbol,
            signal_ts=order.signal_ts,
            submitted_ts=order.submitted_ts,
            eligible_ts=order.eligible_ts,
            order_type=order.order_type,
            side=order.intent.side,
            status="PENDING_AT_END_OF_DATA",
            requested_size=order.intent.size,
            planned_price=order.intent.entry_price,
            bars_short_of_horizon=int(bars_short),
            notes=tuple(order.notes) + ((detail,) if detail else ()),
        )


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------


class Journal:
    """Accumulates the run's rows and the histograms section 12 asks for.

    Pure accumulation: no I/O, no clock, no config beyond the one flag that
    governs row volume. `database/store.py` persists what this holds; keeping
    the two apart means a run can be inspected in memory without a database
    and written to a database without re-running it.
    """

    def __init__(
        self,
        *,
        run_id: str,
        record_wait_signals: bool = True,
        split_label: SplitPhase | None = None,
    ) -> None:
        self.run_id = run_id
        self.record_wait_signals = bool(record_wait_signals)
        self.split_label = split_label

        self._signals: list[SignalRecord] = []
        self._orders: list[OrderRecord] = []
        self._trades: list[TradeRecord] = []
        self._equity: list[EquityPoint] = []

        self._action_counts: Counter[str] = Counter()
        self._wait_counts: Counter[str] = Counter()
        self._stage_counts: Counter[str] = Counter()
        self._miss_counts: Counter[str] = Counter()
        self._exit_counts: Counter[str] = Counter()
        self._signals_seen = 0
        self._waits_dropped = 0
        self._orders_submitted = 0

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Journal(run={self.run_id} signals_seen={self._signals_seen} "
            f"rows={len(self._signals)} trades={len(self._trades)})"
        )

    # --- recording -----------------------------------------------------

    def record_signal(
        self,
        signal: Signal,
        *,
        blocking_stage: str = "",
        stages_run: tuple[str, ...] = (),
        intent_built: bool = False,
        order_submitted: bool = False,
        equity_at_signal: float | None = None,
        sizing_reason: str = "",
        limit_breach: str = "",
    ) -> SignalRecord | None:
        """Record one bar's decision. Returns the row, or None if not persisted.

        The counters move on EVERY call. The row is appended for every
        directional signal, and for a WAIT only when `record_wait_signals` is
        set -- a dropped WAIT is counted twice over (in `wait_counts` and in
        `waits_dropped`) so the run report can say exactly how many rows were
        not kept.
        """
        self._signals_seen += 1
        self._action_counts[signal.action.value] += 1
        if signal.wait_reason is not None:
            self._wait_counts[signal.wait_reason.value] += 1
        if blocking_stage:
            self._stage_counts[blocking_stage] += 1
        if order_submitted:
            self._orders_submitted += 1

        if signal.action is SignalAction.WAIT and not self.record_wait_signals:
            self._waits_dropped += 1
            return None

        record = SignalRecord.from_signal(
            signal,
            run_id=self.run_id,
            blocking_stage=blocking_stage,
            stages_run=stages_run,
            intent_built=intent_built,
            order_submitted=order_submitted,
            equity_at_signal=equity_at_signal,
            sizing_reason=sizing_reason,
            limit_breach=limit_breach,
            split_label=self.split_label,
        )
        self._signals.append(record)
        return record

    def record_order(self, fill: EntryFill) -> OrderRecord:
        """Record one resolved entry attempt, filled or missed."""
        record = OrderRecord.from_fill(fill, run_id=self.run_id)
        if not fill.filled:
            self._miss_counts[fill.miss_reason or "unspecified"] += 1
        self._orders.append(record)
        return record

    def record_unresolved_order(
        self, order: PendingOrder, *, bars_short: int, detail: str = ""
    ) -> OrderRecord:
        record = OrderRecord.unresolved(
            order, run_id=self.run_id, bars_short=bars_short, detail=detail
        )
        self._miss_counts["pending_at_end_of_data"] += 1
        self._orders.append(record)
        return record

    def record_trade(self, trade: TradeRecord) -> TradeRecord:
        if trade.run_id != self.run_id:
            raise JournalError(
                f"trade {trade.trade_id} belongs to run {trade.run_id!r}, not "
                f"{self.run_id!r}; mixing runs in one journal would mix their "
                "provenance and the rows could not be attributed afterwards"
            )
        self._trades.append(trade)
        self._exit_counts[trade.exit_reason.value] += 1
        return trade

    def record_equity(
        self, point: EquityPoint | None, *, replace_same_ts: bool = False
    ) -> EquityPoint | None:
        """Append an equity sample, or replace the one already at that instant.

        `replace_same_ts` exists for the end of a run: the last bar has already
        been marked when a position is flattened, and appending would leave two
        samples at one timestamp -- which any return or drawdown series reads
        as two bars, one of them with a zero time step.
        """
        if point is None:
            return None
        if replace_same_ts and self._equity and self._equity[-1].ts == point.ts:
            self._equity[-1] = point
        else:
            self._equity.append(point)
        return point

    # --- rows ----------------------------------------------------------

    @property
    def signals(self) -> tuple[SignalRecord, ...]:
        return tuple(self._signals)

    @property
    def orders(self) -> tuple[OrderRecord, ...]:
        return tuple(self._orders)

    @property
    def trades(self) -> tuple[TradeRecord, ...]:
        return tuple(self._trades)

    @property
    def equity(self) -> tuple[EquityPoint, ...]:
        return tuple(self._equity)

    @property
    def signals_seen(self) -> int:
        """Bars evaluated, including WAITs whose rows were not kept."""
        return self._signals_seen

    @property
    def waits_dropped(self) -> int:
        return self._waits_dropped

    @property
    def orders_submitted(self) -> int:
        return self._orders_submitted

    # --- histograms ----------------------------------------------------

    def action_counts(self) -> dict[str, int]:
        return dict(sorted(self._action_counts.items()))

    def wait_counts(self) -> dict[str, int]:
        """How often each `WaitReason` fired. The rejection analysis."""
        return dict(sorted(self._wait_counts.items()))

    def blocking_stage_counts(self) -> dict[str, int]:
        """How often each pipeline stage was the one that blocked."""
        return dict(sorted(self._stage_counts.items()))

    def miss_counts(self) -> dict[str, int]:
        return dict(sorted(self._miss_counts.items()))

    def exit_reason_counts(self) -> dict[str, int]:
        return dict(sorted(self._exit_counts.items()))

    def terminal_gate(self) -> tuple[str, int] | None:
        """The reason that blocked most often, or None over no WAITs.

        None rather than an arbitrary pick, and the count travels with the
        name so that "the terminal gate" is never quoted without its n.
        """
        if not self._wait_counts:
            return None
        reason, count = max(
            sorted(self._wait_counts.items()), key=lambda item: item[1]
        )
        return reason, count

    # --- cost drag -----------------------------------------------------

    def cost_drag_by_exit_reason(self) -> dict[str, dict[str, float]]:
        """Realized cost drag, split by how the trade exited.

        Per reason: `n`, the summed costs, the summed R at risk, and the mean
        drag in R. A reason with no trades is absent, and the whole mapping is
        empty over an empty trade list.

        Never collapsed into one figure. The measured Phase 7a asymmetry on NQ
        (a stop pays a market order's slippage plus the spread; a target is a
        resting limit and pays neither) means a single average drag would
        overstate the cost of the trades that win and understate the cost of
        the trades that lose -- in the direction that flatters the system.
        """
        totals: dict[str, dict[str, float]] = {}
        for trade in self._trades:
            key = trade.exit_reason.value
            bucket = totals.setdefault(
                key, {"n": 0.0, "costs": 0.0, "risk_dollars": 0.0, "drag_r": 0.0}
            )
            bucket["n"] += 1.0
            bucket["costs"] += trade.costs.total
            bucket["risk_dollars"] += trade.risk_dollars
            bucket["drag_r"] += trade.cost_drag_r
        for bucket in totals.values():
            n = bucket["n"]
            bucket["mean_costs"] = bucket["costs"] / n
            bucket["mean_drag_r"] = bucket["drag_r"] / n
        return dict(sorted(totals.items()))

    def cost_totals(self) -> dict[str, float]:
        """Itemized cost sums over the closed trades.

        Sums, which are defined over an empty set: every entry is 0.0 when
        nothing traded, and that is a true statement about money spent.
        """
        fields = (
            "commission",
            "exchange_fees",
            "entry_slippage",
            "exit_slippage",
            "spread_cost",
        )
        totals = {name: 0.0 for name in fields}
        totals["total"] = 0.0
        for trade in self._trades:
            for name in fields:
                totals[name] += getattr(trade.costs, name)
            totals["total"] += trade.costs.total
        return totals

    # --- the report ----------------------------------------------------

    def rejection_report(self) -> dict[str, object]:
        """Why the run did what it did, in counts.

        This is the report that answers "the system took no trades -- where
        did they go?", and it is the first thing to read on an empty run.
        """
        terminal = self.terminal_gate()
        return {
            "bars_evaluated": self._signals_seen,
            "signal_rows_kept": len(self._signals),
            "waits_dropped_by_config": self._waits_dropped,
            "actions": self.action_counts(),
            "wait_reasons": self.wait_counts(),
            "blocking_stages": self.blocking_stage_counts(),
            "orders_submitted": self._orders_submitted,
            "order_outcomes": self.miss_counts(),
            "exit_reasons": self.exit_reason_counts(),
            "terminal_gate": None if terminal is None else {
                "reason": terminal[0],
                "count": terminal[1],
            },
        }

    def describe(self) -> dict[str, object]:
        """Counts and sums only. No ratio is computed here at any n."""
        return {
            "run_id": self.run_id,
            "bars_evaluated": self._signals_seen,
            "signals_recorded": len(self._signals),
            "orders_recorded": len(self._orders),
            "trades": len(self._trades),
            "equity_points": len(self._equity),
            "record_wait_signals": self.record_wait_signals,
            "cost_totals": self.cost_totals(),
            "cost_drag_by_exit_reason": self.cost_drag_by_exit_reason(),
            "cost_drag_note": (
                "cost drag is reported per exit reason and never as one figure: a "
                "target exit is a resting limit that crosses no spread, a stop exit is "
                "a market order that pays additional slippage, and their average "
                "describes neither"
            ),
            "rejection": self.rejection_report(),
        }
