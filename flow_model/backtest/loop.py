"""The event loop: section 9's six steps, in section 9's order.

Section 9 fixes the loop verbatim, and the ORDER IS THE SPECIFICATION:

    for ts in clock:                      # bar closes, chronological
        ledger.mark_to_market(ts)
        manager.update_open_positions(ts) # stop/target/time exits first
        signal = signal_engine.evaluate(MarketView(symbol, ts))
        journal.record_signal(signal)      # including WAITs + reasons
        if signal.action is not WAIT:
            intent = risk_manager.build_intent(signal)
            if intent: execution.submit(intent, ts)
        execution.process_pending(ts)      # fills resolve on later bars only

Each ordering constraint is load-bearing, and getting one wrong flatters the
results silently rather than raising:

* **mark_to_market first**, on THIS bar's prices, so the kill switch and the
  drawdown series see the bar they are about to act on. Evaluated on last
  bar's equity, a kill switch fires a bar late.
* **exits before entries.** A new entry sized before the open position exited
  is sized against the wrong open heat and the wrong position count, and
  `max_open_positions=1` is violated for one bar. It matters for the daily
  loss limit too: a loss realized on this bar must be visible to the limit
  check that gates this bar's entry.
* **a MarketView, not raw bars.** The firewall is the strongest safety
  property in the project and the loop does not route around it. The engine
  gets a view built at `ts`; the ledger and the execution model get bars the
  loop has already stepped past, handed to them as arguments.
* **a signal recorded for every bar**, WAITs included, because section 12
  asks how often each gate fires and the answer cannot be reconstructed
  afterwards.
* **process_pending last**, and fills resolve on later bars only. A fill on
  the bar that produced the signal has used that bar's future.
  `backtest/execution.py` enforces the timing; this loop's job is not to undo
  it, which is why `_process_pending` waits for an order's whole horizon to
  close before resolving it (see `_PendingEntry`).

That order is not merely written down here: `_PhaseTracker` enforces it at
runtime on every bar. A step out of order, a step repeated, or a mandatory
step skipped raises `LoopOrderError`. It costs one integer comparison per
step, which is the right price for the invariant the whole phase rests on.

## Zero trades is the live case

Phase 6 committed the finding (f952d3e) that this system takes NO trades: over
2500 consecutive bars with every feed present the highest Flow Score observed
is 65.23, and the lowest setup threshold is SCALP_1R's `min_flow_score` of
70.0, so `score_below_threshold` is the terminal gate. A run today must
therefore complete and produce a valid, EMPTY result. Everything in
`RunResult` is a count, a sum or an explicit `None` for that reason: counts
and sums are defined over an empty set, ratios are not. No win rate, profit
factor or expectancy is computed anywhere in this module at any n -- those
belong to `analytics/`, whose job includes saying "undefined, n=0" where a
`0.0` would read as a measurement.

## Determinism

The loop draws nothing. It holds no generator and reads no clock: `run_id` is
content-derived from the config hash, the data hash, the seed and the split, so
two runs of the same config over the same data produce identical `trade_id`s
and byte-identical rows. Wall-clock time appears exactly once, as
`RunProvenance.created_at`, which is excluded from `result_fingerprint()` for
that reason. The only stochastic terms in the whole backtest are
`ExecutionConfig.random_latency_ms` and `random_miss_probability`, and
`ExecutionModel` draws those from per-order named streams
(`core/determinism.rng`), so adding a stream cannot shift another's draws.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence

import numpy as np
from pydantic import Field

from flow_model import __version__ as CODE_VERSION
from flow_model.backtest.costs import VolatilityContext
from flow_model.backtest.execution import ExecutionModel
from flow_model.backtest.fill_timing import PendingOrder, min_fill_offset
from flow_model.backtest.fills import EntryFill
from flow_model.backtest.journal import Journal, OrderRecord, SignalRecord
from flow_model.backtest.ledger import (
    EntryContext,
    Ledger,
    LedgerSnapshot,
    OpenPosition,
    SessionFlattenPolicy,
)
from flow_model.core.contracts import (
    Bar,
    EquityPoint,
    QuoteSnapshot,
    TradeIntent,
    TradeRecord,
)
from flow_model.core.determinism import stable_hash
from flow_model.core.enums import (
    ExitReason,
    Session,
    SignalAction,
    SplitPhase,
    StrEnum,
)
from flow_model.core.model import FrozenModel
from flow_model.data.market_view import MarketView
from flow_model.data.series import from_ns, to_ns
from flow_model.risk.limits import RiskLimitManager
from flow_model.risk.sizing import size_position
from flow_model.signals.engine import EngineDecision, SignalEngine
from flow_model.utils.logging import get_logger

__all__ = [
    "BacktestError",
    "BacktestLoop",
    "LoopOrderError",
    "LoopPhase",
    "MANDATORY_PHASES",
    "MarketSource",
    "PositionManager",
    "RunProvenance",
    "RunResult",
    "SECTION_9_ORDER",
    "StoreMarketSource",
    "VolatilityTracker",
    "build_intent",
]

logger = get_logger("backtest.loop")


class BacktestError(RuntimeError):
    """Raised when a run cannot proceed honestly."""


class LoopOrderError(BacktestError):
    """Raised when section 9's step order is violated."""


# ---------------------------------------------------------------------------
# The order, made enforceable
# ---------------------------------------------------------------------------


class LoopPhase(StrEnum):
    """One step of section 9's loop body."""

    MARK_TO_MARKET = "mark_to_market"
    UPDATE_OPEN_POSITIONS = "update_open_positions"
    EVALUATE = "evaluate"
    RECORD_SIGNAL = "record_signal"
    BUILD_AND_SUBMIT = "build_and_submit"
    PROCESS_PENDING = "process_pending"


#: Section 9's order, as data. The tracker checks against this tuple, so the
#: implemented order is verifiable rather than asserted in prose.
SECTION_9_ORDER: tuple[LoopPhase, ...] = (
    LoopPhase.MARK_TO_MARKET,
    LoopPhase.UPDATE_OPEN_POSITIONS,
    LoopPhase.EVALUATE,
    LoopPhase.RECORD_SIGNAL,
    LoopPhase.BUILD_AND_SUBMIT,
    LoopPhase.PROCESS_PENDING,
)

#: Every step except the conditional one. `build_and_submit` runs only when the
#: signal is directional, so it is the one step a bar may legitimately skip.
MANDATORY_PHASES: tuple[LoopPhase, ...] = tuple(
    phase for phase in SECTION_9_ORDER if phase is not LoopPhase.BUILD_AND_SUBMIT
)

_PHASE_INDEX = {phase: index for index, phase in enumerate(SECTION_9_ORDER)}


class _PhaseTracker:
    """Enforces section 9's step order, per bar, at runtime.

    Steps may be SKIPPED (only `build_and_submit` legitimately is) but never
    reordered or repeated, and the mandatory steps must all have run by the
    end of the bar. A tracker rather than a comment because the ordering is
    the specification: every one of the four constraints in this module's
    docstring fails silently, in the flattering direction, and a test that
    only checks the result cannot see the order the result came from.
    """

    __slots__ = ("bars", "_ts", "_last", "_seen")

    def __init__(self) -> None:
        self.bars = 0
        self._ts: datetime | None = None
        self._last = -1
        self._seen: set[LoopPhase] = set()

    def begin_bar(self, ts: datetime) -> None:
        if self._ts is not None:
            raise LoopOrderError(
                f"bar {self._ts.isoformat()} was never finished before "
                f"{ts.isoformat()} began"
            )
        self._ts = ts
        self._last = -1
        self._seen = set()

    def enter(self, phase: LoopPhase) -> None:
        if self._ts is None:
            raise LoopOrderError(f"{phase.value} ran outside a bar")
        index = _PHASE_INDEX[phase]
        if index <= self._last:
            previous = SECTION_9_ORDER[self._last]
            raise LoopOrderError(
                f"{self._ts.isoformat()}: {phase.value} ran after {previous.value}. "
                f"Section 9's order is {' -> '.join(p.value for p in SECTION_9_ORDER)}, "
                "and each constraint in it changes results silently when broken"
            )
        self._last = index
        self._seen.add(phase)

    def end_bar(self) -> None:
        if self._ts is None:
            raise LoopOrderError("end_bar without begin_bar")
        missing = [p.value for p in MANDATORY_PHASES if p not in self._seen]
        if missing:
            raise LoopOrderError(
                f"{self._ts.isoformat()}: bar finished without {', '.join(missing)}"
            )
        self._ts = None
        self.bars += 1


# ---------------------------------------------------------------------------
# Market access for the execution side
# ---------------------------------------------------------------------------


class MarketSource(Protocol):
    """What the loop needs that a `MarketView` deliberately cannot give.

    The execution model prices fills against bars; the firewall is about what
    the STRATEGY may see, and `MarketView` is built for that -- no accessor
    takes a timestamp. The execution side needs the bar that just closed, so
    it gets a source with a cursor the loop advances, which can serve the
    present and the past and has no way to reach the future.
    """

    symbol: str

    def clock_ns(self) -> np.ndarray:
        """Bar close instants as int64 UTC nanoseconds, chronological."""

    def advance(self, ts_ns: int) -> None:
        """Move the cursor to `ts_ns`. Must not go backwards."""

    def bar(self) -> Bar | None:
        """The bar that closed at the cursor, or None if this symbol had none."""

    def quote(self, ts: datetime | None = None) -> QuoteSnapshot | None:
        """The last quote at or before `ts` (default: the cursor)."""

    def intrabar(self, bar: Bar) -> tuple[Bar, ...]:
        """Finer-granularity bars inside `bar`, or () when unavailable."""

    def view(self) -> MarketView:
        """A point-in-time view at the cursor."""

    def data_hash(self) -> str:
        """Provenance for the dataset this source reads."""

    def describe(self) -> dict[str, object]:
        """What the dataset is, for the run record."""


class StoreMarketSource:
    """A `MarketSource` over a `DataStore`, with a cursor that cannot rewind.

    The cursor is the structural guarantee: `advance` refuses to go backwards,
    every accessor is relative to the cursor, and no accessor takes a future
    timestamp. A caller holding this object cannot read tomorrow's bar, which
    is the same property `MarketView` provides one layer up.
    """

    def __init__(self, store, symbol: str, *, intrabar_interval: int | None = None) -> None:
        self.store = store
        self.symbol = symbol
        data = store.get(symbol)
        self._data = data
        self._bars = data.primary_bars
        self.interval_seconds = int(data.primary_interval)
        self._quotes = data.quotes
        self._intrabar_interval = (
            None if intrabar_interval is None else int(intrabar_interval)
        )
        self._intrabar = (
            data.bars.get(int(intrabar_interval)) if intrabar_interval else None
        )
        if intrabar_interval and self._intrabar is None:
            raise BacktestError(
                f"{symbol}: no bars loaded at the requested intrabar interval "
                f"{intrabar_interval}s, so intrabar resolution would silently fall "
                "back to the pessimistic rule while the run claimed to use it"
            )
        self._cursor_ns: int | None = None
        self._index = -1

    @classmethod
    def from_config(cls, store, symbol: str, config) -> "StoreMarketSource":
        """Take the intrabar interval from `config.data.intrabar_interval_seconds`."""
        interval = config.data.intrabar_interval_seconds
        data = store.get(symbol)
        if interval is not None and int(interval) not in data.bars:
            # Configured but not loaded is not an error here: the dataset
            # simply has no finer series, and section 9's pessimistic rule
            # applies. Saying so beats refusing to run.
            logger.info(
                "%s: intrabar_interval_seconds=%s is configured but not loaded; "
                "exits resolve by the pessimistic same-bar rule",
                symbol,
                interval,
            )
            interval = None
        return cls(store, symbol, intrabar_interval=interval)

    # --- the cursor ----------------------------------------------------

    def clock_ns(self) -> np.ndarray:
        return self.store.timeline(self.symbol)

    def advance(self, ts_ns: int) -> None:
        ts_ns = int(ts_ns)
        if self._cursor_ns is not None and ts_ns < self._cursor_ns:
            raise BacktestError(
                f"{self.symbol}: the clock moved backwards from "
                f"{from_ns(self._cursor_ns).isoformat()} to {from_ns(ts_ns).isoformat()}"
            )
        self._cursor_ns = ts_ns
        count = self._bars.visible_count(ts_ns)
        self._index = count - 1

    def _require_cursor(self) -> int:
        if self._cursor_ns is None:
            raise BacktestError(f"{self.symbol}: the source has not been advanced yet")
        return self._cursor_ns

    # --- accessors -----------------------------------------------------

    def bar(self) -> Bar | None:
        cursor = self._require_cursor()
        if self._index < 0:
            return None
        if int(self._bars.ts_ns[self._index]) != cursor:
            # A merged multi-symbol clock steps at instants this symbol has no
            # bar for. None, rather than the previous bar: the previous bar has
            # already been marked and acted on, and offering it again would
            # double-count it.
            return None
        return self._bars.bar_at(self._index)

    def quote(self, ts: datetime | None = None) -> QuoteSnapshot | None:
        cursor = self._require_cursor()
        # The cutoff is checked BEFORE the "no quotes loaded" exit. A guard
        # that only fires when the data happens to be present is a guard that
        # passes silently on every bar-only dataset and then fails the first
        # time someone loads quotes.
        cutoff = cursor if ts is None else to_ns(ts)
        if cutoff > cursor:
            raise BacktestError(
                f"{self.symbol}: a quote was requested at {from_ns(cutoff).isoformat()}, "
                f"after the cursor at {from_ns(cursor).isoformat()}. A fill priced off a "
                "quote that had not happened yet is lookahead"
            )
        if self._quotes is None or not len(self._quotes):
            return None
        count = self._quotes.visible_count(cutoff)
        if count <= 0:
            return None
        return self._quotes.quote_at(count - 1)

    def intrabar(self, bar: Bar) -> tuple[Bar, ...]:
        if self._intrabar is None or not len(self._intrabar):
            return ()
        cursor = self._require_cursor()
        start = to_ns(bar.open_ts) + 1
        end = min(to_ns(bar.close_ts), cursor) + 1
        if end <= start:
            return ()
        return self._intrabar.slice_range(start, end).to_bars()

    def view(self) -> MarketView:
        cursor = self._require_cursor()
        return self.store.view(self.symbol, from_ns(cursor), now_ns=cursor)

    def data_hash(self) -> str:
        return self.store.combined_data_hash()

    def describe(self) -> dict[str, object]:
        described = dict(self._data.describe())
        described["intrabar_interval_seconds"] = self._intrabar_interval
        described["store_latency_seconds"] = self.store.latency_seconds
        return described


# ---------------------------------------------------------------------------
# The volatility ratio the cost model needs
# ---------------------------------------------------------------------------


class VolatilityTracker:
    """Supplies `VolatilityContext` (ATR over its own median) from measurements.

    `ExecutionConfig.vol_slippage_coefficient` is documented in units of
    "extra slippage ticks per 1.0 of (ATR / median ATR - 1)", and no feature
    publishes a median ATR -- `features/volatility.py` computes one internally
    for `vol_expansion` and emits only the squashed result. So the ratio is
    formed here, from the ATRs the loop has already seen.

    Two properties make that honest rather than convenient:

    * **Point-in-time.** Only ATRs from bars already processed enter the
      window. The median at bar t is a function of bars <= t.
    * **No collapse of absence into 1.0.** Until the window is FULL the
      context is `VolatilityContext.unmeasured()`, not a ratio of 1.0. A
      median of one observation equals that observation and would report "ATR
      is exactly at its median" on the first bar -- the precise collapse that
      type exists to prevent. Unmeasured declines an entry (execution refuses
      to charge zero volatility slippage for an unmeasured ratio), and since
      the window only ever grows, a position can only have been opened once it
      was full -- so no open position can later meet an unmeasurable exit.

    The window length is `config.levels.atr_median_window`, an existing config
    field whose stated purpose is "Window for the ATR median". Reusing it is a
    judgement call -- it was declared for 14.4's volatility-regularity term --
    and it is overridable, but it introduces no new constant.
    """

    __slots__ = ("window", "_atrs", "_last_atr", "_last_ts", "_observations")

    def __init__(self, *, window: int) -> None:
        if window < 2:
            raise BacktestError(
                f"an ATR median window of {window} cannot produce a median that "
                "differs from its single observation"
            )
        self.window = int(window)
        self._atrs: deque[float] = deque(maxlen=self.window)
        self._last_atr: float | None = None
        self._last_ts: datetime | None = None
        self._observations = 0

    @classmethod
    def from_config(cls, config, *, window: int | None = None) -> "VolatilityTracker":
        return cls(window=int(config.levels.atr_median_window if window is None else window))

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def is_measured(self) -> bool:
        return len(self._atrs) >= self.window

    def observe(self, atr: float | None, ts: datetime) -> None:
        """Record one bar's ATR. A missing or non-positive ATR is not recorded."""
        if atr is None or not math.isfinite(atr) or atr <= 0.0:
            return
        self._atrs.append(float(atr))
        self._last_atr = float(atr)
        self._last_ts = ts
        self._observations += 1

    def context(self, *, atr: float | None = None, ts: datetime | None = None) -> VolatilityContext:
        """The context for a fill at `ts`, using `atr` when the caller holds one.

        A caller that holds the ATR as of the SIGNAL bar (the entry path) passes
        it, so the fill is priced with the measurement the decision was made
        on rather than with a later one.
        """
        if not self.is_measured:
            return VolatilityContext.unmeasured(
                f"the ATR median window is not full ({len(self._atrs)}/{self.window} "
                "measured ATRs), so ATR/median ATR has not been measured. Reported as "
                "absent rather than as a ratio of 1.0, which would claim a measurement "
                "that was never made"
            )
        value = atr if atr is not None else self._last_atr
        notes: list[str] = []
        if atr is None and self._last_ts is not None and ts is not None and self._last_ts < ts:
            notes.append(
                f"ATR carried forward from {self._last_ts.isoformat()} because this bar "
                "published none; the measurement is real but older than the fill it "
                "prices"
            )
        median = float(np.median(np.asarray(self._atrs, dtype=np.float64)))
        return VolatilityContext(atr=value, median_atr=median, notes=tuple(notes))


# ---------------------------------------------------------------------------
# The risk manager's output
# ---------------------------------------------------------------------------


def build_intent(
    decision: EngineDecision,
    *,
    config,
    spec,
    sizing_equity: float | None = None,
) -> TradeIntent | None:
    """Section 9's `risk_manager.build_intent(signal)`.

    Takes the DECISION rather than the `Signal`, because `Signal` is a frozen
    Phase 1 contract carrying no price geometry -- `EntryPlan`'s own docstring
    says so -- and the plan it needs travels beside the signal on
    `EngineDecision`. `evaluate()` is literally `decide(...).signal`, so this
    changes nothing about what the engine decides; it reads the risk stage's
    output instead of throwing it away and guessing it back.

    `sizing_equity` is `config.risk.compound_equity` being honoured. The
    engine takes ONE equity number and uses it for both sizing and the section
    8 limit checks, and under fixed-fractional sizing those two want different
    numbers: the limits must see real mark-to-market equity (a kill switch
    evaluated against a constant would read a profitable account as being in
    drawdown), while the size must come off the starting equity. So the loop
    hands the engine the true equity and, when the two differ, the position is
    re-sized here through the same pure `size_position` function the engine
    used. When they agree -- the default, compounding -- the plan's intent is
    returned untouched.
    """
    if decision.signal.action is SignalAction.WAIT:
        return None
    plan = decision.plan
    if plan is None:
        raise BacktestError(
            f"{decision.signal.symbol} at {decision.signal.ts.isoformat()}: a "
            f"{decision.signal.action.value} signal arrived with no EntryPlan. The risk "
            "stage produces the plan, so a directional signal without one means the "
            "pipeline was short-circuited and there is no sized, bracketed trade to send"
        )
    if sizing_equity is None or math.isclose(
        float(sizing_equity), plan.equity_at_signal, rel_tol=0.0, abs_tol=1e-9
    ):
        return plan.to_trade_intent()

    setup = config.setup(plan.setup)
    if plan.atr is None:
        raise BacktestError(
            f"{plan.symbol}: re-sizing at a fixed equity needs the ATR the stop-width "
            "cap is measured in, and the plan carries none. Sizing with the cap removed "
            "is what signals/engine.py refuses by name"
        )
    resized = size_position(
        spec=spec,
        side=plan.side,
        entry_price=plan.entry_price,
        stop_price=(
            plan.structural_stop_price
            if plan.structural_stop_price is not None
            else plan.stop_price
        ),
        reward_risk=float(setup.reward_risk),
        equity=float(sizing_equity),
        risk_pct=float(config.risk.risk_per_trade_pct),
        max_risk_pct=float(config.risk.max_risk_per_trade_pct),
        min_stop_ticks=int(setup.min_stop_ticks),
        max_stop_distance=float(setup.max_stop_atr_multiple) * float(plan.atr),
        min_reward_risk=float(setup.min_reward_risk),
        allow_fractional=bool(spec.allow_fractional_size),
    )
    if not resized.accepted:
        logger.info(
            "%s at %s: no intent -- fixed-fractional re-sizing refused the trade (%s)",
            plan.symbol,
            plan.signal_ts.isoformat(),
            resized.reason,
        )
        return None
    return TradeIntent(
        symbol=plan.symbol,
        signal_ts=plan.signal_ts,
        side=plan.side,
        setup=plan.setup,
        entry_price=resized.entry_price,
        stop_price=resized.stop_price,
        target_price=resized.target_price,
        size=resized.size,
        risk_dollars=resized.risk_dollars,
        point_value=plan.point_value,
        equity_at_signal=float(sizing_equity),
        flow_score=plan.flow_score,
        regime=plan.regime,
        max_hold_bars=plan.max_hold_bars,
        reasons=plan.reasons
        + (
            f"risk: re-sized at a FIXED equity of {float(sizing_equity):.2f} "
            f"(compound_equity=False); the engine sized {plan.size:g} at "
            f"{plan.equity_at_signal:.2f} mark-to-market equity",
        ),
    )


# ---------------------------------------------------------------------------
# Open positions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PendingEntry:
    """A submitted order plus everything its fill will need.

    The volatility context and the signal-time context are captured HERE, at
    submission, not looked up when the fill resolves. The fill resolves on a
    later bar, and pricing it with that bar's measurements would charge the
    entry with volatility the decision never saw.

    `horizon_bars` is how many bars must close after the signal bar before the
    order's whole wait horizon has elapsed. `_process_pending` holds the order
    until then, because `FillWindow` marks the last bar IT holds as the last
    candidate: resolving against a truncated window would declare a resting
    entry expired on a horizon that has not finished.
    """

    order: PendingOrder
    context: EntryContext
    vol: VolatilityContext
    min_offset: int
    horizon_bars: int


class PositionManager:
    """`manager.update_open_positions(ts)`: the exits, in the right order.

    Stop and target come from `ExecutionModel.resolve_exit`, which owns the gap
    rule and the same-bar rule. Only if neither level was reached does the
    ledger's own verdict apply -- time stop, session flatten, kill switch --
    priced through `exit_at_close`, so a ledger-decided flatten pays the same
    costs as any other market exit.

    The order within the bar is deliberate: the market's levels came first in
    time. Flattening at the close while the stop had already been touched
    intrabar would book a better price than reality.
    """

    def __init__(
        self,
        *,
        ledger: Ledger,
        execution: ExecutionModel,
        journal: Journal,
        limits: RiskLimitManager,
        session_policy: SessionFlattenPolicy,
        volatility: VolatilityTracker,
        session_date,
        kill_switch_drawdown_pct: float,
    ) -> None:
        self.ledger = ledger
        self.execution = execution
        self.journal = journal
        self.limits = limits
        self.session_policy = session_policy
        self.volatility = volatility
        self.session_date = session_date
        self.kill_switch_drawdown_pct = float(kill_switch_drawdown_pct)
        self.bars_without_a_bar = 0

    # --- opening -------------------------------------------------------

    def open_from_fill(
        self, intent: TradeIntent, fill: EntryFill, context: EntryContext
    ) -> OpenPosition:
        position = self.ledger.open_position(intent=intent, entry=fill, context=context)
        self.limits.on_position_opened(position.symbol, position.risk_dollars)
        return position

    # --- the per-bar update --------------------------------------------

    def update_open_positions(
        self,
        ts: datetime,
        *,
        bar: Bar | None,
        quote: QuoteSnapshot | None = None,
        intrabar: Sequence[Bar] | None = None,
    ) -> tuple[TradeRecord, ...]:
        """Resolve every open position against this bar. Exits before entries."""
        closed: list[TradeRecord] = []
        positions = self.ledger.open_positions
        if not positions:
            return ()
        if bar is None:
            # No bar for this symbol at this instant: nothing to resolve
            # against. The position is held and the bar is counted. Holding is
            # the only honest option -- there is no price with which to exit --
            # and the count is what makes a data gap visible afterwards.
            self.bars_without_a_bar += 1
            for position in positions:
                self.ledger.advance_bar(position, ts)
            return ()

        for position in positions:
            self.ledger.advance_bar(position, ts)
            record = self.resolve_one(
                position, ts, bar=bar, quote=quote, intrabar=intrabar
            )
            if record is not None:
                closed.append(record)
        return tuple(closed)

    def resolve_one(
        self,
        position: OpenPosition,
        ts: datetime,
        *,
        bar: Bar,
        quote: QuoteSnapshot | None = None,
        intrabar: Sequence[Bar] | None = None,
    ) -> TradeRecord | None:
        """Resolve ONE position against one bar. Returns its record, or None.

        Separate from `update_open_positions` because the entry path needs
        exactly this and nothing else: when an entry fills, the bar it filled
        on must be checked for the stop and the target, and re-running the
        whole book would advance every OTHER position's holding time a second
        time for the same bar -- which fires their time stops early.

        Does NOT advance the holding time; its caller owns that, because
        "which bars count as held" is a property of the bar, not of the
        resolution.
        """
        if bar.close_ts < position.entry_ts:
            return None
        position.observe(bar)
        vol = self.volatility.context(ts=ts)
        fill = self.execution.resolve_exit(
            position.plan,
            bar,
            vol=vol,
            intrabar=tuple(intrabar or ()),
            quote=quote,
        )
        if fill.resolved:
            return self._close(position, fill)
        reason = self._forced_reason(position, ts)
        if reason is None:
            return None
        forced = self.execution.exit_at_close(
            position.plan, bar, reason, vol=vol, quote=quote
        )
        return self._close(position, forced)

    def flatten_all(
        self,
        ts: datetime,
        *,
        bar: Bar | None,
        reason: ExitReason,
        quote: QuoteSnapshot | None = None,
    ) -> tuple[TradeRecord, ...]:
        """Force every position flat at `bar`'s close, for `reason`.

        Used for the end of the backtest. With no bar to price against, the
        positions stay open and are reported as such: a position closed at a
        price that was never observed is a fabricated trade.
        """
        if bar is None:
            return ()
        closed: list[TradeRecord] = []
        for position in self.ledger.open_positions:
            if bar.close_ts < position.entry_ts:
                continue
            position.observe(bar)
            fill = self.execution.exit_at_close(
                position.plan,
                bar,
                reason,
                vol=self.volatility.context(ts=ts),
                quote=quote,
            )
            closed.append(self._close(position, fill))
        return tuple(closed)

    # --- internals -----------------------------------------------------

    def _forced_reason(self, position: OpenPosition, ts: datetime) -> ExitReason | None:
        """The ledger's own exit verdict for this position, or None.

        The kill switch is checked first because it is the most serious fact
        about the bar, and the drawdown comparison is made against
        `config.risk.kill_switch_drawdown_pct` -- the same field
        `RiskLimitManager` reads, not a second threshold. The comparison is
        duplicated rather than delegated because `RiskLimitManager.check` is
        not a query: it increments trip counts and arms cooldowns, so calling
        it to ask a question would corrupt the rejection analysis it feeds.
        That manager remains the only authority over whether a NEW position
        may open.
        """
        if self.limits.killed or self.ledger.drawdown_fraction() >= self.kill_switch_drawdown_pct:
            return ExitReason.KILL_SWITCH
        if self.ledger.time_stop_due(position):
            return ExitReason.TIME_STOP
        if self.session_policy.due(position.signal_ts, ts):
            return ExitReason.SESSION_CLOSE
        return None

    def _close(self, position: OpenPosition, fill) -> TradeRecord:
        costs = self.execution.round_turn_costs(position.entry, fill)
        record = self.ledger.close_position(position, fill, costs)
        self.journal.record_trade(record)
        self.limits.on_trade_closed(
            ts=record.exit_ts,
            session_date=self.session_date(record.exit_ts),
            symbol=record.symbol,
            pnl=record.pnl,
            risk_dollars=position.risk_dollars,
        )
        return record


# ---------------------------------------------------------------------------
# The run record
# ---------------------------------------------------------------------------


class RunProvenance(FrozenModel):
    """What a result must carry to be a result at all.

    Section 12 and the brief both insist: a performance number whose inputs
    cannot be reconstructed is not a number, it is an anecdote. Everything
    needed to rebuild the run is here -- the config hash, the data hash, the
    seed, the git commit, the split phase and the code version -- and
    `run_id` is derived from the first four so that it is reproducible rather
    than allocated.

    `created_at` is the one wall-clock field, and it is excluded from
    `RunResult.result_fingerprint()`: a timestamp in the identity of a result
    would make two identical runs look different.
    """

    run_id: str
    symbol: str
    config_hash: str
    data_hash: str
    seed: int
    git_commit: str = ""
    git_dirty: bool = False
    code_version: str = ""
    split_label: SplitPhase | None = None
    split_name: str = ""
    created_at: datetime | None = None
    starting_equity: float = 0.0
    compound_equity: bool = True
    record_wait_signals: bool = True
    record_equity_every_bar: bool = True
    bars_in_clock: int = 0
    bars_processed: int = 0
    max_bars: int | None = None
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    interval_seconds: int = 0
    dataset: dict[str, object] = Field(default_factory=dict)

    @property
    def clock_truncated(self) -> bool:
        return self.bars_processed < self.bars_in_clock

    def inputs_fingerprint(self) -> str:
        """Hash of everything that can change a result. Excludes `created_at`."""
        return stable_hash(
            {
                "config_hash": self.config_hash,
                "data_hash": self.data_hash,
                "seed": self.seed,
                "split_label": self.split_label,
                "code_version": self.code_version,
                "git_commit": self.git_commit,
                "max_bars": self.max_bars,
            }
        )


class RunResult(FrozenModel):
    """One backtest run: its provenance, its rows, and counts.

    Deliberately carries NO performance ratio. The brief's rule -- "never
    fabricate historical performance" -- has a quieter corollary: do not
    report a ratio whose denominator is zero as though it were measured. Win
    rate, profit factor, expectancy and Sharpe are all undefined over the
    empty trade set this system currently produces, and they are computed in
    `analytics/` where "undefined, n=0" can be said properly. What is here is
    counts, sums and the rejection histogram -- all defined at n=0 -- plus
    cost drag split by exit reason, which is the one performance-adjacent
    number that must never be collapsed into a single figure.
    """

    provenance: RunProvenance
    trades: tuple[TradeRecord, ...] = ()
    equity: tuple[EquityPoint, ...] = ()
    signals: tuple[SignalRecord, ...] = ()
    orders: tuple[OrderRecord, ...] = ()
    final: LedgerSnapshot | None = None
    rejection: dict[str, object] = Field(default_factory=dict)
    cost_drag_by_exit_reason: dict[str, dict[str, float]] = Field(default_factory=dict)
    cost_totals: dict[str, float] = Field(default_factory=dict)
    ledger_diagnostics: dict[str, float] = Field(default_factory=dict)
    limit_trips: dict[str, int] = Field(default_factory=dict)
    bars_evaluated: int = 0
    waits_dropped: int = 0
    open_at_end: int = 0
    unresolved_orders: int = 0
    submissions_declined: int = 0
    notes: tuple[str, ...] = ()

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def is_empty(self) -> bool:
        """True when the run took no trades -- the current expected outcome."""
        return not self.trades

    def result_fingerprint(self) -> str:
        """Content hash of the rows a determinism test compares.

        Trades, equity and signal rows -- not the provenance, whose
        `created_at` moves with the wall clock.
        """
        return stable_hash(
            {
                "trades": [t.to_init_dict() for t in self.trades],
                "equity": [e.to_init_dict() for e in self.equity],
                "signals": [s.to_init_dict() for s in self.signals],
                "orders": [o.to_init_dict() for o in self.orders],
            }
        )

    def summary(self) -> dict[str, object]:
        """Counts and sums. Every ratio is left to `analytics/`."""
        wins = sum(1 for t in self.trades if t.is_win)
        losses = sum(1 for t in self.trades if t.is_loss)
        return {
            "run_id": self.provenance.run_id,
            "symbol": self.provenance.symbol,
            "split_label": (
                None if self.provenance.split_label is None
                else self.provenance.split_label.value
            ),
            "config_hash": self.provenance.config_hash,
            "data_hash": self.provenance.data_hash,
            "bars_processed": self.provenance.bars_processed,
            "bars_evaluated": self.bars_evaluated,
            "trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "scratches": len(self.trades) - wins - losses,
            "realized_pnl": None if self.final is None else self.final.realized_pnl,
            "ending_equity": None if self.final is None else self.final.equity,
            "open_at_end": self.open_at_end,
            "unresolved_orders": self.unresolved_orders,
            "submissions_declined": self.submissions_declined,
            "cost_totals": dict(self.cost_totals),
            "cost_drag_by_exit_reason": {
                k: dict(v) for k, v in self.cost_drag_by_exit_reason.items()
            },
            "ratios": (
                "not computed here: win rate, profit factor, expectancy and Sharpe are "
                "undefined over an empty trade set and are analytics' responsibility, "
                "which reports 'undefined, n=0' rather than 0.0"
            ),
            "rejection": dict(self.rejection),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class BacktestLoop:
    """Section 9's event loop for one symbol.

    One symbol per loop because the signal engine is per-symbol by contract --
    `decide_from` refuses stage inputs for another symbol, since the feature
    computers carry that symbol's `InstrumentSpec` -- and `evaluate(view)`
    takes a view of one symbol. The ledger, journal and limit manager are all
    symbol-keyed, so a multi-symbol driver is a loop over sources sharing one
    ledger; that driver is not built here and `config.backtest.symbols` with
    more than one entry is run one symbol at a time by the caller.

    No market data in the constructor: `run(source)` takes the data.
    """

    def __init__(
        self,
        config,
        symbol: str,
        *,
        engine: SignalEngine | None = None,
        execution: ExecutionModel | None = None,
        limits: RiskLimitManager | None = None,
        calendar=None,
        split_label: SplitPhase | None = None,
        split_name: str = "",
        starting_equity: float | None = None,
        git_commit: str | None = None,
        git_dirty: bool | None = None,
        repo_root: str = ".",
        atr_median_window: int | None = None,
    ) -> None:
        if symbol not in config.instruments:
            raise BacktestError(
                f"no InstrumentSpec for {symbol!r}; known: {sorted(config.instruments)}"
            )
        self.config = config
        self.symbol = symbol
        self.spec = config.spec(symbol)
        self.split_label = split_label
        self.split_name = split_name
        self.repo_root = repo_root
        self._git = (git_commit, git_dirty)

        if engine is not None and calendar is not None and engine.calendar is not calendar:
            raise BacktestError(
                "the engine was built with a different calendar than the one passed to "
                "the loop. One run has one session calendar: two would gate entries on "
                "one schedule and flatten positions on another"
            )
        self.engine = engine if engine is not None else SignalEngine(
            config, symbol, calendar=calendar
        )
        self.calendar = self.engine.calendar if calendar is None else calendar
        self.execution = (
            execution if execution is not None else ExecutionModel.from_config(config, symbol)
        )
        self.limits = limits if limits is not None else RiskLimitManager(
            config.risk, starting_equity
        )
        self.starting_equity = float(
            config.risk.starting_equity if starting_equity is None else starting_equity
        )
        self.atr_median_window = atr_median_window
        self.tracker = _PhaseTracker()

    # --- run id --------------------------------------------------------

    def run_id_for(self, data_hash: str) -> str:
        """A deterministic run id: a hash of the run's inputs.

        Content-derived, because `run_id` is a field of every `TradeRecord`
        and two runs of the same config over the same data are required to
        produce byte-identical trades. A uuid or a timestamp here would make
        the determinism test unsatisfiable by construction.
        """
        digest = stable_hash(
            {
                "config_hash": self.config.config_hash,
                "data_hash": data_hash,
                "symbol": self.symbol,
                "seed": int(self.config.backtest.seed),
                "split_label": self.split_label,
                "split_name": self.split_name,
                "code_version": CODE_VERSION,
            }
        )
        return f"run-{digest}"

    # --- the loop ------------------------------------------------------

    def run(self, source, *, clock_ns: Sequence[int] | None = None) -> RunResult:
        """Walk the clock. Returns a `RunResult`, empty trade list included."""
        backtest = self.config.backtest
        data_hash = source.data_hash()
        run_id = self.run_id_for(data_hash)

        clock = np.asarray(
            source.clock_ns() if clock_ns is None else clock_ns, dtype=np.int64
        )
        bars_in_clock = int(clock.size)
        if backtest.max_bars is not None:
            clock = clock[: int(backtest.max_bars)]

        ledger = Ledger(
            self.config,
            run_id=run_id,
            config_hash=self.config.config_hash,
            data_hash=data_hash,
            split_label=self.split_label,
            starting_equity=self.starting_equity,
        )
        journal = Journal(
            run_id=run_id,
            record_wait_signals=bool(backtest.record_wait_signals),
            split_label=self.split_label,
        )
        volatility = VolatilityTracker.from_config(
            self.config, window=self.atr_median_window
        )
        session_policy = SessionFlattenPolicy(self.config.session, self.calendar, self.spec)
        manager = PositionManager(
            ledger=ledger,
            execution=self.execution,
            journal=journal,
            limits=self.limits,
            session_policy=session_policy,
            volatility=volatility,
            session_date=self.engine.session_date,
            kill_switch_drawdown_pct=self.config.risk.kill_switch_drawdown_pct,
        )

        notes: list[str] = []
        if session_policy.note:
            notes.append(session_policy.note)
        if self.calendar is None:
            notes.append(
                "no session calendar was supplied: trade records carry Session.CLOSED "
                "rather than a measured session bucket, and the session breakdown in "
                "section 12 cannot be produced from this run"
            )
        if backtest.max_bars is not None and int(clock.size) < bars_in_clock:
            notes.append(
                f"config.backtest.max_bars={backtest.max_bars} truncated the clock from "
                f"{bars_in_clock} bars; this is a debug run and its counts are not the "
                "dataset's"
            )

        pending: list[_PendingEntry] = []
        submissions_declined = 0
        recent: deque[Bar] = deque(maxlen=self._recent_window())
        previous_ns: int | None = None
        first_ts: datetime | None = None
        last_ts: datetime | None = None
        last_bar: Bar | None = None

        for index in range(int(clock.size)):
            ts_ns = int(clock[index])
            if previous_ns is not None and ts_ns <= previous_ns:
                raise BacktestError(
                    f"{self.symbol}: the clock is not strictly ascending at "
                    f"{from_ns(ts_ns).isoformat()}; 'the next bar' is meaningless on an "
                    "unordered clock and a fill could resolve before its own signal"
                )
            previous_ns = ts_ns
            ts = from_ns(ts_ns)
            if first_ts is None:
                first_ts = ts
            last_ts = ts

            source.advance(ts_ns)
            bar = source.bar()
            if bar is not None:
                recent.append(bar)
                last_bar = bar
            quote = source.quote()
            intrabar = source.intrabar(bar) if bar is not None else ()

            self.tracker.begin_bar(ts)

            # 1. mark to market, on THIS bar's prices.
            self.tracker.enter(LoopPhase.MARK_TO_MARKET)
            prices = None if bar is None else {self.symbol: bar.close}
            # `mark_to_market` returns None on a bar it chose not to sample
            # (per `record_equity_every_bar`); the ledger always samples the
            # FIRST bar, and the run always closes the curve on the last.
            journal.record_equity(ledger.mark_to_market(ts, prices))
            self.limits.mark_equity(ledger.equity)

            # 2. exits before entries.
            self.tracker.enter(LoopPhase.UPDATE_OPEN_POSITIONS)
            manager.update_open_positions(ts, bar=bar, quote=quote, intrabar=intrabar)

            # 3. the signal, from a view built at ts.
            self.tracker.enter(LoopPhase.EVALUATE)
            view = source.view()
            if backtest.fail_on_lookahead_assertion:
                view.assert_no_future_leak()
            equity_for_limits = ledger.equity
            decision = self.engine.decide(
                view, equity=equity_for_limits, limits=self.limits
            )
            atr = None if decision.features is None else decision.features.get("atr")
            volatility.observe(atr, ts)

            # 4. record every signal, WAITs included.
            self.tracker.enter(LoopPhase.RECORD_SIGNAL)
            intent: TradeIntent | None = None
            if decision.signal.action is not SignalAction.WAIT:
                intent = build_intent(
                    decision,
                    config=self.config,
                    spec=self.spec,
                    sizing_equity=ledger.sizing_equity(),
                )
            # An order in flight is risk the limit manager cannot see: it
            # learns about a position at the FILL, and between submission and
            # fill `max_open_positions` is satisfied by a book that is about
            # to grow. Two signals on consecutive bars therefore both pass the
            # limit and both fill, and the account carries double the intended
            # size for as long as they both live. The count is enforced here,
            # against the same `config.risk.max_open_positions` the manager
            # reads, counting live orders alongside open positions.
            live = len(pending) + len(ledger.open_positions)
            declined = intent is not None and live >= int(
                self.config.risk.max_open_positions
            )
            if declined:
                submissions_declined += 1
            blocking = decision.blocking
            limit_breach = "" if decision.limits is None else decision.limits.breach
            if declined:
                limit_breach = "max_open_positions_in_flight"
            journal.record_signal(
                decision.signal,
                blocking_stage="" if blocking is None else blocking.stage.value,
                stages_run=tuple(stage.value for stage in decision.stages),
                intent_built=intent is not None,
                order_submitted=intent is not None and not declined,
                equity_at_signal=ledger.sizing_equity(),
                sizing_reason="" if decision.sizing is None else decision.sizing.reason,
                limit_breach=limit_breach,
            )

            # 5. submit.
            if intent is not None and not declined:
                self.tracker.enter(LoopPhase.BUILD_AND_SUBMIT)
                pending.append(
                    self._submit(
                        intent,
                        ts,
                        decision=decision,
                        volatility=volatility,
                        atr=atr,
                    )
                )

            # 6. fills resolve on later bars only.
            self.tracker.enter(LoopPhase.PROCESS_PENDING)
            self._process_pending(
                ts,
                pending=pending,
                recent=recent,
                source=source,
                manager=manager,
                journal=journal,
                bar=bar,
                quote=quote,
                intrabar=intrabar,
            )

            self.tracker.end_bar()

        # --- after the clock -------------------------------------------
        open_at_end = len(ledger.open_positions)
        if open_at_end and last_ts is not None:
            flattened = manager.flatten_all(
                last_ts,
                bar=last_bar,
                reason=ExitReason.END_OF_BACKTEST,
                quote=source.quote() if last_bar is not None else None,
            )
            if flattened:
                notes.append(
                    f"{len(flattened)} position(s) were still open when the data ended "
                    "and were flattened at the last bar's close with "
                    "exit_reason=END_OF_BACKTEST. They paid a full exit's costs, and "
                    "they are not trades the strategy chose to close: exclude them or "
                    "report them separately"
                )
            elif ledger.open_positions:
                notes.append(
                    f"{len(ledger.open_positions)} position(s) were still open when the "
                    "data ended and could not be priced (no final bar), so they are "
                    "reported open rather than closed at an unobserved price"
                )

        for entry in pending:
            available = sum(
                1 for b in recent if b.close_ts > entry.order.signal_ts
            )
            short = max(0, entry.horizon_bars - available)
            journal.record_unresolved_order(
                entry.order,
                bars_short=short,
                detail=(
                    f"the data ended {short} bar(s) before this order's wait horizon "
                    f"elapsed (eligible from offset {entry.min_offset}, horizon "
                    f"{entry.horizon_bars} bars, {available} available), so it is "
                    "recorded unresolved rather than declared expired on a horizon that "
                    "never ran"
                ),
            )
        if pending:
            notes.append(
                f"{len(pending)} order(s) were still live when the data ended; they are "
                "recorded as PENDING_AT_END_OF_DATA and were not filled"
            )
        if submissions_declined:
            notes.append(
                f"{submissions_declined} intent(s) were built and NOT submitted because "
                f"max_open_positions={self.config.risk.max_open_positions} was already "
                "committed to open positions or orders in flight. The signal rows carry "
                "intent_built=True with order_submitted=False, so these are "
                "distinguishable from bars that produced no intent at all"
            )

        # Always close the curve on the run's last bar. With per-bar recording
        # on, this REPLACES that bar's sample, which was taken before the
        # end-of-run flatten; with it off, it is what gives the curve its
        # second endpoint, so a total return has a span to be measured over.
        if last_ts is not None:
            journal.record_equity(
                ledger.record_equity_point(last_ts, replace_same_ts=True),
                replace_same_ts=True,
            )

        ledger.assert_consistent(journal.trades)
        notes.extend(ledger.notes)
        if not journal.trades:
            terminal = journal.terminal_gate()
            notes.append(
                "this run took NO trades. That is the expected outcome at the committed "
                "Phase 6 configuration, not a failure of the loop"
                + (
                    f"; the gate that blocked most often was {terminal[0]} "
                    f"({terminal[1]} bars)"
                    if terminal is not None
                    else ""
                )
                + ". Every ratio over this trade set is undefined, n=0 -- not 0.0"
            )

        provenance = self._provenance(
            run_id=run_id,
            data_hash=data_hash,
            source=source,
            bars_in_clock=bars_in_clock,
            bars_processed=int(clock.size),
            first_ts=first_ts,
            last_ts=last_ts,
        )
        return RunResult(
            provenance=provenance,
            trades=journal.trades,
            equity=journal.equity,
            signals=journal.signals,
            orders=journal.orders,
            final=ledger.snapshot(last_ts),
            rejection=journal.rejection_report(),
            cost_drag_by_exit_reason=journal.cost_drag_by_exit_reason(),
            cost_totals=journal.cost_totals(),
            ledger_diagnostics=ledger.diagnostics(),
            limit_trips=dict(sorted(self.limits.trip_counts().items())),
            bars_evaluated=journal.signals_seen,
            waits_dropped=journal.waits_dropped,
            open_at_end=len(ledger.open_positions),
            unresolved_orders=len(pending),
            submissions_declined=submissions_declined,
            notes=tuple(notes),
        )

    # --- steps ---------------------------------------------------------

    def _recent_window(self) -> int:
        """How many closed bars the fill windows can possibly need.

        Bounded from the config rather than kept unbounded, so a ten-year run
        does not hold its whole history in a deque. Latency, the configured
        entry delay and the wait horizon are all the window can reach, plus
        two bars of slack for the signal bar itself and for a rounding edge.
        """
        execution = self.config.execution
        latency = (float(execution.latency_ms) + float(execution.random_latency_ms)) / 1000.0
        interval = int(self.config.data.primary_interval_seconds)
        offset = min_fill_offset(
            latency_seconds=latency,
            interval_seconds=interval,
            entry_delay_bars=int(execution.entry_delay_bars),
        )
        return offset + int(execution.max_entry_wait_bars) + 2

    def _submit(
        self,
        intent: TradeIntent,
        ts: datetime,
        *,
        decision: EngineDecision,
        volatility: VolatilityTracker,
        atr: float | None,
    ) -> _PendingEntry:
        order = self.execution.submit(intent, ts)
        interval = int(self.config.data.primary_interval_seconds)
        offset = min_fill_offset(
            latency_seconds=(order.eligible_ts - order.signal_ts).total_seconds(),
            interval_seconds=interval,
            entry_delay_bars=order.entry_delay_bars,
        )
        return _PendingEntry(
            order=order,
            context=self._entry_context(decision, atr=atr),
            vol=volatility.context(atr=atr, ts=ts),
            min_offset=offset,
            horizon_bars=offset + order.wait_bars - 1,
        )

    def _entry_context(
        self, decision: EngineDecision, *, atr: float | None
    ) -> EntryContext:
        signal = decision.signal
        features = decision.features
        vol_percentile = None if features is None else features.get("atr_percentile")
        session = Session.CLOSED
        if self.calendar is not None:
            session = self.calendar.session_of(signal.ts, self.spec)
        plan = decision.plan
        return EntryContext(
            flow_score=float(signal.flow_score or 0.0),
            regime=signal.regime,
            component_points=dict(signal.component_points),
            volatility=atr,
            vol_percentile=vol_percentile,
            data_quality=signal.data_quality,
            session=session,
            entry_reason=" | ".join(plan.reasons if plan is not None else signal.reasons),
        )

    def _process_pending(
        self,
        ts: datetime,
        *,
        pending: list[_PendingEntry],
        recent: deque,
        source,
        manager: PositionManager,
        journal: Journal,
        bar: Bar | None,
        quote: QuoteSnapshot | None,
        intrabar: Sequence[Bar],
    ) -> None:
        """Resolve the orders whose whole wait horizon has now closed.

        An order is held until `horizon_bars` bars have closed after its
        signal bar. `FillWindow` treats the last bar it holds as the last
        candidate, so resolving against a partial window would report
        `entry_wait_expired` on a horizon that had not expired -- a miss the
        market never produced. Waiting also means no order can resolve on the
        bar that created it, which is section 9's lookahead invariant.
        """
        if not pending:
            return
        still: list[_PendingEntry] = []
        for entry in pending:
            candidates = tuple(
                b for b in recent if b.close_ts >= entry.order.signal_ts
            )
            after = sum(1 for b in candidates if b.close_ts > entry.order.signal_ts)
            if after < entry.horizon_bars:
                still.append(entry)
                continue
            window = self.execution.open_window(entry.order, candidates)
            fill_quote = None
            if window.bars:
                fill_quote = source.quote(window.bars[0].close_ts)
            fill = self.execution.resolve_entry(
                entry.order, window, vol=entry.vol, quote=fill_quote
            )
            journal.record_order(fill)
            if not fill.filled:
                continue
            position = manager.open_from_fill(entry.order.intent, fill, entry.context)
            # The entry bar is a bar the position was open for: the fill
            # printed at its OPEN, so the stop and the target genuinely came
            # afterwards within it. `resolve_exit` permits an exit on the entry
            # bar for that reason, and skipping the check would let a stop that
            # was hit on the entry bar exit later, at a price the market never
            # offered the position. Only the bar the fill landed on is
            # inspected, and only if the loop has already stepped past it.
            fill_bar = next(
                (b for b in candidates if b.close_ts == fill.fill_ts), None
            )
            if fill_bar is not None:
                manager.resolve_one(
                    position,
                    fill_bar.close_ts,
                    bar=fill_bar,
                    quote=source.quote(fill_bar.close_ts),
                    intrabar=(
                        intrabar
                        if bar is not None and fill_bar.close_ts == bar.close_ts
                        else source.intrabar(fill_bar)
                    ),
                )
        pending[:] = still

    # --- provenance ----------------------------------------------------

    def _provenance(
        self,
        *,
        run_id: str,
        data_hash: str,
        source,
        bars_in_clock: int,
        bars_processed: int,
        first_ts: datetime | None,
        last_ts: datetime | None,
    ) -> RunProvenance:
        commit, dirty = self._git
        if commit is None or dirty is None:
            commit, dirty = _git_state(self.repo_root)
        return RunProvenance(
            run_id=run_id,
            symbol=self.symbol,
            config_hash=self.config.config_hash,
            data_hash=data_hash,
            seed=int(self.config.backtest.seed),
            git_commit=commit,
            git_dirty=bool(dirty),
            code_version=CODE_VERSION,
            split_label=self.split_label,
            split_name=self.split_name,
            created_at=datetime.now(timezone.utc),
            starting_equity=self.starting_equity,
            compound_equity=bool(self.config.risk.compound_equity),
            record_wait_signals=bool(self.config.backtest.record_wait_signals),
            record_equity_every_bar=bool(self.config.backtest.record_equity_every_bar),
            bars_in_clock=bars_in_clock,
            bars_processed=bars_processed,
            max_bars=self.config.backtest.max_bars,
            first_ts=first_ts,
            last_ts=last_ts,
            interval_seconds=int(self.config.data.primary_interval_seconds),
            dataset=dict(source.describe()),
        )


def _git_state(repo_root: str) -> tuple[str, bool]:
    """(commit, dirty), reusing the experiment log's implementation.

    Imported lazily and by function, not by module: `validation/` is the
    layer that owns provenance, and a second copy of two subprocess calls in
    `backtest/` would be a second thing to keep correct.
    """
    from flow_model.validation.experiment_log import git_state

    return git_state(repo_root)
