"""The execution model: section 9's fill and cost model, composed.

The event loop, the ledger and the journal are a separate module; this one
answers three questions and nothing else:

    submit(intent, now)               -> when could this fill at the earliest
    resolve_entry(order, window, ...)  -> did it fill, how much, at what price
    resolve_exit(plan, bar, ...)       -> did the stop or the target get hit

The parts it composes each own one concern, and each states the asymmetry it
is built around:

* `backtest/fill_timing.py` -- eligibility. "Fills resolve on later bars only",
  made structural: `FillWindow` physically excludes the signal bar.
* `backtest/fills.py` -- the per-bar stop/target rules. Stops gap and targets
  do not; stops trigger on a touch and targets need a trade-through;
  unobserved orderings resolve against the trade and are tagged.
* `backtest/costs.py` -- spread, slippage, commissions and fees, itemized into
  `CostBreakdown` and never netted into one number.

## What this module adds on top of them

**Unmeasurable quantities decline the fill.** No bar volume means the
participation limit cannot be evaluated, so the entry is declined rather than
assumed unconstrained -- and unconstrained is cheapest exactly on the thinnest
bars. An unmeasured ATR/median-ATR ratio with a non-zero
`vol_slippage_coefficient` declines the entry rather than charging zero
volatility slippage. `signals/engine.py` carries the scar: there, a missing
ATR "does not degrade the cap, it REMOVES it", and a 20x-ATR stop was accepted
by a setup that caps the stop at 2x.

**A plan the market has already invalidated is declined, not filled.** If a
LONG's fill price has gapped below its own stop then `stop > entry`, the
bracket is inverted, and "stopping out" would book a PROFIT -- a fictional win
manufactured by a gap. Same past the target. Both are refused by name.

**Determinism.** `random_latency_ms` and `random_miss_probability` draw from
NAMED streams via `core/determinism.rng`, keyed by the ORDER's identity
(symbol, signal timestamp, side) rather than by a running counter. Adding a
stochastic term therefore cannot shift an existing one's draws, and no draw
depends on how many orders preceded it, so a re-run, a resumed run and a
single-order unit test all produce the same number. A cost model that is not
reproducible makes every downstream figure unfalsifiable.

**The plan is never changed.** `TradeIntent`'s contract says the execution
model "may fill it worse, partially, or not at all, but it may not change the
plan". Nothing here recomputes a stop, a target or a size from the realized
fill. Section 0.2 -- "R is defined once, at entry, and never revised" -- is
what that protects: a stop re-derived from a worse fill holds R constant by
moving the thesis, and `TradeRecord` carries `planned_entry` beside
`entry_price` so the slippage between them stays visible.

A consequence worth stating plainly, because it surprises: a stopped-out trade
does NOT realize -1R. The entry filled away from `planned_entry` while
`risk_dollars` stays defined by the planned stop distance, so a long filled
two points high on a twenty-point stop realizes about -1.16R once costs are
included. That is the honest number. A backtest reporting -1.00R on every stop
has quietly re-derived R from the fill.
"""

from __future__ import annotations

import math
from datetime import datetime

from flow_model.backtest.costs import (
    CostModel,
    VolatilityContext,
    adverse_price,
    combine_costs,
)
from flow_model.backtest.fill_timing import (
    ExecutionError,
    FillTimingError,
    FillWindow,
    PendingOrder,
    min_fill_offset,
)
from flow_model.backtest.fills import (
    BarResolution,
    EntryFill,
    ExitFill,
    ExitPlan,
    FillStatus,
    MissReason,
    resolve_bar_exit,
    validated_intrabar,
)
from flow_model.core.contracts import Bar, CostBreakdown, QuoteSnapshot, TradeIntent
from flow_model.core.determinism import rng
from flow_model.core.enums import ExitReason, Side
from flow_model.core.instruments import InstrumentSpec

__all__ = [
    "BarResolution",
    "EntryFill",
    "ExecutionError",
    "ExecutionModel",
    "ExitFill",
    "ExitPlan",
    "FillStatus",
    "FillTimingError",
    "FillWindow",
    "MissReason",
    "PendingOrder",
    "min_fill_offset",
    "resolve_bar_exit",
    "validated_intrabar",
]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class ExecutionModel:
    """Section 9's fill and cost model. Config plus an `InstrumentSpec`.

    No market data in the constructor, and no accumulated state: every method
    is a function of its arguments plus config, so a replay and a single call
    at the same timestamp produce the same fill. `regime/detector.py` records
    why that matters -- a stateful component there "would have made backtest
    results depend on evaluation order, and that was caught as a HIGH
    finding".
    """

    __slots__ = ("execution", "spec", "costs", "master_seed")

    #: Names of the seeded streams this model draws from. Each is derived per
    #: ORDER, so adding a stream cannot shift another's draws and a draw never
    #: depends on how many orders preceded it.
    STREAM_LATENCY = "random_latency"
    STREAM_MISS = "random_miss"
    STREAM_ROOT = "execution"

    def __init__(self, execution, spec: InstrumentSpec, *, master_seed: int) -> None:
        """`execution` is an `ExecutionConfig`; `master_seed` seeds the streams.

        Refuses, at construction, a resting entry with a zero-bar horizon.
        `ExecutionConfig`'s own validator catches that for `entry_order_type
        ="limit"` ("a limit entry with max_entry_wait_bars=0 can never fill")
        but not for `"stop"`, which has the same property for the same
        reason. Raised here rather than silently returning
        `no_eligible_bar` on every bar of a ten-year run, following
        `risk/limits.py`: "a config that states an impossible intention
        should fail loudly at construction rather than be silently clamped on
        every bar".
        """
        if execution.entry_order_type in ("limit", "stop") and execution.max_entry_wait_bars == 0:
            raise ExecutionError(
                f"entry_order_type={execution.entry_order_type!r} with "
                "max_entry_wait_bars=0 is an order that can never fill: a resting "
                "entry needs at least one bar in which price can trade through it"
            )
        self.execution = execution
        self.spec = spec
        self.costs = CostModel(execution, spec)
        self.master_seed = int(master_seed)

    @classmethod
    def from_config(cls, config, symbol: str) -> "ExecutionModel":
        """Build from a `FlowModelConfig` and a symbol.

        Takes `config.backtest.seed` as the master seed: these streams are the
        backtester's, and section 9 is the backtester.
        """
        return cls(
            config.execution,
            config.spec(symbol),
            master_seed=config.backtest.seed,
        )

    # --- streams -------------------------------------------------------

    def _stream(self, name: str, intent: TradeIntent):
        """A generator named for this order and this purpose.

        Keyed by the order's identity rather than by a counter so the draw is
        a pure function of the order. `core/determinism.rng` guarantees
        independence between names.
        """
        return rng(
            self.master_seed,
            self.STREAM_ROOT,
            name,
            intent.symbol,
            intent.signal_ts.isoformat(),
            intent.side.value,
        )

    # --- submission ----------------------------------------------------

    def submit(self, intent: TradeIntent, now: datetime) -> PendingOrder:
        """Turn an intent into a pending order with an eligibility instant.

        `now` is the instant the order is sent, which is the signal bar's
        close in the backtest loop. The latency jitter is drawn here, once per
        order, and only when `random_latency_ms` is non-zero -- the default
        path performs no draw at all, so the default backtest is reproducible
        without reference to any generator.
        """
        if now < intent.signal_ts:
            raise FillTimingError(
                f"submitting at {now.isoformat()} precedes the signal at "
                f"{intent.signal_ts.isoformat()}"
            )
        if intent.symbol != self.spec.symbol:
            raise ExecutionError(
                f"intent for {intent.symbol!r} submitted to a model for "
                f"{self.spec.symbol!r}"
            )
        notes: list[str] = []
        jitter = 0.0
        spread_ms = float(self.execution.random_latency_ms)
        if spread_ms > 0.0:
            draw = float(self._stream(self.STREAM_LATENCY, intent).uniform(0.0, spread_ms))
            jitter = draw / 1000.0
            notes.append(
                f"random_latency_ms={spread_ms:g} drew {draw:.3f}ms from the "
                f"'{self.STREAM_ROOT}/{self.STREAM_LATENCY}' stream for this order"
            )
        return PendingOrder(
            intent=intent,
            submitted_ts=now,
            order_type=self.execution.entry_order_type,
            latency_seconds=float(self.execution.latency_ms) / 1000.0,
            jitter_seconds=jitter,
            entry_delay_bars=int(self.execution.entry_delay_bars),
            max_entry_wait_bars=int(self.execution.max_entry_wait_bars),
            notes=tuple(notes),
        )

    def open_window(self, order: PendingOrder, bars) -> FillWindow:
        """The bars this order may fill from. See `FillWindow`."""
        return FillWindow(order, bars)

    # --- entries -------------------------------------------------------

    def resolve_entry(
        self,
        order: PendingOrder,
        window: FillWindow,
        *,
        vol: VolatilityContext,
        quote: QuoteSnapshot | None = None,
    ) -> EntryFill:
        """Fill, partially fill, or miss one entry.

        The probabilistic miss is drawn FIRST, before any price or volume is
        looked at. Applying it after would make the stress-test miss rate
        conditional on the outcome -- the orders that survive the price checks
        are not a random sample of orders -- and the resulting bias would run
        in whichever direction the data happened to point.
        """
        intent = order.intent
        if window.symbol != intent.symbol:
            raise ExecutionError(
                f"window for {window.symbol!r} passed with an intent for "
                f"{intent.symbol!r}"
            )
        # A `FillWindow` is derived from ONE order: its eligibility floor and
        # its wait horizon are that order's. Resolving a different order
        # against it reuses the wrong floor, and the natural event-loop
        # optimization -- build one window per bar, resolve every pending
        # order against it -- produces exactly that pairing. The result is a
        # fill that lands before the order was eligible, at a price the order
        # could not have reached, and nothing downstream can see it because
        # every individual record is internally consistent.
        if (
            window.signal_ts != order.signal_ts
            or window.eligible_ts != order.eligible_ts
            or window.wait_bars != order.wait_bars
        ):
            raise FillTimingError(
                f"{intent.symbol}: this window was built for a different order "
                f"(window signal={window.signal_ts.isoformat()} "
                f"eligible={window.eligible_ts.isoformat()} wait={window.wait_bars}; "
                f"order signal={order.signal_ts.isoformat()} "
                f"eligible={order.eligible_ts.isoformat()} wait={order.wait_bars}). "
                "A window carries its order's eligibility floor, so a crossed pair "
                "fills at a bar the order had not yet reached"
            )
        miss = self._missed(order, intent)
        if miss is not None:
            return miss

        for offset, bar in window.enumerate_offsets():
            attempt = self._try_bar(
                order=order,
                bar=bar,
                offset=offset,
                vol=vol,
                quote=quote,
                last_candidate=(offset == window.first_offset + len(window) - 1),
            )
            if attempt is not None:
                return attempt

        # Reached when the window held no candidate bar at all. Every
        # non-empty window ends in a fill or an explicit miss inside the loop,
        # so this is the "the data ran out" path and not a silent default.
        return self._miss(
            order,
            MissReason.NO_ELIGIBLE_BAR,
            f"no bar closes after the signal at {order.signal_ts.isoformat()} within "
            f"the order's {window.wait_bars}-bar horizon starting at offset "
            f"{window.min_offset}, so there was nowhere a fill could legally resolve",
            *window.notes,
        )

    def _missed(self, order: PendingOrder, intent: TradeIntent) -> EntryFill | None:
        probability = float(self.execution.random_miss_probability)
        if probability <= 0.0:
            return None
        draw = float(self._stream(self.STREAM_MISS, intent).random())
        if draw >= probability:
            return None
        return self._miss(
            order,
            MissReason.RANDOM_MISS,
            f"random_miss_probability={probability:g} and the "
            f"'{self.STREAM_ROOT}/{self.STREAM_MISS}' stream drew {draw:.6f} for this "
            "order. Drawn before any price or volume was inspected, so the miss is "
            "independent of the outcome it would have had",
        )

    def _try_bar(
        self,
        *,
        order: PendingOrder,
        bar: Bar,
        offset: int,
        vol: VolatilityContext,
        quote: QuoteSnapshot | None,
        last_candidate: bool,
    ) -> EntryFill | None:
        """One candidate bar: a fill, a definite miss, or None to keep waiting.

        None means "this bar did not trigger, try the next one". A returned
        MISSED fill means the attempt is over -- insufficient volume or an
        invalidated plan is not something a later bar repairs by chance.
        """
        intent = order.intent
        notes: list[str] = []
        passive = False

        if order.order_type == "market":
            reference = bar.open
            notes.append(
                f"market entry at the open of the bar closing {bar.close_ts.isoformat()} "
                f"(offset {offset}); the signal bar's own prices are unreachable from "
                "the fill window"
            )
        elif order.order_type == "limit":
            limit = intent.entry_price
            through = bar.low < limit if intent.side is Side.LONG else bar.high > limit
            if not through:
                if not last_candidate:
                    return None
                return self._miss(
                    order,
                    MissReason.LIMIT_NOT_TRADED_THROUGH,
                    f"price never traded through the {intent.side.value} limit {limit:g} "
                    f"within max_entry_wait_bars={order.max_entry_wait_bars}; the last "
                    f"candidate bar spanned [{bar.low:g}, {bar.high:g}]. A touch alone "
                    "is not treated as a fill: queue position is unobservable from bar "
                    "data",
                )
            reference = limit
            passive = True
            notes.append(
                f"resting {intent.side.value} limit at {limit:g} traded through on the "
                f"bar closing {bar.close_ts.isoformat()} (offset {offset}); filled at "
                "the limit, which is the best a limit can do"
            )
        else:  # stop entry
            trigger = intent.entry_price
            touched = bar.high >= trigger if intent.side is Side.LONG else bar.low <= trigger
            if not touched:
                if not last_candidate:
                    return None
                return self._miss(
                    order,
                    MissReason.STOP_NOT_TRIGGERED,
                    f"the {intent.side.value} stop entry at {trigger:g} was never "
                    f"touched within max_entry_wait_bars={order.max_entry_wait_bars}; "
                    f"the last candidate bar spanned [{bar.low:g}, {bar.high:g}]",
                )
            gapped = bar.open >= trigger if intent.side is Side.LONG else bar.open <= trigger
            reference = bar.open if gapped else trigger
            if gapped:
                notes.append(
                    f"stop entry at {trigger:g} was already through at the open "
                    f"{bar.open:g}: filled at the OPEN. The gap belongs in the price "
                    "path, not hidden behind the trigger price"
                )
            else:
                notes.append(
                    f"stop entry triggered at {trigger:g} on the bar closing "
                    f"{bar.close_ts.isoformat()} (offset {offset})"
                )

        # --- the plan may already be dead on arrival.
        invalidation = self._plan_invalidated(intent, reference)
        if invalidation is not None:
            reason, detail = invalidation
            return self._miss(order, reason, detail, *notes)

        # --- participation and partial fills.
        sized = self._fillable_size(order, bar)
        if isinstance(sized, EntryFill):
            return sized
        filled_size, size_notes = sized
        notes.extend(size_notes)

        # --- costs. A refusal to price is a refusal to fill.
        is_stop_order = order.order_type == "stop"
        slippage = self.costs.slippage(
            size=filled_size, vol=vol, is_stop_order=is_stop_order, passive=passive
        )
        if not slippage.measurable:
            return self._miss(
                order, MissReason.COST_UNMEASURABLE, slippage.refusal, *slippage.notes, *notes
            )
        spread = self.costs.spread(quote, fill_ts=bar.close_ts)
        costs = self.costs.entry_costs(size=filled_size, spread=spread, slippage=slippage)

        charged_ticks = slippage.ticks + (0.0 if slippage.passive else spread.half_ticks)
        effective = adverse_price(
            reference, charged_ticks, self.spec.tick_size, against_sign=intent.side.sign
        )
        lag = max(0.0, (order.eligible_ts - bar.open_ts).total_seconds())
        if lag > 0.0 and order.order_type == "market":
            notes.append(
                f"the fill bar had been open {lag:g}s when the order became eligible, so "
                "its OPEN print marginally predates eligibility. Bar data holds no finer "
                "price; the lag is recorded rather than asserted negligible"
            )

        full = filled_size >= intent.size - 1e-9
        return EntryFill(
            status=FillStatus.FILLED if full else FillStatus.PARTIAL,
            symbol=intent.symbol,
            side=intent.side,
            signal_ts=intent.signal_ts,
            submitted_ts=order.submitted_ts,
            eligible_ts=order.eligible_ts,
            order_type=order.order_type,
            fill_ts=bar.close_ts,
            fill_bar_offset=offset,
            reference_price=reference,
            effective_price=effective,
            planned_price=intent.entry_price,
            requested_size=intent.size,
            filled_size=filled_size,
            participation_cap=round(
                float(self.execution.max_participation_rate) * bar.volume, 10
            ),
            bar_volume=bar.volume,
            costs=costs,
            spread=spread,
            slippage=slippage,
            eligibility_lag_seconds=lag,
            notes=tuple(order.notes) + tuple(notes),
        )

    def _plan_invalidated(
        self, intent: TradeIntent, reference: float
    ) -> tuple[str, str] | None:
        """Refuse a fill whose price has already broken the plan's geometry.

        A LONG filled at or below its stop has `stop >= entry`: the bracket is
        inverted, and "stopping out" would book a PROFIT at the stop. The
        trade would appear in the record as a win that the market never
        offered. The same inversion happens past the target, where the reward
        the plan was sized for has already been taken by the gap.

        Declined rather than filled-and-adjusted, because adjusting means
        moving the stop or the target, and `TradeIntent` forbids this module
        from changing the plan.
        """
        side = intent.side
        if side is Side.LONG:
            if reference <= intent.stop_price:
                return (
                    MissReason.FILL_BEYOND_STOP,
                    f"the fill reference {reference:g} is at or below the LONG stop "
                    f"{intent.stop_price:g}, so the plan's geometry is inverted and its "
                    "stop sits above its entry. Filling it would let the position "
                    "'stop out' at a profit",
                )
            if reference >= intent.target_price:
                return (
                    MissReason.FILL_BEYOND_TARGET,
                    f"the fill reference {reference:g} is at or above the LONG target "
                    f"{intent.target_price:g}: the move the plan was sized for has "
                    "already happened and the remaining reward/risk is not the setup's",
                )
            return None
        if reference >= intent.stop_price:
            return (
                MissReason.FILL_BEYOND_STOP,
                f"the fill reference {reference:g} is at or above the SHORT stop "
                f"{intent.stop_price:g}, so the plan's geometry is inverted",
            )
        if reference <= intent.target_price:
            return (
                MissReason.FILL_BEYOND_TARGET,
                f"the fill reference {reference:g} is at or below the SHORT target "
                f"{intent.target_price:g}: the planned move has already happened",
            )
        return None

    def _fillable_size(self, order: PendingOrder, bar: Bar):
        """Section 9: `min(size, participation_rate * bar_volume)`.

        Returns `(size, notes)` or a MISSED `EntryFill`. The refusals here are
        the ones the brief's "a partial fill treated as whole" clause is
        about, plus the unmeasurable case: a bar with no measured volume gives
        no participation limit, there is no configured volume fallback to
        degrade to, so the fill is declined. Assuming unconstrained liquidity
        on a bar whose volume was never recorded is the cheap guess, and it is
        cheapest exactly on thin bars.
        """
        intent = order.intent
        requested = intent.size
        notes: list[str] = []

        if not math.isfinite(bar.volume) or bar.volume <= 0.0:
            return self._miss(
                order,
                MissReason.NO_VOLUME_MEASUREMENT,
                f"the bar closing {bar.close_ts.isoformat()} reports volume "
                f"{bar.volume!r}, so max_participation_rate="
                f"{self.execution.max_participation_rate:g} yields no fillable size. No "
                "volume fallback is configured and none is assumed: an unconstrained "
                "fill on an unmeasured bar is free liquidity on exactly the bars least "
                "likely to have any",
            )

        cap = float(self.execution.max_participation_rate) * bar.volume
        filled = min(requested, cap)
        if filled < requested:
            notes.append(
                f"participation cap {cap:g} ({self.execution.max_participation_rate:g} "
                f"of {bar.volume:g} volume) is below the requested {requested:g}"
            )
        if filled > self.spec.max_size:
            notes.append(
                f"clamped to the instrument's max_size {self.spec.max_size:g} from "
                f"{filled:g}"
            )
            filled = float(self.spec.max_size)

        if not self.spec.allow_fractional_size:
            floored = float(math.floor(filled + 1e-9))
            if floored != filled:
                notes.append(
                    f"{self.spec.symbol} does not trade fractional size; {filled:g} "
                    f"floored to {floored:g}"
                )
            filled = floored

        if filled < self.spec.min_size or filled <= 0.0:
            return self._miss(
                order,
                MissReason.SIZE_ROUNDS_TO_ZERO,
                f"the fillable size came to {filled:g}, below {self.spec.symbol}'s "
                f"min_size {self.spec.min_size:g}",
                *notes,
            )

        if filled < requested - 1e-9:
            if not self.execution.allow_partial_fills:
                return self._miss(
                    order,
                    MissReason.PARTIAL_FILLS_DISABLED,
                    f"only {filled:g} of {requested:g} fits under the participation cap "
                    "and allow_partial_fills is False. The order is abandoned rather "
                    "than filled whole: filling whole would take more of the bar's "
                    "volume than the cap says we can",
                    *notes,
                )
            fraction = filled / requested
            minimum = float(self.execution.min_partial_fraction)
            if fraction < minimum:
                return self._miss(
                    order,
                    MissReason.PARTICIPATION_BELOW_MINIMUM,
                    f"fill fraction {fraction:.4f} is below min_partial_fraction "
                    f"{minimum:g}, so the entry is abandoned as its own field describes",
                    *notes,
                )
            notes.append(
                f"PARTIAL fill: {filled:g} of {requested:g} ({fraction:.4f}). The "
                "position is smaller than the plan and R is correspondingly smaller; "
                "nothing here rescales the stop to restore it"
            )
        return filled, notes

    def _miss(self, order: PendingOrder, reason: str, *details: str) -> EntryFill:
        intent = order.intent
        return EntryFill(
            status=FillStatus.MISSED,
            symbol=intent.symbol,
            side=intent.side,
            signal_ts=intent.signal_ts,
            submitted_ts=order.submitted_ts,
            eligible_ts=order.eligible_ts,
            order_type=order.order_type,
            planned_price=intent.entry_price,
            requested_size=intent.size,
            miss_reason=reason,
            notes=tuple(order.notes) + tuple(d for d in details if d),
        )

    # --- exits ---------------------------------------------------------

    def resolve_exit(
        self,
        plan: ExitPlan,
        bar: Bar,
        *,
        vol: VolatilityContext,
        intrabar=None,
        quote: QuoteSnapshot | None = None,
    ) -> ExitFill:
        """Whether this bar closed the position, and at what cost.

        The bar must close strictly after the entry; an exit on the entry bar
        itself is allowed (the entry was at that bar's open, so the stop and
        the target genuinely came afterwards) and is not allowed to precede
        the entry.

        `intrabar` is section 9's finer-granularity resolution. When usable it
        decides the stop/target ordering by observation and the trade is NOT
        tagged ambiguous; when the finest bar is itself ambiguous the
        pessimistic rule applies at that granularity and the tag is set there.
        """
        if bar.symbol != plan.symbol:
            raise ExecutionError(
                f"bar for {bar.symbol!r} passed to an exit for {plan.symbol!r}"
            )
        if not bar.is_valid:
            raise ExecutionError(
                f"{plan.symbol}: structurally invalid bar at {bar.close_ts.isoformat()}"
            )
        if bar.close_ts < plan.entry_ts:
            raise FillTimingError(
                f"{plan.symbol}: exit bar closes {bar.close_ts.isoformat()}, before the "
                f"entry at {plan.entry_ts.isoformat()}"
            )

        notes: list[str] = []
        finer, coverage_notes = validated_intrabar(plan, bar, intrabar)
        notes.extend(coverage_notes)

        resolution: BarResolution | None = None
        used = 0
        if finer:
            for fine in finer:
                used += 1
                resolution = resolve_bar_exit(
                    plan,
                    fine,
                    pessimistic_same_bar=bool(self.execution.pessimistic_same_bar),
                    fill_gaps_at_open=bool(self.execution.fill_gaps_at_open),
                )
                if resolution is not None:
                    notes.append(
                        f"resolved on finer {fine.interval_seconds}s bar "
                        f"{used}/{len(finer)} closing {fine.close_ts.isoformat()}, so the "
                        "stop/target ordering was observed rather than assumed"
                    )
                    break
            if resolution is None:
                used = len(finer)
                coarse = resolve_bar_exit(
                    plan,
                    bar,
                    pessimistic_same_bar=bool(self.execution.pessimistic_same_bar),
                    fill_gaps_at_open=bool(self.execution.fill_gaps_at_open),
                )
                if coarse is not None:
                    notes.append(
                        "the finer bars resolved nothing but the parent bar's range "
                        "reaches a level: the two series disagree, so the parent's "
                        "pessimistic resolution is used and the fill is tagged ambiguous"
                    )
                    resolution = coarse.replace(ambiguous=True)
        else:
            resolution = resolve_bar_exit(
                plan,
                bar,
                pessimistic_same_bar=bool(self.execution.pessimistic_same_bar),
                fill_gaps_at_open=bool(self.execution.fill_gaps_at_open),
            )

        if resolution is None:
            return ExitFill(
                resolved=False,
                symbol=plan.symbol,
                side=plan.side,
                size=plan.size,
                intrabar_bars_used=used,
                notes=tuple(notes),
            )

        is_stop = resolution.reason is ExitReason.STOP
        return self._price_exit(
            plan=plan,
            bar=bar,
            reason=resolution.reason,
            reference=resolution.price,
            is_stop_order=is_stop,
            gapped=resolution.gapped,
            ambiguous=resolution.ambiguous,
            vol=vol,
            quote=quote,
            intrabar_bars_used=used,
            notes=tuple(notes) + resolution.notes,
        )

    def exit_at_close(
        self,
        plan: ExitPlan,
        bar: Bar,
        reason: ExitReason,
        *,
        vol: VolatilityContext,
        quote: QuoteSnapshot | None = None,
    ) -> ExitFill:
        """Price a forced flatten at a bar's close.

        For the exits the LEDGER decides -- a time stop, a session flatten, a
        risk limit, the end of the backtest. `STOP` and `TARGET` are refused
        here: routing them through this path would skip the gap rule and the
        same-bar rule, which is precisely how a stop comes to be filled at its
        limit price.
        """
        if reason in (ExitReason.STOP, ExitReason.TARGET):
            raise ExecutionError(
                f"{reason.value} must be resolved by resolve_exit, which applies the gap "
                "and same-bar rules. Pricing it at a bar close would skip both"
            )
        if bar.symbol != plan.symbol:
            raise ExecutionError(
                f"bar for {bar.symbol!r} passed to an exit for {plan.symbol!r}"
            )
        # The same check `resolve_exit` makes, for a stronger reason: this path
        # prices the exit at `bar.close` directly, so a structurally invalid
        # bar does not merely resolve nothing -- it becomes the exit PRICE. A
        # NaN close produced a NaN fill price carrying a real dollar cost, and
        # the ledger would have marked a position out at it.
        if not bar.is_valid:
            raise ExecutionError(
                f"{plan.symbol}: structurally invalid bar at {bar.close_ts.isoformat()} "
                "(OHLC ordering or volume); its close cannot be used as an exit price"
            )
        if bar.close_ts < plan.entry_ts:
            raise FillTimingError(
                f"{plan.symbol}: flatten bar closes {bar.close_ts.isoformat()}, before "
                f"the entry at {plan.entry_ts.isoformat()}"
            )
        return self._price_exit(
            plan=plan,
            bar=bar,
            reason=reason,
            reference=bar.close,
            is_stop_order=False,
            gapped=False,
            ambiguous=False,
            vol=vol,
            quote=quote,
            intrabar_bars_used=0,
            notes=(
                f"{reason.value} priced at the close {bar.close:g} of the bar ending "
                f"{bar.close_ts.isoformat()}, as a market order",
            ),
        )

    def _price_exit(
        self,
        *,
        plan: ExitPlan,
        bar: Bar,
        reason: ExitReason,
        reference: float,
        is_stop_order: bool,
        gapped: bool,
        ambiguous: bool,
        vol: VolatilityContext,
        quote: QuoteSnapshot | None,
        intrabar_bars_used: int,
        notes: tuple[str, ...],
    ) -> ExitFill:
        """Attach costs to a resolved exit.

        A target exit is a resting limit and pays no slippage and no spread,
        for the reason `costs.slippage` gives -- including when the bar GAPPED
        through it, since a limit that the market opens past still fills at
        the limit and is still the resting side of the book.  A stop exit is a
        market order and additionally pays
        `stop_order_extra_slippage_ticks`.

        An unmeasurable cost CANNOT decline an exit the way it declines an
        entry: the position is already open and the price has already reached
        the level, so refusing to exit would hold a position the market closed.
        It raises instead, because a backtest that cannot price its exits must
        stop rather than quietly book them free.
        """
        passive = reason is ExitReason.TARGET
        slippage = self.costs.slippage(
            size=plan.size, vol=vol, is_stop_order=is_stop_order, passive=passive
        )
        if not slippage.measurable:
            raise ExecutionError(
                f"{plan.symbol}: cannot price the {reason.value} exit at "
                f"{bar.close_ts.isoformat()} -- {slippage.refusal}. The position is "
                "already open and the level is already reached, so this exit cannot be "
                "declined the way an entry can; it is raised rather than booked at zero "
                "cost"
            )
        spread = self.costs.spread(quote, fill_ts=bar.close_ts)
        costs = self.costs.exit_costs(size=plan.size, spread=spread, slippage=slippage)
        charged_ticks = slippage.ticks + (0.0 if slippage.passive else spread.half_ticks)
        # Exiting is the opposite trade: a LONG sells out, so a LOWER price hurts.
        effective = adverse_price(
            reference, charged_ticks, self.spec.tick_size, against_sign=-plan.side.sign
        )
        extra: tuple[str, ...] = ()
        if passive:
            extra = (
                "target exit treated as a resting limit: no spread crossed and no "
                "slippage charged, because a limit cannot fill worse than its price",
            )
        return ExitFill(
            resolved=True,
            symbol=plan.symbol,
            side=plan.side,
            exit_reason=reason,
            fill_ts=bar.close_ts,
            reference_price=reference,
            effective_price=effective,
            size=plan.size,
            gapped=gapped,
            ambiguous_fill=ambiguous,
            intrabar_bars_used=intrabar_bars_used,
            costs=costs,
            spread=spread,
            slippage=slippage,
            notes=notes + extra,
        )

    # --- round turn ----------------------------------------------------

    def round_turn_costs(self, entry: EntryFill, exit_fill: ExitFill) -> CostBreakdown:
        """One trade's costs, itemized, from its two sides.

        The field-wise sum of the two breakdowns -- never a single total.
        Section 12 needs cost drag attributable to a term, and
        `TradeRecord.cost_drag_r` divides this by R to produce the win rate
        the trade must beat to break even.
        """
        if not entry.filled:
            raise ExecutionError("a missed entry has no round turn")
        if not exit_fill.resolved:
            raise ExecutionError("an unresolved exit has no round turn")
        return combine_costs(entry.costs, exit_fill.costs)
