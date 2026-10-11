"""Fill outcomes, and the per-bar rules that decide them.

ARCHITECTURE.md section 9's stop/target resolution, as pure functions over one
`ExitPlan` and one `Bar`, plus the frozen records the `ExecutionModel` returns.

## Each rule is asymmetric, and the asymmetry is the content

Every clause in section 9 is a place where an error has a DIRECTION, and the
flattering direction is the dangerous one. These four are the whole reason
this module is not three lines of `if`:

* **Stops gap, targets do not.** A bar opening through the stop fills at the
  OPEN -- section 9 calls filling at the stop instead "the single most common
  backtest overstatement". A bar opening through the TARGET fills at the
  target, NOT at the better open: a target is a resting limit order and a
  resting limit receives its limit price. One of those two gaps is in the
  trade's favour and it is the one not taken.
* **Stops trigger on a touch; targets need a trade-through.** A stop is a
  market order fired by a print at or through its level, so `low <= stop`
  fires it. A limit at the target needs `high > target`, strictly: a print AT
  our limit does not prove our order filled, because queue position is
  unobservable from bar data. Inclusive for the stop, strict for the target.
* **Ambiguity resolves against the trade and is TAGGED.** When a bar's range
  holds both levels and no finer bars exist, the touch order was never
  observed. `pessimistic_same_bar` (default true) picks the stop, and
  `ambiguous=True` is set under EITHER setting -- the tag records the missing
  observation, not the configuration. Untagging the optimistic setting would
  make a run's sensitivity to the assumption unmeasurable, which is the
  opposite of what section 9 asks the tag for.
* **A gap is not an ambiguity.** The open is chronologically first, so a bar
  that opens through a level resolved there and nothing later in the bar can
  change it. Gap fills are tagged `gapped`, never `ambiguous`, so the
  ambiguous count stays a count of genuinely unobserved orderings.

## Intrabar resolution, and when finer bars are not "available"

Section 9: "Intrabar resolution uses finer-granularity bars when available;
otherwise the pessimistic rule is applied". Available has to mean usable.
A finer series that fails to reach the parent bar's own extremes covers only
part of it, and resolving on it decides every ambiguity by whichever half
happened to be supplied -- so it is rejected, the pessimistic rule applies,
and the trade is tagged. The same test applies to TIME: a series that touches
both of the parent's extremes but leaves a hole in its span has not observed
the ordering either, and a resolution "by observation" would clear
`ambiguous_fill` and hide that. A finer series reaching FURTHER in PRICE than
the parent is used, because it shows the larger excursion, and the
disagreement is reported as a data-quality finding rather than smoothed over.
"""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import Field, model_validator

from flow_model.backtest.costs import SlippageEstimate, SpreadEstimate
from flow_model.backtest.fill_timing import ExecutionError, FillTimingError
from flow_model.core.contracts import Bar, CostBreakdown, TradeIntent
from flow_model.core.enums import ExitReason, Side, StrEnum
from flow_model.core.model import FrozenModel

__all__ = [
    "BarResolution",
    "EntryFill",
    "ExitFill",
    "ExitPlan",
    "FillStatus",
    "MissReason",
    "resolve_bar_exit",
    "validated_intrabar",
]


class FillStatus(StrEnum):
    """Outcome of one entry attempt."""

    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    MISSED = "MISSED"


class MissReason(StrEnum):
    """Why an entry was not taken, one code per cause.

    Distinct codes rather than one `missed` flag, for the same reason
    `risk/sizing.SizingRejection` keeps them apart: section 12's rejection
    analysis asks how often each cause fires, which is unanswerable when
    every refusal reports the same thing.

    These are ENTRY outcomes and are deliberately NOT `ExitReason` values.
    `core/enums.ExitReason` enumerates how a position CLOSED; an order that
    never filled opened no position and has nothing to close.
    """

    NO_ELIGIBLE_BAR = "no_eligible_bar"
    LIMIT_NOT_TRADED_THROUGH = "limit_not_traded_through"
    STOP_NOT_TRIGGERED = "stop_not_triggered"
    RANDOM_MISS = "random_miss"
    NO_VOLUME_MEASUREMENT = "no_volume_measurement"
    PARTICIPATION_BELOW_MINIMUM = "participation_below_minimum"
    PARTIAL_FILLS_DISABLED = "partial_fills_disabled"
    SIZE_ROUNDS_TO_ZERO = "size_rounds_to_zero"
    COST_UNMEASURABLE = "cost_unmeasurable"
    FILL_BEYOND_STOP = "fill_beyond_stop"
    FILL_BEYOND_TARGET = "fill_beyond_target"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class EntryFill(FrozenModel):
    """What happened to one entry attempt.

    `reference_price` is what belongs in `TradeRecord.entry_price`: a price
    off the price path, with spread and slippage charged in `costs` instead of
    folded into it (see `backtest/costs.py` on why both at once double-charges
    and either alone under-reports). `effective_price` is the diagnostic
    including those costs, and is not the recorded fill.
    """

    status: FillStatus
    symbol: str
    side: Side
    signal_ts: datetime
    submitted_ts: datetime
    eligible_ts: datetime
    order_type: str

    fill_ts: datetime | None = Field(
        default=None,
        description=(
            "The fill bar's CLOSE. A market fill's reference price is that bar's "
            "OPEN, so the recorded instant is up to one bar later than the price "
            "it carries -- bar data cannot say where in the bar the order filled. "
            "The close is taken rather than the open because the open of the first "
            "eligible bar equals the signal instant, and an entry_ts equal to "
            "signal_ts is indistinguishable from the same-bar fill section 9 "
            "forbids. The cost is a holding time overstated by under one bar; the "
            "alternative hides the invariant this module exists to keep."
        ),
    )
    fill_bar_offset: int = Field(ge=0, default=0)
    reference_price: float | None = None
    effective_price: float | None = None
    planned_price: float = Field(gt=0)

    requested_size: float = Field(gt=0)
    filled_size: float = Field(ge=0.0, default=0.0)
    participation_cap: float | None = None
    bar_volume: float | None = None

    costs: CostBreakdown = Field(default_factory=CostBreakdown)
    spread: SpreadEstimate | None = None
    slippage: SlippageEstimate | None = None

    eligibility_lag_seconds: float = Field(
        ge=0.0,
        default=0.0,
        description=(
            "How long the fill bar had been open when the order became eligible. "
            "Non-zero means the bar's OPEN print, used as the fill reference, "
            "slightly predates eligibility -- the one residual bar data cannot "
            "remove. Recorded per fill so a latency sweep measures it."
        ),
    )
    ambiguous_fill: bool = False
    miss_reason: str = ""
    notes: tuple[str, ...] = ()

    @property
    def filled(self) -> bool:
        return self.status in (FillStatus.FILLED, FillStatus.PARTIAL)

    @property
    def fill_fraction(self) -> float:
        return round(self.filled_size / self.requested_size, 10)

    @model_validator(mode="after")
    def _check(self) -> "EntryFill":
        for name in ("reference_price", "effective_price"):
            price = getattr(self, name)
            if price is not None and not math.isfinite(price):
                raise ValueError(
                    f"{name} is {price!r}: a non-finite fill price is not a price. "
                    "It survives every downstream arithmetic check silently -- "
                    "`math.isclose(nan, nan)` is False, so TradeRecord's own "
                    "re-derivation of gross_pnl rejects the row long after the "
                    "ledger has already marked a position out at it"
                )
        if self.status is FillStatus.MISSED:
            if not self.miss_reason:
                raise ValueError("a MISSED fill must state a miss_reason")
            if self.filled_size != 0.0:
                raise ValueError("a MISSED fill cannot have a filled size")
            if self.fill_ts is not None or self.reference_price is not None:
                raise ValueError("a MISSED fill has no fill timestamp or price")
            if self.costs.total != 0.0:
                raise ValueError(
                    "a MISSED fill costs nothing: an order that never filled paid "
                    "no commission and crossed no spread"
                )
            return self
        if self.miss_reason:
            raise ValueError("a filled order must not state a miss_reason")
        if self.fill_ts is None or self.reference_price is None:
            raise ValueError("a filled order must carry a fill timestamp and price")
        if self.filled_size <= 0.0:
            raise ValueError("a filled order must have a positive filled size")
        if self.filled_size > self.requested_size + 1e-9:
            raise ValueError("filled_size exceeds the requested size")
        if self.fill_ts <= self.signal_ts:
            raise FillTimingError(
                f"fill at {self.fill_ts.isoformat()} is at or before the signal at "
                f"{self.signal_ts.isoformat()}: section 9 requires fills to resolve "
                "on later bars only"
            )
        if self.fill_bar_offset < 1:
            raise FillTimingError(
                f"fill_bar_offset {self.fill_bar_offset} would place the fill on the "
                "signal bar"
            )
        if self.fill_ts < self.eligible_ts:
            raise FillTimingError(
                f"fill at {self.fill_ts.isoformat()} resolves on a bar that closed "
                f"before the order became eligible at {self.eligible_ts.isoformat()}: "
                "latency that an order is not held to is latency that was not modelled"
            )
        if self.status is FillStatus.FILLED and self.filled_size < self.requested_size - 1e-9:
            raise ValueError("a FILLED order filled short; its status should be PARTIAL")
        if self.status is FillStatus.PARTIAL and self.filled_size >= self.requested_size - 1e-9:
            raise ValueError("a PARTIAL order filled in full; its status should be FILLED")
        return self


class ExitPlan(FrozenModel):
    """The stop/target bracket of an open position, as the resolver reads it.

    Built from a `TradeIntent` and the size actually filled. The prices are
    COPIED from the intent and are never recomputed here: `TradeIntent` says
    the execution model "may not change the plan", and section 0.2 fixes R at
    entry.
    """

    symbol: str
    side: Side
    entry_ts: datetime
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    target_price: float = Field(gt=0)
    size: float = Field(gt=0)

    @classmethod
    def from_intent(
        cls, intent: TradeIntent, *, entry_ts: datetime, entry_price: float, size: float
    ) -> "ExitPlan":
        return cls(
            symbol=intent.symbol,
            side=intent.side,
            entry_ts=entry_ts,
            entry_price=entry_price,
            stop_price=intent.stop_price,
            target_price=intent.target_price,
            size=size,
        )

    @model_validator(mode="after")
    def _check(self) -> "ExitPlan":
        if self.side is Side.LONG:
            if not (self.stop_price < self.entry_price < self.target_price):
                raise ValueError(
                    f"LONG geometry is inverted: stop {self.stop_price} < entry "
                    f"{self.entry_price} < target {self.target_price} does not hold"
                )
        else:
            if not (self.target_price < self.entry_price < self.stop_price):
                raise ValueError(
                    f"SHORT geometry is inverted: target {self.target_price} < entry "
                    f"{self.entry_price} < stop {self.stop_price} does not hold"
                )
        return self


class ExitFill(FrozenModel):
    """Whether a bar closed the position, and on what terms.

    `resolved=False` means neither level was reached in this bar; the ledger
    keeps the position open. Nothing here decides a time stop or a session
    flatten -- those are the ledger's, and `ExecutionModel.exit_at_close`
    prices them once the ledger has decided.
    """

    resolved: bool
    symbol: str
    side: Side
    exit_reason: ExitReason | None = None
    fill_ts: datetime | None = None
    reference_price: float | None = None
    effective_price: float | None = None
    size: float = Field(gt=0)

    gapped: bool = False
    ambiguous_fill: bool = False
    intrabar_bars_used: int = Field(ge=0, default=0)

    costs: CostBreakdown = Field(default_factory=CostBreakdown)
    spread: SpreadEstimate | None = None
    slippage: SlippageEstimate | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "ExitFill":
        for name in ("reference_price", "effective_price"):
            price = getattr(self, name)
            if price is not None and not math.isfinite(price):
                raise ValueError(
                    f"{name} is {price!r}: a non-finite exit price is not a price, "
                    "and it reaches the equity curve before anything re-derives it"
                )
        if not self.resolved:
            if self.exit_reason is not None or self.reference_price is not None:
                raise ValueError("an unresolved bar names no exit reason and no price")
            if self.costs.total != 0.0:
                raise ValueError("an unresolved bar costs nothing")
            return self
        if self.exit_reason is None or self.reference_price is None:
            raise ValueError("a resolved exit must carry a reason and a price")
        if self.fill_ts is None:
            raise ValueError("a resolved exit must carry a fill timestamp")
        return self


# ---------------------------------------------------------------------------
# Pure bar resolution
# ---------------------------------------------------------------------------


class BarResolution(FrozenModel):
    """Which level one bar hit, at what price, and how certainly.

    `gapped` and `ambiguous` are kept apart deliberately. A gap is resolved by
    the bar's own OPEN, which is chronologically first and therefore certain;
    an ambiguity is an ordering that was never observed. Collapsing the two
    would make the ambiguous count -- the one section 9 asks to be measurable
    -- include fills that are not ambiguous at all.
    """

    reason: ExitReason
    price: float
    gapped: bool = False
    ambiguous: bool = False
    notes: tuple[str, ...] = ()


def _stop_touched(plan: ExitPlan, bar: Bar) -> bool:
    """Inclusive. A print AT the stop triggers a stop-market order."""
    return bar.low <= plan.stop_price if plan.side is Side.LONG else bar.high >= plan.stop_price


def _target_traded_through(plan: ExitPlan, bar: Bar) -> bool:
    """Strict. A print AT our resting limit does not prove our order filled.

    Queue position is unobservable from bar data, so a touch is not evidence
    of a fill. The inclusive reading would hand the trade its full target on
    every bar that merely reached it, which is the flattering direction.
    """
    return (
        bar.high > plan.target_price
        if plan.side is Side.LONG
        else bar.low < plan.target_price
    )


def _opened_through_stop(plan: ExitPlan, bar: Bar) -> bool:
    return bar.open <= plan.stop_price if plan.side is Side.LONG else bar.open >= plan.stop_price


def _opened_through_target(plan: ExitPlan, bar: Bar) -> bool:
    """Strict, for the same reason `_target_traded_through` is strict.

    The open print is chronologically first, but a print AT our resting limit
    is still not evidence that our order filled -- queue position is
    unobservable from bar data whether the print is the open or the high. The
    inclusive reading handed a full target win to every bar that merely OPENED
    at the target, and target prices and bar opens both sit on the tick grid,
    so exact equality is an ordinary occurrence rather than a corner case. It
    was also the flattering direction twice over: a bar that opened at the
    target and then traded down through the stop booked the target instead of
    the stop, and it did so tagged `gapped` rather than `ambiguous`.

    With the strict test, an open exactly at the target falls through to the
    range checks below, where `high > target` must hold on its own.
    """
    return (
        bar.open > plan.target_price
        if plan.side is Side.LONG
        else bar.open < plan.target_price
    )


def resolve_bar_exit(
    plan: ExitPlan,
    bar: Bar,
    *,
    pessimistic_same_bar: bool = True,
    fill_gaps_at_open: bool = True,
) -> BarResolution | None:
    """Which of the stop and the target this ONE bar resolved, if either.

    The order of the checks is the order of the clauses in section 9, and each
    one is the asymmetric reading:

    1. **The open is chronologically first**, so a bar that opens through a
       level resolved at the open and nothing later in the bar can change it.
       That makes a gap UNAMBIGUOUS -- it is tagged `gapped`, not `ambiguous`.
       A stop gap fills at the OPEN (`fill_gaps_at_open`), the gap included in
       the price path. A target gap fills at the TARGET, because a resting
       limit order receives its limit price and does not collect a favourable
       gap.
    2. **Both levels inside the range** is the ambiguous case: the touch order
       is unobserved. `pessimistic_same_bar` picks the stop, and either choice
       is tagged `ambiguous`, because what makes it ambiguous is the missing
       observation, not the setting.
    3. One level only: unambiguous, filled at that level.

    Returns None when neither level was reached.
    """
    notes: list[str] = []

    if _opened_through_stop(plan, bar):
        if fill_gaps_at_open:
            price = bar.open
            notes.append(
                f"bar opened at {bar.open:g}, already through the stop "
                f"{plan.stop_price:g}: filled at the OPEN, not at the stop. Filling at "
                "the stop here is section 9's 'single most common backtest "
                "overstatement'"
            )
        else:
            price = plan.stop_price
            notes.append(
                f"bar opened at {bar.open:g}, through the stop {plan.stop_price:g}, but "
                "fill_gaps_at_open is False, so the fill is priced at the stop. This "
                "overstates results and the setting's own description says so"
            )
        return BarResolution(
            reason=ExitReason.STOP, price=price, gapped=True, notes=tuple(notes)
        )

    if _opened_through_target(plan, bar):
        notes.append(
            f"bar opened at {bar.open:g}, already through the target "
            f"{plan.target_price:g}: filled at the TARGET, not at the better open. A "
            "resting limit receives its limit price; collecting the gap would be the "
            "stop-gap rule run in the trade's favour"
        )
        return BarResolution(
            reason=ExitReason.TARGET,
            price=plan.target_price,
            gapped=True,
            notes=tuple(notes),
        )

    stop_hit = _stop_touched(plan, bar)
    target_hit = _target_traded_through(plan, bar)

    if stop_hit and target_hit:
        if pessimistic_same_bar:
            notes.append(
                f"bar range [{bar.low:g}, {bar.high:g}] contains both the stop "
                f"{plan.stop_price:g} and the target {plan.target_price:g}; the touch "
                "order is unobserved, so pessimistic_same_bar resolves to the STOP and "
                "the trade is tagged ambiguous_fill"
            )
            return BarResolution(
                reason=ExitReason.STOP,
                price=plan.stop_price,
                ambiguous=True,
                notes=tuple(notes),
            )
        notes.append(
            f"bar range [{bar.low:g}, {bar.high:g}] contains both the stop "
            f"{plan.stop_price:g} and the target {plan.target_price:g}; "
            "pessimistic_same_bar is False, so this resolves to the TARGET on an "
            "unobserved ordering. Still tagged ambiguous_fill, because the tag records "
            "the missing observation and not the setting"
        )
        return BarResolution(
            reason=ExitReason.TARGET,
            price=plan.target_price,
            ambiguous=True,
            notes=tuple(notes),
        )

    if stop_hit:
        return BarResolution(reason=ExitReason.STOP, price=plan.stop_price)
    if target_hit:
        return BarResolution(reason=ExitReason.TARGET, price=plan.target_price)
    return None


def validated_intrabar(plan: ExitPlan, bar: Bar, intrabar) -> tuple[tuple[Bar, ...], list[str]]:
    """Finer bars usable for intrabar resolution, or `()` with the reason why.

    Section 9: "Intrabar resolution uses finer-granularity bars when
    available; otherwise the pessimistic rule is applied". "Available" has to
    mean usable, and four things make a finer series unusable. Each one falls
    back to the pessimistic rule rather than resolving on partial evidence,
    because a finer series that covers only the calm half of a bar resolves
    every ambiguity in whichever direction it happens to cover.
    """
    notes: list[str] = []
    finer = tuple(intrabar or ())
    if not finer:
        return (), notes

    previous: datetime | None = None
    for fine in finer:
        if fine.symbol != plan.symbol:
            raise ExecutionError(
                f"intrabar bar for {fine.symbol!r} supplied for {plan.symbol!r}"
            )
        if not fine.is_valid:
            raise ExecutionError(
                f"{plan.symbol}: structurally invalid intrabar bar at "
                f"{fine.close_ts.isoformat()}"
            )
        if previous is not None and fine.close_ts <= previous:
            raise ExecutionError(
                f"{plan.symbol}: intrabar bars are not strictly ascending at "
                f"{fine.close_ts.isoformat()}"
            )
        previous = fine.close_ts
        if fine.close_ts <= bar.open_ts or fine.close_ts > bar.close_ts:
            raise FillTimingError(
                f"{plan.symbol}: intrabar bar closing at {fine.close_ts.isoformat()} "
                f"falls outside its parent bar ({bar.open_ts.isoformat()}, "
                f"{bar.close_ts.isoformat()}]. A finer bar from outside the parent is "
                "either a duplicate of another bar's range or data from the future"
            )
        if int(fine.interval_seconds) >= int(bar.interval_seconds):
            raise ExecutionError(
                f"{plan.symbol}: intrabar interval {fine.interval_seconds}s is not finer "
                f"than the parent's {bar.interval_seconds}s"
            )

    low = min(f.low for f in finer)
    high = max(f.high for f in finer)
    if low > bar.low or high < bar.high:
        notes.append(
            f"finer bars span [{low:g}, {high:g}] but the parent bar spans "
            f"[{bar.low:g}, {bar.high:g}]: the finer series does not cover the parent's "
            "extremes, so it is NOT used and the pessimistic rule applies. Resolving on "
            "partial intrabar coverage decides the ambiguity by which half of the bar "
            "happened to be supplied"
        )
        return (), notes
    if low < bar.low or high > bar.high:
        notes.append(
            f"finer bars reach further than the parent ([{low:g}, {high:g}] vs "
            f"[{bar.low:g}, {bar.high:g}]); the finer series is used because it shows "
            "the larger excursion, but the two series disagree and that is a data "
            "quality finding"
        )

    # Reaching the parent's price extremes is not the same as covering its
    # TIME. A series that spans only part of the parent's span can still touch
    # both extremes, and resolving on it then decides the stop/target ordering
    # from whichever MINUTES happened to be supplied -- the same defect as
    # partial price coverage, and worse, because a resolution "by observation"
    # clears `ambiguous_fill` and makes the sensitivity invisible. A hole is
    # treated as unobserved rather than as a quiet session: bar data cannot
    # distinguish "no trades printed" from "these bars were not delivered",
    # and only one of those two readings is safe.
    holes: list[str] = []
    if finer[0].open_ts != bar.open_ts:
        holes.append(
            f"it starts at {finer[0].open_ts.isoformat()} rather than the parent's "
            f"open {bar.open_ts.isoformat()}"
        )
    if finer[-1].close_ts != bar.close_ts:
        holes.append(
            f"it ends at {finer[-1].close_ts.isoformat()} rather than the parent's "
            f"close {bar.close_ts.isoformat()}"
        )
    for previous_bar, next_bar in zip(finer, finer[1:]):
        gap_seconds = (next_bar.open_ts - previous_bar.close_ts).total_seconds()
        if gap_seconds != 0.0:
            holes.append(
                f"there is a {gap_seconds:g}s discontinuity at "
                f"{previous_bar.close_ts.isoformat()}"
            )
    if holes:
        notes.append(
            "the finer series does not cover the parent's time span ("
            + "; ".join(holes)
            + "), so it is NOT used and the pessimistic rule applies. Touching the "
            "parent's high and low is not the same as observing the order in which "
            "they were touched"
        )
        return (), notes
    return finer, notes


