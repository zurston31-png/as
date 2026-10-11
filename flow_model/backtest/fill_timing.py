"""Fill eligibility: the lookahead surface of the execution model.

ARCHITECTURE.md section 9's loop invariant is one line -- "fills resolve on
later bars only" -- and it is the line whose violation does not produce a
wrong number in one place but silently invalidates every number the system
will ever produce. A fill resolving on the bar that produced the signal has
used that bar's own future, and the backtest looks BETTER for it, not worse.

It lives in its own module for the same reason `data/market_view.py` does:
the invariant is worth being able to read in full.

## Structural, not procedural

`FillWindow` copies the pattern `market_view.py` describes: at construction it
DROPS the signal bar, every bar before the order became eligible, and every
bar past the order's wait horizon, and retains no reference to the rest. The
candidate bars are what the object holds; the signal bar is ABSENT, not
merely un-requested. Code that reaches into a private attribute still cannot
find it.

No accessor takes a timestamp, enforced by introspection in
`tests/unit/test_execution_smoke.py` exactly as `test_market_view.py` enforces
it on `MarketView`. A caller can only ask for "the candidates, in order",
which cannot be aimed at a bar of its choosing.

## The floor is arithmetic

    min_fill_offset = floor(latency_seconds / interval) + 1 + entry_delay_bars

The `+ 1` on a non-negative floor IS the invariant: there is no
configuration, `latency_ms=0` included, under which a fill lands on the
signal bar. The division is section 9's "latency_ms shifts eligible fill
time" -- at `latency_ms=250` against 300-second bars the order is eligible on
the next bar, and at `latency_ms=400_000` it is eligible two bars out.

`floor(x) + 1` rather than `max(1, ceil(x))`: the two agree except when the
latency is a whole number of bars, where `ceil` returned the bar whose OPEN
is the signal instant itself and so priced a full bar of latency as a
zero-latency fill.

## One residual, admitted rather than hidden

A market order fills at the chosen bar's OPEN print, and that print can
precede the eligibility instant by up to one bar interval -- 250ms of it at
the default latency, and `latency mod interval` whenever the latency exceeds a
bar. Bar data holds no finer price, so there is no more honest choice
available at this granularity. The magnitude is recorded per fill as
`EntryFill.eligibility_lag_seconds`, so a sweep of `latency_ms` measures its
effect instead of a comment asserting it is negligible.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from pydantic import Field, model_validator

from flow_model.core.contracts import Bar, TradeIntent
from flow_model.core.model import FrozenModel

__all__ = [
    "ExecutionError",
    "FillTimingError",
    "FillWindow",
    "PendingOrder",
    "min_fill_offset",
]


class ExecutionError(RuntimeError):
    """Raised when a caller violates this module's declared contract.

    A bar for the wrong symbol, a non-monotonic bar list, mixed intervals, a
    structurally invalid bar. Missing or declining DATA never raises -- it
    produces a MISSED fill with a reason, mirroring the signal layer's WAIT.
    """


class FillTimingError(ExecutionError):
    """Raised when a fill would resolve at or before the bar that signalled it.

    The analogue of `data/market_view.LookaheadError`, and raised for the same
    reason: the one invariant whose violation makes every downstream number
    worthless gets an exception of its own rather than a boolean.
    """


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


class PendingOrder(FrozenModel):
    """A submitted intent plus the instant it could first be worked.

    `eligible_ts` is an absolute instant, deliberately not a bar count: the
    bar-offset floor needs a bar interval, which is a property of the data and
    is resolved by `FillWindow` where the bars actually are.
    """

    intent: TradeIntent
    submitted_ts: datetime
    order_type: str = Field(pattern="^(market|limit|stop)$")
    latency_seconds: float = Field(ge=0.0)
    jitter_seconds: float = Field(ge=0.0, default=0.0)
    entry_delay_bars: int = Field(ge=0, default=0)
    max_entry_wait_bars: int = Field(ge=0, default=1)
    notes: tuple[str, ...] = ()

    @property
    def symbol(self) -> str:
        return self.intent.symbol

    @property
    def signal_ts(self) -> datetime:
        return self.intent.signal_ts

    @property
    def eligible_ts(self) -> datetime:
        """The earliest instant this order could be worked."""
        return self.submitted_ts + timedelta(
            seconds=self.latency_seconds + self.jitter_seconds
        )

    @property
    def wait_bars(self) -> int:
        """How many bars the order stays live.

        Section 9 scopes `max_entry_wait_bars` to resting entries: "a limit
        entry is missed if price never trades through by max_entry_wait_bars".
        A MARKET order gets exactly one bar. Re-offering a market order on
        later bars would let a backtest chase an entry it missed, at prices
        the signal never saw, and `ExecutionConfig`'s own validator confirms
        the reading -- it rejects `limit` with `max_entry_wait_bars=0` because
        that is an order that "can never fill".
        """
        return 1 if self.order_type == "market" else int(self.max_entry_wait_bars)

    @model_validator(mode="after")
    def _check(self) -> "PendingOrder":
        if self.submitted_ts < self.intent.signal_ts:
            raise FillTimingError(
                f"order submitted at {self.submitted_ts.isoformat()} precedes its signal "
                f"at {self.intent.signal_ts.isoformat()}"
            )
        return self


def min_fill_offset(
    *,
    latency_seconds: float,
    interval_seconds: int,
    entry_delay_bars: int = 0,
) -> int:
    """The first bar, counted after the signal bar, that may host a fill.

        floor(latency_seconds / interval_seconds) + 1 + entry_delay_bars

    The `+ 1` is section 9's loop invariant made arithmetic: `floor` of a
    non-negative number is non-negative, so there is no configuration,
    `latency_ms = 0` included, under which this returns 0 and lets a fill land
    on the bar that produced the signal.

    `floor(x) + 1` equals `ceil(x)` for every non-integer `x` -- 250ms against
    300s bars is offset 1, 400s is 2, 700s is 3 -- and differs exactly where
    `ceil` degenerates: at a latency that is a whole number of bars. With
    `ceil`, 300s of latency against 300s bars returned offset 1, whose OPEN
    print is the signal instant itself, so a full bar of latency bought a
    fill at the price the signal saw -- the zero-latency fill, reached by
    configuring the opposite. `floor(x) + 1` returns 2 there, which is the
    first bar whose open is at or after the eligibility instant, and keeps the
    residual lag below one bar interval for every latency rather than for
    almost every latency.
    """
    if interval_seconds <= 0:
        raise ExecutionError(f"interval_seconds must be positive, got {interval_seconds}")
    if latency_seconds < 0:
        raise FillTimingError(
            f"latency_seconds={latency_seconds} is negative, which would make an order "
            "fillable before it was sent"
        )
    if entry_delay_bars < 0:
        raise ExecutionError(f"entry_delay_bars must be non-negative, got {entry_delay_bars}")
    latency_bars = math.floor(latency_seconds / interval_seconds)
    return latency_bars + 1 + int(entry_delay_bars)


class FillWindow:
    """The bars an order is allowed to be filled from -- and no others.

    Built from a `PendingOrder` and a chronological bar list. At construction
    it drops the signal bar, every bar before the eligibility floor, and every
    bar past the order's wait horizon, keeping only the candidates. The
    dropped bars are not retained in any attribute: like `MarketView`, the
    guarantee is that the signal bar is ABSENT, not merely un-requested.

    No accessor takes a timestamp, for the reason `MarketView` gives -- a
    caller can only ask for the candidates in order, which cannot be aimed at
    a bar of its choosing.
    """

    __slots__ = (
        "symbol",
        "signal_ts",
        "eligible_ts",
        "interval_seconds",
        "min_offset",
        "wait_bars",
        "first_offset",
        "_bars",
        "_notes",
    )

    def __init__(self, order: PendingOrder, bars) -> None:
        candidates = tuple(bars)
        self.symbol = order.symbol
        self.signal_ts = order.signal_ts
        self.eligible_ts = order.eligible_ts
        self.wait_bars = order.wait_bars
        self._notes: tuple[str, ...] = ()

        if not candidates:
            self.interval_seconds = 0
            self.min_offset = 0
            self.first_offset = 0
            self._bars: tuple[Bar, ...] = ()
            return

        interval = int(candidates[0].interval_seconds)
        previous: datetime | None = None
        for bar in candidates:
            if bar.symbol != order.symbol:
                raise ExecutionError(
                    f"bar for {bar.symbol!r} supplied to a window for {order.symbol!r}"
                )
            if int(bar.interval_seconds) != interval:
                raise ExecutionError(
                    f"{order.symbol}: mixed bar intervals ({interval} and "
                    f"{bar.interval_seconds}) in one fill window; the eligibility floor "
                    "is counted in bars and cannot be counted in two units at once"
                )
            if not bar.is_valid:
                raise ExecutionError(
                    f"{order.symbol}: structurally invalid bar at "
                    f"{bar.close_ts.isoformat()} (OHLC ordering or volume)"
                )
            if previous is not None and bar.close_ts <= previous:
                raise ExecutionError(
                    f"{order.symbol}: bars are not strictly ascending at "
                    f"{bar.close_ts.isoformat()}; an unordered list makes 'the next "
                    "bar' meaningless"
                )
            previous = bar.close_ts

        self.interval_seconds = interval
        self.min_offset = min_fill_offset(
            latency_seconds=(order.eligible_ts - order.signal_ts).total_seconds(),
            interval_seconds=interval,
            entry_delay_bars=order.entry_delay_bars,
        )

        # Offsets are counted over bars that CLOSE after the signal instant, so
        # the signal bar (close_ts == signal_ts) is offset 0 and is excluded by
        # the floor of 1 rather than by a separate rule.
        later = [bar for bar in candidates if bar.close_ts > order.signal_ts]
        kept: list[Bar] = []
        first_offset = 0
        for index, bar in enumerate(later, start=1):
            if index < self.min_offset:
                continue
            if index >= self.min_offset + self.wait_bars:
                break
            if not kept:
                first_offset = index
            kept.append(bar)

        self.first_offset = first_offset
        self._bars = tuple(kept)
        notes: list[str] = []
        if len(later) < len(candidates):
            notes.append(
                f"{len(candidates) - len(later)} bar(s) at or before the signal instant "
                "were dropped at construction and are not reachable from this window"
            )
        self._notes = tuple(notes)
        self.assert_resolves_after_signal()

    # --- identity ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._bars)

    def __bool__(self) -> bool:
        return bool(self._bars)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"FillWindow({self.symbol} signal={self.signal_ts.isoformat()} "
            f"min_offset={self.min_offset} candidates={len(self._bars)})"
        )

    @property
    def bars(self) -> tuple[Bar, ...]:
        """The candidate bars, oldest first. Never the full history."""
        return self._bars

    @property
    def notes(self) -> tuple[str, ...]:
        return self._notes

    def enumerate_offsets(self):
        """`(offset, bar)` pairs, where offset counts bars after the signal bar."""
        return tuple(
            (self.first_offset + i, bar) for i, bar in enumerate(self._bars)
        )

    # --- self-check ----------------------------------------------------

    def assert_resolves_after_signal(self) -> None:
        """Verify this window's own invariants.

        Checks every retained bar rather than the first, for the reason
        `MarketView.assert_no_future_leak` gives: a method whose entire job is
        to detect a leak and which inspects one row cannot detect a leak in
        another row.
        """
        for bar in self._bars:
            if bar.close_ts <= self.signal_ts:
                raise FillTimingError(
                    f"{self.symbol}: a candidate bar closing at "
                    f"{bar.close_ts.isoformat()} is at or before the signal instant "
                    f"{self.signal_ts.isoformat()}; a fill there would have used that "
                    "bar's own future"
                )
            if bar.open_ts < self.signal_ts:
                raise FillTimingError(
                    f"{self.symbol}: a candidate bar opening at "
                    f"{bar.open_ts.isoformat()} starts before the signal instant "
                    f"{self.signal_ts.isoformat()}, so part of its range predates the "
                    "signal"
                )
            if bar.close_ts < self.eligible_ts:
                raise FillTimingError(
                    f"{self.symbol}: a candidate bar closing at "
                    f"{bar.close_ts.isoformat()} ends before the order became eligible "
                    f"at {self.eligible_ts.isoformat()}"
                )


