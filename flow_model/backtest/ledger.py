"""The ledger: open positions, equity, mark-to-market and exposure.

Section 9 gives the ledger three jobs and this module does only those three:

    ledger.mark_to_market(ts)   -- revalue at THIS bar's prices, emit equity
    ledger.open_position(...)   -- register a filled entry
    ledger.close_position(...)  -- turn an exit fill into a `TradeRecord`

plus the two exits the ledger DECIDES rather than the market: the time stop
(`max_hold_bars`) and the end-of-session flatten. It decides them and prices
nothing: `ExecutionModel.exit_at_close` prices them, so a ledger-decided
flatten pays commission, spread and slippage like any other market exit
instead of being free. `SessionFlattenPolicy` holds the session half of that
decision; it reads a calendar, never market data.

## Equity, and why costs are realized when they are paid

Three quantities are kept apart because collapsing them hides a real
overstatement:

    realized_pnl       sum of closed TradeRecord.pnl (already net of costs)
    open_entry_costs   the entry side of every position still open
    unrealized         gross mark-to-market of open positions

    realized_equity = starting + realized_pnl - open_entry_costs
    equity          = realized_equity + unrealized

The middle term is the one that is easy to drop. An entry's commission and
spread are paid at the entry, not at the exit, and a ledger that books the
whole round turn at the exit carries an equity curve that is too high for the
entire life of the trade -- which is exactly the curve the kill switch and the
drawdown series are read off. The two halves still sum to
`gross_pnl - costs.total`, so no cost is counted twice.

## compound_equity

`config.risk.compound_equity` decides which equity the next trade is sized
from: the current mark-to-market equity (compounding) or the starting equity
(fixed fractional). Section 11 records that this "changes the drawdown
distribution materially", so it is honoured in exactly one place --
`sizing_equity()`, the number the loop hands to the signal engine -- and it is
observable from the outside in three ways: the flag on `LedgerSnapshot`, the
divergence between `equity` and `sizing_equity`, and
`TradeIntent.equity_at_signal` on every trade, which is constant under fixed
fractional and varies under compounding.

## R, and partial fills -- TWO quantities, deliberately

This module used to compute its own `risk_dollars` from the realized entry and
the filled size. That was a defect: it silently RE-DERIVED R, which section 0.2
forbids ("R is defined once, at entry, and never revised"), and it disagreed
with `TradeIntent._check_geometry` -- the validator at the point R is actually
decided -- and with both committed test suites.

The two quantities are now separate, because one number cannot serve both
consumers honestly:

1. `risk_dollars` -- THE R UNIT, for every performance figure. It is
   `intent.risk_dollars`, CARRIED THROUGH untouched, never recomputed here:

       risk_dollars = |planned_entry - stop_price| * point_value * size

   Planned entry, requested size, both fixed at signal time. An entry that
   triggers two points worse than planned therefore realizes about -1.10R, not
   a tidy -1.00R, and a partial fill scales reported R down by the fill
   fraction. Both are real execution outcomes and both stay visible in the R
   distribution instead of being normalized away. `TradeRecord` validates this
   identity, so a slip here is a construction error rather than a plausible
   number in a report.

2. `carried_risk_dollars` -- THE LIVE EXPOSURE, for section 8's risk limits:

       carried_risk_dollars = |entry_price - stop_price| * point_value * filled_size

   Realized entry, filled size: the dollars the account can actually lose from
   where the position really is to where the stop really is. `open_risk_dollars`
   sums THIS, because a heat limit denominated in planned risk would understate
   the money at risk whenever an entry slipped.

Reporting R against the realized entry would make every stop read exactly
-1.00R and erase execution quality from the record; sizing the heat limit on
planned risk would hide an overage. Hence both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from pydantic import Field

from flow_model.backtest.fills import EntryFill, ExitFill, ExitPlan
from flow_model.core.contracts import (
    Bar,
    CostBreakdown,
    EquityPoint,
    TradeIntent,
    TradeRecord,
)
from flow_model.core.enums import (
    Component,
    DataQuality,
    ExitReason,
    Regime,
    Session,
    SetupType,
    Side,
    SplitPhase,
)
from flow_model.core.model import FrozenModel
from flow_model.utils.logging import get_logger

__all__ = [
    "EntryContext",
    "Ledger",
    "LedgerError",
    "LedgerSnapshot",
    "OpenPosition",
    "SessionFlattenPolicy",
]

logger = get_logger("backtest.ledger")


class LedgerError(RuntimeError):
    """Raised when the ledger is asked for something arithmetically impossible."""


# ---------------------------------------------------------------------------
# What a trade needs to remember about the bar that produced it
# ---------------------------------------------------------------------------


class EntryContext(FrozenModel):
    """The signal-time context a `TradeRecord` carries but a fill does not.

    Captured once, at the signal bar, and copied into the record at the exit.
    Captured rather than re-derived because re-deriving it at exit time would
    read the exit bar's features and label the trade with the regime it closed
    in instead of the one it was taken in.
    """

    flow_score: float
    regime: Regime
    component_points: dict[Component, float] = Field(default_factory=dict)
    volatility: float | None = None
    vol_percentile: float | None = None
    data_quality: DataQuality = DataQuality.GOOD
    session: Session = Session.CLOSED
    entry_reason: str = ""

    def points(self, component: Component) -> float:
        """Weighted points this component contributed, or 0.0 if absent.

        0.0 is the honest value for a component that contributed nothing to
        the score -- `TradeRecord`'s per-component fields are `ge=0.0`
        point contributions, not measurements, and a disabled component contributes
        zero points by `ComponentScore.points`'s own definition.
        """
        return float(self.component_points.get(component, 0.0))


# ---------------------------------------------------------------------------
# An open position
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OpenPosition:
    """One live position, with the two quantities that accumulate while it is.

    Mutable, unlike every contract in `core/contracts.py`, and deliberately:
    `bars_held` and the excursion extremes are per-bar accumulations, and a
    frozen record replaced on every bar of a ten-year run would be a lot of
    garbage to express the same thing. Everything that is a decision -- the
    bracket, the intent, the fill, the signal context -- is a frozen object
    held by reference.
    """

    trade_id: str
    intent: TradeIntent
    entry: EntryFill
    plan: ExitPlan
    context: EntryContext
    risk_dollars: float
    mark_price: float
    bars_held: int = 0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    stale_mark_bars: int = 0

    # --- identity ------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self.plan.symbol

    @property
    def side(self) -> Side:
        return self.plan.side

    @property
    def setup(self) -> SetupType:
        return self.intent.setup

    @property
    def size(self) -> float:
        """The size actually filled. The position is this big, not intent.size."""
        return self.plan.size

    @property
    def entry_price(self) -> float:
        return self.plan.entry_price

    @property
    def entry_ts(self) -> datetime:
        return self.plan.entry_ts

    @property
    def signal_ts(self) -> datetime:
        return self.intent.signal_ts

    @property
    def point_value(self) -> float:
        return self.intent.point_value

    @property
    def max_hold_bars(self) -> int | None:
        return self.intent.max_hold_bars

    @property
    def notional(self) -> float:
        """Exposure at the last mark, in account currency."""
        return abs(self.size * self.point_value * self.mark_price)

    @property
    def carried_risk_dollars(self) -> float:
        """Dollars the account can actually lose from here to the stop.

        REALIZED entry, FILLED size -- deliberately not `risk_dollars`, which is
        the planned R unit fixed at signal time (see this module's docstring).
        Section 8's limits consume THIS: a heat limit denominated in planned
        risk understates the money at risk whenever an entry slipped, and
        understating exposure is the one direction a hard limit may not err in.
        """
        return abs(self.entry_price - self.plan.stop_price) * self.point_value * self.size

    # --- arithmetic ----------------------------------------------------

    def gross_pnl_at(self, price: float) -> float:
        """Mark-to-market PnL at `price`, before any exit cost."""
        return self.side.sign * (price - self.entry_price) * self.point_value * self.size

    def r_at(self, price: float) -> float:
        """Gross PnL at `price` expressed in R."""
        if self.risk_dollars <= 0:
            raise LedgerError(
                f"{self.trade_id}: risk_dollars is {self.risk_dollars}, so R is "
                "undefined and no R multiple may be reported for this position"
            )
        return self.gross_pnl_at(price) / self.risk_dollars

    def observe(self, bar: Bar) -> None:
        """Update the excursion extremes from one bar the position was open for.

        Both extremes are taken from the bar's own high and low, so a bar that
        ran against the position and came back is recorded as having done so.
        The two are clamped at zero in their own direction because
        `TradeRecord` declares `mae_r <= 0` and `mfe_r >= 0`: a trade that was
        never once underwater has an adverse excursion of zero, not a positive
        one.
        """
        if bar.symbol != self.symbol:
            raise LedgerError(
                f"{self.trade_id}: bar for {bar.symbol!r} offered to a position in "
                f"{self.symbol!r}"
            )
        if not bar.is_valid:
            raise LedgerError(
                f"{self.trade_id}: structurally invalid bar at "
                f"{bar.close_ts.isoformat()} cannot update an excursion"
            )
        favourable = bar.high if self.side is Side.LONG else bar.low
        adverse = bar.low if self.side is Side.LONG else bar.high
        self.mfe_r = max(self.mfe_r, max(0.0, self.r_at(favourable)))
        self.mae_r = min(self.mae_r, min(0.0, self.r_at(adverse)))

    def observe_price(self, price: float) -> None:
        """Fold a single price (an exit print) into the excursions."""
        r = self.r_at(price)
        self.mfe_r = max(self.mfe_r, max(0.0, r))
        self.mae_r = min(self.mae_r, min(0.0, r))


# ---------------------------------------------------------------------------
# The session-flatten decision
# ---------------------------------------------------------------------------


class SessionFlattenPolicy:
    """Whether a position must be flat because its session has ended.

    Reads `SessionFilterConfig.flatten_at_session_close` and a calendar. A
    session schedule is not market data -- next Friday's close is known today
    -- so consulting it is not lookahead, and the calendar's own half-day and
    DST handling is what makes "the close" the right instant rather than a
    fixed wall time.

    The policy fires only for a position ENTERED inside the regular session.
    With `trade_rth_only=False` entries happen outside RTH on purpose, and a
    rule keyed on "we are not in RTH now" would flatten those on the bar after
    they opened.
    """

    __slots__ = ("session", "calendar", "spec", "_enabled", "_note")

    def __init__(self, session_config, calendar, spec) -> None:
        self.session = session_config
        self.calendar = calendar
        self.spec = spec
        self._note = ""
        self._enabled = bool(getattr(session_config, "flatten_at_session_close", False))
        if self._enabled and calendar is None:
            self._enabled = False
            self._note = (
                "session.flatten_at_session_close is set but no calendar was supplied, "
                "so no session flatten can be decided and positions are held across "
                "the close. Reported rather than silently approximated with a fixed "
                "wall-clock time, which would be wrong on every half day and twice a "
                "year at the DST boundary"
            )
        elif self._enabled and getattr(spec, "rth", None) is None:
            self._enabled = False
            self._note = (
                f"{getattr(spec, 'symbol', '?')} declares no rth window, so there is no "
                "session close to flatten at"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def note(self) -> str:
        """Why the policy is inert, when it is. Empty when it is active."""
        return self._note

    def due(self, opened_ts: datetime, ts: datetime) -> bool:
        """True when a position opened at `opened_ts` must be flat by `ts`.

        The caller passes the position's SIGNAL instant, not its fill instant.
        `is_rth` is half-open -- "a bar stamped at the close belongs to the
        closing auction, not to the next minute of trading" -- and a fill's
        timestamp is its bar's CLOSE, which can be exactly the session close.
        Keyed on the fill, such a position would read as never having been in
        the session and would be held overnight by a rule written to prevent
        exactly that. The signal instant is the bar the session gate admitted,
        so it is the instant that says which session the trade belongs to.
        """
        if not self._enabled:
            return False
        calendar = self.calendar
        if not calendar.is_rth(opened_ts, self.spec):
            return False
        if not calendar.is_rth(ts, self.spec):
            return True
        return calendar.session_date(ts, self.spec) != calendar.session_date(
            opened_ts, self.spec
        )


# ---------------------------------------------------------------------------
# A point-in-time read of the whole account
# ---------------------------------------------------------------------------


class LedgerSnapshot(FrozenModel):
    """The account at one instant, including what sizing will be based on.

    `EquityPoint` is a frozen Phase 1 contract with five fields and is not
    widened here; this is the richer read that the run report and the
    dashboard want, and the two are consistent by construction because both
    are built from the same accumulators.
    """

    ts: datetime
    equity: float
    realized_equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_entry_costs: float
    sizing_equity: float
    compound_equity: bool
    open_positions: int
    closed_trades: int
    open_risk_dollars: float
    open_notional: float
    high_water_mark: float

    @property
    def drawdown_fraction(self) -> float:
        """Fraction below the high-water mark, in [0, 1]. 0.0 when at a peak."""
        if self.high_water_mark <= 0:
            return 0.0
        return max(0.0, (self.high_water_mark - self.equity) / self.high_water_mark)

    @property
    def exposure_fraction(self) -> float | None:
        """Open notional over equity, or None when equity is non-positive.

        None rather than a large or negative number: with equity at or below
        zero the ratio is not an exposure, and reporting one would be reporting
        a leverage figure about an account that no longer exists.
        """
        if self.equity <= 0:
            return None
        return self.open_notional / self.equity


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


class Ledger:
    """Positions, equity and exposure for one run.

    No market data in the constructor and no data access of any kind: prices
    arrive as arguments to `mark_to_market`, and bars arrive as arguments to
    `observe`. The ledger cannot read a price it was not handed, which is the
    same property `MarketView` gives the feature layer.
    """

    def __init__(
        self,
        config,
        *,
        run_id: str,
        config_hash: str = "",
        data_hash: str = "",
        split_label: SplitPhase | None = None,
        starting_equity: float | None = None,
        record_equity_every_bar: bool | None = None,
    ) -> None:
        risk = config.risk
        equity = float(
            risk.starting_equity if starting_equity is None else starting_equity
        )
        if not math.isfinite(equity) or equity <= 0:
            raise LedgerError(f"starting equity must be positive and finite, got {equity}")

        self.config = config
        self.run_id = run_id
        self.config_hash = config_hash or getattr(config, "config_hash", "")
        self.data_hash = data_hash
        self.split_label = split_label
        self.starting_equity = equity
        self.compound_equity = bool(risk.compound_equity)
        self.record_equity_every_bar = bool(
            config.backtest.record_equity_every_bar
            if record_equity_every_bar is None
            else record_equity_every_bar
        )

        self._positions: dict[str, OpenPosition] = {}
        self._realized_pnl = 0.0
        self._open_entry_costs = 0.0
        self._closed_trades = 0
        self._high_water_mark = equity
        self._equity_curve: list[EquityPoint] = []
        self._last_marked_ts: datetime | None = None
        self._marks_seen = 0
        self._stale_marks = 0
        self._next_sequence = 1
        self._notes: list[str] = []

    # --- identity ------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Ledger(run={self.run_id} equity={self.equity:.2f} "
            f"open={len(self._positions)} closed={self._closed_trades})"
        )

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(self._notes)

    # --- money ---------------------------------------------------------

    @property
    def realized_pnl(self) -> float:
        """Net PnL of closed trades. 0.0 over no trades, which is a SUM and is
        therefore defined at n=0 -- unlike a win rate or a profit factor."""
        return self._realized_pnl

    @property
    def open_entry_costs(self) -> float:
        return self._open_entry_costs

    @property
    def realized_equity(self) -> float:
        return self.starting_equity + self._realized_pnl - self._open_entry_costs

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            position.gross_pnl_at(position.mark_price)
            for position in self._positions.values()
        )

    @property
    def equity(self) -> float:
        return self.realized_equity + self.unrealized_pnl

    @property
    def high_water_mark(self) -> float:
        return self._high_water_mark

    def drawdown_fraction(self) -> float:
        """Current drawdown from the high-water mark, as a fraction in [0, 1]."""
        if self._high_water_mark <= 0:
            return 0.0
        return max(0.0, (self._high_water_mark - self.equity) / self._high_water_mark)

    def sizing_equity(self) -> float:
        """The equity the next trade is sized from. The one place
        `compound_equity` is honoured."""
        return self.equity if self.compound_equity else self.starting_equity

    # --- positions -----------------------------------------------------

    @property
    def open_positions(self) -> tuple[OpenPosition, ...]:
        """Open positions in the order they were opened. An order, so that a
        run's exit sequence does not depend on dict or set iteration."""
        return tuple(self._positions.values())

    @property
    def closed_trades(self) -> int:
        return self._closed_trades

    def is_flat(self) -> bool:
        return not self._positions

    def open_risk_dollars(self) -> float:
        """Sum of the open positions' CARRIED exposure -- the portfolio heat
        section 8 limits. Not a sum of R: see this module's docstring on why the
        planned R unit and the realized exposure are different numbers."""
        return sum(
            position.carried_risk_dollars for position in self._positions.values()
        )

    def open_notional(self) -> float:
        return sum(position.notional for position in self._positions.values())

    def exposure_fraction(self) -> float | None:
        equity = self.equity
        if equity <= 0:
            return None
        return self.open_notional() / equity

    def next_trade_id(self) -> str:
        """A deterministic trade id: the run id plus a counter.

        Content-derived rather than a uuid or a timestamp, because `trade_id`
        is a field of `TradeRecord` and two runs of the same config over the
        same data are required to produce byte-identical trades.
        """
        trade_id = f"{self.run_id}-t{self._next_sequence:06d}"
        self._next_sequence += 1
        return trade_id

    # --- mark to market ------------------------------------------------

    def mark_to_market(
        self,
        ts: datetime,
        prices: Mapping[str, float] | None = None,
        *,
        record_equity: bool | None = None,
    ) -> EquityPoint | None:
        """Revalue at `ts` and (per config) append an `EquityPoint`.

        Called FIRST in section 9's loop, with this bar's own prices, so the
        kill switch and the drawdown series see the bar they are about to act
        on rather than the previous one. A kill switch evaluated on stale
        equity fires a bar late, at a worse price than it claims.

        A symbol with no price in `prices` keeps its previous mark and the
        position counts the bar as stale. That is a measurement -- "this
        position was revalued at a price from an earlier bar" -- and it is
        counted rather than hidden, because an un-updated mark silently
        freezes the drawdown.

        Returns the appended `EquityPoint`, or None when this bar was not
        recorded.
        """
        if self._last_marked_ts is not None and ts < self._last_marked_ts:
            raise LedgerError(
                f"mark_to_market went backwards: {ts.isoformat()} after "
                f"{self._last_marked_ts.isoformat()}. The clock is chronological and "
                "an out-of-order mark would revalue the account at a price it has "
                "already moved past"
            )
        for position in self._positions.values():
            price = None if prices is None else prices.get(position.symbol)
            if price is None:
                position.stale_mark_bars += 1
                self._stale_marks += 1
                if self._stale_marks == 1:
                    self._notes.append(
                        f"{position.symbol} had no price at {ts.isoformat()} while a "
                        "position was open, so that bar was marked at the previous "
                        "bar's price. An un-updated mark freezes the equity curve and "
                        "the drawdown with it; the count is in "
                        "`Ledger.diagnostics()['stale_marks']`"
                    )
            elif not math.isfinite(price):
                raise LedgerError(
                    f"{position.symbol}: mark price {price!r} at {ts.isoformat()} is not "
                    "finite; a non-finite mark propagates into equity, the high-water "
                    "mark and every drawdown computed from them"
                )
            else:
                position.mark_price = float(price)

        self._last_marked_ts = ts
        self._marks_seen += 1
        equity = self.equity
        if equity > self._high_water_mark:
            self._high_water_mark = equity

        should_record = (
            self.record_equity_every_bar if record_equity is None else bool(record_equity)
        )
        if not should_record:
            # With per-bar recording off, the curve keeps the bars where
            # REALIZED equity moved, plus the first bar. Those are the points a
            # realized-PnL drawdown is computed from; recording nothing at all
            # would leave the run with no equity series, and recording "bars
            # where equity changed" would be every bar as soon as a position is
            # open, which is the setting the caller just turned off.
            previous = self._equity_curve[-1] if self._equity_curve else None
            if previous is not None and math.isclose(
                previous.realized_equity, self.realized_equity, rel_tol=0.0, abs_tol=1e-9
            ):
                return None
        return self._append_equity(ts)

    def _append_equity(self, ts: datetime) -> EquityPoint:
        # The high-water mark is raised HERE as well as in `mark_to_market`,
        # because an equity sample may be taken after equity moved without a
        # mark: `record_equity_point` closes the curve on the run's last bar,
        # and the exits of that bar are resolved AFTER it was marked (section
        # 9's order). A trade that exits at its target above that bar's close
        # leaves the final sample above the mark it was taken from, and a point
        # whose `high_water_mark` is below its own `equity` reports
        # `drawdown_pct` as a POSITIVE number -- which `EquityPoint` defines as
        # the fraction BELOW the peak, and which no drawdown series can
        # contain. `mark_to_market` keeps its own update because it must raise
        # the mark on bars it does not sample.
        equity = self.equity
        if equity > self._high_water_mark:
            self._high_water_mark = equity
        point = EquityPoint(
            ts=ts,
            equity=self.equity,
            realized_equity=self.realized_equity,
            open_positions=len(self._positions),
            open_risk_dollars=self.open_risk_dollars(),
            high_water_mark=self._high_water_mark,
        )
        self._equity_curve.append(point)
        return point

    def record_equity_point(
        self, ts: datetime, *, replace_same_ts: bool = False
    ) -> EquityPoint:
        """Force one equity sample, whatever the per-bar setting.

        Used for the first and last bar of a run, so that a run always has at
        least the two endpoints of an equity curve.

        `replace_same_ts` overwrites an existing sample at the same instant
        instead of appending a second one. The end-of-run flatten needs it: the
        bar has already been marked, and appending would leave two points at
        one timestamp -- a curve that any downstream drawdown or return series
        would read as two bars, one of them with a zero time step.
        """
        if (
            replace_same_ts
            and self._equity_curve
            and self._equity_curve[-1].ts == ts
        ):
            self._equity_curve.pop()
        return self._append_equity(ts)

    def equity_curve(self) -> tuple[EquityPoint, ...]:
        return tuple(self._equity_curve)

    # --- opening -------------------------------------------------------

    def open_position(
        self,
        *,
        intent: TradeIntent,
        entry: EntryFill,
        context: EntryContext,
        trade_id: str | None = None,
    ) -> OpenPosition:
        """Register a filled entry. Returns the live position.

        The bracket is rebuilt from the intent at the price that actually
        filled, so the stop and the target are the ones the plan named and the
        geometry is re-validated against the real entry. `ExitPlan` refuses an
        inverted bracket, which is what catches a fill that gapped past its own
        stop before anything can book a fictional win on it.
        """
        if not entry.filled:
            raise LedgerError(
                f"{intent.symbol}: a {entry.status.value} entry opens no position"
            )
        if entry.reference_price is None or entry.fill_ts is None:
            raise LedgerError(
                f"{intent.symbol}: a filled entry must carry a price and a timestamp"
            )
        plan = ExitPlan.from_intent(
            intent,
            entry_ts=entry.fill_ts,
            entry_price=float(entry.reference_price),
            size=float(entry.filled_size),
        )
        # CARRIED, not recomputed. `intent.risk_dollars` was validated against
        # the planned geometry by `TradeIntent._check_geometry` at the moment
        # the trade was sized; recomputing it here from the fill is exactly the
        # revision section 0.2 forbids, and it is what this module used to do.
        risk_dollars = float(intent.risk_dollars)
        if not math.isfinite(risk_dollars) or risk_dollars <= 0:
            raise LedgerError(
                f"{intent.symbol}: the intent carries risk_dollars {risk_dollars!r}; R "
                "would be undefined and every R multiple on this trade meaningless"
            )
        position = OpenPosition(
            trade_id=trade_id or self.next_trade_id(),
            intent=intent,
            entry=entry,
            plan=plan,
            context=context,
            risk_dollars=risk_dollars,
            mark_price=plan.entry_price,
        )
        if position.trade_id in self._positions:
            raise LedgerError(f"duplicate trade_id {position.trade_id}")
        self._positions[position.trade_id] = position
        self._open_entry_costs += entry.costs.total
        logger.debug(
            "opened %s %s %g @ %g (R=%.2f, entry costs %.2f)",
            position.trade_id,
            position.side.value,
            position.size,
            position.entry_price,
            risk_dollars,
            entry.costs.total,
        )
        return position

    # --- the exits the ledger decides ----------------------------------

    def time_stop_due(self, position: OpenPosition) -> bool:
        """True when the position has been held its configured maximum.

        `bars_held` counts bars that CLOSED after the entry bar, so
        `max_hold_bars=24` means twenty-four bars after the one the fill
        printed in. A position with no `max_hold_bars` never times out.
        """
        limit = position.max_hold_bars
        if limit is None:
            return False
        return position.bars_held >= int(limit)

    def advance_bar(self, position: OpenPosition, ts: datetime) -> None:
        """Count one elapsed bar against a position's holding time."""
        if ts > position.entry_ts:
            position.bars_held += 1

    # --- closing -------------------------------------------------------

    def close_position(
        self,
        position: OpenPosition,
        exit_fill: ExitFill,
        costs: CostBreakdown,
        *,
        exit_reason: ExitReason | None = None,
    ) -> TradeRecord:
        """Close a position and return its `TradeRecord`.

        `costs` is the ROUND TURN, from `ExecutionModel.round_turn_costs`: the
        field-wise sum of the two sides. `gross_pnl` is re-derived from the two
        reference prices and the filled size, and `TradeRecord`'s own validator
        re-derives it again and checks `pnl == gross_pnl - costs.total`, so an
        arithmetic slip here is a construction error rather than a plausible
        number in a report.
        """
        if position.trade_id not in self._positions:
            raise LedgerError(f"{position.trade_id} is not an open position")
        if not exit_fill.resolved:
            raise LedgerError(
                f"{position.trade_id}: an unresolved exit closes nothing"
            )
        if exit_fill.reference_price is None or exit_fill.fill_ts is None:
            raise LedgerError(
                f"{position.trade_id}: a resolved exit must carry a price and a timestamp"
            )
        reason = exit_reason or exit_fill.exit_reason
        if reason is None:
            raise LedgerError(f"{position.trade_id}: an exit must name a reason")

        exit_price = float(exit_fill.reference_price)
        position.observe_price(exit_price)
        gross = position.gross_pnl_at(exit_price)
        net = gross - costs.total

        record = TradeRecord(
            trade_id=position.trade_id,
            run_id=self.run_id,
            symbol=position.symbol,
            config_hash=self.config_hash,
            data_hash=self.data_hash,
            split_label=self.split_label,
            signal_ts=position.signal_ts,
            entry_ts=position.entry_ts,
            exit_ts=exit_fill.fill_ts,
            bars_held=position.bars_held,
            session=position.context.session,
            side=position.side,
            setup=position.setup,
            planned_entry=position.intent.entry_price,
            stop_price=position.plan.stop_price,
            target_price=position.plan.target_price,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.intent.size,
            filled_size=position.size,
            point_value=position.point_value,
            exit_reason=reason,
            ambiguous_fill=bool(position.entry.ambiguous_fill or exit_fill.ambiguous_fill),
            risk_dollars=position.risk_dollars,
            gross_pnl=gross,
            costs=costs,
            pnl=net,
            mae_r=position.mae_r,
            mfe_r=position.mfe_r,
            flow_score=position.context.flow_score,
            options_flow_score=position.context.points(Component.OPTIONS_FLOW),
            order_flow_score=position.context.points(Component.ORDER_FLOW),
            structure_score=position.context.points(Component.STRUCTURE),
            liquidity_score=position.context.points(Component.LIQUIDITY),
            momentum_score=position.context.points(Component.VOL_MOMENTUM),
            regime=position.context.regime,
            volatility=position.context.volatility,
            vol_percentile=position.context.vol_percentile,
            data_quality=position.context.data_quality,
            entry_reason=position.context.entry_reason,
            exit_reason_detail=" | ".join(exit_fill.notes),
        )

        del self._positions[position.trade_id]
        self._open_entry_costs -= position.entry.costs.total
        self._realized_pnl += net
        self._closed_trades += 1
        logger.debug(
            "closed %s %s @ %g: %.2f net (%s)",
            record.trade_id,
            record.side.value,
            exit_price,
            net,
            reason.value,
        )
        return record

    # --- reporting -----------------------------------------------------

    def snapshot(self, ts: datetime | None = None) -> LedgerSnapshot:
        return LedgerSnapshot(
            ts=ts or self._last_marked_ts or datetime.min,
            equity=self.equity,
            realized_equity=self.realized_equity,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            open_entry_costs=self._open_entry_costs,
            sizing_equity=self.sizing_equity(),
            compound_equity=self.compound_equity,
            open_positions=len(self._positions),
            closed_trades=self._closed_trades,
            open_risk_dollars=self.open_risk_dollars(),
            open_notional=self.open_notional(),
            high_water_mark=self._high_water_mark,
        )

    def diagnostics(self) -> dict[str, float]:
        """Counters a run report quotes verbatim. Counts and sums only."""
        return {
            "marks": float(self._marks_seen),
            "stale_marks": float(self._stale_marks),
            "equity_points": float(len(self._equity_curve)),
            "closed_trades": float(self._closed_trades),
            "open_positions": float(len(self._positions)),
        }

    def assert_consistent(self, trades: Iterable[TradeRecord]) -> None:
        """Check the accumulators against the records they were built from.

        `realized_pnl` must equal the sum of the closed trades' net PnL. The
        two are computed from the same numbers by different routes -- one
        accumulated bar by bar, one summed at the end -- so a disagreement
        means a position was closed without its record, or a record exists for
        a position the ledger never released.
        """
        total = 0.0
        count = 0
        for record in trades:
            total += record.pnl
            count += 1
        if count != self._closed_trades:
            raise LedgerError(
                f"{count} trade records against {self._closed_trades} closed positions"
            )
        if not math.isclose(total, self._realized_pnl, rel_tol=1e-9, abs_tol=1e-6):
            raise LedgerError(
                f"realized PnL {self._realized_pnl} does not match the sum of the "
                f"trade records ({total})"
            )
