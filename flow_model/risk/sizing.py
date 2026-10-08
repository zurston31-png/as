"""Position sizing: pure functions, no state, no clock, no I/O.

ARCHITECTURE.md section 8 fixes the arithmetic:

    risk_dollars   = equity * risk_pct          (capped by max_risk_pct)
    stop_distance  = |entry - stop|
    risk_per_unit  = stop_distance * point_value
    size           = floor(risk_dollars / risk_per_unit)
    actual_risk    = size * risk_per_unit
    R              = actual_risk
    target         = entry +/- stop_distance * rr_multiple

Two things this module takes seriously that a naive implementation does not.

**R is defined once, at entry, and never re-derived.** The project's whole
accounting rests on `R = |entry - stop| * point_value * size`, fixed when the
trade is opened, and realized R being `pnl_net / risk_dollars`. A sizing
routine that returns a stop the instrument cannot actually trade -- a price
off the tick grid -- produces a `stop_distance` that is not realizable, and
therefore an R that is not realizable either. Every price this module emits
is on the instrument's tick grid, and `risk_dollars` is recomputed from the
ROUNDED prices rather than the requested ones. `TradeIntent._check_geometry`
rejects the alternative, which is how this was caught.

**Rounding is conservative in a single direction.** Snapping to the grid can
only move a trade's economics one way here: the stop rounds AWAY from entry
(a wider stop, more risk per unit, fewer contracts) and the target rounds
TOWARD entry (less reward). Both make the trade look worse. The alternative
-- nearest-tick rounding -- would sometimes narrow the stop and widen the
target, manufacturing a fraction of a basis point of edge on every trade in
the backtest. That is small, systematic, and in the flattering direction,
which is the signature of the errors this project exists to avoid.

A consequence worth stating: because the target rounds inward, the realized
`planned_r_multiple` of a trade can fall below the setup's own
`min_reward_risk`, and such a trade is REJECTED rather than relabelled. On a
coarse tick grid relative to the stop distance, the grid genuinely cannot
express the trade the setup asked for, and `TradeRecord.r_label_is_honest`
exists precisely to stop a "1R scalp" quietly becoming a 0.8R one.
"""

from __future__ import annotations

import math

from pydantic import Field

from flow_model.core.contracts import TradeIntent
from flow_model.core.enums import Side
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel

__all__ = [
    "SizingRejection",
    "SizingDecision",
    "round_to_tick",
    "stop_distance_of",
    "risk_per_unit",
    "raw_size",
    "size_position",
]


class SizingRejection:
    """Namespace of rejection reasons. Strings so the journal can group them.

    Distinct reasons rather than one 'rejected' flag: section 12's rejection
    analysis asks how often each gate fires, which is unanswerable if every
    refusal reports the same cause.
    """

    STOP_TOO_TIGHT = "stop_too_tight"
    STOP_TOO_WIDE = "stop_too_wide"
    SIZE_ZERO = "size_zero"
    RISK_EXCEEDS_CAP = "risk_exceeds_cap"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    REWARD_RISK_BELOW_MINIMUM = "reward_risk_below_minimum"
    NON_FINITE_INPUT = "non_finite_input"
    EQUITY_NON_POSITIVE = "equity_non_positive"


class SizingDecision(FrozenModel):
    """The outcome of sizing one trade: either a full plan, or a refusal.

    Carries the numbers even when rejected, where they are known, because
    "rejected because size came out 0 at an 18.5-point stop" is actionable
    and "rejected" is not.
    """

    accepted: bool
    reason: str = ""
    size: float = Field(default=0.0, ge=0.0)
    risk_dollars: float = Field(default=0.0, ge=0.0)
    entry_price: float = Field(default=0.0, ge=0.0)
    stop_price: float = Field(default=0.0, ge=0.0)
    target_price: float = Field(default=0.0, ge=0.0)
    stop_distance: float = Field(default=0.0, ge=0.0)
    stop_ticks: float = Field(default=0.0, ge=0.0)
    planned_r_multiple: float = Field(default=0.0, ge=0.0)
    risk_pct_of_equity: float = Field(default=0.0, ge=0.0)
    requested_risk_dollars: float = Field(default=0.0, ge=0.0)
    notes: tuple[str, ...] = ()

    def require(self) -> "SizingDecision":
        """Self, or raise. For call sites that treat a refusal as a bug."""
        if not self.accepted:
            raise ValueError(f"sizing was rejected: {self.reason} ({'; '.join(self.notes)})")
        return self


def round_to_tick(price: float, tick_size: float, *, mode: str = "nearest") -> float:
    """Snap `price` to the tick grid.

    `mode` is "nearest", "up" or "down". Division before rounding is done on
    the ratio rather than by repeated subtraction so the result does not drift
    with the magnitude of the price.
    """
    if tick_size <= 0 or not math.isfinite(tick_size):
        raise ValueError(f"tick_size must be positive and finite, got {tick_size!r}")
    if not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")
    ratio = price / tick_size
    if mode == "nearest":
        ticks = math.floor(ratio + 0.5)
    elif mode == "up":
        ticks = math.ceil(ratio)
    elif mode == "down":
        ticks = math.floor(ratio)
    else:
        raise ValueError(f"unknown rounding mode {mode!r}; use nearest, up or down")
    # Re-round the product: tick_size is usually not representable in binary
    # (0.25 is, 0.01 is not), so ticks * tick_size can land a few ULPs off a
    # grid point and make an equality assertion on the grid fail.
    return round(ticks * tick_size, 10)


def stop_distance_of(entry_price: float, stop_price: float) -> float:
    """|entry - stop|. The denominator of every R in the system."""
    return abs(entry_price - stop_price)


def risk_per_unit(stop_distance: float, point_value: float) -> float:
    """Dollars risked per contract or share."""
    return stop_distance * point_value


def raw_size(risk_dollars: float, per_unit: float, *, allow_fractional: bool) -> float:
    """Units affordable at this risk budget.

    Floored, not rounded, for integer instruments: rounding UP would exceed
    the risk budget, which is the one direction the budget may not move.
    """
    if per_unit <= 0:
        return 0.0
    exact = risk_dollars / per_unit
    if allow_fractional:
        return exact
    return float(math.floor(exact))


def size_position(
    *,
    spec: InstrumentSpec,
    side: Side,
    entry_price: float,
    stop_price: float,
    reward_risk: float,
    equity: float,
    risk_pct: float,
    max_risk_pct: float,
    min_stop_ticks: int = 1,
    max_stop_distance: float | None = None,
    min_reward_risk: float | None = None,
    allow_fractional: bool = False,
) -> SizingDecision:
    """Size one trade, or refuse it with a reason.

    `stop_price` is where the thesis is falsified -- for the section 14 setups
    it comes from the structure, beyond the level -- and the target is derived
    from it by `reward_risk`. The stop is an input, never something this
    function chooses, because a stop fitted to a desired position size is a
    stop with no informational content.

    `max_stop_distance` is the setup's `max_stop_atr_multiple * ATR`, resolved
    by the caller: this module does not read market data.
    """
    notes: list[str] = []

    def reject(reason: str, **fields: float) -> SizingDecision:
        return SizingDecision(accepted=False, reason=reason, notes=tuple(notes), **fields)

    for name, value in (
        ("entry_price", entry_price),
        ("stop_price", stop_price),
        ("reward_risk", reward_risk),
        ("equity", equity),
        ("risk_pct", risk_pct),
        ("max_risk_pct", max_risk_pct),
    ):
        if not math.isfinite(value):
            notes.append(f"{name} is not finite ({value!r})")
            return reject(SizingRejection.NON_FINITE_INPUT)

    if equity <= 0:
        notes.append(f"equity is {equity}; a non-positive account cannot take risk")
        return reject(SizingRejection.EQUITY_NON_POSITIVE)
    if reward_risk <= 0:
        notes.append(f"reward_risk must be positive, got {reward_risk}")
        return reject(SizingRejection.DEGENERATE_GEOMETRY)

    # --- the hard cap. Section 8: "the manager cannot be configured to exceed
    # max_risk_per_trade_pct". Capping here rather than validating and raising
    # means a misconfigured risk_pct degrades to the cap instead of halting a
    # ten-year backtest on bar one -- but it is recorded, because silently
    # trading a different size than configured is its own dishonesty.
    effective_risk_pct = risk_pct
    if risk_pct > max_risk_pct:
        notes.append(
            f"risk_pct {risk_pct} exceeds max_risk_per_trade_pct {max_risk_pct}; "
            f"capped to {max_risk_pct}"
        )
        effective_risk_pct = max_risk_pct

    tick = spec.tick_size
    point_value = spec.point_value

    # --- snap to the grid, conservatively and in one direction only.
    entry = round_to_tick(entry_price, tick, mode="nearest")
    if side is Side.LONG:
        stop = round_to_tick(stop_price, tick, mode="down")   # wider
    else:
        stop = round_to_tick(stop_price, tick, mode="up")     # wider
    if entry != round_to_tick(entry_price, tick, mode="nearest"):  # pragma: no cover
        notes.append("entry rounding disagreed with itself")
    if stop != stop_price:
        notes.append(f"stop snapped {stop_price} -> {stop} (away from entry)")

    distance = stop_distance_of(entry, stop)
    if distance <= 0:
        notes.append(f"entry {entry} and stop {stop} coincide on the tick grid")
        return reject(SizingRejection.DEGENERATE_GEOMETRY, entry_price=entry, stop_price=stop)

    # Side consistency: a LONG whose stop sits above entry is a caller bug, not
    # a tradable plan. Caught here rather than by TradeIntent so the reason is
    # a journal-able refusal instead of a validation traceback mid-backtest.
    if side is Side.LONG and stop >= entry:
        notes.append(f"LONG stop {stop} is not below entry {entry}")
        return reject(SizingRejection.DEGENERATE_GEOMETRY, entry_price=entry, stop_price=stop)
    if side is Side.SHORT and stop <= entry:
        notes.append(f"SHORT stop {stop} is not above entry {entry}")
        return reject(SizingRejection.DEGENERATE_GEOMETRY, entry_price=entry, stop_price=stop)

    stop_ticks = round(distance / tick, 6)
    if stop_ticks < min_stop_ticks:
        notes.append(
            f"stop is {stop_ticks} ticks, below the setup's minimum {min_stop_ticks}; "
            "a stop inside the noise floor is a coin flip with a spread attached"
        )
        return reject(
            SizingRejection.STOP_TOO_TIGHT,
            entry_price=entry, stop_price=stop, stop_distance=distance, stop_ticks=stop_ticks,
        )
    if max_stop_distance is not None and distance > max_stop_distance:
        notes.append(
            f"stop distance {distance} exceeds the setup's cap {max_stop_distance} "
            "(max_stop_atr_multiple * ATR)"
        )
        return reject(
            SizingRejection.STOP_TOO_WIDE,
            entry_price=entry, stop_price=stop, stop_distance=distance, stop_ticks=stop_ticks,
        )

    # --- target, rounded TOWARD entry so reward is never overstated.
    raw_target = (
        entry + distance * reward_risk if side is Side.LONG else entry - distance * reward_risk
    )
    target = round_to_tick(raw_target, tick, mode="down" if side is Side.LONG else "up")
    if target != raw_target:
        notes.append(f"target snapped {round(raw_target, 10)} -> {target} (toward entry)")

    target_distance = abs(target - entry)
    if target_distance <= 0:
        notes.append("target collapsed onto entry after rounding toward it")
        return reject(
            SizingRejection.DEGENERATE_GEOMETRY,
            entry_price=entry, stop_price=stop, stop_distance=distance, stop_ticks=stop_ticks,
        )

    planned_r = round(target_distance / distance, 6)
    if min_reward_risk is not None and planned_r < min_reward_risk:
        notes.append(
            f"achievable reward/risk {planned_r} is below the setup's minimum "
            f"{min_reward_risk} once prices are on the tick grid. Rejected rather "
            f"than relabelled: a trade that cannot reach its stated R is not that setup"
        )
        return reject(
            SizingRejection.REWARD_RISK_BELOW_MINIMUM,
            entry_price=entry, stop_price=stop, target_price=target,
            stop_distance=distance, stop_ticks=stop_ticks, planned_r_multiple=planned_r,
        )

    # --- size from the ROUNDED geometry.
    requested = equity * effective_risk_pct
    per_unit = risk_per_unit(distance, point_value)
    size = raw_size(requested, per_unit, allow_fractional=allow_fractional)
    if size <= 0:
        notes.append(
            f"a {distance}-point stop costs {round(per_unit, 4)} per unit against a "
            f"{round(requested, 2)} budget, so not even one unit fits"
        )
        return reject(
            SizingRejection.SIZE_ZERO,
            entry_price=entry, stop_price=stop, target_price=target,
            stop_distance=distance, stop_ticks=stop_ticks, planned_r_multiple=planned_r,
            requested_risk_dollars=requested,
        )

    actual_risk = size * per_unit
    cap = equity * max_risk_pct
    # With floored integer sizing and risk_pct <= max_risk_pct this cannot
    # trip; it can under allow_fractional, and it is kept unconditionally
    # because it is the invariant the whole risk layer promises, and an
    # invariant asserted only where it is expected to hold is decoration.
    if actual_risk > cap * (1.0 + 1e-9):
        notes.append(
            f"actual risk {round(actual_risk, 4)} exceeds the hard cap {round(cap, 4)}"
        )
        return reject(
            SizingRejection.RISK_EXCEEDS_CAP,
            entry_price=entry, stop_price=stop, target_price=target,
            stop_distance=distance, stop_ticks=stop_ticks, planned_r_multiple=planned_r,
            size=size, requested_risk_dollars=requested,
        )

    return SizingDecision(
        accepted=True,
        size=size,
        risk_dollars=round(actual_risk, 10),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        stop_distance=distance,
        stop_ticks=stop_ticks,
        planned_r_multiple=planned_r,
        risk_pct_of_equity=round(actual_risk / equity, 10),
        requested_risk_dollars=round(requested, 10),
        notes=tuple(notes),
    )
