"""Price adjustments that must agree with the market: split checks and spin-offs.

Pure functions, no I/O.

**Split checks.** A recorded split of ratio r (shares after / shares before)
should move the raw price by about 1/r overnight; see ``check_split`` for the
verdicts. Examples from real data: Alpaca records United Technologies on
2020-04-03 as a 1-for-2 reverse split (r = 0.5, price should double), but the
price fell from 86.01 to 49.93 (contradicted, not applied). Top Ships' 1-for-15
reverse split in 2017 moved the price 5x, not 15x (magnitude mismatch).

**Spin-offs.** When a company distributes shares of another company, a holder
keeps the parent share and receives ``ratio`` child shares. The value received
on the ex-date is D = sum(ratio x child close on the ex-date), and the holder's
return over the ex-date is ``(parent close on + D) / close before - 1``. Raw
parent prices before the ex-date are divided by ``factor = (close on + D) /
close on``, which makes the adjusted series show exactly that holder return.
(Dividing by ``close before / (close before - D)`` instead, as simple dividend
adjustments do, assumes the parent did not otherwise move that day; for UTX,
where D was almost half the price, it would have shown +9.9 % instead of +5.2 %.)
A holder return beyond ``SPIN_OFF_MAX_ABS_RETURN`` flags the record.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

#: A split is contradicted when the observed move differs from the expected one
#: by more than this factor (either way).
SPLIT_TOLERANCE = 2.0

#: A move in the split's direction at least this large shows a split probably happened.
MIN_SPLIT_MOVE = 1.5

#: A spin-off record is consistent with prices when the holder's return over the
#: ex-date, distributed shares included, stays within this range.
SPIN_OFF_MAX_ABS_RETURN = 0.25


def check_split(ratio: float, close_before: float | None, close_on: float | None) -> str:
    """``consistent``, ``magnitude_mismatch``, ``contradicted`` or ``no_prices``.

    * consistent - the price moved as the split implies, within ``SPLIT_TOLERANCE``.
      Applied.
    * magnitude_mismatch - the price moved a lot (at least ``MIN_SPLIT_MOVE``) in the
      split's direction, but by a different amount: a split probably happened, together
      with a large market move or with a different ratio than recorded. Applied, and the
      security is held out of the universe until someone reviews it.
    * contradicted - the price did not move, or moved the other way. Not applied.
    * no_prices - no trade on one side. Applied, unverified.
    """
    if close_before is None or close_on is None or close_before <= 0 or close_on <= 0:
        return "no_prices"
    if ratio <= 0:
        return "contradicted"
    observed = math.log(close_before / close_on)  # what the prices say the ratio was, in logs
    expected = math.log(ratio)
    if abs(observed - expected) <= math.log(SPLIT_TOLERANCE):
        return "consistent"
    if observed * expected > 0 and abs(observed) >= math.log(MIN_SPLIT_MOVE):
        return "magnitude_mismatch"
    return "contradicted"


@dataclass(frozen=True)
class SpinOffResult:
    distribution_value: float | None
    factor: float | None
    holder_return: float | None
    status: str  # price_consistent, price_inconsistent, prices_missing


def spin_off_adjustment(
    close_before: float | None,
    close_on: float | None,
    children: Sequence[tuple[float, float | None]],
) -> SpinOffResult:
    """``children`` are (ratio, child close on the ex-date) pairs."""
    if close_before is None or close_on is None or any(price is None for _, price in children):
        return SpinOffResult(None, None, None, "prices_missing")
    value = math.fsum(ratio * price for ratio, price in children)
    holder_return = (close_on + value) / close_before - 1.0
    if value >= close_before:
        return SpinOffResult(value, None, holder_return, "price_inconsistent")
    status = (
        "price_consistent"
        if abs(holder_return) <= SPIN_OFF_MAX_ABS_RETURN
        else "price_inconsistent"
    )
    return SpinOffResult(value, (close_on + value) / close_on, holder_return, status)


def distribution_factors(
    sessions: Iterable[date], spin_offs: Sequence[tuple[date, float]]
) -> dict[date, float]:
    """Product of the factors of every spin-off whose ex-date comes after each session."""
    result = {}
    for session in sessions:
        factor = 1.0
        for ex_date, spin_factor in spin_offs:
            if session < ex_date:
                factor *= spin_factor
        result[session] = factor
    return result
