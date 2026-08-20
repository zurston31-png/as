"""One exit rule, shared by every strategy in the shadow system.

WHY THEY ALL GET THE SAME ONE

A challenger differs from the champion in how it SCORES an entry. If each
also got its own exit rule, a challenger that looked better would leave no
way to tell which half did the work, and the paired comparison - built at
some cost to isolate a single variable - would be measuring two.

So `Challenger.stop_loss_pct` and `.take_profit_pct` are recorded and
deliberately unused here. When entry scoring has been settled by evidence,
varying the exit becomes the next experiment; running both at once would
answer neither.

WHAT THIS IS NOT

It is not the live exit manager. That one ratchets on every price tick,
takes partial profits, and reads a rolling sample buffer for momentum and
trend-reversal exits - none of which the shadow path collects, because
collecting it would mean a per-token tick loop for hypothetical trades.

This walks CLOSED CANDLES and implements the four rules that survive that
translation honestly: stop-loss, take-profit, break-even, trailing stop,
plus a maximum hold. A shadow return therefore measures entry quality
under a fixed, simple exit - not what the live exit manager would have
achieved. Reading it as the latter would overstate what has been shown.

INTRABAR ORDER IS UNKNOWABLE, SO IT IS ASSUMED AGAINST THE TRADE

Within one bar the feed gives a high and a low but not which came first.
When a bar touches both the stop and the target, this takes the stop. The
alternative - assuming the good one landed first - would manufacture
winners out of ambiguity, and the bias would be invisible in the numbers
it produced.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.config import settings
from app.data.candles import Candle, Timeframe

STOP = "stop-loss"
TRAILING = "trailing stop"
BREAK_EVEN = "break-even stop"
TAKE_PROFIT = "take-profit"
MAX_HOLD = "max holding time"


@dataclass(frozen=True)
class ExitPolicy:
    stop_loss_pct: float
    take_profit_pct: float
    trailing_enabled: bool
    trailing_activation_pct: float
    trailing_distance_pct: float
    break_even_enabled: bool
    break_even_trigger_pct: float
    break_even_buffer_pct: float
    max_hold_hours: float

    @classmethod
    def from_settings(cls) -> "ExitPolicy":
        """Read the same numbers the live paper system uses.

        Shared values rather than a second set, so a shadow return is not
        quietly measuring a different trade than the one the bot would
        take. The MECHANISM is simpler here; the levels are not different.
        """
        return cls(
            stop_loss_pct=settings.STOP_LOSS_PCT,
            take_profit_pct=settings.TAKE_PROFIT_PCT,
            trailing_enabled=settings.TRAILING_STOP_ENABLED,
            trailing_activation_pct=settings.TRAILING_STOP_ACTIVATION_PCT,
            trailing_distance_pct=settings.TRAILING_STOP_DISTANCE_PCT,
            break_even_enabled=settings.BREAK_EVEN_ENABLED,
            break_even_trigger_pct=settings.BREAK_EVEN_TRIGGER_PCT,
            break_even_buffer_pct=settings.BREAK_EVEN_BUFFER_PCT,
            max_hold_hours=settings.MAX_POSITION_AGE_HOURS,
        )

    def fingerprint(self) -> str:
        """A short label stored on every resolved row.

        Without it, a dataset spanning a settings change would silently mix
        outcomes produced under different exit rules, and nothing in the
        table would say so.
        """
        bits = [f"sl{self.stop_loss_pct:g}", f"tp{self.take_profit_pct:g}"]
        if self.break_even_enabled:
            bits.append(f"be{self.break_even_trigger_pct:g}+{self.break_even_buffer_pct:g}")
        if self.trailing_enabled:
            bits.append(f"tr{self.trailing_activation_pct:g}/{self.trailing_distance_pct:g}")
        bits.append(f"hold{self.max_hold_hours:g}h")
        return ",".join(bits)


@dataclass
class Walk:
    """The result of stepping one hypothetical position through candles."""

    exit_price: float | None = None
    exit_at: dt.datetime | None = None
    exit_reason: str | None = None
    max_favorable_pct: float | None = None
    max_adverse_pct: float | None = None
    last_price: float | None = None
    bars: int = 0

    @property
    def closed(self) -> bool:
        return self.exit_price is not None


def walk(
    policy: ExitPolicy,
    *,
    entry_price: float,
    opened_at: dt.datetime,
    candles: list[Candle],
    timeframe: Timeframe,
) -> Walk:
    """Step through post-entry bars and find where the exit rule fires.

    `candles` must already be restricted to bars that opened at or after
    entry AND had closed by the resolution instant - this function has no
    idea what "now" is, deliberately, so it cannot read a bar the caller
    should not have given it. Look-ahead is the caller's constraint to
    honour, and keeping the check there means one place enforces it.
    """
    result = Walk()
    if entry_price <= 0:
        return result

    stop = entry_price * (1 - policy.stop_loss_pct)
    target = entry_price * (1 + policy.take_profit_pct)
    peak = entry_price
    break_even_applied = False
    trailing_active = False
    deadline = opened_at + dt.timedelta(hours=policy.max_hold_hours)
    interval = dt.timedelta(seconds=timeframe.seconds)

    for bar in candles:
        # The hold clock is checked on the bar's OPEN. A bar that opens
        # after the deadline belongs to a trade that should already be
        # flat, so it is not allowed to contribute a high, a low, or an
        # exit price.
        if bar.timestamp >= deadline:
            result.exit_price = result.last_price if result.last_price is not None else entry_price
            result.exit_at = deadline
            result.exit_reason = f"{MAX_HOLD} reached ({policy.max_hold_hours:g}h)"
            return result

        result.bars += 1
        result.last_price = bar.close
        closed_at = bar.timestamp + interval

        # The envelope comes from the bar's own extremes, so it captures
        # what the trade actually lived through rather than a sample of
        # closes. Both are gross price moves - fees are charged once, at
        # the exit, and charging them here too would double-count.
        high_move = (bar.high / entry_price - 1) * 100
        low_move = (bar.low / entry_price - 1) * 100
        result.max_favorable_pct = (
            high_move if result.max_favorable_pct is None
            else max(result.max_favorable_pct, high_move)
        )
        result.max_adverse_pct = (
            low_move if result.max_adverse_pct is None
            else min(result.max_adverse_pct, low_move)
        )

        # Which stop mechanism is governing right now. Once trailing has
        # ratcheted, a breach is a trailing stop and the record should say
        # so - "stop-loss hit" would misattribute it to the entry risk.
        label = TRAILING if trailing_active else (BREAK_EVEN if break_even_applied else STOP)

        # A gap through a level fills at the open, not at the level - the
        # price was never available in between. Checked before the
        # intrabar rules because at the open there is no ambiguity about
        # what happened first.
        if bar.open <= stop:
            result.exit_price, result.exit_at, result.exit_reason = bar.open, bar.timestamp, label
            return result
        if bar.open >= target:
            result.exit_price, result.exit_at, result.exit_reason = bar.open, bar.timestamp, TAKE_PROFIT
            return result
        # Both touched inside the bar: the stop wins. See the module note.
        if bar.low <= stop:
            result.exit_price, result.exit_at, result.exit_reason = stop, closed_at, label
            return result
        if bar.high >= target:
            result.exit_price, result.exit_at, result.exit_reason = target, closed_at, TAKE_PROFIT
            return result

        # Ratchets last, so they take effect from the NEXT bar. Moving the
        # stop up using the same bar's high and then testing that bar's
        # low against it would exit on a level the trade never saw.
        if policy.break_even_enabled and not break_even_applied:
            if bar.high >= entry_price * (1 + policy.break_even_trigger_pct):
                stop = max(stop, entry_price * (1 + policy.break_even_buffer_pct))
                break_even_applied = True
        if policy.trailing_enabled:
            peak = max(peak, bar.high)
            if (peak / entry_price - 1) >= policy.trailing_activation_pct:
                trailing_active = True
            if trailing_active:
                stop = max(stop, peak * (1 - policy.trailing_distance_pct))

    return result
