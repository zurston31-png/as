"""Smart-exit logic layered on top of the fixed stop-loss/take-profit set at
entry (app/risk/manager.py).

Every check here reads the position's OWN observed price history since
entry (`highest_price_since_entry`, `recent_prices`) rather than a live
OHLCV feed - the bot does not have one wired in for on-chain memecoins yet
(app/data/ has a full candle/indicator stack, but today it only feeds the
backtester, not live position monitoring; see the data-source open item in
project notes). That keeps every exit here honest: it reacts to prices this
specific position actually traded through during the trade, never a
fabricated indicator value.

`ExitManager.evaluate()` is the single entry point the position monitor
calls once per tick. It always records the tick first, then:

  1. break-even stop                - once far enough in profit, ratchets
                                       the stop up to guarantee the trade can
                                       no longer become a loser (moves the
                                       stop, does not exit by itself)
  2. trailing stop                  - once activated, ratchets the stop up
                                       behind the peak price and never
                                       loosens it (moves the stop, does not
                                       exit by itself)
  3. stop-loss / take-profit        - the position's non-negotiable risk
                                       boundary, checked AFTER the ratchets
                                       above so a breach is reported as
                                       whichever mechanism is actually
                                       governing the stop right now (plain /
                                       break-even / trailing), not a generic
                                       label that hides which one fired
  4. partial profit-take            - locks in some of the position once, at
                                       a configured profit level
  5. momentum-loss exit             - a sharp drop off a recent local peak,
                                       faster than the trailing stop's
                                       configured distance would catch
  6. trend-reversal exit            - two consecutive lower highs after the
                                       position's peak, in the recent samples
  7. time-based exit                - the position has been open too long

Steps 4-7 are strategy refinements layered on top of step 3, not a
replacement for it - stop-loss and take-profit still fire even if every
other rule is disabled.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app import models
from app.config import settings

MAX_RECENT_PRICE_SAMPLES = 30


@dataclass
class ExitAction:
    kind: str  # "none" | "full" | "partial"
    reason: str = ""
    fraction: float = 0.0  # only meaningful for "partial"


NO_EXIT = ExitAction(kind="none")


def record_price_tick(position: models.Position, price: float, now: dt.datetime | None = None) -> None:
    """Update the high/low water marks and the rolling sample buffer.

    Called on every monitor tick regardless of whether an exit fires, so the
    buffers stay populated even for positions that never trigger anything.

    Both extremes are tracked, not just the peak. The peak alone drives the
    trailing stop, but a post-mortem needs the trough too: a trade that
    closed +5% after dipping -30% is a different trade from one that never
    dipped, and the closing price cannot tell them apart.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    if position.highest_price_since_entry is None or price > position.highest_price_since_entry:
        position.highest_price_since_entry = price
    if position.lowest_price_since_entry is None or price < position.lowest_price_since_entry:
        position.lowest_price_since_entry = price

    samples = list(position.recent_prices or [])
    samples.append([now.isoformat(), price])
    position.recent_prices = samples[-MAX_RECENT_PRICE_SAMPLES:]


def record_liquidity_tick(position: models.Position, liquidity_usd: float | None) -> None:
    """Track pool depth alongside price.

    Seeds the entry level on the first reading rather than at fill time, so
    a position opened before this existed still gets a baseline instead of
    being permanently unassessable.
    """
    if liquidity_usd is None or liquidity_usd <= 0:
        return
    if position.liquidity_at_entry_usd is None:
        position.liquidity_at_entry_usd = liquidity_usd
    if position.lowest_liquidity_usd is None or liquidity_usd < position.lowest_liquidity_usd:
        position.lowest_liquidity_usd = liquidity_usd


def evaluate_liquidity(
    position: models.Position, liquidity_usd: float | None
) -> ExitAction:
    """Close, trim, or hold based on how much of the pool is left.

    Deliberately separate from the price-based ladder in ExitManager,
    because it answers a different question. Every other exit asks "is this
    trade going badly?"; this one asks "will there still be something to
    sell into?". A drained pool looks fine on price right up until the stop
    fills at whatever is left, which is usually nothing.

    A MISSING reading is not a drop. The feed going quiet for one tick is
    common and would otherwise dump every open position at once - turning a
    provider hiccup into a portfolio-wide market sell.
    """
    if not settings.LIQUIDITY_EXIT_ENABLED or liquidity_usd is None or liquidity_usd <= 0:
        return NO_EXIT

    if liquidity_usd < settings.LIQUIDITY_EXIT_FLOOR_USD:
        return ExitAction(
            kind="full",
            reason=(
                f"liquidity ${liquidity_usd:,.0f} is below the ${settings.LIQUIDITY_EXIT_FLOOR_USD:,.0f} "
                "floor - too thin to exit cleanly at any price"
            ),
        )

    entry = position.liquidity_at_entry_usd
    if not entry or entry <= 0:
        return NO_EXIT

    remaining = liquidity_usd / entry
    if remaining <= (1 - settings.LIQUIDITY_EXIT_DROP_PCT):
        return ExitAction(
            kind="full",
            reason=(
                f"liquidity fell {(1 - remaining) * 100:.0f}% since entry "
                f"(${entry:,.0f} to ${liquidity_usd:,.0f}) - the pool is being drained"
            ),
        )
    if remaining <= (1 - settings.LIQUIDITY_WARN_DROP_PCT):
        return ExitAction(
            kind="partial",
            fraction=0.5,
            reason=(
                f"liquidity fell {(1 - remaining) * 100:.0f}% since entry "
                f"(${entry:,.0f} to ${liquidity_usd:,.0f}) - trimming while there is depth to sell into"
            ),
        )
    return NO_EXIT


class ExitManager:
    def __init__(
        self,
        *,
        trailing_enabled: bool | None = None,
        trailing_activation_pct: float | None = None,
        trailing_distance_pct: float | None = None,
        break_even_enabled: bool | None = None,
        break_even_trigger_pct: float | None = None,
        break_even_buffer_pct: float | None = None,
        partial_enabled: bool | None = None,
        partial_trigger_pct: float | None = None,
        partial_size_pct: float | None = None,
        momentum_enabled: bool | None = None,
        momentum_lookback: int | None = None,
        momentum_drop_pct: float | None = None,
        trend_reversal_enabled: bool | None = None,
        trend_reversal_min_samples: int | None = None,
        time_exit_enabled: bool | None = None,
        max_position_age_hours: float | None = None,
    ):
        """Every parameter defaults to the matching `settings.*` value when
        omitted, so `ExitManager()` behaves exactly as before - this is only
        for callers (the backtester) that need deterministic, config-driven
        exit rules independent of the live environment's `.env`, while
        running the IDENTICAL exit logic live trading uses.
        """
        self.trailing_enabled = trailing_enabled if trailing_enabled is not None else settings.TRAILING_STOP_ENABLED
        self.trailing_activation_pct = (
            trailing_activation_pct if trailing_activation_pct is not None
            else settings.TRAILING_STOP_ACTIVATION_PCT
        )
        self.trailing_distance_pct = (
            trailing_distance_pct if trailing_distance_pct is not None else settings.TRAILING_STOP_DISTANCE_PCT
        )

        self.break_even_enabled = break_even_enabled if break_even_enabled is not None else settings.BREAK_EVEN_ENABLED
        self.break_even_trigger_pct = (
            break_even_trigger_pct if break_even_trigger_pct is not None else settings.BREAK_EVEN_TRIGGER_PCT
        )
        self.break_even_buffer_pct = (
            break_even_buffer_pct if break_even_buffer_pct is not None else settings.BREAK_EVEN_BUFFER_PCT
        )

        self.partial_enabled = partial_enabled if partial_enabled is not None else settings.PARTIAL_TAKE_PROFIT_ENABLED
        self.partial_trigger_pct = (
            partial_trigger_pct if partial_trigger_pct is not None else settings.PARTIAL_TAKE_PROFIT_TRIGGER_PCT
        )
        self.partial_size_pct = (
            partial_size_pct if partial_size_pct is not None else settings.PARTIAL_TAKE_PROFIT_SIZE_PCT
        )

        self.momentum_enabled = momentum_enabled if momentum_enabled is not None else settings.MOMENTUM_EXIT_ENABLED
        self.momentum_lookback = (
            momentum_lookback if momentum_lookback is not None else settings.MOMENTUM_EXIT_LOOKBACK_SAMPLES
        )
        self.momentum_drop_pct = (
            momentum_drop_pct if momentum_drop_pct is not None else settings.MOMENTUM_EXIT_DROP_PCT
        )

        self.trend_reversal_enabled = (
            trend_reversal_enabled if trend_reversal_enabled is not None else settings.TREND_REVERSAL_EXIT_ENABLED
        )
        self.trend_reversal_min_samples = (
            trend_reversal_min_samples if trend_reversal_min_samples is not None
            else settings.TREND_REVERSAL_MIN_SAMPLES
        )

        self.time_exit_enabled = time_exit_enabled if time_exit_enabled is not None else settings.TIME_BASED_EXIT_ENABLED
        self.max_position_age_hours = (
            max_position_age_hours if max_position_age_hours is not None else settings.MAX_POSITION_AGE_HOURS
        )

    def evaluate(
        self, position: models.Position, current_price: float, now: dt.datetime | None = None
    ) -> ExitAction:
        now = now or dt.datetime.now(dt.timezone.utc)
        record_price_tick(position, current_price, now)

        entry = position.entry_price
        gain_pct = (current_price / entry) - 1 if entry else 0.0

        # 1. break-even stop (adjusts the stop, does not itself exit)
        if self.break_even_enabled and not position.break_even_applied and gain_pct >= self.break_even_trigger_pct:
            new_stop = entry * (1 + self.break_even_buffer_pct)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
            position.break_even_applied = True

        # 2. trailing stop (adjusts the stop, does not itself exit)
        if self.trailing_enabled:
            peak = position.highest_price_since_entry or current_price
            peak_gain_pct = (peak / entry) - 1 if entry else 0.0
            if peak_gain_pct >= self.trailing_activation_pct:
                position.trailing_stop_active = True
            if position.trailing_stop_active:
                trail_stop = peak * (1 - self.trailing_distance_pct)
                if trail_stop > position.stop_loss:
                    position.stop_loss = trail_stop

        # 3. stop-loss / take-profit - the position's non-negotiable risk
        #    boundary. Checked AFTER the ratchets above so a breach is
        #    labeled with whichever stop mechanism is actually governing
        #    right now: once trailing has activated, an unmoved-since-entry
        #    stop is no longer reachable except through the trailing logic
        #    that has been ratcheting it, so a breach there is a trailing
        #    stop, not the original fixed one - the trade journal should
        #    say so rather than a generic "stop-loss hit".
        if current_price <= position.stop_loss:
            if position.trailing_stop_active:
                peak = position.highest_price_since_entry or current_price
                return ExitAction("full", f"trailing stop hit at ${current_price:.8f} (peak ${peak:.8f})")
            if position.break_even_applied:
                return ExitAction("full", f"break-even stop hit at ${current_price:.8f}")
            return ExitAction("full", f"stop-loss hit at ${current_price:.8f}")
        if current_price >= position.take_profit:
            return ExitAction("full", f"take-profit hit at ${current_price:.8f}")

        # 4. partial profit-take (fires once)
        if self.partial_enabled and not position.partial_exit_taken and gain_pct >= self.partial_trigger_pct:
            position.partial_exit_taken = True
            return ExitAction(
                "partial",
                f"partial profit-take at +{gain_pct * 100:.1f}% (${current_price:.8f})",
                fraction=self.partial_size_pct,
            )

        # 5. momentum-loss exit
        if self.momentum_enabled:
            reason = self._momentum_loss(position, current_price)
            if reason:
                return ExitAction("full", reason)

        # 6. trend-reversal exit
        if self.trend_reversal_enabled:
            reason = self._trend_reversal(position)
            if reason:
                return ExitAction("full", reason)

        # 7. time-based exit
        if self.time_exit_enabled and position.opened_at:
            opened = position.opened_at
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=dt.timezone.utc)
            age_hours = (now - opened).total_seconds() / 3600
            if age_hours >= self.max_position_age_hours:
                return ExitAction("full", f"max holding time reached ({age_hours:.1f}h)")

        return NO_EXIT

    def _momentum_loss(self, position: models.Position, current_price: float) -> str | None:
        samples = position.recent_prices or []
        if len(samples) < 2:
            return None
        window = samples[-self.momentum_lookback:]
        recent_peak = max(p for _, p in window)
        if recent_peak <= 0:
            return None
        drop = (recent_peak - current_price) / recent_peak
        if drop >= self.momentum_drop_pct:
            return f"momentum loss: price fell {drop * 100:.1f}% from recent peak ${recent_peak:.8f}"
        return None

    def _trend_reversal(self, position: models.Position) -> str | None:
        samples = position.recent_prices or []
        if len(samples) < self.trend_reversal_min_samples:
            return None
        prices = [p for _, p in samples[-self.trend_reversal_min_samples:]]

        # Crude pivot-high detection over the small sample window: the
        # position ran up to a peak, then printed two strictly lower
        # "highs" in a row after it - a real turn, not one noisy tick.
        peak_idx = max(range(len(prices)), key=lambda i: prices[i])
        after_peak = prices[peak_idx + 1:]
        if len(after_peak) < 2:
            return None  # peak too recent to confirm a reversal yet
        if after_peak[-1] < after_peak[-2] < prices[peak_idx]:
            return f"trend reversal: lower highs after peak ${prices[peak_idx]:.8f}"
        return None
