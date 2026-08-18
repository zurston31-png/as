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
    """Update peak-price tracking and the rolling sample buffer.

    Called on every monitor tick regardless of whether an exit fires, so the
    buffers stay populated even for positions that never trigger anything.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    if position.highest_price_since_entry is None or price > position.highest_price_since_entry:
        position.highest_price_since_entry = price

    samples = list(position.recent_prices or [])
    samples.append([now.isoformat(), price])
    position.recent_prices = samples[-MAX_RECENT_PRICE_SAMPLES:]


class ExitManager:
    def __init__(self):
        self.trailing_enabled = settings.TRAILING_STOP_ENABLED
        self.trailing_activation_pct = settings.TRAILING_STOP_ACTIVATION_PCT
        self.trailing_distance_pct = settings.TRAILING_STOP_DISTANCE_PCT

        self.break_even_enabled = settings.BREAK_EVEN_ENABLED
        self.break_even_trigger_pct = settings.BREAK_EVEN_TRIGGER_PCT
        self.break_even_buffer_pct = settings.BREAK_EVEN_BUFFER_PCT

        self.partial_enabled = settings.PARTIAL_TAKE_PROFIT_ENABLED
        self.partial_trigger_pct = settings.PARTIAL_TAKE_PROFIT_TRIGGER_PCT
        self.partial_size_pct = settings.PARTIAL_TAKE_PROFIT_SIZE_PCT

        self.momentum_enabled = settings.MOMENTUM_EXIT_ENABLED
        self.momentum_lookback = settings.MOMENTUM_EXIT_LOOKBACK_SAMPLES
        self.momentum_drop_pct = settings.MOMENTUM_EXIT_DROP_PCT

        self.trend_reversal_enabled = settings.TREND_REVERSAL_EXIT_ENABLED
        self.trend_reversal_min_samples = settings.TREND_REVERSAL_MIN_SAMPLES

        self.time_exit_enabled = settings.TIME_BASED_EXIT_ENABLED
        self.max_position_age_hours = settings.MAX_POSITION_AGE_HOURS

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
