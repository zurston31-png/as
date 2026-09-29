"""Pure-python indicator maths.

No numpy on purpose: the whole runtime stays installable with `pip install
fastapi uvicorn anthropic pyyaml`, which matters when you want this running on
the same machine as your charts without a toolchain.

Every function returns a list the same length as its input, padded at the front
with ``None`` for bars where the value is not yet defined. That keeps index
alignment with the candle series trivial.
"""

from __future__ import annotations

from typing import Optional, Sequence

Num = Optional[float]


def sma(values: Sequence[float], period: int) -> list[Num]:
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[Num] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        out.append(running / period if i >= period - 1 else None)
    return out


def ema(values: Sequence[float], period: int) -> list[Num]:
    """Standard EMA seeded with the SMA of the first `period` bars."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[Num] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1.0)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> list[Num]:
    """Wilder's RSI."""
    out: list[Num] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[Num]:
    out: list[Num] = [None] * len(highs)
    for i in range(len(highs)):
        if i == 0:
            out[i] = highs[i] - lows[i]
            continue
        prev_close = closes[i - 1]
        out[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[Num]:
    """Wilder-smoothed ATR."""
    tr = true_range(highs, lows, closes)
    out: list[Num] = [None] * len(highs)
    if len(highs) < period:
        return out
    seed = sum(v for v in tr[:period] if v is not None) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(highs)):
        prev = (prev * (period - 1) + (tr[i] or 0.0)) / period
        out[i] = prev
    return out


def session_vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    session_starts: Sequence[bool],
) -> list[Num]:
    """Volume-weighted average price, reset wherever ``session_starts`` is True.

    Falls back to a plain typical-price average when volume is absent (some
    index and forex feeds report zero volume) so the line is still usable.
    """
    out: list[Num] = []
    cum_pv = cum_v = 0.0
    cum_p = 0.0
    count = 0
    for i in range(len(closes)):
        if session_starts[i] or i == 0:
            cum_pv = cum_v = cum_p = 0.0
            count = 0
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        vol = volumes[i] if i < len(volumes) else 0.0
        cum_pv += tp * vol
        cum_v += vol
        cum_p += tp
        count += 1
        out.append(cum_pv / cum_v if cum_v > 0 else cum_p / count)
    return out


def rolling_mean(values: Sequence[float], period: int) -> list[Num]:
    return sma(values, period)


def stdev(values: Sequence[float], period: int) -> list[Num]:
    out: list[Num] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = sum(window) / period
        var = sum((v - mean) ** 2 for v in window) / period
        out[i] = var ** 0.5
    return out


def crossed_above(fast: Sequence[Num], slow: Sequence[Num], lookback: int = 1) -> Optional[int]:
    """Bars since `fast` last crossed up through `slow`, or None.

    0 means the cross happened on the most recent bar.
    """
    n = len(fast)
    for back in range(0, min(lookback, n - 1)):
        i = n - 1 - back
        f0, s0, f1, s1 = fast[i], slow[i], fast[i - 1], slow[i - 1]
        if None in (f0, s0, f1, s1):
            continue
        if f1 <= s1 and f0 > s0:
            return back
    return None


def crossed_below(fast: Sequence[Num], slow: Sequence[Num], lookback: int = 1) -> Optional[int]:
    n = len(fast)
    for back in range(0, min(lookback, n - 1)):
        i = n - 1 - back
        f0, s0, f1, s1 = fast[i], slow[i], fast[i - 1], slow[i - 1]
        if None in (f0, s0, f1, s1):
            continue
        if f1 >= s1 and f0 < s0:
            return back
    return None


def last(values: Sequence[Num]) -> Num:
    for v in reversed(values):
        if v is not None:
            return v
    return None
