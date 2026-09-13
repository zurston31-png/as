"""Technical indicators, as pure functions over plain lists.

Every function takes oldest-first values and returns oldest-first values,
padded at the front with None for the warm-up period so the output always
lines up index-for-index with the input. A caller reading `rsi(closes)[-1]`
gets the value for the most recent candle, and a None means "not enough
history yet" rather than a silently wrong number.

No numpy or pandas on purpose. These series are hundreds of points, where
the speed difference does not matter, and both libraries ship compiled
wheels that break installs on new Python versions — which this project has
already been bitten by once.

Conventions follow Wilder for RSI and ATR (smoothing factor 1/n, not 2/(n+1))
since that is what charting platforms display.
"""
from __future__ import annotations

import math

Series = list[float | None]


# ---------------------------------------------------------------------------
# moving averages
# ---------------------------------------------------------------------------

def sma(values: list[float], period: int) -> Series:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: Series = [None] * len(values)
    if len(values) < period:
        return out
    window = sum(values[:period])
    out[period - 1] = window / period
    for i in range(period, len(values)):
        window += values[i] - values[i - period]
        out[i] = window / period
    return out


def ema(values: list[float], period: int) -> Series:
    """Exponential moving average, seeded with the first SMA.

    Seeding from an SMA rather than the first value is what charting
    platforms do; starting from values[0] makes the early output diverge
    noticeably.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    out: Series = [None] * len(values)
    if len(values) < period:
        return out

    multiplier = 2 / (period + 1)
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = (values[i] - current) * multiplier + current
        out[i] = current
    return out


def wilder_smooth(values: list[float], period: int) -> Series:
    """Wilder's smoothing (RMA), used by RSI and ATR."""
    out: Series = [None] * len(values)
    if len(values) < period:
        return out
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = (current * (period - 1) + values[i]) / period
        out[i] = current
    return out


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------

def rsi(closes: list[float], period: int = 14) -> Series:
    """Relative Strength Index, Wilder's method. 0-100."""
    out: Series = [None] * len(closes)
    if len(closes) <= period:
        return out

    gains, losses = [], []
    for prev, curr in zip(closes, closes[1:]):
        change = curr - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)

    for i in range(len(gains)):
        g, l = avg_gain[i], avg_loss[i]
        if g is None or l is None:
            continue
        if l == 0:
            out[i + 1] = 100.0
        else:
            rs = g / l
            out[i + 1] = 100 - (100 / (1 + rs))
    return out


def macd(
    closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[Series, Series, Series]:
    """MACD line, signal line, histogram."""
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)

    macd_line: Series = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(fast_ema, slow_ema)
    ]

    # The signal line is an EMA of the MACD line, which only exists once the
    # slow EMA has warmed up, so compute it on the defined tail and pad back.
    defined = [(i, v) for i, v in enumerate(macd_line) if v is not None]
    signal_line: Series = [None] * len(closes)
    if len(defined) >= signal:
        values = [v for _, v in defined]
        smoothed = ema(values, signal)
        for (original_index, _), smoothed_value in zip(defined, smoothed):
            signal_line[original_index] = smoothed_value

    histogram: Series = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, histogram


def momentum(closes: list[float], period: int = 10) -> Series:
    """Rate of change over `period` candles, as a fraction (0.05 == +5%)."""
    out: Series = [None] * len(closes)
    for i in range(period, len(closes)):
        past = closes[i - period]
        if past:
            out[i] = (closes[i] - past) / past
    return out


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------

def true_range(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    out = [highs[0] - lows[0]] if highs else []
    for i in range(1, len(highs)):
        prev_close = closes[i - 1]
        out.append(max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        ))
    return out


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> Series:
    """Average True Range, Wilder's smoothing. Absolute price units."""
    if not highs:
        return []
    return wilder_smooth(true_range(highs, lows, closes), period)


def atr_percent(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> Series:
    """ATR as a fraction of price, which is comparable across tokens."""
    values = atr(highs, lows, closes, period)
    return [
        (a / c) if (a is not None and c) else None
        for a, c in zip(values, closes)
    ]


def bollinger_bands(
    closes: list[float], period: int = 20, std_devs: float = 2.0
) -> tuple[Series, Series, Series]:
    """Upper band, middle (SMA), lower band."""
    middle = sma(closes, period)
    upper: Series = [None] * len(closes)
    lower: Series = [None] * len(closes)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1: i + 1]
        mean = middle[i]
        if mean is None:
            continue
        variance = sum((v - mean) ** 2 for v in window) / period
        deviation = math.sqrt(variance)
        upper[i] = mean + deviation * std_devs
        lower[i] = mean - deviation * std_devs

    return upper, middle, lower


def bollinger_width(closes: list[float], period: int = 20, std_devs: float = 2.0) -> Series:
    """Band width relative to the middle band — a squeeze/expansion measure."""
    upper, middle, lower = bollinger_bands(closes, period, std_devs)
    return [
        ((u - l) / m) if (u is not None and l is not None and m) else None
        for u, m, l in zip(upper, middle, lower)
    ]


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------

def vwap(
    highs: list[float], lows: list[float], closes: list[float], volumes: list[float],
    period: int | None = None,
) -> Series:
    """Volume weighted average price.

    Crypto trades continuously with no daily session to anchor to, so this
    defaults to a cumulative VWAP over the whole series. Pass `period` for a
    rolling window instead.
    """
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    out: Series = [None] * len(closes)

    if period is None:
        cumulative_pv = 0.0
        cumulative_v = 0.0
        for i in range(len(closes)):
            cumulative_pv += typical[i] * volumes[i]
            cumulative_v += volumes[i]
            out[i] = (cumulative_pv / cumulative_v) if cumulative_v else None
        return out

    for i in range(period - 1, len(closes)):
        window_v = sum(volumes[i - period + 1: i + 1])
        if window_v:
            window_pv = sum(typical[j] * volumes[j] for j in range(i - period + 1, i + 1))
            out[i] = window_pv / window_v
    return out


def relative_volume(volumes: list[float], period: int = 20) -> Series:
    """Current volume as a multiple of its own average. 2.0 == twice normal."""
    averages = sma(volumes, period)
    return [
        (v / a) if (a is not None and a > 0) else None
        for v, a in zip(volumes, averages)
    ]


def volume_spike(volumes: list[float], period: int = 20, multiple: float = 2.0) -> list[bool]:
    """Whether each bar's volume exceeds `multiple` times its average."""
    relative = relative_volume(volumes, period)
    return [(r is not None and r >= multiple) for r in relative]


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def swing_highs(highs: list[float], lookback: int = 3) -> list[int]:
    """Indices of pivot highs — bars higher than `lookback` bars either side."""
    out = []
    for i in range(lookback, len(highs) - lookback):
        window = highs[i - lookback: i + lookback + 1]
        if highs[i] == max(window) and window.count(highs[i]) == 1:
            out.append(i)
    return out


def swing_lows(lows: list[float], lookback: int = 3) -> list[int]:
    out = []
    for i in range(lookback, len(lows) - lookback):
        window = lows[i - lookback: i + lookback + 1]
        if lows[i] == min(window) and window.count(lows[i]) == 1:
            out.append(i)
    return out


def support_resistance(
    highs: list[float], lows: list[float], lookback: int = 3, max_levels: int = 3
) -> tuple[list[float], list[float]]:
    """The most recent pivot lows (support) and pivot highs (resistance).

    Deliberately simple: recent pivots, newest first. Clustering nearby
    levels would be an improvement, but an unclustered recent pivot is
    already a far better reference than a fixed percentage.
    """
    resistance = [highs[i] for i in reversed(swing_highs(highs, lookback))][:max_levels]
    support = [lows[i] for i in reversed(swing_lows(lows, lookback))][:max_levels]
    return support, resistance


def price_structure(highs: list[float], lows: list[float], lookback: int = 3) -> str:
    """Classify structure as uptrend / downtrend / ranging.

    Higher highs and higher lows is an uptrend; lower highs and lower lows a
    downtrend; anything mixed is ranging.
    """
    high_pivots = swing_highs(highs, lookback)
    low_pivots = swing_lows(lows, lookback)
    if len(high_pivots) < 2 or len(low_pivots) < 2:
        return "ranging"

    higher_highs = highs[high_pivots[-1]] > highs[high_pivots[-2]]
    higher_lows = lows[low_pivots[-1]] > lows[low_pivots[-2]]

    if higher_highs and higher_lows:
        return "uptrend"
    if not higher_highs and not higher_lows:
        return "downtrend"
    return "ranging"


def n_period_high(highs: list[float], period: int = 20) -> Series:
    """Highest high of the PRECEDING `period` bars, excluding the current one.

    Excluding the current bar is what makes a breakout test meaningful — a
    bar cannot break out above a level it helped define.
    """
    out: Series = [None] * len(highs)
    for i in range(period, len(highs)):
        out[i] = max(highs[i - period: i])
    return out


def n_period_low(lows: list[float], period: int = 20) -> Series:
    out: Series = [None] * len(lows)
    for i in range(period, len(lows)):
        out[i] = min(lows[i - period: i])
    return out


def trend_direction(closes: list[float]) -> str:
    """Trend from EMA stacking: bullish, bearish, or neutral.

    Needs 200 candles. With fewer, falls back to the 9/21 relationship and
    reports neutral rather than pretending to know.
    """
    if len(closes) >= 200:
        e50 = ema(closes, 50)[-1]
        e200 = ema(closes, 200)[-1]
        e9 = ema(closes, 9)[-1]
        e21 = ema(closes, 21)[-1]
        if None in (e9, e21, e50, e200):
            return "neutral"
        if e9 > e21 > e50 > e200:
            return "bullish"
        if e9 < e21 < e50 < e200:
            return "bearish"
        if e50 > e200:
            return "bullish"
        if e50 < e200:
            return "bearish"
        return "neutral"

    if len(closes) >= 21:
        e9 = ema(closes, 9)[-1]
        e21 = ema(closes, 21)[-1]
        if e9 is None or e21 is None:
            return "neutral"
        if e9 > e21 * 1.001:
            return "bullish"
        if e9 < e21 * 0.999:
            return "bearish"
    return "neutral"


def crossed_above(fast: Series, slow: Series, index: int = -1) -> bool:
    """Whether `fast` crossed above `slow` on the bar at `index`."""
    try:
        f_now, s_now = fast[index], slow[index]
        f_prev, s_prev = fast[index - 1], slow[index - 1]
    except IndexError:
        return False
    if None in (f_now, s_now, f_prev, s_prev):
        return False
    return f_prev <= s_prev and f_now > s_now


def crossed_below(fast: Series, slow: Series, index: int = -1) -> bool:
    try:
        f_now, s_now = fast[index], slow[index]
        f_prev, s_prev = fast[index - 1], slow[index - 1]
    except IndexError:
        return False
    if None in (f_now, s_now, f_prev, s_prev):
        return False
    return f_prev >= s_prev and f_now < s_now
