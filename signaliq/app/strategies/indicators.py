"""Technical indicators over a series of completed bars.

The one rule every function here obeys: **the value at position ``i`` is
computed only from inputs at positions ``<= i``.** Nothing looks at a later
bar. The tests check this for every indicator by cutting the series off and by
changing the future and confirming that no earlier value moves.

Two families exist, and the difference matters for trading rules:

* indicators that include the current bar (``sma``, ``atr``, ``rsi``, ...):
  the value at ``i`` is what is known once bar ``i`` has completed;
* ``prior_highest`` and ``prior_lowest``: the extreme of the ``window`` bars
  *before* bar ``i``, excluding bar ``i`` itself, so that bar ``i`` can be
  compared with it ("closed above the highest high of the prior 55 sessions").

Inputs are plain sequences of numbers, oldest first. Outputs are lists of the
same length holding ``None`` wherever there is not yet enough history. A value
is never guessed or carried over to fill such a position.
"""

import math
from collections.abc import Sequence

Series = list[float | None]


def _check_window(window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be a whole number of at least 1, got {window!r}")


def _check_same_length(*series: Sequence[float]) -> int:
    lengths = {len(values) for values in series}
    if len(lengths) != 1:
        raise ValueError(f"input series must have the same length, got {sorted(lengths)}")
    return lengths.pop()


def sma(values: Sequence[float], window: int) -> Series:
    """Simple moving average of the last ``window`` values, the current one included."""
    _check_window(window)
    return [
        math.fsum(values[i - window + 1 : i + 1]) / window if i >= window - 1 else None
        for i in range(len(values))
    ]


def prior_highest(values: Sequence[float], window: int) -> Series:
    """Highest of the ``window`` values BEFORE each position (the position itself excluded)."""
    _check_window(window)
    return [max(values[i - window : i]) if i >= window else None for i in range(len(values))]


def prior_lowest(values: Sequence[float], window: int) -> Series:
    """Lowest of the ``window`` values BEFORE each position (the position itself excluded)."""
    _check_window(window)
    return [min(values[i - window : i]) if i >= window else None for i in range(len(values))]


def rate_of_change(values: Sequence[float], window: int) -> Series:
    """Relative change against the value ``window`` positions earlier (0.10 means +10 %)."""
    _check_window(window)
    return [
        values[i] / values[i - window] - 1.0 if i >= window and values[i - window] != 0 else None
        for i in range(len(values))
    ]


def true_range(high: Sequence[float], low: Sequence[float], close: Sequence[float]) -> Series:
    """True range of each bar: the largest of high - low and the gaps to the previous close.

    The first bar has no previous close, so its true range is undefined.
    """
    length = _check_same_length(high, low, close)
    return [
        max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        if i >= 1
        else None
        for i in range(length)
    ]


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], window: int) -> Series:
    """Average true range: the simple mean of the last ``window`` true ranges.

    A plain mean is used (not Wilder's smoothing), so the value depends on
    exactly ``window`` + 1 bars and on nothing earlier.
    """
    _check_window(window)
    ranges = true_range(high, low, close)
    return [
        math.fsum(ranges[i - window + 1 : i + 1]) / window if i >= window else None  # type: ignore[arg-type]
        for i in range(len(ranges))
    ]


def rsi(values: Sequence[float], window: int) -> Series:
    """Relative strength index with Wilder's smoothing, from 0 to 100.

    The first value appears after ``window`` changes. Each later value is
    built from the one before it, so it depends on the whole history up to
    its own position, and on nothing after it.
    """
    _check_window(window)
    result: Series = [None] * len(values)
    if len(values) <= window:
        return result

    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    average_gain = math.fsum(max(change, 0.0) for change in changes[:window]) / window
    average_loss = math.fsum(max(-change, 0.0) for change in changes[:window]) / window

    def index(gain: float, loss: float) -> float:
        if loss == 0:
            # No losses in the window. With no gains either the series is flat.
            return 100.0 if gain > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    result[window] = index(average_gain, average_loss)
    for i in range(window + 1, len(values)):
        change = changes[i - 1]
        average_gain = (average_gain * (window - 1) + max(change, 0.0)) / window
        average_loss = (average_loss * (window - 1) + max(-change, 0.0)) / window
        result[i] = index(average_gain, average_loss)
    return result
