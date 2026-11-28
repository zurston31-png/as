"""Indicator tests: correct values and, above all, no look-ahead.

Every series here is a SYNTHETIC TEST FIXTURE: numbers written by hand or drawn
from a seeded random generator. None of it is market data.
"""

import random

import pytest

from app.strategies.indicators import (
    atr,
    prior_highest,
    prior_lowest,
    rate_of_change,
    rsi,
    sma,
    true_range,
)

LENGTH = 120


def synthetic_bars(seed: int, length: int = LENGTH) -> dict[str, list[float]]:
    """A random walk with consistent highs and lows (fixture, not market data)."""
    generator = random.Random(seed)
    close, high, low = [], [], []
    price = 100.0
    for _ in range(length):
        price = max(1.0, price * (1.0 + generator.uniform(-0.04, 0.04)))
        top = price * (1.0 + generator.uniform(0.0, 0.03))
        bottom = price * (1.0 - generator.uniform(0.0, 0.03))
        close.append(round(price, 4))
        high.append(round(top, 4))
        low.append(round(bottom, 4))
    return {"close": close, "high": high, "low": low}


#: name -> function of a bars dict. Every indicator is listed here once, so the
#: no-look-ahead tests below cover all of them.
INDICATORS = {
    "sma_5": lambda bars: sma(bars["close"], 5),
    "sma_20": lambda bars: sma(bars["close"], 20),
    "prior_highest_10": lambda bars: prior_highest(bars["high"], 10),
    "prior_lowest_10": lambda bars: prior_lowest(bars["low"], 10),
    "rate_of_change_10": lambda bars: rate_of_change(bars["close"], 10),
    "true_range": lambda bars: true_range(bars["high"], bars["low"], bars["close"]),
    "atr_14": lambda bars: atr(bars["high"], bars["low"], bars["close"], 14),
    "rsi_2": lambda bars: rsi(bars["close"], 2),
    "rsi_14": lambda bars: rsi(bars["close"], 14),
}


def test_every_indicator_of_the_module_is_covered_by_the_look_ahead_tests():
    import app.strategies.indicators as module

    public = {
        name
        for name, value in vars(module).items()
        if callable(value) and not name.startswith("_") and value.__module__ == module.__name__
    }
    covered = {name.rstrip("0123456789").rstrip("_") for name in INDICATORS}

    assert public == covered


# =============================================================================
# No look-ahead
# =============================================================================


@pytest.mark.parametrize("name", sorted(INDICATORS))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_value_at_each_bar_is_the_same_when_later_bars_do_not_exist(name, seed):
    """Cut the series after bar i: the value at i must be what the full series gave."""
    indicator = INDICATORS[name]
    bars = synthetic_bars(seed)
    full = indicator(bars)

    assert len(full) == LENGTH
    for i in range(LENGTH):
        truncated = indicator({key: values[: i + 1] for key, values in bars.items()})
        assert truncated == full[: i + 1], f"{name} at bar {i} changed when later bars were removed"


@pytest.mark.parametrize("name", sorted(INDICATORS))
@pytest.mark.parametrize("cut", [1, 30, 77, LENGTH - 1])
def test_changing_later_bars_never_changes_an_earlier_value(name, cut):
    """Replace everything after bar ``cut`` with wildly different numbers."""
    indicator = INDICATORS[name]
    bars = synthetic_bars(seed=7)
    changed = {
        "close": bars["close"][: cut + 1] + [9999.0 + i for i in range(LENGTH - cut - 1)],
        "high": bars["high"][: cut + 1] + [19999.0 + i for i in range(LENGTH - cut - 1)],
        "low": bars["low"][: cut + 1] + [0.5] * (LENGTH - cut - 1),
    }

    before, after = indicator(bars), indicator(changed)

    assert after[: cut + 1] == before[: cut + 1]
    if cut < LENGTH - 1:
        assert after != before, "the changed future must be visible in the later values"


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_the_current_bar_cannot_be_seen_before_it_exists(name):
    """Appending one more bar leaves every existing value exactly as it was."""
    indicator = INDICATORS[name]
    bars = synthetic_bars(seed=11)
    shorter = {key: values[:-1] for key, values in bars.items()}

    assert indicator(bars)[:-1] == indicator(shorter)


def test_prior_extremes_exclude_the_current_bar():
    # The spike at position 3 must not be its own "prior high"; it first shows at position 4.
    highs = [10.0, 11.0, 12.0, 50.0, 13.0, 14.0]
    lows = [10.0, 9.0, 8.0, 1.0, 7.0, 6.0]

    assert prior_highest(highs, 3) == [None, None, None, 12.0, 50.0, 50.0]
    assert prior_lowest(lows, 3) == [None, None, None, 8.0, 1.0, 1.0]
    # So "today's value beats the prior extreme" is a real comparison, never value > itself.
    assert highs[3] > prior_highest(highs, 3)[3]
    assert lows[3] < prior_lowest(lows, 3)[3]


# =============================================================================
# Values, checked by hand
# =============================================================================


def test_sma_values():
    assert sma([1.0, 2.0, 3.0, 4.0, 5.0], 3) == [None, None, 2.0, 3.0, 4.0]
    assert sma([4.0, 6.0], 1) == [4.0, 6.0]
    assert sma([1.0, 2.0], 5) == [None, None]
    assert sma([], 3) == []


def test_prior_extremes_need_a_full_window_of_earlier_bars():
    assert prior_highest([5.0, 3.0, 4.0], 3) == [None, None, None]
    assert prior_highest([5.0, 3.0, 4.0, 1.0], 3) == [None, None, None, 5.0]
    assert prior_lowest([5.0, 3.0, 4.0, 9.0], 2) == [None, None, 3.0, 3.0]


def test_rate_of_change_values():
    assert rate_of_change([100.0, 110.0, 99.0, 121.0], 2) == [
        None,
        None,
        pytest.approx(-0.01),
        pytest.approx(0.10),
    ]
    assert rate_of_change([0.0, 5.0], 1) == [None, None]  # no division by zero


def test_true_range_uses_the_gap_to_the_previous_close():
    high = [10.0, 11.0, 15.0, 9.0]
    low = [9.0, 10.0, 14.0, 8.0]
    close = [9.5, 10.5, 14.5, 8.5]

    assert true_range(high, low, close) == [
        None,  # no previous close
        pytest.approx(1.5),  # high 11 - previous close 9.5
        pytest.approx(4.5),  # gap up: high 15 - previous close 10.5
        pytest.approx(6.5),  # gap down: previous close 14.5 - low 8
    ]


def test_atr_is_the_plain_mean_of_the_last_true_ranges():
    high = [10.0, 11.0, 15.0, 9.0, 10.0]
    low = [9.0, 10.0, 14.0, 8.0, 9.0]
    close = [9.5, 10.5, 14.5, 8.5, 9.5]
    # True ranges: None, 1.5, 4.5, 6.5, 1.5

    assert atr(high, low, close, 2) == [
        None,
        None,
        pytest.approx(3.0),
        pytest.approx(5.5),
        pytest.approx(4.0),
    ]
    assert atr(high, low, close, 4) == [None, None, None, None, pytest.approx(3.5)]
    assert atr(high, low, close, 5) == [None] * 5


def test_rsi_values_with_wilder_smoothing():
    # Changes: +1, +1, -1, +2. Window 2.
    values = [10.0, 11.0, 12.0, 11.0, 13.0]

    result = rsi(values, 2)

    assert result[:2] == [None, None]
    assert result[2] == 100.0  # two gains, no loss
    # gain (1*1 + 0)/2 = 0.5, loss (0*1 + 1)/2 = 0.5  ->  50
    assert result[3] == pytest.approx(50.0)
    # gain (0.5 + 2)/2 = 1.25, loss (0.5 + 0)/2 = 0.25  ->  100 - 100/6
    assert result[4] == pytest.approx(100.0 - 100.0 / 6.0)


def test_rsi_edge_cases():
    assert rsi([5.0, 5.0, 5.0, 5.0], 2) == [None, None, 50.0, 50.0]  # flat
    assert rsi([5.0, 4.0, 3.0], 2) == [None, None, 0.0]  # only losses
    assert rsi([5.0, 6.0], 2) == [None, None]  # not enough history
    assert all(
        value is None or 0.0 <= value <= 100.0 for value in rsi(synthetic_bars(5)["close"], 3)
    )


@pytest.mark.parametrize("window", [0, -1, 1.5, True, None])
def test_invalid_windows_are_refused(window):
    with pytest.raises(ValueError, match="window"):
        sma([1.0, 2.0, 3.0], window)
    with pytest.raises(ValueError, match="window"):
        rsi([1.0, 2.0, 3.0], window)


def test_series_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        true_range([1.0, 2.0], [1.0], [1.0, 2.0])


def test_indicators_do_not_modify_their_inputs():
    bars = synthetic_bars(seed=3)
    copy = {key: list(values) for key, values in bars.items()}

    for indicator in INDICATORS.values():
        indicator(bars)

    assert bars == copy
