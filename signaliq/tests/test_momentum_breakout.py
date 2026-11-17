"""Strategy A (momentum breakout) rule tests.

SYNTHETIC TEST FIXTURES ONLY. Every price is invented to put one rule on the
edge of firing. Nothing here is market data and no number is a performance result.
"""

from dataclasses import replace
from datetime import date

import pytest

from app.strategies.data import DailyBar
from app.strategies.momentum_breakout import (
    STRATEGY_NAME,
    MomentumBreakout,
    MomentumBreakoutParameters,
)
from app.strategies.simulator import (
    DECISION_SKIPPED,
    DECISION_TAKEN,
    EXIT_SIGNAL,
    EXIT_STOP,
    REASON_GAP_ABOVE_LIMIT,
    simulate,
)
from tests.strategy_fixtures import random_walk_bars, sessions_between

#: Small windows so every rule can be checked by hand on a few bars.
SMALL = MomentumBreakoutParameters(
    breakout_window=3, exit_window=2, trend_window=4, atr_window=2, max_positions=5
)
SESSIONS = sessions_between(date(2024, 1, 2), date(2024, 3, 28))


def up_then_down(seed):
    """A seeded random walk that rises, then falls, so both entries and exits occur."""
    up = random_walk_bars(seed, SESSIONS[:40], drift=0.008)
    down = random_walk_bars(seed + 100, SESSIONS[40:], start=up[-1].close, drift=-0.012)
    return up + down


def bars_from(prices, spread=1.0):
    """Bars whose close is ``price``, high = close + spread, low = close - spread, open = close."""
    return [
        DailyBar(SESSIONS[i], price, price + spread, price - spread, price, 1000)
        for i, price in enumerate(prices)
    ]


def plans(prices, parameters=SMALL, spread=1.0):
    return MomentumBreakout(parameters).build_plans(bars_from(prices, spread))


# =============================================================================
# Entry signal
# =============================================================================


def test_close_above_the_prior_high_and_the_trend_average_is_an_entry_signal():
    # Prior 3 highs before bar 4: 11, 11, 11. Close 12.5 > 11, and > 4-bar average 10.625.
    result = plans([10, 10, 10, 10, 12.5])

    assert [plan.entry_signal for plan in result] == [False, False, False, False, True]
    signal = result[4]
    assert signal.indicators == {
        "close": 12.5,
        "prior_3_session_high": 11.0,
        "sma_4": pytest.approx(10.625),
        "atr_2": pytest.approx((1 * 2 + (13.5 - 10)) / 2),  # true ranges 2.0 and 3.5
        "change_3_sessions": pytest.approx(0.25),
    }
    assert signal.rank_value == pytest.approx(0.25)


def test_close_equal_to_the_prior_high_is_not_a_breakout():
    assert plans([10, 10, 10, 10, 11.0])[4].entry_signal is False
    assert plans([10, 10, 10, 10, 11.0001])[4].entry_signal is True


def test_the_bar_being_judged_is_not_part_of_its_own_breakout_level():
    # The high of bar 4 is 13.5; the level it is compared with is still 11 (bars 1-3).
    result = plans([10, 10, 10, 10, 12.5])

    assert result[4].indicators["prior_3_session_high"] == 11.0


def test_no_entry_below_the_trend_average_even_on_a_breakout():
    # Bar 6 closes at 12, above the prior 3-bar high of 9.5 (bars 3-5).
    prices = [30, 30, 30, 9, 9, 9, 12]

    # 6-bar average (30+30+9+9+9+12)/6 = 16.5: the close is below it, so no entry.
    below = plans(prices, replace(SMALL, trend_window=6), spread=0.5)[6]
    # 4-bar average (9+9+9+12)/4 = 9.75: the same breakout now counts.
    above = plans(prices, replace(SMALL, trend_window=4), spread=0.5)[6]

    assert below.entry_signal is False
    assert above.entry_signal is True
    assert above.indicators["prior_3_session_high"] == 9.5


def test_close_exactly_on_the_trend_average_is_not_above_it():
    # A 1-bar "average" equals the close itself: never strictly above.
    parameters = replace(SMALL, trend_window=1)

    assert plans([10, 10, 10, 10, 12.5], parameters)[4].entry_signal is False


def test_no_signal_until_every_indicator_has_enough_history():
    parameters = replace(SMALL, breakout_window=3, trend_window=6)
    # Each close beats the previous high (close + 1) by 1, so every bar is a breakout once ready.
    rising = [10, 12, 14, 16, 18, 20, 22, 24]

    result = plans(rising, parameters)

    # Bar 5 is the first with a 6-bar average; nothing may fire earlier.
    assert [plan.entry_signal for plan in result[:5]] == [False] * 5
    assert result[5].entry_signal is True


# =============================================================================
# Exit signal
# =============================================================================


def test_close_below_the_prior_lows_is_an_exit_signal():
    # Prior 2 lows before bar 3: 9 and 9. Close 7.9 < 9.
    result = plans([10, 10, 10, 7.9])

    assert result[3].exit_signal is True
    assert [plan.exit_signal for plan in result[:3]] == [False, False, False]


def test_close_equal_to_the_prior_low_is_not_an_exit():
    assert plans([10, 10, 10, 9.0])[3].exit_signal is False


# =============================================================================
# Stop distance and entry limit
# =============================================================================


def test_stop_distance_is_two_atr_and_the_entry_limit_is_one_atr_above_the_close():
    result = plans([10, 10, 10, 10, 12.5])

    atr = result[4].indicators["atr_2"]
    assert result[4].stop_distance == pytest.approx(2.0 * atr)
    assert result[4].max_entry_price == pytest.approx(12.5 + 1.0 * atr)


def test_plans_without_an_entry_signal_carry_no_stop_or_limit():
    result = plans([10, 10, 10, 10, 10])

    assert all(plan.stop_distance is None and plan.max_entry_price is None for plan in result)
    assert all(plan.indicators == {} for plan in result)


# =============================================================================
# No look-ahead at the strategy level
# =============================================================================


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_each_plan_is_the_same_when_later_bars_do_not_exist(seed):
    parameters = replace(SMALL, breakout_window=10, exit_window=5, trend_window=20, atr_window=5)
    bars = up_then_down(seed)
    strategy = MomentumBreakout(parameters)
    full = strategy.build_plans(bars)

    assert any(plan.entry_signal for plan in full), "the fixture must produce signals"
    assert any(plan.exit_signal for plan in full)
    for i in range(len(bars)):
        assert strategy.build_plans(bars[: i + 1]) == full[: i + 1], f"plan {i} used a later bar"


def test_changing_the_future_never_changes_an_earlier_plan():
    parameters = replace(SMALL, breakout_window=10, exit_window=5, trend_window=20, atr_window=5)
    bars = up_then_down(9)
    cut = 40
    future = [
        replace(bar, open=bar.open * 3, high=bar.high * 5, low=bar.low / 4)
        for bar in bars[cut + 1 :]
    ]
    strategy = MomentumBreakout(parameters)

    assert (
        strategy.build_plans(bars[: cut + 1] + future)[: cut + 1]
        == strategy.build_plans(bars)[: cut + 1]
    )


# =============================================================================
# Through the simulator
# =============================================================================


def test_a_breakout_is_bought_at_the_next_open_with_the_agreed_size_and_stop():
    prices = [10, 10, 10, 10, 12.5, 13, 14, 15]
    bars = {"TESTA": bars_from(prices)}

    result = simulate(
        MomentumBreakout(SMALL),
        bars,
        period_start=SESSIONS[0],
        period_end=SESSIONS[7],
        starting_equity=100_000.0,
        slippage_bps=0.0,
    )

    signal = next(s for s in result.signals if s.decision == DECISION_TAKEN)
    atr = signal.indicators["atr_2"]  # 2.75
    # Risk 0.5 % = 500 / (2 * 2.75) = 90.9 -> 90 shares; 20 % cap = 20,000 / 12.5 = 1,600.
    assert signal.planned_shares == 90
    trade = result.trades[0]
    assert (trade.entry_session, trade.entry_reference_price) == (SESSIONS[5], 13)
    assert trade.initial_stop == pytest.approx(13 - 2 * atr)


def test_entry_is_cancelled_when_the_next_open_runs_more_than_one_atr_above_the_close():
    bars = bars_from([10, 10, 10, 10, 12.5, 13, 13, 13])
    atr = 2.75
    bars[5] = replace(bars[5], open=12.5 + atr + 0.01, high=20)

    result = simulate(
        MomentumBreakout(SMALL),
        {"TESTA": bars},
        period_start=SESSIONS[0],
        period_end=SESSIONS[7],
        starting_equity=100_000.0,
        slippage_bps=0.0,
    )

    first = result.signals[0]
    assert (first.signal_session, first.decision, first.reason) == (
        SESSIONS[4],
        DECISION_SKIPPED,
        REASON_GAP_ABOVE_LIMIT,
    )


def test_the_exit_signal_closes_the_position_at_the_next_open():
    prices = [10, 10, 10, 10, 12.5, 13, 14, 15, 15, 10.5, 10]
    bars = {"TESTA": [replace(bar, low=bar.close - 0.5) for bar in bars_from(prices)]}

    result = simulate(
        MomentumBreakout(replace(SMALL, stop_atr_multiple=10.0)),  # keep the stop out of the way
        bars,
        period_start=SESSIONS[0],
        period_end=SESSIONS[10],
        starting_equity=100_000.0,
        slippage_bps=0.0,
    )

    (trade,) = result.trades
    assert trade.exit_reason == EXIT_SIGNAL
    assert trade.exit_session == SESSIONS[10]  # the close of 10.5 on SESSIONS[9] broke the lows
    assert trade.exit_reference_price == 10


def test_losing_trades_are_recorded():
    prices = [10, 10, 10, 10, 12.5, 13, 8, 8]
    bars = {"TESTA": bars_from(prices)}

    result = simulate(
        MomentumBreakout(SMALL),
        bars,
        period_start=SESSIONS[0],
        period_end=SESSIONS[7],
        starting_equity=100_000.0,
        slippage_bps=5.0,
    )

    (trade,) = result.trades
    assert trade.exit_reason in (EXIT_STOP, "stop_gap")
    assert trade.net_pnl < 0
    assert result.ending_equity < 100_000.0


# =============================================================================
# Parameters
# =============================================================================


def test_the_default_parameters_are_the_agreed_ones():
    assert MomentumBreakoutParameters().to_dict() == {
        "breakout_window": 55,
        "exit_window": 20,
        "trend_window": 200,
        "atr_window": 20,
        "stop_atr_multiple": 2.0,
        "max_entry_gap_atr": 1.0,
        "risk_fraction": 0.005,
        "max_position_fraction": 0.20,
        "max_positions": 5,
    }
    assert MomentumBreakout().name == STRATEGY_NAME == "momentum_breakout"


def test_parameters_can_be_changed_from_text_and_are_validated():
    parameters = MomentumBreakoutParameters.from_overrides(
        {"breakout_window": "40", "risk_fraction": "0.01"}
    )

    assert (parameters.breakout_window, parameters.risk_fraction) == (40, 0.01)
    with pytest.raises(ValueError, match="unknown parameter"):
        MomentumBreakoutParameters.from_overrides({"take_profit": "3"})
    with pytest.raises(ValueError, match="needs a number"):
        MomentumBreakoutParameters.from_overrides({"breakout_window": "fifty"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"breakout_window": 0},
        {"trend_window": 1.5},
        {"max_positions": 0},
        {"stop_atr_multiple": 0.0},
        {"risk_fraction": 0.0},
        {"risk_fraction": 1.5},
        {"max_position_fraction": -0.1},
    ],
)
def test_invalid_parameters_are_refused(overrides):
    with pytest.raises(ValueError):
        MomentumBreakoutParameters(**overrides)
