"""Simulator tests: fills, stops, sizing, slots, bookkeeping and no look-ahead.

SYNTHETIC TEST FIXTURES ONLY. The prices are invented so that each expected
number can be worked out by hand. Nothing here is market data and no figure
below is a performance result.
"""

from datetime import date

import pytest

from app.strategies.data import DailyBar
from app.strategies.simulator import (
    DECISION_SKIPPED,
    DECISION_TAKEN,
    EXIT_PERIOD_END,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_STOP_GAP,
    REASON_ALREADY_IN_POSITION,
    REASON_GAP_ABOVE_LIMIT,
    REASON_INSUFFICIENT_CASH,
    REASON_NO_BAR,
    REASON_NO_FREE_SLOT,
    REASON_PERIOD_ENDS,
    REASON_POSITION_TOO_SMALL,
    simulate,
)
from tests.strategy_fixtures import SESSIONS, ScriptedStrategy, entry, exit_, flat_bars, make_bars

D = SESSIONS  # D[0] = 2 January 2024, ... D[8] = 12 January 2024
EQUITY = 100_000.0


def run(bars, plans, *, slippage_bps=0.0, start=D[0], end=D[-1], equity=EQUITY, **sizing):
    strategy = ScriptedStrategy(bars, plans, **sizing)
    result = simulate(
        strategy,
        bars,
        period_start=start,
        period_end=end,
        starting_equity=equity,
        slippage_bps=slippage_bps,
    )
    return result, strategy


# =============================================================================
# Entries
# =============================================================================


def test_signal_is_bought_at_the_next_open_and_sold_at_the_open_after_the_exit_signal():
    bars = {
        "TESTA": make_bars(
            [
                (50, 50, 50, 50),  # D0: entry signal at this close
                (51, 53, 50.5, 52),  # D1: bought at the open, 51
                (52, 55, 52, 54),
                (54, 58, 54, 57),  # D3: exit signal at this close
                (60, 61, 59, 60),  # D4: sold at the open, 60
                (60, 60, 60, 60),
            ]
        )
    }
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0), D[3]: exit_(D[3])}}

    result, _ = run(bars, plans, slippage_bps=10.0)

    (signal,) = result.signals
    assert (signal.symbol, signal.signal_session, signal.decision) == (
        "TESTA",
        D[0],
        DECISION_TAKEN,
    )
    assert signal.close == 50
    # 0.5 % of 100,000 = 500 at risk; 500 / 5 = 100 shares (the 20 % cap would allow 400).
    assert signal.planned_shares == 100

    (trade,) = result.trades
    assert (trade.entry_session, trade.exit_session, trade.exit_reason) == (D[1], D[4], EXIT_SIGNAL)
    assert trade.shares == 100
    assert trade.entry_reference_price == 51
    assert trade.entry_price == pytest.approx(51 * 1.001)  # 10 bp worse
    assert trade.initial_stop == pytest.approx(51 * 1.001 - 5.0)
    assert trade.exit_reference_price == 60
    assert trade.exit_price == pytest.approx(60 * 0.999)  # 10 bp worse
    assert trade.gross_pnl == pytest.approx(900.0)
    assert trade.slippage_cost == pytest.approx(100 * (51 * 0.001 + 60 * 0.001))
    assert trade.net_pnl == pytest.approx(900.0 - 11.1)
    assert trade.risk_amount == pytest.approx(500.0)
    assert trade.r_multiple == pytest.approx(888.9 / 500.0)
    assert trade.sessions_held == 3
    assert trade.equity_at_signal == EQUITY


def test_nothing_is_bought_on_the_signal_day_itself():
    bars = {"TESTA": make_bars([(50, 50, 50, 50), (70, 70, 70, 70), (70, 70, 70, 70)])}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    assert result.equity[0].open_positions == 0  # still flat at the close of the signal day
    assert result.equity[0].equity == EQUITY
    assert result.trades[0].entry_session == D[1]
    assert result.trades[0].entry_reference_price == 70  # the next open, whatever it is


def test_entry_is_not_filled_when_the_open_is_above_the_limit():
    bars = {"TESTA": make_bars([(50, 50, 50, 50), (53.01, 54, 53, 54), (54, 54, 54, 54)])}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0, limit=53.0)}}

    result, _ = run(bars, plans)

    (signal,) = result.signals
    assert (signal.decision, signal.reason) == (DECISION_SKIPPED, REASON_GAP_ABOVE_LIMIT)
    assert signal.planned_shares == 100  # what was planned stays on record
    assert result.trades == []
    assert result.ending_equity == EQUITY


def test_entry_is_filled_when_the_open_is_exactly_at_the_limit():
    bars = {"TESTA": make_bars([(50, 50, 50, 50), (53.0, 54, 53, 54), (54, 54, 54, 54)])}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0, limit=53.0)}}

    result, _ = run(bars, plans)

    assert result.signals[0].decision == DECISION_TAKEN


# =============================================================================
# Stops
# =============================================================================


def test_stop_inside_the_bar_is_sold_at_the_stop_price():
    bars = {
        "TESTA": make_bars(
            [(50, 50, 50, 50), (50, 51, 49, 50), (50, 50, 44.5, 48), (48, 48, 48, 48)]
        )
    }
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    (trade,) = result.trades
    assert trade.initial_stop == 45.0
    assert (trade.exit_session, trade.exit_reason) == (D[2], EXIT_STOP)
    assert trade.exit_reference_price == 45.0
    assert trade.net_pnl == pytest.approx(-500.0)  # exactly the planned risk without slippage
    assert trade.r_multiple == pytest.approx(-1.0)


def test_a_low_exactly_at_the_stop_counts_as_hit():
    bars = {"TESTA": make_bars([(50, 50, 50, 50), (50, 51, 45.0, 50), (50, 50, 50, 50)])}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    assert result.trades[0].exit_reason == EXIT_STOP


def test_a_gap_below_the_stop_is_sold_at_the_open_not_at_the_stop():
    bars = {
        "TESTA": make_bars([(50, 50, 50, 50), (50, 51, 49, 50), (40, 41, 39, 40), (40, 40, 40, 40)])
    }
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    (trade,) = result.trades
    assert (trade.exit_session, trade.exit_reason) == (D[2], EXIT_STOP_GAP)
    assert trade.exit_reference_price == 40  # worse than the stop of 45
    assert trade.net_pnl == pytest.approx(-1000.0)
    assert trade.r_multiple == pytest.approx(-2.0)  # a loss larger than planned is recorded as such


def test_a_position_can_be_stopped_out_on_its_entry_day():
    bars = {"TESTA": make_bars([(50, 50, 50, 50), (50, 50, 44, 46), (46, 46, 46, 46)])}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    (trade,) = result.trades
    assert (trade.entry_session, trade.exit_session) == (D[1], D[1])
    assert (trade.exit_reason, trade.sessions_held) == (EXIT_STOP, 0)


def test_the_stop_never_moves_after_entry():
    bars = {
        "TESTA": make_bars(
            [
                (50, 50, 50, 50),
                (50, 60, 50, 60),
                (60, 80, 60, 80),
                (80, 80, 45.5, 70),
                (70, 70, 44, 60),
            ]
        )
    }
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    (trade,) = result.trades
    assert (trade.exit_session, trade.exit_reference_price) == (D[4], 45.0)


def test_slippage_also_makes_a_stop_exit_worse():
    bars = {
        "TESTA": make_bars([(50, 50, 50, 50), (50, 50, 50, 50), (50, 50, 40, 44), (44, 44, 44, 44)])
    }
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans, slippage_bps=20.0)

    (trade,) = result.trades
    stop = 50 * 1.002 - 5.0
    assert trade.exit_reference_price == pytest.approx(stop)
    assert trade.exit_price == pytest.approx(stop * 0.998)
    assert trade.net_pnl < -trade.risk_amount  # worse than the planned loss
    assert trade.r_multiple < -1.0


# =============================================================================
# Position size
# =============================================================================


def test_position_is_capped_at_the_maximum_share_of_equity():
    bars = {"TESTA": flat_bars(50.0, 4)}
    # 500 at risk / 0.5 = 1,000 shares = 50,000; the 20 % cap allows 20,000 / 50 = 400.
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=0.5)}}

    result, _ = run(bars, plans)

    assert result.signals[0].planned_shares == 400
    assert result.trades[0].shares == 400


def test_size_follows_the_equity_at_the_signal_not_the_starting_equity():
    bars = {
        "TESTA": make_bars(
            [(50, 50, 50, 50), (50, 50, 50, 50), (50, 50, 45, 45), (45, 45, 45, 45)]
        ),
        "TESTB": make_bars([(20, 20, 20, 20)] * 6),
    }
    plans = {
        "TESTA": {D[0]: entry(D[0], stop_distance=5.0)},  # loses 500 on D2
        "TESTB": {D[3]: entry(D[3], stop_distance=1.0)},
    }

    result, _ = run(bars, plans)

    second = result.signals[1]
    assert second.symbol == "TESTB"
    # Equity is 99,500 after the loss: floor(0.005 * 99,500 / 1.0) = 497, not 500.
    assert second.planned_shares == 497
    assert result.trades[1].equity_at_signal == pytest.approx(99_500.0)


def test_signal_too_small_to_buy_one_share_is_recorded_and_skipped():
    bars = {"TESTA": flat_bars(50.0, 4)}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans, equity=900.0)  # 0.5 % of 900 = 4.5 at risk: less than one share

    (signal,) = result.signals
    assert (signal.decision, signal.reason, signal.planned_shares) == (
        DECISION_SKIPPED,
        REASON_POSITION_TOO_SMALL,
        0,
    )
    assert result.trades == []


def test_fewer_shares_are_bought_when_the_cash_does_not_cover_the_plan():
    bars = {
        "TESTA": make_bars([(50, 50, 50, 50), (50, 50, 50, 50), (50, 50, 50, 50)]),
        "TESTB": make_bars([(50, 50, 50, 50), (90, 90, 90, 90), (90, 90, 90, 90)]),
    }
    plans = {
        "TESTA": {D[0]: entry(D[0], stop_distance=0.1, rank=2.0)},
        "TESTB": {D[0]: entry(D[0], stop_distance=0.1, rank=1.0)},
    }

    # Each plan is the 60 % cap: 600 / 50 = 12 shares. TESTA costs 600 of the 1,000;
    # TESTB opens at 90, so only floor(400 / 90) = 4 of its 12 shares can be paid for.
    result, _ = run(bars, plans, equity=1000.0, max_position_fraction=0.60)

    by_symbol = {trade.symbol: trade for trade in result.trades}
    assert by_symbol["TESTA"].shares == 12
    assert by_symbol["TESTB"].shares == 4
    assert result.signals[1].planned_shares == 12
    assert min(point.cash for point in result.equity) >= 0  # the account never borrows


def test_signal_is_skipped_when_no_cash_is_left_for_a_single_share():
    bars = {
        "TESTA": make_bars([(50, 50, 50, 50), (50, 50, 50, 50), (50, 50, 50, 50)]),
        "TESTB": make_bars([(50, 50, 50, 50), (50, 50, 50, 50), (50, 50, 50, 50)]),
    }
    plans = {
        "TESTA": {D[0]: entry(D[0], stop_distance=0.1, rank=2.0)},
        "TESTB": {D[0]: entry(D[0], stop_distance=0.1, rank=1.0)},
    }

    result, _ = run(bars, plans, equity=1000.0, max_position_fraction=1.0)

    assert [(s.symbol, s.decision, s.reason) for s in result.signals] == [
        ("TESTA", DECISION_TAKEN, None),
        ("TESTB", DECISION_SKIPPED, REASON_INSUFFICIENT_CASH),
    ]


# =============================================================================
# Slots and ranking
# =============================================================================


def test_strongest_signal_gets_the_free_slot_and_the_other_is_recorded_as_skipped():
    bars = {symbol: flat_bars(50.0, 4) for symbol in ("TESTA", "TESTB", "TESTC")}
    plans = {
        "TESTA": {D[0]: entry(D[0], stop_distance=5.0, rank=0.10)},
        "TESTB": {D[0]: entry(D[0], stop_distance=5.0, rank=0.30)},
        "TESTC": {D[0]: entry(D[0], stop_distance=5.0, rank=0.20)},
    }

    result, _ = run(bars, plans, max_positions=2)

    outcome = {s.symbol: (s.rank, s.decision, s.reason) for s in result.signals}
    assert outcome == {
        "TESTB": (1, DECISION_TAKEN, None),
        "TESTC": (2, DECISION_TAKEN, None),
        "TESTA": (3, DECISION_SKIPPED, REASON_NO_FREE_SLOT),
    }
    assert max(point.open_positions for point in result.equity) == 2


def test_equal_strength_is_ordered_by_symbol_so_the_result_is_always_the_same():
    bars = {symbol: flat_bars(50.0, 4) for symbol in ("TESTB", "TESTA")}
    plans = {
        "TESTB": {D[0]: entry(D[0], stop_distance=5.0, rank=0.2)},
        "TESTA": {D[0]: entry(D[0], stop_distance=5.0, rank=0.2)},
    }

    result, _ = run(bars, plans, max_positions=1)

    assert {s.symbol: s.decision for s in result.signals} == {
        "TESTA": DECISION_TAKEN,
        "TESTB": DECISION_SKIPPED,
    }


def test_a_slot_freed_by_an_exit_at_the_next_open_can_be_used_at_that_open():
    bars = {symbol: flat_bars(50.0, 6) for symbol in ("TESTA", "TESTB")}
    plans = {
        "TESTA": {D[0]: entry(D[0], stop_distance=5.0), D[2]: exit_(D[2])},
        "TESTB": {D[2]: entry(D[2], stop_distance=5.0)},
    }

    result, _ = run(bars, plans, max_positions=1)

    by_symbol = {trade.symbol: trade for trade in result.trades}
    assert by_symbol["TESTA"].exit_session == D[3]
    assert by_symbol["TESTB"].entry_session == D[3]
    assert max(point.open_positions for point in result.equity) == 1


def test_signal_for_a_symbol_already_held_is_recorded_but_not_traded():
    bars = {"TESTA": flat_bars(50.0, 5)}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0), D[2]: entry(D[2], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    assert [(s.signal_session, s.decision, s.reason) for s in result.signals] == [
        (D[0], DECISION_TAKEN, None),
        (D[2], DECISION_SKIPPED, REASON_ALREADY_IN_POSITION),
    ]
    assert len(result.trades) == 1


def test_entry_is_skipped_when_the_symbol_has_no_bar_on_the_entry_session():
    sessions_without_d1 = [D[0], D[2], D[3]]
    bars = {
        "TESTA": make_bars([(50, 50, 50, 50)] * 3, sessions_without_d1),
        "TESTB": flat_bars(20.0, 4),  # keeps D1 in the calendar of the run
    }
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    assert (result.signals[0].decision, result.signals[0].reason) == (
        DECISION_SKIPPED,
        REASON_NO_BAR,
    )
    assert result.trades == []  # nothing was invented for the missing session


# =============================================================================
# The period
# =============================================================================


def test_open_positions_are_closed_at_the_last_close_and_marked_as_period_end():
    bars = {"TESTA": make_bars([(50, 50, 50, 50), (50, 52, 50, 52), (52, 56, 52, 55)])}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    (trade,) = result.trades
    assert (trade.exit_session, trade.exit_reason) == (D[2], EXIT_PERIOD_END)
    assert trade.exit_reference_price == 55
    assert result.equity[-1].open_positions == 0
    assert result.ending_equity == pytest.approx(EQUITY + 100 * 5)


def test_signal_on_the_last_session_is_recorded_but_cannot_be_traded():
    bars = {"TESTA": flat_bars(50.0, 3)}
    plans = {"TESTA": {D[2]: entry(D[2], stop_distance=5.0)}}

    result, _ = run(bars, plans)

    (signal,) = result.signals
    assert (signal.signal_session, signal.decision, signal.reason) == (
        D[2],
        DECISION_SKIPPED,
        REASON_PERIOD_ENDS,
    )


def test_signals_before_the_period_are_ignored_and_bars_after_it_are_never_read():
    bars = {"TESTA": flat_bars(50.0, 9)}
    plans = {
        "TESTA": {
            D[1]: entry(D[1], stop_distance=5.0),  # before the period
            D[3]: entry(D[3], stop_distance=5.0),  # inside
            D[7]: entry(D[7], stop_distance=5.0),  # after
        }
    }

    result, strategy = run(bars, plans, start=D[2], end=D[5])

    assert [signal.signal_session for signal in result.signals] == [D[3]]
    assert [point.session for point in result.equity] == D[2:6]
    # The strategy was handed history before the period, and not one bar after it.
    assert [bar.session for bar in strategy.seen[0]] == D[:6]


def test_bars_after_the_period_cannot_change_the_result():
    plans = {"TESTA": {D[1]: entry(D[1], stop_distance=5.0), D[3]: exit_(D[3])}}
    calm = make_bars([(50, 51, 49, 50)] * 9)
    wild = calm[:6] + make_bars([(500, 900, 1, 700)] * 3, SESSIONS[6:])

    first, _ = run({"TESTA": calm}, plans, end=D[5])
    second, _ = run({"TESTA": wild}, plans, end=D[5])

    assert first.fingerprint() == second.fingerprint()


def test_invalid_requests_are_refused():
    bars = {"TESTA": flat_bars()}
    with pytest.raises(ValueError, match="period_start"):
        run(bars, {}, start=D[3], end=D[1])
    with pytest.raises(ValueError, match="starting_equity"):
        run(bars, {}, equity=0.0)
    with pytest.raises(ValueError, match="slippage"):
        run(bars, {}, slippage_bps=-1.0)


def test_entry_signal_without_a_stop_distance_is_an_error_not_a_guess():
    bars = {"TESTA": flat_bars(50.0, 3)}
    plans = {"TESTA": {D[0]: entry(D[0], stop_distance=0.0)}}

    with pytest.raises(ValueError, match="stop distance"):
        run(bars, plans)


# =============================================================================
# Bookkeeping
# =============================================================================


def busy_scenario():
    """Several overlapping trades: a winner, a stop-out, a gap stop and a forced close."""
    bars = {
        "TESTA": make_bars(
            [(50, 50, 50, 50), (51, 53, 50.5, 52), (52, 55, 52, 54), (54, 58, 54, 57),
             (60, 61, 59, 60), (60, 60, 60, 60), (60, 60, 60, 60), (60, 60, 60, 60)]
        ),
        "TESTB": make_bars(
            [(20, 20, 20, 20), (20, 21, 19.5, 20), (20, 20, 17.5, 18), (18, 18, 18, 18),
             (18, 19, 18, 19), (19, 19, 19, 19), (15, 15, 14, 14), (14, 14, 14, 14)]
        ),
        "TESTC": make_bars(
            [(80, 80, 80, 80), (80, 80, 80, 80), (80, 80, 80, 80), (80, 82, 79, 81),
             (81, 84, 81, 83), (83, 86, 83, 85), (85, 88, 85, 87), (87, 90, 87, 89)]
        ),
    }  # fmt: skip
    plans = {
        "TESTA": {D[0]: entry(D[0], stop_distance=5.0, rank=3), D[3]: exit_(D[3])},
        "TESTB": {
            D[0]: entry(D[0], stop_distance=2.0, rank=2),  # stopped at 18 on D2
            D[3]: entry(D[3], stop_distance=2.0, rank=1),  # gaps through its stop on D6
        },
        "TESTC": {D[2]: entry(D[2], stop_distance=4.0, rank=1)},  # still open at the end
    }
    return bars, plans


def test_every_trade_and_the_equity_curve_add_up():
    bars, plans = busy_scenario()

    result, _ = run(bars, plans, slippage_bps=5.0, end=D[7])

    assert sorted(trade.exit_reason for trade in result.trades) == sorted(
        [EXIT_SIGNAL, EXIT_STOP, EXIT_STOP_GAP, EXIT_PERIOD_END]
    )
    assert any(trade.net_pnl > 0 for trade in result.trades)
    assert any(trade.net_pnl < 0 for trade in result.trades)  # losing trades are recorded too

    # The account ends at exactly the starting equity plus every trade's net result.
    assert result.ending_equity == pytest.approx(EQUITY + sum(t.net_pnl for t in result.trades))
    for trade in result.trades:
        assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.slippage_cost)
        assert trade.slippage_cost > 0
        assert trade.r_multiple == pytest.approx(trade.net_pnl / trade.risk_amount)
        assert trade.entry_session <= trade.exit_session
    for point in result.equity:
        assert point.equity == pytest.approx(point.cash + point.positions_value)
        assert point.cash >= 0
    assert [point.session for point in result.equity] == D[:8]  # one point per session, in order


def test_every_taken_signal_has_exactly_one_trade_and_skipped_signals_have_none():
    bars, plans = busy_scenario()

    result, _ = run(bars, plans, end=D[7], max_positions=2)

    taken = [i for i, signal in enumerate(result.signals) if signal.decision == DECISION_TAKEN]
    assert sorted(trade.signal_index for trade in result.trades) == taken
    for index, signal in enumerate(result.signals):
        assert (signal.reason is None) == (index in taken)  # every skipped signal says why
    for trade in result.trades:
        signal = result.signals[trade.signal_index]
        assert (signal.symbol, signal.signal_session) < (trade.symbol, trade.entry_session)


def test_higher_assumed_costs_never_improve_a_trade():
    bars, plans = busy_scenario()

    results = [run(bars, plans, slippage_bps=bps, end=D[7])[0] for bps in (0.0, 5.0, 10.0, 20.0)]

    endings = [result.ending_equity for result in results]
    assert endings == sorted(endings, reverse=True)
    assert len(set(endings)) == 4
    assert results[0].trades[0].slippage_cost == 0.0


# =============================================================================
# Reproducibility and independence
# =============================================================================


def test_the_same_input_always_gives_the_same_result():
    bars, plans = busy_scenario()

    first, _ = run(bars, plans, slippage_bps=5.0, end=D[7])
    second, _ = run(bars, plans, slippage_bps=5.0, end=D[7])
    other_costs, _ = run(bars, plans, slippage_bps=10.0, end=D[7])

    assert first.fingerprint() == second.fingerprint()
    assert first.signals == second.signals
    assert first.trades == second.trades
    assert first.equity == second.equity
    assert other_costs.fingerprint() != first.fingerprint()


def test_the_order_of_the_symbols_given_does_not_matter():
    bars, plans = busy_scenario()
    reordered = {symbol: bars[symbol] for symbol in ("TESTC", "TESTA", "TESTB")}

    first, _ = run(bars, plans, end=D[7])
    second, _ = run(reordered, plans, end=D[7])

    assert first.fingerprint() == second.fingerprint()


def test_a_simulation_keeps_no_state_between_runs_or_strategies():
    bars, plans = busy_scenario()
    alone, _ = run(bars, plans, end=D[7])

    # Another "strategy" trading the same symbols in between must not affect it.
    run(bars, {"TESTC": {D[0]: entry(D[0], stop_distance=1.0)}}, end=D[7], equity=5_000.0)
    again, _ = run(bars, plans, end=D[7])

    assert again.fingerprint() == alone.fingerprint()
    assert again.starting_equity == EQUITY


def test_the_inputs_are_not_modified():
    bars, plans = busy_scenario()
    before = {symbol: list(items) for symbol, items in bars.items()}

    run(bars, plans, end=D[7])

    assert bars == before
    assert all(isinstance(bar, DailyBar) for items in bars.values() for bar in items)


def test_an_empty_period_leaves_the_account_untouched():
    result, _ = run({"TESTA": flat_bars()}, {}, start=date(2024, 3, 1), end=date(2024, 3, 29))

    assert (result.signals, result.trades, result.equity) == ([], [], [])
    assert result.ending_equity == EQUITY
