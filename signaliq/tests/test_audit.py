"""Audit tests. SYNTHETIC TEST FIXTURES ONLY: invented trades and invented bars.
No number here is a performance result."""

from datetime import date

import pytest
from sqlalchemy import delete, func, select, text

from app.database import session_scope
from app.models import StrategyRun, StrategyTrade
from app.strategies.assumptions import IN_SAMPLE_END, IN_SAMPLE_START, Period
from app.strategies.audit import (
    EquityView,
    TradeView,
    audit_run,
    buy_and_hold,
    concentration,
    drawdown,
    latest_runs,
    symbol_breakdown,
    trade_overlap,
    trade_statistics,
)
from app.strategies.data import DailyBar, DailyData
from app.strategies.momentum_breakout import MomentumBreakout, MomentumBreakoutParameters
from app.strategies.runner import run_strategy
from tests.strategy_fixtures import random_walk_bars, sessions_between, store_daily_bars

D = [date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9, 10, 11, 12)]


def trade(trade_id, symbol, net, *, r=None, held=3, reason="exit_signal", entry=D[0], exit_=D[3]):
    return TradeView(trade_id, symbol, entry, exit_, 10, 100.0, 100.0, reason, held,
                     net + 1.0, 1.0, net, 100.0, net / 100.0 if r is None else r)  # fmt: skip


# =============================================================================
# Trade statistics, by hand
# =============================================================================


def test_trade_statistics_by_hand():
    trades = [
        trade(1, "TESTA", 300.0, held=10),
        trade(2, "TESTA", -100.0, held=2, reason="stop"),
        trade(3, "TESTB", 100.0, held=6),
        trade(4, "TESTB", -50.0, held=1, reason="stop"),
        trade(5, "TESTC", 0.0, held=3),
    ]

    s = trade_statistics(trades)

    assert (s.trades, s.wins, s.losses, s.breakeven) == (5, 2, 2, 1)
    assert s.win_rate == pytest.approx(0.4)
    assert (s.gross_profit, s.gross_loss, s.net_profit) == (400.0, -150.0, 250.0)
    assert s.profit_factor == pytest.approx(400 / 150)
    assert (s.average_win, s.average_loss) == (200.0, -75.0)
    assert s.payoff_ratio == pytest.approx(200 / 75)
    assert s.expectancy == pytest.approx(50.0)
    assert s.expectancy_r == pytest.approx(0.5)
    assert s.total_slippage == 5.0
    assert s.average_sessions_held == pytest.approx(4.4)
    assert s.median_sessions_held == 3.0
    assert (s.average_sessions_held_wins, s.average_sessions_held_losses) == (8.0, 1.5)
    assert [t.trade_id for t in s.largest_wins] == [1, 3]
    assert [t.trade_id for t in s.largest_losses] == [2, 4]
    assert s.exits_by_reason == {"exit_signal": 3, "stop": 2}


def test_profit_factor_is_undefined_without_losses_and_stats_without_trades():
    assert trade_statistics([trade(1, "TESTA", 10.0)]).profit_factor is None
    empty = trade_statistics([])
    assert (empty.trades, empty.win_rate, empty.expectancy, empty.profit_factor) == (
        0,
        None,
        None,
        None,
    )


def test_longest_losing_streak_follows_the_exit_order():
    trades = [
        trade(1, "TESTA", -1.0, exit_=D[1]),
        trade(2, "TESTA", -1.0, exit_=D[2]),
        trade(3, "TESTA", 5.0, exit_=D[3]),
        trade(4, "TESTA", -1.0, exit_=D[4]),
        trade(5, "TESTA", -1.0, exit_=D[5]),
        trade(6, "TESTA", -1.0, exit_=D[6]),
    ]
    assert trade_statistics(trades).max_consecutive_losses == 3


# =============================================================================
# Drawdown and recovery, by hand
# =============================================================================


def curve(values):
    return [EquityView(D[i], value, 1) for i, value in enumerate(values)]


def test_drawdown_peak_trough_and_recovery():
    # Start 100; high 120 on D1; low 90 on D3 (-25 %); back to 120 on D6.
    d = drawdown(100.0, curve([110, 120, 100, 90, 105, 115, 121, 118]))

    assert d.max_drawdown == pytest.approx(0.25)
    assert d.max_drawdown_amount == pytest.approx(30.0)
    assert (d.peak_session, d.trough_session, d.recovery_session) == (D[1], D[3], D[6])
    assert (d.sessions_peak_to_trough, d.sessions_trough_to_recovery, d.sessions_underwater) == (
        2,
        3,
        5,
    )
    assert d.recovered


def test_a_drawdown_from_the_starting_equity_and_without_recovery():
    d = drawdown(100.0, curve([95, 80, 85, 90]))

    assert d.max_drawdown == pytest.approx(0.20)
    assert d.peak_session is None  # the peak was the starting equity
    assert d.trough_session == D[1]
    assert not d.recovered
    assert d.recovery_session is None
    assert d.sessions_underwater == 4  # from before the first session to the last one


def test_no_drawdown_when_equity_only_rises():
    d = drawdown(100.0, curve([100, 101, 105]))
    assert (d.max_drawdown, d.trough_session, d.recovered) == (0.0, None, True)


# =============================================================================
# Per symbol and concentration
# =============================================================================


def test_symbol_breakdown_and_shares():
    trades = [trade(1, "TESTA", 600.0), trade(2, "TESTB", 500.0), trade(3, "TESTB", -100.0)]

    rows = symbol_breakdown(trades, 100_000.0, ["TESTA", "TESTB", "TESTC"])

    by = {row.symbol: row for row in rows}
    assert [row.symbol for row in rows] == ["TESTA", "TESTB", "TESTC"]
    assert by["TESTA"].strategy_return == pytest.approx(0.006)
    assert by["TESTA"].share_of_net_profit == pytest.approx(0.6)
    assert by["TESTB"].share_of_net_profit == pytest.approx(0.4)
    assert by["TESTB"].share_of_gross_profit == pytest.approx(500 / 1100)
    assert (by["TESTC"].trades, by["TESTC"].net_profit) == (0, 0.0)  # listed even without trades


def test_shares_of_net_profit_are_not_given_when_the_total_is_a_loss():
    rows = symbol_breakdown([trade(1, "TESTA", 100.0), trade(2, "TESTB", -300.0)], 1000.0, [])
    assert all(row.share_of_net_profit is None for row in rows)


def test_heavy_dependence_is_flagged_when_one_symbol_carries_the_result():
    trades = [trade(1, "TESTA", 1000.0), trade(2, "TESTB", -300.0), trade(3, "TESTC", -200.0)]

    c = concentration(trades, symbol_breakdown(trades, 1.0, []))

    assert c.heavily_dependent is True
    assert c.best_symbol == "TESTA"
    assert c.best_symbol_share_of_net_profit == pytest.approx(2.0)
    assert c.net_profit_without_best_symbol == pytest.approx(-500.0)
    assert "without TESTA" in c.reason


def test_a_spread_out_result_is_not_flagged():
    trades = [
        trade(i, symbol, 100.0) for i, symbol in enumerate(["TESTA", "TESTB", "TESTC", "TESTD"] * 2)
    ]

    c = concentration(trades, symbol_breakdown(trades, 1.0, []))

    assert c.heavily_dependent is False
    assert c.best_symbol_share_of_net_profit == pytest.approx(0.25)
    assert c.top_five_trades_share_of_net_profit == pytest.approx(5 / 8)


def test_dependence_is_not_judged_for_a_losing_run():
    trades = [trade(1, "TESTA", -10.0)]
    assert concentration(trades, symbol_breakdown(trades, 1.0, [])).heavily_dependent is None


# =============================================================================
# Buy-and-hold
# =============================================================================


def test_buy_and_hold_uses_the_first_open_and_the_last_close_of_the_period():
    bars = [
        DailyBar(D[0], 9.0, 9, 9, 9.5, 1),  # before the period
        DailyBar(D[1], 10.0, 12, 9, 11.0, 1),
        DailyBar(D[2], 11.0, 13, 8, 8.8, 1),
        DailyBar(D[3], 9.0, 15, 9, 15.0, 1),
        DailyBar(D[4], 15.0, 16, 14, 16.0, 1),  # after the period
    ]
    data = DailyData({"TESTA": bars}, {}, "alpaca", "iex", "split", "x")

    (row,) = buy_and_hold(data, D[1], D[3], 10.0)

    assert (row.entry_price, row.exit_price) == (10.0, 15.0)
    assert row.total_return == pytest.approx(0.5)
    assert row.total_return_after_costs == pytest.approx(15 * 0.999 / (10 * 1.001) - 1)
    assert row.max_drawdown == pytest.approx(1 - 8.8 / 11.0)


# =============================================================================
# Against stored runs (synthetic bars)
# =============================================================================

SMALL = MomentumBreakoutParameters(breakout_window=10, exit_window=5, trend_window=20, atr_window=5)


@pytest.fixture
def three_runs(settings, session_factory, engine):
    sessions = sessions_between(date(2021, 3, 1), date(2024, 12, 31))
    for seed, symbol in enumerate(("TESTA", "TESTB"), start=1):
        store_daily_bars(session_factory, symbol, random_walk_bars(seed, sessions, drift=0.0015))
    period = Period("in_sample", IN_SAMPLE_START, IN_SAMPLE_END)
    return [
        run_strategy(
            session_factory,
            settings,
            MomentumBreakout(SMALL),
            period,
            cost_scenario=name,
            symbols=["TESTA", "TESTB"],
        ).run_id
        for name in ("base", "worse", "severe")
    ]


def test_audit_of_a_stored_run_includes_every_trade_and_passes_integrity(
    session_factory, three_runs
):
    with session_scope(session_factory) as session:
        audit = audit_run(session, three_runs[0])
        stored = session.execute(
            select(func.count())
            .select_from(StrategyTrade)
            .where(StrategyTrade.run_id == three_runs[0])
        ).scalar_one()
        losers = session.execute(
            select(func.count())
            .select_from(StrategyTrade)
            .where(StrategyTrade.run_id == three_runs[0], StrategyTrade.net_pnl < 0)
        ).scalar_one()

    assert audit.integrity.ok, audit.integrity.problems
    assert audit.statistics.trades == len(audit.trades) == stored > 0
    assert audit.statistics.losses == audit.integrity.losing_trades_stored == losers > 0
    assert audit.statistics.net_profit == pytest.approx(audit.ending_equity - audit.starting_equity)
    assert sum(row.trades for row in audit.symbols) == stored
    assert sum(row.net_profit for row in audit.symbols) == pytest.approx(
        audit.statistics.net_profit
    )


def test_a_deleted_losing_trade_is_detected(session_factory, three_runs, engine):
    with session_scope(session_factory) as session:
        loser = (
            session.execute(
                select(StrategyTrade.id).where(
                    StrategyTrade.run_id == three_runs[0], StrategyTrade.net_pnl < 0
                )
            )
            .scalars()
            .first()
        )
        session.execute(delete(StrategyTrade).where(StrategyTrade.id == loser))

    with session_scope(session_factory) as session:
        audit = audit_run(session, three_runs[0])

    assert not audit.integrity.ok
    assert not audit.integrity.every_taken_signal_has_one_trade
    assert not audit.integrity.ending_equity_matches_trades
    assert any("run header" in problem for problem in audit.integrity.problems)


def test_latest_runs_returns_one_run_per_cost_scenario_and_overlap_is_reported(
    session_factory, three_runs
):
    with session_scope(session_factory) as session:
        runs = latest_runs(session, "momentum_breakout", "in_sample")
        audits = [audit_run(session, run.id) for run in runs]

    assert [run.cost_scenario for run in runs] == ["base", "worse", "severe"]
    assert [run.id for run in runs] == three_runs
    overlap = trade_overlap(audits)
    assert overlap["common_to_all"] <= min(overlap["per_run"])
    endings = [audit.ending_equity for audit in audits]
    assert endings == sorted(endings, reverse=True)


def test_the_audit_changes_nothing(session_factory, three_runs, engine):
    def everything():
        with engine.connect() as connection:
            return {
                table: connection.execute(text(f"SELECT * FROM {table} ORDER BY 1")).all()
                for table in (
                    "bars",
                    "strategy_runs",
                    "strategy_signals",
                    "strategy_trades",
                    "strategy_equity",
                )
            }

    before = everything()
    with session_scope(session_factory) as session:
        audit_run(session, three_runs[0])
        assert session.execute(select(func.count()).select_from(StrategyRun)).scalar_one() == 3

    assert everything() == before


def test_cli_prints_the_audit_and_checks_the_trade_count(run_script, settings, three_runs):
    small = settings.model_copy(update={"symbols": ("TESTA", "TESTB")})

    code, out, err = run_script("audit_strategy", "momentum_breakout", use_settings=small)
    assert code == 0, err
    assert "READ-ONLY AUDIT" in out
    assert "COST SCENARIOS COMPARED" in out
    assert "Result: PASSED" in out
    assert out.count("\nRUN ") == 3

    code, _, err = run_script("audit_strategy", "momentum_breakout", "--expect-trades", "99999",
                              use_settings=small)  # fmt: skip
    assert code == 1
    assert "TRADE COUNT CHECK FAILED" in err


def test_cli_refuses_when_there_is_nothing_to_audit(run_script, engine):
    code, _, err = run_script("audit_strategy", "momentum_breakout")
    assert code == 2
    assert "no stored in_sample run" in err
    assert "Nothing was simulated" in err
