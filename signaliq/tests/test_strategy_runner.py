"""Tests for loading bars into a simulation, recording runs and protecting the out-of-sample period.

SYNTHETIC TEST FIXTURES ONLY: the bars are invented and stored in a throw-away
database. Nothing here is market data and no figure is a performance result.
"""

from dataclasses import replace
from datetime import date

import pytest
from sqlalchemy import func, select, text

from app.database import session_scope
from app.models import (
    Bar,
    StrategyEquityPoint,
    StrategyParameterFreeze,
    StrategyRun,
    StrategySignal,
    StrategyTrade,
)
from app.strategies.assumptions import (
    DAILY_COST_SCENARIOS,
    IN_SAMPLE_END,
    IN_SAMPLE_START,
    OUT_OF_SAMPLE_START,
    STARTING_EQUITY,
    Period,
)
from app.strategies.data import fingerprint_bars, load_daily_bars
from app.strategies.momentum_breakout import MomentumBreakout, MomentumBreakoutParameters
from app.strategies.runner import (
    StrategyRunError,
    build_strategy,
    freeze_parameters,
    get_freeze,
    run_strategy,
    stored_run_rows,
)
from tests.strategy_fixtures import random_walk_bars, sessions_between, store_daily_bars

#: Small windows, so a few months of fixture bars produce signals.
SMALL = MomentumBreakoutParameters(breakout_window=10, exit_window=5, trend_window=20, atr_window=5)

#: History before the in-sample period (indicator warm-up), the in-sample
#: period itself, and the start of the reserved period.
FIXTURE_SESSIONS = sessions_between(date(2021, 3, 1), date(2025, 3, 31))
STRAY = date(2019, 6, 3)  # an isolated early bar, like SPY's bar of 2018-11-01


@pytest.fixture
def stored(settings, session_factory, engine):
    """Fixture bars for TESTA and TESTB, plus one stray early TESTA bar."""
    bars = {
        "TESTA": random_walk_bars(1, FIXTURE_SESSIONS, drift=0.002),
        "TESTB": random_walk_bars(2, FIXTURE_SESSIONS, drift=0.001),
    }
    for symbol, items in bars.items():
        store_daily_bars(session_factory, symbol, items)
    store_daily_bars(session_factory, "TESTA", random_walk_bars(3, [STRAY]))
    return bars


def in_sample():
    return Period("in_sample", IN_SAMPLE_START, IN_SAMPLE_END)


def out_of_sample():
    return Period("out_of_sample", OUT_OF_SAMPLE_START, None)


# =============================================================================
# Loading bars
# =============================================================================


def test_bars_before_the_unbroken_history_are_excluded_but_stay_in_the_database(
    settings, session_factory, stored
):
    with session_scope(session_factory) as session:
        data = load_daily_bars(session, settings, ["TESTA", "TESTB"])
        stored_count = session.execute(
            select(func.count()).select_from(Bar).where(Bar.symbol == "TESTA")
        ).scalar_one()

    assert data.excluded == {"TESTA": [STRAY], "TESTB": []}
    assert data.bars["TESTA"][0].session == FIXTURE_SESSIONS[0]
    assert STRAY not in {bar.session for bar in data.bars["TESTA"]}
    assert stored_count == len(FIXTURE_SESSIONS) + 1  # the stray bar was not deleted


def test_no_bar_after_the_end_date_is_read(settings, session_factory, stored):
    with session_scope(session_factory) as session:
        data = load_daily_bars(session, settings, ["TESTA"], end=IN_SAMPLE_END)

    assert data.last_session <= IN_SAMPLE_END
    assert max(bar.session for bar in data.bars["TESTA"]) == max(
        day for day in FIXTURE_SESSIONS if day <= IN_SAMPLE_END
    )


def test_bars_of_another_feed_or_adjustment_never_mix_in(settings, session_factory, stored):
    # The bar key has no feed column, so the foreign bars go on dates TESTB has no iex bar for.
    april = sessions_between(date(2025, 4, 1), date(2025, 4, 30))
    may = sessions_between(date(2025, 5, 1), date(2025, 5, 30))
    store_daily_bars(session_factory, "TESTB", random_walk_bars(4, april), feed="sip")
    store_daily_bars(session_factory, "TESTB", random_walk_bars(5, may), adjustment="raw")

    with session_scope(session_factory) as session:
        data = load_daily_bars(session, settings, ["TESTB"])

    assert data.last_session == FIXTURE_SESSIONS[-1]
    assert (data.feed, data.adjustment) == ("iex", "split")


def test_the_data_fingerprint_follows_the_bars(settings, session_factory, stored):
    with session_scope(session_factory) as session:
        first = load_daily_bars(session, settings, ["TESTA", "TESTB"])
        again = load_daily_bars(session, settings, ["TESTA", "TESTB"])
        shorter = load_daily_bars(session, settings, ["TESTA", "TESTB"], end=IN_SAMPLE_END)

    assert first.fingerprint == again.fingerprint == fingerprint_bars(first.bars)
    assert shorter.fingerprint != first.fingerprint
    changed = {
        **first.bars,
        "TESTA": [replace(first.bars["TESTA"][0], close=1.0), *first.bars["TESTA"][1:]],
    }
    assert fingerprint_bars(changed) != first.fingerprint


# =============================================================================
# Running and recording
# =============================================================================


def run_small(session_factory, settings, period=None, **kwargs):
    return run_strategy(
        session_factory,
        settings,
        MomentumBreakout(SMALL),
        period or in_sample(),
        symbols=["TESTA", "TESTB"],
        **kwargs,
    )


def test_a_run_records_every_signal_trade_and_equity_point(settings, session_factory, stored):
    summary = run_small(session_factory, settings)

    assert summary.signals_total > summary.signals_taken > 0
    assert summary.trades_total == summary.signals_taken
    assert summary.excluded_sessions == {"TESTA": [STRAY]}
    with session_scope(session_factory) as session:
        run = session.get(StrategyRun, summary.run_id)
        rows = stored_run_rows(session, summary.run_id)
        assert (run.strategy, run.period, run.status) == (
            "momentum_breakout",
            "in_sample",
            "completed",
        )
        assert (run.period_start, run.period_end) == (IN_SAMPLE_START, date(2024, 12, 31))
        assert run.parameters == SMALL.to_dict()
        assert (run.cost_scenario, run.slippage_bps, run.starting_equity) == (
            "base",
            5.0,
            STARTING_EQUITY,
        )
        assert (run.feed, run.adjustment, run.timeframe) == ("iex", "split", "1Day")
        assert run.data_last_session <= IN_SAMPLE_END
        assert run.result_fingerprint == summary.result_fingerprint

        assert len(rows["signals"]) == summary.signals_total
        assert sum(1 for signal in rows["signals"] if signal.decision == "skipped") > 0
        assert all(signal.reason for signal in rows["signals"] if signal.decision == "skipped")
        assert len(rows["trades"]) == summary.trades_total
        assert {trade.signal_id for trade in rows["trades"]} == {
            signal.id for signal in rows["signals"] if signal.decision == "taken"
        }
        assert any(trade.net_pnl < 0 for trade in rows["trades"])  # losers are kept
        assert all(IN_SAMPLE_START <= point.session <= IN_SAMPLE_END for point in rows["equity"])
        assert rows["equity"][-1].equity == pytest.approx(run.ending_equity)
        assert run.ending_equity == pytest.approx(
            STARTING_EQUITY + sum(trade.net_pnl for trade in rows["trades"])
        )


def test_the_same_run_twice_gives_identical_records(settings, session_factory, stored):
    first = run_small(session_factory, settings)
    second = run_small(session_factory, settings)

    assert first.run_id != second.run_id
    assert (first.data_fingerprint, first.result_fingerprint) == (
        second.data_fingerprint,
        second.result_fingerprint,
    )

    def content(run_id):
        with session_scope(session_factory) as session:
            rows = stored_run_rows(session, run_id)
            drop = {"id", "run_id", "signal_id", "_sa_instance_state"}
            return {
                kind: [{k: v for k, v in vars(row).items() if k not in drop} for row in items]
                for kind, items in rows.items()
            }

    assert content(first.run_id) == content(second.run_id)


def test_each_cost_scenario_is_a_separate_run_and_worse_costs_never_help(
    settings, session_factory, stored
):
    summaries = [
        run_small(session_factory, settings, cost_scenario=name) for name in DAILY_COST_SCENARIOS
    ]

    assert [s.slippage_bps for s in summaries] == [5.0, 10.0, 20.0]
    assert len({s.run_id for s in summaries}) == 3
    endings = [s.ending_equity for s in summaries]
    assert endings == sorted(endings, reverse=True)


def test_unknown_cost_scenario_and_symbols_are_refused(settings, session_factory, stored):
    with pytest.raises(StrategyRunError, match="cost scenario"):
        run_small(session_factory, settings, cost_scenario="free")
    with pytest.raises(StrategyRunError, match="universe"):
        run_strategy(
            session_factory, settings, MomentumBreakout(SMALL), in_sample(), symbols=["IBM"]
        )


def test_a_run_without_data_in_the_period_is_refused(settings, session_factory, engine):
    with pytest.raises(StrategyRunError, match="No stored daily bars"):
        run_small(session_factory, settings)


def test_running_a_strategy_never_changes_market_data(settings, session_factory, stored, engine):
    def market_rows():
        with engine.connect() as connection:
            return {
                table: connection.execute(text(f"SELECT * FROM {table} ORDER BY 1, 2")).all()
                for table in ("bars", "ingestion_runs", "data_quality_checks")
            }

    before = market_rows()
    run_small(session_factory, settings)

    assert market_rows() == before


# =============================================================================
# Out-of-sample protection
# =============================================================================


def test_the_reserved_period_cannot_be_run_before_the_parameters_are_frozen(
    settings, session_factory, stored
):
    with pytest.raises(StrategyRunError, match="freeze the parameters"):
        run_small(session_factory, settings, out_of_sample())
    # A custom period that reaches into the reserved dates is refused the same way.
    with pytest.raises(StrategyRunError, match="reserved"):
        run_small(session_factory, settings, Period("custom", date(2024, 6, 1), date(2025, 1, 2)))
    with session_scope(session_factory) as session:
        assert session.execute(select(func.count()).select_from(StrategyRun)).scalar_one() == 0


def test_in_sample_runs_read_no_bar_from_the_reserved_period(settings, session_factory, stored):
    summary = run_small(session_factory, settings)

    with session_scope(session_factory) as session:
        run = session.get(StrategyRun, summary.run_id)
        last_signal = session.execute(select(func.max(StrategySignal.signal_session))).scalar_one()
        last_exit = session.execute(select(func.max(StrategyTrade.exit_session))).scalar_one()
        last_point = session.execute(select(func.max(StrategyEquityPoint.session))).scalar_one()

    assert run.data_last_session < OUT_OF_SAMPLE_START
    assert max(last_signal, last_exit, last_point) < OUT_OF_SAMPLE_START


def test_in_sample_results_do_not_depend_on_reserved_bars(
    settings, session_factory, stored, engine
):
    first = run_small(session_factory, settings)
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE bars SET close = close * 3, high = high * 3 WHERE timestamp >= '2025-01-01'"
            )
        )

    second = run_small(session_factory, settings)

    assert second.result_fingerprint == first.result_fingerprint
    assert second.data_fingerprint == first.data_fingerprint


def test_after_freezing_the_reserved_period_runs_only_with_the_frozen_parameters(
    settings, session_factory, stored
):
    with session_scope(session_factory) as session:
        freeze_parameters(session, MomentumBreakout(SMALL))

    summary = run_small(session_factory, settings, out_of_sample())
    assert summary.period == "out_of_sample"
    assert summary.period_start == OUT_OF_SAMPLE_START
    assert summary.period_end == FIXTURE_SESSIONS[-1]

    tuned = MomentumBreakout(replace(SMALL, breakout_window=12))
    with pytest.raises(StrategyRunError, match="frozen parameters"):
        run_strategy(session_factory, settings, tuned, out_of_sample(), symbols=["TESTA", "TESTB"])


def test_frozen_parameters_cannot_be_replaced_but_freezing_again_is_harmless(
    settings, session_factory, stored
):
    with session_scope(session_factory) as session:
        first = freeze_parameters(session, MomentumBreakout(SMALL))
        frozen_at = first.frozen_at
    with session_scope(session_factory) as session:
        again = freeze_parameters(session, MomentumBreakout(SMALL))
        assert again.frozen_at == frozen_at
    with (
        session_scope(session_factory) as session,
        pytest.raises(StrategyRunError, match="cannot be replaced"),
    ):
        freeze_parameters(session, MomentumBreakout(replace(SMALL, exit_window=7)))
    with session_scope(session_factory) as session:
        assert get_freeze(session, "momentum_breakout").parameters == SMALL.to_dict()
        assert (
            session.execute(select(func.count()).select_from(StrategyParameterFreeze)).scalar_one()
            == 1
        )


def test_a_changed_version_of_the_rules_cannot_use_the_frozen_period(
    settings, session_factory, stored, monkeypatch
):
    with session_scope(session_factory) as session:
        freeze_parameters(session, MomentumBreakout(SMALL))
    monkeypatch.setattr(MomentumBreakout, "version", "1.1")

    with pytest.raises(StrategyRunError, match="different version"):
        run_small(session_factory, settings, out_of_sample())


# =============================================================================
# Building strategies by name
# =============================================================================


def test_strategies_are_built_by_name_only_from_the_known_list():
    assert build_strategy("momentum_breakout").parameters == MomentumBreakoutParameters()
    assert build_strategy("momentum_breakout", {"exit_window": "10"}).parameters.exit_window == 10
    with pytest.raises(StrategyRunError, match="Unknown strategy"):
        build_strategy("vwap_reclaim")  # not implemented yet
    with pytest.raises(StrategyRunError, match="Invalid parameters"):
        build_strategy("momentum_breakout", {"risk_fraction": "2"})


# =============================================================================
# Command line
# =============================================================================


def test_cli_runs_the_in_sample_period_and_prints_a_summary(
    run_script, settings, session_factory, stored
):
    small = settings.model_copy(update={"symbols": ("TESTA", "TESTB")})
    overrides = [f"--param={name}={value}" for name, value in SMALL.to_dict().items()]

    code, out, err = run_script(
        "run_strategy", "momentum_breakout", "--period", "in_sample", *overrides, use_settings=small
    )

    assert code == 0, err
    assert "SIMULATION ONLY" in out
    assert "Signals recorded:" in out
    assert "Excluded for TESTA:  1 stored session(s)" in out
    assert "Result fingerprint:" in out


def test_cli_refuses_the_reserved_period_without_a_freeze(
    run_script, settings, session_factory, stored
):
    small = settings.model_copy(update={"symbols": ("TESTA", "TESTB")})

    code, _, err = run_script(
        "run_strategy", "momentum_breakout", "--period", "out_of_sample", use_settings=small
    )

    assert code == 2
    assert "reserved" in err


def test_cli_freezes_parameters_once(run_script, settings, session_factory, stored):
    code, out, _ = run_script("run_strategy", "momentum_breakout", "--freeze-parameters")
    assert code == 0
    assert "breakout_window = 55" in out

    code, _, err = run_script(
        "run_strategy", "momentum_breakout", "--freeze-parameters", "--param", "breakout_window=40"
    )
    assert code == 2
    assert "cannot be replaced" in err


def test_cli_rejects_bad_input(run_script, stored):
    code, _, err = run_script(
        "run_strategy", "momentum_breakout", "--period", "in_sample", "--param", "nonsense"
    )
    assert code == 2
    code, _, err = run_script(
        "run_strategy", "momentum_breakout", "--period", "in_sample", "--param", "breakout_window=0"
    )
    assert code == 2
    assert "Invalid parameters" in err
