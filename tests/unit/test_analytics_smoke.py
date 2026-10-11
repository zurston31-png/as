"""Smoke checks for analytics/metrics.py and analytics/breakdowns.py.

Not the exhaustive suite (a dedicated pass follows). These cover the
contracts the module exists to keep, in the order they matter:

1. the n=0 case, which is what a backtest run today actually produces;
2. the split-phase refusal, proved by a raise rather than by a docstring;
3. INSUFFICIENT cells, proved by the absence of a number;
4. the arithmetic, against hand-computed values at the measured cost hurdle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flow_model.analytics import breakdowns as B
from flow_model.analytics import metrics as M
from flow_model.core.contracts import CostBreakdown, EquityPoint
from flow_model.core.enums import ExitReason, Regime, SplitPhase

T0 = datetime(2020, 6, 1, 14, 0, tzinfo=timezone.utc)

#: Phase 7a measured one NQ round turn, size 2, 10-point stop (R = $400):
#: $21.49 through the target (a resting limit) and $47.74 through the stop (a
#: market order paying stop_order_extra_slippage_ticks). `make_trade`'s
#: defaults give the same R = $400 (20 points x $20 x 1), so these dollar
#: figures carry the same R drag here: 0.0537R and 0.1194R.
TARGET_COST = 21.49
STOP_COST = 47.74


def flat_equity(n: int = 2500, equity: float = 100_000.0) -> list[EquityPoint]:
    return [
        EquityPoint(
            ts=T0 + timedelta(minutes=5 * i),
            equity=equity,
            realized_equity=equity,
            high_water_mark=equity,
        )
        for i in range(n)
    ]


def curve(values: list[float], minutes: int = 1440) -> list[EquityPoint]:
    return [
        EquityPoint(ts=T0 + timedelta(minutes=minutes * i), equity=v, realized_equity=v)
        for i, v in enumerate(values)
    ]


# --- 1. the empty trade set, which is the live case -----------------------


def test_the_zero_trade_case_completes_and_reports_undefined_not_zero():
    """The system takes no trades today, so this is the first thing that runs.

    Nothing crashes, nothing divides by zero, and no rate comes back as 0.0.
    """
    summary = M.performance_summary([], flat_equity(), split_label=SplitPhase.TEST)

    assert summary.n_trades == 0
    assert summary.sample is M.SampleStatus.EMPTY

    for name in ("win_rate", "loss_rate", "expectancy", "mean_r", "median_r",
                 "profit_factor", "average_win_r", "average_loss_r",
                 "mean_cost_drag_r", "mean_holding_seconds"):
        value = summary.trade_metrics[name]
        assert not value.is_defined, f"{name} must not be a number at n=0"
        assert value.undefined is M.Undefined.NO_TRADES
        assert value.n == 0

    # Counts and sums over an empty set are their identity element, which is a
    # measurement: no trades were taken and no money moved.
    assert summary.trade_metrics["trade_count"].require() == 0.0
    assert summary.trade_metrics["total_pnl"].require() == 0.0
    assert summary.trade_metrics["total_costs"].require() == 0.0

    # A flat curve HAS a drawdown measurement (zero depth, observed), but no
    # dispersion to form a ratio from.
    assert summary.drawdown.max_drawdown_pct.require() == 0.0
    assert summary.sharpe.undefined is M.Undefined.ZERO_DISPERSION
    assert summary.sortino.undefined is M.Undefined.ZERO_DISPERSION
    assert summary.total_return.require() == 0.0
    assert summary.trades_per_year.require() == 0.0  # zero trades per year, measured
    assert any("n=0" in note for note in summary.notes)


def test_every_registered_metric_survives_an_empty_trade_set():
    for name, function in M.TRADE_METRICS.items():
        value = function([], split_label=None)
        assert isinstance(value, M.MetricValue), name
        assert value.n == 0, name
        if value.is_defined:
            # Only counts and sums may be defined here.
            assert value.require() == 0.0, name


def test_an_empty_breakdown_keeps_its_shape_over_every_dimension():
    """A dimension with a declared domain still emits its cells, so a report
    over zero trades needs no special case anywhere downstream."""
    tables = B.breakdown_all([], split_label=None)
    assert set(tables) == {d.name for d in B.default_dimensions()}
    assert len(tables["regime"].cells) == len(tuple(Regime))
    assert len(tables["day_of_week"].cells) == 7
    assert len(tables["exit_reason"].cells) == len(tuple(ExitReason))
    for table in tables.values():
        assert table.n_trades == 0
        for cell in table.cells:
            assert cell.status is M.SampleStatus.EMPTY
            assert all(not v.is_defined for v in cell.metrics.values())


# --- 2. the split-phase refusal, structural -------------------------------


def test_mixing_split_phases_raises_rather_than_merging(make_trade):
    mixed = [
        make_trade(trade_id="a", split_label=SplitPhase.TRAIN),
        make_trade(trade_id="b", split_label=SplitPhase.TEST),
    ]
    with pytest.raises(M.MixedSplitError):
        M.win_rate(mixed, split_label=SplitPhase.TRAIN)
    with pytest.raises(M.MixedSplitError):
        M.performance_summary(mixed, split_label=SplitPhase.TRAIN)
    with pytest.raises(M.MixedSplitError):
        B.breakdown(mixed, "regime", split_label=SplitPhase.TRAIN)


def test_every_registered_metric_carries_the_refusal(make_trade):
    """The guard is a property of the registry, not of the metrics that
    happened to be written carefully."""
    mixed = [
        make_trade(trade_id="a", split_label=SplitPhase.TRAIN),
        make_trade(trade_id="b", split_label=SplitPhase.VALIDATION),
    ]
    for name, function in M.TRADE_METRICS.items():
        with pytest.raises(M.MixedSplitError):
            function(mixed, split_label=SplitPhase.TRAIN)


def test_a_mislabelled_single_phase_set_also_raises(make_trade):
    trades = [make_trade(split_label=SplitPhase.TRAIN)]
    with pytest.raises(M.SplitLabelMismatchError):
        M.expectancy_r(trades, split_label=SplitPhase.TEST)


def test_the_cross_phase_view_separates_and_never_combines(make_trade):
    mixed = [
        make_trade(trade_id=f"tr{i}", split_label=SplitPhase.TRAIN) for i in range(3)
    ] + [make_trade(trade_id="te", split_label=SplitPhase.TEST)]
    table = B.breakdown_by_split_phase(mixed, include_empty=False)
    assert table.spans_phases is True
    assert [c.key for c in table.cells] == ["TRAIN", "TEST"]
    assert [c.n for c in table.cells] == [3, 1]
    # No combined row anywhere.
    assert all(c.key in {"TRAIN", "TEST"} for c in table.cells)


# --- 3. small samples are refused, not displayed --------------------------


def test_a_three_trade_cell_is_insufficient_and_holds_no_number(make_trade):
    trades = [make_trade(trade_id=f"t{i}", regime=Regime.CHOP) for i in range(3)]
    table = B.breakdown(trades, "regime", split_label=SplitPhase.TRAIN)
    cell = table.cell("CHOP")

    assert cell.n == 3 and cell.n_wins == 3          # the counts are data
    assert cell.status is M.SampleStatus.INSUFFICIENT
    assert cell.is_reportable is False
    for value in cell.metrics.values():              # the rates are not
        assert not value.is_defined
        assert value.undefined is M.Undefined.INSUFFICIENT_SAMPLE
    assert cell.metrics["win_rate"].n == 3           # N travels with the refusal


def test_a_cell_at_min_sample_n_is_reported(make_trade, config):
    n = config.analytics.min_sample_n
    trades = [make_trade(trade_id=f"t{i}", regime=Regime.CHOP) for i in range(n)]
    cell = B.breakdown(trades, "regime", split_label=SplitPhase.TRAIN).cell("CHOP")
    assert cell.status is M.SampleStatus.ADEQUATE
    assert cell.metrics["win_rate"].require() == pytest.approx(1.0)


def test_a_dimension_may_not_drop_a_trade(make_trade):
    trades = [make_trade(trade_id=f"t{i}") for i in range(4)]
    for dimension in B.default_dimensions():
        table = B.breakdown(trades, dimension, split_label=SplitPhase.TRAIN)
        assert table.n_trades == 4, dimension.name


# --- 4. the arithmetic, at the measured cost hurdle -----------------------


@pytest.fixture
def book(make_trade):
    """Three +1R targets and two -1R stops, each paying its measured cost."""
    wins = [
        make_trade(
            trade_id=f"w{i}",
            costs=CostBreakdown(commission=TARGET_COST),
            exit_reason=ExitReason.TARGET,
            signal_ts=T0 + timedelta(minutes=10 * i),
        )
        for i in range(3)
    ]
    losses = [
        make_trade(
            trade_id=f"l{i}",
            exit_price=17980.0,
            costs=CostBreakdown(commission=STOP_COST),
            exit_reason=ExitReason.STOP,
            signal_ts=T0 + timedelta(minutes=100 + 10 * i),
        )
        for i in range(2)
    ]
    return wins + losses


def test_metrics_read_r_from_the_record_rather_than_re_deriving_it(book):
    """Section 0.2: R is defined once. A second derivation would drift."""
    expected = sum(t.r_multiple for t in book) / len(book)
    assert M.mean_r(book, split_label=SplitPhase.TRAIN).require() == pytest.approx(expected)
    assert M.expectancy_r(book, split_label=SplitPhase.TRAIN).require() == pytest.approx(
        expected
    )


def test_expectancy_and_profit_factor_are_net_of_costs(book):
    # +1R target less $21.49 on a $400 R; -1R stop less $47.74.
    win_r = (400.0 - TARGET_COST) / 400.0
    loss_r = -(400.0 + STOP_COST) / 400.0
    assert M.average_win_r(book, split_label=SplitPhase.TRAIN).require() == pytest.approx(
        win_r
    )
    assert M.average_loss_r(book, split_label=SplitPhase.TRAIN).require() == pytest.approx(
        loss_r
    )
    assert M.expectancy_r(book, split_label=SplitPhase.TRAIN).require() == pytest.approx(
        (3 * win_r + 2 * loss_r) / 5
    )
    assert M.profit_factor(book, split_label=SplitPhase.TRAIN).require() == pytest.approx(
        (3 * win_r) / abs(2 * loss_r)
    )
    assert M.win_rate(book, split_label=SplitPhase.TRAIN).require() == pytest.approx(0.6)


def test_breakeven_win_rate_is_derived_and_lands_on_the_measured_hurdle(book):
    """The brief's figure: breakeven for a symmetric 1R system at these costs
    is 54.19%, not 50%. It is COMPUTED from this sample's average win and
    average loss, never quoted from a constant."""
    measured = M.breakeven_win_rate(book, split_label=SplitPhase.TRAIN).require()
    assert measured == pytest.approx(0.5419, abs=5e-5)
    assert measured > 0.5


def test_cost_drag_is_broken_down_by_exit_reason(book):
    report = M.cost_drag_by_exit_reason(book, split_label=SplitPhase.TRAIN)
    # Exact, from the record's own costs and risk; and within rounding of the
    # 0.0537R / 0.1194R the brief quotes from the Phase 7a measurement.
    assert report.by_reason_r[ExitReason.TARGET].require() == pytest.approx(
        TARGET_COST / 400.0
    )
    assert report.by_reason_r[ExitReason.STOP].require() == pytest.approx(
        STOP_COST / 400.0
    )
    assert report.by_reason_r[ExitReason.TARGET].require() == pytest.approx(0.0537, abs=1e-4)
    assert report.by_reason_r[ExitReason.STOP].require() == pytest.approx(0.1194, abs=1e-4)
    # A stop costs more than twice a target, so one figure describes neither.
    assert report.by_reason_r[ExitReason.STOP].require() > (
        2 * report.by_reason_r[ExitReason.TARGET].require()
    )
    assert report.exit_counts == {ExitReason.STOP: 2, ExitReason.TARGET: 3}
    assert report.mix_weighted_r.is_defined
    assert "exit_counts" in report.mix_weighted_r.note


def test_profit_factor_with_no_losses_is_undefined_not_infinite(make_trade):
    wins = [make_trade(trade_id=f"w{i}") for i in range(3)]
    value = M.profit_factor(wins, split_label=SplitPhase.TRAIN)
    assert not value.is_defined
    assert value.undefined is M.Undefined.NO_LOSSES
    with pytest.raises(M.UndefinedMetricError):
        value.require()


def test_profit_factor_with_no_wins_is_zero_which_is_a_measurement(make_trade):
    losses = [
        make_trade(trade_id=f"l{i}", exit_price=17980.0, exit_reason=ExitReason.STOP)
        for i in range(2)
    ]
    assert M.profit_factor(losses, split_label=SplitPhase.TRAIN).require() == 0.0


def test_streaks_run_in_exit_order(make_trade):
    pattern = [True, False, False, True, True, True]
    trades = [
        make_trade(
            trade_id=f"t{i}",
            exit_price=None if win else 17980.0,
            exit_reason=ExitReason.TARGET if win else ExitReason.STOP,
            signal_ts=T0 + timedelta(minutes=60 * i),
        )
        for i, win in enumerate(pattern)
    ]
    profile = M.streaks(list(reversed(trades)), split_label=SplitPhase.TRAIN)
    assert profile.max_consecutive_wins.require() == 3
    assert profile.max_consecutive_losses.require() == 2
    assert profile.current_streak.require() == 3


# --- the equity series ----------------------------------------------------


def test_sharpe_below_the_declared_minimum_is_undefined():
    value = M.sharpe_ratio(curve([100.0, 101.0, 102.0]), split_label=None)
    assert value.undefined is M.Undefined.INSUFFICIENT_OBSERVATIONS
    assert str(M.MIN_RETURN_OBSERVATIONS) in value.note


def test_drawdown_depth_recovery_and_what_is_still_open():
    points = curve([100.0, 120.0, 90.0, 120.0, 110.0])
    profile = M.drawdown_profile(points, split_label=None)
    assert profile.n_episodes == 2
    assert profile.n_unrecovered == 1
    assert profile.max_drawdown_pct.require() == pytest.approx(-0.25)
    assert profile.max_drawdown_dollars.require() == pytest.approx(-30.0)
    assert profile.average_drawdown_pct.require() == pytest.approx(
        (-30.0 / 120.0 + -10.0 / 120.0) / 2
    )
    assert profile.longest_recovery_seconds.require() == pytest.approx(2 * 86400.0)
    assert profile.seconds_underwater_at_end.require() == pytest.approx(86400.0)


def test_an_unrecovered_drawdown_has_no_recovery_time():
    profile = M.drawdown_profile(curve([100.0, 90.0, 95.0]), split_label=None)
    assert profile.longest_recovery_seconds.undefined is M.Undefined.NOT_RECOVERED
    assert profile.n_unrecovered == 1


def test_a_backwards_equity_series_is_rejected_rather_than_sorted():
    points = curve([100.0, 101.0])
    with pytest.raises(M.NonChronologicalEquityError):
        M.total_return(list(reversed(points)), split_label=None)


def test_a_ruined_account_has_no_return_ratio():
    assert M.total_return(curve([0.0, 50.0]), split_label=None).undefined is (
        M.Undefined.NON_POSITIVE_EQUITY
    )
    assert M.cagr(curve([100.0, 0.0]), split_label=None).undefined is (
        M.Undefined.NON_POSITIVE_EQUITY
    )


# --- the undefined representation itself ----------------------------------


def test_a_metric_value_holds_either_a_number_or_a_reason_never_both():
    with pytest.raises(ValueError):
        M.MetricValue(name="x", value=1.0, undefined=M.Undefined.NO_TRADES)
    with pytest.raises(ValueError):
        M.MetricValue(name="x")
    with pytest.raises(ValueError):
        M.MetricValue(name="x", value=float("inf"))


def test_metric_names_match_their_registry_keys(book):
    """A typo between a function's reported name and its registry key would
    mislabel a column in every report."""
    for name, function in M.TRADE_METRICS.items():
        value = function(book, split_label=SplitPhase.TRAIN)
        assert value.name in {name, f"{name}_r"}, (
            f"{name} reports itself as {value.name}"
        )


def test_unknown_metric_names_are_rejected(book):
    with pytest.raises(KeyError):
        M.compute_trade_metrics(book, split_label=SplitPhase.TRAIN, names=["sharpe"])
    with pytest.raises(KeyError):
        B.breakdown(book, "regime", split_label=SplitPhase.TRAIN, metrics=["nope"])
    with pytest.raises(KeyError):
        B.dimension_by_name("moon_phase")


def test_the_policy_comes_from_config_and_nothing_else(config):
    policy = M.MetricPolicy.from_config(config)
    assert policy.min_sample_n == config.analytics.min_sample_n
    assert policy.flow_score_buckets == config.analytics.flow_score_buckets
    assert policy.min_return_observations == M.MIN_RETURN_OBSERVATIONS
    assert policy == M.MetricPolicy.default()


def test_results_are_deterministic(book):
    first = M.performance_summary(book, flat_equity(100), split_label=SplitPhase.TRAIN)
    second = M.performance_summary(
        list(reversed(book)), flat_equity(100), split_label=SplitPhase.TRAIN
    )
    assert first.to_init_dict() == second.to_init_dict()


# --- targets are reported, never optimized toward -------------------------


def test_a_target_comparison_reports_the_measurement_first(book, config):
    summary = M.performance_summary(book, split_label=SplitPhase.TRAIN)
    comparisons = M.compare_to_research_targets(summary, config.research_targets)
    combined = comparisons[0]
    assert combined.measured.name == "win_rate"
    assert combined.measured.require() == pytest.approx(0.6)
    assert combined.target == config.research_targets.combined_10y_win_rate
    assert combined.met is False
    assert "never selects a parameter" in combined.note


def test_an_undefined_measurement_meets_nothing_and_fails_nothing(config):
    summary = M.performance_summary([], split_label=SplitPhase.TEST)
    for comparison in M.compare_to_research_targets(summary, config.research_targets):
        assert comparison.met is None


# --- bucket labels must agree with the declared domain --------------------


@pytest.mark.parametrize(
    "score", [0.0, 59.9, 60.0, 65.23, 69.999, 70.0, 84.9, 85.0, 99.9, 100.0]
)
def test_every_in_range_flow_score_lands_in_a_declared_bucket(score, config):
    """Caught a real bug: the top bucket is closed `[85,100]` in the domain, so
    a score of 85 labelled `[85,100)` produced TWO cells for one bucket -- one
    declared-but-empty and one observed."""
    edges = config.analytics.flow_score_buckets
    assert B.flow_score_bucket_label(score, edges) in B.flow_score_bucket_domain(edges)


def test_a_flow_score_outside_the_configured_buckets_is_visible(config):
    edges = config.analytics.flow_score_buckets
    assert B.flow_score_bucket_label(-1.0, edges) not in B.flow_score_bucket_domain(edges)
    assert B.flow_score_bucket_label(101.0, edges) not in B.flow_score_bucket_domain(edges)


def test_the_flow_score_dimension_emits_one_cell_per_bucket(make_trade, config):
    trades = [
        make_trade(trade_id=f"s{i}", flow_score=score)
        for i, score in enumerate([85.0, 90.0, 100.0, 70.0])
    ]
    table = B.breakdown(trades, "flow_score_bucket", split_label=SplitPhase.TRAIN)
    keys = [c.key for c in table.cells]
    assert len(keys) == len(set(keys))
    assert len(keys) == len(B.flow_score_bucket_domain(config.analytics.flow_score_buckets))
    assert table.cell("[85,100]").n == 3
