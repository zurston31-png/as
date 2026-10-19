"""Adversarial tests for validation/consistency.py.

This module's reason to exist is one ranking: "72% stable" must score above "90%
in one window and badly in the rest". So the tests are organized around the ways
that ranking, or the honesty of the numbers it rests on, could silently fail.

**The ranking itself.** `test_the_whole_point_*` score the two axes of section
10's sentence with the real formula at several window counts, and check the
preference against the closed-form boundary `beta(k)` from both sides -- just
below it the stable axis must win, just above it the spiky one must, because a
formula that always prefers the uniform axis would be ignoring the level and
would rank a flat 10% above everything.

**A dilution of the catastrophe.** The dispersion measure must not shrink as
windows are added. One test computes the sample standard deviation and the
worst-cell shortfall over the same one-catastrophic-window shape at k = 2, 4, 25
and pins the directions: `v / sqrt(k)` falls, `v (k-1) / k` rises.

**A zero that reads as a measurement.** The committed system takes no trades
(PROVENANCE.md 2.4), so the n = 0 report is the live case: the score, its lower
bound and every statement must be an explicit refusal, and `describe()` -- what a
reader actually sees -- must contain no rate. Dispersion over a SINGLE window is
the same hazard one step up: 0.0 there would read as "every window agreed", so it
is undefined while `level` and `worst` stay measured.

**A rate projected from three trades.** A cell below `min_sample_n` must
contribute nothing AND hold no number, so the suppression cannot be undone by
reading the object.

**A composite that flatters itself.** With some terms undefined the score must
refuse rather than renormalize over the defined ones, and the lower bound must
stay a bound (it cannot exceed any completion).

Every config object is built LOCALLY (`AnalyticsConfig(min_sample_n=2)` so a
two-trade cell sits exactly on the boundary; the shipped default is 30).
Arithmetic: NQ at $20/point, size 1, a 20-point stop, so R = $400 and a trade
exiting at its 1R target is +1R, one exiting at its stop is -1R.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from flow_model.analytics.breakdowns import breakdown
from flow_model.analytics.metrics import (
    MetricPolicy,
    MetricValue,
    MixedSplitError,
    SampleStatus,
    Undefined,
)
from flow_model.config.schema import AnalyticsConfig, ConsistencyConfig
from flow_model.core.enums import Regime, SplitPhase
from flow_model.validation.consistency import (
    CONSISTENCY_REGIMES,
    MIN_CELLS_FOR_DISPERSION,
    STABILITY_METRICS,
    CellMeasurement,
    ConsistencyAxis,
    ConsistencyError,
    ConsistencyReport,
    StabilityMetric,
    StabilityStatement,
    cells_from_breakdown,
    consistency_from_trades,
    consistency_report,
    preference_check,
    quality,
    regime_dimension,
    regime_year_dimension,
    rest_quality_boundary,
    year_dimension,
)

WIN_RATE = StabilityMetric.WIN_RATE
TRAIN = SplitPhase.TRAIN


@pytest.fixture
def policy() -> MetricPolicy:
    """min_sample_n = 2, built locally: a two-trade cell is then exactly
    adequate and a one-trade cell exactly INSUFFICIENT."""
    return MetricPolicy.from_analytics(AnalyticsConfig(min_sample_n=2))


def cell(key: str, value: float | None, *, n: int = 1, metric: StabilityMetric = WIN_RATE,
         axis: ConsistencyAxis = ConsistencyAxis.WINDOW, min_sample_n: int = 1):
    """One synthetic cell carrying `value` for `metric` (and 0.0 elsewhere)."""
    return CellMeasurement.build(
        axis=axis, key=key, n_trades=n, min_sample_n=min_sample_n,
        metrics={
            m: (
                MetricValue.measured(m.value, value if m is metric else 0.0, n=n)
                if value is not None or m is not metric
                else MetricValue.undefined_because(m.value, Undefined.NO_LOSSES, n=n)
            )
            for m in STABILITY_METRICS
        },
    )


def axis_of(values, *, metric: StabilityMetric = WIN_RATE, n: int = 1) -> StabilityStatement:
    cells = [cell(f"c{i:02d}", v, n=n, metric=metric) for i, v in enumerate(values)]
    return StabilityStatement.over(cells, metric, ConsistencyAxis.WINDOW)


# ---------------------------------------------------------------------------
# The whole point: consistency over peak
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_cells", [2, 3, 4, 6, 10, 25])
def test_the_whole_point_72_everywhere_beats_90_once(n_cells: int) -> None:
    """72% in every window scores above 90% in one and 30% in the rest."""
    check = preference_check(n_cells=n_cells)
    assert check.prefers_stable
    assert check.margin > 0.0
    assert check.stable.stability.require() == pytest.approx(0.72)


def test_the_whole_point_margin_is_the_hand_computed_number() -> None:
    """k=4: A = 0.72, B = sqrt(mean 0.45 x worst 0.30) = 0.3674234614."""
    check = preference_check(n_cells=4)
    assert check.spiky.level.require() == pytest.approx(0.45)
    assert check.spiky.worst.require() == pytest.approx(0.30)
    assert check.spiky.stability.require() == pytest.approx(math.sqrt(0.45 * 0.30))
    assert check.spiky.stability.require() == pytest.approx(0.3674234614, abs=1e-9)


@pytest.mark.parametrize("n_cells", [2, 4, 10])
def test_the_boundary_holds_from_both_sides(n_cells: int) -> None:
    """The ranking must follow beta(k) exactly, not always favour the uniform
    axis: just below the boundary the stable axis wins, just above it loses. A
    formula that always preferred flatness would rank a flat 10% top."""
    boundary = rest_quality_boundary(0.72, 0.90, n_cells)
    below = preference_check(rest_value=boundary - 1e-6, n_cells=n_cells)
    above = preference_check(rest_value=boundary + 1e-6, n_cells=n_cells)
    assert below.prefers_stable
    assert not above.prefers_stable
    assert below.boundary == pytest.approx(boundary)


def test_the_boundary_rises_toward_the_stable_level_with_more_cells() -> None:
    """beta(k) -> v from below, so more windows make the spiky axis harder to
    rescue by a mediocre remainder. Values quoted in the module docstring."""
    betas = [rest_quality_boundary(0.72, 0.90, k) for k in (2, 4, 10)]
    assert betas == pytest.approx([0.6632386, 0.6948077, 0.7105919], abs=1e-6)
    assert betas[0] < betas[1] < betas[2] < 0.72


@pytest.mark.parametrize("value", [0.1, 0.3, 0.72, 0.9, 1.0])
@pytest.mark.parametrize("n_cells", [2, 5, 9])
def test_a_uniform_axis_scores_exactly_its_own_quality(value: float, n_cells: int) -> None:
    """L = W, so S = L for every k: no penalty is invented and no credit is given
    for merely having more windows."""
    statement = axis_of([value] * n_cells)
    assert statement.dispersion.require() == pytest.approx(0.0)
    assert statement.stability.require() == pytest.approx(value)


def test_one_catastrophic_window_is_not_averaged_away() -> None:
    """A zero-quality cell forces S = 0 however many good cells surround it."""
    for good in (2, 5, 20):
        statement = axis_of([0.9] * good + [0.0])
        assert statement.worst.require() == 0.0
        assert statement.stability.require() == 0.0


def test_the_dispersion_measure_does_not_dilute_while_a_moment_would() -> None:
    """The arithmetic the docstring claims: over one cell at 0 and k-1 at v, the
    sample standard deviation is v/sqrt(k) and FALLS with k, while the worst-cell
    shortfall is v(k-1)/k and RISES. A catastrophe cannot be buried by adding
    folds."""
    v = 0.72
    stdevs, shortfalls = [], []
    for k in (2, 4, 25):
        q = [v] * (k - 1) + [0.0]
        mean = sum(q) / k
        stdev = math.sqrt(sum((x - mean) ** 2 for x in q) / (k - 1))
        assert stdev == pytest.approx(v / math.sqrt(k))
        statement = axis_of(q)
        assert statement.dispersion.require() == pytest.approx(v * (k - 1) / k)
        stdevs.append(stdev)
        shortfalls.append(statement.dispersion.require())
    assert stdevs[0] > stdevs[1] > stdevs[2]
    assert shortfalls[0] < shortfalls[1] < shortfalls[2]


def test_stability_is_the_geometric_mean_of_level_and_worst() -> None:
    statement = axis_of([0.9, 0.6, 0.3])
    level, worst = statement.level.require(), statement.worst.require()
    assert level == pytest.approx(0.6)
    assert worst == pytest.approx(0.3)
    assert statement.stability.require() == pytest.approx(math.sqrt(level * worst))


def test_a_mostly_good_axis_with_one_bad_cell_loses_to_a_uniform_mediocre_one() -> None:
    """The brief's preference, restated: 90/90/90/50 scores below a flat 65."""
    spiky = axis_of([0.9, 0.9, 0.9, 0.5]).stability.require()
    flat = axis_of([0.65] * 4).stability.require()
    assert spiky == pytest.approx(math.sqrt(0.8 * 0.5))
    assert flat == pytest.approx(0.65)
    assert flat > spiky


# ---------------------------------------------------------------------------
# n = 0 and the single window
# ---------------------------------------------------------------------------


def test_the_live_case_builds_a_whole_report_and_reports_no_number(policy) -> None:
    """Zero trades is the committed system's actual behaviour (PROVENANCE.md 2.4),
    so the report must complete and refuse, not raise and not print a rate."""
    report = consistency_from_trades([], split_label=TRAIN, years=(2016, 2017), policy=policy)
    assert report.score.undefined is Undefined.NO_TRADES
    assert report.score.value is None
    assert report.score.n == 0
    assert report.score_lower_bound.value is None, "a bound of 0.0 would read as measured"
    assert report.score_lower_bound.undefined is Undefined.NO_TRADES
    assert report.coverage.require() == 0.0
    for statements in report.by_axis.values():
        for statement in statements:
            assert statement.n_cells_contributing == 0
            for stat in (statement.level, statement.worst, statement.dispersion,
                         statement.stability):
                assert stat.value is None
                assert stat.undefined is Undefined.NO_TRADES
    rendered = "\n".join(report.describe())
    assert "undefined" in rendered
    assert "win_rate=0" not in rendered


def test_dispersion_over_a_single_window_is_undefined_not_zero() -> None:
    """0.0 would make one lucky window maximally consistent -- the exact
    inversion of this module's purpose. level and worst stay measured."""
    statement = axis_of([0.9])
    assert statement.n_cells_contributing == 1
    assert statement.level.require() == pytest.approx(0.9)
    assert statement.worst.require() == pytest.approx(0.9)
    for stat in (statement.dispersion, statement.stability):
        assert stat.value is None
        assert stat.undefined is Undefined.INSUFFICIENT_OBSERVATIONS
        assert "not 0.0" in stat.note
        assert stat.n == 1


def test_a_single_window_study_has_no_score(policy) -> None:
    report = consistency_report(
        split_label=TRAIN,
        window_cells=[cell("w0", 0.9, n=2, min_sample_n=2)],
        regime_cells=[cell("r0", 0.9, n=2, axis=ConsistencyAxis.REGIME, min_sample_n=2)],
        year_cells=[cell("y0", 0.9, n=2, axis=ConsistencyAxis.YEAR, min_sample_n=2)],
        min_sample_n=policy.min_sample_n,
    )
    assert report.score.value is None
    assert report.score.undefined is Undefined.INSUFFICIENT_OBSERVATIONS


def test_min_cells_for_dispersion_is_walk_forwards_constant() -> None:
    from flow_model.validation.walk_forward import MIN_FOLDS_FOR_DISPERSION

    assert MIN_CELLS_FOR_DISPERSION is MIN_FOLDS_FOR_DISPERSION == 2


# ---------------------------------------------------------------------------
# Suppression below min_sample_n
# ---------------------------------------------------------------------------


def test_a_thin_cell_contributes_nothing_and_holds_no_number() -> None:
    thin = cell("thin", 1.0, n=1, min_sample_n=5)
    assert thin.sample is SampleStatus.INSUFFICIENT
    assert all(not v.is_defined for v in thin.metrics.values())
    assert thin.metric(WIN_RATE).undefined is Undefined.INSUFFICIENT_SAMPLE
    assert thin.n_trades == 1, "the count survives suppression; only the rate is refused"
    statement = StabilityStatement.over([thin], WIN_RATE, ConsistencyAxis.WINDOW)
    assert statement.n_cells_contributing == 0
    assert statement.stability.undefined is Undefined.INSUFFICIENT_SAMPLE


def test_an_empty_cell_is_refused_as_no_trades_not_insufficient() -> None:
    empty = cell("empty", 1.0, n=0, min_sample_n=5)
    assert empty.sample is SampleStatus.EMPTY
    assert empty.metric(WIN_RATE).undefined is Undefined.NO_TRADES


def test_a_suppressed_cell_cannot_be_constructed_holding_a_value() -> None:
    """The suppression is structural: the validator refuses the object, so no
    code path can smuggle a three-trade win rate past it."""
    with pytest.raises(ValueError, match="must hold no numbers"):
        CellMeasurement(
            axis=ConsistencyAxis.WINDOW, key="k", n_trades=1, sample=SampleStatus.INSUFFICIENT,
            min_sample_n=5,
            metrics={m.value: MetricValue.measured(m.value, 1.0, n=1) for m in STABILITY_METRICS},
        )


def test_a_cell_must_carry_all_four_metrics() -> None:
    with pytest.raises(ValueError, match="section 10 names"):
        CellMeasurement(
            axis=ConsistencyAxis.WINDOW, key="k", n_trades=4, sample=SampleStatus.ADEQUATE,
            min_sample_n=2, metrics={"win_rate": MetricValue.measured("win_rate", 0.5, n=4)},
        )


# ---------------------------------------------------------------------------
# The quality maps
# ---------------------------------------------------------------------------


def test_quality_maps_are_the_declared_ones() -> None:
    assert quality(WIN_RATE, 0.72) == pytest.approx(0.72)
    assert quality(StabilityMetric.PROFIT_FACTOR, 1.0) == pytest.approx(0.5)
    assert quality(StabilityMetric.PROFIT_FACTOR, 3.0) == pytest.approx(0.75)
    assert quality(StabilityMetric.PROFIT_FACTOR, 0.0) == 0.0
    assert quality(StabilityMetric.EXPECTANCY, 0.0) == pytest.approx(0.5)
    assert quality(StabilityMetric.EXPECTANCY, 1.0) == pytest.approx(0.75)
    assert quality(StabilityMetric.EXPECTANCY, -1.0) == pytest.approx(0.25)
    assert quality(StabilityMetric.MAX_DRAWDOWN, -0.2) == pytest.approx(0.8)
    assert quality(StabilityMetric.MAX_DRAWDOWN, 0.0) == 1.0
    assert quality(StabilityMetric.MAX_DRAWDOWN, -1.0) == 0.0


def test_the_drawdown_map_is_sign_agnostic_and_clipped() -> None:
    """`DrawdownProfile` reports negative fractions; a positive one is not a gain."""
    assert quality(StabilityMetric.MAX_DRAWDOWN, 0.3) == quality(
        StabilityMetric.MAX_DRAWDOWN, -0.3
    )
    assert quality(StabilityMetric.MAX_DRAWDOWN, -2.5) == 0.0
    assert quality(WIN_RATE, 1.4) == 1.0


@pytest.mark.parametrize("metric", list(STABILITY_METRICS))
def test_every_quality_map_is_monotone_in_better(metric: StabilityMetric) -> None:
    """A cell that improves can never lower the score."""
    sign = -1.0 if metric is StabilityMetric.MAX_DRAWDOWN else 1.0
    grid = [sign * x for x in (0.0, 0.05, 0.2, 0.5, 0.9, 1.0)]
    values = [quality(metric, v) for v in grid]
    assert values == sorted(values, reverse=metric is StabilityMetric.MAX_DRAWDOWN)


def test_a_non_finite_measurement_is_refused_rather_than_scored() -> None:
    with pytest.raises(ConsistencyError, match="not finite"):
        quality(WIN_RATE, float("nan"))


# ---------------------------------------------------------------------------
# Axes, weights, coverage
# ---------------------------------------------------------------------------


def test_the_regime_axis_is_the_six_tradable_regimes() -> None:
    assert len(CONSISTENCY_REGIMES) == 6
    assert Regime.UNKNOWN not in CONSISTENCY_REGIMES
    assert regime_dimension().domain == CONSISTENCY_REGIMES


def test_the_year_and_joint_axes_keep_declared_cells_with_no_trades(policy) -> None:
    years = (2015, 2016, 2017)
    year_cells = cells_from_breakdown(
        breakdown([], year_dimension(years), split_label=TRAIN, policy=policy),
        ConsistencyAxis.YEAR,
    )
    joint_cells = cells_from_breakdown(
        breakdown([], regime_year_dimension(years), split_label=TRAIN, policy=policy),
        ConsistencyAxis.REGIME_YEAR,
    )
    assert [c.key for c in year_cells] == ["2015", "2016", "2017"]
    assert len(joint_cells) == 3 * 6
    assert joint_cells[0].key == "2015|LOW_VOL"
    assert all(c.sample is SampleStatus.EMPTY for c in (*year_cells, *joint_cells))


def test_the_joint_axis_is_reported_and_carries_no_weight(policy) -> None:
    report = consistency_from_trades([], split_label=TRAIN, years=(2016,), policy=policy)
    assert tuple(s.metric for s in report.joint) == STABILITY_METRICS
    assert all(t.axis is not ConsistencyAxis.REGIME_YEAR for t in report.terms)
    assert "regime_year" in "\n".join(report.describe())


def test_the_weights_are_the_committed_ones_and_sum_to_one(policy) -> None:
    weights = ConsistencyConfig()
    report = consistency_from_trades(
        [], split_label=TRAIN, years=(2016,), policy=policy, weights=weights
    )
    assert len(report.terms) == 12
    assert sum(t.weight for t in report.terms) == pytest.approx(1.0)
    window = {t.metric: t.weight for t in report.terms if t.axis is ConsistencyAxis.WINDOW}
    assert window[WIN_RATE] == weights.weight_oos_win_rate_stability
    assert window[StabilityMetric.MAX_DRAWDOWN] == weights.weight_drawdown_stability
    regime = sum(t.weight for t in report.terms if t.axis is ConsistencyAxis.REGIME)
    assert regime == pytest.approx(weights.weight_regime_stability)


def test_a_statement_from_the_wrong_axis_is_refused(policy) -> None:
    with pytest.raises(ValueError, match="another axis"):
        ConsistencyReport(
            split_label=TRAIN, min_sample_n=2, weights=ConsistencyConfig(),
            window=tuple(
                StabilityStatement.over([], m, ConsistencyAxis.REGIME) for m in STABILITY_METRICS
            ),
            regime=tuple(
                StabilityStatement.over([], m, ConsistencyAxis.REGIME) for m in STABILITY_METRICS
            ),
            year=tuple(
                StabilityStatement.over([], m, ConsistencyAxis.YEAR) for m in STABILITY_METRICS
            ),
            joint=tuple(
                StabilityStatement.over([], m, ConsistencyAxis.REGIME_YEAR)
                for m in STABILITY_METRICS
            ),
        )


def _full_report(window, regime, year, *, min_sample_n: int = 2) -> ConsistencyReport:
    """A report whose cells carry all four metrics, so coverage can reach 1.0."""
    def cells(values, axis):
        return [
            CellMeasurement.build(
                axis=axis, key=f"{axis.value}{i}", n_trades=min_sample_n,
                min_sample_n=min_sample_n,
                metrics={
                    StabilityMetric.WIN_RATE: MetricValue.measured("win_rate", v, n=2),
                    StabilityMetric.PROFIT_FACTOR: MetricValue.measured("profit_factor", 2.0, n=2),
                    StabilityMetric.MAX_DRAWDOWN: MetricValue.measured(
                        "max_drawdown_pct", -0.1, n=2
                    ),
                    StabilityMetric.EXPECTANCY: MetricValue.measured("expectancy_r", 0.2, n=2),
                },
            )
            for i, v in enumerate(values)
        ]

    return consistency_report(
        split_label=TRAIN,
        window_cells=cells(window, ConsistencyAxis.WINDOW),
        regime_cells=cells(regime, ConsistencyAxis.REGIME),
        year_cells=cells(year, ConsistencyAxis.YEAR),
        min_sample_n=min_sample_n,
    )


def test_a_fully_covered_score_is_a_number_and_the_bound_agrees() -> None:
    report = _full_report([0.72] * 4, [0.72] * 6, [0.72] * 3)
    assert report.coverage.require() == pytest.approx(1.0)
    assert report.score.is_defined
    assert report.score_lower_bound.require() == pytest.approx(report.score.require())
    assert 0.0 <= report.score.require() <= 1.0


def test_the_stable_book_outscores_the_spiky_book_end_to_end() -> None:
    """The ranking survives the whole composite, not just one statement."""
    stable = _full_report([0.72] * 4, [0.72] * 6, [0.72] * 3)
    spiky = _full_report(
        [0.90, 0.30, 0.30, 0.30], [0.90, *([0.30] * 5)], [0.90, 0.30, 0.30]
    )
    assert stable.score.require() > spiky.score.require()


def test_a_partial_score_refuses_rather_than_renormalizing() -> None:
    """Max drawdown is not computable from a breakdown cell, so a trades-only
    report is partially covered. The score must refuse, naming the reason, and the
    lower bound must stay a bound -- below the number a best-case completion of
    the missing terms would give."""
    def no_drawdown_cells(values, axis):
        return [
            CellMeasurement.build(
                axis=axis, key=f"{axis.value}{i}", n_trades=2, min_sample_n=2,
                metrics={
                    StabilityMetric.WIN_RATE: MetricValue.measured("win_rate", v, n=2),
                    StabilityMetric.PROFIT_FACTOR: MetricValue.measured("profit_factor", 2.0, n=2),
                    StabilityMetric.EXPECTANCY: MetricValue.measured("expectancy_r", 0.2, n=2),
                    StabilityMetric.MAX_DRAWDOWN: MetricValue.undefined_because(
                        "max_drawdown_pct", Undefined.NO_EQUITY_SERIES, n=2
                    ),
                },
            )
            for i, v in enumerate(values)
        ]

    partial = consistency_report(
        split_label=TRAIN,
        window_cells=no_drawdown_cells([0.72] * 4, ConsistencyAxis.WINDOW),
        regime_cells=no_drawdown_cells([0.72] * 6, ConsistencyAxis.REGIME),
        year_cells=no_drawdown_cells([0.72] * 3, ConsistencyAxis.YEAR),
        min_sample_n=2,
    )
    assert partial.score.value is None
    assert partial.score.undefined is Undefined.NO_EQUITY_SERIES
    assert "Not renormalized" in partial.score.note
    # The drawdown weight is 0.15 on the window axis plus 0.15 of the regime and
    # year weights: 0.15 + (0.15 + 0.10) * 0.15 / 0.75 = 0.20 of the total.
    assert partial.coverage.require() == pytest.approx(0.80)
    bound = partial.score_lower_bound.require()
    best_completion = bound + 0.20 * 1.0
    assert bound < best_completion
    # And the bound is exactly the defined terms' weighted sum, nothing rescaled.
    defined = [t for t in partial.terms if t.value.is_defined]
    assert bound == pytest.approx(sum(t.weight * t.value.require() for t in defined))


# ---------------------------------------------------------------------------
# Trades in, and the refusals inherited from analytics
# ---------------------------------------------------------------------------


def test_a_populated_report_reaches_the_regime_and_year_axes(make_trade, policy) -> None:
    """Two years x two regimes, one winner and one loser in each cell: the win
    rate is 0.5 everywhere, so a uniform axis scores exactly 0.5."""
    plan = ((2016, Regime.LOW_VOL), (2016, Regime.LOW_VOL), (2017, Regime.CHOP),
            (2017, Regime.CHOP))
    trades = [
        make_trade(
            trade_id=f"t{index}", regime=regime,
            signal_ts=datetime(year, 6, 1, 10, 0, tzinfo=timezone.utc),
            entry=18000.0, stop=17980.0,
            exit_price=18020.0 if index % 2 == 0 else 17980.0,
        )
        for index, (year, regime) in enumerate(plan)
    ]
    report = consistency_from_trades(
        trades, split_label=TRAIN, years=(2016, 2017), policy=policy
    )
    year_wr = next(s for s in report.year if s.metric is WIN_RATE)
    assert [c.n_trades for c in year_wr.cells] == [2, 2]
    assert year_wr.n_cells_contributing == 2
    assert year_wr.level.require() == pytest.approx(0.5)
    assert year_wr.stability.require() == pytest.approx(0.5)
    regime_wr = next(s for s in report.regime if s.metric is WIN_RATE)
    assert regime_wr.n_cells == 6
    assert regime_wr.n_cells_contributing == 2
    assert "LOW_VOL" in "\n".join(regime_wr.describe())


def test_max_drawdown_from_a_breakdown_cell_is_undefined_for_want_of_equity(
    make_trade, policy
) -> None:
    trades = [make_trade(trade_id=f"t{i}") for i in range(2)]
    report = consistency_from_trades(trades, split_label=TRAIN, years=(2020,), policy=policy)
    drawdown = next(s for s in report.year if s.metric is StabilityMetric.MAX_DRAWDOWN)
    assert drawdown.per_cell[0].undefined is Undefined.NO_EQUITY_SERIES
    assert drawdown.stability.value is None


def test_mixing_split_phases_is_refused(make_trade, policy) -> None:
    """Section 0.8's structural refusal, inherited through `breakdown`."""
    trades = [
        make_trade(trade_id="a", split_label=SplitPhase.TRAIN),
        make_trade(trade_id="b", split_label=SplitPhase.TEST),
    ]
    with pytest.raises(MixedSplitError):
        consistency_from_trades(trades, split_label=TRAIN, years=(2020,), policy=policy)


def test_the_report_is_deterministic(make_trade, policy) -> None:
    """Section 0.5: same inputs, bit-identical output. No clock, no randomness."""
    trades = [make_trade(trade_id=f"t{i}") for i in range(3)]
    first = consistency_from_trades(trades, split_label=TRAIN, years=(2020,), policy=policy)
    second = consistency_from_trades(trades, split_label=TRAIN, years=(2020,), policy=policy)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_the_score_cannot_be_held_without_its_cells(policy) -> None:
    """`score` is a computed projection, so there is no constructor that takes a
    number and no way to serialize one without the cells behind it."""
    report = _full_report([0.72] * 2, [0.72] * 2, [0.72] * 2)
    dumped = report.model_dump(mode="json")
    assert "score" in dumped and dumped["score"]["value"] is not None
    assert dumped["window"][0]["cells"], "the cells travel with the score"
    assert "score" not in report.to_init_dict()
    with pytest.raises(ValueError):
        ConsistencyReport(
            split_label=TRAIN, min_sample_n=2, weights=ConsistencyConfig(),
            window=(), regime=(), year=(), joint=(), score=0.99,
        )
