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

PART 2 (below the first banner) adds, with every number hand-computed from THE
FORMULA before it is asserted:

  * the headline ranking scored as a WHOLE BOOK rather than one statement, and
    then shown not to be an accident -- the level is pinned at 0.72 and the
    spread widened, so `score == sqrt(0.72 (0.72 - s))` falls strictly and
    monotonically against dispersion `D == s`;
  * seven GAMING ATTEMPTS, with what each one actually buys reported rather
    than assumed. Two of them worked and are now refused structurally; one
    partly works and is reported as a limitation; one succeeds BY DESIGN
    (a steadily losing book outscores a profitable one with a wipeout, which is
    what "dispersion-penalized stability" means) and is pinned so nobody reads
    the score as a profitability verdict;
  * regimes and years jointly: a book whose marginals both read "perfectly
    stable at 0.5" while the regime x year grid reads stability 0.0, which the
    composite cannot see because the joint grid carries no weight.

`_metric_values_for` inverts all four declared quality maps, so one spread can
be driven through all four metrics at once and the whole twelve-term composite
collapses to a single hand-computable `S`.
"""

from __future__ import annotations

import json
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


# ===========================================================================
# PART 2: the headline property, hand-computed end to end; the gaming
# attempts; and the three structural holes those attempts went through.
#
# Every number below is computed by hand from THE FORMULA before it is
# asserted. `_metric_values_for` inverts all four declared quality maps, so one
# spread can be driven through all four metrics at once and the WHOLE composite
# collapses to a single hand-computable S -- an assertion on the score itself,
# not on one statement.
# ===========================================================================

PF = StabilityMetric.PROFIT_FACTOR
DD = StabilityMetric.MAX_DRAWDOWN
EXP = StabilityMetric.EXPECTANCY

#: `MetricValue.name` each metric must be filed under (its canonical name).
CANONICAL = {m: m.value for m in STABILITY_METRICS}


def _metric_values_for(q: float) -> dict[StabilityMetric, float]:
    """The four metric values whose declared quality maps ALL land on `q`.

    Section 10's maps, inverted by hand:

        win rate       w  = q                              from q = w
        profit factor  PF = q / (1 - q)                    from q = PF / (PF + 1)
        max drawdown   d  = -(1 - q)                       from q = 1 - |d|
        expectancy     E  = r / (1 - |r|), r = 2q - 1       from q = (1 + E/(|E|+1)) / 2

    `0 < q < 1` strictly: at q = 0 the expectancy inverse divides by zero, which
    is the honest statement that no finite expectancy maps to quality 0.
    """
    assert 0.0 < q < 1.0, "the expectancy map has no finite inverse at 0 or 1"
    r = 2.0 * q - 1.0
    return {
        WIN_RATE: q,
        PF: q / (1.0 - q),
        DD: -(1.0 - q),
        EXP: r / (1.0 - abs(r)),
    }


def _uniform_cells(qualities, axis: ConsistencyAxis, *, n: int = 2):
    """One cell per quality, every metric carrying the value that maps to it."""
    return [
        CellMeasurement.build(
            axis=axis, key=f"{axis.value}{index}", n_trades=n, min_sample_n=n,
            metrics={
                m: MetricValue.measured(CANONICAL[m], value, n=n)
                for m, value in _metric_values_for(q).items()
            },
        )
        for index, q in enumerate(qualities)
    ]


def _book(window, regime=None, year=None, *, n: int = 2) -> ConsistencyReport:
    """A fully covered report. With one quality vector on all three axes every
    one of the twelve terms is the same `S`, so `score == sqrt(L * W)` exactly."""
    return consistency_report(
        split_label=TRAIN, min_sample_n=n,
        window_cells=_uniform_cells(window, ConsistencyAxis.WINDOW, n=n),
        regime_cells=_uniform_cells(
            window if regime is None else regime, ConsistencyAxis.REGIME, n=n
        ),
        year_cells=_uniform_cells(
            window if year is None else year, ConsistencyAxis.YEAR, n=n
        ),
    )


def test_the_maps_invert_exactly_so_the_composite_is_hand_computable() -> None:
    """The inverse above is only a test fixture if it is exact; check it is."""
    for q in (0.08, 0.30, 0.50, 0.72, 0.90, 0.9333333333333333):
        for metric, value in _metric_values_for(q).items():
            assert quality(metric, value) == pytest.approx(q, abs=1e-15), metric


# ---------------------------------------------------------------------------
# THE HEADLINE PROPERTY, scored as a whole book
# ---------------------------------------------------------------------------


def test_the_headline_property_on_the_whole_score_not_one_statement() -> None:
    """72% in EVERY window against 90% in one and 30% in the rest, as composites.

    Hand-computed, k = 4, qualities read as win rates because that map is the
    identity and the other three carry the value that maps to the same quality:

        A: L = W = 0.72                      -> S = 0.72        -> score 0.72
        B: L = (0.90 + 3 x 0.30) / 4 = 0.45
           W = 0.30                          -> S = sqrt(0.135) -> score 0.3674234614

    Every one of the twelve weighted terms is that same S, and the weights sum to
    1.0, so the score IS S. Margin 0.72 - 0.3674234614 = 0.3525765386.
    """
    stable = _book([0.72] * 4)
    spiky = _book([0.90, 0.30, 0.30, 0.30])

    assert stable.window[0].level.require() == pytest.approx(0.72)
    assert stable.window[0].worst.require() == pytest.approx(0.72)
    assert stable.score.require() == pytest.approx(0.72, abs=1e-12)

    assert spiky.window[0].level.require() == pytest.approx(0.45)
    assert spiky.window[0].worst.require() == pytest.approx(0.30)
    assert spiky.score.require() == pytest.approx(math.sqrt(0.135), abs=1e-12)
    assert spiky.score.require() == pytest.approx(0.3674234614, abs=1e-9)

    assert stable.score.require() > spiky.score.require()
    assert stable.score.require() - spiky.score.require() == pytest.approx(
        0.3525765386, abs=1e-9
    )
    # Both are fully measured, so the ranking is not an artefact of coverage.
    assert stable.coverage.require() == pytest.approx(1.0)
    assert spiky.coverage.require() == pytest.approx(1.0)


#: Level pinned at 0.72, worst cell at 0.72 - s, so D = s exactly and
#: S = sqrt(0.72 * (0.72 - s)). Hand-computed; two are exact decimals
#: (s = 0.40 -> 0.48 and s = 0.64 -> 0.24) and the rest are irrational.
DISPERSION_SWEEP = (
    (0.00, 0.7200000000),
    (0.08, 0.6788225099),
    (0.16, 0.6349803147),
    (0.24, 0.5878775383),
    (0.32, 0.5366563146),
    (0.40, 0.4800000000),
    (0.48, 0.4156921938),
    (0.56, 0.3394112550),
    (0.64, 0.2400000000),
)


def test_the_ranking_is_not_an_accident_the_score_moves_against_dispersion() -> None:
    """Hold the LEVEL at 0.72 and widen the spread: the score must fall, strictly
    and monotonically, with dispersion.

    Cells `(0.72 - s, 0.72 + s/3, 0.72 + s/3, 0.72 + s/3)` keep the mean at
    exactly 0.72 -- `(0.72 - s + 3 x (0.72 + s/3)) / 4 = 2.88 / 4` -- so the only
    thing changing is the spread. Then by hand

        L = 0.72,  W = 0.72 - s,  D = L - W = s,  S = sqrt(0.72 (0.72 - s))

    and because the level is held fixed the fall cannot be a level effect. The
    headline ranking is therefore a property of the measure, not of the two books
    the brief happens to name.
    """
    scores, dispersions = [], []
    for spread, expected in DISPERSION_SWEEP:
        book = _book([0.72 - spread] + [0.72 + spread / 3.0] * 3)
        statement = book.window[0]
        assert statement.level.require() == pytest.approx(0.72, abs=1e-12)
        assert statement.worst.require() == pytest.approx(0.72 - spread, abs=1e-12)
        assert statement.dispersion.require() == pytest.approx(spread, abs=1e-12)
        assert statement.stability.require() == pytest.approx(expected, abs=1e-9)
        assert book.score.require() == pytest.approx(expected, abs=1e-9)
        assert book.coverage.require() == pytest.approx(1.0)
        scores.append(book.score.require())
        dispersions.append(statement.dispersion.require())

    assert dispersions == sorted(dispersions)
    assert scores == sorted(scores, reverse=True)
    assert all(a > b for a, b in zip(scores, scores[1:])), "strictly, not weakly"
    # Exact decimals, pinned apart from the irrational ones.
    assert scores[0] == pytest.approx(0.72, abs=1e-12)
    assert scores[5] == pytest.approx(0.48, abs=1e-12)
    assert scores[8] == pytest.approx(0.24, abs=1e-12)


def test_the_preference_holds_for_profit_factor_and_expectancy_too() -> None:
    """The brief's three numbers pushed through the other two increasing maps.

    Profit factor: q(0.72) = 0.72/1.72 = 0.4186046512, q(0.90) = 0.9/1.9 =
    0.4736842105, q(0.30) = 0.3/1.3 = 0.2307692308, so
    L = (0.4736842105 + 3 x 0.2307692308)/4 = 0.2914979757, W = 0.2307692308,
    S = sqrt(0.0672687634) = 0.2593622247 against a flat 0.4186046512.

    Expectancy (R): q(0.72) = (1 + 0.72/1.72)/2 = 0.7093023256,
    q(0.90) = 0.7368421053, q(0.30) = 0.6153846154, so L = 0.6457489879,
    W = 0.6153846154, S = sqrt(0.3973840) = 0.6303840040.
    """
    pf = preference_check(metric=PF, n_cells=4)
    assert pf.stable.stability.require() == pytest.approx(0.4186046512, abs=1e-9)
    assert pf.spiky.stability.require() == pytest.approx(0.2593622247, abs=1e-9)
    assert pf.boundary == pytest.approx(0.4108204258, abs=1e-9)
    assert pf.prefers_stable

    expectancy = preference_check(metric=EXP, n_cells=4)
    assert expectancy.stable.stability.require() == pytest.approx(0.7093023256, abs=1e-9)
    assert expectancy.spiky.stability.require() == pytest.approx(0.6303840040, abs=1e-9)
    assert expectancy.boundary == pytest.approx(0.7053805311, abs=1e-9)
    assert expectancy.prefers_stable


def test_the_mean_absolute_deviation_dilutes_as_the_docstring_claims() -> None:
    """The third statistic the docstring rules out. One cell at 0 and k-1 at v:
    MAD = 2 v (k-1) / k^2, which SHRINKS -- 0.36v at k=2, 0.375v at k=4 (the
    maximum, at k=2 and k=3 it is 0.5v and 0.444v) and 0.0768v at k=25 -- while
    D = v (k-1)/k rises. Checked so the claim is measured, not asserted."""
    v = 0.72
    mads, shortfalls = [], []
    for k in (2, 4, 25):
        q = [v] * (k - 1) + [0.0]
        mean = sum(q) / k
        mad = sum(abs(x - mean) for x in q) / k
        assert mad == pytest.approx(2.0 * v * (k - 1) / k**2)
        mads.append(mad)
        shortfalls.append(axis_of(q).dispersion.require())
    assert mads[0] > mads[1] > mads[2]
    assert shortfalls[0] < shortfalls[1] < shortfalls[2]
    assert mads == pytest.approx([0.36, 0.27, 0.0552960], abs=1e-7)
    assert shortfalls == pytest.approx([0.36, 0.54, 0.6912], abs=1e-7)


def test_one_catastrophic_window_is_not_averaged_away_with_numbers() -> None:
    """Three windows at 0.90 and one at 0.0. By hand L = 2.70/4 = 0.675 -- which
    is what an averaging report would print -- while W = 0.0, D = 0.675 and
    S = sqrt(0.675 x 0) = 0.0. The catastrophe takes the whole book, on every
    axis and in the composite, and the level stays visible beside it."""
    book = _book([0.90, 0.90, 0.90, 1e-12])
    statement = book.window[0]
    assert statement.level.require() == pytest.approx(0.675, abs=1e-9)
    assert statement.worst.require() == pytest.approx(0.0, abs=1e-9)
    assert statement.dispersion.require() == pytest.approx(0.675, abs=1e-9)
    assert statement.stability.require() == pytest.approx(0.0, abs=1e-6)
    assert book.score.require() == pytest.approx(0.0, abs=1e-6)
    # The averaging alternative would have scored it 0.675; it is 2.7e-7 instead.
    assert book.score.require() < 1e-5
    # And the worst cell is exactly 0 when the quality really is 0.
    exact = axis_of([0.90, 0.90, 0.90, 0.0])
    assert exact.level.require() == pytest.approx(0.675)
    assert exact.dispersion.require() == pytest.approx(0.675)
    assert exact.stability.require() == 0.0


def test_the_boundary_refuses_a_single_cell() -> None:
    with pytest.raises(ConsistencyError, match="dispersion over one cell is undefined"):
        rest_quality_boundary(0.72, 0.90, 1)


# ---------------------------------------------------------------------------
# The undefined cases, tightened: not 0.0, and not 100
# ---------------------------------------------------------------------------


def test_n_zero_is_undefined_and_is_neither_zero_nor_one_hundred(policy) -> None:
    """The live case (PROVENANCE.md 2.4: 283 of 19542 bars reach the scorer, max
    Flow Score 66.991221 against 70.0, so zero trades). Every refusal must stay a
    refusal: `None`, with a reason -- never 0.0, which reads as a measured worst
    case, and never a 0..100 rescaling of the [0, 1] quality scale."""
    report = consistency_from_trades([], split_label=TRAIN, years=(2016, 2017), policy=policy)
    for value in (report.score, report.score_lower_bound):
        assert value.value is None
        assert value.value != 0.0 and value.value != 100.0
        assert value.undefined is Undefined.NO_TRADES
        assert value.n == 0
    assert "0.0" not in str(report.score), str(report.score)
    assert report.score.unit == "quality"
    for statements in report.by_axis.values():
        for statement in statements:
            assert statement.qualities == ()
            for stat in (statement.level, statement.worst, statement.dispersion,
                         statement.stability):
                assert stat.value is None
                assert stat.unit == "quality"
    # coverage IS a measured 0.0 -- 0% of the committed weight was measured --
    # and that is the one honest zero in the report.
    assert report.coverage.require() == 0.0
    assert report.coverage.unit == "fraction"


def test_a_single_window_book_cannot_outrank_a_four_window_one(policy) -> None:
    """The flattering failure mode, as a ranking. One window at quality 1.0 --
    a perfect lucky fold -- must not beat 72% in four windows. It cannot, because
    it has NO score at all: dispersion over one cell is undefined, so there is no
    number to compare and `score_lower_bound` refuses too rather than reading 0.0
    as a measurement."""
    lucky = consistency_report(
        split_label=TRAIN, min_sample_n=2,
        window_cells=_uniform_cells([0.999999], ConsistencyAxis.WINDOW),
        regime_cells=_uniform_cells([0.999999], ConsistencyAxis.REGIME),
        year_cells=_uniform_cells([0.999999], ConsistencyAxis.YEAR),
    )
    for statement in lucky.window:
        assert statement.n_cells_contributing == 1
        assert statement.level.require() == pytest.approx(0.999999, abs=1e-9)
        assert statement.dispersion.value is None
        assert statement.stability.value is None
        assert statement.dispersion.undefined is Undefined.INSUFFICIENT_OBSERVATIONS
        assert "not 0.0" in statement.dispersion.note
    assert lucky.score.value is None
    assert lucky.score_lower_bound.value is None
    assert lucky.coverage.require() == 0.0

    honest = _book([0.72] * 4)
    assert honest.score.require() == pytest.approx(0.72, abs=1e-12)
    with pytest.raises(ValueError):
        lucky.score.require()


# ---------------------------------------------------------------------------
# Suppression, as arithmetic: an INSUFFICIENT cell contributes NO win rate
# ---------------------------------------------------------------------------


def test_an_insufficient_cell_contributes_no_win_rate_to_the_level() -> None:
    """A three-window axis whose best window is thin. min_sample_n = 2, so the
    one-trade 0.95 window is INSUFFICIENT and contributes nothing:

        k = 2 (not 3),  L = (0.40 + 0.60)/2 = 0.50,  W = 0.40
        D = 0.10,  S = sqrt(0.50 x 0.40) = sqrt(0.20) = 0.4472135955

    If the thin 0.95 had been let in, L would read 0.65 and S 0.5099019514. The
    0.95 must appear nowhere in the serialized statement either -- suppression
    that a reader can undo is not suppression."""
    cells = [
        cell("thin", 0.95, n=1, min_sample_n=2),
        cell("w1", 0.40, n=5, min_sample_n=2),
        cell("w2", 0.60, n=5, min_sample_n=2),
    ]
    statement = StabilityStatement.over(cells, WIN_RATE, ConsistencyAxis.WINDOW)
    assert statement.n_cells == 3
    assert statement.n_cells_contributing == 2
    assert statement.qualities == pytest.approx((0.40, 0.60))
    assert statement.level.require() == pytest.approx(0.50)
    assert statement.worst.require() == pytest.approx(0.40)
    assert statement.dispersion.require() == pytest.approx(0.10)
    assert statement.stability.require() == pytest.approx(math.sqrt(0.20))
    assert statement.stability.require() == pytest.approx(0.4472135955, abs=1e-9)
    assert statement.level.require() != pytest.approx(0.65), "the thin cell leaked in"
    assert statement.undefined_reasons == {"insufficient_sample": 1}
    dumped = json.dumps(statement.model_dump(mode="json"))
    assert "0.95" not in dumped, "the suppressed rate is still readable off the object"
    assert '"n_trades": 1' in dumped, "the COUNT survives; only the rate is refused"


# ---------------------------------------------------------------------------
# GAMING ATTEMPTS
# ---------------------------------------------------------------------------


def _realistic_cells(
    axis, win_rate, profit_factor, drawdown, expectancy, *, k: int = 4, n: int = 2
):
    """Cells in the metrics' own units, so a gaming book reads like a real book.
    A one-element vector is the same value in all `k` cells."""

    def spread(values):
        assert len(values) in (1, k), f"{len(values)} values for {k} cells"
        return list(values) * k if len(values) == 1 else list(values)

    w, p, d, e = (spread(v) for v in (win_rate, profit_factor, drawdown, expectancy))
    return [
        CellMeasurement.build(
            axis=axis, key=f"{axis.value}{i}", n_trades=n, min_sample_n=n,
            metrics={
                WIN_RATE: MetricValue.measured("win_rate", w[i], n=n),
                PF: MetricValue.measured("profit_factor", p[i], n=n),
                DD: MetricValue.measured("max_drawdown_pct", d[i], n=n),
                EXP: MetricValue.measured("expectancy_r", e[i], n=n),
            },
        )
        for i in range(k)
    ]


def _realistic_book(
    win_rate, profit_factor, drawdown, expectancy, *, k: int = 4
) -> ConsistencyReport:
    args = (win_rate, profit_factor, drawdown, expectancy)
    return consistency_report(
        split_label=TRAIN, min_sample_n=2,
        window_cells=_realistic_cells(ConsistencyAxis.WINDOW, *args, k=k),
        regime_cells=_realistic_cells(ConsistencyAxis.REGIME, *args, k=k),
        year_cells=_realistic_cells(ConsistencyAxis.YEAR, *args, k=k),
    )


def test_gaming_one_great_window_among_many_thin_ones_gets_no_score() -> None:
    """ATTEMPT 1: show one good fold and let the rest fall below min_sample_n.

    It buys nothing: the thin cells contribute nothing, so k = 1 and dispersion,
    stability and therefore the score are all UNDEFINED. The one good number
    cannot be published as a consistency score at all."""
    for thin in (1, 5, 20):
        cells = [cell("good", 0.95, n=2, min_sample_n=2)]
        cells += [cell(f"thin{i}", 0.95, n=1, min_sample_n=2) for i in range(thin)]
        statement = StabilityStatement.over(cells, WIN_RATE, ConsistencyAxis.WINDOW)
        assert statement.n_cells_contributing == 1
        assert statement.n_cells == thin + 1
        assert statement.stability.value is None
        assert statement.dispersion.value is None
        assert statement.undefined_reasons == {"insufficient_sample": thin}
        assert f"1 of {thin + 1} cell(s) contributed" in statement.stability.note


def test_gaming_two_good_windows_among_twenty_thin_ones_is_unpenalised_but_stated() -> None:
    """ATTEMPT 2, and it PARTLY WORKS -- reported rather than asserted away.

    Two adequate windows at 0.95 plus twenty INSUFFICIENT ones at 0.10 score
    EXACTLY what two adequate windows alone score: 0.95. The thin cells are
    suppressed, so twenty refused folds cost the score nothing, and `coverage`
    does not see them either -- it measures the share of committed WEIGHT that
    was measured, not the share of cells that contributed.

    What the module does guarantee is that the dilution is VISIBLE and
    structural: `n_cells_contributing` is 2 of 22, the note on every aggregate
    says so, and the per-cell lines come first in `describe()`. A reader who
    reads only the number is not protected by the formula here; a reader who
    reads the statement is."""
    two_only = StabilityStatement.over(
        [cell(f"good{j}", 0.95, n=2, min_sample_n=2) for j in range(2)],
        WIN_RATE, ConsistencyAxis.WINDOW,
    )
    padded = StabilityStatement.over(
        [cell(f"good{j}", 0.95, n=2, min_sample_n=2) for j in range(2)]
        + [cell(f"thin{i}", 0.10, n=1, min_sample_n=2) for i in range(20)],
        WIN_RATE, ConsistencyAxis.WINDOW,
    )
    assert two_only.stability.require() == pytest.approx(0.95)
    assert padded.stability.require() == pytest.approx(0.95)
    assert padded.stability.require() == two_only.stability.require(), (
        "MEASURED: twenty suppressed windows cost the stability nothing"
    )
    # Visible, though: 2 of 22, and the reason histogram names the twenty.
    assert (padded.n_cells_contributing, padded.n_cells) == (2, 22)
    assert "2 of 22 cell(s) contributed" in padded.stability.note
    assert padded.undefined_reasons == {"insufficient_sample": 20}
    lines = padded.describe()
    assert lines[0].endswith("over 22 cell(s):")
    assert sum(1 for line in lines if "n_trades=1" in line) == 20
    # And the per-cell lines precede the aggregate, so the number is never alone.
    assert lines.index(f"  {padded.stability}") > 20


def test_gaming_a_uniformly_terrible_book_scores_below_breakeven() -> None:
    """ATTEMPT 3: get a low dispersion by being equally bad everywhere.

    A book that is 35% win rate, 0.80 profit factor, -0.05R expectancy and a 30%
    drawdown in EVERY window has dispersion 0.0 on every axis. Hand-computed
    qualities and score:

        q(win rate 0.35)    = 0.35
        q(PF 0.80)          = 0.80/1.80          = 0.4444444444
        q(drawdown -0.30)   = 1 - 0.30           = 0.70
        q(expectancy -0.05) = (1 - 0.05/1.05)/2  = 0.4761904762

        score = (0.20 x 0.35 + 0.20 x 0.4444444444
                 + 0.15 x 0.70 + 0.20 x 0.4761904762) / 0.75
              = 0.3591269841 / 0.75 = 0.4788359788

    It is NOT rewarded: 0.4789 sits below the flat-breakeven book's 0.56 and well
    below the stable-72% book's 0.7265787546. But it is not floored either, and
    that is the honest reading of the design -- both profitability maps send
    breakeven to 0.5, so 0.4789 means "stable, and just under breakeven". The
    score is a STABILITY score; it never claims the book makes money."""
    terrible = _realistic_book([0.35], [0.80], [-0.30], [-0.05])
    breakeven = _realistic_book([0.50], [1.00], [-0.20], [0.00])
    good = _realistic_book([0.72], [2.50], [-0.10], [0.30])

    assert terrible.window[0].dispersion.require() == 0.0
    assert terrible.score.require() == pytest.approx(0.4788359788, abs=1e-9)
    assert breakeven.score.require() == pytest.approx(0.56, abs=1e-12)
    assert good.score.require() == pytest.approx(0.7265787546, abs=1e-9)
    assert terrible.score.require() < breakeven.score.require() < good.score.require()
    # Each axis is fully measured, so the ordering is not a coverage artefact.
    for book in (terrible, breakeven, good):
        assert book.coverage.require() == pytest.approx(1.0)


def test_gaming_a_terrible_uniform_book_outscores_a_profitable_one_with_a_wipeout() -> None:
    """ATTEMPT 3, continued -- and this one SUCCEEDS, by design. Reported.

    A book that loses steadily (0.4788359788) outscores a book that is excellent
    in three windows and wiped out in the fourth (0.2183260793). Hand-computed
    for the spiky book:

        win rate      q = (0.90, 0.90, 0.90, 0.00)  L = 0.675   W = 0  S = 0
        profit factor q = (0.75, 0.75, 0.75, 0.00)  L = 0.5625  W = 0  S = 0
        drawdown      q = (0.95, 0.95, 0.95, 0.40)  L = 0.8125  W = 0.40
                                                    S = sqrt(0.325) = 0.5700877125
        expectancy    q = (0.6875 x3, 0.2631578947) L = 0.5814394737
                                                    W = 0.2631578947
                                                    S = 0.3911570131

        score = (0.15 x 0.5700877125 + 0.20 x 0.3911570131) / 0.75
              = 0.1637445595 / 0.75 = 0.2183260793

    That ranking IS the brief's instruction -- "one catastrophic window must not
    be averaged away" -- so it is pinned here rather than treated as a bug. The
    consequence a reader must know: this number cannot be used to choose between
    two systems on profitability, only on stability. ARCHITECTURE.md 10 asks for
    "dispersion-penalized stability", and that is what it is."""
    terrible = _realistic_book([0.35], [0.80], [-0.30], [-0.05])
    spiky = _realistic_book(
        [0.90, 0.90, 0.90, 0.00], [3.0, 3.0, 3.0, 0.0],
        [-0.05, -0.05, -0.05, -0.60], [0.6, 0.6, 0.6, -0.9],
    )
    assert spiky.window[2].stability.require() == pytest.approx(0.5700877125, abs=1e-9)
    assert spiky.window[3].stability.require() == pytest.approx(0.3911570131, abs=1e-9)
    assert spiky.window[0].stability.require() == 0.0
    assert spiky.window[1].stability.require() == 0.0
    assert spiky.score.require() == pytest.approx(0.2183260793, abs=1e-9)
    assert terrible.score.require() > spiky.score.require(), (
        "MEASURED: 0.4788 > 0.2183. Stability, not profitability."
    )


def test_gaming_zero_expectancy_everywhere_scores_the_breakeven_midpoint() -> None:
    """ATTEMPT 4: zero expectancy, zero spread -- the lowest-dispersion book that
    is not actively losing. Expectancy 0.0R maps to quality 0.5 exactly (both
    profitability maps send breakeven to 0.5 by construction), so its stability
    is 0.5 and no amount of flatness pushes it higher.

    With a 0% drawdown as well -- a book that never traded badly because it
    barely traded -- the whole score is (0.2x0.5 + 0.2x0.5 + 0.15x1.0 +
    0.2x0.5)/0.75 = 0.45/0.75 = 0.60. That is the ceiling flatness alone can buy
    at breakeven, and it is visibly not the 0.72 a genuinely 72%-stable book
    gets."""
    flat = _realistic_book([0.50], [1.00], [0.00], [0.00])
    assert quality(EXP, 0.0) == pytest.approx(0.5)
    assert quality(PF, 1.0) == pytest.approx(0.5)
    for statement in flat.window:
        assert statement.dispersion.require() == 0.0
    assert flat.window[3].stability.require() == pytest.approx(0.5)
    assert flat.score.require() == pytest.approx(0.60, abs=1e-12)
    assert flat.score.require() < _book([0.72] * 4).score.require()


def test_gaming_a_flawless_book_has_no_score_because_profit_factor_refuses(
    make_trade, policy
) -> None:
    """ATTEMPT 5: win every trade. A profit factor with no losing trade has no
    denominator, so `analytics` reports NO_LOSSES and the profit-factor term
    refuses; the score then refuses rather than renormalizing over the terms that
    did resolve. An unbroken winning streak buys no consistency score."""
    trades = [
        make_trade(trade_id=f"t{i}", signal_ts=datetime(2016, 6, 1, 10, 0, tzinfo=timezone.utc))
        for i in range(4)
    ]
    report = consistency_from_trades(trades, split_label=TRAIN, years=(2016,), policy=policy)
    profit_factor = next(s for s in report.regime if s.metric is PF)
    assert profit_factor.undefined_reasons["no_losses"] == 1
    assert profit_factor.stability.value is None
    assert report.score.value is None
    assert "Not renormalized" in report.score.note


def test_gaming_dropping_the_bad_window_from_three_of_four_metrics_is_refused() -> None:
    """ATTEMPT 6, and it WORKED until this test: assemble an axis whose four
    metrics are measured over DIFFERENT cell sets, so the catastrophic window
    stays visible in the win-rate statement -- where a reader looks for it -- and
    is absent from the other three.

    MEASURED before the fix: window drawdown stability 0.95 and a score of
    0.5918803418803419 for a book whose third window had a 90% drawdown and no
    winners. Honest (the same window present in all four statements): drawdown
    stability 0.2581988897471611 and a score of 0.4881101753424161 -- the gaming
    bought 0.104 of score and hid a 90% drawdown behind a 0.95.

    `ConsistencyReport` now refuses the construction: an axis IS one set of
    cells."""
    def cells(axis, keys_and_values):
        return [
            CellMeasurement.build(
                axis=axis, key=key, n_trades=5, min_sample_n=2,
                metrics={
                    WIN_RATE: MetricValue.measured("win_rate", win, n=5),
                    PF: MetricValue.measured("profit_factor", 2.0, n=5),
                    DD: MetricValue.measured("max_drawdown_pct", drawdown, n=5),
                    EXP: MetricValue.measured("expectancy_r", 0.3, n=5),
                },
            )
            for key, win, drawdown in keys_and_values
        ]

    pair = [("c0", 0.9, -0.05), ("c1", 0.9, -0.05)]
    withbad = pair + [("c2", 0.0, -0.90)]

    def window(spec):
        return cells(ConsistencyAxis.WINDOW, spec)

    flat = cells(ConsistencyAxis.REGIME, pair)
    years = cells(ConsistencyAxis.YEAR, pair)

    with pytest.raises(ValueError, match="an axis is one set of cells"):
        ConsistencyReport(
            split_label=TRAIN, min_sample_n=2, weights=ConsistencyConfig(),
            window=(
                StabilityStatement.over(window(withbad), WIN_RATE, ConsistencyAxis.WINDOW),
                StabilityStatement.over(window(pair), PF, ConsistencyAxis.WINDOW),
                StabilityStatement.over(window(pair), DD, ConsistencyAxis.WINDOW),
                StabilityStatement.over(window(pair), EXP, ConsistencyAxis.WINDOW),
            ),
            regime=tuple(
                StabilityStatement.over(flat, m, ConsistencyAxis.REGIME)
                for m in STABILITY_METRICS
            ),
            year=tuple(
                StabilityStatement.over(years, m, ConsistencyAxis.YEAR)
                for m in STABILITY_METRICS
            ),
            joint=tuple(
                StabilityStatement.over([], m, ConsistencyAxis.REGIME_YEAR)
                for m in STABILITY_METRICS
            ),
        )

    honest = consistency_report(
        split_label=TRAIN, min_sample_n=2, window_cells=window(withbad),
        regime_cells=flat, year_cells=years,
    )
    assert honest.window[2].stability.require() == pytest.approx(0.2581988897, abs=1e-9)
    assert honest.score.require() == pytest.approx(0.4881101753, abs=1e-9)


def test_gaming_a_metric_filed_under_another_metrics_name_is_refused() -> None:
    """ATTEMPT 7, and it WORKED until this test: `_pick` finds a metric by its
    enum key, so a `MetricValue` named anything at all could be filed as the win
    rate and scored by the win-rate map.

    MEASURED before the fix: a Sharpe ratio of 3.4 filed under `WIN_RATE` gave a
    window win-rate stability of 1.0 (the map clips to [0, 1]). True value: the
    cell must not be constructible. `StabilityMetric`'s docstring claimed exactly
    this guarantee; nothing enforced it."""
    sharpe = {
        WIN_RATE: MetricValue.measured("sharpe_ratio", 3.4, n=5),
        PF: MetricValue.measured("profit_factor", 1.0, n=5),
        DD: MetricValue.measured("max_drawdown_pct", 0.0, n=5),
        EXP: MetricValue.measured("expectancy_r", 0.0, n=5),
    }
    with pytest.raises(ValueError, match="names that are not their own"):
        CellMeasurement.build(
            axis=ConsistencyAxis.WINDOW, key="k", n_trades=5, min_sample_n=2, metrics=sharpe
        )
    # The alias each metric declares is still accepted, so `analytics`' own
    # names ("expectancy" -> expectancy_r, "max_drawdown") keep working.
    ok = dict(sharpe)
    ok[WIN_RATE] = MetricValue.measured("win_rate", 0.4, n=5)
    ok[EXP] = MetricValue.measured("expectancy", 0.0, n=5)
    ok[DD] = MetricValue.measured("max_drawdown", 0.0, n=5)
    built = CellMeasurement.build(
        axis=ConsistencyAxis.WINDOW, key="k", n_trades=5, min_sample_n=2, metrics=ok
    )
    assert built.metric(WIN_RATE).require() == pytest.approx(0.4)


def test_the_preference_check_refuses_a_shape_the_guarantee_does_not_cover() -> None:
    """`beta(k)` is derived for ONE shape: `v` everywhere against `p` once and `b`
    in the other k-1. Max drawdown's quality map DECREASES, so `peak_value=0.90`
    (a 90% drawdown) maps to quality 0.10 and `rest_value=0.30` to 0.70 -- the
    "spiky" axis becomes one BAD cell among three good ones, which is a different
    comparison entirely.

    MEASURED before the fix: `preference_check(metric=MAX_DRAWDOWN).boundary`
    returned 0.2270547968 for qualities (0.10, 0.70, 0.70, 0.70). True value:
    none -- beta does not describe that shape, so the object must not be built.
    `prefers_stable` and `margin` were right all along (they call the real
    formula); only `boundary` was a number about a comparison nobody made."""
    with pytest.raises(ValueError, match="one peak, the rest equal"):
        preference_check(metric=DD)
    with pytest.raises(ValueError, match="one peak, the rest equal"):
        preference_check(rest_value=0.90)
    with pytest.raises(ValueError, match="needs at least"):
        preference_check(n_cells=1)
    # A drawdown comparison is expressible -- it just has to be built with the
    # peak as the SHALLOW window: 5% once, 60% in the rest.
    drawdown = preference_check(
        metric=DD, stable_value=0.28, peak_value=0.05, rest_value=0.60, n_cells=4
    )
    assert drawdown.stable.qualities == pytest.approx((0.72,) * 4)
    assert drawdown.spiky.qualities == pytest.approx((0.95, 0.40, 0.40, 0.40))
    assert drawdown.stable.stability.require() == pytest.approx(0.72)
    assert drawdown.spiky.stability.require() == pytest.approx(math.sqrt(0.5375 * 0.40))
    assert drawdown.prefers_stable


# ---------------------------------------------------------------------------
# Regimes and years: separately, and jointly
# ---------------------------------------------------------------------------


def test_the_regime_and_year_axes_carry_their_own_committed_weight() -> None:
    """Three axes with three different stabilities, so each committed weight is
    visible in the composite instead of cancelling.

        window  uniform 0.72        -> S = 0.72           weight 0.75
        regime  uniform 0.90        -> S = 0.90           weight 0.15
        year    (0.90, 0.30, 0.30)  -> L = 0.50, W = 0.30
                                       S = sqrt(0.15)
                                         = 0.3872983346   weight 0.10

        score = 0.75 x 0.72 + 0.15 x 0.90 + 0.10 x 0.3872983346
              = 0.54 + 0.135 + 0.0387298335 = 0.7137298335

    The inner weighting of the regime and year axes is the same four metric
    weights renormalized over 0.75, so with one quality vector per axis each axis
    contributes its own weight times its own S and nothing else.
    """
    report = consistency_report(
        split_label=TRAIN, min_sample_n=2,
        window_cells=_uniform_cells([0.72] * 4, ConsistencyAxis.WINDOW),
        regime_cells=_uniform_cells([0.90] * 6, ConsistencyAxis.REGIME),
        year_cells=_uniform_cells([0.90, 0.30, 0.30], ConsistencyAxis.YEAR),
    )
    assert report.window[0].stability.require() == pytest.approx(0.72, abs=1e-12)
    assert report.regime[0].stability.require() == pytest.approx(0.90, abs=1e-12)
    assert report.year[0].level.require() == pytest.approx(0.50, abs=1e-12)
    assert report.year[0].worst.require() == pytest.approx(0.30, abs=1e-12)
    assert report.year[0].stability.require() == pytest.approx(0.3872983346, abs=1e-9)
    assert report.score.require() == pytest.approx(0.7137298335, abs=1e-9)

    by_axis = {}
    for term in report.terms:
        by_axis[term.axis] = by_axis.get(term.axis, 0.0) + term.weight
    assert by_axis[ConsistencyAxis.WINDOW] == pytest.approx(0.75)
    assert by_axis[ConsistencyAxis.REGIME] == pytest.approx(0.15)
    assert by_axis[ConsistencyAxis.YEAR] == pytest.approx(0.10)
    assert ConsistencyAxis.REGIME_YEAR not in by_axis


def test_a_catastrophe_visible_only_jointly_is_reported_but_carries_no_weight(
    make_trade, policy
) -> None:
    """Section 10 asks for "across regimes and across years"; the joint grid is
    why "separately" is not enough.

    Eight trades, min_sample_n = 2: LOW_VOL wins twice in 2016 and loses twice in
    2017; CHOP does the opposite. Both MARGINALS then read perfectly stable --

        regime  LOW_VOL 2/4 = 0.5,  CHOP 2/4 = 0.5  -> L = W = 0.5, D = 0.0, S = 0.5
        year    2016    2/4 = 0.5,  2017  2/4 = 0.5 -> L = W = 0.5, D = 0.0, S = 0.5

    -- while the joint grid shows the regimes inverting:

        joint   (2016|LOW_VOL, 2016|CHOP, 2017|LOW_VOL, 2017|CHOP)
                = (1.0, 0.0, 0.0, 1.0)  -> L = 0.5, W = 0.0, D = 0.5, S = 0.0

    So the joint grid is the only axis that sees it. REPORTED FINDING: it carries
    no weight, by the stated design (`ConsistencyConfig` commits six weights and
    a seventh would be a parameter this module invented for itself), so the
    COMPOSITE is blind to this failure -- it is in the report, in full, and not in
    the score. A reader of `score` alone would call this book stable."""
    plan = (
        (2016, Regime.LOW_VOL, True), (2016, Regime.LOW_VOL, True),
        (2017, Regime.LOW_VOL, False), (2017, Regime.LOW_VOL, False),
        (2016, Regime.CHOP, False), (2016, Regime.CHOP, False),
        (2017, Regime.CHOP, True), (2017, Regime.CHOP, True),
    )
    trades = [
        make_trade(
            trade_id=f"t{i}", regime=regime,
            signal_ts=datetime(year, 6, 1, 10, 0, tzinfo=timezone.utc),
            entry=18000.0, stop=17980.0, exit_price=18020.0 if win else 17980.0,
        )
        for i, (year, regime, win) in enumerate(plan)
    ]
    report = consistency_from_trades(
        trades, split_label=TRAIN, years=(2016, 2017), policy=policy
    )

    regime_wr = next(s for s in report.regime if s.metric is WIN_RATE)
    year_wr = next(s for s in report.year if s.metric is WIN_RATE)
    joint_wr = next(s for s in report.joint if s.metric is WIN_RATE)

    for marginal in (regime_wr, year_wr):
        assert marginal.qualities == pytest.approx((0.5, 0.5))
        assert marginal.level.require() == pytest.approx(0.5)
        assert marginal.worst.require() == pytest.approx(0.5)
        assert marginal.dispersion.require() == pytest.approx(0.0)
        assert marginal.stability.require() == pytest.approx(0.5)

    assert joint_wr.n_cells == 2 * 6
    assert joint_wr.n_cells_contributing == 4
    assert [c.key for c in joint_wr.cells if c.n_trades] == [
        "2016|LOW_VOL", "2016|CHOP", "2017|LOW_VOL", "2017|CHOP"
    ]
    assert joint_wr.qualities == pytest.approx((1.0, 0.0, 0.0, 1.0))
    assert joint_wr.level.require() == pytest.approx(0.5)
    assert joint_wr.worst.require() == 0.0
    assert joint_wr.dispersion.require() == pytest.approx(0.5)
    assert joint_wr.stability.require() == 0.0

    # The finding, pinned: nothing weighted sees the joint grid.
    assert all(t.axis is not ConsistencyAxis.REGIME_YEAR for t in report.terms)
    rendered = "\n".join(report.describe())
    assert "axis regime_year (reported, unweighted)" in rendered
    assert "2017|LOW_VOL" in rendered


def test_an_untradable_regime_reaches_the_joint_grid_instead_of_crashing(
    make_trade, policy
) -> None:
    """`CONSISTENCY_REGIMES` excludes `Regime.UNKNOWN` ("insufficient warmup;
    never tradable"), but `TradeRecord.regime` does not forbid it, so a trade can
    carry it -- from a database read, a degradation run, or a fixture.

    MEASURED before the fix: `consistency_from_trades` died with a bare
    `ValueError: tuple.index(x): x not in tuple`, raised from
    `regime_year_dimension`'s sort key, taking the whole consistency report with
    it. True behaviour: the regime axis already reported such a trade as an extra
    cell (its dimension has no sort key), so the joint grid now does too. The
    trade is REPORTED, sorted after the six declared regimes, and nothing is
    hidden or fatal."""
    trades = [
        make_trade(
            trade_id=f"u{i}", regime=Regime.UNKNOWN,
            signal_ts=datetime(2016, 6, 1, 10, 0, tzinfo=timezone.utc),
        )
        for i in range(2)
    ]
    report = consistency_from_trades(trades, split_label=TRAIN, years=(2016,), policy=policy)
    regime_keys = [c.key for c in report.regime[0].cells]
    joint_keys = [c.key for c in report.joint[0].cells]
    assert regime_keys == [*(str(r) for r in CONSISTENCY_REGIMES), "UNKNOWN"]
    assert joint_keys == [*(f"2016|{r}" for r in CONSISTENCY_REGIMES), "2016|UNKNOWN"]
    unknown = next(c for c in report.joint[0].cells if c.key == "2016|UNKNOWN")
    assert unknown.n_trades == 2
    assert unknown.sample is SampleStatus.ADEQUATE
    # Deterministic: the same trades sort the same way every time (section 0.5).
    again = consistency_from_trades(trades, split_label=TRAIN, years=(2016,), policy=policy)
    assert report.model_dump(mode="json") == again.model_dump(mode="json")


# ---------------------------------------------------------------------------
# What a reader is shown
# ---------------------------------------------------------------------------


def test_the_declared_suppression_threshold_must_be_the_applied_one() -> None:
    """`min_sample_n` reaches `consistency_report` as its own argument, separately
    from the cells, so the number printed in the header and the number the cells
    were suppressed at could differ.

    MEASURED before the fix: a report built with `min_sample_n=30` over cells
    suppressed at 1 printed `consistency split=None min_sample_n=30` above a
    score of 0.9000000000000002 computed from FOUR ONE-TRADE CELLS. Section 12
    calls small-sample rates "the primary way people fool themselves"; this
    printed a reassurance beside exactly that. True behaviour: refused."""
    thin = [
        CellMeasurement.build(
            axis=axis, key=f"{axis.value}{i}", n_trades=1, min_sample_n=1,
            metrics={m: MetricValue.measured(CANONICAL[m], v, n=1)
                     for m, v in _metric_values_for(0.9).items()},
        )
        for axis in (ConsistencyAxis.WINDOW,) for i in range(4)
    ]
    regime = _uniform_cells([0.9] * 4, ConsistencyAxis.REGIME, n=1)
    year = _uniform_cells([0.9] * 4, ConsistencyAxis.YEAR, n=1)
    with pytest.raises(ValueError, match="must be the threshold that was applied"):
        consistency_report(
            split_label=TRAIN, min_sample_n=30, window_cells=thin,
            regime_cells=regime, year_cells=year,
        )
    # Declared == applied is accepted, and then the header is true.
    honest = consistency_report(
        split_label=TRAIN, min_sample_n=1, window_cells=thin,
        regime_cells=regime, year_cells=year,
    )
    assert honest.min_sample_n == 1
    assert "min_sample_n=1" in honest.describe()[0]
    assert honest.score.require() == pytest.approx(0.9, abs=1e-12)


def test_at_n_zero_every_line_a_reader_sees_is_a_refusal(policy) -> None:
    """`describe()` is what a reader actually reads, so the refusal has to survive
    rendering. At n = 0 the only number in the whole report is `coverage=0`, which
    is a true measurement (0% of the committed weight was measured); all 4 axes x
    4 metrics x 4 statistics, all 12 terms, the score and its lower bound render
    as `undefined[no_trades]`."""
    report = consistency_from_trades([], split_label=TRAIN, years=(2016, 2017), policy=policy)
    lines = report.describe()
    stats = [line for line in lines if "_level=" in line or "_worst=" in line
             or "_dispersion=" in line or "_stability=" in line]
    assert len(stats) == 4 * 4 * 4 + 12, "16 statements x 4 statistics, plus 12 terms"
    assert all("undefined[no_trades]" in line for line in stats), stats
    assert "consistency_score=undefined[no_trades] (n=0)" in lines
    assert "consistency_score_lower_bound=undefined[no_trades] (n=0)" in lines
    assert "consistency_score_coverage=0 fraction (n=0)" in lines
    numeric = [line for line in lines if "undefined[" not in line and "=" in line
               and "min_sample_n" not in line and "n_trades=0" not in line]
    assert numeric == ["consistency_score_coverage=0 fraction (n=0)"], numeric


def test_a_zero_weight_term_that_refuses_still_refuses_the_score() -> None:
    """REPORTED, not fixed: `coverage` can read 1.0 while `score` refuses.

    `coverage` is the share of committed WEIGHT that was measured, so a term
    carrying weight 0.0 that refuses costs no coverage -- yet `_combine` refuses
    the score on any undefined term, weight or no weight. MEASURED with a locally
    built config that drops the win-rate weight to 0.0 and an axis whose win rate
    is undefined: coverage 1.0, score undefined[no_trades], lower bound 0.9.

    Both statements are individually true and the score's note names the term that
    refused, so nothing is hidden. Left alone deliberately: making the score skip
    zero-weight terms would weaken a refusal, and making coverage count terms
    rather than weight would contradict its stated definition. Unreachable at the
    committed config, where all six weights are non-zero."""
    weights = ConsistencyConfig(
        weight_oos_win_rate_stability=0.0, weight_profit_factor_stability=0.30,
        weight_drawdown_stability=0.15, weight_expectancy_stability=0.30,
        weight_regime_stability=0.15, weight_year_stability=0.10,
    )

    def no_win_rate(qualities, axis):
        out = []
        for index, q in enumerate(qualities):
            metrics = {
                m: MetricValue.measured(CANONICAL[m], v, n=2)
                for m, v in _metric_values_for(q).items()
            }
            metrics[WIN_RATE] = MetricValue.undefined_because(
                "win_rate", Undefined.NO_TRADES, n=2
            )
            out.append(CellMeasurement.build(
                axis=axis, key=f"{axis.value}{index}", n_trades=2, min_sample_n=2,
                metrics=metrics,
            ))
        return out

    report = consistency_report(
        split_label=TRAIN, min_sample_n=2, weights=weights,
        window_cells=no_win_rate([0.9] * 4, ConsistencyAxis.WINDOW),
        regime_cells=no_win_rate([0.9] * 4, ConsistencyAxis.REGIME),
        year_cells=no_win_rate([0.9] * 4, ConsistencyAxis.YEAR),
    )
    assert report.coverage.require() == pytest.approx(1.0)
    assert report.score.value is None
    assert report.score_lower_bound.require() == pytest.approx(0.9, abs=1e-12)
    assert "weight_oos_win_rate_stability" in report.score.note
    assert "9 of 12 term(s) defined" in report.score.note
    zero_weight = [t.label for t in report.terms if t.weight == 0.0]
    assert zero_weight == [
        "weight_oos_win_rate_stability",
        "weight_regime_stability/win_rate",
        "weight_year_stability/win_rate",
    ]
    assert all(not t.value.is_defined for t in report.terms if t.weight == 0.0)
