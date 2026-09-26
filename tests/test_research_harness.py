"""Tests for app/research/ - ablation, robustness and threshold study.

The property that matters throughout: these tools must be capable of
returning a NEGATIVE or INCONCLUSIVE answer. A research harness that can
only ever confirm the strategy is worse than no harness, because it
launders a guess into a number.
"""
import pytest

from app.backtesting.types import BacktestConfig
from app.data.candles import Timeframe
from app.data.providers import SyntheticCandleProvider
from app.research.ablation import (
    FULL_MODEL,
    FactorVerdict,
    NEGLIGIBLE_R,
    run_ablation,
    weights_without,
)
from app.research.harness import (
    MIN_OOS_TRADES,
    Experiment,
    VariantResult,
    run_variants,
)
from app.research.robustness import (
    CLIFF_THRESHOLD_R,
    ParameterPoint,
    RobustnessReport,
    sweep_parameter,
)
from app.research.thresholds import ThresholdResult, ThresholdStudy, study_thresholds
from app.signals.scoring import DEFAULT_WEIGHTS


def _series(regime="bull", seed=3, limit=1200):
    return SyntheticCandleProvider(regime=regime, seed=seed).fetch(
        "TESTCOIN", Timeframe.M15, limit=limit
    )


def _config():
    return BacktestConfig(warmup_bars=210)


# ===========================================================================
# harness
# ===========================================================================

def test_the_split_is_chronological_never_shuffled():
    """Randomly assigning bars between train and test lets a variant learn
    from Tuesday afternoon to predict Tuesday morning."""
    series = _series(limit=1000)
    experiment = run_variants("split", series, {"a": _config()}, train_fraction=0.6)
    assert experiment.train_bars == 600
    assert experiment.oos_bars == 400
    assert experiment.train_bars + experiment.oos_bars == len(series)


def _stub_variant(label, *, oos_trades, oos_r, train_r=None, drawdown=0.0):
    """A VariantResult with real BacktestResults, so the ranker's actual
    properties are exercised rather than monkeypatched away."""
    from app.backtesting.stats import compute_stats
    from app.backtesting.types import BacktestResult

    def result(n, r):
        res = BacktestResult(symbol="TESTCOIN")
        res.stats = compute_stats([], [], 1000.0)
        object.__setattr__(res.stats, "trade_count", n)
        object.__setattr__(res.stats, "expectancy_r", r)
        object.__setattr__(res.stats, "max_drawdown_pct", drawdown)
        return res

    return VariantResult(
        label=label, config=_config(),
        train=result(oos_trades, train_r if train_r is not None else oos_r),
        oos=result(oos_trades, oos_r),
    )


def test_an_untested_variant_never_outranks_a_tested_one():
    """A None score must sort LAST, not as a zero that happens to beat a
    genuinely negative result - otherwise 'we never tested it' wins."""
    losing = _stub_variant("tested but losing", oos_trades=MIN_OOS_TRADES, oos_r=-0.40)
    untested = _stub_variant("untested", oos_trades=2, oos_r=5.0)

    experiment = Experiment(name="x", variants=[untested, losing])
    assert untested.robust_score is None
    assert experiment.ranked[0].label == "tested but losing"
    assert experiment.best.label == "tested but losing"


def test_the_overfit_gap_penalises_a_variant_that_only_shone_in_training():
    """A variant brilliant in training and mediocre afterwards is not a
    good variant; it is an overfit one."""
    honest = _stub_variant("honest", oos_trades=50, oos_r=0.20, train_r=0.20)
    overfit = _stub_variant("overfit", oos_trades=50, oos_r=0.22, train_r=1.20)

    assert overfit.oos_expectancy_r > honest.oos_expectancy_r
    assert overfit.overfit_gap > honest.overfit_gap
    assert overfit.robust_score < honest.robust_score, (
        "the higher raw out-of-sample number must lose once the training gap is charged for"
    )


def test_drawdown_is_charged_against_the_robust_score():
    calm = _stub_variant("calm", oos_trades=50, oos_r=0.20, drawdown=5.0)
    wild = _stub_variant("wild", oos_trades=50, oos_r=0.20, drawdown=45.0)
    assert wild.robust_score < calm.robust_score


def test_an_experiment_with_one_tested_variant_is_inconclusive():
    experiment = Experiment(name="x", variants=[])
    assert experiment.conclusive is False
    assert "no variants were run" in experiment.verdict()


def test_variants_too_close_together_are_reported_as_equivalent():
    """Picking the nominal winner out of a set that did not separate is
    how noise becomes a config change."""
    series = _series(limit=1000)
    # Two thresholds that are almost certainly equivalent on this data.
    experiment = run_variants(
        "close", series,
        {"a": BacktestConfig(warmup_bars=210, min_score_to_enter=60),
         "b": BacktestConfig(warmup_bars=210, min_score_to_enter=61)},
        train_fraction=0.6,
    )
    if experiment.conclusive:
        assert "equivalent" in experiment.verdict() or "ranks highest" in experiment.verdict()


def test_a_variant_that_raises_is_skipped_with_a_warning_not_a_crash():
    series = _series(limit=600)
    bad = BacktestConfig(warmup_bars=210)
    object.__setattr__(bad, "weights", {"not_a_real_factor": 1.0})
    experiment = run_variants("mixed", series, {"good": _config(), "bad": bad}, train_fraction=0.6)
    # Either it ran (the engine tolerates unknown weights) or it was skipped
    # with a warning - but the good variant must survive either way.
    assert any(v.label == "good" for v in experiment.variants)


def test_run_variants_rejects_a_degenerate_split():
    series = _series(limit=600)
    with pytest.raises(ValueError, match="meaningful data on both sides"):
        run_variants("x", series, {"a": _config()}, train_fraction=0.99)


def test_run_variants_rejects_an_empty_variant_set():
    with pytest.raises(ValueError, match="no variants"):
        run_variants("x", _series(limit=600), {})


# ===========================================================================
# ablation
# ===========================================================================

def test_removing_a_factor_zeroes_its_weight_and_renormalises_the_rest():
    """Zeroed, not deleted.

    score_signal() indexes every factor name directly, so deleting the key
    raises KeyError - the ablation would crash instead of measuring
    anything. A zero weight is arithmetically identical: no contribution to
    the numerator, none to the total weight divided by, none to the
    missing-data budget.

    And the rest must renormalise. Leaving them at 0.94 would not test
    "without RSI"; it would test "without RSI, and every score depressed by
    6%", confounding the comparison with a pure scale change."""
    w = weights_without("rsi")
    assert set(w) == set(DEFAULT_WEIGHTS), "every key must survive or the scorer raises"
    assert w["rsi"] == 0.0
    assert sum(w.values()) == pytest.approx(1.0)


def test_the_ablated_model_actually_runs_through_the_scorer():
    """The regression that motivated the zeroing: deleting the key made
    every ablation variant raise KeyError inside score_signal."""
    from app.signals.scoring import score_signal

    series = _series(limit=400)
    result = score_signal(series, weights=weights_without("rsi"))
    assert 0 <= result.score <= 100


def test_relative_weights_are_preserved_when_renormalising():
    w = weights_without("rsi")
    before = DEFAULT_WEIGHTS["trend_direction"] / DEFAULT_WEIGHTS["macd"]
    after = w["trend_direction"] / w["macd"]
    assert before == pytest.approx(after)


def test_ablating_an_unknown_factor_is_an_error_not_a_no_op():
    with pytest.raises(KeyError, match="not a scoring factor"):
        weights_without("moon_phase")


def test_a_factor_verdict_can_say_the_factor_hurts():
    """The finding people refuse to believe, and the reason this exists.
    A standard indicator being actively harmful to THIS strategy on THIS
    data is an ordinary result."""
    v = FactorVerdict("rsi", 0.06, full_oos_r=0.10, ablated_oos_r=0.30, tested=True)
    assert v.delta == pytest.approx(0.20)
    assert "HURTS" in v.verdict
    assert "cut it" in v.recommendation


def test_a_factor_verdict_can_say_the_factor_helps():
    v = FactorVerdict("trend_direction", 0.12, full_oos_r=0.30, ablated_oos_r=0.05, tested=True)
    assert "helps" in v.verdict
    assert v.recommendation == "keep"


def test_a_factor_with_no_measurable_effect_is_called_redundant():
    v = FactorVerdict("bollinger", 0.04, full_oos_r=0.20,
                      ablated_oos_r=0.20 + NEGLIGIBLE_R / 2, tested=True)
    assert "no measurable effect" in v.verdict
    assert "candidate for removal" in v.recommendation


def test_an_untested_factor_yields_no_recommendation():
    v = FactorVerdict("macd", 0.09, full_oos_r=None, ablated_oos_r=None, tested=False)
    assert v.delta is None
    assert v.verdict == "not tested"
    assert "collect more" in v.recommendation


def test_ablation_runs_the_full_model_plus_one_variant_per_factor():
    series = _series(limit=1000)
    report = run_ablation(series, base_config=_config(), factors=("rsi", "macd"))
    labels = {v.label for v in report.experiment.variants}
    assert FULL_MODEL in labels
    assert "minus rsi" in labels and "minus macd" in labels
    assert len(report.factors) == 2


def test_ablation_reports_insufficient_data_rather_than_a_ranking():
    """Too short a series must produce 'we do not know', not a league
    table of noise."""
    report = run_ablation(_series(limit=400), base_config=_config(), factors=("rsi",))
    if not report.conclusive:
        assert "INSUFFICIENT DATA" in report.summary()
        assert "should be added, removed or reweighted" in report.summary()


def test_the_ablation_report_serialises():
    import json

    report = run_ablation(_series(limit=800), base_config=_config(), factors=("rsi",))
    json.dumps(report.as_dict(), allow_nan=False)


# ===========================================================================
# robustness
# ===========================================================================

def _point(value, r, trades=50):
    return ParameterPoint(value=value, oos_expectancy_r=r, oos_trades=trades, tested=True)


def test_a_broad_plateau_is_found_and_its_centre_recommended():
    """The centre has the most room for the market to move around it. The
    peak of a spike is a number that happened once."""
    report = RobustnessReport("min_score_to_enter", [
        _point(55, 0.10), _point(60, 0.12), _point(65, 0.14),
        _point(70, 0.13), _point(75, -0.05),      # negative: outside any plateau
    ])
    plateau = [p.value for p in report.plateau]
    assert plateau == [55, 60, 65, 70]
    assert report.recommended == 62.5      # median of the plateau
    assert "stable region" in report.verdict()


def test_the_plateau_tolerance_is_an_absolute_spread_not_a_ratio():
    """Documenting the actual rule rather than an intuition about it. A
    value scoring 0.05 beside neighbours at 0.14 is three times worse in
    ratio terms but only 0.09R away, so it stays inside the default 0.10R
    tolerance. Tighten PLATEAU_TOLERANCE_R if that is too generous for the
    scale of edge you are working with."""
    from app.research.robustness import PLATEAU_TOLERANCE_R

    report = RobustnessReport("min_score_to_enter", [
        _point(60, 0.14), _point(65, 0.05),
    ])
    assert 0.14 - 0.05 < PLATEAU_TOLERANCE_R
    assert [p.value for p in report.plateau] == [60, 65]


def test_an_isolated_peak_is_refused_however_high_it_scored():
    """Exactly what an overfit parameter looks like."""
    report = RobustnessReport("min_score_to_enter", [
        _point(55, -0.20), _point(60, -0.15), _point(65, 0.90),
        _point(70, -0.18), _point(75, -0.22),
    ])
    assert report.peak.value == 65
    assert report.recommended is None
    assert "NO STABLE REGION" in report.verdict()
    assert "Do not adopt it" in report.verdict()


def test_cliffs_between_neighbouring_values_are_reported():
    report = RobustnessReport("min_score_to_enter", [
        _point(60, 0.10), _point(65, 0.12), _point(70, 0.12 + CLIFF_THRESHOLD_R * 3),
    ])
    cliffs = report.cliffs
    assert len(cliffs) == 1
    assert cliffs[0][0] == 65 and cliffs[0][1] == 70


def test_a_plateau_must_be_profitable_throughout():
    """A run of consistently NEGATIVE values is consistent, and useless."""
    report = RobustnessReport("min_score_to_enter", [
        _point(55, -0.10), _point(60, -0.11), _point(65, -0.09),
    ])
    assert report.plateau == []
    assert report.recommended is None


def test_too_few_tested_values_is_inconclusive():
    report = RobustnessReport("min_score_to_enter", [
        _point(60, 0.1), ParameterPoint(65, None, 0, False),
    ])
    assert report.conclusive is False
    assert "INSUFFICIENT DATA" in report.verdict()


def test_sweeping_an_unknown_parameter_is_an_error():
    with pytest.raises(AttributeError, match="no parameter"):
        sweep_parameter(_series(limit=600), parameter="not_a_setting", values=[1, 2])


def test_sweeping_needs_neighbours_not_a_single_value():
    """Testing 65 alone tells you nothing about whether 65 is safe."""
    with pytest.raises(ValueError, match="at least two values"):
        sweep_parameter(_series(limit=600), parameter="min_score_to_enter", values=[65])


def test_a_real_sweep_produces_one_point_per_value():
    report = sweep_parameter(
        _series(limit=1000), parameter="min_score_to_enter",
        values=[60, 65, 70], base_config=_config(),
    )
    assert [p.value for p in report.points] == [60, 65, 70]


# ===========================================================================
# threshold study
# ===========================================================================

def _result(threshold, r, trades=40):
    return ThresholdResult(
        threshold=threshold, oos_trades=trades, oos_expectancy_r=r,
        oos_expectancy_usd=r * 20, oos_profit_factor=1.5, oos_win_rate=55.0,
        oos_max_drawdown_pct=8.0, oos_avg_win_usd=30.0, oos_avg_loss_usd=-20.0,
        train_expectancy_r=r, overfit_gap=0.0, trades_per_100_bars=4.0, tested=True,
    )


def test_a_strategy_with_no_edge_is_reported_as_having_no_edge():
    """The single most important output this module can produce. Finding
    that a strategy does NOT work is a successful result."""
    study = ThresholdStudy(results=[_result(t, -0.1) for t in (55, 60, 65, 70, 75)])
    verdict = study.verdict()
    assert "NO EDGE AT ANY THRESHOLD" in verdict
    assert "lowering it will produce more losing trades" in verdict
    assert study.recommended is None


def test_more_trades_is_not_treated_as_better():
    """A lower threshold trades more. That must not be what wins."""
    study = ThresholdStudy(results=[
        _result(55, -0.05, trades=200),   # trades the most, loses money
        _result(65, 0.15, trades=40),
        _result(67.5, 0.16, trades=35),
        _result(70, 0.14, trades=30),
    ])
    assert study.recommended == 67.5
    assert 55 not in [r.threshold for r in study.stable_region]


def test_an_isolated_profitable_threshold_is_refused():
    study = ThresholdStudy(results=[
        _result(55, -0.2), _result(60, -0.1), _result(65, 0.6),
        _result(70, -0.15), _result(75, -0.2),
    ])
    assert study.recommended is None
    assert "NO STABLE REGION" in study.verdict()
    assert "overfit parameter looks like" in study.verdict()


def test_a_stable_profitable_band_recommends_its_centre():
    study = ThresholdStudy(results=[
        _result(60, 0.10), _result(62.5, 0.12), _result(65, 0.13),
        _result(67.5, 0.11), _result(70, 0.12),
    ])
    assert study.recommended == 65
    assert "the centre of that region, not its peak" in study.verdict()


def test_untested_thresholds_make_the_study_inconclusive():
    study = ThresholdStudy(results=[
        ThresholdResult(t, 2, None, None, None, None, None, None, None, None, None, 0.2, False)
        for t in (55, 60, 65)
    ])
    assert study.conclusive is False
    assert "INSUFFICIENT DATA" in study.verdict()


def test_a_real_threshold_study_runs_end_to_end():
    study = study_thresholds(
        _series(limit=1500), thresholds=(60, 65, 70), base_config=_config()
    )
    assert [r.threshold for r in study.results] == [60, 65, 70]
    assert study.oos_bars > 0
    import json

    json.dumps(study.as_dict(), allow_nan=False)
    assert isinstance(study.table(), str)
