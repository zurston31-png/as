"""Tests for the evidence report and regime persistence.

The report's job is to distinguish "we measured a bad strategy" from "we
have not measured anything", because those look identical in a table of
zeroes and lead to opposite decisions.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.evidence import MIN_FOR_A_NUMBER, MIN_PER_REGIME, build_evidence_report
from app.signals.market_regime import (
    DEEP_LIQUIDITY_USD, THIN_LIQUIDITY_USD, LiquidityRegime,
    classify_full, classify_liquidity,
)

NOW = dt.datetime.now(dt.timezone.utc)


def _row(db, *, event=1, ret=5.0, regime="bull_trend/normal_volatility/deep_liquidity",
         mfe=None, mae=None, horizon=60):
    observed = NOW - dt.timedelta(hours=3)
    db.add(models.ForwardReturn(
        pipeline_event_id=event, token_address=f"M{event}", symbol="T",
        observed_at=observed, score=70.0, price_at_signal=0.01,
        horizon_minutes=horizon, due_at=observed + dt.timedelta(minutes=horizon),
        return_pct=ret, market_regime=regime,
        max_favorable_pct=mfe, max_adverse_pct=mae,
    ))


# ---------------------------------------------------------------------------
# empty is not the same as bad
# ---------------------------------------------------------------------------

def test_an_empty_dataset_says_so_instead_of_printing_zeroes(clean_db):
    """"expectancy 0.00R over 0 trades" reads as a measurement of a bad
    strategy. It is the absence of a measurement of any strategy."""
    report = build_evidence_report(clean_db)

    assert report.completed_samples == 0
    assert report.promotion_ready is False
    assert "no completed observations" in report.readiness()

    expectancy = next(m for m in report.measures if m.label == "expectancy (net)")
    assert expectancy.value is None
    assert "not measurable" in expectancy.render()


def test_every_measure_carries_its_sample_size(clean_db):
    for i in range(5):
        _row(clean_db, event=i, ret=4.0)
    clean_db.commit()

    report = build_evidence_report(clean_db)
    for measure in report.measures:
        assert "n=" in measure.render()


def test_a_statistic_below_its_floor_is_marked_not_hidden(clean_db):
    """Hiding it would lose the fact that some data exists; showing it
    plain would let it be read as evidence."""
    for i in range(MIN_FOR_A_NUMBER - 5):
        _row(clean_db, event=i, ret=4.0)
    clean_db.commit()

    expectancy = next(
        m for m in build_evidence_report(clean_db).measures if m.label == "expectancy (net)"
    )
    assert expectancy.value is not None
    assert expectancy.trustworthy is False
    assert "below floor" in expectancy.render()


def test_costs_are_subtracted_from_expectancy(clean_db):
    """A +1% gross average is a loss after a ~2.3% round trip."""
    for i in range(MIN_FOR_A_NUMBER + 5):
        _row(clean_db, event=i, ret=1.0)
    clean_db.commit()

    expectancy = next(
        m for m in build_evidence_report(clean_db).measures if m.label == "expectancy (net)"
    )
    assert expectancy.value < 0


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------

def test_regimes_are_grouped_one_axis_at_a_time(clean_db):
    """The full cross product is 36 cells. This bot will never fill them,
    and slicing that finely produces cells of three trades under a table
    that looks authoritative."""
    for i in range(20):
        _row(clean_db, event=i, ret=5.0,
             regime="bull_trend/high_volatility/low_liquidity")
    clean_db.commit()

    by_regime = build_evidence_report(clean_db).by_regime
    assert set(by_regime) == {"bull_trend", "high_volatility", "low_liquidity"}
    assert by_regime["bull_trend"].samples == 20


def test_one_regime_is_not_enough_to_start_drawing_conclusions(clean_db):
    """Everything measured in one market condition cannot distinguish an
    edge from a bet on that condition continuing."""
    for i in range(MIN_FOR_A_NUMBER + 20):
        _row(clean_db, event=i, ret=8.0,
             regime="bull_trend/normal_volatility/deep_liquidity")
    clean_db.commit()

    report = build_evidence_report(clean_db)
    assert report.completed_samples >= MIN_FOR_A_NUMBER
    # all three axes are single-valued, so no second CONDITION exists
    assert "different market condition" in report.next_experiment


def test_a_missing_regime_is_named_as_a_weakness(clean_db):
    for i in range(5):
        _row(clean_db, event=i, ret=3.0, regime=None)
    clean_db.commit()

    weaknesses = build_evidence_report(clean_db).weaknesses
    assert any("market regime" in w.lower() for w in weaknesses)


def test_thin_regimes_are_named(clean_db):
    for i in range(MIN_PER_REGIME + 10):
        _row(clean_db, event=i, ret=5.0, regime="bull_trend/normal_volatility/deep_liquidity")
    for i in range(3):
        _row(clean_db, event=500 + i, ret=5.0, regime="bear_trend/high_volatility/low_liquidity")
    clean_db.commit()

    weaknesses = build_evidence_report(clean_db).weaknesses
    assert any("below the" in w and "floor" in w for w in weaknesses)


# ---------------------------------------------------------------------------
# integrity feeds the report
# ---------------------------------------------------------------------------

def test_corrupt_rows_are_excluded_before_anything_is_averaged(clean_db):
    for i in range(MIN_FOR_A_NUMBER + 5):
        _row(clean_db, event=i, ret=5.0)
    _row(clean_db, event=1, ret=5.0)                    # duplicate
    _row(clean_db, event=999, ret=5_000_000.0)          # impossible
    clean_db.commit()

    report = build_evidence_report(clean_db)
    expectancy = next(m for m in report.measures if m.label == "expectancy (net)")
    assert expectancy.value == pytest.approx(5.0 - 2.3, abs=0.01), (
        "an impossible move leaked into the average"
    )


def test_a_heavily_corrupted_dataset_blocks_readiness(clean_db):
    for i in range(MIN_FOR_A_NUMBER + 10):
        _row(clean_db, event=i, ret=5.0)
    for i in range(20):
        _row(clean_db, event=i, ret=5.0)                # all duplicates
    clean_db.commit()

    report = build_evidence_report(clean_db)
    assert report.promotion_ready is False


def test_the_report_never_claims_readiness_without_two_regimes(clean_db):
    for i in range(200):
        _row(clean_db, event=i, ret=9.0,
             regime="bull_trend/normal_volatility/deep_liquidity")
    clean_db.commit()

    assert build_evidence_report(clean_db).promotion_ready is False


# ---------------------------------------------------------------------------
# the liquidity axis
# ---------------------------------------------------------------------------

def test_liquidity_bands_are_classified_from_depth():
    assert classify_liquidity(THIN_LIQUIDITY_USD - 1) is LiquidityRegime.THIN
    assert classify_liquidity(DEEP_LIQUIDITY_USD + 1) is LiquidityRegime.DEEP
    assert classify_liquidity((THIN_LIQUIDITY_USD + DEEP_LIQUIDITY_USD) / 2) is LiquidityRegime.MODERATE


def test_missing_depth_is_unknown_not_inferred_from_volume():
    """A token can trade heavily in a pool about to be drained. Treating
    turnover as depth is how a thin pool passes for a deep one."""
    assert classify_liquidity(None) is LiquidityRegime.UNKNOWN
    assert classify_liquidity(0) is LiquidityRegime.UNKNOWN


def test_a_regime_records_the_features_it_was_judged_from():
    """A bare label cannot be audited: "sideways" six weeks ago is not
    reviewable unless the numbers behind it sit next to it."""
    condition = classify_full(None, liquidity_usd=9_000.0)
    features = condition.features()

    assert features["liquidity"] == "low_liquidity"
    assert features["liquidity_usd"] == 9_000.0
    assert features["thin_below_usd"] == THIN_LIQUIDITY_USD
    assert "notes" in features


def test_no_candles_means_unknown_trend_rather_than_a_guess():
    condition = classify_full(None, liquidity_usd=500_000.0)
    assert condition.trend.value == "unknown"
    assert condition.liquidity is LiquidityRegime.DEEP
    assert "unassessable" in " ".join(condition.notes)


def test_two_genuinely_different_conditions_do_unlock_readiness():
    """The bar has to be passable, or it is just an off switch."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        for row in db.query(models.ForwardReturn).all():
            db.delete(row)
        db.commit()
        for i in range(MIN_PER_REGIME + 5):
            _row(db, event=i, ret=6.0, regime="bull_trend/normal_volatility/deep_liquidity")
        for i in range(MIN_PER_REGIME + 5):
            _row(db, event=900 + i, ret=2.0, regime="bear_trend/normal_volatility/deep_liquidity")
        db.commit()

        report = build_evidence_report(db)
        assert "trend" in report.contrasting_axes
        assert report.promotion_ready is True
        assert "READY to begin drawing conclusions" in report.readiness()
        assert "not a verdict that the strategy works" in report.readiness()
    finally:
        for row in db.query(models.ForwardReturn).all():
            db.delete(row)
        db.commit()
        db.close()


def test_one_condition_expanding_to_three_labels_is_not_three_regimes():
    """The bug this guards: bull/normal/deep produces three axis labels,
    and counting those as three regimes would let a single market
    condition satisfy a bar that exists to require more than one."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        for row in db.query(models.ForwardReturn).all():
            db.delete(row)
        db.commit()
        for i in range(MIN_FOR_A_NUMBER + 40):
            _row(db, event=i, ret=7.0, regime="bull_trend/normal_volatility/deep_liquidity")
        db.commit()

        report = build_evidence_report(db)
        assert len(report.by_regime) == 3, "three axis labels, as expected"
        assert report.contrasting_axes == [], "but zero axes with contrast"
        assert report.promotion_ready is False
    finally:
        for row in db.query(models.ForwardReturn).all():
            db.delete(row)
        db.commit()
        db.close()
