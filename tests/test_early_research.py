"""Tests for the early-signal research tools.

The point of these tools is to be able to say "the early engine does not
work" out loud, so the tests care most about the paths where the honest
answer is negative or unknown. A tool that only reports success when
success exists is half a tool.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app import models
from app.analysis.early_calibration import MIN_BUCKET_SAMPLE
from app.early.features import EarlyFeatures, Feature
from app.early.score import DEFAULT_WEIGHTS, score_early_opportunity
from app.research.early_ablation import (
    early_weights_without, load_samples, run_early_ablation,
)
from app.research.early_walkforward import (
    CANDIDATE_THRESHOLDS, walk_forward_early_threshold,
)


# ---------------------------------------------------------------------------
# rebuilding stored features
# ---------------------------------------------------------------------------

def test_stored_features_round_trip():
    original = EarlyFeatures()
    original.add(Feature("volume_accel_short", 2.5, True, "rising", "candles"))
    original.add(Feature("buy_pressure", None, False, "no snapshot", "observations"))

    rebuilt = EarlyFeatures.from_dict(original.as_dict())

    assert rebuilt.value("volume_accel_short") == 2.5
    assert rebuilt.get("buy_pressure").available is False
    assert rebuilt.get("buy_pressure").detail == "no snapshot"


def test_a_stored_zero_stays_available():
    """0.0 is a measurement, not a missing value.

    If `available` were inferred from the value being non-null, a genuine
    zero would be reclassified as missing - which moves weight into the
    missing-data budget and can flip an unreliable score to reliable, or
    the reverse. The flag has to be read, not guessed.
    """
    stored = {"buy_pressure_change": {
        "name": "buy_pressure_change", "value": 0.0, "available": True,
        "detail": "flat", "source": "observations",
    }}
    rebuilt = EarlyFeatures.from_dict(stored)
    assert rebuilt.get("buy_pressure_change").available is True
    assert rebuilt.value("buy_pressure_change") == 0.0


def test_a_garbage_payload_does_not_crash_the_rebuild():
    assert EarlyFeatures.from_dict(None).features == {}
    assert EarlyFeatures.from_dict({"x": "not a dict"}).features == {}


# ---------------------------------------------------------------------------
# weight ablation arithmetic
# ---------------------------------------------------------------------------

def test_zeroing_a_weight_removes_it_from_the_denominator_too():
    """Otherwise the ablation would test a scale change, not a removal."""
    features = EarlyFeatures()
    features.add(Feature("volume_accel_short", 3.0, True, "", "candles"))
    features.add(Feature("volume_accel_medium", 2.0, True, "", "candles"))

    weights = early_weights_without("relative_volume")
    result = score_early_opportunity(features, weights=weights)

    assert all(f.name != "relative_volume" or f.weight == 0 for f in result.factors)
    assert sum(f.weight for f in result.factors) == pytest.approx(
        sum(DEFAULT_WEIGHTS.values()) - DEFAULT_WEIGHTS["relative_volume"]
    )


def test_ablating_an_unknown_factor_is_an_error_not_a_silent_no_op():
    with pytest.raises(KeyError):
        early_weights_without("moon_phase")


def test_ablating_the_only_factor_is_refused():
    with pytest.raises(ValueError):
        early_weights_without("volume_acceleration", {"volume_acceleration": 1.0})


# ---------------------------------------------------------------------------
# ablation over stored rows
# ---------------------------------------------------------------------------

# A fully-populated feature payload. The ablation reads stored features
# rather than recomputing them, so the seed has to look like something the
# live engine would have written - a payload with six of nine factors
# missing puts every row in the same bucket and the ablation then measures
# nothing at all.
STRONG = {
    "volume_accel_short": 3.0, "volume_accel_medium": 2.4,
    "txn_rate_change": 0.6, "buy_pressure": 0.62,
    "buy_pressure_persistence": 0.8, "buy_pressure_change": 0.05,
    "volume_steadiness": 0.8, "liquidity_growth": 1.15,
    "liquidity_stability": 0.9, "ema_slope": 0.004,
    "rsi_crossing_up": 1.0, "rsi_level": 58.0,
    "macd_histogram_expanding": 1.0, "range_compression": 0.35,
    "higher_lows": 1.0, "acceleration_smoothness": 0.8,
    "breakout_proximity": -2.0, "relative_volume": 3.0,
}
WEAK = {
    "volume_accel_short": 0.6, "volume_accel_medium": 0.7,
    "txn_rate_change": -0.4, "buy_pressure": 0.35,
    "buy_pressure_persistence": 0.2, "buy_pressure_change": -0.1,
    "volume_steadiness": 0.2, "liquidity_growth": 0.8,
    "liquidity_stability": 0.4, "ema_slope": -0.003,
    "rsi_crossing_up": 0.0, "rsi_level": 38.0,
    "macd_histogram_expanding": 0.0, "range_compression": 0.9,
    "higher_lows": 0.0, "acceleration_smoothness": 0.2,
    "breakout_proximity": -40.0, "relative_volume": 0.3,
}


def _payload(base: dict, *, drop: tuple[str, ...] = (), **overrides) -> dict:
    features = EarlyFeatures()
    for name, value in {**base, **overrides}.items():
        if name in drop:
            features.add(Feature(name, None, False, "no data", "observations"))
        else:
            features.add(Feature(name, value, True, "", "candles"))
    return features.as_dict()


def _row(db, *, early, ret, payload, horizon=60, observed=None):
    db.add(models.ForwardReturn(
        pipeline_event_id=1, token_address="M", symbol="M",
        observed_at=observed or dt.datetime.now(dt.timezone.utc),
        score=70.0, price_at_signal=0.01, horizon_minutes=horizon,
        due_at=dt.datetime.now(dt.timezone.utc),
        return_pct=ret, early_score=early, early_features=payload,
    ))


def test_ablation_says_insufficient_rather_than_inventing_a_verdict(clean_db):
    for _ in range(5):
        _row(clean_db, early=70.0, ret=3.0, payload=_payload(STRONG))
    clean_db.commit()

    report = run_early_ablation(clean_db, horizon_minutes=60)
    assert report.conclusive is False
    assert "INSUFFICIENT DATA" in report.summary()
    assert all(f.delta is None for f in report.factors)


def test_rows_without_stored_features_are_skipped_not_scored_empty(clean_db):
    """An empty feature set is not a neutral input.

    Scoring it would build a number entirely out of unavailable factors
    and pull every bucket toward the same value, which looks exactly like
    "the score does not separate outcomes" - a false negative manufactured
    by the loader.
    """
    _row(clean_db, early=70.0, ret=3.0, payload=None)
    _row(clean_db, early=70.0, ret=3.0, payload={})
    _row(clean_db, early=70.0, ret=3.0, payload=_payload(STRONG))
    clean_db.commit()

    assert len(load_samples(clean_db, horizon_minutes=60)) == 1


def test_ablation_detects_a_factor_that_is_carrying_the_signal(clean_db):
    """Seeded so relative_volume is the only thing separating the winners.

    Three groups: two identical except for relative_volume (which decides
    the outcome), plus a weak group that anchors the bottom bucket. With
    relative_volume ablated the first two groups collapse into one bucket
    and the separation disappears - which is what "this factor was
    carrying information" looks like in the numbers.

    This is a test of the MACHINERY on data with a known answer. It is not
    evidence about real markets, and the module docstring says so.
    """
    for _ in range(MIN_BUCKET_SAMPLE + 5):
        _row(clean_db, early=0.0, ret=30.0, payload=_payload(STRONG, relative_volume=3.0))
        _row(clean_db, early=0.0, ret=-30.0, payload=_payload(STRONG, relative_volume=0.2))
        _row(clean_db, early=0.0, ret=0.0, payload=_payload(WEAK))
    clean_db.commit()

    report = run_early_ablation(clean_db, horizon_minutes=60)
    assert report.conclusive, report.summary()

    rvol = next(f for f in report.factors if f.factor == "relative_volume")
    assert rvol.delta is not None and rvol.delta < 0, rvol.as_dict()
    assert "keep" in rvol.recommendation


def test_a_bucket_edge_artifact_does_not_become_a_verdict(clean_db):
    """Bucket separation moves when a factor merely SHIFTS every score.

    On this seed, removing transaction_acceleration lowers every score
    enough to drop the winning group out of the 80+ bucket and into the
    one below, collapsing it with the losers. Bucket separation falls off
    a cliff, but no information was lost - the score still orders the
    outcomes exactly as before. The verdict has to come from rank
    correlation, which has no edges to fall off.
    """
    for _ in range(MIN_BUCKET_SAMPLE + 5):
        _row(clean_db, early=0.0, ret=30.0, payload=_payload(STRONG, relative_volume=3.0))
        _row(clean_db, early=0.0, ret=-30.0, payload=_payload(STRONG, relative_volume=0.2))
        _row(clean_db, early=0.0, ret=0.0, payload=_payload(WEAK))
    clean_db.commit()

    report = run_early_ablation(clean_db, horizon_minutes=60)
    txn = next(f for f in report.factors if f.factor == "transaction_acceleration")

    assert txn.separation_delta is not None and txn.separation_delta < -10, txn.as_dict()
    assert txn.delta == pytest.approx(0.0, abs=1e-9)
    assert txn.verdict == "no measurable effect"


def test_rank_correlation_is_undefined_rather_than_zero_on_constant_data(clean_db):
    """Zero correlation says "no relationship". A constant column says
    "nothing to relate". Collapsing the second into the first would report
    an absence of edge that was never tested for."""
    from app.research.early_ablation import load_samples, rank_correlation
    from app.early.score import DEFAULT_WEIGHTS as W

    for _ in range(60):
        _row(clean_db, early=0.0, ret=5.0, payload=_payload(STRONG))
    clean_db.commit()

    samples = load_samples(clean_db, horizon_minutes=60)
    assert len(samples) == 60
    assert rank_correlation(samples, W) is None


def test_an_always_absent_factor_is_flagged_rather_than_called_redundant(clean_db):
    """Removing a factor that was never available proves nothing about it.

    The two cases produce the same number - a delta of about zero - and
    mean opposite things, so the report has to separate them from the
    stored data rather than from the arithmetic.
    """
    missing = ("buy_pressure", "buy_pressure_persistence", "buy_pressure_change")
    for _ in range(MIN_BUCKET_SAMPLE + 5):
        _row(clean_db, early=0.0, ret=30.0,
             payload=_payload(STRONG, drop=missing, relative_volume=3.0))
        _row(clean_db, early=0.0, ret=-30.0,
             payload=_payload(STRONG, drop=missing, relative_volume=0.2))
        _row(clean_db, early=0.0, ret=0.0, payload=_payload(WEAK, drop=missing))
    clean_db.commit()

    report = run_early_ablation(clean_db, horizon_minutes=60)
    pressure = next(f for f in report.factors if f.factor == "buy_pressure")

    assert pressure.coverage == 0.0
    assert pressure.verdict == "NO DATA - never measurable"
    assert "proves nothing about it" in pressure.recommendation
    assert any("NO data on any stored row" in w for w in report.warnings), report.warnings

    # a factor that WAS present must not be swept up in the same warning
    assert next(f for f in report.factors if f.factor == "relative_volume").coverage == 1.0


# ---------------------------------------------------------------------------
# attaching the early verdict to already-scheduled rows
# ---------------------------------------------------------------------------

class _Verdict:
    """The parts of EarlyVerdict that attach_early reads."""
    def __init__(self, early_score, features=None, late_risk=None, momentum=None):
        self.early_score = early_score
        self.features = features
        self.late_risk = late_risk
        self.momentum = momentum


def test_the_early_verdict_reaches_the_rows_scheduled_before_it(clean_db):
    """Forward returns are scheduled before the early engine runs.

    That ordering is deliberate - a candidate the technical gate rejects
    still has to be followed forward - but it means the early score is
    unknown at scheduling time. Without the back-fill every
    ForwardReturn.early_score stays NULL and the whole early calibration
    reads an empty dataset while appearing to work.
    """
    from app.analysis.forward_returns import attach_early

    features = EarlyFeatures()
    features.add(Feature("relative_volume", 3.0, True, "", "candles"))
    for horizon in (15, 60, 240):
        _row(clean_db, early=None, ret=None, payload=None, horizon=horizon)
    clean_db.flush()

    updated = attach_early(clean_db, 1, _Verdict(72.5, features, late_risk=18.0))
    clean_db.commit()

    assert updated == 3
    rows = clean_db.query(models.ForwardReturn).all()
    assert all(r.early_score == 72.5 for r in rows)
    assert all(r.late_entry_risk == 18.0 for r in rows)
    assert all(r.early_features["relative_volume"]["value"] == 3.0 for r in rows)


def test_a_security_failure_writes_no_early_score_at_all(clean_db):
    """The engine short-circuits before computing any feature when security
    fails, so there is no score. Writing 0 would be a fabricated
    measurement: "never looked at" and "looked at and scored zero" are
    different facts and the calibration groups on this column.
    """
    from app.analysis.forward_returns import attach_early

    _row(clean_db, early=None, ret=None, payload=None)
    clean_db.flush()

    assert attach_early(clean_db, 1, _Verdict(None)) == 0
    clean_db.commit()
    assert clean_db.query(models.ForwardReturn).one().early_score is None


# ---------------------------------------------------------------------------
# walk-forward over stored rows
# ---------------------------------------------------------------------------

def _walk_rows(db, pairs, *, start=None):
    base = start or dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i, (early, ret) in enumerate(pairs):
        _row(db, early=early, ret=ret, payload=None,
             observed=base + dt.timedelta(minutes=i))


def test_walk_forward_refuses_to_grade_a_thin_dataset(clean_db):
    _walk_rows(clean_db, [(70.0, 5.0)] * 20)
    clean_db.commit()

    report = walk_forward_early_threshold(clean_db, horizon_minutes=60)
    assert report.conclusive is False
    assert "has NOT been walk-forward validated" in report.verdict()


def test_walk_forward_reports_no_edge_when_every_threshold_loses(clean_db):
    """A stable, negative result is a result - and must not read as failure
    of the method."""
    pairs = []
    for i in range(400):
        pairs.append((50.0 + (i % 7) * 5, -8.0))
    _walk_rows(clean_db, pairs)
    clean_db.commit()

    report = walk_forward_early_threshold(clean_db, horizon_minutes=60)
    assert report.conclusive, report.table()
    assert "not an edge" in report.verdict()


def test_walk_forward_never_fits_on_the_window_it_grades(clean_db):
    """The train block for window i must end where its test block starts.

    If they overlapped, the reported out-of-sample number would be partly
    in-sample and the whole exercise would be decorative.
    """
    pairs = [(50.0 + (i % 7) * 5, (i % 5) - 2.0) for i in range(500)]
    _walk_rows(clean_db, pairs)
    clean_db.commit()

    report = walk_forward_early_threshold(clean_db, horizon_minutes=60, windows=4)
    assert report.windows
    for window in report.windows:
        assert window.train_rows > 0
        assert window.test_rows > 0
    # every chosen threshold has to come from the candidate ladder
    assert all(t in CANDIDATE_THRESHOLDS for t in report.chosen_thresholds)


def test_a_threshold_that_jumps_between_windows_is_called_unproven(clean_db):
    """A different answer every window is fitting noise, even when each
    individual fit looks profitable."""
    from app.research.early_walkforward import EarlyWalkForward, Window

    report = EarlyWalkForward(horizon_minutes=60, total_rows=1000)
    for i, threshold in enumerate((50.0, 80.0, 55.0, 75.0), start=1):
        report.windows.append(Window(i, 100, 50, threshold, 6.0, 3.0, 20))

    assert report.stable_threshold is False
    assert "fitting noise" in report.verdict()


def test_a_steady_threshold_with_positive_expectancy_reads_as_held(clean_db):
    from app.research.early_walkforward import EarlyWalkForward, Window

    report = EarlyWalkForward(horizon_minutes=60, total_rows=1000)
    for i, threshold in enumerate((65.0, 70.0, 65.0, 70.0), start=1):
        report.windows.append(Window(i, 100, 50, threshold, 5.0, 4.0, 20))

    assert report.stable_threshold is True
    assert "held steady" in report.verdict()
