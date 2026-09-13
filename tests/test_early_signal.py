"""Tests for the Early Signal Engine.

Most of these are about what the engine REFUSES to do. A momentum detector
that only ever finds reasons to buy is worse than none, because it launders
enthusiasm into a number - so the properties worth pinning down are the
vetoes: security overrides everything, a finished move is rejected however
strong it looks, missing data is not a low score, and a collapsing token is
never "quiet".
"""
import datetime as dt

import pytest

from app.config import settings
from app.data.candles import Timeframe
from app.data.providers import SyntheticCandleProvider
from app.early import features as fm
from app.early.classifier import MomentumClass, classify
from app.early.engine import Decision, evaluate
from app.early.late_entry import Stage, assess
from app.early.score import DEFAULT_WEIGHTS, score_early_opportunity
from app.services.price_feed import MarketSnapshot

NOW = dt.datetime.now(dt.timezone.utc)


def series(regime="bull", seed=3, limit=300, timeframe=Timeframe.M5):
    return SyntheticCandleProvider(regime=regime, seed=seed).fetch("T", timeframe, limit=limit)


def snapshot(**overrides) -> MarketSnapshot:
    defaults = dict(
        price_usd=0.004, liquidity_usd=180_000.0, volume_24h_usd=250_000.0,
        buys_24h=600, sells_24h=420, price_change_1h_pct=5.0, price_change_24h_pct=20.0,
        pair_created_at=NOW - dt.timedelta(days=4), fdv_usd=900_000.0,
        token_address="MintEarly", volume_1h_usd=14_000.0, market_cap_usd=800_000.0,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


class Obs:
    """A stand-in for models.TokenObservation."""

    def __init__(self, minutes_ago, buys, sells, liquidity):
        self.observed_at = NOW - dt.timedelta(minutes=minutes_ago)
        self.buys_1h, self.sells_1h, self.liquidity_usd = buys, sells, liquidity


def rising_flow():
    return [Obs(12, 400, 300, 170_000.0), Obs(8, 440, 320, 174_000.0),
            Obs(4, 500, 350, 178_000.0), Obs(0, 580, 380, 182_000.0)]


def falling_flow():
    return [Obs(12, 600, 300, 190_000.0), Obs(8, 520, 380, 182_000.0),
            Obs(4, 460, 460, 172_000.0), Obs(0, 400, 540, 160_000.0)]


# ===========================================================================
# features: measurable vs honestly unavailable
# ===========================================================================

def test_wallet_level_features_are_reported_unavailable_not_approximated():
    """A transaction count is not a participant count. One wallet can make
    two hundred trades, and treating that as two hundred participants would
    invert the exact signal the feature exists to detect."""
    f = fm.extract(series=series(), market=snapshot(), observations=rising_flow())
    for name in ("unique_buyers", "new_buyers", "repeat_buyers", "wallet_concentration"):
        feature = f.get(name)
        assert feature.available is False
        assert feature.value is None
        assert feature.detail, f"{name} must say WHY it is unavailable"


def test_order_book_depth_is_not_pretended_to_exist():
    f = fm.extract(series=series(), market=snapshot(), observations=[])
    assert f.get("order_book_depth").available is False
    assert "no order book" in f.get("order_book_depth").detail


def test_flow_features_need_two_observations_far_enough_apart():
    """DexScreener reports transaction counts only over 1h/24h windows, so
    the rate of change has no source but differencing stored snapshots."""
    none_stored = fm.extract(series=series(), market=snapshot(), observations=[])
    assert none_stored.get("txn_rate_change").available is False
    assert "differenced" in none_stored.get("txn_rate_change").detail

    too_close = fm.extract(series=series(), market=snapshot(),
                           observations=[Obs(0.2, 400, 300, 1.0), Obs(0, 500, 320, 1.0)])
    assert too_close.get("txn_rate_change").available is False
    assert "rounding" in too_close.get("txn_rate_change").detail

    usable = fm.extract(series=series(), market=snapshot(), observations=rising_flow())
    assert usable.get("txn_rate_change").available is True


def test_buy_pressure_needs_enough_trades_to_mean_anything():
    """Three buys and one sell is not 75% buy pressure."""
    thin = fm.extract(series=series(), market=snapshot(),
                      observations=[Obs(10, 3, 1, 1.0), Obs(0, 4, 1, 1.0)])
    assert thin.get("buy_pressure").available is False


def test_persistence_needs_several_readings_not_one():
    """One observation of 70% buys is a moment; four in a row is a
    condition, and only the second is evidence."""
    two = fm.extract(series=series(), market=snapshot(),
                     observations=[Obs(8, 400, 300, 1.0), Obs(0, 500, 320, 1.0)])
    assert two.get("buy_pressure_persistence").available is False

    four = fm.extract(series=series(), market=snapshot(), observations=rising_flow())
    assert four.get("buy_pressure_persistence").available is True


def test_volume_steadiness_separates_a_climb_from_one_bar():
    f = fm.extract(series=series(), market=snapshot(), observations=[])
    steadiness = f.get("volume_steadiness")
    assert steadiness.available
    assert 0.0 <= steadiness.value <= 1.0


def test_no_candles_marks_every_candle_feature_unavailable():
    f = fm.extract(series=None, market=snapshot(), observations=rising_flow())
    for name in ("volume_accel_short", "rsi_level", "breakout_proximity", "relative_volume"):
        assert f.get(name).available is False
    # ...while snapshot and flow features still work.
    assert f.get("volume_to_liquidity").available is True
    assert f.get("buy_pressure").available is True


# ===========================================================================
# the score
# ===========================================================================

def test_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_missing_inputs_make_the_score_unreliable_rather_than_low():
    """A score built on nothing is not a bad score, it is no score - and a
    low-but-reliable-looking number would sail through a threshold."""
    result = score_early_opportunity(fm.extract(series=None, market=None, observations=[]))
    assert result.reliable is False
    assert result.unavailable
    assert any("unassessable, not average" in w for w in result.warnings)


def test_a_full_dataset_produces_a_reliable_score():
    result = score_early_opportunity(
        fm.extract(series=series(), market=snapshot(), observations=rising_flow())
    )
    assert result.reliable is True
    assert 0 <= result.score <= 100


def test_an_extreme_volume_spike_scores_worse_than_a_healthy_pickup():
    """40x is not 13 times better than 3x. It is a move already underway,
    and probably one actor."""
    from app.early.score import score_volume_acceleration

    class Stub(fm.EarlyFeatures):
        def __init__(self, short):
            super().__init__()
            self.add(fm.Feature("volume_accel_short", short, True, "", "candles"))
            self.add(fm.Feature("volume_accel_medium", short, True, "", "candles"))

    healthy = score_volume_acceleration(Stub(2.5), 0.18)
    exploding = score_volume_acceleration(Stub(25.0), 0.18)
    assert exploding.score < healthy.score
    assert "not early" in exploding.reason


def test_buy_pressure_that_does_not_persist_scores_lower():
    from app.early.score import score_buy_pressure

    def stub(pressure, persistence):
        f = fm.EarlyFeatures()
        f.add(fm.Feature("buy_pressure", pressure, True, "", "observations"))
        f.add(fm.Feature("buy_pressure_persistence", persistence, True, "", "observations"))
        return f

    sustained = score_buy_pressure(stub(0.65, 1.0), 0.14)
    fleeting = score_buy_pressure(stub(0.65, 0.2), 0.14)
    assert fleeting.score < sustained.score


def test_being_far_above_the_breakout_scores_worst_not_best():
    """The naive version of breakout detection rewards being far above the
    range, which is exactly the state of a move that already happened."""
    from app.early.score import score_breakout_position

    def stub(distance):
        f = fm.EarlyFeatures()
        f.add(fm.Feature("breakout_proximity", distance, True, "", "candles"))
        return f

    approaching = score_breakout_position(stub(-3.0), 0.08)
    at_it = score_breakout_position(stub(1.0), 0.08)
    long_gone = score_breakout_position(stub(45.0), 0.08)

    assert at_it.score >= approaching.score > long_gone.score
    assert "already happened" in long_gone.reason


def test_zeroing_a_weight_removes_a_factor_from_both_sides_of_the_ratio():
    """What makes leave-one-out ablation meaningful rather than a scale
    change."""
    f = fm.extract(series=series(), market=snapshot(), observations=rising_flow())
    weights = dict(DEFAULT_WEIGHTS)
    weights["relative_volume"] = 0.0
    ablated = score_early_opportunity(f, weights=weights)
    assert 0 <= ablated.score <= 100


# ===========================================================================
# late entry - the anti-chase system
# ===========================================================================

def test_a_finished_move_is_flagged_late_or_overextended():
    late = assess(fm.extract(series=series("pump"), market=snapshot(), observations=[]))
    assert late.stage in (Stage.LATE, Stage.OVEREXTENDED)
    assert late.blocking is True
    assert late.flags


def test_an_early_setup_is_not_flagged_late():
    early = assess(fm.extract(series=series("bull"), market=snapshot(), observations=[]))
    assert early.stage.enterable


def test_late_risk_is_separate_from_the_early_score():
    """A token can be genuinely strong AND already gone. Averaging the two
    into one number would hide exactly that case."""
    f = fm.extract(series=series("pump"), market=snapshot(), observations=rising_flow())
    early = score_early_opportunity(f)
    late = assess(f)
    assert early.score > 0
    assert late.risk > 0
    # They are different numbers answering different questions.
    assert early.score != late.risk


def test_no_price_history_means_late_not_early():
    """'We cannot tell where in the move this is' must not become an
    invitation to enter."""
    late = assess(fm.extract(series=None, market=snapshot(), observations=[]))
    assert late.assessable is False
    assert late.stage is Stage.LATE


# ===========================================================================
# the classifier
# ===========================================================================

def test_a_collapsing_token_is_never_classified_as_accumulation():
    """The bug this test exists for: `quiet` was a one-sided test, so a
    token down 63% satisfied 'price still quiet' and was classified
    ACCUMULATION at stage EARLY with zero late risk - it would have gone
    onto the watchlist as a promising candidate."""
    result = classify(fm.extract(series=series("crash"), market=snapshot(), observations=[]))
    assert result.label is MomentumClass.DISTRIBUTION
    assert result.label.preferred is False
    assert "exit volume" in result.reason


def test_a_healthy_trend_is_classified_rather_than_falling_through_to_unknown():
    """A trend at 15-60% travelled used to match no class at all, so
    `preferred` was always False and nothing could ever confirm."""
    result = classify(fm.extract(series=series("bull"), market=snapshot(), observations=[]))
    assert result.label is not MomentumClass.UNKNOWN
    assert result.label.preferred is True


def test_a_completed_pump_is_classified_late_not_preferred():
    result = classify(fm.extract(series=series("pump"), market=snapshot(), observations=[]))
    assert result.label in (MomentumClass.LATE_MOMENTUM, MomentumClass.PARABOLIC)
    assert result.label.preferred is False


def test_volume_concentrated_in_one_bar_is_suspicious():
    f = fm.EarlyFeatures()
    f.add(fm.Feature("volume_steadiness", 0.05, True, "", "candles"))
    f.add(fm.Feature("volume_accel_short", 4.0, True, "", "candles"))
    f.add(fm.Feature("return_long", 5.0, True, "", "candles"))
    assert classify(f).label is MomentumClass.SUSPICIOUS


def test_liquidity_leaving_into_a_rising_price_is_suspicious():
    f = fm.EarlyFeatures()
    f.add(fm.Feature("liquidity_growth", 0.7, True, "", "observations"))
    f.add(fm.Feature("return_short", 25.0, True, "", "candles"))
    f.add(fm.Feature("return_long", 30.0, True, "", "candles"))
    result = classify(f)
    assert result.label is MomentumClass.SUSPICIOUS
    assert "shape of an exit" in result.reason


def test_only_accumulation_and_breakout_are_preferred():
    assert MomentumClass.ACCUMULATION.preferred
    assert MomentumClass.BREAKOUT.preferred
    for other in (MomentumClass.LATE_MOMENTUM, MomentumClass.PARABOLIC,
                  MomentumClass.DISTRIBUTION, MomentumClass.SUSPICIOUS, MomentumClass.UNKNOWN):
        assert not other.preferred


# ===========================================================================
# the engine's decisions
# ===========================================================================

def test_a_security_failure_vetoes_the_best_possible_early_signal():
    """An excellent early signal must NEVER override a critical security
    failure - and the cheapest way to guarantee that is to never let the
    two meet."""
    verdict = evaluate(
        series=series("bull"), market=snapshot(), observations=rising_flow(),
        security_passed=False, security_reason="mint authority still active",
        technical_score=99.0,
    )
    assert verdict.decision is Decision.SKIP
    assert "no early signal overrides this" in verdict.reason
    assert verdict.early is None, "the early score must not even be computed"


def test_a_late_token_is_skipped_however_strong_the_score():
    verdict = evaluate(
        series=series("pump"), market=snapshot(), observations=rising_flow(),
        security_passed=True, technical_score=95.0,
    )
    assert verdict.decision is Decision.SKIP
    assert "too late to enter" in verdict.reason


def test_missing_data_is_skipped_rather_than_scored_low():
    verdict = evaluate(series=None, market=None, observations=[], security_passed=True)
    assert verdict.decision is Decision.SKIP
    assert "unreliable" in verdict.reason


def test_a_promising_but_unconfirmed_token_is_watched_not_discarded():
    """The whole point of the third state: without it, every promising
    candidate forces a choice between chasing and missing."""
    verdict = evaluate(
        series=series("bull"), market=snapshot(), observations=rising_flow(),
        security_passed=True, technical_score=20.0,     # technical says no
    )
    assert verdict.decision is Decision.WATCH
    assert "confirmation" in verdict.reason or "unconfirmed" in verdict.reason


def test_the_engine_cannot_trade_while_may_trade_is_false(monkeypatch):
    """The default posture, and the switch that separates research from
    trading."""
    monkeypatch.setattr(settings, "EARLY_SIGNAL_MAY_TRADE", False)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_REQUIRE_TECHNICAL", False)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_CONFIRM_THRESHOLD", 0.0)

    verdict = evaluate(
        series=series("bull"), market=snapshot(), observations=rising_flow(),
        security_passed=True, technical_score=90.0,
    )
    assert verdict.decision is not Decision.PAPER_BUY
    assert "EARLY_SIGNAL_MAY_TRADE is false" in verdict.reason
    assert "unvalidated priors" in verdict.reason
    # The note has to say what enabling it would require, not just that it
    # is off - otherwise the switch reads as an arbitrary obstacle.
    assert any("calibration" in n for n in verdict.notes)


def test_enabling_the_switch_allows_a_confirmed_candidate_through(monkeypatch):
    monkeypatch.setattr(settings, "EARLY_SIGNAL_MAY_TRADE", True)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_REQUIRE_TECHNICAL", False)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_CONFIRM_THRESHOLD", 0.0)

    verdict = evaluate(
        series=series("bull"), market=snapshot(), observations=rising_flow(),
        security_passed=True, technical_score=90.0,
    )
    assert verdict.decision is Decision.PAPER_BUY
    assert "CONFIRMED" in verdict.reason


def test_requiring_technical_confirmation_holds_a_strong_early_signal_at_watch(monkeypatch):
    """Combination strategy C: the early engine finds the candidate, the
    existing strategy confirms the entry."""
    monkeypatch.setattr(settings, "EARLY_SIGNAL_MAY_TRADE", True)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_REQUIRE_TECHNICAL", True)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_CONFIRM_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "MIN_SIGNAL_SCORE_TO_ENTER", 65.0)

    held = evaluate(series=series("bull"), market=snapshot(), observations=rising_flow(),
                    security_passed=True, technical_score=40.0)
    assert held.decision is Decision.WATCH
    assert "technical confirmation is not there yet" in held.reason

    through = evaluate(series=series("bull"), market=snapshot(), observations=rising_flow(),
                       security_passed=True, technical_score=80.0)
    assert through.decision is Decision.PAPER_BUY


def test_every_verdict_carries_all_four_scores_and_its_reasoning():
    verdict = evaluate(
        series=series("bull"), market=snapshot(), observations=rising_flow(),
        security_passed=True, security_score=22.0,
        technical_score=71.0, market_quality_score=68.0,
    )
    payload = verdict.as_dict()
    for key in ("early_score", "technical_score", "security_score",
                "market_quality_score", "late_entry_risk", "decision", "reason"):
        assert key in payload
    assert verdict.explain()

    import json
    json.dumps(payload, allow_nan=False)
