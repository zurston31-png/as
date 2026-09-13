"""Signal scoring and market regime classification.

The properties that matter here are not "does it return a number" but:
  - the score orders market regimes sensibly,
  - a high threshold is selective yet actually reachable,
  - missing data lands at neutral and is flagged, never scored as bullish,
  - every score can be explained factor by factor.
"""
import datetime as dt
import random

import pytest

from app.data.candles import Candle, CandleSeries, Timeframe
from app.data.providers import SyntheticCandleProvider
from app.signals.indicators import rsi
from app.signals.market_regime import (
    MarketCondition, TrendRegime, VolatilityRegime, classify, suits_strategy,
)
from app.signals.scoring import (
    DEFAULT_WEIGHTS, NEUTRAL, score_breakout, score_macd, score_momentum,
    score_relative_volume, score_rsi, score_signal, score_trend_direction,
)

UTC = dt.timezone.utc
START = dt.datetime(2026, 1, 1, tzinfo=UTC)


def synth(regime, n=300, seed=42, timeframe=Timeframe.H1):
    return SyntheticCandleProvider(regime, seed=seed).fetch("TEST", timeframe, n)


def bar(i, price, volume=100_000.0, spread=0.004):
    return Candle(
        START + dt.timedelta(hours=i),
        open=price * (1 - spread / 3), high=price * (1 + spread),
        low=price * (1 - spread), close=price, volume=volume,
    )


def ideal_breakout_series():
    """An uptrend with pullbacks, a squeeze, then a breakout on the last bar."""
    rng = random.Random(4)
    candles, price = [], 1.0
    for i in range(250):
        price *= 1.010 if (i % 7) < 5 else 0.988
        candles.append(bar(i, price, 100_000 * (0.8 + rng.random() * 0.4)))
    peak = price
    for i in range(250, 278):
        price = peak * (0.985 + rng.random() * 0.008)
        candles.append(bar(i, price, 70_000 * (0.8 + rng.random() * 0.4), spread=0.002))
    candles.append(bar(278, peak * 1.025, 400_000, spread=0.006))
    return CandleSeries("IDEAL", Timeframe.H1, candles)


# ---------------------------------------------------------------------------
# ordering and calibration
# ---------------------------------------------------------------------------

def test_score_orders_regimes_sensibly():
    scores = {r: score_signal(synth(r)).score for r in ("crash", "bear", "sideways", "bull")}
    assert scores["crash"] < scores["bear"] < scores["sideways"] < scores["bull"]


def test_bull_scores_above_neutral_and_crash_below():
    assert score_signal(synth("bull")).score > 55
    assert score_signal(synth("crash")).score < 45


def test_a_well_timed_breakout_clears_a_strict_threshold():
    """If a textbook setup cannot reach 75, the gate is unreachable and the
    bot would never trade."""
    assert score_signal(ideal_breakout_series()).score >= 75


def test_ordinary_conditions_do_not_clear_the_threshold():
    """The gate must reject the average case, or it is not a filter."""
    for regime in ("sideways", "bear", "crash", "high_volatility"):
        assert score_signal(synth(regime)).score < 75


def test_score_is_always_within_bounds():
    for regime in SyntheticCandleProvider.REGIMES:
        for seed in (1, 2, 3):
            score = score_signal(synth(regime, seed=seed)).score
            assert 0 <= score <= 100


def test_direction_follows_the_score():
    assert score_signal(ideal_breakout_series()).direction == "long"
    assert score_signal(synth("crash")).direction in ("short", "neutral")


# ---------------------------------------------------------------------------
# explainability
# ---------------------------------------------------------------------------

def test_every_factor_is_reported():
    result = score_signal(synth("bull"))
    assert len(result.factors) == len(DEFAULT_WEIGHTS)
    assert {f.name for f in result.factors} == set(DEFAULT_WEIGHTS)


def test_points_never_exceed_the_factor_weight():
    for f in score_signal(synth("bull")).factors:
        assert 0 <= f.points <= f.max_points + 1e-9


def test_contributions_sum_to_the_score():
    result = score_signal(synth("bull"))
    assert sum(f.points for f in result.factors) == pytest.approx(result.score, abs=0.01)


def test_every_factor_carries_a_reason():
    for f in score_signal(synth("bull")).factors:
        assert f.reason and len(f.reason) > 5


def test_breakdown_is_human_readable():
    text = score_signal(synth("bull")).breakdown()
    assert "Signal score" in text
    for name in DEFAULT_WEIGHTS:
        assert name in text


def test_supporting_and_opposing_split_correctly():
    result = score_signal(synth("bull"))
    assert all(f.score > 0.55 for f in result.supporting)
    assert all(f.score < 0.45 for f in result.opposing)


def test_as_dict_is_journal_ready():
    payload = score_signal(synth("bull")).as_dict()
    assert set(payload) >= {"score", "direction", "reliable", "factors"}
    assert len(payload["factors"]) == len(DEFAULT_WEIGHTS)
    assert set(payload["factors"][0]) >= {"name", "score", "weight", "points", "reason"}


# ---------------------------------------------------------------------------
# missing data must not read as bullish
# ---------------------------------------------------------------------------

def test_missing_data_scores_neutral_not_bullish():
    short = synth("bull", n=25)
    result = score_signal(short)
    for f in result.factors:
        if not f.available:
            assert f.score == NEUTRAL


def test_mostly_missing_data_marks_the_score_unreliable():
    """25 candles still supports RSI, EMA9/21, Bollinger and volume, so only
    ~21% of the weight is missing. It takes a genuinely short series to cross
    the 35% bar — which is the honest answer, not a reason to lower it."""
    result = score_signal(synth("bull", n=12))
    assert not result.reliable
    assert result.warnings
    assert "no data" in result.warnings[0]


def test_a_short_but_workable_series_stays_reliable():
    result = score_signal(synth("bull", n=25))
    assert result.reliable
    assert {f.name for f in result.unavailable} == {"macd", "multi_timeframe"}


def test_a_full_series_is_reliable():
    assert score_signal(synth("bull", n=300)).reliable


def test_unavailable_factors_are_listed():
    result = score_signal(synth("bull", n=25))
    assert result.unavailable
    assert all(not f.available for f in result.unavailable)


# ---------------------------------------------------------------------------
# individual factors
# ---------------------------------------------------------------------------

def test_rsi_factor_penalises_overbought():
    rising = [float(v) for v in range(1, 60)]          # RSI pinned at 100
    assert score_rsi(rising, 1.0).score < 0.3


def test_rsi_factor_rewards_healthy_momentum():
    """Mild net-upward drift lands RSI in the 55-65 band. Rising 2 bars in 3
    pins it near 75, which the scorer correctly calls extended instead."""
    closes = [100.0]
    for i in range(80):
        closes.append(closes[-1] * (1.004 if i % 2 else 0.9965))
    value = rsi(closes, 14)[-1]
    assert 50 <= value <= 70, f"fixture produced RSI {value:.1f}, outside the healthy band"
    assert score_rsi(closes, 1.0).score >= 0.6


def test_macd_factor_rewards_a_fresh_cross():
    closes = [100.0] * 40 + [100 * (1.01 ** i) for i in range(1, 15)]
    assert score_macd(closes, 1.0).score >= 0.8


def test_macd_factor_penalises_a_downtrend():
    assert score_macd([float(v) for v in range(100, 20, -1)], 1.0).score < 0.3


def test_relative_volume_factor_rewards_participation():
    quiet = [100.0] * 40
    busy = [100.0] * 39 + [400.0]
    assert score_relative_volume(busy, 1.0).score > score_relative_volume(quiet, 1.0).score


def test_breakout_factor_requires_closing_above_the_level():
    below = CandleSeries("T", Timeframe.H1, [bar(i, 100 + (i % 5)) for i in range(40)])
    assert score_breakout(below, 1.0).score < 0.5

    # A modest break. A 20% leap above the level scores lower on purpose —
    # that is chasing, not breaking out.
    candles = [bar(i, 100.0) for i in range(40)] + [bar(40, 103.0)]
    above = CandleSeries("T", Timeframe.H1, candles)
    assert score_breakout(above, 1.0).score > 0.8


def test_breakout_factor_penalises_chasing_an_extended_move():
    candles = [bar(i, 100.0) for i in range(40)] + [bar(40, 130.0)]
    extended = CandleSeries("T", Timeframe.H1, candles)
    factor = score_breakout(extended, 1.0)
    assert factor.score < 0.7
    assert "extended" in factor.reason


def test_momentum_factor_distinguishes_direction():
    up = [100 * (1.01 ** i) for i in range(30)]
    down = [100 * (0.99 ** i) for i in range(30)]
    assert score_momentum(up, 1.0).score > score_momentum(down, 1.0).score


def test_trend_direction_factor_reads_the_stack():
    assert score_trend_direction([float(v) for v in range(1, 400)], 1.0).score > 0.9
    assert score_trend_direction([float(v) for v in range(400, 1, -1)], 1.0).score < 0.1


def test_factors_on_empty_history_are_unavailable_not_crashing():
    for factor in (score_rsi([], 1.0), score_macd([], 1.0), score_momentum([], 1.0)):
        assert not factor.available
        assert factor.score == NEUTRAL


def test_weights_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        score_signal(synth("bull"), weights={k: 0.0 for k in DEFAULT_WEIGHTS})


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# market regime
# ---------------------------------------------------------------------------

def test_regime_identifies_trend_direction():
    assert classify(synth("bull")).trend is TrendRegime.BULL
    assert classify(synth("bear")).trend is TrendRegime.BEAR


def test_regime_identifies_volatility():
    assert classify(synth("high_volatility")).volatility is VolatilityRegime.HIGH
    assert classify(synth("low_volatility")).volatility is VolatilityRegime.LOW


def test_regime_axes_are_independent():
    """A market can be both trending and volatile; one label would lose that."""
    condition = classify(synth("crash"))
    assert condition.trend is TrendRegime.BEAR
    assert condition.volatility is VolatilityRegime.HIGH


def test_regime_is_unknown_without_enough_history():
    condition = classify(synth("bull", n=20))
    assert condition.trend is TrendRegime.UNKNOWN
    assert not condition.is_tradeable
    assert condition.notes


def test_regime_describe_is_informative():
    text = classify(synth("bull")).describe()
    assert "bull_trend" in text and "ATR" in text


def test_strategy_gate_allows_a_matching_regime():
    condition = classify(synth("bull"))
    allowed, reason = suits_strategy(condition, {"bull_trend"})
    assert allowed and "matches" in reason


def test_strategy_gate_blocks_a_mismatched_regime():
    condition = classify(synth("bear"))
    allowed, reason = suits_strategy(condition, {"bull_trend"})
    assert not allowed
    assert "bear_trend" in reason


def test_strategy_gate_blocks_an_unknown_regime():
    """An unclassifiable market must not be treated as permission to trade."""
    condition = classify(synth("bull", n=20))
    allowed, reason = suits_strategy(condition, {"bull_trend"})
    assert not allowed
    assert "could not be determined" in reason


def test_regime_agnostic_strategy_always_allowed():
    allowed, _ = suits_strategy(classify(synth("crash")), None)
    assert allowed


def test_volatility_regime_matches_on_either_axis():
    condition = classify(synth("high_volatility"))
    allowed, _ = suits_strategy(condition, {"high_volatility"})
    assert allowed
