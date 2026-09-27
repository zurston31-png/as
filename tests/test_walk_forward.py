"""Tests for app/backtesting/walk_forward.py.

The property that actually matters here is that config SELECTION happens
from train data only - validation and out-of-sample results must never be
able to influence which candidate wins. test_selection_uses_only_train_score
and test_out_of_sample_data_cannot_influence_the_chosen_config both check
that directly rather than just trusting the implementation.
"""
import datetime as dt

import pytest

from app.backtesting.types import BacktestConfig
from app.backtesting.walk_forward import run_walk_forward
from app.data.candles import Candle, CandleSeries, Timeframe
from app.data.providers import SyntheticCandleProvider

WARMUP = 210


def _series(regime: str, seed: int = 1, limit: int = 800):
    return SyntheticCandleProvider(regime=regime, seed=seed).fetch("TESTCOIN", Timeframe.M15, limit=limit)


def _concat(a: CandleSeries, b: CandleSeries) -> CandleSeries:
    """Chronologically join two series (same timeframe), continuing straight
    on from `a`'s last timestamp - used to build a series with a clean
    regime change partway through, so a specific window can be forced to
    fall on one side of it."""
    interval = dt.timedelta(seconds=a.timeframe.seconds)
    offset = a.candles[-1].timestamp + interval - b.candles[0].timestamp
    shifted = [
        Candle(timestamp=c.timestamp + offset, open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume)
        for c in b.candles
    ]
    return CandleSeries(a.symbol, a.timeframe, a.candles + shifted)


def test_split_windows_are_chronological_and_non_overlapping():
    series = _series("bull")
    result = run_walk_forward(series, BacktestConfig(warmup_bars=WARMUP))
    # Every trade in an earlier window must have happened before every
    # trade in a later window - proof the windows didn't overlap.
    if result.train.trades and result.validation.trades:
        assert result.train.trades[-1].exit_time <= result.validation.trades[0].entry_time
    if result.validation.trades and result.out_of_sample.trades:
        assert result.validation.trades[-1].exit_time <= result.out_of_sample.trades[0].entry_time


def test_single_config_shorthand_produces_one_candidate():
    series = _series("bull")
    config = BacktestConfig(warmup_bars=WARMUP)
    result = run_walk_forward(series, config)
    assert len(result.candidates) == 1
    assert result.chosen_config is config


def test_selection_uses_only_train_score():
    """The chosen candidate must be exactly the one with the highest
    train_score - if selection ever started reading validation/out-of-sample
    performance, this is the check that would catch it."""
    series = _series("bull")
    configs = {
        "loose": BacktestConfig(warmup_bars=WARMUP, min_score_to_enter=60.0),
        "strict": BacktestConfig(warmup_bars=WARMUP, min_score_to_enter=90.0),
    }
    result = run_walk_forward(series, configs, selection_metric="expectancy_r")
    best_by_train = max(result.candidates, key=lambda c: c.train_score)
    assert result.chosen_label == best_by_train.label
    assert result.chosen_config is best_by_train.config


def test_dict_labels_are_preserved_on_every_candidate():
    series = _series("bull")
    configs = {"a": BacktestConfig(warmup_bars=WARMUP), "b": BacktestConfig(warmup_bars=WARMUP, fee_pct=0.05)}
    result = run_walk_forward(series, configs)
    assert {c.label for c in result.candidates} == {"a", "b"}


def test_list_of_configs_gets_generated_labels():
    series = _series("bull")
    configs = [BacktestConfig(warmup_bars=WARMUP), BacktestConfig(warmup_bars=WARMUP, fee_pct=0.05)]
    result = run_walk_forward(series, configs)
    assert {c.label for c in result.candidates} == {"candidate_0", "candidate_1"}


def test_too_few_train_trades_produces_a_warning():
    # An absurdly strict score threshold guarantees near-zero trades.
    series = _series("bull")
    result = run_walk_forward(
        series, BacktestConfig(warmup_bars=WARMUP, min_score_to_enter=99.9), min_train_trades=5,
    )
    assert any("not statistically meaningful" in w for w in result.warnings)


def test_overfitting_warning_fires_when_validation_regime_flips():
    """Build a series that is pure bull for train and pure bear for
    validation - a config tuned to look great on the bull training window
    should show a clear train-to-validation degradation."""
    bull = _series("bull", limit=430)
    bear = _series("bear", seed=2, limit=430)
    series = _concat(bull, bear)
    result = run_walk_forward(series, BacktestConfig(warmup_bars=WARMUP))
    assert any("overfitting" in w for w in result.warnings)


def test_out_of_sample_data_cannot_influence_the_chosen_config():
    """Same idea as the overfitting test, from the other direction: even
    though the second half of this series is a regime the 'loose' config
    would do far worse in, selection (train-only, all in the bull half)
    must still be decided before out-of-sample is ever backtested."""
    bull = _series("bull", limit=430)
    bear = _series("bear", seed=2, limit=430)
    series = _concat(bull, bear)
    configs = {
        "loose": BacktestConfig(warmup_bars=WARMUP, min_score_to_enter=60.0),
        "strict": BacktestConfig(warmup_bars=WARMUP, min_score_to_enter=95.0),
    }
    result = run_walk_forward(series, configs)
    best_by_train = max(result.candidates, key=lambda c: c.train_score)
    assert result.chosen_label == best_by_train.label


def test_metric_value_treats_none_as_worst_not_a_crash():
    """A candidate with zero trades has expectancy_r=None - it must never
    win a comparison against a candidate that actually has a positive
    score, and must never raise from comparing None to a float."""
    series = _series("bear")  # regime filter -> 0 trades for any config here
    result = run_walk_forward(series, BacktestConfig(warmup_bars=WARMUP))
    assert result.train.stats.trade_count == 0
    assert result.candidates[0].train_score == float("-inf")
