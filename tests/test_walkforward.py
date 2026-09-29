"""Walk-forward folds, the sealed holdout, and the ablation harness."""

from __future__ import annotations

import asyncio

import pytest
from conftest import make_candles

from tradebot.backtest.ablation import run_ablation
from tradebot.backtest.confirmers import (
    CachedConfirmer,
    NullConfirmer,
    RandomConfirmer,
    signal_key,
)
from tradebot.backtest.runner import Backtester, permissive_risk
from tradebot.backtest.walkforward import (
    HoldoutLedger,
    combine,
    develop,
    split_folds,
)
from tradebot.data.synthetic import SyntheticSource
from tradebot.models import AIVerdict, Decision, Side, TradeSignal


def bars(n=2000, seed=5):
    return SyntheticSource("TEST", "5m", seed=seed).history(n)


# ------------------------------------------------------------------ folds

def test_folds_are_consecutive_and_do_not_overlap():
    candles = make_candles([100 + i * 0.1 for i in range(2000)])
    folds = split_folds(candles, folds=4, test_fraction=0.4)

    assert len(folds) == 4
    for fold in folds:
        assert fold.train and fold.test
        assert fold.train[-1].ts < fold.test[0].ts      # test is strictly after train
    for a, b in zip(folds, folds[1:]):
        assert a.test[-1].ts < b.test[0].ts             # tests don't overlap


def test_anchored_folds_expand_the_training_window():
    candles = make_candles([100.0] * 2000)
    folds = split_folds(candles, folds=3, test_fraction=0.3, anchored=True)
    lengths = [len(f.train) for f in folds]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]
    assert all(f.train[0].ts == candles[0].ts for f in folds)


def test_rolling_folds_keep_a_constant_training_window():
    candles = make_candles([100.0] * 2000)
    folds = split_folds(candles, folds=3, test_fraction=0.3, anchored=False)
    assert len({len(f.train) for f in folds}) == 1


def test_too_little_history_is_refused_with_an_explanation():
    with pytest.raises(ValueError, match="not enough history"):
        split_folds(make_candles([100.0] * 320), folds=4, test_fraction=0.5)


# ---------------------------------------------------------------- holdout

def test_the_holdout_is_the_most_recent_tail():
    candles = make_candles([100.0] * 1000)
    development, holdout = develop(candles, 0.2)
    assert len(development) == 800 and len(holdout) == 200
    assert development[-1].ts < holdout[0].ts


def test_a_zero_fraction_holds_nothing_back():
    candles = make_candles([100.0] * 100)
    development, holdout = develop(candles, 0.0)
    assert len(development) == 100 and holdout == []


def test_the_ledger_records_every_look(tmp_path):
    ledger = HoldoutLedger.load(tmp_path / "ledger.json")
    assert ledger.runs == 0 and ledger.warning() == ""

    ledger.record("cfg-a", "data.csv", 500)
    assert HoldoutLedger.load(tmp_path / "ledger.json").runs == 1
    assert ledger.warning() == ""                   # one look is legitimate


def test_repeated_looks_with_the_same_config_are_benign(tmp_path):
    ledger = HoldoutLedger.load(tmp_path / "ledger.json")
    for _ in range(3):
        ledger.record("cfg-a", "data.csv", 500)
    assert "still out-of-sample" in ledger.warning()


def test_looking_with_different_configs_burns_the_holdout(tmp_path):
    ledger = HoldoutLedger.load(tmp_path / "ledger.json")
    ledger.record("cfg-a", "data.csv", 500)
    ledger.record("cfg-b", "data.csv", 500)

    warning = ledger.warning()
    assert "no longer a clean out-of-sample test" in warning
    assert ledger.distinct_configs == 2


def test_combining_folds_orders_trades_in_time(loose_config):
    results = [
        asyncio.run(Backtester(loose_config, NullConfirmer(), f"f{i}", f"t{i}")
                    .run_candles(chunk))
        for i, chunk in enumerate([bars(700, seed=1), bars(700, seed=2)])
    ]
    combined = combine(results, loose_config.risk.starting_equity)
    assert combined.trades == sum(r.metrics.trades for r in results)


# -------------------------------------------------------------- confirmers

def test_the_verdict_cache_is_keyed_by_the_decision():
    a = TradeSignal("X", "5m", Side.LONG, 100, 98, 104, 108, strategy="s")
    b = TradeSignal("X", "5m", Side.LONG, 100, 98, 104, 108, strategy="s")
    b.decision_ts = a.decision_ts = a.ts
    c = TradeSignal("X", "5m", Side.SHORT, 100, 102, 96, 92, strategy="s")
    c.decision_ts = a.ts
    assert signal_key(a) == signal_key(b)
    assert signal_key(a) != signal_key(c)


def test_cached_verdicts_are_reused_across_arms(tmp_path):
    class Inner:
        last_error = ""

        def __init__(self):
            self.calls = 0

        async def confirm(self, *_args, **_kwargs):
            self.calls += 1
            return AIVerdict(Decision.CONFIRM, 0.9, "ok")

        def accepts(self, verdict):
            return True, "ok"

    inner = Inner()
    cache = CachedConfirmer(inner, tmp_path / "verdicts.json")
    signal = TradeSignal("X", "5m", Side.LONG, 100, 98, 104, 108, strategy="s")

    asyncio.run(cache.confirm(signal, None, []))
    asyncio.run(cache.confirm(signal, None, []))
    assert inner.calls == 1 and cache.hits == 1

    # A fresh confirmer over the same file still hits.
    again = CachedConfirmer(inner, tmp_path / "verdicts.json")
    asyncio.run(again.confirm(signal, None, []))
    assert inner.calls == 1


def test_failures_are_never_cached(tmp_path):
    class Failing:
        last_error = "boom"

        async def confirm(self, *_args, **_kwargs):
            return AIVerdict(Decision.UNAVAILABLE, 0.0, "timed out")

        def accepts(self, verdict):
            return False, "unavailable"

    cache = CachedConfirmer(Failing(), tmp_path / "verdicts.json")
    signal = TradeSignal("X", "5m", Side.LONG, 100, 98, 104, 108, strategy="s")
    asyncio.run(cache.confirm(signal, None, []))
    assert cache.cache == {}


def test_the_random_confirmer_hits_its_target_rate():
    confirmer = RandomConfirmer(0.4, seed=1)
    for _ in range(2000):
        asyncio.run(confirmer.confirm())
    assert 0.36 < confirmer.vetoes / confirmer.calls < 0.44


def test_the_random_confirmer_is_deterministic_per_seed():
    a = RandomConfirmer(0.5, seed=7)
    b = RandomConfirmer(0.5, seed=7)
    for _ in range(50):
        asyncio.run(a.confirm())
        asyncio.run(b.confirm())
    assert a.vetoes == b.vetoes


# --------------------------------------------------------------- ablation

def test_permissive_risk_lifts_the_daily_caps(config):
    loose = permissive_risk(config)
    assert loose.risk.max_trades_per_day > 1000
    assert config.risk.max_trades_per_day == 5        # the original is untouched


def test_ablation_runs_every_arm(loose_config, tmp_path):
    report = asyncio.run(run_ablation(
        loose_config, bars(900), use_live_ai=False,
        cache_path=tmp_path / "verdicts.json", control_runs=2))
    assert [a.label for a in report.arms] == ["rules", "rules+ai", "rules+ai+risk"]
    assert len(report.random_arms) == 2


def test_ablation_says_so_when_there_are_no_ai_verdicts(loose_config, tmp_path):
    """Without verdicts the AI arms are the rules arm; claiming otherwise is a lie."""
    report = asyncio.run(run_ablation(
        loose_config, bars(900), use_live_ai=False,
        cache_path=tmp_path / "empty.json", control_runs=1))
    verdict = report.verdict()
    assert verdict.get("inconclusive")
    assert "meaningless" in verdict["conclusion"]


def test_a_forced_veto_rate_reaches_the_control_arms(loose_config, tmp_path):
    report = asyncio.run(run_ablation(
        loose_config, bars(1200), use_live_ai=False,
        cache_path=tmp_path / "v.json", control_runs=2, veto_rate=0.9))
    assert report.veto_rate == 0.9 and report.veto_rate_source == "forced"
    rules = report.by_label("rules")
    # Vetoing 90% at random must visibly reduce the trade count.
    assert all(a.metrics.trades < rules.metrics.trades for a in report.random_arms
               if rules.metrics.trades > 5)
