"""Rule ablation: which rule removes the setups, and does removing it help?"""

from __future__ import annotations

import asyncio

import pytest

from tradebot.backtest.metrics import compute
from tradebot.backtest.ruleablation import (
    build_variants,
    expand,
    run_rule_ablation,
    with_rules,
)
from tradebot.data.synthetic import SyntheticSource
from tradebot.models import ClosedTrade, Side, utcnow
from tradebot.strategy.presets import PRESETS
from tradebot.strategy.spec import StrategySpec


def bars(n=1200, seed=5):
    return SyntheticSource("TEST", "5m", seed=seed).history(n)


def spec_of(rules):
    return StrategySpec.from_config({"name": "t", "rules": rules}, PRESETS)


# --------------------------------------------------------------- variants

def test_a_preset_is_expanded_into_explicit_rules(config):
    config.strategy = {"preset": "ema_vwap_rsi"}
    spec = expand(config)
    assert len(spec.active()) > 4
    assert all(isinstance(r.rule, str) for r in spec.active())


def test_incremental_variants_stack_up_in_order(config):
    spec = spec_of(["ema_stack", "vwap_position", "rsi_window"])
    _, incremental, _ = build_variants(spec, "incremental")

    assert [len(v.rules) for v in incremental] == [1, 2, 3]
    assert [v.added for v in incremental] == ["ema_stack", "vwap_position", "rsi_window"]
    assert [r["rule"] for r in incremental[-1].rules] == [
        "ema_stack", "vwap_position", "rsi_window"]


def test_leave_one_out_drops_exactly_one_required_rule(config):
    spec = spec_of(["ema_stack", "vwap_position", "rsi_window"])
    baseline, _, loo = build_variants(spec, "leave-one-out")

    assert len(baseline.rules) == 3
    assert {v.removed for v in loo} == {"ema_stack", "vwap_position", "rsi_window"}
    for variant in loo:
        assert len(variant.rules) == 2
        assert variant.removed not in [r["rule"] for r in variant.rules]


def test_advisory_rules_are_not_dropped_by_leave_one_out(config):
    """They aren't blocking anything, so removing them answers nothing."""
    spec = spec_of([
        {"rule": "ema_stack", "mode": "required"},
        {"rule": "volume_confirmation", "mode": "advisory"},
    ])
    _, _, loo = build_variants(spec, "leave-one-out")
    assert [v.removed for v in loo] == ["ema_stack"]


def test_every_variant_keeps_at_least_one_required_rule(config):
    """A strategy with only advisory rules is rejected by the spec."""
    spec = spec_of([
        {"rule": "ema_stack", "mode": "required"},
        {"rule": "volume_confirmation", "mode": "advisory"},
    ])
    _, incremental, loo = build_variants(spec, "both")
    for variant in [*incremental, *loo]:
        assert any(r["mode"] == "required" for r in variant.rules)
        StrategySpec.from_config({"rules": variant.rules}, PRESETS)   # must parse


def test_with_rules_does_not_mutate_the_original(config):
    config.strategy = {"preset": "ema_vwap_rsi", "min_score": 0.6}
    variant = with_rules(config, [{"rule": "ema_stack", "mode": "required"}], "v")
    assert config.strategy == {"preset": "ema_vwap_rsi", "min_score": 0.6}
    assert variant.strategy["rules"][0]["rule"] == "ema_stack"
    assert variant.strategy["min_score"] == 0.6        # other settings carry over
    assert "preset" not in variant.strategy            # or it would win over rules


# ----------------------------------------------------------------- report

def test_ablation_runs_every_variant(loose_config):
    report = asyncio.run(run_rule_ablation(loose_config, bars(900), "both"))
    assert report.baseline.result is not None
    assert all(v.result is not None for v in report.incremental)
    assert all(v.result is not None for v in report.leave_one_out)
    assert report.bars == 900


def test_dropping_a_rule_never_reduces_the_qualifying_count(loose_config):
    """Fewer required conditions can only let more bars qualify."""
    report = asyncio.run(run_rule_ablation(loose_config, bars(1500), "leave-one-out"))
    base = report.baseline.result.qualified
    for variant in report.leave_one_out:
        assert variant.result.qualified >= base, variant.removed


def test_admitted_signals_are_not_monotonic_and_that_is_expected(loose_config):
    """The gates couple decisions across time, so `signals` can fall as rules relax.

    This is exactly why the report judges strictness on qualifying bars: an
    earlier entry occupies the book and suppresses later signals, so a looser
    rule set can admit fewer of them.
    """
    report = asyncio.run(run_rule_ablation(loose_config, bars(1500), "leave-one-out"))
    base_signals = len(report.baseline.result.signals)
    counts = [len(v.result.signals) for v in report.leave_one_out]
    # Not asserting a direction - only that the report never claims one.
    assert counts and base_signals >= 0
    for f in report.findings():
        assert f["setups_it_removes"] == f["qualified_without_it"] - f["qualified_with_it"]


def test_findings_attribute_setups_to_the_rule_that_removes_them(loose_config):
    report = asyncio.run(run_rule_ablation(loose_config, bars(1500), "leave-one-out"))
    findings = report.findings()
    assert findings
    for f in findings:
        assert f["setups_it_removes"] == f["qualified_without_it"] - f["qualified_with_it"]
        assert f["setups_it_removes"] >= 0
    # Sorted most-restrictive first.
    assert findings == sorted(findings, key=lambda f: -f["setups_it_removes"])


def test_thin_samples_are_flagged_rather_than_reported_as_conclusions(loose_config):
    report = asyncio.run(run_rule_ablation(loose_config, bars(700), "leave-one-out"))
    for f in report.findings():
        if report.baseline.result.metrics.trades < 30:
            assert "hint, not a result" in f["reading"]


def test_the_report_renders(loose_config):
    report = asyncio.run(run_rule_ablation(loose_config, bars(900), "both"))
    text = report.render()
    assert "Incremental" in text and "Leave-one-out" in text
    assert "expectancy" in text


# ---------------------------------------------------------------- metrics

def trade(exit_reason: str, pnl: float = 100.0, r: float = 2.0) -> ClosedTrade:
    now = utcnow()
    return ClosedTrade("x", "TEST", Side.LONG, 100, 104, 1, pnl, r, now, now, exit_reason)


def test_target_hit_rates():
    trades = [
        trade("tp2"),                     # reached TP1 and TP2
        trade("tp1"),                     # reached TP1
        trade("stop_after_tp1", -10, -0.1),  # took its partial, then stopped out
        trade("stop", -100, -1.0),        # never reached either
        trade("end_of_data", 5, 0.1),
    ]
    m = compute(trades, 100_000)
    assert m.tp1_hit_rate == pytest.approx(3 / 5)
    assert m.tp2_hit_rate == pytest.approx(1 / 5)


def test_hit_rates_are_zero_without_trades():
    m = compute([], 100_000)
    assert m.tp1_hit_rate == 0.0 and m.tp2_hit_rate == 0.0


def test_hit_rates_appear_in_the_rendered_metrics():
    assert "TP1" in compute([trade("tp2")], 100_000).render()
