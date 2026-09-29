"""The strategy config is a contract: it must fail loudly, at startup."""

from __future__ import annotations

import pytest

from tradebot.strategy.presets import PRESETS
from tradebot.strategy.spec import StrategySpec, StrategySpecError


def spec(data):
    return StrategySpec.from_config(data, PRESETS)


def test_a_preset_expands_into_rules():
    s = spec({"preset": "ema_vwap_rsi"})
    assert [r.rule for r in s.active()][:2] == ["ema_stack", "ema_cross_fresh"]
    assert s.entry.tp1_r == 2.0


def test_rules_from_config_replace_the_preset():
    s = spec({"preset": "ema_vwap_rsi",
              "rules": [{"rule": "rsi_window", "mode": "required"}]})
    assert [r.rule for r in s.active()] == ["rsi_window"]


def test_params_come_from_config():
    s = spec({"rules": [{"rule": "rsi_window", "mode": "required",
                         "params": {"long_min": 45, "long_max": 65}}]})
    assert s.rules[0].params == {"long_min": 45, "long_max": 65}


def test_modes_are_honoured():
    s = spec({"rules": [
        {"rule": "ema_stack", "mode": "required"},
        {"rule": "volume_confirmation", "mode": "advisory"},
        {"rule": "liquidity_sweep", "mode": "disabled"},
    ]})
    assert [r.rule for r in s.active()] == ["ema_stack", "volume_confirmation"]
    assert [r.rule for r in s.rules if r.required] == ["ema_stack"]


def test_a_bare_string_is_a_required_rule():
    s = spec({"rules": ["ema_stack", "vwap_position"]})
    assert all(r.required for r in s.active())


def test_targets_and_stops_are_configurable():
    s = spec({"rules": ["ema_stack"],
              "entry": {"tp1_r": 1.5, "tp2_r": 3.0, "min_rr": 1.2,
                        "stop": {"method": "fixed_ticks", "fixed_ticks": 20}}})
    assert (s.entry.tp1_r, s.entry.tp2_r, s.entry.min_rr) == (1.5, 3.0, 1.2)
    assert s.entry.stop.method == "fixed_ticks" and s.entry.stop.fixed_ticks == 20


@pytest.mark.parametrize("bad,message", [
    ({"rules": []}, "non-empty"),
    ({"rules": [{"rule": "no_such_rule"}]}, "unknown rule"),
    ({"rules": [{"mode": "required"}]}, "missing a 'rule'"),
    ({"rules": [{"rule": "ema_stack", "mode": "sometimes"}]}, "mode"),
    ({"rules": [{"rule": "ema_stack"}, {"rule": "ema_stack"}]}, "twice"),
    ({"rules": [{"rule": "ema_stack", "mode": "advisory"}]}, "at least one rule must be required"),
    ({"preset": "nope"}, "unknown"),
    ({"rules": ["ema_stack"], "min_score": 1.5}, "between 0 and 1"),
    ({"rules": ["ema_stack"], "entry": {"tp1_r": 0}}, "positive"),
    ({"rules": ["ema_stack"], "entry": {"stop": {"method": "vibes"}}}, "stop method"),
])
def test_bad_config_is_rejected_with_a_useful_message(bad, message):
    with pytest.raises(StrategySpecError) as exc:
        spec(bad)
    assert message in str(exc.value)


def test_warmup_is_derived_from_the_longest_lookback():
    short = spec({"rules": [{"rule": "rsi_window", "params": {"period": 14}}],
                  "structure_window": 10})
    long_ = spec({"rules": [{"rule": "rsi_window", "params": {"period": 200}}],
                  "structure_window": 10})
    assert long_.min_bars() > short.min_bars() >= 19


def test_overlays_follow_the_configured_periods():
    s = spec({"rules": [{"rule": "ema_stack", "params": {"fast": 5, "slow": 50}},
                        "vwap_position"]})
    assert {o.get("period") for o in s.overlays()} == {5, 50, None}
