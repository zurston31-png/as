"""Configuration loading, validation and hashing."""

from __future__ import annotations

from datetime import date

import pytest

from flow_model.config.loader import (
    ConfigError,
    apply_override,
    deep_merge,
    default_config,
    load_config,
    load_raw,
    parse_override,
    save_config,
)
from flow_model.config.schema import FlowModelConfig, FlowScoreConfig, SetupConfig
from flow_model.core.enums import Component, Regime, SetupType


# --- loading ---------------------------------------------------------------


def test_defaults_load_and_validate(config):
    assert config.name == "flow_model_default"
    assert set(config.instruments) == {"NQ", "ES", "GC", "QQQ", "SPX"}
    assert config.backtest.symbols == ("NQ", "ES")


def test_all_five_required_markets_are_configured(config):
    """The brief names NQ, ES, GC, SPX, QQQ."""
    assert {"NQ", "ES", "GC", "SPX", "QQQ"} <= set(config.instruments)


def test_real_contract_economics(config):
    assert config.spec("NQ").point_value == pytest.approx(20.0)
    assert config.spec("ES").point_value == pytest.approx(50.0)
    assert config.spec("GC").point_value == pytest.approx(100.0)
    assert config.spec("QQQ").point_value == pytest.approx(1.0)


def test_spx_ships_as_reference_only(config):
    """An index cannot be filled; shipping it tradable would fake the costs."""
    assert config.spec("SPX").tradable is False


def test_unknown_symbol_raises_with_a_useful_message(config):
    with pytest.raises(KeyError, match="NQ"):
        config.spec("CL")


def test_three_setups_with_honest_r_targets(config):
    assert set(config.setups) == {SetupType.SCALP_1R, SetupType.SETUP_2R, SetupType.DIRECTIONAL_3R}
    assert config.setup(SetupType.SCALP_1R).reward_risk == pytest.approx(1.0)
    assert config.setup(SetupType.SETUP_2R).reward_risk == pytest.approx(2.0)
    assert config.setup(SetupType.DIRECTIONAL_3R).reward_risk >= 3.0


def test_scalp_excludes_chop_by_default(config):
    """The brief expects degradation in chop; the default declines to trade it."""
    assert Regime.CHOP not in config.setup(SetupType.SCALP_1R).allowed_regimes


def test_warmup_covers_the_longest_lookback(config):
    f = config.features
    assert f.warmup_bars > max(f.atr_percentile_lookback, f.structure_lookback_bars)


# --- hashing ---------------------------------------------------------------


def test_config_hash_is_stable_across_loads(config):
    assert default_config().config_hash == config.config_hash


def test_cosmetic_changes_do_not_create_a_new_experiment_identity(config):
    assert config.replace(name="other").config_hash == config.config_hash
    assert config.replace(description="notes").config_hash == config.config_hash
    assert load_config(overrides=["logging.level=DEBUG"]).config_hash == config.config_hash
    assert load_config(overrides=["paths.reports_dir=/tmp/x"]).config_hash == config.config_hash


@pytest.mark.parametrize(
    "override",
    [
        "risk.risk_per_trade_pct=0.0025",
        "setups.SCALP_1R.min_flow_score=75",
        "regime.high_vol_percentile=0.85",
        "execution.base_slippage_ticks=2.0",
        "monte_carlo.block_length=40",
        "data.data_latency_seconds=5.0",
    ],
)
def test_result_affecting_changes_create_a_new_identity(config, override):
    assert load_config(overrides=[override]).config_hash != config.config_hash


def test_research_targets_excluded_from_hash(config):
    """Targets are not inputs to the model, so they cannot change its identity."""
    changed = config.replace(
        research_targets=config.research_targets.replace(combined_10y_win_rate=0.5)
    )
    assert changed.config_hash == config.config_hash


# --- overrides -------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,path,value",
    [
        ("risk.risk_per_trade_pct=0.0025", ["risk", "risk_per_trade_pct"], 0.0025),
        ("backtest.symbols=[NQ,ES]", ["backtest", "symbols"], ["NQ", "ES"]),
        ("monte_carlo.n_simulations=100000", ["monte_carlo", "n_simulations"], 100000),
        ("flow_score.strict_component_availability=false",
         ["flow_score", "strict_component_availability"], False),
        ("seal.enabled=yes", ["seal", "enabled"], True),
        ("backtest.start=2016-01-01", ["backtest", "start"], date(2016, 1, 1)),
    ],
)
def test_parse_override(expression, path, value):
    assert parse_override(expression) == (path, value)


def test_parse_override_rejects_malformed():
    with pytest.raises(ConfigError, match="dotted.key=value"):
        parse_override("risk.risk_per_trade_pct")
    with pytest.raises(ConfigError, match="empty key"):
        parse_override("=0.5")


def test_overrides_apply(config):
    c = load_config(overrides=["risk.risk_per_trade_pct=0.0025", "backtest.symbols=[NQ]"])
    assert c.risk.risk_per_trade_pct == pytest.approx(0.0025)
    assert c.backtest.symbols == ("NQ",)


def test_deep_merge_replaces_sequences_rather_than_appending():
    merged = deep_merge({"a": [1, 2, 3], "b": {"x": 1, "y": 2}}, {"a": [9], "b": {"y": 5}})
    assert merged["a"] == [9]
    assert merged["b"] == {"x": 1, "y": 5}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_apply_override_refuses_to_bury_a_scalar():
    with pytest.raises(ConfigError, match="not a mapping"):
        apply_override({"risk": 1}, ["risk", "risk_per_trade_pct"], 0.1)


def test_yaml_round_trip_preserves_identity(config, tmp_path):
    path = save_config(config, tmp_path / "resolved.yaml")
    assert load_config([path], include_defaults=False).config_hash == config.config_hash


def test_missing_config_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config(["/nonexistent/path.yaml"])


def test_user_file_overrides_defaults(tmp_path):
    p = tmp_path / "mine.yaml"
    p.write_text("risk:\n  risk_per_trade_pct: 0.0025\n")
    assert load_config([p]).risk.risk_per_trade_pct == pytest.approx(0.0025)


# --- validation: things that must be rejected ------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        (["flow_score.weights.order_flow=40"], "weights sum to"),
        (["risk.risk_per_trade_pct=0.02"], "exceeds max_risk_per_trade_pct"),
        (["risk.risk_per_trade_pct=0.5"], "Percentages are fractions"),
        (["risk.max_risk_per_trade_pct=0.5"], "Percentages are fractions"),
        (["setups.SCALP_1R.reward_risk=0.4"], "its name asserts"),
        (["setups.SETUP_2R.reward_risk=1.2"], "its name asserts"),
        (["setups.DIRECTIONAL_3R.reward_risk=2.0"], "at least 3R"),
        (["backtest.symbols=[CL]"], "no InstrumentSpec"),
        (["backtest.symbols=[SPX]"], "tradable=False"),
        (["setups.SCALP_1R.allowed_regimes=[UNKNOWN]"], "never tradable"),
        (["setups.SCALP_1R.allowed_regimes=[]"], "must not be empty"),
        (["regime.low_vol_percentile=0.9"], "below high_vol_percentile"),
        (["seal.sealed_end=2030-01-01"], "never be reachable"),
        (["seal.sealed_start=2010-01-01"], "before the backtest period"),
        (["walk_forward.step_months=24"], "out-of-sample periods would be skipped"),
        (["risk.kill_switch_drawdown_pct=0.01"], "must exceed daily_loss_limit_pct"),
        (["risk.max_portfolio_heat_pct=0.001"], "no trade could ever open"),
        (["risk.weekly_loss_limit_pct=0.01"], "below daily_loss_limit_pct"),
        (["execution.entry_order_type=limit", "execution.max_entry_wait_bars=0"], "can never fill"),
        (["data.intrabar_interval_seconds=600"], "finer than"),
        (["data.higher_intervals_seconds=[60]"], "must exceed primary"),
        (["consistency.weight_year_stability=0.5"], "weights sum to"),
        (["monte_carlo.percentiles=[50.0,25.0]"], "sorted ascending"),
        (["monte_carlo.percentiles=[0.0,50.0]"], "strictly between 0 and 100"),
        (["monte_carlo.drawdown_thresholds=[10.0]"], "fractions in"),
        (["analytics.flow_score_buckets=[80.0,60.0]"], "ascending"),
        (["backtest.end=2010-01-01"], "after start"),
        (["setups.SCALP_1R.min_vol_percentile=0.99"], "below max_vol_percentile"),
        (["setups.SCALP_1R.min_reward_risk=1.5"], "exceeds the setup's own"),
        (["setups.SCALP_1R.max_stop_atr_multiple=0.5"], "below stop_atr_multiple"),
    ],
)
def test_invalid_configurations_are_rejected(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(overrides=overrides)


def test_disabling_every_setup_is_rejected():
    with pytest.raises(ConfigError, match="could never trade"):
        load_config(overrides=[
            "setups.SCALP_1R.enabled=false",
            "setups.SETUP_2R.enabled=false",
            "setups.DIRECTIONAL_3R.enabled=false",
        ])


def test_missing_component_weight_is_rejected():
    raw = load_raw()
    del raw["flow_score"]["weights"]["liquidity"]
    with pytest.raises(Exception, match="missing weights"):
        FlowModelConfig(**raw)


def test_negative_component_weight_is_rejected():
    with pytest.raises(Exception):
        FlowScoreConfig(weights={
            Component.OPTIONS_FLOW: -5.0, Component.ORDER_FLOW: 45.0,
            Component.STRUCTURE: 20.0, Component.LIQUIDITY: 15.0,
            Component.VOL_MOMENTUM: 25.0,
        })


def test_instrument_registry_key_must_match_symbol():
    raw = load_raw()
    raw["instruments"]["NQ"]["symbol"] = "MNQ"
    with pytest.raises(Exception, match="does not match"):
        FlowModelConfig(**raw)


def test_execution_proxy_must_exist():
    raw = load_raw()
    raw["instruments"]["SPX"]["tradable"] = True
    raw["instruments"]["SPX"]["execution_proxy"] = "SPY"
    with pytest.raises(Exception, match="not a known instrument"):
        FlowModelConfig(**raw)


def test_spx_becomes_tradable_via_a_real_proxy():
    c = load_config(overrides=[
        "instruments.SPX.tradable=true",
        "instruments.SPX.execution_proxy=ES",
        "backtest.symbols=[SPX]",
    ])
    assert c.spec("SPX").execution_proxy == "ES"


def test_equal_weight_baseline_is_expressible():
    """Weight sweeps in Phase 8 need this to be a legal config."""
    c = load_config(overrides=[
        "flow_score.weights.options_flow=20",
        "flow_score.weights.order_flow=20",
        "flow_score.weights.structure=20",
        "flow_score.weights.liquidity=20",
        "flow_score.weights.vol_momentum=20",
    ])
    assert len(set(c.flow_score.weights.values())) == 1


def test_all_risk_presets_from_the_brief_are_valid():
    for pct in (0.0025, 0.005, 0.0075, 0.01):
        c = load_config(overrides=[f"risk.risk_per_trade_pct={pct}"])
        assert c.risk.risk_per_trade_pct == pytest.approx(pct)


def test_all_robustness_score_thresholds_are_valid(config):
    for threshold in config.robustness.flow_score_thresholds:
        c = load_config(overrides=[f"setups.SCALP_1R.min_flow_score={threshold}"])
        assert c.setup(SetupType.SCALP_1R).min_flow_score == pytest.approx(threshold)


def test_strict_component_availability_defaults_on(config):
    """Redistributing a missing component's weight inflates scores."""
    assert config.flow_score.strict_component_availability is True
    assert config.flow_score.redistribute_disabled_weight is False


def test_pessimistic_execution_defaults(config):
    assert config.execution.pessimistic_same_bar is True
    assert config.execution.fill_gaps_at_open is True
    assert config.execution.base_slippage_ticks >= 1.0
    assert config.execution.latency_ms > 0


def test_analytics_refuses_mixed_splits_by_default(config):
    assert config.analytics.allow_mixed_split_aggregation is False
    assert config.analytics.min_sample_n >= 30


def test_feed_requirements_declared_for_every_component(config):
    assert set(config.flow_score.feed_requirements) == set(Component)
    orderflow = config.flow_score.feed_requirements[Component.ORDER_FLOW]
    assert any(r.feed == "tick_aggregate" and r.required for r in orderflow)
