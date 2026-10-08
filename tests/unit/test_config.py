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
from flow_model.config.schema import (
    BAR_VOLUME_DELTA_PROXIES,
    FeatureConfig,
    FlowModelConfig,
    FlowScoreConfig,
    OptionsFlowConfig,
    OptionsFlowWeights,
    OrderFlowConfig,
    OrderFlowWeights,
    SetupConfig,
)
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


# --- order flow and options flow (Phases 4 and 5) --------------------------
#
# These two sections were added in one pass so the Phase 4 and Phase 5
# implementations never have to edit schema.py. Each non-negotiable validator
# below is exercised against the bad configuration it exists to refuse: a
# validator with no test is a validator nobody can rely on.


def test_both_new_sections_are_reachable_from_the_loader(config):
    assert isinstance(config.order_flow, OrderFlowConfig)
    assert isinstance(config.options_flow, OptionsFlowConfig)


def test_the_phase_1_placeholders_have_exactly_one_home_now():
    """`cvd_period`, `absorption_lookback` and `options_lookback_days` were
    parked on FeatureConfig in Phase 1. They now live in the two sections that
    own those parameter surfaces. Two homes for one window means a computer
    reads whichever its author remembered, and both configurations load."""
    assert "cvd_period" not in FeatureConfig.model_fields
    assert "absorption_lookback" not in FeatureConfig.model_fields
    assert "options_lookback_days" not in FeatureConfig.model_fields
    assert "cvd_lookback_bars" in OrderFlowConfig.model_fields
    assert "absorption_lookback_bars" in OrderFlowConfig.model_fields
    assert "lookback_snapshots" in OptionsFlowConfig.model_fields


def test_both_sections_change_the_experiment_identity(config):
    """They feed features, so they are results-affecting by definition."""
    assert load_config(overrides=["order_flow.cvd_lookback_bars=30"]).config_hash \
        != config.config_hash
    assert load_config(overrides=["options_flow.lookback_snapshots=40"]).config_hash \
        != config.config_hash


def test_order_flow_covers_every_feature_section_5_names(config):
    """signed delta, CVD slope, aggression ratio, absorption at a level,
    trade-size distribution. A missing knob becomes a hardcoded constant."""
    o = config.order_flow
    assert o.delta_percentile_lookback > 0        # signed delta
    assert o.cvd_lookback_bars > 0                # CVD slope (the lookback)
    assert o.aggression_lookback_bars > 0         # aggression ratio
    assert o.absorption_lookback_bars > 0         # absorption at a level
    assert o.trade_size_percentile_lookback > 0   # trade-size distribution
    assert set(OrderFlowWeights.model_fields) == {
        "signed_delta", "cvd_slope", "aggression_ratio", "absorption",
        "trade_size_distribution",
    }


def test_options_flow_covers_every_feature_section_5_names(config):
    """net premium, delta-weighted volume, OI change, 25d skew, gamma proxy."""
    assert set(OptionsFlowWeights.model_fields) == {
        "net_premium", "delta_weighted_volume", "oi_change", "skew_25d",
        "gamma_exposure",
    }
    assert config.options_flow.lookback_snapshots > 0
    assert config.options_flow.use_gamma_exposure_proxy is True


# --- the no-proxy rule (non-negotiable, ARCHITECTURE section 5) ------------


def test_no_proxy_rule_is_the_default(config):
    assert config.order_flow.allow_bar_volume_delta_proxy is False
    assert config.order_flow.require_aggressor_classification is True


def test_enabling_the_bar_volume_delta_proxy_is_refused():
    """Section 5 accepts NO proxy for order flow. A bar-volume delta is
    indistinguishable from real order flow to every caller above, including
    the lookahead audit -- which would pass."""
    with pytest.raises(ConfigError, match="known-bad estimator"):
        load_config(overrides=["order_flow.allow_bar_volume_delta_proxy=true"])


def test_accepting_an_undeclared_classification_method_is_refused():
    with pytest.raises(ConfigError, match="back door"):
        load_config(overrides=["order_flow.require_aggressor_classification=false"])


@pytest.mark.parametrize(
    "denylist",
    [
        "[]",
        "[bar_volume]",
        "[bar_volume,tick_rule_on_bars,volume_split]",
    ],
)
def test_emptying_the_proxy_denylist_is_refused(denylist):
    """A denylist that can be emptied is a default, not a rule."""
    with pytest.raises(ConfigError, match="no longer rejects"):
        load_config(overrides=[f"order_flow.rejected_classification_methods={denylist}"])


def test_the_proxy_denylist_covers_the_known_bad_names(config):
    assert BAR_VOLUME_DELTA_PROXIES <= set(config.order_flow.rejected_classification_methods)


def test_the_proxy_denylist_is_case_normalized():
    """'Bar_Volume' slipping past the membership test is the one failure this
    field exists to prevent, so the comparison may not depend on case."""
    c = load_config(overrides=[
        "order_flow.rejected_classification_methods="
        "[BAR_VOLUME,Bar_Volume_Tick_Rule,Tick_Rule_On_Bars,"
        "UPTICK_DOWNTICK_ON_BARS,VolumeSplit,volume_split]"
    ])
    methods = c.order_flow.rejected_classification_methods
    assert BAR_VOLUME_DELTA_PROXIES <= set(methods)
    assert methods == tuple(dict.fromkeys(methods)), "duplicates survived"
    assert all(m == m.lower() for m in methods)


def test_zero_classification_coverage_is_refused():
    """Accepting a tape with no known aggressor side at all is the proxy under
    another name, so the bound is strictly above zero."""
    with pytest.raises(ConfigError):
        load_config(overrides=["order_flow.min_classification_coverage=0.0"])


# --- the EOD cap (non-negotiable, ARCHITECTURE section 5) ------------------


def test_eod_degraded_path_is_capped_and_can_bind(config):
    assert 0.0 < config.options_flow.eod_degraded_cap_fraction < 1.0
    assert config.options_flow.require_intraday_prints is False


@pytest.mark.parametrize("fraction", ["1.0", "1.5", "2.0"])
def test_a_cap_that_could_not_bind_is_refused(fraction):
    """A cap at or above the component's full weight is not a cap, and the
    DEGRADED flag would then describe a restriction never applied."""
    with pytest.raises(ConfigError, match="not a cap"):
        load_config(overrides=[f"options_flow.eod_degraded_cap_fraction={fraction}"])


def test_the_cap_is_a_fraction_so_it_survives_a_weight_sweep(config):
    """Expressed in absolute points, a 10-point cap stops binding the moment
    Phase 8 lowers the component's weight to 8, and nothing would say so."""
    c = load_config(overrides=[
        "flow_score.weights.options_flow=8",
        "flow_score.weights.order_flow=37",
    ])
    cap_points = c.options_flow.eod_degraded_cap_fraction * c.flow_score.weights[
        Component.OPTIONS_FLOW
    ]
    assert cap_points < c.flow_score.weights[Component.OPTIONS_FLOW]


def test_synthesizing_missing_options_fields_is_refused():
    with pytest.raises(ConfigError, match="never synthesized"):
        load_config(overrides=["options_flow.never_synthesize_missing_fields=false"])


def test_a_disabled_gamma_proxy_must_have_its_weight_zeroed():
    """Otherwise the weights still sum to 1.0 while 0.15 of the sub-score can
    never be earned: a ceiling below 100% that no caller can see."""
    with pytest.raises(ConfigError, match="no caller could see"):
        load_config(overrides=["options_flow.use_gamma_exposure_proxy=false"])


def test_disabling_the_gamma_proxy_is_legal_once_the_weight_is_moved():
    """Phase 8 needs the ablation to be an expressible configuration."""
    c = load_config(overrides=[
        "options_flow.use_gamma_exposure_proxy=false",
        "options_flow.weights.gamma_exposure=0.0",
        "options_flow.weights.skew_25d=0.30",
    ])
    assert c.options_flow.use_gamma_exposure_proxy is False
    assert c.options_flow.weights.gamma_exposure == pytest.approx(0.0)


# --- weights and availability floors --------------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        (["order_flow.weights.signed_delta=0.40"], "order-flow weights sum to"),
        (["options_flow.weights.net_premium=0.40"], "options-flow weights sum to"),
    ],
)
def test_new_weight_blocks_must_sum_to_one(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(overrides=overrides)


def test_dropped_terms_are_not_redistributed_so_a_floor_exists(config):
    """Section 14.5's rule applied to both components: a missing term's weight
    stays lost. Without a floor the sub-score reads 'balanced' when it means
    'never measured'."""
    assert 0.0 < config.order_flow.min_available_weight_fraction <= 1.0
    assert 0.0 < config.options_flow.min_available_weight_fraction <= 1.0


def test_premium_alone_cannot_satisfy_the_options_availability_floor(config):
    """Four of the five OptionsSnapshot inputs are optional; only premium is
    always present. A premium-only chain must report UNAVAILABLE."""
    w = config.options_flow.weights
    assert w.net_premium < config.options_flow.min_available_weight_fraction
    assert w.net_premium + w.delta_weighted_volume \
        >= config.options_flow.min_available_weight_fraction


# --- warmup honesty -------------------------------------------------------


def test_order_flow_warmup_covers_composed_windows_not_the_longest_one(config):
    """Under-declaring warmup was a real HIGH bug in the regime detector (268
    declared, 291 read) and the lookahead audit could not see it, because
    nothing read the future -- the label at bar t just depended on where the
    caller started loading. Ranking a smoothed value inside a percentile
    window reads `lookback + smoothing - 1` bars, not `lookback`."""
    o = config.order_flow
    longest_single_window = max(
        o.delta_percentile_lookback,
        o.trade_size_percentile_lookback,
        o.cvd_lookback_bars,
        o.absorption_lookback_bars,
        o.aggression_lookback_bars,
        o.delta_smoothing_bars,
    )
    assert o.warmup_bars > longest_single_window
    assert o.warmup_bars >= (
        o.trade_size_percentile_lookback + o.aggression_lookback_bars - 1
    )
    assert o.warmup_bars >= (
        o.delta_percentile_lookback + o.absorption_lookback_bars - 1
    )


def test_order_flow_warmup_is_exactly_the_max_of_the_composed_chains(config):
    """The arithmetic pinned, recomputed here rather than trusted. If the
    property is ever rewritten to take the longest single window instead, this
    is what fails."""
    o = config.order_flow
    assert o.warmup_bars == 1 + max(
        o.delta_percentile_lookback + o.delta_smoothing_bars - 1,
        o.delta_percentile_lookback + o.absorption_lookback_bars - 1,
        o.trade_size_percentile_lookback + o.aggression_lookback_bars - 1,
        o.cvd_lookback_bars,
    )


@pytest.mark.parametrize(
    "field,low,high,extra",
    [
        ("delta_smoothing_bars", 1, 50, ["order_flow.trade_size_percentile_lookback=20"]),
        ("absorption_lookback_bars", 1, 50, ["order_flow.trade_size_percentile_lookback=20"]),
        ("delta_percentile_lookback", 60, 300, []),
        ("trade_size_percentile_lookback", 60, 300, []),
        ("aggression_lookback_bars", 1, 50, []),
        ("cvd_lookback_bars", 20, 600, []),
    ],
)
def test_every_window_is_inside_the_order_flow_warmup(field, low, high, extra):
    """Not that warmup rises on any change -- it is a max, so a non-dominant
    window grows without moving it -- but that NO window is left OUT of the
    max. A window absent from the computation is the under-declaration bug
    lying in wait for the Phase 8 sweep that makes that window the longest.
    `extra` puts the window under test on the dominant chain first; without it
    the test would pass vacuously for the short windows."""
    lo = load_config(overrides=[*extra, f"order_flow.{field}={low}"]).order_flow
    hi = load_config(overrides=[*extra, f"order_flow.{field}={high}"]).order_flow
    assert hi.warmup_bars > lo.warmup_bars


def test_options_warmup_is_counted_in_snapshots_not_days(config):
    """The feed's cadence is unobservable from config: EOD chains give roughly
    one snapshot per session, OPRA prints give many. The old name
    `options_lookback_days` asserted a cadence this section does not."""
    assert config.options_flow.warmup_snapshots > config.options_flow.lookback_snapshots
    assert "days" not in " ".join(OptionsFlowConfig.model_fields)


# --- coherence ------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        (["order_flow.delta_smoothing_bars=80"], "smoothed value would be ranked"),
        (["order_flow.absorption_lookback_bars=60"], "not shorter than delta_percentile"),
        (["order_flow.aggression_lookback_bars=120"],
         "not shorter than trade_size_percentile"),
        (["options_flow.min_snapshot_volume=-1"], "greater than or equal to 0"),
        (["order_flow.cvd_slope_squash_scale=0"], "greater than 0"),
        (["order_flow.large_trade_percentile=1.0"], "less than 1"),
        (["order_flow.absorption_delta_percentile=1.0"], "less than 1"),
        (["options_flow.lookback_snapshots=2"], "greater than 2"),
    ],
)
def test_incoherent_new_settings_are_rejected(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(overrides=overrides)


def test_the_phase_8_sweep_ranges_for_the_new_sections_are_all_legal():
    """Constants are hypotheses. If a plausible sweep value does not load, the
    sweep silently stops at the config instead of at a measurement."""
    for lookback in (20, 40, 60, 120, 252):
        assert load_config(
            overrides=[f"order_flow.delta_percentile_lookback={lookback}"]
        ).order_flow.delta_percentile_lookback == lookback
    for coverage in (0.25, 0.5, 0.6, 0.75, 0.9, 1.0):
        assert load_config(
            overrides=[f"order_flow.min_classification_coverage={coverage}"]
        ).order_flow.min_classification_coverage == pytest.approx(coverage)
    for cap in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert load_config(
            overrides=[f"options_flow.eod_degraded_cap_fraction={cap}"]
        ).options_flow.eod_degraded_cap_fraction == pytest.approx(cap)
