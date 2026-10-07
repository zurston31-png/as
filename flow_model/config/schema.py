"""Configuration schema.

Every tunable in the system is declared here, validated by Pydantic, and
loaded from YAML. Nothing reads a magic number from code.

Three conventions:

* **Percentages are fractions.** `risk_per_trade_pct = 0.005` means 0.5%.
  A validator rejects values above 0.25 to catch the common 50-vs-0.5 error.
* **Weights are hypotheses.** The 20/25/20/15/20 point scheme from the brief
  is a declared prior, not a finding. It is swept in Phase 8.
* **Targets are not objectives.** `ResearchTargets` records the brief's
  desired win rates so the final report can state whether they were met or
  refuted. No module outside `analytics`/`validation` may read it, and a
  test enforces that -- otherwise "do not optimize for a desired win rate"
  is just a comment.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field, computed_field, model_validator

from flow_model.core.determinism import DEFAULT_SEED, stable_hash
from flow_model.core.enums import (
    Component,
    DataQuality,
    Feed,
    MonteCarloMethod,
    Regime,
    SetupType,
)
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel

MAX_SANE_PCT = 0.25  # 25% -- anything larger is almost certainly a unit error


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class FeedRequirement(FrozenModel):
    """What a component needs from the data layer, and whether it is optional.

    This is the contract that makes honest degradation possible: the data
    layer reports which feeds exist, and the signal engine disables the
    components whose required feeds are absent rather than substituting a
    proxy.
    """

    feed: Feed
    minimum_quality: DataQuality = DataQuality.DEGRADED
    required: bool = True


class DataConfig(FrozenModel):
    provider: str = Field(
        default="csv",
        description="Adapter name: csv | parquet | synthetic | <vendor>.",
    )
    root_path: str = "data_cache"
    primary_interval_seconds: int = Field(gt=0, default=300)
    higher_intervals_seconds: tuple[int, ...] = (900, 3600, 86400)
    intrabar_interval_seconds: int | None = Field(
        default=60,
        gt=0,
        description=(
            "Finer bars used to resolve stop-vs-target ordering within a "
            "primary bar. None forces the pessimistic same-bar rule."
        ),
    )

    timezone: str = "America/New_York"
    data_latency_seconds: float = Field(
        ge=0.0,
        default=0.0,
        description=(
            "Point-in-time offset. A bar closing at t is only visible at "
            "t + data_latency_seconds. Models feed delay honestly."
        ),
    )
    max_staleness_seconds: float = Field(
        gt=0.0,
        default=120.0,
        description="Default age beyond which a feed is graded STALE and trading halts.",
    )
    max_staleness_seconds_by_feed: dict[Feed, float] = Field(
        default_factory=lambda: {Feed.OPTIONS_SNAPSHOT: 345_600.0},
        description=(
            "Per-feed overrides, in seconds. One global bound cannot fit both "
            "a 5-minute bar feed and a once-per-session options chain: an "
            "end-of-day chain is a session old by construction, so a 120s "
            "limit graded it STALE on 98.7% of bars while the availability "
            "report claimed all 100 points were real. The options default is "
            "four days, which spans a long weekend -- a Friday chain is ~63 "
            "hours old at Monday's open and is normal, not frozen -- while "
            "still catching a feed that has genuinely stopped updating."
        ),
    )
    stale_cadence_multiple: float = Field(
        ge=1.0,
        default=2.5,
        description=(
            "A feed is also allowed this many of its OWN observed intervals "
            "before being called stale, so 'stale' means missed observations "
            "rather than elapsed wall-clock. The configured limit is a floor, "
            "never a ceiling, so tightening it cannot be undone by cadence."
        ),
    )
    min_quality_to_trade: DataQuality = DataQuality.DEGRADED

    drop_partial_bars: bool = True
    max_gap_bars_tolerated: int = Field(ge=0, default=3)
    outlier_sigma: float = Field(
        gt=0.0,
        default=10.0,
        description=(
            "Bars whose return exceeds this many robust sigma are quarantined. "
            "Raising it to 15 more than halves recall (0.75 -> 0.31) against "
            "the synthetic generator's injected outliers, so 10 is the "
            "measured operating point, not a round number."
        ),
    )
    outlier_window_bars: int = Field(
        gt=10,
        default=500,
        description=(
            "Trailing window for the robust dispersion estimate. Must span "
            "several volatility regimes: the generator's LOW_VOL to HIGH_VOL "
            "ratio is 5.5x and DIRECTIONAL_SHIFT lasts ~6 bars, so a 50-bar "
            "window flagged the first bars after every vol switch as bad "
            "prints -- 11 false positives on a dataset with zero injected "
            "outliers, precision 0.52. Measured against ground truth: 50 "
            "bars gives precision 0.52, 100 gives 0.50, 250 gives 0.65, 500 "
            "gives 0.80 at unchanged recall."
        ),
    )
    outlier_min_history_bars: int = Field(
        gt=1,
        default=20,
        description=(
            "Bars of trailing history required before a bar can be "
            "quarantined at all. Earlier bars are never judged: there is "
            "nothing to judge them against, and judging them on later data "
            "is the lookahead this avoids."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> "DataConfig":
        if self.intrabar_interval_seconds is not None:
            if self.intrabar_interval_seconds >= self.primary_interval_seconds:
                raise ValueError(
                    "intrabar_interval_seconds must be finer than primary_interval_seconds"
                )
        for hi in self.higher_intervals_seconds:
            if hi <= self.primary_interval_seconds:
                raise ValueError(
                    f"higher interval {hi}s must exceed primary "
                    f"{self.primary_interval_seconds}s"
                )
        return self


# ---------------------------------------------------------------------------
# Features and regime
# ---------------------------------------------------------------------------


class FeatureConfig(FrozenModel):
    atr_period: int = Field(gt=1, default=14)
    atr_percentile_lookback: int = Field(gt=10, default=252)
    realized_vol_period: int = Field(gt=1, default=20)
    vol_of_vol_period: int = Field(gt=1, default=20)

    swing_lookback: int = Field(gt=1, default=5)
    swing_atr_multiple: float = Field(
        gt=0.0, default=0.5, description="Minimum swing size in ATR units."
    )
    structure_lookback_bars: int = Field(gt=1, default=50)
    vwap_anchor: str = Field(default="session", pattern="^(session|day|week)$")

    momentum_period: int = Field(gt=1, default=10)
    efficiency_period: int = Field(gt=1, default=20)

    volume_percentile_lookback: int = Field(gt=10, default=60)
    spread_percentile_lookback: int = Field(gt=10, default=60)
    time_of_day_buckets: int = Field(gt=1, default=26)

    cvd_period: int = Field(gt=1, default=20)
    absorption_lookback: int = Field(gt=1, default=10)

    options_lookback_days: int = Field(gt=1, default=20)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warmup_bars(self) -> int:
        """Bars required before any feature is valid.

        The signal engine returns WAIT('warmup') until this many bars exist,
        which prevents the classic bug of trading on a half-filled rolling
        window.
        """
        return max(
            self.atr_percentile_lookback,
            self.structure_lookback_bars,
            self.volume_percentile_lookback,
            self.spread_percentile_lookback,
            self.realized_vol_period + self.vol_of_vol_period,
            self.efficiency_period,
            self.cvd_period,
        ) + 1


class RegimeConfig(FrozenModel):
    """Thresholds for the quantitative regime classifier.

    Classification order is fixed in code (DIRECTIONAL_SHIFT first, then
    vol extremes, then trend, else CHOP); only the thresholds are tunable.
    """

    high_vol_percentile: float = Field(gt=0.0, lt=1.0, default=0.80)
    low_vol_percentile: float = Field(gt=0.0, lt=1.0, default=0.20)
    trend_efficiency_threshold: float = Field(gt=0.0, lt=1.0, default=0.40)
    trend_strength_threshold: float = Field(ge=0.0, default=0.25)
    cusum_threshold: float = Field(gt=0.0, default=4.0)
    cusum_drift: float = Field(ge=0.0, default=0.5)
    min_regime_bars: int = Field(
        ge=1, default=3, description="Hysteresis: bars of confirmation before switching."
    )
    shift_decay_bars: int = Field(
        ge=1, default=10, description="How long DIRECTIONAL_SHIFT persists after triggering."
    )

    @model_validator(mode="after")
    def _check(self) -> "RegimeConfig":
        if self.low_vol_percentile >= self.high_vol_percentile:
            raise ValueError("low_vol_percentile must be below high_vol_percentile")
        return self


# ---------------------------------------------------------------------------
# Flow Score and setups
# ---------------------------------------------------------------------------


class FlowScoreConfig(FrozenModel):
    weights: dict[Component, float] = Field(
        default_factory=lambda: {
            Component.OPTIONS_FLOW: 20.0,
            Component.ORDER_FLOW: 25.0,
            Component.STRUCTURE: 20.0,
            Component.LIQUIDITY: 15.0,
            Component.VOL_MOMENTUM: 20.0,
        }
    )
    total_points: float = Field(gt=0, default=100.0)

    strict_component_availability: bool = Field(
        default=True,
        description=(
            "If true, a component whose required feeds are missing makes the "
            "engine refuse to score (WAIT), rather than silently "
            "redistributing its weight to the remaining components. "
            "Redistribution inflates scores and is off by default."
        ),
    )
    redistribute_disabled_weight: bool = Field(
        default=False,
        description="Only consulted when strict_component_availability is false.",
    )

    enabled_components: dict[Component, bool] = Field(
        default_factory=lambda: {c: True for c in Component}
    )

    max_opposing_points: float = Field(
        ge=0.0,
        default=12.0,
        description=(
            "A trade is rejected if components opposing its direction hold "
            "more than this many weighted points (the contradiction gate)."
        ),
    )
    options_contradiction_points: float = Field(
        ge=0.0,
        default=14.0,
        description="Options-flow opposition above this blocks the trade.",
    )

    feed_requirements: dict[Component, tuple[FeedRequirement, ...]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _check(self) -> "FlowScoreConfig":
        missing = set(Component) - set(self.weights)
        if missing:
            raise ValueError(f"missing weights for components: {sorted(m.value for m in missing)}")
        total = sum(self.weights.values())
        if abs(total - self.total_points) > 1e-6:
            raise ValueError(
                f"component weights sum to {total}, expected {self.total_points}"
            )
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("component weights must be non-negative")
        if self.max_opposing_points > self.total_points:
            raise ValueError("max_opposing_points cannot exceed total_points")
        return self


class SetupConfig(FrozenModel):
    """Definition of one trade archetype.

    `reward_risk` is the *planned* R multiple and is enforced: a SCALP_1R
    whose target is 0.4x its stop is a configuration error, not a 90% win
    rate.
    """

    setup: SetupType
    enabled: bool = True
    reward_risk: float = Field(gt=0.0)
    min_flow_score: float = Field(ge=0.0, le=100.0)
    allowed_regimes: tuple[Regime, ...]
    min_vol_percentile: float = Field(ge=0.0, le=1.0, default=0.0)
    max_vol_percentile: float = Field(ge=0.0, le=1.0, default=1.0)
    stop_atr_multiple: float = Field(gt=0.0, default=1.0)
    min_stop_ticks: int = Field(gt=0, default=4)
    max_stop_atr_multiple: float = Field(gt=0.0, default=3.0)
    max_hold_bars: int = Field(gt=0, default=24)
    min_reward_risk: float = Field(
        gt=0.0, default=0.9, description="Reject the trade if achievable R:R falls below this."
    )
    require_orderflow_confirmation: bool = True
    require_structure_confirmation: bool = True

    @model_validator(mode="after")
    def _check(self) -> "SetupConfig":
        if self.min_vol_percentile >= self.max_vol_percentile:
            raise ValueError(
                f"{self.setup.value}: min_vol_percentile must be below max_vol_percentile"
            )
        if Regime.UNKNOWN in self.allowed_regimes:
            raise ValueError(
                f"{self.setup.value}: UNKNOWN regime is never tradable "
                "(it means the warmup window is incomplete)"
            )
        if not self.allowed_regimes:
            raise ValueError(f"{self.setup.value}: allowed_regimes must not be empty")
        expected = {
            SetupType.SCALP_1R: 1.0,
            SetupType.SETUP_2R: 2.0,
            SetupType.DIRECTIONAL_3R: 3.0,
        }[self.setup]
        if self.setup is SetupType.DIRECTIONAL_3R:
            if self.reward_risk < expected - 1e-9:
                raise ValueError(
                    f"DIRECTIONAL_3R must target at least 3R, got {self.reward_risk}"
                )
        elif abs(self.reward_risk - expected) > 0.05:
            raise ValueError(
                f"{self.setup.value} declares reward_risk={self.reward_risk}; "
                f"its name asserts {expected}R. Rename the setup or fix the ratio -- "
                "a mislabelled R target produces a flattering but meaningless win rate."
            )
        if self.min_reward_risk > self.reward_risk:
            raise ValueError(
                f"{self.setup.value}: min_reward_risk exceeds the setup's own reward_risk"
            )
        if self.max_stop_atr_multiple < self.stop_atr_multiple:
            raise ValueError(f"{self.setup.value}: max_stop_atr_multiple below stop_atr_multiple")
        return self


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class LevelSignificanceWeights(FrozenModel):
    """Weights for the level significance score S (ARCHITECTURE.md 14.3).

    "Major" is not a judgement in this system: it is `S >= s_major`.
    """

    touch_count: float = Field(ge=0.0, default=0.30)
    rejection_magnitude: float = Field(ge=0.0, default=0.20)
    volume_at_level: float = Field(ge=0.0, default=0.20)
    htf_confluence: float = Field(ge=0.0, default=0.15)
    age_decay: float = Field(ge=0.0, default=0.05)
    anchor_bonus: float = Field(ge=0.0, default=0.10)

    @model_validator(mode="after")
    def _check(self) -> "LevelSignificanceWeights":
        total = (
            self.touch_count + self.rejection_magnitude + self.volume_at_level
            + self.htf_confluence + self.age_decay + self.anchor_bonus
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"significance weights sum to {total}, expected 1.0")
        return self


class LevelCleanlinessWeights(FrozenModel):
    """Weights for the approach cleanliness score C (ARCHITECTURE.md 14.4).

    "Clean" is not a judgement either: it is `C >= c_min`.
    """

    approach_efficiency: float = Field(ge=0.0, default=0.30)
    recent_touch_density: float = Field(ge=0.0, default=0.25)
    bar_overlap: float = Field(ge=0.0, default=0.20)
    level_integrity: float = Field(ge=0.0, default=0.15)
    volatility_regularity: float = Field(ge=0.0, default=0.10)

    @model_validator(mode="after")
    def _check(self) -> "LevelCleanlinessWeights":
        total = (
            self.approach_efficiency + self.recent_touch_density + self.bar_overlap
            + self.level_integrity + self.volatility_regularity
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"cleanliness weights sum to {total}, expected 1.0")
        return self


class RejectionWeights(FrozenModel):
    """Weights for the rejection confirmation score R (ARCHITECTURE.md 14.5).

    `order_flow` is dropped WITHOUT redistribution when no tick feed exists,
    so these need not sum to 1 after that drop -- the lost weight is lost,
    which is the honest degradation path.
    """

    close_position: float = Field(ge=0.0, default=0.40)
    displacement: float = Field(ge=0.0, default=0.35)
    order_flow: float = Field(ge=0.0, default=0.25)

    @model_validator(mode="after")
    def _check(self) -> "RejectionWeights":
        total = self.close_position + self.displacement + self.order_flow
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rejection weights sum to {total}, expected 1.0")
        return self


class StructureMagnitudeWeights(FrozenModel):
    """Weights combining S, C and R into the STRUCTURE component's magnitude.

    ARCHITECTURE.md 14.6 names `w_S`, `w_C` and `w_R` but gave them no config
    field, so Phase 3 would have had to hardcode them -- which is exactly the
    situation `levels` exists to prevent.
    """

    significance: float = Field(ge=0.0, default=0.40)
    cleanliness: float = Field(ge=0.0, default=0.35)
    rejection: float = Field(ge=0.0, default=0.25)

    @model_validator(mode="after")
    def _check(self) -> "StructureMagnitudeWeights":
        total = self.significance + self.cleanliness + self.rejection
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"structure magnitude weights sum to {total}, expected 1.0")
        return self


class StructureLevelConfig(FrozenModel):
    """Support/resistance zone detection, scoring and gating.

    Every constant here is a hypothesis. This is the largest parameter
    surface in the project and therefore the largest overfitting risk, which
    is why `robustness.py` sweeps it against training folds only.
    """

    # --- pivot detection ---
    pivot_confirm_bars: int = Field(
        gt=0,
        default=3,
        description=(
            "Right-side bars required before a pivot is CONFIRMED. A swing "
            "high is not known until this many bars after it forms; a centred "
            "argmax window evaluated at t would read bars after t."
        ),
    )
    pivot_prominence_atr: float = Field(gt=0.0, default=0.40)

    # --- clustering ---
    zone_band_atr: float = Field(
        gt=0.0, default=0.25, description="Cluster merge distance, in ATR units."
    )
    min_zone_width_ticks: int = Field(gt=0, default=2)
    max_zones_tracked: int = Field(gt=0, default=24)
    lookback_bars: int = Field(gt=10, default=500)

    # --- anchors ---
    use_prior_session_levels: bool = True
    use_overnight_levels: bool = True
    use_opening_range: bool = True
    opening_range_minutes: float = Field(gt=0.0, default=30.0)
    use_vwap_levels: bool = True
    use_round_numbers: bool = True
    round_increment_points: dict[str, float] = Field(
        default_factory=lambda: {"NQ": 100.0, "ES": 25.0, "GC": 10.0, "QQQ": 5.0, "SPX": 25.0}
    )

    # --- significance ---
    significance_weights: LevelSignificanceWeights = Field(
        default_factory=LevelSignificanceWeights
    )
    touch_cap: int = Field(gt=0, default=4)
    touch_separation_atr: float = Field(gt=0.0, default=0.75)
    touch_separation_bars: int = Field(gt=0, default=5)
    rejection_horizon_bars: int = Field(gt=0, default=8)
    rejection_reference_atr: float = Field(gt=0.0, default=1.5)
    age_decay_lambda_bars: float = Field(gt=0.0, default=500.0)
    min_significance: float = Field(
        ge=0.0, le=1.0, default=0.60, description="The gate that defines 'major'."
    )

    # --- cleanliness ---
    cleanliness_weights: LevelCleanlinessWeights = Field(
        default_factory=LevelCleanlinessWeights
    )
    approach_bars: int = Field(gt=1, default=10)
    efficiency_reference: float = Field(gt=0.0, le=1.0, default=0.45)
    recent_window_bars: int = Field(gt=1, default=30)
    max_recent_touches: int = Field(gt=0, default=3)
    overlap_reference: float = Field(gt=0.0, le=1.0, default=0.70)
    failed_break_horizon_bars: int = Field(gt=0, default=4)
    min_cleanliness: float = Field(
        ge=0.0, le=1.0, default=0.55, description="The gate that defines 'clean'."
    )

    # --- rejection ---
    rejection_weights: RejectionWeights = Field(default_factory=RejectionWeights)
    magnitude_weights: StructureMagnitudeWeights = Field(
        default_factory=StructureMagnitudeWeights
    )
    atr_median_window: int = Field(
        gt=1,
        default=50,
        description="Window for the ATR median in the volatility-regularity term (14.4).",
    )
    displacement_reference_atr: float = Field(gt=0.0, default=0.50)
    require_close_back_outside: bool = Field(
        default=True,
        description=(
            "The binary rejection requirement: the bar's extreme entered the "
            "zone and the close returned outside it. Disabling this removes "
            "the only non-negotiable part of the trigger."
        ),
    )

    # --- entry / stop / target ---
    entry_atr_window: float = Field(
        gt=0.0,
        default=0.75,
        description="Price must be within this many ATR of the zone to consider entry.",
    )
    stop_buffer_atr: float = Field(
        gt=0.0,
        default=0.25,
        description="Stop placed beyond the zone edge AND beyond the rejection bar extreme.",
    )
    target_requires_major_zone: bool = Field(
        default=True,
        description=(
            "Target the next opposing zone with S >= min_significance. This is "
            "what makes setup selection a measurement rather than a parameter: "
            "achievable R:R is read off the structure, and a trade whose next "
            "level is too close is declined rather than retargeted."
        ),
    )
    fallback_target_atr: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "ATR-multiple target used when no opposing major zone exists. None "
            "means decline the trade instead, which is the default: inventing a "
            "target is how a structure-based setup silently becomes an "
            "arbitrary-R setup."
        ),
    )

    @model_validator(mode="after")
    def _check_non_negotiables(self) -> "StructureLevelConfig":
        """Refuse the combinations ARCHITECTURE section 14 calls non-negotiable.

        The config was able to switch off every gate the specification
        describes as required, which would leave a "structure-based" setup
        with no structural requirement at all -- and nothing in the code
        would have said so.
        """
        if not self.require_close_back_outside:
            raise ValueError(
                "require_close_back_outside=False removes the only non-negotiable "
                "part of the rejection trigger (section 14.5: the bar's extreme "
                "entered the zone AND the close returned outside it). Without it "
                "a 'rejection' is any bar that touched the level."
            )
        if self.min_significance <= 0.0 and self.min_cleanliness <= 0.0:
            raise ValueError(
                "min_significance and min_cleanliness are both zero, so every "
                "price cluster is both 'major' and 'clean' and section 14's two "
                "gates admit everything. Set at least one above zero."
            )
        if self.fallback_target_atr is not None and self.target_requires_major_zone:
            raise ValueError(
                "fallback_target_atr is set while target_requires_major_zone is "
                "True, which is contradictory: section 14.6 makes achievable R:R a "
                "measurement off the structure, and an ATR fallback turns a "
                "declined trade into an arbitrary-R trade. Choose one."
            )
        return self

    @model_validator(mode="after")
    def _check(self) -> "StructureLevelConfig":
        if self.approach_bars > self.recent_window_bars:
            raise ValueError(
                "approach_bars exceeds recent_window_bars: the approach window "
                "would extend beyond the window used to judge recent activity"
            )
        if self.recent_window_bars > self.lookback_bars:
            raise ValueError("recent_window_bars exceeds lookback_bars")
        if self.touch_cap > self.lookback_bars:
            raise ValueError("touch_cap exceeds lookback_bars")
        return self


class RiskConfig(FrozenModel):
    starting_equity: float = Field(gt=0.0, default=100_000.0)
    risk_per_trade_pct: float = Field(gt=0.0, default=0.005)
    max_risk_per_trade_pct: float = Field(gt=0.0, default=0.01)

    daily_loss_limit_pct: float = Field(gt=0.0, default=0.02)
    weekly_loss_limit_pct: float | None = Field(default=0.05, gt=0.0)
    max_consecutive_losses: int = Field(gt=0, default=4)
    cooldown_minutes_after_loss: float = Field(ge=0.0, default=0.0)
    cooldown_minutes_after_limit: float = Field(ge=0.0, default=1440.0)

    max_open_positions: int = Field(gt=0, default=1)
    max_portfolio_heat_pct: float = Field(
        gt=0.0, default=0.02, description="Cap on the sum of open risk."
    )
    max_positions_per_symbol: int = Field(gt=0, default=1)

    kill_switch_drawdown_pct: float = Field(
        gt=0.0, default=0.15, description="Halt all trading at this drawdown from high-water mark."
    )
    kill_switch_is_terminal: bool = Field(
        default=True, description="If true, the kill switch cannot be auto-reset within a run."
    )

    compound_equity: bool = Field(
        default=True,
        description=(
            "Size from current equity (compounding) vs starting equity (fixed). "
            "Materially changes the drawdown distribution, so it is explicit."
        ),
    )
    allow_fractional_contracts: bool = False

    @model_validator(mode="after")
    def _check(self) -> "RiskConfig":
        pcts = {
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_risk_per_trade_pct": self.max_risk_per_trade_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "max_portfolio_heat_pct": self.max_portfolio_heat_pct,
            "kill_switch_drawdown_pct": self.kill_switch_drawdown_pct,
        }
        for name, value in pcts.items():
            if value > MAX_SANE_PCT:
                raise ValueError(
                    f"{name}={value} exceeds {MAX_SANE_PCT}. Percentages are "
                    f"fractions: 0.5% is 0.005, not 0.5."
                )
        if self.risk_per_trade_pct > self.max_risk_per_trade_pct:
            raise ValueError(
                f"risk_per_trade_pct ({self.risk_per_trade_pct}) exceeds "
                f"max_risk_per_trade_pct ({self.max_risk_per_trade_pct})"
            )
        if self.max_portfolio_heat_pct < self.risk_per_trade_pct:
            raise ValueError(
                "max_portfolio_heat_pct is below risk_per_trade_pct: no trade could ever open"
            )
        if self.weekly_loss_limit_pct is not None:
            if self.weekly_loss_limit_pct < self.daily_loss_limit_pct:
                raise ValueError("weekly_loss_limit_pct is below daily_loss_limit_pct")
        if self.kill_switch_drawdown_pct <= self.daily_loss_limit_pct:
            raise ValueError(
                "kill_switch_drawdown_pct must exceed daily_loss_limit_pct, "
                "or the daily limit can never bind"
            )
        return self


# ---------------------------------------------------------------------------
# Execution / backtest
# ---------------------------------------------------------------------------


class ExecutionConfig(FrozenModel):
    """Cost and fill realism. Defaults are deliberately pessimistic."""

    use_quotes_when_available: bool = True
    spread_ticks_override: float | None = Field(default=None, gt=0.0)

    base_slippage_ticks: float = Field(ge=0.0, default=1.0)
    vol_slippage_coefficient: float = Field(
        ge=0.0,
        default=0.5,
        description="Extra slippage ticks per 1.0 of (ATR / median ATR - 1).",
    )
    stop_order_extra_slippage_ticks: float = Field(
        ge=0.0, default=1.0, description="Stops are market orders and pay more."
    )
    size_impact_ticks_per_unit: float = Field(ge=0.0, default=0.0)

    latency_ms: float = Field(ge=0.0, default=250.0)
    random_latency_ms: float = Field(
        ge=0.0, default=0.0, description="Uniform jitter added to latency (stress testing)."
    )

    max_participation_rate: float = Field(
        gt=0.0,
        le=1.0,
        default=0.05,
        description="Max fraction of a bar's volume we assume we can take.",
    )
    allow_partial_fills: bool = True
    min_partial_fraction: float = Field(
        gt=0.0, le=1.0, default=0.5, description="Below this fill fraction, abandon the entry."
    )

    entry_order_type: str = Field(default="market", pattern="^(market|limit|stop)$")
    max_entry_wait_bars: int = Field(ge=0, default=1)
    random_miss_probability: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Synthetic missed-trade rate for stress tests."
    )
    entry_delay_bars: int = Field(
        ge=0, default=0, description="Forced entry delay for robustness testing."
    )

    fill_gaps_at_open: bool = Field(
        default=True,
        description=(
            "If a bar gaps past the stop, fill at the bar open rather than the "
            "stop price. Disabling this overstates results."
        ),
    )
    pessimistic_same_bar: bool = Field(
        default=True,
        description=(
            "When a bar's range contains both stop and target and no intrabar "
            "data exists, resolve to the stop."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> "ExecutionConfig":
        if self.entry_order_type == "limit" and self.max_entry_wait_bars == 0:
            raise ValueError(
                "a limit entry with max_entry_wait_bars=0 can never fill"
            )
        return self


class SessionFilterConfig(FrozenModel):
    trade_rth_only: bool = True
    skip_first_minutes: float = Field(
        ge=0.0, default=5.0, description="Avoid the opening auction's artificial spreads."
    )
    skip_last_minutes: float = Field(ge=0.0, default=10.0)
    flatten_at_session_close: bool = True
    skip_holidays: bool = True
    skip_half_days: bool = True
    blackout_minutes_around_events: float = Field(
        ge=0.0, default=0.0, description="Economic-release blackout, when a calendar is supplied."
    )


class BacktestConfig(FrozenModel):
    symbols: tuple[str, ...] = ("NQ", "ES")
    start: date = date(2015, 1, 1)
    end: date = date(2025, 1, 1)
    seed: int = DEFAULT_SEED
    record_wait_signals: bool = Field(
        default=True,
        description="Persist WAITs so gate-rejection rates can be analysed.",
    )
    record_equity_every_bar: bool = True
    fail_on_lookahead_assertion: bool = True
    max_bars: int | None = Field(default=None, gt=0, description="Debug cap.")

    @model_validator(mode="after")
    def _check(self) -> "BacktestConfig":
        if self.end <= self.start:
            raise ValueError("backtest end must be after start")
        if not self.symbols:
            raise ValueError("at least one symbol must be configured")
        return self


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class WalkForwardConfig(FrozenModel):
    train_months: int = Field(gt=0, default=36)
    validation_months: int = Field(gt=0, default=12)
    test_months: int = Field(gt=0, default=12)
    step_months: int = Field(gt=0, default=12)
    anchored: bool = Field(
        default=False, description="True keeps the train start fixed and grows the window."
    )
    embargo_days: int = Field(
        ge=0,
        default=5,
        description=(
            "Gap inserted between windows so a position open across a "
            "boundary cannot leak information into the next window."
        ),
    )
    purge_days: int = Field(
        ge=0, default=1, description="Bars dropped at the end of a window (label overlap purge)."
    )
    min_trades_per_fold: int = Field(
        ge=0, default=30, description="Folds below this are reported as INSUFFICIENT, not averaged in."
    )

    @model_validator(mode="after")
    def _check(self) -> "WalkForwardConfig":
        if self.step_months > self.test_months:
            raise ValueError(
                "step_months exceeds test_months: out-of-sample periods would be skipped"
            )
        return self


class SealConfig(FrozenModel):
    """The untouched final out-of-sample window."""

    enabled: bool = True
    sealed_start: date = date(2023, 1, 1)
    sealed_end: date = date(2025, 1, 1)
    max_opens: int = Field(
        ge=1,
        default=1,
        description=(
            "How many times the seal may be opened. Default 1: one final "
            "confirmation run. Further opens raise."
        ),
    )
    audit_path: str = "reports/seal_audit.jsonl"

    @model_validator(mode="after")
    def _check(self) -> "SealConfig":
        if self.sealed_end <= self.sealed_start:
            raise ValueError("sealed_end must be after sealed_start")
        return self


class MonteCarloConfig(FrozenModel):
    n_simulations: int = Field(ge=100, default=10_000)
    method: MonteCarloMethod = MonteCarloMethod.BLOCK
    block_length: int = Field(gt=0, default=20)
    seed: int = DEFAULT_SEED
    compound: bool = True
    drawdown_thresholds: tuple[float, ...] = (0.10, 0.20, 0.30)
    streak_thresholds: tuple[int, ...] = (5, 10, 20)
    percentiles: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0)
    n_curves_to_store: int = Field(ge=0, default=200)
    trades_per_simulation: int | None = Field(
        default=None, gt=0, description="Defaults to the observed trade count."
    )

    @model_validator(mode="after")
    def _check(self) -> "MonteCarloConfig":
        if not all(0.0 < p < 100.0 for p in self.percentiles):
            raise ValueError("percentiles must lie strictly between 0 and 100")
        if list(self.percentiles) != sorted(self.percentiles):
            raise ValueError("percentiles must be sorted ascending")
        if not all(0.0 < d < 1.0 for d in self.drawdown_thresholds):
            raise ValueError("drawdown_thresholds are fractions in (0, 1)")
        return self


class RobustnessConfig(FrozenModel):
    flow_score_thresholds: tuple[float, ...] = (60.0, 65.0, 70.0, 75.0, 80.0, 85.0)
    risk_pcts: tuple[float, ...] = (0.0025, 0.005, 0.0075, 0.01)
    slippage_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    spread_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    entry_delay_bars: tuple[int, ...] = (0, 1, 2)
    miss_probabilities: tuple[float, ...] = (0.0, 0.05, 0.10)
    win_rate_haircuts: tuple[float, ...] = (0.0, 0.05, 0.10)
    fragility_profit_factor_floor: float = Field(
        gt=0.0,
        default=1.1,
        description="A strategy is flagged FRAGILE if any neighbouring parameter drops it below this.",
    )
    fragility_max_relative_drop: float = Field(
        gt=0.0,
        default=0.40,
        description="Flag FRAGILE if expectancy falls more than this fraction on a one-step parameter change.",
    )


class AnalyticsConfig(FrozenModel):
    risk_free_rate_annual: float = Field(ge=0.0, default=0.04)
    trading_days_per_year: int = Field(gt=0, default=252)
    min_sample_n: int = Field(
        ge=1,
        default=30,
        description=(
            "Breakdown cells below this N are reported INSUFFICIENT rather "
            "than shown as a win rate. Small-sample win rates are the main "
            "way a researcher fools themselves."
        ),
    )
    flow_score_buckets: tuple[float, ...] = (0.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 100.0)
    allow_mixed_split_aggregation: bool = Field(
        default=False,
        description="If false, metrics refuse to combine IS and OOS trades in one number.",
    )

    @model_validator(mode="after")
    def _check(self) -> "AnalyticsConfig":
        if list(self.flow_score_buckets) != sorted(self.flow_score_buckets):
            raise ValueError("flow_score_buckets must be ascending")
        return self


class ConsistencyConfig(FrozenModel):
    """Weights for the consistency score.

    The brief prefers 72% stable over 90% in one window, so the score
    rewards low dispersion across periods and regimes, not peak performance.
    """

    weight_oos_win_rate_stability: float = Field(ge=0.0, default=0.20)
    weight_profit_factor_stability: float = Field(ge=0.0, default=0.20)
    weight_drawdown_stability: float = Field(ge=0.0, default=0.15)
    weight_expectancy_stability: float = Field(ge=0.0, default=0.20)
    weight_regime_stability: float = Field(ge=0.0, default=0.15)
    weight_year_stability: float = Field(ge=0.0, default=0.10)

    @model_validator(mode="after")
    def _check(self) -> "ConsistencyConfig":
        total = (
            self.weight_oos_win_rate_stability
            + self.weight_profit_factor_stability
            + self.weight_drawdown_stability
            + self.weight_expectancy_stability
            + self.weight_regime_stability
            + self.weight_year_stability
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"consistency weights sum to {total}, expected 1.0")
        return self


class ResearchTargets(FrozenModel):
    """Targets from the research brief, recorded as falsifiable hypotheses.

    READ-ONLY CONTEXT: these values must never influence signal generation,
    sizing, parameter selection, or any optimizer. They exist so the final
    report can state, per target, whether the data met or refuted it.
    `tests/unit/test_no_target_leakage.py` fails the build if any module
    outside `analytics`/`validation`/`config` references them.
    """

    scalp_1r_win_rate_favourable_regimes: tuple[float, float] = (0.88, 0.92)
    expected_chop_win_rate_reduction: float = 0.05
    combined_10y_win_rate: float = 0.72
    note: str = (
        "Hypotheses from the brief, not objectives. A symmetric 1R system at "
        "88-92% implies an annualized Sharpe above 25 at 500 trades/year, "
        "which is far outside anything documented; it is recorded here to be "
        "tested and most likely refuted. Report measured values, never these."
    )


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


class PathsConfig(FrozenModel):
    reports_dir: str = "reports"
    database_path: str = "reports/flow_model.db"
    experiment_db_path: str = "reports/experiments.db"
    log_dir: str = "reports/logs"
    cache_dir: str = "data_cache"


class LoggingConfig(FrozenModel):
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    json_format: bool = False
    log_to_file: bool = True
    console: bool = True
    include_run_id: bool = True


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class FlowModelConfig(FrozenModel):
    """Root configuration. `config_hash` identifies it in the experiment log."""

    name: str = "flow_model_default"
    description: str = ""
    seed: int = DEFAULT_SEED

    instruments: dict[str, InstrumentSpec] = Field(default_factory=dict)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    flow_score: FlowScoreConfig = Field(default_factory=FlowScoreConfig)
    levels: StructureLevelConfig = Field(default_factory=StructureLevelConfig)
    setups: dict[SetupType, SetupConfig] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    session: SessionFilterConfig = Field(default_factory=SessionFilterConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    seal: SealConfig = Field(default_factory=SealConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    research_targets: ResearchTargets = Field(default_factory=ResearchTargets)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def config_hash(self) -> str:
        """Content hash of everything that can change a result.

        `name`, `description`, `logging` and `paths` are excluded: renaming a
        config or changing a log level must not create a new experiment
        identity, or the experiment log fills with spurious entries.
        """
        payload = self.to_init_dict()
        for key in ("name", "description", "logging", "paths", "research_targets"):
            payload.pop(key, None)
        return stable_hash(payload)

    def spec(self, symbol: str) -> InstrumentSpec:
        try:
            return self.instruments[symbol]
        except KeyError:
            raise KeyError(
                f"no InstrumentSpec for {symbol!r}; known: {sorted(self.instruments)}"
            ) from None

    def setup(self, setup: SetupType) -> SetupConfig:
        try:
            return self.setups[setup]
        except KeyError:
            raise KeyError(f"no SetupConfig for {setup.value}") from None

    def enabled_setups(self) -> tuple[SetupConfig, ...]:
        return tuple(s for s in self.setups.values() if s.enabled)

    @model_validator(mode="after")
    def _check(self) -> "FlowModelConfig":
        for symbol in self.backtest.symbols:
            if symbol not in self.instruments:
                raise ValueError(
                    f"backtest symbol {symbol!r} has no InstrumentSpec "
                    f"(known: {sorted(self.instruments)})"
                )
            spec = self.instruments[symbol]
            if not spec.tradable:
                raise ValueError(
                    f"backtest symbol {symbol!r} is marked tradable=False "
                    "(reference-only instrument); remove it from backtest.symbols "
                    "or give it an execution_proxy"
                )
        for key, spec in self.instruments.items():
            if key != spec.symbol:
                raise ValueError(
                    f"instrument registry key {key!r} does not match spec.symbol {spec.symbol!r}"
                )
            if spec.execution_proxy and spec.execution_proxy not in self.instruments:
                raise ValueError(
                    f"{key}: execution_proxy {spec.execution_proxy!r} is not a known instrument"
                )
        for key, setup in self.setups.items():
            if key != setup.setup:
                raise ValueError(
                    f"setup registry key {key} does not match setup.setup {setup.setup}"
                )
        if not self.enabled_setups():
            raise ValueError("no setups are enabled; the system could never trade")
        if self.seal.enabled:
            if self.seal.sealed_start < self.backtest.start:
                raise ValueError("sealed window starts before the backtest period")
            if self.seal.sealed_end > self.backtest.end:
                raise ValueError(
                    "sealed window ends after the backtest period; it would never be reachable"
                )
        return self
