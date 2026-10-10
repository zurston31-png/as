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

from collections.abc import Iterable
from datetime import date
from typing import Any

from pydantic import Field, computed_field, field_validator, model_validator

from flow_model.core.determinism import DEFAULT_SEED, stable_hash
from flow_model.core.enums import (
    Component,
    DataQuality,
    Feed,
    KronosContamination,
    KronosMode,
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

    # Phase 1 also parked `cvd_period`, `absorption_lookback` and
    # `options_lookback_days` here as placeholders. They now live in
    # `OrderFlowConfig` and `OptionsFlowConfig`, which own the Phases 4 and 5
    # parameter surfaces. Two homes for one window is one too many: the pair
    # loads cleanly, and the computer reads whichever the author happened to
    # remember.

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
    cusum_window_bars: int = Field(
        gt=2,
        default=60,
        description=(
            "Returns in the CUSUM window. The statistic standardizes the "
            "returns inside this window with the window's own mean and "
            "stdev and accumulates from zero at its start, so the window "
            "sets both the estimation sample and the accumulation span. "
            "The default 60 is a little under one 78-bar RTH session at a "
            "5-minute interval: long enough that the stdev is estimated "
            "from 60 observations (relative standard error ~1/sqrt(2n) = "
            "9%) and short enough that a detected break is local rather "
            "than something that happened a session ago. Added in Phase 3 "
            "because section 6 names a CUSUM statistic without naming its "
            "window; see regime/detector.py for what the length implies "
            "about the smallest detectable shift."
        ),
    )
    hysteresis_search_bars: int = Field(
        ge=1,
        default=24,
        description=(
            "How far back the hysteresis rule looks for the most recent "
            "confirmed run of `min_regime_bars` identical raw labels. This "
            "is the horizon that replaces mutable detector state: the "
            "confirmed label is a pure function of the raw labels in this "
            "trailing window, so it cannot depend on call order. A bar "
            "with no completed run anywhere in the window is reported as "
            "UNKNOWN rather than carrying a label forward indefinitely. "
            "The default is 8x the default `min_regime_bars`. Added in "
            "Phase 3; see regime/detector.py."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> "RegimeConfig":
        if self.low_vol_percentile >= self.high_vol_percentile:
            raise ValueError("low_vol_percentile must be below high_vol_percentile")
        if self.hysteresis_search_bars < self.min_regime_bars:
            raise ValueError(
                f"hysteresis_search_bars={self.hysteresis_search_bars} is shorter "
                f"than min_regime_bars={self.min_regime_bars}; no run of "
                "min_regime_bars identical labels could ever fit in the search "
                "window, so every bar would be reported UNKNOWN"
            )
        return self


# ---------------------------------------------------------------------------
# Order flow and options flow (Phases 4 and 5)
# ---------------------------------------------------------------------------

#: Aggressor-classification methods that derive a side from BAR data rather
#: than from trades. ARCHITECTURE.md section 5 calls bar-volume-only "delta" a
#: known-bad estimator and refuses to substitute it silently; these are the
#: names such an estimator ships under. `OrderFlowConfig` refuses to drop any
#: of them from its denylist, because a denylist that can be emptied is not a
#: rule -- it is a default.
BAR_VOLUME_DELTA_PROXIES: frozenset[str] = frozenset(
    {
        "bar_volume",
        "bar_volume_tick_rule",
        "tick_rule_on_bars",
        "uptick_downtick_on_bars",
        "volume_split",
    }
)


class OrderFlowWeights(FrozenModel):
    """Weights combining the five order-flow terms into the component magnitude.

    ARCHITECTURE.md section 5 names the five features and gives them no
    weights, so Phase 4 would otherwise hardcode `w_*` -- exactly the
    situation `StructureMagnitudeWeights` was added to prevent.

    A term whose input is absent is DROPPED WITHOUT REDISTRIBUTION, the rule
    section 14.5 states for `s_flow`, applied to the whole component: the
    lost weight stays lost, so missing data lowers the sub-score instead of
    being reallocated into confidence the tape does not support.
    `OrderFlowConfig.min_available_weight_fraction` is the floor below which
    the component must report UNAVAILABLE rather than a small number that
    reads like "no flow".

    The split below is a declared prior, not a measurement. Nothing has been
    fitted; it is swept in Phase 8 on training folds only.
    """

    signed_delta: float = Field(ge=0.0, default=0.25)
    cvd_slope: float = Field(ge=0.0, default=0.25)
    aggression_ratio: float = Field(ge=0.0, default=0.20)
    absorption: float = Field(ge=0.0, default=0.15)
    trade_size_distribution: float = Field(ge=0.0, default=0.15)

    @model_validator(mode="after")
    def _check(self) -> "OrderFlowWeights":
        total = (
            self.signed_delta + self.cvd_slope + self.aggression_ratio
            + self.absorption + self.trade_size_distribution
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"order-flow weights sum to {total}, expected 1.0")
        return self


class OrderFlowConfig(FrozenModel):
    """Order-flow features (ARCHITECTURE.md section 5, Phase 4).

    Required data is tick-level trades carrying an aggressor side. There is
    **no degraded path**: section 5 accepts no proxy, so an absent or
    unclassified tick feed disables the component and reports its 25 points
    UNAVAILABLE, without redistribution. The switches that hold that line are
    `allow_bar_volume_delta_proxy` and `require_aggressor_classification`,
    and `_check_non_negotiables` refuses to let either be turned off.

    Every window and threshold here is a hypothesis. None was chosen by
    looking at a result; most are judgement calls anchored on a prior already
    declared elsewhere in this file, and the few that carry forward a Phase-1
    placeholder say so.
    """

    # --- signed delta ---
    delta_smoothing_bars: int = Field(
        gt=0,
        default=3,
        description=(
            "Bars of per-bar delta averaged before the signed-delta reading "
            "is taken. One 5-minute bar's delta is a small sample of a noisy "
            "quantity, and a feature that flips sign every bar cannot confirm "
            "a direction. 1 disables the smoothing, which is the honest way "
            "to measure what the smoothing is worth."
        ),
    )
    delta_percentile_lookback: int = Field(
        gt=10,
        default=60,
        description=(
            "Trailing window the signed delta is percentile-ranked within, so "
            "'large delta' means large for THIS symbol's recent tape rather "
            "than a contract count that is meaningless across NQ, GC and QQQ. "
            "A percentile rank is used instead of a z-score because the score "
            "is a weighted sum and an unbounded term lets one print dominate "
            "it (features/base.py). 60 matches the liquidity layer's "
            "`volume_percentile_lookback`, so the two percentile features "
            "describe the same stretch of tape."
        ),
    )

    # --- CVD slope ---
    cvd_lookback_bars: int = Field(
        gt=2,
        default=20,
        description=(
            "Bars of delta cumulated into the CVD path whose slope is the "
            "feature. One window, not two: cumulating over one span and "
            "regressing over another makes the reported slope depend on a "
            "relationship between two constants that nobody tuned on purpose. "
            "Carried from Phase 1's `FeatureConfig.cvd_period`, which was "
            "itself a declared prior."
        ),
    )
    cvd_slope_squash_scale: float = Field(
        gt=0.0,
        default=0.25,
        description=(
            "The DIMENSIONLESS CVD slope -- contracts per bar divided by the "
            "window's own mean classified volume per bar -- that `squash` maps "
            "to about 0.76. Dividing by the window's own volume is what makes "
            "the term comparable across instruments and across volume "
            "regimes; the scale then only sets how quickly it saturates. 0.25 "
            "means a CVD climbing by a quarter of a typical bar's classified "
            "volume per bar already reads as strong, which is a guess."
        ),
    )

    # --- aggression ratio ---
    aggression_lookback_bars: int = Field(
        gt=0,
        default=10,
        description=(
            "Bars pooled before the aggression ratio is formed. The ratio is "
            "built from TRADE COUNTS (buy_trades vs sell_trades), which is a "
            "different measurement from `TickAggregate.delta_ratio`'s "
            "volume weighting: many small lifts and one large one give the "
            "same delta and a very different count ratio. Counts in a single "
            "bar are few, so they are pooled."
        ),
    )
    min_classified_trades: int = Field(
        gt=0,
        default=20,
        description=(
            "Classified trades required in the pooled window before the "
            "aggression ratio is reported at all. Below it the term is "
            "dropped, not neutralized: a 3-trade ratio of 2:1 is noise, and "
            "emitting 0.67 for it is indistinguishable downstream from a real "
            "two-thirds reading."
        ),
    )
    min_classification_coverage: float = Field(
        gt=0.0,
        le=1.0,
        default=0.60,
        description=(
            "Fraction of window volume that must carry a known aggressor "
            "(`TickAggregate.classification_coverage`) before any order-flow "
            "feature is reported. Below it the component is disabled, because "
            "a delta whose sign is set by a minority of the tape is the "
            "known-bad estimator section 5 refuses under another name. The "
            "bound is `gt=0.0` rather than `ge=0.0` for that reason -- 0.0 "
            "would accept a feed with no classification at all. 0.60 is a "
            "judgement call and a Phase 8 sweep candidate."
        ),
    )

    # --- absorption at a level ---
    absorption_lookback_bars: int = Field(
        gt=0,
        default=10,
        description=(
            "Bars over which absorption is measured: sustained one-sided "
            "delta that does NOT move price. Carried from Phase 1's "
            "`FeatureConfig.absorption_lookback`."
        ),
    )
    absorption_delta_percentile: float = Field(
        gt=0.0,
        lt=1.0,
        default=0.80,
        description=(
            "The window's summed delta must rank at or above this percentile "
            "of its own history before absorption is considered, so 'heavy "
            "one-sided flow' is relative to the symbol. 0.80 is the "
            "percentile ARCHITECTURE section 6 already uses for "
            "`high_vol_percentile`; reusing it keeps one notion of 'extreme' "
            "in the system rather than inventing a second."
        ),
    )
    absorption_max_displacement_atr: float = Field(
        gt=0.0,
        default=0.25,
        description=(
            "Net price change over the window, in ATR units, below which the "
            "flow counts as absorbed. This is the 'result' half of "
            "effort-versus-result: heavy delta WITH displacement is a drive, "
            "not absorption, and scoring both the same way would make the "
            "term unreadable. 0.25 mirrors `levels.zone_band_atr`."
        ),
    )
    absorption_level_window_atr: float = Field(
        gt=0.0,
        default=0.25,
        description=(
            "Band, in ATR units, within which the window's closes must stay "
            "for the window to count as 'at one level'. Section 5 says "
            "'absorption at level', and this is how that is measured WITHOUT "
            "the order-flow computer taking a dependency on the structure "
            "layer's zone list: a computer that needed another computer's "
            "output could not be audited for lookahead in isolation, which is "
            "how every other FeatureComputer in the package is audited. The "
            "structure layer still gates on its own zones in `StructureGate`; "
            "this term only reports that flow was absorbed somewhere flat."
        ),
    )

    # --- trade-size distribution ---
    trade_size_percentile_lookback: int = Field(
        gt=10,
        default=120,
        description=(
            "Trailing window for percentile-ranking average and maximum trade "
            "size. Longer than `delta_percentile_lookback` because a size "
            "distribution's tail needs more observations than a volume "
            "percentile does before its 90th percentile means anything. 120 "
            "is a judgement call."
        ),
    )
    large_trade_percentile: float = Field(
        gt=0.0,
        lt=1.0,
        default=0.90,
        description=(
            "Percentile of its own history at or above which a window's "
            "maximum trade size counts as a large-trade event. NOTE on what "
            "is NOT here: bucket EDGES were considered and rejected. "
            "`TickAggregate` carries `max_trade_size` and trade counts, not a "
            "size histogram, so absolute edges would describe a measurement "
            "the feed does not supply -- and in contracts they would be "
            "instrument-specific besides. A relative threshold measures what "
            "the data actually contains. A large print is an observation "
            "about size, not evidence about who traded."
        ),
    )

    # --- combination and availability ---
    weights: OrderFlowWeights = Field(default_factory=OrderFlowWeights)
    min_available_weight_fraction: float = Field(
        gt=0.0,
        le=1.0,
        default=0.50,
        description=(
            "Terms carrying at least this fraction of the total weight must "
            "have real inputs, or the component reports UNAVAILABLE. Without "
            "the floor, dropping terms without redistribution (the honest "
            "rule) produces a systematically small sub-score that reads "
            "downstream as 'the tape is balanced' when it actually means "
            "'most of this was never measured'. Principle 4: missing data "
            "produces WAIT, never a guess."
        ),
    )

    # --- the non-negotiables ---
    allow_bar_volume_delta_proxy: bool = Field(
        default=False,
        description=(
            "The switch that governs the no-proxy rule. Section 5: a delta "
            "estimated from bar volume is a known-bad estimator and is NOT "
            "substituted when the tick feed is missing. Enabling it is "
            "refused by `_check_non_negotiables`; the field exists so the "
            "rule is visible and testable in config rather than implied by "
            "the absence of code."
        ),
    )
    require_aggressor_classification: bool = Field(
        default=True,
        description=(
            "A tick feed must declare HOW it classified the aggressor "
            "(`TickSeries.classification_method`). The series default is "
            "'unknown', and an undeclared method is not a classification: "
            "volume with an unexplained side is bar volume with a label. "
            "Disabling this is refused."
        ),
    )
    rejected_classification_methods: tuple[str, ...] = Field(
        default=(
            "bar_volume",
            "bar_volume_tick_rule",
            "tick_rule_on_bars",
            "uptick_downtick_on_bars",
            "volume_split",
        ),
        description=(
            "Classification methods that disable the component on sight. A "
            "denylist rather than an allowlist, because the honest methods "
            "are open-ended ('bid_ask', an exchange's own tag) while the "
            "known-bad ones are a short named set. Entries are lower-cased on "
            "load so the comparison is unambiguous; every name in "
            "`BAR_VOLUME_DELTA_PROXIES` must remain present."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warmup_bars(self) -> int:
        """A FLOOR for the warmup a Phase-4 computer may declare.

        Composed windows, not the longest single window: ranking a
        `delta_smoothing_bars` average inside a `delta_percentile_lookback`
        window reads `lookback + smoothing - 1` bars, not `lookback`. That
        arithmetic is the whole content of this property, because getting it
        wrong is a HIGH-severity bug that the lookahead audit cannot see --
        the regime detector declared 268 and read 291, which made the label at
        bar t depend on where the caller started loading while nothing read
        the future.

        It is a floor and not an answer, in the same way
        `FeatureConfig.warmup_bars` is (see the note at the top of
        features/volatility.py). A computer that composes these windows more
        deeply must declare more. None may declare less.
        """
        return max(
            self.delta_percentile_lookback + self.delta_smoothing_bars - 1,
            self.delta_percentile_lookback + self.absorption_lookback_bars - 1,
            self.trade_size_percentile_lookback + self.aggression_lookback_bars - 1,
            self.cvd_lookback_bars,
        ) + 1

    @field_validator("rejected_classification_methods", mode="before")
    @classmethod
    def _normalize_methods(cls, value: Any) -> Any:
        """Lower-case and de-duplicate, order preserved.

        The subset check below and the consumer's membership test have to
        agree on case, and 'Bar_Volume' slipping past a denylist is the one
        failure this field exists to prevent.
        """
        if isinstance(value, str) or not isinstance(value, Iterable):
            return value
        seen: dict[str, None] = {}
        for item in value:
            if not isinstance(item, str):
                return value
            seen.setdefault(item.strip().lower(), None)
        return tuple(seen)

    @model_validator(mode="after")
    def _check_non_negotiables(self) -> "OrderFlowConfig":
        """Refuse the configurations ARCHITECTURE section 5 calls impossible.

        Section 5's order-flow row is the only one in the table with no
        degraded path at all. That is a strong claim, and a config able to
        quietly undo it would make the claim decorative.
        """
        if self.allow_bar_volume_delta_proxy:
            raise ValueError(
                "allow_bar_volume_delta_proxy=True enables a bar-volume-derived "
                "delta proxy. ARCHITECTURE.md section 5 accepts NO proxy for "
                "order flow: bar-volume-only 'delta' is a known-bad estimator, "
                "and substituting it would be indistinguishable from real "
                "order flow to every caller above -- including the lookahead "
                "audit, which would pass. The component is disabled and its 25 "
                "points are reported UNAVAILABLE instead, without "
                "redistribution."
            )
        if not self.require_aggressor_classification:
            raise ValueError(
                "require_aggressor_classification=False accepts a tick feed "
                "that does not say how it assigned the aggressor side. An "
                "undeclared method ('unknown', the TickSeries default) is not "
                "a classification, so this is the bar-volume proxy admitted "
                "through the back door."
            )
        missing = sorted(BAR_VOLUME_DELTA_PROXIES - set(self.rejected_classification_methods))
        if missing:
            raise ValueError(
                f"rejected_classification_methods no longer rejects {missing}. "
                "These are the names a bar-volume-derived delta ships under; "
                "removing one admits the proxy that allow_bar_volume_delta_proxy "
                "exists to refuse. A denylist that can be emptied is a default, "
                "not a rule."
            )
        return self

    @model_validator(mode="after")
    def _check(self) -> "OrderFlowConfig":
        if self.delta_smoothing_bars > self.delta_percentile_lookback:
            raise ValueError(
                f"delta_smoothing_bars={self.delta_smoothing_bars} exceeds "
                f"delta_percentile_lookback={self.delta_percentile_lookback}: the "
                "smoothed value would be ranked against fewer observations than "
                "it is built from, so its percentile would be nearly constant"
            )
        if self.absorption_lookback_bars >= self.delta_percentile_lookback:
            raise ValueError(
                f"absorption_lookback_bars={self.absorption_lookback_bars} is not "
                f"shorter than delta_percentile_lookback="
                f"{self.delta_percentile_lookback}: absorption_delta_percentile "
                "ranks the absorption window's own delta against that history, "
                "and a window cannot be an extreme of a history no longer than "
                "itself"
            )
        if self.aggression_lookback_bars >= self.trade_size_percentile_lookback:
            raise ValueError(
                f"aggression_lookback_bars={self.aggression_lookback_bars} is not "
                f"shorter than trade_size_percentile_lookback="
                f"{self.trade_size_percentile_lookback}: the pooled trade-size "
                "statistic would be ranked against a history no longer than the "
                "pool it came from"
            )
        return self


class OptionsFlowWeights(FrozenModel):
    """Weights combining the five options-flow terms (section 5, Phase 5).

    Same reasoning as `OrderFlowWeights`: section 5 names the features and no
    weights, and a term whose input is `None` is dropped WITHOUT
    redistribution rather than read as zero. `OptionsSnapshot` is explicit
    that `None` means NOT SUPPLIED while 0.0 is a real observation, so a
    dropped term and a zero term must not produce the same sub-score.

    Four of the five inputs are optional in the contract; only net premium is
    always present. With the defaults below, net premium alone carries 0.25,
    which is under `OptionsFlowConfig.min_available_weight_fraction` -- so a
    chain supplying premium and nothing else reports UNAVAILABLE rather than a
    quarter-strength opinion.

    None of these numbers is a finding.
    """

    net_premium: float = Field(ge=0.0, default=0.25)
    delta_weighted_volume: float = Field(ge=0.0, default=0.25)
    oi_change: float = Field(ge=0.0, default=0.20)
    skew_25d: float = Field(ge=0.0, default=0.15)
    gamma_exposure: float = Field(ge=0.0, default=0.15)

    @model_validator(mode="after")
    def _check(self) -> "OptionsFlowWeights":
        total = (
            self.net_premium + self.delta_weighted_volume + self.oi_change
            + self.skew_25d + self.gamma_exposure
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"options-flow weights sum to {total}, expected 1.0")
        return self


class OptionsFlowConfig(FrozenModel):
    """Options-flow features (ARCHITECTURE.md section 5, Phase 5).

    Ideal data is OPRA trade prints; the degraded path is an end-of-day chain
    plus open interest, which section 5 grades DEGRADED and caps. The cap is
    `eod_degraded_cap_fraction` and the measurement that triggers it is
    `OptionsSnapshot.is_intraday`, not a config guess about the vendor.

    What this component is NOT: section 5 explicitly rejects reading options
    flow as evidence of institutional intent. Premium has an ambiguous sign --
    a large call print may be an opening bet, a closing sale or a hedge leg --
    so every field here measures IMBALANCE and positioning, and no field
    names an actor. Nothing is ever synthesized to fill a gap
    (`never_synthesize_missing_fields`).
    """

    # --- normalization window ---
    lookback_snapshots: int = Field(
        gt=2,
        default=20,
        description=(
            "Trailing OPTIONS SNAPSHOTS every term is percentile-ranked or "
            "differenced against. Counted in snapshots, not days: the feed's "
            "cadence is whatever it is -- EOD chains give roughly one per "
            "session, OPRA prints give many -- and converting to days would "
            "require this section to assume a cadence it cannot observe. "
            "Carried from Phase 1's `FeatureConfig.options_lookback_days`, "
            "whose name asserted the cadence this one does not. One window "
            "serves all five terms so that the Phase 8 sweep moves one knob "
            "rather than five correlated ones."
        ),
    )
    min_snapshot_volume: float = Field(
        ge=0.0,
        default=1.0,
        description=(
            "Total option volume (calls plus puts) a snapshot must carry "
            "before its imbalances are reported. `premium_imbalance` and "
            "`put_call_volume_ratio` are ratios, and `safe_divide` turns a "
            "zero denominator into 0.0 -- which is a NEUTRAL reading, "
            "indistinguishable from a genuinely balanced chain. This gate "
            "turns 'nothing traded' into DEGRADED instead."
        ),
    )

    # --- term scaling ---
    oi_change_squash_scale: float = Field(
        gt=0.0,
        default=0.05,
        description=(
            "Net open-interest change (call minus put) as a FRACTION of total "
            "open interest that `squash` maps to about 0.76. Expressed as a "
            "fraction so it is comparable across underlyings and across the "
            "growth of a chain over ten years; a contract count would not be. "
            "0.05 says a 5% one-session shift in net OI already reads as "
            "large, which is a judgement call."
        ),
    )
    use_gamma_exposure_proxy: bool = Field(
        default=True,
        description=(
            "Whether the gamma-exposure proxy term is computed. An ablation "
            "switch for Phase 8, not a data switch: absence of the input is "
            "already handled by `OptionsSnapshot.gamma_exposure_proxy` being "
            "None. It is a PROXY -- an OI-and-price construction, not a "
            "dealer inventory, which nobody outside a dealer can observe. "
            "Turning it off requires zeroing its weight (see `_check`)."
        ),
    )

    # --- combination and availability ---
    weights: OptionsFlowWeights = Field(default_factory=OptionsFlowWeights)
    min_available_weight_fraction: float = Field(
        gt=0.0,
        le=1.0,
        default=0.50,
        description=(
            "Terms carrying at least this fraction of the total weight must "
            "have non-None inputs, or the component reports UNAVAILABLE. Four "
            "of the five inputs are optional in `OptionsSnapshot`, so without "
            "this floor a chain carrying premium alone would still produce a "
            "number, and that number would be small for lack of data while "
            "reading as 'options flow is neutral'."
        ),
    )

    # --- the degraded path ---
    eod_degraded_cap_fraction: float = Field(
        gt=0.0,
        default=0.50,
        description=(
            "Cap on the options sub-score when the chain is end-of-day only "
            "(`OptionsSnapshot.is_intraday` is False), as a fraction of the "
            "component's full weight. Section 5 says the sub-score is 'capped "
            "and flagged' on that path and names no number; 0.50 is a "
            "judgement call. A FRACTION rather than absolute points so the "
            "cap survives the Phase 8 weight sweep: a cap of 10 points stops "
            "binding the moment the sweep lowers the component's weight to 8, "
            "and nothing would have said so. `_check_cap_can_bind` refuses a "
            "fraction at or above 1.0, which is not a cap."
        ),
    )
    require_intraday_prints: bool = Field(
        default=False,
        description=(
            "If True, an EOD-only chain disables the component outright "
            "instead of capping it. False follows section 5, which keeps the "
            "degraded path and caps it -- EOD data can still carry "
            "positioning, just not flow TIMING. True is the stricter research "
            "choice and makes the cap moot; it is offered because 'is a capped "
            "EOD sub-score worth anything' is a question for measurement, not "
            "for this docstring."
        ),
    )
    never_synthesize_missing_fields: bool = Field(
        default=True,
        description=(
            "Absent fields stay absent. Section 5: options data is 'Never "
            "synthesized.' An interpolated IV surface or a back-filled OI "
            "change is a fabricated observation, and a feature built on one "
            "cannot be distinguished downstream from a measured feature. "
            "Disabling this is refused."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warmup_snapshots(self) -> int:
        """Snapshots required before any options term is valid.

        In SNAPSHOTS, deliberately. A Phase-5 computer declares `warmup_bars`
        in bars, and the conversion needs the feed's observed cadence, which
        only a view can supply. Converting here would mean assuming a cadence
        -- the same mistake the old `options_lookback_days` name made.
        """
        return self.lookback_snapshots + 1

    @model_validator(mode="after")
    def _check_non_negotiables(self) -> "OptionsFlowConfig":
        if not self.never_synthesize_missing_fields:
            raise ValueError(
                "never_synthesize_missing_fields=False permits filling absent "
                "chain fields. ARCHITECTURE.md section 5 says options data is "
                "never synthesized, and the researcher's brief forbids "
                "inventing missing options data outright. `OptionsSnapshot` "
                "distinguishes None (not supplied) from 0.0 (observed as zero) "
                "precisely so that absence survives into the feature layer; "
                "synthesizing erases that distinction and the audit cannot "
                "recover it."
            )
        return self

    @model_validator(mode="after")
    def _check_cap_can_bind(self) -> "OptionsFlowConfig":
        if self.eod_degraded_cap_fraction >= 1.0:
            raise ValueError(
                f"eod_degraded_cap_fraction={self.eod_degraded_cap_fraction} is at "
                "or above 1.0, which is the component's full weight -- that is "
                "not a cap. Section 5 requires the EOD-degraded options "
                "sub-score to be capped AND flagged; a cap that cannot bind "
                "leaves the flag describing a restriction that was never "
                "applied, which is worse than having neither."
            )
        return self

    @model_validator(mode="after")
    def _check(self) -> "OptionsFlowConfig":
        if not self.use_gamma_exposure_proxy and self.weights.gamma_exposure > 0.0:
            raise ValueError(
                f"use_gamma_exposure_proxy is False while its weight is "
                f"{self.weights.gamma_exposure}. The weights sum to 1.0, so a "
                "term that can never be computed would permanently remove that "
                "much of the sub-score while the component still claimed full "
                "availability -- a ceiling below 100% that no caller could see. "
                "Set weights.gamma_exposure to 0.0 and redistribute the rest, or "
                "leave the proxy enabled."
            )
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
    min_reward_risk: float = Field(
        gt=0.0,
        default=1.0,
        description=(
            "Achievable R:R below this declines the trade (14.6, "
            "WAIT('rr_too_low')). The default is 1.0 rather than SetupConfig's "
            "0.9 because 14.6's setup bands start at 1.0: a value in [0.9, 1.0) "
            "would pass this gate and then match no setup class, which is a "
            "trade declined with a misleading reason. Per-setup "
            "`SetupConfig.min_reward_risk` still applies afterwards and may be "
            "stricter; this is the structure layer's own floor."
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


class KronosConfig(FrozenModel):
    """The Kronos candlestick foundation model, as an OPTIONAL feature source.

    Kronos (github.com/shiyu-coder/Kronos, MIT) is a decoder-only transformer
    pre-trained on K-line sequences from "over 45 global exchanges". It is
    wired in here because the researcher asked for it, and it is wired in
    QUARANTINED, because of one fact that governs everything else about it.

    **The model publishes no training-data cutoff.** This project's research
    window is 2015-01-01 to 2023-01-01 with a sealed holdout from 2023-01-01
    to 2025-01-01. A model released in 2025 and trained on recent data from
    global exchanges has, in all likelihood, already seen every bar of both.
    Features derived from it during a historical backtest are therefore
    informed by the outcome they are being used to predict.

    That is lookahead, and it is a KIND of lookahead this project's defences
    cannot see. `validation/lookahead.py` checks truncation invariance,
    future-mutation invariance, warmup honesty and determinism; a pre-trained
    model passes all four trivially, because the leak is in the WEIGHTS and
    not in the data-access pattern. `MarketView` can guarantee that no future
    bar was read at bar t. It cannot guarantee that no future bar was read in
    2024 by whoever trained the checkpoint. `tests/unit/test_kronos.py`
    demonstrates that blindness on purpose, so a passing audit is never
    mistaken for evidence of no contamination.

    The consequence is a clean line rather than a ban:

    * **Live and paper-forward signals are sound.** A bar that has not
      happened yet cannot have been in anyone's training set, so a forecast
      for it carries no leakage.
    * **Historical backtesting is not sound** while the cutoff is unknown.
      `contamination_policy` defaults to REFUSE, so the feature computer
      declines rather than quietly producing a number that would flatter
      every metric downstream.

    Set `pretrain_cutoff` if a cutoff is ever published or a checkpoint is
    trained in-house: bars strictly after it are then clean, and the guard
    permits them automatically.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Off by default. Enabling it adds a torch dependency and a learned "
            "component to a system whose premise is explicit rules, so it is an "
            "opt-in experiment rather than part of the baseline."
        ),
    )

    # --- what to load -------------------------------------------------
    model_repo: str = Field(
        default="NeoQuasar/Kronos-small",
        description="Hugging Face repo for the predictor. small=24.7M, base=102.3M.",
    )
    tokenizer_repo: str = Field(
        default="NeoQuasar/Kronos-Tokenizer-base",
        description=(
            "Hugging Face repo for the tokenizer. Must match the model: the base "
            "tokenizer pairs with small/base, Tokenizer-2k pairs with mini."
        ),
    )
    module_path: str = Field(
        default="",
        description=(
            "Directory holding Kronos's own `model` package, for a source checkout. "
            "Empty means it is expected to be importable already. Kronos is not on "
            "PyPI, so one of the two must hold."
        ),
    )
    device: str = Field(
        default="cpu",
        description=(
            "cpu or cuda:N. cpu is the default because CUDA kernel selection is a "
            "second source of run-to-run variation on top of sampling."
        ),
    )

    # --- the forecast -------------------------------------------------
    max_context: int = Field(
        default=512, gt=0,
        description=(
            "Bars fed to the model. 512 for small/base, 2048 for mini; the "
            "predictor truncates beyond its own limit, so a larger value here "
            "silently does nothing."
        ),
    )
    pred_len: int = Field(
        default=12, gt=0,
        description=(
            "Forecast horizon in bars. Defaults to 12, which is SCALP_1R's "
            "max_hold_bars, so the forecast covers the trade it would inform "
            "rather than an arbitrary distance."
        ),
    )
    sample_count: int = Field(
        default=32, gt=0,
        description=(
            "Sampled paths per bar. The features are statistics OF this sample, "
            "so a path count this low is itself a source of variance; it is the "
            "denominator of kronos_up_probability and is reported as such."
        ),
    )
    temperature: float = Field(
        default=1.0, gt=0.0,
        description="Sampling temperature T. Lower concentrates the paths.",
    )
    top_p: float = Field(
        default=0.9, gt=0.0, le=1.0, description="Nucleus-sampling cutoff.",
    )
    seed: int = Field(
        default=DEFAULT_SEED,
        description=(
            "Seeds torch before every forecast, so the same view gives the same "
            "paths. Without it the computer would fail the lookahead audit's "
            "determinism check -- which is the one of the four checks that a "
            "sampling model can genuinely fail."
        ),
    )

    # --- the quarantine -----------------------------------------------
    pretrain_cutoff: date | None = Field(
        default=None,
        description=(
            "The last date the checkpoint's training data covers. None means "
            "UNKNOWN, which is Kronos's published state and is treated as "
            "contaminating every historical bar."
        ),
    )
    mode: KronosMode = Field(
        default=KronosMode.RESEARCH,
        description=(
            "RESEARCH (a replay over history) or LIVE (forward signals). It cannot "
            "be inferred from a MarketView -- in a backtest the view's cutoff IS the "
            "simulated present, so a frontier test is true on every bar of a replay "
            "and proves nothing. RESEARCH is the default because it is the "
            "conservative reading and because a backtest is the dangerous case."
        ),
    )
    contamination_policy: KronosContamination = Field(
        default=KronosContamination.REFUSE,
        description=(
            "What to do for a bar that the checkpoint may have trained on. "
            "REFUSE declines it (the default). FLAG computes it and marks the "
            "vector DEGRADED, for deliberately studying the contaminated "
            "signal. ALLOW computes it silently and is never appropriate for a "
            "reported backtest."
        ),
    )
    include_in_flow_score: bool = Field(
        default=False,
        description=(
            "Off by default, and not merely as caution. Section 7 requires the "
            "component weights to sum to 100 and every setup threshold was "
            "calibrated against that scale, so admitting a sixth component "
            "silently rescales every gate. Phase 8 can sweep an alternative "
            "weighting that includes it; the baseline does not."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> "KronosConfig":
        if self.contamination_policy is KronosContamination.ALLOW and self.pretrain_cutoff is None:
            raise ValueError(
                "contamination_policy=ALLOW with pretrain_cutoff=None would compute "
                "forecasts from a checkpoint of unknown provenance and report them "
                "as clean. If you mean to study the contaminated signal, use FLAG, "
                "which produces the same numbers and marks them DEGRADED."
            )
        if "mini" in self.model_repo.lower() and "2k" not in self.tokenizer_repo.lower():
            raise ValueError(
                f"model_repo {self.model_repo!r} is a mini checkpoint but "
                f"tokenizer_repo {self.tokenizer_repo!r} is not the 2k tokenizer; "
                "a mismatched tokenizer produces tokens the model never saw"
            )
        return self


class FlowModelConfig(FrozenModel):
    """Root configuration. `config_hash` identifies it in the experiment log."""

    name: str = "flow_model_default"
    description: str = ""
    seed: int = DEFAULT_SEED

    instruments: dict[str, InstrumentSpec] = Field(default_factory=dict)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    order_flow: OrderFlowConfig = Field(default_factory=OrderFlowConfig)
    options_flow: OptionsFlowConfig = Field(default_factory=OptionsFlowConfig)
    flow_score: FlowScoreConfig = Field(default_factory=FlowScoreConfig)
    levels: StructureLevelConfig = Field(default_factory=StructureLevelConfig)
    kronos: KronosConfig = Field(default_factory=KronosConfig)
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
