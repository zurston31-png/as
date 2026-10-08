"""Adversarial tests for the five component scorers.

`signals/scoring_bars.py` (STRUCTURE, LIQUIDITY, VOL_MOMENTUM) and
`signals/scoring_flow.py` (ORDER_FLOW, OPTIONS_FLOW) are the two halves of one
five-component set, and the Flow Score is the sum of what they emit. Neither
module computes a feature, gates a trade or selects a setup, so almost nothing
in them is arithmetic -- they are a *translation layer*, and the ways a
translation layer goes wrong are specific. Six of them, and the file is
organized behind them.

**1. Magnitude and direction touching (section 7).** The rule the whole phase
turns on:

    "Direction is handled separately from magnitude: each component emits
     (magnitude in [0,1], direction in {-1,0,+1}). The Flow Score is the
     magnitude aggregate; direction agreement is enforced by the gates. This
     avoids the common bug where a strong bearish component inflates a
     bullish score."

The failure is silent and flattering: a bearish component whose magnitude was
quietly reduced, or a direction scaled by its own strength, still produces
plausible numbers, and every backtest reads as a system that "knew" to sit
out. So it is tested per component and in both directions -- flip only the
direction key and the magnitude must be bit-identical; scale only the
magnitude and the direction must not move -- and then at the aggregate, where
one strongly bearish component among four bullish ones must leave the
magnitude aggregate HIGH and leave weighted opposition against BOTH sides.
The second half of the brief's requirement ("no trade is taken in either
direction") belongs to the gates and is not asserted here; what is asserted
is that nothing these scorers emit lets a gate read a majority direction as
permission.

**2. UNAVAILABLE collapsing into a measured zero.** This is the one the
project exists around, and the trap is that the two states are *numerically
identical*: a balanced tape and an absent tape both give `magnitude=0.0`,
`direction=0` and `points=0.0`. `test_a_balanced_tape_and_an_absent_tape_...`
asserts that equality first, as the premise, and only then asserts which
fields separate them -- `enabled`, `quality`, and the detail flags -- so the
distinction is tested where it actually lives rather than in a note nobody
reads. Both readings come from the REAL `features/orderflow.py` on two tapes
that differ only by the presence of the tick feed.

**3. Degradation that stops at the scorer.** Section 5 says liquidity
"degrades to volume-percentile only, flagged DEGRADED" and that the EOD
options sub-score is "capped AND flagged". Both chains run feature module ->
per-key grade -> `ComponentScore`, and both are invisible if the last link
drops: a DEGRADED liquidity score reported GOOD, or a capped options
magnitude whose cap flag never arrives, is a number on a different scale
wearing the right name. Tested through the real computers on datasets built
to bind each one, and then on hand-built vectors where the arithmetic is
exact.

**4. Bounds.** Every magnitude must be in [0, 1] and every direction in
{-1, 0, +1} -- not at one hand-picked bar but across a long synthetic run
through the real feature layer, with the run's own coverage asserted first so
the bounds test cannot pass vacuously on bars that were all unavailable.

**5. Reading a key it never declared.** Each scorer declares `reads`. A
scorer that quietly consumed a key outside it would make its docstring, its
`features_used` and the lookahead audit's traceability wrong at once. Tested
by perturbing every one of the 96 feature keys the seven real computers emit,
one at a time, and requiring the `ComponentScore` to be byte-identical
whenever the perturbed key is not declared.

**6. A weight that is not the configured weight.** Weights are HYPOTHESES for
the Phase 8 sweep. Every weight-sensitive expectation here is stated against a
locally-built config, and a second, deliberately different weight set is run
through the same assertions, so a hardcoded 20 or 25 fails. Nothing in this
file reads `config/defaults.yaml`: the two weight sets, the EOD cap fraction
and the opposition limit are declared in this file, because a test coupled to
the YAML fails when an unrelated section is edited and silently changes its
expected values when a default moves.

Two things this file does NOT claim. It makes no statement about predictive
content -- the only data available at this phase is `data/synthetic.py` and
hand-built vectors, and a magnitude measured off a generator is a measurement
of the generator. And it does not assert that the two modules AGREE
everywhere: two divergences were found and are pinned as divergences, with
the reasoning in their docstrings, rather than quietly blessed.

Builders are local, as in `tests/unit/test_levels.py` and
`tests/unit/test_structure.py`: file ownership, and a failure localizes here
instead of to a shared fixture. `full_values()` emits every key all seven
computers declare -- asserted against the computers themselves in
`test_the_hand_built_vector_covers_every_key_...` -- so a scorer handed one of
these vectors is never looking at a narrower world than the real bundle gives
it.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.config.schema import (
    BacktestConfig,
    FlowModelConfig,
    FlowScoreConfig,
    OptionsFlowConfig,
    SetupConfig,
)
from flow_model.core.contracts import ComponentScore, FeatureVector, FlowScore
from flow_model.core.enums import (
    Component,
    DataQuality,
    Feed,
    InstrumentType,
    Regime,
    SetupType,
    Side,
)
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.calendar import TradingCalendar
from flow_model.data.market_view import MarketView
from flow_model.data.quality import ComponentAvailability
from flow_model.data.series import BarSeries, TickSeries, from_ns, to_ns_array
from flow_model.data.store import SymbolData
from flow_model.data.synthetic import SyntheticConfig, SyntheticMarketGenerator
from flow_model.features.base import FeatureBundle
from flow_model.features.levels import LevelFeatures
from flow_model.features.liquidity import NO_QUOTE_SCORE_CAP, LiquidityFeatures
from flow_model.features.momentum import MomentumFeatures
from flow_model.features.optionsflow import OptionsFlowFeatures
from flow_model.features.orderflow import OrderFlowFeatures
from flow_model.features.structure import StructureFeatures
from flow_model.features.volatility import VolatilityFeatures
from flow_model.signals.scoring_bars import (
    REASON_COMPONENT_DISABLED,
    REASON_FEED_UNAVAILABLE,
    VOL_MOMENTUM_WEIGHT_MOMENTUM,
    VOL_MOMENTUM_WEIGHT_VOLATILITY,
    LiquidityScorer,
    ScorerError,
    StructureScorer,
    VolMomentumScorer,
    bars_scorers,
)
from flow_model.signals.scoring_flow import (
    OPTIONS_FLOW_READS,
    ORDER_FLOW_READS,
    OptionsFlowScorer,
    OrderFlowScorer,
    ScoringError,
    as_components,
    feed_dependent_scorers,
)

SYMBOL = "NQ"
INTERVAL = 300
TS = datetime(2024, 3, 1, 15, 0, tzinfo=timezone.utc)
TS_NEXT = TS + timedelta(seconds=INTERVAL)

#: Section 7's PRIOR weights, declared here rather than read from
#: `config/defaults.yaml`. They are a hypothesis for the Phase 8 sweep, and
#: every hand-computed number in this file is stated against them: STRUCTURE
#: 20 + LIQUIDITY 15 + VOL_MOMENTUM 20 = 55 bars-only points, ORDER_FLOW 25 +
#: OPTIONS_FLOW 20 = the 45 that need a feed bars cannot supply.
PRIOR_WEIGHTS: dict[Component, float] = {
    Component.STRUCTURE: 20.0,
    Component.LIQUIDITY: 15.0,
    Component.VOL_MOMENTUM: 20.0,
    Component.ORDER_FLOW: 25.0,
    Component.OPTIONS_FLOW: 20.0,
}

#: A second weight set, every entry different from the prior and still summing
#: to 100. Not a proposal and not tuned -- it exists so that a hardcoded 20 or
#: 25 anywhere in either scorer module fails a test instead of agreeing with
#: the default by coincidence.
SWEPT_WEIGHTS: dict[Component, float] = {
    Component.STRUCTURE: 30.0,
    Component.LIQUIDITY: 5.0,
    Component.VOL_MOMENTUM: 10.0,
    Component.ORDER_FLOW: 40.0,
    Component.OPTIONS_FLOW: 15.0,
}

#: Declared locally for the same reason as the weights. `OptionsFlowConfig`
#: refuses a fraction at or above 1.0 (a cap that cannot bind), so 0.5 is a
#: value the validator accepts and the EOD path actually reaches.
EOD_CAP = 0.5

#: `config.flow_score.max_opposing_points`, declared locally. The entry filter
#: is the gates' business, not a scorer's, but the aggregate assertions in
#: "one component disagrees" need a number to compare against and must not
#: read it from the YAML.
MAX_OPPOSING_POINTS = 12.0

#: The magnitude every hand-built component carries unless a test says
#: otherwise. 0.9 of everything makes the aggregate 0.9 * 100 = 90.0 exactly.
BIG = 0.9


# ---------------------------------------------------------------------------
# configuration, built here and not loaded
# ---------------------------------------------------------------------------


def nq_spec() -> InstrumentSpec:
    """A liquid index future with an RTH window, so the session-aware
    computers (liquidity's time-of-day profile, structure's anchors) have a
    session to work against."""
    return InstrumentSpec(
        symbol=SYMBOL,
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        typical_spread_ticks=1.0,
        min_spread_ticks=1.0,
        rth=SessionWindow(name="rth", start="09:30", end="16:00"),
    )


def flow_score_config(**overrides) -> FlowScoreConfig:
    fields: dict = {
        "weights": dict(PRIOR_WEIGHTS),
        "total_points": 100.0,
        "strict_component_availability": True,
        "redistribute_disabled_weight": False,
        "enabled_components": {c: True for c in Component},
        "max_opposing_points": MAX_OPPOSING_POINTS,
        "options_contradiction_points": 14.0,
    }
    fields.update(overrides)
    return FlowScoreConfig(**fields)


def root_config(**overrides) -> FlowModelConfig:
    """The root config the two feed-dependent scorers require.

    One enabled setup, because `FlowModelConfig` refuses a configuration in
    which nothing could ever trade. Nothing in either scorer module reads
    `setups`, and nothing here reads `research_targets`.
    """
    fields: dict = {
        "instruments": {SYMBOL: nq_spec()},
        "backtest": BacktestConfig(symbols=(SYMBOL,)),
        "setups": {
            SetupType.SCALP_1R: SetupConfig(
                setup=SetupType.SCALP_1R,
                reward_risk=1.0,
                min_flow_score=70.0,
                allowed_regimes=(Regime.TRENDING_UP,),
            )
        },
        "flow_score": flow_score_config(),
        "options_flow": OptionsFlowConfig(eod_degraded_cap_fraction=EOD_CAP),
    }
    fields.update(overrides)
    return FlowModelConfig(**fields)


def real_computers(config: FlowModelConfig, calendar: TradingCalendar | None = None):
    """The seven computers behind the five components, in one bundle order."""
    spec = config.spec(SYMBOL)
    return (
        VolatilityFeatures(config.features),
        MomentumFeatures(config.features),
        LiquidityFeatures(config.features, spec),
        StructureFeatures(config.features, config.levels, spec, calendar),
        LevelFeatures(config.features, config.levels, spec, calendar),
        OrderFlowFeatures(config.order_flow, config.features),
        OptionsFlowFeatures(config.options_flow),
    )


def scorer_for(component: Component, config: FlowModelConfig):
    """The one scorer that produces `component`.

    The three bars-only scorers take `config.flow_score`; the two
    feed-dependent ones take the root config, because they read
    `config.options_flow`. Both forms are configuration only.
    """
    return {
        Component.STRUCTURE: lambda: StructureScorer(config.flow_score),
        Component.LIQUIDITY: lambda: LiquidityScorer(config.flow_score),
        Component.VOL_MOMENTUM: lambda: VolMomentumScorer(config.flow_score),
        Component.ORDER_FLOW: lambda: OrderFlowScorer(config),
        Component.OPTIONS_FLOW: lambda: OptionsFlowScorer(config),
    }[component]()


def all_scorers(config: FlowModelConfig) -> tuple:
    return tuple(bars_scorers(config.flow_score)) + tuple(feed_dependent_scorers(config))


def component_score(
    scorer, features: FeatureVector, availability: ComponentAvailability | None = None
) -> ComponentScore:
    """One `ComponentScore` out of either module.

    `scoring_bars` returns the contract object straight out of `score()`;
    `scoring_flow` returns a `ComponentScoring` and offers `score_component()`
    for exactly this. Reconciling the two signatures in one place is what the
    aggregator does too.
    """
    adapter = getattr(scorer, "score_component", None)
    if adapter is not None:
        return adapter(features, availability)
    return scorer.score(features, availability)


# ---------------------------------------------------------------------------
# hand-built feature vectors
# ---------------------------------------------------------------------------


def full_values(
    *,
    # levels.py -- STRUCTURE's magnitude and direction
    structure_magnitude: float = BIG,
    level_direction: float = 1.0,
    # structure.py -- STRUCTURE's corroborating geometry (diagnostics only)
    structure_direction: float = 1.0,
    bos_direction: float = 1.0,
    # liquidity.py
    liquidity_score: float = BIG,
    depth_imbalance: float = 0.1,
    # volatility.py + momentum.py -- VOL_MOMENTUM
    vol_regime_score: float = BIG,
    momentum_score: float = BIG,
    momentum_direction: float = 1.0,
    # orderflow.py
    order_flow_magnitude: float = BIG,
    order_flow_direction: float = 1.0,
    order_flow_available: float = 1.0,
    order_flow_available_weight: float = 1.0,
    # optionsflow.py
    options_flow_magnitude: float = BIG,
    options_flow_uncapped_magnitude: float = BIG,
    options_flow_direction: float = 1.0,
    options_eod_capped: float = 0.0,
    options_flow_timing_available: float = 1.0,
    options_available_weight: float = 1.0,
    options_terms_available: float = 5.0,
) -> dict[str, float]:
    """Every key the seven real computers declare, with plausible values.

    A full vector rather than a per-component slice on purpose: a scorer
    handed only its own keys cannot demonstrate that it ignores the other 80,
    which is risk 5. The key set is asserted against the computers in
    `test_the_hand_built_vector_covers_every_key_the_feature_layer_declares`,
    so a key added to a feature module fails here instead of silently being
    absent.
    """
    return {
        # --- levels.py (21) ------------------------------------------
        "zone_count": 6.0,
        "major_zone_count": 2.0,
        "nearest_zone_price": 18_000.0,
        "nearest_zone_width": 12.5,
        "nearest_zone_distance_atr": 0.4,
        "nearest_zone_gap_atr": 3.1,
        "level_significance": 0.82,
        "level_cleanliness": 0.74,
        "level_rejection": 0.66,
        "level_direction": level_direction,
        "level_touch_count": 4.0,
        "level_recent_touches": 2.0,
        "rejection_confirmed": 1.0,
        "level_stop_price": 17_975.0,
        "level_target_price": 18_075.0,
        "achievable_rr": 2.4,
        "level_setup_class": 2.0,
        "structure_magnitude": structure_magnitude,
        "structure_gate_passed": 1.0,
        "level_gate_reason": 0.0,
        "level_flow_available": 0.0,
        # --- structure.py (15) ---------------------------------------
        "swing_high": 18_050.0,
        "swing_low": 17_950.0,
        "structure_direction": structure_direction,
        "bos_direction": bos_direction,
        "choch": 0.0,
        "bars_since_pivot": 7.0,
        "vwap": 18_010.0,
        "vwap_deviation_atr": 0.3,
        "opening_range_high": 18_040.0,
        "opening_range_low": 17_980.0,
        "opening_range_position": 0.6,
        "prior_session_high": 18_060.0,
        "prior_session_low": 17_940.0,
        "prior_session_close": 18_005.0,
        "structure_score": 0.71,
        # --- liquidity.py (9) ----------------------------------------
        "spread_ticks": 1.0,
        "spread_percentile": 0.35,
        "depth_imbalance": depth_imbalance,
        "volume_percentile": 0.62,
        "relative_volume": 1.15,
        "volume_trend": 0.2,
        "dollar_volume": 45_000_000.0,
        "participation_cost_ticks": 1.3,
        "liquidity_score": liquidity_score,
        # --- volatility.py (9) ---------------------------------------
        "atr": 25.0,
        "atr_pct": 0.0014,
        "atr_percentile": 0.55,
        "realized_vol": 0.18,
        "parkinson_vol": 0.21,
        "garman_klass_vol": 0.19,
        "vol_of_vol": 0.3,
        "vol_expansion": 0.25,
        "vol_regime_score": vol_regime_score,
        # --- momentum.py (9) -----------------------------------------
        "roc": 0.004,
        "roc_atr": 2.8,
        "efficiency_ratio": 0.68,
        "direction": momentum_direction,
        "acceleration": 0.4,
        "momentum_persistence": 0.6,
        "up_bar_fraction": 0.7,
        "close_position": 0.85,
        "momentum_score": momentum_score,
        # --- orderflow.py (15) ---------------------------------------
        "order_flow_magnitude": order_flow_magnitude,
        "order_flow_direction": order_flow_direction,
        "order_flow_available": order_flow_available,
        "order_flow_available_weight": order_flow_available_weight,
        "signed_delta": 0.55,
        "signed_delta_contracts": 1_240.0,
        "cvd_slope": 0.42,
        "cvd_slope_normalized": 0.09,
        "aggression_ratio": 0.63,
        "absorption": 0.0,
        "absorption_direction": 0.0,
        "avg_trade_size_percentile": 0.58,
        "max_trade_size_percentile": 0.64,
        "large_trade_event": 0.0,
        "classification_coverage": 0.97,
        # --- optionsflow.py (18) -------------------------------------
        "options_net_premium_imbalance": 0.5,
        "options_delta_volume_imbalance": 0.4,
        "options_oi_change_imbalance": 0.3,
        "options_skew_25d_pressure": 0.2,
        "options_gamma_exposure_pressure": 0.1,
        "options_net_premium": 215_000.0,
        "options_skew_25d": 0.125,
        "options_total_volume": 12_000.0,
        "options_snapshot_age_seconds": 30.0,
        "options_available_weight": options_available_weight,
        "options_terms_available": options_terms_available,
        "options_flow_timing_available": options_flow_timing_available,
        "options_eod_capped": options_eod_capped,
        "options_sign_agreement": 1.0,
        "options_imbalance_consensus": 0.35,
        "options_flow_uncapped_magnitude": options_flow_uncapped_magnitude,
        "options_flow_magnitude": options_flow_magnitude,
        "options_flow_direction": options_flow_direction,
    }


def vector(
    values: dict[str, float],
    *,
    grades: dict[str, DataQuality] | None = None,
    default_grade: DataQuality = DataQuality.GOOD,
    ts: datetime = TS,
    notes: tuple[str, ...] = (),
) -> FeatureVector:
    by_key = {key: default_grade for key in values}
    by_key.update(grades or {})
    return FeatureVector(
        symbol=SYMBOL, ts=ts, values=values, quality_by_key=by_key, notes=notes
    )


def magnitude_kwargs(component: Component, magnitude: float) -> dict[str, float]:
    """`full_values` kwargs that give `component` exactly `magnitude`.

    VOL_MOMENTUM's magnitude is the convex blend of its two halves, so setting
    both halves to `m` gives `w_v*m + w_m*m = m` for any convex split.
    OPTIONS_FLOW's capped and uncapped magnitudes must agree when the cap flag
    is clear, which is the scorer's third invariant.
    """
    return {
        Component.STRUCTURE: {"structure_magnitude": magnitude},
        Component.LIQUIDITY: {"liquidity_score": magnitude},
        Component.VOL_MOMENTUM: {
            "vol_regime_score": magnitude,
            "momentum_score": magnitude,
        },
        Component.ORDER_FLOW: {"order_flow_magnitude": magnitude},
        Component.OPTIONS_FLOW: {
            "options_flow_magnitude": magnitude,
            "options_flow_uncapped_magnitude": magnitude,
        },
    }[component]


def direction_kwargs(component: Component, sign: float) -> dict[str, float]:
    """`full_values` kwargs that point `component` at `sign`.

    LIQUIDITY has no entry: it emits no direction at all, which is its own
    test rather than an omission here.
    """
    return {
        Component.STRUCTURE: {"level_direction": sign},
        Component.LIQUIDITY: {},
        Component.VOL_MOMENTUM: {"momentum_direction": sign},
        Component.ORDER_FLOW: {"order_flow_direction": sign},
        Component.OPTIONS_FLOW: {"options_flow_direction": sign},
    }[component]


#: The four components that carry a direction, and the one that does not.
DIRECTIONAL = (
    Component.STRUCTURE,
    Component.VOL_MOMENTUM,
    Component.ORDER_FLOW,
    Component.OPTIONS_FLOW,
)
EVERY_COMPONENT = (
    Component.STRUCTURE,
    Component.LIQUIDITY,
    Component.VOL_MOMENTUM,
    Component.ORDER_FLOW,
    Component.OPTIONS_FLOW,
)


# ---------------------------------------------------------------------------
# synthetic market builders
# ---------------------------------------------------------------------------


def hand_built_bars(n: int, mid: float = 18_000.0) -> BarSeries:
    """A flat bar grid. Used only where the tape's content is irrelevant and
    the presence or absence of a FEED is the whole point."""
    stamps = to_ns_array(TS + timedelta(seconds=INTERVAL * (i + 1)) for i in range(n))
    close = np.full(n, float(mid))
    return BarSeries(
        symbol=SYMBOL,
        ts_ns=stamps,
        interval_seconds=INTERVAL,
        columns={
            "open": close.copy(),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close.copy(),
            "volume": np.full(n, 1000.0),
        },
    )


def balanced_ticks(n: int) -> TickSeries:
    """A tape with buy volume exactly equal to sell volume on every bar.

    The point of the fixture: this is a MEASUREMENT of a balanced tape, and
    `features/orderflow.py` reports it as a magnitude of exactly 0.0 -- the
    same number an absent feed produces. An honest classification method is
    declared, so nothing else in the module can disable the component.
    """
    stamps = to_ns_array(TS + timedelta(seconds=INTERVAL * (i + 1)) for i in range(n))
    return TickSeries(
        symbol=SYMBOL,
        ts_ns=stamps,
        columns={
            "buy_volume": np.full(n, 50.0),
            "sell_volume": np.full(n, 50.0),
            "buy_trades": np.full(n, 5.0),
            "sell_trades": np.full(n, 5.0),
            "max_trade_size": np.full(n, 10.0),
        },
        meta={"classification_method": "bid_ask_quote"},
    )


def view_at_last_bar(data: SymbolData) -> MarketView:
    cutoff = int(data.primary_bars.ts_ns[-1])
    return MarketView(data, now=from_ns(cutoff), now_ns=cutoff)


#: Datasets and their scored runs, built once per shape. The real feature
#: layer is not cheap enough to rebuild per test and not expensive enough to
#: need a fixture; a module-level memo keeps the builders local, which is the
#: convention these test files follow.
_RUNS: dict[tuple, list] = {}

#: The run window and sampling stride. 1560 bars is the longest warmup in the
#: bundle (`features/liquidity.py`), so sampling starts there; the stride is
#: coprime with the 78-bar session so the sample does not land on the same
#: minute of every session.
RUN_START = date(2024, 1, 2)
RUN_END = date(2024, 4, 1)
RUN_WARMUP = 1560
RUN_STRIDE = 41


def scored_run(*, feeds: str, options_intraday: bool = False) -> list:
    """`[(ts, {component: ComponentScore})]` over a long synthetic run.

    `feeds` is "all" (bars + quotes + ticks + an options chain) or "bars"
    (bars alone, which is the data shape the brief says real research faces).
    Every bar goes through the real seven computers and all five real scorers.
    """
    key = (feeds, options_intraday)
    if key in _RUNS:
        return _RUNS[key]

    config = root_config()
    spec = config.spec(SYMBOL)
    calendar = TradingCalendar()
    dataset = SyntheticMarketGenerator(
        SyntheticConfig(options_intraday=options_intraday), seed=7
    ).generate(SYMBOL, spec, RUN_START, RUN_END, INTERVAL, calendar=calendar)

    if feeds == "all":
        data = SymbolData(
            symbol=SYMBOL,
            primary_interval=INTERVAL,
            bars={INTERVAL: dataset.bars},
            ticks={INTERVAL: dataset.ticks},
            quotes=dataset.quotes,
            options=dataset.options,
        )
    elif feeds == "bars":
        data = SymbolData(
            symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: dataset.bars}
        )
    else:  # pragma: no cover - programming error in a test
        raise AssertionError(f"unknown feed shape {feeds!r}")

    bundle = FeatureBundle(real_computers(config, calendar))
    scorers = all_scorers(config)
    out = []
    for index in range(RUN_WARMUP, len(dataset.bars.ts_ns), RUN_STRIDE):
        cutoff = int(dataset.bars.ts_ns[index])
        features = bundle.compute(MarketView(data, now=from_ns(cutoff), now_ns=cutoff))
        out.append(
            (
                features.ts,
                {
                    s.component: component_score(s, features)
                    for s in scorers
                },
            )
        )
    _RUNS[key] = out
    return out


# ---------------------------------------------------------------------------
# premises -- asserted before anything relies on them
# ---------------------------------------------------------------------------


def test_the_local_config_carries_the_weights_the_arithmetic_assumes():
    """Every hand-computed number below is stated against these weights.

    If `FlowScoreConfig` ever rejected them, or if this file's two weight sets
    stopped summing to the total, the arithmetic in the rest of the file would
    be comparing against numbers nobody computed. Asserted first so that a
    failure here is unmistakably the premise and not the subject.
    """
    config = root_config()
    assert config.flow_score.weights == PRIOR_WEIGHTS
    assert sum(PRIOR_WEIGHTS.values()) == 100.0 == config.flow_score.total_points
    assert sum(SWEPT_WEIGHTS.values()) == 100.0
    # the two sets must disagree on every component, or a hardcoded weight
    # could pass both
    assert all(PRIOR_WEIGHTS[c] != SWEPT_WEIGHTS[c] for c in Component)
    # 20 + 15 + 20 = 55 bars-only points; 25 + 20 = 45 that need a feed
    bars_only = (
        PRIOR_WEIGHTS[Component.STRUCTURE]
        + PRIOR_WEIGHTS[Component.LIQUIDITY]
        + PRIOR_WEIGHTS[Component.VOL_MOMENTUM]
    )
    assert bars_only == 55.0
    assert config.options_flow.eod_degraded_cap_fraction == EOD_CAP
    assert config.flow_score.max_opposing_points == MAX_OPPOSING_POINTS


def test_the_hand_built_vector_covers_every_key_the_feature_layer_declares():
    """`full_values()` must be the real bundle's key set, exactly.

    A missing key would make a scorer report UNAVAILABLE for a reason the test
    did not intend (and `scoring_flow` raise outright), and an extra key would
    be a name no module emits -- a perturbation test over it would prove
    nothing. Both directions are checked.
    """
    config = root_config()
    declared: set[str] = set()
    for computer in real_computers(config):
        declared |= set(computer.keys)
    built = set(full_values())
    assert built - declared == set(), "builder invents keys no computer emits"
    assert declared - built == set(), "builder is missing keys the bundle emits"
    assert len(built) == 96


def test_the_five_scorers_cover_the_five_components_exactly_once():
    """One scorer per component, no duplicates, nothing left out.

    `as_components` raises on a duplicate; what it cannot catch is a component
    nobody scores, which would quietly leave weight out of `max_points`.
    """
    config = root_config()
    produced = [s.component for s in all_scorers(config)]
    assert sorted(c.value for c in produced) == sorted(c.value for c in Component)
    assert len(produced) == len(set(produced)) == 5


def test_every_key_a_scorer_reads_is_produced_by_a_real_feature_computer():
    """A declared read key that nothing emits is a silent hole.

    For `scoring_bars` an orphaned `detail_keys` entry simply never appears in
    `detail` -- no error, no diagnostic, and a trade record missing a field it
    claims to carry. For `scoring_flow` it is worse: `_check_keys` would raise
    on every bar. Either way the typo is invisible until this runs.
    """
    config = root_config()
    declared: set[str] = set()
    for computer in real_computers(config):
        declared |= set(computer.keys)
    for scorer in all_scorers(config):
        orphans = sorted(set(scorer.reads) - declared)
        assert orphans == [], f"{type(scorer).__name__} reads unproduced {orphans}"


# ---------------------------------------------------------------------------
# risk 1: magnitude and direction never touch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", DIRECTIONAL, ids=lambda c: c.value)
def test_flipping_only_the_direction_leaves_the_magnitude_bit_identical(component):
    """Section 7, the half that is per component.

    The bug section 7 names needs only one multiplication: a magnitude signed
    by, scaled by or conditioned on its own direction. It would be invisible
    from the aggregate -- scores would simply be lower on bearish bars, which
    reads as a conservative system. Here the ONLY thing that changes between
    the two vectors is the component's direction key, so any movement in the
    magnitude is that multiplication.
    """
    config = root_config()
    scorer = scorer_for(component, config)
    bullish = component_score(
        scorer, vector(full_values(**{**magnitude_kwargs(component, BIG),
                                      **direction_kwargs(component, 1.0)}))
    )
    bearish = component_score(
        scorer, vector(full_values(**{**magnitude_kwargs(component, BIG),
                                      **direction_kwargs(component, -1.0)}))
    )
    assert bullish.direction == 1
    assert bearish.direction == -1
    # bit-identical, not approximately equal: there is no arithmetic between
    # them that could legitimately differ
    assert bearish.magnitude == bullish.magnitude == BIG
    assert bearish.points == bullish.points
    assert bearish.weight == bullish.weight
    assert bearish.quality is bullish.quality
    assert bearish.enabled is bullish.enabled


@pytest.mark.parametrize("component", DIRECTIONAL, ids=lambda c: c.value)
@pytest.mark.parametrize("magnitude", [0.0, 0.05, 0.5, 0.95, 1.0])
def test_scaling_only_the_magnitude_leaves_the_direction_untouched(
    component, magnitude
):
    """The mirror of the previous test, and the less obvious half.

    A direction derived from a magnitude -- "strong enough to call a side" --
    would turn a three-valued vote into a thresholded magnitude, which section
    7 separates precisely so the gates can refuse a disagreement regardless of
    how small it is. Swept across the closed interval including both ends,
    because 0.0 is where a magnitude-driven direction would collapse to 0.
    """
    config = root_config()
    scorer = scorer_for(component, config)
    score = component_score(
        scorer,
        vector(
            full_values(
                **{
                    **magnitude_kwargs(component, magnitude),
                    **direction_kwargs(component, -1.0),
                }
            )
        ),
    )
    assert score.magnitude == magnitude
    assert score.direction == -1, "a zero-magnitude bearish reading is still bearish"


@pytest.mark.parametrize("component", DIRECTIONAL, ids=lambda c: c.value)
def test_a_mirrored_pair_has_identical_magnitudes_and_opposite_directions(component):
    """Reflecting a reading must reflect only its sign.

    Stronger than the flip test: both halves of the pair are built
    independently from the same magnitude, and their `points` must be equal --
    so neither sign is worth more of the component's weight than the other.
    `opposes`/`agrees_with` must mirror exactly, which is what the gates read.
    """
    config = root_config()
    scorer = scorer_for(component, config)
    up = component_score(
        scorer,
        vector(full_values(**{**magnitude_kwargs(component, 0.8),
                              **direction_kwargs(component, 1.0)})),
    )
    down = component_score(
        scorer,
        vector(full_values(**{**magnitude_kwargs(component, 0.8),
                              **direction_kwargs(component, -1.0)})),
    )
    assert up.magnitude == down.magnitude == 0.8
    assert up.points == down.points == 0.8 * PRIOR_WEIGHTS[component]
    assert up.direction == -down.direction
    assert up.agrees_with(Side.LONG) and up.opposes(Side.SHORT)
    assert down.agrees_with(Side.SHORT) and down.opposes(Side.LONG)


@pytest.mark.parametrize("depth", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_liquidity_emits_no_direction_whatever_the_depth_imbalance_says(depth):
    """LIQUIDITY is a permission, not an opinion, and that has consequences.

    `depth_imbalance` is the one signed key in `features/liquidity.py`, and
    without a quote feed it is a fallback constant -- a direction read off it
    would be manufactured from configuration on exactly the dataset the
    component is supposed to degrade on. Emitting 0 also keeps the component's
    15 points out of `FlowScore.opposing_points`, which is correct: thin
    liquidity must block a trade through `WaitReason.LIQUIDITY`, where the
    reason is recorded, and not as a silent directional veto.
    """
    config = root_config()
    score = component_score(
        scorer_for(Component.LIQUIDITY, config),
        vector(full_values(depth_imbalance=depth)),
    )
    assert score.direction == 0
    assert score.detail["depth_imbalance"] == depth, "still reported, just not voted"
    assert not score.opposes(Side.LONG) and not score.opposes(Side.SHORT)


def test_a_bearish_component_contributes_its_full_weighted_points():
    """"A lot is happening, downwards" must be a LOT of points.

    ORDER_FLOW at magnitude 0.9 with direction -1 is 0.9 * 25 = 22.5 points,
    the same number a bullish 0.9 would contribute. Written out because the
    temptation in a bullish-biased pipeline is to discount it.
    """
    config = root_config()
    score = component_score(
        scorer_for(Component.ORDER_FLOW, config),
        vector(full_values(order_flow_magnitude=0.9, order_flow_direction=-1.0)),
    )
    # 0.9 * 25.0 = 22.5
    assert score.points == pytest.approx(22.5)
    assert score.direction == -1
    assert score.enabled is True


def test_the_aggregate_is_ninety_of_one_hundred_with_one_component_bearish():
    """The brief's adversarial case, the half that lives in these modules.

    Hand-chosen so the aggregate is a number written down rather than read
    back: every component at magnitude 0.9, so the score is

        0.9*20 + 0.9*15 + 0.9*20 + 0.9*25 + 0.9*20 = 0.9 * 100 = 90.0

    ORDER_FLOW points the other way. The magnitude aggregate must stay at 90.0
    -- it does not net, and the disagreement costs nothing -- while weighted
    opposition exists against BOTH sides:

        opposing a LONG  = ORDER_FLOW                       = 22.5
        opposing a SHORT = STRUCTURE + VOL_MOMENTUM + OPTIONS = 18 + 18 + 18 = 54.0

    Both are above `max_opposing_points` (12.0), so the entry filter has
    grounds to refuse either side. `net_direction()` is +1 here, which is
    exactly the majority vote section 7 warns about: a gate that read it would
    take the long that 22.5 opposing points forbid. Refusing is the gates'
    job; what is asserted here is that nothing in the scorers hides the
    disagreement from them.
    """
    config = root_config()
    scores = {}
    for component in EVERY_COMPONENT:
        kwargs = {**magnitude_kwargs(component, 0.9)}
        if component is Component.ORDER_FLOW:
            kwargs.update(direction_kwargs(component, -1.0))
        elif component is not Component.LIQUIDITY:
            kwargs.update(direction_kwargs(component, 1.0))
        scores[component] = component_score(
            scorer_for(component, config), vector(full_values(**kwargs))
        )

    flow = FlowScore(
        symbol=SYMBOL, ts=TS, components=scores, max_points=100.0
    )
    assert flow.score == pytest.approx(90.0)
    assert flow.available_points == pytest.approx(100.0)
    assert flow.opposing_points(Side.LONG) == pytest.approx(22.5)
    assert flow.opposing_points(Side.SHORT) == pytest.approx(54.0)
    assert flow.opposing_points(Side.LONG) > MAX_OPPOSING_POINTS
    assert flow.opposing_points(Side.SHORT) > MAX_OPPOSING_POINTS
    assert flow.net_direction() == 1


def test_the_aggregate_does_not_cancel_a_disagreement():
    """A netting aggregate would read 31.5 here and 90.0 is the right answer.

    The same five components at magnitude 0.9, with ORDER_FLOW bearish. If
    magnitude and direction were multiplied before summing, the 22.5 bearish
    points would subtract and LIQUIDITY's 13.5 would vanish entirely for
    having no direction at all:

        18.0 + 0.0 + 18.0 - 22.5 + 18.0 = 31.5

    The aggregate is a MAGNITUDE aggregate, so it is 0.9 * 100 = 90.0 and
    carries no directional opinion. Asserted against the netted number
    explicitly so the test cannot pass for a near-miss reason.

    (31.5 is also the answer my own first draft of this test got wrong: I
    wrote 67.5 - 22.5 = 45.0, forgetting that a direction of 0 removes a
    component from a netted sum rather than adding its magnitude. The netting
    arithmetic is less obvious than it looks, which is part of why the
    separation is worth enforcing structurally.)
    """
    config = root_config()
    scores = {}
    for component in EVERY_COMPONENT:
        kwargs = dict(magnitude_kwargs(component, 0.9))
        sign = -1.0 if component is Component.ORDER_FLOW else 1.0
        kwargs.update(direction_kwargs(component, sign))
        scores[component] = component_score(
            scorer_for(component, config), vector(full_values(**kwargs))
        )
    flow = FlowScore(symbol=SYMBOL, ts=TS, components=scores, max_points=100.0)
    netted = sum(c.direction * c.points for c in scores.values())
    assert netted == pytest.approx(31.5)
    assert scores[Component.LIQUIDITY].direction == 0
    assert scores[Component.LIQUIDITY].points == pytest.approx(13.5)
    assert flow.score == pytest.approx(90.0)
    assert flow.score != pytest.approx(netted)


def test_structure_reports_geometry_disagreement_without_folding_it_in():
    """14.6 fixes the magnitude formula; corroboration is reported, not mixed.

    `features/structure.py`'s pivot sequence is independent evidence, and a
    scorer that averaged `structure_score` into the magnitude, or let
    `structure_direction` override `level_direction`, would be producing a
    different component than section 14.6 defines while using its name. So
    flipping the geometry against the level must move the reported agreement
    product and nothing else:

        geometry_agreement = structure_direction * level_direction
                           = -1 * +1 = -1

    Whether a disagreement should block the trade is a gate question -- gates
    express "must have", scores express "how good" (14.6) -- so the number is
    put where a gate can read it and no further.
    """
    config = root_config()
    scorer = scorer_for(Component.STRUCTURE, config)
    agreeing = component_score(
        scorer, vector(full_values(level_direction=1.0, structure_direction=1.0,
                                   bos_direction=1.0))
    )
    disagreeing = component_score(
        scorer, vector(full_values(level_direction=1.0, structure_direction=-1.0,
                                   bos_direction=-1.0))
    )
    assert agreeing.detail["geometry_agreement"] == 1.0
    assert disagreeing.detail["geometry_agreement"] == -1.0
    assert disagreeing.detail["bos_agreement"] == -1.0
    # the component itself is untouched by the disagreement
    assert disagreeing.magnitude == agreeing.magnitude == BIG
    assert disagreeing.direction == agreeing.direction == 1
    assert disagreeing.points == agreeing.points


def test_the_vol_momentum_magnitude_is_the_declared_convex_blend():
    """Exact arithmetic, read off the module's own declared weights.

    With `vol_regime_score = 0.8` and `momentum_score = 0.4` and the declared
    0.5 / 0.5 split:

        magnitude = 0.5 * 0.8 + 0.5 * 0.4 = 0.60
        points    = 0.60 * 20.0           = 12.00
        volatility_term = 0.5 * 0.8 = 0.40,  momentum_term = 0.5 * 0.4 = 0.20

    The split is the one weight `scoring_bars.py` introduces -- no config
    field carries it -- so it is pinned here as a declared prior. A future
    sweepable field would change these numbers and should fail this test
    rather than slide past it.
    """
    assert VOL_MOMENTUM_WEIGHT_VOLATILITY == 0.5
    assert VOL_MOMENTUM_WEIGHT_MOMENTUM == 0.5
    config = root_config()
    score = component_score(
        scorer_for(Component.VOL_MOMENTUM, config),
        vector(full_values(vol_regime_score=0.8, momentum_score=0.4)),
    )
    assert score.magnitude == pytest.approx(0.60)
    assert score.points == pytest.approx(12.00)
    assert score.detail["volatility_term"] == pytest.approx(0.40)
    assert score.detail["momentum_term"] == pytest.approx(0.20)


def test_vol_momentum_does_not_zero_on_a_bar_outside_the_volatility_band():
    """A deliberate decision with a consequence, pinned so it stays visible.

    `vol_regime_score` is 0 at BOTH volatility extremes, so a multiplicative
    blend would zero the whole 20-point component on a dead or a panicking
    tape. The blend is additive, following 14.6 ("using a product instead
    would let one weak term zero an otherwise strong setup, which is a gate's
    job, not a score's"), so an untradable-volatility bar with strong momentum
    still contributes

        0.5 * 0.0 + 0.5 * 0.9 = 0.45  ->  0.45 * 20 = 9.0 points

    of the component's 20. That is 9 points of Flow Score on a bar section 3's
    `VolatilityGate` is supposed to reject, and it is the gate -- not this
    scorer -- that must reject it. Recorded rather than fixed by zeroing the
    score, because an invisible zero is worse than a stated rejection.
    """
    config = root_config()
    score = component_score(
        scorer_for(Component.VOL_MOMENTUM, config),
        vector(full_values(vol_regime_score=0.0, momentum_score=0.9)),
    )
    assert score.magnitude == pytest.approx(0.45)
    assert score.points == pytest.approx(9.0)
    assert score.enabled is True, "not a reason to call the component unmeasured"


# ---------------------------------------------------------------------------
# risk 2: UNAVAILABLE is not a measured zero
# ---------------------------------------------------------------------------


def _order_flow_through_the_real_computer(*, with_ticks: bool) -> ComponentScore:
    """ORDER_FLOW scored off the real computer, with and without a tick feed.

    200 bars clears `features/orderflow.py`'s 130-bar warmup, and the two
    datasets differ by exactly one thing: whether the tick series is attached.
    """
    config = root_config()
    bars = hand_built_bars(200)
    data = SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: bars},
        ticks={INTERVAL: balanced_ticks(200)} if with_ticks else None,
    )
    features = FeatureBundle([OrderFlowFeatures(config.order_flow, config.features)]).compute(
        view_at_last_bar(data)
    )
    return component_score(scorer_for(Component.ORDER_FLOW, config), features)


def test_a_balanced_tape_and_an_absent_tape_produce_the_same_numbers():
    """The premise of the whole risk, asserted before the distinction is.

    A tape with buy volume exactly equal to sell volume is a MEASUREMENT, and
    `features/orderflow.py` reports its magnitude as exactly 0.0 -- the same
    magnitude, the same direction and the same points an absent feed produces.
    So every numeric field is useless for telling them apart, and anything
    downstream that reads `magnitude` alone cannot distinguish "the tape is
    balanced" from "there is no tape". That is why the next test exists.
    """
    measured = _order_flow_through_the_real_computer(with_ticks=True)
    absent = _order_flow_through_the_real_computer(with_ticks=False)
    assert measured.magnitude == absent.magnitude == 0.0
    assert measured.direction == absent.direction == 0
    assert measured.points == absent.points == 0.0
    assert measured.weight == absent.weight == 25.0


def test_a_balanced_tape_and_an_absent_tape_are_distinguishable_in_the_score():
    """The distinction must live in the `ComponentScore`, not in a note.

    Four independent fields separate them, and each is load-bearing somewhere:

    * `enabled` -- what `FlowScore.available_points` and the strict-mode
      refusal read. This is THE field; the magnitude is not.
    * `quality` -- GOOD/DEGRADED is a measurement, MISSING is not one.
    * `detail["measured"]` -- the question a consumer should ask instead of
      reading `magnitude`, and the one that survives into the trade record.
    * `detail["attainable_points"]` -- 0.0 when nothing was measured, and the
      reachable ceiling when something was. On this tape one of the five terms
      was dropped, so the ceiling is 25.0 * 0.85 = 21.25 rather than 25.0, and
      the dropped weight was not reallocated.

    A fifth, at the aggregate: `available_points` differs by exactly the
    configured 25, which is how the shortfall stays reportable.
    """
    measured = _order_flow_through_the_real_computer(with_ticks=True)
    absent = _order_flow_through_the_real_computer(with_ticks=False)

    assert measured.enabled is True and absent.enabled is False
    assert measured.quality is not DataQuality.MISSING
    assert absent.quality is DataQuality.MISSING
    assert measured.detail["measured"] == 1.0
    assert absent.detail["measured"] == 0.0
    # 25.0 * 0.85 = 21.25: the absorption term had no inputs on a tape with no
    # price rejection, and its 0.15 of the weight was NOT redistributed.
    assert measured.detail["order_flow_available_weight"] == pytest.approx(0.85)
    assert measured.detail["attainable_points"] == pytest.approx(21.25)
    assert absent.detail["attainable_points"] == 0.0
    # neither was ablated in configuration; both report that separately
    assert measured.detail["component_enabled_in_config"] == 1.0
    assert absent.detail["component_enabled_in_config"] == 1.0


def test_neither_a_balanced_tape_nor_an_absent_tape_opposes_either_side():
    """An unobserved tape must never read as disagreement.

    Both states have `direction == 0`, so `opposes()` is False for both sides
    and neither contributes to `opposing_points`. That is correct and it is
    also the trap: it means a gate CANNOT distinguish them through the
    directional interface, and a setup that needs order-flow confirmation has
    to consult `enabled` (section 5's "no proxy" rule, surfaced through
    `require_orderflow_confirmation`). Reporting "order flow disagreed" about
    a tape nobody observed would be inventing data.
    """
    measured = _order_flow_through_the_real_computer(with_ticks=True)
    absent = _order_flow_through_the_real_computer(with_ticks=False)
    for score in (measured, absent):
        assert not score.opposes(Side.LONG)
        assert not score.opposes(Side.SHORT)
        assert not score.agrees_with(Side.LONG)
        assert not score.agrees_with(Side.SHORT)
    assert measured.enabled is not absent.enabled, "the only usable difference"


def test_an_unavailable_component_keeps_its_configured_weight():
    """The gap is reported by `max_points - available_points`, not hidden.

    An UNAVAILABLE component keeps `weight` at its configured value and turns
    `enabled` off. Zeroing the weight instead would make a 100-point scale out
    of 75 points and there would be nothing left to report. With both
    feed-dependent components absent:

        available = 20 + 15 + 20 = 55.0,  max = 100.0,  shortfall = 45.0

    which is the brief's "55.0 of 100.0 computable", reconstructed from the
    component scores alone.
    """
    config = root_config()
    absent_grades = {
        key: DataQuality.MISSING
        for key in ORDER_FLOW_READS + OPTIONS_FLOW_READS
    }
    features = vector(
        full_values(order_flow_available=0.0),
        grades=absent_grades,
    )
    scores = {
        s.component: component_score(s, features) for s in all_scorers(config)
    }
    flow = FlowScore(symbol=SYMBOL, ts=TS, components=scores, max_points=100.0)
    assert flow.available_points == pytest.approx(55.0)
    assert flow.max_points == 100.0
    assert flow.max_points - flow.available_points == pytest.approx(45.0)
    assert scores[Component.ORDER_FLOW].weight == 25.0
    assert scores[Component.OPTIONS_FLOW].weight == 20.0
    assert scores[Component.ORDER_FLOW].enabled is False
    assert scores[Component.OPTIONS_FLOW].enabled is False


def test_enabled_is_the_one_question_all_five_components_answer_alike():
    """`enabled` is the uniform interface; the detail keys are NOT.

    The two modules spell the same fact differently in `detail`:
    `scoring_bars` writes `available` (1.0/0.0) with a `quality_reason` code,
    `scoring_flow` writes `measured` with `attainable_points`. Neither carries
    the other's key, so a consumer that asked `detail["measured"]` of all five
    components would get a `KeyError` on three of them and one that asked
    `detail["available"]` would get it on two.

    REPORTED, not fixed: the aggregator reads `ComponentScore.enabled`, which
    both modules set identically, so nothing downstream is wrong today. What
    this pins is that `enabled` really is uniform -- so the divergence stays a
    naming inconsistency in a diagnostic rather than becoming a correctness
    problem if someone starts reading `detail` instead.
    """
    config = root_config()
    features = vector(full_values())
    scores = {s.component: component_score(s, features) for s in all_scorers(config)}
    assert all(s.enabled is True for s in scores.values())

    bars_detail = set(scores[Component.STRUCTURE].detail)
    flow_detail = set(scores[Component.ORDER_FLOW].detail)
    assert "available" in bars_detail and "available" not in flow_detail
    assert "measured" in flow_detail and "measured" not in bars_detail


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_an_ablated_component_is_unavailable_and_not_a_measured_zero(component):
    """A Phase 8 ablation is not a measurement of zero.

    `enabled_components` is the ablation switch. Turning a component off must
    produce the same UNAVAILABLE shape a missing feed does -- weight kept,
    `enabled` False -- because an ablated component whose weight was spread
    over the survivors would make the ablation unmeasurable: the score would
    not fall and the experiment would read as "the component contributed
    nothing".
    """
    config = root_config(
        flow_score=flow_score_config(
            enabled_components={c: c is not component for c in Component}
        )
    )
    score = component_score(scorer_for(component, config), vector(full_values()))
    assert score.enabled is False
    assert score.magnitude == 0.0
    assert score.direction == 0
    assert score.weight == PRIOR_WEIGHTS[component], "weight kept, so the gap shows"
    assert score.quality is DataQuality.MISSING
    reason_field = score.detail.get("quality_reason")
    if reason_field is not None:  # scoring_bars encodes the reason as a code
        assert reason_field == REASON_COMPONENT_DISABLED
    else:  # scoring_flow carries it in words on the ComponentScoring
        assert score.detail["component_enabled_in_config"] == 0.0


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_a_dataset_refusal_makes_the_component_unavailable_in_both_modules(component):
    """`data/quality.py`'s verdict is CONSUMED, not re-derived.

    `availability()` has already decided which components a dataset can
    support. A scorer that re-derived it from feeds and feed requirements
    would be a second source of truth that could disagree. Here the vector
    looks perfectly healthy and the refusal alone must disable the component
    -- in both modules, so all five answer to one availability report.
    """
    config = root_config()
    verdict = ComponentAvailability(
        component=component,
        computable=False,
        quality=DataQuality.MISSING,
        weight=PRIOR_WEIGHTS[component],
        missing_required_feeds=(Feed.TICK_AGGREGATE,),
        note="hand-built refusal for this test",
    )
    score = component_score(
        scorer_for(component, config), vector(full_values()), verdict
    )
    assert score.enabled is False
    assert score.quality is DataQuality.MISSING
    assert score.magnitude == 0.0
    assert score.weight == PRIOR_WEIGHTS[component]


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_a_verdict_about_another_component_is_rejected_in_both_modules(component):
    """Scoring one component against another's feed verdict reports the wrong
    gap, so it raises instead of producing a confidently wrong number."""
    config = root_config()
    other = Component.LIQUIDITY if component is not Component.LIQUIDITY else Component.STRUCTURE
    verdict = ComponentAvailability(
        component=other,
        computable=False,
        quality=DataQuality.MISSING,
        weight=PRIOR_WEIGHTS[other],
    )
    with pytest.raises((ScorerError, ScoringError), match="wrong gap"):
        component_score(scorer_for(component, config), vector(full_values()), verdict)


def test_a_bars_only_run_reports_fifty_five_of_one_hundred_on_every_bar():
    """Not one bar: every bar of a long run, through the real feature layer.

    A single-bar assertion would pass on a dataset that happened to have one
    unusable bar. The claim being tested is structural -- the 45 points have
    no feed for the WHOLE run -- so the two feed-dependent components must be
    unavailable on every sampled bar and the three bars-only components
    available on every one, which puts `available_points` at exactly 55.0
    throughout.
    """
    run = scored_run(feeds="bars")
    assert len(run) >= 50, "premise: the run must have enough bars to mean anything"
    for ts, scores in run:
        flow = FlowScore(symbol=SYMBOL, ts=ts, components=scores, max_points=100.0)
        assert flow.available_points == pytest.approx(55.0), ts
        assert scores[Component.ORDER_FLOW].enabled is False, ts
        assert scores[Component.OPTIONS_FLOW].enabled is False, ts
        for component in (
            Component.STRUCTURE,
            Component.LIQUIDITY,
            Component.VOL_MOMENTUM,
        ):
            assert scores[component].enabled is True, (component, ts)
        assert 0.0 <= flow.score <= flow.available_points, ts


# ---------------------------------------------------------------------------
# risk 3: degradation that must survive the scorer
# ---------------------------------------------------------------------------


def test_the_no_quote_liquidity_cap_and_its_degraded_flag_both_arrive():
    """Section 5's "degrades to volume-percentile only, flagged DEGRADED".

    The chain is `features/liquidity.py` caps `liquidity_score` at
    `NO_QUOTE_SCORE_CAP` and grades the key DEGRADED -> the scorer's quality
    is the worst grade over the keys the magnitude is built from -> the
    `ComponentScore` is DEGRADED. If the last link dropped, the degradation
    report would end in the feature layer where nothing acts on it.

    Run bars-only over the real layer and take the first bar where the cap
    actually BINDS, which is the stronger claim than "the cap is in force":
    there the magnitude is exactly the cap, so the component contributes
    exactly `0.5 * 15 = 7.5` of its 15 points. The premise that such a bar
    exists is asserted, or the test would pass without testing anything.
    """
    assert NO_QUOTE_SCORE_CAP == 0.5, "the arithmetic below is stated against this"
    run = scored_run(feeds="bars")
    binding = [
        (ts, scores[Component.LIQUIDITY])
        for ts, scores in run
        if scores[Component.LIQUIDITY].magnitude >= NO_QUOTE_SCORE_CAP
    ]
    assert binding, "premise: no sampled bar binds the no-quote cap"
    ts, score = binding[0]
    assert score.magnitude == NO_QUOTE_SCORE_CAP
    assert score.points == pytest.approx(7.5)  # 0.5 * 15.0
    assert score.quality is DataQuality.DEGRADED
    assert score.enabled is True
    assert score.detail["spread_measured"] == 0.0, "the spread was configuration"
    assert score.detail["available"] == 1.0, "degraded is still a measurement"


def test_no_liquidity_magnitude_in_a_bars_only_run_exceeds_the_cap():
    """The cap is not a one-bar coincidence.

    `NO_QUOTE_SCORE_CAP` binds half the component's evidence away, and the
    consequence -- at most 7.5 of 15 points, on every bar of a quote-less
    dataset -- is the kind of thing that is true at the bar someone checked
    and false twenty bars later if the cap is applied in the wrong branch.
    """
    run = scored_run(feeds="bars")
    worst = max(scores[Component.LIQUIDITY].magnitude for _, scores in run)
    assert worst <= NO_QUOTE_SCORE_CAP
    for ts, scores in run:
        score = scores[Component.LIQUIDITY]
        assert score.points <= 7.5 + 1e-12, ts
        assert score.quality is DataQuality.DEGRADED, ts


def test_liquidity_degrades_when_only_the_time_of_day_profile_is_a_fallback():
    """The other half of the degradation, and it must not be confused.

    `features/liquidity.py` grades `spread_ticks` DEGRADED exactly when the
    spread is a fallback and `relative_volume` DEGRADED exactly when the
    time-of-day bucket could not supply a median. Those are different
    shortfalls with different consequences, and the scorer reads the grades
    rather than re-deriving either from the values -- so a bar whose spread
    was measured but whose volume profile was not must report
    `spread_measured = 1.0` and `profile_measured = 0.0`, and still be
    DEGRADED overall.
    """
    config = root_config()
    score = component_score(
        scorer_for(Component.LIQUIDITY, config),
        vector(
            full_values(liquidity_score=0.62),
            grades={
                "relative_volume": DataQuality.DEGRADED,
                "volume_trend": DataQuality.DEGRADED,
                "liquidity_score": DataQuality.DEGRADED,
                "participation_cost_ticks": DataQuality.DEGRADED,
            },
        ),
    )
    assert score.quality is DataQuality.DEGRADED
    assert score.detail["spread_measured"] == 1.0
    assert score.detail["profile_measured"] == 0.0
    assert score.magnitude == 0.62, "a grade is not a haircut"
    assert score.points == pytest.approx(0.62 * 15.0)


def test_liquidity_stays_enabled_when_it_degrades_so_its_points_stay_on_the_scale():
    """Section 5 says "degrades", not "unavailable", and the difference is 15
    points of scale.

    Marking a degraded liquidity reading UNAVAILABLE would shrink
    `available_points` to 85 and, under `strict_component_availability`,
    trigger a refusal section 5 never asks for -- on every bar of every
    quote-less dataset, which is most real research data.
    """
    config = root_config()
    features = vector(
        full_values(liquidity_score=0.5),
        grades={"liquidity_score": DataQuality.DEGRADED,
                "spread_ticks": DataQuality.DEGRADED,
                "spread_percentile": DataQuality.DEGRADED},
    )
    scores = {s.component: component_score(s, features) for s in all_scorers(config)}
    flow = FlowScore(symbol=SYMBOL, ts=TS, components=scores, max_points=100.0)
    assert scores[Component.LIQUIDITY].enabled is True
    assert flow.available_points == pytest.approx(100.0)
    # FlowScore.quality is the worst over enabled components, so the
    # degradation reaches the decision rather than stopping at the component
    assert flow.quality is DataQuality.DEGRADED


def test_the_eod_options_cap_and_its_flag_both_reach_the_component_score():
    """Section 5 requires the EOD sub-score "capped AND flagged".

    Hand-built so every number is written down. An end-of-day chain whose
    uncapped magnitude is 0.80, capped at `eod_degraded_cap_fraction = 0.50`:

        magnitude          = min(0.80, 0.50)        = 0.50
        points             = 0.50 * 20.0            = 10.00
        points withheld    = 20.0 * (0.80 - 0.50)   =  6.00
        attainable points  = 20.0 * min(1.0, 0.50)  = 10.00

    All four must be on the `ComponentScore` itself: a consumer must never
    have to go back to the `FeatureVector` to find out that the number it is
    holding was capped.
    """
    config = root_config()
    score = component_score(
        scorer_for(Component.OPTIONS_FLOW, config),
        vector(
            full_values(
                options_flow_magnitude=0.50,
                options_flow_uncapped_magnitude=0.80,
                options_eod_capped=1.0,
                options_flow_timing_available=0.0,
            ),
            grades={"options_flow_magnitude": DataQuality.DEGRADED},
        ),
    )
    assert score.magnitude == 0.50
    assert score.points == pytest.approx(10.00)
    assert score.detail["options_eod_capped"] == 1.0
    assert score.detail["options_flow_uncapped_magnitude"] == 0.80
    assert score.detail["eod_cap_fraction"] == EOD_CAP
    assert score.detail["points_withheld_by_cap"] == pytest.approx(6.00)
    assert score.detail["attainable_points"] == pytest.approx(10.00)
    assert score.quality is DataQuality.DEGRADED
    assert score.enabled is True, "EOD positioning is a measurement, not an absence"


def test_the_eod_ceiling_is_the_tighter_of_the_cap_and_the_available_weight():
    """Two independent ceilings, and the reported one must be the binding one.

    Terms carrying 0.4 of the weight had no inputs, so the uncapped magnitude
    could reach at most 0.60; the EOD cap is 0.50. On an EOD chain the ceiling
    is `min(0.60, 0.50) = 0.50 -> 10.0` points, and on the intraday twin of
    the same reading it is `0.60 -> 12.0`. Reporting 1.0 as the ceiling in
    either case would make a magnitude read against a reachable maximum it
    never had.
    """
    config = root_config()
    scorer = scorer_for(Component.OPTIONS_FLOW, config)
    eod = component_score(
        scorer,
        vector(
            full_values(
                options_flow_magnitude=0.50,
                options_flow_uncapped_magnitude=0.60,
                options_available_weight=0.60,
                options_terms_available=3.0,
                options_eod_capped=1.0,
                options_flow_timing_available=0.0,
            ),
            grades={"options_flow_magnitude": DataQuality.DEGRADED},
        ),
    )
    intraday = component_score(
        scorer,
        vector(
            full_values(
                options_flow_magnitude=0.55,
                options_flow_uncapped_magnitude=0.55,
                options_available_weight=0.60,
                options_terms_available=3.0,
                options_eod_capped=0.0,
                options_flow_timing_available=1.0,
            ),
            grades={"options_flow_magnitude": DataQuality.DEGRADED},
        ),
    )
    assert eod.detail["attainable_points"] == pytest.approx(10.0)  # 20 * min(.6,.5)
    assert intraday.detail["attainable_points"] == pytest.approx(12.0)  # 20 * 0.6


def test_the_eod_cap_does_not_scale_or_suppress_the_direction():
    """The cap is a cap on MAGNITUDE. A capped direction would be section 7's
    conflation wearing a degradation's clothes.

    An end-of-day chain and its intraday twin, same bearish lean: the
    magnitudes differ (0.50 against 0.80) and the directions do not. The cap's
    effect on the direction's INFLUENCE is already correct without touching
    it, because `opposing_points` weights a direction by `points` and the cap
    has already reduced those: 10.0 opposing points instead of 16.0.
    """
    config = root_config()
    scorer = scorer_for(Component.OPTIONS_FLOW, config)
    eod = component_score(
        scorer,
        vector(
            full_values(
                options_flow_magnitude=0.50,
                options_flow_uncapped_magnitude=0.80,
                options_flow_direction=-1.0,
                options_eod_capped=1.0,
                options_flow_timing_available=0.0,
            ),
            grades={"options_flow_magnitude": DataQuality.DEGRADED},
        ),
    )
    intraday = component_score(
        scorer,
        vector(
            full_values(
                options_flow_magnitude=0.80,
                options_flow_uncapped_magnitude=0.80,
                options_flow_direction=-1.0,
            ),
        ),
    )
    assert eod.direction == intraday.direction == -1
    assert eod.magnitude == 0.50 and intraday.magnitude == 0.80
    # 0.50 * 20 = 10.0 against 0.80 * 20 = 16.0
    assert eod.points == pytest.approx(10.0)
    assert intraday.points == pytest.approx(16.0)


def test_the_eod_cap_binds_through_the_real_computer_and_the_detail_adds_up():
    """The same chain end to end, where the numbers are the generator's.

    A hand-built vector can satisfy every invariant and still not be what
    `features/optionsflow.py` emits. So the run is done through the real
    computer on an EOD chain, and the arithmetic is checked as a RELATION
    between the fields the score reports rather than against a constant read
    back from the implementation:

        magnitude        == eod_cap_fraction            (the cap bound)
        uncapped         >  magnitude                   (it had somewhere to fall from)
        withheld         == weight * (uncapped - magnitude)
        attainable       >= points                      (a ceiling is above the value)
    """
    run = scored_run(feeds="all", options_intraday=False)
    capped = [
        scores[Component.OPTIONS_FLOW]
        for _, scores in run
        if scores[Component.OPTIONS_FLOW].detail.get("options_eod_capped") == 1.0
    ]
    assert capped, "premise: the EOD cap must bind somewhere in the run"
    for score in capped:
        assert score.quality is DataQuality.DEGRADED
        assert score.enabled is True
        assert score.magnitude == pytest.approx(EOD_CAP)
        uncapped = score.detail["options_flow_uncapped_magnitude"]
        assert uncapped > score.magnitude
        assert score.detail["points_withheld_by_cap"] == pytest.approx(
            20.0 * (uncapped - score.magnitude)
        )
        assert score.detail["attainable_points"] >= score.points - 1e-12


def test_an_intraday_chain_is_never_capped_and_carries_no_reason():
    """The control for the cap tests: the same generator, intraday chains.

    If the cap were applied on the wrong branch -- to every bar rather than to
    end-of-day ones -- the previous tests would still pass. This is the half
    that fails in that case: on intraday chains the flag is never raised,
    nothing is withheld, and a fully-available reading is graded GOOD with an
    empty reason, which `ComponentScoring` documents as the only state with
    nothing to say.
    """
    config = root_config()
    scorer = OptionsFlowScorer(config)
    run = scored_run(feeds="all", options_intraday=True)
    measured = [scores[Component.OPTIONS_FLOW] for _, scores in run
                if scores[Component.OPTIONS_FLOW].enabled]
    assert measured, "premise: the intraday run must measure the component somewhere"
    for score in measured:
        assert score.detail["options_eod_capped"] == 0.0
        assert score.detail["points_withheld_by_cap"] == 0.0
        assert score.detail["options_flow_timing_available"] == 1.0

    good = component_score(scorer, vector(full_values()))
    assert good.quality is DataQuality.GOOD
    assert scorer.score(vector(full_values())).reason == ""


def test_a_capped_magnitude_that_is_not_the_configured_cap_is_a_mismatch():
    """FIXED BUG: the configuration cross-check was one-sided.

    `features/optionsflow.py` applies `magnitude = min(uncapped, cap)` and
    raises the flag only when the cap BOUND, so a flagged bar leaves the
    magnitude AT the fraction exactly. The scorer's docstring says as much
    ("flag set => the magnitude equals the configured
    eod_degraded_cap_fraction"), but the check only looked for a magnitude
    ABOVE the cap. A scorer built from a config whose cap is LOOSER than the
    computer's therefore passed every invariant while reporting
    `eod_cap_fraction` and `attainable_points` for a cap that was never
    applied -- the mirror image of the unflagged reduction the module calls
    "exactly the silent degradation section 5 requires to be visible".

    Both directions now raise. The 0.30 case is the one that used to pass.
    """
    config = root_config()
    scorer = OptionsFlowScorer(config)
    for magnitude in (0.30, 0.80):
        with pytest.raises(ScoringError, match="different configurations"):
            scorer.score(
                vector(
                    full_values(
                        options_flow_magnitude=magnitude,
                        options_flow_uncapped_magnitude=0.90,
                        options_eod_capped=1.0,
                        options_flow_timing_available=0.0,
                    ),
                    grades={"options_flow_magnitude": DataQuality.DEGRADED},
                )
            )
    # and the matching configuration still scores
    tight = root_config(options_flow=OptionsFlowConfig(eod_degraded_cap_fraction=0.30))
    scored = OptionsFlowScorer(tight).score(
        vector(
            full_values(
                options_flow_magnitude=0.30,
                options_flow_uncapped_magnitude=0.90,
                options_eod_capped=1.0,
                options_flow_timing_available=0.0,
            ),
            grades={"options_flow_magnitude": DataQuality.DEGRADED},
        )
    )
    assert scored.magnitude == 0.30
    # 20.0 * (0.90 - 0.30) = 12.0
    assert scored.score.detail["points_withheld_by_cap"] == pytest.approx(12.0)


def test_an_unflagged_reduction_raises():
    """The pair that makes the previous test's asymmetry visible.

    A magnitude below its uncapped value with the flag clear means something
    reduced the score and nothing says what. Raising rather than repairing is
    the right call: every available repair -- trust the flag, trust the
    magnitudes, recompute the cap -- picks one of two contradictory statements
    about one bar.
    """
    config = root_config()
    with pytest.raises(ScoringError, match="unflagged reduction"):
        OptionsFlowScorer(config).score(
            vector(
                full_values(
                    options_flow_magnitude=0.30,
                    options_flow_uncapped_magnitude=0.90,
                    options_eod_capped=0.0,
                )
            )
        )


def test_a_dropped_order_flow_term_lowers_the_attainable_ceiling_not_the_weight():
    """DEGRADED order flow: measured, on a lower ceiling, nothing reallocated.

    With 0.6 of the term weight carrying real inputs, the attainable ceiling
    is `25.0 * 0.6 = 15.0` points while `weight` stays at the configured 25.0
    and the component stays enabled. The dropped 0.4 is reported as a lower
    ceiling, never as a smaller weight -- a smaller weight would shrink
    `available_points` and read as a missing feed.
    """
    config = root_config()
    score = component_score(
        scorer_for(Component.ORDER_FLOW, config),
        vector(
            full_values(order_flow_magnitude=0.5, order_flow_available_weight=0.6),
            grades={
                "order_flow_magnitude": DataQuality.DEGRADED,
                "order_flow_direction": DataQuality.DEGRADED,
            },
        ),
    )
    assert score.enabled is True
    assert score.quality is DataQuality.DEGRADED
    assert score.weight == 25.0
    assert score.detail["attainable_points"] == pytest.approx(15.0)
    assert score.points == pytest.approx(12.5)  # 0.5 * 25.0


# ---------------------------------------------------------------------------
# risk 4: bounds across a long run
# ---------------------------------------------------------------------------


def test_the_long_run_actually_exercises_the_paths_the_bounds_tests_rely_on():
    """Coverage premise, so the bounds tests below cannot pass vacuously.

    A run in which every component were unavailable would satisfy every bound
    trivially. Asserted: all five components are measured somewhere, both
    signs and a flat reading appear, the options cap binds on some bars and
    the component is unavailable on others, and the sample is long enough to
    be worth the name.
    """
    run = scored_run(feeds="all", options_intraday=False)
    assert len(run) >= 50
    seen_directions: dict[Component, set[int]] = {c: set() for c in Component}
    measured: dict[Component, int] = {c: 0 for c in Component}
    for _, scores in run:
        for component, score in scores.items():
            seen_directions[component].add(score.direction)
            measured[component] += int(score.enabled)
    for component in EVERY_COMPONENT:
        assert measured[component] > 0, component
    for component in DIRECTIONAL:
        assert seen_directions[component] & {-1, 1}, component
    assert seen_directions[Component.LIQUIDITY] == {0}
    capped = sum(
        1
        for _, scores in run
        if scores[Component.OPTIONS_FLOW].detail.get("options_eod_capped") == 1.0
    )
    assert 0 < capped < len(run), "both the capped and uncapped paths must appear"
    assert measured[Component.OPTIONS_FLOW] < len(run), (
        "the unavailable options path must appear too"
    )


@pytest.mark.parametrize("feeds", ["all", "bars"])
def test_every_magnitude_and_direction_stays_in_bounds_over_a_long_run(feeds):
    """The contract's bounds, on real feature output rather than by hand.

    `ComponentScore` validates `magnitude` in [0, 1] and `direction` in
    [-1, 1] on construction, so a violation raises inside the scorer rather
    than reaching here -- which means this test is really asserting that a
    long run does not raise, plus the two things the contract does NOT check:
    that the direction is an INTEGER in {-1, 0, +1} (not 0.4 rounded by
    pydantic) and that points never exceed the configured weight.
    """
    run = scored_run(feeds=feeds)
    for ts, scores in run:
        for component, score in scores.items():
            assert isinstance(score.magnitude, float)
            assert 0.0 <= score.magnitude <= 1.0, (component, ts, score.magnitude)
            assert score.direction in (-1, 0, 1), (component, ts, score.direction)
            assert isinstance(score.direction, int)
            assert score.points <= score.weight + 1e-12, (component, ts)
            assert math.isfinite(score.points)


@pytest.mark.parametrize("feeds", ["all", "bars"])
def test_an_unavailable_score_is_always_a_zero_magnitude_and_a_zero_direction(feeds):
    """An unavailable component must cast no vote and carry no number.

    A non-zero magnitude on a disabled component would be harmless today
    (`points` is 0.0 when `enabled` is False) and lethal the moment someone
    read `magnitude` directly -- which the modules' own docstrings warn is
    "meaningless unless measured". A non-zero direction would be worse: it
    would put an unobserved component into `opposing_points`.
    """
    run = scored_run(feeds=feeds)
    unavailable = 0
    for ts, scores in run:
        for component, score in scores.items():
            if score.enabled:
                continue
            unavailable += 1
            assert score.magnitude == 0.0, (component, ts)
            assert score.direction == 0, (component, ts)
            assert score.points == 0.0, (component, ts)
            assert score.quality is DataQuality.MISSING, (component, ts)
            assert score.weight > 0.0, "the configured weight is kept"
    if feeds == "bars":
        assert unavailable == 2 * len(run), "both feed-dependent components, every bar"
    else:
        assert unavailable > 0


def test_every_detail_value_is_finite_over_a_long_run():
    """`detail` travels into the trade record and into SQLite.

    A NaN there is not a loud failure: it serializes, it compares unequal to
    itself, and it quietly poisons any later aggregation over the column.
    `scoring_bars` filters non-finite diagnostics out of `detail`; this is the
    assertion that nothing slips past for either module.
    """
    run = scored_run(feeds="all")
    for ts, scores in run:
        for component, score in scores.items():
            for key, value in score.detail.items():
                assert math.isfinite(value), (component, ts, key, value)


# ---------------------------------------------------------------------------
# risk 5: reading a key it never declared
# ---------------------------------------------------------------------------


def _perturbed(value: float) -> float:
    """A different finite value, whatever the original was."""
    return value * 3.0 + 1.25 if value != 0.0 else 7.5


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_no_scorer_is_moved_by_a_key_it_does_not_declare(component):
    """Each scorer's `reads` must be the whole truth about what it consumes.

    Every one of the 96 keys the real bundle emits is perturbed, one at a
    time, and the resulting `ComponentScore` must be byte-identical whenever
    the perturbed key is not in `reads`. An undeclared read would make the
    scorer's docstring, its `features_used` tuple and the traceability from a
    feature-module change to the scorers it breaks all wrong at once -- and it
    is exactly the kind of thing that happens when a diagnostic is added to a
    blend "temporarily".

    The complementary half is checked too: perturbing a DECLARED key must move
    something, or the declaration is overstating what the scorer uses.
    """
    config = root_config()
    scorer = scorer_for(component, config)
    declared = set(scorer.reads)
    baseline_values = full_values()
    baseline = component_score(scorer, vector(baseline_values))

    unmoved_declared: list[str] = []
    for key in sorted(baseline_values):
        bumped = dict(baseline_values)
        bumped[key] = _perturbed(baseline_values[key])
        if key in declared:
            # A declared key may be a direction or a flag whose perturbed value
            # breaks its own contract; those raise, which is itself movement.
            try:
                after = component_score(scorer, vector(bumped))
            except (ScorerError, ScoringError, ValueError):
                continue
            if after == baseline:
                unmoved_declared.append(key)
            continue
        after = component_score(scorer, vector(bumped))
        assert after == baseline, (
            f"{type(scorer).__name__} moved when undeclared key {key!r} changed"
        )

    assert unmoved_declared == [], (
        f"{type(scorer).__name__} declares {unmoved_declared} but nothing in the "
        "ComponentScore depends on them"
    )


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_every_key_a_scorer_reads_is_named_in_its_own_docstring(component):
    """The docstrings are the traceability, so they have to be complete.

    Both modules promise that a change to a feature module is traceable to the
    scorers it breaks "from either end": the code declares `reads` and the
    class docstring restates it. A key present in one and absent from the
    other breaks the half a human actually reads.
    """
    config = root_config()
    scorer = scorer_for(component, config)
    doc = type(scorer).__doc__ or ""
    missing = sorted(key for key in scorer.reads if key not in doc)
    assert missing == [], f"{type(scorer).__name__} reads undocumented {missing}"


def test_no_magnitude_is_built_from_an_unbounded_natural_unit_key():
    """One extreme observation must not be able to dominate a component.

    The feature modules emit unbounded natural-unit diagnostics next to their
    bounded scores -- `achievable_rr`, `atr`, `spread_ticks`, `dollar_volume`,
    `signed_delta_contracts`, `options_net_premium` -- and a weighted sum that
    read one of those would let a single outlier swamp a 20-point component.
    Checked structurally for the bars scorers (which declare
    `magnitude_keys`) and behaviourally for the feed-dependent ones, by
    driving one such key to an absurd value and requiring the magnitude not to
    move.
    """
    config = root_config()
    unbounded = (
        "achievable_rr",
        "atr",
        "spread_ticks",
        "dollar_volume",
        "signed_delta_contracts",
        "cvd_slope_normalized",
        "options_net_premium",
        "options_skew_25d",
        "options_total_volume",
        "nearest_zone_price",
    )
    for scorer in bars_scorers(config.flow_score):
        assert not set(scorer.magnitude_keys) & set(unbounded)

    baseline_values = full_values()
    for component in (Component.ORDER_FLOW, Component.OPTIONS_FLOW):
        scorer = scorer_for(component, config)
        baseline = component_score(scorer, vector(baseline_values))
        wild = dict(baseline_values)
        for key in unbounded:
            wild[key] = 1e12
        assert component_score(scorer, vector(wild)) == baseline


def test_the_feed_dependent_scorers_do_not_pass_unbounded_keys_into_detail():
    """`detail` is where a later consumer would go looking for them.

    `scoring_flow` states that the unbounded natural-unit diagnostics are
    neither read nor passed through, so nothing downstream can find them in
    `detail` and sum them. (`scoring_bars` takes the opposite position and
    reports them in `detail` deliberately -- a stated divergence, defensible
    only as long as nothing sums `detail`, which is why this assertion is
    scoped to the module that makes the promise.)
    """
    config = root_config()
    features = vector(full_values())
    for component in (Component.ORDER_FLOW, Component.OPTIONS_FLOW):
        detail = component_score(scorer_for(component, config), features).detail
        for key in (
            "signed_delta_contracts",
            "cvd_slope_normalized",
            "options_net_premium",
            "options_skew_25d",
            "options_total_volume",
            "options_snapshot_age_seconds",
        ):
            assert key not in detail, (component, key)


# ---------------------------------------------------------------------------
# risk 6: the weight is the configured weight, and nothing is redistributed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_the_weight_comes_from_config_for_every_component(component):
    """A hardcoded 20 or 25 would agree with the prior and fail here.

    Both weight sets are run through the same scorer on the same vector. The
    magnitude must not move (it is a function of the features alone) and the
    points must move with the weight.
    """
    prior = scorer_for(component, root_config())
    swept = scorer_for(
        component, root_config(flow_score=flow_score_config(weights=dict(SWEPT_WEIGHTS)))
    )
    features = vector(full_values(**magnitude_kwargs(component, 0.5)))
    a = component_score(prior, features)
    b = component_score(swept, features)
    assert a.magnitude == b.magnitude == 0.5
    assert a.weight == PRIOR_WEIGHTS[component]
    assert b.weight == SWEPT_WEIGHTS[component]
    assert a.points == pytest.approx(0.5 * PRIOR_WEIGHTS[component])
    assert b.points == pytest.approx(0.5 * SWEPT_WEIGHTS[component])


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("redistribute", [True, False])
def test_no_scorer_redistributes_weight_under_any_flag_combination(
    strict, redistribute
):
    """"This module never redistributes weight, under any configuration."

    Tested behaviourally rather than by reading the source: all four
    combinations of `strict_component_availability` and
    `redistribute_disabled_weight` must produce byte-identical component
    scores, including on a vector where two components are unavailable. The
    flags are the AGGREGATOR's question -- whether a Flow Score may be emitted
    from an incomplete set -- and a scorer that consulted them would be a
    second place where the decision is made.
    """
    baseline_config = root_config()
    flagged_config = root_config(
        flow_score=flow_score_config(
            strict_component_availability=strict,
            redistribute_disabled_weight=redistribute,
        )
    )
    features = vector(
        full_values(order_flow_available=0.0),
        grades={
            key: DataQuality.MISSING
            for key in ORDER_FLOW_READS + OPTIONS_FLOW_READS
        },
    )
    baseline = {
        s.component: component_score(s, features) for s in all_scorers(baseline_config)
    }
    flagged = {
        s.component: component_score(s, features) for s in all_scorers(flagged_config)
    }
    assert baseline == flagged
    assert sum(s.weight for s in flagged.values()) == pytest.approx(100.0)
    enabled_weight = sum(s.weight for s in flagged.values() if s.enabled)
    assert enabled_weight == pytest.approx(55.0), "still 55, never spread to 100"


def test_as_components_feeds_a_flow_score_and_refuses_a_duplicate():
    """The handoff to the aggregate, and the one way it can corrupt silently.

    `as_components` is how a set of `ComponentScoring`s becomes the dict
    `FlowScore` takes. A duplicate component would let one score overwrite
    another and the aggregate would be short a component without saying so.
    """
    config = root_config()
    features = vector(full_values())
    scorings = [s.score(features) for s in feed_dependent_scorers(config)]
    mapped = as_components(scorings)
    assert set(mapped) == {Component.ORDER_FLOW, Component.OPTIONS_FLOW}
    with pytest.raises(ScoringError, match="one score per component"):
        as_components(scorings + [scorings[0]])


# ---------------------------------------------------------------------------
# bound enforcement: a broken producer must be loud
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
@pytest.mark.parametrize("bad", [1.4, -0.2])
def test_a_magnitude_outside_the_unit_interval_raises_rather_than_clamping(
    component, bad
):
    """Clamping would turn a feature-layer regression into a quiet 20 points.

    A magnitude of 1.4 silently clipped to 1.0 is a component reporting its
    full weight on a broken measurement; a negative one would subtract points
    from the Flow Score. Both modules raise instead, and both must, because a
    component score is the last place the error is attributable to the module
    that produced it.
    """
    config = root_config()
    with pytest.raises((ScorerError, ScoringError)):
        component_score(
            scorer_for(component, config),
            vector(full_values(**magnitude_kwargs(component, bad))),
        )


@pytest.mark.parametrize("component", DIRECTIONAL, ids=lambda c: c.value)
@pytest.mark.parametrize("bad", [0.5, -0.5, 2.0])
def test_a_fractional_direction_raises_in_both_modules(component, bad):
    """FIXED BUG: `scoring_bars.unit_direction` reduced a fraction to a sign.

    Section 7 defines direction as a three-valued vote.
    `signals/scoring_flow.py` enforces it -- a value not within 1e-9 of -1, 0
    or +1 raises, "a fractional direction would be a magnitude wearing a
    direction's name". `scoring_bars.unit_direction` raised only OUTSIDE
    [-1, 1] and silently signed anything inside it, on the argument that "the
    keys in UNIT_DIRECTION_KEYS already emit exactly -1.0, 0.0 or +1.0, so
    nothing is lost" -- an assumption about the producing module that this was
    the one place able to check.

    The consequence of signing a fraction: a 0.3 "weakly bullish"
    `level_direction` becomes a FULL +1 vote carrying all 20 of STRUCTURE's
    points into `FlowScore.opposing_points`, and section 3's order-flow gate
    blocks on any opposing direction regardless of magnitude -- so a
    fractional direction would be promoted into a hard veto. Worse, STRUCTURE
    is where the pipeline takes the trade's SIDE from.

    Latent rather than live: every direction key in the real feature layer
    emits exactly -1.0, 0.0 or +1.0 today (verified across a long synthetic
    run). Fixed anyway, because the enforcement layer's whole job is to catch
    a producing module that broke its contract, and its sister module caught
    this one.
    """
    config = root_config()
    with pytest.raises((ScorerError, ScoringError)):
        component_score(
            scorer_for(component, config),
            vector(full_values(**direction_kwargs(component, bad))),
        )


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_a_magnitude_a_float_hair_above_one_is_accepted(component):
    """Float hygiene, not a transform.

    Both modules allow a violation of one part in 1e-9, which is the scale at
    which a weighted sum of five terms lands on 0.9999999999999999 instead of
    1.0. Raising there would make a correct feature module fail on arithmetic
    noise; the tolerance is far too narrow to absorb a real regression, which
    the previous test covers.
    """
    config = root_config()
    score = component_score(
        scorer_for(component, config),
        vector(full_values(**magnitude_kwargs(component, 1.0 + 5e-10))),
    )
    assert score.magnitude == 1.0
    assert score.points == pytest.approx(PRIOR_WEIGHTS[component])


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_a_non_finite_magnitude_raises(component):
    """A NaN magnitude would make `points` NaN and poison the aggregate.

    `FeatureVector.require` is the first line here and raises `ValueError`;
    what matters is that neither module has a path that lets a NaN through to
    `ComponentScore`, where pydantic's `ge`/`le` bounds do not catch it.
    """
    config = root_config()
    with pytest.raises((ScorerError, ScoringError, ValueError)):
        component_score(
            scorer_for(component, config),
            vector(full_values(**magnitude_kwargs(component, float("nan")))),
        )


def test_a_renamed_feature_key_is_loud_in_both_modules():
    """A key that vanished must not become a narrower score.

    `scoring_flow` raises and names the owning module. `scoring_bars` takes
    the other road for a REQUIRED key -- it reports the component UNAVAILABLE,
    which is the honest answer when the key carrying the magnitude is not
    there -- and both are safe. What neither may do is score the remainder,
    because a score computed from the keys that survived is on a different
    scale than the keys it kept.
    """
    config = root_config()
    without_magnitude = dict(full_values())
    del without_magnitude["order_flow_magnitude"]
    with pytest.raises(ScoringError, match="features/orderflow.py"):
        component_score(
            scorer_for(Component.ORDER_FLOW, config), vector(without_magnitude)
        )

    without_structure = dict(full_values())
    del without_structure["structure_magnitude"]
    score = component_score(
        scorer_for(Component.STRUCTURE, config), vector(without_structure)
    )
    assert score.enabled is False
    assert score.magnitude == 0.0
    assert score.weight == 20.0


# ---------------------------------------------------------------------------
# determinism and statelessness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_scoring_two_bars_in_either_order_gives_the_same_answers(component):
    """A scorer that accumulated anything would make the answer path-dependent.

    Bar B is scored cold, then again after bar A, and the two answers must be
    equal. This is the property that makes a backtest and a live run
    comparable: the same bar cannot score differently because of what was
    evaluated before it.
    """
    config = root_config()
    scorer = scorer_for(component, config)
    a = vector(full_values(**magnitude_kwargs(component, 0.3)), ts=TS)
    b = vector(full_values(**magnitude_kwargs(component, 0.7)), ts=TS_NEXT)

    cold = component_score(scorer, b)
    component_score(scorer, a)
    warm = component_score(scorer, b)
    assert cold == warm
    assert cold.magnitude == 0.7


@pytest.mark.parametrize("component", EVERY_COMPONENT, ids=lambda c: c.value)
def test_two_scorers_built_from_two_equal_configs_agree_bit_for_bit(component):
    """Nothing is captured at construction.

    `__init__` takes configuration only -- there is no market-data parameter
    and no accessor through which one could arrive -- which is what makes the
    point-in-time guarantee inherited from the audited feature layer rather
    than re-argued in these modules. Two independently built scorers on two
    independently built but equal configs must therefore be
    indistinguishable.
    """
    first = scorer_for(component, root_config())
    second = scorer_for(component, root_config())
    features = vector(full_values(**magnitude_kwargs(component, 0.42)))
    assert component_score(first, features) == component_score(second, features)
    assert component_score(first, features).to_init_dict() == (
        component_score(second, features).to_init_dict()
    )


def test_the_order_the_scorers_are_called_in_does_not_matter():
    """Five independent translations, so the loop order is free.

    If any scorer shared mutable state with another -- a memo, a cache, a
    module-level accumulator -- the aggregate would depend on iteration order,
    and `FlowScore` keys by `Component` precisely so that nothing downstream
    imposes one.
    """
    config = root_config()
    features = vector(full_values())
    forward = {s.component: component_score(s, features) for s in all_scorers(config)}
    backward = {
        s.component: component_score(s, features)
        for s in reversed(all_scorers(config))
    }
    assert forward == backward
    assert FlowScore(
        symbol=SYMBOL, ts=TS, components=forward, max_points=100.0
    ).score == FlowScore(
        symbol=SYMBOL, ts=TS, components=backward, max_points=100.0
    ).score


def test_a_constructor_refuses_anything_that_is_not_configuration():
    """The one leak the lookahead audit cannot see through.

    A full-sample statistic captured at construction is invisible to
    `validation/lookahead.py` when it is handed an instance, which is why
    `features/base.py` bans a market-data constructor parameter for computers
    and both scorer modules keep the ban. A `FeatureVector` is the closest
    thing to market data a caller might plausibly pass by mistake.
    """
    config = root_config()
    features = vector(full_values())
    with pytest.raises(ScorerError):
        StructureScorer(features)  # type: ignore[arg-type]
    with pytest.raises(ScoringError):
        OrderFlowScorer(config.flow_score)  # type: ignore[arg-type]
    with pytest.raises(ScoringError):
        OptionsFlowScorer(features)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# divergences between the two modules, pinned rather than blessed
# ---------------------------------------------------------------------------


def test_the_two_modules_disagree_about_a_stale_key():
    """REPORTED DIVERGENCE, not an endorsement of either side.

    `scoring_bars` documents and implements `STALE -> available, flagged (a
    real measurement, late)`: the component keeps its weight on the scale and
    `DataConfig.min_quality_to_trade` is the `DataQualityGate`'s to apply at
    pipeline stage 2, because a scorer that applied it too would be a second,
    divergent source of truth. `scoring_flow` treats a STALE magnitude key as
    UNAVAILABLE -- "the value present on the key is a placeholder" -- which
    drops the weight off the scale and, under strict mode, turns a late feed
    into the structural `COMPONENT_DISABLED` refusal that the engine
    deliberately keeps distinct from `DATA_QUALITY`.

    They cannot both be right: these are five components of ONE score on ONE
    scale, and as written the scale depends on WHICH component went stale.
    `scoring_flow`'s own docstring also says staleness is graded once, by
    `data/quality.py` at the top of the pipeline, and "a scorer that re-judged
    it would be a second, unsynchronized staleness rule" -- which is what the
    STALE branch is.

    NOT FIXED, because it is unreachable through the real feature layer: no
    feature computer stamps STALE on any key (`data/quality.py` grades FEEDS
    stale, not features), and `availability()` never returns a STALE
    `ComponentAvailability`. So this pins the divergence where someone adding
    a STALE-stamping computer will see it, instead of discovering it as a
    40-point scale shift in Phase 7.
    """
    config = root_config()
    stale_everything = {key: DataQuality.STALE for key in full_values()}
    features = vector(full_values(), grades=stale_everything)

    liquidity = component_score(scorer_for(Component.LIQUIDITY, config), features)
    assert liquidity.enabled is True
    assert liquidity.quality is DataQuality.STALE
    assert liquidity.magnitude == BIG
    assert liquidity.points == pytest.approx(BIG * 15.0)

    order_flow = component_score(scorer_for(Component.ORDER_FLOW, config), features)
    assert order_flow.enabled is False
    assert order_flow.quality is DataQuality.MISSING
    assert order_flow.magnitude == 0.0

    flow = FlowScore(
        symbol=SYMBOL,
        ts=TS,
        components={
            s.component: component_score(s, features) for s in all_scorers(config)
        },
        max_points=100.0,
    )
    # 55 of 100, from staleness alone: the divergence in one number.
    assert flow.available_points == pytest.approx(55.0)


def test_the_bars_scorer_explains_a_dataset_level_refusal_the_way_it_scores_it():
    """FIXED BUG: `explain()` and `score()` told two different stories.

    `explain()` promises to say "the same thing as `detail['quality_reason']`"
    and to name the specific keys a code cannot. On a dataset verdict of
    `computable=True, quality=MISSING` -- a combination `ComponentAvailability`
    permits -- `score()` returns UNAVAILABLE with `REASON_FEED_UNAVAILABLE`,
    while `explain()` fell through to the per-key branch and reported
    "magnitude/direction key(s) [] graded MISSING": an empty key list, because
    the MISSING grade came from the dataset and not from any key. A reason
    naming no keys would have gone straight into `Signal.reasons`.

    Latent rather than live -- `availability()` only returns GOOD or DEGRADED
    alongside `computable=True` -- so the fix is to the explanation, not to
    the decision, which was already right.
    """
    config = root_config()
    scorer = scorer_for(Component.LIQUIDITY, config)
    features = vector(full_values())
    verdict = ComponentAvailability(
        component=Component.LIQUIDITY,
        computable=True,
        quality=DataQuality.MISSING,
        weight=15.0,
        note="dataset verdict with no usable grade",
    )
    score = component_score(scorer, features, verdict)
    reasons = scorer.explain(features, verdict)

    assert score.enabled is False
    assert score.detail["quality_reason"] == REASON_FEED_UNAVAILABLE
    assert len(reasons) == 1
    assert "UNAVAILABLE" in reasons[0]
    assert "[]" not in reasons[0], "a reason must not name an empty key list"
    assert "not redistributed" in reasons[0]


def test_a_degraded_dataset_verdict_reaches_the_bars_scorer_but_not_the_flow_scorer():
    """REPORTED DIVERGENCE in how far the dataset verdict is consumed.

    `scoring_bars` ANDs `availability.quality` into the component grade, "so a
    dataset graded DEGRADED cannot be reported GOOD on a bar that happens to
    look clean". `scoring_flow` consumes only the `computable=False` refusal
    and ignores the verdict's grade entirely, although `scoring_bars` claims
    it "applies the same rule to the other two components".

    NOT FIXED, and the consequence is bounded: `FlowScore.quality` is the
    worst grade over enabled components, so an aggregate containing any
    bars-only component on a degraded dataset is DEGRADED either way. What
    diverges is the PER-COMPONENT grade and the per-component reason, which is
    what section 12's rejection analysis reads. Which side is right is a
    genuine question -- a dataset-level DEGRADED is not a statement about this
    bar, and `scoring_flow` defers such judgements to the stage-2 gate on
    principle -- so it is reported rather than resolved by changing a module.
    """
    config = root_config()
    features = vector(full_values())

    liquidity_verdict = ComponentAvailability(
        component=Component.LIQUIDITY,
        computable=True,
        quality=DataQuality.DEGRADED,
        weight=15.0,
        degraded_feeds=(Feed.QUOTES,),
        note="no quote feed: spread and depth terms unavailable, volume only",
    )
    liquidity = component_score(
        scorer_for(Component.LIQUIDITY, config), features, liquidity_verdict
    )
    assert liquidity.quality is DataQuality.DEGRADED, "the dataset grade is ANDed in"
    assert liquidity.magnitude == BIG, "a grade is not a haircut"

    options_verdict = ComponentAvailability(
        component=Component.OPTIONS_FLOW,
        computable=True,
        quality=DataQuality.DEGRADED,
        weight=20.0,
        degraded_feeds=(Feed.OPTIONS_SNAPSHOT,),
        note="end-of-day options only: positioning without flow timing",
    )
    options = component_score(
        scorer_for(Component.OPTIONS_FLOW, config), features, options_verdict
    )
    assert options.quality is DataQuality.GOOD, (
        "the dataset grade is NOT consumed here -- the divergence this pins"
    )
