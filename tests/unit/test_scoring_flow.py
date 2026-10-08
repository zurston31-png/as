"""Smoke checks for the two feed-dependent component scorers.

Deliberately NOT the exhaustive suite -- a dedicated test agent follows.
What is here is the set of properties `signals/scoring_flow.py` would be
pointless without, so that a regression in any of them fails now rather
than in Phase 7:

1. UNAVAILABLE is not a zero magnitude, and the two are distinguishable in
   the emitted `ComponentScore` (`test_unavailable_and_a_measured_zero_...`).
2. The 45 points do not get redistributed: a bars-only assembly reports 55.0
   of 100.0 available (`test_bars_only_reports_...`).
3. The options EOD cap and its flag both survive into the `ComponentScore`,
   and the cap is shown BINDING against the intraday twin of the same chain.
4. Direction is carried separately: a strongly bearish order-flow reading
   emits a LARGE magnitude and `direction=-1`, and its mirror image emits a
   bit-identical magnitude with the sign flipped.
5. Determinism, bit-for-bit.
6. The invariant checks have teeth -- each raises `ScoringError` rather than
   scoring a vector whose keys contradict each other.

The cross-component half of the brief's adversarial case (one component
bearish, the rest bullish, high aggregate magnitude, no trade either way)
spans the aggregator and the gates, which another agent owns, and cannot be
asserted from this file. Point 4 is the half that lives here.

Nothing here measures predictive content, and nothing may: the only data at
this phase is `data/synthetic.py` and hand-built tapes.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.config.loader import default_config, load_config
from flow_model.config.schema import FeatureConfig, OrderFlowConfig
from flow_model.core.contracts import ComponentScore, FeatureVector, FlowScore
from flow_model.core.enums import Component, DataQuality, Feed
from flow_model.data.market_view import MarketView
from flow_model.data.quality import ComponentAvailability, QualityGrader
from flow_model.data.series import BarSeries, OptionsSeries, TickSeries, from_ns, to_ns_array
from flow_model.data.store import SymbolData
from flow_model.features.optionsflow import OptionsFlowFeatures
from flow_model.features.orderflow import OrderFlowFeatures
from flow_model.signals.scoring_flow import (
    OPTIONS_FLOW_NOTE_PREFIX,
    OPTIONS_FLOW_READS,
    ORDER_FLOW_NOTE_PREFIX,
    ORDER_FLOW_READS,
    ComponentScoring,
    OptionsFlowScorer,
    OrderFlowScorer,
    ScoringError,
    as_components,
    feed_dependent_scorers,
)

SYMBOL = "NQ"
INTERVAL = 300
TS0 = datetime(2021, 3, 1, 14, 35, tzinfo=timezone.utc)
HONEST_METHOD = "bid_ask_quote"

#: The exact-tape background row from `tests/unit/test_optionsflow.py`, whose
#: five terms are hand-computed there. Reproduced rather than imported so a
#: failure localizes to one file (that file's own convention).
BACKGROUND = {
    "call_volume": 600.0,
    "put_volume": 400.0,
    "call_premium": 300.0,
    "put_premium": 100.0,
    "call_oi": 6000.0,
    "put_oi": 4000.0,
    "call_oi_change": 500.0,
    "put_oi_change": 100.0,
    "delta_weighted_call_volume": 70.0,
    "delta_weighted_put_volume": 30.0,
    "iv_25d_put": 0.375,
    "iv_25d_call": 0.25,
    "gamma_exposure_proxy": 100.0,
}
NOW = {**BACKGROUND, "iv_25d_put": 0.5, "gamma_exposure_proxy": 500.0}

#: sum(w_i * |t_i|) over the exact tape's five terms, at the shipped weights.
UNCAPPED_EXACT = (
    0.25 * 0.5
    + 0.25 * 0.4
    + 0.20 * math.tanh(((500.0 - 100.0) / (6000.0 + 4000.0)) / 0.05)
    + 0.15 * 1.0
    + 0.15 * 1.0
)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def stamps(n: int, offset: int = 0) -> np.ndarray:
    return to_ns_array(
        TS0 + timedelta(seconds=INTERVAL * (offset + i + 1)) for i in range(n)
    )


def bar_series(n: int, mid: float = 100.0) -> BarSeries:
    close = np.full(n, float(mid))
    return BarSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n),
        interval_seconds=INTERVAL,
        columns={
            "open": close.copy(),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close.copy(),
            "volume": np.full(n, 1000.0),
        },
    )


def tick_series(n: int, *, buy: float = 60.0, sell: float = 40.0) -> TickSeries:
    return TickSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n),
        columns={
            "buy_volume": np.full(n, float(buy)),
            "sell_volume": np.full(n, float(sell)),
            "buy_trades": np.full(n, 6.0),
            "sell_trades": np.full(n, 4.0),
            "max_trade_size": np.full(n, 10.0),
        },
        meta={"classification_method": HONEST_METHOD},
    )


def options_series(rows: list[dict], *, intraday: bool) -> OptionsSeries:
    present = sorted({name for row in rows for name in row})
    return OptionsSeries(
        symbol=SYMBOL,
        ts_ns=stamps(len(rows)),
        columns={
            name: np.array(
                [float(row.get(name, float("nan"))) for row in rows], dtype=np.float64
            )
            for name in present
        },
        meta={
            "is_intraday": bool(intraday),
            "source": "test_intraday" if intraday else "test_eod",
        },
    )


def exact_tape() -> list[dict]:
    return [dict(BACKGROUND) for _ in range(20)] + [dict(NOW)]


def view_of(data: SymbolData) -> MarketView:
    cutoff = int(data.primary_bars.ts_ns[-1])
    return MarketView(data, now=from_ns(cutoff), now_ns=cutoff)


def order_flow_computer() -> OrderFlowFeatures:
    return OrderFlowFeatures(OrderFlowConfig(), FeatureConfig())


def options_computer() -> OptionsFlowFeatures:
    return OptionsFlowFeatures(load_config().options_flow)


def order_flow_vector(
    *, quality: DataQuality = DataQuality.GOOD, ts: datetime | None = None, **overrides
) -> FeatureVector:
    """A hand-built order-flow vector, every declared key present.

    The key set comes from the computer rather than a literal list, so a key
    added to `features/orderflow.py` appears here automatically and an absent
    one is a failure rather than a silently narrower vector.
    """
    base = {
        "order_flow_magnitude": 0.4,
        "order_flow_direction": 1.0,
        "order_flow_available": 1.0,
        "order_flow_available_weight": 1.0,
        "signed_delta": 0.5,
        "signed_delta_contracts": 120.0,
        "cvd_slope": 0.3,
        "cvd_slope_normalized": 0.08,
        "aggression_ratio": 0.6,
        "absorption": 0.0,
        "absorption_direction": 0.0,
        "avg_trade_size_percentile": 0.55,
        "max_trade_size_percentile": 0.6,
        "large_trade_event": 0.0,
        "classification_coverage": 1.0,
    }
    assert set(base) == set(order_flow_computer().keys), (
        "this builder is out of step with features/orderflow.py's declared keys"
    )
    base.update(overrides)
    return FeatureVector(
        symbol=SYMBOL,
        ts=ts or TS0,
        values=base,
        quality_by_key={key: quality for key in base},
    )


def options_flow_vector(
    *, quality: DataQuality = DataQuality.GOOD, **overrides
) -> FeatureVector:
    base = {
        "options_net_premium_imbalance": 0.5,
        "options_delta_volume_imbalance": 0.4,
        "options_oi_change_imbalance": 0.3,
        "options_skew_25d_pressure": 0.2,
        "options_gamma_exposure_pressure": 0.1,
        "options_net_premium": 200.0,
        "options_skew_25d": 0.125,
        "options_total_volume": 1000.0,
        "options_snapshot_age_seconds": 0.0,
        "options_available_weight": 1.0,
        "options_terms_available": 5.0,
        "options_flow_timing_available": 1.0,
        "options_eod_capped": 0.0,
        "options_sign_agreement": 1.0,
        "options_imbalance_consensus": 0.35,
        "options_flow_uncapped_magnitude": 0.35,
        "options_flow_magnitude": 0.35,
        "options_flow_direction": 1.0,
    }
    assert set(base) == set(options_computer().keys), (
        "this builder is out of step with features/optionsflow.py's declared keys"
    )
    base.update(overrides)
    return FeatureVector(
        symbol=SYMBOL,
        ts=TS0,
        values=base,
        quality_by_key={key: quality for key in base},
    )


@pytest.fixture
def scorers(config):
    return feed_dependent_scorers(config)


# ---------------------------------------------------------------------------
# premises
# ---------------------------------------------------------------------------


def test_the_note_prefixes_match_the_producing_computers():
    """The reason text is filtered out of the bundle's merged notes by prefix.

    A rename in `features/` would silently mute every reason this module
    quotes, so the linkage is asserted rather than assumed. The two spellings
    genuinely differ upstream.
    """
    assert ORDER_FLOW_NOTE_PREFIX == f"{OrderFlowFeatures.name}:"
    assert OPTIONS_FLOW_NOTE_PREFIX == f"{OptionsFlowFeatures.name}:"


def test_the_declared_read_sets_are_subsets_of_the_producers_keys():
    assert set(ORDER_FLOW_READS) <= set(order_flow_computer().keys)
    assert set(OPTIONS_FLOW_READS) <= set(options_computer().keys)


def test_unbounded_diagnostics_are_neither_read_nor_passed_through(scorers):
    """A weighted sum must not reach an unbounded natural-unit diagnostic."""
    banned = {
        "signed_delta_contracts",
        "cvd_slope_normalized",
        "options_net_premium",
        "options_skew_25d",
        "options_total_volume",
    }
    order, options = scorers
    assert banned.isdisjoint(order.reads)
    assert banned.isdisjoint(options.reads)
    assert banned.isdisjoint(order.score(order_flow_vector()).score.detail)
    assert banned.isdisjoint(options.score(options_flow_vector()).score.detail)


def test_weights_come_from_config_and_are_not_hardcoded():
    """A scorer reports whatever the config says, including a swept weight."""
    config = default_config()
    order, options = feed_dependent_scorers(config)
    assert order.weight == config.flow_score.weights[Component.ORDER_FLOW]
    assert options.weight == config.flow_score.weights[Component.OPTIONS_FLOW]

    swept = config.replace(
        flow_score=config.flow_score.replace(
            weights={
                Component.ORDER_FLOW: 10.0,
                Component.OPTIONS_FLOW: 10.0,
                Component.STRUCTURE: 30.0,
                Component.LIQUIDITY: 20.0,
                Component.VOL_MOMENTUM: 30.0,
            }
        )
    )
    assert OrderFlowScorer(swept).weight == 10.0
    assert OrderFlowScorer(swept).score(order_flow_vector()).score.weight == 10.0


def test_a_scorer_refuses_anything_but_configuration():
    with pytest.raises(ScoringError, match="FlowModelConfig and nothing else"):
        OrderFlowScorer(order_flow_vector())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# UNAVAILABLE is not a zero
# ---------------------------------------------------------------------------


def test_a_bars_only_tape_makes_order_flow_unavailable_through_the_real_computer(
    scorers,
):
    order, _ = scorers
    data = SymbolData(
        symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bar_series(200)}
    )
    vector = order_flow_computer()._not_ready(
        view_of(data), "required feed(s) absent: ['tick_aggregate']"
    )
    result = order.score(vector)

    assert result.measured is False
    assert result.score.enabled is False
    assert result.quality is DataQuality.MISSING
    assert result.points == 0.0
    assert result.direction == 0
    assert result.weight == 25.0  # the configured prior, still on the score
    assert result.score.detail["measured"] == 0.0
    assert "UNAVAILABLE" in result.reason
    assert "rather than redistributed" in result.reason


def test_unavailable_and_a_measured_zero_are_distinguishable(scorers):
    """The distinction the whole module exists to preserve.

    "never measured" and "measured, and the tape is balanced" are the same
    magnitude and must not be the same `ComponentScore`.
    """
    order, _ = scorers
    never_measured = order.score(
        order_flow_vector(
            quality=DataQuality.MISSING,
            order_flow_available=0.0,
            order_flow_magnitude=0.0,
            order_flow_direction=0.0,
        )
    )
    balanced_tape = order.score(
        order_flow_vector(order_flow_magnitude=0.0, order_flow_direction=0.0)
    )

    assert never_measured.magnitude == balanced_tape.magnitude == 0.0
    assert never_measured.measured is False and balanced_tape.measured is True
    assert never_measured.quality is DataQuality.MISSING
    assert balanced_tape.quality is DataQuality.GOOD
    assert never_measured.score != balanced_tape.score


def test_bars_only_reports_55_of_100_points_and_redistributes_nothing(scorers):
    """The honest consequence, assembled end to end.

    The other three components are hand-built here -- a separate agent owns
    their scorers -- purely so the arithmetic of the gap is exercised.
    """
    order, options = scorers
    unavailable = [
        order.score(order_flow_vector(quality=DataQuality.MISSING, order_flow_available=0.0)),
        options.score(options_flow_vector(quality=DataQuality.MISSING)),
    ]
    bars_only = {
        Component.STRUCTURE: 20.0,
        Component.LIQUIDITY: 15.0,
        Component.VOL_MOMENTUM: 20.0,
    }
    components = as_components(unavailable)
    for component, weight in bars_only.items():
        components[component] = ComponentScore(
            component=component, magnitude=1.0, direction=1, weight=weight
        )

    score = FlowScore(symbol=SYMBOL, ts=TS0, components=components)
    assert score.max_points == 100.0
    assert score.available_points == 55.0
    assert score.score == 55.0
    # The 45 unavailable points are reported, not spread over the rest: a
    # redistributing implementation would have produced 100.0 here.
    assert score.points_for(Component.ORDER_FLOW) == 0.0
    assert score.points_for(Component.OPTIONS_FLOW) == 0.0


def test_a_dataset_level_refusal_is_consumed_not_re_derived(scorers, config):
    """`data/quality.py` already decided; the scorer propagates its verdict."""
    order, _ = scorers
    data = SymbolData(
        symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bar_series(200)}
    )
    report = QualityGrader(config.data, config.flow_score).availability(data)
    verdict = report.components[Component.ORDER_FLOW]
    assert verdict.computable is False

    # The vector itself looks perfectly healthy; the dataset verdict still wins.
    result = order.score(order_flow_vector(), availability=verdict)
    assert result.measured is False
    assert result.quality is DataQuality.MISSING
    assert "data/quality.py reports" in result.reason
    assert Feed.TICK_AGGREGATE.value in result.reason


def test_a_computable_dataset_does_not_override_a_missing_vector(scorers):
    order, _ = scorers
    verdict = ComponentAvailability(
        component=Component.ORDER_FLOW,
        computable=True,
        quality=DataQuality.GOOD,
        weight=25.0,
    )
    result = order.score(
        order_flow_vector(quality=DataQuality.MISSING, order_flow_available=0.0),
        availability=verdict,
    )
    assert result.measured is False


def test_an_availability_report_for_the_wrong_component_raises(scorers):
    order, _ = scorers
    verdict = ComponentAvailability(
        component=Component.LIQUIDITY,
        computable=False,
        quality=DataQuality.MISSING,
        weight=15.0,
    )
    with pytest.raises(ScoringError, match="scoring one component against"):
        order.score(order_flow_vector(), availability=verdict)


def test_configuration_ablation_is_unavailable_and_says_so(config):
    ablated = config.replace(
        flow_score=config.flow_score.replace(
            enabled_components={
                **config.flow_score.enabled_components,
                Component.ORDER_FLOW: False,
            }
        )
    )
    result = OrderFlowScorer(ablated).score(order_flow_vector())
    assert result.measured is False
    assert result.weight == 25.0
    assert result.score.detail["component_enabled_in_config"] == 0.0
    assert "disabled in configuration" in result.reason


# ---------------------------------------------------------------------------
# the options cap
# ---------------------------------------------------------------------------


def test_the_eod_cap_binds_and_both_halves_reach_the_component_score(scorers, config):
    """One chain, two `is_intraday` flags. The difference is the cap."""
    _, options = scorers
    computer = options_computer()
    rows = exact_tape()

    def scored(intraday: bool) -> ComponentScoring:
        data = SymbolData(
            symbol=SYMBOL,
            primary_interval=INTERVAL,
            bars={INTERVAL: bar_series(len(rows))},
            options=options_series(rows, intraday=intraday),
        )
        return options.score(computer.compute(view_of(data)))

    intraday, eod = scored(True), scored(False)
    cap = config.options_flow.eod_degraded_cap_fraction

    # premise: the uncapped magnitude is above the cap, so the cap can bind
    assert intraday.magnitude == pytest.approx(UNCAPPED_EXACT)
    assert UNCAPPED_EXACT > cap

    assert intraday.quality is DataQuality.GOOD
    assert intraday.score.detail["options_eod_capped"] == 0.0
    assert intraday.score.detail["points_withheld_by_cap"] == 0.0

    assert eod.measured is True
    assert eod.quality is DataQuality.DEGRADED
    assert eod.magnitude == pytest.approx(cap)
    assert eod.score.detail["options_eod_capped"] == 1.0
    assert eod.score.detail["options_flow_uncapped_magnitude"] == pytest.approx(
        UNCAPPED_EXACT
    )
    assert eod.score.detail["eod_cap_fraction"] == cap
    assert eod.score.detail["points_withheld_by_cap"] == pytest.approx(
        options.weight * (UNCAPPED_EXACT - cap)
    )
    assert eod.score.detail["attainable_points"] == pytest.approx(options.weight * cap)
    assert "cap bound" in eod.reason
    assert "end-of-day" in eod.reason

    # the cap is a MAGNITUDE cap: direction is identical on both paths
    assert eod.direction == intraday.direction != 0


def test_the_cap_does_not_scale_or_suppress_the_direction(scorers):
    _, options = scorers
    capped = options.score(
        options_flow_vector(
            options_flow_timing_available=0.0,
            options_eod_capped=1.0,
            options_flow_uncapped_magnitude=0.9,
            options_flow_magnitude=0.5,
            options_flow_direction=-1.0,
            quality=DataQuality.DEGRADED,
        )
    )
    assert capped.direction == -1
    assert capped.magnitude == 0.5
    assert capped.score.detail["options_flow_uncapped_magnitude"] == 0.9


@pytest.mark.parametrize(
    "overrides, match",
    [
        # magnitude above the uncapped value: a cap can only reduce
        (
            {"options_flow_magnitude": 0.6, "options_flow_uncapped_magnitude": 0.5},
            "above options_flow_uncapped_magnitude",
        ),
        # flag set while nothing was reduced
        (
            {
                "options_eod_capped": 1.0,
                "options_flow_timing_available": 0.0,
                "options_flow_magnitude": 0.35,
                "options_flow_uncapped_magnitude": 0.35,
            },
            "the magnitudes say it did not",
        ),
        # flag set on an intraday chain
        (
            {
                "options_eod_capped": 1.0,
                "options_flow_timing_available": 1.0,
                "options_flow_magnitude": 0.5,
                "options_flow_uncapped_magnitude": 0.9,
            },
            "intraday chain is never capped",
        ),
        # capped magnitude above the configured cap
        (
            {
                "options_eod_capped": 1.0,
                "options_flow_timing_available": 0.0,
                "options_flow_magnitude": 0.8,
                "options_flow_uncapped_magnitude": 0.9,
            },
            "different configurations",
        ),
        # an unflagged reduction
        (
            {
                "options_eod_capped": 0.0,
                "options_flow_magnitude": 0.3,
                "options_flow_uncapped_magnitude": 0.9,
            },
            "unflagged reduction",
        ),
    ],
)
def test_the_cap_invariants_raise_rather_than_picking_a_story(
    scorers, overrides, match
):
    _, options = scorers
    with pytest.raises(ScoringError, match=match):
        options.score(options_flow_vector(**overrides))


# ---------------------------------------------------------------------------
# direction carried separately
# ---------------------------------------------------------------------------


def test_a_strongly_bearish_component_emits_a_large_magnitude(scorers):
    """Section 7's rule, the half this file owns.

    A bearish reading is a LARGE magnitude with `direction=-1`. The magnitude
    does not shrink because the direction is negative, and the sign does not
    leak into the magnitude. Refusing the long is the gates' job.
    """
    order, _ = scorers
    bearish = order.score(
        order_flow_vector(
            order_flow_magnitude=0.95,
            order_flow_direction=-1.0,
            signed_delta=-0.9,
            cvd_slope=-0.9,
            aggression_ratio=0.05,
        )
    )
    bullish = order.score(
        order_flow_vector(
            order_flow_magnitude=0.95,
            order_flow_direction=1.0,
            signed_delta=0.9,
            cvd_slope=0.9,
            aggression_ratio=0.95,
        )
    )

    assert bearish.magnitude == bullish.magnitude == 0.95
    assert bearish.points == bullish.points == pytest.approx(0.95 * 25.0)
    assert bearish.direction == -1 and bullish.direction == 1
    # and the contradiction gate can see the opposition it must act on
    from flow_model.core.enums import Side

    assert bearish.score.opposes(Side.LONG) is True
    assert bearish.score.agrees_with(Side.SHORT) is True


@pytest.mark.parametrize("bad", [0.5, -0.5, 2.0, -2.0])
def test_a_fractional_direction_raises(scorers, bad):
    order, _ = scorers
    with pytest.raises(ScoringError, match="three-valued"):
        order.score(order_flow_vector(order_flow_direction=bad))


@pytest.mark.parametrize("bad", [1.5, -0.25])
def test_a_magnitude_outside_the_unit_interval_raises(scorers, bad):
    order, _ = scorers
    with pytest.raises(ScoringError, match=r"outside \[0, 1\]"):
        order.score(order_flow_vector(order_flow_magnitude=bad))


def test_a_renamed_feature_key_raises_and_names_its_owner(scorers):
    order, _ = scorers
    vector = order_flow_vector()
    narrowed = FeatureVector(
        symbol=vector.symbol,
        ts=vector.ts,
        values={k: v for k, v in vector.values.items() if k != "cvd_slope"},
        quality_by_key={
            k: q for k, q in vector.quality_by_key.items() if k != "cvd_slope"
        },
    )
    with pytest.raises(ScoringError, match="features/orderflow.py"):
        order.score(narrowed)


# ---------------------------------------------------------------------------
# degradation and determinism
# ---------------------------------------------------------------------------


def test_a_dropped_order_flow_term_degrades_and_lowers_the_attainable_ceiling(scorers):
    order, _ = scorers
    result = order.score(
        order_flow_vector(
            quality=DataQuality.DEGRADED, order_flow_available_weight=0.85
        )
    )
    assert result.measured is True
    assert result.quality is DataQuality.DEGRADED
    assert result.score.detail["attainable_points"] == pytest.approx(0.85 * 25.0)
    assert "NOT redistributed" in result.reason


def test_a_good_reading_carries_no_reason(scorers):
    order, options = scorers
    assert order.score(order_flow_vector()).reason == ""
    assert options.score(options_flow_vector()).reason == ""


def test_two_calls_on_one_vector_agree_bit_for_bit(scorers):
    order, options = scorers
    for scorer, vector in ((order, order_flow_vector()), (options, options_flow_vector())):
        first, second = scorer.score(vector), scorer.score(vector)
        assert first.score.model_dump_json() == second.score.model_dump_json()
        assert first == second


def test_two_independently_built_scorers_agree(config):
    vector = order_flow_vector()
    assert OrderFlowScorer(config).score(vector) == OrderFlowScorer(config).score(vector)


def test_real_order_flow_through_the_real_computer_scores(scorers):
    """End to end on a tick-fed tape: a measured component, bounded."""
    order, _ = scorers
    n = 200
    data = SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: bar_series(n)},
        ticks={INTERVAL: tick_series(n)},
    )
    result = order.score(order_flow_computer().compute(view_of(data)))
    assert result.measured is True
    assert 0.0 <= result.magnitude <= 1.0
    assert result.direction in (-1, 0, 1)
    assert result.quality in (DataQuality.GOOD, DataQuality.DEGRADED)
    assert result.score.features_used == ORDER_FLOW_READS


def test_as_components_refuses_two_scorings_of_one_component(scorers):
    order, _ = scorers
    twice = [order.score(order_flow_vector()), order.score(order_flow_vector())]
    with pytest.raises(ScoringError, match="one score per component"):
        as_components(twice)


def test_as_components_feeds_a_flow_score_directly(scorers):
    order, options = scorers
    components = as_components(
        [order.score(order_flow_vector()), options.score(options_flow_vector())]
    )
    assert set(components) == {Component.ORDER_FLOW, Component.OPTIONS_FLOW}
    assert all(isinstance(c, ComponentScore) for c in components.values())


def test_score_component_is_the_bare_contract_object(scorers):
    """The adapter for a caller that wants all five components in one shape."""
    order, options = scorers
    for scorer, vector in ((order, order_flow_vector()), (options, options_flow_vector())):
        bare = scorer.score_component(vector)
        assert isinstance(bare, ComponentScore)
        assert bare == scorer.score(vector).score
