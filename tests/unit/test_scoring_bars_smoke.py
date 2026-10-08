"""Smoke check for `signals/scoring_bars.py`.

Deliberately NOT the exhaustive suite -- a dedicated test agent follows.
What this file establishes is that the module runs against the real feature
layer, that its declared contracts hold, and that the one rule the phase
turns on is actually enforced:

    section 7: "Direction is handled separately from magnitude ... This
    avoids the common bug where a strong bearish component inflates a
    bullish score."

`test_a_bearish_component_keeps_its_magnitude_and_its_sign` and
`test_conflicting_components_oppose_both_sides` are that rule. They assert
that a strongly bearish component contributes a LARGE magnitude with
`direction = -1`, and that when one component disagrees with the rest the
resulting `FlowScore` has opposing weight against BOTH sides -- so no gate
can read a majority direction off it. The second half of the adversarial
requirement ("no trade is taken in either direction") belongs to the gates,
which this module does not own.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from flow_model.config.loader import load_config
from flow_model.core.contracts import ComponentScore, FeatureVector, FlowScore
from flow_model.core.enums import Component, DataQuality, Side
from flow_model.data.calendar import TradingCalendar
from flow_model.data.market_view import MarketView
from flow_model.data.series import from_ns
from flow_model.data.store import SymbolData
from flow_model.data.quality import ComponentAvailability
from flow_model.data.synthetic import SyntheticConfig, SyntheticMarketGenerator
from flow_model.features.base import FeatureBundle
from flow_model.features.levels import LevelFeatures
from flow_model.features.liquidity import LiquidityFeatures
from flow_model.features.momentum import MomentumFeatures
from flow_model.features.structure import StructureFeatures
from flow_model.features.volatility import VolatilityFeatures
from flow_model.signals.scoring_bars import (
    BOUNDED_MAGNITUDE_KEYS,
    REASON_COMPONENT_DISABLED,
    REASON_FEED_UNAVAILABLE,
    REASON_KEYS_ABSENT,
    REASON_KEY_MISSING,
    REASON_TEXT,
    VOL_MOMENTUM_WEIGHT_MOMENTUM,
    VOL_MOMENTUM_WEIGHT_VOLATILITY,
    BarsComponentScorer,
    LiquidityScorer,
    ScorerError,
    StructureScorer,
    VolMomentumScorer,
    bars_scorers,
    unit_direction,
)

SYMBOL = "NQ"
INTERVAL = 300
TS = datetime(2024, 3, 1, 15, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# hand-built vectors
# ---------------------------------------------------------------------------


def vector(values: dict[str, float], quality: dict[str, DataQuality] | None = None):
    """A FeatureVector carrying exactly `values`, GOOD unless told otherwise."""
    grades = {key: DataQuality.GOOD for key in values}
    grades.update(quality or {})
    return FeatureVector(symbol=SYMBOL, ts=TS, values=values, quality_by_key=grades)


def bullish_structure(magnitude: float = 0.9) -> dict[str, float]:
    return {
        "structure_magnitude": magnitude,
        "level_direction": 1.0,
        "level_significance": 0.8,
        "level_cleanliness": 0.85,
        "level_rejection": 0.7,
        "structure_direction": 1.0,
        "bos_direction": 1.0,
        "achievable_rr": 2.4,
        "nearest_zone_price": 18_000.0,
    }


def bearish_vol_momentum(magnitude_inputs: tuple[float, float] = (0.95, 0.95)):
    vol, mom = magnitude_inputs
    return {
        "vol_regime_score": vol,
        "momentum_score": mom,
        "direction": -1.0,
        "atr_percentile": 0.55,
        "vol_expansion": 0.4,
        "roc": -0.02,
        "roc_atr": -3.1,
        "efficiency_ratio": 0.9,
    }


@pytest.fixture
def flow_score_config():
    return load_config().flow_score


# ---------------------------------------------------------------------------
# section 7: magnitude and direction are separate
# ---------------------------------------------------------------------------


def test_a_bearish_component_keeps_its_magnitude_and_its_sign(flow_score_config):
    """A hard bearish impulse is a LOT of evidence, pointing down.

    The magnitude must not shrink because the move is down, and the
    direction must not grow because the move is large.
    """
    scorer = VolMomentumScorer(flow_score_config)
    down = scorer.score(vector(bearish_vol_momentum()))

    up_values = bearish_vol_momentum()
    up_values["direction"] = 1.0
    up_values["roc"], up_values["roc_atr"] = 0.02, 3.1
    up = scorer.score(vector(up_values))

    assert down.direction == -1
    assert up.direction == 1
    # Same magnitude inputs, so the same magnitude: flipping the sign of the
    # move changed nothing about how much evidence there is.
    assert down.magnitude == up.magnitude
    assert down.magnitude > 0.9
    assert down.points == pytest.approx(up.points)


def test_conflicting_components_oppose_both_sides(flow_score_config):
    """One strongly bearish component against a strongly bullish rest.

    (a) the aggregate magnitude is high, and (b) whichever side a gate were
    to consider, there is opposing weight against it -- so nothing in these
    scores lets a caller take the majority direction.
    """
    structure = StructureScorer(flow_score_config).score(vector(bullish_structure()))
    volmom = VolMomentumScorer(flow_score_config).score(
        vector(bearish_vol_momentum())
    )
    liquidity = LiquidityScorer(flow_score_config).score(
        vector({"liquidity_score": 0.9, "spread_ticks": 1.0, "relative_volume": 1.1})
    )

    score = FlowScore(
        symbol=SYMBOL,
        ts=TS,
        components={c.component: c for c in (structure, volmom, liquidity)},
        max_points=flow_score_config.total_points,
    )

    # (a) the magnitude aggregate is high: the disagreement did not cancel.
    assert score.score > 0.85 * score.available_points

    # (b) both sides are opposed. A long is opposed by VOL_MOMENTUM, a short
    # by STRUCTURE, so no side reads as the majority.
    assert score.opposing_points(Side.LONG) > 0.0
    assert score.opposing_points(Side.SHORT) > 0.0

    # LIQUIDITY holds 15 points and opposes neither side, because it has no
    # directional opinion at all.
    assert liquidity.direction == 0
    assert not liquidity.opposes(Side.LONG)
    assert not liquidity.opposes(Side.SHORT)


def test_magnitude_and_direction_keys_are_disjoint_for_every_scorer(flow_score_config):
    for scorer in bars_scorers(flow_score_config):
        assert not set(scorer.magnitude_keys) & set(scorer.direction_keys)
        assert set(scorer.magnitude_keys) <= BOUNDED_MAGNITUDE_KEYS


# ---------------------------------------------------------------------------
# weights come from config
# ---------------------------------------------------------------------------


def test_weight_is_read_from_config_and_not_hardcoded(flow_score_config):
    for scorer in bars_scorers(flow_score_config):
        assert scorer.weight == flow_score_config.weights[scorer.component]


def test_a_reweighted_config_moves_the_points(flow_score_config):
    """The weight is a hypothesis; the scorer must follow it, not a literal."""
    weights = dict(flow_score_config.weights)
    moved = 5.0
    weights[Component.STRUCTURE] += moved
    weights[Component.LIQUIDITY] -= moved
    altered = flow_score_config.replace(weights=weights)

    base = StructureScorer(flow_score_config).score(vector(bullish_structure(1.0)))
    bumped = StructureScorer(altered).score(vector(bullish_structure(1.0)))
    assert bumped.points == pytest.approx(base.points + moved)


# ---------------------------------------------------------------------------
# UNAVAILABLE is not zero
# ---------------------------------------------------------------------------


def test_a_missing_key_is_unavailable_not_a_zero_score(flow_score_config):
    values = bullish_structure()
    graded = {"structure_magnitude": DataQuality.MISSING}
    score = StructureScorer(flow_score_config).score(vector(values, graded))

    assert score.enabled is False
    assert score.quality is DataQuality.MISSING
    assert score.magnitude == 0.0
    assert score.direction == 0
    # The weight is KEPT, which is what reports the shortfall.
    assert score.weight == flow_score_config.weights[Component.STRUCTURE]
    assert score.points == 0.0
    assert score.detail["available"] == 0.0
    assert score.detail["quality_reason"] == REASON_KEY_MISSING
    # No feature value is copied into detail: the vector's values in this
    # state are placeholders, and a placeholder in a diagnostic reads as a
    # measurement.
    assert "level_significance" not in score.detail
    assert any("UNAVAILABLE" in r for r in StructureScorer(flow_score_config).explain(
        vector(values, graded)
    ))


def test_a_measured_zero_is_available(flow_score_config):
    score = StructureScorer(flow_score_config).score(vector(bullish_structure(0.0)))
    assert score.enabled is True
    assert score.quality is DataQuality.GOOD
    assert score.magnitude == 0.0
    assert score.points == 0.0
    assert score.detail["available"] == 1.0


def test_unavailable_weight_drops_out_of_available_points(flow_score_config):
    """The arithmetic that makes the shortfall reportable."""
    missing = StructureScorer(flow_score_config).score(
        vector(bullish_structure(), {"structure_magnitude": DataQuality.MISSING})
    )
    measured_zero = LiquidityScorer(flow_score_config).score(
        vector({"liquidity_score": 0.0})
    )
    score = FlowScore(
        symbol=SYMBOL,
        ts=TS,
        components={c.component: c for c in (missing, measured_zero)},
        max_points=flow_score_config.total_points,
    )
    assert score.score == 0.0
    assert score.available_points == pytest.approx(
        flow_score_config.weights[Component.LIQUIDITY]
    )


def test_an_absent_key_is_unavailable(flow_score_config):
    score = VolMomentumScorer(flow_score_config).score(
        vector({"vol_regime_score": 0.5, "momentum_score": 0.5})  # no `direction`
    )
    assert score.enabled is False
    assert score.detail["quality_reason"] == REASON_KEYS_ABSENT
    assert score.detail["absent_count"] == 1.0


def test_a_config_disabled_component_says_so(flow_score_config):
    enabled = dict(flow_score_config.enabled_components)
    enabled[Component.LIQUIDITY] = False
    altered = flow_score_config.replace(enabled_components=enabled)
    scorer = LiquidityScorer(altered)
    score = scorer.score(vector({"liquidity_score": 0.9}))
    assert score.enabled is False
    assert score.detail["quality_reason"] == REASON_COMPONENT_DISABLED
    assert scorer.explain(vector({"liquidity_score": 0.9})) == (
        f"liquidity: {REASON_TEXT[REASON_COMPONENT_DISABLED]}",
    )


# ---------------------------------------------------------------------------
# liquidity carries its DEGRADED flag through
# ---------------------------------------------------------------------------


def test_degraded_liquidity_stays_enabled_and_stays_flagged(flow_score_config):
    """Section 5 says "flagged DEGRADED", not "unavailable"."""
    score = LiquidityScorer(flow_score_config).score(
        vector(
            {
                "liquidity_score": 0.5,
                "spread_ticks": 1.0,
                "spread_percentile": 0.5,
                "depth_imbalance": 0.0,
                "relative_volume": 1.0,
            },
            {
                "liquidity_score": DataQuality.DEGRADED,
                "spread_ticks": DataQuality.DEGRADED,
                "spread_percentile": DataQuality.DEGRADED,
                "depth_imbalance": DataQuality.DEGRADED,
            },
        )
    )
    assert score.enabled is True
    assert score.quality is DataQuality.DEGRADED
    assert score.weight == flow_score_config.weights[Component.LIQUIDITY]
    assert score.detail["spread_measured"] == 0.0
    assert score.detail["profile_measured"] == 1.0

    # and the flag reaches the FlowScore, which is the whole point
    aggregate = FlowScore(
        symbol=SYMBOL,
        ts=TS,
        components={score.component: score},
        max_points=flow_score_config.total_points,
    )
    assert aggregate.quality is DataQuality.DEGRADED


def test_a_degraded_diagnostic_does_not_downgrade_a_measured_component(
    flow_score_config,
):
    score = LiquidityScorer(flow_score_config).score(
        vector(
            {"liquidity_score": 0.8, "depth_imbalance": 0.0},
            {"depth_imbalance": DataQuality.DEGRADED},
        )
    )
    assert score.quality is DataQuality.GOOD
    assert score.detail["degraded_diagnostic_count"] == 1.0


# ---------------------------------------------------------------------------
# contract enforcement
# ---------------------------------------------------------------------------


def test_an_unbounded_magnitude_key_is_rejected_at_construction(flow_score_config):
    class Bad(BarsComponentScorer):
        component = Component.STRUCTURE
        magnitude_keys = ("achievable_rr",)  # unbounded by design

        def _magnitude(self, values):  # pragma: no cover - never constructed
            return values["achievable_rr"]

    with pytest.raises(ScorerError, match="BOUNDED_MAGNITUDE_KEYS"):
        Bad(flow_score_config)


def test_a_signed_continuous_key_is_not_a_direction(flow_score_config):
    class Bad(BarsComponentScorer):
        component = Component.VOL_MOMENTUM
        magnitude_keys = ("momentum_score",)
        direction_keys = ("vol_expansion",)

        def _magnitude(self, values):  # pragma: no cover - never constructed
            return values["momentum_score"]

    with pytest.raises(ScorerError, match="UNIT_DIRECTION_KEYS"):
        Bad(flow_score_config)


def test_an_out_of_range_magnitude_input_raises_rather_than_clamping(flow_score_config):
    values = bullish_structure()
    values["structure_magnitude"] = 1.4
    with pytest.raises(ScorerError, match="outside its declared"):
        StructureScorer(flow_score_config).score(vector(values))


def test_an_out_of_range_direction_raises(flow_score_config):
    values = bullish_structure()
    values["level_direction"] = 2.0
    with pytest.raises(ScorerError, match="outside"):
        StructureScorer(flow_score_config).score(vector(values))


@pytest.mark.parametrize(
    "value, expected", [(-1.0, -1), (0.0, 0), (1.0, 1), (-0.0, 0)]
)
def test_unit_direction_is_sign_extraction(value, expected):
    assert unit_direction(value) == expected


def test_vol_momentum_blend_is_convex():
    assert VOL_MOMENTUM_WEIGHT_VOLATILITY + VOL_MOMENTUM_WEIGHT_MOMENTUM == 1.0


def test_vol_momentum_magnitude_is_the_declared_blend(flow_score_config):
    score = VolMomentumScorer(flow_score_config).score(
        vector({"vol_regime_score": 1.0, "momentum_score": 0.0, "direction": 0.0})
    )
    assert score.magnitude == pytest.approx(VOL_MOMENTUM_WEIGHT_VOLATILITY)
    assert score.detail["volatility_term"] == pytest.approx(
        VOL_MOMENTUM_WEIGHT_VOLATILITY
    )
    assert score.detail["momentum_term"] == 0.0


def test_vol_momentum_is_a_sum_not_a_product(flow_score_config):
    """A dead tape must not zero the component; that is the VolatilityGate's job."""
    score = VolMomentumScorer(flow_score_config).score(
        vector({"vol_regime_score": 0.0, "momentum_score": 1.0, "direction": 1.0})
    )
    assert score.magnitude == pytest.approx(VOL_MOMENTUM_WEIGHT_MOMENTUM)
    assert score.magnitude > 0.0


def test_structure_reports_geometry_agreement_without_folding_it_in(flow_score_config):
    """14.6 fixes the magnitude and the direction; geometry is reported."""
    agree = bullish_structure()
    disagree = dict(agree, structure_direction=-1.0, bos_direction=-1.0)

    a = StructureScorer(flow_score_config).score(vector(agree))
    d = StructureScorer(flow_score_config).score(vector(disagree))

    assert a.detail["geometry_agreement"] == 1.0
    assert d.detail["geometry_agreement"] == -1.0
    # The magnitude and direction are 14.6's and are unchanged by the
    # disagreement: whether it blocks the trade is a gate's decision.
    assert a.magnitude == d.magnitude
    assert a.direction == d.direction == 1


def test_every_scorer_documents_the_keys_it_reads(flow_score_config):
    """The traceability requirement, checked rather than trusted."""
    for scorer in bars_scorers(flow_score_config):
        doc = type(scorer).__doc__ or ""
        assert "Feature keys read" in doc
        for key in scorer.reads:
            assert key in doc, f"{scorer.name} reads {key!r} without documenting it"


def test_scores_are_deterministic(flow_score_config):
    for scorer in bars_scorers(flow_score_config):
        values = {
            **bullish_structure(),
            **bearish_vol_momentum(),
            "liquidity_score": 0.63,
            "spread_ticks": 1.25,
            "relative_volume": 1.17,
        }
        first = scorer.score(vector(values))
        second = scorer.score(vector(values))
        assert first.to_init_dict() == second.to_init_dict()
        assert first.magnitude == second.magnitude
        assert first.points == second.points


# ---------------------------------------------------------------------------
# end to end against the real feature layer
# ---------------------------------------------------------------------------


def _bundle_and_view(*, with_quotes: bool):
    """The five bars-only computers run on synthetic NQ, scored at the last bar."""
    config = load_config()
    spec = config.spec(SYMBOL)
    calendar = TradingCalendar()
    dataset = SyntheticMarketGenerator(SyntheticConfig(), seed=11).generate(
        SYMBOL, spec, date(2024, 1, 2), date(2024, 6, 1), INTERVAL, calendar=calendar
    )
    data = SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: dataset.bars},
        quotes=dataset.quotes if with_quotes else None,
    )
    last_ns = int(dataset.bars.ts_ns[-1])
    view = MarketView(data, now=from_ns(last_ns), now_ns=last_ns)
    bundle = FeatureBundle(
        [
            VolatilityFeatures(config.features),
            MomentumFeatures(config.features),
            LiquidityFeatures(config.features, spec),
            StructureFeatures(config.features, config.levels, spec, calendar),
            LevelFeatures(config.features, config.levels, spec, calendar),
        ]
    )
    return config, bundle.compute(view)


def test_runs_against_the_real_feature_layer_with_quotes():
    config, features = _bundle_and_view(with_quotes=True)
    scores = [s.score(features) for s in bars_scorers(config.flow_score)]

    assert {s.component for s in scores} == {
        Component.STRUCTURE,
        Component.LIQUIDITY,
        Component.VOL_MOMENTUM,
    }
    for score in scores:
        assert isinstance(score, ComponentScore)
        assert 0.0 <= score.magnitude <= 1.0
        assert score.direction in (-1, 0, 1)
        assert score.enabled is True
        assert score.quality.rank >= DataQuality.DEGRADED.rank
        assert score.detail["available"] == 1.0

    aggregate = FlowScore(
        symbol=SYMBOL,
        ts=features.ts,
        components={s.component: s for s in scores},
        max_points=config.flow_score.total_points,
    )
    # 55 of 100 points: the other two components are not here, and this is
    # NOT a Flow Score. The aggregate agent is the one that must refuse.
    assert aggregate.available_points == pytest.approx(55.0)
    assert 0.0 <= aggregate.score <= aggregate.available_points


def test_runs_bars_only_and_liquidity_reports_degraded():
    """No quote feed: liquidity degrades, the other two do not."""
    config, features = _bundle_and_view(with_quotes=False)
    by_component = {
        s.component: s for s in (sc.score(features) for sc in bars_scorers(config.flow_score))
    }

    liquidity = by_component[Component.LIQUIDITY]
    assert liquidity.enabled is True
    assert liquidity.quality is DataQuality.DEGRADED
    assert liquidity.detail["spread_measured"] == 0.0
    assert liquidity.magnitude <= 0.5  # NO_QUOTE_SCORE_CAP, carried through

    assert by_component[Component.VOL_MOMENTUM].quality is DataQuality.GOOD
    # STRUCTURE is bars-only too; it may be DEGRADED for the single-interval
    # reasons levels.py documents (no higher timeframe, no tick feed), which
    # is a different shortfall and must not be mistaken for the quote one.
    assert by_component[Component.STRUCTURE].enabled is True


def test_scoring_the_same_bar_twice_agrees_bit_for_bit():
    config, features = _bundle_and_view(with_quotes=True)
    scorers = bars_scorers(config.flow_score)
    first = [s.score(features).to_init_dict() for s in scorers]
    second = [s.score(features).to_init_dict() for s in scorers]
    assert first == second


def test_structure_is_degraded_on_a_single_interval_dataset():
    """A MEASURED consequence, recorded because it is surprising.

    `features/levels.py` grades its whole vector DEGRADED when `s_htf` has
    no higher interval to confirm against and `s_flow` has no tick feed. The
    dataset here carries one bar interval and no ticks, so STRUCTURE is
    DEGRADED on every bar *even when the quote feed is present* -- and since
    `FlowScore.quality` is the worst over enabled components, the aggregate
    is DEGRADED on every bar too.

    That is not this module's doing and it is not a bug: it is the honest
    reading of a single-interval dataset, and `levels.py` says so in its own
    docstring ("the maximum attainable S is 1 - htf_confluence = 0.85").
    What matters here is that the scorer propagates it instead of rounding
    it up to GOOD. Recorded as a measurement rather than fixed by relaxing
    a grade.
    """
    config, features = _bundle_and_view(with_quotes=True)
    structure = StructureScorer(config.flow_score).score(features)
    assert structure.quality is DataQuality.DEGRADED
    assert structure.enabled is True

    aggregate = FlowScore(
        symbol=SYMBOL,
        ts=features.ts,
        components={
            s.component: s for s in (sc.score(features) for sc in bars_scorers(config.flow_score))
        },
        max_points=config.flow_score.total_points,
    )
    assert aggregate.quality is DataQuality.DEGRADED


def test_explain_does_not_repeat_the_component_grade_as_a_diagnostic(flow_score_config):
    """Every levels key degrades together, so listing all 21 says nothing new."""
    values = bullish_structure()
    graded = {key: DataQuality.DEGRADED for key in values}
    reasons = StructureScorer(flow_score_config).explain(vector(values, graded))
    assert len(reasons) == 1
    assert "magnitude/direction key(s)" in reasons[0]
    assert "diagnostic key(s)" not in reasons[0]


def test_explain_truncates_a_long_key_list(flow_score_config):
    from flow_model.signals.scoring_bars import EXPLAIN_KEY_LIMIT

    scorer = StructureScorer(flow_score_config)
    values = {key: 0.0 for key in scorer.reads}
    values["level_direction"] = 1.0
    graded = {key: DataQuality.MISSING for key in scorer.required_keys}
    # Force a long absent list instead: drop every required key but one.
    sparse = {k: v for k, v in values.items() if k not in scorer.required_keys}
    reasons = scorer.explain(vector(sparse, graded))
    assert len(reasons) == 1
    assert "absent" in reasons[0]

    long_list = tuple(f"k{i}" for i in range(EXPLAIN_KEY_LIMIT + 3))
    from flow_model.signals.scoring_bars import _named

    assert _named(long_list).endswith("and 3 more")
    assert _named(long_list[:EXPLAIN_KEY_LIMIT]) == str(list(long_list[:EXPLAIN_KEY_LIMIT]))


# ---------------------------------------------------------------------------
# the dataset-level availability verdict is consumed, not re-derived
# ---------------------------------------------------------------------------


def _availability(component, *, computable, quality, weight, feeds=()):
    return ComponentAvailability(
        component=component,
        computable=computable,
        quality=quality,
        weight=weight,
        missing_required_feeds=tuple(feeds),
        note="synthetic verdict for the smoke check",
    )


def test_an_availability_refusal_makes_the_component_unavailable(flow_score_config):
    from flow_model.core.enums import Feed

    verdict = _availability(
        Component.LIQUIDITY,
        computable=False,
        quality=DataQuality.MISSING,
        weight=flow_score_config.weights[Component.LIQUIDITY],
        feeds=(Feed.BARS,),
    )
    scorer = LiquidityScorer(flow_score_config)
    # The vector itself looks perfectly fine; the dataset verdict is what refuses.
    good = vector({"liquidity_score": 0.9})
    score = scorer.score(good, verdict)
    assert score.enabled is False
    assert score.quality is DataQuality.MISSING
    assert score.detail["quality_reason"] == REASON_FEED_UNAVAILABLE
    assert score.detail["missing_feed_count"] == 1.0
    assert "UNAVAILABLE" in scorer.explain(good, verdict)[0]
    assert "bars" in scorer.explain(good, verdict)[0]


def test_a_computable_verdict_does_not_override_a_missing_key(flow_score_config):
    """`availability()` reads the series; the vector reads this instant."""
    verdict = _availability(
        Component.LIQUIDITY,
        computable=True,
        quality=DataQuality.GOOD,
        weight=flow_score_config.weights[Component.LIQUIDITY],
    )
    score = LiquidityScorer(flow_score_config).score(
        vector({"liquidity_score": 0.9}, {"liquidity_score": DataQuality.MISSING}),
        verdict,
    )
    assert score.enabled is False
    assert score.detail["quality_reason"] == REASON_KEY_MISSING


def test_a_degraded_dataset_downgrades_a_clean_looking_bar(flow_score_config):
    verdict = _availability(
        Component.VOL_MOMENTUM,
        computable=True,
        quality=DataQuality.DEGRADED,
        weight=flow_score_config.weights[Component.VOL_MOMENTUM],
    )
    features = vector(
        {"vol_regime_score": 0.8, "momentum_score": 0.8, "direction": 1.0}
    )
    scorer = VolMomentumScorer(flow_score_config)
    assert scorer.score(features).quality is DataQuality.GOOD
    downgraded = scorer.score(features, verdict)
    assert downgraded.quality is DataQuality.DEGRADED
    assert downgraded.enabled is True
    # the magnitude is untouched: a grade is not a haircut
    assert downgraded.magnitude == scorer.score(features).magnitude
    assert any("data/quality.py" in r for r in scorer.explain(features, verdict))


def test_another_components_verdict_is_rejected(flow_score_config):
    verdict = _availability(
        Component.ORDER_FLOW,
        computable=False,
        quality=DataQuality.MISSING,
        weight=flow_score_config.weights[Component.ORDER_FLOW],
    )
    with pytest.raises(ScorerError, match="wrong gap"):
        LiquidityScorer(flow_score_config).score(
            vector({"liquidity_score": 0.9}), verdict
        )


# ---------------------------------------------------------------------------
# constructed the same way as the other two scorers
# ---------------------------------------------------------------------------


def test_scorers_accept_the_root_config_or_just_flow_score():
    config = load_config()
    from_root = bars_scorers(config)
    from_section = bars_scorers(config.flow_score)
    features = vector(bullish_structure())
    assert from_root[0].score(features).to_init_dict() == (
        from_section[0].score(features).to_init_dict()
    )
    for a, b in zip(from_root, from_section):
        assert a.component is b.component
        assert a.weight == b.weight


def test_a_constructor_refuses_anything_that_is_not_config():
    class Bars:
        close = 100.0

    with pytest.raises(ScorerError, match="nothing else"):
        StructureScorer(Bars())  # type: ignore[arg-type]


def test_all_five_scorers_construct_from_one_config_object():
    """The interface the aggregator depends on."""
    from flow_model.signals.scoring_flow import feed_dependent_scorers

    config = load_config()
    everything = bars_scorers(config) + feed_dependent_scorers(config)
    assert {s.component for s in everything} == set(Component)
    assert sum(s.weight for s in everything) == pytest.approx(
        config.flow_score.total_points
    )
