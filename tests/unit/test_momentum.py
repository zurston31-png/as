"""Momentum features, checked against independently computed arithmetic.

The strategy here is to avoid testing the implementation against itself.
Three kinds of check, in descending order of how much they prove:

1. **Closed-form fixtures.** Four bar shapes are constructed so that every
   feature has an exact value derivable by hand on paper, with no floating
   point slack: a linear ramp whose true range is constant (so Wilder's ATR
   is exactly the step), a wide-range ramp (so the dead band can be
   straddled from both sides), a perfectly flat tape, and a perfect zigzag
   (whose lag-1 return autocorrelation is exactly -(m-1)/m). Where a value
   is not a round number -- acceleration, which passes through a tanh -- the
   formula is re-derived in the test from the raw closes rather than copied
   from the module.

2. **Independent re-implementations.** Wilder's ATR and the lag-1
   autocorrelation are each written a second time here as plain Python
   loops, deliberately in a different code shape from the module's NumPy
   form, and the features that divide by them are checked against those.

3. **Ground truth.** `SyntheticDataset.regime_labels` are drawn *before* the
   prices are drawn from them, so they are causes rather than annotations.
   `efficiency_ratio` is the quantity ARCHITECTURE.md section 6 uses to
   separate TRENDING from CHOP, and that separation is measured here against
   those labels on regime-pure windows rather than asserted.

Plus the lookahead audit with `factory=`, and a control showing the audit
actually bites this computer -- a passing audit on a computer the harness
could not fail would prove nothing.

Fixtures are local by design: `tests/conftest.py` is shared and this module
must be readable on its own.
"""

from __future__ import annotations

import collections
import inspect
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.config.schema import FeatureConfig
from flow_model.core.contracts import FeatureVector
from flow_model.core.enums import DataQuality, Feed, InstrumentType, Regime
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.market_view import MarketView
from flow_model.data.series import BarSeries, from_ns, to_ns
from flow_model.data.store import SymbolData
from flow_model.data.synthetic import SyntheticConfig, SyntheticDataset, SyntheticMarketGenerator
from flow_model.features.base import FeatureBundle
from flow_model.features.momentum import (
    ACCELERATION_REFERENCE_BAR_RANGES,
    DIRECTION_DEAD_BAND_ATR,
    MOMENTUM_SCORE_ROC_ATR_REFERENCE,
    MOMENTUM_SCORE_WEIGHT_ACCELERATION,
    MOMENTUM_SCORE_WEIGHT_EFFICIENCY,
    MOMENTUM_SCORE_WEIGHT_ROC_ATR,
    MomentumFeatures,
)
from flow_model.validation.lookahead import assert_no_lookahead, audit_computer

SYMBOL = "NQ"
INTERVAL = 300
T0 = datetime(2021, 1, 4, 14, 35, tzinfo=timezone.utc)  # 09:35 ET, on the 5-min grid

EXPECTED_KEYS = (
    "roc",
    "roc_atr",
    "efficiency_ratio",
    "direction",
    "acceleration",
    "momentum_persistence",
    "up_bar_fraction",
    "close_position",
    "momentum_score",
)


# --- local fixtures --------------------------------------------------------


@pytest.fixture
def config() -> FeatureConfig:
    return load_config().features


@pytest.fixture
def momentum(config) -> MomentumFeatures:
    return MomentumFeatures(config)


def bars_from(opens, highs, lows, closes) -> BarSeries:
    """A BarSeries on the interval grid from four explicit price arrays."""
    n = len(closes)
    ts = np.array(
        [to_ns(T0 + timedelta(seconds=INTERVAL * (i + 1))) for i in range(n)], dtype=np.int64
    )
    return BarSeries(
        symbol=SYMBOL,
        ts_ns=ts,
        interval_seconds=INTERVAL,
        columns={
            "open": np.asarray(opens, dtype=np.float64),
            "high": np.asarray(highs, dtype=np.float64),
            "low": np.asarray(lows, dtype=np.float64),
            "close": np.asarray(closes, dtype=np.float64),
            "volume": np.full(n, 1000.0),
        },
    )


def bars_from_closes(closes) -> BarSeries:
    """Minimal bars around a close path: open at the prior close, no shadows."""
    close = np.asarray(closes, dtype=np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    return bars_from(open_, np.maximum(open_, close), np.minimum(open_, close), close)


def ramp_bars(n: int = 60, base: float = 100.0, step: float = 1.0) -> BarSeries:
    """Linear ramp with `high == close` and `low == open`.

    Every true range is exactly `step`: the bar's own range is `step`, the
    gap from the prior close to this high is `step`, and the gap to this low
    is zero. Wilder's ATR of a constant series is that constant, so the ATR
    is exactly `step` with no floating-point slack -- which is what makes
    `roc_atr`, `direction` and `acceleration` exactly predictable here. The
    last close is the window high, so `close_position` is exactly 1.0.
    """
    close = np.array([base + (i + 1) * step for i in range(n)], dtype=np.float64)
    open_ = np.array([base + i * step for i in range(n)], dtype=np.float64)
    return bars_from(open_, np.maximum(open_, close), np.minimum(open_, close), close)


def wide_ramp_bars(
    n: int = 60, base: float = 1000.0, step: float = 1.0, width: float = 10.0
) -> BarSeries:
    """Ramp of `step` per bar inside bars of constant range `width`.

    Each true range is exactly `width` provided `width / 2 >= |step|`, so the
    ATR is exactly `width` while the net move over `momentum_period` bars is
    `momentum_period * step`. That decouples the move from the volatility and
    is what lets the direction dead band be straddled from both sides.
    """
    assert width / 2.0 >= abs(step), "a bar's half-range must cover one step"
    close = np.array([base + i * step for i in range(n)], dtype=np.float64)
    open_ = close - step
    return bars_from(open_, close + width / 2.0, close - width / 2.0, close)


def inverted_ramp_bars(n: int = 60, base: float = 100.0, step: float = 1.0) -> BarSeries:
    """Falling ramp with `low == close`, so the last close is the window low."""
    close = np.array([base - (i + 1) * step for i in range(n)], dtype=np.float64)
    open_ = np.array([base - i * step for i in range(n)], dtype=np.float64)
    return bars_from(np.maximum(open_, close), np.maximum(open_, close), close, close)


def flat_bars(n: int = 60, price: float = 500.0) -> BarSeries:
    """Open == high == low == close on every bar. Every denominator is zero."""
    level = np.full(n, price, dtype=np.float64)
    return bars_from(level, level, level, level)


def zigzag_bars(n: int = 60, base: float = 1000.0, amplitude: float = 10.0) -> BarSeries:
    """Close alternates between `base` and `base + amplitude`.

    True range is exactly `amplitude` on every bar after the first, so the
    ATR is exactly `amplitude`. With an even-length efficiency window the net
    displacement is exactly zero, so `efficiency_ratio` is exactly 0.0.
    """
    close = np.array(
        [base + (amplitude if i % 2 else 0.0) for i in range(n)], dtype=np.float64
    )
    open_ = np.concatenate(([base], close[:-1]))
    return bars_from(open_, np.maximum(open_, close), np.minimum(open_, close), close)


def symbol_data(bars: BarSeries) -> SymbolData:
    return SymbolData(symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: bars})


def view_at(bars: BarSeries, index: int = -1) -> MarketView:
    """A view at the close of bar `index` (negative indexes from the end)."""
    return view_of(symbol_data(bars), bars.ts_ns, index)


def view_of(data: SymbolData, stamps: np.ndarray, index: int) -> MarketView:
    """A view at `stamps[index]`, built without touching the whole series.

    `BarSeries.timestamps()` materializes every datetime in the series, so
    calling it inside a per-bar loop turns a chronological pass into an
    O(n^2) one -- 45 seconds for the regime sweeps below, measured. The
    integer stamp is passed as `now_ns` as well, which is also what
    `DataStore` does: `from_ns` truncates below the microsecond, and a
    cutoff built from the truncated datetime alone would sit just before
    the bar's own close.
    """
    stamp = int(stamps[len(stamps) + index if index < 0 else index])
    return MarketView(data, now=from_ns(stamp), now_ns=stamp)


def values(momentum: MomentumFeatures, bars: BarSeries, index: int = -1) -> dict[str, float]:
    vector = momentum.compute(view_at(bars, index))
    assert vector.warmup_complete, vector.notes
    return dict(vector.values)


# --- independent re-implementations ---------------------------------------


def independent_wilder_atr(bars: BarSeries, period: int, window: int) -> float:
    """Wilder ATR over the last `window` bars, as an explicit Python loop.

    Written in a different shape from the module's vectorized form on
    purpose: a copy of the same NumPy expression would agree with a bug.
    """
    high = bars.col("high")[-window:].tolist()
    low = bars.col("low")[-window:].tolist()
    close = bars.col("close")[-window:].tolist()
    true_ranges = []
    for i in range(1, len(close)):
        true_ranges.append(
            max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        )
    average = sum(true_ranges[:period]) / period
    for value in true_ranges[period:]:
        average = (average * (period - 1) + value) / period
    return average


def independent_lag1_autocorrelation(series: list[float]) -> float:
    """sum d_i d_{i+1} / sum d_i^2 over the mean-removed series, by loop."""
    mean = sum(series) / len(series)
    deviations = [x - mean for x in series]
    denominator = sum(d * d for d in deviations)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(
        deviations[i] * deviations[i + 1] for i in range(len(deviations) - 1)
    )
    return numerator / denominator


def independent_efficiency_ratio(closes: list[float]) -> float:
    travel = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    if travel == 0.0:
        return 0.0
    return abs(closes[-1] - closes[0]) / travel


# --- synthetic ground truth -----------------------------------------------


def nq_spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol=SYMBOL,
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
    )


def generate(seed: int, end: date) -> SyntheticDataset:
    """Bars-only generated data: this module needs no other feed."""
    return SyntheticMarketGenerator(
        SyntheticConfig(include_quotes=False, include_ticks=False, include_options=False),
        seed=seed,
    ).generate(SYMBOL, nq_spec(), date(2021, 1, 4), end, INTERVAL)


@pytest.fixture(scope="module")
def labelled() -> SyntheticDataset:
    """Six months of five-minute bars with ground-truth regime labels."""
    return generate(seed=17, end=date(2021, 7, 1))


@pytest.fixture(scope="module")
def regime_samples(labelled) -> dict[Regime, dict[str, list[float]]]:
    """Features bucketed by ground-truth regime, over regime-pure windows only.

    One pass over the dataset, shared by the two separation tests. A window
    that spans a regime transition is a mixture of both and would blur the
    contrast being measured, so only windows whose every label agrees are
    kept -- the purity span is the efficiency window, since that is the one
    the measured feature reads.
    """
    config = load_config().features
    computer = MomentumFeatures(config)
    data = symbol_data(labelled.bars)
    labels = labelled.label_series()
    stamps = labelled.bars.ts_ns
    span = config.efficiency_period

    samples: dict[Regime, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for index in range(computer.warmup_bars - 1, len(stamps)):
        window = labels[index - span : index + 1]
        label = window[0]
        if not all(other == label for other in window):
            continue
        vector = computer.compute(view_of(data, stamps, index))
        for key in ("efficiency_ratio", "momentum_score", "momentum_persistence"):
            samples[label][key].append(vector.values[key])
    return samples


@pytest.fixture(scope="module")
def audit_data() -> SymbolData:
    """A dataset comfortably longer than twice the computer's warmup."""
    dataset = generate(seed=23, end=date(2021, 3, 1))
    assert len(dataset.bars) > 2 * MomentumFeatures(load_config().features).warmup_bars
    return symbol_data(dataset.bars)


# --- the declared contract ------------------------------------------------


def test_output_keys_are_exactly_the_declared_keys(momentum, config):
    vector = momentum.compute(view_at(ramp_bars(n=80)))
    assert momentum.keys == EXPECTED_KEYS
    assert tuple(vector.values) == EXPECTED_KEYS
    assert set(vector.quality_by_key) == set(EXPECTED_KEYS)


def test_required_feed_is_bars_only(momentum):
    assert momentum.required_feeds == frozenset({Feed.BARS})
    assert momentum.optional_feeds == frozenset()


def test_constructor_takes_configuration_only():
    """The one leak the audit harness cannot see through is a full-sample
    constant captured at construction. It is banned at the signature."""
    parameters = list(inspect.signature(MomentumFeatures.__init__).parameters.values())
    assert [p.name for p in parameters] == ["self", "config"]
    annotation = parameters[1].annotation
    assert annotation in (FeatureConfig, "FeatureConfig")
    forbidden = ("symboldata", "barseries", "datastore", "marketview", "ndarray", "series")
    assert not any(token in str(annotation).lower() for token in forbidden)


def test_warmup_covers_every_window_including_the_extra_acceleration_bar(momentum, config):
    """`acceleration` needs the ROC one bar ago, so the close window is
    `momentum_period + 2`, not `+ 1`."""
    assert momentum.roc_window_bars == config.momentum_period + 2
    assert momentum.efficiency_window_bars == config.efficiency_period + 1
    assert momentum.atr_window_bars == 2 * config.atr_period + 1
    assert momentum.warmup_bars == max(
        config.momentum_period + 2,
        config.efficiency_period + 1,
        2 * config.atr_period + 1,
    )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"efficiency_period": 100}, 101),
        ({"atr_period": 60}, 121),
        ({"momentum_period": 200}, 202),
    ],
)
def test_warmup_tracks_whichever_configured_window_is_longest(overrides, expected):
    assert MomentumFeatures(FeatureConfig(**overrides)).warmup_bars == expected


def test_one_bar_short_of_warmup_is_not_ready(momentum):
    bars = ramp_bars(n=momentum.warmup_bars - 1)
    vector = momentum.compute(view_at(bars))
    assert vector.warmup_complete is False
    assert vector.quality is DataQuality.MISSING
    assert set(vector.values) == set(EXPECTED_KEYS)
    assert "warmup incomplete" in vector.notes[0]


def test_exactly_warmup_bars_is_ready(momentum):
    vector = momentum.compute(view_at(ramp_bars(n=momentum.warmup_bars)))
    assert vector.warmup_complete is True
    assert vector.quality is DataQuality.GOOD
    assert vector.notes == ()


def test_the_bundle_gates_on_this_computers_warmup(momentum):
    """The bundle enforces warmup too; the computer must agree with it rather
    than rely on it, because the audit calls `compute` directly."""
    bundle = FeatureBundle([momentum])
    assert bundle.warmup_bars == momentum.warmup_bars
    assert bundle.keys == tuple(sorted(EXPECTED_KEYS))
    short = bundle.compute(view_at(ramp_bars(n=momentum.warmup_bars - 1)))
    assert short.warmup_complete is False
    full = bundle.compute(view_at(ramp_bars(n=momentum.warmup_bars)))
    assert full.warmup_complete is True


# --- the local ATR --------------------------------------------------------


def test_local_atr_matches_an_independent_wilder_loop(momentum, config):
    bars = bars_from_closes(
        [1000.0 + 7.0 * math.sin(i / 4.0) + 0.3 * i for i in range(90)]
    )
    assert momentum.wilder_atr(view_at(bars)) == pytest.approx(
        independent_wilder_atr(bars, config.atr_period, momentum.atr_window_bars),
        rel=1e-12,
    )


def test_local_atr_of_a_constant_true_range_is_that_constant(momentum):
    """Wilder's recursion is a weighted mean, so a constant input is a fixed
    point. This is what makes the ramp fixture exact."""
    assert momentum.wilder_atr(view_at(ramp_bars(n=80, step=1.0))) == 1.0
    assert momentum.wilder_atr(view_at(ramp_bars(n=80, step=0.25))) == pytest.approx(0.25)
    assert momentum.wilder_atr(view_at(zigzag_bars(amplitude=10.0))) == pytest.approx(10.0)


def test_local_atr_of_a_flat_window_is_zero_not_an_error(momentum):
    assert momentum.wilder_atr(view_at(flat_bars())) == 0.0


# --- roc ------------------------------------------------------------------


def test_roc_is_the_plain_n_bar_fractional_return(momentum, config):
    bars = ramp_bars(n=60, base=100.0, step=1.0)
    closes = bars.col("close").tolist()
    n = config.momentum_period
    expected = closes[-1] / closes[-1 - n] - 1.0
    assert values(momentum, bars)["roc"] == pytest.approx(expected, rel=1e-12)
    # and the closed form: ten unit steps off a base of 150.
    assert expected == pytest.approx(10.0 / 150.0, rel=1e-12)


def test_roc_is_negative_on_a_falling_tape(momentum):
    assert values(momentum, inverted_ramp_bars(n=60))["roc"] < 0.0


def test_roc_is_zero_on_a_flat_tape(momentum):
    assert values(momentum, flat_bars())["roc"] == 0.0


# --- roc_atr --------------------------------------------------------------


def test_roc_atr_is_the_move_measured_in_average_true_ranges(momentum, config):
    """Ten unit steps against an ATR of exactly 1.0 is exactly 10 ATR."""
    assert values(momentum, ramp_bars(n=60, step=1.0))["roc_atr"] == pytest.approx(
        float(config.momentum_period), rel=1e-12
    )


def test_roc_atr_is_scale_free_across_instruments(momentum):
    """Tripling every price triples the move and the ATR, so the ratio must
    not move. This is the property that makes the feature comparable across
    NQ and QQQ, and it is the reason the component score reads `roc_atr`
    rather than `roc`."""
    path = [1000.0 + 0.02 * (i + 1) ** 2 for i in range(60)]
    plain = values(momentum, bars_from_closes(path))
    tripled = values(momentum, bars_from_closes([3.0 * p for p in path]))
    assert tripled["roc_atr"] == pytest.approx(plain["roc_atr"], rel=1e-12)
    assert tripled["momentum_score"] == pytest.approx(plain["momentum_score"], rel=1e-12)


def test_roc_atr_is_zero_rather_than_infinite_when_the_atr_is_zero(momentum):
    assert values(momentum, flat_bars())["roc_atr"] == 0.0


def test_roc_atr_matches_the_move_over_an_independently_computed_atr(momentum, config):
    bars = bars_from_closes([900.0 + 11.0 * math.sin(i / 7.0) + 0.5 * i for i in range(90)])
    closes = bars.col("close").tolist()
    move = closes[-1] - closes[-1 - config.momentum_period]
    atr = independent_wilder_atr(bars, config.atr_period, momentum.atr_window_bars)
    assert values(momentum, bars)["roc_atr"] == pytest.approx(move / atr, rel=1e-12)


# --- efficiency_ratio -----------------------------------------------------


def test_efficiency_ratio_of_a_monotone_ramp_is_exactly_one(momentum):
    """Net displacement equals total travel when no step reverses."""
    assert values(momentum, ramp_bars(n=60))["efficiency_ratio"] == 1.0
    assert values(momentum, inverted_ramp_bars(n=60))["efficiency_ratio"] == 1.0


def test_efficiency_ratio_of_a_perfect_zigzag_is_zero(momentum):
    assert values(momentum, zigzag_bars())["efficiency_ratio"] == 0.0


def test_efficiency_ratio_of_a_flat_tape_is_zero_not_one(momentum):
    """The denominator is zero here. 1.0 would make the regime detector read
    a dead market as a perfect trend, so the fallback is 0.0."""
    assert values(momentum, flat_bars())["efficiency_ratio"] == 0.0


def test_efficiency_ratio_matches_an_independent_computation(momentum, config):
    bars = bars_from_closes(
        [1200.0 + 9.0 * math.sin(i / 3.0) + 0.4 * i for i in range(90)]
    )
    window = bars.col("close").tolist()[-(config.efficiency_period + 1) :]
    assert values(momentum, bars)["efficiency_ratio"] == pytest.approx(
        independent_efficiency_ratio(window), rel=1e-12
    )


def test_efficiency_ratio_stays_in_the_unit_interval_on_generated_data(momentum, labelled):
    data = symbol_data(labelled.bars)
    stamps = labelled.bars.ts_ns
    for index in range(momentum.warmup_bars - 1, len(stamps), 211):
        vector = momentum.compute(view_of(data, stamps, index))
        assert 0.0 <= vector.values["efficiency_ratio"] <= 1.0


def test_efficiency_ratio_separates_trending_from_chop_against_ground_truth(regime_samples):
    """The claim ARCHITECTURE.md section 6 rests on, measured.

    `regime_labels` were drawn before the prices were drawn from them, so
    they are causes rather than annotations. Only regime-pure windows count:
    a window spanning a transition is a mixture and would blur exactly the
    contrast being measured.
    """
    trending = (
        regime_samples[Regime.TRENDING_UP]["efficiency_ratio"]
        + regime_samples[Regime.TRENDING_DOWN]["efficiency_ratio"]
    )
    chop = regime_samples[Regime.CHOP]["efficiency_ratio"]
    assert len(trending) >= 500 and len(chop) >= 500, "not enough regime-pure windows"

    trending_mean = float(np.mean(trending))
    chop_mean = float(np.mean(chop))
    assert trending_mean > 1.5 * chop_mean, (
        f"efficiency_ratio does not separate the regimes: TRENDING {trending_mean:.4f} "
        f"vs CHOP {chop_mean:.4f} (ratio {trending_mean / chop_mean:.3f})"
    )


def test_momentum_score_is_higher_in_trending_than_in_chop(regime_samples):
    """The component magnitude inherits the separation, which is the point of
    feeding `efficiency_ratio` into it."""
    trending = float(
        np.mean(
            regime_samples[Regime.TRENDING_UP]["momentum_score"]
            + regime_samples[Regime.TRENDING_DOWN]["momentum_score"]
        )
    )
    chop = float(np.mean(regime_samples[Regime.CHOP]["momentum_score"]))
    assert trending > chop * 1.2, f"TRENDING {trending:.4f} vs CHOP {chop:.4f}"


# --- direction ------------------------------------------------------------


def test_direction_is_plus_one_on_a_rising_tape(momentum):
    assert values(momentum, ramp_bars(n=60))["direction"] == 1.0


def test_direction_is_minus_one_on_a_falling_tape(momentum):
    assert values(momentum, wide_ramp_bars(n=60, step=-1.0))["direction"] == -1.0


def test_direction_is_zero_on_a_flat_tape(momentum):
    assert values(momentum, flat_bars())["direction"] == 0.0


def test_direction_is_zero_when_bars_travelled_but_the_net_move_did_not(momentum):
    """A zigzag of amplitude 10 has an ATR of 10 and a net move of zero: it is
    emphatically not a flat tape, and it is emphatically not directional."""
    row = values(momentum, zigzag_bars(amplitude=10.0))
    assert row["direction"] == 0.0
    assert row["roc_atr"] == 0.0


def test_direction_is_zero_inside_the_dead_band_and_signed_just_outside(momentum, config):
    """The band is `DIRECTION_DEAD_BAND_ATR` average true ranges wide. The
    wide ramp fixes the ATR at `width` and the net move at
    `momentum_period * step`, so the band can be straddled by moving `step`
    alone, with everything else held constant."""
    width = 10.0
    n = config.momentum_period
    band = DIRECTION_DEAD_BAND_ATR * width

    inside = 0.8 * band / n
    outside = 1.2 * band / n
    assert values(momentum, wide_ramp_bars(step=inside, width=width))["direction"] == 0.0
    assert values(momentum, wide_ramp_bars(step=outside, width=width))["direction"] == 1.0
    assert values(momentum, wide_ramp_bars(step=-inside, width=width))["direction"] == 0.0
    assert values(momentum, wide_ramp_bars(step=-outside, width=width))["direction"] == -1.0


def test_the_dead_band_is_measured_in_atr_units_not_price_units(momentum, config):
    """The same net move is directional against a narrow bar range and not
    directional against a wide one. A dead band in price units could not
    express that, and would mean something different on every instrument."""
    step = 0.3
    n = config.momentum_period
    narrow = step * n / DIRECTION_DEAD_BAND_ATR * 0.5  # move is 2x the band
    wide = step * n / DIRECTION_DEAD_BAND_ATR * 2.0  # move is half the band
    assert values(momentum, wide_ramp_bars(step=step, width=narrow))["direction"] == 1.0
    assert values(momentum, wide_ramp_bars(step=step, width=wide))["direction"] == 0.0


def test_direction_is_always_one_of_three_values(momentum, labelled):
    data = symbol_data(labelled.bars)
    stamps = labelled.bars.ts_ns
    seen = set()
    for index in range(momentum.warmup_bars - 1, len(stamps), 97):
        vector = momentum.compute(view_of(data, stamps, index))
        seen.add(vector.values["direction"])
    assert seen and seen <= {-1.0, 0.0, 1.0}


# --- acceleration ---------------------------------------------------------


def test_acceleration_is_positive_when_the_move_is_speeding_up(momentum):
    """A convex close path: each bar's n-bar return exceeds the last one's."""
    convex = [1000.0 + 0.02 * (i + 1) ** 2 for i in range(60)]
    assert values(momentum, bars_from_closes(convex))["acceleration"] > 0.0


def test_acceleration_is_negative_when_the_move_is_slowing_down(momentum):
    concave = [1000.0 + 20.0 * math.sqrt(i + 1) for i in range(60)]
    assert values(momentum, bars_from_closes(concave))["acceleration"] < 0.0


def test_an_arithmetic_ramp_decelerates_in_percentage_terms(momentum):
    """Worth stating because it looks wrong at a glance: adding a constant
    price step each bar is a *shrinking* return as the price rises, so a
    straight line up has negative ROC acceleration."""
    assert values(momentum, ramp_bars(n=60))["acceleration"] < 0.0


def test_acceleration_matches_the_formula_rederived_from_the_closes(momentum, config):
    """ROC change per bar, converted into bar-ranges, through tanh."""
    bars = ramp_bars(n=60, base=100.0, step=1.0)
    closes = bars.col("close").tolist()
    n = config.momentum_period
    atr = 1.0  # exact for this fixture, asserted elsewhere

    roc = closes[-1] / closes[-1 - n] - 1.0
    roc_previous = closes[-2] / closes[-2 - n] - 1.0
    normalized = (roc - roc_previous) / (atr / closes[-1])
    expected = math.tanh(normalized / ACCELERATION_REFERENCE_BAR_RANGES)

    assert values(momentum, bars)["acceleration"] == pytest.approx(expected, rel=1e-12)


def test_acceleration_is_zero_on_a_flat_tape(momentum):
    assert values(momentum, flat_bars())["acceleration"] == 0.0


def test_acceleration_is_bounded_in_minus_one_to_one(momentum):
    """A step change big enough to saturate the tanh must still be bounded:
    an unbounded acceleration would dominate the weighted sum it feeds."""
    burst = [1000.0] * 45 + [1000.0 + 3.0 * i**3 for i in range(1, 16)]
    row = values(momentum, bars_from_closes(burst))
    assert -1.0 <= row["acceleration"] <= 1.0
    assert row["acceleration"] > 0.99, "this path should saturate the squash"


def test_a_constant_growth_rate_is_not_acceleration(momentum):
    """Worth pinning down, because it is the distinction the feature exists
    to make: a geometric path has a constant n-bar ROC, so however steep it
    is, it is not accelerating. The doubling series gives returns that are
    exactly 1.0 in binary floating point, so the answer is exactly zero."""
    doubling = [1024.0 * 2.0**i for i in range(60)]
    assert values(momentum, bars_from_closes(doubling))["acceleration"] == 0.0


def test_acceleration_is_bounded_on_generated_data(momentum, labelled):
    data = symbol_data(labelled.bars)
    stamps = labelled.bars.ts_ns
    for index in range(momentum.warmup_bars - 1, len(stamps), 173):
        vector = momentum.compute(view_of(data, stamps, index))
        assert -1.0 <= vector.values["acceleration"] <= 1.0


# --- momentum_persistence -------------------------------------------------


def test_persistence_of_an_alternating_return_series_is_exactly_minus_m_over_m(
    momentum, config
):
    """The zigzag's returns alternate between two values, so the mean-removed
    deviations alternate in sign with constant magnitude and the estimator
    collapses to -(m-1)/m exactly, with m the number of returns."""
    m = config.efficiency_period
    assert values(momentum, zigzag_bars())["momentum_persistence"] == pytest.approx(
        -(m - 1) / m, rel=1e-12
    )


def test_persistence_is_positive_on_a_smoothly_trending_tape(momentum):
    assert values(momentum, ramp_bars(n=60))["momentum_persistence"] > 0.75


def test_persistence_matches_an_independent_autocorrelation(momentum, config):
    bars = bars_from_closes(
        [800.0 + 6.0 * math.sin(i / 2.5) + 0.2 * i for i in range(90)]
    )
    closes = bars.col("close").tolist()[-(config.efficiency_period + 1) :]
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    assert len(returns) == config.efficiency_period
    assert values(momentum, bars)["momentum_persistence"] == pytest.approx(
        independent_lag1_autocorrelation(returns), rel=1e-10
    )


def test_persistence_abstains_rather_than_claiming_certainty_without_variance(momentum):
    """A series with no return variance has no innovations to correlate, so
    there is nothing to measure. 0.0 is an abstention; 1.0 would be a
    fabrication, reporting a dead tape as perfectly persistent.

    Two such series: a flat tape, whose returns are all exactly zero, and a
    doubling series, whose returns are all exactly 1.0 -- chosen because
    powers of two are exact in binary floating point. A nominally geometric
    path at some other ratio is NOT one of these cases: its returns carry
    rounding noise in the last bits, and this feature then reports the
    autocorrelation of that noise, which is why the module documents the
    guard as exact-zero variance rather than near-zero.
    """
    assert values(momentum, flat_bars())["momentum_persistence"] == 0.0
    doubling = [1024.0 * 2.0**i for i in range(60)]
    assert values(momentum, bars_from_closes(doubling))["momentum_persistence"] == 0.0


def test_persistence_recovers_the_sign_of_the_generators_own_ar1_coefficient(
    regime_samples,
):
    """The generator draws TRENDING with `mean_persistence = +0.35` and CHOP
    with `-0.35`, then draws the prices from those. This feature is supposed
    to measure exactly that, so the recovered sign is checked against the
    parameter that produced the tape rather than against a plausible-looking
    number."""
    trending = float(
        np.mean(
            regime_samples[Regime.TRENDING_UP]["momentum_persistence"]
            + regime_samples[Regime.TRENDING_DOWN]["momentum_persistence"]
        )
    )
    chop = float(np.mean(regime_samples[Regime.CHOP]["momentum_persistence"]))
    assert trending > 0.1, f"TRENDING persistence {trending:+.4f} should be positive"
    assert chop < -0.1, f"CHOP persistence {chop:+.4f} should be negative"


def test_persistence_is_bounded_on_generated_data(momentum, labelled):
    data = symbol_data(labelled.bars)
    stamps = labelled.bars.ts_ns
    for index in range(momentum.warmup_bars - 1, len(stamps), 173):
        vector = momentum.compute(view_of(data, stamps, index))
        assert -1.0 <= vector.values["momentum_persistence"] <= 1.0


def test_persistence_window_is_the_efficiency_period_of_returns(momentum, config):
    """Stated in the docstring and asserted here so the two cannot drift: the
    two trend diagnostics describe the same stretch of tape."""
    assert momentum.efficiency_window_bars == config.efficiency_period + 1


# --- up_bar_fraction ------------------------------------------------------


def test_up_bar_fraction_is_one_when_every_bar_closed_above_its_open(momentum):
    assert values(momentum, ramp_bars(n=60))["up_bar_fraction"] == 1.0


def test_up_bar_fraction_is_zero_when_every_bar_closed_below_its_open(momentum):
    assert values(momentum, wide_ramp_bars(n=60, step=-1.0))["up_bar_fraction"] == 0.0


def test_up_bar_fraction_is_one_half_on_an_alternating_tape(momentum, config):
    assert config.momentum_period % 2 == 0, "this fixture needs an even window"
    assert values(momentum, zigzag_bars())["up_bar_fraction"] == 0.5


def test_up_bar_fraction_counts_a_doji_as_neither_up_nor_down(momentum):
    assert values(momentum, flat_bars())["up_bar_fraction"] == 0.0


def test_up_bar_fraction_matches_a_hand_count(momentum, config):
    n = config.momentum_period
    closes = [1000.0 + 3.0 * math.sin(i / 1.7) for i in range(60)]
    bars = bars_from_closes(closes)
    opens = bars.col("open")[-n:]
    expected = sum(1 for o, c in zip(opens, bars.col("close")[-n:]) if c > o) / n
    assert values(momentum, bars)["up_bar_fraction"] == pytest.approx(expected)
    assert 0.0 < expected < 1.0, "the fixture should be a mixture, not a corner case"


# --- close_position -------------------------------------------------------


def test_close_position_is_one_when_closing_at_the_window_high(momentum):
    assert values(momentum, ramp_bars(n=60))["close_position"] == 1.0


def test_close_position_is_zero_when_closing_at_the_window_low(momentum):
    assert values(momentum, inverted_ramp_bars(n=60))["close_position"] == 0.0


def test_close_position_is_the_midpoint_of_a_degenerate_range(momentum):
    """A flat window puts the close at the high and the low at once; 0.5 is
    the only answer that does not assert a position it cannot know."""
    assert values(momentum, flat_bars())["close_position"] == 0.5


def test_close_position_matches_a_closed_form_mid_range_value(momentum, config):
    """On the wide ramp the last close sits `width / 2` above the window low
    and `width / 2` below the window high, with `(n - 1) * step` of drift
    between them: (9 * 1 + 5) / (9 * 1 + 10) = 14 / 19."""
    n = config.momentum_period
    step, width = 1.0, 10.0
    expected = ((n - 1) * step + width / 2.0) / ((n - 1) * step + width)
    assert values(momentum, wide_ramp_bars(step=step, width=width))[
        "close_position"
    ] == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(14.0 / 19.0, rel=1e-12)


def test_close_position_matches_an_independent_window_computation(momentum, config):
    n = config.momentum_period
    bars = bars_from_closes([700.0 + 5.0 * math.sin(i / 2.0) for i in range(60)])
    high = float(max(bars.col("high")[-n:]))
    low = float(min(bars.col("low")[-n:]))
    close = float(bars.col("close")[-1])
    assert values(momentum, bars)["close_position"] == pytest.approx(
        (close - low) / (high - low), rel=1e-12
    )


def test_close_position_is_bounded_on_generated_data(momentum, labelled):
    data = symbol_data(labelled.bars)
    stamps = labelled.bars.ts_ns
    for index in range(momentum.warmup_bars - 1, len(stamps), 173):
        vector = momentum.compute(view_of(data, stamps, index))
        assert 0.0 <= vector.values["close_position"] <= 1.0


# --- momentum_score -------------------------------------------------------


def test_score_weights_are_a_convex_combination(momentum):
    """This is what bounds the score, so it is asserted rather than trusted."""
    total = (
        MOMENTUM_SCORE_WEIGHT_ROC_ATR
        + MOMENTUM_SCORE_WEIGHT_EFFICIENCY
        + MOMENTUM_SCORE_WEIGHT_ACCELERATION
    )
    assert total == pytest.approx(1.0, abs=1e-12)
    assert sum(momentum.score_weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(weight >= 0.0 for weight in momentum.score_weights.values())


def test_score_is_the_declared_combination_of_the_emitted_features(momentum):
    """Re-derived from the vector's own other columns, so the published
    formula and the implementation cannot drift apart."""
    bars = bars_from_closes([1000.0 + 0.02 * (i + 1) ** 2 for i in range(60)])
    row = values(momentum, bars)

    displacement = math.tanh(
        abs(row["roc_atr"]) / MOMENTUM_SCORE_ROC_ATR_REFERENCE
    )
    support = max(0.0, row["acceleration"] * row["direction"])
    expected = (
        MOMENTUM_SCORE_WEIGHT_ROC_ATR * displacement
        + MOMENTUM_SCORE_WEIGHT_EFFICIENCY * row["efficiency_ratio"]
        + MOMENTUM_SCORE_WEIGHT_ACCELERATION * support
    )
    assert row["momentum_score"] == pytest.approx(expected, rel=1e-12)


def test_score_is_a_magnitude_and_ignores_the_sign_of_the_move(momentum, config):
    """The component carries its direction separately, so a symmetric down
    move must score the same as the up move."""
    step, width = 1.0, 10.0
    up = values(momentum, wide_ramp_bars(step=step, width=width))
    down = values(momentum, wide_ramp_bars(step=-step, width=width))
    assert up["direction"] == 1.0 and down["direction"] == -1.0
    assert down["momentum_score"] == pytest.approx(up["momentum_score"], abs=5e-4)


def test_score_of_a_strong_clean_trend_exceeds_that_of_a_zigzag(momentum):
    building = values(momentum, bars_from_closes([1000.0 + 0.02 * (i + 1) ** 2 for i in range(60)]))
    straight = values(momentum, ramp_bars(n=60))
    chop = values(momentum, zigzag_bars())
    assert building["momentum_score"] > 0.8
    assert straight["momentum_score"] > 0.75
    assert chop["momentum_score"] < 0.05
    # Displacement and efficiency saturate on both trends; the gap between
    # them is the acceleration term, and only the convex path earns it.
    assert building["momentum_score"] > straight["momentum_score"]


def test_score_of_a_dead_tape_is_exactly_zero(momentum):
    """No displacement, no efficiency, no direction, so no evidence of
    momentum and no points. A term whose neutral value were 0.5 would have
    put `w_accel / 2` -- a tenth of the component -- on a tape that has not
    moved, which is the floor this formulation exists to avoid."""
    assert values(momentum, flat_bars())["momentum_score"] == 0.0
    assert values(momentum, zigzag_bars())["momentum_score"] == 0.0
    assert MOMENTUM_SCORE_WEIGHT_ACCELERATION > 0.0, "otherwise this proves nothing"


def test_score_stays_in_the_unit_interval_on_generated_data(momentum, labelled):
    data = symbol_data(labelled.bars)
    stamps = labelled.bars.ts_ns
    observed = []
    for index in range(momentum.warmup_bars - 1, len(stamps), 59):
        vector = momentum.compute(view_of(data, stamps, index))
        score = vector.values["momentum_score"]
        assert 0.0 <= score <= 1.0
        observed.append(score)
    assert len(observed) > 100
    assert min(observed) < 0.4 < max(observed), "the score should actually vary"


def test_acceleration_against_the_move_scores_below_acceleration_with_it(momentum):
    """`acceleration * direction` is the whole point of that product: a move
    that is running out of steam must not score like one that is building.

    Both paths here rise monotonically with a comparable displacement, so
    `efficiency_ratio` is 1.0 on each and `displacement` saturates on each;
    the whole difference in score is the acceleration term.
    """
    building = values(momentum, bars_from_closes([1000.0 + 0.02 * (i + 1) ** 2 for i in range(60)]))
    fading = values(
        momentum, bars_from_closes([1000.0 + 20.0 * math.sqrt(i + 1) for i in range(60)])
    )
    assert building["acceleration"] > 0.0 > fading["acceleration"]
    assert building["direction"] == fading["direction"] == 1.0
    assert building["efficiency_ratio"] == fading["efficiency_ratio"] == 1.0
    assert building["momentum_score"] > fading["momentum_score"]


def test_a_fading_move_loses_its_acceleration_credit_rather_than_being_penalised(momentum):
    """Negative acceleration contributes zero, not a negative. The signed
    feature stays on the vector for the gates to read; the magnitude just
    stops crediting a move that is giving back its speed."""
    fading = values(momentum, bars_from_closes([1000.0 + 20.0 * math.sqrt(i + 1) for i in range(60)]))
    expected_without_acceleration = (
        MOMENTUM_SCORE_WEIGHT_ROC_ATR
        * math.tanh(abs(fading["roc_atr"]) / MOMENTUM_SCORE_ROC_ATR_REFERENCE)
        + MOMENTUM_SCORE_WEIGHT_EFFICIENCY * fading["efficiency_ratio"]
    )
    assert fading["acceleration"] < 0.0
    assert fading["momentum_score"] == pytest.approx(
        expected_without_acceleration, rel=1e-12
    )


# --- degenerate inputs ----------------------------------------------------


def test_every_feature_is_finite_on_a_perfectly_flat_series(momentum):
    """`_vector` raises on a non-finite value, so this reaching a dict at all
    is the assertion; the explicit check names the column if it regresses."""
    row = values(momentum, flat_bars())
    for key, value in row.items():
        assert math.isfinite(value), f"{key} is not finite on a flat series"
    assert row["direction"] == 0.0


def test_every_feature_is_finite_across_generated_data(momentum, labelled):
    data = symbol_data(labelled.bars)
    stamps = labelled.bars.ts_ns
    for index in range(momentum.warmup_bars - 1, len(stamps), 37):
        vector = momentum.compute(view_of(data, stamps, index))
        for key, value in vector.values.items():
            assert math.isfinite(value), f"{key} not finite at bar {index}"


def test_a_single_price_spike_does_not_unbound_any_feature(momentum):
    """One extreme observation is the failure mode percentile ranks and
    squashes exist to prevent: a z-score here would let it dominate."""
    path = [1000.0] * 59 + [40_000.0]
    row = values(momentum, bars_from_closes(path))
    assert -1.0 <= row["acceleration"] <= 1.0
    assert 0.0 <= row["efficiency_ratio"] <= 1.0
    assert 0.0 <= row["close_position"] <= 1.0
    assert 0.0 <= row["momentum_score"] <= 1.0
    assert -1.0 <= row["momentum_persistence"] <= 1.0


# --- lookahead ------------------------------------------------------------


@pytest.mark.lookahead
def test_lookahead_audit_passes_with_a_factory(audit_data, config):
    """`factory=` rather than an instance: that is what rebuilds the computer
    against each altered dataset and so catches a full-sample constant
    captured in `__init__`."""
    result = audit_computer(
        data=audit_data, factory=lambda data: MomentumFeatures(config), sample=40
    )
    assert_no_lookahead([result])
    assert result.bars_checked == 40
    assert result.keys_checked == tuple(sorted(EXPECTED_KEYS))


@pytest.mark.lookahead
def test_the_audit_would_catch_a_construction_time_capture(audit_data, config):
    """A passing audit on a computer the harness could not fail proves
    nothing, so the harness is shown to bite here.

    `_ConstructionCapture` deliberately does not subclass `FeatureComputer`:
    its constructor takes market data, which the suite-wide feature contract
    test forbids, and a banned shape should not be registered as a subclass
    merely to be used as a control.
    """
    clean = audit_computer(
        data=audit_data, factory=lambda data: MomentumFeatures(config), sample=12
    )
    assert clean.passed

    cheat = audit_computer(
        data=audit_data,
        factory=lambda data: _ConstructionCapture(config, data),
        sample=12,
    )
    assert not cheat.passed
    assert {finding.kind for finding in cheat.findings} == {"truncation", "mutation"}
    assert all(finding.key == "roc" for finding in cheat.findings)


@pytest.mark.lookahead
def test_features_at_a_bar_ignore_every_later_bar(momentum, labelled):
    """The firewall makes this true structurally; it is checked anyway,
    because that is the one property whose failure flatters a backtest
    instead of breaking it."""
    bars = labelled.bars
    index = momentum.warmup_bars + 40
    full = momentum.compute(view_at(bars, index))

    closes = bars.col("close").copy()
    closes[index + 1 :] *= 5.0
    highs = bars.col("high").copy()
    highs[index + 1 :] *= 5.0
    lows = bars.col("low").copy()
    lows[index + 1 :] *= 5.0
    opens = bars.col("open").copy()
    opens[index + 1 :] *= 5.0
    altered = BarSeries(
        symbol=SYMBOL,
        ts_ns=bars.ts_ns,
        interval_seconds=INTERVAL,
        columns={
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": bars.col("volume").copy(),
        },
    )
    assert momentum.compute(view_at(altered, index)).values == full.values

    truncated = bars.prefix(index + 1)
    assert momentum.compute(view_at(truncated, -1)).values == full.values


def test_two_separately_constructed_computers_agree_exactly(config, labelled):
    """Determinism: nothing here reads a clock or an unseeded generator."""
    bars = labelled.bars
    index = MomentumFeatures(config).warmup_bars + 17
    first = MomentumFeatures(config).compute(view_at(bars, index))
    second = MomentumFeatures(config).compute(view_at(bars, index))
    assert first.values == second.values


# --- the control used by the audit-bites test -----------------------------


class _ConstructionCapture:
    """A deliberately-cheating stand-in: it reads the full sample in its
    constructor and leaks that constant into a feature.

    Not a `FeatureComputer` subclass, on purpose -- see the test that uses
    it. `audit_computer` only needs `name`, `warmup_bars` and `compute`.
    """

    name = "construction_capture"

    def __init__(self, config: FeatureConfig, data: SymbolData) -> None:
        self._inner = MomentumFeatures(config)
        self._captured = float(np.mean(data.primary_bars.col("close")))

    @property
    def warmup_bars(self) -> int:
        return self._inner.warmup_bars

    def compute(self, view: MarketView) -> FeatureVector:
        vector = self._inner.compute(view)
        if not vector.warmup_complete:
            return vector
        leaked = dict(vector.values)
        leaked["roc"] = leaked["roc"] + self._captured * 1e-6
        return vector.model_copy(update={"values": leaked})
