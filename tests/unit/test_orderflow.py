"""Adversarial tests for the order-flow features.

`features/orderflow.py` is the one row of ARCHITECTURE.md section 5 that has no
degraded path at all:

    "No proxy accepted. Component disabled and the 25 points are reported
     UNAVAILABLE; the weight is NOT redistributed. Bar-volume-only 'delta' is a
     known-bad estimator and is not substituted silently."

Section 13's gate for Phase 4 is literally "Audit passes; no proxy
substitution", so that sentence -- not the formulas -- is what this file is
organized around. Seven things here can be silently wrong.

**A proxy could be substituted and nothing above would know.** A delta derived
from bar volume, a tick rule applied to bar closes, a side inferred from the
close's location in the range or from quote imbalance: every one produces a
plausible number in the right range, and the lookahead audit would pass it,
because none of them reads the future. The refusal cannot therefore be tested
by reading the code -- it has to be tested by handing the computer data that
*invites* a proxy and showing that nothing comes out.
`test_bars_only_disables_the_component_and_substitutes_nothing` is the whole
contract in one place: all fifteen keys exactly 0.0, every key MISSING and not
one DEGRADED, `warmup_complete` False, and a note that names section 5, the
word UNAVAILABLE, the non-redistribution and the four constructions it did not
perform. `test_two_opposite_bar_volume_tapes_produce_the_identical_refusal`
then drives the point: a tape whose every bar closes up on 50_000 lots and the
mirror tape that closes down on the same volume give byte-identical output,
which no volume-derived estimator could.

**MISSING and DEGRADED are different claims.** DEGRADED means "a named term
was dropped and its weight was not redistributed"; MISSING means "the
component is off". Section 5 permits the first for options and liquidity and
only the second for order flow, so every disabling condition is checked for
MISSING-everywhere and the degraded-but-usable paths are checked for the
absence of MISSING.

**A dead term in a weighted sum still looks like a score.** The magnitude is a
weighted net of four signed terms plus a size term, and if one were always 0
the magnitude would still move with the others. Every named feature in section
5's table is therefore pinned on a tape where its arithmetic is exact:
`signed_delta = -0.4` from eighteen of sixty ranked observations,
`cvd_slope = tanh(0.8)` from a CVD climbing 20 contracts a bar against a mean
classified volume of 100, `aggression_ratio = 0.6` from six buy prints in ten,
`absorption = 1.0` on a frozen tape, `max_trade_size_percentile = 0.90`
exactly at the large-trade boundary. And one of them IS dead at the defaults:
`test_absorption_is_structurally_zero_on_five_minute_data` measures the
excursion band independently and shows 0.15 of the component's weight cannot
be earned, while `order_flow_available_weight` still reports 1.0.

**Direction must not be able to inflate a magnitude (section 7).** The
magnitude is `|weighted net|`, so contradictory terms cancel.
`test_an_absorption_vote_against_the_delta_cancels_exactly_its_weight` holds
the tick feed bit-identical and moves only the bar closes, so the 0.15 drop is
the cancellation and provably not a dropped term.

**Under-declared warmup is a HIGH-severity bug the audit cannot see** -- the
regime detector declared 268 and read 291, which made the label at bar *t*
depend on where the caller started loading while nothing read the future.
`test_nothing_before_the_declared_warmup_reaches_the_output` therefore computes
the same bar twice, once from 400 rows and once from the last 130 of them, with
a non-degenerate tail so every term is a live measurement, and requires
bit-equality.

**The audit gate must be provably non-vacuous.** Three deliberate cheaters are
constructed and the harness is required to report findings for each:
construction-time capture (72 findings), an under-declared warmup (3), unseeded
jitter (111). The second needed care and the result is itself a finding: the
harness's warmup check can only see "reported ready too early", so a computer
that under-declares *and* still refuses a short window produces zero findings.
That is exactly why the arithmetic warmup test above exists.

Five things in here are honest gaps or pinned defects rather than passing
claims, and each says so where it sits: the structurally-zero absorption band,
`classification_coverage`'s blindness to a collapse on the newest bars, the
undetected interior tick/bar gap, the size term's direction-free contribution
to a magnitude whose stated principle is that direction-free evidence should
not raise it, and `signed_delta`'s inexact antisymmetry.

**What is NOT tested here, and why.** Four things are left untested on purpose
rather than covered badly. (1) Interior tick/bar row alignment cannot be
checked from outside: `MarketView` exposes no tick timestamps, so a test can
construct the gap but cannot assert the module should have seen it. (2) Five
branches of the module are unreachable and have no test:
`compute`'s "the tick or bar feed has no visible observation", which
`has_feed` already guarantees against; `_read_windows`'s
"short unclassified_volume window", because `tick_column` returns either an
empty array or a full one once the warmup gate has passed; its three
negative-volume checks, which `TickSeries.NON_NEGATIVE` refuses at construction
(pinned by `test_the_negative_volume_branches_are_dead_because_the_series_
refuses_them`); and `_cvd_slope`'s non-finite guard, because `safe_divide`
cannot return a non-finite value. This header said TWO until an independent
mutation sweep went looking, which is worth recording: a claim about what
cannot be tested is itself a claim, and this one was wrong. (3)
`classification_method` lives in `TickSeries.meta` and is therefore
series-level, so a feed that changed methods mid-history is indistinguishable
from one that always used its current method; nothing at this layer can tell.
(4) `_DeltaTerm.measured` is set and never read by the module, so no EMITTED
value distinguishes a degenerate delta window's stated 0.0 from a measured
median rank -- only the note does, and the key's quality stays GOOD, because
this module's DEGRADED means "a term was dropped" and nothing was. It is now
asserted directly, with the other four diagnostic fields the term classes
carry, in `test_the_diagnostic_fields_the_term_classes_carry_are_actually_
asserted`: those classes claimed the fields were "asserted" in a test and not
one of them was. And the
non-redistribution of the 25 points is only HALF enforceable here: this module
reports availability honestly and reallocates nothing, which is what the tests
above check; the final refusal to turn an unavailable component into 100
attainable points lives in `signals/flow_score.py` and
`strict_component_availability`, which are Phase 6.

Builders are local on purpose: file ownership, and so a failure localizes here
rather than to a shared fixture.

**The tape, and why its arithmetic is exact.** Every bar is `high = mid + 1`,
`low = mid - 1`, `close = open = mid`, and `mid` moves by at most 1.0 per bar.
Then the true range is
`max(2, |mid_i + 1 - mid_{i-1}|, |mid_i - 1 - mid_{i-1}|) == 2.0` on every bar,
so the Wilder seed over fourteen ranges is exactly 2.0 -- which makes the
absorption displacement band `0.25 * 2.0 = 0.5` points and the level band the
same, numbers written into the tests rather than read out of the
implementation. `test_the_tape_premise_the_local_atr_is_exactly_two` asserts
the premise before anything relies on it, recomputing the ATR from
`volatility.true_range` and `volatility.wilder_atr_series` directly rather than
from the module under test.

A NOTE ON WHAT IS NOT TESTED HERE, DELIBERATELY. The only dataset available is
`data/synthetic.py`, whose tick stream is generated from the SAME innovations
as its bars. It is used for mechanics only -- bounds, finiteness, determinism,
degradation, lookahead, and the one measured frequency above. No test in this
file correlates any feature with a forward return, and no such number is
quoted, because it would describe the generator and not the feature. The
temptation was real: a one-line check that `signed_delta` leads the next bar's
return would have looked like validation and would have been worth nothing.
"""

from __future__ import annotations

import inspect
import math
from datetime import date, datetime, timedelta, timezone
from typing import get_type_hints

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.config.schema import (
    FeatureConfig,
    OrderFlowConfig,
    OrderFlowWeights,
)
from flow_model.core.contracts import FeatureVector, TickAggregate
from flow_model.core.enums import DataQuality, Feed, InstrumentType
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.calendar import TradingCalendar
from flow_model.data.market_view import MarketView
from flow_model.data.base import SchemaError
from flow_model.data.series import (
    BarSeries,
    QuoteSeries,
    TickSeries,
    from_ns,
    to_ns,
)
from flow_model.data.store import SymbolData
from flow_model.data.synthetic import SyntheticConfig, SyntheticMarketGenerator
from flow_model.features.base import FeatureBundle, FeatureComputer, FeatureError
from flow_model.features.orderflow import (
    ABSORPTION_TERM_COUNT,
    NEUTRAL_RATIO,
    REFUSED_PROXY_CONSTRUCTIONS,
    UNAVAILABLE_PERCENTILE,
    UNKNOWN_CLASSIFICATION_METHOD,
    OrderFlowFeatures,
    excess_above,
    inside_band,
    is_degenerate,
    rolling_max,
    rolling_mean,
    rolling_sum,
    signed_rank,
)
from flow_model.features.volatility import true_range, wilder_atr_series
from flow_model.validation.lookahead import assert_no_lookahead, audit_computer

SYMBOL = "NQ"
INTERVAL = 300
T0 = datetime(2024, 3, 4, 14, 35, tzinfo=timezone.utc)
STEP_NS = INTERVAL * 1_000_000_000

#: The declared contract, written out rather than read from the config, so that
#: a config edit fails `test_the_config_defaults_these_tests_are_written_against`
#: loudly instead of silently changing what every other test here means.
WARMUP = 130
SMOOTHING = 3
DELTA_LOOKBACK = 60
CVD_LOOKBACK = 20
CVD_SQUASH = 0.25
POOL = 10
MIN_TRADES = 20
MIN_COVERAGE = 0.60
ABSORPTION_LOOKBACK = 10
ABSORPTION_PERCENTILE = 0.80
MAX_DISPLACEMENT_ATR = 0.25
LEVEL_BAND_ATR = 0.25
SIZE_LOOKBACK = 120
LARGE_PERCENTILE = 0.90
MIN_AVAILABLE_FRACTION = 0.50
ATR_PERIOD = 14

W_DELTA = 0.25
W_CVD = 0.25
W_AGGRESSION = 0.20
W_ABSORPTION = 0.15
W_SIZE = 0.15

#: The read windows the module declares in its docstring table.
TICK_ROWS = 129          # trade-size chain: SIZE_LOOKBACK + POOL - 1
BAR_ROWS = 15            # ATR_PERIOD + 1

KEYS = (
    "order_flow_magnitude",
    "order_flow_direction",
    "order_flow_available",
    "order_flow_available_weight",
    "signed_delta",
    "signed_delta_contracts",
    "cvd_slope",
    "cvd_slope_normalized",
    "aggression_ratio",
    "absorption",
    "absorption_direction",
    "avg_trade_size_percentile",
    "max_trade_size_percentile",
    "large_trade_event",
    "classification_coverage",
)

#: Keys bounded to [0, 1] and [-1, 1] by construction, per the module docstring.
UNIT_KEYS = (
    "order_flow_magnitude",
    "order_flow_available",
    "order_flow_available_weight",
    "aggression_ratio",
    "absorption",
    "avg_trade_size_percentile",
    "max_trade_size_percentile",
    "large_trade_event",
    "classification_coverage",
)
SIGNED_KEYS = ("signed_delta", "cvd_slope", "order_flow_direction", "absorption_direction")
#: Reported in natural / dimensionless units: diagnostics, unbounded by design.
DIAGNOSTIC_KEYS = ("signed_delta_contracts", "cvd_slope_normalized")

HONEST_METHOD = "bid_ask"


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def order_flow(**overrides) -> OrderFlowConfig:
    return OrderFlowConfig(**overrides)


def features(**overrides) -> FeatureConfig:
    return FeatureConfig(**overrides)


def computer(**overrides) -> OrderFlowFeatures:
    """`OrderFlowFeatures(OrderFlowConfig, FeatureConfig)`.

    Note the argument order: this computer takes its OWN section first and the
    shared `FeatureConfig` second, the reverse of `LiquidityFeatures`,
    `StructureFeatures` and `LevelFeatures`. Swapping them raises rather than
    mis-computing, so it is a readability wart and not a hazard, but it is why
    every construction in this file goes through here.
    """
    feature_overrides = {k: overrides.pop(k) for k in ("atr_period",) if k in overrides}
    return OrderFlowFeatures(order_flow(**overrides), features(**feature_overrides))


def nq_spec(**overrides) -> InstrumentSpec:
    fields = dict(
        symbol=SYMBOL,
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        typical_spread_ticks=1.0,
        min_spread_ticks=1.0,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
        eth=SessionWindow(name="ETH", start="18:00", end="17:00"),
    )
    fields.update(overrides)
    return InstrumentSpec(**fields)


def stamps(n: int, offset: int = 0) -> np.ndarray:
    """`n` strictly ascending bar-close stamps on a contiguous grid."""
    return np.array([to_ns(T0) + STEP_NS * (offset + i) for i in range(n)], dtype=np.int64)


def bars_from_mids(mids, offset: int = 0, volume=1000.0, flat: bool = False) -> BarSeries:
    """Bars with `high = mid + 1`, `low = mid - 1`, `open = close = mid`.

    On a tape whose `mid` moves by at most 1.0 per bar the true range is
    `max(2, |mid_i + 1 - mid_{i-1}|, |mid_i - 1 - mid_{i-1}|) == 2.0` on every
    bar, so the Wilder seed is exactly 2.0. `flat=True` collapses the bar to a
    single price, which makes every true range 0.0 and the ATR undefined -- the
    one input the absorption term can be missing.
    """
    mids = np.asarray(mids, dtype=np.float64)
    n = mids.size
    volumes = np.full(n, float(volume)) if np.isscalar(volume) else np.asarray(volume, float)
    span = 0.0 if flat else 1.0
    return BarSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n, offset),
        interval_seconds=INTERVAL,
        columns={
            "open": mids.copy(),
            "high": mids + span,
            "low": mids - span,
            "close": mids.copy(),
            "volume": volumes,
        },
    )


def tick_series(
    n: int,
    *,
    buy=60.0,
    sell=40.0,
    unclassified=None,
    buy_trades=6.0,
    sell_trades=4.0,
    max_trade=None,
    method: str = HONEST_METHOD,
    offset: int = 0,
    ts_ns=None,
) -> TickSeries:
    """A tick-aggregate series. `None` for a column means NOT SUPPLIED.

    Scalars are broadcast so a test can say "sixty lots bought a bar" without
    writing an array. `buy_trades`/`sell_trades`/`max_trade` default to a
    supplied-but-constant column except `max_trade`, which defaults to absent,
    because most tests here are about the two required columns.
    """

    def column(value):
        if value is None:
            return None
        if np.isscalar(value):
            return np.full(n, float(value))
        return np.asarray(value, dtype=np.float64)

    columns = {"buy_volume": column(buy), "sell_volume": column(sell)}
    for name, value in (
        ("unclassified_volume", unclassified),
        ("buy_trades", buy_trades),
        ("sell_trades", sell_trades),
        ("max_trade_size", max_trade),
    ):
        materialized = column(value)
        if materialized is not None:
            columns[name] = materialized
    return TickSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n, offset) if ts_ns is None else np.asarray(ts_ns, dtype=np.int64),
        columns=columns,
        meta={"classification_method": method},
    )


def symbol_data(bars: BarSeries, ticks: TickSeries | None = None, quotes=None) -> SymbolData:
    return SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: bars},
        ticks={INTERVAL: ticks} if ticks is not None else None,
        quotes=quotes,
    )


def tape(mids, *, offset: int = 0, flat: bool = False, volume=1000.0, **tick_kwargs) -> SymbolData:
    """Bars and an aligned tick feed of the same length and cadence."""
    mids = np.asarray(mids, dtype=np.float64)
    bars = bars_from_mids(mids, offset=offset, volume=volume, flat=flat)
    ticks = tick_series(mids.size, offset=offset, **tick_kwargs)
    return symbol_data(bars, ticks)


def flat_tape(n: int = 160, mid: float = 100.0, **tick_kwargs) -> SymbolData:
    return tape(np.full(n, float(mid)), **tick_kwargs)


def view_at(data: SymbolData, index: int = -1) -> MarketView:
    """A view whose cutoff is bar `index`'s close, so that bar is visible."""
    bars = data.primary_bars
    position = index if index >= 0 else len(bars) + index
    ts = int(bars.ts_ns[position])
    return MarketView(data, now=from_ns(ts), now_ns=ts)


def quote_feed(n: int, *, offset: int = 0) -> QuoteSeries:
    """A one-sided book: the richest L1 data a quote feed can carry.

    Used to show that it buys the order-flow component nothing. A book 100:1
    bid-heavy is exactly what a quote-imbalance proxy would read as aggressive
    buying.
    """
    return QuoteSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n, offset),
        columns={
            "bid": np.full(n, 99.75),
            "ask": np.full(n, 100.25),
            "bid_size": np.full(n, 500.0),
            "ask_size": np.full(n, 5.0),
        },
    )


def local_atr(view: MarketView, period: int = ATR_PERIOD) -> float:
    """The Wilder seed over the last `period` true ranges, recomputed here.

    Computed from `volatility.true_range` and `volatility.wilder_atr_series`
    directly -- the same shared helpers `orderflow.py` imports, but called from
    the test, so an expected band is this file's arithmetic rather than the
    module's answer handed back to itself.
    """
    rows = period + 1
    ranges = true_range(view.highs(rows), view.lows(rows), view.closes(rows))
    return float(wilder_atr_series(ranges, period)[-1])


def note_with(vector: FeatureVector, fragment: str) -> str:
    """The one note containing `fragment`, or a failure naming what was there."""
    hits = [note for note in vector.notes if fragment in note]
    assert len(hits) == 1, f"expected one note containing {fragment!r}, got {vector.notes}"
    return hits[0]


def has_note(vector: FeatureVector, fragment: str) -> bool:
    return any(fragment in note for note in vector.notes)


def synthetic_data(
    *, include_ticks: bool = True, seed: int = 7, months: int = 2
) -> SymbolData:
    """Generated 5-minute NQ spanning well over 2x `warmup_bars`.

    MECHANICS ONLY. The generator builds its tick stream from the same
    innovations as its bars, so a correlation between any feature here and a
    forward return would be a property of the generator. None is computed.
    """
    generator = SyntheticMarketGenerator(
        SyntheticConfig(
            include_ticks=include_ticks, include_quotes=False, include_options=False
        ),
        seed=seed,
    )
    dataset = generator.generate(
        SYMBOL,
        nq_spec(),
        date(2024, 1, 2),
        date(2024, 1 + months, 1),
        INTERVAL,
        calendar=TradingCalendar(),
    )
    return SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: dataset.bars},
        ticks={INTERVAL: dataset.ticks} if dataset.ticks is not None else None,
    )


# ---------------------------------------------------------------------------
# premises
# ---------------------------------------------------------------------------


def test_the_config_defaults_these_tests_are_written_against():
    """Every hand-computed number below is a number only at these defaults.

    Without this test a config edit would turn `signed_delta == -0.4` from a
    check of the implementation into a check of nothing: the tape would be built
    against one lookback and the assertion written against another, and the
    failure would read as an arithmetic bug in `orderflow.py`.
    """
    config = order_flow()
    assert config.delta_smoothing_bars == SMOOTHING
    assert config.delta_percentile_lookback == DELTA_LOOKBACK
    assert config.cvd_lookback_bars == CVD_LOOKBACK
    assert config.cvd_slope_squash_scale == CVD_SQUASH
    assert config.aggression_lookback_bars == POOL
    assert config.min_classified_trades == MIN_TRADES
    assert config.min_classification_coverage == MIN_COVERAGE
    assert config.absorption_lookback_bars == ABSORPTION_LOOKBACK
    assert config.absorption_delta_percentile == ABSORPTION_PERCENTILE
    assert config.absorption_max_displacement_atr == MAX_DISPLACEMENT_ATR
    assert config.absorption_level_window_atr == LEVEL_BAND_ATR
    assert config.trade_size_percentile_lookback == SIZE_LOOKBACK
    assert config.large_trade_percentile == LARGE_PERCENTILE
    assert config.min_available_weight_fraction == MIN_AVAILABLE_FRACTION
    assert config.warmup_bars == WARMUP
    assert features().atr_period == ATR_PERIOD

    weights = config.weights
    assert (weights.signed_delta, weights.cvd_slope, weights.aggression_ratio) == (
        W_DELTA,
        W_CVD,
        W_AGGRESSION,
    )
    assert (weights.absorption, weights.trade_size_distribution) == (W_ABSORPTION, W_SIZE)
    # The magnitude is bounded by this sum, not by the clamp that follows it.
    assert sum(
        (W_DELTA, W_CVD, W_AGGRESSION, W_ABSORPTION, W_SIZE)
    ) == pytest.approx(1.0)
    # The chain arithmetic the warmup declaration rests on.
    assert SIZE_LOOKBACK + POOL - 1 == TICK_ROWS
    assert DELTA_LOOKBACK + SMOOTHING - 1 == 62
    assert DELTA_LOOKBACK + ABSORPTION_LOOKBACK - 1 == 69
    assert ATR_PERIOD + 1 == BAR_ROWS


def test_the_tape_premise_the_local_atr_is_exactly_two():
    """Every absorption threshold below is a number because of this.

    `high = mid + 1`, `low = mid - 1`, `close = mid` with `|d mid| <= 1` makes
    every true range `max(2, |d mid + 1|, |d mid - 1|) == 2`, so the Wilder seed
    over fourteen of them is 2.0 to the bit. Then the displacement band is
    `0.25 * 2.0 = 0.5` points and the level band the same, and a test can write
    0.5 instead of asking the module what it used.
    """
    view = view_at(flat_tape())
    assert local_atr(view) == 2.0

    drifting = np.cumsum(np.resize([0.9, -0.9, 0.4, -0.4], 160)) + 100.0
    assert local_atr(view_at(tape(drifting))) == 2.0

    # And the one case that has no scale at all.
    assert local_atr(view_at(tape(np.full(160, 100.0), flat=True))) == 0.0


# ---------------------------------------------------------------------------
# the declared contract
# ---------------------------------------------------------------------------


def test_the_declared_keys_are_the_five_features_section_five_names():
    """`keys` is the contract `FeatureBundle` checks the output against, so a
    renamed key is a silently missing feature downstream. Section 5 names five
    features; this module emits each as a bounded term, keeps the unbounded
    diagnostic beside the two that have one, and adds the availability triple
    that the no-proxy rule is reported through."""
    comp = computer()
    assert comp.keys == KEYS
    assert len(set(KEYS)) == 15
    assert comp.name == "orderflow"
    # Each of the five named features is present under a name that says what it
    # is -- never "institutional", "whale" or "smart money" (section 0.3).
    for named in (
        "signed_delta",
        "cvd_slope",
        "aggression_ratio",
        "absorption",
        "max_trade_size_percentile",
    ):
        assert named in KEYS
    joined = " ".join(KEYS)
    for banned in ("institution", "whale", "smart_money", "dark_pool"):
        assert banned not in joined


def test_warmup_is_the_longest_chain_floored_at_the_config_floor():
    """Under-declaring this is the HIGH-severity bug the lookahead audit cannot
    see. The longest tick chain is the pooled trade size then its percentile,
    `120 + 10 - 1 = 129` rows; the longest bar chain is the local ATR's
    `14 + 1 = 15`; `OrderFlowConfig.warmup_bars` publishes 130 as a floor no
    computer may declare below. So 130, one bar conservative, which is the safe
    direction."""
    comp = computer()
    assert comp.warmup_bars == WARMUP
    assert comp.warmup_bars >= TICK_ROWS
    assert comp.warmup_bars >= BAR_ROWS
    assert comp.warmup_bars >= order_flow().warmup_bars

    # The declaration tracks the windows rather than being a constant: widen a
    # chain past the floor and it must grow with it.
    wider = computer(trade_size_percentile_lookback=400)
    assert wider.warmup_bars == 400 + POOL - 1 + 1 == 410
    # ... including a bar chain, which the config floor does not know about.
    long_atr = computer(atr_period=500)
    assert long_atr.warmup_bars == 501


def test_the_tick_feed_is_required_and_nothing_is_optional():
    """The whole point of the row. `TICK_AGGREGATE` optional would mean "runs
    without it", and what it would run on is a proxy. Quotes are not optional
    either: they could classify ticks, but classification is the data layer's
    job (`TickSeries.classification_method`), and doing it here from L1 would be
    this module inventing an aggressor side."""
    comp = computer()
    assert comp.required_feeds == frozenset({Feed.BARS, Feed.TICK_AGGREGATE})
    assert comp.optional_feeds == frozenset()
    assert Feed.QUOTES not in (comp.required_feeds | comp.optional_feeds)
    assert Feed.OPTIONS_SNAPSHOT not in (comp.required_feeds | comp.optional_feeds)
    assert not comp.feeds_available(view_at(symbol_data(bars_from_mids(np.full(160, 100.0)))))
    assert comp.feeds_available(view_at(flat_tape()))


def test_the_constructor_takes_configuration_only():
    """Local echo of the package-wide ban in `test_feature_contracts.py`. A
    computer handed a series could capture a full-sample statistic at
    construction, which is the one leak `validation.lookahead` cannot see
    through when it is given an instance rather than a factory.

    `orderflow.py` uses postponed annotations, so `__init__.__annotations__`
    holds STRINGS: they are resolved against the module globals with
    `get_type_hints` before comparing, because comparing a `str` to a class
    passes silently and has already done so twice in this project."""
    parameters = inspect.signature(OrderFlowFeatures.__init__).parameters
    assert list(parameters) == ["self", "config", "features"]

    hints = get_type_hints(OrderFlowFeatures.__init__)
    assert hints["config"] is OrderFlowConfig
    assert hints["features"] is FeatureConfig
    for name in ("config", "features"):
        assert not isinstance(hints[name], str), f"{name} hint was not resolved"

    rendered = str(inspect.signature(OrderFlowFeatures.__init__))
    for banned in (
        "SymbolData",
        "DataStore",
        "BarSeries",
        "TickSeries",
        "ColumnSeries",
        "MarketView",
        "FeatureVector",
        "SyntheticDataset",
        "ndarray",
        "DataFrame",
        "Series",
    ):
        assert banned not in rendered, f"{banned} must not appear in the constructor"


def test_the_constructor_refuses_a_config_that_claims_a_proxy_is_allowed():
    """`OrderFlowConfig` already refuses this, so reaching the computer means a
    caller built the config through `model_construct` or an equivalent bypass.
    The component would still not substitute anything -- there is no code that
    could -- but a config asserting that it may is a disagreement about the
    rule, and refusing to run is the right answer to that."""
    base = order_flow()
    stated = {name: getattr(base, name) for name in OrderFlowConfig.model_fields}

    permissive = OrderFlowConfig.model_construct(
        **{**stated, "allow_bar_volume_delta_proxy": True}
    )
    with pytest.raises(FeatureError, match="accepts NO proxy"):
        OrderFlowFeatures(permissive, features())

    unclassified_ok = OrderFlowConfig.model_construct(
        **{**stated, "require_aggressor_classification": False}
    )
    with pytest.raises(FeatureError, match="require_aggressor_classification"):
        OrderFlowFeatures(unclassified_ok, features())

    # And the schema refuses both before a computer ever sees them.
    with pytest.raises(ValueError, match="known-bad estimator"):
        OrderFlowConfig(allow_bar_volume_delta_proxy=True)
    with pytest.raises(ValueError, match="back door"):
        OrderFlowConfig(require_aggressor_classification=False)


def test_the_constructor_refuses_weights_that_sum_to_zero():
    """The magnitude divides by the total weight. A zero total is a component
    that could never report anything, and a ZeroDivisionError at bar 130 of a
    backtest is a worse way to find out."""
    base = order_flow()
    dead = OrderFlowWeights.model_construct(
        signed_delta=0.0,
        cvd_slope=0.0,
        aggression_ratio=0.0,
        absorption=0.0,
        trade_size_distribution=0.0,
    )
    broken = OrderFlowConfig.model_construct(
        **{**{name: getattr(base, name) for name in OrderFlowConfig.model_fields}, "weights": dead}
    )
    with pytest.raises(FeatureError, match="sum to zero"):
        OrderFlowFeatures(broken, features())


def test_the_rejected_proxy_list_names_what_section_five_rejects():
    """The refusal is greppable on purpose: a rule implied by absent code cannot
    be tested, and the next person to need a delta when the tick feed is down
    will not find the absent code."""
    assert REFUSED_PROXY_CONSTRUCTIONS[0] == "bar volume as signed delta"
    joined = " | ".join(REFUSED_PROXY_CONSTRUCTIONS).lower()
    for construction in ("bar volume", "tick rule", "close location", "quote imbalance"):
        assert construction in joined
    assert UNKNOWN_CLASSIFICATION_METHOD == "unknown"
    # The TickSeries default really is the value the module refuses, rather than
    # a string it invented.
    assert tick_series(2).classification_method == HONEST_METHOD
    bare = TickSeries(
        symbol=SYMBOL,
        ts_ns=stamps(2),
        columns={"buy_volume": np.ones(2), "sell_volume": np.ones(2)},
    )
    assert bare.classification_method == UNKNOWN_CLASSIFICATION_METHOD
    # Every known-bad name the config ships must stay on the denylist.
    for name in ("bar_volume", "tick_rule_on_bars", "volume_split"):
        assert name in order_flow().rejected_classification_methods


# ---------------------------------------------------------------------------
# the UNAVAILABLE contract -- the Phase 4 gate
# ---------------------------------------------------------------------------


def assert_fully_unavailable(vector: FeatureVector) -> str:
    """The whole disablement contract, in one place, as section 5 states it.

    Returns the single note so a caller can assert on the reason as well. Every
    disabling condition in the module routes through here, which is the point:
    the contract is a property of the component, not of one branch.
    """
    assert set(vector.values) == set(KEYS)
    assert set(vector.values.values()) == {0.0}, vector.values
    assert set(vector.quality_by_key) == set(KEYS)
    assert set(vector.quality_by_key.values()) == {DataQuality.MISSING}
    assert DataQuality.DEGRADED not in set(vector.quality_by_key.values())
    assert vector.quality is DataQuality.MISSING
    assert not vector.warmup_complete
    # The availability report itself must say unavailable, not merely be zero
    # because everything is zero.
    assert vector.values["order_flow_available"] == 0.0
    assert vector.values["order_flow_available_weight"] == 0.0
    assert vector.values["order_flow_magnitude"] == 0.0
    assert vector.values["order_flow_direction"] == 0.0
    # Nothing that looks like a measurement survives: a reader cannot mistake
    # 0.5 for a neutral ratio or 1.0 for full coverage.
    assert vector.values["aggression_ratio"] != NEUTRAL_RATIO
    assert vector.values["avg_trade_size_percentile"] != UNAVAILABLE_PERCENTILE
    assert vector.values["classification_coverage"] == 0.0
    assert len(vector.notes) == 1
    note = vector.notes[0]
    assert "ORDER FLOW UNAVAILABLE" in note
    assert "section 5" in note
    assert "no proxy" in note
    assert "25 points" in note and "UNAVAILABLE" in note
    assert "NOT redistributed" in note
    for refused in ("bar-volume delta", "tick rule on bars", "close-location", "quote-imbalance"):
        assert refused in note, refused
    return note


def test_bars_only_disables_the_component_and_substitutes_nothing():
    """THE PHASE 4 GATE, as a whole contract rather than a branch.

    Section 5's order-flow row is the only one in the table with no degraded
    path, so a bars-only feed must produce disablement and not a flagged
    estimate: every key exactly 0.0, every key MISSING and not one DEGRADED,
    `warmup_complete` False, and one note that names the rule, the 25 points,
    the non-redistribution and each construction that was not performed. The
    tape handed in is the most inviting possible: 400 bars of real OHLCV with
    50_000 lots a bar, which is everything a bar-volume delta would need."""
    bars = bars_from_mids(100.0 + np.cumsum(np.resize([0.5, -0.25], 400)), volume=50_000.0)
    vector = computer().compute(view_at(symbol_data(bars)))
    note = assert_fully_unavailable(vector)
    assert "tick-aggregate feed is absent" in note

    # And a quote feed -- a 100:1 bid-heavy book, which a quote-imbalance proxy
    # would read as heavy aggressive buying -- buys nothing either.
    with_quotes = computer().compute(
        view_at(symbol_data(bars, quotes=quote_feed(400)))
    )
    assert_fully_unavailable(with_quotes)
    assert with_quotes.values == vector.values


def test_two_opposite_bar_volume_tapes_produce_the_identical_refusal():
    """No volume-derived estimator could pass this.

    One tape closes up on every bar, the other closes down on every bar, both on
    50_000 lots. Any of the five refused constructions -- signed bar volume, a
    tick rule on closes, close-location-in-range, a configured split -- would
    give these two tapes opposite deltas. The outputs are byte-identical zeros,
    which is the only reading consistent with "nothing was substituted"."""
    n = 200
    rising = BarSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n),
        interval_seconds=INTERVAL,
        columns={
            "open": np.full(n, 99.5),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),   # closes at the top of its range
            "volume": np.full(n, 50_000.0),
        },
    )
    falling = BarSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n),
        interval_seconds=INTERVAL,
        columns={
            "open": np.full(n, 100.5),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 99.5),    # closes at the bottom of it
            "volume": np.full(n, 50_000.0),
        },
    )
    comp = computer()
    up = comp.compute(view_at(symbol_data(rising)))
    down = comp.compute(view_at(symbol_data(falling)))
    assert_fully_unavailable(up)
    assert_fully_unavailable(down)
    assert up.values == down.values
    assert up.notes == down.notes
    assert up.quality_by_key == down.quality_by_key


def test_an_undeclared_classification_method_disables_the_component():
    """`TickSeries.classification_method` defaults to 'unknown', and volume with
    an unexplained side is bar volume with a label. A feed that will not say how
    it assigned the aggressor is the proxy admitted through the back door, which
    is why `require_aggressor_classification` cannot be switched off."""
    for method in (UNKNOWN_CLASSIFICATION_METHOD, "UNKNOWN", "", "   "):
        vector = computer().compute(view_at(flat_tape(method=method)))
        note = assert_fully_unavailable(vector)
        assert "not a classification" in note
        assert "bar volume with a label" in note


def test_a_known_bad_method_is_refused_whatever_its_case_or_padding():
    """`'Bar_Volume'` slipping past a denylist is the one failure the field's
    normalizer exists to prevent, and the computer lower-cases a second time
    because a config built by `model_construct` skips the normalizer."""
    for method in ("bar_volume", "TICK_RULE_ON_BARS", " Volume_Split ", "Uptick_Downtick_On_Bars"):
        vector = computer().compute(view_at(flat_tape(method=method)))
        note = assert_fully_unavailable(vector)
        assert "rejected_classification_methods" in note


def test_an_honest_method_name_is_accepted_whatever_its_case():
    """A denylist, not an allowlist: the honest methods are open-ended (an
    exchange's own tag, 'bid_ask', the synthetic generator's
    'synthetic_aggressor') while the known-bad ones are a short named set.
    Rejecting an unrecognized honest name would disable the component on every
    real feed whose vendor spells it differently."""
    for method in ("bid_ask", "Bid_Ask", "exchange_aggressor_tag", "synthetic_aggressor"):
        vector = computer().compute(view_at(flat_tape(method=method)))
        assert vector.values["order_flow_available"] == 1.0, method
        assert vector.warmup_complete


def test_coverage_below_the_floor_disables_rather_than_degrades():
    """A delta whose sign is set by a minority of the tape is the known-bad
    estimator under another name, so it is MISSING and not DEGRADED. Hand
    arithmetic: 30 bought + 20 sold against 50 unclassified is 50/100 = 0.50
    coverage, below the 0.60 floor."""
    vector = computer().compute(
        view_at(flat_tape(buy=30.0, sell=20.0, unclassified=50.0))
    )
    note = assert_fully_unavailable(vector)
    assert "classification coverage 0.5000" in note
    assert "min_classification_coverage 0.6000" in note
    assert f"{TICK_ROWS}-row window" in note


def test_coverage_exactly_at_the_floor_is_accepted():
    """The floor is a floor, not a strict bound: 36 + 24 classified against 40
    unclassified is exactly 60/100, and refusing it would make the configured
    0.60 mean 0.60-plus-an-epsilon."""
    vector = computer().compute(
        view_at(flat_tape(buy=36.0, sell=24.0, unclassified=40.0))
    )
    assert vector.values["classification_coverage"] == pytest.approx(0.60)
    assert vector.values["order_flow_available"] == 1.0
    assert vector.warmup_complete
    assert vector.quality_of("classification_coverage") is DataQuality.GOOD


def test_classification_coverage_is_the_classified_share_of_window_volume():
    """Hand-computed over the whole read window: 129 rows of 60 bought, 40 sold
    and 25 unclassified is 129*100 / (129*100 + 129*25) = 100/125 = 0.80."""
    vector = computer().compute(
        view_at(flat_tape(buy=60.0, sell=40.0, unclassified=25.0))
    )
    assert vector.values["classification_coverage"] == pytest.approx(0.80)

    # An absent column is absence, not zero: the module takes coverage from the
    # feed's own columns and says in a note that it did NOT cross-check against
    # bar volume, because bar volume and tick volume are different aggregations.
    without = computer().compute(view_at(flat_tape(unclassified=None)))
    assert without.values["classification_coverage"] == 1.0
    note = note_with(without, "no unclassified_volume column")
    assert "NOT cross-checked against bar volume" in note

    # No volume at all is 0.0, not 1.0: "nothing was unclassified" and "nothing
    # was classified either" are different facts and the second disables.
    empty = computer().compute(view_at(flat_tape(buy=0.0, sell=0.0, unclassified=None)))
    assert_fully_unavailable(empty)


def test_a_component_below_the_available_weight_floor_reports_unavailable():
    """Dropping terms without redistribution is the honest rule, and without a
    floor it yields a systematically small sub-score that reads downstream as
    'the tape is balanced' when it means 'most of this was never measured'.

    Hand arithmetic: a feed with no trade counts and no max size drops
    `aggression_ratio` (0.20) and the whole trade-size term (0.15); killing the
    classified volume in the newest 20 rows also drops `cvd_slope` (0.25), which
    has no scale to be made dimensionless with. 0.25 + 0.15 = 0.40 of the weight
    is left, below the 0.50 floor."""
    buy = np.full(160, 60.0)
    sell = np.full(160, 40.0)
    buy[-CVD_LOOKBACK:] = 0.0
    sell[-CVD_LOOKBACK:] = 0.0
    vector = computer().compute(
        view_at(
            flat_tape(
                buy=buy,
                sell=sell,
                unclassified=None,
                buy_trades=None,
                sell_trades=None,
                max_trade=None,
            )
        )
    )
    note = assert_fully_unavailable(vector)
    assert "only 0.4000 of the term weight has real inputs" in note
    assert "floor 0.5000" in note
    assert "cvd_slope" in note and "aggression_ratio" in note
    assert "trade_size_distribution" in note
    assert "never redistributed" in note


def test_a_non_finite_required_column_disables_rather_than_propagating():
    """A NaN in the delta window would reach the score as a NaN magnitude, and
    `_vector` would raise from inside a feature computer rather than reporting.
    A required input that is dirty is a reason to disable the component."""
    buy = np.full(160, 60.0)
    buy[-50] = np.nan
    assert_fully_unavailable(computer().compute(view_at(flat_tape(buy=buy))))

    # Outside the 129-row read window the same NaN is not read at all, which is
    # the warmup declaration doing its job rather than luck.
    outside = np.full(160, 60.0)
    outside[0] = np.nan
    clean = computer().compute(view_at(flat_tape(buy=outside)))
    assert clean.values["order_flow_available"] == 1.0


def test_an_absent_newest_tick_row_is_unavailable_not_an_extrapolation():
    """The tick and bar feeds have independent cadences, so the newest visible
    tick aggregate can close at a different instant from the newest bar. The
    absorption term would then compare a delta window against prices from a
    different stretch of tape, which is a silent misalignment and not a small
    error."""
    n = 160
    bars = bars_from_mids(np.full(n, 100.0))
    lagging = tick_series(n, ts_ns=stamps(n) + STEP_NS // 2)
    view = view_at(symbol_data(bars, lagging))
    assert view.bar_count() == n and view.tick_count() == n - 1
    note = assert_fully_unavailable(computer().compute(view))
    assert "but the newest bar closes at" in note


def test_the_bundle_reports_the_absent_tick_feed_rather_than_running():
    """`FeatureBundle` is the other caller, and it must not run the computer
    against empty arrays. Both the bundle's own feed check and the computer's
    internal one have to hold, because the lookahead audit calls the computer
    directly and the signal engine goes through the bundle."""
    bars = bars_from_mids(np.full(200, 100.0))
    merged = FeatureBundle([computer()]).compute(view_at(symbol_data(bars)))
    assert not merged.warmup_complete
    assert set(merged.values) == set(KEYS)
    assert set(merged.values.values()) == {0.0}
    assert any("tick_aggregate" in note for note in merged.notes)

    ready = FeatureBundle([computer()]).compute(view_at(flat_tape(max_trade=np.arange(160.0) + 1)))
    assert ready.warmup_complete
    assert ready.values["order_flow_available"] == 1.0


# ---------------------------------------------------------------------------
# warmup honesty
# ---------------------------------------------------------------------------


def test_not_ready_one_bar_before_warmup_and_ready_at_exactly_warmup():
    """A rolling window that is only half full produces a number that looks like
    a feature and is not one, and the boundary is where an off-by-one lives."""
    comp = computer()
    early = comp.compute(view_at(flat_tape(n=WARMUP - 1)))
    note = assert_fully_unavailable(early)
    assert f"warmup incomplete: {WARMUP - 1} of {WARMUP} bars" in note

    exact = comp.compute(view_at(flat_tape(n=WARMUP)))
    assert exact.warmup_complete
    assert exact.values["order_flow_available"] == 1.0


def test_bar_warmup_is_a_floor_on_the_tick_check_and_never_a_substitute():
    """The two feeds have independent cadences. A dataset with 400 bars and 129
    tick rows satisfies the bar check and has one row too few for the trade-size
    chain, and computing from it would mean reading a shorter window than the
    computer declares."""
    bars = bars_from_mids(np.full(400, 100.0))
    short = tick_series(TICK_ROWS, offset=400 - TICK_ROWS)
    note = assert_fully_unavailable(computer().compute(view_at(symbol_data(bars, short))))
    assert f"{TICK_ROWS} of {WARMUP} tick observations" in note
    assert "own cadence" in note


def test_nothing_before_the_declared_warmup_reaches_the_output():
    """HIGH SEVERITY IF THIS FAILS. The regime detector declared 268 bars and
    read 291, which made the label at bar *t* depend on where the caller started
    loading -- and the lookahead audit passed it, because nothing read the
    future.

    The same bar is computed twice: once from 400 rows, once from exactly the
    last 130 of them re-stamped to the same instants. The 270 rows that differ
    are wildly different (mids near 500 against 100, volumes and prints two
    orders of magnitude apart). Every one of the fifteen values must match bit
    for bit. The tail is deliberately NON-degenerate -- random mids, volumes,
    counts and print sizes -- because on a constant tail several terms collapse
    to a stated placeholder and would match whatever the prefix was."""
    rng = np.random.default_rng(11)
    head, tail = 270, WARMUP
    tail_mids = 100.0 + np.cumsum(rng.uniform(-0.4, 0.4, tail))
    tail_cols = dict(
        buy=rng.uniform(40, 160, tail),
        sell=rng.uniform(40, 160, tail),
        buy_trades=np.floor(rng.uniform(4, 26, tail)),
        sell_trades=np.floor(rng.uniform(4, 26, tail)),
        max_trade=np.floor(rng.uniform(1, 40, tail)),
    )
    head_mids = 500.0 + np.cumsum(rng.uniform(-0.9, 0.9, head))

    long_data = tape(
        np.concatenate([head_mids, tail_mids]),
        buy=np.concatenate([rng.uniform(0, 9000, head), tail_cols["buy"]]),
        sell=np.concatenate([rng.uniform(0, 9000, head), tail_cols["sell"]]),
        buy_trades=np.concatenate([np.floor(rng.uniform(0, 900, head)), tail_cols["buy_trades"]]),
        sell_trades=np.concatenate([np.floor(rng.uniform(0, 900, head)), tail_cols["sell_trades"]]),
        max_trade=np.concatenate([np.floor(rng.uniform(0, 9000, head)), tail_cols["max_trade"]]),
    )
    short_data = tape(tail_mids, offset=head, **tail_cols)

    comp = computer()
    full = comp.compute(view_at(long_data))
    minimal = comp.compute(view_at(short_data))
    assert full.warmup_complete and minimal.warmup_complete
    # The premise: this tail exercises every term, so a match is informative.
    assert minimal.values["order_flow_available_weight"] == 1.0
    assert minimal.values["order_flow_magnitude"] > 0.0
    assert minimal.values["absorption"] == 0.0 or minimal.values["absorption_direction"] != 0.0

    differing = {k: (full.values[k], minimal.values[k]) for k in KEYS if full.values[k] != minimal.values[k]}
    assert not differing, (
        "a value at this bar moved when rows before the declared warmup changed, "
        f"so the computer reads further back than warmup_bars={WARMUP}: {differing}"
    )
    assert full.notes == minimal.notes


# ---------------------------------------------------------------------------
# signed delta
# ---------------------------------------------------------------------------


def test_signed_delta_is_the_ranked_smoothed_delta_mapped_onto_minus_one_to_one():
    """Hand-computed, with the smoothing switched off so the window IS the last
    sixty deltas and every observation is a number written here.

    `delta_smoothing_bars=1` is a legal configuration whose own field
    description calls it "the honest way to measure what the smoothing is
    worth". The window is then 17 rows of -5, 42 rows of +11 and a current
    value of +3, so `percentile_rank` counts 17 + 1 = 18 observations at or
    below the current one and
        signed_delta = 2 * (18 / 60) - 1 = 0.6 - 1 = -0.4
    A positive delta reading NEGATIVE is the term's documented behaviour, not a
    sign error: it says "less buy-pressured than this tape's own recent normal",
    and `signed_delta_contracts` is emitted alongside so the absolute fact (+3)
    is not lost."""
    comp = computer(delta_smoothing_bars=1)
    deltas = np.zeros(160)
    window = np.concatenate([np.full(17, -5.0), np.full(42, 11.0), [3.0]])
    assert window.size == DELTA_LOOKBACK
    assert int(np.count_nonzero(window <= 3.0)) == 18
    deltas[-DELTA_LOOKBACK:] = window

    vector = comp.compute(
        view_at(
            flat_tape(
                buy=100.0 + np.where(deltas > 0, deltas, 0.0),
                sell=100.0 + np.where(deltas < 0, -deltas, 0.0),
            )
        )
    )
    assert vector.values["signed_delta"] == pytest.approx(2 * 18 / 60 - 1)
    assert vector.values["signed_delta"] == pytest.approx(-0.4)
    assert vector.values["signed_delta_contracts"] == pytest.approx(3.0)
    assert vector.quality_of("signed_delta") is DataQuality.GOOD


def test_signed_delta_contracts_is_the_mean_of_the_last_smoothing_bars():
    """The diagnostic beside the bounded term: a natural unit, unbounded, which
    no weighted sum may read. At the default smoothing it is the mean of three
    per-bar deltas, so 10, 20 and 60 average to 30 -- not the newest value and
    not a sum."""
    deltas = np.zeros(160)
    deltas[-3:] = [10.0, 20.0, 60.0]
    vector = computer().compute(
        view_at(
            flat_tape(
                buy=100.0 + np.where(deltas > 0, deltas, 0.0),
                sell=100.0 + np.where(deltas < 0, -deltas, 0.0),
            )
        )
    )
    assert vector.values["signed_delta_contracts"] == pytest.approx((10 + 20 + 60) / 3)


def test_a_ramping_delta_reaches_plus_one_at_its_own_maximum():
    """The top of the range has to be reachable, or the term's upper half is
    decoration. On a strictly rising delta the newest smoothed value is the
    maximum of its own window, so every observation is at or below it and the
    rank is exactly 1.0."""
    deltas = np.arange(160.0)
    vector = computer().compute(
        view_at(flat_tape(buy=200.0 + deltas, sell=np.full(160, 200.0)))
    )
    assert vector.values["signed_delta"] == 1.0


def test_a_dead_delta_window_is_reported_balanced_not_maximal():
    """`percentile_rank` counts ties as "at or below", so a window in which every
    smoothed delta is identical would rank 1.0 and report a dead or perfectly
    balanced tape as MAXIMUM buy pressure -- the inverse of the truth, and the
    same tie inflation `features/liquidity.py` documents for quoted spreads."""
    vector = computer().compute(view_at(flat_tape(buy=50.0, sell=50.0)))
    assert vector.values["signed_delta"] == 0.0
    assert vector.values["signed_delta_contracts"] == 0.0
    note = note_with(vector, "smoothed delta observations")
    assert "reported as 0.0 (balanced)" in note
    assert "+1.0" in note      # names the wrong answer it avoided

    # A constant but one-sided tape is the same case: the delta is +20 every
    # bar, there is no distribution to be extreme of, and the term says so.
    constant = computer().compute(view_at(flat_tape(buy=60.0, sell=40.0)))
    assert constant.values["signed_delta"] == 0.0
    assert constant.values["signed_delta_contracts"] == pytest.approx(20.0)
    # Still AVAILABLE: a balanced tape is a measurement, so its weight stays in.
    assert constant.quality_of("signed_delta") is DataQuality.GOOD


def test_signed_delta_is_not_exactly_antisymmetric_here_reported():
    """A DEFECT THIS FILE REPORTS RATHER THAN REPAIRS, pinned so it cannot
    change unnoticed.

    `base.percentile_rank` counts ties as "at or below", so the maximum of a
    window ranks 1.0 while its minimum ranks `k / P` for `k` tied observations
    and never 0. On two mirror-image tapes -- identical but for the side the
    aggression is on -- the buy tape reads +1.0 and the sell tape reads
    -0.7333, because eight of the sixty smoothed observations tie at the
    minimum. The attainable range is `[2/P - 1, +1] = [-0.9667, +1.0]`, not
    `[-1, +1]`.

    Not repaired here: `percentile_rank` is `features/base.py`'s shared
    definition and every percentile feature in the system is ranked with it, so
    changing the tie rule is a package-wide decision and a Phase 8 question, not
    a local fix. Reported because `orderflow.py` documents the tie inflation for
    a DEGENERATE window and not for this, and a reader would reasonably assume a
    symmetric term."""
    comp = computer()
    heavy = np.full(160, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0

    bullish = comp.compute(
        view_at(flat_tape(buy=100.0 + heavy, sell=np.full(160, 100.0)))
    ).values["signed_delta"]
    bearish = comp.compute(
        view_at(flat_tape(buy=np.full(160, 100.0), sell=100.0 + heavy))
    ).values["signed_delta"]

    assert bullish == 1.0
    # 8 of the 60 smoothed observations are windows wholly inside the 10-bar
    # heavy block, so 8 tie at the minimum: 2 * 8/60 - 1 = -0.73333...
    assert bearish == pytest.approx(2 * 8 / 60 - 1)
    assert bearish == pytest.approx(-0.7333333, abs=1e-6)
    assert abs(bullish) != pytest.approx(abs(bearish))
    # The floor of the range, for the record.
    assert signed_rank(np.arange(float(DELTA_LOOKBACK)), 0.0) == pytest.approx(
        2 / DELTA_LOOKBACK - 1
    )


# ---------------------------------------------------------------------------
# CVD slope
# ---------------------------------------------------------------------------


def test_cvd_slope_is_the_volume_normalized_ols_slope_squashed():
    """Hand-computed end to end. With 60 bought and 40 sold every bar the delta
    is +20 and the CVD path over 20 bars is 20, 40, ... 400 -- a straight line
    of slope 20 contracts per bar. The window's mean classified volume is
    60 + 40 = 100, so

        cvd_slope_normalized = 20 / 100 = 0.2
        cvd_slope            = tanh(0.2 / 0.25) = tanh(0.8) = 0.664036770...

    The division is what makes the term dimensionless and comparable across NQ,
    GC and QQQ; the squash scale only sets how fast it saturates."""
    vector = computer().compute(view_at(flat_tape(buy=60.0, sell=40.0)))
    assert vector.values["cvd_slope_normalized"] == pytest.approx(0.2)
    assert vector.values["cvd_slope"] == pytest.approx(math.tanh(0.8))
    assert vector.values["cvd_slope"] == pytest.approx(0.6640367702678490)
    assert vector.quality_of("cvd_slope") is DataQuality.GOOD

    # The scale is the value that maps to about 0.76: a CVD climbing a quarter
    # of a typical bar's classified volume per bar.
    at_scale = computer().compute(view_at(flat_tape(buy=62.5, sell=37.5)))
    assert at_scale.values["cvd_slope_normalized"] == pytest.approx(CVD_SQUASH)
    assert at_scale.values["cvd_slope"] == pytest.approx(math.tanh(1.0))

    # Sign is carried, not discarded: `signed_squash`, not `squash`.
    mirrored = computer().compute(view_at(flat_tape(buy=40.0, sell=60.0)))
    assert mirrored.values["cvd_slope"] == pytest.approx(-math.tanh(0.8))
    assert mirrored.values["cvd_slope_normalized"] == pytest.approx(-0.2)


def test_a_flat_cvd_path_is_zero_slope_not_a_tie_inflated_rank():
    """Unlike `signed_delta` this term has a meaningful absolute centre, so a
    balanced tape is 0.0 by arithmetic rather than by a special case."""
    vector = computer().compute(view_at(flat_tape(buy=50.0, sell=50.0)))
    assert vector.values["cvd_slope_normalized"] == 0.0
    assert vector.values["cvd_slope"] == 0.0
    assert vector.quality_of("cvd_slope") is DataQuality.GOOD


def test_cvd_slope_is_dropped_when_its_window_has_no_classified_volume():
    """The slope is in contracts per bar, which means something different on
    every instrument; without a volume to divide by there is no scale to express
    it in, and reporting the raw slope would be a number nobody can compare. So
    the term is dropped and its 0.25 is removed from the attainable magnitude
    rather than reallocated."""
    buy = np.full(160, 60.0)
    sell = np.full(160, 40.0)
    buy[-CVD_LOOKBACK:] = 0.0
    sell[-CVD_LOOKBACK:] = 0.0
    vector = computer().compute(
        view_at(flat_tape(buy=buy, sell=sell, max_trade=np.arange(160.0) + 1.0))
    )
    assert vector.values["order_flow_available"] == 1.0
    assert vector.values["cvd_slope"] == 0.0
    assert vector.values["cvd_slope_normalized"] == 0.0
    assert vector.quality_of("cvd_slope") is DataQuality.DEGRADED
    assert vector.quality_of("cvd_slope_normalized") is DataQuality.DEGRADED
    note = note_with(vector, "CVD window")
    assert "DROPPED" in note and "not redistributed" in note
    # 1.00 - 0.25 (cvd) - 0.20 (aggression: the newest pool has no volume but
    # does have counts, so this one survives) ... check the weight arithmetic
    # rather than assuming which terms fell.
    assert vector.values["order_flow_available_weight"] < 1.0
    assert vector.values["order_flow_available_weight"] <= 1.0 - W_CVD


# ---------------------------------------------------------------------------
# aggression ratio
# ---------------------------------------------------------------------------


def test_aggression_ratio_is_the_pooled_buy_share_of_trade_counts():
    """Hand-computed: six buy prints and four sell prints a bar, pooled over ten
    bars, is 60 / (60 + 40) = 0.6. Built from COUNTS and not volume, which is a
    different measurement from `TickAggregate.delta_ratio`: many small lifts and
    one large one give the same delta and a very different count ratio."""
    vector = computer().compute(view_at(flat_tape(buy_trades=6.0, sell_trades=4.0)))
    assert vector.values["aggression_ratio"] == pytest.approx(0.6)
    assert vector.quality_of("aggression_ratio") is DataQuality.GOOD

    # It is a COUNT ratio: hold the counts and move the volume, and it does not
    # budge. A volume-weighted stand-in would.
    heavy_buys = computer().compute(
        view_at(flat_tape(buy=900.0, sell=10.0, buy_trades=6.0, sell_trades=4.0))
    )
    assert heavy_buys.values["aggression_ratio"] == pytest.approx(0.6)

    one_sided = computer().compute(view_at(flat_tape(buy_trades=10.0, sell_trades=0.0)))
    assert one_sided.values["aggression_ratio"] == 1.0


def test_a_thin_trade_pool_drops_the_term_rather_than_neutralising_it():
    """A 3-trade 2:1 ratio emitting 0.67 is indistinguishable downstream from a
    real two-thirds reading, so the term is dropped. The boundary is exact: 2.0
    prints a bar pooled over ten bars is 20, the configured minimum, and 1.9 is
    19."""
    comp = computer()
    thin = comp.compute(view_at(flat_tape(buy_trades=1.9 * 0.6, sell_trades=1.9 * 0.4)))
    assert thin.values["aggression_ratio"] == NEUTRAL_RATIO
    assert thin.quality_of("aggression_ratio") is DataQuality.DEGRADED
    note = note_with(thin, "classified trades pooled over")
    assert "19 classified trades" in note
    assert "DROPPED, not neutralised" in note
    assert "not redistributed" in note
    assert thin.values["order_flow_available_weight"] == pytest.approx(1.0 - W_AGGRESSION - W_SIZE)

    at_minimum = comp.compute(view_at(flat_tape(buy_trades=1.2, sell_trades=0.8)))
    assert at_minimum.values["aggression_ratio"] == pytest.approx(0.6)
    assert at_minimum.quality_of("aggression_ratio") is DataQuality.GOOD


def test_a_balanced_pool_and_a_dropped_one_are_separated_only_by_quality():
    """0.5 is both a genuinely balanced buy share and the placeholder for a
    dropped term, which is unavoidable -- the key must hold a finite float. The
    key's own `DataQuality` is what separates them, which is why
    `FeatureVector` carries a status per key rather than one per vector. If this
    ever stopped holding, a caller could not tell "the tape is balanced" from
    "this was never measured"."""
    balanced = computer().compute(view_at(flat_tape(buy_trades=5.0, sell_trades=5.0)))
    dropped = computer().compute(view_at(flat_tape(buy_trades=None, sell_trades=None)))
    assert balanced.values["aggression_ratio"] == NEUTRAL_RATIO
    assert dropped.values["aggression_ratio"] == NEUTRAL_RATIO
    assert balanced.values["aggression_ratio"] == dropped.values["aggression_ratio"]
    assert balanced.quality_of("aggression_ratio") is DataQuality.GOOD
    assert dropped.quality_of("aggression_ratio") is DataQuality.DEGRADED
    assert has_note(dropped, "no buy_trades/sell_trades")
    # Neither tape carries `max_trade_size`, so both carry a note about the
    # trade-size term; what distinguishes them is that no note on the balanced
    # tape says anything about the aggression term.
    assert not has_note(balanced, "aggression")
    assert not has_note(balanced, "classified trades pooled")


def test_a_feed_with_no_trade_counts_never_derives_them_from_volume():
    """`TickAggregate.delta_ratio` would give a volume-weighted number in the
    same range, and substituting it would silently replace a count ratio with a
    different measurement. The note says so by name."""
    vector = computer().compute(
        view_at(flat_tape(buy=900.0, sell=10.0, buy_trades=None, sell_trades=None))
    )
    assert vector.values["aggression_ratio"] == NEUTRAL_RATIO
    note = note_with(vector, "no buy_trades/sell_trades")
    assert "delta_ratio" in note
    assert "different measurement" in note
    # A 90:1 volume imbalance moved it not at all.
    balanced_volume = computer().compute(
        view_at(flat_tape(buy=50.0, sell=50.0, buy_trades=None, sell_trades=None))
    )
    assert balanced_volume.values["aggression_ratio"] == vector.values["aggression_ratio"]


def test_a_broken_trade_count_column_is_treated_as_not_supplied_and_said_so():
    """`TickSeries.NON_NEGATIVE` covers the volume columns and not the counts, so
    a negative count reaches the computer. "The feed carries no counts" would be
    the wrong explanation for a column that exists and is broken, so both notes
    are emitted."""
    vector = computer().compute(
        view_at(flat_tape(buy_trades=-1.0, max_trade=np.arange(160.0) + 1.0))
    )
    assert vector.values["order_flow_available"] == 1.0
    assert vector.values["aggression_ratio"] == NEUTRAL_RATIO
    assert vector.quality_of("aggression_ratio") is DataQuality.DEGRADED
    present_but_broken = note_with(vector, "columns are present but")
    assert "NOT SUPPLIED" in present_but_broken
    assert has_note(vector, "no buy_trades/sell_trades columns")
    # The tail moment still works, so the size term survives on one moment.
    assert vector.quality_of("max_trade_size_percentile") is DataQuality.GOOD
    assert vector.values["order_flow_available_weight"] == pytest.approx(1.0 - W_AGGRESSION)


# ---------------------------------------------------------------------------
# absorption at a level
# ---------------------------------------------------------------------------


def test_absorption_is_one_when_heavy_one_sided_flow_does_not_move_price():
    """The top of the term's range has to be reachable on real geometry, or the
    geometric mean is decoration.

    The tape: every close is 100.0, so the displacement over the window is 0.0
    (`quiet = 1 - 0/2.0/0.25 = 1`) and the ten-close excursion is 0.0
    (`flat = 1`). The delta is +1 a bar for a hundred bars and then +100 for ten,
    so the newest 10-bar window sum is 1000, the maximum of its own sixty, which
    ranks 1.0 and gives `effort = (1.0 - 0.80) / 0.20 = 1.0`. Then

        absorption = (1 * 1 * 1) ** (1/3) = 1.0
    """
    heavy = np.full(160, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    view = view_at(flat_tape(buy=100.0 + heavy, sell=100.0, max_trade=np.arange(160.0) + 1.0))
    assert local_atr(view) == 2.0
    vector = computer().compute(view)
    assert vector.values["absorption"] == 1.0
    assert vector.quality_of("absorption") is DataQuality.GOOD
    assert ABSORPTION_TERM_COUNT == 3


def test_absorption_votes_against_the_aggressor():
    """A DECLARED PRIOR, not a fact: the reading is that the passive side is in
    control when aggression fails to move price. It is tested because the sign
    has to be the sign the module says it is -- absorbed buying bearish,
    absorbed selling bullish -- and because this is the one term whose vote
    opposes `signed_delta` on the same bar."""
    heavy = np.full(160, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0

    absorbed_buying = computer().compute(
        view_at(flat_tape(buy=100.0 + heavy, sell=100.0))
    )
    assert absorbed_buying.values["absorption"] == 1.0
    assert absorbed_buying.values["absorption_direction"] == -1.0

    absorbed_selling = computer().compute(
        view_at(flat_tape(buy=100.0, sell=100.0 + heavy))
    )
    assert absorbed_selling.values["absorption"] == 1.0
    assert absorbed_selling.values["absorption_direction"] == 1.0

    # No strength, no vote: the direction is not a leftover sign.
    quiet_tape = computer().compute(view_at(flat_tape(buy=50.0, sell=50.0)))
    assert quiet_tape.values["absorption"] == 0.0
    assert quiet_tape.values["absorption_direction"] == 0.0


def test_displacement_turns_absorption_off_because_a_drive_is_not_absorption():
    """Heavy delta WITH displacement is a drive, and scoring the two the same
    way would make the term unreadable. The geometric mean is an AND: a zero on
    any factor sends the result to zero.

    The ticks are bit-identical to the absorbed case above; only the closes
    move, by 0.9 a bar over the window. Against an ATR of 2.0 that is a 9.0-point
    displacement, 4.5 ATR, far past the 0.25 ATR band, so `quiet = 0` and the
    whole term is 0 however heavy the flow was."""
    heavy = np.full(160, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    mids = np.full(160, 100.0)
    mids[-(ABSORPTION_LOOKBACK + 1):] = 100.0 + np.arange(ABSORPTION_LOOKBACK + 1) * 0.9

    view = view_at(tape(mids, buy=100.0 + heavy, sell=100.0))
    assert local_atr(view) == 2.0
    closes = view.closes(ABSORPTION_LOOKBACK + 1)
    assert abs(float(closes[-1]) - float(closes[0])) == pytest.approx(9.0)
    assert 9.0 / 2.0 > MAX_DISPLACEMENT_ATR

    vector = computer().compute(view)
    assert vector.values["absorption"] == 0.0
    assert vector.values["absorption_direction"] == 0.0
    assert vector.quality_of("absorption") is DataQuality.GOOD   # measured, not dropped


def test_the_three_absorption_factors_are_exact_on_a_one_bar_window():
    """The geometric mean's arithmetic, pinned factor by factor.

    `absorption_lookback_bars=1` makes the window sum the bar's own delta and
    the excursion band a single close, so `flat == 1` by construction and the
    remaining two factors are numbers written here. The window is then the last
    sixty deltas directly:

      * 48 of 60 at or below the current value is a rank of exactly 0.80, the
        configured threshold, and `excess_above` is 0 AT the threshold, so
        `effort = 0` and the whole term is 0.
      * 49 of 60 is 0.816666..., so `effort = (0.8166667 - 0.8) / 0.2 = 1/12`.
      * the last close moves 0.25 points against an ATR of 2.0, which is
        0.125 ATR, half the 0.25 ATR band, so `quiet = 1 - 0.5 = 0.5`.
      * `absorption = (1/12 * 0.5 * 1) ** (1/3) = 0.3466806...`
    """
    comp = computer(absorption_lookback_bars=1)
    assert comp.warmup_bars == WARMUP      # the config floor still binds

    for at_or_below, expected_effort in ((48, 0.0), (49, 1 / 12)):
        window = np.empty(DELTA_LOOKBACK)
        window[: at_or_below - 1] = 1.0        # strictly below the current value
        window[at_or_below - 1 :] = 9.0        # strictly above it
        window[-1] = 5.0                       # the current value
        assert int(np.count_nonzero(window <= 5.0)) == at_or_below

        deltas = np.full(160, 7.0)
        deltas[-DELTA_LOOKBACK:] = window
        mids = np.full(160, 100.0)
        mids[-1] = 100.25                      # |d close| = 0.25 -> quiet = 0.5

        view = view_at(
            tape(
                mids,
                buy=100.0 + np.where(deltas > 0, deltas, 0.0),
                sell=100.0 + np.where(deltas < 0, -deltas, 0.0),
            )
        )
        assert local_atr(view) == 2.0
        expected = (expected_effort * 0.5 * 1.0) ** (1 / ABSORPTION_TERM_COUNT)
        vector = comp.compute(view)
        assert vector.values["absorption"] == pytest.approx(expected)
        assert vector.values["absorption_direction"] == (0.0 if expected == 0.0 else -1.0)

    assert excess_above(ABSORPTION_PERCENTILE, ABSORPTION_PERCENTILE) == 0.0
    assert excess_above(1.0, ABSORPTION_PERCENTILE) == 1.0
    assert inside_band(MAX_DISPLACEMENT_ATR, MAX_DISPLACEMENT_ATR) == 0.0
    assert inside_band(0.0, MAX_DISPLACEMENT_ATR) == 1.0
    assert inside_band(MAX_DISPLACEMENT_ATR / 2, MAX_DISPLACEMENT_ATR) == pytest.approx(0.5)


def test_a_perfectly_balanced_window_reports_no_effort_rather_than_maximum():
    """On a tape where every 10-bar delta sums to exactly 0 a tie-inclusive
    percentile rank would return 1.0 and report MAXIMUM one-sided flow on a tape
    with none. Absorption is about heavy one-sided flow; a window with none has
    nothing to absorb."""
    vector = computer().compute(view_at(flat_tape(buy=50.0, sell=50.0)))
    assert vector.values["absorption"] == 0.0
    assert vector.values["absorption_direction"] == 0.0
    assert vector.quality_of("absorption") is DataQuality.GOOD


def test_absorption_is_dropped_when_the_local_atr_has_no_scale():
    """The one input the term can be missing. A frozen tape -- every bar a single
    price -- has no volatility scale, so there is no ATR-relative geometry, and
    dividing by zero would make the band infinite and the term 1.0 on a tape
    where nothing happened."""
    heavy = np.full(160, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    view = view_at(
        tape(np.full(160, 100.0), flat=True, buy=100.0 + heavy, sell=100.0)
    )
    assert local_atr(view) == 0.0
    vector = computer().compute(view)
    assert vector.values["order_flow_available"] == 1.0
    assert vector.values["absorption"] == 0.0
    assert vector.values["absorption_direction"] == 0.0
    assert vector.quality_of("absorption") is DataQuality.DEGRADED
    assert vector.quality_of("absorption_direction") is DataQuality.DEGRADED
    note = note_with(vector, "local ATR")
    assert "DROPPED" in note and "not redistributed" in note
    # Only the absorption weight leaves. The trade-size term survives on its
    # CENTRAL moment: the classified volume per bar moves with the delta while
    # the print counts stay at ten, so the pooled average is a real
    # distribution even though this feed carries no `max_trade_size`.
    assert vector.quality_of("avg_trade_size_percentile") is DataQuality.GOOD
    assert vector.values["order_flow_available_weight"] == pytest.approx(
        1.0 - W_ABSORPTION
    )


def test_absorption_reads_no_zone_list_so_it_cannot_see_a_structure_level():
    """"At a level" is measured as local flatness, not by reading
    `features/levels.py`'s zones. The reason is auditability: a computer that
    consumed another computer's output could not be audited for lookahead in
    isolation, which is how every other computer in this package is audited, and
    it would couple a 130-bar warmup to the 514-bar level stack.

    The consequence is testable and worth stating: two tapes with identical
    local geometry report identical absorption even though one sits on a level
    that has been touched six times and the other on untouched air. The term
    reports that aggression was absorbed somewhere FLAT, which is a weaker claim
    than "at a level", and `StructureGate` still gates on real zones."""
    heavy = np.full(200, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    comp = computer()

    # A tape that visited 100.0 repeatedly before settling there.
    touched = np.full(200, 100.0)
    touched[20:40] = np.resize([100.0, 99.0, 100.0, 101.0], 20)
    touched[60:80] = np.resize([100.0, 98.5, 100.0, 101.5], 20)
    # A tape that arrived at 100.0 from a long way off and never saw it before.
    untouched = np.concatenate([np.linspace(60.0, 100.0, 180), np.full(20, 100.0)])
    assert float(np.abs(np.diff(untouched)).max()) <= 1.0

    first = comp.compute(view_at(tape(touched, buy=100.0 + heavy, sell=100.0)))
    second = comp.compute(view_at(tape(untouched, buy=100.0 + heavy, sell=100.0)))
    assert first.values["absorption"] == second.values["absorption"] == 1.0
    assert first.values["absorption_direction"] == second.values["absorption_direction"]
    # And the warmup is the order-flow chain, not the level stack's.
    assert comp.warmup_bars == WARMUP


# ---------------------------------------------------------------------------
# the trade-size distribution
# ---------------------------------------------------------------------------


def test_the_large_trade_event_boundary_is_exact():
    """Hand-computed with the pool set to one bar, so the 120 ranked
    observations ARE the last 120 `max_trade_size` values and every one of them
    is a number written here.

    107 values at or below the current one is 107/120 = 0.891666..., below the
    0.90 threshold; 108 is exactly 0.90 and the event fires, because the
    threshold is inclusive ("at or above")."""
    comp = computer(aggression_lookback_bars=1)
    assert comp.warmup_bars == SIZE_LOOKBACK + 1 - 1 + 1 == 121

    for at_or_below, fires, rank in ((108, 1.0, 0.90), (107, 0.0, 107 / 120)):
        pooled = np.empty(SIZE_LOOKBACK)
        pooled[: at_or_below - 1] = 1.0
        pooled[at_or_below - 1 :] = 9.0
        pooled[-1] = 5.0
        assert int(np.count_nonzero(pooled <= 5.0)) == at_or_below
        sizes = np.full(140, 1.0)
        sizes[-SIZE_LOOKBACK:] = pooled

        vector = comp.compute(
            view_at(flat_tape(n=140, buy_trades=12.0, sell_trades=8.0, max_trade=sizes))
        )
        assert vector.values["max_trade_size_percentile"] == pytest.approx(rank)
        assert vector.values["large_trade_event"] == fires
        assert vector.quality_of("large_trade_event") is DataQuality.GOOD
        # The pool is still 1 bar, so min_classified_trades needs 20 in it.
        assert vector.values["aggression_ratio"] == pytest.approx(0.6)


def test_the_tail_moment_is_the_rank_of_the_pooled_maximum_print():
    """Pooled over ten bars and ranked within 120 pooled observations, so the
    newest reading is the largest print of the last ten bars compared with the
    largest print of each preceding ten-bar block. On a strictly rising print
    size the newest pooled maximum is the maximum of its own history."""
    vector = computer().compute(
        view_at(flat_tape(max_trade=np.arange(160.0) + 1.0))
    )
    assert vector.values["max_trade_size_percentile"] == 1.0
    assert vector.values["large_trade_event"] == 1.0
    assert rolling_max(np.arange(10.0), 3).tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_the_central_moment_is_pooled_volume_over_pooled_trade_count():
    """Hand-computed, including the pooling arithmetic, which is where a window
    off-by-one would hide.

    Every bar carries 6 + 4 = 10 prints. The last fifteen rows carry
    60 + 40 = 100 lots and the rest carry 120 + 80 = 200, so a ten-bar pool
    wholly inside the light tail averages 1000 / 100 = 10.0 contracts a print
    and one wholly outside it averages 20.0. A pool straddling the boundary is
    strictly between the two.

    A fifteen-row tail admits `15 - 10 + 1 = 6` pools wholly inside it, and
    those six are the only observations at or below the newest one, so

        avg_trade_size_percentile = 6 / 120 = 0.05

    The average is pooled-sum over pooled-sum and not a mean of per-bar
    averages, which differ whenever the bars carry different counts."""
    tail = POOL + 5
    buy = np.full(160, 120.0)
    sell = np.full(160, 80.0)
    buy[-tail:] = 60.0
    sell[-tail:] = 40.0
    vector = computer().compute(
        view_at(flat_tape(buy=buy, sell=sell, buy_trades=6.0, sell_trades=4.0))
    )
    assert tail - POOL + 1 == 6
    assert vector.values["avg_trade_size_percentile"] == pytest.approx(6 / 120)
    assert vector.values["avg_trade_size_percentile"] == pytest.approx(0.05)
    assert vector.quality_of("avg_trade_size_percentile") is DataQuality.GOOD
    assert rolling_sum(np.arange(5.0), 2).tolist() == [1.0, 3.0, 5.0, 7.0]
    assert rolling_mean(np.arange(5.0), 2).tolist() == [0.5, 1.5, 2.5, 3.5]


def test_the_size_strength_is_zero_for_an_ordinary_distribution():
    """`clamp01(2 * mean(ranks) - 1)`: an ordinary or small trade-size
    distribution contributes nothing and only an unusually heavy one
    contributes. The term casts no directional vote at all, because
    `TickAggregate` reports `max_trade_size` without a side -- inferring the
    side from the bar's delta would be exactly the invention this module
    refuses.

    The two tapes below have identical print sizes and opposite deltas. The size
    contribution to the magnitude is the same in both, so the term's effect can
    be read off the difference from a third tape with small prints."""
    sizes = np.full(160, 1.0)
    sizes[-POOL:] = 500.0
    comp = computer()

    # An unusually heavy tail with no directional terms at all: the magnitude is
    # the size weight times the strength and nothing else.
    heavy = comp.compute(
        view_at(flat_tape(buy=50.0, sell=50.0, buy_trades=10.0, sell_trades=10.0, max_trade=sizes))
    )
    assert heavy.values["signed_delta"] == 0.0
    assert heavy.values["cvd_slope"] == 0.0
    assert heavy.values["aggression_ratio"] == pytest.approx(0.5)
    assert heavy.values["absorption"] == 0.0
    assert heavy.values["max_trade_size_percentile"] == 1.0
    # avg is degenerate on a constant-volume tape, so the strength is the tail
    # rank alone: clamp01(2 * 1.0 - 1) = 1.0, contributing exactly W_SIZE.
    assert heavy.values["order_flow_magnitude"] == pytest.approx(W_SIZE)

    # An ORDINARY distribution contributes nothing, which is the shape's whole
    # point. With the pool set to one bar the 120 ranked observations are the
    # last 120 print sizes directly: 59 below the current value and 60 above it
    # puts the current one at a rank of exactly 60/120 = 0.5, so
    # `clamp01(2 * 0.5 - 1) = 0.0` and the magnitude is 0.0 rather than
    # "something small".
    single = computer(aggression_lookback_bars=1)
    pooled = np.empty(SIZE_LOOKBACK)
    pooled[:59] = 1.0
    pooled[59:] = 9.0
    pooled[-1] = 5.0
    assert int(np.count_nonzero(pooled <= 5.0)) == 60
    sizes_ordinary = np.full(140, 1.0)
    sizes_ordinary[-SIZE_LOOKBACK:] = pooled
    ordinary = single.compute(
        view_at(
            flat_tape(
                n=140,
                buy=50.0,
                sell=50.0,
                buy_trades=10.0,
                sell_trades=10.0,
                max_trade=sizes_ordinary,
            )
        )
    )
    assert ordinary.values["max_trade_size_percentile"] == pytest.approx(0.5)
    assert ordinary.values["large_trade_event"] == 0.0
    assert ordinary.values["order_flow_available_weight"] == 1.0
    assert ordinary.values["order_flow_magnitude"] == 0.0
    assert ordinary.values["order_flow_direction"] == 0.0

    # And a tape with one print size in every pool is unavailable rather than
    # ranked: `rolling_max` over ten bars that all contain a 9.0 gives 9.0
    # everywhere, including the pool holding the single 1.0, so the 120 pooled
    # maxima are identical and the moment reports no measurement at all.
    flattened = comp.compute(
        view_at(
            flat_tape(
                buy=50.0,
                sell=50.0,
                buy_trades=10.0,
                sell_trades=10.0,
                max_trade=np.concatenate([np.full(159, 9.0), [1.0]]),
            )
        )
    )
    assert flattened.values["max_trade_size_percentile"] == UNAVAILABLE_PERCENTILE
    assert flattened.quality_of("max_trade_size_percentile") is DataQuality.DEGRADED
    assert flattened.values["order_flow_available_weight"] == pytest.approx(1.0 - W_SIZE)


def test_one_print_size_everywhere_is_unavailable_not_a_rank_of_one():
    """A tie-inclusive rank of 1.0 on a tape with a single print size would fire
    `large_trade_event` on every bar. The rank is reported as unavailable
    instead, and the flag degrades with it: 0.0 from an unmeasured tail means
    "not measured", not "no large print"."""
    vector = computer().compute(view_at(flat_tape(max_trade=5.0)))
    assert vector.values["max_trade_size_percentile"] == UNAVAILABLE_PERCENTILE
    assert vector.values["large_trade_event"] == 0.0
    assert vector.quality_of("max_trade_size_percentile") is DataQuality.DEGRADED
    assert vector.quality_of("large_trade_event") is DataQuality.DEGRADED
    note = note_with(vector, "pooled maximum")
    assert "unavailable" in note
    assert "one print size" in note
    assert is_degenerate(np.full(5, 5.0))
    assert not is_degenerate(np.array([5.0, 5.0, 6.0]))


def test_the_two_moments_fail_separately_and_are_graded_separately():
    """They fail for different reasons -- a feed can carry `max_trade_size` and
    no counts, or a tape can have one print size and a perfectly good average --
    so grading both by a count would report a measured percentile as DEGRADED
    because the other one was not measured. The term as a whole survives on
    either moment, because "the distribution is unusually heavy" is answerable
    from one."""
    tail_only = computer().compute(
        view_at(
            flat_tape(
                buy_trades=None, sell_trades=None, max_trade=np.arange(160.0) + 1.0
            )
        )
    )
    assert tail_only.quality_of("max_trade_size_percentile") is DataQuality.GOOD
    assert tail_only.quality_of("avg_trade_size_percentile") is DataQuality.DEGRADED
    assert tail_only.values["avg_trade_size_percentile"] == UNAVAILABLE_PERCENTILE
    # The size term is in, the aggression term is out: 1.00 - 0.20 = 0.80.
    assert tail_only.values["order_flow_available_weight"] == pytest.approx(1.0 - W_AGGRESSION)

    buy = np.full(160, 120.0)
    buy[-POOL:] = 60.0
    central_only = computer().compute(
        view_at(flat_tape(buy=buy, sell=80.0, max_trade=5.0))
    )
    assert central_only.quality_of("avg_trade_size_percentile") is DataQuality.GOOD
    assert central_only.quality_of("max_trade_size_percentile") is DataQuality.DEGRADED
    assert central_only.values["order_flow_available_weight"] == 1.0


def test_a_feed_carrying_only_the_two_required_columns_drops_two_terms():
    """The minimum honest tick feed: classified buy and sell volume and nothing
    else. `aggression_ratio` (0.20) and the whole trade-size term (0.15) are
    dropped, so 0.65 of the weight has real inputs -- above the 0.50 floor, so
    the component still reports, at 65% of its attainable magnitude. Nothing is
    reallocated to make up the difference."""
    vector = computer().compute(
        view_at(
            flat_tape(
                unclassified=None, buy_trades=None, sell_trades=None, max_trade=None
            )
        )
    )
    assert vector.values["order_flow_available"] == 1.0
    assert vector.values["order_flow_available_weight"] == pytest.approx(
        W_DELTA + W_CVD + W_ABSORPTION
    )
    assert vector.values["order_flow_available_weight"] == pytest.approx(0.65)
    assert vector.values["aggression_ratio"] == NEUTRAL_RATIO
    assert vector.values["avg_trade_size_percentile"] == UNAVAILABLE_PERCENTILE
    assert vector.values["max_trade_size_percentile"] == UNAVAILABLE_PERCENTILE
    for key in ("aggression_ratio", "avg_trade_size_percentile", "max_trade_size_percentile"):
        assert vector.quality_of(key) is DataQuality.DEGRADED
    assert DataQuality.MISSING not in set(vector.quality_by_key.values())
    assert vector.quality is DataQuality.DEGRADED
    assert vector.warmup_complete


def test_a_broken_max_trade_column_is_treated_as_not_supplied_and_said_so():
    """`max_trade_size` is not in `TickSeries.NON_NEGATIVE`, so a negative print
    size reaches the computer. It is absence rather than a measurement of zero,
    and the note distinguishes "present but broken" from "not carried"."""
    vector = computer().compute(view_at(flat_tape(max_trade=-2.0)))
    assert vector.values["order_flow_available"] == 1.0
    assert vector.values["max_trade_size_percentile"] == UNAVAILABLE_PERCENTILE
    assert vector.quality_of("max_trade_size_percentile") is DataQuality.DEGRADED
    note = note_with(vector, "max_trade_size column is present but")
    assert "NOT SUPPLIED" in note
    assert has_note(vector, "no max_trade_size column")


# ---------------------------------------------------------------------------
# composition: magnitude and a separately carried direction
# ---------------------------------------------------------------------------


def test_the_magnitude_is_the_weighted_net_of_every_term_hand_computed():
    """A sum hides a dead term, so all five are pinned at once on one tape.

    The tape: a constant +20 delta a bar on 100 lots of classified volume, 6
    buy prints and 4 sell prints, frozen closes, one print size. Then

      signed_delta       = 0.0            (degenerate window, stated as balanced)
      cvd_slope          = tanh(0.8)      = 0.6640367702678490
      aggression signed  = 2*0.6 - 1      = 0.2
      absorption signed  = 0.0            (every window sum ties, no effort)
      size               dropped          (one print size, constant average)

      dir_sum   = 0.25*0.0 + 0.25*0.6640367702678490 + 0.20*0.2 + 0.15*0.0
                = 0.16600919256696226 + 0.04
                = 0.20600919256696226
      magnitude = |dir_sum| + 0  (size dropped)
      direction = +1
      available_weight = 1.00 - 0.15 = 0.85
    """
    vector = computer().compute(view_at(flat_tape(buy=60.0, sell=40.0, max_trade=5.0)))
    expected_dir_sum = (
        W_DELTA * 0.0
        + W_CVD * math.tanh(0.8)
        + W_AGGRESSION * 0.2
        + W_ABSORPTION * 0.0
    )
    assert expected_dir_sum == pytest.approx(0.20600919256696226)
    assert vector.values["order_flow_magnitude"] == pytest.approx(expected_dir_sum)
    assert vector.values["order_flow_direction"] == 1.0
    assert vector.values["order_flow_available_weight"] == pytest.approx(1.0 - W_SIZE)
    assert vector.quality_of("order_flow_magnitude") is DataQuality.DEGRADED
    assert vector.quality_of("order_flow_direction") is DataQuality.DEGRADED


def test_direction_is_carried_separately_and_the_magnitude_never_carries_it():
    """ARCHITECTURE section 7's requirement, and the bug it exists to prevent: a
    strongly bearish component inflating a bullish score. The magnitude is in
    [0, 1] on both of two mirror-image tapes and the sign lives only in
    `order_flow_direction`."""
    comp = computer()
    heavy = np.full(160, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    sizes = np.arange(160.0) + 1.0

    bullish = comp.compute(view_at(flat_tape(buy=100.0 + heavy, sell=100.0, max_trade=sizes)))
    bearish = comp.compute(view_at(flat_tape(buy=100.0, sell=100.0 + heavy, max_trade=sizes)))

    assert bullish.values["order_flow_direction"] == 1.0
    assert bearish.values["order_flow_direction"] == -1.0
    for vector in (bullish, bearish):
        assert 0.0 <= vector.values["order_flow_magnitude"] <= 1.0
        assert vector.values["order_flow_direction"] in (-1.0, 0.0, 1.0)
    # The bearish tape's magnitude is a magnitude: positive, not negative, and
    # it cannot be added to a bullish score without reading the direction.
    assert bearish.values["order_flow_magnitude"] > 0.0


def test_an_absorption_vote_against_the_delta_cancels_exactly_its_weight():
    """The composition's defining choice: `|weighted net|`, so contradictory
    terms CANCEL and a tape nobody can read reports a low magnitude instead of a
    confident one.

    The two tapes share a bit-identical tick feed -- the same deltas, counts and
    print sizes -- and differ only in the bar closes, which only the absorption
    geometry reads. On the frozen tape `absorption = 1.0` votes -1 against a
    `signed_delta` of +1; on the displaced tape the term measures 0.0. Both
    report `available_weight = 1.0`, so the difference is cancellation and
    provably not a dropped term:

        magnitude(frozen) = magnitude(displaced) - 0.15
    """
    comp = computer()
    heavy = np.full(160, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    tick_kwargs = dict(buy=100.0 + heavy, sell=100.0, max_trade=np.arange(160.0) + 1.0)

    frozen_mids = np.full(160, 100.0)
    moved_mids = np.full(160, 100.0)
    moved_mids[-(ABSORPTION_LOOKBACK + 1):] = 100.0 + np.arange(ABSORPTION_LOOKBACK + 1) * 0.9

    conflicted = comp.compute(view_at(tape(frozen_mids, **tick_kwargs)))
    agreed = comp.compute(view_at(tape(moved_mids, **tick_kwargs)))

    # The premise: every tick-derived key is identical, so only absorption moved.
    for key in (
        "signed_delta",
        "signed_delta_contracts",
        "cvd_slope",
        "cvd_slope_normalized",
        "aggression_ratio",
        "avg_trade_size_percentile",
        "max_trade_size_percentile",
        "large_trade_event",
        "classification_coverage",
    ):
        assert conflicted.values[key] == agreed.values[key], key
    assert conflicted.values["order_flow_available_weight"] == 1.0
    assert agreed.values["order_flow_available_weight"] == 1.0

    assert conflicted.values["absorption"] == 1.0
    assert conflicted.values["absorption_direction"] == -1.0
    assert agreed.values["absorption"] == 0.0
    assert conflicted.values["order_flow_direction"] == agreed.values["order_flow_direction"] == 1.0
    assert agreed.values["order_flow_magnitude"] - conflicted.values[
        "order_flow_magnitude"
    ] == pytest.approx(W_ABSORPTION)
    assert conflicted.values["order_flow_magnitude"] < agreed.values["order_flow_magnitude"]


def test_a_dropped_term_leaves_its_weight_out_rather_than_reallocating_it():
    """Section 14.5's rule for `s_flow`, applied to the whole component: the lost
    weight stays lost, so missing data LOWERS the sub-score instead of being
    reallocated into confidence the tape does not support.

    Two tapes with an identical delta, CVD and absorption geometry; one carries
    trade counts and one does not. A renormalizing implementation would report
    the same magnitude for both, scaled back up by 1 / 0.80. The weight
    arithmetic is asserted rather than a note, because a note is not a number."""
    comp = computer()
    sizes = np.arange(160.0) + 1.0
    with_counts = comp.compute(
        view_at(flat_tape(buy=60.0, sell=40.0, buy_trades=6.0, sell_trades=4.0, max_trade=sizes))
    )
    without_counts = comp.compute(
        view_at(flat_tape(buy=60.0, sell=40.0, buy_trades=None, sell_trades=None, max_trade=sizes))
    )

    assert with_counts.values["order_flow_available_weight"] == 1.0
    assert without_counts.values["order_flow_available_weight"] == pytest.approx(
        1.0 - W_AGGRESSION
    )
    # The aggression term was contributing 0.20 * (2*0.6 - 1) = +0.04.
    contribution = W_AGGRESSION * (2 * 0.6 - 1)
    assert contribution == pytest.approx(0.04)
    assert with_counts.values["order_flow_magnitude"] - without_counts.values[
        "order_flow_magnitude"
    ] == pytest.approx(contribution)

    # What a renormalizing implementation would have produced, written out so
    # the assertion is on the weight and not on a note.
    renormalized = without_counts.values["order_flow_magnitude"] / (1.0 - W_AGGRESSION)
    assert without_counts.values["order_flow_magnitude"] < renormalized
    assert without_counts.values["order_flow_magnitude"] != pytest.approx(renormalized)


def test_the_magnitude_never_exceeds_the_available_weight():
    """The honest ceiling: with terms dropped the attainable magnitude falls to
    `order_flow_available_weight`, and a magnitude above it would mean something
    was reallocated. Checked over 200 pathological tapes rather than one,
    because the ceiling is an invariant of the composition and not of a case."""
    rng = np.random.default_rng(2024)
    comp = computer()
    checked = 0
    for trial in range(200):
        n = 170
        mids = 100.0 + np.cumsum(rng.uniform(-1.0, 1.0, n))
        kwargs = dict(
            buy=rng.uniform(0, 500, n),
            sell=rng.uniform(0, 500, n),
            buy_trades=np.floor(rng.uniform(0, 40, n)) if trial % 3 else None,
            sell_trades=np.floor(rng.uniform(0, 40, n)) if trial % 3 else None,
            max_trade=np.floor(rng.uniform(0, 900, n)) if trial % 4 else None,
            unclassified=rng.uniform(0, 120, n) if trial % 5 else None,
        )
        vector = comp.compute(view_at(tape(mids, **kwargs)))
        if vector.values["order_flow_available"] == 0.0:
            continue
        checked += 1
        magnitude = vector.values["order_flow_magnitude"]
        available = vector.values["order_flow_available_weight"]
        assert magnitude <= available + 1e-12, (trial, magnitude, available)
        assert available >= MIN_AVAILABLE_FRACTION - 1e-12
        if vector.values["order_flow_direction"] != 0.0:
            assert magnitude > 0.0
    assert checked > 100, f"only {checked} tapes reached the available path"


def test_the_size_term_adds_magnitude_with_no_direction_here_reported():
    """A TENSION THIS FILE REPORTS RATHER THAN REPAIRS.

    The composition's stated reason for netting the directional terms is that
    the alternative "would report 'strong order flow, direction uncertain',
    which is a high score for a tape nobody can read". The trade-size term does
    precisely that: it carries no direction and is ADDED to the magnitude, so a
    tape whose four directional terms cancel exactly and whose print
    distribution is saturated reports

        order_flow_magnitude = 0.15,  order_flow_direction = 0.0

    with `available_weight = 1.0` and every term GOOD. The tape below forces it:
    buy volume equals sell volume (delta 0 every bar, so `signed_delta` is a
    stated 0.0, the CVD slope is 0.0 and the absorption window sums to 0), buy
    prints equal sell prints (ratio exactly 0.5, signed 0.0), and one
    unmistakably large print at the end lifts the tail rank to 1.0.

    Not repaired: whether a direction-free magnitude is right is a Phase 8
    composition question, the docstring already names the alternative it
    rejected, and changing it here would be choosing a composition by looking at
    a result. Reported because 0.15 of a bounded score arriving with no
    direction is exactly what section 7 asks a reader to watch for."""
    sizes = np.full(160, 1.0)
    sizes[20] = 50.0
    sizes[60] = 30.0
    sizes[-1] = 99.0
    vector = computer().compute(
        view_at(
            flat_tape(
                buy=50.0, sell=50.0, buy_trades=10.0, sell_trades=10.0, max_trade=sizes
            )
        )
    )
    assert vector.values["signed_delta"] == 0.0
    assert vector.values["cvd_slope"] == 0.0
    assert vector.values["aggression_ratio"] == pytest.approx(0.5)
    assert vector.values["absorption"] == 0.0
    assert vector.values["max_trade_size_percentile"] == 1.0
    assert vector.values["order_flow_available_weight"] == 1.0

    assert vector.values["order_flow_magnitude"] == pytest.approx(W_SIZE)
    assert vector.values["order_flow_magnitude"] == pytest.approx(0.15)
    assert vector.values["order_flow_direction"] == 0.0


# ---------------------------------------------------------------------------
# the output contract
# ---------------------------------------------------------------------------


def test_emitted_keys_and_qualities_match_the_declaration_exactly():
    """`FeatureBundle` checks the output against `keys`, so a key promised and
    not delivered is an error at the source rather than a `None` propagating
    into a score. Both paths are checked: a key missing only on the UNAVAILABLE
    branch would be the one nobody notices."""
    comp = computer()
    for vector in (
        comp.compute(view_at(flat_tape(max_trade=np.arange(160.0) + 1.0))),
        comp.compute(view_at(symbol_data(bars_from_mids(np.full(160, 100.0))))),
        comp.compute(view_at(flat_tape(n=WARMUP - 1))),
    ):
        assert set(vector.values) == set(comp.keys)
        assert set(vector.quality_by_key) == set(comp.keys)
        assert tuple(vector.values) == comp.keys     # order preserved too
        for key in comp.keys:
            assert vector.quality_of(key) is not None
            assert isinstance(vector.values[key], float)


def test_every_value_is_finite_and_bounded_over_a_long_synthetic_run():
    """MECHANICS ONLY -- bounds and finiteness on generated data, and no
    statement about predictive content, because the generator builds its tick
    stream from the same innovations as its bars.

    A non-finite feature propagates into every score that reads it, and an
    unbounded one lets a single extreme observation dominate a weighted sum
    (`features/base.py`). Nine keys are in [0, 1], four in [-1, 1], and the two
    diagnostics are unbounded BY DESIGN, so only their finiteness is checkable
    over a long run."""
    data = synthetic_data()
    comp = computer()
    assert len(data.primary_bars) > 2 * comp.warmup_bars
    seen_available = 0
    for index in range(comp.warmup_bars - 1, len(data.primary_bars), 37):
        vector = comp.compute(view_at(data, index))
        for key, value in vector.values.items():
            assert math.isfinite(value), (index, key, value)
        if vector.values["order_flow_available"] == 0.0:
            continue
        seen_available += 1
        for key in UNIT_KEYS:
            assert 0.0 <= vector.values[key] <= 1.0, (index, key, vector.values[key])
        for key in SIGNED_KEYS:
            assert -1.0 <= vector.values[key] <= 1.0, (index, key, vector.values[key])
        assert vector.values["order_flow_direction"] in (-1.0, 0.0, 1.0)
        assert vector.values["absorption_direction"] in (-1.0, 0.0, 1.0)
        assert vector.values["large_trade_event"] in (0.0, 1.0)
        for key in DIAGNOSTIC_KEYS:
            assert math.isfinite(vector.values[key])
    assert seen_available > 50, f"only {seen_available} bars reached the available path"


def test_bar_columns_the_module_does_not_read_move_nothing():
    """`required_feeds` is a claim about what is read, and the claim here is
    bars for price geometry only -- high, low and close, over fifteen bars. Open
    and volume are not read, so perturbing them must move nothing. If bar volume
    ever did reach a feature here, that is the proxy section 5 forbids, and this
    is the test that would catch it."""
    mids = np.full(160, 100.0)
    tick_kwargs = dict(buy=60.0, sell=40.0, max_trade=np.arange(160.0) + 1.0)
    comp = computer()

    plain = comp.compute(view_at(tape(mids, **tick_kwargs)))
    loud = BarSeries(
        symbol=SYMBOL,
        ts_ns=stamps(160),
        interval_seconds=INTERVAL,
        columns={
            "open": mids + 0.75,            # a different open, same high/low/close
            "high": mids + 1.0,
            "low": mids - 1.0,
            "close": mids.copy(),
            "volume": np.full(160, 987_654.0),
        },
    )
    perturbed = comp.compute(
        view_at(symbol_data(loud, tick_series(160, **tick_kwargs)))
    )
    assert plain.values == perturbed.values
    assert plain.notes == perturbed.notes
    assert plain.quality_by_key == perturbed.quality_by_key

    # And a quote feed, which the module declares it does not read at all.
    with_quotes = comp.compute(
        view_at(symbol_data(bars_from_mids(mids), tick_series(160, **tick_kwargs), quotes=quote_feed(160)))
    )
    assert with_quotes.values == plain.values


def test_output_is_deterministic_both_ways():
    """No clock, no RNG, no mutable state. Twice on the same view, and twice
    from independently constructed computers, because `__init__` stores only
    integers and floats read from configuration -- a computer that cached a
    window between calls would pass the first check and fail the second."""
    data = flat_tape(max_trade=np.arange(160.0) + 1.0)
    view = view_at(data)

    comp = computer()
    once, twice = comp.compute(view), comp.compute(view)
    assert once.values == twice.values
    assert once.notes == twice.notes
    assert once.quality_by_key == twice.quality_by_key

    rebuilt = computer().compute(view_at(data))
    assert rebuilt.values == once.values
    assert rebuilt.notes == once.notes
    assert rebuilt.quality_by_key == once.quality_by_key


def test_features_are_unchanged_by_later_bars():
    """Truncation invariance, stated against this module directly rather than
    only through the harness: the value at bar *i* must not move when the bars
    and tick rows after *i* are removed."""
    data = synthetic_data()
    comp = computer()
    index = comp.warmup_bars + 40
    cutoff = int(data.primary_bars.ts_ns[index])
    full = comp.compute(view_at(data, index)).values

    truncated = SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: data.primary_bars.prefix(data.primary_bars.visible_count(cutoff))},
        ticks={
            INTERVAL: data.ticks[INTERVAL].prefix(
                data.ticks[INTERVAL].visible_count(cutoff)
            )
        },
    )
    assert len(truncated.primary_bars) == index + 1
    assert comp.compute(view_at(truncated, index)).values == full


# ---------------------------------------------------------------------------
# the lookahead audit, and proof that the gate is not vacuous
# ---------------------------------------------------------------------------


def test_lookahead_audit_passes_with_a_factory():
    """`factory=`, not an instance: that is the only form that catches a
    full-sample constant captured in `__init__`, which is the one leak the
    view-based firewall cannot close."""
    data = synthetic_data()
    config, feature_config = order_flow(), features()
    comp = OrderFlowFeatures(config, feature_config)
    assert len(data.primary_bars) > 2 * comp.warmup_bars

    result = audit_computer(
        data=data,
        factory=lambda d: OrderFlowFeatures(config, feature_config),
        sample=40,
    )
    assert result.passed, result.summary()
    assert result.bars_checked > 0
    assert result.keys_checked == tuple(sorted(comp.keys))
    assert_no_lookahead([result])


def test_lookahead_audit_passes_on_the_unavailable_path_too():
    """The disablement path has its own branches, and an audit that only ever
    saw the available path would not have exercised them. A bars-only dataset
    must still be truncation- and mutation-invariant and must still report
    `warmup_complete=False` before warmup."""
    data = synthetic_data(include_ticks=False)
    assert data.ticks == {}
    config, feature_config = order_flow(), features()
    result = audit_computer(
        data=data,
        factory=lambda d: OrderFlowFeatures(config, feature_config),
        sample=25,
    )
    assert result.passed, result.summary()
    assert result.bars_checked > 0
    assert_no_lookahead([result])


def test_the_audit_gate_catches_a_constructor_that_captures_the_whole_sample():
    """GUARDS THE GUARD. A gate that cannot fail manufactures the appearance of
    validation, so the cheat is constructed and the harness is required to
    report it. This is the one leak `MarketView` cannot close: the constant is
    taken before any view exists, so checks 1 and 2 only see it because the
    factory rebuilds the computer against the altered dataset."""

    class _CapturesAtInit(OrderFlowFeatures):
        name = "orderflow_capture"

        def __init__(self, config, feature_config, whole_sample) -> None:
            super().__init__(config, feature_config)
            self._captured = float(whole_sample.primary_bars.col("close").mean())

        def compute(self, view):
            vector = super().compute(view)
            if not vector.warmup_complete:
                return vector
            values = dict(vector.values)
            values["cvd_slope_normalized"] += self._captured
            return vector.replace(values=values)

    data = synthetic_data()
    config, feature_config = order_flow(), features()
    result = audit_computer(
        data=data,
        factory=lambda d: _CapturesAtInit(config, feature_config, d),
        sample=40,
    )
    assert not result.passed
    # 72 findings as measured; asserted as a floor so that a harness that
    # detects MORE does not fail here, while one that detects less does.
    assert len(result.findings) >= 72, result.summary()
    assert {f.kind for f in result.findings} == {"truncation", "mutation"}
    with pytest.raises(AssertionError, match="lookahead audit failed"):
        assert_no_lookahead([result])


def test_the_audit_gate_catches_a_computer_that_reports_ready_too_early():
    """GUARDS THE GUARD, and reports the gate's boundary.

    The harness's warmup check fires on `warmup_complete=True` at a bar before
    the DECLARED warmup, so the cheat has to report ready while its window is
    short -- which is the observable form of the regime detector's bug (a
    half-filled rolling window handed out as a feature).

    THE BOUNDARY, WHICH IS ITSELF A FINDING: a computer that under-declares
    `warmup_bars` and still REFUSES a short window produces zero findings here,
    because every pre-warmup bar it is asked about reports not-ready. Pure
    under-declaration is therefore invisible to this harness, which is why
    `test_nothing_before_the_declared_warmup_reaches_the_output` exists and does
    the arithmetic directly."""

    class _ReportsReadyEarly(OrderFlowFeatures):
        name = "orderflow_short_warmup"

        @property
        def warmup_bars(self) -> int:
            return 20

        def compute(self, view):
            values = {key: 0.0 for key in self.keys}
            values["signed_delta_contracts"] = float(view.deltas(TICK_ROWS).sum())
            values["classification_coverage"] = 1.0
            return self._vector(view, values)      # warmup_complete=True, always

    data = synthetic_data()
    config, feature_config = order_flow(), features()
    result = audit_computer(
        data=data,
        factory=lambda d: _ReportsReadyEarly(config, feature_config),
        sample=40,
    )
    assert not result.passed
    # 3 findings as measured: `_sample_indices` adds exactly three pre-warmup
    # bars ({0, warmup - 2, warmup // 2}) so that check 3 is actually
    # evaluated, and each of them catches this cheat. Asserted as a floor.
    assert len(result.findings) >= 3, result.summary()
    assert {f.kind for f in result.findings} == {"warmup"}

    # The boundary, measured: under-declaring alone is not caught.
    class _UnderDeclaresOnly(OrderFlowFeatures):
        name = "orderflow_under_declares"

        @property
        def warmup_bars(self) -> int:
            return 20

    silent = audit_computer(
        data=data,
        factory=lambda d: _UnderDeclaresOnly(config, feature_config),
        sample=40,
    )
    assert silent.passed, "the harness was expected NOT to see this; see the docstring"
    assert len(silent.findings) == 0


def test_the_audit_gate_catches_unseeded_jitter():
    """GUARDS THE GUARD. Check 4 computes twice on the identical view, so a
    computer reading a clock or an unseeded generator is caught per bar -- the
    check that makes the truncation and mutation checks meaningful as regression
    tests on `MarketView` rather than no-ops."""

    class _Jitters(OrderFlowFeatures):
        name = "orderflow_jitter"

        def compute(self, view):
            vector = super().compute(view)
            if not vector.warmup_complete:
                return vector
            values = dict(vector.values)
            values["signed_delta_contracts"] += float(
                np.random.default_rng().normal(0.0, 1.0)
            )
            return vector.replace(values=values)

    data = synthetic_data()
    config, feature_config = order_flow(), features()
    result = audit_computer(
        data=data,
        factory=lambda d: _Jitters(config, feature_config),
        sample=40,
    )
    assert not result.passed
    # 111 findings as measured, across all three kinds; asserted as a floor.
    assert len(result.findings) >= 111, result.summary()
    assert "determinism" in {f.kind for f in result.findings}


# ---------------------------------------------------------------------------
# defects this file REPORTS rather than repairs
# ---------------------------------------------------------------------------


def test_absorption_is_structurally_zero_on_five_minute_data_here_reported():
    """A DEAD TERM IN A WEIGHTED SUM. 0.15 of this component's weight cannot be
    earned at the default constants, and nothing downstream can see it.

    The band is recomputed HERE from the view's own closes and a local ATR built
    from `volatility.true_range` / `volatility.wilder_atr_series`, so the
    measurement is this file's arithmetic and not the module's answer handed
    back to itself. Over 5-minute synthetic NQ the ten-close excursion never
    falls below 0.25 ATR -- the measured minimum is above 0.38 and the median
    above 1.2 -- so `flat = clamp01(1 - band / 0.25)` is identically 0 and the
    geometric mean with it.

    The cause is a scale mismatch, not a coding error: 0.25 ATR is anchored on
    `levels.zone_band_atr`, a cluster MERGE DISTANCE between two price levels,
    while this term compares it against the excursion of ten consecutive closes,
    which for a random walk runs about `sqrt(10) = 3.16` ATR.

    NOT TUNED HERE. The constant is a hypothesis for the Phase 8 sweep and
    moving it to make the term fire would be choosing a value by looking at a
    result. Two things are asserted instead: that the term really is dead on
    this data, and that `order_flow_available_weight` reports 1.0 while it is --
    the invisible ceiling, which is the part that belongs in an
    `OrderFlowConfig` validator. The term is NOT dead code: the hand-built
    geometry in `test_absorption_is_one_when_heavy_one_sided_flow_does_not_move_price`
    reaches 1.0."""
    data = synthetic_data()
    comp = computer()
    bands: list[float] = []
    checked = 0
    for index in range(comp.warmup_bars - 1, len(data.primary_bars), 23):
        view = view_at(data, index)
        vector = comp.compute(view)
        if vector.values["order_flow_available"] == 0.0:
            continue
        checked += 1
        atr = local_atr(view)
        window = view.closes(ABSORPTION_LOOKBACK)
        bands.append((float(window.max()) - float(window.min())) / atr)
        assert vector.values["absorption"] == 0.0, index
        assert vector.values["absorption_direction"] == 0.0
        # The term is AVAILABLE while structurally zero: its inputs exist and
        # the measurement is real, so its weight stays in the denominator and
        # the ceiling is invisible.
        assert vector.quality_of("absorption") is DataQuality.GOOD
        assert vector.values["order_flow_available_weight"] == 1.0

    assert checked > 100, f"only {checked} bars sampled"
    measured = np.array(bands)
    assert float(measured.min()) > LEVEL_BAND_ATR, (
        "the excursion band dropped below the configured level window on this "
        "data, so the finding recorded in this test no longer holds and the "
        "docstring must be remeasured"
    )
    assert float(np.median(measured)) > 1.0
    # What the threshold would have to be to describe a flat ten-close window.
    assert LEVEL_BAND_ATR * math.sqrt(ABSORPTION_LOOKBACK) == pytest.approx(0.7905694)


def test_window_coverage_cannot_see_a_collapse_on_the_newest_bars_here_reported():
    """A DEFECT THIS FILE REPORTS RATHER THAN REPAIRS.

    `classification_coverage` is measured over the WHOLE 129-row read window.
    The documented rationale is that every term's comparison set spans that
    window, and the documented cost is that "a collapse confined to the last few
    bars moves it only a few percent". That cost is the defect: on the bar being
    traded the delta's sign can be set by almost none of the tape while the gate
    sees a healthy number.

    Hand arithmetic. 119 rows carry 60 bought, 40 sold and nothing unclassified;
    the newest 10 rows carry 3 bought, 2 sold and 95 unclassified, a per-row
    coverage of 0.05. The window figure is

        (119 * 100 + 10 * 5) / (119 * 100 + 10 * 100) = 11950 / 12900 = 0.92636

    comfortably above the 0.60 floor. So the component reports available, every
    key is GOOD, `available_weight` is 1.0 and there is not one note -- while
    95% of the newest bar's volume has no aggressor at all. The gate exists
    precisely to refuse "a delta whose sign is set by a minority of the tape".

    NOT REPAIRED: the fix is a sixth disabling condition plus a config field
    (a per-bar coverage floor beside the window one), which is a design and
    schema change rather than something a test file may decide. Pinned so that
    a future change to the gate's window shows up here."""
    n = 160
    buy = np.full(n, 60.0)
    sell = np.full(n, 40.0)
    unclassified = np.zeros(n)
    buy[-POOL:] = 3.0
    sell[-POOL:] = 2.0
    unclassified[-POOL:] = 95.0

    data = flat_tape(
        n=n,
        buy=buy,
        sell=sell,
        unclassified=unclassified,
        buy_trades=6.0,
        sell_trades=4.0,
        max_trade=np.arange(float(n)) + 1.0,
    )
    view = view_at(data)
    vector = computer().compute(view)

    expected = (119 * 100 + 10 * 5) / (119 * 100 + 10 * 100)
    assert expected == pytest.approx(0.9263565891472868)
    assert vector.values["classification_coverage"] == pytest.approx(expected)
    assert vector.values["classification_coverage"] > MIN_COVERAGE

    # The newest row's own coverage, which nothing in the module reads.
    newest = view.last_tick_aggregate()
    assert isinstance(newest, TickAggregate)
    assert newest.classification_coverage == pytest.approx(0.05)
    assert newest.classification_coverage < MIN_COVERAGE

    assert vector.values["order_flow_available"] == 1.0
    assert vector.values["order_flow_available_weight"] == 1.0
    assert vector.quality is DataQuality.GOOD
    assert vector.quality_of("classification_coverage") is DataQuality.GOOD
    assert vector.notes == ()


def test_an_interior_tick_gap_is_not_detected_here_reported():
    """A KNOWN LIMITATION THIS FILE PINS.

    The absorption term reads deltas from the tick feed and prices from the bar
    feed, which is only meaningful if row *i* of one window describes the same
    interval as row *i* of the other. `compute` verifies the RIGHT EDGE -- the
    newest tick aggregate and the newest bar must close at the same instant --
    and that both windows are full. Interior per-row alignment is not verified,
    because `MarketView` exposes no tick timestamps and materializing 130
    contract objects per bar costs several times the whole computation.

    So a feed with one interior row missing inside the absorption window
    computes normally: its 10-row delta window spans 11 bars while its 10-close
    price window spans 10. The fix is a `tick_timestamps(n)` accessor on
    `MarketView` beside `bar_timestamps(n)`, which is a data-layer change.
    Pinned here so that when that accessor arrives, this test fails and says
    where to look."""
    n = 170
    bars = bars_from_mids(np.full(n, 100.0))
    keep = np.ones(n, dtype=bool)
    keep[n - 5] = False                       # inside the newest 10 rows
    gapped_stamps = stamps(n)[keep]
    ticks = tick_series(int(gapped_stamps.size), ts_ns=gapped_stamps)

    view = view_at(symbol_data(bars, ticks))
    assert view.bar_count() == n
    assert view.tick_count() == n - 1
    # The right edge still agrees, which is all the module checks.
    assert view.last_tick_aggregate().close_ts == view.last_bar().close_ts

    vector = computer().compute(view)
    assert vector.values["order_flow_available"] == 1.0
    assert not has_note(vector, "gap")
    assert not has_note(vector, "align")


def test_a_tick_aggregate_cannot_express_a_not_supplied_optional_field():
    """A CONTRACT NOTE, recorded because a reader is likely to assume otherwise.

    `OptionsSnapshot` declares its optional fields `float | None`, where None
    means NOT SUPPLIED and 0.0 is a meaningful observation. `TickAggregate` does
    NOT: `unclassified_volume`, `buy_trades`, `sell_trades` and `max_trade_size`
    are plain numbers defaulting to 0.0 / 0, and `TickSeries.tick_at` fills an
    absent column with that default. So a materialized `TickAggregate` cannot
    distinguish "no unclassified volume" from "unclassified volume not
    reported", and "no prints" from "print counts not carried".

    `orderflow.py` is right not to use it for that: it asks
    `MarketView.tick_column`, which returns an EMPTY array for an absent column,
    and `_Windows` keeps that absence as None. This test states the reason that
    indirection exists, so nobody later "simplifies" it into reading
    `TickAggregate` fields and silently turns absence into zero."""
    supplied = TickAggregate(
        symbol=SYMBOL,
        close_ts=T0,
        buy_volume=60.0,
        sell_volume=40.0,
        unclassified_volume=0.0,
        buy_trades=0,
        max_trade_size=0.0,
        classification_method=HONEST_METHOD,
    )
    absent_series = TickSeries(
        symbol=SYMBOL,
        ts_ns=stamps(2),
        columns={"buy_volume": np.full(2, 60.0), "sell_volume": np.full(2, 40.0)},
        meta={"classification_method": HONEST_METHOD},
    )
    materialized = absent_series.tick_at(1)
    # Identical fields from two different facts.
    assert materialized.unclassified_volume == supplied.unclassified_volume == 0.0
    assert materialized.buy_trades == supplied.buy_trades == 0
    assert materialized.max_trade_size == supplied.max_trade_size == 0.0
    assert materialized.classification_coverage == 1.0

    # The route the module actually uses keeps the two apart.
    view = view_at(
        symbol_data(bars_from_mids(np.full(160, 100.0)), tick_series(160, max_trade=None))
    )
    assert view.tick_column("max_trade_size", 10).size == 0
    assert view.tick_column("buy_volume", 10).size == 10
    present = view_at(flat_tape(max_trade=0.0))
    assert present.tick_column("max_trade_size", 10).size == 10
    assert np.all(present.tick_column("max_trade_size", 10) == 0.0)


def test_a_degenerate_absorption_window_reports_zero_without_a_note_here_reported():
    """A SMALL ASYMMETRY, pinned.

    `_signed_delta` emits a note when its window is degenerate and says what
    the tie-inclusive rank would wrongly have reported. The absorption term
    takes the same precaution in code -- all-equal window sums report no effort
    rather than a rank of 1.0 -- but emits NO note, so a persistently one-sided
    tape reports `absorption = 0.0` with GOOD quality and nothing to read. The
    module's stated reason for not noting a zero absorption is that `flat == 0`
    would fire on nearly every bar and drown the notes that matter; that
    argument does not cover this branch, which fires rarely.

    Reported rather than repaired: adding a note is a behaviour change to
    another module's note discipline, and the value itself is correct."""
    vector = computer().compute(view_at(flat_tape(buy=60.0, sell=40.0, max_trade=5.0)))
    # Every 10-bar window sums to exactly +200, so there is no distribution to
    # be extreme of and no effort is reported.
    assert vector.values["absorption"] == 0.0
    assert vector.quality_of("absorption") is DataQuality.GOOD
    assert not has_note(vector, "absorption")
    # The delta term, on the same tape and the same situation, does say so.
    assert has_note(vector, "smoothed delta observations")


# ---------------------------------------------------------------------------
# the shared helpers this module exposes
# ---------------------------------------------------------------------------


def test_the_rolling_helpers_refuse_a_window_longer_than_their_input():
    """A sliding window over too few values would otherwise return an empty
    array and the caller would index `[-1]` of nothing. Raising names the
    window that was too long."""
    with pytest.raises(FeatureError, match="at least 5 values"):
        rolling_sum(np.arange(4.0), 5)
    with pytest.raises(FeatureError, match="at least 5"):
        rolling_max(np.arange(4.0), 5)
    with pytest.raises(FeatureError, match="at least 5"):
        rolling_mean(np.arange(4.0), 5)
    for helper in (rolling_sum, rolling_max, rolling_mean):
        with pytest.raises(FeatureError, match="at least 1"):
            helper(np.arange(4.0), 0)
    # Length is values - window + 1, which is the arithmetic every chain rests on.
    assert rolling_sum(np.arange(10.0), 3).size == 8
    assert rolling_max(np.arange(10.0), 10).size == 1


def test_signed_rank_and_the_band_helpers_are_bounded_and_refuse_bad_limits():
    """Bounded transforms only, and never a raw z-score: an unbounded term lets
    one extreme observation dominate a weighted component score."""
    window = np.arange(100.0)
    assert signed_rank(window, -1.0) == -1.0
    assert signed_rank(window, 1000.0) == 1.0
    assert signed_rank(window, 49.0) == pytest.approx(0.0)
    assert signed_rank(np.full(10, 3.0), 3.0) == 0.0        # degenerate, not +1
    assert signed_rank(np.zeros(0), 1.0) == 0.0
    assert signed_rank(np.array([np.nan, np.nan]), 1.0) == 0.0

    with pytest.raises(FeatureError, match="leaves no room above it"):
        excess_above(0.5, 1.0)
    with pytest.raises(FeatureError, match="must be positive"):
        inside_band(1.0, 0.0)
    assert 0.0 <= excess_above(0.95, 0.9) <= 1.0
    assert 0.0 <= inside_band(1e9, 0.25) <= 1.0
    assert inside_band(1e9, 0.25) == 0.0


def test_the_synthetic_dataset_this_file_uses_carries_no_usable_outcome():
    """A standing reminder, as a test, of why nothing here is a predictive claim.

    The generator's tick stream is built from the SAME innovations as its bars,
    so any correlation between a feature in this file and a synthetic forward
    return would be a property of the generator. The dataset this file builds is
    therefore constructed without labels and the features are exercised for
    mechanics only -- bounds, finiteness, determinism, degradation, lookahead
    and arithmetic. The assertion below is a reminder with teeth: the
    `SymbolData` these tests pass around has no outcome column of any kind to
    regress against, so a later well-meant addition would have to go and fetch
    one, which is the moment to stop and reread this docstring."""
    data = synthetic_data()
    assert set(data.primary_bars.columns) == {"open", "high", "low", "close", "volume"}
    assert data.options is None
    assert data.quotes is None
    assert not hasattr(data, "labels")
    for column in data.ticks[INTERVAL].columns:
        assert column in set(TickSeries.REQUIRED) | set(TickSeries.OPTIONAL)


# ---------------------------------------------------------------------------
# second pass: holes an independent mutation sweep found in the tests above
# ---------------------------------------------------------------------------
#
# The tests in this file were mutation-tested a second time by a reviewer who
# had not written them, against 28 deliberate breaks of `orderflow.py` that
# were not in the implementer's own list. Six survived, and the four that were
# not behaviourally inert at the defaults are pinned below. A mutation the
# suite does not catch is a coverage hole whatever the test count is, and in
# three of the four cases the reason was the same: the FIXTURE made the window
# length unobservable, because a constant per-bar delta gives the identical
# answer over 20 bars or 21.


def test_the_cvd_window_is_exactly_cvd_lookback_bars_long():
    """SURVIVING MUTATION: `deltas[-(C + 1):]` passed all 76 tests above.

    `test_cvd_slope_is_the_volume_normalized_ols_slope_squashed` pins the
    arithmetic on a tape whose delta is +20 on EVERY bar. The CVD path is then a
    perfect straight line, and the OLS slope of a straight line is the same over
    twenty points or twenty-one, so that test cannot see a window length at all.
    This one uses a delta that reverses halfway, which makes the length visible.

    Hand arithmetic. The newest twenty deltas are +20 ten times then -20 ten
    times, so the CVD path is 20, 40 ... 200, 180 ... 20, 0. With `x = 0..19`,
    `sum((x - 9.5) ** 2) = 20 * (20 ** 2 - 1) / 12 = 665` and
    `sum((x - 9.5) * y) = -1000`, so

        slope                = -1000 / 665 = -200 / 133 contracts per bar
        cvd_slope_normalized = (-200 / 133) / 120 = -5 / 399
        cvd_slope            = -tanh((5 / 399) / 0.25) = -tanh(20 / 399)

    where 120 is the mean classified volume per bar (70 + 50 on the up half,
    50 + 70 on the down half). The off-by-one is not a small error here: a
    twenty-one-bar window prepends one more +20 bar and the slope of the
    resulting symmetric path is ZERO."""
    n = 170
    buy = np.full(n, 70.0)
    sell = np.full(n, 50.0)
    buy[-CVD_LOOKBACK // 2 :] = 50.0            # the newest ten bars reverse
    sell[-CVD_LOOKBACK // 2 :] = 70.0
    view = view_at(flat_tape(n=n, buy=buy, sell=sell, max_trade=np.arange(float(n)) + 1.0))
    vector = computer().compute(view)

    # The slope, derived here from the OLS definition rather than from the
    # module's `least_squares_slope`.
    path = np.cumsum(np.concatenate([np.full(10, 20.0), np.full(10, -20.0)]))
    x = np.arange(path.size, dtype=np.float64) - 9.5
    slope = float(np.dot(x, path - path.mean()) / np.dot(x, x))
    assert float(np.dot(x, x)) == 665.0
    assert slope == pytest.approx(-200 / 133)

    assert vector.values["cvd_slope_normalized"] == pytest.approx(slope / 120.0)
    assert vector.values["cvd_slope_normalized"] == pytest.approx(-5 / 399)
    assert vector.values["cvd_slope"] == pytest.approx(-math.tanh(20 / 399))
    assert vector.quality_of("cvd_slope") is DataQuality.GOOD

    # What the off-by-one would have produced, written out so the assertion is
    # on a number and not on the absence of a failure.
    longer = np.cumsum(np.concatenate([[20.0], np.full(10, 20.0), np.full(10, -20.0)]))
    xl = np.arange(longer.size, dtype=np.float64) - longer.size / 2 + 0.5
    off_by_one = float(np.dot(xl, longer - longer.mean()) / np.dot(xl, xl))
    assert off_by_one == pytest.approx(0.0, abs=1e-9)
    assert vector.values["cvd_slope_normalized"] != pytest.approx(off_by_one / 120.0)

    # And the bar one step beyond the window moves nothing, which is the other
    # half of "the window is exactly C long".
    beyond = buy.copy()
    beyond[-CVD_LOOKBACK - 1] = 10.0            # a wildly different delta there
    moved = computer().compute(
        view_at(flat_tape(n=n, buy=beyond, sell=sell, max_trade=np.arange(float(n)) + 1.0))
    )
    assert moved.values["cvd_slope"] == vector.values["cvd_slope"]
    assert moved.values["cvd_slope_normalized"] == vector.values["cvd_slope_normalized"]


def test_the_absorption_effort_window_is_the_full_composed_chain():
    """SURVIVING MUTATION: `_absorption_chain = P` (dropping the `+ A - 1`)
    passed all 76 tests above.

    Every test that reaches a non-zero `absorption` does it either on a frozen
    tape, where the current window sum is the maximum of its own history and
    ranks 1.0 whatever the history's length, or with
    `absorption_lookback_bars=1`, where `P + A - 1` IS `P` and the mutation is
    the identity. So the comparison set's size was never observable.

    The construction below makes it observable by choosing the sixty window sums
    directly. A sliding ten-bar sum is `cumulative[j + 10] - cumulative[j]`, so
    any sequence of sums can be realized by building the cumulative path and
    differencing it. The nine OLDEST sums -- exactly the nine the short chain
    would not see -- are set far above the current one, and forty-nine of the
    sixty are at or below it:

        rank   = 49 / 60 = 0.81666...        effort = (49/60 - 0.80) / 0.20 = 1/12
        absorption = (1/12 * 1 * 1) ** (1/3) = 0.436790...

    on a flat tape, where `quiet` and `flat` are both exactly 1. The short chain
    sees fifty-one sums, forty-nine of them at or below the current one, so its
    rank is 49/51 = 0.9607... and its effort 0.8039... -- a different number,
    written out below."""
    chain = DELTA_LOOKBACK + ABSORPTION_LOOKBACK - 1
    assert chain == 69

    sums = np.empty(DELTA_LOOKBACK)
    sums[:9] = 1000.0                           # only the full chain sees these
    sums[9:57] = np.linspace(10.0, 99.0, 48)    # 48 strictly below the current
    sums[57] = 400.0
    sums[58] = 500.0
    sums[59] = 100.0                            # the current window sum
    assert int(np.count_nonzero(np.abs(sums) <= 100.0)) == 49
    assert int(np.count_nonzero(np.abs(sums[9:]) <= 100.0)) == 49

    cumulative = np.zeros(chain + 1)
    for j in range(DELTA_LOOKBACK):
        cumulative[j + ABSORPTION_LOOKBACK] = cumulative[j] + sums[j]
    chain_deltas = np.diff(cumulative)
    assert chain_deltas.size == chain
    realized = rolling_sum(chain_deltas, ABSORPTION_LOOKBACK)
    assert realized.size == DELTA_LOOKBACK
    assert np.allclose(realized, sums)

    n = 170
    deltas = np.zeros(n)
    deltas[-chain:] = chain_deltas
    base = 2000.0
    view = view_at(
        flat_tape(
            n=n,
            buy=base + np.where(deltas > 0.0, deltas, 0.0),
            sell=base + np.where(deltas < 0.0, -deltas, 0.0),
            max_trade=np.arange(float(n)) + 1.0,
        )
    )
    assert local_atr(view) == 2.0
    vector = computer().compute(view)

    effort = (49 / 60 - ABSORPTION_PERCENTILE) / (1.0 - ABSORPTION_PERCENTILE)
    assert effort == pytest.approx(1 / 12)
    expected = (effort * 1.0 * 1.0) ** (1 / ABSORPTION_TERM_COUNT)
    assert expected == pytest.approx(0.4367902323681)
    assert vector.values["absorption"] == pytest.approx(expected)
    assert vector.values["absorption_direction"] == -1.0      # absorbed buying

    short_effort = (49 / 51 - ABSORPTION_PERCENTILE) / (1.0 - ABSORPTION_PERCENTILE)
    short = short_effort ** (1 / ABSORPTION_TERM_COUNT)
    assert short == pytest.approx(0.9298321540)
    assert vector.values["absorption"] != pytest.approx(short)


def test_the_absorption_geometry_reads_one_bar_more_than_its_lookback():
    """SURVIVING MUTATIONS: both `_bar_rows` formulas that drop the absorption
    geometry passed all 76 tests above, because at the default `atr_period=14`
    the ATR window (15 bars) dominates `absorption_lookback_bars + 1` (11) and
    the two mutations are the identity. Only a config where the absorption
    window is the longer of the two can see them, and no test used one.

    `atr_period=5` makes the ATR window six bars and the absorption geometry
    eleven. `quiet` reads the close from BEFORE the window -- `closes[-1 - A]`,
    the eleventh back -- so a computer that read only six bars would index past
    the start of its own window. The tape puts the only price difference exactly
    there:

        displacement = |100.00 - 100.25| = 0.25 points = 0.125 ATR
        quiet        = 1 - 0.125 / 0.25 = 0.5
        absorption   = (1 * 0.5 * 1) ** (1/3) = 0.793700...
    """
    n = 170
    comp = computer(atr_period=5, absorption_lookback_bars=ABSORPTION_LOOKBACK)
    assert comp.warmup_bars == WARMUP

    mids = np.full(n, 100.0)
    mids[-(ABSORPTION_LOOKBACK + 1)] = 100.25     # the reference close, nothing else
    heavy = np.full(n, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    tick_kwargs = dict(buy=100.0 + heavy, sell=100.0, max_trade=np.arange(float(n)) + 1.0)

    view = view_at(tape(mids, **tick_kwargs))
    assert local_atr(view, period=5) == 2.0
    assert float(np.ptp(view.closes(ABSORPTION_LOOKBACK))) == 0.0   # the band is flat
    vector = comp.compute(view)

    expected = (1.0 * 0.5 * 1.0) ** (1 / ABSORPTION_TERM_COUNT)
    assert expected == pytest.approx(0.7937005259841)
    assert vector.values["absorption"] == pytest.approx(expected)
    assert vector.values["absorption_direction"] == -1.0

    # One bar further back is outside both windows and must move nothing.
    further = np.full(n, 100.0)
    further[-(ABSORPTION_LOOKBACK + 2)] = 100.25
    outside = comp.compute(view_at(tape(further, **tick_kwargs)))
    assert outside.values["absorption"] == 1.0                    # quiet is 1 again
    assert outside.values["absorption"] != pytest.approx(expected)


def test_a_non_finite_trade_count_on_the_newest_row_is_refused_not_raised():
    """A DEFECT FOUND AND FIXED on the second pass.

    `TickSeries.NON_NEGATIVE` compares with `nanmin`, so a NaN in `buy_trades`
    passes series validation, and `_read_windows` is written to treat a broken
    count column as NOT SUPPLIED. But it never ran: `compute` materializes the
    newest `TickAggregate` first -- the only route `MarketView` offers to the
    feed's declared classification method and its right-edge timestamp -- and
    `TickSeries.tick_at` coerces the counts with `int(...)`, which raises
    `ValueError` on a NaN. The result was an unhandled exception out of a
    feature computer, which `FeatureBundle` does not catch and principle 4
    forbids: missing data produces WAIT, never a crash.

    The root cause is in the data layer (`tick_at` should carry a non-finite
    optional field as not-supplied rather than coerce it), and until it moves
    there this module refuses rather than measures: a feed whose newest row
    cannot be read is one whose right edge cannot be verified."""
    n = 170
    counts = np.full(n, 6.0)
    counts[-1] = np.nan                       # the newest row only
    vector = computer().compute(
        view_at(flat_tape(n=n, buy_trades=counts, sell_trades=4.0))
    )
    note = assert_fully_unavailable(vector)
    assert "newest tick aggregate cannot be read" in note
    assert "non-finite buy_trades or sell_trades" in note

    # One row older and the documented handling applies instead: the columns are
    # treated as NOT SUPPLIED, the two terms that need them are dropped, and the
    # component stays available on the terms that do not.
    older = np.full(n, 6.0)
    older[-2] = np.nan
    degraded = computer().compute(
        view_at(flat_tape(n=n, buy_trades=older, sell_trades=4.0))
    )
    assert degraded.values["order_flow_available"] == 1.0
    assert note_with(degraded, "treated as NOT SUPPLIED")
    assert degraded.values["order_flow_available_weight"] == pytest.approx(
        1.0 - W_AGGRESSION - W_SIZE
    )


def test_volumes_that_overflow_double_precision_are_refused_not_raised():
    """A DEFECT FOUND AND FIXED on the second pass.

    `_read_windows` checked that every per-row volume was finite, which does not
    make their SUMS finite -- and the smoothed delta, the CVD cumulation, the
    absorption window sums and the pooled trade size are all sums over that
    window. A feed carrying about 1e308 lots a bar overflowed to an infinite
    `signed_delta_contracts`, and the base class's finiteness guard turned that
    into a `FeatureError` out of `compute` rather than the reported refusal this
    module promises everywhere else.

    Volumes are non-negative, so the window total bounds every sub-window sum
    and one check covers all of them."""
    n = 170
    with np.errstate(over="ignore", invalid="ignore"):
        vector = computer().compute(
            view_at(flat_tape(n=n, buy=1e308, sell=1e307))
        )
        note = assert_fully_unavailable(vector)
        assert "outside double precision" in note

        # The unclassified column has its own total, and its own reason.
        unclassified = computer().compute(
            view_at(flat_tape(n=n, buy=1.0, sell=1.0, unclassified=1e308))
        )
        assert_fully_unavailable(unclassified)
        assert "unclassified_volume window sums" in unclassified.notes[0]

    # A large but non-overflowing tape still computes: the guard is on the
    # arithmetic and not a cap on how busy an instrument may be.
    busy = computer().compute(view_at(flat_tape(n=n, buy=6e290, sell=4e290)))
    assert busy.values["order_flow_available"] == 1.0
    assert math.isfinite(busy.values["signed_delta_contracts"])


def test_no_bar_price_reaches_any_emitted_value_at_the_default_constants():
    """A DEFECT THIS FILE REPORTS RATHER THAN REPAIRS -- the sharp form.

    `test_absorption_is_structurally_zero_on_five_minute_data_here_reported`
    establishes that 0.15 of the weight cannot be earned at the default
    `absorption_level_window_atr`. The consequence is stronger than that, and
    is the version worth pinning: absorption is the ONLY term that reads a bar
    price, so while it is structurally zero the bar feed contributes nothing to
    any of the fifteen emitted values. `BARS` is in `required_feeds`, and at the
    default constants it functions purely as a gate -- it decides whether a
    vector is emitted (through the price-sanity checks, the right-edge alignment
    check and the ATR-is-zero drop) and never what is in one.

    Below: sixty randomized bar tapes at price levels from hundreds to
    thousands and ranges from 0.01 to 40 points, with the tick feed held
    bit-identical. Not one emitted value moves. `test_bar_columns_the_module_
    does_not_read_move_nothing` above holds `high`, `low` and `close` fixed and
    perturbs only `open` and `volume`, so it reads as though the geometry
    mattered; it does not, here, yet.

    NOT REPAIRED, for the same reason the structural-zero test gives: the
    constant is a Phase 8 hypothesis and moving it to make the term fire would
    be choosing a value by looking at a result. Pinned so that whoever changes
    `absorption_level_window_atr` -- or scales it by
    `sqrt(absorption_lookback_bars)` -- sees this test fail and knows the bar
    feed has started to matter."""
    rng = np.random.default_rng(20240304)
    n = 170
    tick_kwargs = dict(
        buy=rng.uniform(10.0, 300.0, n),
        sell=rng.uniform(10.0, 300.0, n),
        buy_trades=np.floor(rng.uniform(3.0, 30.0, n)),
        sell_trades=np.floor(rng.uniform(3.0, 30.0, n)),
        max_trade=np.floor(rng.uniform(1.0, 500.0, n)),
    )
    ticks = tick_series(n, **tick_kwargs)
    comp = computer()

    def emitted(mids, span):
        bars = BarSeries(
            symbol=SYMBOL,
            ts_ns=stamps(n),
            interval_seconds=INTERVAL,
            columns={
                "open": mids.copy(),
                "high": mids + span,
                "low": mids - span,
                "close": mids.copy(),
                "volume": np.full(n, 1000.0),
            },
        )
        return comp.compute(view_at(symbol_data(bars, ticks)))

    baseline = emitted(np.full(n, 100.0) + np.cumsum(rng.uniform(-1.0, 1.0, n)), np.full(n, 1.0))
    assert baseline.values["order_flow_available"] == 1.0
    assert baseline.values["absorption"] == 0.0
    assert baseline.values["order_flow_available_weight"] == 1.0

    checked = 0
    for _ in range(60):
        mids = 500.0 + np.abs(
            1000.0 * rng.uniform(0.5, 2.0) + np.cumsum(rng.uniform(-50.0, 50.0, n))
        )
        other = emitted(mids, rng.uniform(0.01, 40.0, n))
        if other.values["order_flow_available"] == 0.0:
            continue                      # the gate fired; that is bars mattering
        checked += 1
        assert other.values == baseline.values
        assert other.quality_by_key == baseline.quality_by_key
        assert other.notes == baseline.notes
    assert checked > 50, f"only {checked} tapes reached the available path"


def test_the_negative_volume_branches_are_dead_because_the_series_refuses_them():
    """Why two more of this module's branches cannot be given a test.

    `_read_windows` refuses a negative `buy_volume`, `sell_volume` or
    `unclassified_volume`, and `TickSeries.NON_NEGATIVE` covers all three, so no
    legally constructed feed can reach those lines. They are kept as a guard
    against a data-layer change; this test is what would fail if that guard ever
    became load-bearing without anyone noticing, and it is why the count of
    unreachable branches in this file's header is five rather than two.

    The NaN half of the same checks is NOT dead: `NON_NEGATIVE` compares with
    `nanmin`, which a NaN passes."""
    n = 170
    for column in ("buy_volume", "sell_volume", "unclassified_volume"):
        values = np.full(n, 10.0)
        values[-1] = -1.0
        columns = {"buy_volume": np.full(n, 60.0), "sell_volume": np.full(n, 40.0)}
        columns[column] = values
        with pytest.raises(SchemaError, match="negative values"):
            TickSeries(
                symbol=SYMBOL,
                ts_ns=stamps(n),
                columns=columns,
                meta={"classification_method": HONEST_METHOD},
            )

    # The NaN half is reachable, and is reported rather than propagated.
    nan_volume = np.full(n, 60.0)
    nan_volume[-1] = np.nan
    vector = computer().compute(view_at(flat_tape(n=n, buy=nan_volume)))
    note = assert_fully_unavailable(vector)
    assert "buy_volume window is not finite" in note


def test_the_central_moment_reads_classified_volume_and_not_one_side():
    """SURVIVING MUTATION: `rolling_sum(windows.buys[...])` in place of
    `windows.classified_volume` passed all 83 tests above.

    `test_the_central_moment_is_pooled_volume_over_pooled_trade_count` uses
    120/80 and 60/40, both a 60:40 split, so a buy-side-only average is the
    correct one scaled by 0.6 -- and a percentile RANK is scale-invariant, so
    the two implementations agree exactly. The split has to VARY for the
    difference to be observable, and it is tested in both directions here.

    Tape one: the buy side is 60 lots on every bar and only the sell side moves
    (300 outside the newest fifteen rows, 40 inside them), so the classified
    total falls from 360 to 100 while the buy side never changes. The six pools
    wholly inside the light tail are the only observations at or below the
    newest, exactly as in the test above, so the rank is 6/120 -- while a
    buy-side-only average would be 600/100 = 6.0 on every one of the 120 pools,
    a degenerate window reported as UNAVAILABLE.

    Tape two is the mirror: the split flips but the total does not, so the
    correct average is identical on every pool (a degenerate window, DEGRADED
    and 0.5) while a buy-side-only average would have found a clean 6/120."""
    n = 170
    tail = POOL + 5
    comp = computer()

    moving_total = np.full(n, 300.0)
    moving_total[-tail:] = 40.0
    one = comp.compute(
        view_at(
            flat_tape(n=n, buy=60.0, sell=moving_total, buy_trades=6.0, sell_trades=4.0)
        )
    )
    assert one.values["avg_trade_size_percentile"] == pytest.approx(6 / 120)
    assert one.values["avg_trade_size_percentile"] == pytest.approx(0.05)
    assert one.quality_of("avg_trade_size_percentile") is DataQuality.GOOD
    assert not has_note(one, "average trade size in the window is identical")

    flipped_buy = np.full(n, 300.0)
    flipped_sell = np.full(n, 60.0)
    flipped_buy[-tail:] = 60.0
    flipped_sell[-tail:] = 300.0
    two = comp.compute(
        view_at(
            flat_tape(
                n=n, buy=flipped_buy, sell=flipped_sell, buy_trades=6.0, sell_trades=4.0
            )
        )
    )
    assert two.values["avg_trade_size_percentile"] == UNAVAILABLE_PERCENTILE
    assert two.quality_of("avg_trade_size_percentile") is DataQuality.DEGRADED
    assert note_with(two, "average trade size in the window is identical")
    # 360 lots over 10 prints on every bar of the window, both sides summed.
    assert (flipped_buy + flipped_sell).tolist() == [360.0] * n


def test_the_tail_moment_ranks_the_pooled_maximum_and_not_the_newest_row():
    """SURVIVING MUTATION: ranking `windows.max_trade[-1]` instead of
    `pooled[-1]` passed all 83 tests above.

    `test_the_tail_moment_is_the_rank_of_the_pooled_maximum_print` uses a
    strictly rising print size, where the newest ten-bar maximum IS the newest
    row, so the pooling is invisible. The point of pooling is that a large print
    five bars ago is still the largest print of the last ten bars, and the tape
    below says so: every bar carries a one-lot maximum except a 400-lot print
    twenty bars back and a 500-lot print five bars back.

    Over the 129-row window `rolling_max(..., 10)` gives 120 pools: the five
    ending at or after the 500-lot print hold 500, the ten covering the 400-lot
    print hold 400, and the remaining 105 hold 1.0. The newest pool holds 500,
    the maximum of its own history, so the rank is 1.0 and the event fires.
    Ranking the newest ROW's one lot instead would find 105 of 120 at or below
    it -- a rank of 0.875, below the 0.90 threshold, and no event."""
    n = 170
    sizes = np.full(n, 1.0)
    sizes[-20] = 400.0
    sizes[-5] = 500.0
    vector = computer().compute(view_at(flat_tape(n=n, max_trade=sizes)))

    pooled = rolling_max(sizes[-TICK_ROWS:], POOL)
    assert pooled.size == SIZE_LOOKBACK
    assert int(np.count_nonzero(pooled == 500.0)) == 5
    assert int(np.count_nonzero(pooled == 400.0)) == 10
    assert int(np.count_nonzero(pooled == 1.0)) == 105
    assert float(pooled[-1]) == 500.0

    assert vector.values["max_trade_size_percentile"] == 1.0
    assert vector.values["large_trade_event"] == 1.0
    assert vector.quality_of("max_trade_size_percentile") is DataQuality.GOOD

    newest_row_rank = int(np.count_nonzero(pooled <= sizes[-1])) / pooled.size
    assert newest_row_rank == pytest.approx(0.875)
    assert newest_row_rank < LARGE_PERCENTILE
    assert vector.values["max_trade_size_percentile"] != pytest.approx(newest_row_rank)


def test_both_availability_floors_are_exact_with_no_slack_below_them():
    """SURVIVING MUTATIONS: loosening either floor by 0.05 passed all 83 tests
    above. Both floors were pinned AT the configured value and from well below
    it -- 0.50 coverage against a 0.60 floor, 0.40 of the weight against 0.50 --
    and a gate pinned only at its boundary and at a distant point is a gate
    whose boundary can move by a few percent unnoticed. That matters here
    because both floors exist to decide between a measurement and a refusal.

    Coverage: 360 + 239 classified against 401 unclassified is 599/1000 = 0.599,
    one thousandth below the floor, and must still disable.

    The weight floor needs a non-default configuration to be pinned from just
    below, which is itself worth recording: the attainable fractions are the
    subset sums of the five weights, and the only value in [0.45, 0.50) is
    0.25 + 0.20 = 0.45, which cannot occur -- dropping `cvd_slope` requires the
    newest twenty rows to carry no classified volume, and that makes the pooled
    average trade size a live distribution, so the trade-size term survives
    whenever `cvd_slope` dies. At the default floor the two gates are therefore
    indistinguishable; at a floor of 0.56 the reachable 0.55 separates them."""
    n = 170
    just_below = computer().compute(
        view_at(flat_tape(n=n, buy=360.0, sell=239.0, unclassified=401.0))
    )
    assert 599 / 1000 < MIN_COVERAGE
    note = assert_fully_unavailable(just_below)
    assert "classification coverage 0.5990" in note
    assert "min_classification_coverage 0.6000" in note

    buy = np.full(n, 60.0)
    sell = np.full(n, 40.0)
    buy[-CVD_LOOKBACK:] = 0.0
    sell[-CVD_LOOKBACK:] = 0.0
    tick_kwargs = dict(
        buy=buy,
        sell=sell,
        buy_trades=None,
        sell_trades=None,
        max_trade=np.arange(float(n)) + 1.0,
    )
    # 1.00 - 0.25 (cvd, no volume to scale the slope) - 0.20 (aggression, no
    # counts) = 0.55, with absorption and the trade-size tail still measured.
    at_default = computer().compute(view_at(flat_tape(n=n, **tick_kwargs)))
    assert at_default.values["order_flow_available"] == 1.0
    assert at_default.values["order_flow_available_weight"] == pytest.approx(0.55)
    assert at_default.values["order_flow_available_weight"] > MIN_AVAILABLE_FRACTION

    strict = computer(min_available_weight_fraction=0.56).compute(
        view_at(flat_tape(n=n, **tick_kwargs))
    )
    strict_note = assert_fully_unavailable(strict)
    assert "only 0.5500 of the term weight has real inputs" in strict_note
    assert "floor 0.5600" in strict_note


def test_the_audit_gate_is_blind_to_a_proxy_substitution_which_is_the_point():
    """GUARDS THE GUARD, for the half of the Phase 4 gate the audit cannot hold.

    Section 13's gate is "Audit passes; no proxy substitution", and this file's
    header claims the two clauses are independent: a proxy reads only past data,
    so the lookahead harness would pass it, and the refusal therefore has to be
    tested behaviourally. That is a claim about the harness, and until it is
    demonstrated it is only a claim -- so the cheater is built.

    `_SubstitutesAProxy` does exactly what section 5 names as the known-bad
    estimator. On a bars-only dataset it signs bar volume by the close's
    location within each bar's range, cumulates it into a CVD path, squashes the
    slope, derives an aggression ratio from it and reports the component
    AVAILABLE with DEGRADED quality and an honest note. It declares its warmup
    truthfully and reads nothing after `now`, so it is a pure function of the
    past -- which is all `validation/lookahead.py` checks.

    The harness returns ZERO findings and `assert_no_lookahead` is clean. That
    is the finding: "Audit passes" is satisfied by a component that has
    substituted a proxy for all 25 points, and the only thing standing between
    this module and that outcome is
    `test_bars_only_disables_the_component_and_substitutes_nothing` and the
    tapes that go with it. The three cheaters above prove the harness can fail;
    this one marks the edge of what it can see."""

    class _SubstitutesAProxy(OrderFlowFeatures):
        name = "orderflow_proxy"

        @property
        def required_feeds(self):
            return frozenset({Feed.BARS})       # the tick feed made optional

        def compute(self, view):
            vector = super().compute(view)
            if vector.values["order_flow_available"] == 1.0:
                return vector
            if view.bar_count() < self.warmup_bars:
                return vector
            rows = TICK_ROWS
            highs, lows = view.highs(rows), view.lows(rows)
            closes, volumes = view.closes(rows), view.volumes(rows)
            span = np.where(highs - lows > 0.0, highs - lows, 1.0)
            # Bar volume signed by the close's location in the range: two of the
            # five constructions `REFUSED_PROXY_CONSTRUCTIONS` names, combined.
            signed = volumes * (2.0 * (closes - lows) / span - 1.0)
            scale = max(float(volumes.mean()), 1.0)
            path = np.cumsum(signed[-CVD_LOOKBACK:])
            x = np.arange(path.size, dtype=np.float64)
            x = x - x.mean()
            slope = float(np.dot(x, path - path.mean()) / np.dot(x, x))
            delta = float(signed[-SMOOTHING:].mean())
            values = {key: 0.0 for key in self.keys}
            values["signed_delta_contracts"] = delta
            values["signed_delta"] = float(np.tanh(delta / scale))
            values["cvd_slope"] = float(np.tanh(slope / scale))
            values["aggression_ratio"] = 0.5 + 0.5 * values["signed_delta"]
            values["order_flow_magnitude"] = abs(values["signed_delta"]) * 0.5
            values["order_flow_direction"] = float(np.sign(values["signed_delta"]))
            values["order_flow_available"] = 1.0
            values["order_flow_available_weight"] = 1.0
            values["classification_coverage"] = 1.0
            return self._vector(
                view,
                values,
                quality=DataQuality.DEGRADED,
                notes=("orderflow_proxy: delta estimated from bar volume (DEGRADED)",),
            )

    data = synthetic_data(include_ticks=False)
    assert data.ticks == {}
    config, feature_config = order_flow(), features()

    # It really does substitute: a reading, not a refusal, on bars alone.
    proxy = _SubstitutesAProxy(config, feature_config)
    emitted = proxy.compute(view_at(data, len(data.primary_bars) - 1))
    assert emitted.values["order_flow_available"] == 1.0
    assert emitted.values["signed_delta_contracts"] != 0.0
    assert emitted.quality is DataQuality.DEGRADED

    # And the harness sees nothing wrong with it.
    result = audit_computer(
        data=data,
        factory=lambda d: _SubstitutesAProxy(config, feature_config),
        sample=40,
    )
    assert result.passed, result.summary()
    assert result.findings == ()
    assert result.bars_checked > 0
    assert_no_lookahead([result])               # clean, on a pure proxy

    # The real computer, on the identical dataset, refuses.
    honest = OrderFlowFeatures(config, feature_config).compute(
        view_at(data, len(data.primary_bars) - 1)
    )
    note = assert_fully_unavailable(honest)
    assert "tick-aggregate feed is absent" in note

    # Why no caller above could have told the two apart: the cheat emits the
    # same fifteen keys, in the same bounds, with a quality and a note that read
    # as an honest degradation. Only the refusal itself distinguishes them.
    assert set(emitted.values) == set(honest.values) == set(KEYS)
    for key in UNIT_KEYS:
        assert 0.0 <= emitted.values[key] <= 1.0
    for key in SIGNED_KEYS:
        assert -1.0 <= emitted.values[key] <= 1.0
    assert emitted.warmup_complete and not honest.warmup_complete


def test_which_absorption_measurement_goes_in_which_band():
    """SURVIVING MUTATIONS: swapping the two configured bands, and swapping the
    two MEASUREMENTS between them, both passed all 87 tests above.

    `absorption_max_displacement_atr` and `absorption_level_window_atr` both
    default to 0.25, so at the defaults the two bands are interchangeable and
    nothing can tell which measurement is compared against which. Every test
    above also uses a tape where the net displacement and the within-window
    excursion are BOTH zero (a flat tape, where both factors are exactly 1) or
    where the level band is structurally unreachable, so the distinction the
    module docstring argues for at length -- "`quiet` is the NET change and
    reads its reference close from before the window; `flat` is the whole
    excursion WITHIN it... a round trip is absorption, a drift is not" -- was
    entirely untested.

    This pins it with distinct bands (0.5 ATR for displacement, 2.0 for the
    level window) and two tapes with the IDENTICAL within-window excursion of
    1.0 point against an ATR of 2.0:

      * a round trip, up a point and back, ending on its own reference close:
        displacement 0 so `quiet = 1`, excursion 0.5 ATR so
        `flat = 1 - 0.5 / 2.0 = 0.75`, and `absorption = 0.75 ** (1/3)`.
      * a drift, up a point monotonically: displacement 0.5 ATR, exactly the
        configured band, so `quiet = 0` and the term is 0 however heavy the
        flow -- a drive, not absorption.

    Swapping the bands turns the round trip to 0 (its excursion would be
    measured against the 0.5 band); swapping the measurements does the same.
    Both are therefore caught, and the module's stated asymmetry is now a
    tested property rather than a paragraph."""
    n = 170
    comp = computer(
        absorption_max_displacement_atr=0.5, absorption_level_window_atr=2.0
    )
    heavy = np.full(n, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    tick_kwargs = dict(
        buy=100.0 + heavy, sell=100.0, max_trade=np.arange(float(n)) + 1.0
    )

    round_trip = np.full(n, 100.0)
    round_trip[-ABSORPTION_LOOKBACK:] = [
        100.5, 101.0, 100.5, 100.0, 100.5, 101.0, 100.5, 100.0, 100.5, 100.0
    ]
    drift = np.full(n, 100.0)
    drift[-ABSORPTION_LOOKBACK:] = np.linspace(100.0, 101.0, ABSORPTION_LOOKBACK)

    for label, mids, expected, direction in (
        ("round trip", round_trip, (1.0 * 1.0 * 0.75) ** (1 / ABSORPTION_TERM_COUNT), -1.0),
        ("drift", drift, 0.0, 0.0),
    ):
        view = view_at(tape(mids, **tick_kwargs))
        assert local_atr(view) == 2.0, label
        window = view.closes(ABSORPTION_LOOKBACK + 1)
        # The premise both tapes share: the same excursion inside the window.
        assert float(np.ptp(window[1:])) == pytest.approx(1.0), label
        vector = comp.compute(view)
        assert vector.values["absorption"] == pytest.approx(expected), label
        assert vector.values["absorption_direction"] == direction, label

    assert (0.75) ** (1 / ABSORPTION_TERM_COUNT) == pytest.approx(0.9085602964160)
    # What a swap would have produced for the round trip: its 0.5 ATR excursion
    # measured against the 0.5 ATR displacement band is exactly 0.
    assert inside_band(0.5, 0.5) == 0.0


def test_the_local_atr_window_is_atr_period_and_not_the_whole_bar_window():
    """SURVIVING MUTATION: `rows = self._bar_rows` in place of
    `self._atr_period + 1` passed all 87 tests above.

    At the defaults `_bar_rows` IS `atr_period + 1`, so the mutation is the
    identity; and `test_the_absorption_geometry_reads_one_bar_more_than_its_
    lookback` uses a tape whose every true range is 2.0, so the window length
    could not matter there either. Both have to be false at once for the ATR's
    own window to be observable.

    `atr_period=5` with `absorption_lookback_bars=10` makes the bar window
    eleven rows and the ATR window six. The bar RANGE is five points on rows
    -10 to -7 and one point on the newest six, so the five true ranges the ATR
    may read are all 2.0 and the ten it must not read are
    `[10, 10, 10, 10, 2, 2, 2, 2, 2, 2]`. The Wilder figure over the wrong
    window is 4.0972, not 2.0, and with the reference close 0.25 points away:

        correct: quiet = 1 - (0.25 / 2.0000) / 0.25 = 0.5     -> 0.793700...
        wrong:   quiet = 1 - (0.25 / 4.0972) / 0.25 = 0.7559  -> 0.910...
    """
    n = 170
    comp = computer(atr_period=5, absorption_lookback_bars=ABSORPTION_LOOKBACK)
    mids = np.full(n, 100.0)
    mids[-(ABSORPTION_LOOKBACK + 1)] = 100.25
    span = np.full(n, 1.0)
    span[-ABSORPTION_LOOKBACK:-6] = 5.0
    heavy = np.full(n, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0

    bars = BarSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n),
        interval_seconds=INTERVAL,
        columns={
            "open": mids.copy(),
            "high": mids + span,
            "low": mids - span,
            "close": mids.copy(),
            "volume": np.full(n, 1000.0),
        },
    )
    ticks = tick_series(
        n, buy=100.0 + heavy, sell=100.0, max_trade=np.arange(float(n)) + 1.0
    )
    view = view_at(symbol_data(bars, ticks))

    assert local_atr(view, period=5) == 2.0
    wrong_window = true_range(view.highs(11), view.lows(11), view.closes(11))
    assert wrong_window.tolist() == [10.0] * 4 + [2.0] * 6
    wrong_atr = float(wilder_atr_series(wrong_window, 5)[-1])
    assert wrong_atr == pytest.approx(4.09715199999)

    vector = comp.compute(view)
    expected = (1.0 * 0.5 * 1.0) ** (1 / ABSORPTION_TERM_COUNT)
    assert vector.values["absorption"] == pytest.approx(expected)
    assert vector.values["absorption"] == pytest.approx(0.7937005259841)

    wrong_quiet = 1.0 - (0.25 / wrong_atr) / MAX_DISPLACEMENT_ATR
    assert wrong_quiet == pytest.approx(0.7559280202)
    assert vector.values["absorption"] != pytest.approx(
        wrong_quiet ** (1 / ABSORPTION_TERM_COUNT)
    )


def test_the_diagnostic_fields_the_term_classes_carry_are_actually_asserted():
    """The term classes say of their extra fields: "They are reachable from a
    test, which is where they are asserted." That was not true of a single one
    of them -- `_AbsorptionTerm.effort`, `.quiet`, `.flat`,
    `_AggressionTerm.trades` and `_DeltaTerm.measured` were all set and never
    read, by the module or by this file. This test makes the sentence true, and
    it is the only place `_DeltaTerm.measured` is observable at all: it is not
    emitted as a feature, and the delta term is always `available`, so no
    output distinguishes a degenerate window's stated 0.0 from a measured
    median rank. The note does; the quality does not.

    The three absorption factors matter beyond tidiness: `absorption` is their
    geometric mean, so any pair of factors with the same product is
    indistinguishable from the emitted value alone."""
    n = 170
    comp = computer(
        absorption_max_displacement_atr=0.5, absorption_level_window_atr=2.0
    )
    heavy = np.full(n, 1.0)
    heavy[-ABSORPTION_LOOKBACK:] = 100.0
    mids = np.full(n, 100.0)
    mids[-ABSORPTION_LOOKBACK:] = [
        100.5, 101.0, 100.5, 100.0, 100.5, 101.0, 100.5, 100.0, 100.5, 100.0
    ]
    view = view_at(
        tape(
            mids,
            buy=100.0 + heavy,
            sell=100.0,
            buy_trades=6.0,
            sell_trades=4.0,
            max_trade=np.arange(float(n)) + 1.0,
        )
    )
    windows = comp._read_windows(view, [])
    assert not isinstance(windows, str), windows

    absorption = comp._absorption(windows, [])
    assert absorption.effort == 1.0          # the heaviest window sum of its 60
    assert absorption.quiet == 1.0           # ends on its own reference close
    assert absorption.flat == pytest.approx(0.75)
    assert absorption.strength == pytest.approx(
        (absorption.effort * absorption.quiet * absorption.flat)
        ** (1 / ABSORPTION_TERM_COUNT)
    )

    aggression = comp._aggression_ratio(windows, [])
    assert aggression.trades == pytest.approx(POOL * 10.0)   # 6 + 4 a bar, pooled
    assert aggression.trades > MIN_TRADES
    assert aggression.ratio == pytest.approx(0.6)

    # `measured` is True for a real rank and False for a stated 0.0, and the
    # emitted vector cannot tell the two apart.
    delta = comp._signed_delta(windows, [])
    assert delta.measured is True
    assert delta.available is True

    flat_windows = comp._read_windows(view_at(flat_tape(n=n)), [])
    assert not isinstance(flat_windows, str)
    dead = comp._signed_delta(flat_windows, [])
    assert dead.measured is False
    assert dead.available is True            # a balanced tape is a measurement
    assert dead.signed == 0.0
    emitted = computer().compute(view_at(flat_tape(n=n)))
    assert emitted.values["signed_delta"] == 0.0
    assert emitted.quality_of("signed_delta") is DataQuality.GOOD
    assert note_with(emitted, "smoothed delta observations equals")
