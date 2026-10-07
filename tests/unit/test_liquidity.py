"""Adversarial tests for the liquidity features.

Two risks dominate this module, and the tests are organized around them.

**The degradation path is the point.** ARCHITECTURE.md section 5 gives
liquidity L1 quotes as its ideal feed and bar volume as its degraded one,
and the brief for this module forbids a silent zero and an estimate dressed
as a measurement. So the no-quote path is tested as a *whole contract*:
every spread-derived feature reports its named fallback, the note naming
each fallback is present, the per-key quality drops to DEGRADED, and
`liquidity_score` is capped -- and the cap is shown to be *binding*, by
computing what the uncapped score would have been.

**`relative_volume` is the feature most worth getting right.** Intraday
volume is U-shaped, so a flat percentile calls every opening bar unusual and
the component becomes a clock. `test_relative_volume_is_time_of_day_aware`
is the test that decides whether this module is worth having: the same
absolute volume (1000) is placed at a session open and at midday on a
hand-built U-shaped profile, and `volume_percentile` is required to be 1.0
in *both* cases -- it genuinely cannot tell them apart -- while
`relative_volume` is required to be exactly 1.0 at the open and exactly 10/3
at midday.

The geometry that makes the hand-computed values exact: 13 bars per NQ RTH
session (1800-second bars over 09:30-16:00) and `time_of_day_buckets = 13`,
so each bucket holds exactly one bar per session and bar `i` of the dataset
falls in bucket `i % 13`. With `volume_percentile_lookback = 12` the profile
window is 12 x 13 = 156 bars, which is exactly twelve observations per
bucket. Every median below is therefore a median of twelve numbers the test
chose.

Builders are local on purpose: file ownership, and so a failure localizes
here rather than to a shared fixture.
"""

from __future__ import annotations

import inspect
import math
from datetime import date, datetime, timedelta, timezone
from typing import get_type_hints
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from flow_model.config.schema import FeatureConfig
from flow_model.core.enums import DataQuality, Feed, InstrumentType
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.market_view import MarketView
from flow_model.data.series import NS_PER_SECOND, BarSeries, QuoteSeries, from_ns, to_ns
from flow_model.data.store import SymbolData
from flow_model.data.synthetic import SyntheticConfig, SyntheticMarketGenerator
from flow_model.features.base import FeatureBundle, FeatureError
from flow_model.features.liquidity import (
    MIN_BUCKET_SAMPLES,
    MIN_SPREAD_SAMPLE_SHARE,
    NO_QUOTE_SCORE_CAP,
    RELATIVE_VOLUME_FLOOR,
    TIGHTNESS_WITHOUT_QUOTES,
    VOLUME_TREND_REFERENCE_CHANGE,
    LiquidityFeatures,
    bar_open_minutes_of_day,
    bucket_medians,
    least_squares_slope,
    session_span_minutes,
    signed_squash,
    time_of_day_buckets,
    utc_offset_minutes,
)
from flow_model.validation.lookahead import assert_no_lookahead, audit_computer

SYMBOL = "NQ"
TZ = ZoneInfo("America/New_York")

#: 1800-second bars over the 09:30-16:00 NQ session: 390 / 30 = 13 per day.
INTERVAL = 1800
BARS_PER_SESSION = 13
BUCKETS = 13
LOOKBACK = 12

#: 12 x 13: twelve observations in each of thirteen time-of-day buckets.
PROFILE_BARS = LOOKBACK * BUCKETS

BASE = 100.0
TICK = 0.25
POINT_VALUE = 5.0 / 0.25  # tick_value / tick_size

#: A U-shaped session volume profile, one value per bar of the session. The
#: open is the busiest bar and the midday trough is flat, which is the shape
#: `relative_volume` exists to divide out.
U_PROFILE = (1000.0, 700.0, 500.0, 400.0, 350.0, 300.0, 300.0,
             300.0, 350.0, 400.0, 500.0, 700.0, 900.0)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def nq_spec(**overrides) -> InstrumentSpec:
    """An NQ-shaped spec. tick 0.25, $5/tick, 1-tick typical spread, RTH session."""
    fields = dict(
        symbol=SYMBOL,
        instrument_type=InstrumentType.FUTURE,
        tick_size=TICK,
        tick_value=5.0,
        typical_spread_ticks=1.0,
        min_spread_ticks=1.0,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
    )
    fields.update(overrides)
    return InstrumentSpec(**fields)


def index_spec() -> InstrumentSpec:
    """A cash index: no RTH window at all, like SPX in defaults.yaml."""
    return InstrumentSpec(
        symbol="SPX",
        instrument_type=InstrumentType.INDEX,
        tick_size=0.01,
        tick_value=0.01,
        tradable=False,
    )


def config(**overrides) -> FeatureConfig:
    fields = dict(
        volume_percentile_lookback=LOOKBACK,
        spread_percentile_lookback=LOOKBACK,
        time_of_day_buckets=BUCKETS,
    )
    fields.update(overrides)
    return FeatureConfig(**fields)


def session_stamps(
    sessions: int,
    bars: int = BARS_PER_SESSION,
    interval: int = INTERVAL,
    first_day: date = date(2024, 1, 2),
) -> np.ndarray:
    """Bar-CLOSE stamps for `sessions` consecutive weekday RTH sessions.

    Built from a real 09:30 America/New_York open converted to UTC, so the
    minutes-into-session arithmetic under test is exercised against genuine
    timestamps rather than a grid that happens to line up.
    """
    out: list[int] = []
    day, produced = first_day, 0
    while produced < sessions:
        if day.weekday() < 5:
            local_open = datetime(day.year, day.month, day.day, 9, 30, tzinfo=TZ)
            base = to_ns(local_open.astimezone(timezone.utc))
            out.extend(base + k * interval * NS_PER_SECOND for k in range(1, bars + 1))
            produced += 1
        day += timedelta(days=1)
    return np.array(out, dtype=np.int64)


def make_bars(
    stamps: np.ndarray,
    volumes: np.ndarray,
    close: float = BASE,
    symbol: str = SYMBOL,
    interval: int = INTERVAL,
) -> BarSeries:
    """Flat OHLC at `close` with the given volumes, so price never confounds."""
    flat = np.full(stamps.size, float(close), dtype=np.float64)
    return BarSeries(
        symbol=symbol,
        ts_ns=stamps,
        interval_seconds=interval,
        columns={
            "open": flat, "high": flat, "low": flat, "close": flat,
            "volume": np.asarray(volumes, dtype=np.float64),
        },
    )


def make_quotes(
    stamps: np.ndarray,
    spread_ticks: np.ndarray,
    bid_size: np.ndarray | float = 20.0,
    ask_size: np.ndarray | float = 20.0,
    symbol: str = SYMBOL,
    tick: float = TICK,
) -> QuoteSeries:
    """One quote per bar. `spread_ticks` may be negative, giving a crossed book."""
    n = stamps.size
    spread = np.asarray(spread_ticks, dtype=np.float64)
    bid = np.full(n, BASE, dtype=np.float64)
    return QuoteSeries(
        symbol=symbol,
        ts_ns=stamps,
        columns={
            "bid": bid,
            "ask": bid + spread * tick,
            "bid_size": np.broadcast_to(np.asarray(bid_size, dtype=np.float64), (n,)).copy(),
            "ask_size": np.broadcast_to(np.asarray(ask_size, dtype=np.float64), (n,)).copy(),
        },
    )


def make_data(
    bars: BarSeries, quotes: QuoteSeries | None = None, symbol: str = SYMBOL
) -> SymbolData:
    return SymbolData(
        symbol=symbol, primary_interval=INTERVAL, bars={INTERVAL: bars}, quotes=quotes
    )


def view_at(data: SymbolData, index: int) -> MarketView:
    stamp = int(data.primary_bars.ts_ns[index])
    return MarketView(data, now=from_ns(stamp), now_ns=stamp)


def u_volumes(sessions: int) -> np.ndarray:
    """`sessions` repetitions of the U-shaped profile."""
    return np.tile(np.array(U_PROFILE, dtype=np.float64), sessions)


def flat_dataset(
    sessions: int = 14, volume: float = 1000.0, spread_ticks: float = 1.0
) -> SymbolData:
    """Constant volume everywhere, so every bucket median is `volume` exactly."""
    stamps = session_stamps(sessions)
    volumes = np.full(stamps.size, float(volume), dtype=np.float64)
    quotes = make_quotes(stamps, np.full(stamps.size, float(spread_ticks)))
    return make_data(make_bars(stamps, volumes), quotes)


def ramp_dataset(rising: bool = True, sessions: int = 14) -> tuple[SymbolData, int]:
    """Flat volume, except the final 12 bars which ramp by 100 per bar.

    Returns `(data, index)` where `index` is the last bar. Every bucket then
    holds eleven observations of 1000 and one ramped observation above it, so
    every bucket median is exactly 1000 and the trailing twelve deflated
    volumes are exactly 1.0, 1.1, ... 2.1 (or the reverse).
    """
    stamps = session_stamps(sessions)
    volumes = np.full(stamps.size, 1000.0, dtype=np.float64)
    steps = np.arange(LOOKBACK, dtype=np.float64)
    ramp = 1000.0 + 100.0 * (steps if rising else steps[::-1])
    volumes[-LOOKBACK:] = ramp
    quotes = make_quotes(stamps, np.ones(stamps.size))
    return make_data(make_bars(stamps, volumes), quotes), stamps.size - 1


def computer(spec: InstrumentSpec | None = None, **config_overrides) -> LiquidityFeatures:
    return LiquidityFeatures(config(**config_overrides), spec or nq_spec())


def note_matching(vector, fragment: str) -> str | None:
    return next((note for note in vector.notes if fragment in note), None)


# ---------------------------------------------------------------------------
# the declared contract
# ---------------------------------------------------------------------------


def test_declares_its_contract():
    comp = computer()
    assert comp.name == "liquidity"
    assert comp.required_feeds == frozenset({Feed.BARS})
    assert comp.optional_feeds == frozenset({Feed.QUOTES})
    assert comp.keys == (
        "spread_ticks", "spread_percentile", "depth_imbalance", "volume_percentile",
        "relative_volume", "volume_trend", "dollar_volume",
        "participation_cost_ticks", "liquidity_score",
    )


def test_warmup_is_the_time_of_day_profile_window():
    """The profile needs `volume_percentile_lookback` observations in each of
    `time_of_day_buckets` buckets, and the buckets partition the session, so
    the window is their product. Under-declaring it means a median over two
    or three correlated observations per bucket."""
    comp = computer()
    assert comp.warmup_bars == PROFILE_BARS == 156
    assert computer(time_of_day_buckets=26, volume_percentile_lookback=60).warmup_bars == 1560
    # The spread lookback is a floor, never the binding term on defaults.
    assert computer(spread_percentile_lookback=5000).warmup_bars == 5000


def test_constructor_takes_configuration_only():
    """Local echo of the package-wide ban in test_feature_contracts.py. A
    computer that accepted a series could capture a full-sample statistic,
    which is the one leak the lookahead audit cannot see through."""
    parameters = inspect.signature(LiquidityFeatures.__init__).parameters
    assert list(parameters) == ["self", "config", "spec"]
    # `liquidity.py` uses postponed annotations, so `__annotations__` holds
    # strings; resolve them against the module globals before comparing.
    hints = get_type_hints(LiquidityFeatures.__init__)
    assert hints["config"] is FeatureConfig
    assert hints["spec"] is InstrumentSpec


def test_output_keys_match_the_declaration_exactly():
    data, index = ramp_dataset()
    vector = computer().compute(view_at(data, index))
    assert tuple(sorted(vector.values)) == tuple(sorted(computer().keys))
    assert all(math.isfinite(value) for value in vector.values.values())


def test_every_key_carries_a_quality():
    data, index = ramp_dataset()
    vector = computer().compute(view_at(data, index))
    assert set(vector.quality_by_key) == set(computer().keys)


def test_all_bounded_features_stay_in_range_across_a_session():
    """The five bounded features are what a component score may read. One
    excursion past the bound would let a single bar dominate a weighted sum."""
    data = flat_dataset(sessions=16)
    comp = computer()
    for index in range(comp.warmup_bars - 1, len(data.primary_bars)):
        values = comp.compute(view_at(data, index)).values
        assert 0.0 <= values["spread_percentile"] <= 1.0
        assert 0.0 <= values["volume_percentile"] <= 1.0
        assert 0.0 <= values["liquidity_score"] <= 1.0
        assert -1.0 <= values["depth_imbalance"] <= 1.0
        assert -1.0 <= values["volume_trend"] <= 1.0


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------


def test_not_ready_one_bar_before_warmup():
    """Off by one here produces a number from a half-filled profile that
    looks exactly like a feature."""
    data = flat_dataset(sessions=14)
    comp = computer()
    vector = comp.compute(view_at(data, comp.warmup_bars - 2))
    assert not vector.warmup_complete
    assert vector.quality is DataQuality.MISSING
    assert note_matching(vector, "warmup incomplete")


def test_ready_at_exactly_warmup():
    data = flat_dataset(sessions=14)
    comp = computer()
    vector = comp.compute(view_at(data, comp.warmup_bars - 1))
    assert vector.warmup_complete
    assert vector.values["relative_volume"] == pytest.approx(1.0)


def test_not_ready_without_a_bar_feed():
    empty = BarSeries(
        symbol=SYMBOL,
        ts_ns=np.zeros(0, dtype=np.int64),
        interval_seconds=INTERVAL,
        columns={name: np.zeros(0) for name in
                 ("open", "high", "low", "close", "volume")},
    )
    data = make_data(empty)
    vector = computer().compute(
        MarketView(data, now=datetime(2024, 1, 2, 15, tzinfo=timezone.utc))
    )
    assert not vector.warmup_complete
    assert note_matching(vector, "bars feed absent")


def test_non_finite_volume_is_reported_not_ready():
    """`BarSeries` validates OHLC finiteness but not volume's, so a nan
    volume reaches this computer and would propagate into every ratio."""
    stamps = session_stamps(14)
    volumes = np.full(stamps.size, 1000.0)
    volumes[-3] = np.nan
    data = make_data(make_bars(stamps, volumes))
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert not vector.warmup_complete
    assert note_matching(vector, "non-finite")


# ---------------------------------------------------------------------------
# session geometry
# ---------------------------------------------------------------------------


def test_bar_open_minutes_is_the_session_open_for_the_first_bar():
    """The first bar of the session closes at 10:00 and OPENS at 09:30, so
    its minutes-of-day is 570. Bucketing on the close instead would leave
    bucket 0 empty and fold the last bar onto its neighbour."""
    stamps = session_stamps(2)
    minutes = bar_open_minutes_of_day(stamps, TZ, INTERVAL)
    assert minutes[0] == pytest.approx(9 * 60 + 30)
    assert minutes[1] == pytest.approx(10 * 60)
    assert minutes[BARS_PER_SESSION - 1] == pytest.approx(15 * 60 + 30)
    assert minutes[BARS_PER_SESSION] == pytest.approx(9 * 60 + 30)


def test_bar_open_minutes_is_dst_correct():
    """09:30 New York is 14:30 UTC in January and 13:30 UTC in July. A
    fixed offset would put every July bucket one hour out, and only those
    sessions would be wrong -- which no aggregate would reveal."""
    winter = bar_open_minutes_of_day(session_stamps(1, first_day=date(2024, 1, 3)), TZ, INTERVAL)
    summer = bar_open_minutes_of_day(session_stamps(1, first_day=date(2024, 7, 10)), TZ, INTERVAL)
    assert winter[0] == pytest.approx(570.0)
    assert summer[0] == pytest.approx(570.0)


def test_utc_offset_resolves_a_day_containing_a_dst_transition():
    """US spring-forward 2024 is 2024-03-10 at 02:00 local. Bars on both
    sides of it sit in one UTC day, so the per-day grouping must detect the
    transition and fall back to a per-instant lookup."""
    midnight = to_ns(datetime(2024, 3, 10, 5, tzinfo=timezone.utc)) // NS_PER_SECOND
    before = midnight + 3600          # 01:00 EST
    after = midnight + 4 * 3600       # 05:00 EDT (03:00 local + 1h)
    offsets = utc_offset_minutes(np.array([before, after], dtype=np.int64), TZ)
    assert offsets[0] == pytest.approx(-300.0)
    assert offsets[1] == pytest.approx(-240.0)


def test_bar_open_minutes_rejects_a_non_positive_interval():
    with pytest.raises(FeatureError, match="not positive"):
        bar_open_minutes_of_day(session_stamps(1), TZ, 0)


def test_session_span_minutes():
    assert session_span_minutes(nq_spec()) == pytest.approx(390.0)
    assert session_span_minutes(index_spec()) is None
    wrapping = nq_spec(rth=SessionWindow(name="ETH", start="18:00", end="17:00"))
    assert session_span_minutes(wrapping) == pytest.approx(1380.0)
    whole_day = nq_spec(rth=SessionWindow(name="ALL", start="18:00", end="18:00"))
    assert session_span_minutes(whole_day) == pytest.approx(1440.0)


def test_time_of_day_buckets_give_one_bar_per_bucket_here():
    """The geometry the hand-computed values rest on: 13 bars per session
    and 13 buckets, so bar `i` of the dataset is in bucket `i % 13`."""
    stamps = session_stamps(3)
    minutes = bar_open_minutes_of_day(stamps, TZ, INTERVAL)
    buckets = time_of_day_buckets(minutes, nq_spec(), BUCKETS)
    assert buckets.tolist() == [i % BARS_PER_SESSION for i in range(stamps.size)]


def test_time_of_day_buckets_separate_out_of_session_bars():
    """An overnight bar's volume is not comparable with a session bar's, so
    it gets its own bucket rather than being folded into the open."""
    stamps = session_stamps(1)
    extra = np.array([stamps[-1] + INTERVAL * NS_PER_SECOND], dtype=np.int64)
    minutes = bar_open_minutes_of_day(np.concatenate([stamps, extra]), TZ, INTERVAL)
    buckets = time_of_day_buckets(minutes, nq_spec(), BUCKETS)
    assert buckets[-1] == BUCKETS
    assert set(buckets[:-1].tolist()) == set(range(BUCKETS))


def test_time_of_day_buckets_without_an_rth_window_collapse_to_one():
    stamps = session_stamps(2)
    minutes = bar_open_minutes_of_day(stamps, TZ, INTERVAL)
    buckets = time_of_day_buckets(minutes, index_spec(), BUCKETS)
    assert buckets.tolist() == [0] * stamps.size


def test_time_of_day_buckets_rejects_a_bucket_count_below_one():
    with pytest.raises(FeatureError, match="at least 1"):
        time_of_day_buckets(np.array([570.0]), nq_spec(), 0)


# ---------------------------------------------------------------------------
# the helper functions
# ---------------------------------------------------------------------------


def test_bucket_medians_hand_computed():
    values = np.array([10.0, 100.0, 20.0, 200.0, 30.0, 300.0], dtype=np.float64)
    buckets = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    medians, counts = bucket_medians(values, buckets, 3)
    assert medians.tolist() == [20.0, 200.0, 0.0]
    assert counts.tolist() == [3, 3, 0]


def test_bucket_medians_distinguish_an_empty_bucket_from_a_zero_median():
    """Collapsing the two is how a thin bucket becomes a divide-by-zero."""
    medians, counts = bucket_medians(
        np.array([0.0, 0.0]), np.array([1, 1], dtype=np.int64), 3
    )
    assert medians.tolist() == [0.0, 0.0, 0.0]
    assert counts.tolist() == [0, 2, 0]


def test_least_squares_slope_is_exact_on_a_line():
    assert least_squares_slope(np.array([1.0, 3.0, 5.0, 7.0])) == pytest.approx(2.0)
    assert least_squares_slope(np.array([7.0, 5.0, 3.0, 1.0])) == pytest.approx(-2.0)
    assert least_squares_slope(np.array([4.0, 4.0, 4.0])) == pytest.approx(0.0)
    assert least_squares_slope(np.array([5.0])) == 0.0
    assert least_squares_slope(np.zeros(0)) == 0.0


def test_signed_squash_keeps_the_sign():
    """`base.squash` takes the magnitude by design, which is right for a
    magnitude and wrong for a direction: collapsing participation must not
    read the same as rising participation."""
    assert signed_squash(1.0, 1.0) == pytest.approx(math.tanh(1.0))
    assert signed_squash(-1.0, 1.0) == pytest.approx(-math.tanh(1.0))
    assert signed_squash(0.0, 1.0) == 0.0
    assert signed_squash(float("nan"), 1.0) == 0.0
    # tanh saturates at exactly 1.0 in float64 for any |x| beyond ~19, and
    # `squash` documents a closed [0, 1], so the endpoint is in contract.
    assert -1.0 <= signed_squash(-1e9, 1.0) <= -0.999
    assert signed_squash(1e9, 1.0) == -signed_squash(-1e9, 1.0)


# ---------------------------------------------------------------------------
# spread
# ---------------------------------------------------------------------------


def test_spread_ticks_is_the_quoted_spread_in_ticks():
    """bid 100.00, ask 100.50, tick 0.25 -> 2 ticks."""
    stamps = session_stamps(14)
    quotes = make_quotes(stamps, np.full(stamps.size, 2.0))
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)), quotes)
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["spread_ticks"] == pytest.approx(2.0)
    assert vector.quality_of("spread_ticks") is DataQuality.GOOD


def test_spread_ticks_falls_back_to_the_configured_typical_without_quotes():
    """A named configured fallback, a note that says so, and DEGRADED -- never
    a silent zero, which would read as "the spread is zero"."""
    data = flat_dataset()
    bars_only = make_data(data.primary_bars, quotes=None)
    spec = nq_spec(typical_spread_ticks=3.0, min_spread_ticks=1.0)
    vector = LiquidityFeatures(config(), spec).compute(
        view_at(bars_only, len(bars_only.primary_bars) - 1)
    )
    assert vector.values["spread_ticks"] == pytest.approx(3.0)
    assert vector.quality_of("spread_ticks") is DataQuality.DEGRADED
    note = note_matching(vector, "no quote feed")
    assert note is not None and "typical_spread_ticks=3" in note
    assert "not a measurement" in note


def test_spread_ticks_falls_back_on_a_crossed_quote():
    """`ask < bid` is a feed artifact, not a tighter market: a negative
    spread is no spread at all."""
    stamps = session_stamps(14)
    spreads = np.ones(stamps.size)
    spreads[-1] = -4.0
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, spreads))
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["spread_ticks"] == pytest.approx(1.0)  # the fallback
    assert vector.quality_of("spread_ticks") is DataQuality.DEGRADED
    assert note_matching(vector, "crossed")
    assert vector.values["liquidity_score"] <= NO_QUOTE_SCORE_CAP


def test_spread_ticks_reports_a_locked_quote_as_zero_with_a_note():
    """bid == ask is a real quoted state, so zero is the honest reading and
    the feature is still a measurement -- but it is worth naming."""
    stamps = session_stamps(14)
    spreads = np.ones(stamps.size)
    spreads[-1] = 0.0
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, spreads))
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["spread_ticks"] == 0.0
    assert vector.quality_of("spread_ticks") is DataQuality.GOOD
    assert note_matching(vector, "locked")


def test_spread_ticks_falls_back_on_a_non_finite_quote():
    stamps = session_stamps(14)
    spreads = np.ones(stamps.size)
    spreads[-1] = np.nan
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, spreads))
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["spread_ticks"] == pytest.approx(1.0)
    assert note_matching(vector, "non-finite side")


def test_spread_percentile_hand_computed():
    """Trailing twelve spreads are 1..11 ticks plus a current 3, so four of
    the twelve observations are at or below 3: 4 / 12 = 1/3."""
    stamps = session_stamps(14)
    spreads = np.ones(stamps.size)
    spreads[-LOOKBACK:] = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 3.])
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, spreads))
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["spread_ticks"] == pytest.approx(3.0)
    assert vector.values["spread_percentile"] == pytest.approx(4.0 / 12.0)


def test_spread_percentile_is_unavailable_without_quotes():
    data = flat_dataset()
    bars_only = make_data(data.primary_bars, quotes=None)
    vector = computer().compute(view_at(bars_only, len(bars_only.primary_bars) - 1))
    assert vector.values["spread_percentile"] == 0.5
    assert vector.quality_of("spread_percentile") is DataQuality.DEGRADED
    assert note_matching(vector, "spread_percentile reported as 0.5")


def test_spread_percentile_is_unavailable_when_too_few_quotes_are_usable():
    """Seven of the trailing twelve are crossed, leaving five usable against
    a floor of ceil(0.5 * 12) = 6. A percentile over five observations is a
    different statistic from the one the config asked for."""
    stamps = session_stamps(14)
    spreads = np.ones(stamps.size)
    spreads[-LOOKBACK:-LOOKBACK + 7] = -2.0
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, spreads))
    vector = computer().compute(view_at(data, stamps.size - 1))
    floor = math.ceil(MIN_SPREAD_SAMPLE_SHARE * LOOKBACK)
    assert floor == 6
    assert vector.values["spread_percentile"] == 0.5
    note = note_matching(vector, "usable of")
    assert note is not None and "only 5" in note


def test_spread_percentile_notes_a_partially_usable_window():
    """One crossed observation of twelve leaves eleven, above the floor, so
    the percentile is computed -- and the smaller sample is declared."""
    stamps = session_stamps(14)
    spreads = np.ones(stamps.size)
    spreads[-LOOKBACK] = -2.0
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, spreads))
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["spread_percentile"] == pytest.approx(1.0)
    assert vector.quality_of("spread_percentile") is DataQuality.GOOD
    assert note_matching(vector, "computed from 11 usable")


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------


def test_depth_imbalance_hand_computed():
    """bid_size 30, ask_size 10: (30 - 10) / 40 = 0.5."""
    stamps = session_stamps(14)
    data = make_data(
        make_bars(stamps, np.full(stamps.size, 1000.0)),
        make_quotes(stamps, np.ones(stamps.size), bid_size=30.0, ask_size=10.0),
    )
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["depth_imbalance"] == pytest.approx(0.5)
    assert vector.quality_of("depth_imbalance") is DataQuality.GOOD


def test_depth_imbalance_saturates_on_a_one_sided_book():
    stamps = session_stamps(14)
    data = make_data(
        make_bars(stamps, np.full(stamps.size, 1000.0)),
        make_quotes(stamps, np.ones(stamps.size), bid_size=5.0, ask_size=0.0),
    )
    assert computer().compute(view_at(data, stamps.size - 1)).values[
        "depth_imbalance"
    ] == pytest.approx(1.0)


def test_depth_imbalance_on_an_empty_book_is_an_undefined_zero():
    """An empty book is not a balanced book. The zero is a placeholder and
    the note says which of the two it is."""
    stamps = session_stamps(14)
    data = make_data(
        make_bars(stamps, np.full(stamps.size, 1000.0)),
        make_quotes(stamps, np.ones(stamps.size), bid_size=0.0, ask_size=0.0),
    )
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["depth_imbalance"] == 0.0
    assert vector.quality_of("depth_imbalance") is DataQuality.DEGRADED
    note = note_matching(vector, "book is empty")
    assert note is not None and "not balanced" in note


def test_depth_imbalance_is_zero_without_a_quote_feed():
    data = flat_dataset()
    bars_only = make_data(data.primary_bars, quotes=None)
    vector = computer().compute(view_at(bars_only, len(bars_only.primary_bars) - 1))
    assert vector.values["depth_imbalance"] == 0.0
    assert vector.quality_of("depth_imbalance") is DataQuality.DEGRADED
    assert note_matching(vector, "depth_imbalance reported as 0.0")


# ---------------------------------------------------------------------------
# volume: the percentile, and the time-of-day comparison
# ---------------------------------------------------------------------------


def test_volume_percentile_hand_computed():
    """Trailing twelve volumes 100, 200, ... 1200 with the current bar at
    1200: twelve of twelve are at or below it, so 1.0. Shift the current bar
    to 650 and seven of twelve are, so 7/12."""
    stamps = session_stamps(14)
    volumes = np.full(stamps.size, 1000.0)
    volumes[-LOOKBACK:] = np.arange(1, LOOKBACK + 1, dtype=np.float64) * 100.0
    data = make_data(make_bars(stamps, volumes), make_quotes(stamps, np.ones(stamps.size)))
    assert computer().compute(view_at(data, stamps.size - 1)).values[
        "volume_percentile"
    ] == pytest.approx(1.0)

    volumes[-1] = 650.0
    data = make_data(make_bars(stamps, volumes), make_quotes(stamps, np.ones(stamps.size)))
    assert computer().compute(view_at(data, stamps.size - 1)).values[
        "volume_percentile"
    ] == pytest.approx(7.0 / 12.0)


def test_volume_percentile_is_pinned_to_zero_on_a_dead_tape():
    """"Fraction at or below" would rank a zero volume at 1.0 among other
    zeros, so a halted tape would read as maximum participation."""
    stamps = session_stamps(14)
    data = make_data(make_bars(stamps, np.zeros(stamps.size)),
                     make_quotes(stamps, np.ones(stamps.size)))
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["volume_percentile"] == 0.0
    assert note_matching(vector, "volume_percentile pinned to 0.0")


def test_relative_volume_is_one_on_a_stationary_profile():
    data = flat_dataset()
    vector = computer().compute(view_at(data, len(data.primary_bars) - 1))
    assert vector.values["relative_volume"] == pytest.approx(1.0)
    assert vector.quality_of("relative_volume") is DataQuality.GOOD


def test_relative_volume_is_time_of_day_aware():
    """The test that decides whether this module is worth having.

    The same absolute volume (1000) sits at a session open and at midday on
    a U-shaped profile. `volume_percentile` is 1.0 for BOTH -- it ranks the
    bar against the clock and cannot tell them apart -- while
    `relative_volume` is exactly 1.0 at the open (ordinary for the open) and
    exactly 1000 / 300 = 10/3 at midday (three and a third times normal for
    that bucket). A liquidity component fed the percentile alone is a clock.
    """
    sessions = 14
    stamps = session_stamps(sessions)
    volumes = u_volumes(sessions)
    midday_index = BARS_PER_SESSION * 12 + 6
    volumes[midday_index] = 1000.0
    data = make_data(make_bars(stamps, volumes), make_quotes(stamps, np.ones(stamps.size)))
    comp = computer()

    open_index = BARS_PER_SESSION * 12
    at_open = comp.compute(view_at(data, open_index)).values
    at_midday = comp.compute(view_at(data, midday_index)).values

    assert volumes[open_index] == volumes[midday_index] == 1000.0
    assert at_open["volume_percentile"] == pytest.approx(1.0)
    assert at_midday["volume_percentile"] == pytest.approx(1.0)

    assert at_open["relative_volume"] == pytest.approx(1.0)
    assert at_midday["relative_volume"] == pytest.approx(10.0 / 3.0)
    assert at_midday["relative_volume"] > 3.0 * at_open["relative_volume"]


def test_the_time_of_day_profile_reproduces_the_u_shape():
    """Every bar of a stationary U-shaped session must read `relative_volume`
    1.0, from the 1000-share open to the 300-share trough. A flat comparison
    would instead run from above 2 down to below 0.7."""
    sessions = 15
    stamps = session_stamps(sessions)
    volumes = u_volumes(sessions)
    data = make_data(make_bars(stamps, volumes), make_quotes(stamps, np.ones(stamps.size)))
    comp = computer()
    session_start = BARS_PER_SESSION * 13
    ratios = [
        comp.compute(view_at(data, session_start + k)).values["relative_volume"]
        for k in range(BARS_PER_SESSION)
    ]
    assert ratios == pytest.approx([1.0] * BARS_PER_SESSION)


def test_relative_volume_falls_back_for_a_bucket_below_the_sample_floor():
    """An out-of-session bar lands in the out-of-session bucket, which a pure
    RTH feed never populates. A median of one observation is that
    observation, so the window median is used instead and the note says the
    comparison went flat."""
    stamps = session_stamps(14)
    extra = np.array([stamps[-1] + INTERVAL * NS_PER_SECOND], dtype=np.int64)
    stamps = np.concatenate([stamps, extra])
    volumes = np.full(stamps.size, 1000.0)
    volumes[-1] = 2500.0
    data = make_data(make_bars(stamps, volumes), make_quotes(stamps, np.ones(stamps.size)))
    vector = computer().compute(view_at(data, stamps.size - 1))

    assert vector.values["relative_volume"] == pytest.approx(2.5)
    note = note_matching(vector, f"floor {MIN_BUCKET_SAMPLES}")
    assert note is not None and "flat comparison" in note
    assert vector.quality_of("relative_volume") is DataQuality.DEGRADED
    assert vector.quality_of("volume_trend") is DataQuality.DEGRADED
    assert vector.quality_of("volume_percentile") is DataQuality.GOOD


def test_relative_volume_is_pinned_to_one_on_a_zero_volume_window():
    """Nothing to compare against, so 1.0 and say so -- not a division that
    `_vector` would reject as non-finite."""
    stamps = session_stamps(14)
    data = make_data(make_bars(stamps, np.zeros(stamps.size)),
                     make_quotes(stamps, np.ones(stamps.size)))
    vector = computer().compute(view_at(data, stamps.size - 1))
    assert vector.values["relative_volume"] == pytest.approx(1.0)
    assert note_matching(vector, "no volume anywhere")


def test_an_instrument_with_no_session_degrades_to_a_flat_comparison():
    """SPX in defaults.yaml declares no RTH window, so there is no session to
    bucket. The flat comparison is stated rather than silently performed."""
    stamps = session_stamps(14)
    volumes = np.full(stamps.size, 1000.0)
    volumes[-1] = 4000.0
    bars = make_bars(stamps, volumes, symbol="SPX")
    data = SymbolData(symbol="SPX", primary_interval=INTERVAL, bars={INTERVAL: bars})
    vector = LiquidityFeatures(config(), index_spec()).compute(
        view_at(data, stamps.size - 1)
    )
    assert vector.values["relative_volume"] == pytest.approx(4.0)
    note = note_matching(vector, "no RTH window")
    assert note is not None and "flat comparison" in note


# ---------------------------------------------------------------------------
# volume trend
# ---------------------------------------------------------------------------


def test_volume_trend_hand_computed_on_a_rising_ramp():
    """The twelve trailing deflated volumes are exactly 1.0, 1.1, ... 2.1, a
    perfect line of slope 0.1 per bar. The squash scale is one whole multiple
    of normal participation across the window, 1 / 11, so the feature is
    tanh(0.1 / (1/11)) = tanh(1.1)."""
    data, index = ramp_dataset(rising=True)
    values = computer().compute(view_at(data, index)).values
    scale = VOLUME_TREND_REFERENCE_CHANGE / (LOOKBACK - 1)
    assert values["relative_volume"] == pytest.approx(2.1)
    assert values["volume_trend"] == pytest.approx(math.tanh(0.1 / scale))
    assert values["volume_trend"] == pytest.approx(math.tanh(1.1))


def test_volume_trend_is_the_negative_of_itself_on_a_falling_ramp():
    """Rising and falling participation must be distinguishable; a
    magnitude-only squash would report them identically."""
    rising, index = ramp_dataset(rising=True)
    falling, _ = ramp_dataset(rising=False)
    comp = computer()
    up = comp.compute(view_at(rising, index)).values["volume_trend"]
    down = comp.compute(view_at(falling, index)).values["volume_trend"]
    assert up == pytest.approx(-down)
    assert up > 0.0 > down


def test_volume_trend_is_zero_on_a_stationary_tape():
    data = flat_dataset()
    assert computer().compute(view_at(data, len(data.primary_bars) - 1)).values[
        "volume_trend"
    ] == pytest.approx(0.0, abs=1e-12)


def test_volume_trend_is_bounded_by_an_extreme_ramp():
    stamps = session_stamps(14)
    volumes = np.full(stamps.size, 1000.0)
    volumes[-LOOKBACK:] = 1000.0 * np.power(10.0, np.arange(LOOKBACK))
    data = make_data(make_bars(stamps, volumes), make_quotes(stamps, np.ones(stamps.size)))
    trend = computer().compute(view_at(data, stamps.size - 1)).values["volume_trend"]
    assert 0.99 < trend <= 1.0


# ---------------------------------------------------------------------------
# dollar volume and participation cost
# ---------------------------------------------------------------------------


def test_dollar_volume_hand_computed():
    """volume x close x point_value = 1000 x 100 x 20 = 2_000_000."""
    data = flat_dataset()
    assert nq_spec().point_value == pytest.approx(POINT_VALUE)
    assert computer().compute(view_at(data, len(data.primary_bars) - 1)).values[
        "dollar_volume"
    ] == pytest.approx(1000.0 * BASE * POINT_VALUE)


def test_participation_cost_hand_computed():
    """A 2-tick spread at normal participation: 2/2 + 1/1 = 2 ticks."""
    stamps = session_stamps(14)
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, np.full(stamps.size, 2.0)))
    values = computer().compute(view_at(data, stamps.size - 1)).values
    assert values["relative_volume"] == pytest.approx(1.0)
    assert values["participation_cost_ticks"] == pytest.approx(2.0)


def test_participation_cost_scales_with_the_inverse_of_relative_volume():
    """At 2.1x normal participation the slippage term is 1 / 2.1 ticks."""
    data, index = ramp_dataset(rising=True)
    values = computer().compute(view_at(data, index)).values
    assert values["participation_cost_ticks"] == pytest.approx(0.5 + 1.0 / 2.1)


def test_participation_cost_floors_the_half_spread_at_one_increment():
    """A locked quote does not make taking liquidity free."""
    stamps = session_stamps(14)
    spreads = np.ones(stamps.size)
    spreads[-1] = 0.0
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, spreads))
    values = computer().compute(view_at(data, stamps.size - 1)).values
    assert values["spread_ticks"] == 0.0
    assert values["participation_cost_ticks"] == pytest.approx(0.5 + 1.0)


def test_participation_cost_saturates_on_a_vanishing_bar():
    """Without the floor the slippage term diverges as participation goes to
    zero; with it the term stops at ten minimum increments."""
    stamps = session_stamps(14)
    volumes = np.full(stamps.size, 1000.0)
    volumes[-1] = 1.0
    data = make_data(make_bars(stamps, volumes), make_quotes(stamps, np.ones(stamps.size)))
    values = computer().compute(view_at(data, stamps.size - 1)).values
    assert values["relative_volume"] < RELATIVE_VOLUME_FLOOR
    assert values["participation_cost_ticks"] == pytest.approx(
        0.5 + 1.0 / RELATIVE_VOLUME_FLOOR
    )


def test_participation_cost_scales_with_the_instrument():
    """GC's minimum increment is the base slippage unit, so the same relative
    conditions cost the same number of ticks on a different contract -- which
    is the point of deriving the unit from the spec rather than hardcoding it."""
    stamps = session_stamps(14)
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, np.full(stamps.size, 2.0)))
    gc_like = nq_spec(tick_size=0.1, tick_value=10.0,
                      typical_spread_ticks=2.0, min_spread_ticks=2.0)
    values = LiquidityFeatures(config(), gc_like).compute(
        view_at(data, stamps.size - 1)
    ).values
    # The quote is 0.5 price units wide, which is 5 ticks at a 0.1 tick size.
    assert values["spread_ticks"] == pytest.approx(5.0)
    assert values["participation_cost_ticks"] == pytest.approx(5.0 / 2.0 + 2.0 / 1.0)


# ---------------------------------------------------------------------------
# the component score
# ---------------------------------------------------------------------------


def test_liquidity_score_is_one_when_tight_and_maximally_busy():
    """A 1-tick NQ spread is the instrument's typical, so tightness is 1.0;
    the ramp puts the current deflated volume at the top of its window, so
    adequacy is 1.0. sqrt(1 x 1) = 1."""
    data, index = ramp_dataset(rising=True)
    values = computer().compute(view_at(data, index)).values
    assert values["spread_ticks"] == pytest.approx(1.0)
    assert values["liquidity_score"] == pytest.approx(1.0)


def test_liquidity_score_falls_when_the_spread_widens():
    """Eight times the typical spread gives tightness 1/8, so the score is
    sqrt(0.125) even though volume is unchanged: either term failing drags
    the geometric mean down."""
    data, index = ramp_dataset(rising=True)
    stamps = data.primary_bars.ts_ns
    spreads = np.ones(stamps.size)
    spreads[-1] = 8.0
    wide = make_data(data.primary_bars, make_quotes(stamps, spreads))
    values = computer().compute(view_at(wide, index)).values
    assert values["spread_ticks"] == pytest.approx(8.0)
    assert values["liquidity_score"] == pytest.approx(math.sqrt(0.125))


def test_liquidity_score_falls_when_participation_is_thin():
    """A falling ramp leaves the current bar at the bottom of its own
    deflated window: adequacy 1/12 against a tight spread."""
    data, index = ramp_dataset(rising=False)
    values = computer().compute(view_at(data, index)).values
    assert values["relative_volume"] == pytest.approx(1.0)
    assert values["liquidity_score"] == pytest.approx(math.sqrt(1.0 / 12.0))


def test_liquidity_score_is_relative_to_the_instruments_own_typical_spread():
    """"Tight" means tight for this contract. A 2-tick spread is ordinary on
    an instrument whose typical spread is 2 ticks and twice normal on one
    whose typical is 1, and the score must say so."""
    stamps = session_stamps(14)
    data = make_data(make_bars(stamps, np.full(stamps.size, 1000.0)),
                     make_quotes(stamps, np.full(stamps.size, 2.0)))
    index = stamps.size - 1
    narrow = LiquidityFeatures(config(), nq_spec(typical_spread_ticks=1.0))
    wide = LiquidityFeatures(config(), nq_spec(typical_spread_ticks=2.0))
    tight_for_it = wide.compute(view_at(data, index)).values["liquidity_score"]
    wide_for_it = narrow.compute(view_at(data, index)).values["liquidity_score"]
    assert tight_for_it == pytest.approx(wide_for_it * math.sqrt(2.0))


def test_liquidity_score_is_capped_without_quotes_and_the_cap_is_binding():
    """Half the component's evidence is missing, so half its range goes. The
    cap is shown to BITE: the same bar with quotes scores 1.0, and without
    them the uncapped formula would still have reached
    sqrt(TIGHTNESS_WITHOUT_QUOTES x 1) = 0.707."""
    data, index = ramp_dataset(rising=True)
    bars_only = make_data(data.primary_bars, quotes=None)
    comp = computer()

    with_quotes = comp.compute(view_at(data, index)).values
    without = comp.compute(view_at(bars_only, index))

    assert with_quotes["liquidity_score"] == pytest.approx(1.0)
    uncapped = math.sqrt(TIGHTNESS_WITHOUT_QUOTES * 1.0)
    assert uncapped == pytest.approx(0.7071, abs=1e-4)
    assert uncapped > NO_QUOTE_SCORE_CAP
    assert without.values["liquidity_score"] == pytest.approx(NO_QUOTE_SCORE_CAP)
    note = note_matching(without, "liquidity_score capped")
    assert note is not None and "0.50" in note
    assert "fallbacks rather than measurements" in note


def test_the_no_quote_path_reports_every_spread_feature_as_a_fallback():
    """The whole degradation contract in one place: named fallbacks, notes,
    per-key quality, the aggregate quality, and the cap."""
    data, index = ramp_dataset(rising=True)
    spec = nq_spec(typical_spread_ticks=2.0, min_spread_ticks=1.0)
    bars_only = make_data(data.primary_bars, quotes=None)
    vector = LiquidityFeatures(config(), spec).compute(view_at(bars_only, index))

    assert vector.values["spread_ticks"] == pytest.approx(2.0)
    assert vector.values["spread_percentile"] == 0.5
    assert vector.values["depth_imbalance"] == 0.0
    assert vector.values["liquidity_score"] == pytest.approx(NO_QUOTE_SCORE_CAP)
    # The cost estimate still exists, built from the fallback spread.
    assert vector.values["participation_cost_ticks"] == pytest.approx(
        2.0 / 2.0 + 1.0 / 2.1
    )

    for key in ("spread_ticks", "spread_percentile", "depth_imbalance",
                "participation_cost_ticks", "liquidity_score"):
        assert vector.quality_of(key) is DataQuality.DEGRADED, key
    for key in ("volume_percentile", "relative_volume", "volume_trend",
                "dollar_volume"):
        assert vector.quality_of(key) is DataQuality.GOOD, key
    assert vector.quality is DataQuality.DEGRADED
    assert vector.warmup_complete
    assert len(vector.notes) >= 3


def test_quote_path_is_good_quality_throughout():
    data, index = ramp_dataset(rising=True)
    vector = computer().compute(view_at(data, index))
    assert vector.quality is DataQuality.GOOD
    assert vector.notes == ()


# ---------------------------------------------------------------------------
# bundle integration
# ---------------------------------------------------------------------------


def test_bundle_reports_warmup_and_merges_the_keys():
    data = flat_dataset(sessions=14)
    bundle = FeatureBundle([computer()])
    assert bundle.warmup_bars == PROFILE_BARS
    early = bundle.compute(view_at(data, 10))
    assert not early.warmup_complete
    late = bundle.compute(view_at(data, len(data.primary_bars) - 1))
    assert late.warmup_complete
    assert set(late.values) == set(computer().keys)


# ---------------------------------------------------------------------------
# the lookahead audit
# ---------------------------------------------------------------------------


def synthetic_data(include_quotes: bool = True, interval: int = INTERVAL) -> SymbolData:
    """A generated dataset spanning well over 2x `warmup_bars`."""
    spec = nq_spec()
    generator = SyntheticMarketGenerator(
        SyntheticConfig(
            include_quotes=include_quotes, include_ticks=False, include_options=False
        ),
        seed=11,
    )
    dataset = generator.generate(SYMBOL, spec, date(2024, 1, 2), date(2024, 6, 1), interval)
    return SymbolData(
        symbol=SYMBOL,
        primary_interval=interval,
        bars={interval: dataset.bars},
        quotes=dataset.quotes,
    )


def test_lookahead_audit_passes_with_a_factory():
    """`factory=`, not an instance: that is the only form that catches a
    full-sample constant captured in `__init__`, which is the one leak the
    view-based firewall cannot close."""
    data = synthetic_data()
    spec, features = nq_spec(), config()
    assert len(data.primary_bars) > 2 * LiquidityFeatures(features, spec).warmup_bars
    result = audit_computer(
        data=data, factory=lambda d: LiquidityFeatures(features, spec), sample=40
    )
    assert result.passed, result.summary()
    assert result.bars_checked > 0
    assert result.keys_checked == tuple(sorted(LiquidityFeatures(features, spec).keys))
    assert_no_lookahead([result])


def test_lookahead_audit_passes_on_the_degraded_path_too():
    """The no-quote path has its own branches, including the cap, and an
    audit that only ever saw the quote path would not have exercised them."""
    data = synthetic_data(include_quotes=False)
    spec, features = nq_spec(), config()
    result = audit_computer(
        data=data, factory=lambda d: LiquidityFeatures(features, spec), sample=25
    )
    assert result.passed, result.summary()
    assert result.bars_checked > 0
    assert_no_lookahead([result])


def test_output_is_deterministic_across_rebuilt_computers():
    """A second computer built from the same config must agree bit for bit.
    `audit_computer` checks this per sampled bar; this states it directly."""
    data = synthetic_data()
    index = len(data.primary_bars) - 1
    first = LiquidityFeatures(config(), nq_spec()).compute(view_at(data, index))
    second = LiquidityFeatures(config(), nq_spec()).compute(view_at(data, index))
    assert first.values == second.values
    assert first.notes == second.notes


def test_features_are_unchanged_by_later_bars():
    """Truncation invariance, stated against this module directly rather than
    only through the harness: the value at bar i must not move when the bars
    after i are removed."""
    data = synthetic_data()
    comp = computer()
    index = comp.warmup_bars + 40
    cutoff = int(data.primary_bars.ts_ns[index])
    full = comp.compute(view_at(data, index)).values

    kept = data.primary_bars.visible_count(cutoff)
    truncated = SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: data.primary_bars.prefix(kept)},
        quotes=data.quotes.prefix(data.quotes.visible_count(cutoff)),
    )
    assert len(truncated.primary_bars) == index + 1
    short = comp.compute(view_at(truncated, index)).values
    assert full == short
