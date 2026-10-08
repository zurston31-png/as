"""Adversarial tests for the support/resistance level features.

`features/levels.py` is ARCHITECTURE.md section 14 in code: the translation of
the researcher's "use clean price action at major support and resistance
levels" into three bounded scalars, four gates, a determined stop and a
measured target. Nothing above it will ever check its arithmetic -- the
STRUCTURE component reads `structure_magnitude`, the signal engine reads
`structure_gate_passed`, and both are plausible-looking numbers in [0, 1]
whatever the terms underneath did. Six things in it can be silently wrong, and
the tests are organized around them.

**The six terms of significance (14.3).** `S` is a weighted sum of six
bounded terms, and a sum hides a dead term: if `s_volume` were always 0 or
`s_age` always 1, `S` would still move with the other five and still look
like a score. `test_significance_is_the_weighted_sum_of_all_six_terms`
therefore pins every term at once on a tape where each is forced by
construction -- `s_touch = 3/4` from three placed touches, `s_reject = 1.0`
from a 3.75-point departure against a 2.0 ATR, `s_volume = 4/8` from a volume
profile written out bucket by bucket below, `s_htf = 1/1` from one
higher-interval pivot, `s_age = 1.0` because `tau > 0`, `s_anchor = 1` from a
round number -- so `S = 0.825` is six numbers and not one. The two subtleties
14.3 calls out get their own tests: `s_age` decays *only* when `tau == 0`
(a touched level is confirmed, not stale), and a "distinct" touch needs a
departure of `d_sep * ATR` or `n_sep` bars, so one twenty-bar consolidation
must count four touches and not twenty.

**The five terms of cleanliness (14.4)**, the same way, and on the same tape.

**The deliberate tension between S and C (14.4).** The single most important
test in this file is `test_the_same_six_touches_are_clean_spread_out_and_filthy_packed`.
Section 14.4 claims `tau = 6` spread over 400 bars with `rho = 0` in the last
30 is *both* major and clean, while the same `tau = 6` packed into the last 30
bars is major and filthy and declined. If that does not hold, the two-score
split buys nothing and 14.7's third falsification condition is already met at
the arithmetic level, before any outcome data exists. The two tapes here carry
an identical `tau = 6` and an identical `S = 0.75`, and differ only in *when*
the touches happened: C is 0.80 and the gate passes, against C = 0.343 and
`level_gate_reason == GATE_CLEANLINESS`.

**The rejection requirement is a gate, not a term (14.5).** "The extreme
entered the zone AND the close returned outside it" is binary. The failure
mode is treating it as a scored term, which would turn every bar that merely
grazed a level into a rejection with a low `R` -- and `R` feeds a sum, so a
low `R` still trades. Three tests take the three ways to fail it: closing
back *inside* the band, never entering it, and switching the requirement off
in configuration (which the constructor refuses).

**`s_flow` is dropped WITHOUT redistribution (14.5).** With no tick feed,
`R` must be *lower* than it would be with flow agreement, not renormalized
back up to the same number. Renormalizing manufactures confidence from data
that does not exist, which is the whole content of
`strict_component_availability`. The test computes both numbers: 0.80 with
agreeing flow, 0.55 without, and 0.733 is what a renormalizing implementation
would have produced -- so the assertion is on the weight, not on a note.

**Setup selection is a measurement, not a parameter (14.6).** `achievable_rr`
is read off the structure, and the four bands are tested by moving *only* the
rejection bar's close over one fixed geometry: the zone stays at 100, the stop
stays at 99.25, the target stays at 104, and the setup class walks 3 -> 2 -> 1
-> declined as the entry rises. A geometry whose next level is 0.9R away must
decline the trade and keep reporting the real 104 target, not shrink it.

Three things in here are honest gaps rather than passing tests, and each says
so where it sits. `s_volume` is pinned only on the one tape whose volume
profile is written out bucket by bucket, because a percentile derived from the
implementation is not a check of it
(`test_s_volume_is_the_percentile_rank_of_the_hand_built_profile`). Seven keys
are unbounded by design, so over a long run only their finiteness, sign and
relations are checkable and their values are pinned on the hand-built tapes
alone (`test_every_emitted_key_is_finite_and_bounded_over_a_long_run`). And
`test_s_volume_cannot_tell_an_untraded_level_from_a_busy_one_here_reported`
pins a measurement defect this file REPORTS rather than repairs: on a sparse
volume profile an untraded level inherits the rank of every empty bucket, so
moving the whole region's volume away from a level does not change S at all.

Builders are local on purpose: file ownership, and so a failure localizes here
rather than to a shared fixture.

**The tape, and why its arithmetic is exact.** Every bar is
`high = mid + 1`, `low = mid - 1`, `close = mid`, and `mid` moves by at most
1.0 per bar. Then the true range is
`max(2, |mid_i + 1 - mid_{i-1}|, |mid_i - 1 - mid_{i-1}|) == 2.0` on *every*
bar, so the Wilder ATR is exactly 2.0 everywhere -- which makes the zone band
`0.25 * 2.0 = 0.5`, the touch separation `0.75 * 2.0 = 1.5`, the stop buffer
`0.25 * 2.0 = 0.5` and the proximity window `0.75 * 2.0 = 1.5` numbers written
in the tests rather than numbers read out of the implementation. The tests
assert the ATR is 2.0 before relying on it.
"""

from __future__ import annotations

import inspect
import math
from datetime import date, datetime, timedelta, timezone
from typing import get_type_hints
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.config.schema import FeatureConfig, StructureLevelConfig
from flow_model.core.enums import DataQuality, Feed, InstrumentType
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.base import SessionCalendarProtocol
from flow_model.data.calendar import TradingCalendar
from flow_model.data.market_view import MarketView
from flow_model.data.series import BarSeries, TickSeries, from_ns, to_ns
from flow_model.data.store import SymbolData
from flow_model.features.base import FeatureError
from flow_model.features.levels import (
    DIRECTION_NONE,
    DIRECTION_RESISTANCE,
    DIRECTION_SUPPORT,
    GATE_CLEANLINESS,
    GATE_NO_TARGET,
    GATE_NO_ZONE,
    GATE_PASSED,
    GATE_PROXIMITY,
    GATE_REJECTION,
    GATE_REWARD_RISK,
    GATE_SIGNIFICANCE,
    HTF_PIVOT_REGION_BARS,
    SETUP_RR_BANDS,
    LevelCandidate,
    LevelFeatures,
    Zone,
    age_term,
    band_volume_profile,
    build_zone,
    cleanliness_score,
    close_position_term,
    cluster_levels,
    density_term,
    displacement_term,
    distinct_touches,
    efficiency_term,
    failed_break_count,
    flow_term,
    htf_term,
    integrity_term,
    kaufman_efficiency,
    mean_adjacent_overlap,
    overlap_term,
    rejection_magnitude_term,
    rejection_score,
    setup_class_for,
    significance_score,
    structure_magnitude,
    touch_term,
    volatility_regularity_term,
    volume_in_band,
)
from flow_model.features.structure import StructureFeatures, atr_aligned, true_range
from flow_model.validation.lookahead import assert_no_lookahead, audit_computer

SYMBOL = "NQ"
TZ = ZoneInfo("America/New_York")
T0 = datetime(2024, 1, 2, 14, 35, tzinfo=timezone.utc)

FAST = 300
SLOW = 1800
HTF = 900

#: The hand-built tape's three prices. `BASELINE` is where it idles, `VISIT`
#: is the mid it drops to, and `ZONE` is that bar's low -- the level.
BASELINE = 103.0
VISIT = 101.0
ZONE = 100.0

#: The tape's constant true range, and therefore its constant Wilder ATR.
ATR = 2.0

#: `atr_period=2` plus a 60-bar region: 62 bars, so absolute bar index
#: `i` is region index `i - OFFSET`.
WINDOW = 62
REGION = 60
OFFSET = WINDOW - REGION


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def nq_spec(**overrides) -> InstrumentSpec:
    """An NQ-shaped spec: tick 0.25, RTH 09:30-16:00, ETH wrapping midnight."""
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


def short_atr(**overrides) -> FeatureConfig:
    """`atr_period=2`, so the 62-bar window yields a 60-bar region."""
    fields = dict(atr_period=2)
    fields.update(overrides)
    return FeatureConfig(**fields)


def levels_config(**overrides) -> StructureLevelConfig:
    """A 60-bar region with every session anchor off: zones come from the
    pivots and the round numbers only, which are the two sources a
    calendar-free tape can produce honestly.

    `recent_window_bars` and `approach_bars` are declared explicitly because
    the schema bounds them against `lookback_bars`.
    """
    fields = dict(
        lookback_bars=REGION,
        pivot_confirm_bars=3,
        pivot_prominence_atr=0.4,
        recent_window_bars=30,
        approach_bars=10,
        use_prior_session_levels=False,
        use_overnight_levels=False,
        use_opening_range=False,
        use_vwap_levels=False,
        use_round_numbers=False,
    )
    fields.update(overrides)
    return StructureLevelConfig(**fields)


def anchored_levels(**overrides) -> StructureLevelConfig:
    """`levels_config` plus round numbers every 4 points.

    The tape visits [100, 104], so the only multiples of 4 inside it are 100
    and 104: one anchor on the support being tested and one on the resistance
    that serves as its target. That is the whole reason the increment is 4 --
    it buys exactly two anchors at prices the tests name.
    """
    fields = dict(use_round_numbers=True, round_increment_points={SYMBOL: 4.0})
    fields.update(overrides)
    return levels_config(**fields)


def support_only_levels(**overrides) -> StructureLevelConfig:
    """Round numbers every 100 points: the only multiple inside [100, 104] is
    100 itself, so the support exists and has nothing above it to target."""
    fields = dict(use_round_numbers=True, round_increment_points={SYMBOL: 100.0})
    fields.update(overrides)
    return levels_config(**fields)


def grid_stamps(n: int, interval: int = FAST) -> np.ndarray:
    step = int(interval) * 1_000_000_000
    return np.array([to_ns(T0) + step * i for i in range(n)], dtype=np.int64)


def series_from(stamps, highs, lows, closes, volumes, interval=FAST) -> BarSeries:
    return BarSeries(
        symbol=SYMBOL,
        ts_ns=np.asarray(stamps, dtype=np.int64),
        interval_seconds=int(interval),
        columns={
            "open": np.asarray(closes, dtype=np.float64).copy(),
            "high": np.asarray(highs, dtype=np.float64),
            "low": np.asarray(lows, dtype=np.float64),
            "close": np.asarray(closes, dtype=np.float64),
            "volume": np.asarray(volumes, dtype=np.float64),
        },
    )


def support_mids(visits=(12, 21, 30), single=5, n=WINDOW) -> np.ndarray:
    """Mids for the support tape, in ABSOLUTE bar indices (region = index - 2).

    A `visit` drops the mid to 101 for two bars; `single` drops it for one.
    Both are approached and left through a 102 bar so no step exceeds 1.0 and
    the true range stays exactly 2.0.

    The one-bar `single` dip is the only bar prominent enough to be a
    confirmed swing low (its low is 1.0 below its neighbours', and the
    threshold is `0.4 * ATR = 0.8`), so it is what CREATES the zone. The
    two-bar visits are deliberately NOT pivots -- each has an equally low
    neighbour, so its prominence is 0 -- which is what makes them countable
    touches rather than excluded forming bars.
    """
    mids = np.full(int(n), BASELINE)
    if single is not None:
        mids[single - 1] = 102.0
        mids[single] = VISIT
        mids[single + 1] = 102.0
    for start in visits:
        mids[start - 1] = 102.0
        mids[start] = VISIT
        mids[start + 1] = VISIT
        mids[start + 2] = 102.0
    # The tail: four 102 bars, then the evaluation bar.
    mids[-5:-1] = 102.0
    mids[-1] = VISIT
    return mids


def resistance_mids(**kwargs) -> np.ndarray:
    """The support tape reflected about 102.0, so the level under test is the
    RESISTANCE at 104.

    `mid -> 204 - mid` maps the baseline 103 to 101 and the visit 101 to 103,
    which preserves every step size and therefore the constant 2.0 true range
    exactly. A reflected tape is the only honest way to test 14.6's mirrored
    stop: re-deriving the short geometry by hand would let the same sign error
    into the test and the implementation at once.
    """
    return 204.0 - support_mids(**kwargs)


def tape(
    mids=None,
    *,
    high=102.0,
    low=ZONE,
    close=101.5,
    volume=1000.0,
    volume_mid=VISIT,
    interval=FAST,
) -> BarSeries:
    """The support tape. The LAST bar's high/low/close are given explicitly
    because it is the rejection bar every gate test turns on.

    Volume sits only on the bars that visit the level, and is zero elsewhere.
    That is what makes the volume-at-price profile writable by hand: only one
    bar shape contributes to it.
    """
    mids = support_mids() if mids is None else np.asarray(mids, dtype=np.float64)
    highs = mids + 1.0
    lows = mids - 1.0
    closes = mids.copy()
    highs[-1], lows[-1], closes[-1] = float(high), float(low), float(close)
    volumes = np.where(np.isclose(mids, float(volume_mid)), float(volume), 0.0)
    return series_from(grid_stamps(mids.size, interval), highs, lows, closes, volumes, interval)


def resistance_tape(**overrides) -> BarSeries:
    fields = dict(high=104.0, low=102.0, close=102.5, volume_mid=103.0)
    fields.update(overrides)
    return tape(resistance_mids(), **fields)


def htf_bars(*, pivot_low=ZONE, n=70, interval=HTF, end_index=WINDOW - 1) -> BarSeries:
    """A higher-interval series idling at 110 with ONE prominent swing low.

    `n = 70` clears `atr_period + max(HTF_PIVOT_REGION_BARS, 2k+1) = 62`, so
    the interval counts as AVAILABLE for `s_htf`; the single low is 10 points
    below its neighbours, far past the `0.4 * ATR` prominence floor, so the
    interval also counts as CONFIRMING when `pivot_low` falls in the zone.
    """
    mids = np.full(int(n), 110.0)
    highs = mids + 1.0
    lows = mids - 1.0
    lows[int(n) // 2] = float(pivot_low)
    end = to_ns(T0) + int(end_index) * FAST * 1_000_000_000
    step = int(interval) * 1_000_000_000
    stamps = np.array([end - step * (int(n) - 1 - i) for i in range(int(n))], dtype=np.int64)
    return series_from(stamps, highs, lows, mids.copy(), np.full(int(n), 1000.0), interval)


def ticks_for(bars: BarSeries, delta: float) -> dict[int, TickSeries]:
    """One tick aggregate per bar, all with the same signed delta."""
    size = len(bars)
    buy = np.full(size, float(delta) if delta > 0 else 0.0)
    sell = np.full(size, 0.0 if delta > 0 else -float(delta))
    return {
        bars.interval_seconds: TickSeries(
            symbol=SYMBOL,
            ts_ns=bars.ts_ns.copy(),
            columns={"buy_volume": buy, "sell_volume": sell},
        )
    }


def data_of(bars: BarSeries, *, ticks=None, higher=None) -> SymbolData:
    series = {bars.interval_seconds: bars}
    if higher is not None:
        series[higher.interval_seconds] = higher
    return SymbolData(
        symbol=SYMBOL,
        primary_interval=bars.interval_seconds,
        bars=series,
        ticks=ticks,
    )


def view_at(data: SymbolData, index: int = -1) -> MarketView:
    bars = data.primary_bars
    position = index if index >= 0 else len(bars) + index
    stamp = int(bars.ts_ns[position])
    return MarketView(data, now=from_ns(stamp), now_ns=stamp)


def empty_view(interval: int = FAST) -> MarketView:
    bars = BarSeries.empty(SYMBOL, interval)
    return MarketView(
        SymbolData(symbol=SYMBOL, primary_interval=interval, bars={interval: bars}),
        now=T0,
        now_ns=to_ns(T0),
    )


def computer(levels=None, features=None, calendar=None, higher=()) -> LevelFeatures:
    return LevelFeatures(
        features if features is not None else short_atr(),
        levels if levels is not None else levels_config(),
        nq_spec(),
        calendar=calendar,
        higher_intervals=higher,
    )


def values_of(bars: BarSeries, levels=None, **kwargs) -> dict[str, float]:
    return computer(levels=levels, **kwargs).compute(view_at(data_of(bars))).values


def candidate(price: float, **overrides) -> LevelCandidate:
    fields = dict(
        price=float(price),
        origin_age_bars=0,
        known_age_bars=0,
        is_anchor=False,
        source="test",
    )
    fields.update(overrides)
    return LevelCandidate(**fields)


def note_with(vector, fragment: str) -> str:
    """The one note containing `fragment`, or a failure naming what was there."""
    hits = [note for note in vector.notes if fragment in note]
    assert len(hits) == 1, f"expected one note containing {fragment!r}, got {vector.notes}"
    return hits[0]


def has_note(vector, fragment: str) -> bool:
    return any(fragment in note for note in vector.notes)


@pytest.fixture(scope="module")
def synthetic_data() -> SymbolData:
    """~3 months of 5-minute NQ bars: comfortably over 2x the 514-bar warmup."""
    from flow_model.data import SyntheticConfig, SyntheticMarketGenerator

    config = load_config()
    dataset = SyntheticMarketGenerator(SyntheticConfig(), seed=7).generate(
        SYMBOL,
        config.spec(SYMBOL),
        date(2017, 1, 1),
        date(2017, 4, 1),
        FAST,
        calendar=TradingCalendar(),
    )
    return SymbolData(symbol=SYMBOL, primary_interval=FAST, bars={FAST: dataset.bars})


# ---------------------------------------------------------------------------
# the tape itself: the scale every later number is measured against
# ---------------------------------------------------------------------------


def test_every_true_range_on_the_hand_built_tape_is_exactly_two():
    """The premise of every hand-computed number below. `mid` moves by at most
    1.0 per bar on a bar of range 2.0, so
    `TR = max(2, |mid+1 - prev|, |mid-1 - prev|) == 2` -- and therefore the
    Wilder ATR is 2.0, the zone band is 0.5, the stop buffer is 0.5 and the
    proximity window is 1.5. If this fails, every arithmetic assertion in this
    file is measuring a different tape than its comment claims."""
    bars = tape()
    assert len(bars) == WINDOW
    ranges = true_range(bars.col("high"), bars.col("low"), bars.col("close"))
    assert ranges.size == WINDOW - 1
    assert np.all(ranges == ATR)

    atr = atr_aligned(bars.col("high"), bars.col("low"), bars.col("close"), 2)
    assert atr.size == REGION
    assert np.all(atr == ATR)


def test_the_tape_carries_exactly_one_confirmed_pivot_at_the_level():
    """The two-bar visits must NOT be pivots -- each has an equally low
    neighbour -- or they would be excluded from the touch count as forming
    bars and `tau` would collapse. The one-bar dip must be, or there is no
    zone at all without round numbers."""
    bars = tape()
    atr = atr_aligned(bars.col("high"), bars.col("low"), bars.col("close"), 2)
    from flow_model.features.structure import find_pivots

    pivots = find_pivots(
        bars.col("high")[-REGION:],
        bars.col("low")[-REGION:],
        atr,
        confirm_bars=3,
        prominence_atr=0.4,
    )
    assert [(p.index, p.price, p.is_high) for p in pivots] == [(5 - OFFSET, ZONE, False)]
    # prominence = (101 - 1) vs the best neighbour low (102 - 1), in ATR units
    assert pivots[0].prominence_atr == pytest.approx(1.0 / ATR)


# ---------------------------------------------------------------------------
# 14.2 clustering into zones
# ---------------------------------------------------------------------------


def test_two_prices_inside_the_band_are_one_level():
    """14.2's own example: 18245.00 and 18248.25 are one level at a 4-point
    band. The whole point of clustering is that a tick-level difference is not
    a different level."""
    levels = [candidate(18245.00), candidate(18248.25)]
    clusters = cluster_levels(levels, 4.0)
    assert len(clusters) == 1
    assert [c.price for c in clusters[0]] == [18245.00, 18248.25]


def test_single_linkage_chains_across_a_span_wider_than_the_band():
    """Single linkage is what 14.2 names, and chaining is its documented cost:
    three prices each 3.25 apart merge at a 4-point band even though the ends
    are 6.5 apart. Pinned because it is the mechanism by which a dense ladder
    of pivots becomes one wide zone, and `zone_width` is the only warning."""
    prices = [18245.00, 18248.25, 18251.50]
    clusters = cluster_levels([candidate(p) for p in prices], 4.0)
    assert len(clusters) == 1
    assert [c.price for c in clusters[0]] == prices


def test_a_gap_of_exactly_the_band_starts_a_new_cluster():
    """The comparison is strict (`gap < band` merges), so the band is an open
    bound. Tested because an off-by-one-tick reading here changes how many
    zones exist on every bar."""
    merged = cluster_levels([candidate(100.0), candidate(100.49)], 0.5)
    split = cluster_levels([candidate(100.0), candidate(100.50)], 0.5)
    assert len(merged) == 1
    assert len(split) == 2


def test_a_zero_band_puts_every_level_in_its_own_cluster():
    """Including two levels at the SAME price: `0 < 0` is false, so identical
    prices do not merge either. Unreachable from `compute` (the band is
    `zone_band_atr * ATR` and both are positive) but the function is public."""
    assert len(cluster_levels([candidate(100.0), candidate(100.0)], 0.0)) == 2


def test_clustering_refuses_a_negative_or_non_finite_band():
    with pytest.raises(FeatureError, match="non-negative finite"):
        cluster_levels([candidate(100.0)], -0.5)
    with pytest.raises(FeatureError, match="non-negative finite"):
        cluster_levels([candidate(100.0)], float("nan"))


def test_clustering_nothing_yields_nothing_and_orders_by_price():
    assert cluster_levels([], 1.0) == ()
    clusters = cluster_levels([candidate(105.0), candidate(100.0), candidate(102.0)], 0.5)
    assert [c[0].price for c in clusters] == [100.0, 102.0, 105.0]


def test_the_zone_price_is_volume_weighted_not_bar_counted():
    """14.2: `zone_price = volume-weighted mean of members`. The weights
    (1, 99) on prices (100, 101) give 100.99; a count-weighted mean would give
    100.5, which no choice of weights can produce here."""
    zone = build_zone([candidate(100.0), candidate(101.0)], [1.0, 99.0], 0.5)
    assert zone.price == pytest.approx((100.0 * 1.0 + 101.0 * 99.0) / 100.0)
    assert zone.price == pytest.approx(100.99)
    assert zone.volume_weighted


def test_with_no_volume_the_zone_price_is_the_median_and_says_so():
    """14.2 specifies the MEDIAN when there is no volume profile.
    `volume_weighted_average` falls back to the mean, so its flag has to be
    consulted and its value thrown away: on (100, 101, 110) the median is 101
    and the mean is 103.667."""
    members = [candidate(100.0), candidate(101.0), candidate(110.0)]
    zone = build_zone(members, [0.0, 0.0, 0.0], 0.5)
    assert zone.price == 101.0
    assert zone.price != pytest.approx((100.0 + 101.0 + 110.0) / 3.0)
    assert not zone.volume_weighted


def test_the_zone_width_is_exactly_the_specified_width_and_contains_every_member():
    """14.2: `zone_width = max(member spread, min_width_ticks * tick_size)`.
    The volume-weighted price (100.99) sits near one edge of a 1.0-wide
    cluster, so a band CENTRED on it would start at 100.49 and exclude the
    member at 100.0. The band is therefore shifted, not widened: [100, 101],
    width exactly 1.0. A widened band would have been [100.0, 101.49] -- 49%
    wider than 14.2's width -- and `zone.low`/`zone.high` are read by the
    touch test, by 14.5's rejection requirement and by the `s_volume`
    scaling, so an inflated band loosens all three."""
    zone = build_zone([candidate(100.0), candidate(101.0)], [1.0, 99.0], 0.5)
    assert zone.price == pytest.approx(100.99)
    assert (zone.low, zone.high) == (100.0, 101.0)
    assert zone.width == pytest.approx(max(1.0, 0.5))
    assert zone.low <= 100.0 and 101.0 <= zone.high

    mirrored = build_zone([candidate(100.0), candidate(101.0)], [99.0, 1.0], 0.5)
    assert mirrored.price == pytest.approx(100.01)
    assert (mirrored.low, mirrored.high) == (100.0, 101.0)
    assert mirrored.width == pytest.approx(1.0)


def test_a_single_member_zone_is_the_minimum_width_centred_on_its_price():
    """Spread 0, so the width floor binds: `min_width_ticks * tick_size`
    = 2 * 0.25 = 0.5, giving the [99.75, 100.25] band every later test names."""
    zone = build_zone([candidate(ZONE)], [1000.0], 2 * 0.25)
    assert (zone.low, zone.price, zone.high) == (99.75, 100.0, 100.25)
    assert zone.width == 0.5
    assert zone.member_count == 1


def test_build_zone_refuses_input_it_cannot_interpret():
    with pytest.raises(FeatureError, match="at least one member"):
        build_zone([], [], 0.5)
    with pytest.raises(FeatureError, match="one weight per member"):
        build_zone([candidate(100.0)], [1.0, 2.0], 0.5)
    with pytest.raises(FeatureError, match="non-negative and finite"):
        build_zone([candidate(100.0)], [1.0], -1.0)


def test_the_zone_keeps_the_provenance_its_scores_need():
    """`has_anchor` feeds `s_anchor`, `newest_origin_age_bars` feeds `s_age`,
    and `origin_indices` is what excludes a level's own forming bar from its
    touch count. `known_age_bars` is when the zone BEGAN -- the oldest member's
    confirmation, not the newest's."""
    old = candidate(100.0, origin_age_bars=40, known_age_bars=37, origin_index=7)
    new = candidate(100.2, origin_age_bars=5, known_age_bars=2, is_anchor=True, origin_index=42)
    zone = build_zone([old, new], [1.0, 1.0], 0.5)
    assert zone.has_anchor
    assert zone.newest_origin_age_bars == 5
    assert zone.oldest_origin_age_bars == 40
    assert zone.known_age_bars == 37
    assert zone.origin_indices == (7, 42)
    assert zone.sources == ("test", "test")
    assert zone.member_count == 2


def test_a_member_whose_forming_bar_predates_the_region_is_not_excluded_from_touches():
    """`origin_index = -1` means "formed before the measured region", and it
    must not leak into `origin_indices` as a real index -- -1 would exclude the
    LAST bar of the region under Python indexing semantics if anything ever
    indexed with it."""
    zone = build_zone([candidate(100.0, origin_index=-1)], [1.0], 0.5)
    assert zone.origin_indices == ()


def test_the_gap_to_a_price_is_zero_inside_the_band_and_the_band_distance_outside():
    """`gap_to` is what the proximity gate and the nearest-zone choice both
    read, so it has to be the distance to the BAND and not to `price`."""
    zone = build_zone([candidate(ZONE)], [1.0], 0.5)
    assert zone.gap_to(100.0) == 0.0
    assert zone.gap_to(100.25) == 0.0
    assert zone.gap_to(101.5) == pytest.approx(1.25)
    assert zone.gap_to(99.0) == pytest.approx(0.75)
    assert zone.contains(99.75) and not zone.contains(99.74)


def test_a_zone_cannot_be_built_outside_its_own_band():
    """The invariant that makes a touch well defined: if `price` could sit
    outside `[low, high]`, whether a bar touched the level would depend on
    which member happened to carry the volume."""
    with pytest.raises(ValueError, match="outside its own band"):
        Zone(
            price=105.0,
            low=99.0,
            high=101.0,
            member_count=1,
            has_anchor=False,
            newest_origin_age_bars=0,
            oldest_origin_age_bars=0,
            known_age_bars=0,
            volume_weighted=False,
            origin_indices=(),
            sources=("x",),
        )


def test_a_level_confirmed_before_it_formed_is_refused():
    """14.1's existence proof. `known_age_bars > origin_age_bars` would mean
    the level became knowable before the bar that printed it closed, which is
    the exact shape of a lookahead leak in a swing-pivot implementation."""
    with pytest.raises(ValueError, match="confirmed before it formed"):
        candidate(100.0, origin_age_bars=3, known_age_bars=9)
    with pytest.raises(ValueError, match="not finite"):
        candidate(float("inf"))


# ---------------------------------------------------------------------------
# 14.3 `s_volume`: volume at price, from bars
# ---------------------------------------------------------------------------


def test_a_bars_volume_is_spread_uniformly_across_its_own_range():
    """The documented approximation, and the one every `s_volume` rests on. A
    bar spanning [100, 102] with 1000 lots overlaps [99.75, 100.25] by 0.25 of
    its 2.0 range, so 0.125 * 1000 = 125 lots are attributed to the band."""
    got = volume_in_band(
        np.array([102.0]), np.array([100.0]), np.array([1000.0]), 99.75, 100.25
    )
    assert got == pytest.approx(125.0)


def test_a_bar_that_misses_the_band_contributes_nothing():
    got = volume_in_band(
        np.array([104.0]), np.array([102.0]), np.array([1000.0]), 99.75, 100.25
    )
    assert got == 0.0


def test_a_zero_range_bar_is_attributed_entirely_to_the_band_holding_its_price():
    """Spreading a volume over a zero-width range is a division by zero, and
    dropping the bar would lose a halted bar's entire volume. A flat bar
    outside the band still contributes nothing."""
    inside = volume_in_band(
        np.array([100.0]), np.array([100.0]), np.array([700.0]), 99.75, 100.25
    )
    outside = volume_in_band(
        np.array([103.0]), np.array([103.0]), np.array([700.0]), 99.75, 100.25
    )
    assert inside == pytest.approx(700.0)
    assert outside == 0.0


def test_volume_in_band_refuses_misaligned_arrays_and_an_inverted_band():
    with pytest.raises(FeatureError, match="equal-length"):
        volume_in_band(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]), 0.0, 1.0)
    assert volume_in_band(np.array([1.0]), np.array([0.0]), np.array([5.0]), 1.0, 0.0) == 0.0
    assert volume_in_band(np.array([]), np.array([]), np.array([]), 0.0, 1.0) == 0.0


def test_the_profile_is_the_same_attribution_computed_for_every_bucket():
    """The distribution `s_volume` percentile-ranks against. Four bars
    spanning [100, 102] with 1000 lots each put 0.25 of every bar in each of
    the four buckets below 102 and nothing above it: 1000 lots per bucket, and
    zero in the top four."""
    highs = np.full(4, 102.0)
    lows = np.full(4, 100.0)
    volumes = np.full(4, 1000.0)
    edges = 100.0 + 0.5 * np.arange(9, dtype=np.float64)
    profile = band_volume_profile(highs, lows, volumes, edges)
    assert profile.size == 8
    assert profile[:4] == pytest.approx([1000.0] * 4)
    assert profile[4:] == pytest.approx([0.0] * 4)
    # Nothing is lost and nothing is invented: the edges span every bar.
    assert profile.sum() == pytest.approx(volumes.sum())


def test_a_flat_bar_lands_in_exactly_one_profile_bucket_and_is_not_double_counted():
    highs = np.array([102.0, 100.0])
    lows = np.array([100.0, 100.0])
    volumes = np.array([1000.0, 400.0])
    edges = 100.0 + 0.5 * np.arange(9, dtype=np.float64)
    profile = band_volume_profile(highs, lows, volumes, edges)
    assert profile.sum() == pytest.approx(1400.0)
    assert profile[0] == pytest.approx(250.0 + 400.0)


def test_the_profile_refuses_misaligned_bars_or_fewer_than_two_edges():
    with pytest.raises(FeatureError, match="equal-length bar arrays"):
        band_volume_profile(
            np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]), np.array([0.0, 1.0])
        )
    with pytest.raises(FeatureError, match="at least two edges"):
        band_volume_profile(np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# 14.3 `tau`: distinct touches
# ---------------------------------------------------------------------------


def touch_arrays(shapes):
    """`shapes` is a list of (high, low); ATR is 2.0 at every bar."""
    highs = np.array([h for h, _ in shapes], dtype=np.float64)
    lows = np.array([l for _, l in shapes], dtype=np.float64)
    return highs, lows, np.full(highs.size, ATR)


def test_a_bar_whose_range_intersects_the_band_is_a_touch():
    highs, lows, atr = touch_arrays([(100.5, 99.5), (104.0, 102.0), (99.80, 98.0)])
    assert distinct_touches(
        highs, lows, atr, low=99.75, high=100.25, separation_atr=0.75, separation_bars=5
    ) == (0, 2)


def test_one_long_consolidation_is_not_twenty_touches():
    """14.3's own words, and the reason the thinning exists at all: twenty
    consecutive bars grinding against the level with `n_sep = 5` count four
    touches, not twenty. The `or` in 14.3's rule makes it four rather than
    one, which 14.3 asks for and this module declines to tighten."""
    highs, lows, atr = touch_arrays([(100.5, 99.5)] * 20)
    counted = distinct_touches(
        highs, lows, atr, low=99.75, high=100.25, separation_atr=0.75, separation_bars=5
    )
    assert counted == (0, 5, 10, 15)
    assert len(counted) == 4 != 20


def test_two_touches_closer_than_n_sep_with_no_departure_are_one_touch():
    """Bars 1 and 2 stay inside the band, so nothing departed and bar 3 is the
    same test continuing."""
    highs, lows, atr = touch_arrays([(100.5, 99.5)] * 4)
    assert distinct_touches(
        highs, lows, atr, low=99.75, high=100.25, separation_atr=0.75, separation_bars=5
    ) == (0,)


def test_a_departure_of_exactly_d_sep_atr_makes_them_two_touches():
    """`d_sep * ATR = 0.75 * 2.0 = 1.5`. The intervening bars sit at a low of
    101.75, which is 101.75 - 100.25 = 1.5 above the band's top edge -- exactly
    the threshold, and the comparison is `>=`."""
    highs, lows, atr = touch_arrays(
        [(100.5, 99.5), (102.75, 101.75), (102.75, 101.75), (100.5, 99.5)]
    )
    assert distinct_touches(
        highs, lows, atr, low=99.75, high=100.25, separation_atr=0.75, separation_bars=5
    ) == (0, 3)


def test_a_departure_one_hundredth_short_of_the_threshold_does_not():
    """1.49 ATR-points of departure against a 1.5 threshold. Paired with the
    test above so the boundary is pinned from both sides rather than asserted
    once in the middle of a wide interval."""
    highs, lows, atr = touch_arrays(
        [(100.5, 99.5), (102.74, 101.74), (102.74, 101.74), (100.5, 99.5)]
    )
    assert distinct_touches(
        highs, lows, atr, low=99.75, high=100.25, separation_atr=0.75, separation_bars=5
    ) == (0,)


def test_the_departure_is_measured_against_the_atr_at_the_candidate_bar():
    """A departure belongs to the volatility of its own era: the same 1.5-point
    excursion is distinct at ATR 2.0 (threshold 1.5) and not distinct at ATR
    4.0 (threshold 3.0). Measuring it against the evaluation bar's ATR would
    silently rescale every historical touch."""
    highs, lows, _ = touch_arrays(
        [(100.5, 99.5), (102.75, 101.75), (102.75, 101.75), (100.5, 99.5)]
    )
    wide = np.array([ATR, ATR, ATR, 4.0])
    assert distinct_touches(
        highs, lows, wide, low=99.75, high=100.25, separation_atr=0.75, separation_bars=5
    ) == (0,)


def test_a_bar_with_no_atr_falls_back_to_the_bar_count_rule_only():
    """No positive ATR means no scale on which to measure a departure. The
    bar-count half of 14.3's `or` still applies, so a touch 5 bars later still
    counts -- the bar is not dropped, it just cannot be separated by price."""
    highs, lows, _ = touch_arrays(
        [(100.5, 99.5), (102.75, 101.75), (102.75, 101.75), (100.5, 99.5)]
    )
    dead = np.array([ATR, ATR, ATR, 0.0])
    assert distinct_touches(
        highs, lows, dead, low=99.75, high=100.25, separation_atr=0.75, separation_bars=5
    ) == (0,)

    spaced_h, spaced_l, _ = touch_arrays([(100.5, 99.5)] + [(104.0, 102.0)] * 4 + [(100.5, 99.5)])
    assert distinct_touches(
        spaced_h,
        spaced_l,
        np.zeros(6),
        low=99.75,
        high=100.25,
        separation_atr=0.75,
        separation_bars=5,
    ) == (0, 5)


def test_a_levels_own_forming_bar_is_not_one_of_its_touches():
    """A swing low touches its own price by construction. Counting it would
    give every pivot zone `tau >= 1` and make 14.3's age term -- defined only
    for `tau == 0` -- unreachable for every level in the system, which is how
    a conditional term quietly becomes a constant."""
    highs, lows, atr = touch_arrays([(100.5, 99.5), (104.0, 102.0), (100.5, 99.5)])
    kwargs = dict(low=99.75, high=100.25, separation_atr=0.75, separation_bars=5)
    assert distinct_touches(highs, lows, atr, **kwargs) == (0, 2)
    assert distinct_touches(highs, lows, atr, exclude=frozenset({0}), **kwargs) == (2,)
    assert distinct_touches(highs, lows, atr, exclude=frozenset({0, 2}), **kwargs) == ()


def test_distinct_touches_refuses_degenerate_separation_settings():
    highs, lows, atr = touch_arrays([(100.5, 99.5)])
    kwargs = dict(low=99.75, high=100.25)
    with pytest.raises(FeatureError, match="at least 1"):
        distinct_touches(highs, lows, atr, separation_atr=0.75, separation_bars=0, **kwargs)
    with pytest.raises(FeatureError, match="positive and finite"):
        distinct_touches(highs, lows, atr, separation_atr=0.0, separation_bars=5, **kwargs)
    with pytest.raises(FeatureError, match="equal-length"):
        distinct_touches(
            highs, np.array([1.0, 2.0]), atr, separation_atr=0.75, separation_bars=5, **kwargs
        )


# ---------------------------------------------------------------------------
# 14.4 approach quality: efficiency, overlap, failed breaks
# ---------------------------------------------------------------------------


def test_a_straight_line_approach_is_perfectly_efficient():
    assert kaufman_efficiency(np.array([100.0, 101.0, 102.0, 103.0])) == pytest.approx(1.0)


def test_a_round_trip_has_no_efficiency_at_all():
    assert kaufman_efficiency(np.array([100.0, 101.0, 100.0])) == 0.0


def test_a_partial_retrace_is_the_ratio_of_net_to_gross():
    """net 1.0 over gross 3.0: a grind into the level, which is exactly what
    14.4 means by an approach that is not clean."""
    assert kaufman_efficiency(np.array([100.0, 102.0, 101.0])) == pytest.approx(1.0 / 3.0)


def test_a_halted_tape_is_not_a_clean_approach():
    """`0/0` has to resolve somewhere, and 1.0 would call a frozen tape the
    cleanest possible approach to a level."""
    assert kaufman_efficiency(np.array([100.0, 100.0, 100.0])) == 0.0
    assert kaufman_efficiency(np.array([100.0])) == 0.0
    assert kaufman_efficiency(np.array([])) == 0.0


def test_identical_adjacent_bars_are_maximal_churn():
    highs, lows = np.full(4, 102.0), np.full(4, 100.0)
    assert mean_adjacent_overlap(highs, lows) == pytest.approx(1.0)


def test_disjoint_adjacent_bars_do_not_overlap_at_all():
    highs = np.array([101.0, 103.0])
    lows = np.array([100.0, 102.0])
    assert mean_adjacent_overlap(highs, lows) == 0.0


def test_the_overlap_denominator_is_the_union_not_the_smaller_range():
    """An inside bar must not score a flat 1.0. [100,104] then [101,103] share
    2.0 of a 4.0 union, so 0.5 -- and a one-point step on two-point bars
    (the hand-built tape's own geometry) shares 1.0 of a 3.0 union, so 1/3."""
    inside = mean_adjacent_overlap(np.array([104.0, 103.0]), np.array([100.0, 101.0]))
    stepped = mean_adjacent_overlap(np.array([102.0, 103.0]), np.array([100.0, 101.0]))
    assert inside == pytest.approx(0.5)
    assert stepped == pytest.approx(1.0 / 3.0)


def test_two_halted_bars_at_the_same_price_are_maximal_churn():
    """A zero union: the alternative readings are 0 (perfectly clean) and
    nan. A frozen tape is churn."""
    assert mean_adjacent_overlap(np.array([100.0, 100.0]), np.array([100.0, 100.0])) == 1.0


def test_overlap_needs_two_bars_and_refuses_misaligned_input():
    assert mean_adjacent_overlap(np.array([100.0]), np.array([99.0])) == 0.0
    with pytest.raises(FeatureError, match="equal-length"):
        mean_adjacent_overlap(np.array([1.0, 2.0]), np.array([1.0]))


def test_a_failed_break_is_counted_once_per_episode_not_once_per_bar():
    """14.4 defines phi as failed BREAKS. A four-bar excursion that came back
    is one failed break; counting per bar would make `s_integrity` a function
    of how long price stayed out rather than of how often it returned."""
    closes = np.array([100.0, 105.0, 105.0, 105.0, 100.0])
    assert failed_break_count(closes, 99.75, 100.25, 4) == 1


def test_an_excursion_that_does_not_come_back_within_the_horizon_is_not_a_failed_break():
    """Five bars beyond the band against a 2-bar horizon: price broke the
    level and held, which is the opposite of a failed break."""
    closes = np.array([100.0, 105.0, 105.0, 105.0, 105.0, 105.0, 100.0])
    assert failed_break_count(closes, 99.75, 100.25, 2) == 0


def test_a_flip_straight_through_the_band_is_not_a_failed_break():
    """Documented explicitly in the module: price did not return to the level,
    it went through it. Without this the same move would be counted as two
    failed breaks and halve `s_integrity`."""
    through = np.array([100.0, 105.0, 95.0, 90.0])
    assert failed_break_count(through, 99.75, 100.25, 4) == 0

    reverted = np.array([100.0, 105.0, 95.0, 100.0])
    assert failed_break_count(reverted, 99.75, 100.25, 4) == 2


def test_a_tape_that_never_leaves_the_band_has_no_failed_breaks():
    assert failed_break_count(np.full(10, 100.0), 99.75, 100.25, 4) == 0
    assert failed_break_count(np.array([]), 99.75, 100.25, 4) == 0


def test_failed_break_count_refuses_a_horizon_below_one():
    with pytest.raises(FeatureError, match="at least 1"):
        failed_break_count(np.array([100.0]), 99.0, 101.0, 0)


# ---------------------------------------------------------------------------
# the bounded terms of S (14.3)
# ---------------------------------------------------------------------------


def test_s_touch_is_the_capped_touch_count():
    """`min(tau, tau_cap) / tau_cap` with `tau_cap = 4`. The cap is what keeps
    one much-tested level from dominating a weighted sum."""
    assert touch_term(0, 4) == 0.0
    assert touch_term(1, 4) == 0.25
    assert touch_term(3, 4) == 0.75
    assert touch_term(4, 4) == 1.0
    assert touch_term(40, 4) == 1.0
    with pytest.raises(FeatureError, match="at least 1"):
        touch_term(1, 0)


def test_s_reject_is_the_median_displacement_over_the_reference():
    """`clip(median(r_j) / r_ref)`. The MEDIAN, not the mean: on
    (0.5, 1.0, 9.0) the median is 1.0 and the mean is 3.5, so a single
    spectacular bounce cannot carry a level."""
    assert rejection_magnitude_term([0.5, 1.0, 9.0], 1.5) == pytest.approx(1.0 / 1.5)
    assert rejection_magnitude_term([3.0], 1.5) == 1.0
    assert rejection_magnitude_term([0.0, 0.0], 1.5) == 0.0


def test_a_level_with_no_completed_rejection_measurement_scores_zero():
    """0 is the honest reading of "no evidence" for a term that rewards
    evidence; 0.5 would credit an untested level with half a bounce."""
    assert rejection_magnitude_term([], 1.5) == 0.0
    assert rejection_magnitude_term([float("nan")], 1.5) == 0.0
    with pytest.raises(FeatureError, match="positive and finite"):
        rejection_magnitude_term([1.0], 0.0)


def test_s_htf_is_confirming_intervals_over_available_ones():
    assert htf_term(0, 2) == 0.0
    assert htf_term(1, 2) == 0.5
    assert htf_term(2, 2) == 1.0
    with pytest.raises(FeatureError, match=r"must lie in \[0, 2\]"):
        htf_term(3, 2)


def test_no_available_higher_interval_scores_zero_rather_than_neutral():
    """`0/0`. 0.5 would be a vote of half-confluence from a dataset that
    cannot see a second timeframe at all, and because the weight is not
    redistributed the consequence is a cap on S, stated in the module
    docstring rather than hidden."""
    assert htf_term(0, 0) == 0.0


def test_s_age_decays_only_while_the_level_is_untested():
    """The conditional IS the term. `lambda = 500` bars, so an untested level
    500 bars old scores `exp(-1) = 0.3679` and the same level with one touch
    scores 1.0 -- a tested level is confirmed, not stale."""
    assert age_term(0, 0, 500.0) == pytest.approx(1.0)
    assert age_term(250, 0, 500.0) == pytest.approx(math.exp(-0.5))
    assert age_term(500, 0, 500.0) == pytest.approx(math.exp(-1.0))
    assert age_term(5000, 0, 500.0) == pytest.approx(math.exp(-10.0))

    for touches in (1, 2, 40):
        assert age_term(5000, touches, 500.0) == 1.0


def test_s_age_treats_a_negative_age_as_zero_and_refuses_a_dead_lambda():
    assert age_term(-7, 0, 500.0) == pytest.approx(1.0)
    with pytest.raises(FeatureError, match="positive and finite"):
        age_term(10, 0, 0.0)


# ---------------------------------------------------------------------------
# the bounded terms of C (14.4)
# ---------------------------------------------------------------------------


def test_s_density_falls_linearly_with_recent_touches_and_floors_at_zero():
    """`clip(1 - rho / rho_max)` with `rho_max = 3`. This is the term that
    opposes `s_touch`, and 14.7's third falsification condition is about
    exactly that opposition, so the sign has to be right: more recent touches
    means LESS clean."""
    assert density_term(0, 3) == 1.0
    assert density_term(1, 3) == pytest.approx(2.0 / 3.0)
    assert density_term(2, 3) == pytest.approx(1.0 / 3.0)
    assert density_term(3, 3) == 0.0
    assert density_term(9, 3) == 0.0
    with pytest.raises(FeatureError, match="at least 1"):
        density_term(1, 0)


def test_s_eff_and_s_overlap_are_the_clipped_ratios_14_4_states():
    assert efficiency_term(0.45, 0.45) == 1.0
    assert efficiency_term(0.225, 0.45) == pytest.approx(0.5)
    assert efficiency_term(0.90, 0.45) == 1.0
    assert efficiency_term(0.0, 0.45) == 0.0

    assert overlap_term(0.0, 0.70) == 1.0
    assert overlap_term(0.35, 0.70) == pytest.approx(0.5)
    assert overlap_term(0.70, 0.70) == 0.0
    assert overlap_term(0.95, 0.70) == 0.0

    with pytest.raises(FeatureError, match="positive and finite"):
        efficiency_term(0.5, 0.0)
    with pytest.raises(FeatureError, match="positive and finite"):
        overlap_term(0.5, 0.0)


def test_s_integrity_decays_with_failed_breaks_and_never_reaches_zero():
    """`1/(1 + phi)`: a judgement call labelled as one in the module. The
    "never zero" half is load-bearing -- 14.6 says a term that could zero the
    sum is doing a gate's job, and a level poked five times still exists."""
    assert integrity_term(0) == 1.0
    assert integrity_term(1) == 0.5
    assert integrity_term(2) == pytest.approx(1.0 / 3.0)
    assert integrity_term(3) == 0.25
    assert integrity_term(100) > 0.0
    with pytest.raises(FeatureError, match="cannot be negative"):
        integrity_term(-1)


def test_s_vol_penalizes_a_flush_and_a_dead_tape_symmetrically():
    """`1 - clip(|log(ATR/median)| / log 2)`. A doubling and a halving both
    score 0; the symmetry is the point, because a panic flush and a frozen
    tape are both not "ordinary volatility"."""
    assert volatility_regularity_term(2.0, 2.0) == 1.0
    assert volatility_regularity_term(4.0, 2.0) == 0.0
    assert volatility_regularity_term(1.0, 2.0) == 0.0
    assert volatility_regularity_term(8.0, 2.0) == 0.0
    # sqrt(2) in either direction is half a doubling in log space.
    assert volatility_regularity_term(2.0 * math.sqrt(2.0), 2.0) == pytest.approx(0.5)
    assert volatility_regularity_term(2.0 / math.sqrt(2.0), 2.0) == pytest.approx(0.5)


def test_s_vol_has_no_value_on_a_tape_with_no_range():
    """The logarithm has none either, and 0.0 is the conservative reading."""
    assert volatility_regularity_term(0.0, 2.0) == 0.0
    assert volatility_regularity_term(2.0, 0.0) == 0.0
    assert volatility_regularity_term(float("nan"), 2.0) == 0.0


# ---------------------------------------------------------------------------
# the bounded terms of R (14.5)
# ---------------------------------------------------------------------------


def test_s_close_is_the_bars_close_position_mirrored_for_a_short():
    """14.5 writes `p = (close - low) / (high - low)`, which is the support
    case. For a short at resistance the quantity must be mirrored, exactly as
    14.6 mirrors the stop -- otherwise a short rejection bar is REWARDED for
    closing on its high, which is the opposite of a rejection."""
    assert close_position_term(102.0, 100.0, 102.0, DIRECTION_SUPPORT) == 1.0
    assert close_position_term(102.0, 100.0, 101.5, DIRECTION_SUPPORT) == pytest.approx(0.5)
    assert close_position_term(102.0, 100.0, 101.0, DIRECTION_SUPPORT) == 0.0
    assert close_position_term(102.0, 100.0, 100.0, DIRECTION_SUPPORT) == 0.0

    assert close_position_term(102.0, 100.0, 100.0, DIRECTION_RESISTANCE) == 1.0
    assert close_position_term(102.0, 100.0, 100.5, DIRECTION_RESISTANCE) == pytest.approx(0.5)
    assert close_position_term(102.0, 100.0, 102.0, DIRECTION_RESISTANCE) == 0.0


def test_s_close_denies_the_term_to_a_zero_range_bar_or_a_directionless_one():
    """A bar has no position within itself, and 0.5 would credit it with half
    a rejection it did not make."""
    assert close_position_term(100.0, 100.0, 100.0, DIRECTION_SUPPORT) == 0.0
    assert close_position_term(102.0, 100.0, 102.0, DIRECTION_NONE) == 0.0


def test_s_disp_is_the_distance_from_the_level_in_disp_ref_atr_units():
    """`disp_ref * ATR = 0.5 * 2.0 = 1.0`, so a close 1.5 points from the zone
    clips to 1.0 and one 0.5 points away scores 0.5."""
    assert displacement_term(101.5, ZONE, ATR, 0.5) == 1.0
    assert displacement_term(100.5, ZONE, ATR, 0.5) == pytest.approx(0.5)
    assert displacement_term(ZONE, ZONE, ATR, 0.5) == 0.0
    assert displacement_term(99.5, ZONE, ATR, 0.5) == pytest.approx(0.5)


def test_s_disp_has_no_scale_without_an_atr():
    assert displacement_term(101.5, ZONE, 0.0, 0.5) == 0.0
    with pytest.raises(FeatureError, match="positive and finite"):
        displacement_term(101.5, ZONE, ATR, 0.0)


def test_s_flow_is_binary_agreement_and_balanced_flow_agrees_with_nothing():
    """14.5 states it as a sign agreement. A delta of exactly zero is measured
    balance, not confirmation, so it scores 0 -- and the CALLER, not this
    function, is responsible for passing None when there is no feed at all."""
    assert flow_term(500.0, DIRECTION_SUPPORT) == 1.0
    assert flow_term(-500.0, DIRECTION_RESISTANCE) == 1.0
    assert flow_term(-500.0, DIRECTION_SUPPORT) == 0.0
    assert flow_term(500.0, DIRECTION_RESISTANCE) == 0.0
    assert flow_term(0.0, DIRECTION_SUPPORT) == 0.0
    assert flow_term(500.0, DIRECTION_NONE) == 0.0
    assert flow_term(float("nan"), DIRECTION_SUPPORT) == 0.0


# ---------------------------------------------------------------------------
# the three scores and the magnitude
# ---------------------------------------------------------------------------


def test_significance_is_the_sum_of_six_weighted_terms():
    """Written out term by term with the default weights, so a reweighting in
    `schema.py` that this function did not follow fails here."""
    weights = StructureLevelConfig().significance_weights
    got = significance_score(
        weights,
        s_touch=0.75,
        s_reject=1.0,
        s_volume=0.5,
        s_htf=1.0,
        s_age=1.0,
        s_anchor=1.0,
    )
    expected = (
        0.30 * 0.75 + 0.20 * 1.0 + 0.20 * 0.5 + 0.15 * 1.0 + 0.05 * 1.0 + 0.10 * 1.0
    )
    assert expected == pytest.approx(0.825)
    assert got == pytest.approx(expected)


def test_significance_is_bounded_by_its_weights_summing_to_one():
    """No clipping is doing load-bearing work: every term is in [0, 1] and the
    weights sum to 1, so S spans exactly [0, 1]."""
    weights = StructureLevelConfig().significance_weights
    terms = ("s_touch", "s_reject", "s_volume", "s_htf", "s_age", "s_anchor")
    assert significance_score(weights, **{t: 1.0 for t in terms}) == pytest.approx(1.0)
    assert significance_score(weights, **{t: 0.0 for t in terms}) == 0.0


def test_cleanliness_is_the_sum_of_five_weighted_terms():
    weights = StructureLevelConfig().cleanliness_weights
    got = cleanliness_score(
        weights, s_eff=1.0, s_density=1.0, s_overlap=0.0, s_integrity=1.0, s_vol=1.0
    )
    expected = 0.30 * 1.0 + 0.25 * 1.0 + 0.20 * 0.0 + 0.15 * 1.0 + 0.10 * 1.0
    assert expected == pytest.approx(0.80)
    assert got == pytest.approx(expected)

    terms = ("s_eff", "s_density", "s_overlap", "s_integrity", "s_vol")
    assert cleanliness_score(weights, **{t: 1.0 for t in terms}) == pytest.approx(1.0)
    assert cleanliness_score(weights, **{t: 0.0 for t in terms}) == 0.0


def test_dropping_s_flow_subtracts_its_weight_and_never_renormalizes():
    """The arithmetic heart of 14.5's degradation rule. With `s_close = 0.5`
    and `s_disp = 1.0`, R is 0.80 when flow agrees and 0.55 when there is no
    feed. A renormalizing implementation would have returned
    `0.55 / 0.75 = 0.7333`, which is indistinguishable from flow that
    half-agreed -- confidence manufactured from data that does not exist."""
    weights = StructureLevelConfig().rejection_weights
    with_flow = rejection_score(weights, s_close=0.5, s_disp=1.0, s_flow=1.0)
    without = rejection_score(weights, s_close=0.5, s_disp=1.0, s_flow=None)

    assert with_flow == pytest.approx(0.40 * 0.5 + 0.35 * 1.0 + 0.25 * 1.0)
    assert with_flow == pytest.approx(0.80)
    assert without == pytest.approx(0.40 * 0.5 + 0.35 * 1.0)
    assert without == pytest.approx(0.55)
    assert with_flow - without == pytest.approx(weights.order_flow)
    assert without != pytest.approx(0.55 / 0.75)


def test_without_a_tick_feed_the_rejection_score_cannot_exceed_three_quarters():
    """The cap is a reported fact, so it has to be a real one: perfect close
    position and perfect displacement with no flow reach exactly
    `close_position + displacement`."""
    weights = StructureLevelConfig().rejection_weights
    best = rejection_score(weights, s_close=1.0, s_disp=1.0, s_flow=None)
    assert best == pytest.approx(weights.close_position + weights.displacement)
    assert best == pytest.approx(0.75)
    assert rejection_score(weights, s_close=1.0, s_disp=1.0, s_flow=1.0) == pytest.approx(1.0)


def test_the_magnitude_is_a_weighted_sum_and_not_a_product():
    """14.6 states the reason: a product lets one weak term zero an otherwise
    strong setup, which is a gate's job and not a score's. S=0.8, C=0.8, R=0
    scores 0.60 as a sum and 0.00 as a product."""
    weights = StructureLevelConfig().magnitude_weights
    got = structure_magnitude(weights, 0.8, 0.8, 0.0)
    assert got == pytest.approx(0.40 * 0.8 + 0.35 * 0.8 + 0.25 * 0.0)
    assert got == pytest.approx(0.60)
    assert got != 0.0
    assert structure_magnitude(weights, 1.0, 1.0, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 14.6 setup selection as a measurement
# ---------------------------------------------------------------------------


def test_the_setup_bands_are_section_14_6s_numbers_verbatim():
    """The constants, before any geometry: the encoded class IS the setup's R
    multiple, so the encoding cannot drift from the thing it names."""
    assert SETUP_RR_BANDS == ((2.7, 3.0), (1.8, 2.0), (1.0, 1.0))


def test_each_band_edge_reads_off_the_measured_reward_to_risk():
    """Both sides of all three edges. The bands are half-open from below, so
    1.8 is a SETUP_2R and 1.7999 is a SCALP_1R."""
    assert setup_class_for(1.0, 1.0) == 1.0
    assert setup_class_for(1.7999, 1.0) == 1.0
    assert setup_class_for(1.8, 1.0) == 2.0
    assert setup_class_for(2.6999, 1.0) == 2.0
    assert setup_class_for(2.7, 1.0) == 3.0
    assert setup_class_for(12.0, 1.0) == 3.0


def test_an_achievable_rr_below_the_floor_declines_rather_than_demoting():
    """14.6's `rr < min_reward_risk -> WAIT("rr_too_low")`. 0 is "declined",
    not "a smaller setup"."""
    assert setup_class_for(0.9, 1.0) == 0.0
    assert setup_class_for(0.0, 1.0) == 0.0
    assert setup_class_for(float("nan"), 1.0) == 0.0


def test_an_rr_above_the_floor_but_below_the_lowest_band_is_still_declined():
    """The reason `StructureLevelConfig.min_reward_risk` defaults to 1.0: a
    floor of 0.9 would pass this gate at rr=0.95 and then match no band. The
    function declines rather than inventing a class, so the misleading
    combination is impossible to reach silently."""
    assert setup_class_for(0.95, 0.9) == 0.0
    assert setup_class_for(1.0, 0.9) == 1.0


# ---------------------------------------------------------------------------
# 14.3 end to end: all six terms of S on one hand-built zone
# ---------------------------------------------------------------------------
#
# The zone, written out once because the next several tests all read it:
#
#   members     the confirmed swing low at 100.0 (the one-bar dip at bar 5)
#               and the round-number anchor at 100.0
#   band        spread 0, so the width floor binds: 2 ticks * 0.25 = 0.5,
#               giving [99.75, 100.25] centred on 100.0
#   touches     the three two-bar visits at bars 12, 21 and 30 -- region
#               indices 10, 19 and 28, ten apart and so comfortably past the
#               5-bar separation rule. The one-bar dip is the zone's own
#               forming bar and the last bar is the evaluation bar, and both
#               are excluded, so tau is 3 and not 5.
#   profile     volume sits only on the bars whose mid is 101, whose range is
#               [100, 102]. The region spans [100, 104], the bucket width is
#               the zone band 0.5, so there are exactly 8 buckets. Each
#               [100,102] bar puts 0.25 of its volume in each of the four
#               buckets below 102 and nothing above, so the profile is
#               (V, V, V, V, 0, 0, 0, 0) for V = 0.25 * (total visit volume).
#               The zone band [99.75, 100.25] catches only 0.25/2.0 = 0.125 of
#               each such bar, which is 0.5 * V, and the band is exactly one
#               bucket wide so no rescaling applies. Four of the eight buckets
#               are at or below it, so s_volume = 4/8 = 0.5.


def test_significance_is_the_weighted_sum_of_all_six_terms():
    """The test this file exists for on the S side. Every one of 14.3's six
    terms is forced to a value the geometry determines, so `S = 0.825` is six
    independent numbers rather than one aggregate that happens to look right:

        s_touch  = min(3, 4) / 4                                   = 0.75
        s_reject = clip(1.875 / 1.5)                               = 1.0
                   (3.75 points of departure against a 2.0 ATR)
        s_volume = 4 of 8 profile buckets at or below the zone's     = 0.50
        s_htf    = 1 confirming higher interval of 1 available      = 1.0
        s_age    = 1.0 because tau > 0, NOT exp(-56/500) = 0.894    = 1.0
        s_anchor = the round number at 100.0 is in the band         = 1.0

        S = .30(.75) + .20(1) + .20(.5) + .15(1) + .05(1) + .10(1) = 0.825
    """
    data = data_of(tape(), higher=htf_bars())
    vector = computer(levels=anchored_levels(), higher=(HTF,)).compute(view_at(data))
    values = vector.values

    assert values["zone_count"] == 2.0
    assert values["nearest_zone_price"] == ZONE
    assert values["nearest_zone_width"] == 0.5
    assert values["level_touch_count"] == 3.0
    assert values["level_recent_touches"] == 0.0

    expected = (
        0.30 * 0.75 + 0.20 * 1.0 + 0.20 * 0.5 + 0.15 * 1.0 + 0.05 * 1.0 + 0.10 * 1.0
    )
    assert expected == pytest.approx(0.825)
    assert values["level_significance"] == pytest.approx(expected)
    assert values["level_significance"] >= anchored_levels().min_significance


def test_s_anchor_contributes_exactly_its_weight_and_nothing_else_moves():
    """Toggling `use_round_numbers` removes the anchor from the zone and
    leaves the pivot. Every other term is a function of the bars, which did
    not change, so S must fall by exactly `anchor_bonus = 0.10`. A larger drop
    would mean the anchor was feeding some other term too."""
    data = data_of(tape(), higher=htf_bars())
    with_anchor = computer(levels=anchored_levels(), higher=(HTF,)).compute(view_at(data))
    without = computer(levels=levels_config(), higher=(HTF,)).compute(view_at(data))

    assert with_anchor.values["level_significance"] == pytest.approx(0.825)
    assert without.values["level_significance"] == pytest.approx(0.825 - 0.10)
    # The zone is still there, still at the same price, still the same width.
    assert without.values["nearest_zone_price"] == ZONE
    assert without.values["nearest_zone_width"] == 0.5
    assert without.values["level_touch_count"] == 3.0


def test_s_htf_contributes_exactly_its_weight_when_a_higher_interval_confirms():
    """Three states of the same term, on identical primary bars: one
    higher-interval pivot inside the zone (1/1), one outside it (0/1), and no
    higher interval at all (0/0, the term dropped with its weight kept)."""
    confirming = data_of(tape(), higher=htf_bars(pivot_low=ZONE))
    elsewhere = data_of(tape(), higher=htf_bars(pivot_low=90.0))
    comp = computer(levels=anchored_levels(), higher=(HTF,))

    assert comp.compute(view_at(confirming)).values["level_significance"] == pytest.approx(0.825)
    assert comp.compute(view_at(elsewhere)).values["level_significance"] == pytest.approx(0.675)
    assert values_of(tape(), anchored_levels())["level_significance"] == pytest.approx(0.675)
    assert 0.825 - 0.675 == pytest.approx(anchored_levels().significance_weights.htf_confluence)


def test_an_unreadable_higher_interval_lowers_the_denominator_rather_than_voting_no():
    """"We cannot see the hourly chart yet" and "the hourly chart has no swing
    here" are different statements, and only the second is a vote. With a
    confirming 15-minute interval plus an hourly one that is too short to hold
    a confirmed pivot, `s_htf` is 1/1 and S is 0.825; with the hourly long
    enough but unconfirming it is 1/2 and S drops by half the weight."""
    short = data_of(tape(), higher=htf_bars())
    short.bars[3600] = htf_bars(n=40, interval=3600)
    seen = data_of(tape(), higher=htf_bars())
    seen.bars[3600] = htf_bars(n=70, interval=3600, pivot_low=90.0)
    comp = computer(levels=anchored_levels(), higher=(HTF, 3600))

    truncated = comp.compute(view_at(short))
    assert truncated.values["level_significance"] == pytest.approx(0.825)
    assert "not counted as an available higher interval" in note_with(truncated, "3600s interval")

    assert comp.compute(view_at(seen)).values["level_significance"] == pytest.approx(
        0.675 + 0.15 * 0.5
    )


def test_s_age_decays_with_age_only_while_the_level_is_untested():
    """End-to-end form of 14.3's conditional, isolated to one term. Removing
    the two-bar visits leaves `tau = 0` (the only bar that touches the zone is
    its own forming bar), and then moving the forming bar from bar 5 to bar 42
    changes nothing else -- same zone price, same band, same volume profile,
    same approach -- so the whole difference in S is the age term:

        bar 5  -> age 56 bars -> 0.20(0.5) + 0.05 exp(-56/500)  = 0.14470
        bar 42 -> age 19 bars -> 0.20(0.5) + 0.05 exp(-19/500)  = 0.14814
    """
    comp = computer(levels=levels_config())
    old = comp.compute(view_at(data_of(tape(support_mids(visits=(), single=5))))).values
    recent = comp.compute(view_at(data_of(tape(support_mids(visits=(), single=42))))).values

    assert old["level_touch_count"] == 0.0 and recent["level_touch_count"] == 0.0
    assert old["nearest_zone_price"] == recent["nearest_zone_price"] == ZONE

    assert old["level_significance"] == pytest.approx(
        0.20 * 0.5 + 0.05 * math.exp(-56.0 / 500.0)
    )
    assert recent["level_significance"] == pytest.approx(
        0.20 * 0.5 + 0.05 * math.exp(-19.0 / 500.0)
    )
    assert recent["level_significance"] > old["level_significance"]
    assert recent["level_significance"] - old["level_significance"] == pytest.approx(
        0.05 * (math.exp(-19.0 / 500.0) - math.exp(-56.0 / 500.0))
    )


def test_a_touched_level_does_not_decay_however_old_its_oldest_member_is():
    """The other half of the conditional, end to end: the three-visit tape's
    zone was formed 56 bars ago, so a decaying `s_age` would contribute
    `0.05 * 0.894 = 0.0447` instead of 0.05 and S would be 0.6697 rather than
    0.675. 0.0053 is small, which is exactly why a dropped conditional here
    would never be noticed from the output."""
    values = values_of(tape(), anchored_levels())
    assert values["level_touch_count"] == 3.0
    assert values["level_significance"] == pytest.approx(0.675)
    decayed = 0.675 - 0.05 * (1.0 - math.exp(-56.0 / 500.0))
    assert decayed == pytest.approx(0.66970, abs=1e-5)
    assert values["level_significance"] != pytest.approx(decayed)


def test_s_volume_is_the_percentile_rank_of_the_hand_built_profile():
    """`s_volume` is pinned here and only here, because this is the one tape
    whose volume-at-price profile is written out bucket by bucket (see the
    block comment above): (V,V,V,V,0,0,0,0) with the zone band holding 0.5V,
    so four of eight buckets are at or below it and the rank is 0.5.

    Deliberate coverage gap: `s_volume` is NOT pinned on a tape whose profile
    has to be derived rather than constructed, because the percentile would
    then be whatever the implementation computed. What IS checked beyond this
    tape is the direction -- concentrating the region's volume into the zone
    raises the rank to 1.0."""
    plain = values_of(tape(), anchored_levels())
    assert plain["level_significance"] == pytest.approx(0.675)
    # Solve the weighted sum for the only unknown: 0.675 = 0.575 + 0.20 * s_volume.
    assert (plain["level_significance"] - 0.575) / 0.20 == pytest.approx(0.5)

    # Flat bars sitting exactly on the level put ALL of their volume in the
    # zone band, and also all of it in the one bucket [100.0, 100.5) that
    # contains 100.0. Five sloped [100, 102] bars still carry 1000 lots each
    # (the one-bar dip, the second bar of each visit, and the rejection bar),
    # and those give bucket 0 a 0.25 share against the band's 0.125, so
    #
    #     bucket 0  = 3,000,000 + 0.25  * 5 * 1000 = 3,001,250
    #     the band  = 3,000,000 + 0.125 * 5 * 1000 = 3,000,625
    #
    # Bucket 0 is the only bucket above the band, bucket 1 holds 1250 and
    # buckets 4-7 hold nothing, so the rank is 7/8 and not 8/8.
    mids = support_mids()
    bars = tape(mids)
    highs, lows, closes = bars.col("high").copy(), bars.col("low").copy(), bars.col("close").copy()
    volumes = bars.col("volume").copy()
    for index in (12, 21, 30):
        highs[index] = lows[index] = closes[index] = ZONE
        volumes[index] = 1_000_000.0
    concentrated = series_from(bars.ts_ns, highs, lows, closes, volumes)
    loaded = values_of(concentrated, anchored_levels())
    assert loaded["level_significance"] > plain["level_significance"]
    assert (loaded["level_significance"] - 0.575) / 0.20 == pytest.approx(7.0 / 8.0)


def test_s_volume_cannot_tell_an_untraded_level_from_a_busy_one_here_reported():
    """**A measurement defect in 14.3's `s_volume`, pinned rather than fixed.**

    `percentile_rank` is the fraction of buckets AT OR BELOW the zone's
    volume, so a zone holding literally nothing ties every empty bucket and
    inherits their count. On this tape four of the eight buckets are empty, so
    an untraded level scores `s_volume = 4/8 = 0.5` -- the same 0.5 a level
    holding half the busiest bucket's volume scores.

    Moving EVERY lot in the region from the level (the 101-mid bars) to the
    baseline (the 103-mid bars) therefore leaves S at 0.675, unchanged, even
    though the zone's band now holds zero volume. That contradicts 14.3's own
    rationale for the term ("High-volume nodes are real levels") and it is
    worth 0.10 of S, the same as the anchor bonus.

    It is NOT fixed here, deliberately. The implementation does exactly what
    14.3's formula says; it is the formula that misbehaves on a sparse
    profile, and the candidate repairs (rank strictly below, rank only against
    non-empty buckets, floor an empty zone at 0) are each a new judgement call
    about a term section 14.7 pre-registers for a Phase 8 sweep. This test
    exists so the current behaviour cannot drift unnoticed and so a future
    repair shows up as a failure here rather than as a quiet change in every
    S in the system."""
    at_level = values_of(tape(), anchored_levels())
    away = values_of(tape(volume_mid=BASELINE), anchored_levels())

    region_slice = slice(-REGION, None)
    bars = tape(volume_mid=BASELINE)
    assert volume_in_band(
        bars.col("high")[region_slice],
        bars.col("low")[region_slice],
        bars.col("volume")[region_slice],
        99.75,
        100.25,
    ) == 0.0
    assert band_volume_profile(
        bars.col("high")[region_slice],
        bars.col("low")[region_slice],
        bars.col("volume")[region_slice],
        100.0 + 0.5 * np.arange(9, dtype=np.float64),
    ) == pytest.approx([0.0, 0.0, 0.0, 0.0, 10000.0, 10000.0, 10000.0, 10000.0])

    assert away["level_significance"] == pytest.approx(0.675)
    assert away["level_significance"] == pytest.approx(at_level["level_significance"])
    assert (away["level_significance"] - 0.575) / 0.20 == pytest.approx(0.5)


def test_a_region_with_no_volume_at_all_falls_back_to_the_median_and_degrades():
    """The other end of the volume path: with nothing traded anywhere there is
    no distribution to rank against and no distribution to weight the zone
    price by, so `s_volume` is 0 for every zone -- NOT the 0.5 that ties on an
    empty profile -- and the vector says both things happened."""
    vector = computer(levels=anchored_levels()).compute(
        view_at(data_of(tape(volume=0.0)))
    )
    note = note_with(vector, "no volume anywhere")
    assert "member median rather than a volume-weighted mean" in note
    assert vector.quality is DataQuality.DEGRADED
    assert vector.values["level_significance"] == pytest.approx(0.575)
    assert vector.values["nearest_zone_price"] == ZONE


def test_one_long_consolidation_does_not_inflate_the_touch_count_end_to_end():
    """The thinning, reached through `compute` rather than through
    `distinct_touches` directly: twenty consecutive bars pressed against the
    level report `tau = 4`, which caps `s_touch` at 1.0 the same way four
    separate tests would -- but 20 raw touches would also have capped at 1.0,
    so the number that proves the thinning ran is `level_touch_count` itself,
    which is emitted for exactly this reason (14.7's third condition needs
    it)."""
    mids = support_mids(visits=(), single=None)
    mids[19] = 102.0
    mids[20:40] = VISIT
    mids[40] = 102.0
    values = values_of(tape(mids), anchored_levels())

    assert values["level_touch_count"] == 4.0
    assert values["level_recent_touches"] == 1.0
    # 0.55 + 0.20 * s_reject, with s_reject = median(0.875, 0.875, 0.875, 1.875)/1.5:
    # the three touches inside the consolidation can only depart 1.75 points
    # (the bar's own range) before the eight-bar horizon is up.
    assert values["level_significance"] == pytest.approx(0.55 + 0.20 * (0.875 / 1.5))
    assert values["level_significance"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# 14.4 end to end: the five terms of C
# ---------------------------------------------------------------------------


def test_cleanliness_is_the_weighted_sum_of_all_five_terms():
    """The C side of the same tape. Each term is forced by the tail geometry,
    which is 103 x6, 102 x4, then the rejection bar:

        s_eff       net 1.5 over gross 1.5 (monotone) -> E = 1,
                    clip(1/0.45)                                     = 1.0
        s_density   rho = 0 in the last 30 bars                      = 1.0
        s_overlap   mean adjacent overlap 23/27 = 0.8519 over a 0.70
                    reference, so clip(1 - 1.217) floors at          = 0.0
        s_integrity no close returns inside the band in the last 30   = 1.0
        s_vol       ATR 2.0 against a median ATR of 2.0              = 1.0

        C = .30(1) + .25(1) + .20(0) + .15(1) + .10(1)               = 0.80
    """
    values = values_of(tape(), anchored_levels())
    expected = 0.30 * 1.0 + 0.25 * 1.0 + 0.20 * 0.0 + 0.15 * 1.0 + 0.10 * 1.0
    assert expected == pytest.approx(0.80)
    assert values["level_cleanliness"] == pytest.approx(expected)
    assert values["level_cleanliness"] >= anchored_levels().min_cleanliness

    # The 0.0 is the overlap term flooring, not a missing term: the tail idles,
    # and an idling tape is churn by 14.4's measure.
    tail_overlap = mean_adjacent_overlap(
        tape().col("high")[-10:], tape().col("low")[-10:]
    )
    assert tail_overlap == pytest.approx(23.0 / 27.0)
    assert overlap_term(tail_overlap, 0.70) == 0.0


def test_a_grind_into_the_level_lowers_cleanliness_through_the_efficiency_term():
    """14.4's first claim: "a clean approach is directional; a grind into the
    level is not". The tail is replaced by a 103/102 oscillation covering the
    same ground, and BOTH approach terms move -- in opposite directions, which
    is worth pinning rather than hiding:

        s_eff      net 1.5 over gross 7.5 -> E = 0.2, clip(0.2/0.45) = 0.4444
        s_overlap  an oscillating tape overlaps LESS than an idling one:
                   mean 11/27 = 0.4074, so clip(1 - 0.4074/0.70)    = 0.4180

        C = .30(.4444) + .25(1) + .20(.4180) + .15(1) + .10(1)      = 0.7169

    So the grind costs 0.167 of cleanliness through efficiency and wins back
    0.084 through overlap. The net is a fall, but it is NOT the fall the
    efficiency term alone implies, and a reader who assumed otherwise would
    mis-read every C in the output."""
    clean = values_of(tape(), anchored_levels())
    assert clean["level_cleanliness"] == pytest.approx(0.80)

    mids = support_mids()
    mids[-9:-1] = [103.0, 102.0, 103.0, 102.0, 103.0, 102.0, 103.0, 102.0]
    bars = tape(mids)
    grind = values_of(bars, anchored_levels())

    assert kaufman_efficiency(bars.col("close")[-11:]) == pytest.approx(0.2)
    assert mean_adjacent_overlap(bars.col("high")[-10:], bars.col("low")[-10:]) == pytest.approx(
        11.0 / 27.0
    )
    expected = (
        0.30 * efficiency_term(0.2, 0.45)
        + 0.25 * 1.0
        + 0.20 * overlap_term(11.0 / 27.0, 0.70)
        + 0.15 * 1.0
        + 0.10 * 1.0
    )
    assert expected == pytest.approx(0.716931, abs=1e-6)
    assert grind["level_cleanliness"] == pytest.approx(expected)
    assert grind["level_cleanliness"] < clean["level_cleanliness"]
    # The touch count did not move, so the fall is approach quality alone.
    assert grind["level_touch_count"] == clean["level_touch_count"] == 3.0


def test_a_failed_break_lowers_cleanliness_through_density_and_integrity():
    """14.4's `s_integrity`: a close beyond the zone followed by a close back
    inside, within `h_fail` bars, is a failed break. The tape steps down to a
    mid of 99 (close 99, below the band) and back to a mid of 100 (close
    100.0, inside it) -- one episode, in steps of exactly 1.0 so the true
    range never leaves 2.0.

    The poke necessarily lands inside the 30-bar recent window, because that
    is the window `phi` is measured over, so it also adds a recent touch.
    Both terms are written out rather than one being isolated:

        s_eff       the tail is untouched, so still                  = 1.0
        s_density   rho = 1 of rho_max 3                             = 2/3
        s_overlap   the tail is untouched, so still floored at       = 0.0
        s_integrity 1/(1 + 1)                                        = 0.5
        s_vol       ATR 2.0 against a 2.0 median                     = 1.0

        C = .30(1) + .25(2/3) + .20(0) + .15(.5) + .10(1)            = 0.6417
    """
    clean = values_of(tape(), anchored_levels())
    mids = support_mids()
    for index, mid in zip(
        range(43, 52), (103.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0, 102.0, 103.0)
    ):
        mids[index] = mid
    bars = tape(mids)
    poked = values_of(bars, anchored_levels())

    assert np.all(true_range(bars.col("high"), bars.col("low"), bars.col("close")) == ATR)
    assert failed_break_count(bars.col("close")[-30:], 99.75, 100.25, 4) == 1
    assert poked["level_touch_count"] == 4.0
    assert poked["level_recent_touches"] == 1.0

    expected = 0.30 * 1.0 + 0.25 * (2.0 / 3.0) + 0.20 * 0.0 + 0.15 * 0.5 + 0.10 * 1.0
    assert expected == pytest.approx(0.641667, abs=1e-6)
    assert poked["level_cleanliness"] == pytest.approx(expected)
    assert poked["level_cleanliness"] < clean["level_cleanliness"]


# ---------------------------------------------------------------------------
# 14.4 the deliberate tension between S and C
# ---------------------------------------------------------------------------

#: A 500-bar region (plus `atr_period = 2`), which is the smallest window in
#: which "six touches spread over 400 bars" is expressible at all.
WIDE_REGION = 500
WIDE_WINDOW = WIDE_REGION + 2


def wide_levels(**overrides) -> StructureLevelConfig:
    fields = dict(
        lookback_bars=WIDE_REGION,
        use_round_numbers=True,
        round_increment_points={SYMBOL: 4.0},
    )
    fields.update(overrides)
    return levels_config(**fields)


def _wide_base() -> np.ndarray:
    return np.full(WIDE_WINDOW, BASELINE)


def _wide_tail(mids: np.ndarray) -> np.ndarray:
    """Two 102 bars and then the evaluation bar: a clean, monotone approach."""
    mids[-3] = 102.0
    mids[-2] = 102.0
    mids[-1] = VISIT
    return mids


def spread_touch_mids() -> np.ndarray:
    """Six two-bar visits at region indices 20, 90, 160, 230, 300 and 370.

    All six are more than 30 bars from the end, so `rho = 0` and the level is
    established rather than under attack.
    """
    mids = _wide_base()
    for region in (20, 90, 160, 230, 300, 370):
        at = region + 2
        mids[at - 1] = 102.0
        mids[at] = VISIT
        mids[at + 1] = VISIT
        mids[at + 2] = 102.0
    return _wide_tail(mids)


def packed_touch_mids() -> np.ndarray:
    """The same six touches, packed into the last 30 bars.

    Six five-bar cycles of (101, 101, 102, 103, 102) starting at region index
    470, so the counted touches land on region 470, 475, 480, 485, 490 and
    495 -- exactly 5 bars apart, which is `touch_separation_bars`, so all six
    are distinct and all six are inside `recent_window_bars = 30`. A 102 bar
    is placed just before the first cycle so no step exceeds 1.0.
    """
    mids = _wide_base()
    mids[469 + 2] = 102.0
    for cycle in range(6):
        start = 470 + 5 * cycle
        for offset, mid in enumerate((VISIT, VISIT, 102.0, BASELINE, 102.0)):
            mids[start + offset + 2] = mid
    mids[498 + 2] = 102.0
    mids[499 + 2] = VISIT
    return mids


def wide_tape(mids) -> BarSeries:
    return tape(mids)


def test_both_tension_tapes_carry_an_identical_level_and_an_identical_tau():
    """The premise of the next test, separated out so a failure says which
    half broke. If the two tapes differed in `tau`, S, the zone price or the
    band, the comparison below would be measuring something else."""
    spread = values_of(wide_tape(spread_touch_mids()), wide_levels())
    packed = values_of(wide_tape(packed_touch_mids()), wide_levels())

    for values in (spread, packed):
        assert values["nearest_zone_price"] == ZONE
        assert values["nearest_zone_width"] == 0.5
        assert values["zone_count"] == 2.0
        assert values["level_touch_count"] == 6.0
        # S = .30(1) + .20(1) + .20(.5) + .15(0) + .05(1) + .10(1)
        assert values["level_significance"] == pytest.approx(0.75)
        assert values["level_significance"] >= wide_levels().min_significance
        assert values["level_rejection"] == pytest.approx(0.55)


def test_the_same_six_touches_are_clean_spread_out_and_filthy_packed():
    """**The test this file exists for.** Section 14.4 claims the S/C split
    expresses a distinction that one score could not: `tau = 6` spread over
    400 bars with `rho = 0` is a well-established level and is traded, while
    the same `tau = 6` packed into the last 30 bars is a level under active
    attack and is declined. Both tapes here have `tau = 6` and `S = 0.75`; the
    only difference is WHEN the touches happened.

    Spread out:

        s_eff 1.0, s_density 1.0 (rho = 0), s_overlap 0.0, s_integrity 1.0,
        s_vol 1.0  ->  C = .30 + .25 + 0 + .15 + .10 = 0.80

    Packed:

        s_density  rho = 6 against rho_max 3, so clip(1 - 2)      = 0.0
        s_eff      the sawtooth's net 0.5 over gross 6.5 is E = 1/13,
                   so clip((1/13)/0.45)                           = 0.1709
        s_overlap  mean adjacent overlap 5/9, clip(1 - 0.7937)    = 0.2063
        s_integrity no close returns inside the band              = 1.0
        s_vol      ATR 2.0 against a 2.0 median                   = 1.0

        C = .30(.1709) + .25(0) + .20(.2063) + .15(1) + .10(1)    = 0.3426

    If this ever reports both tapes clean, 14.7's third falsification
    condition ("if s_density and s_touch have the same sign of association
    with outcome, the S/C split is unjustified") has already been met at the
    arithmetic level and the two scores should collapse to one.
    """
    spread = values_of(wide_tape(spread_touch_mids()), wide_levels())
    packed = values_of(wide_tape(packed_touch_mids()), wide_levels())
    config = wide_levels()

    # Spread out: major AND clean, and it trades.
    assert spread["level_recent_touches"] == 0.0
    assert spread["level_cleanliness"] == pytest.approx(0.80)
    assert spread["level_cleanliness"] >= config.min_cleanliness
    assert spread["structure_gate_passed"] == 1.0
    assert spread["level_gate_reason"] == GATE_PASSED
    assert spread["level_setup_class"] == 1.0

    # Packed: major and FILTHY, and the system declines it on cleanliness --
    # not on significance, and not on the rejection requirement, both of which
    # it still satisfies.
    assert packed["level_recent_touches"] == 6.0
    expected = (
        0.30 * efficiency_term(1.0 / 13.0, 0.45)
        + 0.25 * 0.0
        + 0.20 * overlap_term(5.0 / 9.0, 0.70)
        + 0.15 * 1.0
        + 0.10 * 1.0
    )
    assert expected == pytest.approx(0.342552, abs=1e-6)
    assert packed["level_cleanliness"] == pytest.approx(expected)
    assert packed["level_cleanliness"] < config.min_cleanliness
    assert packed["structure_gate_passed"] == 0.0
    assert packed["level_gate_reason"] == GATE_CLEANLINESS
    assert packed["rejection_confirmed"] == 1.0
    assert packed["level_significance"] >= config.min_significance


def test_recent_touches_never_exceed_the_total_touch_count():
    """`rho` reuses the global thinning rather than re-thinning from the
    window start, and that is what makes the pair comparable at all: two
    different thinnings of the same bars could make the "recent" count exceed
    the total and `s_density` would then be measuring a different population
    than `s_touch`."""
    for mids in (spread_touch_mids(), packed_touch_mids()):
        values = values_of(wide_tape(mids), wide_levels())
        assert values["level_recent_touches"] <= values["level_touch_count"]


# ---------------------------------------------------------------------------
# 14.5 the rejection requirement is a binary gate
# ---------------------------------------------------------------------------


def test_a_bar_that_enters_the_zone_and_closes_back_outside_is_a_rejection():
    """The positive case, so the three negatives below are not all passing for
    the same uninteresting reason. Low 100.0 is inside the [99.75, 100.25]
    band and the close of 101.5 is above it."""
    values = values_of(tape(), anchored_levels())
    assert values["level_direction"] == DIRECTION_SUPPORT
    assert values["rejection_confirmed"] == 1.0
    assert values["level_gate_reason"] == GATE_PASSED


def test_a_bar_that_enters_the_zone_but_closes_inside_it_is_declined_outright():
    """14.5's requirement is a conjunction, and this is the half that is
    easiest to lose: the bar DID trade into the level, so every scored term
    still has a value, and a "rejection" that merely required contact would
    trade here with `R` around 0.2. A close of 100.2 is inside the band, so
    there is no direction and no rejection."""
    values = values_of(tape(high=102.0, low=ZONE, close=100.2), anchored_levels())
    assert values["level_direction"] == DIRECTION_NONE
    assert values["rejection_confirmed"] == 0.0
    assert values["structure_gate_passed"] == 0.0
    assert values["level_gate_reason"] == GATE_REJECTION
    # S and C both passed, so the rejection gate is what did the work.
    assert values["level_significance"] >= anchored_levels().min_significance
    assert values["level_cleanliness"] >= anchored_levels().min_cleanliness


def test_a_bar_that_closes_outside_the_zone_without_entering_it_is_declined():
    """The other half of the conjunction. The bar's low of 100.5 never reaches
    the band's top edge at 100.25, so nothing was rejected -- price simply did
    not get there. Everything else about the setup is intact, including a
    measured 1.11 R:R, which is exactly why this has to be a gate: the scored
    terms give no reason to decline."""
    values = values_of(tape(high=102.5, low=100.5, close=101.5), anchored_levels())
    assert values["level_direction"] == DIRECTION_SUPPORT
    assert values["rejection_confirmed"] == 0.0
    assert values["achievable_rr"] == pytest.approx(1.0 / 0.9)
    assert values["level_gate_reason"] == GATE_REJECTION
    assert values["structure_gate_passed"] == 0.0


def test_a_short_rejection_at_resistance_is_the_exact_mirror():
    """The support tape reflected about 102.0. Every number mirrors: the zone
    is the 104 round number, the direction is -1, S is 0.675 and C is 0.80 as
    before, and `s_close` is measured from the bar's HIGH."""
    values = values_of(resistance_tape(), anchored_levels())
    assert values["nearest_zone_price"] == 104.0
    assert values["level_direction"] == DIRECTION_RESISTANCE
    assert values["rejection_confirmed"] == 1.0
    assert values["level_touch_count"] == 3.0
    assert values["level_significance"] == pytest.approx(0.675)
    assert values["level_cleanliness"] == pytest.approx(0.80)
    # s_close = (104 - 102.5) / (104 - 102) = 0.75 -> 0.5;  s_disp clips to 1.0
    assert values["level_rejection"] == pytest.approx(0.40 * 0.5 + 0.35 * 1.0)
    assert values["structure_gate_passed"] == 1.0


def test_the_binary_requirement_cannot_be_switched_off_in_configuration():
    """Both layers refuse it, and the duplication is deliberate: the schema
    stops the config from being built, and the computer stops an unvalidated
    one from being used. Without the requirement every bar that touched a
    level is a "rejection", which is the single largest way this module could
    be made to look profitable."""
    with pytest.raises(ValueError, match="require_close_back_outside=False"):
        StructureLevelConfig(require_close_back_outside=False)

    unvalidated = StructureLevelConfig.model_construct(
        **{**StructureLevelConfig().model_dump(), "require_close_back_outside": False}
    )
    assert unvalidated.require_close_back_outside is False
    with pytest.raises(FeatureError, match="non-negotiable"):
        computer(levels=unvalidated)


# ---------------------------------------------------------------------------
# 14.5 `s_flow` is dropped, and its weight is NOT redistributed
# ---------------------------------------------------------------------------


def test_without_a_tick_feed_the_rejection_score_is_lower_not_renormalized():
    """The whole degradation contract in one place, and the assertion is on
    the WEIGHT rather than on a note. The same bar, the same zone, the same
    `s_close = 0.5` and `s_disp = 1.0`:

        agreeing flow   0.40(0.5) + 0.35(1) + 0.25(1) = 0.80
        no tick feed    0.40(0.5) + 0.35(1)           = 0.55
        renormalized    0.55 / 0.75                   = 0.7333  <- NOT this

    A renormalizing implementation would report 0.7333 for a bars-only
    rejection, which is indistinguishable from one that order flow partly
    confirmed. That is confidence manufactured from data that does not exist,
    and `strict_component_availability` exists to forbid it."""
    bars = tape()
    with_flow = computer(levels=anchored_levels()).compute(
        view_at(data_of(bars, ticks=ticks_for(bars, 500.0)))
    )
    without = computer(levels=anchored_levels()).compute(view_at(data_of(bars)))
    weights = anchored_levels().rejection_weights

    assert with_flow.values["level_rejection"] == pytest.approx(0.80)
    assert without.values["level_rejection"] == pytest.approx(0.55)
    assert without.values["level_rejection"] < with_flow.values["level_rejection"]
    assert with_flow.values["level_rejection"] - without.values["level_rejection"] == (
        pytest.approx(weights.order_flow)
    )
    renormalized = 0.55 / (weights.close_position + weights.displacement)
    assert renormalized == pytest.approx(0.73333, abs=1e-5)
    assert without.values["level_rejection"] != pytest.approx(renormalized)


def test_the_missing_tick_feed_is_declared_in_the_flag_the_note_and_the_quality():
    """Dropping a term is arithmetically identical to scoring it 0, so the
    OUTPUT has to say which happened -- otherwise a bars-only rejection and a
    flow-contradicted one are the same number with no way to tell them
    apart. `level_flow_available` is that distinction, and it is why the key
    exists."""
    bars = tape()
    without = computer(levels=anchored_levels()).compute(view_at(data_of(bars)))
    against = computer(levels=anchored_levels()).compute(
        view_at(data_of(bars, ticks=ticks_for(bars, -500.0)))
    )

    assert without.values["level_flow_available"] == 0.0
    assert without.quality is DataQuality.DEGRADED
    note = note_with(without, "s_flow is dropped")
    assert "NOT redistributed" in note and "capped at 0.75" in note

    # Flow that contradicts the trade scores the same 0 -- which is correct,
    # and is exactly why the flag rather than the score carries the fact.
    assert against.values["level_rejection"] == pytest.approx(0.55)
    assert against.values["level_flow_available"] == 1.0
    assert not has_note(against, "s_flow is dropped")


def test_the_magnitude_inherits_the_rejection_cap():
    """14.6's magnitude is `w_S*S + w_C*C + w_R*R`, so a capped R caps the
    component's score too: 0.0625 of the 20 STRUCTURE points are unreachable
    on a bars-only dataset, and that is a fact the output should carry rather
    than a note nobody reads."""
    bars = tape()
    with_flow = computer(levels=anchored_levels()).compute(
        view_at(data_of(bars, ticks=ticks_for(bars, 500.0)))
    )
    without = computer(levels=anchored_levels()).compute(view_at(data_of(bars)))
    config = anchored_levels()
    lost = config.magnitude_weights.rejection * config.rejection_weights.order_flow

    assert lost == pytest.approx(0.0625)
    assert with_flow.values["structure_magnitude"] - without.values[
        "structure_magnitude"
    ] == pytest.approx(lost)
    assert without.values["structure_magnitude"] == pytest.approx(
        0.40 * 0.675 + 0.35 * 0.80 + 0.25 * 0.55
    )
    assert with_flow.values["structure_magnitude"] == pytest.approx(
        0.40 * 0.675 + 0.35 * 0.80 + 0.25 * 0.80
    )


def test_a_fully_fed_vector_is_good_quality():
    """The degradation flags must not be stuck on, or DEGRADED stops carrying
    information. With volume, a visible higher interval and a tick feed -- and
    the session anchors switched off rather than absent -- nothing is missing
    and the vector grades GOOD with no notes at all."""
    bars = tape()
    data = data_of(bars, ticks=ticks_for(bars, 500.0), higher=htf_bars())
    vector = computer(levels=anchored_levels(), higher=(HTF,)).compute(view_at(data))
    assert vector.quality is DataQuality.GOOD
    assert vector.notes == ()
    assert vector.values["level_flow_available"] == 1.0


# ---------------------------------------------------------------------------
# 14.6 the determined stop
# ---------------------------------------------------------------------------


def test_the_long_stop_sits_beyond_the_zone_by_the_buffer():
    """14.6: `stop = min(z_lo, rejection_bar.low) - b_stop * ATR`. Here the
    zone's low (99.75) is below the bar's low (100.0), so the zone sets it:
    `99.75 - 0.25 * 2.0 = 99.25`. This is the point of the whole section --
    the stop is where the thesis is falsified, BEYOND the level, rather than
    at an ATR multiple from the entry."""
    values = values_of(tape(), anchored_levels())
    assert values["level_stop_price"] == pytest.approx(99.25)
    assert values["level_stop_price"] < 99.75  # strictly beyond the band
    # and not an ATR multiple from the entry, which would be 101.5 - 0.5
    assert values["level_stop_price"] != pytest.approx(101.5 - 0.25 * ATR)


def test_a_rejection_bar_that_wicked_past_the_zone_sets_the_stop_itself():
    """The `min` in 14.6's formula, exercised from the other side: a bar whose
    low is 99.0 pierced the band, so the stop goes below the BAR. The pierce
    raises the true range to 3.0 and therefore the Wilder ATR to
    `(2.0 + 3.0) / 2 = 2.5`, so the buffer is 0.625 and the stop is 98.375.

    The widened stop then kills the trade on its own measurement:
    `(104 - 101.5) / (101.5 - 98.375) = 0.8`, which is below
    `min_reward_risk`, and 14.6 declines rather than pulling the target in."""
    bars = tape(low=99.0)
    atr = atr_aligned(bars.col("high"), bars.col("low"), bars.col("close"), 2)
    assert float(atr[-1]) == pytest.approx(2.5)

    values = values_of(bars, anchored_levels())
    assert values["level_stop_price"] == pytest.approx(99.0 - 0.25 * 2.5)
    assert values["level_stop_price"] == pytest.approx(98.375)
    assert values["achievable_rr"] == pytest.approx(0.8)
    assert values["level_setup_class"] == 0.0
    assert values["level_gate_reason"] == GATE_REWARD_RISK
    assert values["level_target_price"] == 104.0


def test_the_short_stop_is_the_mirrored_formula():
    """`stop = max(z_hi, rejection_bar.high) + b_stop * ATR`. The reflected
    tape puts the zone at [103.75, 104.25] and the bar's high at 104.0, so the
    zone's top sets it: `104.25 + 0.5 = 104.75`."""
    values = values_of(resistance_tape(), anchored_levels())
    assert values["level_stop_price"] == pytest.approx(104.75)
    assert values["level_stop_price"] > 104.25
    # Symmetric with the long: the same 2.25 of risk from a 2.5-point entry gap
    assert abs(values["level_stop_price"] - 102.5) == pytest.approx(
        abs(99.25 - 101.5)
    )


def test_a_bar_inside_the_band_gets_no_stop_rather_than_a_fabricated_one():
    """Direction 0 means there is nothing to place a stop beyond. The emitted
    value is the close, which is a non-trade marker consistent with
    `rejection_confirmed = 0`, rather than a sentinel that could be regressed
    on by accident."""
    values = values_of(tape(close=100.2), anchored_levels())
    assert values["level_direction"] == DIRECTION_NONE
    assert values["level_stop_price"] == pytest.approx(100.2)
    assert values["achievable_rr"] == 0.0


# ---------------------------------------------------------------------------
# 14.6 the measured target and the setup bands
# ---------------------------------------------------------------------------


def test_the_setup_class_is_read_off_the_geometry_and_not_configured():
    """14.6's claim in its strongest form: nothing in the configuration
    changes between these four runs. The zone stays at 100, the stop stays at
    99.25 and the target stays at the 104 round number; only the rejection
    bar's CLOSE moves, and the setup walks the bands:

        close 100.50 -> (104 - 100.50) / 1.25 = 2.80 -> DIRECTIONAL_3R
        close 100.75 -> (104 - 100.75) / 1.50 = 2.17 -> SETUP_2R
        close 101.50 -> (104 - 101.50) / 2.25 = 1.11 -> SCALP_1R
        close 101.75 -> (104 - 101.75) / 2.50 = 0.90 -> declined
    """
    expected = {100.50: (2.80, 3.0), 100.75: (13.0 / 6.0, 2.0), 101.50: (10.0 / 9.0, 1.0)}
    for close, (rr, setup) in expected.items():
        values = values_of(tape(close=close), anchored_levels())
        assert values["level_stop_price"] == pytest.approx(99.25), close
        assert values["level_target_price"] == 104.0, close
        assert values["achievable_rr"] == pytest.approx(rr), close
        assert values["level_setup_class"] == setup, close
        assert values["structure_gate_passed"] == 1.0, close


def test_a_reward_to_risk_below_the_floor_declines_and_keeps_the_real_target():
    """14.6's `rr_too_low`, and the failure mode it exists to stop: the honest
    answer is "no trade", not "a 0.9R target called a 1R scalp". The emitted
    target stays at the real 104 level, so a reader can see WHY it was
    declined rather than finding a shrunken target that looks fine."""
    values = values_of(tape(close=101.75), anchored_levels())
    assert values["achievable_rr"] == pytest.approx(0.9)
    assert values["achievable_rr"] < anchored_levels().min_reward_risk
    assert values["level_setup_class"] == 0.0
    assert values["structure_gate_passed"] == 0.0
    assert values["level_gate_reason"] == GATE_REWARD_RISK
    assert values["level_target_price"] == 104.0
    # The stop was not moved in either, so R is still |entry - stop|.
    assert values["level_stop_price"] == pytest.approx(99.25)


def test_an_opposing_zone_that_is_not_major_is_not_a_target():
    """14.6: "the next opposing zone with `S >= s_major`". A lone swing high
    at 105 is a zone but not a major one (no touches, no anchor), so with
    `target_requires_major_zone` on there is nothing to target and the trade
    is declined; with it off the same zone is targeted at 1.56R. One flag,
    two outcomes, same bars."""
    mids = support_mids()
    mids[40] = 104.0  # a lone prominent high, so the zone's price is 105.0
    bars = tape(mids)

    strict = values_of(bars, support_only_levels())
    loose = values_of(bars, support_only_levels(target_requires_major_zone=False))

    assert strict["zone_count"] == loose["zone_count"] == 2.0
    assert strict["major_zone_count"] == 1.0
    assert strict["level_gate_reason"] == GATE_NO_TARGET
    assert strict["achievable_rr"] == 0.0
    assert strict["level_target_price"] == pytest.approx(101.5)

    assert loose["level_target_price"] == 105.0
    assert loose["achievable_rr"] == pytest.approx(3.5 / 2.25)
    assert loose["level_setup_class"] == 1.0
    assert loose["structure_gate_passed"] == 1.0


def test_with_no_zone_beyond_price_the_trade_is_declined_not_retargeted():
    """The default, and the reason `fallback_target_atr` defaults to None:
    inventing a target is how a structure-based setup silently becomes an
    arbitrary-R setup."""
    vector = computer(levels=support_only_levels()).compute(view_at(data_of(tape())))
    assert vector.values["zone_count"] == 1.0
    assert vector.values["level_gate_reason"] == GATE_NO_TARGET
    assert vector.values["achievable_rr"] == 0.0
    note = note_with(vector, "no opposing zone")
    assert "declined rather than retargeted" in note


def test_a_configured_fallback_target_is_reported_as_configured_and_degrades():
    """The escape hatch exists, and using it has to be visible: an ATR-multiple
    target makes the R:R a configured number rather than a measurement off the
    structure, so the vector degrades and the note says exactly that."""
    config = support_only_levels(target_requires_major_zone=False, fallback_target_atr=2.0)
    vector = computer(levels=config).compute(view_at(data_of(tape())))

    assert vector.values["level_target_price"] == pytest.approx(101.5 + 2.0 * ATR)
    assert vector.values["achievable_rr"] == pytest.approx(4.0 / 2.25)
    assert vector.values["level_setup_class"] == 1.0
    assert vector.quality is DataQuality.DEGRADED
    note = note_with(vector, "fallback_target_atr")
    assert "configured number rather than a measurement" in note


# ---------------------------------------------------------------------------
# 14.6 the gates, in order
# ---------------------------------------------------------------------------


def test_a_zone_outside_the_entry_window_is_declined_on_proximity_first():
    """14.6 runs proximity before significance, and the order is what makes
    `level_gate_reason` interpretable. This bar fails both -- the gap is
    `1.75 / 2.0 = 0.875` ATR against a 0.75 window, and S is 0.575 against a
    0.60 floor -- and the reported reason is the FIRST failure."""
    values = values_of(tape(close=102.0), levels_config())
    assert values["nearest_zone_gap_atr"] == pytest.approx(0.875)
    assert values["nearest_zone_gap_atr"] > levels_config().entry_atr_window
    assert values["level_significance"] < levels_config().min_significance
    assert values["level_gate_reason"] == GATE_PROXIMITY


def test_an_insignificant_zone_is_declined_on_significance():
    """Without the round-number anchor the zone loses `s_anchor`, and 0.575 is
    below the 0.60 that defines "major". The bar is a textbook rejection
    otherwise, which is the point: "major" is a gate and not a preference."""
    values = values_of(tape(), levels_config())
    assert values["level_significance"] == pytest.approx(0.575)
    assert values["rejection_confirmed"] == 1.0
    assert values["level_cleanliness"] >= levels_config().min_cleanliness
    assert values["level_gate_reason"] == GATE_SIGNIFICANCE


def test_cleanliness_is_judged_before_the_rejection_requirement():
    """Also an ordering test, from the other end of the list: the packed tape
    with a bar that never entered the band fails cleanliness AND the binary
    requirement, and reports cleanliness."""
    bars = tape(packed_touch_mids(), high=102.5, low=100.5, close=101.5)
    values = values_of(bars, wide_levels())
    assert values["level_significance"] >= wide_levels().min_significance
    assert values["level_cleanliness"] < wide_levels().min_cleanliness
    assert values["rejection_confirmed"] == 0.0
    assert values["level_gate_reason"] == GATE_CLEANLINESS


def test_the_gate_reason_and_the_gate_flag_never_disagree():
    """`structure_gate_passed` is what the signal engine reads and
    `level_gate_reason` is what rejection analysis reads, so they have to be
    the same fact. Six of the eight reasons are constructible on these tapes,
    and each is named here rather than discovered -- the pair must agree AND
    the reason must be the one the geometry implies."""
    cases = [
        (tape(), anchored_levels(), GATE_PASSED),
        (tape(close=102.0), levels_config(), GATE_PROXIMITY),
        (tape(), levels_config(), GATE_SIGNIFICANCE),
        (tape(packed_touch_mids()), wide_levels(), GATE_CLEANLINESS),
        (tape(close=100.2), anchored_levels(), GATE_REJECTION),
        (tape(), support_only_levels(), GATE_NO_TARGET),
        (tape(close=101.75), anchored_levels(), GATE_REWARD_RISK),
    ]
    for bars, config, reason in cases:
        values = values_of(bars, config)
        assert values["level_gate_reason"] == reason, (reason, values["level_gate_reason"])
        assert (values["structure_gate_passed"] == 1.0) == (reason == GATE_PASSED)
        if reason == GATE_PASSED:
            assert values["level_setup_class"] in (1.0, 2.0, 3.0)
    assert len({reason for _, _, reason in cases}) == 7


def test_a_bar_with_no_zone_at_all_reports_the_no_zone_reason():
    """A flat tape has no prominent pivot and -- with round numbers off -- no
    anchor, so there is no level to gate on. The price-valued keys repeat the
    close rather than carrying a sentinel, because every emitted value must be
    finite and a -1 in a price field is what gets regressed on by accident."""
    mids = np.full(WINDOW, BASELINE)
    bars = series_from(
        grid_stamps(WINDOW), mids + 1.0, mids - 1.0, mids.copy(), np.full(WINDOW, 1000.0)
    )
    vector = computer(levels=levels_config()).compute(view_at(data_of(bars)))

    assert vector.warmup_complete
    assert vector.values["zone_count"] == 0.0
    assert vector.values["level_gate_reason"] == GATE_NO_ZONE
    assert vector.values["nearest_zone_price"] == BASELINE
    assert vector.values["level_stop_price"] == BASELINE
    assert vector.values["level_target_price"] == BASELINE
    assert all(math.isfinite(value) for value in vector.values.values())
    assert note_with(vector, "no anchor and no confirmed pivot")


# ---------------------------------------------------------------------------
# 14.1 anchors: the objectively-defined levels, or honest absences
# ---------------------------------------------------------------------------

THREE_DAYS = [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
SLOW_BARS_PER_SESSION = 13


def session_levels(**overrides) -> StructureLevelConfig:
    """The smallest legal pivot region (`lookback_bars` has a `gt=10` bound),
    so a three-session dataset of 13-bar sessions clears the 13-bar warmup and
    the anchors are what the test is about."""
    fields = dict(
        lookback_bars=11,
        recent_window_bars=11,
        approach_bars=10,
        atr_median_window=11,
        use_prior_session_levels=True,
        use_overnight_levels=True,
        use_opening_range=True,
        use_vwap_levels=True,
        use_round_numbers=False,
        opening_range_minutes=30.0,
    )
    fields.update(overrides)
    return levels_config(**fields)


def rth_stamps(days, interval: int) -> np.ndarray:
    """Bar-close stamps on the 09:30-16:00 RTH grid, built from exchange-local
    wall times so the UTC stamps move with DST exactly as a real feed's would."""
    out: list[int] = []
    per = int(6.5 * 3600 // interval)
    for day in days:
        open_local = datetime(day.year, day.month, day.day, 9, 30, tzinfo=TZ)
        for step in range(1, per + 1):
            out.append(to_ns(open_local + timedelta(seconds=interval * step)))
    return np.array(out, dtype=np.int64)


def rth_series(days, mids, interval: int) -> BarSeries:
    mids = np.asarray(mids, dtype=np.float64)
    return series_from(
        rth_stamps(days, interval),
        mids + 1.0,
        mids - 1.0,
        mids.copy(),
        np.full(mids.size, 1000.0),
        interval,
    )


def session_mids(values, interval: int) -> np.ndarray:
    per = int(6.5 * 3600 // interval)
    return np.concatenate([np.full(per, float(value)) for value in values])


def test_the_prior_session_anchors_match_structure_features_on_the_same_bars():
    """The module docstring promises this test by name. `levels.py` does its
    own session grouping rather than reusing `StructureFeatures.compute`,
    because it needs the bar that FORMED each extreme and that computer does
    not expose it -- so the cost is a second bisection over
    `calendar.session_date` and the risk is that the two drift apart and the
    system reports two different "prior session highs".

    Each session trades at a different price entirely (100, 200, 300), so a
    grouping that was off by one session would be wrong by 100 points rather
    than by a rounding error."""
    calendar = TradingCalendar()
    config = session_levels()
    bars = rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    data = data_of(bars)
    view = view_at(data)

    structure = StructureFeatures(short_atr(), config, nq_spec(), calendar=calendar).compute(view)
    levels = LevelFeatures(short_atr(), config, nq_spec(), calendar=calendar)

    requested = levels._session_window_bars(SLOW)
    notes: list[str] = []
    anchors, _ = levels._anchor_levels(
        view.bar_timestamps(requested),
        view.column("high", requested),
        view.column("low", requested),
        view.column("close", requested),
        view.column("volume", requested),
        levels._region,
        SLOW,
        False,
        notes,
    )
    by_source = {anchor.source: anchor.price for anchor in anchors}

    assert by_source["prior_session_high"] == structure.values["prior_session_high"] == 201.0
    assert by_source["prior_session_low"] == structure.values["prior_session_low"] == 199.0
    assert by_source["prior_session_close"] == structure.values["prior_session_close"] == 200.0
    assert by_source["opening_range_high"] == structure.values["opening_range_high"] == 301.0
    assert by_source["opening_range_low"] == structure.values["opening_range_low"] == 299.0
    # The VWAP of a session that traded at one price is that price.
    assert by_source["vwap"] == pytest.approx(300.0)


def test_a_prior_session_extremes_forming_bar_is_older_than_its_confirmation():
    """Both ages are needed downstream and they are NOT equal for a period
    extreme: the prior session's high was printed by one bar (here the first
    bar of the 200 session, 25 bars back) but is only final when that session
    ends (13 bars back). The forming bar is what gets excluded from the touch
    count; the confirmation bar is 14.1's existence proof."""
    calendar = TradingCalendar()
    config = session_levels()
    bars = rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    view = view_at(data_of(bars))
    levels = LevelFeatures(short_atr(), config, nq_spec(), calendar=calendar)
    requested = levels._session_window_bars(SLOW)
    anchors, _ = levels._anchor_levels(
        view.bar_timestamps(requested),
        view.column("high", requested),
        view.column("low", requested),
        view.column("close", requested),
        view.column("volume", requested),
        levels._region,
        SLOW,
        False,
        [],
    )
    high = next(a for a in anchors if a.source == "prior_session_high")
    assert high.origin_age_bars == 2 * SLOW_BARS_PER_SESSION - 1 == 25
    assert high.known_age_bars == SLOW_BARS_PER_SESSION == 13
    assert high.known_age_bars < high.origin_age_bars
    assert high.is_anchor


def test_without_a_calendar_the_session_anchors_are_absent_not_substituted():
    """`StructureFeatures` has to emit a number for `prior_session_high`
    either way and substitutes a window extreme. This module emits a SET of
    levels and has no such obligation, so it omits them and says so -- a
    window extreme labelled "prior session high" would raise `s_anchor` for a
    level nobody is watching, which is a fabricated input to S rather than a
    conservative one."""
    vector = computer(
        levels=levels_config(
            use_prior_session_levels=True,
            use_overnight_levels=True,
            use_opening_range=True,
            use_vwap_levels=True,
        )
    ).compute(view_at(data_of(tape())))

    assert vector.quality is DataQuality.DEGRADED
    note = note_with(vector, "no calendar supplied")
    assert "ABSENT, not substituted" in note
    # The pivot zone still exists, and it still has no anchor.
    assert vector.values["zone_count"] == 1.0
    assert vector.values["level_significance"] == pytest.approx(0.575)


def test_round_numbers_without_a_configured_increment_are_a_config_statement_not_a_defect():
    """`use_round_numbers` on with no entry for this symbol means the operator
    asked for something undefined, which is a configuration fact. The note
    says so and the quality is NOT downgraded -- conflating it with missing
    data would make every such config look like a broken feed."""
    config = levels_config(use_round_numbers=True, round_increment_points={"ES": 25.0})
    vector = computer(levels=config).compute(view_at(data_of(tape())))
    note = note_with(vector, "round_increment_points")
    assert "configuration statement, not a data defect" in note
    assert vector.values["zone_count"] == 1.0  # the pivot zone only


def test_too_many_round_numbers_collapse_into_one_wall_and_the_cap_is_reported():
    """The documented cost of 14.2's single linkage, at its worst: a 0.05-point
    increment over a 4-point range generates more multiples than
    `ROUND_NUMBER_MAX`, the nearest 64 are kept, and then every one of them is
    within the 0.5 band of the next so the whole ladder chains into a single
    3.15-wide zone that also swallows the pivot. `nearest_zone_width` is the
    only warning a reader gets, which is why it is emitted."""
    config = levels_config(use_round_numbers=True, round_increment_points={SYMBOL: 0.05})
    vector = computer(levels=config).compute(view_at(data_of(tape())))

    note = note_with(vector, "yields more than")
    assert "kept the 64 nearest to price" in note
    assert vector.values["zone_count"] == 1.0
    assert vector.values["nearest_zone_width"] > 3.0
    assert vector.values["nearest_zone_width"] == pytest.approx(3.15)


def test_zones_beyond_max_zones_tracked_are_dropped_nearest_first_with_a_note():
    """Dropping a zone can remove the TARGET, which changes the trade rather
    than only the bookkeeping -- so the note says so and the gate reports no
    target instead of silently finding a nearer one."""
    vector = computer(levels=anchored_levels(max_zones_tracked=1)).compute(
        view_at(data_of(tape()))
    )
    assert vector.values["zone_count"] == 1.0
    assert vector.values["nearest_zone_price"] == ZONE
    assert vector.values["level_gate_reason"] == GATE_NO_TARGET
    note = note_with(vector, "max_zones_tracked")
    assert "a target beyond them is not visible" in note


# ---------------------------------------------------------------------------
# warmup honesty
# ---------------------------------------------------------------------------


def test_warmup_bars_is_514_and_states_both_of_its_terms():
    """514 is declared, and it is exactly `atr_period + region` where the
    region is `max(lookback_bars, 2 * pivot_confirm_bars + 1)`: one aligned
    ATR for every bar of the measured region, and nothing else."""
    config = load_config()
    comp = LevelFeatures(config.features, config.levels, config.spec(SYMBOL))
    region = max(config.levels.lookback_bars, 2 * config.levels.pivot_confirm_bars + 1)

    assert comp.warmup_bars == 514
    assert comp.warmup_bars == config.features.atr_period + region
    assert (config.features.atr_period, region) == (14, 500)
    assert computer().warmup_bars == WINDOW == 62


def test_the_region_floor_keeps_a_tiny_lookback_from_declaring_an_unusable_warmup():
    """`lookback_bars = 11` with `pivot_confirm_bars = 10` would give a 11-bar
    region in which no centred 21-bar window fits, so the computer would pass
    its own warmup and then find no pivot ever. The `2k + 1` floor is what
    stops that, and it has to be reflected in `warmup_bars` or the declaration
    is wrong."""
    comp = computer(levels=levels_config(lookback_bars=11, recent_window_bars=11, pivot_confirm_bars=10))
    assert comp.warmup_bars == 2 + (2 * 10 + 1)


def test_not_ready_one_bar_before_warmup_and_ready_exactly_at_warmup():
    bars = tape()
    comp = computer(levels=anchored_levels())
    early = comp.compute(view_at(data_of(bars), WINDOW - 2))
    ready = comp.compute(view_at(data_of(bars), WINDOW - 1))

    assert not early.warmup_complete
    assert early.quality is DataQuality.MISSING
    assert f"warmup incomplete: {WINDOW - 1} of {WINDOW} bars" in early.notes[0]
    assert ready.warmup_complete


def test_extra_history_in_front_does_not_move_a_single_value():
    """The operational content of the 514 claim on the calendar-free path:
    with no calendar the computer requests exactly `warmup_bars` bars, so
    history before them cannot be read. 100 bars at a mid of 50 -- half the
    price of everything that follows -- are prepended, and every emitted value
    must be bit-identical. A computation that reached further back would move
    `nearest_zone_price` (the round numbers are generated from the region's
    visited range) long before it moved anything subtle."""
    padded = np.concatenate([np.full(100, 50.0), support_mids()])
    reference = values_of(tape(), anchored_levels())
    extended = values_of(tape(padded), anchored_levels())
    assert extended == reference


def test_a_window_that_cannot_reach_the_prior_session_says_so_and_degrades():
    """The session window is the one thing that reaches further back than
    `warmup_bars`, and the module's defence is that it REPORTS when it could
    not reach. A single session of bars clears the 13-bar warmup and still has
    no prior session, so the anchors are absent rather than invented."""
    calendar = TradingCalendar()
    bars = rth_series(THREE_DAYS[:1], session_mids([100.0], SLOW), SLOW)
    vector = LevelFeatures(
        short_atr(), session_levels(), nq_spec(), calendar=calendar
    ).compute(view_at(data_of(bars)))

    assert vector.warmup_complete
    assert vector.quality is DataQuality.DEGRADED
    note = note_with(vector, "belong to the current ")
    assert "prior-session anchors are absent" in note


def test_the_session_window_reaches_further_back_than_warmup_but_changes_nothing(
    synthetic_data,
):
    """`warmup_bars` is 514 while the session window asks for
    `ceil(4 days / interval) + 1 = 1153` bars of 5-minute data. That is
    declared in the module docstring rather than folded into `warmup_bars`
    (which has no access to the bar interval), and it is defensible exactly as
    long as the anchors stop changing once both session boundaries are
    visible. Here they do: truncating the dataset to 514, 1153 and 2000 bars
    ending at the same evaluation bar gives bit-identical vectors, with and
    without a calendar."""
    bars = synthetic_data.primary_bars
    index = 3000
    stamp = int(bars.ts_ns[index])
    config = load_config()
    spec = config.spec(SYMBOL)

    def truncated(keep: int) -> SymbolData:
        start = max(0, index + 1 - keep)
        columns = {name: bars.col(name)[start : index + 1].copy() for name in bars.columns}
        series = BarSeries(
            symbol=SYMBOL,
            ts_ns=bars.ts_ns[start : index + 1].copy(),
            interval_seconds=FAST,
            columns=columns,
        )
        return SymbolData(symbol=SYMBOL, primary_interval=FAST, bars={FAST: series})

    for calendar in (None, TradingCalendar()):
        comp = LevelFeatures(config.features, config.levels, spec, calendar=calendar)
        assert comp._session_window_bars(FAST) == 1153 > comp.warmup_bars
        reference = comp.compute(
            MarketView(synthetic_data, now=from_ns(stamp), now_ns=stamp)
        ).values
        for keep in (514, 1153, 2000):
            got = comp.compute(
                MarketView(truncated(keep), now=from_ns(stamp), now_ns=stamp)
            ).values
            assert got == reference, (calendar is not None, keep)


def test_the_higher_interval_window_is_declared_by_its_own_degradation():
    """The other window that is not in `warmup_bars`:
    `atr_period + max(HTF_PIVOT_REGION_BARS, 2k+1)` bars of EACH higher
    interval, which at a daily interval is months of history. It is not folded
    into `warmup_bars` either, and the defence is the same: an interval with
    too little history is reported as not available and lowers `s_htf`'s
    denominator instead of producing a number."""
    comp = computer(levels=anchored_levels(), higher=(HTF,))
    assert comp._htf_window == 2 + max(HTF_PIVOT_REGION_BARS, 2 * 3 + 1)
    short = computer(levels=anchored_levels(), higher=(HTF,)).compute(
        view_at(data_of(tape(), higher=htf_bars(n=comp._htf_window - 1)))
    )
    note = note_with(short, f"{HTF}s interval holds")
    assert "not counted as an available higher interval" in note
    assert short.quality is DataQuality.DEGRADED


def test_a_tape_with_no_range_at_the_evaluation_bar_measures_nothing():
    """Every length in section 14 is ATR-scaled -- the band, the touch
    separation, the rejection displacement, the stop buffer and the proximity
    window -- so with no range there is no scale for any of them. Reporting
    zeros as measurements would be worse than reporting nothing."""
    flat = np.full(WINDOW, BASELINE)
    bars = series_from(grid_stamps(WINDOW), flat, flat.copy(), flat.copy(), np.full(WINDOW, 1000.0))
    vector = computer().compute(view_at(data_of(bars)))
    assert not vector.warmup_complete
    assert vector.quality is DataQuality.MISSING
    assert "every length in section 14 is ATR-scaled" in vector.notes[0]


# ---------------------------------------------------------------------------
# the declared contract
# ---------------------------------------------------------------------------


def test_the_emitted_keys_are_exactly_the_declared_twenty_one():
    """`FeatureBundle` checks that a computer produced the keys it promised, so
    a renamed key here is an error at the source rather than a `None` that
    propagates into the STRUCTURE component."""
    comp = computer(levels=anchored_levels())
    assert comp.keys == (
        "zone_count",
        "major_zone_count",
        "nearest_zone_price",
        "nearest_zone_width",
        "nearest_zone_distance_atr",
        "nearest_zone_gap_atr",
        "level_significance",
        "level_cleanliness",
        "level_rejection",
        "level_direction",
        "level_touch_count",
        "level_recent_touches",
        "rejection_confirmed",
        "level_stop_price",
        "level_target_price",
        "achievable_rr",
        "level_setup_class",
        "structure_magnitude",
        "structure_gate_passed",
        "level_gate_reason",
        "level_flow_available",
    )
    assert len(comp.keys) == len(set(comp.keys)) == 21

    vector = comp.compute(view_at(data_of(tape())))
    assert tuple(sorted(vector.values)) == tuple(sorted(comp.keys))
    assert set(vector.quality_by_key) == set(comp.keys)
    assert all(math.isfinite(value) for value in vector.values.values())


def test_required_feeds_is_bars_and_tick_aggregates_are_optional():
    """An honest `required_feeds` matters in both directions: a dishonest
    extra requirement would disable the component on a bars-only dataset it
    can in fact run on, and declaring ticks required would make 14.5's
    documented degradation path unreachable."""
    comp = computer()
    assert comp.name == "levels"
    assert comp.required_feeds == frozenset({Feed.BARS})
    assert comp.optional_feeds == frozenset({Feed.TICK_AGGREGATE})
    # And the honesty is load-bearing: a bars-only view IS served.
    assert comp.feeds_available(view_at(data_of(tape())))
    assert computer(levels=anchored_levels()).compute(
        view_at(data_of(tape()))
    ).warmup_complete


def test_an_empty_view_is_not_ready_and_names_the_absent_feed():
    vector = computer().compute(empty_view())
    assert not vector.warmup_complete
    assert "bars feed absent" in vector.notes[0]


def test_a_not_ready_vector_is_zeros_and_missing_for_every_key():
    """The zeros are placeholders and must never be read, which is what
    `warmup_complete=False` plus MISSING is for -- including for the
    price-valued keys, where 0.0 is not a plausible price."""
    vector = computer().compute(empty_view())
    assert set(vector.values) == set(computer().keys)
    assert set(vector.values.values()) == {0.0}
    assert vector.quality is DataQuality.MISSING
    assert set(vector.quality_by_key.values()) == {DataQuality.MISSING}


def test_the_constructor_takes_configuration_only():
    """Local echo of the package-wide ban in `test_feature_contracts.py`. A
    computer handed a series could capture a full-sample statistic at
    construction, which is the one leak `validation.lookahead` cannot see
    through when it is given an instance rather than a factory.

    `levels.py` uses postponed annotations, so `__init__.__annotations__`
    holds STRINGS: they are resolved against the module globals with
    `get_type_hints` before comparing, because comparing a `str` to a class
    passes silently and has already done so twice in this project."""
    parameters = inspect.signature(LevelFeatures.__init__).parameters
    assert list(parameters) == [
        "self",
        "config",
        "levels",
        "spec",
        "calendar",
        "higher_intervals",
    ]
    assert parameters["calendar"].default is None
    assert parameters["higher_intervals"].default == ()

    hints = get_type_hints(LevelFeatures.__init__)
    assert hints["config"] is FeatureConfig
    assert hints["levels"] is StructureLevelConfig
    assert hints["spec"] is InstrumentSpec
    assert hints["calendar"] == (SessionCalendarProtocol | None)
    assert hints["higher_intervals"] == tuple[int, ...]
    for name in ("config", "levels", "spec"):
        assert not isinstance(hints[name], str), f"{name} hint was not resolved"

    rendered = str(inspect.signature(LevelFeatures.__init__))
    for banned in (
        "SymbolData",
        "BarSeries",
        "ColumnSeries",
        "DataStore",
        "MarketView",
        "SyntheticDataset",
        "ndarray",
        "DataFrame",
        "Series",
    ):
        assert banned not in rendered, f"{banned} must not appear in the constructor"


def test_an_anchor_this_module_has_no_definition_for_is_refused():
    """`FeatureConfig` constrains `vwap_anchor` by pattern, so this branch is
    only reachable past the schema -- which is exactly when a silent default
    would be most damaging, because the VWAP anchor decides which bars the
    anchor levels are computed from."""
    unvalidated = FeatureConfig.model_construct(vwap_anchor="minute", atr_period=2)
    with pytest.raises(FeatureError, match="not one of"):
        LevelFeatures(unvalidated, levels_config(), nq_spec())


def test_the_declared_keys_do_not_depend_on_the_anchor_flags():
    """A key that appeared and disappeared with a config flag would make the
    bundle's key-collision check depend on configuration. The VALUES are
    expected to differ -- that is what the flags are for -- so only the key
    set is pinned."""
    off = levels_config()
    on = anchored_levels()
    assert computer(levels=off).keys == computer(levels=on).keys
    first = computer(levels=off).compute(view_at(data_of(tape())))
    second = computer(levels=on).compute(view_at(data_of(tape())))
    assert set(first.values) == set(second.values)
    assert first.values != second.values


# ---------------------------------------------------------------------------
# lookahead, determinism, and a long run
# ---------------------------------------------------------------------------


def test_the_lookahead_audit_passes_with_a_factory(synthetic_data):
    """`factory=` rather than an instance: that is the form that rebuilds the
    computer against each truncated and each future-mutated dataset, so a
    full-sample constant captured in `__init__` would change with it and be
    caught. Handing the audit an instance would not test that at all.

    This is the test that would catch 14.1's `i <= t - k_confirm` rule being
    lost, which is the failure the whole section is built around: swing levels
    sampled with future information sit exactly where price later turned, and
    every backtested rejection off them looks clean."""
    config = load_config()
    spec = config.spec(SYMBOL)
    comp = LevelFeatures(config.features, config.levels, spec)
    assert len(synthetic_data.primary_bars) >= 2 * comp.warmup_bars

    result = audit_computer(
        data=synthetic_data,
        factory=lambda d: LevelFeatures(config.features, config.levels, spec),
        sample=40,
    )
    assert_no_lookahead([result])
    assert result.bars_checked >= 20
    assert set(result.keys_checked) == set(comp.keys)


def test_the_lookahead_audit_also_passes_with_a_real_calendar(synthetic_data):
    """A second configuration, because the calendar path is the one that reads
    the session window -- the widest window this module touches, and the only
    one that reaches past `warmup_bars`."""
    config = load_config()
    spec = config.spec(SYMBOL)
    calendar = TradingCalendar()
    result = audit_computer(
        data=synthetic_data,
        factory=lambda d: LevelFeatures(
            config.features, config.levels, spec, calendar=calendar
        ),
        sample=40,
    )
    assert_no_lookahead([result])


def test_two_computations_on_the_identical_view_agree_exactly():
    """Bit-for-bit, notes included. A computer that read a clock or an
    unseeded generator would fail here before the audit ever ran."""
    comp = computer(levels=anchored_levels())
    view = view_at(data_of(tape()))
    first, second = comp.compute(view), comp.compute(view)
    assert first.values == second.values
    assert first.notes == second.notes
    assert first.quality_by_key == second.quality_by_key


def test_two_independently_constructed_computers_agree_bit_for_bit():
    """No state survives a call, so a fresh computer must give the same
    answer -- principle 5 stated at the level of one computer. Tested on the
    calendar path too, because that is where the only cached-looking object
    (the timezone) lives."""
    bars = rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    view = view_at(data_of(bars))
    first = LevelFeatures(
        short_atr(), session_levels(), nq_spec(), calendar=TradingCalendar()
    ).compute(view)
    second = LevelFeatures(
        short_atr(), session_levels(), nq_spec(), calendar=TradingCalendar()
    ).compute(view)
    assert first.values == second.values
    assert first.notes == second.notes

    assert values_of(tape(), anchored_levels()) == values_of(tape(), anchored_levels())


def test_every_emitted_key_is_finite_and_bounded_over_a_long_run(synthetic_data):
    """Three months of 5-minute bars through the real accessor path. The
    bounded keys are the ones a component score may read, and one excursion
    past a bound would let a single bar dominate a weighted sum.

    Deliberate coverage gap, stated rather than papered over: five keys are
    unbounded BY DESIGN -- `nearest_zone_price`, `nearest_zone_width`,
    `level_stop_price`, `level_target_price` are prices,
    `nearest_zone_distance_atr` is a signed ATR distance,
    `nearest_zone_gap_atr` is an unsigned one and `achievable_rr` must stay
    unbounded because squashing it would destroy the only quantity in section
    14 that turns setup selection into a measurement. For those, only
    finiteness, sign and the relations between them are checkable here; their
    VALUES are pinned on the hand-built tapes above and nowhere else."""
    config = load_config()
    comp = LevelFeatures(
        config.features, config.levels, config.spec(SYMBOL), calendar=TradingCalendar()
    )
    bars = synthetic_data.primary_bars
    reasons: set[float] = set()
    passed = 0
    scored = 0

    for index in range(comp.warmup_bars - 1, len(bars), 11):
        stamp = int(bars.ts_ns[index])
        vector = comp.compute(MarketView(synthetic_data, now=from_ns(stamp), now_ns=stamp))
        values = vector.values
        assert vector.warmup_complete
        scored += 1

        for key, value in values.items():
            assert math.isfinite(value), f"{key} is {value} at bar {index}"
        for key in (
            "level_significance",
            "level_cleanliness",
            "level_rejection",
            "structure_magnitude",
        ):
            assert 0.0 <= values[key] <= 1.0, (key, values[key], index)
        assert values["level_direction"] in (-1.0, 0.0, 1.0)
        assert values["level_setup_class"] in (0.0, 1.0, 2.0, 3.0)
        assert values["rejection_confirmed"] in (0.0, 1.0)
        assert values["structure_gate_passed"] in (0.0, 1.0)
        assert values["level_flow_available"] in (0.0, 1.0)
        assert 0.0 <= values["level_gate_reason"] <= 7.0
        assert values["level_gate_reason"] == int(values["level_gate_reason"])
        assert values["achievable_rr"] >= 0.0
        assert values["nearest_zone_gap_atr"] >= 0.0
        assert values["level_recent_touches"] <= values["level_touch_count"]
        assert values["major_zone_count"] <= values["zone_count"]
        assert (values["zone_count"] == 0.0) == (values["level_gate_reason"] == GATE_NO_ZONE)
        if values["zone_count"] > 0.0:
            assert values["nearest_zone_width"] > 0.0
        # 14.6's stop sits BEYOND the level, which is a sign relation and
        # therefore checkable without knowing the price.
        if values["level_direction"] > 0.0:
            assert values["level_stop_price"] < values["nearest_zone_price"]
        if values["level_direction"] < 0.0:
            assert values["level_stop_price"] > values["nearest_zone_price"]
        if values["rejection_confirmed"] == 1.0:
            assert values["level_direction"] != 0.0
        if values["structure_gate_passed"] == 1.0:
            assert values["level_significance"] >= config.levels.min_significance
            assert values["level_cleanliness"] >= config.levels.min_cleanliness
            assert values["rejection_confirmed"] == 1.0
            assert values["achievable_rr"] >= config.levels.min_reward_risk
            assert values["level_setup_class"] in (1.0, 2.0, 3.0)
            passed += 1
        # The module docstring's stated consequence of not redistributing
        # `htf_confluence`: on a single-interval dataset S cannot exceed 0.85.
        assert values["level_flow_available"] == 0.0
        assert values["level_significance"] <= 1.0 - (
            config.levels.significance_weights.htf_confluence
        ) + 1e-12
        reasons.add(values["level_gate_reason"])

    assert scored > 300
    assert passed > 0, "no bar ever cleared the gates: the invariants prove nothing"
    assert {GATE_PASSED, GATE_SIGNIFICANCE, GATE_CLEANLINESS, GATE_REJECTION} <= reasons
