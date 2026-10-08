"""Adversarial tests for the market-structure features.

`structure.py` emits the only bars-only component ARCHITECTURE.md section 5
calls "full function on bars alone", which means nothing upstream will ever
degrade it for us: whatever it says is what the STRUCTURE component scores
on. Four things in it can be silently wrong, and the tests are organized
around them.

**Pivot confirmation lag (section 14.1).** This is the one the module exists
to enforce and the one the rest of the file is arranged behind. A bar `i` is
a confirmed swing high at evaluation time `t` only when `i <= t - k`, so a
centred `argmax` evaluated at `t` is a read of bars after `t`. The failure is
invisible from above -- swing levels sampled with future information sit
exactly where price later turned, and every backtested rejection trade looks
clean -- so it is tested twice: directly on `find_pivots` by extending the
input array one bar at a time, and end-to-end through `compute` on a series
carrying two pivots at known bars, where `swing_high` must *not* switch to
the newer one until exactly `k` bars after it forms. The series is built so
that the newer pivot is LOWER than the older one, because otherwise the
window-high fallback would report the newer level before it was confirmed and
the test would pass for the wrong reason.

**Session anchors.** `prior_session_*` and `opening_range_*` are functions of
the trading calendar, and a calendar error is invisible: a session boundary
shifted by an hour still produces plausible numbers. These are tested across
a weekend, across a holiday the exchange was shut for, on bars that fall on a
holiday, on the first bar of a session, inside a still-forming opening range,
and on both DST boundaries -- the DST tests put a distinctive price *only* in
the first 30 minutes of the post-transition session, so a frozen UTC offset
would read the wrong six bars and report a value the test names.

**VWAP anchoring.** The hand-computed VWAP uses volumes `(1, 1, 98)` on
prices `(100, 110, 120)`, so the volume-weighted answer is 119.7 and an
arithmetic mean would be 110.0 -- a price-averaged "VWAP" cannot survive that
arithmetic. The reset is tested by making each session trade at a different
price entirely, so a VWAP that failed to re-anchor would be off by 100.

**Warmup honesty.** `warmup_bars` is 514 and is the pivot window only, while
`compute` *reads* up to `SESSION_WINDOW_DAYS` days of bars for the session
anchors. That is defensible exactly as long as the anchors stop changing once
both session boundaries are visible, and is reported when they are not, so
both halves are tested: extra history in front must not move a single value,
and a window that cannot reach back over the prior session must say so.

Builders are local on purpose: file ownership, and so a failure localizes
here rather than to a shared fixture. Two worlds, because the risks need
different geometry:

* `grid_series` lays bars on a contiguous interval grid with no calendar. The
  session anchors are then substituted (and the vector DEGRADED), which these
  tests ignore -- they are about pivot geometry, breaks and the score, and a
  contiguous grid keeps the bar indices in the test identical to the bar
  indices in the arithmetic.
* `rth_series` lays bars on the real 09:30-16:00 NQ RTH grid in exchange-local
  time for a named list of session dates, with a real `TradingCalendar`. That
  is the only way to test an anchor, because the quantity being tested *is*
  the calendar's answer.

Every baseline bar is `open == close == mid`, `high == mid + 1`,
`low == mid - 1`. That makes the typical price exactly `mid` (so a VWAP is a
weighted mean of numbers the test chose) and the true range exactly 2.0 on a
flat stretch (so ATR, and therefore every prominence threshold, is a number
the test can write down).
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
from flow_model.data.series import BarSeries, from_ns, to_ns
from flow_model.data.store import SymbolData
from flow_model.features.base import FeatureError
from flow_model.features.structure import (
    SESSION_WINDOW_DAYS,
    STRUCTURE_SCORE_WEIGHTS,
    VWAP_DEVIATION_SCALE,
    Pivot,
    StructureFeatures,
    _clip01,
    atr_aligned,
    find_pivots,
    sequence_direction,
    structure_geometry_score,
    true_range,
    typical_price,
    volume_weighted_average,
    wilder_atr_series,
)
from flow_model.validation.lookahead import assert_no_lookahead, audit_computer

SYMBOL = "NQ"
TZ = ZoneInfo("America/New_York")

#: A plausible instant for the calendar-free world. Every firewall comparison
#: is on int64 ns, so the wall-clock value only has to keep a failure message
#: readable.
T0 = datetime(2024, 1, 2, 14, 35, tzinfo=timezone.utc)

#: 5-minute and 30-minute bars over the 09:30-16:00 NQ session: 78 and 13
#: bars per session. The 30-minute grid keeps the session arrays small; the
#: 5-minute grid is the only one on which a 30-minute opening range is more
#: than one bar, so it is the one the partial-range tests use.
FAST = 300
SLOW = 1800
FAST_BARS_PER_SESSION = 78
SLOW_BARS_PER_SESSION = 13

BASE = 100.0


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def nq_spec(**overrides) -> InstrumentSpec:
    """An NQ-shaped spec: RTH 09:30-16:00, ETH 18:00-17:00 wrapping midnight.

    The wrapping ETH window is what makes `session_date` roll at 18:00
    exchange-local, which is the behaviour the overnight tests depend on.
    """
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


def pivot_levels(**overrides) -> StructureLevelConfig:
    """A 60-bar pivot region: wide enough for four pivots at known bars.

    `recent_window_bars` and `touch_cap` have schema bounds relative to
    `lookback_bars`, so the shrunken lookback has to be declared alongside
    them or the config refuses to build.
    """
    fields = dict(
        pivot_confirm_bars=3,
        pivot_prominence_atr=0.4,
        lookback_bars=60,
        opening_range_minutes=30.0,
    )
    fields.update(overrides)
    return StructureLevelConfig(**fields)


def session_levels(**overrides) -> StructureLevelConfig:
    """The smallest legal pivot region (`lookback_bars` has a `gt=10` bound).

    Used by the session-anchor tests, where the pivots are irrelevant and the
    point is to keep `warmup_bars` at 13 so a two- or three-session dataset
    clears warmup.
    """
    fields = dict(
        pivot_confirm_bars=3,
        pivot_prominence_atr=0.4,
        lookback_bars=11,
        recent_window_bars=11,
        approach_bars=10,
        opening_range_minutes=30.0,
    )
    fields.update(overrides)
    return StructureLevelConfig(**fields)


def short_atr(**overrides) -> FeatureConfig:
    """`atr_period=2`, so a hand-written ATR fits in a few bars."""
    fields = dict(atr_period=2)
    fields.update(overrides)
    return FeatureConfig(**fields)


def grid_stamps(n: int, interval: int = FAST) -> np.ndarray:
    """`n` strictly ascending bar-close stamps on a contiguous grid."""
    step = int(interval) * 1_000_000_000
    return np.array([to_ns(T0) + step * i for i in range(n)], dtype=np.int64)


def rth_stamps(days, interval: int) -> np.ndarray:
    """Bar-close stamps on the 09:30-16:00 RTH grid for each session date.

    Built from exchange-local wall times and converted, so the UTC stamps move
    with DST exactly as a real feed's would. That is what makes the DST tests
    discriminating rather than tautological.
    """
    out: list[int] = []
    per = int(6.5 * 3600 // interval)
    for day in days:
        open_local = datetime(day.year, day.month, day.day, 9, 30, tzinfo=TZ)
        for i in range(1, per + 1):
            out.append(to_ns(open_local + timedelta(seconds=interval * i)))
    return np.array(out, dtype=np.int64)


def series_from(stamps, opens, highs, lows, closes, volumes, interval) -> BarSeries:
    opens = np.asarray(opens, dtype=np.float64)
    return BarSeries(
        symbol=SYMBOL,
        ts_ns=np.asarray(stamps, dtype=np.int64),
        interval_seconds=int(interval),
        columns={
            "open": opens,
            "high": np.asarray(highs, dtype=np.float64),
            "low": np.asarray(lows, dtype=np.float64),
            "close": np.asarray(closes, dtype=np.float64),
            "volume": np.asarray(volumes, dtype=np.float64),
        },
    )


def from_mids(stamps, mids, interval, volumes=None) -> BarSeries:
    """Bars whose typical price is exactly `mids[i]` and range exactly 2.0.

    `open == close == mid`, `high == mid + 1`, `low == mid - 1`. Then
    `(h + l + c) / 3 == mid`, and on a stretch of constant `mid` the true
    range is `max(2, 1, 1) == 2`.
    """
    mids = np.asarray(mids, dtype=np.float64)
    vols = np.full(mids.size, 1000.0) if volumes is None else volumes
    return series_from(
        stamps, mids.copy(), mids + 1.0, mids - 1.0, mids.copy(), vols, interval
    )


def grid_series(mids, interval: int = FAST, volumes=None) -> BarSeries:
    return from_mids(grid_stamps(len(mids), interval), mids, interval, volumes)


def rth_series(days, mids, interval: int, volumes=None) -> BarSeries:
    return from_mids(rth_stamps(days, interval), mids, interval, volumes)


def session_mids(values, interval: int) -> np.ndarray:
    """One constant `mid` per session date, expanded to a bar array."""
    per = int(6.5 * 3600 // interval)
    return np.concatenate([np.full(per, float(v)) for v in values])


def symbol_data(series: BarSeries) -> SymbolData:
    return SymbolData(
        symbol=series.symbol,
        primary_interval=series.interval_seconds,
        bars={series.interval_seconds: series},
    )


def view_at(series: BarSeries, index: int = -1) -> MarketView:
    """A view whose cutoff is bar `index`'s close, so that bar is visible."""
    position = index if index >= 0 else len(series) + index
    ts = int(series.ts_ns[position])
    return MarketView(symbol_data(series), now=from_ns(ts), now_ns=ts)


def empty_view(interval: int = FAST) -> MarketView:
    series = BarSeries.empty(SYMBOL, interval)
    return MarketView(symbol_data(series), now=T0, now_ns=to_ns(T0))


def computer(levels=None, features=None, calendar=None) -> StructureFeatures:
    return StructureFeatures(
        features if features is not None else short_atr(),
        levels if levels is not None else session_levels(),
        nq_spec(),
        calendar=calendar,
    )


def note_with(vector, fragment: str) -> str:
    """The one note containing `fragment`, or a failure naming what was there."""
    hits = [note for note in vector.notes if fragment in note]
    assert len(hits) == 1, f"expected one note containing {fragment!r}, got {vector.notes}"
    return hits[0]


def has_note(vector, fragment: str) -> bool:
    return any(fragment in note for note in vector.notes)


def two_pivot_mids(n: int = 80) -> np.ndarray:
    """A flat tape carrying a swing high at bar 50 and a LOWER one at bar 70.

    The newer pivot being the lower one is the whole point: `swing_high` falls
    back to the window high when nothing is confirmed, so a series whose newer
    pivot was also its highest bar would report the new level early *through
    the fallback* and a confirmation-lag test on it would pass vacuously.
    """
    return np.full(n, BASE)


def two_pivot_series(n: int = 80) -> BarSeries:
    mids = two_pivot_mids(n)
    highs = mids + 1.0
    highs[50] = BASE + 5.0
    highs[70] = BASE + 3.0
    return series_from(
        grid_stamps(n), mids.copy(), highs, mids - 1.0, mids.copy(),
        np.full(n, 1000.0), FAST,
    )


def zigzag_series(n: int = 70, tail=None) -> BarSeries:
    """HH/HL geometry: lows at bars 20 and 40, highs at bars 30 and 50.

    low 95 -> high 105 -> low 96 -> high 107, so the last two confirmed
    pivots of each kind are a higher high AND a higher low, which is the only
    shape `sequence_direction` calls an uptrend. `tail` replaces the last
    three bars; three is `pivot_confirm_bars`, so whatever it does cannot
    itself confirm a pivot and the structure under test stays put.
    """
    mids = np.full(n, BASE)
    opens, highs, lows, closes = mids.copy(), mids + 1.0, mids - 1.0, mids.copy()
    lows[20] = 95.0
    highs[30] = 105.0
    lows[40] = 96.0
    highs[50] = 107.0
    if tail is not None:
        mid, half = float(tail), 0.5
        for i in (n - 3, n - 2, n - 1):
            opens[i] = closes[i] = mid
            highs[i] = mid + half
            lows[i] = mid - half
    return series_from(
        grid_stamps(n), opens, highs, lows, closes, np.full(n, 1000.0), FAST
    )


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
    return symbol_data(dataset.bars)


# ---------------------------------------------------------------------------
# the local ATR chain -- the scale every prominence is measured against
# ---------------------------------------------------------------------------


def test_true_range_matches_the_three_way_max_computed_by_hand():
    """A wrong ATR does not produce a wrong ATR feature here -- this module
    does not emit one. It rescales every pivot prominence threshold, so the
    error shows up as a different *set of pivots*, which no bound checks."""
    highs = np.array([10.0, 12.0, 11.0, 20.0, 15.0, 16.0])
    lows = np.array([9.0, 11.0, 8.0, 19.0, 6.0, 14.0])
    closes = np.array([9.5, 11.5, 10.0, 19.5, 14.0, 15.0])
    # bar 1: h-l=1.0,  |12-9.5|=2.5,  |11-9.5|=1.5   -> 2.5
    # bar 2: h-l=3.0,  |11-11.5|=0.5, |8-11.5|=3.5   -> 3.5
    # bar 3: h-l=1.0,  |20-10|=10.0,  |19-10|=9.0    -> 10.0
    # bar 4: h-l=9.0,  |15-19.5|=4.5, |6-19.5|=13.5  -> 13.5
    # bar 5: h-l=2.0,  |16-14|=2.0,   |14-14|=0.0    -> 2.0
    assert true_range(highs, lows, closes).tolist() == [2.5, 3.5, 10.0, 13.5, 2.0]


def test_true_range_drops_the_first_bar_rather_than_seeding_with_high_minus_low():
    """The first bar of a window has no previous close inside the window.
    Seeding it with `high - low` biases the first range downward and then
    feeds that bias straight into Wilder's seed."""
    series = grid_series([BASE] * 20)
    ranges = true_range(
        np.asarray(series.col("high")),
        np.asarray(series.col("low")),
        np.asarray(series.col("close")),
    )
    assert ranges.size == len(series) - 1
    assert ranges.tolist() == [2.0] * 19  # max(2, 1, 1) on a flat tape


def test_true_range_rejects_arrays_of_different_lengths():
    with pytest.raises(FeatureError, match="equal-length"):
        true_range(np.zeros(5), np.zeros(4), np.zeros(5))


def test_a_single_bar_window_has_no_true_range_at_all():
    one = np.array([1.0])
    assert true_range(one, one, one).size == 0


def test_wilder_atr_matches_an_exact_worked_example():
    """Period 2 over TR = 2.5, 3.5, 10, 13.5, 2, as exact halves.

    seed = (2.5 + 3.5) / 2 = 3.0
    next = (3.0 * 1 + 10)  / 2 = 6.5
    next = (6.5 * 1 + 13.5)/ 2 = 10.0
    next = (10.0 * 1 + 2)  / 2 = 6.0
    """
    out = wilder_atr_series(np.array([2.5, 3.5, 10.0, 13.5, 2.0]), 2)
    assert out.tolist() == pytest.approx([3.0, 6.5, 10.0, 6.0])


def test_wilder_atr_series_length_is_ranges_minus_period_plus_one():
    for size, period in ((40, 14), (12, 2), (514, 14)):
        assert wilder_atr_series(np.ones(size), period).size == size - period + 1


def test_wilder_atr_rejects_a_window_shorter_than_its_period():
    with pytest.raises(FeatureError, match="needs at least"):
        wilder_atr_series(np.ones(13), 14)


def test_wilder_atr_rejects_a_non_positive_period():
    with pytest.raises(FeatureError, match="at least 1"):
        wilder_atr_series(np.ones(13), 0)


def test_atr_aligned_puts_each_bars_atr_at_that_bars_index():
    """The alignment is the whole reason this helper exists: `find_pivots`
    compares a price distance at bar `i` against `atr[i]`, so an off-by-period
    here scales every pivot's prominence by a different bar's volatility --
    which changes the pivot set without changing anything that looks wrong."""
    highs = np.array([10.0, 12.0, 11.0, 20.0, 15.0, 16.0])
    lows = np.array([9.0, 11.0, 8.0, 19.0, 6.0, 14.0])
    closes = np.array([9.5, 11.5, 10.0, 19.5, 14.0, 15.0])
    out = atr_aligned(highs, lows, closes, 2)
    # TR = [2.5, 3.5, 10, 13.5, 2] for bars 1..5; the Wilder seed is the mean
    # of the first two, which is the ATR at bar 2, so out[j] is bar 2 + j.
    assert out.size == highs.size - 2
    assert out.tolist() == pytest.approx([3.0, 6.5, 10.0, 6.0])


def test_atr_aligned_rejects_a_window_too_short_for_one_value():
    """Returning an empty array instead would hand a caller something it then
    indexes at [-1]."""
    with pytest.raises(FeatureError, match="needs at least 15 bars"):
        atr_aligned(np.ones(14), np.ones(14), np.ones(14), 14)


def test_the_pivot_window_yields_exactly_one_atr_per_region_bar():
    """`warmup_bars = atr_period + pivot_region` is sized so the ATR series
    covers the search region exactly. One bar short and `compute` raises
    rather than silently searching a misaligned region -- which is the
    identity `compute` re-asserts at runtime."""
    config, levels = FeatureConfig(), StructureLevelConfig()
    comp = StructureFeatures(config, levels, nq_spec())
    window = comp.warmup_bars
    assert window == config.atr_period + levels.lookback_bars == 514
    flat = np.full(window, BASE)
    assert atr_aligned(flat + 1.0, flat - 1.0, flat, config.atr_period).size == 514 - 14
    assert (
        atr_aligned(flat[:-1] + 1.0, flat[:-1] - 1.0, flat[:-1], config.atr_period).size
        == 499
    )


def test_typical_price_is_the_three_way_mean():
    highs = np.array([10.0, 20.0])
    lows = np.array([4.0, 10.0])
    closes = np.array([7.0, 18.0])
    assert typical_price(highs, lows, closes).tolist() == pytest.approx(
        [(10.0 + 4.0 + 7.0) / 3.0, (20.0 + 10.0 + 18.0) / 3.0]
    )


def test_typical_price_rejects_arrays_of_different_lengths():
    with pytest.raises(FeatureError, match="equal-length"):
        typical_price(np.zeros(3), np.zeros(2), np.zeros(3))


# ---------------------------------------------------------------------------
# pivot confirmation lag -- section 14.1's load-bearing condition
# ---------------------------------------------------------------------------


def test_a_pivot_does_not_exist_until_k_confirm_bars_after_it_forms():
    """THE test in this file. Section 14.1: bar `i` is a confirmed swing high
    at evaluation time `t` only when `i <= t - k_confirm`. `find_pivots` is
    handed arrays that END at `t`, so the condition is the upper bound of the
    centre range -- and the way to check a slice bound is a slice bound is to
    extend the array one bar at a time and watch when the pivot appears.

    A centred argmax that ignored the bound would report the spike at bar 10
    from `t = 10` onward, which is a swing level the market had not yet
    revealed and the single most damaging error this module can make."""
    highs = np.full(21, BASE + 0.5)
    lows = np.full(21, BASE - 0.5)
    highs[10] = BASE + 5.0
    atr = np.full(21, 1.0)

    for last in range(10, 13):  # t = 10, 11, 12: fewer than k=3 bars after
        found = find_pivots(
            highs[: last + 1], lows[: last + 1], atr[: last + 1],
            confirm_bars=3, prominence_atr=0.4,
        )
        assert found == (), f"pivot at bar 10 reported at t={last}, before t-k"

    for last in range(13, 17):  # t >= 10 + 3: the pivot is knowable
        found = find_pivots(
            highs[: last + 1], lows[: last + 1], atr[: last + 1],
            confirm_bars=3, prominence_atr=0.4,
        )
        assert [p.index for p in found] == [10], f"pivot at bar 10 missing at t={last}"
        assert found[0].price == BASE + 5.0
        assert found[0].confirmed_index == 13


def test_the_confirmation_index_is_the_extreme_plus_k_confirm():
    """14.1's `confirmed_at_ts`, in window coordinates. It is the number a
    consumer uses to decide whether a level existed yet, so it is not allowed
    to be the extreme's own index."""
    highs = np.full(31, BASE + 0.5)
    lows = np.full(31, BASE - 0.5)
    highs[12] = BASE + 5.0
    for k in (1, 2, 5, 8):
        found = find_pivots(
            highs, lows, np.full(31, 1.0), confirm_bars=k, prominence_atr=0.4
        )
        assert [p.confirmed_index for p in found] == [12 + k]


def test_no_centre_ever_comes_from_the_last_k_bars():
    """The upper bound stated as a property rather than as one example: every
    centre must have `k` real bars after it inside the array."""
    rng = np.random.default_rng(4)
    highs = BASE + rng.uniform(0.0, 5.0, 200)
    lows = highs - 1.0 - rng.uniform(0.0, 5.0, 200)
    for k in (1, 3, 7):
        found = find_pivots(
            highs, lows, np.full(200, 1.0), confirm_bars=k, prominence_atr=0.4
        )
        assert found, "a random tape should produce some pivots"
        assert max(p.index for p in found) <= 200 - 1 - k
        assert min(p.index for p in found) >= k


def test_a_bar_that_only_ties_the_window_maximum_is_not_a_pivot():
    """14.1 writes the centred-extreme condition as an equality and the
    prominence condition separately. With `atr > 0` and `m > 0` the threshold
    is strictly positive, so the equality is implied -- a tie has prominence
    zero. Without that, a double top would report two pivots at the same
    price and `sequence_direction` would read a flat pair as a trend."""
    highs = np.full(11, BASE)
    lows = np.full(11, BASE - 1.0)
    atr = np.full(11, 1.0)
    assert find_pivots(highs, lows, atr, confirm_bars=3, prominence_atr=0.4) == ()

    highs[2] = BASE + 2.0  # a second bar at the same high as the centre
    highs[5] = BASE + 2.0
    found = find_pivots(highs, lows, atr, confirm_bars=3, prominence_atr=0.4)
    assert [p.index for p in found] == [], "a tie inside the window is not an extreme"


def test_prominence_exactly_at_the_threshold_qualifies_and_a_hair_below_does_not():
    """`>=` rather than `>`, and the boundary is where a config sweep will
    spend most of its time, so it is pinned rather than left to inference."""
    lows = np.full(11, BASE - 1.0)
    atr = np.full(11, 1.0)
    at = np.full(11, BASE)
    at[5] = BASE + 0.4  # prominence 0.4 == 0.4 * ATR 1.0
    found = find_pivots(at, lows, atr, confirm_bars=3, prominence_atr=0.4)
    assert [p.index for p in found] == [5]
    assert found[0].prominence_atr == pytest.approx(0.4)

    below = np.full(11, BASE)
    below[5] = BASE + 0.39999
    assert find_pivots(below, lows, atr, confirm_bars=3, prominence_atr=0.4) == ()


def test_prominence_atr_is_the_measured_margin_in_atr_units():
    """Reported rather than recomputed downstream, so section 14.3 can weight
    a pivot by how decisively it cleared the bar."""
    highs = np.full(11, BASE)
    highs[5] = BASE + 3.0
    found = find_pivots(
        highs, np.full(11, BASE - 1.0), np.full(11, 2.0),
        confirm_bars=3, prominence_atr=0.4,
    )
    assert found[0].prominence_atr == pytest.approx(3.0 / 2.0)  # 3 points / ATR 2


def test_a_bar_with_no_atr_produces_no_pivot_at_all():
    """With `ATR[i] == 0` the threshold is zero, and a `>= 0` test makes every
    bar of a halted tape both a swing high and a swing low. There is no scale
    on which prominence can be measured there, so the candidate is skipped."""
    highs = np.full(11, BASE)
    highs[5] = BASE + 3.0
    assert find_pivots(
        highs, np.full(11, BASE - 1.0), np.zeros(11),
        confirm_bars=3, prominence_atr=0.4,
    ) == ()
    partial = np.full(11, 1.0)
    partial[5] = 0.0
    assert find_pivots(
        highs, np.full(11, BASE - 1.0), partial, confirm_bars=3, prominence_atr=0.4
    ) == ()


def test_a_non_finite_atr_bar_is_skipped_rather_than_producing_a_nan_pivot():
    highs = np.full(11, BASE)
    highs[5] = BASE + 3.0
    atr = np.full(11, 1.0)
    atr[5] = np.nan
    assert find_pivots(
        highs, np.full(11, BASE - 1.0), atr, confirm_bars=3, prominence_atr=0.4
    ) == ()


def test_an_outside_bar_is_reported_as_two_pivots_not_one():
    """A bar that strictly dominates its neighbours in both directions really
    is both a swing high and a swing low. Collapsing it to one would drop a
    level the clustering in 14.2 is entitled to see."""
    highs = np.full(11, BASE)
    lows = np.full(11, BASE - 1.0)
    highs[5] = BASE + 2.0
    lows[5] = BASE - 3.0
    found = find_pivots(highs, lows, np.full(11, 1.0), confirm_bars=3, prominence_atr=0.4)
    assert [(p.index, p.is_high) for p in found] == [(5, True), (5, False)]


def test_pivots_come_back_sorted_by_index_with_highs_first_at_a_tie():
    highs = np.full(31, BASE)
    lows = np.full(31, BASE - 1.0)
    for i in (5, 12, 20):
        highs[i] = BASE + 2.0
    lows[12] = BASE - 4.0
    found = find_pivots(highs, lows, np.full(31, 1.0), confirm_bars=3, prominence_atr=0.4)
    assert [(p.index, p.is_high) for p in found] == [
        (5, True), (12, True), (12, False), (20, True)
    ]


def test_a_window_shorter_than_one_centred_span_yields_nothing():
    """`2k + 1` bars are needed before any centre has neighbours on both
    sides. Returning nothing is the only honest answer; raising would make a
    short view an error rather than a wait."""
    for total in (1, 5, 6):
        assert find_pivots(
            np.full(total, BASE), np.full(total, BASE - 1.0), np.full(total, 1.0),
            confirm_bars=3, prominence_atr=0.4,
        ) == ()


def test_find_pivots_rejects_a_misaligned_atr():
    with pytest.raises(FeatureError, match="equal-length"):
        find_pivots(
            np.zeros(20), np.zeros(20), np.zeros(19), confirm_bars=3, prominence_atr=0.4
        )


def test_find_pivots_rejects_zero_right_side_confirmation():
    """`confirm_bars=0` turns the centre range into `[0, len-1]`, so the
    evaluation bar itself becomes a candidate pivot. That is the lookahead
    this module exists to prevent, so it is refused rather than allowed."""
    with pytest.raises(FeatureError, match="at least 1"):
        find_pivots(
            np.zeros(20), np.zeros(20), np.ones(20), confirm_bars=0, prominence_atr=0.4
        )


def test_find_pivots_rejects_a_non_positive_prominence():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(FeatureError, match="positive finite"):
            find_pivots(
                np.zeros(20), np.zeros(20), np.ones(20),
                confirm_bars=3, prominence_atr=bad,
            )


def test_a_pivot_refuses_to_claim_confirmation_at_or_before_its_own_bar():
    """The model-level echo of the slice bound: a level is not known until at
    least one bar after it forms, whoever constructs the object."""
    with pytest.raises(ValueError, match="must exceed index"):
        Pivot(index=5, confirmed_index=5, price=BASE, is_high=True, prominence_atr=1.0)
    with pytest.raises(ValueError, match="must exceed index"):
        Pivot(index=5, confirmed_index=4, price=BASE, is_high=True, prominence_atr=1.0)


def test_a_pivot_refuses_a_non_finite_price():
    with pytest.raises(ValueError, match="not finite"):
        Pivot(
            index=5, confirmed_index=8, price=float("nan"), is_high=True,
            prominence_atr=1.0,
        )


def test_bars_since_counts_from_the_extreme_in_window_coordinates():
    pivot = Pivot(index=10, confirmed_index=13, price=BASE, is_high=True, prominence_atr=1.0)
    assert pivot.bars_since(13) == 3
    assert pivot.bars_since(40) == 30


# ---------------------------------------------------------------------------
# confirmation lag, end to end through `compute`
# ---------------------------------------------------------------------------


def test_swing_high_does_not_switch_to_a_newer_pivot_until_it_confirms():
    """The same rule through the real accessor path, where the window slides
    and the ATR is rebuilt on every call.

    Bar 50 carries a high of 105 and bar 70 a high of 103. The newer pivot is
    the LOWER one on purpose: `swing_high` falls back to the window high when
    nothing is confirmed, so if the newer pivot were also the highest bar the
    fallback would report it early and this test would pass for the wrong
    reason. With `pivot_confirm_bars = 3`, `swing_high` must read 105 at bars
    71 and 72 and 103 from bar 73 -- exactly `70 + 3`."""
    comp = computer(levels=pivot_levels())
    series = two_pivot_series()
    assert comp.warmup_bars == 2 + 60

    for t in (68, 71, 72):
        values = comp.compute(view_at(series, t)).values
        assert values["swing_high"] == BASE + 5.0, f"bar 70's pivot leaked at t={t}"
        assert values["bars_since_pivot"] == float(t - 50)

    for t in (73, 74, 76):
        values = comp.compute(view_at(series, t)).values
        assert values["swing_high"] == BASE + 3.0, f"bar 70's pivot missing at t={t}"
        assert values["bars_since_pivot"] == float(t - 70)


def test_bars_since_pivot_is_measured_from_the_extreme_not_the_confirmation():
    """At the confirmation bar the newest pivot is already `k` bars old. A
    `bars_since_pivot` of 0 there would claim the market turned on the bar it
    was merely found on, and any rule thresholding freshness would fire a
    confirmation lag too late."""
    comp = computer(levels=pivot_levels())
    values = comp.compute(view_at(two_pivot_series(), 73)).values
    assert values["bars_since_pivot"] == 3.0 == float(pivot_levels().pivot_confirm_bars)


def test_with_no_confirmed_pivot_bars_since_reports_a_sentinel_above_every_real_value():
    """A flat tape confirms nothing. Reporting 0, or the region length minus
    one, would be indistinguishable from a measurement; the sentinel is
    chosen to sit strictly above the oldest attainable value so a threshold
    on it cannot mistake 'none' for 'old'."""
    comp = computer(levels=pivot_levels())
    levels = pivot_levels()
    vector = comp.compute(view_at(grid_series([BASE] * 80)))
    oldest_attainable = levels.lookback_bars - 1 - levels.pivot_confirm_bars
    assert vector.values["bars_since_pivot"] == float(levels.lookback_bars)
    assert vector.values["bars_since_pivot"] > oldest_attainable
    assert "sentinel" in note_with(vector, "no confirmed pivot")


def test_without_a_confirmed_pivot_the_swings_fall_back_and_say_so():
    """The fallback is the window extreme, which is NOT a pivot, and the note
    says so. Silently reporting it as `swing_high` would put a level into
    14.2's clustering that never survived a confirmation test."""
    comp = computer(levels=pivot_levels())
    vector = comp.compute(view_at(grid_series([BASE] * 80)))
    assert vector.values["swing_high"] == BASE + 1.0
    assert vector.values["swing_low"] == BASE - 1.0
    assert "not a pivot" in note_with(vector, "no confirmed swing high")
    assert "not a pivot" in note_with(vector, "no confirmed swing low")


def test_a_flat_tape_reports_a_zero_atr_rather_than_dividing_by_it():
    """`high == low == close` on every bar makes every true range zero. Every
    ATR-scaled quantity here is then undefined, and the module has to produce
    fifteen finite numbers anyway."""
    comp = computer(levels=pivot_levels())
    flat = np.full(80, BASE)
    series = series_from(
        grid_stamps(80), flat, flat.copy(), flat.copy(), flat.copy(),
        np.full(80, 1000.0), FAST,
    )
    vector = comp.compute(view_at(series))
    assert has_note(vector, "ATR at the evaluation bar is zero")
    assert all(math.isfinite(value) for value in vector.values.values())
    assert vector.values["vwap_deviation_atr"] == 0.0
    assert vector.values["structure_score"] == 0.0
    assert vector.values["opening_range_position"] == 0.5
    assert has_note(vector, "opening range has zero width")


# ---------------------------------------------------------------------------
# HH/HL sequence
# ---------------------------------------------------------------------------


def pivot_at(price: float, index: int, is_high: bool) -> Pivot:
    return Pivot(
        index=index, confirmed_index=index + 3, price=price, is_high=is_high,
        prominence_atr=1.0,
    )


def test_a_higher_high_with_a_higher_low_is_an_uptrend():
    highs = (pivot_at(100.0, 10, True), pivot_at(105.0, 30, True))
    lows = (pivot_at(95.0, 20, False), pivot_at(96.0, 40, False))
    assert sequence_direction(highs, lows) == 1.0


def test_a_lower_high_with_a_lower_low_is_a_downtrend():
    highs = (pivot_at(105.0, 10, True), pivot_at(100.0, 30, True))
    lows = (pivot_at(96.0, 20, False), pivot_at(95.0, 40, False))
    assert sequence_direction(highs, lows) == -1.0


def test_a_widening_range_is_not_a_trend():
    """A higher high with a LOWER low is price expansion, not direction.
    Calling it +1 would make the feature agree with volatility, and a range
    is precisely where a structure trade should not fire."""
    highs = (pivot_at(100.0, 10, True), pivot_at(105.0, 30, True))
    lows = (pivot_at(96.0, 20, False), pivot_at(95.0, 40, False))
    assert sequence_direction(highs, lows) == 0.0


def test_an_equal_high_breaks_the_sequence_in_both_directions():
    """`>` and `<`, not `>=`. A double top is not a higher high."""
    highs = (pivot_at(105.0, 10, True), pivot_at(105.0, 30, True))
    lows = (pivot_at(95.0, 20, False), pivot_at(96.0, 40, False))
    assert sequence_direction(highs, lows) == 0.0


def test_fewer_than_two_pivots_of_either_kind_is_no_direction():
    high = (pivot_at(105.0, 10, True),)
    lows = (pivot_at(95.0, 20, False), pivot_at(96.0, 40, False))
    assert sequence_direction(high, lows) == 0.0
    assert sequence_direction(lows, high) == 0.0
    assert sequence_direction((), ()) == 0.0


def test_structure_direction_reads_the_last_two_confirmed_pivots_of_each_kind():
    """End to end on the zigzag: low 95 -> high 105 -> low 96 -> high 107 is
    the only shape that is an uptrend, and the emitted swings must be the
    MOST RECENT confirmed pivots, not the most extreme ones."""
    comp = computer(levels=pivot_levels())
    values = comp.compute(view_at(zigzag_series())).values
    assert values["structure_direction"] == 1.0
    assert values["swing_high"] == 107.0
    assert values["swing_low"] == 96.0  # the higher, more recent low -- not 95


# ---------------------------------------------------------------------------
# break of structure and change of character
# ---------------------------------------------------------------------------


def test_a_close_above_the_most_recent_confirmed_high_breaks_structure_up():
    """The three final bars do the breaking, and three is `pivot_confirm_bars`
    -- so nothing they contain can itself confirm a new pivot, and the level
    being broken is unambiguously the one the test placed at bar 50."""
    comp = computer(levels=pivot_levels())
    values = comp.compute(view_at(zigzag_series(tail=110.0))).values
    assert values["swing_high"] == 107.0
    assert values["bos_direction"] == 1.0


def test_a_close_below_the_most_recent_confirmed_low_breaks_structure_down():
    comp = computer(levels=pivot_levels())
    values = comp.compute(view_at(zigzag_series(tail=94.0))).values
    assert values["swing_low"] == 96.0
    assert values["bos_direction"] == -1.0


def test_a_close_inside_the_confirmed_swing_range_breaks_nothing():
    """The negative case, without which the two above would also pass on an
    implementation that always reported a break."""
    comp = computer(levels=pivot_levels())
    values = comp.compute(view_at(zigzag_series())).values
    assert values["swing_low"] < BASE < values["swing_high"]
    assert values["bos_direction"] == 0.0
    assert values["choch"] == 0.0


def test_a_break_against_an_established_sequence_is_a_change_of_character():
    """An uptrend sequence (HH and HL) broken to the DOWNSIDE. This is the
    only configuration that fires `choch`, and it is the one reading in this
    module that a careless implementation would make equal to `bos != 0`."""
    comp = computer(levels=pivot_levels())
    values = comp.compute(view_at(zigzag_series(tail=94.0))).values
    assert values["structure_direction"] == 1.0
    assert values["bos_direction"] == -1.0
    assert values["choch"] == 1.0


def test_a_break_that_runs_with_the_sequence_is_not_a_change_of_character():
    comp = computer(levels=pivot_levels())
    values = comp.compute(view_at(zigzag_series(tail=110.0))).values
    assert values["structure_direction"] == 1.0
    assert values["bos_direction"] == 1.0
    assert values["choch"] == 0.0


def test_a_break_with_no_established_sequence_is_not_a_change_of_character():
    """One confirmed high and no second one: there is no character to change.
    The series carries a single pivot at bar 50 and then closes above it."""
    comp = computer(levels=pivot_levels())
    mids = np.full(70, BASE)
    opens, highs, lows, closes = mids.copy(), mids + 1.0, mids - 1.0, mids.copy()
    highs[50] = 105.0
    for i in (67, 68, 69):
        opens[i] = closes[i] = 106.0
        highs[i] = 106.5
        lows[i] = 105.5
    series = series_from(
        grid_stamps(70), opens, highs, lows, closes, np.full(70, 1000.0), FAST
    )
    values = comp.compute(view_at(series)).values
    assert values["bos_direction"] == 1.0
    assert values["structure_direction"] == 0.0
    assert values["choch"] == 0.0


def test_no_confirmed_pivot_means_no_break_however_the_tape_moved():
    """`bos_direction` must come off a confirmed pivot, never off the window
    extreme the swings fall back to -- a break of a level that was never a
    level is not a break of structure."""
    comp = computer(levels=pivot_levels())
    rng = np.random.default_rng(9)
    drift = BASE + np.cumsum(rng.normal(0.0, 0.01, 80))  # too smooth to confirm
    vector = comp.compute(view_at(grid_series(drift)))
    assert has_note(vector, "no confirmed swing high")
    assert has_note(vector, "no confirmed swing low")
    assert vector.values["bos_direction"] == 0.0
    assert vector.values["choch"] == 0.0


def test_a_close_beyond_both_swings_is_resolved_by_pivot_recency():
    """Reachable when the most recent confirmed LOW sits above the most
    recent confirmed HIGH -- a one-way move that printed a higher low before
    printing a new high. The structure actually in force is the one confirmed
    more recently, so its break decides the sign, and the note says so rather
    than leaving an arbitrary choice unexplained.

    Geometry: a high pivot of 102 at bar 30, then a rally, then a low pivot
    of 105 at bar 50, then a drift back to a close of 102.5 -- above the
    swing high and below the swing low at the same time."""
    comp = computer(levels=pivot_levels())
    n = 70
    mid = np.empty(n)
    mid[0:41] = BASE
    mid[41:45] = [101.5, 103.0, 104.5, 106.0]
    mid[45:56] = 106.0
    mid[56:70] = np.linspace(105.75, 102.5, 14)
    opens, highs, lows, closes = mid.copy(), mid + 0.2, mid - 0.2, mid.copy()
    highs[30] = 102.0
    lows[50] = 105.0
    series = series_from(
        grid_stamps(n), opens, highs, lows, closes, np.full(n, 1000.0), FAST
    )
    vector = comp.compute(view_at(series))
    assert vector.values["swing_high"] == 102.0
    assert vector.values["swing_low"] == 105.0
    assert float(closes[-1]) == 102.5  # beyond both
    assert vector.values["bos_direction"] == -1.0  # the low at bar 50 is the newer
    assert "resolved by pivot recency" in note_with(vector, "beyond both")


# ---------------------------------------------------------------------------
# VWAP: volume-weighted, and anchored
# ---------------------------------------------------------------------------

#: Three consecutive trading days. Used by every session-anchor test that
#: does not need a weekend, a holiday or a DST transition: three sessions is
#: the smallest number that puts the PRIOR session somewhere other than bar
#: zero of the window, which is what keeps the quality GOOD and therefore
#: keeps the truncation tests below meaningful.
THREE_DAYS = [date(2024, 1, 17), date(2024, 1, 18), date(2024, 1, 19)]


def test_volume_weighted_average_is_weighted_by_volume_not_by_bar_count():
    """Prices 100, 110, 120 with volumes 1, 1, 98:

        (100*1 + 110*1 + 120*98) / 100 = 11970 / 100 = 119.7

    An arithmetic mean gives 110.0. The volumes are deliberately lopsided so
    the two answers are ten points apart -- a "VWAP" that was secretly a mean
    cannot survive this arithmetic, where on balanced volume it would."""
    prices = np.array([100.0, 110.0, 120.0])
    volumes = np.array([1.0, 1.0, 98.0])
    vwap, weighted = volume_weighted_average(prices, volumes)
    assert weighted is True
    assert vwap == pytest.approx(119.7)
    assert float(prices.mean()) == 110.0


def test_a_window_with_no_volume_reports_the_mean_and_flags_that_it_did():
    """There is no volume distribution to weight by, so the unweighted mean is
    the honest fallback -- but it is a different statistic, so the flag comes
    back rather than being swallowed."""
    prices = np.array([100.0, 110.0, 120.0])
    assert volume_weighted_average(prices, np.zeros(3)) == (110.0, False)


def test_volume_weighted_average_rejects_empty_or_misaligned_input():
    with pytest.raises(FeatureError, match="non-empty"):
        volume_weighted_average(np.zeros(0), np.zeros(0))
    with pytest.raises(FeatureError, match="equal-length"):
        volume_weighted_average(np.zeros(3), np.zeros(2))


def test_the_session_vwap_is_volume_weighted_end_to_end():
    """The same lopsided volumes through the real path, on the first three
    bars of a session so the anchor is unambiguous."""
    comp = computer(calendar=TradingCalendar())
    mids = session_mids([100.0, 100.0, 100.0], SLOW)
    mids[26:29] = [100.0, 110.0, 120.0]  # the visible bars of session three
    volumes = np.full(mids.size, 1.0)
    volumes[28] = 98.0
    vector = comp.compute(view_at(rth_series(THREE_DAYS, mids, SLOW, volumes), 28))
    assert vector.values["vwap"] == pytest.approx(119.7)
    assert vector.quality == DataQuality.GOOD


def test_the_vwap_resets_at_the_session_anchor():
    """Each session trades at a completely different price: 100, then 200,
    then 300. A VWAP that failed to re-anchor would report roughly 200 at the
    end of the third session, so the test is a hundred points wide."""
    comp = computer(calendar=TradingCalendar())
    series = rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    vector = comp.compute(view_at(series))
    assert vector.values["vwap"] == pytest.approx(300.0)
    assert vector.quality == DataQuality.GOOD


def test_a_single_zero_volume_bar_drops_out_of_the_weighting():
    """The data layer flags but keeps zero-volume bars, so they arrive here.
    Prices 100, 110, 130 with volumes 1, 0, 1 weight to 115.0; including the
    zero-volume bar as an equal member would give 113.33."""
    comp = computer(calendar=TradingCalendar())
    mids = session_mids([100.0, 100.0, 100.0], SLOW)
    mids[26:29] = [100.0, 110.0, 130.0]
    volumes = np.full(mids.size, 1.0)
    volumes[27] = 0.0
    vector = comp.compute(view_at(rth_series(THREE_DAYS, mids, SLOW, volumes), 28))
    assert vector.values["vwap"] == pytest.approx(115.0)
    assert vector.quality == DataQuality.GOOD
    assert not has_note(vector, "unweighted mean")


def test_a_session_with_no_volume_at_all_degrades_rather_than_renaming_the_mean():
    """A VWAP that is quietly an arithmetic mean is a different statistic
    wearing the same name, which is this module's own definition of a
    substitution -- so it downgrades like every other one.

    This asserted GOOD before the fix: the note was emitted but
    `substituted` was never set, so the vector claimed full quality for a
    number computed without the weights its key names."""
    comp = computer(calendar=TradingCalendar())
    mids = session_mids([100.0, 100.0, 100.0], SLOW)
    mids[26:29] = [100.0, 110.0, 130.0]
    volumes = np.full(mids.size, 1.0)
    volumes[26:29] = 0.0
    vector = comp.compute(view_at(rth_series(THREE_DAYS, mids, SLOW, volumes), 28))
    assert vector.values["vwap"] == pytest.approx((100.0 + 110.0 + 130.0) / 3.0)
    assert "unweighted mean typical price" in note_with(vector, "total volume over")
    assert vector.quality == DataQuality.DEGRADED


def test_vwap_deviation_is_the_close_minus_the_vwap_in_atr_units():
    """Hand-computed over the third session: twelve bars at a typical price of
    300 and a last bar at 310.

        vwap = (300 * 12 + 310) / 13 = 3910 / 13 = 300.76923...

    The ATR comes off the 13-bar pivot window, which is exactly that session.
    Its first bar contributes no true range, the next eleven are flat at
    `max(2, 1, 1) = 2`, and the last is `max(2, |311-300|, |309-300|) = 11`:

        seed  = (2 + 2) / 2 = 2        (and stays 2 through the flat bars)
        final = (2 * 1 + 11) / 2 = 6.5
    """
    comp = computer(calendar=TradingCalendar())
    mids = session_mids([100.0, 200.0, 300.0], SLOW)
    mids[-1] = 310.0
    vector = comp.compute(view_at(rth_series(THREE_DAYS, mids, SLOW)))
    expected_vwap = (300.0 * 12 + 310.0) / 13.0
    assert vector.values["vwap"] == pytest.approx(expected_vwap)
    assert vector.values["vwap_deviation_atr"] == pytest.approx(
        (310.0 - expected_vwap) / 6.5
    )


def test_the_week_anchor_runs_from_the_first_session_of_the_iso_week():
    """`vwap_anchor` is config, and the three settings are different
    quantities. Thursday and Friday of ISO week 3 trade at 100, Monday at 200
    and Tuesday at 300; a week-anchored VWAP on Tuesday is the mean of Monday
    and Tuesday (250), while a session-anchored one is 300."""
    days = [date(2024, 1, 18), date(2024, 1, 19), date(2024, 1, 22), date(2024, 1, 23)]
    assert [d.isocalendar()[:2] for d in days] == [
        (2024, 3), (2024, 3), (2024, 4), (2024, 4)
    ]
    series = rth_series(days, session_mids([100.0, 100.0, 200.0, 300.0], SLOW), SLOW)
    weekly = computer(features=short_atr(vwap_anchor="week"), calendar=TradingCalendar())
    sessionly = computer(calendar=TradingCalendar())
    assert weekly.compute(view_at(series)).values["vwap"] == pytest.approx(250.0)
    assert sessionly.compute(view_at(series)).values["vwap"] == pytest.approx(300.0)


def test_the_day_anchor_rolls_at_local_midnight_and_the_session_anchor_does_not():
    """The three `vwap_anchor` settings are different quantities, and for an
    instrument whose session wraps midnight `session` and `day` disagree by
    construction: an 18:00 bar starts a new trade date but not a new calendar
    day. Tested where the two must differ -- an overnight group that straddles
    local midnight.

    Ten Sunday-evening bars trade at 200 and five post-midnight bars at 300,
    all inside Monday's trade date. The session anchor averages all fifteen
    (233.33); the day anchor re-anchors at 00:00 and reports 300."""
    calendar = TradingCalendar()
    sunday_evening = datetime(2024, 1, 21, 19, 0, tzinfo=TZ)
    monday_early = datetime(2024, 1, 22, 0, 0, tzinfo=TZ)
    assert calendar.session_date(sunday_evening, nq_spec()) == date(2024, 1, 22)
    assert calendar.session_date(monday_early, nq_spec()) == date(2024, 1, 22)

    stamps = np.array(
        list(rth_stamps([date(2024, 1, 19)], SLOW))
        + [to_ns(sunday_evening + timedelta(minutes=30 * i)) for i in range(10)]
        + [to_ns(monday_early + timedelta(minutes=30 * i)) for i in range(5)],
        dtype=np.int64,
    )
    mids = np.concatenate(
        (np.full(SLOW_BARS_PER_SESSION, 100.0), np.full(10, 200.0), np.full(5, 300.0))
    )
    series = from_mids(stamps, mids, SLOW)

    by_session = computer(calendar=calendar).compute(view_at(series)).values["vwap"]
    by_day = (
        computer(features=short_atr(vwap_anchor="day"), calendar=calendar)
        .compute(view_at(series))
        .values["vwap"]
    )
    assert by_session == pytest.approx((200.0 * 10 + 300.0 * 5) / 15.0)
    assert by_day == pytest.approx(300.0)


def test_the_day_anchor_survives_the_repeated_hour_of_the_autumn_transition():
    """`_first_index_of_group` finds the anchor by bisection, which is only
    valid because the group key is non-decreasing in the timestamp. The
    autumn fall-back is where that could break: 01:00 EDT and 01:00 EST are
    an hour apart in UTC and identical on the wall clock.

    Five bars are placed across the repeated hour on 2024-11-03 (00:30 EDT,
    01:00 EDT, 01:30 EDT, 01:00 EST, 01:30 EST -- ascending in UTC, not in
    local time) and all share the local date, so the day anchor must sit at
    the first of them and the VWAP must be 400 rather than a blend with the
    previous day's 100."""
    repeated = [
        datetime(2024, 11, 3, 4, 30, tzinfo=timezone.utc) + timedelta(minutes=30 * i)
        for i in range(5)
    ]
    local = [ts.astimezone(TZ) for ts in repeated]
    assert [ts.date() for ts in local] == [date(2024, 11, 3)] * 5
    assert [(ts.hour, ts.minute) for ts in local] == [
        (0, 30), (1, 0), (1, 30), (1, 0), (1, 30)
    ]

    saturday = datetime(2024, 11, 2, 12, 0, tzinfo=TZ)
    stamps = np.array(
        [to_ns(saturday + timedelta(minutes=30 * i)) for i in range(13)]
        + [to_ns(ts) for ts in repeated],
        dtype=np.int64,
    )
    mids = np.concatenate((np.full(13, 100.0), np.full(5, 400.0)))
    comp = computer(features=short_atr(vwap_anchor="day"), calendar=TradingCalendar())
    assert comp.compute(view_at(from_mids(stamps, mids, SLOW))).values[
        "vwap"
    ] == pytest.approx(400.0)


def test_each_anchor_declares_how_far_back_a_boundary_can_lie():
    """Monday's prior session is Friday, three calendar days back, so a
    two-day window would silently substitute Monday's own statistics for
    Friday's. The week anchor needs a full ISO week plus a weekend either
    side."""
    assert SESSION_WINDOW_DAYS == {"session": 4, "day": 4, "week": 11}


# ---------------------------------------------------------------------------
# session anchors: the calendar is the quantity
# ---------------------------------------------------------------------------


def distinctive_session(mids: np.ndarray, start: int, per: int) -> np.ndarray:
    """Give one session an interior high of 205 and an interior low of 195.

    Interior, so the session's CLOSE stays at its baseline -- which is what
    lets a test distinguish `prior_session_close` from `prior_session_high`.
    """
    mids = mids.copy()
    mids[start + 4] = 205.0
    mids[start + 7] = 195.0
    return mids


def test_the_prior_session_is_friday_when_today_is_monday():
    """Across a weekend. `session_date` is pure calendar arithmetic and is not
    rolled forward, so this is a test that the *search* walks back to the
    previous group that has bars rather than assuming yesterday."""
    days = [date(2024, 1, 18), date(2024, 1, 19), date(2024, 1, 22)]  # Thu Fri Mon
    assert days[1].weekday() == 4 and days[2].weekday() == 0
    per = SLOW_BARS_PER_SESSION
    mids = distinctive_session(session_mids([100.0, 200.0, 300.0], SLOW), per, per)
    vector = computer(calendar=TradingCalendar()).compute(
        view_at(rth_series(days, mids, SLOW))
    )
    assert vector.values["prior_session_high"] == 206.0  # Friday's 205 + 1
    assert vector.values["prior_session_low"] == 194.0  # Friday's 195 - 1
    assert vector.values["prior_session_close"] == 200.0  # Friday's LAST close
    assert vector.quality == DataQuality.GOOD


def test_the_prior_session_skips_a_holiday_the_exchange_was_shut_for():
    """Monday 2024-01-15 is MLK Day. Tuesday's prior session is the previous
    Friday, and a prior-session anchor that counted calendar days back would
    land on a day with no bars at all."""
    calendar = TradingCalendar()
    assert not calendar.is_trading_day(date(2024, 1, 15), nq_spec())
    days = [date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 16)]  # Thu Fri Tue
    per = SLOW_BARS_PER_SESSION
    mids = distinctive_session(session_mids([100.0, 200.0, 300.0], SLOW), per, per)
    vector = computer(calendar=calendar).compute(view_at(rth_series(days, mids, SLOW)))
    assert vector.values["prior_session_high"] == 206.0
    assert vector.values["prior_session_low"] == 194.0
    assert vector.values["prior_session_close"] == 200.0
    assert vector.quality == DataQuality.GOOD


def test_bars_that_fall_on_a_holiday_have_no_regular_session_and_degrade():
    """`rth_bounds` returns None for a day the market was shut, so there is no
    opening-range window to be inside. The documented fallback is the first
    bars of the session group, and it is reported rather than passed off as a
    real opening range."""
    days = [date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 15)]  # the last is MLK
    series = rth_series(days, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    vector = computer(calendar=TradingCalendar()).compute(view_at(series))
    assert "holiday" in note_with(vector, "no regular session")
    assert vector.quality == DataQuality.DEGRADED
    assert vector.values["opening_range_high"] == 301.0


def test_the_opening_range_follows_the_clock_across_the_spring_dst_transition():
    """09:30 New York is 14:30 UTC in March and 13:30 UTC in April. A frozen
    UTC offset would read the wrong six bars after the transition.

    Monday 2024-03-11 is the first session on EDT. Its first six 5-minute
    bars (09:35-10:00 local) trade at 230 and the next six at 250, with the
    rest of the session at 200. A correct opening range reports 231; an
    offset frozen at EST would start an hour late and report 201 or 251."""
    days = [date(2024, 3, 6), date(2024, 3, 7), date(2024, 3, 8), date(2024, 3, 11)]
    calendar = TradingCalendar()
    opens = [calendar.rth_bounds(d, nq_spec())[0].hour for d in (days[2], days[3])]
    assert opens == [14, 13], "the DST transition is not where this test thinks"

    per = FAST_BARS_PER_SESSION
    mids = np.full(4 * per, 200.0)
    mids[3 * per : 3 * per + 6] = 230.0
    mids[3 * per + 6 : 3 * per + 12] = 250.0
    vector = computer(calendar=calendar).compute(view_at(rth_series(days, mids, FAST)))
    assert vector.values["opening_range_high"] == 231.0
    assert vector.values["opening_range_low"] == 229.0
    assert vector.quality == DataQuality.GOOD


def test_the_opening_range_follows_the_clock_across_the_autumn_dst_transition():
    """The other direction, because an hour added and an hour removed are not
    the same bug. Monday 2024-11-04 is the first session back on EST."""
    days = [date(2024, 10, 30), date(2024, 10, 31), date(2024, 11, 1), date(2024, 11, 4)]
    calendar = TradingCalendar()
    opens = [calendar.rth_bounds(d, nq_spec())[0].hour for d in (days[2], days[3])]
    assert opens == [13, 14], "the DST transition is not where this test thinks"

    per = FAST_BARS_PER_SESSION
    mids = np.full(4 * per, 200.0)
    mids[3 * per : 3 * per + 6] = 240.0
    mids[3 * per + 6 : 3 * per + 12] = 260.0
    vector = computer(calendar=calendar).compute(view_at(rth_series(days, mids, FAST)))
    assert vector.values["opening_range_high"] == 241.0
    assert vector.values["opening_range_low"] == 239.0
    assert vector.quality == DataQuality.GOOD


def test_a_still_forming_opening_range_reports_the_range_so_far_and_says_so():
    """The question the first bars of a session have to answer. Reporting a
    partially-formed range as a complete one would put a level into 14.2 that
    the session had not finished drawing.

    Six 5-minute bars make up a 30-minute range. The six bars rise, so the
    range after two bars (202) is visibly narrower than the finished range
    (211), and the note names how many of the six have closed."""
    per = FAST_BARS_PER_SESSION
    mids = np.full(3 * per, 200.0)
    mids[per : per + 6] = 150.0  # yesterday's opening range, far away
    mids[2 * per : 2 * per + 6] = [200.0, 201.0, 202.0, 203.0, 204.0, 210.0]
    series = rth_series(THREE_DAYS, mids, FAST)
    comp = computer(calendar=TradingCalendar())

    first = comp.compute(view_at(series, 2 * per))
    assert first.values["opening_range_high"] == 201.0
    assert "1 of" in note_with(first, "still forming")

    second = comp.compute(view_at(series, 2 * per + 1))
    assert second.values["opening_range_high"] == 202.0
    assert second.values["opening_range_low"] == 199.0
    assert "2 of its 6 bars" in note_with(second, "still forming")

    # And nothing from yesterday's range leaked in while today's was forming.
    assert second.values["opening_range_high"] != 151.0


def test_a_complete_opening_range_does_not_claim_to_be_forming():
    """The sixth bar closes exactly at the 30-minute boundary, so the range is
    complete at that bar -- an off-by-one on the `<= end` comparison would
    either drop that bar or keep calling the range unfinished for ever."""
    per = FAST_BARS_PER_SESSION
    mids = np.full(3 * per, 200.0)
    mids[2 * per : 2 * per + 6] = [200.0, 201.0, 202.0, 203.0, 204.0, 210.0]
    series = rth_series(THREE_DAYS, mids, FAST)
    comp = computer(calendar=TradingCalendar())

    for offset in (5, 6, 20):
        vector = comp.compute(view_at(series, 2 * per + offset))
        assert vector.values["opening_range_high"] == 211.0
        assert vector.values["opening_range_low"] == 199.0
        assert not has_note(vector, "still forming"), f"offset {offset}"


def test_a_bar_before_the_regular_open_reports_session_to_date_extremes():
    """An overnight bar belongs to the next trade date (the session rolls at
    18:00), so it is inside a session whose opening range has not started.
    There is nothing to report, and yesterday's range is not an answer -- so
    the module reports the session so far, says so, and degrades.

    The three Sunday-evening bars trade at 310, 330, 320, nowhere near either
    the previous session or the opening range that is still hours away."""
    calendar = TradingCalendar()
    sunday = datetime(2024, 1, 21, 19, 0, tzinfo=TZ)
    assert calendar.session_date(sunday, nq_spec()) == date(2024, 1, 22)

    stamps = np.concatenate(
        (
            rth_stamps([date(2024, 1, 18), date(2024, 1, 19)], SLOW),
            np.array(
                [to_ns(sunday + timedelta(minutes=30 * i)) for i in range(3)],
                dtype=np.int64,
            ),
        )
    )
    mids = np.concatenate(
        (
            session_mids([100.0, 200.0], SLOW),
            np.array([310.0, 330.0, 320.0]),
        )
    )
    vector = computer(calendar=calendar).compute(
        view_at(from_mids(stamps, mids, SLOW))
    )
    assert has_note(vector, "no bar of this session has closed inside the opening-range")
    assert vector.quality == DataQuality.DEGRADED
    assert vector.values["opening_range_high"] == 331.0  # the evening bars only
    assert vector.values["opening_range_low"] == 309.0
    assert vector.values["prior_session_close"] == 200.0  # Friday, not the window edge


def test_opening_range_position_is_clipped_to_the_unit_interval():
    """A bounded feature, so a breakout far above the range reports 1.0 rather
    than a number a weighted sum could be dominated by. The cost -- that the
    size of the breakout is not recoverable from this key -- is the reason the
    unclipped distance is published as `vwap_deviation_atr` instead."""
    per = SLOW_BARS_PER_SESSION
    comp = computer(calendar=TradingCalendar())
    for tail, expected in ((130.0, 1.0), (70.0, 0.0)):
        mids = np.full(3 * per, 100.0)
        mids[2 * per + 1 :] = tail
        vector = comp.compute(view_at(rth_series(THREE_DAYS, mids, SLOW)))
        assert vector.values["opening_range_high"] == 101.0
        assert vector.values["opening_range_position"] == expected


def test_a_zero_width_opening_range_reports_a_half_position():
    """A defined fallback for a ratio that does not exist, which is what
    `_vector` demands instead of a non-finite value."""
    per = SLOW_BARS_PER_SESSION
    mids = np.full(3 * per, 100.0)
    stamps = rth_stamps(THREE_DAYS, SLOW)
    series = series_from(
        stamps, mids.copy(), mids.copy(), mids.copy(), mids.copy(),
        np.full(mids.size, 1000.0), SLOW,
    )
    vector = computer(calendar=TradingCalendar()).compute(view_at(series))
    assert vector.values["opening_range_position"] == 0.5
    assert has_note(vector, "zero width")


def test_without_a_calendar_every_session_anchor_is_substituted_and_degraded():
    """There are no sessions without a calendar, so the whole window becomes
    one. A window VWAP anchored eighty bars ago is not a session VWAP, and the
    module says exactly that rather than reporting it under the same name
    silently."""
    comp = computer(levels=pivot_levels())
    series = grid_series(np.full(80, BASE))
    vector = comp.compute(view_at(series))
    assert "is NOT a session VWAP" in note_with(vector, "no calendar supplied")
    assert has_note(vector, "no calendar: the opening range")
    assert vector.quality == DataQuality.DEGRADED
    # The prior-session anchors become window statistics, which the note names.
    assert vector.values["prior_session_high"] == BASE + 1.0
    assert vector.values["prior_session_low"] == BASE - 1.0
    assert vector.values["prior_session_close"] == BASE


# ---------------------------------------------------------------------------
# warmup, and the honesty of the window the module actually reads
# ---------------------------------------------------------------------------


def test_warmup_bars_is_the_pivot_window_and_states_both_of_its_terms():
    """`atr_period` bars are consumed before the first ATR exists, and the
    search region needs one ATR per bar, so the window is their sum. The
    `2k + 1` floor matters on its own: without it a config with
    `lookback_bars=11, pivot_confirm_bars=10` would declare a warmup, pass
    it, and then never find a pivot, because no centred span would fit."""
    assert StructureFeatures(
        FeatureConfig(), StructureLevelConfig(), nq_spec()
    ).warmup_bars == 514 == 14 + 500

    narrow = StructureFeatures(
        FeatureConfig(),
        session_levels(pivot_confirm_bars=10),
        nq_spec(),
    )
    assert narrow.warmup_bars == 14 + (2 * 10 + 1) == 35


def test_not_ready_one_bar_before_warmup_and_ready_exactly_at_warmup():
    """Off by one here produces a number from a half-filled pivot window that
    looks exactly like a feature. `compute` re-checks warmup itself rather
    than trusting the bundle, because the lookahead audit calls computers
    directly including at pre-warmup bars."""
    comp = computer(calendar=TradingCalendar())
    series = rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    n = comp.warmup_bars

    short = comp.compute(view_at(series, n - 2))
    assert not short.warmup_complete
    assert f"warmup incomplete: {n - 1} of {n} bars" in short.notes[0]

    exact = comp.compute(view_at(series, n - 1))
    assert exact.warmup_complete


def test_extra_history_in_front_does_not_move_a_single_value():
    """The claim that makes a 514-bar warmup defensible while `compute` reads
    up to `SESSION_WINDOW_DAYS` days of bars: the anchors are functions of the
    current and prior session alone, so they stop changing as soon as both are
    visible. If they did not, the same bar would produce different features in
    bar 300 of a backtest and in bar 30_000."""
    comp = computer(calendar=TradingCalendar())
    per = SLOW_BARS_PER_SESSION
    rng = np.random.default_rng(3)
    tail = 100.0 + np.cumsum(rng.normal(0.0, 0.2, 3 * per))
    older = 100.0 + np.cumsum(rng.normal(0.0, 0.2, 5 * per))

    short_days = [date(2024, 1, 17), date(2024, 1, 18), date(2024, 1, 19)]
    long_days = [
        date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10),
        date(2024, 1, 11), date(2024, 1, 12),
    ] + short_days

    brief = comp.compute(view_at(rth_series(short_days, tail, SLOW)))
    ample = comp.compute(
        view_at(rth_series(long_days, np.concatenate((older, tail)), SLOW))
    )
    assert brief.values == pytest.approx(ample.values)
    assert brief.quality == ample.quality == DataQuality.GOOD


def test_a_window_that_cannot_reach_over_the_prior_session_degrades():
    """`warmup_bars` is the pivot window, which for an instrument whose
    session is longer than it does not reach back over the prior session at
    all -- and the prior session's high and low are then the extremes of
    however much of it happens to be visible.

    Session B really traded up to 241. A window holding only its last five
    bars reports 201 instead, and the only thing standing between that and a
    fabricated level downstream is the note and the downgrade.

    This asserted GOOD before the fix: `window_is_cut_short` was
    `count > size`, which is true only when the view holds MORE bars than the
    module asked for. The case that matters -- the view holding FEWER -- was
    never detected, so a truncated prior session came back at full quality."""
    per = FAST_BARS_PER_SESSION
    mids = session_mids([100.0, 200.0, 300.0], FAST)
    mids[per + 20] = 240.0  # session B's real high, 20 bars into the session
    stamps = rth_stamps(THREE_DAYS, FAST)
    comp = computer(calendar=TradingCalendar())

    whole = comp.compute(view_at(from_mids(stamps, mids, FAST)))
    assert whole.values["prior_session_high"] == 241.0
    assert whole.quality == DataQuality.GOOD

    keep = slice(2 * per - 5, None)
    truncated = comp.compute(view_at(from_mids(stamps[keep], mids[keep], FAST), 12))
    assert truncated.warmup_complete  # 13 bars is exactly `warmup_bars`
    assert truncated.values["prior_session_high"] == 201.0  # NOT session B's high
    assert "may be truncated" in note_with(truncated, "the prior session starts at")
    assert truncated.quality == DataQuality.DEGRADED


def test_a_dataset_with_no_prior_session_at_all_says_so():
    """One session of bars. There is no prior session to report, so the window
    extremes stand in -- named, not passed off."""
    comp = computer(calendar=TradingCalendar())
    series = rth_series([date(2024, 1, 17)], np.full(SLOW_BARS_PER_SESSION, 100.0), SLOW)
    vector = comp.compute(view_at(series))
    assert "no prior session is visible" in note_with(vector, "no prior session")
    assert vector.quality == DataQuality.DEGRADED
    assert vector.values["prior_session_high"] == 101.0
    assert vector.values["prior_session_low"] == 99.0


# ---------------------------------------------------------------------------
# structure_score
# ---------------------------------------------------------------------------


def test_the_score_weights_sum_to_one_which_is_what_bounds_it():
    assert STRUCTURE_SCORE_WEIGHTS == (0.40, 0.35, 0.25)
    assert sum(STRUCTURE_SCORE_WEIGHTS) == pytest.approx(1.0, abs=1e-12)


def test_the_score_refuses_to_run_if_the_weights_stop_summing_to_one(monkeypatch):
    """The [0, 1] bound comes from that sum and nothing else -- no clipping is
    doing load-bearing work -- so the guard is the only thing that would catch
    a later edit to the constants."""
    monkeypatch.setattr(
        "flow_model.features.structure.STRUCTURE_SCORE_WEIGHTS", (0.5, 0.5, 0.5)
    )
    with pytest.raises(FeatureError, match="expected 1.0"):
        structure_geometry_score(1.0, 1.0, 0.0)


def test_the_score_is_the_weighted_sum_of_the_three_readings():
    """Written out with the literal weights and `math.tanh` rather than by
    calling the module back, so this checks the arithmetic rather than the
    determinism."""
    assert structure_geometry_score(1.0, 1.0, 0.0) == pytest.approx(0.40 + 0.35)
    assert structure_geometry_score(-1.0, -1.0, 0.0) == pytest.approx(0.75)
    assert structure_geometry_score(1.0, 1.0, 2.0) == pytest.approx(
        0.40 + 0.35 + 0.25 * math.tanh(2.0 / VWAP_DEVIATION_SCALE)
    )
    assert structure_geometry_score(0.0, 0.0, -3.0) == pytest.approx(
        0.25 * math.tanh(3.0 / VWAP_DEVIATION_SCALE)
    )


def test_opposing_readings_cancel_toward_zero():
    """The absolute value is the *magnitude* half of the STRUCTURE component;
    14.6 keeps magnitude and direction separate, so a score near zero means
    the readings disagree, not that the setup is bearish."""
    assert structure_geometry_score(1.0, -1.0, 0.0) == pytest.approx(0.05)
    assert structure_geometry_score(-1.0, 1.0, 0.0) == pytest.approx(0.05)
    # An uptrend sequence broken downward with price at VWAP: near-perfect
    # cancellation, and the direction is read off the two emitted signs.
    assert structure_geometry_score(1.0, -1.0, 0.0) < 0.1


def test_the_vwap_term_keeps_the_sign_of_the_deviation():
    """`squash` returns a magnitude, so the sign has to be put back. Dropping
    it would make a close below VWAP agree with an uptrend."""
    above = structure_geometry_score(1.0, 0.0, 1.0)
    below = structure_geometry_score(1.0, 0.0, -1.0)
    assert above > 0.40 > below


def test_a_non_finite_vwap_deviation_contributes_nothing():
    for bad in (float("inf"), float("-inf"), float("nan")):
        assert structure_geometry_score(0.0, 0.0, bad) == 0.0
        assert structure_geometry_score(1.0, 1.0, bad) == pytest.approx(0.75)


def test_the_emitted_score_matches_the_formula_on_the_emitted_readings():
    """End to end, so the composition inside `compute` is checked and not just
    the helper -- a swapped argument there would be invisible to every test
    above."""
    comp = computer(levels=pivot_levels())
    values = comp.compute(view_at(zigzag_series(tail=94.0))).values
    deviation = values["vwap_deviation_atr"]
    expected = abs(
        0.40 * values["structure_direction"]
        + 0.35 * values["bos_direction"]
        + 0.25 * math.copysign(math.tanh(abs(deviation)), deviation)
    )
    assert values["structure_score"] == pytest.approx(expected)


def test_clip01_clamps_both_ends_and_passes_the_interior_through():
    assert _clip01(-5.0) == 0.0
    assert _clip01(0.0) == 0.0
    assert _clip01(0.25) == 0.25
    assert _clip01(1.0) == 1.0
    assert _clip01(17.0) == 1.0


# ---------------------------------------------------------------------------
# the declared contract
# ---------------------------------------------------------------------------


def test_the_emitted_keys_are_exactly_the_declared_fifteen():
    """`FeatureBundle` checks a computer produced the keys it promised, so a
    renamed key here is an error at the source rather than a `None` that
    propagates into a score."""
    comp = computer(calendar=TradingCalendar())
    assert comp.keys == (
        "swing_high", "swing_low", "structure_direction", "bos_direction", "choch",
        "bars_since_pivot", "vwap", "vwap_deviation_atr", "opening_range_high",
        "opening_range_low", "opening_range_position", "prior_session_high",
        "prior_session_low", "prior_session_close", "structure_score",
    )
    assert len(comp.keys) == len(set(comp.keys)) == 15

    vector = comp.compute(
        view_at(rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW))
    )
    assert tuple(sorted(vector.values)) == tuple(sorted(comp.keys))
    assert all(math.isfinite(value) for value in vector.values.values())


def test_every_emitted_key_carries_a_quality_grade():
    comp = computer(calendar=TradingCalendar())
    vector = comp.compute(
        view_at(rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW))
    )
    assert set(vector.quality_by_key) == set(comp.keys)
    assert vector.quality == DataQuality.GOOD


def test_a_substituted_anchor_downgrades_every_key_including_the_pivots():
    """`_vector` carries one quality for the whole vector, so a substituted
    session anchor downgrades the pivot features too. Conservative rather
    than wrong, and the notes name exactly what was substituted -- but it is
    worth pinning, because a reader could otherwise expect per-key grading."""
    comp = computer(levels=pivot_levels())  # no calendar
    vector = comp.compute(view_at(two_pivot_series()))
    assert vector.values["swing_high"] == BASE + 3.0  # the pivot is still right
    assert set(vector.quality_by_key.values()) == {DataQuality.DEGRADED}


def test_required_feeds_is_bars_only_and_nothing_is_optional():
    """Section 5 gives market structure as 'full function on bars alone', so
    a dishonest extra requirement would disable the component on a dataset it
    can in fact run on."""
    comp = computer()
    assert comp.name == "structure"
    assert comp.required_feeds == frozenset({Feed.BARS})
    assert comp.optional_feeds == frozenset()


def test_an_empty_view_is_not_ready_and_names_the_absent_feed():
    comp = computer()
    vector = comp.compute(empty_view())
    assert not vector.warmup_complete
    assert "bars feed absent" in vector.notes[0]


def test_a_not_ready_vector_is_zeros_and_missing_for_every_key():
    """The zeros are placeholders and must never be read, which is what
    `warmup_complete=False` plus MISSING is for."""
    comp = computer()
    vector = comp.compute(empty_view())
    assert set(vector.values) == set(comp.keys)
    assert set(vector.values.values()) == {0.0}
    assert vector.quality == DataQuality.MISSING


def test_the_constructor_takes_configuration_only():
    """Local echo of the package-wide ban in test_feature_contracts.py. A
    computer handed a series could capture a full-sample statistic at
    construction, which is the one leak the lookahead audit cannot see
    through when it is given an instance.

    `structure.py` uses postponed annotations, so `__init__.__annotations__`
    holds strings; they are resolved against the module globals before
    comparing, because comparing a `str` to a class silently passes."""
    parameters = inspect.signature(StructureFeatures.__init__).parameters
    assert list(parameters) == ["self", "config", "levels", "spec", "calendar"]
    assert parameters["calendar"].default is None

    hints = get_type_hints(StructureFeatures.__init__)
    assert hints["config"] is FeatureConfig
    assert hints["levels"] is StructureLevelConfig
    assert hints["spec"] is InstrumentSpec
    assert hints["calendar"] == (SessionCalendarProtocol | None)

    rendered = str(inspect.signature(StructureFeatures.__init__))
    for banned in (
        "SymbolData", "BarSeries", "ColumnSeries", "DataStore", "MarketView",
        "SyntheticDataset", "ndarray", "DataFrame", "Series",
    ):
        assert banned not in rendered, f"{banned} must not appear in the constructor"


def test_an_anchor_this_module_has_no_definition_for_is_refused():
    """`FeatureConfig` constrains `vwap_anchor` by pattern, so this branch is
    only reachable past the schema -- which is exactly when a silent default
    would be most damaging. `model_construct` skips validation to get there."""
    unvalidated = FeatureConfig.model_construct(vwap_anchor="minute")
    assert unvalidated.vwap_anchor == "minute"
    with pytest.raises(FeatureError, match="not one of"):
        StructureFeatures(unvalidated, session_levels(), nq_spec())


def test_the_declared_keys_do_not_depend_on_the_use_flags():
    """`use_opening_range` and friends select which anchors feed the zone
    clustering in `features/levels.py`. A feature that appeared and
    disappeared with a config flag would make the bundle's key-collision
    check depend on configuration."""
    off = session_levels(
        use_prior_session_levels=False,
        use_overnight_levels=False,
        use_opening_range=False,
        use_vwap_levels=False,
        use_round_numbers=False,
    )
    assert computer(levels=off).keys == computer().keys
    series = rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    assert computer(levels=off, calendar=TradingCalendar()).compute(
        view_at(series)
    ).values == pytest.approx(
        computer(calendar=TradingCalendar()).compute(view_at(series)).values
    )


# ---------------------------------------------------------------------------
# lookahead, determinism, and a long run
# ---------------------------------------------------------------------------


def test_the_lookahead_audit_passes_with_a_factory(synthetic_data):
    """`factory=` rather than an instance: that is the form that rebuilds the
    computer against each truncated and each future-mutated dataset, so a
    full-sample constant captured in `__init__` would change with it and be
    caught. Handing the audit an instance would not test that."""
    config = load_config()
    spec = config.spec(SYMBOL)
    comp = StructureFeatures(config.features, config.levels, spec)
    assert len(synthetic_data.primary_bars) >= 2 * comp.warmup_bars

    result = audit_computer(
        data=synthetic_data,
        factory=lambda d: StructureFeatures(config.features, config.levels, spec),
        sample=40,
    )
    assert_no_lookahead([result])
    assert result.bars_checked >= 20
    assert set(result.keys_checked) == set(comp.keys)


def test_the_lookahead_audit_also_passes_with_a_real_calendar(synthetic_data):
    """A second configuration, because the calendar path is where the session
    anchors are computed at all -- the no-calendar path reads none of the
    timestamp machinery the audit is most likely to catch."""
    config = load_config()
    spec = config.spec(SYMBOL)
    calendar = TradingCalendar()
    result = audit_computer(
        data=synthetic_data,
        factory=lambda d: StructureFeatures(
            config.features, config.levels, spec, calendar=calendar
        ),
        sample=40,
    )
    assert_no_lookahead([result])


def test_two_computations_on_the_identical_view_agree_exactly():
    """Bit-for-bit, notes included. A computer that read a clock or an
    unseeded generator would fail here before the audit ever ran."""
    comp = computer(calendar=TradingCalendar())
    view = view_at(
        rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    )
    first, second = comp.compute(view), comp.compute(view)
    assert first.values == second.values
    assert first.notes == second.notes
    assert first.quality_by_key == second.quality_by_key


def test_two_independently_constructed_computers_agree_bit_for_bit():
    """No state survives a call, so a fresh computer must give the same
    answer -- which is principle 5 stated at the level of one computer."""
    series = rth_series(THREE_DAYS, session_mids([100.0, 200.0, 300.0], SLOW), SLOW)
    calendar = TradingCalendar()
    a = computer(calendar=calendar).compute(view_at(series)).values
    b = computer(calendar=TradingCalendar()).compute(view_at(series)).values
    assert a == b


def test_every_value_is_finite_and_every_bounded_one_in_range_over_a_long_run(
    synthetic_data,
):
    """Three months of 5-minute bars through the real accessor path. The three
    bounded keys are the ones a component score may read, and one excursion
    past a bound would let a single bar dominate a weighted sum."""
    config = load_config()
    comp = StructureFeatures(
        config.features, config.levels, config.spec(SYMBOL), calendar=TradingCalendar()
    )
    bars = synthetic_data.primary_bars
    scored = 0
    for i in range(comp.warmup_bars - 1, len(bars), 17):
        ts = int(bars.ts_ns[i])
        vector = comp.compute(MarketView(synthetic_data, now=from_ns(ts), now_ns=ts))
        assert vector.warmup_complete
        for key, value in vector.values.items():
            assert math.isfinite(value), f"{key} is {value} at bar {i}"
        assert 0.0 <= vector.values["structure_score"] <= 1.0
        assert 0.0 <= vector.values["opening_range_position"] <= 1.0
        assert vector.values["structure_direction"] in (-1.0, 0.0, 1.0)
        assert vector.values["bos_direction"] in (-1.0, 0.0, 1.0)
        assert vector.values["choch"] in (0.0, 1.0)
        assert vector.values["bars_since_pivot"] >= 0.0
        assert vector.values["prior_session_low"] <= vector.values["prior_session_high"]
        assert (
            vector.values["opening_range_low"] <= vector.values["opening_range_high"]
        )
        scored += 1
    assert scored > 200


def test_a_change_of_character_never_fires_without_both_a_break_and_a_sequence(
    synthetic_data,
):
    """The invariant behind `choch`, checked against real-shaped data rather
    than only the three hand-built cases: it is strictly the conjunction, so
    a `choch` on a bar with no break or no established sequence would mean the
    condition had been loosened."""
    config = load_config()
    comp = StructureFeatures(
        config.features, config.levels, config.spec(SYMBOL), calendar=TradingCalendar()
    )
    bars = synthetic_data.primary_bars
    fired = 0
    for i in range(comp.warmup_bars - 1, len(bars), 13):
        ts = int(bars.ts_ns[i])
        values = comp.compute(
            MarketView(synthetic_data, now=from_ns(ts), now_ns=ts)
        ).values
        if values["choch"] == 1.0:
            assert values["bos_direction"] != 0.0
            assert values["structure_direction"] != 0.0
            assert values["bos_direction"] == -values["structure_direction"]
            fired += 1
        else:
            assert (
                values["bos_direction"] == 0.0
                or values["structure_direction"] == 0.0
                or values["bos_direction"] == values["structure_direction"]
            )
    assert fired > 0, "no change of character in three months: the test proves nothing"
