"""Liquidity features: quoted spread, depth, volume against the time-of-day
curve, the cost of taking liquidity, and the bounded score the LIQUIDITY
component reads.

ARCHITECTURE.md section 5 gives this component 15 of the 100 points, names
L1 quotes as the ideal feed and bar volume as the degraded one, and requires
that the degraded path be *flagged* rather than silently substituted. That
requirement is the organizing principle of this module, so it is stated
precisely before anything else.

## The degradation contract

`required_feeds` is `{BARS}`; `QUOTES` is optional. Half of what liquidity
means -- how wide the book is and which side of it is heavier -- is simply
not observable from bars. When the quote feed is absent this module does
three things and no others:

* **Reports a named fallback.** `spread_ticks` becomes
  `spec.typical_spread_ticks`, which is a *configured* number describing
  what the instrument's spread usually is. It is not a measurement of this
  bar and the module never implies that it is.
* **Says so in a note.** Every vector carries a note naming the fallback and
  the reason. A silent zero would read as "the spread is zero", which is the
  inverse of the truth, and `_vector` would not catch it because zero is
  finite.
* **Caps `liquidity_score` at `NO_QUOTE_SCORE_CAP`.** Two of the score's
  inputs (`spread_percentile`, and `depth_imbalance` by way of the book)
  are then fallbacks rather than observations, so the score may not claim
  the top of its range. The cap is binding, not decorative: see that
  constant.

The cap and the fallbacks apply whenever the spread is *not a measurement*,
which includes a present-but-crossed quote as well as an absent feed. A
crossed quote (`ask < bid`) is a known artifact of real feeds --
`QuoteSeries.crossed_count` exists precisely because they occur -- and a
negative spread is not a tighter market, it is no spread at all.

This module does not grade staleness. `data/quality.py` grades a feed
against its own cadence and the configured bound; a feature computer that
re-implemented that test would be a second, divergent source of truth.

## Why `relative_volume` exists, and why it is the feature that matters

Intraday volume has a pronounced U-shape: the first and last bars of a
session carry several times the volume of a midday bar. A plain percentile
rank of volume against the trailing window therefore ranks *every* opening
bar near 1.0 and *every* midday bar near 0.0, whatever the tape is actually
doing. A liquidity component fed that number is a clock with extra steps.

`relative_volume` divides the current bar's volume by the median volume of
the bars that share its **time-of-day bucket**, so an opening bar is
compared with other opening bars. `volume_percentile` is kept and emitted
anyway, because the two together are informative: when `volume_percentile`
is 1.0 and `relative_volume` is 1.0, the bar is busy *for the clock* and
ordinary *for the session* -- which is the single most common reading at the
open and the one a raw percentile cannot express.

### The bucket

A bar is attributed to the bucket containing its **open**, not its close.
Bar close timestamps run from `session_open + interval` to `session_close`,
so bucketing on the close puts the last bar of the session one bucket past
the end (clamped back onto its neighbour) and leaves the first bucket empty.
The open is also the semantically right choice: a bar's volume accumulates
over `[open, close)`.

Minutes-into-session come from the instrument's own `rth` window and
`timezone`, which are configuration (an `InstrumentSpec` is contract
economics, not market data, so it is allowed in `__init__`; a `BarSeries` is
not). The session span is `(rth.end - rth.start) mod 1440`, which handles a
24-hour window and a window that wraps midnight without a second code path.
Bars outside the RTH window are real -- overnight futures bars exist -- and
their volume is not comparable with a session bar's, so they are not forced
into bucket 0. They get a bucket of their own, index `time_of_day_buckets`,
and are compared with each other. An instrument that declares no `rth` at
all (the SPX cash index in `defaults.yaml`) has no session to bucket, so it
degrades to a single bucket -- a flat comparison -- and says so in a note.

### The profile window, and why `warmup_bars` is large

The profile needs enough observations *per bucket*, not in total. The
buckets partition the session into equal time spans, so a window of `N`
bars distributes roughly `N / buckets` bars into each one. Requiring
`volume_percentile_lookback` observations per bucket therefore sizes the
window at

    profile_bars = volume_percentile_lookback * time_of_day_buckets

which is 60 x 26 = 1560 bars on defaults: twenty 5-minute sessions. That
is the dominant term in `warmup_bars` and, through `FeatureBundle`, in the
whole bundle's warmup. The cost is stated rather than hidden, and the
alternative -- a profile built from two or three sessions -- is a median
over two or three correlated observations per bucket, which is not a profile.

`FeatureConfig` has no dedicated field for this window, so the per-bucket
sample target is `volume_percentile_lookback`, the same field that sizes the
volume percentile. That is a deliberate pun with one honest reading -- the
history you demand before ranking a volume is the history you demand before
claiming to know what is normal at this time of day -- and a weight sweep
moves both together. A dedicated `relative_volume_profile_bars` field would
be better; adding one means editing `config/schema.py`, which this module
does not own, so it is reported as a deviation instead.

The current bar is part of its own comparison set, as it is for
`volume_percentile` here and for `atr_percentile` in `features/volatility.py`.
With `volume_percentile_lookback` samples in the bucket its contribution to
the median is immaterial, and excluding it would make the current bar the
only bar in the window deflated by a different profile from every other.

## Why `liquidity_score` reads the spread LEVEL and not `spread_percentile`

`percentile_rank` in `features/base.py` is the fraction of the window *at or
below* the value. On a near-continuous series (an ATR) that is the right
definition. On a quoted spread it is not, because the distribution is
discrete and piles up on the minimum increment: an NQ spread is one tick on
the large majority of observations, so the *tightest possible* book ranks
near 1.0 -- "at or below" counts all the ties -- and a tightness term of
`1 - spread_percentile` would read a maximally tight market as maximally
wide. Measured over 462 sampled bars of an 18_642-bar synthetic NQ series
(`SyntheticConfig` defaults, seed 7, 300s bars, `spread_ticks_mean = 1.0`),
`spread_percentile` has minimum 0.833, median 0.933 and maximum 1.0 -- it
never once drops below 0.83, because the spread is almost never above the
floor and "at or below" counts every tie.

`spread_percentile` is still emitted, because ARCHITECTURE.md section 5
names it and because "is the spread wider than it has recently been" is a
real question. But the tie inflation is a property of the shared helper,
which this module does not own, so the score uses a transform that does not
depend on the shape of the spread distribution:

    tightness = clamp01(spec.typical_spread_ticks
                        / max(spread_ticks, spec.min_spread_ticks))

1.0 at or below the instrument's own typical spread, 0.5 at twice it, 0.25
at four times. The reference is the instrument rather than an absolute tick
count, so a two-tick GC spread reads as ordinary and a two-tick NQ spread
reads as twice normal -- which is the truth about those two contracts.

When the spread is a fallback the ratio would be exactly 1.0 by
construction (the fallback *is* `typical_spread_ticks`), reporting a
configured constant as a maximally tight book. That is the precise failure
this module exists to avoid, so tightness is pinned at
`TIGHTNESS_WITHOUT_QUOTES` instead and the score is capped.

`liquidity_score` is then `sqrt(tightness * adequacy)`, a geometric mean, so
either term failing drags it down and a zero on either sends it to zero.
Over the same 462 sampled bars the score has minimum 0.091, quartiles
0.487 / 0.680 / 0.856 and maximum 1.0, and `relative_volume` has median
1.009 (it is a ratio to a median, so a median near 1.0 is the check that the
deflation is doing what it claims) with a 0.26-3.50 range.

## Boundedness

`spread_percentile`, `volume_percentile`, `liquidity_score` are in [0, 1];
`depth_imbalance` and `volume_trend` are in [-1, 1]. Those are the five a
component score may read, and each is bounded *by construction* (a
percentile rank or a `tanh` squash) rather than by a clamp applied to an
unbounded quantity.

`spread_ticks`, `relative_volume`, `dollar_volume` and
`participation_cost_ticks` are in natural units (ticks, a ratio, currency,
ticks) and are diagnostics: they are reported, plotted and regressed on, and
a weighted sum must not read them, because one extreme observation in an
unbounded feature dominates the sum. `liquidity_score` reads the *bounded*
transforms of `relative_volume`, never the ratio itself.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from flow_model.config.schema import FeatureConfig
from flow_model.core.contracts import FeatureVector
from flow_model.core.enums import DataQuality, Feed
from flow_model.core.instruments import InstrumentSpec
from flow_model.data.market_view import MarketView
from flow_model.data.series import NS_PER_SECOND
from flow_model.features.base import (
    FeatureComputer,
    FeatureError,
    percentile_rank,
    safe_divide,
    squash,
)

#: Minutes in a day. Session arithmetic is modular in this.
MINUTES_PER_DAY = 1440.0

#: Seconds in a day, for grouping timestamps by UTC date.
SECONDS_PER_DAY = 86_400

#: Observations a time-of-day bucket needs before its median is used.
#:
#: A median over fewer than this is dominated by any one of its members, and
#: `relative_volume` would then measure that member rather than the session's
#: shape. A bucket below the floor falls back to the median of the whole
#: profile window -- a flat comparison, which is what the feature exists to
#: avoid -- so the fallback is reported in a note whenever it affects the
#: current bar. The profile window is sized to put
#: `volume_percentile_lookback` (default 60) observations in each in-session
#: bucket, so on a clean RTH feed this floor is never reached; it exists for
#: the out-of-session bucket, for a short first session, and for a feed whose
#: bars do not cover the whole window.
MIN_BUCKET_SAMPLES = 5

#: Ceiling on `liquidity_score` when the spread is a fallback, not a measurement.
#:
#: Binding by arithmetic, not by gesture. With no quote feed
#: `spread_percentile` is reported as 0.5, so the score's tightness term is
#: fixed at 0.5 and the geometric mean is `sqrt(0.5 * adequacy)`, which
#: reaches 0.707 at full volume adequacy. The cap therefore removes the top
#: 0.207 of the attainable range -- every reading that would have required
#: a tight *measured* spread to justify it. Half the component's evidence is
#: missing; half its range is the honest consequence.
NO_QUOTE_SCORE_CAP = 0.5

#: Floor on `relative_volume` inside the participation-cost formula.
#:
#: The slippage term is `min_spread_ticks / relative_volume`, which diverges
#: as participation goes to zero. A bar with a tenth of normal volume is
#: already an extreme; below that the formula is extrapolating past anything
#: it could have been calibrated on, so the term saturates at ten minimum
#: increments instead of running to infinity. Only the cost formula is
#: floored -- the emitted `relative_volume` is the measured ratio.
RELATIVE_VOLUME_FLOOR = 0.1

#: The score's tightness term when the spread is a fallback, not a measurement.
#:
#: Not derived from the fallback spread. The fallback IS
#: `spec.typical_spread_ticks`, so the tightness ratio would evaluate to
#: exactly 1.0 and report a configured constant as the tightest book the
#: instrument can show. 0.5 is the no-information value, and it is what makes
#: `NO_QUOTE_SCORE_CAP` arithmetically binding: the uncapped score is then
#: `sqrt(0.5 * adequacy)`, which reaches 0.707.
TIGHTNESS_WITHOUT_QUOTES = 0.5

#: Share of the requested spread observations that must be usable.
#:
#: `spread_percentile` asks for `spread_percentile_lookback` quotes and drops
#: the crossed ones. A percentile computed from what is left describes a
#: smaller sample than the one declared, and below this share it is a
#: different statistic altogether, so it is reported as unavailable (0.5)
#: with a note rather than passed off as the configured lookback.
MIN_SPREAD_SAMPLE_SHARE = 0.5

#: Change in relative volume, across the trend window, that maps to tanh(1).
#:
#: `volume_trend` is `squash(slope, VOLUME_TREND_REFERENCE_CHANGE / (k - 1))`
#: over a `k`-bar window, so a rise of one whole multiple of normal
#: participation from the start of the window to its end scores 0.76 and
#: twice that scores 0.96. Expressing the scale as a change *across the
#: window* rather than per bar makes it independent of the window length, so
#: sweeping `volume_percentile_lookback` does not silently rescale the
#: feature.
VOLUME_TREND_REFERENCE_CHANGE = 1.0


# ---------------------------------------------------------------------------
# session geometry
# ---------------------------------------------------------------------------


def utc_offset_minutes(seconds: np.ndarray, tz: ZoneInfo) -> np.ndarray:
    """UTC offset, in minutes, of each instant in `seconds` (epoch seconds).

    Grouped by UTC date, which is two timezone lookups per day rather than
    one per bar. The grouping is only valid while the offset is constant
    across a UTC day, so that is *checked* rather than assumed: a day whose
    first and last second disagree contains a DST transition and is resolved
    per instant. Assuming it instead would put a one-hour error into every
    bar of two sessions a year, and those two sessions would then be the only
    ones whose time-of-day buckets were wrong -- a defect that no aggregate
    would show.
    """
    out = np.empty(seconds.size, dtype=np.float64)
    if seconds.size == 0:
        return out
    days = seconds // SECONDS_PER_DAY
    for day in np.unique(days):
        mask = days == day
        start = int(day) * SECONDS_PER_DAY
        first = _offset_minutes(start, tz)
        last = _offset_minutes(start + SECONDS_PER_DAY - 1, tz)
        if first == last:
            out[mask] = first
        else:
            for index in np.flatnonzero(mask):
                out[index] = _offset_minutes(int(seconds[index]), tz)
    return out


def _offset_minutes(epoch_seconds: int, tz: ZoneInfo) -> float:
    local = datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).astimezone(tz)
    offset = local.utcoffset()
    if offset is None:  # pragma: no cover - ZoneInfo always supplies one
        raise FeatureError(f"timezone {tz!r} reported no UTC offset")
    return offset.total_seconds() / 60.0


def bar_open_minutes_of_day(
    close_ts_ns: np.ndarray, tz: ZoneInfo, interval_seconds: int
) -> np.ndarray:
    """Local minutes-of-day of each bar's OPEN, in [0, 1440).

    `close_ts_ns` holds bar CLOSE stamps (`BarSeries.ts_ns`), so the open is
    one interval earlier. See the module docstring on why the open and not
    the close.
    """
    if interval_seconds <= 0:
        raise FeatureError(
            f"bar interval {interval_seconds!r} is not positive, so a bar's open "
            "cannot be located and time-of-day bucketing is undefined"
        )
    open_seconds = close_ts_ns // NS_PER_SECOND - int(interval_seconds)
    offsets = utc_offset_minutes(open_seconds, tz)
    minutes = (open_seconds % SECONDS_PER_DAY) / 60.0 + offsets
    return np.mod(minutes, MINUTES_PER_DAY)


def session_span_minutes(spec: InstrumentSpec) -> float | None:
    """Length of the instrument's RTH session in minutes, or None if it has none.

    `(end - start) mod 1440`, so a window that wraps midnight and a
    24-hour window both work without a second branch. A window whose start
    equals its end is read as the whole day rather than as zero length,
    because a zero-length session has no bars and `SessionWindow` would
    otherwise be describing nothing.
    """
    window = spec.rth
    if window is None:
        return None
    span = (window.end_minutes - window.start_minutes) % int(MINUTES_PER_DAY)
    return float(span) if span else MINUTES_PER_DAY


def time_of_day_buckets(
    open_minutes: np.ndarray, spec: InstrumentSpec, buckets: int
) -> np.ndarray:
    """Bucket index per bar, in `[0, buckets]` inclusive.

    `0 .. buckets - 1` partition the RTH session into equal time spans.
    Index `buckets` is the out-of-session bucket: an overnight bar is
    compared with other overnight bars rather than being folded into the
    open. An instrument with no declared RTH window has every bar in bucket
    0, which is a flat comparison and is reported as such by the caller.
    """
    if buckets < 1:
        raise FeatureError(f"time_of_day_buckets must be at least 1, got {buckets}")
    span = session_span_minutes(spec)
    if span is None:
        return np.zeros(open_minutes.size, dtype=np.int64)
    assert spec.rth is not None  # implied by span not None
    offset = np.mod(open_minutes - float(spec.rth.start_minutes), MINUTES_PER_DAY)
    index = np.floor(offset / span * buckets).astype(np.int64)
    # `offset == span` is impossible after the modulo, but floating point can
    # still land `index` on `buckets` for an offset a hair under the span.
    index = np.minimum(index, buckets - 1)
    return np.where(offset < span, index, buckets)


def bucket_medians(
    values: np.ndarray, buckets: np.ndarray, bucket_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Median and sample count of `values` within each bucket.

    Returns `(medians, counts)`, both length `bucket_count`. An empty bucket
    gets a median of 0.0 and a count of 0; the caller decides what to do with
    it, because "no observations" and "a median of zero" are different facts
    and collapsing them is how a thin bucket turns into a divide-by-zero.
    """
    medians = np.zeros(bucket_count, dtype=np.float64)
    counts = np.bincount(buckets, minlength=bucket_count).astype(np.int64)
    if counts.size != bucket_count:
        raise FeatureError(
            f"bucket index out of range: {counts.size} buckets observed but "
            f"{bucket_count} declared"
        )
    order = np.argsort(buckets, kind="stable")
    ordered = values[order]
    edges = np.concatenate(([0], np.cumsum(counts)))
    for bucket in range(bucket_count):
        low, high = int(edges[bucket]), int(edges[bucket + 1])
        if high > low:
            medians[bucket] = float(np.median(ordered[low:high]))
    return medians, counts


def least_squares_slope(values: np.ndarray) -> float:
    """Slope of `values` against its own index, per step.

    `cov(x, y) / var(x)` with `x = 0 .. n-1`, which for an evenly spaced x is
    the ordinary least-squares slope. Returns 0.0 for fewer than two points,
    where a slope is not defined rather than zero -- but the callers in this
    module guarantee a full window, so the branch is a guard, not a result.
    """
    n = values.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(x_centered, values - values.mean()) / denominator)


def signed_squash(value: float, scale: float) -> float:
    """`squash` with the sign of the argument reapplied, giving [-1, 1].

    `base.squash` takes the absolute value by design, which is right for a
    magnitude and wrong for a direction: a component that read only the
    magnitude of a volume trend would treat collapsing participation as a
    reason to trade.
    """
    if not math.isfinite(value) or value == 0.0:
        return 0.0
    return math.copysign(squash(value, scale), value)


# ---------------------------------------------------------------------------
# the computer
# ---------------------------------------------------------------------------


class LiquidityFeatures(FeatureComputer):
    """Spread, depth, time-of-day-adjusted volume, and the LIQUIDITY score.

    Construction takes a `FeatureConfig` and an `InstrumentSpec` -- tick
    size, typical spread, point value, session window, timezone -- and
    nothing else. No series, no view, no array: capturing a full-sample
    statistic at construction is the one leak `validation.lookahead` cannot
    see through when handed an instance, so it is banned at the constructor
    and `tests/unit/test_feature_contracts.py` enforces the ban.
    """

    name = "liquidity"

    def __init__(self, config: FeatureConfig, spec: InstrumentSpec) -> None:
        self.config = config
        self.spec = spec
        self._volume_lookback = int(config.volume_percentile_lookback)
        self._spread_lookback = int(config.spread_percentile_lookback)
        self._bucket_count = int(config.time_of_day_buckets)
        # The profile needs `volume_percentile_lookback` observations in each
        # of `time_of_day_buckets` buckets; the buckets partition the session
        # evenly, so that is their product. See the module docstring.
        self._profile_bars = self._volume_lookback * self._bucket_count
        self._window_bars = max(
            self._volume_lookback, self._spread_lookback, self._profile_bars
        )
        # One bucket per session span, plus one for everything outside it.
        self._total_buckets = self._bucket_count + 1
        self._tz = ZoneInfo(spec.timezone)
        self._has_session = session_span_minutes(spec) is not None

    # --- declared contract ---------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """Bars needed before any output is meaningful.

        Dominated by the time-of-day profile. `spread_percentile_lookback` is
        included as a floor even though it counts quote observations rather
        than bars: quote cadence is independent of bar cadence, so the spread
        percentile additionally checks its own sample count at compute time
        and reports 0.5 with a note when it is short. The bar warmup is a
        floor on that check, never a substitute for it.
        """
        return self._window_bars

    @property
    def keys(self) -> tuple[str, ...]:
        return (
            "spread_ticks",
            "spread_percentile",
            "depth_imbalance",
            "volume_percentile",
            "relative_volume",
            "volume_trend",
            "dollar_volume",
            "participation_cost_ticks",
            "liquidity_score",
        )

    @property
    def required_feeds(self) -> frozenset[Feed]:
        return frozenset({Feed.BARS})

    @property
    def optional_feeds(self) -> frozenset[Feed]:
        return frozenset({Feed.QUOTES})

    # --- computation ---------------------------------------------------

    def compute(self, view: MarketView) -> FeatureVector:
        """Features at `view.now` from the last `warmup_bars` bars and the
        visible quote history.

        Warmup is re-checked here rather than left to `FeatureBundle`: the
        lookahead audit calls computers directly, including at pre-warmup
        bars, and a computer that reported `warmup_complete=True` with a
        half-filled window would be handing out a number built from fewer
        observations than it claims.
        """
        if not self.feeds_available(view):
            return self._not_ready(view, "bars feed absent")

        count = view.bar_count()
        if count < self._window_bars:
            return self._not_ready(
                view, f"warmup incomplete: {count} of {self._window_bars} bars"
            )

        interval = int(view.primary_interval)
        if interval <= 0:
            return self._not_ready(view, f"non-positive bar interval: {interval}s")

        n = self._window_bars
        volumes = view.column("volume", n)
        closes = view.column("close", n)
        stamps = view.bar_timestamps(n)
        if not (volumes.size == closes.size == stamps.size == n):
            return self._not_ready(
                view,
                f"short window: volume={volumes.size} close={closes.size} "
                f"ts={stamps.size}, expected {n} each",
            )
        if not np.all(np.isfinite(volumes)) or np.any(volumes < 0.0):
            # Every volume quantity below is a ratio or a median of these.
            # A nan survives `BarSeries` validation (only OHLC is checked for
            # finiteness) and would propagate into every volume feature.
            return self._not_ready(view, "volume window is non-finite or negative")
        if not np.all(np.isfinite(closes)):
            return self._not_ready(view, "close window is non-finite")

        notes: list[str] = []

        # --- quotes: spread, its percentile, and depth ------------------
        spread_ticks, spread_measured = self._spread_ticks(view, notes)
        spread_percentile, percentile_measured = self._spread_percentile(
            view, spread_ticks, spread_measured, notes
        )
        depth_imbalance, depth_measured = self._depth_imbalance(view, notes)

        # --- the time-of-day volume profile ----------------------------
        open_minutes = bar_open_minutes_of_day(stamps, self._tz, interval)
        buckets = time_of_day_buckets(open_minutes, self.spec, self._bucket_count)
        medians, counts = bucket_medians(volumes, buckets, self._total_buckets)
        window_median = float(np.median(volumes))

        if not self._has_session:
            notes.append(
                f"{self.name}: {self.spec.symbol} declares no RTH window, so there "
                "is no session to bucket; relative_volume compares against the "
                "whole trailing window (a flat comparison)"
            )

        # A bucket below the sample floor, or with no volume at all, cannot
        # supply a median; the window median is the stated fallback.
        usable = (counts >= MIN_BUCKET_SAMPLES) & (medians > 0.0)
        effective = np.where(usable, medians, window_median)
        current_bucket = int(buckets[-1])
        profile_degraded = not bool(usable[current_bucket])
        if profile_degraded:
            notes.append(
                f"{self.name}: time-of-day bucket {current_bucket} of "
                f"{self._total_buckets} holds {int(counts[current_bucket])} "
                f"observations (floor {MIN_BUCKET_SAMPLES}) with median "
                f"{medians[current_bucket]:.6g}; relative_volume falls back to the "
                f"window median {window_median:.6g}, which is a flat comparison"
            )
        if window_median <= 0.0 and profile_degraded:
            notes.append(
                f"{self.name}: no volume anywhere in the trailing {n}-bar window; "
                "relative_volume pinned to 1.0"
            )

        denominators = effective[buckets]
        safe = denominators > 0.0
        deflated = np.where(safe, volumes / np.where(safe, denominators, 1.0), 1.0)

        relative_volume = float(deflated[-1])
        tail = deflated[-self._volume_lookback :]
        volume_window = volumes[-self._volume_lookback :]

        # --- volume percentile, trend ----------------------------------
        if float(volume_window.max()) <= 0.0:
            # "Fraction at or below" would rank a zero volume at 1.0 among
            # other zeros, so a halted tape would read as maximum
            # participation -- the inverse of the truth. Zero is the floor.
            volume_percentile = 0.0
            notes.append(
                f"{self.name}: every bar in the {self._volume_lookback}-bar volume "
                "window is zero; volume_percentile pinned to 0.0"
            )
        else:
            volume_percentile = percentile_rank(volume_window, float(volume_window[-1]))

        trend_scale = VOLUME_TREND_REFERENCE_CHANGE / float(max(tail.size - 1, 1))
        volume_trend = signed_squash(least_squares_slope(tail), trend_scale)

        # --- natural-unit diagnostics ----------------------------------
        dollar_volume = (
            float(volume_window[-1]) * float(closes[-1]) * float(self.spec.point_value)
        )
        participation_cost = self._participation_cost(spread_ticks, relative_volume)

        # --- the component score ---------------------------------------
        # Geometric mean of two [0, 1] terms, which is an AND: either one
        # failing drags the score down, and a zero on either sends it to zero.
        # A bar at its instrument's typical spread and at its own median
        # time-of-day volume scores sqrt(1.0 * 0.5) = 0.71.
        #
        # `adequacy` ranks the time-of-day-DEFLATED volume, not the raw
        # volume: ranking the raw volume is what makes the component a clock,
        # and it is the mistake this whole module is built around avoiding.
        # The deflated series has median 1.0 by construction, so adequacy is
        # centred at 0.5 and an opening bar no longer starts from 1.0.
        tightness = self._tightness(spread_ticks, spread_measured)
        adequacy = percentile_rank(tail, relative_volume)
        score = math.sqrt(tightness * adequacy)
        if not spread_measured:
            score = min(score, NO_QUOTE_SCORE_CAP)
            notes.append(
                f"{self.name}: liquidity_score capped at {NO_QUOTE_SCORE_CAP:.2f} "
                "because spread_ticks and spread_percentile are fallbacks rather "
                "than measurements"
            )

        values = {
            "spread_ticks": spread_ticks,
            "spread_percentile": spread_percentile,
            "depth_imbalance": depth_imbalance,
            "volume_percentile": volume_percentile,
            "relative_volume": relative_volume,
            "volume_trend": volume_trend,
            "dollar_volume": dollar_volume,
            "participation_cost_ticks": participation_cost,
            "liquidity_score": _clamp01(score),
        }
        return self._vector(view, values, notes=tuple(notes)).replace(
            quality_by_key=self._quality_by_key(
                spread_measured=spread_measured,
                percentile_measured=percentile_measured,
                depth_measured=depth_measured,
                profile_degraded=profile_degraded,
            )
        )

    # --- quote-derived features ----------------------------------------

    def _spread_ticks(self, view: MarketView, notes: list[str]) -> tuple[float, bool]:
        """Current quoted spread in ticks, and whether it was measured.

        Falls back to `spec.typical_spread_ticks` -- configuration, not an
        observation -- when there is no quote feed, when the newest quote is
        not finite, or when it is crossed. A crossed quote is not a tighter
        market; it is a feed artifact, and `QuoteSeries.crossed_count` exists
        because real feeds produce them.
        """
        fallback = float(self.spec.typical_spread_ticks)
        quote = view.quote() if view.has_feed(Feed.QUOTES) else None
        if quote is None:
            notes.append(
                f"{self.name}: no quote feed; spread_ticks falls back to "
                f"spec.typical_spread_ticks={fallback:g} (configuration, not a "
                "measurement of this bar)"
            )
            return fallback, False

        if not (math.isfinite(quote.bid) and math.isfinite(quote.ask)):
            notes.append(
                f"{self.name}: newest quote has a non-finite side "
                f"(bid={quote.bid} ask={quote.ask}); spread_ticks falls back to "
                f"spec.typical_spread_ticks={fallback:g}"
            )
            return fallback, False

        raw = (float(quote.ask) - float(quote.bid)) / float(self.spec.tick_size)
        if raw < 0.0:
            notes.append(
                f"{self.name}: newest quote is crossed (bid={quote.bid:g} > "
                f"ask={quote.ask:g}), which is no spread rather than a tight one; "
                f"spread_ticks falls back to spec.typical_spread_ticks={fallback:g}"
            )
            return fallback, False
        if raw == 0.0:
            # A locked market is a real quoted state and zero is the honest
            # reading of it. The cost model separately refuses to charge less
            # than one minimum increment; see `_participation_cost`.
            notes.append(
                f"{self.name}: newest quote is locked (bid == ask == "
                f"{quote.bid:g}); spread_ticks is 0.0 as quoted"
            )
        return raw, True

    def _spread_percentile(
        self,
        view: MarketView,
        spread_ticks: float,
        spread_measured: bool,
        notes: list[str],
    ) -> tuple[float, bool]:
        """Rank of the current spread among the trailing quoted spreads.

        The lookback counts QUOTE OBSERVATIONS, not bars: the quote feed has
        its own cadence and `spread_percentile_lookback` bars of history is
        not `spread_percentile_lookback` quotes. Crossed observations are
        dropped from the comparison set, and if that leaves less than
        `MIN_SPREAD_SAMPLE_SHARE` of the requested sample the percentile is
        reported as unavailable (0.5) rather than computed from a sample the
        caller did not ask for.
        """
        if not spread_measured:
            notes.append(
                f"{self.name}: spread_percentile reported as 0.5 (unavailable) "
                "because the current spread is a fallback, not a measurement"
            )
            return 0.5, False

        window = view.spreads(self._spread_lookback) / float(self.spec.tick_size)
        valid = window[np.isfinite(window) & (window >= 0.0)]
        minimum = max(1, int(math.ceil(MIN_SPREAD_SAMPLE_SHARE * self._spread_lookback)))
        if valid.size < minimum:
            notes.append(
                f"{self.name}: only {valid.size} usable of "
                f"{self._spread_lookback} requested quote observations "
                f"(floor {minimum}); spread_percentile reported as 0.5 (unavailable)"
            )
            return 0.5, False
        if valid.size < self._spread_lookback:
            notes.append(
                f"{self.name}: spread_percentile computed from {valid.size} usable "
                f"of {self._spread_lookback} requested quote observations"
            )
        return percentile_rank(valid, spread_ticks), True

    def _depth_imbalance(self, view: MarketView, notes: list[str]) -> tuple[float, bool]:
        """`(bid_size - ask_size) / (bid_size + ask_size)`, in [-1, 1].

        0.0 with a note when there is no quote feed, when the sizes are not
        finite, or when both sides are empty. An empty book is not balanced;
        0.0 is a stated placeholder and the note says which of the two it is.
        """
        quote = view.quote() if view.has_feed(Feed.QUOTES) else None
        if quote is None:
            notes.append(
                f"{self.name}: no quote feed; depth_imbalance reported as 0.0 "
                "(unavailable, not balanced)"
            )
            return 0.0, False

        bid_size, ask_size = float(quote.bid_size), float(quote.ask_size)
        if not (math.isfinite(bid_size) and math.isfinite(ask_size)):
            notes.append(
                f"{self.name}: quote sizes are not finite (bid_size={bid_size} "
                f"ask_size={ask_size}); depth_imbalance reported as 0.0"
            )
            return 0.0, False
        total = bid_size + ask_size
        if total <= 0.0:
            notes.append(
                f"{self.name}: book is empty at the top (bid_size={bid_size:g} "
                f"ask_size={ask_size:g}); depth_imbalance reported as 0.0 "
                "(undefined, not balanced)"
            )
            return 0.0, False
        return _clamp(safe_divide(bid_size - ask_size, total, 0.0), -1.0, 1.0), True

    # --- score terms ---------------------------------------------------

    def _tightness(self, spread_ticks: float, spread_measured: bool) -> float:
        """How tight the book is, in [0, 1], relative to this instrument.

            tightness = clamp01(spec.typical_spread_ticks
                                / max(spread_ticks, spec.min_spread_ticks))

        1.0 at or below the instrument's typical spread, 0.5 at twice it,
        0.25 at four times. Deliberately NOT `1 - spread_percentile`: the
        spread distribution is discrete and piles up on the minimum
        increment, so a tie-inclusive percentile rank puts the tightest
        possible book near 1.0 and the tightness term near 0. See the module
        docstring.

        Pinned at `TIGHTNESS_WITHOUT_QUOTES` when the spread is a fallback,
        because the fallback is `typical_spread_ticks` itself and the ratio
        would therefore report a configured constant as a perfect book.
        """
        if not spread_measured:
            return TIGHTNESS_WITHOUT_QUOTES
        reference = float(self.spec.typical_spread_ticks)
        floor = float(self.spec.min_spread_ticks)
        return _clamp01(safe_divide(reference, max(float(spread_ticks), floor), 0.0))

    # --- cost ----------------------------------------------------------

    def _participation_cost(self, spread_ticks: float, relative_volume: float) -> float:
        """Estimated cost, in ticks, of taking liquidity on this bar.

            effective = max(spread_ticks, spec.min_spread_ticks)
            cost      = effective / 2
                      + spec.min_spread_ticks
                        / max(relative_volume, RELATIVE_VOLUME_FLOOR)

        The first term is the half-spread: crossing the book costs half of it
        relative to the mid. The second is slippage beyond the touch, scaled
        by the inverse of participation -- a thin bar absorbs less size at the
        quote. Its base unit is one *minimum increment* for the instrument
        (`spec.min_spread_ticks`), so at normal participation
        (`relative_volume == 1`) taking liquidity costs the half-spread plus
        one tick, and at a quarter of normal volume it costs the half-spread
        plus four. `FeatureConfig` carries no slippage field, and reaching
        into the execution model's cost config from a feature computer would
        give the system two sources of truth for slippage; deriving the base
        unit from the instrument instead keeps it to one and scales it
        correctly across NQ, GC and QQQ.

        `effective` floors the half-spread at the minimum increment because a
        locked or zero-width quote does not make taking liquidity free. The
        `RELATIVE_VOLUME_FLOOR` caps the slippage term at ten increments; see
        that constant.

        This is an estimate and it is in natural units, so no component score
        reads it. It feeds the execution-realism comparison in Phase 7: the
        backtester's own fill model is the authority on costs, and this
        feature exists to be checked against it.
        """
        minimum = float(self.spec.min_spread_ticks)
        effective = max(float(spread_ticks), minimum)
        participation = max(float(relative_volume), RELATIVE_VOLUME_FLOOR)
        return effective / 2.0 + minimum / participation

    # --- quality -------------------------------------------------------

    def _quality_by_key(
        self,
        *,
        spread_measured: bool,
        percentile_measured: bool,
        depth_measured: bool,
        profile_degraded: bool,
    ) -> dict[str, DataQuality]:
        """Per-key quality, so a caller can tell which features degraded.

        `FeatureVector` carries a status per key precisely so that "the
        spread is a fallback" does not have to be reported as "the volume
        features are unreliable". The aggregate `FeatureVector.quality` is
        the worst of these, so a consumer that gates on the vector as a whole
        still sees DEGRADED.
        """
        good, degraded = DataQuality.GOOD, DataQuality.DEGRADED
        quality = {
            "spread_ticks": good if spread_measured else degraded,
            "spread_percentile": good if percentile_measured else degraded,
            "depth_imbalance": good if depth_measured else degraded,
            # Reads bars only: neither the quote feed nor the profile.
            "volume_percentile": good,
            "dollar_volume": good,
            "relative_volume": degraded if profile_degraded else good,
            "volume_trend": degraded if profile_degraded else good,
            "participation_cost_ticks": (
                good if spread_measured and not profile_degraded else degraded
            ),
            "liquidity_score": (
                good if percentile_measured and not profile_degraded else degraded
            ),
        }
        missing = set(self.keys) - set(quality)
        if missing:  # pragma: no cover - guarded by test_every_key_has_a_quality
            raise FeatureError(f"{self.name}: no quality declared for {sorted(missing)}")
        return quality


def _clamp(value: float, low: float, high: float) -> float:
    return low if value <= low else (high if value >= high else float(value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)
