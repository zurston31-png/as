"""Volatility features: ATR, the ATR percentile, three range estimators,
vol-of-vol, and the two bounded scores the VOL_MOMENTUM component reads.

`atr_percentile` is the most-read number in the system: the regime detector
grades HIGH_VOL/LOW_VOL on it (ARCHITECTURE.md section 6) and every setup's
volatility band gates on it (section 3). Everything else in this module
exists either to produce it honestly or to say something about volatility
that a single percentile cannot.

## The windows, and why `warmup_bars` is larger than the config's own

Every quantity here is a function of exactly the last `warmup_bars` bars --
never of "all the history there is". A fixed window makes the output a
stationary function of its input: a feature whose window silently grew with
the dataset would behave differently in bar 300 of a backtest than in bar
30_000, and no test would notice.

The window is sized by the longest *chain*, not the longest single lookback:

* `atr_percentile` needs `atr_percentile_lookback` (default 252) ATR values,
  and each ATR value needs `atr_period` (default 14) true ranges before it
  exists. A true range needs the previous bar's close. So the ATR chain
  needs `atr_period + atr_percentile_lookback` = 266 bars, which yields
  exactly 265 true ranges and exactly 252 ATR values.
* `vol_of_vol` needs `vol_of_vol_period` realized-vol values, and each needs
  `realized_vol_period` log returns: `realized_vol_period +
  vol_of_vol_period` = 40 bars.

`warmup_bars` is the max of the two, 266 on defaults. Note that
`FeatureConfig.warmup_bars` computes 253 (`atr_percentile_lookback + 1`),
which is 13 bars short of what a 252-deep ATR percentile actually requires;
it omits the `atr_period` the ATR chain consumes before its first value
exists. `FeatureBundle.warmup_bars` takes the max over its computers, so
declaring the true number here is sufficient and nothing downstream reads a
half-filled window -- but the config's own figure is optimistic and is
reported as a deviation rather than silently matched.

## True range and the first bar of the window

`TR_i = max(high_i - low_i, |high_i - close_{i-1}|, |low_i - close_{i-1}|)`.

The first bar in the window has no previous close *inside the window*, and
reaching outside it would make the window length a lie. So that bar
contributes **no true range at all**: it serves only as `close_{i-1}` for
the second bar. The window is sized with that bar included, so skipping it
still leaves exactly the `atr_period + atr_percentile_lookback - 1` true
ranges the ATR chain needs. The alternative convention -- seeding TR with
`high - low` on the first bar -- biases the first true range downward and
then feeds that bias into the Wilder seed, which is worse than dropping a
bar the window was sized to spare.

## Wilder's ATR, and where the recursion is anchored

Wilder's average is a recursion, `ATR_i = (ATR_{i-1}*(p-1) + TR_i) / p`,
seeded by the simple mean of the first `p` true ranges. A recursion needs an
anchor, and the only anchor available to a point-in-time computer is the
start of its own window. So the whole ATR series is rebuilt from the window
start on every call.

The consequence is worth stating plainly: the ATR value this computer
reports for a given bar is not bit-identical to the value it reported for
that same bar one call earlier, because the anchor has moved one bar. That
is not lookahead -- every input is strictly in the past -- and it is not a
problem for the feature that matters, because `atr_percentile` and the
`vol_expansion` median are computed over an ATR series that shares *one*
anchor. A percentile needs its comparison set to be internally consistent,
and it is. A caller that needs a single bar's ATR to be stable across calls
wants a stateful incremental estimator, which is a different object with a
different failure mode (state divergence) and is deliberately not this one.

## Annualization

`bars_per_year` is derived from the view's own bar interval, not configured,
so a 5-minute and a daily series annualize correctly without a second
source of truth. Intraday: 252 sessions x 23_400 regular-trading-hours
seconds / interval -- 19_656 bars/year at 300s, which is 78 bars/session,
matching the generator's own session grid. Daily or coarser: 252 x 86_400 /
interval, so a daily bar gives exactly 252. The two branches agree at the
boundary by construction, and neither reads a config field, because an
annualization factor that disagreed with the data's actual cadence would
rescale every volatility number in the system.

## Boundedness

`atr_percentile`, `vol_expansion` and `vol_regime_score` are bounded by
construction, and they are the three a component score may read.
`atr`, `atr_pct`, `realized_vol`, `parkinson_vol`, `garman_klass_vol` and
`vol_of_vol` are in natural units (price, fraction of price, annualized
fraction, coefficient of variation) and are diagnostics: they are reported,
plotted and regressed on, but a weighted sum must not read them, because a
single extreme observation in an unbounded feature dominates the sum.
"""

from __future__ import annotations

import math

import numpy as np

from flow_model.config.schema import FeatureConfig
from flow_model.core.contracts import FeatureVector
from flow_model.data.market_view import MarketView
from flow_model.features.base import (
    FeatureComputer,
    FeatureError,
    percentile_rank,
    safe_divide,
    squash,
)

#: Seconds in a calendar day. The intraday/daily branch point for annualization.
SECONDS_PER_DAY = 86_400.0

#: US equity-index trading sessions per year. The conventional 252.
TRADING_DAYS_PER_YEAR = 252.0

#: Regular-trading-hours seconds in one session (09:30-16:00 = 6.5 hours).
#: Matches `InstrumentSpec.rth` for the five instruments this system trades,
#: and therefore the synthetic generator's own session grid.
RTH_SECONDS_PER_SESSION = 6.5 * 3600.0

#: Parkinson's normalizing constant: `4 ln 2`.
FOUR_LN2 = 4.0 * math.log(2.0)

#: Garman-Klass's coefficient on the squared log body: `2 ln 2 - 1` ~ 0.3863.
GK_BODY_COEFFICIENT = 2.0 * math.log(2.0) - 1.0

#: `squash` scale for `vol_expansion`, in units of log(ATR / median ATR).
#:
#: `ln 2`, so one doubling of the ATR against its trailing median maps to
#: tanh(1) = 0.76 and one halving to -0.76. The tidy story is not the reason
#: it was kept; the measured distribution is. Over 2.5 years of 5-minute
#: synthetic NQ bars (SyntheticConfig defaults, seed 7, 48_611 scored bars)
#: |log(atr / median atr)| has median 0.268, p75 0.564, p90 0.827, p99 1.338
#: and max 1.782. At this scale that distribution lands at 0.37 / 0.65 /
#: 0.83 / 0.96 / 0.99: the bulk of the mass spreads across the middle of
#: [-1, 1] and only the top 1% of bars reach the flat part of the tanh.
#: Tighter scales were rejected on the same measurement -- 0.45 puts p90 at
#: 0.95 and 0.30 puts it at 0.99, which throws away the distinction between
#: a large expansion and an enormous one, and that distinction is most of
#: what the feature is for.
VOL_EXPANSION_SCALE = math.log(2.0)

#: Hard bound on the `log(atr / median atr)` handed to `squash`.
#:
#: `20 * VOL_EXPANSION_SCALE`, at which `tanh` is within 8e-18 of 1 -- below
#: double precision, so the clamp cannot change any value that the formula
#: could otherwise have produced. It exists for one reason: `squash` returns
#: **0.0** for a non-finite argument (a sensible default for a magnitude, and
#: the wrong answer here), so `log(0)` from a fully compressed ATR would be
#: reported as "no change in volatility" rather than as a total collapse.
#: Keeping the argument finite upstream is the fix; weakening `squash` is not.
LOG_RATIO_LIMIT = 20.0 * VOL_EXPANSION_SCALE

#: Plateau edges of the tradable volatility band, as ATR percentiles.
#:
#: These mirror `RegimeConfig.low_vol_percentile` (0.20) and
#: `RegimeConfig.high_vol_percentile` (0.80): the band is precisely the
#: percentile range the regime classifier does *not* label LOW_VOL or
#: HIGH_VOL. They are module constants rather than config fields because a
#: `FeatureComputer.__init__` takes `FeatureConfig` only -- by contract, so
#: that no computer can be handed market data -- and `FeatureConfig` carries
#: no band. Sweeping the regime thresholds therefore does not sweep these;
#: that coupling is reported as a deviation rather than papered over.
TRADABLE_BAND_LOW = 0.20
TRADABLE_BAND_HIGH = 0.80


def bars_per_year(interval_seconds: int) -> float:
    """Bars in a trading year at this bar interval. See the module docstring.

    Intraday intervals scale the 6.5-hour session; daily and coarser
    intervals scale the calendar day, so a daily bar returns exactly 252.
    """
    seconds = float(interval_seconds)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise FeatureError(
            f"bar interval {interval_seconds!r} is not a positive number of seconds; "
            "an annualization factor cannot be derived from it"
        )
    if seconds >= SECONDS_PER_DAY:
        return TRADING_DAYS_PER_YEAR * SECONDS_PER_DAY / seconds
    return TRADING_DAYS_PER_YEAR * RTH_SECONDS_PER_SESSION / seconds


def true_range(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> np.ndarray:
    """`max(h-l, |h-prev_close|, |l-prev_close|)`, one value per bar *after* the first.

    Returns `len(highs) - 1` values: the first bar has no previous close and
    contributes only its close. See the module docstring on why that bar is
    dropped rather than seeded with `high - low`.
    """
    if not (highs.size == lows.size == closes.size):
        raise FeatureError(
            f"true_range needs equal-length arrays, got high={highs.size} "
            f"low={lows.size} close={closes.size}"
        )
    if highs.size < 2:
        return np.zeros(0, dtype=np.float64)
    prev_close = closes[:-1]
    hi = highs[1:]
    lo = lows[1:]
    return np.maximum(
        hi - lo, np.maximum(np.abs(hi - prev_close), np.abs(lo - prev_close))
    )


def wilder_atr_series(ranges: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothed average of `ranges`, as a series.

    `out[0]` is the simple mean of the first `period` values (Wilder's seed);
    each later element applies `(prev*(period-1) + value) / period`. Length is
    `len(ranges) - period + 1`, so the caller knows exactly how many values a
    window of true ranges yields.

    The loop is explicit rather than a vectorized closed form. The closed
    form needs `(1 - 1/period)**-i`, which is a growing factor multiplied
    back down again, and trades readable arithmetic for a numerical
    stability question nobody wants to re-derive during an audit. Measured
    at the default 252-element output: ~51 microseconds per call, about a
    fifth of `VolatilityFeatures.compute`'s ~234 microseconds, the rest of
    which is NumPy call overhead on 266-element arrays. At ~4300 bars per
    second a full pass over the 48_876-bar synthetic dataset costs ~11
    seconds, which is not the bottleneck worth optimizing against.
    """
    if period < 1:
        raise FeatureError(f"Wilder period must be at least 1, got {period}")
    if ranges.size < period:
        raise FeatureError(
            f"Wilder ATR over {period} periods needs at least {period} ranges, "
            f"got {ranges.size}"
        )
    out = np.empty(ranges.size - period + 1, dtype=np.float64)
    level = float(np.mean(ranges[:period]))
    out[0] = level
    for i in range(period, ranges.size):
        level = (level * (period - 1) + float(ranges[i])) / period
        out[i - period + 1] = level
    return out


def signed_expansion(atr: float, median_atr: float) -> float:
    """`vol_expansion`: the signed, bounded log ratio of ATR to its trailing median.

        squash(log(atr / median_atr), VOL_EXPANSION_SCALE), with the sign of
        the log ratio reapplied.

    Positive means the ATR is above its own trailing median (expanding),
    negative means below it (compressing). The sign has to be reapplied by
    hand because `squash` takes `abs` by design: a component score that read
    only the magnitude would treat a collapse in range as a reason to trade.

    Three degenerate inputs, each with a stated answer rather than a nan:

    * `median_atr <= 0` -- the entire trailing window was flat, so there is
      no reference to expand from. Returns 0.0: neither expansion nor
      compression is the honest reading, and it is also what the ratio gives
      in the only way this is reachable through `compute` (an all-flat window
      makes `atr` zero too, so the ratio is 0/0).
    * `atr <= 0` against a positive median -- total compression. Returns the
      negative saturation rather than 0.0.
    * an extreme but finite ratio -- clamped at `LOG_RATIO_LIMIT`, which is
      invisible at double precision. See that constant.
    """
    if not math.isfinite(atr) or not math.isfinite(median_atr):
        return 0.0
    if median_atr <= 0.0:
        return 0.0
    if atr <= 0.0:
        return -squash(LOG_RATIO_LIMIT, VOL_EXPANSION_SCALE)
    log_ratio = max(-LOG_RATIO_LIMIT, min(LOG_RATIO_LIMIT, math.log(atr / median_atr)))
    return math.copysign(squash(log_ratio, VOL_EXPANSION_SCALE), log_ratio)


def _clamp01(value: float) -> float:
    return 0.0 if value <= 0.0 else (1.0 if value >= 1.0 else float(value))


def _smoothstep(value: float) -> float:
    """`3x^2 - 2x^3` on [0, 1]. Monotone, and flat at both ends."""
    x = _clamp01(value)
    return x * x * (3.0 - 2.0 * x)


def tradable_band_score(percentile: float) -> float:
    """`vol_regime_score`: 1 inside the tradable band, 0 at both extremes.

    Two one-sided ramps multiplied together, then smoothed:

        dead  = clamp(p / TRADABLE_BAND_LOW)              # 0 at p=0, 1 at p>=0.20
        panic = clamp((1 - p) / (1 - TRADABLE_BAND_HIGH)) # 1 at p<=0.80, 0 at p=1
        score = smoothstep(dead * panic)

    So the score is exactly 1 across [0.20, 0.80], exactly 0 at p=0 and
    p=1, and falls off smoothly in between (0.5 at p=0.10 and at p=0.90,
    0.16 at p=0.95). Both tails are penalized, which is the point: a dead
    tape has no range to pay for the stop, and a panic tape has range the
    stop distance cannot keep up with. A score that only punished high
    volatility would happily trade a tape that does not move.

    `smoothstep` rather than the raw product so the derivative vanishes at
    the plateau edges: a feature with a kink at exactly the regime threshold
    makes a weight sweep discontinuous at the one place it matters.
    """
    if not 0.0 < TRADABLE_BAND_LOW < TRADABLE_BAND_HIGH < 1.0:
        raise FeatureError(
            f"tradable band ({TRADABLE_BAND_LOW}, {TRADABLE_BAND_HIGH}) must be an "
            "ordered pair strictly inside (0, 1)"
        )
    p = _clamp01(percentile)
    dead = _clamp01(p / TRADABLE_BAND_LOW)
    panic = _clamp01((1.0 - p) / (1.0 - TRADABLE_BAND_HIGH))
    return _smoothstep(dead * panic)


class VolatilityFeatures(FeatureComputer):
    """ATR, the ATR percentile, three range estimators, and two bounded scores.

    Reads bars only, through `view.column(...)`. Construction takes
    `FeatureConfig` and nothing else -- no series, no view, no array -- which
    is the one leak `validation.lookahead` cannot see through and is
    therefore banned at the constructor.
    """

    name = "volatility"

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._atr_period = int(config.atr_period)
        self._atr_lookback = int(config.atr_percentile_lookback)
        self._rv_period = int(config.realized_vol_period)
        self._vov_period = int(config.vol_of_vol_period)

        # The two chains, spelled out. See the module docstring.
        self._atr_chain_bars = self._atr_period + self._atr_lookback
        self._vov_chain_bars = self._rv_period + self._vov_period
        self._window_bars = max(self._atr_chain_bars, self._vov_chain_bars)

    # --- declared contract ---------------------------------------------

    @property
    def warmup_bars(self) -> int:
        return self._window_bars

    @property
    def keys(self) -> tuple[str, ...]:
        return (
            "atr",
            "atr_pct",
            "atr_percentile",
            "realized_vol",
            "parkinson_vol",
            "garman_klass_vol",
            "vol_of_vol",
            "vol_expansion",
            "vol_regime_score",
        )

    # --- computation ---------------------------------------------------

    def compute(self, view: MarketView) -> FeatureVector:
        """Features at `view.now` from the last `warmup_bars` bars.

        `compute` re-checks warmup itself rather than relying on
        `FeatureBundle` to do it: the lookahead audit calls computers
        directly, including at pre-warmup bars, and a computer that reported
        `warmup_complete=True` with a half-filled window would fail check 3
        -- correctly, because something that bypasses the bundle would then
        read a number built from fewer bars than it claims.
        """
        if not self.feeds_available(view):
            return self._not_ready(view, "bars feed absent")

        count = view.bar_count()
        if count < self._window_bars:
            return self._not_ready(
                view, f"warmup incomplete: {count} of {self._window_bars} bars"
            )

        n = self._window_bars
        opens = view.column("open", n)
        highs = view.column("high", n)
        lows = view.column("low", n)
        closes = view.column("close", n)
        if not (opens.size == highs.size == lows.size == closes.size == n):
            return self._not_ready(
                view,
                f"short OHLC window: open={opens.size} high={highs.size} "
                f"low={lows.size} close={closes.size}, expected {n} each",
            )
        for label, prices in (
            ("open", opens), ("high", highs), ("low", lows), ("close", closes)
        ):
            if not np.all(np.isfinite(prices)) or not np.all(prices > 0.0):
                # Every estimator below is a log of a price ratio. A
                # non-positive price makes them undefined, and emitting a
                # fabricated zero would be read as "no volatility".
                return self._not_ready(
                    view, f"{label} contains a non-positive or non-finite price"
                )

        interval = int(view.primary_interval)
        if interval <= 0:
            return self._not_ready(view, f"non-positive bar interval: {interval}s")
        annualize = math.sqrt(bars_per_year(interval))

        notes: list[str] = []

        # --- ATR, its percentile, and its trailing median ---------------
        ranges = true_range(highs, lows, closes)
        atr_series = wilder_atr_series(ranges, self._atr_period)
        if atr_series.size < self._atr_lookback:
            raise FeatureError(
                f"{self.name}: window of {n} bars yielded {atr_series.size} ATR values "
                f"but the percentile needs {self._atr_lookback}; warmup_bars is wrong"
            )
        atr_window = atr_series[-self._atr_lookback :]
        atr = float(atr_series[-1])

        if float(atr_window.max()) <= 0.0:
            # A window with no range anywhere. "Fraction at or below" would
            # rank a zero ATR at 1.0 among other zeros, and the regime
            # detector would read a halted tape as HIGH_VOL -- the exact
            # inverse of the truth. Zero range is the floor of volatility.
            atr_percentile = 0.0
            notes.append(
                "every ATR in the lookback is zero; percentile pinned to 0.0"
            )
        else:
            atr_percentile = percentile_rank(atr_window, atr)

        median_atr = float(np.median(atr_window))
        vol_expansion = signed_expansion(atr, median_atr)

        # --- return-based and range-based volatility --------------------
        log_returns = np.diff(np.log(closes))
        rv_tail = log_returns[-(self._rv_period + self._vov_period - 1) :]
        rv_windows = np.lib.stride_tricks.sliding_window_view(
            rv_tail, self._rv_period
        )
        if rv_windows.shape[0] != self._vov_period:
            raise FeatureError(
                f"{self.name}: {rv_windows.shape[0]} realized-vol windows but "
                f"vol_of_vol_period is {self._vov_period}; warmup_bars is wrong"
            )
        # ddof=1: these are sample standard deviations of a sample of
        # returns, not of a population.
        rv_series = rv_windows.std(axis=1, ddof=1) * annualize
        realized_vol = float(rv_series[-1])
        # A coefficient of variation, so the annualization factor cancels and
        # the number is scale-free in both price and bar interval.
        vol_of_vol = safe_divide(
            float(rv_series.std(ddof=1)), float(rv_series.mean()), 0.0
        )

        r = self._rv_period
        log_hl = np.log(highs[-r:] / lows[-r:])
        # A flat bar gives ln(1) = 0, a well-defined zero contribution -- the
        # hazard in these estimators is a non-positive price, screened above,
        # not a zero range. An entirely flat window returns 0.0, which is the
        # correct estimate and not a division by zero.
        parkinson_vol = math.sqrt(float(np.mean(log_hl * log_hl)) / FOUR_LN2) * annualize

        log_body = np.log(closes[-r:] / opens[-r:])
        gk_terms = 0.5 * (log_hl * log_hl) - GK_BODY_COEFFICIENT * (log_body * log_body)
        gk_mean = float(np.mean(gk_terms))
        # The per-bar GK term cannot actually be negative while OHLC ordering
        # holds: high >= max(open, close) and low <= min(open, close) give
        # ln(h/l) >= |ln(c/o)|, and 0.5 > 2 ln 2 - 1 = 0.3863, so the term is
        # at least 0.1137 * ln(c/o)^2 >= 0. `BarSeries` validates that
        # ordering on construction. The clamp is the floor for the one way it
        # could still arrive negative -- an ordering-violating bar reaching
        # this computer past the schema check -- because a negative variance
        # estimate has no square root and `_vector` would raise on the nan.
        if gk_mean < 0.0:
            notes.append(
                f"Garman-Klass mean term is negative ({gk_mean:.3e}), which requires "
                "an OHLC ordering violation; clamped to zero"
            )
        garman_klass_vol = math.sqrt(max(gk_mean, 0.0)) * annualize

        return self._vector(
            view,
            {
                "atr": atr,
                "atr_pct": safe_divide(atr, float(closes[-1]), 0.0),
                "atr_percentile": atr_percentile,
                "realized_vol": realized_vol,
                "parkinson_vol": parkinson_vol,
                "garman_klass_vol": garman_klass_vol,
                "vol_of_vol": vol_of_vol,
                "vol_expansion": vol_expansion,
                "vol_regime_score": tradable_band_score(atr_percentile),
            },
            notes=tuple(notes),
        )
