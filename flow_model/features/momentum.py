"""Momentum features for the VOL_MOMENTUM component.

ARCHITECTURE.md section 5 gives this component "ROC, efficiency ratio,
momentum acceleration" and section 6 makes the Kaufman efficiency ratio the
quantity that separates TRENDING from CHOP. Everything here is a named
formula on OHLC bars alone, so the component is at full function with
bars-only data.

Three decisions worth stating, because each one is a place where a
plausible alternative would be wrong:

* **Every directional move is also reported in ATR units.** `roc` is a
  fractional return and is therefore comparable across instruments but not
  across volatility regimes: a 0.3% move is a shrug in HIGH_VOL and a
  stampede in LOW_VOL. `roc_atr` divides the same move by one bar's average
  true range, which makes it comparable across both. That is the input the
  component score reads; `roc` is emitted for interpretability and for the
  analytics breakdowns.

* **The ATR is computed here, locally, rather than imported.** `volatility.py`
  owns the published `atr` feature; this module needs an ATR only as a
  scaling denominator, and a cross-import between two feature modules would
  couple them for no benefit. The two are independent implementations of
  Wilder's average true range over `config.atr_period`, which is a mild
  duplication and a deliberate one: nothing in `momentum` can break when
  `volatility` changes its window conventions. No key emitted here collides
  with a volatility key, which `FeatureBundle` would reject anyway.

* **The ATR window is finite and declared.** Wilder's recursion is an
  infinite-memory filter, so an ATR seeded at the start of whatever history
  happens to be loaded would make a feature's value depend on how much data
  the caller loaded. That is not lookahead -- truncation only removes the
  future -- but it is non-reproducibility, and it would make a backtest's
  features differ from a live run's. The ATR here is seeded with the simple
  mean of the first `atr_period` true ranges inside a window of
  `2 * atr_period + 1` bars and smoothed forward over the remaining
  `atr_period`, so it is a pure function of a fixed, declared window.

Boundedness: `momentum_score` is the quantity that reaches a weighted sum,
and it is a convex combination of three terms each bounded in [0, 1]. No
z-score appears anywhere in this module -- one extreme observation cannot
dominate the component.

Degenerate inputs are handled explicitly rather than by luck. A perfectly
flat series has no summed absolute travel for `efficiency_ratio` to divide
by, no return variance for `momentum_persistence` to correlate, and no
high-low range for `close_position` to place a close within; each has a
stated fallback below, and `_vector` would raise rather than let a NaN
through if one were missed.
"""

from __future__ import annotations

import math

import numpy as np

from flow_model.config.schema import FeatureConfig
from flow_model.core.contracts import FeatureVector
from flow_model.data.market_view import MarketView
from flow_model.features.base import FeatureComputer, safe_divide, squash

# ---------------------------------------------------------------------------
# Parameters that FeatureConfig does not yet carry.
#
# These three belong in `FeatureConfig` next to `momentum_period` and
# `efficiency_period`; they are named module constants here only because
# `config/schema.py` is owned elsewhere this phase. They are reported as
# deviations so the gap is on the record rather than buried as a literal in
# an expression. Nothing else in this module hardcodes a number.
# ---------------------------------------------------------------------------

#: Half a bar's average true range. A net move smaller than this over the
#: whole `momentum_period` window is not a direction: it is a tape that has
#: gone nowhere while individual bars travelled several times as far. At
#: 0.25 ATR the band is deliberately narrow -- a 10-bar random walk covers
#: roughly 2 ATR, so this zeroes the flattest few percent of bars rather
#: than imposing a view about what counts as a real trend.
DIRECTION_DEAD_BAND_ATR: float = 0.25

#: Reference for the acceleration squash, in bar-ranges per bar. The change
#: in an n-bar ROC between consecutive bars is, for a random walk, about one
#: bar's worth of travel; 1.0 therefore maps a typical acceleration to
#: tanh(1) ~ 0.76 and leaves the extremes room without saturating.
ACCELERATION_REFERENCE_BAR_RANGES: float = 1.0

#: Reference for the `roc_atr` squash inside `momentum_score`, in ATR units.
#: A `momentum_period`-bar move of two average bar ranges maps to ~0.76.
MOMENTUM_SCORE_ROC_ATR_REFERENCE: float = 2.0

#: Convex weights combining the three terms of `momentum_score`. They sum to
#: 1.0, which is what bounds the score in [0, 1]; `_check_weights` enforces
#: that at import so a future edit cannot quietly unbound it. The split is a
#: declared prior in the sense of ARCHITECTURE.md principle 7, not a finding:
#: displacement leads, quality of travel qualifies it, and acceleration is a
#: tilt rather than a driver.
MOMENTUM_SCORE_WEIGHT_ROC_ATR: float = 0.45
MOMENTUM_SCORE_WEIGHT_EFFICIENCY: float = 0.35
MOMENTUM_SCORE_WEIGHT_ACCELERATION: float = 0.20


def _check_weights() -> None:
    total = (
        MOMENTUM_SCORE_WEIGHT_ROC_ATR
        + MOMENTUM_SCORE_WEIGHT_EFFICIENCY
        + MOMENTUM_SCORE_WEIGHT_ACCELERATION
    )
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"momentum_score weights sum to {total}, expected 1.0. The score the "
            "VOL_MOMENTUM component reads is bounded in [0, 1] because it is a "
            "convex combination; a sum above 1 unbounds it and a sum below 1 caps "
            "it below its documented range"
        )


_check_weights()


def _signed_squash(value: float, scale: float) -> float:
    """`squash` with the sign kept: tanh(value / scale), in [-1, 1].

    `base.squash` takes the absolute value because it exists for magnitudes.
    Acceleration has a meaningful sign -- "speeding up" versus "slowing
    down" -- so the sign is reattached rather than a second tanh written
    here.
    """
    magnitude = squash(value, scale)
    # `+ 0.0` collapses the negative zero `copysign` produces from a -0.0
    # input, so a zero acceleration compares equal to 0.0 and prints as 0.0.
    return math.copysign(magnitude, value) + 0.0


def _lag1_autocorrelation(values: np.ndarray) -> float:
    """Lag-1 autocorrelation of `values`, in [-1, 1]; 0.0 with no variance.

    The autocovariance estimator, with both sums taken over the full
    mean-removed window:

        rho = sum_i d_i * d_{i+1} / sum_i d_i^2,   d_i = x_i - mean(x)

    Cauchy-Schwarz bounds the numerator by the denominator, so this form
    cannot leave [-1, 1]. The alternative that normalizes by the two
    half-window standard deviations separately can exceed 1 on short
    windows, which would hand an out-of-range number to a bounded score.
    """
    if values.size < 3:
        # Two observations always give exactly -0.5 (their deviations are
        # equal and opposite by construction), which is a constant carrying
        # no information about the tape. Abstain rather than emit it.
        return 0.0
    deviations = values - float(np.mean(values))
    denominator = float(np.dot(deviations, deviations))
    if denominator <= 0.0:
        # A series with no variance has no innovations to correlate. 0.0 is
        # "no evidence either way", which is the honest answer; 1.0 would
        # report a dead tape as perfectly trending.
        return 0.0
    numerator = float(np.dot(deviations[:-1], deviations[1:]))
    return float(np.clip(numerator / denominator, -1.0, 1.0))


class MomentumFeatures(FeatureComputer):
    """Rate of change, travel efficiency, acceleration and persistence.

    Construction takes configuration only. It is given no `SymbolData`, no
    `BarSeries` and no array of prices, because a statistic captured at
    construction time from the full sample is the one lookahead leak the
    audit harness in `validation/lookahead.py` cannot see through when it is
    handed an already-built instance.
    """

    name = "momentum"

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.momentum_period = int(config.momentum_period)
        self.efficiency_period = int(config.efficiency_period)
        self.atr_period = int(config.atr_period)

        #: Bars read to produce the local Wilder ATR: `atr_period` true
        #: ranges to seed the average and `atr_period` to smooth it, plus one
        #: bar because the first true range needs a previous close.
        self.atr_window_bars = 2 * self.atr_period + 1

        #: Closes read for `roc` and for `acceleration`'s previous ROC: the
        #: n-bar move needs n+1 closes and the move one bar ago needs one
        #: more.
        self.roc_window_bars = self.momentum_period + 2

        #: Closes read for `efficiency_ratio` and for the return window of
        #: `momentum_persistence`. The two share a window deliberately: both
        #: are trend diagnostics, and describing the same stretch of tape
        #: makes them comparable to each other and to the regime detector,
        #: which reads `efficiency_ratio` over exactly this window.
        self.efficiency_window_bars = self.efficiency_period + 1

        self.direction_dead_band_atr = DIRECTION_DEAD_BAND_ATR
        self.acceleration_reference = ACCELERATION_REFERENCE_BAR_RANGES
        self.score_roc_atr_reference = MOMENTUM_SCORE_ROC_ATR_REFERENCE
        self.score_weights: dict[str, float] = {
            "roc_atr": MOMENTUM_SCORE_WEIGHT_ROC_ATR,
            "efficiency_ratio": MOMENTUM_SCORE_WEIGHT_EFFICIENCY,
            "acceleration": MOMENTUM_SCORE_WEIGHT_ACCELERATION,
        }

    # --- declared contract --------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """The longest window this computer reads.

        `roc_window_bars` already includes the extra bar `acceleration`
        needs for its previous ROC, so a caller cannot get an acceleration
        computed against a half-formed predecessor.
        """
        return max(
            self.roc_window_bars,
            self.efficiency_window_bars,
            self.atr_window_bars,
            self.momentum_period,
        )

    @property
    def keys(self) -> tuple[str, ...]:
        return (
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

    # --- local ATR -----------------------------------------------------

    def wilder_atr(self, view: MarketView) -> float:
        """Wilder ATR over the last `atr_window_bars` bars; 0.0 if flat.

            TR_i = max(high_i - low_i, |high_i - close_{i-1}|, |low_i - close_{i-1}|)
            ATR_seed = mean(TR_1 .. TR_p)
            ATR_i    = (ATR_{i-1} * (p - 1) + TR_i) / p      for i > p

        with `p = config.atr_period`. Public because the tests check the
        features that divide by it against this same quantity, and a private
        name would only have forced them to reach through it.
        """
        bars = self.atr_window_bars
        highs = view.highs(bars)
        lows = view.lows(bars)
        closes = view.closes(bars)
        if highs.size < bars or lows.size < bars or closes.size < bars:
            return 0.0

        previous_close = closes[:-1]
        true_range = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - previous_close), np.abs(lows[1:] - previous_close)
            ),
        )
        period = self.atr_period
        if true_range.size < period:
            return 0.0

        atr = float(np.mean(true_range[:period]))
        for value in true_range[period:]:
            atr = (atr * (period - 1) + float(value)) / period
        if not math.isfinite(atr) or atr <= 0.0:
            # A window in which every bar has zero range. Not an error: the
            # callers below divide through `safe_divide`, which returns a
            # defined fallback rather than an inf.
            return 0.0
        return atr

    # --- the features --------------------------------------------------

    def compute(self, view: MarketView) -> FeatureVector:
        if not self.feeds_available(view):
            return self._not_ready(view, "bar feed absent")
        if not view.warmup_ok(self.warmup_bars):
            return self._not_ready(
                view,
                f"warmup incomplete: {view.bar_count()} of {self.warmup_bars} bars",
            )

        n = self.momentum_period
        closes = view.closes(self.roc_window_bars)
        if closes.size < self.roc_window_bars:
            return self._not_ready(view, "close window short of its declared length")

        close_now = float(closes[-1])
        close_then = float(closes[-1 - n])
        close_previous = float(closes[-2])
        close_then_previous = float(closes[-2 - n])
        move = close_now - close_then

        atr = self.wilder_atr(view)

        # roc: the plain fractional n-bar return. safe_divide rather than a
        # bare quotient because a non-positive close is constructible in a
        # BarSeries (only volume is constrained non-negative) and an inf
        # here would be raised by `_vector` several lines later with no clue
        # as to which column caused it.
        roc = safe_divide(move, close_then, default=0.0)

        # roc_atr: the same move in bar-ranges. Scale-free, so it is
        # comparable across instruments and across volatility regimes.
        roc_atr = safe_divide(move, atr, default=0.0)

        efficiency_ratio = self._efficiency_ratio(view)
        direction = self._direction(move, atr)
        acceleration = self._acceleration(
            roc=roc,
            close_previous=close_previous,
            close_then_previous=close_then_previous,
            close_now=close_now,
            atr=atr,
        )
        momentum_persistence = self._persistence(view)
        up_bar_fraction = self._up_bar_fraction(view)
        close_position = self._close_position(view, close_now)
        momentum_score = self._momentum_score(
            roc_atr=roc_atr,
            efficiency_ratio=efficiency_ratio,
            acceleration=acceleration,
            direction=direction,
        )

        return self._vector(
            view,
            {
                "roc": roc,
                "roc_atr": roc_atr,
                "efficiency_ratio": efficiency_ratio,
                "direction": direction,
                "acceleration": acceleration,
                "momentum_persistence": momentum_persistence,
                "up_bar_fraction": up_bar_fraction,
                "close_position": close_position,
                "momentum_score": momentum_score,
            },
        )

    def _efficiency_ratio(self, view: MarketView) -> float:
        """Kaufman's efficiency ratio over `config.efficiency_period`, in [0, 1].

            |close_t - close_{t-k}| / sum_{i=t-k+1}^{t} |close_i - close_{i-1}|

        Net displacement divided by total travel: 1.0 when every step went
        the same way, near 0 when the steps cancelled. This is the quantity
        ARCHITECTURE.md section 6 uses to separate TRENDING from CHOP.

        A perfectly flat window makes the denominator zero. The fallback is
        0.0, not 1.0: a tape that has not moved has no direction to have been
        efficient about, and 1.0 would make the regime detector read a dead
        market as a perfect trend.
        """
        closes = view.closes(self.efficiency_window_bars)
        if closes.size < 2:
            return 0.0
        steps = np.diff(closes)
        travel = float(np.sum(np.abs(steps)))
        displacement = abs(float(closes[-1]) - float(closes[0]))
        return float(np.clip(safe_divide(displacement, travel, default=0.0), 0.0, 1.0))

    def _direction(self, move: float, atr: float) -> float:
        """Sign of the `momentum_period` move, in {-1, 0, +1}.

        Zero inside a dead band of `direction_dead_band_atr` average true
        ranges, so a tape whose bars travelled far while the net move went
        nowhere is not reported as directional. With no measurable range at
        all the band is undefined, and any nonzero move is then
        unambiguously large relative to a market that does not move, so the
        exact sign is used.
        """
        if not math.isfinite(move):
            return 0.0
        if atr <= 0.0:
            return 0.0 if move == 0.0 else math.copysign(1.0, move)
        if abs(move) < self.direction_dead_band_atr * atr:
            return 0.0
        return math.copysign(1.0, move)

    def _acceleration(
        self,
        *,
        roc: float,
        close_previous: float,
        close_then_previous: float,
        close_now: float,
        atr: float,
    ) -> float:
        """Change in ROC between this bar and the last, squashed into [-1, 1].

            roc_previous = close_{t-1} / close_{t-1-n} - 1
            d            = roc - roc_previous
            normalized   = d / (atr / close_t)
            acceleration = tanh(normalized / reference)

        Positive when the move is speeding up. The division by `atr /
        close_t` converts a fractional return into bar-ranges, which is what
        makes the squash reference meaningful on NQ and on QQQ at once: a
        fixed scale applied to a raw fractional ROC difference would saturate
        on one instrument and never move on another.
        """
        roc_previous = safe_divide(
            close_previous - close_then_previous, close_then_previous, default=0.0
        )
        change = roc - roc_previous
        atr_fraction = safe_divide(atr, close_now, default=0.0)
        normalized = safe_divide(change, atr_fraction, default=0.0)
        return _signed_squash(normalized, self.acceleration_reference)

    def _persistence(self, view: MarketView) -> float:
        """Lag-1 autocorrelation of the trailing returns, in [-1, 1].

        The window is `config.efficiency_period` returns -- the same stretch
        `efficiency_ratio` measures travel over. Positive means a push
        tended to be followed by a push in the same direction (trending);
        negative means it tended to be given back (mean-reverting).

        A series whose returns are all identical -- a flat tape, or a
        geometric path whose ratio is exact in binary floating point -- has
        no return variance and therefore no innovations to correlate, and
        reports 0.0. That is a deliberate abstention rather than a claim of
        perfect persistence. The guard is exact-zero variance and not a
        tolerance: a path that is only *nominally* geometric carries rounding
        noise in the last bits of its returns, and this feature then reports
        the autocorrelation of that noise. The result is still bounded in
        [-1, 1] so it cannot distort a score, and no real tape has returns
        constant to within a few ULPs, so a relative epsilon here would be a
        threshold invented to handle a case that does not occur.
        """
        closes = view.closes(self.efficiency_window_bars)
        if closes.size < 3:
            return 0.0
        previous = closes[:-1]
        returns = np.divide(
            np.diff(closes),
            previous,
            out=np.zeros(closes.size - 1, dtype=np.float64),
            where=previous != 0.0,
        )
        returns[~np.isfinite(returns)] = 0.0
        return _lag1_autocorrelation(returns)

    def _up_bar_fraction(self, view: MarketView) -> float:
        """Fraction of the trailing `momentum_period` bars that closed above
        their open, in [0, 1].

        A doji -- close exactly equal to open -- counts as neither up nor
        down, so a perfectly flat window reports 0.0 rather than 1.0.
        """
        n = self.momentum_period
        opens = view.opens(n)
        closes = view.closes(n)
        if opens.size == 0 or opens.size != closes.size:
            return 0.0
        return float(np.count_nonzero(closes > opens)) / float(opens.size)

    def _close_position(self, view: MarketView, close_now: float) -> float:
        """Where the last close sits in the trailing high-low range, in [0, 1].

            (close_t - min(low)) / (max(high) - min(low))

        over the trailing `momentum_period` bars. 1.0 is a close at the
        window high. A window with no range at all places the close at the
        high and the low simultaneously, so the fallback is 0.5 -- the
        midpoint -- rather than either extreme.
        """
        n = self.momentum_period
        highs = view.highs(n)
        lows = view.lows(n)
        if highs.size == 0 or lows.size == 0:
            return 0.5
        window_high = float(np.max(highs))
        window_low = float(np.min(lows))
        span = window_high - window_low
        if span <= 0.0:
            return 0.5
        return float(np.clip((close_now - window_low) / span, 0.0, 1.0))

    def _momentum_score(
        self,
        *,
        roc_atr: float,
        efficiency_ratio: float,
        acceleration: float,
        direction: float,
    ) -> float:
        """Bounded [0, 1] magnitude for the VOL_MOMENTUM component.

            displacement = squash(|roc_atr|, score_roc_atr_reference)   in [0, 1]
            quality      = efficiency_ratio                             in [0, 1]
            support      = max(0, acceleration * direction)             in [0, 1]
            score        = w_d * displacement + w_q * quality + w_s * support

        A magnitude, not a direction: `direction` is the component's sign and
        is carried separately, so `displacement` uses the absolute move.

        Acceleration enters multiplied by `direction`, which turns it from
        "speeding up" into "speeding up *the way the move is already
        going*". Inside the direction dead band `direction` is 0, so the term
        is 0: there is no move for an acceleration to be reinforcing.

        `max(0, ...)` rather than a rescaling of [-1, 1] into [0, 1]. Every
        term here means "this much evidence of momentum", so 0 has to mean
        "none" -- a term whose neutral value were 0.5 would give the score a
        floor of `w_s / 2`, which on the default weights is 0.10 and would
        put two of the VOL_MOMENTUM component's twenty points on a tape that
        has not moved. A move that is fading instead of building therefore
        gets no acceleration credit rather than negative credit; the signed
        `acceleration` feature is emitted separately for the gates and the
        breakdowns to read, and is not discarded.

        The weights sum to 1 and each term is in [0, 1], so the convex
        combination is bounded by construction rather than by a clip. The
        clip is kept for float safety only.
        """
        displacement = squash(roc_atr, self.score_roc_atr_reference)
        quality = float(np.clip(efficiency_ratio, 0.0, 1.0))
        support = max(0.0, acceleration * direction)
        score = (
            self.score_weights["roc_atr"] * displacement
            + self.score_weights["efficiency_ratio"] * quality
            + self.score_weights["acceleration"] * support
        )
        return float(np.clip(score, 0.0, 1.0))
