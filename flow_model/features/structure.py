"""Market structure: confirmed swing pivots and session geometry.

This module provides the two halves of ARCHITECTURE.md section 14.1's level
inventory that can be stated without clustering: the **swing pivots**, which
require detection and are where retail support/resistance code silently
acquires lookahead, and the **anchor levels**, which are objectively defined
and carry no detection risk at all. `features/levels.py` consumes both and
turns them into zones with significance/cleanliness/rejection scores; that
is deliberately not done here.

## The one rule this module exists to enforce

A swing high at bar `i` is CONFIRMED at evaluation time `t` only when all
three of these hold (14.1):

    i <= t - k                                    (right-side confirmation)
    high[i] == max(high[i-k : i+k+1])             (centred extreme)
    high[i] - max(neighbours) >= m * ATR[i]       (ATR-scaled prominence)

with `k = levels.pivot_confirm_bars` and `m = levels.pivot_prominence_atr`.

The first condition is the load-bearing one. A centred `argmax` evaluated at
`t` reads `k` bars *after* `t`, so a pivot "found" at `t` is a pivot the
market had not yet revealed. The failure is invisible from above: swing
levels sampled with future information sit exactly where price later turned,
every backtested rejection trade looks clean, and nothing downstream
complains. `find_pivots` therefore takes arrays that END at the evaluation
bar and emits only centres in `[k, len-1-k]`, which is that condition
written as a slice bound rather than as an assertion. Every pivot carries
`confirmed_index = index + k`, and a pivot does not exist in any window whose
last index is below it.

The third condition is why a bar with no ATR produces no pivot. When
`ATR[i] == 0` the prominence threshold is zero, and an exactly-flat tape
would then report *every* bar as both a swing high and a swing low. There is
no scale against which prominence can be measured on a halted tape, so such
candidates are skipped and said so in a note. With `ATR[i] > 0` and `m > 0`
the threshold is strictly positive, which makes the centred-extreme
condition strict automatically: a bar that merely ties the window maximum
has prominence 0 and is rejected.

## ATR is computed here, locally

`features/volatility.py` owns the system's reported ATR, and this module
deliberately does not import it: the two were written concurrently, and a
feature module that depends on a sibling's internals for a quantity it needs
on every bar couples their release schedules for no benefit. The formulas
are the standard ones (Wilder, `config.atr_period`) and are spelled out in
`true_range` and `wilder_atr_series`.

One consequence of a point-in-time Wilder recursion is worth restating:
Wilder's average is seeded by the simple mean of the first `period` true
ranges, and the only anchor available to a point-in-time computer is the
start of its own window. The whole ATR series is therefore rebuilt from the
window start on every call, so this module's ATR for a given bar is not
bit-identical to the value it reported for that bar one call earlier. That
is not lookahead -- every input is strictly in the past -- and it does not
affect the pivot set in any way that matters, because the prominence test
compares an ATR to a price distance rather than to another ATR.

## Session geometry, and what a missing calendar costs

Section 14.1's anchors are defined relative to a *session*, and a session is
whatever the calendar says it is. With a `TradingCalendar`:

* bars are grouped by `calendar.session_date`, which rolls at the ETH open
  for an instrument whose session wraps midnight -- so a 19:00 Monday bar
  and a 02:00 Tuesday bar belong to the same trade date, and the prior
  session's high is the high of that whole trade date rather than of a
  calendar day;
* the VWAP anchor is the first bar of the current group, per
  `config.vwap_anchor` (see `_vwap_group` for what `session`, `day` and
  `week` each mean);
* the opening range is bounded by `calendar.rth_bounds`, so it is the first
  `levels.opening_range_minutes` of the *regular* session and shortens
  correctly on a half day.

Without a calendar there are no sessions. The whole visible window is then
treated as one session: the VWAP runs over all of it, the opening range is
its first `opening_range_minutes`-worth of bars, and the prior-session
anchors are replaced by window statistics. **That is not the same quantity**
-- a window VWAP anchored 1153 bars ago is not a session VWAP -- so every
substitution is reported in a note and the vector's quality is downgraded to
DEGRADED. `_vector` carries one quality for every key, so a substituted
session anchor downgrades the pivot features too; that is conservative
rather than wrong, and the notes name exactly what was substituted.

`rth_bounds` returns a session's open and close from calendar arithmetic
alone. Reading the *boundary* of a session that has not finished is not
lookahead: no market data after the cutoff is consulted, and the
opening-range extremes are taken only over bars the view actually holds. A
view sitting inside the opening range reports the range **so far**, with a
note, because that is what was knowable.

## Boundedness

`structure_score` is the only feature here a component score may read, and
it is bounded [0, 1] by construction (a weighted sum of terms each in
[-1, 1], with weights summing to 1, then an absolute value).
`structure_direction`, `bos_direction` and `choch` are bounded by their own
definitions. Everything else -- the swing and anchor prices, `vwap`,
`bars_since_pivot` -- is in natural units and is a diagnostic: reported,
plotted and regressed on, but never summed into a score, because an
unbounded term lets one extreme observation dominate the sum.
`vwap_deviation_atr` is signed and scale-free but unbounded, which is why
`structure_score` passes it through `squash` rather than reading it raw.

## Window sizes

    pivot window   = atr_period + pivot_region
    pivot_region   = max(levels.lookback_bars, 2 * pivot_confirm_bars + 1)
    session window = max(pivot window, bars in SESSION_WINDOW_DAYS days)

`n` bars yield `n - 1` true ranges (the first bar has no previous close
inside the window and reaching outside it would make the window length a
lie), which yield `n - atr_period` Wilder values, aligned to the last
`n - atr_period` bars. Sizing the pivot window as `atr_period + region`
therefore gives exactly one ATR per bar of the pivot search region, which is
what the prominence test needs. The `2k + 1` floor guarantees the region is
wide enough to hold at least one centred window; without it a config with
`lookback_bars=11, pivot_confirm_bars=10` would declare a warmup, pass it,
and then find no pivot ever.

`warmup_bars` is the pivot window only. The session window is an upper bound
on how far back a session boundary can lie, not a quantity the features
depend on: the anchors are functions of the current and prior session alone,
so they stop changing as soon as both are visible. When the window does not
reach back far enough to certify that -- the window was cut short by the
view and the prior session starts at its first bar -- that is reported and
downgraded rather than assumed.
"""

from __future__ import annotations

import math
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import Field, model_validator

from flow_model.config.schema import FeatureConfig, StructureLevelConfig
from flow_model.core.contracts import FeatureVector
from flow_model.core.enums import DataQuality
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel
from flow_model.data.base import SessionCalendarProtocol
from flow_model.data.market_view import MarketView
from flow_model.data.series import NS_PER_SECOND, from_ns, to_ns
from flow_model.features.base import (
    FeatureComputer,
    FeatureError,
    safe_divide,
    squash,
)

__all__ = [
    "Pivot",
    "SESSION_WINDOW_DAYS",
    "STRUCTURE_SCORE_WEIGHTS",
    "StructureFeatures",
    "VWAP_DEVIATION_SCALE",
    "atr_aligned",
    "find_pivots",
    "sequence_direction",
    "structure_geometry_score",
    "true_range",
    "typical_price",
    "volume_weighted_average",
    "wilder_atr_series",
]

#: Seconds in a calendar day.
SECONDS_PER_DAY = 86_400.0

#: Calendar days of bars to read so the current and prior session boundaries
#: are both inside the window, per `config.vwap_anchor`.
#:
#: Four days rather than two for the session/day anchors because Monday's
#: *prior session* is Friday, three calendar days back, and a window sized
#: for two days would silently substitute Monday's own statistics for
#: Friday's. Eleven for the week anchor: a full ISO week plus the weekend on
#: either side. These are upper bounds on the distance to a boundary, not
#: lookbacks the features average over -- see the module docstring.
SESSION_WINDOW_DAYS: dict[str, int] = {"session": 4, "day": 4, "week": 11}

#: Weights of the three terms in `structure_score`, in order: the HH/HL
#: sequence direction, the break of structure, and the squashed signed VWAP
#: deviation. They sum to 1.0, which is what bounds the score.
#:
#: The sequence gets the most weight because it is the only term built from
#: *confirmed* pivots -- several bars of corroboration each -- while a break
#: is a single close and a VWAP deviation is a single bar's displacement. The
#: break gets more than the VWAP term because it is a discrete event at a
#: level the market has already respected, where the VWAP deviation is a
#: continuous distance that a trending session carries all day.
#:
#: These are module constants rather than config fields, and that is a
#: deviation worth stating: `FeatureComputer.__init__` takes configuration
#: only -- by contract, so that no computer can be handed market data -- and
#: neither `FeatureConfig` nor `StructureLevelConfig` carries weights for
#: this score. Phase 8's sweep over `StructureMagnitudeWeights` therefore
#: does not sweep these three. Adding them to the schema is a config-layer
#: change and this module does not own that file.
STRUCTURE_SCORE_WEIGHTS: tuple[float, float, float] = (0.40, 0.35, 0.25)

#: `squash` scale for the VWAP deviation term, in ATR units.
#:
#: One ATR of displacement from the session VWAP maps to tanh(1) = 0.76, two
#: ATR to 0.96. One ATR is the natural unit here because it is also the unit
#: the stop is quoted in (14.6, `stop_buffer_atr`): a close one ATR above
#: VWAP is roughly one stop-width of extension, which is the point at which
#: "price is above VWAP" stops being a tie-break and starts being a stated
#: displacement.
VWAP_DEVIATION_SCALE = 1.0


def _clip01(value: float) -> float:
    return 0.0 if value <= 0.0 else (1.0 if value >= 1.0 else float(value))


# ---------------------------------------------------------------------------
# ATR, locally
# ---------------------------------------------------------------------------


def true_range(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> np.ndarray:
    """`max(h-l, |h-prev_close|, |l-prev_close|)`, one value per bar after the first.

    Returns `len(highs) - 1` values. The first bar of the window has no
    previous close *inside* the window and reaching outside it would make the
    window length a lie, so that bar contributes no true range at all and
    serves only as `close_{i-1}` for the second. Seeding the first range with
    `high - low` instead biases it downward and then feeds that bias into
    Wilder's seed.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
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

        ATR_seed = mean(TR_0 .. TR_{p-1})
        ATR_i    = (ATR_{i-1} * (p - 1) + TR_i) / p

    `out[0]` is the seed; length is `len(ranges) - period + 1`, so a caller
    knows exactly how many values a window of true ranges yields. The loop is
    explicit rather than the vectorized closed form, which needs
    `(1 - 1/period)**-i` -- a growing factor multiplied back down -- and
    trades readable arithmetic for a numerical-stability question nobody
    wants to re-derive during an audit.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    period = int(period)
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


def atr_aligned(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> np.ndarray:
    """Wilder ATR aligned to the LAST `len(highs) - period` bars.

    `out[j]` is the ATR at bar `period + j` of the input arrays, so
    `out[-1]` is the ATR at the evaluation bar. The alignment is the whole
    point of this helper: the prominence test in `find_pivots` compares a
    price distance at bar `i` against the ATR at bar `i`, and an
    off-by-`period` here would scale every pivot's prominence by the ATR of
    a different bar.

    Raises when the window is too short to produce even one value, rather
    than returning an empty array a caller might then index.
    """
    period = int(period)
    needed = period + 1
    if np.asarray(highs).size < needed:
        raise FeatureError(
            f"atr_aligned over {period} periods needs at least {needed} bars, "
            f"got {np.asarray(highs).size}"
        )
    return wilder_atr_series(true_range(highs, lows, closes), period)


def typical_price(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> np.ndarray:
    """`(high + low + close) / 3`, the per-bar price proxy for VWAP.

    Bar data carries no trade-by-trade prices, so a volume-weighted average
    needs one price per bar. The typical price is used rather than the close
    because the close is where the bar happened to settle: on a wide bar that
    closed on its low, the close understates where the bar's volume actually
    traded, and a VWAP built from closes would then sit below the level the
    session's participants are watching.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    if not (highs.size == lows.size == closes.size):
        raise FeatureError(
            f"typical_price needs equal-length arrays, got high={highs.size} "
            f"low={lows.size} close={closes.size}"
        )
    return (highs + lows + closes) / 3.0


def volume_weighted_average(
    prices: np.ndarray, volumes: np.ndarray
) -> tuple[float, bool]:
    """`(vwap, volume_weighted)` over the given bars.

    Returns the simple mean with `volume_weighted=False` when the total
    volume is zero or non-finite: a zero-volume window has no volume
    distribution to weight by, and the unweighted mean is the honest
    fallback. The flag is returned rather than swallowed so the caller can
    say so in a note -- a VWAP that is quietly an arithmetic mean is a
    different statistic wearing the same name.
    """
    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    if prices.size == 0 or prices.size != volumes.size:
        raise FeatureError(
            f"volume_weighted_average needs equal-length non-empty arrays, got "
            f"price={prices.size} volume={volumes.size}"
        )
    total = float(volumes.sum())
    if not math.isfinite(total) or total <= 0.0:
        return float(prices.mean()), False
    weighted = float((prices * volumes).sum()) / total
    if not math.isfinite(weighted):
        return float(prices.mean()), False
    return weighted, True


# ---------------------------------------------------------------------------
# pivots
# ---------------------------------------------------------------------------


class Pivot(FrozenModel):
    """One confirmed swing extreme.

    `index` and `confirmed_index` are positions in the arrays that were
    handed to `find_pivots`, not absolute bar numbers: a point-in-time
    computer re-reads a rolling window, so the same market pivot has a
    different `index` one bar later. What is invariant is the relationship
    `confirmed_index = index + pivot_confirm_bars` and the fact that
    `find_pivots` never returns a pivot whose `confirmed_index` exceeds the
    last index of its input. The level does not exist before its
    confirmation bar, which is 14.1's `confirmed_at_ts` expressed in window
    coordinates.
    """

    index: int = Field(ge=0, description="Bar index of the extreme.")
    confirmed_index: int = Field(
        ge=0, description="index + pivot_confirm_bars: when the pivot became known."
    )
    price: float = Field(description="high at `index` for a swing high, low for a low.")
    is_high: bool
    prominence_atr: float = Field(
        ge=0.0,
        description="(price - best neighbour) / ATR[index]. The measured margin.",
    )

    @model_validator(mode="after")
    def _confirmation_is_in_the_future_of_the_extreme(self) -> "Pivot":
        if self.confirmed_index <= self.index:
            raise ValueError(
                f"pivot at {self.index} claims confirmation at {self.confirmed_index}: "
                "a swing is not known until at least one bar after it forms, so "
                "confirmed_index must exceed index"
            )
        if not math.isfinite(self.price):
            raise ValueError(f"pivot price is not finite: {self.price}")
        return self

    def bars_since(self, evaluation_index: int) -> int:
        """Bars between the extreme and `evaluation_index`, in window coordinates."""
        return int(evaluation_index) - self.index


def find_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    *,
    confirm_bars: int,
    prominence_atr: float,
) -> tuple[Pivot, ...]:
    """Confirmed pivots only, given arrays that end at the evaluation bar.

    All three arrays are indexed by the same bars, and `atr[i]` must be the
    ATR *at* bar `i` (see `atr_aligned`). The last index is the evaluation
    bar, so the right-side confirmation rule `i <= t - confirm_bars` becomes
    the upper bound of the centre range:

        centres i in [confirm_bars, len - 1 - confirm_bars]

    which is exactly the set of bars that have `confirm_bars` neighbours on
    both sides inside the arrays. A centred extreme nearer the end than that
    would need bars the caller does not have, and inventing one is how a
    swing level ends up sitting where price had not yet turned.

    A centre is a swing high when

        highs[i] - max(highs[i-k : i+k+1] without i) >= prominence_atr * atr[i]

    and a swing low under the mirrored condition on `lows`. The
    centred-extreme condition `highs[i] == max(window)` from 14.1 is implied
    rather than tested separately: with `atr[i] > 0` and `prominence_atr > 0`
    the threshold is strictly positive, so a bar that only ties the window
    maximum has prominence 0 and fails. Bars whose ATR is zero or non-finite
    are skipped entirely -- there is no scale on which to measure prominence,
    and without the skip every bar of a flat tape would qualify as both a
    high and a low.

    A single bar can be both: an outside bar that strictly dominates its
    neighbours in both directions is two pivots, and is returned as two.
    Results are sorted by `index`, highs before lows at the same index.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    if not (highs.size == lows.size == atr.size):
        raise FeatureError(
            f"find_pivots needs equal-length arrays, got high={highs.size} "
            f"low={lows.size} atr={atr.size}; a misaligned ATR would scale every "
            "prominence by the wrong bar's volatility"
        )
    k = int(confirm_bars)
    if k < 1:
        raise FeatureError(
            f"pivot_confirm_bars must be at least 1, got {k}: with no right-side "
            "confirmation a centred extreme at the last bar would be reported "
            "before the market had revealed it"
        )
    margin = float(prominence_atr)
    if not (math.isfinite(margin) and margin > 0.0):
        raise FeatureError(
            f"pivot_prominence_atr must be a positive finite number, got "
            f"{prominence_atr!r}: a non-positive threshold makes every bar that "
            "ties its window maximum a pivot"
        )

    total = highs.size
    width = 2 * k + 1
    if total < width:
        return ()

    high_windows = np.lib.stride_tricks.sliding_window_view(highs, width)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows, width)
    centres = np.arange(k, total - k)
    scale = atr[k : total - k]
    usable = np.isfinite(scale) & (scale > 0.0)
    threshold = margin * scale

    high_prominence = high_windows[:, k] - np.maximum(
        high_windows[:, :k].max(axis=1), high_windows[:, k + 1 :].max(axis=1)
    )
    low_prominence = (
        np.minimum(
            low_windows[:, :k].min(axis=1), low_windows[:, k + 1 :].min(axis=1)
        )
        - low_windows[:, k]
    )
    high_ok = usable & np.isfinite(high_prominence) & (high_prominence >= threshold)
    low_ok = usable & np.isfinite(low_prominence) & (low_prominence >= threshold)

    found: list[Pivot] = []
    for position in np.flatnonzero(high_ok):
        index = int(centres[position])
        found.append(
            Pivot(
                index=index,
                confirmed_index=index + k,
                price=float(highs[index]),
                is_high=True,
                prominence_atr=float(high_prominence[position] / scale[position]),
            )
        )
    for position in np.flatnonzero(low_ok):
        index = int(centres[position])
        found.append(
            Pivot(
                index=index,
                confirmed_index=index + k,
                price=float(lows[index]),
                is_high=False,
                prominence_atr=float(low_prominence[position] / scale[position]),
            )
        )
    found.sort(key=lambda pivot: (pivot.index, not pivot.is_high))
    return tuple(found)


def sequence_direction(
    highs: tuple[Pivot, ...], lows: tuple[Pivot, ...]
) -> float:
    """`structure_direction`: +1 for HH *and* HL, -1 for LH *and* LL, else 0.

    Read off the last two confirmed pivots of each kind:

        +1  when highs[-1] > highs[-2] and lows[-1] > lows[-2]
        -1  when highs[-1] < highs[-2] and lows[-1] < lows[-2]
         0  otherwise, including fewer than two pivots of either kind

    Both legs are required in each direction. A higher high with a lower low
    is a widening range, not an uptrend, and reporting it as +1 would make
    the feature agree with price expansion rather than with direction -- the
    two are different things and the range case is precisely where a
    structure trade should not fire.
    """
    if len(highs) < 2 or len(lows) < 2:
        return 0.0
    higher_high = highs[-1].price > highs[-2].price
    higher_low = lows[-1].price > lows[-2].price
    lower_high = highs[-1].price < highs[-2].price
    lower_low = lows[-1].price < lows[-2].price
    if higher_high and higher_low:
        return 1.0
    if lower_high and lower_low:
        return -1.0
    return 0.0


def structure_geometry_score(
    structure_direction: float, bos_direction: float, vwap_deviation_atr: float
) -> float:
    """`structure_score`: how much the three geometry readings agree, in [0, 1].

        v       = sign(dev) * squash(|dev|, VWAP_DEVIATION_SCALE)      in (-1, 1)
        aligned = 0.40 * structure_direction + 0.35 * bos_direction + 0.25 * v
        score   = |aligned|

    The weights sum to 1 and each term is in [-1, 1], so `aligned` is in
    [-1, 1] and the score is in [0, 1] by construction -- no clipping is
    doing load-bearing work, and no single term can dominate the sum the way
    a raw z-score would.

    The absolute value is deliberate and matters. This is the *magnitude*
    half of the STRUCTURE component; 14.6 keeps magnitude and direction
    separate so that a gate expresses "must have" and a score expresses "how
    good". A score of 0 therefore means the readings cancel -- an uptrend
    sequence broken to the downside while price sits at VWAP -- and a score
    near 1 means all three point the same way. The direction itself is read
    off `structure_direction` and `bos_direction`, which are emitted
    alongside.
    """
    weight_direction, weight_bos, weight_vwap = STRUCTURE_SCORE_WEIGHTS
    total = weight_direction + weight_bos + weight_vwap
    if abs(total - 1.0) > 1e-9:
        raise FeatureError(
            f"STRUCTURE_SCORE_WEIGHTS sum to {total}, expected 1.0; the score's "
            "[0, 1] bound comes from that sum and nothing else"
        )
    if math.isfinite(vwap_deviation_atr):
        deviation = math.copysign(
            squash(vwap_deviation_atr, VWAP_DEVIATION_SCALE), vwap_deviation_atr
        )
    else:
        deviation = 0.0
    aligned = (
        weight_direction * float(structure_direction)
        + weight_bos * float(bos_direction)
        + weight_vwap * deviation
    )
    return _clip01(abs(aligned))


# ---------------------------------------------------------------------------
# the computer
# ---------------------------------------------------------------------------


class StructureFeatures(FeatureComputer):
    """Confirmed swing geometry plus the session anchors of 14.1.

    Construction takes configuration only -- `FeatureConfig`,
    `StructureLevelConfig`, `InstrumentSpec` and an optional calendar -- and
    no series, view or array. That is the one leak `validation.lookahead`
    cannot see through when handed an instance, so it is banned at the
    constructor and the ban is tested by
    `tests/unit/test_feature_contracts.py`.

    `levels.use_opening_range`, `use_prior_session_levels` and the other
    `use_*` flags are NOT read here. They select which anchors feed the zone
    clustering in `features/levels.py`; this module measures the geometry
    unconditionally, because `keys` is a fixed declaration and a feature that
    appeared and disappeared with a config flag would make the bundle's
    key-collision check depend on configuration.
    """

    name = "structure"

    def __init__(
        self,
        config: FeatureConfig,
        levels: StructureLevelConfig,
        spec: InstrumentSpec,
        calendar: SessionCalendarProtocol | None = None,
    ) -> None:
        self.config = config
        self.levels = levels
        self.spec = spec
        self.calendar = calendar

        self._atr_period = int(config.atr_period)
        self._anchor = str(config.vwap_anchor)
        if self._anchor not in SESSION_WINDOW_DAYS:
            raise FeatureError(
                f"vwap_anchor={self._anchor!r} is not one of "
                f"{sorted(SESSION_WINDOW_DAYS)}; this module cannot anchor a VWAP "
                "to a period it has no definition for"
            )
        self._confirm_bars = int(levels.pivot_confirm_bars)
        self._prominence_atr = float(levels.pivot_prominence_atr)
        self._opening_range_minutes = float(levels.opening_range_minutes)

        # The pivot search region: one ATR per bar, and wide enough for at
        # least one centred confirmation window. See the module docstring.
        self._pivot_region = max(
            int(levels.lookback_bars), 2 * self._confirm_bars + 1
        )
        self._window_bars = self._atr_period + self._pivot_region
        self._timezone = ZoneInfo(spec.timezone)

    # --- declared contract ---------------------------------------------

    @property
    def warmup_bars(self) -> int:
        return self._window_bars

    @property
    def keys(self) -> tuple[str, ...]:
        return (
            "swing_high",
            "swing_low",
            "structure_direction",
            "bos_direction",
            "choch",
            "bars_since_pivot",
            "vwap",
            "vwap_deviation_atr",
            "opening_range_high",
            "opening_range_low",
            "opening_range_position",
            "prior_session_high",
            "prior_session_low",
            "prior_session_close",
            "structure_score",
        )

    # --- session grouping ----------------------------------------------

    def _session_group(self, ts_ns_value: int) -> date:
        """The calendar's trade date for one bar close.

        Rolls at the ETH open for an instrument whose session wraps
        midnight, so the prior session's high spans the overnight bars that
        belong to that trade date rather than to the previous calendar day.
        """
        assert self.calendar is not None
        return self.calendar.session_date(from_ns(int(ts_ns_value)), self.spec)

    def _vwap_group(self, ts_ns_value: int):
        """The VWAP anchor group for one bar close, per `config.vwap_anchor`.

        * `session` -- the calendar's trade date. Rolls at the ETH open.
        * `day` -- the exchange-local calendar date. Rolls at local midnight,
          so it deliberately differs from `session` for a contract whose
          session wraps midnight: an 18:00 bar starts a new trade date but
          not a new calendar day.
        * `week` -- the ISO (year, week) of the trade date, so the anchor is
          the first session of the ISO week.

        All three are non-decreasing in the timestamp, which is what lets
        `_first_index_of_group` find a boundary by bisection instead of
        converting every bar's timestamp. Local time is non-decreasing in UTC
        even across a DST fall-back (the repeated hour is the same local date
        and on the same side of the ETH open), so monotonicity holds.
        """
        if self._anchor == "session":
            return self._session_group(ts_ns_value)
        if self._anchor == "day":
            return from_ns(int(ts_ns_value)).astimezone(self._timezone).date()
        iso = self._session_group(ts_ns_value).isocalendar()
        return (iso[0], iso[1])

    @staticmethod
    def _first_index_of_group(stamps: np.ndarray, group, limit: int, grouper) -> int:
        """First index in `[0, limit)` whose group is not before `group`.

        Bisection over a non-decreasing key. `grouper` is called O(log n)
        times rather than n, which matters: each call converts a nanosecond
        stamp through `zoneinfo`, and doing that for every bar of a
        1153-bar window on every bar of a backtest costs more than the rest
        of this computer put together.
        """
        left, right = 0, int(limit)
        while left < right:
            middle = (left + right) // 2
            if grouper(int(stamps[middle])) < group:
                left = middle + 1
            else:
                right = middle
        return left

    def _session_window_bars(self, interval_seconds: int) -> int:
        """Bars to read so both session boundaries are inside the window."""
        days = SESSION_WINDOW_DAYS[self._anchor]
        spanning = int(math.ceil(days * SECONDS_PER_DAY / float(interval_seconds))) + 1
        return max(self._window_bars, spanning)

    def _opening_range_bars(self, interval_seconds: int) -> int:
        """Bars covering `opening_range_minutes`, at least one."""
        minutes_in_seconds = self._opening_range_minutes * 60.0
        return max(1, int(math.ceil(minutes_in_seconds / float(interval_seconds))))

    # --- computation ---------------------------------------------------

    def compute(self, view: MarketView) -> FeatureVector:
        """Features at `view.now`, from bars at or before its cutoff.

        `compute` re-checks warmup itself rather than relying on
        `FeatureBundle`: the lookahead audit calls computers directly,
        including at pre-warmup bars, and a computer reporting
        `warmup_complete=True` on a half-filled window would fail the
        harness's warmup-honesty check -- correctly, because anything that
        bypassed the bundle would then read a number built from fewer bars
        than it claims.
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

        requested = self._session_window_bars(interval)
        highs = view.column("high", requested)
        lows = view.column("low", requested)
        closes = view.column("close", requested)
        volumes = view.column("volume", requested)
        stamps = view.bar_timestamps(requested)
        size = closes.size
        if not (
            highs.size == lows.size == volumes.size == stamps.size == size
            and size >= self._window_bars
        ):
            return self._not_ready(
                view,
                f"short OHLCV window: high={highs.size} low={lows.size} "
                f"close={closes.size} volume={volumes.size} ts={stamps.size}, "
                f"expected at least {self._window_bars} of each",
            )
        for label, values in (("high", highs), ("low", lows), ("close", closes)):
            if not np.all(np.isfinite(values)):
                return self._not_ready(view, f"{label} contains a non-finite price")
        if not np.all(np.isfinite(volumes)) or not np.all(volumes >= 0.0):
            return self._not_ready(
                view, "volume contains a negative or non-finite value"
            )

        notes: list[str] = []
        # True when this module asked for more bars than the view holds, so a
        # boundary found at index 0 might be the window's edge rather than a
        # real session start. When False, index 0 is the start of all the
        # history there is and the boundary is as certain as anything can be.
        window_is_cut_short = count > size
        substituted = False

        # --- pivots --------------------------------------------------
        pivot_highs = highs[-self._window_bars :]
        pivot_lows = lows[-self._window_bars :]
        pivot_closes = closes[-self._window_bars :]
        atr_region = atr_aligned(
            pivot_highs, pivot_lows, pivot_closes, self._atr_period
        )
        region = int(atr_region.size)
        if region != self._pivot_region:
            raise FeatureError(
                f"{self.name}: a {self._window_bars}-bar window yielded {region} ATR "
                f"values but the pivot region is {self._pivot_region}; warmup_bars "
                "is wrong"
            )
        region_highs = pivot_highs[self._atr_period :]
        region_lows = pivot_lows[self._atr_period :]
        atr = float(atr_region[-1])
        if atr <= 0.0:
            notes.append(
                "ATR at the evaluation bar is zero: no pivot can be confirmed on a "
                "tape with no range, and vwap_deviation_atr has no scale"
            )

        pivots = find_pivots(
            region_highs,
            region_lows,
            atr_region,
            confirm_bars=self._confirm_bars,
            prominence_atr=self._prominence_atr,
        )
        confirmed_highs = tuple(p for p in pivots if p.is_high)
        confirmed_lows = tuple(p for p in pivots if not p.is_high)
        evaluation_index = region - 1

        if confirmed_highs:
            swing_high = confirmed_highs[-1].price
        else:
            swing_high = float(region_highs.max())
            notes.append(
                f"no confirmed swing high in the last {region} bars; swing_high "
                "falls back to the window high, which is not a pivot"
            )
        if confirmed_lows:
            swing_low = confirmed_lows[-1].price
        else:
            swing_low = float(region_lows.min())
            notes.append(
                f"no confirmed swing low in the last {region} bars; swing_low "
                "falls back to the window low, which is not a pivot"
            )

        if pivots:
            bars_since_pivot = float(
                evaluation_index - max(pivot.index for pivot in pivots)
            )
        else:
            # Strictly above every attainable value (the newest confirmable
            # pivot sits at region - 1 - confirm_bars), so the sentinel cannot
            # be mistaken for a measurement and is monotone in the right
            # direction for anything that thresholds it.
            bars_since_pivot = float(region)
            notes.append(
                f"no confirmed pivot in the last {region} bars; bars_since_pivot "
                f"reports the sentinel {region}, which exceeds every attainable value"
            )

        structure_direction = sequence_direction(confirmed_highs, confirmed_lows)

        # --- break of structure --------------------------------------
        last_close = float(closes[-1])
        broke_up = bool(confirmed_highs) and last_close > swing_high
        broke_down = bool(confirmed_lows) and last_close < swing_low
        if broke_up and broke_down:
            # Reachable when the most recent confirmed low sits ABOVE the most
            # recent confirmed high -- a strong one-way move that printed a
            # higher low before printing a new high. The more recently
            # confirmed pivot is the structure that was actually in force, so
            # its break decides; an exact tie (one outside bar that is both)
            # is reported as no break rather than as an arbitrary sign.
            newest_high = confirmed_highs[-1].index
            newest_low = confirmed_lows[-1].index
            if newest_high > newest_low:
                bos_direction = 1.0
            elif newest_low > newest_high:
                bos_direction = -1.0
            else:
                bos_direction = 0.0
            notes.append(
                f"close {last_close} is beyond both the swing high {swing_high} and "
                f"the swing low {swing_low}; resolved by pivot recency to "
                f"bos_direction={bos_direction}"
            )
        elif broke_up:
            bos_direction = 1.0
        elif broke_down:
            bos_direction = -1.0
        else:
            bos_direction = 0.0

        # Change of character, as a sequence condition: a break that runs
        # against the established HH/HL (or LH/LL) sequence. No break, or no
        # established sequence, is not a change of character.
        choch = (
            1.0
            if bos_direction != 0.0
            and structure_direction != 0.0
            and bos_direction == -structure_direction
            else 0.0
        )

        # --- session grouping ----------------------------------------
        typical = typical_price(highs, lows, closes)
        if self.calendar is None:
            vwap_anchor_start = 0
            session_start = 0
            notes.append(
                f"no calendar supplied: the VWAP, the opening range and the "
                f"prior-session anchors are computed over the whole visible window "
                f"of {size} bars instead of over a session. A window VWAP anchored "
                f"{size} bars ago is NOT a session VWAP"
            )
            substituted = True
        else:
            vwap_anchor_start = self._first_index_of_group(
                stamps, self._vwap_group(stamps[-1]), size, self._vwap_group
            )
            session_start = self._first_index_of_group(
                stamps, self._session_group(stamps[-1]), size, self._session_group
            )
            if vwap_anchor_start == 0 and window_is_cut_short:
                notes.append(
                    f"the {self._anchor} VWAP anchor is at the first visible bar of a "
                    f"window cut short at {size} of {count} bars, so the anchor may "
                    "predate the window and the VWAP may be truncated"
                )
                substituted = True

        # --- VWAP -----------------------------------------------------
        vwap, volume_weighted = volume_weighted_average(
            typical[vwap_anchor_start:], volumes[vwap_anchor_start:]
        )
        if not volume_weighted:
            notes.append(
                f"total volume over the {size - vwap_anchor_start} bars since the "
                "anchor is zero, so vwap is the unweighted mean typical price"
            )
        vwap_deviation_atr = safe_divide(last_close - vwap, atr, 0.0)

        # --- prior session -------------------------------------------
        if self.calendar is None or session_start == 0:
            prior_session_high = float(highs.max())
            prior_session_low = float(lows.min())
            prior_session_close = float(closes[0])
            if self.calendar is not None:
                notes.append(
                    f"no prior session is visible: all {size} bars belong to the "
                    "current session, so prior_session_high/low are the window "
                    "extremes and prior_session_close is the oldest visible close"
                )
                substituted = True
        else:
            prior_start = self._first_index_of_group(
                stamps,
                self._session_group(stamps[session_start - 1]),
                session_start,
                self._session_group,
            )
            prior_session_high = float(highs[prior_start:session_start].max())
            prior_session_low = float(lows[prior_start:session_start].min())
            prior_session_close = float(closes[session_start - 1])
            if prior_start == 0 and window_is_cut_short:
                notes.append(
                    f"the prior session starts at the first visible bar of a window "
                    f"cut short at {size} of {count} bars, so its high and low may "
                    "be truncated"
                )
                substituted = True

        # --- opening range -------------------------------------------
        or_start, or_stop = session_start, size
        if self.calendar is None:
            or_stop = min(size, self._opening_range_bars(interval))
            notes.append(
                f"no calendar: the opening range is the first "
                f"{self._opening_range_bars(interval)} bars of the visible window, "
                "not the first minutes of a regular session"
            )
        else:
            bounds = self._rth_bounds(stamps[-1])
            if bounds is None:
                or_stop = min(size, session_start + self._opening_range_bars(interval))
                notes.append(
                    f"the calendar reports no regular session for this trade date "
                    f"(holiday, or no RTH window declared for {self.spec.symbol}); "
                    f"the opening range is the first "
                    f"{self._opening_range_bars(interval)} bars of the session group"
                )
                substituted = True
            else:
                open_ns = to_ns(bounds[0])
                end_ns = open_ns + int(
                    round(self._opening_range_minutes * 60.0 * NS_PER_SECOND)
                )
                group = stamps[session_start:size]
                inside = np.flatnonzero((group > open_ns) & (group <= end_ns))
                if inside.size:
                    or_start = session_start + int(inside[0])
                    or_stop = session_start + int(inside[-1]) + 1
                    if or_stop == size and int(stamps[-1]) < end_ns:
                        notes.append(
                            f"the opening range is still forming: {inside.size} of "
                            f"its {self._opening_range_bars(interval)} bars have "
                            "closed, so the extremes are the range SO FAR"
                        )
                else:
                    notes.append(
                        "no bar of this session has closed inside the opening-range "
                        "window yet; opening_range_high/low are the session-to-date "
                        "extremes instead"
                    )
                    substituted = True

        opening_range_high = float(highs[or_start:or_stop].max())
        opening_range_low = float(lows[or_start:or_stop].min())
        span = opening_range_high - opening_range_low
        if span > 0.0:
            opening_range_position = _clip01((last_close - opening_range_low) / span)
        else:
            opening_range_position = 0.5
            notes.append(
                "the opening range has zero width, so opening_range_position is "
                "0.5 rather than a division by zero"
            )

        structure_score = structure_geometry_score(
            structure_direction, bos_direction, vwap_deviation_atr
        )

        return self._vector(
            view,
            {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "structure_direction": structure_direction,
                "bos_direction": bos_direction,
                "choch": choch,
                "bars_since_pivot": bars_since_pivot,
                "vwap": vwap,
                "vwap_deviation_atr": vwap_deviation_atr,
                "opening_range_high": opening_range_high,
                "opening_range_low": opening_range_low,
                "opening_range_position": opening_range_position,
                "prior_session_high": prior_session_high,
                "prior_session_low": prior_session_low,
                "prior_session_close": prior_session_close,
                "structure_score": structure_score,
            },
            quality=DataQuality.DEGRADED if substituted else DataQuality.GOOD,
            notes=tuple(notes),
        )

    def _rth_bounds(self, ts_ns_value: int):
        """The regular session's open/close for a bar's trade date, or None.

        `rth_bounds` is on `TradingCalendar` but not on
        `SessionCalendarProtocol`, so it is reached through `getattr`: a
        conforming calendar that lacks it gets the documented fallback
        (the first bars of the session group) with a note, rather than an
        `AttributeError` swallowed somewhere above.

        This reads a session boundary, never market data, so it is
        calendar arithmetic rather than a peek at the future: the
        opening-range extremes below are taken only over bars the view
        actually holds.
        """
        assert self.calendar is not None
        bounds_of = getattr(self.calendar, "rth_bounds", None)
        if bounds_of is None:
            return None
        return bounds_of(self._session_group(ts_ns_value), self.spec)
