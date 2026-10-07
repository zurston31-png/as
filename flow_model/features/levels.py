"""Support/resistance zones: the measurement that replaces a judgement.

ARCHITECTURE.md section 14 exists because the researcher's instruction was
"use clean price action at major support and resistance levels", and per
principle 3 that sentence cannot enter the codebase as written. "Clean" and
"major" are judgements. This module is the translation: three bounded
scalars, four gates built on them, and a stop and a target that are
*determined* by the structure rather than chosen.

    S  significance   (14.3)  -- "major" is the gate `S >= min_significance`
    C  cleanliness    (14.4)  -- "clean" is the gate `C >= min_cleanliness`
    R  rejection      (14.5)  -- the trigger, gated by a binary requirement

and then 14.6:

    magnitude     = w_S*S + w_C*C + w_R*R
    direction     = +1 at support, -1 at resistance
    stop          = min(zone_low, bar.low) - stop_buffer_atr * ATR   (long)
    target        = the next opposing zone with S >= min_significance
    achievable_rr = |target - entry| / |entry - stop|

Honest framing, carried over from section 14 itself: formalizing
support/resistance makes it **testable**, not profitable. This module adds
no performance claim, and section 14.7's five falsification conditions are
the things that would retire it. Two of them are directly readable from the
keys emitted here (`level_significance` and `level_cleanliness` bucketed
against outcome), and one needs `level_touch_count` against
`level_recent_touches`, which is why both are emitted as natural-unit
diagnostics rather than only as weighted terms.

## What is reused rather than rebuilt

`features/structure.py` already owns confirmed pivot detection and the ATR
alignment it needs, and this module imports `find_pivots`, `atr_aligned`,
`typical_price` and `volume_weighted_average` from it rather than restating
them. In particular `find_pivots` enforces 14.1's right-side confirmation
rule (`i <= t - k_confirm`) as a slice bound, so a pivot cannot exist in a
window whose last index is before its confirmation bar. That is the single
rule this whole construction rests on, and it is enforced in exactly one
place.

What is NOT reused is `StructureFeatures.compute`. Three of this module's
anchors need quantities that computer does not expose -- the VWAP anchor
index (for the +/-1 sigma bands), the RTH open boundary (for the overnight
high and low), and the bar that *formed* each session extreme (for the age
term and for excluding a level's own formation bar from its touch count) --
so the session grouping is done once here instead of being computed twice
and then half-reconstructed. The cost is a second implementation of the
same bisection over `calendar.session_date`, and the risk is that the two
drift apart; `tests/unit/test_levels.py` pins this module's prior-session
anchors against `StructureFeatures`' own on the same bars so a drift fails
the build rather than producing two different "prior session highs".

## No calendar means no sessions, and no session anchors

`StructureFeatures` substitutes window statistics for the session anchors
when no calendar is supplied, because its `keys` are a fixed declaration and
it must emit a number for `prior_session_high` either way. This module has
no such obligation: it emits a *set* of levels, and the honest size of that
set when there is no calendar is "the ones that do not need a session". So
without a calendar the prior-session, overnight, opening-range and VWAP
anchors are **absent**, not substituted, the vector is DEGRADED, and a note
says so. Inventing a "prior session high" out of a window extreme would put
a fabricated anchor into `s_anchor` and raise S for a level nobody is
watching.

## The deliberate tension between S and C (14.4)

Historical touches raise S. Recent touches lower C. That is not an
inconsistency: `tau = 6` spread over 400 bars with `rho = 0` in the last 30
is a well-established level, and the same `tau = 6` packed into the last 30
bars is a level under active attack. The second is declined. Keeping these
as two scores rather than one is what makes that distinction expressible,
and 14.7's third falsification test is precisely "if `s_density` and
`s_touch` have the same sign of association with outcome, this split is
unjustified".

## Degradation is subtraction, never redistribution

`s_flow` is dropped when `MarketView.deltas()` is empty, **and its weight is
not redistributed** (14.5, and `strict_component_availability` in general).
Arithmetically, dropping a term of weight `w` is identical to scoring that
term 0 and keeping `w`, which is how it is implemented; what matters is what
is NOT done -- renormalizing the remaining weights to sum to 1 would
manufacture confidence from data that does not exist. With no tick feed, R
is therefore capped at `close_position + displacement` = 0.75 of its
default weight, and that cap is a reported fact rather than a hidden one.

The same subtraction applies to `s_htf` when no higher interval is visible.
Section 14.3 defines that term as "confirming higher intervals / available",
which is 0/0 when the dataset carries a single interval; the term is scored
0 and the vector is DEGRADED. The consequence is worth stating plainly
rather than smoothing over: on a single-interval bars-only dataset the
maximum attainable S is `1 - htf_confluence` = 0.85 with default weights, so
the `min_significance = 0.60` gate is being asked for 0.60 out of 0.85.

## Boundedness

Every term of S, C and R is in [0, 1] by construction (a clipped ratio, a
`percentile_rank`, or a binary), and every weight set is validated to sum to
1, so S, C, R and `structure_magnitude` are in [0, 1] without any clipping
doing load-bearing work. No raw z-score appears anywhere: an unbounded term
lets one extreme observation dominate a weighted sum.

Four keys are deliberately NOT bounded, because they are measurements in
natural units that a component score must never sum: `nearest_zone_price`,
`nearest_zone_width`, `level_stop_price`, `level_target_price`, plus the
scale-free-but-unbounded `nearest_zone_distance_atr`, `nearest_zone_gap_atr`
and `achievable_rr`. `achievable_rr` in particular must stay unbounded: it
is compared against thresholds, and squashing it would destroy the only
quantity in section 14 that turns setup selection into a measurement.

## Window sizes and warmup

    region       = max(levels.lookback_bars, 2 * pivot_confirm_bars + 1)
    pivot window = features.atr_period + region
    warmup_bars  = pivot window

The region is every bar for which an aligned ATR exists, and it is also the
window over which touches, failed breaks, the volume profile and the ATR
median are measured -- deliberately one window, so that "within the
lookback" means one thing.

The session window (up to `SESSION_WINDOW_DAYS[vwap_anchor]` days of bars)
is *larger* than `warmup_bars` at intraday intervals and is deliberately not
part of it, exactly as in `features/structure.py`. Two reasons. It is an
upper bound on how far back a session boundary can lie rather than a
lookback anything averages over, so the anchors stop changing as soon as the
current and prior sessions are both visible; and `warmup_bars` is a property
with no access to the bar interval, so the bar count spanning four days is
not expressible there at all. When the window is cut short, every affected
anchor is reported in a note and the vector is DEGRADED.

## Key encodings

Three keys are categorical and are encoded numerically, because
`FeatureVector` holds floats:

`level_direction`: +1 long at support, -1 short at resistance, 0 none
(price inside the zone, or no zone).

`level_setup_class`: the setup's own R multiple -- 0 declined, 1 SCALP_1R,
2 SETUP_2R, 3 DIRECTIONAL_3R -- so the encoding is the quantity it names.

`level_gate_reason`: which gate of 14.6 failed first, 0 when none did.
1 no zone at all, 2 proximity, 3 significance, 4 cleanliness, 5 the binary
rejection requirement, 6 no opposing major zone to target, 7 achievable R:R
below `min_reward_risk`. Reasons 1-5 map to `WaitReason.NO_STRUCTURE` and
6-7 to `WaitReason.RR_TOO_LOW`; the finer code is kept so that rejection
analysis can say *which* gate did the work.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import Field, model_validator

from flow_model.config.schema import (
    FeatureConfig,
    LevelCleanlinessWeights,
    LevelSignificanceWeights,
    RejectionWeights,
    StructureLevelConfig,
    StructureMagnitudeWeights,
)
from flow_model.core.contracts import FeatureVector
from flow_model.core.enums import DataQuality, Feed
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel
from flow_model.data.base import SessionCalendarProtocol
from flow_model.data.market_view import MarketView
from flow_model.data.series import NS_PER_SECOND, from_ns, to_ns
from flow_model.features.base import (
    FeatureComputer,
    FeatureError,
    percentile_rank,
    safe_divide,
)
from flow_model.features.structure import (
    SECONDS_PER_DAY,
    SESSION_WINDOW_DAYS,
    atr_aligned,
    find_pivots,
    typical_price,
    volume_weighted_average,
)

__all__ = [
    "DIRECTION_NONE",
    "DIRECTION_RESISTANCE",
    "DIRECTION_SUPPORT",
    "GATE_CLEANLINESS",
    "GATE_NO_TARGET",
    "GATE_NO_ZONE",
    "GATE_PASSED",
    "GATE_PROXIMITY",
    "GATE_REJECTION",
    "GATE_REWARD_RISK",
    "GATE_SIGNIFICANCE",
    "HTF_PIVOT_REGION_BARS",
    "LevelCandidate",
    "LevelFeatures",
    "ROUND_NUMBER_MAX",
    "SETUP_RR_BANDS",
    "VOLUME_PROFILE_MAX_BUCKETS",
    "Zone",
    "age_term",
    "band_volume_profile",
    "build_zone",
    "cleanliness_score",
    "close_position_term",
    "cluster_levels",
    "density_term",
    "displacement_term",
    "distinct_touches",
    "efficiency_term",
    "failed_break_count",
    "flow_term",
    "htf_term",
    "integrity_term",
    "kaufman_efficiency",
    "mean_adjacent_overlap",
    "overlap_term",
    "rejection_magnitude_term",
    "rejection_score",
    "setup_class_for",
    "significance_score",
    "touch_term",
    "volatility_regularity_term",
    "volume_in_band",
]


# ---------------------------------------------------------------------------
# encodings and module constants
# ---------------------------------------------------------------------------

#: `level_direction`. +1 is a long at support, -1 a short at resistance.
DIRECTION_SUPPORT = 1.0
DIRECTION_NONE = 0.0
DIRECTION_RESISTANCE = -1.0

#: `level_gate_reason`: which gate of 14.6 failed first.
GATE_PASSED = 0.0
GATE_NO_ZONE = 1.0
GATE_PROXIMITY = 2.0
GATE_SIGNIFICANCE = 3.0
GATE_CLEANLINESS = 4.0
GATE_REJECTION = 5.0
GATE_NO_TARGET = 6.0
GATE_REWARD_RISK = 7.0

#: 14.6's setup bands, as `(achievable_rr floor, level_setup_class)`, highest
#: floor first. The class value IS the setup's R multiple, so the encoding
#: cannot drift from the thing it names. The floors are section 14.6's
#: numbers verbatim; they are module constants rather than config fields for
#: the same reason `STRUCTURE_SCORE_WEIGHTS` is -- a `FeatureComputer` takes
#: configuration only and `StructureLevelConfig` carries no band edges, and
#: adding them is a config-layer change this module does not own.
SETUP_RR_BANDS: tuple[tuple[float, float], ...] = (
    (2.7, 3.0),
    (1.8, 2.0),
    (1.0, 1.0),
)

#: Upper bound on the number of price buckets in the volume-at-price profile
#: that `s_volume` ranks against. The bucket width is the zone band
#: (`zone_band_atr * ATR`), so 256 buckets covers a region whose high-to-low
#: range is 64 ATR at the default 0.25 band -- comfortably more than a
#: 500-bar range reaches in practice. A judgement call about cost, not a
#: tuning parameter: the profile is a (bars x buckets) array built on every
#: bar, and when the cap binds the buckets are widened and a note says so,
#: which changes the resolution of a percentile rank and nothing else.
VOLUME_PROFILE_MAX_BUCKETS = 256

#: Bars of each higher interval searched for a confirming pivot (`s_htf`).
#: A judgement call: 60 bars is five hours of 15-minute bars, two and a half
#: sessions of hourly bars and three months of daily bars, which is enough to
#: hold several confirmed pivots at every interval `DataConfig` ships. An
#: interval with fewer than `atr_period + this` visible bars is counted as
#: NOT AVAILABLE rather than as unconfirming, so a short higher-interval
#: history lowers the denominator instead of silently voting no.
HTF_PIVOT_REGION_BARS = 60

#: Cap on round-number anchors, nearest to price first. Round numbers are
#: generated across the region's visited price range, so a `round_increment`
#: far smaller than that range would otherwise generate thousands of levels
#: and merge them into one wall. Hitting the cap is noted.
ROUND_NUMBER_MAX = 64


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return 0.0 if value <= 0.0 else (1.0 if value >= 1.0 else float(value))


# ---------------------------------------------------------------------------
# levels and zones (14.1, 14.2)
# ---------------------------------------------------------------------------


class LevelCandidate(FrozenModel):
    """One price the market might respect, before clustering.

    Ages rather than indices, deliberately. A point-in-time computer re-reads
    a rolling window, so the same market level sits at a different array index
    on every bar; what is invariant is how many bars ago it formed and how
    many bars ago it became knowable.

    `known_age_bars >= 0` is the existence proof required by 14.1: a level
    whose confirmation bar has not closed yet has a negative known age and is
    refused here rather than silently entering a zone. For a swing pivot the
    two ages differ by `pivot_confirm_bars`; for an anchor they are usually
    equal, because an anchor is known as soon as the bar defining it closes.
    """

    price: float
    origin_age_bars: int = Field(
        ge=0, description="Bars from the bar that FORMED the level to now."
    )
    known_age_bars: int = Field(
        ge=0, description="Bars from the bar that CONFIRMED the level to now."
    )
    is_anchor: bool = Field(
        description="True for the objectively-defined anchors of 14.1's table."
    )
    source: str = Field(description="Which rule produced it, for notes and tests.")
    origin_index: int = Field(
        ge=-1,
        default=-1,
        description=(
            "Index of the forming bar in the measurement region, or -1 when it "
            "lies before the region. Used only to exclude a level's own forming "
            "bar from its touch count."
        ),
    )

    @model_validator(mode="after")
    def _finite_and_ordered(self) -> "LevelCandidate":
        if not math.isfinite(self.price):
            raise ValueError(f"level price is not finite: {self.price}")
        if self.known_age_bars > self.origin_age_bars:
            raise ValueError(
                f"{self.source}: known_age_bars={self.known_age_bars} exceeds "
                f"origin_age_bars={self.origin_age_bars}, which would mean the "
                "level was confirmed before it formed"
            )
        return self


class Zone(FrozenModel):
    """A merged cluster of levels: 14.2's band, with its provenance kept.

    `low` and `high` are the band the touch and rejection tests use, and the
    band always contains every member. A zone that did not contain its own
    members would make a touch depend on which member happened to carry the
    volume that set `price`.
    """

    price: float = Field(description="Volume-weighted member mean, or the median.")
    low: float
    high: float
    member_count: int = Field(gt=0)
    has_anchor: bool
    #: Age of the NEWEST member. A zone re-confirmed by a recent pivot is not
    #: a stale level, so this is what the age term decays on.
    newest_origin_age_bars: int = Field(ge=0)
    oldest_origin_age_bars: int = Field(ge=0)
    known_age_bars: int = Field(
        ge=0, description="Age of the EARLIEST-known member: when the zone began."
    )
    volume_weighted: bool = Field(
        description="False when `price` fell back to the member median."
    )
    origin_indices: tuple[int, ...] = Field(
        description="Region indices of the member forming bars, excluded from touches."
    )
    sources: tuple[str, ...]

    @model_validator(mode="after")
    def _ordered(self) -> "Zone":
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(f"zone band is not finite: [{self.low}, {self.high}]")
        if self.high < self.low:
            raise ValueError(f"zone high {self.high} is below its low {self.low}")
        if not (self.low <= self.price <= self.high):
            raise ValueError(
                f"zone price {self.price} lies outside its own band "
                f"[{self.low}, {self.high}]"
            )
        if self.newest_origin_age_bars > self.oldest_origin_age_bars:
            raise ValueError("newest member cannot be older than the oldest")
        return self

    @property
    def width(self) -> float:
        return self.high - self.low

    def gap_to(self, price: float) -> float:
        """Distance from `price` to the BAND, zero when inside it."""
        return max(self.low - price, price - self.high, 0.0)

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


def cluster_levels(
    levels: Sequence[LevelCandidate], band: float
) -> tuple[tuple[LevelCandidate, ...], ...]:
    """Single-linkage agglomeration while the price gap is below `band` (14.2).

    Levels are sorted by price and a new cluster starts whenever the gap to
    the previous level is `band` or more. Single linkage means a chain of
    levels each within `band` of the next merges into one zone even when its
    ends are far apart -- 18245.00 and 18248.25 are one level, and so are
    18245.00, 18248.25 and 18251.50 at a 4-point band. That is the documented
    behaviour of the rule section 14.2 names, and it is also its main cost: a
    dense ladder of pivots can chain into one wide zone. `zone_width` is
    reported for exactly that reason.

    The comparison is strict (`gap < band` merges), so a `band` of zero puts
    every distinct price in its own cluster rather than merging all of them.
    Clusters come back ordered by price, members ordered by price within.
    """
    if band < 0.0 or not math.isfinite(band):
        raise FeatureError(
            f"cluster band must be a non-negative finite number, got {band!r}"
        )
    if not levels:
        return ()
    ordered = sorted(levels, key=lambda level: (level.price, level.source))
    clusters: list[list[LevelCandidate]] = [[ordered[0]]]
    for level in ordered[1:]:
        if level.price - clusters[-1][-1].price < band:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return tuple(tuple(cluster) for cluster in clusters)


def build_zone(
    members: Sequence[LevelCandidate],
    weights: Sequence[float],
    min_width: float,
) -> Zone:
    """One zone from one cluster (14.2).

        zone_price = volume-weighted mean of members (median when no profile)
        zone_width = max(member spread, min_width_ticks * tick_size)

    `weights` is the volume at each member's price. When their total is zero
    or non-finite there is no volume distribution to weight by and the
    **median** is used, which is what 14.2 specifies -- not the mean that
    `volume_weighted_average` falls back to, so its flag is consulted and its
    fallback value discarded.

    The band is the specified width centred on `zone_price`, then widened if
    necessary so that it contains every member. The widening is the one place
    this differs from a literal reading of 14.2: a volume-weighted price can
    sit near one edge of a wide cluster, and a band centred there would
    exclude members of the zone it was built from.
    """
    if not members:
        raise FeatureError("build_zone needs at least one member")
    if len(weights) != len(members):
        raise FeatureError(
            f"build_zone needs one weight per member, got {len(weights)} weights "
            f"for {len(members)} members"
        )
    if min_width < 0.0 or not math.isfinite(min_width):
        raise FeatureError(f"min_width must be non-negative and finite, got {min_width!r}")

    prices = np.array([member.price for member in members], dtype=np.float64)
    weighted_price, volume_weighted = volume_weighted_average(
        prices, np.asarray(weights, dtype=np.float64)
    )
    price = weighted_price if volume_weighted else float(np.median(prices))

    spread = float(prices.max() - prices.min())
    width = max(spread, min_width)
    half = width / 2.0
    low = min(price - half, float(prices.min()))
    high = max(price + half, float(prices.max()))
    return Zone(
        price=price,
        low=low,
        high=high,
        member_count=len(members),
        has_anchor=any(member.is_anchor for member in members),
        newest_origin_age_bars=min(member.origin_age_bars for member in members),
        oldest_origin_age_bars=max(member.origin_age_bars for member in members),
        known_age_bars=min(member.known_age_bars for member in members),
        volume_weighted=volume_weighted,
        origin_indices=tuple(
            sorted({member.origin_index for member in members if member.origin_index >= 0})
        ),
        sources=tuple(member.source for member in members),
    )


# ---------------------------------------------------------------------------
# volume at price (14.3 `s_volume`)
# ---------------------------------------------------------------------------


def volume_in_band(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    low: float,
    high: float,
) -> float:
    """Volume traded inside `[low, high]`, from bars (14.3's `s_volume` input).

    Bar data carries no price-by-price volume, so each bar's volume is spread
    **uniformly across its own range** and the part falling inside the band is
    counted. That is the standard bar-derived volume profile and it is an
    approximation with a known bias: real intrabar volume concentrates near
    the close and near the extremes, not uniformly, so a wide bar's volume is
    under-attributed to its own close. The alternative is a tick feed, which
    `MarketView.deltas()` reports as absent rather than estimating; this
    module uses the uniform spread and says so rather than claiming a
    measurement it does not have.

    A zero-range bar is attributed entirely to the band containing its price,
    because spreading a volume over a zero-width range is a division by zero
    and dropping it would lose a halted bar's entire volume.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    if not (highs.size == lows.size == volumes.size):
        raise FeatureError(
            f"volume_in_band needs equal-length arrays, got high={highs.size} "
            f"low={lows.size} volume={volumes.size}"
        )
    if highs.size == 0 or not math.isfinite(low) or not math.isfinite(high) or high < low:
        return 0.0

    ranges = highs - lows
    wide = ranges > 0.0
    overlap = np.clip(
        np.minimum(highs, high) - np.maximum(lows, low), 0.0, None
    )
    share = np.where(wide, overlap / np.where(wide, ranges, 1.0), 0.0)
    total = float((volumes * share).sum())

    if not np.all(wide):
        flat = ~wide
        inside = flat & (lows >= low) & (lows <= high)
        total += float(volumes[inside].sum())
    return total if math.isfinite(total) else 0.0


def band_volume_profile(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Volume in each of the price buckets delimited by `edges`.

    The same uniform-spread attribution as `volume_in_band`, computed for
    every bucket at once. The distribution of these bucket volumes is what
    `s_volume` percentile-ranks a zone against, which is why the buckets are
    all one width: ranking against unequal-width bins would rank width rather
    than volume.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if not (highs.size == lows.size == volumes.size):
        raise FeatureError(
            f"band_volume_profile needs equal-length bar arrays, got "
            f"high={highs.size} low={lows.size} volume={volumes.size}"
        )
    if edges.size < 2:
        raise FeatureError(f"band_volume_profile needs at least two edges, got {edges.size}")
    buckets = edges.size - 1
    out = np.zeros(buckets, dtype=np.float64)
    if highs.size == 0:
        return out

    ranges = (highs - lows)[:, None]
    wide = ranges > 0.0
    overlap = np.clip(
        np.minimum(highs[:, None], edges[None, 1:])
        - np.maximum(lows[:, None], edges[None, :-1]),
        0.0,
        None,
    )
    share = np.where(wide, overlap / np.where(wide, ranges, 1.0), 0.0)
    out += (volumes[:, None] * share).sum(axis=0)

    flat = (highs - lows) <= 0.0
    if np.any(flat):
        width = float(edges[1] - edges[0])
        if width > 0.0:
            index = np.clip(
                ((lows[flat] - edges[0]) / width).astype(np.int64), 0, buckets - 1
            )
            np.add.at(out, index, volumes[flat])
    return out


# ---------------------------------------------------------------------------
# touches, approach quality, failed breaks (14.3, 14.4)
# ---------------------------------------------------------------------------


def distinct_touches(
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    *,
    low: float,
    high: float,
    separation_atr: float,
    separation_bars: int,
    exclude: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    """Indices of the DISTINCT touches of `[low, high]` (14.3's `tau`).

    A bar touches the band when its own range intersects it
    (`high_i >= low and low_i <= high`). Touches are then thinned so that one
    long consolidation is not counted as twenty tests: a touch counts only
    when, since the last counted touch, either

      * at least `separation_bars` bars have elapsed, **or**
      * some intervening bar departed the band by `separation_atr * ATR`.

    The `or` is section 14.3's wording. It is weaker than an `and` would be:
    a twenty-bar grind against the level with `separation_bars = 5` counts
    four touches, not one. That is still "not twenty", which is what 14.3
    asks for, and it is deliberately not tightened here -- 14.3 names the
    rule and section 0.7 makes the constants hypotheses to be swept, not
    quantities to be improved by whoever implements them.

    `exclude` holds the indices of the bars that FORMED the level. They are
    not touches: a swing high touches its own price by construction, and
    counting it would give every pivot zone `tau >= 1`, which would make
    14.3's age term -- defined only for `tau == 0` -- unreachable for every
    level in the system.

    The ATR used for the departure test is the ATR **at the candidate bar**,
    so a departure is measured against the volatility of its own era. A bar
    whose ATR is not positive has no scale on which to measure a departure,
    and only the bar-count test applies there.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    if not (highs.size == lows.size == atr.size):
        raise FeatureError(
            f"distinct_touches needs equal-length arrays, got high={highs.size} "
            f"low={lows.size} atr={atr.size}"
        )
    if separation_bars < 1:
        raise FeatureError(
            f"touch_separation_bars must be at least 1, got {separation_bars}"
        )
    if not (math.isfinite(separation_atr) and separation_atr > 0.0):
        raise FeatureError(
            f"touch_separation_atr must be positive and finite, got {separation_atr!r}"
        )

    touching = np.flatnonzero((highs >= low) & (lows <= high))
    counted: list[int] = []
    last = -1
    for index in touching:
        position = int(index)
        if position in exclude:
            continue
        if last < 0:
            counted.append(position)
            last = position
            continue
        if position - last >= separation_bars:
            counted.append(position)
            last = position
            continue
        scale = float(atr[position])
        if scale > 0.0 and math.isfinite(scale) and position - last > 1:
            between_high = highs[last + 1 : position]
            between_low = lows[last + 1 : position]
            departure = float(
                np.maximum(between_low - high, low - between_high).max()
            )
            if departure >= separation_atr * scale:
                counted.append(position)
                last = position
    return tuple(counted)


def kaufman_efficiency(closes: np.ndarray) -> float:
    """`|net change| / sum(|bar-to-bar change|)` over the window, in [0, 1].

    14.4's approach-quality term: a clean approach is directional, and a
    grind into the level is not. A window with no movement at all returns
    0.0, not 1.0 -- a halted tape is not a clean approach to anything, and
    `0/0` has to resolve somewhere.
    """
    closes = np.asarray(closes, dtype=np.float64)
    if closes.size < 2:
        return 0.0
    steps = np.abs(np.diff(closes))
    total = float(steps.sum())
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    return _clip01(abs(float(closes[-1] - closes[0])) / total)


def mean_adjacent_overlap(highs: np.ndarray, lows: np.ndarray) -> float:
    """Mean overlap of adjacent bar ranges, in [0, 1] (14.4's `O`).

    Each adjacent pair contributes `shared extent / combined extent` -- the
    overlap of the two price intervals divided by their union. Identical bars
    give 1, an inside bar gives the ratio of the ranges, disjoint bars give
    0. The union is used as the denominator rather than the smaller range,
    which would score every inside bar a flat 1.0 and lose the distinction
    between a narrow inside bar and a repeat of the same bar.

    Two bars that both have zero range at the same price have a zero union
    and score 1.0: that is a halted tape, which is maximal churn.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    if highs.size != lows.size:
        raise FeatureError(
            f"mean_adjacent_overlap needs equal-length arrays, got high={highs.size} "
            f"low={lows.size}"
        )
    if highs.size < 2:
        return 0.0
    overlap = np.clip(
        np.minimum(highs[1:], highs[:-1]) - np.maximum(lows[1:], lows[:-1]), 0.0, None
    )
    union = np.maximum(highs[1:], highs[:-1]) - np.minimum(lows[1:], lows[:-1])
    positive = union > 0.0
    fraction = np.where(positive, overlap / np.where(positive, union, 1.0), 1.0)
    return _clip01(float(fraction.mean()))


def failed_break_count(
    closes: np.ndarray, low: float, high: float, horizon: int
) -> int:
    """Failed breaks of `[low, high]` in the window (14.4's `phi`).

    14.4: "a failed break is a close beyond the zone followed by a close back
    inside within `h_fail` bars". Counted by EPISODE rather than by bar: a
    maximal run of consecutive closes beyond the band on one side is one
    break, and it failed when some close returns inside the band within
    `horizon` bars of the run's first close. Counting per bar would score a
    four-bar excursion as four failed breaks and make `s_integrity` a
    function of how long price stayed out rather than of how often it came
    back.

    A run that flips straight to the other side of the band without a close
    inside is not a failed break: price did not return to the level, it went
    through it.
    """
    closes = np.asarray(closes, dtype=np.float64)
    horizon = int(horizon)
    if horizon < 1:
        raise FeatureError(f"failed_break_horizon_bars must be at least 1, got {horizon}")
    if closes.size == 0 or high < low:
        return 0
    side = np.where(closes > high, 1, np.where(closes < low, -1, 0))
    count = 0
    index = 0
    total = side.size
    while index < total:
        current = int(side[index])
        if current == 0:
            index += 1
            continue
        end = index
        while end < total and int(side[end]) == current:
            end += 1
        limit = min(total, index + horizon + 1)
        if np.any(side[index + 1 : limit] == 0):
            count += 1
        index = end
    return count


# ---------------------------------------------------------------------------
# the bounded terms of S, C and R (14.3, 14.4, 14.5)
# ---------------------------------------------------------------------------


def touch_term(touches: int, cap: int) -> float:
    """14.3 `s_touch = min(tau, tau_cap) / tau_cap`."""
    if cap < 1:
        raise FeatureError(f"touch_cap must be at least 1, got {cap}")
    return _clip01(min(int(touches), int(cap)) / float(cap))


def rejection_magnitude_term(
    displacements_atr: Sequence[float], reference: float
) -> float:
    """14.3 `s_reject = clip(median(r_j) / r_ref, 0, 1)`.

    `displacements_atr` is one value per touch whose rejection horizon fits
    inside the measured region. An empty sequence returns 0.0: a level with
    no completed rejection measurement has not been shown to produce moves,
    and 0 is the honest reading of "no evidence" for a term that is supposed
    to reward evidence.
    """
    if not (math.isfinite(reference) and reference > 0.0):
        raise FeatureError(
            f"rejection_reference_atr must be positive and finite, got {reference!r}"
        )
    values = [float(v) for v in displacements_atr if math.isfinite(float(v))]
    if not values:
        return 0.0
    return _clip01(float(np.median(values)) / reference)


def htf_term(confirming: int, available: int) -> float:
    """14.3 `s_htf = confirming higher intervals / available`.

    `available == 0` returns 0.0 rather than raising or defaulting to a
    neutral 0.5. The term's weight is NOT redistributed, so a dataset with
    one interval simply cannot reach the top of S -- see the module
    docstring; this is the same subtraction 14.5 applies to `s_flow`.
    """
    if available <= 0:
        return 0.0
    if confirming < 0 or confirming > available:
        raise FeatureError(
            f"confirming intervals ({confirming}) must lie in [0, {available}]"
        )
    return _clip01(confirming / float(available))


def age_term(age_bars: int, touches: int, decay_lambda: float) -> float:
    """14.3 `s_age = exp(-a / lambda)`, and **1.0 whenever `tau > 0`**.

    The conditional is the whole point of the term and is easy to drop. An
    *untested* level decays with age: nobody has traded against it and the
    longer that stays true the less it means. A level with touches has been
    tested, which makes it confirmed rather than stale, so no decay applies
    and the term contributes its full weight.
    """
    if not (math.isfinite(decay_lambda) and decay_lambda > 0.0):
        raise FeatureError(
            f"age_decay_lambda_bars must be positive and finite, got {decay_lambda!r}"
        )
    if int(touches) > 0:
        return 1.0
    return _clip01(math.exp(-max(int(age_bars), 0) / decay_lambda))


def density_term(recent_touches: int, maximum: int) -> float:
    """14.4 `s_density = clip(1 - rho / rho_max, 0, 1)`."""
    if maximum < 1:
        raise FeatureError(f"max_recent_touches must be at least 1, got {maximum}")
    return _clip01(1.0 - int(recent_touches) / float(maximum))


def efficiency_term(efficiency: float, reference: float) -> float:
    """14.4 `s_eff = clip(E / E_ref, 0, 1)`."""
    if not (math.isfinite(reference) and reference > 0.0):
        raise FeatureError(
            f"efficiency_reference must be positive and finite, got {reference!r}"
        )
    return _clip01(float(efficiency) / reference)


def overlap_term(overlap: float, reference: float) -> float:
    """14.4 `s_overlap = clip(1 - O / O_ref, 0, 1)`."""
    if not (math.isfinite(reference) and reference > 0.0):
        raise FeatureError(
            f"overlap_reference must be positive and finite, got {reference!r}"
        )
    return _clip01(1.0 - float(overlap) / reference)


def integrity_term(failed_breaks: int) -> float:
    """14.4 `s_integrity`, which "decays with phi = failed breaks".

        s_integrity = 1 / (1 + phi)

    **A judgement call, labelled as one.** Section 14.4 names the input and
    the direction but gives no formula, so this is a choice: `1/(1 + phi)` is
    bounded, strictly decreasing, needs no new tunable constant, and gives
    1.0, 0.50, 0.33, 0.25 for zero to three failed breaks. It never reaches
    exactly zero, which is deliberate -- a level poked five times still
    exists, it is simply not clean, and a term that hit 0 would let one input
    zero a weighted sum, which is a gate's job rather than a score's (14.6).

    The alternative considered was `clip(1 - phi / phi_max)` with a new
    `max_failed_breaks` config field. It was rejected because it introduces
    a constant into the largest overfitting surface in the project to express
    something `1/(1 + phi)` already expresses monotonically.
    """
    if failed_breaks < 0:
        raise FeatureError(f"failed break count cannot be negative, got {failed_breaks}")
    return _clip01(1.0 / (1.0 + int(failed_breaks)))


def volatility_regularity_term(atr_now: float, atr_median: float) -> float:
    """14.4 `s_vol = 1 - clip(|log(ATR_t / median ATR)| / log 2, 0, 1)`.

    Penalizes a panic flush and a dead tape symmetrically: a doubling or a
    halving of ATR against its own median both score 0, and ATR at its median
    scores 1. Returns 0.0 when either value is non-positive, because the
    logarithm has no value there and a halted tape is not ordinary
    volatility.
    """
    if not (
        math.isfinite(atr_now)
        and math.isfinite(atr_median)
        and atr_now > 0.0
        and atr_median > 0.0
    ):
        return 0.0
    return _clip01(1.0 - abs(math.log(atr_now / atr_median)) / math.log(2.0))


def close_position_term(
    high: float, low: float, close: float, direction: float
) -> float:
    """14.5 `s_close = clip((p - 0.5) / 0.5, 0, 1)`.

    14.5 writes `p = (close - low) / (high - low)`, which is the support
    case: a long rejection closes near the bar's high. For a short at
    resistance the quantity is mirrored to `(high - close) / (high - low)`,
    exactly as 14.6 mirrors the stop formula. Scoring an unmirrored `p` would
    reward a short rejection bar for closing on its high.

    A zero-range bar has no position within itself; it returns 0.0, which
    denies it the term rather than crediting it with a neutral 0.5.
    """
    span = float(high) - float(low)
    if not math.isfinite(span) or span <= 0.0:
        return 0.0
    if direction > 0.0:
        position = (float(close) - float(low)) / span
    elif direction < 0.0:
        position = (float(high) - float(close)) / span
    else:
        return 0.0
    return _clip01((position - 0.5) / 0.5)


def displacement_term(
    close: float, zone_price: float, atr: float, reference: float
) -> float:
    """14.5 `s_disp = clip(|close - zone_price| / (disp_ref * ATR), 0, 1)`."""
    if not (math.isfinite(reference) and reference > 0.0):
        raise FeatureError(
            f"displacement_reference_atr must be positive and finite, got {reference!r}"
        )
    scale = reference * float(atr)
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    return _clip01(abs(float(close) - float(zone_price)) / scale)


def flow_term(delta: float, direction: float) -> float:
    """14.5 `s_flow`: 1.0 when the signed delta agrees with the direction.

    Binary, as 14.5 states it. A delta of exactly zero agrees with nothing
    and scores 0: balanced flow is not confirmation. The CALLER is
    responsible for not calling this at all when no tick feed exists --
    `MarketView.deltas()` returns an empty array there, and passing a zero
    in its place would be indistinguishable from measured balanced flow.
    """
    if not math.isfinite(delta) or delta == 0.0 or direction == 0.0:
        return 0.0
    return 1.0 if math.copysign(1.0, delta) == math.copysign(1.0, direction) else 0.0


def setup_class_for(achievable_rr: float, min_reward_risk: float) -> float:
    """14.6's setup bands, as the numeric `level_setup_class`.

        rr < min_reward_risk        -> 0 (declined, WAIT "rr_too_low")
        1.0 <= rr < 1.8             -> 1 (SCALP_1R)
        1.8 <= rr < 2.7             -> 2 (SETUP_2R)
        rr >= 2.7                   -> 3 (DIRECTIONAL_3R)

    This is the mechanism section 14.6 calls "setup selection becomes a
    measurement, not a parameter": the class is read off the distance to the
    next significant level divided by the stop distance, and nothing
    configures it. An `rr` that clears `min_reward_risk` but falls below the
    lowest band still returns 0 -- declined, not promoted -- which is why
    `StructureLevelConfig.min_reward_risk` defaults to the lowest band edge.
    """
    rr = float(achievable_rr)
    if not math.isfinite(rr) or rr < float(min_reward_risk):
        return 0.0
    for floor, encoded in SETUP_RR_BANDS:
        if rr >= floor:
            return encoded
    return 0.0


# ---------------------------------------------------------------------------
# the three scores (14.3, 14.4, 14.5)
# ---------------------------------------------------------------------------


def significance_score(
    weights: LevelSignificanceWeights,
    *,
    s_touch: float,
    s_reject: float,
    s_volume: float,
    s_htf: float,
    s_age: float,
    s_anchor: float,
) -> float:
    """14.3 `S = sum(w_i * s_i)`, in [0, 1].

    The bound comes from the weights summing to 1 (validated by
    `LevelSignificanceWeights`) and every term being in [0, 1]; no clipping
    is doing load-bearing work.
    """
    return _clip01(
        weights.touch_count * _clip01(s_touch)
        + weights.rejection_magnitude * _clip01(s_reject)
        + weights.volume_at_level * _clip01(s_volume)
        + weights.htf_confluence * _clip01(s_htf)
        + weights.age_decay * _clip01(s_age)
        + weights.anchor_bonus * _clip01(s_anchor)
    )


def cleanliness_score(
    weights: LevelCleanlinessWeights,
    *,
    s_eff: float,
    s_density: float,
    s_overlap: float,
    s_integrity: float,
    s_vol: float,
) -> float:
    """14.4 `C = sum(v_i * s_i)`, in [0, 1]."""
    return _clip01(
        weights.approach_efficiency * _clip01(s_eff)
        + weights.recent_touch_density * _clip01(s_density)
        + weights.bar_overlap * _clip01(s_overlap)
        + weights.level_integrity * _clip01(s_integrity)
        + weights.volatility_regularity * _clip01(s_vol)
    )


def rejection_score(
    weights: RejectionWeights,
    *,
    s_close: float,
    s_disp: float,
    s_flow: float | None,
) -> float:
    """14.5 `R`, with `s_flow` DROPPED and its weight NOT redistributed.

    `s_flow=None` means no tick feed. The term then contributes nothing and
    the remaining weights are left alone, so R cannot exceed
    `close_position + displacement` -- 0.75 on the defaults. Renormalizing
    would make a bars-only rejection indistinguishable from one confirmed by
    order flow, which is the specific dishonesty `strict_component_availability`
    exists to prevent.
    """
    total = weights.close_position * _clip01(s_close) + weights.displacement * _clip01(
        s_disp
    )
    if s_flow is not None:
        total += weights.order_flow * _clip01(s_flow)
    return _clip01(total)


def structure_magnitude(
    weights: StructureMagnitudeWeights,
    significance: float,
    cleanliness: float,
    rejection: float,
) -> float:
    """14.6 `magnitude = w_S*S + w_C*C + w_R*R`, in [0, 1].

    A weighted sum rather than a product, deliberately: 14.6 states that a
    product would let one weak term zero an otherwise strong setup, which is
    a gate's job and not a score's. The gates run first and separately.
    """
    return _clip01(
        weights.significance * _clip01(significance)
        + weights.cleanliness * _clip01(cleanliness)
        + weights.rejection * _clip01(rejection)
    )


# ---------------------------------------------------------------------------
# the computer
# ---------------------------------------------------------------------------


def _first_index_of_group(stamps: np.ndarray, group, limit: int, grouper) -> int:
    """First index in `[0, limit)` whose group is not before `group`.

    Bisection over a non-decreasing key, for the same reason
    `StructureFeatures` uses one: `grouper` converts a nanosecond stamp
    through `zoneinfo`, and doing that for every bar of a four-day window on
    every bar of a backtest costs more than the rest of this module put
    together.
    """
    left, right = 0, int(limit)
    while left < right:
        middle = (left + right) // 2
        if grouper(int(stamps[middle])) < group:
            left = middle + 1
        else:
            right = middle
    return left


class LevelFeatures(FeatureComputer):
    """Section 14 end to end: zones, S, C, R, the gates, the stop and the target.

    Construction takes configuration only -- a `FeatureConfig`, a
    `StructureLevelConfig`, an `InstrumentSpec`, an optional calendar and the
    higher bar intervals to look for confluence in (a tuple of ints mirroring
    `DataConfig.higher_intervals_seconds`). No series, view, array or
    dataset: capturing a full-sample statistic at construction is the one
    leak `validation.lookahead` cannot see through when handed an instance,
    so it is banned at the constructor and
    `tests/unit/test_feature_contracts.py` enforces the ban package-wide.

    Every number emitted here is a function of the visible prefix of one
    `MarketView`. Zones cannot exist before the bar that confirms them
    because the only detected level is a pivot from `find_pivots`, which
    never returns a pivot whose confirmation bar is past the end of its
    input.
    """

    name = "levels"

    def __init__(
        self,
        config: FeatureConfig,
        levels: StructureLevelConfig,
        spec: InstrumentSpec,
        calendar: SessionCalendarProtocol | None = None,
        higher_intervals: tuple[int, ...] = (),
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

        # One ATR per bar of the measurement region, and wide enough to hold
        # at least one centred confirmation window -- the same sizing as
        # `StructureFeatures`, and for the same reason: without the `2k + 1`
        # floor a config with `lookback_bars=11, pivot_confirm_bars=10` would
        # declare a warmup, pass it, and then find no pivot ever.
        self._region = max(int(levels.lookback_bars), 2 * self._confirm_bars + 1)
        self._window_bars = self._atr_period + self._region
        self._timezone = ZoneInfo(spec.timezone)

        self._higher_intervals = tuple(
            sorted({int(i) for i in higher_intervals if int(i) > 0})
        )
        self._htf_window = self._atr_period + max(
            HTF_PIVOT_REGION_BARS, 2 * self._confirm_bars + 1
        )

    # --- declared contract ---------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """`atr_period + region`: one aligned ATR per bar of the region.

        The session window is deliberately not included -- see the module
        docstring. It is an upper bound on the distance back to a session
        boundary rather than a window anything averages over, and it is not
        expressible here because `warmup_bars` has no access to the bar
        interval. A window that does not reach a prior session reports the
        affected anchors as ABSENT, with a note, and downgrades to DEGRADED.
        """
        return self._window_bars

    @property
    def keys(self) -> tuple[str, ...]:
        return (
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

    @property
    def required_feeds(self) -> frozenset[Feed]:
        return frozenset({Feed.BARS})

    @property
    def optional_feeds(self) -> frozenset[Feed]:
        """Tick aggregates improve R (14.5's `s_flow`) and are never required.

        Their absence subtracts `rejection_weights.order_flow` from the
        attainable R rather than being filled in from bar volume, which is a
        known-bad estimator (section 5).
        """
        return frozenset({Feed.TICK_AGGREGATE})

    # --- session grouping (see the module docstring on the duplication) ---

    def _session_group(self, ts_ns_value: int) -> date:
        assert self.calendar is not None
        return self.calendar.session_date(from_ns(int(ts_ns_value)), self.spec)

    def _vwap_group(self, ts_ns_value: int):
        """The VWAP anchor group, per `features.vwap_anchor`.

        `session` is the calendar's trade date (rolls at the ETH open), `day`
        the exchange-local calendar date, `week` the ISO week of the trade
        date. All three are non-decreasing in the timestamp, which is what
        lets `_first_index_of_group` bisect.
        """
        if self._anchor == "session":
            return self._session_group(ts_ns_value)
        if self._anchor == "day":
            return from_ns(int(ts_ns_value)).astimezone(self._timezone).date()
        iso = self._session_group(ts_ns_value).isocalendar()
        return (iso[0], iso[1])

    def _session_window_bars(self, interval_seconds: int) -> int:
        days = SESSION_WINDOW_DAYS[self._anchor]
        spanning = int(math.ceil(days * SECONDS_PER_DAY / float(interval_seconds))) + 1
        return max(self._window_bars, spanning)

    def _opening_range_bars(self, interval_seconds: int) -> int:
        return max(
            1, int(math.ceil(self._opening_range_minutes * 60.0 / float(interval_seconds)))
        )

    def _rth_bounds(self, ts_ns_value: int):
        """The regular session's open/close for a bar's trade date, or None.

        Reached through `getattr` because `rth_bounds` is on
        `TradingCalendar` but not on `SessionCalendarProtocol`: a conforming
        calendar without it gets the documented absence and a note rather
        than an `AttributeError` swallowed somewhere above. This reads a
        session boundary from calendar arithmetic, never market data.
        """
        assert self.calendar is not None
        bounds_of = getattr(self.calendar, "rth_bounds", None)
        if bounds_of is None:
            return None
        return bounds_of(self._session_group(ts_ns_value), self.spec)

    # --- 14.1 anchors --------------------------------------------------

    def _anchor_levels(
        self,
        stamps: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        region: int,
        interval: int,
        cut_short: bool,
        notes: list[str],
    ) -> tuple[tuple[LevelCandidate, ...], bool]:
        """14.1's anchor table, or honest absences. Returns `(levels, degraded)`.

        Each anchor carries the bar that FORMED it and the bar at which it
        became KNOWN, which differ for a period extreme: the prior session's
        high was printed by one bar but is only final when the session ends.
        Both are needed downstream -- the forming bar is excluded from the
        zone's touch count, and the known bar is the existence proof.

        Nothing here is substituted. Without a calendar there are no
        sessions, so the session-derived anchors are absent and the vector is
        DEGRADED; inventing them out of window statistics would feed a
        fabricated anchor into `s_anchor` and raise S for a level nobody is
        watching.
        """
        out: list[LevelCandidate] = []
        degraded = False
        size = int(closes.size)
        evaluation = size - 1
        offset = size - region

        def add(price: float, origin: int, known: int, source: str) -> None:
            value = float(price)
            if not math.isfinite(value):
                notes.append(f"{self.name}: anchor {source} is not finite; dropped")
                return
            out.append(
                LevelCandidate(
                    price=value,
                    origin_age_bars=evaluation - int(origin),
                    known_age_bars=evaluation - int(known),
                    is_anchor=True,
                    source=source,
                    origin_index=(int(origin) - offset) if int(origin) >= offset else -1,
                )
            )

        # --- round numbers: no session needed --------------------------
        if self.levels.use_round_numbers:
            increment = self.levels.round_increment_points.get(self.spec.symbol)
            if increment is None or not (math.isfinite(increment) and increment > 0.0):
                notes.append(
                    f"{self.name}: use_round_numbers is on but "
                    f"levels.round_increment_points has no positive entry for "
                    f"{self.spec.symbol!r}, so no round-number anchors exist. This "
                    "is a configuration statement, not a data defect, so the "
                    "quality is not downgraded"
                )
            else:
                low = float(lows[-region:].min())
                high = float(highs[-region:].max())
                first = int(math.ceil(low / increment))
                last = int(math.floor(high / increment))
                multiples = [k * increment for k in range(first, last + 1)]
                if len(multiples) > ROUND_NUMBER_MAX:
                    reference = float(closes[-1])
                    multiples = sorted(
                        multiples, key=lambda price: (abs(price - reference), price)
                    )[:ROUND_NUMBER_MAX]
                    notes.append(
                        f"{self.name}: round_increment={increment:g} yields more than "
                        f"{ROUND_NUMBER_MAX} multiples across the region's "
                        f"{high - low:g}-point range; kept the {ROUND_NUMBER_MAX} "
                        "nearest to price"
                    )
                for price in multiples:
                    # A round number is defined timelessly: it has no forming
                    # bar and does not go stale, so its age is zero.
                    add(price, evaluation, evaluation, "round_number")

        wants_session = (
            self.levels.use_prior_session_levels
            or self.levels.use_overnight_levels
            or self.levels.use_opening_range
            or self.levels.use_vwap_levels
        )
        if self.calendar is None:
            if wants_session:
                notes.append(
                    f"{self.name}: no calendar supplied, so there are no sessions. "
                    "The prior-session, overnight, opening-range and VWAP anchors "
                    "of 14.1 are ABSENT, not substituted by window statistics -- a "
                    "window extreme labelled 'prior session high' would raise "
                    "s_anchor for a level nobody is watching"
                )
                degraded = True
            return tuple(out), degraded

        session_start = _first_index_of_group(
            stamps, self._session_group(int(stamps[-1])), size, self._session_group
        )
        bounds = self._rth_bounds(int(stamps[-1]))

        # --- prior session high / low / close --------------------------
        if self.levels.use_prior_session_levels:
            if session_start == 0:
                notes.append(
                    f"{self.name}: all {size} visible bars belong to the current "
                    "session, so the prior-session anchors are absent"
                )
                degraded = True
            else:
                prior_start = _first_index_of_group(
                    stamps,
                    self._session_group(int(stamps[session_start - 1])),
                    session_start,
                    self._session_group,
                )
                high_slice = highs[prior_start:session_start]
                low_slice = lows[prior_start:session_start]
                high_at = prior_start + int(np.argmax(high_slice))
                low_at = prior_start + int(np.argmin(low_slice))
                add(
                    float(high_slice[high_at - prior_start]),
                    high_at,
                    session_start - 1,
                    "prior_session_high",
                )
                add(
                    float(low_slice[low_at - prior_start]),
                    low_at,
                    session_start - 1,
                    "prior_session_low",
                )
                add(
                    float(closes[session_start - 1]),
                    session_start - 1,
                    session_start - 1,
                    "prior_session_close",
                )
                if prior_start == 0 and cut_short:
                    notes.append(
                        f"{self.name}: the prior session starts at the first visible "
                        f"bar of a window cut short at {size} bars, so its high and "
                        "low may be truncated"
                    )
                    degraded = True

        # --- overnight high / low (the ETH range before the RTH open) ---
        if self.levels.use_overnight_levels:
            if bounds is None:
                notes.append(
                    f"{self.name}: the calendar reports no regular session for this "
                    f"trade date (holiday, or no RTH window declared for "
                    f"{self.spec.symbol}), so there is no RTH open to bound an "
                    "overnight range; the overnight anchors are absent"
                )
                degraded = True
            else:
                open_ns = to_ns(bounds[0])
                group = stamps[session_start:size]
                before = np.flatnonzero(group <= open_ns)
                if before.size == 0:
                    notes.append(
                        f"{self.name}: no bar of this session closed at or before the "
                        "RTH open, so there is no overnight range to anchor on"
                    )
                    if cut_short:
                        degraded = True
                else:
                    start = session_start + int(before[0])
                    stop = session_start + int(before[-1]) + 1
                    high_at = start + int(np.argmax(highs[start:stop]))
                    low_at = start + int(np.argmin(lows[start:stop]))
                    add(float(highs[high_at]), high_at, stop - 1, "overnight_high")
                    add(float(lows[low_at]), low_at, stop - 1, "overnight_low")
                    if stop == size:
                        notes.append(
                            f"{self.name}: the RTH open has not arrived yet, so the "
                            "overnight anchors are the overnight range SO FAR"
                        )

        # --- opening range high / low ----------------------------------
        if self.levels.use_opening_range:
            if bounds is None:
                notes.append(
                    f"{self.name}: no regular session for this trade date, so the "
                    "opening-range anchors are absent"
                )
                degraded = True
            else:
                open_ns = to_ns(bounds[0])
                end_ns = open_ns + int(
                    round(self._opening_range_minutes * 60.0 * NS_PER_SECOND)
                )
                group = stamps[session_start:size]
                inside = np.flatnonzero((group > open_ns) & (group <= end_ns))
                if inside.size == 0:
                    notes.append(
                        f"{self.name}: no bar of this session has closed inside the "
                        f"{self._opening_range_minutes:g}-minute opening range yet, so "
                        "the opening-range anchors are absent"
                    )
                else:
                    start = session_start + int(inside[0])
                    stop = session_start + int(inside[-1]) + 1
                    high_at = start + int(np.argmax(highs[start:stop]))
                    low_at = start + int(np.argmin(lows[start:stop]))
                    add(float(highs[high_at]), high_at, stop - 1, "opening_range_high")
                    add(float(lows[low_at]), low_at, stop - 1, "opening_range_low")
                    if stop == size and int(stamps[-1]) < end_ns:
                        notes.append(
                            f"{self.name}: the opening range is still forming "
                            f"({inside.size} of {self._opening_range_bars(interval)} "
                            "bars closed), so its anchors are the range SO FAR"
                        )

        # --- session VWAP and its +/-1 sigma bands ----------------------
        if self.levels.use_vwap_levels:
            vwap_start = _first_index_of_group(
                stamps, self._vwap_group(int(stamps[-1])), size, self._vwap_group
            )
            prices = typical_price(
                highs[vwap_start:], lows[vwap_start:], closes[vwap_start:]
            )
            weights = volumes[vwap_start:]
            vwap, volume_weighted = volume_weighted_average(prices, weights)
            # Recomputed from the anchor on every bar, so a VWAP level has no
            # forming bar and no staleness: its age is zero by construction.
            add(vwap, evaluation, evaluation, "vwap")

            total = float(weights.sum())
            if volume_weighted and total > 0.0:
                variance = float((((prices - vwap) ** 2) * weights).sum() / total)
            else:
                variance = float(prices.var()) if prices.size else 0.0
                notes.append(
                    f"{self.name}: total volume over the {prices.size} bars since the "
                    "VWAP anchor is zero, so the VWAP is an unweighted mean typical "
                    "price and its sigma is an unweighted standard deviation -- "
                    "different statistics under the same names"
                )
                degraded = True
            sigma = math.sqrt(variance) if variance > 0.0 else 0.0
            if sigma > 0.0:
                add(vwap + sigma, evaluation, evaluation, "vwap_plus_sigma")
                add(vwap - sigma, evaluation, evaluation, "vwap_minus_sigma")
            else:
                notes.append(
                    f"{self.name}: the typical price has not varied since the VWAP "
                    "anchor, so the +/-1 sigma bands coincide with the VWAP and are "
                    "not emitted as separate anchors"
                )
            if vwap_start == 0 and cut_short:
                notes.append(
                    f"{self.name}: the {self._anchor} VWAP anchor is at the first "
                    f"visible bar of a window cut short at {size} bars, so the anchor "
                    "may predate the window and the VWAP may be truncated"
                )
                degraded = True

        return tuple(out), degraded

    # --- 14.3 `s_htf` ---------------------------------------------------

    def _htf_pivot_prices(
        self, view: MarketView, notes: list[str]
    ) -> tuple[tuple[np.ndarray, ...], int]:
        """Confirmed pivot prices at each AVAILABLE higher interval.

        `available` is the denominator of 14.3's `s_htf`. An interval with
        too little visible history is NOT available: it lowers the
        denominator rather than silently voting no, because "we cannot see
        the hourly chart yet" and "the hourly chart has no swing here" are
        different statements. An interval that is available and has no
        confirmed pivot does stay in the denominator -- that one genuinely
        is a vote of no confluence.

        Confluence is measured against confirmed higher-interval *pivots*,
        using the same `pivot_confirm_bars` and `pivot_prominence_atr` as the
        primary interval, so a higher-interval level is subject to the same
        right-side confirmation rule and cannot be read before it formed.
        """
        prices: list[np.ndarray] = []
        available = 0
        for interval in self._higher_intervals:
            visible = view.bar_count(interval)
            if visible < self._htf_window:
                notes.append(
                    f"{self.name}: the {interval}s interval holds {visible} of the "
                    f"{self._htf_window} bars needed for a confirmed pivot, so it is "
                    "not counted as an available higher interval for s_htf"
                )
                continue
            highs = view.column("high", self._htf_window, interval)
            lows = view.column("low", self._htf_window, interval)
            closes = view.column("close", self._htf_window, interval)
            if not (highs.size == lows.size == closes.size == self._htf_window):
                notes.append(
                    f"{self.name}: the {interval}s interval returned a short OHLC "
                    f"window ({highs.size}/{lows.size}/{closes.size}); not counted"
                )
                continue
            if not (
                np.all(np.isfinite(highs))
                and np.all(np.isfinite(lows))
                and np.all(np.isfinite(closes))
            ):
                notes.append(
                    f"{self.name}: the {interval}s interval has a non-finite price in "
                    "its window; not counted"
                )
                continue
            atr = atr_aligned(highs, lows, closes, self._atr_period)
            region = int(atr.size)
            found = find_pivots(
                highs[-region:],
                lows[-region:],
                atr,
                confirm_bars=self._confirm_bars,
                prominence_atr=self._prominence_atr,
            )
            available += 1
            prices.append(
                np.array([pivot.price for pivot in found], dtype=np.float64)
            )
        if available == 0:
            notes.append(
                f"{self.name}: no higher interval is available, so 14.3's s_htf is "
                "0/0. The term is scored 0 and its weight is NOT redistributed, "
                "which caps the attainable significance below 1.0"
            )
        return tuple(prices), available
