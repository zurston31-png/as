"""Order-flow features: signed delta, CVD slope, aggression ratio, absorption
at a level, and the trade-size distribution -- plus the bounded magnitude and
the separately-carried direction that the ORDER_FLOW component reads.

ARCHITECTURE.md section 5 gives this component 25 of the 100 points, names
those five features, and states a requirement that no other row in that
table states:

    "No proxy accepted. Component disabled and the 25 points are reported
     UNAVAILABLE; the weight is NOT redistributed. Bar-volume-only 'delta'
     is a known-bad estimator and is not substituted silently."

That sentence, not the formulas, is the organizing principle of this module,
so it is stated precisely before anything else.

## What this component measures

Aggression. Every number here is built from trades that carried an aggressor
side: volume that lifted an offer (`TickAggregate.buy_volume`), volume that
hit a bid (`sell_volume`), the counts of those trades, and the largest print
in a bar. The component answers one question -- *which side of the book is
paying the spread, how hard, and is price moving as a result* -- and nothing
else.

## What it does NOT claim

* **Nothing about who traded.** A large print is an observation about size.
  It is not evidence of an institution, a bank or a "whale";
  ARCHITECTURE.md section 0.3 bans those words from the code, and the
  feature is therefore named `max_trade_size_percentile`, which is what it
  is. One 500-lot print and a hundred 5-lot prints are different
  measurements, not different actors.
* **Nothing about intent.** `absorption` is a measurement -- heavy one-sided
  aggression that did not move price -- and the directional vote it casts
  (*against* the aggressor) is a DECLARED PRIOR to be swept and tested in
  Phase 8, not an established fact. See "Absorption" below.
* **Nothing about predictive content.** No constant here was chosen by
  looking at a result, and nothing in this module has been shown to predict
  anything. The only dataset available is `data/synthetic.py`, whose tick
  stream is generated from the SAME innovations as its bars; it can validate
  mechanics (bounds, determinism, degradation, lookahead, arithmetic) and it
  cannot establish predictive content. A correlation between these features
  and synthetic forward returns would be a property of the generator.

## The measurement sample

Every measured number quoted below comes from one sample, named here so it
is not restated five times: **1130 bars**, taken every 7th bar past warmup
from the 8034 five-minute synthetic NQ bars that
`SyntheticMarketGenerator(SyntheticConfig(), seed=7)` produces for
2017-01-01 to 2017-06-01 with `include_ticks=True`. Below it is "the
sample".

Every measured number quoted in this docstring is a MECHANICAL property --
a bound, a frequency, a correlation between two of this module's own
features, a distribution shape. None of them is evidence that any feature
here predicts anything, and no correlation with synthetic forward returns
was computed, because the generator builds its tick stream from the same
innovations as its bars and any such number would describe the generator.

## The feed, and exactly what happens without it

`required_feeds` is `{BARS, TICK_AGGREGATE}`. Bars supply the price geometry
absorption needs (a local ATR and the window's closes); the tick aggregate
supplies everything else.

Section 5 permits *degradation* for options and liquidity. For order flow it
permits only **disablement**, and this module has no degraded path to the
component as a whole. When the tick feed is absent, or present but
unusable, `compute` returns `FeatureComputer._not_ready(...)`: every key is
0.0, `warmup_complete` is False, every key's `DataQuality` is **MISSING**
(never DEGRADED), `order_flow_available` is 0.0, and a note names the
reason. In that state `FeatureBundle` refuses to produce a score, the
component's 25 points are reported UNAVAILABLE, and **the weight is not
redistributed** -- there is no code path in this file that rescales anything
to compensate for a missing term, and `signals/flow_score.py` is where the
non-redistribution is finally enforced.

The five disabling conditions, each reported by name:

1. The tick-aggregate feed is absent at this instant (`has_feed`).
2. Fewer than `warmup_bars` bars or tick rows are visible.
3. The tick feed does not declare how it classified the aggressor side
   (`TickSeries.classification_method == "unknown"`, the series default), or
   declares a method on `OrderFlowConfig.rejected_classification_methods` --
   the names a bar-volume-derived delta ships under.
4. Measured `classification_coverage` over the read window is below
   `OrderFlowConfig.min_classification_coverage`. A delta whose sign is set
   by a minority of the tape is the known-bad estimator under another name.
5. Terms carrying less than `min_available_weight_fraction` of the total
   term weight have real inputs. Dropping terms without redistribution is
   the honest rule; without this floor it yields a systematically small
   sub-score that reads downstream as "the tape is balanced" when it means
   "most of this was never measured".
6. A required window is short, dirty (a non-finite value) or arithmetically
   unusable: its volumes are finite per row but sum outside double
   precision, so every rolling sum, mean and cumulation over it would be
   infinite. A required input that cannot be read is a reason to disable the
   component, never to estimate around.
7. The newest visible tick row cannot be materialized at all. The feed's
   declared classification method and its right-edge timestamp are only
   reachable through a `TickAggregate`, and `TickSeries.tick_at` coerces the
   trade counts with `int(...)`, which raises on a NaN. Without that row
   neither gate 3 nor the alignment check can be evaluated, so the component
   is disabled. The underlying fix belongs in the data layer -- a non-finite
   optional field should arrive as not-supplied rather than as an exception
   -- and until it moves there this is the honest response here.
8. The newest tick aggregate and the newest bar do not close at the same
   instant (`_alignment_error`), so the absorption term would compare a
   delta window with prices from a different stretch of tape.

Conditions 6 and 7 were added after an independent reviewer found that each
was previously an unhandled exception out of `compute` -- a `FeatureError` on
the overflowing sum and a `ValueError` on the NaN count -- rather than a
reported refusal. `FeatureBundle` does not catch either, so principle 4
("missing data produces WAIT, never a guess") was being satisfied by a
traceback.

What this module will **not** do under any of those conditions, because each
would be indistinguishable from real order flow to every caller above --
including the lookahead audit, which would pass:

* derive a delta from bar volume, signed or unsigned;
* apply an uptick/downtick (tick-rule) classification to bar closes;
* infer a side from the close's location within the bar's range;
* infer a side from quote imbalance;
* split bar volume by any ratio, configured or estimated.

`MarketView.deltas()` returns an EMPTY array rather than an estimate when
there is no tick feed, which is the same rule enforced one layer down. The
two switches that hold the line in configuration --
`allow_bar_volume_delta_proxy` and `require_aggressor_classification` --
cannot be flipped (`OrderFlowConfig` refuses the config), and `__init__`
here refuses them a second time in case a caller bypasses that validator.

## The five features

Throughout, `S = delta_smoothing_bars`, `P = delta_percentile_lookback`,
`C = cvd_lookback_bars`, `G = aggression_lookback_bars`,
`A = absorption_lookback_bars`, `T = trade_size_percentile_lookback`.

### 1. `signed_delta` -- in [-1, 1]

`signed_delta_contracts` is the mean per-bar delta over the last `S` bars,
in contracts: a natural unit, unbounded, a diagnostic that no weighted sum
may read. The bounded term is its percentile rank within the last `P`
values of that same smoothed series, mapped to [-1, 1]:

    signed_delta = 2 * percentile_rank(smoothed_window, smoothed_now) - 1

A percentile, not a z-score, because the component score is a weighted sum
and an unbounded term lets one print dominate it (`features/base.py`).

The consequence of ranking the *signed* value within its own history is
worth stating plainly, because it is a behaviour and not a bug: the term
reads "more buy-pressured than this tape's own recent normal", not "net
buying". On a tape with a persistent one-sided delta the window re-centres,
so a bar with a positive but below-median delta reads negative.
`signed_delta_contracts` is emitted alongside precisely so that the absolute
fact is not lost, and the re-centring is a Phase-8 hypothesis, not a
finding.

A window in which every smoothed delta is identical (a perfectly balanced
tape) is reported as 0.0 with a note rather than passed to
`percentile_rank`, which counts ties as "at or below" and would report a
dead tape as maximum buy pressure -- the inverse of the truth, and the same
tie-inflation that `features/liquidity.py` documents for quoted spreads.

### 2. `cvd_slope` -- in [-1, 1]

Cumulate the per-bar delta over `C` bars and take the ordinary
least-squares slope of that path against bar index. The slope is in
contracts per bar, which is not comparable across NQ, GC and QQQ, so it is
divided by the window's own mean classified volume per bar:

    cvd_slope_normalized = slope(cumsum(delta_window)) / mean(buy + sell)
    cvd_slope            = signed_squash(cvd_slope_normalized,
                                        cvd_slope_squash_scale)

One window, not two: the cumulation and the regression span the same bars,
so the reported slope does not depend on a ratio between two constants
nobody chose on purpose. The division is what makes the term dimensionless
and comparable; the squash scale only sets how fast it saturates.

**On the overlap with `signed_delta`, measured rather than assumed.** The
OLS slope of a cumulative sum is a weighted mean of its increments, so this
term and `signed_delta` are built from the same deltas and the default
weights give them 0.50 of the component between them. That looked like a
redundancy worth flagging, and the measurement says it mostly is not: over the
sample, `corr(signed_delta, cvd_slope) = 0.075` and
`corr(signed_delta_contracts, cvd_slope_normalized) = 0.205`. The windows
are what separate them -- a 3-bar smoothed delta ranked over 60 bars against
a 20-bar regression -- so the two terms are nearly independent at these
defaults and would collapse into one if `delta_smoothing_bars` were swept up
toward `cvd_lookback_bars`. That coupling is a Phase-8 note, not a reason to
change a default here.

### 3. `aggression_ratio` -- in [0, 1], 0.5 = balanced

    aggression_ratio = sum(buy_trades) / sum(buy_trades + sell_trades)

pooled over `G` bars, and the score reads `2 * aggression_ratio - 1`.

Deliberately built from trade COUNTS, which is a different measurement from
`TickAggregate.delta_ratio`'s volume weighting: many small lifts and one
large one give the same delta and a very different count ratio. Counts in a
single bar are few, so they are pooled.

Below `min_classified_trades` in the pool the term is **dropped, not
neutralised**. A 3-trade 2:1 ratio emitting 0.67 is indistinguishable
downstream from a real two-thirds reading, so the emitted value is 0.5
(`NEUTRAL_RATIO`), the key is graded DEGRADED, a note says so, and the
term's weight is removed from both the magnitude and the available-weight
fraction -- not reallocated.

Unlike `signed_delta`, this term is centred absolutely at 0.5 rather than
percentile-ranked against its own history. Two of the five terms are
relative (signed delta, trade size) and three are absolute (aggression,
CVD-over-volume, absorption geometry in ATR units). That asymmetry is a
declared choice: a count ratio has a meaningful absolute centre (balanced)
while a contract count does not. It is also a Phase-8 candidate.

### 4. `absorption` -- in [0, 1], with `absorption_direction` in {-1, 0, +1}

Effort versus result, measured over `A` bars, as the geometric mean of three
[0, 1] terms:

    effort = clamp01((rank - absorption_delta_percentile)
                     / (1 - absorption_delta_percentile))
             where rank = percentile_rank(|window sums| history, |sum now|)
    quiet  = clamp01(1 - |close[-1] - close[-1-A]| / ATR
                         / absorption_max_displacement_atr)
    flat   = clamp01(1 - (max(close[-A:]) - min(close[-A:])) / ATR
                         / absorption_level_window_atr)

    absorption = (effort * quiet * flat) ** (1/3)

A geometric mean is an AND: a zero on any term sends the result to zero,
which is what "heavy one-sided flow, at one level, that did not move price"
means. Heavy delta WITH displacement is a drive, not absorption, and
scoring the two the same way would make the term unreadable.

`quiet` and `flat` are not redundant. `quiet` is the NET change across the
window and reads its reference close from *before* the window, so it sees a
gap; `flat` is the whole excursion *within* the window, so it separates a
window that drifted 0.2 ATR from one that ran 0.2 ATR up and came back. A
round trip is absorption; a drift is not.

**"At a level" is measured as local flatness, not by reading the structure
layer's zone list.** `features/levels.py` already exposes nearest-zone
geometry, and composing with it was considered and rejected: a
`FeatureComputer` that consumed another computer's output could not be
audited for lookahead in isolation, which is how every other computer in
this package is audited, and it would couple the order-flow component's
warmup to the 514-bar level stack. `StructureGate` still gates on real
zones (section 14.6); this term only reports that aggression was absorbed
somewhere flat, and `absorption_level_window_atr` mirrors
`levels.zone_band_atr` so that "flat" means the same width in both places.

**MEASURED: with the default constants this term is identically zero on
five-minute data, and 0.15 of the component's weight can never be earned.**
Over the sample, `effort > 0` on 18.8% of windows and `quiet > 0`
on 16.9%, both working as intended -- but `flat > 0` on **0.0%**. The
10-close excursion `band_atr` has minimum 0.2813, median 1.3026 and p95
2.7791, and `absorption_level_window_atr` is 0.25, so the band condition is
never met and the geometric mean is always zero. The reason is a scale
mismatch rather than a coding error: 0.25 ATR is anchored on
`levels.zone_band_atr`, which is a *cluster merge distance between two
price levels*, while this term compares it against the excursion of ten
consecutive closes, which for a random walk runs about `sqrt(A)` ATR -- 3.16
ATR at `A = 10`. Measured thresholds: `band_atr < 0.5` on 1.6% of windows,
`< 0.79` (= 0.25 * sqrt(10)) on 12.3%, `< 1.0` on 27.4%.

This is reported, not fixed. The constant is a hypothesis for the Phase 8
sweep and tuning it here to make the term fire would be choosing a value by
looking at a result. The minimal honest repairs, for whoever owns
`config/schema.py`: scale the band by `sqrt(absorption_lookback_bars)` so
the threshold is a statement about flatness rather than about a single
bar, or raise the default to roughly 0.75-1.0 and say what event frequency
it is meant to describe. Note also that the term stays *available* while it
is zero -- its inputs exist and the measurement is real -- so
`order_flow_available_weight` reports 1.0 and nothing downstream can see
that 0.15 of the magnitude is unreachable. That is the same invisible
ceiling `OrderFlowConfig` and `OptionsFlowConfig` already refuse in other
forms, and it belongs in a config validator.

**The direction this term votes, and why it is a prior.**
`absorption_direction` is `-sign(window delta)`: absorbed buying votes
bearish, absorbed selling votes bullish, on the reading that the passive
side is the one in control when aggression fails to move price. That is a
standard interpretation and it is still only an interpretation. It is also
the one term whose vote OPPOSES `signed_delta` on the same bar, so the two
cancel in the composition below -- which is the intended behaviour:
contradictory evidence should produce a small magnitude, not a confident
one.

The ATR used is **local to this computer** -- the Wilder seed over the last
`FeatureConfig.atr_period` true ranges -- and is deliberately not
`VolatilityFeatures`' `atr`, which is anchored at the start of its own
266-bar window. The two are close in the middle of the distribution and not
interchangeable in the tails: over the sampled bars their correlation is
0.968 and the ratio local/volatility has median 0.991 and quartiles 0.923 /
1.052, but its 5th percentile is 0.733 and its extremes are 0.308 and 1.433.
A 15-bar seed reprices a volatility collapse within a session while a
266-bar Wilder recursion is still carrying the old level, which for a
*local* geometric scale is the more useful behaviour and is why the
difference is not treated as an error. The two differ by their anchor (volatility.py documents
this), and reading another computer's output is the dependency rejected
above. `features/volatility.py`'s `true_range` and `wilder_atr_series` are
imported so that the true-range CONVENTION has one definition in the
system; they are pure array functions, not a computer, and importing them
creates no cycle.

### 5. The trade-size distribution -- two percentiles and an event flag

Two moments of the distribution, pooled over `G` bars and each percentile-
ranked within the last `T` pooled observations:

    avg_trade_size_percentile  -- rank of sum(classified volume)
                                  / sum(classified trades)   (central)
    max_trade_size_percentile  -- rank of max(max_trade_size) (tail)
    large_trade_event          -- 1.0 when the tail rank is at or above
                                  large_trade_percentile, else 0.0

The score reads `clamp01(2 * mean(available ranks) - 1)`, so an ordinary or
small distribution contributes nothing and only an unusually heavy one
contributes. This term casts **no directional vote**: `TickAggregate`
reports `max_trade_size` without a side, so which way the large prints
leaned is not in the data. The term is magnitude-only because of the feed,
not because of a modelling preference, and inferring the side from the
bar's delta would be exactly the kind of invention this module refuses.

Bucket EDGES were rejected in the config for a reason that applies here
too: the feed supplies a maximum and counts, not a histogram, so absolute
edges would describe a measurement that was never made.

`percentile_rank` counts ties as "at or below", and `max_trade_size` is a
discrete quantity, so a pooled maximum that ties with much of its history
ranks high -- the tie inflation `features/liquidity.py` documents for quoted
spreads. Measured over the sample, the inflation is mild but real:
`max_trade_size_percentile` has median 0.533 and p90 0.917, and
`large_trade_event` fires on 16.2% of bars at a threshold of 0.90, where an
untied rank would fire on about 10%. Reported, not corrected by moving
`large_trade_percentile`. A tape with a single print size everywhere is a
different case and is handled explicitly: the rank is reported as
`UNAVAILABLE_PERCENTILE` and the event flag degrades with it, because a
tie-inclusive rank of 1.0 would otherwise fire the flag on every bar.

## Composition: magnitude, and direction carried separately

ARCHITECTURE.md section 7 requires direction to be carried separately from
magnitude, because the common bug is a strong bearish component inflating a
bullish score. So:

    dir_sum   = SUM over AVAILABLE directional terms of w_i * signed_i
    magnitude = |dir_sum| + w_size * size_strength   (clamped to [0, 1])
    direction = sign(dir_sum)                        in {-1, 0, +1}

`|dir_sum|` rather than a sum of magnitudes, which means **conflicting terms
cancel**. A bar with a strongly positive signed delta and an active
absorption reading nets out near zero and reports a low magnitude with no
confident direction, because the evidence is contradictory. The alternative
-- averaging the term magnitudes and taking the sign of the net -- would
report "strong order flow, direction uncertain", which is a high score for a
tape nobody can read. This is a declared choice and a Phase-8 candidate.

The weights sum to 1.0 (`OrderFlowWeights` enforces it), and every term is
in [-1, 1], so the magnitude is bounded by construction rather than by the
clamp; the clamp is a guard against a future weight edit, not a transform.

**Dropped terms are dropped.** A term whose input is absent contributes
nothing and its weight is NOT reallocated to the others, so the attainable
magnitude falls to `order_flow_available_weight`. That is the honest
consequence of missing data (section 5, and 14.5 for the same rule applied
to `s_flow`), and it is why `min_available_weight_fraction` exists: below
that floor the component reports UNAVAILABLE instead of a quarter-strength
opinion.

A consequence of composing four signed terms, stated with the measurement
rather than in advance: the magnitude reaches 1.0 only when all four
directional terms saturate in the SAME direction and the size term
saturates too, so real readings sit low in [0, 1]. Over the sample
the distribution is minimum 0.0001, p25 0.105, median
0.195, p75 0.312, p90 0.405, p99 0.564 and **maximum 0.660** -- the top
third of the range was never reached, and with `absorption` structurally
zero (above) 0.15 of the weight was unreachable throughout. If Phase 6 finds
the component systematically under-contributes against the other four, the
fix belongs in the Phase 8 weight sweep or in `signals/flow_score.py`'s
normalization. It does not belong in a constant tuned here, and the
cancellation that produces the low readings is the intended behaviour, not
the defect.

## Numeric encodings

Every emitted value is a finite float (the base class rejects non-finite),
so the categorical facts are encoded:

| key | encoding |
|---|---|
| `order_flow_direction` | -1.0 bearish, 0.0 none, +1.0 bullish |
| `order_flow_available` | 1.0 usable, 0.0 UNAVAILABLE |
| `absorption_direction` | -1.0 absorbed buying, 0.0 none, +1.0 absorbed selling |
| `large_trade_event` | 1.0 the pooled max print ranks at or above `large_trade_percentile` |
| `aggression_ratio` | buy share of classified trades |

0.5 is both a balanced `aggression_ratio` and the placeholder emitted when
the term is dropped, and `UNAVAILABLE_PERCENTILE` (also 0.5) is both a
median rank and the placeholder for an unmeasured one. The key's own
`DataQuality` is what separates the two -- GOOD for a measurement, DEGRADED
for a placeholder -- which is why `FeatureVector` carries a status per key
rather than one per vector.

## Boundedness

`order_flow_magnitude`, `absorption`, `aggression_ratio`,
`avg_trade_size_percentile`, `max_trade_size_percentile` and
`classification_coverage` are in [0, 1]; `signed_delta` and `cvd_slope` are
in [-1, 1]. Each is a percentile rank, a share, or a `tanh` squash --
bounded by construction, never a clamp applied to an unbounded quantity, and
never a z-score.

`signed_delta_contracts` and `cvd_slope_normalized` are in natural and
dimensionless units respectively. They are diagnostics: reported, plotted
and regressed on, and a weighted sum must not read them, because one extreme
observation in an unbounded feature dominates the sum.

## Warmup: the longest chain actually read

Windows compose, and under-declaring `warmup_bars` is a HIGH-severity bug
the lookahead audit cannot see -- the regime detector declared 268 and read
291, which made the label at bar *t* depend on where the caller started
loading while nothing read the future. The chains, on defaults:

| chain | arithmetic | rows |
|---|---|---|
| smoothed delta, then its percentile | `P + S - 1` | 62 |
| absorption window sum, then its percentile | `P + A - 1` | 69 |
| pooled trade size, then its percentile | `T + G - 1` | 129 |
| CVD cumulation and regression | `C` | 20 |
| pooled aggression counts | `G` | 10 |
| local ATR (bars) | `atr_period + 1` | 15 |
| absorption price geometry (bars) | `A + 1` | 11 |

The longest tick chain is 129 rows and the longest bar chain is 15 bars.
`OrderFlowConfig.warmup_bars` publishes 130 (the same maxima plus one) and
documents itself as a FLOOR that no computer may declare below, so this
computer declares `max(129, 15, 130) = 130`. It reads 129 tick rows and 15
bars, so the declaration is one bar conservative -- the safe direction to be
wrong in, and stated here rather than silently matched.

Tick rows are counted in TICK OBSERVATIONS and bars in BARS. The two feeds
have independent cadences, so `compute` re-checks that it actually received
the rows it asked for and reports not-ready instead of computing from a
shorter window than it declared.

## What is verified about tick/bar alignment, and what is not

The absorption term reads deltas from the tick feed and prices from the bar
feed, which is only meaningful if row *i* of one window describes the same
interval as row *i* of the other. `compute` verifies the right edge: the
newest visible tick aggregate's `close_ts` must equal the newest visible
bar's `close_ts`, and both windows must be full.

Interior per-row alignment is NOT verified, because `MarketView` exposes no
tick timestamps -- `tick_column` returns values only, and
`tick_aggregates(n)` materializes `n` contract objects at a measured 622
microseconds for a 130-row window, which is 2.7x the whole of
`VolatilityFeatures.compute` and far too expensive per bar. A tick feed with
an interior gap would therefore have its delta window span more bars than
its price window. This is reported as a known limitation; the fix is a
`tick_timestamps(n)` accessor on `MarketView` alongside
`bar_timestamps(n)`, which is a data-layer change and not this module's to
make.

## Cost

Measured at 467 microseconds per `compute` on a 130-bar window, against 325
for `VolatilityFeatures.compute` on its 266-bar window, on the same machine
and the same synthetic series. The work is NumPy call overhead on short
arrays plus two materialized contract objects (the newest tick row and the
newest bar, each read once -- see `compute`). A full pass over the
8034-bar sample costs about 3.8 seconds.

## Determinism

No clock, no RNG, no mutable state: every value is a pure function of the
view's windows. Two computations on an identical view agree bit-for-bit, and
two independently constructed computers agree, because `__init__` stores
only integers and floats read from configuration. It accepts CONFIGURATION
ONLY -- a `FeatureConfig` and an `OrderFlowConfig`, never a series, view,
array or dataset -- because a full-sample statistic captured at construction
is the one leak `validation.lookahead` cannot see through when handed an
instance, and `tests/unit/test_feature_contracts.py` enforces the ban
package-wide.
"""

from __future__ import annotations

import math

import numpy as np

from flow_model.config.schema import FeatureConfig, OrderFlowConfig
from flow_model.core.contracts import Bar, FeatureVector, TickAggregate
from flow_model.core.enums import DataQuality, Feed
from flow_model.data.market_view import MarketView
from flow_model.features.base import (
    FeatureComputer,
    FeatureError,
    percentile_rank,
    safe_divide,
)

# `least_squares_slope` and `signed_squash` are pure array/scalar functions
# that already exist in `features/liquidity.py`, and `true_range` /
# `wilder_atr_series` in `features/volatility.py`. They are imported rather
# than restated so that each formula -- an OLS slope, a signed tanh squash,
# the true-range convention -- has ONE definition in the system. None of them
# is a `FeatureComputer`, and this module never reads another computer's
# `FeatureVector`: that is the dependency that would break per-computer
# lookahead auditing, and it is not taken here.
from flow_model.features.liquidity import least_squares_slope, signed_squash
from flow_model.features.volatility import true_range, wilder_atr_series

#: The `TickSeries` default `classification_method`.
#:
#: An undeclared method is not a classification: volume with an unexplained
#: side is bar volume with a label. `OrderFlowConfig.
#: require_aggressor_classification` is True and cannot be turned off, so a
#: feed carrying this value disables the component.
UNKNOWN_CLASSIFICATION_METHOD = "unknown"

#: The value emitted for `aggression_ratio` when the term is dropped.
#:
#: 0.5 is also the value a genuinely balanced pool produces, which is
#: unavoidable -- the key must hold a finite float -- so the two are
#: distinguished by the key's `DataQuality` (DEGRADED when dropped, GOOD when
#: measured) and by a note. `min_available_weight_fraction` is what stops a
#: run of placeholders from adding up to a usable component.
NEUTRAL_RATIO = 0.5

#: The value emitted for a percentile that could not be measured.
#:
#: The no-information point of a [0, 1] rank, and the same convention
#: `features/liquidity.py` uses for an unavailable `spread_percentile`. It
#: maps to a strength of 0.0 in `_size_strength`, so an unmeasured
#: distribution contributes nothing rather than something small.
UNAVAILABLE_PERCENTILE = 0.5

#: Terms in the absorption geometric mean: effort, quiet, flat.
ABSORPTION_TERM_COUNT = 3

#: Proxy constructions this module refuses to perform, named so that the
#: refusal is greppable and testable rather than implied by absent code.
#:
#: Section 5 rejects a bar-volume-derived delta by name; the rest are the
#: same substitution in other clothing. Every one of them would produce a
#: plausible number that no caller above -- including the lookahead audit,
#: which reads only past data and would pass -- could distinguish from real
#: order flow.
REFUSED_PROXY_CONSTRUCTIONS: tuple[str, ...] = (
    "bar volume as signed delta",
    "tick rule (uptick/downtick) applied to bar closes",
    "close location within the bar range as an aggressor side",
    "quote imbalance as an aggressor side",
    "any configured or estimated split of bar volume into buy and sell",
)


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Sums of every consecutive `window`-long block, oldest-to-newest.

    Length `values.size - window + 1`. A sliding window rather than a
    difference of cumulative sums: the cumsum form loses precision by
    cancellation when a long prefix dwarfs the block, and these blocks are a
    few hundred elements at most, so the honest arithmetic is free.
    """
    if window < 1:
        raise FeatureError(f"rolling window must be at least 1, got {window}")
    if values.size < window:
        raise FeatureError(
            f"rolling_sum over {window} needs at least {window} values, "
            f"got {values.size}"
        )
    return np.lib.stride_tricks.sliding_window_view(values, window).sum(axis=-1)


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """Maxima of every consecutive `window`-long block, oldest-to-newest."""
    if window < 1:
        raise FeatureError(f"rolling window must be at least 1, got {window}")
    if values.size < window:
        raise FeatureError(
            f"rolling_max over {window} needs at least {window} values, "
            f"got {values.size}"
        )
    return np.lib.stride_tricks.sliding_window_view(values, window).max(axis=-1)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Means of every consecutive `window`-long block, oldest-to-newest."""
    if window < 1:
        raise FeatureError(f"rolling window must be at least 1, got {window}")
    if values.size < window:
        raise FeatureError(
            f"rolling_mean over {window} needs at least {window} values, "
            f"got {values.size}"
        )
    return np.lib.stride_tricks.sliding_window_view(values, window).mean(axis=-1)


def signed_rank(window: np.ndarray, value: float) -> float:
    """`2 * percentile_rank(window, value) - 1`, in [-1, 1].

    0.0 at the window's median, +1 at its maximum, -1 below everything in
    it. A degenerate window -- every observation identical -- returns 0.0
    rather than the +1.0 that a tie-inclusive rank would produce, because a
    dead or perfectly balanced tape is not an extreme of anything. The
    caller is expected to note the degeneracy; see `signed_delta` in the
    module docstring.
    """
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return 0.0
    if float(finite.max()) == float(finite.min()):
        return 0.0
    return 2.0 * percentile_rank(finite, value) - 1.0


def is_degenerate(window: np.ndarray) -> bool:
    """Whether every finite observation in `window` holds the same value."""
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return True
    return float(finite.max()) == float(finite.min())


def excess_above(rank: float, threshold: float) -> float:
    """How far a [0, 1] rank sits above `threshold`, rescaled to [0, 1].

    0.0 at or below the threshold, 1.0 at a rank of 1.0. Used so that
    "extreme" is a graded term rather than a step: a gate's job is to be
    binary (section 14.6), a score's is to say how good.
    """
    if threshold >= 1.0:
        raise FeatureError(
            f"threshold {threshold} leaves no room above it; a percentile "
            "threshold must be below 1.0"
        )
    return _clamp01((rank - threshold) / (1.0 - threshold))


def inside_band(measurement: float, limit: float) -> float:
    """How far inside a band a non-negative measurement sits, in [0, 1].

    1.0 at zero, 0.0 at or beyond `limit`. The graded form of "price barely
    moved": `1 - measurement / limit`, clamped.
    """
    if limit <= 0.0:
        raise FeatureError(f"band limit must be positive, got {limit}")
    return _clamp01(1.0 - measurement / limit)


class OrderFlowFeatures(FeatureComputer):
    """The five order-flow features, a bounded magnitude, and a direction.

    Construction takes a `FeatureConfig` (for `atr_period` only) and an
    `OrderFlowConfig`, and nothing else. No series, no view, no array: see
    the module docstring on why that is a contract rather than a style.
    """

    name = "orderflow"

    def __init__(self, config: OrderFlowConfig, features: FeatureConfig) -> None:
        # The no-proxy switches, checked a second time. `OrderFlowConfig`'s
        # own validator refuses both, so reaching this branch means a caller
        # built the config through `model_construct` or an equivalent bypass.
        # The component would still not substitute a proxy -- there is no code
        # here that could -- but a config asserting that it may is a
        # disagreement about the rule, and the right response is to refuse to
        # run rather than to run correctly while the config says otherwise.
        if config.allow_bar_volume_delta_proxy:
            raise FeatureError(
                "allow_bar_volume_delta_proxy is True. ARCHITECTURE.md section 5 "
                "accepts NO proxy for order flow: a bar-volume-derived delta is a "
                "known-bad estimator and would be indistinguishable from real order "
                "flow to every caller above, the lookahead audit included. This "
                "computer reports the component UNAVAILABLE instead; it has no "
                "proxy path to enable."
            )
        if not config.require_aggressor_classification:
            raise FeatureError(
                "require_aggressor_classification is False, which would accept a "
                "tick feed that does not say how it assigned the aggressor side. An "
                f"undeclared method ({UNKNOWN_CLASSIFICATION_METHOD!r}, the "
                "TickSeries default) is not a classification, so this is the "
                "bar-volume proxy admitted through the back door."
            )

        self.config = config
        self.features = features

        self._smoothing = int(config.delta_smoothing_bars)
        self._delta_lookback = int(config.delta_percentile_lookback)
        self._cvd_lookback = int(config.cvd_lookback_bars)
        self._aggression_lookback = int(config.aggression_lookback_bars)
        self._absorption_lookback = int(config.absorption_lookback_bars)
        self._size_lookback = int(config.trade_size_percentile_lookback)
        self._atr_period = int(features.atr_period)

        # The composed chains, spelled out. See the module docstring's table.
        self._delta_chain = self._delta_lookback + self._smoothing - 1
        self._absorption_chain = self._delta_lookback + self._absorption_lookback - 1
        self._size_chain = self._size_lookback + self._aggression_lookback - 1
        self._tick_rows = max(
            self._delta_chain,
            self._absorption_chain,
            self._size_chain,
            self._cvd_lookback,
            self._aggression_lookback,
        )
        # Bars are read only for the absorption geometry: a local ATR, and the
        # window's closes plus the one before it.
        self._bar_rows = max(self._atr_period + 1, self._absorption_lookback + 1)

        self._weights = config.weights
        self._total_weight = (
            self._weights.signed_delta
            + self._weights.cvd_slope
            + self._weights.aggression_ratio
            + self._weights.absorption
            + self._weights.trade_size_distribution
        )
        if self._total_weight <= 0.0:
            raise FeatureError(
                "order-flow term weights sum to zero, so the component could "
                "never report anything; OrderFlowWeights requires 1.0"
            )
        self._rejected_methods = frozenset(
            method.strip().lower() for method in config.rejected_classification_methods
        )

    # --- declared contract ---------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """Bars required before any output is meaningful.

        The longest chain this computer reads, floored at
        `OrderFlowConfig.warmup_bars`, which the config documents as a bound
        no computer may declare below. On defaults the config's floor (130)
        is one bar above the longest chain (129), so the floor binds and the
        declaration is one bar conservative.
        """
        return max(self._tick_rows, self._bar_rows, int(self.config.warmup_bars))

    @property
    def keys(self) -> tuple[str, ...]:
        return (
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

    @property
    def required_feeds(self) -> frozenset[Feed]:
        """Bars and tick aggregates, both required.

        `TICK_AGGREGATE` is required rather than optional, which is the whole
        point: section 5 gives this component no degraded path, so an absent
        tick feed disables it instead of switching it to an estimate.
        """
        return frozenset({Feed.BARS, Feed.TICK_AGGREGATE})

    @property
    def optional_feeds(self) -> frozenset[Feed]:
        """None.

        Quotes could classify ticks, but classification is the DATA LAYER's
        job (`TickSeries.classification_method`); doing it here from L1 would
        be this module inventing an aggressor side, which is the refusal in
        `REFUSED_PROXY_CONSTRUCTIONS`.
        """
        return frozenset()

    # --- computation ---------------------------------------------------

    def compute(self, view: MarketView) -> FeatureVector:
        """Features at `view.now`, or an UNAVAILABLE vector with the reason.

        Warmup and feed availability are re-checked here rather than left to
        `FeatureBundle`: the lookahead audit calls computers directly,
        including at pre-warmup bars and against datasets with no tick feed,
        and a computer that reported `warmup_complete=True` with a half-
        filled window would be handing out a number built from fewer
        observations than it claims.
        """
        if not view.has_feed(Feed.BARS):
            return self._unavailable(view, "the bars feed is absent")
        if not view.has_feed(Feed.TICK_AGGREGATE):
            return self._unavailable(
                view,
                "the tick-aggregate feed is absent, so there is no classified "
                "aggressor volume to measure",
            )

        required = self.warmup_bars
        if view.bar_count() < required:
            return self._unavailable(
                view, f"warmup incomplete: {view.bar_count()} of {required} bars"
            )
        if view.tick_count() < required:
            return self._unavailable(
                view,
                f"warmup incomplete: {view.tick_count()} of {required} tick "
                "observations (the tick feed has its own cadence, so bar warmup "
                "is a floor on this check and never a substitute for it)",
            )

        # The newest visible tick row, read ONCE: it carries both the feed's
        # declared classification method (from the series meta) and the
        # right-edge timestamp the alignment check needs, and materializing
        # a contract object twice for one row is waste on a per-bar path.
        try:
            newest_tick = view.last_tick_aggregate()
        except (ValueError, TypeError) as error:
            # `TickSeries.tick_at` coerces `buy_trades`/`sell_trades` with
            # `int(...)`, which raises on a NaN. `_read_windows` treats a broken
            # count column as NOT SUPPLIED, but it never runs: materializing the
            # newest row happens first, because the feed's declared
            # classification method and the right-edge timestamp are only
            # reachable through a `TickAggregate`. Without that row there is no
            # way to check either gate, so the component is disabled rather than
            # measured -- and, critically, rather than raising out of a feature
            # computer, which `FeatureBundle` does not catch.
            return self._unavailable(
                view,
                "the newest tick aggregate cannot be read "
                f"({type(error).__name__}: {error}); a non-finite buy_trades or "
                "sell_trades count on that row causes this, and a feed whose "
                "newest row cannot be read is one whose right edge cannot be "
                "verified",
            )
        newest_bar = view.last_bar()
        if newest_tick is None or newest_bar is None:
            return self._unavailable(
                view, "the tick or bar feed has no visible observation"
            )

        method_error = self._classification_method_error(newest_tick)
        if method_error is not None:
            return self._unavailable(view, method_error)

        alignment_error = self._alignment_error(newest_tick, newest_bar)
        if alignment_error is not None:
            return self._unavailable(view, alignment_error)

        notes: list[str] = []
        windows = self._read_windows(view, notes)
        if isinstance(windows, str):
            return self._unavailable(view, windows)

        coverage = self._coverage(windows)
        if coverage < float(self.config.min_classification_coverage):
            return self._unavailable(
                view,
                f"classification coverage {coverage:.4f} is below "
                f"min_classification_coverage "
                f"{self.config.min_classification_coverage:.4f} over the "
                f"{self._tick_rows}-row window: the delta's sign would be set by a "
                "minority of the tape, which is the known-bad estimator section 5 "
                "refuses under another name",
            )

        if not windows.has_unclassified_column:
            notes.append(
                f"{self.name}: the tick feed reports no unclassified_volume column, "
                "so classification_coverage is taken as 1.0 from the feed's own "
                "columns. It is NOT cross-checked against bar volume: bar volume "
                "and tick volume come from different aggregations, and a mismatch "
                "between them is not evidence of unclassified flow"
            )

        delta = self._signed_delta(windows, notes)
        cvd = self._cvd_slope(windows, notes)
        aggression = self._aggression_ratio(windows, notes)
        absorption = self._absorption(windows, notes)
        size = self._trade_size(windows, notes)

        available_weight = (
            (self._weights.signed_delta if delta.available else 0.0)
            + (self._weights.cvd_slope if cvd.available else 0.0)
            + (self._weights.aggression_ratio if aggression.available else 0.0)
            + (self._weights.absorption if absorption.available else 0.0)
            + (self._weights.trade_size_distribution if size.available else 0.0)
        )
        available_fraction = available_weight / self._total_weight
        floor = float(self.config.min_available_weight_fraction)
        if available_fraction < floor:
            dropped = [
                name
                for name, term in (
                    ("signed_delta", delta),
                    ("cvd_slope", cvd),
                    ("aggression_ratio", aggression),
                    ("absorption", absorption),
                    ("trade_size_distribution", size),
                )
                if not term.available
            ]
            return self._unavailable(
                view,
                f"only {available_fraction:.4f} of the term weight has real inputs "
                f"(floor {floor:.4f}); dropped terms: {dropped}. The weight of a "
                "dropped term is never redistributed, so a sub-score built from "
                "what is left would read downstream as 'the tape is balanced' when "
                "it means 'most of this was never measured'",
            )

        # Direction and magnitude, section 7: separate, and the magnitude is
        # the NET of the directional votes so that contradictory evidence
        # produces a small number instead of a confident one.
        directional_sum = (
            (self._weights.signed_delta * delta.signed if delta.available else 0.0)
            + (self._weights.cvd_slope * cvd.signed if cvd.available else 0.0)
            + (
                self._weights.aggression_ratio * aggression.signed
                if aggression.available
                else 0.0
            )
            + (
                self._weights.absorption * absorption.signed
                if absorption.available
                else 0.0
            )
        )
        size_contribution = (
            self._weights.trade_size_distribution * size.strength
            if size.available
            else 0.0
        )
        magnitude = _clamp01(
            (abs(directional_sum) + size_contribution) / self._total_weight
        )
        direction = 0.0
        if directional_sum > 0.0:
            direction = 1.0
        elif directional_sum < 0.0:
            direction = -1.0

        values = {
            "order_flow_magnitude": magnitude,
            "order_flow_direction": direction,
            "order_flow_available": 1.0,
            "order_flow_available_weight": _clamp01(available_fraction),
            "signed_delta": delta.signed,
            "signed_delta_contracts": delta.contracts,
            "cvd_slope": cvd.signed,
            "cvd_slope_normalized": cvd.normalized,
            "aggression_ratio": aggression.ratio,
            "absorption": absorption.strength,
            "absorption_direction": absorption.direction,
            "avg_trade_size_percentile": size.avg_percentile,
            "max_trade_size_percentile": size.max_percentile,
            "large_trade_event": size.large_event,
            "classification_coverage": _clamp01(coverage),
        }
        quality = self._quality_by_key(
            delta=delta, cvd=cvd, aggression=aggression, absorption=absorption, size=size
        )
        return self._vector(view, values, notes=tuple(notes)).replace(
            quality_by_key=quality
        )

    # --- the UNAVAILABLE path ------------------------------------------

    def _unavailable(self, view: MarketView, reason: str) -> FeatureVector:
        """The component disabled: every key 0.0, every key MISSING.

        The one exit this module has when its data is not good enough.
        Section 5 permits degradation for options and liquidity; for order
        flow it permits only disablement, so there is deliberately no variant
        of this method that substitutes an estimate, flags it DEGRADED and
        carries on. The 25 points are UNAVAILABLE and are not redistributed.
        """
        return self._not_ready(
            view,
            f"ORDER FLOW UNAVAILABLE: {reason}. ARCHITECTURE.md section 5 accepts no "
            "proxy for order flow, so the component is disabled and its 25 points "
            "are reported UNAVAILABLE; the weight is NOT redistributed. No "
            "bar-volume delta, tick rule on bars, close-location or quote-imbalance "
            "estimate is substituted",
            quality=DataQuality.MISSING,
        )

    # --- gates ---------------------------------------------------------

    def _classification_method_error(self, aggregate: TickAggregate) -> str | None:
        """Why this feed's aggressor classification is unacceptable, or None.

        The method lives in `TickSeries.meta` and reaches here through
        `TickAggregate.classification_method` on the newest visible row,
        which is the only route `MarketView` offers.
        """
        method = str(aggregate.classification_method).strip().lower()
        if not method or method == UNKNOWN_CLASSIFICATION_METHOD:
            return (
                f"the tick feed declares classification_method="
                f"{aggregate.classification_method!r}, and an undeclared method is "
                "not a classification -- volume with an unexplained side is bar "
                "volume with a label"
            )
        if method in self._rejected_methods:
            return (
                f"the tick feed declares classification_method="
                f"{aggregate.classification_method!r}, which is on "
                "OrderFlowConfig.rejected_classification_methods: it is a name a "
                "bar-volume-derived delta ships under"
            )
        return None

    def _alignment_error(self, aggregate: TickAggregate, bar: Bar) -> str | None:
        """Why the tick and bar windows do not line up at the right edge.

        Only the right edge is checked; the module docstring states what that
        does and does not establish, and why a per-row check is not available
        at an acceptable cost.
        """
        if aggregate.close_ts != bar.close_ts:
            return (
                f"the newest tick aggregate closes at {aggregate.close_ts.isoformat()} "
                f"but the newest bar closes at {bar.close_ts.isoformat()}; the "
                "absorption term would compare a delta window with prices from a "
                "different stretch of tape"
            )
        return None

    # --- windows -------------------------------------------------------

    def _read_windows(self, view: MarketView, notes: list[str]) -> "_Windows | str":
        """Every array this computer reads, or a reason it cannot proceed.

        Returns a string rather than raising so that a short or dirty window
        becomes a reported UNAVAILABLE rather than an exception out of a
        feature computer. A REQUIRED input that is short or dirty is a
        reason to disable the component; an OPTIONAL column that is present
        but unusable is treated as not supplied, and the note says which of
        the two happened. Optional tick columns are returned as empty arrays
        when the feed does not carry them, and an empty array means NOT
        SUPPLIED -- never zero. That distinction is why `aggression_ratio`
        and the trade-size terms can be dropped rather than silently read as
        a tape with no trades in it.
        """
        rows = self._tick_rows
        deltas = view.deltas(rows)
        buys = view.tick_column("buy_volume", rows)
        sells = view.tick_column("sell_volume", rows)
        if not (deltas.size == buys.size == sells.size == rows):
            return (
                f"short tick window: delta={deltas.size} buy={buys.size} "
                f"sell={sells.size}, expected {rows} each"
            )
        for label, values in (("buy_volume", buys), ("sell_volume", sells)):
            if not np.all(np.isfinite(values)):
                return f"the {label} window is not finite"
            if np.any(values < 0.0):
                # Unreachable through a legally constructed `TickSeries`, whose
                # `NON_NEGATIVE` covers both volume columns. Kept as a guard
                # against a data-layer change, and named in the test file's list
                # of branches that cannot be exercised from outside.
                return f"the {label} window contains a negative volume"
        # Every per-row value being finite does not make their SUMS finite, and
        # the smoothed delta, the CVD cumulation, the absorption window sums and
        # the pooled trade size are all sums over this window. Volumes are
        # non-negative, so the total bounds every sub-window sum: one check here
        # covers all of them. Without it a feed carrying ~1e308 lots overflows
        # to an infinite `signed_delta_contracts`, and the base class's
        # finiteness guard turns that into a `FeatureError` out of `compute`
        # instead of the reported refusal this module promises.
        total_classified = float(buys.sum()) + float(sells.sum())
        if not math.isfinite(total_classified):
            return (
                "the classified-volume window sums to a value outside double "
                "precision, so every rolling sum, mean and cumulation built from "
                "it would be infinite rather than a measurement"
            )

        unclassified = view.tick_column("unclassified_volume", rows)
        has_unclassified = unclassified.size == rows
        if unclassified.size not in (0, rows):
            return (
                f"short unclassified_volume window: {unclassified.size} of {rows}"
            )
        if has_unclassified and (
            not np.all(np.isfinite(unclassified)) or np.any(unclassified < 0.0)
        ):
            # As above, the negative half is unreachable through a legal
            # `TickSeries`; the non-finite half is not, because `NON_NEGATIVE`
            # compares with `nanmin` and lets a NaN through.
            return "the unclassified_volume window is not finite or is negative"
        if has_unclassified and not math.isfinite(float(unclassified.sum())):
            return (
                "the unclassified_volume window sums to a value outside double "
                "precision, so the classification coverage cannot be measured"
            )

        buy_trades = view.tick_column("buy_trades", rows)
        sell_trades = view.tick_column("sell_trades", rows)
        has_trades = buy_trades.size == rows and sell_trades.size == rows
        if has_trades and not (
            np.all(np.isfinite(buy_trades))
            and np.all(np.isfinite(sell_trades))
            and np.all(buy_trades >= 0.0)
            and np.all(sell_trades >= 0.0)
        ):
            # `TickSeries.NON_NEGATIVE` covers the volume columns and not the
            # counts, so a negative or non-finite count reaches here. It is
            # treated as NOT SUPPLIED and said so, because the alternative
            # notes above ("the feed carries no counts") would be the wrong
            # explanation for a column that exists and is broken.
            notes.append(
                f"{self.name}: the buy_trades/sell_trades columns are present but "
                "contain a negative or non-finite count over the read window, so "
                "they are treated as NOT SUPPLIED"
            )
            has_trades = False

        max_trade = view.tick_column("max_trade_size", rows)
        has_max_trade = max_trade.size == rows
        if has_max_trade and not (
            np.all(np.isfinite(max_trade)) and np.all(max_trade >= 0.0)
        ):
            notes.append(
                f"{self.name}: the max_trade_size column is present but contains a "
                "negative or non-finite size over the read window, so it is treated "
                "as NOT SUPPLIED"
            )
            has_max_trade = False

        bar_rows = self._bar_rows
        highs = view.column("high", bar_rows)
        lows = view.column("low", bar_rows)
        closes = view.column("close", bar_rows)
        if not (highs.size == lows.size == closes.size == bar_rows):
            return (
                f"short bar window: high={highs.size} low={lows.size} "
                f"close={closes.size}, expected {bar_rows} each"
            )
        for label, prices in (("high", highs), ("low", lows), ("close", closes)):
            if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
                return f"the {label} window contains a non-positive or non-finite price"

        return _Windows(
            deltas=deltas,
            buys=buys,
            sells=sells,
            unclassified=unclassified if has_unclassified else None,
            buy_trades=buy_trades if has_trades else None,
            sell_trades=sell_trades if has_trades else None,
            max_trade=max_trade if has_max_trade else None,
            highs=highs,
            lows=lows,
            closes=closes,
        )

    def _coverage(self, windows: "_Windows") -> float:
        """Share of window volume carrying a known aggressor, in [0, 1].

        Measured over the WHOLE read window rather than the newest bar,
        because every term's comparison set spans that window and a
        percentile computed inside it is only as trustworthy as the window's
        classification. The cost of that choice is stated: a collapse in
        classification confined to the last few bars moves this number by
        only a few percent, so it is a check on the comparison set and not an
        alarm on the current bar.
        """
        classified = float(windows.classified_volume.sum())
        if windows.unclassified is None:
            # The column is absent: the feed reports no unclassified volume.
            # Noted by the caller; see `_read_windows` on absence vs zero.
            return 1.0 if classified > 0.0 else 0.0
        total = classified + float(windows.unclassified.sum())
        return safe_divide(classified, total, 0.0)

    # --- the five terms ------------------------------------------------

    def _signed_delta(self, windows: "_Windows", notes: list[str]) -> "_DeltaTerm":
        """`signed_delta` and `signed_delta_contracts`.

        Always available once the component's gates have passed: it reads
        only the delta array, which the tick feed's required columns
        guarantee.
        """
        deltas = windows.deltas[-self._delta_chain :]
        smoothed = rolling_mean(deltas, self._smoothing)
        window = smoothed[-self._delta_lookback :]
        current = float(window[-1])
        if is_degenerate(window):
            notes.append(
                f"{self.name}: every one of the {window.size} smoothed delta "
                f"observations equals {current:.6g}, so there is no distribution to "
                "rank against; signed_delta reported as 0.0 (balanced). A "
                "tie-inclusive percentile rank would have reported +1.0 here, which "
                "is the inverse of the truth"
            )
            return _DeltaTerm(signed=0.0, contracts=current, available=True, measured=False)
        return _DeltaTerm(
            signed=_clamp(signed_rank(window, current), -1.0, 1.0),
            contracts=current,
            available=True,
            measured=True,
        )

    def _cvd_slope(self, windows: "_Windows", notes: list[str]) -> "_CvdTerm":
        """`cvd_slope` and `cvd_slope_normalized`.

        Dropped when the window's mean classified volume is zero: the slope
        would then have no scale to be expressed in, and reporting the raw
        contracts-per-bar slope instead would be a number that means
        something different on every instrument.
        """
        deltas = windows.deltas[-self._cvd_lookback :]
        classified = windows.classified_volume[-self._cvd_lookback :]
        mean_classified = float(classified.mean())
        slope = least_squares_slope(np.cumsum(deltas))
        if mean_classified <= 0.0:
            notes.append(
                f"{self.name}: no classified volume in the {self._cvd_lookback}-bar "
                "CVD window, so the slope has no scale to be made dimensionless "
                "with; the cvd_slope term is DROPPED and its weight "
                f"({self._weights.cvd_slope:g}) is not redistributed"
            )
            return _CvdTerm(signed=0.0, normalized=0.0, available=False)
        normalized = safe_divide(slope, mean_classified, 0.0)
        if not math.isfinite(normalized):
            notes.append(
                f"{self.name}: the normalized CVD slope is not finite; the "
                "cvd_slope term is DROPPED"
            )
            return _CvdTerm(signed=0.0, normalized=0.0, available=False)
        return _CvdTerm(
            signed=_clamp(
                signed_squash(normalized, float(self.config.cvd_slope_squash_scale)),
                -1.0,
                1.0,
            ),
            normalized=normalized,
            available=True,
        )

    def _aggression_ratio(
        self, windows: "_Windows", notes: list[str]
    ) -> "_AggressionTerm":
        """`aggression_ratio`, pooled over `aggression_lookback_bars`.

        Dropped -- not neutralised -- when the feed carries no trade counts
        or when the pool holds fewer than `min_classified_trades`.
        """
        if windows.buy_trades is None or windows.sell_trades is None:
            notes.append(
                f"{self.name}: the tick feed carries no buy_trades/sell_trades "
                "columns, so a count-based aggression ratio was never measured; the "
                "term is DROPPED and its weight "
                f"({self._weights.aggression_ratio:g}) is not redistributed. "
                "Deriving it from volume instead would silently replace a count "
                "ratio with TickAggregate.delta_ratio, which is a different "
                "measurement"
            )
            return _AggressionTerm(
                ratio=NEUTRAL_RATIO, signed=0.0, trades=0.0, available=False
            )

        window = self._aggression_lookback
        buy = float(windows.buy_trades[-window:].sum())
        sell = float(windows.sell_trades[-window:].sum())
        classified = buy + sell
        minimum = float(self.config.min_classified_trades)
        if classified < minimum:
            notes.append(
                f"{self.name}: {classified:.0f} classified trades pooled over "
                f"{window} bars is below min_classified_trades {minimum:.0f}; the "
                "aggression_ratio term is DROPPED, not neutralised -- a 3-trade 2:1 "
                "ratio emitting 0.67 is indistinguishable downstream from a real "
                "two-thirds reading. Its weight "
                f"({self._weights.aggression_ratio:g}) is not redistributed"
            )
            return _AggressionTerm(
                ratio=NEUTRAL_RATIO, signed=0.0, trades=classified, available=False
            )
        ratio = _clamp01(safe_divide(buy, classified, NEUTRAL_RATIO))
        return _AggressionTerm(
            ratio=ratio,
            signed=_clamp(2.0 * ratio - 1.0, -1.0, 1.0),
            trades=classified,
            available=True,
        )

    def _absorption(self, windows: "_Windows", notes: list[str]) -> "_AbsorptionTerm":
        """`absorption` and `absorption_direction`.

        Effort versus result over `absorption_lookback_bars`; see the module
        docstring for the three terms and the geometric mean. Dropped when
        the local ATR is zero or non-finite, which is the only input that can
        be missing: a window with no volatility scale has no ATR-relative
        geometry, and a frozen tape is not an absorption reading.
        """
        window = self._absorption_lookback
        sums = rolling_sum(
            windows.deltas[-self._absorption_chain :], window
        )[-self._delta_lookback :]
        current_sum = float(sums[-1])

        atr = self._local_atr(windows)
        if atr is None:
            notes.append(
                f"{self.name}: the local ATR over {self._atr_period} true ranges is "
                "zero or non-finite, so displacement and the level band have no "
                "scale; the absorption term is DROPPED and its weight "
                f"({self._weights.absorption:g}) is not redistributed"
            )
            return _AbsorptionTerm(
                strength=0.0,
                direction=0.0,
                signed=0.0,
                available=False,
                effort=0.0,
                quiet=0.0,
                flat=0.0,
            )

        magnitudes = np.abs(sums)
        if current_sum == 0.0 or is_degenerate(magnitudes):
            # Absorption is about HEAVY ONE-SIDED flow. A window whose delta
            # sums to exactly zero has none to absorb, and a window whose
            # |sums| are all identical has no distribution to be extreme of
            # -- on a perfectly balanced tape every sum is 0.0 and a
            # tie-inclusive percentile rank would return 1.0, reporting
            # maximum one-sided flow on a tape with none. Both cases report
            # no effort, which sends the geometric mean to zero.
            effort = 0.0
        else:
            effort = excess_above(
                percentile_rank(magnitudes, abs(current_sum)),
                float(self.config.absorption_delta_percentile),
            )
        closes = windows.closes
        displacement_atr = abs(float(closes[-1]) - float(closes[-1 - window])) / atr
        band_atr = (
            float(closes[-window:].max()) - float(closes[-window:].min())
        ) / atr
        quiet = inside_band(
            displacement_atr, float(self.config.absorption_max_displacement_atr)
        )
        flat = inside_band(
            band_atr, float(self.config.absorption_level_window_atr)
        )
        strength = _clamp01(
            math.pow(effort * quiet * flat, 1.0 / ABSORPTION_TERM_COUNT)
        )

        direction = 0.0
        if strength > 0.0 and current_sum != 0.0:
            # Against the aggressor: absorbed buying votes bearish. A DECLARED
            # PRIOR, not a fact -- see the module docstring.
            direction = -1.0 if current_sum > 0.0 else 1.0
        return _AbsorptionTerm(
            strength=strength,
            direction=direction,
            signed=_clamp(direction * strength, -1.0, 1.0),
            available=True,
            effort=effort,
            quiet=quiet,
            flat=flat,
        )

    def _local_atr(self, windows: "_Windows") -> float | None:
        """The Wilder seed over the last `atr_period` true ranges, or None.

        A LOCAL scale for the absorption geometry. Deliberately not
        `VolatilityFeatures`' `atr`: reading another computer's output is the
        dependency that would break per-computer lookahead auditing, and the
        two values differ anyway because Wilder's recursion is anchored at
        the start of whichever window computed it (volatility.py documents
        this). `true_range` and `wilder_atr_series` are imported from that
        module so the true-range convention has one definition.

        None when the ATR is zero or non-finite, which the caller reports as
        a dropped term rather than dividing by.
        """
        rows = self._atr_period + 1
        ranges = true_range(
            windows.highs[-rows:], windows.lows[-rows:], windows.closes[-rows:]
        )
        series = wilder_atr_series(ranges, self._atr_period)
        atr = float(series[-1])
        if not math.isfinite(atr) or atr <= 0.0:
            return None
        return atr

    def _trade_size(self, windows: "_Windows", notes: list[str]) -> "_SizeTerm":
        """The trade-size distribution: two percentiles and an event flag.

        Each moment is dropped independently. The term as a whole is
        available while at least one of the two is, because "the distribution
        is unusually heavy" is answerable from either moment alone; it is
        dropped when the feed supplies neither.
        """
        pool = self._aggression_lookback
        rows = self._size_chain
        ranks: list[float] = []

        max_percentile = UNAVAILABLE_PERCENTILE
        large_event = 0.0
        max_measured = False
        if windows.max_trade is None:
            notes.append(
                f"{self.name}: the tick feed carries no max_trade_size column, so "
                "the tail of the trade-size distribution was never measured; "
                "max_trade_size_percentile is reported as "
                f"{UNAVAILABLE_PERCENTILE} (unavailable, not median)"
            )
        else:
            pooled = rolling_max(windows.max_trade[-rows:], pool)[
                -self._size_lookback :
            ]
            if is_degenerate(pooled):
                notes.append(
                    f"{self.name}: every one of the {pooled.size} pooled maximum "
                    "trade sizes is identical, so there is no distribution to rank "
                    f"against; max_trade_size_percentile reported as "
                    f"{UNAVAILABLE_PERCENTILE} (unavailable). A tie-inclusive "
                    "percentile rank would have reported 1.0 and fired "
                    "large_trade_event on a tape with one print size"
                )
            else:
                max_percentile = _clamp01(
                    percentile_rank(pooled, float(pooled[-1]))
                )
                max_measured = True
                ranks.append(max_percentile)
                if max_percentile >= float(self.config.large_trade_percentile):
                    large_event = 1.0

        avg_percentile = UNAVAILABLE_PERCENTILE
        avg_measured = False
        if windows.buy_trades is None or windows.sell_trades is None:
            notes.append(
                f"{self.name}: the tick feed carries no trade counts, so an average "
                "trade size cannot be formed; avg_trade_size_percentile is reported "
                f"as {UNAVAILABLE_PERCENTILE} (unavailable, not median)"
            )
        else:
            volume_sums = rolling_sum(windows.classified_volume[-rows:], pool)
            trade_sums = rolling_sum(windows.classified_trades[-rows:], pool)
            usable = trade_sums > 0.0
            averages = np.where(
                usable, volume_sums / np.where(usable, trade_sums, 1.0), np.nan
            )[-self._size_lookback :]
            if not math.isfinite(float(averages[-1])):
                notes.append(
                    f"{self.name}: no classified trades in the newest {pool}-bar "
                    "pool, so the current average trade size is undefined; "
                    f"avg_trade_size_percentile is reported as "
                    f"{UNAVAILABLE_PERCENTILE} (unavailable)"
                )
            elif is_degenerate(averages):
                notes.append(
                    f"{self.name}: every pooled average trade size in the window is "
                    "identical, so there is no distribution to rank against; "
                    f"avg_trade_size_percentile reported as "
                    f"{UNAVAILABLE_PERCENTILE} (unavailable)"
                )
            else:
                avg_percentile = _clamp01(
                    percentile_rank(averages, float(averages[-1]))
                )
                avg_measured = True
                ranks.append(avg_percentile)

        if not ranks:
            notes.append(
                f"{self.name}: neither moment of the trade-size distribution is "
                "available; the term is DROPPED and its weight "
                f"({self._weights.trade_size_distribution:g}) is not redistributed"
            )
            return _SizeTerm(
                strength=0.0,
                avg_percentile=avg_percentile,
                max_percentile=max_percentile,
                large_event=large_event,
                avg_measured=False,
                max_measured=False,
            )

        strength = _clamp01(2.0 * (sum(ranks) / len(ranks)) - 1.0)
        return _SizeTerm(
            strength=strength,
            avg_percentile=avg_percentile,
            max_percentile=max_percentile,
            large_event=large_event,
            avg_measured=avg_measured,
            max_measured=max_measured,
        )

    # --- quality -------------------------------------------------------

    def _quality_by_key(
        self,
        *,
        delta: "_DeltaTerm",
        cvd: "_CvdTerm",
        aggression: "_AggressionTerm",
        absorption: "_AbsorptionTerm",
        size: "_SizeTerm",
    ) -> dict[str, DataQuality]:
        """Per-key quality, so a caller can tell which terms were measured.

        DEGRADED here has exactly one meaning: **a named term was dropped and
        its weight was not redistributed.** It never means an estimate was
        substituted -- there is no substitution anywhere in this module. When
        the component as a whole cannot be measured the vector comes from
        `_unavailable` and every key is MISSING instead, which is the
        distinction section 5 draws between options/liquidity degradation and
        order-flow disablement.
        """
        good, degraded = DataQuality.GOOD, DataQuality.DEGRADED
        all_terms = (
            delta.available
            and cvd.available
            and aggression.available
            and absorption.available
            and size.available
        )
        composite = good if all_terms else degraded
        quality = {
            "order_flow_magnitude": composite,
            "order_flow_direction": composite,
            # Availability itself is always a measurement: it is the report
            # that something else was not measured.
            "order_flow_available": good,
            "order_flow_available_weight": good,
            "classification_coverage": good,
            "signed_delta": good if delta.available else degraded,
            "signed_delta_contracts": good if delta.available else degraded,
            "cvd_slope": good if cvd.available else degraded,
            "cvd_slope_normalized": good if cvd.available else degraded,
            "aggression_ratio": good if aggression.available else degraded,
            "absorption": good if absorption.available else degraded,
            "absorption_direction": good if absorption.available else degraded,
            # Graded per moment, not per term: a measured average is a
            # measurement even when the tail was never observed.
            "avg_trade_size_percentile": good if size.avg_measured else degraded,
            "max_trade_size_percentile": good if size.max_measured else degraded,
            # Derived from the tail rank alone, so it degrades with it: 0.0
            # from an unmeasured tail means "not measured", not "no large
            # print".
            "large_trade_event": good if size.max_measured else degraded,
        }
        missing = set(self.keys) - set(quality)
        if missing:  # pragma: no cover - guarded by a key/quality parity test
            raise FeatureError(f"{self.name}: no quality declared for {sorted(missing)}")
        return quality


# ---------------------------------------------------------------------------
# term results and read windows
# ---------------------------------------------------------------------------
#
# Plain slotted classes rather than tuples, so that a term's availability
# travels with its value and cannot be read without it. `available=False`
# means the term was DROPPED: its value is a stated placeholder, its weight
# leaves the magnitude, and nothing reallocates it.
#
# Each also carries the detail behind its term -- the three absorption
# factors, the pooled trade count, whether a window was degenerate -- which
# `compute` does not emit as features. That is deliberate: those numbers
# diagnose a term and a weight sweep reads them, but adding a key per factor
# would widen the contract with values no caller could act on. They are
# reachable from a test, which is where they are asserted.


class _Windows:
    """Every array one `compute` call reads, with absence preserved.

    An optional tick column that the feed does not carry is `None`, never an
    array of zeros. `TickAggregate` draws the same distinction for its
    optional fields and `OptionsSnapshot` documents why: collapsing absence
    into 0.0 feeds a feature a measurement that was never made.
    """

    __slots__ = (
        "deltas",
        "buys",
        "sells",
        "unclassified",
        "buy_trades",
        "sell_trades",
        "max_trade",
        "highs",
        "lows",
        "closes",
        "classified_volume",
        "classified_trades",
    )

    def __init__(
        self,
        *,
        deltas: np.ndarray,
        buys: np.ndarray,
        sells: np.ndarray,
        unclassified: np.ndarray | None,
        buy_trades: np.ndarray | None,
        sell_trades: np.ndarray | None,
        max_trade: np.ndarray | None,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> None:
        self.deltas = deltas
        self.buys = buys
        self.sells = sells
        self.unclassified = unclassified
        self.buy_trades = buy_trades
        self.sell_trades = sell_trades
        self.max_trade = max_trade
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.classified_volume = buys + sells
        self.classified_trades = (
            buy_trades + sell_trades
            if buy_trades is not None and sell_trades is not None
            else None
        )

    @property
    def has_unclassified_column(self) -> bool:
        return self.unclassified is not None


class _DeltaTerm:
    """`signed_delta` in [-1, 1] and the smoothed delta in contracts."""

    __slots__ = ("signed", "contracts", "available", "measured")

    def __init__(
        self, *, signed: float, contracts: float, available: bool, measured: bool
    ) -> None:
        self.signed = signed
        self.contracts = contracts
        self.available = available
        #: False when the window was degenerate and 0.0 is a stated reading
        #: rather than a rank. The term is still available: a balanced tape
        #: is a measurement.
        self.measured = measured


class _CvdTerm:
    """`cvd_slope` in [-1, 1] and the dimensionless slope behind it."""

    __slots__ = ("signed", "normalized", "available")

    def __init__(self, *, signed: float, normalized: float, available: bool) -> None:
        self.signed = signed
        self.normalized = normalized
        self.available = available


class _AggressionTerm:
    """`aggression_ratio` in [0, 1] and the pooled classified trade count."""

    __slots__ = ("ratio", "signed", "trades", "available")

    def __init__(
        self, *, ratio: float, signed: float, trades: float, available: bool
    ) -> None:
        self.ratio = ratio
        self.signed = signed
        self.trades = trades
        self.available = available


class _AbsorptionTerm:
    """`absorption` in [0, 1], its direction, and the three terms behind it."""

    __slots__ = ("strength", "direction", "signed", "available", "effort", "quiet", "flat")

    def __init__(
        self,
        *,
        strength: float,
        direction: float,
        signed: float,
        available: bool,
        effort: float,
        quiet: float,
        flat: float,
    ) -> None:
        self.strength = strength
        self.direction = direction
        self.signed = signed
        self.available = available
        #: The three factors, kept for diagnosis and for the Phase 8 sweep.
        #: They are not emitted as features: three more keys to describe one
        #: term would widen the contract without adding a measurement a
        #: caller could act on.
        self.effort = effort
        self.quiet = quiet
        self.flat = flat


class _SizeTerm:
    """The trade-size distribution: a strength in [0, 1] and two percentiles.

    The two moments are tracked SEPARATELY rather than counted, because they
    fail separately: a feed can carry `max_trade_size` and no trade counts,
    or a tape can have one print size (a degenerate tail) and a perfectly
    good average. Grading both keys by a count would report a measured
    percentile as DEGRADED because the other one was not measured.
    """

    __slots__ = (
        "strength",
        "avg_percentile",
        "max_percentile",
        "large_event",
        "avg_measured",
        "max_measured",
    )

    def __init__(
        self,
        *,
        strength: float,
        avg_percentile: float,
        max_percentile: float,
        large_event: float,
        avg_measured: bool,
        max_measured: bool,
    ) -> None:
        self.strength = strength
        self.avg_percentile = avg_percentile
        self.max_percentile = max_percentile
        self.large_event = large_event
        self.avg_measured = avg_measured
        self.max_measured = max_measured

    @property
    def available(self) -> bool:
        """Whether the term enters the magnitude at all.

        One moment is enough: "the distribution is unusually heavy" is
        answerable from either the central or the tail rank alone, and the
        key for the unmeasured one reports `UNAVAILABLE_PERCENTILE` with
        DEGRADED quality.
        """
        return self.avg_measured or self.max_measured


def _clamp(value: float, low: float, high: float) -> float:
    return low if value <= low else (high if value >= high else float(value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)
