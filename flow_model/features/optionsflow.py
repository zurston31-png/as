"""Options-flow features: the five chain-level imbalance and positioning
measurements ARCHITECTURE.md section 5 names, and the bounded
(magnitude, direction) pair the OPTIONS_FLOW component reads.

Section 5 gives this component 20 of the 100 points, names its features as
"net premium (call-prem minus put-prem), delta-weighted volume, OI change,
25d skew, gamma-exposure proxy", names OPRA trade prints as the ideal feed
and an end-of-day chain plus open interest as the degraded one, and says of
the degraded path:

    "Degrades to DEGRADED with daily granularity; OptionsFlow sub-score
     capped and flagged. Never synthesized."

Those two sentences, not the formulas, are the organizing principle of this
module, so they are stated precisely before anything else.

## What this component measures, and what it does NOT claim

Section 5's "explicitly rejected interpretations" paragraph is binding:

    "options flow is treated as a measurable order-imbalance and positioning
     signal, not as evidence of institutional intent. Large premium prints
     have ambiguous sign (they may be hedges, spreads, or closing trades);
     the feature set therefore measures imbalance and realized
     dealer-hedging pressure proxies, with no claim about who traded or
     why."

So every number here answers one question: **how unbalanced is this chain
between its call side and its put side, and how large is that imbalance
against this same chain's own recent history?** Nothing here answers *who*
traded, *why*, or whether the imbalance was opened or closed. In particular:

* A call-heavy premium imbalance is a call-heavy premium imbalance. It is
  not a bullish bet, a "sweep", conviction, or an institution. The same
  print is produced by a customer buying calls to open, a customer selling
  calls to close, a dealer hedging a short put, and one leg of a spread
  whose other leg is in a different expiry. The measurement cannot separate
  them and this module does not pretend to.
* `options_flow_direction` is the sign of the **measured call-side-versus-
  put-side imbalance**, carried separately from the magnitude because
  section 7 requires it ("this avoids the common bug where a strong bearish
  component inflates a bullish score"). That a call-side imbalance should
  precede an up move is a *hypothesis*, testable only against the
  pre-registered criteria of section 15. It is not asserted anywhere in
  this module and no result here supports it.
* `options_gamma_exposure_pressure` is a **proxy**. Dealer gamma inventory
  is not observable by anyone outside a dealer. `OptionsSnapshot` supplies
  an OI-and-price construction under that name, and this module reports how
  large it is relative to its own recent history. Its *sign* depends on
  which side of the chain dealers are on, which is exactly the unobservable
  part, so the sign is never read: the term is a magnitude only and casts no
  directional vote.

## The feed, and what happens without it

`required_feeds` is `{Feed.OPTIONS_SNAPSHOT}` and nothing else. **No bar
column is read anywhere in this module** -- not a close, not a volume, not a
timestamp -- so `Feed.BARS` would be a requirement this computer does not
have, and declaring it would make the component unavailable on a dataset it
could have scored.

Three distinct outcomes, each reported rather than papered over:

1. **No options feed at all.** `FeatureBundle` sees the absent required feed
   and emits a vector with `warmup_complete=False` and
   `DataQuality.MISSING`, noting `required feed(s) absent:
   ['options_snapshot']`. The component is disabled and its 20 points are
   UNAVAILABLE. Nothing is estimated from bars, from the underlying, or from
   anything else: there is no bar-derived proxy for an options chain, and a
   fabricated one would be indistinguishable downstream from a measured one.
2. **Feed present, fewer than `warmup_snapshots` snapshots visible.** The
   percentile windows are not full, so `compute` returns `_not_ready` with
   the observed and required counts. A half-filled window produces a number
   that looks like a feature and is not one.
3. **Feed present and warm, but too few of its optional fields supplied.**
   See "Availability" below: the component reports UNAVAILABLE rather than a
   fraction-strength opinion.

## Degradation: the EOD path, which is the main deliverable of Phase 5

`OptionsSnapshot.is_intraday` is the measurement that decides this, through
`MarketView.options_are_intraday()`. It is a property of the data, never a
config guess about a vendor.

What an end-of-day chain can and cannot express is specific. It carries the
session's aggregate premium, volume, open interest, implied vols and the
gamma proxy -- so all five terms are computable from it, and reporting them
as unavailable would throw away real measurements. What it cannot carry is
**flow timing**: whether the imbalance happened in the first ten minutes or
the last, and therefore whether it is current at the instant a signal is
taken. The reading describes a session that has already closed.

So on the EOD path this module does exactly four things:

* **Flags it.** `options_flow_timing_available` is 0.0, and the two trade-flow
  terms (`options_net_premium_imbalance`,
  `options_delta_volume_imbalance`) carry `DataQuality.DEGRADED` with a note
  naming the fallback: each is a *whole-session aggregate* standing in for
  flow within the current bar, which is what the feed supplies and not what
  the feature wants. The positioning and price terms (OI change, skew, the
  gamma proxy) are DEGRADED too, because a positioning reading taken at a
  session boundary is also not a reading at `view.now`.
* **Caps the sub-score.** `options_flow_magnitude` is capped at
  `config.eod_degraded_cap_fraction` (default 0.50). The magnitude is
  already expressed as a fraction of the component's weight, so the
  configured *fraction* is the cap directly, and the cap keeps binding when
  the Phase 8 sweep changes the component's point allocation.
* **Shows the cap binding.** `options_flow_uncapped_magnitude` reports what
  the magnitude would have been, and `options_eod_capped` is 1.0 on exactly
  those bars where the cap changed the answer. A cap that is never reached
  is not a cap; these two keys make that checkable at runtime instead of
  being asserted in a docstring.
* **Never synthesizes.** No intraday path is reconstructed, interpolated or
  back-filled from the daily aggregate. `config.never_synthesize_missing_fields`
  is refused if set False and this module has no code that would need it.

`config.require_intraday_prints` (default False) is the stricter research
option: True disables the component outright on an EOD chain instead of
capping it, which makes the cap moot. False is what section 5 specifies.

This module does **not** grade staleness. `data/quality.py` already does,
against each feed's own observed cadence, with
`max_staleness_seconds_by_feed[options_snapshot] = 345_600s` (four days)
precisely so that an EOD chain -- a session old by construction -- is not
graded STALE for being what it is. `options_snapshot_age_seconds` is
reported here as a diagnostic so a consumer can see the age; re-deriving
the grade would be a second, divergent source of truth.

## The window, and why `warmup_bars` is a floor rather than the real gate

Every term is compared against the last `config.warmup_snapshots` snapshots
(21 on defaults: `lookback_snapshots` predecessors plus the current one).
The current snapshot is a member of its own comparison set, as the current
bar is for `atr_percentile` in `features/volatility.py` and
`volume_percentile` in `features/liquidity.py`; with 21 samples its
contribution to the comparison is immaterial, and excluding it would make
the current observation the only one ranked against a different window.

The window is counted in **snapshots**, which is the unit
`OptionsFlowConfig` deliberately uses: an EOD chain gives roughly one
snapshot per session and an intraday feed gives many, so a window in days
or bars would assert a cadence this layer cannot observe.

`FeatureComputer.warmup_bars` is nevertheless declared in bars, so it is
declared here as a **lower bound**: `warmup_snapshots` bars. The bound rests
on this system modelling at most one chain snapshot per bar, and that is a
CONVENTION rather than an invariant, so it is stated as one.
`QualityGrader._presence_coverage` in `data/quality.py` encodes the
convention -- it grades an intraday chain as "expected once per bar" and
`round(min(options_count / bars, 1.0), 6)` caps its coverage at 1.0 -- but
capping is not rejecting: nothing in the data layer refuses a chain carrying
two snapshots per bar, and on such a chain this computer reports
`warmup_complete=True` at bar index 10 while declaring 21, which is a
finding the lookahead audit's check 3 would raise. No such dataset exists in
the system and `FeatureBundle` gates on bars as well, so the direction is
safe; it is recorded here rather than claimed away, because "provable" would
be the wrong word for a convention.

In the direction that matters for a half-filled window the declaration never
over-claims: it never reports readiness before `warmup_snapshots` snapshots
exist, and it never over-states a bar dependency this module does not have.
It does UNDER-state the bars an EOD chain needs -- one snapshot per session
means 21 snapshots take roughly 21 sessions, which is some sixteen hundred
5-minute bars -- and the snapshot gate below, not this number, is what
refuses to score until then.

It is a floor and not a substitute. `compute` additionally checks
`view.options_count()` against `warmup_snapshots` on every call and reports
`_not_ready` when it is short, exactly as `features/liquidity.py` checks its
quote sample count separately from its bar warmup. The honest consequence,
reported rather than hidden: the lookahead harness's warmup-honesty check
(check 3 of `validation/lookahead.py`) tests the *bar* declaration, so for
this computer it is a weak test, and the snapshot gate has to be tested
directly.

## The five terms

Each term produces one bounded number plus an availability flag. A term
whose input is `None` in `OptionsSnapshot` is **dropped**, and its weight is
**not redistributed** -- the rule section 14.5 states for `s_flow`, applied
to the whole component. `None` means NOT SUPPLIED and `0.0` is a real
observation, so a dropped term and a zero term must not produce the same
sub-score; redistributing would convert missing data into confidence the
chain does not support.

Signs are all stated the same way: **positive means the call side is
heavier**, so the four signed terms are directly comparable.

### 1. Net premium -- `options_net_premium_imbalance`, in [-1, 1]

        (call_premium - put_premium) / (call_premium + put_premium)

A dimensionless imbalance rather than the dollar difference, so it is
comparable across underlyings and across a decade of premium inflation, and
bounded by construction rather than by a clamp. `options_net_premium` is
emitted beside it in dollars as a diagnostic.

Computed from the raw fields rather than through
`OptionsSnapshot.premium_imbalance`, which returns `0.0` when total premium
is zero. Zero is the reading for a perfectly balanced chain, so it cannot
also be the reading for a chain with no premium at all. A zero denominator
drops the term instead.

### 2. Delta-weighted volume -- `options_delta_volume_imbalance`, in [-1, 1]

        (|dw_call_volume| - |dw_put_volume|)
            / (|dw_call_volume| + |dw_put_volume|)

Delta-weighted volume is the closest thing in the contract to "how much
directional exposure changed hands", in delta units rather than contracts.

The absolute values are deliberate and are not cosmetic. Put delta is
negative under the standard convention, and the contract does not say
whether a provider signs `delta_weighted_put_volume` by it. If a provider
does, `dwcv - dwpv` would add the two sides instead of differencing them
and the denominator would approach zero -- producing a wild imbalance from
ordinary data. Defining the term on magnitudes makes it invariant to that
convention, at the price of ignoring a sign the provider may have supplied;
the invariance is worth more than the sign, because a convention mismatch
here is silent and catastrophic while the lost sign is recoverable from the
call/put split that remains.

### 3. OI change -- `options_oi_change_imbalance`, in [-1, 1]

        net      = call_oi_change - put_oi_change
        fraction = net / (call_oi + put_oi)
        term     = sign(fraction) * squash(|fraction|, oi_change_squash_scale)

A fraction of total open interest rather than a contract count, so it is
comparable across underlyings and across the growth of a chain over years.
`squash` (a `tanh`) rather than a z-score, so one extreme session cannot
dominate the weighted sum.

**`call_oi_change` is never reconstructed by differencing `call_oi`.** The
levels are in the window and the arithmetic is trivial, which is exactly why
this needs saying: a differenced level would be indistinguishable
downstream from a supplied `call_oi_change`, it assumes the two snapshots
are adjacent sessions when a gap in the feed means they are not, and it
erases the `None`-versus-`0.0` distinction `OptionsSnapshot` exists to
preserve. `config.never_synthesize_missing_fields` and the brief's "never
invent missing options data" both forbid it. The term is dropped instead.

### 4. 25-delta skew -- `options_skew_25d_pressure`, in [-1, 1]

        skew = iv_25d_put - iv_25d_call                  (the diagnostic)
        p    = percentile_rank(skew over the window, skew now)
        term = 2p - 1

The *level* of index skew is structurally positive and says little about the
current state -- the synthetic generator's own base skew is 0.12 before any
signal is applied. Its position within its own recent distribution is the
measurement. `2p - 1` maps the rank onto [-1, 1] with 0 at the window
median, which keeps the term on the same scale as the other three.

Sign: `skew_25d` positive means the 25-delta put is bid relative to the
25-delta call, which is relatively stronger demand for downside protection
-- the **put** side. So this term's vote is **inverted** before it enters
the call-side consensus. It is a price asymmetry, not a trade imbalance,
which is why it is the one term whose sign convention has to be converted.

### 5. Gamma-exposure proxy -- `options_gamma_exposure_pressure`, in [0, 1]

        percentile_rank(|proxy| over the window, |proxy| now)

A magnitude only, for the reason given at the top: the sign of a dealer
hedging flow depends on which side of the chain dealers hold, and nobody
outside a dealer observes that. So this term contributes to the magnitude
and casts no directional vote. `config.use_gamma_exposure_proxy` is an
ablation switch for Phase 8; turning it off requires zeroing its weight, so
the drop costs nothing.

## Combination, availability, and the direction

With `w_i` the five weights (summing to 1.0) and `t_i` the terms:

        available_weight = sum of w_i over terms with usable inputs
        uncapped         = sum of w_i * |t_i| over those same terms
        magnitude        = min(uncapped, cap)   on the EOD path
                         = uncapped             on the intraday path

`magnitude` is therefore in [0, available_weight] and in [0, 1], and it
falls when data is missing. That is the point.

**The availability floor.** When `available_weight <
config.min_available_weight_fraction` (default 0.50) the component reports
UNAVAILABLE rather than a number. Four of the five inputs are optional in
`OptionsSnapshot` and only net premium is always present; a premium-only
chain carries 0.25, which is below the floor, so it reports UNAVAILABLE
instead of a quarter-strength reading that would look like "options flow is
neutral" while meaning "most of this was never measured".

**The volume gate.** When `call_volume + put_volume <
config.min_snapshot_volume` the two trade-flow terms are dropped: a ratio
built from nothing traded is not a measurement of balance. The positioning
and price terms survive, because open interest is a stock and implied vol is
a price, and neither needs the session to have traded. The surviving weight
is 0.50 on defaults, exactly at the availability floor, so such a snapshot
is reported DEGRADED rather than unavailable -- which is what the field's
own documentation specifies.

**Direction.** The three call-versus-put imbalance terms vote with their own
sign and the skew term votes inverted; the gamma proxy does not vote.

        consensus = sum(w_i * vote_i) / sum(w_i)        over voting terms
        agreement = (voting weight whose vote agrees with sign(consensus))
                        / (voting weight carrying any sign)
        direction = sign(consensus)  if consensus != 0 and agreement > 0.5
                  = 0               otherwise

`agreement` is the explicit, reported encoding of section 5's ambiguity: the
individual terms measure different things, and when the things they measure
point opposite ways there is no imbalance direction to report, however large
the imbalance magnitudes are. The threshold is a strict majority of the
signed voting weight -- half the weight pointing each way is disagreement,
not a direction -- so it is not a tuned number. A term that measured exactly
zero imbalance dilutes the consensus (it is a real reading of "no imbalance
on this term") but is excluded from `agreement`, which asks only about the
terms that point somewhere.

No magnitude dead band is applied to the direction, unlike
`features/momentum.py`. The reason is section 7: magnitude is carried
separately, so a small-but-signed reading already reaches the gates as a
weak component, and a dead band would be a second, untested threshold doing
work the magnitude already does.

## Boundedness

`options_net_premium_imbalance`, `options_delta_volume_imbalance`,
`options_oi_change_imbalance`, `options_skew_25d_pressure` and
`options_imbalance_consensus` are in [-1, 1];
`options_gamma_exposure_pressure`, `options_sign_agreement`,
`options_available_weight`, `options_flow_magnitude` and
`options_flow_uncapped_magnitude` are in [0, 1]. Each is bounded *by
construction* -- a ratio of same-signed quantities, a percentile rank or a
`tanh` -- never by a clamp applied to an unbounded quantity, and never a raw
z-score, because one extreme observation in an unbounded feature dominates
every weighted sum that reads it.

`options_net_premium` (dollars), `options_skew_25d` (implied-vol points),
`options_total_volume` (contracts) and `options_snapshot_age_seconds`
(seconds) are in natural units and are diagnostics: they are reported,
plotted and regressed on, and a weighted sum must not read them.

## Key encodings

`FeatureVector` holds floats, so four keys are categorical and encoded
numerically:

`options_flow_direction`: +1 the measured call-side imbalance dominates,
-1 the put-side imbalance dominates, 0 no direction (the voting terms
disagree, the consensus is exactly zero, or no voting term is available).

`options_flow_timing_available`: 1 an intraday chain, so the terms describe
flow inside the current session; 0 an end-of-day chain, so they describe a
closed session's aggregate.

`options_eod_capped`: 1 the EOD cap changed `options_flow_magnitude` on this
bar, 0 it did not. On the EOD path the cap is in force whether or not it
binds; this key reports binding, which is the stronger claim.

`options_terms_available`: how many of the five terms produced a value, 0 to
5. Read with `options_available_weight`, which is the quantity that actually
matters, since the terms do not carry equal weight.

## Determinism and construction

No clock, no generator, no state. `__init__` takes an `OptionsFlowConfig`
and nothing else: no series, no view, no array. A full-sample statistic
captured at construction is the one leak `validation.lookahead` cannot see
through when handed an instance, so it is banned at the constructor and
`tests/unit/test_feature_contracts.py` enforces the ban package-wide.

## What is NOT claimed, one more time

Nothing in this module has been shown to have predictive content. The only
data available at Phase 5 is `data/synthetic.py`, whose options series is
generated from the same innovations as its bars, so any correlation between
these features and its forward returns is a property of the generator and
not evidence about markets. The constants below are hypotheses for the
Phase 8 sweep, and the two that are module-level rather than configured are
named as such.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from flow_model.config.schema import OptionsFlowConfig
from flow_model.core.contracts import FeatureVector, OptionsSnapshot
from flow_model.core.enums import DataQuality, Feed
from flow_model.data.market_view import MarketView
from flow_model.features.base import (
    FeatureComputer,
    FeatureError,
    percentile_rank,
    safe_divide,
    squash,
)

# ---------------------------------------------------------------------------
# encodings and module constants
# ---------------------------------------------------------------------------

#: `options_flow_direction`. +1 is a call-side imbalance, -1 a put-side one.
#: A *measured imbalance*, not a view: see the module docstring.
DIRECTION_CALL_SIDE = 1.0
DIRECTION_NONE = 0.0
DIRECTION_PUT_SIDE = -1.0

#: `options_flow_timing_available` and `options_eod_capped`.
FLAG_SET = 1.0
FLAG_CLEAR = 0.0

#: Share of the *signed* voting weight that must agree before a direction is
#: reported. A strict majority, which is why the comparison below is `>`
#: rather than `>=`: weight split exactly half each way is disagreement, and
#: calling it a direction would report the tie as a verdict.
#:
#: Not a tuned number -- there is no value other than "half" that follows
#: from "the terms that point somewhere mostly point the same way" -- and so
#: it is a module constant rather than a swept config field. If Phase 8
#: wants to sweep it, it needs a field in `OptionsFlowConfig`, which this
#: module does not own; the coupling is reported as a deviation rather than
#: worked around by editing the config layer.
MIN_SIGN_AGREEMENT = 0.5

#: Fraction of the requested snapshot window that must carry a finite value
#: before a percentile rank is computed from it.
#:
#: The two percentile terms (skew, gamma proxy) read OPTIONAL columns, which
#: may be present but hold `nan` on individual rows -- `OptionsSeries`
#: validates only the six required columns. A rank computed from four
#: surviving observations is not the rank over 21 that the caller asked for,
#: so the term is dropped instead.
#:
#: 0.50 is anchored on `OptionsFlowConfig.min_available_weight_fraction`,
#: the config's own "at least half the thing you asked for" floor, for want
#: of a dedicated field. That coupling means the Phase 8 sweep moves the
#: availability floor without moving this, which is reported as a deviation.
MIN_PERCENTILE_SAMPLE_SHARE = 0.50

#: Floating-point slack on the availability-floor comparison. NOT a
#: threshold and not tunable: `min_available_weight_fraction` is a decimal
#: fraction and so are the five weights, and summing a SUBSET of them in
#: float64 can land one ULP below the nominal total. On the defaults the
#: surviving weight of a zero-volume chain (0.20 + 0.15 + 0.15) happens to
#: evaluate to exactly 0.50, but 46 of the two-decimal weight triples that
#: nominally sum to 0.50 evaluate to 0.49999999999999994 -- so without this
#: slack a Phase 8 weight sweep that lands on one of them would report the
#: component UNAVAILABLE while its own note read "available term weight
#: 0.5000 is below the floor 0.5000". The floor means "at least this
#: fraction" (`OptionsFlowConfig.min_available_weight_fraction`), so a
#: weight at the floor is available and a representation error must not
#: decide otherwise.
WEIGHT_EPSILON = 1e-9

#: Term identifiers. Used as dict keys for the weights and in notes, so a
#: typo names itself instead of silently dropping a term's weight.
TERM_NET_PREMIUM = "net_premium"
TERM_DELTA_VOLUME = "delta_weighted_volume"
TERM_OI_CHANGE = "oi_change"
TERM_SKEW = "skew_25d"
TERM_GAMMA = "gamma_exposure"

#: Every term, in the order they are combined. Fixed, so the weighted sum is
#: evaluated in one order and two runs agree bit-for-bit.
TERM_ORDER: tuple[str, ...] = (
    TERM_NET_PREMIUM,
    TERM_DELTA_VOLUME,
    TERM_OI_CHANGE,
    TERM_SKEW,
    TERM_GAMMA,
)

#: Options-chain columns the percentile windows read. Named here so that a
#: schema change in `OptionsSeries.OPTIONAL` shows up as a failing read
#: rather than as an empty window silently dropping a term.
SKEW_COLUMNS = ("iv_25d_put", "iv_25d_call")
GAMMA_COLUMN = "gamma_exposure_proxy"


class Term(NamedTuple):
    """One of the five section-5 features, after bounding.

    `value` is signed in [-1, 1] for the four imbalance terms and a
    magnitude in [0, 1] for the gamma proxy. `vote_factor` converts the
    term's own sign convention to the shared one (positive = call side
    heavier); it is 0.0 for a term that may not vote on direction at all.
    """

    name: str
    value: float
    available: bool
    vote_factor: float

    @property
    def magnitude(self) -> float:
        """The term's contribution to the component magnitude, in [0, 1]."""
        return abs(self.value) if self.available else 0.0

    @property
    def vote(self) -> float:
        """The term's signed contribution to the direction consensus."""
        if not self.available or self.vote_factor == 0.0:
            return 0.0
        return self.vote_factor * self.value


def _clamp(value: float, low: float, high: float) -> float:
    return low if value <= low else (high if value >= high else float(value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _signed_squash(value: float, scale: float) -> float:
    """`squash` with the sign of `value` reapplied, in [-1, 1].

    `squash` takes `abs` by design, because a component *magnitude* must not
    depend on a sign. A signed term has to put the sign back by hand, and
    doing it in one named place keeps the four signed terms on one
    convention.
    """
    if not math.isfinite(value) or value == 0.0:
        return 0.0
    return math.copysign(squash(value, scale), value)


def imbalance_ratio(call_side: float, put_side: float) -> float | None:
    """`(call - put) / (call + put)` in [-1, 1], or None if undefined.

    None rather than 0.0 when the denominator is not positive. 0.0 is the
    reading for a *balanced* chain, so it cannot also be the reading for a
    chain with nothing in it -- that conflation is the one thing
    `safe_divide`'s default would do here, and a caller downstream could not
    tell the two apart.

    Both arguments are expected non-negative (the caller passes magnitudes);
    a negative input makes the ratio leave [-1, 1], so it is rejected rather
    than clamped into looking valid.
    """
    if not (math.isfinite(call_side) and math.isfinite(put_side)):
        return None
    if call_side < 0.0 or put_side < 0.0:
        return None
    total = call_side + put_side
    if total <= 0.0:
        return None
    return _clamp(safe_divide(call_side - put_side, total, 0.0), -1.0, 1.0)


def relative_rank(window: np.ndarray, value: float, minimum: int) -> float | None:
    """Percentile rank of `value` among the finite members of `window`.

    Returns None -- "not measured" -- in the two cases where a rank would be
    a number without a meaning:

    * fewer than `minimum` finite observations. `percentile_rank` would
      happily rank against three samples; three samples are not the window
      the caller asked for.
    * a window with no dispersion (`max == min`). `percentile_rank` counts
      the values *at or below* `value`, so every tie counts and a constant
      window ranks its own repeated value at 1.0 -- "maximally elevated"
      from data that found no variation at all. The same tie inflation is
      documented for quoted spreads in `features/liquidity.py`; it is a
      property of the shared helper, which this module does not own.
    """
    if window.size == 0:
        return None
    finite = window[np.isfinite(window)]
    if finite.size < max(1, int(minimum)):
        return None
    if float(finite.max()) == float(finite.min()):
        return None
    if not math.isfinite(value):
        return None
    return _clamp01(percentile_rank(finite, value))


def partial_window_note(
    computer: str, column: str, window: np.ndarray, requested: int
) -> str | None:
    """A note when a percentile was ranked over fewer rows than requested.

    `relative_rank` DROPS a term whose window carries fewer than
    `MIN_PERCENTILE_SAMPLE_SHARE` finite observations, but between that floor
    and a full window it ranks against whatever survived and returns a
    number. Both percentile terms read OPTIONAL chain columns, which may be
    present and hold `nan` on individual rows, so "ranked over 21" and
    "ranked over 11" are both reachable and were previously indistinguishable
    in the output -- the term's own note claimed it was dropped "rather than
    ranked against a window the caller did not ask for", which is what the
    code does only below the floor. The note makes the smaller sample
    visible; it does not change the value, and whether such a term should
    also be graded DEGRADED is a contract question for the component's
    consumers rather than something this function may decide.
    """
    finite = int(np.count_nonzero(np.isfinite(window)))
    if finite >= int(requested):
        return None
    return (
        f"{computer}: the {column} percentile is ranked over {finite} finite "
        f"observations of the {int(requested)} requested; the rank is a rank "
        "within what the chain supplied, not within a full window"
    )


class OptionsFlowFeatures(FeatureComputer):
    """The five section-5 options-chain measurements, bounded and combined.

    Construction takes an `OptionsFlowConfig` and nothing else -- no series,
    no view, no array, no dataset. See the module docstring on why that is a
    hard rule rather than a preference.
    """

    name = "options_flow"

    def __init__(self, config: OptionsFlowConfig) -> None:
        self.config = config
        self._window = int(config.warmup_snapshots)
        self._min_volume = float(config.min_snapshot_volume)
        self._oi_scale = float(config.oi_change_squash_scale)
        self._use_gamma = bool(config.use_gamma_exposure_proxy)
        self._cap = float(config.eod_degraded_cap_fraction)
        self._require_intraday = bool(config.require_intraday_prints)
        self._min_available = float(config.min_available_weight_fraction)
        self._min_samples = max(
            1, int(math.ceil(MIN_PERCENTILE_SAMPLE_SHARE * self._window))
        )
        weights = config.weights
        self._weights: dict[str, float] = {
            TERM_NET_PREMIUM: float(weights.net_premium),
            TERM_DELTA_VOLUME: float(weights.delta_weighted_volume),
            TERM_OI_CHANGE: float(weights.oi_change),
            TERM_SKEW: float(weights.skew_25d),
            TERM_GAMMA: float(weights.gamma_exposure),
        }
        if set(self._weights) != set(TERM_ORDER):
            raise FeatureError(
                f"{self.name}: weight table {sorted(self._weights)} does not match "
                f"the term order {sorted(TERM_ORDER)}"
            )

    # --- declared contract ---------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """A provable lower bound in bars, not the real gate.

        `warmup_snapshots` snapshots cannot exist before `warmup_snapshots`
        bars do, because this system models at most one chain snapshot per
        bar. `compute` checks the snapshot count itself on every call; see
        the module docstring for why both checks exist.
        """
        return self._window

    @property
    def keys(self) -> tuple[str, ...]:
        return (
            # the five named section-5 features, bounded
            "options_net_premium_imbalance",
            "options_delta_volume_imbalance",
            "options_oi_change_imbalance",
            "options_skew_25d_pressure",
            "options_gamma_exposure_pressure",
            # natural-unit diagnostics; a weighted sum must not read these
            "options_net_premium",
            "options_skew_25d",
            "options_total_volume",
            "options_snapshot_age_seconds",
            # availability and the degradation contract
            "options_available_weight",
            "options_terms_available",
            "options_flow_timing_available",
            "options_eod_capped",
            # combination
            "options_sign_agreement",
            "options_imbalance_consensus",
            "options_flow_uncapped_magnitude",
            "options_flow_magnitude",
            "options_flow_direction",
        )

    @property
    def required_feeds(self) -> frozenset[Feed]:
        """The options chain, and nothing else. No bar column is read."""
        return frozenset({Feed.OPTIONS_SNAPSHOT})

    @property
    def optional_feeds(self) -> frozenset[Feed]:
        return frozenset()

    # --- computation ---------------------------------------------------

    def compute(self, view: MarketView) -> FeatureVector:
        """Features at `view.now` from the last `warmup_snapshots` snapshots.

        Feed presence and the snapshot warmup are re-checked here rather
        than left to `FeatureBundle`: the lookahead audit calls computers
        directly, including at pre-warmup bars, and a computer that reported
        `warmup_complete=True` with a half-filled window would be handing
        out a number built from fewer observations than it claims.
        """
        if not self.feeds_available(view):
            return self._not_ready(view, "options_snapshot feed absent")

        count = view.options_count()
        if count < self._window:
            return self._not_ready(
                view,
                f"warmup incomplete: {count} of {self._window} options snapshots",
            )

        snapshot = view.options_snapshot()
        if snapshot is None:  # pragma: no cover - has_feed already guarantees one
            return self._not_ready(view, "no visible options snapshot")

        # The six REQUIRED chain columns are screened for finiteness here
        # because `OptionsSeries` does not screen them: its non-negativity
        # check uses `np.nanmin`, which ignores a `nan` rather than rejecting
        # it, and `OptionsSeries.snapshot_at` converts a non-finite value to
        # None only for the OPTIONAL columns. A `nan` premium would otherwise
        # reach `_vector`, which raises on a non-finite feature -- turning a
        # bad row into a crash in the middle of a backtest instead of a
        # reported not-ready. Same screen, same reason, as the OHLC check in
        # `features/volatility.py`.
        required = {
            "call_volume": snapshot.call_volume,
            "put_volume": snapshot.put_volume,
            "call_premium": snapshot.call_premium,
            "put_premium": snapshot.put_premium,
            "call_oi": snapshot.call_oi,
            "put_oi": snapshot.put_oi,
        }
        unusable = sorted(
            name for name, value in required.items() if not math.isfinite(value)
        )
        if unusable:
            return self._not_ready(
                view,
                f"the newest options snapshot has non-finite required field(s) "
                f"{unusable}; no term is computed from a partially broken chain",
            )

        intraday = bool(view.options_are_intraday())
        if self._require_intraday and not intraday:
            return self._not_ready(
                view,
                "require_intraday_prints is set and the chain is end-of-day only, "
                "so the component is disabled rather than capped",
            )

        notes: list[str] = []
        terms = self._terms(view, snapshot, intraday, notes)

        available_weight = sum(
            self._weights[t.name] for t in terms if t.available
        )
        terms_available = sum(1 for t in terms if t.available)
        if available_weight < self._min_available - WEIGHT_EPSILON:
            dropped = sorted(t.name for t in terms if not t.available)
            return self._not_ready(
                view,
                f"available term weight {available_weight:.4f} is below the floor "
                f"{self._min_available:.4f}; terms without usable inputs: {dropped}. "
                "The component reports UNAVAILABLE rather than a fraction-strength "
                "sub-score that would read as 'options flow is neutral'",
            )

        uncapped = sum(self._weights[t.name] * t.magnitude for t in terms)
        uncapped = _clamp01(uncapped)

        magnitude = uncapped
        capped = FLAG_CLEAR
        if not intraday:
            magnitude = min(uncapped, self._cap)
            if uncapped > self._cap:
                capped = FLAG_SET
                notes.append(
                    f"{self.name}: options_flow_magnitude capped at "
                    f"{self._cap:.4f} (eod_degraded_cap_fraction) from "
                    f"{uncapped:.4f}; an end-of-day chain carries no flow timing"
                )
            else:
                notes.append(
                    f"{self.name}: the end-of-day cap {self._cap:.4f} is in force "
                    f"but did not bind this bar (uncapped {uncapped:.4f})"
                )

        consensus, agreement, direction = self._direction(terms)

        age = view.feed_age_seconds(Feed.OPTIONS_SNAPSHOT)
        if age is None or not math.isfinite(age):  # pragma: no cover - feed is present
            age = 0.0
            notes.append(
                f"{self.name}: the options feed reported no age; "
                "options_snapshot_age_seconds is 0.0 (unavailable, not fresh)"
            )

        by_name = {t.name: t for t in terms}
        values = {
            "options_net_premium_imbalance": by_name[TERM_NET_PREMIUM].value,
            "options_delta_volume_imbalance": by_name[TERM_DELTA_VOLUME].value,
            "options_oi_change_imbalance": by_name[TERM_OI_CHANGE].value,
            "options_skew_25d_pressure": by_name[TERM_SKEW].value,
            "options_gamma_exposure_pressure": by_name[TERM_GAMMA].value,
            "options_net_premium": float(snapshot.call_premium - snapshot.put_premium),
            # The 0.0 is a placeholder when the chain supplied no 25-delta
            # legs, and `_quality_by_key` marks it DEGRADED in that case so it
            # cannot be read as "the skew is flat".
            "options_skew_25d": (
                0.0 if snapshot.skew_25d is None else float(snapshot.skew_25d)
            ),
            "options_total_volume": float(snapshot.total_volume),
            "options_snapshot_age_seconds": float(age),
            "options_available_weight": _clamp01(available_weight),
            "options_terms_available": float(terms_available),
            "options_flow_timing_available": FLAG_SET if intraday else FLAG_CLEAR,
            "options_eod_capped": capped,
            "options_sign_agreement": agreement,
            "options_imbalance_consensus": consensus,
            "options_flow_uncapped_magnitude": uncapped,
            "options_flow_magnitude": _clamp01(magnitude),
            "options_flow_direction": direction,
        }
        return self._vector(view, values, notes=tuple(notes)).replace(
            quality_by_key=self._quality_by_key(
                terms,
                intraday=intraday,
                skew_supplied=snapshot.skew_25d is not None,
            )
        )

    # --- the five terms -------------------------------------------------

    def _terms(
        self,
        view: MarketView,
        snapshot: OptionsSnapshot,
        intraday: bool,
        notes: list[str],
    ) -> tuple[Term, ...]:
        """The five bounded terms, in `TERM_ORDER`."""
        traded_enough = self._volume_gate(snapshot, notes)
        terms = (
            self._net_premium_term(snapshot, traded_enough, notes),
            self._delta_volume_term(snapshot, traded_enough, notes),
            self._oi_change_term(snapshot, notes),
            self._skew_term(view, snapshot, notes),
            self._gamma_term(view, snapshot, notes),
        )
        if tuple(t.name for t in terms) != TERM_ORDER:  # pragma: no cover - guard
            raise FeatureError(
                f"{self.name}: term order {[t.name for t in terms]} does not match "
                f"{list(TERM_ORDER)}"
            )
        if not intraday:
            notes.append(
                f"{self.name}: end-of-day chain, so every term is a closed-session "
                "aggregate standing in for a reading at this instant; the two "
                "trade-flow terms in particular describe the whole session's "
                "premium and delta-weighted volume, not flow inside the current bar"
            )
        return terms

    def _volume_gate(self, snapshot: OptionsSnapshot, notes: list[str]) -> bool:
        """Whether the chain traded enough for its trade-flow ratios to mean anything."""
        total = float(snapshot.total_volume)
        if not math.isfinite(total) or total < self._min_volume:
            notes.append(
                f"{self.name}: chain volume {total:g} is below "
                f"min_snapshot_volume={self._min_volume:g}; the net-premium and "
                "delta-weighted-volume terms are DROPPED (a ratio of nothing "
                "traded is not a balanced chain). The open-interest, skew and "
                "gamma terms are unaffected: open interest is a stock and "
                "implied vol is a price"
            )
            return False
        return True

    def _net_premium_term(
        self, snapshot: OptionsSnapshot, traded_enough: bool, notes: list[str]
    ) -> Term:
        """Section 5's "net premium (call-prem minus put-prem)", as an imbalance."""
        if not traded_enough:
            return Term(TERM_NET_PREMIUM, 0.0, False, 1.0)
        ratio = imbalance_ratio(
            float(snapshot.call_premium), float(snapshot.put_premium)
        )
        if ratio is None:
            notes.append(
                f"{self.name}: total option premium is "
                f"{snapshot.call_premium + snapshot.put_premium:g}, so the premium "
                "imbalance is undefined; the net-premium term is DROPPED (0.0 here "
                "is a placeholder, not a balanced chain)"
            )
            return Term(TERM_NET_PREMIUM, 0.0, False, 1.0)
        return Term(TERM_NET_PREMIUM, ratio, True, 1.0)

    def _delta_volume_term(
        self, snapshot: OptionsSnapshot, traded_enough: bool, notes: list[str]
    ) -> Term:
        """Section 5's "delta-weighted volume", as an imbalance of magnitudes."""
        if not traded_enough:
            return Term(TERM_DELTA_VOLUME, 0.0, False, 1.0)
        call_side = snapshot.delta_weighted_call_volume
        put_side = snapshot.delta_weighted_put_volume
        if call_side is None or put_side is None:
            missing = [
                name
                for name, value in (
                    ("delta_weighted_call_volume", call_side),
                    ("delta_weighted_put_volume", put_side),
                )
                if value is None
            ]
            notes.append(
                f"{self.name}: {missing} not supplied by the chain; the "
                "delta-weighted-volume term is DROPPED without redistributing its "
                "weight. None means NOT SUPPLIED and is not read as 0.0"
            )
            return Term(TERM_DELTA_VOLUME, 0.0, False, 1.0)
        # Magnitudes: see the module docstring on put-delta sign conventions.
        ratio = imbalance_ratio(abs(float(call_side)), abs(float(put_side)))
        if ratio is None:
            notes.append(
                f"{self.name}: delta-weighted volume is zero on both sides, so its "
                "imbalance is undefined; the term is DROPPED"
            )
            return Term(TERM_DELTA_VOLUME, 0.0, False, 1.0)
        return Term(TERM_DELTA_VOLUME, ratio, True, 1.0)

    def _oi_change_term(self, snapshot: OptionsSnapshot, notes: list[str]) -> Term:
        """Section 5's "OI change", as a squashed fraction of total open interest."""
        call_change = snapshot.call_oi_change
        put_change = snapshot.put_oi_change
        if call_change is None or put_change is None:
            missing = [
                name
                for name, value in (
                    ("call_oi_change", call_change),
                    ("put_oi_change", put_change),
                )
                if value is None
            ]
            notes.append(
                f"{self.name}: {missing} not supplied by the chain; the OI-change "
                "term is DROPPED. It is deliberately NOT reconstructed by "
                "differencing call_oi/put_oi across snapshots: that would be "
                "indistinguishable downstream from a supplied observation, it "
                "assumes two adjacent snapshots are adjacent sessions, and "
                "never_synthesize_missing_fields forbids it"
            )
            return Term(TERM_OI_CHANGE, 0.0, False, 1.0)
        total_oi = float(snapshot.call_oi) + float(snapshot.put_oi)
        if not math.isfinite(total_oi) or total_oi <= 0.0:
            notes.append(
                f"{self.name}: total open interest is {total_oi:g}, so a net OI "
                "change has no denominator to be a fraction of; the OI-change term "
                "is DROPPED"
            )
            return Term(TERM_OI_CHANGE, 0.0, False, 1.0)
        net = float(call_change) - float(put_change)
        fraction = safe_divide(net, total_oi, 0.0)
        return Term(
            TERM_OI_CHANGE, _signed_squash(fraction, self._oi_scale), True, 1.0
        )

    def _skew_term(
        self, view: MarketView, snapshot: OptionsSnapshot, notes: list[str]
    ) -> Term:
        """Section 5's "25d skew", as a rank within its own recent distribution.

        Votes with an INVERTED sign: a high relative 25-delta put-IV premium
        is relatively stronger demand for downside protection, which is the
        put side of the shared convention.
        """
        current = snapshot.skew_25d
        if current is None:
            notes.append(
                f"{self.name}: iv_25d_put and/or iv_25d_call not supplied by the "
                "chain, so skew_25d is undefined; the skew term is DROPPED"
            )
            return Term(TERM_SKEW, 0.0, False, -1.0)
        puts = view.options_column(SKEW_COLUMNS[0], self._window)
        calls = view.options_column(SKEW_COLUMNS[1], self._window)
        if puts.size != self._window or calls.size != self._window:
            notes.append(
                f"{self.name}: skew window is short ({SKEW_COLUMNS[0]}={puts.size}, "
                f"{SKEW_COLUMNS[1]}={calls.size}, expected {self._window} each); "
                "the skew term is DROPPED rather than ranked against a window the "
                "caller did not ask for"
            )
            return Term(TERM_SKEW, 0.0, False, -1.0)
        rank = relative_rank(puts - calls, float(current), self._min_samples)
        if rank is None:
            notes.append(
                f"{self.name}: the {self._window}-snapshot skew window has fewer "
                f"than {self._min_samples} finite observations or no dispersion at "
                "all, so a percentile rank would be a number without a meaning; "
                "the skew term is DROPPED"
            )
            return Term(TERM_SKEW, 0.0, False, -1.0)
        partial = partial_window_note(
            self.name, "skew_25d", puts - calls, self._window
        )
        if partial is not None:
            notes.append(partial)
        # 2p - 1: the window median maps to 0.0, so the term shares the scale
        # of the three imbalance ratios.
        return Term(TERM_SKEW, _clamp(2.0 * rank - 1.0, -1.0, 1.0), True, -1.0)

    def _gamma_term(
        self, view: MarketView, snapshot: OptionsSnapshot, notes: list[str]
    ) -> Term:
        """Section 5's "gamma-exposure proxy", as a magnitude only.

        `vote_factor` is 0.0: the sign of a dealer hedging flow depends on
        which side of the chain dealers hold, which nobody outside a dealer
        observes. A magnitude is what the data supports.
        """
        if not self._use_gamma:
            notes.append(
                f"{self.name}: use_gamma_exposure_proxy is off (a Phase 8 ablation, "
                "not missing data); the gamma term is not computed. Its weight is "
                "required to be 0.0 in that configuration, so nothing is lost"
            )
            return Term(TERM_GAMMA, 0.0, False, 0.0)
        current = snapshot.gamma_exposure_proxy
        if current is None:
            notes.append(
                f"{self.name}: {GAMMA_COLUMN} not supplied by the chain; the "
                "gamma-proxy term is DROPPED without redistributing its weight"
            )
            return Term(TERM_GAMMA, 0.0, False, 0.0)
        window = view.options_column(GAMMA_COLUMN, self._window)
        if window.size != self._window:
            notes.append(
                f"{self.name}: gamma window is short ({window.size}, expected "
                f"{self._window}); the gamma-proxy term is DROPPED"
            )
            return Term(TERM_GAMMA, 0.0, False, 0.0)
        rank = relative_rank(
            np.abs(window), abs(float(current)), self._min_samples
        )
        if rank is None:
            notes.append(
                f"{self.name}: the {self._window}-snapshot gamma-proxy window has "
                f"fewer than {self._min_samples} finite observations or no "
                "dispersion at all; the gamma-proxy term is DROPPED"
            )
            return Term(TERM_GAMMA, 0.0, False, 0.0)
        partial = partial_window_note(
            self.name, GAMMA_COLUMN, window, self._window
        )
        if partial is not None:
            notes.append(partial)
        return Term(TERM_GAMMA, _clamp01(rank), True, 0.0)

    # --- combination ----------------------------------------------------

    def _direction(self, terms: tuple[Term, ...]) -> tuple[float, float, float]:
        """`(consensus, agreement, direction)` from the voting terms.

        See the module docstring. The consensus denominator is the whole
        available voting weight, so a term that measured no imbalance
        dilutes it; the agreement denominator is only the voting weight that
        carries a sign, because agreement asks about the terms that point
        somewhere.
        """
        voting = [t for t in terms if t.available and t.vote_factor != 0.0]
        voting_weight = sum(self._weights[t.name] for t in voting)
        if voting_weight <= 0.0:
            return 0.0, 0.0, DIRECTION_NONE

        consensus = _clamp(
            safe_divide(
                sum(self._weights[t.name] * t.vote for t in voting),
                voting_weight,
                0.0,
            ),
            -1.0,
            1.0,
        )

        signed = [t for t in voting if t.vote != 0.0]
        signed_weight = sum(self._weights[t.name] for t in signed)
        if consensus == 0.0 or signed_weight <= 0.0:
            return consensus, 0.0, DIRECTION_NONE

        consensus_sign = math.copysign(1.0, consensus)
        agree_weight = sum(
            self._weights[t.name]
            for t in signed
            if math.copysign(1.0, t.vote) == consensus_sign
        )
        agreement = _clamp01(safe_divide(agree_weight, signed_weight, 0.0))
        if agreement <= MIN_SIGN_AGREEMENT:
            return consensus, agreement, DIRECTION_NONE
        return (
            consensus,
            agreement,
            DIRECTION_CALL_SIDE if consensus > 0.0 else DIRECTION_PUT_SIDE,
        )

    # --- per-key quality ------------------------------------------------

    def _quality_by_key(
        self, terms: tuple[Term, ...], *, intraday: bool, skew_supplied: bool
    ) -> dict[str, DataQuality]:
        """Per-key quality, so a caller can tell which terms degraded.

        GOOD or DEGRADED only. MISSING is reserved for `_not_ready`, which
        is the state in which the component as a whole is UNAVAILABLE and
        every value is a placeholder -- a dropped single term is not that,
        and reporting it as MISSING would make `FeatureVector.quality`
        MISSING for the whole bundle and block every other component.

        The rule separating the terms from the diagnostics: a **term** claims
        to describe the chain's imbalance at `view.now`, so an end-of-day
        snapshot degrades it. A **diagnostic** is a required field read
        straight off the snapshot, or a property of the feed itself, and an
        EOD snapshot's premium, volume and age are exactly as observed --
        `options_snapshot_age_seconds` is reported precisely so a consumer
        can see how old "as observed" is. Grading every key DEGRADED on the
        EOD path would carry no information at all.
        """
        good, degraded = DataQuality.GOOD, DataQuality.DEGRADED
        available = {t.name: t.available for t in terms}

        def term_quality(name: str) -> DataQuality:
            # An end-of-day chain degrades every term: it is a closed
            # session's aggregate, not a reading at `view.now`.
            return good if available[name] and intraday else degraded

        # Completeness is measured in WEIGHT, not in term count, because
        # weight is what the magnitude actually loses. A term disabled by
        # configuration carries zero weight (`OptionsFlowConfig._check`
        # refuses to disable the gamma proxy while its weight is positive),
        # so the Phase 8 ablation loses nothing and the combination keys stay
        # GOOD. A term dropped for missing data costs its whole weight and
        # the combination says so, even though the component stayed above its
        # availability floor.
        #
        # The ablated term's OWN key is still DEGRADED: it holds the 0.0
        # placeholder rather than a measurement. So an ablation run reports
        # `FeatureVector.quality` (the worst over all keys) as DEGRADED while
        # `options_flow_magnitude` reports GOOD -- which is why
        # `quality_by_key` exists, and is a distinction a Phase 8 comparison
        # has to read per key rather than gating on the aggregate.
        lost_weight = sum(
            self._weights[t.name] for t in terms if not t.available
        )
        combined = good if lost_weight <= 0.0 and intraday else degraded

        quality = {
            "options_net_premium_imbalance": term_quality(TERM_NET_PREMIUM),
            "options_delta_volume_imbalance": term_quality(TERM_DELTA_VOLUME),
            "options_oi_change_imbalance": term_quality(TERM_OI_CHANGE),
            "options_skew_25d_pressure": term_quality(TERM_SKEW),
            "options_gamma_exposure_pressure": term_quality(TERM_GAMMA),
            # Diagnostics of the chain and the feed itself. These are
            # measured whatever the terms do, and an EOD snapshot's premium,
            # volume and age are exactly as observed.
            "options_net_premium": good,
            "options_total_volume": good,
            "options_snapshot_age_seconds": good,
            "options_available_weight": good,
            "options_terms_available": good,
            "options_flow_timing_available": good,
            "options_eod_capped": good,
            # skew_25d is a diagnostic but it is only a measurement when the
            # chain supplied both legs; otherwise it is the 0.0 placeholder.
            # Keyed on whether the CHAIN supplied them, not on whether the
            # skew TERM survived: the term also drops when its 21-snapshot
            # window has no dispersion or too few finite observations, and in
            # that case the emitted diagnostic is still the measured
            # `iv_25d_put - iv_25d_call`. Grading a real measurement DEGRADED
            # because a percentile could not be formed from it reported the
            # wrong thing about the wrong key.
            "options_skew_25d": good if skew_supplied else degraded,
            "options_sign_agreement": combined,
            "options_imbalance_consensus": combined,
            "options_flow_uncapped_magnitude": combined,
            "options_flow_magnitude": combined,
            "options_flow_direction": combined,
        }
        missing = set(self.keys) - set(quality)
        if missing:  # pragma: no cover - guarded by the key/quality test
            raise FeatureError(f"{self.name}: no quality declared for {sorted(missing)}")
        return quality
