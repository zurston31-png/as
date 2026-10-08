"""Mathematical regime classification (ARCHITECTURE.md section 6).

Five measurements per bar, from point-in-time bars only, then a fixed
classification order with thresholds from YAML, then hysteresis. No bar is
ever labelled by hand, and no threshold in this module was moved after it
was scored -- the confusion matrix in `tests/unit/test_regime.py` is
measured at the shipped defaults and asserts what was measured, including
where that is unflattering.

## The diagnostic vector

`RegimeState` carries the primary label *plus* every measurement, because
regimes are not mutually exclusive in reality: a bar can be simultaneously
in the top vol decile and travelling efficiently in one direction, and the
classifier has to pick one name for it. Phase 8 breaks results down by both,
so discarding the vector and keeping only the label would throw away the
only evidence that the taxonomy is or is not carving the tape correctly.

* `vol_pct` -- percentile rank of Wilder ATR(`atr_period`) inside a rolling
  `atr_percentile_lookback`-value window of ATRs. Bounded in [0, 1] by
  construction. Numerically identical to `VolatilityFeatures.atr_percentile`
  at the same bar, which `test_regime.py` asserts directly.
* `trend_tau` -- Kendall tau-b of close against bar index over
  `efficiency_period` bars, in [-1, 1]. `trend_strength` is `|trend_tau|`.
* `efficiency` -- Kaufman efficiency ratio over `efficiency_period` bars:
  net displacement over summed absolute travel, in [0, 1].
* `vol_of_vol` -- `VolatilityFeatures`' own definition: the coefficient of
  variation (stdev / mean) of the rolling realized-vol series. Section 6
  says "stdev of rolling realized vol"; the landed volatility module
  normalizes that stdev by the mean so the number is scale-free in both
  price and bar interval, and introducing a second, scale-dependent
  `vol_of_vol` here would mean two features with one name. The landed
  definition is reused and the deviation from section 6's wording is
  recorded here rather than resolved silently.
* `shift_stat` -- two-sided CUSUM of standardized returns over
  `cusum_window_bars`, with allowance `cusum_drift`. `shift_peak` is its
  maximum over the trailing `shift_decay_bars` bars, which is the quantity
  the DIRECTIONAL_SHIFT branch actually tests: section 6's threshold check
  and `shift_decay_bars`' persistence are one comparison,
  `shift_peak > cusum_threshold`.

Two bounded companions are emitted for the scoring layer, which may only
read bounded quantities: `shift_score = squash(shift_stat, cusum_threshold)`
and `trend_strength`. `shift_stat`, `vol_of_vol` and `realized_vol` are
diagnostics in natural units: they are compared against their own
thresholds and plotted, and a weighted sum must not read them.

## Kendall tau rather than ADX, and why one statistic carries both

Section 6 offers "|Kendall tau of close vs time| over N bars, or ADX(14)".
Kendall tau is used, for three reasons. It is bounded in [-1, 1] by
construction, so it satisfies the no-unbounded-transform rule without a
second normalization; ADX is a 0-100 smoothed average whose scale depends on
the instrument and which needs its own trailing percentile before it is
comparable, which is a third rolling chain and a third window to declare.
It is rank-based, so a single outlier print cannot move it much, and bad
prints are exactly what `data/clean.py` exists to find and does not always
catch. And it needs one window, not the three nested Wilder chains ADX
needs (+DI, -DI, DX, then the smoothing of DX), each of which would have to
be anchored inside this module's own window.

The *sign* of the same tau is the `slope` of section 6's trend branches.
Using one statistic for both strength and direction means they cannot
contradict each other: with an OLS slope for direction and a tau for
strength, a window could report a strong trend and a slope of the opposite
sign, and the classifier would have no defined answer. A tau of exactly
zero -- equal numbers of concordant and discordant pairs -- is not a
direction, and such a bar falls through to CHOP.

`N` is `FeatureConfig.efficiency_period`, so `trend_tau` and `efficiency`
describe the same stretch of tape. `momentum.py` already states that the
regime detector reads efficiency over exactly that window; sharing it makes
the two trend diagnostics comparable to each other and to
`MomentumFeatures.efficiency_ratio`, which `test_regime.py` asserts is the
same number.

## The CUSUM, and what its window can and cannot see

For the last `cusum_window_bars` log returns `r_1..r_C`:

    mu, sd = mean(r), stdev(r, ddof=1)
    z_i    = (r_i - mu) / sd
    S+_i   = max(0, S+_{i-1} + z_i - cusum_drift),   S+_0 = 0
    S-_i   = max(0, S-_{i-1} - z_i - cusum_drift),   S-_0 = 0
    shift_stat = max(S+_C, S-_C)

Standardizing with the window's *own* mean is deliberate and has a
consequence worth stating: a shift that spans the entire window is absorbed
into `mu` and is invisible. This statistic detects a break *inside* its
window, not a level difference between this window and some earlier one. A
detector that compared against a longer baseline would detect the latter
too, at the cost of a second window and a second staleness question.

The window length bounds what is detectable. Starting from zero, `S+`
cannot exceed `C * (delta - cusum_drift)` for a sustained standardized
shift of `delta`, so reaching `cusum_threshold` inside the window requires

    delta > cusum_drift + cusum_threshold / cusum_window_bars

which is 0.5 + 4/60 = 0.567 standard deviations per bar at the defaults.
That is a strong requirement, and it is reported as a finding rather than
tuned away: see the DIRECTIONAL_SHIFT row of the confusion matrix in
`test_regime.py`, where the synthetic generator's own shift regime has a
per-bar drift-to-vol ratio of 0.0005/0.0018 = 0.28 -- below `cusum_drift`
alone, so its mean increment is negative and the statistic can only reach
the threshold on a noise burst.

`shift_decay_bars` makes the trigger persist: a bar is raw-labelled
DIRECTIONAL_SHIFT when `shift_stat > cusum_threshold` on *any* of the last
`shift_decay_bars` bars. That is bounded memory, not state -- it is a
maximum over a trailing window, computable from the visible prefix alone.

## Hysteresis without mutable state -- the load-bearing design decision

The obvious implementation keeps `self._label` and `self._count` and
updates them on each `classify` call. That is wrong here, and not merely
inelegant. A detector that remembers its last answer returns a different
label for the *same* `MarketView` depending on what it was asked before,
so a backtest's labels depend on the order bars were evaluated in, a
re-run of a single bar disagrees with the original run, and two backtests
that differ only in where they started produce different regimes for the
same day. ARCHITECTURE.md principle 5 requires bit-identical results from
the same config and data; a stateful detector cannot provide that, and the
failure is invisible because every individual run looks self-consistent.

So the label sequence is recomputed from the visible prefix on every call.
The stateful automaton -- "switch to label L once L has been the raw label
for `min_regime_bars` consecutive bars" -- is replaced by its closed form:

    confirmed(t) = raw(s*),  s* = max{ s <= t : raw[s-k+1 .. s] are all equal }

with `k = min_regime_bars`. These are the same function. The automaton's
state changes exactly when a run of `k` identical raw labels completes, and
holds otherwise, so its state at `t` is the label of the most recently
completed `k`-run -- which is what the closed form evaluates. The closed
form is a pure function of the raw labels, so it is order-independent by
construction rather than by discipline.

The search for `s*` is bounded by `hysteresis_search_bars`, because an
unbounded search would make each call O(bars so far) and the whole backtest
quadratic. A bar with no completed `k`-run anywhere in that window has had
no label stable for `k` bars in the whole horizon, and is reported as
`Regime.UNKNOWN` -- never tradable -- rather than carrying a stale label
forward from beyond the horizon. Measured on the scored dataset in
`test_regime.py`, that fallback fires on a small minority of bars and its
rate is asserted, so a future change that made it common would fail.

The bound also makes `classify` exactly equal to `label_span` over the
whole series at the same bar, because `maximum.accumulate` over a span
holding the last `hysteresis_search_bars` raw labels finds the same `s*` as
one over the entire history whenever `s*` is inside the horizon, and
reports UNKNOWN in both when it is not. `test_regime.py` asserts that
equality bar by bar.

`bars_in_regime` is censored at `hysteresis_search_bars` for the same
reason: an uncensored run length would depend on how much history the
caller happened to load, which is the non-reproducibility this module is
built to avoid. It is a floor, and reads as such.

## Per-bar anchoring, and why this module rebuilds the ATR chain per bar

`vol_pct` at bar `t` is computed from exactly the `atr_period +
atr_percentile_lookback` bars ending at `t`, with Wilder's recursion
anchored at the start of that window -- the same convention
`VolatilityFeatures` uses and documents. The cheaper alternative is one
Wilder chain over the whole span with sub-windows taken from it, which
costs one pass instead of `count` passes. It is rejected: the ATR values
would then be anchored at the span start, so `vol_pct` at bar `t` would
depend on how many bars the caller asked for, and "the regime at bar t"
would stop being a well-defined quantity. That is not lookahead -- every
input is still strictly in the past -- but it is exactly the
non-reproducibility the hysteresis design above exists to prevent, and it
would be inconsistent with the published `atr_percentile`.

Everything else (realized vol, vol-of-vol, efficiency, Kendall tau, the
CUSUM recursion) is window-local and is computed for the whole span in one
vectorized pass.

## Warmup

`warmup_bars` is the longest *chain*, not the longest single lookback:

    inst   = max(atr_period + atr_percentile_lookback,          # 266
                 realized_vol_period + vol_of_vol_period,       #  40
                 efficiency_period + 1,                         #  21
                 cusum_window_bars + 1)                         #  61
    raw    = max(inst, cusum_window_bars + shift_decay_bars)    # 266, 70
    warmup = raw + min_regime_bars - 1                          # 268

266 bars for one ATR percentile, then `shift_decay_bars - 1` more only if
the CUSUM chain is the binding one, then `min_regime_bars - 1` more because
a confirmed label needs that many raw labels behind it. 268 on the shipped
defaults. `FeatureConfig.warmup_bars` reports 253, which is short of this
for the same reason `volatility.py` records: it omits the `atr_period` the
ATR chain consumes before its first value exists, and it knows nothing
about the regime config's own windows. `FeatureBundle.warmup_bars` takes the
max over its computers, so declaring the true number here is sufficient --
but the config's figure is optimistic and is reported as a deviation rather
than silently matched.

## Construction takes configuration only

No series, no view, no array, no dataset. A full-sample statistic captured
in `__init__` is the one lookahead the audit harness cannot see through
when it is handed an instance, and `tests/unit/test_feature_contracts.py`
discovers this class and asserts the constructor cannot accept market data.
`test_regime.py` runs the audit with `factory=`, which is the form that
rebuilds the detector against the altered dataset and so catches a captured
constant, and separately proves the gate is not vacuous by subclassing this
detector into three deliberate cheats and requiring findings for each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flow_model.config.schema import FeatureConfig, RegimeConfig
from flow_model.core.contracts import FeatureVector, RegimeState
from flow_model.core.enums import Regime
from flow_model.data.market_view import MarketView
from flow_model.features.base import (
    FeatureComputer,
    FeatureError,
    percentile_rank,
    safe_divide,
    squash,
)
from flow_model.features.volatility import bars_per_year, true_range, wilder_atr_series

#: Integer encoding of the labels this module emits, in code order.
#:
#: `regime_code` in the feature vector is an index into this tuple, and the
#: confusion matrix in the tests indexes it too. UNKNOWN is code 0 so that
#: "no label" is the zero value, and the remaining order matches section 6's
#: classification order so a code is readable as a precedence rank.
REGIME_CODES: tuple[Regime, ...] = (
    Regime.UNKNOWN,
    Regime.DIRECTIONAL_SHIFT,
    Regime.HIGH_VOL,
    Regime.LOW_VOL,
    Regime.TRENDING_UP,
    Regime.TRENDING_DOWN,
    Regime.CHOP,
)

#: Reverse lookup, built once.
CODE_BY_REGIME: dict[Regime, int] = {r: i for i, r in enumerate(REGIME_CODES)}

UNKNOWN_CODE = CODE_BY_REGIME[Regime.UNKNOWN]


def regime_of_code(code: int) -> Regime:
    """The label for an integer code, raising on a code this module cannot emit."""
    index = int(code)
    if not 0 <= index < len(REGIME_CODES):
        raise FeatureError(
            f"regime code {code!r} is outside the declared encoding "
            f"{[r.value for r in REGIME_CODES]}"
        )
    return REGIME_CODES[index]


# ---------------------------------------------------------------------------
# the measurements, as free functions on arrays
# ---------------------------------------------------------------------------


def kendall_tau_vs_time(values: np.ndarray) -> float:
    """Kendall tau-b of `values` against their own index, in [-1, 1].

        tau_b = (C - D) / sqrt(n0 * (n0 - n2))

    with `n0 = n(n-1)/2`, `C - D = sum_{i<j} sign(y_j - y_i)` and `n2` the
    number of tied pairs in `y`. The index has no ties, so the usual `n1`
    term is zero and drops out.

    Two degenerate inputs: fewer than two values has no pair to be
    concordant about, and a window in which every value is equal has
    `n0 == n2`. Both return 0.0 -- "no evidence of direction" -- rather than
    a nan or a 1.0 that would read a halted tape as a perfect trend.
    """
    y = np.asarray(values, dtype=np.float64)
    n = y.size
    if n < 2:
        return 0.0
    # sign_matrix[i, j] = sign(y_j - y_i); only the strict upper triangle is
    # a pair with i < j.
    differences = y[None, :] - y[:, None]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    signs = np.sign(differences[upper])
    concordance = float(signs.sum())
    pairs = 0.5 * n * (n - 1)
    ties = float(np.count_nonzero(signs == 0.0))
    denominator = pairs * (pairs - ties)
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(concordance / math.sqrt(denominator), -1.0, 1.0))


def kaufman_efficiency(values: np.ndarray) -> float:
    """Kaufman efficiency ratio of `values`, in [0, 1].

        |y_last - y_first| / sum_i |y_i - y_{i-1}|

    Net displacement over total travel: 1.0 when every step went the same
    way, near 0 when the steps cancelled. A flat window has no travel to
    divide by and returns 0.0, not 1.0 -- a tape that has not moved has no
    direction to have been efficient about.

    Identical to `MomentumFeatures.efficiency_ratio` over the same window,
    which `test_regime.py` asserts rather than assumes.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.size < 2:
        return 0.0
    steps = np.diff(y)
    travel = float(np.abs(steps).sum())
    displacement = abs(float(y[-1]) - float(y[0]))
    return float(np.clip(safe_divide(displacement, travel, default=0.0), 0.0, 1.0))


def cusum_statistic(returns: np.ndarray, drift: float) -> float:
    """Two-sided CUSUM of standardized `returns`. See the module docstring.

    Returns 0.0 when the window has no return dispersion: with `sd == 0`
    there is no standardized scale, and a break in a series that never moves
    is not a break.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if not math.isfinite(sd) or sd <= 0.0:
        return 0.0
    z = (r - float(r.mean())) / sd
    high = 0.0
    low = 0.0
    for value in z:
        high = max(0.0, high + float(value) - drift)
        low = max(0.0, low - float(value) - drift)
    return max(high, low)


def _rolling_kendall_tau(closes: np.ndarray, window: int, chunk: int = 2048) -> np.ndarray:
    """`kendall_tau_vs_time` over every trailing `window + 1`-close window.

    `out[j]` is the tau of `closes[j : j + window + 1]`, so it belongs to bar
    index `j + window`. Chunked rather than one big broadcast: the pair
    matrix is `rows x (window+1) x (window+1)`, which is 21 x 21 doubles per
    row at the defaults and would be several hundred megabytes over a
    multi-year dataset in one allocation.
    """
    span = int(window) + 1
    n = closes.size
    rows = n - span + 1
    if rows <= 0:
        return np.zeros(0, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(closes, span)
    pairs = 0.5 * span * (span - 1)
    upper = np.triu(np.ones((span, span), dtype=bool), k=1)
    out = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, int(chunk)):
        stop = min(rows, start + int(chunk))
        block = windows[start:stop]
        # differences[b, i, j] = block[b, j] - block[b, i]
        differences = block[:, None, :] - block[:, :, None]
        signs = np.sign(differences[:, upper])
        concordance = signs.sum(axis=1)
        ties = (signs == 0.0).sum(axis=1).astype(np.float64)
        denominator = pairs * (pairs - ties)
        tau = np.zeros(stop - start, dtype=np.float64)
        usable = denominator > 0.0
        tau[usable] = concordance[usable] / np.sqrt(denominator[usable])
        out[start:stop] = np.clip(tau, -1.0, 1.0)
    return out


def _rolling_cusum(log_returns: np.ndarray, window: int, drift: float) -> np.ndarray:
    """`cusum_statistic` over every trailing `window` of `log_returns`.

    `out[j]` is the statistic over `log_returns[j : j + window]`, which ends
    at bar index `j + window` (return `i` spans bars `i` to `i + 1`).

    The recursion is run across bars rather than along them: `window`
    vector steps over all rows at once instead of `rows` Python loops of
    `window` steps. Bit-identical to `cusum_statistic` on each window, which
    `test_regime.py` asserts.
    """
    w = int(window)
    rows = log_returns.size - w + 1
    if rows <= 0:
        return np.zeros(0, dtype=np.float64)
    blocks = np.lib.stride_tricks.sliding_window_view(log_returns, w)
    sd = blocks.std(axis=1, ddof=1)
    mean = blocks.mean(axis=1)
    usable = np.isfinite(sd) & (sd > 0.0)
    safe_sd = np.where(usable, sd, 1.0)
    z = (blocks - mean[:, None]) / safe_sd[:, None]
    high = np.zeros(rows, dtype=np.float64)
    low = np.zeros(rows, dtype=np.float64)
    for i in range(w):
        column = z[:, i]
        high = np.maximum(0.0, high + column - drift)
        low = np.maximum(0.0, low - column - drift)
    return np.where(usable, np.maximum(high, low), 0.0)


# ---------------------------------------------------------------------------
# the span of labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeSpan:
    """Diagnostics and labels for a contiguous run of bars.

    Every array has the same length and is aligned to `bar_index`, which
    holds the position of each labelled bar inside the array of closes the
    span was computed from. A span is the output of one vectorized pass; it
    is what `classify` reads its newest element from and what analytics
    reads whole.

    `raw_code` is the classification before hysteresis and `code` after it.
    Both are kept: the difference between them *is* the hysteresis lag, and
    measuring that lag needs both series.
    """

    bar_index: np.ndarray
    vol_pct: np.ndarray
    trend_tau: np.ndarray
    efficiency: np.ndarray
    realized_vol: np.ndarray
    vol_of_vol: np.ndarray
    shift_stat: np.ndarray
    shift_peak: np.ndarray
    raw_code: np.ndarray
    code: np.ndarray
    bars_in_regime: np.ndarray

    def __len__(self) -> int:
        return int(self.bar_index.size)

    def regimes(self) -> list[Regime]:
        return [regime_of_code(int(c)) for c in self.code]

    def at(self, position: int) -> dict[str, float]:
        """The diagnostic vector at one position in the span, as plain floats."""
        return {
            "vol_pct": float(self.vol_pct[position]),
            "trend_tau": float(self.trend_tau[position]),
            "trend_strength": abs(float(self.trend_tau[position])),
            "efficiency": float(self.efficiency[position]),
            "realized_vol": float(self.realized_vol[position]),
            "vol_of_vol": float(self.vol_of_vol[position]),
            "shift_stat": float(self.shift_stat[position]),
            "shift_peak": float(self.shift_peak[position]),
        }


class RegimeDetector(FeatureComputer):
    """Section 6's classifier: five measurements, a fixed order, hysteresis.

    Construction takes `RegimeConfig` and `FeatureConfig` and nothing else.
    It is given no `SymbolData`, no `BarSeries` and no array of prices,
    because a statistic captured at construction time from the full sample
    is the one lookahead leak `validation/lookahead.py` cannot see through
    when it is handed an already-built instance.
    """

    name = "regime"

    def __init__(self, config: RegimeConfig, features: FeatureConfig) -> None:
        self.config = config
        self.features = features

        self._atr_period = int(features.atr_period)
        self._atr_lookback = int(features.atr_percentile_lookback)
        self._rv_period = int(features.realized_vol_period)
        self._vov_period = int(features.vol_of_vol_period)
        self._trend_period = int(features.efficiency_period)

        self._cusum_window = int(config.cusum_window_bars)
        self._cusum_threshold = float(config.cusum_threshold)
        self._cusum_drift = float(config.cusum_drift)
        self._decay_bars = int(config.shift_decay_bars)
        self._min_regime_bars = int(config.min_regime_bars)
        self._search_bars = int(config.hysteresis_search_bars)

        self._high_vol = float(config.high_vol_percentile)
        self._low_vol = float(config.low_vol_percentile)
        self._trend_eff = float(config.trend_efficiency_threshold)

        # The chains, spelled out. See the module docstring on warmup.
        self._atr_chain_bars = self._atr_period + self._atr_lookback
        self._vov_chain_bars = self._rv_period + self._vov_period
        self._trend_bars = self._trend_period + 1
        self._cusum_bars = self._cusum_window + 1
        self._inst_bars = max(
            self._atr_chain_bars, self._vov_chain_bars, self._trend_bars, self._cusum_bars
        )
        #: Bars behind one RAW label: the instantaneous window, or the CUSUM
        #: window extended by the decay lookback when that is longer.
        self._raw_bars = max(self._inst_bars, self._cusum_window + self._decay_bars)
        #: Bars behind one CONFIRMED label.
        self._warmup_bars = self._raw_bars + self._min_regime_bars - 1
        #: Bars `classify` reads. Enough raw labels for a full hysteresis
        #: horizon (`hysteresis_search_bars`) *plus* the `min_regime_bars - 1`
        #: behind the oldest run that horizon may select. Without those extra
        #: raw labels the run ending at the far edge of the horizon would be
        #: invisible to `classify` and visible to a whole-series pass, and the
        #: two would disagree on exactly the bars where a regime had just
        #: been confirmed.
        self._classify_bars = (
            self._raw_bars + self._search_bars + self._min_regime_bars - 2
        )

    # --- declared contract ---------------------------------------------

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    @property
    def keys(self) -> tuple[str, ...]:
        return (
            "vol_pct",
            "trend_tau",
            "trend_strength",
            "efficiency",
            "realized_vol",
            "vol_of_vol",
            "shift_stat",
            "shift_peak",
            "shift_score",
            "regime_code",
            "raw_regime_code",
            "regime_confidence",
            "bars_in_regime",
        )

    # --- the vectorized pass -------------------------------------------

    def label_span(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        interval_seconds: int,
    ) -> RegimeSpan:
        """Diagnostics and labels for every bar of `closes` that has both.

        The arrays are a contiguous window of bars, oldest to newest, and
        the returned span covers indices `warmup_bars - 1 .. len(closes)-1`.
        Public because analytics and the scoring harness need the whole
        series and a `MarketView` can only be aimed at one instant;
        `classify` calls it with the last `_classify_bars` bars, and
        `test_regime.py` asserts the two agree bar by bar.

        This method takes arrays rather than a view on purpose, and that is
        not a hole in the firewall: it is called with arrays a view handed
        over, it holds no state, and the constructor -- the only place a
        full-sample statistic could hide -- still refuses market data.
        """
        h = np.asarray(highs, dtype=np.float64)
        low = np.asarray(lows, dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)
        if not (h.size == low.size == c.size):
            raise FeatureError(
                f"label_span needs equal-length arrays, got high={h.size} "
                f"low={low.size} close={c.size}"
            )
        n = c.size
        if n < self._warmup_bars:
            raise FeatureError(
                f"{self.name}: label_span needs at least {self._warmup_bars} bars, "
                f"got {n}"
            )
        if not np.all(np.isfinite(c)) or not np.all(c > 0.0):
            raise FeatureError(
                f"{self.name}: closes contain a non-positive or non-finite price; "
                "every return and log ratio below would be undefined"
            )

        annualize = math.sqrt(bars_per_year(int(interval_seconds)))

        # --- vol_pct: one Wilder chain per bar, anchored in its own window.
        first_inst = self._inst_bars - 1
        inst_count = n - first_inst
        vol_pct = np.empty(inst_count, dtype=np.float64)
        chain = self._atr_chain_bars
        for position in range(inst_count):
            end = first_inst + position + 1
            start = end - chain
            ranges = true_range(h[start:end], low[start:end], c[start:end])
            atr_series = wilder_atr_series(ranges, self._atr_period)
            window = atr_series[-self._atr_lookback :]
            if window.size != self._atr_lookback:
                raise FeatureError(
                    f"{self.name}: {chain} bars yielded {atr_series.size} ATR values "
                    f"but the percentile needs {self._atr_lookback}; warmup is wrong"
                )
            if float(window.max()) <= 0.0:
                # Matches VolatilityFeatures: "fraction at or below" would
                # rank a zero ATR at 1.0 among other zeros and report a
                # halted tape as HIGH_VOL, the exact inverse of the truth.
                vol_pct[position] = 0.0
            else:
                vol_pct[position] = percentile_rank(window, float(atr_series[-1]))

        # --- realized vol and vol-of-vol, vectorized.
        log_returns = np.diff(np.log(c))
        rv_all = (
            np.lib.stride_tricks.sliding_window_view(log_returns, self._rv_period).std(
                axis=1, ddof=1
            )
            * annualize
        )
        # rv_all[j] ends at bar j + rv_period.
        rv_index = np.arange(first_inst, n) - self._rv_period
        realized_vol = rv_all[rv_index]
        vov_windows = np.lib.stride_tricks.sliding_window_view(rv_all, self._vov_period)
        # vov_windows[j] is the vov_period realized vols ending at rv_all[j + vov_period - 1],
        # i.e. at bar j + vov_period - 1 + rv_period.
        vov_index = rv_index - self._vov_period + 1
        vov_block = vov_windows[vov_index]
        numerator = vov_block.std(axis=1, ddof=1)
        denominator = vov_block.mean(axis=1)
        vol_of_vol = np.array(
            [safe_divide(float(a), float(b), 0.0) for a, b in zip(numerator, denominator)],
            dtype=np.float64,
        )

        # --- efficiency and Kendall tau over the trend window.
        steps = np.abs(np.diff(c))
        travel_all = np.lib.stride_tricks.sliding_window_view(
            steps, self._trend_period
        ).sum(axis=1)
        # travel_all[j] spans bars j .. j + trend_period.
        trend_index = np.arange(first_inst, n) - self._trend_period
        travel = travel_all[trend_index]
        displacement = np.abs(c[first_inst:] - c[trend_index])
        efficiency = np.array(
            [
                float(np.clip(safe_divide(float(d), float(t), 0.0), 0.0, 1.0))
                for d, t in zip(displacement, travel)
            ],
            dtype=np.float64,
        )
        tau_all = _rolling_kendall_tau(c, self._trend_period)
        # tau_all[j] belongs to bar j + trend_period.
        trend_tau = tau_all[trend_index]

        # --- CUSUM, over the instantaneous window plus the decay lookback.
        first_shift = max(0, first_inst - (self._decay_bars - 1))
        cusum_all = _rolling_cusum(log_returns, self._cusum_window, self._cusum_drift)
        # cusum_all[j] ends at bar j + cusum_window.
        shift_index = np.arange(first_shift, n) - self._cusum_window
        shift_extended = cusum_all[shift_index]
        shift_stat = shift_extended[first_inst - first_shift :]

        # --- raw labels, then hysteresis.
        #
        # `shift_peak` is the largest CUSUM statistic in the trailing
        # `shift_decay_bars` bars, which is how `shift_decay_bars` is applied:
        # the trigger persists for that many bars, and a maximum over a
        # trailing window is bounded memory computed from the visible prefix,
        # not detector state.
        decay_windows = np.lib.stride_tricks.sliding_window_view(
            shift_extended, self._decay_bars
        )
        # decay_windows[j] covers the decay_bars bars ending at
        # first_shift + j + decay_bars - 1.
        shift_peak_extended = decay_windows.max(axis=1)
        raw_first_bar = first_shift + self._decay_bars - 1
        raw_bar_start = max(raw_first_bar, self._raw_bars - 1)
        offset = raw_bar_start - first_inst
        if offset < 0:
            raise FeatureError(
                f"{self.name}: raw labels start at bar {raw_bar_start} but "
                f"diagnostics start at {first_inst}; warmup is wrong"
            )
        shift_peak_raw = shift_peak_extended[raw_bar_start - raw_first_bar :]
        raw_code = self._raw_codes(
            vol_pct=vol_pct[offset:],
            efficiency=efficiency[offset:],
            trend_tau=trend_tau[offset:],
            shift_peak=shift_peak_raw,
        )
        code = self._confirm(raw_code)
        run = self._run_lengths(code)

        start = offset + (self._warmup_bars - 1 - raw_bar_start)
        if start < 0:
            raise FeatureError(
                f"{self.name}: confirmed labels would start before warmup "
                f"({start}); the window arithmetic is wrong"
            )
        label_start = start - offset
        return RegimeSpan(
            bar_index=np.arange(self._warmup_bars - 1, n, dtype=np.int64),
            vol_pct=vol_pct[start:],
            trend_tau=trend_tau[start:],
            efficiency=efficiency[start:],
            realized_vol=realized_vol[start:],
            vol_of_vol=vol_of_vol[start:],
            shift_stat=shift_stat[start:],
            shift_peak=shift_peak_raw[label_start:],
            raw_code=raw_code[label_start:],
            code=code[label_start:],
            bars_in_regime=run[label_start:],
        )

    # --- classification -------------------------------------------------

    def _raw_codes(
        self,
        *,
        vol_pct: np.ndarray,
        efficiency: np.ndarray,
        trend_tau: np.ndarray,
        shift_peak: np.ndarray,
    ) -> np.ndarray:
        """Section 6's table, applied in its stated order.

        The order is load-bearing, not cosmetic: a break in the mean is
        almost always also a volatility expansion, so DIRECTIONAL_SHIFT is
        tested first or it would never be reachable -- HIGH_VOL would absorb
        every one of its bars. `test_regime.py` constructs a bar satisfying
        both and asserts the shift wins.

        A window with `efficiency >= trend_efficiency_threshold` and a tau of
        exactly zero has equal concordant and discordant pairs, which is not
        a direction, and falls through to CHOP.
        """
        count = vol_pct.size
        if not (efficiency.size == trend_tau.size == count) or shift_peak.size != count:
            raise FeatureError(
                f"{self.name}: misaligned raw-label inputs "
                f"(vol={vol_pct.size} eff={efficiency.size} tau={trend_tau.size} "
                f"shift={shift_peak.size})"
            )
        out = np.full(count, CODE_BY_REGIME[Regime.CHOP], dtype=np.int16)
        trending = efficiency >= self._trend_eff
        out[trending & (trend_tau > 0.0)] = CODE_BY_REGIME[Regime.TRENDING_UP]
        out[trending & (trend_tau < 0.0)] = CODE_BY_REGIME[Regime.TRENDING_DOWN]
        out[vol_pct <= self._low_vol] = CODE_BY_REGIME[Regime.LOW_VOL]
        out[vol_pct >= self._high_vol] = CODE_BY_REGIME[Regime.HIGH_VOL]
        out[shift_peak > self._cusum_threshold] = CODE_BY_REGIME[Regime.DIRECTIONAL_SHIFT]
        return out

    def _confirm(self, raw_code: np.ndarray) -> np.ndarray:
        """The hysteresis closed form. See the module docstring.

        `confirmed(t)` is the label of the most recently completed run of
        `min_regime_bars` identical raw labels, searched back at most
        `hysteresis_search_bars`, and UNKNOWN when there is none. Pure in
        `raw_code`, so it cannot depend on call order.
        """
        k = self._min_regime_bars
        count = raw_code.size
        out = np.full(count, UNKNOWN_CODE, dtype=np.int16)
        if count < k:
            return out
        runs = np.lib.stride_tricks.sliding_window_view(raw_code, k)
        # completed[j] marks a k-run ending at index j + k - 1.
        completed = (runs == runs[:, :1]).all(axis=1)
        marker = np.full(count, -1, dtype=np.int64)
        marker[k - 1 :] = np.where(completed, np.arange(k - 1, count), -1)
        last = np.maximum.accumulate(marker)
        positions = np.arange(count, dtype=np.int64)
        # A marker older than the horizon is not carried forward: the tape
        # has had no label stable for k bars anywhere in the window.
        visible = last >= (positions - self._search_bars + 1)
        found = visible & (last >= 0)
        out[found] = raw_code[last[found]]
        return out

    def _run_lengths(self, code: np.ndarray) -> np.ndarray:
        """Consecutive identical confirmed labels ending at each position.

        Censored at `hysteresis_search_bars`, and a **floor** rather than an
        exact count: it is read off the span the caller asked for, and
        `classify` asks for a span holding one hysteresis horizon, so a
        regime that has held longer than the horizon reports the horizon.
        A span computed over a longer stretch of history can also report a
        larger number for a bar near its own left edge, where `classify`
        cannot see the markers that would extend the run. Every other field
        of the span is exactly equal between the two paths; this one reads
        as "at least", and `test_regime.py` asserts the inequality rather
        than pretending to an equality that a censored statistic cannot
        have.
        """
        count = code.size
        out = np.zeros(count, dtype=np.int64)
        if count == 0:
            return out
        run = 1
        out[0] = 1
        for i in range(1, count):
            run = run + 1 if code[i] == code[i - 1] else 1
            out[i] = run
        return np.minimum(out, self._search_bars)

    def confidence(self, label: Regime, diagnostics: dict[str, float]) -> float:
        """Normalized margin by which `label`'s condition holds at this bar.

        A margin, **not a probability**: there is no likelihood model here
        and calling it one would invite it to be multiplied by things it
        cannot be multiplied by. Each branch's margin is the distance of the
        deciding statistic past its own threshold, divided by the distance
        from that threshold to the statistic's bound, clamped to [0, 1]:

            DIRECTIONAL_SHIFT  squash(shift_peak - threshold, threshold)
            HIGH_VOL           (vol_pct - high) / (1 - high)
            LOW_VOL            (low - vol_pct) / low
            TRENDING_*         (efficiency - trend_eff) / (1 - trend_eff)
            CHOP               min over the three distances *to* a trigger,
                               each normalized the same way
            UNKNOWN            0.0

        `squash` for the shift because `shift_stat` has no upper bound to
        normalize against; everything else divides by a real distance. The
        shift branch reads `shift_peak` -- the largest statistic inside the
        decay window -- rather than this bar's own, because that peak is
        what put the label there; using the current bar's value would read
        0.0 for every bar of a decaying shift.

        During hysteresis the confirmed label can disagree with the current
        bar's own measurements, and the margin is then negative and clamps
        to 0.0. That is the honest reading and a useful one: it says the
        label is being held by confirmation rather than supported by this
        bar.
        """
        vol = diagnostics["vol_pct"]
        efficiency = diagnostics["efficiency"]
        shift = diagnostics["shift_stat"]
        peak = diagnostics.get("shift_peak", shift)
        if label is Regime.UNKNOWN:
            return 0.0
        if label is Regime.DIRECTIONAL_SHIFT:
            return squash(max(0.0, peak - self._cusum_threshold), self._cusum_threshold)
        if label is Regime.HIGH_VOL:
            return _clamp01((vol - self._high_vol) / (1.0 - self._high_vol))
        if label is Regime.LOW_VOL:
            return _clamp01((self._low_vol - vol) / self._low_vol)
        if label in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
            return _clamp01((efficiency - self._trend_eff) / (1.0 - self._trend_eff))
        # CHOP: how far this bar is from being anything else. Half the band
        # width normalizes the vol term so the midpoint of the tradable band
        # scores 1.0 and either threshold scores 0.0.
        half_band = 0.5 * (self._high_vol - self._low_vol)
        to_vol_edge = min(vol - self._low_vol, self._high_vol - vol) / half_band
        to_trend = (self._trend_eff - efficiency) / self._trend_eff
        to_shift = (self._cusum_threshold - peak) / self._cusum_threshold
        return _clamp01(min(to_vol_edge, to_trend, to_shift))

    def classify(self, view: MarketView) -> RegimeState:
        """The regime at `view.now`, with the full diagnostic vector.

        Takes a `MarketView` rather than section 3's `FeatureVector`. The
        deviation is forced and is recorded rather than worked around:
        hysteresis is a function of the raw label *sequence*, and a single
        bar's feature vector cannot supply the previous bars' labels. Taking
        the view keeps the detector a pure function of point-in-time data,
        which is what the no-state design above requires. The diagnostics it
        emits duplicate three of `VolatilityFeatures`' and one of
        `MomentumFeatures`' by construction, and `test_regime.py` asserts
        they are numerically identical rather than merely similar.
        """
        span = self._span_from_view(view)
        if span is None:
            return RegimeState(
                symbol=view.symbol,
                ts=view.now,
                regime=Regime.UNKNOWN,
                confidence=0.0,
                bars_in_regime=0,
                diagnostics={},
            )
        position = len(span) - 1
        diagnostics = span.at(position)
        label = regime_of_code(int(span.code[position]))
        raw_label = regime_of_code(int(span.raw_code[position]))
        return RegimeState(
            symbol=view.symbol,
            ts=view.now,
            regime=label,
            confidence=self.confidence(label, diagnostics),
            vol_percentile=diagnostics["vol_pct"],
            trend_strength=diagnostics["trend_strength"],
            efficiency_ratio=diagnostics["efficiency"],
            vol_of_vol=diagnostics["vol_of_vol"],
            shift_statistic=diagnostics["shift_stat"],
            bars_in_regime=int(span.bars_in_regime[position]),
            diagnostics={
                **diagnostics,
                "shift_score": squash(diagnostics["shift_stat"], self._cusum_threshold),
                "regime_code": float(span.code[position]),
                "raw_regime_code": float(CODE_BY_REGIME[raw_label]),
            },
        )

    def compute(self, view: MarketView) -> FeatureVector:
        """The diagnostic vector as a `FeatureComputer` output.

        Exists so the detector goes through the same machinery every other
        feature does: `FeatureBundle` key checking, the finite-value guard,
        and `validation.lookahead`, which reads `compute(...).values` and is
        therefore auditing the *label* as well as the measurements behind it.
        `classify` is the richer API; this is the audited one.

        Warmup is re-checked here rather than left to `FeatureBundle`: the
        audit calls computers directly, including at pre-warmup bars.
        """
        if not self.feeds_available(view):
            return self._not_ready(view, "bars feed absent")
        count = view.bar_count()
        if count < self._warmup_bars:
            return self._not_ready(
                view, f"warmup incomplete: {count} of {self._warmup_bars} bars"
            )
        span = self._span_from_view(view)
        if span is None:  # pragma: no cover - the count check above precludes it
            return self._not_ready(view, "no labelled bar in the visible prefix")

        position = len(span) - 1
        diagnostics = span.at(position)
        label = regime_of_code(int(span.code[position]))
        notes: list[str] = []
        if label is Regime.UNKNOWN:
            notes.append(
                f"no run of {self._min_regime_bars} identical raw labels within the "
                f"last {self._search_bars} bars; reported UNKNOWN rather than "
                "carrying a label forward from beyond the horizon"
            )
        return self._vector(
            view,
            {
                **diagnostics,
                "shift_score": squash(diagnostics["shift_stat"], self._cusum_threshold),
                "regime_code": float(span.code[position]),
                "raw_regime_code": float(span.raw_code[position]),
                "regime_confidence": self.confidence(label, diagnostics),
                "bars_in_regime": float(span.bars_in_regime[position]),
            },
            notes=tuple(notes),
        )

    # --- view plumbing --------------------------------------------------

    def _span_from_view(self, view: MarketView) -> RegimeSpan | None:
        """The hysteresis horizon ending at `view.now`, or None before warmup.

        Reads `_classify_bars` bars so the horizon is full whenever enough
        history exists. Shorter near the start of a dataset, which is
        correct: the horizon is "the last `hysteresis_search_bars` raw
        labels", and fewer than that exist there.
        """
        if not self.feeds_available(view):
            return None
        count = view.bar_count()
        if count < self._warmup_bars:
            return None
        n = min(count, self._classify_bars)
        highs = view.column("high", n)
        lows = view.column("low", n)
        closes = view.column("close", n)
        if not (highs.size == lows.size == closes.size == n):
            raise FeatureError(
                f"{self.name}: short OHLC window (high={highs.size} low={lows.size} "
                f"close={closes.size}, expected {n} each)"
            )
        return self.label_span(highs, lows, closes, int(view.primary_interval))


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return 0.0 if value <= 0.0 else (1.0 if value >= 1.0 else float(value))
