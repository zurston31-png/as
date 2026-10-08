"""Adversarial tests for the mathematical regime detector.

`regime/detector.py` landed unvalidated and with no tests. It is the module
ARCHITECTURE.md section 6 specifies, every later phase breaks its results
down by its label, and nothing above it can tell a wrong label from a right
one -- a mislabelled tape still produces plausible per-regime statistics.
Five things in it can be silently wrong, and the tests are organized around
them.

**Accuracy against truth, measured and not tuned.** This is the one the
file exists for. `data/synthetic.py` draws its regime path from a Markov
chain FIRST and then draws prices conditioned on it, so
`SyntheticDataset.regime_labels` is the *cause* of the bars rather than
somebody's reading of a chart. That makes a confusion matrix possible, and
the confusion matrix is the only thing that can falsify the taxonomy. It is
computed once, at the shipped defaults from `load_config().regime`, over
9_460 labelled 5-minute bars, and the measured numbers are written into the
assertions and the comments below. **Overall accuracy is 0.337.** Nothing
in `detector.py` or `defaults.yaml` was moved to improve that number, and
two thresholds this file believes are miscalibrated (`cusum_threshold` and
`trend_efficiency_threshold`) are left exactly where they shipped, with the
evidence recorded in `test_the_shift_threshold_is_a_null_percentile_not_a
_detection_threshold` and `test_the_trend_threshold_barely_separates_a_
trending_tape_from_a_random_walk`. A detector tuned against the data it is
scored on measures nothing; a detector scoring 0.337 and reported honestly
says the section 6 taxonomy is largely *not* recoverable from 5-minute
price alone, which is a result the researcher needs before Phase 8 builds
breakdowns on top of it.

The errors are systematic, not noise, and in three identifiable ways:

1. `vol_pct` is a percentile *rank* in a trailing window, so about
   `low_vol_percentile + (1 - high_vol_percentile)` = 0.40 of all bars sit
   at or beyond one of the two vol thresholds **in every regime, by
   construction**. Measured: 0.505 of bars satisfy one of the two vol
   conditions (both tails run above their nominal quantile because
   `percentile_rank` counts "at or below" and a smoothed ATR produces long
   near-ties), and 0.418 end up labelled LOW_VOL or HIGH_VOL. Since
   section 6 checks the vol branches before the trend branches, those bars
   can never be labelled TRENDING whatever their geometry.
2. DIRECTIONAL_SHIFT over-fires by a factor of six: it claims 0.174 of
   bars against a true share of 0.030, for a precision of 0.030 -- the base
   rate, i.e. no information. `cusum_threshold = 4.0` is the 99.2nd
   percentile of the statistic's iid null, which `shift_decay_bars = 10`
   spreads into a 3.7% per-bar alarm rate before any real structure is
   involved.
3. TRENDING_UP and TRENDING_DOWN have recalls of 0.098 and 0.059 against
   supports of 1_750 and 1_846. The trend condition holds on only 0.306 of
   truly-trending bars (a Kaufman ratio over 20 bars has a driftless-walk
   mean of exactly `1/sqrt(20)` = 0.224, and 0.40 is far into that null's
   right tail), and of the bars where it *does* hold, 0.637 are preempted
   by an earlier branch.

So the directions are both confirmed: CHOP does absorb more than its share
(3_093 predictions against 1_988 true bars, 0.703 of them wrong), and the
trend branches effectively never fire. Neither is tuned away here. Cohen's
kappa over the whole matrix is 0.2017 and balanced accuracy (macro recall)
is 0.3098 against a 0.1667 chance level, which are the two summary numbers
that survive the unbalanced supports.

**Hysteresis as a pure function.** The obvious implementation keeps
`self._label` across calls. That would make a backtest's labels depend on
the order bars were evaluated in, so a re-run of one bar disagrees with the
original run and two backtests differing only in start date produce
different regimes for the same day -- a violation of principle 5 that is
invisible because each run is internally consistent. The module uses a
closed form instead, and the tests verify three separate things: that the
closed form *is* section 6's automaton (simulated independently over all
9_460 measured raw labels), that two computes on one view agree bit for
bit, and that a fresh detector agrees with one already used on other data.
It does. The measured hysteresis lag is exactly `min_regime_bars - 1` = 2
bars over 614 sustained raw-label runs (500 at 2 bars, 114 at 0 because the
confirmed label had not left yet; never 1 and never 3), which is the real
and irreducible cost of the rule. Confirmation also *costs* 1.4 points of
accuracy -- the raw labels score 0.351 -- so `min_regime_bars` has to be
justified by the flapping it prevents and not by the classification.

**Warmup honesty, and the bug it was hiding.** `warmup_bars` was 268 --
`_raw_bars + min_regime_bars - 1`, the point at which a confirmed label
first *exists* -- while `_span_from_view` reads up to `_classify_bars` =
291 bars so the hysteresis horizon is full. Those are not the same number,
and the difference is not cosmetic. 268 bars leaves 3 raw labels in the
horizon instead of 24, and fed exactly three the closed form reports
UNKNOWN -- "not tradable" -- on **1_710 of 9_458 scored bars, 18.1%**,
every one of which has a perfectly good label once the full horizon is
visible. The label at bar `t` therefore depended on how much history the
caller had loaded, which is exactly the path-dependence the stateless
hysteresis design exists to prevent, arriving through the warmup contract
instead of through mutable state and invisible to the lookahead audit
because nothing reads the future. `warmup_bars` is now `_classify_bars` =
291, and both halves are tested: cutting history to 291 moves none of the
thirteen keys at any sampled bar, and the three-label horizon moves 18% of
the labels.

Separately, section 6 asks for the ATR percentile over "a rolling
252-session window" and the implementation uses 252 *bars*. At the
5-minute interval these tests score, that is 3.2 sessions, and the whole
291-bar warmup is 3.7 sessions. The discrepancy is measured in
`test_the_declared_warmup_does_not_cover_a_252_session_window` and
reported, not silently accepted: a vol percentile taken over three
sessions answers "loud for this afternoon" where section 6's answers
"loud for this year".

**A guard that was exactly true only for `np.zeros`.** `cusum_statistic`
refused to standardize a window when `sd <= 0.0`. A window of *identical
non-zero* returns -- a tape stepping a tick a bar, or any stretch whose log
returns round to the same double -- has a floating-point stdev of order
`sqrt(n) * eps * scale` rather than zero, so the guard passed and the
leftover rounding dust was divided by it into standardized increments of
order one. 60 identical returns of 0.01 produced `shift_stat = 29.5`
against a threshold of 4.0, so the highest-precedence label in section 6
fired on a perfectly uniform tape out of pure float error. Fixed with a
relative floor; the zeros case passing is what made it look covered. See
`test_a_window_of_identical_non_zero_returns_is_not_a_break`.

**Branch order.** Section 6's order is load-bearing. A break in the mean is
almost always also a volatility expansion, so if HIGH_VOL were checked
first DIRECTIONAL_SHIFT would be unreachable. Each branch gets a test
driven by a constructed series that triggers exactly it, and the two
overlapping cases (shift with vol in the top decile, trend with vol at an
extreme) get tests that pin which one wins.

**The output contract.** Thirteen keys, every one finite, `required_feeds`
honest down to which *columns* are read, and `RegimeState` carrying the
primary label *and* the whole diagnostic vector -- section 6 requires both,
because regimes are not mutually exclusive in reality and Phase 8 breaks
results down by both.

Builders are local on purpose, and all four rest on one trick: **a flat
close series with a per-bar half-range.** With `close == open == BASE` and
`high/low = BASE +/- u_i`, the true range is exactly `2 * u_i`, every log
return is exactly zero, and therefore `efficiency`, `trend_tau` and
`shift_stat` are all exactly 0.0 while `vol_pct` is a pure function of the
`u` schedule. That decouples the four measurements completely, so each
branch can be aimed at one at a time with arithmetic the test can write
down. No bar anywhere in this file is labelled by hand.
"""

from __future__ import annotations

import inspect
import math
from datetime import date, datetime, timezone
from typing import get_type_hints

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.config.schema import FeatureConfig, RegimeConfig
from flow_model.core.contracts import RegimeState
from flow_model.core.enums import DataQuality, Feed, Regime
from flow_model.data.calendar import TradingCalendar
from flow_model.data.market_view import MarketView
from flow_model.data.series import BarSeries, from_ns, to_ns
from flow_model.data.store import SymbolData
from flow_model.features.base import FeatureError, percentile_rank
from flow_model.features.momentum import MomentumFeatures
from flow_model.features.volatility import (
    VolatilityFeatures,
    bars_per_year,
    true_range,
    wilder_atr_series,
)
from flow_model.regime.detector import (
    CODE_BY_REGIME,
    REGIME_CODES,
    UNKNOWN_CODE,
    RegimeDetector,
    RegimeSpan,
    _rolling_cusum,
    _rolling_kendall_tau,
    cusum_statistic,
    kaufman_efficiency,
    kendall_tau_vs_time,
    regime_of_code,
)
from flow_model.validation.lookahead import assert_no_lookahead, audit_computer

SYMBOL = "NQ"
INTERVAL = 300

#: A plausible instant. Every firewall comparison inside `MarketView` is on
#: int64 ns, so the wall-clock value only has to keep a message readable.
T0 = datetime(2024, 1, 2, 14, 35, tzinfo=timezone.utc)

#: The flat level every constructed series trades at. 100.0 so a 1% jump is
#: exactly 1.0 and the log return is exactly `log(1.01)`.
BASE = 100.0

#: The two half-ranges the alternating builder cycles between: true ranges
#: of 2.0 and 1.0.
U_HI = 1.0
U_LO = 0.5

#: The scored dataset: 5-minute NQ bars, Jan-Jun 2017, seed 7. 9_750 bars,
#: of which 9_460 clear the 291-bar warmup. Long enough that the rarest
#: generated regime (DIRECTIONAL_SHIFT, mean dwell 6 bars) still has 282
#: bars of support.
SCORED_START = date(2017, 1, 1)
SCORED_END = date(2017, 7, 1)
SCORED_SEED = 7


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def detector(regime=None, features=None) -> RegimeDetector:
    config = load_config()
    return RegimeDetector(
        regime if regime is not None else config.regime,
        features if features is not None else config.features,
    )


def grid_stamps(n: int, interval: int = INTERVAL) -> np.ndarray:
    """`n` strictly ascending bar-close stamps on a contiguous grid."""
    step = int(interval) * 1_000_000_000
    return np.array([to_ns(T0) + step * i for i in range(n)], dtype=np.int64)


def series_from(closes: np.ndarray, half_ranges: np.ndarray) -> BarSeries:
    """Bars with `open == close == closes[i]` and `high/low = close +/- u_i`.

    The OHLC ordering `l <= o <= h` and `l <= c <= h` holds for any positive
    `u`, so no builder in this file can produce a series `BarSeries` refuses.
    """
    closes = np.asarray(closes, dtype=np.float64)
    u = np.asarray(half_ranges, dtype=np.float64)
    return BarSeries(
        symbol=SYMBOL,
        ts_ns=grid_stamps(closes.size),
        interval_seconds=INTERVAL,
        columns={
            "open": closes.copy(),
            "high": closes + u,
            "low": closes - u,
            "close": closes.copy(),
            "volume": np.full(closes.size, 1000.0),
        },
    )


def alternating_half_ranges(n: int, hi: float = U_HI, lo: float = U_LO) -> np.ndarray:
    """`u` alternating hi/lo every bar and ending on **lo**.

    Ending on lo is the whole point. With ranges alternating between `2*hi`
    and `2*lo`, Wilder's ATR settles into a two-point cycle, so half the
    252-value percentile window sits on the upper point and half on the
    lower. A final bar on the lower point therefore ranks at about 0.5 --
    inside the tradable band -- while a final bar on the upper point ranks
    at 1.0 and would be HIGH_VOL. The parity is asserted by every test that
    uses this, via the `vol_pct` precondition.
    """
    u = np.empty(int(n), dtype=np.float64)
    u[::2], u[1::2] = hi, lo
    if (int(n) - 1) % 2 == 0:
        u = np.roll(u, 1)
    return u


def flat_series(n: int, half_ranges=None) -> BarSeries:
    """A completely flat tape: every close `BASE`.

    Every log return is exactly 0.0, so `efficiency`, `trend_tau` and
    `shift_stat` are exactly 0.0 and `vol_pct` is the only live
    measurement.
    """
    closes = np.full(int(n), BASE)
    u = alternating_half_ranges(n) if half_ranges is None else np.asarray(half_ranges)
    return series_from(closes, u)


def jump_series(n: int, bars_back: int, factor: float = 1.01) -> BarSeries:
    """A flat tape that steps to `BASE * factor` `bars_back` bars from the end.

    The step is the only non-zero log return in the CUSUM window, which is
    what makes the statistic hand-computable: see
    `test_a_single_step_drives_the_cusum_to_a_hand_computed_value`.
    """
    closes = np.full(int(n), BASE)
    closes[int(n) - 1 - int(bars_back) :] = BASE * float(factor)
    return series_from(closes, alternating_half_ranges(n))


def ramp_series(n: int, per_bar: float) -> BarSeries:
    """Geometric closes, `BASE * (1 + per_bar)**i`, with alternating ranges.

    Geometric rather than linear so every log return is *exactly* equal,
    which drives the CUSUM window's stdev to zero and `shift_stat` to 0.0 by
    `cusum_statistic`'s own zero-dispersion branch. A monotone close series
    has `efficiency == 1.0` and `|trend_tau| == 1.0` exactly, so the trend
    branch fires on the strongest possible evidence while the vol and shift
    branches are pinned out of the way.

    `per_bar` must stay far below `U_LO`, or the per-bar step would exceed
    the half-range and the true range would stop being `2 * u`.
    """
    closes = BASE * (1.0 + float(per_bar)) ** np.arange(int(n), dtype=np.float64)
    return series_from(closes, alternating_half_ranges(n))


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


def empty_view() -> MarketView:
    series = BarSeries.empty(SYMBOL, INTERVAL)
    return MarketView(symbol_data(series), now=T0, now_ns=to_ns(T0))


def span_of(series: BarSeries, comp: RegimeDetector | None = None) -> RegimeSpan:
    comp = comp if comp is not None else detector()
    return comp.label_span(
        series.col("high"), series.col("low"), series.col("close"), INTERVAL
    )


def last_of(series: BarSeries, comp: RegimeDetector | None = None):
    """`(diagnostics, confirmed label, raw label)` at the newest bar."""
    comp = comp if comp is not None else detector()
    span = span_of(series, comp)
    position = len(span) - 1
    return (
        span.at(position),
        regime_of_code(int(span.code[position])),
        regime_of_code(int(span.raw_code[position])),
    )


def truncate(series: BarSeries, keep: int) -> BarSeries:
    """The last `keep` bars, as a series in its own right.

    Used by the warmup tests: cutting history from the *left* is the only
    way to ask whether an answer depends on how much past the caller
    loaded.
    """
    return BarSeries(
        symbol=series.symbol,
        ts_ns=series.ts_ns[-int(keep) :],
        interval_seconds=series.interval_seconds,
        columns={name: series.col(name)[-int(keep) :] for name in series.columns},
    )


def slow_hysteresis(raw_code: np.ndarray, k: int) -> np.ndarray:
    """Section 6's hysteresis as the stateful automaton it is specified as.

    "A regime change requires `min_regime_bars` consecutive qualifying
    bars": hold the current label, count the run of identical raw labels,
    and switch when the count reaches `k`. Written independently and
    deliberately statefully, as the thing `_confirm`'s closed form claims to
    equal. Unbounded -- it has no `hysteresis_search_bars` -- so it agrees
    with the closed form exactly as long as the horizon never binds, which
    is separately asserted.
    """
    out = np.full(raw_code.size, UNKNOWN_CODE, dtype=np.int16)
    label = UNKNOWN_CODE
    run = 0
    previous = None
    for i, value in enumerate(raw_code):
        run = run + 1 if value == previous else 1
        previous = value
        if run >= k:
            label = int(value)
        out[i] = label
    return out


# ---------------------------------------------------------------------------
# the scored dataset and its confusion matrix -- computed once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored():
    """Labels and ground truth over 9_460 bars, plus the confusion matrix.

    Module-scoped because `label_span` rebuilds one Wilder chain per bar and
    costs ~1.4s over this dataset; every accuracy test reads this one pass.
    """
    from flow_model.data import SyntheticConfig, SyntheticMarketGenerator

    config = load_config()
    dataset = SyntheticMarketGenerator(SyntheticConfig(), seed=SCORED_SEED).generate(
        SYMBOL,
        config.spec(SYMBOL),
        SCORED_START,
        SCORED_END,
        INTERVAL,
        calendar=TradingCalendar(),
    )
    comp = RegimeDetector(config.regime, config.features)
    bars = dataset.bars
    span = comp.label_span(
        bars.col("high"), bars.col("low"), bars.col("close"), INTERVAL
    )
    # The generator's own code order is its config's, not this module's, so
    # the codes are translated through the Regime enum rather than assumed
    # to line up. A silent mis-join here would invent an accuracy number.
    truth = np.array(
        [
            CODE_BY_REGIME[Regime(dataset.regime_names[int(code)])]
            for code in dataset.regime_labels[span.bar_index]
        ],
        dtype=np.int64,
    )
    size = len(REGIME_CODES)
    confusion = np.zeros((size, size), dtype=np.int64)
    for actual, predicted in zip(truth, span.code.astype(np.int64)):
        confusion[actual, predicted] += 1
    return {
        "dataset": dataset,
        "data": symbol_data(bars),
        "detector": comp,
        "span": span,
        "truth": truth,
        "confusion": confusion,
    }


def precision_recall(confusion: np.ndarray, label: Regime) -> tuple[float, float, int]:
    index = CODE_BY_REGIME[label]
    support = int(confusion[index].sum())
    predicted = int(confusion[:, index].sum())
    hits = int(confusion[index, index])
    precision = hits / predicted if predicted else float("nan")
    recall = hits / support if support else float("nan")
    return precision, recall, support


# ---------------------------------------------------------------------------
# the integer encoding
# ---------------------------------------------------------------------------


def test_the_encoding_covers_every_regime_the_enum_declares():
    """`regime_code` is a float in a feature vector, so a label missing from
    `REGIME_CODES` would be unrepresentable -- and a weighted score reading
    that float would quietly use whatever the nearest code meant."""
    assert set(REGIME_CODES) == set(Regime)
    assert len(REGIME_CODES) == len(set(REGIME_CODES)) == len(Regime)


def test_unknown_is_code_zero():
    """Zero is what `_not_ready` fills every key with, so "no label" has to
    be the zero code or a pre-warmup vector would read as a tradable
    regime."""
    assert REGIME_CODES[0] is Regime.UNKNOWN
    assert UNKNOWN_CODE == 0
    assert CODE_BY_REGIME[Regime.UNKNOWN] == 0


def test_the_code_order_is_section_sixs_precedence_order():
    """A code is meant to read as a precedence rank, which is only true if
    the tuple is in the classification order. DIRECTIONAL_SHIFT is checked
    first and so must be code 1; CHOP is the fallthrough and so must be
    last."""
    assert REGIME_CODES[1:] == (
        Regime.DIRECTIONAL_SHIFT,
        Regime.HIGH_VOL,
        Regime.LOW_VOL,
        Regime.TRENDING_UP,
        Regime.TRENDING_DOWN,
        Regime.CHOP,
    )


def test_the_reverse_lookup_round_trips():
    """Two tables that disagree would mislabel every bar in one direction
    only, which a confusion matrix would show as a permutation rather than
    as an error."""
    for code, label in enumerate(REGIME_CODES):
        assert CODE_BY_REGIME[label] == code
        assert regime_of_code(code) is label
        assert regime_of_code(float(code)) is label


def test_a_code_outside_the_encoding_is_refused():
    """`regime_code` arrives back as a float from a feature vector. Clamping
    or wrapping an out-of-range one would turn a bug upstream into a
    plausible label."""
    for bad in (-1, len(REGIME_CODES), 99):
        with pytest.raises(FeatureError, match="outside the declared encoding"):
            regime_of_code(bad)


# ---------------------------------------------------------------------------
# Kendall tau-b: the trend statistic, by hand
# ---------------------------------------------------------------------------


def test_kendall_tau_on_a_four_point_zigzag_computed_by_hand():
    """The pair census, written out. y = [1, 3, 2, 4] has six pairs:
    (1,3)+ (1,2)+ (1,4)+ (3,2)- (3,4)+ (2,4)+, so C - D = 5 - 1 = 4.
    n0 = 4*3/2 = 6 and there are no ties, so tau_b = 4 / sqrt(6*6) = 2/3.

    A sign error or an off-by-one in the upper-triangle mask would still
    produce a number in [-1, 1], which is why the value is checked and not
    just the bound."""
    assert kendall_tau_vs_time(np.array([1.0, 3.0, 2.0, 4.0])) == pytest.approx(4.0 / 6.0)


def test_kendall_tau_b_applies_the_tie_correction():
    """y = [1, 2, 2, 3]: pairs (1,2)+ (1,2)+ (1,3)+ (2,2)=0 (2,3)+ (2,3)+,
    so C - D = 5 and n2 = 1 tied pair. tau_b = 5 / sqrt(6 * (6 - 1)) =
    5/sqrt(30) = 0.912871.

    Tau-a would report 5/6 = 0.8333. The difference matters because a
    halted or tick-grid-quantized tape is full of ties, and tau-a would
    read those ties as evidence *against* a trend rather than as no
    evidence."""
    assert kendall_tau_vs_time(np.array([1.0, 2.0, 2.0, 3.0])) == pytest.approx(
        5.0 / math.sqrt(30.0)
    )


def test_kendall_tau_is_exactly_one_on_a_monotone_window():
    """The bound has to be attained, not merely respected: `trend_strength`
    is `|tau|` and feeds a bounded score, so a statistic that could only
    reach 0.9 on a perfect ramp would compress the whole top of the
    range."""
    assert kendall_tau_vs_time(np.arange(1.0, 9.0)) == 1.0
    assert kendall_tau_vs_time(np.arange(9.0, 1.0, -1.0)) == -1.0


def test_kendall_tau_is_rank_based_and_so_ignores_the_size_of_a_step():
    """The reason this module uses tau rather than an OLS slope. Two windows
    with the same ordering must score identically however violent one
    step is, because `data/clean.py` does not catch every bad print and a
    slope-based trend statistic would read one 10% spike as a trend."""
    gentle = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spiked = np.array([1.0, 2.0, 3.0, 4.0, 500.0])
    assert kendall_tau_vs_time(gentle) == kendall_tau_vs_time(spiked) == 1.0


def test_a_halted_tape_has_no_direction_rather_than_a_perfect_one():
    """Every value equal means n0 == n2, so the tau-b denominator is zero.
    Returning 1.0 there -- which a naive `C - D` over a zero denominator
    guarded by a default of 1 would -- labels a halted instrument
    TRENDING_UP and makes it tradable."""
    assert kendall_tau_vs_time(np.full(9, 7.0)) == 0.0


def test_a_window_too_short_to_hold_a_pair_returns_zero():
    """One value has no pair to be concordant about. The classifier reads
    the sign of this, so a nan here becomes a nan comparison and falls
    through to CHOP silently instead of being refused."""
    assert kendall_tau_vs_time(np.array([5.0])) == 0.0
    assert kendall_tau_vs_time(np.zeros(0)) == 0.0


def test_the_rolling_tau_matches_the_scalar_one_window_for_window():
    """`_rolling_kendall_tau` is a chunked broadcast rewrite of the scalar
    function and the module asserts they agree. They have to agree
    *exactly*: `trend_tau > 0.0` is a strict comparison, so a 1e-16
    disagreement at a sign change flips a label."""
    closes = np.array(
        [100.0 + 3.0 * math.sin(i / 4.0) + 0.05 * i for i in range(140)]
    )
    window = 20
    rolled = _rolling_kendall_tau(closes, window)
    scalar = np.array(
        [
            kendall_tau_vs_time(closes[j : j + window + 1])
            for j in range(closes.size - window)
        ]
    )
    assert rolled.size == scalar.size == closes.size - window
    assert np.array_equal(rolled, scalar)


def test_the_rolling_tau_is_chunk_boundary_independent():
    """The chunking exists to bound peak memory. A chunk-dependent answer
    would make the statistic depend on the dataset length, which is the
    reproducibility failure this module is built around."""
    closes = np.array([100.0 + 2.0 * math.cos(i / 3.0) for i in range(90)])
    whole = _rolling_kendall_tau(closes, 20, chunk=4096)
    for chunk in (1, 7, 33):
        assert np.array_equal(_rolling_kendall_tau(closes, 20, chunk=chunk), whole)


def test_the_rolling_tau_returns_nothing_when_no_window_fits():
    """An empty result rather than a short one: a partially-filled window
    would be a tau over fewer bars than declared, reported under the same
    name."""
    assert _rolling_kendall_tau(np.arange(5.0), 20).size == 0


# ---------------------------------------------------------------------------
# Kaufman efficiency: the other trend statistic, by hand
# ---------------------------------------------------------------------------


def test_kaufman_efficiency_computed_by_hand():
    """[10, 11, 10, 13]: displacement |13 - 10| = 3, travel 1 + 1 + 3 = 5,
    so the ratio is 0.6. Section 6 writes exactly this formula, and the only
    way to catch a numerator that used `max - min` (which would give 3/5
    here too, by coincidence) is to also check a window where they differ --
    see the next test."""
    assert kaufman_efficiency(np.array([10.0, 11.0, 10.0, 13.0])) == pytest.approx(0.6)


def test_the_numerator_is_net_displacement_and_not_the_window_range():
    """[10, 20, 10]: displacement |10 - 10| = 0, travel 20, so the ratio is
    0.0. A `max - min` numerator would report 10/20 = 0.5 and label a
    round trip a half-efficient trend."""
    assert kaufman_efficiency(np.array([10.0, 20.0, 10.0])) == 0.0


def test_kaufman_efficiency_is_exactly_one_on_a_monotone_window():
    """1.0 must be attainable: `trend_efficiency_threshold` is compared
    against this with `>=`, and the confidence margin divides by
    `1 - trend_eff`, so a statistic that topped out below 1.0 would never
    report full confidence."""
    assert kaufman_efficiency(np.arange(5.0, 15.0)) == 1.0
    assert kaufman_efficiency(np.arange(15.0, 5.0, -1.0)) == 1.0


def test_a_flat_window_has_no_direction_to_have_been_efficient_about():
    """Zero travel. `0/0` as 1.0 -- which is what a "perfectly efficient"
    reading of a halted tape would be -- makes the trend branch fire on an
    instrument that has not traded."""
    assert kaufman_efficiency(np.full(6, 42.0)) == 0.0


def test_kaufman_efficiency_is_bounded_in_the_unit_interval():
    """Displacement can never exceed travel, so this is a property of the
    arithmetic rather than of the clip. Asserting it over random windows is
    the cheapest guard against a future rewrite that reversed them."""
    generator = np.random.default_rng(4)
    for _ in range(50):
        window = 100.0 + generator.normal(size=21).cumsum()
        assert 0.0 <= kaufman_efficiency(window) <= 1.0


def test_kaufman_efficiency_equals_momentums_published_ratio(scored):
    """`momentum.py` documents that the regime detector reads efficiency over
    exactly `efficiency_period`, and this module's docstring claims the two
    numbers are identical. Two features with one name and different values
    is worse than either: a breakdown by `efficiency_ratio` would not match
    a breakdown by regime."""
    config = load_config()
    momentum = MomentumFeatures(config.features)
    comp, span = scored["detector"], scored["span"]
    bars = scored["dataset"].bars
    for offset in (0, 1, 500, 4000, len(span) - 1):
        bar = int(span.bar_index[offset])
        view = view_at(bars, bar)
        assert momentum.compute(view).values["efficiency_ratio"] == float(
            span.efficiency[offset]
        )
        assert comp.classify(view).efficiency_ratio == float(span.efficiency[offset])


# ---------------------------------------------------------------------------
# the CUSUM: the shift statistic, by hand
# ---------------------------------------------------------------------------


def test_the_cusum_on_an_alternating_window_computed_by_hand():
    """r = [1, -1, 1, -1]. mean 0, stdev(ddof=1) = sqrt(4/3) = 1.154701, so
    z = +/-0.866025. S+ runs 0.366025, 0, 0.366025, 0 and S- runs 0,
    0.366025, 0, 0.366025, so the statistic is max(0, 0.366025) = 0.366025.

    Hand-running both arms matters: a one-sided CUSUM would return 0.0 here
    and would be blind to every downward break."""
    expected = 1.0 / math.sqrt(4.0 / 3.0) - 0.5
    assert cusum_statistic(np.array([1.0, -1.0, 1.0, -1.0]), 0.5) == pytest.approx(
        expected
    )
    assert expected == pytest.approx(0.3660254037844387)


def test_the_cusum_detects_a_downward_break_as_readily_as_an_upward_one():
    """Sign symmetry. The statistic is `max(S+, S-)`, so negating the
    returns must leave it unchanged -- otherwise DIRECTIONAL_SHIFT would be
    a long-only label and the synthetic generator's own per-episode random
    shift sign would halve its recall."""
    returns = np.array([0.1, -0.2, 0.05, 1.4, 0.0, -0.1])
    assert cusum_statistic(returns, 0.3) == pytest.approx(
        cusum_statistic(-returns, 0.3)
    )


def test_a_single_step_drives_the_cusum_to_a_hand_computed_value():
    """`C - 1` zero returns and one step `r` at the end of the window. The
    whole thing collapses to a closed form in `C` alone:

        mean = r/C
        sum of squared deviations = r**2 * (C - 1)/C
        sd  = r / sqrt(C)
        z   = (r - r/C) / (r / sqrt(C)) = (C - 1)/sqrt(C)

    so at `C = 60` the step's own z is 59/sqrt(60) = 7.6168672 and every
    quiet bar's is -1/sqrt(60) = -0.1290994. The negatives cannot lift S+
    off zero, so S+ = 7.6168672 - 0.5 = 7.1168672 -- the value the shift
    branch is tested against, and 1.78x `cusum_threshold`.

    `r` cancelling out entirely is the point: standardizing inside the
    window makes the statistic blind to how big the break was in price
    terms, which is why `shift_stat` is a diagnostic in natural units and
    `shift_score` is the bounded companion.

    (The first draft of this test wrote the stdev out as
    `sqrt((1-1/C)**2 + (C-1)*(1/C)**2)/sqrt(C-1)` and then mis-divided it by
    hand. The expectation was wrong, not the module; reducing it to
    `r/sqrt(C)` left no arithmetic to get wrong.)"""
    window = 60
    returns = np.zeros(window)
    returns[-1] = 0.00995  # log(1.01), the jump builder's step
    sd = returns[-1] / math.sqrt(window)
    assert float(returns.std(ddof=1)) == pytest.approx(sd, rel=1e-12)
    expected = (window - 1) / math.sqrt(window) - 0.5
    assert expected == pytest.approx(7.1168672, abs=1e-7)
    assert cusum_statistic(returns, 0.5) == pytest.approx(expected, rel=1e-12)


def test_the_cusum_decays_by_drift_plus_one_quiet_bars_z_per_bar():
    """After the step leaves the newest position, each further quiet bar
    costs `cusum_drift + 1/sqrt(60) = 0.6290994`. That rate is what makes
    `shift_decay_bars` legible: the raw trigger survives
    ceil((7.1168672 - 4.0) / 0.6290994) = 5 bars on the statistic alone, and
    `shift_decay_bars - 1 = 9` more on the trailing maximum, for 14 bars in
    all -- which `test_the_shift_trigger_persists_for_exactly_fourteen_bars`
    measures end to end."""
    window = 60
    decay = 0.5 + 1.0 / math.sqrt(window)
    assert decay == pytest.approx(0.6290994, abs=1e-7)
    returns = np.zeros(window)
    returns[-1] = 0.00995
    first = cusum_statistic(returns, 0.5)
    shifted = np.zeros(window)
    shifted[-2] = 0.00995
    assert first - cusum_statistic(shifted, 0.5) == pytest.approx(decay, rel=1e-10)
    assert math.ceil((first - 4.0) / decay) == 5


def test_a_window_that_never_moved_has_no_break_in_it():
    """Zero dispersion means there is no standardized scale. Dividing by it
    yields inf, then nan through the recursion, and the `_vector` finite
    guard would raise on a perfectly ordinary halted stretch.

    The second case is the bug this test found -- see
    `test_a_window_of_identical_non_zero_returns_is_not_a_break` for what it
    did and why `sd > 0` was not enough."""
    assert cusum_statistic(np.zeros(60), 0.5) == 0.0
    assert cusum_statistic(np.full(30, 0.004), 0.5) == 0.0


def test_a_window_of_identical_non_zero_returns_is_not_a_break():
    """**The bug.** `cusum_statistic` guarded its standardization with
    `sd <= 0.0`, which is exactly true for `np.zeros` and almost never true
    for anything else. A window of *identical non-zero* returns has a
    floating-point stdev of order `sqrt(n) * eps * scale` -- the mean is
    computed by summation and does not land back on the common value -- so
    the guard passed, the leftover rounding dust was divided by that stdev
    into standardized increments of order one, and the CUSUM accumulated
    them. Measured before the fix: 60 identical returns of 0.01 gave
    `shift_stat = 29.5`, and 30 of 0.004 gave 14.5, against a threshold of
    4.0.

    That is not a rounding wobble in a diagnostic. DIRECTIONAL_SHIFT is the
    highest-precedence branch in section 6, so a uniform tape -- one
    stepping a tick a bar, or any stretch whose log returns round to the
    same double -- was labelled with a mean break out of nothing but float
    error, and that label preempted every other branch. The `np.zeros` case
    passing is what made it look covered.

    Fixed by testing the dispersion against the window's own scale rather
    than against zero: `RELATIVE_DISPERSION_FLOOR = 1e-9`, six orders above
    the rounding floor and four below the ~1e-5 relative spacing tick
    quantization puts between two genuinely different returns."""
    for magnitude in (0.004, 0.01, 1.0, 1e-6):
        for size in (3, 30, 60, 250):
            assert cusum_statistic(np.full(size, magnitude), 0.5) == 0.0, (
                magnitude,
                size,
            )
    # The same window with one genuinely different return is still a break.
    window = np.full(60, 0.004)
    window[-1] = 0.03
    assert cusum_statistic(window, 0.5) > 4.0


def test_a_resolvable_dispersion_is_still_standardized():
    """The floor must not swallow a real window. Tick quantization puts at
    least ~1e-5 relative between two different returns on an index future
    and the floor is 1e-9, so a window whose returns differ by a single tick
    is four orders above it and is standardized exactly as before."""
    from flow_model.regime.detector import RELATIVE_DISPERSION_FLOOR

    assert RELATIVE_DISPERSION_FLOOR == 1e-9
    base = 0.004
    window = np.full(60, base)
    window[-1] = base * (1.0 + 1e-5)
    assert float(window.std(ddof=1)) > RELATIVE_DISPERSION_FLOOR * base
    # The odd return has to be at the *end* of the window: the statistic is
    # the accumulation's value at the newest bar, and 29 quiet bars after a
    # mid-window step decay it back to zero. (My first version put it at
    # index 30 and read 0.0 -- the same mistake, in the opposite direction,
    # as the hand-computed single-step test above.)
    assert cusum_statistic(window, 0.5) == pytest.approx(
        59.0 / math.sqrt(60.0) - 0.5, rel=1e-6
    )


def test_a_window_too_short_for_a_stdev_returns_zero():
    """`ddof=1` over one observation is nan, not a small number."""
    assert cusum_statistic(np.array([0.01]), 0.5) == 0.0
    assert cusum_statistic(np.zeros(0), 0.5) == 0.0


def test_a_shift_spanning_the_whole_window_is_absorbed_into_its_own_mean():
    """The documented and deliberate blind spot, stated as a test so a future
    reader cannot mistake it for a bug. The window standardizes with its own
    mean, so a step that happened before the window opened -- a pure level
    difference -- is invisible. Detecting that would need a second, longer
    baseline window and a second staleness question."""
    inside = np.concatenate([np.zeros(30), np.full(30, 0.01)])
    before = np.full(60, 0.01)
    assert cusum_statistic(inside, 0.5) > 4.0
    # Before the dispersion-floor fix this read 29.5 rather than 0.0, which
    # is how the blind spot stayed hidden: the one window the docstring
    # calls invisible was the loudest alarm in the module.
    assert cusum_statistic(before, 0.5) == 0.0


def test_the_rolling_cusum_matches_the_scalar_one_window_for_window():
    """The vectorized recursion runs across bars instead of along them. The
    module claims bit-identity; `shift_peak > cusum_threshold` is a strict
    comparison, so near the threshold a float discrepancy is a label
    change."""
    generator = np.random.default_rng(11)
    returns = generator.normal(0.0, 0.002, size=300)
    window, drift = 60, 0.5
    rolled = _rolling_cusum(returns, window, drift)
    scalar = np.array(
        [
            cusum_statistic(returns[j : j + window], drift)
            for j in range(returns.size - window + 1)
        ]
    )
    assert rolled.size == scalar.size == returns.size - window + 1
    assert np.array_equal(rolled, scalar)


def test_the_rolling_cusum_zeroes_a_zero_dispersion_window_too():
    """The vectorized path divides by a substituted stdev of 1.0 to keep the
    broadcast shape, then masks. A missing mask would emit the raw
    accumulation of unstandardized returns, which for a flat window is an
    accumulation of `-drift` -- clamped to zero by luck, not by design.
    Checked on a window that is flat but not zero."""
    returns = np.concatenate([np.full(70, 0.003), np.zeros(5)])
    rolled = _rolling_cusum(returns, 60, 0.5)
    assert rolled[0] == 0.0
    assert np.all(np.isfinite(rolled))


def test_the_rolling_cusum_returns_nothing_when_no_window_fits():
    assert _rolling_cusum(np.zeros(10), 60, 0.5).size == 0


# ---------------------------------------------------------------------------
# the measurements the span reports, and their identity with the published ones
# ---------------------------------------------------------------------------


def test_vol_pct_is_volatilitys_published_atr_percentile(scored):
    """`atr_percentile` is the most-read number in the system and this module
    recomputes it rather than importing the value. Identical, not similar:
    `high_vol_percentile` is compared with `>=`, so a 1e-16 difference at
    0.80 is a different label, and a *breakdown* by `atr_percentile` that
    disagreed with the regime it supposedly produced would be unfalsifiable
    rather than wrong."""
    config = load_config()
    volatility = VolatilityFeatures(config.features)
    span, bars = scored["span"], scored["dataset"].bars
    for offset in (0, 1, 37, 2500, 9000, len(span) - 1):
        bar = int(span.bar_index[offset])
        assert volatility.compute(view_at(bars, bar)).values["atr_percentile"] == float(
            span.vol_pct[offset]
        )


def test_realized_vol_and_vol_of_vol_are_volatilitys_published_values(scored):
    """Same argument, and `vol_of_vol` additionally records a deviation from
    section 6 -- it is a coefficient of variation, not a bare stdev. Reusing
    the landed definition is the choice; the test is that it is *actually*
    reused rather than re-derived into a second number with one name."""
    config = load_config()
    volatility = VolatilityFeatures(config.features)
    span, bars = scored["span"], scored["dataset"].bars
    for offset in (0, 123, 5000, len(span) - 1):
        bar = int(span.bar_index[offset])
        values = volatility.compute(view_at(bars, bar)).values
        assert values["realized_vol"] == float(span.realized_vol[offset])
        assert values["vol_of_vol"] == float(span.vol_of_vol[offset])


def test_the_annualization_is_volatilitys_bars_per_year():
    """`realized_vol` is a diagnostic in natural units and is plotted
    alongside `VolatilityFeatures`'. Two annualization constants would make
    the two series differ by a fixed factor that looks like a model
    difference."""
    assert bars_per_year(INTERVAL) == pytest.approx(252.0 * 6.5 * 3600.0 / 300.0)
    span = span_of(flat_series(300))
    assert np.all(span.realized_vol == 0.0)  # a flat tape has no realized vol


def test_vol_pct_is_the_rank_of_the_atr_chain_this_module_rebuilds():
    """The per-bar anchoring claim, checked against the arithmetic rather
    than against the other module. At bar `t`, `vol_pct` must be the rank of
    the last of `atr_percentile_lookback` ATR values from a Wilder chain
    seeded inside the `atr_period + atr_percentile_lookback` bars ending at
    `t` -- not a chain seeded at the start of whatever span the caller
    asked for."""
    config = load_config()
    comp = detector()
    series = flat_series(comp.warmup_bars + 60)
    span = span_of(series, comp)
    chain = config.features.atr_period + config.features.atr_percentile_lookback
    for offset in (0, 17, len(span) - 1):
        end = int(span.bar_index[offset]) + 1
        ranges = true_range(
            series.col("high")[end - chain : end],
            series.col("low")[end - chain : end],
            series.col("close")[end - chain : end],
        )
        atr = wilder_atr_series(ranges, config.features.atr_period)
        window = atr[-config.features.atr_percentile_lookback :]
        assert window.size == config.features.atr_percentile_lookback
        assert float(span.vol_pct[offset]) == percentile_rank(window, float(atr[-1]))


def test_a_halted_tape_reports_the_lowest_vol_rather_than_the_highest():
    """Every ATR zero means every value ties, and "fraction at or below"
    would rank a zero at 1.0 among zeros -- labelling a halted instrument
    HIGH_VOL, the exact inverse of the truth, and making it the most
    aggressively traded thing in the book."""
    comp = detector()
    series = series_from(
        np.full(comp.warmup_bars, BASE), np.zeros(comp.warmup_bars)
    )
    diagnostics, label, _ = last_of(series, comp)
    assert diagnostics["vol_pct"] == 0.0
    assert label is Regime.LOW_VOL


def test_the_flat_builder_pins_three_of_the_four_measurements_at_zero():
    """The precondition every branch test below depends on. A flat close
    series has zero log returns, so `efficiency`, `trend_tau` and
    `shift_stat` are *exactly* zero and `vol_pct` is the only live input.
    If this stopped holding, the vol-branch tests would be passing for an
    unknown reason."""
    diagnostics, _, _ = last_of(flat_series(detector().warmup_bars))
    assert diagnostics["efficiency"] == 0.0
    assert diagnostics["trend_tau"] == 0.0
    assert diagnostics["trend_strength"] == 0.0
    assert diagnostics["shift_stat"] == 0.0
    assert diagnostics["shift_peak"] == 0.0
    assert diagnostics["realized_vol"] == 0.0


def test_the_alternating_builder_lands_inside_the_tradable_band():
    """The other precondition: ranges cycling between 2.0 and 1.0 put
    Wilder's ATR into a two-point cycle, so a final bar on the lower point
    ranks at about half the 252-value window. Measured 0.5000 exactly, which
    is as hand-computable as this gets -- 126 of 252 values at or below.
    The opposite parity ranks at 1.0, which the next test uses."""
    diagnostics, label, _ = last_of(flat_series(detector().warmup_bars))
    assert diagnostics["vol_pct"] == pytest.approx(0.5, abs=0.01)
    config = load_config()
    assert config.regime.low_vol_percentile < diagnostics["vol_pct"]
    assert diagnostics["vol_pct"] < config.regime.high_vol_percentile
    assert label is Regime.CHOP


# ---------------------------------------------------------------------------
# one test per classification branch
# ---------------------------------------------------------------------------


def test_the_chop_branch_is_the_fallthrough():
    """Nothing triggered: vol inside the band, efficiency 0, no shift. CHOP
    is the only branch with no condition of its own, so it is also the only
    one that can absorb a bug in any of the others -- which is what the
    confusion matrix below reports it doing."""
    diagnostics, label, raw = last_of(flat_series(detector().warmup_bars + 20))
    assert (label, raw) == (Regime.CHOP, Regime.CHOP)
    assert diagnostics["shift_peak"] == 0.0
    assert diagnostics["efficiency"] == 0.0


def test_the_low_vol_branch_fires_on_a_contracting_range():
    """Half-ranges shrinking monotonically make the final ATR the smallest
    value in its own percentile window, so `vol_pct` is 1/252 = 0.003968 --
    below `low_vol_percentile` by a wide margin, and hand-computable because
    exactly one of the 252 values (the last) is at or below itself."""
    comp = detector()
    n = comp.warmup_bars + 20
    series = series_from(np.full(n, BASE), np.linspace(2.0, 0.5, n))
    diagnostics, label, raw = last_of(series, comp)
    assert diagnostics["vol_pct"] == pytest.approx(1.0 / 252.0)
    assert (label, raw) == (Regime.LOW_VOL, Regime.LOW_VOL)


def test_the_high_vol_branch_fires_on_an_expanding_range():
    """The mirror image: monotonically growing half-ranges make the final ATR
    the largest, so `vol_pct` is exactly 1.0 -- all 252 values at or below
    it."""
    comp = detector()
    n = comp.warmup_bars + 20
    series = series_from(np.full(n, BASE), np.linspace(0.5, 2.0, n))
    diagnostics, label, raw = last_of(series, comp)
    assert diagnostics["vol_pct"] == 1.0
    assert (label, raw) == (Regime.HIGH_VOL, Regime.HIGH_VOL)


def test_the_trending_up_branch_fires_on_a_monotone_ramp():
    """A geometric ramp has `efficiency == 1.0` and `trend_tau == 1.0`
    exactly, and -- because every log return is *identical* -- a CUSUM
    window with zero dispersion and therefore `shift_stat == 0.0`. The
    alternating half-ranges hold `vol_pct` in the band. So only the trend
    branch can fire, and the direction comes from the sign of the same tau
    that supplied the strength."""
    comp = detector()
    diagnostics, label, raw = last_of(ramp_series(comp.warmup_bars + 20, 1e-4), comp)
    assert diagnostics["efficiency"] == 1.0
    assert diagnostics["trend_tau"] == 1.0
    assert diagnostics["shift_stat"] == 0.0
    assert (label, raw) == (Regime.TRENDING_UP, Regime.TRENDING_UP)


def test_the_trending_down_branch_fires_on_a_monotone_decline():
    """Same series negated in log space. Strength and direction come from one
    statistic, so they cannot disagree -- which an OLS slope paired with a
    separate tau could, leaving the classifier with no defined answer."""
    comp = detector()
    diagnostics, label, raw = last_of(ramp_series(comp.warmup_bars + 20, -1e-4), comp)
    assert diagnostics["efficiency"] == 1.0
    assert diagnostics["trend_tau"] == -1.0
    assert (label, raw) == (Regime.TRENDING_DOWN, Regime.TRENDING_DOWN)


def test_the_directional_shift_branch_fires_on_a_single_step():
    """One 1% step three bars from the end. `shift_peak` is
    59/sqrt(60) - 0.5 = 7.1168672 by the hand computation above, against a
    threshold of 4.0, and the step has been the newest-window event for four
    bars so hysteresis has confirmed it."""
    comp = detector()
    diagnostics, label, raw = last_of(jump_series(comp.warmup_bars + 20, 3), comp)
    assert diagnostics["shift_peak"] == pytest.approx(
        59.0 / math.sqrt(60.0) - 0.5, rel=1e-10
    )
    assert (label, raw) == (Regime.DIRECTIONAL_SHIFT, Regime.DIRECTIONAL_SHIFT)


def test_the_unknown_branch_fires_when_no_label_holds_for_min_regime_bars():
    """UNKNOWN is the hysteresis fallback, not a classification branch, and
    it is unreachable from any series this file can build -- on the scored
    dataset it fires on zero of 9_460 bars. So it is tested where it lives:
    a raw sequence that alternates every bar completes no run of
    `min_regime_bars`, and every bar must report UNKNOWN rather than
    carrying a label forward from beyond the horizon."""
    comp = detector()
    raw = np.array([2, 3] * 10, dtype=np.int16)
    assert np.all(comp._confirm(raw) == UNKNOWN_CODE)


def test_a_tau_of_exactly_zero_is_not_a_direction_and_falls_to_chop():
    """Section 6 gives TRENDING_UP `slope > 0` and TRENDING_DOWN `slope < 0`
    and names no branch for zero. Equal concordant and discordant pairs is
    not a direction, so such a bar must fall through rather than be assigned
    to whichever branch happens to be written with `>=`."""
    comp = detector()
    codes = comp._raw_codes(
        vol_pct=np.array([0.5]),
        efficiency=np.array([1.0]),
        trend_tau=np.array([0.0]),
        shift_peak=np.array([0.0]),
    )
    assert regime_of_code(int(codes[0])) is Regime.CHOP


def test_the_trend_branch_is_inclusive_at_its_threshold_and_vol_exclusive_inside():
    """Section 6 writes `efficiency >= trend_eff`, `vol_pct >= high_vol_pct`
    and `vol_pct <= low_vol_pct`. All three are inclusive, so a bar sitting
    exactly on a threshold is *in* the branch. Off-by-one-epsilon here is
    invisible on real data and changes the band the CHOP confidence margin
    normalizes against."""
    config = load_config()
    comp = detector()
    eff, high, low = (
        config.regime.trend_efficiency_threshold,
        config.regime.high_vol_percentile,
        config.regime.low_vol_percentile,
    )
    codes = comp._raw_codes(
        vol_pct=np.array([0.5, 0.5, high, low, high - 1e-9, low + 1e-9]),
        efficiency=np.array([eff, eff - 1e-9, 0.0, 0.0, 0.0, 0.0]),
        trend_tau=np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        shift_peak=np.zeros(6),
    )
    assert [regime_of_code(int(c)) for c in codes] == [
        Regime.TRENDING_UP,
        Regime.CHOP,
        Regime.HIGH_VOL,
        Regime.LOW_VOL,
        Regime.CHOP,
        Regime.CHOP,
    ]


def test_the_shift_branch_is_strict_at_its_threshold():
    """Section 6 writes `shift_stat > cusum_threshold`, strictly. The
    statistic's value *at* the threshold is reachable on a flat tape (both
    are 0 only when the threshold is 0), and more importantly the
    DIRECTIONAL_SHIFT confidence margin is `squash(peak - threshold,
    threshold)`, which is 0.0 exactly at the boundary -- a label with zero
    confidence should not have been assigned."""
    config = load_config()
    comp = detector()
    threshold = config.regime.cusum_threshold
    codes = comp._raw_codes(
        vol_pct=np.full(3, 0.5),
        efficiency=np.zeros(3),
        trend_tau=np.zeros(3),
        shift_peak=np.array([threshold, threshold + 1e-9, threshold - 1e-9]),
    )
    assert [regime_of_code(int(c)) for c in codes] == [
        Regime.CHOP,
        Regime.DIRECTIONAL_SHIFT,
        Regime.CHOP,
    ]


def test_the_trend_strength_threshold_is_inert_configuration():
    """`RegimeConfig.trend_strength_threshold` (0.25, and in
    `defaults.yaml`) is read by nothing. Section 6 names `trend_strength` as
    a *measurement* but its classification table thresholds `efficiency` and
    the slope only, so there is no branch for this number to govern -- and
    the detector does not invent one.

    Asserted rather than left implicit for two reasons. A researcher tuning
    thresholds will reach for it, and a knob that silently does nothing is
    worse than one that is absent -- it looks like the trend branch has been
    tried at a second setting when it has not. And if a future change *does*
    wire it in, this test fails and forces the decision to be deliberate
    rather than arriving as a side effect.

    A tightened `trend_strength_threshold` would be the natural fix for the
    trend branch's 0.44 precision: requiring `|tau| >= 0.25` as well as
    `efficiency >= 0.40` would drop the weakest-ordered windows. That is a
    change to section 6's table, not a threshold move, so it is reported and
    not made."""
    config = load_config()
    assert config.regime.trend_strength_threshold == 0.25
    comp = detector()
    series = ramp_series(comp.warmup_bars + 20, 1e-4)
    baseline = comp.compute(view_at(series)).values
    assert baseline["trend_strength"] == 1.0
    for value in (0.0, 0.99):
        other = detector(
            regime=RegimeConfig(
                **{**config.regime.model_dump(), "trend_strength_threshold": value}
            )
        )
        assert other.compute(view_at(series)).values == baseline
    # Nothing in the module's own source mentions it either.
    import inspect as _inspect

    import flow_model.regime.detector as module

    assert "trend_strength_threshold" not in _inspect.getsource(module)


def test_misaligned_raw_label_inputs_are_refused():
    """Four arrays that must be the same length and are sliced from four
    different index computations. A silent broadcast would label bars with
    another bar's measurements, which produces a plausible regime series and
    no error anywhere."""
    comp = detector()
    with pytest.raises(FeatureError, match="misaligned raw-label inputs"):
        comp._raw_codes(
            vol_pct=np.zeros(5),
            efficiency=np.zeros(4),
            trend_tau=np.zeros(5),
            shift_peak=np.zeros(5),
        )


# ---------------------------------------------------------------------------
# branch order: the part of section 6 that is load-bearing
# ---------------------------------------------------------------------------


def test_directional_shift_preempts_high_vol_when_both_conditions_hold():
    """The ordering section 6 calls out, and the reason it is called out: a
    break in the mean is almost always also a volatility expansion, so if
    HIGH_VOL were tested first DIRECTIONAL_SHIFT would be unreachable --
    every one of its bars would be absorbed. The series makes both
    conditions true at once (growing ranges put `vol_pct` at exactly 1.0,
    a step puts `shift_peak` at 7.117) and the shift must win."""
    config = load_config()
    comp = detector()
    n = comp.warmup_bars + 20
    closes = np.full(n, BASE)
    closes[n - 4 :] = BASE * 1.01
    series = series_from(closes, np.linspace(0.5, 2.0, n))
    diagnostics, label, raw = last_of(series, comp)
    assert diagnostics["vol_pct"] >= config.regime.high_vol_percentile
    assert diagnostics["shift_peak"] > config.regime.cusum_threshold
    assert (label, raw) == (Regime.DIRECTIONAL_SHIFT, Regime.DIRECTIONAL_SHIFT)


def test_directional_shift_preempts_low_vol_too():
    """The same ordering on the other vol extreme. Worth its own test because
    a contracting range is the case where a break is *most* informative --
    a step out of a quiet stretch -- and where an implementation that
    assigned LOW_VOL last would silently relabel it."""
    config = load_config()
    comp = detector()
    n = comp.warmup_bars + 20
    closes = np.full(n, BASE)
    closes[n - 4 :] = BASE * 1.01
    series = series_from(closes, np.linspace(2.0, 0.5, n))
    diagnostics, label, _ = last_of(series, comp)
    assert diagnostics["vol_pct"] <= config.regime.low_vol_percentile
    assert diagnostics["shift_peak"] > config.regime.cusum_threshold
    assert label is Regime.DIRECTIONAL_SHIFT


def test_a_vol_extreme_preempts_a_perfect_trend():
    """Section 6 lists the vol branches *before* the trend branches, so a
    monotone ramp whose ATR percentile is at an extreme is labelled by its
    vol and not by its trend. This is the single largest systematic error in
    the confusion matrix below -- it is also what section 6 specifies, so it
    is pinned here as behaviour and reported there as a cost."""
    comp = detector()
    n = comp.warmup_bars + 20
    closes = BASE * (1.0 + 1e-4) ** np.arange(n)
    for schedule, expected in (
        (np.linspace(0.5, 2.0, n), Regime.HIGH_VOL),
        (np.linspace(2.0, 0.5, n), Regime.LOW_VOL),
    ):
        diagnostics, label, _ = last_of(series_from(closes, schedule), comp)
        assert diagnostics["efficiency"] == 1.0
        assert abs(diagnostics["trend_tau"]) == 1.0
        assert label is expected


def test_the_schema_makes_the_two_vol_conditions_mutually_exclusive():
    """Which is why the order *between* HIGH_VOL and LOW_VOL is unreachable
    through legal configuration: `low >= high` is refused, so no `vol_pct`
    can satisfy both. Worth asserting, because it is the guard that makes
    `model_construct` the only way into the next test's branch."""
    with pytest.raises(ValueError, match="low_vol_percentile must be below"):
        RegimeConfig(low_vol_percentile=0.6, high_vol_percentile=0.4)
    with pytest.raises(ValueError, match="low_vol_percentile must be below"):
        RegimeConfig(low_vol_percentile=0.5, high_vol_percentile=0.5)


def test_high_vol_wins_over_low_vol_if_the_schema_guard_is_ever_relaxed():
    """Section 6 lists HIGH_VOL before LOW_VOL, and the implementation
    assigns LOW_VOL and then overwrites it -- so that precedence lives in
    statement order and nothing legal can reach it. `model_construct` skips
    validation to get there, which is exactly when a silently wrong order
    would be most damaging: a future widening of the schema would otherwise
    flip every overlapping bar with no test failing.

    My first attempt built the inverted config directly and died in
    pydantic. That expectation was wrong, not the schema."""
    inverted = RegimeConfig.model_construct(
        **{
            **RegimeConfig().model_dump(),
            "low_vol_percentile": 0.6,
            "high_vol_percentile": 0.4,
        }
    )
    assert inverted.low_vol_percentile > inverted.high_vol_percentile
    comp = detector(regime=inverted)
    codes = comp._raw_codes(
        vol_pct=np.array([0.5]),
        efficiency=np.zeros(1),
        trend_tau=np.zeros(1),
        shift_peak=np.zeros(1),
    )
    assert regime_of_code(int(codes[0])) is Regime.HIGH_VOL


def test_the_shift_trigger_persists_for_exactly_fourteen_bars():
    """How `shift_decay_bars` is applied, measured. The statistic itself
    clears 4.0 for 5 bars (7.117157 falling by 0.629099 a bar), and the
    trailing maximum over `shift_decay_bars = 10` extends that by 9, so the
    raw label is DIRECTIONAL_SHIFT for 5 + 9 = 14 consecutive bars and then
    stops. A trailing maximum is bounded memory computed from the visible
    prefix -- not detector state -- which is why it is allowed here at
    all."""
    comp = detector()
    n = comp.warmup_bars + 60
    series = jump_series(n, 40)
    span = span_of(series, comp)
    shift_code = CODE_BY_REGIME[Regime.DIRECTIONAL_SHIFT]
    firing = np.flatnonzero(span.raw_code == shift_code)
    assert firing.size == 14
    assert np.array_equal(firing, np.arange(firing[0], firing[0] + 14))
    # 5 bars from the statistic: (7.117157 - 4.0) / 0.629099 = 4.95 -> 5.
    assert int((span.shift_stat > load_config().regime.cusum_threshold).sum()) == 5


# ---------------------------------------------------------------------------
# hysteresis
# ---------------------------------------------------------------------------


def test_the_closed_form_is_section_sixs_automaton_on_a_hand_sequence():
    """k = 3. raw = [A, A, A, B, B, B]: the automaton switches to A at index
    2 (its first completed 3-run), holds A at 3 and 4 while B is still
    accumulating, and switches to B at 5. Indices 0 and 1 have no completed
    run behind them at all and are UNKNOWN."""
    comp = detector(regime=RegimeConfig(min_regime_bars=3, hysteresis_search_bars=24))
    out = comp._confirm(np.array([2, 2, 2, 3, 3, 3], dtype=np.int16))
    assert list(out) == [UNKNOWN_CODE, UNKNOWN_CODE, 2, 2, 2, 3]


def test_a_single_bar_flap_is_suppressed():
    """`min_regime_bars` exists for exactly this. raw = [A,A,A,B,A,A,A,A]:
    the lone B never completes a 3-run, so the confirmed label stays A
    across it and the single bar never reaches a backtest."""
    comp = detector(regime=RegimeConfig(min_regime_bars=3, hysteresis_search_bars=24))
    out = comp._confirm(np.array([2, 2, 2, 3, 2, 2, 2, 2], dtype=np.int16))
    assert list(out) == [UNKNOWN_CODE, UNKNOWN_CODE] + [2] * 6


def test_a_two_bar_flap_is_suppressed_and_a_three_bar_run_is_not():
    """The boundary of the rule, at `min_regime_bars = 3`. Two bars of B
    hold; the third switches. Checked as a pair so an off-by-one in either
    direction fails."""
    comp = detector(regime=RegimeConfig(min_regime_bars=3, hysteresis_search_bars=24))
    two = comp._confirm(np.array([2, 2, 2, 3, 3, 2, 2, 2], dtype=np.int16))
    three = comp._confirm(np.array([2, 2, 2, 3, 3, 3, 2, 2], dtype=np.int16))
    # [A,A,A,B,B,A,A,A]: B never completes a 3-run, so A holds throughout.
    assert list(two)[2:] == [2] * 6
    # [A,A,A,B,B,B,A,A]: B completes at index 5, and the *trailing* two A
    # bars are themselves only a 2-run, so B holds to the end. My first
    # expectation had it reverting to A at index 6 -- which would have been
    # the rule applied in one direction only, and would have made a 2-bar
    # flap suppressible on the way in but not on the way out.
    assert list(three)[2:] == [2, 2, 2, 3, 3, 3]


def test_a_single_bar_raw_shift_is_suppressed_end_to_end():
    """The flap rule through the whole stack rather than through `_confirm`.
    A 1% step on the *final* bar makes exactly one bar's raw label
    DIRECTIONAL_SHIFT -- the step has not yet been in the window long enough
    for the trailing maximum to cover a second bar -- and the confirmed
    label must stay CHOP. Two bars of it must also be suppressed, and three
    must not: that is `min_regime_bars` measured on a price series."""
    comp = detector()
    n = comp.warmup_bars + 20
    shift_code = CODE_BY_REGIME[Regime.DIRECTIONAL_SHIFT]
    observed = []
    for bars_back in (0, 1, 2):
        span = span_of(jump_series(n, bars_back), comp)
        position = len(span) - 1
        assert int(span.raw_code[position]) == shift_code
        assert int((span.raw_code == shift_code).sum()) == bars_back + 1
        observed.append(regime_of_code(int(span.code[position])))
    assert observed == [Regime.CHOP, Regime.CHOP, Regime.DIRECTIONAL_SHIFT]


def test_the_closed_form_agrees_with_the_stateful_automaton_on_every_scored_bar(scored):
    """The load-bearing claim, checked against an independent and
    deliberately *stateful* implementation of section 6's wording over all
    9_460 measured raw labels. The closed form exists so that the label
    sequence is a pure function of the visible prefix; it is only allowed to
    exist if it computes the same thing the specification describes."""
    comp = scored["detector"]
    span = scored["span"]
    expected = slow_hysteresis(span.raw_code, comp._min_regime_bars)
    # The automaton has to be seeded, and the span is a *slice*: its first
    # bar already has 25 raw labels behind it that the slice does not carry,
    # so the automaton reads UNKNOWN there where the detector (which sees
    # them) does not. Everything from the automaton's own first completed
    # run onward depends only on raw labels inside the slice, and is
    # therefore seed-independent and comparable. My first version compared
    # from bar zero and blamed the module for the seed.
    started = int(np.flatnonzero(expected != UNKNOWN_CODE)[0])
    assert started == comp._min_regime_bars - 1 == 2
    assert np.array_equal(span.code[started:], expected[started:])
    assert span.code.size - started == 9_458


def test_the_hysteresis_horizon_never_binds_on_the_scored_dataset(scored):
    """Which is why the previous test can compare against an *unbounded*
    automaton. Measured: UNKNOWN on 0 of 9_460 bars -- with
    `hysteresis_search_bars = 24` and `min_regime_bars = 3` some 3-run
    always exists inside the horizon. Asserted as a small rate rather than
    as zero, because zero is a property of this tape and a loose bound is
    what a future change should have to break."""
    span = scored["span"]
    rate = float((span.code == UNKNOWN_CODE).mean())
    assert rate == 0.0  # measured: 0 of 9460
    assert rate <= 0.02


def test_the_measured_hysteresis_lag_is_exactly_min_regime_bars_minus_one(scored):
    """The real cost of the rule, measured rather than asserted from theory.
    Over the 614 raw-label runs on the scored dataset that last at least
    `min_regime_bars`, the confirmed label reaches the new raw label after
    exactly 2 bars -- `min_regime_bars - 1` -- on 500 of them, and after 0
    bars on the other 114, which are the runs where the confirmed label
    already *was* that label (the raw series flapped away and back). Mean
    1.629, median 2, max 2. No run takes 1 bar or 3: the lag is the rule's
    own constant and not a distribution.

    It cannot be zero. Two bars at a 5-minute interval is ten minutes of
    holding a label the tape has already left, and that is worth knowing
    before Phase 8 attributes a loss to a regime."""
    comp, span = scored["detector"], scored["span"]
    raw, code = span.raw_code, span.code
    k = comp._min_regime_bars
    lags = []
    start = 0
    while start < raw.size:
        stop = start
        while stop + 1 < raw.size and raw[stop + 1] == raw[start]:
            stop += 1
        if stop - start + 1 >= k and start > 0:
            hit = next(
                (i for i in range(start, stop + 1) if code[i] == raw[start]), None
            )
            if hit is not None:
                lags.append(hit - start)
        start = stop + 1
    lags = np.array(lags)
    assert lags.size == 614
    assert lags.max() == k - 1 == 2
    assert set(np.unique(lags).tolist()) == {0, k - 1}
    assert int((lags == k - 1).sum()) == 500
    assert int((lags == 0).sum()) == 114
    assert lags.mean() == pytest.approx(1.629, abs=0.005)


def test_bars_in_regime_is_censored_at_the_hysteresis_horizon(scored):
    """A run length read off an uncensored history would depend on how much
    the caller happened to load, which is the non-reproducibility the whole
    design avoids. It is a floor, and the censor is the horizon."""
    comp, span = scored["detector"], scored["span"]
    assert span.bars_in_regime.min() >= 1
    assert span.bars_in_regime.max() == comp._search_bars
    assert np.all(span.bars_in_regime <= comp._search_bars)


# ---------------------------------------------------------------------------
# statefulness -- the risk that would make every backtest path-dependent
# ---------------------------------------------------------------------------


def test_two_computes_on_the_identical_view_agree_bit_for_bit():
    """A detector holding `self._label` would pass this and fail the next
    test, so both are needed. This one also catches a computer that read a
    clock or an unseeded generator."""
    comp = detector()
    view = view_at(ramp_series(comp.warmup_bars + 30, 1e-4))
    first, second = comp.compute(view), comp.compute(view)
    assert first.values == second.values
    assert first.notes == second.notes
    assert first.quality_by_key == second.quality_by_key


def test_a_fresh_detector_agrees_with_one_already_used_on_other_data():
    """The test that actually catches remembered state. A detector that
    carried its last label across calls would answer differently here
    depending on what it had been shown before, so a backtest's labels would
    depend on the order bars were evaluated in, a re-run of one bar would
    disagree with the original run, and two backtests differing only in
    start date would report different regimes for the same day. Principle 5
    requires bit-identical results from the same config and data."""
    config = load_config()
    series = flat_series(config.regime and 340)
    target = view_at(series)
    fresh = RegimeDetector(config.regime, config.features)
    used = RegimeDetector(config.regime, config.features)
    # Walk the used one through deliberately contradictory tape first.
    for other in (
        ramp_series(340, 1e-4),
        ramp_series(340, -1e-4),
        jump_series(340, 2),
    ):
        used.compute(view_at(other))
    assert used.compute(target).values == fresh.compute(target).values
    assert used.classify(target) == fresh.classify(target)


def test_evaluating_bars_backwards_gives_the_same_labels_as_forwards():
    """Order independence stated directly. A stateful detector is
    self-consistent in any single pass, so the only way to see the failure
    is to run the same bars in two orders and compare."""
    comp = detector()
    series = jump_series(comp.warmup_bars + 40, 20)
    bars = list(range(comp.warmup_bars - 1, len(series)))
    forwards = {t: comp.compute(view_at(series, t)).values for t in bars}
    backwards = {t: comp.compute(view_at(series, t)).values for t in reversed(bars)}
    assert forwards == backwards


def test_classify_equals_a_whole_series_pass_bar_by_bar(scored):
    """`classify` reads the last `_classify_bars` bars; `label_span` reads
    everything. The module claims they agree at the same bar, and analytics
    uses the span while the live path uses `classify` -- a disagreement would
    mean the research and the production label are different features."""
    comp, span, bars = scored["detector"], scored["span"], scored["dataset"].bars
    offsets = [0, 1, 2, len(span) // 2, len(span) - 2, len(span) - 1]
    for offset in offsets:
        bar = int(span.bar_index[offset])
        state = comp.classify(view_at(bars, bar))
        assert state.regime is regime_of_code(int(span.code[offset]))
        assert state.vol_percentile == float(span.vol_pct[offset])
        assert state.efficiency_ratio == float(span.efficiency[offset])
        assert state.vol_of_vol == float(span.vol_of_vol[offset])
        assert state.shift_statistic == float(span.shift_stat[offset])
        assert state.trend_strength == abs(float(span.trend_tau[offset]))
        assert state.bars_in_regime == int(span.bars_in_regime[offset])


def test_the_detector_holds_no_attribute_that_is_not_derived_from_config():
    """A structural echo of the two tests above: every instance attribute
    must be a scalar read off the two config objects at construction. An
    array or a label here would be state by definition, and the determinism
    tests would only catch it if they happened to exercise the right
    order."""
    comp = detector()
    for name, value in vars(comp).items():
        if name in ("config", "features"):
            continue
        assert isinstance(value, (int, float)), f"{name} is {type(value).__name__}"


# ---------------------------------------------------------------------------
# warmup honesty
# ---------------------------------------------------------------------------


def test_warmup_is_the_number_of_bars_classify_actually_reads():
    """The bug this file found. `warmup_bars` was `_raw_bars +
    min_regime_bars - 1` = 268, the point at which a confirmed label first
    exists, while `_span_from_view` reads `_classify_bars` = 291 so the
    hysteresis horizon is full. Declaring the smaller number made the label
    at bar `t` depend on how much history the caller had loaded -- the exact
    path-dependence the stateless hysteresis design exists to prevent,
    arriving through the warmup contract instead of through mutable state,
    and invisible to the lookahead audit because nothing reads the
    future."""
    comp = detector()
    assert comp.warmup_bars == comp._classify_bars
    assert comp.warmup_bars > comp._first_confirmable_bars


def test_the_warmup_chain_is_the_longest_chain_and_not_the_longest_window():
    """266 = atr_period + atr_percentile_lookback, because Wilder's seed
    consumes `atr_period` true ranges before the first ATR value exists;
    then `hysteresis_search_bars` more raw labels for a full horizon and
    `min_regime_bars - 1` behind the oldest run that horizon may select.

        266 + 24 + 3 - 2 = 291"""
    config = load_config()
    comp = detector()
    instantaneous = max(
        config.features.atr_period + config.features.atr_percentile_lookback,
        config.features.realized_vol_period + config.features.vol_of_vol_period,
        config.features.efficiency_period + 1,
        config.regime.cusum_window_bars + 1,
    )
    assert instantaneous == 266
    raw = max(
        instantaneous,
        config.regime.cusum_window_bars + config.regime.shift_decay_bars,
    )
    assert raw == 266
    assert comp.warmup_bars == (
        raw + config.regime.hysteresis_search_bars + config.regime.min_regime_bars - 2
    )
    assert comp.warmup_bars == 291


def test_extra_history_in_front_does_not_move_a_single_value(scored):
    """Half of warmup honesty: once `warmup_bars` bars are visible, nothing
    further back may change an answer. Checked by cutting history from the
    *left* to exactly `warmup_bars` at 30 sampled bars and requiring every
    one of the thirteen keys to be bit-identical."""
    comp, bars = scored["detector"], scored["dataset"].bars
    for bar in range(comp.warmup_bars - 1, len(bars), 311):
        full = comp.compute(view_at(bars, bar)).values
        window = truncate(
            BarSeries(
                symbol=bars.symbol,
                ts_ns=bars.ts_ns[: bar + 1],
                interval_seconds=INTERVAL,
                columns={name: bars.col(name)[: bar + 1] for name in bars.columns},
            ),
            comp.warmup_bars,
        )
        assert comp.compute(view_at(window)).values == full, f"bar {bar}"


def test_the_old_268_bar_warmup_would_have_starved_the_hysteresis_horizon(scored):
    """The other half, and the evidence that the fix was not cosmetic.

    268 bars leaves `268 - _raw_bars + 1 = 3` raw labels in the horizon
    instead of 24, so the confirmed label is whatever those three say. Fed
    exactly three raw labels at every scored bar, the closed form reports
    UNKNOWN -- "no label has been stable for `min_regime_bars`; not
    tradable" -- on **1_710 of 9_458 bars, 18.1%**, every one of which has a
    perfectly good label once the full horizon is visible.

    Checked by slicing the raw labels rather than by truncating the bar
    series, because the fix means a 268-bar series is now (correctly)
    refused as pre-warmup, and comparing a ready vector against a
    `_not_ready` vector of zeros would prove nothing. This way the
    comparison is exactly the arithmetic the old `warmup_bars` implied."""
    comp, span = scored["detector"], scored["span"]
    k = comp._min_regime_bars
    assert comp._first_confirmable_bars - comp._raw_bars + 1 == k == 3
    starved = np.array(
        [int(comp._confirm(span.raw_code[i - k + 1 : i + 1])[-1])
         for i in range(k - 1, span.raw_code.size)],
        dtype=np.int16,
    )
    full = span.code[k - 1 :]
    differ = starved != full
    assert starved.size == 9_458
    assert int(differ.sum()) == 1_710
    assert float(differ.mean()) == pytest.approx(0.181, abs=0.002)
    # Every disagreement is the starved horizon reporting "not tradable".
    assert np.all(starved[differ] == UNKNOWN_CODE)
    assert float((span.code == UNKNOWN_CODE).mean()) == 0.0


def test_the_declared_warmup_does_not_cover_a_252_session_window():
    """Section 6 asks for the ATR percentile over "a rolling 252-session
    window". `atr_percentile_lookback` is 252 *bars*, and at the 5-minute
    interval these tests score that is 252/78 = 3.23 sessions; the whole
    291-bar warmup is 3.73. A literal reading of section 6 would need
    252 * 78 + 14 = 19_670 bars -- 100 sessions of a year's data spent
    before the first label.

    Reported, not resolved. The implemented feature answers "loud for this
    afternoon" where section 6's answers "loud for this year", and those are
    different features with one name; which one the research wants is not a
    question a test can settle."""
    config = load_config()
    comp = detector()
    bars_per_session = int(6.5 * 3600 // INTERVAL)
    assert bars_per_session == 78
    assert config.features.atr_percentile_lookback == 252
    assert comp.warmup_bars / bars_per_session == pytest.approx(3.73, abs=0.01)
    literal = 252 * bars_per_session + config.features.atr_period
    assert literal == 19_670
    assert comp.warmup_bars < literal / 60


def test_warmup_exceeds_the_feature_configs_own_optimistic_figure():
    """`FeatureConfig.warmup_bars` reports 253: it omits the `atr_period`
    the ATR chain consumes before its first value exists and knows nothing
    about the regime config's windows. `FeatureBundle` takes the max over
    its computers, so declaring the true number here is what keeps a bundle
    honest -- but the deviation is recorded rather than matched."""
    config = load_config()
    assert config.features.warmup_bars == 253
    assert detector().warmup_bars > config.features.warmup_bars


def test_label_span_refuses_a_window_shorter_than_warmup():
    """Not a silent short span: the caller would then get fewer labels than
    bars and align them by position."""
    comp = detector()
    series = flat_series(comp.warmup_bars - 1)
    with pytest.raises(FeatureError, match="needs at least 291 bars"):
        span_of(series, comp)


def test_label_span_labels_exactly_one_bar_at_exactly_warmup():
    """The boundary. `bar_index` must start at `warmup_bars - 1` and every
    array in the span must have that one element -- an off-by-one here
    misaligns every diagnostic against every label."""
    comp = detector()
    span = span_of(flat_series(comp.warmup_bars), comp)
    assert len(span) == 1
    assert span.bar_index.tolist() == [comp.warmup_bars - 1]
    for name in (
        "vol_pct", "trend_tau", "efficiency", "realized_vol", "vol_of_vol",
        "shift_stat", "shift_peak", "raw_code", "code", "bars_in_regime",
    ):
        assert getattr(span, name).size == 1, name


def test_every_span_array_is_aligned_to_bar_index(scored):
    """Eleven arrays sliced from five different index computations. Equal
    lengths is the cheapest check that none of them is off by one, and an
    off-by-one would attribute each bar's label to its neighbour's
    measurements."""
    span = scored["span"]
    expected = len(span)
    for name in (
        "bar_index", "vol_pct", "trend_tau", "efficiency", "realized_vol",
        "vol_of_vol", "shift_stat", "shift_peak", "raw_code", "code",
        "bars_in_regime",
    ):
        assert getattr(span, name).size == expected, name
    assert span.bar_index[0] == scored["detector"].warmup_bars - 1
    assert span.bar_index[-1] == len(scored["dataset"].bars) - 1
    assert np.array_equal(np.diff(span.bar_index), np.ones(expected - 1, dtype=np.int64))


def test_label_span_refuses_unequal_arrays():
    comp = detector()
    with pytest.raises(FeatureError, match="equal-length arrays"):
        comp.label_span(np.zeros(300), np.zeros(299), np.zeros(300), INTERVAL)


def test_label_span_refuses_a_non_positive_price():
    """Every return below is a log ratio. A zero or negative close yields
    -inf and then nan through the CUSUM, and the `_vector` finite guard
    would raise somewhere far from the cause."""
    comp = detector()
    closes = np.full(comp.warmup_bars, BASE)
    closes[10] = 0.0
    with pytest.raises(FeatureError, match="non-positive or non-finite price"):
        comp.label_span(closes + 1.0, closes - 1.0, closes, INTERVAL)


# ---------------------------------------------------------------------------
# the output contract
# ---------------------------------------------------------------------------


def test_the_emitted_keys_match_the_declared_keys_exactly():
    """`_vector` enforces this, so the test is really that `compute` goes
    through `_vector` at all -- a path that built a `FeatureVector` directly
    would skip both the key check and the finite guard."""
    comp = detector()
    vector = comp.compute(view_at(flat_series(comp.warmup_bars + 5)))
    assert tuple(vector.values) == comp.keys
    assert len(comp.keys) == 13
    assert len(set(comp.keys)) == 13


def test_every_emitted_value_is_finite_across_every_branch():
    """A non-finite feature propagates into every score that reads it. Each
    of the five series below drives a different branch, so the guard is
    exercised where the arithmetic is most degenerate -- a halted tape, a
    zero-dispersion CUSUM window, a zero-travel efficiency ratio."""
    comp = detector()
    n = comp.warmup_bars + 10
    for series in (
        flat_series(n),
        series_from(np.full(n, BASE), np.zeros(n)),
        series_from(np.full(n, BASE), np.linspace(2.0, 0.5, n)),
        series_from(np.full(n, BASE), np.linspace(0.5, 2.0, n)),
        ramp_series(n, 1e-4),
        jump_series(n, 3),
    ):
        values = comp.compute(view_at(series)).values
        assert all(math.isfinite(v) for v in values.values()), values


def test_every_bounded_key_is_actually_bounded(scored):
    """The scoring layer may only read bounded quantities. `vol_pct`,
    `trend_strength`, `efficiency`, `shift_score` and `regime_confidence`
    are the five that claim to be, and they are checked over all 9_460
    scored bars rather than on a constructed series, because the bound has
    to hold on tape nobody designed."""
    comp, span = scored["detector"], scored["span"]
    assert np.all((span.vol_pct >= 0.0) & (span.vol_pct <= 1.0))
    assert np.all((span.trend_tau >= -1.0) & (span.trend_tau <= 1.0))
    assert np.all((span.efficiency >= 0.0) & (span.efficiency <= 1.0))
    assert np.all(span.shift_stat >= 0.0)
    assert np.all(span.shift_peak >= span.shift_stat)
    bars = scored["dataset"].bars
    for offset in (0, 4000, len(span) - 1):
        values = comp.compute(view_at(bars, int(span.bar_index[offset]))).values
        assert 0.0 <= values["shift_score"] <= 1.0
        assert 0.0 <= values["regime_confidence"] <= 1.0
        assert 0.0 <= values["trend_strength"] <= 1.0


def test_the_unbounded_diagnostics_are_the_ones_the_docstring_names():
    """`shift_stat`, `vol_of_vol` and `realized_vol` are diagnostics in
    natural units, compared against their own thresholds and plotted. The
    test is that bounded companions exist for the two the scoring layer
    needs, so a weighted sum never has an excuse to read the raw one."""
    comp = detector()
    assert "shift_score" in comp.keys and "shift_stat" in comp.keys
    assert "trend_strength" in comp.keys and "trend_tau" in comp.keys
    values = comp.compute(view_at(jump_series(comp.warmup_bars + 5, 2))).values
    assert values["shift_score"] == pytest.approx(
        math.tanh(values["shift_stat"] / load_config().regime.cusum_threshold)
    )
    assert values["trend_strength"] == abs(values["trend_tau"])


def test_required_feeds_is_bars_and_nothing_else():
    """Declaring a feed it does not need would make the detector degrade a
    whole bundle whenever that feed was absent; declaring one it *does* need
    as optional would let it emit silent rubbish. It reads high, low and
    close, so BARS is the honest answer."""
    comp = detector()
    assert comp.required_feeds == frozenset({Feed.BARS})
    assert comp.optional_feeds == frozenset()


def test_only_the_high_low_and_close_columns_are_read():
    """`required_feeds` honesty at column granularity. Replacing `open` and
    `volume` with values no instrument would print must not move a single
    output -- if it did, the detector would be reading a column its
    documentation does not mention and the lookahead audit would be
    perturbing the wrong thing."""
    comp = detector()
    n = comp.warmup_bars + 10
    closes = np.full(n, BASE)
    u = alternating_half_ranges(n)
    plain = series_from(closes, u)
    odd = BarSeries(
        symbol=SYMBOL,
        ts_ns=plain.ts_ns,
        interval_seconds=INTERVAL,
        columns={
            "open": closes - u,  # the low, not the close
            "high": closes + u,
            "low": closes - u,
            "close": closes.copy(),
            "volume": np.arange(n, dtype=np.float64) * 7.0,
        },
    )
    assert comp.compute(view_at(odd)).values == comp.compute(view_at(plain)).values


def test_a_pre_warmup_view_is_not_ready_rather_than_wrong():
    """`compute` re-checks warmup itself because the audit calls computers
    directly at pre-warmup bars. The zeros it returns are placeholders and
    must be marked as such, or a bundle would score them."""
    comp = detector()
    vector = comp.compute(view_at(flat_series(comp.warmup_bars - 1)))
    assert not vector.warmup_complete
    assert set(vector.values) == set(comp.keys)
    assert set(vector.values.values()) == {0.0}
    assert vector.quality == DataQuality.MISSING
    assert any("warmup incomplete" in note for note in vector.notes)


def test_an_absent_bars_feed_is_not_ready_rather_than_an_exception():
    comp = detector()
    vector = comp.compute(empty_view())
    assert not vector.warmup_complete
    assert set(vector.values) == set(comp.keys)
    assert vector.quality == DataQuality.MISSING


def test_classify_before_warmup_returns_unknown_and_is_not_tradable():
    """`RegimeState.is_tradable` is what downstream gates on, so a
    pre-warmup state has to be UNKNOWN rather than a plausible label with
    low confidence."""
    comp = detector()
    state = comp.classify(view_at(flat_series(comp.warmup_bars - 1)))
    assert state.regime is Regime.UNKNOWN
    assert not state.is_tradable
    assert state.confidence == 0.0
    assert state.diagnostics == {}


def test_the_regime_state_carries_the_label_and_the_whole_diagnostic_vector():
    """Section 6: "Regimes are *not* mutually exclusive in reality, so
    `RegimeState` carries the primary label plus the full diagnostic
    vector, and analytics breaks results down by both." A state carrying
    only the label would make the Phase 8 breakdown impossible to build and
    the taxonomy impossible to falsify -- which is the one thing the
    confusion matrix below is for."""
    comp = detector()
    state = comp.classify(view_at(jump_series(comp.warmup_bars + 10, 3)))
    assert isinstance(state, RegimeState)
    assert state.regime is Regime.DIRECTIONAL_SHIFT
    assert state.is_tradable
    for named in (
        "vol_percentile", "trend_strength", "efficiency_ratio", "vol_of_vol",
        "shift_statistic",
    ):
        assert getattr(state, named) is not None, named
    for key in (
        "vol_pct", "trend_tau", "trend_strength", "efficiency", "realized_vol",
        "vol_of_vol", "shift_stat", "shift_peak", "shift_score", "regime_code",
        "raw_regime_code",
    ):
        assert key in state.diagnostics, key
    assert state.diagnostics["regime_code"] == CODE_BY_REGIME[Regime.DIRECTIONAL_SHIFT]


def test_the_raw_code_is_reported_alongside_the_confirmed_one(scored):
    """The difference between them *is* the hysteresis lag, so measuring the
    lag needs both -- and a bar whose confirmed label disagrees with its own
    raw label is a bar being held by confirmation, which is exactly what an
    attribution wants to know."""
    comp, span, bars = scored["detector"], scored["span"], scored["dataset"].bars
    disagree = np.flatnonzero(span.code != span.raw_code)
    assert disagree.size == 1_381  # of 9_460 bars
    offset = int(disagree[0])
    values = comp.compute(view_at(bars, int(span.bar_index[offset]))).values
    assert values["regime_code"] == float(span.code[offset])
    assert values["raw_regime_code"] == float(span.raw_code[offset])
    assert values["regime_code"] != values["raw_regime_code"]


def test_a_held_label_reports_zero_confidence_rather_than_a_negative_margin():
    """`confidence` is a normalized margin, not a probability. During
    hysteresis the confirmed label can disagree with the current bar's
    measurements and the margin is then negative; clamping to 0.0 says
    honestly that the label is held by confirmation rather than supported by
    this bar. A negative confidence would also violate `RegimeState`'s own
    `ge=0.0` bound and raise far from the cause."""
    comp = detector()
    assert comp.confidence(
        Regime.HIGH_VOL,
        {"vol_pct": 0.1, "efficiency": 0.0, "shift_stat": 0.0, "shift_peak": 0.0},
    ) == 0.0
    assert comp.confidence(
        Regime.LOW_VOL,
        {"vol_pct": 0.9, "efficiency": 0.0, "shift_stat": 0.0, "shift_peak": 0.0},
    ) == 0.0
    assert comp.confidence(
        Regime.UNKNOWN,
        {"vol_pct": 0.5, "efficiency": 0.0, "shift_stat": 0.0, "shift_peak": 0.0},
    ) == 0.0


def test_each_confidence_margin_reaches_one_at_its_statistics_bound():
    """A margin that saturated below 1.0 would compress the whole top of the
    range and make a borderline label indistinguishable from an emphatic
    one. HIGH_VOL at `vol_pct = 1`, LOW_VOL at 0, TRENDING at efficiency 1,
    and CHOP at the midpoint of the tradable band with nothing else
    triggering."""
    config, comp = load_config(), detector()
    flat = {"efficiency": 0.0, "shift_stat": 0.0, "shift_peak": 0.0}
    assert comp.confidence(Regime.HIGH_VOL, {"vol_pct": 1.0, **flat}) == 1.0
    assert comp.confidence(Regime.LOW_VOL, {"vol_pct": 0.0, **flat}) == 1.0
    assert comp.confidence(
        Regime.TRENDING_UP,
        {"vol_pct": 0.5, "efficiency": 1.0, "shift_stat": 0.0, "shift_peak": 0.0},
    ) == 1.0
    midpoint = 0.5 * (config.regime.low_vol_percentile + config.regime.high_vol_percentile)
    # 1 - 2e-16 rather than exactly 1.0: `0.8 - 0.2` and `0.5 - 0.2` are not
    # the same double, so the midpoint margin divides two values differing in
    # the last bit. Float noise in a diagnostic margin, not a logic error,
    # and not worth an epsilon in `_clamp01` that would hide a real
    # overshoot.
    assert comp.confidence(Regime.CHOP, {"vol_pct": midpoint, **flat}) == pytest.approx(
        1.0, abs=1e-12
    )


def test_each_confidence_margin_is_zero_exactly_at_its_threshold():
    """The other end. A bar sitting on its branch's threshold has no margin,
    and CHOP has three thresholds to be adjacent to, so its margin is the
    minimum over all three distances."""
    config, comp = load_config(), detector()
    regime = config.regime
    flat = {"efficiency": 0.0, "shift_stat": 0.0, "shift_peak": 0.0}
    assert comp.confidence(
        Regime.HIGH_VOL, {"vol_pct": regime.high_vol_percentile, **flat}
    ) == 0.0
    assert comp.confidence(
        Regime.LOW_VOL, {"vol_pct": regime.low_vol_percentile, **flat}
    ) == 0.0
    assert comp.confidence(
        Regime.TRENDING_UP,
        {
            "vol_pct": 0.5,
            "efficiency": regime.trend_efficiency_threshold,
            "shift_stat": 0.0,
            "shift_peak": 0.0,
        },
    ) == 0.0
    midpoint = 0.5 * (regime.low_vol_percentile + regime.high_vol_percentile)
    for near in (
        {"vol_pct": regime.low_vol_percentile, **flat},
        {"vol_pct": regime.high_vol_percentile, **flat},
        {
            "vol_pct": midpoint,
            "efficiency": regime.trend_efficiency_threshold,
            "shift_stat": 0.0,
            "shift_peak": 0.0,
        },
        {
            "vol_pct": midpoint,
            "efficiency": 0.0,
            "shift_stat": regime.cusum_threshold,
            "shift_peak": regime.cusum_threshold,
        },
    ):
        assert comp.confidence(Regime.CHOP, near) == 0.0


def test_the_shift_confidence_reads_the_peak_and_not_this_bars_statistic():
    """The peak inside the decay window is what put the label there. Reading
    the current bar's own statistic would report 0.0 for every bar of a
    decaying shift -- a label with no confidence on the bars where it is
    most often assigned."""
    comp = detector()
    decayed = {
        "vol_pct": 0.5,
        "efficiency": 0.0,
        "shift_stat": 0.0,
        "shift_peak": 8.0,
    }
    assert comp.confidence(Regime.DIRECTIONAL_SHIFT, decayed) > 0.0
    assert comp.confidence(Regime.DIRECTIONAL_SHIFT, decayed) == pytest.approx(
        math.tanh((8.0 - 4.0) / 4.0)
    )


def test_a_span_position_renders_as_plain_floats():
    """`RegimeSpan.at` is what both `classify` and `compute` read their
    diagnostics from. NumPy scalars leaking through would serialize
    differently and compare unequal to the same value read back from a
    report."""
    span = span_of(flat_series(detector().warmup_bars + 3))
    rendered = span.at(len(span) - 1)
    assert all(type(value) is float for value in rendered.values())
    assert set(rendered) == {
        "vol_pct", "trend_tau", "trend_strength", "efficiency", "realized_vol",
        "vol_of_vol", "shift_stat", "shift_peak",
    }
    assert span.regimes() == [regime_of_code(int(c)) for c in span.code]


# ---------------------------------------------------------------------------
# construction and the lookahead audit
# ---------------------------------------------------------------------------


def test_the_constructor_takes_configuration_only():
    """Local echo of the package-wide ban in test_feature_contracts.py. A
    detector handed a series could capture a full-sample statistic at
    construction, and that is the one leak `validation/lookahead.py` cannot
    see through when it is given an already-built instance.

    `detector.py` uses postponed annotations, so `__init__.__annotations__`
    holds strings; they are resolved with `get_type_hints` before comparing,
    because comparing a `str` to a class silently passes -- a test bug that
    has already bitten this project twice."""
    parameters = inspect.signature(RegimeDetector.__init__).parameters
    assert list(parameters) == ["self", "config", "features"]

    hints = get_type_hints(RegimeDetector.__init__)
    assert hints["config"] is RegimeConfig
    assert hints["features"] is FeatureConfig

    rendered = str(inspect.signature(RegimeDetector.__init__))
    for banned in (
        "SymbolData", "BarSeries", "ColumnSeries", "DataStore", "MarketView",
        "SyntheticDataset", "ndarray", "DataFrame", "Series",
    ):
        assert banned not in rendered, f"{banned} must not appear in the constructor"


def test_the_lookahead_audit_passes_with_a_factory(scored):
    """`factory=` rather than an instance: that is the form that rebuilds the
    detector against each truncated and each future-mutated dataset, so a
    full-sample constant captured in `__init__` would change with it and be
    caught. Handing the audit an instance would not test that."""
    config = load_config()
    comp = scored["detector"]
    assert len(scored["data"].primary_bars) >= 2 * comp.warmup_bars
    result = audit_computer(
        data=scored["data"],
        factory=lambda d: RegimeDetector(config.regime, config.features),
        sample=40,
    )
    assert_no_lookahead([result])
    assert result.bars_checked >= 20
    assert set(result.keys_checked) == set(comp.keys)


def test_the_lookahead_audit_passes_on_a_config_whose_decay_outruns_its_windows(scored):
    """A second configuration, chosen because it is the one where the index
    arithmetic is most fragile. When `shift_decay_bars` exceeds the
    instantaneous window, the CUSUM's own start index goes *negative* --
    `arange(first_shift, n) - cusum_window` -- and a negative index into
    `cusum_all` reads from the end of the array, which is a future bar. The
    entries it poisons are provably discarded by the slice that follows, and
    this is the test that holds that proof to account rather than trusting
    the algebra."""
    features = FeatureConfig(
        atr_period=2,
        atr_percentile_lookback=11,
        realized_vol_period=2,
        vol_of_vol_period=2,
        efficiency_period=2,
    )
    regime = RegimeConfig(
        cusum_window_bars=3,
        shift_decay_bars=20,
        min_regime_bars=2,
        hysteresis_search_bars=5,
    )
    comp = RegimeDetector(regime, features)
    assert comp._decay_bars > comp._inst_bars - 1 - comp._cusum_window
    result = audit_computer(
        data=scored["data"],
        factory=lambda d: RegimeDetector(regime, features),
        sample=40,
    )
    assert_no_lookahead([result])


def test_the_audit_catches_a_detector_that_captured_a_full_sample_constant():
    """Proof the gate is not vacuous, and proof that `factory=` is the form
    that makes it non-vacuous.

    A `MarketView` cannot be made to hand over a future bar, so the leak the
    constructor ban exists to stop is the only one reachable here: a
    statistic captured from the whole dataset at construction time. This
    subclass takes the dataset and normalizes `vol_pct` by the full-sample
    mean ATR percentile -- an entirely plausible-looking "percentile
    relative to this instrument's own history" that is in fact computed
    from bars the view has not reached.

    Handed to the audit as a `factory`, it is rebuilt against each truncated
    and each future-mutated dataset, the captured constant moves, and the
    audit reports it. Handed the same object as an `instance` it would not
    be rebuilt and the audit would find nothing -- which is the second half
    of the test, and the reason the two passes above pass `factory=`."""

    class CapturedAtInit(RegimeDetector):
        def __init__(self, config, features, dataset) -> None:
            super().__init__(config, features)
            # "Scale the percentile against this instrument's own average
            # level" -- computed from every bar in the dataset, including
            # the ones no view has reached.
            self._leak = float(dataset.primary_bars.col("close").mean())

        def compute(self, view):
            vector = super().compute(view)
            if not vector.warmup_complete:
                return vector
            values = dict(vector.values)
            values["vol_pct"] = float(
                np.clip(0.5 * values["vol_pct"] * (18_000.0 / self._leak), 0.0, 1.0)
            )
            return self._vector(view, values, notes=vector.notes)

    from flow_model.data import SyntheticConfig, SyntheticMarketGenerator

    config = load_config()
    dataset = SyntheticMarketGenerator(SyntheticConfig(), seed=3).generate(
        SYMBOL,
        config.spec(SYMBOL),
        date(2017, 1, 1),
        date(2017, 3, 1),
        INTERVAL,
        calendar=TradingCalendar(),
    )
    data = symbol_data(dataset.bars)

    leaky = audit_computer(
        data=data,
        factory=lambda d: CapturedAtInit(config.regime, config.features, d),
        sample=12,
    )
    assert not leaky.passed
    assert {finding.kind for finding in leaky.findings} <= {"truncation", "mutation"}
    assert "vol_pct" in {finding.key for finding in leaky.findings}
    with pytest.raises(AssertionError, match="lookahead audit failed"):
        assert_no_lookahead([leaky])

    # The same cheat, handed over as an instance: invisible, because nothing
    # rebuilds it. This is why both audits above use `factory=`.
    blind = audit_computer(
        CapturedAtInit(config.regime, config.features, data), data, sample=12
    )
    assert blind.passed


# ---------------------------------------------------------------------------
# the confusion matrix: the measurement this file exists for
# ---------------------------------------------------------------------------


def test_the_ground_truth_is_the_cause_of_the_prices_and_not_an_annotation(scored):
    """What makes the rest of this section meaningful. `synthetic.py` draws
    the regime path from a Markov chain first and draws prices conditioned
    on it, so `regime_labels` is the *cause* of the bars. Checked here
    rather than assumed: every label must be a regime the generator declared
    parameters for, never UNKNOWN (which means "insufficient warmup" and is
    not a thing bars can be generated from), and the labels must cover the
    bars one for one."""
    dataset, truth, span = scored["dataset"], scored["truth"], scored["span"]
    assert len(dataset.regime_labels) == len(dataset.bars)
    assert truth.size == len(span)
    assert UNKNOWN_CODE not in set(truth.tolist())
    assert set(truth.tolist()) == {
        CODE_BY_REGIME[Regime(name)] for name in dataset.regime_names
    }
    assert len(dataset.regime_names) == 6


def test_the_confusion_matrix_accounts_for_every_scored_bar(scored):
    """A matrix that silently dropped bars would flatter whichever regime it
    dropped them from. The row sums are the supports and the column sums the
    predictions, and both must total the span."""
    confusion, span = scored["confusion"], scored["span"]
    assert int(confusion.sum()) == len(span) == 9_460
    assert confusion[UNKNOWN_CODE].sum() == 0  # truth never says UNKNOWN
    assert int(confusion[:, UNKNOWN_CODE].sum()) == 0  # and neither does the detector


def test_overall_accuracy_at_the_shipped_defaults(scored):
    """**The headline number, measured at `load_config().regime` and not
    tuned.** 3_192 of 9_460 bars correct = 0.3374.

    The assertion is `>= 0.30`, which is the measured value with room for
    the dataset to be regenerated, and it is deliberately not a number this
    file worked toward. For scale: always guessing the most common true
    regime (CHOP, 1_988 bars) would score 0.210, and guessing uniformly over
    six labels would score 0.167. So the detector carries real information
    -- about 1.6x the majority-class baseline -- and is nowhere near usable
    as a six-way classifier.

    What that means for the research is the point of measuring it: the
    section 6 taxonomy is largely **not** recoverable from 5-minute price
    alone at these thresholds, and any Phase 8 breakdown by regime is a
    breakdown by a label that is wrong about two bars in three. The three
    mechanisms are isolated in the four tests that follow."""
    confusion = scored["confusion"]
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total
    assert (correct, total) == (3_192, 9_460)
    assert accuracy == pytest.approx(0.3374, abs=0.0005)
    assert accuracy >= 0.30
    # Baselines, computed from the matrix rather than written down.
    majority = float(confusion.sum(axis=1).max()) / total
    assert majority == pytest.approx(0.2101, abs=0.0005)
    assert accuracy > majority


def test_hysteresis_costs_accuracy_rather_than_buying_it(scored):
    """Worth knowing which way the trade goes. The raw labels score 3_321 of
    9_460 = 0.3511 and the confirmed ones 0.3374, so confirmation costs 1.4
    points of accuracy on this tape. That is the expected sign -- the rule
    exists to suppress flapping, not to improve classification -- but it
    means `min_regime_bars` is paid for in both lag *and* accuracy, and the
    justification has to be the flapping it prevents."""
    truth, span = scored["truth"], scored["span"]
    raw_accuracy = float((span.raw_code.astype(np.int64) == truth).mean())
    confirmed_accuracy = float((span.code.astype(np.int64) == truth).mean())
    assert raw_accuracy == pytest.approx(0.3511, abs=0.0005)
    assert confirmed_accuracy == pytest.approx(0.3374, abs=0.0005)
    assert raw_accuracy > confirmed_accuracy


def test_per_regime_precision_and_recall_at_the_shipped_defaults(scored):
    """The whole table, measured. Supports are the generator's own dwell
    times, so they are not balanced and precision matters more than
    accuracy:

        regime             support  precision  recall
        DIRECTIONAL_SHIFT      282     0.030    0.181
        HIGH_VOL             1_640     0.470    0.404
        LOW_VOL              1_954     0.504    0.655
        TRENDING_UP          1_750     0.439    0.098
        TRENDING_DOWN        1_846     0.317    0.059
        CHOP                 1_988     0.297    0.463

    Three readings. The vol branches are the only ones that work at all, and
    they only reach ~0.5 precision. DIRECTIONAL_SHIFT's precision of 0.030
    is *exactly its base rate* (282/9_460 = 0.0298), which means the label
    carries no information whatsoever -- see the next test for why. And the
    trend branches have precision around 0.4 but recall under 0.10: when
    they fire they are better than chance, and they almost never fire.

    The bounds below are loose and one-sided in the direction each finding
    points, so this test fails if the detector gets materially better or
    worse, and does not fail on a reseeded dataset."""
    confusion = scored["confusion"]
    expected = {
        Regime.DIRECTIONAL_SHIFT: (0.030, 0.181, 282),
        Regime.HIGH_VOL: (0.470, 0.404, 1_640),
        Regime.LOW_VOL: (0.504, 0.655, 1_954),
        Regime.TRENDING_UP: (0.439, 0.098, 1_750),
        Regime.TRENDING_DOWN: (0.317, 0.059, 1_846),
        Regime.CHOP: (0.297, 0.463, 1_988),
    }
    for label, (precision, recall, support) in expected.items():
        got_precision, got_recall, got_support = precision_recall(confusion, label)
        assert got_support == support, label
        assert got_precision == pytest.approx(precision, abs=0.002), label
        assert got_recall == pytest.approx(recall, abs=0.002), label
    # No regime is ever perfectly recovered, and none is entirely unreachable.
    for label in expected:
        precision, recall, _ = precision_recall(confusion, label)
        assert 0.0 < recall < 0.70, label
        assert 0.0 < precision < 0.60, label


def test_the_vol_branches_claim_forty_percent_of_bars_by_rank_construction(scored):
    """**Systematic error 1, and the largest.** `vol_pct` is a percentile
    *rank* inside a trailing window, so by construction about
    `low_vol_percentile` of bars sit at or below the low threshold and
    `1 - high_vol_percentile` at or above the high one -- 0.20 + 0.20 = 0.40
    **in every regime, whatever the tape is doing**. Measured: 0.418 of all
    9_460 bars are labelled LOW_VOL or HIGH_VOL, against a true combined
    share of 0.380.

    That is not a bug in the arithmetic; it is what a rank means. It becomes
    a classification problem because section 6 checks the vol branches
    *before* the trend branches, so those 40% of bars can never be labelled
    TRENDING however clean their geometry. A rank-based vol measure and a
    precedence-ordered classifier do not compose, and the fix is a design
    decision (an absolute vol level, or a vol branch that only fires when no
    trend does) rather than a threshold move."""
    config, span, truth = load_config(), scored["span"], scored["truth"]
    regime = config.regime
    low = float((span.vol_pct <= regime.low_vol_percentile).mean())
    high = float((span.vol_pct >= regime.high_vol_percentile).mean())
    # The rank's own construction, independent of any label. Measured 0.276
    # and 0.229 against the nominal 0.20 each -- both tails run *above* the
    # quantile because `percentile_rank` counts "at or below" and a smoothed
    # ATR produces long stretches of near-ties, which pile onto whichever
    # side of the window the current value ties with. The point stands and
    # is stronger: half the tape satisfies a vol condition.
    assert low == pytest.approx(0.276, abs=0.01)
    assert high == pytest.approx(0.229, abs=0.01)
    assert low + high == pytest.approx(0.505, abs=0.01)
    assert low > regime.low_vol_percentile
    assert high > 1.0 - regime.high_vol_percentile
    labelled = np.isin(
        span.code, [CODE_BY_REGIME[Regime.LOW_VOL], CODE_BY_REGIME[Regime.HIGH_VOL]]
    )
    assert float(labelled.mean()) == pytest.approx(0.418, abs=0.005)
    true_share = float(
        np.isin(
            truth, [CODE_BY_REGIME[Regime.LOW_VOL], CODE_BY_REGIME[Regime.HIGH_VOL]]
        ).mean()
    )
    assert true_share == pytest.approx(0.380, abs=0.005)


def test_the_shift_threshold_is_a_null_percentile_not_a_detection_threshold(scored):
    """**Systematic error 2.** DIRECTIONAL_SHIFT fires on 0.174 of bars
    against a true share of 0.030 -- nearly six times too often -- and its
    precision *is* its base rate, so the label carries no information.

    The evidence that this is the threshold and not the statistic is a null
    simulation, which is mathematics and not fitting: under iid returns,
    `P(shift_stat > 4.0)` is 0.0082, and `shift_decay_bars = 10` spreads
    each alarm over ten bars for a per-bar rate of 0.0369 before any real
    structure exists. `cusum_threshold = 4.0` is the 99.2nd percentile of
    the null -- a reasonable *false-alarm* threshold for a rare event, and
    far too loose for a label that is supposed to pick out 3% of bars. On
    the scored tape the rate is 0.174 rather than 0.037 because returns are
    autocorrelated and heteroskedastic, which within-window standardization
    does not remove.

    **The default is left exactly where it shipped.** 6.0 would put the null
    decay rate at 0.0008 and is where this file would start looking, but a
    threshold moved to improve the matrix above would be fitted to the same
    data the matrix is measured on and would mean nothing."""
    config, span, truth = load_config(), scored["span"], scored["truth"]
    threshold = config.regime.cusum_threshold
    assert threshold == 4.0
    fires = float((span.shift_peak > threshold).mean())
    assert fires == pytest.approx(0.174, abs=0.005)
    true_share = float((truth == CODE_BY_REGIME[Regime.DIRECTIONAL_SHIFT]).mean())
    assert true_share == pytest.approx(0.0298, abs=0.002)
    assert fires > 5.0 * true_share

    # The null, simulated. Seeded, so this is a fixed number and not a draw.
    generator = np.random.default_rng(20240102)
    null = _rolling_cusum(
        generator.normal(size=120_000), config.regime.cusum_window_bars,
        config.regime.cusum_drift,
    )
    per_bar = float((null > threshold).mean())
    with_decay = float(
        (
            np.lib.stride_tricks.sliding_window_view(
                null, config.regime.shift_decay_bars
            ).max(axis=1)
            > threshold
        ).mean()
    )
    assert per_bar == pytest.approx(0.0082, abs=0.0015)
    assert with_decay == pytest.approx(0.0369, abs=0.006)
    # Precision cannot beat this even on a perfect tape: the alarm rate alone
    # already exceeds the event's share of bars.
    assert with_decay > true_share


def test_the_trend_threshold_barely_separates_a_trending_tape_from_a_random_walk(scored):
    """**Systematic error 3.** TRENDING recall is 0.098 and 0.059 against
    supports of 1_750 and 1_846, and the branch is not merely preempted --
    its condition does not hold in the first place. Measured: the trend
    condition holds on 0.306 of truly-trending bars.

    Why, as arithmetic rather than opinion. A Kaufman ratio over `N` steps
    of a driftless walk has a mean of exactly `1/sqrt(N)` -- the ratio of
    `E|displacement| = sigma*sqrt(2N/pi)` to `E[travel] = N*sigma*sqrt(2/pi)`
    -- which is 0.2236 at `efficiency_period = 20`. Measured median on the
    scored tape: 0.230. `trend_efficiency_threshold = 0.40` is 1.8x that
    null mean, and `P(ER >= 0.40)` is 0.155 under a pure random walk against
    0.376 under the generator's own TRENDING_UP parameters. A likelihood
    ratio of 2.4 is all the separation the threshold has to work with, and
    it caps recall at 0.376 before precedence takes its cut.

    Then precedence takes its cut: of the 2_286 bars where the condition
    *does* hold, 0.637 are relabelled by an earlier branch. 0.306 * 0.363 =
    0.111, which is the recall the matrix reports.

    **The default is left exactly where it shipped.** A 20-bar efficiency
    ratio at a 5-minute interval may simply not be a trend detector; that is
    a feature-design question, not a threshold to nudge."""
    config, span, truth = load_config(), scored["span"], scored["truth"]
    threshold = config.regime.trend_efficiency_threshold
    period = config.features.efficiency_period
    assert (threshold, period) == (0.40, 20)

    # The null mean, closed form, against the measured median.
    assert 1.0 / math.sqrt(period) == pytest.approx(0.2236, abs=0.0005)
    assert float(np.median(span.efficiency)) == pytest.approx(0.230, abs=0.01)
    assert threshold > 1.7 * (1.0 / math.sqrt(period))

    trending = np.isin(
        truth,
        [CODE_BY_REGIME[Regime.TRENDING_UP], CODE_BY_REGIME[Regime.TRENDING_DOWN]],
    )
    holds = span.efficiency >= threshold
    assert float(holds[trending].mean()) == pytest.approx(0.306, abs=0.01)
    assert int(holds.sum()) == 2_286
    trend_codes = [
        CODE_BY_REGIME[Regime.TRENDING_UP],
        CODE_BY_REGIME[Regime.TRENDING_DOWN],
    ]
    preempted = float((~np.isin(span.raw_code[holds], trend_codes)).mean())
    assert preempted == pytest.approx(0.637, abs=0.01)
    assert preempted > 0.5


def test_chop_absorbs_more_bars_than_it_is_owed(scored):
    """The fourth direction worth naming. CHOP is the only branch with no
    condition of its own, so every bar no other branch claims lands there:
    3_093 predictions against 1_988 true bars, a 1.56x over-assignment, and
    0.703 of those predictions are wrong. 1_368 of them are truly TRENDING
    bars whose efficiency never cleared 0.40.

    That is the same finding as the previous test seen from the other side,
    and it is the one that matters most for Phase 8: a breakdown that
    reports "CHOP" for a third of the tape is reporting "none of the above"
    under a name that implies a measurement."""
    confusion = scored["confusion"]
    chop = CODE_BY_REGIME[Regime.CHOP]
    predicted = int(confusion[:, chop].sum())
    support = int(confusion[chop].sum())
    assert (predicted, support) == (3_093, 1_988)
    assert predicted / support == pytest.approx(1.556, abs=0.01)
    trend_rows = [
        CODE_BY_REGIME[Regime.TRENDING_UP],
        CODE_BY_REGIME[Regime.TRENDING_DOWN],
    ]
    assert int(confusion[trend_rows, chop].sum()) == 1_368
    assert 1.0 - confusion[chop, chop] / predicted == pytest.approx(0.703, abs=0.002)


def test_every_label_is_both_emitted_and_missed_so_no_branch_is_dead(scored):
    """The weakest and most important structural claim: a branch that never
    fires, or one that fires on everything, is a different bug from a
    mislabelling and would not show up in an accuracy number. Every one of
    the six generated regimes is predicted at least once and confused at
    least once, so the matrix is genuinely six-way and not a two-label
    detector wearing six names."""
    confusion = scored["confusion"]
    for label in REGIME_CODES:
        if label is Regime.UNKNOWN:
            continue
        index = CODE_BY_REGIME[label]
        assert confusion[:, index].sum() > 0, f"{label} is never emitted"
        assert confusion[index].sum() > 0, f"{label} is never generated"
        off_diagonal = confusion[:, index].sum() - confusion[index, index]
        assert off_diagonal > 0, f"{label} is never a false positive"
    # And no single label swallows the tape.
    shares = confusion.sum(axis=0) / confusion.sum()
    assert shares.max() == pytest.approx(0.327, abs=0.005)
    assert shares.max() < 0.50


def test_even_the_vol_axis_alone_barely_beats_guessing_neither(scored):
    """The place the taxonomy was most likely to work, and it does not --
    which is the least flattering number in this file and the reason it is
    here.

    Collapse the six labels onto the vol axis alone -- quiet, loud, or
    neither -- and the detector scores 0.6239 against a majority baseline of
    0.6201 (always answering "neither", which is 62% of the truth). That is
    an edge of 0.004: on the coarsest possible question the vol branches are
    *within noise of saying nothing*, even though their individual recalls
    (0.655 and 0.404) look respectable. The two facts are compatible because
    the branches buy their recall with volume -- they claim 0.418 of bars to
    cover a true 0.380 -- and the collapse charges them for the
    over-claiming the per-regime recall hides.

    The brief's registered hypothesis is "88-92% in low- and high-vol"
    (`H-REGIME-WINRATE` in `validation/hypotheses.py`, about win rate rather
    than classification). This is the evidence against its premise from the
    classification side: the regimes it conditions on are not identified
    well enough for a per-regime win rate to mean what the hypothesis
    assumes."""
    span, truth = scored["span"], scored["truth"]
    low, high = CODE_BY_REGIME[Regime.LOW_VOL], CODE_BY_REGIME[Regime.HIGH_VOL]

    def collapse(codes):
        out = np.zeros(codes.size, dtype=np.int64)
        out[codes == low] = 1
        out[codes == high] = 2
        return out

    predicted = collapse(span.code.astype(np.int64))
    actual = collapse(truth)
    accuracy = float((predicted == actual).mean())
    majority = float(np.bincount(actual, minlength=3).max()) / actual.size
    assert accuracy == pytest.approx(0.6239, abs=0.002)
    assert majority == pytest.approx(0.6201, abs=0.002)
    assert accuracy > majority
    assert accuracy - majority < 0.02
    assert accuracy < 0.88  # the registered hypothesis, nowhere near met


def test_cohens_kappa_and_balanced_accuracy_at_the_shipped_defaults(scored):
    """The two summary numbers that survive the unbalanced supports, so the
    headline accuracy cannot be read as either better or worse than it is.

        Cohen's kappa      0.2017   (chance agreement p_e = 0.1700)
        balanced accuracy  0.3098   (macro recall; chance 1/6 = 0.1667)
        macro precision    0.3427

    Kappa around 0.20 is the conventional "slight" band -- the detector
    agrees with truth a fifth of the way from chance to perfect. Balanced
    accuracy of 0.31 against a 0.167 chance level says the same thing
    without the class-prior flattery in the raw 0.337: there is real signal
    here, and it is about a third of what a usable six-way classifier
    needs."""
    confusion = scored["confusion"]
    total = confusion.sum()
    observed = float(np.trace(confusion)) / total
    expected = float(
        ((confusion.sum(axis=1) / total) * (confusion.sum(axis=0) / total)).sum()
    )
    kappa = (observed - expected) / (1.0 - expected)
    assert expected == pytest.approx(0.1700, abs=0.001)
    assert kappa == pytest.approx(0.2017, abs=0.002)
    assert 0.0 < kappa < 0.40

    live = [i for i in range(1, len(REGIME_CODES)) if confusion[i].sum()]
    recalls = [confusion[i, i] / confusion[i].sum() for i in live]
    precisions = [confusion[i, i] / confusion[:, i].sum() for i in live]
    assert len(live) == 6
    assert float(np.mean(recalls)) == pytest.approx(0.3098, abs=0.002)
    assert float(np.mean(precisions)) == pytest.approx(0.3427, abs=0.002)
    assert float(np.mean(recalls)) > 1.0 / len(live)
