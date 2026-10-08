"""Adversarial tests for the options-flow features.

`features/optionsflow.py` is ARCHITECTURE.md section 5's options row in code:
20 of the 100 points, five named chain measurements, and -- the part that
makes it different from every Phase 3 computer -- a *degradation contract*
that section 5 states in two sentences and the Phase 5 gate makes the main
deliverable:

    "Degrades to DEGRADED with daily granularity; OptionsFlow sub-score
     capped and flagged. Never synthesized."

Nothing above this module will ever check any of that. The OPTIONS_FLOW
scorer reads `options_flow_magnitude`, the contradiction gate reads
`options_flow_direction`, and both are plausible-looking numbers in their
ranges whatever happened underneath. Six things here can be silently wrong,
and the tests are organized around them.

**The cap has to BIND, not merely exist.** A cap that is never reached is
not a cap, and "flagged" describing a restriction that was never applied is
worse than having neither. So the cap is tested twice and never by reading
`options_eod_capped` alone. First on the exact tape, where the uncapped
magnitude is a number written into this file
(`0.25*0.5 + 0.25*0.4 + 0.20*tanh(0.8) + 0.15*1 + 0.15*1 = 0.65780735...`)
and the EOD twin of the *same chain* must report exactly `0.50` with the
flag set, while the intraday twin reports the uncapped value with the flag
clear -- the difference is the cap and nothing else. Then empirically, over
every bar of a five-month EOD dataset, where the cap must bind on a large
share of bars and the uncapped distribution is reported. The intraday twin
of that dataset must never apply it.

**`None` is NOT SUPPLIED and `0.0` is an observation.** `OptionsSnapshot`
exists to keep those apart, and the one test that proves they stayed apart
is `test_a_zero_term_and_a_dropped_term_are_distinguishable_in_the_output`:
two chains whose `options_flow_magnitude` is the *identical*
`0.5328073540535698`, one because its premium imbalance measured exactly
zero and one because its premium term was dropped. They are separated by
`options_available_weight` (1.0 against 0.75), by
`options_terms_available` (5 against 4) and by the consensus, whose
denominator includes an available zero term and excludes a dropped one. If
those three ever agreed, the component would be reporting confidence the
chain does not support, and the magnitude alone could not say so.

**No weight is redistributed.** Dropping a term must LOWER the magnitude by
that term's whole contribution, not renormalize back up to the same number.
`test_dropping_a_term_lowers_the_magnitude_and_does_not_renormalize`
computes both numbers -- 0.6578073540535698 with all five terms and
0.5578073540535698 without delta-weighted volume -- plus 0.7437431387380931,
which is what a renormalizing implementation would have produced. The
assertion is on the weight, not on a note.

**Direction is carried separately from magnitude.** Section 7's reason is
explicit: "this avoids the common bug where a strong bearish component
inflates a bullish score". Tested by mirroring a chain side for side: the
magnitude must be bit-identical and the consensus exactly negated. Also
tested at the two places a direction may NOT be reported -- weight split
exactly half each way (`MIN_SIGN_AGREEMENT` is a strict majority, so the
comparison is `>`), and a consensus of exactly zero.

**Warmup honesty, which the lookahead harness cannot check here.**
`warmup_bars` is 21 *bars* while the real gate is 21 *snapshots*, so check 3
of `validation/lookahead.py` is a weak test for this computer -- and
measurably so: an UNDER-declared `warmup_bars` produces zero audit findings
(`test_the_audit_cannot_catch_an_under_declared_warmup_bars_here`), because
the snapshot gate refuses first. The snapshot window is therefore tested
directly: prepending nineteen wildly different older snapshots must not move
a single key, and a subclass that reads thirty snapshots while declaring
twenty-one moves seven of them -- so the test can fail.

**Nothing here asserts intent.** Section 5 rejects reading options flow as
evidence of institutional activity, and no key name or runtime note may say
otherwise. `test_no_key_name_or_runtime_note_claims_intent_or_an_actor`
collects the notes actually emitted across every scenario in this file and
scans them, rather than scanning the docstrings, which quote the rejected
reading in order to reject it.

Three things in here are honest gaps rather than passing tests, each said
where it sits:

* **Nothing in this file tests predictive content, and no test may.** The
  only data at Phase 5 is `data/synthetic.py`, whose options series is
  generated from the same innovations as its bars
  (`_options_series`: `signal = rho * standardized + ...` where
  `standardized` is the session's own realized return). A correlation
  between these features and its forward returns is a property of the
  generator. The synthetic datasets here are used for mechanics only --
  bounds, determinism, the cap's binding rate, lookahead. I was tempted:
  the cap-binding test loops over every bar and the forward return is one
  array away. It is not computed.
* `relative_rank` rejects a window with *no* dispersion, but a partially
  tied window still inflates the rank, because `percentile_rank` counts
  values at or below. That is a property of the shared helper in
  `features/base.py`, documented for quoted spreads in `liquidity.py`, and
  is not pinned here.
* `MIN_SIGN_AGREEMENT` and `MIN_PERCENTILE_SAMPLE_SHARE` are module
  constants the Phase 8 sweep cannot move. They are tested as *encoded*
  (a strict majority, half the window) and not as correct; whether either
  value is right is a measurement question with no measurement yet.

**The tape, and why its arithmetic is exact.** Twenty-one snapshots. Twenty
identical background rows, then one "now" row that differs in exactly two
columns. Every field is a small integer or a binary-exact fraction, chosen so
that each of the five terms is a number written into the test:

    net premium        (300 - 100) / (300 + 100)                 = 0.5
    delta volume       (70 - 30) / (70 + 30)                      = 0.4
    OI change          tanh(((500 - 100) / (6000 + 4000)) / 0.05) = tanh(0.8)
    25d skew           put IV 0.5 vs 0.25 is the window maximum,
                       so p = 21/21 and 2p - 1                    = 1.0
    gamma proxy        |500| is the window maximum, so p          = 1.0

`iv_25d_put`, `iv_25d_call` and the premiums are all exactly representable in
binary, so there is no rounding anywhere in the five terms, and the weighted
sum is exact. `test_the_exact_tape_is_what_this_file_claims_it_is` asserts
the premise before anything relies on it.

Builders are local on purpose: file ownership, and so a failure localizes
here rather than to a shared fixture.
"""

from __future__ import annotations

import inspect
import math
from datetime import date, datetime, timedelta, timezone
from typing import get_type_hints

import numpy as np
import pytest

from flow_model.config.loader import load_config
from flow_model.config.schema import OptionsFlowConfig, OptionsFlowWeights
from flow_model.core.contracts import ComponentScore, FeatureVector
from flow_model.core.enums import Component, DataQuality, Feed
from flow_model.data import SyntheticConfig, SyntheticMarketGenerator, TradingCalendar
from flow_model.data.market_view import MarketView
from flow_model.data.series import BarSeries, OptionsSeries, from_ns, to_ns_array
from flow_model.data.store import SymbolData
from flow_model.features.base import FeatureBundle
from flow_model.features import optionsflow as optionsflow_module
from flow_model.features.optionsflow import (
    DIRECTION_CALL_SIDE,
    DIRECTION_NONE,
    DIRECTION_PUT_SIDE,
    FLAG_CLEAR,
    FLAG_SET,
    GAMMA_COLUMN,
    MIN_PERCENTILE_SAMPLE_SHARE,
    MIN_SIGN_AGREEMENT,
    SKEW_COLUMNS,
    TERM_DELTA_VOLUME,
    TERM_GAMMA,
    TERM_NET_PREMIUM,
    TERM_OI_CHANGE,
    TERM_ORDER,
    TERM_SKEW,
    WEIGHT_EPSILON,
    OptionsFlowFeatures,
    imbalance_ratio,
    partial_window_note,
    relative_rank,
)
from flow_model.features.volatility import VolatilityFeatures
from flow_model.validation.lookahead import (
    _sample_indices,
    assert_no_lookahead,
    audit_computer,
)

SYMBOL = "NQ"
INTERVAL = 300
TS0 = datetime(2021, 3, 1, 14, 35, tzinfo=timezone.utc)

#: Every optional chain column, so a builder can omit one by name and the
#: series still carries the rest. Named here rather than inferred, so a
#: schema change in `OptionsSeries.OPTIONAL` shows up as a failure here.
CHAIN_COLUMNS = (
    "call_volume",
    "put_volume",
    "call_premium",
    "put_premium",
    "call_oi",
    "put_oi",
    "call_oi_change",
    "put_oi_change",
    "delta_weighted_call_volume",
    "delta_weighted_put_volume",
    "iv_25d_put",
    "iv_25d_call",
    "gamma_exposure_proxy",
)

#: The background row of the exact tape. See the module docstring.
BACKGROUND = {
    "call_volume": 600.0,
    "put_volume": 400.0,
    "call_premium": 300.0,
    "put_premium": 100.0,
    "call_oi": 6000.0,
    "put_oi": 4000.0,
    "call_oi_change": 500.0,
    "put_oi_change": 100.0,
    "delta_weighted_call_volume": 70.0,
    "delta_weighted_put_volume": 30.0,
    "iv_25d_put": 0.375,
    "iv_25d_call": 0.25,
    "gamma_exposure_proxy": 100.0,
}

#: The newest row. Differs from the background in exactly two columns, so the
#: two percentile terms rank their own current observation at the top of the
#: window and everything else is the background value.
NOW = {**BACKGROUND, "iv_25d_put": 0.5, "gamma_exposure_proxy": 500.0}

# --- the five hand-computed terms of the exact tape ------------------------

T_NET_PREMIUM = 0.5  # (300 - 100) / (300 + 100)
T_DELTA_VOLUME = 0.4  # (70 - 30) / (70 + 30)
T_OI_CHANGE = math.tanh(((500.0 - 100.0) / (6000.0 + 4000.0)) / 0.05)  # tanh(0.8)
T_SKEW = 1.0  # 2 * (21/21) - 1; the newest skew 0.25 is the window maximum
T_GAMMA = 1.0  # 21/21; |500| is the window maximum of |gamma_exposure_proxy|

W_NET_PREMIUM, W_DELTA_VOLUME, W_OI_CHANGE, W_SKEW, W_GAMMA = (
    0.25,
    0.25,
    0.20,
    0.15,
    0.15,
)

#: sum(w_i * |t_i|) over all five terms of the exact tape.
UNCAPPED = (
    W_NET_PREMIUM * T_NET_PREMIUM
    + W_DELTA_VOLUME * T_DELTA_VOLUME
    + W_OI_CHANGE * T_OI_CHANGE
    + W_SKEW * T_SKEW
    + W_GAMMA * T_GAMMA
)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def options_config(**overrides) -> OptionsFlowConfig:
    """The shipped options-flow config, optionally overridden.

    `replace` rather than `model_copy`: the config's own validators are the
    things that refuse a cap at 1.0 and a disabled gamma term with a positive
    weight, and a test that bypassed them would be testing a config the
    system cannot be run with.
    """
    return load_config().options_flow.replace(**overrides)


def weights(**overrides) -> OptionsFlowWeights:
    return OptionsFlowWeights(**overrides)


def computer(config: OptionsFlowConfig | None = None) -> OptionsFlowFeatures:
    return OptionsFlowFeatures(config if config is not None else options_config())


def stamps(n: int, start: int = 1) -> np.ndarray:
    """`n` timestamps on the bar-close grid, starting at the `start`-th bar."""
    return to_ns_array(
        TS0 + timedelta(seconds=INTERVAL * (start + i)) for i in range(n)
    )


def bar_series(n: int) -> BarSeries:
    """A flat bar tape. Its values are deliberately uninformative.

    `optionsflow.py` declares `required_feeds = {OPTIONS_SNAPSHOT}` and reads
    no bar column, so these bars exist only to satisfy `SymbolData` (which
    requires bars at the primary interval) and to let `MarketView` be built.
    `test_perturbing_every_bar_column_moves_nothing` perturbs them to prove
    they are not read.
    """
    close = np.full(n, 100.0)
    return BarSeries(
        symbol=SYMBOL,
        ts_ns=stamps(n),
        interval_seconds=INTERVAL,
        columns={
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
    )


def chain(rows, *, intraday: bool, start: int = 1) -> OptionsSeries:
    """An `OptionsSeries` from a list of row dicts, one snapshot per bar.

    A column is present iff at least one row names it, and a row that omits
    a present column carries `nan` there -- which `OptionsSeries.snapshot_at`
    converts to `None` for the optional columns, i.e. NOT SUPPLIED. That is
    how this file builds a chain with a `nan` hole as distinct from a chain
    with the column absent entirely.
    """
    present = sorted({name for row in rows for name in row})
    columns = {
        name: np.array(
            [float(row.get(name, float("nan"))) for row in rows], dtype=np.float64
        )
        for name in present
    }
    return OptionsSeries(
        symbol=SYMBOL,
        ts_ns=stamps(len(rows), start=start),
        columns=columns,
        meta={
            "is_intraday": bool(intraday),
            "source": "synthetic_intraday" if intraday else "synthetic_eod",
        },
    )


def dataset(rows, *, intraday: bool = True, n_bars: int | None = None) -> SymbolData:
    total = n_bars if n_bars is not None else len(rows)
    return SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: bar_series(total)},
        options=chain(rows, intraday=intraday),
    )


def view_of(data: SymbolData, index: int = -1) -> MarketView:
    """A view at the close of bar `index` of `data`'s primary bars."""
    series = data.primary_bars
    position = len(series) + index if index < 0 else index
    cutoff = int(series.ts_ns[position])
    return MarketView(data, now=from_ns(cutoff), now_ns=cutoff)


def compute(
    rows,
    *,
    intraday: bool = True,
    n_bars: int | None = None,
    config: OptionsFlowConfig | None = None,
    index: int = -1,
) -> FeatureVector:
    data = dataset(rows, intraday=intraday, n_bars=n_bars)
    return computer(config).compute(view_of(data, index))


def tape(*, now_overrides=None, background_overrides=None, omit=()) -> list[dict]:
    """The 21-row exact tape, with named edits.

    `omit` removes columns from every row, which makes them absent from the
    series and therefore `None` on every snapshot.
    """
    background = {**BACKGROUND, **(background_overrides or {})}
    newest = {**NOW, **(background_overrides or {}), **(now_overrides or {})}
    rows = [dict(background) for _ in range(20)] + [dict(newest)]
    if omit:
        rows = [{k: v for k, v in row.items() if k not in omit} for row in rows]
    return rows


def note_with(vector: FeatureVector, fragment: str) -> str:
    matching = [note for note in vector.notes if fragment in note]
    assert matching, f"no note containing {fragment!r}; notes were {vector.notes}"
    return matching[0]


def synthetic(
    *, intraday: bool, months: int = 5, seed: int = 7
) -> SymbolData:
    """A synthetic NQ dataset with an options chain.

    Mechanics only. The generator's options series is built from the same
    innovations as its bars, so nothing in this file may read a forward
    return from it. See the module docstring.
    """
    generated = SyntheticMarketGenerator(
        SyntheticConfig(include_options=True, options_intraday=intraday), seed=seed
    ).generate(
        SYMBOL,
        load_config().spec(SYMBOL),
        date(2017, 1, 1),
        date(2017, 1 + months, 1),
        INTERVAL,
        calendar=TradingCalendar(),
    )
    return SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: generated.bars},
        options=generated.options,
    )


def walk(data: SymbolData, comp: OptionsFlowFeatures, step: int = 1):
    """Every `step`-th bar's vector, scored ones only."""
    series = data.primary_bars
    for index in range(0, len(series), step):
        cutoff = int(series.ts_ns[index])
        vector = comp.compute(
            MarketView(data, now=from_ns(cutoff), now_ns=cutoff)
        )
        if vector.warmup_complete:
            yield index, vector


# ---------------------------------------------------------------------------
# the premise of the tape
# ---------------------------------------------------------------------------


def test_the_exact_tape_is_what_this_file_claims_it_is():
    """Every hand-computed number below depends on these properties of the
    tape, so they are asserted before anything relies on them -- a tape that
    quietly stopped having a 1000-contract volume or a binary-exact skew
    would make every term in this file wrong in a way that looked like a bug
    in the module."""
    rows = tape()
    assert len(rows) == 21 == options_config().warmup_snapshots

    newest = rows[-1]
    assert newest["call_volume"] + newest["put_volume"] == 1000.0
    assert newest["call_volume"] + newest["put_volume"] >= (
        options_config().min_snapshot_volume
    )
    assert newest["call_premium"] + newest["put_premium"] == 400.0
    assert newest["call_oi"] + newest["put_oi"] == 10000.0

    # Binary-exact, so (put - call) carries no rounding at all.
    assert newest["iv_25d_put"] - newest["iv_25d_call"] == 0.25
    assert rows[0]["iv_25d_put"] - rows[0]["iv_25d_call"] == 0.125

    skews = np.array([row["iv_25d_put"] - row["iv_25d_call"] for row in rows])
    assert skews[-1] == skews.max() and np.count_nonzero(skews == skews.max()) == 1
    gammas = np.abs([row["gamma_exposure_proxy"] for row in rows])
    assert gammas[-1] == gammas.max() and np.count_nonzero(gammas == gammas.max()) == 1

    # The newest row differs from the background in exactly two columns.
    assert {k for k in BACKGROUND if BACKGROUND[k] != NOW[k]} == {
        "iv_25d_put",
        "gamma_exposure_proxy",
    }

    view = view_of(dataset(rows))
    assert view.options_count() == 21 and view.bar_count() == 21
    assert view.options_are_intraday() is True


def test_the_shipped_config_is_the_one_every_number_here_assumes():
    """The hand arithmetic is written against the defaults. If the Phase 8
    sweep ever changes a default, this fails first and names which one,
    instead of twenty term tests failing with unexplained numbers."""
    config = options_config()
    assert config.lookback_snapshots == 20 and config.warmup_snapshots == 21
    assert config.min_snapshot_volume == 1.0
    assert config.oi_change_squash_scale == 0.05
    assert config.min_available_weight_fraction == 0.50
    assert config.eod_degraded_cap_fraction == 0.50
    assert config.require_intraday_prints is False
    assert config.use_gamma_exposure_proxy is True
    assert config.never_synthesize_missing_fields is True
    assert (
        config.weights.net_premium,
        config.weights.delta_weighted_volume,
        config.weights.oi_change,
        config.weights.skew_25d,
        config.weights.gamma_exposure,
    ) == (W_NET_PREMIUM, W_DELTA_VOLUME, W_OI_CHANGE, W_SKEW, W_GAMMA)
    assert TERM_ORDER == (
        TERM_NET_PREMIUM,
        TERM_DELTA_VOLUME,
        TERM_OI_CHANGE,
        TERM_SKEW,
        TERM_GAMMA,
    )


# ---------------------------------------------------------------------------
# the five named section-5 features, hand-computed
# ---------------------------------------------------------------------------


def test_net_premium_is_the_dimensionless_call_minus_put_imbalance():
    """Section 5 names "net premium (call-prem minus put-prem)". The module
    reports the dollar difference as a diagnostic and the *normalized*
    imbalance as the term, because an unnormalized dollar figure is not
    comparable across underlyings and is not bounded by construction -- and
    section 7 forbids an unbounded quantity entering a weighted sum."""
    fv = compute(tape())
    # (300 - 100) / (300 + 100) = 200 / 400
    assert fv.values["options_net_premium_imbalance"] == T_NET_PREMIUM
    assert fv.values["options_net_premium"] == 200.0  # dollars, the diagnostic
    assert fv.values["options_total_volume"] == 1000.0


def test_delta_weighted_volume_is_an_imbalance_of_magnitudes():
    """(70 - 30) / (70 + 30) = 0.4. The term is defined on |dw| for both
    sides because the contract does not say whether a provider signs
    `delta_weighted_put_volume` by the (negative) put delta. If it does,
    differencing the raw values would ADD the two sides and drive the
    denominator toward zero, producing a wild imbalance from ordinary
    data."""
    fv = compute(tape())
    assert fv.values["options_delta_volume_imbalance"] == T_DELTA_VOLUME


def test_delta_weighted_volume_is_invariant_to_the_put_delta_sign_convention():
    """The same chain with `delta_weighted_put_volume` negated is the same
    measurement. This is the test that would fail if someone "simplified"
    the magnitudes away, and the failure it prevents is silent: a provider
    that signs puts negative would otherwise read as a near-total call-side
    imbalance on every snapshot."""
    signed_positive = compute(tape())
    signed_negative = compute(
        tape(background_overrides={"delta_weighted_put_volume": -30.0})
    )
    assert (
        signed_negative.values["options_delta_volume_imbalance"]
        == signed_positive.values["options_delta_volume_imbalance"]
        == T_DELTA_VOLUME
    )


def test_oi_change_is_a_squashed_fraction_of_total_open_interest():
    """net = 500 - 100 = 400; fraction = 400 / 10000 = 0.04;
    term = tanh(0.04 / 0.05) = tanh(0.8).

    A fraction rather than a contract count, so it is comparable across
    underlyings; a `tanh` rather than a z-score, so one extreme session
    cannot dominate the weighted sum."""
    fv = compute(tape())
    assert fv.values["options_oi_change_imbalance"] == pytest.approx(
        math.tanh(0.8), abs=1e-15
    )
    assert T_OI_CHANGE == pytest.approx(0.6640367702678489, abs=1e-15)


def test_oi_change_carries_the_sign_of_the_net_change():
    """Put-side OI growing faster than call-side OI is a put-side reading.
    net = 100 - 600 = -500; fraction = -0.05; term = -tanh(1.0).

    The sign has to be reapplied by hand after `squash`, which takes `abs`
    by design, so this is the one place a sign could be dropped and the term
    would still look like a plausible magnitude."""
    fv = compute(
        tape(background_overrides={"call_oi_change": 100.0, "put_oi_change": 600.0})
    )
    assert fv.values["options_oi_change_imbalance"] == pytest.approx(
        -math.tanh((600.0 - 100.0) / 10000.0 / 0.05), abs=1e-15
    )
    assert fv.values["options_oi_change_imbalance"] < 0.0


def test_the_25_delta_skew_term_is_its_own_rank_not_its_level():
    """The newest skew (0.25) is the maximum of its 21-snapshot window, so
    p = 21/21 and the term is 2p - 1 = 1.0, while the diagnostic reports the
    0.25 level.

    The level of index skew is structurally positive and says little; its
    position within its own recent distribution is the measurement. Ranking
    the level instead would make this term a near-constant +1 on every real
    chain."""
    fv = compute(tape())
    assert fv.values["options_skew_25d_pressure"] == T_SKEW
    assert fv.values["options_skew_25d"] == 0.25  # iv_25d_put - iv_25d_call
    assert fv.quality_by_key["options_skew_25d"] is DataQuality.GOOD


def test_the_skew_term_sits_at_zero_when_the_current_skew_is_the_window_median():
    """Ten snapshots below, ten above, the newest in the middle: p = 11/21
    and 2p - 1 = 1/21. Written out because `2p - 1` is the mapping that puts
    this term on the same [-1, 1] scale as the three imbalance ratios, and a
    sign error in it would be invisible on the tape above (where p = 1)."""
    rows = [dict(BACKGROUND, iv_25d_put=0.3125) for _ in range(10)]  # skew 0.0625
    rows += [dict(BACKGROUND, iv_25d_put=0.625) for _ in range(10)]  # skew 0.375
    rows += [dict(NOW, iv_25d_put=0.5)]  # skew 0.25, the 11th smallest
    fv = compute(rows)
    # p = count(<= 0.25) / 21 = 11 / 21
    assert fv.values["options_skew_25d_pressure"] == pytest.approx(
        2.0 * (11.0 / 21.0) - 1.0, abs=1e-15
    )


def test_the_gamma_proxy_is_a_magnitude_rank_and_carries_no_sign():
    """`|500|` is the maximum of the window's absolute values, so p = 21/21.

    The sign of a dealer hedging flow depends on which side of the chain
    dealers hold, which is the one thing nobody outside a dealer observes.
    So the proxy is ranked on its magnitude, and a chain whose gamma proxy is
    the negative of this one must produce the identical term."""
    positive = compute(tape())
    negated = compute(
        tape(
            background_overrides={"gamma_exposure_proxy": -100.0},
            now_overrides={"gamma_exposure_proxy": -500.0},
        )
    )
    assert positive.values["options_gamma_exposure_pressure"] == T_GAMMA
    assert negated.values["options_gamma_exposure_pressure"] == T_GAMMA
    assert 0.0 <= positive.values["options_gamma_exposure_pressure"] <= 1.0


def test_all_five_terms_and_the_magnitude_are_one_weighted_sum_written_out():
    """The single most important arithmetic test here: a weighted sum hides a
    dead term. If `options_gamma_exposure_pressure` were always 0, or the
    skew term always 1, `options_flow_magnitude` would still move with the
    others and still look like a score. So all five terms, both availability
    keys and the magnitude are pinned at once on a tape where each is forced
    by construction."""
    fv = compute(tape())

    assert fv.values["options_net_premium_imbalance"] == T_NET_PREMIUM
    assert fv.values["options_delta_volume_imbalance"] == T_DELTA_VOLUME
    assert fv.values["options_oi_change_imbalance"] == pytest.approx(
        T_OI_CHANGE, abs=1e-15
    )
    assert fv.values["options_skew_25d_pressure"] == T_SKEW
    assert fv.values["options_gamma_exposure_pressure"] == T_GAMMA

    assert fv.values["options_available_weight"] == 1.0
    assert fv.values["options_terms_available"] == 5.0

    # 0.25*0.5 + 0.25*0.4 + 0.20*tanh(0.8) + 0.15*1.0 + 0.15*1.0
    assert UNCAPPED == pytest.approx(0.6578073540535698, abs=1e-15)
    assert fv.values["options_flow_uncapped_magnitude"] == pytest.approx(
        UNCAPPED, abs=1e-15
    )
    assert fv.values["options_flow_magnitude"] == pytest.approx(UNCAPPED, abs=1e-15)


def test_the_magnitude_never_exceeds_the_weight_that_was_actually_available():
    """`magnitude = sum(w_i * |t_i|)` over available terms and each `|t_i|`
    is at most 1, so the magnitude is bounded by the available weight. That
    is the whole content of "the magnitude falls when data is missing": if it
    could exceed the available weight, a chain with two terms could score as
    high as a chain with five."""
    scenarios = {
        "all five": tape(),
        "no delta volume": tape(
            omit=("delta_weighted_call_volume", "delta_weighted_put_volume")
        ),
        "no oi change": tape(omit=("call_oi_change", "put_oi_change")),
        "no gamma": tape(omit=("gamma_exposure_proxy",)),
        "flat percentile windows": [dict(BACKGROUND) for _ in range(21)],
    }
    for label, rows in scenarios.items():
        fv = compute(rows)
        assert fv.warmup_complete, label
        available = fv.values["options_available_weight"]
        magnitude = fv.values["options_flow_magnitude"]
        assert 0.0 <= magnitude <= available + 1e-12, label
        assert 0.0 <= available <= 1.0, label


# ---------------------------------------------------------------------------
# direction, carried separately from magnitude
# ---------------------------------------------------------------------------


def test_the_direction_consensus_and_agreement_are_hand_computed():
    """The three imbalance terms vote with their own sign; the skew term
    votes INVERTED, because an elevated relative 25-delta put IV is stronger
    demand for downside protection, which is the put side; the gamma proxy
    does not vote.

    consensus = (0.25*0.5 + 0.25*0.4 + 0.20*tanh(0.8) + 0.15*(-1.0)) / 0.85
    agreement = (0.25 + 0.25 + 0.20) / 0.85 = 0.70 / 0.85"""
    fv = compute(tape())
    voting_weight = W_NET_PREMIUM + W_DELTA_VOLUME + W_OI_CHANGE + W_SKEW
    assert voting_weight == 0.85

    expected_consensus = (
        W_NET_PREMIUM * T_NET_PREMIUM
        + W_DELTA_VOLUME * T_DELTA_VOLUME
        + W_OI_CHANGE * T_OI_CHANGE
        + W_SKEW * (-T_SKEW)
    ) / voting_weight
    assert fv.values["options_imbalance_consensus"] == pytest.approx(
        expected_consensus, abs=1e-12
    )
    assert fv.values["options_sign_agreement"] == pytest.approx(
        0.70 / 0.85, abs=1e-12
    )
    assert fv.values["options_flow_direction"] == DIRECTION_CALL_SIDE


def test_an_elevated_put_iv_premium_votes_for_the_put_side():
    """The skew term is the one term whose sign convention has to be
    converted, so it gets its own test: with the three trade imbalances
    forced to the put side AND the skew elevated, every vote must agree and
    the direction must be -1. If the inversion were missing, the skew would
    fight the other three here and the agreement would fall to 0.70/0.85."""
    fv = compute(
        tape(
            background_overrides={
                "call_premium": 100.0,
                "put_premium": 300.0,
                "delta_weighted_call_volume": 30.0,
                "delta_weighted_put_volume": 70.0,
                "call_oi_change": 100.0,
                "put_oi_change": 500.0,
            }
        )
    )
    assert fv.values["options_skew_25d_pressure"] == T_SKEW  # elevated put IV
    assert fv.values["options_sign_agreement"] == 1.0
    assert fv.values["options_flow_direction"] == DIRECTION_PUT_SIDE
    assert fv.values["options_imbalance_consensus"] < 0.0


def test_mirroring_the_chain_negates_the_direction_and_leaves_the_magnitude():
    """Section 7: "direction is handled separately from magnitude... this
    avoids the common bug where a strong bearish component inflates a
    bullish score."

    The two chains here are each other's side-for-side mirror, so a
    magnitude that moved at all would mean the sign had leaked into it. The
    two percentile terms are omitted because a percentile rank of a window
    is not a side-symmetric quantity, which would make the mirror inexact
    for reasons that are not about this property."""
    omitted = ("iv_25d_put", "iv_25d_call", "gamma_exposure_proxy")
    call_heavy = tape(omit=omitted)
    put_heavy = [
        {
            "call_volume": row["put_volume"],
            "put_volume": row["call_volume"],
            "call_premium": row["put_premium"],
            "put_premium": row["call_premium"],
            "call_oi": row["put_oi"],
            "put_oi": row["call_oi"],
            "call_oi_change": row["put_oi_change"],
            "put_oi_change": row["call_oi_change"],
            "delta_weighted_call_volume": row["delta_weighted_put_volume"],
            "delta_weighted_put_volume": row["delta_weighted_call_volume"],
        }
        for row in call_heavy
    ]
    up, down = compute(call_heavy), compute(put_heavy)

    # 0.25*0.5 + 0.25*0.4 + 0.20*tanh(0.8), with skew and gamma dropped
    expected = (
        W_NET_PREMIUM * T_NET_PREMIUM
        + W_DELTA_VOLUME * T_DELTA_VOLUME
        + W_OI_CHANGE * T_OI_CHANGE
    )
    assert up.values["options_flow_magnitude"] == pytest.approx(expected, abs=1e-15)
    assert (
        down.values["options_flow_magnitude"]
        == up.values["options_flow_magnitude"]
    )
    assert down.values["options_imbalance_consensus"] == pytest.approx(
        -up.values["options_imbalance_consensus"], abs=1e-15
    )
    assert up.values["options_flow_direction"] == DIRECTION_CALL_SIDE
    assert down.values["options_flow_direction"] == DIRECTION_PUT_SIDE
    assert (
        down.values["options_sign_agreement"]
        == up.values["options_sign_agreement"]
        == 1.0
    )


def test_the_magnitude_is_a_valid_component_score_for_either_direction():
    """`ComponentScore` constrains magnitude to [0, 1] and direction to
    {-1, 0, +1}, and it is the object the Flow Score sums. A component that
    emitted a magnitude outside [0, 1] -- or a signed magnitude -- would be
    rejected there, at which point the backtest stops rather than scoring
    wrongly; proving it constructs for both sides is cheaper than
    discovering it in Phase 6."""
    for rows in (tape(), tape(background_overrides={"call_premium": 100.0,
                                                    "put_premium": 300.0})):
        fv = compute(rows)
        score = ComponentScore(
            component=Component.OPTIONS_FLOW,
            magnitude=fv.values["options_flow_magnitude"],
            direction=int(fv.values["options_flow_direction"]),
            weight=20.0,
            quality=fv.quality,
        )
        assert 0.0 <= score.magnitude <= 1.0
        assert score.direction in (-1, 0, 1)


def test_weight_split_exactly_half_each_way_is_disagreement_not_a_direction():
    """`MIN_SIGN_AGREEMENT` is a strict majority of the signed voting weight,
    so the comparison is `>` and not `>=`.

    Net premium (+0.5, weight 0.25) against delta-weighted volume (-0.4,
    weight 0.25), with the other three terms dropped: agreement is exactly
    0.25/0.50 = 0.5, the consensus is non-zero (0.025/0.5 = 0.05), and the
    direction must still be 0. Reporting +1 here would turn a tie into a
    verdict."""
    rows = tape(
        background_overrides={
            "delta_weighted_call_volume": 30.0,
            "delta_weighted_put_volume": 70.0,
        },
        omit=(
            "call_oi_change",
            "put_oi_change",
            "iv_25d_put",
            "iv_25d_call",
            "gamma_exposure_proxy",
        ),
    )
    fv = compute(rows)
    assert fv.values["options_available_weight"] == pytest.approx(0.50, abs=1e-12)
    assert fv.values["options_net_premium_imbalance"] == 0.5
    assert fv.values["options_delta_volume_imbalance"] == -0.4
    # (0.25*0.5 + 0.25*(-0.4)) / 0.50 = 0.025 / 0.50
    assert fv.values["options_imbalance_consensus"] == pytest.approx(0.05, abs=1e-12)
    assert fv.values["options_sign_agreement"] == pytest.approx(0.5, abs=1e-12)
    assert fv.values["options_sign_agreement"] == MIN_SIGN_AGREEMENT
    assert fv.values["options_flow_direction"] == DIRECTION_NONE


def test_a_hair_over_half_the_signed_weight_does_report_a_direction():
    """The other side of the strict-majority boundary, so the previous test
    cannot be passing because the direction is always 0. Net premium (0.25)
    and the OI change (0.20) against delta-weighted volume (0.25): the
    agreement is 0.45/0.70, which is over half, and the direction is +1."""
    rows = tape(
        background_overrides={
            "delta_weighted_call_volume": 30.0,
            "delta_weighted_put_volume": 70.0,
        },
        omit=("iv_25d_put", "iv_25d_call", "gamma_exposure_proxy"),
    )
    fv = compute(rows)
    assert fv.values["options_sign_agreement"] == pytest.approx(
        0.45 / 0.70, abs=1e-12
    )
    assert fv.values["options_sign_agreement"] > MIN_SIGN_AGREEMENT
    assert fv.values["options_flow_direction"] == DIRECTION_CALL_SIDE


def test_a_consensus_of_exactly_zero_reports_no_direction():
    """Equal weights, equal magnitudes, opposite signs: +0.5 against -0.5 at
    0.25 each. The consensus is exactly 0.0 and there is no sign to report.

    `options_sign_agreement` is 0.0 here by the module's stated convention
    (agreement is measured against `sign(consensus)`, which is undefined at
    zero) -- NOT because no term carried a sign. Both terms do. A consumer
    that needed to tell "perfectly split" from "nothing signed" would have to
    read `options_terms_available`; that is a reporting limitation, recorded
    here rather than asserted as desirable."""
    rows = tape(
        background_overrides={
            "delta_weighted_call_volume": 30.0,
            "delta_weighted_put_volume": 90.0,  # (30 - 90) / 120 = -0.5
        },
        omit=(
            "call_oi_change",
            "put_oi_change",
            "iv_25d_put",
            "iv_25d_call",
            "gamma_exposure_proxy",
        ),
    )
    fv = compute(rows)
    assert fv.values["options_net_premium_imbalance"] == 0.5
    assert fv.values["options_delta_volume_imbalance"] == -0.5
    assert fv.values["options_imbalance_consensus"] == 0.0
    assert fv.values["options_flow_direction"] == DIRECTION_NONE
    assert fv.values["options_sign_agreement"] == 0.0
    # The magnitudes did not cancel: 0.25*0.5 + 0.25*0.5
    assert fv.values["options_flow_magnitude"] == 0.25


def test_the_agreement_denominator_is_the_signed_weight_not_the_voting_weight():
    """Regression for a hole this file did not cover.

    The module states two different denominators on purpose: the CONSENSUS is
    divided by the whole available voting weight, so a term that measured no
    imbalance dilutes it, while AGREEMENT is divided only by the voting
    weight that carries a sign, "because agreement asks about the terms that
    point somewhere". Nothing pinned the second one -- every earlier fixture
    had all its voting terms signed, which makes the two denominators equal.

    This tape separates them. Premium is balanced (a real 0.0 reading, so the
    term is available and votes nothing), delta-weighted volume is wholly
    call-side, and OI change is almost wholly put-side; skew and the gamma
    proxy are absent:

        voting weight = 0.25 + 0.25 + 0.20 = 0.70
        signed weight =        0.25 + 0.20 = 0.45
        consensus = (0.25*0 + 0.25*1 + 0.20*-tanh(10)) / 0.70 = +0.0714...
        agreement = 0.25 / 0.45 = 0.5555... > 0.5  -> a call-side direction

    Dividing by the voting weight instead gives 0.25/0.70 = 0.357, which is
    below the strict-majority threshold, so the reported DIRECTION -- not
    merely the agreement number -- would flip to none."""
    rows = tape(
        background_overrides={
            "call_premium": 200.0,
            "put_premium": 200.0,
            "delta_weighted_call_volume": 100.0,
            "delta_weighted_put_volume": 0.0,
            "call_oi_change": 0.0,
            "put_oi_change": 5000.0,
        },
        omit=("iv_25d_put", "iv_25d_call", "gamma_exposure_proxy"),
    )
    fv = compute(rows)

    oi_term = -math.tanh((5000.0 / 10000.0) / 0.05)
    assert fv.values["options_net_premium_imbalance"] == 0.0
    assert fv.values["options_delta_volume_imbalance"] == 1.0
    assert fv.values["options_oi_change_imbalance"] == pytest.approx(
        oi_term, abs=1e-15
    )
    assert fv.values["options_available_weight"] == pytest.approx(0.70, abs=1e-12)
    assert fv.values["options_terms_available"] == 3.0

    consensus = (W_DELTA_VOLUME * 1.0 + W_OI_CHANGE * oi_term) / 0.70
    assert fv.values["options_imbalance_consensus"] == pytest.approx(
        consensus, abs=1e-15
    )
    assert consensus > 0.0

    assert fv.values["options_sign_agreement"] == pytest.approx(
        W_DELTA_VOLUME / 0.45, abs=1e-15
    )
    assert fv.values["options_sign_agreement"] > MIN_SIGN_AGREEMENT
    assert fv.values["options_flow_direction"] == DIRECTION_CALL_SIDE

    # The wrong denominator is below the threshold, so the assertion above is
    # discriminating rather than incidentally true.
    assert W_DELTA_VOLUME / 0.70 <= MIN_SIGN_AGREEMENT


def test_agreement_is_measured_against_the_consensus_sign_not_the_weight_majority():
    """Regression for a hole this file did not cover.

    `agreement` asks how much of the signed weight points the way the
    CONSENSUS points. It is not "which side carries more weight", and the two
    differ whenever a small-magnitude vote carries the larger weight: a term
    can hold the weight majority while contributing almost nothing to the
    weighted sum.

    Here delta-weighted volume votes +0.01 on 0.25 of the weight and OI
    change votes -tanh(10) on 0.20, with premium balanced and the two
    percentile terms absent:

        consensus = (0.25*0.01 + 0.20*-tanh(10)) / 0.70 = -0.282...
        agreement = 0.20 / 0.45 = 0.444... <= 0.5  -> NO direction

    Measuring agreement against the heavier side instead would give
    0.25/0.45 = 0.5555 and report a put-side direction (the consensus is
    negative) off a chain where most of the signed weight votes call-side --
    a direction manufactured from disagreement, which is exactly what the
    threshold exists to refuse."""
    rows = tape(
        background_overrides={
            "call_premium": 200.0,
            "put_premium": 200.0,
            "delta_weighted_call_volume": 50.5,
            "delta_weighted_put_volume": 49.5,
            "call_oi_change": 0.0,
            "put_oi_change": 5000.0,
        },
        omit=("iv_25d_put", "iv_25d_call", "gamma_exposure_proxy"),
    )
    fv = compute(rows)

    oi_term = -math.tanh((5000.0 / 10000.0) / 0.05)
    assert fv.values["options_delta_volume_imbalance"] == pytest.approx(
        0.01, abs=1e-15
    )
    assert fv.values["options_oi_change_imbalance"] == pytest.approx(
        oi_term, abs=1e-15
    )

    consensus = (W_DELTA_VOLUME * 0.01 + W_OI_CHANGE * oi_term) / 0.70
    assert fv.values["options_imbalance_consensus"] == pytest.approx(
        consensus, abs=1e-15
    )
    assert consensus < 0.0

    assert fv.values["options_sign_agreement"] == pytest.approx(
        W_OI_CHANGE / 0.45, abs=1e-15
    )
    assert fv.values["options_sign_agreement"] <= MIN_SIGN_AGREEMENT
    assert fv.values["options_flow_direction"] == DIRECTION_NONE

    # The weight majority points the OTHER way and would clear the threshold,
    # so this is a case the two rules genuinely separate.
    assert W_DELTA_VOLUME > W_OI_CHANGE
    assert W_DELTA_VOLUME / 0.45 > MIN_SIGN_AGREEMENT

    # The magnitude is unaffected: it is a sum of |t_i| and carries no sign.
    assert fv.values["options_flow_magnitude"] == pytest.approx(
        W_DELTA_VOLUME * 0.01 + W_OI_CHANGE * abs(oi_term), abs=1e-15
    )


def test_the_gamma_proxy_moves_the_magnitude_and_never_the_direction():
    """Its weight contributes to the magnitude only. Two tapes differing
    ONLY in the gamma proxy's current value must give different magnitudes
    and bit-identical consensus, agreement and direction. If the proxy ever
    acquired a vote, this is where it would show, and the claim it would
    smuggle in -- that dealer hedging has an observable sign -- is exactly
    what section 5 rejects."""
    high = compute(tape())
    low = compute(tape(now_overrides={"gamma_exposure_proxy": 1.0}))

    # |1.0| is now the window minimum: p = 1/21.
    assert low.values["options_gamma_exposure_pressure"] == pytest.approx(
        1.0 / 21.0, abs=1e-15
    )
    assert (
        low.values["options_flow_magnitude"] < high.values["options_flow_magnitude"]
    )
    for key in (
        "options_imbalance_consensus",
        "options_sign_agreement",
        "options_flow_direction",
    ):
        assert low.values[key] == high.values[key], key


# ---------------------------------------------------------------------------
# None is NOT SUPPLIED; 0.0 is an observation
# ---------------------------------------------------------------------------


def test_a_zero_term_and_a_dropped_term_are_distinguishable_in_the_output():
    """`OptionsSnapshot` keeps `None` (NOT SUPPLIED) apart from `0.0` (an
    observation) and this is the test that the distinction survived into the
    feature vector.

    Both chains produce the IDENTICAL magnitude, because a term measuring
    exactly zero contributes exactly zero to a weighted sum of magnitudes and
    so does a dropped term. They are separated by three other keys: the
    available weight (1.0 against 0.75), the term count (5 against 4) and the
    consensus, whose denominator includes an available zero term and excludes
    a dropped one. If all three agreed, a chain that never supplied its
    premium would be indistinguishable from a perfectly balanced one."""
    balanced = compute(
        tape(background_overrides={"call_premium": 200.0, "put_premium": 200.0})
    )
    unsupplied = compute(
        tape(background_overrides={"call_premium": 0.0, "put_premium": 0.0})
    )

    # 0.25*0.0 + 0.25*0.4 + 0.20*tanh(0.8) + 0.15 + 0.15
    expected = (
        W_DELTA_VOLUME * T_DELTA_VOLUME
        + W_OI_CHANGE * T_OI_CHANGE
        + W_SKEW * T_SKEW
        + W_GAMMA * T_GAMMA
    )
    assert balanced.values["options_flow_magnitude"] == pytest.approx(
        expected, abs=1e-15
    )
    assert (
        unsupplied.values["options_flow_magnitude"]
        == balanced.values["options_flow_magnitude"]
    )

    assert balanced.values["options_available_weight"] == 1.0
    assert balanced.values["options_terms_available"] == 5.0
    assert balanced.quality_by_key["options_net_premium_imbalance"] is DataQuality.GOOD

    assert unsupplied.values["options_available_weight"] == 0.75
    assert unsupplied.values["options_terms_available"] == 4.0
    assert (
        unsupplied.quality_by_key["options_net_premium_imbalance"]
        is DataQuality.DEGRADED
    )

    # consensus denominator: 0.85 with the zero term, 0.60 without it.
    numerator = (
        W_DELTA_VOLUME * T_DELTA_VOLUME
        + W_OI_CHANGE * T_OI_CHANGE
        + W_SKEW * (-T_SKEW)
    )
    assert balanced.values["options_imbalance_consensus"] == pytest.approx(
        numerator / 0.85, abs=1e-12
    )
    assert unsupplied.values["options_imbalance_consensus"] == pytest.approx(
        numerator / 0.60, abs=1e-12
    )
    assert (
        balanced.values["options_imbalance_consensus"]
        != unsupplied.values["options_imbalance_consensus"]
    )


def test_dropping_a_term_lowers_the_magnitude_and_does_not_renormalize():
    """Section 14.5's rule, applied to the whole component: a dropped term's
    weight is subtracted and NOT reallocated.

    All five terms give 0.6578073540535698. Without delta-weighted volume the
    answer must be 0.5578073540535698 -- lower by exactly 0.25 * 0.4. A
    renormalizing implementation would divide by the surviving 0.75 and
    report 0.7437431387380931, which is HIGHER than the full-data answer:
    missing data would have manufactured confidence. The assertion is on the
    weight, not on a note."""
    full = compute(tape())
    partial = compute(
        tape(omit=("delta_weighted_call_volume", "delta_weighted_put_volume"))
    )

    expected = UNCAPPED - W_DELTA_VOLUME * T_DELTA_VOLUME
    assert expected == pytest.approx(0.5578073540535698, abs=1e-15)
    assert partial.values["options_flow_magnitude"] == pytest.approx(
        expected, abs=1e-15
    )
    assert partial.values["options_available_weight"] == 0.75
    assert partial.values["options_terms_available"] == 4.0

    renormalized = expected / 0.75
    assert renormalized == pytest.approx(0.7437431387380931, abs=1e-12)
    assert renormalized > full.values["options_flow_magnitude"]
    assert partial.values["options_flow_magnitude"] != pytest.approx(
        renormalized, abs=1e-6
    )
    note_with(partial, "DROPPED without redistributing its weight")
    note_with(partial, "None means NOT SUPPLIED and is not read as 0.0")


def test_oi_change_is_never_reconstructed_by_differencing_the_oi_levels():
    """The chain carries `call_oi` and `put_oi` on all 21 snapshots, so the
    first difference is one `np.diff` away -- which is exactly why this needs
    a test. A differenced level would be indistinguishable downstream from a
    supplied `call_oi_change`, it assumes two adjacent snapshots are adjacent
    sessions (false across a feed gap), and it erases the None-versus-0.0
    distinction. `never_synthesize_missing_fields` and the brief's "never
    invent missing options data" both forbid it.

    The cost is real and is reported rather than hidden: a chain supplying
    only OI levels -- a common vendor shape -- loses the whole 0.20."""
    rows = tape(
        background_overrides={"call_oi": 6000.0, "put_oi": 4000.0},
        omit=("call_oi_change", "put_oi_change"),
    )
    # Make the levels walk, so a differencing implementation would have a
    # non-zero change to report on every snapshot.
    for index, row in enumerate(rows):
        row["call_oi"] = 6000.0 + 100.0 * index
        row["put_oi"] = 4000.0 - 10.0 * index
    fv = compute(rows)

    assert fv.values["options_oi_change_imbalance"] == 0.0
    assert fv.quality_by_key["options_oi_change_imbalance"] is DataQuality.DEGRADED
    assert fv.values["options_available_weight"] == pytest.approx(0.80, abs=1e-12)
    assert fv.values["options_terms_available"] == 4.0
    note = note_with(fv, "NOT reconstructed by differencing")
    assert "never_synthesize_missing_fields" in note


def test_one_supplied_oi_change_side_is_not_enough():
    """`call_oi_change` present and `put_oi_change` absent means the NET
    change was never observed. Reading the absent side as 0.0 would report a
    pure call-side OI build from a chain that said nothing about puts."""
    rows = tape()
    for row in rows:
        row.pop("put_oi_change")
    fv = compute(rows)
    assert fv.values["options_oi_change_imbalance"] == 0.0
    assert fv.values["options_terms_available"] == 4.0
    assert "'put_oi_change'" in note_with(fv, "not supplied by the chain")


def test_a_chain_with_no_open_interest_drops_the_oi_term_rather_than_scoring_zero():
    """A hole this file did not cover: the OI term's DENOMINATOR, as against
    its numerator.

    `call_oi_change` and `put_oi_change` are both supplied and both real, but
    total open interest is zero, so there is nothing for the net change to be
    a fraction OF. `safe_divide` would return its 0.0 default and the term
    would be reported as available with an exactly-zero imbalance -- "open
    interest did not shift", which is a real observation this chain did not
    make. The term is dropped instead, which costs its 0.20 and says so."""
    fv = compute(tape(background_overrides={"call_oi": 0.0, "put_oi": 0.0}))

    assert fv.values["options_oi_change_imbalance"] == 0.0
    assert fv.quality_by_key["options_oi_change_imbalance"] is DataQuality.DEGRADED
    assert fv.values["options_available_weight"] == pytest.approx(0.80, abs=1e-12)
    assert fv.values["options_terms_available"] == 4.0
    # 1.0 - 0.20: the dropped weight is subtracted, not reallocated.
    assert fv.values["options_flow_magnitude"] == pytest.approx(
        UNCAPPED - W_OI_CHANGE * T_OI_CHANGE, abs=1e-15
    )
    note_with(fv, "no denominator to be a fraction of")

    # Supplied-and-zero is the other half of the same distinction: the same
    # chain with a real OI base and a genuinely zero net change DOES score,
    # and scores zero on that term.
    zero_change = compute(
        tape(background_overrides={"call_oi_change": 250.0, "put_oi_change": 250.0})
    )
    assert zero_change.values["options_oi_change_imbalance"] == 0.0
    assert zero_change.quality_by_key["options_oi_change_imbalance"] is (
        DataQuality.GOOD
    )
    assert zero_change.values["options_available_weight"] == 1.0
    assert zero_change.values["options_terms_available"] == 5.0


def test_a_nan_hole_in_an_optional_column_is_absence_not_zero():
    """`OptionsSeries` validates only its six REQUIRED columns, so an
    optional column can be present and hold `nan` on the newest row.
    `snapshot_at` converts that to `None`, i.e. NOT SUPPLIED -- and the term
    must drop rather than read the hole as a zero delta-weighted volume."""
    rows = tape()
    rows[-1] = dict(rows[-1], delta_weighted_call_volume=float("nan"))
    fv = compute(rows)
    assert fv.values["options_delta_volume_imbalance"] == 0.0
    assert fv.values["options_available_weight"] == 0.75
    assert "'delta_weighted_call_volume'" in note_with(
        fv, "not supplied by the chain"
    )


# ---------------------------------------------------------------------------
# the EOD degradation contract -- the Phase 5 gate
# ---------------------------------------------------------------------------


def test_an_eod_chain_flags_the_missing_timing_and_degrades_every_term():
    """The whole degradation contract as one contract.

    `OptionsSnapshot.is_intraday` is a property of the DATA, not a config
    guess about a vendor. On an EOD chain: the timing flag is 0, all five
    term keys and all five combination keys are DEGRADED, the vector's
    aggregate quality is DEGRADED, and a note names the fallback -- each term
    is a closed-session aggregate standing in for a reading at `view.now`,
    the two trade-flow terms in particular being whole-session premium and
    delta-weighted volume rather than flow inside the current bar.

    The diagnostics stay GOOD on purpose: an EOD snapshot's premium, volume
    and age are exactly as observed, and grading every key DEGRADED would
    carry no information at all."""
    fv = compute(tape(), intraday=False)

    assert fv.values["options_flow_timing_available"] == FLAG_CLEAR
    assert fv.warmup_complete is True
    assert fv.quality is DataQuality.DEGRADED

    terms = (
        "options_net_premium_imbalance",
        "options_delta_volume_imbalance",
        "options_oi_change_imbalance",
        "options_skew_25d_pressure",
        "options_gamma_exposure_pressure",
    )
    combination = (
        "options_sign_agreement",
        "options_imbalance_consensus",
        "options_flow_uncapped_magnitude",
        "options_flow_magnitude",
        "options_flow_direction",
    )
    for key in terms + combination:
        assert fv.quality_by_key[key] is DataQuality.DEGRADED, key
    for key in (
        "options_net_premium",
        "options_total_volume",
        "options_snapshot_age_seconds",
        "options_available_weight",
        "options_terms_available",
        "options_flow_timing_available",
        "options_eod_capped",
    ):
        assert fv.quality_by_key[key] is DataQuality.GOOD, key

    fallback = note_with(fv, "closed-session aggregate")
    assert "not flow inside the current bar" in fallback

    intraday = compute(tape(), intraday=True)
    assert intraday.values["options_flow_timing_available"] == FLAG_SET
    assert intraday.quality is DataQuality.GOOD


def test_the_eod_cap_binds_and_the_uncapped_value_says_by_how_much():
    """The Phase 5 gate. A cap that is never reached is not a cap, so it is
    shown binding by computing what the uncapped answer would have been --
    and by scoring the SAME chain both ways, so the only difference is the
    cap.

    uncapped = 0.6578073540535698 on both paths. Intraday: the magnitude is
    that number and the flag is clear. End-of-day: the magnitude is exactly
    0.50 (`eod_degraded_cap_fraction`), the flag is set, and the note names
    both numbers."""
    eod = compute(tape(), intraday=False)
    intraday = compute(tape(), intraday=True)

    assert eod.values["options_flow_uncapped_magnitude"] == pytest.approx(
        UNCAPPED, abs=1e-15
    )
    assert (
        eod.values["options_flow_uncapped_magnitude"]
        == intraday.values["options_flow_uncapped_magnitude"]
    )
    assert UNCAPPED > 0.50

    assert eod.values["options_flow_magnitude"] == 0.50
    assert eod.values["options_eod_capped"] == FLAG_SET
    assert intraday.values["options_flow_magnitude"] == pytest.approx(
        UNCAPPED, abs=1e-15
    )
    assert intraday.values["options_eod_capped"] == FLAG_CLEAR

    note = note_with(eod, "capped at")
    assert "0.5000" in note and "0.6578" in note

    # Nothing else moved. The cap, the flag and the timing flag are the only
    # keys that may differ between the two paths -- in particular no value is
    # substituted, interpolated or back-filled for the missing timing.
    moved = {
        key
        for key in eod.values
        if eod.values[key] != intraday.values[key]
    }
    assert moved == {
        "options_flow_magnitude",
        "options_eod_capped",
        "options_flow_timing_available",
    }


def test_the_eod_cap_is_in_force_but_reports_when_it_did_not_bind():
    """`options_eod_capped` reports BINDING, which is the stronger claim than
    "a cap is configured". On an EOD chain whose uncapped magnitude is below
    the cap the flag stays clear, the magnitude is the uncapped value, and
    the note says the cap was in force and did not bind -- so a consumer
    cannot read a clear flag as "this chain was intraday"; that is what
    `options_flow_timing_available` is for."""
    rows = tape(
        omit=(
            "iv_25d_put",
            "iv_25d_call",
            "gamma_exposure_proxy",
            "call_oi_change",
            "put_oi_change",
        )
    )
    fv = compute(rows, intraday=False)
    # 0.25*0.5 + 0.25*0.4, with three terms dropped
    expected = W_NET_PREMIUM * T_NET_PREMIUM + W_DELTA_VOLUME * T_DELTA_VOLUME
    assert expected == 0.225
    assert fv.values["options_flow_uncapped_magnitude"] == expected
    assert fv.values["options_flow_magnitude"] == expected
    assert fv.values["options_eod_capped"] == FLAG_CLEAR
    assert fv.values["options_flow_timing_available"] == FLAG_CLEAR
    assert "did not bind this bar" in note_with(fv, "is in force")


def test_an_uncapped_magnitude_exactly_at_the_cap_is_not_flagged_as_capped():
    """The boundary. A perfectly one-sided premium and delta-weighted volume
    with the other three terms dropped gives 0.25*1 + 0.25*1 = 0.50, which is
    the cap exactly. `min(0.50, 0.50)` changed nothing, so the flag must stay
    clear: a flag on this bar would claim a restriction that did not alter
    the answer."""
    rows = tape(
        background_overrides={
            "call_premium": 400.0,
            "put_premium": 0.0,
            "delta_weighted_call_volume": 100.0,
            "delta_weighted_put_volume": 0.0,
        },
        omit=(
            "call_oi_change",
            "put_oi_change",
            "iv_25d_put",
            "iv_25d_call",
            "gamma_exposure_proxy",
        ),
    )
    fv = compute(rows, intraday=False)
    assert fv.values["options_flow_uncapped_magnitude"] == 0.50
    assert fv.values["options_flow_magnitude"] == 0.50
    assert fv.values["options_eod_capped"] == FLAG_CLEAR
    assert fv.values["options_flow_direction"] == DIRECTION_CALL_SIDE


def test_the_cap_binds_on_most_bars_of_a_real_eod_dataset_and_never_intraday():
    """The empirical half of "the cap must be shown to be BINDING". The exact
    tape proves the cap works; this proves it is reached by ordinary data
    rather than only by a tape built to reach it.

    Five months of synthetic NQ, scored at every bar. On the EOD chain the
    cap must bind on a large share of scored bars and the uncapped
    distribution must straddle it. On the intraday chain of the same
    generator the cap must never be applied, and the magnitude must be free
    to exceed it -- otherwise the intraday path would be silently capped too
    and the degradation would mean nothing.

    Mechanics only: no forward return is read here. See the module
    docstring."""
    comp = computer()

    eod_uncapped, eod_capped_flags, eod_magnitudes = [], [], []
    for _, vector in walk(synthetic(intraday=False), comp):
        eod_uncapped.append(vector.values["options_flow_uncapped_magnitude"])
        eod_magnitudes.append(vector.values["options_flow_magnitude"])
        eod_capped_flags.append(vector.values["options_eod_capped"])
        assert vector.values["options_flow_timing_available"] == FLAG_CLEAR

    assert len(eod_uncapped) > 1000
    uncapped = np.array(eod_uncapped)
    magnitudes = np.array(eod_magnitudes)
    binding = float(np.mean(eod_capped_flags))

    assert binding > 0.25, f"the cap only bound on {binding:.1%} of scored bars"
    assert uncapped.max() > 0.50 and uncapped.min() < 0.50
    assert magnitudes.max() == 0.50
    assert np.all(magnitudes <= 0.50 + 1e-12)
    # The flag is set on exactly the bars where the cap changed the answer.
    assert np.array_equal(
        np.array(eod_capped_flags) == FLAG_SET, uncapped > 0.50
    )

    intraday_magnitudes = []
    for _, vector in walk(synthetic(intraday=True), comp, step=7):
        assert vector.values["options_eod_capped"] == FLAG_CLEAR
        assert vector.values["options_flow_timing_available"] == FLAG_SET
        assert (
            vector.values["options_flow_magnitude"]
            == vector.values["options_flow_uncapped_magnitude"]
        )
        intraday_magnitudes.append(vector.values["options_flow_magnitude"])
    assert max(intraday_magnitudes) > 0.50


def test_require_intraday_prints_disables_the_component_instead_of_capping_it():
    """The stricter research option. True is not "cap harder": the component
    reports UNAVAILABLE, every value is a placeholder, and the note says
    disabled rather than capped -- which makes the cap moot and is the
    honest way to ask "is a capped EOD sub-score worth anything". The same
    config must still score an intraday chain, or the switch would be a
    kill switch rather than a strictness setting."""
    strict = options_config(require_intraday_prints=True)
    disabled = compute(tape(), intraday=False, config=strict)
    assert disabled.warmup_complete is False
    assert disabled.quality is DataQuality.MISSING
    assert all(value == 0.0 for value in disabled.values.values())
    assert "disabled rather than capped" in note_with(disabled, "end-of-day")

    still_scored = compute(tape(), intraday=True, config=strict)
    assert still_scored.warmup_complete is True
    assert still_scored.values["options_flow_magnitude"] == pytest.approx(
        UNCAPPED, abs=1e-15
    )


def test_the_config_refuses_a_cap_that_cannot_bind_and_refuses_synthesis():
    """Both are section 5 quotations rather than preferences, and both are
    the premise of every cap test above: a cap at or above the component's
    full weight leaves the flag describing a restriction that was never
    applied, and "Never synthesized" is not a default that can be turned
    off."""
    with pytest.raises(ValueError, match="not a cap"):
        options_config(eod_degraded_cap_fraction=1.0)
    with pytest.raises(ValueError, match="never synthesized|never_synthesize"):
        options_config(never_synthesize_missing_fields=False)
    assert 0.0 < options_config().eod_degraded_cap_fraction < 1.0


def test_the_snapshot_age_is_a_diagnostic_and_is_never_graded_here():
    """`data/quality.py` is the sole authority on staleness, against each
    feed's own cadence, with the options bound set to four days precisely so
    that an EOD chain is not graded STALE for being what it is. This module
    reports the age so a consumer can see it and re-derives no grade -- a
    second grade would be a second, divergent source of truth.

    Four bars after the newest snapshot, the age is exactly 4 * 300s."""
    fv = compute(tape(), intraday=False, n_bars=25)
    assert fv.values["options_snapshot_age_seconds"] == 4.0 * INTERVAL
    assert fv.quality_by_key["options_snapshot_age_seconds"] is DataQuality.GOOD
    assert DataQuality.STALE not in set(fv.quality_by_key.values())


# ---------------------------------------------------------------------------
# unavailability: the 20 points are reported absent, never estimated
# ---------------------------------------------------------------------------


def test_with_no_options_feed_the_bundle_reports_missing_and_estimates_nothing():
    """Outcome 1 of four. There is no bar-derived proxy for an options chain,
    and a fabricated one would be indistinguishable downstream from a
    measured one -- so the 20 points are UNAVAILABLE and the weight is not
    redistributed anywhere.

    Through `FeatureBundle` on a bars-only dataset, because that is the path
    the signal engine takes: the bundle sees the absent required feed, skips
    the computer, and merges a vector whose every value is a placeholder the
    base class forbids reading (`warmup_complete=False`, MISSING). The
    volatility computer in the same bundle must still be GOOD on its own
    keys, so the test shows the options component failing alone rather than
    the whole bundle failing to run."""
    config = load_config()
    generated = SyntheticMarketGenerator(
        SyntheticConfig(include_options=False), seed=7
    ).generate(
        SYMBOL,
        config.spec(SYMBOL),
        date(2017, 1, 1),
        date(2017, 3, 1),
        INTERVAL,
        calendar=TradingCalendar(),
    )
    bars_only = SymbolData(
        symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: generated.bars}
    )
    assert Feed.OPTIONS_SNAPSHOT not in bars_only.available_feeds()

    options = computer()
    bundle = FeatureBundle([VolatilityFeatures(config.features), options])
    merged = bundle.compute(view_of(bars_only))

    assert merged.warmup_complete is False
    assert merged.quality is DataQuality.MISSING
    for key in options.keys:
        assert merged.values[key] == 0.0, key
        assert merged.quality_by_key[key] is DataQuality.MISSING, key
    assert any(
        note == "options_flow: required feed(s) absent: ['options_snapshot']"
        for note in merged.notes
    )
    # The other component in the bundle is unaffected: nothing about the
    # absent chain was inferred from the bars that ARE present.
    assert merged.quality_by_key["atr_percentile"] is DataQuality.GOOD


def test_a_premium_only_chain_reports_unavailable_rather_than_a_quarter_score():
    """Outcome 3. Only net premium is always present in `OptionsSnapshot`, so
    a premium-only chain carries 0.25 of the weight -- below the 0.50 floor.
    A number built from a quarter of the requested evidence would read as
    "options flow is neutral" while meaning "most of this was never
    measured", and the note has to name the arithmetic and the dropped
    terms so the gap is diagnosable rather than mysterious."""
    rows = tape(
        omit=(
            "call_oi_change",
            "put_oi_change",
            "delta_weighted_call_volume",
            "delta_weighted_put_volume",
            "iv_25d_put",
            "iv_25d_call",
            "gamma_exposure_proxy",
        )
    )
    fv = compute(rows)
    assert fv.warmup_complete is False
    assert fv.quality is DataQuality.MISSING
    assert all(value == 0.0 for value in fv.values.values())
    note = note_with(fv, "below the floor")
    assert "0.2500" in note and "0.5000" in note
    for term in (TERM_DELTA_VOLUME, TERM_OI_CHANGE, TERM_SKEW, TERM_GAMMA):
        assert term in note


def test_premium_plus_oi_change_is_still_below_the_floor():
    """0.25 + 0.20 = 0.45. The floor is crossed between two realistic vendor
    shapes, so a test at only one of them could pass with the comparison
    inverted."""
    rows = tape(
        omit=(
            "delta_weighted_call_volume",
            "delta_weighted_put_volume",
            "iv_25d_put",
            "iv_25d_call",
            "gamma_exposure_proxy",
        )
    )
    fv = compute(rows)
    assert fv.warmup_complete is False
    assert "0.4500" in note_with(fv, "below the floor")


def test_weight_exactly_at_the_availability_floor_is_scored_not_disabled():
    """`min_available_weight_fraction` means "at least this fraction", so the
    boundary belongs on the available side -- the field's own documentation
    says a zero-volume chain (which keeps 0.20 + 0.15 + 0.15 = 0.50) is
    reported DEGRADED rather than unavailable."""
    rows = tape(background_overrides={"call_volume": 0.0, "put_volume": 0.0})
    fv = compute(rows)
    assert fv.warmup_complete is True
    assert fv.values["options_available_weight"] == pytest.approx(0.50, abs=1e-12)
    assert fv.values["options_terms_available"] == 3.0
    assert fv.quality is DataQuality.DEGRADED


def test_a_float_representation_error_cannot_decide_the_availability_floor():
    """Regression for a defect found by this file.

    `min_available_weight_fraction` and the five weights are decimal
    fractions, and summing a SUBSET of them in float64 can land one ULP
    below the nominal total: 0.03 + 0.29 + 0.18 evaluates to
    0.49999999999999994, not 0.5. With a bare `<` comparison a Phase 8 weight
    sweep landing on any of the 46 two-decimal triples with that property
    disabled the component outright, and said so in a note that read
    "available term weight 0.5000 is below the floor 0.5000". `WEIGHT_EPSILON`
    is representation slack, not a threshold."""
    surviving = 0.0
    for value in (0.03, 0.29, 0.18):
        surviving += value
    assert surviving < 0.50 and surviving == pytest.approx(0.50, abs=1e-15)
    assert 0.0 < WEIGHT_EPSILON < 1e-6

    config = options_config(
        weights=weights(
            net_premium=0.47,
            delta_weighted_volume=0.03,
            oi_change=0.03,
            skew_25d=0.29,
            gamma_exposure=0.18,
        )
    )
    rows = tape(background_overrides={"call_volume": 0.0, "put_volume": 0.0})
    fv = compute(rows, config=config)
    assert fv.warmup_complete is True, "a 1-ULP error disabled the component"
    assert fv.values["options_available_weight"] == surviving


def test_a_short_snapshot_window_is_reported_with_both_counts():
    """Outcome 2. A half-filled percentile window produces a number that
    looks like a feature and is not one, so 20 of 21 snapshots is not
    ready -- and the note carries the observed and required counts, because
    "not ready" without them is undiagnosable on a real feed."""
    fv = compute(tape()[:20], n_bars=21)
    assert fv.warmup_complete is False
    assert fv.quality is DataQuality.MISSING
    assert note_with(fv, "warmup incomplete") == (
        "options_flow: warmup incomplete: 20 of 21 options snapshots"
    )


def test_a_non_finite_required_field_is_reported_and_no_term_is_computed():
    """`OptionsSeries` does not screen its six REQUIRED columns for
    finiteness: the non-negativity check uses `np.nanmin`, which IGNORES a
    `nan`, and `snapshot_at` converts non-finite to None only for the
    optional columns. So a `nan` premium reaches the computer, and without
    this screen it would reach `_vector`, which raises -- turning a bad row
    into a crash in the middle of a backtest instead of a reported
    not-ready. Every offending field must be named."""
    rows = tape()
    rows[-1] = dict(rows[-1], call_premium=float("nan"), put_oi=float("inf"))
    fv = compute(rows)
    assert fv.warmup_complete is False
    assert fv.quality is DataQuality.MISSING
    note = note_with(fv, "non-finite required field")
    assert "'call_premium'" in note and "'put_oi'" in note


def test_a_chain_that_traded_nothing_drops_the_two_trade_flow_terms_only():
    """A ratio built from nothing traded is not a measurement of balance, so
    `min_snapshot_volume` drops net premium and delta-weighted volume. The
    positioning and price terms survive on purpose: open interest is a stock
    and implied vol is a price, and neither needs the session to have
    traded. Dropping all five would throw away three real measurements."""
    rows = tape(background_overrides={"call_volume": 0.0, "put_volume": 0.0})
    fv = compute(rows)

    assert fv.values["options_net_premium_imbalance"] == 0.0
    assert fv.values["options_delta_volume_imbalance"] == 0.0
    assert fv.values["options_oi_change_imbalance"] == pytest.approx(
        T_OI_CHANGE, abs=1e-15
    )
    assert fv.values["options_skew_25d_pressure"] == T_SKEW
    assert fv.values["options_gamma_exposure_pressure"] == T_GAMMA
    # 0.20*tanh(0.8) + 0.15*1.0 + 0.15*1.0
    assert fv.values["options_flow_magnitude"] == pytest.approx(
        W_OI_CHANGE * T_OI_CHANGE + W_SKEW + W_GAMMA, abs=1e-15
    )
    assert fv.values["options_total_volume"] == 0.0
    note = note_with(fv, "min_snapshot_volume")
    assert "open interest is a stock" in note


def test_the_volume_gate_is_a_floor_and_a_chain_exactly_at_it_still_scores():
    """`min_snapshot_volume` is a floor -- "volume a snapshot must carry" --
    so a chain carrying exactly it is scored and only a chain below it drops
    the two trade-flow terms. The boundary was not pinned, and `<` against
    `<=` is a one-character edit that silently disables half the component's
    weight on the thinnest chains that are still measurable.

    Tested against the configured value rather than the literal 1.0, so a
    Phase 8 sweep of the field moves the test with it."""
    floor = options_config().min_snapshot_volume
    assert floor > 0.0

    at_floor = compute(
        tape(background_overrides={"call_volume": floor, "put_volume": 0.0})
    )
    assert at_floor.values["options_total_volume"] == floor
    assert at_floor.values["options_terms_available"] == 5.0
    assert at_floor.values["options_available_weight"] == 1.0
    assert at_floor.values["options_net_premium_imbalance"] == T_NET_PREMIUM
    assert not [note for note in at_floor.notes if "min_snapshot_volume" in note]

    below = compute(
        tape(
            background_overrides={
                "call_volume": math.nextafter(floor, 0.0),
                "put_volume": 0.0,
            }
        )
    )
    assert below.values["options_total_volume"] < floor
    assert below.values["options_terms_available"] == 3.0
    assert below.values["options_net_premium_imbalance"] == 0.0
    note_with(below, "min_snapshot_volume")


def test_a_zero_premium_chain_drops_the_term_rather_than_calling_it_balanced():
    """`OptionsSnapshot.premium_imbalance` returns 0.0 when total premium is
    zero, and 0.0 is the reading for a perfectly BALANCED chain -- it cannot
    also be the reading for a chain with no premium at all. The module
    computes the ratio from the raw fields for exactly this reason, and
    `imbalance_ratio` returns None on a non-positive denominator."""
    assert imbalance_ratio(0.0, 0.0) is None
    assert imbalance_ratio(300.0, 100.0) == 0.5
    assert imbalance_ratio(100.0, 100.0) == 0.0
    # A negative side is rejected, not clamped. The inputs matter: (-1, 1)
    # sums to zero and is refused by the DENOMINATOR branch, so it does not
    # exercise the negative-input guard at all -- with that guard deleted
    # entirely it still returns None. (-1, 3) is the input that reaches it:
    # the quotient would be -2.0, outside [-1, 1].
    assert imbalance_ratio(-1.0, 3.0) is None
    assert imbalance_ratio(3.0, -1.0) is None
    assert imbalance_ratio(-1.0, 1.0) is None  # zero denominator, not the guard
    assert imbalance_ratio(float("nan"), 1.0) is None

    fv = compute(tape(background_overrides={"call_premium": 0.0, "put_premium": 0.0}))
    assert fv.values["options_net_premium_imbalance"] == 0.0
    assert fv.quality_by_key["options_net_premium_imbalance"] is DataQuality.DEGRADED
    assert "is a placeholder, not a balanced chain" in note_with(fv, "undefined")


def test_a_percentile_window_with_no_dispersion_drops_its_term():
    """`percentile_rank` counts values AT OR BELOW, so every tie counts and a
    constant window ranks its own repeated value at 1.0 -- "maximally
    elevated" from data that found no variation at all. Both percentile terms
    must drop instead, which costs 0.30 of the weight and leaves the
    component at 0.70.

    The same tie inflation on a PARTIALLY tied window is not fixed and is not
    tested here: it is a property of the shared helper in `features/base.py`,
    documented for quoted spreads in `liquidity.py`."""
    assert relative_rank(np.array([5.0] * 21), 5.0, 11) is None
    assert relative_rank(np.array([1.0, 2.0, 3.0]), 2.0, 11) is None  # too few
    assert relative_rank(np.array([1.0, 2.0, 3.0]), 3.0, 2) == 1.0
    assert relative_rank(np.zeros(0), 1.0, 1) is None
    assert relative_rank(np.array([1.0, 2.0, 3.0]), float("nan"), 2) is None

    fv = compute([dict(BACKGROUND) for _ in range(21)])
    assert fv.values["options_skew_25d_pressure"] == 0.0
    assert fv.values["options_gamma_exposure_pressure"] == 0.0
    assert fv.values["options_available_weight"] == pytest.approx(0.70, abs=1e-12)
    assert fv.values["options_terms_available"] == 3.0
    note_with(fv, "no dispersion at all")


def test_a_percentile_ranked_over_a_partial_window_says_so():
    """Regression for a defect found by this file.

    `MIN_PERCENTILE_SAMPLE_SHARE` drops a percentile term below half a
    window of finite observations, but between that floor and a full window
    it ranks against whatever survived -- and the term's own note claimed it
    was "DROPPED rather than ranked against a window the caller did not ask
    for", which is what the code does only below the floor. A rank over 11 of
    21 and a rank over 21 of 21 were previously identical in the output.

    The note now names both counts. Whether such a term should additionally
    be graded DEGRADED is a contract question for the component's consumers,
    so the grade is deliberately left alone and reported as an open
    question."""
    assert MIN_PERCENTILE_SAMPLE_SHARE == 0.50
    minimum = math.ceil(MIN_PERCENTILE_SAMPLE_SHARE * 21)
    assert minimum == 11

    rows = tape()
    for index in range(10):  # 11 of 21 skew observations survive
        rows[index] = dict(rows[index], iv_25d_put=float("nan"))
    fv = compute(rows)
    assert fv.values["options_skew_25d_pressure"] == T_SKEW
    note = note_with(fv, "percentile is ranked over")
    assert "11 finite observations of the 21 requested" in note

    for index in range(11):  # one fewer, and the term drops entirely
        rows[index] = dict(rows[index], iv_25d_put=float("nan"))
    dropped = compute(rows)
    assert dropped.values["options_available_weight"] == pytest.approx(
        0.85, abs=1e-12
    )
    note_with(dropped, "fewer than 11 finite observations")

    # The helper itself, so the note is not only reachable through one path.
    assert partial_window_note("c", "col", np.arange(21.0), 21) is None
    assert "20 finite observations of the 21 requested" in partial_window_note(
        "c", "col", np.append(np.arange(20.0), np.nan), 21
    )


def test_the_skew_diagnostic_is_a_measurement_whenever_the_chain_supplied_it():
    """Regression for a defect found by this file.

    `options_skew_25d` is a diagnostic read straight off the snapshot, so its
    grade must key on whether the CHAIN supplied both 25-delta legs -- not on
    whether the percentile TERM survived. The term also drops when its window
    has no dispersion, and the grade then reported DEGRADED about a perfectly
    good `iv_25d_put - iv_25d_call` measurement, which is the wrong thing
    said about the wrong key.

    The term's own key is still DEGRADED, because that one does hold a
    placeholder."""
    flat = compute([dict(BACKGROUND) for _ in range(21)])
    assert flat.values["options_skew_25d"] == 0.125  # 0.375 - 0.25, measured
    assert flat.quality_by_key["options_skew_25d"] is DataQuality.GOOD
    assert flat.quality_by_key["options_skew_25d_pressure"] is DataQuality.DEGRADED

    no_legs = compute(tape(omit=("iv_25d_put", "iv_25d_call")))
    assert no_legs.values["options_skew_25d"] == 0.0  # the placeholder
    assert no_legs.quality_by_key["options_skew_25d"] is DataQuality.DEGRADED


def test_the_gamma_ablation_loses_no_weight_and_leaves_one_degraded_key():
    """`use_gamma_exposure_proxy=False` is a Phase 8 ablation, not missing
    data, and the config refuses it while the term's weight is positive --
    otherwise the component would permanently lose that much of its score
    while still claiming full availability.

    The consequence a Phase 8 comparison has to know: completeness is
    measured in WEIGHT, so the five combination keys stay GOOD and no weight
    is lost, but the ablated term's own key holds a 0.0 placeholder and is
    DEGRADED. `FeatureVector.quality` is therefore DEGRADED by construction,
    and an arm gated on the aggregate being GOOD would look degraded for a
    reason that is not about data."""
    with pytest.raises(ValueError, match="use_gamma_exposure_proxy is False"):
        options_config(use_gamma_exposure_proxy=False)

    config = options_config(
        use_gamma_exposure_proxy=False,
        weights=weights(
            net_premium=0.25,
            delta_weighted_volume=0.25,
            oi_change=0.20,
            skew_25d=0.30,
            gamma_exposure=0.0,
        ),
    )
    fv = compute(tape(), config=config)
    assert fv.warmup_complete is True
    assert fv.values["options_available_weight"] == 1.0
    assert fv.values["options_terms_available"] == 4.0
    non_good = {
        key
        for key, quality in fv.quality_by_key.items()
        if quality is not DataQuality.GOOD
    }
    assert non_good == {"options_gamma_exposure_pressure"}
    assert fv.quality is DataQuality.DEGRADED
    # 0.25*0.5 + 0.25*0.4 + 0.20*tanh(0.8) + 0.30*1.0 + 0.0
    assert fv.values["options_flow_magnitude"] == pytest.approx(
        0.25 * T_NET_PREMIUM + 0.25 * T_DELTA_VOLUME + 0.20 * T_OI_CHANGE + 0.30,
        abs=1e-15,
    )
    note_with(fv, "use_gamma_exposure_proxy is off")


def test_a_term_dropped_for_missing_data_degrades_the_combination_keys():
    """The contrast case the gamma-ablation test above needs to mean anything,
    and a hole this file did not cover.

    `_quality_by_key` measures completeness in WEIGHT: a term carrying zero
    weight (the Phase 8 ablation) costs nothing and the combination keys stay
    GOOD, while a term dropped for MISSING DATA costs its whole weight and the
    combination says so even though the component stayed above its
    availability floor. Only the first half was pinned -- and an
    implementation that always reported the combination GOOD on an intraday
    chain passed every test in this file.

    Same chain, same missing column, two reasons, two verdicts."""
    missing = compute(tape(omit=("gamma_exposure_proxy",)))
    assert missing.warmup_complete is True
    assert missing.values["options_flow_timing_available"] == FLAG_SET
    assert missing.values["options_available_weight"] == pytest.approx(
        0.85, abs=1e-12
    )
    assert missing.values["options_terms_available"] == 4.0

    for key in (
        "options_sign_agreement",
        "options_imbalance_consensus",
        "options_flow_uncapped_magnitude",
        "options_flow_magnitude",
        "options_flow_direction",
    ):
        assert missing.quality_by_key[key] is DataQuality.DEGRADED, key
    # The diagnostics are still measurements of the chain itself.
    for key in ("options_net_premium", "options_total_volume", "options_skew_25d"):
        assert missing.quality_by_key[key] is DataQuality.GOOD, key

    # A chain supplying everything, on the same path, is GOOD throughout --
    # so DEGRADED above is the dropped weight talking and not the path.
    whole = compute(tape())
    assert whole.values["options_available_weight"] == 1.0
    assert set(whole.quality_by_key.values()) == {DataQuality.GOOD}


def test_every_unavailable_outcome_is_unreadable_in_the_same_way():
    """`_not_ready`'s zeros are placeholders and must never be read, so every
    route to UNAVAILABLE has to look identical from above:
    `warmup_complete=False`, MISSING on every key, every value 0.0. A route
    that returned MISSING while leaving one real value behind would invite a
    consumer to read it.

    The single note is a real consequence rather than a preference, and is
    pinned so it is at least visible: `_not_ready` (in `features/base.py`)
    builds a fresh vector and discards the per-term notes collected on the
    way, so on the availability-floor route the reason EACH term dropped --
    not supplied, versus a window with no dispersion -- is lost, and only the
    summary note's list of term names survives. That is the base class's
    note shape, not this module's, and it is reported rather than worked
    around here."""
    routes = {
        "short window": (tape()[:20], {"n_bars": 21}),
        "non-finite required field": (
            tape()[:20] + [dict(NOW, call_volume=float("nan"))],
            {},
        ),
        "below the availability floor": (
            tape(
                omit=(
                    "call_oi_change",
                    "put_oi_change",
                    "delta_weighted_call_volume",
                    "delta_weighted_put_volume",
                    "iv_25d_put",
                    "iv_25d_call",
                    "gamma_exposure_proxy",
                )
            ),
            {},
        ),
    }
    for label, (rows, kwargs) in routes.items():
        fv = compute(rows, **kwargs)
        assert fv.warmup_complete is False, label
        assert fv.quality is DataQuality.MISSING, label
        assert set(fv.quality_by_key.values()) == {DataQuality.MISSING}, label
        assert all(value == 0.0 for value in fv.values.values()), label
        assert len(fv.notes) == 1 and fv.notes[0].startswith("options_flow: "), label


# ---------------------------------------------------------------------------
# warmup honesty
# ---------------------------------------------------------------------------


def test_warmup_bars_is_the_snapshot_window_and_the_snapshot_gate_is_separate():
    """`warmup_bars` is declared in bars while the real window is 21
    SNAPSHOTS, so it is a provable lower bound rather than the gate: at most
    one chain snapshot exists per bar, so 21 snapshots cannot exist before 21
    bars do. Both checks have to exist, and the second has to be the binding
    one -- 21 bars with only 20 snapshots must not score."""
    comp = computer()
    assert comp.warmup_bars == options_config().warmup_snapshots == 21

    data = dataset(tape()[:20], n_bars=21)
    view = view_of(data)
    assert view.bar_count() == 21 and view.options_count() == 20
    assert view.warmup_ok(comp.warmup_bars) is True  # the BAR gate is satisfied
    assert comp.compute(view).warmup_complete is False  # the snapshot gate is not


def test_no_key_depends_on_history_older_than_the_declared_window():
    """The regime detector declared 268 bars and read 291, which made the
    label at bar `t` depend on where the caller started loading -- invisible
    to the lookahead audit, because nothing read the future. The direct test
    is tail-equality: prepend nineteen wildly different older snapshots and
    every key must be unchanged.

    `test_the_window_reach_test_can_fail` proves this assertion is live."""
    rows = tape()
    older = [
        dict(
            BACKGROUND,
            call_premium=1.0,
            put_premium=999.0,
            call_oi_change=-9000.0,
            put_oi_change=9000.0,
            delta_weighted_call_volume=1.0,
            delta_weighted_put_volume=999.0,
            iv_25d_put=9.0,
            iv_25d_call=0.01,
            gamma_exposure_proxy=-9e6,
        )
        for _ in range(19)
    ]
    short = compute(rows)
    long = compute(older + rows)
    assert short.warmup_complete and long.warmup_complete
    assert long.values == short.values
    assert long.quality_by_key == short.quality_by_key


def test_the_window_reach_test_can_fail():
    """A test that cannot fail manufactures the appearance of validation. A
    subclass that reads thirty snapshots for its skew window while still
    declaring twenty-one is exactly the regime-detector bug, and the
    tail-equality assertion above has to catch it."""

    class OverReaching(OptionsFlowFeatures):
        name = "over_reaching_options_flow"

        def _skew_term(self, view, snapshot, notes):
            self._window += 9
            try:
                return super()._skew_term(view, snapshot, notes)
            finally:
                self._window -= 9

    comp = OverReaching(options_config())
    rows = tape()
    older = [dict(BACKGROUND, iv_25d_put=9.0) for _ in range(19)]
    short = comp.compute(view_of(dataset(rows)))
    long = comp.compute(view_of(dataset(older + rows)))
    assert short.values != long.values
    moved = {key for key in short.values if short.values[key] != long.values[key]}
    assert "options_skew_25d_pressure" in moved
    assert "options_flow_magnitude" in moved


def test_the_warmup_declaration_is_honest_bar_by_bar_on_a_real_dataset():
    """The declaration has to hold at the boundary, not only on a hand tape.
    On an intraday chain with one snapshot per bar, the first scoreable bar
    must be exactly bar index 20 (the 21st bar), never earlier -- earlier
    would mean a percentile was computed from a partly filled window -- and
    every bar before it must report not-ready."""
    data = synthetic(intraday=True, months=1)
    comp = computer()
    first_scored = None
    for index in range(30):
        cutoff = int(data.primary_bars.ts_ns[index])
        vector = comp.compute(
            MarketView(data, now=from_ns(cutoff), now_ns=cutoff)
        )
        if vector.warmup_complete:
            first_scored = index
            break
        assert vector.quality is DataQuality.MISSING, index
    assert first_scored == 20 == comp.warmup_bars - 1


# ---------------------------------------------------------------------------
# the output contract
# ---------------------------------------------------------------------------


def test_the_emitted_keys_are_exactly_the_declared_keys():
    """`FeatureBundle` checks this too, but only for the keys a computer
    happens to produce on the path it took. Pinning the declared tuple here
    means a key added without a scorer -- or renamed under a consumer -- is a
    failure in this file rather than a `None` propagating into a score."""
    comp = computer()
    assert comp.keys == (
        "options_net_premium_imbalance",
        "options_delta_volume_imbalance",
        "options_oi_change_imbalance",
        "options_skew_25d_pressure",
        "options_gamma_exposure_pressure",
        "options_net_premium",
        "options_skew_25d",
        "options_total_volume",
        "options_snapshot_age_seconds",
        "options_available_weight",
        "options_terms_available",
        "options_flow_timing_available",
        "options_eod_capped",
        "options_sign_agreement",
        "options_imbalance_consensus",
        "options_flow_uncapped_magnitude",
        "options_flow_magnitude",
        "options_flow_direction",
    )
    assert len(set(comp.keys)) == len(comp.keys) == 18
    assert all(key.startswith("options_") for key in comp.keys)

    for rows, kwargs in (
        (tape(), {}),
        (tape(), {"intraday": False}),
        (tape()[:20], {"n_bars": 21}),
    ):
        fv = compute(rows, **kwargs)
        assert set(fv.values) == set(comp.keys)
        assert set(fv.quality_by_key) == set(comp.keys)


def test_the_percentile_windows_name_columns_the_chain_schema_still_has():
    """`view.options_column` returns an EMPTY array for a column the series
    does not have, and the module turns that into a dropped term with a
    "window is short" note. So a column RENAMED in `OptionsSeries.OPTIONAL`
    would not raise anywhere -- it would silently cost the skew and gamma
    terms 0.30 of the weight on every chain, which is under the availability
    floor plus the two trade terms and would look like a vendor problem.

    Pinning the names against the schema makes a rename a failure here."""
    assert set(SKEW_COLUMNS) == {"iv_25d_put", "iv_25d_call"}
    assert GAMMA_COLUMN == "gamma_exposure_proxy"
    for name in SKEW_COLUMNS + (GAMMA_COLUMN,):
        assert name in OptionsSeries.OPTIONAL, name
    # The builders in this file must exercise every column the module reads,
    # or a term could be dropped on every tape here and never be noticed.
    assert set(CHAIN_COLUMNS) <= set(OptionsSeries.REQUIRED) | set(
        OptionsSeries.OPTIONAL
    )
    assert set(OptionsSeries.REQUIRED) <= set(CHAIN_COLUMNS)
    assert set(SKEW_COLUMNS + (GAMMA_COLUMN,)) <= set(CHAIN_COLUMNS)
    assert set(BACKGROUND) == set(CHAIN_COLUMNS)


def test_the_module_docstring_cites_symbols_that_exist():
    """The warmup paragraph's load-bearing citation, pinned.

    `warmup_bars` is a bound in BARS over a window counted in SNAPSHOTS, and
    the only thing standing behind the conversion is the one-snapshot-per-bar
    convention that `data/quality.py` encodes. The docstring named the class
    holding it as `DataQualityGate`, which does not exist -- the class is
    `QualityGrader` -- so a reader checking the one claim that most needs
    checking would have found nothing. A citation to a symbol that is not
    there is worse than no citation, because it reads as verified."""
    from flow_model.data import quality as quality_module

    doc = inspect.getdoc(optionsflow_module) or ""
    assert "DataQualityGate" not in doc
    assert "QualityGrader._presence_coverage" in doc
    assert hasattr(quality_module, "QualityGrader")
    assert hasattr(quality_module.QualityGrader, "_presence_coverage")

    # And the convention really is a cap rather than a rejection, which is
    # what the paragraph now says: two snapshots per bar is accepted, and the
    # computer then reports ready while 21 bars do not yet exist.
    rows = [dict(BACKGROUND) for _ in range(41)] + [dict(NOW)]
    half_bar = to_ns_array(
        TS0 + timedelta(seconds=INTERVAL + INTERVAL * index / 2)
        for index in range(len(rows))
    )
    present = sorted({name for row in rows for name in row})
    dense = OptionsSeries(
        symbol=SYMBOL,
        ts_ns=half_bar,
        columns={
            name: np.array([float(row[name]) for row in rows], dtype=np.float64)
            for name in present
        },
        meta={"is_intraday": True, "source": "two_per_bar"},
    )
    data = SymbolData(
        symbol=SYMBOL,
        primary_interval=INTERVAL,
        bars={INTERVAL: bar_series(25)},
        options=dense,
    )
    comp = computer()
    cutoff = int(data.primary_bars.ts_ns[10])
    view = MarketView(data, now=from_ns(cutoff), now_ns=cutoff)
    assert view.bar_count() == 11 and view.options_count() == 21
    assert comp.compute(view).warmup_complete is True
    assert view.bar_count() < comp.warmup_bars  # the declaration is beaten


def test_required_feeds_is_the_options_chain_and_nothing_else():
    """Declaring `Feed.BARS` would be a requirement this computer does not
    have and would make the component unavailable on a dataset it could have
    scored; declaring fewer than it reads would run it against empty
    arrays."""
    comp = computer()
    assert comp.required_feeds == frozenset({Feed.OPTIONS_SNAPSHOT})
    assert comp.optional_feeds == frozenset()
    assert comp.feeds_available(view_of(dataset(tape()))) is True


def test_perturbing_every_bar_column_moves_nothing():
    """The honest-`required_feeds` test: if a bar column mattered, the
    declaration would be wrong and the component would silently score on a
    bars-only dataset. Open, high, low, close and volume are all replaced
    with wildly different values on the same timestamps."""
    rows = tape()
    options = chain(rows, intraday=True)
    stamped = stamps(len(rows))
    wild_close = np.full(len(rows), 55555.0)
    wild = BarSeries(
        symbol=SYMBOL,
        ts_ns=stamped,
        interval_seconds=INTERVAL,
        columns={
            "open": wild_close - 7.0,
            "high": wild_close + 99.0,
            "low": wild_close - 99.0,
            "close": wild_close,
            "volume": np.full(len(rows), 3.0),
        },
    )
    comp = computer()
    plain = comp.compute(
        view_of(
            SymbolData(
                symbol=SYMBOL,
                primary_interval=INTERVAL,
                bars={INTERVAL: bar_series(len(rows))},
                options=options,
            )
        )
    )
    perturbed = comp.compute(
        view_of(
            SymbolData(
                symbol=SYMBOL,
                primary_interval=INTERVAL,
                bars={INTERVAL: wild},
                options=options,
            )
        )
    )
    assert perturbed.values == plain.values
    assert perturbed.quality_by_key == plain.quality_by_key
    assert perturbed.notes == plain.notes


def test_compute_touches_no_accessor_outside_the_options_feed():
    """Stronger than perturbation, and it names the contract: a view that
    raises on every bar, quote and tick accessor must still produce the full
    vector. If a later edit reaches for `view.closes()` to normalize
    something by the underlying, this fails with the name of the accessor
    instead of producing a feature that quietly needs a feed the component
    does not declare."""

    class OptionsOnlyView:
        allowed = frozenset(
            {
                "symbol",
                "now",
                "has_feed",
                "options_snapshot",
                "options_column",
                "options_count",
                "options_are_intraday",
                "feed_age_seconds",
            }
        )

        def __init__(self, inner):
            self.inner = inner
            self.touched: set[str] = set()

        def __getattr__(self, name):
            if name in {"inner", "touched", "allowed"}:
                raise AttributeError(name)
            if name not in OptionsOnlyView.allowed:
                raise AssertionError(
                    f"options_flow read {name!r}, which is not an options accessor"
                )
            object.__getattribute__(self, "touched").add(name)
            return getattr(object.__getattribute__(self, "inner"), name)

    restricted = OptionsOnlyView(view_of(dataset(tape())))
    fv = computer().compute(restricted)
    assert fv.values["options_flow_magnitude"] == pytest.approx(UNCAPPED, abs=1e-15)
    assert {"options_count", "options_snapshot", "options_column"} <= (
        restricted.touched
    )
    assert restricted.touched <= OptionsOnlyView.allowed


def test_every_key_is_finite_and_within_its_declared_range_over_a_long_run():
    """The term keys, the consensus and the magnitudes are bounded by
    CONSTRUCTION -- a ratio of same-signed quantities, a percentile rank or a
    `tanh`, never a clamp over an unbounded quantity -- so a value outside
    range means a transform was replaced, not that a market was unusual.

    The four natural-unit diagnostics are unbounded by design, so only their
    finiteness and sign are checkable here."""
    bounded_signed = (
        "options_net_premium_imbalance",
        "options_delta_volume_imbalance",
        "options_oi_change_imbalance",
        "options_skew_25d_pressure",
        "options_imbalance_consensus",
    )
    bounded_unit = (
        "options_gamma_exposure_pressure",
        "options_sign_agreement",
        "options_available_weight",
        "options_flow_uncapped_magnitude",
        "options_flow_magnitude",
    )
    comp = computer()
    checked = 0
    for intraday in (True, False):
        for _, fv in walk(synthetic(intraday=intraday, months=2), comp, step=11):
            checked += 1
            for key, value in fv.values.items():
                assert math.isfinite(value), key
            for key in bounded_signed:
                assert -1.0 <= fv.values[key] <= 1.0, key
            for key in bounded_unit:
                assert 0.0 <= fv.values[key] <= 1.0, key
            assert fv.values["options_flow_direction"] in (
                DIRECTION_PUT_SIDE,
                DIRECTION_NONE,
                DIRECTION_CALL_SIDE,
            )
            assert fv.values["options_flow_timing_available"] in (FLAG_CLEAR, FLAG_SET)
            assert fv.values["options_eod_capped"] in (FLAG_CLEAR, FLAG_SET)
            assert 0.0 <= fv.values["options_terms_available"] <= 5.0
            assert (
                fv.values["options_flow_magnitude"]
                <= fv.values["options_flow_uncapped_magnitude"] + 1e-12
            )
            assert (
                fv.values["options_flow_uncapped_magnitude"]
                <= fv.values["options_available_weight"] + 1e-12
            )
            assert fv.values["options_total_volume"] >= 0.0
            assert fv.values["options_snapshot_age_seconds"] >= 0.0
    assert checked > 400


def test_the_quality_of_a_scored_vector_is_never_missing_or_stale():
    """MISSING is reserved for `_not_ready`, where every value is a
    placeholder. A single dropped term reported as MISSING would make
    `FeatureVector.quality` MISSING for the whole merged bundle and block
    every other component -- a chain missing its gamma column would disable
    market structure. STALE is `data/quality.py`'s verdict, never this
    module's."""
    scenarios = (
        tape(),
        tape(omit=("gamma_exposure_proxy",)),
        tape(omit=("call_oi_change", "put_oi_change")),
        [dict(BACKGROUND) for _ in range(21)],
        tape(background_overrides={"call_volume": 0.0, "put_volume": 0.0}),
    )
    for rows in scenarios:
        for intraday in (True, False):
            fv = compute(rows, intraday=intraday)
            assert fv.warmup_complete is True
            assert set(fv.quality_by_key.values()) <= {
                DataQuality.GOOD,
                DataQuality.DEGRADED,
            }


def test_no_key_name_or_runtime_note_claims_intent_or_an_actor():
    """Section 5's rejected interpretations are binding: options flow is "a
    measurable order-imbalance and positioning signal, not... evidence of
    institutional intent", and a large premium print has an ambiguous sign.

    The scan is over the KEY NAMES and the notes actually EMITTED, not over
    the module's prose, which quotes the rejected reading in order to reject
    it. Notes reach the signal journal and the trade record's reason field,
    so a note claiming an actor would end up in the research output."""
    banned = (
        "institution",
        "smart money",
        "smart-money",
        "sweep",
        "conviction",
        "bullish",
        "bearish",
        "whale",
        "unusual activity",
        "intent",
        "retail",
        "dumb money",
    )
    comp = computer()
    for key in comp.keys:
        for word in banned:
            assert word not in key.lower().replace("_", " "), key

    emitted: list[str] = []
    scenarios = [
        (tape(), {}),
        (tape(), {"intraday": False}),
        (tape()[:20], {"n_bars": 21}),
        ([dict(BACKGROUND) for _ in range(21)], {}),
        (tape(omit=("gamma_exposure_proxy",)), {}),
        (tape(omit=("call_oi_change", "put_oi_change")), {}),
        (
            tape(
                omit=(
                    "delta_weighted_call_volume",
                    "delta_weighted_put_volume",
                    "iv_25d_put",
                    "iv_25d_call",
                    "gamma_exposure_proxy",
                )
            ),
            {},
        ),
        (tape(background_overrides={"call_volume": 0.0, "put_volume": 0.0}), {}),
        (tape(background_overrides={"call_premium": 0.0, "put_premium": 0.0}), {}),
        (
            tape(),
            {"intraday": False, "config": options_config(require_intraday_prints=True)},
        ),
    ]
    for rows, kwargs in scenarios:
        emitted.extend(compute(rows, **kwargs).notes)
    # Ten scenarios produce eleven notes, all of them distinct. The counts are
    # asserted so that a scan over an empty list cannot pass, and so that a
    # note added later is read by this test rather than skipped by it.
    assert len(emitted) == 11 and len(set(emitted)) == 11
    for note in emitted:
        lowered = note.lower()
        for word in banned:
            assert word not in lowered, f"{word!r} in note: {note}"
        assert note.startswith("options_flow: ")


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_the_same_view_computed_twice_and_by_two_computers_agrees():
    """ARCHITECTURE.md section 0.5: same config hash plus same data hash
    gives bit-identical results. No clock, no generator, no state -- so both
    directions must hold, and the notes and per-key qualities must match too,
    because a note that varied would change a `TradeRecord`'s reason field
    and break the determinism test one layer up."""
    view = view_of(dataset(tape(), intraday=False))
    first, second = computer().compute(view), computer().compute(view)
    same_instance = computer()
    third, fourth = same_instance.compute(view), same_instance.compute(view)
    for other in (second, third, fourth):
        assert other.values == first.values
        assert other.quality_by_key == first.quality_by_key
        assert other.notes == first.notes
        assert other.warmup_complete == first.warmup_complete


def test_the_term_sum_is_evaluated_in_one_fixed_order():
    """Floating-point addition is not associative, so a weighted sum
    evaluated in a set-iteration order would differ in the last bits between
    runs. `TERM_ORDER` fixes it, and the magnitude must therefore equal the
    sum written in that exact order."""
    fv = compute(tape())
    in_order = 0.0
    for weight, term in (
        (W_NET_PREMIUM, T_NET_PREMIUM),
        (W_DELTA_VOLUME, T_DELTA_VOLUME),
        (W_OI_CHANGE, T_OI_CHANGE),
        (W_SKEW, T_SKEW),
        (W_GAMMA, T_GAMMA),
    ):
        in_order += weight * term
    assert fv.values["options_flow_uncapped_magnitude"] == in_order


# ---------------------------------------------------------------------------
# lookahead
# ---------------------------------------------------------------------------


def test_the_lookahead_audit_passes_on_both_feed_shapes():
    """Verified here rather than taken on trust, and with `factory=` rather
    than an instance: the audit then rebuilds the computer against each
    altered dataset, so a full-sample statistic captured in `__init__` moves
    with it and is reported. Both chain shapes are audited because the EOD
    path takes a different branch through `compute` -- the cap -- and an
    audit of only the intraday path would never execute it."""
    config = load_config().options_flow
    for intraday in (True, False):
        data = synthetic(intraday=intraday, months=3)
        audited = _sample_indices(
            len(data.primary_bars), computer().warmup_bars, 40, 20240101
        )
        result = audit_computer(
            data=data,
            factory=lambda _: OptionsFlowFeatures(config),
            sample=40,
        )
        assert result.bars_checked > 0, "nothing audited is not a pass"
        assert len(result.keys_checked) == 18
        assert_no_lookahead([result])

        # `bars_checked` and `keys_checked` are satisfied by an audit of
        # NOT-READY bars alone: `_not_ready` emits all eighteen keys as
        # zeros, and zeros are trivially truncation- and mutation-invariant.
        # Verified: an audit over 30 bars carrying 5 snapshots reports 13
        # bars checked, 18 keys and no findings while scoring nothing. The
        # EOD chain makes that reachable here -- its 21-snapshot gate is not
        # satisfied until roughly 21 sessions in, which is most of a
        # three-month dataset -- so what the audit actually covered is
        # asserted rather than assumed.
        comp = computer()
        scored = [
            comp.compute(
                MarketView(
                    data,
                    now=from_ns(int(data.primary_bars.ts_ns[index])),
                    now_ns=int(data.primary_bars.ts_ns[index]),
                )
            )
            for index in audited
        ]
        ready = [vector for vector in scored if vector.warmup_complete]
        assert len(ready) > 10, (
            f"intraday={intraday}: only {len(ready)} of {len(audited)} audited "
            "bars produced a scored vector; the rest compare zeros to zeros"
        )
        assert any(
            vector.values["options_flow_magnitude"] > 0.0 for vector in ready
        )
        if not intraday:
            # ...and the branch this second audit exists for -- the cap --
            # was executed on an audited bar, not merely configured.
            assert any(
                vector.values["options_eod_capped"] == FLAG_SET for vector in ready
            )


def test_the_audit_reports_a_constant_captured_at_construction():
    """The gate is not vacuous. This is the one leak `MarketView` cannot
    close, because the capture happens before any view exists, and the only
    reason the audit sees it is `factory=`."""
    data = synthetic(intraday=True, months=2)

    class CaptureCheater(OptionsFlowFeatures):
        name = "capture_cheater"

        def __init__(self, config, captured: float) -> None:
            super().__init__(config)
            self.captured = float(captured)

        def compute(self, view):
            vector = super().compute(view)
            return vector.replace(
                values={
                    **vector.values,
                    "options_flow_magnitude": abs(self.captured) % 1.0,
                }
            )

    config = load_config().options_flow
    result = audit_computer(
        data=data,
        factory=lambda d: CaptureCheater(
            config, float(np.mean(d.options.col("call_premium")))
        ),
        sample=40,
    )
    assert len(result.findings) > 0
    assert {f.kind for f in result.findings} == {"truncation", "mutation"}
    assert all(f.key == "options_flow_magnitude" for f in result.findings)
    with pytest.raises(AssertionError):
        assert_no_lookahead([result])


def test_the_audit_reports_unseeded_jitter():
    """Check 4. All randomness in this system flows through named, seeded
    generators; an unseeded one makes two computations on the identical view
    disagree, which is also what a clock read would look like."""
    data = synthetic(intraday=True, months=2)
    config = load_config().options_flow

    class JitterCheater(OptionsFlowFeatures):
        name = "jitter_cheater"

        def compute(self, view):
            import random

            vector = super().compute(view)
            return vector.replace(
                values={
                    **vector.values,
                    "options_flow_magnitude": min(
                        1.0,
                        vector.values["options_flow_magnitude"]
                        + random.random() * 1e-3,
                    ),
                }
            )

    result = audit_computer(
        data=data, factory=lambda _: JitterCheater(config), sample=40
    )
    assert "determinism" in {f.kind for f in result.findings}
    assert len(result.findings) > 0


def test_the_audit_reports_a_warmup_declaration_the_computer_does_not_honour():
    """Check 3 is live for this computer in ONE direction: a declaration the
    computer beats. A subclass declaring 42 bars while reporting ready at 21
    snapshots is caught, which is what makes the next test's result
    meaningful rather than an artefact of a dead check."""
    data = synthetic(intraday=True, months=2)
    config = load_config().options_flow

    class OverDeclaring(OptionsFlowFeatures):
        name = "over_declaring_cheater"

        @property
        def warmup_bars(self) -> int:
            return 2 * super().warmup_bars

    result = audit_computer(
        data=data, factory=lambda _: OverDeclaring(config), sample=40
    )
    assert {f.kind for f in result.findings} == {"warmup"}
    assert len(result.findings) > 0


def test_the_audit_cannot_catch_an_under_declared_warmup_bars_here():
    """An honest negative result, recorded because a reader would otherwise
    assume the audit covers this.

    Check 3 fires when a computer reports ready BEFORE its declared warmup. A
    subclass declaring `warmup_bars = 2` reports ready no earlier than its
    21-snapshot gate allows, so the check never fires and the audit returns
    ZERO findings on a declaration that is off by nineteen bars. The
    regime-detector bug (268 declared, 291 read) was invisible to this audit
    for the same reason: nothing read the future.

    What covers it is
    `test_no_key_depends_on_history_older_than_the_declared_window`, whose
    liveness `test_the_window_reach_test_can_fail` establishes."""
    data = synthetic(intraday=True, months=2)
    config = load_config().options_flow

    class UnderDeclaring(OptionsFlowFeatures):
        name = "under_declaring_cheater"

        @property
        def warmup_bars(self) -> int:
            return 2

    result = audit_computer(
        data=data, factory=lambda _: UnderDeclaring(config), sample=40
    )
    assert result.findings == ()
    assert UnderDeclaring(config).warmup_bars < computer().warmup_bars


# ---------------------------------------------------------------------------
# the constructor ban
# ---------------------------------------------------------------------------


def test_the_constructor_takes_configuration_only():
    """Local echo of the package-wide ban in `test_feature_contracts.py`. A
    computer handed a series could capture a full-sample statistic at
    construction, which is the one leak `validation.lookahead` cannot see
    through when it is given an instance rather than a factory -- and
    `test_the_audit_reports_a_constant_captured_at_construction` above shows
    exactly what that looks like.

    `optionsflow.py` uses postponed annotations, so
    `__init__.__annotations__` holds STRINGS: they are resolved against the
    module globals with `get_type_hints` before comparing, because comparing
    a `str` to a class passes silently and has already done so twice in this
    project."""
    parameters = inspect.signature(OptionsFlowFeatures.__init__).parameters
    assert list(parameters) == ["self", "config"]

    hints = get_type_hints(OptionsFlowFeatures.__init__)
    assert hints["config"] is OptionsFlowConfig
    assert not isinstance(hints["config"], str), "the config hint was not resolved"

    rendered = str(inspect.signature(OptionsFlowFeatures.__init__))
    for word in (
        "SymbolData",
        "OptionsSeries",
        "BarSeries",
        "ColumnSeries",
        "DataStore",
        "MarketView",
        "OptionsSnapshot",
        "FeatureVector",
        "SyntheticDataset",
        "ndarray",
        "DataFrame",
        "Series",
    ):
        assert word not in rendered, f"{word} must not appear in the constructor"

    for name in ("data", "dataset", "store", "series", "view", "bars", "chain"):
        assert name not in parameters, name


def test_every_one_of_the_five_weights_reaches_the_magnitude():
    """A term identifier that never reached the weight table would drop that
    term's weight from the magnitude while `options_terms_available` still
    counted it -- the dead-term failure again, but from the config side
    rather than the term side.

    So each of the five weights is moved in turn and the magnitude must land
    on the sum recomputed with the new weights. On the exact tape the terms
    are fixed, so each expected number is `sum(w_i * t_i)` written out with
    only one pair of weights changed."""
    terms = {
        TERM_NET_PREMIUM: T_NET_PREMIUM,
        TERM_DELTA_VOLUME: T_DELTA_VOLUME,
        TERM_OI_CHANGE: T_OI_CHANGE,
        TERM_SKEW: T_SKEW,
        TERM_GAMMA: T_GAMMA,
    }
    assert set(terms) == set(TERM_ORDER)
    base = {
        TERM_NET_PREMIUM: W_NET_PREMIUM,
        TERM_DELTA_VOLUME: W_DELTA_VOLUME,
        TERM_OI_CHANGE: W_OI_CHANGE,
        TERM_SKEW: W_SKEW,
        TERM_GAMMA: W_GAMMA,
    }
    field = {
        TERM_NET_PREMIUM: "net_premium",
        TERM_DELTA_VOLUME: "delta_weighted_volume",
        TERM_OI_CHANGE: "oi_change",
        TERM_SKEW: "skew_25d",
        TERM_GAMMA: "gamma_exposure",
    }
    assert computer().name == "options_flow"

    for moved in TERM_ORDER:
        donor = TERM_DELTA_VOLUME if moved != TERM_DELTA_VOLUME else TERM_SKEW
        shifted = dict(base)
        shifted[moved] += 0.05
        shifted[donor] -= 0.05
        config = options_config(
            weights=weights(**{field[name]: shifted[name] for name in TERM_ORDER})
        )
        fv = compute(tape(), config=config)
        expected = 0.0
        for name in TERM_ORDER:
            expected += shifted[name] * terms[name]
        assert fv.values["options_flow_magnitude"] == pytest.approx(
            expected, abs=1e-12
        ), moved
        # The move changed the answer, so the assertion above is not passing
        # because every weight vector happens to give the same number.
        assert expected != pytest.approx(UNCAPPED, abs=1e-6), moved
