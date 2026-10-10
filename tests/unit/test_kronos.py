"""Tests for the Kronos integration, organized around one question.

Kronos is the first component in this project whose correctness is not
decidable from the code. Every other feature is an explicit function of
visible bars, so a test can hand it a tape and check the arithmetic. This one
is a checkpoint someone else trained on data nobody here can enumerate, and
the thing that could be wrong with it -- that it has already seen the
outcomes it is forecasting -- leaves no trace in the data-access pattern.

So the tests are organized around the quarantine rather than around the
forecast:

**The audit is blind, and that is demonstrated, not asserted.**
`test_the_lookahead_audit_passes_a_deliberate_oracle` wires in an adapter
that literally reads the future and shows `validation/lookahead.py` reporting
no findings. If that test ever fails, the audit has become able to catch
weight-borne leakage and this module's defensiveness can be revisited. While
it passes, a green audit on `KronosFeatures` means nothing about
contamination, and the test exists so nobody concludes otherwise.

**The guard is the real defence**, so its truth table is tested exhaustively
rather than by example: unknown cutoff against a historical bar, unknown
cutoff against the live frontier, a declared cutoff on both sides, and each
of the three policies.

**Nothing is substituted when the dependency is absent.** torch is not
installed in this environment, which makes the unavailable path the DEFAULT
path here and therefore the best-tested one. That is a coverage gap in the
opposite direction, stated plainly at the bottom.

The feature arithmetic is tested through `paths_to_features`, which is pure
and takes the sampled closes directly -- so the statistics are checked by
hand without needing a checkpoint at all.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import get_type_hints

import numpy as np
import pytest

from flow_model.config.schema import KronosConfig
from flow_model.core.enums import DataQuality, Feed, KronosContamination, KronosMode
from flow_model.features.kronos import (
    DIRECTION_DEAD_BAND,
    EXPECTED_MOVE_SCALE_ATR,
    KRONOS_KEYS,
    ContaminationGuard,
    ForecastPaths,
    KronosAdapter,
    KronosFeatures,
    KronosUnavailable,
    paths_to_features,
)

UTC = timezone.utc

#: Inside the project's research window, so it is exactly the case that must
#: be refused: a bar the checkpoint has almost certainly trained on.
HISTORICAL = datetime(2017, 6, 1, 14, 35, tzinfo=UTC)
#: After the sealed window, standing in for "now" in a live run.
LIVE = datetime(2026, 6, 1, 14, 35, tzinfo=UTC)


# ---------------------------------------------------------------------------
# local builders. Local rather than shared, so a failure localizes here.
# ---------------------------------------------------------------------------

SYMBOL, INTERVAL = "NQ", 300


@pytest.fixture(scope="module")
def synthetic_symbol_data():
    """A short synthetic NQ dataset whose last bar sits inside the research
    window, so the default guard verdict on it is the one that matters."""
    from flow_model.config.loader import load_config
    from flow_model.data import SyntheticConfig, SyntheticMarketGenerator, TradingCalendar
    from flow_model.data.store import SymbolData

    config = load_config()
    dataset = SyntheticMarketGenerator(SyntheticConfig(), seed=3).generate(
        SYMBOL, config.spec(SYMBOL), date(2017, 1, 1), date(2017, 3, 1),
        INTERVAL, calendar=TradingCalendar(),
    )
    return SymbolData(
        symbol=SYMBOL, primary_interval=INTERVAL, bars={INTERVAL: dataset.bars}
    )


def view_at(data, index: int = -1):
    """A view whose cutoff is bar `index`'s close, so that bar is visible."""
    from flow_model.data.market_view import MarketView
    from flow_model.data.series import from_ns

    series = data.bars[INTERVAL]
    position = index if index >= 0 else len(series) + index
    ts = int(series.ts_ns[position])
    return MarketView(data, now=from_ns(ts), now_ns=ts)


def cfg(**overrides) -> KronosConfig:
    fields = dict(enabled=True, max_context=16, pred_len=4, sample_count=8)
    fields.update(overrides)
    return KronosConfig(**fields)


# ---------------------------------------------------------------------------
# the blindness, demonstrated
# ---------------------------------------------------------------------------


class _OracleAdapter:
    """An adapter that cheats as hard as it is possible to cheat.

    It ignores the input frame entirely and returns a forecast built from a
    future it was handed at construction. This is a perfect stand-in for a
    checkpoint that memorised the tape: the forecast is correct because the
    answer was known, not because anything was inferred.
    """

    def __init__(self, future_close: float) -> None:
        self.future_close = future_close

    def forecast(self, frame, x_timestamp, y_timestamp, anchor_close):
        return ForecastPaths(
            terminal_closes=np.full(8, self.future_close, dtype=np.float64),
            anchor_close=float(anchor_close),
            horizon_bars=4,
        )


def test_the_lookahead_audit_passes_a_deliberate_oracle(synthetic_symbol_data):
    """The most important test in this file.

    `validation/lookahead.py` checks truncation invariance, future-mutation
    invariance, warmup honesty and determinism. An oracle passes all four:
    truncating the dataset does not change what it knows, mutating future bars
    does not change what it knows, it declares its context honestly, and it is
    deterministic. The leak is in the oracle, not in the access pattern.

    So this test asserts the audit reports NO findings against a computer that
    is cheating outright. It is the reason `ContaminationGuard` exists and the
    reason a green audit on this module proves nothing about contamination.
    """
    from flow_model.validation.lookahead import assert_no_lookahead, audit_computer

    def factory(data):
        return KronosFeatures(
            cfg(pretrain_cutoff=date(2010, 1, 1), contamination_policy=KronosContamination.ALLOW),
            adapter=_OracleAdapter(future_close=999_999.0),
        )

    result = audit_computer(data=synthetic_symbol_data, factory=factory, sample=12)
    assert_no_lookahead([result])          # the oracle is NOT caught
    assert result.bars_checked > 0
    assert set(result.keys_checked) == set(KRONOS_KEYS)


def test_the_guard_catches_what_the_audit_cannot(synthetic_symbol_data):
    """The complement of the test above, on the same oracle: with the default
    policy and no declared cutoff, the bar is refused before the oracle is
    ever consulted."""
    computer = KronosFeatures(cfg(), adapter=_OracleAdapter(future_close=999_999.0))
    view = view_at(synthetic_symbol_data, -1)
    vector = computer.compute(view)
    assert vector.quality is DataQuality.MISSING
    assert vector.values["kronos_contaminated"] == 1.0
    assert vector.values["kronos_up_probability"] == 0.0     # never computed
    assert any("REFUSED" in note for note in vector.notes)


# ---------------------------------------------------------------------------
# the guard's truth table
# ---------------------------------------------------------------------------


def test_unknown_cutoff_contaminates_every_bar_of_a_replay():
    """Kronos's published state. A checkpoint that will not say what it read
    must be assumed to have read everything that already happened."""
    guard = ContaminationGuard(pretrain_cutoff=None, mode=KronosMode.RESEARCH)
    verdict = guard.verdict(HISTORICAL)
    assert verdict.contaminated is True
    assert verdict.refused is True
    assert verdict.quality is DataQuality.MISSING
    assert "publishes no training-data cutoff" in verdict.reason


def test_research_mode_refuses_regardless_of_how_recent_the_bar_is():
    """The bug this pins, found by test_the_guard_catches_what_the_audit_cannot.

    The guard's first version compared the bar against the VIEW's cutoff and
    called a bar at the frontier clean. In a backtest the view's cutoff IS the
    current bar, so that test was true on every bar of a replay: the guard
    would have waved through every contaminated historical bar while appearing
    to work. Mode is now declared by the caller, and in RESEARCH mode the bar's
    own recency is irrelevant -- even a 2026 bar is part of a replay.
    """
    guard = ContaminationGuard(pretrain_cutoff=None, mode=KronosMode.RESEARCH)
    for ts in (HISTORICAL, LIVE, datetime(2099, 1, 1, tzinfo=UTC)):
        assert guard.verdict(ts).refused is True, ts


def test_live_mode_permits_an_unknown_cutoff():
    """The clean line, and the whole reason this integration is usable at all.
    A bar that has not happened cannot be in anyone's training set, whatever
    the cutoff is, so forward signals are sound even with provenance
    unknown."""
    guard = ContaminationGuard(pretrain_cutoff=None, mode=KronosMode.LIVE)
    verdict = guard.verdict(LIVE)
    assert verdict.contaminated is False
    assert verdict.permitted is True
    assert verdict.quality is DataQuality.GOOD
    assert "had not happened when any checkpoint" in verdict.reason


def test_research_is_the_default_mode():
    """The conservative reading, and the dangerous case. A default of LIVE
    would make a backtest silently clean."""
    assert KronosConfig().mode is KronosMode.RESEARCH
    assert ContaminationGuard(pretrain_cutoff=None).verdict(HISTORICAL).refused is True


@pytest.mark.parametrize(
    "bar_day,expect_contaminated",
    [
        (date(2019, 12, 31), True),    # before the cutoff
        (date(2020, 1, 1), True),      # ON the cutoff -- inclusive, the data is in
        (date(2020, 1, 2), False),     # after it
    ],
)
def test_a_declared_cutoff_is_inclusive_on_its_own_date(bar_day, expect_contaminated):
    """Inclusive because a cutoff names the last date COVERED. Treating it as
    exclusive would admit one day of training data as clean, and a one-day
    error at the boundary is the kind that survives review."""
    guard = ContaminationGuard(pretrain_cutoff=date(2020, 1, 1))
    ts = datetime(bar_day.year, bar_day.month, bar_day.day, 15, 0, tzinfo=UTC)
    assert guard.verdict(ts).contaminated is expect_contaminated


def test_flag_computes_and_marks_degraded():
    """For deliberately measuring the contaminated signal. Worth having: a
    model that has seen the answers gives an UPPER BOUND no clean model can
    beat, which is a reference point rather than a result."""
    guard = ContaminationGuard(None, policy=KronosContamination.FLAG)
    verdict = guard.verdict(HISTORICAL)
    assert verdict.contaminated is True
    assert verdict.permitted is True
    assert verdict.quality is DataQuality.DEGRADED
    assert "upper bound, not a" in verdict.reason


def test_allow_requires_a_declared_cutoff():
    """ALLOW plus an unknown cutoff would report a forecast from a checkpoint
    of unknown provenance as clean, which is the one combination that can
    silently corrupt a reported backtest."""
    with pytest.raises(ValueError, match="unknown provenance"):
        cfg(contamination_policy=KronosContamination.ALLOW, pretrain_cutoff=None)
    # with a cutoff it is accepted
    ok = cfg(contamination_policy=KronosContamination.ALLOW, pretrain_cutoff=date(2020, 1, 1))
    assert ok.contamination_policy is KronosContamination.ALLOW


def test_the_guard_takes_the_clock_as_an_argument():
    """A guard that read the wall clock would give different verdicts on a
    replay, and a backtest is a replay. Same rule as risk/limits.py."""
    import ast
    import inspect

    from flow_model.features import kronos as module

    tree = ast.parse(inspect.getsource(module.ContaminationGuard))
    bad = [
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"now", "utcnow", "today", "time"}
    ]
    assert not bad, f"ContaminationGuard reads the clock: {bad}"


def test_a_mini_checkpoint_demands_the_2k_tokenizer():
    """A mismatched tokenizer emits tokens the model never saw during
    training, which produces confident nonsense rather than an error."""
    with pytest.raises(ValueError, match="2k tokenizer"):
        cfg(model_repo="NeoQuasar/Kronos-mini", tokenizer_repo="NeoQuasar/Kronos-Tokenizer-base")
    cfg(model_repo="NeoQuasar/Kronos-mini", tokenizer_repo="NeoQuasar/Kronos-Tokenizer-2k")


# ---------------------------------------------------------------------------
# the feature arithmetic, by hand
# ---------------------------------------------------------------------------


def paths(closes, anchor=100.0, horizon=4) -> ForecastPaths:
    return ForecastPaths(
        terminal_closes=np.asarray(closes, dtype=np.float64),
        anchor_close=anchor, horizon_bars=horizon,
    )


def test_up_probability_and_agreement_are_hand_computable():
    """6 of 8 paths above a 100.0 anchor: p = 0.75, and agreement is
    |0.75 - 0.5| * 2 = 0.5. Agreement carries only the strength of consensus,
    with the direction held separately, because section 7 requires magnitude
    and direction to be separable."""
    out = paths_to_features(paths([101, 102, 103, 104, 105, 106, 99, 98]), atr=2.0)
    assert out["kronos_up_probability"] == 0.75
    assert out["kronos_agreement"] == 0.5
    assert out["kronos_direction"] == 1.0


def test_an_even_split_reports_no_direction():
    """4 up, 4 down: p = 0.5, inside the dead band, so direction is 0 and
    agreement is 0. A 32-path sample has a standard error near 0.09 on a
    probability this close to even money, so calling a side here would be
    reporting sampling noise."""
    out = paths_to_features(paths([101, 102, 103, 104, 99, 98, 97, 96]), atr=2.0)
    assert out["kronos_up_probability"] == 0.5
    assert out["kronos_direction"] == 0.0
    assert out["kronos_agreement"] == 0.0


def test_the_dead_band_boundary_is_strict_on_both_sides():
    """p exactly at 0.5 + the band is NOT a direction; just past it is."""
    n = 1000
    at_edge = int(n * (0.5 + DIRECTION_DEAD_BAND))
    inside = paths_to_features(
        paths([101.0] * at_edge + [99.0] * (n - at_edge)), atr=1.0
    )
    assert inside["kronos_up_probability"] == pytest.approx(0.5 + DIRECTION_DEAD_BAND)
    assert inside["kronos_direction"] == 0.0
    beyond = paths_to_features(
        paths([101.0] * (at_edge + 1) + [99.0] * (n - at_edge - 1)), atr=1.0
    )
    assert beyond["kronos_direction"] == 1.0


def test_the_expected_move_is_in_atr_units_and_signed():
    """Mean of the paths is 102.0 against a 100.0 anchor, so the move is
    +2.0; at ATR 2.0 that is exactly 1.0 ATR, which the squash maps to
    tanh(1.0) at the configured scale of one ATR."""
    out = paths_to_features(paths([101, 102, 103, 102]), atr=2.0)
    assert out["kronos_expected_move_atr"] == pytest.approx(1.0)
    assert out["kronos_expected_move_score"] == pytest.approx(
        math.tanh(1.0 / EXPECTED_MOVE_SCALE_ATR)
    )


def test_a_down_forecast_does_not_read_as_a_weak_up_forecast():
    """The reason the score is a SIGNED squash. `base.squash` takes the
    magnitude by design, so without reapplying the sign a forecast fall would
    be indistinguishable from a forecast rise of the same size."""
    up = paths_to_features(paths([102, 102, 102, 102], anchor=100.0), atr=2.0)
    down = paths_to_features(paths([98, 98, 98, 98], anchor=100.0), atr=2.0)
    assert up["kronos_expected_move_score"] == pytest.approx(-down["kronos_expected_move_score"])
    assert down["kronos_expected_move_score"] < 0.0
    assert down["kronos_direction"] == -1.0


def test_dispersion_is_the_sample_spread_in_atr_units():
    closes = [98.0, 100.0, 102.0, 104.0]
    expected = float(np.std(np.asarray(closes), ddof=1)) / 2.0
    out = paths_to_features(paths(closes), atr=2.0)
    assert out["kronos_path_dispersion_atr"] == pytest.approx(expected)


def test_without_an_atr_the_denominated_keys_are_zero_not_guessed():
    """`signals/gates.py` and `signals/engine.py` both decline a bar whose ATR
    was never measured -- a HIGH bug was fixed in exactly that place -- so
    this module reports 0.0 and expects the caller to decline, rather than
    inventing a scale."""
    for bad_atr in (None, 0.0, -1.0, float("nan"), float("inf")):
        out = paths_to_features(paths([101, 102, 103, 104]), atr=bad_atr)
        assert out["kronos_expected_move_atr"] == 0.0
        assert out["kronos_path_dispersion_atr"] == 0.0
        assert out["kronos_expected_move_score"] == 0.0
        # the unscaled statistics still work
        assert out["kronos_up_probability"] == 1.0


def test_the_sample_size_is_emitted_alongside_the_statistics():
    """Because every other key is a statistic OF the sample. A reader who
    cannot see the denominator cannot know that a probability of 0.75 came
    from eight paths."""
    out = paths_to_features(paths([101, 102, 103, 104, 105, 106, 99, 98]), atr=1.0)
    assert out["kronos_sample_count"] == 8.0
    assert out["kronos_horizon_bars"] == 4.0


def test_every_scaled_key_stays_bounded_under_an_absurd_forecast():
    """A 1000x move must not produce an unbounded feature: an unbounded input
    to a weighted score lets one observation dominate it."""
    out = paths_to_features(paths([100_000.0] * 8, anchor=100.0), atr=1.0)
    assert -1.0 <= out["kronos_expected_move_score"] <= 1.0
    assert 0.0 <= out["kronos_up_probability"] <= 1.0
    assert 0.0 <= out["kronos_agreement"] <= 1.0


def test_forecast_paths_rejects_degenerate_input():
    for bad in (np.zeros(0), np.asarray([np.nan]), np.asarray([np.inf])):
        with pytest.raises(Exception):
            ForecastPaths(terminal_closes=bad, anchor_close=100.0, horizon_bars=4)
    with pytest.raises(Exception):
        ForecastPaths(terminal_closes=np.asarray([1.0]), anchor_close=0.0, horizon_bars=4)


# ---------------------------------------------------------------------------
# the computer's contract and its degraded paths
# ---------------------------------------------------------------------------


def test_disabled_reports_missing_and_emits_no_forecast(synthetic_symbol_data):
    computer = KronosFeatures(KronosConfig(max_context=16))
    vector = computer.compute(view_at(synthetic_symbol_data, -1))
    assert vector.quality is DataQuality.MISSING
    assert vector.values["kronos_available"] == 0.0
    assert any("disabled" in n for n in vector.notes)


def test_a_missing_dependency_degrades_and_substitutes_nothing(synthetic_symbol_data):
    """torch is not installed here, so this is the live path rather than a
    hypothetical. Nothing may be substituted: the brief's rule against
    inventing missing data applies to a missing model as much as a missing
    feed."""
    computer = KronosFeatures(
        cfg(pretrain_cutoff=date(2010, 1, 1)),   # bar is clean, so the guard permits
        adapter=KronosAdapter(cfg(pretrain_cutoff=date(2010, 1, 1))),
    )
    vector = computer.compute(view_at(synthetic_symbol_data, -1))
    assert vector.quality is DataQuality.MISSING
    assert vector.values["kronos_available"] == 0.0
    assert all(v == 0.0 for k, v in vector.values.items() if k != "kronos_contaminated")
    assert any("not importable" in n or "torch" in n for n in vector.notes)


def test_the_adapter_caches_its_failure(synthetic_symbol_data):
    """Retrying a missing dependency once per bar would turn a typo into a
    very slow backtest rather than a fast error."""
    adapter = KronosAdapter(cfg())
    with pytest.raises(KronosUnavailable):
        adapter.load()
    assert adapter._load_error is not None
    with pytest.raises(KronosUnavailable):
        adapter.load()
    assert adapter.available() is False


def test_a_short_view_is_not_ready_rather_than_forecast(synthetic_symbol_data):
    computer = KronosFeatures(cfg(max_context=100_000))
    vector = computer.compute(view_at(synthetic_symbol_data, -1))
    assert vector.warmup_complete is False
    assert vector.quality is DataQuality.MISSING


def test_the_declared_contract(synthetic_symbol_data):
    computer = KronosFeatures(cfg(max_context=64))
    assert computer.keys == KRONOS_KEYS
    assert computer.warmup_bars == 64
    assert computer.required_feeds == frozenset({Feed.BARS})
    vector = computer.compute(view_at(synthetic_symbol_data, -1))
    assert tuple(sorted(vector.values)) == tuple(sorted(KRONOS_KEYS))
    assert all(math.isfinite(v) for v in vector.values.values())


def test_warmup_equals_the_model_context_and_is_not_minimised():
    """The forecast is conditioned on the whole context, so a shorter history
    is a DIFFERENT model input rather than a noisier one. Declaring less than
    the context would make the feature depend on where the caller started
    loading -- the same bug class that was HIGH severity in the regime
    detector."""
    for context in (16, 128, 512, 2048):
        assert KronosFeatures(cfg(max_context=context)).warmup_bars == context


def test_the_constructor_takes_configuration_only():
    """The package-wide ban, echoed. Sharply ironic here: the ban exists
    because a full-sample statistic captured at construction is the one leak
    the audit cannot see, and this module's whole problem is a DIFFERENT leak
    that same audit also cannot see. The ban still applies; it is just not
    sufficient."""
    import inspect

    params = list(inspect.signature(KronosFeatures.__init__).parameters)
    assert params == ["self", "config", "adapter"]
    hints = get_type_hints(KronosFeatures.__init__)
    assert not isinstance(hints.get("adapter"), str)


def test_two_computes_on_one_view_agree_bit_for_bit(synthetic_symbol_data):
    computer = KronosFeatures(
        cfg(pretrain_cutoff=date(2010, 1, 1), contamination_policy=KronosContamination.ALLOW),
        adapter=_OracleAdapter(future_close=123.0),
    )
    view = view_at(synthetic_symbol_data, -1)
    first, second = computer.compute(view), computer.compute(view)
    assert first.values == second.values
    assert first.notes == second.notes


def test_kronos_is_not_in_the_flow_score_by_default():
    """Section 7 fixes the five weights at a sum of 100 and every setup
    threshold was calibrated on that scale, so a sixth component silently
    rescales every gate."""
    assert KronosConfig().include_in_flow_score is False
    assert KronosConfig().enabled is False


def test_nothing_in_the_module_claims_the_forecast_works():
    """The brief: "Never say the system is profitable before testing it." No
    out-of-sample forecast accuracy has been measured by this project, and it
    cannot be until a cutoff is known."""
    import inspect

    from flow_model.features import kronos as module

    import re

    text = inspect.getsource(module).lower()
    # Word boundaries, not substrings. The first version of this check matched
    # "proven" inside "provenance" -- the same substring-vs-word false positive
    # that has now bitten this project three times (the hypotheses leakage
    # guard and the wall-clock guard were the other two, both fixed by
    # matching structure instead of text).
    for claim in ("proven", "profitable", "outperforms", "beats the market"):
        assert not re.search(rf"\b{re.escape(claim)}\b", text), f"module asserts {claim!r}"


# ---------------------------------------------------------------------------
# the cutoff question, settled by arithmetic rather than by a date
# ---------------------------------------------------------------------------


def test_no_credible_cutoff_clears_this_project_research_window():
    """The decisive result of the cutoff investigation, and the reason the
    quarantine is permanent for backtesting rather than provisional.

    I went looking for Kronos's training-data cutoff: the paper is arXiv
    2508.02739 (AAAI 2026), the corpus is "over 12 billion K-line records from
    45 global exchanges", and NEITHER the paper page, the repo, nor the model
    cards publish a cutoff. A third-party paper claims the corpus ends June
    2024, which I could not verify from this container. The authors' own
    finetune config loads Qlib data to 2025-06-05 and holds out 2024-07-01
    onward for backtesting -- circumstantial support for a mid-2024 corpus
    end, and no more than that.

    But the exact date turns out not to matter, which is why this test is
    arithmetic rather than a lookup. This project's research window ends
    2023-01-01. For Kronos to be clean across it, its corpus would have to end
    BEFORE 2015-01-01 -- and a model released in 2025, trained on 12 billion
    recent K-lines from 45 exchanges, cannot have a pre-2015 cutoff.

    So every candidate cutoff leaves 100% of the research window
    contaminated, and the only clean path is mode=LIVE. If this test ever
    fails it means the research window moved, and the conclusion needs
    redoing rather than assuming.
    """
    from flow_model.config.loader import load_config

    config = load_config()
    research_start = config.backtest.start
    guard_cutoffs = (
        date(2024, 6, 30),   # the third-party claim
        date(2025, 6, 5),    # the authors' own finetune data end
        date(2023, 12, 31),  # a generously early alternative
    )
    for cutoff in guard_cutoffs:
        assert cutoff >= research_start, (
            f"cutoff {cutoff} predates the research window start {research_start}; "
            "the arithmetic in this test's docstring needs redoing"
        )
        guard = ContaminationGuard(pretrain_cutoff=cutoff, mode=KronosMode.RESEARCH)
        first_bar = datetime(
            research_start.year, research_start.month, research_start.day, 15, 0, tzinfo=UTC
        )
        assert guard.verdict(first_bar).contaminated is True


def test_the_live_path_is_the_only_clean_one_and_it_works():
    """The complement: having established that no cutoff rescues the backtest,
    confirm the path that IS sound is not also blocked. Otherwise the
    integration would be unusable rather than merely restricted."""
    guard = ContaminationGuard(pretrain_cutoff=None, mode=KronosMode.LIVE)
    verdict = guard.verdict(datetime(2026, 6, 1, 14, 35, tzinfo=UTC))
    assert verdict.permitted is True
    assert verdict.contaminated is False
    assert verdict.quality is DataQuality.GOOD


def test_the_default_cutoff_stays_unknown_despite_the_third_party_claim():
    """Deliberate. Setting pretrain_cutoff=2024-06-30 on an unverified
    secondhand claim would convert a rumour into a license to backtest, and
    the guard exists to stop exactly that. It stays None until someone
    verifies a cutoff or trains a checkpoint whose cutoff they know."""
    assert KronosConfig().pretrain_cutoff is None
