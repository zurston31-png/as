"""Phase 11's gate: `live/paper.py`, the paper path, tested against the backtest.

Four things are established here, in the order their failure would matter.

**1. The zero-trade case is tested FIRST, because it is the only case.**
PROVENANCE.md 4a measured the population Flow Score over 19542 synthetic NQ
5-minute bars: 11239 bars carry a score, the maximum is 69.6335 against a
70.0 threshold, and ZERO bars reach 70.0. So every bar this module can emit
is a WAIT, and the tests that matter are the ones about WAITs: that each
carries its reason, that none carries a geometry, and that a session summary
over them completes without computing a ratio at n=0. A rate over zero
signals is `undefined[input_not_supplied]`, never 0.0.

**2. Backtest/live parity, bar for bar, with a proof that it can fail.**
`test_the_two_paths_are_identical_field_for_field` runs `BacktestLoop` and
`PaperSession` over the SAME `DataStore` and compares every field in
`PARITY_FIELDS` plus `stages_run` and `blocking_stage`. A parity test that
cannot fail is worse than none, so three separate non-vacuity proofs follow
it: a locally-built config variant, a bar slice shifted by one bar, and -- the
strongest -- a per-field mutation sweep that mutates each compared field in
turn and asserts the comparator names that field. The config variant is built
with `load_config(overrides=...)` in this file; `config/defaults.yaml` is not
touched and no threshold or weight is moved. It exists to make a comparison
FAIL, which is the opposite of tuning.

**3. "It cannot place an order" is checked, not trusted.** The module ships
`audit_cannot_place_orders()` and `audit_import_closure()`. Both are run
against the shipped file AND against deliberately bad inputs: stub module
sources that import `requests` or call `place_order`, and fake repository
trees whose `paper.py` reaches a network import two hops away or reaches
`flow_model.backtest`. Every match is on the AST, never on raw source text --
`paper.py` contains the strings "place_order" and "requests" inside its own
guard lists, so a text-matching test would flag the module it exists to
clear. `test_a_text_matching_guard_would_have_flagged_the_clean_module` pins
that trap open so nobody re-lays it.

**4. One engine, not two.** The session must run `SignalEngine.decide()` from
`flow_model/signals/engine.py`, not a subclass that overrides it and not
re-derived gate logic. Tested three ways: construction refuses an override of
`decide`, `decide_from` or `evaluate`; an instance-level spy proves `observe`
goes through `decide()`; and an AST read of the real file proves the module
imports no gate, scorer, setup, regime or feature module at all, so there is
nothing in it that *could* re-derive a score.

Everything here is synthetic. No real market bar has ever entered this system,
and `test_every_emitted_signal_carries_its_origin_warning` asserts that every
signal says so on its own face.

Two test doubles appear, and neither moves a threshold, a weight or a config
default:

* HAND-BUILT `Signal` and `EntryPlan` objects, for the directional branch of
  `PaperSignal`'s validator. The real engine emits no directional signal at
  the committed config, so the branch is otherwise unreachable; lowering a
  threshold to reach it is the move the brief forbids, so the records are
  constructed directly. They fabricate no performance claim -- what they test
  is that a LONG must carry its whole plan and that `planned_risk_dollars` is
  carried rather than recomputed.
* STUB MODULE SOURCES and FAKE REPOSITORY TREES, as strings and tmp_path
  files, for the two audits' non-vacuity. None is importable and none is part
  of the package.
"""

from __future__ import annotations

import ast
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from flow_model.analytics.metrics import MetricValue, Undefined
from flow_model.backtest.loop import BacktestLoop, StoreMarketSource
from flow_model.config.loader import default_config, load_config
from flow_model.core.contracts import Signal
from flow_model.core.determinism import stable_hash
from flow_model.core.enums import (
    Component,
    DataQuality,
    Feed,
    Regime,
    SetupType,
    SignalAction,
    Side,
    WaitReason,
)
from flow_model.dashboard.build import DataOrigin, origin_notice
from flow_model.data import DataStore, SyntheticAdapter, SyntheticConfig, TradingCalendar
from flow_model.data.quality import QualityGrader
from flow_model.data.store import SymbolData
from flow_model.live import paper as paper_module
from flow_model.live.paper import (
    FORBIDDEN_CALL_NAMES,
    FORBIDDEN_IDENTIFIER_WORDS,
    GUARD_IDENTIFIER_EXEMPTIONS,
    NETWORK_ROOTS,
    PERMITTED_IMPORT_ROOTS,
    SYNTHETIC_WARNING,
    UNDECLARED_WARNING,
    PaperError,
    PaperSession,
    PaperSessionSummary,
    PaperSignal,
    audit_cannot_place_orders,
    audit_import_closure,
    origin_warning,
    reachable_repository_modules,
)
from flow_model.risk.limits import RiskLimitManager
from flow_model.signals.engine import EntryPlan, SignalEngine

SYMBOL = "NQ"
T0 = datetime(2016, 1, 5, 15, 0, tzinfo=timezone.utc)

#: The section 12 fields both paths carry on their per-bar record. The paper
#: path's are read off `PaperSignal.signal` -- the engine's own contract
#: object -- so what is compared is the `Signal`, not a projection of it.
PARITY_FIELDS: tuple[str, ...] = (
    "symbol",
    "ts",
    "action",
    "side",
    "setup",
    "flow_score",
    "confidence",
    "regime",
    "data_quality",
    "wait_reason",
    "wait_detail",
    "reasons",
    "component_points",
    "reference_price",
    "atr",
)

#: Carried beside the signal on both paths: the audit trail of which stages
#: ran and which one stopped the bar.
PARITY_TRAIL_FIELDS: tuple[str, ...] = ("stages_run", "blocking_stage")


# ---------------------------------------------------------------------------
# Data. 2016 bars, far from the sealed window (2023-01-01 .. 2025-01-01),
# which is neither requested nor opened anywhere in this file.
# ---------------------------------------------------------------------------


def build_store(start: date, end: date, config) -> tuple[DataStore, TradingCalendar]:
    """A synthetic store for one window. The config is the committed default."""
    calendar = TradingCalendar()
    adapter = SyntheticAdapter(
        config=SyntheticConfig(),
        seed=config.seed,
        spec_by_symbol=dict(config.instruments),
        interval_seconds=config.data.primary_interval_seconds,
        calendar=calendar,
    )
    store = DataStore(latency_seconds=config.data.data_latency_seconds)
    store.load(
        adapter,
        SYMBOL,
        start,
        end,
        primary_interval=config.data.primary_interval_seconds,
    )
    return store, calendar


@pytest.fixture(scope="module")
def config():
    return default_config()


@pytest.fixture(scope="module")
def dataset(config):
    """A five-week slice: long enough to clear warmup and reach real gates."""
    store, calendar = build_store(date(2016, 1, 4), date(2016, 2, 8), config)
    return store, calendar


@pytest.fixture(scope="module")
def short_dataset(config):
    """A shorter slice, for the tests that only need two paths to disagree."""
    store, calendar = build_store(date(2016, 1, 4), date(2016, 1, 12), config)
    return store, calendar


class SlicedFeed:
    """A `ViewFeed` over a prefix of a `StoreMarketSource`'s clock.

    Satisfies `paper.ViewFeed` structurally, which is the point: the protocol
    is narrower than `backtest.loop.MarketSource` so that the paper path does
    not depend on `flow_model.backtest`, and something other than
    `StoreMarketSource` ought to be able to satisfy it.
    """

    def __init__(self, source, clock) -> None:
        self._source = source
        self._clock = tuple(int(ts) for ts in clock)

    def clock_ns(self):
        return self._clock

    def advance(self, ts_ns: int) -> None:
        self._source.advance(int(ts_ns))

    def view(self):
        return self._source.view()


def backtest_signals(config, store, calendar, clock=None):
    """The backtest path's per-bar rows."""
    loop = BacktestLoop(
        config, SYMBOL, calendar=calendar, git_commit="", git_dirty=False
    )
    result = loop.run(
        StoreMarketSource.from_config(store, SYMBOL, config), clock_ns=clock
    )
    return result


def paper_signals(config, store, calendar, clock=None):
    """The paper path's per-bar records, over the same feed.

    `equity` and `limits` are handed over explicitly so parity is exact by
    construction rather than by coincidence: the loop passes `ledger.equity`
    and its own `RiskLimitManager` into `decide()`, and at zero trades the
    ledger never moves off `starting_equity`.
    """
    equity = float(config.risk.starting_equity)
    session = PaperSession.from_store(
        config,
        SYMBOL,
        store,
        calendar=calendar,
        equity=equity,
        limits=RiskLimitManager(config.risk, equity),
    )
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    ticks = source.clock_ns() if clock is None else clock
    return session, session.replay_feed(SlicedFeed(source, ticks))


@pytest.fixture(scope="module")
def both_paths(config, dataset):
    """The backtest result and the paper records over one identical dataset."""
    store, calendar = dataset
    result = backtest_signals(config, store, calendar)
    session, records = paper_signals(config, store, calendar)
    return result, session, records


@pytest.fixture(scope="module")
def emitted(both_paths):
    _, _, records = both_paths
    return records


# ---------------------------------------------------------------------------
# 1. THE ZERO-TRADE CASE. Not an edge case -- the only case.
# ---------------------------------------------------------------------------


def test_every_bar_emits_wait_at_the_committed_config(emitted):
    """PROVENANCE 4a: zero bars reach 70.0, so there is nothing but WAIT.

    This is asserted over every bar rather than sampled, because "mostly
    WAIT" and "only WAIT" are different findings and only one of them is the
    measured one.
    """
    assert emitted, "the session observed no bars at all"
    assert all(record.action is SignalAction.WAIT for record in emitted)
    assert {record.action for record in emitted} == {SignalAction.WAIT}
    assert all(record.side is None for record in emitted)
    assert all(record.setup is None for record in emitted)


def test_every_wait_carries_its_reason(emitted):
    """A WAIT without a reason is an unexplained rejection, which makes the
    rejection histogram section 12 asks for impossible to build."""
    assert all(record.wait_reason is not None for record in emitted)
    assert all(isinstance(record.wait_reason, WaitReason) for record in emitted)
    # The reasons are not all one value: the pipeline short-circuits at
    # several different stages over this window.
    assert len({record.wait_reason for record in emitted}) > 1


def test_no_wait_carries_a_geometry(emitted):
    """0.0 for an entry that was never planned reads as a measurement."""
    for record in emitted:
        assert record.entry_price is None
        assert record.stop_price is None
        assert record.target_price is None
        assert record.reward_risk is None
        assert record.size is None
        assert record.planned_risk_dollars is None
        assert record.point_value is None
        assert record.achievable_reward_risk is None
        assert record.max_hold_bars is None


def test_no_observed_bar_reaches_the_lowest_setup_threshold(config, emitted):
    """The measured fact behind the zero-trade case, re-measured here.

    PROVENANCE 4a's figure of record is a full-year population maximum of
    69.6335 against 70.0. This window is five weeks, so its own maximum is
    lower; what is asserted is the claim that holds at any window length --
    no bar reaches the lowest setup threshold.
    """
    thresholds = [setup.min_flow_score for setup in config.setups.values()]
    assert thresholds, "the config declares no setups"
    floor = min(thresholds)
    assert floor == pytest.approx(70.0), (
        "this test is written against a 70.0 lowest setup threshold; the "
        f"config now says {floor}"
    )
    scores = [r.flow_score for r in emitted if r.flow_score is not None]
    assert scores, "no bar reached the Flow Score stage, so nothing was measured"
    assert max(scores) < floor
    assert not [s for s in scores if s >= floor]
    # Bounded 0..100 by the Phase 1 contract, which is what the threshold is
    # compared against.
    assert all(0.0 <= s <= 100.0 for s in scores)


def test_the_session_summary_completes_at_zero_trades(both_paths):
    """No crash, and counts that add up, over a run with nothing in it."""
    _, session, records = both_paths
    summary = session.summary()
    assert isinstance(summary, PaperSessionSummary)
    assert summary.symbol == SYMBOL
    assert summary.bars_observed == len(records)
    assert summary.directional_bars == 0
    assert summary.by_action == {SignalAction.WAIT: len(records)}
    assert sum(summary.by_wait_reason.values()) == len(records)
    assert summary.first_ts == records[0].ts.isoformat()
    assert summary.last_ts == records[-1].ts.isoformat()
    assert summary.summary_lines()
    # The counts are defined at n=0 and are reported; it is only the RATIOS
    # that are withheld.
    assert summary.scored_bars == len(
        [r for r in records if r.flow_score is not None]
    )


def test_a_rate_over_zero_signals_is_undefined_never_zero(config, short_dataset):
    """n=0 is the case this module will actually be run in on day one.

    A session that has observed nothing must say `undefined`, because 0.0
    reads as "measured, and the answer was none".
    """
    store, calendar = short_dataset
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    assert session.signals == ()
    summary = session.summary()
    assert summary.bars_observed == 0
    for metric in (
        summary.scored_fraction,
        summary.directional_fraction,
        summary.max_flow_score,
    ):
        assert isinstance(metric, MetricValue)
        assert metric.is_defined is False
        assert metric.value is None
        assert metric.or_none() is None
        assert metric.undefined is Undefined.INPUT_NOT_SUPPLIED
        assert metric.n == 0
        assert metric.note
        assert "undefined" in str(metric)
        with pytest.raises(Exception):
            metric.require()
    # And no second representation of absence: no 0.0 anywhere in the lines.
    assert summary.summary_lines()
    assert summary.by_action == {}
    assert summary.by_wait_reason == {}
    assert summary.first_ts == ""
    assert summary.last_ts == ""


def test_max_flow_score_is_undefined_when_no_bar_was_scored(config, short_dataset):
    """A maximum over an empty set is undefined, not zero, not minus infinity."""
    store, calendar = short_dataset
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    # The first bars of any window are in warmup, so none of them is scored.
    session.replay_feed(SlicedFeed(source, list(source.clock_ns())[:40]))
    summary = session.summary()
    assert summary.bars_observed == 40
    assert summary.scored_bars == 0
    assert summary.max_flow_score.value is None
    assert summary.max_flow_score.undefined is Undefined.INPUT_NOT_SUPPLIED
    # But the rate over OBSERVED bars is now defined, and is a real 0.0: zero
    # of forty bars were scored, which is a measurement, not an absence.
    assert summary.scored_fraction.is_defined is True
    assert summary.scored_fraction.value == 0.0
    assert summary.scored_fraction.n == 40


def test_the_summary_is_defined_over_a_scored_window(both_paths):
    """Where a denominator exists, the rate is reported as a number."""
    _, session, records = both_paths
    summary = session.summary()
    assert summary.scored_fraction.is_defined is True
    assert summary.scored_fraction.n == len(records)
    assert summary.directional_fraction.is_defined is True
    assert summary.directional_fraction.value == 0.0
    assert summary.max_flow_score.is_defined is True
    assert summary.max_flow_score.n == summary.scored_bars
    assert summary.max_flow_score.value == max(
        r.flow_score for r in records if r.flow_score is not None
    )


def test_the_summary_says_no_instruction_to_act_is_produced(both_paths):
    _, session, _ = both_paths
    lines = session.summary().summary_lines()
    assert any("No instruction to act is produced" in line for line in lines)


# ---------------------------------------------------------------------------
# 2. BACKTEST/LIVE PARITY -- the Phase 11 gate.
# ---------------------------------------------------------------------------


def parity_mismatches(rows, records) -> list[str]:
    """Every field on which the two paths disagree. Empty means identical.

    `rows` are `SignalRecord`s from `BacktestLoop`; `records` are
    `PaperSignal`s. The signal fields are read off `record.signal`, the
    engine's own contract object, so this compares the `Signal` rather than a
    projection of it.
    """
    problems: list[str] = []
    if len(rows) != len(records):
        problems.append(f"bar count: backtest {len(rows)} vs paper {len(records)}")
    for row, record in zip(rows, records):
        for field in PARITY_FIELDS:
            left = getattr(row, field)
            right = getattr(record.signal, field)
            if left != right:
                problems.append(f"{field}: backtest {left!r} vs paper {right!r}")
        for field in PARITY_TRAIL_FIELDS:
            left = getattr(row, field)
            right = getattr(record, field)
            if left != right:
                problems.append(f"{field}: backtest {left!r} vs paper {right!r}")
    return problems


def parity_fingerprint(items, read) -> str:
    """A content hash of every compared field, in bar order."""
    return stable_hash(
        [
            {
                field: _hashable(getattr(read(item), field))
                for field in PARITY_FIELDS
            }
            for item in items
        ]
    )


def _hashable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {getattr(k, "value", str(k)): v for k, v in sorted(
            value.items(), key=lambda kv: getattr(kv[0], "value", str(kv[0]))
        )}
    return getattr(value, "value", value)


def test_the_two_paths_see_the_same_number_of_bars(both_paths):
    result, _, records = both_paths
    assert len(records) == len(result.signals)
    assert len(records) == result.provenance.bars_processed
    assert result.waits_dropped == 0, (
        "record_wait_signals is off, so the backtest kept fewer rows than bars "
        "and a parity comparison would be over a filtered set"
    )


def test_the_two_paths_are_identical_field_for_field(both_paths):
    """THE PHASE 11 GATE. Not similar -- identical, bar for bar.

    Section 12: "reuses the *identical* `signal_engine` object as the
    backtest (no parallel implementation; this is enforced by a test that
    runs both paths over the same synthetic data and asserts identical
    signals)."
    """
    result, _, records = both_paths
    assert result.trade_count == 0, (
        "the committed config took a trade; this test's equity-parity "
        "argument assumes the ledger never moves off starting_equity"
    )
    mismatches = parity_mismatches(result.signals, records)
    assert mismatches == [], (
        f"{len(mismatches)} parity mismatch(es); first five: {mismatches[:5]}"
    )


def test_the_parity_window_is_wide_enough_to_mean_something(both_paths):
    """A gate over three warmup bars would pass forever.

    Pinned so that a later shrink of the fixture window shows up here rather
    than as a quietly weaker Phase 11 gate.
    """
    result, _, records = both_paths
    assert len(records) >= 1500, f"only {len(records)} bars compared"
    reasons = {record.wait_reason for record in records}
    assert len(reasons) >= 8, f"only {len(reasons)} distinct wait reason(s) exercised"
    stages = {stage for record in records for stage in record.stages_run}
    assert len(stages) >= 9, f"only {len(stages)} pipeline stage(s) reached"
    assert "flow_score" in stages, "no bar reached the Flow Score stage"
    assert len(records) * (len(PARITY_FIELDS) + len(PARITY_TRAIL_FIELDS)) > 20000
    assert len(result.signals) == len(records)


def test_the_two_paths_agree_that_no_geometry_was_produced(both_paths):
    """The entry/stop/target/R:R half of section 12's list, at zero trades.

    The backtest never built an intent and the paper path carries no plan.
    Both are statements of the same fact and they must not diverge: a paper
    path reporting a geometry on a bar the backtest declined to size would be
    the drift the gate exists to catch.
    """
    result, _, records = both_paths
    assert not any(row.intent_built for row in result.signals)
    assert not any(row.order_submitted for row in result.signals)
    assert result.orders == ()
    assert result.submissions_declined == 0
    assert all(record.entry_price is None for record in records)
    assert all(record.stop_price is None for record in records)
    assert all(record.target_price is None for record in records)
    assert all(record.reward_risk is None for record in records)
    assert all(record.planned_risk_dollars is None for record in records)


def test_the_two_paths_hash_identically(both_paths):
    """The same statement as one number, so "identical" is not field-by-field
    luck across a list someone later shortens."""
    result, _, records = both_paths
    assert parity_fingerprint(result.signals, lambda row: row) == parity_fingerprint(
        records, lambda record: record.signal
    )


def test_the_parity_field_list_covers_every_signal_field():
    """A field added to `Signal` and left out of `PARITY_FIELDS` would drift
    between the two paths with the gate still green."""
    declared = set(Signal.model_fields) | {"side"}
    missing = declared - set(PARITY_FIELDS)
    assert missing == set(), (
        f"`Signal` carries {sorted(missing)}, which the parity comparison does "
        "not look at"
    )


# --- non-vacuity: the comparison must be able to fail ----------------------


def test_the_parity_comparison_fails_on_a_locally_perturbed_config(
    config, short_dataset
):
    """NON-VACUITY 1: a different config on one path must break parity.

    The variant is built HERE with `load_config(overrides=...)`;
    `config/defaults.yaml` is untouched and nothing is tuned. Raising the
    data-quality floor from DEGRADED to GOOD is chosen because it changes the
    decision at a gate that every bar passes through, so the two paths cannot
    agree by accident. The point is that the comparison FAILS -- if it
    passes, the comparator is not reading the fields it claims to.
    """
    store, calendar = short_dataset
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    clock = list(source.clock_ns())[:300]

    perturbed = load_config(overrides=["data.min_quality_to_trade=GOOD"])
    assert perturbed.config_hash != config.config_hash
    assert perturbed.data.min_quality_to_trade != config.data.min_quality_to_trade

    result = backtest_signals(config, store, calendar, clock=clock)
    _, records = paper_signals(perturbed, store, calendar, clock=clock)

    mismatches = parity_mismatches(result.signals, records)
    assert mismatches, (
        "the two paths agreed although one ran a different config, so the "
        "parity comparison cannot detect a drifted implementation"
    )
    assert any("wait_reason" in problem for problem in mismatches)


def test_the_parity_comparison_fails_on_a_shifted_bar_slice(config, short_dataset):
    """NON-VACUITY 2: the comparison must notice misaligned bars.

    Two paths over the same config but clocks offset by one bar. If this
    passed, parity would be comparing unordered bags rather than bar for bar.
    """
    store, calendar = short_dataset
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    clock = list(source.clock_ns())
    result = backtest_signals(config, store, calendar, clock=clock[:200])
    _, records = paper_signals(config, store, calendar, clock=clock[1:201])

    mismatches = parity_mismatches(result.signals, records)
    assert mismatches
    assert any("ts" in problem for problem in mismatches)


MUTATIONS: dict[str, object] = {
    "symbol": "ES",
    "ts": T0 + timedelta(days=900),
    "action": SignalAction.LONG,
    "setup": SetupType.SCALP_1R,
    "flow_score": 99.0,
    "confidence": 0.5,
    "regime": Regime.TRENDING_UP,
    "data_quality": DataQuality.GOOD,
    "wait_reason": WaitReason.RISK_LIMIT,
    "wait_detail": "a detail no engine produced",
    "reasons": ("a reason no engine produced",),
    "component_points": {Component.STRUCTURE: 123.0},
    "reference_price": 123456.0,
    "atr": 98765.0,
}


@pytest.mark.parametrize("field", sorted(MUTATIONS))
def test_the_comparator_catches_a_mutation_in_every_compared_field(
    both_paths, field
):
    """NON-VACUITY 3, the strongest: each field is really compared.

    One field of one paper record is mutated and the comparator must name
    that field. A field that is in `PARITY_FIELDS` but never actually read --
    a typo, a property that silently returns the same thing on both sides --
    passes the parity test forever and fails here.

    `model_copy` is used deliberately: it skips validation, which is what
    builds a record no engine could have produced.
    """
    result, _, records = both_paths
    index = len(records) // 2
    original = getattr(records[index].signal, field)
    mutant_value = MUTATIONS[field]
    assert original != mutant_value, (
        f"the mutation for {field!r} equals the real value, so this test "
        "would pass without the comparator doing anything"
    )

    mutated_signal = records[index].signal.model_copy(update={field: mutant_value})
    mutated = records[index].model_copy(update={"signal": mutated_signal})
    spoiled = list(records)
    spoiled[index] = mutated

    mismatches = parity_mismatches(result.signals, spoiled)
    assert mismatches, f"mutating {field!r} did not break the comparison"
    assert any(problem.startswith(f"{field}:") for problem in mismatches), (
        f"the comparator noticed something but did not name {field!r}: "
        f"{mismatches}"
    )


@pytest.mark.parametrize(
    "field,mutant_value",
    [("stages_run", ("a_stage_that_does_not_exist",)), ("blocking_stage", "nowhere")],
)
def test_the_comparator_catches_a_mutation_in_the_audit_trail(
    both_paths, field, mutant_value
):
    """The stages ran and the stage that stopped the bar are compared too."""
    result, _, records = both_paths
    index = len(records) // 2
    assert getattr(records[index], field) != mutant_value
    spoiled = list(records)
    spoiled[index] = records[index].model_copy(update={field: mutant_value})
    mismatches = parity_mismatches(result.signals, spoiled)
    assert any(problem.startswith(f"{field}:") for problem in mismatches)


def test_the_comparator_catches_a_dropped_bar(both_paths):
    result, _, records = both_paths
    mismatches = parity_mismatches(result.signals, list(records)[:-1])
    assert any("bar count" in problem for problem in mismatches)


def test_replaying_one_dataset_twice_yields_identical_signals(config, short_dataset):
    """No clock is read, so a replay is reproducible.

    Every timestamp on a `PaperSignal` comes from the view that produced it.
    If a wall clock leaked in anywhere, these two fingerprints would differ.
    """
    store, calendar = short_dataset
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    clock = list(source.clock_ns())[:150]
    _, first = paper_signals(config, store, calendar, clock=clock)
    _, second = paper_signals(config, store, calendar, clock=clock)
    assert parity_fingerprint(first, lambda r: r.signal) == parity_fingerprint(
        second, lambda r: r.signal
    )
    assert [r.ts for r in first] == [r.ts for r in second]
    assert [r.summary_lines() for r in first] == [r.summary_lines() for r in second]


def test_every_timestamp_comes_from_its_own_view(config, short_dataset):
    """The signal's ts is the view's instant, not "now"."""
    store, calendar = short_dataset
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    clock = list(source.clock_ns())[:60]
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    records = session.replay_feed(SlicedFeed(source, clock))
    stamps = [int(r.ts.timestamp() * 1_000_000_000) for r in records]
    assert stamps == clock
    assert all(r.ts == r.signal.ts for r in records)
    # All of them are 2016 bars, so nothing was stamped with today's date.
    assert all(r.ts.year == 2016 for r in records)


# ---------------------------------------------------------------------------
# 3. IT CANNOT PLACE AN ORDER.
# ---------------------------------------------------------------------------


def test_the_self_audit_is_clean_on_the_shipped_module():
    """`audit_cannot_place_orders()` over `paper.py` itself: empty, as promised.

    This shipped returning SIX offences, all of them the guard flagging its
    own declarations (`EXECUTION_PACKAGE` three times, the audit function's
    own name twice, and its `getattr` call). A guard that cannot clear the
    module it guards clears nothing, because a real violation would arrive as
    a seventh line nobody was watching.
    """
    assert audit_cannot_place_orders() == ()


def test_the_closure_audit_is_clean_on_the_shipped_repository():
    """Nothing reachable from this module can open a socket or build a fill."""
    offences = audit_import_closure()
    assert offences == (), f"reachable hazards: {offences}"
    reachable = reachable_repository_modules()
    assert len(reachable) > 20, (
        "the closure walked almost nothing, so its empty result says little"
    )
    assert "flow_model.signals.engine" in reachable
    assert not [m for m in reachable if m.startswith("flow_model.backtest")]


#: Module sources that MUST be reported. Each is a real way this module could
#: acquire the ability to act, written as source rather than described.
BAD_SOURCES: dict[str, str] = {
    "imports requests": "import requests\n",
    "imports from requests": "from requests import Session\n",
    "imports a broker sdk": "import ib_insync\n",
    "imports socket": "import socket\n",
    "imports asyncio": "import asyncio\n",
    "imports subprocess": "import subprocess\n",
    "calls place_order on an object": (
        "import flow_model\n\n\ndef go(client):\n    return client.place_order(1)\n"
    ),
    "calls a bare place_order": (
        "import flow_model\n\n\ndef go():\n    return place_order(1)\n"
    ),
    "calls submit_order": (
        "import flow_model\n\n\ndef go(client):\n    return client.submit_order()\n"
    ),
    "names a position": (
        "import flow_model\n\n\ndef go(book):\n    return book.on_position_opened()\n"
    ),
    "function-local import": (
        "import flow_model\n\n\ndef go():\n    import ast\n\n    return ast\n"
    ),
    "relative import": "from . import sibling\n",
    "async def": "import flow_model\n\n\nasync def go():\n    return 1\n",
    "await": (
        "import flow_model\n\n\nasync def go(thing):\n    return await thing()\n"
    ),
    "calls getattr": (
        "import flow_model\n\n\ndef go(x):\n    return getattr(x, 'y')\n"
    ),
    "calls eval": "import flow_model\n\n\ndef go(s):\n    return eval(s)\n",
    "calls open": "import flow_model\n\n\ndef go(p):\n    return open(p)\n",
    "reads a clock": "import flow_model\n\n\ndef go(c):\n    return c.now()\n",
}


@pytest.mark.parametrize("label", sorted(BAD_SOURCES))
def test_the_self_audit_trips_on_a_module_that_could_act(label):
    """NON-VACUITY: the checker is fed a module that DOES the forbidden thing.

    Every one of these must be named. A checker that returns an empty tuple
    for `paper.py` and also for a file that imports `requests` has told us
    nothing about `paper.py`.
    """
    offences = audit_cannot_place_orders(BAD_SOURCES[label])
    assert offences, f"{label}: the audit cleared a module that can act"
    assert all(isinstance(line, str) and line for line in offences)


#: Module sources that MUST NOT be reported. This project has had three false
#: positives from substring guards, one of which flagged accurate prose in a
#: docstring, so the clean cases are tested as hard as the dirty ones.
CLEAN_SOURCES: dict[str, str] = {
    "forbidden words in a docstring": (
        '"""This module will never place_order and never import requests."""\n'
        "import flow_model\n"
    ),
    "forbidden words in a string literal": (
        'import flow_model\n\nGUARD = "requests socket place_order submit_order"\n'
    ),
    "is_tradable is not trade": (
        "import flow_model\n\n\ndef is_tradable(x):\n    return bool(x)\n"
    ),
    "reasons is not send": "import flow_model\n\nreasons = ()\n",
    "session is not a requests session": (
        "import flow_model\n\n\nclass PaperSession:\n    pass\n"
    ),
    "every permitted root": (
        "from __future__ import annotations\n\n"
        "import ast\nfrom collections.abc import Mapping\nfrom pathlib import Path\n"
        "from typing import Protocol\n\nfrom pydantic import Field\n\n"
        "import flow_model\n"
    ),
}


@pytest.mark.parametrize("label", sorted(CLEAN_SOURCES))
def test_the_self_audit_clears_a_module_that_only_talks_about_acting(label):
    """Matched on the AST, so a string cannot trip it and cannot hide in it."""
    offences = audit_cannot_place_orders(CLEAN_SOURCES[label])
    assert offences == (), f"{label}: false positive -- {offences}"


def test_a_text_matching_guard_would_have_flagged_the_clean_module():
    """The trap, pinned open.

    `paper.py` contains the literal strings "requests" and "place_order"
    inside its own guard lists and docstring. A test that grepped the source
    instead of parsing it would report the module it exists to clear. This
    asserts the strings really are there, so the next person to reach for
    `in source` sees why that does not work.
    """
    source = Path(paper_module.__file__).read_text(encoding="utf-8")
    assert "requests" in source
    assert "place_order" in source
    assert "socket" in source
    # And yet the AST audit is clean, which is the whole point.
    assert audit_cannot_place_orders(source) == ()


def test_the_audit_matches_identifier_segments_not_substrings():
    """`is_tradable` is not a `trade`, and `reasons` is not a `send`."""
    assert "trade" in FORBIDDEN_IDENTIFIER_WORDS
    assert "send" in FORBIDDEN_IDENTIFIER_WORDS
    assert "tradable" not in FORBIDDEN_IDENTIFIER_WORDS
    assert "reasons" not in FORBIDDEN_IDENTIFIER_WORDS
    assert audit_cannot_place_orders(
        "import flow_model\n\n\ndef is_tradable(reasons):\n    return reasons\n"
    ) == ()
    # But the segment itself is caught.
    assert audit_cannot_place_orders(
        "import flow_model\n\n\ndef is_trade(x):\n    return x\n"
    )


def test_the_import_rule_is_an_allowlist_not_a_denylist(config):
    """A denylist of broker SDKs is a list of the ones that existed when it
    was written. Any root outside the allowlist is reported, including one
    that is on no hazard list at all."""
    assert "requests" not in PERMITTED_IMPORT_ROOTS
    assert not (PERMITTED_IMPORT_ROOTS & NETWORK_ROOTS)
    # A harmless-looking, entirely unknown root is still refused.
    offences = audit_cannot_place_orders("import some_broker_sdk_from_2030\n")
    assert offences
    assert any("PERMITTED_IMPORT_ROOTS" in line for line in offences)
    # And the module's own roots are exactly the permitted ones.
    tree = ast.parse(Path(paper_module.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert roots <= PERMITTED_IMPORT_ROOTS, f"unexpected roots: {roots}"
    assert not (roots & NETWORK_ROOTS)


def test_the_guard_exemption_is_one_name_and_is_load_bearing(monkeypatch):
    """Every exemption is a word the scan can no longer see on that spelling.

    So the set is pinned at exactly one member -- the public audit's own name,
    which cannot say what it checks without containing `place` and `orders` --
    and the exemption is shown to be load-bearing by removing it and watching
    the audit flag the module again.
    """
    assert GUARD_IDENTIFIER_EXEMPTIONS == frozenset({"audit_cannot_place_orders"})
    monkeypatch.setattr(paper_module, "GUARD_IDENTIFIER_EXEMPTIONS", frozenset())
    offences = audit_cannot_place_orders()
    assert offences, "the exemption is exempting nothing and can be deleted"
    assert all("audit_cannot_place_orders" in line for line in offences), (
        f"the exemption is hiding more than its own name: {offences}"
    )


def test_the_exemption_does_not_let_a_module_that_acts_through():
    """The residual, bounded: a module can borrow the exempt name and is
    still caught by everything else in it."""
    assert audit_cannot_place_orders(
        "import requests\n\n\ndef audit_cannot_place_orders(client):\n"
        "    return client.send(1)\n"
    )
    assert audit_cannot_place_orders(
        "import flow_model\n\n\ndef audit_cannot_place_orders(client):\n"
        "    return client.submit_order(1)\n"
    )


def test_the_forbidden_call_list_covers_dynamic_access_and_every_clock():
    for name in ("eval", "exec", "compile", "__import__", "getattr", "open"):
        assert name in FORBIDDEN_CALL_NAMES
    for name in ("now", "utcnow", "today", "time", "monotonic", "sleep"):
        assert name in FORBIDDEN_CALL_NAMES


# --- the closure audit, against fake repository trees ----------------------


def write_fake_repo(root: Path, paper_source: str, others: dict[str, str]) -> Path:
    """A fake repository whose `flow_model/live/paper.py` is what we say.

    `audit_import_closure(root)` reads files off disk rather than importing
    them, so a tree like this exercises it without any of this being
    importable.
    """
    (root / "flow_model" / "live").mkdir(parents=True, exist_ok=True)
    (root / "flow_model" / "__init__.py").write_text("", encoding="utf-8")
    (root / "flow_model" / "live" / "__init__.py").write_text("", encoding="utf-8")
    (root / "flow_model" / "live" / "paper.py").write_text(
        paper_source, encoding="utf-8"
    )
    for relative, source in others.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def test_the_closure_audit_trips_on_a_network_import_two_hops_away(tmp_path):
    """NON-VACUITY: the closure really is transitive.

    `paper.py` -> `helper` -> `deeper` -> `requests`. A one-hop check would
    clear this tree, which is exactly the hazard the audit exists for: the
    broker SDK will not be imported here, it will be imported by something
    that something here imports.
    """
    root = write_fake_repo(
        tmp_path / "two_hops",
        "from flow_model.live import helper\n",
        {
            "flow_model/live/helper.py": "from flow_model.live import deeper\n",
            "flow_model/live/deeper.py": "import requests\n",
        },
    )
    offences = audit_import_closure(root)
    assert offences
    assert any("deeper" in line and "requests" in line for line in offences)


def test_the_closure_audit_trips_on_a_direct_network_import(tmp_path):
    root = write_fake_repo(tmp_path / "direct", "import socket\n", {})
    offences = audit_import_closure(root)
    assert offences
    assert any("socket" in line for line in offences)


def test_the_closure_audit_trips_on_reaching_the_order_machinery(tmp_path):
    """`flow_model.backtest` is where an intent becomes a fill. Reaching it
    from the paper path is reported even though nothing there is a network
    import."""
    root = write_fake_repo(
        tmp_path / "execution",
        "from flow_model.backtest import loop\n",
        {
            "flow_model/backtest/__init__.py": "",
            "flow_model/backtest/loop.py": "VALUE = 1\n",
        },
    )
    offences = audit_import_closure(root)
    assert offences
    assert any("flow_model.backtest" in line for line in offences)


def test_the_closure_audit_clears_a_clean_tree(tmp_path):
    """So the trips above are the audit working, not the audit always firing."""
    root = write_fake_repo(
        tmp_path / "clean",
        "from flow_model.live import helper\n",
        {"flow_model/live/helper.py": "import math\n"},
    )
    assert audit_import_closure(root) == ()
    assert reachable_repository_modules(root) == (
        "flow_model.live",
        "flow_model.live.helper",
    )


def test_the_module_has_no_order_placing_surface():
    """Nothing in the public surface offers a way to act."""
    surface = [name for name in dir(paper_module) if not name.startswith("_")]
    callables = [
        name
        for name in surface
        if callable(getattr(paper_module, name))
    ]
    for name in callables:
        segments = set(name.lower().split("_"))
        offending = segments & (FORBIDDEN_IDENTIFIER_WORDS - GUARD_IDENTIFIER_EXEMPTIONS)
        assert not offending or name in GUARD_IDENTIFIER_EXEMPTIONS, (
            f"public callable {name!r} is named after acting: {offending}"
        )


# ---------------------------------------------------------------------------
# 4. ONE ENGINE, NOT TWO.
# ---------------------------------------------------------------------------


def test_the_session_runs_the_backtests_signal_engine(config, dataset):
    """The same class, from the same module, not a look-alike."""
    store, calendar = dataset
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    assert isinstance(session.engine, SignalEngine)
    assert type(session.engine) is SignalEngine
    assert type(session.engine).__module__ == "flow_model.signals.engine"
    assert session.engine.symbol == SYMBOL
    # The entry point is `decide`, and it is SignalEngine's own.
    assert type(session.engine).decide is SignalEngine.decide
    assert type(session.engine).decide_from is SignalEngine.decide_from
    assert type(session.engine).evaluate is SignalEngine.evaluate


def test_the_session_accepts_the_very_engine_the_backtest_holds(config, dataset):
    """Section 12's "identical object": one engine, handed to both."""
    store, calendar = dataset
    engine = SignalEngine(config, SYMBOL, calendar=calendar)
    loop = BacktestLoop(
        config, SYMBOL, engine=engine, calendar=calendar, git_commit="", git_dirty=False
    )
    session = PaperSession(config, SYMBOL, engine=engine, calendar=calendar)
    assert session.engine is engine
    assert loop.engine is engine
    assert session.engine is loop.engine


@pytest.mark.parametrize("method", ["decide", "decide_from", "evaluate"])
def test_a_subclass_that_overrides_the_pipeline_is_refused(config, method):
    """The one way a parallel implementation could hide behind the right type.

    An `isinstance` check passes for all three of these. Comparing the bound
    methods against `SignalEngine`'s own is what catches them.
    """
    namespace = {method: lambda self, *a, **k: None}
    sneaky = type("SneakyEngine", (SignalEngine,), namespace)
    with pytest.raises(PaperError, match=f"overrides SignalEngine.{method}"):
        PaperSession(config, SYMBOL, engine=sneaky(config, SYMBOL))


def test_a_subclass_that_overrides_nothing_is_accepted(config):
    """The check is about the pipeline, not about the class tree. A subclass
    that adds a helper and overrides no stage still runs the same code."""
    plain = type("PlainSubclass", (SignalEngine,), {"extra": lambda self: 1})
    session = PaperSession(config, SYMBOL, engine=plain(config, SYMBOL))
    assert type(session.engine).decide is SignalEngine.decide


def test_something_that_is_not_the_engine_is_refused(config):
    class LooksLikeAnEngine:
        symbol = SYMBOL

        def decide(self, view, **kwargs):
            raise AssertionError("never called")

    with pytest.raises(PaperError, match="runs the backtest's SignalEngine"):
        PaperSession(config, SYMBOL, engine=LooksLikeAnEngine())


def test_an_engine_built_for_another_symbol_is_refused(config):
    with pytest.raises(PaperError, match="is built for"):
        PaperSession(config, SYMBOL, engine=SignalEngine(config, "ES"))


def test_an_engine_with_a_different_calendar_is_refused(config):
    engine = SignalEngine(config, SYMBOL, calendar=TradingCalendar())
    with pytest.raises(PaperError, match="different session calendar"):
        PaperSession(config, SYMBOL, engine=engine, calendar=TradingCalendar())


def test_observing_a_bar_goes_through_engine_decide(config, short_dataset):
    """A spy on the instance, installed AFTER construction so the class-level
    override check is unaffected. It records the call and delegates, so what
    is proved is that `observe` produced its record by calling `decide()` --
    not that it could have."""
    store, calendar = short_dataset
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    clock = list(source.clock_ns())[:5]

    seen: list[tuple] = []
    real = session.engine.decide

    def spy(view, **kwargs):
        seen.append((view.now, kwargs.get("equity"), kwargs.get("limits")))
        return real(view, **kwargs)

    session.engine.decide = spy
    records = session.replay_feed(SlicedFeed(source, clock))

    assert len(seen) == len(records) == len(clock)
    assert [call[0] for call in seen] == [record.ts for record in records]


def test_the_module_imports_no_gate_scorer_or_feature_module():
    """There is nothing in this module that COULD re-derive a score.

    A parallel implementation would need the gates, the scorers, the setup
    selector, the regime detector or the feature layer. None of them is
    imported, which is a stronger statement than "it does not call them".
    """
    tree = ast.parse(Path(paper_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = (
        "flow_model.signals.gates",
        "flow_model.signals.setups",
        "flow_model.signals.flow_score",
        "flow_model.signals.scoring_bars",
        "flow_model.signals.scoring_flow",
        "flow_model.regime",
        "flow_model.regime.detector",
        "flow_model.features",
        "flow_model.risk.sizing",
        "flow_model.backtest",
    )
    for module in forbidden:
        assert not any(
            name == module or name.startswith(f"{module}.") for name in imported
        ), f"{module} is imported, so a second implementation is possible here"
    # The one signals module it may use is the engine.
    assert "flow_model.signals.engine" in imported


def test_the_only_engine_method_the_session_calls_is_decide():
    """`self.engine.<x>` across the whole file, read off the AST.

    `decide` is the pipeline entry point. `symbol`, `availability` and
    `calendar` are read for the construction-time contract checks. Anything
    else appearing here would be the paper path reaching into a stage.
    """
    tree = ast.parse(Path(paper_module.__file__).read_text(encoding="utf-8"))
    touched = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "engine"
    }
    assert "decide" in touched
    assert touched <= {"decide", "symbol", "availability", "calendar"}, (
        f"the session reaches further into the engine than decide(): {touched}"
    )


def test_the_paper_path_does_not_move_the_risk_state(config, short_dataset):
    """A paper session cannot mutate the risk state a real run would.

    The forbidden word list contains `position` and `trade`, so this module
    physically cannot name `on_position_opened` or `on_trade_closed`. That is
    a structural guarantee rather than a promise, and it is checked both ways:
    the words are forbidden, and the manager handed in comes back untouched.
    """
    assert "position" in FORBIDDEN_IDENTIFIER_WORDS
    assert "trade" in FORBIDDEN_IDENTIFIER_WORDS
    store, calendar = short_dataset
    equity = float(config.risk.starting_equity)
    limits = RiskLimitManager(config.risk, equity)

    def risk_state(manager) -> dict[str, object]:
        """Everything `on_position_opened` / `on_trade_closed` would move."""
        return {
            "killed": manager.killed,
            "consecutive_losses": manager.consecutive_losses,
            "open_positions": manager.open_positions(),
            "portfolio_heat": manager.portfolio_heat(),
            "trip_counts": dict(manager.trip_counts()),
        }

    before = risk_state(limits)
    session = PaperSession(
        config,
        SYMBOL,
        calendar=calendar,
        equity=equity,
        limits=limits,
        availability=QualityGrader(config.data, config.flow_score).availability(
            store.get(SYMBOL)
        ),
        dataset=store.get(SYMBOL).describe(),
    )
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    session.replay_feed(SlicedFeed(source, list(source.clock_ns())[:60]))
    assert risk_state(limits) == before
    assert limits.open_positions() == 0
    assert limits.portfolio_heat() == 0.0
    assert limits.killed is False


# ---------------------------------------------------------------------------
# 5. EVERY SECTION 12 FIELD, ON THE SIGNAL.
# ---------------------------------------------------------------------------

#: Section 12, verbatim: "Emits LONG/SHORT/WAIT + score + confidence +
#: entry/stop/target + R:R + regime + reasons + timestamp + data-quality
#: status." Mapped onto attribute names, as data, so the list cannot quietly
#: shrink.
SECTION_12_SURFACE: dict[str, tuple[str, ...]] = {
    "LONG/SHORT/WAIT": ("action", "side"),
    "score": ("flow_score",),
    "confidence": ("confidence",),
    "entry/stop/target": ("entry_price", "stop_price", "target_price"),
    "R:R": ("reward_risk", "achievable_reward_risk"),
    "regime": ("regime",),
    "reasons": ("reasons",),
    "timestamp": ("ts",),
    "data-quality status": (
        "data_quality",
        "overall_feed_quality",
        "feed_quality",
        "blocking_feeds",
        "quality_note",
        "available_points",
        "total_points",
        "incomputable_components",
    ),
}


def test_every_section_12_field_is_present_on_an_emitted_signal(emitted):
    """Present, not inferable. A reader must not have to deduce the status.

    On a WAIT the geometry fields are present and explicitly `None` -- which
    is a different statement from absent, and a different statement again
    from 0.0.
    """
    record = emitted[len(emitted) // 2]
    for item, attributes in SECTION_12_SURFACE.items():
        for attribute in attributes:
            assert hasattr(record, attribute), f"{item}: no {attribute!r}"
            getattr(record, attribute)
    assert isinstance(record.action, SignalAction)
    assert isinstance(record.regime, Regime)
    assert isinstance(record.data_quality, DataQuality)
    assert isinstance(record.ts, datetime)
    assert record.ts.tzinfo is not None
    assert isinstance(record.reasons, tuple) and record.reasons
    assert isinstance(record.confidence, float)
    assert record.symbol == SYMBOL


def test_the_projected_fields_agree_with_the_carried_signal(emitted):
    """`PaperSignal` carries the engine's `Signal` and projects it, so there
    is exactly one statement of each field. A property that drifted from the
    object it reads would make the parity test compare the wrong thing."""
    for record in emitted[::97]:
        assert record.symbol == record.signal.symbol
        assert record.ts == record.signal.ts
        assert record.action == record.signal.action
        assert record.side == record.signal.side
        assert record.setup == record.signal.setup
        assert record.flow_score == record.signal.flow_score
        assert record.confidence == record.signal.confidence
        assert record.regime == record.signal.regime
        assert record.data_quality == record.signal.data_quality
        assert record.wait_reason == record.signal.wait_reason
        assert record.wait_detail == record.signal.wait_detail
        assert record.reasons == record.signal.reasons
        assert record.component_points == record.signal.component_points


def test_the_data_quality_status_names_the_feeds_and_the_stage(emitted):
    """Section 12's "data-quality status" is per-feed, not one word."""
    blocked = [r for r in emitted if r.wait_reason is WaitReason.DATA_QUALITY]
    assert blocked, "no bar was blocked on data quality over this window"
    record = blocked[0]
    assert isinstance(record.overall_feed_quality, DataQuality)
    assert record.feed_quality, "no per-feed grades were carried"
    assert all(isinstance(feed, Feed) for feed in record.feed_quality)
    assert all(
        isinstance(status, DataQuality) for status in record.feed_quality.values()
    )
    assert record.blocking_feeds
    assert record.quality_note
    assert record.blocking_stage
    assert record.stages_run
    assert record.blocking_stage in record.stages_run


def test_the_wait_reason_and_its_detail_reach_the_reader(emitted):
    record = emitted[len(emitted) // 2]
    lines = record.summary_lines()
    assert lines
    assert any(record.action.value in line for line in lines)
    assert any("WAIT (" in line for line in lines)
    assert any("none -- no plan was made" in line for line in lines)
    assert any(record.regime.value in line for line in lines)


# --- the 55-of-100 a bars-only feed really has -----------------------------


@pytest.fixture(scope="module")
def bars_only(config, short_dataset):
    """The same bars with the tick, quote and options feeds removed.

    Order flow requires a classified tick feed and options flow requires an
    options snapshot; neither exists on a bars-only dataset, so STRUCTURE 20
    + LIQUIDITY 15 + VOL_MOMENTUM 20 = 55 of the 100 points remain. The
    fingerprint is carried over so the dataset still declares itself
    synthetic.
    """
    rich_store, calendar = short_dataset
    rich = rich_store.get(SYMBOL)
    store = DataStore(latency_seconds=config.data.data_latency_seconds)
    store.add(
        SymbolData(
            symbol=SYMBOL,
            primary_interval=rich.primary_interval,
            bars=dict(rich.bars),
            fingerprint=rich.fingerprint,
        )
    )
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    records = session.replay_feed(SlicedFeed(source, list(source.clock_ns())[:120]))
    return store, session, records


def test_a_bars_only_feed_declares_55_of_100_on_every_signal(bars_only):
    """The brief's number, measured and ON the signal rather than inferred.

    The data-quality gate stops a bars-only bar before the Flow Score stage,
    so the figure comes from the session's `AvailabilityReport` -- and the
    signal says which, in `availability_note`, so a reader is not left to
    guess whether 55 was measured at this bar or promised for the dataset.
    """
    store, _, records = bars_only
    data = store.get(SYMBOL)
    assert data.available_feeds() == frozenset({Feed.BARS})
    report = QualityGrader(
        default_config().data, default_config().flow_score
    ).availability(data)
    assert report.available_points == 55.0
    assert report.total_points == 100.0
    assert set(report.incomputable_components) == {
        Component.OPTIONS_FLOW,
        Component.ORDER_FLOW,
    }

    assert records
    for record in records:
        assert record.available_points == 55.0
        assert record.total_points == 100.0
        assert record.unavailable_points == 45.0
        assert set(record.incomputable_components) == {
            Component.OPTIONS_FLOW,
            Component.ORDER_FLOW,
        }
        assert record.availability_note
    # And it is in the words a reader sees, not only in the fields.
    lines = records[-1].summary_lines()
    assert any("55 of 100" in line for line in lines)
    assert any("options_flow" in line for line in lines)


def test_a_bars_only_feed_waits_and_says_which_feeds_blocked_it(bars_only):
    """The two absent feeds are named on every bar they block.

    The FIRST bar of a window additionally lists `bars`, because one bar of
    history grades the bar feed DEGRADED -- so the exact-set assertion is
    made from the second bar on, and the subset assertion everywhere. Writing
    it the other way round reported a defect that was not there.
    """
    _, _, records = bars_only
    blocked = [r for r in records if r.wait_reason is WaitReason.DATA_QUALITY]
    assert blocked
    structural = {Feed.TICK_AGGREGATE, Feed.OPTIONS_SNAPSHOT}
    for record in blocked:
        assert record.action is SignalAction.WAIT
        assert record.flow_score is None
        assert structural <= set(record.blocking_feeds)
        assert record.overall_feed_quality is DataQuality.MISSING
        assert record.feed_quality[Feed.TICK_AGGREGATE] is DataQuality.MISSING
        assert record.feed_quality[Feed.OPTIONS_SNAPSHOT] is DataQuality.MISSING
    assert set(blocked[0].blocking_feeds) == structural | {Feed.BARS}
    assert blocked[0].feed_quality[Feed.BARS] is DataQuality.DEGRADED
    for record in blocked[1:]:
        assert set(record.blocking_feeds) == structural
        assert record.feed_quality[Feed.BARS] is DataQuality.GOOD


def test_an_unmeasured_shortfall_is_declared_not_subtracted(config, short_dataset):
    """With no `AvailabilityReport` and no Flow Score stage, the shortfall is
    reported as unmeasured rather than as a full 100.

    `unavailable_points` returns None rather than a difference, because
    `100 - None` is the kind of arithmetic that turns an absence into a zero.
    """
    store, calendar = short_dataset
    engine = SignalEngine(config, SYMBOL, calendar=calendar)
    assert engine.availability is None
    session = PaperSession(config, SYMBOL, engine=engine, calendar=calendar)
    assert session.availability is None
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    records = session.replay_feed(SlicedFeed(source, list(source.clock_ns())[:30]))
    assert records
    for record in records:
        assert record.available_points is None
        assert record.total_points is None
        assert record.unavailable_points is None
        assert "not measured" in record.availability_note
        assert record.incomputable_components == ()


def test_a_scored_bar_reports_the_availability_of_that_bar(emitted):
    """Where the pipeline reached the Flow Score stage, the numbers are the
    ones the decision actually used, and the note says so."""
    scored = [r for r in emitted if r.flow_score is not None]
    assert scored
    for record in scored:
        assert record.availability_note == "from this bar's Flow Score stage"
        assert record.total_points == 100.0
        assert record.available_points == 100.0
        assert record.incomputable_components == ()
        assert record.unavailable_points == 0.0


def test_the_availability_report_must_be_for_this_symbol(config, short_dataset):
    store, calendar = short_dataset
    other = build_store(date(2016, 1, 4), date(2016, 1, 7), config)[0]
    grader = QualityGrader(config.data, config.flow_score)
    wrong = grader.availability(other.get(SYMBOL)).model_copy(update={"symbol": "ES"})
    with pytest.raises(PaperError, match="availability report is for"):
        PaperSession(config, SYMBOL, calendar=calendar, availability=wrong)
    assert store is not None


# --- where the bars came from ---------------------------------------------


def test_every_emitted_signal_carries_its_origin_warning(emitted):
    """No real market bar has ever entered this system, and every signal in
    this suite says so on its own face."""
    for record in emitted[::53]:
        assert record.dataset_synthetic is True
        assert record.dataset_source == "synthetic"
        assert record.origin_warning == SYNTHETIC_WARNING
        assert record.needs_origin_warning is True
        # The warning is the FIRST line a reader sees, before any number.
        assert record.summary_lines()[0] == SYNTHETIC_WARNING


def test_the_session_summary_carries_the_origin_warning_too(both_paths):
    _, session, _ = both_paths
    summary = session.summary()
    assert summary.origin_warning == SYNTHETIC_WARNING
    assert summary.summary_lines()[0] == SYNTHETIC_WARNING


@pytest.mark.parametrize(
    "source,synthetic,expected",
    [
        ("synthetic", True, SYNTHETIC_WARNING),
        ("databento", True, SYNTHETIC_WARNING),
        ("", None, UNDECLARED_WARNING),
        ("", False, UNDECLARED_WARNING),
        ("unknown", False, UNDECLARED_WARNING),
        ("unknown", None, UNDECLARED_WARNING),
        ("databento", None, UNDECLARED_WARNING),
        ("databento", False, ""),
    ],
)
def test_silence_about_the_origin_is_not_a_claim_of_real_bars(
    source, synthetic, expected
):
    """The default is suspicion. Only a dataset positively declaring
    `synthetic=False` from a named source gets no warning."""
    assert origin_warning(source, synthetic) == expected


@pytest.mark.parametrize(
    "source,synthetic",
    [
        ("synthetic", True),
        ("databento", True),
        ("", None),
        ("", False),
        ("unknown", False),
        ("unknown", None),
        ("databento", None),
        ("databento", False),
    ],
)
def test_the_paper_warning_agrees_with_the_dashboard_taxonomy(source, synthetic):
    """`dashboard.build.DataOrigin` is the authoritative taxonomy.

    The two read the same two fields of `SymbolData.describe()`, and this
    pins them to agree on WHEN a warning is owed. They are deliberately not
    one function -- importing the dashboard would put Plotly in this module's
    import surface -- so the agreement is tested instead of assumed.
    """
    dataset = {"source": source, "synthetic": synthetic}
    notice = origin_notice(dataset)
    owed = bool(origin_warning(source, synthetic))
    assert owed == (notice.origin is not DataOrigin.REAL)
    if notice.origin is DataOrigin.SYNTHETIC:
        assert origin_warning(source, synthetic) == SYNTHETIC_WARNING
    elif notice.origin is DataOrigin.UNDECLARED:
        assert origin_warning(source, synthetic) == UNDECLARED_WARNING
    else:
        assert origin_warning(source, synthetic) == ""


def test_a_signal_may_not_understate_where_its_bars_came_from():
    """The validator refuses a record whose warning disagrees with its own
    declared fields, so the warning cannot be edited off a synthetic signal."""
    wait = Signal(
        symbol=SYMBOL,
        ts=T0,
        action=SignalAction.WAIT,
        wait_reason=WaitReason.WARMUP,
    )
    with pytest.raises(PaperError, match="understate"):
        PaperSignal(
            signal=wait,
            dataset_source="synthetic",
            dataset_synthetic=True,
            origin_warning="",
        )
    with pytest.raises(PaperError, match="understate"):
        PaperSignal(signal=wait, dataset_source="", dataset_synthetic=None,
                    origin_warning="")
    ok = PaperSignal(
        signal=wait,
        dataset_source="synthetic",
        dataset_synthetic=True,
        origin_warning=SYNTHETIC_WARNING,
    )
    assert ok.needs_origin_warning is True


def test_the_default_origin_state_is_the_warning():
    """A record built with no declaration at all still warns."""
    wait = Signal(
        symbol=SYMBOL, ts=T0, action=SignalAction.WAIT, wait_reason=WaitReason.WARMUP
    )
    record = PaperSignal(signal=wait)
    assert record.origin_warning == UNDECLARED_WARNING
    assert record.needs_origin_warning is True
    session = PaperSession(default_config(), SYMBOL)
    assert session.origin_warning == UNDECLARED_WARNING


# ---------------------------------------------------------------------------
# 6. THE DIRECTIONAL BRANCH, AND SECTION 0.2's R.
# ---------------------------------------------------------------------------


def directional_signal(action: SignalAction = SignalAction.LONG) -> Signal:
    """A hand-built directional `Signal`.

    The real engine emits none at the committed config, and lowering a
    threshold to produce one is the move the brief forbids, so it is
    constructed. It is not a performance claim: what it tests is the
    validator's directional branch.
    """
    return Signal(
        symbol=SYMBOL,
        ts=T0,
        action=action,
        setup=SetupType.SETUP_2R,
        flow_score=72.5,
        confidence=0.8,
        regime=Regime.TRENDING_UP,
        data_quality=DataQuality.GOOD,
        reasons=("hand-built for the directional branch",),
    )


def full_plan_fields() -> dict[str, object]:
    """Entry 18000, stop 17990, target 18020, size 2, point value 20.

    R = |18000 - 17990| * 20 * 2 = 400.0, which is section 0.2's PLANNED unit.
    """
    return {
        "entry_price": 18000.0,
        "stop_price": 17990.0,
        "target_price": 18020.0,
        "reward_risk": 2.0,
        "achievable_reward_risk": 2.4,
        "size": 2.0,
        "planned_risk_dollars": 400.0,
        "point_value": 20.0,
        "max_hold_bars": 24,
    }


def test_a_directional_signal_must_carry_its_whole_plan():
    """A partial plan cannot be read as a trade idea."""
    full = full_plan_fields()
    record = PaperSignal(signal=directional_signal(), **full)
    assert record.action is SignalAction.LONG
    assert record.side is Side.LONG
    assert record.reward_risk == 2.0
    for omitted in (
        "entry_price",
        "stop_price",
        "target_price",
        "reward_risk",
        "size",
        "planned_risk_dollars",
        "point_value",
    ):
        partial = {k: v for k, v in full.items() if k != omitted}
        with pytest.raises(PaperError, match="must carry the whole plan"):
            PaperSignal(signal=directional_signal(), **partial)


def test_a_wait_may_not_carry_any_part_of_a_plan():
    """Reporting 0.0 for a geometry that was never planned reads as a
    measurement, so the validator refuses the field at all, not just zero."""
    wait = Signal(
        symbol=SYMBOL, ts=T0, action=SignalAction.WAIT, wait_reason=WaitReason.WARMUP
    )
    assert PaperSignal(signal=wait).entry_price is None
    for field, value in full_plan_fields().items():
        if field == "achievable_reward_risk" or field == "max_hold_bars":
            continue
        with pytest.raises(PaperError, match="a WAIT carries no entry"):
            PaperSignal(signal=wait, **{field: value})


def test_a_geometry_of_zero_is_refused_outright():
    """`gt=0.0` on every price, so 0.0 cannot be stored even on a LONG."""
    full = full_plan_fields()
    for field in ("entry_price", "stop_price", "target_price", "size", "point_value"):
        with pytest.raises(Exception):
            PaperSignal(signal=directional_signal(), **{**full, field: 0.0})


def test_the_planned_risk_is_section_0_2s_planned_unit():
    """R = |planned_entry - stop_price| * point_value * size.

    Checked against the contract that owns the identity rather than against a
    second copy of the formula: `EntryPlan.to_trade_intent()` goes through
    `TradeIntent`, whose validator raises if `risk_dollars` disagrees.
    """
    plan = EntryPlan(
        symbol=SYMBOL,
        signal_ts=T0,
        side=Side.LONG,
        setup=SetupType.SETUP_2R,
        entry_price=18000.0,
        stop_price=17990.0,
        target_price=18020.0,
        size=2.0,
        risk_dollars=400.0,
        point_value=20.0,
        equity_at_signal=100000.0,
        stop_distance=10.0,
        stop_ticks=40.0,
        planned_r_multiple=2.0,
        achievable_rr=2.4,
        max_hold_bars=24,
        flow_score=72.5,
        regime=Regime.TRENDING_UP,
    )
    intent = plan.to_trade_intent()
    assert intent.risk_dollars == 400.0
    assert intent.risk_dollars == pytest.approx(
        abs(plan.entry_price - plan.stop_price) * plan.point_value * plan.size
    )
    record = PaperSignal(signal=directional_signal(), **full_plan_fields())
    assert record.planned_risk_dollars == plan.risk_dollars


def test_the_planned_risk_is_carried_not_recomputed():
    """There is no third definition of R in this module.

    `EntryPlan` validates its geometry but not the R identity (`TradeIntent`
    owns that), so a plan can be built whose `risk_dollars` is NOT the
    product of its own fields. If `paper.py` recomputed R from entry, stop,
    size and point value, it would overwrite that number; carrying it means
    the odd value survives, which is how this test tells the two apart.
    """
    record = PaperSignal(
        signal=directional_signal(),
        **{**full_plan_fields(), "planned_risk_dollars": 777.0},
    )
    recomputed = abs(18000.0 - 17990.0) * 20.0 * 2.0
    assert recomputed == 400.0
    assert record.planned_risk_dollars == 777.0, (
        "the module recomputed R instead of carrying the planned unit"
    )


def test_a_short_plan_is_accepted_in_the_other_direction():
    record = PaperSignal(
        signal=directional_signal(SignalAction.SHORT),
        entry_price=18000.0,
        stop_price=18010.0,
        target_price=17980.0,
        reward_risk=2.0,
        achievable_reward_risk=2.1,
        size=2.0,
        planned_risk_dollars=400.0,
        point_value=20.0,
        max_hold_bars=24,
    )
    assert record.side is Side.SHORT
    lines = record.summary_lines()
    assert any("entry=18000.0" in line for line in lines)
    assert any("planned risk=400.0" in line for line in lines)
    assert any("section 0.2 planned unit" in line for line in lines)


# ---------------------------------------------------------------------------
# 7. POINT-IN-TIME. A paper feed may not re-read a bar it has passed.
# ---------------------------------------------------------------------------


def test_a_repeated_bar_is_refused(config, short_dataset):
    store, calendar = short_dataset
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    clock = list(source.clock_ns())
    source.advance(clock[10])
    view = source.view()
    session.observe(view)
    with pytest.raises(PaperError, match="already observed"):
        session.observe(view)


def test_a_rewound_bar_is_refused(config, short_dataset):
    """Re-reading a passed bar would let a later signal be computed from an
    earlier instant, which is lookahead's mirror image."""
    store, calendar = short_dataset
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    clock = list(source.clock_ns())
    source.advance(clock[10])
    session.observe(source.view())
    earlier = StoreMarketSource.from_config(store, SYMBOL, config)
    earlier.advance(clock[4])
    with pytest.raises(PaperError, match="point-in-time"):
        session.observe(earlier.view())
    assert len(session.signals) == 1


def test_a_view_for_another_symbol_is_refused(config, short_dataset):
    store, calendar = short_dataset
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)

    class ForeignView:
        symbol = "ES"

    with pytest.raises(PaperError, match="was handed a view for"):
        session.observe(ForeignView())


def test_signals_accumulate_in_clock_order_and_replay_returns_its_own_slice(
    config, short_dataset
):
    store, calendar = short_dataset
    session = PaperSession.from_store(config, SYMBOL, store, calendar=calendar)
    source = StoreMarketSource.from_config(store, SYMBOL, config)
    clock = list(source.clock_ns())
    first = session.replay_feed(SlicedFeed(source, clock[:20]))
    second = session.replay_feed(SlicedFeed(source, clock[20:40]))
    assert len(first) == len(second) == 20
    assert session.signals == first + second
    stamps = [record.ts for record in session.signals]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    assert session.summary().bars_observed == 40
