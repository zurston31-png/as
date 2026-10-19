"""Tests for `validation/walk_forward.py` -- the Phase 8 gate.

The module was committed untested, so nothing here trusts its docstrings. Each
claim it makes is either reproduced from the dates on paper or shown to be the
behaviour of the code.

WHAT THE GATE ASKS, AND WHERE IT IS

* purge and embargo verified AND proved distinct -- `TestPurge`,
  `TestEmbargo`, `TestPurgeIsNotEmbargo`.
* the gap's boundary arithmetic, inclusive vs exclusive -- `TestDayUnit`,
  which rebuilds every boundary from the half-open interval and the purge's
  stated meaning ("how long a position can stay open") rather than from the
  module's own arithmetic.
* folds reported separately, with the mechanism asserted and shown
  non-vacuous -- `TestNoAggregateOnlyOutput`.
* the seal: a refused read raises AND spends none of the one-open budget --
  `TestTheSeal`.
* n=0 folds: undefined, not 0.0, with the fold count still reported --
  `TestZeroTradeFolds`. This is the live case (PROVENANCE.md 2.4: 283 of
  19542 bars reach the scorer and none clears the threshold), so it is
  tested first in the file's logic and most heavily.
* rolling vs anchored with explicit boundary dates -- `TestRollingVsAnchored`,
  which writes the dates out rather than recomputing them.
* determinism -- `TestDeterminism`, including across processes with different
  `PYTHONHASHSEED`.

Every config is built in the test. `conftest.config` reads the committed
`defaults.yaml` and is used only where the committed numbers are the subject
(`TestCommittedConfig`); nothing here writes a config value.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import typing
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from flow_model.analytics.metrics import (
    MetricPolicy,
    MetricValue,
    SampleStatus,
    SplitLabelMismatchError,
    Undefined,
    performance_summary,
    sample_status,
)
from flow_model.backtest.loop import RunProvenance, RunResult
from flow_model.core.contracts import EquityPoint
from flow_model.core.enums import SplitPhase
from flow_model.validation.experiment_log import ExperimentLog
from flow_model.validation.splits import (
    DataAccessGuard,
    DateRange,
    SealAudit,
    SealedDataAccessError,
    Split,
    SplitRegistry,
)
from flow_model.validation.walk_forward import (
    FOLD_PHASES,
    MIN_FOLDS_FOR_DISPERSION,
    CrossFoldStatement,
    EmbargoViolationError,
    Fold,
    FoldMode,
    FoldResult,
    FoldSet,
    FoldWindowResult,
    InsufficientRangeError,
    PurgeViolationError,
    WalkForwardError,
    WalkForwardReport,
    WindowSpec,
    evaluate_folds,
    generate_folds,
)

# ---------------------------------------------------------------------------
# Dates and helpers
# ---------------------------------------------------------------------------

#: ARCHITECTURE.md 0.6 / config `seal.*`. Named here so a test can say "the
#: seal" without reading the committed config, and so a typo in one of the two
#: is a failure rather than a silently different window.
SEALED = DateRange(start=date(2023, 1, 1), end=date(2025, 1, 1))

#: The research window. Everything before the seal.
RESEARCH = DateRange(start=date(2015, 1, 1), end=date(2023, 1, 1))

#: Rates and ratios: undefined at n=0, never 0.0.
RATE_AND_RATIO_METRICS = (
    "win_rate",
    "loss_rate",
    "scratch_rate",
    "expectancy",
    "expectancy_r",
    "mean_r",
    "median_r",
    "average_win_r",
    "average_loss_r",
    "win_loss_ratio",
    "profit_factor",
    "breakeven_win_rate",
    "mean_cost_drag_r",
    "mean_mae_r",
    "mean_mfe_r",
    "mean_holding_seconds",
    "max_consecutive_losses",
)

#: Counts and sums: 0.0 over no trades IS a measurement, and the flat equity
#: curve says so independently (`metrics.total_pnl`).
COUNT_AND_SUM_METRICS = (
    "trade_count",
    "total_pnl",
    "gross_profit",
    "gross_loss",
    "total_costs",
)

AGGREGATES = ("mean", "stdev", "spread", "minimum", "maximum")


def shift(day: date, days: int) -> date:
    """Day arithmetic written out, so a test never borrows the module's."""
    return date.fromordinal(day.toordinal() + days)


def rng(start: date, end: date) -> DateRange:
    return DateRange(start=start, end=end)


def make_spec(**changes) -> WindowSpec:
    """A small, readable geometry. Day counts chosen so the dates are legible.

    train 30 / validation 10 / test 10 / step 10, purge 1 + embargo 2 = gap 3,
    span 56. Built in the test, never read from a file.
    """
    fields = {
        "train_days": 30,
        "validation_days": 10,
        "test_days": 10,
        "step_days": 10,
        "purge_days": 1,
        "embargo_days": 2,
        "min_trades_per_fold": 0,
    }
    fields.update(changes)
    return WindowSpec(**fields)


def mark(day: date) -> EquityPoint:
    """One flat equity sample, stamped midday UTC on `day`."""
    return EquityPoint(
        ts=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        + timedelta(hours=12),
        equity=100_000.0,
        realized_equity=100_000.0,
        high_water_mark=100_000.0,
    )


def make_fold(
    *,
    train: tuple[date, date],
    validation: tuple[date, date],
    test: tuple[date, date],
    purge_days: int,
    embargo_days: int,
    index: int = 0,
    mode: FoldMode = FoldMode.ROLLING,
) -> Fold:
    """A hand-built fold -- the shape `Fold._check` exists to refuse."""
    return Fold(
        index=index,
        mode=mode,
        train=rng(*train),
        validation=rng(*validation),
        test=rng(*test),
        purge_days=purge_days,
        embargo_days=embargo_days,
    )


@pytest.fixture
def open_guard(tmp_path) -> DataAccessGuard:
    """A guard with no seal, for geometry tests that never approach it."""
    return DataAccessGuard(
        sealed=None, audit=SealAudit(tmp_path / "no_seal_audit.jsonl"), enabled=False
    )


@pytest.fixture
def audit_path(tmp_path):
    return tmp_path / "seal_audit.jsonl"


@pytest.fixture
def sealed_guard(audit_path) -> DataAccessGuard:
    """The committed seal shape: one open, audited, enabled."""
    return DataAccessGuard(
        sealed=SEALED, audit=SealAudit(audit_path), max_opens=1, enabled=True
    )


class Cfg:
    """The smallest thing `evaluate_folds` accepts as a config."""

    config_hash = "cfg0000"


class FakeRunner:
    """A `Runner` that returns whatever the test says, and counts its calls.

    Takes the place of `robustness.StoreRunner` so the n=0 path -- the live
    one -- is testable without a data store, which is the reason the module
    injects a runner at all.
    """

    def __init__(self, plan=None, *, phase=SplitPhase.TEST, run_id=None, equity=False):
        self.plan = plan or {}
        self.phase = phase
        self.forced_run_id = run_id
        self.equity = equity
        self.calls: list[str] = []

    def trades_for(self, split: Split, make_trade):
        index = int(split.name.split("_")[1])
        plan = self.plan.get(index, ()) if split.phase is self.phase else ()
        base = datetime.combine(split.range.start, datetime.min.time(), tzinfo=timezone.utc)
        out = []
        for offset, r_multiple in enumerate(plan):
            entry, stop = 18000.0, 17980.0
            out.append(
                make_trade(
                    trade_id=f"{split.name}_{offset}",
                    entry=entry,
                    stop=stop,
                    exit_price=entry + (entry - stop) * r_multiple,
                    signal_ts=base + timedelta(days=offset, hours=10),
                    split_label=split.phase,
                )
            )
        return tuple(out)

    def equity_for(self, split: Split):
        """Two points inside the window: its first day and its last."""
        if not self.equity:
            return ()
        return tuple(
            mark(day) for day in (split.range.start, shift(split.range.end, -1))
        )

    def bind(self, make_trade):
        def run(config, split: Split) -> RunResult:
            self.calls.append(split.name)
            trades = self.trades_for(split, make_trade)
            equity = self.equity_for(split)
            return RunResult(
                provenance=RunProvenance(
                    run_id=self.forced_run_id or f"run-{split.name}",
                    symbol="NQ",
                    config_hash=Cfg.config_hash,
                    data_hash=f"data-{split.name}",
                    seed=20240101,
                    split_label=split.phase,
                    split_name=split.name,
                ),
                trades=trades,
                equity=equity,
            )

        return run


@pytest.fixture
def empty_folds(open_guard) -> FoldSet:
    """Four folds over 2016 Q1. Explicit so every later date is checkable."""
    return generate_folds(
        rng(date(2016, 1, 1), date(2016, 4, 1)), make_spec(), guard=open_guard
    )


def report_over(folds: FoldSet, runner: FakeRunner, make_trade, **kwargs) -> WalkForwardReport:
    return evaluate_folds(folds, Cfg(), runner.bind(make_trade), **kwargs)


# ---------------------------------------------------------------------------
# The unit is the calendar day, and the interval is half-open
# ---------------------------------------------------------------------------


class TestDayUnit:
    """Boundary arithmetic, rebuilt from the half-open interval."""

    def test_a_window_holds_exactly_its_configured_day_count(self, empty_folds):
        spec = empty_folds.spec
        for fold in empty_folds:
            assert fold.train.days == spec.train_days
            assert fold.validation.days == spec.validation_days
            assert fold.test.days == spec.test_days

    def test_the_end_date_is_exclusive(self, empty_folds):
        fold = empty_folds[0]
        assert fold.train.contains(fold.train.start)
        assert fold.train.contains(shift(fold.train.end, -1))
        assert not fold.train.contains(fold.train.end)
        # Which is why the day count is a plain subtraction.
        assert (fold.train.end - fold.train.start).days == fold.train.days

    def test_the_gap_is_the_count_of_days_in_neither_window(self, empty_folds):
        spec = empty_folds.spec
        for fold in empty_folds:
            for earlier, later in ((fold.train, fold.validation), (fold.validation, fold.test)):
                excluded = [
                    shift(earlier.end, offset)
                    for offset in range((later.start - earlier.end).days)
                ]
                assert len(excluded) == spec.gap_days
                for day in excluded:
                    assert not earlier.contains(day)
                    assert not later.contains(day)

    def test_the_gap_is_purge_plus_embargo(self, empty_folds):
        spec = empty_folds.spec
        assert spec.gap_days == spec.purge_days + spec.embargo_days
        for fold in empty_folds:
            assert (fold.validation.start - fold.train.end).days == spec.gap_days
            assert (fold.test.start - fold.validation.end).days == spec.gap_days

    def test_a_gap_of_exactly_n_satisfies_a_requirement_of_n(self):
        """The off-by-one that would be a leak, checked in both directions."""
        purge, embargo = 1, 5
        required = purge + embargo
        train = (date(2016, 1, 1), date(2016, 2, 1))
        validation_start = shift(train[1], required)
        accepted = make_fold(
            train=train,
            validation=(validation_start, shift(validation_start, 10)),
            test=(
                shift(validation_start, 10 + required),
                shift(validation_start, 20 + required),
            ),
            purge_days=purge,
            embargo_days=embargo,
        )
        assert (accepted.validation.start - accepted.train.end).days == required

        one_short = shift(train[1], required - 1)
        with pytest.raises(EmbargoViolationError):
            make_fold(
                train=train,
                validation=(one_short, shift(one_short, 10)),
                test=(shift(one_short, 10 + required), shift(one_short, 20 + required)),
                purge_days=purge,
                embargo_days=embargo,
            )

    def test_splits_assert_embargo_agrees_on_the_exactly_n_boundary(self):
        """`splits.assert_embargo` is the second, independent check. If the two
        disagreed about whether N days clears N, one of them would be the leak."""
        registry = SplitRegistry(
            splits=(
                Split(
                    name="train",
                    phase=SplitPhase.TRAIN,
                    range=rng(date(2016, 1, 1), date(2016, 2, 1)),
                ),
                Split(
                    name="validation",
                    phase=SplitPhase.VALIDATION,
                    range=rng(date(2016, 2, 7), date(2016, 3, 1)),
                ),
            )
        )
        assert (date(2016, 2, 7) - date(2016, 2, 1)).days == 6
        registry.assert_embargo(6)
        with pytest.raises(ValueError, match="embargo violated"):
            registry.assert_embargo(7)

    def test_the_folds_two_checks_agree_on_exactly_n(self):
        """The same boundary through `Fold`, which is the route Phase 8 uses.

        `Fold._check` enforces the gap at construction and `Fold.assert_embargo`
        re-runs it through `splits.SplitRegistry`. A fold accepted by the first
        and refused by the second at the same gap would mean one of the two is
        the leak, and nothing downstream would say which.
        """
        purge, embargo = 1, 5
        required = purge + embargo
        train = (date(2016, 1, 1), date(2016, 2, 1))
        validation_start = shift(train[1], required)
        fold = make_fold(
            train=train,
            validation=(validation_start, shift(validation_start, 10)),
            test=(
                shift(validation_start, 10 + required),
                shift(validation_start, 20 + required),
            ),
            purge_days=purge,
            embargo_days=embargo,
        )
        assert (fold.validation.start - fold.train.end).days == required
        fold.assert_embargo(required)  # exactly N clears a requirement of N
        with pytest.raises(EmbargoViolationError, match="a gap of 6"):
            fold.assert_embargo(required + 1)

    def test_span_is_train_start_to_test_end_and_excludes_the_trailing_embargo(
        self, empty_folds
    ):
        fold = empty_folds[0]
        assert fold.span.start == fold.train.start
        assert fold.span.end == fold.test.end
        assert fold.span.end == fold.embargo_after_test.start
        assert empty_folds.spec.span_days == fold.span.days

    def test_months_resolve_to_days_by_calendar_arithmetic_only(self):
        """`WindowSpec` has no days-per-month constant; the only route from
        months to days is `from_config`, and it must land on real calendars."""
        class MonthConfig:
            class walk_forward:
                train_months = 1
                validation_months = 1
                test_months = 1
                step_months = 1
                anchored = False
                embargo_days = 0
                purge_days = 0
                min_trades_per_fold = 0

        # 2016-01-31 + 1 month clamps into February rather than gaining a day.
        spec = WindowSpec.from_config(MonthConfig, date(2016, 1, 31))
        assert spec.train_days == (date(2016, 2, 29) - date(2016, 1, 31)).days == 29


# ---------------------------------------------------------------------------
# PURGE
# ---------------------------------------------------------------------------


class TestPurge:
    """The purge removes the end of the EARLIER window.

    The module states the purge's meaning as "how long a position can stay
    open". These tests take that literally: a trade opened on the last day the
    earlier window holds, and still open `purge_days` later, must have its
    whole outcome window finished before the later window begins.
    """

    PURGE = 3

    def layout(self, validation_start: date) -> dict:
        train = (date(2016, 1, 1), date(2016, 2, 1))
        return {
            "train": train,
            "validation": (validation_start, shift(validation_start, 10)),
            "test": (shift(validation_start, 13), shift(validation_start, 23)),
            "purge_days": self.PURGE,
            "embargo_days": 0,
        }

    def test_a_trade_open_from_the_end_of_train_into_the_later_window_is_refused(self):
        train_end = date(2016, 2, 1)
        last_train_day = shift(train_end, -1)
        assert last_train_day == date(2016, 1, 31)
        # Opened on the last training day, still open PURGE days later: its
        # outcome is settled on this day, by prices from whatever window this
        # day belongs to.
        outcome_day = shift(last_train_day, self.PURGE)
        assert outcome_day == date(2016, 2, 3)

        leaky = self.layout(outcome_day)
        with pytest.raises(PurgeViolationError) as exc:
            make_fold(**leaky)
        # The leak, named: the trade's outcome day is inside the later window.
        assert rng(*leaky["validation"]).contains(outcome_day)
        assert "purge_days=3" in str(exc.value)

    def test_the_purge_removes_it_when_the_gap_covers_the_whole_outcome_window(self):
        outcome_day = shift(shift(date(2016, 2, 1), -1), self.PURGE)
        clean = self.layout(shift(outcome_day, 1))
        fold = make_fold(**clean)
        assert not fold.validation.contains(outcome_day)
        assert (fold.validation.start - fold.train.end).days == self.PURGE

    def test_every_generated_fold_clears_the_purge_at_both_interior_boundaries(
        self, open_guard
    ):
        spec = make_spec(purge_days=4, embargo_days=0)
        folds = generate_folds(
            rng(date(2016, 1, 1), date(2016, 6, 1)), spec, guard=open_guard
        )
        assert len(folds) > 1
        for fold in folds:
            last_train_day = shift(fold.train.end, -1)
            assert not fold.validation.contains(shift(last_train_day, spec.purge_days))
            last_validation_day = shift(fold.validation.end, -1)
            assert not fold.test.contains(shift(last_validation_day, spec.purge_days))

    def test_a_trade_from_the_end_of_train_cannot_be_open_inside_test(self, open_guard):
        """The gate names the TEST window, and that is the longer reach.

        `_check_gap` runs on adjacent boundaries, so the purge is enforced at
        train -> validation. Test is further away by validation_days plus a
        second gap, so stating the test case separately is stating that the
        nearer boundary is the binding one -- and if the gaps were ever
        rearranged so test came nearer, this is what would catch it.
        """
        spec = make_spec(purge_days=4, embargo_days=2)
        folds = generate_folds(
            rng(date(2016, 1, 1), date(2016, 6, 1)), spec, guard=open_guard
        )
        assert len(folds) > 1
        for fold in folds:
            outcome_day = shift(shift(fold.train.end, -1), spec.purge_days)
            assert not fold.validation.contains(outcome_day)
            assert not fold.test.contains(outcome_day)
            assert outcome_day < fold.validation.start <= fold.test.start
            # The nearer boundary is the one the purge has to clear.
            assert (fold.validation.start - fold.train.end).days < (
                fold.test.start - fold.train.end
            ).days

    def test_a_negative_gap_is_a_purge_violation_even_at_purge_zero(self):
        """Overlapping windows are the shape a sign error takes, and they must
        not slip through on a spec that asks for no purge at all."""
        with pytest.raises(PurgeViolationError):
            make_fold(
                train=(date(2016, 1, 1), date(2016, 3, 1)),
                validation=(date(2016, 2, 1), date(2016, 2, 11)),
                test=(date(2016, 3, 1), date(2016, 3, 11)),
                purge_days=0,
                embargo_days=0,
            )


# ---------------------------------------------------------------------------
# EMBARGO
# ---------------------------------------------------------------------------


class TestEmbargo:
    """The embargo is a dead band that FOLLOWS an evaluation window."""

    def test_the_dead_band_begins_at_the_test_windows_exclusive_end(self, empty_folds):
        fold = empty_folds[0]
        shadow = fold.embargo_after_test
        assert shadow == rng(fold.test.end, shift(fold.test.end, fold.embargo_days))
        assert shadow.days == fold.embargo_days
        assert not fold.test.contains(shadow.start)

    def test_a_zero_width_embargo_is_absent_rather_than_degenerate(self, open_guard):
        folds = generate_folds(
            rng(date(2016, 1, 1), date(2016, 6, 1)),
            make_spec(purge_days=1, embargo_days=0),
            guard=open_guard,
        )
        assert folds[0].embargo_after_test is None

    def test_a_gap_that_clears_the_purge_but_not_the_embargo_is_refused(self):
        purge, embargo = 1, 5
        train = (date(2016, 1, 1), date(2016, 2, 1))
        validation_start = shift(train[1], 3)
        assert 3 >= purge  # the purge fits
        assert 3 < purge + embargo  # the embargo does not
        with pytest.raises(EmbargoViolationError) as exc:
            make_fold(
                train=train,
                validation=(validation_start, shift(validation_start, 10)),
                test=(shift(validation_start, 16), shift(validation_start, 26)),
                purge_days=purge,
                embargo_days=embargo,
            )
        assert "6 required" in str(exc.value)

    def test_a_later_fold_may_not_begin_training_inside_an_earlier_tests_shadow(self):
        spec = make_spec(train_days=10, purge_days=0, embargo_days=3)
        first = make_fold(
            index=0,
            train=(date(2016, 1, 1), date(2016, 1, 11)),
            validation=(date(2016, 1, 14), date(2016, 1, 24)),
            test=(date(2016, 1, 27), date(2016, 2, 6)),
            purge_days=0,
            embargo_days=3,
        )
        assert first.embargo_after_test == rng(date(2016, 2, 6), date(2016, 2, 9))

        def second(train_start: date) -> Fold:
            return make_fold(
                index=1,
                train=(train_start, shift(train_start, 10)),
                validation=(shift(train_start, 13), shift(train_start, 23)),
                test=(shift(train_start, 26), shift(train_start, 36)),
                purge_days=0,
                embargo_days=3,
            )

        inside = second(date(2016, 2, 7))
        assert first.embargo_after_test.contains(inside.train.start)
        with pytest.raises(EmbargoViolationError, match="inside"):
            FoldSet(
                spec=spec,
                data_range=rng(date(2016, 1, 1), date(2016, 4, 1)),
                folds=(first, inside),
            )

        # One day past the shadow: the boundary is exclusive, so this is fine.
        outside = second(date(2016, 2, 9))
        assert not first.embargo_after_test.contains(outside.train.start)
        FoldSet(
            spec=spec,
            data_range=rng(date(2016, 1, 1), date(2016, 4, 1)),
            folds=(first, outside),
        )

    def test_reusing_an_earlier_test_window_as_later_training_data_is_allowed(
        self, open_guard
    ):
        """A rolling walk-forward does exactly this, and `splits.py` says so.
        The rule is about where a train window BEGINS, not what it spans."""
        folds = generate_folds(
            rng(date(2016, 1, 1), date(2016, 6, 1)), make_spec(), guard=open_guard
        )
        overlaps = [
            (later.index, earlier.index)
            for earlier in folds
            for later in folds[earlier.index + 1 :]
            if later.train.overlaps(earlier.test) or later.validation.overlaps(earlier.test)
        ]
        assert overlaps, "the geometry under test does not exercise the reuse case"

    def test_fold_assert_embargo_also_covers_the_boundary_to_the_seal(self):
        """Phase 8's job per section 10: call `splits.assert_embargo` itself."""
        def fold_ending(test_end: date) -> Fold:
            test_start = shift(test_end, -90)
            validation_end = shift(test_start, -6)
            validation_start = shift(validation_end, -90)
            train_end = shift(validation_start, -6)
            return make_fold(
                train=(shift(train_end, -365), train_end),
                validation=(validation_start, validation_end),
                test=(test_start, test_end),
                purge_days=1,
                embargo_days=5,
            )

        clear = fold_ending(shift(SEALED.start, -6))
        clear.assert_embargo(6, sealed=SEALED)

        too_close = fold_ending(shift(SEALED.start, -5))
        with pytest.raises(EmbargoViolationError, match="embargo violated"):
            too_close.assert_embargo(6, sealed=SEALED)

    def test_a_test_window_overlapping_the_seal_is_not_called_an_embargo_problem(self):
        """A seal leak reported as a short embargo tells the reader to widen the
        embargo when what they did was reach into the holdout. The registry owns
        the seal and keeps its own words."""
        fold = make_fold(
            train=(date(2021, 1, 1), date(2022, 1, 1)),
            validation=(date(2022, 1, 7), date(2022, 6, 1)),
            test=(date(2022, 6, 7), date(2023, 2, 1)),
            purge_days=1,
            embargo_days=5,
        )
        assert fold.test.overlaps(SEALED)
        with pytest.raises(ValidationError, match="SEALED_OOS") as exc:
            fold.assert_embargo(6, sealed=SEALED)
        assert not isinstance(exc.value, EmbargoViolationError)


# ---------------------------------------------------------------------------
# PURGE IS NOT EMBARGO
# ---------------------------------------------------------------------------


class TestPurgeIsNotEmbargo:
    """Conflating the two is the classic bug, so the types must differ."""

    def test_the_two_error_types_are_unrelated(self):
        assert PurgeViolationError is not EmbargoViolationError
        assert not issubclass(PurgeViolationError, EmbargoViolationError)
        assert not issubclass(EmbargoViolationError, PurgeViolationError)
        assert issubclass(PurgeViolationError, WalkForwardError)
        assert issubclass(EmbargoViolationError, WalkForwardError)

    @staticmethod
    def fold_with_gap(gap: int, purge: int, embargo: int) -> Fold:
        """One shape, one knob: only the train -> validation gap moves."""
        train = (date(2016, 1, 1), date(2016, 2, 1))
        validation_start = shift(train[1], gap)
        required = purge + embargo
        return make_fold(
            train=train,
            validation=(validation_start, shift(validation_start, 10)),
            test=(
                shift(validation_start, 10 + required),
                shift(validation_start, 20 + required),
            ),
            purge_days=purge,
            embargo_days=embargo,
        )

    def test_short_of_the_purge_raises_purge_and_not_embargo(self):
        with pytest.raises(PurgeViolationError) as exc:
            self.fold_with_gap(gap=0, purge=2, embargo=5)
        assert not isinstance(exc.value, EmbargoViolationError)

    def test_short_of_only_the_embargo_raises_embargo_and_not_purge(self):
        with pytest.raises(EmbargoViolationError) as exc:
            self.fold_with_gap(gap=3, purge=1, embargo=5)
        assert not isinstance(exc.value, PurgeViolationError)

    def test_the_same_shape_passes_once_the_gap_is_long_enough(self):
        """Non-vacuity for the two tests above: the shape itself is fine."""
        fold = self.fold_with_gap(gap=6, purge=1, embargo=5)
        assert (fold.validation.start - fold.train.end).days == 6

    def test_satisfying_the_embargo_but_not_the_purge_is_still_refused(self):
        """The classic bug would check `gap >= embargo_days` and stop.

        purge 10, embargo 2, gap 3: the gap is longer than the embargo, so a
        conflated check passes it, and a position opened at the end of train
        is still open seven days into the later window.
        """
        purge, embargo, gap = 10, 2, 3
        assert gap >= embargo
        assert gap < purge
        with pytest.raises(PurgeViolationError) as exc:
            self.fold_with_gap(gap=gap, purge=purge, embargo=embargo)
        assert "purge_days=10" in str(exc.value)
        # And the leak the conflated check would have admitted:
        leaked = self.fold_with_gap(gap=purge + embargo, purge=purge, embargo=embargo)
        last_train_day = shift(leaked.train.end, -1)
        assert not leaked.validation.contains(shift(last_train_day, purge))

    def test_satisfying_the_purge_but_not_the_embargo_is_also_refused(self):
        """The mirror image, so neither check is doing both jobs."""
        with pytest.raises(EmbargoViolationError):
            self.fold_with_gap(gap=10, purge=10, embargo=2)

    def test_the_embargo_check_on_its_own_accepts_what_the_purge_refuses(self):
        """The classic bug, run as the bug: only the embargo check is applied.

        `splits.SplitRegistry.assert_embargo` is the embargo check in the
        module that owns the seal, and it takes ONE number. Handed the embargo
        alone it passes this layout; `Fold` refuses the same dates. If the two
        were one check, this layout would be in a fold set.
        """
        purge, embargo, gap = 10, 2, 3
        train = rng(date(2016, 1, 1), date(2016, 2, 1))
        validation = rng(shift(train.end, gap), shift(train.end, gap + 10))
        embargo_only = SplitRegistry(
            splits=(
                Split(name="train", phase=SplitPhase.TRAIN, range=train),
                Split(name="validation", phase=SplitPhase.VALIDATION, range=validation),
            )
        )
        embargo_only.assert_embargo(embargo)  # the conflated check: passes
        # `splits` raises its own `ValueError`; only `Fold.assert_embargo`
        # relabels it, which is why the purge/embargo types live in this module.
        with pytest.raises(ValueError, match="embargo violated"):
            embargo_only.assert_embargo(purge + embargo)  # the real requirement
        with pytest.raises(PurgeViolationError, match="purge_days=10"):
            self.fold_with_gap(gap=gap, purge=purge, embargo=embargo)

    def test_the_generator_refuses_a_later_train_start_inside_an_earlier_shadow(
        self, open_guard
    ):
        """The cross-fold embargo is live inside `generate_folds`, not only on
        hand-built sets, and whether it bites is pure arithmetic.

        train 30 / validation 10 / test 10, purge 1 + embargo 9 -> gap 10 and
        span 70. Fold k trains from day 10k, so fold 7 trains from day 70 --
        which is exactly where fold 0's test window ends, and so the first day
        of its embargo [70, 79). Seven folds fit in 130 days and are accepted;
        the eighth is the one that lands in the shadow.
        """
        spec = make_spec(purge_days=1, embargo_days=9)
        assert (spec.gap_days, spec.span_days) == (10, 70)
        start = date(2016, 1, 1)

        seven = generate_folds(rng(start, shift(start, 130)), spec, guard=open_guard)
        assert len(seven) == 7  # non-vacuity: this geometry does generate folds
        assert seven[0].embargo_after_test == rng(shift(start, 70), shift(start, 79))

        with pytest.raises(EmbargoViolationError, match="fold_007 starts training") as exc:
            generate_folds(rng(start, shift(start, 140)), spec, guard=open_guard)
        assert not isinstance(exc.value, PurgeViolationError)
        assert str(shift(start, 70)) in str(exc.value)

    def test_the_refusal_is_the_modules_own_type_not_a_pydantic_wrapper(self):
        """Regression. `Fold` checks in a `model_validator`, and pydantic
        converts a `ValueError` raised there into a `ValidationError`. While
        `WalkForwardError` subclassed `ValueError`, both refusals arrived as
        `pydantic_core.ValidationError`: `isinstance(exc, PurgeViolationError)`
        was False for the purge case and for the embargo case alike, which made
        them interchangeable in every way a caller can observe."""
        for gap, purge, embargo, expected in (
            (0, 2, 5, PurgeViolationError),
            (3, 1, 5, EmbargoViolationError),
        ):
            with pytest.raises(expected) as exc:
                self.fold_with_gap(gap=gap, purge=purge, embargo=embargo)
            assert not isinstance(exc.value, ValidationError)
            assert type(exc.value) is expected

    def test_set_level_and_statement_level_refusals_are_typed_too(self, empty_folds):
        """Every class here checks inside a `model_validator`, so each one was
        affected by the same wrapping, not just `Fold`."""
        with pytest.raises(WalkForwardError, match="indices must") as exc:
            FoldSet(
                spec=empty_folds.spec,
                data_range=empty_folds.data_range,
                folds=empty_folds.folds[1:],
            )
        assert not isinstance(exc.value, ValidationError)

        with pytest.raises(WalkForwardError, match="every value must name its fold") as exc:
            CrossFoldStatement(
                metric="win_rate",
                phase=SplitPhase.TEST,
                fold_indices=(0,),
                per_fold=(),
                fold_n_trades=(),
            )
        assert not isinstance(exc.value, ValidationError)

        with pytest.raises(WalkForwardError, match="holes in its OOS coverage") as exc:
            make_spec(step_days=20, test_days=10)
        assert not isinstance(exc.value, ValidationError)


# ---------------------------------------------------------------------------
# No aggregate-only output
# ---------------------------------------------------------------------------


class TestNoAggregateOnlyOutput:
    """Section 10: "every fold is reported separately; there is no
    aggregate-only output". The mechanism is the shape of the type."""

    @pytest.fixture
    def statement(self, empty_folds, make_trade) -> CrossFoldStatement:
        runner = FakeRunner({0: (1.0, 1.0, -1.0), 1: (-1.0, 1.0, -1.0), 2: (1.0, -1.0)})
        report = report_over(empty_folds, runner, make_trade)
        return report.cross_fold("win_rate", SplitPhase.TEST)

    def test_the_per_fold_vector_is_a_required_constructor_field(self):
        fields = CrossFoldStatement.model_fields
        for name in ("fold_indices", "per_fold", "fold_n_trades"):
            assert name in fields
            assert fields[name].is_required()

    def test_the_aggregates_are_computed_projections_not_fields(self):
        for name in AGGREGATES:
            assert name not in CrossFoldStatement.model_fields
            assert name in CrossFoldStatement.model_computed_fields

    def test_no_constructor_accepts_an_aggregate(self, statement):
        payload = statement.to_init_dict()
        # Non-vacuous: the payload on its own rebuilds the statement.
        assert CrossFoldStatement(**payload).mean == statement.mean
        with pytest.raises(ValidationError):
            CrossFoldStatement(**payload, mean=0.99)

    def test_no_setter(self, statement):
        with pytest.raises(ValidationError):
            statement.mean = MetricValue.measured("win_rate_mean", 0.99, n=99)
        assert statement.mean.value == pytest.approx(statement.mean.value)

    def test_a_serialized_aggregate_carries_its_folds(self, statement):
        dumped = statement.model_dump()
        assert len(dumped["per_fold"]) == statement.n_folds
        assert tuple(dumped["fold_indices"]) == statement.fold_indices
        assert tuple(dumped["fold_n_trades"]) == statement.fold_n_trades
        for name in AGGREGATES:
            assert name in dumped
        # Round-tripped through JSON, the folds are still there.
        rebuilt = json.loads(json.dumps(statement.model_dump(mode="json")))
        assert len(rebuilt["per_fold"]) == statement.n_folds

    def test_there_is_no_cached_scalar_to_hold_without_the_folds(self, statement):
        payload = statement.to_init_dict()
        assert "mean" not in payload
        assert "spread" not in payload
        assert len(payload["per_fold"]) == statement.n_folds

    def test_keeping_the_mean_while_dropping_the_folds_is_refused(self, statement):
        assert statement.mean.is_defined  # non-vacuity
        with pytest.raises(WalkForwardError, match="every value must name its fold"):
            statement.replace(per_fold=())
        with pytest.raises(WalkForwardError, match="the size of"):
            statement.replace(per_fold=(), fold_indices=())
        emptied = statement.replace(per_fold=(), fold_indices=(), fold_n_trades=())
        assert not emptied.mean.is_defined
        assert emptied.mean.undefined is Undefined.NO_TRADES

    def test_a_statement_cannot_mix_two_metrics(self, statement):
        other = MetricValue.measured("profit_factor", 1.5, n=3)
        with pytest.raises(WalkForwardError, match="mixes values named"):
            statement.replace(per_fold=statement.per_fold[:-1] + (other,))

    def test_describe_prints_every_fold_before_the_aggregate(self, statement):
        lines = statement.describe()
        fold_lines = [i for i, line in enumerate(lines) if line.strip().startswith("fold_")]
        aggregate_lines = [
            i
            for i, line in enumerate(lines)
            if any(f"_{suffix}=" in line for suffix in ("mean", "stdev", "spread", "min", "max"))
        ]
        assert len(fold_lines) == statement.n_folds
        assert len(aggregate_lines) == 5
        assert max(fold_lines) < min(aggregate_lines)

    def test_the_spread_is_always_stated_alongside_the_mean(self, statement):
        for suffix, value in (
            ("mean", statement.mean),
            ("stdev", statement.stdev),
            ("spread", statement.spread),
            ("min", statement.minimum),
            ("max", statement.maximum),
        ):
            assert value.name == f"{statement.metric_name}_{suffix}"
            assert value.unit == statement.unit

    def test_dispersion_is_undefined_below_two_folds(self, empty_folds, make_trade):
        assert MIN_FOLDS_FOR_DISPERSION == 2
        runner = FakeRunner({0: (1.0, -1.0)})
        report = report_over(empty_folds, runner, make_trade)
        one = report.cross_fold("win_rate", SplitPhase.TEST)
        assert one.n_folds_defined == 1
        assert one.mean.is_defined
        assert not one.stdev.is_defined
        assert one.stdev.undefined is Undefined.INSUFFICIENT_OBSERVATIONS
        assert not one.spread.is_defined

        runner = FakeRunner({0: (1.0, -1.0), 1: (1.0, 1.0)})
        two = report_over(empty_folds, runner, make_trade).cross_fold(
            "win_rate", SplitPhase.TEST
        )
        assert two.n_folds_defined == 2
        assert two.stdev.is_defined
        assert two.spread.value == pytest.approx(0.5)

    def test_the_mean_is_unweighted_over_the_folds_that_are_defined(
        self, empty_folds, make_trade
    ):
        """A trade-weighted average across folds is a pooled statistic wearing a
        per-fold label. Fold 0: one win in four. Fold 1: one win in one."""
        runner = FakeRunner({0: (1.0, -1.0, -1.0, -1.0), 1: (1.0,)})
        statement = report_over(empty_folds, runner, make_trade).cross_fold(
            "win_rate", SplitPhase.TEST
        )
        assert statement.per_fold[0].value == pytest.approx(0.25)
        assert statement.per_fold[1].value == pytest.approx(1.0)
        assert statement.mean.value == pytest.approx(0.625)
        # Pooled would be 2/5 = 0.4. The difference is the point.
        assert statement.mean.value != pytest.approx(0.4)
        assert statement.n_trades == 5

    def test_the_n_on_an_aggregate_counts_folds_and_the_note_counts_trades(
        self, empty_folds, make_trade
    ):
        runner = FakeRunner({0: (1.0, -1.0), 1: (-1.0,)})
        statement = report_over(empty_folds, runner, make_trade).cross_fold(
            "win_rate", SplitPhase.TEST
        )
        assert statement.mean.n == statement.n_folds_defined == 2
        assert statement.n_trades == 3
        assert "2 of 4 fold(s) contributed, 3 trade(s) in total" in statement.mean.note

    def test_undefined_folds_stay_in_the_vector_with_their_reasons(
        self, empty_folds, make_trade
    ):
        runner = FakeRunner({0: (1.0, -1.0)})
        statement = report_over(empty_folds, runner, make_trade).cross_fold(
            "win_rate", SplitPhase.TEST
        )
        assert statement.n_folds == 4
        assert len(statement.per_fold) == 4
        assert statement.undefined_reasons == {"no_trades": 3}
        assert "no_trades x3" in statement.mean.note

    def test_a_thin_fold_is_labelled_insufficient_and_still_reported(
        self, open_guard, make_trade
    ):
        """`min_trades_per_fold` labels; it does not filter. A fold dropped for
        thinness is a bad period hidden, so the fold stays in the vector and its
        `sample` says what it is."""
        folds = generate_folds(
            rng(date(2016, 1, 1), date(2016, 4, 1)),
            make_spec(min_trades_per_fold=30),
            guard=open_guard,
        )
        runner = FakeRunner({0: (1.0, -1.0), 1: (-1.0, -1.0)})
        report = report_over(folds, runner, make_trade)
        windows = report.window_results(SplitPhase.TEST)
        assert [w.sample for w in windows] == [
            SampleStatus.INSUFFICIENT,
            SampleStatus.INSUFFICIENT,
            SampleStatus.EMPTY,
            SampleStatus.EMPTY,
        ]
        assert all(w.is_reportable for w in windows)
        statement = report.cross_fold("win_rate", SplitPhase.TEST)
        assert statement.n_folds == 4
        assert statement.n_folds_defined == 2
        assert statement.n_trades == 4

    def test_the_report_exposes_no_bare_aggregate_and_no_ranking(self):
        """If someone later adds `best_fold` or a method returning a float, this
        fails. Return types are read with `get_type_hints`, because under
        `from __future__ import annotations` `__annotations__` holds strings."""
        assert typing.get_type_hints(WalkForwardReport.cross_fold)["return"] is (
            CrossFoldStatement
        )
        public = {
            name
            for name, member in vars(WalkForwardReport).items()
            if not name.startswith("_") and callable(member)
        }
        assert public == {
            "cross_fold",
            "describe",
            "phases",
            "summary",
            "trade_counts",
            "window_results",
        }
        for name in public:
            hints = typing.get_type_hints(getattr(WalkForwardReport, name))
            assert hints.get("return") not in (float, int, "float")
        for forbidden in ("best_fold", "best", "rank", "ranking", "mean", "aggregate"):
            assert not hasattr(WalkForwardReport, forbidden)

    def test_summary_returns_one_statement_per_metric_each_with_its_folds(
        self, empty_folds, make_trade
    ):
        runner = FakeRunner({0: (1.0, -1.0), 1: (-1.0,)})
        report = report_over(empty_folds, runner, make_trade)
        statements = report.summary(("win_rate", "expectancy"), SplitPhase.TEST)
        assert len(statements) == 2
        for statement in statements:
            assert len(statement.per_fold) == report.n_folds


# ---------------------------------------------------------------------------
# THE SEAL
# ---------------------------------------------------------------------------


class TestTheSeal:
    """A refused read is not an open, and must not spend the one-open budget."""

    def test_a_fold_request_touching_the_seal_raises(self, sealed_guard):
        with pytest.raises(SealedDataAccessError, match="sealed"):
            generate_folds(
                rng(date(2021, 1, 1), date(2023, 6, 1)),
                make_spec(train_days=365, validation_days=90, test_days=90, step_days=90),
                guard=sealed_guard,
            )

    def test_the_refusal_writes_no_seal_audit_entry(self, sealed_guard, audit_path):
        with pytest.raises(SealedDataAccessError):
            generate_folds(
                rng(date(2021, 1, 1), date(2023, 6, 1)),
                make_spec(train_days=365, validation_days=90, test_days=90, step_days=90),
                guard=sealed_guard,
            )
        assert not audit_path.exists()
        assert sealed_guard.audit.entries() == []
        assert sealed_guard.opens_used() == 0
        assert sealed_guard.opens_remaining() == 1
        assert not sealed_guard.is_unsealed

    def test_a_request_wholly_inside_the_seal_raises_and_spends_nothing(
        self, sealed_guard, audit_path
    ):
        with pytest.raises(SealedDataAccessError):
            generate_folds(
                rng(date(2023, 2, 1), date(2024, 12, 1)),
                make_spec(train_days=100, validation_days=30, test_days=30, step_days=30),
                guard=sealed_guard,
            )
        assert not audit_path.exists()
        assert sealed_guard.opens_remaining() == 1

    def test_the_whole_range_is_checked_before_any_fold_is_built(self, sealed_guard):
        """The first fold of this request is entirely inside the research window,
        so a per-fold-only check would build it before refusing. The refusal has
        to come from the range, or a partial fold set leaks out."""
        spec = make_spec(train_days=365, validation_days=90, test_days=90, step_days=90)
        early = generate_folds(
            rng(date(2015, 1, 1), date(2017, 1, 1)),
            spec,
            guard=DataAccessGuard(sealed=SEALED, audit=sealed_guard.audit, enabled=True),
        )
        assert len(early) >= 1  # non-vacuity: the geometry does fit
        with pytest.raises(SealedDataAccessError):
            generate_folds(rng(date(2015, 1, 1), date(2024, 1, 1)), spec, guard=sealed_guard)

    def test_nothing_is_clipped_to_make_a_request_fit(self, sealed_guard):
        """`clip_to_allowed` exists on the guard and is deliberately not used
        here: a fold set silently missing its last two years is worse than a
        refusal."""
        request = rng(date(2021, 1, 1), date(2024, 1, 1))
        assert sealed_guard.clip_to_allowed(request.start, request.end) == rng(
            date(2021, 1, 1), SEALED.start
        )
        with pytest.raises(SealedDataAccessError):
            generate_folds(
                request,
                make_spec(train_days=365, validation_days=90, test_days=90, step_days=90),
                guard=sealed_guard,
            )

    def test_the_research_window_up_to_the_seal_is_allowed(self, sealed_guard, audit_path):
        """The boundary is half-open, so a range ENDING at the seal's start does
        not touch it."""
        assert not sealed_guard.touches_seal(RESEARCH.start, RESEARCH.end)
        folds = generate_folds(
            RESEARCH,
            make_spec(train_days=365, validation_days=90, test_days=90, step_days=90),
            guard=sealed_guard,
            sealed=SEALED,
        )
        assert len(folds) > 1
        for fold in folds:
            assert not fold.span.overlaps(SEALED)
            assert fold.test.end <= SEALED.start
        assert not audit_path.exists()
        assert sealed_guard.opens_remaining() == 1

    def test_one_day_over_the_boundary_is_refused(self, sealed_guard):
        assert sealed_guard.touches_seal(date(2022, 1, 1), shift(SEALED.start, 1))
        with pytest.raises(SealedDataAccessError):
            generate_folds(
                rng(date(2019, 1, 1), shift(SEALED.start, 1)),
                make_spec(train_days=365, validation_days=90, test_days=90, step_days=90),
                guard=sealed_guard,
            )

    def test_evaluation_rechecks_the_guard_per_fold(
        self, open_guard, sealed_guard, audit_path, make_trade
    ):
        """A fold set built with one guard and evaluated under another -- the
        case a stale pickle or a hand-built `FoldSet` produces -- is refused at
        evaluation, and that refusal spends nothing either."""
        folds = generate_folds(
            rng(date(2022, 1, 1), date(2024, 6, 1)),
            make_spec(train_days=200, validation_days=100, test_days=100, step_days=100),
            guard=open_guard,
        )
        assert any(fold.span.overlaps(SEALED) for fold in folds)
        with pytest.raises(SealedDataAccessError):
            report_over(folds, FakeRunner(), make_trade, guard=sealed_guard)
        assert not audit_path.exists()
        assert sealed_guard.opens_remaining() == 1

    def test_the_seal_window_matches_the_committed_configuration(self, config):
        assert config.seal.sealed_start == SEALED.start
        assert config.seal.sealed_end == SEALED.end
        assert config.seal.max_opens == 1

    def test_a_genuine_open_does_write_the_audit(self, sealed_guard, audit_path):
        """Non-vacuity for every `not audit_path.exists()` above.

        If an open did not write the file either, "the refusal wrote nothing"
        would be true of a guard that never records anything, and the whole
        seal section would pass on a broken audit log. No sealed data is read
        here: the budget is spent and the row inspected, nothing is loaded.
        """
        assert not audit_path.exists()
        with sealed_guard.unseal(
            "non-vacuity check for the refusal tests",
            experiment_id="exp_test",
            approved_by="test",
        ) as entry:
            assert sealed_guard.is_unsealed
            assert entry.sequence == 1
        assert audit_path.exists()
        assert [e.experiment_id for e in sealed_guard.audit.entries()] == ["exp_test"]
        assert sealed_guard.opens_used() == 1
        assert sealed_guard.opens_remaining() == 0
        assert not sealed_guard.is_unsealed

    def test_a_refused_fold_request_leaves_the_one_open_unspent(
        self, sealed_guard, audit_path
    ):
        """The budget after a refusal is not merely reported as 1 -- it is
        still usable, which is the thing "must not spend the budget" means."""
        spec = make_spec(train_days=365, validation_days=90, test_days=90, step_days=90)
        for _ in range(3):
            with pytest.raises(SealedDataAccessError):
                generate_folds(rng(date(2021, 1, 1), date(2024, 1, 1)), spec, guard=sealed_guard)
        assert not audit_path.exists()
        assert sealed_guard.opens_remaining() == 1
        with sealed_guard.unseal(
            "the single final confirmation",
            experiment_id="exp_final",
            approved_by="test",
        ) as entry:
            assert entry.sequence == 1  # the refusals consumed nothing
        assert sealed_guard.opens_used() == 1

    def test_the_last_test_window_may_not_abut_the_seal(self, sealed_guard, audit_path):
        """Regression, and the seal boundary stated as arithmetic.

        A fold set ending ON the seal's first day does not OVERLAP it --
        `DateRange` is half-open -- so the guard's non-overlap check passes it.
        What refuses it is the TEST-to-SEALED_OOS gap, and that gap was checked
        only when the caller passed `sealed=`. Omitted, a last test window
        ending 2023-01-01 was accepted with a gap of 0 against a required 6: a
        position opened on the last test day and still open would have its
        outcome set by sealed prices. `sealed` now defaults to the guard's own
        range, so no caller has to remember.

        train 333 -> span 365 -> gap 0, refused. train 328 -> gap 5, refused.
        train 327 -> gap 6, which is exactly purge+embargo, accepted.
        """
        data_range = rng(date(2022, 1, 1), SEALED.start)
        assert not sealed_guard.touches_seal(data_range.start, data_range.end)

        def build(train_days: int) -> FoldSet:
            spec = make_spec(
                train_days=train_days,
                validation_days=10,
                test_days=10,
                step_days=10,
                purge_days=1,
                embargo_days=5,
            )
            assert spec.gap_days == 6
            return generate_folds(data_range, spec, guard=sealed_guard)

        with pytest.raises(EmbargoViolationError, match="a gap of 0 day") as exc:
            build(333)
        assert "sealed_oos" in str(exc.value)
        with pytest.raises(EmbargoViolationError, match="a gap of 5 day"):
            build(328)

        clears = build(327)
        assert (SEALED.start - clears[-1].test.end).days == 6
        assert not clears[-1].test.overlaps(SEALED)
        # None of the three refusals or the acceptance touched the budget.
        assert not audit_path.exists()
        assert sealed_guard.opens_remaining() == 1

    def test_the_guard_supplies_the_seal_to_evaluation_too(
        self, sealed_guard, open_guard, make_trade
    ):
        """`evaluate_folds` takes the same default, so a fold set built
        elsewhere is still measured against the holdout boundary."""
        spec = make_spec(
            train_days=333,
            validation_days=10,
            test_days=10,
            step_days=10,
            purge_days=1,
            embargo_days=5,
        )
        folds = generate_folds(rng(date(2022, 1, 1), SEALED.start), spec, guard=open_guard)
        assert folds[-1].test.end == SEALED.start  # the open guard knows no seal
        with pytest.raises(EmbargoViolationError, match="a gap of 0 day"):
            report_over(folds, FakeRunner(), make_trade, guard=sealed_guard)
        # Non-vacuity: with no guard there is no seal to check against.
        assert report_over(folds, FakeRunner(), make_trade).n_folds == len(folds)

    def test_a_disabled_guard_contributes_no_seal(self, tmp_path, make_trade):
        """`seal.enabled = false` means there is no holdout, so the gap to one
        is not a thing to enforce. The guard's own `enabled` decides, not the
        presence of a range on it."""
        audit = SealAudit(tmp_path / "disabled.jsonl")
        disabled = DataAccessGuard(sealed=SEALED, audit=audit, enabled=False)
        assert disabled.sealed == SEALED and not disabled.enabled
        spec = make_spec(
            train_days=333,
            validation_days=10,
            test_days=10,
            step_days=10,
            purge_days=1,
            embargo_days=5,
        )
        folds = generate_folds(rng(date(2022, 1, 1), SEALED.start), spec, guard=disabled)
        assert folds[-1].test.end == SEALED.start


# ---------------------------------------------------------------------------
# n = 0 -- THE LIVE CASE
# ---------------------------------------------------------------------------


class TestZeroTradeFolds:
    """The committed configuration takes no trades, so every fold is empty.

    An empty fold is a valid, reportable fold: not an error, and not a zero win
    rate.
    """

    @pytest.fixture
    def report(self, empty_folds, make_trade) -> WalkForwardReport:
        return report_over(empty_folds, FakeRunner(), make_trade)

    def test_every_fold_and_window_is_empty(self, report):
        assert report.n_folds == 4
        for phase in FOLD_PHASES:
            assert report.trade_counts(phase) == (0, 0, 0, 0)
            for window in report.window_results(phase):
                assert window.is_empty
                assert window.n_trades == 0
                assert window.sample is SampleStatus.EMPTY

    def test_an_empty_fold_is_reportable_and_never_dropped(self, report):
        for phase in FOLD_PHASES:
            windows = report.window_results(phase)
            assert len(windows) == report.n_folds
            assert all(window.is_reportable for window in windows)

    @pytest.mark.parametrize("metric", RATE_AND_RATIO_METRICS)
    @pytest.mark.parametrize("phase", FOLD_PHASES)
    def test_every_rate_and_ratio_is_undefined_not_zero(self, report, metric, phase):
        statement = report.cross_fold(metric, phase)
        for aggregate in AGGREGATES:
            value = getattr(statement, aggregate)
            assert value.value is None, f"{metric}.{aggregate} reported a number"
            assert value.undefined is Undefined.NO_TRADES
            assert value.n == 0

    @pytest.mark.parametrize("metric", RATE_AND_RATIO_METRICS)
    def test_every_per_fold_rate_is_undefined_too(self, report, metric):
        statement = report.cross_fold(metric, SplitPhase.TEST)
        assert all(not value.is_defined for value in statement.per_fold)
        assert statement.undefined_reasons == {"no_trades": 4}

    @pytest.mark.parametrize("metric", COUNT_AND_SUM_METRICS)
    def test_counts_and_sums_are_measured_zero(self, report, metric):
        """0.0 here IS the measurement: no trades happened and no money moved.
        Distinguishing this from an undefined rate is the whole point."""
        statement = report.cross_fold(metric, SplitPhase.TEST)
        assert statement.mean.value == pytest.approx(0.0)
        assert statement.mean.n == 4
        assert statement.spread.value == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "metric",
        ("periods_per_year", "sharpe", "sortino", "total_return", "cagr", "trades_per_year"),
    )
    def test_the_equity_side_metrics_are_undefined_too(self, report, metric):
        """`cross_fold` reads `trade_metrics`, so the equity-derived numbers --
        the ones PROVENANCE 2.6 records a defect in, a `total_return` of 0.039
        published across a NaN curve -- are asserted on the window directly.
        Nothing here is 0.0, and the drawdown profile is undefined with it."""
        for phase in FOLD_PHASES:
            for window in report.window_results(phase):
                value = getattr(window.performance, metric)
                assert value.value is None, f"{metric} reported a number at n=0"
                assert value.undefined is not None
                assert window.performance.drawdown.max_drawdown_pct.value is None
                assert window.performance.drawdown.n_points == 0

    def test_the_fold_count_is_still_reported(self, report):
        statement = report.cross_fold("win_rate", SplitPhase.TEST)
        assert statement.n_folds == 4
        assert statement.n_folds_defined == 0
        assert statement.n_trades == 0
        assert len(statement.fold_indices) == 4
        assert statement.fold_indices == (0, 1, 2, 3)
        assert "0 of 4 fold(s) contributed" in statement.mean.note

    def test_the_report_says_the_whole_study_is_empty(self, report):
        assert report.notes
        note = report.notes[0]
        assert "n=0" in note
        assert "4 fold(s)" in note

    def test_an_empty_study_describes_itself_without_raising(self, report):
        lines = report.describe(("win_rate", "profit_factor", "expectancy"))
        assert lines
        assert any("undefined" in line for line in lines)

    def test_a_fold_with_no_folds_at_all_is_still_a_coherent_statement(self):
        statement = CrossFoldStatement(
            metric="win_rate",
            phase=SplitPhase.TEST,
            fold_indices=(),
            per_fold=(),
            fold_n_trades=(),
        )
        assert statement.n_folds == 0
        assert statement.mean.undefined is Undefined.NO_TRADES
        assert statement.mean.note == "no folds"

    def test_no_trades_is_reserved_for_folds_that_really_held_none(
        self, empty_folds, make_trade
    ):
        """Regression. Nine trades across four folds, every one a loss:
        `average_win_r` is undefined in each fold for want of a WINNER, and the
        aggregate used to report `no_trades` with "0 trade(s) in total". A reader
        of the aggregate alone would have concluded the system stood flat through
        a period in which it lost nine times."""
        runner = FakeRunner(
            {0: (-1.0, -1.0, -1.0), 1: (-1.0, -1.0, -1.0), 2: (-1.0, -1.0), 3: (-1.0,)}
        )
        report = report_over(empty_folds, runner, make_trade)
        assert report.trade_counts(SplitPhase.TEST) == (3, 3, 2, 1)
        statement = report.cross_fold("average_win_r", SplitPhase.TEST)
        assert statement.n_trades == 9
        assert statement.mean.undefined is Undefined.NO_WINS
        assert statement.undefined_reasons == {"no_wins": 4}
        assert "9 trade(s) in total" in statement.mean.note
        # The losses themselves are not undefined, and say so.
        losses = report.cross_fold("average_loss_r", SplitPhase.TEST)
        assert losses.mean.value == pytest.approx(-1.0)
        assert losses.n_trades == 9

    def test_a_metrics_own_n_is_not_the_fold_trade_count(self, empty_folds, make_trade):
        """Why `fold_n_trades` is carried rather than summed off `per_fold`:
        `average_win_r.n` counts winners, not trades."""
        runner = FakeRunner({0: (1.0, -1.0, -1.0), 1: (1.0, 1.0, -1.0)})
        report = report_over(empty_folds, runner, make_trade)
        statement = report.cross_fold("average_win_r", SplitPhase.TEST)
        assert sum(value.n for value in statement.per_fold) == 3  # winners
        assert statement.n_trades == 6  # trades
        assert statement.fold_n_trades == (3, 3, 0, 0)

    def test_mixed_undefined_reasons_do_not_claim_one_of_them(
        self, empty_folds, make_trade
    ):
        runner = FakeRunner({0: (-1.0, -1.0)})
        report = report_over(empty_folds, runner, make_trade)
        statement = report.cross_fold("average_win_r", SplitPhase.TEST)
        assert statement.undefined_reasons == {"no_trades": 3, "no_wins": 1}
        assert statement.mean.undefined is Undefined.INSUFFICIENT_OBSERVATIONS
        assert "no_trades x3" in statement.mean.note
        assert "no_wins x1" in statement.mean.note


# ---------------------------------------------------------------------------
# ROLLING vs ANCHORED
# ---------------------------------------------------------------------------


class TestRollingVsAnchored:
    """Boundary dates written out, not recomputed.

    Spec: train 30 / validation 10 / test 10 / step 10, purge 1 + embargo 2.
    Data range [2016-01-01, 2016-04-01). span = 30 + 3 + 10 + 3 + 10 = 56, so
    folds start at offsets 0, 10, 20, 30 and the fourth ends on 2016-03-27.
    """

    ROLLING_TRAIN = (
        (date(2016, 1, 1), date(2016, 1, 31)),
        (date(2016, 1, 11), date(2016, 2, 10)),
        (date(2016, 1, 21), date(2016, 2, 20)),
        (date(2016, 1, 31), date(2016, 3, 1)),
    )
    ANCHORED_TRAIN = (
        (date(2016, 1, 1), date(2016, 1, 31)),
        (date(2016, 1, 1), date(2016, 2, 10)),
        (date(2016, 1, 1), date(2016, 2, 20)),
        (date(2016, 1, 1), date(2016, 3, 1)),
    )
    VALIDATION = (
        (date(2016, 2, 3), date(2016, 2, 13)),
        (date(2016, 2, 13), date(2016, 2, 23)),
        (date(2016, 2, 23), date(2016, 3, 4)),
        (date(2016, 3, 4), date(2016, 3, 14)),
    )
    TEST = (
        (date(2016, 2, 16), date(2016, 2, 26)),
        (date(2016, 2, 26), date(2016, 3, 7)),
        (date(2016, 3, 7), date(2016, 3, 17)),
        (date(2016, 3, 17), date(2016, 3, 27)),
    )

    def build(self, mode: FoldMode, guard: DataAccessGuard) -> FoldSet:
        return generate_folds(
            rng(date(2016, 1, 1), date(2016, 4, 1)),
            make_spec(mode=mode),
            guard=guard,
        )

    def test_rolling_train_window_slides_at_fixed_width(self, open_guard):
        folds = self.build(FoldMode.ROLLING, open_guard)
        assert len(folds) == 4
        assert tuple((f.train.start, f.train.end) for f in folds) == self.ROLLING_TRAIN
        assert {f.train.days for f in folds} == {30}

    def test_anchored_train_window_grows_from_a_fixed_start(self, open_guard):
        folds = self.build(FoldMode.ANCHORED, open_guard)
        assert len(folds) == 4
        assert tuple((f.train.start, f.train.end) for f in folds) == self.ANCHORED_TRAIN
        assert {f.train.start for f in folds} == {date(2016, 1, 1)}
        assert [f.train.days for f in folds] == [30, 40, 50, 60]

    @pytest.mark.parametrize("mode", (FoldMode.ROLLING, FoldMode.ANCHORED))
    def test_both_modes_evaluate_the_same_windows(self, open_guard, mode):
        folds = self.build(mode, open_guard)
        assert tuple((f.validation.start, f.validation.end) for f in folds) == self.VALIDATION
        assert tuple((f.test.start, f.test.end) for f in folds) == self.TEST

    def test_the_modes_differ_only_in_the_train_window(self, open_guard):
        rolling = self.build(FoldMode.ROLLING, open_guard)
        anchored = self.build(FoldMode.ANCHORED, open_guard)
        assert [f.test for f in rolling] == [f.test for f in anchored]
        assert [f.train for f in rolling] != [f.train for f in anchored]
        assert rolling.fingerprint() != anchored.fingerprint()

    def test_test_windows_meet_exactly_once_so_oos_coverage_has_no_holes(self, open_guard):
        folds = self.build(FoldMode.ROLLING, open_guard)
        for earlier, later in zip(folds, folds[1:]):
            assert later.test.start == earlier.test.end
            assert not earlier.test.overlaps(later.test)
        assert folds.test_coverage == rng(date(2016, 2, 16), date(2016, 3, 27))

    def test_one_study_uses_one_mode(self, open_guard):
        folds = self.build(FoldMode.ROLLING, open_guard)
        with pytest.raises(WalkForwardError, match="one study uses one mode"):
            FoldSet(
                spec=make_spec(mode=FoldMode.ANCHORED),
                data_range=folds.data_range,
                folds=folds.folds,
            )

    def test_fold_indices_must_be_chronological_zero_to_n(self, open_guard):
        folds = self.build(FoldMode.ROLLING, open_guard)
        with pytest.raises(WalkForwardError, match="indices must"):
            FoldSet(spec=folds.spec, data_range=folds.data_range, folds=folds.folds[1:])

    def test_a_fold_may_not_apply_a_different_purge_than_the_set(self, open_guard):
        folds = self.build(FoldMode.ROLLING, open_guard)
        with pytest.raises(WalkForwardError, match="applies purge/embargo"):
            FoldSet(
                spec=make_spec(purge_days=2, embargo_days=1),
                data_range=folds.data_range,
                folds=folds.folds,
            )

    @staticmethod
    def pair(second_offset: int) -> tuple[Fold, Fold]:
        """Two hand-built folds, the later one shifted by `second_offset` days.

        `WindowSpec` now refuses a step shorter than the test window, so
        `generate_folds` can no longer produce overlapping test windows -- which
        is exactly why the `FoldSet` check still needs testing directly. A
        fixture, a range read from a file, or a window nudged to "use a bit more
        data" arrives here instead.
        """
        def one(index: int, offset: int) -> Fold:
            start = shift(date(2016, 1, 1), offset)
            return make_fold(
                index=index,
                train=(start, shift(start, 10)),
                validation=(shift(start, 10), shift(start, 20)),
                test=(shift(start, 20), shift(start, 30)),
                purge_days=0,
                embargo_days=0,
            )

        return one(0, 0), one(1, second_offset)

    def test_overlapping_test_windows_are_refused(self):
        first, second = self.pair(5)
        assert first.test.overlaps(second.test)
        assert first.test.intersection(second.test) == rng(date(2016, 1, 26), date(2016, 1, 31))
        assert second.test.start > first.test.start
        with pytest.raises(WalkForwardError, match="double-count"):
            FoldSet(
                spec=make_spec(
                    train_days=10, test_days=10, step_days=10, purge_days=0, embargo_days=0
                ),
                data_range=rng(date(2016, 1, 1), date(2016, 2, 5)),
                folds=(first, second),
            )

    def test_non_overlapping_test_windows_are_accepted(self):
        """Non-vacuity: the same two folds, stepped by a full test window."""
        first, second = self.pair(10)
        assert not first.test.overlaps(second.test)
        folds = FoldSet(
            spec=make_spec(
                train_days=10, test_days=10, step_days=10, purge_days=0, embargo_days=0
            ),
            data_range=rng(date(2016, 1, 1), date(2016, 2, 10)),
            folds=(first, second),
        )
        assert len(folds) == 2

    def test_test_windows_that_do_not_advance_are_refused(self):
        """A zero or negative step is the shape a sign error takes."""
        first, _ = self.pair(10)
        twin = first.replace(index=1)
        assert twin.test.start == first.test.start
        with pytest.raises(WalkForwardError, match="Folds must advance in time"):
            FoldSet(
                spec=make_spec(
                    train_days=10, test_days=10, step_days=10, purge_days=0, embargo_days=0
                ),
                data_range=rng(date(2016, 1, 1), date(2016, 2, 10)),
                folds=(first, twin),
            )

    def test_no_fold_reaches_past_the_data_range(self, open_guard):
        folds = self.build(FoldMode.ROLLING, open_guard)
        assert folds[-1].test.end <= folds.data_range.end
        with pytest.raises(WalkForwardError, match="outside the data range"):
            FoldSet(
                spec=folds.spec,
                data_range=rng(date(2016, 1, 1), date(2016, 3, 1)),
                folds=folds.folds,
            )


# ---------------------------------------------------------------------------
# DETERMINISM
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same inputs, bit-identical output. No clock, no global RNG."""

    def test_two_generations_of_the_same_request_are_identical(self, open_guard):
        first = generate_folds(
            rng(date(2016, 1, 1), date(2016, 4, 1)), make_spec(), guard=open_guard
        )
        second = generate_folds(
            rng(date(2016, 1, 1), date(2016, 4, 1)), make_spec(), guard=open_guard
        )
        assert first.folds == second.folds
        assert first.fingerprint() == second.fingerprint()
        assert first.to_init_dict() == second.to_init_dict()
        assert first.describe() == second.describe()

    def test_the_geometry_holds_no_timestamp(self, empty_folds):
        """A clock read would show up here, and would make two identical
        requests look different."""

        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from walk(item)
            else:
                yield value

        assert not [v for v in walk(empty_folds.to_init_dict()) if isinstance(v, datetime)]

    def test_the_fingerprint_is_stable_across_processes_and_hash_seeds(self, empty_folds):
        """`PYTHONHASHSEED` changes dict and set iteration order. A fingerprint
        that moved with it would be unreproducible between runs."""
        script = (
            "import sys; sys.path.insert(0, '.');"
            "from datetime import date;"
            "from flow_model.validation.splits import DataAccessGuard, DateRange, SealAudit;"
            "from flow_model.validation.walk_forward import WindowSpec, generate_folds;"
            "g = DataAccessGuard(sealed=None, audit=SealAudit('/dev/null'), enabled=False);"
            "s = WindowSpec(train_days=30, validation_days=10, test_days=10, step_days=10,"
            " purge_days=1, embargo_days=2, min_trades_per_fold=0);"
            "f = generate_folds(DateRange(start=date(2016,1,1), end=date(2016,4,1)), s, guard=g);"
            "print(f.fingerprint())"
        )
        seen = set()
        for seed in ("0", "1", "12345"):
            done = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
                check=True,
            )
            seen.add(done.stdout.strip())
        assert len(seen) == 1
        assert seen == {empty_folds.fingerprint()}

    def test_a_different_request_gets_a_different_fingerprint(self, open_guard):
        base = generate_folds(
            rng(date(2016, 1, 1), date(2016, 4, 1)), make_spec(), guard=open_guard
        )
        wider = generate_folds(
            rng(date(2016, 1, 1), date(2016, 4, 2)), make_spec(), guard=open_guard
        )
        other_gap = generate_folds(
            rng(date(2016, 1, 1), date(2016, 4, 1)),
            make_spec(purge_days=2, embargo_days=1),
            guard=open_guard,
        )
        assert len({base.fingerprint(), wider.fingerprint(), other_gap.fingerprint()}) == 3

    def test_the_spec_fingerprint_separates_the_purge_from_the_embargo(self):
        """gap_days is the same 3 in both, so a fingerprint over the gap alone
        would call these two specs identical."""
        one = make_spec(purge_days=1, embargo_days=2)
        two = make_spec(purge_days=2, embargo_days=1)
        assert one.gap_days == two.gap_days == 3
        assert one.fingerprint() != two.fingerprint()

    def test_evaluation_is_reproducible(self, empty_folds, make_trade):
        runner_a = FakeRunner({0: (1.0, -1.0), 1: (-1.0,)})
        runner_b = FakeRunner({0: (1.0, -1.0), 1: (-1.0,)})
        first = report_over(empty_folds, runner_a, make_trade)
        second = report_over(empty_folds, runner_b, make_trade)
        assert first.to_init_dict() == second.to_init_dict()
        assert first.data_hash == second.data_hash
        assert first.describe(("win_rate",)) == second.describe(("win_rate",))

    def test_the_reports_data_hash_is_order_independent(self, empty_folds, make_trade):
        """It digests a SORTED set of the per-window data hashes, so the hash
        does not depend on the order a set happened to iterate in."""
        report = report_over(empty_folds, FakeRunner(), make_trade)
        assert report.data_hash
        again = report_over(empty_folds, FakeRunner(), make_trade)
        assert report.data_hash == again.data_hash


# ---------------------------------------------------------------------------
# generate_folds: refusals and the guard
# ---------------------------------------------------------------------------


class TestGenerateFolds:
    def test_the_guard_is_required_and_keyword_only(self):
        """"An optional guard is a guard that gets omitted." Asserted on the
        signature rather than taken from the docstring."""
        parameter = inspect.signature(generate_folds).parameters["guard"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
        assert typing.get_type_hints(generate_folds)["guard"] is DataAccessGuard
        with pytest.raises(TypeError):
            generate_folds(rng(date(2016, 1, 1), date(2016, 4, 1)), make_spec())

    def test_no_fold_fitting_raises_rather_than_returning_an_empty_set(self, open_guard):
        """An empty `FoldSet` would flow downstream and surface as "undefined,
        n=0" -- the honest report of an empty market and a dishonest report of
        an impossible spec."""
        with pytest.raises(InsufficientRangeError, match="no fold fits"):
            generate_folds(
                rng(date(2016, 1, 1), date(2016, 2, 1)), make_spec(), guard=open_guard
            )

    def test_a_range_exactly_one_span_wide_yields_exactly_one_fold(self, open_guard):
        spec = make_spec()
        start = date(2016, 1, 1)
        folds = generate_folds(rng(start, shift(start, spec.span_days)), spec, guard=open_guard)
        assert len(folds) == 1
        assert folds[0].test.end == shift(start, spec.span_days)
        with pytest.raises(InsufficientRangeError):
            generate_folds(
                rng(start, shift(start, spec.span_days - 1)), spec, guard=open_guard
            )

    def test_a_step_shorter_than_the_test_window_is_refused_at_the_spec(self):
        """Regression. `step_days < test_days` makes consecutive test windows
        overlap, which `FoldSet` refuses -- so the spec could be built, folds
        could be generated, and the refusal arrived from inside pydantic as a
        `ValidationError` about two folds that were only ever the symptom."""
        with pytest.raises(WalkForwardError, match="would overlap"):
            make_spec(step_days=5, test_days=10)
        # Non-vacuous: the same spec with a matching step is accepted.
        assert make_spec(step_days=10, test_days=10).step_days == 10

    def test_a_step_longer_than_the_test_window_is_refused(self):
        with pytest.raises(WalkForwardError, match="holes in its OOS coverage"):
            make_spec(step_days=20, test_days=10)

    def test_a_fold_holds_no_sealed_window(self):
        assert SplitPhase.SEALED_OOS not in FOLD_PHASES
        assert FOLD_PHASES == (SplitPhase.TRAIN, SplitPhase.VALIDATION, SplitPhase.TEST)

    def test_split_refuses_a_phase_a_fold_does_not_hold(self, empty_folds):
        fold = empty_folds[0]
        assert fold.split(SplitPhase.TEST).name == "fold_000_test"
        assert fold.split(SplitPhase.TEST).range == fold.test
        with pytest.raises(WalkForwardError, match="holds no"):
            fold.split(SplitPhase.SEALED_OOS)

    def test_a_folds_registry_enforces_the_splits_rules(self, empty_folds):
        registry = empty_folds[0].registry()
        assert [split.name for split in registry.splits] == [
            "fold_000_train",
            "fold_000_validation",
            "fold_000_test",
        ]
        registry.assert_embargo(empty_folds.spec.gap_days)

    def test_fold_ids_are_zero_padded_and_match_their_index(self, empty_folds):
        assert [fold.fold_id for fold in empty_folds] == [
            "fold_000",
            "fold_001",
            "fold_002",
            "fold_003",
        ]

    def test_the_set_supports_len_iteration_and_indexing(self, empty_folds):
        assert len(empty_folds) == empty_folds.n_folds == 4
        assert [fold.index for fold in empty_folds] == [0, 1, 2, 3]
        assert empty_folds[2].index == 2


# ---------------------------------------------------------------------------
# evaluate_folds
# ---------------------------------------------------------------------------


class TestEvaluateFolds:
    def test_the_runner_is_called_once_per_fold_window(self, empty_folds, make_trade):
        runner = FakeRunner()
        report_over(empty_folds, runner, make_trade)
        assert len(runner.calls) == 4 * len(FOLD_PHASES)
        assert len(set(runner.calls)) == len(runner.calls)
        assert runner.calls[:3] == [
            "fold_000_train",
            "fold_000_validation",
            "fold_000_test",
        ]

    def test_a_repeated_run_id_is_refused(self, empty_folds, make_trade):
        """The symptom a reused `BacktestLoop` produces. One loop, one run."""
        runner = FakeRunner(run_id="run-same")
        with pytest.raises(WalkForwardError, match="run_id"):
            report_over(empty_folds, runner, make_trade)

    def test_rows_dated_outside_the_window_are_refused(self, empty_folds, make_trade):
        """Regression. `run_id` uniqueness does NOT catch an unclipped clock:
        `run_id_for` hashes the split name, so a runner walking the whole dataset
        under each fold's name still returns distinct ids. Nothing else tied a
        fold's numbers to a fold's dates."""

        class Wandering(FakeRunner):
            def trades_for(self, split, make_trade):
                if split.phase is not SplitPhase.TEST:
                    return ()
                return (
                    make_trade(
                        trade_id=f"{split.name}_outside",
                        signal_ts=datetime(2023, 6, 1, 10, tzinfo=timezone.utc),
                        split_label=split.phase,
                    ),
                )

        with pytest.raises(WalkForwardError, match="outside the window"):
            report_over(empty_folds, Wandering(), make_trade)

    def test_an_equity_curve_wider_than_the_fold_is_refused(self, empty_folds, make_trade):
        """Total return, CAGR, Sharpe and the drawdown profile all come off this
        series, so a series wider than the fold reports another period's path."""

        class WholeStudy(FakeRunner):
            def equity_for(self, split):
                span = empty_folds.data_range
                return (mark(span.start), mark(shift(span.end, -1)))

        with pytest.raises(WalkForwardError, match="outside the window"):
            report_over(empty_folds, WholeStudy(), make_trade)

    def test_rows_inside_the_window_are_accepted(self, empty_folds, make_trade):
        """Non-vacuity for the two refusals above: a correctly clipped runner,
        with both trades and an equity curve, passes."""
        runner = FakeRunner({0: (1.0,)}, equity=True)
        report = report_over(empty_folds, runner, make_trade)
        assert report.trade_counts(SplitPhase.TEST) == (1, 0, 0, 0)
        window = report.window_results(SplitPhase.TEST)[0]
        assert window.performance.total_return.is_defined

    def test_the_last_day_of_a_window_is_inside_it_and_midnight_after_is_not(
        self, empty_folds, make_trade
    ):
        """The half-open boundary, as `evaluate_folds` enforces it."""
        fold = empty_folds[0]
        last_day = shift(fold.test.end, -1)

        def runner_stamped(stamp: datetime) -> FakeRunner:
            class Stamped(FakeRunner):
                def trades_for(self, split, mk):
                    if split.phase is not SplitPhase.TEST or split.name != "fold_000_test":
                        return ()
                    return (
                        mk(
                            trade_id="edge",
                            signal_ts=stamp,
                            hold_minutes=1.0,
                            split_label=split.phase,
                        ),
                    )

            return Stamped()

        inside = datetime.combine(last_day, datetime.min.time(), tzinfo=timezone.utc)
        report = report_over(empty_folds, runner_stamped(inside), make_trade)
        assert report.trade_counts(SplitPhase.TEST) == (1, 0, 0, 0)

        outside = datetime.combine(fold.test.end, datetime.min.time(), tzinfo=timezone.utc)
        with pytest.raises(WalkForwardError, match="outside the window"):
            report_over(empty_folds, runner_stamped(outside), make_trade)

    def test_trades_labelled_with_another_phase_are_refused(self, empty_folds, make_trade):
        class Mislabelled(FakeRunner):
            def trades_for(self, split, mk):
                if split.phase is not SplitPhase.TEST:
                    return ()
                base = datetime.combine(
                    split.range.start, datetime.min.time(), tzinfo=timezone.utc
                )
                return (
                    mk(
                        trade_id=f"{split.name}_x",
                        signal_ts=base + timedelta(hours=10),
                        split_label=SplitPhase.TRAIN,
                    ),
                )

        with pytest.raises(SplitLabelMismatchError):
            report_over(empty_folds, Mislabelled(), make_trade)

    def test_a_window_result_must_agree_with_its_own_summary(self, empty_folds, make_trade):
        report = report_over(empty_folds, FakeRunner({0: (1.0,)}), make_trade)
        window = report.window_results(SplitPhase.TEST)[0]
        assert window.performance.split_label is SplitPhase.TEST
        assert window.performance.n_trades == window.n_trades == 1
        with pytest.raises(WalkForwardError, match="filed"):
            window.replace(phase=SplitPhase.VALIDATION, split_id="x")
        with pytest.raises(WalkForwardError, match="disagrees"):
            window.replace(n_trades=7)

    def test_asking_for_no_phase_is_refused(self, empty_folds, make_trade):
        with pytest.raises(WalkForwardError, match="no fold phase to evaluate"):
            report_over(empty_folds, FakeRunner(), make_trade, phases=())
        with pytest.raises(WalkForwardError, match="no fold phase to evaluate"):
            report_over(
                empty_folds, FakeRunner(), make_trade, phases=(SplitPhase.SEALED_OOS,)
            )

    def test_a_restricted_phase_list_reports_only_those_windows(
        self, empty_folds, make_trade
    ):
        report = report_over(
            empty_folds, FakeRunner(), make_trade, phases=(SplitPhase.TEST,)
        )
        assert report.phases() == (SplitPhase.TEST,)
        assert all(len(result.windows) == 1 for result in report.folds)
        with pytest.raises(KeyError):
            report.window_results(SplitPhase.TRAIN)

    def test_phases_are_reported_in_chronological_order(self, empty_folds, make_trade):
        report = report_over(
            empty_folds,
            FakeRunner(),
            make_trade,
            phases=(SplitPhase.TEST, SplitPhase.TRAIN),
        )
        assert report.phases() == (SplitPhase.TRAIN, SplitPhase.TEST)

    def test_every_window_is_written_to_the_experiment_log(
        self, empty_folds, make_trade, tmp_path
    ):
        log = ExperimentLog(tmp_path / "experiments.db")
        report = report_over(
            empty_folds, FakeRunner(), make_trade, log=log, reason="phase 8 fold study"
        )
        rows = log.query()
        assert len(rows) == 4 * len(FOLD_PHASES) == log.count()
        assert len({row.split_id for row in rows}) == len(rows)
        assert log.phase_touches(Cfg.config_hash, SplitPhase.TEST) == 4
        payload = rows[-1].results
        assert payload["n_trades"] == 0
        assert payload["purge_days"] == empty_folds.spec.purge_days
        assert payload["embargo_days"] == empty_folds.spec.embargo_days
        assert payload["range"]["start"] == empty_folds[-1].test.start.isoformat()
        for result in report.folds:
            for window in result.windows:
                assert window.experiment_id

    def test_the_log_rows_carry_the_fold_set_fingerprint(
        self, empty_folds, make_trade, tmp_path
    ):
        log = ExperimentLog(tmp_path / "experiments.db")
        report_over(empty_folds, FakeRunner(), make_trade, log=log)
        assert all(empty_folds.fingerprint() in row.notes for row in log.query())

    def test_a_report_shorter_than_its_fold_set_is_refused(self, empty_folds, make_trade):
        report = report_over(empty_folds, FakeRunner(), make_trade)
        with pytest.raises(WalkForwardError, match="dropped a period"):
            report.replace(folds=report.folds[:-1])

    def test_a_report_whose_folds_do_not_match_the_set_is_refused(
        self, empty_folds, make_trade
    ):
        report = report_over(empty_folds, FakeRunner(), make_trade)
        shuffled = (report.folds[1], report.folds[0]) + report.folds[2:]
        with pytest.raises(WalkForwardError, match="does not match"):
            report.replace(folds=shuffled)

    def test_every_fold_must_report_the_same_phases(self, empty_folds, make_trade):
        """Regression. `phases()` reads the first fold, so a fold missing a
        window made `cross_fold` a mean over a different set of folds than the
        one it named."""
        report = report_over(empty_folds, FakeRunner(), make_trade)
        thinned = FoldResult(
            fold=report.folds[1].fold,
            windows=tuple(
                w for w in report.folds[1].windows if w.phase is not SplitPhase.VALIDATION
            ),
        )
        with pytest.raises(WalkForwardError, match="Every fold reports on"):
            report.replace(folds=(report.folds[0], thinned) + report.folds[2:])

    def test_one_fold_cannot_hold_two_results_for_the_same_phase(
        self, empty_folds, make_trade
    ):
        report = report_over(empty_folds, FakeRunner(), make_trade)
        result = report.folds[0]
        with pytest.raises(WalkForwardError, match="two results for the same phase"):
            result.replace(windows=result.windows + (result.windows[0],))

    def test_a_window_must_report_on_its_folds_own_range(self, empty_folds, make_trade):
        report = report_over(empty_folds, FakeRunner(), make_trade)
        result = report.folds[0]
        window = result.window(SplitPhase.TEST)
        moved = window.replace(range=rng(date(2017, 1, 1), date(2017, 1, 11)))
        with pytest.raises(WalkForwardError, match="reports on"):
            result.replace(
                windows=tuple(
                    moved if w.phase is SplitPhase.TEST else w for w in result.windows
                )
            )

    def test_an_unknown_metric_name_says_what_is_available(self, empty_folds, make_trade):
        report = report_over(empty_folds, FakeRunner(), make_trade)
        with pytest.raises(KeyError, match="not in this window's summary"):
            report.cross_fold("sharpe_over_folds", SplitPhase.TEST)

    def test_describe_lists_the_geometry_then_the_folds_then_the_statements(
        self, empty_folds, make_trade
    ):
        report = report_over(empty_folds, FakeRunner({0: (1.0, -1.0)}), make_trade)
        lines = report.describe(("win_rate",))
        assert lines[0].startswith("4 fold(s) [ROLLING]")
        assert any("fold_003" in line for line in lines)
        statement_head = next(i for i, line in enumerate(lines) if "win_rate [" in line)
        assert any("fold_000" in line for line in lines[:statement_head])
        assert any(line.strip().startswith("win_rate_mean=") for line in lines)

    def test_an_unlabelled_trade_is_refused(self, empty_folds, make_trade):
        """Section 0.8: no performance claim without a split label. An
        unlabelled trade filed under the fold's phase would be an unlabelled
        number wearing an OOS label."""

        class Unlabelled(FakeRunner):
            def trades_for(self, split, mk):
                if split.phase is not SplitPhase.TEST:
                    return ()
                base = datetime.combine(
                    split.range.start, datetime.min.time(), tzinfo=timezone.utc
                )
                return (
                    mk(
                        trade_id=f"{split.name}_x",
                        signal_ts=base + timedelta(hours=10),
                        split_label=None,
                    ),
                )

        with pytest.raises(SplitLabelMismatchError):
            report_over(empty_folds, Unlabelled(), make_trade)

    def test_a_window_for_a_phase_the_fold_does_not_hold_is_a_typed_refusal(
        self, empty_folds
    ):
        """Regression. `FoldResult._check` indexed `fold.ranges()[phase]`
        directly, so a window carrying SEALED_OOS -- a holdout result filed as
        a fold's own -- was refused by a bare `KeyError` whose entire message
        was `<SplitPhase.SEALED_OOS: 'SEALED_OOS'>`. `except WalkForwardError`
        around report assembly did not catch it, and the message named neither
        the fold nor what a fold actually holds. `Fold.split` already refused
        the same phase with a `WalkForwardError`; the two now agree."""
        sealed_window = FoldWindowResult(
            fold_index=0,
            phase=SplitPhase.SEALED_OOS,
            split_id="fold_000_sealed_oos",
            range=SEALED,
            n_trades=0,
            sample=sample_status(0, 1),
            min_trades_per_fold=0,
            performance=performance_summary((), (), split_label=SplitPhase.SEALED_OOS),
        )
        with pytest.raises(WalkForwardError, match="a phase fold_000 does not hold") as exc:
            FoldResult(fold=empty_folds[0], windows=(sealed_window,))
        assert not isinstance(exc.value, KeyError)
        assert "SEALED_OOS" in str(exc.value)
        # Non-vacuity: a window for a phase the fold does hold is accepted.
        assert FoldResult(
            fold=empty_folds[0],
            windows=(
                FoldWindowResult(
                    fold_index=0,
                    phase=SplitPhase.TEST,
                    split_id="fold_000_test",
                    range=empty_folds[0].test,
                    n_trades=0,
                    sample=sample_status(0, 1),
                    min_trades_per_fold=0,
                    performance=performance_summary((), (), split_label=SplitPhase.TEST),
                ),
            ),
        ).window(SplitPhase.TEST).phase is SplitPhase.TEST

    def test_the_log_row_grades_the_fold_against_min_trades_per_fold(
        self, open_guard, make_trade, tmp_path
    ):
        """Regression. The row recorded `summary.sample`, graded against the
        analytics `min_sample_n`, under the key `sample` on a FOLD row whose
        own verdict comes from `min_trades_per_fold`. With min_trades_per_fold
        5 and min_sample_n 2, a 3-trade fold the report calls INSUFFICIENT was
        logged ADEQUATE, and neither threshold was in the row to re-derive it
        from. Section 10 builds the overfitting report from this table."""
        folds = generate_folds(
            rng(date(2016, 1, 1), date(2016, 4, 1)),
            make_spec(min_trades_per_fold=5),
            guard=open_guard,
        )
        policy = MetricPolicy(
            min_sample_n=2,
            risk_free_rate_annual=0.0,
            trading_days_per_year=252,
            flow_score_buckets=(70.0, 80.0),
        )
        log = ExperimentLog(tmp_path / "experiments.db")
        report = report_over(
            folds, FakeRunner({0: (1.0, -1.0, -1.0)}), make_trade, policy=policy, log=log
        )
        window = report.window_results(SplitPhase.TEST)[0]
        assert window.n_trades == 3
        assert window.sample is SampleStatus.INSUFFICIENT
        assert window.performance.sample is SampleStatus.ADEQUATE  # the two differ

        row = next(r for r in log.query() if r.split_id == window.split_id)
        assert row.results["sample"] == SampleStatus.INSUFFICIENT.value
        assert row.results["min_trades_per_fold"] == 5
        # The analytics verdict is still recorded, under a name that says so.
        assert row.results["analytics_sample"] == SampleStatus.ADEQUATE.value
        assert row.results["min_sample_n"] == 2


# ---------------------------------------------------------------------------
# The committed configuration
# ---------------------------------------------------------------------------


class TestCommittedConfig:
    """What the committed `walk_forward.*` actually produces. Read, never set."""

    def test_the_committed_months_resolve_to_the_days_this_study_uses(self, config):
        spec = WindowSpec.from_config(config, RESEARCH.start)
        assert (spec.train_days, spec.validation_days, spec.test_days) == (1096, 365, 365)
        assert (spec.purge_days, spec.embargo_days, spec.gap_days) == (1, 5, 6)
        assert spec.step_days == spec.test_days == 365
        assert spec.span_days == 1838
        assert spec.mode is FoldMode.ROLLING
        assert spec.min_trades_per_fold == config.walk_forward.min_trades_per_fold

    def test_the_research_window_holds_three_folds_at_the_committed_spec(
        self, config, sealed_guard
    ):
        spec = WindowSpec.from_config(config, RESEARCH.start)
        folds = generate_folds(RESEARCH, spec, guard=sealed_guard, sealed=SEALED)
        assert len(folds) == 3
        assert folds[0].train.start == date(2015, 1, 1)
        assert folds[0].test == rng(date(2019, 1, 13), date(2020, 1, 13))
        assert folds[-1].test == rng(date(2021, 1, 12), date(2022, 1, 12))
        assert folds.test_coverage == rng(date(2019, 1, 13), date(2022, 1, 12))
        assert folds[-1].test.end < SEALED.start

    def test_equal_month_counts_give_equal_day_widths_at_every_anchor(self, config):
        """Regression. `step_months` used to be resolved from the anchor while
        `test_months` was resolved from the test window's start, so equal month
        counts could give unequal day counts. At `anchor=2016-02-29` -- a real
        NQ session -- the committed config gave step_days=365 against
        test_days=366, and the folds double-counted one out-of-sample day."""
        assert config.walk_forward.step_months == config.walk_forward.test_months
        leap = WindowSpec.from_config(config, date(2016, 2, 29))
        assert leap.step_days == leap.test_days == 366
        day = RESEARCH.start
        while day < RESEARCH.end:
            spec = WindowSpec.from_config(config, day)
            assert spec.step_days == spec.test_days, day
            day = shift(day, 1)

    def test_a_leap_anchor_now_builds_folds_that_do_not_double_count(
        self, config, open_guard
    ):
        spec = WindowSpec.from_config(config, date(2016, 2, 29))
        folds = generate_folds(
            rng(date(2016, 2, 29), date(2023, 1, 1)), spec, guard=open_guard
        )
        assert len(folds) > 1
        for earlier, later in zip(folds, folds[1:]):
            assert later.test.start == earlier.test.end
            assert not earlier.test.overlaps(later.test)

    def test_the_committed_embargo_and_purge_are_distinct_numbers(self, config):
        """If they were equal, a test that conflated them could still pass."""
        assert config.walk_forward.purge_days != config.walk_forward.embargo_days
