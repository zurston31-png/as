"""Walk-forward folds, and per-fold reporting (ARCHITECTURE.md section 10).

Section 10 asks for "configurable train/validate/test windows, rolling or
anchored, with a PURGE + EMBARGO gap between windows so a trade open across a
boundary cannot leak", and for "every fold reported separately; there is no
aggregate-only output". This module is that and nothing else: it builds folds,
runs an injected `Runner` over them, and reports. It owns no data loading,
computes no metric of its own, and has no notion of a "best" fold.

THE UNIT IS THE CALENDAR DAY
----------------------------
Every length here -- train, validation, test, step, purge, embargo -- is a
number of **calendar days**, and every boundary is a `datetime.date`. Trading
sessions are deliberately not a supported unit: there is no `train_sessions`
field and no session-counting method, because supporting both invites the
reader to guess which one a number is, and a 36 that means sessions where days
were meant is a nine-fold error in the window width with no other symptom.
`WindowSpec.from_config` resolves the committed `walk_forward.*_months` values
against one date by true calendar arithmetic, so the month counts in
`config/defaults.yaml` keep their meaning with no days-per-month constant
invented anywhere.

PURGE IS NOT EMBARGO
--------------------
They are different repairs for different leaks, and conflating them is the
usual bug, so they are separate fields with separate errors:

* **PURGE** removes observations from the END of the EARLIER window -- the one
  being fitted or selected on. A position opened near the end of train and
  still open when the next window begins has its outcome determined by the
  next window's prices, so that trade's label is contaminated even though its
  entry is not. The purge width states how long a position can stay open, and
  it comes off the fitting side of the boundary.

* **EMBARGO** is a dead band that FOLLOWS an evaluation window. Returns are
  serially correlated, so the days just after a window are partly predictable
  from it; letting the next window begin inside that shadow carries
  information backwards across the seam. The embargo width states how long
  that correlation persists, and it delays the later window rather than
  trimming the earlier one.

Inside a fold both apply at each interior boundary, so the gap between
consecutive windows is `purge_days + embargo_days` (`WindowSpec.gap_days`), and
a `Fold` whose own ranges do not honour it raises at construction:
`PurgeViolationError` when the gap is short of the purge alone,
`EmbargoViolationError` when the purge fits and the embargo does not. The gap
is checked again through `splits.SplitRegistry.assert_embargo`, which section
10 names as Phase 8's job -- available there and deliberately not automatic.

Across folds the embargo is the span `Fold.embargo_after_test`, and the rule is
that no later fold's train window may BEGIN inside it. It is not a ban on a
later train window spanning an earlier test window: `splits.py` records that a
rolling walk-forward legitimately reuses an earlier fold's test range as later
training data, and that reuse is the mechanism of a rolling study. What the
rule forbids is a train window whose earliest observation is drawn from an
evaluation window's correlation shadow with nothing before it to dilute.

WHY THERE IS NO AGGREGATE-ONLY OUTPUT
-------------------------------------
A mean across folds is the most misleading number this project could print.
Five folds at 2.0R and one at -11R average to a positive expectancy no account
would have survived, and an average quoted without the dispersion and the fold
count is indistinguishable from one that is real. Section 10 also wants "72%
stable" preferred over "90% in one window", which cannot be stated without the
spread.

So the aggregate is not a number this module stores. `CrossFoldStatement` takes
the per-fold `MetricValue`s -- and each fold's own trade count -- as REQUIRED
constructor fields, and exposes `mean`, `stdev`, `spread`, `minimum`,
`maximum`, `n_folds`, `n_folds_defined` and `n_trades` only as computed
projections of those vectors: no setter, no cached scalar, no constructor
accepting a mean. Keeping the mean while dropping the folds is not something a
caller can do, `model_dump()` included, and `describe()` prints every fold
first.

ZERO TRADES IS THE LIVE CASE
----------------------------
The committed configuration takes no trades, so every fold is empty. An empty
fold is a valid, reportable fold: not an error, and not a zero win rate.
`FoldWindowResult` carries `SampleStatus.EMPTY` and a `PerformanceSummary` in
which every rate and ratio is `Undefined.NO_TRADES`. A `CrossFoldStatement`
over folds that are all empty is undefined with n=0 -- n on an aggregate counts
the FOLDS that contributed -- with the per-fold reasons listed, never 0.0.

`generate_folds` is a pure function of the data range and the spec: no clock
read, nothing drawn, so `FoldSet.fingerprint()` is stable across processes and
`fold_set.folds` is byte-identical for identical inputs.
"""

from __future__ import annotations

import calendar as _calendar
import math
from datetime import date
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from pydantic import Field, computed_field, model_validator

from flow_model.analytics.metrics import (
    STDEV_DDOF,
    MetricPolicy,
    MetricValue,
    PerformanceSummary,
    SampleStatus,
    Undefined,
    performance_summary,
    sample_status,
)
from flow_model.core.determinism import stable_hash
from flow_model.core.enums import SplitPhase, StrEnum
from flow_model.core.model import FrozenModel
from flow_model.utils.logging import get_logger
from flow_model.validation.experiment_log import ExperimentLog
from flow_model.validation.splits import (
    DataAccessGuard,
    DateRange,
    Split,
    SplitRegistry,
)

logger = get_logger("validation.walk_forward")

#: The windows a fold holds, in the order they occur in time. SEALED_OOS is
#: never one of them: a fold cannot be built over the holdout.
FOLD_PHASES: tuple[SplitPhase, ...] = (
    SplitPhase.TRAIN,
    SplitPhase.VALIDATION,
    SplitPhase.TEST,
)

#: Dispersion needs two observations. One fold has a mean and no spread, and a
#: spread of 0.0 over a single fold reads as "every fold agreed", which is the
#: opposite of what one fold means.
MIN_FOLDS_FOR_DISPERSION: int = 2


class WalkForwardError(Exception):
    """Base for every refusal in this module.

    NOT a `ValueError`, and that is the whole point. Pydantic converts a
    `ValueError` raised inside a `model_validator` into a `ValidationError`,
    and `Fold`, `FoldSet`, `FoldWindowResult`, `FoldResult`,
    `CrossFoldStatement` and `WalkForwardReport` do all of their checking
    there. While this base was `ValueError`, `Fold(...)` on a gap short of the
    purge raised `pydantic_core.ValidationError`:
    `isinstance(exc, PurgeViolationError)` was False, so was
    `isinstance(exc, EmbargoViolationError)`, `except PurgeViolationError` did
    not catch it, and the purge/embargo distinction this module exists to keep
    survived only as prose inside `exc.errors()[0]["ctx"]["error"]`. Two
    errors a caller cannot tell apart are one error.

    Pydantic passes any other exception type through untouched, so the error a
    caller sees is now the error this module raised.
    """


class PurgeViolationError(WalkForwardError):
    """A fold boundary leaves less room than `purge_days`."""


class EmbargoViolationError(WalkForwardError):
    """A gap clears the purge but not the embargo that must follow it."""


class InsufficientRangeError(WalkForwardError):
    """The data range cannot hold even one fold at this spec."""


class FoldMode(StrEnum):
    """How the train window moves from fold to fold."""

    ROLLING = "ROLLING"
    """Fixed-width train window that slides forward by `step_days`."""

    ANCHORED = "ANCHORED"
    """Train window starts at a fixed date and grows by `step_days`."""


def _shift(day: date, days: int) -> date:
    return date.fromordinal(day.toordinal() + days)


#: The only arithmetic `CrossFoldStatement` does. Keyed by the suffix each one
#: appends to the metric name, so an aggregate cannot be computed under a name
#: that does not describe it.
_AGGREGATORS: dict[str, Any] = {
    "mean": lambda v: np.mean(v),
    "stdev": lambda v: np.std(v, ddof=STDEV_DDOF),
    "min": min,
    "max": max,
    "spread": lambda v: max(v) - min(v),
}


def _add_months(day: date, months: int) -> date:
    """Calendar-month addition, clamping to the target month's length.

    Used only by `WindowSpec.from_config`. Clamping rather than rolling over
    keeps 31 January + 1 month inside February, so a month count never
    silently gains a day.
    """
    total = (day.year * 12 + day.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    return date(year, month, min(day.day, _calendar.monthrange(year, month)[1]))


class WindowSpec(FrozenModel):
    """Fold geometry, in calendar days. Pure data; no dates in it.

    One spec describes every fold in a `FoldSet`, which is what makes the folds
    comparable: a mean over folds of unequal width is a weighted average
    pretending to be an unweighted one.
    """

    train_days: int = Field(gt=0)
    validation_days: int = Field(gt=0)
    test_days: int = Field(gt=0)
    step_days: int = Field(gt=0)
    purge_days: int = Field(ge=0)
    embargo_days: int = Field(ge=0)
    mode: FoldMode = FoldMode.ROLLING
    min_trades_per_fold: int = Field(ge=0, default=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gap_days(self) -> int:
        """Days between consecutive windows: the purge plus the embargo."""
        return self.purge_days + self.embargo_days

    @computed_field  # type: ignore[prop-decorator]
    @property
    def span_days(self) -> int:
        """Calendar days from a fold's train start to its test end."""
        return (
            self.train_days
            + self.gap_days
            + self.validation_days
            + self.gap_days
            + self.test_days
        )

    @model_validator(mode="after")
    def _check(self) -> "WindowSpec":
        if self.step_days > self.test_days:
            raise WalkForwardError(
                f"step_days={self.step_days} exceeds test_days={self.test_days}: "
                "consecutive folds would leave out-of-sample days that no fold "
                "evaluates, and a walk-forward with holes in its OOS coverage "
                "reports on a period it chose after the fact"
            )
        if self.step_days < self.test_days:
            raise WalkForwardError(
                f"step_days={self.step_days} is short of test_days={self.test_days}: "
                f"consecutive test windows would overlap by "
                f"{self.test_days - self.step_days} day(s), and `FoldSet` refuses "
                "that -- double-counting the same out-of-sample days flatters the "
                "fold count and the dispersion at once. Refused here rather than "
                "after the folds are built, so the spec is named as the cause "
                "instead of two folds that were only ever a symptom. The step that "
                "makes test windows meet exactly once is step_days == test_days"
            )
        return self

    @classmethod
    def from_config(cls, config: Any, anchor: date) -> "WindowSpec":
        """Build from `config.walk_forward`, resolved against `anchor`.

        The committed configuration states window lengths in months, and a
        month is not a fixed number of days, so the conversion is done once by
        calendar arithmetic from `anchor` (normally the data range's start) and
        every fold then uses the resulting day widths. Two intended
        consequences: the folds are all exactly the same width, and no
        days-per-month constant exists in this module to be wrong.

        `step_months` is resolved from the TEST window's start, not from
        `anchor`. A step is what makes consecutive test windows meet, so it has
        to be counted over the same stretch of calendar the test window covers
        or equal month counts stop meaning equal day counts. Resolved from
        `anchor` it did not: with the committed 36/12/12/12 months and
        `anchor=2016-02-29` it gave step_days=365 against test_days=366,
        because `_add_months(2016-02-29, 12)` clamps to 2017-02-28 while the
        test window [2020-02-28, 2021-02-28) contains 29 February 2020. The
        folds then overlapped on exactly one out-of-sample day. At
        `anchor=2015-01-01` both readings give 365, so nothing committed moves.
        """
        wf = config.walk_forward
        train_end = _add_months(anchor, wf.train_months)
        validation_end = _add_months(train_end, wf.validation_months)
        test_end = _add_months(validation_end, wf.test_months)
        step_end = _add_months(validation_end, wf.step_months)
        return cls(
            train_days=(train_end - anchor).days,
            validation_days=(validation_end - train_end).days,
            test_days=(test_end - validation_end).days,
            step_days=(step_end - validation_end).days,
            purge_days=wf.purge_days,
            embargo_days=wf.embargo_days,
            mode=FoldMode.ANCHORED if wf.anchored else FoldMode.ROLLING,
            min_trades_per_fold=wf.min_trades_per_fold,
        )

    def fingerprint(self) -> str:
        return stable_hash(self.to_init_dict())


class Fold(FrozenModel):
    """One fold: three dated windows, with the gaps already taken out.

    Constructing a `Fold` whose gaps fall short of `purge_days + embargo_days`
    raises. That is the point: `generate_folds` produces correct geometry by
    arithmetic, but a hand-built fold -- a fixture, a range read from a file, a
    window nudged to "use a bit more data" -- is where the gap goes missing,
    and it fails here instead of being accepted and reported.
    """

    index: int = Field(ge=0)
    mode: FoldMode
    train: DateRange
    validation: DateRange
    test: DateRange
    purge_days: int = Field(ge=0)
    embargo_days: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fold_id(self) -> str:
        return f"fold_{self.index:03d}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def span(self) -> DateRange:
        """Train start to test end. Excludes the trailing embargo."""
        return DateRange(start=self.train.start, end=self.test.end)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embargo_after_test(self) -> DateRange | None:
        """The dead band following this fold's test window, if any.

        `None` when `embargo_days` is zero, rather than an empty range: a
        `DateRange` cannot be empty, and a zero-width band is better said by
        its absence than by a degenerate object.
        """
        if self.embargo_days <= 0:
            return None
        return DateRange(
            start=self.test.end, end=_shift(self.test.end, self.embargo_days)
        )

    def ranges(self) -> dict[SplitPhase, DateRange]:
        return {
            SplitPhase.TRAIN: self.train,
            SplitPhase.VALIDATION: self.validation,
            SplitPhase.TEST: self.test,
        }

    def split(self, phase: SplitPhase) -> Split:
        """This fold's window for `phase`, as a named `Split`."""
        window = self.ranges().get(phase)
        if window is None:
            raise WalkForwardError(
                f"a fold holds no {phase.value} window; it holds "
                f"{[p.value for p in FOLD_PHASES]}"
            )
        return Split(name=f"{self.fold_id}_{phase.value.lower()}", phase=phase, range=window)

    def registry(self, sealed: DateRange | None = None) -> SplitRegistry:
        """This fold as a `SplitRegistry`, optionally including the seal.

        Built rather than stored, so the registry's own checks -- no overlap
        between a fitting and an evaluation phase, chronological order, no TEST
        inside SEALED_OOS -- run on every fold turned into one. Passing
        `sealed` is how a test window that has crept up against the holdout is
        caught: the registry rejects an overlap, and `assert_embargo` rejects a
        gap too small even where they do not overlap.
        """
        splits = [self.split(phase) for phase in FOLD_PHASES]
        if sealed is not None:
            splits.append(Split(name="sealed_oos", phase=SplitPhase.SEALED_OOS, range=sealed))
        return SplitRegistry(splits=tuple(splits))

    def assert_embargo(self, min_gap_days: int, sealed: DateRange | None = None) -> None:
        """Run `splits.assert_embargo` over this fold.

        Section 10: the embargo length is a configured research choice, so
        `SplitRegistry` does not check it automatically and Phase 8 is what
        calls it. This is that call -- a second, independent check on the
        geometry `Fold._check` already enforced, which additionally covers the
        TEST-to-SEALED_OOS boundary when `sealed` is given.

        The registry is BUILT OUTSIDE the try, so its own refusals keep their
        own identity. A test window that overlaps the holdout is a seal leak,
        not a short embargo, and relabelling it `EmbargoViolationError` --
        which is what building it inside did -- tells the reader to widen the
        embargo when what they actually did was reach into the sealed window.
        `splits.SplitRegistry` already says that in the words of the module
        that owns the seal.
        """
        registry = self.registry(sealed)
        try:
            registry.assert_embargo(min_gap_days)
        except ValueError as exc:
            raise EmbargoViolationError(f"{self.fold_id}: {exc}") from None

    def _check_gap(self, earlier: DateRange, later: DateRange, label: str) -> None:
        gap = (later.start - earlier.end).days
        if gap < self.purge_days:
            raise PurgeViolationError(
                f"{self.fold_id} {label}: {earlier} then {later}, a gap of {gap} "
                f"day(s), below purge_days={self.purge_days}. A position opened in the "
                f"earlier window and still open when the later one begins has its "
                f"outcome set by the later window's prices, so the purge comes off the "
                f"earlier window's end"
            )
        required = self.purge_days + self.embargo_days
        if gap < required:
            raise EmbargoViolationError(
                f"{self.fold_id} {label}: {earlier} then {later}, a gap of {gap} "
                f"day(s). The purge of {self.purge_days} fits, the embargo of "
                f"{self.embargo_days} that must follow it does not ({required} "
                f"required). Days after a window are partly predictable from it, so "
                f"the later window may not begin in that shadow"
            )

    @model_validator(mode="after")
    def _check(self) -> "Fold":
        self._check_gap(self.train, self.validation, "train -> validation")
        self._check_gap(self.validation, self.test, "validation -> test")
        return self

    def describe(self) -> str:
        embargo = self.embargo_after_test
        return (
            f"{self.fold_id} [{self.mode.value}] train{self.train} "
            f"validation{self.validation} test{self.test} "
            f"purge={self.purge_days}d embargo={self.embargo_days}d"
            + (f" embargo_after_test{embargo}" if embargo is not None else "")
        )


class FoldSet(FrozenModel):
    """Every fold of one walk-forward study, in chronological order.

    Cross-fold geometry is checked here rather than in `Fold`, because it is a
    relationship between folds: test windows must advance and must not overlap,
    and no fold's train window may begin inside an earlier fold's post-test
    embargo.
    """

    spec: WindowSpec
    data_range: DateRange
    folds: tuple[Fold, ...]

    def __len__(self) -> int:
        return len(self.folds)

    def __iter__(self) -> Iterator[Fold]:  # type: ignore[override]
        return iter(self.folds)

    def __getitem__(self, index: int) -> Fold:
        return self.folds[index]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def test_coverage(self) -> DateRange | None:
        """First test start to last test end, or None with no folds."""
        if not self.folds:
            return None
        return DateRange(start=self.folds[0].test.start, end=self.folds[-1].test.end)

    def _check_one(self, position: int, fold: Fold) -> None:
        if fold.index != position:
            raise WalkForwardError(
                f"fold at position {position} is indexed {fold.index}; indices must "
                "be 0..n-1 in chronological order, because every per-fold report is "
                "keyed by that index"
            )
        if fold.mode is not self.spec.mode:
            raise WalkForwardError(
                f"{fold.fold_id} is {fold.mode.value} in a {self.spec.mode.value} "
                "fold set; one study uses one mode"
            )
        if (fold.purge_days, fold.embargo_days) != (
            self.spec.purge_days,
            self.spec.embargo_days,
        ):
            raise WalkForwardError(
                f"{fold.fold_id} applies purge/embargo {fold.purge_days}/"
                f"{fold.embargo_days} against the set's {self.spec.purge_days}/"
                f"{self.spec.embargo_days}"
            )
        if not (
            self.data_range.start <= fold.train.start and fold.test.end <= self.data_range.end
        ):
            raise WalkForwardError(
                f"{fold.fold_id} spans {fold.span}, outside the data range "
                f"{self.data_range}. A fold reaching past the data it was built from "
                "reports on bars that do not exist"
            )

    @model_validator(mode="after")
    def _check(self) -> "FoldSet":
        for position, fold in enumerate(self.folds):
            self._check_one(position, fold)

        for earlier, later in zip(self.folds, self.folds[1:]):
            if later.test.start <= earlier.test.start:
                raise WalkForwardError(
                    f"{later.fold_id} tests {later.test}, not after {earlier.fold_id}'s "
                    f"{earlier.test}. Folds must advance in time; a zero or negative "
                    "step is the shape a sign error takes"
                )
            if earlier.test.overlaps(later.test):
                raise WalkForwardError(
                    f"{earlier.fold_id} and {later.fold_id} both evaluate "
                    f"{earlier.test.intersection(later.test)}. Overlapping test windows "
                    "double-count the same out-of-sample days, flattering the fold "
                    "count and the dispersion at once"
                )

        for earlier in self.folds:
            shadow = earlier.embargo_after_test
            if shadow is None:
                continue
            for later in self.folds[earlier.index + 1 :]:
                if shadow.contains(later.train.start):
                    raise EmbargoViolationError(
                        f"{later.fold_id} starts training on {later.train.start}, inside "
                        f"the embargo {shadow} following {earlier.fold_id}'s test window. "
                        f"Its earliest training observation would be drawn from that "
                        f"window's correlation shadow with nothing before it to dilute"
                    )
        return self

    def assert_embargo(self, sealed: DateRange | None = None) -> None:
        """Call `splits.assert_embargo` on every fold, with the spec's gap."""
        for fold in self.folds:
            fold.assert_embargo(self.spec.gap_days, sealed=sealed)

    def fingerprint(self) -> str:
        """Content hash of the geometry. Identical inputs, identical hash."""
        return stable_hash(
            {
                "spec": self.spec.to_init_dict(),
                "data_range": self.data_range.to_init_dict(),
                "folds": [fold.to_init_dict() for fold in self.folds],
            }
        )

    def describe(self) -> tuple[str, ...]:
        head = (
            f"{self.n_folds} fold(s) [{self.spec.mode.value}] over {self.data_range}: "
            f"train={self.spec.train_days}d validation={self.spec.validation_days}d "
            f"test={self.spec.test_days}d step={self.spec.step_days}d "
            f"purge={self.spec.purge_days}d embargo={self.spec.embargo_days}d "
            f"fingerprint={self.fingerprint()}"
        )
        return (head, *(fold.describe() for fold in self.folds))


def generate_folds(
    data_range: DateRange,
    spec: WindowSpec,
    *,
    guard: DataAccessGuard,
    sealed: DateRange | None = None,
) -> FoldSet:
    """Every fold that fits inside `data_range`, as a pure function.

    `guard` is required rather than optional. A fold generator is the one place
    in the project that turns a date-arithmetic mistake into a read of the
    holdout, and an optional guard is a guard that gets omitted. The whole
    requested range is checked before any fold is built, and each fold's span
    again, so a request touching the sealed window raises
    `SealedDataAccessError` instead of being quietly trimmed -- the trimming
    path exists on the guard (`clip_to_allowed`) and is deliberately not used
    here, because a fold set silently missing its last two years is worse than
    a refusal.

    Raises `InsufficientRangeError` when not one fold fits, rather than
    returning an empty set: no fold is a misconfiguration, and an empty
    `FoldSet` would flow downstream and surface as "undefined, n=0", which is
    the honest report of an empty market and a dishonest report of an
    impossible spec.
    """
    guard.check_access(
        data_range.start,
        data_range.end,
        purpose=f"walk-forward fold generation over {data_range}",
    )

    folds: list[Fold] = []
    index = 0
    while True:
        offset = index * spec.step_days
        if spec.mode is FoldMode.ANCHORED:
            train_start = data_range.start
            train_end = _shift(data_range.start, spec.train_days + offset)
        else:
            train_start = _shift(data_range.start, offset)
            train_end = _shift(train_start, spec.train_days)
        validation_start = _shift(train_end, spec.gap_days)
        validation_end = _shift(validation_start, spec.validation_days)
        test_start = _shift(validation_end, spec.gap_days)
        test_end = _shift(test_start, spec.test_days)
        if test_end > data_range.end:
            break
        fold = Fold(
            index=index,
            mode=spec.mode,
            train=DateRange(start=train_start, end=train_end),
            validation=DateRange(start=validation_start, end=validation_end),
            test=DateRange(start=test_start, end=test_end),
            purge_days=spec.purge_days,
            embargo_days=spec.embargo_days,
        )
        guard.check_access(fold.span.start, fold.span.end, purpose=f"walk-forward {fold.fold_id}")
        folds.append(fold)
        index += 1

    if not folds:
        raise InsufficientRangeError(
            f"no fold fits: {data_range} holds {data_range.days} day(s) and one fold of "
            f"this spec spans {spec.span_days} (train {spec.train_days} + gap "
            f"{spec.gap_days} + validation {spec.validation_days} + gap {spec.gap_days} "
            f"+ test {spec.test_days}). Widen the data range or build a spec whose "
            f"windows fit it; nothing here will shorten a window to make a fold appear"
        )

    fold_set = FoldSet(spec=spec, data_range=data_range, folds=tuple(folds))
    fold_set.assert_embargo(sealed=sealed)
    logger.info(
        "generated %d walk-forward fold(s) over %s (fingerprint %s)",
        fold_set.n_folds,
        data_range,
        fold_set.fingerprint(),
    )
    return fold_set


class FoldWindowResult(FrozenModel):
    """One (fold, phase) evaluation. Every number carries its own n.

    An empty window is a result, not a failure: `sample` is
    `SampleStatus.EMPTY`, `performance` holds `Undefined.NO_TRADES` in place of
    every rate and ratio, and `is_reportable` is True.
    """

    fold_index: int = Field(ge=0)
    phase: SplitPhase
    split_id: str
    range: DateRange
    n_trades: int = Field(ge=0)
    sample: SampleStatus
    min_trades_per_fold: int = Field(ge=0)
    performance: PerformanceSummary
    run_id: str = ""
    data_hash: str = ""
    experiment_id: str = ""
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.n_trades == 0

    @property
    def is_reportable(self) -> bool:
        """Always True. An empty fold is reported, never dropped.

        Named because "is this reportable" is the question a caller asks, and
        the answer being unconditional is the behaviour worth pinning: a fold
        filtered out for thinness is a bad period hidden.
        """
        return True

    def metric(self, name: str) -> MetricValue:
        """One metric from this window's summary, by registry name."""
        value = self.performance.trade_metrics.get(name)
        if value is None:
            raise KeyError(
                f"{name!r} is not in this window's summary; it holds "
                f"{sorted(self.performance.trade_metrics)}"
            )
        return value

    @model_validator(mode="after")
    def _check(self) -> "FoldWindowResult":
        if self.performance.split_label is not self.phase:
            raise WalkForwardError(
                f"{self.split_id}: the window is {self.phase.value} but its summary is "
                f"labelled {self.performance.split_label}. A per-fold number filed "
                "under the wrong phase is an in-sample result reported as OOS"
            )
        if self.performance.n_trades != self.n_trades:
            raise WalkForwardError(
                f"{self.split_id}: n_trades={self.n_trades} disagrees with its "
                f"summary's {self.performance.n_trades}"
            )
        expected = sample_status(self.n_trades, max(1, self.min_trades_per_fold))
        if self.sample is not expected:
            raise WalkForwardError(
                f"{self.split_id}: sample={self.sample.value} but n={self.n_trades} "
                f"against min_trades_per_fold={self.min_trades_per_fold} is "
                f"{expected.value}"
            )
        return self

    def describe(self) -> str:
        return (
            f"fold_{self.fold_index:03d} {self.phase.value} {self.range} "
            f"n={self.n_trades} sample={self.sample.value}"
        )


class FoldResult(FrozenModel):
    """One fold, with one result per window. No window may be missing."""

    fold: Fold
    windows: tuple[FoldWindowResult, ...]

    def window(self, phase: SplitPhase) -> FoldWindowResult:
        for window in self.windows:
            if window.phase is phase:
                return window
        raise KeyError(
            f"{self.fold.fold_id} has no {phase.value} window; it has "
            f"{[w.phase.value for w in self.windows]}"
        )

    @model_validator(mode="after")
    def _check(self) -> "FoldResult":
        phases = [window.phase for window in self.windows]
        if len(set(phases)) != len(phases):
            raise WalkForwardError(
                f"{self.fold.fold_id} has two results for the same phase: {phases}"
            )
        ranges = self.fold.ranges()
        for window in self.windows:
            if window.fold_index != self.fold.index:
                raise WalkForwardError(
                    f"{window.split_id} is indexed {window.fold_index} inside "
                    f"{self.fold.fold_id}"
                )
            if ranges[window.phase] != window.range:
                raise WalkForwardError(
                    f"{window.split_id} reports on {window.range} but "
                    f"{self.fold.fold_id}'s {window.phase.value} window is "
                    f"{ranges[window.phase]}"
                )
        return self

    def describe(self) -> tuple[str, ...]:
        return (self.fold.describe(), *(w.describe() for w in self.windows))


class CrossFoldStatement(FrozenModel):
    """A metric across folds. The per-fold values are the object.

    `mean`, `stdev`, `spread`, `minimum` and `maximum` are computed projections
    of `per_fold`, not stored fields. There is no constructor that takes a
    mean, no way to set one, and no way to hold this type without the vector it
    was derived from -- the structural form of section 10's "no aggregate-only
    output". The vector is in `model_dump()` for the same reason: a serialized
    aggregate carries its folds with it.

    Undefined folds stay in the vector with their reasons rather than being
    dropped; `n_folds` is the fold count and `n_folds_defined` is how many
    contributed. Reporting a mean over the two folds that happened to have
    trades, out of six, without saying so, is how a bad period gets hidden.
    """

    metric: str
    phase: SplitPhase
    fold_indices: tuple[int, ...]
    per_fold: tuple[MetricValue, ...]
    fold_n_trades: tuple[int, ...]
    """How many trades each fold's window held, in `fold_indices` order.

    Required, and carried rather than read off `per_fold`, because a
    `MetricValue`'s own `n` is the number of observations BEHIND THAT METRIC,
    which is not the trade count for any metric conditioned on an outcome.
    `average_win_r` counts winners; over four folds holding nine losing trades
    between them it counts zero, and `n_trades` summed from it used to report
    "0 trade(s) in total" for a study that had taken nine trades and lost all
    nine. That is a bad period reported as an empty one.
    """

    @model_validator(mode="after")
    def _check(self) -> "CrossFoldStatement":
        if len(self.fold_indices) != len(self.per_fold):
            raise WalkForwardError(
                f"{self.metric}: {len(self.fold_indices)} fold index(es) against "
                f"{len(self.per_fold)} value(s); every value must name its fold"
            )
        if len(self.fold_n_trades) != len(self.per_fold):
            raise WalkForwardError(
                f"{self.metric}: {len(self.fold_n_trades)} trade count(s) against "
                f"{len(self.per_fold)} value(s); every value must carry the size of "
                "the sample its fold held"
            )
        if any(count < 0 for count in self.fold_n_trades):
            raise WalkForwardError(
                f"{self.metric}: negative fold trade count in {self.fold_n_trades}"
            )
        if len(set(self.fold_indices)) != len(self.fold_indices):
            raise WalkForwardError(f"{self.metric}: duplicate fold indices")
        names = {value.name for value in self.per_fold}
        if len(names) > 1:
            raise WalkForwardError(
                f"{self.metric}: the vector mixes values named {sorted(names)}. A "
                "statement about one metric cannot be assembled from several. (The "
                "requested key may differ from the value's own name -- TRADE_METRICS "
                "aliases 'expectancy' to expectancy_r -- so the rule is that every "
                "fold agrees, not that it matches the key.)"
            )
        return self

    @classmethod
    def over(
        cls, results: Sequence[FoldResult], metric: str, phase: SplitPhase
    ) -> "CrossFoldStatement":
        """Collect `metric` for `phase` from every fold, in fold order."""
        windows = [result.window(phase) for result in results]
        return cls(
            metric=metric,
            phase=phase,
            fold_indices=tuple(w.fold_index for w in windows),
            per_fold=tuple(w.metric(metric) for w in windows),
            fold_n_trades=tuple(w.n_trades for w in windows),
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def metric_name(self) -> str:
        """The canonical name the folds' own values carry.

        Differs from `metric` when the lookup key is a `TRADE_METRICS` alias.
        The aggregates are named from this, so a serialized statement cannot
        claim a name the underlying measurement does not use.
        """
        for value in self.per_fold:
            return value.name
        return self.metric

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_folds(self) -> int:
        return len(self.per_fold)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_folds_defined(self) -> int:
        return sum(1 for value in self.per_fold if value.is_defined)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_trades(self) -> int:
        """Trades behind the whole statement, defined folds or not.

        Summed from `fold_n_trades`, which is the window's own trade count. A
        `MetricValue.n` would have been the count of observations the metric
        itself was taken over -- winners for `average_win_r`, zero for any
        undefined value -- and summing those reported no trades for folds that
        had them.
        """
        return sum(self.fold_n_trades)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def undefined_reasons(self) -> dict[str, int]:
        """How many folds were undefined, by reason. Empty when none were."""
        counts: dict[str, int] = {}
        for value in self.per_fold:
            if value.undefined is not None:
                counts[value.undefined.value] = counts.get(value.undefined.value, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def defined_values(self) -> tuple[float, ...]:
        return tuple(v.value for v in self.per_fold if v.value is not None)

    @property
    def unit(self) -> str:
        """The unit the folds' own values carry, so an aggregate keeps it."""
        for value in self.per_fold:
            return value.unit
        return ""

    def _coverage_note(self) -> str:
        """Fold accounting, carried on every aggregate this class returns.

        The `n` on an aggregate counts FOLDS, not trades: the sample a
        cross-fold mean is taken over is the folds. The trade count goes here
        instead, because dropping it would let "mean win rate over 4 folds"
        stand without saying the four folds hold two trades between them.
        """
        reasons = self.undefined_reasons
        head = (
            f"{self.n_folds_defined} of {self.n_folds} fold(s) contributed, "
            f"{self.n_trades} trade(s) in total"
        )
        if not reasons:
            return head
        listed = ", ".join(f"{reason} x{count}" for reason, count in reasons.items())
        return f"{head}; undefined: {listed}"

    def _undefined_reason(self) -> Undefined:
        """Why an aggregate has no number, taken from the folds themselves.

        `NO_TRADES` is reserved for a statement whose folds really held no
        trades. Before, any statement with nothing to average said `NO_TRADES`,
        so a metric undefined for want of a WINNER -- four folds, nine trades,
        nine losses, `average_win_r` undefined `NO_WINS` in every one --
        aggregated to "no trades". A reader of the aggregate alone would have
        concluded the system stood flat through a period in which it lost nine
        times, which is the one direction this project refuses to be wrong in.

        When the folds disagree about why they are undefined, no single reason
        is theirs; the aggregate says it has too few observations and
        `undefined_reasons` carries the breakdown.
        """
        if self.n_folds_defined > 0:
            return Undefined.INSUFFICIENT_OBSERVATIONS
        if self.n_trades == 0:
            return Undefined.NO_TRADES
        reasons = {value.undefined for value in self.per_fold if value.undefined is not None}
        if len(reasons) == 1:
            return reasons.pop()
        return Undefined.INSUFFICIENT_OBSERVATIONS

    def _aggregate(self, suffix: str, values: Sequence[float], minimum: int) -> MetricValue:
        name = f"{self.metric_name}_{suffix}"
        if self.n_folds == 0:
            return MetricValue.undefined_because(
                name, Undefined.NO_TRADES, n=0, unit=self.unit, note="no folds"
            )
        if len(values) < minimum:
            return MetricValue.undefined_because(
                name,
                self._undefined_reason(),
                n=len(values),
                unit=self.unit,
                note=self._coverage_note(),
            )
        number = float(_AGGREGATORS[suffix](values))
        if not math.isfinite(number):
            return MetricValue.undefined_because(
                name,
                Undefined.NOT_REPRESENTABLE,
                n=len(values),
                unit=self.unit,
                note=self._coverage_note(),
            )
        return MetricValue.measured(
            name, number, n=len(values), unit=self.unit, note=self._coverage_note()
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean(self) -> MetricValue:
        """Unweighted mean over the folds that are defined.

        Unweighted on purpose: a trade-weighted average across folds is a
        pooled statistic wearing a per-fold label, and it lets one busy fold
        speak for the study. `n` is the number of folds that contributed, so an
        all-empty study reads "undefined, n=0", and the note carries the fold
        accounting and the total trade count.
        """
        return self._aggregate("mean", self.defined_values, 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def stdev(self) -> MetricValue:
        """Sample standard deviation across folds. Undefined below two."""
        return self._aggregate("stdev", self.defined_values, MIN_FOLDS_FOR_DISPERSION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> MetricValue:
        """Maximum minus minimum across folds. Undefined below two."""
        return self._aggregate("spread", self.defined_values, MIN_FOLDS_FOR_DISPERSION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def minimum(self) -> MetricValue:
        return self._aggregate("min", self.defined_values, 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def maximum(self) -> MetricValue:
        return self._aggregate("max", self.defined_values, 1)

    def describe(self) -> tuple[str, ...]:
        """Per-fold lines first, then the aggregate. Never the other way."""
        per_fold = tuple(
            f"  fold_{index:03d}: {value}"
            for index, value in zip(self.fold_indices, self.per_fold)
        )
        return (
            f"{self.metric_name} [{self.phase.value}] over {self.n_folds} fold(s):",
            *per_fold,
            f"  {self.mean}",
            f"  {self.stdev}",
            f"  {self.spread}",
            f"  {self.minimum}",
            f"  {self.maximum}",
        )


class WalkForwardReport(FrozenModel):
    """The whole study: the geometry, every fold, and nothing aggregate-only.

    The only route to a cross-fold number is `cross_fold`, which builds a
    `CrossFoldStatement` from the folds this report holds. There is no method
    returning a bare mean, no `best_fold` and no ranking: a Phase 8 report that
    names its best fold has selected on the evaluation data, and the absence of
    the method is the enforcement.
    """

    fold_set: FoldSet
    folds: tuple[FoldResult, ...]
    config_hash: str = ""
    data_hash: str = ""
    notes: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @model_validator(mode="after")
    def _check(self) -> "WalkForwardReport":
        if len(self.folds) != len(self.fold_set.folds):
            raise WalkForwardError(
                f"{len(self.folds)} fold result(s) against {len(self.fold_set.folds)} "
                "generated fold(s). Every fold is reported, empty ones included; a "
                "report shorter than its fold set has dropped a period"
            )
        for result, fold in zip(self.folds, self.fold_set.folds):
            if result.fold != fold:
                raise WalkForwardError(
                    f"fold result {result.fold.fold_id} does not match the generated "
                    f"{fold.fold_id}"
                )
        if self.folds:
            expected = {w.phase for w in self.folds[0].windows}
            for result in self.folds[1:]:
                present = {w.phase for w in result.windows}
                if present != expected:
                    raise WalkForwardError(
                        f"{result.fold.fold_id} reports on "
                        f"{sorted(p.value for p in present)} against "
                        f"{self.folds[0].fold.fold_id}'s "
                        f"{sorted(p.value for p in expected)}. Every fold reports on "
                        "the same phases or they are not comparable: `phases()` reads "
                        "the first fold, and a cross-fold statement about a phase one "
                        "fold is missing is a mean over a different set of folds than "
                        "the one it names"
                    )
        return self

    def phases(self) -> tuple[SplitPhase, ...]:
        """The phases every fold reports on, in chronological order.

        Read off the first fold, which `_check` has established speaks for all
        of them.
        """
        if not self.folds:
            return ()
        common = [w.phase for w in self.folds[0].windows]
        return tuple(p for p in FOLD_PHASES if p in common)

    def window_results(self, phase: SplitPhase) -> tuple[FoldWindowResult, ...]:
        return tuple(result.window(phase) for result in self.folds)

    def trade_counts(self, phase: SplitPhase) -> tuple[int, ...]:
        return tuple(w.n_trades for w in self.window_results(phase))

    def cross_fold(self, metric: str, phase: SplitPhase) -> CrossFoldStatement:
        """A cross-fold statement about `metric`, carrying every fold."""
        return CrossFoldStatement.over(self.folds, metric, phase)

    def summary(
        self, metrics: Sequence[str], phase: SplitPhase
    ) -> tuple[CrossFoldStatement, ...]:
        return tuple(self.cross_fold(metric, phase) for metric in metrics)

    def describe(self, metrics: Sequence[str] = ()) -> tuple[str, ...]:
        lines: list[str] = list(self.fold_set.describe())
        for result in self.folds:
            lines.extend(result.describe())
        for phase in self.phases():
            for metric in metrics:
                lines.extend(self.cross_fold(metric, phase).describe())
        lines.extend(self.notes)
        return tuple(lines)


def _window_payload(
    fold: Fold, phase: SplitPhase, split: Split, summary: PerformanceSummary
) -> dict[str, Any]:
    """The experiment-log row body for one fold window."""
    metrics: Mapping[str, MetricValue] = summary.trade_metrics
    return {
        "fold_index": fold.index,
        "fold_id": fold.fold_id,
        "phase": phase.value,
        "split_id": split.name,
        "range": {"start": split.range.start.isoformat(), "end": split.range.end.isoformat()},
        "purge_days": fold.purge_days,
        "embargo_days": fold.embargo_days,
        "n_trades": summary.n_trades,
        "sample": summary.sample.value,
        "metrics": {name: value.model_dump(mode="json") for name, value in metrics.items()},
    }


def _assert_rows_inside(split: Split, result: Any) -> None:
    """Refuse a runner's rows that are dated outside the fold's own window.

    `run_id` uniqueness catches a reused `BacktestLoop`. It does NOT catch a
    clock that was never clipped: a runner that walks the whole dataset for
    every fold returns a different `data_hash` per window only by accident, and
    nothing else here ties a fold's numbers to a fold's dates. So the rows are
    checked against the window rather than the runner being trusted to have
    clipped -- a fold whose trades or equity curve quietly span its neighbours,
    or the seal, reports an in-sample number under an out-of-sample label, and
    it reports it with a plausible `split_id` attached.

    `DateRange` is half-open and `contains` normalizes to UTC, so a row stamped
    at midnight UTC on the window's end date belongs to the NEXT window. That
    is the same boundary `robustness.StoreRunner` clips its clock on
    (`start <= ts < end`), by construction rather than by coincidence.
    """
    window = split.range
    for trade in result.trades:
        for field in ("signal_ts", "entry_ts", "exit_ts"):
            stamp = getattr(trade, field)
            if not window.contains(stamp):
                raise WalkForwardError(
                    f"{split.name}: trade {trade.trade_id} has {field}={stamp}, "
                    f"outside the window {window}. The runner's clock was not clipped "
                    f"to this fold, so this window's numbers are measured over days "
                    f"that belong to another window -- or to the holdout"
                )
    for point in result.equity:
        if not window.contains(point.ts):
            raise WalkForwardError(
                f"{split.name}: an equity point is stamped {point.ts}, outside the "
                f"window {window}. Total return, CAGR, Sharpe and the drawdown "
                f"profile are all taken from this series, so a series wider than the "
                f"fold reports another period's path under this fold's label"
            )


def evaluate_folds(
    fold_set: FoldSet,
    config: Any,
    runner: Any,
    *,
    phases: Sequence[SplitPhase] = FOLD_PHASES,
    policy: MetricPolicy | None = None,
    guard: DataAccessGuard | None = None,
    sealed: DateRange | None = None,
    log: ExperimentLog | None = None,
    reason: str = "",
    parent_experiment_id: str | None = None,
) -> WalkForwardReport:
    """Run `runner(config, split)` once per fold window and report each.

    `runner` is any `(config, Split) -> RunResult` callable, injected so this
    module drives folds without owning data loading and so the n=0 path is
    testable without a data store. `validation/robustness.py` declares the same
    shape as `Runner` and ships a `StoreRunner` for it; this module does not
    import either, so a fold study runs whether or not that module is present.

    Whatever the runner is, it must build a FRESH `BacktestLoop` per call. That
    is not this module's to fake: a second `run()` on one loop raises, because
    `RiskLimitManager` accumulates, and reusing a loop across folds would carry
    one fold's daily-loss and consecutive-loss state into the next. This
    function calls `runner` exactly once per (fold, phase) and refuses
    duplicate `run_id`s, which is the symptom a reused loop would produce.

    A mis-clipped clock is a SEPARATE failure and `run_id` does not catch it:
    `BacktestLoop.run_id_for` hashes the split name among its inputs, so a
    runner that walks the whole dataset under each fold's name still hands back
    distinct ids. `_assert_rows_inside` checks every trade and equity row
    against the window's own dates instead.

    Every window is written to the experiment log when one is supplied, with
    its own `split_id` and phase, so section 10's overfitting report is
    generated from that table rather than from this object.
    """
    ordered = tuple(p for p in FOLD_PHASES if p in tuple(phases))
    if not ordered:
        raise WalkForwardError(
            f"no fold phase to evaluate: asked for {[p.value for p in phases]}, and a "
            f"fold holds {[p.value for p in FOLD_PHASES]}"
        )

    config_hash = str(getattr(config, "config_hash", ""))
    results: list[FoldResult] = []
    seen_run_ids: dict[str, str] = {}
    data_hashes: set[str] = set()

    for fold in fold_set.folds:
        if guard is not None:
            guard.check_access(
                fold.span.start,
                fold.span.end,
                purpose=f"walk-forward evaluation {fold.fold_id}",
            )
        fold.assert_embargo(fold_set.spec.gap_days, sealed=sealed)

        windows: list[FoldWindowResult] = []
        for phase in ordered:
            split = fold.split(phase)
            result = runner(config, split)
            _assert_rows_inside(split, result)
            run_id = str(result.provenance.run_id)
            if run_id in seen_run_ids:
                raise WalkForwardError(
                    f"{split.name} returned run_id {run_id}, already produced by "
                    f"{seen_run_ids[run_id]}. Two fold windows cannot be the same run: "
                    "either a loop was reused across folds or the clock was not "
                    "clipped to the fold, and in both cases the per-fold numbers are "
                    "not per-fold"
                )
            seen_run_ids[run_id] = split.name
            data_hashes.add(str(result.provenance.data_hash))

            summary = performance_summary(
                result.trades, result.equity, split_label=phase, policy=policy
            )
            experiment_id = ""
            if log is not None:
                record = log.log(
                    f"walk_forward {fold.fold_id} {phase.value}",
                    config_hash or str(result.provenance.config_hash),
                    phase,
                    data_hash=str(result.provenance.data_hash),
                    split_id=split.name,
                    reason=reason,
                    results=_window_payload(fold, phase, split, summary),
                    notes=f"fold_set={fold_set.fingerprint()} run_id={run_id}",
                    parent_experiment_id=parent_experiment_id,
                )
                experiment_id = record.experiment_id

            windows.append(
                FoldWindowResult(
                    fold_index=fold.index,
                    phase=phase,
                    split_id=split.name,
                    range=split.range,
                    n_trades=len(result.trades),
                    sample=sample_status(
                        len(result.trades), max(1, fold_set.spec.min_trades_per_fold)
                    ),
                    min_trades_per_fold=fold_set.spec.min_trades_per_fold,
                    performance=summary,
                    run_id=run_id,
                    data_hash=str(result.provenance.data_hash),
                    experiment_id=experiment_id,
                )
            )
        results.append(FoldResult(fold=fold, windows=tuple(windows)))

    notes: list[str] = []
    if all(w.is_empty for r in results for w in r.windows):
        notes.append(
            f"every window of every fold is empty (n=0) across {len(results)} fold(s). "
            "This is the committed configuration's measured behaviour, not a failure "
            "of the fold machinery: no cross-fold rate, ratio or mean is a number, and "
            "each is reported undefined with its reason."
        )
    return WalkForwardReport(
        fold_set=fold_set,
        folds=tuple(results),
        config_hash=config_hash,
        data_hash=stable_hash(sorted(data_hashes)) if data_hashes else "",
        notes=tuple(notes),
    )
