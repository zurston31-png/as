"""Date splits and the sealed out-of-sample guard.

The brief's hardest requirement is "keep a completely untouched final OOS
dataset" and "prevent accidental test-set contamination". A comment cannot
enforce that, so this module makes it a runtime error:

    * every data request passes through `DataAccessGuard.check_access`
    * a request overlapping the sealed window raises `SealedDataAccessError`
    * the only way through is `guard.unseal(reason=..., experiment_id=...)`,
      which appends an immutable audit entry and is budgeted by `max_opens`
    * the audit log is on disk, so the budget survives process restarts --
      you cannot reset it by rerunning the script

The failure mode this prevents is not malice. It is running a sweep that
happens to span 2015-2025 and quietly fitting on the holdout.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import Field, computed_field, model_validator

from flow_model.core.enums import SplitPhase
from flow_model.core.model import FrozenModel
from flow_model.utils.logging import get_logger

logger = get_logger("validation.splits")


class SealedDataAccessError(RuntimeError):
    """Raised when code reaches into the sealed out-of-sample window."""


class SealBudgetExhaustedError(RuntimeError):
    """Raised when the sealed window has already been opened `max_opens` times."""


def _as_date(value: date | datetime | str) -> date:
    """The calendar date a timestamp belongs to, in UTC.

    The UTC normalization is load-bearing. An aware datetime's `.date()` is
    its date *in its own zone*, so the single instant 2016-01-01T02:30Z and
    2015-12-31T21:30-05:00 -- the same moment -- used to return different
    dates, and therefore landed on opposite sides of a seal boundary
    depending only on which zone the caller happened to construct it in. A
    seal boundary that moves with the caller's tzinfo is not a boundary.
    Everything in this project stores int64 UTC nanoseconds, so UTC is the
    convention the rest of the system already uses.

    A naive datetime is taken as already-UTC rather than guessed at.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


class DateRange(FrozenModel):
    """Half-open interval [start, end)."""

    start: date
    end: date

    @model_validator(mode="after")
    def _check(self) -> "DateRange":
        if self.end <= self.start:
            raise ValueError(f"DateRange end ({self.end}) must be after start ({self.start})")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def contains(self, value: date | datetime | str) -> bool:
        d = _as_date(value)
        return self.start <= d < self.end

    def overlaps(self, other: "DateRange") -> bool:
        return self.start < other.end and other.start < self.end

    def intersection(self, other: "DateRange") -> "DateRange | None":
        if not self.overlaps(other):
            return None
        return DateRange(start=max(self.start, other.start), end=min(self.end, other.end))

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.start} -> {self.end})"


class Split(FrozenModel):
    """A named date range with a validation phase."""

    name: str
    phase: SplitPhase
    range: DateRange

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.name}({self.phase.value}){self.range}"


class SealAuditEntry(FrozenModel):
    """One immutable record of the sealed window being opened."""

    ts: datetime
    reason: str
    experiment_id: str
    approved_by: str
    config_hash: str = ""
    requested: DateRange | None = None
    pid: int = 0
    sequence: int = 0


class SealAudit:
    """Append-only JSONL audit log for seal openings."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def entries(self) -> list[SealAuditEntry]:
        if not self.path.exists():
            return []
        out: list[SealAuditEntry] = []
        for line_no, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(SealAuditEntry.from_mapping(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                # A corrupt audit line must not be silently skipped: that is
                # exactly how an opening would get lost.
                raise RuntimeError(
                    f"{self.path}:{line_no} is not a readable seal audit entry: {exc}"
                ) from exc
        return out

    def open_count(self) -> int:
        return len(self.entries())

    def record(self, entry: SealAuditEntry) -> SealAuditEntry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = entry.replace(sequence=self.open_count() + 1, pid=os.getpid())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_init_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return entry


class SplitRegistry(FrozenModel):
    """The project's date splits, with structural guarantees.

    Enforced: no TRAIN or VALIDATION range may overlap a TEST or SEALED_OOS
    range, and TRAIN may not overlap VALIDATION. Overlap between those is
    leakage by construction, not a judgement call.
    """

    splits: tuple[Split, ...] = ()

    def by_phase(self, phase: SplitPhase) -> tuple[Split, ...]:
        return tuple(s for s in self.splits if s.phase is phase)

    def get(self, name: str) -> Split:
        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(f"no split named {name!r}; known: {[s.name for s in self.splits]}")

    def phase_of(self, value: date | datetime | str) -> SplitPhase | None:
        """Which phase a timestamp belongs to, if any.

        Sealed takes precedence, so a date inside the seal can never be
        reported as TRAIN by an overlapping definition.
        """
        d = _as_date(value)
        for phase in (SplitPhase.SEALED_OOS, SplitPhase.TEST, SplitPhase.VALIDATION, SplitPhase.TRAIN):
            for split in self.by_phase(phase):
                if split.range.contains(d):
                    return phase
        return None

    def add(self, split: Split) -> "SplitRegistry":
        return SplitRegistry(splits=self.splits + (split,))

    @model_validator(mode="after")
    def _no_leaky_overlap(self) -> "SplitRegistry":
        names = [s.name for s in self.splits]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate split names: {sorted(duplicates)}")

        fit_phases = (SplitPhase.TRAIN, SplitPhase.VALIDATION)
        eval_phases = (SplitPhase.TEST, SplitPhase.SEALED_OOS)
        for fit in (s for s in self.splits if s.phase in fit_phases):
            for ev in (s for s in self.splits if s.phase in eval_phases):
                if fit.range.overlaps(ev.range):
                    raise ValueError(
                        f"leakage: {fit.name} ({fit.phase.value}) {fit.range} overlaps "
                        f"{ev.name} ({ev.phase.value}) {ev.range}"
                    )
        trains = self.by_phase(SplitPhase.TRAIN)
        vals = self.by_phase(SplitPhase.VALIDATION)
        for t in trains:
            for v in vals:
                if t.range.overlaps(v.range):
                    raise ValueError(
                        f"leakage: TRAIN {t.name} {t.range} overlaps VALIDATION {v.name} {v.range}"
                    )

        # TEST overlapping SEALED_OOS was accepted until this check existed.
        # It is incoherent rather than immediately leaky -- `DataAccessGuard`
        # still blocks the read, because it works on dates and not on phase
        # labels -- but it means a researcher who believes they hold two
        # years of TEST actually holds one, and discovers the rest only as a
        # SealedDataAccessError mid-run. `phase_of` reporting SEALED for the
        # contested dates papered over the contradiction instead of naming it.
        for ev in self.by_phase(SplitPhase.TEST):
            for sealed in self.by_phase(SplitPhase.SEALED_OOS):
                if ev.range.overlaps(sealed.range):
                    raise ValueError(
                        f"leakage: TEST {ev.name} {ev.range} overlaps SEALED_OOS "
                        f"{sealed.name} {sealed.range}. The holdout must be "
                        f"disjoint from every window that is evaluated more "
                        f"than once, or it is not untouched."
                    )

        self._assert_chronological()
        return self

    def _assert_chronological(self) -> None:
        """Later phases must come after earlier ones in wall-clock time.

        Non-overlap alone does not make a split honest: TRAIN [2020, 2021)
        with TEST [2010, 2011) does not overlap and is still a model fitted
        on the future and evaluated on the past. The brief is explicit --
        "chronological only" and "Never randomly mix future data into
        training" -- and this is the structural form of that rule, because
        Phase 8 builds these registries by arithmetic and a sign error in
        fold stepping produces exactly this shape with no other symptom.

        One registry describes one fold plus the global seal, which is why a
        global ordering is the right rule here. Across folds, a rolling
        walk-forward legitimately reuses an earlier fold's TEST window as a
        later fold's TRAIN data; that is a relationship between registries,
        not within one, and `_no_leaky_overlap` would reject it inside a
        single registry anyway.
        """
        rank = {
            SplitPhase.TRAIN: 0,
            SplitPhase.VALIDATION: 1,
            SplitPhase.TEST: 2,
            SplitPhase.SEALED_OOS: 3,
        }
        for earlier in self.splits:
            for later in self.splits:
                if rank[earlier.phase] >= rank[later.phase]:
                    continue
                if earlier.range.end > later.range.start:
                    raise ValueError(
                        f"non-chronological split: {earlier.name} "
                        f"({earlier.phase.value}) {earlier.range} must end at "
                        f"or before {later.name} ({later.phase.value}) "
                        f"{later.range} begins. Fitting on data that postdates "
                        f"the evaluation window inflates every result and the "
                        f"brief forbids it."
                    )

    def assert_embargo(self, min_gap_days: int) -> None:
        """Require a gap between each phase and the next.

        Separate from the always-on chronology check because the embargo
        length is a configured research choice (`walk_forward.embargo_days`,
        default 5) rather than a structural truth, and because this class
        does not read config. Phase 8 calls it when it builds folds.

        The reason an embargo is needed at all: a trade opened two days
        before a boundary and closed after it is scored in one window using
        price action from the next, so adjacent-but-touching ranges leak at
        the seam even though they do not overlap.
        """
        if min_gap_days <= 0:
            return
        rank = {
            SplitPhase.TRAIN: 0,
            SplitPhase.VALIDATION: 1,
            SplitPhase.TEST: 2,
            SplitPhase.SEALED_OOS: 3,
        }
        for earlier in self.splits:
            for later in self.splits:
                if rank[earlier.phase] + 1 != rank[later.phase]:
                    continue
                gap = (later.range.start - earlier.range.end).days
                if gap < min_gap_days:
                    raise ValueError(
                        f"embargo violated: {earlier.name} ends {earlier.range.end} "
                        f"and {later.name} starts {later.range.start}, a gap of "
                        f"{gap} day(s), below the required {min_gap_days}. A "
                        f"position held across the boundary would be scored in "
                        f"one window using the next window's prices."
                    )


class DataAccessGuard:
    """Gatekeeper for every historical data request.

    Usage (the data layer calls this; researchers normally never touch it):

        guard.check_access(start, end, purpose="backtest NQ 2016-2019")

    To run the single final confirmation:

        with guard.unseal("final OOS confirmation", experiment_id=..., approved_by="me"):
            ...
    """

    def __init__(
        self,
        sealed: DateRange | None,
        audit: SealAudit,
        max_opens: int = 1,
        enabled: bool = True,
        config_hash: str = "",
    ) -> None:
        self.sealed = sealed
        self.audit = audit
        self.max_opens = max_opens
        self.enabled = enabled and sealed is not None
        self.config_hash = config_hash
        self._active_token: SealAuditEntry | None = None

    # --- queries -------------------------------------------------------

    @property
    def is_unsealed(self) -> bool:
        return self._active_token is not None

    def opens_used(self) -> int:
        return self.audit.open_count()

    def opens_remaining(self) -> int:
        return max(0, self.max_opens - self.opens_used())

    def touches_seal(self, start: date | datetime | str, end: date | datetime | str) -> bool:
        if not self.enabled or self.sealed is None:
            return False
        requested = DateRange(start=_as_date(start), end=_as_date(end))
        return requested.overlaps(self.sealed)

    # --- enforcement ---------------------------------------------------

    def check_access(
        self,
        start: date | datetime | str,
        end: date | datetime | str,
        purpose: str = "",
    ) -> None:
        """Raise unless this request is allowed."""
        if not self.touches_seal(start, end):
            return
        if self.is_unsealed:
            return
        assert self.sealed is not None  # guarded by touches_seal
        raise SealedDataAccessError(
            f"request {_as_date(start)} -> {_as_date(end)} overlaps the sealed "
            f"out-of-sample window {self.sealed}. "
            f"Purpose given: {purpose or '(none)'}. "
            f"This window is reserved for a single final confirmation run. "
            f"If that is genuinely what you are doing, wrap the call in "
            f"guard.unseal(reason=..., experiment_id=..., approved_by=...). "
            f"Opens remaining: {self.opens_remaining()}/{self.max_opens}."
        )

    def allowed_ranges(
        self, start: date | datetime | str, end: date | datetime | str
    ) -> tuple[DateRange, ...]:
        """Every part of the request that is not sealed, in order.

        Returns up to two ranges, because a request can straddle the seal and
        have allowed data on both sides. `clip_to_allowed` cannot express
        that -- it returns a single range -- and used to resolve the straddle
        by returning the earlier side and silently dropping the later one. A
        sweep asking for "everything available" over [2010, 2026) with a seal
        at [2023, 2025) received [2010, 2023) and lost 2025 without a word.
        """
        requested = DateRange(start=_as_date(start), end=_as_date(end))
        if not self.enabled or self.sealed is None or not requested.overlaps(self.sealed):
            return (requested,)
        out: list[DateRange] = []
        if requested.start < self.sealed.start:
            out.append(DateRange(start=requested.start, end=self.sealed.start))
        if requested.end > self.sealed.end:
            out.append(DateRange(start=self.sealed.end, end=requested.end))
        return tuple(out)

    def clip_to_allowed(
        self, start: date | datetime | str, end: date | datetime | str
    ) -> DateRange | None:
        """Trim a request so it stops at the seal boundary.

        For sweeps that legitimately want "everything available": they get
        everything up to the seal, rather than an exception or the holdout.
        Returns None if nothing outside the seal remains.

        Raises when the request straddles the seal with allowed data on BOTH
        sides, because a single `DateRange` cannot represent the gap and
        every way of resolving it silently is wrong: returning one side
        discards real data, and returning the span re-admits the holdout.
        `allowed_ranges` handles that case explicitly.
        """
        parts = self.allowed_ranges(start, end)
        if not parts:
            return None
        if len(parts) > 1:
            raise SealedDataAccessError(
                f"request {_as_date(start)} -> {_as_date(end)} straddles the "
                f"sealed window {self.sealed}, leaving allowed data on both "
                f"sides ({', '.join(str(p) for p in parts)}). A single range "
                f"cannot express that gap. Call allowed_ranges() and load each "
                f"part, or request one side at a time."
            )
        return parts[0]

    @contextmanager
    def unseal(
        self,
        reason: str,
        experiment_id: str,
        approved_by: str,
        requested: DateRange | None = None,
    ) -> Iterator[SealAuditEntry]:
        """Open the seal for the duration of the block. Budgeted and audited."""
        if not self.enabled:
            yield SealAuditEntry(
                ts=datetime.now(tz=timezone.utc),
                reason=reason or "(seal disabled)",
                experiment_id=experiment_id,
                approved_by=approved_by,
            )
            return
        if self.is_unsealed:
            raise RuntimeError("the seal is already open in this context; do not nest unseals")
        if not reason.strip():
            raise ValueError("unsealing requires a non-empty reason; it goes in the audit log")
        if not experiment_id.strip():
            raise ValueError("unsealing requires an experiment_id so the run is attributable")
        used = self.opens_used()
        if used >= self.max_opens:
            previous = self.audit.entries()
            raise SealBudgetExhaustedError(
                f"the sealed window has already been opened {used} time(s), "
                f"budget is {self.max_opens}. Previous opens: "
                + "; ".join(f"{e.ts.date()} {e.experiment_id}: {e.reason}" for e in previous)
                + ". Opening it again would make the holdout an in-sample dataset. "
                "If you have genuinely redesigned the study, raise seal.max_opens "
                "deliberately and record why -- do not delete the audit log."
            )
        entry = self.audit.record(
            SealAuditEntry(
                ts=datetime.now(tz=timezone.utc),
                reason=reason,
                experiment_id=experiment_id,
                approved_by=approved_by,
                config_hash=self.config_hash,
                requested=requested or self.sealed,
            )
        )
        logger.warning(
            "SEALED OOS WINDOW OPENED (%d/%d): %s",
            entry.sequence,
            self.max_opens,
            reason,
            extra={"experiment_id": experiment_id, "approved_by": approved_by},
        )
        self._active_token = entry
        try:
            yield entry
        finally:
            self._active_token = None


def build_registry(
    train: tuple[date, date] | None = None,
    validation: tuple[date, date] | None = None,
    test: tuple[date, date] | None = None,
    sealed: tuple[date, date] | None = None,
) -> SplitRegistry:
    """Convenience builder for the common four-way split."""
    splits: list[Split] = []
    for name, phase, bounds in (
        ("train", SplitPhase.TRAIN, train),
        ("validation", SplitPhase.VALIDATION, validation),
        ("test", SplitPhase.TEST, test),
        ("sealed_oos", SplitPhase.SEALED_OOS, sealed),
    ):
        if bounds is not None:
            splits.append(
                Split(name=name, phase=phase, range=DateRange(start=bounds[0], end=bounds[1]))
            )
    return SplitRegistry(splits=tuple(splits))


def guard_from_config(config, audit_path: str | Path | None = None) -> DataAccessGuard:
    """Build the guard from a `FlowModelConfig`."""
    seal = config.seal
    sealed_range = (
        DateRange(start=seal.sealed_start, end=seal.sealed_end) if seal.enabled else None
    )
    return DataAccessGuard(
        sealed=sealed_range,
        audit=SealAudit(audit_path or seal.audit_path),
        max_opens=seal.max_opens,
        enabled=seal.enabled,
        config_hash=config.config_hash,
    )
