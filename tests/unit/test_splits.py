"""Date splits and the sealed out-of-sample guard.

These are the tests that make "keep a completely untouched final OOS
dataset" enforceable rather than aspirational.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from flow_model.config.loader import load_config
from flow_model.core.enums import SplitPhase
from flow_model.validation.splits import (
    DataAccessGuard,
    DateRange,
    SealAudit,
    SealBudgetExhaustedError,
    SealedDataAccessError,
    Split,
    SplitRegistry,
    build_registry,
    guard_from_config,
)


# --- DateRange -------------------------------------------------------------


def test_range_is_half_open():
    r = DateRange(start=date(2020, 1, 1), end=date(2021, 1, 1))
    assert r.contains(date(2020, 1, 1))
    assert not r.contains(date(2021, 1, 1))
    assert r.contains(date(2020, 12, 31))


def test_range_accepts_datetimes_and_iso_strings():
    r = DateRange(start=date(2020, 1, 1), end=date(2021, 1, 1))
    assert r.contains(datetime(2020, 6, 1, 14, 30, tzinfo=timezone.utc))
    assert r.contains("2020-06-01")
    assert r.contains("2020-06-01T14:30:00+00:00")


def test_empty_or_inverted_range_rejected():
    with pytest.raises(ValueError, match="must be after start"):
        DateRange(start=date(2021, 1, 1), end=date(2020, 1, 1))
    with pytest.raises(ValueError):
        DateRange(start=date(2020, 1, 1), end=date(2020, 1, 1))


def test_adjacent_ranges_do_not_overlap():
    a = DateRange(start=date(2019, 1, 1), end=date(2020, 1, 1))
    b = DateRange(start=date(2020, 1, 1), end=date(2021, 1, 1))
    assert not a.overlaps(b) and not b.overlaps(a)
    assert a.intersection(b) is None


def test_overlap_and_intersection():
    a = DateRange(start=date(2019, 1, 1), end=date(2021, 1, 1))
    b = DateRange(start=date(2020, 1, 1), end=date(2022, 1, 1))
    assert a.overlaps(b)
    assert a.intersection(b) == DateRange(start=date(2020, 1, 1), end=date(2021, 1, 1))


def test_days():
    assert DateRange(start=date(2020, 1, 1), end=date(2020, 1, 11)).days == 10


# --- registry --------------------------------------------------------------


def test_registry_phase_lookup():
    reg = build_registry(
        train=(date(2015, 1, 1), date(2019, 1, 1)),
        validation=(date(2019, 1, 1), date(2020, 1, 1)),
        test=(date(2020, 1, 1), date(2023, 1, 1)),
        sealed=(date(2023, 1, 1), date(2025, 1, 1)),
    )
    assert reg.phase_of(date(2017, 6, 1)) is SplitPhase.TRAIN
    assert reg.phase_of(date(2019, 6, 1)) is SplitPhase.VALIDATION
    assert reg.phase_of(date(2021, 6, 1)) is SplitPhase.TEST
    assert reg.phase_of(date(2024, 6, 1)) is SplitPhase.SEALED_OOS
    assert reg.phase_of(date(2030, 1, 1)) is None


def test_registry_rejects_train_overlapping_sealed():
    """Overlap between a fitting window and the holdout is leakage by
    construction, so it cannot be expressed."""
    with pytest.raises(ValueError, match="leakage"):
        build_registry(
            train=(date(2015, 1, 1), date(2024, 1, 1)),
            sealed=(date(2023, 1, 1), date(2025, 1, 1)),
        )


def test_registry_rejects_train_overlapping_test():
    with pytest.raises(ValueError, match="leakage"):
        build_registry(
            train=(date(2015, 1, 1), date(2021, 1, 1)),
            test=(date(2020, 1, 1), date(2023, 1, 1)),
        )


def test_registry_rejects_train_overlapping_validation():
    with pytest.raises(ValueError, match="leakage"):
        build_registry(
            train=(date(2015, 1, 1), date(2020, 1, 1)),
            validation=(date(2019, 1, 1), date(2021, 1, 1)),
        )


def test_registry_rejects_duplicate_names():
    r = DateRange(start=date(2015, 1, 1), end=date(2016, 1, 1))
    with pytest.raises(ValueError, match="duplicate split names"):
        SplitRegistry(splits=(
            Split(name="a", phase=SplitPhase.TRAIN, range=r),
            Split(name="a", phase=SplitPhase.TRAIN, range=r),
        ))


def test_sealed_phase_wins_lookup_ties():
    """If definitions ever overlap, a sealed date must never read as TRAIN."""
    reg = SplitRegistry(splits=(
        Split(name="sealed_oos", phase=SplitPhase.SEALED_OOS,
              range=DateRange(start=date(2023, 1, 1), end=date(2025, 1, 1))),
        Split(name="test2", phase=SplitPhase.TEST,
              range=DateRange(start=date(2022, 1, 1), end=date(2024, 1, 1))),
    ))
    assert reg.phase_of(date(2023, 6, 1)) is SplitPhase.SEALED_OOS


def test_registry_get_and_add():
    reg = build_registry(train=(date(2015, 1, 1), date(2016, 1, 1)))
    assert reg.get("train").phase is SplitPhase.TRAIN
    with pytest.raises(KeyError):
        reg.get("nope")
    bigger = reg.add(Split(name="t2", phase=SplitPhase.TEST,
                           range=DateRange(start=date(2017, 1, 1), end=date(2018, 1, 1))))
    assert len(bigger.splits) == 2


# --- the seal --------------------------------------------------------------


@pytest.fixture
def guard(tmp_path):
    return DataAccessGuard(
        sealed=DateRange(start=date(2023, 1, 1), end=date(2025, 1, 1)),
        audit=SealAudit(tmp_path / "seal_audit.jsonl"),
        max_opens=1,
    )


def test_non_sealed_access_is_allowed(guard):
    guard.check_access(date(2016, 1, 1), date(2019, 1, 1), purpose="training")


def test_access_stopping_at_the_seal_boundary_is_allowed(guard):
    guard.check_access(date(2016, 1, 1), date(2023, 1, 1), purpose="all usable data")


def test_a_sweep_spanning_the_whole_decade_is_blocked(guard):
    """The realistic contamination path: a sweep over 2015-2025."""
    with pytest.raises(SealedDataAccessError, match="sealed out-of-sample window"):
        guard.check_access(date(2015, 1, 1), date(2025, 1, 1), purpose="parameter sweep")


def test_access_wholly_inside_the_seal_is_blocked(guard):
    with pytest.raises(SealedDataAccessError):
        guard.check_access(date(2023, 6, 1), date(2024, 6, 1))


def test_error_message_names_the_remedy_and_the_budget(guard):
    with pytest.raises(SealedDataAccessError) as excinfo:
        guard.check_access(date(2024, 1, 1), date(2024, 6, 1), purpose="curiosity")
    text = str(excinfo.value)
    assert "guard.unseal" in text
    assert "Opens remaining: 1/1" in text
    assert "curiosity" in text


def test_clip_to_allowed_trims_at_the_boundary(guard):
    clipped = guard.clip_to_allowed(date(2015, 1, 1), date(2025, 1, 1))
    assert clipped == DateRange(start=date(2015, 1, 1), end=date(2023, 1, 1))


def test_clip_to_allowed_returns_none_when_fully_sealed(guard):
    assert guard.clip_to_allowed(date(2023, 6, 1), date(2024, 1, 1)) is None


def test_clip_to_allowed_passes_through_clean_ranges(guard):
    r = guard.clip_to_allowed(date(2016, 1, 1), date(2017, 1, 1))
    assert r == DateRange(start=date(2016, 1, 1), end=date(2017, 1, 1))


def test_unseal_permits_access_and_closes_afterwards(guard):
    with guard.unseal("final confirmation", experiment_id="exp_1", approved_by="me"):
        assert guard.is_unsealed
        guard.check_access(date(2023, 1, 1), date(2025, 1, 1))
    assert not guard.is_unsealed
    with pytest.raises(SealedDataAccessError):
        guard.check_access(date(2023, 1, 1), date(2025, 1, 1))


def test_unseal_is_budgeted(guard):
    with guard.unseal("first", experiment_id="exp_1", approved_by="me"):
        pass
    with pytest.raises(SealBudgetExhaustedError, match="already been opened"):
        with guard.unseal("second", experiment_id="exp_2", approved_by="me"):
            pass


def test_budget_survives_a_fresh_process(tmp_path):
    """Re-running the script must not reset the budget."""
    audit_path = tmp_path / "audit.jsonl"
    sealed = DateRange(start=date(2023, 1, 1), end=date(2025, 1, 1))
    first = DataAccessGuard(sealed=sealed, audit=SealAudit(audit_path), max_opens=1)
    with first.unseal("final", experiment_id="exp_1", approved_by="me"):
        pass
    second = DataAccessGuard(sealed=sealed, audit=SealAudit(audit_path), max_opens=1)
    assert second.opens_used() == 1
    assert second.opens_remaining() == 0
    with pytest.raises(SealBudgetExhaustedError):
        with second.unseal("sneak", experiment_id="exp_2", approved_by="me"):
            pass


def test_unseal_requires_a_reason_and_an_experiment_id(guard):
    with pytest.raises(ValueError, match="non-empty reason"):
        with guard.unseal("   ", experiment_id="exp_1", approved_by="me"):
            pass
    with pytest.raises(ValueError, match="experiment_id"):
        with guard.unseal("reason", experiment_id="", approved_by="me"):
            pass
    assert guard.opens_used() == 0      # rejected attempts are not charged


def test_unseal_cannot_be_nested(guard):
    with guard.unseal("outer", experiment_id="exp_1", approved_by="me"):
        with pytest.raises(RuntimeError, match="already open"):
            with guard.unseal("inner", experiment_id="exp_2", approved_by="me"):
                pass


def test_seal_closes_even_if_the_block_raises(guard):
    with pytest.raises(RuntimeError, match="boom"):
        with guard.unseal("final", experiment_id="exp_1", approved_by="me"):
            raise RuntimeError("boom")
    assert not guard.is_unsealed


def test_audit_records_who_what_and_why(guard):
    with guard.unseal("final OOS run", experiment_id="exp_42", approved_by="researcher"):
        pass
    entries = guard.audit.entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.reason == "final OOS run"
    assert entry.experiment_id == "exp_42"
    assert entry.approved_by == "researcher"
    assert entry.sequence == 1
    assert entry.pid > 0


def test_corrupt_audit_line_raises_rather_than_being_skipped(tmp_path):
    """A lost audit line is a lost open; silence is the wrong failure mode."""
    path = tmp_path / "audit.jsonl"
    path.write_text('{"not": "an entry"\n')
    with pytest.raises(RuntimeError, match="not a readable seal audit entry"):
        SealAudit(path).entries()


def test_disabled_seal_allows_everything(tmp_path):
    guard = DataAccessGuard(sealed=None, audit=SealAudit(tmp_path / "a.jsonl"), enabled=False)
    guard.check_access(date(2015, 1, 1), date(2030, 1, 1))
    assert not guard.touches_seal(date(2024, 1, 1), date(2024, 6, 1))
    with guard.unseal("noop", experiment_id="x", approved_by="y"):
        pass
    assert guard.opens_used() == 0


def test_guard_from_config(tmp_path):
    config = load_config(overrides=[f"seal.audit_path={tmp_path}/a.jsonl"])
    guard = guard_from_config(config)
    assert guard.sealed == DateRange(start=config.seal.sealed_start, end=config.seal.sealed_end)
    assert guard.max_opens == config.seal.max_opens
    assert guard.config_hash == config.config_hash
    with pytest.raises(SealedDataAccessError):
        guard.check_access(config.backtest.start, config.backtest.end, purpose="full backtest")


def test_default_config_seal_covers_the_last_two_years(config):
    assert config.seal.enabled
    assert (config.seal.sealed_end - config.seal.sealed_start).days >= 365
    assert config.seal.sealed_end == config.backtest.end
