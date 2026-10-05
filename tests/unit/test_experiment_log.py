"""Experiment log -- the mechanical part of the anti-overfitting rules."""

from __future__ import annotations

import pytest

from flow_model.config.loader import load_config
from flow_model.core.enums import SplitPhase
from flow_model.validation.experiment_log import (
    ExperimentBudgetError,
    ExperimentLog,
    git_state,
)


@pytest.fixture
def log(tmp_path):
    with ExperimentLog(tmp_path / "experiments.db", repo_root=tmp_path) as instance:
        yield instance


def test_log_and_retrieve(log, config):
    record = log.log("baseline", config.config_hash, SplitPhase.TRAIN,
                     dataset_id="NQ_5m", results={"trades": 0})
    assert log.get(record.experiment_id) == record
    assert log.count() == 1


def test_ids_are_unique_and_sequential(log, config):
    ids = [log.log(f"r{i}", config.config_hash, SplitPhase.TRAIN).experiment_id for i in range(5)]
    assert len(set(ids)) == 5
    assert [log.get(i).sequence for i in ids] == [1, 2, 3, 4, 5]


def test_results_keep_their_numeric_types(log, config):
    """canonical_json (for hashing) stringifies floats; storage must not."""
    record = log.log("r", config.config_hash, SplitPhase.TRAIN,
                     results={"win_rate": 0.7234, "trades": 412, "pf": 1.63})
    stored = log.get(record.experiment_id).results
    assert stored["win_rate"] == pytest.approx(0.7234)
    assert isinstance(stored["win_rate"], float)
    assert isinstance(stored["trades"], int)


def test_update_results_merges(log, config):
    record = log.log("r", config.config_hash, SplitPhase.TRAIN, results={"a": 1})
    log.update_results(record.experiment_id, {"b": 2}, notes="second pass")
    refreshed = log.get(record.experiment_id)
    assert refreshed.results == {"a": 1, "b": 2}
    assert refreshed.notes == "second pass"


def test_parameter_change_without_a_reason_is_rejected(log, config):
    """An untracked optimization loop starts exactly here."""
    with pytest.raises(ValueError, match="must state a reason"):
        log.log("tweak", config.config_hash, SplitPhase.TRAIN,
                params_changed={"setups.SCALP_1R.min_flow_score": [70, 75]})


def test_parameter_change_with_a_reason_is_accepted(log, config):
    record = log.log("tweak", config.config_hash, SplitPhase.TRAIN,
                     params_changed={"x": [1, 2]},
                     reason="too many low-conviction entries in training folds")
    assert record.params_changed == {"x": [1, 2]}
    assert record.reason


def test_lineage_walks_to_the_root(log, config):
    a = log.log("a", config.config_hash, SplitPhase.TRAIN)
    b = log.log("b", config.config_hash, SplitPhase.TRAIN, parent_experiment_id=a.experiment_id)
    c = log.log("c", config.config_hash, SplitPhase.TRAIN, parent_experiment_id=b.experiment_id)
    chain = [r.experiment_id for r in log.lineage(c.experiment_id)]
    assert chain == [a.experiment_id, b.experiment_id, c.experiment_id]


def test_test_set_budget_blocks_a_second_evaluation(log, config):
    log.assert_phase_budget(config.config_hash, SplitPhase.TEST, max_touches=1)
    log.log("first test run", config.config_hash, SplitPhase.TEST)
    with pytest.raises(ExperimentBudgetError, match="turns it into a"):
        log.assert_phase_budget(config.config_hash, SplitPhase.TEST, max_touches=1)


def test_budget_is_per_config_not_global(log, config):
    other = load_config(overrides=["setups.SCALP_1R.min_flow_score=75"])
    log.log("run A", config.config_hash, SplitPhase.TEST)
    # A different config has its own budget -- but the count of distinct
    # configs tried is what the overfitting report surfaces.
    log.assert_phase_budget(other.config_hash, SplitPhase.TEST, max_touches=1)


def test_training_phases_are_never_budgeted(log, config):
    for _ in range(10):
        log.log("sweep", config.config_hash, SplitPhase.TRAIN)
    assert log.assert_phase_budget(config.config_hash, SplitPhase.TRAIN, max_touches=1) is None
    assert log.assert_phase_budget(config.config_hash, SplitPhase.VALIDATION, max_touches=1) is None


def test_sealed_phase_is_budgeted(log, config):
    log.log("final", config.config_hash, SplitPhase.SEALED_OOS, seal_opened=True)
    with pytest.raises(ExperimentBudgetError):
        log.assert_phase_budget(config.config_hash, SplitPhase.SEALED_OOS, max_touches=1)


def test_phase_touch_counts(log, config):
    log.log("a", config.config_hash, SplitPhase.TRAIN)
    log.log("b", config.config_hash, SplitPhase.TEST)
    log.log("c", config.config_hash, SplitPhase.TEST)
    assert log.phase_touches(config.config_hash, SplitPhase.TEST) == 2
    assert log.total_phase_touches(SplitPhase.TRAIN) == 1


def test_overfitting_summary_counts_effective_trials(log, config):
    """The multiple-comparisons denominator must come from the record, not memory."""
    for threshold in (60, 65, 70, 75, 80, 85):
        candidate = load_config(overrides=[f"setups.SCALP_1R.min_flow_score={threshold}"])
        log.log(f"sweep {threshold}", candidate.config_hash, SplitPhase.TRAIN)
        log.log(f"eval {threshold}", candidate.config_hash, SplitPhase.TEST)
    summary = log.overfitting_summary()
    assert summary["effective_trials_on_evaluation_data"] == 6
    assert summary["distinct_configs"] == 6
    assert summary["by_phase"]["TEST"]["runs"] == 6


def test_overfitting_summary_flags_dirty_trees_and_unexplained_changes(log, config):
    log.log("clean", config.config_hash, SplitPhase.TRAIN)
    summary = log.overfitting_summary()
    assert "runs_from_dirty_tree" in summary
    assert summary["param_changes_without_reason"] == 0


def test_query_filters(log, config):
    log.log("a", config.config_hash, SplitPhase.TRAIN)
    log.log("b", config.config_hash, SplitPhase.TEST)
    assert len(log.query(phase=SplitPhase.TEST)) == 1
    assert len(log.query(config_hash=config.config_hash)) == 2
    assert len(log.query(limit=1)) == 1
    assert log.query(config_hash="nonexistent") == []


def test_missing_experiment_raises(log):
    with pytest.raises(KeyError):
        log.get("exp_does_not_exist")


def test_persistence_across_connections(tmp_path, config):
    path = tmp_path / "e.db"
    with ExperimentLog(path, repo_root=tmp_path) as first:
        first.log("a", config.config_hash, SplitPhase.TRAIN)
    with ExperimentLog(path, repo_root=tmp_path) as second:
        assert second.count() == 1


def test_config_json_is_readable_back(log, config):
    import json

    record = log.log("a", config.config_hash, SplitPhase.TRAIN,
                     config=config.to_init_dict(mode="json"))
    restored = json.loads(log.get(record.experiment_id).config_json)
    assert restored["risk"]["risk_per_trade_pct"] == pytest.approx(0.005)


def test_git_state_handles_a_non_repo(tmp_path):
    sha, dirty = git_state(tmp_path)
    assert isinstance(sha, str) and isinstance(dirty, bool)


def test_to_rows_is_flat(log, config):
    log.log("a", config.config_hash, SplitPhase.TRAIN, results={"x": 1.0})
    row = log.to_rows()[0]
    assert set(row) >= {"experiment_id", "phase", "config_hash", "results", "git_commit"}
    assert all(not isinstance(v, (dict, list)) for v in row.values())
