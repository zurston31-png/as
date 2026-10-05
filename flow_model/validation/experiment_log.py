"""Experiment log.

Overfitting is a *process* failure, not a statistical one: it happens when a
researcher tries many things against the same evaluation data and reports
the best. You cannot detect that from the final backtest. You can only
detect it from a record of how many things were tried.

So every run writes a row here: what config, what data, which phase, what
changed, and *why*. The Phase 8 overfitting report is generated from this
table. Two mechanisms do real work:

  * `params_changed` requires a non-empty `reason` -- a parameter change with
    no stated rationale is rejected.
  * `assert_phase_budget` raises once a config family has been evaluated
    against TEST or SEALED_OOS more times than allowed, which is the
    mechanical form of "do not optimize repeatedly against the same test set".
"""

from __future__ import annotations

import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import Field

from flow_model.core.determinism import stable_hash
from flow_model.utils.serialization import dumps as json_dumps
from flow_model.utils.serialization import loads as json_loads
from flow_model.core.enums import SplitPhase
from flow_model.core.model import FrozenModel
from flow_model.utils.logging import get_logger

logger = get_logger("validation.experiment_log")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id        TEXT PRIMARY KEY,
    sequence             INTEGER NOT NULL,
    created_at           TEXT    NOT NULL,
    name                 TEXT    NOT NULL,
    git_commit           TEXT    NOT NULL DEFAULT '',
    git_dirty            INTEGER NOT NULL DEFAULT 0,
    config_hash          TEXT    NOT NULL,
    config_json          TEXT    NOT NULL DEFAULT '',
    dataset_id           TEXT    NOT NULL DEFAULT '',
    data_hash            TEXT    NOT NULL DEFAULT '',
    split_id             TEXT    NOT NULL DEFAULT '',
    phase                TEXT    NOT NULL,
    params_changed_json  TEXT    NOT NULL DEFAULT '{}',
    reason               TEXT    NOT NULL DEFAULT '',
    results_json         TEXT    NOT NULL DEFAULT '{}',
    notes                TEXT    NOT NULL DEFAULT '',
    parent_experiment_id TEXT,
    seal_opened          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_experiments_config ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_experiments_phase  ON experiments(phase);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
"""

_EVALUATION_PHASES = (SplitPhase.TEST, SplitPhase.SEALED_OOS)


class ExperimentBudgetError(RuntimeError):
    """Raised when a config family has exhausted its evaluation-phase budget."""


class ExperimentRecord(FrozenModel):
    """One logged experiment."""

    experiment_id: str
    sequence: int = 0
    created_at: datetime
    name: str
    config_hash: str
    phase: SplitPhase
    git_commit: str = ""
    git_dirty: bool = False
    config_json: str = ""
    dataset_id: str = ""
    data_hash: str = ""
    split_id: str = ""
    params_changed: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    results: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    parent_experiment_id: str | None = None
    seal_opened: bool = False

    @property
    def is_evaluation(self) -> bool:
        return self.phase in _EVALUATION_PHASES


def git_state(repo_root: str | Path = ".") -> tuple[str, bool]:
    """(commit_sha, is_dirty). Empty sha when git is unavailable.

    Recorded because a result is only reproducible if you know which code
    produced it; `git_dirty=True` is a warning that the code was uncommitted.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return sha, bool(status)
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return "", False


class ExperimentLog:
    """SQLite-backed experiment registry."""

    def __init__(self, path: str | Path, repo_root: str | Path = ".") -> None:
        self.path = Path(path)
        self.repo_root = Path(repo_root)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writing -------------------------------------------------------

    def log(
        self,
        name: str,
        config_hash: str,
        phase: SplitPhase,
        *,
        config: Any = None,
        dataset_id: str = "",
        data_hash: str = "",
        split_id: str = "",
        params_changed: Mapping[str, Any] | None = None,
        reason: str = "",
        results: Mapping[str, Any] | None = None,
        notes: str = "",
        parent_experiment_id: str | None = None,
        seal_opened: bool = False,
    ) -> ExperimentRecord:
        """Append an experiment. Returns the stored record."""
        params = dict(params_changed or {})
        if params and not reason.strip():
            raise ValueError(
                "a parameter change must state a reason. Changing parameters "
                "without recording why is how an untracked optimization loop "
                f"forms. Changed: {sorted(params)}"
            )
        sha, dirty = git_state(self.repo_root)
        sequence = self.count() + 1
        created = datetime.now(tz=timezone.utc)
        experiment_id = "exp_{:05d}_{}".format(
            sequence,
            stable_hash(
                [config_hash, phase.value, params, created.isoformat(), uuid.uuid4().hex],
                length=8,
            ),
        )
        record = ExperimentRecord(
            experiment_id=experiment_id,
            sequence=sequence,
            created_at=created,
            name=name,
            config_hash=config_hash,
            phase=phase,
            git_commit=sha,
            git_dirty=dirty,
            config_json=json_dumps(config) if config is not None else "",
            dataset_id=dataset_id,
            data_hash=data_hash,
            split_id=split_id,
            params_changed=params,
            reason=reason,
            results=dict(results or {}),
            notes=notes,
            parent_experiment_id=parent_experiment_id,
            seal_opened=seal_opened,
        )
        self._conn.execute(
            """INSERT INTO experiments (
                   experiment_id, sequence, created_at, name, git_commit, git_dirty,
                   config_hash, config_json, dataset_id, data_hash, split_id, phase,
                   params_changed_json, reason, results_json, notes,
                   parent_experiment_id, seal_opened)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.experiment_id,
                record.sequence,
                record.created_at.isoformat(),
                record.name,
                record.git_commit,
                int(record.git_dirty),
                record.config_hash,
                record.config_json,
                record.dataset_id,
                record.data_hash,
                record.split_id,
                record.phase.value,
                json_dumps(record.params_changed),
                record.reason,
                json_dumps(record.results),
                record.notes,
                record.parent_experiment_id,
                int(record.seal_opened),
            ),
        )
        self._conn.commit()
        logger.info(
            "logged experiment %s (%s, phase=%s)",
            record.experiment_id,
            name,
            phase.value,
            extra={"config_hash": config_hash, "dirty": dirty},
        )
        if record.git_dirty:
            logger.warning(
                "experiment %s was run from a dirty working tree; it is not "
                "reproducible from git alone",
                record.experiment_id,
            )
        return record

    def update_results(
        self, experiment_id: str, results: Mapping[str, Any], notes: str | None = None
    ) -> ExperimentRecord:
        """Attach results to an already-logged experiment."""
        existing = self.get(experiment_id)
        merged = {**existing.results, **dict(results)}
        self._conn.execute(
            "UPDATE experiments SET results_json = ?, notes = ? WHERE experiment_id = ?",
            (json_dumps(merged), notes if notes is not None else existing.notes, experiment_id),
        )
        self._conn.commit()
        return self.get(experiment_id)

    # --- reading -------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            sequence=row["sequence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            name=row["name"],
            config_hash=row["config_hash"],
            phase=SplitPhase(row["phase"]),
            git_commit=row["git_commit"],
            git_dirty=bool(row["git_dirty"]),
            config_json=row["config_json"],
            dataset_id=row["dataset_id"],
            data_hash=row["data_hash"],
            split_id=row["split_id"],
            params_changed=json_loads(row["params_changed_json"]) or {},
            reason=row["reason"],
            results=json_loads(row["results_json"]) or {},
            notes=row["notes"],
            parent_experiment_id=row["parent_experiment_id"],
            seal_opened=bool(row["seal_opened"]),
        )

    def get(self, experiment_id: str) -> ExperimentRecord:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no experiment {experiment_id!r}")
        return self._row_to_record(row)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])

    def query(
        self,
        config_hash: str | None = None,
        phase: SplitPhase | None = None,
        limit: int | None = None,
    ) -> list[ExperimentRecord]:
        sql = "SELECT * FROM experiments WHERE 1=1"
        args: list[Any] = []
        if config_hash is not None:
            sql += " AND config_hash = ?"
            args.append(config_hash)
        if phase is not None:
            sql += " AND phase = ?"
            args.append(phase.value)
        sql += " ORDER BY sequence ASC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        return [self._row_to_record(r) for r in self._conn.execute(sql, args)]

    def lineage(self, experiment_id: str) -> list[ExperimentRecord]:
        """Chain from the root ancestor to `experiment_id`."""
        chain: list[ExperimentRecord] = []
        seen: set[str] = set()
        current: str | None = experiment_id
        while current:
            if current in seen:
                raise RuntimeError(f"cycle in experiment lineage at {current}")
            seen.add(current)
            record = self.get(current)
            chain.append(record)
            current = record.parent_experiment_id
        return list(reversed(chain))

    # --- anti-overfitting ----------------------------------------------

    def phase_touches(self, config_hash: str, phase: SplitPhase) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE config_hash = ? AND phase = ?",
                (config_hash, phase.value),
            ).fetchone()[0]
        )

    def total_phase_touches(self, phase: SplitPhase) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE phase = ?", (phase.value,)
            ).fetchone()[0]
        )

    def assert_phase_budget(
        self, config_hash: str, phase: SplitPhase, max_touches: int = 1
    ) -> None:
        """Raise if this config has already been evaluated on `phase` too often.

        The mechanical form of "do not optimize parameters repeatedly against
        the same test set".
        """
        if phase not in _EVALUATION_PHASES:
            return
        used = self.phase_touches(config_hash, phase)
        if used >= max_touches:
            previous = self.query(config_hash=config_hash, phase=phase)
            raise ExperimentBudgetError(
                f"config {config_hash} has already been evaluated on {phase.value} "
                f"{used} time(s) (budget {max_touches}). Prior runs: "
                + "; ".join(f"{e.experiment_id} ({e.created_at.date()}) {e.name}" for e in previous)
                + f". Repeated evaluation on {phase.value} turns it into a "
                "training set. Evaluate on TRAIN/VALIDATION instead."
            )

    def distinct_configs(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT config_hash) FROM experiments"
            ).fetchone()[0]
        )

    def overfitting_summary(self) -> dict[str, Any]:
        """Inputs for the Phase 8 overfitting report.

        `effective_trials` is the count of distinct configurations ever
        evaluated against TEST or SEALED_OOS -- the multiple-comparisons
        denominator. A best-of-N result needs a deflated significance
        threshold, and N comes from here, not from memory.
        """
        rows = self._conn.execute(
            """SELECT phase, COUNT(*) AS n, COUNT(DISTINCT config_hash) AS configs
               FROM experiments GROUP BY phase"""
        ).fetchall()
        by_phase = {r["phase"]: {"runs": r["n"], "distinct_configs": r["configs"]} for r in rows}
        eval_configs = int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT config_hash) FROM experiments WHERE phase IN (?,?)",
                (SplitPhase.TEST.value, SplitPhase.SEALED_OOS.value),
            ).fetchone()[0]
        )
        dirty = int(
            self._conn.execute("SELECT COUNT(*) FROM experiments WHERE git_dirty = 1").fetchone()[0]
        )
        unexplained = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM experiments "
                "WHERE params_changed_json NOT IN ('{}','') AND TRIM(reason) = ''"
            ).fetchone()[0]
        )
        return {
            "total_experiments": self.count(),
            "distinct_configs": self.distinct_configs(),
            "by_phase": by_phase,
            "effective_trials_on_evaluation_data": eval_configs,
            "seal_opens_logged": int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM experiments WHERE seal_opened = 1"
                ).fetchone()[0]
            ),
            "runs_from_dirty_tree": dirty,
            "param_changes_without_reason": unexplained,
        }

    def to_rows(self, records: Iterable[ExperimentRecord] | None = None) -> list[dict[str, Any]]:
        """Flat rows for reporting."""
        source = records if records is not None else self.query()
        return [
            {
                "experiment_id": r.experiment_id,
                "created_at": r.created_at.isoformat(),
                "name": r.name,
                "phase": r.phase.value,
                "config_hash": r.config_hash,
                "dataset_id": r.dataset_id,
                "params_changed": json_dumps(r.params_changed),
                "reason": r.reason,
                "results": json_dumps(r.results),
                "git_commit": r.git_commit[:8],
                "git_dirty": r.git_dirty,
            }
            for r in source
        ]
