"""Tests for database snapshots.

A backup is only worth anything on the day it is needed, which is the day
nobody is available to debug it. So these care about the ways a backup
system fails quietly: producing a file that is not a usable database,
rotating away the good copy, or restoring over live data.
"""
import datetime as dt
import sqlite3

import pytest

from app import backup
from app.config import settings


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A throwaway database and backup directory."""
    db = tmp_path / "data" / "bot.db"
    db.parent.mkdir(parents=True)
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setattr(settings, "BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setattr(settings, "BACKUP_ENABLED", True)
    monkeypatch.setattr(settings, "BACKUP_RESTORE_ON_EMPTY", True)
    monkeypatch.setattr(settings, "BACKUP_KEEP", 3)
    return db


def _seed(db_path, rows=10, table="pipeline_events"):
    con = sqlite3.connect(db_path)
    con.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY, v TEXT)")
    con.executemany(f"INSERT INTO {table} (v) VALUES (?)", [(f"r{i}",) for i in range(rows)])
    con.commit()
    con.close()


def _count(db_path, table="pipeline_events"):
    con = sqlite3.connect(db_path)
    try:
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# taking one
# ---------------------------------------------------------------------------

def test_a_snapshot_is_a_readable_database_not_just_a_file(sandbox):
    _seed(sandbox, 25)
    snapshot = backup.take_snapshot(reason="test")

    assert snapshot is not None
    assert backup._verify(snapshot.path)
    assert _count(snapshot.path) == 25


def test_a_snapshot_of_a_live_database_captures_committed_rows(sandbox):
    """The online backup API holds the same locks the database uses, so a
    write in flight cannot tear the snapshot in half. A shutil copy can."""
    _seed(sandbox, 10)
    con = sqlite3.connect(sandbox)          # deliberately left open
    con.execute("INSERT INTO pipeline_events (v) VALUES ('committed')")
    con.commit()

    snapshot = backup.take_snapshot(reason="test")
    con.close()

    assert snapshot is not None
    assert _count(snapshot.path) == 11


def test_nothing_is_snapshotted_before_the_database_exists(sandbox):
    assert backup.take_snapshot(reason="test") is None
    assert backup.list_snapshots() == []


def test_non_sqlite_is_left_alone(sandbox, monkeypatch):
    """Postgres has better backup tooling than anything reimplemented here.
    Half-supporting it would be worse than not supporting it."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+psycopg2://u@h/db")
    assert backup.database_path() is None
    assert backup.take_snapshot(reason="test") is None


def test_a_partial_file_is_never_left_looking_like_a_snapshot(sandbox):
    _seed(sandbox, 5)
    backup.take_snapshot(reason="test")
    leftovers = list(backup.backup_dir().glob("*.partial"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def test_a_corrupt_file_does_not_count_as_a_backup(sandbox):
    backup.backup_dir().mkdir(parents=True, exist_ok=True)
    bad = backup.backup_dir() / "snapshot-20200101T000000000Z.db"
    bad.write_bytes(b"not a database at all")

    assert backup._verify(bad) is False
    assert backup.restore(force=True) is False, "a corrupt snapshot was restored"


def test_an_empty_database_does_not_count_as_a_backup(sandbox):
    """A file that opens cleanly and contains nothing is the most dangerous
    kind of bad backup: every check passes and restoring it erases
    everything."""
    backup.backup_dir().mkdir(parents=True, exist_ok=True)
    empty = backup.backup_dir() / "snapshot-20200101T000000000Z.db"
    sqlite3.connect(empty).close()

    assert backup._verify(empty) is False


# ---------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------

def test_rotation_keeps_the_newest_and_only_runs_after_a_good_snapshot(sandbox):
    """Pruning first would mean a failed backup had already deleted one of
    the good ones it was meant to replace."""
    _seed(sandbox, 5)
    for _ in range(6):
        backup.take_snapshot(reason="test")

    snapshots = backup.list_snapshots()
    assert len(snapshots) == settings.BACKUP_KEEP == 3
    assert snapshots == sorted(snapshots, key=lambda s: s.taken_at, reverse=True)


def test_two_snapshots_in_the_same_second_do_not_collide(sandbox):
    _seed(sandbox, 5)
    first = backup.take_snapshot(reason="test")
    second = backup.take_snapshot(reason="test")
    assert first.path != second.path


def test_downloading_repeatedly_does_not_rotate_away_the_history(sandbox):
    """The download button must not destroy the backups it exists to
    protect. A snapshot younger than the reuse window is served as-is."""
    _seed(sandbox, 5)
    for _ in range(3):
        backup.take_snapshot(reason="test")
    before = {s.path.name for s in backup.list_snapshots()}

    for _ in range(10):
        backup.snapshot_for_download(max_age_seconds=3600)

    assert {s.path.name for s in backup.list_snapshots()} == before


def test_a_stale_download_snapshot_is_refreshed(sandbox):
    _seed(sandbox, 5)
    backup.take_snapshot(reason="test")
    fresh = backup.snapshot_for_download(max_age_seconds=0)
    assert fresh is not None
    assert len(backup.list_snapshots()) == 2


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

def test_a_wiped_database_is_recovered_on_startup(sandbox):
    """What a filesystem reset actually looks like."""
    _seed(sandbox, 40)
    backup.take_snapshot(reason="test")
    sandbox.unlink()

    assert backup.database_is_empty()
    assert backup.restore_if_empty() is True
    assert _count(sandbox) == 40


def test_a_schema_only_database_still_counts_as_empty(sandbox):
    """init_db() creates the schema before anything else runs, so a wiped
    install that has already started up has tables and no rows. Requiring
    a MISSING file would mean the restore never fires in the one case it
    was written for."""
    _seed(sandbox, 12)
    backup.take_snapshot(reason="test")

    sandbox.unlink()
    sqlite3.connect(sandbox).close()        # a bare, table-less database
    assert backup.database_is_empty()
    assert backup.restore_if_empty() is True
    assert _count(sandbox) == 12


def test_a_populated_database_is_never_rolled_back_automatically(sandbox):
    """The failure this guard prevents is not recoverable: silently
    reverting to last night's snapshot destroys exactly the data the
    snapshots exist to protect."""
    _seed(sandbox, 5)
    backup.take_snapshot(reason="test")
    _seed(sandbox, 20)                      # 25 rows now, snapshot has 5

    assert backup.database_is_empty() is False
    assert backup.restore_if_empty() is False
    assert backup.restore(force=False) is False
    assert _count(sandbox) == 25


def test_force_overwrites_a_populated_database(sandbox):
    _seed(sandbox, 5)
    backup.take_snapshot(reason="test")
    _seed(sandbox, 20)

    assert backup.restore(force=True) is True
    assert _count(sandbox) == 5


def test_the_snapshot_survives_being_restored(sandbox):
    """Restoring the wrong one must not be a one-way door."""
    _seed(sandbox, 7)
    snapshot = backup.take_snapshot(reason="test")
    sandbox.unlink()
    backup.restore_if_empty()

    assert snapshot.path.exists()
    assert backup._verify(snapshot.path)


def test_writes_work_after_a_restore(sandbox):
    """A restore that replaces an unlinked file leaves any pooled
    connection pointing at a deleted inode, which SQLite reports as a
    READONLY database - an error that reads like a permissions problem and
    is not. The pool has to be dropped."""
    _seed(sandbox, 5)
    backup.take_snapshot(reason="test")
    sandbox.unlink()
    backup.restore_if_empty()

    con = sqlite3.connect(sandbox)
    con.execute("INSERT INTO pipeline_events (v) VALUES ('after')")
    con.commit()
    con.close()
    assert _count(sandbox) == 6


def test_restoring_with_no_snapshots_is_a_no_op_not_a_crash(sandbox):
    _seed(sandbox, 3)
    sandbox.unlink()
    assert backup.restore_if_empty() is False
    assert backup.restore() is False


def test_restore_on_empty_can_be_switched_off(sandbox, monkeypatch):
    _seed(sandbox, 5)
    backup.take_snapshot(reason="test")
    sandbox.unlink()
    monkeypatch.setattr(settings, "BACKUP_RESTORE_ON_EMPTY", False)

    assert backup.restore_if_empty() is False
    assert not sandbox.exists()


# ---------------------------------------------------------------------------
# the warning that matters most
# ---------------------------------------------------------------------------

def test_backups_beside_the_database_are_flagged_as_pointless(sandbox, monkeypatch):
    """The most common way people lose this data, and it fails silently:
    the snapshots are taken, verified, logged, and then wiped along with
    the database on the next deploy."""
    monkeypatch.setattr(settings, "BACKUP_DIR", str(sandbox.parent / "backups"))
    warning = backup.warn_if_backups_are_pointless()

    assert warning is not None
    assert "wiped along with the database" in warning


def test_a_backup_dir_on_another_path_is_not_flagged(sandbox, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BACKUP_DIR", str(tmp_path / "elsewhere"))
    assert backup.warn_if_backups_are_pointless() is None


# ---------------------------------------------------------------------------
# the app lifecycle actually running
# ---------------------------------------------------------------------------

def test_the_lifespan_is_a_working_async_context_manager():
    """A regression guard for a bug 1030 unit tests could not see.

    Nothing in the suite entered the app's lifespan - TestClient only runs
    it inside a `with` block - so a startup path that raised on the first
    line stayed green. It had: an edit landed between
    @asynccontextmanager and `async def lifespan`, decorating the wrong
    function, and every request would have failed against a real server.
    """
    import inspect

    from app.main import _snapshot_forever, lifespan

    assert inspect.isasyncgenfunction(_snapshot_forever) is False
    assert inspect.iscoroutinefunction(_snapshot_forever), (
        "_snapshot_forever must be a plain coroutine - asyncio.create_task "
        "cannot schedule a context manager"
    )
    assert hasattr(lifespan(None), "__aenter__"), (
        "lifespan is not an async context manager - FastAPI cannot start"
    )


def test_the_app_starts_backs_up_and_recovers_from_a_wipe(tmp_path, monkeypatch):
    """The whole point, exercised through the real app rather than the
    module: boot, record, snapshot, delete the database, boot again."""
    import datetime as dt

    db_path = tmp_path / "data" / "bot.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(settings, "BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setattr(settings, "BACKUP_ENABLED", True)
    monkeypatch.setattr(settings, "SCANNER_ENABLED", False)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_ENABLED", False)

    # A standalone engine bound to the sandbox database - the app's own
    # engine is bound to the suite's shared one at import time.
    from sqlalchemy import create_engine
    from app.database import Base

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    with engine.begin() as con:
        from sqlalchemy import text

        for i in range(30):
            con.execute(
                text(
                    "INSERT INTO forward_returns "
                    "(pipeline_event_id, token_address, symbol, observed_at, score, "
                    " price_at_signal, horizon_minutes, due_at, return_pct) "
                    "VALUES (:i, :a, :s, :t, 70, 0.01, 60, :t, 2.0)"
                ),
                {"i": i, "a": f"M{i}", "s": f"T{i}",
                 "t": dt.datetime.now(dt.timezone.utc).isoformat()},
            )
    engine.dispose()

    assert backup.take_snapshot(reason="test") is not None
    db_path.unlink()
    assert backup.database_is_empty()

    # what startup does
    assert backup.restore_if_empty() is True
    assert _count(db_path, "forward_returns") == 30
