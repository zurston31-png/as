"""Snapshots of the database, so a reset does not erase the dataset.

WHY A PLAIN FILE COPY IS NOT ENOUGH

The bot writes continuously - the position monitor, the scanner, the
watchlist loop and the forward-return worker all commit on their own
schedules. Copying the .db file with shutil while a write is in flight
produces a file that opens fine and is silently corrupt in the middle,
which is the worst possible failure for a backup: it looks like it worked
and only turns out to be worthless on the day it is needed. SQLite's
online backup API copies page by page under the same locking the database
itself uses, and is the only supported way to snapshot a live one.

Every snapshot is verified with PRAGMA integrity_check before it is
allowed to count as a backup. An unverified backup is a guess.

WHAT THIS DOES AND DOES NOT PROTECT AGAINST

    process restart, crash, redeploy    yes, if BACKUP_DIR survives
    accidental deletion of the db       yes
    a host with an EPHEMERAL filesystem yes ONLY IF BACKUP_DIR points at a
                                        mounted volume or synced directory

That last one is the case people actually hit. On Railway, Render, Fly or
a plain Heroku dyno the whole filesystem is replaced on every deploy, and
a backup written next to the database dies with it. Pointing BACKUP_DIR at
a mounted volume is what makes any of this work; there is a warning at
startup when it appears to be on the same ephemeral disk.

RESTORE IS DELIBERATELY CONSERVATIVE

Startup restores automatically ONLY when the database is missing or has no
tables. It will never overwrite a database that already holds rows: a bot
that silently rolled its own history back to last night's snapshot because
of a transient error would destroy exactly the data this module exists to
protect. Overwriting a populated database is a manual command.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

BACKUP_SUFFIX = ".db"
BACKUP_PREFIX = "snapshot-"


def is_sqlite() -> bool:
    return settings.DATABASE_URL.startswith("sqlite")


def database_path() -> Path | None:
    """Filesystem path of the SQLite database, or None for other engines.

    Postgres and friends have their own backup tooling that is better than
    anything this module would reimplement, so they are left alone rather
    than half-supported.
    """
    if not is_sqlite():
        return None
    raw = settings.DATABASE_URL.split("sqlite:///", 1)[-1]
    return Path(raw).resolve()


def backup_dir() -> Path:
    return Path(settings.BACKUP_DIR).resolve()


@dataclass
class Snapshot:
    path: Path
    taken_at: dt.datetime
    size_bytes: int

    @property
    def age_hours(self) -> float:
        return (dt.datetime.now(dt.timezone.utc) - self.taken_at).total_seconds() / 3600

    def as_dict(self) -> dict:
        return {
            "file": self.path.name,
            "taken_at": self.taken_at.isoformat(),
            "size_bytes": self.size_bytes,
            "age_hours": round(self.age_hours, 2),
        }


def list_snapshots() -> list[Snapshot]:
    """Existing snapshots, newest first."""
    directory = backup_dir()
    if not directory.is_dir():
        return []
    snapshots = []
    for path in directory.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"):
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshots.append(
            Snapshot(
                path=path,
                taken_at=dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc),
                size_bytes=stat.st_size,
            )
        )
    return sorted(snapshots, key=lambda s: s.taken_at, reverse=True)


def latest_snapshot() -> Snapshot | None:
    snapshots = list_snapshots()
    return snapshots[0] if snapshots else None


def _verify(path: Path) -> bool:
    """Is this file a readable, internally consistent SQLite database?

    A snapshot that has not been checked is not a backup, it is a file.
    """
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            result = con.execute("PRAGMA integrity_check").fetchone()
            tables = con.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
    except sqlite3.Error as exc:
        logger.error("snapshot %s failed to open: %s", path.name, exc)
        return False

    if not result or result[0] != "ok":
        logger.error("snapshot %s failed integrity_check: %s", path.name, result)
        return False
    if not tables:
        logger.error("snapshot %s has no tables", path.name)
        return False
    return True


def snapshot_for_download(*, max_age_seconds: float = 60.0) -> Snapshot | None:
    """A current snapshot to hand to a browser, without churning history.

    Taking a fresh one on every request would let someone refreshing the
    download page rotate out every older snapshot within a minute -
    destroying backup history through the very button meant to protect it.
    So a snapshot younger than a minute is reused as-is.
    """
    newest = latest_snapshot()
    if newest is not None and newest.age_hours * 3600 <= max_age_seconds:
        return newest
    return take_snapshot(reason="download") or newest


def take_snapshot(*, reason: str = "scheduled") -> Snapshot | None:
    """Copy the live database, verify it, and rotate old snapshots.

    Returns None rather than raising on any failure. A backup that cannot
    be taken must not be able to bring the bot down - losing a snapshot is
    recoverable, losing the process that would have taken the next one is
    not.
    """
    source = database_path()
    if source is None:
        logger.debug("backups skipped: DATABASE_URL is not sqlite")
        return None
    if not source.exists():
        logger.debug("backups skipped: %s does not exist yet", source)
        return None

    directory = backup_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("cannot create backup directory %s: %s", directory, exc)
        return None

    # Millisecond resolution. Two snapshots in the same second - a manual one
    # landing on a scheduled one - would otherwise share a filename and the
    # second would silently replace the first.
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")[:-4] + "Z"
    # Written to a .partial name first. A crash mid-copy then leaves an
    # obviously-incomplete file rather than something that looks like a
    # usable snapshot and is not.
    final = directory / f"{BACKUP_PREFIX}{stamp}{BACKUP_SUFFIX}"
    partial = final.with_suffix(".partial")

    try:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src, \
                sqlite3.connect(partial) as dst:
            # The online backup API, not a file copy: it holds the same
            # locks the database uses, so a write in flight cannot tear the
            # snapshot in half.
            src.backup(dst)
    except sqlite3.Error as exc:
        logger.error("snapshot failed: %s", exc)
        partial.unlink(missing_ok=True)
        return None

    if not _verify(partial):
        partial.unlink(missing_ok=True)
        return None

    try:
        partial.replace(final)
    except OSError as exc:
        logger.error("could not finalise snapshot %s: %s", final.name, exc)
        partial.unlink(missing_ok=True)
        return None

    snapshot = Snapshot(
        path=final,
        taken_at=dt.datetime.now(dt.timezone.utc),
        size_bytes=final.stat().st_size,
    )
    logger.info(
        "database snapshot taken (%s): %s, %.1f KB",
        reason, final.name, snapshot.size_bytes / 1024,
    )
    _rotate()
    return snapshot


def _rotate() -> int:
    """Delete the oldest snapshots beyond BACKUP_KEEP.

    Rotation happens AFTER a successful verified snapshot, never before.
    Pruning first would mean a failed backup had already destroyed one of
    the good ones it was meant to replace.
    """
    keep = max(settings.BACKUP_KEEP, 1)
    doomed = list_snapshots()[keep:]
    for snapshot in doomed:
        try:
            snapshot.path.unlink()
            logger.debug("rotated out old snapshot %s", snapshot.path.name)
        except OSError as exc:
            logger.warning("could not remove old snapshot %s: %s", snapshot.path.name, exc)
    return len(doomed)


def database_is_empty() -> bool:
    """True when there is no database, or one with no tables.

    "No tables" rather than "no rows": init_db() creates the schema before
    anything else runs, so a freshly-created empty schema is the normal
    state of a wiped install and must still count as empty.
    """
    source = database_path()
    if source is None or not source.exists() or source.stat().st_size == 0:
        return True
    try:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as con:
            tables = con.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
    except sqlite3.Error:
        return True
    return tables == 0


def _row_count() -> int:
    """Rows across the tables that carry the research dataset.

    Used only to decide whether a database is worth protecting from an
    automatic restore. Counts the append-only history rather than every
    table, because a bot that has merely started up already has bookkeeping
    rows - the strategy-version and upstream-health tables - and would
    otherwise look populated when it holds no research data at all.
    """
    source = database_path()
    if source is None or not source.exists():
        return 0
    total = 0
    try:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as con:
            for table in ("pipeline_events", "forward_returns", "signals",
                          "trades", "positions", "token_observations", "watchlist"):
                try:
                    total += con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                except sqlite3.Error:
                    continue      # table not present in an older schema
    except sqlite3.Error:
        return 0
    return total


def restore(snapshot: Snapshot | None = None, *, force: bool = False) -> bool:
    """Replace the live database with a snapshot.

    Refuses to overwrite a database that already holds research rows
    unless `force` is set. Automatic restore over live data is how a
    transient read error turns into silently rolling the dataset back to
    last night - the exact loss this module exists to prevent.
    """
    snapshot = snapshot or latest_snapshot()
    if snapshot is None:
        logger.info("no snapshot available to restore")
        return False

    target = database_path()
    if target is None:
        logger.error("restore skipped: DATABASE_URL is not sqlite")
        return False

    existing = _row_count()
    if existing and not force:
        logger.error(
            "refusing to restore over a database holding %d rows - "
            "run `python scripts/backup.py restore --force` if that is really intended",
            existing,
        )
        return False

    if not _verify(snapshot.path):
        logger.error("refusing to restore %s: it did not verify", snapshot.path.name)
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy through SQLite again rather than moving the file, so the
        # snapshot survives the restore and can be restored a second time
        # if this one turns out to be the wrong choice.
        with sqlite3.connect(f"file:{snapshot.path}?mode=ro", uri=True) as src, \
                sqlite3.connect(target) as dst:
            src.backup(dst)
    except (sqlite3.Error, OSError) as exc:
        logger.error("restore from %s failed: %s", snapshot.path.name, exc)
        return False

    # Drop every pooled connection. SQLAlchemy's pool may still hold handles
    # to the file that was just replaced - and if the old one was unlinked
    # rather than overwritten, those handles point at a deleted inode that
    # SQLite reports as a READONLY database. Reads keep working, writes fail
    # with an error that looks like a permissions problem and is not.
    # Harmless when the pool is already cold, which it is at startup.
    try:
        from app.database import engine

        engine.dispose()
    except Exception:
        logger.exception("could not dispose the connection pool after restore")

    logger.warning(
        "database restored from %s (taken %.1fh ago) - anything recorded after that "
        "snapshot is not in this database",
        snapshot.path.name, snapshot.age_hours,
    )
    return True


def restore_if_empty() -> bool:
    """Startup hook: bring back the dataset after the disk was wiped."""
    if not settings.BACKUP_ENABLED or not settings.BACKUP_RESTORE_ON_EMPTY:
        return False
    if not database_is_empty():
        return False
    snapshot = latest_snapshot()
    if snapshot is None:
        return False
    logger.warning(
        "database is empty but a snapshot from %s exists - restoring it. "
        "This is what a filesystem reset looks like.",
        snapshot.taken_at.isoformat(),
    )
    return restore(snapshot)


def warn_if_backups_are_pointless() -> str | None:
    """Is BACKUP_DIR on the same disk that gets wiped?

    A backup written next to the database dies with it on any host whose
    filesystem is replaced on deploy - which is the single most common way
    people lose this data, and it fails silently: the snapshots are taken,
    verified, logged, and then thrown away with everything else.
    """
    source = database_path()
    if source is None or not settings.BACKUP_ENABLED:
        return None
    directory = backup_dir()
    try:
        same_disk = os.path.commonpath([directory, source.parent]) == str(source.parent)
    except ValueError:
        same_disk = False
    if same_disk:
        return (
            f"BACKUP_DIR ({directory}) is inside the database's own directory "
            f"({source.parent}). On a host that replaces the filesystem on deploy - "
            "Railway, Render, Fly, a plain Heroku dyno - the snapshots are wiped along "
            "with the database they were protecting. Point BACKUP_DIR at a mounted "
            "volume, or download snapshots from /backup/download."
        )
    return None
