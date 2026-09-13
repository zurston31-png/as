#!/usr/bin/env python3
"""Database snapshots from the terminal.

    python scripts/backup.py status              what exists, and will it survive
    python scripts/backup.py now                 take one immediately
    python scripts/backup.py restore             restore the newest snapshot
    python scripts/backup.py restore --file X    restore a specific one
    python scripts/backup.py restore --force     overwrite a populated database

Restore refuses to overwrite a database that already holds research rows
unless --force is given. That guard is the point: rolling the dataset back
to last night by accident destroys exactly what the snapshots exist to
protect, and it is not recoverable by taking another snapshot.

The bot does not need to be stopped to take a snapshot - it uses SQLite's
online backup API. It DOES need to be stopped to restore one.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backup  # noqa: E402
from app.config import settings  # noqa: E402

RULE = "=" * 78


def cmd_status(args) -> int:
    print(RULE)
    print(" DATABASE SNAPSHOTS")
    print(RULE)

    source = backup.database_path()
    if source is None:
        print(f" DATABASE_URL is {settings.DATABASE_URL!r}, which is not SQLite.")
        print(" This tool only handles SQLite. Postgres and MySQL have their own")
        print(" backup tooling, which is better than anything reimplemented here.")
        return 0

    print(f" database : {source}  ({'exists' if source.exists() else 'MISSING'})")
    print(f" snapshots: {backup.backup_dir()}")
    print(f" schedule : every {settings.BACKUP_INTERVAL_MINUTES}m, keeping {settings.BACKUP_KEEP}"
          f"{'' if settings.BACKUP_ENABLED else '  (DISABLED)'}")
    print(f" restore on empty: {settings.BACKUP_RESTORE_ON_EMPTY}")

    warning = backup.warn_if_backups_are_pointless()
    if warning:
        print(f"\n WARNING: {warning}")

    snapshots = backup.list_snapshots()
    if not snapshots:
        print("\n No snapshots yet.")
        return 0

    print(f"\n {len(snapshots)} snapshot(s), newest first:")
    print(f"   {'file':<34}{'age':>10}{'size':>12}")
    for snapshot in snapshots:
        age = (
            f"{snapshot.age_hours:.1f}h" if snapshot.age_hours < 48
            else f"{snapshot.age_hours / 24:.1f}d"
        )
        print(f"   {snapshot.path.name:<34}{age:>10}{snapshot.size_bytes / 1024:>10.1f} KB")
    return 0


def cmd_now(args) -> int:
    snapshot = backup.take_snapshot(reason="manual")
    if snapshot is None:
        print("Snapshot failed. Nothing was rotated - see the log output above.")
        return 1
    print(f"Wrote {snapshot.path} ({snapshot.size_bytes / 1024:.1f} KB), verified.")
    return 0


def cmd_restore(args) -> int:
    snapshot = None
    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = backup.backup_dir() / path
        if not path.exists():
            print(f"No such snapshot: {path}")
            return 1
        import datetime as dt

        snapshot = backup.Snapshot(
            path=path,
            taken_at=dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc),
            size_bytes=path.stat().st_size,
        )

    if backup.restore(snapshot, force=args.force):
        print("Restored. Anything recorded after that snapshot is not in this database.")
        return 0
    print("Restore did not happen - see the message above.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="what exists, and will it survive a reset")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("now", help="take a verified snapshot immediately")
    p.set_defaults(func=cmd_now)

    p = sub.add_parser("restore", help="restore a snapshot over the database")
    p.add_argument("--file", help="snapshot filename (default: the newest)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a database that already holds rows")
    p.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
