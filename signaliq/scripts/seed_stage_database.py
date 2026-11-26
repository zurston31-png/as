"""Start a new research database from an archived one, without modifying or copying its SEC cache.

Every table of the parent is copied row for row (timestamps included) into a new database,
except the SEC answers in ``external_responses``: those stay in the parent, which later runs
read through the read-only parent cache (``app.universe.sec_cache``,
SIGNALIQ_SIP_RESEARCH_PARENT_DB). The parent is opened read-only at the SQLite level, so it
cannot be modified. After copying, every table's fingerprint is compared with the parent's
(``external_responses`` as merged with the parent), and any difference is an error.

    python scripts/seed_stage_database.py --parent data/universe_stage.db --out NEW.db
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from table_fingerprints import fingerprints

from app.universe.database import open_research_database


def seed(parent: Path, out: Path) -> list[str]:
    parent, out = parent.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(f"{out} exists: a seed starts a new database only")
    if parent == out:
        raise ValueError("the new database must be another file")
    # The schema comes from the migrations, as for any new research database.
    open_research_database(f"sqlite:///{out.as_posix()}", read_only=False).dispose()
    connection = sqlite3.connect(f"file:{out.as_posix()}?mode=rw", uri=True)
    connection.execute(f"ATTACH DATABASE 'file:{parent.as_posix()}?mode=ro' AS parent")
    tables = [
        r[0]
        for r in connection.execute(
            "select name from parent.sqlite_master where type='table' order by 1"
        )
    ]
    copied = []
    with connection:
        for table in tables:
            mine = [r[1] for r in connection.execute(f'pragma main.table_info("{table}")')]
            theirs = [r[1] for r in connection.execute(f'pragma parent.table_info("{table}")')]
            if mine != theirs:
                raise RuntimeError(f"{table}: columns differ ({mine} against {theirs})")
            columns = ",".join(f'"{c}"' for c in mine)
            connection.execute(f'delete from main."{table}"')
            where = " where source <> 'sec'" if table == "external_responses" else ""
            connection.execute(
                f'insert into main."{table}" ({columns}) '
                f'select {columns} from parent."{table}"{where}'
            )
            copied.append(table)
    connection.execute("DETACH DATABASE parent")
    connection.close()
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    copied = seed(args.parent, args.out)
    print(f"copied {len(copied)} tables from {args.parent} (read-only) into {args.out}")
    print("verifying every table against the parent ...", flush=True)
    theirs = {t: (n, d) for t, n, d, _ in fingerprints(str(args.parent.resolve()))}
    mine = {t: (n, d) for t, n, d, _ in fingerprints(str(args.out.resolve()), str(args.parent))}
    bad = 0
    for table in sorted(theirs):
        same = mine.get(table) == theirs[table]
        bad += not same
        print(f"  {'ok  ' if same else 'DIFF'} {table}: {mine.get(table)} / parent {theirs[table]}")
    print("RESULT:", "every table equal to the parent" if not bad else f"{bad} tables differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
