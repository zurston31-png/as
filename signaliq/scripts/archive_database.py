"""Archive a research database to compressed files that can be stored anywhere, and get it back.

    python scripts/archive_database.py export  DB  --out DIR
    python scripts/archive_database.py verify  DIR [--against DB]
    python scripts/archive_database.py restore DIR --out NEW.db [--compress-sec]

**export** reads the database read-only (SQLite ``mode=ro``) and writes:

* ``NNNN-<table>.jsonl.gz``: every row of every table, one JSON array per line, all
  columns (timestamps included), in the order of all non-timestamp columns; a large table
  is split into several files (``--chunk-mb`` of uncompressed rows each), and the export
  fails if any compressed file exceeds ``--max-file-mb`` (default 95 MB, under GitHub's
  100 MB limit per file);
* ``manifest.json``: the exact schema (SQLite's own CREATE statements), the columns, each
  file's sha256, size and row count, and per table: the row count, the fingerprint of
  ``scripts/table_fingerprints.py`` (all columns but timestamps) and a second fingerprint
  over all columns including timestamps.

**verify** checks every file's sha256, then decompresses every row and recomputes both
fingerprints of every table, which must equal the manifest's; with ``--against`` it also
compares them with the live database. Nothing is assumed: every row is read back.

**restore** recreates the schema exactly and inserts every row into a new database (it
refuses to overwrite a file). With ``--compress-sec``, SEC answers longer than 64 KB are
stored zlib-compressed (the format ``app.universe.sec_cache`` reads), so a 20 GB cache fits
in about 2 GB; ``table_fingerprints.py --logical`` then decodes them, and the restored
database's fingerprints must equal the original's.
"""

import argparse
import base64
import gzip
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

FORMAT = "signaliq-research-archive/1"


def _connect_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _encode(value):
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    return value


def _decode(value):
    if isinstance(value, dict) and set(value) == {"$bytes"}:
        return base64.b64decode(value["$bytes"])
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _Fingerprints:
    """Both fingerprints of one table, fed rows in all-non-timestamp-column order."""

    def __init__(self, columns: list[str]):
        self.keep = [i for i, c in enumerate(columns) if not c.endswith("_at")]
        self.core, self.full, self.rows = hashlib.sha256(), hashlib.sha256(), 0

    def add(self, row: tuple) -> None:
        self.core.update(repr(tuple(row[i] for i in self.keep)).encode())
        self.full.update(repr(tuple(row)).encode())
        self.rows += 1

    def result(self) -> dict:
        return {
            "rows": self.rows,
            "fingerprint": self.core.hexdigest()[:16],
            "fingerprint_all_columns": self.full.hexdigest()[:16],
        }


def _tables(connection) -> list[str]:
    return [
        r[0]
        for r in connection.execute(
            "select name from sqlite_master where type='table' and name not like 'sqlite_%' "
            "order by 1"
        )
    ]


def _columns(connection, table) -> list[str]:
    return [r[1] for r in connection.execute(f'pragma table_info("{table}")')]


def _ordered(connection, table, columns):
    order = ",".join(f'"{c}"' for c in columns if not c.endswith("_at"))
    selected = ",".join(f'"{c}"' for c in columns)
    return connection.execute(f'select {selected} from "{table}" order by {order}')


class _ChunkWriter:
    """Gzip files of at most ``limit`` uncompressed bytes per table, recorded in ``files``."""

    def __init__(self, out: Path, limit: int, files: list):
        self.out, self.limit, self.files = out, limit, files
        self.number, self.handle = 0, None

    def start(self, table: str) -> None:
        self.table, self.wrote_any = table, False

    def _open(self) -> None:
        self.number += 1
        self.name, self.size, self.rows = f"{self.number:04d}-{self.table}.jsonl.gz", 0, 0
        self.handle = gzip.open(  # noqa: SIM115 - closed in _close
            self.out / self.name, "wt", encoding="utf-8", compresslevel=6
        )
        self.wrote_any = True

    def _close(self) -> None:
        self.handle.close()
        self.handle = None
        path = self.out / self.name
        self.files.append(
            {
                "file": self.name,
                "table": self.table,
                "rows": self.rows,
                "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            }
        )

    def write(self, line: str) -> None:
        if self.handle is None or self.size >= self.limit:
            if self.handle is not None:
                self._close()
            self._open()
        self.handle.write(line)
        self.size += len(line)
        self.rows += 1

    def finish(self) -> None:
        if not self.wrote_any:  # an empty table still gets a file, so restore sees it
            self._open()
        if self.handle is not None:
            self._close()


def export(db: Path, out: Path, chunk_mb: int) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    connection = _connect_ro(db)
    manifest = {
        "format": FORMAT,
        "source": db.name,
        "schema": [
            r[0]
            for r in connection.execute(
                "select sql from sqlite_master where sql is not null and name not like 'sqlite_%' "
                "order by case type when 'table' then 0 else 1 end, name"
            )
        ],
        "tables": {},
        "files": [],
    }
    writer = _ChunkWriter(out, chunk_mb * 1024 * 1024, manifest["files"])
    for table in _tables(connection):
        columns = _columns(connection, table)
        prints = _Fingerprints(columns)
        writer.start(table)
        for row in _ordered(connection, table, columns):
            writer.write(json.dumps([_encode(v) for v in row], ensure_ascii=False) + "\n")
            prints.add(row)
        writer.finish()
        manifest["tables"][table] = {"columns": columns, **prints.result()}
        print(f"  {table}: {prints.rows} rows", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _rows(directory: Path, entry: dict):
    with gzip.open(directory / entry["file"], "rt", encoding="utf-8") as f:
        for line in f:
            yield tuple(_decode(v) for v in json.loads(line))


def verify(directory: Path, against: Path | None) -> int:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        print(f"FAIL unknown archive format {manifest.get('format')}")
        return 1
    problems = 0
    for entry in manifest["files"]:
        if _sha_file(directory / entry["file"]) != entry["sha256"]:
            print(f"FAIL sha256 of {entry['file']}")
            problems += 1
    print(f"files: {len(manifest['files'])}, sha256 checked; problems: {problems}", flush=True)
    live = {}
    if against is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from table_fingerprints import fingerprints

        live = {t: (n, d) for t, n, d, _ in fingerprints(str(against))}
    for table, info in manifest["tables"].items():
        prints = _Fingerprints(info["columns"])
        rows_in_files = 0
        for entry in (e for e in manifest["files"] if e["table"] == table):
            count = 0
            for row in _rows(directory, entry):
                prints.add(row)
                count += 1
            rows_in_files += count
            if count != entry["rows"]:
                print(f"FAIL {entry['file']}: {count} rows, manifest {entry['rows']}")
                problems += 1
        got = prints.result()
        same = all(got[k] == info[k] for k in ("rows", "fingerprint", "fingerprint_all_columns"))
        live_same = table not in live or live[table] == (got["rows"], got["fingerprint"])
        problems += (not same) + (not live_same)
        print(
            f"  {'ok  ' if same and live_same else 'FAIL'} {table}: {got['rows']} rows, "
            f"{got['fingerprint']} / {got['fingerprint_all_columns']}"
            + (f"; live database {live[table]}" if table in live else "")
        )
    print("RESULT:", "PASS" if not problems else f"FAIL ({problems})")
    return 0 if not problems else 1


def restore(directory: Path, out: Path, compress_sec: bool) -> int:
    if out.exists():
        print(f"{out} exists: restore writes a new database only")
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.universe import sec_cache

    manifest = json.loads((directory / "manifest.json").read_text())
    connection = sqlite3.connect(out)
    connection.execute("PRAGMA journal_mode=WAL")
    tables = [s for s in manifest["schema"] if s.upper().startswith("CREATE TABLE")]
    others = [s for s in manifest["schema"] if not s.upper().startswith("CREATE TABLE")]
    for statement in tables:
        connection.execute(statement)
    for table, info in manifest["tables"].items():
        columns = info["columns"]
        response = columns.index("response") if table == "external_responses" else None
        placeholders = ",".join("?" for _ in columns)
        names = ",".join(f'"{c}"' for c in columns)
        for entry in (e for e in manifest["files"] if e["table"] == table):
            batch = []
            for row in _rows(directory, entry):
                if compress_sec and response is not None and row[0] == "sec":
                    stored = sec_cache.encode(json.loads(row[response]))
                    if isinstance(stored, dict):
                        row = (*row[:response], json.dumps(stored), *row[response + 1 :])
                batch.append(row)
                if len(batch) >= 2000:
                    connection.executemany(
                        f'insert into "{table}" ({names}) values ({placeholders})', batch
                    )
                    batch = []
            if batch:
                connection.executemany(
                    f'insert into "{table}" ({names}) values ({placeholders})', batch
                )
            connection.commit()
        print(f"  {table}: restored", flush=True)
    for statement in others:
        connection.execute(statement)
    connection.commit()
    connection.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("db", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--chunk-mb", type=int, default=256, help="uncompressed MB per file")
    e.add_argument(
        "--max-file-mb",
        type=int,
        default=95,
        help="largest compressed file allowed (GitHub refuses files over 100 MB)",
    )
    v = sub.add_parser("verify")
    v.add_argument("directory", type=Path)
    v.add_argument("--against", type=Path)
    r = sub.add_parser("restore")
    r.add_argument("directory", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--compress-sec", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        manifest = export(args.db, args.out, args.chunk_mb)
        too_big = [f for f in manifest["files"] if f["bytes"] > args.max_file_mb * 1024 * 1024]
        total = sum(f["bytes"] for f in manifest["files"])
        print(f"files: {len(manifest['files'])}, {total / 1024**2:.0f} MB compressed")
        for f in too_big:
            print(f"FAIL {f['file']} is {f['bytes'] / 1024**2:.0f} MB (limit {args.max_file_mb})")
        return 1 if too_big else 0
    if args.command == "verify":
        return verify(args.directory, args.against)
    return restore(args.directory, args.out, args.compress_sec)


if __name__ == "__main__":
    sys.exit(main())
