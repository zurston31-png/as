"""Per-table fingerprints of a research database, read-only.

For every table: sha256 over ``repr(row)`` of every row, all columns except timestamps
(``*_at``), in the order of all those columns; the first 16 hex characters are printed.

With ``--parent``, ``external_responses`` is fingerprinted as the cache the database reads:
its own rows plus the parent's rows it does not override (``app.universe.sec_cache``). The
two are merged in key order, which for this table (primary key source, request_key) is the
same order as sorting by all columns, so a database whose own cache is empty and whose
parent is X gives X's fingerprint exactly.

With ``--logical``, SEC answers stored compressed (``app.universe.sec_cache``, as a
restored archive stores them) are hashed as the JSON they encode, so a restored database
gives the original's fingerprints.

    python scripts/table_fingerprints.py DB [--parent PARENT_DB] [--logical]
"""

import argparse
import base64
import hashlib
import json
import sqlite3
import sys
import zlib


def _connect(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _columns(connection, table):
    names = [r[1] for r in connection.execute(f'pragma table_info("{table}")')]
    return [n for n in names if not n.endswith("_at")], [n for n in names if n.endswith("_at")]


def _rows(connection, table, columns):
    selected = ",".join(f'"{c}"' for c in columns)
    return connection.execute(f'select {selected} from "{table}" order by {selected}')


def _merged(own, parent, key_width):
    """Rows of two key-ordered streams; on equal keys the first stream's row wins."""
    a, b = next(own, None), next(parent, None)
    while a is not None or b is not None:
        if b is None or (a is not None and a[:key_width] < b[:key_width]):
            yield a
            a = next(own, None)
        elif a is None or b[:key_width] < a[:key_width]:
            yield b
            b = next(parent, None)
        else:
            yield a
            a, b = next(own, None), next(parent, None)


def _logical(rows, response_index):
    """Rows with compressed SEC answers (``sec_cache``) replaced by the JSON they encode."""
    for row in rows:
        value = row[response_index]
        if isinstance(value, str) and value.startswith('{"zlib_b64"'):
            decoded = json.loads(value)
            if set(decoded) == {"zlib_b64"}:
                text = zlib.decompress(base64.b64decode(decoded["zlib_b64"])).decode("utf-8")
                row = (*row[:response_index], json.dumps(text), *row[response_index + 1 :])
        yield row


def fingerprints(
    path: str, parent: str | None = None, logical: bool = False
) -> list[tuple[str, int, str, list[str]]]:
    connection = _connect(path)
    parent_connection = _connect(parent) if parent else None
    result = []
    tables = [
        r[0]
        for r in connection.execute(
            "select name from sqlite_master where type='table' order by 1"
        ).fetchall()
    ]
    for table in tables:
        columns, left_out = _columns(connection, table)
        rows = _rows(connection, table, columns)
        if table == "external_responses" and parent_connection is not None:
            if columns[:2] != ["source", "request_key"]:
                raise RuntimeError(f"unexpected external_responses columns: {columns}")
            rows = _merged(iter(rows), iter(_rows(parent_connection, table, columns)), 2)
        if table == "external_responses" and logical:
            rows = _logical(rows, columns.index("response"))
        digest, count = hashlib.sha256(), 0
        for row in rows:
            digest.update(repr(tuple(row)).encode())
            count += 1
        result.append((table, count, digest.hexdigest()[:16], left_out))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    parser.add_argument("--parent")
    parser.add_argument(
        "--logical", action="store_true", help="decode compressed SEC answers before hashing"
    )
    args = parser.parse_args()
    for table, count, digest, left_out in fingerprints(args.database, args.parent, args.logical):
        print(f"{table}: {count} rows, sha256 {digest} (columns left out: {left_out or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
