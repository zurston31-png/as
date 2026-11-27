"""Summarise or verify a raw provider response store (read-only).

    python scripts/response_store_tool.py summary data/run/alpaca_responses.db
    python scripts/response_store_tool.py verify  data/run/alpaca_responses.db

``summary`` prints the number of stored responses, their total size and a fingerprint over
(request key, body sha256). ``verify`` also decompresses every body and checks its sha256 and
length; it exits 1 if any body is damaged.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.response_store import ResponseStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("summary", "verify"))
    parser.add_argument("store", type=Path)
    args = parser.parse_args()

    store = ResponseStore(args.store, "replay")  # read-only
    info = store.summary()
    print(
        f"responses: {info['responses']}; body bytes: {info['body_bytes']}; "
        f"fingerprint: {info['fingerprint']}"
    )
    connection = sqlite3.connect(f"file:{args.store}?mode=ro", uri=True)
    for path, count in connection.execute(
        "select path, count(*) from responses group by path order by path"
    ):
        print(f"  {path}: {count}")
    first, last = connection.execute(
        "select min(fetched_at), max(fetched_at) from responses"
    ).fetchone()
    print(f"  recorded between {first} and {last}")
    try:
        total, failed, errors = connection.execute(
            "select count(*), sum(status is not null and status <> 200), sum(status is null)"
            " from attempts"
        ).fetchone()
        print(f"  HTTP attempts: {total}; non-200: {failed or 0}; no response: {errors or 0}")
    except sqlite3.OperationalError:
        print("  HTTP attempts: not logged (store made before attempt logging)")
    connection.close()
    if args.command == "verify":
        problems = store.verify()
        for problem in problems:
            print(f"DAMAGED {problem}")
        print(f"bodies checked: {info['responses']}; damaged: {len(problems)}")
        print("RESULT: PASS" if not problems else "RESULT: FAIL")
        store.close()
        return 1 if problems else 0
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
