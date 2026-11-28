#!/usr/bin/env python3
# ruff: noqa: E501
"""Membership differences between two datasets, decision by decision (read-only).

python research/results/phase1f/scripts/compare_membership.py A_DB B_DB
"""

import sqlite3
import sys

COLUMNS = [
    "rule_fingerprint",
    "eligible",
    "reason",
    "raw_close",
    "median_dollar_volume",
    "rank",
    "member",
]


def load(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = {}
    for d, sid, *rest in db.execute(
        f"select decision_session, security_id, {', '.join(COLUMNS)} from membership"
    ):
        rows.setdefault(str(d)[:10], {})[sid] = rest
    symbol = dict(
        db.execute("select security_id, max(symbol) from security_bars group by security_id")
    )
    return rows, symbol


def main() -> int:
    (a, symbol), (b, _) = load(sys.argv[1]), load(sys.argv[2])
    print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}")
    print(f"decisions: {len(a)} vs {len(b)}")
    changed = 0
    for d in sorted(a.keys() | b.keys()):
        ra, rb = a.get(d, {}), b.get(d, {})
        diff = {sid for sid in ra.keys() | rb.keys() if ra.get(sid) != rb.get(sid)}
        if not diff:
            continue
        changed += 1
        cols = sorted({c for sid in diff if sid in ra and sid in rb
                       for c, x, y in zip(COLUMNS, ra[sid], rb[sid], strict=True) if x != y})  # fmt: skip
        ma = {sid for sid, r in ra.items() if r[-1]}
        mb = {sid for sid, r in rb.items() if r[-1]}
        print(f"{d}: {len(diff)} rows differ (columns {cols}; rows only in one: "
              f"{len(ra.keys() ^ rb.keys())}); members {len(ma)} vs {len(mb)}, "
              f"only in A {sorted(symbol.get(s, s) for s in ma - mb)}, only in B {sorted(symbol.get(s, s) for s in mb - ma)}")  # fmt: skip
    print(f"decisions with any difference: {changed} of {len(a.keys() | b.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
