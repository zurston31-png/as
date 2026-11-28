#!/usr/bin/env python3
# ruff: noqa: E501
"""Fix the stale free-text notes in the hybrid dataset (version 3), as approved after Phase 1f.

``batch_symbols.reason`` and ``candidates.reason`` quote the 100th member's median of the proof
round that added the symbol. The hybrid inherited Stage 1.1's text, which quotes the value
computed from the defective 2021-04-19 bars. The correct text is the one version 2 wrote
(same bars, same membership). Only rows whose reason differs from version 2 are changed, and
only their ``reason`` column; every other column must already be equal. Runs once: it refuses
unless the hybrid still has the sha256 reviewed in Phase 1f. Stage 1.1 is not opened.

    python research/results/phase1g/scripts/fix_reason_notes.py
"""

import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HYBRID = ROOT / "data" / "stage_1_1_hybrid" / "universe_stage.db"
V2 = ROOT / "data" / "phase1f_v2_rebuild" / "universe_stage.db"
REVIEWED = "d08e272e66c5aa5b7e4f7473fe7b41b93bb671af56bda345fd81dd2146158528"
LOG = ROOT / "research" / "results" / "phase1g" / "reason_notes_fixed.json"
NTH = re.compile(r"on (\d{4}-\d{2}-\d{2}):.*N-th member ([\d,]+)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    before = sha256(HYBRID)
    if before != REVIEWED:
        print(f"refusing: hybrid sha256 {before} is not the reviewed {REVIEWED}")
        return 2
    db = sqlite3.connect(HYBRID)
    db.execute(f"attach database 'file:{V2.as_posix()}?mode=ro' as v2")
    log = []
    for table in ("batch_symbols", "candidates"):
        cols = [r[1] for r in db.execute(f"pragma main.table_info({table})")]
        rest = [c for c in cols if c != "reason"]
        diff = db.execute(
            f"select {', '.join('h.' + c for c in cols)}, v.reason from main.{table} h"
            f" join v2.{table} v using (symbol) where h.reason is not v.reason"
        ).fetchall()
        n_h, n_v = (
            db.execute(f"select count(*) from {s}.{table}").fetchone()[0] for s in ("main", "v2")
        )
        other = db.execute(
            f"select count(*) from (select {', '.join(rest)} from main.{table}"
            f" except select {', '.join(rest)} from v2.{table})"
        ).fetchone()[0]
        if n_h != n_v or other:
            print(f"STOP: {table}: {n_h} vs {n_v} rows, {other} rows differ outside reason")
            return 1
        for *row, new in diff:
            old = dict(zip(cols, row, strict=True))
            day, nth = NTH.search(new).groups()
            m100 = db.execute(
                "select min(median_dollar_volume) from membership where member and decision_session like ?",
                (day + "%",),
            ).fetchone()[0]
            assert f"{m100:,.0f}" == nth, (
                table,
                old["symbol"],
                m100,
                nth,
            )  # text matches this dataset
            db.execute(f"update main.{table} set reason = ? where symbol = ?", (new, old["symbol"]))
            log.append({"table": table, "symbol": old["symbol"], "old": old["reason"], "new": new,
                        "decision": day, "hybrid_100th_member_median": m100})  # fmt: skip
    db.commit()
    db.execute("detach database v2")
    db.close()
    after = sha256(HYBRID)
    LOG.write_text(
        json.dumps({"sha256_before": before, "sha256_after": after, "rows": log}, indent=1)
    )
    print(f"fixed {len(log)} notes: " + ", ".join(f"{r['table']}.{r['symbol']}" for r in log))
    print(f"hybrid sha256 before {before}\nhybrid sha256 after  {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
