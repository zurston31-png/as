#!/usr/bin/env python3
# ruff: noqa: E501
"""Build dataset version 3 ("hybrid"): Stage 1.1 with Alpaca's repaired 2021-04-19 bars.

- Seeds a new database from Stage 1.1 (read-only parent; Stage 1.1 is never written).
- Control: membership recomputed on the untouched seed must equal Stage 1.1's stored
  membership, row for row, before anything is changed.
- Replaces every 2021-04-19 bar (security_bars, ticker_bars) that differs in the version 2
  rebuild (the Phase 1b recording, rebuilt by offline replay). Every replacement is logged with
  old and new values and its source (store, response key, body sha256, fetch time).
- 2021-10-25 and 2022-03-08 are left as they are (flagged and bounded separately).
- Recomputes all 59 membership decisions with the project's rule and runs the look-ahead checks
  B2 (data cut off at each decision) and B3 (later prices x 10).

    PYTHONPATH=. SIGNALIQ_SEC_OFFLINE=1 python research/results/phase1f/scripts/build_hybrid.py
"""

import json
import os
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

DAY = "2021-04-19"
STAGE11 = ROOT / "data" / "universe_stage.db"
V2 = ROOT / "data" / "phase1f_v2_rebuild" / "universe_stage.db"
V2_STORE = ROOT / "data" / "phase1b_record" / "alpaca_responses.db"
OUT = ROOT / "data" / "stage_1_1_hybrid" / "universe_stage.db"
RESULTS = ROOT / "research" / "results" / "phase1f"
KEYS = {"security_bars": ("security_id", "session"), "ticker_bars": ("symbol", "session")}


def main() -> int:
    if OUT.exists():
        print(f"refusing: {OUT} exists")
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seeded = subprocess.run(
        [sys.executable, "scripts/seed_stage_database.py", "--parent", str(STAGE11), "--out", str(OUT)],
        cwd=ROOT, capture_output=True, text=True,
    )  # fmt: skip
    print("seed:", seeded.stdout.strip().splitlines()[-1] if seeded.stdout else seeded.stderr)
    if seeded.returncode:
        return 1
    os.environ["SIGNALIQ_SIP_RESEARCH_DATABASE_URL"] = f"sqlite:///{OUT.as_posix()}"
    os.environ["SIGNALIQ_SIP_RESEARCH_PARENT_DB"] = str(STAGE11)
    os.environ["SIGNALIQ_SEC_OFFLINE"] = "1"
    from app.database import create_session_factory, session_scope
    from app.universe import checks
    from app.universe import pipeline as p
    from app.universe.database import open_research_database
    from app.universe.membership import MembershipRule

    rule = MembershipRule(size=100)
    engine = open_research_database()
    factory = create_session_factory(engine)
    columns = "decision_session, security_id, rule_fingerprint, eligible, reason, raw_close, median_dollar_volume, rank, member"  # fmt: skip

    def stored(db):
        return sorted(db.execute(f"select {columns} from membership").fetchall(), key=str)

    # Control: recompute on the untouched seed (rolled back), compare with Stage 1.1's rows.
    raw = sqlite3.connect(OUT)
    before = stored(raw)
    with session_scope(factory) as s:
        p.decide_membership(s, rule)
        s.flush()
        control = sorted(
            s.connection().exec_driver_sql(f"select {columns} from membership").fetchall(),
            key=str,
        )
        s.rollback()
    ok = [tuple(r) for r in control] == [tuple(r) for r in before]
    print(f"control: recomputed membership on the untouched seed equals Stage 1.1's: {ok} ({len(before)} rows)")  # fmt: skip
    if not ok:
        return 1

    # Replace the 2021-04-19 bars that differ in version 2, logging each one.
    raw.execute(f"attach database 'file:{V2.as_posix()}?mode=ro' as v2")
    store = sqlite3.connect(f"file:{V2_STORE.as_posix()}?mode=ro", uri=True)
    sources: dict[tuple[str, str], tuple] = {}
    for key, params, sha, fetched, body in store.execute(
        "select key, params, body_sha256, fetched_at, body from responses where path='/v2/stocks/bars'"
    ):
        q = json.loads(params)
        if not (q["start"][:10] <= DAY <= q["end"][:10]):
            continue
        for sym, bars in (json.loads(zlib.decompress(body)).get("bars") or {}).items():
            if any(b["t"].startswith(DAY) for b in bars):
                sources.setdefault((sym, q["adjustment"]), (key, sha, fetched))
    log = []
    for table, (k1, k2) in KEYS.items():
        cols = [r[1] for r in raw.execute(f"pragma main.table_info({table})")]
        n_main = raw.execute(
            f"select count(*) from main.{table} where session=?", (DAY,)
        ).fetchone()[0]
        n_v2 = raw.execute(f"select count(*) from v2.{table} where session=?", (DAY,)).fetchone()[0]
        if n_main != n_v2:
            print(f"STOP: {table} has {n_main} rows on {DAY} here and {n_v2} in version 2")
            return 1
        old_rows = {
            (r[cols.index(k1)], r[cols.index(k2)]): r
            for r in raw.execute(f"select * from main.{table} where session=?", (DAY,))
        }
        for new in raw.execute(f"select * from v2.{table} where session=?", (DAY,)).fetchall():
            key = (new[cols.index(k1)], new[cols.index(k2)])
            old = old_rows.get(key)
            if old is None:
                print(f"STOP: {table} {key} missing here")
                return 1
            if tuple(old) == tuple(new):
                continue
            changed = [c for c, a, b in zip(cols, old, new, strict=True) if a != b]
            sym = new[cols.index("symbol")]
            raw.execute(
                f"update main.{table} set " + ", ".join(f"{c}=?" for c in changed)
                + f" where {k1}=? and {k2}=?",
                [new[cols.index(c)] for c in changed] + list(key),
            )  # fmt: skip
            src = sources.get((sym, "raw"), ("?", "?", "?"))
            log.append(
                {
                    "table": table, "key": list(key), "symbol": sym, "changed": changed,
                    "old": dict(zip(cols, old, strict=True)), "new": dict(zip(cols, new, strict=True)),
                    "source_store": "data/phase1b_record/alpaca_responses.db",
                    "source_key": src[0], "source_body_sha256": src[1], "source_fetched_at": src[2],
                    "split_source": sources.get((sym, "split"), ("?",))[0],
                }
            )  # fmt: skip
    raw.commit()
    raw.execute("detach database v2")
    raw.close()
    (OUT.parent / "replacements.json").write_text(json.dumps(log, indent=1, default=str))
    with (RESULTS / "replacements_provenance.tsv").open("w") as f:
        f.write(
            "table\tkey\tsymbol\tcolumns_changed\tsource_body_sha256\tsource_fetched_at\tsource_key\n"
        )
        for r in log:
            f.write(
                f"{r['table']}\t{r['key'][0]}|{r['key'][1]}\t{r['symbol']}\t{','.join(r['changed'])}"
                f"\t{r['source_body_sha256']}\t{r['source_fetched_at']}\t{r['source_key']}\n"
            )  # fmt: skip
    print(f"replaced: {sum(r['table']=='security_bars' for r in log)} security_bars rows, "
          f"{sum(r['table']=='ticker_bars' for r in log)} ticker_bars rows (all on {DAY}); "
          f"without a located raw source: {sum(r['source_key']=='?' for r in log)}")  # fmt: skip

    # Recompute all 59 decisions; look-ahead checks B2 and B3.
    with session_scope(factory) as s:
        decisions = p.decide_membership(s, rule)
    with session_scope(factory) as s:
        differ = checks.recompute_with_truncated_data(s, rule, p.DATA_START)
        cut = decisions[len(decisions) // 2]
        tamper = checks.tamper_future(s, rule, p.DATA_START, cut)
    print(f"decisions recomputed: {len(decisions)}")
    print(f"B2: recomputed from data cut off at each decision: {len(decisions) - len(differ)} of {len(decisions)} identical {differ}")  # fmt: skip
    print(f"B3: every price and volume after {cut} multiplied by 10, earlier decisions unchanged: {tamper}")  # fmt: skip
    engine.dispose()
    return 0 if not differ and tamper else 1


if __name__ == "__main__":
    sys.exit(main())
