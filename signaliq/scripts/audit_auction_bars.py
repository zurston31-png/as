#!/usr/bin/env python3
"""Audit: auction-only daily bars of normally liquid securities (proposed threshold G1).

A bar is flagged when its security had a median of at least 1,000 trades over the 20 market
sessions just before it (sessions without a bar count as 0, so a long suspension never looks
liquid) and the bar itself has at most 5 trades. The baseline is per security, never per
ticker, so a reused ticker does not inherit another company's history.

Every flagged bar must appear in the reviewed list (``--explained``, a TSV with the columns
session, security_id, explanation, treatment). The audit passes only when no flagged bar is
missing from that list. It reads the database read-only and changes nothing.

    python scripts/audit_auction_bars.py --db data/universe_stage.db [--explained FILE]
        [--out flagged.tsv]
"""

import argparse
import csv
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASELINE_SESSIONS = 20
LIQUID_MEDIAN_TRADES = 1000
MAX_SUSPECT_TRADES = 5


def flagged_bars(db: sqlite3.Connection) -> list[dict]:
    calendar = [r[0] for r in db.execute("select distinct session from security_bars order by 1")]
    index = {d: i for i, d in enumerate(calendar)}
    bars: dict[int, dict[str, tuple]] = defaultdict(dict)
    for sid, session, symbol, o, h, low, c, v, n in db.execute(
        "select security_id, session, symbol, open, high, low, close, volume, trade_count "
        "from security_bars"
    ):
        bars[sid][session] = (symbol, o, h, low, c, v, n)
    found = []
    for sid, by_session in bars.items():
        for session, (symbol, o, h, low, c, _v, n) in by_session.items():
            i = index[session]
            if n is None or n > MAX_SUSPECT_TRADES or i < BASELINE_SESSIONS:
                continue
            before = [calendar[j] for j in range(i - BASELINE_SESSIONS, i)]
            normal = statistics.median((by_session.get(d, (None,) * 7)[6] or 0) for d in before)
            if normal >= LIQUID_MEDIAN_TRADES:
                found.append(
                    {
                        "session": session,
                        "security_id": sid,
                        "symbol": symbol,
                        "normal_trades": normal,
                        "trades": n,
                        "auction_only_shape": h == max(o, c) and low == min(o, c),
                    }
                )
    return sorted(found, key=lambda r: (r["session"], r["symbol"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--explained", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    found = flagged_bars(db)
    explained = set()
    if args.explained:
        with args.explained.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["explanation"].strip() and row["treatment"].strip():
                    explained.add((row["session"], int(row["security_id"])))
    missing = [r for r in found if (r["session"], r["security_id"]) not in explained]
    print(f"Auction-only bar audit of {args.db.name} (read-only)")
    print(
        f"rule: median >= {LIQUID_MEDIAN_TRADES} trades over the {BASELINE_SESSIONS} market "
        f"sessions before (missing = 0), and <= {MAX_SUSPECT_TRADES} trades"
    )
    print(f"flagged bars: {len(found)}")
    for session, count in sorted(Counter(r["session"] for r in found).items()):
        print(f"  {session}: {count}")
    print(f"flagged bars without a reviewed explanation and treatment: {len(missing)}")
    if args.out:
        with args.out.open("w") as f:
            f.write("session\tsecurity_id\tsymbol\tnormal_trades\ttrades\tauction_only_shape\n")
            for r in found:
                f.write(
                    f"{r['session']}\t{r['security_id']}\t{r['symbol']}\t{r['normal_trades']:g}"
                    f"\t{r['trades']}\t{r['auction_only_shape']}\n"
                )
    print(f"RESULT: {'PASS' if not missing else 'FAIL'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
