#!/usr/bin/env python3
"""Audit: daily bars that lost most of their trades (proposed; stored data only, read-only).

For every bar of a normally liquid security (median of at least 1,000 trades over the 20
market sessions before it, sessions without a bar counting as 0), the trade count is compared
with that 20-session median. A bar below 10% of it is an alert. Each alert is labelled, never
repaired or excluded:

- ``systemic``: on a session where at least 5% of liquid securities, and at least 10, alert at
  once (on ordinary sessions at most one does). A potential provider-side missing-data event;
- ``after_volume_spike``: the 20-session median is more than 3 times the median of the 60
  sessions before those 20, so a normal quiet day looks low against an inflated baseline;
- ``last_bar``: the security's last bar (a partial final day is common before a delisting);
- ``isolated``: none of the above. A potential missing-data bar that needs review.

Half trading days (scheduled 13:00 closes) are tagged; on those days liquid securities trade
about 35-60% of normal, far above the 10% line, so they do not alert by themselves.
Alerts are potential anomalies. A defect is *confirmed* only by a reviewed entry
(``--confirmed`` TSV: session, security_id or *, evidence) backed by recorded evidence.

    python scripts/audit_partial_days.py --db DB [--confirmed FILE] [--out alerts.tsv]
"""

import argparse
import csv
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.validation import market_early_closes

BASELINE = 20
LONG_BASELINE = 60
LIQUID_MEDIAN_TRADES = 1000
ALERT_RATIO = 0.10
SYSTEMIC_SHARE = 0.05
SYSTEMIC_MIN = 10
SPIKE_FACTOR = 3.0


def alerts(db: sqlite3.Connection) -> tuple[list[dict], dict[str, int]]:
    calendar = [r[0] for r in db.execute("select distinct session from security_bars order by 1")]
    index = {d: i for i, d in enumerate(calendar)}
    trades: dict[int, dict[str, int]] = defaultdict(dict)
    symbol: dict[tuple[int, str], str] = {}
    for sid, session, sym, n in db.execute(
        "select security_id, session, symbol, trade_count from security_bars"
    ):
        trades[sid][session] = n or 0
        symbol[(sid, session)] = sym
    liquid: Counter = Counter()
    found = []
    for sid, by_session in trades.items():
        last = max(by_session)
        for session, n in by_session.items():
            i = index[session]
            if i < BASELINE:
                continue
            normal = statistics.median(
                by_session.get(calendar[j], 0) for j in range(i - BASELINE, i)
            )
            if normal < LIQUID_MEDIAN_TRADES:
                continue
            liquid[session] += 1
            if n >= ALERT_RATIO * normal:
                continue
            older = [
                by_session.get(calendar[j], 0)
                for j in range(max(0, i - BASELINE - LONG_BASELINE), i - BASELINE)
            ]
            long_normal = statistics.median(older) if older else None
            found.append(
                {
                    "session": session,
                    "security_id": sid,
                    "symbol": symbol[(sid, session)],
                    "trades": n,
                    "normal_trades": normal,
                    "ratio": n / normal,
                    "spike": bool(long_normal) and normal > SPIKE_FACTOR * long_normal,
                    "last_bar": session == last,
                }
            )
    per_session = Counter(a["session"] for a in found)
    early = {
        d.isoformat() for year in {int(s[:4]) for s in calendar} for d in market_early_closes(year)
    }
    for a in found:
        s = a["session"]
        if per_session[s] >= max(SYSTEMIC_MIN, SYSTEMIC_SHARE * liquid[s]):
            a["label"] = "systemic"
        elif a["spike"]:
            a["label"] = "after_volume_spike"
        elif a["last_bar"]:
            a["label"] = "last_bar"
        else:
            a["label"] = "isolated"
        a["half_day"] = s in early
    return sorted(found, key=lambda a: (a["session"], a["symbol"])), dict(liquid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--confirmed", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    found, liquid = alerts(db)
    confirmed: dict[tuple[str, str], str] = {}
    if args.confirmed:
        with args.confirmed.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                confirmed[(row["session"], row["security_id"])] = row["evidence"]
    for a in found:
        a["status"] = (
            "confirmed"
            if (a["session"], str(a["security_id"])) in confirmed
            or (a["session"], "*") in confirmed
            else "potential"
        )
    print(f"Partial-day audit of {args.db.name} (read-only)")
    print(
        f"rule: trades < {ALERT_RATIO:.0%} of the 20-session median, for securities with a "
        f"median >= {LIQUID_MEDIAN_TRADES}; systemic when >= {SYSTEMIC_SHARE:.0%} of liquid "
        f"securities (and >= {SYSTEMIC_MIN}) alert on the same session"
    )
    print(f"liquid security-sessions examined: {sum(liquid.values())}; alerts: {len(found)}")
    by_label = Counter((a["label"], a["status"]) for a in found)
    for (label, status), count in sorted(by_label.items()):
        print(f"  {label:20} {status:10} {count}")
    systemic = Counter(a["session"] for a in found if a["label"] == "systemic")
    for session, count in sorted(systemic.items()):
        print(f"  systemic session {session}: {count} of {liquid[session]} liquid securities")
    print(f"half-day alerts: {sum(a['half_day'] for a in found)}")
    if args.out:
        with args.out.open("w") as f:
            f.write(
                "session\tsecurity_id\tsymbol\ttrades\tnormal_trades\tratio\tlabel\tstatus"
                "\thalf_day\n"
            )
            for a in found:
                f.write(
                    f"{a['session']}\t{a['security_id']}\t{a['symbol']}\t{a['trades']}"
                    f"\t{a['normal_trades']:g}\t{a['ratio']:.4f}\t{a['label']}\t{a['status']}"
                    f"\t{a['half_day']}\n"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
