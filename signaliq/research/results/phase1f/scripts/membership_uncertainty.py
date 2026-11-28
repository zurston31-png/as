#!/usr/bin/env python3
# ruff: noqa: E501
"""Membership uncertainty from flagged bars that cannot be repaired (2021-10-25, 2022-03-08).

A flagged bar lost trades, so its true volume is at least the stored one; its security's
63-session median can only rise, and by at most one step (the bar's value at +infinity). A
decision is **certain** when no eligible non-member with a flagged bar in its window could rise
above the 100th member's median. Otherwise each such non-member is listed with the true dollar
volume its flagged bar would need. Eligibility cannot change: a flagged bar is still an active
session, and the price filter uses the decision day only.

    python research/results/phase1f/scripts/membership_uncertainty.py DB [--all-bars]

--all-bars: conservative variant; every bar on 2021-10-25 and 2022-03-08 counts as possibly
short, not only the bars the partial-day audit flags.
"""

import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_partial_days import alerts  # noqa: E402

from app.universe import pipeline as p  # noqa: E402

UNREPAIRED = {"2021-10-25", "2022-03-08"}


def main() -> int:
    db_path = Path(sys.argv[1]).resolve()
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    found, _ = alerts(db)
    flagged = {(a["security_id"], a["session"]) for a in found if a["label"] == "systemic"}
    if "--all-bars" in sys.argv:
        flagged = set(
            db.execute(
                "select security_id, session from security_bars where session in (?, ?)",
                sorted(UNREPAIRED),
            ).fetchall()
        )
    leftover = sorted({s for _, s in flagged} - UNREPAIRED)
    print(f"Membership uncertainty in {db_path.name}")
    print(
        f"flagged bars ({'every bar on the two sessions' if '--all-bars' in sys.argv else 'systemic'}): {len(flagged)}; sessions {sorted({s for _, s in flagged})}"
    )
    if leftover:
        print(f"WARNING: flagged bars on sessions outside 2021-10-25/2022-03-08: {leftover}")
    calendar = [d.isoformat() for d in p.trading_days(p.DATA_START, p.DATA_END)]
    index = {d: i for i, d in enumerate(calendar)}
    value = defaultdict(dict)
    symbol = {}
    for sid, s, c, v, sym in db.execute(
        "select security_id, session, close, volume, symbol from security_bars"
    ):
        value[sid][s] = c * v
        symbol[sid] = sym
    rows = defaultdict(list)
    for d, sid, el, med, rank, mem in db.execute(
        "select decision_session, security_id, eligible, median_dollar_volume, rank, member from membership"
    ):
        rows[str(d)[:10]].append((sid, el, med, rank, mem))
    uncertain_total = 0
    for d in sorted(rows):
        i = index[d]
        window = calendar[i - 62 : i + 1]
        hit = UNREPAIRED & set(window)
        if not hit:
            continue
        members = [(sid, med) for sid, el, med, rank, mem in rows[d] if mem]
        m100 = min(med for _, med in members)
        last = min(members, key=lambda x: x[1])[0]
        affected = [r for r in rows[d] if r[1] and any((r[0], h) in flagged for h in hit)]
        checked = 0
        lines = []
        for sid, _, med, rank, mem in affected:
            vals = [value[sid].get(s, 0.0) for s in window]
            assert math.isclose(statistics.median(vals), med, rel_tol=1e-9), (d, sid)
            if mem:
                continue  # a member's median can only rise: it stays a member
            checked += 1
            bad = [k for k, s in enumerate(window) if (sid, s) in flagged]
            top = list(vals)
            for k in bad:
                top[k] = math.inf
            ceiling = statistics.median(top)
            if ceiling <= m100:
                continue
            lo, hi = max(vals[k] for k in bad), m100 * 100
            for _ in range(200):
                mid = (lo + hi) / 2
                trial = list(vals)
                for k in bad:
                    trial[k] = mid
                lo, hi = (mid, hi) if statistics.median(trial) <= m100 else (lo, mid)
            k = bad[0]
            neigh = [value[sid].get(window[j], 0.0) for j in (k - 1, k + 1) if 0 <= j < len(window)]
            lines.append(
                f"    {symbol[sid]} (rank {rank}, median {med:,.0f}; at most {ceiling:,.0f}): enters if its "
                f"{window[k]} dollar volume was above {hi:,.0f} (stored {vals[k]:,.0f}; neighbouring sessions "
                + ", ".join(f"{x:,.0f}" for x in neigh)
                + ")"
            )
        state = "UNCERTAIN" if lines else "certain"
        uncertain_total += bool(lines)
        print(
            f"\n{d}: window holds {sorted(hit)}; {len(affected)} eligible securities with a flagged bar, "
            f"{checked} of them non-members; 100th member {symbol[last]} at {m100:,.0f} -> {state}"
        )
        for line in lines:
            print(line)
    print(f"\ndecisions uncertain: {uncertain_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
