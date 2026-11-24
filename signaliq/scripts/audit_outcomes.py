#!/usr/bin/env python3
"""Audit every ended security's outcome, confidence label, evidence and exit value. Read-only.

    python scripts/audit_outcomes.py

Checks, for each ended security in the research database:

1. it has a confidence label (two independent sources, one source, unresolved, ...);
2. a resolved outcome has exit evidence; "two sources" means two different source
   families (all SEC records count as one), "one source" exactly one;
3. an unresolved outcome carries no value;
4. every value has a basis and every basis a value; an assumed zero is labelled as an
   assumption with the recovery not confirmed.

Exits with status 1 if any check fails.
"""

import sqlite3
import sys
from collections import Counter
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "sip_research.db"
LABELS = {
    "confirmed": "two independent sources",
    "single_source": "lower confidence (one source)",
    "unresolved": "unresolved",
    "conflict": "conflict",
    "not_an_exit": "not an exit",
}
EXIT_EVIDENCE = {
    "acquired",
    "acquired_cash",
    "acquired_stock",
    "acquired_stock_and_cash",
    "deal_terms",
    "bankruptcy",
    "failure",
    "redeemed",
}


def main() -> int:
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ended = db.execute(
        "select security_id, last_symbol, outcome, outcome_status, assumed_exit_value, "
        "value_basis, exit_assumption from securities where status='ended' order by security_id"
    ).fetchall()
    problems = []
    for sid, sym, _outcome, status, value, basis, note in ended:
        evidence = db.execute(
            "select source, category from exit_evidence where security_id=?", (sid,)
        ).fetchall()
        if status not in LABELS:
            problems.append((sid, sym, f"no confidence label ({status})"))
        exits = [(s, c) for s, c in evidence if c in EXIT_EVIDENCE]
        families = {"sec" if s.startswith("sec") else s for s, _ in exits}
        if status in ("confirmed", "single_source") and not exits:
            problems.append((sid, sym, "resolved outcome without exit evidence"))
        if status == "confirmed" and len(families) < 2:
            problems.append((sid, sym, f"two sources claimed, families {sorted(families)}"))
        if status == "single_source" and len(families) != 1:
            problems.append((sid, sym, f"one source claimed, families {sorted(families)}"))
        if status == "unresolved" and value is not None:
            problems.append((sid, sym, "unresolved but valued"))
        if basis == "assumed_zero" and (
            "assumption" not in (note or "") or "not confirmed" not in (note or "")
        ):
            problems.append((sid, sym, "assumed zero not labelled as an assumption"))
        if value is None and basis not in (None, "none"):
            problems.append((sid, sym, "basis without value"))
        if value is not None and basis in (None, "none"):
            problems.append((sid, sym, "value without basis"))
    print(f"Ended securities: {len(ended)}")
    print(f"Confidence labels: {dict(Counter(LABELS.get(r[3], 'MISSING') for r in ended))}")
    print(f"Value basis: {dict(Counter(r[5] for r in ended))}")
    print(
        "Values presented as payouts (stated cash or verified deal terms): "
        f"{sum(1 for r in ended if r[5] in ('stated_cash', 'deal_terms'))}"
    )
    print(
        "Zero values that are assumptions (event confirmed, recovery not confirmed): "
        f"{sum(1 for r in ended if r[5] == 'assumed_zero')}"
    )
    print(f"Problems: {len(problems)}")
    for problem in problems:
        print(f"  PROBLEM {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
