"""Compare membership decision by decision between two databases (read-only)."""
import sqlite3, sys
def load(p):
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    out = {}
    for row in c.execute("select decision_session, security_id, rule_fingerprint, eligible, reason, raw_close, median_dollar_volume, rank, member from membership order by 1, 2"):
        out.setdefault(row[0], []).append(row[1:])
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
assert a, "no membership rows"
differ = [d for d in sorted(a.keys() | b.keys()) if a.get(d) != b.get(d)]
members = {d: sorted(r[0] for r in rows if r[-1]) for d, rows in a.items()}
print(f"decisions: {len(a)} vs {len(b)}; rows {sum(map(len, a.values()))} vs {sum(map(len, b.values()))}")
print(f"members per decision: min {min(map(len, members.values()))}, max {max(map(len, members.values()))}")
print(f"decisions differing (all columns, every row): {len(differ)} {differ[:5]}")
print("MEMBERSHIP: ALL DECISIONS IDENTICAL" if not differ and len(a) == 59 else "MEMBERSHIP: DIFFERENCE FOUND")
