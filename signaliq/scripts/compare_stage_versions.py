#!/usr/bin/env python3
"""What changed between two runs of a stage: membership, types, issuers, outcomes, values.
Read-only on both databases.

    python scripts/compare_stage_versions.py OLD.db NEW.db [--report PATH]

Security ids are permanent (the registry), so the two runs are compared id by id. For
every membership decision: the members added and removed, each with the reason the other
run gave (not eligible because..., rank). Then every security whose type, issuer at its
last trade, outcome or exit value changed.
"""

import argparse
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def table(db, sql):
    return db.execute(sql).fetchall()


def why(rows: dict, sid: int, day) -> str:
    """How a run treated a security at a decision: its rank, or why it was not eligible."""
    r = rows.get((day, sid))
    if r is None:
        return "not in that run"
    return f"rank {r[4]}" if r[2] else f"not eligible: {r[3]}"


def nesting(old_rows: dict, new_rows: dict, k: int) -> dict:
    """Per decision of the old run: (old members, the new run's top ``k``) where they differ.

    Rows are membership rows keyed (decision, security_id) with (..., eligible, reason, rank,
    member). The top ``k`` of the new run are its eligible securities ranked 1..k. With the
    same ranking rule, a larger run's top ``k`` must equal the smaller run's members unless
    the evidence about a security in both runs changed; every difference must be explained.
    """
    differences = {}
    for day in sorted({key[0] for key in old_rows}):
        was = {key[1] for key, r in old_rows.items() if key[0] == day and r[5]}
        top = {
            key[1]
            for key, r in new_rows.items()
            if key[0] == day and r[2] and r[4] is not None and r[4] <= k
        }
        if was != top:
            differences[day] = (was, top)
    return differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--nesting",
        type=int,
        metavar="K",
        help="also check that the new run's top K equal the old run's members at every decision",
    )
    args = parser.parse_args()
    lines: list[str] = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    old, new = connect(args.old), connect(args.new)
    sec_sql = (
        "select security_id, last_symbol, status, security_type, type_status, outcome, "
        "outcome_status, assumed_exit_value, value_basis from securities"
    )
    a = {r[0]: r for r in table(old, sec_sql)}
    b = {r[0]: r for r in table(new, sec_sql)}
    reg = {
        r[0]: r
        for r in table(new, "select security_id, status, merged_into from security_registry")
    }
    name = {**{k: v[1] for k, v in a.items()}, **{k: v[1] for k, v in b.items()}}
    say(f"Old: {args.old.name} ({len(a)} securities); new: {args.new.name} ({len(b)} securities)")
    gone = sorted(set(a) - set(b))
    say(f"Securities only in the old run: {len(gone)}")
    for sid in gone:
        r = reg.get(sid)
        say(
            f"  id {sid} {name[sid]}: "
            + (
                f"now part of id {r[2]}"
                if r and r[1] == "merged"
                else f"registry status {r[1] if r else None}"
            )
        )
    added = sorted(set(b) - set(a))
    say(
        f"Securities only in the new run: {len(added)}"
        + "".join(f"\n  id {s} {name[s]}" for s in added)
    )

    m_sql = "select decision_session, security_id, eligible, reason, rank, member from membership"
    ma = {(r[0], r[1]): r for r in table(old, m_sql)}
    mb = {(r[0], r[1]): r for r in table(new, m_sql)}
    days = sorted({k[0] for k in ma} | {k[0] for k in mb})
    say("\nMEMBERSHIP")
    changed_days = 0
    entering, leaving = Counter(), Counter()
    for day in days:
        was = {k[1] for k, r in ma.items() if k[0] == day and r[5]}
        now = {k[1] for k, r in mb.items() if k[0] == day and r[5]}
        if was == now:
            continue
        changed_days += 1
        ins, outs = sorted(now - was), sorted(was - now)
        entering.update(ins)
        leaving.update(outs)

        say(
            f"  {day}: +{len(ins)} -{len(outs)}; in: "
            + ", ".join(f"{name.get(s)} (old: {why(ma, s, day)})" for s in ins)
            + "; out: "
            + ", ".join(f"{name.get(s)} (new: {why(mb, s, day)})" for s in outs)
        )
    say(f"Decisions with a different member list: {changed_days} of {len(days)}")
    say(
        "Securities that entered at some decision: "
        + ", ".join(f"{name.get(s)} x{n}" for s, n in entering.most_common())
    )
    say(
        "Securities that left at some decision: "
        + ", ".join(f"{name.get(s)} x{n}" for s, n in leaving.most_common())
    )
    distinct_a = {k[1] for k, r in ma.items() if r[5]}
    distinct_b = {k[1] for k, r in mb.items() if r[5]}
    say(f"Distinct members: old {len(distinct_a)}, new {len(distinct_b)}")

    def reasons(rows):
        return Counter(r[3] for r in rows.values() if not r[2])

    say(f"Not-eligible reasons, old: {dict(sorted(reasons(ma).items()))}")
    say(f"Not-eligible reasons, new: {dict(sorted(reasons(mb).items()))}")

    if args.nesting:
        say(f"\nNESTING (the new run's top {args.nesting} against the old run's members)")
        differences = nesting(ma, mb, args.nesting)
        for day, (was, top) in differences.items():
            say(
                f"  {day}: only in the old members: "
                + ", ".join(f"{name.get(s)} (new: {why(mb, s, day)})" for s in sorted(was - top))
                + "; only in the new top: "
                + ", ".join(f"{name.get(s)} (old: {why(ma, s, day)})" for s in sorted(top - was))
            )
        days_old = sorted({key[0] for key in ma})
        say(
            "  RESULT: "
            + (
                f"nested at every decision ({len(days_old)})"
                if not differences
                else f"{len(differences)} of {len(days_old)} decisions differ; "
                "each must be explained"
            )
        )

    say("\nTYPES (security_type, type_status)")
    grid = Counter()
    for sid in sorted(set(a) & set(b)):
        before, after = (a[sid][3], a[sid][4]), (b[sid][3], b[sid][4])
        if before != after:
            grid[(before, after)] += 1
    for (before, after), n in sorted(grid.items(), key=lambda x: -x[1]):
        say(f"  {before[0]}/{before[1]} -> {after[0]}/{after[1]}: {n}")
    for sid in sorted(set(a) & set(b)):
        before, after = (a[sid][3], a[sid][4]), (b[sid][3], b[sid][4])
        if before != after:
            say(f"    id {sid} {name[sid]}: {before[0]}/{before[1]} -> {after[0]}/{after[1]}")

    say("\nISSUER AT THE LAST TRADE")
    ia = {r[0]: r for r in table(old, "select security_id, cik, name, how from security_issuers")}
    ib = {r[0]: r for r in table(new, "select security_id, cik, name, how from security_issuers")}
    say(
        f"  Known: old {len(ia)}, new {len(ib)}; same CIK "
        f"{sum(1 for s in ia if s in ib and ia[s][1] == ib[s][1])}"
    )
    for label, ids in (
        ("found only now", sorted(set(ib) - set(ia))),
        ("lost now", sorted(set(ia) - set(ib))),
        ("different CIK", sorted(s for s in set(ia) & set(ib) if ia[s][1] != ib[s][1])),
    ):
        say(f"  {label}: {len(ids)}")
        for s in ids:
            say(
                f"    id {s} {name.get(s)}: old {ia[s][1:] if s in ia else None} -> "
                f"new {ib[s][1:] if s in ib else None}"
            )

    say("\nOUTCOMES AND EXIT VALUES (ended securities)")
    for sid in sorted(set(a) & set(b)):
        x, y = a[sid], b[sid]
        if x[2] != y[2] or (x[5], x[6], x[7], x[8]) != (y[5], y[6], y[7], y[8]):
            say(
                f"  id {sid} {name[sid]}: status {x[2]} -> {y[2]}; "
                f"outcome {x[5]}/{x[6]} -> {y[5]}/{y[6]}; "
                f"value {x[7]} ({x[8]}) -> {y[7]} ({y[8]})"
            )
    by_basis = defaultdict(Counter)
    for label, rows in (("old", a), ("new", b)):
        for r in rows.values():
            if r[2] == "ended":
                by_basis[label][r[8]] += 1
    say(f"  Value basis, old: {dict(sorted(by_basis['old'].items()))}")
    say(f"  Value basis, new: {dict(sorted(by_basis['new'].items()))}")
    old.close()
    new.close()
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
