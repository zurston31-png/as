"""Independent audit of the stage's completeness proof (read-only, separate code).

The stage (``app.universe.stage``) fetches daily data only for tickers that rank within
3 x N by monthly traded value in some screen month, and then for every ticker whose upper
bound on the 63-session median dollar volume reaches the N-th member's at some decision.
It reports "left at the end: 0". This script re-derives that from the stored raw tables
with its own code, opening the database read-only through SQLite itself. It does not import
the stage, membership or pipeline modules.

Checks:

1. Pool: every candidate has a recorded monthly fetch (a cache marker, even when Alpaca had
   no bars), so no candidate was skipped silently.
2. Screen: each candidate's best monthly rank (traded value = vwap x volume, months with a
   trade, 2019-09 to 2024-11), recomputed, equals the stored one, and the selection follows
   from it (rank <= 3 x N gives "screen").
3. Threshold: at each decision, every member's 63-session median of raw close x volume
   (sessions without a trade count as 0) recomputed from daily bars equals the stored one,
   and the N-th member's value is the minimum.
4. Bound premises, checked on the data rather than assumed:
   - each decision's 63-session window starts no earlier than three calendar months
     before the decision month (so four monthly bars cover it);
   - the decision is the last session of its month.
5. Bound: for every ticker without daily data and every decision, the upper bound
   2 x sum(monthly high x volume over the four months) / 63 (zero if the decision month has
   no trade or its high is below the $5 minimum) is below the N-th member's value. The
   count of tickers whose bound reaches it must be 0.
6. Added tickers: each "completeness" candidate's stated bound, recomputed, matches, and
   it was given daily data (it is in the batch).

    python scripts/audit_completeness.py [--db DB] [--report F]
    (without --db: the file of SIGNALIQ_SIP_RESEARCH_DATABASE_URL)
"""

import argparse
import math
import os
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

SCREEN_FIRST, SCREEN_LAST = date(2019, 9, 1), date(2024, 11, 1)
MARGIN = 3
WINDOW = 63
MIN_PRICE = 5.0
REASON = re.compile(
    r"could have been a member on (\d{4}-\d{2}-\d{2}): dollar-volume bound ([\d,]+) >= "
    r"N-th member ([\d,]+)"
)


def _month(day: date) -> date:
    return date(day.year, day.month, 1)


def _back(month: date, k: int) -> date:
    y, m = divmod(month.year * 12 + month.month - 1 - k, 12)
    return date(y, m + 1, 1)


def _day(value) -> date:
    return date.fromisoformat(str(value)[:10])


def _database_path(argument: str | None) -> Path:
    if argument:
        return Path(argument)
    url = os.environ.get("SIGNALIQ_SIP_RESEARCH_DATABASE_URL", "")
    if not url.startswith("sqlite:///"):
        raise SystemExit("give --db or set SIGNALIQ_SIP_RESEARCH_DATABASE_URL")
    return Path(url.removeprefix("sqlite:///"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    path = _database_path(args.db).resolve()
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    lines: list[str] = []
    failures = 0

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    def check(ok: bool, text: str):
        nonlocal failures
        failures += not ok
        say(f"  {'ok  ' if ok else 'FAIL'} {text}")

    say(f"Completeness audit of {path.name} (opened read-only)")

    # Decisions, members and N.
    members: dict[date, list[tuple[int, float, int]]] = defaultdict(list)
    for decision, sid, median, rank in db.execute(
        "select decision_session, security_id, median_dollar_volume, rank from membership "
        "where member = 1"
    ):
        members[_day(decision)].append((sid, median, rank))
    sizes = {len(v) for v in members.values()}
    if len(sizes) != 1:
        say(f"members per decision vary: {sorted(sizes)}")
    n = max(sizes)
    decisions = sorted(members)
    say(f"Decisions: {len(decisions)} ({decisions[0]} to {decisions[-1]}); N = {n}")

    candidates = {
        r[0]: {"rank": r[1], "selection": r[2], "reason": r[3]}
        for r in db.execute("select symbol, best_rank, selection, reason from candidates")
    }
    counts = defaultdict(int)
    for c in candidates.values():
        counts[c["selection"]] += 1
    say(f"Candidates: {len(candidates)}; by selection {dict(sorted(counts.items()))}")

    # 1. Pool.
    say("\n1. POOL")
    fetched = {
        r[0]
        for r in db.execute(
            "select request_key from external_responses where source = 'monthly_fetched'"
        )
    }
    missing = sorted(set(candidates) - fetched)
    check(
        not missing,
        "candidates with a recorded monthly fetch: "
        f"{len(candidates) - len(missing)} of {len(candidates)}"
        + (f"; missing: {missing[:20]}" if missing else ""),
    )

    # 2. Screen.
    say("\n2. SCREEN")
    monthly: dict[str, dict[date, tuple[float, float, int, int]]] = defaultdict(dict)
    by_month: dict[date, list[tuple[float, str]]] = defaultdict(list)
    for symbol, month, high, volume, vwap, trades in db.execute(
        "select symbol, month, high, volume, vwap, trade_count from monthly_bars"
    ):
        m = _day(month)
        monthly[symbol][m] = (high, vwap, volume, trades)
        if volume > 0 and trades > 0 and SCREEN_FIRST <= m <= SCREEN_LAST:
            by_month[m].append((vwap * volume, symbol))
    best: dict[str, int] = {}
    for m in sorted(by_month):
        for rank, (_, symbol) in enumerate(sorted(by_month[m], key=lambda x: (-x[0], x[1])), 1):
            if rank < best.get(symbol, math.inf):
                best[symbol] = rank
    rank_diff = [s for s, c in candidates.items() if c["rank"] != best.get(s)]
    check(
        not rank_diff,
        f"best monthly ranks recomputed and equal to the stored ones: "
        f"{len(candidates) - len(rank_diff)} of {len(candidates)}"
        + (f"; differ: {rank_diff[:20]}" if rank_diff else ""),
    )
    wrong = [
        s
        for s, c in candidates.items()
        if c["selection"] != "completeness"
        and (c["selection"] == "screen") != (best.get(s) is not None and best[s] <= MARGIN * n)
    ]
    within = sum(1 for s in candidates if best.get(s) is not None and best[s] <= MARGIN * n)
    check(
        not wrong,
        f"tickers ranked within {MARGIN} x N = {MARGIN * n} in some month: {within}; "
        f"selection consistent with the rank for every candidate"
        + (f"; inconsistent: {wrong[:20]}" if wrong else ""),
    )

    # 3. Threshold from daily bars.
    say("\n3. N-TH MEMBER, RECOMPUTED FROM DAILY BARS")
    calendar = sorted(
        {_day(r[0]) for r in db.execute("select distinct session from security_bars")}
    )
    member_ids = {sid for v in members.values() for sid, _, _ in v}
    daily: dict[int, dict[date, float]] = defaultdict(dict)
    for sid, session, close, volume in db.execute(
        "select security_id, session, close, volume from security_bars"
    ):
        if sid in member_ids:
            daily[sid][_day(session)] = close * volume
    index = {d: i for i, d in enumerate(calendar)}
    threshold: dict[date, float] = {}
    total_members = sum(len(v) for v in members.values())
    mismatches = []
    premise = []
    for d in decisions:
        i = index.get(d)
        if i is None or i + 1 < WINDOW:
            premise.append(f"{d}: not in the calendar or too early")
            continue
        window = calendar[i - WINDOW + 1 : i + 1]
        if window[0] < _back(_month(d), 3):
            premise.append(f"{d}: window starts {window[0]}, before {_back(_month(d), 3)}")
        if i + 1 < len(calendar) and _month(calendar[i + 1]) == _month(d):
            premise.append(f"{d}: not the last session of its month")
        values = []
        for sid, stored, _rank in members[d]:
            median = statistics.median(daily[sid].get(s, 0.0) for s in window)
            values.append(median)
            if not math.isclose(median, stored or 0.0, rel_tol=1e-9, abs_tol=1e-6):
                mismatches.append(f"{d} id {sid}: {median} against stored {stored}")
        threshold[d] = min(values) if len(values) >= n else 0.0
        stored_min = min((v or 0.0) for _, v, _ in members[d])
        if not math.isclose(threshold[d], stored_min, rel_tol=1e-9, abs_tol=1e-6):
            mismatches.append(f"{d}: N-th member {threshold[d]} against stored {stored_min}")
    check(
        not mismatches,
        f"member medians recomputed and equal to the stored ones: "
        f"{total_members - len(mismatches)} of {total_members}"
        + (f"; differ: {mismatches[:10]}" if mismatches else ""),
    )
    say(
        f"  N-th member's 63-session median dollar volume: lowest {min(threshold.values()):,.0f} "
        f"({min(threshold, key=threshold.get)}), highest {max(threshold.values()):,.0f} "
        f"({max(threshold, key=threshold.get)})"
    )

    say("\n4. PREMISES OF THE BOUND (checked on the calendar)")
    check(
        not premise,
        "every decision is the last session of its month and its 63-session window "
        "lies within the decision month and the three before it"
        + (f"; problems: {premise}" if premise else ""),
    )

    # 5. Bound for every ticker without daily data.
    say("\n5. TICKERS WITHOUT DAILY DATA")

    def bound(symbol: str, d: date) -> float:
        bars = monthly.get(symbol, {})
        now = bars.get(_month(d))
        if now is None or now[2] <= 0 or now[0] < MIN_PRICE:
            return 0.0
        total = 0.0
        for k in range(4):
            b = bars.get(_back(_month(d), k))
            if b is not None and b[2] > 0:
                total += b[0] * b[2]
        return 2.0 * total / WINDOW

    left_out = [s for s, c in candidates.items() if c["selection"] == "not_selected"]
    reaching = []
    closest = (math.inf, None, None)
    for symbol in left_out:
        for d in decisions:
            b = bound(symbol, d)
            if b <= 0:
                continue
            gap = threshold[d] / b
            if b >= threshold[d]:
                reaching.append(f"{symbol} on {d}: bound {b:,.0f} >= {threshold[d]:,.0f}")
            if gap < closest[0]:
                closest = (gap, symbol, d)
    check(
        not reaching,
        f"tickers without daily data whose bound reaches the N-th member at some decision: "
        f"{len(reaching)} (of {len(left_out)} checked at {len(decisions)} decisions)"
        + (f": {reaching[:20]}" if reaching else ""),
    )
    if closest[1]:
        say(f"  closest: {closest[1]} on {closest[2]}, N-th member {closest[0]:.3f} x its bound")

    # 6. Tickers added by the proof.
    say("\n6. TICKERS ADDED BY THE COMPLETENESS PROOF")
    batch = {r[0] for r in db.execute("select symbol from batch_symbols")}
    added = {s: c for s, c in candidates.items() if c["selection"] == "completeness"}
    bad_reason, not_in_batch, above_final = [], [], 0
    for symbol, c in sorted(added.items()):
        if symbol not in batch:
            not_in_batch.append(symbol)
        found = REASON.search(c["reason"])
        if not found:
            bad_reason.append(f"{symbol}: unreadable reason")
            continue
        d, stated = date.fromisoformat(found.group(1)), float(found.group(2).replace(",", ""))
        nth = float(found.group(3).replace(",", ""))
        again = bound(symbol, d)
        if abs(again - stated) > 1.0 or stated < nth:
            bad_reason.append(
                f"{symbol} {d}: bound {again:,.0f} against stated {stated:,.0f} (N-th {nth:,.0f})"
            )
        above_final += again >= threshold.get(d, math.inf)
    check(
        not bad_reason,
        f"added tickers whose stated bound recomputes and reached the N-th member "
        f"of that round: {len(added) - len(bad_reason)} of {len(added)}"
        + (f"; problems: {bad_reason[:20]}" if bad_reason else ""),
    )
    check(
        not not_in_batch,
        "added tickers given daily data (in the batch): "
        f"{len(added) - len(not_in_batch)} of {len(added)}"
        + (f"; missing: {not_in_batch}" if not_in_batch else ""),
    )
    say(
        f"  of these, {above_final} still reach the final N-th member"
        + (
            "; the others reached the lower threshold of an earlier round, before the added "
            "tickers raised it"
            if above_final < len(added)
            else ""
        )
    )

    say("")
    say(f"RESULT: {'PASS' if failures == 0 else f'FAIL ({failures} checks)'}")
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
