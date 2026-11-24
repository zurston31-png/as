#!/usr/bin/env python3
"""Which securities kept out for a conflicting or unknown type would have been members?
Read-only; changes nothing.

    python scripts/excluded_type_impact.py [--report PATH]

Recomputes point-in-time membership exactly as pipeline.decide_membership does (same rule,
raw daily bars, held-out split checks, blank-check exclusions), once as stored and once
with every security whose type is in conflict or unknown treated as common stock. Lists
each such security that would then have been a member, with the decisions, and the
members it would have displaced. This measures the effect of the existing type rule
(only types backed by evidence count); it does not change it.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.data.validation import trading_days
from app.database import create_session_factory, session_scope
from app.universe.database import open_research_database
from app.universe.evidence import blank_check_exclusions
from app.universe.membership import DailyRow, MembershipRule, decide, month_end_sessions
from app.universe.models import Security, SecurityBar, SplitCheck, TypeEvidence
from app.universe.pipeline import (
    DATA_END,
    DATA_START,
    MEMBERSHIP_FIRST_DECISION,
    MEMBERSHIP_LAST_DECISION,
)
from app.universe.sectypes import USABLE_STATUSES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rule = MembershipRule(size=args.size)
    lines: list[str] = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    engine = open_research_database()
    with session_scope(create_session_factory(engine)) as s:
        history = defaultdict(list)
        for bar in s.execute(
            select(SecurityBar).order_by(SecurityBar.security_id, SecurityBar.session)
        ).scalars():
            history[bar.security_id].append(DailyRow(bar.session, bar.close, bar.volume))
        securities = {x.security_id: x for x in s.execute(select(Security)).scalars()}
        stored = {
            sid: x.security_type if x.type_status in USABLE_STATUSES else "unknown"
            for sid, x in securities.items()
        }
        open_types = {
            sid for sid, x in securities.items() if x.type_status in ("conflict", "unknown")
        }
        trial = {sid: ("common" if sid in open_types else t) for sid, t in stored.items()}
        votes = defaultdict(list)
        for e in s.execute(select(TypeEvidence)).scalars():
            votes[e.security_id].append(f"{e.source}={e.security_type}")
        held = {
            c.security_id: "price_adjustment_under_review"
            for c in s.execute(
                select(SplitCheck).where(SplitCheck.verdict == "magnitude_mismatch")
            ).scalars()
            if c.security_id is not None
        }
        dated = blank_check_exclusions(s)
        calendar = trading_days(DATA_START, DATA_END)
        gained = defaultdict(list)
        displaced = defaultdict(list)
        for when in month_end_sessions(MEMBERSHIP_FIRST_DECISION, MEMBERSHIP_LAST_DECISION):
            cal = [d for d in calendar if d <= when]
            before = {
                d.security_id
                for d in decide(when, history, stored, rule, cal, held, dated)
                if d.member
            }
            after = {
                d.security_id
                for d in decide(when, history, trial, rule, cal, held, dated)
                if d.member
            }
            for sid in after - before:
                gained[sid].append(when)
            for sid in before - after:
                displaced[sid].append(when)
        say(f"Securities with a conflicting or unknown type: {len(open_types)}")
        say(f"Of these, would have been a member if treated as common stock: {len(gained)}")
        for sid, days in sorted(gained.items(), key=lambda x: -len(x[1])):
            x = securities[sid]
            say(
                f"  id {sid} {x.last_symbol} ({x.type_status}, {x.status}): {len(days)} decisions "
                f"{days[0]}..{days[-1]}; evidence: {'; '.join(votes[sid]) or 'none'}"
            )
        say(f"Members that would have been displaced: {len(displaced)}")
        for sid, days in sorted(displaced.items(), key=lambda x: -len(x[1])):
            say(f"  id {sid} {securities[sid].last_symbol}: {len(days)} decisions")
    engine.dispose()
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
