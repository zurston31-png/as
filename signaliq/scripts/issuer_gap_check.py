#!/usr/bin/env python3
"""Which common stocks have no identified SEC issuer, and could it matter for membership?
Cache only (no network), read-only.

    python scripts/issuer_gap_check.py [--report PATH]

Applies the report's rules (4e): a common stock without dated issuer evidence and without
a documented reason (a curated one, or trading only before the first membership decision).
For each, shows each ticker period, today's holder of the ticker in SEC's ticker list (a
candidate only, never evidence) and how many filings that company's cached submissions
list during the period, and the security's membership (decisions as a member, best rank,
decisions where it was eligible): without an issuer there is no SIC, so a blank-check
company would not be excluded. Also lists, for every member, the issuer evidence behind
each membership decision. Changes nothing.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.universe import curated, evidence, sec
from app.universe import pipeline as p
from app.universe import sec_history as sh
from app.universe.database import research_database_url
from app.universe.models import Membership, Security, SecurityIssuer, TickerHistory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    lines: list[str] = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    path = research_database_url().removeprefix("sqlite:///")
    engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(f"file:{path}?mode=ro", uri=True),
    )
    with Session(engine) as s:
        client = sec.CachedSec(s, network=False)
        today = evidence._ticker_today(client)
        issuers = set(s.execute(select(SecurityIssuer.security_id)).scalars())
        gaps = [
            x
            for x in s.execute(select(Security).order_by(Security.security_id)).scalars()
            if x.security_id not in issuers
            and x.security_type == "common"
            and (x.last_symbol, x.last_trade_session) not in curated.CURATED_NO_SEC_ISSUER
            and x.last_trade_session >= p.MEMBERSHIP_FIRST_DECISION
        ]
        say(f"Common stocks without an issuer and without a documented reason: {len(gaps)}")
        for x in gaps:
            say(
                f"\nid {x.security_id} {x.last_symbol} ({x.status}, {x.type_status}); "
                f"traded {x.first_session}..{x.last_trade_session}"
            )
            for h in s.execute(
                select(TickerHistory)
                .where(TickerHistory.security_id == x.security_id)
                .order_by(TickerHistory.valid_from)
            ).scalars():
                held = today.get(h.symbol)
                text = f"  {h.symbol} {h.valid_from}..{h.valid_to}: "
                if held is None:
                    text += "not in SEC's ticker list today"
                elif x.status == "ended":
                    text += (
                        f"today held by CIK {held[0]} ({held[1]}); not used, the security "
                        "ended (step 2 is for securities still trading)"
                    )
                else:
                    company = sh.submissions(client, held[0])
                    filed = [d for _, d, _ in (company.filings if company else ())]
                    inside = sum(1 for d in filed if h.valid_from <= d <= h.valid_to)
                    text += (
                        f"today held by CIK {held[0]} ({held[1]}); its cached submissions list "
                        f"{len(filed)} filings before 2025"
                        + (f" ({min(filed)}..{max(filed)})" if filed else "")
                        + f", {inside} inside this ticker period"
                    )
                say(text)
            rows = list(
                s.execute(
                    select(Membership)
                    .where(Membership.security_id == x.security_id)
                    .order_by(Membership.decision_session)
                ).scalars()
            )
            members = [r.decision_session for r in rows if r.member]
            eligible = [r for r in rows if r.eligible]
            ranks = [r.rank for r in rows if r.rank is not None]
            say(
                f"  membership: member at {len(members)} of {len(rows)} decisions"
                + (f" ({members[0]}..{members[-1]})" if members else "")
                + f"; eligible at {len(eligible)}; best rank {min(ranks) if ranks else None}"
            )
        points = evidence.stored_issuer_points(s)
        members = sorted(
            {
                m.security_id
                for m in s.execute(select(Membership).where(Membership.member == 1)).scalars()
            }
        )
        say(f"\nMembers: {len(members)}; issuer behind each membership decision:")
        missing = later = 0
        for sid in members:
            days = [
                m.decision_session
                for m in s.execute(
                    select(Membership).where(Membership.security_id == sid, Membership.member == 1)
                ).scalars()
            ]
            held = [evidence.issuer_on(points.get(sid, []), d) for d in days]
            missing += sum(1 for h, _ in held if h is None)
            later += sum(1 for h, flag in held if h is not None and flag)
            ciks = sorted({h.cik for h, _ in held if h is not None})
            kinds = sorted({h.how for h, _ in held if h is not None})
            say(
                f"  id {sid} {s.get(Security, sid).last_symbol}: {len(days)} decisions; "
                f"CIK {', '.join(ciks) or 'NONE'}; evidence {', '.join(kinds) or 'none'}"
            )
        say(f"Member decisions without an issuer: {missing}; resting on a later document: {later}")
    engine.dispose()
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
