"""Evidence listings for the two proposed rules, read-only (nothing is adopted or stored).

1. Dated security types: every security whose own cover pages give more than one class
   type over time, with each run of pages (dates, form, the class title, the document),
   and every decision at which the dated type differs from the undated one.
2. Co-registrant choice: every ticker period where two companies' cover pages overlapped,
   the company the current rule keeps, the company the proposed rule would keep, and why
   (from the shared cover page itself).

The database is opened read-only and SEC answers come only from the cache: a missing
answer is listed, never requested. Membership impact is measured by full runs with the
rule switched on (see research/plans/phase1/), compared with Stage 1.1.

    SIGNALIQ_SIP_RESEARCH_READ_ONLY=1 SIGNALIQ_SEC_OFFLINE=1 \\
    SIGNALIQ_SIP_RESEARCH_DATABASE_URL=sqlite:///data/universe_stage.db \\
    python scripts/measure_rules.py --dated-types REPORT1 --coregistrant REPORT2
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import astuple
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from app.database import create_session_factory, session_scope
from app.universe import evidence, rules
from app.universe.database import open_research_database
from app.universe.membership import MembershipRule, month_end_sessions
from app.universe.models import Membership, Security
from app.universe.pipeline import (
    MEMBERSHIP_FIRST_DECISION,
    MEMBERSHIP_LAST_DECISION,
    membership_decisions,
)
from app.universe.sectypes import USABLE_STATUSES

STAGE_SIZE = 100


def _membership_check(s, type_changes, lines) -> None:
    """Exact membership impact: the run's own membership code, recomputed without storing."""
    rule = MembershipRule(size=STAGE_SIZE)
    stored = {
        (m.decision_session, m.security_id): (m.eligible, m.reason, m.rank, m.member)
        for m in s.execute(select(Membership)).scalars()
    }
    _, baseline = membership_decisions(s, rule)
    again = {
        (d.session, d.security_id): (int(d.eligible), d.reason, d.rank, int(d.member))
        for d in baseline
    }
    same = again == {k: (int(v[0]), v[1], v[2], int(v[3])) for k, v in stored.items()}
    lines.append(
        f"Membership recomputed with the rule off: {'identical to' if same else 'DIFFERENT FROM'} "
        f"the stored Stage 1.1 membership ({len(again)} security-decisions)"
    )
    _, with_rule = membership_decisions(s, rule, type_changes)
    members = defaultdict(set)
    members_rule = defaultdict(set)
    for d in baseline:
        if d.member:
            members[d.session].add(d.security_id)
    for d in with_rule:
        if d.member:
            members_rule[d.session].add(d.security_id)
    changed = [d for d in sorted(members) if members[d] != members_rule[d]]
    lines.append(f"Membership with the rule on: {len(changed)} of {len(members)} decisions differ")
    for d in changed:
        lines.append(
            f"  {d}: in {sorted(members_rule[d] - members[d])}, "
            f"out {sorted(members[d] - members_rule[d])}"
        )


def dated_types(s, out: Path) -> None:
    lines = []
    client = evidence.StrictCache(s)
    classes = rules.cover_classes(client, evidence.stored_issuer_points(s))
    changes = rules.type_changes(classes)
    decisions = month_end_sessions(MEMBERSHIP_FIRST_DECISION, MEMBERSHIP_LAST_DECISION)
    pages = sum(len(v) for v in classes.values())
    unread = sum(1 for v in classes.values() for c in v if c.why == rules.NOT_READ)
    conclusive = sum(1 for v in classes.values() for c in v if c.kind)
    other_missing = sorted(k for k in client.missing if not k.startswith("cover:"))
    lines.append(
        "DATED SECURITY TYPES (proposed rule, measured on the Stage 1.1 data, not adopted)"
    )
    lines.append(
        f"Cover pages in the issuer evidence: {pages} for {len(classes)} securities. In the "
        f"cache and read: {pages - unread}; conclusive (one recognised class): {conclusive}; "
        f"not in the cache, so not read: {unread}."
    )
    lines.append(
        "The issuer search reads the first and last cover page of each company's run and the "
        "pages its halving probes; the pages in between are evidence of the ticker from the "
        "filing list, but their class titles were never read. Reading them all is a fetch of "
        f"{unread} pages, which Phase 1 does not do."
    )
    if other_missing:
        lines.append(f"OTHER ANSWERS MISSING (filing lists): {other_missing[:20]}")
    lines.append(
        f"\nSecurities whose read cover pages give more than one type over time: {len(changes)}"
    )
    for sid, all_pages in sorted(changes.items()):
        security = s.get(Security, sid)
        undated = security.security_type if security.type_status in USABLE_STATUSES else "unknown"
        not_read = sum(1 for c in all_pages if c.why == rules.NOT_READ)
        lines.append(
            f"\n  id {sid} {security.last_symbol}: undated type {security.security_type} "
            f"({security.type_status}), counted for membership as {undated}; cover pages "
            f"{len(all_pages)}, of which not read {not_read}"
        )
        for kind, first, last, count in rules.type_runs(all_pages):
            lines.append(f"    {kind:9} {first.filed}..{last.filed}  {count} conclusive pages read")
            lines.append(f"      first: {first.filed} {first.form} CIK {first.cik} '{first.title}'")
            lines.append(f"             {first.url}")
            if last is not first:
                lines.append(f"      last:  {last.filed} {last.form} CIK {last.cik} '{last.title}'")
                lines.append(f"             {last.url}")
        rows = []
        for d in decisions:
            kind, why = rules.dated_type(all_pages, d)
            effective = kind or undated
            rows.append((d, effective, why))
        flips = [(d, k, w) for d, k, w in rows if (k == "common") != (undated == "common")]
        spans = []
        for d, k, w in rows:
            if spans and spans[-1][1] == k and spans[-1][2] == w:
                spans[-1][3] = d
            else:
                spans.append([d, k, w, d])
        lines.append("    type at each decision under the rule:")
        for first_d, k, w, last_d in spans:
            lines.append(f"      {first_d}..{last_d}: {k} ({w})")
        lines.append(
            f"    decisions where eligibility would change: {len(flips)}"
            + (f" ({flips[0][0]}..{flips[-1][0]})" if flips else "")
        )
    lines.append("")
    _membership_check(s, changes, lines)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2]), flush=True)


def coregistrant(s, out: Path) -> None:
    lines = ["CO-REGISTRANT CHOICE (proposed rule, measured, not adopted)"]
    stored = evidence.stored_issuer_points(s)
    client = evidence.StrictCache(s)
    rules.CHOICES.clear()
    import os

    os.environ[rules.COREGISTRANT_VARIABLE] = "1"
    try:
        replay, _conflicts = evidence.issuer_points(s, client, evidence.sec_findings(s, client))
    finally:
        os.environ.pop(rules.COREGISTRANT_VARIABLE, None)
    choices = [(sid, c) for sid, cs in sorted(rules.CHOICES.items()) for c in cs]
    changed = [(sid, c) for sid, c in choices if c.changed]
    undecided = [(sid, c) for sid, c in choices if c.reason.endswith("current rule")]
    lines.append(
        f"Overlapping ticker periods examined: {len(choices)}; the proposed rule keeps the other "
        f"company in {len(changed)}; the cover page decides as the current rule in "
        f"{len(choices) - len(changed) - len(undecided)}; undecided (current rule stands) in "
        f"{len(undecided)}; answers not in the cache: {len(client.missing)}"
    )
    securities = {x.security_id: x for x in s.execute(select(Security)).scalars()}
    for title, group in (
        ("WOULD CHANGE", changed),
        ("UNDECIDED (current rule stands)", undecided),
        ("SAME AS NOW", [x for x in choices if x not in changed and x not in undecided]),
    ):
        lines.append(f"\n{title}: {len(group)}")
        for sid, c in group:
            name = {p.cik: p.name for p in stored.get(sid, []) + replay.get(sid, [])}
            lines.append(
                f"  id {sid} {securities[sid].last_symbol} ({c.symbol}): keep CIK {c.kept_cik} "
                f"{name.get(c.kept_cik, '')}, not CIK {c.other_cik} {name.get(c.other_cik, '')}"
            )
            lines.append(f"    why: {c.reason}")
            if c.url:
                lines.append(f"    shared cover page: {c.url}")
    differ = sorted(
        sid
        for sid in set(stored) | set(replay)
        if sorted(map(astuple, stored.get(sid, []))) != sorted(map(astuple, replay.get(sid, [])))
    )
    lines.append(f"\nSecurities whose dated issuer evidence would differ: {len(differ)}")
    for sid in differ:
        before = evidence.issuer_on(stored[sid], securities[sid].last_trade_session)[0]
        after = evidence.issuer_on(replay[sid], securities[sid].last_trade_session)[0]
        lines.append(
            f"  id {sid} {securities[sid].last_symbol}: issuer at the last trade CIK {before.cik} "
            f"{before.name} -> CIK {after.cik} {after.name}"
        )
    if client.missing:
        lines.append(f"MISSING from the cache: {sorted(client.missing)[:20]}")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2]), flush=True)


def coregistrant_requirements(s, out: Path) -> None:
    """Every SEC answer the issuer step would need with the co-registrant rule on and that is
    not in the cache, without fetching any. Cache only (``StrictCache``)."""
    import math
    import os

    from app.universe import sec, sec_history

    lines = ["CO-REGISTRANT RULE: SEC ANSWERS NEEDED AND NOT IN THE CACHE (nothing fetched)"]
    stored = evidence.stored_issuer_points(s)
    client = evidence.StrictCache(s)
    os.environ[rules.COREGISTRANT_VARIABLE] = "1"
    try:
        records, _conflicts = evidence.issuer_records(s, client, evidence.sec_findings(s, client))
    finally:
        os.environ.pop(rules.COREGISTRANT_VARIABLE, None)
    missing = sorted(client.missing)
    lines.append(
        "Method: the issuer step (issuer_records, the only step that requests SEC answers;"
        " every later step reads the cache only) run with the rule on against a cache that"
        " records each answer it lacks instead of fetching it."
    )
    lines.append(f"\nFirst-level answers missing: {len(missing)}")
    lines += [f"  {key}" for key in missing]

    securities = {x.security_id: x for x in s.execute(select(Security)).scalars()}
    lines.append("\nCompanies the rule adds to a security's issuer evidence:")
    total_known, total_later, total_bound = len(missing), 0, 0
    for sid, _last, points, _sics in records:
        before = {p.cik for p in stored.get(sid, [])}
        for cik in sorted({p.cik for p in points} - before):
            security = securities[sid]
            first = security.first_session - timedelta(days=365)
            last = security.last_trade_session
            name = next(p.name for p in points if p.cik == cik)
            lines.append(
                f"  id {sid} {security.last_symbol}: CIK {cik} {name}; SIC history from {first}"
                f" to {last}"
            )
            text = client.get(sec_history.SUBMISSIONS_URL.format(cik=str(int(cik)).zfill(10)))
            if not text:
                lines.append("    submissions file: NOT in the cache")
                continue
            pages = sec_history.older_pages(text)
            have = [p for p in pages if client.get(sec.SUBMISSIONS_PAGE_URL.format(name=p))]
            lines.append(
                f"    submissions file: in the cache; older pages listed {len(pages)},"
                f" in the cache {len(have)}: {pages}"
            )
            company = sec_history.parse_submissions(text)
            dated = sorted({(f, a) for _, f, a in company.filings}) if company else []
            oldest = dated[0][0] if dated else None
            inside = [d for d in dated if first <= d[0] <= last]
            prior = [d for d in dated if d[0] < first]
            settled = bool(prior) or not pages or len(have) == len(pages)
            n = len(inside) + (1 if prior else 0)
            if settled:
                ends = ([prior[-1]] if prior else []) + ([inside[-1]] if inside else [])
                lines.append(
                    f"    filing headers that must be read (both ends of the interval, known"
                    f" now): {len(ends)}"
                )
                for filed, accession in ends:
                    folder = accession.replace("-", "")
                    url = (
                        f"{sec_history.ARCHIVES}/Archives/edgar/data/{int(cik)}/{folder}/"
                        f"{accession}.hdr.sgml"
                    )
                    state = "in the cache" if client.get(url) else "NOT in the cache"
                    lines.append(f"      {filed} {url} ({state})")
            else:
                total_later += 1
                lines.append(
                    f"    the cached file lists filings back to {oldest} only; the filing"
                    f" before {first} is on an older page, so 1 of the 2 end headers is known"
                    " only after that page is fetched"
                )
            bound = math.ceil(math.log2(max(n, 2)))
            total_bound += bound
            lines.append(
                f"    if the SIC changed inside the interval: up to {bound} more headers per"
                f" change (halving over about {n} filings); if not: none"
            )
    lines.append(
        f"\nTotal: {total_known} answers known now to be needed; {total_later} more known only"
        f" after an older page is read; up to {total_bound} more per SIC change. A changed"
        " issuer can also change later issuer-confirmed rename decisions, and a rename merge"
        " starts another pass whose new securities may need further answers; that is known"
        " only once the step above completes."
    )
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:3]), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dated-types", type=Path)
    parser.add_argument("--coregistrant", type=Path)
    parser.add_argument("--coregistrant-requirements", type=Path)
    args = parser.parse_args()
    engine = open_research_database()
    with session_scope(create_session_factory(engine)) as s:
        if args.dated_types:
            dated_types(s, args.dated_types)
        if args.coregistrant:
            coregistrant(s, args.coregistrant)
        if args.coregistrant_requirements:
            coregistrant_requirements(s, args.coregistrant_requirements)
    return 0


if __name__ == "__main__":
    sys.exit(main())
