#!/usr/bin/env python3
"""Independent audit of the point-in-time issuers, from raw cached SEC answers. Read-only.

    python scripts/audit_issuers.py [--report PATH]

Does not use the pipeline's issuer code to judge its output. Checks:

1. **Every answer is cached.** The issuer step is replayed for every security against the
   cache alone, recording each request whose answer is missing; there must be none (so no
   lookup can have been read as "no issuer"), and no request goes to the network.
2. **Every piece of evidence stands up.** For each stored ``issuer_evidence`` row:
   its kind is the delisting notice, a cover page or the company's own filing (never
   SEC's ticker list as of today); a cover page or filing lies in that company's own EDGAR
   folder, is listed in the company's cached filing list with the same filing date, and
   was filed while the security traded under that ticker. The first and the last cover
   page of each run (ticker, company) is re-read and must list the ticker, followed by its
   exchange, under "Trading Symbol(s)".
3. **Every membership decision has an issuer**, and it rests on evidence filed by the
   decision date (or is counted as resting on a later document).
4. **Every SIC row belongs to an issuer of that security**, and every blank-check exclusion
   and every member is re-derived from the SIC rows on file by the decision date, for the
   issuer of that day: exclusions must be blank-check companies then; members must not be.
5. A fingerprint of the stored evidence, so two runs can be compared.

Exits with status 1 if any check fails.
"""

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from dataclasses import astuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.database import create_session_factory, session_scope
from app.universe import evidence, sec
from app.universe import sec_history as sh
from app.universe.database import open_research_database
from app.universe.models import (
    IssuerEvidence,
    IssuerSic,
    Membership,
    SecurityIssuer,
    TickerHistory,
)

KINDS = {"cover_page", "form25", "full_text"}
EXCHANGES = re.compile(
    r"(?:the\s+)?(?:nasdaq|new\s+york\s+stock\s+exchange|nyse|cboe|bats|chicago\s+stock|american\s+stock)",
    re.I,
)


def listed_on_cover(text: str, ticker: str) -> bool:
    """Written separately from ``sec_history.cover_page_symbol``: the ticker as a whole
    capitalised word within 600 characters after "Trading Symbol", with an exchange name
    starting within the next 20 characters."""
    for heading in re.finditer(r"trading\s+symbol", text, re.I):
        window = text[heading.end() : heading.end() + 600]
        start = 0
        while (i := window.find(ticker, start)) != -1:
            start = i + 1
            before = window[i - 1] if i else " "
            after = window[i + len(ticker)] if i + len(ticker) < len(window) else " "
            if before.isalnum() or before in ".&'-" or after.isalnum() or after in "&'-.":
                continue
            # The name must start within 20 characters; it may run past them.
            rest = window[i + len(ticker) : i + len(ticker) + 60].replace("\u200b", " ")
            m = EXCHANGES.search(rest)
            if (
                m
                and m.start() <= 20
                and not rest[: m.start()].replace("\xa0", " ").strip(" ,;:|()-")
            ):
                return True
    return False


#: Words that do not identify a company (written separately from sec_history._NAME_NOISE).
NOT_NAMES = {
    "AG",
    "AND",
    "CL",
    "CLASS",
    "CO",
    "COM",
    "COMMON",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "DEL",
    "ETF",
    "FUND",
    "GROUP",
    "HOLDING",
    "HOLDINGS",
    "INC",
    "INCORPORATED",
    "LIMITED",
    "LLC",
    "LP",
    "LTD",
    "NEW",
    "NV",
    "OF",
    "PLC",
    "SA",
    "SHARES",
    "SHS",
    "THE",
    "TRUST",
    "US",
    "USA",
}


def names_itself(text: str, ticker: str, company: str, reach: int = 60) -> bool:
    """Written separately from ``sec_history.own_ticker_mention``: "(NYSE: XOM)" or
    "(Nasdaq: XOM)" with every word of the company's own name in the ``reach``
    characters before it, and no "formerly" just before."""

    def words(t: str) -> set[str]:
        return set(re.findall(r"[A-Z0-9]+", t.replace("\\n", " ").upper()))

    name = words(re.sub(r"/[^/]*/", " ", company)) - NOT_NAMES
    if not name:
        return False
    exchange = r"(?:NASDAQ|NYSE(?:\s+AMERICAN|\s+ARCA|\s+MKT)?)"
    for m in re.finditer(rf"{exchange}\s*:\s*{re.escape(ticker)}(?![A-Za-z0-9])", text, re.I):
        before = text[max(0, m.start() - reach) : m.start()]
        if "formerly" not in before[-30:].lower() and name <= words(before):
            return True
    return False


def sic_on(rows, day):
    known = [sic for d, sic in sorted(rows) if d <= day and sic]
    return known[-1] if known else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    lines: list[str] = []
    failures = 0

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    engine = open_research_database()
    with session_scope(create_session_factory(engine)) as s:
        # 1. Cache completeness: the issuer step replayed against the cache alone.
        strict = evidence.StrictCache(s)
        replay, _ = evidence.issuer_points(s, strict, evidence.sec_findings(s, strict))
        stored_points = evidence.stored_issuer_points(s)
        same = {
            sid
            for sid in set(replay) | set(stored_points)
            if sorted(map(astuple, replay.get(sid, [])))
            == sorted(map(astuple, stored_points.get(sid, [])))
        }
        say("1. CACHE COMPLETENESS")
        say(f"  Issuer step replayed from the cache: requests to SEC {strict.requests}")
        say(f"  Answers needed but not cached: {len(strict.missing)}")
        for key in sorted(strict.missing)[:20]:
            say(f"    MISSING {key}")
        differ = sorted((set(replay) | set(stored_points)) - same)
        say(f"  Securities whose replayed evidence differs from the stored evidence: {len(differ)}")
        failures += bool(strict.missing) + bool(strict.requests) + bool(differ)

        # 2. Evidence.
        history = defaultdict(list)
        for h in s.execute(select(TickerHistory)).scalars():
            history[h.security_id].append(h)
        rows = list(s.execute(select(IssuerEvidence).order_by(IssuerEvidence.id)).scalars())
        client = sec.CachedSec(s, network=False)
        filings_of: dict[str, dict[str, object] | None] = {}
        problems = []
        for e in rows:
            if e.how not in KINDS:
                problems.append(f"id {e.security_id} {e.symbol}: evidence kind {e.how}")
                continue
            periods = [h for h in history[e.security_id] if h.symbol == e.symbol]
            if not periods:
                problems.append(f"id {e.security_id}: {e.symbol} is not one of its tickers")
                continue
            if e.how == "form25":
                continue
            if f"/edgar/data/{int(e.cik)}/" not in e.url:
                problems.append(
                    f"id {e.security_id} {e.symbol}: {e.url} not in CIK {e.cik}'s folder"
                )
            accession = e.url.split("/")[-2]
            if e.cik not in filings_of:
                company = sh.submissions(client, e.cik)
                filings_of[e.cik] = (
                    {acc.replace("-", ""): filed for _, filed, acc in company.filings}
                    if company
                    else None
                )
            listed = filings_of[e.cik]
            if listed is None:
                problems.append(f"CIK {e.cik}: filing list not cached")
            elif listed.get(accession) != e.filed:
                problems.append(
                    f"id {e.security_id} {e.symbol}: {accession} filed {listed.get(accession)} "
                    f"in CIK {e.cik}'s list, evidence says {e.filed}"
                )
            slack = 30 if e.how == "full_text" else 0
            if not any(
                (h.valid_from - e.filed).days <= slack and (e.filed - h.valid_to).days <= slack
                for h in periods
            ):
                problems.append(
                    f"id {e.security_id} {e.symbol}: {e.filed} outside its ticker period"
                )
        own_filings = 0
        for e in rows:
            if e.how != "full_text":
                continue
            raw = client.get(e.url)
            if raw is None:
                problems.append(f"id {e.security_id} {e.symbol}: own filing {e.url} not cached")
            elif names_itself(sec.plain_text(raw), e.symbol, e.name):
                own_filings += 1
            else:
                problems.append(
                    f"id {e.security_id} {e.symbol}: {e.url} does not show {e.name} naming "
                    f"{e.symbol} as its own"
                )
        runs = defaultdict(list)
        for e in rows:
            if e.how == "cover_page":
                runs[(e.security_id, e.symbol, e.cik)].append(e)
        confirmed_ends = 0
        for (sid, symbol, _cik), run in runs.items():
            run.sort(key=lambda e: e.filed)
            for e in {run[0].id: run[0], run[-1].id: run[-1]}.values():
                text = client.cover_text(e.url)
                if text is None:
                    problems.append(f"id {sid} {symbol}: cover page {e.url} not cached")
                elif listed_on_cover(text, symbol):
                    confirmed_ends += 1
                else:
                    end = "first" if e is run[0] else "last"
                    problems.append(f"id {sid} {symbol}: {end} cover page {e.url} does not list it")
        say("\n2. EVIDENCE")
        kinds = dict(sorted(_count(e.how for e in rows).items()))
        say(f"  Evidence rows: {len(rows)}; kinds {kinds}")
        say(
            f"  Cover-page runs (security, ticker, company): {len(runs)}; "
            f"ends re-read and listing the ticker: {confirmed_ends}"
        )
        say(
            f"  Own-filing evidence re-read, the company naming the ticker as its own: "
            f"{own_filings} of {sum(e.how == 'full_text' for e in rows)}"
        )
        issuers = list(s.execute(select(SecurityIssuer)).scalars())
        bad_how = [i for i in issuers if i.how not in KINDS]
        say(
            f"  Issuers at the last trade: {len(issuers)}; resting on a kind other than "
            f"cover page, Form 25 or own filing: {len(bad_how)}"
        )
        say(f"  Problems: {len(problems)}")
        for p in problems:
            say(f"    {p}")
        failures += bool(problems) + bool(bad_how)

        # 3 and 4. Membership decisions.
        sics = defaultdict(list)
        for r in s.execute(select(IssuerSic)).scalars():
            sics[(r.security_id, r.cik)].append((r.as_of, r.sic))
        stray = [k for k in sics if k[1] not in {p.cik for p in stored_points.get(k[0], [])}]
        renames = {sid: sorted(h.valid_from for h in items)[1:] for sid, items in history.items()}
        no_issuer = later = unsupported = member_blank = 0
        examples = []
        decisions = list(s.execute(select(Membership)).scalars())
        for m in decisions:
            pts = stored_points.get(m.security_id, [])
            if not (m.member or m.reason == "blank_check_company"):
                continue
            by_then = [p for p in pts if p.filed <= m.decision_session]
            held = (
                max(by_then, key=lambda p: (p.filed, p.how == "form25"))
                if by_then
                else (min(pts, key=lambda p: p.filed) if pts else None)
            )
            if held is None:
                no_issuer += m.member
                if m.reason == "blank_check_company":
                    unsupported += 1
                continue
            later += not by_then
            rows_then = [
                x for x in sics.get((m.security_id, held.cik), []) if x[0] <= m.decision_session
            ]
            first_6770 = min((d for d, sic in rows_then if sic == sh.BLANK_CHECK_SIC), default=None)
            renamed = first_6770 and any(
                first_6770 <= r <= m.decision_session for r in renames.get(m.security_id, [])
            )
            blank = (
                bool(first_6770) and not renamed and sic_on(rows_then, m.decision_session) == "6770"
            )
            if m.reason == "blank_check_company" and not blank:
                unsupported += 1
                examples.append(
                    f"excluded but not blank-check: id {m.security_id} {m.decision_session}"
                )
            if m.member and blank:
                member_blank += 1
                examples.append(
                    f"member while blank-check: id {m.security_id} {m.decision_session}"
                )
        say("\n3. MEMBERSHIP DECISIONS")
        members = sum(1 for m in decisions if m.member)
        say(f"  Member decisions: {members}; without an issuer: {no_issuer}")
        say(f"  Member or blank-check decisions whose issuer rests on a later document: {later}")
        say("\n4. SIC AND BLANK-CHECK COMPANIES")
        say(f"  SIC rows for a company that is not an issuer of that security: {len(stray)}")
        say(
            f"  Blank-check exclusions not re-derived from the SIC on file by then: {unsupported}; "
            f"members that were blank-check companies then: {member_blank}"
        )
        for x in examples[:20]:
            say(f"    {x}")
        failures += bool(no_issuer) + bool(stray) + bool(unsupported) + bool(member_blank)

        # 5. Fingerprint.
        digest = hashlib.sha256()
        for e in sorted(
            rows, key=lambda e: (e.security_id, e.symbol, e.filed, e.cik, e.how, e.url)
        ):
            digest.update(repr((e.security_id, e.symbol, e.filed, e.cik, e.how, e.url)).encode())
        say("\n5. FINGERPRINT")
        say(f"  issuer_evidence: {len(rows)} rows, sha256 {digest.hexdigest()[:16]}")
        say(f"\nRESULT: {'FAIL' if failures else 'PASS'}")
    engine.dispose()
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return 1 if failures else 0


def _count(items):
    out: dict = defaultdict(int)
    for x in items:
        out[x] += 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
