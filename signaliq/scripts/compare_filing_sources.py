#!/usr/bin/env python3
"""Equivalence check: browse-edgar filing lists against SEC's JSON service. Read-only for
existing cache rows (it only adds the JSON answers it fetches).

    python scripts/compare_filing_sources.py [--live] [--report PATH]

Company filing lists asked BY CIK (the Form 25, 8-K and Form 15 lists sec.find reads) are
the lists the pipeline takes from SEC's JSON service. For every such list in the research
database's cache that came from www.sec.gov/cgi-bin/browse-edgar, builds the same list from
data.sec.gov/submissions (CachedSec.company_json) and compares:

- whether a company was found at all, and its CIK;
- every filing: accession number, filing date, form type, and the 8-K item numbers;
- the order of the list (newest first; filings of one day by acceptance time) and its
  length (at most 100);
- the company name and SIC code (today's values in both services).

With --live, every list that differs is asked of browse-edgar again (retried; an unanswered
request is a failure, never a match). A difference is explained only when today's
browse-edgar answer equals the JSON list exactly, i.e. the cached browse-edgar answer was
wrong when fetched.

Lists asked BY TICKER stay on browse-edgar (SEC's JSON ticker list knows only today's
tickers, browse-edgar also resolves retired ones); sec.find uses only their CIK, so they are
reported by the CIK each service gives, for information.

Requests cached under a digest key (long URLs) are recovered by recomputing the key for
every candidate date and form type. Then the difficult cases (bankruptcies, failures,
acquisitions, other delistings, reused tickers, First Republic, Signature Bank) are listed
from the v5 validation batch's outcomes.

Exits with status 1 if any CIK list cannot be recovered or built, or any difference in a
CIK list is not explained by a live answer.
"""

import argparse
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from sqlalchemy import select

from app.database import create_session_factory, session_scope
from app.universe import sec
from app.universe.database import V2_DATABASE, open_research_database
from app.universe.models import ExternalResponse

V5_DB = V2_DATABASE
FORMS = ("", "25", "8-K", "15")
PREFIX = sec.BROWSE_URL + "?CIK="
RETIRED_KEY = re.compile(r"#-?\d{1,20}$")


def recover_params(key: str) -> tuple[str, date, str] | None:
    """(CIK or ticker, dateb, type) of a cached browse-edgar company request."""
    if "#" not in key:
        query = dict(parse_qsl(urlsplit(key).query, keep_blank_values=True))
        day = query["dateb"]
        return query["CIK"], date(int(day[:4]), int(day[4:6]), int(day[6:])), query.get("type", "")
    head = key.split("#")[0]
    found = re.match(re.escape(PREFIX) + r"([^&]+)&action=getcompany&count=100&dateb=(\d*)", head)
    if not found:
        return None
    who, partial = found.groups()
    day = date(2010, 1, 1)
    while day <= date(2026, 12, 31):
        text = day.strftime("%Y%m%d")
        if text.startswith(partial):
            for form in FORMS:
                params = {
                    "action": "getcompany",
                    "CIK": who,
                    "type": form,
                    "owner": "include",
                    "dateb": text,
                    "count": "100",
                    "output": "atom",
                }
                if sec.cache_key(sec.BROWSE_URL, params) == key:
                    return who, day, form
        day += timedelta(days=1)
    return None


def items_of(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"\d\.\d\d", text or ""))))


def rows(company: sec.Company | None) -> list[tuple[str, date, str, tuple]]:
    if company is None:
        return []
    return [(sec.accession_of(f.href), f.filed, f.form, items_of(f.items)) for f in company.filings]


def norm(text: str | None) -> str:
    return " ".join((text or "").upper().replace("&AMP;", "&").split())


def browse_live(who: str, form: str, before: date) -> str | None:
    """browse-edgar's answer today (retried through its passing 503s); None if none came."""
    params = {
        "action": "getcompany",
        "CIK": who,
        "type": form,
        "owner": "include",
        "dateb": before.strftime("%Y%m%d"),
        "count": "100",
        "output": "atom",
    }
    for _ in range(20):
        time.sleep(sec.PAUSE_SECONDS)
        try:
            r = httpx.get(
                sec.BROWSE_URL, params=params, headers={"User-Agent": sec.USER_AGENT}, timeout=60
            )
        except httpx.TransportError:
            r = None
        if r is not None and r.status_code == 200:
            return r.text
        time.sleep(5)
    return None


def same(old: sec.Company | None, new: sec.Company | None) -> bool:
    if old is None or new is None:
        return old is None and new is None
    return old.cik == new.cik and rows(old) == rows(new)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--live", action="store_true", help="re-ask browse-edgar for differences")
    args = parser.parse_args()
    out: list[str] = []

    def say(line: str = "") -> None:
        print(line, flush=True)
        out.append(line)

    engine = open_research_database()
    factory = create_session_factory(engine)
    with session_scope(factory) as s:
        cached = list(
            s.execute(
                select(ExternalResponse.request_key, ExternalResponse.response).where(
                    ExternalResponse.source == "sec",
                    ExternalResponse.request_key.like(PREFIX + "%"),
                )
            )
        )
        client = sec.CachedSec(s, network=True)
        by_cik, by_ticker = {}, {}
        unrecovered, unfetched, retired = [], [], []
        for i, (key, response) in enumerate(sorted(cached)):
            if RETIRED_KEY.search(key):
                # Keyed with Python's per-process hash() before the sha256 key was adopted:
                # no run can compute that key again, so the pipeline never reads these rows.
                retired.append(key)
                continue
            params = recover_params(key)
            if params is None:
                unrecovered.append(key)
                continue
            who, before, form = params
            old = sec.parse_company_feed(response) if response else None
            try:
                if who.isdigit():
                    by_cik[key] = (who, before, form, old, client.company_json(who, before, form))
                else:
                    by_ticker[key] = (who, before, form, old, client.ticker_cik(who))
            except sec.SecUnavailable as exc:
                unfetched.append((key, str(exc)))
            if i % 100 == 0:
                s.commit()
        s.commit()
        client.close()

        say(f"browse-edgar company filing lists in the cache: {len(cached)}")
        say(
            "  retired key format (Python hash(), unreadable by any run, not compared): "
            f"{len(retired)} for {len({re.search(r'CIK=([^&]+)', k).group(1) for k in retired})} "
            "companies, each of which also has lists under the current key"
        )
        say(f"  parameters not recovered: {len(unrecovered)}")
        say(f"  JSON list not built (no SEC answer): {len(unfetched)}")
        for key in unrecovered:
            say(f"    NOT RECOVERED {key}")
        for key, why in unfetched:
            say(f"    NOT BUILT {key}: {why}")
        say(f"  asked by CIK (switched to the JSON service): {len(by_cik)}")
        say(f"  asked by ticker (stay on browse-edgar): {len(by_ticker)}")

        kinds = Counter()
        details = defaultdict(list)
        unexplained = 0
        for _key, (who, before, form, old, new) in sorted(by_cik.items()):
            label = f"CIK {who} type={form or '-'} dateb={before}"
            if same(old, new):
                kinds["identical" if old else "identical: no company in either"] += 1
            else:
                a, b = rows(old), rows(new)
                desc = (
                    f"{label}: cached browse-edgar {old and old.cik} {len(a)} filings, "
                    f"JSON {new and new.cik} {len(b)} filings; only browse-edgar "
                    f"{[(x[0], str(x[1]), x[2]) for x in sorted(set(a) - set(b))[:4]]}; "
                    f"only JSON {[(x[0], str(x[1]), x[2]) for x in sorted(set(b) - set(a))[:4]]}"
                    + (" (same filings, other order)" if a != b and sorted(a) == sorted(b) else "")
                )
                verdict = "UNEXPLAINED (not re-asked; run with --live)"
                if args.live:
                    text = browse_live(who, form, before)
                    live = sec.parse_company_feed(text) if text else None
                    if text is None:
                        verdict = "UNEXPLAINED: browse-edgar gave no answer today"
                    elif same(live, new):
                        verdict = "explained: browse-edgar today answers exactly the JSON list"
                    elif same(live, old):
                        verdict = "UNEXPLAINED: browse-edgar today repeats the cached answer"
                    else:
                        verdict = "UNEXPLAINED: browse-edgar today differs from both"
                unexplained += verdict.startswith("UNEXPLAINED")
                kinds["differs: " + verdict] += 1
                details["differs"].append(f"{desc} -> {verdict}")
            if old and new and norm(old.name) != norm(new.name):
                kinds["name differs (today's name in both)"] += 1
                details["name differs"].append(f"{label}: {old.name!r} vs {new.name!r}")
            if old and new and (old.sic or "") != (new.sic or ""):
                kinds["SIC differs (today's SIC in both)"] += 1
                details["SIC differs"].append(f"{label}: {old.sic} vs {new.sic}")

        say("\nLISTS ASKED BY CIK (switched to SEC's JSON service)")
        for kind, n in sorted(kinds.items(), key=lambda x: -x[1]):
            say(f"  {n:5d}  {kind}")
        for kind, lines in details.items():
            say(f"\n  {kind} ({len(lines)}):")
            for line in lines:
                say(f"    {line}")

        tick = Counter()
        tick_lines = []
        for _key, (who, before, _form, old, new_cik) in sorted(by_ticker.items()):
            old_cik = old.cik if old else None
            kind = (
                "same CIK"
                if old_cik == new_cik
                else "CIK only from browse-edgar (a ticker not in SEC's list today)"
                if new_cik is None
                else "CIK only from SEC's ticker list today"
                if old_cik is None
                else "different CIK (the ticker has a new holder today)"
            )
            tick[kind] += 1
            if kind != "same CIK":
                tick_lines.append(f"{who} dateb={before}: browse-edgar {old_cik}, JSON {new_cik}")
        say("\nLISTS ASKED BY TICKER (stay on browse-edgar; sec.find uses only the CIK)")
        for kind, n in sorted(tick.items(), key=lambda x: -x[1]):
            say(f"  {n:5d}  {kind}")
        for line in tick_lines:
            say(f"    {line}")

        say("\nDIFFICULT CASES (v5 validation batch outcomes; lists asked by CIK)")
        difficult_cases(by_cik, by_ticker, say)
        say(
            f"\nRESULT: {'PASS' if not (unrecovered or unfetched or unexplained) else 'FAIL'}"
            f" ({unexplained} unexplained differences, {len(unrecovered)} not recovered, "
            f"{len(unfetched)} not built)"
        )

    engine.dispose()
    if args.report:
        args.report.write_text("\n".join(out) + "\n")
    return 1 if unrecovered or unfetched or unexplained else 0


def difficult_cases(by_cik, by_ticker, say) -> None:
    """Per case: every list asked for its issuer's CIK, or by any of its tickers."""
    v5 = sqlite3.connect(f"file:{V5_DB}?mode=ro", uri=True)
    tickers = defaultdict(set)
    for sid, symbol in v5.execute("select security_id, symbol from ticker_history"):
        tickers[sid].add(symbol)
    issuer = dict(v5.execute("select security_id, cik from security_issuers"))
    reused = {
        s for (s,) in v5.execute("select symbol from segment_links where decision='kept_separate'")
    }
    lists = defaultdict(list)
    for who, before, form, old, new in by_cik.values():
        lists[who.zfill(10)].append((form, before, same(old, new)))
    ticker_cik = defaultdict(set)
    for who, _before, _form, old, new_cik in by_ticker.values():
        ticker_cik[who].add((old.cik if old else None, new_cik))
    groups = defaultdict(list)
    for sid, symbol, outcome in v5.execute(
        "select security_id, last_symbol, outcome from securities where status='ended'"
    ):
        group = (
            "First Republic / Signature Bank"
            if symbol in ("FRC", "SBNY")
            else "bankruptcy or failure"
            if outcome in ("bankruptcy", "failure")
            else "acquired"
            if (outcome or "").startswith("acquired")
            else f"other delisting ({outcome or 'unresolved'})"
        )
        groups[group].append((sid, symbol))
    for sid in sorted(sid for sid, t in tickers.items() if t & reused):
        groups["reused ticker (kept-separate securities)"].append(
            (sid, sorted(tickers[sid] & reused)[0])
        )
    for group, members in sorted(groups.items()):
        same_n = total = 0
        lines = []
        for sid, symbol in sorted(members, key=lambda m: m[1]):
            cik = issuer.get(sid)
            mine = lists.get(cik, []) if cik else []
            ok = sum(x[2] for x in mine)
            same_n += ok
            total += len(mine)
            ciks = sorted(
                {c for t in tickers[sid] for pair in ticker_cik.get(t, ()) for c in pair if c}
            )
            lines.append(
                f"    {symbol:6} id {sid:3}: issuer CIK {cik or 'none (no SEC issuer)'}; "
                f"{ok} of {len(mine)} CIK lists identical"
                + (f"; ticker lookups gave {ciks}" if not cik else "")
            )
        say(f"  {group}: {len(members)} securities, {same_n} of {total} CIK lists identical")
        for line in lines:
            say(line)


if __name__ == "__main__":
    raise SystemExit(main())
