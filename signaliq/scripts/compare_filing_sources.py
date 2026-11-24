#!/usr/bin/env python3
"""Equivalence check: browse-edgar filing lists against SEC's JSON service. Read-only for
existing cache rows (it only adds the JSON answers it fetches).

    python scripts/compare_filing_sources.py [--report PATH]

For every company filing list in the research database's cache that came from
www.sec.gov/cgi-bin/browse-edgar (action=getcompany, by CIK or ticker), builds the same
list from data.sec.gov/submissions (CachedSec.company_json) and compares:

- whether a company was found at all, and its CIK;
- every filing: accession number, filing date, form type, and the 8-K item numbers;
- the order of the list (newest first) and its length (browse-edgar returns at most 100);
- the company name and SIC code (today's values in both services).

Requests the cache keyed by a digest (long URLs) are recovered by recomputing the key for
every candidate date and form type. Every discrepancy is printed with its details. Then the
difficult cases (bankruptcies, failures, acquisitions, other delistings, reused tickers,
First Republic, Signature Bank) are listed from the v5 validation batch's outcomes.

Exits with status 1 if any list cannot be recovered or fetched, or any discrepancy remains
that is not one of the explained kinds printed in the summary.
"""

import argparse
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.database import create_session_factory, session_scope
from app.universe import sec
from app.universe.database import open_research_database
from app.universe.models import ExternalResponse

V5_DB = Path(__file__).resolve().parent.parent / "data" / "sip_research.db"
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
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
        say(f"browse-edgar company filing lists in the cache: {len(cached)}")
        client = sec.CachedSec(s, network=True)
        results = {}  # key -> (who, before, form, old, new)
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
                new = client.company_json(who, before, form)
            except sec.SecUnavailable as exc:
                unfetched.append((key, str(exc)))
                continue
            results[key] = (who, before, form, old, new)
            if i % 100 == 0:
                s.commit()
                print(f"  ... {i} of {len(cached)} compared", flush=True)
        s.commit()
        client.close()

        say(
            "retired key format (Python hash(), unreadable by any run, not compared): "
            f"{len(retired)}"
            f" for {len({re.search(r'CIK=([^&]+)', k).group(1) for k in retired})} companies"
        )
        recovered = len(cached) - len(retired) - len(unrecovered)
        say(f"parameters recovered: {recovered} (not recovered: {len(unrecovered)})")
        say(f"JSON lists built: {len(results)} (not fetched: {len(unfetched)})")
        for key in unrecovered:
            say(f"  NOT RECOVERED {key}")
        for key, why in unfetched:
            say(f"  NOT FETCHED {key}: {why}")

        kinds = Counter()
        details = defaultdict(list)
        for _key, (who, before, form, old, new) in sorted(results.items()):
            label = f"{who} type={form or '-'} dateb={before}"
            a, b = rows(old), rows(new)
            if old is None and new is None:
                kinds["identical: no company in either"] += 1
                continue
            if (old is None) != (new is None):
                kind = "company found only in " + ("JSON" if old is None else "browse-edgar")
                kinds[kind] += 1
                details[kind].append(f"{label}: browse={old and old.cik} json={new and new.cik}")
                continue
            if old.cik != new.cik:
                kinds["different CIK"] += 1
                details["different CIK"].append(f"{label}: {old.cik} vs {new.cik}")
                continue
            notes = []
            if a == b:
                kinds["identical filings (accession, date, form, items, order)"] += 1
            elif sorted(a) == sorted(b):
                kinds["same filings, different order"] += 1
                notes.append("order")
                details["same filings, different order"].append(
                    f"{label}: first difference at position "
                    f"{next(i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)}"
                )
            else:
                only_old = sorted(set(a) - set(b))
                only_new = sorted(set(b) - set(a))
                kind = "different filings"
                kinds[kind] += 1
                details[kind].append(
                    f"{label}: {len(a)} vs {len(b)} filings; only browse-edgar {only_old[:5]}; "
                    f"only JSON {only_new[:5]}"
                )
            if norm(old.name) != norm(new.name):
                kinds["name differs (today's name)"] += 1
                details["name differs (today's name)"].append(
                    f"{label}: {old.name!r} vs {new.name!r}"
                )
            if (old.sic or "") != (new.sic or ""):
                kinds["SIC differs (today's SIC)"] += 1
                details["SIC differs (today's SIC)"].append(f"{label}: {old.sic} vs {new.sic}")

        say("\nRESULTS")
        for kind, n in sorted(kinds.items(), key=lambda x: -x[1]):
            say(f"  {n:5d}  {kind}")
        for kind, lines in details.items():
            say(f"\n{kind} ({len(lines)}):")
            for line in lines:
                say(f"  {line}")

        say("\nDIFFICULT CASES (v5 validation batch outcomes)")
        difficult_cases(results, say)

    engine.dispose()
    if args.report:
        args.report.write_text("\n".join(out) + "\n")
    material = {
        "company found only in JSON",
        "company found only in browse-edgar",
        "different CIK",
        "different filings",
    }
    bad = unrecovered or unfetched or any(kinds[k] for k in material)
    return 1 if bad else 0


def difficult_cases(results, say) -> None:
    """Per case, every compared list asked for the security's tickers or its issuer's CIK."""
    v5 = sqlite3.connect(f"file:{V5_DB}?mode=ro", uri=True)
    tickers = defaultdict(set)
    for sid, symbol in v5.execute("select security_id, symbol from ticker_history"):
        tickers[sid].add(symbol)
    cik = dict(v5.execute("select security_id, cik from security_issuers"))
    cases = v5.execute(
        "select security_id, last_symbol, outcome, outcome_status from securities "
        "where status='ended' order by outcome, last_symbol"
    ).fetchall()
    reused = {
        s for (s,) in v5.execute("select symbol from segment_links where decision='kept_separate'")
    }
    by_who = defaultdict(list)
    for _key, (who, before, form, old, new) in results.items():
        by_who[who.lstrip("0") if who.isdigit() else who].append((before, form, old, new))
    groups = defaultdict(list)
    for sid, symbol, outcome, _status in cases:
        group = (
            "First Republic / Signature Bank"
            if symbol in ("FRC", "SBNY")
            else "bankruptcy or failure"
            if outcome in ("bankruptcy", "failure")
            else "acquired"
            if (outcome or "").startswith("acquired")
            else "reused ticker"
            if tickers[sid] & reused
            else f"other delisting ({outcome or 'unresolved'})"
        )
        groups[group].append((sid, symbol))
    for sid in {sid for sid, t in tickers.items() if t & reused}:
        symbol = next(iter(tickers[sid]))
        if all(sid != x for x, _ in groups["reused ticker"]):
            groups["reused ticker"].append((sid, symbol))
    for group, members in sorted(groups.items()):
        same = total = 0
        lines = []
        for sid, symbol in members:
            keys = set(tickers[sid]) | ({cik[sid].lstrip("0")} if sid in cik else set())
            lists = [x for k in keys for x in by_who.get(k, [])]
            ok = sum(
                rows(old) == rows(new) and (old is None) == (new is None)
                for _, _, old, new in lists
            )
            same += ok
            total += len(lists)
            lines.append(f"    {symbol:6} id {sid}: {ok} of {len(lists)} lists identical")
        say(f"  {group}: {len(members)} securities, {same} of {total} lists identical")
        for line in lines:
            say(line)


if __name__ == "__main__":
    raise SystemExit(main())
