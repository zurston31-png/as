#!/usr/bin/env python3
"""Which SEC findings change when filing lists asked by CIK come from SEC's JSON service
instead of the cached browse-edgar answers? Cache only (no network), read-only.

    python scripts/filing_source_impact.py [--report PATH]

Recomputes sec_findings (the matched company, its delisting notices, the exit 8-Ks and the
Form 15s for every ended security) twice from the same cache: once reading the cached
browse-edgar lists, once building the lists asked by CIK from the cached JSON answers
(ticker lookups and name searches read browse-edgar in both). Done for the v5 validation
batch (data/sip_research.db) and for the stage database's securities; a security whose
answers are not all cached yet is reported as not comparable, never as unchanged.

The cache is the stage database (SIGNALIQ_SIP_RESEARCH_DATABASE_URL), which holds every
answer the v5 batch used plus the JSON answers.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import create_session_factory, session_scope
from app.universe import evidence, sec
from app.universe.database import open_research_database

V5_URL = (
    f"sqlite:///{(Path(__file__).resolve().parent.parent / 'data' / 'sip_research.db').as_posix()}"
)


class Incomplete(Exception):
    pass


class BrowseClient(sec.CachedSec):
    """Cache only; every list from the cached browse-edgar answers. A missing answer raises."""

    def get(self, url, params=None):
        text = super().get(url, params)
        if text is None:
            raise Incomplete(sec.cache_key(url, params))
        return text


class JsonClient(BrowseClient):
    """As BrowseClient, but a list asked by CIK is built from the cached JSON answers."""

    def company(self, cik_or_ticker, before, form=""):
        if cik_or_ticker.isdigit():
            return self.company_json(cik_or_ticker, before, form)
        return super().company(cik_or_ticker, before, form)


def summary(f: sec.SecFinding | None):
    if f is None:
        return None
    return (
        f.cik,
        f.company,
        tuple(sorted(sec.accession_of(x.href) for x, _ in f.notices)),
        tuple(sorted(sec.accession_of(x.href) for x in f.eight_ks)),
        tuple(sorted(sec.accession_of(x.href) for x in f.other)),
        f.cash,
    )


def per_security(security_session, cache_session, client_type):
    """{security id: finding summary, or "incomplete (...)"}: sec_findings' own steps, one
    security at a time, so a missing answer marks only that security."""
    from collections import defaultdict

    from sqlalchemy import select

    from app.universe import sec_history as sh
    from app.universe.models import Security, TickerHistory

    client = client_type(cache_session, network=False)
    history = defaultdict(list)
    for h in security_session.execute(
        select(TickerHistory).order_by(TickerHistory.valid_from)
    ).scalars():
        history[h.security_id].append(h)
    shared = evidence._shared_tickers(security_session)
    out = {}
    for security in security_session.execute(
        select(Security).where(Security.status == "ended").order_by(Security.security_id)
    ).scalars():
        try:
            tickers = history[security.security_id]
            symbols, names = evidence._sec_search_keys(security_session, tickers, shared)
            text_ciks = []
            for h in reversed(tickers):
                found_text = sh.ticker_candidates(client, h.symbol, h.valid_from, h.valid_to)
                for cik in sh.ranked_candidates(found_text):
                    if sh.confirm_writer(client, found_text[cik], h.symbol):
                        text_ciks.append(cik)
            out[security.security_id] = summary(
                sec.find(client, symbols, names, security.last_trade_session, text_ciks)
            )
        except Incomplete as exc:
            out[security.security_id] = f"incomplete ({exc})"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    lines: list[str] = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    cache_engine = open_research_database()
    cache = create_session_factory(cache_engine)
    v5_engine = open_research_database(V5_URL)
    v5 = create_session_factory(v5_engine)
    status = 0
    for label, factory in (("v5 validation batch", v5), ("Stage 1 (stage database)", cache)):
        with session_scope(factory) as securities, session_scope(cache) as answers:
            old = per_security(securities, answers, BrowseClient)
            new = per_security(securities, answers, JsonClient)
            names = dict(
                securities.execute(
                    __import__("sqlalchemy").text(
                        "select security_id, last_symbol from securities where status='ended'"
                    )
                ).all()
            )
        both = [
            sid for sid in old if not isinstance(old[sid], str) and not isinstance(new[sid], str)
        ]
        incomplete = [sid for sid in old if isinstance(old[sid], str) or isinstance(new[sid], str)]
        changed = [sid for sid in both if old[sid] != new[sid]]
        say(f"{label}: {len(old)} ended securities")
        say(f"  comparable (all answers cached for both): {len(both)}")
        say(f"  not comparable yet (an answer not cached): {len(incomplete)}")
        say(f"  findings unchanged: {len(both) - len(changed)}")
        say(f"  findings changed: {len(changed)}")
        for sid in changed:
            say(f"    {names.get(sid)} id {sid}:")
            say(f"      browse-edgar lists: {old[sid]}")
            say(f"      JSON lists:         {new[sid]}")
        status |= bool(changed)
        say()
    cache_engine.dispose()
    v5_engine.dispose()
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
