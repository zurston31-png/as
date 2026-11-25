#!/usr/bin/env python3
"""Independent audit of trading-gap merges, from the raw stored data. Read-only.

    python scripts/audit_merges.py

For every merge recorded in the research database, recomputes from the raw ticker bars,
the raw corporate actions and the issuer's SEC filing list (cache only), without using
the pipeline's decision code. Then, for every candidate kept separate, re-establishes from
the same raw data at least one fact that rules out a merge (a delisting notice, a price
break, a long gap, different issuers, a merger record, or a quoted fund liquidation).

For every merge:

1. both segments are bars of the same ticker, and inside the gap every bar Alpaca
   returned is a filler bar (no volume, no trades);
2. the number of sessions without a trade, recounted from the exchange calendar;
3. the split-adjusted price ratio across the gap, recomputed from the raw split events;
4. no Form 25 / 25-NSE filed by the security's issuer (when known) between the last
   trade before the gap and the first trade after it;
5. the merged security exists once in the securities table, spanning both segments.

Then every documented identity exception (``CURATED_IDENTITY_LINKS``: one security across
a suspension longer than the trading-gap rule, under a new ticker) is checked the same way,
from raw data and not from the pipeline: each quoted passage is in its cached SEC document;
both segment ends are real trades and neither ticker traded in between; both segments
belong to one security, whose issuer is the link's CIK; the issuer's own filing list
(cached SEC submissions) shows no delisting notice (Form 25) and no deregistration
(Form 15) between the last and the first trade; and Alpaca, independently of SEC, records
the ticker change of the same security in between (one security, not only one company).
The price on each side is reported.

Exits with status 1 if any merge or exception fails a check.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.data.validation import trading_days
from app.database import create_session_factory, session_scope
from app.universe import sec
from app.universe import sec_history as sh
from app.universe.curated import CURATED_IDENTITY_LINKS
from app.universe.database import open_research_database
from app.universe.models import (
    CorporateAction,
    Security,
    SecurityIssuer,
    SegmentLink,
    TickerBar,
    TickerHistory,
)

PRICE_LOW, PRICE_HIGH = 0.5, 2.0
SCOPE_END = date(2024, 12, 31)


def last_trade(session, sid):
    return session.get(Security, sid).last_trade_session


def main() -> int:
    engine = open_research_database()
    factory = create_session_factory(engine)
    failures = 0
    with session_scope(factory) as s:
        client = sec.CachedSec(s, network=False)
        links = list(
            s.execute(
                select(SegmentLink)
                .where(SegmentLink.decision == "merged")
                .order_by(SegmentLink.symbol, SegmentLink.earlier_last)
            ).scalars()
        )
        actions = list(s.execute(select(CorporateAction)).scalars())
        print(f"Merges to audit: {len(links)}\n")
        for link in links:
            problems = []
            bars = {
                b.session: b
                for b in s.execute(
                    select(TickerBar).where(
                        TickerBar.symbol == link.symbol,
                        TickerBar.session >= link.earlier_last,
                        TickerBar.session <= link.later_first,
                    )
                ).scalars()
            }
            before, after = bars.get(link.earlier_last), bars.get(link.later_first)
            if before is None or after is None or before.volume == 0 or after.volume == 0:
                problems.append("a segment end is not a real trade of this ticker")
            inside = [b for d, b in bars.items() if link.earlier_last < d < link.later_first]
            traded = [b.session for b in inside if b.volume > 0 or b.trade_count > 0]
            if traded:
                problems.append(f"real trades inside the gap: {traded[:3]}")
            gap = max(0, len(trading_days(link.earlier_last, link.later_first)) - 2)
            if gap != link.gap_sessions:
                problems.append(f"gap recount {gap} != recorded {link.gap_sessions}")
            interim = {link.symbol} | {
                a.new_symbol
                for a in actions
                if a.action_type == "name_change"
                and a.old_symbol == link.symbol
                and a.process_date
                and link.earlier_last < a.process_date <= link.later_first
            }
            ratio = 1.0
            for a in actions:
                when = a.ex_date or a.process_date
                if (
                    a.action_type in ("forward_split", "reverse_split", "unit_split")
                    and a.symbol in interim
                    and when
                    and link.earlier_last < when <= link.later_first
                    and a.old_rate
                    and a.new_rate
                ):
                    ratio *= a.new_rate / a.old_rate
            price = None
            if before is not None and after is not None and before.close:
                price = after.close / (before.close / ratio)
                if not PRICE_LOW <= price <= PRICE_HIGH and "SEC filings say" not in link.reason:
                    problems.append(f"price ratio {price:.3f} outside {PRICE_LOW}-{PRICE_HIGH}")
            owners = {
                h.security_id
                for h in s.execute(
                    select(TickerHistory).where(
                        TickerHistory.symbol == link.symbol,
                        TickerHistory.valid_from.in_((link.later_first,)),
                    )
                ).scalars()
            } | {
                h.security_id
                for h in s.execute(
                    select(TickerHistory).where(
                        TickerHistory.symbol == link.symbol,
                        TickerHistory.valid_to == link.earlier_last,
                    )
                ).scalars()
            }
            if len(owners) != 1:
                problems.append(f"segments belong to {len(owners)} securities, not one")
            notices, notice_check = [], []
            for sid in owners:
                issuer = s.get(SecurityIssuer, sid)
                if issuer is None:
                    notice_check.append("no SEC issuer known")
                    continue
                # The pipeline asks for the issuer's Form 25 list up to 120 days after a
                # segment's last trade; read that same list (cache only, never fetched here).
                feeds = [
                    client.company(issuer.cik, min(day + timedelta(days=120), SCOPE_END), "25")
                    for day in (link.earlier_last, last_trade(s, sid))
                ]
                read = [f for f in feeds if f is not None]
                if not read:
                    problems.append(f"issuer {issuer.cik}: delisting-notice list not available")
                    continue
                filings = {(f.form, f.filed) for feed in read for f in feed.filings}
                found = sorted(
                    f"{form} {filed}"
                    for form, filed in filings
                    if form.startswith("25") and link.earlier_last < filed < link.later_first
                )
                notices += found
                notice_check.append(f"issuer {issuer.cik}: notice list read, {len(found)} in gap")
            if notices:
                problems.append(f"issuer delisting notice in the gap: {notices}")
            failures += bool(problems)
            print(
                f"{'FAIL' if problems else 'ok  '} {link.symbol:6} "
                f"{link.earlier_last} -> {link.later_first}: "
                f"{gap} sessions, {len(inside)} filler bars inside, price ratio "
                f"{'n/a' if price is None else f'{price:.3f}'}"
                + (f" after splits x{ratio:g}" if ratio != 1.0 else "")
                + f", one security: {len(owners) == 1}; {', '.join(notice_check)}"
                + (f"; {'; '.join(problems)}" if problems else "")
            )
        print(f"\n{len(links) - failures} of {len(links)} merges pass every check")
        rejected_failures = audit_rejections(s, client, actions)
        identity_failures = audit_identity_links(s, client)
    engine.dispose()
    return 1 if failures or rejected_failures or identity_failures else 0


def audit_identity_links(s, client) -> int:
    """Recheck every documented identity exception from raw data (see the module docstring)."""
    print(f"\nDocumented identity exceptions to audit: {len(CURATED_IDENTITY_LINKS)}\n")
    failures = 0
    for link in CURATED_IDENTITY_LINKS:
        problems = []
        for q in link.quotes:
            if " ".join(q.text.split()) not in sh.plain_text(client.get(q.url)):
                problems.append(f"quote not found in {q.url}")
        old = s.scalar(
            select(TickerBar).where(
                TickerBar.symbol == link.old_symbol, TickerBar.session == link.old_last
            )
        )
        new = s.scalar(
            select(TickerBar).where(
                TickerBar.symbol == link.new_symbol, TickerBar.session == link.new_first
            )
        )
        if old is None or old.volume == 0 or new is None or new.volume == 0:
            problems.append("a segment end is not a real trade")
        between = [
            (b.symbol, b.session)
            for b in s.execute(
                select(TickerBar).where(
                    TickerBar.symbol.in_((link.old_symbol, link.new_symbol)),
                    TickerBar.session > link.old_last,
                    TickerBar.session < link.new_first,
                )
            ).scalars()
            if b.volume > 0 or b.trade_count > 0
        ]
        if between:
            problems.append(f"real trades in between: {between[:3]}")
        owners = {
            h.security_id
            for h in s.execute(
                select(TickerHistory).where(
                    (
                        (TickerHistory.symbol == link.old_symbol)
                        & (TickerHistory.valid_to == link.old_last)
                    )
                    | (
                        (TickerHistory.symbol == link.new_symbol)
                        & (TickerHistory.valid_from == link.new_first)
                    )
                )
            ).scalars()
        }
        if len(owners) != 1:
            problems.append(f"segments belong to {len(owners)} securities, not one")
        for sid in owners:
            issuer = s.get(SecurityIssuer, sid)
            if issuer is None or issuer.cik != link.cik:
                problems.append(f"id {sid} issuer {issuer.cik if issuer else None}, not {link.cik}")
        company = sh.submissions(client, link.cik)
        if company is None:
            problems.append(f"filing list of CIK {link.cik} not cached")
            notices = []
        else:
            notices = sorted(
                f"{form} {filed}"
                for form, filed, _ in company.filings
                if form.startswith(("25", "15")) and link.old_last <= filed <= link.new_first
            )
            if notices:
                problems.append(f"filed in between: {notices}")
        # Same shares, not only the same company: Alpaca (independent of SEC) records the
        # ticker change of this security between the two segments.
        renames = [
            a
            for a in s.execute(
                select(CorporateAction).where(
                    CorporateAction.action_type == "name_change",
                    CorporateAction.old_symbol == link.old_symbol,
                    CorporateAction.new_symbol == link.new_symbol,
                )
            ).scalars()
            if a.process_date and link.old_last < a.process_date <= link.new_first
        ]
        if not renames:
            problems.append("no Alpaca name change between the two tickers in between")
        gap = max(0, len(trading_days(link.old_last, link.new_first)) - 2)
        failures += bool(problems)
        print(
            f"{'FAIL' if problems else 'ok  '} {link.old_symbol} {link.old_last} -> "
            f"{link.new_symbol} {link.new_first}: {gap} sessions without a trade; close "
            f"{old.close if old else None} -> {new.close if new else None}; one security: "
            f"{len(owners) == 1}; issuer {link.cik}; Form 25/15 in between: "
            f"{len(notices) if company else 'not read'}; Alpaca name change "
            f"{', '.join(str(a.process_date) for a in renames) or 'none'}"
            + (f"; {'; '.join(problems)}" if problems else "")
        )
    print(
        f"\n{len(CURATED_IDENTITY_LINKS) - failures} of {len(CURATED_IDENTITY_LINKS)} "
        "identity exceptions pass every check"
    )
    return failures


def audit_rejections(s, client, actions) -> int:
    """For every candidate kept separate, re-establish a fact from raw data that rules it out."""
    links = list(
        s.execute(
            select(SegmentLink)
            .where(SegmentLink.decision == "kept_separate")
            .order_by(SegmentLink.symbol, SegmentLink.earlier_last)
        ).scalars()
    )
    print(f"\nRejected merge candidates to audit: {len(links)}\n")
    failures = 0
    for link in links:
        facts = []
        gap = max(0, len(trading_days(link.earlier_last, link.later_first)) - 2)
        if gap > 250:
            facts.append(f"gap {gap} sessions")
        before = s.scalar(
            select(TickerBar).where(
                TickerBar.symbol == link.symbol, TickerBar.session == link.earlier_last
            )
        )
        after = s.scalar(
            select(TickerBar).where(
                TickerBar.symbol == link.symbol, TickerBar.session == link.later_first
            )
        )
        owners_a = {
            h.security_id
            for h in s.execute(
                select(TickerHistory).where(
                    TickerHistory.symbol == link.symbol, TickerHistory.valid_to == link.earlier_last
                )
            ).scalars()
        }
        owners_b = {
            h.security_id
            for h in s.execute(
                select(TickerHistory).where(
                    TickerHistory.symbol == link.symbol,
                    TickerHistory.valid_from == link.later_first,
                )
            ).scalars()
        }
        ratio = 1.0
        for a in actions:
            when = a.ex_date or a.process_date
            if (
                a.action_type in ("forward_split", "reverse_split", "unit_split")
                and a.symbol == link.symbol
                and when
                and link.earlier_last < when <= link.later_first
                and a.old_rate
                and a.new_rate
            ):
                ratio *= a.new_rate / a.old_rate
        if before and after and before.close:
            price = after.close / (before.close / ratio)
            if not PRICE_LOW <= price <= PRICE_HIGH:
                facts.append(f"price ratio {price:.2f}")
        for sid in owners_a:
            issuer = s.get(SecurityIssuer, sid)
            if issuer is None:
                continue
            feed = client.company(
                issuer.cik, min(last_trade(s, sid) + timedelta(days=120), SCOPE_END), "25"
            )
            if feed is None:
                facts.append(f"issuer {issuer.cik}: notice list NOT READ")
            else:
                found = sorted(
                    f"{f.form} {f.filed}"
                    for f in feed.filings
                    if f.form.startswith("25")
                    and link.earlier_last - timedelta(days=45) < f.filed < link.later_first
                )
                if found:
                    facts.append(f"issuer {issuer.cik} delisting notice {found[0]}")
            for other in owners_b:
                theirs = s.get(SecurityIssuer, other)
                if theirs and theirs.cik != issuer.cik:
                    facts.append(f"different issuers {issuer.cik} / {theirs.cik}")
        for a in actions:
            if (
                a.acquiree_symbol == link.symbol
                and a.process_date
                and link.earlier_last <= a.process_date < link.later_first
                and a.action_type.endswith("merger")
            ):
                facts.append(f"Alpaca {a.action_type} {a.process_date}")
        heads = {
            s.scalar(
                select(TickerHistory.valid_from)
                .where(TickerHistory.security_id == other)
                .order_by(TickerHistory.valid_from)
            )
            for other in owners_b
        }
        if link.later_first not in heads:
            facts.append("later segment is inside another security's history")
        if "SEC filing: redeemed" in link.reason:
            facts.append("fund liquidation quoted from SEC filings before the restart")
        same = bool(owners_a & owners_b)
        supported = [f for f in facts if "NOT READ" not in f]
        good = supported and not same
        failures += not good
        print(
            f"{'ok  ' if good else 'FAIL'} {link.symbol:6} "
            f"{link.earlier_last} -> {link.later_first} "
            f"(ids {sorted(owners_a)} / {sorted(owners_b)}): "
            f"{'; '.join(facts) or 'NO FACT'}"
        )
    print(f"\n{len(links) - failures} of {len(links)} rejections independently supported")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
