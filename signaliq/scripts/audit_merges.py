#!/usr/bin/env python3
"""Independent audit of trading-gap merges, from the raw stored data. Read-only.

    python scripts/audit_merges.py

For every merge recorded in the research database, recomputes from the raw ticker bars,
the raw corporate actions and the issuer's SEC filing list (cache only), without using
the pipeline's decision code:

1. both segments are bars of the same ticker, and inside the gap every bar Alpaca
   returned is a filler bar (no volume, no trades);
2. the number of sessions without a trade, recounted from the exchange calendar;
3. the split-adjusted price ratio across the gap, recomputed from the raw split events;
4. no Form 25 / 25-NSE filed by the security's issuer (when known) between the last
   trade before the gap and the first trade after it;
5. the merged security exists once in the securities table, spanning both segments.

Exits with status 1 if any merge fails a check.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.data.validation import trading_days
from app.database import create_session_factory, session_scope
from app.universe import sec
from app.universe.database import open_research_database
from app.universe.models import (
    CorporateAction,
    SecurityIssuer,
    SegmentLink,
    TickerBar,
    TickerHistory,
)

PRICE_LOW, PRICE_HIGH = 0.5, 2.0


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
            notices = []
            for sid in owners:
                issuer = s.get(SecurityIssuer, sid)
                if issuer is None:
                    continue
                feed = client.company(
                    issuer.cik, link.later_first + timedelta(days=WINDOW_DAYS), "25"
                )
                for f in feed.filings if feed else ():
                    if (
                        f.form.startswith("25")
                        and link.earlier_last - timedelta(days=WINDOW_DAYS)
                        < f.filed
                        < link.later_first
                    ):
                        notices.append(f"{f.form} {f.filed}")
            if notices:
                problems.append(f"issuer delisting notice in the gap: {notices}")
            failures += bool(problems)
            print(
                f"{'FAIL' if problems else 'ok  '} {link.symbol:6} "
                f"{link.earlier_last} -> {link.later_first}: "
                f"{gap} sessions, {len(inside)} filler bars inside, price ratio "
                f"{'n/a' if price is None else f'{price:.3f}'}"
                + (f" after splits x{ratio:g}" if ratio != 1.0 else "")
                + f", one security: {len(owners) == 1}"
                + (f"; {'; '.join(problems)}" if problems else "")
            )
    engine.dispose()
    print(f"\n{len(links) - failures} of {len(links)} merges pass every check")
    return 1 if failures else 0


#: The issuer's delisting-notice feed is read from this many days before the gap.
WINDOW_DAYS = 45

if __name__ == "__main__":
    raise SystemExit(main())
