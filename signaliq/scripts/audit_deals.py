#!/usr/bin/env python3
"""Independent audit of stock and cash-and-stock deal values, from raw data. Read-only.

    python scripts/audit_deals.py [--report PATH]

For every curated deal matched to an ended security (``CURATED_DEALS``), without the
pipeline's valuation code:

1. every quoted passage is in its cached SEC document;
2. the effective date is written in the target's own filing ("May 8, 2020");
3. the target has no trade after the effective date;
4. the acquirer's first real trade (volume > 0) on or after the effective date, read
   from the raw cached Alpaca bars, is the price used;
5. value = cash + shares x that close, recomputed, equals the stored deal value and the
   security's exit value (when verified); and it lies within 10% of the target's last
   close, or the value is not used.

Exits with status 1 if any check fails.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select

from app.data.validation import NEW_YORK
from app.database import create_session_factory, session_scope
from app.universe import sec
from app.universe import sec_history as sh
from app.universe.curated import CURATED_DEALS
from app.universe.database import open_research_database
from app.universe.models import DealValue, ExternalResponse, Security, SecurityBar

TOLERANCE = 0.10


def written(day) -> set[str]:
    return {f"{day:%B} {day.day}, {day.year}", f"{day:%B} {day.day} , {day.year}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    lines: list[str] = []
    failures = checked = 0

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    engine = open_research_database()
    with session_scope(create_session_factory(engine)) as s:
        client = sec.CachedSec(s, network=False)
        ended = {
            (x.last_symbol, x.last_trade_session): x
            for x in s.execute(select(Security).where(Security.status == "ended")).scalars()
        }
        stored = {d.security_id: d for d in s.execute(select(DealValue)).scalars()}
        for deal in CURATED_DEALS:
            security = ended.get((deal.target_symbol, deal.last_trade))
            if security is None:
                continue
            checked += 1
            problems = []
            docs = {q.url: sh.plain_text(client.get(q.url)) for q in deal.quotes}
            missing = [q for q in deal.quotes if " ".join(q.text.split()) not in docs[q.url]]
            if missing:
                problems.append(f"{len(missing)} quote(s) not found")
            if not any(w in text for text in docs.values() for w in written(deal.effective_date)):
                problems.append(f"effective date {deal.effective_date} not written in the filing")
            after = s.scalar(
                select(func.count()).where(
                    SecurityBar.security_id == security.security_id,
                    SecurityBar.session > deal.effective_date,
                )
            )
            if after:
                problems.append(f"target traded {after} sessions after the effective date")
            last_close = s.scalar(
                select(SecurityBar.close).where(
                    SecurityBar.security_id == security.security_id,
                    SecurityBar.session == security.last_trade_session,
                )
            )
            raw = s.get(
                ExternalResponse,
                ("alpaca_deal", f"{deal.acquirer_symbol}:{deal.effective_date.isoformat()}"),
            )
            bars = []
            for item in raw.response if raw else []:
                day = datetime.fromisoformat(item["t"].replace("Z", "+00:00")).astimezone(NEW_YORK)
                if day.date() >= deal.effective_date and int(item["v"]) > 0:
                    bars.append((day.date(), float(item["c"])))
            bars.sort()
            if not bars:
                problems.append(
                    f"no raw {deal.acquirer_symbol} trade on or after the effective date"
                )
                value = None
            else:
                session_day, close = bars[0]
                value = deal.cash_per_share + deal.shares_per_share * close
            row = stored.get(security.security_id)
            near = bool(value and last_close and abs(value / last_close - 1) <= TOLERANCE)
            if row is None:
                problems.append("no stored deal value")
            else:
                if value is not None and (row.value is None or abs(row.value - value) > 1e-9):
                    problems.append(f"stored value {row.value} != recomputed {value}")
                if (row.status == "verified") != (near and not missing):
                    problems.append(f"stored status {row.status}, recomputed near={near}")
                if row.status == "verified" and (
                    security.assumed_exit_value is None
                    or abs(security.assumed_exit_value - value) > 1e-9
                ):
                    problems.append(f"exit value {security.assumed_exit_value} != {value}")
            failures += bool(problems)
            say(
                f"{'FAIL' if problems else 'ok  '} {deal.target_symbol:5} "
                f"effective {deal.effective_date}: "
                f"{deal.cash_per_share:g} cash + {deal.shares_per_share:g} {deal.acquirer_symbol}"
                + (
                    f" @ {close} ({session_day}) = {value:.4f}; target last close {last_close} "
                    f"({value / last_close - 1:+.1%})"
                    if bars and last_close
                    else ""
                )
                + f"; quotes {len(deal.quotes) - len(missing)}/{len(deal.quotes)}"
                + f"; stored {row.status if row else None}, "
                f"exit value {security.assumed_exit_value}"
                + (f"; {'; '.join(problems)}" if problems else "")
            )
        say(f"\n{checked - failures} of {checked} deals pass every check")
    engine.dispose()
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
