#!/usr/bin/env python3
"""Run the SIP universe validation batch and print the findings.

Downloads SIP daily bars (raw and split-adjusted) and corporate actions for
the validation batch, up to 2024-12-31, and writes everything to the SIP
research database (data/sip_research.db by default). The market data database
and the Strategy A results are not read or written. Nothing is simulated.

    python scripts/universe_validation_batch.py [--report FILE]
"""

import argparse
import io
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, func, select

from app.config import configure_logging, get_settings
from app.database import create_session_factory, session_scope
from app.universe import checks
from app.universe import pipeline as p
from app.universe.batch import KNOWN_CASES
from app.universe.database import open_research_database, research_database_url
from app.universe.membership import MembershipRule
from app.universe.models import (
    BatchSymbol,
    CorporateAction,
    Membership,
    Security,
    TickerBar,
    TickerHistory,
)

RULE = MembershipRule()
MAX_DISCOVERY_ROUNDS = 6


def run(settings) -> dict:
    engine = open_research_database()
    factory = create_session_factory(engine)
    facts: dict = {"database": Path(research_database_url().removeprefix("sqlite:///")).name}
    try:
        with p.sip_client(settings) as client:
            with session_scope(factory) as session:
                batch = p.build_batch(session, client)
                facts["batch_size_initial"] = len(batch)
            with session_scope(factory) as session:
                # Every run starts from no stored events, so its result never depends on an
                # earlier run. Renames are then followed until no new ticker appears.
                session.execute(delete(CorporateAction))
                to_fetch = [r.symbol for r in session.execute(select(BatchSymbol)).scalars()]
                facts["discovered"] = []
                for _round in range(MAX_DISCOVERY_ROUNDS):
                    for i in range(0, len(to_fetch), 20):
                        p.store_corporate_actions(
                            session,
                            p.fetch_corporate_actions(
                                client,
                                symbols=to_fetch[i : i + 20],
                                types=p.CA_TYPES,
                                first=p.DATA_START,
                                last=p.DATA_END,
                            ),
                        )
                    discovered = p.discover_linked_symbols(session, rounds=1)
                    facts["discovered"] += [(r.symbol, r.discovered_from) for r in discovered]
                    to_fetch = [r.symbol for r in discovered]
                    if not to_fetch:
                        break
                facts["discovery_complete"] = not to_fetch
            with session_scope(factory) as session:
                symbols = [r.symbol for r in session.execute(select(BatchSymbol)).scalars()]
                p.store_ticker_bars(session, p.fetch_literal_bars(client, symbols))
            with session_scope(factory) as session:
                assembled = p.assemble(session)
                p.store_securities(session, assembled)
                p.store_security_bars(session, assembled)
                facts["fillers"] = assembled.fillers_by_symbol
                facts["entity_check"] = p.attach_alpaca_adjusted(session, client, settings)
            with session_scope(factory) as session:
                facts["decisions"] = p.decide_membership(session, RULE)
    finally:
        engine.dispose()
    return facts


def report(facts: dict) -> str:
    engine = open_research_database()
    factory = create_session_factory(engine)
    out = io.StringIO()
    with redirect_stdout(out), session_scope(factory) as s:
        batch = list(s.execute(select(BatchSymbol).order_by(BatchSymbol.symbol)).scalars())
        securities = {x.security_id: x for x in s.execute(select(Security)).scalars()}
        history = defaultdict(list)
        for h in s.execute(select(TickerHistory).order_by(TickerHistory.valid_from)).scalars():
            history[h.security_id].append(h)
        by_symbol = defaultdict(list)
        for sid, items in history.items():
            for h in items:
                by_symbol[h.symbol].append(sid)
        bar_counts = dict(
            s.execute(select(TickerBar.symbol, func.count()).group_by(TickerBar.symbol)).all()
        )

        print("SIP UNIVERSE VALIDATION BATCH - research data only, no strategy, no backtest")
        print(f"Research database: {facts['database']}")
        print("Bars: SIP daily, 2016-01-01 to 2024-12-31 (nothing from 2025 on was read)\n")

        print("1. BATCH")
        print(
            f"  Tickers requested: {len(batch)}  by selection: {dict(Counter(b.selection.split(':')[0] for b in batch))}"
        )
        print(f"  By category: {dict(sorted(Counter(b.category for b in batch).items()))}")
        print(
            f"  Discovered through renames: {', '.join(f'{a} (from {b})' for a, b in facts['discovered']) or 'none'}"
        )
        print(
            f"  Rename discovery complete (no new ticker in the last round): {facts['discovery_complete']}"
        )
        no_bars = sorted(b.symbol for b in batch if bar_counts.get(b.symbol, 0) == 0)
        print(
            f"  Tickers with no SIP bars 2016-2024 ({len(no_bars)}): {', '.join(no_bars) or 'none'}"
        )
        total_fill = sum(facts["fillers"].values())
        heavy = sorted(((v, k) for k, v in facts["fillers"].items() if v), reverse=True)[:12]
        print(
            f"  Filler bars (no trade, price copied) dropped: {total_fill:,} in "
            f"{sum(1 for v in facts['fillers'].values() if v)} tickers; most: "
            + ", ".join(f"{k} {v}" for v, k in heavy)
        )

        print("\n2. SECURITIES (internal ids)")
        print(
            f"  Securities: {len(securities)}; status {dict(Counter(x.status for x in securities.values()))}"
        )
        print(
            f"  Exit types: {dict(sorted(Counter(x.exit_type for x in securities.values()).items()))}"
        )
        multi = [sid for sid, items in history.items() if len(items) > 1]
        print(f"  Securities with more than one ticker (renames linked): {len(multi)}")
        reused = {sym: sids for sym, sids in by_symbol.items() if len(set(sids)) > 1}
        print(
            f"  Tickers used by more than one security: {len(reused)}: "
            + ", ".join(f"{k} -> ids {sorted(set(v))}" for k, v in sorted(reused.items()))
        )

        print("\n3. KNOWN CASES: what the data shows against what really happened")
        for case in KNOWN_CASES:
            sids = sorted(set(by_symbol.get(case.symbol, [])))
            print(f"  {case.symbol:6} expected: {case.expected}")
            if not sids:
                print(
                    "         RESULT: no security found (no SIP bars for this ticker in 2016-2024)"
                )
            for sid in sids:
                sec = securities[sid]
                path = " -> ".join(f"{h.symbol} {h.valid_from}..{h.valid_to}" for h in history[sid])
                value = (
                    ""
                    if sec.assumed_exit_value is None
                    else f", assumed value {sec.assumed_exit_value}"
                )
                print(f"         id {sid}: {path}; {sec.status}, exit {sec.exit_type}{value}")

        print("\n4. TICKER HISTORY AGAINST ALPACA'S OWN COMPANY MAPPING (raw closes and volumes)")
        ec = facts["entity_check"]
        full = sum(1 for v in ec.values() if v["mismatch"] == 0 and v["missing"] == 0)
        print(
            f"  Securities whose every real trade matches Alpaca's mapped series for their last ticker: {full} of {len(ec)}"
        )
        for sid, v in sorted(ec.items()):
            if v["mismatch"] or v["missing"]:
                sec = securities[sid]
                print(
                    f"    id {sid} ({' -> '.join(h.symbol for h in history[sid])}, last trade {sec.last_trade_session}): "
                    f"{v['match']} match, {v['mismatch']} differ, {v['missing']} not in Alpaca's mapped series"
                )

        print("\n5. RAW AGAINST SPLIT-ADJUSTED PRICES")
        diffs = checks.adjustment_differences(s)
        splits_n = s.execute(
            select(func.count())
            .select_from(CorporateAction)
            .where(
                CorporateAction.action_type.in_(("forward_split", "reverse_split", "unit_split"))
            )
        ).scalar_one()
        compared = sum(v["compared"] for v in diffs.values())
        within = sum(v["within"] for v in diffs.values())
        print(f"  Split events recorded for batch tickers: {splits_n}")
        print(
            f"  Sessions compared (our raw/factor against Alpaca's adjusted close): {compared:,}; within 0.1%: {within:,}"
        )
        bad = sorted(
            ((v["worst"], sid, v) for sid, v in diffs.items() if v["worst"] > 1e-3), reverse=True
        )
        print(f"  Securities with any session off by more than 0.1%: {len(bad)}")
        for worst, sid, v in bad[:15]:
            print(
                f"    id {sid} ({' -> '.join(h.symbol for h in history[sid])}): worst {worst:.1%} on {v['worst_session']}, "
                f"{v['compared'] - v['within']} of {v['compared']} sessions off"
            )
        jumps = checks.large_jumps(s)
        unexplained = [j for j in jumps if not j.explained_by_split]
        print(
            f"  Day-to-day raw moves beyond +80%/-45%: {len(jumps)}; explained by a recorded split: "
            f"{len(jumps) - len(unexplained)}; not explained: {len(unexplained)}"
        )
        for j in unexplained[:20]:
            note = {
                True: "adjusted series jumps too (a real move)",
                False: "adjusted series does NOT jump (split missing from our events)",
                None: "no adjusted price",
            }[j.adjusted_also_jumps]
            print(
                f"    id {j.security_id} {j.symbol} {j.session}: {j.previous_close} -> {j.close} ({note})"
            )
        leaks = checks.adjusted_price_would_change_eligibility(s, RULE)
        print(
            f"  Decisions where the $5 filter would answer differently on adjusted prices: {len(leaks)}"
            + (
                f" (e.g. {', '.join(f'id {sid} on {d}: raw {r:.2f}, adjusted {a:.2f}' for d, sid, r, a in leaks[:3])})"
                if leaks
                else ""
            )
        )

        print("\n6. POINT-IN-TIME MEMBERSHIP (validation rule: top 25 of the batch)")
        print(f"  Rule: {RULE.to_dict()}  fingerprint {RULE.fingerprint()[:16]}")
        decisions = facts["decisions"]
        print(f"  Decisions: {len(decisions)} month-ends, {decisions[0]} to {decisions[-1]}")
        members = checks.stored_members(s)
        sizes = [len(v) for v in members.values()]
        print(f"  Members per decision: min {min(sizes)}, max {max(sizes)}")
        ended = {sid for sid, x in securities.items() if x.status == "ended"}
        ever = {sid for v in members.values() for sid in v}
        ended_members = sorted(ever & ended)
        print(
            f"  Distinct members: {len(ever)}, of which later ended (delisted while or after being a member): {len(ended_members)}"
        )
        for sid in ended_members:
            sec = securities[sid]
            last_member = max(d for d, v in members.items() if sid in v)
            print(
                f"    id {sid} {sec.last_symbol}: member until the {last_member} decision, last trade "
                f"{sec.last_trade_session}, exit {sec.exit_type}"
            )
        after_exit = [
            (sid, d)
            for d, v in members.items()
            for sid in v
            if d > securities[sid].last_trade_session
        ]
        print(f"  Members chosen after their last trade: {len(after_exit)}")
        reasons = Counter(
            r.reason
            for r in s.execute(select(Membership).where(Membership.eligible == 0)).scalars()
        )
        print(f"  Not-eligible reasons (all decisions): {dict(reasons)}")
        differ = checks.recompute_with_truncated_data(s, RULE, p.DATA_START)
        print(
            f"  Recomputed from data cut off at each decision: {len(decisions) - len(differ)} of {len(decisions)} identical"
        )
        cut = decisions[len(decisions) // 2]
        print(
            f"  Every price and volume after {cut} multiplied by 10: decisions up to then unchanged: "
            f"{checks.tamper_future(s, RULE, p.DATA_START, cut)}"
        )
        unknown = sorted(
            {
                securities[sid].last_symbol
                for sid in ever
                if securities[sid].declared_type == "unknown"
            }
        )
        print(
            f"  Members whose security type is not known (ETF/preferred/etc. not ruled out): {len(unknown)}: {', '.join(unknown)}"
        )
    engine.dispose()
    return out.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/universe_validation_batch.py")
    parser.add_argument("--report", type=Path, help="also write the report to this file")
    parser.add_argument(
        "--report-only", action="store_true", help="report on the stored batch without fetching"
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("WARNING")
    facts = {} if args.report_only else run(settings)
    if args.report_only:
        print("--report-only needs the facts of a run; run without it.", file=sys.stderr)
        return 2
    text = report(facts)
    print(text)
    if args.report:
        args.report.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
