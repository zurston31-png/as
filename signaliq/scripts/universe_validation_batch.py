#!/usr/bin/env python3
"""Run the SIP universe validation batch and print the findings.

Downloads SIP daily bars and corporate actions for the validation batch, up to
2024-12-31, gathers independent evidence (OpenFIGI, Wikidata, hand-entered
corrections checked against prices, SEC EDGAR delisting notices and 8-Ks) and
writes everything to the SIP research
database (data/sip_research.db by default). The market data database and the
Strategy A results are not read or written. Nothing is simulated.

    python scripts/universe_validation_batch.py [--report FILE] [--refresh]

OpenFIGI, Wikidata and SEC answers are stored and reused on later runs; --refresh
asks them again. SEC requests are read-only and identify the project with a
contact address in the User-Agent, as SEC requires.
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
from app.data.validation import trading_days
from app.database import create_session_factory, session_scope
from app.universe import checks, evidence
from app.universe import pipeline as p
from app.universe.batch import KNOWN_CASES
from app.universe.database import open_research_database, research_database_url
from app.universe.membership import MembershipRule
from app.universe.models import (
    BatchSymbol,
    CorporateAction,
    ExitEvidence,
    ExternalResponse,
    Membership,
    Security,
    SpinOff,
    SplitCheck,
    TickerBar,
    TickerHistory,
    TypeEvidence,
)

RULE = MembershipRule()
MAX_DISCOVERY_ROUNDS = 6


def run(settings, refresh: bool) -> dict:
    engine = open_research_database()
    factory = create_session_factory(engine)
    facts: dict = {"database": Path(research_database_url().removeprefix("sqlite:///")).name}
    try:
        with p.sip_client(settings) as client:
            with session_scope(factory) as session:
                facts["batch_size_initial"] = len(p.build_batch(session, client))
            with session_scope(factory) as session:
                # Every run starts from no stored corporate actions, so its result never
                # depends on an earlier run. Renames are followed until no new ticker appears.
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
                # Spin-off records name the parent "source_symbol", which a symbol filter misses.
                facts["spin_off_records"] = p.store_corporate_actions(
                    session,
                    p.fetch_corporate_actions(
                        client, symbols=None, types="spin_off", first=p.DATA_START, last=p.DATA_END
                    ),
                )
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
                evidence.apply_price_adjustments(session, client, p.utc_window)
            with session_scope(factory) as session:
                tickers = evidence.plain_tickers(session)
                facts["openfigi_requests"] = evidence.fetch_openfigi(
                    session, tickers, refresh=refresh
                )
            with session_scope(factory) as session:
                facts["wikidata_requests"], facts["wikidata_missing"] = evidence.fetch_wikidata(
                    session, tickers, refresh=refresh
                )
            with session_scope(factory) as session:
                facts["sec_requests"], facts["sec_failures"] = evidence.fetch_sec(
                    session, refresh=refresh
                )
            with session_scope(factory) as session:
                evidence.resolve_types(session)
                evidence.resolve_outcomes(session, trading_days(p.DATA_START, p.DATA_END))
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
        path = lambda sid: " -> ".join(h.symbol for h in history[sid])  # noqa: E731
        bar_counts = dict(
            s.execute(select(TickerBar.symbol, func.count()).group_by(TickerBar.symbol)).all()
        )
        exit_ev = defaultdict(list)
        for e in s.execute(select(ExitEvidence).order_by(ExitEvidence.id)).scalars():
            exit_ev[e.security_id].append(e)
        type_ev = defaultdict(list)
        for e in s.execute(select(TypeEvidence).order_by(TypeEvidence.id)).scalars():
            type_ev[e.security_id].append(e)

        print("SIP UNIVERSE VALIDATION BATCH (v3) - research data only, no strategy, no backtest")
        print(f"Research database: {facts['database']}")
        print(
            "Bars and corporate actions: SIP daily, 2016-01-01 to 2024-12-31. Evidence dated 2025 or later is ignored."
        )
        print(
            "Independent sources: Alpaca corporate actions, OpenFIGI (security type), Wikidata (company records),"
        )
        print(
            "SEC EDGAR (delisting notices and 8-Ks, read-only), hand-entered corrections checked against prices.\n"
        )

        print("1. BATCH")
        print(
            f"  Tickers: {len(batch)}  by selection: {dict(Counter(b.selection.split(':')[0] for b in batch))}"
        )
        print(
            f"  Rename discovery complete: {facts['discovery_complete']}; discovered: {len(facts['discovered'])}"
        )
        no_bars = sorted(b.symbol for b in batch if bar_counts.get(b.symbol, 0) == 0)
        print(f"  Tickers with no SIP bars 2016-2024 ({len(no_bars)}): {', '.join(no_bars)}")
        print(f"  Filler bars dropped: {sum(facts['fillers'].values()):,}")
        print(
            f"  Securities: {len(securities)}; status {dict(Counter(x.status for x in securities.values()))}; "
            f"with more than one ticker: {sum(1 for v in history.values() if len(v) > 1)}"
        )

        print("\n2. SOURCE COVERAGE")
        figi = {
            r.request_key: r.response
            for r in s.execute(
                select(ExternalResponse).where(ExternalResponse.source == "openfigi")
            ).scalars()
        }
        wiki = {
            r.request_key: r.response
            for r in s.execute(
                select(ExternalResponse).where(ExternalResponse.source == "wikidata")
            ).scalars()
        }
        print(
            f"  OpenFIGI: {sum(1 for v in figi.values() if v)} of {len(figi)} tickers found "
            f"(requests this run: {facts.get('openfigi_requests', 0)})"
        )
        print(
            f"  Wikidata: {sum(1 for v in wiki.values() if v)} of {len(wiki)} tickers have a company item "
            f"(requests this run: {facts.get('wikidata_requests', 0)})"
        )
        missing = facts.get("wikidata_missing", [])
        print(
            f"  Wikidata did not answer for {len(missing)} tickers this run (not looked up, not guessed)"
            + (f": {', '.join(missing)}" if missing else "")
        )
        print(f"  Alpaca spin-off records 2016-2024: {facts.get('spin_off_records', 0)}")
        sec_cached = s.scalar(
            select(func.count())
            .select_from(ExternalResponse)
            .where(ExternalResponse.source == "sec")
        )
        print(
            f"  SEC: {sec_cached} cached responses (requests this run: {facts.get('sec_requests', 0)}, "
            f"failed: {len(facts.get('sec_failures', []))})"
        )

        print("\n3. SECURITY TYPES")
        grid = Counter((x.security_type, x.type_status) for x in securities.values())
        for (kind, status), n in sorted(grid.items()):
            print(f"  {kind:12} {status:14} {n:4}")
        print("  Conflicts (sources disagree):")
        for sid, x in sorted(securities.items()):
            if x.type_status == "conflict":
                print(
                    f"    id {sid} {path(sid)}: "
                    + "; ".join(
                        f"{e.source}={e.security_type} ({e.detail[:60]})"
                        for e in type_ev[sid]
                        if e.source != "declared"
                    )
                )
        unknown = [sid for sid, x in securities.items() if x.type_status == "unknown"]
        print(
            f"  Unknown type ({len(unknown)}): "
            + ", ".join(f"id {sid} {path(sid)}" for sid in sorted(unknown))
        )
        mism = []
        for sid, x in securities.items():
            declared = {e.security_type for e in type_ev[sid] if e.source == "declared"}
            if declared and x.security_type not in declared:
                mism.append(
                    f"id {sid} {path(sid)}: declared {'/'.join(declared)}, resolved {x.security_type} ({x.type_status})"
                )
        print(
            f"  Known cases whose resolved type differs from the declared one: {len(mism)}"
            + "".join(f"\n    {m}" for m in mism)
        )

        print("\n4. EXIT OUTCOMES (ended securities)")
        ended = {sid: x for sid, x in securities.items() if x.status == "ended"}
        grid = Counter((x.outcome, x.outcome_status) for x in ended.values())
        for (outcome, status), n in sorted(grid.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            print(f"  {outcome:24} {status:14} {n:4}")
        print(
            f"  Value assumed: {sum(1 for x in ended.values() if x.assumed_exit_value is not None)} "
            f"(total loss {sum(1 for x in ended.values() if x.assumed_exit_value == 0.0)}); "
            f"no value assumed: {sum(1 for x in ended.values() if x.assumed_exit_value is None)}"
        )
        for status in ("confirmed", "single_source", "conflict", "unresolved"):
            rows = sorted((sid, x) for sid, x in ended.items() if x.outcome_status == status)
            print(f"  --- {status} ({len(rows)}) ---")
            for sid, x in rows:
                value = "none" if x.assumed_exit_value is None else x.assumed_exit_value
                print(
                    f"    id {sid} {path(sid)} (last trade {x.last_trade_session}): {x.outcome}, value {value}"
                )
                for e in exit_ev[sid]:
                    print(f"        [{e.source}] {e.category}: {e.detail[:140]}")
        print("  Known cases (expected against the evidence):")
        for case in KNOWN_CASES:
            for sid in sorted(set(by_symbol.get(case.symbol, []))):
                x = securities[sid]
                if x.status == "ended":
                    print(
                        f"    {case.symbol:6} id {sid}: {x.outcome} ({x.outcome_status}) | expected: {case.expected}"
                    )

        print("\n4b. SEC CROSS-CHECK (every ended security)")
        print(
            "  A company is accepted only if it filed a Form 25/25-NSE between 45 days before and 90 days after"
        )
        print(
            "  the last trade (before 2025). Search keys (tickers, names) only propose candidates."
        )
        sec_found = evidence.cached_sec_findings(s)
        matched = {sid: f for sid, f in sec_found.items() if f.cik}
        print(f"  Matched: {len(matched)} of {len(sec_found)}")
        for sid, f in sorted(sec_found.items()):
            x = securities[sid]
            head = f"    id {sid} {path(sid)} (last trade {x.last_trade_session}): "
            if not f.cik:
                print(head + f"NO MATCH - {f.note}")
                continue
            print(
                head
                + f"{f.company} (CIK {f.cik}, SIC {f.sic or 'none'}{', blank check' if f.sic == '6770' else ''})"
            )
            for filing, n in f.notices:
                print(
                    f"        {filing.form} {filing.filed} {n.exchange}: '{n.class_description}', {n.rule.replace('17 CFR 240.', '')}"
                )
            for k in f.eight_ks:
                print(f"        8-K {k.filed}: {k.items}")
            if f.note:
                print(f"        note: {f.note}")
            print(
                f"        -> type {x.security_type} ({x.type_status}); outcome {x.outcome} ({x.outcome_status})"
            )

        print("\n5. SPLITS AND SPIN-OFFS")
        verdicts = Counter(c.verdict for c in s.execute(select(SplitCheck)).scalars())
        print(f"  Splits owned by batch securities, checked against prices: {dict(verdicts)}")
        for c in s.execute(select(SplitCheck).where(SplitCheck.verdict != "consistent")).scalars():
            print(
                f"    {c.verdict}: {c.symbol} {c.ex_date} ratio {c.ratio:g}, close before {c.close_before}, on {c.close_on}"
                + (" -> NOT applied" if c.verdict == "contradicted" else " -> applied, unverified")
            )
        for so in s.execute(select(SpinOff)).scalars():
            kids = ", ".join(
                f"{c['ratio']:g} {c['symbol']} at {c['close_on_ex_date']}" for c in so.children
            )
            print(
                f"  Spin-off id {so.security_id} {so.parent_symbol} {so.ex_date} [{so.source}]: {kids}; parent close "
                f"{so.parent_close_before} -> {so.parent_close_on}; distributed value {so.distribution_value}; "
                f"holder return {'n/a' if so.holder_return is None else f'{so.holder_return:+.1%}'}; factor "
                f"{so.factor}; {so.status}"
                + (" -> applied" if so.status == "price_consistent" else " -> NOT applied")
            )
        diffs = checks.adjustment_differences(s)
        compared = sum(v["compared"] for v in diffs.values())
        within = sum(v["within"] for v in diffs.values())
        print(
            f"  Our split adjustment against Alpaca's split-adjusted close: {within:,} of {compared:,} sessions within 0.1%"
        )
        for worst, sid, v in sorted(
            ((v["worst"], sid, v) for sid, v in diffs.items() if v["worst"] > 1e-3), reverse=True
        ):
            print(
                f"    id {sid} ({path(sid)}): {v['compared'] - v['within']} of {v['compared']} sessions off, worst {worst:.1%} on {v['worst_session']}"
            )
        jumps = checks.large_jumps(s)
        unexplained = [j for j in jumps if not j.explained_by_split]
        print(
            f"  Raw moves beyond +80%/-45%: {len(jumps)}; at an applied split {len(jumps) - len(unexplained)}; "
            f"real moves {sum(1 for j in unexplained if j.adjusted_also_jumps)}; split missing "
            f"{sum(1 for j in unexplained if j.adjusted_also_jumps is False)}; no adjusted price "
            f"{sum(1 for j in unexplained if j.adjusted_also_jumps is None)}"
        )
        for j in unexplained:
            if j.adjusted_also_jumps is False:
                print(
                    f"    missing split? id {j.security_id} {j.symbol} {j.session}: {j.previous_close} -> {j.close}"
                )

        print(
            "\n6. POINT-IN-TIME MEMBERSHIP (validation rule: top 25 eligible common stocks of the batch)"
        )
        print(f"  Rule: {RULE.to_dict()}  fingerprint {RULE.fingerprint()[:16]}")
        decisions = facts["decisions"]
        members = checks.stored_members(s)
        sizes = [len(v) for v in members.values()]
        print(
            f"  Decisions: {len(decisions)} ({decisions[0]} to {decisions[-1]}); members per decision {min(sizes)}-{max(sizes)}"
        )
        ever = {sid for v in members.values() for sid in v}
        print(
            f"  Distinct members: {len(ever)}; types: {dict(Counter((securities[sid].security_type, securities[sid].type_status) for sid in ever))}"
        )
        reasons = Counter(
            r.reason
            for r in s.execute(select(Membership).where(Membership.eligible == 0)).scalars()
        )
        print(f"  Not-eligible reasons (all decisions): {dict(reasons)}")
        print("  Members that later ended:")
        for sid in sorted(ever):
            x = securities[sid]
            if x.status == "ended":
                last_member = max(d for d, v in members.items() if sid in v)
                value = "none" if x.assumed_exit_value is None else x.assumed_exit_value
                print(
                    f"    id {sid} {path(sid)}: member until {last_member}, last trade {x.last_trade_session}, "
                    f"{x.outcome} ({x.outcome_status}), value {value}"
                )
        after_exit = [
            (sid, d)
            for d, v in members.items()
            for sid in v
            if d > securities[sid].last_trade_session
        ]
        print(f"  Members chosen after their last trade: {len(after_exit)}")
        differ = checks.recompute_with_truncated_data(s, RULE, p.DATA_START)
        print(
            f"  Recomputed from data cut off at each decision: {len(decisions) - len(differ)} of {len(decisions)} identical"
        )
        cut = decisions[len(decisions) // 2]
        print(
            f"  Every price and volume after {cut} multiplied by 10: earlier decisions unchanged: "
            f"{checks.tamper_future(s, RULE, p.DATA_START, cut)}"
        )
        leaks = checks.adjusted_price_would_change_eligibility(s, RULE)
        print(
            f"  Decisions where the $5 filter would answer differently on adjusted prices: {len(leaks)}"
        )
    engine.dispose()
    return out.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/universe_validation_batch.py")
    parser.add_argument("--report", type=Path, help="also write the report to this file")
    parser.add_argument(
        "--refresh", action="store_true", help="ask OpenFIGI, Wikidata and SEC again"
    )
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    text = report(run(get_settings(), args.refresh))
    print(text)
    if args.report:
        args.report.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
