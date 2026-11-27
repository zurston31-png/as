#!/usr/bin/env python3
"""Run the SIP universe validation batch and print the findings.

Downloads SIP daily bars and corporate actions for the validation batch, up to
2024-12-31, gathers independent evidence (OpenFIGI, Wikidata, hand-entered
corrections checked against prices, SEC EDGAR delisting notices and 8-Ks) and
writes everything to the SIP research database named by
SIGNALIQ_SIP_RESEARCH_DATABASE_URL (required: there is no default, and the archived
databases are refused). The market data database and the
Strategy A results are not read or written. Nothing is simulated.

    python scripts/universe_validation_batch.py [--report FILE] [--refresh]

OpenFIGI, Wikidata and SEC answers are stored and reused on later runs; --refresh
asks them again. SEC requests are read-only and identify the project with a
contact address in the User-Agent, as SEC requires.
"""

import argparse
import io
import re
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import timedelta
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, func, select

from app.config import configure_logging, get_settings
from app.data.validation import trading_days
from app.database import create_session_factory, session_scope
from app.universe import checks, curated, evidence, rules, sec_cache, sectypes
from app.universe import pipeline as p
from app.universe import segments as seg
from app.universe import stage as universe_stage
from app.universe.batch import KNOWN_CASES
from app.universe.database import open_research_database, research_database_url
from app.universe.membership import MembershipRule
from app.universe.models import (
    BatchSymbol,
    Candidate,
    CorporateAction,
    DealValue,
    ExitEvidence,
    ExternalResponse,
    IssuerEvidence,
    IssuerSic,
    Membership,
    Security,
    SecurityBar,
    SecurityIssuer,
    SecurityRegistry,
    SegmentLink,
    SpinOff,
    SplitCheck,
    TickerBar,
    TickerHistory,
    TypeEvidence,
)

RULE = MembershipRule()
MAX_MERGE_PASSES = 4
MAX_COMPLETENESS_ROUNDS = 4


def stage_pool(session, client, settings):
    """Candidate pool and monthly bars for a stage (see app.universe.stage)."""
    sec_client = evidence.sec.CachedSec(session, network=True)
    try:
        return universe_stage.collect_pool(
            session,
            client,
            fetch_actions=p.fetch_corporate_actions,
            utc_window=p.utc_window,
            sec_get=sec_client.get,
            user_agent=evidence.USER_AGENT,
        )
    finally:
        sec_client.close()


MAX_DISCOVERY_ROUNDS = 6


SEC_STEPS = (("curated", "curated documents"), ("sec", "evidence"), ("issuer", "issuers"))


def _sec_cached(session) -> int:
    """Distinct cached SEC answers, including a read-only parent cache's (``sec_cache``)."""
    return sec_cache.count(session)


def _count_sec(facts: dict, step: str, result: tuple[int, int, list[str]]) -> None:
    """Add one SEC fetch's (requests, answers stored, failures) to the run's totals."""
    made, stored, failed = result
    facts[f"{step}_requests"] = facts.get(f"{step}_requests", 0) + made
    facts[f"{step}_stored"] = facts.get(f"{step}_stored", 0) + stored
    facts[f"{step}_failures"] = facts.get(f"{step}_failures", []) + failed


def _print_dated_types(s) -> None:
    """The dated-type rule's changes (measurement runs only)."""
    client = evidence.StrictCache(s)
    changes = rules.type_changes(rules.cover_classes(client, evidence.stored_issuer_points(s)))
    print(f"  Dated-type rule: securities whose own cover pages changed class: {len(changes)}")
    for sid, pages in sorted(changes.items()):
        security = s.get(Security, sid)
        print(
            f"    id {sid} {security.last_symbol}: undated type {security.security_type} "
            f"({security.type_status})"
        )
        for kind, first, last, count in rules.type_runs(pages):
            print(
                f"      {kind} {first.filed}..{last.filed} ({count} cover pages): "
                f"first {first.form} '{first.title}' [{first.url}]"
            )


def run(settings, refresh: bool, stage: int | None = None) -> dict:
    engine = open_research_database()
    factory = create_session_factory(engine)
    facts: dict = {"database": Path(research_database_url().removeprefix("sqlite:///")).name}
    facts["refresh"] = refresh
    with session_scope(factory) as session:
        facts["sec_cached_at_start"] = _sec_cached(session)
    try:
        with p.sip_client(settings) as client:
            if stage:
                with session_scope(factory) as session:
                    pool, facts["pool"] = stage_pool(session, client, settings)
            forced: dict[str, str] = {}
            facts["completeness_rounds"] = []
            for _round in range(MAX_COMPLETENESS_ROUNDS if stage else 1):
                with session_scope(factory) as session:
                    if stage:
                        selected = universe_stage.screen_candidates(session, pool, stage, forced)
                        facts["batch_size_initial"] = universe_stage.build_stage_batch(
                            session, selected
                        )
                    else:
                        facts["batch_size_initial"] = len(p.build_batch(session, client))
                with session_scope(factory) as session:
                    # Every run starts from no stored corporate actions, so its result never
                    # depends on an earlier run. Renames are followed until no new ticker appears.
                    session.execute(delete(CorporateAction))
                    # Renames that Alpaca lacks, read from SEC filings and re-verified each run.
                    _count_sec(
                        facts, "curated", evidence.fetch_curated_documents(session, refresh=refresh)
                    )
                    facts["renames"] = [
                        (r.old_symbol, r.new_symbol, r.date, len(r.quotes), len(missing))
                        for r, missing in evidence.apply_curated_renames(session)
                    ]
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
                            client,
                            symbols=None,
                            types="spin_off",
                            first=p.DATA_START,
                            last=p.DATA_END,
                        ),
                    )
                with session_scope(factory) as session:
                    symbols = [r.symbol for r in session.execute(select(BatchSymbol)).scalars()]
                    p.store_ticker_bars(session, p.fetch_literal_bars(client, symbols))
                    # Trading-gap merges are derived on every run, from no stored links.
                    session.execute(delete(SegmentLink))
                # Identity and evidence are built, then trading gaps decided from that evidence;
                # securities are rebuilt with the merges until no new merge appears.
                facts["merge_passes"] = 0
                for _pass in range(MAX_MERGE_PASSES):
                    facts["merge_passes"] += 1
                    with session_scope(factory) as session:
                        assembled = p.assemble(session)
                        p.store_securities(session, assembled)
                        p.store_security_bars(session, assembled)
                        facts["fillers"] = assembled.fillers_by_symbol
                        facts["entity_check"] = p.attach_alpaca_adjusted(session, client, settings)
                    with session_scope(factory) as session:
                        evidence.apply_price_adjustments(session, client, p.utc_window)
                    if _pass == 0:
                        with session_scope(factory) as session:
                            tickers = evidence.plain_tickers(session)
                            facts["openfigi_requests"] = evidence.fetch_openfigi(
                                session, tickers, refresh=refresh
                            )
                        with session_scope(factory) as session:
                            facts["wikidata_requests"], facts["wikidata_missing"] = (
                                evidence.fetch_wikidata(session, tickers, refresh=refresh)
                            )
                    # SEC requests are added up over every pass and step, so the report
                    # accounts for each one (checked against the growth of the cache).
                    for step, fetch in (
                        ("sec", evidence.fetch_sec),
                        ("issuer", evidence.fetch_issuers),
                    ):
                        with session_scope(factory) as session:
                            _count_sec(facts, step, fetch(session, refresh=refresh))
                    with session_scope(factory) as session:
                        facts["issuer_conflicts"] = evidence.store_issuers(session)
                    with session_scope(factory) as session:
                        evidence.value_deals(session, client, p.utc_window)
                    with session_scope(factory) as session:
                        evidence.resolve_types(session)
                        evidence.resolve_outcomes(session, trading_days(p.DATA_START, p.DATA_END))
                    with session_scope(factory) as session:
                        new = evidence.decide_gap_links(session)
                        renamed = evidence.decide_issuer_renames(session)
                        if not any(row.decision == "merged" for row in new) and not renamed:
                            break
                facts["handovers"] = assembled.handovers
                facts["merges_stable"] = not any(row.decision == "merged" for row in new) and (
                    not renamed
                )
                with session_scope(factory) as session:
                    facts["decisions"] = p.decide_membership(session, RULE)
                if not stage:
                    break
                with session_scope(factory) as session:
                    gaps = universe_stage.completeness_gaps(
                        session,
                        universe_stage.nth_member_volume(session, RULE.size),
                        RULE.min_price,
                    )
                facts["completeness_rounds"].append(len(gaps))
                if not gaps:
                    break
                for symbol, (decision, bound, threshold) in gaps.items():
                    forced[symbol] = (
                        f"could have been a member on {decision}: dollar-volume bound "
                        f"{bound:,.0f} >= N-th member {threshold:,.0f}"
                    )
            facts["completeness_gaps_final"] = len(gaps) if stage else None
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

        if facts.get("pool"):
            print(
                f"SIP UNIVERSE STAGE {RULE.size} (top {RULE.size} eligible common stocks per month) - research data only, no strategy, no backtest"
            )
        else:
            print(
                "SIP UNIVERSE VALIDATION BATCH (v5) - research data only, no strategy, no backtest"
            )
        if facts.get("pool"):
            print(
                "Version 1.1 (Stage 1 cleanup): point-in-time issuers from dated SEC evidence, explicit\n"
                "security-type rules, the documented YNDX/NBIS identity exception, deal terms for AGN,\n"
                "WORK and RTN. Selection rule, membership logic and exit-value treatment unchanged."
            )
        print(f"Research database: {facts['database']}")
        if rules.enabled_rules():
            # Only measurement runs print this: with no rule switched on, the report is
            # unchanged.
            print(
                f"MEASUREMENT RUN, NOT ADOPTED: proposed rules switched on: "
                f"{', '.join(rules.enabled_rules())}"
            )
        print(
            "Bars and corporate actions: SIP daily, 2016-01-01 to 2024-12-31. Evidence dated 2025 or later is ignored."
        )
        print(
            "Independent sources: Alpaca corporate actions, OpenFIGI (security type), Wikidata (company records),"
        )
        print(
            "SEC EDGAR (delisting notices, 8-Ks, full-text search, filing headers; read-only), facts read from SEC\n"
            "filings by hand (each re-verified against the cached document every run), corrections checked against prices.\n"
        )

        if facts.get("pool"):
            pool_facts = facts["pool"]
            cands = list(s.execute(select(Candidate).order_by(Candidate.symbol)).scalars())
            print("0. CANDIDATE POOL, SCREEN AND COMPLETENESS")
            print(
                f"  Sources (tickers named, plain tickers only after filtering): {pool_facts['source_sizes']}"
            )
            print(
                f"  Alpaca corporate actions read: {pool_facts['alpaca_events']}; Wikidata listing rows: {pool_facts['wikidata_rows']}"
            )
            print(f"  Candidate pool: {len(cands)} tickers")
            only = Counter()
            for c in cands:
                srcs = c.sources.split(",")
                if len(srcs) == 1:
                    only[srcs[0]] += 1
            print(f"  Tickers named by one source only: {dict(only)}")
            print(
                f"  With at least one month of trades 2016-2024: {sum(1 for c in cands if c.months_with_trades)}"
            )
            sel = Counter(c.selection for c in cands)
            print(
                f"  Daily data fetched for: {sel.get('screen', 0) + sel.get('completeness', 0)} "
                f"(screen, rank <= {universe_stage.MARGIN * RULE.size} in a month 2019-09..2024-11: {sel.get('screen', 0)}; "
                f"added by the completeness proof: {sel.get('completeness', 0)}); not selected: {sel.get('not_selected', 0)}"
            )
            print(
                f"  Completeness rounds (tickers that could have been members, per round): {facts.get('completeness_rounds')}; "
                f"left at the end: {facts.get('completeness_gaps_final')}"
            )
            for c in cands:
                if c.selection == "completeness":
                    print(f"    added {c.symbol}: {c.reason}")
            known = universe_stage.validation_tickers()
            for source in ("sec_ticker_list", "alpaca_corporate_actions", "wikidata_listing"):
                found = sorted(
                    k for k in known if any(c.symbol == k and source in c.sources for c in cands)
                )
                print(f"  Named validation cases found by {source}: {len(found)} of {len(known)}")
            automated = {
                c.symbol for c in cands if set(c.sources.split(",")) - {"validation_batch"}
            }
            missed = sorted(known - automated)
            print(
                f"  Named validation cases found by no automated source: {len(missed)}"
                + (f" ({', '.join(missed)})" if missed else "")
            )
            print(
                "  Each candidate's selection and reason is stored in table candidates (every inclusion and exclusion)."
            )
            print()

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
        sec_cached = _sec_cached(s)
        made = sum(facts.get(f"{k}_requests", 0) for k, _ in SEC_STEPS)
        stored = sum(facts.get(f"{k}_stored", 0) for k, _ in SEC_STEPS)
        failed = sum(len(facts.get(f"{k}_failures", [])) for k, _ in SEC_STEPS)
        print(f"  SEC: {sec_cached} cached responses (requests this run: {made}, failed: {failed})")
        for k, label in SEC_STEPS:
            print(
                f"    {label}: {facts.get(f'{k}_requests', 0)} HTTP responses (retries included), "
                f"{facts.get(f'{k}_stored', 0)} answers stored, "
                f"{len(facts.get(f'{k}_failures', []))} failed"
            )
        start = facts.get("sec_cached_at_start")
        if start is not None:
            grown = sec_cached - start
            if facts.get("refresh"):
                verdict = "not comparable (--refresh replaces cached answers)"
            else:
                verdict = "every new answer accounted for" if grown == stored else "MISMATCH"
            print(
                f"    cache at start {start}, now {sec_cached}: {grown} new; "
                f"answers stored this run {stored}: {verdict}"
            )

        print("\n3. SECURITY TYPES")
        print("  Classification rules (app.universe.sectypes):")
        for rule_text in sectypes.CLASSIFICATION_RULES:
            print(f"    - {rule_text}")
        print(f"  Statuses usable for membership: {', '.join(sorted(sectypes.USABLE_STATUSES))}")
        grid = Counter((x.security_type, x.type_status) for x in securities.values())
        for (kind, status), n in sorted(grid.items()):
            print(f"  {kind:12} {status:14} {n:4}")
        print(
            "  Resolved (sources disagreed; settled by the issuer's own SEC description, outvoted"
            " evidence shown):"
        )
        for sid, x in sorted(securities.items()):
            if x.type_status == "resolved":
                print(
                    f"    id {sid} {path(sid)} -> {x.security_type}: "
                    + "; ".join(
                        f"{e.source}={e.security_type} ({e.detail[:70]})"
                        for e in type_ev[sid]
                        if e.source != "declared"
                    )
                )
        print("  Conflicts (sources disagree, not settled):")
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
        groups = {
            "failure (bankruptcy, bank failure)": ("bankruptcy", "failure"),
            "acquisition": (
                "acquired",
                "acquired_cash",
                "acquired_stock",
                "acquired_stock_and_cash",
            ),
            "redemption (SPAC trust, fund liquidation, called warrants or notes)": ("redeemed",),
            "not an exit (trading gap of the same instrument)": ("trading_gap",),
            "conflict": ("conflict",),
            "unresolved": ("unknown", "resumed_or_fragment"),
        }
        print("  By kind of end:")
        for label, kinds in groups.items():
            rows = [x for x in ended.values() if x.outcome in kinds]
            common = sum(1 for x in rows if x.security_type == "common")
            print(f"    {label:70} {len(rows):3}  (common stock: {common})")
        print("  Basis of the value (the status above describes the event, not the value):")
        for basis, n in sorted(Counter(x.value_basis for x in ended.values()).items()):
            print(f"    {basis:14} {n:3}")
        print(
            "    assumed_zero = event confirmed (bankruptcy or failure), shareholder recovery NOT confirmed;"
        )
        print("    zero is the agreed conservative modelling assumption, to be stress-tested.")
        for status in ("confirmed", "single_source", "conflict", "not_an_exit", "unresolved"):
            rows = sorted((sid, x) for sid, x in ended.items() if x.outcome_status == status)
            print(f"  --- {status} ({len(rows)}) ---")
            for sid, x in rows:
                value = "none" if x.assumed_exit_value is None else x.assumed_exit_value
                print(
                    f"    id {sid} {path(sid)} (last trade {x.last_trade_session}): {x.outcome}, value {value} [{x.value_basis}]"
                    + (
                        "  LOWER CONFIDENCE: one source"
                        if x.outcome_status == "single_source"
                        else ""
                    )
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
        known_issuer = {i.security_id: i for i in s.execute(select(SecurityIssuer)).scalars()}

        def notice_read(f) -> bool:
            cik = str(int(f.cik)).zfill(10)
            return any(str(int(n.issuer_cik or 0)).zfill(10) == cik for _, n in f.notices)

        print(
            f"  Matched: {len(matched)} of {len(sec_found)}; the matched company's Form 25 notice read "
            f"and naming it as the issuer: {sum(map(notice_read, matched.values()))}"
        )
        print(
            "  A match whose notice was not read is a candidate only: it is not taken as the issuer"
        )
        print(
            "  (section 4e has the issuer from dated evidence), but its 8-Ks are still read below."
        )
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
            if not notice_read(f):
                known = known_issuer.get(sid)
                print(
                    "        candidate only: Form 25 notice not read; issuer from dated evidence: "
                    + (
                        f"{known.name} (CIK {known.cik}, {known.how})"
                        + (
                            ", the same company"
                            if known.cik == str(int(f.cik)).zfill(10)
                            else ", a DIFFERENT company"
                        )
                        if known
                        else "none"
                    )
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

        print(
            "\n4c. FACTS READ FROM SEC FILINGS (re-verified: every quoted passage must be in the cached document)"
        )
        check = evidence.sec.CachedSec(s, network=False)
        for label, entries in (
            ("rename", curated.CURATED_RENAMES),
            ("fact", curated.CURATED_FACTS),
            ("deal", curated.CURATED_DEALS),
        ):
            for e in entries:
                missing = evidence.missing_quotes(check, e.quotes)
                name = (
                    f"{e.old_symbol} -> {e.new_symbol} {e.date}"
                    if label == "rename"
                    else f"{getattr(e, 'symbol', getattr(e, 'target_symbol', ''))} (last trade {e.last_trade})"
                )
                print(
                    f"  {label:6} {name:32} quotes found {len(e.quotes) - len(missing)}/{len(e.quotes)}"
                    + ("" if not missing else "  -> NOT USED")
                )
        print(
            "  Sources: "
            + ", ".join(
                sorted(
                    {
                        q.url.split("/data/")[1].split("/")[0]
                        for e in (
                            *curated.CURATED_RENAMES,
                            *curated.CURATED_FACTS,
                            *curated.CURATED_DEALS,
                        )
                        for q in e.quotes
                    }
                )
            )
            + " (CIKs of the filers quoted)"
        )

        print(
            "\n4d. VALUE OF STOCK AND CASH-AND-STOCK ACQUISITIONS (terms quoted from the SEC filing)"
        )
        print(
            "  value = cash + shares x acquirer's raw SIP close on the effective date (first session with a trade)"
        )
        print(f"  used only if within {evidence.DEAL_TOLERANCE:.0%} of the target's last close")
        for d in s.execute(select(DealValue).order_by(DealValue.security_id)).scalars():
            x = securities[d.security_id]
            gap = (
                f"{d.value / d.target_last_close - 1:+.1%}"
                if d.value is not None and d.target_last_close
                else "n/a"
            )
            print(
                f"    id {d.security_id} {path(d.security_id):8} {d.cash_per_share:g} cash + {d.shares_per_share:g} {d.acquirer_symbol} "
                f"@ {d.acquirer_close} ({d.acquirer_session}) = {d.value if d.value is None else round(d.value, 4)}; "
                f"target last close {d.target_last_close} ({gap}); {d.status} -> outcome {x.outcome}, value {x.assumed_exit_value}"
            )

        print(
            "\n4e. ISSUERS AND BLANK-CHECK COMPANIES (SIC 6770, point in time from filing headers)"
        )
        issuers = {i.security_id: i for i in s.execute(select(SecurityIssuer)).scalars()}
        points = evidence.stored_issuer_points(s)
        print(
            "  Issuer of a ticker on a day: the company of the latest dated evidence by then -\n"
            "  its own cover page listing the ticker under 'Trading Symbol(s)' (cover_page, 2019 on),\n"
            "  the delisting notice at the last trade (form25), or a filing naming the ticker as its\n"
            "  own (full_text). SEC's ticker list as of today only proposes candidates."
        )
        print(
            f"  Issuer at the last trade known for {len(issuers)} of {len(securities)} securities, by evidence: "
            f"{dict(sorted(Counter(i.how for i in issuers.values()).items()))}"
        )
        print(
            f"  Dated evidence: {sum(len(v) for v in points.values())} points; by kind "
            f"{dict(sorted(Counter(pt.how for v in points.values() for pt in v).items()))}"
        )
        multi = {sid: v for sid, v in points.items() if len({pt.cik for pt in v}) > 1}
        print(
            f"  Securities whose ticker was held by more than one company over time: {len(multi)}"
        )
        for sid, v in sorted(multi.items()):
            spans = []
            for cik in dict.fromkeys(pt.cik for pt in v):
                mine = [pt for pt in v if pt.cik == cik]
                spans.append(
                    f"CIK {cik} {mine[0].name[:40]} {mine[0].filed}..{mine[-1].filed} ({len(mine)})"
                )
            print(f"    id {sid} {path(sid)}: " + "; ".join(spans))
        conflicts = facts.get("issuer_conflicts") or {}
        print(
            f"  Ticker periods where two companies' cover pages overlapped (the one with more kept): "
            f"{sum(len(v) for v in conflicts.values())}"
        )
        for sid, items in sorted(conflicts.items()):
            for item in items:
                print(f"    id {sid} {path(sid)}: {item}")
        common_ids = [sid for sid, x in securities.items() if x.security_type == "common"]
        print(
            f"  Common stocks with an issuer (so SIC can be tested): {sum(1 for sid in common_ids if sid in issuers)} of {len(common_ids)}"
        )
        print("  Securities with no identified SEC issuer, and why:")
        unexplained_issuers = 0
        for sid in sorted(set(securities) - set(issuers)):
            x = securities[sid]
            reason = curated.CURATED_NO_SEC_ISSUER.get((x.last_symbol, x.last_trade_session))
            if reason is None and x.security_type != "common":
                reason = (
                    f"{x.security_type}, not common stock: outside the universe whatever its issuer"
                )
            if reason is None and x.last_trade_session < p.MEMBERSHIP_FIRST_DECISION:
                reason = (
                    f"traded only before the first membership decision ({p.MEMBERSHIP_FIRST_DECISION}): "
                    "no membership decision reads its SIC"
                )
            if reason is None:
                unexplained_issuers += 1
                reason = "UNEXPLAINED: a blank-check company here would not be caught"
            print(f"    id {sid} {path(sid)} ({x.status}): {reason}")
        print(
            f"  Common stocks without an issuer and without a documented reason: {unexplained_issuers}"
        )
        later_identity = []
        for m in s.execute(
            select(Membership).where(
                (Membership.eligible == 1) | (Membership.reason == "blank_check_company")
            )
        ).scalars():
            if m.security_id in points:
                held, later = evidence.issuer_on(points[m.security_id], m.decision_session)
                if later:
                    later_identity.append((m.security_id, m.decision_session, held))
        print(
            "  Decisions (eligible or excluded as blank-check) whose issuer rests on a document filed\n"
            f"  after the decision (no earlier evidence; the SIC used is still only what was on file): {len(later_identity)}"
        )
        for sid, day, held in later_identity:
            print(
                f"    id {sid} {path(sid)} on {day}: CIK {held.cik} from {held.how} filed {held.filed}"
            )
        sics = defaultdict(list)
        for r in s.execute(select(IssuerSic).order_by(IssuerSic.as_of)).scalars():
            sics[(r.security_id, r.cik)].append((r.as_of, r.sic))
        print("  Issuers with SIC 6770 (blank-check) at some point, SIC by filing date:")
        for sid, cik in sorted(sics):
            if any(sic == "6770" for _, sic in sics[(sid, cik)]):
                print(
                    f"    id {sid} {path(sid)} (CIK {cik}): "
                    + ", ".join(f"{sic} from {day}" for day, sic in sics[(sid, cik)])
                )

        print("\n4f. TRADING-GAP MERGES (same ticker after a pause: one security or two?)")
        print(
            f"  Passes until stable: {facts.get('merge_passes')}; stable: {facts.get('merges_stable')}"
        )
        links = list(
            s.execute(
                select(SegmentLink).order_by(SegmentLink.symbol, SegmentLink.earlier_last)
            ).scalars()
        )
        decisions = Counter(r.decision for r in links)
        print(
            f"  Same-ticker candidates: {decisions['merged'] + decisions['kept_separate']}; "
            f"merged {decisions['merged']}, kept separate {decisions['kept_separate']}"
        )
        print(
            "  Renames confirmed by the issuer (consecutive securities, different tickers, same "
            f"registrant on its cover pages): renamed {decisions['renamed']}, "
            f"rejected {decisions['rename_rejected']}"
        )
        for r in links:
            print(
                f"    {r.decision:13} {r.symbol:6} {r.earlier_last} -> {r.later_first} ({r.gap_sessions} sessions): {r.reason[:170]}"
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
            f"\n6. POINT-IN-TIME MEMBERSHIP (rule: top {RULE.size} eligible common stocks "
            "of the batch)"
        )
        print(f"  Rule: {RULE.to_dict()}  fingerprint {RULE.fingerprint()[:16]}")
        if rules.dated_types_enabled():
            _print_dated_types(s)
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
        print(
            "    (the filter uses raw prices, as traded on the day; split-adjusted prices would embed later splits)"
        )

        print("\n7. CROSS-CHECKS OF MERGER VALUES AND CORPORATE ACTIONS")
        cash_pairs = []
        for sid in sorted(ended):
            alp = [
                e for e in exit_ev[sid] if e.source == "alpaca" and e.category == "acquired_cash"
            ]
            sec_cash = [
                e for e in exit_ev[sid] if e.source == "sec" and e.category == "acquired_cash"
            ]
            if alp and sec_cash:
                cash_pairs.append((sid, alp[0].detail, sec_cash[0].detail))
        print(f"  Cash deals stated by both Alpaca and the SEC 8-K: {len(cash_pairs)}")
        agree = 0
        for sid, a, b in cash_pairs:
            first = re.search(r"([0-9]+(?:\.[0-9]+)?) cash per share", a)
            second = re.search(r"cash per share ([0-9]+(?:\.[0-9]+)?)", b)
            same = bool(first and second) and float(first.group(1)) == float(
                second.group(1).replace(",", "")
            )
            agree += same
            print(
                f"    id {sid} {path(sid)}: Alpaca {first.group(1) if first else '?'}, SEC 8-K "
                f"{second.group(1) if second else '?'}" + ("" if same else "  <- DIFFER")
            )
        print(f"    agreeing on the amount: {agree} of {len(cash_pairs)}")
        notices = {}
        for sid, f in sec_found.items():
            for filing, n in f.notices:
                if "(a)(3)" in n.rule:
                    notices.setdefault(sid, []).append(filing.filed)
        for d in s.execute(select(DealValue).order_by(DealValue.security_id)).scalars():
            near = notices.get(d.security_id, [])
            off = min((abs((day - d.effective_date).days) for day in near), default=None)
            print(
                f"  Deal id {d.security_id} {path(d.security_id)}: effective {d.effective_date}, merger delisting notice "
                + (f"{off} day(s) away" if off is not None else "not found")
                + ("" if off is not None and off <= 4 else "  <- CHECK")
            )
        for r in curated.CURATED_RENAMES:
            ids = [
                sid
                for sid in history
                if [h.symbol for h in history[sid]][:2] == [r.old_symbol, r.new_symbol]
                or any(
                    a.symbol == r.old_symbol and b.symbol == r.new_symbol
                    for a, b in zip(history[sid], history[sid][1:], strict=False)
                )
            ]
            if not ids:
                present = any(
                    h.symbol in (r.old_symbol, r.new_symbol) for hs in history.values() for h in hs
                )
                print(
                    f"  Curated rename {r.old_symbol} -> {r.new_symbol}: NOT linked"
                    if present
                    else f"  Curated rename {r.old_symbol} -> {r.new_symbol}: neither ticker is "
                    "in this batch (nothing to link)"
                )
                continue
            sid = ids[0]
            hs = history[sid]
            k = next(
                i
                for i in range(len(hs) - 1)
                if hs[i].symbol == r.old_symbol and hs[i + 1].symbol == r.new_symbol
            )
            from app.universe.models import SecurityBar as _SB

            def adjusted(day, sid=sid):
                bar = s.scalar(select(_SB).where(_SB.security_id == sid, _SB.session == day))
                return (
                    None
                    if bar is None
                    else bar.close / (bar.split_factor * bar.distribution_factor)
                )

            before, after = adjusted(hs[k].valid_to), adjusted(hs[k + 1].valid_from)
            print(
                f"  Curated rename {r.old_symbol} -> {r.new_symbol}: linked in id {sid} ({path(sid)}); "
                f"split-adjusted last close {before:.4g} on {hs[k].valid_to}, first close {after:.4g} on {hs[k + 1].valid_from}"
                + (f" ({after / before - 1:+.1%})" if before and after else "")
            )

        print("\n8. SURVIVORSHIP AND LOOK-AHEAD AUDIT")
        print("  Survivorship:")
        print(
            f"    Batch built from: {dict(Counter(b.selection.split(':')[0] for b in batch))} "
            "(historical corporate-action events and named cases; not today's ticker list)"
        )
        print(
            f"    Securities: {len(securities)}, of which ended before 2025: {len(ended)} "
            f"({sum(1 for x in ended.values() if x.security_type == 'common')} common stocks)"
        )
        ended_members = sorted(sid for sid in ever if securities[sid].status == "ended")
        print(f"    Members that later ended: {len(ended_members)} of {len(ever)} distinct members")
        print(
            f"      their exit value basis: {dict(Counter(securities[sid].value_basis for sid in ended_members))}"
        )
        print(
            f"      their outcome status: {dict(Counter(securities[sid].outcome_status for sid in ended_members))}"
        )
        no_value = [sid for sid in ended_members if securities[sid].assumed_exit_value is None]
        print(
            "      members with no exit value: "
            + (
                ", ".join(f"id {sid} {path(sid)} ({securities[sid].outcome})" for sid in no_value)
                or "none"
            )
        )
        dropped = [
            sid
            for sid, x in ended.items()
            if x.security_type == "unknown" or x.type_status in ("unknown", "conflict")
        ]
        print(
            f"    Ended securities kept out only because their type is unknown: {len(dropped)}: "
            + ", ".join(f"id {sid} {path(sid)}" for sid in sorted(dropped))
        )
        blank = Counter(
            r.reason
            for r in s.execute(
                select(Membership).where(Membership.reason == "blank_check_company")
            ).scalars()
        )
        print(f"    Decisions excluded as blank-check company: {sum(blank.values())}")
        print("  Look-ahead:")
        print(
            f"    Recompute with data cut at each decision: {len(decisions) - len(differ)}/{len(decisions)} identical; "
            f"future prices x10: earlier decisions unchanged"
        )
        print(f"    Members chosen after their last trade: {len(after_exit)}")
        late = {
            "security bars": s.scalar(select(func.max(SecurityBar.session))),
            "exit evidence": s.scalar(select(func.max(ExitEvidence.event_date))),
            "SIC filings": s.scalar(select(func.max(IssuerSic.as_of))),
            "issuer evidence": s.scalar(select(func.max(IssuerEvidence.filed))),
            "deal effective dates": s.scalar(select(func.max(DealValue.effective_date))),
            "acquirer price sessions": s.scalar(select(func.max(DealValue.acquirer_session))),
        }
        print(
            "    Latest date in: "
            + "; ".join(f"{k} {v}" for k, v in late.items())
            + f"  -> all before 2025: {all(v is None or v < p.DATA_END.replace(year=2025, month=1, day=1) for v in late.values())}"
        )
        later_sic = 0
        dated_ex = evidence.blank_check_exclusions(s)
        rows_sic = defaultdict(list)
        for r in s.execute(select(IssuerSic)).scalars():
            rows_sic[(r.security_id, r.cik)].append((r.as_of, r.sic))
        held_points = evidence.stored_issuer_points(s)
        for m in s.execute(
            select(Membership).where(Membership.reason == "blank_check_company")
        ).scalars():
            # Re-derived from the issuer of the day and from the SIC rows and ticker changes
            # seen by the decision date only.
            held, _ = evidence.issuer_on(held_points.get(m.security_id, []), m.decision_session)
            known = [
                (day, sic)
                for day, sic in sorted(rows_sic[(m.security_id, held.cik if held else "")])
                if day <= m.decision_session
            ]
            seen = [
                h.valid_from
                for h in history[m.security_id][1:]
                if h.valid_from <= m.decision_session
            ]
            later_sic += not evidence.blank_check_on(known, seen, m.decision_session)
        print(
            f"    Blank-check exclusions not supported by a SIC on file by the decision date: {later_sic} "
            f"(of {sum(blank.values())}; {len(dated_ex)} securities have SIC history)"
        )
        print(
            "    Membership reads: raw daily bars up to the decision, the security type, held-out split checks and"
        )
        print(
            "    SIC filed by then. It never reads exit outcomes, exit values or any price after the decision."
        )
        print(
            "    Known limitation: the security type comes from present-day sources and applies to the whole history"
        )
        print(
            "    (a type is not a price or an outcome, but a later reclassification would be applied retroactively)."
        )
        print(
            "    Trading-gap merges use the fact that trading resumed (later data) to decide identity only; a merged"
        )
        print(
            "    security's bars up to any date are the same as before the merge, and the recompute tests above hold."
        )

        print("\n8b. PERMANENT SECURITY IDS")
        reg = list(
            s.execute(select(SecurityRegistry).order_by(SecurityRegistry.security_id)).scalars()
        )
        print(
            f"  Registered identities: {len(reg)}; by status: {dict(Counter(r.status for r in reg))}"
        )
        active_ids = {r.security_id for r in reg if r.status == "active"}
        print(
            f"  Every security in this run has a registered id: {set(securities) <= active_ids}; "
            f"registered active ids not in this run: {len(active_ids - set(securities))}"
        )
        for r in reg:
            if r.status == "merged":
                print(
                    f"    id {r.security_id} ({r.first_symbol} from {r.first_session}) is now part of id {r.merged_into}"
                )

        print("\n9. DUPLICATE-SECURITY AUDIT")
        owners = defaultdict(set)
        for bar in s.execute(
            select(SecurityBar.symbol, SecurityBar.session, SecurityBar.security_id)
        ).all():
            owners[(bar[0], bar[1])].add(bar[2])
        overlaps = {k: v for k, v in owners.items() if len(v) > 1}
        print(f"  Ticker-days held by more than one security: {len(overlaps)}")
        link_keys = {(r.symbol, r.earlier_last, r.later_first) for r in links}
        successors = []
        for sid, items in history.items():
            tail = items[-1]
            later = sorted(
                (h.valid_from, osid)
                for osid in history
                for h in history[osid]
                if osid != sid and h.symbol == tail.symbol and h.valid_from > tail.valid_to
            )
            if later:
                successors.append((tail.symbol, tail.valid_to, later[0][0], sid, later[0][1]))
        undocumented = [x for x in successors if (x[0], x[1], x[2]) not in link_keys]
        print(
            f"  Same-ticker successions between two securities: {len(successors)}; each with a recorded decision: "
            f"{len(successors) - len(undocumented)}"
            + "".join(
                f"\n    UNDOCUMENTED {a} {b} -> {c} (id {d} -> id {e})"
                for a, b, c, d, e in undocumented
            )
        )
        kept = [r for r in links if r.decision == "kept_separate"]
        print(
            f"  Kept separate, with the reason (genuinely different listings): {len(kept)} (listed in 4f)"
        )
        by_cik = defaultdict(list)
        for i in issuers.values():
            by_cik[i.cik].append(i.security_id)
        suspects = []
        for cik, ids in by_cik.items():
            ids = sorted(ids, key=lambda k: securities[k].first_session)
            for a, b in pairwise(ids):
                xa, xb = securities[a], securities[b]
                if (
                    xa.security_type == xb.security_type
                    and xb.first_session > xa.last_trade_session
                ):
                    gap = len(trading_days(xa.last_trade_session, xb.first_session)) - 2
                    suspects.append((cik, a, b, gap))
        kept_by_pair = {
            (r.symbol, r.earlier_last, r.later_first): r
            for r in links
            if r.decision in ("kept_separate", "rename_rejected")
        }
        unexplained_pairs = 0
        print(
            f"  Same issuer and same type, one after the other (could be one security): {len(suspects)}"
        )
        for c, a, b, g in suspects:
            link = kept_by_pair.get(
                (history[a][-1].symbol, history[a][-1].valid_to, history[b][0].valid_from)
            )
            unexplained_pairs += link is None
            print(
                f"    CIK {c}: id {a} {path(a)} (to {securities[a].last_trade_session}) and id {b} {path(b)} "
                f"(from {securities[b].first_session}), {g} sessions apart: "
                + (f"kept separate because {link.reason[:150]}" if link else "UNEXPLAINED")
            )
        print(f"  Possible duplicates without a recorded reason: {unexplained_pairs}")
        identity = evidence.curated_identity_links(s)
        print(
            f"  Documented identity exceptions (one security across a suspension longer than "
            f"{evidence.TRADING_GAP_MAX_SESSIONS} sessions; checked by scripts/audit_merges.py): {len(identity)}"
        )
        for link, missing in identity:
            link_id = evidence.identity_link_id(link)
            joined = [
                sid for sid, items in history.items() if any(h.linked_by == link_id for h in items)
            ]
            gap = len(trading_days(link.old_last, link.new_first)) - 2
            print(
                f"    {link.old_symbol} (last trade {link.old_last}) -> {link.new_symbol} (first trade "
                f"{link.new_first}), CIK {link.cik}, {gap} sessions without a trade; quotes found "
                f"{len(link.quotes) - len(missing)} of {len(link.quotes)}; "
                + (
                    f"applied: id {joined[0]} {path(joined[0])}"
                    if joined
                    else "NOT APPLIED (quotes missing or segments not found)"
                )
            )
            print(f"      {link.note}")
        handovers = facts.get("handovers") or []
        points = evidence.stored_issuer_points(s)

        def issuer_text(sid, day):
            point, later = evidence.issuer_on(points.get(sid, []), day)
            if point is None:
                return "no issuer evidence"
            return f"CIK {point.cik} {point.name[:30]}" + (" (later document)" if later else "")

        print(
            f"  Tickers that passed to another company without a pause, cut into one security "
            f"per company (checked by scripts/audit_merges.py): {len(handovers)}"
        )
        for h in handovers:
            before = next(
                (
                    sid
                    for sid, items in history.items()
                    if any(
                        t.symbol == h.symbol
                        and t.valid_to < h.cut
                        and seg.sessions_between(t.valid_to, h.cut) <= seg.MAX_GAP_SESSIONS
                        for t in items
                    )
                ),
                None,
            )
            after = next(
                (
                    sid
                    for sid, items in history.items()
                    if any(t.symbol == h.symbol and t.valid_from == h.cut for t in items)
                ),
                None,
            )
            how = (
                f"rename {h.renamed_from} -> {h.symbol}"
                if h.renamed_from
                else "documented exception"
            )
            print(
                f"    {h.symbol} cut at {h.cut} ({how}); earlier holder's exit: {h.exit_note[:120]}"
            )
            print(
                "      earlier holder: "
                + (
                    f"id {before} {path(before)}, "
                    f"{issuer_text(before, h.cut - timedelta(days=1))} before the cut; "
                    f"{securities[before].status}"
                    + (
                        f", outcome {securities[before].outcome} "
                        f"({securities[before].outcome_status}), value "
                        f"{securities[before].assumed_exit_value}"
                        if securities[before].status == "ended"
                        else ""
                    )
                    if before is not None
                    else "NOT FOUND"
                )
            )
            print(
                "      new holder: "
                + (
                    f"id {after} {path(after)}, {issuer_text(after, h.cut)} at the cut"
                    if after is not None
                    else "NOT FOUND"
                )
            )

        print("\n10. EXIT-VALUE AUDIT (no value without evidence)")
        deals = {d.security_id: d for d in s.execute(select(DealValue)).scalars()}
        problems = []

        def numbers(text):
            return {float(n) for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text or "")}

        for sid, x in sorted(ended.items()):
            v = x.assumed_exit_value
            if v is None:
                if x.value_basis not in (None, "none"):
                    problems.append((sid, "no value but a basis"))
                continue
            if x.value_basis == "assumed_zero":
                if v != 0.0 or not any(
                    e.category in ("bankruptcy", "failure") for e in exit_ev[sid]
                ):
                    problems.append((sid, "assumed zero without a bankruptcy or failure record"))
            elif x.value_basis == "deal_terms":
                d = deals.get(sid)
                if (
                    d is None
                    or d.status != "verified"
                    or d.value is None
                    or abs(d.value - v) > 1e-9
                ):
                    problems.append((sid, "deal value without verified terms"))
            elif x.value_basis == "stated_cash":
                if not any(abs(n - v) < 1e-9 for e in exit_ev[sid] for n in numbers(e.detail)):
                    problems.append((sid, f"cash {v} not stated in any evidence"))
            else:
                problems.append((sid, f"unknown basis {x.value_basis}"))
        valued = [x for x in ended.values() if x.assumed_exit_value is not None]
        print(
            f"  Ended securities with a value: {len(valued)}; each traced to its evidence: {len(valued) - len(problems)}"
        )
        for sid, why in problems:
            print(f"    PROBLEM id {sid} {path(sid)}: {why}")
        print(
            f"  Values by basis: {dict(Counter(x.value_basis for x in valued))}; assumed zero is a modelling assumption,"
        )
        print(
            "  not an observed recovery. No value is assumed for: "
            + ", ".join(
                f"{path(sid)} ({x.outcome})"
                for sid, x in sorted(ended.items())
                if x.assumed_exit_value is None and x.outcome != "trading_gap"
            )
        )

        print("\n11. CONFIDENCE OF OUTCOMES")
        conf = Counter(x.outcome_status for x in ended.values())
        print(
            f"  Two independent sources: {conf.get('confirmed', 0)}; LOWER CONFIDENCE, one source: {conf.get('single_source', 0)}; "
            f"not an exit: {conf.get('not_an_exit', 0)}; unresolved: {conf.get('unresolved', 0)}; conflict: {conf.get('conflict', 0)}"
        )
        one = sorted(sid for sid, x in ended.items() if x.outcome_status == "single_source")
        print(
            "  One-source outcomes: "
            + ", ".join(f"{path(sid)} ({securities[sid].outcome})" for sid in one)
        )
        print(
            "  Members that later ended, by confidence: "
            f"{dict(Counter(securities[sid].outcome_status for sid in ended_members))}"
        )
    engine.dispose()
    return out.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/universe_validation_batch.py")
    parser.add_argument("--report", type=Path, help="also write the report to this file")
    parser.add_argument(
        "--refresh", action="store_true", help="ask OpenFIGI, Wikidata and SEC again"
    )
    parser.add_argument(
        "--stage",
        type=int,
        help="build the staged universe of this size (top N per month) in its own database",
    )
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    global RULE
    if args.stage:
        RULE = MembershipRule(size=args.stage)
    text = report(run(get_settings(), args.refresh, args.stage))
    print(text)
    if args.report:
        args.report.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
