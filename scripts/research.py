#!/usr/bin/env python3
"""Strategy research CLI.

    python scripts/research.py report                  the validation report
    python scripts/research.py distribution            what the scorer produces
    python scripts/research.py calibration             does the score predict?
    python scripts/research.py funnel                  where candidates die
    python scripts/research.py thresholds  <symbol>    the threshold ladder
    python scripts/research.py ablate      <symbol>    which factors earn their weight
    python scripts/research.py sweep       <symbol> <param> <v1,v2,...>

    python scripts/research.py readiness               how much data is still needed
    python scripts/research.py evidence                is there enough evidence yet?
    python scripts/research.py shadow                  champion vs challengers, paired
    python scripts/research.py integrity               observations that must not be counted
    python scripts/research.py preflight               can this machine collect at all?
    python scripts/research.py resolve   [symbols...]  which mint does a symbol mean?
    python scripts/research.py collection              is the paper run collecting cleanly?
    python scripts/research.py counterfactual          what the filters rejected, and its worth
    python scripts/research.py degradation             has recent behaviour drifted from baseline?
    python scripts/research.py diagnose                triage the recorded data
    python scripts/research.py changelog               what autopilot changed, and why
    python scripts/research.py replay                  thresholds on YOUR recorded history
    python scripts/research.py postmortem              per-trade autopsies
    python scripts/research.py early                   the Early Signal Engine
    python scripts/research.py early-ablate            which early factors earn it
    python scripts/research.py early-walkforward       is the early threshold stable?
    python scripts/research.py modes       <symbol>    A vs B vs C

The database-backed commands read the bot's own database and need no
network. The backtest commands need candle history for the symbol - by default
from the live provider, or with --synthetic from the generator (useful for
exercising the machinery, useless for drawing conclusions about markets).

Everything here is read-only. Running it never places, cancels or modifies
anything.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.calibration import HORIZONS_MINUTES, build_calibration  # noqa: E402
from app.analysis.forward_returns import coverage  # noqa: E402
from app.analysis.research_report import build_research_report  # noqa: E402
from app.analysis.score_distribution import build_score_distribution  # noqa: E402
from app.analysis.stage_funnel import build_stage_funnel  # noqa: E402
from app.backtesting.types import BacktestConfig  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.pipeline import MARKET_QUALITY, SECURITY, TECHNICAL_SCORE  # noqa: E402

RULE = "=" * 78


def _series(symbol: str, *, synthetic: bool, limit: int):
    """Candle history for a backtest-driven command."""
    from app.data.candles import Timeframe

    if synthetic:
        from app.data.providers import SyntheticCandleProvider

        print(
            "  Using SYNTHETIC candles. This exercises the machinery and says NOTHING\n"
            "  about real markets - do not read any number below as a market result.\n"
        )
        return SyntheticCandleProvider(regime="bull", seed=7).fetch(symbol, Timeframe.M15, limit=limit)

    import asyncio

    from app.data.live_provider import fetch_live_series

    series = asyncio.run(fetch_live_series("solana", symbol, Timeframe.M15, limit=limit))
    if series is None or not len(series):
        print(
            f"  No live candle history for {symbol}. Either the token address is wrong, the\n"
            "  provider is unreachable, or the pool is too new. Nothing can be researched\n"
            "  without history - use --synthetic only to check the machinery runs."
        )
        return None
    return series


def cmd_report(args) -> int:
    db = SessionLocal()
    try:
        print(build_research_report(db, strategy_version=args.version).render())
    finally:
        db.close()
    return 0


def cmd_distribution(args) -> int:
    db = SessionLocal()
    try:
        for stage in (TECHNICAL_SCORE, MARKET_QUALITY, SECURITY):
            dist = build_score_distribution(db, stage=stage, include_unreliable=args.include_unreliable)
            print(dist.summary())
            if dist.histogram:
                print("  histogram:")
                widest = max(c for _b, c in dist.histogram) or 1
                for bucket, count in dist.histogram:
                    bar = "#" * int(40 * count / widest)
                    print(f"    {bucket:>8}  {count:>5}  {bar}")
            print()
    finally:
        db.close()
    return 0


def cmd_calibration(args) -> int:
    db = SessionLocal()
    try:
        stats = coverage(db)
        print(RULE)
        print(" SCORE CALIBRATION - does a higher score precede a better outcome?")
        print(RULE)
        print(
            f" dataset: {stats['resolved']} resolved / {stats['pending']} pending / "
            f"{stats['unmeasurable']} unmeasurable  ({stats['coverage_pct']}% coverage)"
        )
        if stats["resolved"] == 0:
            print("\n No forward returns resolved yet - nothing to calibrate against.")
            print(" This accumulates on its own while the bot runs.")
            return 0
        print()
        for horizon in HORIZONS_MINUTES:
            table = build_calibration(db, horizon_minutes=horizon)
            print(f" {horizon}m: {table.verdict()}")
            error = table.calibration_error_pct
            if error is not None:
                rho = (
                    f"{table.rank_correlation:+.3f}"
                    if table.rank_correlation is not None else "n/a"
                )
                print(
                    f"    calibration error {error:>6.2f} pts   spread "
                    f"{table.spread_pct:>6.2f} pts   rank corr {rho:>7}   "
                    f"[{table.calibration_grade()}]"
                )
            usable = [b for b in table.buckets if b.sample_size]
            if usable:
                print(f"    {'bucket':<10}{'n':>6}{'mean %':>10}{'median %':>10}"
                      f"{'win %':>8}{'net %':>10}")
                for b in usable:
                    mark = " " if b.meaningful else "*"
                    print(
                        f"  {mark} {b.bucket:<10}{b.sample_size:>6}"
                        f"{b.mean_return_pct:>+10.2f}{b.median_return_pct:>+10.2f}"
                        f"{b.win_rate_pct:>8.0f}{b.mean_net_of_costs_pct:>+10.2f}"
                    )
            print()
        print(" * fewer than 30 measured outcomes - shown, but not evidence")
    finally:
        db.close()
    return 0


def cmd_funnel(args) -> int:
    db = SessionLocal()
    try:
        funnel = build_stage_funnel(db, window_hours=args.hours)
        print(RULE)
        print(f" SCANNER FUNNEL  (last {args.hours or 'all'} hours)")
        print(RULE)
        print(f" {funnel.explain()}")
        print()
        print(f"  {'stage':<18}{'entered':>9}{'passed':>9}{'rejected':>10}{'pass rate':>11}")
        for stage in funnel.stages:
            rate = f"{stage.pass_rate * 100:.1f}%" if stage.pass_rate is not None else "no data"
            print(f"  {stage.stage:<18}{stage.entered:>9}{stage.passed:>9}"
                  f"{stage.rejected:>10}{rate:>11}")

        prescreen = funnel.prescreen
        if prescreen.evaluated:
            print(f"\n  pre-screen breakdown over {prescreen.evaluated} tokens:")
            print(f"    {'check':<16}{'passed':>8}{'failed':>8}{'pass rate':>11}")
            for entry in prescreen.as_dict()["checks"]:
                rate = f"{entry['pass_rate_pct']:.1f}%" if entry["pass_rate_pct"] is not None else "-"
                print(f"    {entry['name']:<16}{entry['passed']:>8}{entry['failed']:>8}{rate:>11}")

        if funnel.rejection_reasons:
            print("\n  top rejection reasons:")
            for stage, reason, count in funnel.rejection_reasons[:12]:
                print(f"    {count:>5}  [{stage}] {reason}")
    finally:
        db.close()
    return 0


def cmd_thresholds(args) -> int:
    from app.research.thresholds import study_thresholds

    series = _series(args.symbol, synthetic=args.synthetic, limit=args.limit)
    if series is None:
        return 1
    print(RULE)
    print(" THRESHOLD STUDY - MIN_SIGNAL_SCORE_TO_ENTER")
    print(RULE)
    study = study_thresholds(series, base_config=BacktestConfig(warmup_bars=args.warmup))
    print(study.table())
    if args.json:
        print(json.dumps(study.as_dict(), indent=2))
    return 0


def cmd_ablate(args) -> int:
    from app.research.ablation import run_ablation

    series = _series(args.symbol, synthetic=args.synthetic, limit=args.limit)
    if series is None:
        return 1
    print(RULE)
    print(" FEATURE ABLATION - does each scoring factor earn its weight?")
    print(RULE)
    report = run_ablation(series, base_config=BacktestConfig(warmup_bars=args.warmup))
    print(report.summary())
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_sweep(args) -> int:
    from app.research.robustness import sweep_parameter

    series = _series(args.symbol, synthetic=args.synthetic, limit=args.limit)
    if series is None:
        return 1
    values = [float(v) for v in args.values.split(",")]
    print(RULE)
    print(f" ROBUSTNESS SWEEP - {args.param}")
    print(RULE)
    report = sweep_parameter(
        series, parameter=args.param, values=values,
        base_config=BacktestConfig(warmup_bars=args.warmup),
    )
    print(report.table())
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_replay(args) -> int:
    from app.research.replay import replay_thresholds

    db = SessionLocal()
    try:
        print(RULE)
        print(" THRESHOLD REPLAY - re-scoring the candidates this bot actually saw")
        print(RULE)
        thresholds = tuple(float(v) for v in args.thresholds.split(","))
        report = replay_thresholds(
            db, horizon_minutes=args.horizon, thresholds=thresholds,
            strategy_version=args.version,
        )
        print(report.table())
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
    finally:
        db.close()
    return 0


def cmd_postmortem(args) -> int:
    from app.analysis.postmortem import recent_postmortems

    db = SessionLocal()
    try:
        print(RULE)
        print(" TRADE POST-MORTEMS")
        print(RULE)
        reports = recent_postmortems(db, limit=args.limit)
        if not reports:
            print(" No closed positions yet.")
            return 0
        for pm in reports:
            print(f"  {pm.headline()}")
            if args.verbose:
                d = pm.as_dict()
                for key in ("entry_price", "exit_price", "fees_usd", "execution_cost_pct",
                            "capture", "liquidity_drop_pct", "signal_score", "samples"):
                    print(f"      {key:<22}{d[key]}")
                print()
        gave_back = sum(1 for p in reports if p.gave_back_a_winner)
        survived = sum(1 for p in reports if p.survived_a_drawdown)
        print(f"\n  {len(reports)} closed | {gave_back} gave back a 20%+ winner "
              f"| {survived} closed green after a 20%+ drawdown")
        print("  MFE/MAE come from polled prices, so both are LOWER bounds on the true path.")
        if args.json:
            print(json.dumps([p.as_dict() for p in reports], indent=2))
    finally:
        db.close()
    return 0


def _pct(value: float | None, *, signed: bool = False) -> str:
    """Format a percentage, or say plainly that there isn't one.

    A missing expectancy prints as "n/a", never as 0.000% - a strategy that
    has not entered anything has no per-trade number, and showing zero
    would read as "it broke even".
    """
    if value is None:
        return "     n/a"
    return f"{value:>+8.3f}%" if signed else f"{value:>8.3f}%"


def cmd_shadow(args) -> int:
    from app.autopilot.promote import evaluate
    from app.shadow.challengers import CHAMPION_ID
    from app.shadow.compare import compare_all
    from app.shadow.resolver import coverage
    from app.strategy.version import current_label

    db = SessionLocal()
    try:
        print(RULE)
        print(" SHADOW CHALLENGERS - paired comparison")
        print(RULE)
        print(f" Champion:            {CHAMPION_ID} ({current_label()})")

        # Printed first, because every number below is conditional on it.
        # A comparison drawn from mostly-unresolved positions describes
        # whichever tokens kept trading long enough to be measured.
        c = coverage(db)
        print(
            f" Outcome coverage:    {c['resolved']}/{c['positions']} resolved "
            f"({c['resolved_pct']}%), {c['open']} still open, "
            f"{c['unmeasurable']} unmeasurable"
        )

        comparisons = compare_all(db)
        if not comparisons:
            print(" Challengers:         none have recorded a decision")
            print("\n INSUFFICIENT_DATA - configure SHADOW_CHALLENGERS and let the bot run.")
            return 0

        for result in comparisons:
            d = result.as_dict()
            print(f"\n Challenger:          {d['challenger_id']}")
            print(f" Paired opportunities:{d['paired']:>8}")
            print(f" Both entered:        {d['both_entered']:>8}")
            print(f" Both rejected:       {d['both_rejected']:>8}")
            print(f" Champion-only:       {d['champion_only']:>8}")
            print(f" Challenger-only:     {d['challenger_only']:>8}")
            print(f" Unresolved entries:  {d['unresolved']:>8}")

            if not result.conclusive or d["difference_pct"] is None:
                print(f"\n {result.verdict()}")
                print(" Promotion gate:      INSUFFICIENT_DATA - not submitted")
                continue

            print("\n Per OPPORTUNITY (declines count as 0%, paired - the gate reads this)")
            print(f"   Champion:          {d['champion_expectancy_pct']:>8.3f}%")
            print(f"   Challenger:        {d['challenger_expectancy_pct']:>8.3f}%")
            print(f"   Difference:        {d['difference_pct']:>+8.3f}%")

            print("\n Per ENTERED TRADE (self-selected samples - reported, not promoted on)")
            print(
                f"   Champion:          {_pct(d['champion_trade_expectancy_pct'])}"
                f"  over {d['champion_trades']} entries"
            )
            print(
                f"   Challenger:        {_pct(d['challenger_trade_expectancy_pct'])}"
                f"  over {d['challenger_trades']} entries"
            )
            print(f"   Difference:        {_pct(d['trade_difference_pct'], signed=True)}")

            champion_arm, challenger_arm = result.arms()
            verdict = evaluate(champion_arm, challenger_arm, attempts=len(comparisons))
            print(f"\n Promotion gate:      {verdict.outcome}")
            print(f"   {verdict.reason()}")
            for bar in verdict.bars:
                print(f"   [{'PASS' if bar.passed else 'FAIL'}] {bar.name:<14} {bar.detail}")

        print(
            "\n Paired means both strategies evaluated the SAME opportunity. Unpaired\n"
            " observations are excluded: a larger unpaired sample measures the wrong\n"
            " thing more precisely.\n\n"
            " The two expectancies answer different questions. Per-opportunity asks\n"
            " what a strategy makes per chance that comes past, and both arms share\n"
            " the same denominator, so it is a paired contrast. Per-trade asks how\n"
            " good the trades are when it does trade, over the entries it chose for\n"
            " itself - informative, but not a controlled comparison, which is why the\n"
            " promotion gate never sees it."
        )
        if args.json:
            print(json.dumps([r.as_dict() for r in comparisons], indent=2))
    finally:
        db.close()
    return 0


def cmd_evidence(args) -> int:
    from app.analysis.evidence import build_evidence_report

    db = SessionLocal()
    try:
        report = build_evidence_report(db, horizon_minutes=args.horizon)
        print(report.render())
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
    finally:
        db.close()
    return 0


def cmd_integrity(args) -> int:
    from app.analysis.integrity import check_all

    db = SessionLocal()
    try:
        print(RULE)
        print(" DATA INTEGRITY")
        print(RULE)
        for name, report in check_all(db).items():
            print(f"\n {name}")
            print(report.render())
        if args.json:
            print(json.dumps(
                {k: v.as_dict() for k, v in check_all(db).items()}, indent=2
            ))
    finally:
        db.close()
    return 0


def cmd_preflight(args) -> int:
    from app.analysis.preflight import environment_summary, run

    print(RULE)
    print(" PREFLIGHT - can this machine actually run a collection?")
    print(RULE)
    for key, value in environment_summary().items():
        print(f"   {key:<14} {value}")
    if args.no_probe:
        print("\n   (--no-probe: upstream APIs not contacted)")
    print()

    report = run(probe_upstreams=not args.no_probe)
    for check in report.checks:
        tag = "" if check.fatal else "  (non-fatal)"
        print(f" [{check.status:<4}] {check.name}{tag}")
        print(f"        {check.detail}")

    print(f"\n {report.verdict()}")
    if report.blocking:
        print(
            "\n The bot degrades quietly by design - a missing price must never take down the\n"
            " position monitor. The cost is that a completely non-functional deployment looks\n"
            " exactly like a quiet market, which is what this command exists to tell apart."
        )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    return 1 if report.blocking else 0


def cmd_resolve(args) -> int:
    import asyncio

    from app.analysis.resolve_symbol import describe_address, resolve_many
    from app.config import settings

    chain = None if args.any_chain else (args.chain or settings.CHAIN)

    def _show(res, *, address_mode=False):
        print(f" {res.symbol}  [{res.verdict()}]")
        if res.error:
            print(f"        could not reach the listing source: {res.error}")
            print()
            return
        if not res.candidates:
            if address_mode:
                print(
                    f"        no pool on {chain or 'any chain'} holds this mint. Either it is not\n"
                    f"        traded on a DEX, or the address is not a token."
                )
            else:
                print(
                    f"        nothing on {chain or 'any chain'} lists this symbol. If the chart is a\n"
                    f"        CEX or index symbol it has no mint, and this bot cannot trade it."
                )
            print()
            return

        for c in res.candidates:
            marker = "->" if c is res.best and res.unambiguous else "  "
            turnover = "" if c.turnover is None else f" · turnover {c.turnover:.4f}"
            print(f"   {marker} {c.token_address}")
            print(
                f"        {c.chain:<10} {(c.name or '?')[:30]:<30} "
                f"liq ${c.liquidity_usd or 0:,.0f} · vol24h ${c.volume_24h_usd or 0:,.0f} "
                f"· {c.pair_count} pair(s){turnover}"
            )
            if not c.live:
                print(f"        REJECTED: {c.why_not_live()}")

        if len(res.live_candidates) > 1 and not res.unambiguous:
            print(
                "        More than one genuinely traded claimant. Do NOT pick by size -\n"
                "        open each on its listing page and match it to your chart."
            )
        print()

    if args.address:
        print(RULE)
        print(" RESOLVE - what is this mint?")
        print(RULE)
        print(f"   chain          {chain or 'any'}")
        print(f"   address        {args.address}\n")
        res = asyncio.run(describe_address(args.address, chain))
        _show(res, address_mode=True)
        return 0 if res.candidates else 1

    symbols = [s.strip() for s in ",".join(args.symbols).split(",") if s.strip()]
    if not symbols:
        symbols = [s.strip() for s in settings.SYMBOLS_WATCHLIST.split(",") if s.strip()]

    print(RULE)
    print(" RESOLVE - which mint does each chart symbol actually mean?")
    print(RULE)
    print(f"   chain          {chain or 'any'}")
    print(f"   symbols        {', '.join(symbols)}")
    print(
        "\n   A symbol is a label, not an identity - anyone can mint a token called\n"
        "   BONK, and copycats do it on purpose. Ranked by traded volume, NOT by\n"
        "   reported liquidity: a pool can claim any figure it likes, and the\n"
        "   copycats claim enormous ones. Volume is what somebody actually did.\n"
    )

    results = asyncio.run(resolve_many(symbols, chain))
    for res in results:
        _show(res)

    resolved = [r for r in results if r.unambiguous]
    print(RULE)
    print(f" {len(resolved)} of {len(results)} symbol(s) resolved to a single traded mint.")
    if len(resolved) < len(results):
        print(
            " The rest need a human. If you already have an address in hand, confirm it\n"
            " directly - that does not depend on the search returning the right token:\n"
            "     python scripts/research.py resolve --address <mint>"
        )
    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
    return 0


def cmd_degradation(args) -> int:
    from app.analysis.degradation import build_degradation

    db = SessionLocal()
    try:
        print(RULE)
        print(" DEGRADATION - has recent behaviour moved away from the baseline?")
        print(RULE)
        report = build_degradation(db, recent_trades=args.recent)
        print(f" version: {report.strategy_version}   trades: {report.total_trades}"
              f"   baseline {report.baseline_n} / recent {report.recent_n}")
        if report.shifts:
            print()
            for shift in report.shifts:
                print(shift.render())
        comparable = [g for g in report.groups if g.comparable]
        if comparable:
            print("\n BY CONDITION (directional - no significance test on a split this thin)")
            for group in comparable:
                delta = f"{group.delta:+.2f}" if group.delta is not None else "n/a"
                print(f"  {group.axis:<12}{group.group:<20}"
                      f"{group.baseline:>8.2f}% -> {group.recent:>8.2f}%  ({delta})"
                      f"  n={group.baseline_n}/{group.recent_n}")
        print(f"\n {report.verdict()}")
        print(
            "\n The baseline is this version's OWN earlier trades. A strategy that never\n"
            " worked will not show as degrading - that question belongs to calibration.\n"
            " Nothing here changes a threshold, halts trading, or loosens a filter."
        )
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
    finally:
        db.close()
    return 0


def cmd_counterfactual(args) -> int:
    from app.analysis.counterfactual import MIN_COHORT, build_counterfactual

    db = SessionLocal()
    try:
        print(RULE)
        print(" REJECTED-SIGNAL COUNTERFACTUAL - what the filters turned down")
        print(RULE)
        report = build_counterfactual(db, horizon_minutes=args.horizon)
        print(f" horizon: {args.horizon}m   accepted cohort: {report.accepted.n}"
              f"   unmatched: {report.unmatched}")
        if report.invisible_stages:
            print(
                f"\n NOT MEASURABLE: {', '.join(report.invisible_stages)} run before "
                "forward-return\n tracking begins, so their rejects have no recorded outcome. "
                "Absent, not innocent."
            )
        print()
        for gate in report.gates:
            tag = " [SAFETY]" if gate.protected else ""
            print(f" [{gate.grade():<17}] {gate.stage}{tag}")
            print(f"   {gate.note()}")
        print(f"\n {report.verdict()}")
        print(
            f"\n Cohorts below {MIN_COHORT} on either side are not compared. Nothing here "
            "changes\n a filter: a result worth acting on becomes a challenger and earns its "
            "way\n through the promotion gate on a paired sample."
        )
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
    finally:
        db.close()
    return 0


def cmd_collection(args) -> int:
    from app.analysis.collection import TARGET_PAIRS, check_collection

    db = SessionLocal()
    try:
        print(RULE)
        print(" COLLECTION HEALTH - is the run producing usable observations?")
        print(RULE)
        report = check_collection(db)
        for check in report.checks:
            print(f"\n [{check.status:<17}] {check.name}")
            print(f"   {check.detail}")
            if check.counts:
                print("   " + "  ".join(f"{k}={v}" for k, v in check.counts.items()))

        print(f"\n{RULE}")
        if report.paired:
            print(" PAIRED SAMPLE PROGRESS")
            for name, n in sorted(report.paired.items()):
                bar = "#" * int(min(n / TARGET_PAIRS, 1.0) * 40)
                print(f"   {name:<16} {n:>5} / {TARGET_PAIRS}  |{bar:<40}|")
        print(f"\n {report.verdict()}")
        print(
            "\n This checks whether the DATA is usable. It says nothing about whether any\n"
            " strategy is good - that question needs the promotion gate, and the gate needs\n"
            " the sample above to be full first."
        )
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
        # A failed check means the observations being written right now are
        # not trustworthy, which is worth a non-zero exit so a cron job or a
        # CI step notices instead of scrolling past.
        return 1 if report.failures else 0
    finally:
        db.close()


def cmd_diagnose(args) -> int:
    from app.autopilot.diagnose import diagnose

    db = SessionLocal()
    try:
        print(RULE)
        print(" DIAGNOSIS - problems visible in the recorded data")
        print(RULE)
        report = diagnose(db)
        print(report.render())
        print(
            "\n  Findings marked HUMAN change what the strategy believes. Autopilot logs\n"
            "  them and stops - acting on a correlation in a few hundred rows is how a\n"
            "  loop fits itself to a fortnight of market."
        )
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
    finally:
        db.close()
    return 0


def cmd_changelog(args) -> int:
    from app.autopilot import changelog

    db = SessionLocal()
    try:
        print(RULE)
        print(" AUTOPILOT CHANGELOG")
        print(RULE)
        print(changelog.render(db, limit=args.limit))
    finally:
        db.close()
    return 0


def cmd_readiness(args) -> int:
    from app.analysis.readiness import build_readiness

    db = SessionLocal()
    try:
        print(RULE)
        print(" DATA READINESS - what can be answered yet, and what is still accumulating")
        print(RULE)
        report = build_readiness(db, horizon_minutes=args.horizon)
        print(report.table())
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
    finally:
        db.close()
    return 0


def cmd_early(args) -> int:
    from app.analysis.early_calibration import (
        build_early_calibration, build_false_positives, build_lead_time,
    )

    db = SessionLocal()
    try:
        print(RULE)
        print(" EARLY SIGNAL ENGINE")
        print(RULE)

        print("\n CALIBRATION - does a higher EARLY score precede a better outcome?\n")
        for horizon in args.horizons:
            table = build_early_calibration(db, horizon_minutes=horizon)
            print(f" {horizon}m: {table.verdict()}")
            usable = [b for b in table.buckets if b.sample_size]
            if usable:
                print(f"    {'bucket':<10}{'n':>6}{'mean %':>9}{'win %':>8}"
                      f"{'net %':>9}{'MFE':>8}{'MAE':>8}")
                for b in usable:
                    mark = " " if b.meaningful else "*"
                    mfe = f"{b.mean_favorable_pct:+.1f}" if b.mean_favorable_pct is not None else "-"
                    mae = f"{b.mean_adverse_pct:+.1f}" if b.mean_adverse_pct is not None else "-"
                    print(f"  {mark} {b.bucket:<10}{b.sample_size:>6}"
                          f"{b.mean_return_pct:>+9.2f}{b.win_rate_pct:>8.0f}"
                          f"{b.expectancy_net_pct:>+9.2f}{mfe:>8}{mae:>8}")
            print()
        print(" * fewer than 30 measured outcomes - shown, but not evidence")

        lead = build_lead_time(db)
        print(f"\n LEAD TIME\n {lead.verdict()}")
        if lead.tracked:
            print(f"    {'move':<10}{'reached':>9}{'detected first':>16}{'share':>8}")
            for row in lead.as_dict()["thresholds"]:
                share = f"{row['share_pct']:.0f}%" if row["share_pct"] is not None else "-"
                print(f"    +{row['threshold_pct']:<9.0f}{row['reached']:>9}"
                      f"{row['detected_before']:>16}{share:>8}")

        fp = build_false_positives(db)
        print(f"\n FALSE POSITIVES\n {fp.verdict()}")
        if fp.by_category:
            print(f"    {'category':<26}{'count':>7}")
            for category, count in fp.by_category:
                print(f"    {category:<26}{count:>7}")
        if fp.by_score_bucket:
            print(f"\n    {'early bucket':<14}{'failed':>8}{'total':>7}{'fail rate':>11}")
            for bucket, (failed, total) in fp.by_score_bucket.items():
                rate = f"{failed / total * 100:.0f}%" if total else "-"
                print(f"    {bucket:<14}{failed:>8}{total:>7}{rate:>11}")

        if args.json:
            print(json.dumps({
                "calibration": [
                    build_early_calibration(db, horizon_minutes=h).as_dict() for h in args.horizons
                ],
                "lead_time": lead.as_dict(),
                "false_positives": fp.as_dict(),
            }, indent=2))
    finally:
        db.close()
    return 0


def cmd_early_ablate(args) -> int:
    from app.research.early_ablation import run_early_ablation

    db = SessionLocal()
    try:
        print(RULE)
        print(" EARLY FEATURE ABLATION - does each early factor earn its weight?")
        print(RULE)
        report = run_early_ablation(db, horizon_minutes=args.horizon)
        print(report.summary())
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
    finally:
        db.close()
    return 0


def cmd_early_walkforward(args) -> int:
    from app.research.early_walkforward import walk_forward_early_threshold

    db = SessionLocal()
    try:
        print(RULE)
        print(" EARLY THRESHOLD WALK-FORWARD - does the level survive out-of-sample?")
        print(RULE)
        report = walk_forward_early_threshold(
            db, horizon_minutes=args.horizon, windows=args.windows
        )
        print(report.table())
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
    finally:
        db.close()
    return 0


def cmd_modes(args) -> int:
    from app.research.strategy_modes import compare_modes

    series = _series(args.symbol, synthetic=args.synthetic, limit=args.limit)
    if series is None:
        return 1
    print(RULE)
    print(" STRATEGY MODES - technical only vs early only vs early+technical")
    print(RULE)
    comparison = compare_modes(series, base_config=BacktestConfig(warmup_bars=args.warmup))
    print(comparison.table())
    if args.json:
        print(json.dumps(comparison.as_dict(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def backtest_args(p):
        p.add_argument("symbol", help="token mint address, or any name when --synthetic")
        p.add_argument("--synthetic", action="store_true",
                       help="generated candles - exercises the machinery, proves nothing")
        p.add_argument("--limit", type=int, default=3000, help="candles to fetch (default 3000)")
        p.add_argument("--warmup", type=int, default=210, help="warmup bars (default 210)")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("report", help="the full validation report")
    p.add_argument("--version", help="restrict to one strategy version label")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("distribution", help="what the scoring engine produces")
    p.add_argument("--include-unreliable", action="store_true")
    p.set_defaults(func=cmd_distribution)

    p = sub.add_parser("calibration", help="does a higher score predict a better outcome?")
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser("funnel", help="where discovered tokens die")
    p.add_argument("--hours", type=float, default=None, help="window (default: all history)")
    p.set_defaults(func=cmd_funnel)

    p = sub.add_parser("thresholds", help="the MIN_SIGNAL_SCORE_TO_ENTER ladder")
    backtest_args(p)
    p.set_defaults(func=cmd_thresholds)

    p = sub.add_parser("ablate", help="leave-one-out over the scoring factors")
    backtest_args(p)
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("sweep", help="one parameter across neighbouring values")
    backtest_args(p)
    p.add_argument("param", help="BacktestConfig field name")
    p.add_argument("values", help="comma-separated values, e.g. 60,62.5,65,67.5,70")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("replay", help="thresholds re-scored on the bot's own history")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--thresholds", default="50,55,60,65,70,75,80")
    p.add_argument("--version", help="restrict to one strategy version label")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("postmortem", help="per-trade autopsies with MFE/MAE")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_postmortem)

    p = sub.add_parser("shadow", help="champion vs challengers, on paired opportunities")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_shadow)

    p = sub.add_parser("evidence", help="is there enough evidence to trust the strategy?")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_evidence)

    p = sub.add_parser("integrity", help="observations that must not be counted")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_integrity)

    p = sub.add_parser("preflight", help="can this machine actually run a collection?")
    p.add_argument("--no-probe", action="store_true", help="skip the live upstream probes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("resolve", help="which mint does a chart symbol actually mean?")
    p.add_argument(
        "symbols", nargs="*",
        help="symbols to resolve; defaults to SYMBOLS_WATCHLIST from .env",
    )
    p.add_argument(
        "--address", help="ask what one specific mint is, instead of searching by symbol",
    )
    p.add_argument("--chain", help="restrict to one chain (default: CHAIN from .env)")
    p.add_argument(
        "--any-chain", action="store_true",
        help="do not filter by chain - shows the same symbol across every chain listing it",
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("degradation", help="has recent behaviour moved away from the baseline?")
    p.add_argument("--recent", type=int, default=30, help="trades in the recent window")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_degradation)

    p = sub.add_parser("counterfactual", help="did the filters reject the better opportunities?")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_counterfactual)

    p = sub.add_parser("collection", help="is the paper run producing usable observations?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_collection)

    p = sub.add_parser("diagnose", help="problems visible in the recorded data")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("changelog", help="what autopilot changed, and why")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_changelog)

    p = sub.add_parser("readiness", help="how much data each question still needs")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_readiness)

    p = sub.add_parser("early", help="Early Signal Engine: calibration, lead time, false positives")
    p.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60, 120],
                   help="forward-return horizons in minutes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_early)

    p = sub.add_parser("early-ablate", help="leave-one-out over the early factors, on stored data")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_early_ablate)

    p = sub.add_parser("early-walkforward", help="is the early threshold stable out-of-sample?")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_early_walkforward)

    p = sub.add_parser("modes", help="A: technical / B: early / C: both")
    backtest_args(p)
    p.set_defaults(func=cmd_modes)

    args = parser.parse_args()
    init_db()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
