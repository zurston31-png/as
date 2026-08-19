#!/usr/bin/env python3
"""Strategy research CLI.

    python scripts/research.py report                  the validation report
    python scripts/research.py distribution            what the scorer produces
    python scripts/research.py calibration             does the score predict?
    python scripts/research.py funnel                  where candidates die
    python scripts/research.py thresholds  <symbol>    the threshold ladder
    python scripts/research.py ablate      <symbol>    which factors earn their weight
    python scripts/research.py sweep       <symbol> <param> <v1,v2,...>

The first four read the bot's own database and need no network. The last
three run backtests and need candle history for the symbol - by default
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

    args = parser.parse_args()
    init_db()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
