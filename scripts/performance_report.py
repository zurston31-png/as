#!/usr/bin/env python3
"""Print the full performance report for the paper-trading record.

    python scripts/performance_report.py
    python scripts/performance_report.py --version v-f412f88b
    python scripts/performance_report.py --list-versions
    python scripts/performance_report.py --json

Answers the question the dashboard's headline numbers cannot: is this
record strong enough to believe? It leads with the strategy's validation
status (EXPERIMENTAL / FAILING / VALIDATED) rather than with the return,
because "+34%" and "on 12 trades" mean very different things and the
second one is the part people skip.

Everything here is read-only. Running it never places, cancels or modifies
anything.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import models  # noqa: E402
from app.analysis.report import build_performance_report  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402

RULE = "=" * 78


def _fmt(value, spec="{:,.2f}", none="n/a"):
    return none if value is None else spec.format(value)


def print_report(report) -> None:
    stats = report.stats

    print(RULE)
    print(f" STRATEGY STATUS: {report.validation.status.value.upper()}")
    print(RULE)
    print(f" {report.validation.headline}")
    print()
    print(report.validation.summary().split("\n", 2)[2])

    if report.warnings:
        print()
        print(" READ THIS FIRST")
        for w in report.warnings:
            print(f"   ! {w}")

    print()
    print(RULE)
    print(" RESULTS")
    print(RULE)
    print(f"  closed trades        {stats.trade_count}")
    print(f"  win rate             {stats.win_rate:.1f}%  ({stats.win_count}W / {stats.loss_count}L)")
    print(f"  net P&L              ${_fmt(report.net_pnl_usd)}")
    print(f"  gross P&L (pre-cost) ${_fmt(report.gross_pnl_usd)}")
    print(f"  expectancy / trade   ${_fmt(stats.expectancy_usd)}")
    print(f"  profit factor        {_fmt(stats.profit_factor, '{:.2f}')}")
    print(f"  max drawdown         {stats.max_drawdown_pct:.1f}%")
    print(f"  avg win / avg loss   ${_fmt(stats.avg_win_usd)} / ${_fmt(stats.avg_loss_usd)}")
    print(f"  longest losing run   {stats.longest_losing_streak}")

    ex = report.extremes
    print(f"  largest win          ${_fmt(ex.largest_win_usd)}  ({ex.largest_win_symbol or 'n/a'})")
    print(f"  largest loss         ${_fmt(ex.largest_loss_usd)}  ({ex.largest_loss_symbol or 'n/a'})")
    if ex.profit_depends_on_one_trade:
        print("   ! one trade produced most of the gross profit - this is a lucky sample, not an edge")

    print()
    print(RULE)
    print(" WHAT IT COST")
    print(RULE)
    c = report.costs
    print(f"  fees                 ${_fmt(c.total_fees_usd)}")
    print(f"  slippage + impact    ${_fmt(c.total_slippage_usd)}")
    print(f"  total execution cost ${_fmt(c.total_execution_cost_usd)}")
    print(f"  avg cost per leg     {_fmt((c.avg_execution_cost_pct or 0) * 100, '{:.3f}')}%")
    print(f"  avg fill delay       {_fmt(c.avg_fill_delay_seconds, '{:.2f}')}s")
    print(f"  cost data coverage   {c.coverage_pct:.0f}% "
          f"({c.legs_missing_cost_data} leg(s) unrecorded)")

    h = report.holding
    print()
    print(RULE)
    print(" HOLDING TIME")
    print(RULE)
    print(f"  average / median     {_fmt(h.avg_hours, '{:.1f}')}h / {_fmt(h.median_hours, '{:.1f}')}h")
    print(f"  shortest / longest   {_fmt(h.shortest_hours, '{:.1f}')}h / {_fmt(h.longest_hours, '{:.1f}')}h")
    print(f"  winners / losers     {_fmt(h.avg_winner_hours, '{:.1f}')}h / "
          f"{_fmt(h.avg_loser_hours, '{:.1f}')}h")
    if h.winners_held_longer is False:
        print("   ! losers are held longer than winners - cutting winners short and riding losers")

    print()
    print(RULE)
    print(" BREAKDOWNS   (buckets marked * have too few trades to mean anything)")
    print(RULE)
    for breakdown in report.breakdowns:
        print(f"\n  by {breakdown.dimension}"
              + (f"   [{breakdown.unknown_count} not recorded]" if breakdown.unknown_count else ""))
        if not breakdown.buckets:
            print("    (no closed trades with this attribute yet)")
            continue
        print(f"    {'bucket':<24}{'trades':>8}{'win %':>8}{'total P&L':>13}{'per trade':>12}")
        for b in breakdown.buckets:
            mark = " " if b.meaningful else "*"
            print(f"  {mark} {b.label:<24}{b.trade_count:>8}{b.win_rate:>7.0f}%"
                  f"{b.total_pnl_usd:>13,.2f}{b.expectancy_usd:>12,.2f}")

    if report.rejections and report.rejections.total:
        print()
        print(RULE)
        print(" WHERE CANDIDATES WERE REJECTED")
        print(RULE)
        for entry in report.rejections.as_dict()["by_reason"]:
            print(f"  {entry['reason']:<32}{entry['count']:>7}  ({entry['share_pct']:.0f}%)")
        print("\n  A filter rejecting a lot is doing its job. The fix for too few trades is")
        print("  better candidates, never a lower filter.")

    if report.monte_carlo:
        print()
        print(RULE)
        print(" MONTE CARLO   (the realized path was one draw; this is the range)")
        print(RULE)
        print("  " + report.monte_carlo.summary().replace("\n", "\n  "))

    if report.version_counts:
        print()
        print(RULE)
        print(" STRATEGY VERSIONS IN THIS RECORD")
        print(RULE)
        for label, count in sorted(report.version_counts.items()):
            print(f"  {label:<20}{count:>6} trade legs")

    print()
    print(RULE)
    print(" Paper trading only. No real funds, no wallet, no private key involved.")
    print(RULE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="restrict the report to one strategy version label")
    parser.add_argument("--list-versions", action="store_true",
                        help="list the strategy versions on record and exit")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--simulations", type=int, default=2_000,
                        help="Monte Carlo paths to run (default 2000)")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.list_versions:
            rows = db.query(models.StrategyVersion).order_by(models.StrategyVersion.created_at).all()
            if not rows:
                print("No strategy versions recorded yet - the bot hasn't processed a signal.")
                return 0
            print(f"{'label':<16}{'first seen':<22}{'last seen':<22}")
            for r in rows:
                print(f"{r.label:<16}{str(r.created_at):<22}{str(r.last_seen_at):<22}")
            return 0

        report = build_performance_report(
            db, strategy_version=args.version, monte_carlo_simulations=args.simulations
        )
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
        else:
            print_report(report)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
