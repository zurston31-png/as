#!/usr/bin/env python3
"""Audit stored strategy runs, read-only.

This command simulates nothing and changes nothing. It reads runs that
scripts/run_strategy.py already stored, and the daily bars of the same period
for the buy-and-hold benchmark, and prints the audit.

Run from the project folder, for example:

    python scripts/audit_strategy.py momentum_breakout        # newest in-sample run per scenario
    python scripts/audit_strategy.py momentum_breakout --run 4 --run 5 --run 6
    python scripts/audit_strategy.py momentum_breakout --expect-trades 62
    python scripts/audit_strategy.py momentum_breakout --benchmark-only   # no runs needed

Exit codes: 0 = audit printed and every integrity check passed, 1 = an
integrity check failed or the trade count differs from --expect-trades,
2 = nothing to audit or the request is invalid.
"""

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

# Allow running this file directly without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigurationError, get_settings
from app.database import (
    create_db_engine,
    create_session_factory,
    current_revision,
    database_file_path,
    head_revision,
    session_scope,
)
from app.models import StrategyRun
from app.strategies.assumptions import DAILY_COST_SCENARIOS, IN_SAMPLE, OUT_OF_SAMPLE
from app.strategies.audit import RunAudit, audit_run, buy_and_hold, latest_runs, trade_overlap
from app.strategies.data import load_daily_bars
from app.strategies.runner import STRATEGIES

PERIODS = {IN_SAMPLE.name: IN_SAMPLE, OUT_OF_SAMPLE.name: OUT_OF_SAMPLE}


def pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value * 100:+.{digits}f}%"


def plain_pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


def num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python scripts/audit_strategy.py",
        description="Read-only performance audit of stored strategy runs.",
    )
    parser.add_argument("strategy", choices=sorted(STRATEGIES))
    parser.add_argument("--period", choices=sorted(PERIODS), default=IN_SAMPLE.name)
    parser.add_argument(
        "--run",
        type=int,
        action="append",
        metavar="RUN_ID",
        help="audit these run ids instead of the newest run per cost scenario",
    )
    parser.add_argument(
        "--expect-trades",
        type=int,
        metavar="N",
        help="fail unless every audited run stored exactly N trades",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="print only the buy-and-hold benchmark for the period",
    )
    parser.add_argument("--json", action="store_true", help="print the audit as JSON")
    return parser.parse_args(argv)


def print_benchmark(rows, first: date, last: date, slippage_bps: float) -> None:
    print(
        f"BUY-AND-HOLD, {first.isoformat()} open to {last.isoformat()} close "
        f"(split-adjusted, dividends NOT included; after costs = {slippage_bps:g} bp per side)"
    )
    print(f"  {'SYMBOL':<7}{'ENTRY':>10}{'EXIT':>10}{'RETURN':>11}{'AFTER COSTS':>13}{'MAX DD':>9}")
    for row in rows:
        print(
            f"  {row.symbol:<7}{row.entry_price:>10.2f}{row.exit_price:>10.2f}"
            f"{pct(row.total_return):>11}{pct(row.total_return_after_costs):>13}"
            f"{plain_pct(row.max_drawdown):>9}"
        )
    if rows:
        average = statistics.fmean(row.total_return for row in rows)
        print(
            f"  Equal-weight portfolio of all {len(rows)} (equal dollars at the start, never "
            f"rebalanced): {pct(average)}"
        )


def print_run(audit: RunAudit, benchmark: dict[str, float]) -> None:
    s, d, c, i = audit.statistics, audit.drawdown, audit.concentration, audit.integrity
    print(
        f"\nRUN {audit.run_id}: {audit.strategy} v{audit.strategy_version}, {audit.period} "
        f"{audit.period_start} to {audit.period_end}, cost scenario {audit.cost_scenario} "
        f"({audit.slippage_bps:g} bp per side)"
    )
    print(
        f"  Starting / ending equity  {money(audit.starting_equity)} / {money(audit.ending_equity)}"
        f"   total return {pct(audit.total_return)}, annualised {pct(audit.annualised_return)}"
    )
    print(
        f"  Volatility (annualised)   {plain_pct(audit.annualised_volatility)}"
        f"   Sharpe ratio (risk-free rate 0) {num(audit.sharpe_ratio)}"
    )
    print(
        f"  Exposure                  {plain_pct(audit.exposure)} of sessions with a position open"
        f" at the close, {num(audit.average_open_positions)} positions on average"
    )

    print("\n  Trades")
    print(
        f"    Trades {s.trades}: {s.wins} wins, {s.losses} losses, {s.breakeven} breakeven"
        f"   win rate {plain_pct(s.win_rate)}"
    )
    print(
        f"    Profit factor {num(s.profit_factor)}   payoff ratio (avg win / avg loss) "
        f"{num(s.payoff_ratio)}"
    )
    print(
        f"    Average win {money(s.average_win)} ({num(s.average_win_r)} R)   "
        f"average loss {money(s.average_loss)} ({num(s.average_loss_r)} R)"
    )
    print(f"    Expectancy {money(s.expectancy)} per trade ({num(s.expectancy_r)} R)")
    print(
        f"    Gross profit {money(s.gross_profit)}   gross loss {money(s.gross_loss)}   "
        f"net {money(s.net_profit)}   assumed slippage paid {money(s.total_slippage)}"
    )
    print(
        f"    Holding period: mean {num(s.average_sessions_held, 1)} sessions, median "
        f"{num(s.median_sessions_held, 1)}; winners {num(s.average_sessions_held_wins, 1)}, "
        f"losers {num(s.average_sessions_held_losses, 1)}"
    )
    print(
        f"    Exits: {', '.join(f'{k} {v}' for k, v in s.exits_by_reason.items()) or 'none'}"
        f"   longest losing streak {s.max_consecutive_losses}"
    )
    for title, items in (("Largest wins", s.largest_wins), ("Largest losses", s.largest_losses)):
        print(f"    {title}:")
        for t in items or []:
            print(
                f"      {t.symbol:<6}{t.entry_session} -> {t.exit_session} {t.shares:>5} sh"
                f"  net {money(t.net_pnl):>11}  {t.r_multiple:+.2f} R  ({t.exit_reason})"
            )
        if not items:
            print("      none")

    print("\n  Drawdown")
    if d.trough_session is None:
        print("    none: equity never fell below an earlier high")
    else:
        recovered = (
            f"recovered {d.recovery_session} after {d.sessions_trough_to_recovery} sessions"
            if d.recovered
            else f"NOT recovered by the end of the period ({d.sessions_underwater} sessions "
            "below the earlier high)"
        )
        print(
            f"    Maximum drawdown {plain_pct(d.max_drawdown)} ({money(d.max_drawdown_amount)}): "
            f"peak {d.peak_session or 'start'} -> trough {d.trough_session} "
            f"({d.sessions_peak_to_trough} sessions), {recovered}"
        )

    print("\n  By symbol (strategy return = net profit / starting equity; not a stand-alone run)")
    print(
        f"    {'SYMBOL':<7}{'TRADES':>7}{'WINS':>6}{'NET':>13}{'STRATEGY':>10}{'B&H':>10}"
        f"{'% NET PROFIT':>14}{'% GROSS PROFIT':>16}"
    )
    for row in audit.symbols:
        print(
            f"    {row.symbol:<7}{row.trades:>7}{row.wins:>6}{money(row.net_profit):>13}"
            f"{pct(row.strategy_return):>10}{pct(benchmark.get(row.symbol)):>10}"
            f"{plain_pct(row.share_of_net_profit):>14}{plain_pct(row.share_of_gross_profit):>16}"
        )

    print("\n  Dependence on single stocks or trades")
    print(
        f"    Best symbol {c.best_symbol}: {plain_pct(c.best_symbol_share_of_net_profit)} of net "
        f"profit; net without it {money(c.net_profit_without_best_symbol)}"
    )
    print(
        f"    Best trade: {plain_pct(c.best_trade_share_of_net_profit)} of net profit; net without "
        f"it {money(c.net_profit_without_best_trade)}; top 5 trades "
        f"{plain_pct(c.top_five_trades_share_of_net_profit)}"
    )
    print(
        f"    Symbols with net profit {c.symbols_with_net_profit}, with net loss "
        f"{c.symbols_with_net_loss}"
    )
    verdict = {True: "YES", False: "no", None: "not applicable"}[c.heavily_dependent]
    print(f"    Heavily dependent: {verdict} ({c.reason})")
    print(
        "    (Removing a symbol only subtracts its recorded trades; the freed cash and slots "
        "are not re-simulated.)"
    )

    print("\n  Integrity")
    print(
        f"    Signals {i.signals_total}: {i.signals_taken} taken, {i.signals_skipped} not taken "
        f"({', '.join(f'{k} {v}' for k, v in i.skipped_by_reason.items()) or 'none'})"
    )
    print(
        f"    Trades stored {i.trades_stored} (run header {i.trades_in_run_header}), of which "
        f"{i.losing_trades_stored} losing"
    )
    print(
        f"    One trade per taken signal: {i.every_taken_signal_has_one_trade}; ending equity = "
        f"start + all net results: {i.ending_equity_matches_trades}; equity curve complete: "
        f"{i.equity_curve_complete}; trade arithmetic and dates: {i.trade_identity_holds}"
    )
    print(f"    Result: {'PASSED' if i.ok else 'FAILED - ' + '; '.join(i.problems)}")


def print_comparison(audits: list[RunAudit]) -> None:
    print("\nCOST SCENARIOS COMPARED")
    same_inputs = (
        len({(a.parameters_hash, a.data_fingerprint, a.period_start, a.period_end) for a in audits})
        == 1
    )
    print(f"  Same parameters, data and period in every run: {same_inputs}")
    rows = [
        ("Slippage per side (bp)", lambda a: f"{a.slippage_bps:g}"),
        ("Total return", lambda a: pct(a.total_return)),
        ("Annualised return", lambda a: pct(a.annualised_return)),
        ("Trades", lambda a: str(a.statistics.trades)),
        ("Win rate", lambda a: plain_pct(a.statistics.win_rate)),
        ("Profit factor", lambda a: num(a.statistics.profit_factor)),
        ("Expectancy ($)", lambda a: money(a.statistics.expectancy)),
        ("Expectancy (R)", lambda a: num(a.statistics.expectancy_r)),
        ("Slippage paid ($)", lambda a: money(a.statistics.total_slippage)),
        ("Maximum drawdown", lambda a: plain_pct(a.drawdown.max_drawdown)),
        ("Sharpe ratio", lambda a: num(a.sharpe_ratio)),
    ]
    print(f"  {'':<24}" + "".join(f"{a.cost_scenario + ' #' + str(a.run_id):>16}" for a in audits))
    for title, value in rows:
        print(f"  {title:<24}" + "".join(f"{value(a):>16}" for a in audits))
    overlap = trade_overlap(audits)
    print(
        f"  Trades identical in every scenario (symbol, entry and exit session): "
        f"{overlap['common_to_all']} of {overlap['per_run']}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    path = database_file_path(settings.database_url)
    if path is not None and not path.exists():
        print(f"Error: database not found at {path}", file=sys.stderr)
        return 2

    engine = create_db_engine(settings.database_url)
    try:
        if current_revision(engine) != head_revision():
            print(
                "Error: the database schema is not current; run  python -m app.database  first. "
                "(This audit only reads and does not upgrade it.)",
                file=sys.stderr,
            )
            return 2
        factory = create_session_factory(engine)
        period = PERIODS[args.period]
        with session_scope(factory) as session:
            if args.run:
                runs = [session.get(StrategyRun, run_id) for run_id in args.run]
                missing = [rid for rid, run in zip(args.run, runs, strict=True) if run is None]
                if missing:
                    print(f"Error: no strategy run with id {missing}", file=sys.stderr)
                    return 2
            else:
                runs = (
                    [] if args.benchmark_only else latest_runs(session, args.strategy, period.name)
                )
            if not runs and not args.benchmark_only:
                print(
                    f"Error: no stored {period.name} run of {args.strategy} to audit. Nothing was "
                    "simulated by this command. Create runs with scripts/run_strategy.py, or use "
                    "--benchmark-only.",
                    file=sys.stderr,
                )
                return 2
            first = runs[0].period_start if runs else period.start
            last = runs[0].period_end if runs else period.end
            data = load_daily_bars(session, settings, list(settings.symbols), end=last)
            slippage = runs[0].slippage_bps if runs else DAILY_COST_SCENARIOS["base"]
            bench = buy_and_hold(data, first, last, slippage)
            audits = [audit_run(session, run.id) for run in runs]
    finally:
        engine.dispose()

    benchmark = {row.symbol: row.total_return for row in bench}
    if args.json:
        print(
            json.dumps(
                {"buy_and_hold": [asdict(r) for r in bench], "runs": [asdict(a) for a in audits]},
                default=str,
                indent=2,
            )
        )
    else:
        print("READ-ONLY AUDIT of stored simulations. Nothing was simulated or changed.\n")
        print_benchmark(bench, first, last, slippage)
        for audit in audits:
            print_run(audit, benchmark)
        if len(audits) > 1:
            print_comparison(audits)

    failed = [a.run_id for a in audits if not a.integrity.ok]
    wrong_count = [
        a.run_id
        for a in audits
        if args.expect_trades is not None and a.statistics.trades != args.expect_trades
    ]
    if wrong_count:
        print(
            f"\nTRADE COUNT CHECK FAILED: expected {args.expect_trades} trades in run(s) "
            f"{wrong_count}",
            file=sys.stderr,
        )
    return 1 if failed or wrong_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
