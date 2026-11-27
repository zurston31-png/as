#!/usr/bin/env python3
"""Simulate ONE strategy on the stored daily bars and record the run.

This is a simulation over historical data. It places no orders, connects to no
broker and cannot trade real money. Each run covers a single strategy with its
own virtual account; strategies are never combined.

Run from the project folder, for example:

    python scripts/run_strategy.py momentum_breakout --period in_sample
    python scripts/run_strategy.py momentum_breakout --period in_sample --cost-scenario worse
    python scripts/run_strategy.py momentum_breakout --period in_sample --param breakout_window=40

    python scripts/run_strategy.py momentum_breakout --freeze-parameters
    python scripts/run_strategy.py momentum_breakout --period out_of_sample

The in-sample period (2021-06-01 to 2024-12-31) can be run freely. Everything
from 2025-01-01 on is reserved: it can only be run after the parameters were
frozen, and only with exactly those parameters. Frozen parameters cannot be
changed afterwards.

Every signal is stored, including the ones that were not traded and the trades
that lost money (tables strategy_runs, strategy_signals, strategy_trades,
strategy_equity).

Exit codes: 0 = done, 2 = the command, the configuration or the request was refused.
"""

import argparse
import sys
from pathlib import Path

# Allow running this file directly (python scripts/run_strategy.py) without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigurationError, get_settings
from app.database import (
    ProtectedDatabase,
    SchemaError,
    create_db_engine,
    create_session_factory,
    database_file_path,
    init_db,
    refuse_protected,
    session_scope,
)
from app.strategies.assumptions import (
    DAILY_COST_SCENARIOS,
    IN_SAMPLE,
    OUT_OF_SAMPLE,
    STARTING_EQUITY,
)
from app.strategies.runner import (
    STRATEGIES,
    RunSummary,
    StrategyRunError,
    build_strategy,
    freeze_parameters,
    run_strategy,
)

PERIODS = {IN_SAMPLE.name: IN_SAMPLE, OUT_OF_SAMPLE.name: OUT_OF_SAMPLE}


def _parameter(value: str) -> tuple[str, str]:
    name, separator, raw = value.partition("=")
    if not separator or not name.strip() or not raw.strip():
        raise argparse.ArgumentTypeError(f"{value!r} is not of the form name=value")
    return name.strip(), raw.strip()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_strategy.py",
        description=(
            "Simulate one strategy on stored daily bars (no orders, no real money) "
            "and record every signal and trade."
        ),
    )
    parser.add_argument("strategy", choices=sorted(STRATEGIES), help="the strategy to simulate")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--period", choices=sorted(PERIODS), help="the period to simulate")
    action.add_argument(
        "--freeze-parameters",
        action="store_true",
        help="fix the parameters ahead of the out-of-sample run (cannot be undone)",
    )
    parser.add_argument(
        "--cost-scenario",
        choices=list(DAILY_COST_SCENARIOS),
        default="base",
        help="assumed slippage per side: "
        + ", ".join(f"{name} = {bps:g} bp" for name, bps in DAILY_COST_SCENARIOS.items())
        + " (default: base)",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        type=_parameter,
        metavar="NAME=VALUE",
        help="change one parameter from its default; repeat for several",
    )
    return parser.parse_args(argv)


def print_summary(summary: RunSummary, database: str) -> None:
    print("SIMULATION ONLY - historical bars, virtual account, no orders were placed.")
    print(f"Run {summary.run_id}: {summary.strategy}, period {summary.period} "
          f"({summary.period_start.isoformat()} to {summary.period_end.isoformat()})")  # fmt: skip
    print(f"Database:           {database}")
    print(f"Cost scenario:      {summary.cost_scenario} ({summary.slippage_bps:g} bp per side, "
          "an assumption)")  # fmt: skip
    print(f"Bars read:          {summary.bars_used:,}")
    for symbol, days in summary.excluded_sessions.items():
        listed = ", ".join(day.isoformat() for day in days[:5]) + (" ..." if len(days) > 5 else "")
        print(f"Excluded for {symbol}:  {len(days)} stored session(s) outside the unbroken "
              f"history ({listed}); kept in the database")  # fmt: skip
    print(f"Signals recorded:   {summary.signals_total:,} "
          f"({summary.signals_taken:,} traded, "
          f"{summary.signals_total - summary.signals_taken:,} not traded)")  # fmt: skip
    for reason, count in summary.signals_skipped_by_reason.items():
        print(f"  not traded, {reason}: {count:,}")
    print(f"Trades recorded:    {summary.trades_total:,}")
    for reason, count in summary.trades_by_exit_reason.items():
        print(f"  closed by {reason}: {count:,}")
    print(f"Starting equity:    {summary.starting_equity:,.2f}")
    print(f"Ending equity:      {summary.ending_equity:,.2f}")
    print(f"Data fingerprint:   {summary.data_fingerprint}")
    print(f"Result fingerprint: {summary.result_fingerprint}")
    print("The same data, parameters and assumptions always give the same result fingerprint.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = get_settings()
        strategy = build_strategy(args.strategy, dict(args.param))
    except (ConfigurationError, StrategyRunError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    path = database_file_path(settings.database_url)
    if path is not None and not path.exists():
        print(
            f"Error: database not found at {path}\n"
            "Create it with:  python -m app.database   and download bars with scripts/ingest.py",
            file=sys.stderr,
        )
        return 2

    try:
        refuse_protected(settings.database_url)
    except ProtectedDatabase as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    engine = create_db_engine(settings.database_url)
    try:
        init_db(engine)
        session_factory = create_session_factory(engine)
        if args.freeze_parameters:
            with session_scope(session_factory) as session:
                freeze = freeze_parameters(session, strategy)
                frozen_at, parameters = freeze.frozen_at, dict(freeze.parameters)
            print(f"Parameters of {strategy.name} (rules version {strategy.version}) are frozen "
                  f"since {frozen_at.strftime('%Y-%m-%dT%H:%M:%SZ')}:")  # fmt: skip
            for name, value in parameters.items():
                print(f"  {name} = {value}")
            print("They cannot be changed. The out-of-sample period can now be run with them.")
            return 0
        summary = run_strategy(
            session_factory,
            settings,
            strategy,
            PERIODS[args.period],
            cost_scenario=args.cost_scenario,
            starting_equity=STARTING_EQUITY,
        )
    except (StrategyRunError, SchemaError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.dispose()

    print_summary(summary, str(path) if path is not None else "in-memory database")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
