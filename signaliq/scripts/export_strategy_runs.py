#!/usr/bin/env python3
"""Export stored strategy runs to plain files, read-only.

Writes, for each run: its trades, signals and equity curve as CSV; and for the
set of runs: their parameters and fingerprints (runs.json), the buy-and-hold
benchmark of the same period (buy_and_hold.csv), the audit as text and JSON,
and SHA-256 checksums of every file written (SHA256SUMS).

Nothing is simulated and the database is not changed. Numbers are written with
Python's repr, so every value round-trips to exactly what the database holds.

    python scripts/export_strategy_runs.py momentum_breakout --run 1 --run 2 --run 3 \
        --out research/results/strategy_a
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Allow running this file directly without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_audit_command():
    """The audit command's module, loaded from its file next to this one."""
    import importlib.util

    path = Path(__file__).resolve().parent / "audit_strategy.py"
    spec = importlib.util.spec_from_file_location("signaliq_audit_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from app.config import ConfigurationError, get_settings
from app.database import (
    create_db_engine,
    create_session_factory,
    current_revision,
    head_revision,
    session_scope,
)
from app.models import format_utc
from app.strategies.audit import audit_run, buy_and_hold, load_run
from app.strategies.data import load_daily_bars
from app.strategies.runner import STRATEGIES

TRADE_COLUMNS = (
    "trade_id",
    "run_id",
    "signal_id",
    "strategy",
    "symbol",
    "shares",
    "entry_session",
    "entry_reference_price",
    "entry_price",
    "equity_at_signal",
    "initial_stop",
    "risk_amount",
    "exit_session",
    "exit_reference_price",
    "exit_price",
    "exit_reason",
    "sessions_held",
    "gross_pnl",
    "slippage_cost",
    "net_pnl",
    "r_multiple",
)
SIGNAL_COLUMNS = (
    "signal_id",
    "run_id",
    "strategy",
    "symbol",
    "signal_session",
    "signal_type",
    "close",
    "rank_value",
    "rank",
    "decision",
    "reason",
    "planned_shares",
    "indicators",
)
EQUITY_COLUMNS = ("run_id", "session", "cash", "positions_value", "equity", "open_positions")
RUN_FIELDS = (
    "id",
    "created_at",
    "finished_at",
    "status",
    "strategy",
    "strategy_version",
    "period",
    "period_start",
    "period_end",
    "symbols",
    "parameters",
    "parameters_hash",
    "starting_equity",
    "cost_scenario",
    "slippage_bps",
    "timeframe",
    "source",
    "feed",
    "adjustment",
    "bars_used",
    "data_first_session",
    "data_last_session",
    "data_fingerprint",
    "result_fingerprint",
    "signals_total",
    "signals_taken",
    "trades_total",
    "ending_equity",
    "error_message",
)


def cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, datetime):
        return format_utc(value)  # the database's own text form, e.g. 2024-01-02T05:00:00.000000Z
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([cell(row[column]) for column in columns])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/export_strategy_runs.py")
    parser.add_argument("strategy", choices=sorted(STRATEGIES))
    parser.add_argument("--run", type=int, action="append", required=True, metavar="RUN_ID")
    parser.add_argument("--out", type=Path, required=True, help="directory to write into")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    engine = create_db_engine(settings.database_url)
    try:
        if current_revision(engine) != head_revision():
            print("Error: the database schema is not current.", file=sys.stderr)
            return 2
        factory = create_session_factory(engine)
        args.out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        runs_meta = []
        with session_scope(factory) as session:
            for run_id in args.run:
                run, signals, trades, equity = load_run(session, run_id)
                if run.strategy != args.strategy:
                    print(
                        f"Error: run {run_id} is {run.strategy}, not {args.strategy}",
                        file=sys.stderr,
                    )
                    return 2
                if run.period == "out_of_sample":
                    print(
                        f"Error: run {run_id} is an out-of-sample run; not exported here.",
                        file=sys.stderr,
                    )
                    return 2
                name = run.cost_scenario
                runs_meta.append({field: getattr(run, field) for field in RUN_FIELDS})
                paths = {
                    "trades": args.out / f"trades_{name}.csv",
                    "signals": args.out / f"signals_{name}.csv",
                    "equity": args.out / f"equity_{name}.csv",
                }
                write_csv(
                    paths["trades"],
                    TRADE_COLUMNS,
                    [
                        {**{c: getattr(t, c, None) for c in TRADE_COLUMNS}, "trade_id": t.id}
                        for t in trades
                    ],
                )
                write_csv(
                    paths["signals"],
                    SIGNAL_COLUMNS,
                    [
                        {**{c: getattr(s, c, None) for c in SIGNAL_COLUMNS}, "signal_id": s.id}
                        for s in signals
                    ],
                )
                write_csv(
                    paths["equity"],
                    EQUITY_COLUMNS,
                    [{c: getattr(p, c) for c in EQUITY_COLUMNS} for p in equity],
                )
                written.extend(paths.values())

            first, last = runs_meta[0]["period_start"], runs_meta[0]["period_end"]
            data = load_daily_bars(session, settings, list(settings.symbols), end=last)
            bench = buy_and_hold(data, first, last, runs_meta[0]["slippage_bps"])
            audits = [audit_run(session, run_id) for run_id in args.run]
    finally:
        engine.dispose()

    bench_path = args.out / "buy_and_hold.csv"
    write_csv(
        bench_path,
        (
            "symbol",
            "first_session",
            "last_session",
            "entry_price",
            "exit_price",
            "total_return",
            "total_return_after_costs",
            "max_drawdown",
        ),
        [asdict(row) for row in bench],
    )
    runs_path = args.out / "runs.json"
    runs_path.write_text(json.dumps(runs_meta, default=cell, indent=2, sort_keys=True) + "\n")
    audit_json = args.out / "audit.json"
    audit_json.write_text(
        json.dumps(
            {"buy_and_hold": [asdict(r) for r in bench], "runs": [asdict(a) for a in audits]},
            default=cell,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    failed = [audit.run_id for audit in audits if not audit.integrity.ok]
    if failed:
        print(
            f"Error: run(s) {failed} failed an integrity check; the report was not written.",
            file=sys.stderr,
        )
        return 1
    # The report text is produced by the audit command's own printer, from the audits above.
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        _load_audit_command().print_report(bench, first, last, runs_meta[0]["slippage_bps"], audits)
    report_path = args.out / "audit_report.txt"
    report_path.write_text(report.getvalue())
    written += [bench_path, runs_path, audit_json, report_path]

    sums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(written, key=lambda p: p.name)
    )
    (args.out / "SHA256SUMS").write_text(sums)
    for path in sorted(written, key=lambda p: p.name):
        print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    print(f"wrote {args.out / 'SHA256SUMS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
