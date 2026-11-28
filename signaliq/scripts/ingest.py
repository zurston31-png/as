#!/usr/bin/env python3
"""Download historical bars from Alpaca into the local SignalIQ database.

Run from the project folder, for example:

    python scripts/ingest.py --symbol AAPL
    python scripts/ingest.py --symbol AAPL --symbol MSFT --start 2020-01-01
    python scripts/ingest.py --all

Running the same command again is safe: bars that are already stored are
skipped, and nothing is duplicated or overwritten.

Exit codes: 0 = every run finished, 1 = at least one run failed,
2 = the command or the configuration is invalid.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Allow running this file directly (python scripts/ingest.py) without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigurationError, Settings, configure_logging, get_settings
from app.data.ingestion import BatchIngestionResult, IngestionError, run_ingestion
from app.database import ProtectedDatabase, database_file_path
from app.models import RunStatus

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a date; use the form YYYY-MM-DD, for example 2016-01-01"
        ) from None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/ingest.py",
        description="Download historical stock bars from Alpaca and store them in SQLite.",
        epilog=(
            "Examples:\n"
            "  python scripts/ingest.py --symbol AAPL\n"
            "  python scripts/ingest.py --symbol AAPL --symbol MSFT --start 2020-01-01\n"
            "  python scripts/ingest.py --all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--symbol",
        "-s",
        action="append",
        metavar="SYMBOL",
        help="symbol to download; repeat the option or separate symbols with commas",
    )
    target.add_argument(
        "--all", action="store_true", help="download every symbol of the configured universe"
    )
    parser.add_argument(
        "--start",
        type=_date,
        metavar="YYYY-MM-DD",
        help="first date to request (default: the configured history start, 2016-01-01)",
    )
    parser.add_argument(
        "--end",
        type=_date,
        metavar="YYYY-MM-DD",
        help=(
            "last date to request (default: the last completed trading session); "
            "a candle that is still forming is never requested"
        ),
    )
    parser.add_argument(
        "--timeframe", help="bar timeframe (default: 1Day, the only one in Milestone 1)"
    )
    parser.add_argument("--log-level", choices=LOG_LEVELS, help="logging detail (default: INFO)")
    return parser.parse_args(argv)


def _symbols(values: list[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def print_summary(batch: BatchIngestionResult, settings: Settings) -> None:
    """Print one line per symbol plus any warnings and errors."""
    location = database_file_path(settings.database_url) or "in-memory database"
    print()
    print(
        f"Ingestion summary  feed={settings.alpaca_feed}  "
        f"adjustment={settings.alpaca_adjustment}  database={location}"
    )
    print()
    header = (
        f"{'SYMBOL':<8}{'STATUS':<24}{'FETCHED':>9}{'INSERTED':>10}{'UNCHANGED':>11}"
        f"{'CONFLICTS':>11}{'REJECTED':>10}{'DUPLICATES':>12}{'MISSING':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in batch.results:
        print(
            f"{result.symbol:<8}{result.status:<24}{result.bars_fetched:>9,}"
            f"{result.bars_inserted:>10,}{result.bars_unchanged:>11,}"
            f"{result.bars_conflicting:>11,}{result.bars_rejected:>10,}"
            f"{result.bars_duplicate:>12,}{result.missing_sessions:>9,}"
        )

    warnings = [
        (result.symbol, message) for result in batch.results for message in result.warning_messages
    ]
    if warnings:
        print("\nWarnings:")
        for symbol, message in warnings:
            print(f"  {symbol}: {message}")

    errors = [(result.symbol, result.error) for result in batch.results if result.error]
    if errors:
        print("\nErrors:")
        for symbol, message in errors:
            print(f"  {symbol}: {message}")
    if batch.not_attempted:
        print(
            "\nNot attempted because the error above would affect them too: "
            + ", ".join(batch.not_attempted)
        )

    if any(result.status == RunStatus.COMPLETED_WITH_ISSUES for result in batch.results):
        print(
            "\nSome bars were rejected, conflicted with stored data or are missing. Nothing was "
            "repaired or overwritten.\nSee the details with:  python scripts/validate.py   "
            "or   GET /quality/report"
        )
    print(
        "\nMISSING counts expected trading sessions without a bar in the requested window "
        "(weekends and holidays excluded)."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(args.log_level or settings.log_level)

    symbols = list(settings.symbols) if args.all else _symbols(args.symbol)
    try:
        batch = run_ingestion(
            settings, symbols, timeframe=args.timeframe, start=args.start, end=args.end
        )
    except (ConfigurationError, IngestionError, ProtectedDatabase) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print_summary(batch, settings)
    return 0 if batch.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
