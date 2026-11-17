#!/usr/bin/env python3
"""Re-validate the bars stored in the SignalIQ database.

Run from the project folder, for example:

    python scripts/validate.py
    python scripts/validate.py --symbol AAPL

Every stored bar is checked again (prices, volume, timestamps, symbols,
metadata) and each symbol is checked for missing trading sessions. Nothing is
changed or repaired: the outcome is recorded in the data_quality_checks table
and printed here.

Exit codes: 0 = no failed check, 1 = at least one check failed,
2 = the command or the configuration is invalid.
"""

import argparse
import sys
from pathlib import Path

# Allow running this file directly (python scripts/validate.py) without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (  # noqa: E402
    ConfigurationError,
    configure_logging,
    get_settings,
    is_valid_symbol,
    is_valid_timeframe,
)
from app.data.validation import (  # noqa: E402
    CHECK_DESCRIPTIONS,
    Finding,
    StoredValidationSummary,
    validate_stored_bars,
)
from app.database import (  # noqa: E402
    create_db_engine,
    create_session_factory,
    database_file_path,
    init_db,
    session_scope,
)
from app.models import CheckStatus, format_utc  # noqa: E402

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/validate.py",
        description="Re-run every data-quality check over the bars stored in SQLite.",
    )
    parser.add_argument(
        "--symbol",
        "-s",
        action="append",
        metavar="SYMBOL",
        help="only validate this symbol; repeat the option or separate symbols with commas",
    )
    parser.add_argument("--timeframe", help="only validate this timeframe, for example 1Day")
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        metavar="N",
        help="how many failures and warnings to print (default: 20; all are stored)",
    )
    parser.add_argument("--log-level", choices=LOG_LEVELS, help="logging detail (default: INFO)")
    return parser.parse_args(argv)


def _describe(finding: Finding) -> str:
    where = " ".join(part for part in (finding.symbol, finding.timeframe) if part)
    when = f" {format_utc(finding.bar_timestamp)}" if finding.bar_timestamp else ""
    return f"  [{finding.check_name}] {where}{when} - {finding.message}"


def print_report(summary: StoredValidationSummary, max_issues: int) -> None:
    print()
    print(
        f"Validated {summary.bars_checked:,} stored bar(s) in {len(summary.groups)} "
        "symbol/timeframe group(s)."
    )
    print(f"Check run id: {summary.check_run_id} (recorded in data_quality_checks)")
    print()

    if not summary.groups:
        print("Nothing to validate: no bars are stored yet. Download some with scripts/ingest.py.")
        return

    header = f"{'CHECK':<28}{'PASSED':>8}{'FAILED':>8}{'WARNING':>9}{'SKIPPED':>9}"
    print(header)
    print("-" * len(header))
    for check_name in CHECK_DESCRIPTIONS:
        counts = summary.counts.get(check_name)
        if counts is None:
            continue
        print(
            f"{check_name:<28}{counts.get(CheckStatus.PASSED, 0):>8,}"
            f"{counts.get(CheckStatus.FAILED, 0):>8,}{counts.get(CheckStatus.WARNING, 0):>9,}"
            f"{counts.get(CheckStatus.SKIPPED, 0):>9,}"
        )

    for title, status in (("Failed checks", CheckStatus.FAILED), ("Warnings", CheckStatus.WARNING)):
        findings = [finding for finding in summary.findings if finding.status == status]
        if not findings:
            continue
        shown = findings[: max(max_issues, 0)]
        print(f"\n{title} (showing {len(shown)} of {len(findings)}):")
        for finding in shown:
            print(_describe(finding))

    print()
    if summary.failed:
        print(
            f"RESULT: FAILED - {summary.failed} problem(s) found, {summary.warnings} warning(s). "
            "Nothing was changed or repaired."
        )
    else:
        print(f"RESULT: PASSED - no problems found, {summary.warnings} warning(s).")
    print(
        "PASSED/SKIPPED count one row per check and symbol; FAILED/WARNING count individual "
        "findings."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(args.log_level or settings.log_level)

    symbols: list[str] | None = None
    if args.symbol:
        symbols = [
            part.strip().upper() for value in args.symbol for part in value.split(",") if part.strip()
        ]
        invalid = [symbol for symbol in symbols if not is_valid_symbol(symbol)]
        if invalid:
            print(f"Error: invalid symbol(s): {', '.join(invalid)}", file=sys.stderr)
            return 2
    if args.timeframe is not None and not is_valid_timeframe(args.timeframe):
        print(
            f"Error: invalid timeframe {args.timeframe!r}; examples: 1Min, 1Hour, 1Day",
            file=sys.stderr,
        )
        return 2

    path = database_file_path(settings.database_url)
    if path is not None and not path.exists():
        print(
            f"Error: database not found at {path}\n"
            "Create it with:  python -m app.database   (then download data with "
            "scripts/ingest.py)",
            file=sys.stderr,
        )
        return 2

    engine = create_db_engine(settings.database_url)
    try:
        init_db(engine)
        with session_scope(create_session_factory(engine)) as session:
            summary = validate_stored_bars(
                session, settings, symbols=symbols, timeframe=args.timeframe
            )
    finally:
        engine.dispose()

    print_report(summary, args.max_issues)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
