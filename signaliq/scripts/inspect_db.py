#!/usr/bin/env python3
"""Show statistics about the SignalIQ database.

Run from the project folder, for example:

    python scripts/inspect_db.py
    python scripts/inspect_db.py --json

This command only reads. It prints how many rows each table holds, the stored
date range per symbol, the ingestion history and the data-quality outcomes.

Exit codes: 0 = statistics printed, 2 = the configuration is invalid or the
database does not exist yet.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running this file directly (python scripts/inspect_db.py) without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import Engine, func, select  # noqa: E402

from app.config import ConfigurationError, Settings, get_settings  # noqa: E402
from app.database import create_db_engine, database_file_path, init_db, table_row_counts  # noqa: E402
from app.models import Bar, CheckStage, DataQualityCheck, IngestionRun  # noqa: E402

LATEST_RUNS_SHOWN = 5


def _iso(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ") if value is not None else None


def collect_statistics(engine: Engine, settings: Settings) -> dict[str, Any]:
    """Gather the statistics as plain data (also used for --json)."""
    path = database_file_path(settings.database_url)
    with engine.connect() as connection:
        coverage = connection.execute(
            select(
                Bar.symbol,
                Bar.timeframe,
                func.count().label("bar_count"),
                func.min(Bar.timestamp).label("first_timestamp"),
                func.max(Bar.timestamp).label("last_timestamp"),
            )
            .group_by(Bar.symbol, Bar.timeframe)
            .order_by(Bar.symbol, Bar.timeframe)
        ).all()
        provenance = connection.execute(
            select(Bar.source, Bar.feed, Bar.adjustment, func.count().label("bar_count"))
            .group_by(Bar.source, Bar.feed, Bar.adjustment)
            .order_by(Bar.source, Bar.feed, Bar.adjustment)
        ).all()
        runs_by_status = connection.execute(
            select(IngestionRun.status, func.count())
            .group_by(IngestionRun.status)
            .order_by(IngestionRun.status)
        ).all()
        latest_runs = connection.execute(
            select(
                IngestionRun.id,
                IngestionRun.started_at,
                IngestionRun.symbol,
                IngestionRun.timeframe,
                IngestionRun.status,
                IngestionRun.bars_fetched,
                IngestionRun.bars_inserted,
                IngestionRun.bars_conflicting,
                IngestionRun.bars_rejected,
                IngestionRun.missing_sessions,
            )
            .order_by(IngestionRun.id.desc())
            .limit(LATEST_RUNS_SHOWN)
        ).all()
        checks_by_status = connection.execute(
            select(DataQualityCheck.status, func.count())
            .group_by(DataQualityCheck.status)
            .order_by(DataQualityCheck.status)
        ).all()
        newest_validation = connection.execute(
            select(DataQualityCheck.check_run_id, DataQualityCheck.created_at)
            .where(DataQualityCheck.stage == CheckStage.STORED)
            .order_by(DataQualityCheck.id.desc())
            .limit(1)
        ).first()
        latest_validation: dict[str, Any] | None = None
        if newest_validation is not None:
            counts = connection.execute(
                select(DataQualityCheck.status, func.count())
                .where(DataQualityCheck.check_run_id == newest_validation.check_run_id)
                .group_by(DataQualityCheck.status)
            ).all()
            latest_validation = {
                "check_run_id": newest_validation.check_run_id,
                "created_at": _iso(newest_validation.created_at),
                "by_status": {status: count for status, count in counts},
            }

    stored_symbols = {row.symbol for row in coverage}
    return {
        "database": {
            "path": str(path) if path is not None else None,
            "size_bytes": path.stat().st_size if path is not None and path.exists() else None,
        },
        "tables": table_row_counts(engine),
        "bars": [
            {
                "symbol": row.symbol,
                "timeframe": row.timeframe,
                "bar_count": row.bar_count,
                "first_timestamp": _iso(row.first_timestamp),
                "last_timestamp": _iso(row.last_timestamp),
                "in_universe": row.symbol in settings.symbols,
            }
            for row in coverage
        ],
        "provenance": [
            {
                "source": row.source,
                "feed": row.feed,
                "adjustment": row.adjustment,
                "bar_count": row.bar_count,
            }
            for row in provenance
        ],
        "universe": {
            "configured": list(settings.symbols),
            "without_data": [symbol for symbol in settings.symbols if symbol not in stored_symbols],
        },
        "ingestion_runs": {
            "by_status": {status: count for status, count in runs_by_status},
            "latest": [
                {
                    "id": row.id,
                    "started_at": _iso(row.started_at),
                    "symbol": row.symbol,
                    "timeframe": row.timeframe,
                    "status": row.status,
                    "bars_fetched": row.bars_fetched,
                    "bars_inserted": row.bars_inserted,
                    "bars_conflicting": row.bars_conflicting,
                    "bars_rejected": row.bars_rejected,
                    "missing_sessions": row.missing_sessions,
                }
                for row in latest_runs
            ],
        },
        "quality_checks": {
            "by_status": {status: count for status, count in checks_by_status},
            "latest_stored_validation": latest_validation,
        },
    }


def _by_status(counts: dict[str, int]) -> str:
    return ", ".join(f"{status}: {count:,}" for status, count in counts.items()) or "none"


def format_statistics(stats: dict[str, Any]) -> str:
    """Render the statistics as readable text."""
    lines: list[str] = []
    database = stats["database"]
    size = database["size_bytes"]
    lines.append(f"Database: {database['path'] or 'in-memory database'}")
    if size is not None:
        lines.append(f"Size:     {size:,} bytes ({size / 1_048_576:.2f} MB)")

    lines.append("\nTables")
    for table, count in stats["tables"].items():
        lines.append(f"  {table:<22}{count:>12,} rows")

    lines.append("\nBars by symbol and timeframe")
    if stats["bars"]:
        lines.append(f"  {'SYMBOL':<8}{'TIMEFRAME':<11}{'BARS':>10}  {'FIRST (UTC)':<22}{'LAST (UTC)':<22}")
        for row in stats["bars"]:
            note = "" if row["in_universe"] else "  <- not in the configured universe"
            lines.append(
                f"  {row['symbol']:<8}{row['timeframe']:<11}{row['bar_count']:>10,}  "
                f"{row['first_timestamp']:<22}{row['last_timestamp']:<22}{note}"
            )
    else:
        lines.append("  (no bars stored yet)")
    if stats["universe"]["without_data"]:
        lines.append(
            "  Configured symbols without data: " + ", ".join(stats["universe"]["without_data"])
        )

    if stats["provenance"]:
        lines.append("\nBars by source, feed and adjustment")
        for row in stats["provenance"]:
            lines.append(
                f"  source={row['source']}  feed={row['feed']}  adjustment={row['adjustment']}"
                f"{row['bar_count']:>12,} bars"
            )

    runs = stats["ingestion_runs"]
    lines.append(f"\nIngestion runs: {_by_status(runs['by_status'])}")
    if runs["latest"]:
        lines.append(f"  Latest {len(runs['latest'])}:")
        for run in runs["latest"]:
            lines.append(
                f"  #{run['id']:<5}{run['started_at']}  {run['symbol']:<7}{run['timeframe']:<6}"
                f"{run['status']:<23}fetched={run['bars_fetched']:,} inserted={run['bars_inserted']:,} "
                f"conflicts={run['bars_conflicting']:,} rejected={run['bars_rejected']:,} "
                f"missing={run['missing_sessions']:,}"
            )

    checks = stats["quality_checks"]
    lines.append(f"\nData-quality check rows: {_by_status(checks['by_status'])}")
    latest = checks["latest_stored_validation"]
    if latest is not None:
        lines.append(
            f"  Latest validation of stored bars ({latest['created_at']}): "
            f"{_by_status(latest['by_status'])}"
        )
    else:
        lines.append("  Stored bars have not been validated yet (run: python scripts/validate.py)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/inspect_db.py",
        description="Show statistics about the SignalIQ SQLite database (read-only).",
    )
    parser.add_argument("--json", action="store_true", help="print the statistics as JSON")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    path = database_file_path(settings.database_url)
    if path is not None and not path.exists():
        print(
            f"Error: database not found at {path}\nCreate it with:  python -m app.database",
            file=sys.stderr,
        )
        return 2

    engine = create_db_engine(settings.database_url)
    try:
        init_db(engine)
        stats = collect_statistics(engine, settings)
    finally:
        engine.dispose()

    print(json.dumps(stats, indent=2) if args.json else format_statistics(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
