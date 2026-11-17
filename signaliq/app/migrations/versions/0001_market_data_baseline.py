"""Market data tables as created by Milestone 1.

This is the schema exactly as ``Base.metadata.create_all`` built it before
migrations existed. A database from that time already has these tables and is
marked as being at this revision instead of running it.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

#: Stored form of the application's UTC timestamp type (ISO-8601 text ending in Z).
UTC_TEXT = sa.String(27)


def upgrade() -> None:
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("started_at", UTC_TEXT, nullable=False),
        sa.Column("finished_at", UTC_TEXT, nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("feed", sa.String(16), nullable=False),
        sa.Column("adjustment", sa.String(16), nullable=False),
        sa.Column("currency", sa.String(8), nullable=False),
        sa.Column("requested_start", UTC_TEXT, nullable=False),
        sa.Column("requested_end", UTC_TEXT, nullable=False),
        sa.Column("pages_fetched", sa.Integer, nullable=False),
        sa.Column("bars_fetched", sa.Integer, nullable=False),
        sa.Column("bars_inserted", sa.Integer, nullable=False),
        sa.Column("bars_unchanged", sa.Integer, nullable=False),
        sa.Column("bars_conflicting", sa.Integer, nullable=False),
        sa.Column("bars_rejected", sa.Integer, nullable=False),
        sa.Column("bars_duplicate", sa.Integer, nullable=False),
        sa.Column("missing_sessions", sa.Integer, nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index("ix_ingestion_runs_symbol_started", "ingestion_runs", ["symbol", "started_at"])

    op.create_table(
        "bars",
        sa.Column("symbol", sa.String(16), primary_key=True),
        sa.Column("timeframe", sa.String(16), primary_key=True),
        sa.Column("timestamp", UTC_TEXT, primary_key=True),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("volume", sa.BigInteger, nullable=False),
        sa.Column("vwap", sa.Float, nullable=False),
        sa.Column("trade_count", sa.BigInteger, nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("feed", sa.String(16), nullable=False),
        sa.Column("adjustment", sa.String(16), nullable=False),
        sa.Column(
            "ingestion_run_id", sa.Integer, sa.ForeignKey("ingestion_runs.id"), nullable=True
        ),
        sa.Column("ingested_at", UTC_TEXT, nullable=False),
    )
    op.create_index("ix_bars_ingestion_run_id", "bars", ["ingestion_run_id"])

    op.create_table(
        "data_quality_checks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("created_at", UTC_TEXT, nullable=False),
        sa.Column("check_run_id", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(16), nullable=False),
        sa.Column(
            "ingestion_run_id", sa.Integer, sa.ForeignKey("ingestion_runs.id"), nullable=True
        ),
        sa.Column("check_name", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=True),
        sa.Column("timeframe", sa.String(16), nullable=True),
        sa.Column("bar_timestamp", UTC_TEXT, nullable=True),
        sa.Column("records_checked", sa.Integer, nullable=True),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("details", sa.JSON, nullable=True),
    )
    op.create_index("ix_data_quality_checks_check_run_id", "data_quality_checks", ["check_run_id"])
    op.create_index(
        "ix_data_quality_checks_ingestion_run_id", "data_quality_checks", ["ingestion_run_id"]
    )
    op.create_index("ix_data_quality_checks_status", "data_quality_checks", ["status"])
    op.create_index(
        "ix_data_quality_checks_symbol_check", "data_quality_checks", ["symbol", "check_name"]
    )


def downgrade() -> None:
    raise NotImplementedError(
        "The market data tables are the baseline and are never dropped by a migration."
    )
