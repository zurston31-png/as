"""Tables for simulated strategy runs: runs, signals, trades, equity, parameter freezes.

Only adds tables. The market data tables are not touched.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

UTC_TEXT = sa.String(27)


def upgrade() -> None:
    op.create_table(
        "strategy_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("created_at", UTC_TEXT, nullable=False),
        sa.Column("finished_at", UTC_TEXT, nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("strategy", sa.String(48), nullable=False),
        sa.Column("strategy_version", sa.String(16), nullable=False),
        sa.Column("period", sa.String(16), nullable=False),
        sa.Column("period_start", sa.Date, nullable=False),
        sa.Column("period_end", sa.Date, nullable=False),
        sa.Column("symbols", sa.JSON, nullable=False),
        sa.Column("parameters", sa.JSON, nullable=False),
        sa.Column("parameters_hash", sa.String(64), nullable=False),
        sa.Column("starting_equity", sa.Float, nullable=False),
        sa.Column("cost_scenario", sa.String(24), nullable=False),
        sa.Column("slippage_bps", sa.Float, nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("feed", sa.String(16), nullable=False),
        sa.Column("adjustment", sa.String(16), nullable=False),
        sa.Column("bars_used", sa.Integer, nullable=False),
        sa.Column("data_first_session", sa.Date, nullable=True),
        sa.Column("data_last_session", sa.Date, nullable=True),
        sa.Column("data_fingerprint", sa.String(64), nullable=False),
        sa.Column("result_fingerprint", sa.String(64), nullable=True),
        sa.Column("signals_total", sa.Integer, nullable=False),
        sa.Column("signals_taken", sa.Integer, nullable=False),
        sa.Column("trades_total", sa.Integer, nullable=False),
        sa.Column("ending_equity", sa.Float, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index("ix_strategy_runs_strategy_period", "strategy_runs", ["strategy", "period"])

    op.create_table(
        "strategy_signals",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("strategy_runs.id"), nullable=False),
        sa.Column("strategy", sa.String(48), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("signal_session", sa.Date, nullable=False),
        sa.Column("signal_type", sa.String(16), nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("indicators", sa.JSON, nullable=False),
        sa.Column("rank_value", sa.Float, nullable=True),
        sa.Column("rank", sa.Integer, nullable=True),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("reason", sa.String(48), nullable=True),
        sa.Column("planned_shares", sa.Integer, nullable=True),
        sa.UniqueConstraint(
            "run_id", "symbol", "signal_session", "signal_type", name="uq_strategy_signals_key"
        ),
    )
    op.create_index("ix_strategy_signals_run_id", "strategy_signals", ["run_id"])

    op.create_table(
        "strategy_trades",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("strategy_runs.id"), nullable=False),
        sa.Column(
            "signal_id",
            sa.Integer,
            sa.ForeignKey("strategy_signals.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("strategy", sa.String(48), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("shares", sa.Integer, nullable=False),
        sa.Column("entry_session", sa.Date, nullable=False),
        sa.Column("entry_reference_price", sa.Float, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("equity_at_signal", sa.Float, nullable=False),
        sa.Column("initial_stop", sa.Float, nullable=False),
        sa.Column("risk_amount", sa.Float, nullable=False),
        sa.Column("exit_session", sa.Date, nullable=False),
        sa.Column("exit_reference_price", sa.Float, nullable=False),
        sa.Column("exit_price", sa.Float, nullable=False),
        sa.Column("exit_reason", sa.String(24), nullable=False),
        sa.Column("sessions_held", sa.Integer, nullable=False),
        sa.Column("gross_pnl", sa.Float, nullable=False),
        sa.Column("slippage_cost", sa.Float, nullable=False),
        sa.Column("net_pnl", sa.Float, nullable=False),
        sa.Column("r_multiple", sa.Float, nullable=False),
    )
    op.create_index("ix_strategy_trades_run_id", "strategy_trades", ["run_id"])

    op.create_table(
        "strategy_equity",
        sa.Column("run_id", sa.Integer, sa.ForeignKey("strategy_runs.id"), primary_key=True),
        sa.Column("session", sa.Date, primary_key=True),
        sa.Column("cash", sa.Float, nullable=False),
        sa.Column("positions_value", sa.Float, nullable=False),
        sa.Column("equity", sa.Float, nullable=False),
        sa.Column("open_positions", sa.Integer, nullable=False),
    )

    op.create_table(
        "strategy_parameter_freezes",
        sa.Column("strategy", sa.String(48), primary_key=True),
        sa.Column("strategy_version", sa.String(16), nullable=False),
        sa.Column("parameters", sa.JSON, nullable=False),
        sa.Column("parameters_hash", sa.String(64), nullable=False),
        sa.Column("frozen_at", UTC_TEXT, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("strategy_parameter_freezes")
    op.drop_table("strategy_equity")
    op.drop_index("ix_strategy_trades_run_id", table_name="strategy_trades")
    op.drop_table("strategy_trades")
    op.drop_index("ix_strategy_signals_run_id", table_name="strategy_signals")
    op.drop_table("strategy_signals")
    op.drop_index("ix_strategy_runs_strategy_period", table_name="strategy_runs")
    op.drop_table("strategy_runs")
