"""permanent security registry, monthly screening bars, candidate pool

Revision ID: r0006
Revises: r0005
"""

import sqlalchemy as sa
from alembic import op

revision = "r0006"
down_revision = "r0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "security_registry",
        sa.Column("security_key", sa.String(length=48), nullable=False),
        sa.Column("security_id", sa.Integer(), nullable=False),
        sa.Column("first_symbol", sa.String(length=16), nullable=False),
        sa.Column("first_session", sa.Date(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("merged_into", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("security_key"),
        sa.UniqueConstraint("security_id"),
    )
    op.create_table(
        "monthly_bars",
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("month", sa.Date(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("vwap", sa.Float(), nullable=False),
        sa.Column("trade_count", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "month"),
    )
    op.create_table(
        "candidates",
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("sources", sa.String(length=160), nullable=False),
        sa.Column("months_with_trades", sa.Integer(), nullable=False),
        sa.Column("best_rank", sa.Integer(), nullable=True),
        sa.Column("best_month", sa.Date(), nullable=True),
        sa.Column("selection", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("symbol"),
    )


def downgrade() -> None:
    op.drop_table("candidates")
    op.drop_table("monthly_bars")
    op.drop_table("security_registry")
