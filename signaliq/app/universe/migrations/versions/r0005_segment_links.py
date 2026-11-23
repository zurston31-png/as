"""trading-gap segment links

Revision ID: r0005
Revises: r0004
"""

import sqlalchemy as sa
from alembic import op

revision = "r0005"
down_revision = "r0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "segment_links",
        sa.Column("link_id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("earlier_last", sa.Date(), nullable=False),
        sa.Column("later_first", sa.Date(), nullable=False),
        sa.Column("gap_sessions", sa.Integer(), nullable=False),
        sa.Column("close_before", sa.Float(), nullable=True),
        sa.Column("close_after", sa.Float(), nullable=True),
        sa.Column("split_ratio", sa.Float(), nullable=False),
        sa.Column("issuer_before", sa.String(length=10), nullable=True),
        sa.Column("issuer_after", sa.String(length=10), nullable=True),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("link_id"),
    )


def downgrade() -> None:
    op.drop_table("segment_links")
