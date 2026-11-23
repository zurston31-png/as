"""issuers, point-in-time SIC and deal values

Revision ID: r0003
Revises: r0002
"""

import sqlalchemy as sa
from alembic import op

revision = "r0003"
down_revision = "r0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "security_issuers",
        sa.Column("security_id", sa.Integer(), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("how", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["security_id"], ["securities.security_id"]),
        sa.PrimaryKeyConstraint("security_id"),
    )
    op.create_table(
        "issuer_sic",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("security_id", sa.Integer(), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("sic", sa.String(length=4), nullable=True),
        sa.ForeignKeyConstraint(["security_id"], ["securities.security_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("issuer_sic", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_issuer_sic_security_id"), ["security_id"], unique=False
        )
    op.create_table(
        "deal_values",
        sa.Column("security_id", sa.Integer(), nullable=False),
        sa.Column("acquirer_symbol", sa.String(length=16), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("cash_per_share", sa.Float(), nullable=False),
        sa.Column("shares_per_share", sa.Float(), nullable=False),
        sa.Column("acquirer_session", sa.Date(), nullable=True),
        sa.Column("acquirer_close", sa.Float(), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("target_last_close", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["security_id"], ["securities.security_id"]),
        sa.PrimaryKeyConstraint("security_id"),
    )


def downgrade() -> None:
    op.drop_table("deal_values")
    with op.batch_alter_table("issuer_sic", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_issuer_sic_security_id"))
    op.drop_table("issuer_sic")
    op.drop_table("security_issuers")
