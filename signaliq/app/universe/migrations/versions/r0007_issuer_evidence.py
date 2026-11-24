"""dated issuer evidence (point-in-time issuer of each ticker period)

Revision ID: r0007
Revises: r0006
"""

import sqlalchemy as sa
from alembic import op

revision = "r0007"
down_revision = "r0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "issuer_evidence",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("security_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("filed", sa.Date(), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("how", sa.String(length=32), nullable=False),
        sa.Column("url", sa.String(length=300), nullable=False),
        sa.ForeignKeyConstraint(["security_id"], ["securities.security_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_issuer_evidence_security_id", "issuer_evidence", ["security_id"])


def downgrade() -> None:
    op.drop_index("ix_issuer_evidence_security_id", table_name="issuer_evidence")
    op.drop_table("issuer_evidence")
