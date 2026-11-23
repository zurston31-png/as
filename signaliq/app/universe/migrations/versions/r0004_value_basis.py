"""value basis of an exit value

Revision ID: r0004
Revises: r0003
"""

import sqlalchemy as sa
from alembic import op

revision = "r0004"
down_revision = "r0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("securities", schema=None) as batch_op:
        batch_op.add_column(sa.Column("value_basis", sa.String(length=16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("securities", schema=None) as batch_op:
        batch_op.drop_column("value_basis")
