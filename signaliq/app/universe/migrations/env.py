"""Alembic environment of the SIP research database (separate version table)."""

from alembic import context

from app.universe.models import ResearchBase

connection = context.config.attributes["connection"]
context.configure(
    connection=connection,
    target_metadata=ResearchBase.metadata,
    version_table="sip_research_alembic_version",
    render_as_batch=True,
)
with context.begin_transaction():
    context.run_migrations()
