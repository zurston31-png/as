"""Alembic environment for SignalIQ.

The application runs migrations itself (see ``app.database.init_db``) and
passes its own connection in. Run from the command line (``alembic upgrade
head``) this file opens the database named by the SignalIQ settings.
"""

from alembic import context

from app.models import Base

target_metadata = Base.metadata


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # SQLite cannot alter most things in place; batch mode rebuilds the table.
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return

    from app.config import get_settings
    from app.database import create_db_engine

    engine = create_db_engine(get_settings().database_url)
    try:
        with engine.connect() as own_connection:
            _run(own_connection)
            own_connection.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    raise SystemExit("Offline (SQL script) migrations are not supported; run against a database.")
run_migrations()
