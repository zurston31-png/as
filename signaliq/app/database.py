"""Database engine, session management and schema initialisation (SQLite).

Initialise the database from the command line with::

    python -m app.database
"""

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, func, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import ConfigurationError, get_settings
from app.models import Base

#: The tables this milestone creates, in the order they are reported.
EXPECTED_TABLES: tuple[str, ...] = ("bars", "ingestion_runs", "data_quality_checks")


def database_file_path(database_url: str) -> Path | None:
    """Return the SQLite file behind ``database_url`` (None for in-memory)."""
    database = make_url(database_url).database
    if not database or database == ":memory:":
        return None
    return Path(database).expanduser()


def create_db_engine(database_url: str) -> Engine:
    """Create the SQLAlchemy engine, creating the database folder if needed."""
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        raise ValueError("Only SQLite databases are supported in Milestone 1.")

    path = database_file_path(database_url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        url,
        # FastAPI serves requests from a thread pool; each request gets its own
        # session, so sharing the connection pool across threads is safe.
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> list[str]:
    """Create any missing tables and return the table names now present.

    Safe to call repeatedly: existing tables and their data are left untouched.
    """
    Base.metadata.create_all(engine)
    return sorted(inspect(engine).get_table_names())


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a transactional scope: commit on success, roll back on error."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def table_row_counts(engine: Engine) -> dict[str, int]:
    """Return the number of rows in each SignalIQ table."""
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for name in EXPECTED_TABLES:
            counts[name] = connection.execute(
                select(func.count()).select_from(Base.metadata.tables[name])
            ).scalar_one()
    return counts


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: create the database and its tables."""
    parser = argparse.ArgumentParser(
        prog="python -m app.database",
        description="Initialise the SignalIQ SQLite database (safe to run more than once).",
    )
    parser.parse_args(argv)

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    engine = create_db_engine(settings.database_url)
    try:
        tables = init_db(engine)
        counts = table_row_counts(engine)
    finally:
        engine.dispose()

    location = database_file_path(settings.database_url) or "in-memory database"
    print(f"Database ready: {location}")
    for table in EXPECTED_TABLES:
        state = f"{counts[table]:>10,} rows" if table in tables else "   MISSING"
        print(f"  {table:<22} {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
