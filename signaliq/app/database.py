"""Database engine, session management and schema migrations (SQLite).

Create or upgrade the database from the command line with::

    python -m app.database

The schema is defined by the Alembic migrations in ``app/migrations``. They
are applied automatically; a database created before migrations existed is
recognised and upgraded without touching its data.
"""

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, create_engine, event, func, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import PROJECT_ROOT, ConfigurationError, get_settings
from app.models import Base

#: The market data tables (Milestone 1), in the order they are reported.
EXPECTED_TABLES: tuple[str, ...] = ("bars", "ingestion_runs", "data_quality_checks")

#: The strategy research tables (Milestone 2). They hold simulated results only.
STRATEGY_TABLES: tuple[str, ...] = (
    "strategy_runs",
    "strategy_signals",
    "strategy_trades",
    "strategy_equity",
    "strategy_parameter_freezes",
)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

#: The revision that describes the tables as Milestone 1 created them.
BASELINE_REVISION = "0001"

_VERSION_TABLE = "alembic_version"


class SchemaError(RuntimeError):
    """The database is in a state the migrations cannot safely continue from."""


_DATA = PROJECT_ROOT / "data"

#: Archived databases that no workflow may write (see research/plans/database_inventory.md).
#: Research databases open them read-only only; the market data writers refuse them.
PROTECTED_DATABASES: dict[Path, str] = {
    (_DATA / "universe_stage.db").resolve(): "the Stage 1.1 research database",
    (_DATA / "universe_stage_v1_0_tables.db").resolve(): "the Stage 1.0 tables",
    (_DATA / "signaliq.db").resolve(): "the IEX market data database",
    (_DATA / "sip_research.db").resolve(): "the validation batch v2 database",
    (_DATA / "phase1b_record" / "alpaca_responses.db").resolve(): (
        "the Phase 1b raw Alpaca responses (recorded 2026-11-27)"
    ),
    (_DATA / "phase1d_anomaly" / "alpaca_responses.db").resolve(): (
        "the raw Alpaca responses of the four approved anomaly requests (2026-11-27)"
    ),
}


class ProtectedDatabase(RuntimeError):
    """An archived database was about to be opened for writing."""


def protected_label(path: Path | None) -> str | None:
    return None if path is None else PROTECTED_DATABASES.get(path.resolve())


def refuse_protected(database_url: str, action: str = "write to") -> None:
    """Raise ProtectedDatabase when ``database_url`` names a protected database."""
    path = database_file_path(database_url)
    label = protected_label(path)
    if label:
        raise ProtectedDatabase(
            f"refusing to {action} {path}: it is {label}, archived and protected. "
            "Set SIGNALIQ_DATABASE_URL to a new database file."
        )


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


def _migration_config(connection: Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def head_revision() -> str:
    """Return the newest schema revision the code knows."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    head = ScriptDirectory.from_config(config).get_current_head()
    assert head is not None
    return head


def current_revision(engine: Engine) -> str | None:
    """Return the schema revision of the database (None before any migration)."""
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def init_db(engine: Engine) -> list[str]:
    """Bring the database schema up to date and return the tables now present.

    Safe to call repeatedly: when the schema is current nothing is done, and a
    migration never removes market data. A database created before migrations
    existed (the three Milestone 1 tables, no version marker) is marked as
    being at the baseline revision and upgraded from there.
    """
    head = head_revision()
    with engine.begin() as connection:
        if MigrationContext.configure(connection).get_current_revision() != head:
            refuse_protected(str(engine.url), "migrate")
            config = _migration_config(connection)
            tables = set(inspect(connection).get_table_names())
            if _VERSION_TABLE not in tables and tables:
                missing = [name for name in EXPECTED_TABLES if name not in tables]
                if missing:
                    raise SchemaError(
                        "The database has tables but no schema version, and it is not a "
                        f"complete SignalIQ database (missing: {', '.join(missing)}). "
                        "Refusing to migrate it."
                    )
                command.stamp(config, BASELINE_REVISION)
            command.upgrade(config, "head")
    return sorted(name for name in inspect(engine).get_table_names() if name != _VERSION_TABLE)


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


def table_row_counts(engine: Engine, tables: tuple[str, ...] = EXPECTED_TABLES) -> dict[str, int]:
    """Return the number of rows in each of ``tables`` (the market data tables by default)."""
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for name in tables:
            counts[name] = connection.execute(
                select(func.count()).select_from(Base.metadata.tables[name])
            ).scalar_one()
    return counts


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: create the database and its tables."""
    parser = argparse.ArgumentParser(
        prog="python -m app.database",
        description=(
            "Create the SignalIQ SQLite database or upgrade its schema "
            "(safe to run more than once)."
        ),
    )
    parser.parse_args(argv)

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    engine = create_db_engine(settings.database_url)
    try:
        before = current_revision(engine)
        tables = init_db(engine)
        after = current_revision(engine)
        counts = table_row_counts(engine, (*EXPECTED_TABLES, *STRATEGY_TABLES))
    except SchemaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.dispose()

    location = database_file_path(settings.database_url) or "in-memory database"
    print(f"Database ready: {location}")
    if before == after:
        print(f"Schema revision: {after} (already up to date)")
    else:
        print(f"Schema revision: {after} (upgraded from {before or 'no version'})")
    for table in (*EXPECTED_TABLES, *STRATEGY_TABLES):
        state = f"{counts[table]:>10,} rows" if table in tables else "   MISSING"
        print(f"  {table:<27} {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
