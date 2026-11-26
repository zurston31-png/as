"""The SIP research database: its own file, its own migrations, its own version table."""

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, inspect

from app.config import PROJECT_ROOT
from app.database import create_db_engine, database_file_path
from app.universe import sec_cache

DEFAULT_URL = f"sqlite:///{(PROJECT_ROOT / 'data' / 'sip_research.db').as_posix()}"
URL_VARIABLE = "SIGNALIQ_SIP_RESEARCH_DATABASE_URL"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
VERSION_TABLE = "sip_research_alembic_version"
READ_ONLY_VARIABLE = "SIGNALIQ_SIP_RESEARCH_READ_ONLY"

#: Archived databases that no workflow may write. They can be opened read-only only
#: (``read_only=True`` or SIGNALIQ_SIP_RESEARCH_READ_ONLY=1), which SQLite itself enforces.
PROTECTED_DATABASES: dict[Path, str] = {
    (PROJECT_ROOT / "data" / "universe_stage.db").resolve(): "the Stage 1.1 research database",
    (PROJECT_ROOT / "data" / "universe_stage_v1_0_tables.db").resolve(): "the Stage 1.0 tables",
    (PROJECT_ROOT / "data" / "signaliq.db").resolve(): "the IEX market data database",
}


class ProtectedDatabase(RuntimeError):
    """An archived database was about to be opened for writing."""


def protected_label(path: Path | None) -> str | None:
    return None if path is None else PROTECTED_DATABASES.get(path.resolve())


def _read_only_requested() -> bool:
    return os.environ.get(READ_ONLY_VARIABLE, "").strip() == "1"


def _open_read_only(path: Path) -> Engine:
    """An engine whose every connection is read-only at the SQLite level (``mode=ro``)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    engine = create_engine(
        f"sqlite:///file:{path.resolve().as_posix()}?mode=ro&uri=true",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    with engine.connect() as connection:
        config = _config(connection)
        head = ScriptDirectory.from_config(config).get_current_head()
        current = MigrationContext.configure(
            connection, opts={"version_table": VERSION_TABLE}
        ).get_current_revision()
    if current != head:
        engine.dispose()
        raise RuntimeError(f"{path} is at schema {current}, not {head}; read-only, so not upgraded")
    return engine


def research_database_url() -> str:
    return os.environ.get(URL_VARIABLE, "").strip() or DEFAULT_URL


def _config(connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def open_research_database(url: str | None = None, read_only: bool | None = None) -> Engine:
    """Open (and create or upgrade) the research database. Refuses the market data database.

    A protected database (``PROTECTED_DATABASES``) is opened only read-only, and only when
    that is asked for; otherwise ``ProtectedDatabase`` is raised before anything is opened.
    A read-only parent cache (``sec_cache``) is attached from SIGNALIQ_SIP_RESEARCH_PARENT_DB.
    """
    url = url or research_database_url()
    path = database_file_path(url)
    read_only = _read_only_requested() if read_only is None else read_only
    label = protected_label(path)
    if label and not read_only:
        raise ProtectedDatabase(
            f"{path} is {label}, archived and protected: it can be opened read-only only "
            f"({READ_ONLY_VARIABLE}=1). Work on a new database instead."
        )
    sec_cache.configure_from_environment()
    parent = sec_cache.parent_path()
    if parent is not None and path is not None and parent == path.resolve():
        raise RuntimeError("the parent cache must be another database than the one opened")
    if read_only:
        if path is None:
            raise ValueError("read-only mode needs a database file")
        return _open_read_only(path)
    engine = create_db_engine(url)
    tables = set(inspect(engine).get_table_names())
    if "bars" in tables or "alembic_version" in tables:
        engine.dispose()
        raise RuntimeError(
            f"{url} is the market data database, not the SIP research database. "
            "The two are kept apart on purpose."
        )
    with engine.begin() as connection:
        config = _config(connection)
        head = ScriptDirectory.from_config(config).get_current_head()
        current = MigrationContext.configure(
            connection, opts={"version_table": VERSION_TABLE}
        ).get_current_revision()
        if current != head:
            command.upgrade(config, "head")
    return engine
