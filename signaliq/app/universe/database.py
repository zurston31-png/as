"""The SIP research database: its own file, its own migrations, its own version table."""

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect

from app.config import PROJECT_ROOT
from app.database import create_db_engine

DEFAULT_URL = f"sqlite:///{(PROJECT_ROOT / 'data' / 'sip_research.db').as_posix()}"
URL_VARIABLE = "SIGNALIQ_SIP_RESEARCH_DATABASE_URL"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
VERSION_TABLE = "sip_research_alembic_version"


def research_database_url() -> str:
    return os.environ.get(URL_VARIABLE, "").strip() or DEFAULT_URL


def _config(connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def open_research_database(url: str | None = None) -> Engine:
    """Open (and create or upgrade) the research database. Refuses the market data database."""
    url = url or research_database_url()
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
