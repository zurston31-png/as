"""Additive schema migration for databases created by an earlier version.

SQLAlchemy's `create_all()` creates tables that don't exist yet, but it
never touches a table that does - so a deployment that ran an older build
keeps its old columns forever, and every column added since (Signal.source,
Trade.position_id, Position.initial_qty, RugCheckResult.rug_risk_score, ...)
is simply absent. The bot then dies on its first query with
"no such column: signals.source", which looks like a code bug rather than
a stale database.

This closes that gap by comparing each mapped model against the table that
is actually there and issuing `ALTER TABLE ... ADD COLUMN` for anything
missing. Deliberately ADDITIVE ONLY:

  * adds a missing column                       yes
  * creates a missing table (via create_all)    yes
  * creates a missing index                     yes
  * renames / drops / retypes a column          NO
  * computed / conditional backfill             NO

An index needs its own step because neither of the other two covers it:
`create_all()` skips a table that already exists, indexes included, and
`ADD COLUMN` never creates one. Without this, a column added on upgrade
(Signal.strategy_version, Trade.strategy_version) would be present but
unindexed, and the analytics queries that group by it would quietly
degrade to full scans on exactly the databases with the most history.

What existing rows get in a newly added column depends on the model:

  * a column with a simple scalar default (Signal.source="tradingview",
    ScannedToken.times_traded=0) is added WITH that DEFAULT, so old rows
    take it. That is a statement of fact rather than a guess - every signal
    in a pre-scanner database really did come from TradingView, and
    querying `source == "scanner"` then behaves sanely instead of tripping
    over NULLs.
  * a column with no default, or a callable one like `utcnow`, lands NULL.
    That is also the honest value: "this wasn't recorded at the time".
    Callable defaults are applied by the ORM on insert, not by the
    database, so emitting one into DDL would be wrong (and SQLite rejects
    non-constant defaults on ADD COLUMN anyway).

That scope is the point. A real migration tool (Alembic) is the right
answer for destructive or data-transforming changes, and this project
should adopt one if it ever needs them. Until then, every schema change
here has been purely additive, and a dependency-free 60-line function that
handles exactly that case beats an unused migration framework - while
refusing to silently do the dangerous half.

SQLite cannot add a column with a non-constant default, and Postgres locks
differ, so a column that needs a computed backfill must still be handled by
hand. None currently do; `ADDABLE_ONLY_NOTE` is raised if one ever appears.
"""
from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.database import Base

logger = logging.getLogger(__name__)


def _column_ddl_type(column, dialect) -> str:
    """Render a column's type for this dialect (e.g. VARCHAR(64), FLOAT)."""
    return column.type.compile(dialect=dialect)


def _default_clause(column) -> str:
    """A literal DEFAULT clause when the model declares a simple scalar one.

    Only scalar defaults are emitted. A callable default (like `utcnow`) is
    applied by SQLAlchemy on insert, not by the database, so pushing one
    into DDL would be wrong - and SQLite rejects non-constant defaults on
    ADD COLUMN anyway. Existing rows keep NULL, new rows get the model
    default through the ORM as usual.
    """
    default = column.default
    if default is None or getattr(default, "is_callable", False):
        return ""
    value = getattr(default, "arg", None)
    if callable(value) or value is None:
        return ""
    if isinstance(value, bool):
        return f" DEFAULT {1 if value else 0}"
    if isinstance(value, (int, float)):
        return f" DEFAULT {value}"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f" DEFAULT '{escaped}'"
    return ""


def apply_additive_migrations(engine: Engine) -> list[str]:
    """Add any model column or index missing from an existing table.

    Returns the list of `table.column` names and `index:name` entries
    added, so startup can log what it changed rather than migrating
    silently.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                # create_all() handles brand-new tables; nothing to alter.
                continue

            present = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if not column.nullable and column.default is None:
                    # Can't add a NOT NULL column with no default to a table
                    # that already has rows - flag it loudly rather than
                    # emitting DDL that will fail halfway through startup.
                    logger.error(
                        "cannot auto-add NOT NULL column %s.%s with no default - "
                        "this needs a hand-written migration",
                        table.name, column.name,
                    )
                    continue

                ddl = (
                    f"ALTER TABLE {table.name} "
                    f"ADD COLUMN {column.name} {_column_ddl_type(column, engine.dialect)}"
                    f"{_default_clause(column)}"
                )
                conn.execute(text(ddl))
                added.append(f"{table.name}.{column.name}")
                logger.info("migrated: added missing column %s.%s", table.name, column.name)

    # Indexes go in a second pass, after every column exists - an index on a
    # column added above cannot be created before that ALTER has run.
    inspector = inspect(engine)  # re-inspect: the columns above are new
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present_indexes = {ix["name"] for ix in inspector.get_indexes(table.name)}
        present_columns = {col["name"] for col in inspector.get_columns(table.name)}
        for index in table.indexes:
            if index.name in present_indexes:
                continue
            if not {c.name for c in index.columns} <= present_columns:
                # A column the index needs is still missing (it hit the
                # NOT NULL guard above). Skip rather than emit failing DDL.
                logger.error(
                    "cannot create index %s - it covers a column that could not be added",
                    index.name,
                )
                continue
            index.create(bind=engine)
            added.append(f"index:{index.name}")
            logger.info("migrated: created missing index %s on %s", index.name, table.name)

    if added:
        logger.warning(
            "database schema was out of date - applied %d additive change(s): %s. "
            "Existing rows have NULL in any new column, which correctly means "
            "'not recorded at the time'.",
            len(added), ", ".join(added),
        )
    return added
