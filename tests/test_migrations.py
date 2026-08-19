"""Tests for app/migrations.py.

The bug these cover is an upgrade breakage, not a fresh-install one:
SQLAlchemy's create_all() creates missing TABLES but never adds a missing
COLUMN to a table that already exists, so every deployment running an
earlier build would come back up with tables missing everything added
since, and die on the first query.
"""
import sqlite3

import pytest
from sqlalchemy import create_engine, inspect

from app.database import Base
from app.migrations import apply_additive_migrations

# Column set as it existed before Stages 8/9 added the score + source fields.
OLD_SIGNALS_DDL = """
CREATE TABLE signals (
    id INTEGER PRIMARY KEY,
    received_at DATETIME,
    symbol VARCHAR(64),
    token_address VARCHAR(128),
    chain VARCHAR(32),
    signal_type VARCHAR(16),
    price FLOAT,
    tv_timestamp DATETIME,
    rsi FLOAT,
    ema9 FLOAT,
    ema21 FLOAT,
    volume FLOAT,
    volume_sma FLOAT,
    breakout_level FLOAT,
    raw_payload JSON
)
"""


@pytest.fixture()
def old_db(tmp_path):
    """A database as an earlier version of the bot would have left it."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute(OLD_SIGNALS_DDL)
    con.execute(
        "INSERT INTO signals (symbol, chain, signal_type, price) VALUES ('LEGACY','solana','buy',0.01)"
    )
    con.commit()
    con.close()
    return path


def _columns(path, table="signals") -> set[str]:
    con = sqlite3.connect(path)
    try:
        return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def test_missing_columns_are_added_to_an_existing_table(old_db):
    engine = create_engine(f"sqlite:///{old_db}")
    before = _columns(old_db)
    assert "source" not in before, "fixture should start without the newer columns"

    added = apply_additive_migrations(engine)

    after = _columns(old_db)
    assert "source" in after
    assert "signal_score" in after
    assert "signal_score_reliable" in after
    assert "signal_score_factors" in after
    assert any(name.endswith(".source") for name in added)


def test_existing_rows_survive_the_migration(old_db):
    engine = create_engine(f"sqlite:///{old_db}")
    apply_additive_migrations(engine)

    con = sqlite3.connect(old_db)
    try:
        row = con.execute("SELECT symbol, source, signal_score FROM signals").fetchone()
    finally:
        con.close()

    assert row[0] == "LEGACY", "the pre-existing row must not be lost"

    # source has a scalar model default, so old rows take it - and here that
    # is a fact rather than a guess: a signal written before the scanner
    # existed genuinely did come from TradingView.
    assert row[1] == "tradingview"

    # signal_score has no default, so it stays NULL - the honest value for
    # "this wasn't being recorded when the row was written".
    assert row[2] is None


def test_migration_is_idempotent(old_db):
    engine = create_engine(f"sqlite:///{old_db}")
    first = apply_additive_migrations(engine)
    second = apply_additive_migrations(engine)
    assert first, "first run should have had work to do"
    assert second == [], "second run must be a no-op"


def test_a_fresh_database_needs_no_migration(tmp_path):
    path = tmp_path / "fresh.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(bind=engine)
    assert apply_additive_migrations(engine) == []


def test_brand_new_tables_are_left_to_create_all(old_db):
    """The scanner's table doesn't exist in the old schema at all. The
    migration must not try to ALTER it - create_all() owns that case."""
    engine = create_engine(f"sqlite:///{old_db}")
    assert "scanned_tokens" not in set(inspect(engine).get_table_names())

    added = apply_additive_migrations(engine)
    assert not any(name.startswith("scanned_tokens.") for name in added)

    Base.metadata.create_all(bind=engine)
    assert "scanned_tokens" in set(inspect(engine).get_table_names())


def test_every_model_column_exists_after_create_all_plus_migrate(old_db):
    """End to end: the exact sequence init_db() runs must leave every table
    matching its model, whatever state the database started in."""
    engine = create_engine(f"sqlite:///{old_db}")
    Base.metadata.create_all(bind=engine)
    apply_additive_migrations(engine)

    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        actual = {col["name"] for col in inspector.get_columns(table.name)}
        expected = {col.name for col in table.columns}
        missing = expected - actual
        assert not missing, f"{table.name} still missing {missing} after migration"
