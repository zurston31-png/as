"""Schema migration tests.

All rows written here are synthetic test fixtures, not market data.
"""

from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from app import database
from app.database import (
    BASELINE_REVISION,
    EXPECTED_TABLES,
    STRATEGY_TABLES,
    SchemaError,
    create_db_engine,
    current_revision,
    head_revision,
    init_db,
    session_scope,
    table_row_counts,
)
from app.models import Base, IngestionRun, RunStatus
from tests.test_database import fixture_bar

STAMP = datetime(2024, 1, 2, 5, 0, tzinfo=UTC)


@pytest.fixture
def blank_engine(tmp_path):
    """An engine on a database file with no tables at all."""
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'migrations.db').as_posix()}")
    yield engine
    engine.dispose()


def create_milestone_1_database(engine) -> None:
    """Build the database the way Milestone 1 did: ``create_all``, no version marker."""
    Base.metadata.create_all(
        engine, tables=[Base.metadata.tables[name] for name in EXPECTED_TABLES]
    )
    with session_scope(database.create_session_factory(engine)) as session:
        run = IngestionRun(
            status=RunStatus.SUCCEEDED,
            symbol="TESTA",
            timeframe="1Day",
            source="alpaca",
            feed="iex",
            adjustment="split",
            currency="USD",
            requested_start=STAMP,
            requested_end=STAMP,
            bars_fetched=1,
            bars_inserted=1,
        )
        session.add(run)
        session.flush()
        session.add(fixture_bar(STAMP, ingestion_run_id=run.id))


def schema_of(engine, tables) -> dict:
    """Columns, keys and indexes of ``tables``, as the database reports them."""
    inspector = inspect(engine)
    return {
        table: {
            "columns": [
                (column["name"], str(column["type"]), column["nullable"])
                for column in inspector.get_columns(table)
            ],
            "primary_key": inspector.get_pk_constraint(table)["constrained_columns"],
            "foreign_keys": sorted(
                (tuple(key["constrained_columns"]), key["referred_table"])
                for key in inspector.get_foreign_keys(table)
            ),
            "indexes": sorted(
                (index["name"], tuple(index["column_names"]), bool(index["unique"]))
                for index in inspector.get_indexes(table)
            ),
        }
        for table in tables
    }


def test_a_new_database_is_migrated_to_the_newest_revision(blank_engine):
    assert current_revision(blank_engine) is None

    tables = init_db(blank_engine)

    assert current_revision(blank_engine) == head_revision()
    assert set(tables) == set(EXPECTED_TABLES) | set(STRATEGY_TABLES)


def test_the_migrated_schema_is_exactly_what_the_models_describe(blank_engine):
    init_db(blank_engine)

    with blank_engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)

    assert differences == []


def test_the_baseline_revision_reproduces_the_milestone_1_schema(blank_engine, tmp_path):
    # One database built the old way, one built by the baseline migration alone.
    create_milestone_1_database(blank_engine)
    migrated = create_db_engine(f"sqlite:///{(tmp_path / 'baseline.db').as_posix()}")
    try:
        with migrated.begin() as connection:
            command.upgrade(database._migration_config(connection), BASELINE_REVISION)

        assert schema_of(migrated, EXPECTED_TABLES) == schema_of(blank_engine, EXPECTED_TABLES)
    finally:
        migrated.dispose()


def test_a_milestone_1_database_is_upgraded_without_touching_its_data(blank_engine):
    create_milestone_1_database(blank_engine)
    market_schema_before = schema_of(blank_engine, EXPECTED_TABLES)
    with blank_engine.connect() as connection:
        bars_before = connection.execute(text("SELECT * FROM bars")).all()
        runs_before = connection.execute(text("SELECT * FROM ingestion_runs")).all()

    tables = init_db(blank_engine)

    assert current_revision(blank_engine) == head_revision()
    assert set(STRATEGY_TABLES) <= set(tables)
    assert schema_of(blank_engine, EXPECTED_TABLES) == market_schema_before
    with blank_engine.connect() as connection:
        assert connection.execute(text("SELECT * FROM bars")).all() == bars_before
        assert connection.execute(text("SELECT * FROM ingestion_runs")).all() == runs_before
    assert table_row_counts(blank_engine, STRATEGY_TABLES) == dict.fromkeys(STRATEGY_TABLES, 0)


def test_migrating_twice_changes_nothing(blank_engine):
    create_milestone_1_database(blank_engine)
    init_db(blank_engine)
    everything = (*EXPECTED_TABLES, *STRATEGY_TABLES)
    schema, counts = schema_of(blank_engine, everything), table_row_counts(blank_engine, everything)

    init_db(blank_engine)

    assert schema_of(blank_engine, everything) == schema
    assert table_row_counts(blank_engine, everything) == counts
    assert current_revision(blank_engine) == head_revision()


def test_an_unrecognised_database_is_refused_and_left_alone(blank_engine):
    with blank_engine.begin() as connection:
        connection.execute(text("CREATE TABLE bars (something TEXT)"))
        connection.execute(text("INSERT INTO bars VALUES ('not a SignalIQ database')"))

    with pytest.raises(SchemaError, match="ingestion_runs"):
        init_db(blank_engine)

    assert inspect(blank_engine).get_table_names() == ["bars"]
    with blank_engine.connect() as connection:
        assert connection.execute(text("SELECT something FROM bars")).scalar_one() == (
            "not a SignalIQ database"
        )


def test_removing_the_strategy_tables_keeps_the_market_data(blank_engine):
    create_milestone_1_database(blank_engine)
    init_db(blank_engine)

    with blank_engine.begin() as connection:
        command.downgrade(database._migration_config(connection), BASELINE_REVISION)

    assert current_revision(blank_engine) == BASELINE_REVISION
    assert set(inspect(blank_engine).get_table_names()) == {*EXPECTED_TABLES, "alembic_version"}
    assert table_row_counts(blank_engine) == {
        "bars": 1,
        "ingestion_runs": 1,
        "data_quality_checks": 0,
    }
    # ... and the upgrade can be applied again.
    init_db(blank_engine)
    assert current_revision(blank_engine) == head_revision()


def test_no_migration_can_drop_the_market_data_tables(blank_engine):
    init_db(blank_engine)

    with (
        pytest.raises(NotImplementedError, match="never dropped"),
        blank_engine.begin() as connection,
    ):
        command.downgrade(database._migration_config(connection), "base")

    assert set(EXPECTED_TABLES) <= set(inspect(blank_engine).get_table_names())
