"""Database tests: schema, insertion, primary key and UTC timestamp handling.

All values are synthetic test fixtures (see ``conftest.py``), not market data.
"""

import json
from datetime import UTC, date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError, StatementError

from app import database
from app.database import create_db_engine, init_db, session_scope, table_row_counts
from app.models import (
    Bar,
    CheckStage,
    CheckStatus,
    DataQualityCheck,
    IngestionRun,
    RunStatus,
    ensure_utc,
    format_utc,
)

REQUIRED_BAR_COLUMNS = {
    "symbol",
    "timeframe",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
    "source",
    "feed",
    "adjustment",
}


def fixture_bar(timestamp: datetime, **overrides) -> Bar:
    """Build a Bar with invented fixture values."""
    values = {
        "symbol": "TESTA",
        "timeframe": "1Day",
        "timestamp": timestamp,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000,
        "vwap": 10.2,
        "trade_count": 25,
        "source": "alpaca",
        "feed": "iex",
        "adjustment": "split",
    }
    values.update(overrides)
    return Bar(**values)


# --- Schema -----------------------------------------------------------------


def test_init_db_creates_the_three_tables(engine):
    assert set(inspect(engine).get_table_names()) == {
        "bars",
        "ingestion_runs",
        "data_quality_checks",
    }


def test_bars_table_has_all_required_columns(engine):
    columns = {column["name"] for column in inspect(engine).get_columns("bars")}
    assert REQUIRED_BAR_COLUMNS <= columns


def test_bars_primary_key_is_symbol_timeframe_timestamp(engine):
    primary_key = inspect(engine).get_pk_constraint("bars")["constrained_columns"]
    assert primary_key == ["symbol", "timeframe", "timestamp"]


def test_init_db_is_idempotent_and_keeps_existing_data(engine, session_factory):
    with session_scope(session_factory) as session:
        session.add(fixture_bar(datetime(2024, 1, 2, 5, 0, tzinfo=UTC)))

    tables = init_db(engine)  # second initialisation

    assert set(tables) == {"bars", "ingestion_runs", "data_quality_checks"}
    assert table_row_counts(engine)["bars"] == 1


def test_database_folder_is_created_when_missing(tmp_path):
    database_url = f"sqlite:///{(tmp_path / 'nested' / 'folder' / 'x.db').as_posix()}"
    engine = create_db_engine(database_url)
    try:
        init_db(engine)
        assert (tmp_path / "nested" / "folder" / "x.db").is_file()
    finally:
        engine.dispose()


def test_non_sqlite_database_is_refused():
    with pytest.raises(ValueError, match="Only SQLite"):
        create_db_engine("postgresql://user@localhost/signaliq")


# --- Insertion and primary key ----------------------------------------------


def test_insert_and_read_back_a_bar(session_factory):
    timestamp = datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    with session_scope(session_factory) as session:
        session.add(fixture_bar(timestamp))

    with session_scope(session_factory) as session:
        stored = session.execute(select(Bar)).scalar_one()
        assert (stored.symbol, stored.timeframe, stored.timestamp) == ("TESTA", "1Day", timestamp)
        assert (stored.open, stored.high, stored.low, stored.close) == (10.0, 11.0, 9.0, 10.5)
        assert (stored.volume, stored.vwap, stored.trade_count) == (1000, 10.2, 25)
        assert (stored.source, stored.feed, stored.adjustment) == ("alpaca", "iex", "split")
        assert stored.ingested_at.tzinfo is not None


def test_duplicate_primary_key_is_rejected_by_the_database(session_factory):
    timestamp = datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    with session_scope(session_factory) as session:
        session.add(fixture_bar(timestamp))

    with pytest.raises(IntegrityError):
        with session_scope(session_factory) as session:
            session.add(fixture_bar(timestamp, close=10.6))

    with session_scope(session_factory) as session:
        stored = session.execute(select(Bar)).scalars().all()
        assert len(stored) == 1
        assert stored[0].close == 10.5  # the original record is untouched


def test_same_timestamp_is_allowed_for_another_symbol_or_timeframe(session_factory):
    timestamp = datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    with session_scope(session_factory) as session:
        session.add(fixture_bar(timestamp))
        session.add(fixture_bar(timestamp, symbol="TESTB"))
        session.add(fixture_bar(timestamp, timeframe="1Hour"))

    with session_scope(session_factory) as session:
        assert len(session.execute(select(Bar)).scalars().all()) == 3


@pytest.mark.parametrize("column", sorted(REQUIRED_BAR_COLUMNS - {"symbol", "timeframe", "timestamp"}))
def test_required_bar_columns_are_not_nullable(session_factory, column):
    with pytest.raises(IntegrityError):
        with session_scope(session_factory) as session:
            session.add(fixture_bar(datetime(2024, 1, 2, 5, 0, tzinfo=UTC), **{column: None}))


# --- UTC timestamp handling ---------------------------------------------------


def test_naive_timestamp_is_rejected(session_factory):
    with pytest.raises(StatementError, match="Naive datetime rejected"):
        with session_scope(session_factory) as session:
            session.add(fixture_bar(datetime(2024, 1, 2, 5, 0)))  # no tzinfo


def test_aware_non_utc_timestamp_is_converted_to_utc(session_factory):
    new_york_midnight = datetime(2024, 1, 2, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    with session_scope(session_factory) as session:
        session.add(fixture_bar(new_york_midnight))

    with session_scope(session_factory) as session:
        stored = session.execute(select(Bar)).scalar_one()
        assert stored.timestamp == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
        assert stored.timestamp.utcoffset() == timedelta(0)


def test_timestamps_are_stored_as_utc_text_ending_in_z(session_factory, engine):
    with session_scope(session_factory) as session:
        session.add(fixture_bar(datetime(2024, 7, 1, 4, 0, tzinfo=UTC)))

    with engine.connect() as connection:
        raw = connection.execute(text("SELECT timestamp, ingested_at FROM bars")).one()

    assert raw.timestamp == "2024-07-01T04:00:00.000000Z"
    assert raw.ingested_at.endswith("Z")


def test_timestamps_read_back_as_timezone_aware_utc(session_factory):
    with session_scope(session_factory) as session:
        session.add(fixture_bar(datetime(2024, 1, 2, 5, 0, tzinfo=timezone(timedelta(hours=-5)))))

    with session_scope(session_factory) as session:
        stored = session.execute(select(Bar)).scalar_one()
        assert stored.timestamp.tzinfo is not None
        assert stored.timestamp == datetime(2024, 1, 2, 10, 0, tzinfo=UTC)


def test_timestamp_range_filter_and_chronological_order(session_factory):
    timestamps = [
        datetime(2024, 1, 4, 5, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 5, 0, tzinfo=UTC),
        datetime(2024, 7, 1, 4, 0, tzinfo=UTC),
        datetime(2024, 1, 3, 5, 0, tzinfo=UTC),
    ]
    with session_scope(session_factory) as session:
        session.add_all(fixture_bar(timestamp) for timestamp in timestamps)

    with session_scope(session_factory) as session:
        rows = (
            session.execute(
                select(Bar.timestamp)
                .where(Bar.timestamp >= datetime(2024, 1, 3, 0, 0, tzinfo=UTC))
                .where(Bar.timestamp <= datetime(2024, 1, 4, 23, 59, tzinfo=UTC))
                .order_by(Bar.timestamp)
            )
            .scalars()
            .all()
        )
    assert rows == [datetime(2024, 1, 3, 5, 0, tzinfo=UTC), datetime(2024, 1, 4, 5, 0, tzinfo=UTC)]


def test_ensure_utc_and_format_utc_helpers():
    aware = datetime(2024, 1, 2, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    assert ensure_utc(aware) == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    assert format_utc(aware) == "2024-01-02T05:00:00.000000Z"
    with pytest.raises(ValueError, match="Naive datetime rejected"):
        ensure_utc(datetime(2024, 1, 2))


# --- Ingestion history and quality checks -------------------------------------


def test_ingestion_run_and_quality_check_are_persisted(session_factory):
    with session_scope(session_factory) as session:
        run = IngestionRun(
            symbol="TESTA",
            timeframe="1Day",
            source="alpaca",
            feed="iex",
            adjustment="split",
            currency="USD",
            requested_start=datetime(2024, 1, 2, 5, 0, tzinfo=UTC),
            requested_end=datetime(2024, 1, 6, 4, 59, 59, tzinfo=UTC),
        )
        session.add(run)
        session.flush()
        session.add(
            DataQualityCheck(
                check_run_id="fixture-check-run",
                stage=CheckStage.INGESTION,
                ingestion_run_id=run.id,
                check_name="high_below_low",
                status=CheckStatus.FAILED,
                symbol="TESTA",
                timeframe="1Day",
                bar_timestamp=datetime(2024, 1, 2, 5, 0, tzinfo=UTC),
                message="fixture finding",
                details={"raw": {"h": 8.0, "l": 9.0}},
            )
        )

    with session_scope(session_factory) as session:
        run = session.execute(select(IngestionRun)).scalar_one()
        assert run.status == RunStatus.RUNNING
        assert run.started_at.tzinfo is not None
        assert run.finished_at is None
        assert (run.bars_fetched, run.bars_inserted, run.bars_rejected) == (0, 0, 0)

        check = session.execute(select(DataQualityCheck)).scalar_one()
        assert check.ingestion_run_id == run.id
        assert check.status == CheckStatus.FAILED
        assert check.details == {"raw": {"h": 8.0, "l": 9.0}}
        assert check.bar_timestamp == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)


def test_foreign_keys_are_enforced(session_factory):
    with pytest.raises(IntegrityError):
        with session_scope(session_factory) as session:
            session.add(
                DataQualityCheck(
                    check_run_id="fixture-check-run",
                    stage=CheckStage.STORED,
                    ingestion_run_id=999_999,  # no such run
                    check_name="negative_volume",
                    status=CheckStatus.PASSED,
                    message="fixture",
                )
            )


def test_session_scope_rolls_back_on_error(session_factory, engine):
    with pytest.raises(RuntimeError, match="boom"):
        with session_scope(session_factory) as session:
            session.add(fixture_bar(datetime(2024, 1, 2, 5, 0, tzinfo=UTC)))
            session.flush()
            raise RuntimeError("boom")

    assert table_row_counts(engine)["bars"] == 0


# --- Command line -------------------------------------------------------------


def test_database_cli_initialises_the_database(settings, monkeypatch, capsys):
    monkeypatch.setattr(database, "get_settings", lambda: settings)

    exit_code = database.main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Database ready" in output
    for table in ("bars", "ingestion_runs", "data_quality_checks"):
        assert table in output


# --- Command line: scripts/inspect_db.py ---------------------------------------


def test_inspect_cli_reports_database_statistics(run_script, store_bars, engine):
    days = [date(2024, 1, day) for day in (2, 3, 4, 5)]
    store_bars("TESTA", days)
    store_bars("ZZZZ", days[:1])  # a symbol outside the configured universe

    exit_code, output, _ = run_script("inspect_db")

    assert exit_code == 0
    assert "test_signaliq.db" in output
    lines = [line.split() for line in output.splitlines()]
    assert ["bars", "5", "rows"] in lines
    assert ["ingestion_runs", "0", "rows"] in lines
    assert ["data_quality_checks", "0", "rows"] in lines
    assert ["TESTA", "1Day", "4", "2024-01-02T05:00:00Z", "2024-01-05T05:00:00Z"] in lines
    assert any("ZZZZ" in line and "not in the configured universe" in line for line in output.splitlines())
    assert "Configured symbols without data: TESTB" in output
    assert "source=alpaca  feed=iex  adjustment=split" in output
    assert "Stored bars have not been validated yet" in output


def test_inspect_cli_can_print_json(run_script, store_bars, engine):
    store_bars("TESTA", [date(2024, 1, 2), date(2024, 1, 3)])

    exit_code, output, _ = run_script("inspect_db", "--json")

    assert exit_code == 0
    stats = json.loads(output)
    assert stats["tables"] == {"bars": 2, "ingestion_runs": 0, "data_quality_checks": 0}
    assert stats["bars"] == [
        {
            "symbol": "TESTA",
            "timeframe": "1Day",
            "bar_count": 2,
            "first_timestamp": "2024-01-02T05:00:00Z",
            "last_timestamp": "2024-01-03T05:00:00Z",
            "in_universe": True,
        }
    ]
    assert stats["provenance"] == [
        {"source": "alpaca", "feed": "iex", "adjustment": "split", "bar_count": 2}
    ]
    assert stats["universe"] == {"configured": ["TESTA", "TESTB"], "without_data": ["TESTB"]}
    assert stats["ingestion_runs"] == {"by_status": {}, "latest": []}
    assert stats["quality_checks"] == {"by_status": {}, "latest_stored_validation": None}
    assert stats["database"]["size_bytes"] > 0


def test_inspect_cli_only_reads(run_script, store_bars, engine):
    store_bars("TESTA", [date(2024, 1, 2)])
    before = table_row_counts(engine)

    run_script("inspect_db")

    assert table_row_counts(engine) == before


def test_inspect_cli_explains_a_missing_database(run_script):
    # No ``engine`` fixture: the database file was never created.
    exit_code, _, errors = run_script("inspect_db")

    assert exit_code == 2
    assert "database not found" in errors
    assert "python -m app.database" in errors
