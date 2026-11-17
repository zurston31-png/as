"""Ingestion tests: storing bars, idempotency, conflicts, pagination, provider
errors, missing data and the "completed candles only" rule.

The provider is mocked and every bar is a SYNTHETIC TEST FIXTURE with invented
values (see ``conftest.py``). Nothing in this file is real market data.
"""

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import func, select, text

from app.config import ConfigurationError
from app.data import ingestion
from app.data.alpaca_client import AlpacaClient
from app.data.ingestion import IngestionError, IngestionService, run_ingestion
from app.data.validation import (
    CHECK_CONFLICTING_RECORD,
    CHECK_DUPLICATE_TIMESTAMP,
    CHECK_HIGH_BELOW_LOW,
    CHECK_MISSING_FIELDS,
    CHECK_NEGATIVE_VOLUME,
    CHECK_SESSION_GAP,
    CHECK_UNEXPECTED_SYMBOL,
    ROW_CHECKS,
)
from app.database import session_scope
from app.models import Bar, CheckStage, CheckStatus, DataQualityCheck, IngestionRun, RunStatus

#: The nine trading sessions from 2 to 12 January 2024.
SESSIONS = [date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9, 10, 11, 12)]
START, END = date(2024, 1, 2), date(2024, 1, 12)


def stored_bars(session_factory) -> list[Bar]:
    with session_scope(session_factory) as session:
        return list(session.execute(select(Bar).order_by(Bar.symbol, Bar.timestamp)).scalars())


def stored_runs(session_factory) -> list[IngestionRun]:
    with session_scope(session_factory) as session:
        return list(session.execute(select(IngestionRun).order_by(IngestionRun.id)).scalars())


def stored_checks(session_factory, *, status: str | None = None) -> list[DataQualityCheck]:
    with session_scope(session_factory) as session:
        query = select(DataQualityCheck).order_by(DataQualityCheck.id)
        if status is not None:
            query = query.where(DataQualityCheck.status == status)
        return list(session.execute(query).scalars())


def bar_count(session_factory) -> int:
    with session_scope(session_factory) as session:
        return session.execute(select(func.count()).select_from(Bar)).scalar_one()


def ingest(service: IngestionService, symbol: str = "TESTA", **kwargs):
    kwargs.setdefault("start", START)
    kwargs.setdefault("end", END)
    return service.ingest_symbol(symbol, **kwargs)


# =============================================================================
# Database insertion
# =============================================================================


def test_ingestion_stores_bars_in_sqlite(ingestion_service, alpaca_mock, session_factory, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    result = ingest(ingestion_service)

    assert result.status == RunStatus.SUCCEEDED
    assert (result.bars_fetched, result.bars_inserted) == (9, 9)
    assert (result.bars_unchanged, result.bars_conflicting) == (0, 0)
    assert (result.bars_rejected, result.bars_duplicate, result.missing_sessions) == (0, 0, 0)
    assert result.error is None
    assert result.ok

    bars = stored_bars(session_factory)
    assert len(bars) == 9
    first = bars[0]
    assert (first.symbol, first.timeframe) == ("TESTA", "1Day")
    assert first.timestamp == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    assert (first.open, first.high, first.low, first.close) == (10.0, 11.0, 9.0, 10.5)
    assert (first.volume, first.vwap, first.trade_count) == (1000, 10.2, 25)
    assert (first.source, first.feed, first.adjustment) == ("alpaca", "iex", "split")
    assert first.ingestion_run_id == result.run_id
    assert first.ingested_at.tzinfo is not None


def test_stored_values_are_exactly_what_the_provider_sent(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    payload = [
        make_bar(day, o=10 + index, h=12 + index, l=9 + index, c=11 + index, v=1000 * (index + 1),
                 n=10 + index, vw=10.5 + index)
        for index, day in enumerate(SESSIONS)
    ]  # fmt: skip
    alpaca_mock.add_bars({"TESTA": payload})

    ingest(ingestion_service)

    for raw, bar in zip(payload, stored_bars(session_factory), strict=True):
        assert (bar.open, bar.high, bar.low, bar.close) == (raw["o"], raw["h"], raw["l"], raw["c"])
        assert (bar.volume, bar.trade_count, bar.vwap) == (raw["v"], raw["n"], raw["vw"])


def test_request_asks_for_the_configured_feed_and_window(ingestion_service, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    ingest(ingestion_service)

    query = alpaca_mock.query()
    assert query["symbols"] == "TESTA"
    assert query["timeframe"] == "1Day"
    assert query["feed"] == "iex"
    assert query["adjustment"] == "split"
    assert query["currency"] == "USD"
    # Whole New York days: 2 January 00:00 to 12 January 23:59:59, in UTC.
    assert query["start"] == "2024-01-02T05:00:00Z"
    assert query["end"] == "2024-01-13T04:59:59Z"


def test_timestamps_are_stored_in_utc_across_a_daylight_saving_change(
    settings, session_factory, alpaca_client, alpaca_mock, engine, make_bar
):
    service = IngestionService(
        settings,
        session_factory,
        alpaca_client,
        now=lambda: datetime(2024, 3, 20, 15, 0, tzinfo=UTC),
    )
    # Friday 8 March 2024 is in winter time, Monday 11 March 2024 in summer time.
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 3, 8)), make_bar(date(2024, 3, 11))]})

    service.ingest_symbol("TESTA", start=date(2024, 3, 8), end=date(2024, 3, 11))

    bars = stored_bars(session_factory)
    assert [bar.timestamp for bar in bars] == [
        datetime(2024, 3, 8, 5, 0, tzinfo=UTC),
        datetime(2024, 3, 11, 4, 0, tzinfo=UTC),
    ]
    assert all(bar.timestamp.utcoffset().total_seconds() == 0 for bar in bars)
    with engine.connect() as connection:
        raw = (
            connection.execute(text("SELECT timestamp FROM bars ORDER BY timestamp"))
            .scalars()
            .all()
        )
    assert raw == ["2024-03-08T05:00:00.000000Z", "2024-03-11T04:00:00.000000Z"]


# =============================================================================
# Idempotent ingestion
# =============================================================================


def test_running_the_same_import_twice_creates_no_duplicates(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    payload = {"TESTA": [make_bar(day) for day in SESSIONS]}
    alpaca_mock.add_bars(payload)
    alpaca_mock.add_bars(payload)

    first = ingest(ingestion_service)
    snapshot = [
        (bar.timestamp, bar.close, bar.ingestion_run_id) for bar in stored_bars(session_factory)
    ]
    second = ingest(ingestion_service)

    assert (first.bars_inserted, first.bars_unchanged) == (9, 0)
    assert (second.bars_fetched, second.bars_inserted, second.bars_unchanged) == (9, 0, 9)
    assert second.status == RunStatus.SUCCEEDED
    assert bar_count(session_factory) == 9
    # The stored rows are untouched: same values, still attributed to the first run.
    assert [
        (bar.timestamp, bar.close, bar.ingestion_run_id) for bar in stored_bars(session_factory)
    ] == snapshot
    assert first.run_id != second.run_id


def test_many_repeated_imports_never_grow_the_table(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    payload = {"TESTA": [make_bar(day) for day in SESSIONS]}
    for _ in range(5):
        alpaca_mock.add_bars(payload)
        ingest(ingestion_service)

    assert bar_count(session_factory) == 9
    runs = stored_runs(session_factory)
    assert [run.bars_inserted for run in runs] == [9, 0, 0, 0, 0]
    assert [run.bars_unchanged for run in runs] == [0, 9, 9, 9, 9]
    assert stored_checks(session_factory, status=CheckStatus.FAILED) == []


def test_overlapping_import_inserts_only_the_new_bars(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS[:4]]})
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    first = ingest(ingestion_service, end=date(2024, 1, 5))
    second = ingest(ingestion_service)

    assert (first.bars_inserted, first.bars_unchanged) == (4, 0)
    assert (second.bars_inserted, second.bars_unchanged) == (5, 4)
    assert bar_count(session_factory) == 9


def test_each_symbol_has_its_own_bars(ingestion_service, alpaca_mock, session_factory, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})
    alpaca_mock.add_bars({"TESTB": [make_bar(day, c=10.7) for day in SESSIONS]})

    batch = ingestion_service.ingest_symbols(["TESTA", "TESTB"], start=START, end=END)

    assert batch.ok
    assert [result.symbol for result in batch.results] == ["TESTA", "TESTB"]
    assert [result.bars_inserted for result in batch.results] == [9, 9]
    assert bar_count(session_factory) == 18
    assert [request.url.params["symbols"] for request in alpaca_mock.requests] == ["TESTA", "TESTB"]


# =============================================================================
# Conflicting records are recorded, never silently replaced
# =============================================================================


def test_conflicting_bar_is_recorded_and_the_stored_bar_is_kept(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    original = [make_bar(day) for day in SESSIONS]
    revised = [make_bar(day) for day in SESSIONS]
    revised[2] = make_bar(SESSIONS[2], c=10.9, v=2000)  # 4 January differs
    alpaca_mock.add_bars({"TESTA": original})
    alpaca_mock.add_bars({"TESTA": revised})

    first = ingest(ingestion_service)
    second = ingest(ingestion_service)

    assert second.status == RunStatus.COMPLETED_WITH_ISSUES
    assert (second.bars_inserted, second.bars_unchanged, second.bars_conflicting) == (0, 8, 1)
    assert second.failures == 1

    # The stored record still holds the original values.
    kept = stored_bars(session_factory)[2]
    assert kept.timestamp == datetime(2024, 1, 4, 5, 0, tzinfo=UTC)
    assert (kept.close, kept.volume) == (10.5, 1000)
    assert kept.ingestion_run_id == first.run_id
    assert bar_count(session_factory) == 9

    # The discrepancy is on record for investigation.
    conflicts = stored_checks(session_factory, status=CheckStatus.FAILED)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.check_name == CHECK_CONFLICTING_RECORD
    assert conflict.stage == CheckStage.INGESTION
    assert conflict.ingestion_run_id == second.run_id
    assert (conflict.symbol, conflict.timeframe) == ("TESTA", "1Day")
    assert conflict.bar_timestamp == datetime(2024, 1, 4, 5, 0, tzinfo=UTC)
    assert conflict.details["differences"] == {
        "close": {"stored": 10.5, "incoming": 10.9},
        "volume": {"stored": 1000, "incoming": 2000},
    }
    assert conflict.details["stored_by_ingestion_run_id"] == first.run_id
    assert conflict.details["incoming_record"] == revised[2]
    assert "the stored record was kept unchanged" in conflict.message


def test_conflict_is_reported_again_on_every_run_until_resolved(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    revised = [make_bar(day, c=10.9) if day == SESSIONS[0] else make_bar(day) for day in SESSIONS]
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})
    alpaca_mock.add_bars({"TESTA": revised})
    alpaca_mock.add_bars({"TESTA": revised})

    ingest(ingestion_service)
    second = ingest(ingestion_service)
    third = ingest(ingestion_service)

    assert (second.bars_conflicting, third.bars_conflicting) == (1, 1)
    assert stored_bars(session_factory)[0].close == 10.5
    assert len(stored_checks(session_factory, status=CheckStatus.FAILED)) == 2


def test_data_from_another_feed_is_never_merged_into_stored_bars(
    ingestion_service, settings, session_factory, alpaca_mock, fixed_now, make_bar
):
    payload = {"TESTA": [make_bar(day) for day in SESSIONS]}
    alpaca_mock.add_bars(payload)
    alpaca_mock.add_bars(payload)
    ingest(ingestion_service)

    other_feed = settings.model_copy(update={"alpaca_feed": "sip"})
    with AlpacaClient(other_feed, transport=alpaca_mock.transport) as sip_client:
        sip_service = IngestionService(
            other_feed, session_factory, sip_client, now=lambda: fixed_now
        )
        result = ingest(sip_service)

    assert alpaca_mock.query()["feed"] == "sip"  # the explicitly configured feed was requested
    assert (result.bars_inserted, result.bars_conflicting) == (0, 9)
    assert {bar.feed for bar in stored_bars(session_factory)} == {"iex"}
    conflicts = stored_checks(session_factory, status=CheckStatus.FAILED)
    assert {check.check_name for check in conflicts} == {CHECK_CONFLICTING_RECORD}
    assert conflicts[0].details["differences"] == {"feed": {"stored": "iex", "incoming": "sip"}}


# =============================================================================
# Invalid and duplicate bars
# =============================================================================


def test_invalid_bars_are_rejected_recorded_and_never_stored(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    payload = [make_bar(day) for day in SESSIONS]
    payload[1] = make_bar(SESSIONS[1], h=8.0)  # 3 January: high below low
    payload[4] = make_bar(SESSIONS[4], v=-5)  # 8 January: negative volume
    payload[6] = make_bar(SESSIONS[6], omit=("vw",))  # 10 January: missing field
    alpaca_mock.add_bars({"TESTA": payload})

    result = ingest(ingestion_service)

    assert result.status == RunStatus.COMPLETED_WITH_ISSUES
    assert (result.bars_fetched, result.bars_inserted, result.bars_rejected) == (9, 6, 3)

    stored_days = [bar.timestamp.day for bar in stored_bars(session_factory)]
    assert stored_days == [2, 4, 5, 9, 11, 12]  # no invalid bar was stored or "repaired"

    failed = stored_checks(session_factory, status=CheckStatus.FAILED)
    row_failures = {check.check_name: check for check in failed if check.check_name in ROW_CHECKS}
    assert set(row_failures) == {CHECK_HIGH_BELOW_LOW, CHECK_NEGATIVE_VOLUME, CHECK_MISSING_FIELDS}
    assert row_failures[CHECK_HIGH_BELOW_LOW].details["record"] == payload[1]
    assert row_failures[CHECK_NEGATIVE_VOLUME].details["record"] == payload[4]
    assert row_failures[CHECK_MISSING_FIELDS].details["record"] == payload[6]
    assert {check.ingestion_run_id for check in failed} == {result.run_id}

    # The rejected sessions are also visible as missing data; nothing was filled in.
    gaps = [check for check in failed if check.check_name == CHECK_SESSION_GAP]
    assert [gap.details["missing_sessions"] for gap in gaps] == [
        ["2024-01-03"],
        ["2024-01-08"],
        ["2024-01-10"],
    ]
    assert result.missing_sessions == 3


def test_identical_duplicate_in_the_payload_is_stored_once(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    payload = [make_bar(day) for day in SESSIONS]
    payload.append(make_bar(SESSIONS[3]))  # 5 January twice, identical
    alpaca_mock.add_bars({"TESTA": payload})

    result = ingest(ingestion_service)

    assert (result.bars_fetched, result.bars_inserted, result.bars_duplicate) == (10, 9, 1)
    assert result.bars_rejected == 0
    assert result.status == RunStatus.COMPLETED_WITH_ISSUES  # the duplicate is on record
    assert bar_count(session_factory) == 9
    failed = stored_checks(session_factory, status=CheckStatus.FAILED)
    assert [check.check_name for check in failed] == [CHECK_DUPLICATE_TIMESTAMP]
    assert failed[0].details["occurrences"] == 2


def test_conflicting_duplicates_in_the_payload_are_all_rejected(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    payload = [make_bar(day) for day in SESSIONS]
    payload.append(make_bar(SESSIONS[3], c=10.9))  # 5 January twice, different close
    alpaca_mock.add_bars({"TESTA": payload})

    result = ingest(ingestion_service)

    assert (result.bars_fetched, result.bars_inserted, result.bars_rejected) == (10, 8, 2)
    assert result.bars_duplicate == 0
    assert date(2024, 1, 5) not in {bar.timestamp.date() for bar in stored_bars(session_factory)}
    assert result.missing_sessions == 1


def test_bars_for_a_symbol_that_was_not_requested_are_rejected(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars(
        {
            "TESTA": [make_bar(day) for day in SESSIONS],
            "TESTB": [make_bar(SESSIONS[0])],  # not asked for in this request
        }
    )

    result = ingest(ingestion_service)

    assert (result.bars_fetched, result.bars_inserted, result.bars_rejected) == (10, 9, 1)
    assert {bar.symbol for bar in stored_bars(session_factory)} == {"TESTA"}
    failed = stored_checks(session_factory, status=CheckStatus.FAILED)
    assert [(check.check_name, check.symbol) for check in failed] == [
        (CHECK_UNEXPECTED_SYMBOL, "TESTB")
    ]


def test_counters_always_add_up_to_the_number_of_bars_fetched(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS[:5]]})
    messy = [make_bar(day) for day in SESSIONS]
    messy[0] = make_bar(SESSIONS[0], c=10.9)  # conflicts with the stored bar
    messy[6] = make_bar(SESSIONS[6], l=-1.0)  # invalid
    messy.append(make_bar(SESSIONS[8]))  # identical duplicate
    alpaca_mock.add_bars({"TESTA": messy})

    ingest(ingestion_service, end=date(2024, 1, 8))
    result = ingest(ingestion_service)

    assert result.bars_fetched == 10
    assert (
        result.bars_inserted,
        result.bars_unchanged,
        result.bars_conflicting,
        result.bars_rejected,
        result.bars_duplicate,
    ) == (3, 4, 1, 1, 1)
    run = stored_runs(session_factory)[-1]
    assert run.bars_fetched == (
        run.bars_inserted
        + run.bars_unchanged
        + run.bars_conflicting
        + run.bars_rejected
        + run.bars_duplicate
    )


# =============================================================================
# Pagination
# =============================================================================


def test_all_pages_are_downloaded_and_stored(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars(
        {"TESTA": [make_bar(day) for day in SESSIONS[:3]]}, next_page_token="tok-1"
    )
    alpaca_mock.add_bars(
        {"TESTA": [make_bar(day) for day in SESSIONS[3:6]]}, next_page_token="tok-2"
    )
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS[6:]]})

    result = ingest(ingestion_service)

    assert result.status == RunStatus.SUCCEEDED
    assert (result.pages_fetched, result.bars_fetched, result.bars_inserted) == (3, 9, 9)
    assert [alpaca_mock.query(index).get("page_token") for index in range(3)] == [
        None,
        "tok-1",
        "tok-2",
    ]
    assert [bar.timestamp.day for bar in stored_bars(session_factory)] == [
        2,
        3,
        4,
        5,
        8,
        9,
        10,
        11,
        12,
    ]
    assert stored_runs(session_factory)[0].pages_fetched == 3


# =============================================================================
# Provider errors
# =============================================================================


def test_authentication_failure_is_recorded_and_stops_the_batch(
    ingestion_service, alpaca_mock, session_factory
):
    alpaca_mock.add_error(401, "fixture unauthorized")

    batch = ingestion_service.ingest_symbols(["TESTA", "TESTB"], start=START, end=END)

    assert not batch.ok
    assert len(batch.results) == 1
    assert batch.not_attempted == ("TESTB",)  # it would fail in exactly the same way
    result = batch.results[0]
    assert result.status == RunStatus.FAILED
    assert result.stops_batch
    assert "AlpacaAuthenticationError" in result.error
    assert "fixture unauthorized" in result.error
    assert len(alpaca_mock.requests) == 1

    runs = stored_runs(session_factory)
    assert len(runs) == 1
    assert runs[0].status == RunStatus.FAILED
    assert runs[0].error_message == result.error
    assert runs[0].finished_at is not None
    assert bar_count(session_factory) == 0


def test_error_message_stored_in_the_database_contains_no_credentials(
    ingestion_service, alpaca_mock, session_factory, settings
):
    alpaca_mock.add_error(403, "fixture forbidden")

    ingest(ingestion_service)

    key_id, secret = settings.require_alpaca_credentials()
    message = stored_runs(session_factory)[0].error_message
    assert "AlpacaPermissionError" in message
    assert key_id not in message
    assert secret not in message


def test_a_rejected_request_for_one_symbol_does_not_stop_the_others(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_error(422, "fixture invalid symbol")
    alpaca_mock.add_bars({"TESTB": [make_bar(day) for day in SESSIONS]})

    batch = ingestion_service.ingest_symbols(["TESTA", "TESTB"], start=START, end=END)

    assert not batch.ok
    assert batch.not_attempted == ()
    assert [result.status for result in batch.results] == [RunStatus.FAILED, RunStatus.SUCCEEDED]
    assert not batch.results[0].stops_batch
    assert {bar.symbol for bar in stored_bars(session_factory)} == {"TESTB"}
    assert [run.status for run in stored_runs(session_factory)] == [
        RunStatus.FAILED,
        RunStatus.SUCCEEDED,
    ]


def test_persistent_server_error_fails_the_run_after_retries(
    ingestion_service, alpaca_mock, session_factory, settings, sleeps
):
    for _ in range(settings.http_max_retries + 1):
        alpaca_mock.add_error(503, "fixture outage")

    result = ingest(ingestion_service)

    assert result.status == RunStatus.FAILED
    assert "AlpacaServerError" in result.error
    assert result.stops_batch
    assert len(sleeps) == settings.http_max_retries
    assert bar_count(session_factory) == 0


def test_transient_error_is_retried_and_the_run_succeeds(
    ingestion_service, alpaca_mock, session_factory, sleeps, make_bar
):
    alpaca_mock.add_error(500, "fixture outage")
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    result = ingest(ingestion_service)

    assert result.status == RunStatus.SUCCEEDED
    assert result.bars_inserted == 9
    assert sleeps == [1.0]


def test_failure_in_the_middle_of_pagination_stores_nothing(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars(
        {"TESTA": [make_bar(day) for day in SESSIONS[:3]]}, next_page_token="tok-1"
    )
    alpaca_mock.add_error(403, "fixture forbidden")

    result = ingest(ingestion_service)

    # A half-downloaded history is never stored as if it were complete.
    assert result.status == RunStatus.FAILED
    assert result.pages_fetched == 1
    assert bar_count(session_factory) == 0
    run = stored_runs(session_factory)[0]
    assert (run.status, run.pages_fetched, run.bars_inserted) == (RunStatus.FAILED, 1, 0)


def test_malformed_provider_response_fails_the_run(ingestion_service, alpaca_mock, session_factory):
    alpaca_mock.add_json({"unexpected": "fixture shape"})

    result = ingest(ingestion_service)

    assert result.status == RunStatus.FAILED
    assert "AlpacaResponseError" in result.error
    assert not result.stops_batch
    assert bar_count(session_factory) == 0


def test_storage_failure_marks_the_run_failed_and_stores_nothing(
    ingestion_service, alpaca_mock, session_factory, make_bar, monkeypatch
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    def explode(**_kwargs):
        raise RuntimeError("fixture storage failure")

    monkeypatch.setattr(ingestion, "build_check_rows", explode)

    with pytest.raises(RuntimeError, match="fixture storage failure"):
        ingest(ingestion_service)

    assert bar_count(session_factory) == 0  # the transaction was rolled back
    run = stored_runs(session_factory)[0]
    assert run.status == RunStatus.FAILED
    assert "fixture storage failure" in run.error_message


# =============================================================================
# Missing data detection
# =============================================================================


def test_a_missing_trading_session_is_detected(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS if day != date(2024, 1, 9)]})

    result = ingest(ingestion_service)

    assert result.status == RunStatus.COMPLETED_WITH_ISSUES
    assert (result.bars_inserted, result.missing_sessions, result.failures) == (8, 1, 1)
    gap = stored_checks(session_factory, status=CheckStatus.FAILED)[0]
    assert gap.check_name == CHECK_SESSION_GAP
    assert gap.details["gap_type"] == "interior"
    assert gap.details["missing_sessions"] == ["2024-01-09"]
    assert stored_runs(session_factory)[0].missing_sessions == 1
    # The gap is reported, not filled: no bar was fabricated for 9 January.
    assert bar_count(session_factory) == 8


def test_weekends_and_holidays_are_not_reported_as_missing(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    # 8-19 January 2024 contains a weekend and Martin Luther King Jr. Day (15 January).
    sessions = [date(2024, 1, day) for day in (8, 9, 10, 11, 12, 16, 17, 18, 19)]
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in sessions]})

    result = ingest(ingestion_service, start=date(2024, 1, 8), end=date(2024, 1, 19))

    assert result.status == RunStatus.SUCCEEDED
    assert (result.missing_sessions, result.failures, result.warnings) == (0, 0, 0)
    gap_rows = [
        check for check in stored_checks(session_factory) if check.check_name == CHECK_SESSION_GAP
    ]
    assert [(row.status, row.records_checked) for row in gap_rows] == [(CheckStatus.PASSED, 9)]


def test_history_that_starts_later_than_requested_is_a_warning(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS[4:]]})  # starts 8 January

    result = ingest(ingestion_service)

    assert result.status == RunStatus.SUCCEEDED  # provider availability is not an error
    assert (result.bars_inserted, result.missing_sessions) == (5, 4)
    assert (result.failures, result.warnings) == (0, 1)
    assert "History starts later than requested" in result.warning_messages[0]
    warning = stored_checks(session_factory, status=CheckStatus.WARNING)[0]
    assert warning.details["first_missing_session"] == "2024-01-02"
    assert warning.details["last_missing_session"] == "2024-01-05"


def test_no_data_from_the_provider_is_reported(ingestion_service, alpaca_mock, session_factory):
    alpaca_mock.add_bars({})

    result = ingest(ingestion_service)

    assert result.status == RunStatus.SUCCEEDED
    assert (result.bars_fetched, result.bars_inserted, result.missing_sessions) == (0, 0, 9)
    assert result.warnings == 1
    assert result.warning_messages[0].startswith("No data:")
    assert bar_count(session_factory) == 0


def test_gap_closes_once_the_missing_bar_arrives(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS if day != date(2024, 1, 9)]})
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    first = ingest(ingestion_service)
    second = ingest(ingestion_service)

    assert (first.missing_sessions, second.missing_sessions) == (1, 0)
    assert (second.bars_inserted, second.bars_unchanged) == (1, 8)
    assert second.status == RunStatus.SUCCEEDED


# =============================================================================
# Completed daily candles only
# =============================================================================


def test_unfinished_candle_of_the_current_session_is_never_requested(
    ingestion_service, alpaca_mock, make_bar
):
    # The fixed clock is Friday 1 March 2024, 10:00 New York: the session is open.
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 2, 29))]})

    result = ingestion_service.ingest_symbol("TESTA")

    query = alpaca_mock.query()
    assert query["start"] == "2024-01-02T05:00:00Z"  # the configured history start
    assert query["end"] == "2024-03-01T04:59:59Z"  # end of Thursday 29 February in New York
    assert result.requested_end == datetime(2024, 3, 1, 4, 59, 59, tzinfo=UTC)


def test_requested_end_in_the_future_is_moved_back_to_the_last_completed_session(
    ingestion_service, alpaca_mock, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 2, 29))]})

    window = ingestion_service.resolve_window(date(2024, 2, 26), date(2024, 12, 31))
    ingestion_service.ingest_symbol("TESTA", start=date(2024, 2, 26), end=date(2024, 12, 31))

    assert window.clamped
    assert window.last_session == date(2024, 2, 29)
    assert alpaca_mock.query()["end"] == "2024-03-01T04:59:59Z"


def test_todays_candle_is_requested_once_it_is_final(
    settings, session_factory, alpaca_client, alpaca_mock, make_bar
):
    friday_evening = datetime(2024, 3, 2, 1, 15, tzinfo=UTC)  # Friday 20:15 New York
    service = IngestionService(settings, session_factory, alpaca_client, now=lambda: friday_evening)
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 3, 1))]})

    service.ingest_symbol("TESTA", start=date(2024, 3, 1))

    assert alpaca_mock.query()["end"] == "2024-03-02T04:59:59Z"  # includes Friday 1 March


def test_window_keeps_an_earlier_requested_end(ingestion_service):
    window = ingestion_service.resolve_window(date(2024, 1, 2), date(2024, 1, 12))

    assert not window.clamped
    assert (window.first_session, window.last_session) == (date(2024, 1, 2), date(2024, 1, 12))
    assert window.start == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    assert window.end == datetime(2024, 1, 13, 4, 59, 59, tzinfo=UTC)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (date(2024, 3, 1), None, "No completed trading session"),  # today is not finished
        (date(2024, 6, 1), date(2024, 6, 30), "No completed trading session"),  # in the future
        (date(2024, 2, 1), date(2024, 1, 1), "is after end date"),
    ],
)
def test_window_without_a_completed_session_is_refused(
    ingestion_service, alpaca_mock, session_factory, start, end, message
):
    with pytest.raises(IngestionError, match=message):
        ingestion_service.ingest_symbol("TESTA", start=start, end=end)

    assert alpaca_mock.requests == []
    assert stored_runs(session_factory) == []


# =============================================================================
# Request validation
# =============================================================================


def test_symbol_outside_the_configured_universe_is_refused(
    ingestion_service, alpaca_mock, session_factory
):
    with pytest.raises(IngestionError, match="not in the configured universe: TESTA, TESTB"):
        ingest(ingestion_service, symbol="NOPE")

    assert alpaca_mock.requests == []
    assert stored_runs(session_factory) == []


def test_symbol_case_and_whitespace_are_normalised(ingestion_service, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    result = ingest(ingestion_service, symbol=" testa ")

    assert result.symbol == "TESTA"
    assert alpaca_mock.query()["symbols"] == "TESTA"


@pytest.mark.parametrize(
    ("timeframe", "message"),
    [("1Hour", "not supported yet"), ("1Min", "not supported yet"), ("1D", "Invalid timeframe")],
)
def test_timeframes_outside_this_milestone_are_refused(
    ingestion_service, alpaca_mock, timeframe, message
):
    with pytest.raises(IngestionError, match=message):
        ingest(ingestion_service, timeframe=timeframe)

    assert alpaca_mock.requests == []


def test_batch_inputs_are_checked_before_anything_is_downloaded(ingestion_service, alpaca_mock):
    with pytest.raises(IngestionError, match="not in the configured universe"):
        ingestion_service.ingest_symbols(["TESTA", "NOPE"], start=START, end=END)
    with pytest.raises(IngestionError, match="No symbols were given"):
        ingestion_service.ingest_symbols([], start=START, end=END)

    assert alpaca_mock.requests == []


def test_repeated_symbols_in_a_batch_are_ingested_once(ingestion_service, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    batch = ingestion_service.ingest_symbols(["TESTA", "testa", "TESTA"], start=START, end=END)

    assert [result.symbol for result in batch.results] == ["TESTA"]
    assert len(alpaca_mock.requests) == 1


# =============================================================================
# Ingestion history and quality records
# =============================================================================


def test_ingestion_history_is_recorded(
    ingestion_service, alpaca_mock, session_factory, fixed_now, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    result = ingest(ingestion_service)

    run = stored_runs(session_factory)[0]
    assert run.id == result.run_id
    assert (run.symbol, run.timeframe, run.status) == ("TESTA", "1Day", RunStatus.SUCCEEDED)
    assert (run.source, run.feed, run.adjustment, run.currency) == ("alpaca", "iex", "split", "USD")
    assert run.requested_start == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    assert run.requested_end == datetime(2024, 1, 13, 4, 59, 59, tzinfo=UTC)
    assert (run.started_at, run.finished_at) == (fixed_now, fixed_now)
    assert (run.pages_fetched, run.bars_fetched, run.bars_inserted) == (1, 9, 9)
    assert run.error_message is None


def test_a_clean_run_records_every_check_as_passed(
    ingestion_service, alpaca_mock, session_factory, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    result = ingest(ingestion_service)

    checks = stored_checks(session_factory)
    assert {check.check_name for check in checks} == {
        *ROW_CHECKS,
        CHECK_CONFLICTING_RECORD,
        CHECK_SESSION_GAP,
    }
    assert len(checks) == 11
    assert {check.status for check in checks} == {CheckStatus.PASSED}
    assert {check.stage for check in checks} == {CheckStage.INGESTION}
    assert {check.ingestion_run_id for check in checks} == {result.run_id}
    assert len({check.check_run_id for check in checks}) == 1
    assert {check.records_checked for check in checks} == {9}


# =============================================================================
# run_ingestion(): the entry point used by the command line
# =============================================================================


def test_run_ingestion_requires_credentials(settings):
    without_credentials = settings.model_copy(
        update={"alpaca_api_key_id": None, "alpaca_api_secret_key": None}
    )

    with pytest.raises(ConfigurationError, match="credentials are not configured"):
        run_ingestion(without_credentials, ["TESTA"])


def test_run_ingestion_creates_the_database_and_stores_bars(
    settings, alpaca_mock, session_factory, make_bar, monkeypatch
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})
    monkeypatch.setattr(
        ingestion,
        "AlpacaClient",
        lambda configured: AlpacaClient(configured, transport=alpaca_mock.transport),
    )

    batch = run_ingestion(settings, ["TESTA"], start=START, end=END)

    assert batch.ok
    assert batch.results[0].bars_inserted == 9
    assert bar_count(session_factory) == 9


# =============================================================================
# Command line: scripts/ingest.py
# =============================================================================

WINDOW_ARGS = ("--start", "2024-01-02", "--end", "2024-01-12")


def test_cli_downloads_one_ticker(run_script, mocked_provider, session_factory, make_bar):
    mocked_provider.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    exit_code, output, _ = run_script("ingest", "--symbol", "TESTA", *WINDOW_ARGS)

    assert exit_code == 0
    assert "Ingestion summary  feed=iex  adjustment=split" in output
    summary_line = next(line for line in output.splitlines() if line.startswith("TESTA"))
    assert summary_line.split() == ["TESTA", "succeeded", "9", "9", "0", "0", "0", "0", "0"]
    assert bar_count(session_factory) == 9
    assert [request.url.params["symbols"] for request in mocked_provider.requests] == ["TESTA"]


def test_cli_downloads_all_tickers_of_the_universe(
    run_script, mocked_provider, session_factory, make_bar
):
    mocked_provider.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})
    mocked_provider.add_bars({"TESTB": [make_bar(day) for day in SESSIONS]})

    exit_code, output, _ = run_script("ingest", "--all", *WINDOW_ARGS)

    assert exit_code == 0
    assert [request.url.params["symbols"] for request in mocked_provider.requests] == [
        "TESTA",
        "TESTB",
    ]
    assert bar_count(session_factory) == 18
    assert "TESTA" in output
    assert "TESTB" in output


@pytest.mark.parametrize(
    "symbol_args",
    [
        ("--symbol", "TESTA,TESTB"),
        ("--symbol", "TESTA", "--symbol", "testb"),
        ("-s", "TESTA", "-s", "TESTB"),
    ],
)
def test_cli_accepts_several_tickers(
    run_script, mocked_provider, session_factory, make_bar, symbol_args
):
    mocked_provider.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})
    mocked_provider.add_bars({"TESTB": [make_bar(day) for day in SESSIONS]})

    exit_code, _, _ = run_script("ingest", *symbol_args, *WINDOW_ARGS)

    assert exit_code == 0
    assert bar_count(session_factory) == 18


def test_cli_can_be_repeated_without_creating_duplicates(
    run_script, mocked_provider, session_factory, make_bar
):
    for _ in range(2):
        mocked_provider.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    first_code, _, _ = run_script("ingest", "--symbol", "TESTA", *WINDOW_ARGS)
    second_code, output, _ = run_script("ingest", "--symbol", "TESTA", *WINDOW_ARGS)

    assert (first_code, second_code) == (0, 0)
    summary_line = next(line for line in output.splitlines() if line.startswith("TESTA"))
    assert summary_line.split() == ["TESTA", "succeeded", "9", "0", "9", "0", "0", "0", "0"]
    assert bar_count(session_factory) == 9


def test_cli_creates_the_database_when_it_does_not_exist(
    run_script, mocked_provider, settings, make_bar
):
    # No ``engine`` fixture: the database file does not exist before the command runs.
    mocked_provider.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})

    exit_code, _, _ = run_script("ingest", "--symbol", "TESTA", *WINDOW_ARGS)

    assert exit_code == 0


@pytest.mark.parametrize("argv", [(), ("--symbol", "TESTA", "--all")])
def test_cli_requires_exactly_one_of_symbol_or_all(run_script, mocked_provider, argv):
    exit_code, _, errors = run_script("ingest", *argv)

    assert exit_code == 2
    assert "--symbol" in errors
    assert "--all" in errors
    assert mocked_provider.requests == []


def test_cli_rejects_a_malformed_date(run_script, mocked_provider):
    exit_code, _, errors = run_script("ingest", "--symbol", "TESTA", "--start", "01/02/2024")

    assert exit_code == 2
    assert "is not a date; use the form YYYY-MM-DD" in errors
    assert mocked_provider.requests == []


def test_cli_explains_missing_credentials(run_script, settings, session_factory):
    without_credentials = settings.model_copy(
        update={"alpaca_api_key_id": None, "alpaca_api_secret_key": None}
    )

    exit_code, _, errors = run_script(
        "ingest", "--symbol", "TESTA", use_settings=without_credentials
    )

    assert exit_code == 2
    assert "Copy .env.example to .env" in errors
    assert bar_count(session_factory) == 0


def test_cli_rejects_a_symbol_outside_the_universe(run_script, mocked_provider):
    exit_code, _, errors = run_script("ingest", "--symbol", "NOPE", *WINDOW_ARGS)

    assert exit_code == 2
    assert "not in the configured universe" in errors
    assert mocked_provider.requests == []


def test_cli_reports_a_failed_run_with_exit_code_1(
    run_script, mocked_provider, session_factory, settings
):
    mocked_provider.add_error(401, "fixture unauthorized")

    exit_code, output, _ = run_script("ingest", "--all", *WINDOW_ARGS)

    assert exit_code == 1
    assert "Errors:" in output
    assert "AlpacaAuthenticationError" in output
    assert "Not attempted because the error above would affect them too: TESTB" in output
    key_id, secret = settings.require_alpaca_credentials()
    assert key_id not in output
    assert secret not in output
    assert bar_count(session_factory) == 0


def test_cli_reports_data_issues_and_warnings(
    run_script, mocked_provider, session_factory, make_bar
):
    payload = [make_bar(day) for day in SESSIONS[2:]]  # history starts on 4 January
    payload[3] = make_bar(SESSIONS[5], v=-5)  # 9 January has a negative volume
    mocked_provider.add_bars({"TESTA": payload})

    exit_code, output, _ = run_script("ingest", "--symbol", "TESTA", *WINDOW_ARGS)

    assert exit_code == 0  # the run finished; the data problems are recorded, not hidden
    summary_line = next(line for line in output.splitlines() if line.startswith("TESTA"))
    assert summary_line.split() == [
        "TESTA",
        "completed_with_issues",
        "7",
        "6",
        "0",
        "0",
        "1",
        "0",
        "3",
    ]
    assert "Warnings:" in output
    assert "History starts later than requested" in output
    assert "Nothing was repaired or overwritten" in output
    assert bar_count(session_factory) == 6
