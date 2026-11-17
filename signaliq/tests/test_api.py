"""API tests: every endpoint, filtering, input validation and error responses.

The database holds only SYNTHETIC TEST FIXTURES with invented values (see
``conftest.py``). Nothing in this file is real market data.
"""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.data.validation import validate_stored_bars
from app.database import session_scope
from app.main import create_app

#: The nine trading sessions from 2 to 12 January 2024.
SESSIONS = [date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9, 10, 11, 12)]

BAR_FIELDS = {
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


def timestamps(response) -> list[str]:
    return [bar["timestamp"] for bar in response.json()["bars"]]


# =============================================================================
# The application starts
# =============================================================================


def test_application_starts_and_documents_its_endpoints(api_client):
    schema = api_client.get("/openapi.json")

    assert schema.status_code == 200
    assert set(schema.json()["paths"]) == {
        "/health",
        "/symbols",
        "/bars/{symbol}",
        "/ingestion/runs",
        "/quality/report",
    }
    assert api_client.get("/docs").status_code == 200


def test_index_points_to_the_documentation(api_client):
    body = api_client.get("/").json()

    assert body["name"] == "SignalIQ"
    assert body["documentation"] == "/docs"
    assert "/bars/{symbol}" in body["endpoints"]


def test_api_starts_without_alpaca_credentials(settings, engine):
    without_credentials = settings.model_copy(
        update={"alpaca_api_key_id": None, "alpaca_api_secret_key": None}
    )

    with TestClient(create_app(without_credentials)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/symbols").status_code == 200


def test_api_creates_the_database_tables_on_startup(settings):
    # No ``engine`` fixture here: the application itself must create the tables.
    with TestClient(create_app(settings)) as client:
        assert client.get("/bars/TESTA").json()["total"] == 0
        assert client.get("/ingestion/runs").json()["total"] == 0


def test_api_is_read_only(api_client):
    for method in ("post", "put", "patch", "delete"):
        assert getattr(api_client, method)("/bars/TESTA").status_code == 405


# =============================================================================
# GET /health
# =============================================================================


def test_health(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["version"] == __version__
    assert body["time_utc"].endswith("Z")


# =============================================================================
# GET /symbols
# =============================================================================


def test_symbols_lists_the_universe_even_when_nothing_is_stored(api_client):
    body = api_client.get("/symbols").json()

    assert body["count"] == 2
    assert "does not represent the entire US stock market" in body["note"]
    assert body["symbols"] == [
        {"symbol": "TESTA", "bar_count": 0, "coverage": []},
        {"symbol": "TESTB", "bar_count": 0, "coverage": []},
    ]


def test_symbols_reports_what_is_stored(api_client, store_bars):
    store_bars("TESTA", SESSIONS)
    store_bars("TESTA", SESSIONS[:2], timeframe="1Hour")
    store_bars("ZZZZ", SESSIONS[:1])  # not in the universe: must not be listed

    body = api_client.get("/symbols").json()

    assert [item["symbol"] for item in body["symbols"]] == ["TESTA", "TESTB"]
    testa = body["symbols"][0]
    assert testa["bar_count"] == 11
    assert testa["coverage"] == [
        {
            "timeframe": "1Day",
            "bar_count": 9,
            "first_timestamp": "2024-01-02T05:00:00Z",
            "last_timestamp": "2024-01-12T05:00:00Z",
        },
        {
            "timeframe": "1Hour",
            "bar_count": 2,
            "first_timestamp": "2024-01-02T05:00:00Z",
            "last_timestamp": "2024-01-03T05:00:00Z",
        },
    ]
    assert body["symbols"][1] == {"symbol": "TESTB", "bar_count": 0, "coverage": []}


# =============================================================================
# GET /bars/{symbol}
# =============================================================================


def test_bars_returns_the_stored_records(api_client, store_bars):
    store_bars("TESTA", SESSIONS)

    response = api_client.get("/bars/TESTA")

    assert response.status_code == 200
    body = response.json()
    assert (body["symbol"], body["timeframe"]) == ("TESTA", "1Day")
    assert (body["total"], body["count"], body["limit"], body["offset"]) == (9, 9, 1000, 0)
    assert (body["start"], body["end"]) == (None, None)
    assert body["bars"][0] == {
        "symbol": "TESTA",
        "timeframe": "1Day",
        "timestamp": "2024-01-02T05:00:00Z",
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
    assert all(set(bar) == BAR_FIELDS for bar in body["bars"])
    assert timestamps(response) == sorted(timestamps(response))  # oldest first by default


def test_bars_only_returns_the_requested_symbol(api_client, store_bars):
    store_bars("TESTA", SESSIONS)
    store_bars("TESTB", SESSIONS[:3], close=10.7)

    body = api_client.get("/bars/TESTB").json()

    assert body["total"] == 3
    assert {bar["symbol"] for bar in body["bars"]} == {"TESTB"}
    assert {bar["close"] for bar in body["bars"]} == {10.7}


def test_bars_symbol_is_case_insensitive(api_client, store_bars):
    store_bars("TESTA", SESSIONS)

    body = api_client.get("/bars/testa").json()

    assert (body["symbol"], body["total"]) == ("TESTA", 9)


def test_known_symbol_without_data_returns_an_empty_list(api_client):
    response = api_client.get("/bars/TESTB")

    assert response.status_code == 200
    assert response.json()["bars"] == []
    assert response.json()["total"] == 0


def test_bars_filter_by_timeframe(api_client, store_bars):
    store_bars("TESTA", SESSIONS)
    store_bars("TESTA", SESSIONS[:2], timeframe="1Hour")

    daily = api_client.get("/bars/TESTA").json()
    explicit_daily = api_client.get("/bars/TESTA", params={"timeframe": "1Day"}).json()
    hourly = api_client.get("/bars/TESTA", params={"timeframe": "1Hour"}).json()
    weekly = api_client.get("/bars/TESTA", params={"timeframe": "1Week"}).json()

    assert daily["total"] == explicit_daily["total"] == 9
    assert (hourly["timeframe"], hourly["total"]) == ("1Hour", 2)
    assert {bar["timeframe"] for bar in hourly["bars"]} == {"1Hour"}
    assert weekly["total"] == 0


def test_bars_filter_by_start_and_end_dates_is_inclusive(api_client, store_bars):
    store_bars("TESTA", SESSIONS)

    response = api_client.get("/bars/TESTA", params={"start": "2024-01-03", "end": "2024-01-08"})

    assert response.status_code == 200
    assert timestamps(response) == [
        "2024-01-03T05:00:00Z",
        "2024-01-04T05:00:00Z",
        "2024-01-05T05:00:00Z",
        "2024-01-08T05:00:00Z",
    ]
    body = response.json()
    # Dates are New York market dates; the applied bounds are echoed in UTC.
    assert body["start"] == "2024-01-03T05:00:00Z"
    assert body["end"] == "2024-01-09T04:59:59.999999Z"
    assert body["total"] == 4


def test_bars_filter_by_start_only_or_end_only(api_client, store_bars):
    store_bars("TESTA", SESSIONS)

    from_start = api_client.get("/bars/TESTA", params={"start": "2024-01-11"})
    until_end = api_client.get("/bars/TESTA", params={"end": "2024-01-03"})

    assert timestamps(from_start) == ["2024-01-11T05:00:00Z", "2024-01-12T05:00:00Z"]
    assert timestamps(until_end) == ["2024-01-02T05:00:00Z", "2024-01-03T05:00:00Z"]


def test_bars_filter_by_a_single_day(api_client, store_bars):
    store_bars("TESTA", SESSIONS)

    response = api_client.get("/bars/TESTA", params={"start": "2024-01-05", "end": "2024-01-05"})

    assert timestamps(response) == ["2024-01-05T05:00:00Z"]


def test_bars_filter_by_exact_utc_date_times(api_client, store_bars):
    store_bars("TESTA", SESSIONS)

    response = api_client.get(
        "/bars/TESTA",
        params={"start": "2024-01-04T05:00:00Z", "end": "2024-01-08T04:59:59Z"},
    )

    # The end is one second before the 8 January bar, so that bar is excluded.
    assert timestamps(response) == ["2024-01-04T05:00:00Z", "2024-01-05T05:00:00Z"]


def test_bars_date_time_with_another_offset_is_converted_to_utc(api_client, store_bars):
    store_bars("TESTA", SESSIONS)

    response = api_client.get("/bars/TESTA", params={"start": "2024-01-11T00:00:00-05:00"})

    assert response.json()["start"] == "2024-01-11T05:00:00Z"
    assert timestamps(response) == ["2024-01-11T05:00:00Z", "2024-01-12T05:00:00Z"]


def test_bars_date_filter_follows_daylight_saving_time(api_client, store_bars):
    store_bars("TESTA", [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3)])

    response = api_client.get("/bars/TESTA", params={"start": "2024-07-02", "end": "2024-07-02"})

    # In summer, midnight in New York is 04:00 UTC.
    assert timestamps(response) == ["2024-07-02T04:00:00Z"]
    assert response.json()["start"] == "2024-07-02T04:00:00Z"


def test_bars_paging_and_ordering(api_client, store_bars):
    store_bars("TESTA", SESSIONS)

    first_page = api_client.get("/bars/TESTA", params={"limit": 4})
    second_page = api_client.get("/bars/TESTA", params={"limit": 4, "offset": 4})
    last_page = api_client.get("/bars/TESTA", params={"limit": 4, "offset": 8})
    newest_first = api_client.get("/bars/TESTA", params={"order": "desc", "limit": 2})

    assert [page.json()["count"] for page in (first_page, second_page, last_page)] == [4, 4, 1]
    assert {page.json()["total"] for page in (first_page, second_page, last_page)} == {9}
    assert timestamps(first_page) + timestamps(second_page) + timestamps(last_page) == [
        f"2024-01-{day:02d}T05:00:00Z" for day in (2, 3, 4, 5, 8, 9, 10, 11, 12)
    ]
    assert timestamps(newest_first) == ["2024-01-12T05:00:00Z", "2024-01-11T05:00:00Z"]


# --- Input validation and error responses ------------------------------------


def test_unknown_symbol_returns_404_with_the_supported_symbols(api_client):
    response = api_client.get("/bars/NOPE")

    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown symbol 'NOPE'. Supported symbols: TESTA, TESTB."}


@pytest.mark.parametrize("symbol", ["TEST!", "123", "WAYTOOLONGSYMBOLNAME", "A B"])
def test_malformed_symbol_returns_422(api_client, symbol):
    response = api_client.get(f"/bars/{symbol}")

    assert response.status_code == 422
    assert "Invalid symbol" in response.json()["detail"]


@pytest.mark.parametrize("timeframe", ["1D", "daily", "0Min", "60Min", "1day"])
def test_invalid_timeframe_returns_422(api_client, timeframe):
    response = api_client.get("/bars/TESTA", params={"timeframe": timeframe})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert f"Invalid timeframe '{timeframe}'" in detail
    assert "1Day" in detail  # the message shows valid examples


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("start", "yesterday"),
        ("start", "2024-13-45"),
        ("end", "01/02/2024"),
        ("end", "2024-01-02T25:00:00Z"),
    ],
)
def test_invalid_date_returns_422_with_guidance(api_client, parameter, value):
    response = api_client.get("/bars/TESTA", params={parameter: value})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail.startswith(f"Invalid {parameter} '{value}'")
    assert "YYYY-MM-DD" in detail


def test_date_time_without_utc_offset_returns_422(api_client):
    response = api_client.get("/bars/TESTA", params={"start": "2024-01-02T05:00:00"})

    assert response.status_code == 422
    assert "needs a UTC offset" in response.json()["detail"]


def test_start_after_end_returns_422(api_client):
    response = api_client.get("/bars/TESTA", params={"start": "2024-02-01", "end": "2024-01-01"})

    assert response.status_code == 422
    assert response.json()["detail"] == "start (2024-02-01) must not be after end (2024-01-01)."


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 10_001},
        {"limit": "many"},
        {"offset": -1},
        {"order": "sideways"},
    ],
)
def test_out_of_range_paging_values_return_422(api_client, params):
    response = api_client.get("/bars/TESTA", params=params)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", next(iter(params))]


# =============================================================================
# GET /ingestion/runs
# =============================================================================


def test_ingestion_runs_empty(api_client):
    assert api_client.get("/ingestion/runs").json() == {
        "total": 0,
        "count": 0,
        "limit": 50,
        "offset": 0,
        "runs": [],
    }


def test_ingestion_runs_lists_history_newest_first(
    api_client, ingestion_service, alpaca_mock, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})
    alpaca_mock.add_error(422, "fixture rejected request")
    ingestion_service.ingest_symbols(
        ["TESTA", "TESTB"], start=date(2024, 1, 2), end=date(2024, 1, 12)
    )

    body = api_client.get("/ingestion/runs").json()

    assert (body["total"], body["count"]) == (2, 2)
    newest, oldest = body["runs"]
    assert (newest["symbol"], newest["status"]) == ("TESTB", "failed")
    assert "fixture rejected request" in newest["error_message"]
    assert (oldest["symbol"], oldest["status"]) == ("TESTA", "succeeded")
    assert oldest["id"] < newest["id"]
    assert (oldest["bars_fetched"], oldest["bars_inserted"], oldest["pages_fetched"]) == (9, 9, 1)
    assert (oldest["source"], oldest["feed"], oldest["adjustment"]) == ("alpaca", "iex", "split")
    assert oldest["requested_start"] == "2024-01-02T05:00:00Z"
    assert oldest["requested_end"] == "2024-01-13T04:59:59Z"
    assert oldest["error_message"] is None


def test_ingestion_runs_filters_and_paging(api_client, ingestion_service, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})
    alpaca_mock.add_error(422, "fixture rejected request")
    alpaca_mock.add_bars({"TESTA": [make_bar(day) for day in SESSIONS]})
    window = {"start": date(2024, 1, 2), "end": date(2024, 1, 12)}
    ingestion_service.ingest_symbols(["TESTA", "TESTB"], **window)
    ingestion_service.ingest_symbol("TESTA", **window)

    by_symbol = api_client.get("/ingestion/runs", params={"symbol": "testa"}).json()
    failed = api_client.get("/ingestion/runs", params={"status": "failed"}).json()
    paged = api_client.get("/ingestion/runs", params={"limit": 1, "offset": 1}).json()

    assert [run["symbol"] for run in by_symbol["runs"]] == ["TESTA", "TESTA"]
    assert [run["symbol"] for run in failed["runs"]] == ["TESTB"]
    assert (paged["total"], paged["count"]) == (3, 1)
    assert paged["runs"][0]["symbol"] == "TESTB"


@pytest.mark.parametrize(
    ("params", "status_code", "expected"),
    [
        ({"status": "exploded"}, 422, "Invalid status 'exploded'"),
        ({"symbol": "NOPE"}, 404, "Unknown symbol 'NOPE'"),
        ({"symbol": "!!"}, 422, "Invalid symbol"),
    ],
)
def test_ingestion_runs_rejects_invalid_filters(api_client, params, status_code, expected):
    response = api_client.get("/ingestion/runs", params=params)

    assert response.status_code == status_code
    assert expected in response.json()["detail"]


def test_ingestion_runs_rejects_out_of_range_paging(api_client):
    assert api_client.get("/ingestion/runs", params={"limit": 0}).status_code == 422
    assert api_client.get("/ingestion/runs", params={"limit": 501}).status_code == 422
    assert api_client.get("/ingestion/runs", params={"offset": -1}).status_code == 422


# =============================================================================
# GET /quality/report
# =============================================================================


def test_quality_report_on_an_empty_database(api_client):
    body = api_client.get("/quality/report").json()

    assert body["bars_stored"] == 0
    assert body["totals"] == {"passed": 0, "failed": 0, "warning": 0, "skipped": 0}
    assert body["checks"] == []
    assert body["latest_stored_validation"] is None
    assert (body["issue_count"], body["issues"]) == (0, [])
    assert body["generated_at"].endswith("Z")


def test_quality_report_shows_validation_failures(
    api_client, ingestion_service, alpaca_mock, make_bar
):
    payload = [make_bar(day) for day in SESSIONS]
    payload[1] = make_bar(SESSIONS[1], h=8.0)  # high below low on 3 January
    alpaca_mock.add_bars({"TESTA": payload})
    ingestion_service.ingest_symbol("TESTA", start=date(2024, 1, 2), end=date(2024, 1, 12))

    body = api_client.get("/quality/report").json()

    assert body["bars_stored"] == 8
    # 11 checks ran: 9 passed, and two produced one finding each.
    assert body["totals"] == {"passed": 9, "failed": 2, "warning": 0, "skipped": 0}
    by_check = {check["check_name"]: check for check in body["checks"]}
    assert len(by_check) == 11
    assert by_check["high_below_low"]["failed"] == 1
    assert by_check["high_below_low"]["description"] == "High is not lower than low"
    assert by_check["trading_session_gap"]["failed"] == 1
    assert by_check["negative_volume"] == {
        "check_name": "negative_volume",
        "description": "Volume is not negative",
        "passed": 1,
        "failed": 0,
        "warning": 0,
        "skipped": 0,
    }

    assert body["issue_count"] == 2
    gap_issue, price_issue = body["issues"]  # newest first
    assert price_issue["check_name"] == "high_below_low"
    assert price_issue["status"] == "failed"
    assert price_issue["stage"] == "ingestion"
    assert (price_issue["symbol"], price_issue["timeframe"]) == ("TESTA", "1Day")
    assert price_issue["bar_timestamp"] == "2024-01-03T05:00:00Z"
    assert price_issue["message"] == "High 8.0 is lower than low 9.0"
    assert price_issue["details"]["record"] == payload[1]
    assert price_issue["ingestion_run_id"] == 1
    assert gap_issue["check_name"] == "trading_session_gap"
    assert gap_issue["details"]["missing_sessions"] == ["2024-01-03"]
    assert body["latest_stored_validation"] is None  # scripts/validate.py has not run


def test_quality_report_includes_the_latest_validation_of_stored_bars(
    api_client, settings, session_factory, store_bars, fixed_now
):
    store_bars("TESTA", SESSIONS)
    store_bars("TESTB", SESSIONS, volume=-1)
    with session_scope(session_factory) as session:
        validate_stored_bars(session, settings, now=fixed_now)
    with session_scope(session_factory) as session:
        latest = validate_stored_bars(session, settings, now=fixed_now)

    body = api_client.get("/quality/report").json()

    assert body["bars_stored"] == 18
    summary = body["latest_stored_validation"]
    assert summary["check_run_id"] == latest.check_run_id
    # Per symbol: 10 checks. TESTA: 9 passed + 1 stale-data warning.
    # TESTB: 8 passed + 9 negative volumes + 1 stale-data warning.
    assert (summary["passed"], summary["failed"], summary["warning"]) == (17, 9, 2)
    # Totals cover both validation runs.
    assert body["totals"] == {"passed": 34, "failed": 18, "warning": 4, "skipped": 0}
    assert body["issue_count"] == 22


def test_quality_report_filters(api_client, settings, session_factory, store_bars, fixed_now):
    store_bars("TESTA", SESSIONS)
    store_bars("TESTB", SESSIONS, volume=-1)
    with session_scope(session_factory) as session:
        validate_stored_bars(session, settings, now=fixed_now)

    only_a = api_client.get("/quality/report", params={"symbol": "testa"}).json()
    only_b = api_client.get("/quality/report", params={"symbol": "TESTB", "limit": 3}).json()
    stored = api_client.get("/quality/report", params={"stage": "stored"}).json()
    ingestion = api_client.get("/quality/report", params={"stage": "ingestion"}).json()

    assert (only_a["symbol"], only_a["bars_stored"]) == ("TESTA", 9)
    assert only_a["totals"] == {"passed": 9, "failed": 0, "warning": 1, "skipped": 0}
    assert only_a["latest_stored_validation"]["failed"] == 0

    assert only_b["totals"]["failed"] == 9
    assert only_b["issue_count"] == 10
    assert len(only_b["issues"]) == 3  # limited, newest first
    assert {issue["symbol"] for issue in only_b["issues"]} == {"TESTB"}
    assert only_b["latest_stored_validation"]["failed"] == 9

    assert stored["totals"]["failed"] == 9
    assert ingestion["totals"] == {"passed": 0, "failed": 0, "warning": 0, "skipped": 0}
    assert ingestion["latest_stored_validation"] is None


@pytest.mark.parametrize(
    ("params", "status_code", "expected"),
    [
        ({"stage": "elsewhere"}, 422, "Invalid stage 'elsewhere'"),
        ({"symbol": "NOPE"}, 404, "Unknown symbol 'NOPE'"),
    ],
)
def test_quality_report_rejects_invalid_filters(api_client, params, status_code, expected):
    response = api_client.get("/quality/report", params=params)

    assert response.status_code == status_code
    assert expected in response.json()["detail"]


def test_quality_report_rejects_out_of_range_limit(api_client):
    assert api_client.get("/quality/report", params={"limit": 0}).status_code == 422
    assert api_client.get("/quality/report", params={"limit": 1001}).status_code == 422
