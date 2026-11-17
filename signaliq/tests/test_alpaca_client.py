"""Alpaca client tests: request shape, pagination, error handling, retries.

Every HTTP response here is a mocked TEST FIXTURE that imitates the shape of
Alpaca's documented bars payload. No test in this file touches the network and
no value in it is real market data.
"""

import logging
from datetime import UTC, date, datetime

import httpx
import pytest
from pydantic import SecretStr

from app.config import (
    DEFAULT_SYMBOLS,
    PROJECT_ROOT,
    ConfigurationError,
    Settings,
    load_settings,
)
from app.data.alpaca_client import (
    AlpacaAPIError,
    AlpacaAuthenticationError,
    AlpacaClient,
    AlpacaNetworkError,
    AlpacaPermissionError,
    AlpacaRateLimitError,
    AlpacaRequestError,
    AlpacaResponseError,
    AlpacaServerError,
)

START = datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
END = datetime(2024, 1, 13, 4, 59, 59, tzinfo=UTC)


def fetch(client: AlpacaClient, symbols=("TESTA",), **kwargs):
    return client.fetch_bars(list(symbols), timeframe="1Day", start=START, end=END, **kwargs)


# --- Request shape -------------------------------------------------------------


def test_request_targets_the_official_bars_endpoint(alpaca_client, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    fetch(alpaca_client)

    request = alpaca_mock.requests[0]
    assert request.method == "GET"
    assert request.url.scheme == "https"
    assert request.url.host == "data.alpaca.markets"
    assert request.url.path == "/v2/stocks/bars"


def test_request_sends_the_configured_feed_adjustment_and_window(
    alpaca_client, alpaca_mock, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    fetch(alpaca_client)

    assert alpaca_mock.query() == {
        "symbols": "TESTA",
        "timeframe": "1Day",
        "start": "2024-01-02T05:00:00Z",
        "end": "2024-01-13T04:59:59Z",
        "limit": "10000",
        "adjustment": "split",
        "feed": "iex",
        "currency": "USD",
        "sort": "asc",
    }


def test_credentials_travel_in_the_alpaca_headers_and_never_in_the_url(
    alpaca_client, alpaca_mock, settings, make_bar
):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    fetch(alpaca_client)

    request = alpaca_mock.requests[0]
    key_id, secret = settings.require_alpaca_credentials()
    assert request.headers["APCA-API-KEY-ID"] == key_id
    assert request.headers["APCA-API-SECRET-KEY"] == secret
    assert key_id not in str(request.url)
    assert secret not in str(request.url)


def test_multiple_symbols_and_custom_page_limit(alpaca_client, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))], "TESTB": []})

    fetch(alpaca_client, symbols=("TESTA", "TESTB"), limit=50)

    assert alpaca_mock.query()["symbols"] == "TESTA,TESTB"
    assert alpaca_mock.query()["limit"] == "50"


def test_http_timeout_is_configured(alpaca_client, settings):
    timeout = alpaca_client._http.timeout
    assert timeout.connect == timeout.read == timeout.write == settings.http_timeout_seconds


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"symbols": []}, "At least one symbol"),
        ({"symbols": ["not a symbol!"]}, "Invalid symbol"),
        ({"timeframe": "1D"}, "Invalid timeframe"),
        ({"start": datetime(2024, 1, 2)}, "timezone-aware"),
        ({"end": datetime(2024, 1, 2)}, "timezone-aware"),
        ({"start": END, "end": START}, "start must not be after end"),
        ({"limit": 0}, "limit must be between"),
        ({"limit": 10_001}, "limit must be between"),
    ],
)
def test_invalid_arguments_are_rejected_before_any_request(
    alpaca_client, alpaca_mock, kwargs, message
):
    arguments = {"symbols": ["TESTA"], "timeframe": "1Day", "start": START, "end": END}
    arguments.update(kwargs)
    symbols = arguments.pop("symbols")

    with pytest.raises(ValueError, match=message):
        alpaca_client.fetch_bars(symbols, **arguments)

    assert alpaca_mock.requests == []


# --- Pagination ------------------------------------------------------------------


def test_pagination_follows_next_page_token_until_exhausted(alpaca_client, alpaca_mock, make_bar):
    days = [date(2024, 1, day) for day in (2, 3, 4, 5, 8)]
    alpaca_mock.add_bars({"TESTA": [make_bar(days[0]), make_bar(days[1])]}, next_page_token="tok-1")
    alpaca_mock.add_bars({"TESTA": [make_bar(days[2]), make_bar(days[3])]}, next_page_token="tok-2")
    alpaca_mock.add_bars({"TESTA": [make_bar(days[4])]}, next_page_token=None)

    bars = fetch(alpaca_client)

    assert len(alpaca_mock.requests) == 3
    assert alpaca_mock.pending == 0
    assert "page_token" not in alpaca_mock.query(0)
    assert alpaca_mock.query(1)["page_token"] == "tok-1"
    assert alpaca_mock.query(2)["page_token"] == "tok-2"
    # Every record from every page is returned, in order, with nothing dropped.
    assert bars == {"TESTA": [make_bar(day) for day in days]}


def test_pagination_keeps_all_other_parameters_identical(alpaca_client, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]}, next_page_token="tok-1")
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 3))]})

    fetch(alpaca_client)

    first, second = alpaca_mock.query(0), alpaca_mock.query(1)
    second.pop("page_token")
    assert first == second


def test_pagination_merges_symbols_that_span_pages(alpaca_client, alpaca_mock, make_bar):
    alpaca_mock.add_bars(
        {"TESTA": [make_bar(date(2024, 1, 2)), make_bar(date(2024, 1, 3))]},
        next_page_token="tok-1",
    )
    alpaca_mock.add_bars(
        {"TESTA": [make_bar(date(2024, 1, 4))], "TESTB": [make_bar(date(2024, 1, 2))]},
        next_page_token="tok-2",
    )
    alpaca_mock.add_bars({"TESTB": [make_bar(date(2024, 1, 3))]})

    bars = fetch(alpaca_client, symbols=("TESTA", "TESTB"))

    assert [bar["t"] for bar in bars["TESTA"]] == [
        "2024-01-02T05:00:00Z",
        "2024-01-03T05:00:00Z",
        "2024-01-04T05:00:00Z",
    ]
    assert [bar["t"] for bar in bars["TESTB"]] == ["2024-01-02T05:00:00Z", "2024-01-03T05:00:00Z"]


def test_iter_bar_pages_reports_page_numbers_and_counts(alpaca_client, alpaca_mock, make_bar):
    alpaca_mock.add_bars(
        {"TESTA": [make_bar(date(2024, 1, 2)), make_bar(date(2024, 1, 3))]},
        next_page_token="tok-1",
    )
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 4))]})

    pages = list(alpaca_client.iter_bar_pages(["TESTA"], timeframe="1Day", start=START, end=END))

    assert [(page.page_number, page.bar_count, page.next_page_token) for page in pages] == [
        (1, 2, "tok-1"),
        (2, 1, None),
    ]


@pytest.mark.parametrize("empty_bars", [{}, None])
def test_no_data_in_window_returns_no_bars(alpaca_client, alpaca_mock, empty_bars):
    alpaca_mock.add_json({"bars": empty_bars, "next_page_token": None})

    assert fetch(alpaca_client) == {}
    assert len(alpaca_mock.requests) == 1


def test_repeated_page_token_stops_instead_of_looping_forever(alpaca_client, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]}, next_page_token="tok-1")
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 3))]}, next_page_token="tok-1")

    with pytest.raises(AlpacaResponseError, match="same next_page_token twice"):
        fetch(alpaca_client)

    # The repeat is noticed as soon as the second page hands back the same token.
    assert len(alpaca_mock.requests) == 2


# --- API error handling ----------------------------------------------------------


@pytest.mark.parametrize(
    ("status_code", "error_type", "expected_text"),
    [
        (401, AlpacaAuthenticationError, "APCA_API_KEY_ID"),
        (403, AlpacaPermissionError, "never substitutes a different feed"),
        (400, AlpacaRequestError, "rejected the request parameters"),
        (422, AlpacaRequestError, "rejected the request parameters"),
        (404, AlpacaRequestError, "endpoint not found"),
        (418, AlpacaRequestError, "Unexpected response"),
    ],
)
def test_client_errors_raise_clear_exceptions_without_retrying(
    alpaca_client, alpaca_mock, sleeps, status_code, error_type, expected_text
):
    alpaca_mock.add_error(status_code, "fixture provider message")

    with pytest.raises(error_type) as excinfo:
        fetch(alpaca_client)

    message = str(excinfo.value)
    assert expected_text in message
    assert "fixture provider message" in message  # the provider's own words are kept
    assert f"HTTP {status_code}" in message
    assert excinfo.value.status_code == status_code
    assert isinstance(excinfo.value, AlpacaAPIError)
    assert len(alpaca_mock.requests) == 1
    assert sleeps == []


def test_refused_feed_is_never_replaced_by_another_feed(alpaca_client, alpaca_mock):
    alpaca_mock.add_error(403, "subscription does not permit querying this feed")

    with pytest.raises(AlpacaPermissionError, match="feed 'iex'"):
        fetch(alpaca_client)

    # Exactly one request was made, with the configured feed: no silent fallback.
    assert [request.url.params["feed"] for request in alpaca_mock.requests] == ["iex"]


def test_error_message_uses_plain_text_bodies_too(alpaca_client, alpaca_mock):
    alpaca_mock.add_text("fixture plain text refusal", status_code=403)

    with pytest.raises(AlpacaPermissionError, match="fixture plain text refusal"):
        fetch(alpaca_client)


def test_request_id_is_reported_for_support(alpaca_client, alpaca_mock):
    alpaca_mock.add_error(400, "bad window", headers={"X-Request-ID": "fixture-request-id"})

    with pytest.raises(AlpacaRequestError) as excinfo:
        fetch(alpaca_client)

    assert excinfo.value.request_id == "fixture-request-id"
    assert "fixture-request-id" in str(excinfo.value)


@pytest.mark.parametrize(
    ("payload", "expected_text"),
    [
        ({"next_page_token": None}, "expected 'bars' field"),
        (["not", "an", "object"], "expected 'bars' field"),
        ({"bars": ["not", "a", "mapping"]}, "not an object keyed by symbol"),
        ({"bars": {"TESTA": "not a list"}}, "'bars.TESTA' is not a list"),
        ({"bars": {}, "next_page_token": 12345}, "'next_page_token' is not a string"),
        ({"bars": {}, "next_page_token": None, "currency": "EUR"}, "unexpected currency"),
    ],
)
def test_malformed_success_bodies_raise_response_error(
    alpaca_client, alpaca_mock, payload, expected_text
):
    alpaca_mock.add_json(payload)

    with pytest.raises(AlpacaResponseError, match=expected_text):
        fetch(alpaca_client)


def test_non_json_success_body_raises_response_error(alpaca_client, alpaca_mock):
    alpaca_mock.add_text("<html>fixture gateway page</html>")

    with pytest.raises(AlpacaResponseError, match="not valid JSON"):
        fetch(alpaca_client)


def test_matching_currency_in_response_is_accepted(alpaca_client, alpaca_mock, make_bar):
    alpaca_mock.add_json(
        {
            "bars": {"TESTA": [make_bar(date(2024, 1, 2))]},
            "next_page_token": None,
            "currency": "USD",
        }
    )

    assert len(fetch(alpaca_client)["TESTA"]) == 1


# --- Rate limiting ---------------------------------------------------------------


def test_rate_limit_waits_for_retry_after_then_succeeds(
    alpaca_client, alpaca_mock, sleeps, make_bar
):
    alpaca_mock.add_error(429, "too many requests", headers={"Retry-After": "7"})
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    bars = fetch(alpaca_client)

    assert len(bars["TESTA"]) == 1
    assert len(alpaca_mock.requests) == 2
    assert sleeps == [7.0]


def test_rate_limit_waits_until_the_reset_time(alpaca_client, alpaca_mock, sleeps, make_bar):
    reset_at = alpaca_mock.CLOCK + 30
    alpaca_mock.add_error(
        429,
        "too many requests",
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(reset_at))},
    )
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    fetch(alpaca_client)

    assert sleeps == [31.0]  # 30 seconds until reset plus a one-second cushion


def test_rate_limit_without_headers_falls_back_to_exponential_backoff(
    alpaca_client, alpaca_mock, sleeps, make_bar
):
    alpaca_mock.add_error(429)
    alpaca_mock.add_error(429)
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    fetch(alpaca_client)

    assert sleeps == [1.0, 2.0]


def test_rate_limit_wait_is_capped(alpaca_client, alpaca_mock, sleeps, make_bar):
    alpaca_mock.add_error(429, headers={"Retry-After": "86400"})
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    fetch(alpaca_client)

    assert sleeps == [120.0]


def test_persistent_rate_limit_raises_after_the_configured_retries(
    alpaca_client, alpaca_mock, sleeps, settings
):
    attempts = settings.http_max_retries + 1
    for _ in range(attempts):
        alpaca_mock.add_error(429, "too many requests", headers={"Retry-After": "1"})

    with pytest.raises(AlpacaRateLimitError, match="rate limit"):
        fetch(alpaca_client)

    assert len(alpaca_mock.requests) == attempts
    assert sleeps == [1.0] * settings.http_max_retries


def test_exhausted_quota_pauses_before_requesting_the_next_page(
    alpaca_client, alpaca_mock, sleeps, make_bar
):
    reset_at = alpaca_mock.CLOCK + 10
    alpaca_mock.add_bars(
        {"TESTA": [make_bar(date(2024, 1, 2))]},
        next_page_token="tok-1",
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(reset_at))},
    )
    alpaca_mock.add_bars(
        {"TESTA": [make_bar(date(2024, 1, 3))]},
        headers={"X-RateLimit-Remaining": "150", "X-RateLimit-Reset": str(int(reset_at))},
    )

    bars = fetch(alpaca_client)

    assert len(bars["TESTA"]) == 2
    assert sleeps == [11.0]


# --- Retry with exponential backoff -------------------------------------------------


def test_server_errors_are_retried_with_exponential_backoff(
    alpaca_client, alpaca_mock, sleeps, make_bar
):
    for status_code in (500, 503, 502):
        alpaca_mock.add_error(status_code, "fixture outage")
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    bars = fetch(alpaca_client)

    assert len(bars["TESTA"]) == 1
    assert len(alpaca_mock.requests) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_backoff_is_capped_at_the_configured_maximum(settings, alpaca_mock, sleeps, make_bar):
    patient = settings.model_copy(update={"http_max_retries": 6})
    for _ in range(6):
        alpaca_mock.add_error(504, "fixture outage")
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    with AlpacaClient(
        patient, transport=alpaca_mock.transport, sleep=sleeps.append, backoff_jitter=0.0
    ) as client:
        fetch(client)

    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_backoff_jitter_only_ever_lengthens_the_wait(settings, alpaca_mock, sleeps, make_bar):
    alpaca_mock.add_error(500)
    alpaca_mock.add_error(500)
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    with AlpacaClient(
        settings, transport=alpaca_mock.transport, sleep=sleeps.append, backoff_jitter=0.5
    ) as client:
        fetch(client)

    assert 1.0 <= sleeps[0] <= 1.5
    assert 2.0 <= sleeps[1] <= 3.0


def test_persistent_server_error_raises_after_the_configured_retries(
    alpaca_client, alpaca_mock, sleeps, settings
):
    attempts = settings.http_max_retries + 1
    for _ in range(attempts):
        alpaca_mock.add_error(503, "fixture outage")

    with pytest.raises(AlpacaServerError, match="HTTP 503: fixture outage") as excinfo:
        fetch(alpaca_client)

    assert excinfo.value.status_code == 503
    assert len(alpaca_mock.requests) == attempts
    assert len(sleeps) == settings.http_max_retries


def test_timeouts_are_retried_and_then_succeed(alpaca_client, alpaca_mock, sleeps, make_bar):
    alpaca_mock.add_exception(httpx.ReadTimeout("fixture timeout"))
    alpaca_mock.add_exception(httpx.ConnectTimeout("fixture timeout"))
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    bars = fetch(alpaca_client)

    assert len(bars["TESTA"]) == 1
    assert sleeps == [1.0, 2.0]


def test_persistent_timeout_raises_network_error(alpaca_client, alpaca_mock, settings):
    attempts = settings.http_max_retries + 1
    for _ in range(attempts):
        alpaca_mock.add_exception(httpx.ReadTimeout("fixture timeout"))

    with pytest.raises(AlpacaNetworkError, match="timed out after 30 seconds"):
        fetch(alpaca_client)

    assert len(alpaca_mock.requests) == attempts


def test_connection_errors_are_retried(alpaca_client, alpaca_mock, sleeps, make_bar):
    alpaca_mock.add_exception(httpx.ConnectError("fixture connection refused"))
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    fetch(alpaca_client)

    assert len(alpaca_mock.requests) == 2
    assert sleeps == [1.0]


def test_persistent_connection_error_raises_network_error(alpaca_client, alpaca_mock, settings):
    for _ in range(settings.http_max_retries + 1):
        alpaca_mock.add_exception(httpx.ConnectError("fixture connection refused"))

    with pytest.raises(AlpacaNetworkError, match="fixture connection refused"):
        fetch(alpaca_client)


def test_proxy_refusal_fails_immediately_with_a_clear_message(alpaca_client, alpaca_mock, sleeps):
    alpaca_mock.add_exception(httpx.ProxyError("403 Forbidden"))

    with pytest.raises(AlpacaNetworkError, match="proxy refused the connection"):
        fetch(alpaca_client)

    assert len(alpaca_mock.requests) == 1
    assert sleeps == []


def test_retry_resumes_on_the_same_page(alpaca_client, alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]}, next_page_token="tok-1")
    alpaca_mock.add_error(500, "fixture outage")
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 3))]})

    bars = fetch(alpaca_client)

    assert len(bars["TESTA"]) == 2
    assert [alpaca_mock.query(index).get("page_token") for index in range(3)] == [
        None,
        "tok-1",
        "tok-1",
    ]


# --- Request logging and secrets -------------------------------------------------------


def test_requests_are_logged_without_credentials(
    alpaca_client, alpaca_mock, settings, caplog, make_bar
):
    caplog.set_level(logging.DEBUG)
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]}, next_page_token="tok-1")
    alpaca_mock.add_error(500, "fixture outage")
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 3))]})
    alpaca_mock.add_error(401, "fixture unauthorized")

    fetch(alpaca_client)
    with pytest.raises(AlpacaAuthenticationError) as excinfo:
        fetch(alpaca_client)

    log_text = caplog.text
    assert "GET /v2/stocks/bars" in log_text
    assert "feed=iex" in log_text
    assert "adjustment=split" in log_text
    assert "status=200" in log_text
    assert "status=500" in log_text
    assert "retrying in" in log_text

    key_id, secret = settings.require_alpaca_credentials()
    for hidden in (key_id, secret):
        assert hidden not in log_text
        assert hidden not in str(excinfo.value)


# --- Credentials and configuration ----------------------------------------------------


def test_client_refuses_to_start_without_credentials(settings):
    without_credentials = settings.model_copy(
        update={"alpaca_api_key_id": None, "alpaca_api_secret_key": None}
    )

    with pytest.raises(ConfigurationError, match="APCA_API_KEY_ID and APCA_API_SECRET_KEY"):
        AlpacaClient(without_credentials)


def test_proxy_supplied_secret_sends_only_the_key_id_header(settings, alpaca_mock, make_bar):
    proxied = settings.model_copy(
        update={"alpaca_api_secret_key": None, "alpaca_secret_from_proxy": True}
    )
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    assert proxied.has_alpaca_credentials
    with AlpacaClient(proxied, transport=alpaca_mock.transport) as client:
        fetch(client)

    request = alpaca_mock.requests[0]
    assert request.headers["APCA-API-KEY-ID"] == settings.require_alpaca_credentials()[0]
    assert "APCA-API-SECRET-KEY" not in request.headers


def test_proxy_supplied_secret_still_needs_the_key_id(settings):
    no_key_id = settings.model_copy(
        update={
            "alpaca_api_key_id": None,
            "alpaca_api_secret_key": None,
            "alpaca_secret_from_proxy": True,
        }
    )

    assert not no_key_id.has_alpaca_credentials
    with pytest.raises(ConfigurationError, match="not configured"):
        AlpacaClient(no_key_id)


def test_a_configured_secret_is_sent_even_when_the_proxy_option_is_on(
    settings, alpaca_mock, make_bar
):
    both = settings.model_copy(update={"alpaca_secret_from_proxy": True})
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})

    with AlpacaClient(both, transport=alpaca_mock.transport) as client:
        fetch(client)

    assert alpaca_mock.requests[0].headers["APCA-API-SECRET-KEY"] == (
        settings.require_alpaca_credentials()[1]
    )


def test_proxy_supplied_secret_401_points_at_the_proxy_not_the_env_file(settings, alpaca_mock):
    proxied = settings.model_copy(
        update={"alpaca_api_secret_key": None, "alpaca_secret_from_proxy": True}
    )
    alpaca_mock.add_error(401)

    with (
        AlpacaClient(proxied, transport=alpaca_mock.transport) as client,
        pytest.raises(AlpacaAuthenticationError, match="SIGNALIQ_ALPACA_SECRET_FROM_PROXY"),
    ):
        fetch(client)


def test_load_settings_reads_the_proxy_supplied_secret_option():
    loaded = load_settings(
        env_file=None,
        environ={"APCA_API_KEY_ID": "fixture-key-only", "SIGNALIQ_ALPACA_SECRET_FROM_PROXY": "true"},
    )

    assert loaded.alpaca_secret_from_proxy
    assert loaded.has_alpaca_credentials
    assert loaded.alpaca_auth_headers() == {"APCA-API-KEY-ID": "fixture-key-only"}

    with pytest.raises(ConfigurationError, match="SIGNALIQ_ALPACA_SECRET_FROM_PROXY"):
        load_settings(env_file=None, environ={"SIGNALIQ_ALPACA_SECRET_FROM_PROXY": "maybe"})


def test_settings_never_reveal_secrets_when_printed(settings):
    key_id, secret = settings.require_alpaca_credentials()

    for rendered in (repr(settings), str(settings), settings.model_dump_json()):
        assert key_id not in rendered
        assert secret not in rendered


def test_default_universe_is_the_ten_milestone_symbols():
    assert Settings().symbols == DEFAULT_SYMBOLS
    assert DEFAULT_SYMBOLS == (
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "AMZN",
        "META",
        "TSLA",
        "GOOGL",
        "SPY",
        "QQQ",
    )


def test_default_request_configuration_matches_the_milestone_brief():
    defaults = Settings()

    assert defaults.alpaca_data_base_url == "https://data.alpaca.markets"
    assert defaults.alpaca_feed == "iex"
    assert defaults.alpaca_adjustment == "split"
    assert defaults.alpaca_currency == "USD"
    assert defaults.default_timeframe == "1Day"
    assert defaults.history_start == date(2016, 1, 1)
    assert not defaults.has_alpaca_credentials


def test_load_settings_reads_the_env_file_and_environment_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "APCA_API_KEY_ID=fixture-key-from-file\n"
        "APCA_API_SECRET_KEY=fixture-secret-from-file\n"
        "SIGNALIQ_HISTORY_START=2020-06-01\n"
        "SIGNALIQ_SYMBOLS=testa, testb\n"
        "SIGNALIQ_LOG_LEVEL=debug\n"
    )

    loaded = load_settings(
        env_file=env_file, environ={"APCA_API_KEY_ID": "fixture-key-from-environment"}
    )

    assert loaded.require_alpaca_credentials() == (
        "fixture-key-from-environment",
        "fixture-secret-from-file",
    )
    assert loaded.history_start == date(2020, 6, 1)
    assert loaded.symbols == ("TESTA", "TESTB")
    assert loaded.log_level == "DEBUG"


def test_load_settings_works_without_an_env_file(tmp_path):
    loaded = load_settings(env_file=tmp_path / "missing.env", environ={})

    assert not loaded.has_alpaca_credentials
    assert loaded.symbols == DEFAULT_SYMBOLS


@pytest.mark.parametrize(
    "environ",
    [
        {
            "APCA_API_KEY_ID": "REPLACE_WITH_YOUR_ALPACA_KEY_ID",
            "APCA_API_SECRET_KEY": "REPLACE_WITH_YOUR_ALPACA_SECRET_KEY",
        },
        {"APCA_API_KEY_ID": "", "APCA_API_SECRET_KEY": "   "},
        {"APCA_API_KEY_ID": "fixture-key-only"},
    ],
)
def test_placeholder_empty_or_partial_credentials_count_as_missing(environ):
    loaded = load_settings(env_file=None, environ=environ)

    assert not loaded.has_alpaca_credentials
    with pytest.raises(ConfigurationError, match="not configured"):
        loaded.require_alpaca_credentials()


def test_example_env_file_contains_placeholders_only():
    loaded = load_settings(env_file=PROJECT_ROOT / ".env.example", environ={})

    assert not loaded.has_alpaca_credentials


@pytest.mark.parametrize(
    ("environ", "expected_text"),
    [
        ({"SIGNALIQ_ALPACA_FEED": "nasdaq"}, "SIGNALIQ_ALPACA_FEED: Value error, unknown feed"),
        ({"SIGNALIQ_ALPACA_ADJUSTMENT": "magic"}, "SIGNALIQ_ALPACA_ADJUSTMENT"),
        ({"SIGNALIQ_ALPACA_DATA_BASE_URL": "http://data.alpaca.markets"}, "https://"),
        ({"SIGNALIQ_DATABASE_URL": "postgresql://localhost/db"}, "only SQLite is supported"),
        ({"SIGNALIQ_HISTORY_START": "yesterday"}, "SIGNALIQ_HISTORY_START"),
        ({"SIGNALIQ_SYMBOLS": "TESTA,TESTA"}, "symbols must be unique"),
        ({"SIGNALIQ_SYMBOLS": "bad symbol!"}, "invalid symbol"),
        ({"SIGNALIQ_HTTP_MAX_RETRIES": "-1"}, "SIGNALIQ_HTTP_MAX_RETRIES"),
        ({"SIGNALIQ_LOG_LEVEL": "LOUD"}, "unknown log level"),
    ],
)
def test_invalid_configuration_is_reported_clearly(environ, expected_text):
    with pytest.raises(ConfigurationError, match=expected_text):
        load_settings(env_file=None, environ=environ)


def test_relative_database_path_is_resolved_from_the_project_folder():
    loaded = load_settings(
        env_file=None, environ={"SIGNALIQ_DATABASE_URL": "sqlite:///data/custom.db"}
    )

    assert loaded.database_url == f"sqlite:///{(PROJECT_ROOT / 'data' / 'custom.db').as_posix()}"


def test_feed_names_are_normalised_but_never_changed(settings):
    assert Settings(alpaca_feed=" IEX ").alpaca_feed == "iex"
    with AlpacaClient(
        settings.model_copy(update={"alpaca_feed": "sip"}),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"bars": {}})),
    ) as client:
        assert client.feed == "sip"
        assert client.adjustment == "split"
        assert client.currency == "USD"


def test_secret_values_are_wrapped_so_they_cannot_leak_by_accident():
    configured = Settings(
        alpaca_api_key_id=SecretStr("fixture-key"),
        alpaca_api_secret_key=SecretStr("fixture-secret"),
    )

    assert "fixture-secret" not in f"{configured.alpaca_api_secret_key}"
    assert configured.require_alpaca_credentials() == ("fixture-key", "fixture-secret")
