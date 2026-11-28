"""Raw provider response storage: record, replay, integrity, and that replay never uses the network.

Every response here is a mocked TEST FIXTURE in the shape of Alpaca's bars payload. Nothing
touches the network and no value is real market data.
"""

import hashlib
import sqlite3
from datetime import UTC, date, datetime

import httpx
import pytest

from app.data import response_store
from app.data.alpaca_client import AlpacaClient
from app.data.response_store import (
    ResponseNotStored,
    ResponseStore,
    ResponseStoreError,
    request_key,
)

START = datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
END = datetime(2024, 1, 13, 4, 59, 59, tzinfo=UTC)


def _client(settings, store, transport=None, sleeps=None):
    return AlpacaClient(
        settings,
        transport=transport,
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
        clock=lambda: 1_700_000_000.0,
        backoff_jitter=0.0,
        store=store,
    )


def _fetch(client, symbols=("TESTA",)):
    return client.fetch_bars(list(symbols), timeframe="1Day", start=START, end=END)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def two_pages(alpaca_mock, make_bar):
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]}, next_page_token="tok-1")
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 3), c=101.25)]})
    return alpaca_mock


def test_replay_reproduces_the_recorded_data_without_the_network(settings, tmp_path, two_pages):
    path = tmp_path / "responses.db"
    recorder = ResponseStore(path, "record")
    with _client(settings, recorder, two_pages.transport) as client:
        recorded = _fetch(client)
    assert recorder.fetched_live == 2
    assert len(two_pages.requests) == 2
    before = recorder.summary()
    recorder.close()
    file_hash = _sha(path)

    player = ResponseStore(path, "replay")
    with _client(settings, player) as client:  # no transport given: NoNetworkTransport
        replayed = _fetch(client)
    assert replayed == recorded
    assert player.served_from_store == 2
    assert player.fetched_live == 0
    assert len(two_pages.requests) == 2  # nothing more reached the provider
    assert player.summary() == before
    player.close()
    assert _sha(path) == file_hash  # replay opened the store read-only


def test_replay_client_has_no_credentials_and_refuses_the_network(settings, tmp_path):
    path = tmp_path / "responses.db"
    ResponseStore(path, "record").close()
    player = ResponseStore(path, "replay")
    client = _client(settings, player)
    assert "APCA-API-KEY-ID" not in client._http.headers
    assert "APCA-API-SECRET-KEY" not in client._http.headers
    with pytest.raises(ResponseStoreError, match="network request was attempted"):
        client._http.get("/v2/stocks/bars")
    client.close()


def test_replay_of_a_request_not_stored_stops(settings, tmp_path, two_pages):
    path = tmp_path / "responses.db"
    recorder = ResponseStore(path, "record")
    with _client(settings, recorder, two_pages.transport) as client:
        _fetch(client)
    recorder.close()
    with (
        _client(settings, ResponseStore(path, "replay")) as client,
        pytest.raises(ResponseNotStored),
    ):
        _fetch(client, symbols=("TESTB",))


def test_a_later_provider_revision_does_not_change_the_replay(
    settings, tmp_path, alpaca_mock, make_bar
):
    """The NEE case: the provider revises history after the run was recorded."""
    path = tmp_path / "responses.db"
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2), c=25.89)]})
    recorder = ResponseStore(path, "record")
    with _client(settings, recorder, alpaca_mock.transport) as client:
        original = _fetch(client)
    recorder.close()

    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2), c=103.57)]})  # revised
    with _client(settings, None, alpaca_mock.transport) as client:
        assert _fetch(client)["TESTA"][0]["c"] == 103.57  # live: the revised value

    with _client(settings, ResponseStore(path, "replay")) as client:
        assert _fetch(client) == original  # replay: the value as originally received
        assert original["TESTA"][0]["c"] == 25.89


def test_record_resumes_from_what_is_stored(settings, tmp_path, two_pages):
    path = tmp_path / "responses.db"
    first = ResponseStore(path, "record")
    with _client(settings, first, two_pages.transport) as client:
        expected = _fetch(client)
    first.close()
    again = ResponseStore(path, "record")
    with _client(settings, again, two_pages.transport) as client:
        assert _fetch(client) == expected
    assert again.fetched_live == 0
    assert again.served_from_store == 2
    assert two_pages.pending == 0
    assert len(two_pages.requests) == 2


def test_errors_are_retried_and_kept_out_of_responses(settings, tmp_path, alpaca_mock, make_bar):
    path = tmp_path / "responses.db"
    alpaca_mock.add_error(500)
    alpaca_mock.add_error(429, headers={"Retry-After": "1"})
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})
    store = ResponseStore(path, "record")
    with _client(settings, store, alpaca_mock.transport, sleeps=[]) as client:
        _fetch(client)
    rows = sqlite3.connect(path).execute("select status, count(*) from responses").fetchall()
    assert rows == [(200, 1)]


def test_a_damaged_body_is_detected(settings, tmp_path, two_pages):
    path = tmp_path / "responses.db"
    recorder = ResponseStore(path, "record")
    with _client(settings, recorder, two_pages.transport) as client:
        _fetch(client)
    recorder.close()
    import zlib

    db = sqlite3.connect(path)
    key, packed = db.execute("select key, body from responses order by key limit 1").fetchone()
    tampered = zlib.decompress(packed).replace(b"100", b"999")
    db.execute("update responses set body = ? where key = ?", (zlib.compress(tampered), key))
    db.commit()
    db.close()

    player = ResponseStore(path, "replay")
    assert len(player.verify()) == 1
    with _client(settings, player) as client, pytest.raises(ResponseStoreError, match="damaged"):
        _fetch(client)


def test_refuses_a_file_that_is_not_a_response_store(tmp_path):
    other = tmp_path / "research.db"
    db = sqlite3.connect(other)
    db.execute("create table membership (x)")
    db.commit()
    db.close()
    before = _sha(other)
    for mode in ("record", "replay"):
        with pytest.raises(ResponseStoreError, match="not a response store"):
            ResponseStore(other, mode)
    assert _sha(other) == before


def test_record_mode_refuses_a_protected_store(tmp_path, monkeypatch):
    from app import database

    path = tmp_path / "archived_responses.db"
    ResponseStore(path, "record").close()
    monkeypatch.setitem(database.PROTECTED_DATABASES, path.resolve(), "a test archive")
    with pytest.raises(ResponseStoreError, match="protected"):
        ResponseStore(path, "record")
    ResponseStore(path, "replay").close()


def test_replay_needs_an_existing_store(tmp_path):
    with pytest.raises(ResponseStoreError, match="existing response store"):
        ResponseStore(tmp_path / "missing.db", "replay")
    assert not (tmp_path / "missing.db").exists()


def test_key_ignores_parameter_order_and_never_holds_credentials():
    a = request_key("/v2/stocks/bars", {"symbols": "A", "limit": 10})
    b = request_key("/v2/stocks/bars", {"limit": 10, "symbols": "A"})
    assert a == b == "/v2/stocks/bars?limit=10&symbols=A"


def test_off_by_default_and_configured_by_environment(settings, tmp_path, monkeypatch):
    monkeypatch.delenv(response_store.PATH_VARIABLE, raising=False)
    monkeypatch.delenv(response_store.MODE_VARIABLE, raising=False)
    response_store.reset_shared()
    assert response_store.from_environment() is None
    client = AlpacaClient(settings, transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert client._store is None
    client.close()

    monkeypatch.setenv(response_store.PATH_VARIABLE, str(tmp_path / "r.db"))
    with pytest.raises(ResponseStoreError, match="set both"):
        response_store.from_environment()
    monkeypatch.setenv(response_store.MODE_VARIABLE, "record")
    store = response_store.from_environment()
    assert store is not None
    assert store.mode == "record"
    assert response_store.from_environment() is store  # one store per process
    response_store.reset_shared()


def _attempts(path):
    return (
        sqlite3.connect(path)
        .execute(
            "select attempt, status, error, body is not null, body_sha256 from attempts order by id"
        )
        .fetchall()
    )


def test_every_attempt_is_logged_with_failed_bodies(settings, tmp_path, alpaca_mock, make_bar):
    path = tmp_path / "responses.db"
    alpaca_mock.add_error(500, "fixture server error")
    alpaca_mock.add_error(429, headers={"Retry-After": "1"})
    alpaca_mock.add_exception(httpx.ReadTimeout("fixture timeout"))
    alpaca_mock.add_bars({"TESTA": [make_bar(date(2024, 1, 2))]})
    store = ResponseStore(path, "record")
    with _client(settings, store, alpaca_mock.transport, sleeps=[]) as client:
        recorded = _fetch(client)
    store.close()
    rows = _attempts(path)
    assert [(a, s) for a, s, *_ in rows] == [(1, 500), (2, 429), (3, None), (4, 200)]
    assert rows[2][2] == "ReadTimeout: fixture timeout"
    assert [has_body for *_, has_body, _ in rows] == [True, True, False, False]  # 200 body once
    stored_sha = sqlite3.connect(path).execute("select body_sha256 from responses").fetchone()[0]
    assert rows[3][4] == stored_sha
    db = sqlite3.connect(path)
    body = db.execute("select body from attempts where status = 500").fetchone()[0]
    assert b"fixture server error" in __import__("zlib").decompress(body)
    assert db.execute("select retry_after from attempts where status = 429").fetchone() == ("1",)
    assert ResponseStore(path, "replay").verify() == []
    with _client(settings, ResponseStore(path, "replay")) as client:
        assert _fetch(client) == recorded  # replay serves the 200 directly


@pytest.mark.parametrize(
    ("queue", "error"),
    [
        ([500] * 4, "AlpacaServerError"),
        ([429] * 4, "AlpacaRateLimitError"),
        ([422], "AlpacaRequestError"),
        (["timeout"] * 4, "AlpacaNetworkError"),
    ],
)
def test_a_recorded_failure_is_replayed_as_the_same_failure(
    settings, tmp_path, alpaca_mock, queue, error
):
    from app.data import alpaca_client as ac

    path = tmp_path / "responses.db"
    for item in queue:
        if item == "timeout":
            alpaca_mock.add_exception(httpx.ReadTimeout("fixture timeout"))
        else:
            alpaca_mock.add_error(item, headers={"Retry-After": "1"} if item == 429 else None)
    with (
        _client(settings, ResponseStore(path, "record"), alpaca_mock.transport, sleeps=[]) as c,
        pytest.raises(getattr(ac, error)),
    ):
        _fetch(c)
    assert len(_attempts(path)) == len(queue)
    sent = len(alpaca_mock.requests)
    with (
        _client(settings, ResponseStore(path, "replay")) as c,
        pytest.raises(getattr(ac, error), match=r"replay|422|Unprocessable|invalid") as raised,
    ):
        _fetch(c)
    assert len(alpaca_mock.requests) == sent  # nothing reached the provider
    assert not isinstance(raised.value, ResponseNotStored)


def test_a_store_made_before_attempts_were_logged_still_replays(settings, tmp_path, two_pages):
    path = tmp_path / "responses.db"
    with _client(settings, ResponseStore(path, "record"), two_pages.transport) as client:
        recorded = _fetch(client)
    db = sqlite3.connect(path)
    db.execute("drop table attempts")
    db.commit()
    db.close()
    player = ResponseStore(path, "replay")
    assert player.verify() == []
    with _client(settings, player) as client:
        assert _fetch(client) == recorded


def test_verify_finds_a_stored_response_without_its_logged_attempt(settings, tmp_path, two_pages):
    path = tmp_path / "responses.db"
    with _client(settings, ResponseStore(path, "record"), two_pages.transport) as client:
        _fetch(client)
    assert ResponseStore(path, "replay").verify() == []
    db = sqlite3.connect(path)
    db.execute("delete from attempts where id = (select min(id) from attempts)")
    db.commit()
    db.close()
    problems = ResponseStore(path, "replay").verify()
    assert len(problems) == 1
    assert "without its logged 200 attempt" in problems[0]
