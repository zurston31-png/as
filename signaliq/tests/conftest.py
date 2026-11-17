"""Shared pytest fixtures for the SignalIQ test suite.

IMPORTANT - TEST FIXTURES ONLY
------------------------------
Every bar value produced in this test suite is a SYNTHETIC TEST FIXTURE. The
numbers are invented purely to exercise code paths. They are not real market
data, they do not describe any real security, and they must never be used as
evidence of market behaviour or performance. The fixture tickers (TESTA,
TESTB) are deliberately fake, and the mocked HTTP responses only imitate the
*shape* of Alpaca's documented bars payload.
"""

from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.data.alpaca_client import AlpacaClient
from app.data.ingestion import IngestionService
from app.database import create_db_engine, create_session_factory, init_db, session_scope
from app.main import create_app
from app.models import Bar

NEW_YORK = ZoneInfo("America/New_York")

#: Deliberately fake tickers so fixtures can never be mistaken for real securities.
FIXTURE_SYMBOLS = ("TESTA", "TESTB")

#: Fake credentials. They only ever reach a mocked transport.
FIXTURE_KEY_ID = "FIXTURE-KEY-ID-NOT-REAL"
FIXTURE_SECRET = "FIXTURE-SECRET-NOT-REAL"


@pytest.fixture
def fixed_now() -> datetime:
    """A fixed "current time": Friday 1 March 2024, 15:00 UTC (10:00 in New York).

    The market is open at this moment, so the last *completed* session is
    Thursday 29 February 2024.
    """
    return datetime(2024, 3, 1, 15, 0, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings for tests: fake credentials, fake tickers, throw-away database."""
    return Settings(
        alpaca_api_key_id=SecretStr(FIXTURE_KEY_ID),
        alpaca_api_secret_key=SecretStr(FIXTURE_SECRET),
        symbols=FIXTURE_SYMBOLS,
        history_start=date(2024, 1, 2),
        database_url=f"sqlite:///{(tmp_path / 'test_signaliq.db').as_posix()}",
        http_max_retries=3,
        http_backoff_base_seconds=1.0,
        http_backoff_max_seconds=8.0,
    )


@pytest.fixture
def engine(settings: Settings) -> Iterator[Engine]:
    """An initialised throw-away SQLite database."""
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(engine)


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def make_bar() -> Callable[..., dict[str, Any]]:
    """Factory for one SYNTHETIC bar in the shape of Alpaca's bars payload.

    Keys follow the provider's documented names (t, o, h, l, c, v, n, vw). The
    timestamp is midnight America/New_York of ``day`` expressed in UTC, which
    is how the provider timestamps daily bars. Prices are invented.
    """

    def _make(day: date, *, omit: tuple[str, ...] = (), **overrides: Any) -> dict[str, Any]:
        midnight_new_york = datetime.combine(day, time(0, 0), tzinfo=NEW_YORK)
        bar: dict[str, Any] = {
            "t": midnight_new_york.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": 10.0,
            "h": 11.0,
            "l": 9.0,
            "c": 10.5,
            "v": 1000,
            "n": 25,
            "vw": 10.2,
        }
        bar.update(overrides)
        for key in omit:
            bar.pop(key, None)
        return bar

    return _make


class AlpacaMock:
    """Scripted stand-in for the Alpaca bars endpoint (TEST FIXTURE).

    Queue responses with the ``add_*`` methods. Each request the client makes
    consumes the next queued item and is recorded in ``requests`` so tests can
    assert on exactly what was sent. Nothing here touches the network.
    """

    #: The fake wall-clock time used by the ``alpaca_client`` fixture.
    CLOCK = 1_700_000_000.0

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._queue: list[httpx.Response | Exception] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("AlpacaMock received a request but no response was queued")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def add_json(
        self, payload: Any, *, status_code: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self._queue.append(httpx.Response(status_code, json=payload, headers=headers))

    def add_bars(
        self,
        bars_by_symbol: dict[str, list[Any]],
        *,
        next_page_token: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Queue a successful page in the provider's documented response shape."""
        self.add_json(
            {"bars": bars_by_symbol, "next_page_token": next_page_token}, headers=headers
        )

    def add_error(
        self,
        status_code: int,
        message: str = "fixture error message",
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.add_json({"message": message}, status_code=status_code, headers=headers)

    def add_text(
        self, text: str, *, status_code: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self._queue.append(httpx.Response(status_code, text=text, headers=headers))

    def add_exception(self, exception: Exception) -> None:
        self._queue.append(exception)

    def query(self, index: int = -1) -> dict[str, str]:
        """Query parameters of a recorded request (the last one by default)."""
        return dict(self.requests[index].url.params)

    @property
    def pending(self) -> int:
        return len(self._queue)


@pytest.fixture
def alpaca_mock() -> AlpacaMock:
    return AlpacaMock()


@pytest.fixture
def sleeps() -> list[float]:
    """Collects every pause the client asks for, instead of really sleeping."""
    return []


@pytest.fixture
def alpaca_client(
    settings: Settings, alpaca_mock: AlpacaMock, sleeps: list[float]
) -> Iterator[AlpacaClient]:
    """An AlpacaClient wired to the mock transport: no network, no real waiting."""
    client = AlpacaClient(
        settings,
        transport=alpaca_mock.transport,
        sleep=sleeps.append,
        clock=lambda: AlpacaMock.CLOCK,
        backoff_jitter=0.0,
    )
    yield client
    client.close()


@pytest.fixture
def store_bars(session_factory: sessionmaker[Session]) -> Callable[..., None]:
    """Insert SYNTHETIC daily bars straight into the database, bypassing ingestion."""

    def _store(symbol: str, days: Iterable[date], *, timeframe: str = "1Day", **overrides: Any) -> None:
        with session_scope(session_factory) as session:
            for day in days:
                midnight_new_york = datetime.combine(day, time(0, 0), tzinfo=NEW_YORK)
                values: dict[str, Any] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": midnight_new_york.astimezone(UTC),
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
                session.add(Bar(**values))

    return _store


@pytest.fixture
def ingestion_service(
    settings: Settings,
    session_factory: sessionmaker[Session],
    alpaca_client: AlpacaClient,
    fixed_now: datetime,
) -> IngestionService:
    """Ingestion wired to the mocked provider, a throw-away database and a fixed clock."""
    return IngestionService(settings, session_factory, alpaca_client, now=lambda: fixed_now)


@pytest.fixture
def api_client(settings: Settings, engine: Engine) -> Iterator[TestClient]:
    """The FastAPI application running against the throw-away test database."""
    with TestClient(create_app(settings)) as client:
        yield client
