"""OPTIONAL live integration tests against the real Alpaca Market Data API.

These tests are skipped automatically unless Alpaca credentials are configured
(``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` in the environment or in the
project's ``.env`` file). When they run they make a handful of small, real
requests for one month of daily bars.

They check structure and behaviour only: authentication, the response shape,
pagination, validation, idempotent storage and the API. They assert nothing
about price levels and are not evidence of market performance.

    Run only these tests:        pytest -m integration -v
    Run everything except them:  pytest -m "not integration"
"""

from datetime import date, time

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import SOURCE_ALPACA, Settings, load_settings
from app.data.alpaca_client import (
    AlpacaAuthenticationError,
    AlpacaClient,
    AlpacaPermissionError,
)
from app.data.ingestion import IngestionService
from app.data.validation import (
    NEW_YORK,
    analyse_session_gaps,
    normalize_alpaca_bars,
    session_date,
    session_window_utc,
    validate_records,
)
from app.database import create_db_engine, create_session_factory, init_db
from app.main import create_app
from app.models import RunStatus

pytestmark = pytest.mark.integration

SYMBOL = "AAPL"
#: A fixed month in the past, so its daily candles are long since final.
FIRST_SESSION, LAST_SESSION = date(2024, 1, 2), date(2024, 1, 31)
EXPECTED_SESSIONS = 21  # trading days in January 2024
PROVIDER_BAR_KEYS = {"t", "o", "h", "l", "c", "v", "n", "vw"}


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    settings = load_settings()
    if not settings.has_alpaca_credentials:
        pytest.skip(
            "Alpaca credentials are not configured (APCA_API_KEY_ID / APCA_API_SECRET_KEY); "
            "live integration tests were not run."
        )
    return settings


@pytest.fixture
def live_client(live_settings: Settings):
    with AlpacaClient(live_settings) as client:
        yield client


def fetch_month(client: AlpacaClient, **kwargs):
    start, end = session_window_utc(FIRST_SESSION, LAST_SESSION)
    return client.fetch_bars([SYMBOL], timeframe="1Day", start=start, end=end, **kwargs)


def test_live_authentication_and_bar_structure(live_client: AlpacaClient):
    bars = fetch_month(live_client)

    # Valid credentials were accepted and data came back for the requested symbol only.
    assert set(bars) == {SYMBOL}
    items = bars[SYMBOL]
    assert items, "the provider returned no bars for the test window"
    assert all(set(bar) >= PROVIDER_BAR_KEYS for bar in items)

    # Every real bar passes every row-level check.
    records = normalize_alpaca_bars(
        bars,
        timeframe="1Day",
        source=SOURCE_ALPACA,
        feed=live_client.feed,
        adjustment=live_client.adjustment,
    )
    result = validate_records(
        records,
        allowed_symbols=(SYMBOL,),
        expected_feed=live_client.feed,
        expected_adjustment=live_client.adjustment,
    )
    assert [finding.message for finding in result.findings] == []

    # Timestamps are UTC, ordered, and mark midnight New York of a trading session.
    stamps = [record["timestamp"] for record in records]
    assert all(stamp.utcoffset().total_seconds() == 0 for stamp in stamps)
    assert stamps == sorted(stamps)
    assert all(stamp.astimezone(NEW_YORK).time() == time(0, 0) for stamp in stamps)

    # The trading calendar agrees with the provider: no missing and no unexpected sessions.
    analysis = analyse_session_gaps(
        (session_date(stamp) for stamp in stamps), FIRST_SESSION, LAST_SESSION
    )
    assert analysis.unexpected_dates == ()
    assert [(gap.kind, gap.sessions) for gap in analysis.gaps] == []
    assert len(items) == EXPECTED_SESSIONS


def test_live_pagination_returns_the_same_bars_as_one_request(live_client: AlpacaClient):
    start, end = session_window_utc(FIRST_SESSION, LAST_SESSION)

    pages = list(
        live_client.iter_bar_pages([SYMBOL], timeframe="1Day", start=start, end=end, limit=5)
    )
    paged = [bar for page in pages for bar in page.bars.get(SYMBOL, [])]
    single = fetch_month(live_client)[SYMBOL]

    assert len(pages) >= 2, "a page size of 5 must need several pages for a month of bars"
    assert all(page.next_page_token for page in pages[:-1])
    assert pages[-1].next_page_token is None
    assert all(page.bar_count <= 5 for page in pages)
    assert paged == single
    assert len({bar["t"] for bar in paged}) == len(paged)  # no bar delivered twice


def test_live_wrong_credentials_are_rejected(live_settings: Settings):
    wrong = live_settings.model_copy(
        update={
            "alpaca_api_key_id": SecretStr("INVALID-KEY-ID-FOR-TEST"),
            "alpaca_api_secret_key": SecretStr("invalid-secret-for-test"),
        }
    )

    with (
        AlpacaClient(wrong) as client,
        pytest.raises((AlpacaAuthenticationError, AlpacaPermissionError)),
    ):
        fetch_month(client)


def test_live_ingestion_is_idempotent_and_served_by_the_api(live_settings: Settings, tmp_path):
    settings = live_settings.model_copy(
        update={
            "symbols": (SYMBOL,),
            "database_url": f"sqlite:///{(tmp_path / 'live_integration.db').as_posix()}",
        }
    )
    engine = create_db_engine(settings.database_url)
    try:
        init_db(engine)
        with AlpacaClient(settings) as client:
            service = IngestionService(settings, create_session_factory(engine), client)
            first = service.ingest_symbol(SYMBOL, start=FIRST_SESSION, end=LAST_SESSION)
            second = service.ingest_symbol(SYMBOL, start=FIRST_SESSION, end=LAST_SESSION)
    finally:
        engine.dispose()

    assert first.status == RunStatus.SUCCEEDED
    assert first.bars_inserted == first.bars_fetched == EXPECTED_SESSIONS
    assert (first.bars_rejected, first.bars_conflicting, first.missing_sessions) == (0, 0, 0)

    # The second import of the same window stores nothing new and finds no differences.
    assert second.status == RunStatus.SUCCEEDED
    assert (second.bars_inserted, second.bars_unchanged) == (0, EXPECTED_SESSIONS)
    assert second.bars_conflicting == 0

    with TestClient(create_app(settings)) as api:
        body = api.get(f"/bars/{SYMBOL}").json()
        runs = api.get("/ingestion/runs").json()
    assert body["total"] == EXPECTED_SESSIONS
    assert {bar["feed"] for bar in body["bars"]} == {settings.alpaca_feed}
    assert {bar["source"] for bar in body["bars"]} == {SOURCE_ALPACA}
    assert {bar["adjustment"] for bar in body["bars"]} == {settings.alpaca_adjustment}
    assert all(bar["timestamp"].endswith("Z") for bar in body["bars"])
    assert runs["total"] == 2
