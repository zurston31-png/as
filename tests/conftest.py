import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point at a throwaway sqlite file and set required env vars *before* any
# `app.*` module is imported, since Settings() is instantiated at import time.
_tmp_db = Path(tempfile.gettempdir()) / "memecoin_bot_test.db"
if _tmp_db.exists():
    _tmp_db.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db}"
os.environ.setdefault("WEBHOOK_SECRET", "test-webhook-secret-please-ignore")
# Must not be the shipped placeholder: the dashboard refuses that outright,
# the same way the webhook refuses its placeholder secret.
os.environ.setdefault("DASHBOARD_PASSWORD", "test-dashboard-password")
os.environ.setdefault("LIVE_TRADING", "false")
os.environ.setdefault("RUGCHECK_ENABLED", "true")

import pytest  # noqa: E402

from app.database import Base, SessionLocal, engine, init_db  # noqa: E402
from app.risk.manager import resume_trading  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _reset_halt_state():
    db = SessionLocal()
    try:
        resume_trading(db)
        db.commit()
    finally:
        db.close()
    yield


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# shared market-data fixtures
# ---------------------------------------------------------------------------
# Integration tests that drive the buy path need a MarketSnapshot to exist,
# because the market-quality gate (app/signals/market_quality.py) rejects a
# candidate it cannot assess. Building one healthy snapshot here - rather
# than stubbing the scorer out - means those tests keep exercising the real
# scorer, so a regression that made every real token score badly would still
# be caught by the integration suite.

def make_market_snapshot(**overrides):
    """A snapshot of a genuinely healthy, tradeable market.

    Deep pool, turnover a little above 1x, steady hour-on-hour volume, small
    average trade size, two-sided flow, an established pool and a normal
    intraday swing - the profile the quality score is meant to pass.
    Override any single field to build the unhealthy variant a test needs.
    """
    import datetime as _dt

    from app.services.price_feed import MarketSnapshot

    now = _dt.datetime.now(_dt.timezone.utc)
    defaults = dict(
        price_usd=0.005,
        liquidity_usd=250_000.0,
        volume_24h_usd=400_000.0,
        buys_24h=1_200,
        sells_24h=900,
        price_change_1h_pct=8.0,
        price_change_24h_pct=25.0,
        pair_created_at=now - _dt.timedelta(days=20),
        fdv_usd=2_000_000.0,
        token_address="HealthyMarketToken1111",
        token_symbol="HEALTHY",
        token_name="Healthy Market Token",
        volume_1h_usd=20_000.0,
        volume_6h_usd=110_000.0,
        volume_5m_usd=1_600.0,
        buys_1h=60,
        sells_1h=45,
        observed_at=now,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


@pytest.fixture()
def healthy_market(monkeypatch):
    """Patch the price feed so the market-quality gate sees a good market."""
    from app.services import price_feed

    async def fake_snapshot(token_address):
        return make_market_snapshot(token_address=token_address)

    monkeypatch.setattr(price_feed, "get_market_snapshot", fake_snapshot)
    return monkeypatch


# ---------------------------------------------------------------------------
# no real sleeping, and no real network
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_retry_sleeping(monkeypatch):
    """Make the HTTP helper's retry backoff instant for the whole suite.

    app/services/http.py retries transient failures with 1-2-4s backoff,
    which is right in production and pure wall-clock cost here. Any test
    that reaches an unreachable host pays it three times over - routing the
    price feed through the helper turned a 32s suite into a 68s one on that
    alone.

    The sleep is skipped, not the retry: attempt counting, Retry-After
    parsing and give-up behaviour all still run exactly as they do live.
    tests/test_http_retry.py additionally records the delays it would have
    slept for, and overrides this with its own stub to do so.

    Patches app.services.http._sleep, NOT asyncio.sleep. The earlier version
    set the attribute on the shared asyncio module, which reaches every
    module in the process - coroutines then never suspend, asyncio.gather
    stops interleaving, and a test written to expose a race passes because
    the race can no longer occur rather than because it was fixed.
    """
    from app.services import http as http_helper

    async def instant(_seconds):
        return None

    monkeypatch.setattr(http_helper, "_sleep", instant)
    yield


@pytest.fixture(autouse=True)
def _reset_api_health():
    """Health counters are process-global, so one test's failures would
    otherwise show up in another's assertions."""
    from app.services import api_health

    api_health.reset()
    yield
    api_health.reset()


@pytest.fixture(autouse=True)
def _coherent_cash_ledger():
    """Keep the test database's books balanced before each test.

    Many tests fabricate Trade rows directly to set up a scenario, without
    going through the buy/sell path that moves the cash ledger. That leaves
    the ledger genuinely inconsistent with the trade record - and the
    kill switch (app/safety/killswitch.py) correctly refuses to open
    positions when the books do not balance, so every downstream
    integration test would fail on an artefact of the fixtures.

    Syncing the ledger here rather than DISABLING the kill switch is the
    deliberate choice: the switch stays armed in every integration test, so
    a regression that breaks accounting or staleness in the real code path
    is still caught. Only the fixture-made discrepancy is removed.

    Accounting itself is covered directly by tests/test_kill_switch.py,
    which builds its own inconsistencies and asserts they are detected.
    """
    from app.safety import reconcile as reconcile_mod
    from app.services import portfolio

    db = SessionLocal()
    try:
        result = reconcile_mod.reconcile(db)
        if not result.balanced:
            portfolio.set_state(db, portfolio.CASH_KEY, result.expected_cash)
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
    yield
