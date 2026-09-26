"""The dashboard has to say WHY trading stopped, not just that it did.

The reason has always been written to `bot_state` beside the halt flag and
recorded as a RiskEvent - and nothing read either back onto the page. The
header showed a bare HALTED badge, and the only place the reason appeared
was the Risk Events table at the bottom, which is capped at 20 rows and
shares them with every rejection. On a busy day the one row explaining why
the bot stopped had already scrolled off, so the honest answer to "why did
it halt?" was unreachable from the dashboard.
"""
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.risk.manager import halt_reason, halt_trading, resume_trading

client = TestClient(app)
AUTH = (settings.DASHBOARD_USERNAME, settings.DASHBOARD_PASSWORD)

REASON = "4 consecutive losing trades - strategy or market conditions may have changed"


@pytest.fixture()
def halted():
    db = SessionLocal()
    try:
        halt_trading(db, REASON)
        db.commit()
        yield
    finally:
        resume_trading(db)
        db.commit()
        db.close()


def test_no_reason_when_not_halted():
    """"" rather than a stale reason: resume clears the flag, and a page
    that showed the last halt's reason while running would read as though
    the bot were still stopped."""
    db = SessionLocal()
    try:
        assert halt_reason(db) == ""
    finally:
        db.close()


def test_the_reason_is_readable_while_halted(halted):
    db = SessionLocal()
    try:
        assert halt_reason(db) == REASON
    finally:
        db.close()


def test_the_dashboard_states_the_reason(halted):
    body = client.get("/", auth=AUTH).text
    assert "Trading is halted" in body
    assert REASON in body


def test_the_reason_says_open_positions_are_still_managed(halted):
    """A halt stops new entries only. Someone reading "halted" and
    assuming their open positions are unmanaged would close them by hand
    for no reason."""
    body = client.get("/", auth=AUTH).text
    assert "Open positions are unaffected" in body


def test_the_banner_is_absent_when_trading_normally():
    body = client.get("/", auth=AUTH).text
    assert "Trading is halted" not in body


def test_the_api_reports_the_reason_too(halted):
    """Whatever reads /api/stats - a phone check, a future alerting hook -
    should not have to scrape HTML for it."""
    payload = client.get("/api/stats", auth=AUTH).json()
    assert payload["halted"] is True
    assert payload["halt_reason"] == REASON
