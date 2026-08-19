"""Dashboard route tests.

These exist because a Starlette signature change broke the dashboard with
a 500 while every other test still passed — the suite had no coverage of
the HTML page at all. Rendering the real template against real rows is the
point here, so keep these hitting the actual route rather than mocking it.
"""
import pytest
from fastapi.testclient import TestClient

from app import models
from app.config import settings
from app.database import SessionLocal
from app.main import app

client = TestClient(app)
AUTH = (settings.DASHBOARD_USERNAME, settings.DASHBOARD_PASSWORD)


@pytest.fixture()
def sample_rows():
    """An open position, a closed round trip, and a risk event, so the
    template renders every table rather than just the empty states."""
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    db = SessionLocal()
    created = []
    try:
        pos = models.Position(
            symbol="DASHCOIN",
            token_address="DashCoinAddress111",
            qty=1234.5,
            entry_price=0.00042,
            stop_loss=0.000357,
            take_profit=0.000546,
            status=models.PositionStatus.OPEN.value,
            opened_at=now,
        )
        buy = models.Trade(
            symbol="DASHCOIN", side="buy", status=models.TradeStatus.FILLED.value,
            mode=models.TradeMode.PAPER.value, size_usd=20.0, qty=1234.5,
            entry_price=0.00042, opened_at=now,
        )
        sell = models.Trade(
            symbol="OLDCOIN", side="sell", status=models.TradeStatus.FILLED.value,
            mode=models.TradeMode.PAPER.value, size_usd=20.0, qty=1000.0,
            exit_price=0.0005, pnl_usd=3.21, pnl_pct=0.16, closed_at=now,
        )
        event = models.RiskEvent(event_type="rug_check_rejected", details="unit test event")
        for row in (pos, buy, sell, event):
            db.add(row)
            created.append(row)
        db.commit()
        yield
    finally:
        for row in created:
            db.delete(row)
        db.commit()
        db.close()


def test_dashboard_requires_auth():
    resp = client.get("/")
    assert resp.status_code == 401


def test_dashboard_rejects_wrong_password():
    resp = client.get("/", auth=(settings.DASHBOARD_USERNAME, "definitely-not-the-password"))
    assert resp.status_code == 401


def test_dashboard_renders_when_empty():
    resp = client.get("/", auth=AUTH)
    assert resp.status_code == 200, resp.text[:500]
    assert "Memecoin Trading Bot" in resp.text


def test_dashboard_renders_positions_trades_and_events(sample_rows):
    resp = client.get("/", auth=AUTH)
    assert resp.status_code == 200, resp.text[:500]
    body = resp.text
    assert "DASHCOIN" in body          # open position row
    assert "OLDCOIN" in body           # closed trade row
    assert "unit test event" in body   # risk event row
    assert "PAPER" in body             # mode badge


def test_api_stats_returns_json():
    resp = client.get("/api/stats", auth=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "PAPER"
    assert body["halted"] is False
    assert isinstance(body["portfolio_value_usd"], (int, float))


def test_halt_and_resume_round_trip():
    from app.dashboard.routes import CSRF_TOKEN

    token = {"csrf_token": CSRF_TOKEN}
    halt = client.post("/api/halt", auth=AUTH, data=token, follow_redirects=False)
    assert halt.status_code == 303
    assert client.get("/api/stats", auth=AUTH).json()["halted"] is True

    resume = client.post("/api/resume", auth=AUTH, data=token, follow_redirects=False)
    assert resume.status_code == 303
    assert client.get("/api/stats", auth=AUTH).json()["halted"] is False


# --- security regressions -------------------------------------------------

def test_state_changing_endpoints_reject_missing_csrf_token():
    """Browsers attach cached Basic Auth to cross-origin form posts, so auth
    alone would let a visited page resume a bot the risk manager halted."""
    for path in ("/api/halt", "/api/resume"):
        resp = client.post(path, auth=AUTH, follow_redirects=False)
        assert resp.status_code == 403, f"{path} accepted a request with no CSRF token"


def test_state_changing_endpoints_reject_wrong_csrf_token():
    for path in ("/api/halt", "/api/resume"):
        resp = client.post(path, auth=AUTH, data={"csrf_token": "not-the-token"},
                           follow_redirects=False)
        assert resp.status_code == 403


def test_csrf_token_is_rendered_into_the_forms():
    from app.dashboard.routes import CSRF_TOKEN

    body = client.get("/", auth=AUTH).text
    assert body.count(f'value="{CSRF_TOKEN}"') >= 1


def test_default_dashboard_password_is_refused(monkeypatch):
    """The webhook refuses its placeholder secret; the dashboard must refuse
    its placeholder password, or a compose deployment that only rotated the
    webhook secret exposes halt/resume on admin/changeme."""
    from app.dashboard.routes import PLACEHOLDER_PASSWORD

    monkeypatch.setattr(settings, "DASHBOARD_PASSWORD", PLACEHOLDER_PASSWORD)
    resp = client.get("/", auth=(settings.DASHBOARD_USERNAME, PLACEHOLDER_PASSWORD))
    assert resp.status_code == 401
    assert "still the default" in resp.json()["detail"]


def test_empty_dashboard_password_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "DASHBOARD_PASSWORD", "")
    assert client.get("/", auth=(settings.DASHBOARD_USERNAME, "")).status_code == 401


# ---------------------------------------------------------------------------
# /performance
# ---------------------------------------------------------------------------

def test_performance_page_requires_auth():
    assert client.get("/performance").status_code == 401
    assert client.get("/api/performance").status_code == 401


def test_performance_page_renders_on_an_empty_record():
    """The empty state is the one an operator sees on day one, and it is
    where a template bug would sit unnoticed longest."""
    resp = client.get("/performance", auth=AUTH)
    assert resp.status_code == 200
    assert "Performance" in resp.text


def test_performance_page_leads_with_the_validation_verdict(sample_rows):
    resp = client.get("/performance", auth=AUTH)
    assert resp.status_code == 200
    body = resp.text
    # The status banner must appear before the P&L numbers - a page that
    # opens with the return and buries the sample size reads as a claim.
    status_at = min(
        (body.index(s) for s in ("EXPERIMENTAL", "FAILING", "VALIDATED") if s in body),
        default=-1,
    )
    assert status_at >= 0
    assert status_at < body.index("Net P&amp;L")


def test_performance_page_never_claims_real_money_readiness(sample_rows):
    body = client.get("/performance", auth=AUTH).text
    assert "not about real execution" in body or "simulated" in body


def test_performance_api_returns_the_report_as_json(sample_rows):
    resp = client.get("/api/performance", auth=AUTH)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["validation"]["status"] in {"experimental", "failing", "validated"}
    assert "costs" in payload and "breakdowns" in payload


def test_performance_page_can_be_filtered_to_one_strategy_version(sample_rows):
    resp = client.get("/api/performance?version=v-doesnotexist", auth=AUTH)
    assert resp.status_code == 200
    assert resp.json()["trade_count"] == 0
    assert resp.json()["strategy_version"] == "v-doesnotexist"
