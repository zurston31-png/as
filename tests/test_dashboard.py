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


# ---------------------------------------------------------------------------
# /pipeline and /token
# ---------------------------------------------------------------------------

def test_pipeline_and_token_pages_require_auth():
    assert client.get("/pipeline").status_code == 401
    assert client.get("/api/pipeline").status_code == 401
    assert client.get("/token/SomeMint111").status_code == 401


def test_pipeline_page_renders_when_nothing_has_been_scanned():
    resp = client.get("/pipeline", auth=AUTH)
    assert resp.status_code == 200
    assert "Funnel" in resp.text


def test_pipeline_page_explains_that_a_narrow_funnel_is_the_design():
    """Without this the page reads as a fault report, and the natural
    response to a fault report is to loosen the filters - which is the one
    thing an operator must not conclude from it."""
    body = client.get("/pipeline", auth=AUTH).text
    assert "narrow funnel is the design" in body
    assert "never that a filter should be lowered" in body


@pytest.fixture()
def scanned_token():
    """A ScannedToken row so the pipeline table renders its cells.

    The empty-state test alone let a broken format string
    ("$%,.0f", which is not valid %-formatting) ship undetected: with no
    rows, the loop body never ran. Any template test that only covers the
    empty state is only covering half the template.
    """
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    db = SessionLocal()
    row = models.ScannedToken(
        token_address="PipelineRenderMint111", symbol="PIPE", chain="solana",
        discovery_source="dexscreener", first_seen_at=now, last_evaluated_at=now,
        evaluation_count=2, times_traded=1, last_stage="traded",
        last_reason="opened a position",
        liquidity_usd=185_000.0, volume_24h_usd=1_250_000.0,
    )
    db.add(row)
    db.commit()
    try:
        yield row
    finally:
        db.delete(row)
        db.commit()
        db.close()


def test_pipeline_page_renders_token_rows_with_formatted_numbers(scanned_token):
    resp = client.get("/pipeline", auth=AUTH)
    assert resp.status_code == 200
    assert "PIPE" in resp.text
    assert "$185,000" in resp.text        # thousands separator, no decimals
    assert "$1,250,000" in resp.text
    assert "/token/PipelineRenderMint111" in resp.text


def test_token_page_renders_a_full_record(scanned_token):
    resp = client.get("/token/PipelineRenderMint111", auth=AUTH)
    assert resp.status_code == 200
    assert "PIPE" in resp.text
    assert "discovered" in resp.text


def test_pipeline_page_shows_upstream_health(sample_rows):
    from app.services import api_health

    api_health.record_success("dexscreener")
    api_health.record_failure("geckoterminal", "HTTP 429")
    resp = client.get("/pipeline", auth=AUTH)
    assert resp.status_code == 200
    assert "dexscreener" in resp.text
    assert "geckoterminal" in resp.text


def test_pipeline_api_returns_json(sample_rows):
    import json

    resp = client.get("/api/pipeline", auth=AUTH)
    assert resp.status_code == 200
    payload = resp.json()
    json.dumps(payload, allow_nan=False)
    assert "stages" in payload["funnel"]
    assert isinstance(payload["health"], list)


def test_token_page_renders_for_an_unknown_address():
    """The honest answer for an address the bot has never seen is "never
    seen", not a 404 or a blank page."""
    resp = client.get("/token/CompletelyUnknownMint999", auth=AUTH)
    assert resp.status_code == 200
    assert "never seen" in resp.text


def test_token_page_shows_the_mint_as_the_identity(sample_rows):
    resp = client.get("/token/CompletelyUnknownMint999", auth=AUTH)
    assert "CompletelyUnknownMint999" in resp.text
    assert "keyed on the mint address, never the symbol" in resp.text


# ---------------------------------------------------------------------------
# research page and the upgraded pipeline funnel
# ---------------------------------------------------------------------------

def test_the_research_page_renders_with_real_data():
    """Rendering an empty page proves little - templates break on the
    branches that only fire once there is something to show."""
    from app import models, pipeline
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        for i in range(40):
            pipeline.record(
                db, stage=pipeline.TECHNICAL_SCORE, symbol=f"RS{i}",
                token_address=f"MintRS{i}", passed=i % 4 == 0,
                score=45.0 + i, detail={"reliable": True},
            )
        db.commit()

        resp = client.get("/research", auth=AUTH)
        assert resp.status_code == 200
        assert "Validation status" in resp.text
        assert "Score distribution" in resp.text
        # The histogram and survival columns only render with a sample.
        assert "≥65" in resp.text
    finally:
        for row in db.query(models.PipelineEvent).all():
            db.delete(row)
        db.commit()
        db.close()


def test_the_research_page_never_claims_validation_on_a_thin_record():
    resp = client.get("/research", auth=AUTH)
    assert resp.status_code == 200
    assert "NOT VALIDATED" in resp.text or "INSUFFICIENT" in resp.text


def test_the_pipeline_page_shows_the_stage_funnel_and_prescreen_breakdown():
    from app import models, pipeline
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        for i in range(12):
            pipeline.record(db, stage=pipeline.DISCOVERED, symbol=f"PF{i}",
                            token_address=f"MintPF{i}", passed=True)
            pipeline.record(
                db, stage=pipeline.PRESCREEN, symbol=f"PF{i}",
                token_address=f"MintPF{i}", passed=i < 2,
                reason="liquidity below scanner minimum",
                detail={"checks": [
                    {"name": "liquidity", "passed": i < 2},
                    {"name": "volume", "passed": True},
                ]},
            )
        db.commit()

        resp = client.get("/pipeline", auth=AUTH)
        assert resp.status_code == 200
        assert "Stage funnel" in resp.text
        assert "Pre-screen breakdown" in resp.text
        assert "liquidity" in resp.text
        # The bottleneck must be named, not left for the reader to spot.
        assert "bottleneck" in resp.text
    finally:
        for row in db.query(models.PipelineEvent).all():
            db.delete(row)
        db.commit()
        db.close()


def test_the_api_research_endpoint_is_json_safe():
    """An all-winners record produces an infinite profit factor, and
    Infinity is not valid JSON - this endpoint returned a 500 in exactly
    the early state it most needs to describe."""
    import json

    resp = client.get("/api/research", auth=AUTH)
    assert resp.status_code == 200
    json.dumps(resp.json(), allow_nan=False)
    assert "report" in resp.json()
    assert "forward_return_coverage" in resp.json()


def test_every_page_links_to_the_research_page():
    for path in ("/", "/pipeline", "/performance", "/journal"):
        resp = client.get(path, auth=AUTH)
        assert resp.status_code == 200
        assert '/research' in resp.text, f"{path} does not link to /research"


def test_the_research_page_requires_auth():
    assert client.get("/research").status_code == 401
    assert client.get("/api/research").status_code == 401
