"""End-to-end tests: the gate chain, the AI veto, and the HTTP surface."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from tradebot.app.server import create_app
from tradebot.backtest.runner import Backtester
from tradebot.data.synthetic import SyntheticSource
from tradebot.models import AIVerdict, Decision
from tradebot.orchestrator import Orchestrator


def synthetic_candles(n=1200, seed=3):
    return SyntheticSource("TEST", "5m", seed=seed).history(n)


@pytest.fixture
def bot(config):
    config.market.trade_session_only = False
    return Orchestrator(config)


async def feed(bot, candles):
    for candle in candles:
        await bot.on_candle(candle)


def test_pipeline_runs_and_produces_state(bot):
    asyncio.run(feed(bot, synthetic_candles(400)))
    state = bot.state()
    assert state["status"]["series_length"] > 300
    assert state["context"]["warm"]
    assert state["risk"]["limits"]["max_trades_per_day"] > 0


def test_ai_rejection_blocks_an_otherwise_valid_setup(config, monkeypatch):
    config.market.trade_session_only = False
    config.ai.enabled = True
    bot = Orchestrator(config)

    async def always_reject(*_args, **_kwargs):
        return AIVerdict(Decision.REJECT, 0.9, "thin volume into a prior high")

    monkeypatch.setattr(bot.confirmation, "confirm", always_reject)
    asyncio.run(feed(bot, synthetic_candles(1200)))

    assert bot.signals, "expected the rules to find at least one setup"
    assert all(not s.actionable for s in bot.signals)
    assert any("AI rejected" in s.blocked_by for s in bot.signals)


def test_low_ai_confidence_blocks_the_trade(config, monkeypatch):
    config.market.trade_session_only = False
    config.ai.enabled = True
    config.ai.confirm_min_confidence = 0.8
    bot = Orchestrator(config)

    async def weak_confirm(*_args, **_kwargs):
        return AIVerdict(Decision.CONFIRM, 0.4, "could go either way")

    monkeypatch.setattr(bot.confirmation, "confirm", weak_confirm)
    asyncio.run(feed(bot, synthetic_candles(1200)))
    assert bot.signals and all("confidence" in s.blocked_by for s in bot.signals)


def test_unavailable_ai_falls_through_when_not_required(config, monkeypatch):
    config.market.trade_session_only = False
    config.ai.enabled = True
    config.ai.required = False
    bot = Orchestrator(config)

    async def unavailable(*_args, **_kwargs):
        return AIVerdict(Decision.UNAVAILABLE, 0.0, "timed out")

    monkeypatch.setattr(bot.confirmation, "confirm", unavailable)
    asyncio.run(feed(bot, synthetic_candles(1200)))
    assert any(s.actionable for s in bot.signals)


def test_unavailable_ai_blocks_when_required(config, monkeypatch):
    config.market.trade_session_only = False
    config.ai.enabled = True
    config.ai.required = True
    bot = Orchestrator(config)

    async def unavailable(*_args, **_kwargs):
        return AIVerdict(Decision.UNAVAILABLE, 0.0, "timed out")

    monkeypatch.setattr(bot.confirmation, "confirm", unavailable)
    asyncio.run(feed(bot, synthetic_candles(1200)))
    assert bot.signals and all(not s.actionable for s in bot.signals)


def test_kill_switch_stops_new_signals(bot):
    bot.kill("manual: test")
    asyncio.run(feed(bot, synthetic_candles(1200)))
    assert all(not s.actionable for s in bot.signals)
    assert all("kill switch" in s.blocked_by for s in bot.signals if s.blocked_by)


def test_the_ai_cannot_invent_a_trade(config, monkeypatch):
    """A 'confirm' on a bar the rules rejected must never reach the user."""
    config.market.trade_session_only = False
    config.ai.enabled = True
    bot = Orchestrator(config)
    calls = []

    async def spy(*args, **kwargs):
        calls.append(args)
        return AIVerdict(Decision.CONFIRM, 1.0, "yes")

    monkeypatch.setattr(bot.confirmation, "confirm", spy)
    asyncio.run(feed(bot, synthetic_candles(1200)))
    # The layer is only ever consulted about setups the rules already passed.
    assert len(calls) == len(bot.signals)


def test_backtest_is_reproducible(config):
    config.market.trade_session_only = False
    candles = synthetic_candles(1500, seed=11)
    a = asyncio.run(Backtester(config).run_candles(candles))
    b = asyncio.run(Backtester(config).run_candles(candles))
    assert a.metrics.to_dict() == b.metrics.to_dict()
    assert a.bars == 1500


def test_backtest_never_touches_live_risk_state(config, tmp_path):
    live_state = tmp_path / "risk.json"
    live_state.write_text('{"trading_day": "2026-01-01", "equity": 12345.0, "trades_today": 99}')
    config.risk.state_path = str(live_state)
    asyncio.run(Backtester(config).run_candles(synthetic_candles(300)))
    assert '"equity": 12345.0' in live_state.read_text()


# --------------------------------------------------------------- http api

@pytest.fixture
def client(config):
    config.feed.source = "webhook"
    config.feed.webhook_secret = "s3cret"
    app = create_app(config)
    with TestClient(app) as c:
        yield c


def test_status_and_state_endpoints(client):
    assert client.get("/api/status").json()["symbol"] == "TEST"
    body = client.get("/api/state").json()
    assert {"status", "risk", "positions", "performance"} <= set(body)


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "Ask the bot" in response.text


def test_kill_and_resume_over_http(client):
    assert client.post("/api/kill", json={"reason": "testing"}).json()["risk"]["kill_switch"]
    assert not client.post("/api/resume").json()["risk"]["kill_switch"]


def test_webhook_requires_the_secret(client):
    payload = {"close": 100, "closed": True}
    assert client.post("/webhook/tradingview", json=payload).status_code == 401
    assert client.post("/webhook/tradingview", json={**payload, "secret": "wrong"}).status_code == 401
    ok = client.post("/webhook/tradingview", json={**payload, "secret": "s3cret"})
    assert ok.status_code == 200 and ok.json()["accepted"]


def test_webhook_rejects_a_payload_with_no_price(client):
    response = client.post("/webhook/tradingview", json={"secret": "s3cret", "note": "buy"})
    assert response.status_code == 422


def test_websocket_sends_state_on_connect(client):
    with client.websocket_connect("/ws") as ws:
        message = ws.receive_json()
        assert message["type"] == "state"
        assert message["data"]["status"]["symbol"] == "TEST"
        ws.send_json({"action": "ping"})
        assert ws.receive_json()["type"] in ("pong", "tick", "state", "evaluation")
