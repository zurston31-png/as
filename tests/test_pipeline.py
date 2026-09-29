"""End-to-end: the gate chain, the AI veto, reconciliation, and the HTTP surface."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from tradebot.app.server import create_app
from tradebot.audit import NullAuditLog
from tradebot.backtest.runner import Backtester
from tradebot.data.synthetic import SyntheticSource
from tradebot.models import AIVerdict, Decision, Side
from tradebot.orchestrator import Orchestrator


def synthetic_candles(n=1200, seed=3):
    return SyntheticSource("TEST", "5m", seed=seed).history(n)


@pytest.fixture
def bot(loose_config):
    return Orchestrator(loose_config, audit=NullAuditLog())


async def feed(bot, candles):
    for candle in candles:
        await bot.on_candle(candle)


def run(bot, candles):
    asyncio.run(feed(bot, candles))
    return bot


def stub_ai(bot, monkeypatch, verdict: AIVerdict):
    async def fixed(*_args, **_kwargs):
        return verdict
    monkeypatch.setattr(bot.confirmation, "confirm", fixed)


# --------------------------------------------------------------- pipeline

def test_pipeline_runs_and_produces_state(bot):
    run(bot, synthetic_candles(400))
    state = bot.state()
    assert state["status"]["series_length"] > 300
    assert state["context"]["bars"] > 300
    assert state["risk"]["limits"]["max_trades_per_day"] > 0
    assert state["status"]["feed"]["bars_seen"] > 300
    assert state["position_state"]["state"] in ("flat", "long", "short")


def test_signals_are_produced(bot):
    run(bot, synthetic_candles(1500))
    assert bot.signals, "expected the loose strategy to find setups"


def test_one_setup_produces_one_signal(bot):
    """No bar-by-bar repeats of the same setup."""
    run(bot, synthetic_candles(1500))
    stamps = [s.decision_ts for s in bot.signals]
    assert len(stamps) == len(set(stamps))
    # And consecutive same-side signals are never on back-to-back bars.
    for a, b in zip(bot.signals, bot.signals[1:]):
        if a.side is b.side:
            assert (b.decision_ts - a.decision_ts).total_seconds() > 300


# ---------------------------------------------------------------- AI veto

def test_ai_rejection_blocks_an_otherwise_valid_setup(loose_config, monkeypatch):
    loose_config.ai.enabled = True
    bot = Orchestrator(loose_config, audit=NullAuditLog())
    stub_ai(bot, monkeypatch, AIVerdict(Decision.REJECT, 0.9, "thin volume into a high"))
    run(bot, synthetic_candles(1500))

    assert bot.signals
    assert all(not s.actionable for s in bot.signals)
    assert any("AI rejected" in s.blocked_by for s in bot.signals)


def test_low_ai_confidence_blocks_the_trade(loose_config, monkeypatch):
    loose_config.ai.enabled = True
    loose_config.ai.confirm_min_confidence = 0.8
    bot = Orchestrator(loose_config, audit=NullAuditLog())
    stub_ai(bot, monkeypatch, AIVerdict(Decision.CONFIRM, 0.4, "could go either way"))
    run(bot, synthetic_candles(1500))
    assert bot.signals and all("confidence" in s.blocked_by for s in bot.signals)


def test_an_unavailable_ai_is_a_veto_by_default(loose_config, monkeypatch):
    loose_config.ai.enabled = True
    loose_config.ai.on_failure = "wait"
    bot = Orchestrator(loose_config, audit=NullAuditLog())
    stub_ai(bot, monkeypatch, AIVerdict(Decision.UNAVAILABLE, 0.0, "timed out"))
    run(bot, synthetic_candles(1500))
    assert bot.signals and all(not s.actionable for s in bot.signals)


def test_rules_only_lets_trades_through_when_the_api_is_down(loose_config, monkeypatch):
    loose_config.ai.enabled = True
    loose_config.ai.on_failure = "rules_only"
    bot = Orchestrator(loose_config, audit=NullAuditLog())
    stub_ai(bot, monkeypatch, AIVerdict(Decision.UNAVAILABLE, 0.0, "timed out"))
    run(bot, synthetic_candles(1500))
    assert any(s.actionable for s in bot.signals)


def test_the_ai_cannot_invent_a_trade(loose_config, monkeypatch):
    """A 'confirm' on a bar the rules rejected must never reach the user."""
    loose_config.ai.enabled = True
    bot = Orchestrator(loose_config, audit=NullAuditLog())
    calls = []

    async def spy(*args, **kwargs):
        calls.append(args)
        return AIVerdict(Decision.CONFIRM, 1.0, "yes")

    monkeypatch.setattr(bot.confirmation, "confirm", spy)
    run(bot, synthetic_candles(1500))
    # The layer is only consulted about setups the rules already passed.
    assert len(calls) == len(bot.signals)


# ----------------------------------------------------------- other gates

def test_kill_switch_stops_new_signals(bot):
    bot.kill("manual: test")
    run(bot, synthetic_candles(1500))
    assert all(not s.actionable for s in bot.signals)
    assert all("kill_switch" in s.blocked_by for s in bot.signals if s.blocked_by)


def test_a_stale_feed_blocks_every_signal(loose_config):
    loose_config.market.trade_session_only = False
    loose_config.feed.max_stale_seconds = 0.001      # everything is instantly stale
    bot = Orchestrator(loose_config, audit=NullAuditLog())
    run(bot, synthetic_candles(1500))
    assert not [s for s in bot.signals if s.actionable]


def test_reconciliation_failure_halts_trading(loose_config, monkeypatch):
    bot = Orchestrator(loose_config, audit=NullAuditLog())
    from tradebot.models import TradeSignal
    # Book a position the broker has never heard of.
    bot.book.record_entry(TradeSignal("TEST", "5m", Side.LONG, 1, 1, 1, 1), 1)

    run(bot, synthetic_candles(1500))
    assert bot.risk.state.kill_switch
    assert any("reconciliation" in s.blocked_by for s in bot.signals)


def test_the_audit_log_records_every_decision(loose_config, tmp_path):
    from tradebot.audit import AuditLog

    path = tmp_path / "audit.jsonl"
    bot = Orchestrator(loose_config, audit=AuditLog(path, "fp"))
    run(bot, synthetic_candles(1500))

    records = [r for r in AuditLog(path).records() if r["event"] == "signal"]
    assert len(records) == len(bot.signals)
    assert AuditLog(path).verify().ok
    for record, signal in zip(records, bot.signals):
        assert record["payload"]["signal_id"] == signal.id
        assert record["payload"]["decision_ts"] == (signal.decision_ts or signal.ts).isoformat()
        assert record["payload"]["strategy"]["checks"]


def test_a_taken_signal_records_its_whole_chain(loose_config, tmp_path, monkeypatch):
    from tradebot.audit import AuditLog

    loose_config.ai.enabled = True
    path = tmp_path / "audit.jsonl"
    bot = Orchestrator(loose_config, audit=AuditLog(path, "fp"))
    stub_ai(bot, monkeypatch, AIVerdict(Decision.CONFIRM, 0.9, "clean"))
    run(bot, synthetic_candles(1500))

    taken = [r for r in AuditLog(path).records()
             if r["event"] == "signal" and r["payload"]["actionable"]]
    assert taken, "expected at least one taken signal"
    payload = taken[0]["payload"]
    assert payload["ai"]["decision"] == "confirm"
    assert payload["risk"]["sizing"]["contracts"] > 0
    assert payload["candle"]["close"]
    assert payload["context"]["features"]
    assert payload["position_state"]["state"] == "flat"   # as it was when decided


# -------------------------------------------------------------- backtest

def test_backtest_is_reproducible(loose_config):
    candles = synthetic_candles(1500, seed=11)
    a = asyncio.run(Backtester(loose_config, tag="rep_a").run_candles(candles))
    b = asyncio.run(Backtester(loose_config, tag="rep_b").run_candles(candles))
    assert a.metrics.to_dict() == b.metrics.to_dict()
    assert a.bars == 1500


def test_backtest_never_touches_live_risk_state(loose_config, tmp_path):
    live_state = tmp_path / "risk.json"
    live_state.write_text('{"trading_day": "2026-01-01", "equity": 12345.0, "trades_today": 99}')
    loose_config.risk.state_path = str(live_state)
    asyncio.run(Backtester(loose_config, tag="isolated").run_candles(synthetic_candles(400)))
    assert '"equity": 12345.0' in live_state.read_text()


def test_backtest_and_live_agree(loose_config):
    """The backtest drives the live orchestrator, so results must match exactly."""
    candles = synthetic_candles(1200, seed=17)
    result = asyncio.run(Backtester(loose_config, tag="agree").run_candles(candles))

    from tradebot.backtest.runner import sandbox
    bot = Orchestrator(sandbox(loose_config, "agree_live"), audit=NullAuditLog())
    bot.monitor.max_stale_seconds = 10 ** 9
    run(bot, candles)
    if bot.broker.positions:
        bot.broker.close_all(candles[-1].close, candles[-1], "end_of_data")

    assert len(bot.signals) == len(result.signals)
    assert [t.pnl for t in bot.broker.closed] == [t.pnl for t in result.trades]


# --------------------------------------------------------------- http api

@pytest.fixture
def client(loose_config):
    loose_config.feed.source = "webhook"
    loose_config.feed.webhook_secret = "s3cret"
    with TestClient(create_app(loose_config)) as c:
        yield c


def test_status_and_state_endpoints(client):
    assert client.get("/api/status").json()["symbol"] == "TEST"
    body = client.get("/api/state").json()
    assert {"status", "risk", "positions", "performance", "position_state"} <= set(body)


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "Ask the bot" in response.text


def test_kill_and_resume_over_http(client):
    assert client.post("/api/kill", json={"reason": "testing"}).json()["risk"]["kill_switch"]
    assert not client.post("/api/resume").json()["risk"]["kill_switch"]


def test_reconcile_endpoint(client):
    body = client.post("/api/reconcile").json()
    assert body["ok"] and body["reconciliation"]["ok"]


def test_audit_endpoint_reports_the_chain(client):
    body = client.get("/api/audit").json()
    assert body["verify"]["ok"]
    assert isinstance(body["records"], list)


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
