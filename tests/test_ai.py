"""The AI layers, with the network stubbed out.

These assert the request shape we send and - more importantly - that every way
a model call can fail resolves to "no trade" rather than to an exception in the
trading loop or an optimistic reading of a broken response.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from conftest import build_context

from tradebot.ai import chat as chat_module
from tradebot.ai import confirm as confirm_module
from tradebot.ai.chat import ChatSession
from tradebot.ai.confirm import ConfirmationLayer, _schema_violation
from tradebot.models import AIVerdict, Decision, Side, TradeSignal


def text_response(payload, stop_reason: str = "end_turn") -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           stop_reason=stop_reason)


class FakeMessages:
    def __init__(self, response=None, error: Exception | None = None):
        self.response, self.error, self.calls = response, error, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.messages = FakeMessages(response, error)


@pytest.fixture
def decision(config, uptrend):
    config.ai.enabled = True
    ctx = build_context(config, uptrend)
    signal = TradeSignal("TEST", "5m", Side.LONG, 100.0, 98.0, 104.0, 108.0,
                         ts=ctx.candle.ts, decision_ts=ctx.decision_ts)
    return signal, ctx


def run_confirm(layer, decision):
    signal, ctx = decision
    return asyncio.run(layer.confirm(signal, ctx, list(ctx.candles)))


# ------------------------------------------------------------- confirmation

def test_confirm_parses_a_verdict(config, decision, monkeypatch):
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: FakeClient(text_response(
        {"decision": "confirm", "confidence": 0.82,
         "rationale": "clean cross above VWAP", "risk_flags": []})))
    verdict = run_confirm(ConfirmationLayer(config), decision)
    assert verdict.decision is Decision.CONFIRM
    assert verdict.confidence == pytest.approx(0.82)
    assert verdict.source == "structured"


def test_confirm_sends_a_constrained_json_request(config, decision, monkeypatch):
    client = FakeClient(text_response(
        {"decision": "wait", "confidence": 0.3, "rationale": "x", "risk_flags": ["thin_volume"]}))
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: client)
    run_confirm(ConfirmationLayer(config), decision)

    sent = client.messages.calls[0]
    assert sent["model"] == config.ai.model
    fmt = sent["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["required"] == ["decision", "confidence", "rationale", "risk_flags"]
    assert fmt["schema"]["additionalProperties"] is False
    assert sent["output_config"]["effort"] == config.ai.effort
    body = sent["messages"][0]["content"]
    assert "indicators" in body and "recent_candles_oldest_first" in body


def test_disabled_layer_never_calls_the_api(config, decision, monkeypatch):
    config.ai.enabled = False
    called = []
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: called.append(1))
    assert run_confirm(ConfirmationLayer(config), decision).decision is Decision.SKIPPED
    assert not called


# ------------------------------------------------- every failure is a veto

@pytest.mark.parametrize("build", [
    pytest.param(lambda: FakeClient(error=RuntimeError("connection reset")), id="api_error"),
    pytest.param(lambda: FakeClient(error=TimeoutError("timed out")), id="timeout"),
    pytest.param(lambda: FakeClient(text_response("not json")), id="unparseable"),
    pytest.param(lambda: FakeClient(text_response(
        {"decision": "confirm", "confidence": 1.0, "rationale": "", "risk_flags": []},
        stop_reason="refusal")), id="refusal"),
    pytest.param(lambda: FakeClient(text_response(
        {"decision": "maybe", "confidence": 0.9, "rationale": "", "risk_flags": []})),
        id="bad_decision_value"),
    pytest.param(lambda: FakeClient(text_response(
        {"decision": "confirm", "confidence": 7, "rationale": "", "risk_flags": []})),
        id="confidence_out_of_range"),
    pytest.param(lambda: FakeClient(text_response(
        {"decision": "confirm", "confidence": "high", "rationale": "", "risk_flags": []})),
        id="confidence_wrong_type"),
    pytest.param(lambda: FakeClient(text_response(["confirm"])), id="not_an_object"),
])
def test_every_failure_mode_becomes_unavailable(config, decision, monkeypatch, build):
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: build())
    verdict = run_confirm(ConfirmationLayer(config), decision)
    assert verdict.decision is Decision.UNAVAILABLE


def test_unavailable_means_wait_by_default(config):
    config.ai.on_failure = "wait"
    allowed, reason = ConfirmationLayer(config).accepts(
        AIVerdict(Decision.UNAVAILABLE, 0.0, "timed out"))
    assert not allowed and "WAIT" in reason


def test_rules_only_is_an_explicit_opt_out(config):
    config.ai.on_failure = "rules_only"
    allowed, reason = ConfirmationLayer(config).accepts(
        AIVerdict(Decision.UNAVAILABLE, 0.0, "timed out"))
    assert allowed and "rules-only" in reason


@pytest.mark.parametrize("decision_value,confidence,allowed", [
    ("confirm", 0.9, True),
    ("confirm", 0.1, False),     # below the configured threshold
    ("wait", 0.9, False),
    ("reject", 0.9, False),
])
def test_accept_policy(config, decision_value, confidence, allowed):
    verdict = AIVerdict(Decision(decision_value), confidence, "because")
    assert ConfirmationLayer(config).accepts(verdict)[0] is allowed


@pytest.mark.parametrize("payload,expected", [
    ({"decision": "confirm", "confidence": 0.8, "rationale": "x", "risk_flags": []}, ""),
    ({"decision": "nope", "confidence": 0.8, "rationale": "x", "risk_flags": []}, "decision"),
    ({"decision": "confirm", "confidence": 1.7, "rationale": "x", "risk_flags": []}, "outside"),
    ({"decision": "confirm", "confidence": True, "rationale": "x", "risk_flags": []}, "number"),
    ({"decision": "confirm", "confidence": 0.5, "rationale": 5, "risk_flags": []}, "rationale"),
    ({"decision": "confirm", "confidence": 0.5, "rationale": "x", "risk_flags": [1]}, "risk_flags"),
    ("nope", "object"),
])
def test_schema_validation(payload, expected):
    assert expected in _schema_violation(payload)


def test_a_missing_api_key_is_explained_not_leaked_as_typeerror(config, decision, monkeypatch):
    """The SDK raises a bare TypeError when no credential resolves."""
    exc = TypeError("Could not resolve authentication method. Expected one of api_key...")
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: FakeClient(error=exc))
    verdict = run_confirm(ConfirmationLayer(config), decision)
    assert verdict.decision is Decision.UNAVAILABLE
    assert "ANTHROPIC_API_KEY" in verdict.rationale


# --------------------------------------------------------------------- chat

class FakeStream:
    def __init__(self, chunks, stop_reason="end_turn"):
        self.chunks, self.stop_reason = chunks, stop_reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    @property
    def text_stream(self):
        async def gen():
            for chunk in self.chunks:
                yield chunk
        return gen()

    async def get_final_message(self):
        return SimpleNamespace(stop_reason=self.stop_reason)


class FakeChatClient:
    def __init__(self, chunks, stop_reason="end_turn"):
        self.calls = []
        outer = self

        class Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                return FakeStream(chunks, stop_reason)

        self.messages = Messages()


def collect(session, message):
    async def go():
        return "".join([chunk async for chunk in session.ask(message)])
    return asyncio.run(go())


def test_chat_streams_and_remembers(config, monkeypatch):
    client = FakeChatClient(["The ", "9 EMA ", "is below the 21."])
    monkeypatch.setattr(chat_module, "get_client", lambda *_: client)
    session = ChatSession(config, lambda: {"risk": {"trades_today": 2}})

    assert collect(session, "why no trade?") == "The 9 EMA is below the 21."
    assert [m["role"] for m in session.transcript()] == ["user", "assistant"]
    collect(session, "and now?")
    assert len(session.transcript()) == 4


def test_chat_attaches_live_state_after_the_user_turn(config, monkeypatch):
    client = FakeChatClient(["ok"])
    monkeypatch.setattr(chat_module, "get_client", lambda *_: client)
    collect(ChatSession(config, lambda: {"risk": {"trades_today": 7}}),
            "how many trades today?")

    messages = client.calls[0]["messages"]
    assert messages[-1]["role"] == "system"          # state rides last
    assert messages[-2]["role"] == "user"            # ...right after the question
    assert '"trades_today": 7' in messages[-1]["content"]
    assert client.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_chat_failure_does_not_poison_the_history(config, monkeypatch):
    class Boom:
        class messages:
            @staticmethod
            def stream(**_):
                raise RuntimeError("network down")

    monkeypatch.setattr(chat_module, "get_client", lambda *_: Boom())
    session = ChatSession(config, dict)
    assert "chat unavailable" in collect(session, "hello")
    assert session.transcript() == []


def test_chat_survives_a_broken_state_provider(config, monkeypatch):
    client = FakeChatClient(["fine"])
    monkeypatch.setattr(chat_module, "get_client", lambda *_: client)

    def broken():
        raise ValueError("state exploded")

    assert collect(ChatSession(config, broken), "hi") == "fine"
    assert "bot state unavailable" in client.calls[0]["messages"][-1]["content"]


def test_chat_history_is_trimmed_and_still_starts_on_a_user_turn(config, monkeypatch):
    config.ai.chat_history_turns = 2
    monkeypatch.setattr(chat_module, "get_client", lambda *_: FakeChatClient(["ok"]))
    session = ChatSession(config, dict)
    for i in range(6):
        collect(session, f"question {i}")
    assert len(session.history) <= 4
    assert session.history[0]["role"] == "user"


def test_chat_without_the_sdk_explains_itself(config, monkeypatch):
    monkeypatch.setattr(chat_module, "get_client", lambda *_: None)
    assert "pip install anthropic" in collect(ChatSession(config, dict), "hi")
