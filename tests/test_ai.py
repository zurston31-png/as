"""The AI layers, with the network stubbed out.

These assert the request shape we send and - more importantly - that every
failure mode of a model call degrades to "no trade" or "rules only" rather than
to an exception in the trading loop.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tradebot.ai import chat as chat_module
from tradebot.ai import confirm as confirm_module
from tradebot.ai.chat import ChatSession
from tradebot.ai.confirm import ConfirmationLayer
from tradebot.models import Decision, Series, Side, TradeSignal
from tradebot.strategy.context import ContextBuilder


def text_response(payload: dict, stop_reason: str = "end_turn") -> SimpleNamespace:
    block = SimpleNamespace(type="text", text=json.dumps(payload))
    return SimpleNamespace(content=[block], stop_reason=stop_reason)


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
def signal_and_context(config, uptrend):
    config.ai.enabled = True
    series = Series("TEST", "5m")
    series.extend(uptrend)
    ctx = ContextBuilder(config).build(series)
    sig = TradeSignal("TEST", "5m", Side.LONG, 100.0, 98.0, 104.0, 108.0, ts=ctx.candle.ts)
    return sig, ctx, series


def run_confirm(layer, signal_and_context):
    sig, ctx, series = signal_and_context
    return asyncio.run(layer.confirm(sig, ctx, series.closed_candles()))


# ------------------------------------------------------------- confirmation

def test_confirm_parses_a_verdict(config, signal_and_context, monkeypatch):
    client = FakeClient(text_response(
        {"decision": "confirm", "confidence": 0.82,
         "rationale": "clean cross above VWAP", "risk_flags": []}))
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: client)

    verdict = run_confirm(ConfirmationLayer(config), signal_and_context)
    assert verdict.decision is Decision.CONFIRM
    assert verdict.confidence == pytest.approx(0.82)
    assert verdict.source == "structured"


def test_confirm_sends_a_constrained_json_request(config, signal_and_context, monkeypatch):
    client = FakeClient(text_response(
        {"decision": "wait", "confidence": 0.3, "rationale": "x", "risk_flags": ["thin_volume"]}))
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: client)
    run_confirm(ConfirmationLayer(config), signal_and_context)

    sent = client.messages.calls[0]
    assert sent["model"] == config.ai.model
    fmt = sent["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["required"] == ["decision", "confidence", "rationale", "risk_flags"]
    assert fmt["schema"]["additionalProperties"] is False
    assert sent["output_config"]["effort"] == config.ai.effort
    # The model gets numbers, not a picture.
    body = sent["messages"][0]["content"]
    assert "indicators" in body and "recent_candles_oldest_first" in body


def test_disabled_layer_never_calls_the_api(config, signal_and_context, monkeypatch):
    config.ai.enabled = False
    called = []
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: called.append(1))
    verdict = run_confirm(ConfirmationLayer(config), signal_and_context)
    assert verdict.decision is Decision.SKIPPED and not called


def test_api_failure_becomes_unavailable_not_an_exception(config, signal_and_context, monkeypatch):
    monkeypatch.setattr(confirm_module, "get_client",
                        lambda *_: FakeClient(error=RuntimeError("connection reset")))
    verdict = run_confirm(ConfirmationLayer(config), signal_and_context)
    assert verdict.decision is Decision.UNAVAILABLE


def test_unparseable_output_becomes_unavailable(config, signal_and_context, monkeypatch):
    bad = SimpleNamespace(content=[SimpleNamespace(type="text", text="not json")],
                          stop_reason="end_turn")
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: FakeClient(bad))
    assert run_confirm(ConfirmationLayer(config), signal_and_context).decision is Decision.UNAVAILABLE


def test_a_refusal_is_handled(config, signal_and_context, monkeypatch):
    refused = text_response({"decision": "confirm", "confidence": 1.0,
                             "rationale": "", "risk_flags": []}, stop_reason="refusal")
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: FakeClient(refused))
    assert run_confirm(ConfirmationLayer(config), signal_and_context).decision is Decision.UNAVAILABLE


@pytest.mark.parametrize("decision,confidence,allowed", [
    ("confirm", 0.9, True),
    ("confirm", 0.1, False),     # below the configured threshold
    ("wait", 0.9, False),
    ("reject", 0.9, False),
])
def test_accept_policy(config, decision, confidence, allowed):
    from tradebot.models import AIVerdict
    layer = ConfirmationLayer(config)
    verdict = AIVerdict(Decision(decision), confidence, "because")
    assert layer.accepts(verdict)[0] is allowed


def test_required_flag_decides_what_unavailable_means(config):
    from tradebot.models import AIVerdict
    verdict = AIVerdict(Decision.UNAVAILABLE, 0.0, "timed out")
    config.ai.required = False
    assert ConfirmationLayer(config).accepts(verdict)[0] is True
    config.ai.required = True
    assert ConfirmationLayer(config).accepts(verdict)[0] is False


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
    session = ChatSession(config, lambda: {"risk": {"trades_today": 7}})
    collect(session, "how many trades today?")

    messages = client.calls[0]["messages"]
    assert messages[-1]["role"] == "system"          # state rides last
    assert messages[-2]["role"] == "user"            # ...right after the question
    assert '"trades_today": 7' in messages[-1]["content"]
    # The persona stays in the cached top-level prefix.
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

    session = ChatSession(config, broken)
    assert collect(session, "hi") == "fine"
    assert "bot state unavailable" in client.calls[0]["messages"][-1]["content"]


def test_chat_history_is_trimmed_and_still_starts_on_a_user_turn(config, monkeypatch):
    config.ai.chat_history_turns = 2
    client = FakeChatClient(["ok"])
    monkeypatch.setattr(chat_module, "get_client", lambda *_: client)
    session = ChatSession(config, dict)
    for i in range(6):
        collect(session, f"question {i}")
    assert len(session.history) <= 4
    assert session.history[0]["role"] == "user"


def test_chat_without_the_sdk_explains_itself(config, monkeypatch):
    monkeypatch.setattr(chat_module, "get_client", lambda *_: None)
    session = ChatSession(config, dict)
    assert "pip install anthropic" in collect(session, "hi")


def test_a_missing_api_key_is_explained_not_leaked_as_typeerror(config, monkeypatch):
    """The SDK raises a bare TypeError when no credential resolves."""
    from tradebot.ai.client import describe_error

    exc = TypeError(
        "Could not resolve authentication method. Expected one of api_key, "
        "auth_token, or credentials to be set."
    )
    assert "ANTHROPIC_API_KEY" in describe_error(exc)

    class NoKey:
        class messages:
            @staticmethod
            def stream(**_):
                raise exc

    monkeypatch.setattr(chat_module, "get_client", lambda *_: NoKey())
    session = ChatSession(config, dict)
    assert "ANTHROPIC_API_KEY" in collect(session, "hi")


def test_confirmation_reports_a_missing_key_clearly(config, signal_and_context, monkeypatch):
    exc = TypeError("Could not resolve authentication method. Expected one of api_key...")
    monkeypatch.setattr(confirm_module, "get_client", lambda *_: FakeClient(error=exc))
    verdict = run_confirm(ConfirmationLayer(config), signal_and_context)
    assert verdict.decision is Decision.UNAVAILABLE
    assert "ANTHROPIC_API_KEY" in verdict.rationale
