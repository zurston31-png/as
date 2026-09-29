"""The audit log: completeness and tamper evidence."""

from __future__ import annotations

import json

from conftest import BASE

from tradebot.audit import AuditLog, fingerprint
from tradebot.models import AIVerdict, Decision, RiskDecision, Side, TradeSignal


def sample_signal() -> TradeSignal:
    signal = TradeSignal("TEST", "5m", Side.LONG, 100, 98, 104, 108, ts=BASE,
                         decision_ts=BASE, strategy="t")
    signal.trigger_candle = {"ts": BASE.isoformat(), "close": 100}
    signal.ai = AIVerdict(Decision.CONFIRM, 0.8, "looks fine", ["thin_volume"])
    signal.risk = RiskDecision(True, "ok", "ok", qty=5, risk_amount=500,
                               sizing={"risk_dollars": 500.0, "contracts": 5})
    signal.actionable = True
    return signal


def test_a_record_carries_the_whole_decision_chain(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl", "cfg123")
    log.signal(sample_signal(), {"price": 100, "features": {"ema_9": 99.5}})

    record = next(log.records())
    payload = record["payload"]
    assert record["event"] == "signal"
    assert record["config_fingerprint"] == "cfg123"
    assert payload["decision_ts"] == BASE.isoformat()
    assert payload["candle"]["close"] == 100
    assert payload["context"]["features"]["ema_9"] == 99.5
    assert payload["ai"]["decision"] == "confirm"
    assert payload["risk"]["sizing"]["contracts"] == 5
    assert payload["levels"] == {"entry": 100, "stop": 98, "tp1": 104, "tp2": 108}


def test_sequence_numbers_are_monotonic(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for _ in range(5):
        log.append("tick", {})
    assert [r["sequence"] for r in log.records()] == [1, 2, 3, 4, 5]


def test_the_chain_verifies(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(10):
        log.append("signal", {"i": i})
    result = log.verify()
    assert result.ok and result.records == 10


def test_an_edited_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.append("signal", {"i": i, "entry": 100 + i})

    lines = path.read_text().strip().splitlines()
    tampered = json.loads(lines[2])
    tampered["payload"]["entry"] = 999            # rewrite history
    lines[2] = json.dumps(tampered, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    result = AuditLog(path).verify()
    assert not result.ok
    assert result.broken_at == 3
    assert "modified" in result.message


def test_a_deleted_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.append("signal", {"i": i})

    lines = path.read_text().strip().splitlines()
    del lines[2]
    path.write_text("\n".join(lines) + "\n")

    result = AuditLog(path).verify()
    assert not result.ok and result.broken_at == 4


def test_the_chain_resumes_across_restarts(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLog(path)
    first.append("a", {})
    first.append("b", {})

    second = AuditLog(path)                       # a restart
    assert second.sequence == 2
    second.append("c", {})

    assert AuditLog(path).verify().ok
    assert [r["sequence"] for r in second.records()] == [1, 2, 3]


def test_a_logging_failure_does_not_raise(tmp_path):
    """A full disk must not take the trading loop down."""
    log = AuditLog(tmp_path / "nope" / "audit.jsonl")
    log.path = tmp_path                            # a directory: writing will fail
    record = log.append("signal", {"x": 1})
    assert record["sequence"] == 1                 # returned anyway


def test_fingerprint_is_stable_and_order_independent():
    a = fingerprint({"risk": {"pct": 0.5}, "market": {"symbol": "X"}})
    b = fingerprint({"market": {"symbol": "X"}, "risk": {"pct": 0.5}})
    c = fingerprint({"market": {"symbol": "Y"}, "risk": {"pct": 0.5}})
    assert a == b and a != c
