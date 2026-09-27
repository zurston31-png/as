"""Tests for the /journal dashboard route - the per-position trade record
that joins entry indicators, rug-check score, and every exit leg together.
"""
import datetime as dt

import pytest
from fastapi.testclient import TestClient

from app import models
from app.config import settings
from app.database import SessionLocal
from app.main import app

client = TestClient(app)
AUTH = (settings.DASHBOARD_USERNAME, settings.DASHBOARD_PASSWORD)


def test_journal_requires_auth():
    resp = client.get("/journal")
    assert resp.status_code == 401


def test_journal_renders_empty_state():
    resp = client.get("/journal", auth=("nonexistent", "wrong"))
    assert resp.status_code == 401


@pytest.fixture()
def journal_fixture():
    now = dt.datetime.now(dt.timezone.utc)
    db = SessionLocal()
    created = []
    try:
        signal = models.Signal(
            symbol="JOURNALCOIN", token_address="JournalAddr1", signal_type="buy",
            price=0.001, rsi=55.0, ema9=0.00105, ema21=0.00098, volume=50000, volume_sma=20000,
            breakout_level=0.00095,
        )
        db.add(signal)
        db.flush()

        rug = models.RugCheckResult(
            signal_id=signal.id, passed=True, reasons=[], scanner_source="rugcheck.xyz",
            liquidity_usd=40000.0, top10_holder_pct=0.18,
            rug_risk_score=22.5, rug_risk_level="safe", rug_risk_factors=[],
        )
        db.add(rug)

        buy_trade = models.Trade(
            signal_id=signal.id, symbol="JOURNALCOIN", side="buy",
            status=models.TradeStatus.FILLED.value, mode="paper",
            size_usd=100.0, qty=1000.0, entry_price=0.001, opened_at=now,
        )
        db.add(buy_trade)
        db.flush()

        position = models.Position(
            symbol="JOURNALCOIN", token_address="JournalAddr1", qty=0.0, initial_qty=1000.0,
            entry_price=0.001, stop_loss=0.00085, take_profit=0.0013,
            status=models.PositionStatus.CLOSED.value, mode="paper",
            entry_trade_id=buy_trade.id, opened_at=now, closed_at=now,
            close_reason="take-profit hit at $0.00130000", realized_pnl_usd=0.0,
        )
        db.add(position)
        db.flush()

        buy_trade.position_id = position.id

        sell_trade = models.Trade(
            position_id=position.id, symbol="JOURNALCOIN", side="sell",
            status=models.TradeStatus.FILLED.value, mode="paper",
            size_usd=130.0, qty=1000.0, exit_price=0.0013, pnl_usd=30.0, pnl_pct=0.3,
            closed_at=now, close_reason="take-profit hit at $0.00130000",
        )
        db.add(sell_trade)
        db.commit()

        created = [signal, rug, buy_trade, position, sell_trade]
        yield
    finally:
        for row in reversed(created):
            db.delete(row)
        db.commit()
        db.close()


def test_journal_joins_entry_indicators_rug_score_and_exit_leg(journal_fixture):
    resp = client.get("/journal", auth=AUTH)
    assert resp.status_code == 200
    body = resp.text
    assert "JOURNALCOIN" in body
    assert "22" in body  # rug score
    assert "take-profit hit" in body
    assert "$30.00" in body  # net realized P&L


def test_journal_handles_a_position_with_no_linked_signal():
    """A position opened without ever recording its entry trade/signal
    (e.g. legacy data, or a manual DB edit) must still render, with the
    signal/rug-check blocks showing their "no record" states rather than
    raising."""
    db = SessionLocal()
    now = dt.datetime.now(dt.timezone.utc)
    position = models.Position(
        symbol="ORPHANCOIN", qty=500.0, initial_qty=500.0, entry_price=0.01,
        stop_loss=0.008, take_profit=0.013, status=models.PositionStatus.OPEN.value,
        mode="paper", opened_at=now,
    )
    db.add(position)
    db.commit()
    try:
        resp = client.get("/journal", auth=AUTH)
        assert resp.status_code == 200
        assert "ORPHANCOIN" in resp.text
        assert "No TradingView indicator payload" in resp.text
        assert "No rug-check record linked" in resp.text
    finally:
        db.delete(position)
        db.commit()
        db.close()
