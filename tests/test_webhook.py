import pytest
from fastapi.testclient import TestClient

import app.services.trading_service as trading_service
from app.config import settings
from app.database import SessionLocal
from app import models
from app.main import app
from app.rugcheck.filters import RugCheckReport
from app.services import price_feed

client = TestClient(app)


@pytest.fixture(autouse=True)
def _patch_network(monkeypatch):
    async def fake_price(instrument):
        return 0.001234

    async def fake_rug_check(chain, token_address):
        return RugCheckReport(passed=True, reasons=[], liquidity_usd=50000.0, dev_wallet_pct=0.05)

    monkeypatch.setattr(price_feed, "get_price_usd", fake_price)
    monkeypatch.setattr(trading_service, "run_rug_checks", fake_rug_check)
    yield


def _payload(symbol: str, signal: str, secret: str | None = None) -> dict:
    return {
        "secret": secret if secret is not None else settings.WEBHOOK_SECRET,
        "symbol": symbol,
        "token_address": f"{symbol}TokenAddress111",
        "chain": "solana",
        "signal": signal,
        "price": 0.001234,
        "rsi": 55,
        "ema9": 0.00125,
        "ema21": 0.00115,
        "volume": 100000,
        "volume_sma": 40000,
        "breakout_level": 0.0012,
    }


def test_webhook_rejects_invalid_secret():
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("BADSECRETCOIN", "buy", secret="wrong-secret"))
    assert resp.status_code == 401


def test_webhook_buy_opens_position_then_sell_closes_it():
    symbol = "WEBHOOKCOIN"

    resp = client.post(settings.WEBHOOK_PATH, json=_payload(symbol, "buy"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"

    db = SessionLocal()
    try:
        position = db.query(models.Position).filter_by(symbol=symbol, status="open").first()
        assert position is not None
        assert position.qty > 0
        assert position.entry_price > 0
        assert position.stop_loss < position.entry_price < position.take_profit

        trade = db.query(models.Trade).filter_by(symbol=symbol, side="buy").first()
        assert trade is not None
        assert trade.status == "filled"
        assert trade.mode == "paper"
        # Trade.position_id (Stage 8) must link the entry leg back to the
        # position it opened - without it there's no way to join an exit
        # leg back to its entry's signal/rug-check context in the journal.
        assert trade.position_id == position.id
    finally:
        db.close()

    resp2 = client.post(settings.WEBHOOK_PATH, json=_payload(symbol, "sell"))
    assert resp2.status_code == 200

    db = SessionLocal()
    try:
        position = db.query(models.Position).filter_by(symbol=symbol).order_by(models.Position.id.desc()).first()
        assert position.status == "closed"
        assert position.close_reason

        sell_trade = db.query(models.Trade).filter_by(symbol=symbol, side="sell").first()
        assert sell_trade is not None
        assert sell_trade.pnl_usd is not None
        assert sell_trade.position_id == position.id
        assert sell_trade.close_reason == position.close_reason
    finally:
        db.close()


def test_webhook_second_buy_ignored_while_position_open():
    symbol = "DUPBUYCOIN"
    client.post(settings.WEBHOOK_PATH, json=_payload(symbol, "buy"))
    client.post(settings.WEBHOOK_PATH, json=_payload(symbol, "buy"))

    db = SessionLocal()
    try:
        open_positions = db.query(models.Position).filter_by(symbol=symbol, status="open").all()
        assert len(open_positions) == 1
    finally:
        db.close()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["live_trading"] is False
