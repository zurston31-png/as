"""Integration tests for Stage 4 smart exits: trading_service.partial_close_position
and the position monitor wiring that calls into ExitManager on every tick.

Unlike tests/test_exit_manager.py (pure logic, no DB), these go through the
real paper execution client and a real DB session, the same way
tests/test_webhook.py exercises the full buy/sell path - proving the pieces
actually fit together, not just that each one is correct in isolation.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.monitor.position_monitor import _evaluate_position
from app.services import price_feed
from app.services.trading_service import partial_close_position


@pytest.fixture(autouse=True)
def _patch_price(monkeypatch):
    state = {"price": 1.0}

    async def fake_price(instrument):
        return state["price"]

    monkeypatch.setattr(price_feed, "get_price_usd", fake_price)
    yield state


def _open_position(db, symbol, **overrides):
    defaults = dict(
        symbol=symbol,
        token_address=f"{symbol}addr",
        qty=100.0,
        initial_qty=100.0,
        entry_price=1.0,
        stop_loss=0.5,
        take_profit=5.0,
        status=models.PositionStatus.OPEN.value,
        opened_at=dt.datetime.now(dt.timezone.utc),
    )
    defaults.update(overrides)
    pos = models.Position(**defaults)
    db.add(pos)
    db.commit()
    db.refresh(pos)
    return pos


def test_partial_close_position_sells_a_fraction_and_leaves_the_position_open(_patch_price):
    db = SessionLocal()
    try:
        pos = _open_position(db, "PARTIALCOIN")

        import asyncio

        asyncio.run(
            partial_close_position(db, pos, fraction=0.5, reason="test partial")
        )
        db.commit()

        assert pos.status == models.PositionStatus.OPEN.value
        assert pos.qty == pytest.approx(50.0)
        assert pos.realized_pnl_usd != 0.0  # paper slippage guarantees a nonzero fill vs entry

        trade = (
            db.query(models.Trade)
            .filter_by(symbol="PARTIALCOIN", side="sell")
            .order_by(models.Trade.id.desc())
            .first()
        )
        assert trade is not None
        assert trade.status == "filled"
        assert trade.qty == pytest.approx(50.0)
        assert trade.pnl_usd is not None
        assert trade.position_id == pos.id
        assert trade.close_reason == "test partial"
    finally:
        db.close()


def test_partial_close_position_with_fraction_one_fully_closes(_patch_price):
    db = SessionLocal()
    try:
        pos = _open_position(db, "FULLFRACCOIN")

        import asyncio

        asyncio.run(
            partial_close_position(db, pos, fraction=1.0, reason="test full via partial path")
        )
        db.commit()

        assert pos.status == models.PositionStatus.CLOSED.value
        assert pos.qty == pytest.approx(0.0)
        assert pos.close_reason == "test full via partial path"
    finally:
        db.close()


def test_position_monitor_tick_triggers_partial_exit_then_a_later_full_close(_patch_price):
    """End to end through the real monitor tick handler: price rises past
    the partial-take trigger (partial sell, position stays open), then rises
    further past take-profit (full close)."""
    original = {
        name: getattr(settings, name)
        for name in (
            "PARTIAL_TAKE_PROFIT_ENABLED", "PARTIAL_TAKE_PROFIT_TRIGGER_PCT", "PARTIAL_TAKE_PROFIT_SIZE_PCT",
            "TRAILING_STOP_ENABLED", "BREAK_EVEN_ENABLED", "MOMENTUM_EXIT_ENABLED", "TREND_REVERSAL_EXIT_ENABLED",
        )
    }
    settings.PARTIAL_TAKE_PROFIT_ENABLED = True
    settings.PARTIAL_TAKE_PROFIT_TRIGGER_PCT = 0.20
    settings.PARTIAL_TAKE_PROFIT_SIZE_PCT = 0.5
    settings.TRAILING_STOP_ENABLED = False
    settings.BREAK_EVEN_ENABLED = False
    settings.MOMENTUM_EXIT_ENABLED = False
    settings.TREND_REVERSAL_EXIT_ENABLED = False

    # position_monitor.exit_manager was constructed at import time (before
    # this test patched settings), so read the new settings into it exactly
    # like a fresh process boot would - not a workaround, just the same
    # "settings are read at construction" contract app.risk.manager.RiskManager
    # already follows.
    import app.monitor.position_monitor as position_monitor
    from app.exits.manager import ExitManager

    original_exit_manager = position_monitor.exit_manager
    position_monitor.exit_manager = ExitManager()

    try:
        db = SessionLocal()
        try:
            pos = _open_position(db, "MONITORFLOWCOIN", stop_loss=0.5, take_profit=1.5)

            import asyncio

            _patch_price["price"] = 1.25  # +25%, past the 20% partial trigger
            asyncio.run(_evaluate_position(db, pos))
            db.commit()
            assert pos.status == models.PositionStatus.OPEN.value
            assert pos.partial_exit_taken is True
            assert pos.qty == pytest.approx(50.0)

            _patch_price["price"] = 1.55  # past take_profit=1.5
            asyncio.run(_evaluate_position(db, pos))
            db.commit()
            assert pos.status == models.PositionStatus.CLOSED.value
            assert "take-profit" in pos.close_reason
        finally:
            db.close()
    finally:
        position_monitor.exit_manager = original_exit_manager
        for name, value in original.items():
            setattr(settings, name, value)
