import pytest

from app import models
from app.config import settings
from app.risk.manager import (
    HARD_MAX_DAILY_LOSS_PCT,
    HARD_MAX_PORTFOLIO_PCT_PER_TRADE,
    RiskManager,
    halt_trading,
    resume_trading,
)


def test_position_size_respects_pct_and_cap():
    rm = RiskManager()
    original = settings.MAX_TRADE_SIZE_USD
    try:
        settings.MAX_TRADE_SIZE_USD = 50
        size = rm.position_size_usd(portfolio_value_usd=10_000)
        assert size == min(10_000 * rm.max_pct_per_trade, 50)
    finally:
        settings.MAX_TRADE_SIZE_USD = original


def test_hard_ceiling_clamps_oversized_pct_config(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PORTFOLIO_PCT_PER_TRADE", 0.9)
    rm = RiskManager()
    assert rm.max_pct_per_trade == HARD_MAX_PORTFOLIO_PCT_PER_TRADE


def test_hard_ceiling_clamps_daily_loss_config(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT_PCT", 0.9)
    rm = RiskManager()
    assert rm.daily_loss_limit_pct == HARD_MAX_DAILY_LOSS_PCT


def test_stop_loss_take_profit_bracket():
    rm = RiskManager()
    sl, tp = rm.stop_loss_take_profit(100.0)
    assert sl == pytest.approx(100 * (1 - rm.stop_loss_pct))
    assert tp == pytest.approx(100 * (1 + rm.take_profit_pct))
    assert sl < 100.0 < tp


def test_check_can_open_position_blocked_when_halted(db_session):
    rm = RiskManager()
    halt_trading(db_session, "unit test halt")
    db_session.commit()

    decision = rm.check_can_open_position(db_session)
    assert not decision.allowed
    assert "halted" in decision.reason

    resume_trading(db_session)
    db_session.commit()
    assert rm.check_can_open_position(db_session).allowed


def test_check_can_open_position_blocked_at_max_concurrent(db_session):
    rm = RiskManager()
    created = []
    for i in range(rm.max_concurrent_positions):
        pos = models.Position(
            symbol=f"MAXPOSCOIN{i}",
            token_address=f"addr{i}",
            qty=1.0,
            entry_price=1.0,
            stop_loss=0.8,
            take_profit=1.5,
            status=models.PositionStatus.OPEN.value,
        )
        db_session.add(pos)
        created.append(pos)
    db_session.commit()

    try:
        decision = rm.check_can_open_position(db_session)
        assert not decision.allowed
        assert "max concurrent positions" in decision.reason
    finally:
        # Don't leak open positions into other tests sharing this DB file.
        for pos in created:
            db_session.delete(pos)
        db_session.commit()


def test_evaluate_daily_loss_halts_past_limit(db_session):
    import datetime as dt

    rm = RiskManager()
    loss_limit = settings.PORTFOLIO_STARTING_BALANCE_USD * rm.daily_loss_limit_pct

    db_session.add(
        models.Trade(
            symbol="COIN",
            side="sell",
            status=models.TradeStatus.FILLED.value,
            pnl_usd=-(loss_limit + 1),
            closed_at=dt.datetime.now(dt.timezone.utc),
        )
    )
    db_session.commit()

    decision = rm.evaluate_daily_loss(db_session)
    assert not decision.allowed
