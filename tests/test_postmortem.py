"""Tests for per-trade post-mortems.

The point of a post-mortem is to distinguish trades that a P&L column
shows as identical, so the tests are mostly about the path: a winner that
gave everything back, a loser that was deeply underwater first, and the
capture ratio that grades the exit rather than the entry.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.postmortem import PostMortem, build_postmortem, recent_postmortems
from app.database import SessionLocal


@pytest.fixture
def db():
    session = SessionLocal()
    def wipe():
        for model in (models.Trade, models.Position, models.Signal, models.RugCheckResult):
            for row in session.query(model).filter(
                getattr(model, "symbol", model.id).isnot(None)
            ).all():
                if getattr(row, "symbol", "") .startswith("PM") or model is models.RugCheckResult:
                    session.delete(row)
        session.commit()
    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


def _closed(db, *, entry=1.0, exit_price=1.1, high=None, low=None,
            reason="take profit", qty=100.0, symbol="PMCOIN"):
    now = dt.datetime.now(dt.timezone.utc)
    position = models.Position(
        symbol=symbol, token_address=f"mint-{symbol}", chain="solana",
        qty=0.0, initial_qty=qty, entry_price=entry,
        stop_loss=entry * 0.8, take_profit=entry * 1.5,
        status=models.PositionStatus.CLOSED.value,
        opened_at=now - dt.timedelta(minutes=90), closed_at=now,
        close_reason=reason,
        highest_price_since_entry=high if high is not None else max(entry, exit_price),
        lowest_price_since_entry=low if low is not None else min(entry, exit_price),
        realized_pnl_usd=(exit_price - entry) * qty,
        recent_prices=[[now.isoformat(), entry]] * 12,
    )
    db.add(position)
    db.flush()

    db.add(models.Trade(
        position_id=position.id, symbol=symbol, side="buy",
        status=models.TradeStatus.FILLED.value, size_usd=entry * qty,
        qty=qty, entry_price=entry, fee_usd=0.30, execution_cost_pct=0.011,
        created_at=now - dt.timedelta(minutes=90),
    ))
    db.add(models.Trade(
        position_id=position.id, symbol=symbol, side="sell",
        status=models.TradeStatus.FILLED.value, size_usd=exit_price * qty,
        qty=qty, exit_price=exit_price, fee_usd=0.32, execution_cost_pct=0.013,
        created_at=now,
    ))
    db.commit()
    return position


def test_a_post_mortem_reports_the_path_not_just_the_outcome(db):
    position = _closed(db, entry=1.0, exit_price=1.05, high=1.40, low=0.70)
    pm = build_postmortem(db, position)

    assert pm.return_pct == pytest.approx(5.0)
    assert pm.max_gain_pct == pytest.approx(40.0)
    assert pm.max_loss_pct == pytest.approx(-30.0)
    assert pm.hold_minutes == pytest.approx(90, abs=1)
    assert pm.exit_reason == "take profit"


def test_a_plus_five_percent_winner_that_first_went_down_thirty_is_not_a_clean_winner(db):
    """The single case that motivates the whole module. In a P&L column
    this is indistinguishable from a trade that rose steadily to +5%, and
    the two call for completely different fixes."""
    steady = build_postmortem(db, _closed(db, entry=1.0, exit_price=1.05,
                                          high=1.05, low=1.0, symbol="PMSTEADY"))
    wild = build_postmortem(db, _closed(db, entry=1.0, exit_price=1.05,
                                        high=1.40, low=0.70, symbol="PMWILD"))

    assert steady.return_pct == pytest.approx(wild.return_pct)
    assert steady.max_loss_pct == pytest.approx(0.0)
    assert wild.max_loss_pct == pytest.approx(-30.0)
    assert wild.survived_a_drawdown is True
    assert steady.survived_a_drawdown is False


def test_capture_grades_the_exit_rather_than_the_entry(db):
    """A trade that peaked at +40% and closed at +5% kept an eighth of what
    it found. That is a fact about the exit logic, and the return column
    cannot express it."""
    pm = build_postmortem(db, _closed(db, entry=1.0, exit_price=1.05, high=1.40, low=0.95))
    assert pm.capture == pytest.approx(0.125, abs=0.001)


def test_giving_back_a_winner_is_flagged(db):
    pm = build_postmortem(db, _closed(db, entry=1.0, exit_price=0.98, high=1.35, low=0.95))
    assert pm.gave_back_a_winner is True
    assert "gave back" in pm.headline()


def test_capture_is_undefined_when_the_trade_never_went_green(db):
    """Dividing by a non-positive peak would produce a number that looks
    like a ratio and means nothing."""
    pm = build_postmortem(db, _closed(db, entry=1.0, exit_price=0.80, high=1.0, low=0.75))
    assert pm.capture is None


def test_fees_are_summed_across_every_leg(db):
    """A round trip taken in pieces pays each time. Reporting only the
    entry fee would understate the cost of the exit logic that split it."""
    pm = build_postmortem(db, _closed(db))
    assert pm.fees_usd == pytest.approx(0.62)


def test_the_exit_price_is_size_weighted_across_partials(db):
    """Averaging a small scalp-out with a large final exit as equals would
    misreport the return on every position that took a partial."""
    now = dt.datetime.now(dt.timezone.utc)
    position = models.Position(
        symbol="PMPARTIAL", token_address="mint-PMPARTIAL", chain="solana",
        qty=0.0, initial_qty=100.0, entry_price=1.0, stop_loss=0.8, take_profit=1.5,
        status=models.PositionStatus.CLOSED.value,
        opened_at=now - dt.timedelta(hours=1), closed_at=now,
        close_reason="trailing stop", highest_price_since_entry=1.5,
        lowest_price_since_entry=0.95, realized_pnl_usd=20.0, recent_prices=[],
    )
    db.add(position)
    db.flush()
    for qty, price in ((20.0, 1.50), (80.0, 1.10)):
        db.add(models.Trade(
            position_id=position.id, symbol="PMPARTIAL", side="sell",
            status=models.TradeStatus.FILLED.value, size_usd=qty * price,
            qty=qty, exit_price=price, fee_usd=0.1, created_at=now,
        ))
    db.commit()

    pm = build_postmortem(db, position)
    # (20*1.50 + 80*1.10) / 100 = 1.18, not the naive (1.50+1.10)/2 = 1.30
    assert pm.exit_price == pytest.approx(1.18)
    assert pm.return_pct == pytest.approx(18.0)


def test_a_liquidity_drop_during_the_hold_is_reported(db):
    position = _closed(db, symbol="PMDRAIN")
    position.liquidity_at_entry_usd = 200_000.0
    position.lowest_liquidity_usd = 40_000.0
    db.commit()

    assert build_postmortem(db, position).liquidity_drop_pct == pytest.approx(80.0)


def test_sample_count_is_reported_so_the_path_is_not_overtrusted(db):
    """MFE/MAE come from polled prices, so they are lower bounds. The
    sample count is what tells the reader how loose those bounds are."""
    pm = build_postmortem(db, _closed(db))
    assert pm.samples == 12


def test_recent_postmortems_returns_newest_first(db):
    _closed(db, symbol="PMOLD")
    _closed(db, symbol="PMNEW")
    results = recent_postmortems(db, limit=10)
    symbols = [p.symbol for p in results if p.symbol.startswith("PM")]
    assert symbols, "no post-mortems were built"
    assert results == sorted(results, key=lambda p: p.closed_at or dt.datetime.min, reverse=True)
