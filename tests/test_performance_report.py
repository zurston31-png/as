"""Tests for app/analysis/report.py.

Integration-level: the report is what an operator actually reads, so what
matters is that it refuses to present a thin or pooled record as if it
were a conclusion.
"""
import datetime as dt
import json
import random

import pytest

from app import models
from app.analysis.report import build_performance_report
from app.analysis.validation import ValidationStatus
from app.database import SessionLocal

NOW = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture()
def clean_db():
    """A session with the trade tables emptied.

    The report reads every Trade in the database, so rows left behind by
    the webhook/scanner integration tests would otherwise leak into these
    assertions. Cleaned on the way in AND out.
    """
    def wipe(session):
        for model in (models.Trade, models.RiskEvent, models.Signal, models.Position):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _add_trade(db, pnl, *, version="v-test0001", fee=0.25, cost_pct=0.006,
               hours_open=5.0, reason="take-profit hit", signal=None):
    trade = models.Trade(
        symbol="REPORTCOIN", side="sell", status=models.TradeStatus.FILLED.value,
        size_usd=100.0, pnl_usd=pnl, pnl_pct=pnl,
        opened_at=NOW - dt.timedelta(hours=hours_open), closed_at=NOW,
        fee_usd=fee, execution_cost_pct=cost_pct, fill_delay_seconds=1.0,
        close_reason=reason, strategy_version=version,
        signal_id=signal.id if signal else None,
    )
    db.add(trade)
    return trade


# ---------------------------------------------------------------------------
# empty and thin records
# ---------------------------------------------------------------------------

def test_an_empty_database_produces_an_experimental_report(clean_db):
    report = build_performance_report(clean_db, monte_carlo_simulations=50)
    assert report.stats.trade_count == 0
    assert report.monte_carlo is None
    assert report.validation.status is ValidationStatus.EXPERIMENTAL


def test_a_handful_of_winning_trades_is_still_experimental(clean_db):
    """The number that must not read as success. Five wins, no losses, and
    the report has to say so plainly rather than showing a 100% win rate
    as an achievement."""
    for _ in range(5):
        _add_trade(clean_db, 20.0)
    clean_db.flush()

    report = build_performance_report(clean_db, monte_carlo_simulations=100,
                                      rng=random.Random(1))
    assert report.stats.win_rate == 100.0
    assert report.validation.status is ValidationStatus.EXPERIMENTAL
    assert report.monte_carlo.reliable is False


# ---------------------------------------------------------------------------
# costs and P&L
# ---------------------------------------------------------------------------

def test_gross_and_net_pnl_differ_by_the_execution_cost(clean_db):
    _add_trade(clean_db, 30.0, fee=0.25, cost_pct=0.006)
    _add_trade(clean_db, -10.0, fee=0.25, cost_pct=0.006)
    clean_db.flush()

    report = build_performance_report(clean_db, monte_carlo_simulations=50,
                                      rng=random.Random(1))
    assert report.net_pnl_usd == pytest.approx(20.0)
    assert report.costs.total_execution_cost_usd == pytest.approx(1.2)
    assert report.gross_pnl_usd == pytest.approx(21.2)


def test_incomplete_cost_data_is_warned_about_not_silently_summed(clean_db):
    _add_trade(clean_db, 10.0, fee=0.25, cost_pct=0.006)
    _add_trade(clean_db, 10.0, fee=None, cost_pct=None)
    clean_db.flush()

    report = build_performance_report(clean_db, monte_carlo_simulations=50,
                                      rng=random.Random(1))
    assert report.costs.cost_data_complete is False
    assert any("understated" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# strategy versions
# ---------------------------------------------------------------------------

def test_pooling_versions_produces_a_loud_warning(clean_db):
    """A win rate spanning a threshold change describes a strategy that
    never existed."""
    _add_trade(clean_db, 10.0, version="v-11111111")
    _add_trade(clean_db, -5.0, version="v-22222222")
    clean_db.flush()

    report = build_performance_report(clean_db, monte_carlo_simulations=50,
                                      rng=random.Random(1))
    assert report.version_counts == {"v-11111111": 1, "v-22222222": 1}
    assert any("pool 2 strategy versions" in w for w in report.warnings)


def test_filtering_to_one_version_drops_the_warning_and_the_other_trades(clean_db):
    _add_trade(clean_db, 10.0, version="v-11111111")
    _add_trade(clean_db, -5.0, version="v-22222222")
    clean_db.flush()

    report = build_performance_report(clean_db, strategy_version="v-11111111",
                                      monte_carlo_simulations=50, rng=random.Random(1))
    assert report.stats.trade_count == 1
    assert not any("pool" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# the pieces are wired together
# ---------------------------------------------------------------------------

def test_the_report_carries_every_section(clean_db):
    signal = models.Signal(
        symbol="REPORTCOIN", chain="solana", signal_type="buy", price=1.0,
        raw_payload={}, signal_score=82.0, market_quality_score=77.0,
    )
    clean_db.add(signal)
    clean_db.flush()
    clean_db.add(models.RugCheckResult(signal_id=signal.id, passed=True, liquidity_usd=120_000.0))
    for pnl in (25.0, -8.0, 14.0):
        _add_trade(clean_db, pnl, signal=signal)
    clean_db.add(models.RiskEvent(event_type="rug_check_rejected", details="SOMECOIN: honeypot"))
    clean_db.flush()

    report = build_performance_report(clean_db, monte_carlo_simulations=100,
                                      rng=random.Random(1))

    dimensions = {b.dimension for b in report.breakdowns}
    assert dimensions == {
        "signal score", "market quality", "entry liquidity USD", "holding time", "exit reason",
    }
    assert report.rejections.total == 1
    assert report.holding.trades_counted == 3
    assert report.extremes.largest_win_usd == 25.0
    assert report.monte_carlo is not None
    assert report.validation is not None


def test_the_report_serialises(clean_db):
    _add_trade(clean_db, 12.0)
    clean_db.flush()
    payload = build_performance_report(clean_db, monte_carlo_simulations=50,
                                       rng=random.Random(1)).as_dict()
    # allow_nan=False is the point: an all-winners record produces an
    # infinite profit factor, and Python's default json.dumps would happily
    # emit `Infinity`, which is not valid JSON. Serialising it raw made
    # /api/performance return a 500 in exactly the early, all-winners state
    # the page most needs to describe.
    json.dumps(payload, allow_nan=False)
    assert payload["validation"]["status"] == "experimental"
    assert payload["profit_factor"] is None
    assert payload["profit_factor_state"] == "no losing trades yet - undefined, not excellent"


def test_a_computed_profit_factor_is_reported_as_a_number(clean_db):
    _add_trade(clean_db, 30.0)
    _add_trade(clean_db, -10.0)
    clean_db.flush()
    payload = build_performance_report(clean_db, monte_carlo_simulations=50,
                                       rng=random.Random(1)).as_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["profit_factor"] == pytest.approx(3.0)
    assert payload["profit_factor_state"] == "computed"


def test_the_report_writes_nothing(clean_db):
    """It is read-only by design: an analytics page that mutated state
    would make the record it reports on depend on who looked at it."""
    _add_trade(clean_db, 12.0)
    clean_db.commit()
    before = clean_db.query(models.Trade).count()

    build_performance_report(clean_db, monte_carlo_simulations=50, rng=random.Random(1))

    assert clean_db.query(models.Trade).count() == before
    assert not clean_db.new and not clean_db.dirty and not clean_db.deleted
