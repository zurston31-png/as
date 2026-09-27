"""The equity curve has to say what happened, not just that something did.

The chart used to be a bare polyline over (time, value) pairs. A step down
told the reader the book lost money and nothing else - not which token, not
which exit rule fired, not how long the position was held - so answering
"what happened here?" meant scrolling to the trades table and matching on a
timestamp by eye. These tests pin the data that makes each point
self-describing, and the geometry the template needs to put a marker
exactly on the line.
"""
import datetime as dt

import pytest
from fastapi.testclient import TestClient

from app import models
from app.config import settings
from app.dashboard.analytics import (
    build_equity_markers,
    compute_equity_curve,
    compute_equity_points,
)
from app.dashboard.charts import CURVE_PAD, CURVE_WIDTH, curve_positions
from app.database import SessionLocal
from app.main import app

START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

client = TestClient(app)
AUTH = (settings.DASHBOARD_USERNAME, settings.DASHBOARD_PASSWORD)


def _sell(minutes: int, pnl_usd: float, **kw) -> models.Trade:
    return models.Trade(
        symbol=kw.pop("symbol", "COIN"),
        side="sell",
        status=models.TradeStatus.FILLED.value,
        pnl_usd=pnl_usd,
        closed_at=START + dt.timedelta(minutes=minutes),
        **kw,
    )


# --- the points themselves -------------------------------------------------

def test_each_point_carries_the_trade_that_produced_it():
    trades = [_sell(10, 50.0, symbol="ONE"), _sell(20, -20.0, symbol="TWO")]
    points = compute_equity_points(trades, 1000.0)

    assert [p.equity_usd for p in points] == [1000.0, 1050.0, 1030.0]
    assert points[0].trade is None and points[0].trade_number == 0
    assert points[1].trade.symbol == "ONE"
    assert points[2].trade.symbol == "TWO"


def test_a_point_knows_the_equity_it_moved_from():
    """`equity_before` is what makes a point readable as a step rather than
    a level: "1050 -> 1030" answers the question the reader is asking, and
    subtracting P&L in the template would drift the moment a trade's P&L
    and the running total disagree."""
    points = compute_equity_points([_sell(10, 50.0), _sell(20, -20.0)], 1000.0)
    assert points[2].equity_before_usd == 1050.0
    assert points[2].equity_usd == 1030.0


def test_peak_and_drawdown_are_recorded_as_of_each_point():
    points = compute_equity_points(
        [_sell(10, 200.0), _sell(20, -300.0), _sell(30, 100.0)], 1000.0
    )
    peaks = [p.peak_usd for p in points]
    assert peaks == [1000.0, 1200.0, 1200.0, 1200.0]
    # 1200 -> 900 is the trough: 25% down from the peak at that moment.
    assert points[2].drawdown_pct == pytest.approx(25.0)
    assert points[3].drawdown_pct == pytest.approx(100 * (1200 - 1000) / 1200)


def test_the_plain_curve_is_derived_from_the_same_points():
    """compute_equity_curve keeps its old contract, but is now a view onto
    compute_equity_points rather than a second implementation - the line
    and the per-point detail cannot describe different numbers."""
    trades = [_sell(30, 30.0), _sell(10, 50.0), _sell(20, -20.0)]  # out of order
    curve = compute_equity_curve(trades, 1000.0)
    points = compute_equity_points(trades, 1000.0)
    assert curve == [(p.at, p.equity_usd) for p in points]
    assert [v for _, v in curve] == [1000.0, 1050.0, 1030.0, 1060.0]


def test_no_closed_trades_produces_no_points():
    assert compute_equity_points([], 1000.0) == []


# --- geometry --------------------------------------------------------------

def test_positions_span_the_padded_width():
    positions = curve_positions([1000.0, 1100.0, 1050.0])
    xs = [x for x, _ in positions]
    assert xs[0] == pytest.approx(CURVE_PAD / CURVE_WIDTH * 100)
    assert xs[-1] == pytest.approx((CURVE_WIDTH - CURVE_PAD) / CURVE_WIDTH * 100)
    assert xs == sorted(xs)


def test_the_highest_value_sits_nearest_the_top():
    """y is measured downward in both SVG and CSS, so the best equity must
    have the SMALLEST percentage. Getting this backwards would draw markers
    mirrored against the line they annotate."""
    (_, y_low), (_, y_high) = curve_positions([1000.0, 1100.0])
    assert y_high < y_low


def test_positions_are_empty_below_two_points():
    """Matches equity_curve_svg, which returns "" for the same input: there
    is no line, so there is nothing to attach a marker to."""
    assert curve_positions([1000.0]) == []
    assert curve_positions([]) == []


# --- markers ---------------------------------------------------------------

def _paired_book() -> list[models.Trade]:
    buy = models.Trade(
        symbol="COIN", side="buy", status=models.TradeStatus.FILLED.value,
        position_id=7, size_usd=20.0, entry_price=0.00042,
        opened_at=START, created_at=START,
    )
    sell = models.Trade(
        symbol="COIN", side="sell", status=models.TradeStatus.FILLED.value,
        position_id=7, size_usd=22.0, exit_price=0.00046,
        pnl_usd=2.0, pnl_pct=9.5, fee_usd=0.05,
        execution_cost_pct=0.004,          # a FRACTION on the row: 0.4%
        close_reason="take profit hit at $0.00046",
        closed_at=START + dt.timedelta(minutes=6),
        strategy_version="v-83c77cda", mode=models.TradeMode.PAPER.value,
        token_address="CoinAddress111",
    )
    return [buy, sell]


def _markers(trades, starting_balance=1000.0):
    return build_equity_markers(compute_equity_points(trades, starting_balance), trades)


def test_execution_cost_reaches_the_template_as_a_percent():
    """`Trade.execution_cost_pct` is a fraction despite its name (see
    docs/GLOSSARY.md). Rendering it raw would print a 0.4% cost as
    "0.00%", which reads as free execution - the exact misreading the fill
    audit exists to prevent."""
    marker = _markers(_paired_book())[-1]
    assert marker.execution_cost_pct == pytest.approx(0.4)


def test_holding_time_comes_from_the_entry_leg():
    """`opened_at` is stamped on the BUY leg; the sell carries only
    `closed_at`. Measuring both off the exit is what made the holding-time
    panel read "-h" for every trade before it was fixed."""
    marker = _markers(_paired_book())[-1]
    assert marker.holding_time == "6m"


def test_entry_and_exit_prices_are_both_resolved():
    marker = _markers(_paired_book())[-1]
    assert marker.entry_price == pytest.approx(0.00042)
    assert marker.exit_price == pytest.approx(0.00046)


def test_the_exit_reason_is_carried_verbatim():
    """Grouped into rules for the breakdowns, but shown in full here: the
    reader inspecting one point wants the price it actually fired at."""
    assert _markers(_paired_book())[-1].exit_reason == "take profit hit at $0.00046"


def test_the_opening_marker_describes_the_starting_balance():
    markers = _markers(_paired_book())
    assert markers[0].is_start is True
    assert markers[0].trade_number == 0
    assert markers[0].symbol is None
    assert markers[0].equity_usd == 1000.0


def test_every_point_gets_exactly_one_marker():
    trades = [_sell(10, 5.0), _sell(20, -5.0), _sell(30, 1.0)]
    assert len(_markers(trades)) == len(compute_equity_points(trades, 1000.0)) == 4
    assert [m.trade_number for m in _markers(trades)] == [0, 1, 2, 3]
    assert all(m.total_trades == 3 for m in _markers(trades))


def test_wins_and_losses_are_distinguished():
    markers = _markers([_sell(10, 5.0), _sell(20, -5.0)])
    assert markers[1].won is True
    assert markers[2].won is False
    assert markers[0].won is None       # the starting balance won nothing


def test_a_trade_with_no_position_id_still_produces_a_marker():
    """Rows written before position_id existed, and any leg the join
    misses, must not break the chart - they lose the entry context only."""
    marker = _markers([_sell(10, 5.0)])[-1]
    assert marker.pnl_usd == 5.0
    assert marker.holding_time is None
    assert marker.entry_price is None


def test_unrecorded_costs_stay_unrecorded():
    """A missing fee is not a zero fee. NULL has to survive to the
    template so it can print "not recorded" rather than "$0.0000"."""
    marker = _markers([_sell(10, 5.0)])[-1]
    assert marker.fee_usd is None
    assert marker.execution_cost_pct is None


# --- the rendered page -----------------------------------------------------

@pytest.fixture()
def closed_round_trip():
    db = SessionLocal()
    rows = _paired_book()
    try:
        for row in rows:
            db.add(row)
        db.commit()
        yield
    finally:
        for row in rows:
            db.delete(row)
        db.commit()
        db.close()


def test_the_dashboard_renders_a_marker_and_a_card_per_point(closed_round_trip):
    """Rendered through the real route: the markers are positioned by
    inline percentages the template computes from the chart geometry, and a
    Jinja error there would otherwise only show up as a 500 in production."""
    resp = client.get("/", auth=AUTH)
    assert resp.status_code == 200, resp.text[:500]
    body = resp.text
    assert 'class="eq-hit"' in body
    assert 'id="eq-card-0"' in body
    assert "take profit hit at $0.00046" in body   # the exit reason, in full
    assert "Held for" in body


def test_the_curve_panel_still_has_an_empty_state():
    """With no closed trades there is no line and no markers - the panel
    must say so rather than rendering an empty interactive chart."""
    db = SessionLocal()
    try:
        has_closed = (
            db.query(models.Trade)
            .filter(models.Trade.pnl_usd.isnot(None), models.Trade.closed_at.isnot(None))
            .count()
        )
    finally:
        db.close()
    if has_closed:
        pytest.skip("database already carries closed trades; empty state not reachable")
    body = client.get("/", auth=AUTH).text
    assert "Not enough closed trades yet" in body
