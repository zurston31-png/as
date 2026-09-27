import datetime as dt

import pytest

from app.dashboard.analytics import compute_equity_curve, compute_portfolio_stats
from app.dashboard.charts import equity_curve_svg
from app import models

START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _trade(days_offset: int, pnl_usd: float | None, closed: bool = True) -> models.Trade:
    return models.Trade(
        symbol="COIN", side="sell", status="filled",
        pnl_usd=pnl_usd,
        closed_at=(START + dt.timedelta(days=days_offset)) if closed else None,
    )


def test_empty_trades_produces_zeroed_stats():
    stats = compute_portfolio_stats([], starting_balance=1000.0)
    assert stats.trade_count == 0
    assert stats.profit_factor is None
    assert stats.current_streak == 0


def test_open_and_unrealized_trades_are_excluded():
    trades = [_trade(0, None), models.Trade(symbol="COIN", side="buy", status="filled")]
    stats = compute_portfolio_stats(trades, 1000.0)
    assert stats.trade_count == 0


def test_win_rate_and_profit_factor():
    trades = [_trade(0, 100), _trade(1, -50), _trade(2, 20)]
    stats = compute_portfolio_stats(trades, 1000.0)
    assert stats.trade_count == 3
    assert stats.win_count == 2
    assert stats.win_rate == pytest.approx(200 / 3)
    assert stats.profit_factor == pytest.approx(120 / 50)


def test_current_streak_sign_reflects_the_most_recent_run():
    winning = [_trade(0, 10), _trade(1, -5), _trade(2, 10), _trade(3, 10)]
    assert compute_portfolio_stats(winning, 1000.0).current_streak == 2

    losing = [_trade(0, 10), _trade(1, -5), _trade(2, -5)]
    assert compute_portfolio_stats(losing, 1000.0).current_streak == -2


def test_equity_curve_is_cumulative_and_ordered():
    trades = [_trade(2, 30), _trade(0, 50), _trade(1, -20)]  # out of order on purpose
    curve = compute_equity_curve(trades, 1000.0)
    values = [v for _, v in curve]
    assert values == [1000.0, 1050.0, 1030.0, 1060.0]


def test_equity_curve_empty_with_no_closed_trades():
    assert compute_equity_curve([], 1000.0) == []


def test_max_drawdown_from_a_dip_and_recovery():
    trades = [_trade(0, 200), _trade(1, -300), _trade(2, 100)]
    stats = compute_portfolio_stats(trades, 1000.0)
    # equity: 1000 -> 1200 -> 900 -> 1000 ; peak 1200, trough 900 = 25% dd
    assert stats.max_drawdown_pct == pytest.approx(25.0)


def test_equity_curve_svg_empty_with_fewer_than_two_points():
    assert equity_curve_svg([(START, 1000.0)]) == ""


def test_equity_curve_svg_renders_a_polyline():
    points = [(START, 1000.0), (START + dt.timedelta(days=1), 1100.0), (START + dt.timedelta(days=2), 1050.0)]
    svg = equity_curve_svg(points)
    assert "<svg" in svg
    assert "<polyline" in svg
    assert "points=" in svg


def test_equity_curve_svg_color_reflects_net_direction():
    up = [(START, 1000.0), (START + dt.timedelta(days=1), 1100.0)]
    down = [(START, 1000.0), (START + dt.timedelta(days=1), 900.0)]
    assert "#3ddc97" in equity_curve_svg(up)
    assert "#ff5c7a" in equity_curve_svg(down)

