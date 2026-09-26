import datetime as dt

import pytest

from app.backtesting.stats import compute_stats
from app.backtesting.types import BacktestTrade

START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _trade(days_offset: int, pnl_usd: float, r_multiple: float = 1.0, hold_days: int = 1) -> BacktestTrade:
    entry = START + dt.timedelta(days=days_offset)
    exit_ = entry + dt.timedelta(days=hold_days)
    return BacktestTrade(
        entry_time=entry, exit_time=exit_, entry_price=1.0, exit_price=1.0 + pnl_usd / 100,
        qty=100.0, size_usd=100.0, fees_usd=0.5, pnl_usd=pnl_usd,
        pnl_pct=pnl_usd / 100, r_multiple=r_multiple, exit_reason="test",
        signal_score=80.0, market_regime="bull_trend / normal_volatility", bars_held=5,
    )


def _equity_curve(values: list[float]) -> list[tuple[dt.datetime, float]]:
    return [(START + dt.timedelta(days=i), v) for i, v in enumerate(values)]


def test_empty_trades_produces_zeroed_stats_not_a_crash():
    stats = compute_stats([], [], starting_balance_usd=1000.0)
    assert stats.trade_count == 0
    assert stats.win_rate == 0.0
    assert stats.profit_factor is None
    assert stats.sharpe_ratio is None
    assert stats.final_balance_usd == 1000.0


def test_win_rate_and_counts():
    trades = [_trade(0, 50), _trade(1, -20), _trade(2, 30), _trade(3, -10)]
    stats = compute_stats(trades, _equity_curve([1000, 1050, 1030, 1060, 1050]), 1000.0)
    assert stats.trade_count == 4
    assert stats.win_count == 2
    assert stats.loss_count == 2
    assert stats.win_rate == pytest.approx(50.0)


def test_profit_factor_is_gross_profit_over_gross_loss():
    trades = [_trade(0, 100), _trade(1, 50), _trade(2, -40), _trade(3, -10)]
    stats = compute_stats(trades, _equity_curve([1000] * 5), 1000.0)
    assert stats.profit_factor == pytest.approx(150 / 50)


def test_profit_factor_is_none_with_no_losses_and_no_wins():
    stats = compute_stats([], [], 1000.0)
    assert stats.profit_factor is None


def test_all_wins_profit_factor_is_infinite():
    trades = [_trade(0, 10), _trade(1, 20)]
    stats = compute_stats(trades, _equity_curve([1000, 1010, 1030]), 1000.0)
    assert stats.profit_factor == float("inf")


def test_expectancy_is_average_pnl_per_trade():
    trades = [_trade(0, 100), _trade(1, -50), _trade(2, 0)]
    stats = compute_stats(trades, _equity_curve([1000] * 4), 1000.0)
    assert stats.expectancy_usd == pytest.approx((100 - 50 + 0) / 3)


def test_avg_r_multiple():
    trades = [_trade(0, 10, r_multiple=1.5), _trade(1, -10, r_multiple=-1.0), _trade(2, 20, r_multiple=2.0)]
    stats = compute_stats(trades, _equity_curve([1000] * 4), 1000.0)
    assert stats.avg_r_multiple == pytest.approx((1.5 - 1.0 + 2.0) / 3)


def test_max_drawdown_from_equity_curve():
    curve = _equity_curve([1000, 1200, 1100, 900, 950, 1300])
    stats = compute_stats([_trade(0, 300)], curve, 1000.0)
    # peak 1200 -> trough 900 = 25% drawdown
    assert stats.max_drawdown_pct == pytest.approx(25.0)


def test_max_drawdown_zero_on_a_straight_line_up():
    curve = _equity_curve([1000, 1100, 1200, 1300])
    stats = compute_stats([_trade(0, 300)], curve, 1000.0)
    assert stats.max_drawdown_pct == pytest.approx(0.0)


def test_longest_losing_and_winning_streaks():
    trades = [
        _trade(0, 10), _trade(1, 10), _trade(2, -10), _trade(3, -10), _trade(4, -10), _trade(5, 10),
    ]
    stats = compute_stats(trades, _equity_curve([1000] * 7), 1000.0)
    assert stats.longest_winning_streak == 2
    assert stats.longest_losing_streak == 3


def test_total_return_matches_final_equity():
    curve = _equity_curve([1000, 1100, 1250])
    stats = compute_stats([_trade(0, 250)], curve, 1000.0)
    assert stats.final_balance_usd == pytest.approx(1250)
    assert stats.total_return_usd == pytest.approx(250)
    assert stats.total_return_pct == pytest.approx(25.0)


def test_sharpe_and_sortino_are_none_with_fewer_than_two_trades():
    stats = compute_stats([_trade(0, 10)], _equity_curve([1000, 1010]), 1000.0)
    assert stats.sharpe_ratio is None
    assert stats.sortino_ratio is None


def test_sharpe_is_none_when_returns_have_zero_variance():
    trades = [_trade(0, 10), _trade(1, 10), _trade(2, 10)]
    stats = compute_stats(trades, _equity_curve([1000] * 4), 1000.0)
    assert stats.sharpe_ratio is None  # identical returns -> zero stdev -> undefined, not divide-by-zero


def test_sortino_uses_only_downside_deviation():
    trades = [_trade(0, 50), _trade(10, 60), _trade(20, -10), _trade(30, 55)]
    stats = compute_stats(trades, _equity_curve([1000] * 5), 1000.0)
    assert stats.sortino_ratio is not None
