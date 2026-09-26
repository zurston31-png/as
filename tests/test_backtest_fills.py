import pytest

from app.backtesting.fills import buy_fill, sell_fill
from app.backtesting.types import BacktestConfig


def test_buy_fill_is_worse_than_the_reference_price():
    config = BacktestConfig(slippage_pct=0.01, spread_pct=0.005, fee_pct=0.0)
    fill = buy_fill(1.0, usd_amount=100.0, config=config)
    assert fill.price > 1.0
    assert fill.price == pytest.approx(1.015)


def test_sell_fill_is_worse_than_the_reference_price():
    config = BacktestConfig(slippage_pct=0.01, spread_pct=0.005, fee_pct=0.0)
    fill = sell_fill(1.0, qty=100.0, config=config)
    assert fill.price < 1.0
    assert fill.price == pytest.approx(0.985)


def test_buy_fill_fee_scales_with_notional():
    config = BacktestConfig(slippage_pct=0.0, spread_pct=0.0, fee_pct=0.01)
    fill = buy_fill(1.0, usd_amount=1000.0, config=config)
    assert fill.fee_usd == pytest.approx(10.0)


def test_sell_fill_fee_scales_with_proceeds():
    config = BacktestConfig(slippage_pct=0.0, spread_pct=0.0, fee_pct=0.01)
    fill = sell_fill(2.0, qty=50.0, config=config)
    assert fill.fee_usd == pytest.approx(1.0)


def test_sell_fill_never_goes_negative_under_extreme_impact():
    config = BacktestConfig(slippage_pct=2.0, spread_pct=2.0, fee_pct=0.0)
    fill = sell_fill(1.0, qty=10.0, config=config)
    assert fill.price >= 0.0
