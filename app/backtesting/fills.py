"""Realistic fill simulation for the backtester.

Three costs are modeled, each applied per side (once on entry, once on
exit) rather than netted round-trip, so a trade's cost is visible on both
legs the way it would be in a real fill:

  slippage  price impact from the order itself moving a thin memecoin pool
  spread    the bid/ask gap you cross just to transact
  fee       the exchange/DEX's own cut

Slippage and spread both push the fill price AGAINST the trader - worse on
a buy (pay more), worse on a sell (receive less) - which is what makes a
backtest that ignores them systematically too optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.backtesting.types import BacktestConfig


@dataclass
class Fill:
    price: float
    fee_usd: float


def buy_fill(reference_price: float, usd_amount: float, config: BacktestConfig) -> Fill:
    impact = config.slippage_pct + config.spread_pct
    price = reference_price * (1 + impact)
    fee_usd = usd_amount * config.fee_pct
    return Fill(price=price, fee_usd=fee_usd)


def sell_fill(reference_price: float, qty: float, config: BacktestConfig) -> Fill:
    impact = config.slippage_pct + config.spread_pct
    price = reference_price * (1 - impact)
    fee_usd = (qty * price) * config.fee_pct
    return Fill(price=max(price, 0.0), fee_usd=fee_usd)
