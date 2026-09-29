"""Broker interface.

`PaperBroker` is the only full implementation and is what stages 1-4 of the
rollout (backtest -> live paper -> alerts) use. `LiveBroker` is a deliberate
stub: filling it in is the *last* step, not the first.
"""

from __future__ import annotations

from typing import Protocol

from ..models import Candle, Position, TradeSignal


class Broker(Protocol):
    def open(self, signal: TradeSignal, qty: float) -> Position: ...
    def on_candle(self, candle: Candle) -> list: ...
    @property
    def positions(self) -> list[Position]: ...


class LiveBrokerNotConfigured(RuntimeError):
    pass


class LiveBroker:
    """Placeholder for a real brokerage adapter.

    Wiring this up is intentionally left as an explicit, separate decision:
    every method raises until someone implements order placement, order-status
    reconciliation, and position sync against the broker's own records. Do not
    make it "work" by forwarding to the paper broker - silent paper fills that
    look like live fills are how accounts get blown up.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise LiveBrokerNotConfigured(
            "execution.mode is 'live' but no live broker adapter is implemented. "
            "Run in 'paper' or 'alerts_only' mode, or implement tradebot/execution/base.py:LiveBroker "
            "against your broker's API (order placement, fill reconciliation, and position sync)."
        )
