from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.config import Config
from tradebot.models import Candle

BASE = datetime(2026, 3, 10, 14, 0, tzinfo=timezone.utc)  # 10:00 New York


@pytest.fixture
def config(tmp_path) -> Config:
    """A config that writes its state into a throwaway directory."""
    cfg = Config()
    cfg.market.symbol = "TEST"
    cfg.market.point_value = 1.0
    cfg.market.tick_size = 0.25
    cfg.market.qty_step = 1.0
    cfg.market.min_qty = 1.0
    cfg.risk.starting_equity = 100_000.0
    cfg.risk.state_path = str(tmp_path / "risk.json")
    cfg.execution.trades_path = str(tmp_path / "trades.jsonl")
    cfg.execution.slippage_ticks = 0.0
    cfg.ai.enabled = False
    cfg.vision.enabled = False
    return cfg


def make_candles(closes, start=BASE, minutes=5, volume=1000.0, spread=1.0):
    """Build a clean series from a list of closes."""
    out = []
    prev = closes[0]
    for i, close in enumerate(closes):
        high = max(prev, close) + spread
        low = min(prev, close) - spread
        out.append(Candle(start + timedelta(minutes=minutes * i), prev, high, low, close, volume))
        prev = close
    return out


@pytest.fixture
def uptrend():
    """A clean, volume-backed advance - should satisfy a long stack."""
    closes = [100 + i * 0.6 for i in range(40)]
    closes += [124 - i * 0.2 for i in range(6)]      # small pullback
    closes += [122.8 + i * 0.8 for i in range(14)]   # resumption
    return make_candles(closes)
