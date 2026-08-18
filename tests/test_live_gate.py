"""Tests for app/signals/live_gate.py - the fail-closed wrapper that turns
live GeckoTerminal candles into a signal score, or None when it can't.
"""
import pytest

from app.config import settings
from app.data.candles import Candle, CandleSeries, Timeframe
import app.signals.live_gate as live_gate
from app.signals.live_gate import evaluate_live_entry_signal

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _series(n: int) -> CandleSeries:
    import datetime as dt

    candles = []
    price = 1.0
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i in range(n):
        price *= 1.001
        candles.append(Candle(
            timestamp=start + dt.timedelta(minutes=15 * i),
            open=price, high=price * 1.01, low=price * 0.99, close=price, volume=10_000,
        ))
    return CandleSeries("TESTCOIN", Timeframe.M15, candles)


async def test_returns_none_without_a_token_address():
    result = await evaluate_live_entry_signal("solana", None, "TESTCOIN")
    assert result is None


async def test_returns_none_when_no_candles_are_fetched(monkeypatch):
    async def fake_fetch(chain, token_address, symbol, timeframe, limit):
        return None

    monkeypatch.setattr(live_gate, "fetch_candles", fake_fetch)
    result = await evaluate_live_entry_signal("solana", "Addr111", "TESTCOIN")
    assert result is None


async def test_returns_none_when_too_few_candles(monkeypatch):
    async def fake_fetch(chain, token_address, symbol, timeframe, limit):
        return _series(10)  # well under SIGNAL_SCORE_MIN_CANDLES

    monkeypatch.setattr(live_gate, "fetch_candles", fake_fetch)
    result = await evaluate_live_entry_signal("solana", "Addr111", "TESTCOIN")
    assert result is None


async def test_returns_a_score_when_enough_candles_are_available(monkeypatch):
    async def fake_fetch(chain, token_address, symbol, timeframe, limit):
        return _series(300)

    monkeypatch.setattr(live_gate, "fetch_candles", fake_fetch)
    result = await evaluate_live_entry_signal("solana", "Addr111", "TESTCOIN")
    assert result is not None
    assert 0 <= result.score <= 100


async def test_returns_none_for_an_invalid_configured_timeframe(monkeypatch):
    monkeypatch.setattr(settings, "SIGNAL_SCORE_TIMEFRAME", "not-a-real-timeframe")
    result = await evaluate_live_entry_signal("solana", "Addr111", "TESTCOIN")
    assert result is None


async def test_passes_the_configured_timeframe_and_limit_through(monkeypatch):
    seen = {}

    async def fake_fetch(chain, token_address, symbol, timeframe, limit):
        seen.update(chain=chain, token_address=token_address, symbol=symbol, timeframe=timeframe, limit=limit)
        return _series(300)

    monkeypatch.setattr(live_gate, "fetch_candles", fake_fetch)
    await evaluate_live_entry_signal("solana", "Addr111", "TESTCOIN")
    assert seen["chain"] == "solana"
    assert seen["token_address"] == "Addr111"
    assert seen["symbol"] == "TESTCOIN"
    assert seen["timeframe"] == Timeframe(settings.SIGNAL_SCORE_TIMEFRAME)
    assert seen["limit"] == settings.SIGNAL_SCORE_CANDLE_LIMIT
