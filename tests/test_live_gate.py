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


# ---------------------------------------------------------------------------
# why the score was unavailable - the five causes must not share one message
# ---------------------------------------------------------------------------

def test_a_blank_address_is_not_reported_as_a_candle_shortage():
    """Regression, from a live run. A BONK signal was rejected with "need
    >=60 live 15m candles" - a count of a fetch that never happened, because
    `evaluate_live_entry_signal` returns None on a blank address before it
    reaches the provider at all.

    BONK is among the most liquid mints on Solana, so the stated reason sent
    the search in exactly the wrong direction. Reporting a measurement that
    was never taken is the same failure as inventing one.
    """
    reason = live_gate.unavailable_reason("solana", None, "BONK")
    assert "no contract address" in reason
    assert "Token/Contract Address" in reason  # names the field to fix
    # Saying "no candles could be requested" is honest - it describes what
    # did not happen. Quoting a threshold is not, because that implies a
    # fetch came back short. Ban the count, not the word.
    assert ">=" not in reason
    assert str(settings.SIGNAL_SCORE_MIN_CANDLES) not in reason


def test_an_unmapped_chain_says_so_rather_than_blaming_the_history():
    reason = live_gate.unavailable_reason("dogecoin", "SomeAddress111", "DOGE")
    assert "no candle provider" in reason
    assert "dogecoin" in reason


def test_an_invalid_timeframe_setting_says_so(monkeypatch):
    monkeypatch.setattr(settings, "SIGNAL_SCORE_TIMEFRAME", "13m")
    reason = live_gate.unavailable_reason("solana", "SomeAddress111", "TESTCOIN")
    assert "not a valid timeframe" in reason


def test_the_genuinely_ambiguous_case_admits_it_is_ambiguous():
    """With an address, a real chain and a valid timeframe, the failure is
    either "no pool for this mint" or "the request failed" - and this code
    cannot tell which without asking again. It must not pick one, and it
    must not quote a candle count it never measured."""
    reason = live_gate.unavailable_reason("solana", "RealMint111", "BONK")
    assert "no pool for this mint or the request failed" in reason
    assert "NOT a measured candle count" in reason
    assert "diagnose_token.py RealMint111" in reason  # names the tool that settles it
