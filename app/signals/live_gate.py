"""Wires the 0-100 signal-scoring engine (app/signals/scoring.py) into LIVE
entry decisions, via live OHLCV candles from GeckoTerminal
(app/data/live_provider.py).

Before this existed, the score was fully built and tested (Stage 2) but
only ever fed the backtester - live trading entered purely on a
TradingView buy alert plus the rug check, with no live candle source to
compute a score from. This module is the missing piece.

Fail-closed by design, same rule the rug-check filter and every risk gate
in this codebase already follow: `evaluate_live_entry_signal` returns None
when a trustworthy score cannot be produced (fetch failure, too few live
candles, unrecognised chain) - the caller (app/services/trading_service.py)
must treat None as "reject", never as "skip the check and let it through".
A signal score this bot could not actually compute is not evidence the
setup was good; it is only evidence the bot didn't look.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.data.candles import Timeframe
from app.data.live_provider import fetch_candles
from app.signals.scoring import SignalScore, score_signal

logger = logging.getLogger(__name__)


async def evaluate_live_entry_signal(chain: str, token_address: str, symbol: str) -> SignalScore | None:
    """Fetch live candles for `token_address` and score the setup, or
    return None if a trustworthy score can't be produced."""
    if not token_address:
        return None

    try:
        timeframe = Timeframe(settings.SIGNAL_SCORE_TIMEFRAME)
    except ValueError:
        logger.error(
            "SIGNAL_SCORE_TIMEFRAME=%r is not a valid timeframe - live signal score unavailable",
            settings.SIGNAL_SCORE_TIMEFRAME,
        )
        return None

    series = await fetch_candles(chain, token_address, symbol, timeframe, settings.SIGNAL_SCORE_CANDLE_LIMIT)
    if series is None or len(series) < settings.SIGNAL_SCORE_MIN_CANDLES:
        got = 0 if series is None else len(series)
        logger.info(
            "insufficient live candle data for %s (%s): got %d, need >=%d",
            symbol, token_address, got, settings.SIGNAL_SCORE_MIN_CANDLES,
        )
        return None

    return score_signal(series)
