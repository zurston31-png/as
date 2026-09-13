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
from app.signals.market_regime import classify_full
from app.data.live_provider import CHAIN_TO_GECKOTERMINAL_NETWORK, fetch_candles
from app.signals.scoring import SignalScore, score_signal

logger = logging.getLogger(__name__)


def unavailable_reason(chain: str, token_address: str | None, symbol: str) -> str:
    """Why a live score could not be produced, without repeating the fetch.

    `evaluate_live_entry_signal` returns None for five different reasons -
    no contract address, an invalid timeframe setting, a chain with no
    candle provider, a failed fetch, and a genuinely short history - and
    the caller used to record all five as "need >=N live candles". That
    reads as a measurement, and for four of the five nothing was ever
    measured: a blank address never reaches the provider at all.

    Reporting a count that was never taken is the same failure as inventing
    one, so the ambiguous case says it is ambiguous and names the tool that
    settles it. Everything here is local state - no network call, because
    this runs on a path that has already failed once.
    """
    if not token_address:
        return (
            f"no contract address on the {symbol} signal, so no candles could be requested - "
            f"set Token/Contract Address in the TradingView indicator settings"
        )

    try:
        Timeframe(settings.SIGNAL_SCORE_TIMEFRAME)
    except ValueError:
        return (
            f"SIGNAL_SCORE_TIMEFRAME={settings.SIGNAL_SCORE_TIMEFRAME!r} is not a valid "
            f"timeframe, so no candles could be requested"
        )

    if chain not in CHAIN_TO_GECKOTERMINAL_NETWORK:
        return (
            f"chain {chain!r} has no candle provider mapping, so {symbol} cannot be scored"
        )

    return (
        f"the candle provider returned no usable history for {symbol} ({token_address}) - "
        f"either it has no pool for this mint or the request failed. This is NOT a measured "
        f"candle count: run scripts/diagnose_token.py {token_address} to tell the two apart"
    )


async def evaluate_live_entry_signal(
    chain: str, token_address: str, symbol: str, liquidity_usd: float | None = None
) -> SignalScore | None:
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

    score = score_signal(series)
    # Classify from the SAME series the score was built on. Fetching
    # separately would cost another request and could land on a different
    # candle, so the recorded regime would not be the one the decision was
    # actually made in.
    score.market_condition = classify_full(series, liquidity_usd=liquidity_usd)
    return score
