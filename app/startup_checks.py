"""Config coherence checks, run once at startup.

Every individual setting in this bot can be perfectly reasonable while the
COMBINATION makes trading impossible. That failure is invisible: the bot
starts fine, the logs look healthy, candidates get discovered, and nothing
ever trades - which is indistinguishable from "the market had no good
setups this week".

That is exactly what happened with the shipped defaults. The scanner
admitted tokens from 6 hours old, but the signal score demanded 60 x 15m
candles = 15 hours of history, so every token between those two ages was
discovered, passed the pre-screen, and was then guaranteed to be rejected -
forever, by construction, with a perfectly ordinary-looking log line.

These checks exist to make that class of mistake loud at boot instead of
silent at runtime. They only ever WARN - none of them stop the bot or
change a value, because a deployer may have a good reason for an unusual
combination and this file has no business overruling them.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.data.candles import Timeframe

logger = logging.getLogger(__name__)

# What the scoring engine actually produces, measured across 120 synthetic
# runs spanning bull/pump/sideways/bear/high-volatility regimes at 120-300
# candles (see the commit that added this file). The score is a weighted
# average of 14 factors, so it regresses toward 50 by construction - the
# practical ceiling is nowhere near 100.
#
#   median  58      90th pct  70      95th pct  74      99th pct  79
#
# Which makes the qualifying rate at each threshold:
#   60 -> ~43%    65 -> ~26%    70 -> ~10%    75 -> ~3%    80 -> ~1%
SCORE_PERCENTILES = {60: 43, 65: 26, 70: 10, 75: 3, 80: 1}
SCORE_EFFECTIVELY_NEVER = 75


def check_config_coherence() -> list[str]:
    """Return a list of human-readable warnings about the current config.

    Logged at startup and surfaced on the dashboard, so "why isn't it
    trading" has an answer visible before you go looking for one.
    """
    warnings: list[str] = []

    # --- the score gate vs the candle requirement ---
    try:
        timeframe = Timeframe(settings.SIGNAL_SCORE_TIMEFRAME)
    except ValueError:
        warnings.append(
            f"SIGNAL_SCORE_TIMEFRAME={settings.SIGNAL_SCORE_TIMEFRAME!r} is not a valid timeframe - "
            "the live signal score will be unavailable and EVERY buy will be rejected"
        )
        timeframe = None

    if timeframe is not None:
        required_hours = settings.SIGNAL_SCORE_MIN_CANDLES * timeframe.seconds / 3600
        if settings.SCANNER_ENABLED and required_hours > settings.SCANNER_MIN_TOKEN_AGE_HOURS:
            warnings.append(
                f"IMPOSSIBLE COMBINATION: the signal score needs "
                f"{settings.SIGNAL_SCORE_MIN_CANDLES} x {timeframe.value} "
                f"({required_hours:.0f}h of history) but the scanner admits tokens from "
                f"{settings.SCANNER_MIN_TOKEN_AGE_HOURS:.0f}h old. Tokens between "
                f"{settings.SCANNER_MIN_TOKEN_AGE_HOURS:.0f}h and {required_hours:.0f}h will pass the "
                f"pre-screen and then ALWAYS fail the score gate. Raise "
                f"SCANNER_MIN_TOKEN_AGE_HOURS to at least {required_hours:.0f}, or lower "
                f"SIGNAL_SCORE_MIN_CANDLES."
            )

    # --- the score threshold vs what the engine actually produces ---
    threshold = settings.MIN_SIGNAL_SCORE_TO_ENTER
    if settings.LIVE_SIGNAL_SCORE_ENABLED and threshold >= SCORE_EFFECTIVELY_NEVER:
        nearest = min(SCORE_PERCENTILES, key=lambda k: abs(k - threshold))
        rate = SCORE_PERCENTILES[nearest]
        warnings.append(
            f"MIN_SIGNAL_SCORE_TO_ENTER={threshold:.0f} is at or above the ~97th percentile of what "
            f"the scoring engine actually produces (roughly {rate}% of setups reach {nearest}). "
            f"Combined with the rug check, expect very few or zero trades. 65 (~26% of setups) "
            f"still rejects three out of four candidates while letting the bot actually trade."
        )

    # --- trade size vs the shallowest pool the scanner will accept ---
    # Price impact is trade_usd / (liquidity/2), so the worst realistic case
    # is the biggest allowed trade against the thinnest allowed pool. If
    # that exceeds the slippage tolerance, those fills simply revert - real
    # behavior, but worth knowing about before it looks like a mystery.
    if settings.SCANNER_ENABLED and settings.SCANNER_MIN_LIQUIDITY_USD > 0:
        worst_impact = settings.MAX_TRADE_SIZE_USD / (settings.SCANNER_MIN_LIQUIDITY_USD / 2)
        tolerance = settings.SLIPPAGE_BPS / 10_000
        if worst_impact > tolerance:
            warnings.append(
                f"MAX_TRADE_SIZE_USD=${settings.MAX_TRADE_SIZE_USD:,.0f} against the thinnest pool the "
                f"scanner accepts (${settings.SCANNER_MIN_LIQUIDITY_USD:,.0f}) implies "
                f"{worst_impact * 100:.1f}% price impact, above the {tolerance * 100:.1f}% "
                f"SLIPPAGE_BPS tolerance - those fills will revert. Either raise "
                f"SCANNER_MIN_LIQUIDITY_USD, lower MAX_TRADE_SIZE_USD, or widen SLIPPAGE_BPS."
            )

    # --- gates that are switched off entirely ---
    if not settings.RUGCHECK_ENABLED:
        warnings.append("RUGCHECK_ENABLED=false - buy signals are NOT being screened for scams/rugs")
    if not settings.LIVE_SIGNAL_SCORE_ENABLED:
        warnings.append("LIVE_SIGNAL_SCORE_ENABLED=false - buys are NOT being scored before entry")

    # --- nothing can find a token to trade ---
    if not settings.SCANNER_ENABLED:
        warnings.append(
            "SCANNER_ENABLED=false - the bot will only ever trade tokens named by an inbound "
            "TradingView alert. If you aren't sending alerts, it will never trade at all."
        )

    # --- risk limits that block everything ---
    if settings.MAX_CONCURRENT_POSITIONS < 1:
        warnings.append("MAX_CONCURRENT_POSITIONS < 1 - no position can ever be opened")
    if settings.MAX_DAILY_TRADES < 1:
        warnings.append("MAX_DAILY_TRADES < 1 - no trade can ever be opened")
    if settings.MAX_TRADE_SIZE_USD <= 0:
        warnings.append("MAX_TRADE_SIZE_USD <= 0 - every position would be sized to nothing")
    if settings.MAX_TOTAL_EXPOSURE_PCT <= 0:
        warnings.append("MAX_TOTAL_EXPOSURE_PCT <= 0 - every position would be sized to nothing")

    return warnings


def log_config_coherence() -> list[str]:
    """Run the checks and log them. Returns the warnings for the dashboard."""
    warnings = check_config_coherence()
    if not warnings:
        logger.info("config coherence check passed - no impossible combinations found")
        return warnings

    logger.warning("=" * 70)
    logger.warning("CONFIG WARNINGS - these may stop the bot from ever trading:")
    for warning in warnings:
        logger.warning("  * %s", warning)
    logger.warning("=" * 70)
    return warnings
