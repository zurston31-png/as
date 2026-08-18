"""Cheap pre-screen for discovered tokens, run BEFORE any further network
call is spent on them.

Every field checked here already arrived in the discovery payload
(app/scanner/discovery.py), so rejecting on it costs nothing. That ordering
is the whole point: a scan cycle can surface hundreds of brand-new mints,
and the expensive stages downstream - the rug check (several scanner
lookups) and the signal score (pool resolution + a candle fetch) - should
only ever run on the handful that could plausibly be traded.

Fail-closed on missing data, same rule as everywhere else in this bot: a
token whose liquidity or volume simply wasn't reported is rejected, not
waved through. "The data source didn't say" is not evidence a brand-new
memecoin is safe to buy.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.scanner.discovery import DiscoveredToken


@dataclass
class FilterVerdict:
    passed: bool
    reason: str = ""


def prescreen(token: DiscoveredToken) -> FilterVerdict:
    """Judge a discovered token on its listing data alone."""
    if token.liquidity_usd is None:
        return FilterVerdict(False, "no liquidity reported by the listing source")
    if token.liquidity_usd < settings.SCANNER_MIN_LIQUIDITY_USD:
        return FilterVerdict(
            False,
            f"liquidity ${token.liquidity_usd:,.0f} below scanner minimum "
            f"${settings.SCANNER_MIN_LIQUIDITY_USD:,.0f}",
        )

    if token.volume_24h_usd is None:
        return FilterVerdict(False, "no 24h volume reported by the listing source")
    if token.volume_24h_usd < settings.SCANNER_MIN_VOLUME_24H_USD:
        return FilterVerdict(
            False,
            f"24h volume ${token.volume_24h_usd:,.0f} below scanner minimum "
            f"${settings.SCANNER_MIN_VOLUME_24H_USD:,.0f}",
        )

    age_hours = token.age_hours
    if age_hours is None:
        return FilterVerdict(False, "no pool creation time reported - token age unknown")
    if age_hours < settings.SCANNER_MIN_TOKEN_AGE_HOURS:
        return FilterVerdict(
            False,
            f"pool is {age_hours:.1f}h old, under the {settings.SCANNER_MIN_TOKEN_AGE_HOURS:.1f}h minimum "
            "(the highest-risk rug window)",
        )
    if settings.SCANNER_MAX_TOKEN_AGE_HOURS > 0 and age_hours > settings.SCANNER_MAX_TOKEN_AGE_HOURS:
        return FilterVerdict(
            False,
            f"pool is {age_hours / 24:.1f}d old, past the {settings.SCANNER_MAX_TOKEN_AGE_HOURS / 24:.1f}d "
            "window this scanner targets",
        )

    # A pool with almost no trades has no meaningful price history to score
    # against, whatever its headline liquidity says.
    buys, sells = token.buys_24h, token.sells_24h
    if buys is None or sells is None:
        return FilterVerdict(False, "no 24h buy/sell counts reported by the listing source")
    total_txns = buys + sells
    if total_txns < settings.SCANNER_MIN_TXNS_24H:
        return FilterVerdict(
            False, f"only {total_txns} trades in 24h, under the {settings.SCANNER_MIN_TXNS_24H} minimum"
        )

    # Overwhelming sell pressure on a brand-new token is the shape of a
    # distribution/exit, not an entry.
    sell_share = sells / total_txns if total_txns else 0.0
    if sell_share >= settings.SCANNER_MAX_SELL_SHARE:
        return FilterVerdict(
            False,
            f"{sell_share * 100:.0f}% of 24h trades are sells "
            f"(limit {settings.SCANNER_MAX_SELL_SHARE * 100:.0f}%) - distribution pressure",
        )

    return FilterVerdict(True, "passed scanner pre-screen")
