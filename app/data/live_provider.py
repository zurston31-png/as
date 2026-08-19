"""Live OHLCV candles from GeckoTerminal (https://www.geckoterminal.com/dex-api),
for wiring the signal-scoring engine (app/signals/scoring.py) into LIVE entry
decisions - everything that engine was built on (app/data/candles.py,
app/data/providers.py) previously only fed the backtester, since the bot had
no live source of actual OHLCV data for on-chain memecoins.

GeckoTerminal was chosen over paid alternatives (e.g. Birdeye) because it
needs no API key and has a generous free tier, at the cost of coarser
1-minute-minimum candle granularity and its own rate limits (roughly 30
requests/minute on the public tier as of this writing - PRICE_POLL_INTERVAL_SECONDS
doesn't apply to this, so don't call it faster than the watchlist size
reasonably needs).

HONESTY NOTE, read before trusting this against a real token: this module
is written from documented/trained knowledge of GeckoTerminal's public API
v2 shape (JSON:API-style `data`/`attributes` nesting, `/tokens/{addr}/pools`
for pool discovery, `/pools/{addr}/ohlcv/{timeframe}` for candles), NOT
verified against a live response from this sandboxed environment - outbound
HTTP to public APIs is proxied/restricted here in a way it won't be on your
actual deployment server. Every parse below is defensive (missing/renamed
fields raise or return None rather than silently producing wrong data), and
the fail-closed design in app/signals/live_gate.py means a shape mismatch
rejects trades rather than admitting bad ones - but you should still run
`scripts/diagnose_token.py <address>` (or watch the first several signals
in the dashboard) against a real token on your deployment before trusting
this gate's verdicts unattended.
"""
from __future__ import annotations

import datetime as dt
import logging

from app.config import settings
from app.data.candles import Candle, CandleSeries, Timeframe
from app.services import http

logger = logging.getLogger(__name__)

# Our internal chain labels (used throughout app/rugcheck/, app/execution/)
# to GeckoTerminal's network slugs. Extend this if you add a chain
# elsewhere in the bot - an unmapped chain fails closed (see fetch_candles).
CHAIN_TO_GECKOTERMINAL_NETWORK = {
    "solana": "solana",
    "ethereum": "eth",
    "bsc": "bsc",
    "polygon": "polygon_pos",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "base": "base",
    "avalanche": "avax",
}

# (GeckoTerminal timeframe path segment, aggregate multiple) per our Timeframe.
_TIMEFRAME_MAP: dict[Timeframe, tuple[str, int]] = {
    Timeframe.M1: ("minute", 1),
    Timeframe.M5: ("minute", 5),
    Timeframe.M15: ("minute", 15),
    Timeframe.H1: ("hour", 1),
    Timeframe.H4: ("hour", 4),
    Timeframe.D1: ("day", 1),
}


async def _get_json(url: str, params: dict | None = None) -> dict | None:
    """Candle fetch, via the shared rate-limit-aware helper.

    GeckoTerminal's free tier has the tightest limit of any source here
    (roughly 30 requests/minute), and both the scanner and the trade path
    call it. A 429 that isn't retried means rejecting a setup that was
    actually fine - so this is the call that most benefits from backing
    off and trying again rather than failing closed immediately.
    """
    return await http.get_json(
        url, params=params, headers={"Accept": "application/json"},
        label=f"GeckoTerminal {url}",
    )


async def _find_primary_pool(network: str, token_address: str) -> str | None:
    """The token's highest-liquidity pool address, for the OHLCV endpoint
    (which is keyed by pool, not by token)."""
    data = await _get_json(f"{settings.GECKOTERMINAL_API_BASE}/networks/{network}/tokens/{token_address}/pools")
    if not data:
        return None

    pools = data.get("data")
    if not isinstance(pools, list) or not pools:
        return None

    def reserve_usd(pool: dict) -> float:
        try:
            return float((pool.get("attributes") or {}).get("reserve_in_usd") or 0)
        except (TypeError, ValueError):
            return 0.0

    best = max(pools, key=reserve_usd)
    try:
        return best["attributes"]["address"]
    except (KeyError, TypeError):
        return None


def _parse_ohlcv_response(data: dict, symbol: str, timeframe: Timeframe) -> CandleSeries | None:
    try:
        rows = data["data"]["attributes"]["ohlcv_list"]
    except (KeyError, TypeError):
        logger.warning("unrecognised GeckoTerminal OHLCV response shape: %s", data)
        return None

    if not isinstance(rows, list):
        return None

    candles: list[Candle] = []
    for row in rows:
        try:
            ts, o, h, l, c, v = row
            candles.append(Candle(
                timestamp=dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc),
                open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v or 0),
            ))
        except (TypeError, ValueError, IndexError):
            continue  # skip a single malformed row rather than discard the whole series

    if not candles:
        return None

    # GeckoTerminal has returned newest-first for this endpoint; sort
    # defensively rather than assume either order, since CandleSeries and
    # every indicator built on it require oldest-first.
    candles.sort(key=lambda c: c.timestamp)
    return CandleSeries(symbol, timeframe, candles)


async def fetch_candles(
    chain: str, token_address: str, symbol: str, timeframe: Timeframe, limit: int
) -> CandleSeries | None:
    """Live OHLCV for one token, or None if it can't be fetched/parsed -
    the caller (app/signals/live_gate.py) must treat None as "no data",
    never as "safe to skip the check"."""
    network = CHAIN_TO_GECKOTERMINAL_NETWORK.get(chain.lower())
    if network is None:
        logger.warning("no GeckoTerminal network mapping for chain %r - live signal score unavailable", chain)
        return None

    pool_address = await _find_primary_pool(network, token_address)
    if not pool_address:
        return None

    path_segment, aggregate = _TIMEFRAME_MAP[timeframe]
    data = await _get_json(
        f"{settings.GECKOTERMINAL_API_BASE}/networks/{network}/pools/{pool_address}/ohlcv/{path_segment}",
        params={"aggregate": aggregate, "limit": min(limit, 1000), "currency": "usd"},
    )
    if not data:
        return None

    return _parse_ohlcv_response(data, symbol, timeframe)
