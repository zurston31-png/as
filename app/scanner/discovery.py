"""Automatic token discovery — finds newly listed tokens to evaluate,
instead of waiting for a TradingView alert to name one.

Two sources, both optional and independently degradable:

  DexScreener  free, no API key. Lists any token once it has a liquidity
               pool and at least one transaction, which is exactly the
               population this bot cares about. Always used.
  Birdeye      needs BIRDEYE_API_KEY. Has a dedicated new-listing endpoint
               (including tokens from launchpads/meme platforms) and
               server-side minimum-liquidity filtering, so the scanner
               doesn't have to pull down and judge every random mint.
               Skipped entirely when no key is configured.

Discovery returns `DiscoveredToken`s carrying the market data already
present in the listing payload (liquidity, 24h volume, buy/sell counts,
price change, pool age). That matters for cost ordering: the scanner can
reject most candidates on those fields alone, BEFORE spending a rug-check
lookup or a candle fetch on them (see app/scanner/filters.py).

HONESTY NOTE, same as app/data/live_provider.py: these endpoint shapes come
from documented/trained knowledge of the DexScreener and Birdeye public
APIs, NOT from a live response verified in this development environment
(outbound HTTP to public APIs is proxied/restricted here in a way it won't
be on your deployment server). Every parse is defensive - an unrecognised
shape yields no tokens rather than malformed ones, and a source that errors
is logged and skipped rather than taking the whole scan down. Run
`python scripts/scan_once.py` on your real server to see what actually
comes back before leaving the scanner running unattended.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass

from app.config import settings
from app.services import http

logger = logging.getLogger(__name__)

DEXSCREENER_TOKEN_PROFILES = "https://api.dexscreener.com/token-profiles/latest/v1"
DEXSCREENER_TOKEN_BOOSTS = "https://api.dexscreener.com/token-boosts/latest/v1"
DEXSCREENER_TOKENS = "https://api.dexscreener.com/latest/dex/tokens/{addresses}"

# DexScreener accepts a comma-separated batch on the tokens endpoint; keeping
# batches modest keeps each URL short and each failure small.
DEXSCREENER_BATCH_SIZE = 25


@dataclass
class DiscoveredToken:
    """One candidate found by the scanner, with whatever market data the
    listing payload already carried. Anything a source didn't report stays
    None - never a fabricated 0, since "no volume reported" and "zero
    volume" mean very different things to the filters downstream.
    """

    token_address: str
    symbol: str
    chain: str
    source: str
    liquidity_usd: float | None = None
    volume_24h_usd: float | None = None
    buys_24h: int | None = None
    sells_24h: int | None = None
    price_usd: float | None = None
    price_change_1h_pct: float | None = None
    price_change_24h_pct: float | None = None
    pair_created_at: dt.datetime | None = None

    @property
    def age_hours(self) -> float | None:
        if self.pair_created_at is None:
            return None
        return (dt.datetime.now(dt.timezone.utc) - self.pair_created_at).total_seconds() / 3600


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _get_json(url: str, *, headers: dict | None = None, params: dict | None = None):
    """Discovery fetch, via the shared rate-limit-aware helper.

    The scanner is the heaviest caller in the bot - a batch of listing plus
    hydration requests on every tick - so it is the most likely to earn a
    429, and the one that most needs to back off rather than silently
    return nothing.
    """
    return await http.get_json(url, headers=headers, params=params, label=f"token discovery {url}")


# ---------------------------------------------------------------------------
# DexScreener
# ---------------------------------------------------------------------------

def _addresses_from_profile_payload(payload, chain: str) -> list[str]:
    """DexScreener's profile/boost endpoints return a flat list of
    {chainId, tokenAddress, ...}. Pull out the addresses for our chain."""
    if not isinstance(payload, list):
        return []
    out = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if (entry.get("chainId") or "").lower() != chain.lower():
            continue
        address = entry.get("tokenAddress")
        if address:
            out.append(str(address))
    return out


def _best_pair(pairs: list[dict]) -> dict | None:
    if not pairs:
        return None
    return max(pairs, key=lambda p: _to_float((p.get("liquidity") or {}).get("usd")) or 0.0)


def _token_from_pair(pair: dict, source: str) -> DiscoveredToken | None:
    base = pair.get("baseToken") or {}
    address = base.get("address")
    if not address:
        return None

    txns_24h = (pair.get("txns") or {}).get("h24") or {}
    price_change = pair.get("priceChange") or {}
    created_ms = pair.get("pairCreatedAt")
    created = (
        dt.datetime.fromtimestamp(created_ms / 1000, tz=dt.timezone.utc)
        if isinstance(created_ms, (int, float)) else None
    )

    return DiscoveredToken(
        token_address=str(address),
        symbol=str(base.get("symbol") or address[:8]),
        chain=str(pair.get("chainId") or settings.CHAIN),
        source=source,
        liquidity_usd=_to_float((pair.get("liquidity") or {}).get("usd")),
        volume_24h_usd=_to_float((pair.get("volume") or {}).get("h24")),
        buys_24h=_to_int(txns_24h.get("buys")),
        sells_24h=_to_int(txns_24h.get("sells")),
        price_usd=_to_float(pair.get("priceUsd")),
        price_change_1h_pct=_to_float(price_change.get("h1")),
        price_change_24h_pct=_to_float(price_change.get("h24")),
        pair_created_at=created,
    )


async def _hydrate_dexscreener(addresses: list[str], chain: str, source: str) -> list[DiscoveredToken]:
    """Turn bare addresses into DiscoveredTokens with full market data."""
    tokens: list[DiscoveredToken] = []
    for start in range(0, len(addresses), DEXSCREENER_BATCH_SIZE):
        batch = addresses[start:start + DEXSCREENER_BATCH_SIZE]
        payload = await _get_json(DEXSCREENER_TOKENS.format(addresses=",".join(batch)))
        pairs = (payload or {}).get("pairs") if isinstance(payload, dict) else None
        if not pairs:
            continue

        # One token can have many pools; keep only its deepest, which is what
        # every other part of this bot prices against too.
        by_token: dict[str, list[dict]] = {}
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            if (pair.get("chainId") or "").lower() != chain.lower():
                continue
            address = ((pair.get("baseToken") or {}).get("address") or "").strip()
            if address:
                by_token.setdefault(address, []).append(pair)

        for address, token_pairs in by_token.items():
            best = _best_pair(token_pairs)
            if best is None:
                continue
            token = _token_from_pair(best, source)
            if token is not None:
                tokens.append(token)
    return tokens


async def discover_dexscreener(chain: str) -> list[DiscoveredToken]:
    """Newly profiled + boosted tokens on `chain`, hydrated with market data."""
    profiles, boosts = await asyncio.gather(
        _get_json(DEXSCREENER_TOKEN_PROFILES),
        _get_json(DEXSCREENER_TOKEN_BOOSTS),
    )

    addresses: list[str] = []
    seen: set[str] = set()
    for payload in (profiles, boosts):
        for address in _addresses_from_profile_payload(payload, chain):
            if address not in seen:
                seen.add(address)
                addresses.append(address)

    if not addresses:
        return []
    return await _hydrate_dexscreener(addresses, chain, source="dexscreener")


# ---------------------------------------------------------------------------
# Birdeye (optional - only when an API key is configured)
# ---------------------------------------------------------------------------

def _birdeye_headers() -> dict:
    return {
        "X-API-KEY": settings.BIRDEYE_API_KEY or "",
        "x-chain": "solana",
        "accept": "application/json",
    }


async def discover_birdeye(chain: str) -> list[DiscoveredToken]:
    """Birdeye's dedicated new-listing feed. Solana only, key required.

    Birdeye's new-listing payload is deliberately thin (it exists to say
    "this exists now", not to describe the market), so the addresses it
    returns are hydrated through DexScreener for the market fields the
    filters need - one source for discovery, one for pricing, rather than
    two half-populated shapes to reconcile downstream.
    """
    if not settings.BIRDEYE_API_KEY:
        return []
    if chain.lower() != "solana":
        return []

    payload = await _get_json(
        f"{settings.BIRDEYE_API_BASE}/defi/v2/tokens/new_listing",
        headers=_birdeye_headers(),
        params={"limit": min(settings.SCANNER_MAX_TOKENS_PER_CYCLE, 50)},
    )
    if not isinstance(payload, dict):
        return []

    items = (payload.get("data") or {}).get("items")
    if not isinstance(items, list):
        logger.warning("unrecognised Birdeye new-listing response shape: %s", str(payload)[:300])
        return []

    addresses = [str(item["address"]) for item in items if isinstance(item, dict) and item.get("address")]
    if not addresses:
        return []
    return await _hydrate_dexscreener(addresses, chain, source="birdeye")


# ---------------------------------------------------------------------------
# combined
# ---------------------------------------------------------------------------

async def discover_tokens(chain: str | None = None) -> list[DiscoveredToken]:
    """Every newly listed candidate both sources know about, deduplicated by
    token address (DexScreener's copy wins on a tie, since it's the one
    carrying the market data either way)."""
    chain = (chain or settings.CHAIN).lower()

    results = await asyncio.gather(
        discover_dexscreener(chain),
        discover_birdeye(chain),
        return_exceptions=True,
    )

    tokens: list[DiscoveredToken] = []
    seen: set[str] = set()
    for result in results:
        if isinstance(result, BaseException):
            logger.warning("a discovery source failed: %s", result)
            continue
        for token in result:
            if token.token_address not in seen:
                seen.add(token.token_address)
                tokens.append(token)

    logger.info("token discovery found %d unique candidate(s) on %s", len(tokens), chain)
    return tokens
