"""Chain-agnostic USD price/liquidity lookups via the DexScreener public API
(https://docs.dexscreener.com/api/reference). No API key required, works
for both Solana and EVM token addresses, which keeps position monitoring
and paper-trading fills independent of which execution backend is active.
"""
import logging

import httpx

logger = logging.getLogger(__name__)

DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex/tokens"


async def _fetch_pairs(token_address: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{DEXSCREENER_BASE}/{token_address}")
        resp.raise_for_status()
        data = resp.json()
    return data.get("pairs") or []


def _best_pair(pairs: list[dict]) -> dict | None:
    if not pairs:
        return None
    return max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))


async def get_price_usd(token_address: str) -> float | None:
    try:
        pairs = await _fetch_pairs(token_address)
    except Exception:
        logger.warning("price lookup failed for %s", token_address, exc_info=True)
        return None
    best = _best_pair(pairs)
    if not best:
        return None
    price = best.get("priceUsd")
    return float(price) if price is not None else None


async def get_liquidity_usd(token_address: str) -> float | None:
    try:
        pairs = await _fetch_pairs(token_address)
    except Exception:
        logger.warning("liquidity lookup failed for %s", token_address, exc_info=True)
        return None
    if not pairs:
        return None
    return max(float((p.get("liquidity") or {}).get("usd") or 0) for p in pairs)
