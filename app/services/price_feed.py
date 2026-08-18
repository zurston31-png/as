"""Chain-agnostic USD price/liquidity lookups via the DexScreener public API
(https://docs.dexscreener.com/api/reference). No API key required, works
for both Solana and EVM token addresses, which keeps position monitoring
and paper-trading fills independent of which execution backend is active.
"""
import datetime as dt
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex/tokens"


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


@dataclass
class MarketSnapshot:
    """Everything usable out of DexScreener's best pair for a token, beyond
    the plain price/liquidity above. Feeds the rug-risk score's
    market-behavior factors (token age, volume, buy/sell balance, price
    swings) - real observed market data rather than another scanner guess.

    Any field DexScreener didn't return for this pair is None, never a
    fabricated 0 - a token with unavailable volume is not the same as a
    token with zero volume.
    """

    price_usd: float | None
    liquidity_usd: float | None
    volume_24h_usd: float | None
    buys_24h: int | None
    sells_24h: int | None
    price_change_1h_pct: float | None
    price_change_24h_pct: float | None
    pair_created_at: dt.datetime | None
    fdv_usd: float | None


async def get_market_snapshot(token_address: str) -> MarketSnapshot | None:
    try:
        pairs = await _fetch_pairs(token_address)
    except Exception:
        logger.warning("market snapshot lookup failed for %s", token_address, exc_info=True)
        return None
    best = _best_pair(pairs)
    if not best:
        return None

    txns_h24 = (best.get("txns") or {}).get("h24") or {}
    price_change = best.get("priceChange") or {}
    created_ms = best.get("pairCreatedAt")
    created = dt.datetime.fromtimestamp(created_ms / 1000, tz=dt.timezone.utc) if created_ms else None

    buys = txns_h24.get("buys")
    sells = txns_h24.get("sells")

    return MarketSnapshot(
        price_usd=_to_float(best.get("priceUsd")),
        liquidity_usd=_to_float((best.get("liquidity") or {}).get("usd")),
        volume_24h_usd=_to_float((best.get("volume") or {}).get("h24")),
        buys_24h=int(buys) if isinstance(buys, (int, float)) else None,
        sells_24h=int(sells) if isinstance(sells, (int, float)) else None,
        price_change_1h_pct=_to_float(price_change.get("h1")),
        price_change_24h_pct=_to_float(price_change.get("h24")),
        pair_created_at=created,
        fdv_usd=_to_float(best.get("fdv")),
    )
