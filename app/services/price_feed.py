"""Chain-agnostic USD price/liquidity lookups via the DexScreener public API
(https://docs.dexscreener.com/api/reference). No API key required, works
for both Solana and EVM token addresses, which keeps position monitoring
and paper-trading fills independent of which execution backend is active.
"""
import datetime as dt
import logging
from dataclasses import dataclass, field

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
    market-behavior factors and the market-quality score
    (app/signals/market_quality.py) - real observed market data rather than
    another scanner guess.

    Any field DexScreener didn't return for this pair is None, never a
    fabricated 0 - a token with unavailable volume is not the same as a
    token with zero volume, and the scores downstream treat those two
    cases very differently.

    Multiple volume/price windows are carried (5m, 1h, 6h, 24h) because
    market quality depends on the SHAPE of activity, not just its total:
    comparing the last hour against the 24h average is what distinguishes
    a steady market from a single wash-traded burst.
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
    # --- identity / venue ---
    token_address: str | None = None
    token_name: str | None = None
    token_symbol: str | None = None
    pair_address: str | None = None
    dex_id: str | None = None
    chain_id: str | None = None
    market_cap_usd: float | None = None
    # --- shorter volume windows ---
    volume_5m_usd: float | None = None
    volume_1h_usd: float | None = None
    volume_6h_usd: float | None = None
    # --- shorter price-change windows ---
    price_change_5m_pct: float | None = None
    price_change_6h_pct: float | None = None
    # --- shorter transaction windows ---
    buys_1h: int | None = None
    sells_1h: int | None = None
    # When this observation was taken. Stale data is worse than no data,
    # because it looks authoritative - callers check this before trusting
    # the rest of the snapshot.
    observed_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    @property
    def total_txns_24h(self) -> int | None:
        if self.buys_24h is None or self.sells_24h is None:
            return None
        return self.buys_24h + self.sells_24h

    @property
    def buy_sell_ratio_24h(self) -> float | None:
        """Buys per sell. None when unknown; inf when there are no sells at
        all, which is itself a meaningful (and suspicious) reading."""
        if self.buys_24h is None or self.sells_24h is None:
            return None
        if self.sells_24h == 0:
            return float("inf") if self.buys_24h else None
        return self.buys_24h / self.sells_24h

    @property
    def age_hours(self) -> float | None:
        if self.pair_created_at is None:
            return None
        return (dt.datetime.now(dt.timezone.utc) - self.pair_created_at).total_seconds() / 3600

    def age_seconds(self, now: dt.datetime | None = None) -> float:
        """How old this OBSERVATION is (not the pool). Feeds stale-data
        rejection - see app/data/staleness.py."""
        now = now or dt.datetime.now(dt.timezone.utc)
        observed = self.observed_at
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=dt.timezone.utc)
        return (now - observed).total_seconds()


async def get_market_snapshot(token_address: str) -> MarketSnapshot | None:
    try:
        pairs = await _fetch_pairs(token_address)
    except Exception:
        logger.warning("market snapshot lookup failed for %s", token_address, exc_info=True)
        return None
    best = _best_pair(pairs)
    if not best:
        return None
    return _snapshot_from_pair(best, token_address)


def _snapshot_from_pair(pair: dict, token_address: str) -> MarketSnapshot:
    """Parse one DexScreener pair into a MarketSnapshot.

    Split out from the fetch so it can be unit-tested against a recorded
    payload without any network involved.
    """
    def window(section: str, key: str):
        return _to_float((pair.get(section) or {}).get(key))

    def txns(key: str) -> tuple[int | None, int | None]:
        entry = (pair.get("txns") or {}).get(key) or {}
        buys, sells = entry.get("buys"), entry.get("sells")
        return (
            int(buys) if isinstance(buys, (int, float)) else None,
            int(sells) if isinstance(sells, (int, float)) else None,
        )

    base = pair.get("baseToken") or {}
    created_ms = pair.get("pairCreatedAt")
    created = (
        dt.datetime.fromtimestamp(created_ms / 1000, tz=dt.timezone.utc)
        if isinstance(created_ms, (int, float)) else None
    )
    buys_24h, sells_24h = txns("h24")
    buys_1h, sells_1h = txns("h1")

    return MarketSnapshot(
        price_usd=_to_float(pair.get("priceUsd")),
        liquidity_usd=_to_float((pair.get("liquidity") or {}).get("usd")),
        volume_24h_usd=window("volume", "h24"),
        buys_24h=buys_24h,
        sells_24h=sells_24h,
        price_change_1h_pct=window("priceChange", "h1"),
        price_change_24h_pct=window("priceChange", "h24"),
        pair_created_at=created,
        fdv_usd=_to_float(pair.get("fdv")),
        token_address=str(base.get("address") or token_address),
        token_name=base.get("name"),
        token_symbol=base.get("symbol"),
        pair_address=pair.get("pairAddress"),
        dex_id=pair.get("dexId"),
        chain_id=pair.get("chainId"),
        market_cap_usd=_to_float(pair.get("marketCap")),
        volume_5m_usd=window("volume", "m5"),
        volume_1h_usd=window("volume", "h1"),
        volume_6h_usd=window("volume", "h6"),
        price_change_5m_pct=window("priceChange", "m5"),
        price_change_6h_pct=window("priceChange", "h6"),
        buys_1h=buys_1h,
        sells_1h=sells_1h,
    )
