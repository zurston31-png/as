"""Turn a chart symbol into the mint address that symbol actually refers to.

This exists because of the rule in `app/identity.py`: the mint is the token
and the symbol is only a label. Anyone can mint a token called BONK, and
copycats are deliberate rather than accidental - sharing a symbol with
something people trust is the entire point of one.

So the Pine script's `Token/Contract Address` input cannot be filled in from
memory or from a search result's first hit. This resolves a symbol against
the live listing data, returns every distinct mint carrying that symbol
ranked by liquidity, and says plainly how much of a gap there is between the
top candidate and the next one. A symbol with two well-funded claimants is
reported as ambiguous rather than silently resolved to the larger.

Nothing here decides anything. It prints candidates for a human to choose
from, because choosing wrong means the bot screens one token and trades
another.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.scanner.discovery import DEXSCREENER_SEARCH, _get_json, _to_float, _to_int

# Below this a "candidate" is noise - a dust pool someone made to squat the
# symbol. Still reported, but never treated as the confident answer.
MIN_CREDIBLE_LIQUIDITY_USD = 25_000.0

# The top candidate is only unambiguous if it dwarfs the runner-up. Two mints
# with comparable liquidity under one symbol is exactly the situation where
# picking by size is a coin flip with real money behind it.
DOMINANCE_RATIO = 5.0


@dataclass
class Candidate:
    token_address: str
    symbol: str
    name: str | None
    chain: str
    liquidity_usd: float | None
    volume_24h_usd: float | None
    price_usd: float | None
    fdv_usd: float | None
    pair_count: int = 0

    @property
    def credible(self) -> bool:
        return (self.liquidity_usd or 0.0) >= MIN_CREDIBLE_LIQUIDITY_USD


@dataclass
class Resolution:
    symbol: str
    requested_chain: str | None
    candidates: list[Candidate] = field(default_factory=list)
    error: str | None = None

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def unambiguous(self) -> bool:
        """One credible claimant, clear of the next by DOMINANCE_RATIO."""
        if not self.candidates or not self.candidates[0].credible:
            return False
        if len(self.candidates) == 1:
            return True
        top = self.candidates[0].liquidity_usd or 0.0
        runner_up = self.candidates[1].liquidity_usd or 0.0
        if runner_up <= 0:
            return True
        return top / runner_up >= DOMINANCE_RATIO

    def verdict(self) -> str:
        if self.error:
            return f"UNRESOLVED - {self.error}"
        if not self.candidates:
            return "NO MATCH - nothing listed under this symbol"
        if not self.candidates[0].credible:
            return (
                f"TOO THIN - best claimant holds only "
                f"${self.candidates[0].liquidity_usd or 0:,.0f} of liquidity"
            )
        if not self.unambiguous:
            return f"AMBIGUOUS - {len([c for c in self.candidates if c.credible])} credible claimants"
        return "RESOLVED"

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "requested_chain": self.requested_chain,
            "verdict": self.verdict(),
            "unambiguous": self.unambiguous,
            "error": self.error,
            "candidates": [
                {
                    "token_address": c.token_address, "symbol": c.symbol, "name": c.name,
                    "chain": c.chain, "liquidity_usd": c.liquidity_usd,
                    "volume_24h_usd": c.volume_24h_usd, "price_usd": c.price_usd,
                    "fdv_usd": c.fdv_usd, "pair_count": c.pair_count,
                }
                for c in self.candidates
            ],
        }


def _fold_pairs(pairs: list[dict], symbol: str, chain: str | None) -> list[Candidate]:
    """Group a search response by mint.

    A token has many pairs; the mint is what identifies it. Liquidity is
    summed across that mint's pairs rather than taken from the deepest one,
    because a token split over three pools is not less liquid than the same
    depth in one.
    """
    wanted = symbol.strip().upper()
    by_mint: dict[str, Candidate] = {}

    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        base = pair.get("baseToken") or {}
        address = base.get("address")
        if not address:
            continue
        # Exact symbol match only. A substring match turns a search for DOGE
        # into a list of every DOGE-suffixed derivative on the chain.
        if str(base.get("symbol") or "").strip().upper() != wanted:
            continue
        pair_chain = str(pair.get("chainId") or "")
        if chain and pair_chain.lower() != chain.lower():
            continue

        existing = by_mint.get(str(address))
        liquidity = _to_float((pair.get("liquidity") or {}).get("usd")) or 0.0
        volume = _to_float((pair.get("volume") or {}).get("h24")) or 0.0

        if existing is None:
            by_mint[str(address)] = Candidate(
                token_address=str(address),
                symbol=str(base.get("symbol") or wanted),
                name=base.get("name"),
                chain=pair_chain,
                liquidity_usd=liquidity,
                volume_24h_usd=volume,
                price_usd=_to_float(pair.get("priceUsd")),
                fdv_usd=_to_float(pair.get("fdv")),
                pair_count=1,
            )
        else:
            existing.liquidity_usd = (existing.liquidity_usd or 0.0) + liquidity
            existing.volume_24h_usd = (existing.volume_24h_usd or 0.0) + volume
            existing.pair_count += 1
            # FDV is a property of the token, not the pair, so any pair that
            # reports it will do - but a pair may omit it.
            if existing.fdv_usd is None:
                existing.fdv_usd = _to_float(pair.get("fdv"))

    ranked = sorted(by_mint.values(), key=lambda c: c.liquidity_usd or 0.0, reverse=True)
    return ranked


async def resolve_symbol(symbol: str, chain: str | None = None, *, limit: int = 5) -> Resolution:
    """Every mint listed under `symbol`, most liquid first."""
    try:
        payload = await _get_json(DEXSCREENER_SEARCH, params={"q": symbol})
    except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
        return Resolution(symbol=symbol, requested_chain=chain, error=f"{type(exc).__name__}: {exc}")

    if not isinstance(payload, dict):
        return Resolution(
            symbol=symbol, requested_chain=chain,
            error="listing source returned no usable response",
        )

    candidates = _fold_pairs(payload.get("pairs") or [], symbol, chain)
    return Resolution(symbol=symbol, requested_chain=chain, candidates=candidates[:limit])


async def resolve_many(symbols: list[str], chain: str | None = None) -> list[Resolution]:
    """Resolved sequentially on purpose - the search endpoint is rate limited,
    and a burst of parallel queries earns a 429 that looks like 'no match'."""
    return [await resolve_symbol(s, chain) for s in symbols]
