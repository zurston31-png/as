"""Turn a chart symbol into the mint address that symbol actually refers to.

This exists because of the rule in `app/identity.py`: the mint is the token
and the symbol is only a label. Anyone can mint a token called BONK, and
copycats are deliberate rather than accidental - sharing a symbol with
something people trust is the entire point of one.

WHY THIS DOES NOT RANK BY LIQUIDITY
-----------------------------------
The first version of this module did, and it recommended scam tokens. A
real run for WIF returned three pools each claiming over a billion dollars
of liquidity against four dollars of daily volume, while the genuine mint -
millions in liquidity, over a million in daily volume, spread across
seventeen pools - was ranked last and then truncated out of the results
entirely.

Reported liquidity is a self-declared number attached to a pool that anyone
can create. Volume is what someone had to actually do. So a pool holding a
fortune that nobody trades is not a deep market, it is a claim with nothing
behind it, and this ranks on traded volume with an explicit turnover test
to throw those claims out.

Nothing here decides anything. It prints candidates for a human to choose
from, because choosing wrong means the bot screens one token and trades
another. None of it is on the trading path: no scoring, threshold, weight
or classifier used for a trade decision reads any of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.scanner.discovery import (
    DEXSCREENER_SEARCH,
    DEXSCREENER_TOKENS,
    _get_json,
    _to_float,
)

# A pool smaller than this cannot absorb a position, whoever minted it.
MIN_CREDIBLE_LIQUIDITY_USD = 25_000.0

# A token nobody trades is not the one a chart is tracking, however much its
# pool claims to hold.
MIN_CREDIBLE_VOLUME_24H_USD = 10_000.0

# Share of the pool that must change hands in a day for the reported
# liquidity to be believable. The fabricated WIF pools scored 3e-9 against
# this - four dollars traded against a billion claimed. Real memecoin pools
# turn over multiples of themselves daily, so this floor is deliberately far
# below anything genuine rather than tuned to sit near it.
MIN_PLAUSIBLE_TURNOVER = 0.001

# The top candidate is only unambiguous if it dwarfs the runner-up on
# volume. Two mints with comparable real trading under one symbol is exactly
# the situation where picking the larger is a coin flip with money behind it.
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
    def turnover(self) -> float | None:
        """24h volume as a fraction of liquidity. None if unmeasurable."""
        liquidity = self.liquidity_usd or 0.0
        if liquidity <= 0:
            return None
        return (self.volume_24h_usd or 0.0) / liquidity

    @property
    def deep_enough(self) -> bool:
        return (self.liquidity_usd or 0.0) >= MIN_CREDIBLE_LIQUIDITY_USD

    @property
    def traded(self) -> bool:
        return (self.volume_24h_usd or 0.0) >= MIN_CREDIBLE_VOLUME_24H_USD

    @property
    def dead_pool(self) -> bool:
        """Claims real depth, but nothing trades against it.

        This is the copycat's signature: a pool minted with a headline
        number and no market. Reported separately from a merely small
        token, because the two need different reactions - one is a fake,
        the other is just illiquid.
        """
        turnover = self.turnover
        return self.deep_enough and turnover is not None and turnover < MIN_PLAUSIBLE_TURNOVER

    @property
    def live(self) -> bool:
        """Deep enough to trade, actually traded, and plausibly so."""
        return self.deep_enough and self.traded and not self.dead_pool

    def why_not_live(self) -> str:
        if self.dead_pool:
            return (
                f"pool claims ${self.liquidity_usd or 0:,.0f} but only "
                f"${self.volume_24h_usd or 0:,.0f} traded in 24h - not a real market"
            )
        if not self.deep_enough:
            return f"only ${self.liquidity_usd or 0:,.0f} of liquidity"
        if not self.traded:
            return f"only ${self.volume_24h_usd or 0:,.0f} of 24h volume"
        return ""


@dataclass
class Resolution:
    symbol: str
    requested_chain: str | None
    candidates: list[Candidate] = field(default_factory=list)
    error: str | None = None

    @property
    def live_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.live]

    @property
    def best(self) -> Candidate | None:
        live = self.live_candidates
        return live[0] if live else None

    @property
    def unambiguous(self) -> bool:
        """One live claimant, clear of the next by DOMINANCE_RATIO on volume."""
        live = self.live_candidates
        if not live:
            return False
        if len(live) == 1:
            return True
        top = live[0].volume_24h_usd or 0.0
        runner_up = live[1].volume_24h_usd or 0.0
        if runner_up <= 0:
            return True
        return top / runner_up >= DOMINANCE_RATIO

    def verdict(self) -> str:
        if self.error:
            return f"UNRESOLVED - {self.error}"
        if not self.candidates:
            return "NO MATCH - nothing listed under this symbol"
        live = self.live_candidates
        if not live:
            fakes = sum(1 for c in self.candidates if c.dead_pool)
            if fakes:
                return (
                    f"NO REAL MARKET - {len(self.candidates)} claimant(s), "
                    f"{fakes} of them pools with no trading"
                )
            return "NO REAL MARKET - nothing under this symbol is both deep enough and traded"
        if not self.unambiguous:
            return f"AMBIGUOUS - {len(live)} genuinely traded claimants"
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
                    "turnover": c.turnover, "live": c.live, "dead_pool": c.dead_pool,
                }
                for c in self.candidates
            ],
        }


def _fold_pairs(pairs: list[dict], symbol: str | None, chain: str | None) -> list[Candidate]:
    """Group a listing response by mint.

    A token has many pairs; the mint is what identifies it. Liquidity and
    volume are summed across that mint's pairs rather than taken from the
    deepest one, because a token split over three pools is not less liquid
    than the same depth in one.

    `symbol=None` accepts every symbol, which is what an address lookup
    wants - there the mint is already the question, not the answer.
    """
    wanted = symbol.strip().upper() if symbol else None
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
        if wanted is not None and str(base.get("symbol") or "").strip().upper() != wanted:
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
                symbol=str(base.get("symbol") or wanted or "?"),
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

    return _rank(list(by_mint.values()))


def _rank(candidates: list[Candidate]) -> list[Candidate]:
    """Genuinely traded mints first, by volume; everything else after.

    Ranking on volume rather than liquidity is the whole point - see the
    module docstring. Sorting live candidates ahead of the rest also means a
    caller that truncates the list keeps the real answers and drops the
    fakes, rather than the other way round.
    """
    return sorted(
        candidates,
        key=lambda c: (c.live, c.volume_24h_usd or 0.0, c.liquidity_usd or 0.0),
        reverse=True,
    )


async def resolve_symbol(symbol: str, chain: str | None = None, *, limit: int = 8) -> Resolution:
    """Every mint listed under `symbol`, genuinely traded ones first.

    `limit` is applied after ranking, never before: truncating an unranked
    list is how the real WIF mint got dropped in favour of three fabricated
    billion-dollar pools.
    """
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
    # Never truncate away a live candidate: an ambiguity the human needs to
    # see must not be hidden by a display cap.
    live = [c for c in candidates if c.live]
    rest = [c for c in candidates if not c.live]
    return Resolution(
        symbol=symbol, requested_chain=chain,
        candidates=live + rest[: max(0, limit - len(live))],
    )


async def resolve_many(symbols: list[str], chain: str | None = None) -> list[Resolution]:
    """Resolved sequentially on purpose - the search endpoint is rate limited,
    and a burst of parallel queries earns a 429 that looks like 'no match'."""
    return [await resolve_symbol(s, chain) for s in symbols]


async def describe_address(address: str, chain: str | None = None) -> Resolution:
    """What is this mint?

    The reverse of `resolve_symbol`, and the reliable direction. Searching by
    symbol depends on the listing source returning the right token among
    however many copycats share the name; asking about a specific mint does
    not. When a symbol search comes back AMBIGUOUS or misses the token
    entirely, this is how an address in hand gets confirmed.
    """
    try:
        payload = await _get_json(DEXSCREENER_TOKENS.format(addresses=address))
    except Exception as exc:  # noqa: BLE001
        return Resolution(symbol=address, requested_chain=chain, error=f"{type(exc).__name__}: {exc}")

    if not isinstance(payload, dict):
        return Resolution(
            symbol=address, requested_chain=chain,
            error="listing source returned no usable response",
        )

    # symbol=None: the address is the question here, so whatever symbol it
    # turns out to carry is the answer rather than a filter.
    candidates = _fold_pairs(payload.get("pairs") or [], None, chain)
    exact = [c for c in candidates if c.token_address.lower() == address.lower()]
    return Resolution(symbol=address, requested_chain=chain, candidates=exact)
