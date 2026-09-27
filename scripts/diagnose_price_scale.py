#!/usr/bin/env python3
"""Prove - or disprove - that the candle feed is returning the wrong token.

WHAT THIS IS FOR

The shadow resolver produced exit prices in the hundreds and thousands for
tokens whose entry prices were small fractions of a cent, clustered tightly
per symbol. That pattern is not a series of outliers; it is the shape of a
feed returning a DIFFERENT asset's chart. The suspected cause is in
app/data/live_provider.py: GeckoTerminal's OHLCV endpoint is keyed by POOL,
a pool has two tokens, and asked without a `token` parameter it answers for
the pool's BASE token. Any token sitting on the QUOTE side of its own
deepest pool therefore gets its counterparty's price history - well-formed
OHLCV, right length, right window, wrong asset.

This script does not assume that. It measures it, against the live APIs,
using a second independent source (DexScreener, via app/services/price_feed)
as the referee:

    side          which side of its deepest pool the token is on
    dexscreener   spot USD price, the second source
    default       last close from the call the bot makes TODAY
    pinned        last close when `token=<address>` is passed
    verdict       which of those two agrees with the referee

A token whose `default` disagrees with DexScreener by orders of magnitude
while `pinned` agrees is a confirmed instance. A token where both agree is
a base-side token, unaffected. A token where NEITHER agrees is something
else and should not be blamed on this.

It is READ-ONLY. It opens the database only to choose which tokens to ask
about, writes nothing, and changes no setting - the `token=` call is made
directly here regardless of GECKOTERMINAL_PIN_TOKEN_SIDE, precisely so the
flag can be evaluated before it is switched on.

Usage:
    # the tokens the shadow system actually resolved, worst first
    python scripts/diagnose_price_scale.py

    # or specific addresses
    python scripts/diagnose_price_scale.py <address> [<address> ...]

    # how many tokens to sample from the database (default 12)
    python scripts/diagnose_price_scale.py --limit 25

Needs outbound HTTPS to api.geckoterminal.com and api.dexscreener.com, so
run it on the deployment host, not in a sandbox.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import models  # noqa: E402
from app.config import settings  # noqa: E402
from app.data.candles import Timeframe  # noqa: E402
from app.data.live_provider import (  # noqa: E402
    CHAIN_TO_GECKOTERMINAL_NETWORK,
    _find_primary_pool,
    _get_json,
    _parse_ohlcv_response,
)
from app.database import SessionLocal  # noqa: E402
from app.services import price_feed  # noqa: E402

# Order-of-magnitude, not a tuning knob. Below this the two sources merely
# differ (a stale bar, a thin pool, a different venue); at or above it they
# are not describing the same asset.
WRONG_ASSET_RATIO = 50.0


async def _last_close(network: str, pool: str, *, token: str | None) -> float | None:
    params = {"aggregate": 15, "limit": 5, "currency": "usd"}
    if token is not None:
        params["token"] = token
    data = await _get_json(
        f"{settings.GECKOTERMINAL_API_BASE}/networks/{network}/pools/{pool}/ohlcv/minute",
        params=params,
    )
    if not data:
        return None
    series = _parse_ohlcv_response(data, "probe", Timeframe.M15)
    if not series or not len(series):
        return None
    return series[-1].close


def _ratio(a: float | None, b: float | None) -> float | None:
    if not a or not b or a <= 0 or b <= 0:
        return None
    low, high = sorted((a, b))
    return high / low


def _agrees(a: float | None, b: float | None) -> bool | None:
    r = _ratio(a, b)
    return None if r is None else r < WRONG_ASSET_RATIO


async def inspect(chain: str, address: str, symbol: str) -> dict:
    out = {"symbol": symbol, "address": address, "chain": chain}
    network = CHAIN_TO_GECKOTERMINAL_NETWORK.get(chain.lower())
    if network is None:
        out["verdict"] = f"UNMEASURED: no GeckoTerminal network for chain {chain!r}"
        return out

    pool, side = await _find_primary_pool(network, address)
    out["pool"] = pool
    out["side"] = side
    if not pool:
        out["verdict"] = "UNMEASURED: no pool found"
        return out

    out["dexscreener"] = await price_feed.get_price_usd(address)
    out["default"] = await _last_close(network, pool, token=None)
    out["pinned"] = await _last_close(network, pool, token=address)

    referee = out["dexscreener"]
    if referee is None:
        out["verdict"] = "UNMEASURED: no second source to check against"
        return out

    default_ok = _agrees(out["default"], referee)
    pinned_ok = _agrees(out["pinned"], referee)

    if default_ok is None and pinned_ok is None:
        out["verdict"] = "UNMEASURED: no candles either way"
    elif default_ok is False and pinned_ok is True:
        out["verdict"] = (
            f"CONFIRMED WRONG ASSET: today's call is off by "
            f"{_ratio(out['default'], referee):,.0f}x; pinning the token fixes it"
        )
    elif default_ok is True:
        out["verdict"] = "OK: today's call already returns this token"
    elif pinned_ok is False:
        out["verdict"] = (
            "DISAGREES BOTH WAYS: neither call matches DexScreener, so this is "
            "not the pool-side bug - investigate separately"
        )
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


def _tokens_from_db(limit: int) -> list[tuple[str, str, str]]:
    db = SessionLocal()
    try:
        chains = {
            row.token_address: row.chain
            for row in db.query(models.ShadowDecision.token_address, models.ShadowDecision.chain).all()
        }
        rows = (
            db.query(models.ShadowPosition)
            .filter(models.ShadowPosition.return_pct.isnot(None))
            .all()
        )
        # Worst first: the biggest recorded returns are where a wrong-asset
        # price does the most visible damage.
        rows.sort(key=lambda r: abs(r.return_pct or 0), reverse=True)
        seen: dict[str, tuple[str, str, str]] = {}
        for row in rows:
            if row.token_address in seen:
                continue
            seen[row.token_address] = (
                chains.get(row.token_address) or settings.CHAIN,
                row.token_address,
                row.symbol or "?",
            )
            if len(seen) >= limit:
                break
        return list(seen.values())
    finally:
        db.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("addresses", nargs="*", help="token addresses to check")
    parser.add_argument("--chain", default=settings.CHAIN, help="chain for addresses given on the command line")
    parser.add_argument("--limit", type=int, default=12, help="how many tokens to sample from the database")
    args = parser.parse_args()

    if args.addresses:
        targets = [(args.chain, a, "?") for a in args.addresses]
    else:
        targets = _tokens_from_db(args.limit)

    if not targets:
        print("nothing to check: no resolved shadow positions and no addresses given")
        return 1

    print(f"checking {len(targets)} token(s) against GeckoTerminal and DexScreener\n")
    results = []
    for chain, address, symbol in targets:
        result = await inspect(chain, address, symbol)
        results.append(result)

        def fmt(v):
            return "-" if v is None else f"{v:.10g}"

        print(f"{symbol:>10}  {address}")
        print(f"            pool={result.get('pool') or '-'}  side={result.get('side') or 'unknown'}")
        print(f"            dexscreener={fmt(result.get('dexscreener'))}  "
              f"default={fmt(result.get('default'))}  pinned={fmt(result.get('pinned'))}")
        print(f"            {result['verdict']}")
        print()

    confirmed = [r for r in results if r["verdict"].startswith("CONFIRMED")]
    ok = [r for r in results if r["verdict"].startswith("OK")]
    unmeasured = [r for r in results if "UNMEASURED" in r["verdict"]]
    print("-" * 70)
    print(f"confirmed wrong asset : {len(confirmed)}")
    print(f"already correct       : {len(ok)}")
    print(f"unmeasured            : {len(unmeasured)}")
    print(f"other                 : {len(results) - len(confirmed) - len(ok) - len(unmeasured)}")
    print()
    if confirmed:
        print("Every CONFIRMED token above was scored, and had its shadow exits walked,")
        print("against another asset's chart. Those observations are not recoverable -")
        print("the right candles were never fetched, so there is nothing to recompute")
        print("from. Setting GECKOTERMINAL_PIN_TOKEN_SIDE=true corrects the feed going")
        print("forward and starts a new collection run; it does not repair the old one.")
    else:
        print("No token here shows the pool-side defect. Do not set")
        print("GECKOTERMINAL_PIN_TOKEN_SIDE on the strength of this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
