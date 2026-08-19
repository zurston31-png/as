"""Stale-data rejection.

Stale data is more dangerous than missing data, because it looks
authoritative. A missing price makes the bot fail closed and skip the
trade; a price from twenty minutes ago sails through every check and gets
used to size a position, set a stop, and decide an exit — all against a
market that has since moved. On a memecoin, twenty minutes is a different
asset.

Two separate notions of "old" are checked, and conflating them was the
trap this module exists to avoid:

  OBSERVATION AGE   how long ago the bot fetched this data. Always
                    knowable, since the bot stamps it on arrival.
  REPORTED AGE      how long ago the SOURCE last saw a trade. Only
                    inferable, and only sometimes - a pool with no volume
                    in the last hour is reporting a real price that simply
                    hasn't moved, which is not the same as a broken feed.

The first is a hard rejection. The second is a warning: a quiet pool is a
market-quality problem (already scored in app/signals/market_quality.py),
not a data-integrity one, and treating it as corruption would reject every
genuinely calm token.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from app.config import settings
from app.services.price_feed import MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class FreshnessVerdict:
    fresh: bool
    reason: str = ""
    observation_age_seconds: float = 0.0


def check_snapshot_freshness(
    snapshot: MarketSnapshot | None, *, now: dt.datetime | None = None
) -> FreshnessVerdict:
    """Is this market observation fresh enough to trade on?

    None is treated as not-fresh with an explicit reason rather than
    raising, so every caller handles the missing and stale cases through
    one code path.
    """
    if snapshot is None:
        return FreshnessVerdict(False, "no market snapshot available")

    age = snapshot.age_seconds(now)
    limit = settings.MAX_MARKET_DATA_AGE_SECONDS
    if age > limit:
        return FreshnessVerdict(
            False,
            f"market data is {age:.0f}s old, past the {limit:.0f}s freshness limit - "
            "refusing to trade on a stale observation",
            observation_age_seconds=age,
        )

    return FreshnessVerdict(True, "market data is fresh", observation_age_seconds=age)


def check_price_sanity(price: float | None, previous_price: float | None = None) -> FreshnessVerdict:
    """Reject prices that cannot be real, and flag implausible jumps.

    A non-positive price is always corrupt. A jump beyond
    MAX_PRICE_JUMP_FACTOR versus the last observation is *probably* a bad
    tick (a decimals bug, a thin-pool print, a feed glitch) rather than a
    genuine move - and acting on it would mean sizing or exiting against a
    number that never existed. Memecoins do genuinely move violently, so
    the default factor is deliberately generous; this catches broken data,
    not volatility.
    """
    if price is None:
        return FreshnessVerdict(False, "no price available")
    if price <= 0:
        return FreshnessVerdict(False, f"price {price} is not positive - corrupt data")

    if previous_price is not None and previous_price > 0:
        factor = max(price / previous_price, previous_price / price)
        if factor > settings.MAX_PRICE_JUMP_FACTOR:
            return FreshnessVerdict(
                False,
                f"price moved {factor:.1f}x versus the previous observation "
                f"({previous_price:.10g} -> {price:.10g}), beyond the "
                f"{settings.MAX_PRICE_JUMP_FACTOR:.0f}x plausibility limit - treating as a bad tick",
            )

    return FreshnessVerdict(True, "price looks sane")
