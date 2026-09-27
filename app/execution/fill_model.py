"""Realistic fill simulation for paper trading.

The old paper engine applied one flat 0.5% buffer to every trade, which
made paper results systematically optimistic in the way that matters most
for memecoins: it charged a $20 trade and a $20,000 trade the SAME cost
against the same thin pool. In reality the second one moves the price
against itself enormously. A strategy can look profitable on paper purely
because the simulator never charged it for the size it was taking.

Five effects are modeled here, all derived from data the bot already
fetches (DexScreener reports pool liquidity, 24h volume, and price change
per pair):

1. PRICE IMPACT — the big one, and the one that was missing entirely.
   For a constant-product AMM (x*y=k), buying with `d` quote-units against
   a quote-side reserve `R` gives:

       tokens_out      = R_t * d / (R + d)
       effective_price = d / tokens_out = (R + d) / R_t
       spot_price      = R / R_t
       effective/spot  = 1 + d/R

   so impact is simply `trade_usd / quote_side_reserve`. DexScreener's
   `liquidity.usd` reports the TOTAL pool value across both sides, so the
   quote side is about half of it, giving `2 * trade_usd / liquidity_usd`.
   This is a real derivation rather than a fudge factor, and it is why a
   trade sized at 5% of a pool costs ~10% in impact.

2. SPREAD — the bid/ask gap crossed just to transact.

3. CONFIRMATION DELAY — a swap is not instant. Between signing and
   inclusion the price keeps moving, and on a volatile memecoin that drift
   can dwarf the spread. Scaled from the token's own recent volatility
   (1h price change) by sqrt(time), the standard random-walk scaling, so a
   quiet token gets little drift and a violent one gets a lot.

4. FAILED FILLS — real DEX swaps carry a slippage tolerance and REVERT
   when the market moves past it. `slippage_bps` was previously accepted by
   the paper engine and then completely ignored, so paper trading never
   experienced a failed fill at all. Now, when impact + adverse drift
   exceeds the tolerance, the fill fails exactly as it would on-chain -
   which is itself an important cost, since a strategy that only fills in
   calm conditions has very different economics from one that always fills.

5. FEES — charged per side (app/config.py PAPER_FEE_PCT).

Randomness is injected through an explicit `rng`, so production is
genuinely stochastic while tests stay reproducible by passing a seeded
Random.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from app.config import settings

# A pool whose liquidity we don't know is treated as this deep. Deliberately
# modest rather than generous: an unknown pool should not be assumed to be
# a forgiving one. The rug filter already rejects genuinely thin pools, so
# anything reaching here has cleared MIN_LIQUIDITY_USD.
ASSUMED_LIQUIDITY_USD = 50_000.0

# Fallback 1h volatility when the market snapshot doesn't report one.
# Memecoins are violent; assuming calm would understate delay drift.
ASSUMED_1H_VOLATILITY_PCT = 15.0

# Ceiling on modeled impact. Beyond this the constant-product formula is
# still "correct" but the trade would never realistically be attempted, and
# an unbounded number produces absurd fills that hide bugs rather than
# expose them.
MAX_IMPACT_PCT = 0.50


@dataclass
class FillOutcome:
    filled: bool
    fill_price: float = 0.0
    impact_pct: float = 0.0
    drift_pct: float = 0.0
    spread_pct: float = 0.0
    fee_pct: float = 0.0
    total_cost_pct: float = 0.0
    delay_seconds: float = 0.0
    failure_reason: str = ""


def price_impact_pct(trade_usd: float, liquidity_usd: float | None) -> float:
    """Constant-product price impact: trade_usd / quote-side reserve.

    See the module docstring for the derivation. `liquidity_usd` is the
    pool's total value across both sides, so the quote side is half.
    """
    pool = liquidity_usd if (liquidity_usd and liquidity_usd > 0) else ASSUMED_LIQUIDITY_USD
    quote_side = pool / 2
    if quote_side <= 0:
        return MAX_IMPACT_PCT
    return min(trade_usd / quote_side, MAX_IMPACT_PCT)


def confirmation_delay_seconds(rng: random.Random) -> float:
    """How long between submitting the swap and it landing on-chain."""
    low = settings.PAPER_MIN_CONFIRM_SECONDS
    high = max(settings.PAPER_MAX_CONFIRM_SECONDS, low)
    return rng.uniform(low, high)


def delay_drift_pct(volatility_1h_pct: float | None, delay_seconds: float, rng: random.Random) -> float:
    """Price movement during confirmation, as a SIGNED fraction.

    Scales the token's own 1h volatility down to the delay window by
    sqrt(t), the standard random-walk scaling: a move over 1 hour of v
    implies a move over t seconds of v * sqrt(t/3600).
    """
    vol_pct = abs(volatility_1h_pct) if volatility_1h_pct is not None else ASSUMED_1H_VOLATILITY_PCT
    hourly = vol_pct / 100.0
    scaled = hourly * math.sqrt(max(delay_seconds, 0.0) / 3600.0)
    return rng.gauss(0.0, scaled)


def simulate_fill(
    *,
    side: str,
    reference_price: float,
    trade_usd: float,
    liquidity_usd: float | None = None,
    volatility_1h_pct: float | None = None,
    slippage_bps: int | None = None,
    rng: random.Random | None = None,
) -> FillOutcome:
    """Simulate one swap leg. `side` is "buy" or "sell".

    Returns a FillOutcome whose `filled` is False when the modeled adverse
    move exceeds the slippage tolerance - a real, common outcome that the
    previous simulator could never produce.
    """
    rng = rng or random.Random()
    slippage_bps = settings.SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    tolerance = slippage_bps / 10_000.0

    impact = price_impact_pct(trade_usd, liquidity_usd)
    spread = settings.PAPER_SPREAD_PCT
    fee = settings.PAPER_FEE_PCT
    delay = confirmation_delay_seconds(rng)
    drift = delay_drift_pct(volatility_1h_pct, delay, rng)

    # Drift helps or hurts depending on direction: a buy is hurt by the
    # price rising during confirmation, a sell by it falling.
    adverse_drift = drift if side == "buy" else -drift

    # What the on-chain slippage check actually tests: how far the
    # execution price moved against you versus the quote. Fees are a known
    # cost taken by the protocol, not slippage, so they are excluded here.
    adverse_move = impact + adverse_drift
    if adverse_move > tolerance:
        return FillOutcome(
            filled=False,
            impact_pct=impact,
            drift_pct=adverse_drift,
            spread_pct=spread,
            fee_pct=fee,
            delay_seconds=delay,
            failure_reason=(
                f"slippage tolerance exceeded: price moved {adverse_move * 100:.2f}% against the trade "
                f"({impact * 100:.2f}% price impact + {adverse_drift * 100:+.2f}% drift over {delay:.1f}s) "
                f"vs {slippage_bps}bps tolerance"
            ),
        )

    total_cost = impact + spread + fee + adverse_drift
    if side == "buy":
        fill_price = reference_price * (1 + total_cost)
    else:
        fill_price = max(reference_price * (1 - total_cost), 0.0)

    return FillOutcome(
        filled=True,
        fill_price=fill_price,
        impact_pct=impact,
        drift_pct=adverse_drift,
        spread_pct=spread,
        fee_pct=fee,
        total_cost_pct=total_cost,
        delay_seconds=delay,
    )
