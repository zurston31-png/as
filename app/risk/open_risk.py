"""Worst-case risk sitting in the open book, and whether the day's
remaining loss budget can absorb it.

THE QUESTION THIS ANSWERS

The daily-loss limit is a statement about how much the account may lose
today. Checking it only against what has already been lost answers a
question about the past. The question that actually protects capital is:

    if every stop in the book were hit right now, would the day's loss
    limit be breached?

If the answer is yes, opening another position adds to a loss that is
already committed. The bot must decline - which is the invariant this
module exists to make enforceable.

THE ONE THING THAT IS EASY TO GET WRONG

A position bought at $1.00, now at $0.90, with a stop at $0.85, has:

    already lost   $0.10 per unit  - unrealized, and ALREADY inside the
                                     day's equity drawdown
    still at risk  $0.05 per unit  - from here down to the stop

The total exposure from entry is $0.15, but adding $0.15 to a drawdown
that already contains $0.10 counts that dime twice and would halt the bot
on a loss half again as large as the one it is really facing.

So `loss_if_all_stops_hit_usd` is measured FROM THE CURRENT MARK, not from
entry. `current_open_drawdown_usd` reports the entry-to-mark part
separately, and the two are disjoint by construction:

    entry -> mark   is in current_open_drawdown_usd  (and in the daily drawdown)
    mark  -> stop   is in loss_if_all_stops_hit_usd  (and is not, yet)

A position already trading BELOW its stop contributes zero further stop
risk: that loss has happened, it is in the drawdown, and the exit is
pending rather than prospective.

BASE VERSUS STRESS

  Base   - every stop fills exactly at its stop price. Optimistic, and
           knowingly so: it is the floor, not a forecast.

  Stress - every stop fills at the stop price minus the DETERMINISTIC part
           of the same fill model that would price the real exit
           (app/execution/fill_model.py): constant-product price impact
           for the exit notional against the pool, plus spread, plus fee.
           Impact is computed against the LOWEST pool depth actually
           recorded for that position (`Position.lowest_liquidity_usd`),
           which is a measurement rather than an assumption.

The random drift term is deliberately excluded from the stress case.
It is a zero-mean draw, and picking one sample of it and calling the
result a worst case would be inventing a number.

WHAT STRESS DOES NOT COVER, AND WHY IT IS NOT SET TO ZERO

Three real ways a stop-out can be worse than the stress figure:

  * the price GAPS through the stop rather than trading down to it
  * liquidity is pulled entirely and the exit does not happen at all
  * several positions stop out at once into the same falling market

None of them can be quantified from what this bot has recorded - it has
no observed gap distribution and no recorded rug event to measure. Per
CLAUDE.md they are reported as unquantified rather than modeled as zero,
via `unmodeled_risks`. `stress_loss_if_all_stops_hit_usd` is a floor on
the bad case, not a ceiling, and nothing here should be read as one.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.execution.fill_model import price_impact_pct
from app.services import price_feed

logger = logging.getLogger(__name__)

# Named so a reader of `unmodeled_risks` knows these are acknowledged
# gaps in the stress figure, not oversights.
UNMODELED_RISKS = (
    "a gap through the stop rather than a trade down to it",
    "a liquidity pull that prevents the exit entirely",
    "correlated stop-outs into the same falling market",
)


@dataclass(frozen=True)
class OpenRisk:
    """Worst-case loss still available to the open book."""

    positions: int

    # Entry -> current mark. Already inside today's equity drawdown.
    current_open_drawdown_usd: float
    # Current mark -> stop. Not yet realized, and disjoint from the above.
    loss_if_all_stops_hit_usd: float
    # As above, plus the deterministic exit cost of getting out.
    stress_loss_if_all_stops_hit_usd: float

    # How much of today's loss budget survives the stress case.
    remaining_daily_loss_budget_usd: float
    remaining_daily_loss_budget_after_open_risk_usd: float

    # Measurement quality. A number computed partly from fallbacks is not
    # the same number as one computed from recorded data.
    unpriced_positions: int
    positions_without_recorded_liquidity: int
    unmodeled_risks: tuple[str, ...] = field(default=UNMODELED_RISKS)

    @property
    def measurable(self) -> bool:
        return self.unpriced_positions == 0

    def as_dict(self) -> dict:
        return asdict(self)


async def _mark(position: models.Position) -> float | None:
    if not position.token_address:
        return None
    try:
        price = await price_feed.get_price_usd(position.token_address)
    except Exception:
        logger.warning("price lookup failed for %s during open-risk assessment",
                       position.symbol, exc_info=True)
        return None
    return price if price and price > 0 else None


def _exit_cost_pct(notional_usd: float, liquidity_usd: float | None) -> float:
    """The deterministic share of the fill model's exit cost.

    Reuses `price_impact_pct` rather than restating the constant-product
    formula so this can never drift away from the model that prices the
    real exit. Drift is excluded - see the module docstring.
    """
    return (
        price_impact_pct(notional_usd, liquidity_usd)
        + settings.PAPER_SPREAD_PCT
        + settings.PAPER_FEE_PCT
    )


async def assess(
    db: Session,
    *,
    remaining_daily_loss_budget_usd: float,
    prospective_size_usd: float = 0.0,
    prospective_stop_loss_pct: float | None = None,
    prospective_liquidity_usd: float | None = None,
) -> OpenRisk:
    """Worst-case loss still ahead of the open book.

    `remaining_daily_loss_budget_usd` comes from the daily-loss assessment
    and is passed in rather than recomputed, so the two can never disagree
    about how much of the day is left.

    A prospective position is included when `prospective_size_usd` is
    given: a trade that has not opened yet still gets to be counted
    against the budget it would consume, which is the only order in which
    the check can prevent anything.
    """
    positions = (
        db.query(models.Position)
        .filter(models.Position.status == models.PositionStatus.OPEN.value)
        .all()
    )

    open_drawdown = 0.0
    to_stop = 0.0
    to_stop_stressed = 0.0
    unpriced = 0
    without_liquidity = 0

    for pos in positions:
        mark = await _mark(pos)
        if mark is None:
            # Valued at cost: no measurable drawdown, and the stop
            # distance is measured from entry because that is the only
            # price available. Counted as unpriced so the caller knows
            # this figure rests on a stale mark.
            unpriced += 1
            mark = pos.entry_price

        open_drawdown += max(pos.entry_price - mark, 0.0) * pos.qty

        # Zero once the mark is already at or below the stop: that loss
        # has happened and lives in the drawdown above.
        per_unit_to_stop = max(mark - pos.stop_loss, 0.0)
        to_stop += per_unit_to_stop * pos.qty

        # Lowest depth actually seen beats depth at entry: a pool that has
        # thinned is the pool the exit will really hit.
        liquidity = pos.lowest_liquidity_usd or pos.liquidity_at_entry_usd
        if not liquidity:
            without_liquidity += 1
        exit_notional = pos.stop_loss * pos.qty
        to_stop_stressed += (
            per_unit_to_stop * pos.qty + exit_notional * _exit_cost_pct(exit_notional, liquidity)
        )

    if prospective_size_usd > 0:
        stop_pct = (
            prospective_stop_loss_pct
            if prospective_stop_loss_pct is not None
            else settings.STOP_LOSS_PCT
        )
        # A position about to be opened has no drawdown yet, so its whole
        # stop distance is still ahead of it - notional * stop_pct, which
        # is exactly the risk the sizing formula intended to take.
        prospective_to_stop = prospective_size_usd * stop_pct
        remaining_notional = prospective_size_usd * (1 - stop_pct)
        to_stop += prospective_to_stop
        to_stop_stressed += prospective_to_stop + remaining_notional * _exit_cost_pct(
            remaining_notional, prospective_liquidity_usd
        )
        if not prospective_liquidity_usd:
            without_liquidity += 1

    return OpenRisk(
        positions=len(positions),
        current_open_drawdown_usd=open_drawdown,
        loss_if_all_stops_hit_usd=to_stop,
        stress_loss_if_all_stops_hit_usd=to_stop_stressed,
        remaining_daily_loss_budget_usd=remaining_daily_loss_budget_usd,
        remaining_daily_loss_budget_after_open_risk_usd=(
            remaining_daily_loss_budget_usd - to_stop_stressed
        ),
        unpriced_positions=unpriced,
        positions_without_recorded_liquidity=without_liquidity,
    )
