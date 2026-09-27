"""How wrong would the cost assumption have to be?

WHY THIS EXISTS

The 97-round-trip verdict was that the champion loses money. But the
losing margin is small and the cost side is not a measurement - it is a
model. `PAPER_FEE_PCT` and `PAPER_SPREAD_PCT` are numbers somebody chose,
and the fill model adds impact and confirmation drift on top of them. The
recorded gross edge is positive; the recorded net edge is negative; the
entire difference is that model.

So the honest question is not "is the strategy good" but "how accurate
does the cost model have to be for that verdict to hold?" This answers
it: the per-leg cost rate at which the record breaks even, and the rate at
which it would clear the validation gate's profit factor.

WHAT THIS IS NOT, AND THE TRAP IT MUST NOT BECOME

It does NOT say what costs really are. It cannot - a paper run has no real
fills to measure, which is the whole point of it being paper.

And the result is not a dial. Finding that the record breaks even at 0.31%
per leg is a statement about how far the assumption would have to be from
the truth, to be checked against what a swap actually costs. It is NEVER
a licence to set PAPER_FEE_PCT to 0.31% because that makes the number
green. Costs are in BEHAVIORAL_SETTINGS precisely so that moving them is
visible and splits the dataset; lowering them to manufacture a pass would
be the purest possible version of the thing this repository's rules exist
to prevent.

COSTS ARE ATTRIBUTED PER POSITION, BOTH LEGS

A round trip pays to get in and to get out. Grouping only the exit legs -
which is what `round_trips` does, because only an exit carries realized
P&L - would count half the cost and halve the apparent sensitivity. So
this walks every filled leg carrying a `position_id`, entry and exit
alike, and attributes both to the position.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app import models
from app.analysis.trade_analytics import _costed_notional

#: The gate's threshold, restated here so the scenario table says
#: something the validation report will agree with.
TARGET_PROFIT_FACTOR = 1.30


@dataclass
class _Position:
    """One position's cost and P&L, both legs folded in."""

    notional_usd: float = 0.0
    cost_usd: float = 0.0
    net_pnl_usd: float = 0.0

    @property
    def gross_pnl_usd(self) -> float:
        """P&L as if execution had been free."""
        return self.net_pnl_usd + self.cost_usd

    def pnl_at(self, rate: float) -> float:
        """What this position would have made at `rate` per leg."""
        return self.gross_pnl_usd - rate * self.notional_usd


@dataclass
class CostScenario:
    cost_rate: float                 # per-leg fraction
    net_pnl_usd: float
    expectancy_usd: float
    profit_factor: float | None


@dataclass
class CostSensitivity:
    positions: int
    #: Notional-weighted per-leg rate actually recorded.
    measured_cost_rate: float | None
    total_notional_usd: float
    total_cost_usd: float
    gross_pnl_usd: float
    net_pnl_usd: float
    #: Per-leg rate at which the record breaks even. None when the record
    #: is gross-negative, where no cost reduction can rescue it - a
    #: distinction worth keeping, because "needs free execution" and
    #: "cannot be fixed by execution at all" are different findings.
    breakeven_cost_rate: float | None
    #: Per-leg rate at which profit factor would reach the gate's 1.30.
    target_pf_cost_rate: float | None
    scenarios: list[CostScenario] = field(default_factory=list)
    #: Filled legs with no recorded cost rate. They are excluded, not
    #: assumed free. CLAUDE.md: unmeasurable is never zero.
    unmeasured_legs: int = 0

    @property
    def gross_edge_per_position(self) -> float | None:
        return (self.gross_pnl_usd / self.positions) if self.positions else None

    @property
    def cost_per_position(self) -> float | None:
        return (self.total_cost_usd / self.positions) if self.positions else None

    @property
    def cost_exceeds_gross_edge(self) -> bool:
        """The headline finding when true: the signal earns, execution eats it."""
        return self.gross_pnl_usd > 0 and self.total_cost_usd > self.gross_pnl_usd


def _profit_factor(positions: list[_Position], rate: float) -> float | None:
    gross_profit = sum(p.pnl_at(rate) for p in positions if p.pnl_at(rate) > 0)
    gross_loss = sum(p.pnl_at(rate) for p in positions if p.pnl_at(rate) < 0)
    if gross_loss:
        return gross_profit / abs(gross_loss)
    return float("inf") if gross_profit else None


def _solve_rate_for_pf(
    positions: list[_Position], target: float, ceiling: float
) -> float | None:
    """Lowest-cost bisection for the rate where profit factor hits `target`.

    Profit factor rises monotonically as cost falls, so bisection is valid
    - but the function is not smooth: positions cross zero one at a time
    as the rate moves, so PF steps rather than glides. Bisection handles
    that; a derivative method would not.

    Returns None when even free execution leaves PF below target, which
    says the shortfall is not an execution problem.
    """
    free = _profit_factor(positions, 0.0)
    if free is None or free < target:
        return None
    if (_profit_factor(positions, ceiling) or 0.0) >= target:
        return ceiling

    lo, hi = 0.0, ceiling
    for _ in range(60):
        mid = (lo + hi) / 2
        pf = _profit_factor(positions, mid)
        if pf is not None and pf >= target:
            lo = mid
        else:
            hi = mid
    return lo


def analyse(trades: list[models.Trade]) -> CostSensitivity:
    """Sensitivity of the recorded record to the per-leg cost rate.

    Only filled legs with BOTH a cost rate and a usable notional are
    included; a leg missing either is counted as unmeasured rather than
    treated as free.
    """
    by_position: dict[int, _Position] = {}
    unmeasured = 0

    for t in trades:
        if t.status != models.TradeStatus.FILLED.value:
            continue
        if t.position_id is None:
            unmeasured += 1
            continue
        if t.execution_cost_pct is None:
            unmeasured += 1
            continue
        notional = _costed_notional(t)
        if not notional:
            unmeasured += 1
            continue

        pos = by_position.setdefault(t.position_id, _Position())
        pos.notional_usd += notional
        pos.cost_usd += t.execution_cost_pct * notional
        if t.pnl_usd is not None:
            pos.net_pnl_usd += t.pnl_usd

    # Only completed positions can be scored: one with no realized P&L is
    # still open, and folding its entry cost in without its outcome would
    # make execution look more expensive than the finished record shows.
    positions = [p for p in by_position.values() if p.net_pnl_usd != 0.0]

    total_notional = sum(p.notional_usd for p in positions)
    total_cost = sum(p.cost_usd for p in positions)
    net = sum(p.net_pnl_usd for p in positions)
    gross = sum(p.gross_pnl_usd for p in positions)

    measured_rate = (total_cost / total_notional) if total_notional else None

    breakeven = None
    if gross > 0 and total_notional:
        breakeven = gross / total_notional

    ceiling = max(measured_rate or 0.0, 0.02)
    target_rate = (
        _solve_rate_for_pf(positions, TARGET_PROFIT_FACTOR, ceiling)
        if positions else None
    )

    scenarios: list[CostScenario] = []
    if positions:
        rates = {0.0, 0.001, 0.002, 0.003, 0.004, 0.005}
        if measured_rate is not None:
            rates.add(round(measured_rate, 5))
        if breakeven is not None:
            rates.add(round(breakeven, 5))
        for rate in sorted(rates):
            pnls = [p.pnl_at(rate) for p in positions]
            total = sum(pnls)
            scenarios.append(CostScenario(
                cost_rate=rate,
                net_pnl_usd=total,
                expectancy_usd=total / len(positions),
                profit_factor=_profit_factor(positions, rate),
            ))

    return CostSensitivity(
        positions=len(positions),
        measured_cost_rate=measured_rate,
        total_notional_usd=total_notional,
        total_cost_usd=total_cost,
        gross_pnl_usd=gross,
        net_pnl_usd=net,
        breakeven_cost_rate=breakeven,
        target_pf_cost_rate=target_rate,
        scenarios=scenarios,
        unmeasured_legs=unmeasured,
    )
