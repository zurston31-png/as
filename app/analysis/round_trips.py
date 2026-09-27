"""Group exit legs into the positions they actually belong to.

The unit problem this module exists to fix
------------------------------------------

`Trade` rows are LEGS, not trades. A position that takes a partial profit
and is later closed writes two filled sell rows, each with its own
`pnl_usd`. Every count built by walking those rows therefore counts that
one position twice.

That is not a rounding issue. It contaminates, in order of severity:

  - the 100-closed-trades evidence threshold, which decides when the
    record is allowed to be read as evidence at all;
  - win rate, because a partial that banked a gain and a final close that
    gave it back score as one win and one loss instead of one outcome;
  - expectancy, whose denominator is inflated by however many partials
    fired;
  - consecutive-loss streaks, because a winning partial can break a streak
    of losing POSITIONS;
  - anything resampling those rows as if they were draws from independent
    trials - Monte Carlo, confidence intervals - since two legs of one
    position share an entry, a signal, a token, a regime and a sizing
    decision, and are about as far from independent as two observations
    can be.

The last one is the reason this is worth a module rather than a helper.
A confidence interval assumes independent observations; feeding it
correlated legs produces an interval that is too narrow, which makes a
thin record look more conclusive than it is. The error runs toward
overconfidence.

What this module does NOT do
----------------------------

It does not change the strategy, and it does not silently replace the
leg-level numbers. Both counts are reported side by side, because they
answer different questions: "how many fills did we make?" is a real
question about execution, and "how many independent bets did we place?"
is the one statistics needs. Only the second is a sample size.

It also does not touch the risk manager's consecutive-loss rule, which has
the same leg-vs-position defect (app/risk/manager.py). That rule decides
when the bot halts, so changing it changes live risk behaviour and belongs
behind its own flag, not in a measurement change.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from app import models



def _aware(moment: dt.datetime) -> dt.datetime:
    """Naive timestamps from SQLite are UTC; make them comparable.

    Rows reloaded from the database come back naive while rows created in
    the current session are aware, and sorting a list holding both raises
    TypeError. Same guard as app/analysis/trade_analytics.py.
    """
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


@dataclass
class RoundTrip:
    """One position's complete life, from entry to final exit."""

    position_id: int | None
    legs: list[models.Trade] = field(default_factory=list)
    #: True when this is a single leg that carried no position_id and so
    #: could not be proven to be a whole position.
    attributed: bool = True

    @property
    def pnl_usd(self) -> float:
        return sum(t.pnl_usd for t in self.legs if t.pnl_usd is not None)

    @property
    def leg_count(self) -> int:
        return len(self.legs)

    @property
    def partial_count(self) -> int:
        """Exits before the last one. Zero for a position closed in one go."""
        return max(0, len(self.legs) - 1)

    @property
    def is_win(self) -> bool:
        """Judged on the WHOLE position, which is the point of the module."""
        return self.pnl_usd > 0

    @property
    def symbol(self) -> str | None:
        return self.legs[0].symbol if self.legs else None

    @property
    def closed_at(self) -> dt.datetime | None:
        """When the position finally ended - its last leg's close."""
        stamps = [_aware(t.closed_at) for t in self.legs if t.closed_at is not None]
        return max(stamps) if stamps else None

    @property
    def strategy_version(self) -> str | None:
        return self.legs[0].strategy_version if self.legs else None


@dataclass
class RoundTripSummary:
    """Both units, side by side, plus how far apart they are."""

    round_trips: list[RoundTrip]
    exit_leg_count: int
    unattributed_leg_count: int

    @property
    def count(self) -> int:
        return len(self.round_trips)

    @property
    def partial_leg_count(self) -> int:
        """Exit legs beyond the first for each position - the overcount."""
        return sum(rt.partial_count for rt in self.round_trips)

    @property
    def positions_with_partials(self) -> int:
        return sum(1 for rt in self.round_trips if rt.partial_count > 0)

    @property
    def counts_agree(self) -> bool:
        return self.exit_leg_count == self.count

    @property
    def wins(self) -> int:
        return sum(1 for rt in self.round_trips if rt.is_win)

    @property
    def win_rate_pct(self) -> float | None:
        return (self.wins / self.count * 100) if self.count else None

    @property
    def total_pnl_usd(self) -> float:
        return sum(rt.pnl_usd for rt in self.round_trips)

    @property
    def expectancy_usd(self) -> float | None:
        return (self.total_pnl_usd / self.count) if self.count else None

    @property
    def profit_factor(self) -> float | None:
        """None when there is nothing to divide, not 0.

        Mirrors compute_stats: an all-winners record has no denominator,
        and reporting infinity as a number invites reading it as strength
        when it is a symptom of a small sample.
        """
        gross_profit = sum(rt.pnl_usd for rt in self.round_trips if rt.pnl_usd > 0)
        gross_loss = sum(rt.pnl_usd for rt in self.round_trips if rt.pnl_usd < 0)
        if gross_loss:
            return gross_profit / abs(gross_loss)
        return float("inf") if gross_profit else None

    @property
    def longest_losing_streak(self) -> int:
        """Consecutive losing POSITIONS, oldest first.

        Differs from the leg-level streak precisely when a winning partial
        sits inside a run of losing positions - which is the case the
        leg-level count gets wrong, and the case a consecutive-loss halt
        is supposed to catch.
        """
        longest = current = 0
        for rt in self.round_trips:
            if rt.pnl_usd < 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    def discrepancy_note(self) -> str | None:
        """One sentence for a report, or None when the two units agree."""
        if self.counts_agree and not self.unattributed_leg_count:
            return None
        parts = [
            f"{self.exit_leg_count} exit legs resolve to {self.count} round trips"
        ]
        if self.partial_leg_count:
            parts.append(
                f"{self.positions_with_partials} position(s) took a partial exit, "
                f"adding {self.partial_leg_count} extra leg(s)"
            )
        if self.unattributed_leg_count:
            parts.append(
                f"{self.unattributed_leg_count} leg(s) carried no position_id and "
                "are counted as one round trip each, which may still overcount"
            )
        return "; ".join(parts) + ". Sample size means round trips, not legs."


def group_round_trips(trades: list[models.Trade]) -> RoundTripSummary:
    """Fold realized exit legs into one RoundTrip per position.

    Only legs with realized P&L are considered - the same filter
    `closed_trades()` applies - because an unfilled or pending leg is not
    an outcome.

    A leg with no `position_id` cannot be proven to belong to any
    position, so it becomes its own round trip AND is counted in
    `unattributed_leg_count`. Silently merging such legs would invent
    positions; silently dropping them would hide realized P&L. Counting
    them while flagging them is the only option that neither fabricates
    nor discards, and the flag is what stops the resulting number being
    read as exact. CLAUDE.md: a measurement that cannot be taken is
    recorded as unmeasurable, never as zero.
    """
    realized = [
        t for t in trades
        if t.pnl_usd is not None and t.closed_at is not None
    ]

    by_position: dict[int, RoundTrip] = {}
    orphans: list[RoundTrip] = []

    for leg in realized:
        if leg.position_id is None:
            orphans.append(RoundTrip(position_id=None, legs=[leg], attributed=False))
            continue
        trip = by_position.get(leg.position_id)
        if trip is None:
            trip = RoundTrip(position_id=leg.position_id)
            by_position[leg.position_id] = trip
        trip.legs.append(leg)

    for trip in by_position.values():
        trip.legs.sort(key=lambda t: _aware(t.closed_at))

    trips = list(by_position.values()) + orphans
    # Oldest completed position first, so streak order is chronological.
    _EPOCH = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    trips.sort(key=lambda rt: _aware(rt.closed_at) if rt.closed_at else _EPOCH)

    return RoundTripSummary(
        round_trips=trips,
        exit_leg_count=len(realized),
        unattributed_leg_count=len(orphans),
    )
