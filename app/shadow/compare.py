"""Paired comparison between the champion and one challenger.

PAIRED is the whole point. Comparing two strategies over all their
observations compares the opportunities as much as the strategies - if the
challenger's sample happened to include a calmer week, its better numbers
say nothing about the challenger. Restricting to opportunities BOTH
evaluated removes that confound entirely, and what is left is the
disagreement.

The cost is sample size: pairing throws away every observation only one
arm saw. That is the right trade. A larger unpaired sample measures the
wrong thing more precisely.

THIS MODULE DOES NOT PROMOTE

It produces Arms and hands them to app/autopilot/promote.py, which owns
the six bars and the multiple-comparison correction. Letting the module
that generates challengers also decide which ones win would let a search
mark its own homework.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.autopilot.promote import Arm
from app.shadow.challengers import CHAMPION_ID
from app.shadow.recorder import BUY, REJECT

# Below this many paired opportunities, the agreement counts are anecdote.
MIN_PAIRS = 20


@dataclass
class PairedComparison:
    challenger_id: str
    paired: int = 0
    both_entered: int = 0
    both_rejected: int = 0
    champion_only: int = 0
    challenger_only: int = 0
    champion_returns: list[float] = field(default_factory=list)
    challenger_returns: list[float] = field(default_factory=list)
    champion_by_regime: dict[str, list[float]] = field(default_factory=dict)
    challenger_by_regime: dict[str, list[float]] = field(default_factory=dict)
    unresolved: int = 0

    @property
    def conclusive(self) -> bool:
        return self.paired >= MIN_PAIRS

    @property
    def agreement_pct(self) -> float | None:
        if not self.paired:
            return None
        return (self.both_entered + self.both_rejected) / self.paired * 100

    def _mean(self, values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @property
    def champion_expectancy(self) -> float | None:
        return self._mean(self.champion_returns)

    @property
    def challenger_expectancy(self) -> float | None:
        return self._mean(self.challenger_returns)

    @property
    def difference(self) -> float | None:
        a, b = self.champion_expectancy, self.challenger_expectancy
        return (b - a) if a is not None and b is not None else None

    def arms(self) -> tuple[Arm, Arm]:
        """Hand the two sides to the promotion gate.

        Returns are in percent here and the gate reads R-multiples, so they
        are divided by 100. Feeding percent straight in would inflate every
        effect by a factor of a hundred and clear the effect bar on noise.
        """
        return (
            Arm(
                label=CHAMPION_ID,
                returns_r=[r / 100 for r in self.champion_returns],
                by_regime={
                    k: [r / 100 for r in v] for k, v in self.champion_by_regime.items()
                },
            ),
            Arm(
                label=self.challenger_id,
                returns_r=[r / 100 for r in self.challenger_returns],
                by_regime={
                    k: [r / 100 for r in v] for k, v in self.challenger_by_regime.items()
                },
            ),
        )

    def verdict(self) -> str:
        if not self.conclusive:
            return (
                f"INSUFFICIENT_DATA: {self.paired} paired opportunities "
                f"(need >={MIN_PAIRS}). Nothing here separates {self.challenger_id} from "
                "the champion, and that is a statement about the sample, not about the "
                "challenger."
            )
        if self.difference is None:
            return (
                f"INSUFFICIENT_DATA: {self.paired} pairs recorded but no resolved outcomes "
                "on both sides, so no expectancy can be compared."
            )
        return (
            f"{self.challenger_id} differs from the champion by {self.difference:+.2f}% per "
            f"trade across {self.paired} paired opportunities "
            f"({self.agreement_pct:.0f}% agreement). The promotion gate decides whether that "
            "clears its bars - this number alone does not."
        )

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 4) if v is not None else None
        return {
            "challenger_id": self.challenger_id,
            "paired": self.paired,
            "conclusive": self.conclusive,
            "both_entered": self.both_entered,
            "both_rejected": self.both_rejected,
            "champion_only": self.champion_only,
            "challenger_only": self.challenger_only,
            "unresolved": self.unresolved,
            "agreement_pct": r(self.agreement_pct),
            "champion_expectancy_pct": r(self.champion_expectancy),
            "challenger_expectancy_pct": r(self.challenger_expectancy),
            "difference_pct": r(self.difference),
            "verdict": self.verdict(),
        }


def compare(db: Session, challenger_id: str) -> PairedComparison:
    """Pair the champion against one challenger on shared opportunities."""
    result = PairedComparison(challenger_id=challenger_id)

    rows = (
        db.query(models.ShadowDecision)
        .filter(models.ShadowDecision.strategy_id.in_([CHAMPION_ID, challenger_id]))
        .all()
    )
    by_opportunity: dict[str, dict[str, models.ShadowDecision]] = defaultdict(dict)
    for row in rows:
        by_opportunity[row.opportunity_id][row.strategy_id] = row

    # Outcomes for hypothetical positions, keyed the same way.
    positions = {
        (p.opportunity_id, p.strategy_id): p
        for p in db.query(models.ShadowPosition).all()
    }

    for oid, arms in by_opportunity.items():
        champion = arms.get(CHAMPION_ID)
        challenger = arms.get(challenger_id)
        # Only opportunities BOTH evaluated. An unpaired observation is
        # not a weaker data point, it is a different comparison.
        if champion is None or challenger is None:
            continue
        result.paired += 1

        champion_in = champion.decision == BUY and bool(champion.fill_succeeded)
        challenger_in = challenger.decision == BUY and bool(challenger.fill_succeeded)

        if champion_in and challenger_in:
            result.both_entered += 1
        elif champion_in:
            result.champion_only += 1
        elif challenger_in:
            result.challenger_only += 1
        else:
            result.both_rejected += 1

        # A strategy that declined has a realised return of zero on this
        # opportunity - it is not missing data, it is a decision with an
        # outcome. Treating it as missing would drop exactly the
        # observations where the two arms disagreed, which is the signal.
        for side, decision, entered, returns, by_regime in (
            ("champion", champion, champion_in, result.champion_returns, result.champion_by_regime),
            ("challenger", challenger, challenger_in, result.challenger_returns, result.challenger_by_regime),
        ):
            if not entered:
                returns.append(0.0)
                if decision.market_regime:
                    for axis in decision.market_regime.split("/"):
                        by_regime.setdefault(axis, []).append(0.0)
                continue
            position = positions.get((oid, decision.strategy_id))
            if position is None or position.return_pct is None:
                # Entered but not yet resolved. Counted, not guessed - a
                # pending outcome recorded as 0% would be a fabricated
                # measurement.
                result.unresolved += 1
                continue
            returns.append(position.return_pct)
            if decision.market_regime:
                for axis in decision.market_regime.split("/"):
                    by_regime.setdefault(axis, []).append(position.return_pct)

    return result


def compare_all(db: Session) -> list[PairedComparison]:
    """Every challenger that has recorded at least one decision."""
    ids = [
        row[0] for row in
        db.query(models.ShadowDecision.strategy_id)
        .filter(models.ShadowDecision.strategy_id != CHAMPION_ID)
        .distinct()
        .all()
    ]
    return [compare(db, strategy_id) for strategy_id in sorted(ids)]
