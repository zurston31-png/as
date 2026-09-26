"""Cross-validating one number against a second source.

When two independent providers report a token's price or liquidity and
they disagree materially, at least one is wrong - and there is no way to
tell which. The dangerous instinct is to pick the more convenient value:
the higher liquidity (which permits a bigger position), or the lower price
(which makes the entry look better). That is not reconciliation, it is
selection bias with extra steps, and it biases every result in the
direction of trading more.

So the rule here is: WHEN SOURCES DISAGREE BEYOND TOLERANCE, DON'T TRADE.
Mark the observation uncertain, log the disagreement with both values, and
skip. A skipped trade costs one opportunity; a trade sized off a wrong
liquidity number costs the position.

Where a value IS needed and the sources merely differ slightly, the more
CONSERVATIVE one is used - lower liquidity, higher price for a buy - never
an average. Averaging two numbers when one of them is wrong produces a
third number that is also wrong, and hides the disagreement.

This module is pure comparison logic. It does no fetching, so it can be
tested without a network and reused for any pair of sources.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Agreement:
    field: str
    primary: float | None
    secondary: float | None
    tolerance: float
    primary_source: str = "primary"
    secondary_source: str = "secondary"

    @property
    def comparable(self) -> bool:
        """Both sources returned a usable number."""
        return (
            self.primary is not None and self.primary > 0
            and self.secondary is not None and self.secondary > 0
        )

    @property
    def relative_gap(self) -> float | None:
        """Difference as a fraction of the SMALLER value.

        Against the smaller one deliberately: a source reporting $10k when
        another reports $100k is a 900% error, and dividing by the larger
        would report a much more forgiving 90%.
        """
        if not self.comparable:
            return None
        low, high = sorted((self.primary, self.secondary))
        return (high - low) / low

    @property
    def agrees(self) -> bool:
        gap = self.relative_gap
        return gap is not None and gap <= self.tolerance

    @property
    def single_source(self) -> bool:
        """Exactly one source answered. Not a disagreement, but not a
        cross-check either - the caller decides whether one is enough."""
        got_primary = self.primary is not None and self.primary > 0
        got_secondary = self.secondary is not None and self.secondary > 0
        return got_primary != got_secondary

    def conservative(self, *, prefer: str) -> float | None:
        """The safer of the two values.

        `prefer` is "low" or "high": low for liquidity and volume (assume
        the thinner pool), high for the price you are about to pay. Never
        an average - averaging a right number with a wrong one produces a
        third wrong number and hides that they disagreed.
        """
        values = [v for v in (self.primary, self.secondary) if v is not None and v > 0]
        if not values:
            return None
        if prefer == "low":
            return min(values)
        if prefer == "high":
            return max(values)
        raise ValueError(f"prefer must be 'low' or 'high', got {prefer!r}")

    @property
    def message(self) -> str:
        if not self.comparable:
            if self.single_source:
                return f"{self.field}: only one source answered - no cross-check possible"
            return f"{self.field}: neither source returned a usable value"
        if self.agrees:
            return (
                f"{self.field}: sources agree within {self.relative_gap * 100:.1f}% "
                f"({self.primary_source} {self.primary:,.6g}, "
                f"{self.secondary_source} {self.secondary:,.6g})"
            )
        return (
            f"{self.field}: sources DISAGREE by {self.relative_gap * 100:.0f}% - "
            f"{self.primary_source} says {self.primary:,.6g}, "
            f"{self.secondary_source} says {self.secondary:,.6g} "
            f"(tolerance {self.tolerance * 100:.0f}%). At least one is wrong and there is no "
            "way to tell which."
        )

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "primary": self.primary,
            "secondary": self.secondary,
            "primary_source": self.primary_source,
            "secondary_source": self.secondary_source,
            "relative_gap_pct": (
                round(self.relative_gap * 100, 2) if self.relative_gap is not None else None
            ),
            "tolerance_pct": round(self.tolerance * 100, 2),
            "comparable": self.comparable,
            "agrees": self.agrees,
            "single_source": self.single_source,
            "message": self.message,
        }


@dataclass
class CrossCheck:
    agreements: list[Agreement]
    require_two_sources: bool = False

    @property
    def disagreements(self) -> list[Agreement]:
        return [a for a in self.agreements if a.comparable and not a.agrees]

    @property
    def uncrossed(self) -> list[Agreement]:
        return [a for a in self.agreements if a.single_source]

    @property
    def trustworthy(self) -> bool:
        """Safe to trade on.

        False on any material disagreement. Also false when a second source
        was required and did not answer - "we could not check" is not
        "we checked and it was fine".
        """
        if self.disagreements:
            return False
        if self.require_two_sources and self.uncrossed:
            return False
        return True

    @property
    def reason(self) -> str:
        if self.disagreements:
            return "; ".join(a.message for a in self.disagreements)
        if self.require_two_sources and self.uncrossed:
            return "; ".join(a.message for a in self.uncrossed)
        return "sources agree"

    def as_dict(self) -> dict:
        return {
            "trustworthy": self.trustworthy,
            "reason": self.reason,
            "agreements": [a.as_dict() for a in self.agreements],
        }


def compare(
    *,
    price: tuple[float | None, float | None] | None = None,
    liquidity: tuple[float | None, float | None] | None = None,
    volume: tuple[float | None, float | None] | None = None,
    price_tolerance: float = 0.05,
    liquidity_tolerance: float = 0.30,
    volume_tolerance: float = 0.50,
    primary_source: str = "dexscreener",
    secondary_source: str = "geckoterminal",
    require_two_sources: bool = False,
) -> CrossCheck:
    """Compare (primary, secondary) readings of each supplied field.

    Tolerances differ per field on purpose, because the sources genuinely
    measure slightly different things. Price should match closely - both
    read the same pool. Liquidity is looser: providers include different
    pools and update at different times. Volume is looser still, since
    windows and deduplication of wash trades vary between them. Setting all
    three to the same tight number would produce constant false alarms and
    train whoever reads them to ignore the check.
    """
    agreements: list[Agreement] = []
    for field_name, pair, tolerance in (
        ("price", price, price_tolerance),
        ("liquidity", liquidity, liquidity_tolerance),
        ("volume", volume, volume_tolerance),
    ):
        if pair is None:
            continue
        agreements.append(
            Agreement(
                field=field_name,
                primary=pair[0],
                secondary=pair[1],
                tolerance=tolerance,
                primary_source=primary_source,
                secondary_source=secondary_source,
            )
        )

    check = CrossCheck(agreements=agreements, require_two_sources=require_two_sources)
    for disagreement in check.disagreements:
        logger.warning("data source disagreement - %s", disagreement.message)
    return check
