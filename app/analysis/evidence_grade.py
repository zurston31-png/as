"""How much a number can be trusted, from how many observations it has.

WHY A LABEL AND NOT JUST A FLOOR

app/analysis/evidence.py already withholds a statistic below a sample
floor, which stops the worst failure: reporting "win rate 0%" over no
trades as though it measured a bad strategy. But a floor is binary, and
either side of it is misleading in its own way. Twenty-one observations
and five hundred both render as a number, and they are not the same
number - one is a direction, the other is a measurement.

So every reported statistic gets a grade, and the grade travels with it.

WHERE THE THRESHOLDS COME FROM

They are not chosen by feel. For a proportion - a win rate - the standard
error is sqrt(p(1-p)/n), worst at p=0.5, and the 95% confidence half-width
is 1.96 times that. Reading that backwards gives the ladder:

    n <  30    half-width worse than +/-18pp.  INSUFFICIENT
    n <  100   half-width worse than +/-10pp.  EARLY
    n <  385   half-width worse than  +/-5pp.  USABLE
    n >= 385   half-width  +/-5pp or better.   STRONG

385 is the familiar "n for +/-5% at 95% confidence" and 30 is the
conventional point below which the normal approximation stops being worth
quoting at all. Each boundary therefore means something specific about
how wide the error bar is, and `half_width_pp` reports that width
directly so the label never has to be taken on trust.

TWO THINGS THIS IS NOT

It is not evidence of an edge. STRONG says the number is measured
precisely; if the number is negative, STRONG means the bot is confidently
losing. Sample size and profitability are orthogonal and conflating them
is how a large, well-measured loss gets read as a good sign.

It is not a promise for RETURNS. The ladder is derived for a proportion.
Memecoin returns are heavy-tailed, and a mean over a heavy-tailed
distribution converges far more slowly than a proportion does - a handful
of extreme outcomes can move it after the sample looks large. For a mean
return the same n therefore buys LESS precision than the grade suggests,
and the grade should be read as an optimistic ceiling. `caveat()` says so
wherever a mean is being graded.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

# 95% two-sided.
Z = 1.96

# Boundaries, in observations. See the module docstring for the derivation.
EARLY_AT = 30
USABLE_AT = 100
STRONG_AT = 385


class EvidenceLevel(str, enum.Enum):
    """How precisely a statistic is pinned down. Not whether it is good."""

    INSUFFICIENT = "INSUFFICIENT"
    EARLY = "EARLY"
    USABLE = "USABLE"
    STRONG = "STRONG"

    @property
    def rank(self) -> int:
        return _RANK[self]

    def __lt__(self, other: "EvidenceLevel") -> bool:
        return self.rank < other.rank


_RANK = {
    EvidenceLevel.INSUFFICIENT: 0,
    EvidenceLevel.EARLY: 1,
    EvidenceLevel.USABLE: 2,
    EvidenceLevel.STRONG: 3,
}

_MEANING = {
    EvidenceLevel.INSUFFICIENT: (
        "not enough observations to say anything - not even the direction. "
        "Report the count, not the statistic"
    ),
    EvidenceLevel.EARLY: (
        "enough to see a direction, not a magnitude. Treat it as a hypothesis "
        "to keep collecting against, never as a result"
    ),
    EvidenceLevel.USABLE: (
        "precise enough to compare against a clearly different number, but not "
        "to split hairs with one a few points away"
    ),
    EvidenceLevel.STRONG: (
        "measured to within about five points. Precise - which says nothing "
        "about whether the number itself is good"
    ),
}


def classify(samples: int) -> EvidenceLevel:
    """The grade for a statistic computed over `samples` observations."""
    if samples < EARLY_AT:
        return EvidenceLevel.INSUFFICIENT
    if samples < USABLE_AT:
        return EvidenceLevel.EARLY
    if samples < STRONG_AT:
        return EvidenceLevel.USABLE
    return EvidenceLevel.STRONG


def half_width_pp(samples: int) -> float | None:
    """95% confidence half-width for a proportion, in percentage points.

    Evaluated at p=0.5, which is the worst case - so the real interval for
    a lopsided win rate is narrower than this, never wider. None below one
    observation, where the quantity is undefined rather than infinite.
    """
    if samples < 1:
        return None
    return Z * math.sqrt(0.25 / samples) * 100


def samples_needed_for(level: EvidenceLevel) -> int:
    return {
        EvidenceLevel.INSUFFICIENT: 0,
        EvidenceLevel.EARLY: EARLY_AT,
        EvidenceLevel.USABLE: USABLE_AT,
        EvidenceLevel.STRONG: STRONG_AT,
    }[level]


def shortfall_to_next(samples: int) -> tuple[EvidenceLevel, int] | None:
    """The next grade up and how many more observations it needs.

    Returns None at STRONG. Phrased as "how many more" rather than "how
    many total" because the actionable question during a collection run is
    always how much further there is to go.
    """
    level = classify(samples)
    if level is EvidenceLevel.STRONG:
        return None
    nxt = {
        EvidenceLevel.INSUFFICIENT: EvidenceLevel.EARLY,
        EvidenceLevel.EARLY: EvidenceLevel.USABLE,
        EvidenceLevel.USABLE: EvidenceLevel.STRONG,
    }[level]
    return nxt, samples_needed_for(nxt) - samples


@dataclass(frozen=True)
class Graded:
    """A statistic, its sample size, and how much weight it can carry."""

    name: str
    value: float | None
    samples: int
    unit: str = ""
    # Set for a mean rather than a proportion. Heavy-tailed returns
    # converge more slowly than the ladder assumes, so the grade is an
    # optimistic ceiling and says so.
    is_mean: bool = False

    @property
    def level(self) -> EvidenceLevel:
        return classify(self.samples)

    @property
    def reportable(self) -> bool:
        """Whether the VALUE should be shown at all.

        INSUFFICIENT withholds it. The sample count is still reported -
        "12 observations so far" is useful and honest, where "win rate
        58%" over twelve is neither.
        """
        return self.value is not None and self.level is not EvidenceLevel.INSUFFICIENT

    def caveat(self) -> str:
        base = _MEANING[self.level]
        if self.is_mean and self.level is not EvidenceLevel.INSUFFICIENT:
            return (
                base + ". This is a mean of a heavy-tailed distribution, so the "
                "grade is an optimistic ceiling - a few extreme outcomes can still move it"
            )
        return base

    def summary(self) -> str:
        """One line, safe to print anywhere."""
        if not self.reportable:
            gap = shortfall_to_next(self.samples)
            need = f" - {gap[1]} more for {gap[0].value}" if gap else ""
            return f"{self.name}: not enough data ({self.samples} observations{need})"
        return (
            f"{self.name}: {self.value:,.2f}{self.unit} "
            f"[{self.level.value}, n={self.samples}]"
        )

    def as_dict(self) -> dict:
        gap = shortfall_to_next(self.samples)
        width = half_width_pp(self.samples)
        return {
            "name": self.name,
            "value": self.value if self.reportable else None,
            "samples": self.samples,
            "unit": self.unit,
            "level": self.level.value,
            "reportable": self.reportable,
            "half_width_pp": round(width, 1) if width is not None else None,
            "next_level": gap[0].value if gap else None,
            "samples_to_next_level": gap[1] if gap else None,
            "caveat": self.caveat(),
        }


def weakest(graded: list[Graded]) -> EvidenceLevel:
    """The grade a set of statistics collectively deserves.

    The weakest link, not the average. A conclusion drawn from four
    measurements is only as good as the flimsiest one, and averaging the
    grades would let three well-sampled numbers launder one that has
    almost no data behind it.
    """
    if not graded:
        return EvidenceLevel.INSUFFICIENT
    return min((g.level for g in graded), key=lambda level: level.rank)
