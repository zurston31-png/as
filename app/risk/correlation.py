"""Correlation risk: five memecoins are not five independent positions.

The concurrent-position limit implicitly assumes diversification. On
Solana memecoins that assumption is close to false. They move together -
with SOL, with overall risk appetite, and with whatever narrative is
running that week. A book of five names can be, in practice, one position
in "Solana memecoins" at five times the intended size, and every
per-position risk limit is satisfied the whole time.

WHAT THIS MEASURES

Pairwise Pearson correlation of RETURNS, not of prices. Two tokens both
drifting upward have correlated price levels almost by definition; what
matters for risk is whether they fall together, which is a property of
their returns.

From those pairwise numbers it computes EFFECTIVE EXPOSURE: what the book
is worth in terms of independent bets.

    effective = sqrt( sum_i sum_j  w_i * w_j * rho_ij )

With five uncorrelated equal positions this is about 45% of the gross - the
familiar diversification benefit. With five perfectly correlated positions
it is 100% of the gross, correctly reporting that the book is one bet.

WHAT IT REFUSES TO DO

It never assumes a correlation it has not measured. A pair with too few
overlapping observations is reported as UNKNOWN, and unknown pairs are
treated as fully correlated for the exposure calculation - the
conservative direction. Assuming independence by default is how a book
that is secretly one position passes every check.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Below this many overlapping return observations, a correlation estimate
# is noise. Two points always correlate perfectly.
MIN_OVERLAP = 20

# Pairs at or above this are treated as "the same bet" for reporting.
HIGH_CORRELATION = 0.70


def returns_from_prices(prices: list[float]) -> list[float]:
    """Simple returns. Non-positive prices break the ratio, so they end the
    usable series rather than producing an infinity."""
    out: list[float] = []
    for previous, current in zip(prices, prices[1:]):
        if previous is None or current is None or previous <= 0:
            continue
        out.append(current / previous - 1)
    return out


def pearson(a: list[float], b: list[float]) -> float | None:
    """Correlation of two equal-length return series, or None.

    None when there is not enough overlap, or when either series is
    perfectly flat - a constant has no correlation with anything, and
    reporting 0 would claim independence that was never measured.
    """
    n = min(len(a), len(b))
    if n < MIN_OVERLAP:
        return None
    a, b = a[-n:], b[-n:]

    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)

    if var_a <= 0 or var_b <= 0:
        return None
    return max(-1.0, min(1.0, cov / math.sqrt(var_a * var_b)))


@dataclass
class Pair:
    a: str
    b: str
    correlation: float | None
    overlap: int

    @property
    def known(self) -> bool:
        return self.correlation is not None

    @property
    def high(self) -> bool:
        return self.known and self.correlation >= HIGH_CORRELATION

    def as_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "correlation": round(self.correlation, 3) if self.known else None,
            "overlap": self.overlap,
            "known": self.known,
            "high": self.high,
        }


@dataclass
class CorrelationReport:
    pairs: list[Pair] = field(default_factory=list)
    gross_exposure_usd: float = 0.0
    effective_exposure_usd: float = 0.0
    positions: int = 0
    unknown_pairs: int = 0

    @property
    def diversification_ratio(self) -> float | None:
        """effective / gross. 1.0 means the book is one bet."""
        if self.gross_exposure_usd <= 0:
            return None
        return self.effective_exposure_usd / self.gross_exposure_usd

    @property
    def high_pairs(self) -> list[Pair]:
        return [p for p in self.pairs if p.high]

    @property
    def independent_bets(self) -> float | None:
        """How many genuinely independent positions the book amounts to."""
        ratio = self.diversification_ratio
        if ratio is None or ratio <= 0:
            return None
        return 1 / (ratio ** 2)

    def as_dict(self) -> dict:
        return {
            "positions": self.positions,
            "gross_exposure_usd": round(self.gross_exposure_usd, 2),
            "effective_exposure_usd": round(self.effective_exposure_usd, 2),
            "diversification_ratio": (
                round(self.diversification_ratio, 3) if self.diversification_ratio is not None else None
            ),
            "independent_bets": (
                round(self.independent_bets, 2) if self.independent_bets is not None else None
            ),
            "unknown_pairs": self.unknown_pairs,
            "high_correlation_pairs": [p.as_dict() for p in self.high_pairs],
            "pairs": [p.as_dict() for p in self.pairs],
        }

    def summary(self) -> str:
        if self.positions < 2:
            return f"{self.positions} open position(s) - correlation is not yet a question."
        ratio = self.diversification_ratio
        lines = [
            f"{self.positions} open positions, ${self.gross_exposure_usd:,.0f} gross, "
            f"${self.effective_exposure_usd:,.0f} effective "
            f"({ratio * 100:.0f}% of gross, ~{self.independent_bets:.1f} independent bets)."
        ]
        if self.high_pairs:
            worst = max(self.high_pairs, key=lambda p: p.correlation)
            lines.append(
                f"  {len(self.high_pairs)} highly correlated pair(s), worst "
                f"{worst.a}/{worst.b} at {worst.correlation:.2f}."
            )
        if self.unknown_pairs:
            lines.append(
                f"  {self.unknown_pairs} pair(s) have too little overlapping history to measure "
                "and are counted as fully correlated - the conservative direction."
            )
        return "\n".join(lines)


def analyse(
    exposures: dict[str, float], return_series: dict[str, list[float]]
) -> CorrelationReport:
    """Effective exposure of a book, given each position's USD value and
    return history.

    `exposures` and `return_series` are keyed by canonical instrument key
    (see app/identity.py) so two mints sharing a symbol stay separate.
    """
    keys = sorted(exposures)
    gross = sum(max(v, 0.0) for v in exposures.values())

    report = CorrelationReport(positions=len(keys), gross_exposure_usd=gross)
    if not keys or gross <= 0:
        return report

    correlations: dict[tuple[str, str], float] = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            rho = pearson(return_series.get(a, []), return_series.get(b, []))
            overlap = min(len(return_series.get(a, [])), len(return_series.get(b, [])))
            report.pairs.append(Pair(a, b, rho, overlap))
            if rho is None:
                report.unknown_pairs += 1
                # Unmeasured means unknown, and unknown is treated as fully
                # correlated. Defaulting to independence is how a book that
                # is secretly one position passes every check.
                correlations[(a, b)] = 1.0
            else:
                correlations[(a, b)] = rho

    total = 0.0
    for a in keys:
        for b in keys:
            wa, wb = max(exposures[a], 0.0), max(exposures[b], 0.0)
            if a == b:
                rho = 1.0
            else:
                rho = correlations.get((a, b)) or correlations.get((b, a)) or 1.0
            total += wa * wb * rho

    report.effective_exposure_usd = math.sqrt(max(total, 0.0))
    return report
