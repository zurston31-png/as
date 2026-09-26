"""Correlation risk applied to the live book.

app/risk/correlation.py is pure arithmetic over exposures and return
series. This is the part that reads the database, decides what to do with
the answer, and - importantly - decides what NOT to do with it.

WHY THE GATE NEEDS EVIDENCE AND THE REPORT DOES NOT

correlation.analyse() treats an unmeasured pair as fully correlated. That
is the right default for a REPORT: a book whose correlations are unknown
should not be described as diversified, and assuming independence is how a
book that is secretly one position passes every check.

It is the wrong default for a BLOCK. On day one there are no stored
prices, so every pair is unknown, so every candidate would look perfectly
correlated with the entire book, and the bot would refuse to open a second
position - forever, since it never collects the observations that would
prove otherwise. A risk check that stops all trading before it has
measured anything is indistinguishable from a broken bot, and the
sensible-looking response to it is to switch the check off, which is
worse than not having it.

So the two are split deliberately:

    report()  conservative - unknown counts as correlated, shown on the
              dashboard so the uncertainty is visible
    gate()    evidential - only MEASURED correlation at or above
              HIGH_CORRELATION contributes to the cluster that can block

This is not the check being softened. It is the difference between
describing risk and refusing a trade: the first should assume the worst,
the second needs a reason.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.early import watchlist as wl
from app.identity import instrument_key
from app.risk.correlation import (
    HIGH_CORRELATION, MIN_OVERLAP, CorrelationReport, analyse, pearson,
    returns_from_prices,
)

logger = logging.getLogger(__name__)


def _open_positions(db: Session) -> list[models.Position]:
    return (
        db.query(models.Position)
        .filter(models.Position.status == models.PositionStatus.OPEN.value)
        .all()
    )


def _exposure_usd(db: Session, position: models.Position) -> float:
    """What this position is worth, marked to the last stored price.

    Uses the stored observation rather than fetching, for two reasons.
    The position monitor writes one on every pass, so it is as fresh as
    anything a fetch would return; and a gate that made one network call
    per open position per candidate would add latency to the buy path at
    exactly the moment latency costs the most.

    Falls back to entry price rather than skipping the position. A
    position whose current value is unknown is still capital at risk, and
    dropping it from the book would understate exposure - the same trap
    app/services/portfolio.py documents for the same reason.
    """
    price = None
    if position.token_address:
        recent = wl.price_series(db, position.token_address, limit=1)
        price = recent[-1] if recent else None
    if not price or price <= 0:
        price = position.entry_price
    return max((position.qty or 0.0) * (price or 0.0), 0.0)


def report(db: Session) -> CorrelationReport:
    """Effective exposure of the open book, unknown counted as correlated."""
    positions = _open_positions(db)
    exposures: dict[str, float] = {}
    series: dict[str, list[float]] = {}

    for position in positions:
        key = instrument_key(position.symbol, position.token_address)
        exposures[key] = exposures.get(key, 0.0) + _exposure_usd(db, position)
        if position.token_address and key not in series:
            series[key] = returns_from_prices(wl.price_series(db, position.token_address))

    return analyse(exposures, series)


@dataclass
class ClusterVerdict:
    """Whether a candidate would pile onto an already-correlated cluster."""
    candidate: str
    cluster_usd: float = 0.0
    portfolio_usd: float = 0.0
    cap_usd: float = 0.0
    correlated_with: list[tuple[str, float]] = field(default_factory=list)
    measured_pairs: int = 0
    unmeasured_pairs: int = 0

    @property
    def blocked(self) -> bool:
        return self.cap_usd > 0 and self.cluster_usd > self.cap_usd

    @property
    def reason(self) -> str:
        if not self.blocked:
            return "no measured correlated cluster over the cap"
        names = ", ".join(
            f"{key} (rho {rho:+.2f})" for key, rho in self.correlated_with[:4]
        )
        return (
            f"correlated cluster ${self.cluster_usd:,.0f} exceeds the "
            f"${self.cap_usd:,.0f} cap ({settings.MAX_CORRELATED_CLUSTER_PCT:.0%} of a "
            f"${self.portfolio_usd:,.0f} portfolio). {self.candidate} moves with {names} - "
            "adding it is increasing one bet, not diversifying."
        )

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "blocked": self.blocked,
            "reason": self.reason,
            "cluster_usd": round(self.cluster_usd, 2),
            "cap_usd": round(self.cap_usd, 2),
            "measured_pairs": self.measured_pairs,
            "unmeasured_pairs": self.unmeasured_pairs,
            "correlated_with": [
                {"key": key, "rho": round(rho, 4)} for key, rho in self.correlated_with
            ],
        }


def gate(
    db: Session,
    *,
    symbol: str,
    token_address: str | None,
    portfolio_value_usd: float,
    proposed_size_usd: float,
) -> ClusterVerdict:
    """Would this candidate push a MEASURED correlated cluster over the cap?

    Only pairs with at least MIN_OVERLAP overlapping observations and a
    correlation at or above HIGH_CORRELATION count toward the cluster.
    Unmeasured pairs are counted and reported but never block - see the
    module docstring for why blocking on an absent measurement would halt
    the bot permanently.
    """
    candidate_key = instrument_key(symbol, token_address)
    verdict = ClusterVerdict(
        candidate=candidate_key,
        portfolio_usd=portfolio_value_usd,
        cap_usd=max(portfolio_value_usd, 0.0) * settings.MAX_CORRELATED_CLUSTER_PCT,
    )
    if not settings.CORRELATION_RISK_ENABLED or not token_address:
        verdict.cap_usd = 0.0        # disabled: cannot block
        return verdict

    candidate_returns = returns_from_prices(wl.price_series(db, token_address))

    for position in _open_positions(db):
        key = instrument_key(position.symbol, position.token_address)
        if key == candidate_key:
            # The same instrument is not a correlated cluster, it is the
            # existing position, and the per-token exposure cap already
            # covers it. Counting it here would double-charge it.
            continue
        if not position.token_address:
            verdict.unmeasured_pairs += 1
            continue

        other = returns_from_prices(wl.price_series(db, position.token_address))
        rho = pearson(candidate_returns, other)
        if rho is None or min(len(candidate_returns), len(other)) < MIN_OVERLAP:
            verdict.unmeasured_pairs += 1
            continue

        verdict.measured_pairs += 1
        if rho >= HIGH_CORRELATION:
            verdict.correlated_with.append((key, rho))
            verdict.cluster_usd += _exposure_usd(db, position)

    if verdict.correlated_with:
        verdict.cluster_usd += max(proposed_size_usd, 0.0)
        verdict.correlated_with.sort(key=lambda pair: pair[1], reverse=True)
    return verdict
