"""Score calibration: does a higher score actually precede a better outcome?

This is the question the whole scoring engine stands or falls on, and it is
not answerable from the trade record. The bot only trades what it already
scored highly, so trades are a censored sample - "our 80+ trades did well"
is compatible with the score being pure noise, because the 55s were never
given the chance to disagree.

The answer key is app/models.ForwardReturn: for every candidate the engine
scored, whether or not it was traded, record the price at the time and then
come back later and record what it did. Grouping those outcomes by score
bucket produces a table that either shows a monotonic relationship or does
not. If 75-rated setups do not out-perform 65-rated ones, the honest
conclusion is that the score does not separate them, and no amount of extra
indicators fixes that.

WHAT THIS DELIBERATELY DOES NOT DO
It never back-fills a missing horizon with the last known price, and never
treats an unmeasurable outcome as a zero return. A token that stopped
trading is not a token that returned 0%; conflating the two is how a
survivorship-biased dataset gets built by accident. Unfilled horizons are
counted and reported separately.

Forward returns are RAW price moves, not strategy results: no stop, no
target, no position sizing. That is on purpose. Mixing exit logic into the
calibration measurement would make it impossible to tell whether a bad
bucket means "the score was wrong" or "the stop was too tight".
`net_of_costs` applies a flat round-trip execution cost so the comparison
is at least made against a realistic bar.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.config import settings

logger = logging.getLogger(__name__)

# The horizons the spec asks for, in minutes.
HORIZONS_MINUTES: tuple[int, ...] = (5, 15, 30, 60, 120, 240, 480, 1440)

# Score buckets. Chosen to straddle every threshold under discussion so the
# table can speak to each of them.
BUCKET_EDGES: tuple[float, ...] = (55, 60, 65, 70, 75, 80)

# A bucket thinner than this cannot separate signal from noise, and is
# reported but never used to justify a change.
MIN_BUCKET_SAMPLE = 30


def bucket_label(score: float) -> str:
    if score < BUCKET_EDGES[0]:
        return f"<{BUCKET_EDGES[0]:g}"
    for low, high in zip(BUCKET_EDGES, BUCKET_EDGES[1:]):
        if score < high:
            return f"{low:g}-{high:g}"
    return f"{BUCKET_EDGES[-1]:g}+"


ALL_BUCKETS: tuple[str, ...] = (
    f"<{BUCKET_EDGES[0]:g}",
    *(f"{low:g}-{high:g}" for low, high in zip(BUCKET_EDGES, BUCKET_EDGES[1:])),
    f"{BUCKET_EDGES[-1]:g}+",
)


def round_trip_cost_pct() -> float:
    """The flat execution cost charged against a raw forward return.

    Spread and fee on both legs, plus the configured slippage tolerance
    once. It is an approximation - real price impact depends on pool depth
    and trade size, which a raw forward return has no notion of - and it is
    deliberately on the pessimistic side, because a calibration table that
    flatters execution is the one that gets a strategy funded.
    """
    per_leg = settings.PAPER_FEE_PCT + settings.PAPER_SPREAD_PCT
    return 2 * per_leg + settings.SLIPPAGE_BPS / 10_000


@dataclass
class BucketOutcome:
    bucket: str
    horizon_minutes: int
    sample_size: int
    unmeasured: int                      # horizon due but no price - NOT zeros
    mean_return_pct: float | None = None
    median_return_pct: float | None = None
    win_rate_pct: float | None = None
    best_return_pct: float | None = None
    worst_return_pct: float | None = None
    mean_net_of_costs_pct: float | None = None

    @property
    def meaningful(self) -> bool:
        return self.sample_size >= MIN_BUCKET_SAMPLE

    @property
    def coverage_pct(self) -> float:
        total = self.sample_size + self.unmeasured
        return (self.sample_size / total * 100) if total else 0.0

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 3) if v is not None else None

        return {
            "bucket": self.bucket,
            "horizon_minutes": self.horizon_minutes,
            "sample_size": self.sample_size,
            "unmeasured": self.unmeasured,
            "coverage_pct": round(self.coverage_pct, 1),
            "meaningful": self.meaningful,
            "mean_return_pct": r(self.mean_return_pct),
            "median_return_pct": r(self.median_return_pct),
            "win_rate_pct": r(self.win_rate_pct),
            "best_return_pct": r(self.best_return_pct),
            "worst_return_pct": r(self.worst_return_pct),
            "mean_net_of_costs_pct": r(self.mean_net_of_costs_pct),
        }


@dataclass
class CalibrationTable:
    horizon_minutes: int
    buckets: list[BucketOutcome] = field(default_factory=list)
    cost_pct: float = 0.0

    @property
    def measurable_buckets(self) -> list[BucketOutcome]:
        return [b for b in self.buckets if b.meaningful]

    @property
    def monotonic(self) -> bool | None:
        """Do mean returns rise with the score bucket?

        The property the score claims to have. None when fewer than two
        buckets carry enough data to compare - which is not evidence either
        way, and must not be reported as a failure.
        """
        usable = [b for b in self.measurable_buckets if b.mean_return_pct is not None]
        if len(usable) < 2:
            return None
        order = {label: i for i, label in enumerate(ALL_BUCKETS)}
        usable.sort(key=lambda b: order.get(b.bucket, 0))
        return all(
            a.mean_return_pct <= b.mean_return_pct for a, b in zip(usable, usable[1:])
        )

    def verdict(self) -> str:
        """Plain-language reading. Says 'not enough data' when that is true."""
        usable = [b for b in self.measurable_buckets if b.mean_return_pct is not None]
        if len(usable) < 2:
            return (
                f"INSUFFICIENT DATA at {self.horizon_minutes}m: fewer than two score buckets have "
                f"{MIN_BUCKET_SAMPLE}+ measured outcomes. Nothing can be concluded about whether "
                "the score predicts anything."
            )
        order = {label: i for i, label in enumerate(ALL_BUCKETS)}
        usable.sort(key=lambda b: order.get(b.bucket, 0))
        low, high = usable[0], usable[-1]
        gap = high.mean_return_pct - low.mean_return_pct
        if self.monotonic and gap > 0:
            return (
                f"At {self.horizon_minutes}m the score separates outcomes: mean return rises from "
                f"{low.mean_return_pct:+.2f}% in {low.bucket} to {high.mean_return_pct:+.2f}% in "
                f"{high.bucket}, monotonically across {len(usable)} buckets."
            )
        if gap > 0:
            return (
                f"At {self.horizon_minutes}m higher scores do better overall "
                f"({low.bucket} {low.mean_return_pct:+.2f}% vs {high.bucket} "
                f"{high.mean_return_pct:+.2f}%) but NOT monotonically - the middle buckets are out "
                "of order, so the score is a weak ranking rather than a reliable one."
            )
        return (
            f"At {self.horizon_minutes}m the score does NOT predict outcomes: {high.bucket} averages "
            f"{high.mean_return_pct:+.2f}% against {low.bucket}'s {low.mean_return_pct:+.2f}%. "
            "A higher threshold would reduce trade count without improving trade quality."
        )

    def as_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "cost_pct": round(self.cost_pct * 100, 3),
            "monotonic": self.monotonic,
            "verdict": self.verdict(),
            "buckets": [b.as_dict() for b in self.buckets],
        }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def build_calibration(
    db: Session,
    *,
    horizon_minutes: int,
    strategy_version: str | None = None,
    since: dt.datetime | None = None,
) -> CalibrationTable:
    """Group measured forward returns at one horizon into score buckets."""
    query = db.query(models.ForwardReturn).filter(
        models.ForwardReturn.horizon_minutes == horizon_minutes,
        models.ForwardReturn.score.isnot(None),
    )
    if strategy_version is not None:
        query = query.filter(models.ForwardReturn.strategy_version == strategy_version)
    if since is not None:
        query = query.filter(models.ForwardReturn.observed_at >= since)

    grouped: dict[str, list[float]] = {label: [] for label in ALL_BUCKETS}
    unmeasured: dict[str, int] = {label: 0 for label in ALL_BUCKETS}

    now = dt.datetime.now(dt.timezone.utc)
    for row in query.all():
        label = bucket_label(row.score)
        if row.return_pct is None:
            due = row.due_at.replace(tzinfo=dt.timezone.utc) if row.due_at.tzinfo is None else row.due_at
            # Only count it as unmeasured once the horizon has actually
            # elapsed. A horizon still in the future is pending, not missing.
            if due <= now:
                unmeasured[label] += 1
            continue
        grouped[label].append(row.return_pct)

    cost = round_trip_cost_pct() * 100
    buckets = []
    for label in ALL_BUCKETS:
        returns = grouped[label]
        if not returns:
            buckets.append(BucketOutcome(label, horizon_minutes, 0, unmeasured[label]))
            continue
        mean = sum(returns) / len(returns)
        buckets.append(
            BucketOutcome(
                bucket=label,
                horizon_minutes=horizon_minutes,
                sample_size=len(returns),
                unmeasured=unmeasured[label],
                mean_return_pct=mean,
                median_return_pct=_median(returns),
                win_rate_pct=sum(1 for r in returns if r > 0) / len(returns) * 100,
                best_return_pct=max(returns),
                worst_return_pct=min(returns),
                mean_net_of_costs_pct=mean - cost,
            )
        )

    return CalibrationTable(
        horizon_minutes=horizon_minutes, buckets=buckets, cost_pct=round_trip_cost_pct()
    )


def build_all_horizons(
    db: Session, *, strategy_version: str | None = None, since: dt.datetime | None = None
) -> list[CalibrationTable]:
    return [
        build_calibration(db, horizon_minutes=h, strategy_version=strategy_version, since=since)
        for h in HORIZONS_MINUTES
    ]
