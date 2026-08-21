"""Is the forward-return worker keeping up?

`forward_returns.coverage()` reports how much of the dataset is resolved.
That is a statement about the dataset, and it degrades slowly and
ambiguously: a worker that stopped an hour ago and one that is merely
young both show coverage below 100%, and only one of them is a problem.

This module asks the operational question instead. Three symptoms, each
of which corrupts the dataset in a different way:

  BACKLOG    Rows past due and unresolved. Harmless briefly - a horizon
             comes due between passes by definition - and fatal if it
             grows, because every row that sits past its lateness
             tolerance is sealed unmeasurable and lost for good. The
             oldest overdue row is the number that matters; a large count
             of rows one minute overdue is just the batch about to run.

  LATENESS   How far past due resolutions are actually landing, measured
             from `actual_elapsed_minutes` against `horizon_minutes`. A
             worker resolving 60-minute horizons at 74 minutes is inside
             tolerance and about to fall outside it, and coverage will
             not show that until rows start being discarded.

  SEALED     Rows closed without a measurement. The reason is recorded on
             each one, and they are split by reason here - a token whose
             price feed died is a fact about the market, while a row
             sealed for lateness is a fact about this worker, and pooling
             them hides the only one that can be fixed.

WHAT "LATE" MEANS IS NOT DEFINED HERE

The threshold comes from `forward_returns.lateness_tolerance_minutes`,
the same function the resolver uses to decide whether to fill a row or
seal it. Restating the rule would let the health check and the resolver
drift apart and disagree about which rows were lost.

NOTHING HERE IS ESTIMATED. Every figure is counted from rows on disk. A
worker that has never run reports IDLE with no rows, which is not the
same as healthy and does not claim to be.
"""
from __future__ import annotations

import datetime as dt
import enum
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis.forward_returns import lateness_tolerance_minutes

# Below this many resolutions, lateness statistics are not reported: three
# samples cannot distinguish a slow worker from three slow rows.
MIN_RESOLUTIONS_FOR_LATENESS = 10


class ResolverStatus(str, enum.Enum):
    IDLE = "IDLE"            # nothing scheduled yet - not a health verdict
    HEALTHY = "HEALTHY"      # nothing overdue past its tolerance
    BEHIND = "BEHIND"        # rows overdue, none yet past tolerance
    LOSING_DATA = "LOSING_DATA"   # rows are past tolerance and will be sealed
    STALLED = "STALLED"      # rows are past tolerance and nothing has resolved


@dataclass
class ResolverHealth:
    status: ResolverStatus = ResolverStatus.IDLE
    detail: str = ""

    scheduled: int = 0
    resolved: int = 0
    pending: int = 0
    sealed_unmeasurable: int = 0

    overdue: int = 0
    overdue_past_tolerance: int = 0
    oldest_overdue_minutes: float | None = None

    # Lateness of rows that DID resolve, in minutes past their horizon.
    median_lateness_minutes: float | None = None
    worst_lateness_minutes: float | None = None
    lateness_samples: int = 0

    sealed_by_reason: dict[str, int] = field(default_factory=dict)
    last_resolution_at: dt.datetime | None = None

    @property
    def healthy(self) -> bool:
        return self.status in (ResolverStatus.HEALTHY, ResolverStatus.IDLE)

    @property
    def unmeasurable_rate_pct(self) -> float | None:
        """Share of CLOSED rows that carry no measurement.

        Computed over closed rows, not all rows: pending rows are not
        failures yet, and including them would make a healthy young
        dataset look like it was losing most of its observations.
        """
        closed = self.resolved + self.sealed_unmeasurable
        if not closed:
            return None
        return self.sealed_unmeasurable / closed * 100

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["healthy"] = self.healthy
        payload["unmeasurable_rate_pct"] = (
            round(self.unmeasurable_rate_pct, 1)
            if self.unmeasurable_rate_pct is not None else None
        )
        payload["last_resolution_at"] = (
            self.last_resolution_at.isoformat() if self.last_resolution_at else None
        )
        return payload


def _aware(moment: dt.datetime | None) -> dt.datetime | None:
    if moment is None:
        return None
    return moment.replace(tzinfo=dt.timezone.utc) if moment.tzinfo is None else moment


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def check_resolver_health(db: Session, *, now: dt.datetime | None = None) -> ResolverHealth:
    """Count the resolver's backlog, lateness and losses from the rows."""
    now = now or dt.datetime.now(dt.timezone.utc)
    health = ResolverHealth()

    rows = db.query(models.ForwardReturn).all()
    health.scheduled = len(rows)
    if not rows:
        health.status = ResolverStatus.IDLE
        health.detail = (
            "no forward returns scheduled yet. Nothing has been lost, and nothing has "
            "been collected either - this is an empty dataset, not a healthy one."
        )
        return health

    lateness: list[float] = []
    last_resolution: dt.datetime | None = None

    for row in rows:
        due = _aware(row.due_at)
        if row.return_pct is not None:
            health.resolved += 1
            measured = _aware(row.measured_at)
            if measured and (last_resolution is None or measured > last_resolution):
                last_resolution = measured
            if row.actual_elapsed_minutes is not None:
                lateness.append(row.actual_elapsed_minutes - row.horizon_minutes)
            continue

        if row.filled_at is not None:
            # Closed without a measurement. The reason is recorded on the
            # row; an unrecorded one is reported as such rather than
            # folded into a neighbouring bucket.
            health.sealed_unmeasurable += 1
            reason = (row.failure_reason or "reason not recorded").strip()
            health.sealed_by_reason[reason] = health.sealed_by_reason.get(reason, 0) + 1
            continue

        health.pending += 1
        if due is None or due > now:
            continue

        overdue_minutes = (now - due).total_seconds() / 60
        health.overdue += 1
        if (
            health.oldest_overdue_minutes is None
            or overdue_minutes > health.oldest_overdue_minutes
        ):
            health.oldest_overdue_minutes = overdue_minutes
        if overdue_minutes > lateness_tolerance_minutes(row.horizon_minutes):
            health.overdue_past_tolerance += 1

    health.last_resolution_at = last_resolution

    # Lateness is only reported with enough resolutions behind it. Three
    # slow rows and a slow worker look identical at n=3.
    health.lateness_samples = len(lateness)
    if len(lateness) >= MIN_RESOLUTIONS_FOR_LATENESS:
        health.median_lateness_minutes = _median(lateness)
        health.worst_lateness_minutes = max(lateness)

    health.status, health.detail = _verdict(health)
    return health


def _verdict(health: ResolverHealth) -> tuple[ResolverStatus, str]:
    if health.overdue_past_tolerance:
        # Past tolerance means these rows can no longer be filled honestly -
        # the resolver will seal them. Distinguish "running but losing" from
        # "not running at all", because they need different fixes.
        if health.resolved == 0:
            return ResolverStatus.STALLED, (
                f"{health.overdue_past_tolerance} row(s) are past the point where they could "
                "still be measured as the horizon they claim, and nothing has ever resolved. "
                "The worker is not running - check FORWARD_RETURNS_ENABLED and the logs. "
                "Every row past tolerance is a permanently lost observation."
            )
        return ResolverStatus.LOSING_DATA, (
            f"{health.overdue_past_tolerance} row(s) are past their lateness tolerance and "
            "will be sealed unmeasurable rather than filled with a price from the wrong "
            "time. The worker is running but not keeping up - raise "
            "FORWARD_RETURN_BATCH_LIMIT, shorten the interval, or accept the loss."
        )

    if health.overdue:
        return ResolverStatus.BEHIND, (
            f"{health.overdue} row(s) are due and not yet resolved, none past tolerance. "
            "A horizon coming due between passes is normal; watch the oldest overdue age "
            "rather than the count."
        )

    if health.resolved == 0 and health.pending:
        return ResolverStatus.IDLE, (
            f"{health.pending} row(s) scheduled, none due yet. Nothing to judge."
        )

    return ResolverStatus.HEALTHY, (
        f"{health.resolved} resolved, {health.pending} pending, nothing overdue."
    )
