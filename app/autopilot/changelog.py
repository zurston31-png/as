"""Writing and reading the record of what the loop did.

Rejections are recorded as prominently as promotions, which is the part
that is easy to skip and expensive to have skipped. An automated search
that only logs its successes will retry a rejected idea on the next cycle,
and the next, until random variation finally lets it through - and the
changelog will show a single clean promotion with no trace of the fifteen
attempts that preceded it. `attempts_against` exists to make that visible.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from app import models

logger = logging.getLogger(__name__)

PROMOTION = "promotion"
ROLLBACK = "rollback"
REMEDY = "remedy"
PROPOSAL = "proposal"
REJECTION = "rejection"


def record(
    db: Session,
    *,
    kind: str,
    target: str,
    summary: str,
    rationale: str = "",
    before: dict | None = None,
    after: dict | None = None,
    evidence: dict | None = None,
    from_version: str | None = None,
    to_version: str | None = None,
    applied: bool = False,
) -> models.AutopilotChange:
    """Append one entry. Never updates an existing row."""
    entry = models.AutopilotChange(
        kind=kind, target=target, summary=summary, rationale=rationale,
        before=before or {}, after=after or {}, evidence=evidence or {},
        from_version=from_version, to_version=to_version, applied=applied,
    )
    db.add(entry)
    db.flush()
    logger.info("autopilot %s on %s: %s", kind, target, summary)
    return entry


def mark_reverted(db: Session, original_id: int, rollback_id: int) -> None:
    """Link a rollback back to what it undid.

    Without the link, a parameter that keeps flipping looks like a series
    of independent decisions instead of the oscillation it is.
    """
    original = db.get(models.AutopilotChange, original_id)
    if original is not None:
        original.reverted_by_id = rollback_id


def history(
    db: Session, *, limit: int = 100, kind: str | None = None, target: str | None = None
) -> list[models.AutopilotChange]:
    query = db.query(models.AutopilotChange)
    if kind:
        query = query.filter(models.AutopilotChange.kind == kind)
    if target:
        query = query.filter(models.AutopilotChange.target == target)
    return query.order_by(models.AutopilotChange.occurred_at.desc()).limit(limit).all()


def attempts_against(db: Session, target: str, *, days: float = 30.0) -> int:
    """How many times the loop has already tried to change this knob.

    Feeds straight into the promotion gate's multiple-comparison
    correction. A search that runs nightly accumulates attempts even when
    each night looks like a fresh single comparison, and without counting
    across cycles the correction understates the real number of tries by
    however many nights it has been running.
    """
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    return (
        db.query(models.AutopilotChange)
        .filter(
            models.AutopilotChange.target == target,
            models.AutopilotChange.occurred_at >= cutoff,
            models.AutopilotChange.kind.in_([PROMOTION, REJECTION]),
        )
        .count()
    )


def is_oscillating(db: Session, target: str, *, window: int = 6, flips: int = 3) -> bool:
    """Has this knob been changed and reverted repeatedly?

    An oscillating parameter is not being optimised, it is being fitted to
    whatever the last few weeks happened to look like. The honest response
    is to stop touching it, not to keep searching for the value that
    finally sticks.
    """
    recent = history(db, limit=window, target=target)
    reverted = sum(1 for row in recent if row.reverted_by_id is not None)
    return reverted >= flips


def render(db: Session, *, limit: int = 30) -> str:
    """The changelog as text, newest first."""
    rows = history(db, limit=limit)
    if not rows:
        return "No automatic changes recorded."

    lines = [f"{len(rows)} most recent autopilot entries:", ""]
    for row in rows:
        stamp = row.occurred_at.strftime("%Y-%m-%d %H:%M") if row.occurred_at else "?"
        flag = "applied" if row.applied else "NOT APPLIED"
        lines.append(f"  {stamp}  [{row.kind}] {row.target}  ({flag})")
        lines.append(f"      {row.summary}")
        if row.rationale:
            lines.append(f"      why: {row.rationale}")
        if row.reverted_by_id:
            lines.append(f"      later reverted by change #{row.reverted_by_id}")
    return "\n".join(lines)
