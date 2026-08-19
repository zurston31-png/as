"""Per-upstream API health tracking.

Every gate in this bot is fail-closed: no data means reject the trade. That
is the right behaviour, and it has one bad consequence — a dead upstream
and a genuinely quiet market produce *identical* output. The bot stops
trading, the logs say "rejected: no data", and nothing anywhere says "your
DexScreener key is being throttled and has been for six hours".

This module is the answer to that. It records, per service, the last
success, the last failure and its reason, and a consecutive-failure count,
so the dashboard can show DEGRADED next to a source instead of leaving the
operator to infer it from an absence.

Two-layer on purpose:

  IN MEMORY   the hot path. Every HTTP call updates a counter, and a
              counter update must not cost a database write - the scanner
              alone makes dozens of calls a minute.
  PERSISTED   debounced to the ApiHealth table, so health survives a
              restart and `scripts/` can read it without being in the same
              process as the bot.

Nothing here ever changes what the bot does. A degraded service does not
relax a gate, and does not get retried harder; it gets *reported*. Health
information that fed back into trading decisions would be a way for a
broken API to lower the bar, which is exactly backwards.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models

logger = logging.getLogger(__name__)

# How long a service can go without a successful call before it is called
# degraded. Generous relative to the poll intervals: a single miss is
# normal on a free public tier and should not light up the dashboard.
STALE_AFTER_SECONDS = 600.0

# Consecutive failures that mark a service degraded regardless of recency.
DEGRADED_AFTER_CONSECUTIVE_FAILURES = 3

# At most one database write per service per this interval. The counters
# are exact in memory; the persisted copy is allowed to lag.
PERSIST_EVERY_SECONDS = 30.0


@dataclass
class ServiceHealth:
    service: str
    last_success_at: dt.datetime | None = None
    last_failure_at: dt.datetime | None = None
    last_error: str | None = None
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    _last_persisted_at: dt.datetime | None = field(default=None, repr=False)

    @property
    def total_calls(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float | None:
        """None when the service has not been called at all this session -
        which is not the same as a 0% success rate."""
        return (self.success_count / self.total_calls * 100) if self.total_calls else None

    def seconds_since_success(self, now: dt.datetime | None = None) -> float | None:
        if self.last_success_at is None:
            return None
        now = now or dt.datetime.now(dt.timezone.utc)
        last = self.last_success_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=dt.timezone.utc)
        return (now - last).total_seconds()

    def status(self, now: dt.datetime | None = None) -> str:
        """One of "unused", "ok", "degraded", "down".

        "unused" is deliberately distinct from "ok": a service nothing has
        called yet is not healthy, it is unknown, and showing it green
        would be a claim the bot cannot support.
        """
        if self.total_calls == 0:
            return "unused"
        if self.consecutive_failures >= DEGRADED_AFTER_CONSECUTIVE_FAILURES:
            return "down"
        since = self.seconds_since_success(now)
        if since is None:
            return "down"           # called, never once succeeded
        if since > STALE_AFTER_SECONDS or self.consecutive_failures:
            return "degraded"
        return "ok"

    def as_dict(self, now: dt.datetime | None = None) -> dict:
        return {
            "service": self.service,
            "status": self.status(now),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": (
                round(self.success_rate, 1) if self.success_rate is not None else None
            ),
            "seconds_since_success": (
                round(self.seconds_since_success(now)) if self.last_success_at else None
            ),
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "last_error": self.last_error,
        }


_registry: dict[str, ServiceHealth] = {}
_lock = threading.Lock()


def record_success(service: str) -> ServiceHealth:
    with _lock:
        health = _registry.setdefault(service, ServiceHealth(service))
        health.last_success_at = dt.datetime.now(dt.timezone.utc)
        health.success_count += 1
        health.consecutive_failures = 0
        return health


def record_failure(service: str, error: str) -> ServiceHealth:
    with _lock:
        health = _registry.setdefault(service, ServiceHealth(service))
        health.last_failure_at = dt.datetime.now(dt.timezone.utc)
        # Truncated: some upstreams return an entire HTML error page, and a
        # dashboard cell is not the place for it.
        health.last_error = (error or "")[:500]
        health.failure_count += 1
        health.consecutive_failures += 1
        if health.consecutive_failures == DEGRADED_AFTER_CONSECUTIVE_FAILURES:
            logger.warning(
                "%s has failed %d times in a row - the bot will keep rejecting trades that "
                "need it, which looks identical to a quiet market. Last error: %s",
                service, health.consecutive_failures, health.last_error,
            )
        return health


def snapshot() -> list[ServiceHealth]:
    """Every tracked service, worst first, so problems sort to the top."""
    order = {"down": 0, "degraded": 1, "unused": 2, "ok": 3}
    with _lock:
        return sorted(_registry.values(), key=lambda h: (order[h.status()], h.service))


def get(service: str) -> ServiceHealth | None:
    with _lock:
        return _registry.get(service)


def reset() -> None:
    """Clear the registry. For tests - the bot itself never calls this."""
    with _lock:
        _registry.clear()


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def persist(db: Session, *, force: bool = False) -> int:
    """Flush the in-memory registry to the ApiHealth table.

    Debounced per service so a busy scanner cycle does not turn into a
    write per HTTP call. Returns how many rows were written. The caller
    commits.
    """
    now = dt.datetime.now(dt.timezone.utc)
    written = 0

    with _lock:
        entries = list(_registry.values())

    for health in entries:
        if not force and health._last_persisted_at is not None:
            elapsed = (now - health._last_persisted_at).total_seconds()
            if elapsed < PERSIST_EVERY_SECONDS:
                continue

        row = db.query(models.ApiHealth).filter_by(service=health.service).first()
        if row is None:
            row = models.ApiHealth(service=health.service)
            db.add(row)
            # The session runs with autoflush=False, so without this the
            # pending INSERT stays invisible to the next lookup and a
            # second persist() before the caller commits inserts a
            # duplicate - which the unique constraint on `service` then
            # rejects, poisoning the whole transaction.
            db.flush()

        row.last_success_at = health.last_success_at
        row.last_failure_at = health.last_failure_at
        row.last_error = health.last_error
        row.success_count = health.success_count
        row.failure_count = health.failure_count
        row.consecutive_failures = health.consecutive_failures
        health._last_persisted_at = now
        written += 1

    return written


def load(db: Session) -> int:
    """Seed the in-memory registry from the table at startup.

    Without this, a restart makes every service look "unused" and the
    dashboard loses the fact that something has been down for an hour.
    Counts are restored as-is; they are lifetime totals, not per-session.
    """
    rows = db.query(models.ApiHealth).all()
    with _lock:
        for row in rows:
            _registry[row.service] = ServiceHealth(
                service=row.service,
                last_success_at=row.last_success_at,
                last_failure_at=row.last_failure_at,
                last_error=row.last_error,
                success_count=row.success_count or 0,
                failure_count=row.failure_count or 0,
                consecutive_failures=row.consecutive_failures or 0,
            )
    return len(rows)
