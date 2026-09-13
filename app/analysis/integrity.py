"""Finding observations that must not be counted.

Every metric downstream is an average over rows. A duplicate, a stale
price, or a timestamp from the future does not make an average slightly
wrong - it makes it wrong in a direction nobody checked, and it does so
silently, because a corrupted row looks exactly like a clean one once it
has been summed.

So this runs BEFORE the statistics, not after, and its output is a set of
row ids to exclude plus the reason each was excluded. Excluded rows are
never deleted: a dataset that quietly shrinks is worse than one with known
holes, and the exclusion count is itself a signal about the data pipeline.

WHAT COUNTS AS CORRUPTION HERE

Only things that cannot be true. A -95% return is not corruption, it is a
memecoin. A price that has not moved in six hours while volume was
reported is corruption, because both cannot be so. The bar is deliberately
"impossible", not "surprising" - a filter tuned to remove surprising
observations removes exactly the tail that carries the result.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models

# A forward return resolved more than this far from its due time was
# measured against the wrong moment.
MAX_RESOLUTION_DRIFT_MINUTES = 30.0

# Price moves beyond this in one horizon are possible in memecoins but are
# far more often a decimals error or a bad parse. Flagged, not deleted -
# and counted separately so the tail can be inspected rather than assumed.
IMPLAUSIBLE_GAIN_PCT = 10_000.0     # +100x
IMPLAUSIBLE_LOSS_PCT = -99.9


@dataclass
class Exclusion:
    table: str
    row_id: int
    code: str
    detail: str

    def as_dict(self) -> dict:
        return {"table": self.table, "row_id": self.row_id,
                "code": self.code, "detail": self.detail}


@dataclass
class IntegrityReport:
    checked: int = 0
    exclusions: list[Exclusion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def excluded_ids(self) -> set[int]:
        return {e.row_id for e in self.exclusions}

    @property
    def by_code(self) -> dict[str, int]:
        return dict(Counter(e.code for e in self.exclusions))

    @property
    def clean(self) -> int:
        return max(self.checked - len(self.excluded_ids), 0)

    @property
    def exclusion_rate(self) -> float | None:
        return (len(self.excluded_ids) / self.checked) if self.checked else None

    def verdict(self) -> str:
        if not self.checked:
            return "No observations to check yet."
        rate = self.exclusion_rate or 0.0
        head = (
            f"{self.clean} of {self.checked} observations usable "
            f"({rate:.1%} excluded)."
        )
        if rate >= 0.20:
            head += (
                " Over a fifth of the dataset is unusable - treat every metric below as "
                "provisional. The pipeline is losing more data than the statistics can "
                "absorb, and fixing that comes before reading any result."
            )
        elif rate >= 0.05:
            head += " Worth watching, but the remaining sample is still readable."
        return head

    def as_dict(self) -> dict:
        return {
            "checked": self.checked,
            "clean": self.clean,
            "excluded": len(self.excluded_ids),
            "exclusion_rate_pct": (
                round(self.exclusion_rate * 100, 2) if self.exclusion_rate is not None else None
            ),
            "verdict": self.verdict(),
            "by_code": self.by_code,
            "warnings": list(self.warnings),
            "examples": [e.as_dict() for e in self.exclusions[:20]],
        }

    def render(self) -> str:
        lines = [self.verdict(), ""]
        if self.by_code:
            lines.append("  excluded by reason:")
            for code, count in sorted(self.by_code.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {count:>6}  {code}")
        for w in self.warnings:
            lines.append(f"\n  WARNING: {w}")
        lines.append(
            "\n  Excluded rows are never deleted. A dataset that quietly shrinks is worse\n"
            "  than one with known holes, and the exclusion count is itself a signal about\n"
            "  the pipeline."
        )
        return "\n".join(lines)


def _aware(moment: dt.datetime | None) -> dt.datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def check_forward_returns(db: Session) -> IntegrityReport:
    """Validate the calibration dataset row by row."""
    report = IntegrityReport()
    rows = db.query(models.ForwardReturn).all()
    report.checked = len(rows)
    if not rows:
        return report

    now = dt.datetime.now(dt.timezone.utc)
    seen: dict[tuple, int] = {}
    missing_regime = 0

    for row in rows:
        observed = _aware(row.observed_at)
        due = _aware(row.due_at)
        filled = _aware(row.filled_at)

        # --- duplicates -------------------------------------------------
        # One row per (candidate, horizon). Two means the scheduler ran
        # twice, and the duplicate would double-weight that candidate in
        # every average it appears in.
        key = (row.pipeline_event_id, row.token_address, row.horizon_minutes)
        if key in seen:
            report.exclusions.append(Exclusion(
                "forward_returns", row.id, "duplicate_outcome",
                f"same candidate and horizon as row {seen[key]}",
            ))
            continue
        seen[key] = row.id

        # --- impossible timestamps --------------------------------------
        if observed and observed > now + dt.timedelta(minutes=5):
            report.exclusions.append(Exclusion(
                "forward_returns", row.id, "future_timestamp",
                f"observed_at {observed.isoformat()} is in the future",
            ))
            continue
        if observed and due and due < observed:
            report.exclusions.append(Exclusion(
                "forward_returns", row.id, "impossible_timestamp",
                "due_at is before observed_at",
            ))
            continue

        # --- look-ahead ---------------------------------------------------
        # A row resolved BEFORE its horizon elapsed was measured against a
        # price that had not happened yet at decision time. This is the
        # single most damaging corruption possible here: it does not add
        # noise, it manufactures an edge.
        if filled and due and filled < due - dt.timedelta(minutes=1):
            report.exclusions.append(Exclusion(
                "forward_returns", row.id, "future_data_leakage",
                f"resolved {(due - filled).total_seconds() / 60:.1f}m before its horizon elapsed",
            ))
            continue
        if filled and due:
            drift = (filled - due).total_seconds() / 60
            if drift > MAX_RESOLUTION_DRIFT_MINUTES:
                report.exclusions.append(Exclusion(
                    "forward_returns", row.id, "stale_resolution",
                    f"resolved {drift:.0f}m after its horizon - measured at the wrong moment",
                ))
                continue

        # --- impossible prices --------------------------------------------
        if row.price_at_signal is not None and row.price_at_signal <= 0:
            report.exclusions.append(Exclusion(
                "forward_returns", row.id, "invalid_price",
                f"signal price {row.price_at_signal}",
            ))
            continue
        if row.return_pct is not None and not (
            IMPLAUSIBLE_LOSS_PCT <= row.return_pct <= IMPLAUSIBLE_GAIN_PCT
        ):
            report.exclusions.append(Exclusion(
                "forward_returns", row.id, "implausible_move",
                f"{row.return_pct:+.1f}% - far more often a decimals or parse error "
                "than a real move",
            ))
            continue

        # --- MFE/MAE coherence ---------------------------------------------
        # The path cannot be inside the endpoints.
        if row.return_pct is not None:
            if row.max_favorable_pct is not None and row.max_favorable_pct < row.return_pct - 0.01:
                report.exclusions.append(Exclusion(
                    "forward_returns", row.id, "incoherent_path",
                    f"MFE {row.max_favorable_pct:+.2f}% below the close {row.return_pct:+.2f}%",
                ))
                continue
            if row.max_adverse_pct is not None and row.max_adverse_pct > row.return_pct + 0.01:
                report.exclusions.append(Exclusion(
                    "forward_returns", row.id, "incoherent_path",
                    f"MAE {row.max_adverse_pct:+.2f}% above the close {row.return_pct:+.2f}%",
                ))
                continue

        if row.return_pct is not None and not row.market_regime:
            missing_regime += 1

    if missing_regime:
        report.warnings.append(
            f"{missing_regime} resolved row(s) carry no market regime. They are still "
            "usable for overall metrics but cannot enter any per-regime comparison, "
            "which is what the promotion gate's consistency bar reads."
        )
    return report


def check_positions(db: Session) -> IntegrityReport:
    """Validate closed positions used for post-mortems."""
    report = IntegrityReport()
    rows = (
        db.query(models.Position)
        .filter(models.Position.status == models.PositionStatus.CLOSED.value)
        .all()
    )
    report.checked = len(rows)

    for row in rows:
        opened, closed = _aware(row.opened_at), _aware(row.closed_at)
        if closed and opened and closed < opened:
            report.exclusions.append(Exclusion(
                "positions", row.id, "impossible_timestamp",
                "closed before it opened",
            ))
            continue
        if row.entry_price is not None and row.entry_price <= 0:
            report.exclusions.append(Exclusion(
                "positions", row.id, "invalid_price",
                f"entry price {row.entry_price}",
            ))
            continue
        # A closed position with no exit leg is an incomplete post-mortem:
        # the return is unknown, so including it would average a real
        # number against a missing one.
        legs = (
            db.query(models.Trade)
            .filter(models.Trade.position_id == row.id, models.Trade.side == "sell")
            .count()
        )
        if legs == 0:
            report.exclusions.append(Exclusion(
                "positions", row.id, "incomplete_postmortem",
                "closed with no sell leg - the realised return is unknown",
            ))
            continue
        if row.highest_price_since_entry is not None and row.lowest_price_since_entry is not None:
            if row.highest_price_since_entry < row.lowest_price_since_entry:
                report.exclusions.append(Exclusion(
                    "positions", row.id, "incoherent_path",
                    "high-water mark below the low-water mark",
                ))
                continue

    without_regime = sum(1 for r in rows if not r.market_regime)
    if without_regime:
        report.warnings.append(
            f"{without_regime} closed position(s) carry no entry regime - opened before "
            "regime persistence existed, so they cannot enter a per-regime comparison."
        )
    return report


def check_all(db: Session) -> dict[str, IntegrityReport]:
    return {
        "forward_returns": check_forward_returns(db),
        "positions": check_positions(db),
    }


def usable_forward_returns(db: Session, **filters) -> list[models.ForwardReturn]:
    """Resolved forward returns with corrupted rows removed.

    The intended entry point for anything that computes a statistic. Going
    round it and querying ForwardReturn directly is how a duplicate ends
    up double-weighted in an expectancy nobody re-checks.
    """
    report = check_forward_returns(db)
    excluded = report.excluded_ids

    query = db.query(models.ForwardReturn).filter(
        models.ForwardReturn.return_pct.isnot(None)
    )
    for column, value in filters.items():
        query = query.filter(getattr(models.ForwardReturn, column) == value)
    return [row for row in query.all() if row.id not in excluded]
