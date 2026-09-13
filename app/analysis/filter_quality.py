"""Was each individual pre-screen threshold right to reject what it rejected?

app/analysis/counterfactual.py already answers this per STAGE. That is not
enough for the pre-screen, because the pre-screen is five thresholds
wearing one label: a stage-level verdict saying "the pre-screen rejects
worse tokens on average" is compatible with liquidity doing all the useful
work while the age floor throws away winners. Tuning needs the split.

Every pre-screen event stores each check's name, verdict, the value that
was measured and the threshold it was measured against
(app/scanner/filters.py::FilterVerdict.as_dict), so the split is a read
over history the bot was already keeping.

WHAT THIS DELIBERATELY DOES NOT DO

It does not recommend a threshold, and it never says a filter is wrong on a
sample too small to support the claim. Below MIN_ARM_SAMPLE in either arm
it reports INSUFFICIENT DATA and stops. A tool that produced a confident
verdict from nine observations would be the most dangerous thing in this
repository, because the verdict it produces is always "you could be
trading more".

Forward returns are NET of the round-trip cost the entry would have paid.
A gross comparison flatters every rejected token, since the rejected ones
never paid the cost that would have eaten the difference.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models, pipeline
from app.analysis.calibration import round_trip_cost_pct

# Below this in EITHER arm, no comparison is reported. Matched to the
# calibration bucket floor: the question has the same shape - is this
# difference real, or is it four coin flips?
MIN_ARM_SAMPLE = 30

# A rejected token is matched to forward returns recorded for the same mint
# shortly after the rejection. Wide enough to cover the scheduling lag,
# tight enough not to capture the same mint's next scan cycle.
MATCH_WINDOW_MINUTES = 30.0

# The gain/loss bands reported per arm. Chosen to straddle what a memecoin
# entry is actually trying to catch, not as round numbers for their own
# sake: +10% roughly clears the round trip, +50% is the outcome the whole
# strategy exists for, and -20% is past where the stop would have fired.
GAIN_BANDS: tuple[float, ...] = (10.0, 25.0, 50.0)
LOSS_BAND = -20.0


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


@dataclass
class ArmStats:
    """Forward returns for one side of one check."""
    label: str
    returns: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.returns)

    @property
    def mean(self) -> float | None:
        return sum(self.returns) / self.n if self.n else None

    @property
    def median(self) -> float | None:
        return _median(self.returns)

    @property
    def win_rate_pct(self) -> float | None:
        if not self.n:
            return None
        return sum(1 for r in self.returns if r > 0) / self.n * 100

    def share_gaining(self, pct: float) -> float | None:
        if not self.n:
            return None
        return sum(1 for r in self.returns if r >= pct) / self.n * 100

    def share_losing(self, pct: float = LOSS_BAND) -> float | None:
        if not self.n:
            return None
        return sum(1 for r in self.returns if r <= pct) / self.n * 100

    @property
    def usable(self) -> bool:
        return self.n >= MIN_ARM_SAMPLE

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "mean_pct": round(self.mean, 2) if self.mean is not None else None,
            "median_pct": round(self.median, 2) if self.median is not None else None,
            "win_rate_pct": round(self.win_rate_pct, 1) if self.win_rate_pct is not None else None,
            "share_gaining": {
                f"+{int(b)}%": (
                    round(self.share_gaining(b), 1) if self.share_gaining(b) is not None else None
                )
                for b in GAIN_BANDS
            },
            "share_losing_20pct": (
                round(self.share_losing(), 1) if self.share_losing() is not None else None
            ),
            "usable": self.usable,
        }


@dataclass
class CheckQuality:
    """One pre-screen threshold, and what its two sides went on to do."""
    check_name: str
    checked: int = 0
    n_passed: int = 0
    n_failed: int = 0
    threshold: float | None = None
    passed_arm: ArmStats = field(default_factory=lambda: ArmStats("passed this check"))
    failed_arm: ArmStats = field(default_factory=lambda: ArmStats("failed this check"))
    # Measured values, for describing where the threshold actually sits in
    # the distribution rather than only whether it fired.
    failed_values: list[float] = field(default_factory=list)

    @property
    def fail_rate_pct(self) -> float | None:
        return (self.n_failed / self.checked * 100) if self.checked else None

    @property
    def measured(self) -> bool:
        return self.passed_arm.usable and self.failed_arm.usable

    def verdict(self) -> str:
        if not self.measured:
            return (
                f"INSUFFICIENT DATA - {self.passed_arm.n} passed / {self.failed_arm.n} failed "
                f"with measured outcomes, need {MIN_ARM_SAMPLE} in each arm"
            )
        passed_median = self.passed_arm.median or 0.0
        failed_median = self.failed_arm.median or 0.0
        gap = passed_median - failed_median
        if gap > 0:
            return (
                f"keeping the better half - median {passed_median:+.1f}% kept vs "
                f"{failed_median:+.1f}% rejected ({gap:+.1f} pts)"
            )
        return (
            f"REJECTING THE BETTER HALF - median {failed_median:+.1f}% rejected vs "
            f"{passed_median:+.1f}% kept ({-gap:+.1f} pts the wrong way). Evidence to "
            f"investigate this threshold, NOT permission to move it"
        )

    def as_dict(self) -> dict:
        return {
            "check": self.check_name,
            "checked": self.checked,
            "passed": self.n_passed,
            "failed": self.n_failed,
            "fail_rate_pct": round(self.fail_rate_pct, 1) if self.fail_rate_pct is not None else None,
            "threshold": self.threshold,
            "median_failed_value": _median(self.failed_values),
            "measured": self.measured,
            "verdict": self.verdict(),
            "passed_arm": self.passed_arm.as_dict(),
            "failed_arm": self.failed_arm.as_dict(),
        }


@dataclass
class FilterQualityReport:
    horizon_minutes: int
    window_hours: float | None
    checks: list[CheckQuality] = field(default_factory=list)
    events_seen: int = 0
    events_with_outcome: int = 0

    @property
    def measurable(self) -> list[CheckQuality]:
        return [c for c in self.checks if c.measured]

    @property
    def flagged(self) -> list[CheckQuality]:
        """Checks whose rejected side outperformed the side they kept."""
        return [
            c for c in self.measurable
            if (c.failed_arm.median or 0.0) > (c.passed_arm.median or 0.0)
        ]

    def summary(self) -> str:
        if not self.events_seen:
            return "No pre-screen events in this window."
        if not self.measurable:
            return (
                f"INSUFFICIENT DATA - {self.events_seen} pre-screen events, "
                f"{self.events_with_outcome} with a measured {self.horizon_minutes}min outcome. "
                f"No check has {MIN_ARM_SAMPLE} in both arms yet, so no threshold can be judged. "
                f"This is the correct answer, not a gap to route around."
            )
        if self.flagged:
            names = ", ".join(c.check_name for c in self.flagged)
            return (
                f"{len(self.measurable)} check(s) measurable; {names} rejected tokens that "
                f"outperformed the ones kept. Investigate before changing anything - a single "
                f"horizon on one window is a screen, not a finding."
            )
        return (
            f"{len(self.measurable)} check(s) measurable, and each kept the better half at "
            f"{self.horizon_minutes}min."
        )

    def as_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "window_hours": self.window_hours,
            "events_seen": self.events_seen,
            "events_with_outcome": self.events_with_outcome,
            "min_arm_sample": MIN_ARM_SAMPLE,
            "summary": self.summary(),
            "flagged": [c.check_name for c in self.flagged],
            "checks": [c.as_dict() for c in self.checks],
        }


def _aware(value: dt.datetime) -> dt.datetime:
    return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)


def build_filter_quality(
    db: Session,
    *,
    horizon_minutes: int = 240,
    window_hours: float | None = None,
    strategy_version: str | None = None,
) -> FilterQualityReport:
    """Per-check forward-return comparison over recorded pre-screen events.

    Matching is by mint plus time rather than by pipeline_event_id, because
    the two arms anchor differently: a rejected token's forward returns hang
    off its PRESCREEN event, while an accepted one's are scheduled later at
    TECHNICAL_SCORE. Keying on the event id would silently compare only the
    rejects against nothing.
    """
    report = FilterQualityReport(horizon_minutes=horizon_minutes, window_hours=window_hours)

    query = db.query(models.PipelineEvent).filter(models.PipelineEvent.stage == pipeline.PRESCREEN)
    if window_hours is not None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
        query = query.filter(models.PipelineEvent.occurred_at >= cutoff)
    if strategy_version is not None:
        query = query.filter(models.PipelineEvent.strategy_version == strategy_version)
    events = query.all()
    report.events_seen = len(events)
    if not events:
        return report

    outcomes = (
        db.query(models.ForwardReturn)
        .filter(
            models.ForwardReturn.horizon_minutes == horizon_minutes,
            models.ForwardReturn.return_pct.isnot(None),
        )
        .all()
    )
    by_mint: dict[str, list[models.ForwardReturn]] = {}
    for row in outcomes:
        by_mint.setdefault(row.token_address, []).append(row)

    cost = round_trip_cost_pct()
    checks: dict[str, CheckQuality] = {}

    for event in events:
        mint = event.token_address
        if not mint:
            continue
        occurred = _aware(event.occurred_at)
        match = None
        for row in by_mint.get(mint, ()):
            delta = (_aware(row.observed_at) - occurred).total_seconds() / 60
            if -1.0 <= delta <= MATCH_WINDOW_MINUTES:
                match = row
                break
        if match is None:
            continue
        report.events_with_outcome += 1
        # Net, so the rejected arm is charged the cost it never paid.
        net = match.return_pct - cost

        for raw in (event.detail or {}).get("checks", []):
            name = raw.get("name")
            if not name:
                continue
            quality = checks.setdefault(name, CheckQuality(check_name=name))
            quality.checked += 1
            if raw.get("threshold") is not None:
                quality.threshold = raw["threshold"]
            if raw.get("passed"):
                quality.n_passed += 1
                quality.passed_arm.returns.append(net)
            else:
                quality.n_failed += 1
                quality.failed_arm.returns.append(net)
                if isinstance(raw.get("value"), (int, float)):
                    quality.failed_values.append(float(raw["value"]))

    report.checks = sorted(checks.values(), key=lambda c: c.n_failed, reverse=True)
    return report
