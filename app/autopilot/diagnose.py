"""Triage over what the bot has already recorded.

Every finding here is derived from rows the bot writes anyway - pipeline
events, risk events, API health, forward returns, positions. Nothing is
inferred from logs, because a log line is a string and a row is a fact.

WHY THIS PROPOSES RATHER THAN PATCHES

Findings come in two kinds, and they are treated very differently.

    OPERATIONAL   a provider is failing, a stage is rejecting everything,
                  data has gone stale. These have bounded, reversible
                  remedies that only ever make the bot MORE conservative,
                  and app/autopilot/remedy.py may apply them unattended.

    STRUCTURAL    a threshold looks wrong, a factor is not earning its
                  weight, an exit is giving back winners. These are
                  explained and logged for a human. They are also exactly
                  the findings a search would love to "fix" automatically,
                  and doing so on the evidence available here - a
                  correlation in a few hundred rows - is how a strategy
                  gets fitted to a fortnight.

A finding is never a fix. Severity says how confident the triage is that
something is wrong, not how confident it is about the remedy.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, pipeline

OPERATIONAL = "operational"
STRUCTURAL = "structural"

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

# A stage that rejects this share of everything is either mis-tuned or
# looking at the wrong population. Either way it is worth surfacing.
CHOKE_POINT_REJECT_RATE = 0.98
MIN_STAGE_SAMPLE = 40

# Consecutive failures before a provider counts as down rather than flaky.
PROVIDER_FAILURE_STREAK = 5

# Share of closed trades that gave back a 20%+ winner before the exit
# logic is worth questioning.
GIVEBACK_SHARE = 0.30
MIN_TRADES_FOR_EXIT_REVIEW = 20


@dataclass
class Finding:
    code: str
    kind: str                 # OPERATIONAL | STRUCTURAL
    severity: str
    title: str
    detail: str
    evidence: dict = field(default_factory=dict)
    proposed: str = ""

    @property
    def auto_remediable(self) -> bool:
        """Only operational findings may be acted on without a human.

        Structural findings change what the strategy believes. Acting on
        those unattended, from the sample sizes available here, is how a
        loop fits itself to a fortnight of market.
        """
        return self.kind == OPERATIONAL

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "evidence": self.evidence,
            "proposed": self.proposed,
            "auto_remediable": self.auto_remediable,
        }


@dataclass
class Diagnosis:
    findings: list[Finding] = field(default_factory=list)
    checked_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == CRITICAL]

    @property
    def remediable(self) -> list[Finding]:
        return [f for f in self.findings if f.auto_remediable]

    @property
    def for_a_human(self) -> list[Finding]:
        return [f for f in self.findings if not f.auto_remediable]

    def headline(self) -> str:
        if not self.findings:
            return "No problems found in the recorded data."
        return (
            f"{len(self.findings)} finding(s): {len(self.critical)} critical, "
            f"{len(self.remediable)} auto-remediable, "
            f"{len(self.for_a_human)} needing a human decision."
        )

    def as_dict(self) -> dict:
        return {
            "checked_at": self.checked_at.isoformat(),
            "headline": self.headline(),
            "findings": [f.as_dict() for f in self.findings],
        }

    def render(self) -> str:
        lines = [self.headline(), ""]
        for f in self.findings:
            tag = "auto" if f.auto_remediable else "HUMAN"
            lines.append(f"  [{f.severity:<8}] [{tag:<5}] {f.title}")
            lines.append(f"      {f.detail}")
            if f.proposed:
                lines.append(f"      proposed: {f.proposed}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def _check_providers(db: Session) -> list[Finding]:
    """Upstreams that have stopped answering."""
    findings = []
    for row in db.query(models.ApiHealth).all():
        streak = getattr(row, "consecutive_failures", 0) or 0
        if streak >= PROVIDER_FAILURE_STREAK:
            findings.append(Finding(
                code="provider_down",
                kind=OPERATIONAL,
                severity=CRITICAL if streak >= PROVIDER_FAILURE_STREAK * 2 else WARNING,
                title=f"{row.service} has failed {streak} times in a row",
                detail=(
                    f"Last error: {getattr(row, 'last_error', None) or 'unrecorded'}. "
                    "Fail-closed means this currently reads as 'no candidates', which is "
                    "indistinguishable from a quiet market until someone looks here."
                ),
                evidence={"service": row.service, "consecutive_failures": streak},
                proposed="back off this provider and keep entries blocked while it is down",
            ))
    return findings


def _check_funnel(db: Session, *, window_hours: float = 24.0) -> list[Finding]:
    """Stages that reject essentially everything."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
    rows = (
        db.query(
            models.PipelineEvent.stage,
            models.PipelineEvent.passed,
            func.count(models.PipelineEvent.id),
        )
        .filter(models.PipelineEvent.occurred_at >= cutoff)
        .group_by(models.PipelineEvent.stage, models.PipelineEvent.passed)
        .all()
    )
    totals: dict[str, dict[bool, int]] = {}
    for stage, passed, count in rows:
        totals.setdefault(stage, {True: 0, False: 0})[bool(passed)] += count

    findings = []
    for stage, counts in totals.items():
        if stage in pipeline.TERMINAL_STAGES:
            continue
        total = counts[True] + counts[False]
        if total < MIN_STAGE_SAMPLE:
            continue
        reject_rate = counts[False] / total
        if reject_rate >= CHOKE_POINT_REJECT_RATE:
            findings.append(Finding(
                code="stage_choke_point",
                kind=STRUCTURAL,
                severity=WARNING,
                title=f"{stage} rejected {reject_rate:.0%} of {total} candidates",
                detail=(
                    "A stage that turns away almost everything is either mis-tuned or "
                    "reading the wrong population. Both are possible and they call for "
                    "opposite fixes, which is why this is not adjusted automatically - "
                    "loosening a gate that is correctly rejecting junk would be the worst "
                    "available outcome."
                ),
                evidence={"stage": stage, "rejected": counts[False], "total": total},
                proposed=(
                    f"inspect `research.py funnel` for {stage} and decide whether the "
                    "threshold or the candidate source is wrong"
                ),
            ))
    return findings


def _check_data_freshness(db: Session) -> list[Finding]:
    """Has anything been recorded recently at all?"""
    latest = db.query(func.max(models.PipelineEvent.occurred_at)).scalar()
    if latest is None:
        return [Finding(
            code="no_activity",
            kind=OPERATIONAL,
            severity=INFO,
            title="nothing recorded yet",
            detail="The bot has not produced a pipeline event. Normal before the first scan.",
            proposed="none - start the bot and let it run",
        )]

    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=dt.timezone.utc)
    quiet_hours = (dt.datetime.now(dt.timezone.utc) - latest).total_seconds() / 3600
    if quiet_hours >= 2:
        return [Finding(
            code="scanner_stalled",
            kind=OPERATIONAL,
            severity=CRITICAL if quiet_hours >= 6 else WARNING,
            title=f"no pipeline activity for {quiet_hours:.1f} hours",
            detail=(
                "Either the scan loop has stopped or every upstream is failing. A stalled "
                "loop looks exactly like a market with no candidates, and neither writes "
                "an error."
            ),
            evidence={"quiet_hours": round(quiet_hours, 2)},
            proposed="restart the scan loop and check provider health",
        )]
    return []


def _check_exit_quality(db: Session) -> list[Finding]:
    """Are winners being handed back?"""
    from app.analysis.postmortem import recent_postmortems

    trades = recent_postmortems(db, limit=200)
    if len(trades) < MIN_TRADES_FOR_EXIT_REVIEW:
        return []

    gave_back = [t for t in trades if t.gave_back_a_winner]
    share = len(gave_back) / len(trades)
    if share < GIVEBACK_SHARE:
        return []

    return [Finding(
        code="exits_give_back_winners",
        kind=STRUCTURAL,
        severity=WARNING,
        title=f"{share:.0%} of closed trades gave back a 20%+ winner",
        detail=(
            f"{len(gave_back)} of {len(trades)} positions were up 20% or more and closed "
            "flat or worse. That is a fact about the exit logic, not the entries - and "
            "tightening the trailing stop to fix it would also cut short the trades that "
            "recover, which the same data cannot tell apart yet."
        ),
        evidence={"gave_back": len(gave_back), "closed": len(trades)},
        proposed=(
            "run `research.py postmortem --verbose` and compare capture ratios before "
            "changing the trailing distance"
        ),
    )]


def _check_unresolved_returns(db: Session) -> list[Finding]:
    """Forward returns that will never resolve."""
    total = db.query(func.count(models.ForwardReturn.id)).scalar() or 0
    if total < 50:
        return []
    unmeasurable = (
        db.query(func.count(models.ForwardReturn.id))
        .filter(models.ForwardReturn.failure_reason.isnot(None))
        .scalar() or 0
    )
    share = unmeasurable / total
    if share < 0.30:
        return []
    return [Finding(
        code="forward_returns_unmeasurable",
        kind=OPERATIONAL,
        severity=WARNING,
        title=f"{share:.0%} of forward returns could not be measured",
        detail=(
            "The calibration dataset is being eaten by price lookups that fail. These are "
            "correctly left NULL rather than counted as 0%, so nothing is corrupted - but "
            "the sample is shrinking faster than the row count suggests."
        ),
        evidence={"unmeasurable": unmeasurable, "total": total},
        proposed="check the price provider's coverage for delisted or dead pools",
    )]


def diagnose(db: Session) -> Diagnosis:
    """Run every check over the recorded data."""
    report = Diagnosis()
    for check in (
        _check_data_freshness,
        _check_providers,
        _check_funnel,
        _check_exit_quality,
        _check_unresolved_returns,
    ):
        try:
            report.findings.extend(check(db))
        except Exception as exc:  # noqa: BLE001
            # One failing check must not blind the rest of the triage.
            report.findings.append(Finding(
                code="diagnostic_failed",
                kind=OPERATIONAL,
                severity=WARNING,
                title=f"the {check.__name__} check raised",
                detail=f"{type(exc).__name__}: {exc}",
                proposed="none - this is a bug in the triage, not in the bot",
            ))

    order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    report.findings.sort(key=lambda f: order.get(f.severity, 3))
    return report
