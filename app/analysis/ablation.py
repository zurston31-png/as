"""What the pre-screen would have let through with one check removed.

Every pre-screen check runs on every candidate and every result is
recorded (app/scanner/filters.py records all five, not just the first
failure), so this needs no simulation and no assumptions: removing a
check from a recorded verdict is a set operation over rows already on
disk.

THE QUESTION IT ANSWERS, AND THE ONE IT DOES NOT

Answers: how many mints does each check turn away that NOTHING ELSE would
have caught? That is the check's marginal contribution, and it is very
different from its raw rejection count. A check that rejects 480 mints of
which 478 also fail three other checks is doing almost nothing - the
funnel would look near-identical without it. One that rejects 12 mints
nobody else touches is the only thing standing between those 12 and the
buy path.

Does not answer: whether turning them away was right. That needs the
forward returns of the rejected tokens, which is
app/analysis/filter_quality.py's job. A check with a large marginal
contribution might be the bot's best filter or its worst; this module
cannot tell, and says so rather than implying the count is a verdict.

WHY "UNIQUELY REJECTED" IS THE HEADLINE AND NOT THE OVERLAP MATRIX

The overlap between five checks has 31 non-empty cells and this bot will
never fill them with enough mints to read. Counting how many candidates
each check is solely responsible for is the same information at the
resolution the data actually supports.

NOTHING IS TUNED HERE. No threshold moves, no check is disabled, and
nothing in this module is wired to anything that trades. It is a counting
report over recorded events.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.pipeline import PRESCREEN

# Below this many evaluated mints, per-check shares are withheld: a
# "100% uniquely rejected" built on two tokens is noise wearing a
# percentage sign.
MIN_MINTS_TO_REPORT = 10


@dataclass
class CheckAblation:
    """One pre-screen check's marginal contribution."""

    check: str
    evaluated: int = 0          # distinct mints this check ran on
    rejected: int = 0           # distinct mints it failed
    uniquely_rejected: int = 0  # ...that no other check also failed
    redundant_rejections: int = 0   # rejected, but caught by something else too

    @property
    def marginal_share(self) -> float | None:
        """Of everything this check rejects, how much only it catches."""
        if self.rejected < MIN_MINTS_TO_REPORT:
            return None
        return self.uniquely_rejected / self.rejected

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["marginal_share_pct"] = (
            round(self.marginal_share * 100, 1) if self.marginal_share is not None else None
        )
        return payload


@dataclass
class AblationReport:
    window_hours: float | None
    mints_evaluated: int = 0
    mints_passing_all: int = 0
    checks: list[CheckAblation] = field(default_factory=list)
    note: str = ""

    @property
    def has_data(self) -> bool:
        return self.mints_evaluated > 0

    def would_pass_without(self, check: str) -> int:
        """How many mints clear the pre-screen if `check` is removed.

        The current pass count plus the ones only that check stopped. No
        other check's verdict changes, because every check already ran on
        every candidate independently.
        """
        row = next((c for c in self.checks if c.check == check), None)
        if row is None:
            return self.mints_passing_all
        return self.mints_passing_all + row.uniquely_rejected

    def as_dict(self) -> dict:
        return {
            "window_hours": self.window_hours,
            "mints_evaluated": self.mints_evaluated,
            "mints_passing_all": self.mints_passing_all,
            "checks": [c.as_dict() for c in self.checks],
            "would_pass_without": {
                c.check: self.would_pass_without(c.check) for c in self.checks
            },
            "note": self.note,
        }


def build_ablation(
    db: Session, *, window_hours: float | None = 168.0, strategy_version: str | None = None
) -> AblationReport:
    """Count each pre-screen check's marginal contribution, per mint."""
    query = db.query(models.PipelineEvent).filter(models.PipelineEvent.stage == PRESCREEN)
    if window_hours is not None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
        query = query.filter(models.PipelineEvent.occurred_at >= cutoff)
    if strategy_version:
        query = query.filter(models.PipelineEvent.strategy_version == strategy_version)

    events = query.order_by(models.PipelineEvent.occurred_at.asc()).all()
    report = AblationReport(window_hours=window_hours)
    if not events:
        report.note = (
            "no pre-screen events recorded in this window. Nothing to ablate - this is an "
            "absence of data, not a finding that every check is redundant."
        )
        return report

    # LAST verdict per mint, not every evaluation. A token re-screened
    # forty times as its liquidity moved would otherwise contribute forty
    # votes, and the most-rescanned token would decide the answer.
    latest: dict[str, dict] = {}
    without_detail = 0
    for event in events:
        if not event.token_address:
            continue
        checks = (event.detail or {}).get("checks") or []
        if not checks:
            without_detail += 1
            continue
        latest[event.token_address] = {
            c["name"]: bool(c.get("passed")) for c in checks if c.get("name")
        }

    if not latest:
        report.note = (
            f"{len(events)} pre-screen event(s) recorded, none carrying per-check detail. "
            "Ablation needs every check's individual verdict; these rows predate that "
            "being recorded."
        )
        return report

    report.mints_evaluated = len(latest)

    per_check: dict[str, CheckAblation] = {}
    for verdicts in latest.values():
        failed = [name for name, ok in verdicts.items() if not ok]
        if not failed:
            report.mints_passing_all += 1
        for name, ok in verdicts.items():
            row = per_check.setdefault(name, CheckAblation(check=name))
            row.evaluated += 1
            if ok:
                continue
            row.rejected += 1
            if len(failed) == 1:
                row.uniquely_rejected += 1
            else:
                row.redundant_rejections += 1

    report.checks = sorted(
        per_check.values(), key=lambda c: (-c.uniquely_rejected, -c.rejected, c.check)
    )

    if without_detail:
        report.note = (
            f"{without_detail} pre-screen event(s) carry no per-check detail and are "
            "excluded - they predate that being recorded, and guessing which checks they "
            "failed would be inventing data."
        )
    return report
