"""One view of how much usable data exists, across every source at once.

The individual coverage numbers already exist - forward returns know their
own fill rate, the shadow resolver knows how many positions closed, the
integrity checker knows what to exclude. Spread across four commands they
are easy to read optimistically: each one looks reasonable on its own, and
nobody multiplies them together.

This puts them on one page and states the product. A run with 70% forward
coverage, 60% of shadow positions resolved and 5% excluded for integrity
does not have "mostly fine" data - it has a paired sample considerably
smaller than the row counts suggest, and the milestone is further away
than any single number implies.

WHAT IT IS FOR

Deciding whether to keep waiting. Not whether the strategy works - no
number here is about returns, deliberately, because a screen that shows
coverage next to expectancy invites reading the second before the first is
adequate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models
from app.analysis.collection import TARGET_PAIRS, check_collection
from app.analysis.forward_returns import coverage as forward_coverage
from app.analysis.integrity import check_all
from app.shadow.challengers import CHAMPION_ID
from app.shadow.resolver import coverage as shadow_coverage
from app.strategy.version import current_label


@dataclass
class Source:
    """One dataset, and how much of it can actually be used."""
    name: str
    total: int
    usable: int
    pending: int = 0
    unusable: int = 0
    note: str = ""

    @property
    def usable_pct(self) -> float:
        return (self.usable / self.total * 100) if self.total else 0.0

    def as_dict(self) -> dict:
        return {"name": self.name, "total": self.total, "usable": self.usable,
                "pending": self.pending, "unusable": self.unusable,
                "usable_pct": round(self.usable_pct, 1), "note": self.note}


@dataclass
class DatasetReport:
    strategy_version: str | None = None
    sources: list[Source] = field(default_factory=list)
    paired: dict[str, int] = field(default_factory=dict)
    target_pairs: int = TARGET_PAIRS
    regimes: dict[str, int] = field(default_factory=dict)
    liquidity: dict[str, int] = field(default_factory=dict)
    integrity_exclusions: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def slowest_arm(self) -> tuple[str, int] | None:
        if not self.paired:
            return None
        return min(self.paired.items(), key=lambda kv: kv[1])

    @property
    def progress_pct(self) -> float:
        arm = self.slowest_arm
        return min(arm[1] / self.target_pairs * 100, 100.0) if arm else 0.0

    @property
    def contrasting_axes(self) -> list[str]:
        """Axes with at least two values represented.

        The consistency bar needs contrast WITHIN an axis. Counting distinct
        full labels instead would let one condition satisfy it three times
        over - a bug this project has already had once.
        """
        axes = []
        if len({k.split("/")[0] for k in self.regimes if k}) >= 2:
            axes.append("trend")
        if len({k.split("/")[1] for k in self.regimes if k and len(k.split("/")) > 1}) >= 2:
            axes.append("volatility")
        if len(self.liquidity) >= 2:
            axes.append("liquidity")
        return axes

    def verdict(self) -> str:
        if self.failures:
            return (
                f"{len(self.failures)} collection check(s) are FAILING. Time spent collecting "
                "right now is producing data that cannot be used."
            )
        arm = self.slowest_arm
        if arm is None:
            return (
                "No paired observations yet. Nothing on this page is a limitation of the "
                "strategy - the run has not produced a sample."
            )
        if not self.contrasting_axes:
            return (
                f"{arm[1]} paired opportunities on the slowest arm ({arm[0]}), but only one "
                "market condition seen so far. The promotion gate's consistency bar cannot be "
                "satisfied by any number of observations from a single regime."
            )
        return (
            f"{arm[1]} / {self.target_pairs} paired opportunities on the slowest arm "
            f"({arm[0]}), {self.progress_pct:.0f}% of the milestone, with contrast on "
            f"{', '.join(self.contrasting_axes)}."
        )

    def as_dict(self) -> dict:
        return {
            "strategy_version": self.strategy_version,
            "sources": [s.as_dict() for s in self.sources],
            "paired": dict(self.paired),
            "target_pairs": self.target_pairs,
            "progress_pct": round(self.progress_pct, 1),
            "regimes": dict(self.regimes),
            "liquidity": dict(self.liquidity),
            "contrasting_axes": self.contrasting_axes,
            "integrity_exclusions": self.integrity_exclusions,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "verdict": self.verdict(),
        }


def build_dataset_report(db: Session) -> DatasetReport:
    """Assemble the coverage picture. Read-only."""
    report = DatasetReport(strategy_version=current_label())

    forward = forward_coverage(db)
    report.sources.append(Source(
        name="forward returns",
        total=forward["total"], usable=forward["resolved"],
        pending=forward["pending"], unusable=forward["unmeasurable"],
        note="every scored candidate, taken or not - the answer key for calibration",
    ))

    shadow = shadow_coverage(db)
    report.sources.append(Source(
        name="shadow positions",
        total=shadow["positions"], usable=shadow["resolved"],
        pending=shadow["open"], unusable=shadow["unmeasurable"],
        note="hypothetical entries walked through the shared exit rule",
    ))

    horizons_total = db.query(models.ShadowHorizonReturn).count()
    horizons_measured = db.query(models.ShadowHorizonReturn).filter(
        models.ShadowHorizonReturn.return_pct.isnot(None)).count()
    report.sources.append(Source(
        name="shadow horizons",
        total=horizons_total, usable=horizons_measured,
        unusable=horizons_total - horizons_measured,
        note="fixed-horizon outcomes, independent of the exit rule",
    ))

    decisions = db.query(models.ShadowDecision).count()
    report.sources.append(Source(
        name="shadow decisions",
        total=decisions, usable=decisions,
        note="one row per strategy per opportunity, refusals included",
    ))

    # Regime spread, from the decisions themselves rather than from a
    # separate count - the grouping the gate reads is the one on these rows.
    for label, n in db.query(
        models.ShadowDecision.market_regime, func.count()
    ).group_by(models.ShadowDecision.market_regime).all():
        if label:
            report.regimes[label] = n
    for label, n in db.query(
        models.ShadowDecision.liquidity_regime, func.count()
    ).group_by(models.ShadowDecision.liquidity_regime).all():
        if label:
            report.liquidity[label] = n

    health = check_collection(db)
    report.paired = dict(health.paired)
    report.failures = [c.name for c in health.checks if c.status == "FAIL"]
    report.warnings = [c.name for c in health.checks if c.status == "WARN"]

    report.integrity_exclusions = sum(
        len(r.exclusions) for r in check_all(db).values()
    )
    return report


def milestone_gap(report: DatasetReport) -> dict:
    """What is still missing before the promotion gate can be run.

    Stated as a shortfall rather than a percentage, because "68% of the
    way" reads as nearly done and "160 more paired opportunities on
    loose-60" reads as what it is.
    """
    arm = report.slowest_arm
    gaps = {}
    if arm is None:
        gaps["paired"] = f"no paired observations yet (target {report.target_pairs} per arm)"
    elif arm[1] < report.target_pairs:
        gaps["paired"] = (
            f"{report.target_pairs - arm[1]} more paired opportunities on {arm[0]}"
        )
    missing_axes = [a for a in ("trend", "volatility", "liquidity")
                    if a not in report.contrasting_axes]
    if missing_axes:
        gaps["contrast"] = (
            f"a second value on: {', '.join(missing_axes)} - the consistency bar needs "
            "contrast within an axis, not a count of labels"
        )
    if report.failures:
        gaps["health"] = f"failing collection checks: {', '.join(report.failures)}"
    return gaps


def champion_decisions(db: Session) -> int:
    return db.query(models.ShadowDecision).filter(
        models.ShadowDecision.strategy_id == CHAMPION_ID).count()
