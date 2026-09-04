"""How much data exists, and how much is still needed before each question
can be answered.

Every research tool in this project reports INSUFFICIENT DATA until it has
enough measured outcomes, which is correct and also unhelpful on its own:
"not yet" gives no idea whether the answer is a day away or a month away,
and the natural response to a wall of "not yet" is to stop checking or to
lower the bar. This turns the same floors into a progress readout.

WHY IT ESTIMATES A TIME AND WHY THAT ESTIMATE IS WEAK

The estimate is the observed accumulation rate extrapolated linearly. It
assumes the bot keeps running and the market keeps producing candidates at
the rate it has been, and neither is reliable - a quiet weekend halves the
rate, a busy hour doubles it. It is a planning aid for deciding when to
look again, not a forecast, and it is labelled as such everywhere it is
shown.

WHAT IT DELIBERATELY DOES NOT DO

It does not suggest lowering any threshold to reach readiness sooner. The
floors exist because below them the answers are noise, and a tool that
offered "or you could just need 10 samples instead of 30" would be the
single most damaging feature in the repository.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.analysis.calibration import MIN_BUCKET_SAMPLE as MIN_TECHNICAL_BUCKET
from app.analysis.early_calibration import (
    EARLY_BUCKETS, MIN_BUCKET_SAMPLE as MIN_EARLY_BUCKET, early_bucket,
)
from app.analysis.monte_carlo import MIN_TRADES_FOR_MONTE_CARLO
from app.analysis.validation import MIN_CLOSED_TRADES
from app.research.early_ablation import MIN_CORRELATION_ROWS, count_samples
from app.research.early_walkforward import MIN_WINDOW_ROWS


@dataclass
class Requirement:
    """One research question and how close the data is to answering it."""
    question: str
    command: str
    have: int
    need: int
    unit: str
    note: str = ""

    @property
    def ready(self) -> bool:
        return self.have >= self.need

    @property
    def remaining(self) -> int:
        return max(self.need - self.have, 0)

    @property
    def progress(self) -> float:
        return min(self.have / self.need, 1.0) if self.need else 1.0

    def eta(self, hours_running: float | None) -> str:
        """Time to readiness at THIS requirement's own observed rate.

        Each quantity accrues at its own speed and they differ by orders
        of magnitude: a busy scanner produces hundreds of scored
        candidates for every closed position. Extrapolating one shared
        candidate rate across all of them would put "100 closed trades"
        a few hours away when it is really weeks, which is exactly the
        kind of optimism that gets someone to stop waiting and start
        tuning.

        So the rate is `have / hours_running` for this row alone, and a
        row still at zero has no measurable rate and gets no number. An
        ETA invented from no rate is worse than no ETA - it gets
        remembered as a date.
        """
        if self.ready:
            return "ready"
        if not hours_running or hours_running <= 0 or self.have <= 0:
            return "unknown - no measurable rate yet"
        per_hour = self.have / hours_running
        hours = self.remaining / per_hour
        if hours < 1:
            return "<1h at the current rate"
        if hours < 48:
            return f"~{hours:.0f}h at the current rate"
        return f"~{hours / 24:.1f} days at the current rate"

    def as_dict(self, hours_running: float | None = None) -> dict:
        return {
            "question": self.question,
            "command": self.command,
            "have": self.have,
            "need": self.need,
            "unit": self.unit,
            "ready": self.ready,
            "progress_pct": round(self.progress * 100, 1),
            "eta": self.eta(hours_running),
            "note": self.note,
        }


@dataclass
class EarlyLookCoverage:
    """How many technically-rejected candidates the early engine got to see.

    EARLY_SIGNAL_TECHNICAL_MARGIN is an unvalidated choice, and it is the
    valve on the early dataset: candidates more than that many points below
    the entry threshold are dropped before the early engine ever looks at
    them. Set too tight, the engine sees almost nothing and its calibration
    never fills; set too loose, every scanned token costs a security lookup
    and a candle fetch.

    Nothing new is written to answer this. Every scored candidate already
    leaves a TECHNICAL_SCORE row carrying its score, so the split is a
    query over history the bot was keeping anyway - which also means it
    answers retrospectively, for a margin that was never in force.
    """
    rejected: int = 0
    admitted: int = 0
    dropped: int = 0
    floor: float = 0.0

    @property
    def admitted_share(self) -> float | None:
        return self.admitted / self.rejected if self.rejected else None

    def note(self) -> str:
        if not self.rejected:
            return "no technically-rejected candidates recorded yet"
        share = self.admitted_share
        line = (
            f"{self.admitted} of {self.rejected} technically-rejected candidates "
            f"({share:.0%}) scored at or above the early-look floor of {self.floor:.0f} "
            f"and were shown to the early engine; {self.dropped} were dropped before it."
        )
        if share is not None and share < 0.10:
            line += (
                " Under 10% - EARLY_SIGNAL_TECHNICAL_MARGIN is starving the early dataset, "
                "and early calibration will take far longer than the ETA above suggests."
            )
        elif share is not None and share > 0.90:
            line += (
                " Over 90% - the margin is barely filtering, so nearly every scanned token "
                "is paying for a security lookup and a candle fetch."
            )
        return line

    def as_dict(self) -> dict:
        return {
            "rejected": self.rejected,
            "admitted": self.admitted,
            "dropped": self.dropped,
            "floor": self.floor,
            "admitted_share_pct": (
                round(self.admitted_share * 100, 1) if self.admitted_share is not None else None
            ),
            "note": self.note(),
        }


def early_look_coverage(db: Session) -> EarlyLookCoverage:
    """Split recorded technical rejections by the early-look floor."""
    from app import pipeline

    floor = settings.MIN_SIGNAL_SCORE_TO_ENTER - settings.EARLY_SIGNAL_TECHNICAL_MARGIN
    coverage = EarlyLookCoverage(floor=floor)

    scores = (
        db.query(models.PipelineEvent.score)
        .filter(
            models.PipelineEvent.stage == pipeline.TECHNICAL_SCORE,
            models.PipelineEvent.passed.is_(False),
            models.PipelineEvent.score.isnot(None),
        )
        .all()
    )
    for (score,) in scores:
        coverage.rejected += 1
        if score >= floor:
            coverage.admitted += 1
        else:
            coverage.dropped += 1
    return coverage


@dataclass
class Readiness:
    requirements: list[Requirement] = field(default_factory=list)
    early_look: EarlyLookCoverage = field(default_factory=EarlyLookCoverage)
    scored_candidates: int = 0
    first_observation: dt.datetime | None = None
    last_observation: dt.datetime | None = None

    @property
    def hours_running(self) -> float | None:
        if not self.first_observation or not self.last_observation:
            return None
        span = (self.last_observation - self.first_observation).total_seconds() / 3600
        return span if span > 0 else None

    @property
    def candidates_per_hour(self) -> float | None:
        hours = self.hours_running
        if not hours or not self.scored_candidates:
            return None
        return self.scored_candidates / hours

    @property
    def ready(self) -> list[Requirement]:
        return [r for r in self.requirements if r.ready]

    def headline(self) -> str:
        if not self.scored_candidates:
            return (
                "Nothing has been recorded yet. Start the bot in paper mode and leave it "
                "running; every question below is answered by data it collects on its own. "
                "Nothing here needs tuning first."
            )
        rate = self.candidates_per_hour
        pace = f" at about {rate:.1f} scored candidates an hour" if rate else ""
        return (
            f"{len(self.ready)} of {len(self.requirements)} questions are answerable. "
            f"{self.scored_candidates} candidates scored{pace}."
        )

    def as_dict(self) -> dict:
        rate = self.candidates_per_hour
        hours = self.hours_running
        return {
            "headline": self.headline(),
            "scored_candidates": self.scored_candidates,
            "hours_running": round(self.hours_running, 2) if self.hours_running else None,
            "candidates_per_hour": round(rate, 2) if rate else None,
            "ready": len(self.ready),
            "total": len(self.requirements),
            "early_look": self.early_look.as_dict(),
            "requirements": [r.as_dict(hours) for r in self.requirements],
        }

    def table(self) -> str:
        hours = self.hours_running
        lines = [self.headline(), ""]
        width = max((len(r.question) for r in self.requirements), default=10)
        for r in self.requirements:
            bar_len = 24
            filled = int(round(r.progress * bar_len))
            bar = "#" * filled + "." * (bar_len - filled)
            mark = "READY" if r.ready else f"{r.have}/{r.need}"
            lines.append(f"  {r.question:<{width}}  [{bar}] {mark:>9}  {r.eta(hours)}")
            if r.note:
                lines.append(f"  {'':<{width}}  {r.note}")
        lines.append("")
        lines.append(
            "  Each ETA extrapolates that row's OWN observed rate, so they differ - closed\n"
            "  positions accrue far slower than scored candidates. A quiet weekend halves\n"
            "  every rate and a busy hour doubles it: use these to decide when to look\n"
            "  again, not as dates. The sample floors are not negotiable - below them the\n"
            "  answers are noise."
        )
        lines.append("")
        lines.append(f"  Early-look valve: {self.early_look.note()}")
        lines.append("")
        lines.append("  Run the command beside a question once it reads READY:")
        for r in self.requirements:
            lines.append(f"    {r.question:<{width}}  {r.command}")
        return "\n".join(lines)


def _second_fullest_bucket(db: Session, column, horizon_minutes: int) -> int:
    """Rows in the SECOND-fullest score bucket for one horizon.

    Calibration compares a top bucket against a bottom one, so it needs
    TWO buckets at the floor and stays INSUFFICIENT DATA until both get
    there. The second-fullest is therefore the number that decides
    readiness: it reaches the floor exactly when the tool starts
    answering.

    Neither of the obvious alternatives works. The total across buckets
    would read 100% while every row sat in `<50`, answering nothing. The
    fullest bucket has the same problem one step later - it hits the floor
    while the tool is still refusing to speak, which is the precise
    optimism this report exists to avoid.
    """
    rows = (
        db.query(column)
        .filter(
            models.ForwardReturn.horizon_minutes == horizon_minutes,
            column.isnot(None),
            models.ForwardReturn.return_pct.isnot(None),
        )
        .all()
    )
    counts: dict[str, int] = {}
    for (score,) in rows:
        counts[early_bucket(score)] = counts.get(early_bucket(score), 0) + 1
    ranked = sorted(counts.values(), reverse=True)
    return ranked[1] if len(ranked) >= 2 else 0


def build_readiness(db: Session, *, horizon_minutes: int = 60) -> Readiness:
    """Count what exists against every floor the research tools enforce."""
    report = Readiness()

    report.scored_candidates = (
        db.query(func.count(func.distinct(models.ForwardReturn.pipeline_event_id))).scalar() or 0
    )
    report.first_observation = db.query(func.min(models.ForwardReturn.observed_at)).scalar()
    report.last_observation = db.query(func.max(models.ForwardReturn.observed_at)).scalar()
    if report.first_observation and report.first_observation.tzinfo is None:
        report.first_observation = report.first_observation.replace(tzinfo=dt.timezone.utc)
    if report.last_observation and report.last_observation.tzinfo is None:
        report.last_observation = report.last_observation.replace(tzinfo=dt.timezone.utc)

    early_rows = (
        db.query(func.count(models.ForwardReturn.id))
        .filter(
            models.ForwardReturn.horizon_minutes == horizon_minutes,
            models.ForwardReturn.early_score.isnot(None),
            models.ForwardReturn.return_pct.isnot(None),
        )
        .scalar() or 0
    )
    # Counted through the ablation's own loader rather than in SQL, so the
    # report cannot promise a sample the ablation then declines.
    with_features = count_samples(db, horizon_minutes=horizon_minutes)
    resolved_watches = (
        db.query(func.count(models.WatchlistEntry.id))
        .filter(models.WatchlistEntry.state.in_(["PAPER_BUY", "FAILED", "SKIPPED", "EXPIRED"]))
        .scalar() or 0
    )
    tracked_watches = db.query(func.count(models.WatchlistEntry.id)).scalar() or 0
    closed_trades = (
        db.query(func.count(models.Position.id))
        .filter(models.Position.status == models.PositionStatus.CLOSED.value)
        .scalar() or 0
    )

    report.early_look = early_look_coverage(db)
    report.requirements = [
        Requirement(
            question="technical calibration",
            command="python scripts/research.py calibration",
            have=_second_fullest_bucket(db, models.ForwardReturn.score, horizon_minutes),
            need=MIN_TECHNICAL_BUCKET,
            unit="resolved rows in the second-fullest score bucket",
            note="two buckets must reach the floor, so the second-fullest is what decides it",
        ),
        Requirement(
            question="early calibration",
            command="python scripts/research.py early",
            have=_second_fullest_bucket(db, models.ForwardReturn.early_score, horizon_minutes),
            need=MIN_EARLY_BUCKET,
            unit="resolved rows in the second-fullest early bucket",
            note=(
                f"{early_rows} early-scored rows resolved across {len(EARLY_BUCKETS)} buckets; "
                "two buckets must reach the floor, so the second-fullest is what decides it"
            ),
        ),
        Requirement(
            question="lead time",
            command="python scripts/research.py early",
            have=tracked_watches,
            need=MIN_EARLY_BUCKET,
            unit="tracked watchlist entries",
        ),
        Requirement(
            question="false positives",
            command="python scripts/research.py early",
            have=resolved_watches,
            need=MIN_EARLY_BUCKET,
            unit="resolved watchlist entries",
        ),
        Requirement(
            question="early ablation",
            command="python scripts/research.py early-ablate",
            have=with_features,
            need=MIN_CORRELATION_ROWS,
            unit="stored feature/outcome pairs",
        ),
        Requirement(
            question="early walk-forward",
            command="python scripts/research.py early-walkforward",
            have=early_rows,
            need=MIN_WINDOW_ROWS * 5,
            unit="early-scored resolved rows",
            note="four windows of at least 20 rows, plus a training block",
        ),
        Requirement(
            question="Monte Carlo",
            command="python scripts/performance_report.py",
            have=closed_trades,
            need=MIN_TRADES_FOR_MONTE_CARLO,
            unit="closed positions",
        ),
        Requirement(
            question="validation gate",
            command="python scripts/research.py report",
            have=closed_trades,
            need=MIN_CLOSED_TRADES,
            unit="closed positions",
        ),
    ]
    return report
