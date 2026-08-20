"""What happened to the candidates the bot turned down.

Every filter looks good from the inside. It rejects things, the things it
rejected are not in the trade log, and the trade log is where performance
gets measured - so a filter that throws away the best setups produces
exactly the same clean report as one that throws away the worst.

The only way to catch that is to follow the rejects forward, which the bot
already does (app/analysis/forward_returns.py schedules a row for every
scored candidate, passed or failed). This module reads those rows, sorts
them by WHICH GATE turned the candidate down, and asks one question per
gate:

    did the opportunities this filter rejected go on to do better,
    after costs, than the ones it let through?

A yes is not permission to loosen anything. It is a place to look.

WHAT IT CANNOT SEE

Forward returns are scheduled at the TECHNICAL_SCORE stage, so a candidate
killed before that - by the prescreen, the history check, the risk gate -
has no recorded outcome at all and cannot appear here. That is a real hole
and it is reported as one rather than rendered as an empty row, because an
empty row reads as "this filter rejects nothing worth having".

SAFETY FILTERS ARE NOT TUNING CANDIDATES

The security and data-quality gates are reported like any other, because
knowing what they cost is legitimate. They are never flagged as a place to
loosen, however good the rejected cohort looks, and this module emits no
recommendation for them at all. A rug pull that would have been profitable
for eleven minutes is not evidence that the rug filter is too strict, and
the one thing an automated search must never be able to do is talk itself
into switching off the safety rail.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models, pipeline
from app.analysis.calibration import round_trip_cost_pct
from app.autopilot.promote import bootstrap_p_value

# The stage at which forward-return tracking begins. Nothing earlier can be
# analysed here, and pretending otherwise would invent coverage.
MEASURABLE_FROM = pipeline.TECHNICAL_SCORE

# Gates that exist to prevent loss, not to select for return. Reported,
# never recommended for loosening. See the module docstring.
PROTECTED_STAGES = frozenset({
    pipeline.SECURITY,
    pipeline.DATA_QUALITY,
    pipeline.RISK,
})

# Below this, a cohort's mean is an anecdote and the comparison is not run.
MIN_COHORT = 30

# How confident before a gate is called out. The gate that promotes
# strategies uses 0.05 with a multiple-comparison correction; this is a
# screening report that promotes nothing, so it uses the same alpha
# uncorrected and says plainly that it is a screen.
ALPHA = 0.05

# How wide the pipeline window around a scored candidate is when matching
# it to the stage that rejected it. Generous enough to cover the security
# and execution calls that follow scoring, tight enough not to capture the
# same token's next scan cycle.
MATCH_WINDOW_MINUTES = 30.0


@dataclass
class Cohort:
    """One group of candidates that shared a fate."""
    label: str
    returns_net: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.returns_net)

    @property
    def mean(self) -> float | None:
        return sum(self.returns_net) / self.n if self.n else None

    @property
    def win_rate(self) -> float | None:
        if not self.n:
            return None
        return sum(1 for r in self.returns_net if r > 0) / self.n * 100

    @property
    def usable(self) -> bool:
        return self.n >= MIN_COHORT


@dataclass
class GateVerdict:
    """What one gate's rejections were worth, against what it accepted."""
    stage: str
    rejected: Cohort
    accepted: Cohort
    p_value: float | None = None

    @property
    def protected(self) -> bool:
        return self.stage in PROTECTED_STAGES

    @property
    def difference(self) -> float | None:
        a, b = self.accepted.mean, self.rejected.mean
        return (b - a) if a is not None and b is not None else None

    @property
    def comparable(self) -> bool:
        return self.rejected.usable and self.accepted.usable

    def grade(self) -> str:
        """PASS = the gate is keeping the better half. FAIL = it is not."""
        if not self.comparable:
            return "INSUFFICIENT_DATA"
        if self.difference is None or self.difference <= 0:
            return "PASS"
        if self.p_value is not None and self.p_value <= ALPHA:
            return "FAIL"
        return "PASS"

    def note(self) -> str:
        if not self.comparable:
            return (
                f"{self.stage}: {self.rejected.n} rejected / {self.accepted.n} accepted with a "
                f"measured outcome - below the {MIN_COHORT} needed on both sides. Nothing can "
                "be concluded, which is a statement about the sample."
            )
        direction = (
            f"rejected {self.rejected.mean:+.2f}% vs accepted {self.accepted.mean:+.2f}% "
            f"net of costs (n={self.rejected.n}/{self.accepted.n})"
        )
        if self.grade() == "PASS":
            return f"{self.stage}: keeping the better half - {direction}."
        if self.protected:
            return (
                f"{self.stage}: {direction}, p={self.p_value:.3f}. This is a SAFETY gate. The "
                "number is the price of the protection, not an argument against it - a rug that "
                "would have paid for eleven minutes is not evidence the filter is too strict, "
                "and nothing here recommends loosening it."
            )
        return (
            f"{self.stage}: rejecting opportunities that OUTPERFORMED what it accepted - "
            f"{direction}, p={self.p_value:.3f}. Worth investigating. This is a screening "
            "result, not a mandate: any actual change goes through the promotion gate as a "
            "challenger, on its own paired sample."
        )

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 4) if v is not None else None
        return {
            "stage": self.stage,
            "protected": self.protected,
            "grade": self.grade(),
            "rejected_n": self.rejected.n,
            "accepted_n": self.accepted.n,
            "rejected_mean_net_pct": r(self.rejected.mean),
            "accepted_mean_net_pct": r(self.accepted.mean),
            "rejected_win_rate_pct": r(self.rejected.win_rate),
            "accepted_win_rate_pct": r(self.accepted.win_rate),
            "difference_pct": r(self.difference),
            "p_value": r(self.p_value),
            "note": self.note(),
        }


@dataclass
class CounterfactualReport:
    horizon_minutes: int
    gates: list[GateVerdict] = field(default_factory=list)
    accepted: Cohort = field(default_factory=lambda: Cohort("accepted"))
    unmatched: int = 0
    invisible_stages: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[GateVerdict]:
        """Gates rejecting better opportunities - excluding the safety ones,
        which are never a tuning target however they score."""
        return [g for g in self.gates if g.grade() == "FAIL" and not g.protected]

    def verdict(self) -> str:
        if not self.accepted.usable:
            return (
                f"INSUFFICIENT_DATA: {self.accepted.n} accepted candidates have a measured "
                f"{self.horizon_minutes}m outcome (need {MIN_COHORT}). Every comparison here is "
                "against that cohort, so none of them can run yet."
            )
        if not self.flagged:
            return (
                f"No tunable gate is rejecting better opportunities than it accepts at "
                f"{self.horizon_minutes}m. The filters are, on this evidence, keeping the "
                "better half."
            )
        names = ", ".join(g.stage for g in self.flagged)
        return (
            f"{len(self.flagged)} gate(s) rejected opportunities that outperformed what they "
            f"accepted at {self.horizon_minutes}m: {names}. Investigate before changing "
            "anything - and change it as a challenger, not as an edit to the champion."
        )

    def as_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "accepted_n": self.accepted.n,
            "unmatched": self.unmatched,
            "invisible_stages": list(self.invisible_stages),
            "flagged": [g.stage for g in self.flagged],
            "gates": [g.as_dict() for g in self.gates],
            "verdict": self.verdict(),
        }


def _stage_rank() -> dict[str, int]:
    return {stage: i for i, stage in enumerate(pipeline.STAGE_ORDER)}


def _fate(events: list[models.PipelineEvent]) -> str | None:
    """The stage that turned this candidate down, or None if none did.

    The EARLIEST failing stage in pipeline order, not the earliest by
    timestamp: two gates can record within the same second, and attributing
    the rejection to whichever row was written first would scatter one
    filter's cost across its neighbours.
    """
    rank = _stage_rank()
    failures = [e for e in events if not e.passed and e.stage not in pipeline.TERMINAL_STAGES]
    if not failures:
        return None
    return min(failures, key=lambda e: rank.get(e.stage, 99)).stage


def build_counterfactual(
    db: Session,
    *,
    horizon_minutes: int = 60,
    strategy_version: str | None = None,
) -> CounterfactualReport:
    """Sort measured outcomes by the gate that rejected them, and compare.

    Restricted to one strategy version when asked, because a filter that
    changed mid-run is two filters and pooling them describes neither.
    """
    report = CounterfactualReport(horizon_minutes=horizon_minutes)
    cost = round_trip_cost_pct() * 100

    query = db.query(models.ForwardReturn).filter(
        models.ForwardReturn.horizon_minutes == horizon_minutes,
        models.ForwardReturn.return_pct.isnot(None),
    )
    if strategy_version is not None:
        query = query.filter(models.ForwardReturn.strategy_version == strategy_version)
    rows = query.all()
    if not rows:
        report.invisible_stages = _invisible_stages()
        return report

    # One pass over the pipeline log for the tokens involved, rather than a
    # query per row - a few hundred candidates would otherwise be a few
    # hundred round trips.
    tokens = {r.token_address for r in rows if r.token_address}
    events_by_token: dict[str, list[models.PipelineEvent]] = {}
    if tokens:
        for event in (
            db.query(models.PipelineEvent)
            .filter(models.PipelineEvent.token_address.in_(tokens))
            .all()
        ):
            events_by_token.setdefault(event.token_address, []).append(event)

    rejected: dict[str, Cohort] = {}
    for row in rows:
        net = row.return_pct - cost
        window = _events_near(events_by_token.get(row.token_address, []), row.observed_at)
        if not window:
            # Scored, followed forward, but no pipeline log to say what
            # became of it. Counted rather than assigned to a cohort:
            # guessing "accepted" would credit the filters with outcomes
            # they never saw.
            report.unmatched += 1
            continue
        stage = _fate(window)
        if stage is None:
            report.accepted.returns_net.append(net)
        else:
            rejected.setdefault(stage, Cohort(stage)).returns_net.append(net)

    rank = _stage_rank()
    for stage in sorted(rejected, key=lambda s: rank.get(s, 99)):
        verdict = GateVerdict(stage=stage, rejected=rejected[stage], accepted=report.accepted)
        if verdict.comparable and (verdict.difference or 0) > 0:
            # Only worth resampling when the rejected side looks better;
            # the p-value answers "could this lift be sampling noise?"
            verdict.p_value = bootstrap_p_value(
                report.accepted.returns_net, rejected[stage].returns_net
            )
        report.gates.append(verdict)

    report.invisible_stages = _invisible_stages()
    return report


def _events_near(events: list[models.PipelineEvent], observed_at) -> list[models.PipelineEvent]:
    """The pipeline events belonging to one candidate's run.

    Matched by time window rather than by id because a candidate produces
    many stage rows and the forward return points at only one of them.
    """
    import datetime as dt

    if observed_at is None:
        return []
    start = observed_at if observed_at.tzinfo else observed_at.replace(tzinfo=dt.timezone.utc)
    window = dt.timedelta(minutes=MATCH_WINDOW_MINUTES)
    out = []
    for event in events:
        moment = event.occurred_at
        if moment is None:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=dt.timezone.utc)
        # Slightly before, because scoring is recorded a moment after the
        # signal timestamp the forward return carries.
        if start - dt.timedelta(minutes=5) <= moment <= start + window:
            out.append(event)
    return out


def _invisible_stages() -> list[str]:
    """Gates that run before forward-return tracking starts.

    Named explicitly. Rendering them as empty rows would read as "this
    filter rejects nothing worth having", which is the opposite of what an
    absent measurement means.
    """
    rank = _stage_rank()
    cutoff = rank[MEASURABLE_FROM]
    return [
        stage for stage in pipeline.FILTER_STAGES
        if rank.get(stage, 99) < cutoff and stage != pipeline.DISCOVERED
    ]
