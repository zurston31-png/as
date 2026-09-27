"""Where candidates die, counted from the pipeline event log.

WHAT THIS ADDS OVER THE STAGE FUNNEL

app/analysis/stage_funnel.py counts EVENTS per stage: how many evaluations
passed and failed at each gate. That is the right shape for "is the
scanner working?" and the wrong shape for "which filter is costing me
candidates?", because one mint re-evaluated forty times and rejected forty
times contributes forty failures to a stage it was never going to clear.
A busy watchlist can make one stubborn token look like a systemic
bottleneck.

So this module counts differently, in three ways that answer different
questions:

  TERMINAL STAGE   For each MINT, the deepest stage it ever reached, and
                   whether it ever got through. One row per mint, so a
                   token evaluated forty times counts once. This is the
                   attribution that actually says where the funnel loses
                   things.

  BY REASON        Within each stage, which reasons come up and how many
                   distinct mints each one stopped. Reasons are collapsed
                   to their category the same way the funnel does it,
                   because the raw strings embed prices and scores and
                   would otherwise produce one category per event.

  NEAR MISSES      For the stages that produce a score, how many
                   rejections were within a small margin of the
                   threshold. A gate that rejects 300 candidates at 20
                   points below the line is doing its job; one that
                   rejects 300 within two points of it is a coin flip
                   wearing a threshold.

WHAT THIS DELIBERATELY DOES NOT DO

It does not recommend a threshold, and it does not compute what "would
have happened" had a filter been looser. The counterfactual question is
answerable - app/analysis/filter_quality.py does it properly, by
comparing forward returns between the tokens a check passed and the ones
it rejected - and it needs forward-return outcomes, not rejection counts.
Reading "liquidity rejected the most candidates" as "liquidity is too
strict" is exactly the inference this data cannot support: the most
prolific filter is usually the first one, and the first one is prolific
because it runs on everything.

EVERY NUMBER HERE COMES FROM A RECORDED PipelineEvent. A stage nothing
was recorded for reports no rows rather than zeros, because "this gate
rejected nobody" and "this gate never ran" are different facts and only
one of them is good news.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis.stage_funnel import _shorten
from app.pipeline import STAGE_ORDER

# Scored stages, and how close to the threshold counts as a near miss.
# Both scores are 0-100, so the margin is in points.
NEAR_MISS_POINTS = 5.0
SCORED_STAGES = ("TECHNICAL_SCORE", "MARKET_QUALITY", "SECURITY")

# Below this many distinct mints, a per-reason rate is not reported. Three
# rejections sharing a reason is not a pattern, and a "100% of candidates"
# built on one mint invites exactly the over-reading this module warns
# about.
MIN_MINTS_FOR_SHARE = 5


@dataclass
class ReasonCount:
    """One rejection category within one stage."""

    stage: str
    reason: str
    mints: int          # distinct mints stopped by this reason
    events: int         # raw evaluations - always >= mints
    example_symbol: str | None = None

    @property
    def repeats_per_mint(self) -> float | None:
        """How often the average affected mint hit this. A high number
        means one token being re-evaluated, not many tokens failing."""
        return (self.events / self.mints) if self.mints else None


@dataclass
class StageRejections:
    """Everything recorded about one gate over the window."""

    stage: str
    mints_reaching: int = 0        # distinct mints this stage ever evaluated
    mints_stopped: int = 0         # distinct mints that NEVER passed it
    mints_passed: int = 0
    events: int = 0
    events_failed: int = 0
    reasons: list[ReasonCount] = field(default_factory=list)
    near_misses: int = 0
    scored_rejections: int = 0
    median_rejected_score: float | None = None

    @property
    def stop_rate(self) -> float | None:
        """Share of mints reaching this gate that never got through it.
        None below the reporting floor rather than a confident fraction of
        four."""
        if self.mints_reaching < MIN_MINTS_FOR_SHARE:
            return None
        return self.mints_stopped / self.mints_reaching

    @property
    def near_miss_share(self) -> float | None:
        if not self.scored_rejections:
            return None
        return self.near_misses / self.scored_rejections

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "mints_reaching": self.mints_reaching,
            "mints_stopped": self.mints_stopped,
            "mints_passed": self.mints_passed,
            "events": self.events,
            "events_failed": self.events_failed,
            "stop_rate_pct": round(self.stop_rate * 100, 1) if self.stop_rate is not None else None,
            "near_misses": self.near_misses,
            "scored_rejections": self.scored_rejections,
            "near_miss_share_pct": (
                round(self.near_miss_share * 100, 1) if self.near_miss_share is not None else None
            ),
            "median_rejected_score": self.median_rejected_score,
            "reasons": [
                {
                    "reason": r.reason,
                    "mints": r.mints,
                    "events": r.events,
                    "repeats_per_mint": (
                        round(r.repeats_per_mint, 1) if r.repeats_per_mint is not None else None
                    ),
                    "example_symbol": r.example_symbol,
                }
                for r in self.reasons
            ],
        }


@dataclass
class RejectionReport:
    window_hours: float | None
    mints_seen: int = 0
    mints_reaching_a_position: int = 0
    stages: list[StageRejections] = field(default_factory=list)
    terminal_stage_counts: dict[str, int] = field(default_factory=dict)
    note: str = ""

    @property
    def has_data(self) -> bool:
        return self.mints_seen > 0

    @property
    def biggest_blocker(self) -> StageRejections | None:
        """The stage that terminally stopped the most distinct mints.

        "Biggest" is a count, not a judgement. The earliest gate usually
        wins this simply because it evaluates everything - see the module
        docstring on why that is not evidence it is too strict.
        """
        candidates = [s for s in self.stages if s.mints_stopped]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.mints_stopped)

    def as_dict(self) -> dict:
        return {
            "window_hours": self.window_hours,
            "mints_seen": self.mints_seen,
            "mints_reaching_a_position": self.mints_reaching_a_position,
            "terminal_stage_counts": dict(self.terminal_stage_counts),
            "stages": [s.as_dict() for s in self.stages],
            "note": self.note,
        }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _threshold_for(stage: str) -> float | None:
    """The configured pass mark for a scored stage, or None.

    Read from settings rather than hard-coded so a near miss stays defined
    against the threshold actually in force. SECURITY is inverted - its
    score is a RISK score, rejected when it is too HIGH - and that is
    handled by the caller.
    """
    from app.config import settings

    return {
        "TECHNICAL_SCORE": settings.MIN_SIGNAL_SCORE_TO_ENTER,
        "MARKET_QUALITY": settings.MIN_MARKET_QUALITY_SCORE,
        "SECURITY": settings.REJECT_RUG_SCORE_ABOVE,
    }.get(stage)


def _is_near_miss(stage: str, score: float, threshold: float) -> bool:
    if stage == "SECURITY":
        # A rug RISK score: rejected for being above the line, so a near
        # miss is just above it.
        return threshold < score <= threshold + NEAR_MISS_POINTS
    return threshold - NEAR_MISS_POINTS <= score < threshold


def build_rejection_report(
    db: Session,
    *,
    window_hours: float | None = 24.0,
    strategy_version: str | None = None,
) -> RejectionReport:
    """Count, per stage, which mints were stopped and why."""
    query = db.query(models.PipelineEvent)
    if window_hours is not None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
        query = query.filter(models.PipelineEvent.occurred_at >= cutoff)
    if strategy_version:
        query = query.filter(models.PipelineEvent.strategy_version == strategy_version)

    events = query.order_by(models.PipelineEvent.occurred_at.asc()).all()

    report = RejectionReport(window_hours=window_hours)
    if not events:
        report.note = (
            "no pipeline events recorded in this window - the scanner either did not run "
            "or found nothing to evaluate. This is not the same as 'nothing was rejected'."
        )
        return report

    # A mint is the unit of attribution. Events with no mint recorded are
    # counted in the event totals but cannot be attributed to one, so they
    # are excluded from the distinct-mint arithmetic rather than being
    # merged under a shared placeholder key - which would make them look
    # like one very busy token.
    stage_index = {name: i for i, name in enumerate(STAGE_ORDER)}

    per_stage: dict[str, StageRejections] = {}
    reached: dict[str, set[str]] = {}
    passed: dict[str, set[str]] = {}
    reason_mints: dict[tuple[str, str], set[str]] = {}
    reason_events: dict[tuple[str, str], int] = {}
    reason_example: dict[tuple[str, str], str] = {}
    rejected_scores: dict[str, list[float]] = {}
    deepest: dict[str, tuple[int, str]] = {}
    all_mints: set[str] = set()

    for event in events:
        stage = event.stage
        row = per_stage.setdefault(stage, StageRejections(stage=stage))
        row.events += 1
        if not event.passed:
            row.events_failed += 1

        mint = event.token_address
        if mint:
            all_mints.add(mint)
            reached.setdefault(stage, set()).add(mint)
            if event.passed:
                passed.setdefault(stage, set()).add(mint)

            depth = stage_index.get(stage)
            if depth is not None:
                current = deepest.get(mint)
                if current is None or depth > current[0]:
                    deepest[mint] = (depth, stage)

        if not event.passed and event.reason:
            key = (stage, _shorten(event.reason))
            reason_events[key] = reason_events.get(key, 0) + 1
            if mint:
                reason_mints.setdefault(key, set()).add(mint)
            reason_example.setdefault(key, event.symbol)

        threshold = _threshold_for(stage)
        if not event.passed and event.score is not None and threshold is not None:
            rejected_scores.setdefault(stage, []).append(event.score)
            row.scored_rejections += 1
            if _is_near_miss(stage, event.score, threshold):
                row.near_misses += 1

    for stage, row in per_stage.items():
        reaching = reached.get(stage, set())
        through = passed.get(stage, set())
        row.mints_reaching = len(reaching)
        row.mints_passed = len(through)
        # Stopped means never got through, over the whole window - not
        # "failed at least once". A token rejected on a cooldown at 09:00
        # and bought at 11:00 was not stopped by the risk gate.
        row.mints_stopped = len(reaching - through)
        row.median_rejected_score = _median(rejected_scores.get(stage, []))

        row.reasons = sorted(
            (
                ReasonCount(
                    stage=stage,
                    reason=reason,
                    mints=len(reason_mints.get((stage, reason), set())),
                    events=count,
                    example_symbol=reason_example.get((stage, reason)),
                )
                for (ev_stage, reason), count in reason_events.items()
                if ev_stage == stage
            ),
            key=lambda r: (-r.mints, -r.events, r.reason),
        )

    report.stages = sorted(
        per_stage.values(),
        key=lambda s: stage_index.get(s.stage, len(STAGE_ORDER)),
    )
    report.mints_seen = len(all_mints)
    report.mints_reaching_a_position = len(passed.get("OPEN_POSITION", set()))

    open_depth = stage_index.get("OPEN_POSITION")
    for mint, (depth, stage) in deepest.items():
        # A mint that reached OPEN_POSITION did not terminate at a gate;
        # counting it as "stopped at OPEN_POSITION" would read as a
        # failure when it is the success case.
        if open_depth is not None and depth >= open_depth:
            continue
        report.terminal_stage_counts[stage] = report.terminal_stage_counts.get(stage, 0) + 1

    unattributed = sum(1 for e in events if not e.token_address)
    if unattributed:
        report.note = (
            f"{unattributed} of {len(events)} events carry no mint address and are counted in "
            "the event totals but not in any distinct-mint figure - they cannot be attributed "
            "to a token."
        )
    return report
