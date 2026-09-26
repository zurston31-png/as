"""What the scoring engine actually produces.

Before arguing about where a threshold belongs, you have to know the shape
of the thing you are thresholding. A 0-100 score built as a weighted average
of fourteen factors regresses toward 50 by construction: its practical
ceiling is nowhere near 100, and a threshold of 75 can quietly mean "trade
the top 3%" rather than "trade good setups".

This module reads the TECHNICAL_SCORE (and MARKET_QUALITY, and SECURITY)
events written by app/pipeline.py and reports the distribution. Those events
are recorded for EVERY token the engine scored, including the ones then
rejected, which is what makes the distribution the real one rather than the
distribution of survivors.

It answers "what would this threshold cost me in candidates?" and nothing
else. Whether the surviving candidates are any *better* is a different
question, and the honest answer needs outcomes - see
app/analysis/calibration.py. A threshold chosen from this module alone
would be chosen on trade frequency, which is exactly the wrong criterion.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.pipeline import TECHNICAL_SCORE

# The thresholds worth reporting a survival rate for. Spans the range an
# operator would plausibly consider, at the granularity the argument is
# actually conducted in.
REPORTED_THRESHOLDS: tuple[float, ...] = (50, 55, 60, 62.5, 65, 67.5, 70, 72.5, 75, 80, 85)

# Below this the percentiles are arithmetic rather than information.
MIN_SAMPLE_FOR_A_DISTRIBUTION = 30


@dataclass
class ThresholdSurvival:
    threshold: float
    qualifying: int
    total: int

    @property
    def share(self) -> float:
        return (self.qualifying / self.total) if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "qualifying": self.qualifying,
            "share_pct": round(self.share * 100, 2),
        }


@dataclass
class ScoreDistribution:
    stage: str
    sample_size: int
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    percentiles: dict[int, float] = field(default_factory=dict)
    survival: list[ThresholdSurvival] = field(default_factory=list)
    histogram: list[tuple[str, int]] = field(default_factory=list)
    unreliable_count: int = 0        # scored, but flagged as built on too little data
    warnings: list[str] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.sample_size >= MIN_SAMPLE_FOR_A_DISTRIBUTION

    def share_reaching(self, threshold: float) -> float | None:
        entry = next((s for s in self.survival if s.threshold == threshold), None)
        return entry.share if entry else None

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "sample_size": self.sample_size,
            "reliable": self.reliable,
            "mean": round(self.mean, 2) if self.mean is not None else None,
            "median": round(self.median, 2) if self.median is not None else None,
            "stdev": round(self.stdev, 2) if self.stdev is not None else None,
            "min": round(self.minimum, 2) if self.minimum is not None else None,
            "max": round(self.maximum, 2) if self.maximum is not None else None,
            "percentiles": {k: round(v, 2) for k, v in sorted(self.percentiles.items())},
            "survival": [s.as_dict() for s in self.survival],
            "histogram": [{"bucket": b, "count": c} for b, c in self.histogram],
            "unreliable_count": self.unreliable_count,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        if not self.sample_size:
            return f"{self.stage}: no scores recorded yet."
        lines = [
            f"{self.stage} score distribution  (n={self.sample_size}"
            + ("" if self.reliable else ", TOO SMALL TO READ")
            + ")",
            f"  mean {self.mean:.1f}   median {self.median:.1f}   sd {self.stdev:.1f}"
            f"   range {self.minimum:.1f}-{self.maximum:.1f}",
            "  percentile  " + "  ".join(f"p{k}={v:.1f}" for k, v in sorted(self.percentiles.items())),
            "  reaching:   " + "  ".join(f"{s.threshold:g}:{s.share * 100:.0f}%" for s in self.survival),
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _percentile(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (pct / 100) * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe_scores(
    scores: list[float], *, stage: str = TECHNICAL_SCORE, unreliable_count: int = 0
) -> ScoreDistribution:
    """Summarise a list of scores. Pure - no database, so it is testable."""
    if not scores:
        return ScoreDistribution(
            stage=stage, sample_size=0, unreliable_count=unreliable_count,
            warnings=["no scores recorded yet - the engine has not scored anything"],
        )

    ordered = sorted(scores)
    n = len(ordered)
    mean = sum(ordered) / n
    variance = sum((s - mean) ** 2 for s in ordered) / n if n > 1 else 0.0

    warnings: list[str] = []
    if n < MIN_SAMPLE_FOR_A_DISTRIBUTION:
        warnings.append(
            f"only {n} scores (need >={MIN_SAMPLE_FOR_A_DISTRIBUTION}) - these percentiles "
            "are arithmetic, not a description of the engine"
        )

    survival = [
        ThresholdSurvival(t, sum(1 for s in ordered if s >= t), n) for t in REPORTED_THRESHOLDS
    ]

    buckets: dict[str, int] = {}
    for score in ordered:
        low = min(int(score // 5) * 5, 95)
        buckets[f"{low}-{low + 5}"] = buckets.get(f"{low}-{low + 5}", 0) + 1
    histogram = sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0]))

    return ScoreDistribution(
        stage=stage,
        sample_size=n,
        mean=mean,
        median=_percentile(ordered, 50),
        stdev=math.sqrt(variance),
        minimum=ordered[0],
        maximum=ordered[-1],
        percentiles={p: _percentile(ordered, p) for p in (5, 10, 25, 50, 75, 90, 95, 99)},
        survival=survival,
        histogram=histogram,
        unreliable_count=unreliable_count,
        warnings=warnings,
    )


def build_score_distribution(
    db: Session,
    *,
    stage: str = TECHNICAL_SCORE,
    window_hours: float | None = None,
    strategy_version: str | None = None,
    include_unreliable: bool = False,
) -> ScoreDistribution:
    """Distribution of every score this stage produced.

    Includes rejected tokens by default and by design. Restricting to the
    ones that passed would describe only what got through the threshold,
    which cannot be used to judge the threshold.

    `include_unreliable` controls whether scores the engine itself flagged
    as built on too much missing data are counted. They are excluded by
    default: those scores are noise around a neutral 50 and would pull the
    whole distribution toward the middle, making the engine look less
    decisive than it is.
    """
    query = db.query(models.PipelineEvent).filter(
        models.PipelineEvent.stage == stage,
        models.PipelineEvent.score.isnot(None),
    )
    if window_hours is not None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
        query = query.filter(models.PipelineEvent.occurred_at >= cutoff)
    if strategy_version is not None:
        query = query.filter(models.PipelineEvent.strategy_version == strategy_version)

    scores: list[float] = []
    unreliable = 0
    for event in query.all():
        flagged_reliable = (event.detail or {}).get("reliable", True)
        if not flagged_reliable:
            unreliable += 1
            if not include_unreliable:
                continue
        scores.append(event.score)

    distribution = describe_scores(scores, stage=stage, unreliable_count=unreliable)
    if unreliable and not include_unreliable:
        distribution.warnings.append(
            f"{unreliable} score(s) excluded as unreliable (too much missing input data). "
            "They are counted here but kept out of the percentiles, where they would pull "
            "the distribution toward a neutral 50."
        )
    return distribution
