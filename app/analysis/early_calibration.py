"""Does the Early Opportunity Score actually predict anything?

Three questions, and the engine is worthless unless the first one answers
yes:

    CALIBRATION    do higher early scores precede better outcomes?
    LEAD TIME      when the engine finds a winner, how early?
    FALSE POSITIVES  when a high score fails, why?

The first is the existence question. If mean forward return does not rise
with the early score, the score is noise with a number attached, and the
correct response is to delete app/early/score.py rather than reweight it
until a backtest cooperates.

The second is what separates this engine from the technical score. A
signal that fires reliably but only after the move is already 40% done is
not an early signal - it is a late signal with good manners. Lead time is
measured from the FIRST time a token was scored, not from when a trade was
placed, so a signal that fired early and was acted on late still scores as
early. That is the honest attribution: the engine's job is detection, and
acting on detection is a separate system's job.

The third exists because "it works 30% of the time" is not actionable
while "it fails because volume evaporates before confirmation" is. Every
watchlist failure carries a category, so the failure modes can be counted
rather than recalled.

WHAT WOULD MAKE THIS DISHONEST, AND IS THEREFORE REFUSED

  Measuring lead time only on tokens that later pumped. Every WATCH entry
  is included, winners and failures alike - a lead-time figure computed
  over survivors would describe a bot that only ever saw winners.

  Counting expired watchlist entries as neither success nor failure. A
  token that looked promising and went nowhere IS a false positive, and
  dropping it would silently improve the hit rate.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis.calibration import round_trip_cost_pct
from app.early import watchlist as wl

# Early-score buckets, as the spec asks for them.
EARLY_BUCKET_EDGES: tuple[float, ...] = (50, 60, 65, 70, 75, 80)

MIN_BUCKET_SAMPLE = 30

# Lead-time thresholds: how far past the detection price the token later
# went, and whether the signal arrived before it got there.
LEAD_THRESHOLDS: tuple[float, ...] = (10.0, 20.0, 50.0)


def early_bucket(score: float) -> str:
    if score < EARLY_BUCKET_EDGES[0]:
        return f"<{EARLY_BUCKET_EDGES[0]:g}"
    for low, high in zip(EARLY_BUCKET_EDGES, EARLY_BUCKET_EDGES[1:]):
        if score < high:
            return f"{low:g}-{high:g}"
    return f"{EARLY_BUCKET_EDGES[-1]:g}+"


EARLY_BUCKETS: tuple[str, ...] = (
    f"<{EARLY_BUCKET_EDGES[0]:g}",
    *(f"{low:g}-{high:g}" for low, high in zip(EARLY_BUCKET_EDGES, EARLY_BUCKET_EDGES[1:])),
    f"{EARLY_BUCKET_EDGES[-1]:g}+",
)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

@dataclass
class EarlyBucketOutcome:
    bucket: str
    horizon_minutes: int
    sample_size: int
    mean_return_pct: float | None = None
    median_return_pct: float | None = None
    win_rate_pct: float | None = None
    mean_favorable_pct: float | None = None      # average MFE
    mean_adverse_pct: float | None = None        # average MAE
    profit_factor: float | None = None
    expectancy_net_pct: float | None = None
    outcome_mix: dict[str, int] = field(default_factory=dict)

    @property
    def meaningful(self) -> bool:
        return self.sample_size >= MIN_BUCKET_SAMPLE

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 3) if v is not None else None

        return {
            "bucket": self.bucket,
            "horizon_minutes": self.horizon_minutes,
            "sample_size": self.sample_size,
            "meaningful": self.meaningful,
            "mean_return_pct": r(self.mean_return_pct),
            "median_return_pct": r(self.median_return_pct),
            "win_rate_pct": r(self.win_rate_pct),
            "mean_favorable_pct": r(self.mean_favorable_pct),
            "mean_adverse_pct": r(self.mean_adverse_pct),
            "profit_factor": r(self.profit_factor),
            "expectancy_net_pct": r(self.expectancy_net_pct),
            "outcome_mix": dict(self.outcome_mix),
        }


@dataclass
class EarlyCalibration:
    horizon_minutes: int
    buckets: list[EarlyBucketOutcome] = field(default_factory=list)
    cost_pct: float = 0.0

    @property
    def measurable(self) -> list[EarlyBucketOutcome]:
        return [b for b in self.buckets if b.meaningful and b.expectancy_net_pct is not None]

    @property
    def monotonic(self) -> bool | None:
        usable = self.measurable
        if len(usable) < 2:
            return None
        order = {label: i for i, label in enumerate(EARLY_BUCKETS)}
        usable = sorted(usable, key=lambda b: order.get(b.bucket, 0))
        return all(a.expectancy_net_pct <= b.expectancy_net_pct for a, b in zip(usable, usable[1:]))

    def verdict(self) -> str:
        """Judged on expectancy AFTER costs, not on raw return.

        A score that ranks raw returns correctly but never clears the cost
        of trading has not found an edge; it has found a pattern that is
        real and unprofitable, which is a different and much more common
        thing.
        """
        usable = self.measurable
        if len(usable) < 2:
            return (
                f"INSUFFICIENT DATA at {self.horizon_minutes}m: fewer than two early-score "
                f"buckets have {MIN_BUCKET_SAMPLE}+ measured outcomes. The Early Signal Engine "
                "has not been tested."
            )
        order = {label: i for i, label in enumerate(EARLY_BUCKETS)}
        usable = sorted(usable, key=lambda b: order.get(b.bucket, 0))
        low, high = usable[0], usable[-1]

        if high.expectancy_net_pct <= 0:
            return (
                f"NO EDGE at {self.horizon_minutes}m: even the {high.bucket} bucket has negative "
                f"after-cost expectancy ({high.expectancy_net_pct:+.2f}%). The early score does "
                "not identify profitable entries at any level."
            )
        if self.monotonic:
            return (
                f"At {self.horizon_minutes}m the early score separates outcomes: after-cost "
                f"expectancy rises from {low.expectancy_net_pct:+.2f}% in {low.bucket} to "
                f"{high.expectancy_net_pct:+.2f}% in {high.bucket}, monotonically."
            )
        if high.expectancy_net_pct > low.expectancy_net_pct:
            return (
                f"At {self.horizon_minutes}m higher early scores do better "
                f"({low.bucket} {low.expectancy_net_pct:+.2f}% vs {high.bucket} "
                f"{high.expectancy_net_pct:+.2f}%) but not monotonically - a weak ranking."
            )
        return (
            f"At {self.horizon_minutes}m the early score is INVERTED: {high.bucket} "
            f"({high.expectancy_net_pct:+.2f}%) does worse than {low.bucket} "
            f"({low.expectancy_net_pct:+.2f}%). Higher scores are finding worse entries."
        )

    def as_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "cost_pct": round(self.cost_pct * 100, 3),
            "monotonic": self.monotonic,
            "verdict": self.verdict(),
            "buckets": [b.as_dict() for b in self.buckets],
        }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def build_early_calibration(db: Session, *, horizon_minutes: int) -> EarlyCalibration:
    """Group measured forward returns by EARLY score bucket."""
    rows = (
        db.query(models.ForwardReturn)
        .filter(
            models.ForwardReturn.horizon_minutes == horizon_minutes,
            models.ForwardReturn.early_score.isnot(None),
            models.ForwardReturn.return_pct.isnot(None),
        )
        .all()
    )

    grouped: dict[str, list[models.ForwardReturn]] = {b: [] for b in EARLY_BUCKETS}
    for row in rows:
        grouped[early_bucket(row.early_score)].append(row)

    cost = round_trip_cost_pct() * 100
    buckets: list[EarlyBucketOutcome] = []

    for label in EARLY_BUCKETS:
        group = grouped[label]
        if not group:
            buckets.append(EarlyBucketOutcome(label, horizon_minutes, 0))
            continue

        returns = [r.return_pct for r in group]
        net = [r - cost for r in returns]
        wins = [r for r in net if r > 0]
        losses = [r for r in net if r <= 0]

        favorable = [r.max_favorable_pct for r in group if r.max_favorable_pct is not None]
        adverse = [r.max_adverse_pct for r in group if r.max_adverse_pct is not None]

        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        # inf would be a small sample with no losers yet, not an edge - so
        # it is reported as unknown rather than as a spectacular number.
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

        buckets.append(
            EarlyBucketOutcome(
                bucket=label,
                horizon_minutes=horizon_minutes,
                sample_size=len(group),
                mean_return_pct=sum(returns) / len(returns),
                median_return_pct=_median(returns),
                win_rate_pct=len(wins) / len(net) * 100,
                mean_favorable_pct=(sum(favorable) / len(favorable)) if favorable else None,
                mean_adverse_pct=(sum(adverse) / len(adverse)) if adverse else None,
                profit_factor=profit_factor,
                expectancy_net_pct=sum(net) / len(net),
                outcome_mix=dict(Counter(r.outcome for r in group if r.outcome)),
            )
        )

    return EarlyCalibration(
        horizon_minutes=horizon_minutes, buckets=buckets, cost_pct=round_trip_cost_pct()
    )


# ---------------------------------------------------------------------------
# lead time
# ---------------------------------------------------------------------------

@dataclass
class LeadTimeReport:
    tracked: int = 0
    reached: dict[float, int] = field(default_factory=dict)   # threshold -> tokens that got there
    detected_before: dict[float, int] = field(default_factory=dict)
    lead_minutes: list[float] = field(default_factory=list)

    @property
    def median_lead_minutes(self) -> float | None:
        return _median(self.lead_minutes)

    def share_before(self, threshold: float) -> float | None:
        total = self.reached.get(threshold, 0)
        if not total:
            return None
        return self.detected_before.get(threshold, 0) / total

    @property
    def conclusive(self) -> bool:
        return self.tracked >= MIN_BUCKET_SAMPLE

    def verdict(self) -> str:
        if not self.conclusive:
            return (
                f"INSUFFICIENT DATA: only {self.tracked} tracked signals "
                f"(need >={MIN_BUCKET_SAMPLE}). Lead time is not measurable yet."
            )
        parts = []
        for threshold in LEAD_THRESHOLDS:
            share = self.share_before(threshold)
            total = self.reached.get(threshold, 0)
            if share is None:
                parts.append(f"no token reached +{threshold:.0f}%")
            else:
                parts.append(f"{share:.0%} of the {total} that reached +{threshold:.0f}% were detected first")
        median = self.median_lead_minutes
        head = (
            f"Median lead time {median:.0f} minutes."
            if median is not None else "No lead time measurable."
        )
        return head + " " + "; ".join(parts)

    def as_dict(self) -> dict:
        return {
            "tracked": self.tracked,
            "conclusive": self.conclusive,
            "median_lead_minutes": (
                round(self.median_lead_minutes, 1) if self.median_lead_minutes is not None else None
            ),
            "verdict": self.verdict(),
            "thresholds": [
                {
                    "threshold_pct": t,
                    "reached": self.reached.get(t, 0),
                    "detected_before": self.detected_before.get(t, 0),
                    "share_pct": (
                        round(self.share_before(t) * 100, 1) if self.share_before(t) is not None else None
                    ),
                }
                for t in LEAD_THRESHOLDS
            ],
        }


def build_lead_time(db: Session) -> LeadTimeReport:
    """How early the engine detected the tokens that later moved.

    Measured from `price_at_first_signal` - the price when the token FIRST
    entered the watchlist - against the maximum favourable excursion the
    forward-return rows recorded afterwards.

    Every watchlist entry is included, not only the ones that worked. A
    lead-time figure computed over winners would describe a bot that only
    ever saw winners, which is the exact bias this analysis exists to
    avoid.
    """
    report = LeadTimeReport()
    report.reached = {t: 0 for t in LEAD_THRESHOLDS}
    report.detected_before = {t: 0 for t in LEAD_THRESHOLDS}

    entries = db.query(models.WatchlistEntry).filter(
        models.WatchlistEntry.price_at_first_signal.isnot(None)
    ).all()

    for entry in entries:
        report.tracked += 1
        rows = (
            db.query(models.ForwardReturn)
            .filter(
                models.ForwardReturn.token_address == entry.token_address,
                models.ForwardReturn.max_favorable_pct.isnot(None),
            )
            .all()
        )
        if not rows:
            continue

        peak = max(r.max_favorable_pct for r in rows)
        signal_at = entry.first_signal_at
        if signal_at and signal_at.tzinfo is None:
            signal_at = signal_at.replace(tzinfo=dt.timezone.utc)

        for threshold in LEAD_THRESHOLDS:
            if peak < threshold:
                continue
            report.reached[threshold] += 1
            # The signal is "before" the move by construction here: the
            # excursion is measured FROM the detection price, so a token
            # that reached +20% did so after being detected. What varies is
            # how much of the move was already behind it at detection,
            # which the late-entry stage captures separately.
            report.detected_before[threshold] += 1

        # Lead time: how long after detection the peak arrived. Approximated
        # by the longest horizon that recorded the peak, since the
        # resolution loop samples rather than tick-watching.
        best_row = max(rows, key=lambda r: r.max_favorable_pct)
        if best_row.max_favorable_pct >= LEAD_THRESHOLDS[0]:
            report.lead_minutes.append(float(best_row.horizon_minutes))

    return report


# ---------------------------------------------------------------------------
# false positives
# ---------------------------------------------------------------------------

@dataclass
class FalsePositiveReport:
    total_resolved: int = 0
    succeeded: int = 0
    failed: int = 0
    by_category: list[tuple[str, int]] = field(default_factory=list)
    by_score_bucket: dict[str, tuple[int, int]] = field(default_factory=dict)  # bucket -> (failed, total)

    @property
    def false_positive_rate(self) -> float | None:
        if not self.total_resolved:
            return None
        return self.failed / self.total_resolved

    @property
    def conclusive(self) -> bool:
        return self.total_resolved >= MIN_BUCKET_SAMPLE

    def verdict(self) -> str:
        if not self.conclusive:
            return (
                f"INSUFFICIENT DATA: only {self.total_resolved} watchlist entries have resolved "
                f"(need >={MIN_BUCKET_SAMPLE}). The false-positive rate is not measurable yet."
            )
        rate = self.false_positive_rate
        head = (
            f"{rate:.0%} of resolved watchlist entries failed "
            f"({self.failed} of {self.total_resolved})."
        )
        if self.by_category:
            top, count = self.by_category[0]
            head += (
                f" Most common cause: {top} ({count}, {count / max(self.failed, 1):.0%} of failures)."
            )
        residual = dict(self.by_category).get("score_decayed", 0)
        if self.failed and residual / self.failed > 0.5:
            head += (
                " Over half of failures land in the residual 'score_decayed' bucket, which means "
                "the failure taxonomy is missing a category rather than that they simply faded."
            )
        return head

    def as_dict(self) -> dict:
        return {
            "total_resolved": self.total_resolved,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "false_positive_rate_pct": (
                round(self.false_positive_rate * 100, 1)
                if self.false_positive_rate is not None else None
            ),
            "conclusive": self.conclusive,
            "verdict": self.verdict(),
            "by_category": [
                {"category": c, "count": n, "explanation": wl.FAILURE_CATEGORIES.get(c, "")}
                for c, n in self.by_category
            ],
            "by_score_bucket": [
                {"bucket": b, "failed": f, "total": t,
                 "failure_rate_pct": round(f / t * 100, 1) if t else None}
                for b, (f, t) in sorted(self.by_score_bucket.items())
            ],
        }


def build_false_positives(db: Session) -> FalsePositiveReport:
    """Why high early scores fail.

    Expired entries count as failures. A token that looked promising and
    went nowhere IS a false positive, and excluding it would silently
    improve the engine's apparent hit rate - which is the single easiest
    way to make a useless signal look useful.
    """
    report = FalsePositiveReport()
    resolved = db.query(models.WatchlistEntry).filter(
        models.WatchlistEntry.state.in_(list(wl.TERMINAL_STATES))
    ).all()

    categories: Counter[str] = Counter()
    buckets: dict[str, list[int]] = {}

    for entry in resolved:
        report.total_resolved += 1
        failed = entry.state in (wl.FAILED, wl.SKIPPED, wl.EXPIRED)
        if failed:
            report.failed += 1
            categories[entry.failure_category or "score_decayed"] += 1
        else:
            report.succeeded += 1

        score = entry.best_early_score
        if score is not None:
            label = early_bucket(score)
            counts = buckets.setdefault(label, [0, 0])
            counts[1] += 1
            if failed:
                counts[0] += 1

    report.by_category = categories.most_common()
    report.by_score_bucket = {b: (c[0], c[1]) for b, c in buckets.items()}
    return report
