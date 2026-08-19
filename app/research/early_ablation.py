"""Leave-one-out over the EARLY signal's factors, replayed on stored data.

This is deliberately not the same machine as app/research/ablation.py.
That one re-runs the whole strategy over a candle series, which works for
the technical score because every one of its factors can be recomputed
from candles. The early score cannot be tested that way: four of its nine
factors (transaction acceleration, buy pressure, volume quality,
liquidity quality) come from differencing successive market snapshots,
and a historical candle series does not carry snapshots. A candle
backtest would silently score those factors as unavailable and then
report that removing them changes nothing - a conclusion about the
backtest, not about the factors.

So this reads what the bot actually recorded. Every ForwardReturn row
carries the feature values AS THEY WERE at signal time plus the return
that followed, which is exactly the (features, outcome) pair an ablation
needs. Re-scoring those stored features with one factor's weight zeroed
gives what the engine WOULD have decided without that factor, with no
look-ahead anywhere: nothing is recomputed from data that arrived later.

WHAT IS BEING MEASURED

Not "does the score change" - zeroing a weight always changes the score.
The question is whether the score still ORDERS the outcomes. Two measures,
because either one alone is misleading:

  rank correlation (primary)  Spearman between score and realised return
                              over every row. Boundary-free, so a factor
                              that merely shifts every score up or down
                              cannot register as a loss of information.

  bucket separation (shown)   after-cost expectancy of the top bucket
                              minus the bottom. Reads directly against the
                              calibration table, but it is measured across
                              fixed bucket EDGES, so removing a factor can
                              collapse two groups into one bucket and show
                              a large drop that is about the edges rather
                              than about the factor. It is reported for
                              continuity with /early and never used to
                              decide the verdict.

    removing it HURTS       the factor was carrying information
    removing it does NOTHING   redundant with the rest; it costs a data
                            dependency and buys nothing
    removing it HELPS       the factor was adding noise

The third case is ordinary and is the reason this exists.

WHAT IT CANNOT TELL YOU

Re-scoring stored features answers "would a different weighting have
ranked these same candidates better". It does NOT answer "would the bot
have found different candidates", because a lower early score can change
which tokens reach the watchlist at all, and those counterfactual tokens
have no stored rows. The sample is the sample the live engine produced.
That is a real limitation and it is reported on the result rather than
left for the reader to infer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis.early_calibration import EARLY_BUCKETS, MIN_BUCKET_SAMPLE, early_bucket
from app.early.features import EarlyFeatures
from app.early.score import DEFAULT_WEIGHTS, score_early_opportunity
from app.analysis.calibration import round_trip_cost_pct

FULL_MODEL = "full model"

# Which stored features each factor actually reads. Kept explicitly rather
# than inferred, because it is what lets the report distinguish "removing
# this factor changed nothing" from "this factor had no data to remove".
# Those look identical in the numbers and mean opposite things: the first
# is a finding about the factor, the second is a finding about the dataset.
# Mirrors the f.value(...) calls in app/early/score.py.
FACTOR_FEATURES: dict[str, tuple[str, ...]] = {
    "volume_acceleration": ("volume_accel_short", "volume_accel_medium"),
    "transaction_acceleration": ("txn_rate_change",),
    "buy_pressure": ("buy_pressure", "buy_pressure_persistence", "buy_pressure_change"),
    "volume_quality": ("volume_steadiness",),
    "liquidity_quality": ("liquidity_growth", "liquidity_stability"),
    "momentum_acceleration": (
        "ema_slope", "rsi_crossing_up", "rsi_level", "macd_histogram_expanding",
    ),
    "price_structure": ("range_compression", "higher_lows", "acceleration_smoothness"),
    "breakout_position": ("breakout_proximity",),
    "relative_volume": ("relative_volume",),
}


def coverage(samples: list["Sample"], factor: str) -> float | None:
    """Share of stored rows on which this factor had ANY data at all.

    None when the factor is not in FACTOR_FEATURES, which would mean the
    map has drifted from the scorers - reported as unknown rather than
    silently as zero.
    """
    names = FACTOR_FEATURES.get(factor)
    if names is None:
        return None
    if not samples:
        return 0.0
    seen = sum(
        1 for s in samples
        if any(s.features.get(name).available for name in names)
    )
    return seen / len(samples)

# A rank-correlation change smaller than this is noise on any sample this
# bot will realistically have. Reported as "no measurable effect", not as
# a small effect.
NEGLIGIBLE_RHO = 0.02

# Below this many rows, a rank correlation is a property of the handful of
# points rather than of the score.
MIN_CORRELATION_ROWS = 40


def early_weights_without(factor: str, base: dict[str, float] | None = None) -> dict[str, float]:
    """The early weight map with `factor` neutralised.

    Zeroed rather than deleted, and NOT manually renormalised:
    score_early_opportunity divides by the sum of the weights it was
    given, so a zero weight already drops out of both the numerator and
    the denominator. Renormalising on top of that would be a no-op at
    best; deleting the key would change which factors are scored at all.
    """
    weights = dict(base or DEFAULT_WEIGHTS)
    if factor not in weights:
        raise KeyError(f"{factor} is not an early-score factor")
    weights[factor] = 0.0
    if sum(weights.values()) <= 0:
        raise ValueError("cannot ablate the only weighted factor")
    return weights


@dataclass
class Sample:
    """One stored (features at signal time, realised return) pair."""
    features: EarlyFeatures
    return_pct: float


def load_samples(db: Session, *, horizon_minutes: int) -> list[Sample]:
    """Stored feature/outcome pairs for one horizon.

    Rows without stored features are skipped rather than scored on an
    empty feature set. An empty feature set is not a neutral input - it
    would produce a score built entirely from unavailable factors and drag
    every bucket toward the same number.
    """
    rows = (
        db.query(models.ForwardReturn)
        .filter(
            models.ForwardReturn.horizon_minutes == horizon_minutes,
            models.ForwardReturn.return_pct.isnot(None),
            models.ForwardReturn.early_features.isnot(None),
        )
        .all()
    )
    samples = []
    for row in rows:
        features = EarlyFeatures.from_dict(row.early_features)
        if features.features:
            samples.append(Sample(features=features, return_pct=row.return_pct))
    return samples


def separation_pct(samples: list[Sample], weights: dict[str, float], cost_pct: float) -> float | None:
    """Top-bucket minus bottom-bucket after-cost expectancy, in percent.

    None when fewer than two buckets have enough measured outcomes - which
    is the normal state early on and must not be reported as zero
    separation. "We cannot tell yet" and "there is no difference" are
    opposite conclusions.
    """
    grouped: dict[str, list[float]] = {b: [] for b in EARLY_BUCKETS}
    for sample in samples:
        result = score_early_opportunity(sample.features, weights=weights)
        grouped[early_bucket(result.score)].append(sample.return_pct - cost_pct)

    # EARLY_BUCKETS is ordered low to high, so the first and last usable
    # entries are the bottom and top buckets.
    usable = [grouped[b] for b in EARLY_BUCKETS if len(grouped[b]) >= MIN_BUCKET_SAMPLE]
    if len(usable) < 2:
        return None
    low = sum(usable[0]) / len(usable[0])
    high = sum(usable[-1]) / len(usable[-1])
    return high - low


def _ranks(values: list[float]) -> list[float]:
    """Fractional ranks, ties averaged.

    Ties are averaged rather than broken arbitrarily. Breaking them by
    position would let the row ORDER leak into the correlation, which on a
    chronologically-sorted dataset is a direct route to inventing a
    relationship that is really just time.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def rank_correlation(samples: list[Sample], weights: dict[str, float]) -> float | None:
    """Spearman correlation between the re-scored early score and the return.

    None below MIN_CORRELATION_ROWS, and None when either side is entirely
    constant - a correlation with a constant is undefined, not zero, and
    reporting it as zero would read as "no relationship" when the truth is
    "no variation to relate".
    """
    if len(samples) < MIN_CORRELATION_ROWS:
        return None
    scores = [score_early_opportunity(s.features, weights=weights).score for s in samples]
    returns = [s.return_pct for s in samples]
    if len(set(scores)) < 2 or len(set(returns)) < 2:
        return None

    x, y = _ranks(scores), _ranks(returns)
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    dx = [v - mean_x for v in x]
    dy = [v - mean_y for v in y]
    denominator = (sum(v * v for v in dx) * sum(v * v for v in dy)) ** 0.5
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


@dataclass
class EarlyFactorVerdict:
    factor: str
    weight: float
    separation_pct: float | None
    baseline_pct: float | None
    coverage: float | None = None      # share of rows where the factor had data
    rho: float | None = None           # rank correlation without this factor
    baseline_rho: float | None = None

    @property
    def separation_delta(self) -> float | None:
        if self.separation_pct is None or self.baseline_pct is None:
            return None
        return self.separation_pct - self.baseline_pct

    @property
    def delta(self) -> float | None:
        """Change in rank correlation. This is what decides the verdict."""
        if self.rho is None or self.baseline_rho is None:
            return None
        return self.rho - self.baseline_rho

    @property
    def verdict(self) -> str:
        if self.coverage == 0.0:
            return "NO DATA - never measurable"
        delta = self.delta
        if delta is None:
            return "untested"
        if delta < -NEGLIGIBLE_RHO:
            return "removing it HURTS"
        if delta > NEGLIGIBLE_RHO:
            return "removing it HELPS"
        return "no measurable effect"

    @property
    def recommendation(self) -> str:
        if self.coverage == 0.0:
            return (
                "this factor had no data on any stored row - removing it proves nothing "
                "about it, only that removing an absent factor changes nothing"
            )
        delta = self.delta
        if delta is None:
            return "not enough measured outcomes to test this factor"
        if delta < -NEGLIGIBLE_RHO:
            return "keep - it is carrying information"
        if delta > NEGLIGIBLE_RHO:
            return "cut it, or cut its weight - it is adding noise"
        return "redundant with the other factors on this sample"

    def as_dict(self) -> dict:
        return {
            "factor": self.factor,
            "weight": self.weight,
            "coverage_pct": round(self.coverage * 100, 1) if self.coverage is not None else None,
            "rank_correlation": round(self.rho, 4) if self.rho is not None else None,
            "delta_rank_correlation": round(self.delta, 4) if self.delta is not None else None,
            "separation_pct": round(self.separation_pct, 3) if self.separation_pct is not None else None,
            "delta_separation_pct": (
                round(self.separation_delta, 3) if self.separation_delta is not None else None
            ),
            "verdict": self.verdict,
            "recommendation": self.recommendation,
        }


@dataclass
class EarlyAblationReport:
    horizon_minutes: int
    samples: int = 0
    baseline_pct: float | None = None
    baseline_rho: float | None = None
    factors: list[EarlyFactorVerdict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def conclusive(self) -> bool:
        return self.baseline_rho is not None and self.samples >= MIN_CORRELATION_ROWS

    def summary(self) -> str:
        if not self.conclusive:
            return (
                f"INSUFFICIENT DATA at {self.horizon_minutes}m: {self.samples} stored "
                f"feature/outcome pairs (need >={MIN_CORRELATION_ROWS} with varying scores "
                "and varying returns). No factor has been tested. This is the expected state "
                "until the bot has run for a while - it is not a finding about any factor."
            )
        separation = (
            f"{self.baseline_pct:+.2f}%" if self.baseline_pct is not None
            else f"unmeasurable (fewer than two buckets reach {MIN_BUCKET_SAMPLE} rows)"
        )
        lines = [
            f"Baseline at {self.horizon_minutes}m over {self.samples} stored pairs: "
            f"rank correlation {self.baseline_rho:+.3f}, bucket separation {separation}.",
            "",
            f"  {'factor':<28}{'weight':>8}{'data %':>8}{'rho':>8}{'d rho':>8}"
            f"{'d sep %':>9}  verdict",
        ]
        for f in sorted(self.factors, key=lambda x: (x.delta is None, x.delta or 0)):
            rho = f"{f.rho:+.3f}" if f.rho is not None else "n/a"
            delta = f"{f.delta:+.3f}" if f.delta is not None else "n/a"
            sep = (
                f"{f.separation_delta:+.2f}" if f.separation_delta is not None else "n/a"
            )
            cov = f"{f.coverage * 100:.0f}" if f.coverage is not None else "?"
            lines.append(
                f"  {f.factor:<28}{f.weight:>8.2f}{cov:>8}{rho:>8}{delta:>8}{sep:>9}  {f.verdict}"
            )
        lines.append("")
        lines.append(
            "  The verdict comes from 'd rho' (change in rank correlation), not from\n"
            "  'd sep %'. Bucket separation is measured across fixed score EDGES, so removing\n"
            "  a factor can collapse two groups into one bucket and post a large drop that is\n"
            "  about the edges rather than about the factor. It is shown for continuity with\n"
            "  the /early calibration table and nothing more."
        )
        lines.append("")
        lines.append(
            "  Re-scored from features stored AT SIGNAL TIME, so there is no look-ahead. But\n"
            "  this only asks whether a different weighting would have RANKED these same\n"
            "  candidates better - it cannot show which tokens a different weighting would\n"
            "  have found, because those tokens were never recorded."
        )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "samples": self.samples,
            "conclusive": self.conclusive,
            "baseline_rank_correlation": (
                round(self.baseline_rho, 4) if self.baseline_rho is not None else None
            ),
            "baseline_separation_pct": (
                round(self.baseline_pct, 3) if self.baseline_pct is not None else None
            ),
            "summary": self.summary(),
            "factors": [f.as_dict() for f in self.factors],
            "warnings": list(self.warnings),
        }


def run_early_ablation(
    db: Session, *, horizon_minutes: int = 60, weights: dict[str, float] | None = None
) -> EarlyAblationReport:
    """Leave-one-out over the early factors, on the bot's own recorded data."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    cost_pct = round_trip_cost_pct() * 100
    samples = load_samples(db, horizon_minutes=horizon_minutes)

    report = EarlyAblationReport(horizon_minutes=horizon_minutes, samples=len(samples))
    report.baseline_pct = separation_pct(samples, weights, cost_pct)
    report.baseline_rho = rank_correlation(samples, weights)

    for factor, weight in weights.items():
        if weight <= 0:
            continue
        ablated = early_weights_without(factor, weights)
        report.factors.append(
            EarlyFactorVerdict(
                factor=factor,
                weight=weight,
                separation_pct=separation_pct(samples, ablated, cost_pct),
                baseline_pct=report.baseline_pct,
                coverage=coverage(samples, factor),
                rho=rank_correlation(samples, ablated),
                baseline_rho=report.baseline_rho,
            )
        )

    absent = [f.factor for f in report.factors if f.coverage == 0.0]
    if absent:
        report.warnings.append(
            "these factors had NO data on any stored row: " + ", ".join(absent)
            + ". Their ablation result says nothing about the factor - only that removing "
            "an already-absent factor changes nothing. Fix the data source before reading "
            "them as redundant."
        )
    return report
