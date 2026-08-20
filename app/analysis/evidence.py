"""One report answering: is there enough evidence to trust this strategy?

Every section can come back empty, and an empty section says so in words
rather than printing a zero. That distinction is the whole design. A
report that renders "expectancy: 0.00R, win rate: 0%" over no trades looks
like a measurement of a bad strategy, when it is the absence of a
measurement of any strategy - and those lead to opposite decisions.

So every number here carries its sample size, and any statistic computed
over fewer observations than its floor is withheld rather than shown
faintly. Nothing is estimated, interpolated, or filled forward.

CORRUPTED ROWS ARE REMOVED BEFORE ANYTHING IS COMPUTED

app/analysis/integrity.py runs first and its exclusions are honoured here,
because an average over a dataset containing a duplicate or a leaked
resolution is wrong in a direction nobody checked.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis import integrity
from app.analysis.calibration import round_trip_cost_pct
from app.config import settings

# Floors below which a statistic is withheld rather than shown.
MIN_FOR_A_NUMBER = 20
MIN_PER_REGIME = 15


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


@dataclass
class Measure:
    """A number that knows how many observations it came from."""
    label: str
    value: float | None
    samples: int
    unit: str = ""
    floor: int = MIN_FOR_A_NUMBER

    @property
    def trustworthy(self) -> bool:
        return self.value is not None and self.samples >= self.floor

    def render(self) -> str:
        if self.value is None:
            return f"{self.label:<26} not measurable          (n={self.samples})"
        marker = "" if self.trustworthy else "  *below floor"
        return f"{self.label:<26} {self.value:>10.3f}{self.unit:<6} (n={self.samples}){marker}"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "value": round(self.value, 4) if self.value is not None else None,
            "samples": self.samples,
            "unit": self.unit,
            "trustworthy": self.trustworthy,
        }


@dataclass
class EvidenceReport:
    champion: str | None = None
    challengers: list[str] = field(default_factory=list)
    measures: list[Measure] = field(default_factory=list)
    by_regime: dict[str, Measure] = field(default_factory=dict)
    # label -> which axis it belongs to (trend / volatility / liquidity).
    # Needed because "two regimes" must mean two values ON ONE AXIS. One
    # market condition expands into three axis labels, and counting those
    # as three regimes would let a single condition satisfy a bar that
    # exists precisely to require more than one.
    regime_axis: dict[str, str] = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    weaknesses: list[str] = field(default_factory=list)
    next_experiment: str = ""
    generated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    @property
    def completed_samples(self) -> int:
        expectancy = next((m for m in self.measures if m.label == "expectancy (net)"), None)
        return expectancy.samples if expectancy else 0

    @property
    def contrasting_axes(self) -> list[str]:
        """Axes where two or more DIFFERENT conditions each have a sample.

        This is what "measured across market conditions" has to mean. A
        book traded entirely in one bull/normal/deep condition produces
        three axis labels, and treating that as three regimes would report
        variety where there is none.
        """
        per_axis: dict[str, int] = defaultdict(int)
        for label, measure in self.by_regime.items():
            if measure.samples >= MIN_PER_REGIME:
                per_axis[self.regime_axis.get(label, "?")] += 1
        return [axis for axis, count in per_axis.items() if count >= 2]

    @property
    def promotion_ready(self) -> bool:
        """Is there enough evidence to trust ANY conclusion yet?

        Requires the overall sample, genuine variation on at least one
        axis, and a dataset that is not mostly exclusions. Deliberately
        conservative: this answers "may we start drawing conclusions",
        not "is the strategy good".
        """
        if self.completed_samples < MIN_FOR_A_NUMBER:
            return False
        if not self.contrasting_axes:
            return False
        rate = (self.integrity.get("forward_returns") or {}).get("exclusion_rate_pct")
        return not (rate is not None and rate >= 20.0)

    def readiness(self) -> str:
        if self.completed_samples == 0:
            return (
                "NOT READY - no completed observations. Nothing in this report is a "
                "finding about the strategy; it is a description of an empty dataset."
            )
        if self.promotion_ready:
            return (
                f"READY to begin drawing conclusions on {self.completed_samples} "
                "observations across multiple regimes. That is permission to READ the "
                "numbers, not a verdict that the strategy works."
            )
        missing = []
        if self.completed_samples < MIN_FOR_A_NUMBER:
            missing.append(f"{MIN_FOR_A_NUMBER - self.completed_samples} more observations")
        if not self.contrasting_axes:
            missing.append(
                f"a contrasting market condition with {MIN_PER_REGIME}+ observations - "
                "everything so far comes from one, so nothing separates an edge from a "
                "bet on those conditions holding"
            )
        return "NOT READY - still needs " + ", and ".join(missing) + "."

    def as_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "champion": self.champion,
            "challengers": self.challengers,
            "completed_samples": self.completed_samples,
            "promotion_ready": self.promotion_ready,
            "contrasting_axes": self.contrasting_axes,
            "readiness": self.readiness(),
            "measures": [m.as_dict() for m in self.measures],
            "by_regime": {k: v.as_dict() for k, v in self.by_regime.items()},
            "integrity": self.integrity,
            "weaknesses": self.weaknesses,
            "next_experiment": self.next_experiment,
        }

    def render(self) -> str:
        lines = [
            "=" * 78,
            " EVIDENCE REPORT",
            "=" * 78,
            f" generated       {self.generated_at.isoformat(timespec='seconds')}",
            f" champion        {self.champion or 'none registered'}",
            f" challengers     {', '.join(self.challengers) if self.challengers else 'none'}",
            "",
            f" PROMOTION READINESS: {self.readiness()}",
            "",
            " MEASURES (sample size beside every number)",
        ]
        for m in self.measures:
            lines.append("   " + m.render())

        lines.append("\n SAMPLES BY MARKET REGIME")
        if not self.by_regime:
            lines.append("   none - no observation carries a regime label yet")
        else:
            for name, m in sorted(self.by_regime.items()):
                lines.append("   " + m.render())

        lines.append("\n DATA QUALITY")
        for table, report in self.integrity.items():
            lines.append(f"   {table}: {report.get('verdict', 'unchecked')}")
            for code, count in sorted((report.get("by_code") or {}).items(), key=lambda kv: -kv[1]):
                lines.append(f"      {count:>5}  {code}")
            for warning in report.get("warnings", []):
                lines.append(f"      WARNING: {warning}")

        lines.append("\n PRIMARY WEAKNESSES")
        for w in self.weaknesses:
            lines.append(f"   - {w}")

        lines.append(f"\n RECOMMENDED NEXT EXPERIMENT\n   {self.next_experiment}")
        lines.append(
            "\n Every number above carries its sample size. Anything marked *below floor\n"
            " is arithmetic, not evidence. Nothing here is estimated or filled forward:\n"
            " a section with no data says so rather than printing a zero."
        )
        return "\n".join(lines)


def build_evidence_report(db: Session, *, horizon_minutes: int = 60) -> EvidenceReport:
    """Assemble the report from clean rows only."""
    report = EvidenceReport()

    checks = integrity.check_all(db)
    report.integrity = {name: r.as_dict() for name, r in checks.items()}
    excluded = checks["forward_returns"].excluded_ids

    version = (
        db.query(models.StrategyVersion)
        .order_by(models.StrategyVersion.last_seen_at.desc())
        .first()
    )
    report.champion = version.label if version else None

    rows = [
        r for r in db.query(models.ForwardReturn).filter(
            models.ForwardReturn.horizon_minutes == horizon_minutes,
            models.ForwardReturn.return_pct.isnot(None),
        ).all()
        if r.id not in excluded
    ]
    returns = [r.return_pct for r in rows]
    cost = round_trip_cost_pct() * 100
    net = [r - cost for r in returns]

    total_scored = db.query(models.ForwardReturn).count()
    accepted = (
        db.query(models.Position)
        .filter(models.Position.status == models.PositionStatus.CLOSED.value)
        .count()
    )

    report.measures = [
        Measure("evaluated signals", float(total_scored), total_scored, floor=1),
        Measure("completed paper trades", float(accepted), accepted, floor=1),
        Measure("expectancy (net)", _mean(net), len(net), "%"),
        Measure("median return", _median(returns), len(returns), "%"),
        Measure(
            "win rate",
            (sum(1 for r in net if r > 0) / len(net) * 100) if net else None,
            len(net), "%",
        ),
        Measure(
            "MFE (avg)",
            _mean([r.max_favorable_pct for r in rows if r.max_favorable_pct is not None]),
            sum(1 for r in rows if r.max_favorable_pct is not None), "%",
        ),
        Measure(
            "MAE (avg)",
            _mean([r.max_adverse_pct for r in rows if r.max_adverse_pct is not None]),
            sum(1 for r in rows if r.max_adverse_pct is not None), "%",
        ),
    ]

    grouped: dict[str, list[float]] = defaultdict(list)
    axis_names = ("trend", "volatility", "liquidity")
    for row in rows:
        if not row.market_regime:
            continue
        # One axis at a time. The full cross product is 36 cells and this
        # bot will never fill them; slicing that finely produces cells of
        # three trades and a table that looks authoritative over nothing.
        for position, label in enumerate(row.market_regime.split("/")):
            grouped[label].append(row.return_pct - cost)
            if position < len(axis_names):
                report.regime_axis[label] = axis_names[position]
    report.by_regime = {
        name: Measure(name, _mean(values), len(values), "%", floor=MIN_PER_REGIME)
        for name, values in grouped.items()
    }

    # --- weaknesses, stated plainly ---------------------------------------
    if not rows:
        report.weaknesses.append(
            "No resolved observations at this horizon. Every metric above is empty, and "
            "an empty metric is not a bad result - it is the absence of one."
        )
    if not grouped:
        report.weaknesses.append(
            "No observation carries a market regime, so per-regime comparison is "
            "impossible and the promotion gate's consistency bar can never pass."
        )
    thin = [n for n, m in report.by_regime.items() if m.samples < MIN_PER_REGIME]
    if thin:
        report.weaknesses.append(
            f"Regimes below the {MIN_PER_REGIME}-observation floor: {', '.join(sorted(thin))}. "
            "An edge measured only where the sample is thick is a bet on those conditions."
        )
    fr_rate = report.integrity["forward_returns"].get("exclusion_rate_pct") or 0.0
    if fr_rate >= 5.0:
        report.weaknesses.append(
            f"{fr_rate:.1f}% of forward returns were excluded as corrupt. Fixing the "
            "pipeline comes before reading any statistic derived from what is left."
        )
    if settings.LIVE_TRADING:
        report.weaknesses.append("LIVE_TRADING is enabled - this report describes real money.")

    # --- what to do next ---------------------------------------------------
    if not rows:
        report.next_experiment = (
            "Run the bot in paper and collect observations. No experiment is meaningful "
            "before there is a dataset - and the first honest question is whether the "
            "score separates outcomes at all, not which threshold is best."
        )
    elif len(rows) < MIN_FOR_A_NUMBER:
        report.next_experiment = (
            f"Keep collecting: {MIN_FOR_A_NUMBER - len(rows)} more resolved observations "
            "before any number here is worth acting on."
        )
    elif not report.contrasting_axes:
        report.next_experiment = (
            "Collect through a different market condition. Everything measured so far "
            "comes from one regime, so nothing distinguishes an edge from a regime bet."
        )
    else:
        report.next_experiment = (
            "Run `research.py replay` and `research.py calibration` on this dataset. "
            "Whether the score ranks outcomes at all comes before tuning any threshold."
        )
    return report
