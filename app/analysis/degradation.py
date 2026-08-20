"""Has the strategy stopped working?

Not "is it losing money today" - that is noise. The question is whether
the recent behaviour has moved away from the behaviour that was validated,
and it has to be asked about more than the P&L line, because P&L is the
last thing to move.

The early symptoms show up in the path, not the result:

    entries stop reaching as far           falling MFE
    entries take more heat before working  deepening MAE
    fills get worse                        rising slippage
    losing streaks get longer              deeper cumulative drawdown

A strategy can hold its expectancy for weeks while all four of those decay,
because a few outsized winners paper over it. By the time the mean moves,
the change is old.

WHAT "BASELINE" MEANS HERE

The earlier trades of the SAME strategy version. Not a hand-set target, and
never trades from a different version - a threshold that moved mid-run
makes the two halves different strategies, and comparing them measures the
edit rather than the decay.

That does mean the baseline is only as good as the run that produced it. A
strategy that never worked will not be reported as degrading, because it
has nothing to degrade from. That is correct: this file answers "has it
changed", and app/analysis/calibration.py answers "was it ever any good".

IT REPORTS. IT DOES NOT REACT.

No threshold is touched, no filter is loosened, nothing is halted. A
degradation finding is a reason for a person to look, and any change that
follows goes through the promotion gate as a challenger on its own paired
sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis.postmortem import PostMortem, build_postmortem
from app.autopilot.promote import bootstrap_p_value
from app.strategy.version import current_label

# Trades needed on each side before the comparison runs at all.
MIN_BASELINE = 30
MIN_RECENT = 20

# The trailing window treated as "recent". Larger than MIN_RECENT so a
# comparison, once possible, is not decided by three trades.
RECENT_TRADES = 30

# Per-group floor for the regime and liquidity breakdowns. Lower than the
# overall floor because a regime split necessarily thins the sample - and
# the finding is directional, not a promotion.
MIN_GROUP = 12

# How far a resampled difference must be from chance before the shift is
# called degradation rather than noise.
ALPHA = 0.05

# A shift must also be MATERIAL, not merely real. The bootstrap answers
# "is this difference distinguishable from chance?", and for a metric that
# barely moves the answer can be an emphatic yes about a meaningless
# amount: a slippage figure that is constant to fifteen decimal places
# still differs in the sixteenth, and resampling a constant reproduces
# that difference every single time, giving p=0.000 on a shift of 2e-16.
#
# So the change has to clear a fraction of the baseline's own magnitude
# before it is a candidate at all. Relative rather than absolute because
# these metrics live on completely different scales - a 0.3-point move is
# nothing in MFE and enormous in slippage.
MIN_RELATIVE_SHIFT = 0.05

# Floor for the relative test, so a baseline at or near zero does not make
# every rounding artefact material.
MIN_ABSOLUTE_SHIFT = 0.01

# Never inspect more than this much history in one pass.
MAX_TRADES = 500

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass
class MetricShift:
    """One measurement, before and after."""
    name: str
    direction: str
    baseline: float | None
    recent: float | None
    baseline_n: int
    recent_n: int
    p_value: float | None = None
    unit: str = "%"

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.recent is None:
            return None
        return self.recent - self.baseline

    @property
    def materiality(self) -> float:
        """How large a move has to be before it is worth testing at all."""
        base = abs(self.baseline) if self.baseline is not None else 0.0
        return max(base * MIN_RELATIVE_SHIFT, MIN_ABSOLUTE_SHIFT)

    @property
    def worse(self) -> bool:
        """Moved the bad way by a material amount. Not yet a finding.

        Materiality is checked HERE rather than only at the p-value,
        because a metric that hardly moves can produce an overwhelming
        p-value about an irrelevant difference - see MIN_RELATIVE_SHIFT.
        """
        if self.delta is None or abs(self.delta) < self.materiality:
            return False
        return self.delta < 0 if self.direction == HIGHER_IS_BETTER else self.delta > 0

    @property
    def degraded(self) -> bool:
        """Moved in the bad direction by more than resampling explains."""
        return self.worse and self.p_value is not None and self.p_value <= ALPHA

    def render(self) -> str:
        if self.baseline is None or self.recent is None:
            return f"  {self.name:<22} not measurable (n={self.baseline_n}/{self.recent_n})"
        flag = "  DEGRADED" if self.degraded else ("  worse" if self.worse else "")
        p = f"  p={self.p_value:.3f}" if self.p_value is not None else ""
        return (
            f"  {self.name:<22} {self.baseline:>9.2f}{self.unit} -> {self.recent:>9.2f}{self.unit}"
            f"  ({self.delta:+.2f}){p}{flag}"
        )

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 4) if v is not None else None
        return {
            "name": self.name, "direction": self.direction,
            "baseline": r(self.baseline), "recent": r(self.recent),
            "delta": r(self.delta), "baseline_n": self.baseline_n,
            "recent_n": self.recent_n, "p_value": r(self.p_value),
            "worse": self.worse, "degraded": self.degraded,
        }


@dataclass
class GroupShift:
    """Expectancy before and after, within one market condition."""
    group: str
    axis: str
    baseline: float | None
    recent: float | None
    baseline_n: int
    recent_n: int

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.recent is None:
            return None
        return self.recent - self.baseline

    @property
    def comparable(self) -> bool:
        return self.baseline_n >= MIN_GROUP and self.recent_n >= MIN_GROUP

    def as_dict(self) -> dict:
        def r(v):
            return round(v, 4) if v is not None else None
        return {"group": self.group, "axis": self.axis, "baseline": r(self.baseline),
                "recent": r(self.recent), "delta": r(self.delta),
                "baseline_n": self.baseline_n, "recent_n": self.recent_n,
                "comparable": self.comparable}


@dataclass
class DegradationReport:
    strategy_version: str | None = None
    total_trades: int = 0
    baseline_n: int = 0
    recent_n: int = 0
    shifts: list[MetricShift] = field(default_factory=list)
    groups: list[GroupShift] = field(default_factory=list)
    note: str = ""

    @property
    def comparable(self) -> bool:
        return self.baseline_n >= MIN_BASELINE and self.recent_n >= MIN_RECENT

    @property
    def degraded(self) -> list[MetricShift]:
        return [s for s in self.shifts if s.degraded]

    @property
    def degraded_groups(self) -> list[GroupShift]:
        """Conditions where expectancy fell, among those with enough data.

        Reported without a significance test and labelled directional: a
        per-regime split is thin by construction, and dressing it up with a
        p-value on twelve trades would be the false confidence this whole
        apparatus exists to avoid.
        """
        return [g for g in self.groups
                if g.comparable and g.delta is not None and g.delta < 0]

    def verdict(self) -> str:
        if self.note:
            return self.note
        if not self.comparable:
            return (
                f"INSUFFICIENT_DATA: {self.baseline_n} baseline / {self.recent_n} recent closed "
                f"trades on {self.strategy_version} (need {MIN_BASELINE}/{MIN_RECENT}). Nothing "
                "here is a statement about the strategy."
            )
        if not self.degraded:
            drifting = [s.name for s in self.shifts if s.worse]
            if drifting:
                return (
                    "No metric has degraded beyond what resampling explains. Drifting the wrong "
                    f"way but within noise: {', '.join(drifting)}. Worth re-checking as the "
                    "sample grows."
                )
            return "No sign of degradation: nothing has moved the wrong way."
        names = ", ".join(s.name for s in self.degraded)
        conditions = ", ".join(g.group for g in self.degraded_groups)
        where = f" Weakest conditions: {conditions}." if conditions else ""
        return (
            f"DEGRADATION: {names} moved the wrong way by more than resampling explains, over "
            f"the last {self.recent_n} trades against a {self.baseline_n}-trade baseline.{where} "
            "This is a reason to look, not a mandate to change anything."
        )

    def as_dict(self) -> dict:
        return {
            "strategy_version": self.strategy_version,
            "total_trades": self.total_trades,
            "baseline_n": self.baseline_n,
            "recent_n": self.recent_n,
            "comparable": self.comparable,
            "degraded": [s.name for s in self.degraded],
            "shifts": [s.as_dict() for s in self.shifts],
            "groups": [g.as_dict() for g in self.groups],
            "verdict": self.verdict(),
        }


def _drawdown(returns: list[float]) -> float | None:
    """Deepest peak-to-trough run of the cumulative return, in points.

    Returned as a NEGATIVE number so it shares a direction with every other
    metric here: less negative is better, and a single sign convention
    across the table is what stops a reader misjudging which way is bad.
    """
    if not returns:
        return None
    cumulative = 0.0
    peak = 0.0
    worst = 0.0
    for r in returns:
        cumulative += r
        peak = max(peak, cumulative)
        worst = min(worst, cumulative - peak)
    return worst


def _values(rows: list[PostMortem], attribute: str) -> list[float]:
    return [v for v in (getattr(r, attribute, None) for r in rows) if v is not None]


def _shift(
    name: str, direction: str, baseline: list[float], recent: list[float], unit: str = "%"
) -> MetricShift:
    """Compare one metric across the two windows, with a resampled p-value.

    The p-value asks a one-sided question - "could the baseline look this
    much better than recent by chance?" - which is the question that
    matters. A two-sided test would also fire on improvement, and an alert
    that goes off when things get better gets muted.
    """
    shift = MetricShift(
        name=name, direction=direction,
        baseline=_mean(baseline), recent=_mean(recent),
        baseline_n=len(baseline), recent_n=len(recent), unit=unit,
    )
    if not shift.worse:
        return shift
    # bootstrap_p_value asks "does the challenger beat the champion?".
    # Degradation is the baseline beating recent, so the baseline goes in
    # as the challenger. For a lower-is-better metric both sides are
    # negated so that "bigger" still means "better".
    if direction == HIGHER_IS_BETTER:
        shift.p_value = bootstrap_p_value(recent, baseline)
    else:
        shift.p_value = bootstrap_p_value([-v for v in recent], [-v for v in baseline])
    return shift


def _regime_axes(label: str | None) -> list[tuple[str, str]]:
    """Split a full regime label into its named axes.

    Reported per axis rather than as a whole label because the combined
    label fragments the sample into a dozen groups of three trades, and a
    dozen groups of three trades is a random-number generator.
    """
    if not label:
        return []
    parts = label.split("/")
    names = ("trend", "volatility", "liquidity")
    return [(names[i], part) for i, part in enumerate(parts) if i < len(names) and part]


def build_degradation(
    db: Session,
    *,
    strategy_version: str | None = None,
    recent_trades: int = RECENT_TRADES,
) -> DegradationReport:
    """Compare the recent window against the earlier trades of one version."""
    version = strategy_version or current_label()
    report = DegradationReport(strategy_version=version)

    query = (
        db.query(models.Position)
        .filter(
            models.Position.status == models.PositionStatus.CLOSED.value,
            models.Position.closed_at.isnot(None),
        )
        .order_by(models.Position.closed_at.asc())
    )
    positions = [p for p in query.limit(MAX_TRADES).all()]

    # One version only. Pooling across a threshold change compares two
    # strategies and calls the difference decay.
    versioned = [p for p in positions if _version_of(db, p) == version]
    report.total_trades = len(versioned)

    if len(versioned) < MIN_BASELINE + MIN_RECENT:
        report.baseline_n = max(len(versioned) - recent_trades, 0)
        report.recent_n = min(len(versioned), recent_trades)
        return report

    rows = [build_postmortem(db, p) for p in versioned]
    regimes = {p.id: getattr(p, "market_regime", None) for p in versioned}

    split = len(rows) - recent_trades
    baseline_rows, recent_rows = rows[:split], rows[split:]
    report.baseline_n, report.recent_n = len(baseline_rows), len(recent_rows)

    base_returns = _values(baseline_rows, "return_pct")
    recent_returns = _values(recent_rows, "return_pct")

    report.shifts = [
        _shift("expectancy", HIGHER_IS_BETTER, base_returns, recent_returns),
        _shift(
            "win rate", HIGHER_IS_BETTER,
            [100.0 if r > 0 else 0.0 for r in base_returns],
            [100.0 if r > 0 else 0.0 for r in recent_returns],
        ),
        _shift("MFE (peak reached)", HIGHER_IS_BETTER,
               _values(baseline_rows, "max_gain_pct"), _values(recent_rows, "max_gain_pct")),
        _shift("MAE (heat taken)", HIGHER_IS_BETTER,
               _values(baseline_rows, "max_loss_pct"), _values(recent_rows, "max_loss_pct")),
        _shift("slippage", LOWER_IS_BETTER,
               _values(baseline_rows, "slippage_pct"), _values(recent_rows, "slippage_pct")),
    ]

    # Drawdown is a property of the SEQUENCE, so it has no per-trade sample
    # to resample. Reported as a shift with no p-value rather than given
    # one it cannot support.
    report.shifts.append(MetricShift(
        name="max drawdown", direction=HIGHER_IS_BETTER,
        baseline=_drawdown(base_returns), recent=_drawdown(recent_returns),
        baseline_n=len(base_returns), recent_n=len(recent_returns),
    ))

    report.groups = _group_shifts(baseline_rows, recent_rows, regimes)
    return report


def _group_shifts(
    baseline_rows: list[PostMortem],
    recent_rows: list[PostMortem],
    regimes: dict[int, str | None],
) -> list[GroupShift]:
    """Expectancy per market condition, before and after."""
    def collect(rows: list[PostMortem]) -> dict[tuple[str, str], list[float]]:
        out: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if row.return_pct is None:
                continue
            for axis, value in _regime_axes(regimes.get(row.position_id)):
                out.setdefault((axis, value), []).append(row.return_pct)
        return out

    before, after = collect(baseline_rows), collect(recent_rows)
    shifts = []
    for key in sorted(set(before) | set(after)):
        axis, value = key
        shifts.append(GroupShift(
            group=value, axis=axis,
            baseline=_mean(before.get(key, [])), recent=_mean(after.get(key, [])),
            baseline_n=len(before.get(key, [])), recent_n=len(after.get(key, [])),
        ))
    return shifts


def _version_of(db: Session, position: models.Position) -> str | None:
    """The strategy version a position was opened under.

    Read from the signal that produced it: the position table does not
    carry the label itself, and inferring it from the current settings
    would stamp every historical trade with today's configuration.
    """
    trade = (
        db.query(models.Trade)
        .filter(models.Trade.id == position.entry_trade_id)
        .first()
    ) if position.entry_trade_id else None
    if trade is None or trade.signal_id is None:
        return None
    signal = db.query(models.Signal).filter(models.Signal.id == trade.signal_id).first()
    return signal.strategy_version if signal else None
