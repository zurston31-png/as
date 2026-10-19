"""The consistency score: dispersion-penalized stability (ARCHITECTURE.md 10).

Section 10 names this module and fixes what it computes:

    `validation/consistency.py` -- the consistency score from the brief:
    dispersion-penalized stability of OOS win rate, profit factor, max
    drawdown, expectancy, across regimes and across years. Explicitly
    rewards "72% stable" over "90% in one window".

That last sentence is the user's strongest stated preference and the
pre-registered `H-CONSISTENCY-OVER-PEAK` is about exactly it, so the ranking is a
property of the FORMULA and not a note in a report: `preference_check` re-derives
it from this code and `rest_quality_boundary` states the exact condition under
which it holds. Like `walk_forward.py` this module computes no metric of its own
-- it consumes `MetricValue`s that `analytics.metrics` produced, so there is no
second definition of a win rate here and no second representation of "undefined"
(`analytics.metrics.Undefined` is the only one).

THE FORMULA
-----------
An AXIS is a set of cells: walk-forward windows, the six regimes, the years, or
the regime x year grid. For one axis and one of the four metrics section 10 names:

1. SUPPRESSION (section 12). A cell with `n < min_sample_n` is INSUFFICIENT and
   contributes nothing; `n = 0` is EMPTY and contributes nothing. Small-sample win
   rates are "the primary way people fool themselves", so the suppressed value is
   not stored anywhere (`CellMeasurement.build`). What remains are the
   CONTRIBUTING cells: adequate n AND a defined value.

2. QUALITY. Each contributing value is mapped to `q` in [0, 1] by a declared
   monotone map, so four metrics on four incompatible scales combine without one
   dominating through its unit:

       win rate       q = w                                   (w in [0, 1])
       profit factor  q = PF / (PF + PROFIT_FACTOR_BREAKEVEN)
       expectancy     q = (1 + E / (|E| + EXPECTANCY_HALF_SATURATION_R)) / 2
       max drawdown   q = 1 - min(|d|, MAX_DRAWDOWN_RUIN_FRACTION)

   Three of the four have no free parameter: a win rate already spans its own
   achievable range, breakeven for a profit factor IS 1.0 by the definition of the
   ratio, a 100% drawdown IS ruin. The fourth needs a scale and takes the metric's
   own unit, one R (section 0.2), rather than a fitted one. Both profitability maps
   send breakeven to 0.5. Every constant is declared below with its provenance;
   there are no others.

3. LEVEL, WORST CELL, DISPERSION and STABILITY over the k contributing cells:

       L = (1/k) sum_i q_i        W = min_i q_i
       D = L - W                  S = sqrt(L * W) = L * sqrt(1 - D / L)

   `S` is the level discounted by the square root of the share of it that survives
   in the worst cell. `D` and `S` are UNDEFINED for k < 2.

The score combines the per-axis `S` with the weights committed in
`config.consistency` (`ConsistencyConfig`) and invents none of its own:

    score =  w_win . S(WINDOW, win rate)     +  w_pf . S(WINDOW, profit factor)
          +  w_dd  . S(WINDOW, max drawdown) +  w_exp . S(WINDOW, expectancy)
          +  w_reg . sum_m (w_m / sum w_m) . S(REGIME, m)
          +  w_yr  . sum_m (w_m / sum w_m) . S(YEAR,   m)

The inner weighting of the regime and year axes is those same four metric weights
renormalized, so the relative importance of the four is stated once;
`ConsistencyReport.terms` reports the identical sum multiplied out flat. The
REGIME_YEAR grid -- "across regimes AND years ... jointly" -- is a REQUIRED field
of `ConsistencyReport`, reported in full but carrying no weight:
`ConsistencyConfig` commits six weights and a seventh would be a parameter this
module invented for itself.

WHY THE DISPERSION MEASURE IS THE WORST-CELL SHORTFALL
------------------------------------------------------
"One catastrophic window must not be averaged away" rules out every averaging
dispersion statistic, and the arithmetic says so exactly. Over k cells, one at
quality 0 and k-1 at quality v:

    sample standard deviation (ddof=1) = v / sqrt(k)
    mean absolute deviation            = 2 v (k - 1) / k^2
    D = L - W                          = v (k - 1) / k

The first two SHRINK as windows are added -- a wipeout among 25 windows has a
fifth of the standard deviation it has among one -- so a study could bury a
catastrophe by reporting more folds. `D` grows toward `v` instead: driven by the
extreme, it cannot be diluted. That is the required direction of outlier
sensitivity, and it is why the measure is an extremum and not a moment.

An extremum says nothing about the shape of the middle, so the formula keeps both
readings: `L` is the typical cell, `W` the worst, `S` their geometric mean. Hence
a UNIFORM axis scores exactly its own quality (L = W, so S = L for every k):
"72% in every window" scores 0.72, no penalty is invented, and no credit is given
for merely having more windows -- only for not having a bad one. And W = 0 forces
S = 0 whatever the other cells did; quality 0 is not a thin sample (that is
suppressed, not scored) but a window with adequate n and no winners at all, or a
100% drawdown, and averaging that away is the failure this module exists to
prevent. The geometric mean has no tunable weight either: the linear alternative,
(1 - lambda) L + lambda W, needs a lambda, and a lambda is a knob that can be
turned until the verdict is the desired one.

THE GUARANTEE
-------------
Write A for "v in every one of k windows" and B for "p in one window, b in the
other k-1", b < v <= p. Then S(A) = v exactly and
S(B) = sqrt(b (b + (p - b) / k)), so S(A) > S(B) exactly when

    (k - 1) b^2 + p b - k v^2 < 0
    <=>  b  <  beta(k) = (-p + sqrt(p^2 + 4 k (k - 1) v^2)) / (2 (k - 1))

At the brief's numbers (v = 0.72, p = 0.90; qualities read as win rates because
that map is the identity) beta(2) = 0.6632, beta(4) = 0.6948, beta(10) = 0.7106,
rising toward v as k grows. So 72% in every window beats 90% in one window
whenever the rest are worse than roughly 66-71%, and for any ordinary reading of
"badly" the margin is wide: at k = 4 with the rest at 30%, A scores 0.72 against
B's sqrt(0.45 * 0.30) = 0.3674. The formula does NOT claim 72% everywhere beats
90% once and 71.9% otherwise -- that would mean ignoring the level, which would
rank a flat 10% above everything -- and `rest_quality_boundary` reports the
boundary rather than hiding it.

THE UNDEFINED CASES
-------------------
n = 0 is the live case. At the committed configuration the system takes zero
trades -- 19542 synthetic NQ 5-minute bars (full year 2016), 283 reach the scorer,
max Flow Score 66.991221 against a 70.0 threshold, so section 14 structure and not
the score is the binding gate (PROVENANCE.md 2.4, commit fe6c31c). Every cell on
every axis is then EMPTY and the whole report builds and prints without a number:
undefined, `Undefined.NO_TRADES`, n = 0.

DISPERSION OVER A SINGLE WINDOW IS UNDEFINED, not 0.0. A lone window has no
spread, and 0.0 reads as "every window agreed" -- which would make one lucky
window the most consistent result this module can report, the exact inversion of
its purpose. `MIN_CELLS_FOR_DISPERSION` is `walk_forward`'s
`MIN_FOLDS_FOR_DISPERSION`, imported rather than restated; below it `level` and
`worst` are still measured (a mean over one cell is that cell) while `dispersion`
and `stability` refuse.

A partially covered score is never renormalized over the terms that happen to be
defined, because dropping the undefined ones and rescaling is how a bad period
disappears. `score` is defined ONLY at full coverage. Beside it,
`score_lower_bound` counts every undefined term as 0 -- sound precisely because
the quality scale is bounded below by 0 -- so it can only understate, and
`coverage` says how much of the weight was measured.

DETERMINISM (section 0.5): pure arithmetic over the values handed in. No clock
read, no randomness, no numpy global state; cell order is the caller's order and
every reported mapping is sorted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from itertools import product

from pydantic import Field, computed_field, model_validator

from flow_model.analytics.breakdowns import Breakdown, Dimension, breakdown, dimension_by_name
from flow_model.analytics.metrics import (
    MetricPolicy,
    MetricValue,
    SampleStatus,
    Undefined,
    sample_status,
)
from flow_model.config.schema import ConsistencyConfig
from flow_model.core.contracts import TradeRecord
from flow_model.core.enums import Regime, SplitPhase, StrEnum
from flow_model.core.model import FrozenModel
from flow_model.validation.walk_forward import MIN_FOLDS_FOR_DISPERSION, WalkForwardReport

# --- Declared constants: all of them, with provenance ----------------------

#: Dispersion needs two cells. IMPORTED from `walk_forward` rather than restated:
#: "one window has a mean and no spread" is the same truth about folds and about
#: regimes, and two copies could drift apart.
MIN_CELLS_FOR_DISPERSION: int = MIN_FOLDS_FOR_DISPERSION

#: The profit factor at which gross profit equals gross loss. NOT a judgement
#: call: 1.0 by the definition of the ratio, and the half-saturation point of
#: `PF / (PF + 1)`, so quality 0.5 means breakeven.
PROFIT_FACTOR_BREAKEVEN: float = 1.0

#: Expectancy alone has no bounded natural range, so its map needs a scale. It
#: takes the metric's OWN unit -- one R, fixed at signal time (section 0.2) -- so
#: +1R sits halfway between breakeven (0.5) and the unreachable ceiling and -1R
#: halfway to the floor. Deliberately NOT fitted to a measured expectancy: the
#: system has taken zero trades so there is nothing to fit, and a scale chosen
#: from results would make the score a function of the results it scores.
EXPECTANCY_HALF_SATURATION_R: float = 1.0

#: The drawdown depth at which quality is 0. NOT a judgement call: a 100%
#: drawdown is ruin and no deeper number exists.
MAX_DRAWDOWN_RUIN_FRACTION: float = 1.0

#: Float slack when checking that `PreferenceCheck`'s two axes have the shape
#: `beta(k)` was derived for. Not a tunable: it exists only so that qualities that
#: are equal in exact arithmetic are not split apart by representation error.
_SHAPE_TOLERANCE: float = 1e-12

#: Unit on every number here -- per-cell quality, level, worst, dispersion,
#: stability, the composite -- all on the same bounded [0, 1] scale.
QUALITY_UNIT: str = "quality"

#: The six regimes the score is taken across. `Regime.UNKNOWN` is excluded as
#: "insufficient warmup; never tradable" (`core.enums`): no trade can be taken in
#: it, so declaring it would put a permanently EMPTY cell on the axis.
CONSISTENCY_REGIMES: tuple[Regime, ...] = tuple(r for r in Regime if r is not Regime.UNKNOWN)

INSUFFICIENT_NOTE = (
    "N below min_sample_n: this cell contributes no win rate and no other rate to the "
    "consistency score (section 12). Its counts are the data; the rate would be an "
    "extrapolation from them."
)
NO_EQUITY_NOTE = (
    "a breakdown cell is a set of trades, not an equity series, so max drawdown is not "
    "computable from it; pass max_drawdown_by_key to supply it"
)
ONE_CELL_NOTE = (
    "; dispersion over a single cell is UNDEFINED, not 0.0 -- a spread of zero reads as "
    "'every window agreed' and would make one lucky window maximally consistent"
)
NO_RESCALE_NOTE = (
    ". Not renormalized over the defined terms: that would drop the periods that refused "
    "and rescale the rest"
)


class ConsistencyError(ValueError):
    """Base for every refusal in this module."""


class ConsistencyAxis(StrEnum):
    """What a set of cells is sliced by. REGIME_YEAR is the joint grid: reported
    in full, never folded into the composite."""

    WINDOW = "window"
    REGIME = "regime"
    YEAR = "year"
    REGIME_YEAR = "regime_year"


class StabilityMetric(StrEnum):
    """The four metrics section 10 names, valued by their canonical
    `MetricValue.name` so a statement cannot be assembled from a metric that calls
    itself something else."""

    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown_pct"
    EXPECTANCY = "expectancy_r"

    @property
    def lookup_keys(self) -> tuple[str, ...]:
        """Keys this metric may appear under; `TRADE_METRICS` aliases `expectancy`."""
        if self is StabilityMetric.EXPECTANCY:
            return ("expectancy_r", "expectancy")
        if self is StabilityMetric.MAX_DRAWDOWN:
            return ("max_drawdown_pct", "max_drawdown")
        return (self.value,)


#: Declared order, used for every tuple of statements here.
STABILITY_METRICS: tuple[StabilityMetric, ...] = (
    StabilityMetric.WIN_RATE,
    StabilityMetric.PROFIT_FACTOR,
    StabilityMetric.MAX_DRAWDOWN,
    StabilityMetric.EXPECTANCY,
)

#: Which committed weight scores which (axis, metric): the four metric weights on
#: the window axis, the regime and year weights on their own axis across all four
#: metrics with these same four renormalized inside. PROVENANCE:
#: `ConsistencyConfig`, `config/schema.py`. The window entries are in
#: `STABILITY_METRICS` order, which `ConsistencyReport` relies on and validates.
WINDOW_WEIGHT_FIELDS: tuple[tuple[str, StabilityMetric], ...] = (
    ("weight_oos_win_rate_stability", StabilityMetric.WIN_RATE),
    ("weight_profit_factor_stability", StabilityMetric.PROFIT_FACTOR),
    ("weight_drawdown_stability", StabilityMetric.MAX_DRAWDOWN),
    ("weight_expectancy_stability", StabilityMetric.EXPECTANCY),
)
AXIS_WEIGHT_FIELDS: tuple[tuple[str, ConsistencyAxis], ...] = (
    ("weight_regime_stability", ConsistencyAxis.REGIME),
    ("weight_year_stability", ConsistencyAxis.YEAR),
)


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def quality(metric: StabilityMetric, value: float) -> float:
    """Step 2 of THE FORMULA: the declared map onto [0, 1], monotone in "better"
    for all four, so a cell that improves can never lower the score."""
    if not math.isfinite(value):
        raise ConsistencyError(
            f"{metric.value}={value} is not finite; a non-finite measurement must reach "
            "this module as an undefined MetricValue, not as a float"
        )
    if metric is StabilityMetric.WIN_RATE:
        return _clip01(value)
    if metric is StabilityMetric.PROFIT_FACTOR:
        pf = max(0.0, value)
        return _clip01(pf / (pf + PROFIT_FACTOR_BREAKEVEN))
    if metric is StabilityMetric.EXPECTANCY:
        return _clip01(0.5 * (1.0 + value / (abs(value) + EXPECTANCY_HALF_SATURATION_R)))
    # Max drawdown. `abs` on purpose: `DrawdownProfile` reports a NEGATIVE fraction,
    # no reading of a drawdown makes a positive one a gain, so the depth is taken
    # sign-agnostically rather than trusted to a caller.
    return _clip01(1.0 - min(abs(value), MAX_DRAWDOWN_RUIN_FRACTION))


def rest_quality_boundary(stable: float, peak: float, n_cells: int) -> float:
    """`beta(k)` from THE GUARANTEE: the quality the other cells must fall below
    for the uniform axis to outscore the spiky one. Qualities in, quality out; for
    win rate the map is the identity, so it reads as a win rate."""
    if n_cells < MIN_CELLS_FOR_DISPERSION:
        raise ConsistencyError(
            f"the boundary needs at least {MIN_CELLS_FOR_DISPERSION} cells; dispersion "
            "over one cell is undefined"
        )
    k = float(n_cells)
    return (-peak + math.sqrt(peak**2 + 4.0 * k * (k - 1.0) * stable**2)) / (2.0 * (k - 1.0))


def _pick(
    metrics: Mapping[StabilityMetric | str, MetricValue], metric: StabilityMetric
) -> MetricValue:
    """One metric out of a mapping that may key it by enum or by alias."""
    for key in (metric, *metric.lookup_keys):
        found = metrics.get(key)  # type: ignore[arg-type]
        if found is not None:
            return found
    raise ConsistencyError(
        f"no value for {metric.value} in the mapping supplied (keys: "
        f"{sorted(str(k) for k in metrics)}). This module computes no metric of its own; "
        "supply it, or supply an explicitly undefined MetricValue"
    )


class CellMeasurement(FrozenModel):
    """One cell of one axis: a count, a status, four possibly-refused numbers.

    Suppression is structural, not promised. `build` derives the status from
    `n_trades` and replaces every metric of a non-ADEQUATE cell with an undefined
    one; the validator then refuses to hold a defined value in such a cell. No
    field carries the hidden number, so reading the object cannot undo it.
    """

    axis: ConsistencyAxis
    key: str
    n_trades: int = Field(ge=0)
    sample: SampleStatus
    min_sample_n: int = Field(ge=1)
    metrics: dict[str, MetricValue]

    @classmethod
    def build(
        cls, *, axis: ConsistencyAxis, key: str, n_trades: int, min_sample_n: int,
        metrics: Mapping[StabilityMetric | str, MetricValue],
    ) -> "CellMeasurement":
        status = sample_status(n_trades, min_sample_n)
        by_name: dict[str, MetricValue] = {}
        for metric in STABILITY_METRICS:
            value = _pick(metrics, metric)
            if status is not SampleStatus.ADEQUATE:
                value = MetricValue.undefined_because(
                    value.name,
                    Undefined.NO_TRADES if n_trades == 0 else Undefined.INSUFFICIENT_SAMPLE,
                    n=n_trades, unit=value.unit,
                    note="" if n_trades == 0 else INSUFFICIENT_NOTE,
                )
            by_name[metric.value] = value
        return cls(
            axis=axis, key=key, n_trades=n_trades, sample=status,
            min_sample_n=min_sample_n, metrics=by_name,
        )

    def metric(self, metric: StabilityMetric) -> MetricValue:
        return self.metrics[metric.value]

    @model_validator(mode="after")
    def _check(self) -> "CellMeasurement":
        where = f"cell {self.axis.value}:{self.key}"
        if set(self.metrics) != {m.value for m in STABILITY_METRICS}:
            raise ConsistencyError(
                f"{where} holds {sorted(self.metrics)}; section 10 names "
                f"{sorted(m.value for m in STABILITY_METRICS)}"
            )
        mismatched = sorted(
            f"{m.value}<-{self.metrics[m.value].name}"
            for m in STABILITY_METRICS
            if self.metrics[m.value].name not in m.lookup_keys
        )
        if mismatched:
            raise ConsistencyError(
                f"{where} files metrics under names that are not their own: {mismatched}. "
                "`StabilityMetric` is valued by the canonical `MetricValue.name`, so a "
                "statement may not be assembled from a metric that calls itself something "
                "else -- a Sharpe ratio filed as a win rate would be scored by the win-rate "
                "map and clipped to quality 1.0"
            )
        if self.sample is not sample_status(self.n_trades, self.min_sample_n):
            raise ConsistencyError(
                f"{where} is {self.sample.value} at n={self.n_trades} against "
                f"min_sample_n={self.min_sample_n}"
            )
        leaked = sorted(k for k, v in self.metrics.items() if v.is_defined)
        if self.sample is not SampleStatus.ADEQUATE and leaked:
            raise ConsistencyError(
                f"{where} is {self.sample.value} but still carries values for {leaked}; a "
                "suppressed cell must hold no numbers"
            )
        return self


class StabilityStatement(FrozenModel):
    """One metric across one axis. The per-cell vector IS the object.

    `level`, `worst`, `dispersion` and `stability` are computed projections of
    `cells`, following `walk_forward.CrossFoldStatement`: no constructor takes a
    stability, there is no setter, and `model_dump()` carries the cells with it.
    An aggregate holdable without the cells behind it is what section 10 forbids.
    """

    axis: ConsistencyAxis
    metric: StabilityMetric
    cells: tuple[CellMeasurement, ...]

    @classmethod
    def over(
        cls, cells: Sequence[CellMeasurement], metric: StabilityMetric, axis: ConsistencyAxis
    ) -> "StabilityStatement":
        return cls(axis=axis, metric=metric, cells=tuple(cells))

    @model_validator(mode="after")
    def _check(self) -> "StabilityStatement":
        where = f"{self.axis.value}/{self.metric.value}"
        keys = [cell.key for cell in self.cells]
        if len(set(keys)) != len(keys):
            raise ConsistencyError(f"{where}: duplicate cell key in {keys}")
        for cell in self.cells:
            if cell.axis is not self.axis:
                raise ConsistencyError(f"{where}: cell {cell.key} is on {cell.axis.value}")
        return self

    @property
    def per_cell(self) -> tuple[MetricValue, ...]:
        """This metric in each cell, in cell order, undefined included -- a gap is
        honest, a dropped cell is a filtered sample."""
        return tuple(cell.metric(self.metric) for cell in self.cells)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_trades(self) -> int:
        return sum(cell.n_trades for cell in self.cells)

    @property
    def qualities(self) -> tuple[float, ...]:
        """Step 2 over the contributing cells, in cell order."""
        return tuple(quality(self.metric, v.value) for v in self.per_cell if v.value is not None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_cells_contributing(self) -> int:
        """Adequate N and a defined value: the `k` of THE FORMULA."""
        return len(self.qualities)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def undefined_reasons(self) -> dict[str, int]:
        """How many cells refused, by reason. Sorted, so a dump is stable."""
        counts: dict[str, int] = {}
        for value in self.per_cell:
            if value.undefined is not None:
                counts[value.undefined.value] = counts.get(value.undefined.value, 0) + 1
        return dict(sorted(counts.items()))

    def _reason(self) -> Undefined:
        """Why nothing contributed: the suppression reasons (the common case at
        n=0), else the first genuine refusal from an adequate cell.
        `undefined_reasons` carries the whole histogram either way."""
        if self.n_trades == 0:
            return Undefined.NO_TRADES
        suppressed = (Undefined.NO_TRADES, Undefined.INSUFFICIENT_SAMPLE)
        reasons = [v.undefined for v in self.per_cell if v.undefined is not None]
        genuine = [r for r in reasons if r not in suppressed]
        if genuine:
            return genuine[0]
        if Undefined.INSUFFICIENT_SAMPLE in reasons:
            return Undefined.INSUFFICIENT_SAMPLE
        return Undefined.NO_TRADES

    def _note(self) -> str:
        head = (
            f"{self.n_cells_contributing} of {self.n_cells} cell(s) contributed, "
            f"{self.n_trades} trade(s) in total"
        )
        listed = ", ".join(f"{r} x{c}" for r, c in self.undefined_reasons.items())
        return f"{head}; undefined: {listed}" if listed else head

    def _stat(self, suffix: str, number: float | None) -> MetricValue:
        """One of L, W, D, S as a `MetricValue`. `n` counts CELLS, not trades, as
        on `CrossFoldStatement`'s aggregates; the trade count is in the note."""
        name = f"{self.axis.value}_{self.metric.value}_{suffix}"
        k = self.n_cells_contributing
        if number is not None:
            return MetricValue.measured(name, number, n=k, unit=QUALITY_UNIT, note=self._note())
        return MetricValue.undefined_because(
            name,
            self._reason() if k == 0 else Undefined.INSUFFICIENT_OBSERVATIONS,
            n=k, unit=QUALITY_UNIT, note=self._note() + (ONE_CELL_NOTE if k else ""),
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def level(self) -> MetricValue:
        """`L`: unweighted mean quality, as `CrossFoldStatement.mean` is unweighted
        -- trade-weighting lets one busy window speak for the study."""
        q = self.qualities
        return self._stat("level", sum(q) / len(q) if q else None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def worst(self) -> MetricValue:
        """`W`: the worst contributing cell."""
        q = self.qualities
        return self._stat("worst", min(q) if q else None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dispersion(self) -> MetricValue:
        """`D = L - W`, the maximum shortfall below the level. Undefined below two."""
        q = self.qualities
        enough = len(q) >= MIN_CELLS_FOR_DISPERSION
        return self._stat("dispersion", sum(q) / len(q) - min(q) if enough else None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def stability(self) -> MetricValue:
        """`S = sqrt(L * W)`, the score's input. Undefined below two cells."""
        q = self.qualities
        enough = len(q) >= MIN_CELLS_FOR_DISPERSION
        return self._stat("stability", math.sqrt(sum(q) / len(q) * min(q)) if enough else None)

    def describe(self) -> tuple[str, ...]:
        """Per-cell lines first, then the aggregate. Never the other way."""
        return (
            f"{self.axis.value}/{self.metric.value} over {self.n_cells} cell(s):",
            *(f"  {c.key}: {c.metric(self.metric)} n_trades={c.n_trades}" for c in self.cells),
            f"  {self.level}",
            f"  {self.worst}",
            f"  {self.dispersion}",
            f"  {self.stability}",
        )


class ScoreTerm(FrozenModel):
    """One weighted input to the score: a committed weight, an axis, a metric and
    the stability that axis measured for it."""

    label: str
    weight: float = Field(ge=0.0)
    axis: ConsistencyAxis
    metric: StabilityMetric
    value: MetricValue


def _combine(
    name: str, terms: Sequence[ScoreTerm]
) -> tuple[MetricValue, MetricValue, MetricValue]:
    """`(value, lower_bound, coverage)` for the weighted mean of `terms`.

    The value is defined ONLY at full coverage: dropping the undefined terms and
    rescaling the rest is how a bad period disappears from a composite. The bound
    counts an undefined term as 0, sound because quality is bounded below by 0, so
    it can only understate.
    """
    total = sum(t.weight for t in terms)
    if not terms or total <= 0.0:
        raise ConsistencyError(
            f"{name}: {len(terms)} term(s) of total weight {total}; a weighted combination "
            "with no weight is a construction error, not a measurement"
        )
    missing = [t for t in terms if not t.value.is_defined]
    defined = len(terms) - len(missing)
    note = f"{defined} of {len(terms)} term(s) defined"
    if missing:
        note += f"; undefined: {', '.join(sorted(t.label for t in missing))}"
    coverage = MetricValue.measured(
        f"{name}_coverage", sum(t.weight for t in terms if t.value.is_defined) / total,
        n=defined, unit="fraction", note=f"{len(terms)} term(s), weight {total:g}",
    )
    if missing:
        value = MetricValue.undefined_because(
            name, missing[0].value.undefined or Undefined.NO_TRADES,
            n=defined, unit=QUALITY_UNIT, note=note + NO_RESCALE_NOTE,
        )
    else:
        value = MetricValue.measured(
            name, sum(t.weight * t.value.require() for t in terms) / total,
            n=len(terms), unit=QUALITY_UNIT, note=note,
        )
    if not defined:
        bound = MetricValue.undefined_because(
            f"{name}_lower_bound", value.undefined or Undefined.NO_TRADES, n=0,
            unit=QUALITY_UNIT,
            note=f"{note}. Nothing contributed, so the bound is not 0.0 either: 0.0 would "
            "read as a measured worst case",
        )
    else:
        bound = MetricValue.measured(
            f"{name}_lower_bound",
            sum(t.weight * (t.value.value or 0.0) for t in terms) / total,
            n=defined, unit=QUALITY_UNIT,
            note=f"{note}. Undefined terms counted as 0, so this can only understate; "
            f"coverage {coverage.require():.4f}",
        )
    return value, bound, coverage


class ConsistencyReport(FrozenModel):
    """The whole score: every axis, every cell, nothing aggregate-only.

    `score` is a computed projection of `terms`, which are projections of the
    statements, which are projections of their cells. No constructor takes a
    score, and this object cannot be held without the cells behind it.
    """

    split_label: SplitPhase | None
    min_sample_n: int = Field(ge=1)
    weights: ConsistencyConfig
    window: tuple[StabilityStatement, ...]
    regime: tuple[StabilityStatement, ...]
    year: tuple[StabilityStatement, ...]
    joint: tuple[StabilityStatement, ...]
    """The regime x year grid: required, so it cannot be dropped, and unweighted,
    so no weight is invented for it (see THE FORMULA)."""

    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "ConsistencyReport":
        for axis, statements in self.by_axis.items():
            if tuple(s.metric for s in statements) != STABILITY_METRICS:
                raise ConsistencyError(
                    f"{axis.value} holds {[s.metric.value for s in statements]}; section 10 "
                    f"names {[m.value for m in STABILITY_METRICS]}, in that order"
                )
            if any(s.axis is not axis for s in statements):
                raise ConsistencyError(f"{axis.value} holds a statement from another axis")
            # An AXIS IS ONE SET OF CELLS (THE FORMULA). Four metrics measured over
            # four different sets would let the catastrophic window be dropped from
            # three of them while staying visible in the fourth, which is the exact
            # "do not hide a bad period" failure this module exists to prevent.
            reference = statements[0]
            for statement in statements[1:]:
                if statement.cells != reference.cells:
                    raise ConsistencyError(
                        f"{axis.value}/{statement.metric.value} is measured over "
                        f"{[c.key for c in statement.cells]} but "
                        f"{axis.value}/{reference.metric.value} over "
                        f"{[c.key for c in reference.cells]}; an axis is one set of cells, "
                        "and four metrics over four different sets lets a bad cell be "
                        "dropped from three of them"
                    )
            # The DECLARED threshold must be the APPLIED one. `min_sample_n` is a
            # separate argument from the cells, so a report could print
            # `min_sample_n=30` over a score computed from one-trade cells -- the
            # section 12 failure mode with a reassurance printed beside it.
            wrong = sorted(
                {cell.min_sample_n for cell in reference.cells} - {self.min_sample_n}
            )
            if wrong:
                raise ConsistencyError(
                    f"{axis.value} holds cells suppressed at min_sample_n={wrong} but the "
                    f"report declares min_sample_n={self.min_sample_n}; the threshold a "
                    "reader is shown must be the threshold that was applied"
                )
        return self

    @property
    def by_axis(self) -> dict[ConsistencyAxis, tuple[StabilityStatement, ...]]:
        return {
            ConsistencyAxis.WINDOW: self.window,
            ConsistencyAxis.REGIME: self.regime,
            ConsistencyAxis.YEAR: self.year,
            ConsistencyAxis.REGIME_YEAR: self.joint,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def terms(self) -> tuple[ScoreTerm, ...]:
        """THE FORMULA's composite, multiplied out into a flat weighted mean over
        twelve stability terms -- four window metrics at their own weights, and
        each of the regime and year axes' four at `w_axis * w_m / sum w_m`. That is
        the same number as the nested form the docstring writes, with one
        advantage: coverage and the lower bound are then accounted per metric
        rather than per axis, so a single refusing metric does not take its whole
        axis out of the bound. The statements are in `STABILITY_METRICS` order,
        which the validator pins.
        """
        inner = tuple(getattr(self.weights, name) for name, _ in WINDOW_WEIGHT_FIELDS)
        scale = sum(inner)
        if scale <= 0.0:
            raise ConsistencyError(
                "the four metric weights sum to 0, so the regime and year axes have no "
                "inner weighting; ConsistencyConfig commits them non-zero"
            )
        window = [
            ScoreTerm(
                label=name, weight=getattr(self.weights, name),
                axis=ConsistencyAxis.WINDOW, metric=s.metric, value=s.stability,
            )
            for (name, _), s in zip(WINDOW_WEIGHT_FIELDS, self.window)
        ]
        grouped = [
            ScoreTerm(
                label=f"{name}/{s.metric.value}",
                weight=getattr(self.weights, name) * weight / scale,
                axis=axis, metric=s.metric, value=s.stability,
            )
            for name, axis in AXIS_WEIGHT_FIELDS
            for s, weight in zip(self.by_axis[axis], inner)
        ]
        return (*window, *grouped)

    @property
    def _combined(self) -> tuple[MetricValue, MetricValue, MetricValue]:
        return _combine("consistency_score", self.terms)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> MetricValue:
        """The consistency score. Defined only at full coverage."""
        return self._combined[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score_lower_bound(self) -> MetricValue:
        """The score with every undefined term counted as 0: a true lower bound."""
        return self._combined[1]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coverage(self) -> MetricValue:
        """The share of the committed weight that was actually measured."""
        return self._combined[2]

    def describe(self) -> tuple[str, ...]:
        """Cells, then terms, then the score. Never the other way."""
        lines = [f"consistency split={self.split_label} min_sample_n={self.min_sample_n}"]
        for axis, statements in self.by_axis.items():
            tag = " (reported, unweighted)" if axis is ConsistencyAxis.REGIME_YEAR else ""
            lines.append(f"axis {axis.value}{tag}:")
            for statement in statements:
                lines.extend(f"  {line}" for line in statement.describe())
        lines.extend(f"{t.label} (weight {t.weight:g}) {t.value}" for t in self.terms)
        lines.extend([str(self.score), str(self.score_lower_bound), str(self.coverage)])
        return (*lines, *self.notes)


# --- Dimensions: the `breakdowns` machinery, reused rather than restated ----


def regime_dimension() -> Dimension:
    """`breakdowns`' regime dimension, redeclared over the six tradable regimes."""
    return replace(dimension_by_name("regime"), domain=CONSISTENCY_REGIMES)


def year_dimension(years: Sequence[int] = ()) -> Dimension:
    """`breakdowns`' year dimension with the study's years as its domain. Passing
    the research window's years keeps a year with no trades on the axis as an EMPTY
    cell instead of silently absent."""
    return replace(dimension_by_name("year"), domain=tuple(years))


def _regime_order(regime: Regime) -> tuple[int, str]:
    """A TOTAL order over regimes: the six declared ones first, anything else after.

    `CONSISTENCY_REGIMES.index(regime)` alone raised a bare
    `ValueError: tuple.index(x): x not in tuple` for a trade carrying
    `Regime.UNKNOWN` -- which `TradeRecord.regime` does not forbid -- and took the
    whole consistency report down from inside a sort key. The regime axis already
    reports such a trade as an extra cell (its dimension has no `sort_key`), so the
    joint grid crashing was also an inconsistency between the two. An undeclared
    regime is now sorted last and REPORTED, not hidden and not fatal.
    """
    if regime in CONSISTENCY_REGIMES:
        return (CONSISTENCY_REGIMES.index(regime), "")
    return (len(CONSISTENCY_REGIMES), str(regime))


def regime_year_dimension(years: Sequence[int] = ()) -> Dimension:
    """The joint grid: (year, regime) per cell, chronological then declared."""
    return Dimension(
        name="regime_year",
        description="calendar year of entry (UTC) x regime at entry",
        key_of=lambda t: (t.entry_ts.year, t.regime),
        label_of=lambda k: f"{int(k[0]):04d}|{k[1]}",
        domain=tuple(product(tuple(years), CONSISTENCY_REGIMES)),
        sort_key=lambda k: (int(k[0]), *_regime_order(k[1])),
    )


# --- Building cells, and the report ----------------------------------------


def _cell(
    axis: ConsistencyAxis, key: str, n_trades: int, min_sample_n: int,
    trade_metrics: Mapping[str, MetricValue], max_drawdown: MetricValue,
) -> CellMeasurement:
    metrics: dict[StabilityMetric | str, MetricValue] = {
        StabilityMetric.MAX_DRAWDOWN: max_drawdown
    }
    for metric in STABILITY_METRICS:
        if metric is not StabilityMetric.MAX_DRAWDOWN:
            metrics[metric] = _pick(trade_metrics, metric)
    return CellMeasurement.build(
        axis=axis, key=key, n_trades=n_trades, min_sample_n=min_sample_n, metrics=metrics
    )


def cells_from_breakdown(
    result: Breakdown, axis: ConsistencyAxis, *, min_sample_n: int | None = None,
    max_drawdown_by_key: Mapping[str, MetricValue] | None = None,
) -> tuple[CellMeasurement, ...]:
    """Cells from an `analytics.breakdowns.Breakdown`, in its cell order.

    Max drawdown needs an equity series and a breakdown cell has none, so it
    arrives undefined (`NO_EQUITY_SERIES`) unless supplied per cell key. That is
    honest and visible in `coverage`; deriving a drawdown from the trade sequence
    here would be a second definition of a number `analytics` owns.
    """
    drawdowns = max_drawdown_by_key or {}
    threshold = result.min_sample_n if min_sample_n is None else min_sample_n
    return tuple(
        _cell(
            axis, cell.key, cell.n, threshold, cell.metrics,
            drawdowns.get(
                cell.key,
                MetricValue.undefined_because(
                    StabilityMetric.MAX_DRAWDOWN.value, Undefined.NO_EQUITY_SERIES,
                    n=cell.n, unit="fraction", note=NO_EQUITY_NOTE,
                ),
            ),
        )
        for cell in result.cells
    )


def cells_from_walk_forward(
    report: WalkForwardReport, phase: SplitPhase, *, min_sample_n: int
) -> tuple[CellMeasurement, ...]:
    """Window cells from one phase of a walk-forward study, in fold order: the
    three trade metrics off each fold's `PerformanceSummary` registry dict, max
    drawdown off its `DrawdownProfile` where the equity curve's numbers live."""
    return tuple(
        _cell(
            ConsistencyAxis.WINDOW, w.split_id, w.n_trades, min_sample_n,
            w.performance.trade_metrics, w.performance.drawdown.max_drawdown_pct,
        )
        for w in (fold.window(phase) for fold in report.folds)
    )


def _statements(
    cells: Sequence[CellMeasurement], axis: ConsistencyAxis
) -> tuple[StabilityStatement, ...]:
    return tuple(StabilityStatement.over(cells, m, axis) for m in STABILITY_METRICS)


def consistency_report(
    *, split_label: SplitPhase | None, window_cells: Sequence[CellMeasurement],
    regime_cells: Sequence[CellMeasurement], year_cells: Sequence[CellMeasurement],
    min_sample_n: int, joint_cells: Sequence[CellMeasurement] = (),
    weights: ConsistencyConfig | None = None, notes: Sequence[str] = (),
) -> ConsistencyReport:
    """Assemble the score from cells already measured. Takes no trades and no
    config beyond the committed weights: everything reported is a projection of
    the `MetricValue`s handed in."""
    return ConsistencyReport(
        split_label=split_label,
        min_sample_n=min_sample_n,
        weights=weights if weights is not None else ConsistencyConfig(),
        window=_statements(window_cells, ConsistencyAxis.WINDOW),
        regime=_statements(regime_cells, ConsistencyAxis.REGIME),
        year=_statements(year_cells, ConsistencyAxis.YEAR),
        joint=_statements(joint_cells, ConsistencyAxis.REGIME_YEAR),
        notes=tuple(notes),
    )


def consistency_from_trades(
    trades: Iterable[TradeRecord], *, split_label: SplitPhase | None,
    window_cells: Sequence[CellMeasurement] = (), years: Sequence[int] = (),
    policy: MetricPolicy | None = None, weights: ConsistencyConfig | None = None,
    notes: Sequence[str] = (),
) -> ConsistencyReport:
    """The regime, year and joint axes from one phase's trades, through the
    `breakdowns` dimensions.

    The window axis is NOT derived here: a window is a walk-forward fold, which
    only `walk_forward.py` can cut, so it is passed in (`cells_from_walk_forward`).
    `breakdown` applies the section 0.8 single-phase refusal to the trade set. Max
    drawdown needs an equity series, so it is undefined on these three axes;
    `cells_from_breakdown` takes it when a caller has it.
    """
    pol = policy if policy is not None else MetricPolicy.default()
    held = tuple(trades)
    cells = {
        axis: cells_from_breakdown(
            breakdown(held, dimension, split_label=split_label, policy=pol),
            axis, min_sample_n=pol.min_sample_n,
        )
        for axis, dimension in (
            (ConsistencyAxis.REGIME, regime_dimension()),
            (ConsistencyAxis.YEAR, year_dimension(years)),
            (ConsistencyAxis.REGIME_YEAR, regime_year_dimension(years)),
        )
    }
    return consistency_report(
        split_label=split_label, window_cells=window_cells,
        regime_cells=cells[ConsistencyAxis.REGIME], year_cells=cells[ConsistencyAxis.YEAR],
        joint_cells=cells[ConsistencyAxis.REGIME_YEAR], min_sample_n=pol.min_sample_n,
        weights=weights, notes=notes,
    )


# --- The property this module exists for, checked from the code -------------


class PreferenceCheck(FrozenModel):
    """"72% stable" against "90% in one window", scored by the real formula.

    Both axes are built through `StabilityStatement`, so this cannot drift from
    the score: it IS the score, on two synthetic axes.
    """

    metric: StabilityMetric
    stable: StabilityStatement
    spiky: StabilityStatement

    @model_validator(mode="after")
    def _check_shape(self) -> "PreferenceCheck":
        """THE GUARANTEE derives `beta(k)` for ONE shape -- `v` in every cell against
        `p` once and `b` in the other `k - 1` -- so `boundary` is a boundary only when
        the axes have that shape. Checked structurally rather than assumed, because a
        DECREASING quality map inverts the two parameters: at `metric=MAX_DRAWDOWN`,
        `peak_value=0.90` is a 90% drawdown and maps to quality 0.10 while
        `rest_value=0.30` maps to 0.70, so the "spiky" axis becomes one BAD cell among
        `k - 1` good ones and `beta` reported 0.2270548 about a comparison that is not
        the one the axes express.
        """
        for statement, tag in ((self.stable, "stable"), (self.spiky, "spiky")):
            if statement.n_cells_contributing < MIN_CELLS_FOR_DISPERSION:
                raise ConsistencyError(
                    f"the {tag} axis has {statement.n_cells_contributing} contributing "
                    f"cell(s); this comparison is about dispersion and needs at least "
                    f"{MIN_CELLS_FOR_DISPERSION}"
                )
        flat = self.stable.qualities
        if max(flat) - min(flat) > _SHAPE_TOLERANCE:
            raise ConsistencyError(
                f"the stable axis is not uniform: qualities {flat}. beta(k) is derived "
                "against 'v in every cell'"
            )
        spiky = self.spiky.qualities
        peak = max(spiky)
        rest = tuple(q for q in spiky if q != peak)
        if len(rest) != len(spiky) - 1 or (rest and max(rest) - min(rest) > _SHAPE_TOLERANCE):
            raise ConsistencyError(
                f"the spiky axis is not 'one peak, the rest equal': qualities {spiky}. "
                "beta(k) is derived for that shape only, so `boundary` would be a number "
                "about a different comparison (a decreasing quality map such as max "
                "drawdown inverts peak and rest)"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def prefers_stable(self) -> bool:
        return self.stable.stability.require() > self.spiky.stability.require()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def margin(self) -> float:
        return self.stable.stability.require() - self.spiky.stability.require()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def boundary(self) -> float:
        """`beta(k)`, read off the two axes themselves -- the uniform axis's level
        is its common quality, the spiky axis's peak is its best cell -- so it can
        disagree with them only by arithmetic, which the tests check both ways."""
        return rest_quality_boundary(
            self.stable.level.require(), max(self.spiky.qualities), self.spiky.n_cells
        )


def preference_check(
    *, stable_value: float = 0.72, peak_value: float = 0.90, rest_value: float = 0.30,
    n_cells: int = 4, metric: StabilityMetric = StabilityMetric.WIN_RATE,
    min_sample_n: int = 1,
) -> PreferenceCheck:
    """Score the two axes of section 10's sentence and report which wins.

    Synthetic cells, each given `min_sample_n` trades so none is suppressed: the
    comparison is about dispersion, and a suppressed cell would test the
    suppression rule instead.
    """
    unit = "R" if metric is StabilityMetric.EXPECTANCY else "fraction"

    def axis(values: Sequence[float], tag: str) -> StabilityStatement:
        cells = tuple(
            CellMeasurement.build(
                axis=ConsistencyAxis.WINDOW, key=f"{tag}_{index:03d}", n_trades=min_sample_n,
                min_sample_n=min_sample_n,
                metrics={
                    other: MetricValue.measured(
                        other.value, value if other is metric else 0.0,
                        n=min_sample_n, unit=unit if other is metric else "",
                    )
                    for other in STABILITY_METRICS
                },
            )
            for index, value in enumerate(values)
        )
        return StabilityStatement.over(cells, metric, ConsistencyAxis.WINDOW)

    return PreferenceCheck(
        metric=metric,
        stable=axis((stable_value,) * n_cells, "stable"),
        spiky=axis((peak_value, *((rest_value,) * (n_cells - 1))), "spiky"),
    )
