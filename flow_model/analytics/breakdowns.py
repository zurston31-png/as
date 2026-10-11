"""The section 12 metrics, sliced by dimension -- with N on every cell.

ARCHITECTURE.md section 12:

    `analytics/breakdowns.py` -- market, year, month, day-of-week, session,
    regime, setup, Flow Score bucket, R target. Each cell reports N, and any
    cell with N below `min_sample_n` is marked `INSUFFICIENT` rather than
    shown as a win rate (small-sample win rates are the primary way people
    fool themselves).

Three design rules follow from that sentence, and they are what this module
is for.

**1. A cell is never a bare float.** `BreakdownCell` is a value *and* a count
*and* possibly a refusal. Below `AnalyticsConfig.min_sample_n` the cell's
metrics are `MetricValue`s carrying `Undefined.INSUFFICIENT_SAMPLE` and the
cell's `status` is `INSUFFICIENT`. The suppressed number is computed and then
dropped unstored -- there is no field holding the hidden value, so the
suppression cannot be undone by reading the object. A 3-trade cell showing
100% is worse than useless, so it does not exist here in any form.

Raw COUNTS survive suppression (`n`, `n_wins`, `n_losses`, `n_scratch`).
Hiding them would be hiding data; the refusal is about the rate, which
projects three wins onto a population, not about the three wins.

**2. Adding a dimension does not touch `metrics.py`, and adding a metric does
not touch this module.** A `Dimension` is a name plus a key function; the
cell's metrics come from `metrics.TRADE_METRICS` by name. The two sides meet
only through that registry.

**3. An empty cell needs no special case.** A dimension may declare its
`domain` -- the full key universe, e.g. every `Regime` -- and keys in the
domain with no trades become cells with `n=0` and `status=EMPTY`. The shape
of a breakdown over the current zero-trade system is therefore the same shape
as a populated one: a full grid of EMPTY cells, no exceptions raised and no
zeros that read as measurements.

## Split phases

Every function here takes `split_label` and inherits the structural refusal
from `metrics.assert_single_phase` (section 0.8). `breakdown_by_split_phase`
is the single exception in the package: it is the sanctioned way to look at
several phases, and it reports them as separate cells with NO combined row.

## Which timestamp a trade is bucketed by

`entry_ts`, the moment the risk was taken -- not the signal and not the exit,
so a trade belongs to the period in which the decision was made. Timestamps
are bucketed as stored, which in this project is UTC; a day-of-week or
session-hour reading for an instrument whose local day differs from the UTC
day is therefore a UTC reading. `Session` is not affected: it is bucketed
from `TradeRecord.session`, which the backtester assigns from the
instrument's own calendar.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass

from pydantic import Field, model_validator

from flow_model.analytics.metrics import (
    DEFAULT_CELL_METRICS,
    MetricPolicy,
    MetricValue,
    SampleStatus,
    TRADE_METRICS,
    Undefined,
    assert_single_phase,
    compute_trade_metrics,
    outcome_counts,
    sample_status,
)
from flow_model.core.contracts import TradeRecord
from flow_model.core.enums import ExitReason, Regime, Session, SetupType, SplitPhase
from flow_model.core.model import FrozenModel

#: Weekday labels, index 0 = Monday to match `datetime.weekday()`. Fixed in
#: English rather than taken from the locale, so a report is byte-identical on
#: every machine (section 0.5, determinism).
DAY_OF_WEEK_LABELS: tuple[str, ...] = (
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
)

#: Decimal places the R-target dimension rounds the planned R multiple to.
#: JUDGEMENT CALL: two places, enough to keep a 1.5R target distinct from a
#: 1.45R one while not splitting one target into cells over tick rounding.
R_TARGET_DECIMALS: int = 2


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dimension:
    """How to slice a trade set.

    A plain function from trade to key, plus presentation. Deliberately not a
    pydantic model: it holds callables and is never serialized -- what gets
    stored is the resulting `Breakdown`, which holds only data.
    """

    name: str
    description: str
    key_of: Callable[[TradeRecord], Hashable]
    label_of: Callable[[Hashable], str] = str
    domain: tuple[Hashable, ...] = ()
    """The full key universe, when it is known in advance (an enum, the seven
    weekdays, the Flow Score buckets). Keys here with no trades still get a
    cell, which is how an empty breakdown keeps its shape."""

    sort_key: Callable[[Hashable], object] | None = None
    """Ordering for keys NOT in `domain` (years, symbols, R targets). Domain
    keys keep their declared order. Defaults to the label."""

    def order_for(self, key: Hashable) -> object:
        if self.sort_key is not None:
            return self.sort_key(key)
        return self.label_of(key)


def flow_score_bucket_label(score: float, edges: Sequence[float]) -> str:
    """Label for the Flow Score bucket `score` falls in.

    Half-open intervals `[lo, hi)`, except the last, which is closed so a
    perfect 100 is not out of range. Scores outside the configured edges get
    an explicit out-of-range label rather than being folded into an end
    bucket, because a score outside `flow_score_buckets` means the buckets
    are mis-configured and that should be visible.
    """
    if len(edges) < 2:
        raise ValueError(
            f"flow_score_buckets needs at least two edges to form a bucket, got {tuple(edges)}"
        )
    if score < edges[0]:
        return f"<{edges[0]:g}"
    if score > edges[-1]:
        return f">{edges[-1]:g}"
    final = len(edges) - 2
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        if index == final:
            # The top bucket is closed, so a perfect 100 belongs to it rather
            # than to a label of its own.
            return f"[{lower:g},{upper:g}]"
        if lower <= score < upper:
            return f"[{lower:g},{upper:g})"
    raise AssertionError("unreachable: edges are ascending and score is in range")


def flow_score_bucket_domain(edges: Sequence[float]) -> tuple[str, ...]:
    """Every in-range bucket label, ascending. Must agree exactly with
    `flow_score_bucket_label`, or a bucket would appear twice in a table --
    once as a declared-but-empty cell and once as an observed key."""
    if len(edges) < 2:
        raise ValueError("flow_score_buckets needs at least two edges")
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip(edges, edges[1:])]
    labels[-1] = f"[{edges[-2]:g},{edges[-1]:g}]"
    return tuple(labels)


def default_dimensions(policy: MetricPolicy | None = None) -> tuple[Dimension, ...]:
    """The section 12 dimensions, plus two the rest of the architecture asks for.

    Section 12 names: market, year, month, day-of-week, session, regime,
    setup, Flow Score bucket, R target. Added here:

    * `exit_reason` -- the researcher's strategy audit needs cost drag by exit
      reason, and the exit mix is the thing that makes a single drag figure
      wrong (`metrics.cost_drag_by_exit_reason` has the uncensored costs).
    * `ambiguous_fill` -- section 9 tags a trade `ambiguous_fill=true` "so the
      sensitivity of results to this assumption is measurable". Measuring it
      means comparing the two cells.
    * `month_of_year` -- section 12's "month" is read here as the
      chronological year-month (the thing a reader plots next to a year); the
      seasonal 1-12 reading is kept as its own dimension rather than chosen
      between.

    A new dimension is one entry in this tuple and nothing else.
    """
    pol = policy if policy is not None else MetricPolicy.default()
    edges = pol.flow_score_buckets
    return (
        Dimension(
            name="market",
            description="instrument symbol",
            key_of=lambda t: t.symbol,
            label_of=str,
        ),
        Dimension(
            name="year",
            description="calendar year of entry (UTC)",
            key_of=lambda t: t.entry_ts.year,
            label_of=lambda k: f"{int(k):04d}",
            sort_key=lambda k: int(k),
        ),
        Dimension(
            name="month",
            description="calendar year-month of entry (UTC), chronological",
            key_of=lambda t: (t.entry_ts.year, t.entry_ts.month),
            label_of=lambda k: f"{k[0]:04d}-{k[1]:02d}",
            sort_key=lambda k: (k[0], k[1]),
        ),
        Dimension(
            name="month_of_year",
            description="month 1-12 of entry, for seasonality",
            key_of=lambda t: t.entry_ts.month,
            label_of=lambda k: f"{int(k):02d}",
            domain=tuple(range(1, 13)),
            sort_key=lambda k: int(k),
        ),
        Dimension(
            name="day_of_week",
            description="weekday of entry (UTC), Monday first",
            key_of=lambda t: t.entry_ts.weekday(),
            label_of=lambda k: DAY_OF_WEEK_LABELS[int(k)],
            domain=tuple(range(7)),
            sort_key=lambda k: int(k),
        ),
        Dimension(
            name="session",
            description="session bucket recorded on the trade",
            key_of=lambda t: t.session,
            label_of=lambda k: str(k),
            domain=tuple(Session),
        ),
        Dimension(
            name="regime",
            description="regime at entry",
            key_of=lambda t: t.regime,
            label_of=lambda k: str(k),
            domain=tuple(Regime),
        ),
        Dimension(
            name="setup",
            description="setup archetype",
            key_of=lambda t: t.setup,
            label_of=lambda k: str(k),
            domain=tuple(SetupType),
        ),
        Dimension(
            name="flow_score_bucket",
            description="Flow Score at entry, in configured buckets",
            key_of=lambda t: flow_score_bucket_label(t.flow_score, edges),
            label_of=str,
            domain=flow_score_bucket_domain(edges),
        ),
        Dimension(
            name="r_target",
            description="planned reward/risk, read from the record's own prices",
            key_of=lambda t: round(t.planned_r_multiple, R_TARGET_DECIMALS),
            label_of=lambda k: f"{float(k):g}R",
            sort_key=lambda k: float(k),
        ),
        Dimension(
            name="exit_reason",
            description="how the position closed",
            key_of=lambda t: t.exit_reason,
            label_of=lambda k: str(k),
            domain=tuple(ExitReason),
        ),
        Dimension(
            name="ambiguous_fill",
            description="section 9 same-bar ambiguity tag",
            key_of=lambda t: bool(t.ambiguous_fill),
            label_of=lambda k: "ambiguous_fill" if k else "clean_fill",
            domain=(False, True),
        ),
    )


SPLIT_PHASE_DIMENSION = Dimension(
    name="split_phase",
    description="validation phase the trade belongs to (the one cross-phase view)",
    key_of=lambda t: t.split_label,
    label_of=lambda k: "UNLABELLED" if k is None else str(k),
    domain=(*tuple(SplitPhase), None),
)


def dimension_by_name(
    name: str, *, policy: MetricPolicy | None = None
) -> Dimension:
    for dimension in default_dimensions(policy):
        if dimension.name == name:
            return dimension
    if name == SPLIT_PHASE_DIMENSION.name:
        return SPLIT_PHASE_DIMENSION
    raise KeyError(f"unknown breakdown dimension: {name!r}")


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

INSUFFICIENT_NOTE = (
    "N below min_sample_n: reported INSUFFICIENT rather than as a rate "
    "(section 12). The counts beside it are the data; the rate would be an "
    "extrapolation from it."
)


class BreakdownCell(FrozenModel):
    """One slice: a count, a status, and metrics that may refuse to be numbers."""

    dimension: str
    key: str
    n: int = Field(ge=0)
    n_wins: int = Field(ge=0)
    n_losses: int = Field(ge=0)
    n_scratch: int = Field(ge=0)
    status: SampleStatus
    metrics: dict[str, MetricValue]

    @property
    def is_reportable(self) -> bool:
        """True only when N justified computing the metrics."""
        return self.status is SampleStatus.ADEQUATE

    def metric(self, name: str) -> MetricValue:
        return self.metrics[name]

    @model_validator(mode="after")
    def _check(self) -> "BreakdownCell":
        if self.n_wins + self.n_losses + self.n_scratch != self.n:
            raise ValueError(
                f"cell {self.dimension}:{self.key} counts "
                f"{self.n_wins}+{self.n_losses}+{self.n_scratch} do not sum to n={self.n}"
            )
        if self.status is not SampleStatus.ADEQUATE:
            leaked = sorted(k for k, v in self.metrics.items() if v.is_defined)
            if leaked:
                raise ValueError(
                    f"cell {self.dimension}:{self.key} is {self.status} but still "
                    f"carries values for {leaked}; a suppressed cell must hold no "
                    "numbers at all"
                )
        return self


def _cell(
    dimension: Dimension,
    key: Hashable,
    trades: Sequence[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy,
    names: Sequence[str],
) -> BreakdownCell:
    wins, losses, scratches = outcome_counts(trades)
    status = sample_status(len(trades), policy.min_sample_n)
    computed = compute_trade_metrics(
        trades, split_label=split_label, policy=policy, names=names
    )
    if status is not SampleStatus.ADEQUATE:
        # The computed values are dropped here and nowhere stored. Computing
        # first is what keeps each suppressed metric's name and unit right.
        computed = {
            name: MetricValue.undefined_because(
                value.name,
                Undefined.NO_TRADES if not trades else Undefined.INSUFFICIENT_SAMPLE,
                n=len(trades),
                unit=value.unit,
                note="" if not trades else INSUFFICIENT_NOTE,
            )
            for name, value in computed.items()
        }
    return BreakdownCell(
        dimension=dimension.name,
        key=dimension.label_of(key),
        n=len(trades),
        n_wins=wins,
        n_losses=losses,
        n_scratch=scratches,
        status=status,
        metrics=computed,
    )


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------


class Breakdown(FrozenModel):
    """One dimension's cells, ordered, with the suppression rule recorded."""

    dimension: str
    description: str
    split_label: SplitPhase | None
    spans_phases: bool = False
    """True only for `breakdown_by_split_phase`, which never emits a combined
    row. Anywhere else, several phases in one breakdown is a raise."""

    min_sample_n: int = Field(ge=1)
    metric_names: tuple[str, ...]
    n_trades: int = Field(ge=0)
    cells: tuple[BreakdownCell, ...]

    @property
    def reportable_cells(self) -> tuple[BreakdownCell, ...]:
        return tuple(c for c in self.cells if c.is_reportable)

    @property
    def n_insufficient(self) -> int:
        return sum(1 for c in self.cells if c.status is SampleStatus.INSUFFICIENT)

    @property
    def n_empty(self) -> int:
        return sum(1 for c in self.cells if c.status is SampleStatus.EMPTY)

    def cell(self, key: str) -> BreakdownCell | None:
        for candidate in self.cells:
            if candidate.key == key:
                return candidate
        return None

    def series(self, metric: str) -> tuple[tuple[str, MetricValue], ...]:
        """(label, value) pairs for one metric, in cell order. Undefined cells
        stay in the series as undefined -- a gap in a plot is honest, a
        dropped cell is a filtered sample."""
        return tuple((c.key, c.metrics[metric]) for c in self.cells)

    def describe(self) -> tuple[str, ...]:
        header = (
            f"{self.dimension} ({self.description}) split={self.split_label} "
            f"n={self.n_trades} cells={len(self.cells)} "
            f"insufficient={self.n_insufficient} empty={self.n_empty} "
            f"min_sample_n={self.min_sample_n}"
        )
        rows = [
            f"  {c.key}: n={c.n} w/l/s={c.n_wins}/{c.n_losses}/{c.n_scratch} "
            f"[{c.status}]"
            for c in self.cells
        ]
        return (header, *rows)

    @model_validator(mode="after")
    def _check(self) -> "Breakdown":
        total = sum(c.n for c in self.cells)
        if total != self.n_trades:
            raise ValueError(
                f"breakdown {self.dimension}: cells hold {total} trades but "
                f"{self.n_trades} were supplied. A dimension may not drop a trade."
            )
        return self


def breakdown(
    trades: Iterable[TradeRecord],
    dimension: Dimension | str,
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    metrics: Sequence[str] = DEFAULT_CELL_METRICS,
    include_empty: bool = True,
) -> Breakdown:
    """Slice `trades` by `dimension` and compute each cell's metrics.

    Refuses a mixed-phase trade set (section 0.8). Every trade lands in
    exactly one cell -- the `Breakdown` validator checks the cell counts sum
    to the input count, so a key function that silently drops trades fails
    here rather than producing a flattering subset.

    `include_empty=False` trims declared-domain cells with no trades. It is a
    presentation choice and the default is to show them, because an empty
    regime is a fact about the sample.
    """
    pol = policy if policy is not None else MetricPolicy.default()
    dim = dimension if isinstance(dimension, Dimension) else dimension_by_name(
        dimension, policy=pol
    )
    materialized = tuple(trades)
    assert_single_phase(materialized, split_label)
    return _build(
        materialized,
        dim,
        split_label=split_label,
        policy=pol,
        metrics=metrics,
        include_empty=include_empty,
        spans_phases=False,
    )


def _build(
    trades: Sequence[TradeRecord],
    dim: Dimension,
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy,
    metrics: Sequence[str],
    include_empty: bool,
    spans_phases: bool,
    phase_of_key: Callable[[Hashable], SplitPhase | None] | None = None,
) -> Breakdown:
    """Group, then compute. `phase_of_key` exists for the one dimension whose
    key IS the split phase: each of its cells must be computed under its own
    label, or the phase guard inside every metric would reject the cell it
    was asked to describe."""
    unknown = [name for name in metrics if name not in TRADE_METRICS]
    if unknown:
        raise KeyError(f"unknown metric name(s): {unknown}")

    groups: dict[Hashable, list[TradeRecord]] = {key: [] for key in dim.domain}
    for trade in trades:
        groups.setdefault(dim.key_of(trade), []).append(trade)

    declared = [key for key in dim.domain if include_empty or groups[key]]
    extra = sorted(
        (key for key in groups if key not in dim.domain), key=dim.order_for
    )
    ordered = [*declared, *extra]

    cells = tuple(
        _cell(
            dim,
            key,
            groups[key],
            split_label=phase_of_key(key) if phase_of_key is not None else split_label,
            policy=policy,
            names=metrics,
        )
        for key in ordered
    )
    return Breakdown(
        dimension=dim.name,
        description=dim.description,
        split_label=split_label,
        spans_phases=spans_phases,
        min_sample_n=policy.min_sample_n,
        metric_names=tuple(metrics),
        n_trades=sum(c.n for c in cells),
        cells=cells,
    )


def breakdown_all(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    dimensions: Sequence[Dimension] | None = None,
    metrics: Sequence[str] = DEFAULT_CELL_METRICS,
    include_empty: bool = True,
) -> dict[str, Breakdown]:
    """Every default dimension, keyed by name, in declared order."""
    pol = policy if policy is not None else MetricPolicy.default()
    materialized = tuple(trades)
    assert_single_phase(materialized, split_label)
    dims = dimensions if dimensions is not None else default_dimensions(pol)
    return {
        dim.name: _build(
            materialized,
            dim,
            split_label=split_label,
            policy=pol,
            metrics=metrics,
            include_empty=include_empty,
            spans_phases=False,
        )
        for dim in dims
    }


def breakdown_by_split_phase(
    trades: Iterable[TradeRecord],
    *,
    policy: MetricPolicy | None = None,
    metrics: Sequence[str] = DEFAULT_CELL_METRICS,
    include_empty: bool = True,
) -> Breakdown:
    """The ONE function allowed to receive several split phases.

    It is how you look at TRAIN and TEST together honestly: one cell each,
    each cell's metrics computed from that phase alone, and no combined row
    anywhere in the result. There is no argument that merges them, because
    section 0.8 forbids the number that would produce.

    Trades with no `split_label` get an UNLABELLED cell rather than being
    folded into a labelled one.
    """
    pol = policy if policy is not None else MetricPolicy.default()
    materialized = tuple(trades)
    return _build(
        materialized,
        SPLIT_PHASE_DIMENSION,
        split_label=None,
        policy=pol,
        metrics=metrics,
        include_empty=include_empty,
        spans_phases=True,
        phase_of_key=lambda key: key,  # the key is the phase
    )
