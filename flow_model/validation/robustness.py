"""Parameter grids and degradation runs (ARCHITECTURE.md 10).

WHY THERE IS NO "BEST CELL"
---------------------------
A sweep that names its best cell is a curve-fitting machine: run it twice and
the second answer is fitted to the first answer's noise. The brief says "do
not optimize parameters repeatedly against the same test set", so this module
reports the WHOLE GRID and has no `best`, `optimal`, `argmax`, `recommended`
or `select` anything. The refusal is structural, not a promise:

1. No method returns a winning cell. Nobody can call what does not exist.
2. `CellResult` and `DegradationResult` refuse to be ordered -- `<`, `<=`,
   `>`, `>=` raise -- so `max(report.cells)` and `sorted(...)` fail by name
   instead of quietly ranking. Picking a winner means writing the key
   function in your own module, where the choice shows up in a diff.
3. A sweep is refused outright on TEST and SEALED_OOS (`SWEEPABLE_PHASES`),
   and the seal is checked again through `DataAccessGuard` when one is given.

Every cell carries its trade count first. The live finding is that the system
takes no trades at the committed configuration (highest Flow Score over a
full year 66.99 against a lowest setup threshold of 70.0), so a 60..85 score
grid takes trades in its low cells and none in its high cells. Those low-cell
trades are not a discovery that the system works at a lower threshold; they
are the output of lowering a gate until something passed. A three-trade cell
is reported as a three-trade cell. Every cell is logged with a reason, so
"how often has this config family touched a phase" is answered from the
experiment-log table (`phase_touches`, `overfitting_summary`), which this
module does not reimplement.

THE FRAGILITY VERDICT, AS A FORMULA
-----------------------------------
Section 10 asks "how much performance decays per unit of parameter change".
With P the metric named by `FRAGILITY_METRIC` (expectancy in R):

    baseline cell B  = the cell whose coordinate on every axis equals the
                       committed config's value for that axis
    neighbour N      = a cell one grid step from B along exactly one axis
    relative_drop(N) = (P(B) - P(N)) / |P(B)|
    step(N)          = |v(N) - v(B)| / |v(B)|
    elasticity(N)    = relative_drop(N) / step(N)
    worst_relative_drop = max over neighbours N of relative_drop(N)
                       and the same maximum per AXIS, because the grid's
                       worst can belong to a parameter the reader is not
                       asking about -- a pre-registered hypothesis names one
                       driver and is scored on that driver's own number

    FRAGILE   if worst_relative_drop > fragility_max_relative_drop, or a
              neighbour's profit factor falls below
              fragility_profit_factor_floor while B's does not
    ROBUST    if neither holds and a neighbour is defined
    UNDEFINED otherwise: no trades, no cell at the committed coordinate, no
              defined neighbour, or P(B) == 0 so the drop has no denominator

Both cutoffs come from `config.robustness`; this module chooses neither. The
one-step relative form is the shape of the pre-registered
`H-SR-CBAND-ROBUST`, so the measurement matches a refutation condition fixed
in advance rather than one invented after the fact. The elasticity is
reported beside it to keep "per unit of parameter change" literal: a 10% drop
across a 7% parameter move is more fragile than the same drop across a 40%
move, and a relative drop alone cannot tell them apart. Every drop behind the
VERDICT is measured FROM the baseline, so a step running into B uses B's
denominator, and `max_abs_elasticity` includes those baseline-measured
neighbour ratios: a `DecayStep` divides by its own lower cell, which for a
neighbour below B understates the decay per unit of parameter change, and the
reported maximum must not be the smaller of the two. `decay_profile` carries
every adjacent pair on every axis, each measured in its own direction, so the
grid speaks even when the verdict is UNDEFINED -- which at the committed
configuration it is, because n=0 makes every P undefined and no verdict is
invented from that.

DEGRADATION RUNS
----------------
Six kinds (section 10): added slippage, wider spreads, N-bar delayed entry,
random execution delay, random missed trades, win-rate haircut.
`default_degradations` runs each at every severity `config.robustness`
commits. Five are re-runs through config (`RERUN`); the win-rate haircut is
arithmetic over the realized R vector (`POST_HOC`), because `TradeRecord`
validates its `gross_pnl` against its price path, so "flip this winner to a
loser" could only be said by inventing an exit price that never traded. A
stress test is not a licence to fabricate a fill.

Direction, honestly: `DegradationSpec` cannot be constructed in the improving
direction at all (`NEUTRAL_SEVERITY` pins the severity that reproduces the
committed config; below it raises) -- that is the structural half. The
POST_HOC haircut is monotone by arithmetic, so a defined improvement there is
a defect here and RAISES. A RERUN degradation can genuinely come out ahead,
because it changes which trades exist: a randomly missed trade can be a
missed loser, and `spread_ticks_override` outranks a real quote, so a pinned
wider spread can still be tighter than the quote it replaced.
A delayed or jittered fill is priced off a LATER BAR, which can open more
favourably than the one it replaced -- so that run can come out ahead with its
trade set untouched. `DirectionCheck.worse_or_equal` records that and the
report says what it means; asserting a monotonicity that does not hold would
either crash honest runs or hide them. Every run that can flatter or can be
vacuous names its own mechanism in `notes`, because "read the run's own notes"
is only an answer if the note is there.

The delayed-entry runs push `max_entry_wait_bars` past 1, the path where bars
between an early fill and the end of the wait horizon were once uninspected
-- worth 2.06R of fabricated performance on one trade (fixed in
`BacktestLoop._process_pending`, f44ddb9). `wait_horizon_violations` re-checks
that invariant on every trade they produce rather than assuming the fix holds.

WHICH IS NOT THE SAME AS REACHING THAT PATH, and the difference is on the
result rather than only here. `PendingOrder.wait_bars` is 1 for a MARKET
order whatever `max_entry_wait_bars` says, because section 9 scopes the wait
horizon to resting entries -- re-offering a market order on later bars would
let a backtest chase an entry it missed. The committed
`execution.entry_order_type` is `market`, so at the committed configuration a
delayed-entry run shifts the fill bar and still builds a SINGLE-bar window:
the multi-bar horizon is not exercised and an empty `wait_horizon_findings`
says nothing about the fix. `_degradation_notes` states that on every such
run, and the path is reached only with a resting `entry_order_type`.

Randomness: `random_latency_ms` and `random_miss_probability` are drawn inside
`ExecutionModel` from named streams seeded by `config.backtest.seed`, so a
stochastic degradation carries a `seed_name` and derives that seed with
`core.determinism.derive_seed`; the haircut draws from `core.determinism.rng`.
The global NumPy random state is never touched.

Undefined values are `analytics.metrics.MetricValue` / `Undefined`; there is
no second representation of "no number". Counts and sums stay measured at n=0
and rates and ratios do not, which is where `analytics.metrics` draws the
line too. Two reasons are reused for ratios this module forms rather than
measures -- `ZERO_DISPERSION` when the baseline P is exactly zero (a relative
drop with no denominator) and `INSUFFICIENT_OBSERVATIONS` when fewer than two
comparable cells exist -- and each such value carries a `note` saying so.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time, timezone
from typing import Any

from pydantic import Field, model_validator

from flow_model.analytics.metrics import (
    Comparison,
    MetricPolicy,
    MetricValue,
    TargetComparison,
    TRADE_METRICS,
    Undefined,
    compare_to_target,
    compute_trade_metrics,
)
from flow_model.backtest.loop import BacktestLoop, RunResult, StoreMarketSource
from flow_model.config.loader import apply_override
from flow_model.core.contracts import TradeRecord
from flow_model.core.determinism import derive_seed, rng
from flow_model.core.enums import ExitReason, SplitPhase, StrEnum
from flow_model.core.model import FrozenModel
from flow_model.data.series import to_ns
from flow_model.utils.logging import get_logger
from flow_model.validation.experiment_log import ExperimentLog
from flow_model.validation.hypotheses import Direction, by_id
from flow_model.validation.splits import DataAccessGuard, Split, SplitRegistry

__all__ = [
    "AxisMode", "CellResult", "DecayStep", "DegradationKind", "DegradationMode",
    "DegradationResult", "DegradationSpec", "DirectionCheck", "FRAGILITY_METRIC",
    "FragilityReport", "FragilityVerdict", "GRID_CELL_METRICS", "GRID_READING_NOTE",
    "GridPoint", "HAIRCUT_METRICS", "HYPOTHESIS_DRIVER_PATHS", "HypothesisReport",
    "NEUTRAL_SEVERITY", "ParameterAxis", "RobustnessError", "RobustnessReport",
    "SWEEPABLE_PHASES", "StoreRunner", "apply_overrides", "axis_for_driver",
    "default_axes", "default_degradations", "fold_profit_concentration",
    "fragility_report", "hypothesis_reports", "read_path", "risk_pct_axis",
    "robustness_report", "run_degradations", "run_grid", "score_threshold_axis",
    "wait_horizon_violations", "win_rate_haircut_metrics",
]

logger = get_logger(__name__)


class RobustnessError(RuntimeError):
    """A sweep was asked for something it refuses to do."""


#: Phases a sweep may run on. TEST and SEALED_OOS are absent deliberately:
#: sweeping a grid against an evaluation window is how a test set becomes a
#: training set.
SWEEPABLE_PHASES: tuple[SplitPhase, ...] = (SplitPhase.TRAIN, SplitPhase.VALIDATION)

#: Reported per cell, trade count first. All are keys of `TRADE_METRICS`;
#: nothing here computes a metric itself.
GRID_CELL_METRICS: tuple[str, ...] = (
    "trade_count", "win_rate", "expectancy", "mean_r", "profit_factor",
    "total_pnl", "max_consecutive_losses", "mean_cost_drag_r",
)

#: The metric the fragility formula is stated over: expectancy in R, the
#: quantity `H-SR-CBAND-ROBUST` is pre-registered against.
FRAGILITY_METRIC = "expectancy"

#: Compared between a degradation and its baseline. Lower is worse for each.
DEGRADATION_METRICS: tuple[str, ...] = (
    "trade_count", "win_rate", "expectancy", "profit_factor", "total_pnl",
)

#: Float slack for "worse or equal": last-bit noise in a sum, not a parameter.
COMPARISON_TOLERANCE = 1e-12

#: Hypothesis drivers mapped to the config path that moves them. A mapping,
#: not a grid: step sizes stay the caller's choice, so no sweep constant is
#: invented here for a driver `RobustnessConfig` declares none for.
HYPOTHESIS_DRIVER_PATHS: dict[str, str] = {"c_band": "levels.zone_band_atr"}


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------


def read_path(config: Any, path: str) -> Any:
    """Read a dotted config path. `setups.SCALP_1R.min_flow_score` works."""
    node: Any = config
    for part in path.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                raise RobustnessError(
                    f"config path {path!r}: no key {part!r} "
                    f"(have {sorted(str(k) for k in node)})"
                )
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            raise RobustnessError(
                f"config path {path!r}: {type(node).__name__} has no field {part!r}"
            )
    return node


def apply_overrides(config: Any, overrides: Mapping[str, Any]) -> Any:
    """A re-validated copy of `config` with dotted paths replaced.

    Goes through `to_init_dict(mode="json")` and the loader's own
    `apply_override`, so the schema re-validates the result (an out-of-range
    risk percentage raises here rather than reaching a run) and `config_hash`
    is recomputed. `config/defaults.yaml` is neither read nor written.
    """
    if not overrides:
        return config
    raw = config.to_init_dict(mode="json")
    for path in sorted(overrides):
        parts = [p for p in path.split(".") if p]
        if not parts:
            raise RobustnessError(f"empty override path in {dict(overrides)!r}")
        apply_override(raw, parts, overrides[path])
    try:
        return type(config)(**raw)
    except Exception as exc:
        raise RobustnessError(
            f"override {dict(overrides)!r} produced an invalid config: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


class AxisMode(StrEnum):
    """How one axis value reaches the config paths it governs."""

    ABSOLUTE = "absolute"
    """Every path is set to the axis value."""

    PRESERVE_OFFSETS = "preserve_offsets"
    """Paths keep their committed spacing and move together.

    JUDGEMENT CALL, stated because it changes what is measured. The committed
    setup ladder is 70/75/80: setting all three to one value collapses three
    setups into one and varies two things at once, while shifting the ladder
    so its lowest rung sits on the axis value keeps its shape and varies one.
    This mode does the latter, anchored on the committed minimum.
    """


class ParameterAxis(FrozenModel):
    """One swept quantity: the paths it moves, and the values it takes."""

    name: str
    paths: tuple[str, ...]
    values: tuple[float, ...]
    mode: AxisMode = AxisMode.ABSOLUTE
    unit: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ParameterAxis":
        if not self.paths:
            raise ValueError(f"axis {self.name!r} governs no config path")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError(f"axis {self.name!r} repeats a config path")
        if len(self.values) < 2:
            raise ValueError(
                f"axis {self.name!r} has {len(self.values)} value(s): a decay per unit "
                "of parameter change needs two points to be a measurement"
            )
        if list(self.values) != sorted(self.values) or len(set(self.values)) != len(
            self.values
        ):
            raise ValueError(f"axis {self.name!r} values must ascend without repeats")
        return self

    def overrides(self, config: Any, value: float) -> dict[str, float]:
        """The config overrides this axis applies for `value`."""
        if self.mode is AxisMode.ABSOLUTE:
            return {path: float(value) for path in self.paths}
        current = {path: float(read_path(config, path)) for path in self.paths}
        anchor = min(current.values())
        return {path: float(value) + (base - anchor) for path, base in current.items()}

    def committed_value(self, config: Any) -> float:
        """The axis coordinate `config` itself sits at."""
        current = [float(read_path(config, path)) for path in self.paths]
        return min(current) if self.mode is AxisMode.PRESERVE_OFFSETS else current[0]


def score_threshold_axis(config: Any) -> ParameterAxis:
    """Flow Score setup gate, over `config.robustness.flow_score_thresholds`."""
    return ParameterAxis(
        name="flow_score_threshold",
        paths=tuple(f"setups.{s.value}.min_flow_score" for s in sorted(config.setups)),
        values=tuple(float(v) for v in config.robustness.flow_score_thresholds),
        mode=AxisMode.PRESERVE_OFFSETS,
        unit="points",
        note="the committed setup ladder is shifted, its spacing unchanged",
    )


def risk_pct_axis(config: Any) -> ParameterAxis:
    """Risk per trade, over `config.robustness.risk_pcts`."""
    return ParameterAxis(
        name="risk_per_trade_pct",
        paths=("risk.risk_per_trade_pct",),
        values=tuple(float(v) for v in config.robustness.risk_pcts),
        unit="fraction of equity",
    )


def default_axes(config: Any) -> tuple[ParameterAxis, ...]:
    """Section 10's two axes, both read from the committed config."""
    return (score_threshold_axis(config), risk_pct_axis(config))


def axis_for_driver(driver: str, values: Sequence[float]) -> ParameterAxis:
    """An axis for a pre-registered hypothesis driver, over caller values.

    The values are the caller's because `RobustnessConfig` commits no grid for
    these drivers, and a step size invented here would be a research choice
    made in the wrong file.
    """
    if driver not in HYPOTHESIS_DRIVER_PATHS:
        raise RobustnessError(
            f"no config path known for hypothesis driver {driver!r}; "
            f"known: {sorted(HYPOTHESIS_DRIVER_PATHS)}"
        )
    path = HYPOTHESIS_DRIVER_PATHS[driver]
    return ParameterAxis(
        name=driver,
        paths=(path,),
        values=tuple(float(v) for v in values),
        note=f"hypothesis driver {driver!r} -> {path}",
    )


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------


class _Unorderable(FrozenModel):
    """A result row that refuses to be ranked.

    `max(cells)`, `min(cells)` and `sorted(cells)` raise, because an
    accidental `max()` downstream picks a best cell just as effectively as a
    method would.
    """

    def _refuse(self) -> "RobustnessError":
        return RobustnessError(
            f"{type(self).__name__} is deliberately not orderable, so max(), min() and "
            "sorted() cannot rank sweep results: a sweep that names its best cell is a "
            "curve fit. Report the whole grid, or write the key function where the "
            "choice is visible"
        )

    def __lt__(self, other: object) -> bool:
        raise self._refuse()

    def __le__(self, other: object) -> bool:
        raise self._refuse()

    def __gt__(self, other: object) -> bool:
        raise self._refuse()

    def __ge__(self, other: object) -> bool:
        raise self._refuse()


class GridPoint(FrozenModel):
    """One cell's coordinates: axis name -> value, in axis order."""

    coordinates: tuple[tuple[str, float], ...]

    @model_validator(mode="after")
    def _check(self) -> "GridPoint":
        names = [name for name, _ in self.coordinates]
        if len(set(names)) != len(names):
            raise ValueError(f"grid point repeats an axis: {names}")
        return self

    def as_mapping(self) -> dict[str, float]:
        return dict(self.coordinates)

    def label(self) -> str:
        return " ".join(f"{name}={value:g}" for name, value in self.coordinates)


class CellResult(_Unorderable):
    """One grid cell: what was changed, what ran, and what was measured."""

    point: GridPoint
    split_name: str
    split_phase: SplitPhase
    config_hash: str
    data_hash: str = ""
    experiment_id: str = ""
    params_changed: dict[str, float] = Field(default_factory=dict)
    n_trades: int = Field(ge=0)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def metric(self, name: str) -> MetricValue:
        """A measured metric, or an explicit undefined carrying its reason."""
        found = self.metrics.get(name)
        if found is not None:
            return found
        return MetricValue.undefined_because(
            name,
            Undefined.NO_TRADES if self.n_trades == 0 else Undefined.INSUFFICIENT_SAMPLE,
            n=self.n_trades,
            note=f"{name} was not computed for this cell",
        )

    def describe(self) -> str:
        body = " ".join(
            str(self.metrics[name]) for name in GRID_CELL_METRICS if name in self.metrics
        )
        return f"{self.split_name} [{self.point.label()}] n={self.n_trades} {body}"


# ---------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------


class DegradationKind(StrEnum):
    """Section 10's six degradation runs."""

    ADDED_SLIPPAGE = "added_slippage"
    WIDER_SPREAD = "wider_spread"
    DELAYED_ENTRY = "delayed_entry"
    RANDOM_EXECUTION_DELAY = "random_execution_delay"
    RANDOM_MISSED_TRADES = "random_missed_trades"
    WIN_RATE_HAIRCUT = "win_rate_haircut"


class DegradationMode(StrEnum):
    RERUN = "rerun"
    """The backtest is re-run with a worse configuration."""

    POST_HOC = "post_hoc"
    """Arithmetic over the realized R vector; no record is altered."""


#: The severity at which each kind reproduces the committed config exactly.
#: Below it is an improvement, which is refused rather than measured.
NEUTRAL_SEVERITY: dict[DegradationKind, float] = {
    DegradationKind.ADDED_SLIPPAGE: 1.0,
    DegradationKind.WIDER_SPREAD: 1.0,
    DegradationKind.DELAYED_ENTRY: 0.0,
    DegradationKind.RANDOM_EXECUTION_DELAY: 0.0,
    DegradationKind.RANDOM_MISSED_TRADES: 0.0,
    DegradationKind.WIN_RATE_HAIRCUT: 0.0,
}

_STOCHASTIC = (
    DegradationKind.RANDOM_EXECUTION_DELAY,
    DegradationKind.RANDOM_MISSED_TRADES,
    DegradationKind.WIN_RATE_HAIRCUT,
)
_PROBABILITIES = (DegradationKind.RANDOM_MISSED_TRADES, DegradationKind.WIN_RATE_HAIRCUT)


class DegradationSpec(FrozenModel):
    """One degradation, in the worse-or-equal direction by construction."""

    kind: DegradationKind
    severity: float
    seed_name: str = ""

    @model_validator(mode="after")
    def _check(self) -> "DegradationSpec":
        neutral = NEUTRAL_SEVERITY[self.kind]
        if self.severity < neutral:
            raise ValueError(
                f"{self.kind.value}: severity {self.severity:g} is below the neutral "
                f"{neutral:g}, which would make the result BETTER than the committed "
                "configuration. A degradation that can improve a result is "
                "misconfigured; there is no negative slippage here"
            )
        if self.kind in _PROBABILITIES and self.severity > 1.0:
            raise ValueError(
                f"{self.kind.value}: severity {self.severity:g} is a probability and "
                "must lie in [0, 1]"
            )
        if self.kind is DegradationKind.DELAYED_ENTRY and self.severity % 1.0:
            raise ValueError(
                f"delayed_entry severity {self.severity:g} is a bar count and must be "
                "a whole number of bars"
            )
        if self.kind in _STOCHASTIC and not self.seed_name.strip():
            raise ValueError(
                f"{self.kind.value} draws random numbers, so it must name its seed "
                "stream: an unnamed stream cannot be reproduced, and the global NumPy "
                "random state is never used here"
            )
        return self

    @property
    def is_neutral(self) -> bool:
        return self.severity == NEUTRAL_SEVERITY[self.kind]

    @property
    def mode(self) -> DegradationMode:
        if self.kind is DegradationKind.WIN_RATE_HAIRCUT:
            return DegradationMode.POST_HOC
        return DegradationMode.RERUN

    def label(self) -> str:
        return f"{self.kind.value}@{self.severity:g}"

    def _seed(self, config: Any) -> int:
        return derive_seed(
            int(config.backtest.seed), "robustness", self.kind.value, self.seed_name
        )

    def overrides(self, config: Any, symbol: str) -> dict[str, Any]:
        """Config overrides for a RERUN degradation. Empty when neutral."""
        kind = self.kind
        if self.mode is DegradationMode.POST_HOC or self.is_neutral:
            return {}
        if kind is DegradationKind.ADDED_SLIPPAGE:
            base = float(config.execution.base_slippage_ticks)
            return {"execution.base_slippage_ticks": base * self.severity}
        if kind is DegradationKind.WIDER_SPREAD:
            typical = float(read_path(config, f"instruments.{symbol}.typical_spread_ticks"))
            return {"execution.spread_ticks_override": typical * self.severity}
        if kind is DegradationKind.DELAYED_ENTRY:
            # The wait horizon is widened past the committed 1 so the delayed
            # entry has somewhere to land -- also the path whose uninspected
            # bars once fabricated 2.06R, re-checked by wait_horizon_violations.
            bars = int(self.severity)
            return {
                "execution.entry_delay_bars": bars,
                "execution.max_entry_wait_bars": max(
                    int(config.execution.max_entry_wait_bars), bars + 1
                ),
            }
        # ExecutionModel seeds its per-order streams from backtest.seed, so a
        # named seed reaches the draws through it.
        field = (
            "execution.random_latency_ms"
            if kind is DegradationKind.RANDOM_EXECUTION_DELAY
            else "execution.random_miss_probability"
        )
        return {field: self.severity, "backtest.seed": self._seed(config)}


def default_degradations(
    config: Any, *, seed_name: str = "default"
) -> tuple[DegradationSpec, ...]:
    """All six kinds, at the severities `config.robustness` commits.

    Neutral severities are kept: each family's neutral cell IS the committed
    configuration, the baseline the rest are read against.

    JUDGEMENT CALL: `RobustnessConfig` commits no grid for the random
    execution delay, so its severities are the committed
    `execution.latency_ms` scaled by (slippage multiplier - 1). That invents
    no number -- a 2.0x slippage run and a 250ms jitter run are the same step
    in two units the config already holds.
    """
    r = config.robustness
    latency = float(config.execution.latency_ms)
    families: tuple[tuple[DegradationKind, Sequence[float]], ...] = (
        (DegradationKind.ADDED_SLIPPAGE, tuple(float(m) for m in r.slippage_multipliers)),
        (DegradationKind.WIDER_SPREAD, tuple(float(m) for m in r.spread_multipliers)),
        (DegradationKind.DELAYED_ENTRY, tuple(float(b) for b in r.entry_delay_bars)),
        (
            DegradationKind.RANDOM_EXECUTION_DELAY,
            tuple(max(0.0, float(m) - 1.0) * latency for m in r.slippage_multipliers),
        ),
        (DegradationKind.RANDOM_MISSED_TRADES, tuple(float(p) for p in r.miss_probabilities)),
        (DegradationKind.WIN_RATE_HAIRCUT, tuple(float(h) for h in r.win_rate_haircuts)),
    )
    return tuple(
        DegradationSpec(
            kind=kind,
            severity=severity,
            seed_name=seed_name if kind in _STOCHASTIC else "",
        )
        for kind, severities in families
        for severity in severities
    )


class DirectionCheck(FrozenModel):
    """One metric, baseline against degraded, and which way it moved."""

    metric: str
    baseline: MetricValue
    degraded: MetricValue
    worse_or_equal: bool | None = None
    note: str = ""

    @classmethod
    def build(cls, metric: str, base: MetricValue, degraded: MetricValue) -> "DirectionCheck":
        before, after = base.or_none(), degraded.or_none()
        verdict: bool | None = None
        note = "undefined on one side, so the direction is not a measurement"
        if before is not None and after is not None:
            verdict = after <= before + COMPARISON_TOLERANCE
            note = (
                ""
                if verdict
                else "the degraded run came out AHEAD of the baseline. Not an edge, and "
                "this check cannot say WHICH mechanism did it -- read the run's own "
                "notes. A RERUN degradation can change which trades exist (a skipped "
                "trade may have been a skipped loser), what they cost (a pinned "
                "spread can be tighter than the quote it replaced) or which bar they "
                "fill on (a delayed or jittered fill is priced off a later bar, which "
                "can open more favourably). Compare the "
                "trade_count check before assuming the trade set moved"
            )
        return cls(
            metric=metric, baseline=base, degraded=degraded,
            worse_or_equal=verdict, note=note,
        )


class DegradationResult(_Unorderable):
    """One degradation run, beside the baseline it is read against."""

    spec: DegradationSpec
    mode: DegradationMode
    split_name: str
    split_phase: SplitPhase
    applied: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = ""
    experiment_id: str = ""
    n_trades: int = Field(ge=0)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    direction: tuple[DirectionCheck, ...] = ()
    wait_horizon_findings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def improved_anywhere(self) -> bool:
        """True if any defined metric came out ahead of the baseline."""
        return any(check.worse_or_equal is False for check in self.direction)

    def describe(self) -> str:
        body = " ".join(str(self.metrics[name]) for name in sorted(self.metrics))
        return f"{self.spec.label()} [{self.mode.value}] n={self.n_trades} {body}"


# ---------------------------------------------------------------------------
# The win-rate haircut: arithmetic, not a fabricated fill
# ---------------------------------------------------------------------------

HAIRCUT_METRICS: tuple[str, ...] = ("trade_count", "win_rate", "expectancy", "profit_factor")

_HAIRCUT_NOTE = (
    "post-hoc arithmetic on the realized R vector: a seeded subset of winners is "
    "re-valued at the sample's mean loss R. No TradeRecord is altered and no fill is "
    "invented, so this stresses the measured distribution; it is not a run and not a "
    "historical result."
)


def _haircut_undefined(reason: Undefined, n: int, note: str) -> dict[str, MetricValue]:
    """Undefined rates, with the trade COUNT still measured.

    A count over an unusable sample is a fact; the rates over it are not.
    `analytics.metrics` draws the line in the same place.
    """
    out = {
        name: MetricValue.undefined_because(f"haircut_{name}", reason, n=n, note=note)
        for name in HAIRCUT_METRICS
    }
    out["trade_count"] = MetricValue.measured(
        "haircut_trade_count", float(n), n=n, unit="trades", note=note
    )
    return out


def win_rate_haircut_metrics(
    trades: Sequence[TradeRecord], *, severity: float, master_seed: int, seed_name: str
) -> dict[str, MetricValue]:
    """Win rate, expectancy and profit factor under a win-rate haircut.

    `ceil(severity * wins)` winners are re-valued at the sample's mean loss R,
    chosen by a named NumPy `Generator`. Ceil rather than round, because a
    haircut that was asked for must bite: rounding 0.05 of three winners to
    zero would report an unchanged result as a degradation. R is read off each
    record (`r_multiple`, net PnL over that record's own `risk_dollars`), so
    this stays correct under either resolution of how `risk_dollars` is
    defined.
    """
    n = len(trades)
    if n == 0:
        return _haircut_undefined(Undefined.NO_TRADES, 0, _HAIRCUT_NOTE)
    values = [float(trade.r_multiple) for trade in trades]
    win_index = [i for i, trade in enumerate(trades) if trade.is_win]
    losses = [values[i] for i, trade in enumerate(trades) if trade.is_loss]
    k = min(math.ceil(severity * len(win_index)) if severity > 0.0 else 0, len(win_index))
    if k and not losses:
        return _haircut_undefined(
            Undefined.NO_LOSSES, n,
            "the sample holds no loss, so there is no measured magnitude to re-value a "
            "winner at. " + _HAIRCUT_NOTE,
        )
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    if k and mean_loss == 0.0:
        return _haircut_undefined(
            Undefined.ZERO_AVERAGE_LOSS, n,
            "losses are present but their mean R is exactly zero, so the haircut has "
            "no magnitude to apply. " + _HAIRCUT_NOTE,
        )
    degraded = list(values)
    if k:
        generator = rng(master_seed, "robustness", "win_rate_haircut", seed_name)
        for offset in sorted(
            int(c) for c in generator.choice(len(win_index), size=k, replace=False)
        ):
            degraded[win_index[offset]] = mean_loss

    wins = sum(1 for value in degraded if value > 0.0)
    gross_profit = sum(value for value in degraded if value > 0.0)
    gross_loss = -sum(value for value in degraded if value < 0.0)
    out = {
        "trade_count": MetricValue.measured(
            "haircut_trade_count", float(n), n=n, unit="trades", note=_HAIRCUT_NOTE
        ),
        "win_rate": MetricValue.measured(
            "haircut_win_rate", wins / n, n=n, unit="fraction", note=_HAIRCUT_NOTE
        ),
        "expectancy": MetricValue.measured(
            "haircut_expectancy", sum(degraded) / n, n=n, unit="R", note=_HAIRCUT_NOTE
        ),
        "profit_factor": (
            MetricValue.measured(
                "haircut_profit_factor", gross_profit / gross_loss, n=n, note=_HAIRCUT_NOTE
            )
            if gross_loss > 0.0
            else MetricValue.undefined_because(
                "haircut_profit_factor", Undefined.NO_LOSSES, n=n,
                note="a profit factor with no losses is undefined, not infinite. "
                + _HAIRCUT_NOTE,
            )
        ),
    }
    return out


# ---------------------------------------------------------------------------
# The wait-horizon invariant
# ---------------------------------------------------------------------------


def wait_horizon_violations(result: RunResult) -> tuple[str, ...]:
    """Trades whose record contradicts the multi-bar wait-horizon fix.

    Five invariants, each one a shape the fixed bug can leave in a record: a
    fill cannot resolve on the bar that produced the signal; an exit cannot
    precede its entry; a position that exited on a LATER bar than it entered
    must have been advanced over at least one bar; a trade whose worst
    excursion went BEYOND its own stop must have exited at the stop -- the
    regression reported such a trade as a TARGET; and a trade that exited AT
    its stop must have an excursion that reached it, or its exit price was on
    no bar the position held. The stop's distance is computed per record from
    that record's own prices and `risk_dollars`, so the check does not depend
    on how `risk_dollars` is defined; a fixed -1.0R bound would raise false
    positives the moment an entry slipped.

    WHAT A RECORD CANNOT SHOW, stated here because the alternative is a
    documented promise standing in for a structural one. The f44ddb9
    regression understated `mae_r` (-0.5 against a true -1.5) BECAUSE it
    skipped the bars, so the record it emitted was internally consistent and
    all five invariants PASS on it -- its excursion never reached its stop,
    and it held a positive number of bars. `bars_held` was understated too (5
    against 7), and an under-count cannot be separated from a clock gap
    without the clock, so only the degenerate zero-bar case is checkable
    here. What verifies the fix is a run over bars whose stop is known to sit
    inside the skipped window, asserted against the loop's own output
    (`tests/unit/test_robustness.py::TestTheWaitHorizonFixHolds`). An empty
    result from this function is the absence of five specific defects, not a
    proof that every bar was inspected.
    """
    findings: list[str] = []
    for trade in result.trades:
        if trade.entry_ts <= trade.signal_ts:
            findings.append(
                f"{trade.trade_id}: entry {trade.entry_ts.isoformat()} does not postdate "
                f"its signal {trade.signal_ts.isoformat()}; the fill used its own bar"
            )
        if trade.exit_ts < trade.entry_ts:
            findings.append(f"{trade.trade_id}: exit precedes entry")
        elif trade.exit_ts > trade.entry_ts and trade.bars_held < 1:
            # Both fills are stamped at their bar's close, so a same-bar
            # entry and exit share a timestamp and bars_held=0 is correct
            # there. A LATER exit means `advance_bar` saw at least one bar
            # close after the entry; zero means no bar between the fill and
            # the exit was ever offered to the position.
            findings.append(
                f"{trade.trade_id}: exit {trade.exit_ts.isoformat()} postdates its entry "
                f"{trade.entry_ts.isoformat()} but bars_held=0, so no bar between the "
                "fill and the exit was ever inspected"
            )
        stop_r = -(
            abs(trade.entry_price - trade.stop_price) * trade.point_value * trade.filled_size
        ) / trade.risk_dollars
        if trade.mae_r < stop_r - 1e-6 and trade.exit_reason is not ExitReason.STOP:
            findings.append(
                f"{trade.trade_id}: mae_r={trade.mae_r:.6f} is beyond its own stop at "
                f"{stop_r:.6f}R, but the exit is {trade.exit_reason.value}. A bar the "
                "position held was never checked for its stop"
            )
        if trade.exit_reason is ExitReason.STOP and trade.mae_r > stop_r + 1e-6:
            # The mirror of the check above, and sound for the same reason:
            # excursions are taken from each held bar's high and low and the
            # exit print is folded in, and the stop is never moved, so a stop
            # exit whose excursion never reached the stop was filled at a
            # price no bar the position held ever traded.
            findings.append(
                f"{trade.trade_id}: the exit is STOP but mae_r={trade.mae_r:.6f} never "
                f"reached its own stop at {stop_r:.6f}R, so the exit price was on no "
                "bar the position held"
            )
    return tuple(findings)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

#: `(config, split) -> RunResult`. Injected so this module drives folds
#: without owning data loading, and so the n=0 path is testable without one.
Runner = Callable[[Any, Split], RunResult]


def _midnight(day: date) -> datetime:
    """UTC midnight on `day`. The seal boundary is UTC, so the clock is too."""
    return datetime.combine(day, time(0, 0), tzinfo=timezone.utc)


class StoreRunner:
    """A `Runner` over a loaded `DataStore`, one fresh loop per call.

    `BacktestLoop` refuses a second `run()` because its `RiskLimitManager`
    accumulates, so a loop is built per call and never reused. The clock is
    clipped to the split's own range, which is what makes a fold a fold.
    """

    def __init__(
        self,
        store: Any,
        symbol: str,
        *,
        calendar: Any = None,
        git_commit: str | None = None,
        git_dirty: bool | None = None,
    ) -> None:
        self.store = store
        self.symbol = symbol
        self.calendar = calendar
        self.git_commit = git_commit
        self.git_dirty = git_dirty

    def __call__(self, config: Any, split: Split) -> RunResult:
        source = StoreMarketSource.from_config(self.store, self.symbol, config)
        start, end = to_ns(_midnight(split.range.start)), to_ns(_midnight(split.range.end))
        clock = [int(ts) for ts in self.store.timeline(self.symbol) if start <= int(ts) < end]
        if not clock:
            raise RobustnessError(
                f"{self.symbol}: no loaded bar falls inside {split.name} "
                f"({split.range}); the store holds no data for this fold"
            )
        loop = BacktestLoop(
            config, self.symbol, calendar=self.calendar,
            split_label=split.phase, split_name=split.name,
            git_commit=self.git_commit, git_dirty=self.git_dirty,
        )
        return loop.run(source, clock_ns=clock)


def _check_split(split: Split, guard: DataAccessGuard | None) -> None:
    if split.phase not in SWEEPABLE_PHASES:
        raise RobustnessError(
            f"{split.name} is {split.phase.value}: a parameter sweep is refused on an "
            f"evaluation phase (sweepable: {[p.value for p in SWEEPABLE_PHASES]}). "
            "Repeatedly fitting a grid against the window that is supposed to judge it "
            "turns that window into a training set"
        )
    if guard is not None:
        guard.check_access(
            split.range.start, split.range.end, purpose=f"robustness sweep {split.name}"
        )


def _measure(
    result: RunResult, split: Split, policy: MetricPolicy | None, names: Sequence[str]
) -> dict[str, MetricValue]:
    return compute_trade_metrics(
        result.trades, split_label=split.phase, policy=policy, names=names
    )


def _payload(metrics: Mapping[str, MetricValue], n_trades: int) -> dict[str, Any]:
    return {
        "n_trades": n_trades,
        "metrics": {name: value.model_dump(mode="json") for name, value in metrics.items()},
    }


def run_grid(
    config: Any,
    registry: SplitRegistry,
    runner: Runner,
    *,
    log: ExperimentLog,
    reason: str,
    split_names: Sequence[str] | None = None,
    axes: Sequence[ParameterAxis] | None = None,
    guard: DataAccessGuard | None = None,
    embargo_days: int | None = None,
    policy: MetricPolicy | None = None,
    parent_experiment_id: str | None = None,
    metrics: Sequence[str] = GRID_CELL_METRICS,
) -> tuple[CellResult, ...]:
    """Run the whole grid on every named split. Returns every cell.

    Calls `registry.assert_embargo` first -- section 10 says Phase 8 is what
    calls it, and a fold boundary with no gap leaks a position held across it.
    Each cell is logged before it runs and has results attached after, so an
    interrupted sweep still leaves its attempts in the table.
    """
    if not reason.strip():
        raise RobustnessError(
            "run_grid needs a reason_for_change: an untracked sweep is how a parameter "
            "gets optimized without anyone being able to count how often the data has "
            "been touched"
        )
    unknown = [name for name in metrics if name not in TRADE_METRICS]
    if unknown:
        raise RobustnessError(f"unknown metric name(s): {unknown}")
    registry.assert_embargo(
        int(config.walk_forward.embargo_days if embargo_days is None else embargo_days)
    )
    axis_list = tuple(axes) if axes is not None else default_axes(config)
    if not axis_list:
        raise RobustnessError("a grid needs at least one axis")
    chosen = (
        [registry.get(name) for name in split_names]
        if split_names is not None
        else [split for split in registry.splits if split.phase in SWEEPABLE_PHASES]
    )
    if not chosen:
        raise RobustnessError(
            "no sweepable split in the registry (have "
            f"{[(s.name, s.phase.value) for s in registry.splits]})"
        )
    for split in chosen:
        _check_split(split, guard)

    cells: list[CellResult] = []
    for split in chosen:
        for combination in itertools.product(*(axis.values for axis in axis_list)):
            pairs = tuple(zip(axis_list, combination))
            point = GridPoint(
                coordinates=tuple((axis.name, float(value)) for axis, value in pairs)
            )
            overrides: dict[str, float] = {}
            for axis, value in pairs:
                overrides.update(axis.overrides(config, float(value)))
            derived = apply_overrides(config, overrides)
            record = log.log(
                name=f"robustness_grid:{split.name}:{point.label()}",
                config_hash=derived.config_hash,
                phase=split.phase,
                split_id=split.name,
                params_changed=overrides,
                reason=f"{reason} | grid cell {point.label()} on {split.name}",
                notes="parameter-grid cell, reported as part of a whole grid; this "
                "module exposes no best cell",
                parent_experiment_id=parent_experiment_id,
            )
            result = runner(derived, split)
            measured = _measure(result, split, policy, metrics)
            log.update_results(record.experiment_id, _payload(measured, result.trade_count))
            cells.append(
                CellResult(
                    point=point, split_name=split.name, split_phase=split.phase,
                    config_hash=derived.config_hash,
                    data_hash=result.provenance.data_hash,
                    experiment_id=record.experiment_id, params_changed=overrides,
                    n_trades=result.trade_count, metrics=measured, notes=result.notes,
                )
            )
            logger.info("grid cell %s on %s: n=%d", point.label(), split.name, result.trade_count)
    return tuple(cells)


def _post_hoc_degradation(
    config: Any,
    split: Split,
    spec: DegradationSpec,
    baseline: RunResult,
    baseline_haircut: Mapping[str, MetricValue],
    *,
    log: ExperimentLog,
    reason: str,
    parent_experiment_id: str,
) -> DegradationResult:
    """The win-rate haircut, whose direction is asserted rather than reported."""
    measured = win_rate_haircut_metrics(
        baseline.trades, severity=spec.severity,
        master_seed=int(config.backtest.seed), seed_name=spec.seed_name,
    )
    checks = tuple(
        DirectionCheck.build(name, baseline_haircut[name], measured[name])
        for name in HAIRCUT_METRICS
    )
    offenders = [check for check in checks if check.worse_or_equal is False]
    if offenders:
        raise RobustnessError(
            f"{spec.label()} improved on the baseline: "
            + "; ".join(f"{c.metric} {c.baseline} -> {c.degraded}" for c in offenders)
            + ". A win-rate haircut is monotone by arithmetic, so this is a defect in "
            "this module, not a finding about the strategy"
        )
    record = log.log(
        name=f"robustness_degradation:{split.name}:{spec.label()}",
        config_hash=config.config_hash,
        phase=split.phase,
        split_id=split.name,
        reason=f"{reason} | {spec.label()} on {split.name}",
        notes=_HAIRCUT_NOTE,
        parent_experiment_id=parent_experiment_id,
    )
    log.update_results(record.experiment_id, _payload(measured, baseline.trade_count))
    return DegradationResult(
        spec=spec, mode=spec.mode, split_name=split.name, split_phase=split.phase,
        config_hash=config.config_hash, experiment_id=record.experiment_id,
        n_trades=baseline.trade_count, metrics=measured, direction=checks,
        notes=(_HAIRCUT_NOTE,),
    )


#: Said on every run that moves the fill to a later bar. Without it the only
#: mechanisms offered for a flattering run are a skipped loser and a pinned
#: spread, and neither is true of a delayed or jittered fill whose trade set
#: never moved.
_LATER_BAR_NOTE = (
    "a delayed fill is a DIFFERENT fill, not the same fill later: the order is priced "
    "off a later bar, which can open more favourably than the one it would have used. "
    "So this run can come out AHEAD of the baseline with its trade set unchanged, and "
    "a flattering result here is that path effect rather than a skipped loser or a "
    "pinned spread"
)


def _degradation_notes(spec: DegradationSpec, config: Any) -> tuple[str, ...]:
    """What this run cannot say, stated on the run rather than in a docstring.

    Three of the six are capable of being VACUOUS at some configuration
    (added slippage at `base_slippage_ticks=0`, a delayed entry under a market
    order, a sub-bar latency jitter) and three of moving the fill price in the
    favourable direction, and a stress test that cannot stress reads as a pass
    unless it says so.
    """
    if spec.is_neutral:
        return ()
    if spec.kind is DegradationKind.ADDED_SLIPPAGE:
        if float(config.execution.base_slippage_ticks) == 0.0:
            return (
                "base_slippage_ticks is 0 and this degradation MULTIPLIES it, so "
                f"severity {spec.severity:g} re-runs the committed configuration "
                "unchanged -- same config hash, same fills. The run is VACUOUS: an "
                "unchanged metric here is arithmetic, not evidence that the strategy "
                "tolerates slippage",
            )
        return ()
    if spec.kind is DegradationKind.WIDER_SPREAD:
        return (
            "spread_ticks_override outranks a real quote by design, so this run both "
            "widens the spread and pins it; where a quote was wider than the pinned "
            "value the run can be cheaper than the baseline",
        )
    if spec.kind is DegradationKind.DELAYED_ENTRY:
        notes = (
            "max_entry_wait_bars is widened, and the invariant of the multi-bar wait "
            "horizon whose uninspected bars once fabricated 2.06R is re-checked per "
            "trade in wait_horizon_findings",
        )
        if str(config.execution.entry_order_type) == "market":
            notes += (
                "entry_order_type is 'market', and PendingOrder.wait_bars is 1 for a "
                "market order whatever max_entry_wait_bars says -- a market order may "
                "not be re-offered on later bars. So this run shifts the fill bar by "
                "entry_delay_bars but builds a SINGLE-bar fill window: it does not "
                "exercise the multi-bar horizon, and its empty wait_horizon_findings "
                "is a statement about this order type, not about the fix. A resting "
                "entry_order_type is what reaches that path",
            )
        return notes + (_LATER_BAR_NOTE,)
    if spec.kind is DegradationKind.RANDOM_EXECUTION_DELAY:
        interval = int(config.data.primary_interval_seconds)
        base_ms = float(config.execution.latency_ms)
        floor_low = math.floor(base_ms / 1000.0 / interval)
        floor_high = math.floor((base_ms + spec.severity) / 1000.0 / interval)
        if floor_low == floor_high:
            return (
                f"this jitter is sub-bar: min_fill_offset floors "
                f"(latency_ms + jitter) / 1000 against a {interval}s bar, and every "
                f"draw in [0, {spec.severity:g}] ms on top of latency_ms="
                f"{base_ms:g} floors to the same {floor_low} bar(s). The fill bar "
                "therefore cannot move, so an unchanged metric here is arithmetic, "
                "not evidence that the strategy tolerates execution delay",
            )
        return (_LATER_BAR_NOTE,)
    return ()


def run_degradations(
    config: Any,
    split: Split,
    runner: Runner,
    symbol: str,
    *,
    log: ExperimentLog,
    reason: str,
    specs: Sequence[DegradationSpec] | None = None,
    guard: DataAccessGuard | None = None,
    policy: MetricPolicy | None = None,
    parent_experiment_id: str | None = None,
) -> tuple[RunResult, tuple[DegradationResult, ...]]:
    """Run every degradation against one split. Returns (baseline, results).

    The baseline is the committed config over the same split, run once. Each
    RERUN spec re-runs it worse; the POST_HOC haircut re-values the baseline's
    own R vector.
    """
    if not reason.strip():
        raise RobustnessError("run_degradations needs a reason_for_change")
    _check_split(split, guard)
    spec_list = tuple(specs) if specs is not None else default_degradations(config)

    root = log.log(
        name=f"robustness_baseline:{split.name}",
        config_hash=config.config_hash,
        phase=split.phase,
        split_id=split.name,
        reason=f"{reason} | degradation baseline on {split.name}",
        notes="the committed configuration, unchanged: the reference every degradation "
        "is read against",
        parent_experiment_id=parent_experiment_id,
    )
    baseline = runner(config, split)
    baseline_metrics = _measure(baseline, split, policy, DEGRADATION_METRICS)
    log.update_results(root.experiment_id, _payload(baseline_metrics, baseline.trade_count))
    baseline_haircut = win_rate_haircut_metrics(
        baseline.trades, severity=0.0,
        master_seed=int(config.backtest.seed), seed_name="baseline",
    )

    out: list[DegradationResult] = []
    for spec in spec_list:
        if spec.mode is DegradationMode.POST_HOC:
            out.append(
                _post_hoc_degradation(
                    config, split, spec, baseline, baseline_haircut,
                    log=log, reason=reason, parent_experiment_id=root.experiment_id,
                )
            )
            continue
        overrides = spec.overrides(config, symbol)
        derived = apply_overrides(config, overrides)
        record = log.log(
            name=f"robustness_degradation:{split.name}:{spec.label()}",
            config_hash=derived.config_hash,
            phase=split.phase,
            split_id=split.name,
            params_changed=overrides,
            reason=f"{reason} | {spec.label()} on {split.name}",
            notes="degradation run: worse costs or worse fills, never better",
            parent_experiment_id=root.experiment_id,
        )
        result = runner(derived, split)
        measured = _measure(result, split, policy, DEGRADATION_METRICS)
        log.update_results(record.experiment_id, _payload(measured, result.trade_count))
        out.append(
            DegradationResult(
                spec=spec, mode=spec.mode, split_name=split.name, split_phase=split.phase,
                applied=dict(overrides), config_hash=derived.config_hash,
                experiment_id=record.experiment_id, n_trades=result.trade_count,
                metrics=measured,
                direction=tuple(
                    DirectionCheck.build(name, baseline_metrics[name], measured[name])
                    for name in DEGRADATION_METRICS
                ),
                wait_horizon_findings=wait_horizon_violations(result),
                notes=_degradation_notes(spec, config) + result.notes,
            )
        )
        logger.info("degradation %s on %s: n=%d", spec.label(), split.name, result.trade_count)
    return baseline, tuple(out)


# ---------------------------------------------------------------------------
# Fragility
# ---------------------------------------------------------------------------


class FragilityVerdict(StrEnum):
    FRAGILE = "FRAGILE"
    ROBUST = "ROBUST"
    UNDEFINED = "UNDEFINED"
    """No measurement was possible -- the n=0 case, which is not a pass."""


class DecayStep(FrozenModel):
    """One adjacent pair on one axis, and the decay between them."""

    axis: str
    metric: str
    from_value: float
    to_value: float
    from_n: int = Field(ge=0)
    to_n: int = Field(ge=0)
    relative_drop: MetricValue
    elasticity: MetricValue
    is_baseline_neighbour: bool = False


class FragilityReport(FrozenModel):
    """The verdict, the numbers behind it, and the cutoffs it used."""

    metric: str
    verdict: FragilityVerdict
    baseline_point: GridPoint | None
    baseline_split: str = ""
    """Which fold the verdict is about. A grid run over several folds holds a
    cell at the committed coordinate in each, and one verdict can only be
    about one of them; `reasons` names the folds this verdict excludes."""

    baseline: MetricValue
    worst_relative_drop: MetricValue
    worst_relative_drop_by_axis: dict[str, MetricValue] = Field(default_factory=dict)
    """The same quantity, per axis, one entry for every axis supplied.

    `worst_relative_drop` is the maximum over EVERY neighbour of the
    committed cell, so on a grid with more than one axis it can come from a
    different parameter than the one a reader is asking about. A
    pre-registered hypothesis names ONE driver, so scoring it needs that
    driver's own number rather than the grid's worst."""

    max_abs_elasticity: MetricValue
    decay_profile: tuple[DecayStep, ...] = ()
    max_relative_drop_allowed: float
    profit_factor_floor: float
    reasons: tuple[str, ...] = ()

    def describe(self) -> tuple[str, ...]:
        return (
            f"fragility[{self.metric}]={self.verdict} (cutoffs: relative_drop>"
            f"{self.max_relative_drop_allowed:g}, profit_factor<"
            f"{self.profit_factor_floor:g})",
            f"baseline {self.baseline}"
            + (f" on {self.baseline_split}" if self.baseline_split else ""),
            f"worst {self.worst_relative_drop}",
            f"elasticity {self.max_abs_elasticity}",
            *self.reasons,
        )


def _undefined(name: str, reason: Undefined, *, n: int = 0, note: str = "") -> MetricValue:
    return MetricValue.undefined_because(name, reason, n=n, note=note)


def _axis_lines(cells: Sequence[CellResult], axis: ParameterAxis) -> list[list[CellResult]]:
    """Cells grouped into lines along `axis`: one group per split and per
    combination of the OTHER coordinates, so adjacent members differ in
    exactly one parameter -- which is what "one step" has to mean."""
    groups: dict[tuple[Any, ...], list[CellResult]] = {}
    for cell in cells:
        key: tuple[Any, ...] = (
            cell.split_name,
            *((name, value) for name, value in cell.point.coordinates if name != axis.name),
        )
        groups.setdefault(key, []).append(cell)
    return list(groups.values())


def _relative_drop(metric: str, base: MetricValue, other: MetricValue) -> MetricValue:
    """(base - other) / |base|, or an explicit undefined with its reason."""
    before, after = base.or_none(), other.or_none()
    name = f"{metric}_relative_drop"
    if before is None:
        return _undefined(
            name, base.undefined or Undefined.NO_TRADES, n=base.n,
            note=f"the baseline {metric} is undefined, so a drop from it is too",
        )
    if after is None:
        return _undefined(
            name, other.undefined or Undefined.NO_TRADES, n=other.n,
            note=f"the compared {metric} is undefined, so a drop to it is too",
        )
    if before == 0.0:
        return _undefined(
            name, Undefined.ZERO_DISPERSION, n=base.n,
            note=f"the baseline {metric} is exactly 0.0, so a RELATIVE drop is a ratio "
            f"with no denominator. The absolute difference is {before - after:.6g}",
        )
    return MetricValue.measured(
        name, (before - after) / abs(before), n=min(base.n, other.n), unit="fraction"
    )


def _decay_profile(
    cells: Sequence[CellResult],
    axes: Sequence[ParameterAxis],
    baseline_key: tuple[str, GridPoint] | None,
    metric: str,
) -> tuple[list[DecayStep], list[tuple[str, CellResult]]]:
    """Every adjacent pair on every axis, plus the baseline's neighbours.

    `baseline_key` is (split_name, point), not a point alone. A grid run over
    several folds holds a cell at the committed coordinate in EVERY fold, so
    matching on the point alone made every fold's cells neighbours of one
    fold's baseline: a drop measured from fold A's expectancy to fold B's is
    not a one-step parameter change, and a defined cell in another fold could
    supply a ROBUST verdict for a fold whose own neighbours were all n=0.
    """
    steps: list[DecayStep] = []
    neighbours: list[tuple[str, CellResult]] = []
    for axis in axes:
        for line in _axis_lines(cells, axis):
            ordered = sorted(line, key=lambda c: c.point.as_mapping()[axis.name])
            for left, right in zip(ordered, ordered[1:]):
                low = left.point.as_mapping()[axis.name]
                high = right.point.as_mapping()[axis.name]
                drop = _relative_drop(metric, left.metric(metric), right.metric(metric))
                step = abs(high - low) / abs(low) if low else 0.0
                if drop.is_defined and step > 0.0:
                    elasticity = MetricValue.measured(
                        f"{metric}_elasticity", drop.require() / step, n=drop.n
                    )
                else:
                    elasticity = _undefined(
                        f"{metric}_elasticity",
                        drop.undefined or Undefined.ZERO_DISPERSION, n=drop.n,
                        note="no relative drop to divide"
                        if not drop.is_defined
                        else "the parameter step is zero relative to its own value",
                    )
                neighbour = False
                if baseline_key is not None and (left.split_name, left.point) == baseline_key:
                    neighbour = True
                    neighbours.append((axis.name, right))
                elif baseline_key is not None and (right.split_name, right.point) == baseline_key:
                    neighbour = True
                    neighbours.append((axis.name, left))
                steps.append(
                    DecayStep(
                        axis=axis.name, metric=metric, from_value=low, to_value=high,
                        from_n=left.n_trades, to_n=right.n_trades,
                        relative_drop=drop, elasticity=elasticity,
                        is_baseline_neighbour=neighbour,
                    )
                )
    return steps, neighbours


def fragility_report(
    config: Any,
    cells: Sequence[CellResult],
    axes: Sequence[ParameterAxis],
    *,
    metric: str = FRAGILITY_METRIC,
) -> FragilityReport:
    """The section 10 fragility verdict. The formula is in the module docstring."""
    max_drop = float(config.robustness.fragility_max_relative_drop)
    pf_floor = float(config.robustness.fragility_profit_factor_floor)
    committed = {axis.name: axis.committed_value(config) for axis in axes}
    # A coordinate for an axis this call was not given has no committed value
    # to be compared against, and defaulting that comparison to the cell's own
    # value made every such coordinate match: a cell at twice the committed
    # risk was then nominated as the committed baseline and its verdict
    # published as the committed configuration's. Refused rather than guessed.
    stray = sorted(
        {
            name
            for cell in cells
            for name, _ in cell.point.coordinates
            if name not in committed
        }
    )
    if stray:
        raise RobustnessError(
            f"the cells carry coordinate(s) {stray} for which no axis was supplied "
            f"(axes given: {sorted(committed)}). The committed coordinate of a missing "
            "axis is unknown, so a cell at an arbitrary value on it would be reported "
            "as the committed baseline. Pass every axis the grid was run over"
        )
    at_baseline = [
        cell
        for cell in cells
        if all(
            math.isclose(value, committed.get(name, value), rel_tol=0.0, abs_tol=1e-12)
            for name, value in cell.point.coordinates
        )
    ]
    baseline_cell = at_baseline[0] if at_baseline else None
    baseline_point = baseline_cell.point if baseline_cell is not None else None
    baseline_split = baseline_cell.split_name if baseline_cell is not None else ""
    baseline_key = (
        (baseline_cell.split_name, baseline_cell.point)
        if baseline_cell is not None
        else None
    )
    baseline = (
        baseline_cell.metric(metric)
        if baseline_cell is not None
        else _undefined(
            metric, Undefined.INSUFFICIENT_OBSERVATIONS,
            note=f"the grid holds no cell at the committed coordinate {committed}, so "
            "there is nothing for a one-step change to be a change FROM",
        )
    )
    steps, neighbours = _decay_profile(cells, axes, baseline_key, metric)

    # Always measured FROM the baseline: reusing a step's own drop would divide
    # by the neighbour's value on steps running INTO the baseline, which is a
    # different quantity wearing the same name.
    drops: list[tuple[str, CellResult, MetricValue]] = []
    for axis_name, cell in neighbours:
        drop = _relative_drop(metric, baseline, cell.metric(metric))
        if drop.is_defined:
            drops.append((axis_name, cell, drop))

    reasons: list[str] = []
    worst_name = f"{metric}_worst_relative_drop"
    other_splits = sorted(
        {cell.split_name for cell in at_baseline[1:]}
        if len(at_baseline) > 1
        else set()
    )
    if other_splits:
        reasons.append(
            f"this verdict covers {baseline_split} only: the grid also holds a cell at "
            f"the committed coordinate in {other_splits}, and a step from one fold's "
            "baseline to another fold's cell is not a one-step parameter change. Call "
            "fragility_report once per fold to read the others"
        )
    if not baseline.is_defined:
        verdict = FragilityVerdict.UNDEFINED
        worst = _undefined(
            worst_name, baseline.undefined or Undefined.NO_TRADES, n=baseline.n,
            note=f"the baseline cell measured {baseline}. A decay per unit of parameter "
            "change cannot be computed from an undefined starting point, and 0.0 would "
            "be a claim the data does not support",
        )
        reasons.append(
            f"undefined: baseline {metric} is {baseline}. This is the live case at the "
            "committed configuration, not a failure of the sweep"
        )
    elif not drops:
        verdict = FragilityVerdict.UNDEFINED
        worst = _undefined(
            worst_name, Undefined.INSUFFICIENT_OBSERVATIONS, n=baseline.n,
            note="no neighbouring cell has a defined metric to compare against",
        )
        reasons.append("undefined: no neighbouring cell of the baseline is defined")
    else:
        worst_axis, worst_cell, worst_drop = max(
            drops, key=lambda item: item[2].require()
        )
        worst_value = worst_drop.require()
        # n is the drop's own n (min of the two cells), NOT the baseline's.
        # A 3-trade neighbour against a 400-trade baseline is a 3-trade
        # measurement, and labelling it n=400 is exactly how a three-trade
        # cell reads as a result.
        worst = MetricValue.measured(
            worst_name, worst_value, n=worst_drop.n, unit="fraction",
            note=f"worst over the baseline's neighbours: one step along {worst_axis} "
            f"to [{worst_cell.point.label()}] on {worst_cell.split_name}, whose own "
            f"trade count is n={worst_cell.n_trades} (baseline n={baseline.n})",
        )
        verdict = FragilityVerdict.ROBUST
        if worst_value > max_drop:
            verdict = FragilityVerdict.FRAGILE
            reasons.append(
                f"fragile: a one-step parameter change along {worst_axis} to "
                f"[{worst_cell.point.label()}] (n={worst_cell.n_trades}) drops {metric} "
                f"by {worst_value:.4f} relative, above the committed "
                f"fragility_max_relative_drop={max_drop:g}"
            )
        base_pf = at_baseline[0].metric("profit_factor")
        if base_pf.is_defined and base_pf.require() >= pf_floor:
            for axis_name, cell, _ in drops:
                pf = cell.metric("profit_factor")
                if pf.is_defined and pf.require() < pf_floor:
                    verdict = FragilityVerdict.FRAGILE
                    reasons.append(
                        f"fragile: one step along {axis_name} to [{cell.point.label()}] "
                        f"drops the profit factor to {pf.require():.4f}, below the "
                        f"committed fragility_profit_factor_floor={pf_floor:g} "
                        f"(n={cell.n_trades})"
                    )
        if verdict is FragilityVerdict.ROBUST:
            reasons.append(
                f"robust at these cutoffs: worst one-step relative drop "
                f"{worst_value:.4f} <= {max_drop:g}, measured from a baseline of "
                f"n={baseline.n} trades against a neighbour of n="
                f"{worst_cell.n_trades}. A verdict over these sample sizes is not "
                "evidence of an edge"
            )

    # The formula's elasticity(N) = relative_drop(N) / step(N), with BOTH
    # measured from the baseline. A DecayStep divides by its own lower cell,
    # so for a neighbour sitting BELOW the baseline on an axis the step's
    # elasticity is a different ratio -- and a smaller one whenever the
    # baseline is the larger value, which understates the decay per unit of
    # parameter change. The baseline-measured value is added to the pool, so
    # the reported maximum can only rise.
    elasticities = [abs(s.elasticity.require()) for s in steps if s.elasticity.is_defined]
    if baseline_cell is not None:
        at = baseline_cell.point.as_mapping()
        for axis_name, cell_, drop in drops:
            from_value = at.get(axis_name)
            to_value = cell_.point.as_mapping().get(axis_name)
            if from_value is None or to_value is None or not from_value:
                continue
            relative_step = abs(to_value - from_value) / abs(from_value)
            if relative_step > 0.0:
                elasticities.append(abs(drop.require() / relative_step))

    by_axis: dict[str, MetricValue] = {}
    for axis in axes:
        name = f"{metric}_worst_relative_drop"
        own = [drop for axis_name, _, drop in drops if axis_name == axis.name]
        if own:
            by_axis[axis.name] = MetricValue.measured(
                name, max(drop.require() for drop in own),
                n=min(drop.n for drop in own), unit="fraction",
                note=f"worst one-step relative drop along {axis.name} alone, measured "
                "from the committed cell",
            )
        elif not baseline.is_defined:
            by_axis[axis.name] = _undefined(
                name, baseline.undefined or Undefined.NO_TRADES, n=baseline.n,
                note=f"the baseline {metric} is undefined, so a drop along "
                f"{axis.name} is too",
            )
        else:
            by_axis[axis.name] = _undefined(
                name, Undefined.INSUFFICIENT_OBSERVATIONS, n=baseline.n,
                note=f"no neighbour of the committed cell along {axis.name} has a "
                f"defined {metric}",
            )

    return FragilityReport(
        metric=metric,
        verdict=verdict,
        baseline_point=baseline_point,
        baseline_split=baseline_split,
        baseline=baseline,
        worst_relative_drop=worst,
        worst_relative_drop_by_axis=by_axis,
        max_abs_elasticity=(
            MetricValue.measured(
                f"{metric}_max_abs_elasticity", max(elasticities), n=len(elasticities)
            )
            if elasticities
            else _undefined(
                f"{metric}_max_abs_elasticity", Undefined.INSUFFICIENT_OBSERVATIONS,
                note="no adjacent pair of cells has a defined metric on both sides",
            )
        ),
        decay_profile=tuple(steps),
        max_relative_drop_allowed=max_drop,
        profit_factor_floor=pf_floor,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Reporting against pre-registered hypotheses
# ---------------------------------------------------------------------------


def fold_profit_concentration(
    folds: Mapping[str, Sequence[TradeRecord]],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Largest fold's share of gross profit, for `H-CONSISTENCY-OVER-PEAK`.

    GROSS profit, not net: a net denominator can be zero or negative and a
    ratio against it would be presentable nonsense. Gross profit is
    non-negative and its denominator vanishes only when no trade won at all,
    which is `NO_WINS`.
    """
    shares: list[float] = []
    n_total = 0
    for name in sorted(folds):
        trades = tuple(folds[name])
        n_total += len(trades)
        value = TRADE_METRICS["gross_profit"](
            trades, split_label=split_label, policy=policy
        ).or_none()
        shares.append(0.0 if value is None else value)
    if n_total == 0:
        return _undefined(
            "fold_profit_concentration", Undefined.NO_TRADES,
            note="no fold holds a trade, so profit cannot be concentrated anywhere",
        )
    total = sum(shares)
    if total <= 0.0:
        return _undefined(
            "fold_profit_concentration", Undefined.NO_WINS, n=n_total,
            note="gross profit across folds is zero, so there is no profit for a single "
            "fold to hold a share of",
        )
    # One fold holds 1.0 of its own profit by arithmetic. The number is still
    # reported -- removing it would hide a measurement -- but it carries the
    # reason it is not a finding, because `H-CONSISTENCY-OVER-PEAK` is refuted
    # above 0.50 and a one-fold sweep would otherwise refute it every time
    # without having compared anything.
    note = (
        ""
        if len(folds) > 1
        else f"only {len(folds)} fold was supplied, so this share is 1.0 by arithmetic "
        "rather than by concentration: there is no second fold for profit to have been "
        "spread across. Read it as not-yet-measured, not as a refutation"
    )
    return MetricValue.measured(
        "fold_profit_concentration", max(shares) / total, n=n_total, unit="fraction",
        note=note,
    )


class HypothesisReport(FrozenModel):
    """A pre-registered hypothesis, reported against rather than restated."""

    hypothesis_id: str
    metric: str
    driver: str
    evaluable: bool
    comparison: TargetComparison | None = None
    refuted: bool | None = None
    consequence: str = ""
    note: str = ""

    def describe(self) -> str:
        if not self.evaluable:
            verdict = "not evaluable"
        elif self.refuted is None:
            verdict = "no measurement"
        else:
            verdict = "REFUTED" if self.refuted else "not refuted"
        body = self.comparison.describe() if self.comparison is not None else self.note
        return f"{self.hypothesis_id}: {verdict} -- {body}"


def _hypothesis_report(
    hypothesis_id: str, measured: MetricValue | None, *, note: str = ""
) -> HypothesisReport:
    h = by_id(hypothesis_id)
    fields = {
        "hypothesis_id": h.hypothesis_id,
        "metric": h.metric,
        "driver": h.driver,
        "consequence": h.consequence,
    }
    if measured is None or h.threshold is None:
        return HypothesisReport(
            **fields,
            evaluable=False,
            note=note
            or f"this sweep produced no measurement of {h.metric!r}; the hypothesis is "
            "neither supported nor refuted by it",
        )
    comparison = compare_to_target(
        measured, h.threshold,
        label=f"{h.hypothesis_id} ({h.metric})",
        comparison=(
            Comparison.AT_MOST if h.direction is Direction.AT_MOST else Comparison.AT_LEAST
        ),
    )
    return HypothesisReport(
        **fields,
        evaluable=True,
        comparison=comparison,
        refuted=None if comparison.met is None else not comparison.met,
        note=note,
    )


def hypothesis_reports(
    fragility: FragilityReport,
    axes: Sequence[ParameterAxis],
    *,
    concentration: MetricValue | None = None,
) -> tuple[HypothesisReport, ...]:
    """Report this sweep against the two hypotheses section 10 names.

    `H-SR-CBAND-ROBUST` is a claim about `c_band`, so it is evaluable only
    when the grid actually swept that driver. Scoring it against the
    score-threshold axis instead would answer a different question with the
    same number.
    """
    path = HYPOTHESIS_DRIVER_PATHS["c_band"]
    # The axis that actually moves c_band, by name or by the path it governs.
    # The number must come from THAT axis: `worst_relative_drop` is the
    # maximum over every neighbour on every axis, so on a joint grid it can
    # be a flow-score step, and scoring a pre-registered claim about c_band
    # with it reports a refutation the driver never produced.
    driver_axis = next(
        (axis for axis in axes if axis.name == "c_band" or path in axis.paths), None
    )
    if driver_axis is not None:
        measured = fragility.worst_relative_drop_by_axis.get(driver_axis.name)
        if measured is None:
            measured = _undefined(
                f"{fragility.metric}_worst_relative_drop",
                Undefined.INSUFFICIENT_OBSERVATIONS,
                note=f"the fragility report carries no per-axis drop for "
                f"{driver_axis.name!r}",
            )
        others = sorted(axis.name for axis in axes if axis.name != driver_axis.name)
        c_band = _hypothesis_report(
            "H-SR-CBAND-ROBUST", measured,
            note=f"measured as the worst one-step relative drop in {fragility.metric} "
            f"across the {driver_axis.name} neighbours of the committed cell ALONE"
            + (
                f"; the grid also swept {others}, whose steps are excluded because this "
                "hypothesis names one driver"
                if others
                else ""
            ),
        )
    else:
        c_band = _hypothesis_report(
            "H-SR-CBAND-ROBUST", None,
            note=f"not evaluable: this grid swept {sorted(a.name for a in axes)} and not "
            f"c_band ({path}). Build the axis with axis_for_driver('c_band', values)",
        )
    consistency = _hypothesis_report(
        "H-CONSISTENCY-OVER-PEAK", concentration,
        note="measured as the largest fold's share of gross profit. The "
        "sign-disagreement half of the refutation condition needs the per-fold "
        "expectancies, which the fold cells carry"
        if concentration is not None
        else "",
    )
    return (c_band, consistency)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


class RobustnessReport(FrozenModel):
    """The whole grid, the degradations, and the fragility verdict.

    No method returns a best, optimal or recommended cell, and the rows refuse
    to be ordered, so nothing downstream can turn this into a parameter choice
    without saying so in its own source.
    """

    axes: tuple[ParameterAxis, ...]
    cells: tuple[CellResult, ...]
    degradations: tuple[DegradationResult, ...] = ()
    fragility: FragilityReport | None = None
    hypotheses: tuple[HypothesisReport, ...] = ()
    baseline_n_trades: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def cells_with_trades(self) -> int:
        return sum(1 for cell in self.cells if cell.n_trades > 0)

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        rows = (*self.cells, *self.degradations)
        return tuple(row.experiment_id for row in rows if row.experiment_id)

    def wait_horizon_findings(self) -> tuple[str, ...]:
        return tuple(f for row in self.degradations for f in row.wait_horizon_findings)

    def describe(self) -> tuple[str, ...]:
        lines = [
            f"robustness: {self.n_cells} grid cell(s), {self.cells_with_trades} with at "
            f"least one trade, {len(self.degradations)} degradation run(s)"
        ]
        lines += [cell.describe() for cell in self.cells]
        lines += [row.describe() for row in self.degradations]
        if self.fragility is not None:
            lines += list(self.fragility.describe())
        lines += [report.describe() for report in self.hypotheses]
        return tuple(lines + list(self.notes))


#: Said on every report, because a grid whose low cells trade and whose high
#: cells do not invites exactly one wrong reading.
GRID_READING_NOTE = (
    "Read this as a grid. Cells at a lower score threshold take trades only because "
    "the gate was lowered until something passed; that is not evidence the system works "
    "at a lower threshold, and every number carries its cell's trade count so a "
    "three-trade cell cannot be mistaken for evidence. No cell here is a "
    "recommendation, and this module exposes no way to ask which one won."
)


def robustness_report(
    config: Any,
    cells: Sequence[CellResult],
    axes: Sequence[ParameterAxis],
    *,
    degradations: Sequence[DegradationResult] = (),
    baseline: RunResult | None = None,
    concentration: MetricValue | None = None,
    metric: str = FRAGILITY_METRIC,
    notes: Sequence[str] = (),
) -> RobustnessReport:
    """Assemble the report: grid, degradations, fragility, hypotheses."""
    fragility = fragility_report(config, cells, axes, metric=metric)
    extra = list(notes)
    improved = [row for row in degradations if row.improved_anywhere]
    if improved:
        extra.append(
            "degradation runs that came out ahead of the baseline on at least one "
            "metric: " + ", ".join(row.spec.label() for row in improved)
            + ". Not a result. A RERUN degradation can change which trades exist (a "
            "skipped trade can be a skipped loser), what they cost (a pinned spread "
            "can be tighter than the quote it replaced, which leaves the trade set "
            "untouched and the run cheaper) or which bar they fill on (a delayed or "
            "jittered fill is priced off a later bar, which can open more favourably "
            "and also leaves the trade set untouched). Each row's own notes say which "
            "applies"
        )
    findings = tuple(f for row in degradations for f in row.wait_horizon_findings)
    delayed = [
        row
        for row in degradations
        if row.spec.kind is DegradationKind.DELAYED_ENTRY and not row.spec.is_neutral
    ]
    if findings:
        extra.append(
            f"WAIT-HORIZON INVARIANT VIOLATED in {len(findings)} trade(s): "
            + "; ".join(findings[:3])
            + (" ..." if len(findings) > 3 else "")
        )
    elif delayed:
        extra.append(
            "wait-horizon invariant holds on every trade the delayed-entry runs produced "
            f"({sum(row.n_trades for row in delayed)} trade(s) inspected). At n=0 the "
            "check is vacuous, which is a statement about the sample, not about the fix"
        )
    return RobustnessReport(
        axes=tuple(axes),
        cells=tuple(cells),
        degradations=tuple(degradations),
        fragility=fragility,
        hypotheses=hypothesis_reports(fragility, axes, concentration=concentration),
        baseline_n_trades=None if baseline is None else baseline.trade_count,
        notes=(GRID_READING_NOTE, *extra),
    )
