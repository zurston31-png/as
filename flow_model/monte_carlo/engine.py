"""Monte Carlo resampling of a realized trade sequence (ARCHITECTURE.md 11).

Three methods, one result TYPE per outcome, and no number that outlives the
sample it came from. Each method's own mechanics and declared artifacts are
documented at the function that implements it; this header states the four
things that are true across all of them.

## 1. n = 0 IS THE LIVE PATH, AND IT IS A DIFFERENT TYPE

The committed config takes **no trades**. The figure of record is the
full-year population scan, not the gated subsample: over 11239 scored
synthetic NQ 5-minute bars in 2016 the maximum Flow Score is 69.6335 against
a 70.0 threshold and zero bars reach it, missing by 0.37 points out of 100
(PROVENANCE.md, "The result" and "The tail, counted"; the gated subsample's
66.991221 in 2.4 is the lower, selected figure).

Read it precisely, because PROVENANCE corrected itself here: the claim is
NOT that the score cannot reach 70. The tail approaches the threshold and
roughly quarters every two points (30 -> 12 -> 4 -> 1), so bars above 70.0
are expected over a long enough window -- rarely. PROVENANCE's own ESTIMATE,
labelled there as an estimate rather than a measurement, is 2 to 5 signals
over the ten-year window before eight gates that reject 97% of bars, so a
plausible trade count of zero to one.

That makes BOTH paths in this module live, not one:

* n=0 -- refused as a type. Resampling an empty list cannot produce a
  distribution of anything: there is no empirical R distribution to
  preserve, no autocorrelation to block, and no regime sequence to estimate
  a chain from.
* n=1 or 2 -- the realistic non-zero case, and section 2 below. A bootstrap
  there is a bootstrap over one or two numbers.

Neither is an edge case to be handled after the interesting work. They are
the work.

The temptation is the shape of a populated result with zeros in it -- ten
thousand flat curves at the starting balance, a fan that is one horizontal
line, `P(drawdown >= 10%) = 0.0`. Each is arithmetically true of the empty
set and each renders as a confident picture of a strategy that never traded.
`P(ruin) = 0` for an account that never opened a position is the most
flattering number in this file's reach.

So the refusal is the TYPE, not a flag on a value:

* `run_monte_carlo` returns `MonteCarloRefusal | MonteCarloDistribution`.
* `MonteCarloRefusal` declares no percentile, probability or fan field, and
  `extra="forbid"` means none can be attached. Reading `terminal_equity` off
  one raises `AttributeError` at the line that made the mistake, and a type
  checker flags it before that.
* `MonteCarloDistribution.n_trades` is `Field(ge=1)`, so the populated type
  cannot be constructed at n=0 at all, and its validator rejects any
  `MetricValue` that is undefined or whose `n` disagrees with `n_trades`.

To check it: read `MonteCarloRefusal`'s field list, and
`MonteCarloDistribution`'s `n_trades: int = Field(ge=1)`.
`tests/unit/test_monte_carlo.py` asserts both, including that constructing
the populated type at n=0 raises.

## 2. SMALL n IS THE OTHER DISHONEST CASE

A bootstrap over 3 trades resamples the same 3 trades. Its percentile spread
is a property of those three numbers, not knowledge about the future, and
10,000 simulations add no information -- they only make the artifact smooth.

* **`n` on every `MetricValue` here is the number of distinct observed trades
  in the resample pool, never the simulation count.** A percentile from
  100,000 sims over 3 trades carries 3 trades of information; writing 100000
  into `n` would be the single most misleading number this module could
  print. The simulation count is a separate, labelled field.
* `sample` is `SampleStatus.INSUFFICIENT` below `min_sample_n`
  (`config.analytics.min_sample_n`, the same declared minimum
  `analytics.breakdowns` suppresses cells on), and a note says in words that
  the spread is an artifact of the pool size.

Both uncertainties are stated rather than merged: simulation error on a
probability is `sqrt(p(1-p)/n_simulations)` and shrinks with compute;
sampling error is governed by `n_trades` and compute cannot touch it.

## 3. THE STATED LIMITATION IS A REQUIRED FIELD

Section 11 requires it surfaced in every MC report. A docstring is skippable
and a class constant does not serialize, so `STATED_LIMITATION` is a required
field on both result types, validated verbatim. It therefore appears in
`model_dump()`, in the JSON a dashboard reads, and first in `describe()`.
There is no constructor path to a percentile that does not carry it.

The regime half of it is enforced too, not just quoted: `RegimeTransitions`
refuses to hold a row for a regime the sample never showed, or for one that
only ever appeared last. See `estimate_transitions` and `_regime_indices`.

## 4. R, DOLLARS, DETERMINISM, PROVENANCE

R is READ off the record (`TradeRecord.r_multiple` = `pnl / risk_dollars` on
the PLANNED unit of section 0.2) and never re-derived. But R is a ratio and
an equity path needs dollars, and `TradeRecord` carries `risk_dollars`
without the equity it was sized against, so the pool cannot supply the
fraction. `MonteCarloPlan.risk_fraction` declares it
(`config.risk.risk_per_trade_pct` by default) and is echoed on the result;
`r_dollars_modeled` is reported beside `mean_risk_dollars_observed` so the
assumption is checkable. There is deliberately no terminal figure quoted in
R: under compounding the dollar value of one R moves along the path, so
dividing total dollars by the initial R would put a unit on it that it does
not have. `_equity_paths` has the two sizing modes and the ruin rule.

Randomness comes only from `core.determinism.rng(seed, "monte_carlo",
method)`; the global NumPy state is never touched and no clock is read.
Draws are chunked to bound memory at 100,000 sims and the chunk size is a
pure function of the path length and one declared constant, so the draw
sequence depends on the inputs alone: same plan + same pool => bit-identical
output.

Monte Carlo adds no information. Every number is a property of the input
trade set and inherits its provenance exactly; the distinct `data_hash` and
`config_hash` values of the resampled records are carried on the result.
Every trade this project has produced came from `data/synthetic.py`
(PROVENANCE.md 1), so a percentile computed from them describes a generator
and not any market. `PROVENANCE_NOTE` says so on every result.

Nothing here places, routes or simulates an order. It reads completed
`TradeRecord`s and resamples numbers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from flow_model.analytics.metrics import (
    MetricValue,
    SampleStatus,
    Undefined,
    assert_single_phase,
    chronological,
    sample_status,
)
from flow_model.config.schema import AnalyticsConfig, MonteCarloConfig, RiskConfig
from flow_model.core.contracts import TradeRecord
from flow_model.core.determinism import rng
from flow_model.core.enums import MonteCarloMethod, Regime, SplitPhase
from flow_model.core.model import FrozenModel

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: ARCHITECTURE.md section 11, verbatim. The source renders `sequence` and
#: `model` as markdown emphasis; capitals carry that emphasis into plain text.
#: `tests/unit/test_monte_carlo.py` checks this against the document itself,
#: so the sentence cannot drift from the architecture it quotes.
STATED_LIMITATION: str = (
    "resampling historical trades cannot produce regimes absent from the sample. "
    "MC bounds SEQUENCE risk, not MODEL risk."
)

#: On every result, because a resampled figure reads like a measurement.
PROVENANCE_NOTE: str = (
    "Monte Carlo adds no information: every figure here is a property of the input "
    "trade set and inherits its provenance exactly. Every trade this project has "
    "produced came from flow_model/data/synthetic.py, so a percentile computed from "
    "them describes that generator and not any market. The input data_hash values "
    "are carried on this result."
)

#: Largest simulation count section 11 commits to supporting. A hard ceiling
#: rather than a documented one, so an accidental extra zero is an error with
#: a name instead of an out-of-memory kill.
MAX_SIMULATIONS: int = 100_000

#: Most step columns an equity fan reports. An exact percentile at a step
#: needs every simulation's value there, so the STEP GRID is subsampled
#: rather than the simulations: 100,000 x 101 floats is bounded, 100,000 x
#: n_trades is not. Shorter paths report every step.
FAN_MAX_STEPS: int = 101

#: Draw budget per chunk, in matrix elements. JUDGEMENT CALL, not a config
#: field: it trades peak memory against loop overhead and changes no result.
#: Part of the determinism contract -- `_chunk_sims` depends on this and the
#: path length alone, so chunking cannot move a number.
SIM_ELEMENT_BUDGET: int = 2_000_000

#: Signed depth edges for the drawdown histogram, ascending, matching the
#: negative-fraction convention of `EquityPoint.drawdown_pct` and
#: `analytics.DrawdownProfile`. Open tails at both ends.
DRAWDOWN_HISTOGRAM_EDGES: tuple[float, ...] = (-0.50, -0.40, -0.30, -0.20, -0.10, -0.05, -0.01)


# ---------------------------------------------------------------------------
# The resolved plan
# ---------------------------------------------------------------------------


class MonteCarloPlan(FrozenModel):
    """Every input the engine reads, resolved from config once and echoed.

    `n_simulations` is `ge=1` here while `MonteCarloConfig.n_simulations` is
    `ge=100`: the config floor states the policy for a reported run, and a
    plan built in a test needs to be cheap. Nothing here changes a default.
    """

    method: MonteCarloMethod
    n_simulations: int = Field(ge=1)
    block_length: int = Field(gt=0)
    seed: int
    compound: bool
    starting_equity: float = Field(gt=0.0)
    risk_fraction: float = Field(gt=0.0, lt=1.0)
    drawdown_thresholds: tuple[float, ...]
    streak_thresholds: tuple[int, ...]
    percentiles: tuple[float, ...]
    n_curves_to_store: int = Field(ge=0)
    min_sample_n: int = Field(ge=1)
    trades_per_simulation: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> "MonteCarloPlan":
        if self.n_simulations > MAX_SIMULATIONS:
            raise ValueError(
                f"n_simulations={self.n_simulations} exceeds MAX_SIMULATIONS="
                f"{MAX_SIMULATIONS}; section 11 commits to supporting that many"
            )
        if not self.percentiles:
            raise ValueError("at least one percentile is required")
        if not all(0.0 < p < 100.0 for p in self.percentiles):
            raise ValueError("percentiles are strictly between 0 and 100")
        if list(self.percentiles) != sorted(self.percentiles):
            raise ValueError("percentiles must be sorted ascending")
        if not all(0.0 < d < 1.0 for d in self.drawdown_thresholds):
            raise ValueError("drawdown_thresholds are fractions in (0, 1)")
        if not all(k > 0 for k in self.streak_thresholds):
            raise ValueError("streak_thresholds are positive trade counts")
        return self

    @classmethod
    def from_parts(
        cls, monte_carlo: MonteCarloConfig, risk: RiskConfig, analytics: AnalyticsConfig
    ) -> "MonteCarloPlan":
        return cls(
            method=monte_carlo.method,
            n_simulations=monte_carlo.n_simulations,
            block_length=monte_carlo.block_length,
            seed=monte_carlo.seed,
            compound=monte_carlo.compound,
            starting_equity=risk.starting_equity,
            risk_fraction=risk.risk_per_trade_pct,
            drawdown_thresholds=monte_carlo.drawdown_thresholds,
            streak_thresholds=monte_carlo.streak_thresholds,
            percentiles=monte_carlo.percentiles,
            n_curves_to_store=monte_carlo.n_curves_to_store,
            min_sample_n=analytics.min_sample_n,
            trades_per_simulation=monte_carlo.trades_per_simulation,
        )

    @classmethod
    def from_config(cls, config: object) -> "MonteCarloPlan":
        """Resolve from the root config.

        `compound` is `config.monte_carlo.compound`, deliberately NOT
        `config.risk.compound_equity`: "what would this look like without
        compounding" has to be askable without changing how the backtest
        sized anything.
        """
        monte_carlo = getattr(config, "monte_carlo", None)
        risk = getattr(config, "risk", None)
        analytics = getattr(config, "analytics", None)
        if (
            not isinstance(monte_carlo, MonteCarloConfig)
            or not isinstance(risk, RiskConfig)
            or not isinstance(analytics, AnalyticsConfig)
        ):
            raise TypeError(
                "MonteCarloPlan.from_config wants the root config (monte_carlo, risk "
                f"and analytics sections), got {type(config).__name__}"
            )
        return cls.from_parts(monte_carlo, risk, analytics)

    @classmethod
    def default(cls) -> "MonteCarloPlan":
        """The schema defaults, constructed in-process (no file, no I/O)."""
        return cls.from_parts(MonteCarloConfig(), RiskConfig(), AnalyticsConfig())

    @property
    def r_dollars(self) -> float:
        """Dollars one R is worth at the starting equity. The declared map."""
        return self.starting_equity * self.risk_fraction


# ---------------------------------------------------------------------------
# The regime chain, with its holes left as holes
# ---------------------------------------------------------------------------


class RegimeTransitions(FrozenModel):
    """An empirical Markov chain over the regimes the trades actually visited.

    The invariant worth a validator: `rows` holds a probability row for a
    state if and only if that state was observed WITH at least one outgoing
    transition. A terminal state and an unobserved regime are both named and
    neither can be given a row, so "no smoothing" is enforced, not asserted.
    """

    states: tuple[Regime, ...]
    """Regimes observed among the trades, in `Regime` declaration order."""

    occupancy: tuple[int, ...]
    """Trades per state, aligned to `states`."""

    counts: tuple[tuple[int, ...], ...]
    """Raw observed transition counts, states x states. No prior added."""

    rows: dict[str, tuple[float, ...]]
    """Estimated rows only, keyed by `Regime` value, aligned to `states`."""

    initial_distribution: tuple[float, ...]
    """Empirical occupancy frequency. DECLARED CHOICE: the alternative is the
    first trade's regime, which is an estimate from a sample of one."""

    terminal_states: tuple[Regime, ...]
    """Observed, zero outgoing transitions, so no row. A path ends here."""

    unobserved_regimes: tuple[Regime, ...]
    """`Regime` members absent from the sample, so absent from the chain.
    `Regime.UNKNOWN` is expected here: section 6 makes it untradable."""

    n_transitions: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> "RegimeTransitions":
        names = {g.value for g in self.states}
        width = len(self.states)
        if len(self.occupancy) != width or len(self.initial_distribution) != width:
            raise ValueError("occupancy and initial_distribution align to states")
        if len(self.counts) != width or any(len(row) != width for row in self.counts):
            raise ValueError("counts must be a square matrix over states")
        for state in self.terminal_states:
            if state.value in self.rows:
                raise ValueError(
                    f"{state.value} has no outgoing transition and therefore no row; a "
                    "uniform prior here would manufacture a regime the sample never "
                    "showed (section 11)"
                )
        for regime in self.unobserved_regimes:
            if regime in self.states or regime.value in self.rows:
                raise ValueError(f"{regime.value} is listed unobserved but present")
        for key, row in self.rows.items():
            if key not in names or len(row) != width:
                raise ValueError(f"row {key!r} must belong to states and align to them")
            if abs(sum(row) - 1.0) > 1e-9 or any(p < 0.0 for p in row):
                raise ValueError(f"row {key!r} is not a probability distribution")
        return self

    def describe(self) -> tuple[str, ...]:
        lines = [
            f"regime chain: {len(self.states)} state(s), "
            f"{self.n_transitions} observed transition(s)"
        ]
        for state, n in zip(self.states, self.occupancy):
            row = self.rows.get(state.value)
            shown = (
                "TERMINAL (no row, paths end here)"
                if row is None
                else str(tuple(round(p, 6) for p in row))
            )
            lines.append(f"  {state.value}: n={n} -> {shown}")
        if self.unobserved_regimes:
            lines.append(
                "  unobserved, given no state: "
                + ", ".join(g.value for g in self.unobserved_regimes)
            )
        return tuple(lines)


def estimate_transitions(regimes: Sequence[Regime]) -> RegimeTransitions:
    """Count transitions between consecutive outcomes. No smoothing anywhere."""
    present = set(regimes)
    states = tuple(g for g in Regime if g in present)
    index = {g: i for i, g in enumerate(states)}
    width = len(states)

    counts = np.zeros((width, width), dtype=np.int64)
    for a, b in zip(regimes, regimes[1:]):
        counts[index[a], index[b]] += 1
    totals = counts.sum(axis=1)
    occupancy = np.array([sum(1 for g in regimes if g is s) for s in states], dtype=np.int64)
    total = int(occupancy.sum())

    return RegimeTransitions(
        states=states,
        occupancy=tuple(int(v) for v in occupancy),
        counts=tuple(tuple(int(v) for v in row) for row in counts),
        rows={
            state.value: tuple(float(p) for p in counts[i] / totals[i])
            for i, state in enumerate(states)
            if totals[i] > 0
        },
        initial_distribution=tuple(float(v / total) for v in occupancy),
        terminal_states=tuple(s for i, s in enumerate(states) if totals[i] == 0),
        unobserved_regimes=tuple(g for g in Regime if g not in present),
        n_transitions=int(counts.sum()),
    )


# ---------------------------------------------------------------------------
# Result types: two of them, because there are two outcomes
# ---------------------------------------------------------------------------


class FanBand(FrozenModel):
    """One percentile curve of the fan.

    NOT a simulated path: each value is the cross-sectional percentile of all
    simulations at that step, so the band is an envelope no single run
    traced. `MonteCarloDistribution.sample_curves` holds real paths for a
    caller that needs one.
    """

    percentile: float = Field(gt=0.0, lt=100.0)
    equity: tuple[float, ...]


class EquityFan(FrozenModel):
    """Per-step percentile envelopes over the simulated equity curves."""

    steps: tuple[int, ...]
    """Trade indices the bands are evaluated at; step 0 is the starting
    equity. Subsampled to at most `FAN_MAX_STEPS` on long paths."""

    bands: tuple[FanBand, ...]
    n_simulations: int = Field(ge=1)
    n_trades: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> "EquityFan":
        if not self.bands:
            raise ValueError("a fan with no band is not a fan")
        if any(len(band.equity) != len(self.steps) for band in self.bands):
            raise ValueError("every band must align to steps")
        return self


class DepthBucket(FrozenModel):
    """One drawdown-histogram bin. Counts of SIMULATIONS, not of trades."""

    lower: float | None
    upper: float | None
    count: int = Field(ge=0)

    @property
    def label(self) -> str:
        low = "-inf" if self.lower is None else f"{self.lower:g}"
        high = "+inf" if self.upper is None else f"{self.upper:g}"
        return f"[{low},{high})"


class MonteCarloReport(FrozenModel):
    """What every MC result carries, distribution or refusal.

    Deliberately holds no percentile, probability or fan. A caller typed to
    this base cannot read a distributional figure without narrowing to
    `MonteCarloDistribution`, and that narrowing is exactly the point where
    `n_trades >= 1` has been established.
    """

    method: MonteCarloMethod
    split_label: SplitPhase | None
    n_trades: int = Field(ge=0)
    plan: MonteCarloPlan
    limitation: str
    """Section 11's stated limitation. Required, validated verbatim, so there
    is no construction path to a result without it."""

    data_hashes: tuple[str, ...] = ()
    config_hashes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_limitation(self) -> "MonteCarloReport":
        if self.limitation != STATED_LIMITATION:
            raise ValueError(
                "limitation must be STATED_LIMITATION verbatim (ARCHITECTURE.md 11); a "
                "weakened or paraphrased version is how it stops being surfaced"
            )
        if PROVENANCE_NOTE not in self.notes:
            raise ValueError("every MC result carries PROVENANCE_NOTE in notes")
        return self

    def describe(self) -> tuple[str, ...]:
        return (
            f"LIMITATION: {self.limitation}",
            f"method={self.method.value} n_trades={self.n_trades} split={self.split_label}",
            *self.notes,
        )


class MonteCarloRefusal(MonteCarloReport):
    """There is no distribution, and the reason is named.

    `Undefined.NO_TRADES` is the live case. This type has no percentile,
    probability or fan field and `extra="forbid"` means none can be added to
    an instance, so a flat fan at the starting balance cannot be smuggled
    through it.
    """

    undefined: Undefined

    @model_validator(mode="after")
    def _check(self) -> "MonteCarloRefusal":
        if (self.n_trades == 0) != (self.undefined is Undefined.NO_TRADES):
            raise ValueError(
                f"n_trades={self.n_trades} and reason={self.undefined} disagree: an "
                "empty pool is refused as NO_TRADES and a non-empty one is not"
            )
        return self

    def describe(self) -> tuple[str, ...]:
        return (
            f"LIMITATION: {self.limitation}",
            f"method={self.method.value} UNDEFINED[{self.undefined.value}] "
            f"n_trades={self.n_trades}: no percentile, no probability, no fan",
            *self.notes,
        )


class MonteCarloDistribution(MonteCarloReport):
    """The populated result. Cannot exist at n=0.

    `n_trades` is `ge=1` and the validator rejects any undefined
    `MetricValue` or any whose `n` disagrees with `n_trades`. Every
    `MetricValue` here therefore reports the POOL size, never the simulation
    count -- see the module docstring, section 2.
    """

    n_trades: int = Field(ge=1)
    n_simulations: int = Field(ge=1)
    path_length: int = Field(ge=1)
    sample: SampleStatus
    min_sample_n: int = Field(ge=1)
    effective_block_length: int | None = None

    r_dollars_modeled: MetricValue
    mean_risk_dollars_observed: MetricValue
    terminal_equity: dict[str, MetricValue]
    drawdown_percentiles: dict[str, MetricValue]
    p_drawdown_at_least: dict[str, MetricValue]
    p_loss_streak_at_least: dict[str, MetricValue]
    worst_drawdown: MetricValue
    median_drawdown: MetricValue
    mean_drawdown: MetricValue
    median_max_loss_streak: MetricValue
    worst_max_loss_streak: MetricValue
    p_ruin: MetricValue
    mean_path_length: MetricValue
    drawdown_histogram: tuple[DepthBucket, ...]
    fan: EquityFan
    sample_curves: tuple[tuple[float, ...], ...] = ()
    transitions: RegimeTransitions | None = None
    n_truncated_paths: int = Field(ge=0, default=0)

    def metric_values(self) -> tuple[MetricValue, ...]:
        """Every `MetricValue` on this result, for the validator and reports."""
        tables: tuple[Mapping[str, MetricValue], ...] = (
            self.terminal_equity,
            self.drawdown_percentiles,
            self.p_drawdown_at_least,
            self.p_loss_streak_at_least,
        )
        return (
            self.r_dollars_modeled,
            self.mean_risk_dollars_observed,
            self.worst_drawdown,
            self.median_drawdown,
            self.mean_drawdown,
            self.median_max_loss_streak,
            self.worst_max_loss_streak,
            self.p_ruin,
            self.mean_path_length,
            *(v for table in tables for v in table.values()),
        )

    @model_validator(mode="after")
    def _check(self) -> "MonteCarloDistribution":
        if not self.terminal_equity:
            raise ValueError("a distribution with no terminal percentile is a refusal")
        for value in self.metric_values():
            if not value.is_defined:
                raise ValueError(
                    f"{value.name} is undefined on a populated distribution; absence is "
                    "MonteCarloRefusal's job, not a field on this type"
                )
            if value.n != self.n_trades:
                raise ValueError(
                    f"{value.name} carries n={value.n} against n_trades={self.n_trades}: "
                    "n is the resample POOL size everywhere in this module, never the "
                    "simulation count"
                )
        if self.fan.n_trades != self.n_trades or self.fan.n_simulations != self.n_simulations:
            raise ValueError("the fan must report the same n as the distribution")
        if (self.transitions is None) != (self.method is not MonteCarloMethod.REGIME_AWARE):
            raise ValueError("the regime-aware method reports a chain; the others do not")
        return self

    def describe(self) -> tuple[str, ...]:
        head = (
            f"LIMITATION: {self.limitation}",
            f"method={self.method.value} n_trades={self.n_trades} "
            f"sims={self.n_simulations} path={self.path_length} "
            f"sample={self.sample.value} split={self.split_label}",
        )
        body = [str(v) for v in self.terminal_equity.values()]
        body += [str(v) for v in self.p_drawdown_at_least.values()]
        body += [str(v) for v in self.p_loss_streak_at_least.values()]
        body += [str(self.worst_drawdown), str(self.p_ruin)]
        chain = self.transitions.describe() if self.transitions is not None else ()
        return (*head, *body, *chain, *self.notes)


MonteCarloResult = MonteCarloRefusal | MonteCarloDistribution


# ---------------------------------------------------------------------------
# Draws: vectorized over simulations, the short loop runs over trades
# ---------------------------------------------------------------------------


def _chunk_sims(path_length: int, n_simulations: int) -> int:
    """Simulations per chunk. A pure function of its arguments and
    `SIM_ELEMENT_BUDGET`, so the draw sequence depends on the inputs alone
    and chunking cannot move a number."""
    return max(1, min(n_simulations, SIM_ELEMENT_BUDGET // max(1, path_length)))


def _iid_indices(
    generator: np.random.Generator, n_pool: int, sims: int, length: int
) -> np.ndarray:
    return generator.integers(0, n_pool, size=(sims, length), dtype=np.int64)


def _block_indices(
    generator: np.random.Generator, n_pool: int, sims: int, length: int, block: int
) -> np.ndarray:
    """Circular moving-block bootstrap.

    Starts are uniform over the whole pool and blocks wrap, so every outcome
    is equally likely to be drawn. The truncating variant under-samples the
    tail of the sequence, which is where the worst clusters often sit.
    """
    n_blocks = -(-length // block)
    starts = generator.integers(0, n_pool, size=(sims, n_blocks), dtype=np.int64)
    offsets = np.arange(block, dtype=np.int64)
    drawn = (starts[:, :, None] + offsets[None, None, :]) % n_pool
    return drawn.reshape(sims, n_blocks * block)[:, :length]


def _regime_indices(
    generator: np.random.Generator,
    transitions: RegimeTransitions,
    regimes: Sequence[Regime],
    sims: int,
    length: int,
) -> np.ndarray:
    """Sequence states from the chain, then draw within the state's stratum.

    Returns a (sims, length) index matrix where -1 marks a step past the end
    of a truncated path: the chain entered a state with no outgoing row, so
    the path ended rather than being continued by invention.
    """
    states = transitions.states
    width = len(states)

    # Dense cumulative rows, for vectorized sampling only. A terminal state's
    # row is left all-zero and is never advanced from (`alive` masks it); the
    # REPORTED chain has no such row, which is the guarantee that matters.
    cumulative = np.zeros((width, width), dtype=np.float64)
    terminal = np.ones(width, dtype=bool)
    for i, state in enumerate(states):
        row = transitions.rows.get(state.value)
        if row is not None:
            cumulative[i] = np.cumsum(np.asarray(row, dtype=np.float64))
            terminal[i] = False

    # Pool indices grouped by state and flattened, so a draw is one gather.
    groups = [[j for j, g in enumerate(regimes) if g is state] for state in states]
    flat = np.array([j for group in groups for j in group], dtype=np.int64)
    sizes = np.array([len(group) for group in groups], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)

    initial = np.cumsum(np.asarray(transitions.initial_distribution, dtype=np.float64))
    state = np.minimum(np.searchsorted(initial, generator.random(sims), side="right"), width - 1)
    u_step = generator.random((sims, length))
    v_draw = generator.random((sims, length))

    indices = np.full((sims, length), -1, dtype=np.int64)
    alive = np.ones(sims, dtype=bool)
    for step in range(length):
        position = np.minimum((v_draw[:, step] * sizes[state]).astype(np.int64), sizes[state] - 1)
        indices[:, step] = np.where(alive, flat[offsets[state] + position], -1)
        ends = terminal[state] & alive
        nxt = np.minimum((cumulative[state] < u_step[:, step, None]).sum(axis=1), width - 1)
        state = np.where(alive & ~ends, nxt, state)
        alive = alive & ~ends
        if not alive.any():
            break
    return indices


# ---------------------------------------------------------------------------
# Path arithmetic
# ---------------------------------------------------------------------------


def _equity_paths(r: np.ndarray, plan: MonteCarloPlan) -> tuple[np.ndarray, np.ndarray]:
    """Equity curves of shape (sims, length + 1) and a ruin mask.

    An account that reaches zero stays at zero: a dead account does not go on
    trading, and letting the arithmetic run through a negative balance would
    report a recovery that could not happen.
    """
    sims, length = r.shape
    equity_0 = plan.starting_equity
    curves = np.empty((sims, length + 1), dtype=np.float64)
    curves[:, 0] = equity_0

    if plan.compound:
        # Overflow is DETECTED by the caller and reported as
        # `Undefined.NOT_REPRESENTABLE`, so a warning here would be noise
        # about a case that is already refused rather than published.
        with np.errstate(over="ignore", invalid="ignore"):
            curves[:, 1:] = equity_0 * np.cumprod(1.0 + plan.risk_fraction * r, axis=1)
    else:
        curves[:, 1:] = equity_0 + plan.r_dollars * np.cumsum(r, axis=1)

    # Read off the CURVE, not off the per-trade factor, in both modes. Under
    # compounding a factor of 1 + f*r is negative only on a loss worse than
    # -1/f R, and testing that alone would miss the other way a compounding
    # account reaches zero: a long enough run of severe losses underflows the
    # product to exactly 0.0 with every factor still positive. Reporting a
    # -100% drawdown beside `p_ruin = 0` would be incoherent, and a balance
    # of 0.0 is the account being gone at the precision the system can
    # represent. The first breach is the first step at or below zero either
    # way, because a negative factor drives that step's curve negative before
    # any later factor can multiply it back positive.
    breach = curves[:, 1:] <= 0.0
    ruined = breach.any(axis=1)
    if ruined.any():
        after = np.arange(length)[None, :] >= np.argmax(breach, axis=1)[:, None]
        curves[:, 1:] = np.where(ruined[:, None] & after, 0.0, curves[:, 1:])
    return curves, ruined


def _max_drawdowns(curves: np.ndarray) -> np.ndarray:
    """Deepest peak-to-trough depth per simulation, as a NEGATIVE fraction.

    Matches `EquityPoint.drawdown_pct` and `analytics.DrawdownProfile`, so a
    drawdown never changes sign between two parts of this system. The running
    peak starts at the starting equity and is therefore always positive.
    """
    return (curves / np.maximum.accumulate(curves, axis=1) - 1.0).min(axis=1)


def _max_loss_streaks(r: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Longest run of losses per simulation.

    A loss is `r < 0`, the sign of `TradeRecord.is_loss` (net P&L). A scratch
    breaks the run, as in `analytics.streaks`. A step past the end of a
    truncated path is neither, so it ends a run without extending it.
    """
    sims, length = r.shape
    losing = (r < 0.0) & valid
    best = np.zeros(sims, dtype=np.int64)
    run = np.zeros(sims, dtype=np.int64)
    for step in range(length):
        run = np.where(losing[:, step], run + 1, 0)
        best = np.maximum(best, run)
    return best


def _fan_steps(path_length: int) -> np.ndarray:
    if path_length + 1 <= FAN_MAX_STEPS:
        return np.arange(path_length + 1, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, path_length, FAN_MAX_STEPS)).astype(np.int64))


def _histogram(depths: np.ndarray) -> tuple[DepthBucket, ...]:
    edges = np.asarray(DRAWDOWN_HISTOGRAM_EDGES, dtype=np.float64)
    counts = np.bincount(np.searchsorted(edges, depths, side="right"), minlength=len(edges) + 1)
    return tuple(
        DepthBucket(
            lower=None if i == 0 else float(edges[i - 1]),
            upper=None if i == len(edges) else float(edges[i]),
            count=int(count),
        )
        for i, count in enumerate(counts)
    )


def _percentile_table(
    name: str, values: np.ndarray, quantiles: Sequence[float], *, unit: str, n: int
) -> dict[str, MetricValue]:
    return {
        f"p{q:g}": MetricValue.measured(
            f"{name}_p{q:g}", float(np.percentile(values, q)), n=n, unit=unit
        )
        for q in quantiles
    }


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

_NOTE_NO_TRADES = (
    "n=0: resampling an empty trade list cannot produce a distribution of anything -- "
    "there is no empirical R distribution to preserve, no autocorrelation to block and "
    "no regime sequence to estimate a chain from. This is the expected shape of a run "
    "at the committed config, which takes no trades; it is not a failure. A flat fan at "
    "the starting balance would be arithmetically true and would read as a picture of a "
    "strategy that never traded, so this type cannot carry one."
)


def _refusal(
    reason: Undefined,
    *,
    plan: MonteCarloPlan,
    split_label: SplitPhase | None,
    n_trades: int,
    notes: Sequence[str],
    data_hashes: tuple[str, ...] = (),
    config_hashes: tuple[str, ...] = (),
) -> MonteCarloRefusal:
    return MonteCarloRefusal(
        method=plan.method,
        split_label=split_label,
        n_trades=n_trades,
        plan=plan,
        limitation=STATED_LIMITATION,
        data_hashes=data_hashes,
        config_hashes=config_hashes,
        notes=(PROVENANCE_NOTE, *notes),
        undefined=reason,
    )


def run_monte_carlo(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    plan: MonteCarloPlan | None = None,
    method: MonteCarloMethod | None = None,
) -> MonteCarloResult:
    """Resample a realized trade sequence. Returns a refusal or a distribution.

    Raises `MixedSplitError` on trades from more than one `SplitPhase`
    (section 0.8, via `analytics.assert_single_phase`): an MC fan over merged
    phases is an unlabelled performance claim.
    """
    resolved = plan if plan is not None else MonteCarloPlan.default()
    if method is not None and method is not resolved.method:
        resolved = resolved.replace(method=method)

    pool = tuple(trades)
    assert_single_phase(pool, split_label)
    n_trades = len(pool)
    notes: list[str] = []

    if n_trades == 0:
        return _refusal(
            Undefined.NO_TRADES,
            plan=resolved,
            split_label=split_label,
            n_trades=0,
            notes=(_NOTE_NO_TRADES,),
        )

    ordered = chronological(pool)
    r_pool = np.array([t.r_multiple for t in ordered], dtype=np.float64)
    regimes = [t.regime for t in ordered]
    data_hashes = tuple(sorted({t.data_hash for t in ordered}))
    config_hashes = tuple(sorted({t.config_hash for t in ordered}))

    path_length = resolved.trades_per_simulation or n_trades
    status = sample_status(n_trades, resolved.min_sample_n)
    if status is SampleStatus.INSUFFICIENT:
        notes.append(
            f"n={n_trades} is below min_sample_n={resolved.min_sample_n}: a bootstrap over "
            f"{n_trades} trade{'' if n_trades == 1 else 's'} resamples the same "
            f"{n_trades}, so the percentile spread below is an artifact of that pool and "
            "not knowledge about the future. More simulations make it smoother, not "
            "better founded."
        )
    if path_length != n_trades:
        notes.append(
            f"path_length={path_length} against a pool of {n_trades}: the simulated "
            "sequence is longer or shorter than anything observed. That extrapolates the "
            "COUNT of trades only; it adds no outcome the pool does not contain."
        )
    if float(r_pool.std()) == 0.0:
        notes.append(
            "every trade in the pool realized the same R, so the terminal distribution is "
            "a point mass. That is a measurement, not a failure."
        )

    # One draw function per method, bound where that method's inputs are
    # validated, so the simulation loop needs no re-narrowing and no assert.
    generator = rng(resolved.seed, "monte_carlo", resolved.method.value)
    transitions: RegimeTransitions | None = None
    effective_block: int | None = None

    if resolved.method is MonteCarloMethod.REGIME_AWARE:
        transitions = estimate_transitions(regimes)
        chain = transitions

        def draw(sims: int) -> np.ndarray:
            return _regime_indices(generator, chain, regimes, sims, path_length)

        notes.append(
            f"regime-aware: {len(transitions.states)} state(s) observed, "
            f"{transitions.n_transitions} transition(s) estimated, "
            f"terminal (no row, paths end there): "
            f"{', '.join(g.value for g in transitions.terminal_states) or 'none'}; "
            f"never observed, given no state: "
            f"{', '.join(g.value for g in transitions.unobserved_regimes) or 'none'}. "
            "Nothing was smoothed into existence: section 11's limitation is exactly "
            "that resampling cannot produce an absent regime."
        )
    elif resolved.method is MonteCarloMethod.BLOCK:
        effective_block = min(resolved.block_length, n_trades)
        block = effective_block

        def draw(sims: int) -> np.ndarray:
            return _block_indices(generator, n_trades, sims, path_length, block)

        clamped = (
            f" (clamped from {resolved.block_length} to the pool size)"
            if effective_block != resolved.block_length
            else ""
        )
        notes.append(
            f"block bootstrap, circular, L={effective_block}{clamped}. Wrapping joins the "
            "last observed outcome to the first, a transition that never occurred."
        )
        if effective_block >= n_trades:
            notes.append(
                f"L={effective_block} covers the whole pool, so every simulation is a "
                f"rotation of the one observed sequence: the spread below is a "
                f"distribution over {n_trades} rotations, not over resampled paths."
            )
    else:

        def draw(sims: int) -> np.ndarray:
            return _iid_indices(generator, n_trades, sims, path_length)

    chunk = _chunk_sims(path_length, resolved.n_simulations)
    steps = _fan_steps(path_length)
    terminal_p: list[np.ndarray] = []
    depth_p: list[np.ndarray] = []
    streak_p: list[np.ndarray] = []
    ruin_p: list[np.ndarray] = []
    length_p: list[np.ndarray] = []
    fan_p: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    n_kept = 0
    done = 0

    while done < resolved.n_simulations:
        sims = min(chunk, resolved.n_simulations - done)
        indices = draw(sims)

        # -1 marks a truncated step. It gathers the last pool element (NumPy
        # wraps) and is then masked to a zero R, so the step is flat.
        valid = indices >= 0
        r_draw = np.where(valid, r_pool[indices], 0.0)
        curves, ruined = _equity_paths(r_draw, resolved)
        if not np.isfinite(curves).all():
            return _refusal(
                Undefined.NOT_REPRESENTABLE,
                plan=resolved,
                split_label=split_label,
                n_trades=n_trades,
                data_hashes=data_hashes,
                config_hashes=config_hashes,
                notes=(
                    f"compounding this pool over {path_length} trades overflows a float, "
                    "so the terminal distribution is not representable. Reported as "
                    "undefined rather than as an infinity or a clipped number that would "
                    "look like a measurement.",
                    *notes,
                ),
            )

        terminal_p.append(curves[:, -1])
        depth_p.append(_max_drawdowns(curves))
        streak_p.append(_max_loss_streaks(r_draw, valid))
        ruin_p.append(ruined)
        length_p.append(valid.sum(axis=1))
        fan_p.append(curves[:, steps])
        if n_kept < resolved.n_curves_to_store:
            take = min(resolved.n_curves_to_store - n_kept, sims)
            kept.append(curves[:take, steps])
            n_kept += take
        done += sims

    terminal = np.concatenate(terminal_p)
    depths = np.concatenate(depth_p)
    runs = np.concatenate(streak_p)
    ruin = np.concatenate(ruin_p)
    lengths = np.concatenate(length_p)
    fan_columns = np.concatenate(fan_p, axis=0)

    truncated = int((lengths < path_length).sum())
    if truncated:
        notes.append(
            f"{truncated} of {resolved.n_simulations} path(s) ended early in a terminal "
            f"regime state (mean realized length {float(lengths.mean()):.4g} against "
            f"{path_length}), so the terminal distribution is a mixture over path lengths."
        )

    def measured(name: str, value: float, unit: str) -> MetricValue:
        return MetricValue.measured(name, float(value), n=n_trades, unit=unit)

    return MonteCarloDistribution(
        method=resolved.method,
        split_label=split_label,
        n_trades=n_trades,
        plan=resolved,
        limitation=STATED_LIMITATION,
        data_hashes=data_hashes,
        config_hashes=config_hashes,
        notes=(PROVENANCE_NOTE, *notes),
        n_simulations=resolved.n_simulations,
        path_length=path_length,
        sample=status,
        min_sample_n=resolved.min_sample_n,
        effective_block_length=effective_block,
        r_dollars_modeled=measured("r_dollars_modeled", resolved.r_dollars, "USD"),
        mean_risk_dollars_observed=measured(
            "mean_risk_dollars_observed", float(np.mean([t.risk_dollars for t in ordered])), "USD"
        ),
        terminal_equity=_percentile_table(
            "terminal_equity", terminal, resolved.percentiles, unit="USD", n=n_trades
        ),
        drawdown_percentiles=_percentile_table(
            "max_drawdown", depths, resolved.percentiles, unit="fraction", n=n_trades
        ),
        p_drawdown_at_least={
            f"{d:g}": measured(f"p_drawdown_at_least_{d:g}", np.mean(depths <= -d), "probability")
            for d in resolved.drawdown_thresholds
        },
        p_loss_streak_at_least={
            f"{k:d}": measured(f"p_loss_streak_at_least_{k:d}", np.mean(runs >= k), "probability")
            for k in resolved.streak_thresholds
        },
        worst_drawdown=measured("worst_max_drawdown", depths.min(), "fraction"),
        median_drawdown=measured("median_max_drawdown", np.median(depths), "fraction"),
        mean_drawdown=measured("mean_max_drawdown", depths.mean(), "fraction"),
        median_max_loss_streak=measured("median_max_loss_streak", np.median(runs), "trades"),
        worst_max_loss_streak=measured("worst_max_loss_streak", runs.max(), "trades"),
        p_ruin=measured("p_ruin", ruin.mean(), "probability"),
        mean_path_length=measured("mean_path_length", lengths.mean(), "trades"),
        drawdown_histogram=_histogram(depths),
        fan=EquityFan(
            steps=tuple(int(s) for s in steps),
            bands=tuple(
                FanBand(
                    percentile=float(q),
                    equity=tuple(float(v) for v in np.percentile(fan_columns, q, axis=0)),
                )
                for q in resolved.percentiles
            ),
            n_simulations=resolved.n_simulations,
            n_trades=n_trades,
        ),
        sample_curves=tuple(
            tuple(float(v) for v in curve) for part in kept for curve in part
        ),
        transitions=transitions,
        n_truncated_paths=truncated,
    )


def compare_methods(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    plan: MonteCarloPlan | None = None,
    methods: Sequence[MonteCarloMethod] = tuple(MonteCarloMethod),
) -> dict[MonteCarloMethod, MonteCarloResult]:
    """Run several methods over the same pool.

    The comparison is the point of having three: iid destroys order, so a
    block or regime-aware result that is materially worse is measuring the
    clustering iid threw away. No method is named best and none is aggregated
    with another -- they answer different questions about one sample, and at
    n=0 all three refuse.
    """
    pool = tuple(trades)
    return {
        m: run_monte_carlo(pool, split_label=split_label, plan=plan, method=m) for m in methods
    }
