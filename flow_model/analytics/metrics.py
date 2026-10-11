"""Performance metrics: pure functions over `list[TradeRecord]` + the equity series.

ARCHITECTURE.md section 12 names what this module computes. Sections 0.2 and
0.8 fix two rules that shape every signature here, and the researcher brief
adds a third.

## 1. The n = 0 contract, which is the live case

The system as committed takes **zero trades**: the highest Flow Score measured
over 2500 consecutive bars with every feed present is 65.23 and the lowest
setup threshold is 70.0, so `score_below_threshold` is the terminal gate
(commit f952d3e). Phase 8 sweeps weights and thresholds against TRAINING
folds; nothing here may widen a gate to produce a sample.

So an empty trade set is not an edge case, it is *the* case, and a metric over
it must say "undefined, n=0" rather than return a number that reads as a
measurement. The rule applied throughout:

* a **count** or a **sum** over an empty set is its identity element -- zero
  trades, zero dollars realized. That is a measurement, and the flat equity
  curve says the same thing independently.
* a **mean**, a **rate** or a **ratio** over an empty set is UNDEFINED. There
  is no such thing as the average of nothing, and `0.0` in a win-rate column
  reads as "we measured, and it lost".
* a profit factor with winners and no losers is UNDEFINED, never infinity.
* Sharpe below a declared minimum number of return observations is UNDEFINED
  (`MIN_RETURN_OBSERVATIONS`), never a ratio computed from two points.
* max drawdown over a flat curve **is** 0.0. Zero depth was observed on every
  point; that is data, not absence of data.

`MetricValue` is the single representation of all of this: a name, a value or
a reason it has none, the N behind it, and a unit. There is no float sentinel
anywhere in this module, because `-1.0` and `0.0` eventually get plotted as
values.

## 2. Split labels (section 0.8, section 12)

"No performance claim without a split label. Analytics refuse to aggregate IS
and OOS into one unlabeled number." Every public function therefore takes a
`split_label`, and the refusal is structural rather than documented: handed
trades carrying more than one `SplitPhase`, these functions RAISE
(`MixedSplitError`). They do not pick one, do not merge, and do not consult a
config flag -- a knob that can switch the refusal off is not a refusal.
`breakdowns.breakdown_by_split_phase` is the one function in the package that
may receive several phases, and it reports them as separate cells with no
combined row.

Three functions take no `split_label`, and the omission is deliberate rather
than an oversight: `simple_returns`, `drawdown_episodes` and
`infer_periods_per_year` describe the SHAPE of an equity series (its returns,
its underwater excursions, its sampling density). None of them is a
performance claim, and `EquityPoint` carries no phase for them to check.
Every function that reports performance takes one.

Note that `AnalyticsConfig.allow_mixed_split_aggregation` (default False) is
deliberately NOT read here. Its default states the policy; honouring a True
would reopen exactly the hole section 0.8 closes.

## 3. R is read, never re-derived (section 0.2)

`R = |entry - stop| * point_value * size`, fixed at entry; realized R is
`pnl_after_costs / risk_dollars`. `TradeRecord.r_multiple` already computes
it, from a `pnl` the record validates against the price path and the itemized
costs. This module reads that field. A second derivation from prices would be
a second definition of R, and it would drift.

For the same reason expectancy and profit factor are net of costs with no
extra arithmetic: `TradeRecord.pnl` is already net, and `is_win` / `is_loss`
classify on the net number.

## 4. Cost drag is broken down by exit reason, never quoted as one figure

Phase 7a measured one NQ round turn, size 2, 10-point stop (R = $400), vol
multiple 1.25: **$47.74 = 0.1194R through the stop** and **$21.49 = 0.0537R
through the target**. A target is a resting limit (no spread, no slippage); a
stop is a market order that also pays `stop_order_extra_slippage_ticks`. So
drag is not one constant, it depends on the exit mix, and
`cost_drag_by_exit_reason` reports per-reason means with the mix beside them.
`breakeven_win_rate` derives the hurdle from the measured average win and
average loss rather than from any stored figure.

## 5. Annualizing Sharpe and Sortino

Section 12 names "Sharpe", which by convention means annualized, so the
equity-curve entry points annualize by `sqrt(periods_per_year)` with the
factor MEASURED from the series' own sampling density
(`infer_periods_per_year`). Two caveats are declared rather than buried:

* the square-root rule assumes independent per-period returns. Intraday
  returns are not independent, so an annualized intraday Sharpe is a
  convention, not an estimate of a yearly ratio. `sharpe_from_returns` with
  `periods_per_year=None` gives the per-period number, whose unit says so.
* a flat curve has no dispersion, so both ratios are refused rather than
  reported (see `Undefined.ZERO_DISPERSION`). Annualizing a constant
  shortfall against the risk-free rate would print a large negative number
  that measures the hurdle, not the strategy.

## 6. Targets

`analytics` is the one package (with `validation`) allowed to read
`config.research_targets`, and only to report whether a measured value met a
target. `compare_to_research_targets` puts the measured value first, labels
the result a comparison, and returns `met=None` when the measurement is
undefined. Nothing here selects a parameter, and no target is an objective.

## Declared constants

Every constant in this module is either a config field or declared below with
its provenance. There are no silent defaults.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime

import numpy as np
from pydantic import Field, model_validator

from flow_model.config.schema import AnalyticsConfig, ResearchTargets
from flow_model.core.contracts import EquityPoint, TradeRecord
from flow_model.core.enums import ExitReason, SplitPhase, StrEnum
from flow_model.core.model import FrozenModel

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Fewest return observations from which an annualized Sharpe/Sortino will be
#: reported. JUDGEMENT CALL -- it is not in ARCHITECTURE.md and not a config
#: field. Below roughly this many observations the standard error of the
#: statistic is the same size as the statistic, so the number carries no
#: information while looking exactly like one that does. It is a named
#: constant rather than an inline `2` so that it can be argued with, and it is
#: deliberately NOT `AnalyticsConfig.min_sample_n` (which counts TRADES in a
#: breakdown cell, a different quantity that happens to have a similar value).
MIN_RETURN_OBSERVATIONS: int = 30

#: Julian year. CONVENTION: used for calendar spans (CAGR, trade frequency),
#: where the quantity is wall-clock time and not trading sessions.
#: Session-based annualization uses `AnalyticsConfig.trading_days_per_year`.
SECONDS_PER_CALENDAR_YEAR: float = 365.25 * 24.0 * 3600.0

#: Sample standard deviation. CONVENTION (ddof=1): the trade/return sample is
#: a sample, not the population.
STDEV_DDOF: int = 1

#: Downside deviation divides by the count of ALL observations, not only the
#: ones below target. CONVENTION: that is the standard Sortino denominator;
#: dividing by the downside count only would flatter a curve whose losses are
#: rare but large.
DOWNSIDE_DDOF: int = 0

#: Percentiles reported for the R distribution. PROVENANCE: ARCHITECTURE.md
#: section 12 fixes this set for the dashboard's percentile table.
R_PERCENTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)

#: Interior edges of the R histogram. JUDGEMENT CALL: half-R granularity over
#: the range the three configured setups can produce (-1R stop through +3R
#: target), with open ends so an outlier is counted rather than clipped.
R_HISTOGRAM_EDGES: tuple[float, ...] = (
    -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AnalyticsError(ValueError):
    """Base for refusals raised by the analytics layer."""


class MixedSplitError(AnalyticsError):
    """Trades from more than one `SplitPhase` were handed to one metric.

    Section 0.8. Merging in-sample and out-of-sample trades into a single
    number is the easiest accidental way to produce a flattering result, so
    this is a raise and not a warning.
    """


class SplitLabelMismatchError(AnalyticsError):
    """The declared `split_label` is not the phase the trades carry."""


class NonChronologicalEquityError(AnalyticsError):
    """The equity series runs backwards.

    Sorting it silently would hide an upstream ordering bug while producing a
    plausible drawdown number, so the series is rejected instead.
    """


class UndefinedMetricError(AnalyticsError):
    """`MetricValue.require()` was called on a metric that has no value."""


# ---------------------------------------------------------------------------
# The one representation of "undefined"
# ---------------------------------------------------------------------------


class Undefined(StrEnum):
    """Why a metric has no value.

    These are *definedness* reasons for a measurement. They are not
    `ExitReason` or `WaitReason` values, they never touch a `TradeRecord`, and
    nothing in `signals/` or `backtest/` reads them.
    """

    NO_TRADES = "no_trades"
    """The trade set is empty -- the current state of the system."""

    NO_WINS = "no_wins"
    NO_LOSSES = "no_losses"
    """A profit factor with no losses is undefined, not infinite."""

    INSUFFICIENT_SAMPLE = "insufficient_sample"
    """N below `AnalyticsConfig.min_sample_n`; see `breakdowns`."""

    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    """Fewer return observations than `MIN_RETURN_OBSERVATIONS`."""

    ZERO_DISPERSION = "zero_dispersion"
    """Standard deviation is exactly zero -- e.g. a flat equity curve, where
    the Sharpe ratio is 0/0 rather than 0."""

    NO_DOWNSIDE = "no_downside"
    """No observation fell below the Sortino target, so downside deviation is
    zero and the ratio is undefined rather than infinite."""

    NO_DRAWDOWN_EPISODE = "no_drawdown_episode"
    """The curve never fell below its running peak, so there is no episode to
    average. Max drawdown is still 0.0, which is a measurement."""

    NOT_RECOVERED = "not_recovered"
    """Every drawdown episode is still open at the end of the series."""

    NO_EQUITY_SERIES = "no_equity_series"
    """Fewer than two equity points were supplied."""

    NON_POSITIVE_EQUITY = "non_positive_equity"
    """Equity reached zero or below, so a ratio-of-equity has no meaning."""

    ZERO_ELAPSED_TIME = "zero_elapsed_time"
    """First and last timestamps coincide, so a per-year rate is 0/0."""

    NO_ANNUALIZATION = "no_annualization"
    """The sampling frequency of the equity series could not be measured."""


class SampleStatus(StrEnum):
    """Whether a cell's N justifies reporting a rate at all."""

    EMPTY = "EMPTY"
    INSUFFICIENT = "INSUFFICIENT"
    ADEQUATE = "ADEQUATE"


def sample_status(n: int, min_sample_n: int) -> SampleStatus:
    if n <= 0:
        return SampleStatus.EMPTY
    if n < min_sample_n:
        return SampleStatus.INSUFFICIENT
    return SampleStatus.ADEQUATE


class MetricValue(FrozenModel):
    """One number, or an explicit statement that there is no number.

    Exactly one of `value` and `undefined` is set. `n` is always present, so a
    reader can judge the number without looking it up elsewhere.
    """

    name: str
    value: float | None = None
    n: int = Field(ge=0, default=0)
    undefined: Undefined | None = None
    unit: str = ""
    note: str = ""

    @classmethod
    def measured(
        cls, name: str, value: float, *, n: int, unit: str = "", note: str = ""
    ) -> "MetricValue":
        return cls(name=name, value=float(value), n=n, unit=unit, note=note)

    @classmethod
    def undefined_because(
        cls, name: str, reason: Undefined, *, n: int = 0, unit: str = "", note: str = ""
    ) -> "MetricValue":
        return cls(name=name, undefined=reason, n=n, unit=unit, note=note)

    @property
    def is_defined(self) -> bool:
        return self.value is not None

    def require(self) -> float:
        """The value, or a raise. For callers that cannot handle absence."""
        if self.value is None:
            raise UndefinedMetricError(
                f"{self.name} is undefined ({self.undefined}, n={self.n})"
            )
        return self.value

    def or_none(self) -> float | None:
        return self.value

    def __str__(self) -> str:  # pragma: no cover - presentation only
        if self.value is None:
            return f"{self.name}=undefined[{self.undefined}] (n={self.n})"
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.name}={self.value:.6g}{unit} (n={self.n})"

    @model_validator(mode="after")
    def _check(self) -> "MetricValue":
        if (self.value is None) == (self.undefined is None):
            raise ValueError(
                "a MetricValue carries either a value or a reason it has none, "
                f"never both and never neither: {self.name!r}"
            )
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(
                f"{self.name} is {self.value}; a non-finite metric must be reported "
                "as undefined with a reason, not as a float"
            )
        return self


# ---------------------------------------------------------------------------
# Policy (the config this module is allowed to read)
# ---------------------------------------------------------------------------


class MetricPolicy(FrozenModel):
    """The configured knobs these functions obey.

    Built from `config.analytics` so that `min_sample_n` and the Flow Score
    buckets live in exactly one place (`config/schema.py`). `research_targets`
    is NOT part of the policy: a target may never reach a computation, only a
    comparison (see `compare_to_research_targets`).
    """

    min_sample_n: int = Field(ge=1)
    risk_free_rate_annual: float = Field(ge=0.0)
    trading_days_per_year: int = Field(gt=0)
    flow_score_buckets: tuple[float, ...]
    min_return_observations: int = Field(ge=2, default=MIN_RETURN_OBSERVATIONS)

    @classmethod
    def from_analytics(cls, analytics: AnalyticsConfig) -> "MetricPolicy":
        return cls(
            min_sample_n=analytics.min_sample_n,
            risk_free_rate_annual=analytics.risk_free_rate_annual,
            trading_days_per_year=analytics.trading_days_per_year,
            flow_score_buckets=analytics.flow_score_buckets,
        )

    @classmethod
    def from_config(cls, config: object) -> "MetricPolicy":
        """Accepts either the root config or an `AnalyticsConfig`."""
        analytics = getattr(config, "analytics", config)
        if not isinstance(analytics, AnalyticsConfig):
            raise TypeError(
                "MetricPolicy.from_config wants the root config or an AnalyticsConfig, "
                f"got {type(config).__name__}"
            )
        return cls.from_analytics(analytics)

    @classmethod
    def default(cls) -> "MetricPolicy":
        """The schema defaults, constructed in-process (no file, no I/O)."""
        return cls.from_analytics(AnalyticsConfig())


def _policy(policy: MetricPolicy | None) -> MetricPolicy:
    return policy if policy is not None else MetricPolicy.default()


# ---------------------------------------------------------------------------
# The split-phase refusal
# ---------------------------------------------------------------------------


def phases_present(trades: Iterable[TradeRecord]) -> tuple[SplitPhase | None, ...]:
    """Distinct `split_label`s in the set, in first-seen order.

    `None` (an unlabelled trade) counts as its own phase: an unlabelled number
    mixed into a labelled one is exactly the unlabelled claim section 0.8
    forbids.
    """
    seen: list[SplitPhase | None] = []
    for trade in trades:
        if trade.split_label not in seen:
            seen.append(trade.split_label)
    return tuple(seen)


def assert_single_phase(
    trades: Sequence[TradeRecord], split_label: SplitPhase | None
) -> None:
    """Raise unless every trade belongs to the declared phase.

    Two separate refusals, because they are two different mistakes:
    `MixedSplitError` means the caller merged phases; `SplitLabelMismatchError`
    means the caller mislabelled a single-phase set.
    """
    present = phases_present(trades)
    if len(present) > 1:
        raise MixedSplitError(
            "refusing to compute one number from several split phases: "
            f"{[str(p) for p in present]}. Report each phase separately "
            "(breakdowns.breakdown_by_split_phase) -- section 0.8."
        )
    if present and present[0] != split_label:
        raise SplitLabelMismatchError(
            f"declared split_label={split_label} but the trades carry "
            f"{present[0]}. A performance number must be attributable."
        )


def _checked(
    trades: Iterable[TradeRecord], split_label: SplitPhase | None
) -> tuple[TradeRecord, ...]:
    materialized = tuple(trades)
    assert_single_phase(materialized, split_label)
    return materialized


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _stdev(values: Sequence[float], ddof: int = STDEV_DDOF) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=ddof))


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _r_values(trades: Sequence[TradeRecord]) -> list[float]:
    """Realized R, READ from the record (section 0.2). Never re-derived."""
    return [t.r_multiple for t in trades]


def _chronological(trades: Sequence[TradeRecord]) -> list[TradeRecord]:
    """Trades in the order their outcomes became known.

    A trade list is a set, so its argument order is not meaningful; streaks
    are. Sorted on exit (when the win or loss was realized), with entry and id
    as deterministic tie-breaks.
    """
    return sorted(trades, key=lambda t: (t.exit_ts, t.entry_ts, t.trade_id))


# ---------------------------------------------------------------------------
# Counts and sums (defined over the empty set)
# ---------------------------------------------------------------------------


def outcome_counts(trades: Sequence[TradeRecord]) -> tuple[int, int, int]:
    """(wins, losses, scratches), classified on P&L NET of costs."""
    wins = sum(1 for t in trades if t.is_win)
    losses = sum(1 for t in trades if t.is_loss)
    return wins, losses, len(trades) - wins - losses


def trade_count(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """How many trades. Zero is a measurement, and currently the right one."""
    t = _checked(trades, split_label)
    return MetricValue.measured("trade_count", float(len(t)), n=len(t), unit="trades")


def total_pnl(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Net dollars. A sum over no trades is 0.0 -- no money changed hands, and
    the flat equity curve says so independently."""
    t = _checked(trades, split_label)
    return MetricValue.measured(
        "total_pnl", sum(x.pnl for x in t), n=len(t), unit="USD"
    )


def gross_profit(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    return MetricValue.measured(
        "gross_profit", sum(x.pnl for x in t if x.is_win), n=len(t), unit="USD"
    )


def gross_loss(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Sum of losing P&L, reported NEGATIVE as it is experienced."""
    t = _checked(trades, split_label)
    return MetricValue.measured(
        "gross_loss", sum(x.pnl for x in t if x.is_loss), n=len(t), unit="USD"
    )


def total_costs(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    return MetricValue.measured(
        "total_costs", sum(x.costs.total for x in t), n=len(t), unit="USD"
    )


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def _rate(name: str, count: int, n: int) -> MetricValue:
    if n == 0:
        return MetricValue.undefined_because(
            name, Undefined.NO_TRADES, n=0, unit="fraction"
        )
    return MetricValue.measured(name, count / n, n=n, unit="fraction")


def win_rate(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Winners / ALL trades.

    The denominator is every trade, scratches included, so nothing is
    quietly excluded to lift the rate; `scratch_rate` reports the excluded
    mass that an "excluding scratches" convention would hide. Classification
    is on net P&L, so a trade that reached its target and gave the gain back
    in costs is a loss here.

    Undefined at n=0. Not 0.0: that reads as "we measured, and it lost".
    """
    t = _checked(trades, split_label)
    wins, _, _ = outcome_counts(t)
    return _rate("win_rate", wins, len(t))


def loss_rate(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    _, losses, _ = outcome_counts(t)
    return _rate("loss_rate", losses, len(t))


def scratch_rate(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Exactly-zero-P&L trades as a fraction. Reported so that `win_rate`'s
    denominator is auditable rather than assumed."""
    t = _checked(trades, split_label)
    _, _, scratches = outcome_counts(t)
    return _rate("scratch_rate", scratches, len(t))


# ---------------------------------------------------------------------------
# R statistics and expectancy
# ---------------------------------------------------------------------------


def mean_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because("mean_r", Undefined.NO_TRADES, unit="R")
    return MetricValue.measured("mean_r", _mean(_r_values(t)), n=len(t), unit="R")


def median_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because("median_r", Undefined.NO_TRADES, unit="R")
    return MetricValue.measured("median_r", _median(_r_values(t)), n=len(t), unit="R")


def expectancy_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Expected R per trade, net of costs.

    This is arithmetically IDENTICAL to `mean_r`, and that is stated rather
    than disguised: with realized R read from the record, the textbook
    `p_win * avg_win - p_loss * |avg_loss|` expands to `sum(R) / n` exactly
    (scratches contribute zero to both). Section 12 names both, so both exist
    and they agree by construction instead of drifting.
    """
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because(
            "expectancy_r", Undefined.NO_TRADES, unit="R"
        )
    return MetricValue.measured("expectancy_r", _mean(_r_values(t)), n=len(t), unit="R")


def expectancy_dollars(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because(
            "expectancy_dollars", Undefined.NO_TRADES, unit="USD"
        )
    return MetricValue.measured(
        "expectancy_dollars", _mean([x.pnl for x in t]), n=len(t), unit="USD"
    )


def stdev_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if len(t) < 2:
        return MetricValue.undefined_because(
            "stdev_r",
            Undefined.NO_TRADES if not t else Undefined.INSUFFICIENT_OBSERVATIONS,
            n=len(t),
            unit="R",
        )
    return MetricValue.measured("stdev_r", _stdev(_r_values(t)), n=len(t), unit="R")


def average_win_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    wins = [x.r_multiple for x in t if x.is_win]
    if not wins:
        return MetricValue.undefined_because(
            "average_win_r",
            Undefined.NO_TRADES if not t else Undefined.NO_WINS,
            n=0,
            unit="R",
        )
    return MetricValue.measured("average_win_r", _mean(wins), n=len(wins), unit="R")


def average_loss_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Mean R of losers, reported NEGATIVE (a loss is not a positive number)."""
    t = _checked(trades, split_label)
    losses = [x.r_multiple for x in t if x.is_loss]
    if not losses:
        return MetricValue.undefined_because(
            "average_loss_r",
            Undefined.NO_TRADES if not t else Undefined.NO_LOSSES,
            n=0,
            unit="R",
        )
    return MetricValue.measured("average_loss_r", _mean(losses), n=len(losses), unit="R")


def win_loss_ratio(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """average win / |average loss|. Undefined without both sides."""
    t = _checked(trades, split_label)
    wins = [x.r_multiple for x in t if x.is_win]
    losses = [x.r_multiple for x in t if x.is_loss]
    if not t:
        return MetricValue.undefined_because("win_loss_ratio", Undefined.NO_TRADES)
    if not wins:
        return MetricValue.undefined_because(
            "win_loss_ratio", Undefined.NO_WINS, n=len(t)
        )
    if not losses:
        return MetricValue.undefined_because(
            "win_loss_ratio", Undefined.NO_LOSSES, n=len(t)
        )
    return MetricValue.measured(
        "win_loss_ratio", _mean(wins) / abs(_mean(losses)), n=len(wins) + len(losses)
    )


def profit_factor(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Gross profit / gross loss, both NET of costs.

    With winners and no losers the quotient is a division by zero. That is
    reported as UNDEFINED (`NO_LOSSES`), never as infinity and never as a
    large float: an "infinite profit factor" is the signature of a sample too
    small to contain a loss.

    With losers and no winners the value is 0.0, which IS a measurement --
    gross profit really was zero.
    """
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because("profit_factor", Undefined.NO_TRADES)
    profit = sum(x.pnl for x in t if x.is_win)
    loss = abs(sum(x.pnl for x in t if x.is_loss))
    if loss == 0.0:
        return MetricValue.undefined_because(
            "profit_factor",
            Undefined.NO_LOSSES,
            n=len(t),
            note="no losing trade in the sample; the ratio has no denominator",
        )
    return MetricValue.measured("profit_factor", profit / loss, n=len(t))


def breakeven_win_rate(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """The win rate at which expectancy is zero, from the MEASURED averages.

    `|avg_loss| / (avg_win + |avg_loss|)`. For a symmetric 1R system the
    costs measured in Phase 7a put this at roughly 54%, not 50% -- which is
    the point of computing it from this sample's own average win and average
    loss rather than quoting a constant. Scratches are in neither average, so
    with scratches present the identity is approximate; `scratch_rate`
    reports how much mass that is.
    """
    t = _checked(trades, split_label)
    wins = [x.r_multiple for x in t if x.is_win]
    losses = [x.r_multiple for x in t if x.is_loss]
    if not t:
        return MetricValue.undefined_because(
            "breakeven_win_rate", Undefined.NO_TRADES, unit="fraction"
        )
    if not wins:
        return MetricValue.undefined_because(
            "breakeven_win_rate", Undefined.NO_WINS, n=len(t), unit="fraction"
        )
    if not losses:
        return MetricValue.undefined_because(
            "breakeven_win_rate", Undefined.NO_LOSSES, n=len(t), unit="fraction"
        )
    avg_win, avg_loss = _mean(wins), abs(_mean(losses))
    return MetricValue.measured(
        "breakeven_win_rate",
        avg_loss / (avg_win + avg_loss),
        n=len(wins) + len(losses),
        unit="fraction",
    )


# ---------------------------------------------------------------------------
# Excursions, holding time, cost drag
# ---------------------------------------------------------------------------

_EXCURSION_NOTE = (
    "mae_r/mfe_r default to 0.0 on TradeRecord, so an unmeasured excursion is "
    "indistinguishable from a trade that never went against or in favour of "
    "the position. Read alongside the fill model's intrabar coverage."
)


def mean_mae_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because("mean_mae_r", Undefined.NO_TRADES, unit="R")
    return MetricValue.measured(
        "mean_mae_r", _mean([x.mae_r for x in t]), n=len(t), unit="R", note=_EXCURSION_NOTE
    )


def worst_mae_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because("worst_mae_r", Undefined.NO_TRADES, unit="R")
    return MetricValue.measured(
        "worst_mae_r", min(x.mae_r for x in t), n=len(t), unit="R", note=_EXCURSION_NOTE
    )


def mean_mfe_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because("mean_mfe_r", Undefined.NO_TRADES, unit="R")
    return MetricValue.measured(
        "mean_mfe_r", _mean([x.mfe_r for x in t]), n=len(t), unit="R", note=_EXCURSION_NOTE
    )


def best_mfe_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because("best_mfe_r", Undefined.NO_TRADES, unit="R")
    return MetricValue.measured(
        "best_mfe_r", max(x.mfe_r for x in t), n=len(t), unit="R", note=_EXCURSION_NOTE
    )


def mean_holding_seconds(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because(
            "mean_holding_seconds", Undefined.NO_TRADES, unit="s"
        )
    return MetricValue.measured(
        "mean_holding_seconds",
        _mean([x.holding_time_seconds for x in t]),
        n=len(t),
        unit="s",
    )


def median_holding_seconds(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because(
            "median_holding_seconds", Undefined.NO_TRADES, unit="s"
        )
    return MetricValue.measured(
        "median_holding_seconds",
        _median([x.holding_time_seconds for x in t]),
        n=len(t),
        unit="s",
    )


def mean_bars_held(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because(
            "mean_bars_held", Undefined.NO_TRADES, unit="bars"
        )
    return MetricValue.measured(
        "mean_bars_held", _mean([float(x.bars_held) for x in t]), n=len(t), unit="bars"
    )


def mean_cost_drag_r(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Mean cost drag in R across ALL exits -- an exit-mix-weighted average.

    Quotable only beside the exit mix, because a stop exit costs more than
    twice a target exit (Phase 7a: 0.1194R vs 0.0537R). Use
    `cost_drag_by_exit_reason` for the figure that does not depend on the mix.
    """
    t = _checked(trades, split_label)
    if not t:
        return MetricValue.undefined_because(
            "mean_cost_drag_r", Undefined.NO_TRADES, unit="R"
        )
    return MetricValue.measured(
        "mean_cost_drag_r",
        _mean([x.cost_drag_r for x in t]),
        n=len(t),
        unit="R",
        note="exit-mix weighted; see cost_drag_by_exit_reason",
    )


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------


class StreakProfile(FrozenModel):
    """Consecutive-outcome runs, in the order outcomes were realized.

    A scratch breaks both streaks: it is not a win and not a loss. Runs are
    measured on exit order, because that is when the account felt it.
    """

    split_label: SplitPhase | None
    n_trades: int = Field(ge=0)
    max_consecutive_wins: MetricValue
    max_consecutive_losses: MetricValue
    current_streak: MetricValue
    """Signed: positive for a run of wins, negative for a run of losses."""


def streaks(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> StreakProfile:
    t = _checked(trades, split_label)
    n = len(t)
    if not t:
        def undefined(name: str) -> MetricValue:
            return MetricValue.undefined_because(
                name, Undefined.NO_TRADES, unit="trades"
            )

        return StreakProfile(
            split_label=split_label,
            n_trades=0,
            max_consecutive_wins=undefined("max_consecutive_wins"),
            max_consecutive_losses=undefined("max_consecutive_losses"),
            current_streak=undefined("current_streak"),
        )

    best_win = best_loss = run_win = run_loss = 0
    for trade in _chronological(t):
        if trade.is_win:
            run_win, run_loss = run_win + 1, 0
        elif trade.is_loss:
            run_loss, run_win = run_loss + 1, 0
        else:
            run_win = run_loss = 0
        best_win, best_loss = max(best_win, run_win), max(best_loss, run_loss)
    current = run_win if run_win else -run_loss
    return StreakProfile(
        split_label=split_label,
        n_trades=n,
        max_consecutive_wins=MetricValue.measured(
            "max_consecutive_wins", float(best_win), n=n, unit="trades"
        ),
        max_consecutive_losses=MetricValue.measured(
            "max_consecutive_losses", float(best_loss), n=n, unit="trades"
        ),
        current_streak=MetricValue.measured(
            "current_streak", float(current), n=n, unit="trades"
        ),
    )


def max_consecutive_wins(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    return streaks(trades, split_label=split_label, policy=policy).max_consecutive_wins


def max_consecutive_losses(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    return streaks(trades, split_label=split_label, policy=policy).max_consecutive_losses


# ---------------------------------------------------------------------------
# R distribution
# ---------------------------------------------------------------------------


class RBucket(FrozenModel):
    """One histogram bin. Counts only -- a share of 3 trades is still 3 trades."""

    lower: float | None
    """None for the open left tail."""

    upper: float | None
    """None for the open right tail."""

    count: int = Field(ge=0)

    @property
    def label(self) -> str:
        low = "-inf" if self.lower is None else f"{self.lower:g}"
        high = "+inf" if self.upper is None else f"{self.upper:g}"
        return f"[{low},{high})"


class RDistribution(FrozenModel):
    """The shape of realized R, not just its mean."""

    split_label: SplitPhase | None
    n_trades: int = Field(ge=0)
    mean: MetricValue
    median: MetricValue
    stdev: MetricValue
    minimum: MetricValue
    maximum: MetricValue
    percentiles: dict[str, MetricValue]
    histogram: tuple[RBucket, ...]
    edges: tuple[float, ...]


def r_distribution(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    edges: Sequence[float] = R_HISTOGRAM_EDGES,
) -> RDistribution:
    """Percentiles and a histogram of realized R.

    With no trades every statistic is undefined and the histogram is a full
    set of zero-count buckets: the shape of the empty result is the same shape
    as a populated one, so a caller (and the dashboard) needs no special case.
    """
    t = _checked(trades, split_label)
    n = len(t)
    bounds = tuple(float(e) for e in edges)
    values = _r_values(t)

    counts = [0] * (len(bounds) + 1)
    for value in values:
        index = int(np.searchsorted(np.asarray(bounds, dtype=float), value, side="right"))
        counts[index] += 1
    histogram = tuple(
        RBucket(
            lower=None if i == 0 else bounds[i - 1],
            upper=None if i == len(bounds) else bounds[i],
            count=count,
        )
        for i, count in enumerate(counts)
    )

    def undefined(name: str) -> MetricValue:
        return MetricValue.undefined_because(name, Undefined.NO_TRADES, unit="R")

    if not t:
        return RDistribution(
            split_label=split_label,
            n_trades=0,
            mean=undefined("mean_r"),
            median=undefined("median_r"),
            stdev=undefined("stdev_r"),
            minimum=undefined("min_r"),
            maximum=undefined("max_r"),
            percentiles={
                f"p{q:g}": undefined(f"r_p{q:g}") for q in R_PERCENTILES
            },
            histogram=histogram,
            edges=bounds,
        )

    return RDistribution(
        split_label=split_label,
        n_trades=n,
        mean=MetricValue.measured("mean_r", _mean(values), n=n, unit="R"),
        median=MetricValue.measured("median_r", _median(values), n=n, unit="R"),
        stdev=(
            MetricValue.measured("stdev_r", _stdev(values), n=n, unit="R")
            if n >= 2
            else MetricValue.undefined_because(
                "stdev_r", Undefined.INSUFFICIENT_OBSERVATIONS, n=n, unit="R"
            )
        ),
        minimum=MetricValue.measured("min_r", min(values), n=n, unit="R"),
        maximum=MetricValue.measured("max_r", max(values), n=n, unit="R"),
        percentiles={
            f"p{q:g}": MetricValue.measured(
                f"r_p{q:g}", _percentile(values, q), n=n, unit="R"
            )
            for q in R_PERCENTILES
        },
        histogram=histogram,
        edges=bounds,
    )


# ---------------------------------------------------------------------------
# Cost drag by exit reason (the brief's strategy-audit requirement)
# ---------------------------------------------------------------------------


class CostDragReport(FrozenModel):
    """Cost drag per exit reason, with the exit mix beside it.

    Phase 7a measured the asymmetry this report exists for: a target is a
    resting limit (no spread, no slippage) while a stop is a market order that
    also pays `stop_order_extra_slippage_ticks`, so one NQ round turn cost
    0.1194R through the stop and 0.0537R through the target. A single
    "cost drag" figure describes neither, which is why `mix_weighted_r`
    is never published without `exit_counts`.
    """

    split_label: SplitPhase | None
    n_trades: int = Field(ge=0)
    by_reason_r: dict[ExitReason, MetricValue]
    by_reason_dollars: dict[ExitReason, MetricValue]
    exit_counts: dict[ExitReason, int]
    mix_weighted_r: MetricValue
    mix_weighted_dollars: MetricValue


def cost_drag_by_exit_reason(
    trades: Iterable[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> CostDragReport:
    """Mean cost drag per exit reason, in R and in dollars.

    Reasons are iterated in `ExitReason` declaration order (determinism), and
    only reasons present in the sample get a cell. No `min_sample_n`
    suppression is applied here, deliberately: a round-turn cost is mechanical
    given size, spread and volatility, not an outcome rate, so a small-N mean
    cost is a cost rather than a claim about profitability. The
    `exit_reason` dimension in `breakdowns` applies the usual suppression to
    the outcome metrics for the same trades.
    """
    t = _checked(trades, split_label)
    n = len(t)
    by_reason_r: dict[ExitReason, MetricValue] = {}
    by_reason_dollars: dict[ExitReason, MetricValue] = {}
    counts: dict[ExitReason, int] = {}
    for reason in ExitReason:
        subset = [x for x in t if x.exit_reason is reason]
        if not subset:
            continue
        counts[reason] = len(subset)
        by_reason_r[reason] = MetricValue.measured(
            f"cost_drag_r:{reason.value}",
            _mean([x.cost_drag_r for x in subset]),
            n=len(subset),
            unit="R",
        )
        by_reason_dollars[reason] = MetricValue.measured(
            f"cost_dollars:{reason.value}",
            _mean([x.costs.total for x in subset]),
            n=len(subset),
            unit="USD",
        )

    if not t:
        return CostDragReport(
            split_label=split_label,
            n_trades=0,
            by_reason_r={},
            by_reason_dollars={},
            exit_counts={},
            mix_weighted_r=MetricValue.undefined_because(
                "mean_cost_drag_r", Undefined.NO_TRADES, unit="R"
            ),
            mix_weighted_dollars=MetricValue.undefined_because(
                "mean_cost_dollars", Undefined.NO_TRADES, unit="USD"
            ),
        )

    return CostDragReport(
        split_label=split_label,
        n_trades=n,
        by_reason_r=by_reason_r,
        by_reason_dollars=by_reason_dollars,
        exit_counts=counts,
        mix_weighted_r=MetricValue.measured(
            "mean_cost_drag_r",
            _mean([x.cost_drag_r for x in t]),
            n=n,
            unit="R",
            note="exit-mix weighted; meaningless without exit_counts",
        ),
        mix_weighted_dollars=MetricValue.measured(
            "mean_cost_dollars", _mean([x.costs.total for x in t]), n=n, unit="USD"
        ),
    )


# ---------------------------------------------------------------------------
# The equity series
# ---------------------------------------------------------------------------


def _ordered_equity(equity: Iterable[EquityPoint]) -> tuple[EquityPoint, ...]:
    """The series as given, after checking it runs forwards.

    Equal timestamps are allowed (several marks at one bar close); a strict
    decrease is a bug upstream. Sorting it here would hide that bug behind a
    plausible drawdown number.
    """
    points = tuple(equity)
    for earlier, later in zip(points, points[1:]):
        if later.ts < earlier.ts:
            raise NonChronologicalEquityError(
                f"equity series goes backwards at {later.ts.isoformat()} "
                f"(after {earlier.ts.isoformat()})"
            )
    return points


def simple_returns(equity: Iterable[EquityPoint]) -> list[float] | None:
    """Per-point simple returns, or None if equity ever reached zero or below.

    A ruined account has no meaningful return series, so the callers report
    `NON_POSITIVE_EQUITY` rather than dividing by it.
    """
    points = _ordered_equity(equity)
    if len(points) < 2:
        return []
    if any(p.equity <= 0.0 for p in points[:-1]):
        return None
    return [
        later.equity / earlier.equity - 1.0
        for earlier, later in zip(points, points[1:])
    ]


def infer_periods_per_year(
    equity: Iterable[EquityPoint], *, policy: MetricPolicy | None = None
) -> MetricValue:
    """Annualization factor MEASURED from the series' own sampling density.

    `observations per calendar date present` x `trading_days_per_year`. The
    alternative -- dividing a Julian year by the median bar spacing -- counts
    hours the market is shut as sampling opportunities and inflates an
    annualized Sharpe by the square root of the overcount, which is exactly
    the direction that flatters. Dates present, rather than dates spanned, is
    the conservative choice for a series with holidays in it.
    """
    pol = _policy(policy)
    points = _ordered_equity(equity)
    if len(points) < 2:
        return MetricValue.undefined_because(
            "periods_per_year", Undefined.NO_EQUITY_SERIES, n=len(points),
            unit="periods/year",
        )
    dates = {p.ts.date() for p in points}
    observations_per_day = (len(points) - 1) / len(dates)
    return MetricValue.measured(
        "periods_per_year",
        observations_per_day * pol.trading_days_per_year,
        n=len(points) - 1,
        unit="periods/year",
        note=f"{observations_per_day:.6g} observations/day x {pol.trading_days_per_year} days",
    )


def _risk_free_per_period(annual_rate: float, periods_per_year: float) -> float:
    """Geometric de-annualization. DECLARED CHOICE: compounding the quoted
    annual rate, not dividing it, so the hurdle is self-consistent with a
    compounded return series."""
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def sharpe_from_returns(
    returns: Sequence[float],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    periods_per_year: float | None = None,
) -> MetricValue:
    """Sharpe ratio over a return series.

    * fewer than `policy.min_return_observations` observations -> UNDEFINED.
      The minimum is a declared constant, not a silent 2.
    * zero standard deviation (a flat curve, which is what zero trades
      produces) -> UNDEFINED. 0/0 is not 0.
    * `periods_per_year=None` -> per-period Sharpe, and the risk-free rate is
      NOT applied, because an annual rate cannot be converted without knowing
      the period. The unit says so.
    """
    pol = _policy(policy)
    if periods_per_year is not None and periods_per_year <= 0.0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    observations = list(returns)
    n = len(observations)
    unit = "ratio" if periods_per_year is not None else "ratio/period"
    if n < pol.min_return_observations:
        return MetricValue.undefined_because(
            "sharpe_ratio",
            Undefined.INSUFFICIENT_OBSERVATIONS,
            n=n,
            unit=unit,
            note=f"needs at least {pol.min_return_observations} return observations",
        )
    deviation = _stdev(observations)
    if deviation == 0.0:
        return MetricValue.undefined_because(
            "sharpe_ratio",
            Undefined.ZERO_DISPERSION,
            n=n,
            unit=unit,
            note="returns have no dispersion; the ratio is 0/0, not 0",
        )
    if periods_per_year is None:
        return MetricValue.measured(
            "sharpe_ratio",
            _mean(observations) / deviation,
            n=n,
            unit=unit,
            note="per period, excess over zero (no annualization supplied)",
        )
    hurdle = _risk_free_per_period(pol.risk_free_rate_annual, periods_per_year)
    return MetricValue.measured(
        "sharpe_ratio",
        (_mean(observations) - hurdle) / deviation * math.sqrt(periods_per_year),
        n=n,
        unit=unit,
        note=(
            f"annualized at {periods_per_year:.6g} periods/year over a "
            f"{pol.risk_free_rate_annual:.4g} annual risk-free rate"
        ),
    )


def sortino_from_returns(
    returns: Sequence[float],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    periods_per_year: float | None = None,
) -> MetricValue:
    """Sortino ratio: excess return over downside deviation.

    Downside deviation divides by ALL observations (`DOWNSIDE_DDOF`), the
    standard definition. With no observation below the target the denominator
    is zero and the ratio is UNDEFINED (`NO_DOWNSIDE`) -- never infinity, and
    never a large number standing in for one.
    """
    pol = _policy(policy)
    if periods_per_year is not None and periods_per_year <= 0.0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    observations = list(returns)
    n = len(observations)
    unit = "ratio" if periods_per_year is not None else "ratio/period"
    if n < pol.min_return_observations:
        return MetricValue.undefined_because(
            "sortino_ratio",
            Undefined.INSUFFICIENT_OBSERVATIONS,
            n=n,
            unit=unit,
            note=f"needs at least {pol.min_return_observations} return observations",
        )
    if _stdev(observations) == 0.0:
        return MetricValue.undefined_because(
            "sortino_ratio",
            Undefined.ZERO_DISPERSION,
            n=n,
            unit=unit,
            note=(
                "returns have no dispersion, so the only shortfall is the "
                "risk-free hurdle itself: the ratio would be -1 per period "
                "(hugely negative once annualized) while measuring no risk at "
                "all. A flat curve -- which is what zero trades produces -- is "
                "reported here as undefined rather than as a catastrophic "
                "number. Read total_return and trade_frequency instead."
            ),
        )
    target = (
        0.0
        if periods_per_year is None
        else _risk_free_per_period(pol.risk_free_rate_annual, periods_per_year)
    )
    shortfalls = [min(r - target, 0.0) for r in observations]
    if not any(s < 0.0 for s in shortfalls):
        return MetricValue.undefined_because(
            "sortino_ratio",
            Undefined.NO_DOWNSIDE,
            n=n,
            unit=unit,
            note="no observation below target, so downside deviation is zero",
        )
    downside = math.sqrt(
        sum(s * s for s in shortfalls) / (len(shortfalls) - DOWNSIDE_DDOF)
    )
    ratio = (_mean(observations) - target) / downside
    if periods_per_year is None:
        return MetricValue.measured(
            "sortino_ratio", ratio, n=n, unit=unit, note="per period, target 0.0"
        )
    return MetricValue.measured(
        "sortino_ratio",
        ratio * math.sqrt(periods_per_year),
        n=n,
        unit=unit,
        note=f"annualized at {periods_per_year:.6g} periods/year",
    )


def _annualization(
    equity: Sequence[EquityPoint],
    policy: MetricPolicy,
    periods_per_year: float | None,
) -> float | None:
    if periods_per_year is not None:
        return periods_per_year
    return infer_periods_per_year(equity, policy=policy).or_none()


def sharpe_ratio(
    equity: Iterable[EquityPoint],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    periods_per_year: float | None = None,
) -> MetricValue:
    """Annualized Sharpe from the equity curve.

    `split_label` is recorded, not verified: `EquityPoint` carries no phase,
    so the caller's label is taken on trust here. That is stated plainly
    rather than implied -- the structural check lives on the trade-based
    functions, which can actually see the phase.
    """
    pol = _policy(policy)
    points = _ordered_equity(equity)
    if len(points) < 2:
        return MetricValue.undefined_because(
            "sharpe_ratio", Undefined.NO_EQUITY_SERIES, n=len(points), unit="ratio"
        )
    returns = simple_returns(points)
    if returns is None:
        return MetricValue.undefined_because(
            "sharpe_ratio", Undefined.NON_POSITIVE_EQUITY, n=len(points), unit="ratio"
        )
    factor = _annualization(points, pol, periods_per_year)
    if factor is None:
        return MetricValue.undefined_because(
            "sharpe_ratio", Undefined.NO_ANNUALIZATION, n=len(returns), unit="ratio"
        )
    return sharpe_from_returns(
        returns, split_label=split_label, policy=pol, periods_per_year=factor
    )


def sortino_ratio(
    equity: Iterable[EquityPoint],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    periods_per_year: float | None = None,
) -> MetricValue:
    """Annualized Sortino from the equity curve. See `sharpe_ratio` on the
    status of `split_label` here."""
    pol = _policy(policy)
    points = _ordered_equity(equity)
    if len(points) < 2:
        return MetricValue.undefined_because(
            "sortino_ratio", Undefined.NO_EQUITY_SERIES, n=len(points), unit="ratio"
        )
    returns = simple_returns(points)
    if returns is None:
        return MetricValue.undefined_because(
            "sortino_ratio", Undefined.NON_POSITIVE_EQUITY, n=len(points), unit="ratio"
        )
    factor = _annualization(points, pol, periods_per_year)
    if factor is None:
        return MetricValue.undefined_because(
            "sortino_ratio", Undefined.NO_ANNUALIZATION, n=len(returns), unit="ratio"
        )
    return sortino_from_returns(
        returns, split_label=split_label, policy=pol, periods_per_year=factor
    )


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class DrawdownEpisode(FrozenModel):
    """One peak-to-trough-to-recovery excursion. Unrecovered episodes are
    kept, not dropped: "do not hide bad periods" starts with the one still
    open at the end of the series."""

    peak_ts: datetime
    peak_equity: float
    trough_ts: datetime
    trough_equity: float
    recovered_ts: datetime | None = None

    @property
    def depth_dollars(self) -> float:
        """Negative."""
        return self.trough_equity - self.peak_equity

    @property
    def depth_pct(self) -> float | None:
        """Negative fraction of the peak, or None if the peak was not positive."""
        if self.peak_equity <= 0.0:
            return None
        return self.depth_dollars / self.peak_equity

    @property
    def is_recovered(self) -> bool:
        return self.recovered_ts is not None

    @property
    def recovery_seconds(self) -> float | None:
        """Peak to recovery of that peak -- the whole underwater stretch."""
        if self.recovered_ts is None:
            return None
        return (self.recovered_ts - self.peak_ts).total_seconds()


def drawdown_episodes(equity: Iterable[EquityPoint]) -> tuple[DrawdownEpisode, ...]:
    """Every excursion below the running peak, in order.

    The running peak is computed from the series itself rather than read from
    `EquityPoint.high_water_mark`, because that field defaults to 0.0 and an
    unset default would silently report zero drawdown. The stored value of the
    FIRST point is still honoured as a lower bound on the starting peak, so a
    series that begins mid-run does not understate the drawdown it is already
    in.
    """
    points = _ordered_equity(equity)
    if not points:
        return ()
    peak = max(points[0].equity, points[0].high_water_mark)
    peak_ts = points[0].ts
    underwater = False
    trough_equity = peak
    trough_ts = peak_ts
    episodes: list[DrawdownEpisode] = []
    for point in points:
        if point.equity >= peak:
            if underwater:
                episodes.append(
                    DrawdownEpisode(
                        peak_ts=peak_ts,
                        peak_equity=peak,
                        trough_ts=trough_ts,
                        trough_equity=trough_equity,
                        recovered_ts=point.ts,
                    )
                )
                underwater = False
            peak, peak_ts = point.equity, point.ts
        elif not underwater:
            underwater = True
            trough_equity, trough_ts = point.equity, point.ts
        elif point.equity < trough_equity:
            trough_equity, trough_ts = point.equity, point.ts
    if underwater:
        # Only the final episode can still be open, and it is KEPT: an
        # unrecovered drawdown is the one a report most wants to drop.
        episodes.append(
            DrawdownEpisode(
                peak_ts=peak_ts,
                peak_equity=peak,
                trough_ts=trough_ts,
                trough_equity=trough_equity,
                recovered_ts=None,
            )
        )
    return tuple(episodes)


class DrawdownProfile(FrozenModel):
    """Max and average drawdown, recovery times, and what is still open.

    Depths are NEGATIVE fractions and NEGATIVE dollars, matching
    `EquityPoint.drawdown_pct` and `TradeRecord.mae_r`, so a drawdown never
    changes sign between two parts of this system.
    """

    split_label: SplitPhase | None
    n_points: int = Field(ge=0)
    n_episodes: int = Field(ge=0)
    n_unrecovered: int = Field(ge=0)
    max_drawdown_pct: MetricValue
    max_drawdown_dollars: MetricValue
    average_drawdown_pct: MetricValue
    longest_recovery_seconds: MetricValue
    mean_recovery_seconds: MetricValue
    seconds_underwater_at_end: MetricValue
    trough_ts: datetime | None = None
    episodes: tuple[DrawdownEpisode, ...] = ()


def drawdown_profile(
    equity: Iterable[EquityPoint],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> DrawdownProfile:
    """Drawdown statistics from the equity curve.

    Max drawdown over a flat curve is 0.0 and that is a MEASUREMENT: zero
    depth was observed at every point. The average over zero episodes is
    undefined, because the mean of an empty set is not zero.
    """
    points = _ordered_equity(equity)
    episodes = drawdown_episodes(points)
    n_points = len(points)

    if n_points == 0:
        def nothing(name: str, unit: str) -> MetricValue:
            return MetricValue.undefined_because(
                name, Undefined.NO_EQUITY_SERIES, n=0, unit=unit
            )

        return DrawdownProfile(
            split_label=split_label,
            n_points=0,
            n_episodes=0,
            n_unrecovered=0,
            max_drawdown_pct=nothing("max_drawdown_pct", "fraction"),
            max_drawdown_dollars=nothing("max_drawdown_dollars", "USD"),
            average_drawdown_pct=nothing("average_drawdown_pct", "fraction"),
            longest_recovery_seconds=nothing("longest_recovery_seconds", "s"),
            mean_recovery_seconds=nothing("mean_recovery_seconds", "s"),
            seconds_underwater_at_end=nothing("seconds_underwater_at_end", "s"),
        )

    depths_pct = [e.depth_pct for e in episodes]
    measurable_pct = [d for d in depths_pct if d is not None]
    unmeasurable = len(depths_pct) - len(measurable_pct)

    if unmeasurable:
        max_pct = MetricValue.undefined_because(
            "max_drawdown_pct",
            Undefined.NON_POSITIVE_EQUITY,
            n=n_points,
            unit="fraction",
            note=f"{unmeasurable} episode(s) peaked at non-positive equity",
        )
        avg_pct = MetricValue.undefined_because(
            "average_drawdown_pct",
            Undefined.NON_POSITIVE_EQUITY,
            n=len(episodes),
            unit="fraction",
        )
    else:
        max_pct = MetricValue.measured(
            "max_drawdown_pct",
            min(measurable_pct) if measurable_pct else 0.0,
            n=n_points,
            unit="fraction",
            note="negative; 0.0 means the curve never fell below its peak",
        )
        avg_pct = (
            MetricValue.measured(
                "average_drawdown_pct",
                _mean(measurable_pct),
                n=len(measurable_pct),
                unit="fraction",
                note="mean trough depth per episode",
            )
            if measurable_pct
            else MetricValue.undefined_because(
                "average_drawdown_pct",
                Undefined.NO_DRAWDOWN_EPISODE,
                n=0,
                unit="fraction",
                note="the curve never fell below its running peak",
            )
        )

    max_dollars = MetricValue.measured(
        "max_drawdown_dollars",
        min((e.depth_dollars for e in episodes), default=0.0),
        n=n_points,
        unit="USD",
        note="negative",
    )

    recovered = [e.recovery_seconds for e in episodes if e.recovery_seconds is not None]
    unrecovered = [e for e in episodes if not e.is_recovered]
    if recovered:
        longest = MetricValue.measured(
            "longest_recovery_seconds", max(recovered), n=len(recovered), unit="s"
        )
        mean_recovery = MetricValue.measured(
            "mean_recovery_seconds", _mean(recovered), n=len(recovered), unit="s"
        )
    else:
        reason = (
            Undefined.NOT_RECOVERED if episodes else Undefined.NO_DRAWDOWN_EPISODE
        )
        note = (
            f"{len(unrecovered)} episode(s) still open at the end of the series"
            if episodes
            else "the curve never fell below its running peak"
        )
        longest = MetricValue.undefined_because(
            "longest_recovery_seconds", reason, n=0, unit="s", note=note
        )
        mean_recovery = MetricValue.undefined_because(
            "mean_recovery_seconds", reason, n=0, unit="s", note=note
        )

    underwater = 0.0
    if unrecovered:
        still_open = unrecovered[-1]
        underwater = (points[-1].ts - still_open.peak_ts).total_seconds()

    deepest = min(episodes, key=lambda e: e.depth_dollars, default=None)
    return DrawdownProfile(
        split_label=split_label,
        n_points=n_points,
        n_episodes=len(episodes),
        n_unrecovered=len(unrecovered),
        max_drawdown_pct=max_pct,
        max_drawdown_dollars=max_dollars,
        average_drawdown_pct=avg_pct,
        longest_recovery_seconds=longest,
        mean_recovery_seconds=mean_recovery,
        seconds_underwater_at_end=MetricValue.measured(
            "seconds_underwater_at_end", underwater, n=n_points, unit="s"
        ),
        trough_ts=deepest.trough_ts if deepest is not None else None,
        episodes=episodes,
    )


def max_drawdown(
    equity: Iterable[EquityPoint],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    return drawdown_profile(
        equity, split_label=split_label, policy=policy
    ).max_drawdown_pct


def average_drawdown(
    equity: Iterable[EquityPoint],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    return drawdown_profile(
        equity, split_label=split_label, policy=policy
    ).average_drawdown_pct


def recovery_time(
    equity: Iterable[EquityPoint],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Longest peak-to-recovery stretch. Undefined while nothing has
    recovered -- an unrecovered drawdown has no recovery time yet, and
    reporting the elapsed time as if it were one would understate it."""
    return drawdown_profile(
        equity, split_label=split_label, policy=policy
    ).longest_recovery_seconds


# ---------------------------------------------------------------------------
# Return, growth rate, frequency
# ---------------------------------------------------------------------------


def total_return(
    equity: Iterable[EquityPoint],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """(last - first) / first over the equity series."""
    points = _ordered_equity(equity)
    if len(points) < 2:
        return MetricValue.undefined_because(
            "total_return", Undefined.NO_EQUITY_SERIES, n=len(points), unit="fraction"
        )
    start, end = points[0].equity, points[-1].equity
    if start <= 0.0:
        return MetricValue.undefined_because(
            "total_return",
            Undefined.NON_POSITIVE_EQUITY,
            n=len(points),
            unit="fraction",
        )
    return MetricValue.measured(
        "total_return", end / start - 1.0, n=len(points), unit="fraction"
    )


def cagr(
    equity: Iterable[EquityPoint],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
) -> MetricValue:
    """Compound annual growth rate over the CALENDAR span of the series.

    Calendar rather than trading time: this is a per-wall-clock-year growth
    rate, and `SECONDS_PER_CALENDAR_YEAR` says which year. A terminal equity
    of zero or less has no geometric growth rate at all, so it is reported
    undefined instead of as -100%.
    """
    points = _ordered_equity(equity)
    if len(points) < 2:
        return MetricValue.undefined_because(
            "cagr", Undefined.NO_EQUITY_SERIES, n=len(points), unit="fraction/year"
        )
    start, end = points[0].equity, points[-1].equity
    if start <= 0.0 or end <= 0.0:
        return MetricValue.undefined_because(
            "cagr",
            Undefined.NON_POSITIVE_EQUITY,
            n=len(points),
            unit="fraction/year",
        )
    years = (points[-1].ts - points[0].ts).total_seconds() / SECONDS_PER_CALENDAR_YEAR
    if years <= 0.0:
        return MetricValue.undefined_because(
            "cagr", Undefined.ZERO_ELAPSED_TIME, n=len(points), unit="fraction/year"
        )
    return MetricValue.measured(
        "cagr",
        (end / start) ** (1.0 / years) - 1.0,
        n=len(points),
        unit="fraction/year",
        note=f"over {years:.6g} calendar years",
    )


def trade_frequency(
    trades: Iterable[TradeRecord],
    equity: Iterable[EquityPoint] = (),
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    span_seconds: float | None = None,
) -> MetricValue:
    """Trades per calendar year over the observation window.

    ZERO trades per year over a known window is a measurement, and it is the
    system's current one. The window comes from `span_seconds` when given,
    otherwise from the equity series -- never from the trades themselves,
    because the span between the first and last of zero trades does not
    exist, and the span between the first and last of three trades is not the
    period they were available to be taken in.
    """
    t = _checked(trades, split_label)
    span = span_seconds
    if span is None:
        points = _ordered_equity(equity)
        if len(points) < 2:
            return MetricValue.undefined_because(
                "trade_frequency",
                Undefined.NO_EQUITY_SERIES,
                n=len(t),
                unit="trades/year",
                note="supply the equity series or an explicit span_seconds",
            )
        span = (points[-1].ts - points[0].ts).total_seconds()
    if span <= 0.0:
        return MetricValue.undefined_because(
            "trade_frequency", Undefined.ZERO_ELAPSED_TIME, n=len(t), unit="trades/year"
        )
    years = span / SECONDS_PER_CALENDAR_YEAR
    return MetricValue.measured(
        "trade_frequency",
        len(t) / years,
        n=len(t),
        unit="trades/year",
        note=f"over {years:.6g} calendar years",
    )


# ---------------------------------------------------------------------------
# The registry (how `breakdowns` stays independent of the metric list)
# ---------------------------------------------------------------------------

#: Every trade-only metric, by name. `breakdowns` iterates this mapping, so
#: adding a metric here is enough to put it in every breakdown cell, and
#: adding a breakdown dimension never touches this module. The keys are the
#: names `validation/hypotheses.py` refers to ("Must match an analytics
#: breakdown key"), so `win_rate` and `expectancy` are spelled as it spells
#: them. `expectancy` and `expectancy_r` are two keys onto ONE function: the
#: first is the name the pre-registration uses, the second says the unit.
#:
#: Dict order is source order, which keeps every report's column order
#: deterministic. Each function here takes `split_label` and therefore
#: carries the section 0.8 refusal: a metric cannot be registered without it.
#: A registered metric: trades in, one `MetricValue` out, split_label obliged.
TradeMetricFn = Callable[..., MetricValue]

TRADE_METRICS: dict[str, TradeMetricFn] = {
    "trade_count": trade_count,
    "win_rate": win_rate,
    "loss_rate": loss_rate,
    "scratch_rate": scratch_rate,
    "expectancy": expectancy_r,
    "expectancy_r": expectancy_r,
    "expectancy_dollars": expectancy_dollars,
    "mean_r": mean_r,
    "median_r": median_r,
    "stdev_r": stdev_r,
    "average_win_r": average_win_r,
    "average_loss_r": average_loss_r,
    "win_loss_ratio": win_loss_ratio,
    "profit_factor": profit_factor,
    "breakeven_win_rate": breakeven_win_rate,
    "total_pnl": total_pnl,
    "gross_profit": gross_profit,
    "gross_loss": gross_loss,
    "total_costs": total_costs,
    "mean_cost_drag_r": mean_cost_drag_r,
    "mean_mae_r": mean_mae_r,
    "worst_mae_r": worst_mae_r,
    "mean_mfe_r": mean_mfe_r,
    "best_mfe_r": best_mfe_r,
    "mean_holding_seconds": mean_holding_seconds,
    "median_holding_seconds": median_holding_seconds,
    "mean_bars_held": mean_bars_held,
    "max_consecutive_wins": max_consecutive_wins,
    "max_consecutive_losses": max_consecutive_losses,
}

#: The subset reported in a breakdown cell by default. JUDGEMENT CALL about
#: width, not about content: the full registry is available by passing
#: `metrics=tuple(TRADE_METRICS)`. Chosen to cover section 12's named
#: quantities plus the cost drag the strategy audit asks for.
DEFAULT_CELL_METRICS: tuple[str, ...] = (
    "trade_count",
    "win_rate",
    "loss_rate",
    "expectancy",
    "mean_r",
    "median_r",
    "profit_factor",
    "average_win_r",
    "average_loss_r",
    "breakeven_win_rate",
    "total_pnl",
    "mean_cost_drag_r",
    "mean_mae_r",
    "mean_mfe_r",
    "mean_holding_seconds",
    "max_consecutive_losses",
)


def compute_trade_metrics(
    trades: Sequence[TradeRecord],
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    names: Sequence[str] = DEFAULT_CELL_METRICS,
) -> dict[str, MetricValue]:
    """Run named registry metrics over one trade set, in the given order."""
    pol = _policy(policy)
    unknown = [name for name in names if name not in TRADE_METRICS]
    if unknown:
        raise KeyError(f"unknown metric name(s): {unknown}")
    out: dict[str, MetricValue] = {}
    for name in names:
        function = TRADE_METRICS[name]
        out[name] = function(trades, split_label=split_label, policy=pol)
    return out


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------

ZERO_TRADE_NOTE = (
    "n=0: every rate, mean and ratio below is undefined BY DEFINITION, not by "
    "failure. The system as configured takes no trades (highest Flow Score "
    "measured 65.23 against a lowest setup threshold of 70.0), so this is the "
    "expected shape of a run today."
)


class PerformanceSummary(FrozenModel):
    """Everything section 12 names, for one split phase, with N on every number.

    `min_sample_n` suppression is NOT applied to these top-level numbers --
    only to breakdown cells, where section 12 mandates it. The asymmetry is
    deliberate: the headline is where N belongs beside the value rather than
    instead of it, and `sample` states the verdict explicitly so a reader
    cannot miss a thin sample. Suppressing the headline too would also hide
    the one honest summary of a thin run.
    """

    split_label: SplitPhase | None
    n_trades: int = Field(ge=0)
    n_wins: int = Field(ge=0)
    n_losses: int = Field(ge=0)
    n_scratch: int = Field(ge=0)
    sample: SampleStatus
    min_sample_n: int = Field(ge=1)
    trade_metrics: dict[str, MetricValue]
    periods_per_year: MetricValue
    sharpe: MetricValue
    sortino: MetricValue
    total_return: MetricValue
    cagr: MetricValue
    trades_per_year: MetricValue
    drawdown: DrawdownProfile
    streaks: StreakProfile
    r_distribution: RDistribution
    cost_drag: CostDragReport
    notes: tuple[str, ...] = ()

    def describe(self) -> tuple[str, ...]:
        """Human-readable lines. Pure string formatting, no I/O."""
        head = (
            f"split={self.split_label} n={self.n_trades} "
            f"(wins={self.n_wins} losses={self.n_losses} scratch={self.n_scratch}) "
            f"sample={self.sample}"
        )
        body = [str(value) for value in self.trade_metrics.values()]
        body += [
            str(self.sharpe),
            str(self.sortino),
            str(self.total_return),
            str(self.cagr),
            str(self.trades_per_year),
            str(self.drawdown.max_drawdown_pct),
            str(self.drawdown.average_drawdown_pct),
            str(self.drawdown.longest_recovery_seconds),
            str(self.streaks.max_consecutive_losses),
        ]
        return (head, *body, *self.notes)


def performance_summary(
    trades: Iterable[TradeRecord],
    equity: Iterable[EquityPoint] = (),
    *,
    split_label: SplitPhase | None,
    policy: MetricPolicy | None = None,
    periods_per_year: float | None = None,
    names: Sequence[str] = tuple(TRADE_METRICS),
) -> PerformanceSummary:
    """The whole section 12 metric set for one split phase.

    Completes on an empty trade set and an empty equity series. Nothing
    divides by zero, nothing returns infinity, and no undefined quantity
    comes back as a float.
    """
    pol = _policy(policy)
    t = _checked(trades, split_label)
    points = _ordered_equity(equity)
    wins, losses, scratches = outcome_counts(t)
    status = sample_status(len(t), pol.min_sample_n)

    notes: list[str] = []
    if not t:
        notes.append(ZERO_TRADE_NOTE)
    elif status is SampleStatus.INSUFFICIENT:
        notes.append(
            f"n={len(t)} is below min_sample_n={pol.min_sample_n}: these values are "
            "measurements of a sample too small to generalize from, and the "
            "breakdown cells for the same trades are reported INSUFFICIENT."
        )
    if not points:
        notes.append(
            "no equity series supplied: Sharpe, Sortino, drawdown, total return, "
            "CAGR and trade frequency are undefined for want of inputs, which is "
            "a different thing from being zero."
        )

    return PerformanceSummary(
        split_label=split_label,
        n_trades=len(t),
        n_wins=wins,
        n_losses=losses,
        n_scratch=scratches,
        sample=status,
        min_sample_n=pol.min_sample_n,
        trade_metrics=compute_trade_metrics(
            t, split_label=split_label, policy=pol, names=names
        ),
        periods_per_year=infer_periods_per_year(points, policy=pol),
        sharpe=sharpe_ratio(
            points, split_label=split_label, policy=pol, periods_per_year=periods_per_year
        ),
        sortino=sortino_ratio(
            points, split_label=split_label, policy=pol, periods_per_year=periods_per_year
        ),
        total_return=total_return(points, split_label=split_label, policy=pol),
        cagr=cagr(points, split_label=split_label, policy=pol),
        trades_per_year=trade_frequency(
            t, points, split_label=split_label, policy=pol
        ),
        drawdown=drawdown_profile(points, split_label=split_label, policy=pol),
        streaks=streaks(t, split_label=split_label, policy=pol),
        r_distribution=r_distribution(t, split_label=split_label, policy=pol),
        cost_drag=cost_drag_by_exit_reason(t, split_label=split_label, policy=pol),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Targets: reported against, never optimized toward
# ---------------------------------------------------------------------------


class Comparison(StrEnum):
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    WITHIN_RANGE = "within_range"


class TargetComparison(FrozenModel):
    """A measured value, and separately, what the brief hoped for.

    The measured value comes first in the field order and in every rendering,
    the target second, and `met` is None when there is nothing to compare.
    This type exists so that a target can be REPORTED. It is never an input
    to a choice; see `tests/unit/test_no_target_leakage.py`.
    """

    label: str
    measured: MetricValue
    comparison: Comparison
    target: float
    target_upper: float | None = None
    met: bool | None = None
    note: str = (
        "comparison only. The target is a pre-registered hypothesis from the "
        "research brief, not an objective, and never selects a parameter."
    )

    @model_validator(mode="after")
    def _check(self) -> "TargetComparison":
        if self.comparison is Comparison.WITHIN_RANGE:
            if self.target_upper is None:
                raise ValueError("WITHIN_RANGE needs target_upper")
            if self.target_upper < self.target:
                raise ValueError("target_upper must not be below target")
        elif self.target_upper is not None:
            raise ValueError(f"target_upper is meaningless for {self.comparison}")
        return self

    def describe(self) -> str:
        bound = (
            f"{self.target:g}..{self.target_upper:g}"
            if self.comparison is Comparison.WITHIN_RANGE
            else f"{self.target:g}"
        )
        verdict = "no measurement" if self.met is None else ("met" if self.met else "not met")
        return f"{self.label}: measured {self.measured} vs {self.comparison} {bound} -> {verdict}"


def compare_to_target(
    measured: MetricValue,
    target: float,
    *,
    label: str = "",
    comparison: Comparison = Comparison.AT_LEAST,
    target_upper: float | None = None,
) -> TargetComparison:
    """Compare a measurement with a stated target, reporting the measurement.

    An undefined measurement yields `met=None`. It does not yield False, and
    it certainly does not yield True: "no data" is not a result either way.
    """
    met: bool | None = None
    value = measured.or_none()
    if value is not None:
        if comparison is Comparison.AT_LEAST:
            met = value >= target
        elif comparison is Comparison.AT_MOST:
            met = value <= target
        else:
            met = target <= value <= (target_upper if target_upper is not None else target)
    return TargetComparison(
        label=label or measured.name,
        measured=measured,
        comparison=comparison,
        target=target,
        target_upper=target_upper,
        met=met,
    )


def compare_to_research_targets(
    summary: PerformanceSummary,
    targets: ResearchTargets,
    *,
    scalp_favourable_win_rate: MetricValue | None = None,
) -> tuple[TargetComparison, ...]:
    """Report the brief's win-rate hypotheses against what was measured.

    This is the ONLY place in `analytics` that reads `research_targets`, it
    reads them only to build comparisons, and the comparison objects carry the
    measured value as their primary field. `scalp_favourable_win_rate` is
    supplied by the caller from a breakdown cell (SCALP_1R in favourable
    regimes); when it is absent or undefined, the comparison says so rather
    than substituting the overall rate.
    """
    comparisons = [
        compare_to_target(
            summary.trade_metrics.get(
                "win_rate",
                MetricValue.undefined_because(
                    "win_rate", Undefined.NO_TRADES, unit="fraction"
                ),
            ),
            targets.combined_10y_win_rate,
            label="combined win rate vs brief hypothesis",
            comparison=Comparison.AT_LEAST,
        )
    ]
    low, high = targets.scalp_1r_win_rate_favourable_regimes
    comparisons.append(
        compare_to_target(
            scalp_favourable_win_rate
            if scalp_favourable_win_rate is not None
            else MetricValue.undefined_because(
                "scalp_1r_favourable_win_rate", Undefined.NO_TRADES, unit="fraction"
            ),
            low,
            label="SCALP_1R win rate in favourable regimes vs brief hypothesis",
            comparison=Comparison.WITHIN_RANGE,
            target_upper=high,
        )
    )
    return tuple(comparisons)
