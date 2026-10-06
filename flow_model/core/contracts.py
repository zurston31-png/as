"""Frozen data contracts.

Hot-path market-data types are `slots` dataclasses (millions of
instantiations in a backtest). Record/serialization types are Pydantic
models, because their invariants are what keep the research honest.

The most important invariant in this file is the definition of R:

    risk_dollars = size * |entry_price - stop_price| * point_value
    r_multiple   = pnl_net / risk_dollars

A trade may not claim a "1R target" unless |target - entry| equals
|entry - stop| within tolerance. `TradeRecord` enforces this so a high win
rate produced by an asymmetric stop/target cannot be mislabelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import Field, computed_field, model_validator

from flow_model.core.model import FrozenModel

from flow_model.core.enums import (
    Component,
    DataQuality,
    ExitReason,
    Regime,
    Session,
    SetupType,
    Side,
    SignalAction,
    SplitPhase,
    WaitReason,
)

# ---------------------------------------------------------------------------
# Market data (hot path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar.

    `close_ts` is the instant the bar *completed*. Signals derived from this
    bar are only actionable at or after `close_ts`; the backtester never
    fills at `close_ts` itself (see backtest.execution).
    """

    symbol: str
    close_ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval_seconds: int
    trades: int | None = None
    vwap: float | None = None

    @property
    def open_ts(self) -> datetime:
        return self.close_ts - timedelta(seconds=self.interval_seconds)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_valid(self) -> bool:
        """Structural sanity: OHLC ordering, non-negative volume, finite."""
        vals = (self.open, self.high, self.low, self.close)
        if not all(math.isfinite(v) for v in vals):
            return False
        if self.volume < 0 or not math.isfinite(self.volume):
            return False
        if self.high < self.low:
            return False
        if not (self.low <= self.open <= self.high):
            return False
        if not (self.low <= self.close <= self.high):
            return False
        return True

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    """Top-of-book at an instant. Used by liquidity features and the fill model."""

    symbol: str
    ts: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def depth_imbalance(self) -> float:
        """(bid_size - ask_size) / total, in [-1, 1]. 0 if book is empty."""
        total = self.bid_size + self.ask_size
        if total <= 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total

    @property
    def is_crossed(self) -> bool:
        return self.ask < self.bid


@dataclass(frozen=True, slots=True)
class TickAggregate:
    """Per-bar aggregation of tick data with aggressor classification.

    `buy_volume` is volume that traded at/above the ask (aggressive buyers);
    `sell_volume` traded at/below the bid. Volume that cannot be classified
    goes to `unclassified_volume` -- it is NOT silently assigned a side.
    """

    symbol: str
    close_ts: datetime
    buy_volume: float
    sell_volume: float
    unclassified_volume: float = 0.0
    buy_trades: int = 0
    sell_trades: int = 0
    max_trade_size: float = 0.0
    classification_method: str = "unknown"

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def classified_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def total_volume(self) -> float:
        return self.classified_volume + self.unclassified_volume

    @property
    def delta_ratio(self) -> float:
        """Normalized delta in [-1, 1]; 0 when nothing is classified."""
        cv = self.classified_volume
        if cv <= 0:
            return 0.0
        return self.delta / cv

    @property
    def classification_coverage(self) -> float:
        """Fraction of volume with a known aggressor. Drives data quality."""
        tv = self.total_volume
        if tv <= 0:
            return 0.0
        return self.classified_volume / tv


@dataclass(frozen=True, slots=True)
class OptionsSnapshot:
    """Chain-level options aggregates for one underlying at one timestamp.

    Deliberately NOT a raw chain: features consume aggregates so that the
    provider-specific chain parsing lives in the data layer.

    `is_intraday` distinguishes true flow (trade prints during the session)
    from end-of-day chain snapshots. EOD-only data can still produce
    positioning features but cannot produce flow timing, so it is graded
    DEGRADED and the options sub-score is capped.
    """

    symbol: str
    ts: datetime
    call_volume: float
    put_volume: float
    call_premium: float
    put_premium: float
    call_oi: float
    put_oi: float
    # None means NOT SUPPLIED. 0.0 is a meaningful value for all four -- "open
    # interest did not change" and "no delta-weighted volume" are real
    # observations -- so collapsing absence into 0.0 would feed a Phase 5
    # feature a measurement that was never made.
    call_oi_change: float | None = None
    put_oi_change: float | None = None
    delta_weighted_call_volume: float | None = None
    delta_weighted_put_volume: float | None = None
    atm_iv: float | None = None
    iv_25d_call: float | None = None
    iv_25d_put: float | None = None
    gamma_exposure_proxy: float | None = None
    underlying_price: float | None = None
    is_intraday: bool = False
    source: str = "unknown"

    @property
    def total_volume(self) -> float:
        return self.call_volume + self.put_volume

    @property
    def put_call_volume_ratio(self) -> float | None:
        if self.call_volume <= 0:
            return None
        return self.put_volume / self.call_volume

    @property
    def net_premium(self) -> float:
        """Call premium minus put premium, in dollars.

        NOTE: sign is ambiguous as a directional signal -- a large call
        premium may be an opening bullish bet, a closing sale, or a hedge
        leg. This value is an imbalance measurement, not evidence of intent.
        """
        return self.call_premium - self.put_premium

    @property
    def premium_imbalance(self) -> float:
        """Net premium normalized to [-1, 1]."""
        total = self.call_premium + self.put_premium
        if total <= 0:
            return 0.0
        return self.net_premium / total

    @property
    def skew_25d(self) -> float | None:
        """Put IV minus call IV at 25-delta. Positive = downside demand."""
        if self.iv_25d_put is None or self.iv_25d_call is None:
            return None
        return self.iv_25d_put - self.iv_25d_call


# ---------------------------------------------------------------------------
# Features / regime / scores
# ---------------------------------------------------------------------------


class FeatureVector(FrozenModel):
    """Named features at one timestamp, each with its own data quality.

    `quality` is the worst status among contributing feeds, so a caller can
    gate on the bundle without inspecting every key.
    """


    symbol: str
    ts: datetime
    values: dict[str, float] = Field(default_factory=dict)
    quality_by_key: dict[str, DataQuality] = Field(default_factory=dict)
    warmup_complete: bool = True
    notes: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quality(self) -> DataQuality:
        if not self.quality_by_key:
            return DataQuality.MISSING
        return DataQuality.worst(*self.quality_by_key.values())

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.values.get(key, default)

    def require(self, key: str) -> float:
        if key not in self.values:
            raise KeyError(f"feature {key!r} not present at {self.ts.isoformat()}")
        v = self.values[key]
        if not math.isfinite(v):
            raise ValueError(f"feature {key!r} is not finite: {v}")
        return v

    def quality_of(self, key: str) -> DataQuality:
        return self.quality_by_key.get(key, DataQuality.MISSING)

    def merge(self, other: "FeatureVector") -> "FeatureVector":
        """Combine two vectors at the same (symbol, ts). Keys must not collide."""
        if other.symbol != self.symbol or other.ts != self.ts:
            raise ValueError("cannot merge FeatureVectors from different symbol/ts")
        clash = set(self.values) & set(other.values)
        if clash:
            raise ValueError(f"duplicate feature keys on merge: {sorted(clash)}")
        return FeatureVector(
            symbol=self.symbol,
            ts=self.ts,
            values={**self.values, **other.values},
            quality_by_key={**self.quality_by_key, **other.quality_by_key},
            warmup_complete=self.warmup_complete and other.warmup_complete,
            notes=self.notes + other.notes,
        )


class RegimeState(FrozenModel):
    """Classified regime plus the diagnostics that produced it.

    Regimes overlap in reality, so the primary label is accompanied by the
    full measurement vector; analytics breaks results down by both.
    """


    symbol: str
    ts: datetime
    regime: Regime
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    vol_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    trend_strength: float | None = None
    efficiency_ratio: float | None = None
    vol_of_vol: float | None = None
    shift_statistic: float | None = None
    bars_in_regime: int = 0
    diagnostics: dict[str, float] = Field(default_factory=dict)

    @property
    def is_tradable(self) -> bool:
        return self.regime is not Regime.UNKNOWN


class ComponentScore(FrozenModel):
    """One component's contribution: bounded magnitude, explicit direction.

    Magnitude and direction are kept separate so that a strongly bearish
    component can never inflate a bullish Flow Score.
    """


    component: Component
    magnitude: float = Field(ge=0.0, le=1.0)
    direction: int = Field(ge=-1, le=1)
    weight: float = Field(ge=0.0)
    quality: DataQuality = DataQuality.GOOD
    enabled: bool = True
    features_used: tuple[str, ...] = ()
    detail: dict[str, float] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def points(self) -> float:
        """Weighted points contributed to the Flow Score."""
        if not self.enabled:
            return 0.0
        return self.magnitude * self.weight

    def agrees_with(self, side: Side) -> bool:
        return self.direction == side.sign

    def opposes(self, side: Side) -> bool:
        return self.direction == -side.sign


class FlowScore(FrozenModel):
    """Aggregate 0-100 score with its component breakdown preserved."""


    symbol: str
    ts: datetime
    components: dict[Component, ComponentScore]
    max_points: float = Field(gt=0, default=100.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> float:
        return round(sum(c.points for c in self.components.values()), 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available_points(self) -> float:
        """Total weight of enabled components. Less than max_points when a
        component is disabled for lack of data -- reported, never hidden."""
        return round(sum(c.weight for c in self.components.values() if c.enabled), 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quality(self) -> DataQuality:
        enabled = [c.quality for c in self.components.values() if c.enabled]
        return DataQuality.worst(*enabled) if enabled else DataQuality.MISSING

    def net_direction(self) -> int:
        """Sign of the weighted directional vote. 0 if balanced."""
        net = sum(c.direction * c.points for c in self.components.values())
        if net > 0:
            return 1
        if net < 0:
            return -1
        return 0

    def points_for(self, component: Component) -> float:
        c = self.components.get(component)
        return c.points if c else 0.0

    def opposing_points(self, side: Side) -> float:
        """Weighted points held by components opposing `side`."""
        return round(
            sum(c.points for c in self.components.values() if c.opposes(side)), 6
        )

    @model_validator(mode="after")
    def _check_bounds(self) -> "FlowScore":
        if self.score > self.max_points + 1e-6:
            raise ValueError(
                f"FlowScore {self.score} exceeds max_points {self.max_points}"
            )
        return self


# ---------------------------------------------------------------------------
# Signals / intents
# ---------------------------------------------------------------------------


class Signal(FrozenModel):
    """The signal engine's decision at one timestamp.

    WAITs are first-class and carry their blocking gate, which is what makes
    rejection analysis ("which gate rejects most trades") possible.
    """


    symbol: str
    ts: datetime
    action: SignalAction
    setup: SetupType | None = None
    flow_score: float | None = Field(default=None, ge=0.0, le=100.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    regime: Regime = Regime.UNKNOWN
    data_quality: DataQuality = DataQuality.MISSING
    wait_reason: WaitReason | None = None
    wait_detail: str = ""
    reasons: tuple[str, ...] = ()
    component_points: dict[Component, float] = Field(default_factory=dict)
    reference_price: float | None = None
    atr: float | None = None

    @property
    def side(self) -> Side | None:
        if self.action is SignalAction.LONG:
            return Side.LONG
        if self.action is SignalAction.SHORT:
            return Side.SHORT
        return None

    @model_validator(mode="after")
    def _check(self) -> "Signal":
        if self.action is SignalAction.WAIT:
            if self.wait_reason is None:
                raise ValueError("a WAIT signal must state a wait_reason")
        else:
            if self.wait_reason is not None:
                raise ValueError("a directional signal must not set wait_reason")
            if self.setup is None:
                raise ValueError("a directional signal must name a setup")
            if self.flow_score is None:
                raise ValueError("a directional signal must carry a flow_score")
        return self


class TradeIntent(FrozenModel):
    """A fully-specified order request. Produced only by the risk manager.

    Every price and size is final here; the execution model may fill it
    worse, partially, or not at all, but it may not change the plan.
    """


    symbol: str
    signal_ts: datetime
    side: Side
    setup: SetupType
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    target_price: float = Field(gt=0)
    size: float = Field(gt=0)
    risk_dollars: float = Field(gt=0)
    point_value: float = Field(gt=0)
    equity_at_signal: float = Field(gt=0)
    flow_score: float = Field(ge=0.0, le=100.0)
    regime: Regime
    max_hold_bars: int | None = Field(default=None, gt=0)
    reasons: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_distance(self) -> float:
        return abs(self.target_price - self.entry_price)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def planned_r_multiple(self) -> float:
        """Reward/risk as planned. This is the 'R' in '2R setup'."""
        return round(self.target_distance / self.stop_distance, 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def potential_reward_dollars(self) -> float:
        return round(self.target_distance * self.point_value * self.size, 10)

    @model_validator(mode="after")
    def _check_geometry(self) -> "TradeIntent":
        if self.stop_distance <= 0:
            raise ValueError("stop_price must differ from entry_price")
        if self.side is Side.LONG:
            if self.stop_price >= self.entry_price:
                raise ValueError("LONG stop must be below entry")
            if self.target_price <= self.entry_price:
                raise ValueError("LONG target must be above entry")
        else:
            if self.stop_price <= self.entry_price:
                raise ValueError("SHORT stop must be above entry")
            if self.target_price >= self.entry_price:
                raise ValueError("SHORT target must be below entry")
        expected_risk = self.stop_distance * self.point_value * self.size
        if not math.isclose(self.risk_dollars, expected_risk, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"risk_dollars {self.risk_dollars} inconsistent with "
                f"size*stop_distance*point_value = {expected_risk}"
            )
        return self


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class CostBreakdown(FrozenModel):
    """Itemized execution costs, so cost drag is measurable, not buried."""


    commission: float = Field(ge=0.0, default=0.0)
    exchange_fees: float = Field(ge=0.0, default=0.0)
    entry_slippage: float = Field(ge=0.0, default=0.0)
    exit_slippage: float = Field(ge=0.0, default=0.0)
    spread_cost: float = Field(ge=0.0, default=0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> float:
        return round(
            self.commission
            + self.exchange_fees
            + self.entry_slippage
            + self.exit_slippage
            + self.spread_cost,
            10,
        )


class TradeRecord(FrozenModel):
    """The audit row for one completed trade.

    Carries every field required by the research brief, plus the provenance
    fields (`split_label`, `config_hash`, `data_hash`) without which a
    performance number cannot be attributed to anything.
    """


    # identity / provenance
    trade_id: str
    run_id: str
    symbol: str
    config_hash: str
    data_hash: str = ""
    split_label: SplitPhase | None = None

    # timing
    signal_ts: datetime
    entry_ts: datetime
    exit_ts: datetime
    bars_held: int = Field(ge=0, default=0)
    session: Session = Session.CLOSED

    # plan
    side: Side
    setup: SetupType
    planned_entry: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    target_price: float = Field(gt=0)

    # execution
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    size: float = Field(gt=0)
    filled_size: float = Field(gt=0)
    point_value: float = Field(gt=0)
    exit_reason: ExitReason
    ambiguous_fill: bool = False

    # money
    risk_dollars: float = Field(gt=0)
    gross_pnl: float
    costs: CostBreakdown = Field(default_factory=CostBreakdown)
    pnl: float  # net of costs

    # excursions
    mae_r: float = Field(le=0.0, default=0.0, description="Max adverse excursion in R (<=0).")
    mfe_r: float = Field(ge=0.0, default=0.0, description="Max favourable excursion in R (>=0).")

    # scores at entry
    flow_score: float = Field(ge=0.0, le=100.0)
    options_flow_score: float = Field(ge=0.0, default=0.0)
    order_flow_score: float = Field(ge=0.0, default=0.0)
    structure_score: float = Field(ge=0.0, default=0.0)
    liquidity_score: float = Field(ge=0.0, default=0.0)
    momentum_score: float = Field(ge=0.0, default=0.0)

    # context at entry
    regime: Regime
    volatility: float | None = Field(default=None, description="ATR at entry, price units.")
    vol_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    data_quality: DataQuality = DataQuality.GOOD

    # narrative
    entry_reason: str = ""
    exit_reason_detail: str = ""

    # --- derived -------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def r_multiple(self) -> float:
        """Realized R: net PnL divided by the risk committed at entry."""
        return round(self.pnl / self.risk_dollars, 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def planned_r_multiple(self) -> float:
        stop_distance = abs(self.planned_entry - self.stop_price)
        if stop_distance <= 0:
            return 0.0
        return round(abs(self.target_price - self.planned_entry) / stop_distance, 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def holding_time_seconds(self) -> float:
        return (self.exit_ts - self.entry_ts).total_seconds()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_drag_r(self) -> float:
        """Costs expressed in R. Makes the cost hurdle directly comparable
        to the win rate required to break even."""
        return round(self.costs.total / self.risk_dollars, 6)

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def is_loss(self) -> bool:
        return self.pnl < 0

    @property
    def is_scratch(self) -> bool:
        return self.pnl == 0

    def r_label_is_honest(self, tolerance: float = 0.05) -> bool:
        """True if the setup's name matches the planned reward/risk.

        Guards against the failure mode where a 'SCALP_1R' actually targets
        0.4R against a full-width stop, producing a flattering win rate.
        """
        expected = {
            SetupType.SCALP_1R: 1.0,
            SetupType.SETUP_2R: 2.0,
            SetupType.DIRECTIONAL_3R: 3.0,
        }[self.setup]
        if self.setup is SetupType.DIRECTIONAL_3R:
            return self.planned_r_multiple >= expected - tolerance
        return abs(self.planned_r_multiple - expected) <= tolerance

    @model_validator(mode="after")
    def _check(self) -> "TradeRecord":
        if self.entry_ts < self.signal_ts:
            raise ValueError(
                "entry_ts precedes signal_ts: a fill cannot happen before its signal"
            )
        if self.exit_ts < self.entry_ts:
            raise ValueError("exit_ts precedes entry_ts")
        if self.filled_size > self.size + 1e-9:
            raise ValueError("filled_size exceeds requested size")
        expected_gross = (
            self.side.sign
            * (self.exit_price - self.entry_price)
            * self.point_value
            * self.filled_size
        )
        if not math.isclose(self.gross_pnl, expected_gross, rel_tol=1e-6, abs_tol=1e-4):
            raise ValueError(
                f"gross_pnl {self.gross_pnl} inconsistent with price path "
                f"(expected {expected_gross})"
            )
        expected_net = self.gross_pnl - self.costs.total
        if not math.isclose(self.pnl, expected_net, rel_tol=1e-6, abs_tol=1e-4):
            raise ValueError(
                f"pnl {self.pnl} != gross_pnl - costs ({expected_net})"
            )
        return self


class EquityPoint(FrozenModel):
    """Equity curve sample, including open risk so heat is auditable."""


    ts: datetime
    equity: float
    realized_equity: float
    open_positions: int = Field(ge=0, default=0)
    open_risk_dollars: float = Field(ge=0.0, default=0.0)
    high_water_mark: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def drawdown_pct(self) -> float:
        if self.high_water_mark <= 0:
            return 0.0
        return round((self.equity - self.high_water_mark) / self.high_water_mark, 10)
