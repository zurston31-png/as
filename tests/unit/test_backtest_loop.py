"""Adversarial tests for the event loop, the ledger, the journal and the database.

Organized around the ways this layer can be quietly wrong. Every one of them
produces a plausible-looking number rather than an exception, which is why
they are tested structurally rather than by eyeballing a report.

**n = 0 is the live case, not an edge case.** Phase 6 committed the finding
(f952d3e) that this system takes NO trades: the highest Flow Score measured
over 2500 bars is 65.23 against a lowest setup threshold of 70.0, so
`score_below_threshold` is the terminal gate. A run today must therefore
complete and produce a valid EMPTY result. The first section below is about
nothing else: that counts and sums -- which are defined over an empty set --
are reported, that no ratio is reported at any n anywhere in `backtest/`, and
that an exit reason with no trades is ABSENT from the cost mapping rather than
present with 0.0. "Profit factor = inf" and "win rate = 0%" are the two
fabrications this section exists to make impossible.

**Section 9's order IS the specification, and every violation of it flatters
the result silently.** Marking after the exits lets a kill switch read last
bar's equity; entering before exiting lets the limit check see the wrong open
heat and the wrong position count; a fill resolving on the bar that produced
the signal has used that bar's future. None of these raises on its own, and a
test that only inspects the result cannot see the order the result came from.
So the order is made OBSERVABLE here: a stub engine records what the limit
manager showed it at the instant it was asked, on a bar where a position exits
and a new signal fires, and the test asserts it saw the post-exit book.

**A bar the position held and nobody looked at.** The lookahead invariant has
a mirror image that is just as dangerous: a bar the position was open for that
no exit check ever inspected. An order with a multi-bar wait horizon fills on
an early candidate while the loop stands on the last one, and the bars in
between belonged to the position. A stop hit there, unseen, exits later at a
price the market never offered -- a realized loss reported as a win. That was
a real bug in this module and `test_every_bar_a_position_held_is_checked_for
_its_stop` is its regression test.

**R that is not the risk that was committed.** Everything downstream divides
by `risk_dollars`. The ledger must use the size that actually FILLED and the
price that actually PRINTED, because a half fill committed half the risk and
the distance from where the position really is to where the stop really is IS
the risk.

**A drawdown that is positive.** `EquityPoint.drawdown_pct` is defined as the
fraction BELOW the high-water mark. A point whose recorded mark is below its
own equity reports a positive drawdown, which is not a drawdown. That too was
a real bug, on the one sample every total return is read off: the last.

**Determinism that is not reproducibility.** Same config hash plus same data
hash must give bit-identical results. That fails in two directions: a
timestamp or a uuid inside the identity of a result, and state carried from a
previous run into the next one. Both are tested, and the fingerprint is itself
tested for sensitivity -- a digest that never changes is not a determinism
test, it is a decoration.

**Storage is not hashing.** `canonical_json` renders floats as fixed-width
strings so a digest is portable, and a value stored with it reads back as the
STRING "7.200000000000e-01". That is a committed bug fix, so the round trip is
tested on exact float values, and on the other silent value change in this
layer: SQLite binds a non-finite double as NULL.

NQ is used throughout because its 0.25 tick and 20-dollar point value make the
arithmetic exact: a 10-point stop is 40 ticks and costs 200 dollars per
contract, so two contracts commit R = 400 dollars. Every expected number below
is derived by hand from those two, with the arithmetic in a comment.

The config objects are built LOCALLY, as `test_risk.py` builds them, and not
from `config/defaults.yaml`: a test coupled to the YAML fails whenever an
unrelated section is edited, and its expected values change silently when a
default moves. The single exception is flagged in its own docstring -- the
test that the COMMITTED configuration takes no trades, where the YAML is the
subject rather than a dependency.

Nothing here lowers a threshold, widens a gate or constructs a configuration
that manufactures trades. The entry path is unreachable with the committed
parameters, so it is reached with a stub engine emitting a hand-built
`EntryPlan`. That fabricates no performance claim: it exists to prove that a
fill lands on a later bar, that every bar the position held is accounted for,
that the ledger's arithmetic closes, and that the record round-trips.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from flow_model.backtest.execution import ExecutionModel
from flow_model.backtest.fills import EntryFill, ExitFill, FillStatus
from flow_model.backtest.journal import Journal, JournalError, OrderRecord, SignalRecord
from flow_model.backtest.ledger import (
    EntryContext,
    Ledger,
    LedgerError,
    LedgerSnapshot,
    SessionFlattenPolicy,
)
from flow_model.backtest.loop import (
    MANDATORY_PHASES,
    SECTION_9_ORDER,
    BacktestError,
    BacktestLoop,
    LoopOrderError,
    LoopPhase,
    PositionManager,
    RunProvenance,
    RunResult,
    StoreMarketSource,
    VolatilityTracker,
    _PhaseTracker,
)
from flow_model.config.schema import (
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    FlowModelConfig,
    RiskConfig,
    SessionFilterConfig,
    SetupConfig,
    StructureLevelConfig,
)
from flow_model.core.contracts import (
    Bar,
    CostBreakdown,
    EquityPoint,
    FeatureVector,
    Signal,
    TradeIntent,
    TradeRecord,
)
from flow_model.core.enums import (
    Component,
    DataQuality,
    ExitReason,
    InstrumentType,
    Regime,
    Session,
    SetupType,
    Side,
    SignalAction,
    SplitPhase,
    WaitReason,
)
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data import DataStore
from flow_model.data.market_view import MarketView
from flow_model.data.series import BarSeries, to_ns
from flow_model.data.store import SymbolData
from flow_model.database.schema import SIGNAL_COLUMNS, TABLES, create_statements
from flow_model.database.store import Database, DatabaseError, signal_row
from flow_model.risk.limits import RiskLimitManager
from flow_model.signals.engine import EngineDecision, EntryPlan
from flow_model.signals.gates import GateResult, GateStage

UTC = timezone.utc
INTERVAL = 300
T0 = datetime(2016, 1, 5, 15, 0, tzinfo=UTC)

#: NQ, written out so the arithmetic below is reading a fixed target.
#: point_value = tick_value / tick_size = 5.0 / 0.25 = 20.0 dollars per point.
TICK = 0.25
POINT_VALUE = 20.0
NQ = InstrumentSpec(
    symbol="NQ",
    instrument_type=InstrumentType.FUTURE,
    tick_size=TICK,
    tick_value=5.0,
    commission_per_side=2.25,
    exchange_fee_per_side=0.37,
    typical_spread_ticks=1.0,
    min_spread_ticks=1.0,
)

#: The plan every stub signal carries: a LONG at 13000 with the stop 10 points
#: below and the target 10 points above, two contracts.
#:   stop_distance  = |13000 - 12990|      = 10 points = 40 ticks
#:   risk_per_unit  = 10 * 20              = 200 dollars
#:   R              = 200 * 2 contracts    = 400 dollars
ENTRY = 13000.0
STOP = 12990.0
TARGET = 13010.0
SIZE = 2.0
R_DOLLARS = 400.0

#: The measured Phase 7a cost hurdle for one NQ round turn at this geometry and
#: a vol multiple of 1.25. Used as the INPUT to the reporting tests, never
#: re-derived there: what is under test is that the drag is reported per exit
#: reason and that the two reasons differ, not the cost model itself.
STOP_EXIT_COSTS = 47.74      # 47.74 / 400 = 0.11935R
TARGET_EXIT_COSTS = 21.49    # 21.49 / 400 = 0.0537250R


# ---------------------------------------------------------------------------
# Local builders. No YAML, no shared fixtures: every value under test is here.
# ---------------------------------------------------------------------------


def setup_config(**over) -> SetupConfig:
    """SCALP_1R with the committed 70.0 threshold written out explicitly.

    The threshold is never moved by any test in this file -- it is here so
    that moving it in the YAML cannot change what these tests measure.
    """
    fields = dict(
        setup=SetupType.SCALP_1R,
        reward_risk=1.0,
        min_flow_score=70.0,
        allowed_regimes=(Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.LOW_VOL),
        stop_atr_multiple=1.0,
        min_stop_ticks=6,
        max_stop_atr_multiple=3.0,
        max_hold_bars=24,
        min_reward_risk=0.9,
    )
    fields.update(over)
    return SetupConfig(**fields)


def risk_config(**over) -> RiskConfig:
    """Section 8's limits, spelled out. Nothing here is read from the YAML."""
    fields = dict(
        starting_equity=100_000.0,
        risk_per_trade_pct=0.005,
        max_risk_per_trade_pct=0.01,
        daily_loss_limit_pct=0.02,
        weekly_loss_limit_pct=0.05,
        max_consecutive_losses=4,
        cooldown_minutes_after_loss=0.0,
        cooldown_minutes_after_limit=1440.0,
        max_open_positions=1,
        max_positions_per_symbol=1,
        max_portfolio_heat_pct=0.02,
        kill_switch_drawdown_pct=0.15,
        kill_switch_is_terminal=True,
        compound_equity=True,
    )
    fields.update(over)
    return RiskConfig(**fields)


def config(*, spec: InstrumentSpec | None = None, **over) -> FlowModelConfig:
    """A whole FlowModelConfig assembled locally from explicit values."""
    instrument = spec or NQ
    fields = dict(
        instruments={instrument.symbol: instrument},
        setups={SetupType.SCALP_1R: setup_config()},
        data=DataConfig(primary_interval_seconds=INTERVAL),
        risk=risk_config(),
        execution=ExecutionConfig(),
        session=SessionFilterConfig(flatten_at_session_close=False),
        backtest=BacktestConfig(symbols=(instrument.symbol,), seed=1234),
        levels=StructureLevelConfig(),
    )
    fields.update(over)
    return FlowModelConfig(**fields)


def bar(
    index: int,
    *,
    open_: float = 13000.0,
    high: float = 13002.0,
    low: float = 12998.0,
    close: float = 13000.0,
    volume: float = 100_000.0,
) -> Bar:
    """One flat NQ bar, `index` intervals after T0. Volume is large enough that
    `max_participation_rate` never caps a two-contract order."""
    return Bar(
        symbol="NQ",
        close_ts=T0 + timedelta(seconds=INTERVAL * index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        interval_seconds=INTERVAL,
    )


def store_of(bars) -> DataStore:
    store = DataStore(latency_seconds=0.0)
    store.add(
        SymbolData(
            symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: BarSeries.from_bars(bars)}
        )
    )
    return store


def plan_at(ts: datetime, *, equity: float = 100_000.0, **over) -> EntryPlan:
    """The hand-built 1R plan. R = 10 points * 20 dollars * 2 = 400 dollars."""
    fields = dict(
        symbol="NQ",
        signal_ts=ts,
        side=Side.LONG,
        setup=SetupType.SCALP_1R,
        entry_price=ENTRY,
        stop_price=STOP,
        target_price=TARGET,
        size=SIZE,
        risk_dollars=R_DOLLARS,
        point_value=POINT_VALUE,
        equity_at_signal=equity,
        stop_distance=10.0,
        stop_ticks=40.0,
        planned_r_multiple=1.0,
        achievable_rr=1.5,
        structural_stop_price=STOP,
        structural_target_price=TARGET,
        atr=10.0,
        max_hold_bars=24,
        flow_score=71.0,
        regime=Regime.TRENDING_UP,
        reasons=("test: a constructed plan, not a measured one",),
    )
    fields.update(over)
    return EntryPlan(**fields)


def intent_at(ts: datetime, *, size: float = SIZE) -> TradeIntent:
    return plan_at(ts).to_trade_intent().replace(
        size=size, risk_dollars=10.0 * POINT_VALUE * size
    )


def entry_fill(
    intent: TradeIntent,
    *,
    price: float = ENTRY,
    size: float = SIZE,
    ts: datetime | None = None,
    costs: CostBreakdown | None = None,
) -> EntryFill:
    return EntryFill(
        status=FillStatus.FILLED if size >= intent.size else FillStatus.PARTIAL,
        symbol=intent.symbol,
        side=intent.side,
        signal_ts=intent.signal_ts,
        submitted_ts=intent.signal_ts,
        eligible_ts=intent.signal_ts,
        order_type="market",
        fill_ts=ts or (intent.signal_ts + timedelta(seconds=INTERVAL)),
        fill_bar_offset=1,
        reference_price=price,
        planned_price=intent.entry_price,
        requested_size=intent.size,
        filled_size=size,
        costs=costs if costs is not None else CostBreakdown(commission=4.5, spread_cost=5.0),
    )


def exit_fill_at(
    price: float,
    ts: datetime,
    *,
    reason: ExitReason = ExitReason.TARGET,
    size: float = SIZE,
) -> ExitFill:
    return ExitFill(
        resolved=True,
        symbol="NQ",
        side=Side.LONG,
        exit_reason=reason,
        fill_ts=ts,
        reference_price=price,
        size=size,
    )


def entry_context(**over) -> EntryContext:
    fields = dict(
        flow_score=71.0,
        regime=Regime.TRENDING_UP,
        component_points={Component.STRUCTURE: 15.0, Component.LIQUIDITY: 9.0},
        volatility=10.0,
        vol_percentile=0.5,
        data_quality=DataQuality.GOOD,
        session=Session.RTH_MID,
        entry_reason="constructed",
    )
    fields.update(over)
    return EntryContext(**fields)


def ledger_of(cfg: FlowModelConfig | None = None, **kwargs) -> Ledger:
    return Ledger(cfg or config(), run_id="run-test", **kwargs)


def trade_record(**over) -> TradeRecord:
    """A self-consistent TradeRecord built locally.

    `gross_pnl` and `pnl` are derived from the price path so a test cannot
    construct an internally inconsistent record -- `TradeRecord`'s own
    validator would reject it anyway, which is the point of deriving them
    here rather than passing them in.
    """
    side: Side = over.pop("side", Side.LONG)
    entry: float = over.pop("entry_price", ENTRY)
    stop: float = over.pop("stop_price", STOP)
    target: float = over.pop("target_price", TARGET)
    exit_price: float = over.pop("exit_price", TARGET)
    size: float = over.pop("size", 1.0)
    point_value: float = over.pop("point_value", POINT_VALUE)
    costs: CostBreakdown = over.pop("costs", CostBreakdown())
    signal_ts: datetime = over.pop("signal_ts", T0)
    gross = side.sign * (exit_price - entry) * point_value * size
    fields = dict(
        trade_id="t1",
        run_id="run-test",
        symbol="NQ",
        config_hash="cfg",
        data_hash="dat",
        split_label=SplitPhase.TRAIN,
        signal_ts=signal_ts,
        entry_ts=signal_ts + timedelta(seconds=INTERVAL),
        exit_ts=signal_ts + timedelta(seconds=2 * INTERVAL),
        bars_held=1,
        session=Session.RTH_MID,
        side=side,
        setup=SetupType.SCALP_1R,
        planned_entry=entry,
        stop_price=stop,
        target_price=target,
        entry_price=entry,
        exit_price=exit_price,
        size=size,
        filled_size=size,
        point_value=point_value,
        exit_reason=ExitReason.TARGET,
        risk_dollars=abs(entry - stop) * point_value * size,
        gross_pnl=gross,
        costs=costs,
        pnl=gross - costs.total,
        flow_score=71.0,
        regime=Regime.TRENDING_UP,
    )
    fields.update(over)
    return TradeRecord(**fields)


def wait_signal(ts: datetime, reason: WaitReason, **over) -> Signal:
    fields = dict(
        symbol="NQ",
        ts=ts,
        action=SignalAction.WAIT,
        wait_reason=reason,
        wait_detail="constructed",
        data_quality=DataQuality.GOOD,
    )
    fields.update(over)
    return Signal(**fields)


def run_result_of(trades=(), equity=(), **over) -> RunResult:
    """A RunResult with the minimum provenance, for reporting-only tests."""
    provenance = RunProvenance(
        run_id="run-test",
        symbol="NQ",
        config_hash="cfg",
        data_hash="dat",
        seed=1234,
        created_at=T0,
        split_label=SplitPhase.TRAIN,
    )
    return RunResult(
        provenance=provenance, trades=tuple(trades), equity=tuple(equity), **over
    )


# ---------------------------------------------------------------------------
# The stub engine. A test double for the ENTRY PATH only.
# ---------------------------------------------------------------------------


class StubEngine:
    """Emits WAIT on every bar except the ones named, and records what it saw.

    It reports no score it measured and no performance it achieved. The real
    engine's verdict on the committed configuration -- no trades -- is tested
    separately against the real config and is not altered here.

    `observed` is what makes section 9's ORDER observable: on every bar it
    records what the `RiskLimitManager` showed at the instant the engine was
    asked. Exits run before the engine is called, so on a bar where a position
    closed, these rows must already reflect the closure.
    """

    def __init__(
        self,
        cfg: FlowModelConfig,
        signal_indices=(),
        *,
        equity: float = 100_000.0,
        plan_factory=None,
        wait_reason: WaitReason = WaitReason.NO_SETUP_MATCH,
        wait_stage: GateStage = GateStage.SETUP,
        flow_score: float | None = None,
    ) -> None:
        self.config = cfg
        self.symbol = "NQ"
        self.spec = cfg.spec("NQ")
        self.calendar = None
        self.signal_indices = tuple(signal_indices)
        self.equity = equity
        self.plan_factory = plan_factory or plan_at
        self.wait_reason = wait_reason
        self.wait_stage = wait_stage
        self.flow_score = flow_score
        self.observed: list[dict] = []
        self.views: list[MarketView] = []

    def session_date(self, ts: datetime) -> date:
        return ts.date()

    def _index(self, ts: datetime) -> int:
        return round((ts - T0).total_seconds() / INTERVAL)

    def decide(self, view, *, equity=None, limits=None) -> EngineDecision:
        ts = view.now
        self.views.append(view)
        self.observed.append(
            dict(
                ts=ts,
                equity=equity,
                open_positions=None if limits is None else limits.open_positions(),
                heat=None if limits is None else limits.portfolio_heat(),
                realized=None if limits is None else limits.realized_for_session(ts.date()),
                consecutive=None if limits is None else limits.consecutive_losses,
                killed=None if limits is None else limits.killed,
            )
        )
        features = FeatureVector(
            symbol="NQ",
            ts=ts,
            values={"atr": 10.0, "atr_percentile": 0.5},
            quality_by_key={"atr": DataQuality.GOOD, "atr_percentile": DataQuality.GOOD},
        )
        if self._index(ts) not in self.signal_indices:
            return EngineDecision(
                signal=wait_signal(
                    ts, self.wait_reason, flow_score=self.flow_score
                ),
                gate_results=(
                    GateResult(
                        stage=self.wait_stage,
                        passed=False,
                        wait_reason=self.wait_reason,
                        detail="stub: not a signal bar",
                    ),
                ),
                features=features,
            )
        return EngineDecision(
            signal=Signal(
                symbol="NQ",
                ts=ts,
                action=SignalAction.LONG,
                setup=SetupType.SCALP_1R,
                flow_score=71.0,
                confidence=0.5,
                regime=Regime.TRENDING_UP,
                data_quality=DataQuality.GOOD,
                reference_price=ENTRY,
                atr=10.0,
                component_points={Component.STRUCTURE: 15.0},
            ),
            gate_results=(GateResult(stage=GateStage.RISK, passed=True),),
            plan=self.plan_factory(ts, equity=self.equity),
            features=features,
        )


class GatedStubEngine(StubEngine):
    """A stub that consults the limit manager the way the real engine does.

    `signals/engine.py` runs `RiskLimitManager.check` as the last gate and
    turns a breach into a WAIT. Modelling that is what lets a test assert
    which book the gating check SAW -- the point of section 9's
    exits-before-entries rule.
    """

    def decide(self, view, *, equity=None, limits=None) -> EngineDecision:
        decision = super().decide(view, equity=equity, limits=limits)
        if decision.signal.action is SignalAction.WAIT or limits is None:
            return decision
        plan = decision.plan
        verdict = limits.check(
            ts=view.now,
            session_date=self.session_date(view.now),
            equity=equity,
            symbol="NQ",
            prospective_risk=plan.risk_dollars,
        )
        if not verdict.allowed:
            return EngineDecision(
                signal=wait_signal(
                    view.now, verdict.wait_reason or WaitReason.RISK_LIMIT
                ).replace(wait_detail=verdict.breach),
                gate_results=(
                    GateResult(
                        stage=GateStage.RISK,
                        passed=False,
                        wait_reason=verdict.wait_reason or WaitReason.RISK_LIMIT,
                        detail=verdict.breach,
                    ),
                ),
                features=decision.features,
                limits=verdict,
            )
        return EngineDecision(
            signal=decision.signal,
            gate_results=decision.gate_results,
            plan=plan,
            features=decision.features,
            limits=verdict,
        )


def run_loop(
    cfg: FlowModelConfig,
    bars,
    signal_indices=(),
    *,
    engine_cls=StubEngine,
    clock_ns=None,
    atr_median_window: int = 3,
    **engine_kwargs,
):
    """Walk `bars` with a stub engine. Returns `(RunResult, engine, loop)`.

    `atr_median_window=3` is passed explicitly so three warm-up bars suffice:
    the volatility tracker refuses to price a fill until its window is FULL
    (an unmeasured ratio is not a ratio of 1.0), and the window length is a
    config value this file does not want to depend on.
    """
    store = store_of(bars)
    engine = engine_cls(cfg, signal_indices, **engine_kwargs)
    loop = BacktestLoop(
        cfg, "NQ", engine=engine, git_commit="", git_dirty=False,
        atr_median_window=atr_median_window,
    )
    result = loop.run(StoreMarketSource(store, "NQ"), clock_ns=clock_ns)
    return result, engine, loop


def warm_bars(count: int = 4) -> list[Bar]:
    """Flat bars 0..count-1, enough to fill a three-observation ATR window."""
    return [bar(i) for i in range(count)]


#: Names no module in `backtest/` may compute, at any n. Each is a ratio whose
#: denominator can be zero, and each has an honest answer ("undefined, n=0")
#: that only `analytics/` is in a position to say.
FORBIDDEN_RATIOS = (
    "win_rate",
    "loss_rate",
    "profit_factor",
    "expectancy",
    "sharpe",
    "sortino",
    "avg_win",
    "avg_loss",
    "payoff_ratio",
)


# ---------------------------------------------------------------------------
# The premise: the arithmetic the rest of the file rests on
# ---------------------------------------------------------------------------


def test_the_nq_arithmetic_the_rest_of_this_file_rests_on():
    """Assert the instrument's own numbers before anything divides by them.

    Every expected value below is derived from these two, so a silent change
    to either would make the rest of the file test nothing.
    """
    assert NQ.tick_size == TICK
    assert NQ.point_value == POINT_VALUE          # 5.0 / 0.25
    assert abs(ENTRY - STOP) / TICK == 40.0        # a 10-point stop is 40 ticks
    assert abs(ENTRY - STOP) * POINT_VALUE * SIZE == R_DOLLARS   # 10*20*2 = 400
    # The plan calls itself 1R, and |target - entry| == |entry - stop| is what
    # entitles it to: 13010 - 13000 == 13000 - 12990.
    assert TARGET - ENTRY == ENTRY - STOP
    assert plan_at(T0).to_trade_intent().planned_r_multiple == 1.0


# ---------------------------------------------------------------------------
# n = 0: the live case. This section comes first because it runs first.
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_run():
    """A run of 40 bars on which every bar is a WAIT for `score_below_threshold`.

    The committed finding, modelled exactly: a flow score of 65.23 against a
    threshold of 70.0 on every bar. The stub keeps the engine's thresholds out
    of it, so what is under test is the loop, the ledger, the journal and the
    database over an empty trade set -- not Phase 6's parameters.
    """
    result, engine, _ = run_loop(
        config(),
        [bar(i) for i in range(40)],
        (),
        wait_reason=WaitReason.SCORE_BELOW_THRESHOLD,
        wait_stage=GateStage.FLOW_SCORE,
        flow_score=65.23,
    )
    return result


def test_a_run_that_takes_no_trades_completes_and_produces_a_valid_result(empty_run):
    """The empty result is the LIVE case. It must be a result, not a crash and
    not a gap: a curve that exists, an equity that has not moved, and counts
    that are zero because nothing happened rather than because nothing was
    counted."""
    assert empty_run.is_empty is True
    assert empty_run.trades == ()
    assert empty_run.trade_count == 0
    assert empty_run.orders == ()
    assert empty_run.bars_evaluated == 40
    assert len(empty_run.signals) == 40
    assert len(empty_run.equity) == 40
    assert empty_run.open_at_end == 0
    assert empty_run.unresolved_orders == 0
    assert empty_run.final is not None
    assert empty_run.final.equity == 100_000.0
    assert empty_run.final.realized_pnl == 0.0
    assert empty_run.final.closed_trades == 0
    assert empty_run.ledger_diagnostics["closed_trades"] == 0.0
    # Nothing moved, so every point sits exactly at the starting equity and at
    # the high-water mark: a flat curve, not an absent one.
    assert {point.equity for point in empty_run.equity} == {100_000.0}
    assert {point.drawdown_pct for point in empty_run.equity} == {0.0}
    assert any("took NO trades" in note for note in empty_run.notes)
    assert any("undefined, n=0" in note for note in empty_run.notes)


def test_no_ratio_is_reported_anywhere_in_the_backtest_at_any_n(empty_run):
    """A win rate over zero trades is UNDEFINED, not 0%. A profit factor with
    no losses is UNDEFINED, not infinity.

    Checked structurally, over every name the run publishes, rather than by
    reading one report: the refusal has to hold for the fields, the summary and
    the journal's own description, or some caller will find the one that
    slipped through and divide by zero in public.
    """
    published = (
        set(RunResult.model_fields)
        | set(empty_run.summary())
        | set(empty_run.rejection)
        | set(empty_run.cost_totals)
        | set(empty_run.ledger_diagnostics)
        | {name for name in dir(RunResult) if not name.startswith("_")}
    )
    for forbidden in FORBIDDEN_RATIOS:
        assert not any(forbidden in name for name in published), forbidden
    summary = empty_run.summary()
    assert "undefined" in summary["ratios"]
    assert "analytics" in summary["ratios"]
    # Sums and counts ARE defined over an empty set, and are reported.
    assert summary["trades"] == 0
    assert summary["wins"] == 0
    assert summary["losses"] == 0
    assert summary["scratches"] == 0
    assert summary["realized_pnl"] == 0.0
    assert empty_run.cost_totals["total"] == 0.0


def test_an_exit_reason_with_no_trades_is_absent_rather_than_zero(empty_run):
    """"No trade has ever exited this way" and "trades exit this way for free"
    are different facts. A mapping that answers the first with 0.0 answers the
    second by accident."""
    assert empty_run.cost_drag_by_exit_reason == {}
    assert empty_run.rejection["exit_reasons"] == {}
    for reason in ExitReason:
        assert reason.value not in empty_run.cost_drag_by_exit_reason
    # And there is no single aggregate drag figure to quote instead.
    assert not any(
        "drag" in name and "exit_reason" not in name
        for name in RunResult.model_fields
    )


def test_the_terminal_gate_is_none_over_no_waits_and_never_quoted_without_its_n():
    """`terminal_gate` must not pick a gate when no WAIT has been recorded, and
    must not hand back a name without the count behind it -- "the terminal gate
    is score_below_threshold" is a different claim at n=6 and at n=6000."""
    journal = Journal(run_id="run-test")
    assert journal.terminal_gate() is None
    assert journal.rejection_report()["terminal_gate"] is None
    journal.record_signal(wait_signal(T0, WaitReason.SCORE_BELOW_THRESHOLD))
    gate = journal.terminal_gate()
    assert gate == ("score_below_threshold", 1)
    assert journal.rejection_report()["terminal_gate"] == {
        "reason": "score_below_threshold",
        "count": 1,
    }


def test_the_terminal_gate_breaks_a_tie_the_same_way_every_time():
    """Two gates with equal counts must not resolve by dict order, or two runs
    of the same data would report different terminal gates."""
    picks = set()
    for order in (
        (WaitReason.WARMUP, WaitReason.REGIME_BLOCKED),
        (WaitReason.REGIME_BLOCKED, WaitReason.WARMUP),
    ):
        journal = Journal(run_id="run-test")
        for index, reason in enumerate(order):
            journal.record_signal(
                wait_signal(T0 + timedelta(seconds=INTERVAL * index), reason)
            )
        picks.add(journal.terminal_gate())
    assert len(picks) == 1
    assert picks.pop()[1] == 1


def test_an_equity_curve_over_no_trades_still_has_its_endpoints(empty_run):
    """A run that measured nothing must still be measurable over something: a
    curve with no second endpoint has no span for a return to be measured
    over, and no timestamps for a drawdown series."""
    stamps = [point.ts for point in empty_run.equity]
    assert stamps == sorted(stamps)
    assert len(stamps) == len(set(stamps))      # no zero-length time step
    assert stamps[0] == T0
    assert stamps[-1] == T0 + timedelta(seconds=INTERVAL * 39)


def test_a_clock_with_no_bars_measures_nothing_and_divides_by_none_of_it():
    """The other n=0: no bars at all. Every count is zero and every optional is
    None -- not 0.0, which would read as a measurement of an empty run."""
    result, _, _ = run_loop(config(), warm_bars(4), (), clock_ns=[])
    assert result.is_empty is True
    assert result.bars_evaluated == 0
    assert result.equity == ()
    assert result.signals == ()
    assert result.provenance.bars_processed == 0
    assert result.provenance.first_ts is None
    assert result.provenance.last_ts is None
    assert result.cost_drag_by_exit_reason == {}
    assert result.rejection["terminal_gate"] is None
    assert result.summary()["trades"] == 0
    assert result.ledger_diagnostics["marks"] == 0.0


def test_exposure_is_undefined_rather_than_a_number_when_the_account_is_gone():
    """With equity at or below zero the ratio is not an exposure. Reporting one
    would be reporting a leverage figure about an account that no longer
    exists."""
    gone = LedgerSnapshot(
        ts=T0,
        equity=0.0,
        realized_equity=0.0,
        realized_pnl=-100_000.0,
        unrealized_pnl=0.0,
        open_entry_costs=0.0,
        sizing_equity=0.0,
        compound_equity=True,
        open_positions=1,
        closed_trades=3,
        open_risk_dollars=400.0,
        open_notional=520_000.0,
        high_water_mark=100_000.0,
    )
    assert gone.exposure_fraction is None
    assert gone.drawdown_fraction == 1.0          # (100000 - 0) / 100000
    alive = gone.replace(equity=50_000.0)
    assert alive.exposure_fraction == pytest.approx(520_000.0 / 50_000.0)
    assert alive.drawdown_fraction == pytest.approx(0.5)


def test_r_is_refused_rather_than_divided_by_zero_when_no_risk_was_committed():
    """R is `pnl / risk_dollars`. With no risk committed there is no R, and the
    position says so instead of returning a number."""
    ledger = ledger_of()
    position = ledger.open_position(
        intent=intent_at(T0), entry=entry_fill(intent_at(T0)), context=entry_context()
    )
    position.risk_dollars = 0.0                   # only reachable by force
    with pytest.raises(LedgerError, match="undefined"):
        position.r_at(ENTRY)


def test_the_empty_run_persists_and_reads_back_as_an_empty_run(tmp_path, empty_run):
    """The empty run is the one that will actually be stored, so it has to
    store: one `runs` row with its provenance and its rejection histogram,
    zero trade rows, and a count of zero that came from counting rather than
    from a failed write."""
    with Database(tmp_path / "empty.db") as database:
        run_id = database.write_run(empty_run)
        assert database.read_trades(run_id) == ()
        assert database.read_orders(run_id) == ()
        assert database.read_signals(run_id) == empty_run.signals
        assert database.read_equity(run_id) == empty_run.equity
        row = database.read_run(run_id)
        assert row is not None
        assert row["trade_count"] == 0
        assert row["bars_evaluated"] == 40
        assert row["result_fingerprint"] == empty_run.result_fingerprint()
        assert row["cost_drag"] == {}
        assert row["cost_totals"]["total"] == 0.0
        assert row["rejection"]["terminal_gate"] == {
            "reason": "score_below_threshold",
            "count": 40,
        }
        assert any("took NO trades" in note for note in row["notes"])
        counts = database.counts()
        assert counts["runs"] == 1
        assert counts["trades"] == 0
        assert counts["signals"] == 40
        # And the stored rows are readable as records, not as text.
        stored = database.read_signals(run_id)[0]
        assert stored.wait_reason is WaitReason.SCORE_BELOW_THRESHOLD
        assert isinstance(stored.flow_score, float)
        assert stored.flow_score == 65.23


def test_the_committed_configuration_still_takes_no_trades_over_real_bars():
    """The one test here that reads `config/defaults.yaml`, because the
    COMMITTED configuration is its subject rather than its dependency.

    Phase 6's finding (f952d3e) is that the highest Flow Score measured is
    65.23 against a lowest threshold of 70.0. If this test ever starts
    producing trades on untouched defaults, either a threshold moved or the
    score changed, and both are Phase 8's business and not a backtester
    change. 2016 bars: the sealed window (2023-01-01 to 2025-01-01) is neither
    requested nor opened.
    """
    from flow_model.config.loader import default_config
    from flow_model.data import SyntheticAdapter, SyntheticConfig, TradingCalendar

    cfg = default_config()
    calendar = TradingCalendar()
    adapter = SyntheticAdapter(
        config=SyntheticConfig(),
        seed=cfg.seed,
        spec_by_symbol=dict(cfg.instruments),
        interval_seconds=cfg.data.primary_interval_seconds,
        calendar=calendar,
    )
    store = DataStore(latency_seconds=cfg.data.data_latency_seconds)
    store.load(
        adapter,
        "NQ",
        date(2016, 1, 4),
        date(2016, 1, 8),
        primary_interval=cfg.data.primary_interval_seconds,
    )
    clock = store.timeline("NQ")[:300]
    result = BacktestLoop(
        cfg, "NQ", calendar=calendar, git_commit="", git_dirty=False
    ).run(StoreMarketSource.from_config(store, "NQ", cfg), clock_ns=clock)
    assert result.is_empty is True
    assert result.orders == ()
    assert result.final.equity == pytest.approx(float(cfg.risk.starting_equity))
    assert result.bars_evaluated == 300
    assert all(row.wait_reason is not None for row in result.signals)
    assert sum(result.rejection["wait_reasons"].values()) == 300
    assert result.cost_drag_by_exit_reason == {}


# ---------------------------------------------------------------------------
# Section 9's order IS the specification
# ---------------------------------------------------------------------------


def exit_then_signal_bars() -> list[Bar]:
    """Bars 0-3 warm up, 3 signals, 4 fills at the open, 5 stops out.

    Bar 5 is the bar the ordering is observable on: an open position exits on
    it AND a new signal fires on it.
    """
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13003.0, low=12999.0, close=13002.0))
    bars.append(bar(5, open_=13001.0, high=13002.0, low=12980.0, close=12985.0))
    bars.append(bar(6))
    bars.append(bar(7))
    return bars


def test_the_limit_check_that_gates_this_bar_sees_the_post_exit_book():
    """EXITS BEFORE ENTRIES, made observable.

    On bar 5 a position exits and a new signal fires. `max_open_positions=1`,
    so if the entry were sized before the exit the limit check would see one
    open position and one open R of heat, and would block the new entry with
    `max_open_positions`. The bar would then look like a risk rejection in
    section 12's histogram when the account was in fact flat -- and the
    opposite ordering error, with the position counted twice, breaches the
    limit for a bar.

    Hand-computed: the trade is a LONG filled at bar 4's open of 13000 and
    stopped at 12990, so gross = (12990 - 13000) * 20 * 2 = -400 dollars
    exactly, and the realized PnL the gating check must see is that minus the
    round turn's costs.
    """
    cfg = config()
    result, engine, _ = run_loop(
        cfg, exit_then_signal_bars(), (3, 5), engine_cls=GatedStubEngine
    )
    first = result.trades[0]
    assert first.exit_reason is ExitReason.STOP
    assert first.exit_ts == bar(5).close_ts
    assert first.gross_pnl == pytest.approx(-400.0)
    assert first.pnl == pytest.approx(first.gross_pnl - first.costs.total)
    assert first.costs.total > 0.0

    seen = {row["ts"]: row for row in engine.observed}
    at_exit_bar = seen[bar(5).close_ts]
    assert at_exit_bar["open_positions"] == 0        # the exit already happened
    assert at_exit_bar["heat"] == 0.0                # and released its heat
    assert at_exit_bar["realized"] == pytest.approx(first.pnl)
    assert at_exit_bar["consecutive"] == 1
    assert at_exit_bar["equity"] == pytest.approx(100_000.0 + first.pnl)

    # The new entry was therefore allowed and submitted, not blocked.
    rows = {row.ts: row for row in result.signals}
    directional = rows[bar(5).close_ts]
    assert directional.action is SignalAction.LONG
    assert directional.intent_built is True
    assert directional.order_submitted is True
    assert directional.limit_breach == ""
    assert len(result.orders) == 2
    assert "max_open_positions" not in result.limit_trips


def test_marking_runs_first_so_the_engine_is_handed_this_bars_equity():
    """MARK TO MARKET FIRST, on THIS bar's prices.

    A kill switch and a drawdown series read off last bar's equity fire a bar
    late, at a worse price than they claim. Bar 5 closes 5 points above bar
    4's close, so the equity the engine is handed on bar 5 must already
    include that move.

    Hand-computed: 5 points * 20 dollars * 2 contracts = 200 dollars of
    unrealized gain, against an equity reduced by the entry's costs, which are
    realized when they are PAID.
    """
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13002.0, low=12999.0, close=13000.0))
    bars.append(bar(5, open_=13000.0, high=13008.0, low=12999.0, close=13005.0))
    bars.append(bar(6))
    result, engine, _ = run_loop(config(), bars, (3,))
    entry_costs = result.orders[0].costs.total
    assert entry_costs > 0.0
    seen = {row["ts"]: row for row in engine.observed}
    # Bar 4: the fill resolves in step 6, AFTER the engine was asked, so the
    # position is not yet visible. That is section 9's order, not a gap.
    assert seen[bar(4).close_ts]["equity"] == pytest.approx(100_000.0)
    # Bar 5: marked at 13005 before anything else ran.
    assert seen[bar(5).close_ts]["equity"] == pytest.approx(
        100_000.0 - entry_costs + 200.0
    )


def test_the_engine_is_handed_a_market_view_built_at_this_bar_never_a_bar():
    """The firewall is the strongest safety property in the project and the
    loop must not route around it. `evaluate` takes a view built at `ts`; the
    bars go to the ledger and the execution model as arguments instead."""
    bars = warm_bars(6)
    result, engine, _ = run_loop(config(), bars, ())
    assert len(engine.views) == len(bars)
    for index, view in enumerate(engine.views):
        assert isinstance(view, MarketView)
        assert view.now == bars[index].close_ts
        view.assert_no_future_leak()
    assert result.provenance.bars_processed == len(bars)


def test_every_bar_records_a_signal_with_its_reason_and_its_blocking_stage():
    """Section 12 asks how often each gate fires. The answer cannot be
    reconstructed after the fact, so a row is written for every bar including
    the WAITs, and the row carries the STAGE as well as the reason -- a WAIT
    for `risk_limit` and a LONG whose order was never submitted are different
    outcomes."""
    result, _, _ = run_loop(
        config(),
        [bar(i) for i in range(12)],
        (),
        wait_reason=WaitReason.SCORE_BELOW_THRESHOLD,
        wait_stage=GateStage.FLOW_SCORE,
    )
    assert len(result.signals) == 12
    assert result.waits_dropped == 0
    assert all(row.action is SignalAction.WAIT for row in result.signals)
    assert all(row.wait_reason is WaitReason.SCORE_BELOW_THRESHOLD for row in result.signals)
    assert all(row.blocking_stage == "flow_score" for row in result.signals)
    assert all(row.stages_run == ("flow_score",) for row in result.signals)
    assert result.rejection["wait_reasons"] == {"score_below_threshold": 12}
    assert result.rejection["blocking_stages"] == {"flow_score": 12}
    assert result.rejection["bars_evaluated"] == 12


def test_dropping_wait_rows_never_drops_the_rejection_counts():
    """`record_wait_signals` governs ROW VOLUME, not the histogram. A rejection
    analysis that silently loses its denominator when a flag is flipped is
    worse than none: the remaining numbers still look like answers."""
    cfg = config()
    sparse = cfg.replace(backtest=cfg.backtest.replace(record_wait_signals=False))
    kept, _, _ = run_loop(cfg, [bar(i) for i in range(9)], ())
    dropped, _, _ = run_loop(sparse, [bar(i) for i in range(9)], ())
    assert len(kept.signals) == 9
    assert dropped.signals == ()
    assert dropped.waits_dropped == 9
    assert dropped.bars_evaluated == 9
    assert kept.rejection["wait_reasons"] == dropped.rejection["wait_reasons"]
    assert dropped.rejection["waits_dropped_by_config"] == 9
    assert dropped.provenance.record_wait_signals is False


@pytest.mark.parametrize(
    "sequence",
    [
        # Marking after the exits: the kill switch reads stale equity.
        (LoopPhase.UPDATE_OPEN_POSITIONS, LoopPhase.MARK_TO_MARKET),
        # Entering before exiting: the limits see the wrong book.
        (
            LoopPhase.MARK_TO_MARKET,
            LoopPhase.EVALUATE,
            LoopPhase.UPDATE_OPEN_POSITIONS,
        ),
        # Filling before the signal was even recorded.
        (
            LoopPhase.MARK_TO_MARKET,
            LoopPhase.UPDATE_OPEN_POSITIONS,
            LoopPhase.PROCESS_PENDING,
            LoopPhase.RECORD_SIGNAL,
        ),
        # The same step twice: a bar marked or filled twice.
        (LoopPhase.MARK_TO_MARKET, LoopPhase.MARK_TO_MARKET),
        (
            LoopPhase.MARK_TO_MARKET,
            LoopPhase.UPDATE_OPEN_POSITIONS,
            LoopPhase.EVALUATE,
            LoopPhase.RECORD_SIGNAL,
            LoopPhase.PROCESS_PENDING,
            LoopPhase.PROCESS_PENDING,
        ),
    ],
)
def test_a_reordered_or_repeated_step_raises_rather_than_flattering_the_result(
    sequence,
):
    """Every one of section 9's constraints fails SILENTLY in the flattering
    direction, so the order is checked at runtime on every bar rather than
    documented and hoped for."""
    tracker = _PhaseTracker()
    tracker.begin_bar(T0)
    with pytest.raises(LoopOrderError):
        for phase in sequence:
            tracker.enter(phase)


def test_a_bar_that_skipped_a_mandatory_step_is_caught_at_its_end():
    """Skipping is not catchable at the step -- a skipped step is simply a step
    that never ran -- so the tracker checks the set at `end_bar`. Only
    `build_and_submit` may be absent, and only because a WAIT bar submits
    nothing."""
    tracker = _PhaseTracker()
    tracker.begin_bar(T0)
    for phase in MANDATORY_PHASES:
        tracker.enter(phase)
    tracker.end_bar()                      # the legal skip
    assert tracker.bars == 1

    tracker.begin_bar(T0 + timedelta(seconds=INTERVAL))
    tracker.enter(LoopPhase.MARK_TO_MARKET)
    tracker.enter(LoopPhase.EVALUATE)
    tracker.enter(LoopPhase.RECORD_SIGNAL)
    tracker.enter(LoopPhase.PROCESS_PENDING)
    with pytest.raises(LoopOrderError, match="update_open_positions"):
        tracker.end_bar()


def test_a_step_outside_a_bar_and_a_bar_inside_a_bar_both_raise():
    """The tracker's own framing: a step that runs between bars has no bar to
    be ordered within, and a bar that begins before the last one finished means
    the loop body was re-entered."""
    tracker = _PhaseTracker()
    with pytest.raises(LoopOrderError, match="outside a bar"):
        tracker.enter(LoopPhase.MARK_TO_MARKET)
    tracker.begin_bar(T0)
    with pytest.raises(LoopOrderError, match="never finished"):
        tracker.begin_bar(T0 + timedelta(seconds=INTERVAL))


def test_section_9_s_order_is_carried_as_data_so_it_can_be_checked():
    """The order is the specification, so it is a tuple the tracker reads
    rather than a sequence of calls a reader has to trace."""
    assert [phase.value for phase in SECTION_9_ORDER] == [
        "mark_to_market",
        "update_open_positions",
        "evaluate",
        "record_signal",
        "build_and_submit",
        "process_pending",
    ]
    assert LoopPhase.BUILD_AND_SUBMIT not in MANDATORY_PHASES
    assert set(MANDATORY_PHASES) == set(SECTION_9_ORDER) - {LoopPhase.BUILD_AND_SUBMIT}


# ---------------------------------------------------------------------------
# The lookahead invariant, and its mirror image
# ---------------------------------------------------------------------------


def through_bars(count: int = 6, *, first: int = 4) -> list[Bar]:
    """Bars that trade through 13000 from both sides, so a market, limit or
    stop entry all fill at 13000 -- and neither the stop (12990) nor the
    target (13010) is reached."""
    bars = warm_bars(first)
    bars += [
        bar(index, open_=13000.0, high=13005.0, low=12995.0, close=13000.0)
        for index in range(first, first + count)
    ]
    return bars


@pytest.mark.parametrize(
    "execution",
    [
        ExecutionConfig(entry_order_type="market", latency_ms=0.0),
        # Latency longer than a bar: the eligibility floor moves out, and must
        # never move IN to the signal bar.
        ExecutionConfig(entry_order_type="market", latency_ms=400_000.0),
        ExecutionConfig(entry_order_type="market", entry_delay_bars=1),
        ExecutionConfig(entry_order_type="limit", max_entry_wait_bars=3),
        ExecutionConfig(entry_order_type="stop", max_entry_wait_bars=2),
        ExecutionConfig(entry_order_type="limit", max_entry_wait_bars=2, latency_ms=0.0),
    ],
)
def test_a_fill_can_never_resolve_on_the_bar_that_produced_the_signal(execution):
    """A fill on the signal bar has used that bar's own future.

    Checked for every entry type and every eligibility floor, because the
    guarantee comes from arithmetic (`min_fill_offset >= 1`) plus a window that
    physically drops the signal bar, and a regression in either would show up
    on exactly one of these combinations.
    """
    cfg = config(execution=execution)
    result, _, _ = run_loop(cfg, through_bars(8), (3,))
    # The scenario must actually fill, or the invariant below is vacuous.
    assert any(order.filled for order in result.orders)
    assert result.trades
    signal_ts = bar(3).close_ts
    for order in result.orders:
        assert order.signal_ts == signal_ts
        if order.fill_ts is not None:
            assert order.fill_ts > signal_ts
            assert order.fill_bar_offset >= 1
    for record in result.trades:
        assert record.entry_ts > record.signal_ts
        assert record.exit_ts >= record.entry_ts


def test_every_bar_a_position_held_is_checked_for_its_stop():
    """A bar the position was open for that no exit check ever inspected.

    This was a real bug. An order with a multi-bar wait horizon is held until
    its WHOLE horizon has closed -- correctly, so that a truncated window
    cannot report a miss on a horizon that never elapsed -- but it can FILL on
    an earlier candidate. Only the fill bar was then checked for an exit, so
    the bars between the fill and the loop's current position belonged to the
    trade and were inspected by nobody.

    The scenario: a resting limit at 13000 with `max_entry_wait_bars=3` fills
    at offset 1, offset 2 trades down to 12985 (clean through the 12990 stop),
    and offset 3 recovers to 13016. Before the fix the position survived
    untouched and exited as a TARGET at 13010 for +0.99R. The honest answer is
    a STOP at 12990:

        gross = (12990 - 13000) * 20 * 2 = -400 dollars = -1R
        mae   = (12985 - 13000) * 20 * 2 / 400 = -1.5R

    so the bug was worth 2.06R of fabricated performance on one trade, in the
    flattering direction, with nothing in the record to show for it.
    """
    cfg = config(execution=ExecutionConfig(entry_order_type="limit", max_entry_wait_bars=3))
    bars = warm_bars(4)
    bars.append(bar(4, open_=12999.0, high=13001.0, low=12995.0, close=12998.0))
    bars.append(bar(5, open_=12998.0, high=12999.0, low=12985.0, close=12988.0))
    bars.append(bar(6, open_=12990.0, high=13016.0, low=12989.0, close=13015.0))
    bars.append(bar(7, open_=13015.0, high=13020.0, low=13014.0, close=13018.0))
    result, _, _ = run_loop(cfg, bars, (3,))

    assert len(result.orders) == 1
    assert result.orders[0].fill_ts == bar(4).close_ts
    assert len(result.trades) == 1
    record = result.trades[0]
    assert record.exit_reason is ExitReason.STOP
    assert record.exit_ts == bar(5).close_ts
    assert record.exit_price == pytest.approx(STOP)
    assert record.gross_pnl == pytest.approx(-400.0)
    assert record.r_multiple < -1.0                  # -1R plus the stop's costs
    assert record.mae_r == pytest.approx(-1.5)       # bar 5's low, not bar 4's
    assert record.bars_held == 1                     # one bar closed after the entry
    assert result.rejection["exit_reasons"] == {"STOP": 1}


def test_the_replayed_bars_are_counted_once_and_only_against_their_own_position():
    """The holding time is a property of the bar, counted once per position.

    Re-running the whole book when a position joins it would charge every
    OTHER open position for the same bars a second time and fire their time
    stops early -- a trade cut short, invisible afterwards. With two positions
    whose fills are five bars apart, each one's `bars_held` must equal exactly
    the number of clock bars that closed after its own entry bar:

        A fills on bar 4, data ends on bar 11  ->  bars 5..11  = 7
        B fills on bar 8, data ends on bar 11  ->  bars 9..11  = 3
    """
    cfg = config(
        execution=ExecutionConfig(entry_order_type="limit", max_entry_wait_bars=3),
        risk=risk_config(
            max_open_positions=2, max_positions_per_symbol=2, max_portfolio_heat_pct=0.01
        ),
    )
    bars = warm_bars(4)
    bars += [
        bar(index, open_=13000.0, high=13005.0, low=12995.0, close=13000.0)
        for index in range(4, 12)
    ]
    result, _, _ = run_loop(
        cfg, bars, (3, 7), plan_factory=lambda ts, *, equity: plan_at(ts, equity=equity, max_hold_bars=100)
    )
    assert len(result.trades) == 2
    by_entry = {record.entry_ts: record for record in result.trades}
    assert set(by_entry) == {bar(4).close_ts, bar(8).close_ts}
    assert by_entry[bar(4).close_ts].bars_held == 7
    assert by_entry[bar(8).close_ts].bars_held == 3
    assert all(record.exit_reason is ExitReason.END_OF_BACKTEST for record in result.trades)


def test_an_order_whose_horizon_outlives_the_data_is_unresolved_not_missed():
    """A miss is a statement about the MARKET; this is a statement about the
    DATASET. Resolving the order against the bars that happen to exist would
    declare a resting entry expired on a horizon that never elapsed -- a
    fabricated outcome -- and dropping it would hide a live order."""
    result, _, _ = run_loop(config(), [bar(0)], (0,))
    assert result.trades == ()
    assert result.unresolved_orders == 1
    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.status == "PENDING_AT_END_OF_DATA"
    assert order.filled is False
    assert order.miss_reason == ""            # not a miss
    assert order.bars_short_of_horizon == 1   # one bar short of a one-bar horizon
    assert order.fill_ts is None
    assert result.rejection["order_outcomes"] == {"pending_at_end_of_data": 1}
    assert any("PENDING_AT_END_OF_DATA" in note for note in result.notes)


# ---------------------------------------------------------------------------
# The ledger's arithmetic
# ---------------------------------------------------------------------------


def test_the_r_unit_is_carried_from_the_intent_and_exposure_is_measured_separately():
    """TWO quantities, and this test exists to keep them apart.

    Section 0.2: "R is defined once, at entry, and never revised." The ledger
    used to recompute `risk_dollars` from the realized fill, which both revised
    R and disagreed with `TradeIntent._check_geometry` -- the validator at the
    point R is actually decided. That is now a carried value, and the realized
    exposure the risk limits need lives under its own name.

    Hand-computed, one contract filled at 13001 where the intent asked for two
    from 13000 against a stop at 12990:
      * the R unit stays |13000 - 12990| * 20 * 2 = 400, the intent's own
        number, so a 1R loss is a 1R loss whatever the fill did;
      * the carried exposure is |13001 - 12990| * 20 * 1 = 220, the dollars the
        account can actually lose from here to the stop, which is what portfolio
        heat must count.
    The two differ here BY CONSTRUCTION -- a slipped, partial fill -- so neither
    assertion can pass by accident if the other quantity is substituted.
    """
    ledger = ledger_of()
    intent = intent_at(T0)
    position = ledger.open_position(
        intent=intent,
        entry=entry_fill(intent, price=13001.0, size=1.0),
        context=entry_context(),
    )
    assert intent.risk_dollars == pytest.approx(400.0)
    assert position.size == 1.0

    # the R unit: carried, never re-derived
    assert position.risk_dollars == pytest.approx(400.0)
    assert position.risk_dollars == pytest.approx(intent.risk_dollars)

    # the exposure: realized entry, filled size
    assert position.carried_risk_dollars == pytest.approx(220.0)
    assert ledger.open_risk_dollars() == pytest.approx(220.0)

    # and they are genuinely different numbers, so the test is not vacuous
    assert position.risk_dollars != pytest.approx(position.carried_risk_dollars)

    # R is reported against the planned unit: one point is 20 dollars of 400.
    assert position.r_at(13002.0) == pytest.approx(20.0 / 400.0)


def test_entry_costs_are_realized_when_they_are_paid_and_counted_once():
    """A ledger that books the whole round turn at the exit carries an equity
    curve that is too high for the entire life of the trade -- and that is the
    curve the kill switch and the drawdown series are read off.

    The two halves must still sum to `gross_pnl - costs.total` exactly, or a
    cost has been counted twice.
    """
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13002.0, low=12999.0, close=13000.0))
    bars.append(bar(5, open_=13000.0, high=13002.0, low=12999.0, close=13001.0))
    bars.append(bar(6, open_=13001.0, high=13015.0, low=13000.0, close=13012.0))
    bars.append(bar(7))
    result, _, _ = run_loop(config(), bars, (3,))
    record = result.trades[0]
    entry_costs = result.orders[0].costs.total
    assert record.exit_reason is ExitReason.TARGET
    # The round turn is strictly more than the entry side alone.
    assert record.costs.total > entry_costs
    assert record.pnl == pytest.approx(record.gross_pnl - record.costs.total)
    # While the position was open, only the ENTRY's costs had been spent.
    during = [
        point
        for point in result.equity
        if record.entry_ts < point.ts < record.exit_ts
    ]
    assert during, "the position must be held across at least one marked bar"
    assert during[0].realized_equity == pytest.approx(100_000.0 - entry_costs)
    assert during[0].open_positions == 1
    assert during[0].open_risk_dollars == pytest.approx(R_DOLLARS)
    # And at the end the account has moved by exactly the trade's net PnL:
    # no cost counted twice, none dropped.
    assert result.final.equity == pytest.approx(100_000.0 + record.pnl)
    assert result.final.open_entry_costs == 0.0
    assert result.final.realized_pnl == pytest.approx(record.pnl)


def test_no_equity_point_may_report_a_positive_drawdown():
    """`EquityPoint.drawdown_pct` is the fraction BELOW the high-water mark, so
    a positive value is not a drawdown and a mark below the point's own equity
    is not a mark.

    This was a real bug on the one sample every total return is read off. The
    run's last equity point is forced AFTER the bar's exits are resolved
    (section 9 marks first), so a trade exiting at its target ABOVE that bar's
    close left the final point above the mark it was taken from. Here the last
    bar reaches the 13010 target and closes at 12995:

        mark at the close: 100000 - costs + (12995 - 13000) * 20 * 2  (below the peak)
        after the exit:    100000 + (13010 - 13000) * 20 * 2 - costs  (a new peak)
    """
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13002.0, low=12999.0, close=13000.0))
    bars.append(bar(5, open_=13000.0, high=13012.0, low=12994.0, close=12995.0))
    result, _, _ = run_loop(config(), bars, (3,))
    record = result.trades[0]
    assert record.exit_reason is ExitReason.TARGET
    assert record.gross_pnl == pytest.approx(400.0)
    assert record.pnl > 0.0
    last = result.equity[-1]
    assert last.equity == pytest.approx(100_000.0 + record.pnl)
    assert last.high_water_mark >= last.equity
    assert last.drawdown_pct <= 0.0
    for point in result.equity:
        assert point.drawdown_pct <= 0.0
        assert point.high_water_mark >= point.equity


def test_a_forced_equity_sample_raises_the_high_water_mark_it_records():
    """The unit-level version: an equity point appended after a profitable
    close must not carry the pre-close peak, or the row disagrees with
    itself."""
    ledger = ledger_of()
    intent = intent_at(T0)
    position = ledger.open_position(
        intent=intent,
        entry=entry_fill(intent, costs=CostBreakdown(commission=4.5)),
        context=entry_context(),
    )
    ledger.mark_to_market(T0 + timedelta(seconds=INTERVAL), {"NQ": ENTRY})
    assert ledger.high_water_mark == pytest.approx(100_000.0)
    # +10 points on 2 contracts = +400 gross, less 9 dollars of round turn.
    record = ledger.close_position(
        position,
        exit_fill_at(TARGET, T0 + timedelta(seconds=2 * INTERVAL)),
        CostBreakdown(commission=9.0),
    )
    assert record.pnl == pytest.approx(391.0)
    point = ledger.record_equity_point(T0 + timedelta(seconds=2 * INTERVAL))
    assert point.equity == pytest.approx(100_391.0)
    assert point.high_water_mark == pytest.approx(100_391.0)
    assert point.drawdown_pct == 0.0


def test_mark_to_market_refuses_to_rewind_or_to_mark_at_a_non_price():
    """An out-of-order mark revalues the account at a price it has already
    moved past; a non-finite mark propagates into equity, the high-water mark
    and every drawdown computed from them."""
    ledger = ledger_of()
    ledger.mark_to_market(T0 + timedelta(seconds=INTERVAL), {"NQ": ENTRY})
    with pytest.raises(LedgerError, match="backwards"):
        ledger.mark_to_market(T0, {"NQ": ENTRY})
    intent = intent_at(T0)
    ledger.open_position(
        intent=intent, entry=entry_fill(intent), context=entry_context()
    )
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(LedgerError, match="finite"):
            ledger.mark_to_market(
                T0 + timedelta(seconds=9 * INTERVAL), {"NQ": bad}
            )


def test_a_fill_at_or_through_its_own_stop_opens_no_position_at_all():
    """A LONG filled at or below its stop has an inverted bracket, and
    "stopping out" would book a PROFIT at the stop -- a win the market never
    offered. The ledger must refuse it and must not leave a half-registered
    position behind."""
    intent = intent_at(T0)
    for price in (STOP, STOP - 5.0):
        ledger = ledger_of()
        with pytest.raises(Exception) as caught:
            ledger.open_position(
                intent=intent,
                entry=entry_fill(intent, price=price),
                context=entry_context(),
            )
        assert "inverted" in str(caught.value)
        assert ledger.open_positions == ()
        assert ledger.open_entry_costs == 0.0
        assert ledger.is_flat() is True


def test_the_time_stop_counts_bars_after_the_entry_bar_and_is_not_free():
    """`max_hold_bars=2` means two bars CLOSED after the one the fill printed
    in, and the ledger decides it while `exit_at_close` prices it -- so a
    ledger-decided flatten pays commission, spread and slippage like any other
    market exit rather than being free.

    Hand-computed: the fill prints on bar 4, so bars 5 and 6 are the two bars
    held and the exit is stamped at bar 6's close.
    """
    cfg = config()
    bars = warm_bars(4)
    bars += [bar(index) for index in range(4, 10)]
    result, _, _ = run_loop(
        cfg,
        bars,
        (3,),
        plan_factory=lambda ts, *, equity: plan_at(ts, equity=equity, max_hold_bars=2),
    )
    record = result.trades[0]
    assert record.exit_reason is ExitReason.TIME_STOP
    assert record.entry_ts == bar(4).close_ts
    assert record.exit_ts == bar(6).close_ts
    assert record.bars_held == 2
    assert record.costs.total > 0.0
    assert record.pnl == pytest.approx(record.gross_pnl - record.costs.total)
    assert result.cost_drag_by_exit_reason["TIME_STOP"]["n"] == 1.0


def test_a_kill_switch_flatten_pays_a_full_exits_costs():
    """The kill switch is the ledger's verdict, not the market's, and it is
    priced through the same exit path. A free flatten would understate the cost
    of the worst exits in the record -- the ones taken in a drawdown.

    Hand-computed: on a 5000-dollar account a 5% kill-switch threshold is 250
    dollars. Two contracts move 40 dollars per point, so a 7-point adverse move
    is 280 dollars -- past the threshold, and short of the 10-point stop, so
    only the kill switch can close this position.
    """
    cfg = config(
        risk=risk_config(
            starting_equity=5_000.0,
            daily_loss_limit_pct=0.04,
            weekly_loss_limit_pct=0.045,
            kill_switch_drawdown_pct=0.05,
            max_portfolio_heat_pct=0.10,
        )
    )
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13001.0, low=12999.0, close=13000.0))
    bars.append(bar(5, open_=13000.0, high=13000.0, low=12993.0, close=12993.0))
    bars.append(bar(6))
    result, _, _ = run_loop(cfg, bars, (3,), equity=5_000.0)
    record = result.trades[0]
    assert record.exit_reason is ExitReason.KILL_SWITCH
    assert record.exit_price == pytest.approx(12993.0)     # the bar's close
    assert record.gross_pnl == pytest.approx(-280.0)       # -7 * 20 * 2
    assert record.costs.total > 0.0
    assert record.pnl < record.gross_pnl
    assert result.cost_drag_by_exit_reason["KILL_SWITCH"]["mean_drag_r"] > 0.0


def test_a_bar_the_symbol_never_printed_holds_the_position_and_is_counted():
    """No bar means no price to exit at, so holding is the only honest option
    -- and the gap is counted, because an un-updated mark freezes the equity
    curve and the drawdown with it."""
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13001.0, low=12999.0, close=13000.0))
    bars.append(bar(6, open_=13000.0, high=13002.0, low=12999.0, close=13001.0))
    bars.append(bar(7))
    store = store_of(bars)
    gap_ts = T0 + timedelta(seconds=5 * INTERVAL)
    clock = sorted(list(store.timeline("NQ")) + [to_ns(gap_ts)])
    cfg = config()
    engine = StubEngine(cfg, (3,))
    loop = BacktestLoop(
        cfg, "NQ", engine=engine, git_commit="", git_dirty=False, atr_median_window=3
    )
    result = loop.run(StoreMarketSource(store, "NQ"), clock_ns=clock)
    assert result.provenance.bars_processed == len(bars) + 1
    assert result.ledger_diagnostics["stale_marks"] == 1.0
    assert any("no price" in note for note in result.notes)
    at_gap = [point for point in result.equity if point.ts == gap_ts]
    assert len(at_gap) == 1
    # Marked at the entry price it already had: the curve froze, visibly.
    assert at_gap[0].open_positions == 1
    assert at_gap[0].equity == pytest.approx(
        100_000.0 - result.orders[0].costs.total
    )


def test_the_ledger_catches_a_closed_position_whose_record_went_missing():
    """`realized_pnl` is accumulated trade by trade and the records are summed
    at the end. The two agree only if every closed position produced exactly
    one record, so a disagreement means a trade is missing from the journal or
    a record exists for a position the ledger never released."""
    ledger = ledger_of()
    intent = intent_at(T0)
    position = ledger.open_position(
        intent=intent, entry=entry_fill(intent), context=entry_context()
    )
    record = ledger.close_position(
        position,
        exit_fill_at(TARGET, T0 + timedelta(seconds=2 * INTERVAL)),
        CostBreakdown(commission=9.0),
    )
    with pytest.raises(LedgerError, match="0 trade records against 1"):
        ledger.assert_consistent([])
    # A record whose PnL disagrees with its own price path cannot even be
    # built -- `TradeRecord` re-derives `gross_pnl` and checks
    # `pnl == gross_pnl - costs.total` -- so the record that disagrees with
    # the ledger here is a consistent record of a DIFFERENT exit: 13015
    # instead of 13010, which is 5 more points, 200 more dollars.
    other = trade_record(
        exit_price=13015.0, size=SIZE, costs=CostBreakdown(commission=9.0)
    )
    assert other.pnl == pytest.approx(record.pnl + 200.0)
    with pytest.raises(LedgerError, match="does not match"):
        ledger.assert_consistent([other])
    ledger.assert_consistent([record])


def test_compound_equity_is_honoured_in_exactly_one_place():
    """Section 11 records that compounding "changes the drawdown distribution
    materially", so it must be honoured -- and observable. Under fixed
    fractional sizing the SIZE comes off the starting equity while the limits
    still see the real mark-to-market equity; collapsing the two would let a
    kill switch read a profitable account as being in drawdown."""
    compounding = ledger_of(config())
    cfg = config()
    fixed = ledger_of(cfg.replace(risk=cfg.risk.replace(compound_equity=False)))
    for ledger in (compounding, fixed):
        intent = intent_at(T0)
        ledger.open_position(
            intent=intent, entry=entry_fill(intent), context=entry_context()
        )
        # +50 points on 2 contracts = +2000 dollars of unrealized gain.
        ledger.mark_to_market(T0 + timedelta(seconds=2 * INTERVAL), {"NQ": 13050.0})
    assert compounding.compound_equity is True
    assert fixed.compound_equity is False
    assert compounding.equity == pytest.approx(fixed.equity)
    assert compounding.equity > 100_000.0
    assert compounding.sizing_equity() == pytest.approx(compounding.equity)
    assert fixed.sizing_equity() == 100_000.0
    # And both report which one they are, so a reader of the run can tell.
    assert fixed.snapshot().compound_equity is False
    assert fixed.snapshot().sizing_equity == 100_000.0


def test_excursions_take_the_bar_extremes_with_the_contracts_own_signs():
    """`TradeRecord` declares `mae_r <= 0` and `mfe_r >= 0`. A trade that was
    never once underwater has an adverse excursion of ZERO, not a positive
    one, and a bar that ran against the position and came back is recorded as
    having done so.

    Hand-computed on R = 400: 4 points against = 4 * 20 * 2 / 400 = 0.4R.
    """
    ledger = ledger_of()
    intent = intent_at(T0)
    position = ledger.open_position(
        intent=intent, entry=entry_fill(intent), context=entry_context()
    )
    position.observe(bar(2, open_=13000.0, high=13004.0, low=12996.0, close=13000.0))
    assert position.mfe_r == pytest.approx(0.4)
    assert position.mae_r == pytest.approx(-0.4)
    never_underwater = ledger_of().open_position(
        intent=intent, entry=entry_fill(intent), context=entry_context()
    )
    never_underwater.observe(
        bar(2, open_=13001.0, high=13006.0, low=13000.0, close=13005.0)
    )
    assert never_underwater.mae_r == 0.0
    assert never_underwater.mfe_r == pytest.approx(0.6)
    # A bar for another symbol, or a structurally invalid one, is not an
    # excursion: it is a bug, and it says so.
    other_symbol = Bar(
        symbol="ES",
        close_ts=bar(3).close_ts,
        open=13000.0,
        high=13002.0,
        low=12998.0,
        close=13000.0,
        volume=100.0,
        interval_seconds=INTERVAL,
    )
    with pytest.raises(LedgerError, match="offered to a position"):
        position.observe(other_symbol)
    invalid = Bar(
        symbol="NQ",
        close_ts=bar(3).close_ts,
        open=13000.0,
        high=12990.0,          # high below low: structurally impossible
        low=13010.0,
        close=13000.0,
        volume=100.0,
        interval_seconds=INTERVAL,
    )
    with pytest.raises(LedgerError, match="invalid bar"):
        position.observe(invalid)


def test_the_signal_bars_context_is_what_the_trade_is_labelled_with():
    """Re-deriving the context at the exit would label the trade with the
    regime it CLOSED in rather than the one it was taken in, which is the
    quiet way a regime breakdown stops meaning anything."""
    ledger = ledger_of()
    intent = intent_at(T0)
    context = entry_context(
        regime=Regime.TRENDING_UP,
        flow_score=71.0,
        component_points={Component.STRUCTURE: 15.0, Component.ORDER_FLOW: 7.5},
        volatility=10.0,
        vol_percentile=0.5,
        session=Session.RTH_OPEN,
    )
    position = ledger.open_position(
        intent=intent, entry=entry_fill(intent), context=context
    )
    record = ledger.close_position(
        position,
        exit_fill_at(TARGET, T0 + timedelta(seconds=3 * INTERVAL)),
        CostBreakdown(commission=9.0),
    )
    assert record.regime is Regime.TRENDING_UP
    assert record.flow_score == 71.0
    assert record.structure_score == 15.0
    assert record.order_flow_score == 7.5
    assert record.session is Session.RTH_OPEN
    assert record.volatility == 10.0
    assert record.vol_percentile == 0.5
    # A component that contributed nothing contributed ZERO points, which is
    # the honest value for a point contribution (not a missing measurement).
    assert record.options_flow_score == 0.0


def test_an_unmeasured_volatility_ratio_is_absent_rather_than_one():
    """A median of one observation equals that observation, so reporting
    ATR/median = 1.0 before the window is full claims "ATR is exactly at its
    median" on the first bar -- a measurement that was never made. Execution
    refuses to price a fill against it, which is what makes the refusal
    load-bearing rather than cosmetic."""
    tracker = VolatilityTracker(window=3)
    assert tracker.context().is_measured is False
    assert tracker.context().multiple is None
    tracker.observe(10.0, T0)
    tracker.observe(12.0, T0)
    assert tracker.context().is_measured is False
    tracker.observe(14.0, T0)
    measured = tracker.context()
    assert measured.is_measured is True
    assert measured.median_atr == pytest.approx(12.0)      # median(10, 12, 14)
    assert measured.multiple == pytest.approx(14.0 / 12.0)
    tracker.observe(None, T0)
    tracker.observe(-1.0, T0)
    tracker.observe(float("nan"), T0)
    assert tracker.observations == 3
    with pytest.raises(BacktestError, match="median"):
        VolatilityTracker(window=1)


# ---------------------------------------------------------------------------
# Determinism, and provenance
# ---------------------------------------------------------------------------


def jittered_bars() -> list[Bar]:
    """Warm-up, a signal bar, then bars that trade through 13000 either way."""
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13003.0, low=12999.0, close=13002.0))
    bars.append(bar(5, open_=13001.0, high=13002.0, low=12980.0, close=12985.0))
    bars += [bar(index) for index in range(6, 12)]
    bars.append(bar(12, open_=13000.0, high=13003.0, low=12999.0, close=13002.0))
    bars.append(bar(13))
    return bars


def test_two_runs_over_the_same_data_are_byte_identical_with_randomness_on():
    """One run, one seed, identical results -- tested where it can actually
    fail, with both stochastic terms in the whole backtest switched on.

    `random_latency_ms` and `random_miss_probability` are the only draws in
    `backtest/`, and they are the only way this requirement can break other
    than by a clock or a uuid. The rows are compared as serialized text, not
    just by digest, so "byte-identical" means what it says.
    """
    from flow_model.utils.serialization import dumps

    cfg = config(
        execution=ExecutionConfig(
            latency_ms=100.0, random_latency_ms=400.0, random_miss_probability=0.4
        )
    )
    bars = jittered_bars()
    first, _, _ = run_loop(cfg, bars, (3, 11))
    second, _, _ = run_loop(cfg, bars, (3, 11))
    assert first.provenance.run_id == second.provenance.run_id
    assert first.result_fingerprint() == second.result_fingerprint()
    for rows in ("trades", "equity", "signals", "orders"):
        left = [row.to_init_dict() for row in getattr(first, rows)]
        right = [row.to_init_dict() for row in getattr(second, rows)]
        assert dumps(left) == dumps(right), rows
    # The scenario must actually exercise the draws, or this proves nothing.
    assert any(order.miss_reason == "random_miss" for order in first.orders)
    assert any(order.eligibility_lag_seconds > 0.0 for order in first.orders)


def test_an_orders_draw_does_not_depend_on_how_many_orders_preceded_it():
    """Each order draws from its own NAMED stream, keyed on the order, so
    adding a stochastic term -- or an earlier order -- cannot shift another
    order's draw. A single shared generator would make every result depend on
    the number of draws before it, which is not reproducibility, it is luck
    that happens to repeat."""
    cfg = config(
        execution=ExecutionConfig(
            latency_ms=100.0, random_latency_ms=400.0, random_miss_probability=0.4
        )
    )
    bars = jittered_bars()
    with_predecessor, _, _ = run_loop(cfg, bars, (3, 11))
    alone, _, _ = run_loop(cfg, bars, (11,))
    late_ts = bar(11).close_ts
    def late(result):
        return [
            (row.status, row.eligibility_lag_seconds, row.miss_reason, row.fill_ts)
            for row in result.orders
            if row.signal_ts == late_ts
        ]
    assert len(late(with_predecessor)) == 1
    assert late(with_predecessor) == late(alone)
    assert len(with_predecessor.orders) == 2
    assert len(alone.orders) == 1


def test_the_result_fingerprint_is_sensitive_to_the_rows_it_covers():
    """A digest that never changes is not a determinism test, it is a
    decoration. Two results differing in one trade, one equity sample or one
    signal row must not share a fingerprint."""
    base = run_result_of(trades=(trade_record(),))
    assert base.result_fingerprint() != run_result_of().result_fingerprint()
    moved = run_result_of(trades=(trade_record(exit_price=13009.0),))
    assert moved.result_fingerprint() != base.result_fingerprint()
    plus_equity = run_result_of(
        trades=(trade_record(),),
        equity=(EquityPoint(ts=T0, equity=1.0, realized_equity=1.0),),
    )
    assert plus_equity.result_fingerprint() != base.result_fingerprint()


def test_the_wall_clock_is_excluded_from_both_fingerprints():
    """`created_at` is the one wall-clock field in the whole run record. Inside
    the identity of a result it would make two identical runs look different,
    which is the determinism requirement failing in the other direction: an
    alarm that can never be silenced."""
    early = RunProvenance(
        run_id="run-test", symbol="NQ", config_hash="cfg", data_hash="dat",
        seed=1234, created_at=T0, split_label=SplitPhase.TRAIN,
    )
    late = early.replace(created_at=T0 + timedelta(days=400))
    assert early.created_at != late.created_at
    assert early.inputs_fingerprint() == late.inputs_fingerprint()
    assert (
        run_result_of().replace(provenance=early).result_fingerprint()
        == run_result_of().replace(provenance=late).result_fingerprint()
    )
    # But a real input difference must change it, or it is hashing nothing.
    assert early.inputs_fingerprint() != early.replace(seed=99).inputs_fingerprint()
    assert (
        early.inputs_fingerprint()
        != early.replace(data_hash="other").inputs_fingerprint()
    )


def test_a_second_run_on_the_same_loop_is_refused_rather_than_served():
    """The limit manager is built in the constructor and is the one object here
    that accumulates: realized session PnL, the consecutive-loss streak, armed
    cooldowns and the kill-switch flag.

    This was a real bug. A second `run()` over the SAME data was gated by the
    first run's losses, so the two runs disagreed -- the determinism
    requirement failing silently, in the direction of fewer trades, which
    reads as a strategy that stopped finding setups. Hand-computed: with a
    daily loss limit of 0.4% of 100000 = 400 dollars, one stopped-out trade
    (-400 gross and costs on top) puts the session past the limit, so every
    signal in a second run would be blocked.
    """
    cfg = config(
        risk=risk_config(daily_loss_limit_pct=0.004, weekly_loss_limit_pct=0.05)
    )
    bars = exit_then_signal_bars()
    store = store_of(bars)
    loop = BacktestLoop(
        cfg,
        "NQ",
        engine=GatedStubEngine(cfg, (3,)),
        git_commit="",
        git_dirty=False,
        atr_median_window=3,
    )
    first = loop.run(StoreMarketSource(store, "NQ"))
    assert len(first.trades) == 1
    assert first.trades[0].pnl < -400.0
    with pytest.raises(BacktestError, match="already run"):
        loop.run(StoreMarketSource(store, "NQ"))
    # A fresh loop over the same data reproduces the first run exactly.
    again, _, _ = run_loop(cfg, bars, (3,), engine_cls=GatedStubEngine)
    assert again.result_fingerprint() == first.result_fingerprint()


def test_the_run_id_separates_two_split_phases_over_the_same_data():
    """`run_id` is content-derived so that `trade_id` can be stable. That makes
    the split part of the identity on purpose: a TRAIN run and a TEST run over
    the same bars must not collide in one database, or an IS row and an OOS row
    would share a primary key and section 0.8's "no performance claim without a
    split label" would have nothing to stand on."""
    cfg = config()
    bars = warm_bars(6)
    ids = {}
    for phase in (SplitPhase.TRAIN, SplitPhase.VALIDATION, SplitPhase.TEST, None):
        store = store_of(bars)
        loop = BacktestLoop(
            cfg, "NQ", engine=StubEngine(cfg, ()), split_label=phase,
            split_name="" if phase is None else f"fold-{phase.value}",
            git_commit="", git_dirty=False, atr_median_window=3,
        )
        result = loop.run(StoreMarketSource(store, "NQ"))
        ids[phase] = result.provenance.run_id
        assert result.provenance.split_label is phase
        assert all(row.split_label is phase for row in result.signals)
    assert len(set(ids.values())) == 4


def test_every_row_carries_the_provenance_that_makes_it_attributable():
    """A performance number whose inputs cannot be reconstructed is not a
    number, it is an anecdote. The run records config hash, data hash, seed,
    git commit, split phase and code version -- and each TRADE carries the
    first four, so a row pulled out of the table alone is still attributable.
    """
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13002.0, low=12999.0, close=13000.0))
    bars.append(bar(5, open_=13001.0, high=13015.0, low=13000.0, close=13012.0))
    cfg = config()
    store = store_of(bars)
    loop = BacktestLoop(
        cfg, "NQ", engine=StubEngine(cfg, (3,)), split_label=SplitPhase.TRAIN,
        split_name="fold-1", git_commit="abc1234", git_dirty=True,
        atr_median_window=3,
    )
    result = loop.run(StoreMarketSource(store, "NQ"))
    provenance = result.provenance
    assert provenance.config_hash == cfg.config_hash
    assert provenance.data_hash == store.combined_data_hash()
    assert provenance.seed == cfg.backtest.seed == 1234
    assert provenance.git_commit == "abc1234"
    assert provenance.git_dirty is True
    assert provenance.code_version
    assert provenance.split_label is SplitPhase.TRAIN
    assert provenance.split_name == "fold-1"
    assert provenance.first_ts == T0
    assert provenance.last_ts == bar(5).close_ts
    assert provenance.interval_seconds == INTERVAL
    assert provenance.bars_in_clock == provenance.bars_processed == 6
    assert provenance.clock_truncated is False
    assert provenance.starting_equity == 100_000.0
    assert provenance.compound_equity is True
    record = result.trades[0]
    assert record.run_id == provenance.run_id
    assert record.config_hash == cfg.config_hash
    assert record.data_hash == provenance.data_hash
    assert record.split_label is SplitPhase.TRAIN
    assert record.trade_id.startswith(provenance.run_id)


def test_a_truncated_clock_says_so_rather_than_passing_for_the_dataset():
    """`max_bars` is a debug cap. A run whose counts are not the dataset's must
    say which it is, or a 300-bar smoke run gets quoted as a ten-year result."""
    cfg = config(backtest=BacktestConfig(symbols=("NQ",), seed=1234, max_bars=5))
    result, _, _ = run_loop(cfg, [bar(index) for index in range(12)], ())
    assert result.provenance.bars_in_clock == 12
    assert result.provenance.bars_processed == 5
    assert result.provenance.max_bars == 5
    assert result.provenance.clock_truncated is True
    assert result.bars_evaluated == 5
    assert any("truncated the clock" in note for note in result.notes)


def test_a_clock_that_is_not_strictly_ascending_is_refused():
    """"The next bar" is meaningless on an unordered clock, and a fill could
    resolve before its own signal."""
    bars = warm_bars(5)
    store = store_of(bars)
    clock = list(store.timeline("NQ"))
    cfg = config()
    for broken in ([clock[2], clock[1]], [clock[1], clock[1]]):
        loop = BacktestLoop(
            cfg, "NQ", engine=StubEngine(cfg, ()), git_commit="", git_dirty=False,
            atr_median_window=3,
        )
        with pytest.raises(BacktestError, match="strictly ascending"):
            loop.run(StoreMarketSource(store, "NQ"), clock_ns=broken)


def test_the_same_config_and_data_reproduce_across_processes(tmp_path):
    """Dict and string hashing are randomized per process, so a digest built
    over a mapping can be stable in one process and unstable across two. The
    check that matters is therefore between PROCESSES with different
    PYTHONHASHSEED values.

    The two runs are compared to each other, so no value from the YAML is
    under test here -- only that the same inputs give the same digest.
    """
    import os
    import subprocess
    import sys

    script = tmp_path / "twice.py"
    script.write_text(
        "\n".join(
            [
                "from datetime import date",
                "from flow_model.config.loader import default_config",
                "from flow_model.data import DataStore, SyntheticAdapter, SyntheticConfig, TradingCalendar",
                "from flow_model.backtest.loop import BacktestLoop, StoreMarketSource",
                "cfg = default_config()",
                "calendar = TradingCalendar()",
                "adapter = SyntheticAdapter(config=SyntheticConfig(), seed=cfg.seed,",
                "    spec_by_symbol=dict(cfg.instruments),",
                "    interval_seconds=cfg.data.primary_interval_seconds, calendar=calendar)",
                "store = DataStore(latency_seconds=cfg.data.data_latency_seconds)",
                "store.load(adapter, 'NQ', date(2016, 1, 4), date(2016, 1, 6),",
                "    primary_interval=cfg.data.primary_interval_seconds)",
                "clock = store.timeline('NQ')[:120]",
                "result = BacktestLoop(cfg, 'NQ', calendar=calendar, git_commit='',",
                "    git_dirty=False).run(",
                "    StoreMarketSource.from_config(store, 'NQ', cfg), clock_ns=clock)",
                "print(result.provenance.run_id, result.result_fingerprint(),",
                "      result.provenance.inputs_fingerprint())",
            ]
        ),
        encoding="utf-8",
    )
    import flow_model

    # The repo root, derived from the package rather than from the working
    # directory, so the subprocess imports the same tree whatever pytest was
    # invoked from.
    root = Path(flow_model.__file__).resolve().parent.parent
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(root))
        done = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, env=env, timeout=600,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        outputs.append(done.stdout.strip())
    assert outputs[0] == outputs[1]
    assert len(outputs[0].split()) == 3


# ---------------------------------------------------------------------------
# Storage is not hashing
# ---------------------------------------------------------------------------


def traded_bars() -> list[Bar]:
    """Two sequential round turns: one through the target, one through the stop.

    bar 3 signals, 4 fills at the open, 6 reaches 13010.
    bar 7 signals, 8 fills at the open, 9 trades down through 12990.
    """
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13002.0, low=12999.0, close=13000.0))
    bars.append(bar(5))
    bars.append(bar(6, open_=13001.0, high=13015.0, low=13000.0, close=13012.0))
    bars.append(bar(7))
    bars.append(bar(8, open_=13000.0, high=13002.0, low=12999.0, close=13000.0))
    bars.append(bar(9, open_=12999.0, high=13000.0, low=12980.0, close=12985.0))
    bars.append(bar(10))
    return bars


@pytest.fixture
def traded_run():
    result, _, _ = run_loop(config(), traded_bars(), (3, 7))
    assert len(result.trades) == 2
    return result


def test_a_trade_record_round_trips_through_sqlite_without_losing_a_digit(tmp_path):
    """Write it, read it back, get an EQUAL object -- including the floats.

    The committed bug this guards: `canonical_json` renders floats as
    fixed-width strings so a digest is portable, and a value stored through it
    read back as the STRING "7.200000000000e-01". The values below are chosen
    to have no short decimal form (0.1 + 0.2, one third) so a round trip
    through text at any precision would fail.
    """
    original = trade_record(
        trade_id="exotic",
        size=2.0,
        exit_price=13007.25,
        costs=CostBreakdown(
            commission=4.5, exchange_fees=0.74, entry_slippage=0.1 + 0.2,
            exit_slippage=1.0 / 3.0, spread_cost=5.0,
        ),
        mae_r=-0.4731,
        mfe_r=1.2345,
        volatility=1.0 / 3.0,
        vol_percentile=0.72,
        flow_score=65.23,
        split_label=SplitPhase.VALIDATION,
        entry_reason="structure: a constructed reason | risk: 2 at 13000",
        exit_reason_detail="target traded through",
    )
    with Database(tmp_path / "round_trip.db") as database:
        assert database.write_trades([original]) == 1
        read_back = database.read_trades()
    assert len(read_back) == 1
    restored = read_back[0]
    assert restored == original
    assert restored.costs == original.costs
    # Not merely equal-ish: the same doubles, and still numbers.
    assert isinstance(restored.vol_percentile, float)
    assert restored.vol_percentile == 0.72
    assert restored.volatility == 1.0 / 3.0
    assert restored.costs.entry_slippage == 0.1 + 0.2
    assert repr(restored.costs.exit_slippage) == repr(1.0 / 3.0)
    assert restored.mae_r == -0.4731
    # And the derived values agree with the fields they are derived from.
    assert restored.r_multiple == original.r_multiple
    assert restored.cost_drag_r == original.cost_drag_r


def test_a_whole_traded_run_round_trips_including_its_rows(tmp_path, traded_run):
    """A run is only a result if it can be read back as the same run: the
    trades, the entry attempts, the signal rows and the equity curve."""
    with Database(tmp_path / "traded.db") as database:
        run_id = database.write_run(traded_run)
        assert run_id == traded_run.provenance.run_id
        assert set(database.read_trades(run_id)) == set(traded_run.trades)
        assert database.read_orders(run_id) == traded_run.orders
        assert database.read_signals(run_id) == traded_run.signals
        assert database.read_equity(run_id) == traded_run.equity
        row = database.read_run(run_id)
        assert row is not None
        assert row["trade_count"] == 2
        assert row["bars_evaluated"] == traded_run.bars_evaluated
        assert row["result_fingerprint"] == traded_run.result_fingerprint()
        assert row["inputs_fingerprint"] == traded_run.provenance.inputs_fingerprint()
        assert row["compound_equity"] is True
        assert row["git_dirty"] is False
        assert set(row["cost_drag"]) == {"STOP", "TARGET"}
        assert database.counts()["trades"] == 2
        assert database.schema_version() >= 1


def test_a_float_in_a_json_column_is_stored_as_a_number(tmp_path, traded_run):
    """The other half of the committed bug: the JSON columns.

    `utils.serialization.dumps` for storage, `canonical_json` for identity.
    A cost drag or a component's points read back as a string would make every
    downstream comparison a string comparison, which silently succeeds.
    """
    journal_row = SignalRecord(
        run_id="run-test", symbol="NQ", ts=T0, action=SignalAction.WAIT,
        wait_reason=WaitReason.SCORE_BELOW_THRESHOLD, flow_score=65.23,
        component_points={Component.STRUCTURE: 0.72},
    )
    stored = dict(zip(SIGNAL_COLUMNS, signal_row(journal_row)))
    assert '"0.72"' not in stored["component_points_json"]
    assert "0.72" in stored["component_points_json"]
    assert isinstance(stored["flow_score"], float)

    with Database(tmp_path / "json.db") as database:
        database.write_run(traded_run)
        row = database.read_run(traded_run.provenance.run_id)
    for bucket in row["cost_drag"].values():
        for value in bucket.values():
            assert isinstance(value, float)
    for value in row["cost_totals"].values():
        assert isinstance(value, float)
    assert row["cost_totals"]["total"] > 0.0


def test_a_non_finite_float_is_refused_rather_than_written_as_a_null(tmp_path):
    """SQLite has no NaN: it binds a non-finite double as NULL.

    This was a real bug. `TradeRecord.volatility` is an optional float with no
    finiteness bound, so a NaN ATR wrote without complaint and read back as
    None -- unequal to the record written, and claiming "no ATR was supplied"
    where the truth was "the ATR was not a number". Storage may not change a
    value silently, so the write is refused and the field is named.
    """
    nan_volatility = trade_record(trade_id="nan", volatility=float("nan"))
    assert math.isnan(nan_volatility.volatility)
    with Database(tmp_path / "nan.db") as database:
        with pytest.raises(DatabaseError, match="volatility"):
            database.write_trades([nan_volatility])
        assert database.counts()["trades"] == 0
        with pytest.raises(DatabaseError, match="mfe_r"):
            database.write_trades([trade_record(trade_id="inf", mfe_r=float("inf"))])
        # An absent measurement is a different thing and stores fine as NULL.
        assert database.write_trades([trade_record(volatility=None)]) == 1
        assert database.read_trades()[0].volatility is None


def test_two_equity_samples_at_one_instant_are_refused_and_nothing_half_lands(
    tmp_path,
):
    """A curve with two samples at one timestamp is read by any return or
    drawdown series as two bars, one of them with a zero-length time step.
    `(run_id, ts)` is the natural key, so the write fails -- and because it is
    one transaction, it must leave NO rows behind. A half-written run is the
    worst of the three outcomes: it looks like a complete run with fewer
    trades.
    """
    duplicated = run_result_of(
        trades=(trade_record(),),
        equity=(
            EquityPoint(ts=T0, equity=100_000.0, realized_equity=100_000.0),
            EquityPoint(ts=T0, equity=100_391.0, realized_equity=100_391.0),
        ),
    )
    with Database(tmp_path / "dup.db") as database:
        with pytest.raises(DatabaseError, match="could not be written"):
            database.write_run(duplicated)
        assert database.counts() == {
            "runs": 0, "trades": 0, "signals": 0, "orders": 0,
            "equity": 0, "experiments": 0, "seal_audit": 0,
        }


def test_writing_the_same_run_twice_raises_and_overwriting_is_explicit(
    tmp_path, traded_run
):
    """The run id is derived from the config hash, the data hash, the seed and
    the split, so a collision means this exact run is already stored. Doubling
    its trade count silently would double its PnL in every aggregate computed
    from the table."""
    with Database(tmp_path / "twice.db") as database:
        run_id = database.write_run(traded_run)
        with pytest.raises(DatabaseError, match="already stored"):
            database.write_run(traded_run)
        assert database.counts()["trades"] == 2
        assert database.write_run(traded_run, replace=True) == run_id
        assert database.counts()["runs"] == 1
        assert database.counts()["trades"] == 2
        assert database.delete_run(run_id) > 0
        assert database.counts()["runs"] == 0
        assert database.read_trades(run_id) == ()
        assert database.read_run(run_id) is None


def test_a_trade_collision_raises_this_modules_own_error(tmp_path):
    """`write_trades` must fail the same way `write_run` does. A caller that
    catches `DatabaseError` around one and a bare `sqlite3.IntegrityError`
    around the other catches neither reliably, and the condition is the same:
    this trade is already stored."""
    with Database(tmp_path / "collide.db") as database:
        database.write_trades([trade_record()])
        with pytest.raises(DatabaseError, match="already stored|primary key"):
            database.write_trades([trade_record()])
        assert database.counts()["trades"] == 1
        # Not the raw driver error, which would escape a DatabaseError handler.
        try:
            database.write_trades([trade_record()])
        except DatabaseError:
            pass
        except sqlite3.IntegrityError:  # pragma: no cover - the bug this pins
            pytest.fail("the raw sqlite3 error escaped the module's own type")


def test_the_schema_declares_every_table_section_12_names():
    """Section 12 names six tables. `orders` is a declared ADDITION: without
    it a missed entry disappears and "the signal fired and the limit never
    traded through" becomes indistinguishable from "no signal fired" -- two
    findings with the same empty trade list, only one of which is about the
    strategy."""
    connection = sqlite3.connect(":memory:")
    for statement in create_statements():
        connection.execute(statement)
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"runs", "trades", "signals", "equity", "experiments", "seal_audit"} <= tables
    assert "orders" in tables and "orders" in TABLES
    # Postgres-compatible: no SQLite-only syntax in the DDL.
    ddl = "\n".join(create_statements()).upper()
    assert "AUTOINCREMENT" not in ddl
    assert "WITHOUT ROWID" not in ddl
    assert "DOUBLE PRECISION" in ddl
    # No column is typed REAL: 4 bytes in Postgres, which would silently
    # round every price and every PnL on a migration. Matched on a word
    # boundary, because REALIZED_EQUITY is a column name, not a type.
    import re

    assert re.search(r"\bREAL\b", ddl) is None


def test_reads_are_deterministic_and_chronological(tmp_path, traded_run):
    """Two reads of one database must give the same sequence, and that sequence
    must be in time order: a caller walking the rows is walking the run."""
    with Database(tmp_path / "order.db") as database:
        database.write_run(traded_run)
        for reader in ("read_trades", "read_signals", "read_orders", "read_equity"):
            first = getattr(database, reader)(traded_run.provenance.run_id)
            second = getattr(database, reader)(traded_run.provenance.run_id)
            assert first == second, reader
        assert [row.ts for row in database.read_signals()] == sorted(
            row.ts for row in traded_run.signals
        )
        assert [point.ts for point in database.read_equity()] == sorted(
            point.ts for point in traded_run.equity
        )
        # Trades are ordered by ENTRY, which is not necessarily the order in
        # which their PnL was realized. Anything that walks them as an equity
        # curve must sort by exit_ts itself.
        entries = [record.entry_ts for record in database.read_trades()]
        assert entries == sorted(entries)


def test_the_seal_audit_mirror_opens_nothing(tmp_path):
    """The JSONL log in `validation/splits.py` stays the source of truth for
    the one-open budget: a budget enforced by a table any caller can UPDATE is
    not a budget. An absent log imports zero rows and creates no file."""
    path = tmp_path / "seal_audit.jsonl"
    with Database(tmp_path / "seal.db") as database:
        assert database.import_seal_audit(path) == 0
        assert database.counts()["seal_audit"] == 0
    assert not path.exists()


# ---------------------------------------------------------------------------
# Honest cost reporting
# ---------------------------------------------------------------------------


def stop_trade(**over) -> TradeRecord:
    """A stopped-out NQ round turn, size 2: R = 400, costs 47.74 dollars.

    gross = (12990 - 13000) * 20 * 2 = -400,  pnl = -447.74,  R = -1.11935
    """
    fields = dict(
        trade_id="stopped",
        size=2.0,
        exit_price=STOP,
        exit_reason=ExitReason.STOP,
        costs=CostBreakdown(commission=9.0, exit_slippage=20.0, spread_cost=18.74),
    )
    fields.update(over)
    return trade_record(**fields)


def target_trade(**over) -> TradeRecord:
    """A target NQ round turn, size 2: R = 400, costs 21.49 dollars.

    gross = (13010 - 13000) * 20 * 2 = +400,  pnl = +378.51,  R = +0.946275
    """
    fields = dict(
        trade_id="targeted",
        size=2.0,
        exit_price=TARGET,
        exit_reason=ExitReason.TARGET,
        costs=CostBreakdown(commission=9.0, entry_slippage=7.49, spread_cost=5.0),
    )
    fields.update(over)
    return trade_record(**fields)


def test_cost_drag_is_reported_per_exit_reason_and_matches_the_measurement():
    """The measured Phase 7a hurdle, reproduced from the records themselves.

    A target is a resting limit -- no spread crossed, no slippage -- while a
    stop is a market order that additionally pays
    `stop_order_extra_slippage_ticks`, so the two exits do not cost the same:

        stop:    47.74 / 400 = 0.11935 R
        target:  21.49 / 400 = 0.0537250 R
    """
    journal = Journal(run_id="run-test")
    journal.record_trade(stop_trade())
    journal.record_trade(target_trade())
    assert journal.trades[0].pnl == pytest.approx(-447.74)
    assert journal.trades[1].pnl == pytest.approx(378.51)
    drag = journal.cost_drag_by_exit_reason()
    assert set(drag) == {"STOP", "TARGET"}
    assert drag["STOP"]["n"] == 1.0
    assert drag["STOP"]["costs"] == pytest.approx(STOP_EXIT_COSTS)
    assert drag["TARGET"]["costs"] == pytest.approx(TARGET_EXIT_COSTS)
    assert drag["STOP"]["mean_drag_r"] == pytest.approx(0.11935)
    assert drag["TARGET"]["mean_drag_r"] == pytest.approx(0.0537250)
    assert drag["STOP"]["mean_drag_r"] > 2.0 * drag["TARGET"]["mean_drag_r"]
    assert journal.cost_totals()["total"] == pytest.approx(69.23)
    # A reason with no trades is absent, not present with 0.0.
    assert "TIME_STOP" not in drag
    assert "KILL_SWITCH" not in drag


def test_the_breakeven_win_rate_implied_by_the_measured_drag_is_not_fifty_percent():
    """Costs move the breakeven win rate of a symmetric 1R system from 50% to
    54.19%, and that figure can only be computed from the two drags
    separately, because wins pay the target's and losses pay the stop's.

        win payoff  = 1 - 0.0537250 =  0.946275 R
        loss payoff = 1 + 0.11935   = -1.119350 R
        w * 0.946275 = (1 - w) * 1.119350
        w = 1.119350 / (0.946275 + 1.119350) = 0.5418941  ->  54.19%
    """
    journal = Journal(run_id="run-test")
    journal.record_trade(stop_trade())
    journal.record_trade(target_trade())
    drag = journal.cost_drag_by_exit_reason()
    win_payoff = 1.0 - drag["TARGET"]["mean_drag_r"]
    loss_payoff = 1.0 + drag["STOP"]["mean_drag_r"]
    breakeven = loss_payoff / (win_payoff + loss_payoff)
    assert win_payoff == pytest.approx(0.946275)
    assert loss_payoff == pytest.approx(1.119350)
    assert round(breakeven * 100, 2) == 54.19
    assert breakeven > 0.5
    # The two records' own R multiples are the same two numbers, so the hurdle
    # is a property of the trade set and not of this arithmetic.
    assert journal.trades[0].r_multiple == pytest.approx(-1.11935)
    assert journal.trades[1].r_multiple == pytest.approx(0.946275)


def test_one_averaged_drag_figure_misstates_every_trade_in_the_set():
    """Why there is no single drag figure anywhere in this module.

    Ten trades, seven stopped and three targeted:

        total costs = 7 * 47.74 + 3 * 21.49 = 398.65, mean 39.865 = 0.0996625 R

    Applying that one figure to every trade leaves the TOTAL R untouched --
    costs are only redistributed -- and misstates each trade in opposite
    directions:

        average win   0.946275 R  ->  0.9003375 R   (understated by 0.0459375)
        average loss -1.119350 R  -> -1.0996625 R   (overstated by 0.0196875)
        payoff ratio  0.845379     ->  0.818740

    So a loss looks better than it was and a win looks worse: the R
    distribution, the avg-win/avg-loss pair and any payoff ratio built from
    them are all wrong, while the headline total agrees -- which is exactly
    the shape of error that survives a sanity check.
    """
    journal = Journal(run_id="run-test")
    for index in range(7):
        journal.record_trade(stop_trade(trade_id=f"stop-{index}"))
    for index in range(3):
        journal.record_trade(target_trade(trade_id=f"target-{index}"))
    totals = journal.cost_totals()
    assert totals["total"] == pytest.approx(398.65)
    averaged_drag = totals["total"] / len(journal.trades) / R_DOLLARS
    assert averaged_drag == pytest.approx(0.0996625)

    drag = journal.cost_drag_by_exit_reason()
    true_win = 1.0 - drag["TARGET"]["mean_drag_r"]
    true_loss = -(1.0 + drag["STOP"]["mean_drag_r"])
    assert true_win == pytest.approx(0.946275)
    assert true_loss == pytest.approx(-1.119350)
    assert 1.0 - averaged_drag == pytest.approx(0.9003375)
    assert -(1.0 + averaged_drag) == pytest.approx(-1.0996625)
    assert true_win / -true_loss == pytest.approx(0.845379, abs=5e-7)
    assert (1.0 - averaged_drag) / (1.0 + averaged_drag) == pytest.approx(
        0.818740, abs=5e-7
    )
    # The total is invariant, which is why averaging survives a spot check:
    # 3 * 0.946275 - 7 * 1.119350 = -4.996625
    true_total = 3 * true_win + 7 * true_loss
    averaged_total = 3 * (1.0 - averaged_drag) - 7 * (1.0 + averaged_drag)
    assert true_total == pytest.approx(-4.996625)
    assert averaged_total == pytest.approx(true_total)
    # And the module offers no such figure to quote.
    assert "mean_drag_r" not in journal.cost_totals()
    assert set(journal.cost_drag_by_exit_reason()) == {"STOP", "TARGET"}


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_one_trade_all_wins_all_losses_and_a_scratch_are_all_counted_not_rated():
    """The three degenerate trade sets, through the run summary.

    All wins makes a profit factor infinite; all losses makes it zero; one
    trade makes every dispersion statistic undefined. The summary therefore
    reports COUNTS -- which are defined in all three -- and names no ratio, so
    there is nothing here to divide by zero.

    A scratch is a third outcome, not a loss: a trade whose costs exactly ate
    its gross is neither, and counting it as a loss would understate the win
    rate a later stage computes. Hand-computed: gross = (13000.5 - 13000) * 20
    * 1 = 10 dollars against 10 dollars of commission, so pnl is exactly 0.
    """
    win = target_trade(trade_id="w1")
    loss = stop_trade(trade_id="l1")
    scratch = trade_record(
        trade_id="s1",
        exit_price=13000.5,
        costs=CostBreakdown(commission=10.0),
    )
    assert scratch.pnl == 0.0
    assert scratch.is_scratch is True
    assert scratch.is_win is False and scratch.is_loss is False

    for trades, expected in (
        ((win,), (1, 1, 0, 0)),
        ((win, target_trade(trade_id="w2")), (2, 2, 0, 0)),
        ((loss, stop_trade(trade_id="l2")), (2, 0, 2, 0)),
        ((win, loss, scratch), (3, 1, 1, 1)),
    ):
        summary = run_result_of(trades=trades).summary()
        assert (
            summary["trades"], summary["wins"], summary["losses"], summary["scratches"]
        ) == expected
        for forbidden in FORBIDDEN_RATIOS:
            assert not any(forbidden in key for key in summary)
        assert "undefined" in summary["ratios"]

    # All wins: the losing bucket is ABSENT from the drag mapping rather than
    # present with a zero, so nothing downstream can read it as a measurement.
    journal = Journal(run_id="run-test")
    journal.record_trade(win)
    journal.record_trade(target_trade(trade_id="w2"))
    assert set(journal.cost_drag_by_exit_reason()) == {"TARGET"}
    assert journal.exit_reason_counts() == {"TARGET": 2}
    assert journal.cost_totals()["total"] == pytest.approx(2 * TARGET_EXIT_COSTS)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        # One position already committed against a limit of one.
        (dict(), "max_open_positions_in_flight"),
        # Room for two positions, but only one per symbol.
        (
            dict(max_open_positions=2, max_positions_per_symbol=1),
            "max_positions_per_symbol_in_flight",
        ),
        # Room for two positions in two symbols, but 2 * 400 dollars of risk
        # against a heat cap of 100000 * 0.005 = 500 dollars.
        (
            dict(
                max_open_positions=2,
                max_positions_per_symbol=2,
                max_portfolio_heat_pct=0.005,
            ),
            "max_portfolio_heat_in_flight",
        ),
    ],
)
def test_an_order_in_flight_counts_against_every_limit_on_the_book(
    overrides, expected
):
    """`RiskLimitManager` learns of a position at the FILL, so between
    submission and fill the book it checked is one that is about to grow.

    Two signals on consecutive bars both pass the limit and both fill, and the
    account carries more than the configuration allows for as long as both
    live. The position count was already compared here against live orders;
    the per-symbol count and the portfolio HEAT were not, which was a real gap
    -- with `max_open_positions=2` two 400-dollar orders both filled against a
    500-dollar cap, 60% over. All three are the same comparison against the
    same `config.risk` fields, so all three are made.
    """
    cfg = config(risk=risk_config(**overrides))
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13002.0, low=12999.0, close=13001.0))
    bars.append(bar(5, open_=13000.0, high=13002.0, low=12999.0, close=13001.0))
    bars += [bar(index) for index in range(6, 9)]
    result, _, _ = run_loop(
        cfg,
        bars,
        (3, 4),
        plan_factory=lambda ts, *, equity: plan_at(ts, equity=equity, max_hold_bars=100),
    )
    assert result.submissions_declined == 1
    assert len(result.orders) == 1
    directional = [row for row in result.signals if row.action is SignalAction.LONG]
    assert len(directional) == 2
    assert directional[0].order_submitted is True
    assert directional[1].intent_built is True       # the intent was built...
    assert directional[1].order_submitted is False   # ...and deliberately not sent
    assert directional[1].limit_breach == expected
    assert any("NOT submitted" in note for note in result.notes)
    # The heat actually carried never exceeds the configured cap.
    cap = 100_000.0 * float(cfg.risk.max_portfolio_heat_pct)
    assert max(point.open_risk_dollars for point in result.equity) <= cap


def test_the_journal_refuses_a_trade_from_another_run():
    """Mixing runs in one journal mixes their provenance, and the rows could
    not be attributed afterwards -- which is the whole point of carrying a
    run id on every row."""
    journal = Journal(run_id="run-mine")
    with pytest.raises(JournalError, match="belongs to run"):
        journal.record_trade(trade_record(run_id="run-theirs"))
    assert journal.trades == ()
    assert journal.exit_reason_counts() == {}
    assert journal.record_trade(trade_record(run_id="run-mine")) is not None


def test_a_position_open_when_the_data_ends_is_flagged_not_quietly_counted():
    """It is not a trade the strategy chose to close, so it is flattened at
    the last bar's close, pays a full exit's costs, and is reported with a note
    saying to exclude it or report it separately."""
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13002.0, low=12999.0, close=13000.0))
    bars.append(bar(5, open_=13000.0, high=13004.0, low=12999.0, close=13003.0))
    result, _, _ = run_loop(
        config(),
        bars,
        (3,),
        plan_factory=lambda ts, *, equity: plan_at(ts, equity=equity, max_hold_bars=100),
    )
    record = result.trades[0]
    assert record.exit_reason is ExitReason.END_OF_BACKTEST
    assert record.exit_ts == bar(5).close_ts
    assert record.exit_price == pytest.approx(13003.0)        # the last close
    assert record.gross_pnl == pytest.approx(120.0)           # 3 * 20 * 2
    assert record.costs.total > 0.0                           # not a free exit
    assert result.open_at_end == 0
    assert any("END_OF_BACKTEST" in note for note in result.notes)
    assert any("exclude them or" in note for note in result.notes)
    assert result.cost_drag_by_exit_reason["END_OF_BACKTEST"]["n"] == 1.0


def test_the_session_flatten_is_inert_without_a_schedule_and_says_which():
    """A session schedule is not market data, so consulting one is not
    lookahead -- but approximating "the close" with a fixed wall-clock time
    would be wrong on every half day and twice a year at the DST boundary. With
    no calendar, or an instrument that declares no RTH window, the policy
    declines to guess and reports why."""
    cfg = config(session=SessionFilterConfig(flatten_at_session_close=True))
    no_calendar = SessionFlattenPolicy(cfg.session, None, cfg.spec("NQ"))
    assert no_calendar.enabled is False
    assert "no calendar" in no_calendar.note
    assert no_calendar.due(T0, T0 + timedelta(days=3)) is False

    class _Calendar:
        def is_rth(self, ts, spec):  # pragma: no cover - never consulted
            raise AssertionError("an inert policy must not read a calendar")

    no_window = SessionFlattenPolicy(cfg.session, _Calendar(), cfg.spec("NQ"))
    assert cfg.spec("NQ").rth is None
    assert no_window.enabled is False
    assert "no rth window" in no_window.note
    assert no_window.due(T0, T0 + timedelta(days=3)) is False

    # With a window and a calendar it fires -- and only for a position whose
    # SIGNAL was inside the session, because a fill stamped exactly at the
    # close would otherwise read as never having been in it.
    with_rth = NQ.replace(rth=SessionWindow(name="RTH", start="09:30", end="16:00"))
    from flow_model.data import TradingCalendar

    calendar = TradingCalendar()
    policy = SessionFlattenPolicy(cfg.session, calendar, with_rth)
    assert policy.enabled is True
    assert policy.note == ""
    in_session = datetime(2016, 1, 5, 15, 0, tzinfo=UTC)       # 10:00 New York
    assert calendar.is_rth(in_session, with_rth)
    assert policy.due(in_session, in_session + timedelta(minutes=30)) is False
    assert policy.due(in_session, datetime(2016, 1, 5, 21, 30, tzinfo=UTC)) is True
    overnight = datetime(2016, 1, 5, 3, 0, tzinfo=UTC)
    assert policy.due(overnight, overnight + timedelta(hours=1)) is False


def test_turning_per_bar_equity_sampling_off_keeps_the_curves_endpoints():
    """`record_equity_every_bar` governs ROW VOLUME. A return measured over a
    zero-length span, or over a curve with one point, is worse than no return
    at all, so the first and last bars are always sampled and no two samples
    share a timestamp."""
    cfg = config()
    sparse = cfg.replace(backtest=cfg.backtest.replace(record_equity_every_bar=False))
    dense, _, _ = run_loop(cfg, traded_bars(), (3, 7))
    thin, _, _ = run_loop(sparse, traded_bars(), (3, 7))
    assert len(dense.equity) == len(traded_bars())
    assert len(thin.equity) < len(dense.equity)
    stamps = [point.ts for point in thin.equity]
    assert stamps == sorted(stamps)
    assert len(stamps) == len(set(stamps))
    assert stamps[0] == T0
    assert stamps[-1] == traded_bars()[-1].close_ts
    # The realized moves are still there: both runs end at the same equity.
    assert thin.equity[-1].equity == pytest.approx(dense.equity[-1].equity)
    assert thin.provenance.record_equity_every_bar is False


def test_a_run_over_one_bar_with_no_signal_is_a_valid_run_of_one_bar():
    """The smallest non-empty clock. One mark, one signal row, one equity
    point, and nothing that needs two bars to be defined."""
    result, _, _ = run_loop(config(), [bar(0)], ())
    assert result.bars_evaluated == 1
    assert len(result.signals) == 1
    assert len(result.equity) == 1
    assert result.equity[0].ts == T0
    assert result.equity[0].equity == 100_000.0
    assert result.provenance.first_ts == result.provenance.last_ts == T0
    assert result.is_empty is True
    assert result.ledger_diagnostics["marks"] == 1.0


def test_the_journal_records_nothing_for_a_bar_that_was_not_sampled():
    """`mark_to_market` returns None on a bar it chose not to sample, and the
    journal must treat that as "no sample" rather than appending a hole."""
    journal = Journal(run_id="run-test")
    assert journal.record_equity(None) is None
    assert journal.equity == ()
    point = EquityPoint(ts=T0, equity=1.0, realized_equity=1.0)
    assert journal.record_equity(point) is point
    # Replacing the sample at the same instant, rather than appending a second
    # one: two points at one timestamp is a zero-length time step.
    replacement = point.replace(equity=2.0)
    journal.record_equity(replacement, replace_same_ts=True)
    assert journal.equity == (replacement,)


def test_a_position_manager_resolves_one_position_without_touching_the_others():
    """Direct regression on the unit the replay depends on: `resolve_one` must
    not advance anyone's holding time, including its own argument's, because
    "which bars count as held" is a property of the bar and its caller owns
    it. Re-running the whole book here would charge every other open position
    for the same bar twice and fire their time stops early."""
    cfg = config(risk=risk_config(max_open_positions=2, max_positions_per_symbol=2))
    ledger = ledger_of(cfg)
    volatility = VolatilityTracker(window=2)
    volatility.observe(10.0, T0)
    volatility.observe(10.0, T0)
    manager = PositionManager(
        ledger=ledger,
        execution=ExecutionModel.from_config(cfg, "NQ"),
        journal=Journal(run_id="run-test"),
        limits=RiskLimitManager(cfg.risk),
        session_policy=SessionFlattenPolicy(cfg.session, None, cfg.spec("NQ")),
        volatility=volatility,
        session_date=lambda ts: ts.date(),
        kill_switch_drawdown_pct=float(cfg.risk.kill_switch_drawdown_pct),
    )
    first_intent = intent_at(T0)
    second_intent = intent_at(T0 + timedelta(seconds=INTERVAL))
    first = ledger.open_position(
        intent=first_intent,
        entry=entry_fill(first_intent, ts=T0 + timedelta(seconds=INTERVAL)),
        context=entry_context(),
        trade_id="first",
    )
    second = ledger.open_position(
        intent=second_intent,
        entry=entry_fill(second_intent, ts=T0 + timedelta(seconds=2 * INTERVAL)),
        context=entry_context(),
        trade_id="second",
    )
    held = bar(3, open_=13000.0, high=13002.0, low=12999.0, close=13001.0)
    assert manager.resolve_one(second, held.close_ts, bar=held) is None
    assert (first.bars_held, second.bars_held) == (0, 0)
    assert manager.update_open_positions(held.close_ts, bar=held) == ()
    assert (first.bars_held, second.bars_held) == (1, 1)


def short_plan(ts: datetime, *, equity: float = 100_000.0) -> EntryPlan:
    """The mirror of the LONG plan: short 13000, stop 13010, target 12990."""
    return plan_at(
        ts,
        equity=equity,
        side=Side.SHORT,
        entry_price=ENTRY,
        stop_price=13010.0,
        target_price=12990.0,
        structural_stop_price=13010.0,
        structural_target_price=12990.0,
    )


@pytest.mark.parametrize(
    "resolving, reason, exit_price, gross, mae_r, mfe_r",
    [
        # Down to 12985: the 12990 target is through. Favourable is the LOW.
        #   gross = -1 * (12990 - 13000) * 20 * 2 = +400
        #   mfe   = -1 * (12985 - 13000) * 20 * 2 / 400 = +1.5
        #   mae   = -1 * (13001 - 13000) * 20 * 2 / 400 = -0.1  (bar 4's high)
        ("target", ExitReason.TARGET, 12990.0, 400.0, -0.1, 1.5),
        # Up to 13020: the 13010 stop is through. Adverse is the HIGH.
        #   gross = -1 * (13010 - 13000) * 20 * 2 = -400
        #   mae   = -1 * (13020 - 13000) * 20 * 2 / 400 = -2.0
        ("stop", ExitReason.STOP, 13010.0, -400.0, -2.0, 0.1),
    ],
)
def test_a_short_is_the_mirror_of_a_long_in_every_sign(
    resolving, reason, exit_price, gross, mae_r, mfe_r
):
    """A sign error in the PnL or in the excursions is invisible on a LONG-only
    test suite and turns half the trade record into fiction.

    `gross_pnl = side.sign * (exit - entry) * point_value * size`, and for a
    SHORT the FAVOURABLE excursion is the bar's low while the adverse one is
    its high. The expected values are hand-computed in the parameters above.
    """
    bars = warm_bars(4)
    bars.append(bar(4, open_=13000.0, high=13001.0, low=12999.0, close=13000.0))
    if resolving == "target":
        bars.append(bar(5, open_=12999.0, high=13000.0, low=12985.0, close=12988.0))
    else:
        bars.append(bar(5, open_=13001.0, high=13020.0, low=13000.0, close=13018.0))
    bars.append(bar(6))
    result, _, _ = run_loop(config(), bars, (3,), plan_factory=short_plan)
    record = result.trades[0]
    assert record.side is Side.SHORT
    assert record.exit_reason is reason
    assert record.exit_price == pytest.approx(exit_price)
    assert record.gross_pnl == pytest.approx(gross)
    assert record.risk_dollars == pytest.approx(R_DOLLARS)
    assert record.mae_r == pytest.approx(mae_r)
    assert record.mfe_r == pytest.approx(mfe_r)
    assert record.pnl == pytest.approx(record.gross_pnl - record.costs.total)
    assert record.r_multiple == pytest.approx(record.pnl / R_DOLLARS)
    # A win is a win and a loss is a loss, on the net number.
    assert record.is_win is (gross > 0)
    assert record.is_loss is (gross < 0)


def test_a_partial_fill_commits_only_the_risk_it_filled():
    """Section 9: "size filled ~ min(size, participation_rate * bar_volume)".

    Hand-computed: `max_participation_rate=0.05` against a bar volume of 20
    caps the fill at 1 contract of the 2 requested, which is above
    `min_partial_fraction=0.5` so the entry stands.

    The R unit stays the intent's 400 (10 points * 20 dollars * 2 requested),
    because R is fixed at signal time. So capturing only half the position
    reports about HALF an R rather than a full one -- the honest statement that
    half the intended risk unit was captured, and the reason `fill_fraction`
    must be read alongside R on a partial. Re-denominating to the filled size
    would report the same R as a full fill and make the shortfall invisible.
    """
    bars = warm_bars(4)
    bars.append(
        bar(4, open_=13000.0, high=13002.0, low=12999.0, close=13000.0, volume=20.0)
    )
    bars.append(bar(5, open_=13001.0, high=13015.0, low=13000.0, close=13012.0))
    bars.append(bar(6))
    result, _, _ = run_loop(config(), bars, (3,))
    order = result.orders[0]
    assert order.status == "PARTIAL"
    assert order.requested_size == 2.0
    assert order.filled_size == 1.0
    assert order.participation_cap == pytest.approx(1.0)      # 0.05 * 20
    record = result.trades[0]
    assert record.size == 2.0                                  # what was asked
    assert record.filled_size == 1.0                           # what was filled
    assert record.risk_dollars == pytest.approx(400.0)         # the planned unit
    assert record.gross_pnl == pytest.approx(200.0)            # 10 * 20 * 1
    assert record.r_multiple == pytest.approx(record.pnl / 400.0)
    # Half the position captured, so about half an R -- not a full one.
    assert 0.0 < record.r_multiple < 0.5
    assert result.final.equity == pytest.approx(100_000.0 + record.pnl)
