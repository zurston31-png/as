"""Smoke check for `backtest/{ledger,journal,loop}.py` and `database/`.

Deliberately NOT the exhaustive suite -- a dedicated test agent follows. What
this file establishes is that the modules run, that section 9's loop order is
enforced rather than merely documented, and that the EMPTY run -- the live
case, not a hypothetical -- completes and reports honestly.

The committed Phase 6 finding (f952d3e) is that this system takes zero trades:
the highest Flow Score observed over 2500 consecutive bars is 65.23 against a
lowest setup threshold of 70.0. So the first thing tested here is a real run
of the real configured engine over real (synthetic) bars producing a valid,
EMPTY result: no crash, no division by zero, and no 0.0 standing in for
"undefined, n=0".

Two kinds of test double appear, and neither touches a threshold, a weight or
a config default:

* CONSTRUCTED `TradeRecord`s and hand-built bars, for the ledger, the cost
  breakdown and the database round trip.
* a stub signal engine that emits one hand-built `EntryPlan`, for the entry
  path -- which is otherwise unreachable, since the real engine takes no
  trades. It fabricates no performance claim: it exists to prove that a fill
  resolves on a LATER bar, that the ledger books it, and that the record
  round-trips. Lowering a threshold to reach the same code is the move the
  brief forbids, so the plan is built directly instead.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from flow_model.backtest.costs import VolatilityContext
from flow_model.backtest.execution import ExecutionModel
from flow_model.backtest.fills import EntryFill, FillStatus
from flow_model.backtest.journal import Journal
from flow_model.backtest.ledger import (
    EntryContext,
    Ledger,
    LedgerError,
    SessionFlattenPolicy,
)
from flow_model.backtest.loop import (
    MANDATORY_PHASES,
    build_intent,
    SECTION_9_ORDER,
    BacktestError,
    BacktestLoop,
    LoopOrderError,
    LoopPhase,
    PositionManager,
    StoreMarketSource,
    VolatilityTracker,
    _PhaseTracker,
)
from flow_model.config.loader import default_config
from flow_model.core.contracts import (
    Bar,
    CostBreakdown,
    FeatureVector,
    Signal,
    TradeIntent,
)
from flow_model.core.enums import (
    Component,
    DataQuality,
    ExitReason,
    Regime,
    SetupType,
    Side,
    SignalAction,
    SplitPhase,
    WaitReason,
)
from flow_model.data import (
    DataStore,
    SyntheticAdapter,
    SyntheticConfig,
    TradingCalendar,
)
from flow_model.data.series import BarSeries
from flow_model.data.store import SymbolData
from flow_model.risk.limits import RiskLimitManager
from flow_model.database.schema import create_statements
from flow_model.database.store import Database, DatabaseError
from flow_model.signals.engine import EngineDecision, EntryPlan

INTERVAL = 300
T0 = datetime(2016, 1, 5, 15, 0, tzinfo=timezone.utc)

# The measured Phase 7a cost hurdle for one NQ round turn, size 2, a 10-point
# stop (R = 400 dollars) at a vol multiple of 1.25. Used here as the INPUT to
# a reporting check, not re-derived: what is under test is that the drag is
# reported per exit reason and that the two reasons differ.
STOP_EXIT_COSTS = 47.74
TARGET_EXIT_COSTS = 21.49
R_DOLLARS = 400.0


# ---------------------------------------------------------------------------
# A real run of the real engine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    """A 2016 slice, far from the sealed window, long enough to clear warmup.

    The config is the committed default, UNCHANGED. Nothing here moves a
    threshold or a weight, and the sealed window (2023-01-01 to 2025-01-01) is
    neither requested nor opened: these are 2016 bars.
    """
    config = default_config()
    calendar = TradingCalendar()
    adapter = SyntheticAdapter(
        config=SyntheticConfig(),
        seed=config.seed,
        spec_by_symbol=dict(config.instruments),
        interval_seconds=config.data.primary_interval_seconds,
        calendar=calendar,
    )
    store = DataStore(latency_seconds=config.data.data_latency_seconds)
    store.load(
        adapter,
        "NQ",
        date(2016, 1, 4),
        date(2016, 2, 8),
        primary_interval=config.data.primary_interval_seconds,
    )
    return config, calendar, store


@pytest.fixture(scope="module")
def empty_run(synthetic):
    config, calendar, store = synthetic
    loop = BacktestLoop(config, "NQ", calendar=calendar, git_commit="", git_dirty=False)
    return loop.run(StoreMarketSource.from_config(store, "NQ", config))


def test_a_run_today_completes_and_produces_a_valid_empty_result(empty_run, synthetic):
    _, _, store = synthetic
    assert empty_run.trades == ()
    assert empty_run.is_empty is True
    assert empty_run.trade_count == 0
    assert empty_run.provenance.bars_processed == len(store.get("NQ").primary_bars)
    assert empty_run.open_at_end == 0
    assert empty_run.unresolved_orders == 0
    # An equity curve exists even with nothing to revalue, and it has not moved.
    assert len(empty_run.equity) == empty_run.provenance.bars_processed
    assert empty_run.final is not None
    assert empty_run.final.equity == pytest.approx(config_starting_equity())
    assert empty_run.final.realized_pnl == 0.0
    assert empty_run.final.open_positions == 0
    assert any("took NO trades" in note for note in empty_run.notes)


def config_starting_equity() -> float:
    return float(default_config().risk.starting_equity)


def test_the_empty_run_reports_sums_and_counts_but_no_ratio(empty_run):
    """Profit factor with no losses is UNDEFINED, not infinity. A win rate over
    zero trades is UNDEFINED, not 0%. So neither is reported here at all."""
    summary = empty_run.summary()
    for forbidden in ("win_rate", "profit_factor", "expectancy", "sharpe", "sortino"):
        assert forbidden not in summary
    assert "undefined" in summary["ratios"]
    # A reason with no trades is ABSENT, not present with 0.0.
    assert empty_run.cost_drag_by_exit_reason == {}
    assert empty_run.rejection["exit_reasons"] == {}
    # Sums ARE defined over an empty set, and are zero.
    assert empty_run.cost_totals["total"] == 0.0
    assert summary["trades"] == 0
    assert summary["realized_pnl"] == 0.0


def test_every_bar_recorded_a_signal_and_every_wait_carries_its_reason(empty_run):
    bars = empty_run.provenance.bars_processed
    assert empty_run.bars_evaluated == bars
    assert len(empty_run.signals) == bars  # record_wait_signals is on by default
    assert empty_run.waits_dropped == 0
    assert all(row.action is SignalAction.WAIT for row in empty_run.signals)
    assert all(row.wait_reason is not None for row in empty_run.signals)
    histogram = empty_run.rejection["wait_reasons"]
    assert sum(histogram.values()) == bars
    assert sum(empty_run.rejection["blocking_stages"].values()) == bars
    terminal = empty_run.rejection["terminal_gate"]
    assert terminal is not None and terminal["count"] > 0


def test_the_run_records_its_own_provenance(empty_run, synthetic):
    config, _, store = synthetic
    provenance = empty_run.provenance
    assert provenance.config_hash == config.config_hash
    assert provenance.data_hash == store.combined_data_hash()
    assert provenance.seed == config.backtest.seed
    assert provenance.code_version
    assert provenance.run_id.startswith("run-")
    assert provenance.interval_seconds == config.data.primary_interval_seconds
    assert provenance.first_ts is not None and provenance.last_ts is not None
    assert provenance.first_ts < provenance.last_ts
    assert provenance.inputs_fingerprint()
    # The dataset's own description travels with the run, synthetic flag and all.
    assert provenance.dataset["symbol"] == "NQ"
    assert provenance.dataset["synthetic"] is True


def test_two_runs_of_the_same_config_over_the_same_data_are_identical(synthetic):
    """One run, one seed, identical results -- including the run id, which is
    content-derived precisely so that `trade_id` can be stable."""
    config, calendar, store = synthetic
    clock = store.timeline("NQ")[:400]
    first = BacktestLoop(
        config, "NQ", calendar=calendar, git_commit="", git_dirty=False
    ).run(StoreMarketSource.from_config(store, "NQ", config), clock_ns=clock)
    second = BacktestLoop(
        config, "NQ", calendar=calendar, git_commit="", git_dirty=False
    ).run(StoreMarketSource.from_config(store, "NQ", config), clock_ns=clock)
    assert first.provenance.run_id == second.provenance.run_id
    assert first.result_fingerprint() == second.result_fingerprint()
    assert first.provenance.inputs_fingerprint() == second.provenance.inputs_fingerprint()


# ---------------------------------------------------------------------------
# Section 9's order, enforced
# ---------------------------------------------------------------------------


def test_section_9_order_is_declared_as_data_in_section_9_order():
    assert [phase.value for phase in SECTION_9_ORDER] == [
        "mark_to_market",
        "update_open_positions",
        "evaluate",
        "record_signal",
        "build_and_submit",
        "process_pending",
    ]
    # Only the conditional step may be skipped.
    assert LoopPhase.BUILD_AND_SUBMIT not in MANDATORY_PHASES
    assert len(MANDATORY_PHASES) == len(SECTION_9_ORDER) - 1


def test_the_tracker_accepts_the_specified_order_and_the_legal_skip():
    tracker = _PhaseTracker()
    tracker.begin_bar(T0)
    for phase in SECTION_9_ORDER:
        tracker.enter(phase)
    tracker.end_bar()
    tracker.begin_bar(T0 + timedelta(seconds=INTERVAL))
    for phase in MANDATORY_PHASES:  # a WAIT bar submits nothing
        tracker.enter(phase)
    tracker.end_bar()
    assert tracker.bars == 2


@pytest.mark.parametrize(
    "sequence",
    [
        # entries before exits: the limits would see the wrong open heat
        (LoopPhase.MARK_TO_MARKET, LoopPhase.EVALUATE, LoopPhase.UPDATE_OPEN_POSITIONS),
        # marking after the exits: the kill switch would read stale equity
        (LoopPhase.UPDATE_OPEN_POSITIONS, LoopPhase.MARK_TO_MARKET),
        # filling before the signal was recorded, or twice
        (LoopPhase.MARK_TO_MARKET, LoopPhase.MARK_TO_MARKET),
        (
            LoopPhase.MARK_TO_MARKET,
            LoopPhase.UPDATE_OPEN_POSITIONS,
            LoopPhase.PROCESS_PENDING,
            LoopPhase.EVALUATE,
        ),
    ],
)
def test_the_tracker_refuses_a_reordered_bar(sequence):
    tracker = _PhaseTracker()
    tracker.begin_bar(T0)
    with pytest.raises(LoopOrderError):
        for phase in sequence:
            tracker.enter(phase)


def test_the_tracker_refuses_a_bar_that_skipped_a_mandatory_step():
    tracker = _PhaseTracker()
    tracker.begin_bar(T0)
    tracker.enter(LoopPhase.MARK_TO_MARKET)
    tracker.enter(LoopPhase.EVALUATE)
    with pytest.raises(LoopOrderError, match="update_open_positions"):
        tracker.end_bar()


# ---------------------------------------------------------------------------
# Hand-built bars and a constructed plan
# ---------------------------------------------------------------------------


def bar(
    index: int,
    *,
    open_: float = 13000.0,
    high: float = 13002.0,
    low: float = 12998.0,
    close: float = 13000.0,
    volume: float = 100_000.0,
) -> Bar:
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


def plan_at(ts: datetime, *, equity: float) -> EntryPlan:
    """A 1R plan on NQ: size 2, a 10-point stop, so R is 400 dollars."""
    return EntryPlan(
        symbol="NQ",
        signal_ts=ts,
        side=Side.LONG,
        setup=SetupType.SCALP_1R,
        entry_price=13000.0,
        stop_price=12990.0,
        target_price=13010.0,
        size=2.0,
        risk_dollars=R_DOLLARS,
        point_value=20.0,
        equity_at_signal=equity,
        stop_distance=10.0,
        stop_ticks=40.0,
        planned_r_multiple=1.0,
        achievable_rr=1.5,
        structural_stop_price=12990.0,
        structural_target_price=13010.0,
        atr=10.0,
        max_hold_bars=24,
        flow_score=71.0,
        regime=Regime.TRENDING_UP,
        reasons=("test: a constructed plan, not a measured one",),
    )


class _StubEngine:
    """Emits one directional signal, with a hand-built plan, on one bar.

    A test double for the ENTRY PATH only. It reports no score it measured and
    no performance it achieved; the real engine's verdict on this
    configuration (no trades) is tested separately and is not altered here.
    """

    def __init__(
        self, config, symbol: str, signal_indices: tuple[int, ...], equity: float
    ) -> None:
        self.config = config
        self.symbol = symbol
        self.spec = config.spec(symbol)
        self.calendar = None
        self.signal_indices = tuple(signal_indices)
        self.equity = equity
        self.seen: list[datetime] = []

    def session_date(self, ts: datetime) -> date:
        return ts.date()

    def decide(self, view, *, equity=None, limits=None) -> EngineDecision:
        ts = view.now
        self.seen.append(ts)
        features = FeatureVector(
            symbol=self.symbol,
            ts=ts,
            values={"atr": 10.0, "atr_percentile": 0.5},
            quality_by_key={"atr": DataQuality.GOOD, "atr_percentile": DataQuality.GOOD},
        )
        index = round((ts - T0).total_seconds() / INTERVAL)
        if index not in self.signal_indices:
            return EngineDecision(
                signal=Signal(
                    symbol=self.symbol,
                    ts=ts,
                    action=SignalAction.WAIT,
                    wait_reason=WaitReason.NO_SETUP_MATCH,
                    wait_detail="stub: not the signal bar",
                    data_quality=DataQuality.GOOD,
                ),
                features=features,
            )
        return EngineDecision(
            signal=Signal(
                symbol=self.symbol,
                ts=ts,
                action=SignalAction.LONG,
                setup=SetupType.SCALP_1R,
                flow_score=71.0,
                confidence=0.5,
                regime=Regime.TRENDING_UP,
                data_quality=DataQuality.GOOD,
                reference_price=13000.0,
                atr=10.0,
                component_points={Component.STRUCTURE: 15.0},
            ),
            plan=plan_at(ts, equity=self.equity),
            features=features,
        )


def _store_of(bars) -> DataStore:
    series = BarSeries.from_bars(bars)
    store = DataStore(latency_seconds=0.0)
    store.add(SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: series}))
    return store


def _stub_bars(*, signal_index: int, resolve: str) -> list[Bar]:
    """Flat bars, the fill bar, one bar held, then the bar that resolves.

    The held bar is deliberate: mark-to-market runs FIRST in section 9's loop
    and the fill resolves LAST, so the equity sample stamped at the entry bar
    predates the position by construction. A position that survives one
    further bar is what makes the paid entry cost observable on the curve.
    """
    bars = [bar(i) for i in range(signal_index + 1)]
    bars.append(bar(signal_index + 1, open_=13000.0, high=13002.0, low=12999.0, close=13001.0))
    bars.append(bar(signal_index + 2, open_=13001.0, high=13003.0, low=12999.0, close=13000.0))
    if resolve == "target":
        bars.append(bar(signal_index + 3, open_=13001.0, high=13015.0, low=13000.0, close=13012.0))
    else:
        bars.append(bar(signal_index + 3, open_=12999.0, high=13000.0, low=12985.0, close=12988.0))
    bars.append(bar(signal_index + 4, open_=13000.0))
    return bars


@pytest.fixture
def stub_run():
    def _run(resolve: str, *, extra_signal: bool = False):
        config = default_config()
        signal_index = 55  # past the ATR median window, so the fill is priceable
        store = _store_of(_stub_bars(signal_index=signal_index, resolve=resolve))
        indices = (signal_index, signal_index + 1) if extra_signal else (signal_index,)
        engine = _StubEngine(
            config, "NQ", indices, float(config.risk.starting_equity)
        )
        loop = BacktestLoop(
            config, "NQ", engine=engine, git_commit="", git_dirty=False
        )
        return loop.run(StoreMarketSource(store, "NQ"))

    return _run


def test_an_entry_fills_on_a_later_bar_and_becomes_a_trade(stub_run):
    result = stub_run("target")
    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.filled is True
    assert order.fill_bar_offset >= 1
    assert order.fill_ts > order.signal_ts  # section 9's lookahead invariant
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_ts > trade.signal_ts
    assert trade.exit_ts > trade.entry_ts
    assert trade.exit_reason is ExitReason.TARGET
    assert trade.exit_price == pytest.approx(13010.0)
    assert trade.gross_pnl == pytest.approx(400.0)  # 10 points, 2 contracts, 20/point
    assert trade.pnl == pytest.approx(trade.gross_pnl - trade.costs.total)
    assert trade.risk_dollars == pytest.approx(R_DOLLARS)
    assert trade.r_multiple == pytest.approx(trade.pnl / R_DOLLARS)
    assert trade.run_id == result.provenance.run_id
    assert trade.config_hash == result.provenance.config_hash


def test_a_stop_exit_costs_more_than_a_target_exit(stub_run):
    """A target is a resting limit; a stop is a market order paying extra
    slippage. The two cannot be quoted as one drag figure."""
    target = stub_run("target").trades[0]
    stop = stub_run("stop").trades[0]
    assert stop.exit_reason is ExitReason.STOP
    assert stop.costs.total > target.costs.total
    assert stop.cost_drag_r > target.cost_drag_r
    assert target.costs.exit_slippage == 0.0


def test_the_equity_curve_sees_the_trade_and_the_entry_costs(stub_run):
    result = stub_run("target")
    trade = result.trades[0]
    assert result.final is not None
    assert result.final.realized_pnl == pytest.approx(trade.pnl)
    assert result.final.equity == pytest.approx(
        float(default_config().risk.starting_equity) + trade.pnl
    )
    assert result.final.open_positions == 0
    start = float(default_config().risk.starting_equity)

    # Mark-to-market runs FIRST and the fill resolves LAST, so the sample
    # stamped at the entry bar cannot yet see the position. That is section 9's
    # ordering, not a gap in the ledger, and the cost of it is one bar of
    # reporting lag on a position's first appearance.
    at_entry = [p for p in result.equity if p.ts == trade.entry_ts]
    assert len(at_entry) == 1
    assert at_entry[0].open_positions == 0
    assert at_entry[0].realized_equity == pytest.approx(start)

    # On the next bar the position is marked, and the entry's commission and
    # spread are already spent: costs are realized when they are PAID, not when
    # the round turn completes.
    during = [p for p in result.equity if trade.entry_ts < p.ts < trade.exit_ts]
    assert during, "the position was held across at least one marked bar"
    assert during[0].open_positions == 1
    assert during[0].open_risk_dollars == pytest.approx(R_DOLLARS)
    assert during[0].realized_equity < start


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def _intent(ts: datetime, *, size: float = 2.0) -> TradeIntent:
    return plan_at(ts, equity=100_000.0).to_trade_intent().replace(
        size=size, risk_dollars=10.0 * 20.0 * size
    )


def _fill(intent: TradeIntent, *, price: float, size: float, ts: datetime) -> EntryFill:
    return EntryFill(
        status=FillStatus.FILLED if size >= intent.size else FillStatus.PARTIAL,
        symbol=intent.symbol,
        side=intent.side,
        signal_ts=intent.signal_ts,
        submitted_ts=intent.signal_ts,
        eligible_ts=intent.signal_ts,
        order_type="market",
        fill_ts=ts,
        fill_bar_offset=1,
        reference_price=price,
        planned_price=intent.entry_price,
        requested_size=intent.size,
        filled_size=size,
        costs=CostBreakdown(commission=4.5, spread_cost=5.0),
    )


def _context() -> EntryContext:
    return EntryContext(
        flow_score=71.0,
        regime=Regime.TRENDING_UP,
        component_points={Component.STRUCTURE: 15.0, Component.LIQUIDITY: 9.0},
        volatility=10.0,
        vol_percentile=0.5,
        data_quality=DataQuality.GOOD,
        entry_reason="constructed",
    )


def _ledger(config=None, **kwargs) -> Ledger:
    return Ledger(config or default_config(), run_id="run-test", **kwargs)


def test_r_is_the_risk_actually_committed_at_the_fill_that_happened():
    """Half a fill commits half the risk, and the entry that happened defines
    the distance to the stop -- not the entry that was planned."""
    ledger = _ledger()
    intent = _intent(T0)
    position = ledger.open_position(
        intent=intent,
        entry=_fill(intent, price=13001.0, size=1.0, ts=T0 + timedelta(seconds=INTERVAL)),
        context=_context(),
    )
    assert position.size == 1.0
    assert position.risk_dollars == pytest.approx(abs(13001.0 - 12990.0) * 20.0 * 1.0)
    assert position.risk_dollars != intent.risk_dollars


def test_entry_costs_are_realized_when_they_are_paid():
    ledger = _ledger()
    intent = _intent(T0)
    ledger.open_position(
        intent=intent,
        entry=_fill(intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=INTERVAL)),
        context=_context(),
    )
    start = float(default_config().risk.starting_equity)
    assert ledger.open_entry_costs == pytest.approx(9.5)
    assert ledger.realized_equity == pytest.approx(start - 9.5)
    # Marked at the entry price, the unrealized PnL is zero and equity is the
    # starting equity minus what has actually been spent.
    point = ledger.mark_to_market(T0 + timedelta(seconds=INTERVAL), {"NQ": 13000.0})
    assert point is not None
    assert point.equity == pytest.approx(start - 9.5)
    assert ledger.unrealized_pnl == pytest.approx(0.0)
    moved = ledger.mark_to_market(T0 + timedelta(seconds=2 * INTERVAL), {"NQ": 13005.0})
    assert moved is not None
    assert moved.equity == pytest.approx(start - 9.5 + 5.0 * 20.0 * 2.0)
    assert moved.open_risk_dollars == pytest.approx(400.0)


def test_excursions_come_from_the_bar_extremes_with_the_contract_s_signs():
    ledger = _ledger()
    intent = _intent(T0)
    position = ledger.open_position(
        intent=intent,
        entry=_fill(intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=INTERVAL)),
        context=_context(),
    )
    position.observe(bar(2, open_=13000.0, high=13004.0, low=12996.0, close=13000.0))
    assert position.mfe_r == pytest.approx(4.0 * 20.0 * 2.0 / 400.0)
    assert position.mae_r == pytest.approx(-4.0 * 20.0 * 2.0 / 400.0)
    # Never the wrong sign: a trade that was never underwater has mae_r 0.0.
    fresh = _ledger().open_position(
        intent=intent,
        entry=_fill(intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=INTERVAL)),
        context=_context(),
    )
    fresh.observe(bar(2, open_=13001.0, high=13006.0, low=13000.0, close=13005.0))
    assert fresh.mae_r == 0.0
    assert fresh.mfe_r > 0.0


def test_mark_to_market_refuses_to_go_backwards_or_to_mark_at_a_non_price():
    ledger = _ledger()
    ledger.mark_to_market(T0 + timedelta(seconds=INTERVAL), {"NQ": 13000.0})
    with pytest.raises(LedgerError, match="backwards"):
        ledger.mark_to_market(T0, {"NQ": 13000.0})
    intent = _intent(T0)
    ledger.open_position(
        intent=intent,
        entry=_fill(intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=INTERVAL)),
        context=_context(),
    )
    with pytest.raises(LedgerError, match="finite"):
        ledger.mark_to_market(T0 + timedelta(seconds=9 * INTERVAL), {"NQ": float("nan")})


def test_compound_equity_decides_what_the_next_trade_is_sized_from():
    config = default_config()
    compounding = _ledger(config)
    fixed = _ledger(config.replace(risk=config.risk.replace(compound_equity=False)))
    assert compounding.compound_equity is True
    assert fixed.compound_equity is False
    for ledger in (compounding, fixed):
        intent = _intent(T0)
        ledger.open_position(
            intent=intent,
            entry=_fill(intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=INTERVAL)),
            context=_context(),
        )
        ledger.mark_to_market(T0 + timedelta(seconds=2 * INTERVAL), {"NQ": 13050.0})
    assert compounding.sizing_equity() == pytest.approx(compounding.equity)
    assert fixed.sizing_equity() == pytest.approx(fixed.starting_equity)
    assert fixed.equity != pytest.approx(fixed.starting_equity)


def test_the_time_stop_is_the_ledger_s_and_counts_bars_after_the_entry_bar():
    ledger = _ledger()
    intent = _intent(T0)
    position = ledger.open_position(
        intent=intent,
        entry=_fill(intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=INTERVAL)),
        context=_context(),
    )
    assert ledger.time_stop_due(position) is False
    ledger.advance_bar(position, position.entry_ts)  # the entry bar itself
    assert position.bars_held == 0
    for step in range(1, int(intent.max_hold_bars) + 1):
        ledger.advance_bar(position, position.entry_ts + timedelta(seconds=INTERVAL * step))
    assert position.bars_held == intent.max_hold_bars
    assert ledger.time_stop_due(position) is True


def test_session_flatten_fires_only_for_a_position_entered_inside_rth():
    config = default_config()
    calendar = TradingCalendar()
    spec = config.spec("NQ")
    policy = SessionFlattenPolicy(config.session, calendar, spec)
    assert policy.enabled is True
    in_rth = datetime(2016, 1, 5, 15, 0, tzinfo=timezone.utc)  # 10:00 New York
    assert calendar.is_rth(in_rth, spec)
    assert policy.due(in_rth, in_rth + timedelta(minutes=30)) is False
    assert policy.due(in_rth, datetime(2016, 1, 5, 21, 30, tzinfo=timezone.utc)) is True
    assert policy.due(in_rth, datetime(2016, 1, 6, 15, 0, tzinfo=timezone.utc)) is True
    overnight = datetime(2016, 1, 5, 3, 0, tzinfo=timezone.utc)
    assert policy.due(overnight, overnight + timedelta(hours=1)) is False
    # No calendar: inert, and it says so rather than guessing a wall-clock close.
    blind = SessionFlattenPolicy(config.session, None, spec)
    assert blind.enabled is False
    assert "no calendar" in blind.note


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------


def _wait(ts: datetime, reason: WaitReason) -> Signal:
    return Signal(
        symbol="NQ",
        ts=ts,
        action=SignalAction.WAIT,
        wait_reason=reason,
        wait_detail="constructed",
        data_quality=DataQuality.GOOD,
    )


def test_dropping_wait_rows_never_drops_the_wait_counts():
    """`record_wait_signals` governs row volume, not the rejection analysis."""
    kept = Journal(run_id="run-test", record_wait_signals=True)
    dropped = Journal(run_id="run-test", record_wait_signals=False)
    for index, reason in enumerate(
        (WaitReason.SCORE_BELOW_THRESHOLD, WaitReason.WARMUP, WaitReason.WARMUP)
    ):
        ts = T0 + timedelta(seconds=INTERVAL * index)
        for journal in (kept, dropped):
            journal.record_signal(_wait(ts, reason), blocking_stage="flow_score")
    assert len(kept.signals) == 3
    assert len(dropped.signals) == 0
    assert dropped.waits_dropped == 3
    assert kept.wait_counts() == dropped.wait_counts() == {
        "score_below_threshold": 1,
        "warmup": 2,
    }
    assert dropped.signals_seen == 3
    assert dropped.terminal_gate() == ("warmup", 2)
    assert kept.blocking_stage_counts() == {"flow_score": 3}


def test_cost_drag_is_broken_down_by_exit_reason_and_never_averaged(make_trade):
    """The measured hurdle: 47.74 dollars through the stop and 21.49 through
    the target on the same 400-dollar R. One average figure would overstate
    the cost of the trades that win and understate the ones that lose."""
    journal = Journal(run_id="test_run")
    journal.record_trade(
        make_trade(
            trade_id="stopped",
            exit_price=17980.0,
            exit_reason=ExitReason.STOP,
            costs=CostBreakdown(commission=9.0, exit_slippage=20.0, spread_cost=18.74),
            point_value=20.0,
            size=1.0,
        )
    )
    journal.record_trade(
        make_trade(
            trade_id="targeted",
            exit_reason=ExitReason.TARGET,
            costs=CostBreakdown(commission=9.0, entry_slippage=7.49, spread_cost=5.0),
            point_value=20.0,
            size=1.0,
        )
    )
    drag = journal.cost_drag_by_exit_reason()
    assert set(drag) == {"STOP", "TARGET"}
    assert drag["STOP"]["n"] == 1.0
    assert drag["STOP"]["costs"] == pytest.approx(STOP_EXIT_COSTS)
    assert drag["TARGET"]["costs"] == pytest.approx(TARGET_EXIT_COSTS)
    assert drag["STOP"]["mean_drag_r"] == pytest.approx(STOP_EXIT_COSTS / R_DOLLARS)
    assert drag["TARGET"]["mean_drag_r"] == pytest.approx(TARGET_EXIT_COSTS / R_DOLLARS)
    assert drag["STOP"]["mean_drag_r"] > drag["TARGET"]["mean_drag_r"]
    # Reasons with no trades are absent, not zero.
    assert "TIME_STOP" not in drag
    assert journal.cost_totals()["total"] == pytest.approx(
        STOP_EXIT_COSTS + TARGET_EXIT_COSTS
    )


def test_an_empty_journal_reports_no_drag_rather_than_a_zero_one():
    journal = Journal(run_id="run-test")
    assert journal.cost_drag_by_exit_reason() == {}
    assert journal.exit_reason_counts() == {}
    assert journal.terminal_gate() is None
    assert journal.cost_totals() == {
        "commission": 0.0,
        "exchange_fees": 0.0,
        "entry_slippage": 0.0,
        "exit_slippage": 0.0,
        "spread_cost": 0.0,
        "total": 0.0,
    }


# ---------------------------------------------------------------------------
# The volatility ratio
# ---------------------------------------------------------------------------


def test_an_unfilled_atr_window_is_unmeasured_and_not_a_ratio_of_one():
    tracker = VolatilityTracker(window=3)
    assert tracker.context().is_measured is False
    tracker.observe(10.0, T0)
    assert tracker.context().is_measured is False
    assert tracker.context().multiple is None
    tracker.observe(12.0, T0)
    tracker.observe(14.0, T0)
    context = tracker.context()
    assert context.is_measured is True
    assert context.median_atr == pytest.approx(12.0)
    assert context.multiple == pytest.approx(14.0 / 12.0)
    assert tracker.context(atr=15.0).multiple == pytest.approx(15.0 / 12.0)
    # A missing or non-positive ATR is not an observation.
    tracker.observe(None, T0)
    tracker.observe(0.0, T0)
    assert tracker.observations == 3


def test_a_window_that_cannot_produce_a_median_is_refused():
    with pytest.raises(BacktestError, match="median"):
        VolatilityTracker(window=1)


def test_an_unmeasured_ratio_declines_an_entry_rather_than_charging_zero():
    """The behaviour the tracker's window rule depends on, asserted against
    the execution model rather than assumed."""
    config = default_config()
    model = ExecutionModel.from_config(config, "NQ")
    assert config.execution.vol_slippage_coefficient > 0.0
    estimate = model.costs.slippage(size=2.0, vol=VolatilityContext.unmeasured("test"))
    assert estimate.measurable is False


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------


def test_the_schema_declares_every_table_section_12_names():
    import sqlite3

    connection = sqlite3.connect(":memory:")
    for statement in create_statements():
        connection.execute(statement)
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"runs", "trades", "signals", "equity", "experiments", "seal_audit"} <= tables


def test_a_trade_record_round_trips_through_sqlite(tmp_path, make_trade):
    """Write it, read it back, get an equal object -- all 41 input fields."""
    original = make_trade(
        costs=CostBreakdown(commission=4.5, exchange_fees=0.74, entry_slippage=5.0),
        split_label=SplitPhase.TRAIN,
        mae_r=-0.4731,
        mfe_r=1.2345,
    )
    with Database(tmp_path / "round_trip.db") as database:
        database.write_trades([original])
        read_back = database.read_trades()
    assert len(read_back) == 1
    assert read_back[0] == original
    assert read_back[0].costs == original.costs
    assert read_back[0].r_multiple == original.r_multiple


def test_a_float_survives_storage_as_a_number(tmp_path):
    """The committed serialization bug: `canonical_json` stringifies floats for
    hashing, and a value stored with it read back as the string
    '7.341000000000e-01'. Storage and hashing are different jobs."""
    journal = Journal(run_id="run-test")
    journal.record_signal(_wait(T0, WaitReason.SCORE_BELOW_THRESHOLD))
    row = journal.signals[0].replace(
        component_points={Component.STRUCTURE: 0.7341}, flow_score=65.23
    )
    from flow_model.database.store import signal_row
    from flow_model.database.schema import SIGNAL_COLUMNS

    stored = dict(zip(SIGNAL_COLUMNS, signal_row(row)))
    assert '"0.7341"' not in stored["component_points_json"]
    assert "0.7341" in stored["component_points_json"]
    assert isinstance(stored["flow_score"], float)


def test_a_whole_empty_run_persists_and_reads_back(tmp_path, empty_run):
    path = tmp_path / "runs.db"
    with Database(path) as database:
        run_id = database.write_run(empty_run)
        assert database.read_trades(run_id) == ()
        signals = database.read_signals(run_id)
        assert len(signals) == len(empty_run.signals)
        assert signals[0].wait_reason is not None
        equity = database.read_equity(run_id)
        assert len(equity) == len(empty_run.equity)
        stored = database.read_run(run_id)
        assert stored is not None
        assert stored["trade_count"] == 0
        assert stored["result_fingerprint"] == empty_run.result_fingerprint()
        assert stored["rejection"]["terminal_gate"]["count"] > 0
        assert stored["compound_equity"] is True
        counts = database.counts()
        assert counts["runs"] == 1
        assert counts["trades"] == 0
        # Writing the same run twice is a primary key violation, not a silent
        # doubling of the trade count.
        with pytest.raises(DatabaseError, match="already stored"):
            database.write_run(empty_run)
        assert database.write_run(empty_run, replace=True) == run_id
        assert database.counts()["runs"] == 1
        database.delete_run(run_id)
        assert database.counts() == dict(counts, runs=0, signals=0, equity=0)


def test_a_traded_run_round_trips_including_its_signal_rows(tmp_path, stub_run):
    result = stub_run("target")
    with Database(tmp_path / "traded.db") as database:
        run_id = database.write_run(result)
        assert database.read_trades(run_id) == result.trades
        assert database.read_orders(run_id) == result.orders
        assert database.read_signals(run_id) == result.signals
        assert database.read_equity(run_id) == result.equity
        row = database.read_run(run_id)
        assert row is not None
        assert row["trade_count"] == 1
        assert row["cost_drag"]["TARGET"]["n"] == 1.0


def test_the_seal_audit_table_mirrors_the_log_without_opening_anything(tmp_path):
    """An empty or absent audit log imports zero rows and creates no file: the
    JSONL stays the source of truth for the one-open budget."""
    path = tmp_path / "seal_audit.jsonl"
    with Database(tmp_path / "seal.db") as database:
        assert database.import_seal_audit(path) == 0
    assert not path.exists()

def test_a_clock_with_no_bars_runs_and_reports_nothing(synthetic):
    """The other n=0: no bars at all. A run over an empty clock is a run that
    measured nothing, and it says so rather than dividing by it."""
    config, calendar, store = synthetic
    loop = BacktestLoop(config, "NQ", calendar=calendar, git_commit="", git_dirty=False)
    result = loop.run(StoreMarketSource.from_config(store, "NQ", config), clock_ns=[])
    assert result.is_empty is True
    assert result.bars_evaluated == 0
    assert result.equity == ()
    assert result.signals == ()
    assert result.provenance.bars_processed == 0
    assert result.provenance.first_ts is None
    assert result.summary()["trades"] == 0
    assert result.cost_drag_by_exit_reason == {}


def test_resolving_one_position_does_not_advance_another_s_holding_time():
    """The holding time is a property of the bar. When a second entry fills,
    the bar it filled on must not count against the first position twice --
    that fires a time stop early, and the trade it cut short is invisible
    afterwards."""
    config = default_config()
    ledger = _ledger(config)
    volatility = VolatilityTracker(window=2)
    volatility.observe(10.0, T0)
    volatility.observe(10.0, T0)
    manager = PositionManager(
        ledger=ledger,
        execution=ExecutionModel.from_config(config, "NQ"),
        journal=Journal(run_id="run-test"),
        limits=RiskLimitManager(config.risk),
        session_policy=SessionFlattenPolicy(config.session, None, config.spec("NQ")),
        volatility=volatility,
        session_date=lambda ts: ts.date(),
        kill_switch_drawdown_pct=float(config.risk.kill_switch_drawdown_pct),
    )
    first_intent = _intent(T0)
    second_intent = _intent(T0 + timedelta(seconds=INTERVAL))
    first = ledger.open_position(
        intent=first_intent,
        entry=_fill(
            first_intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=INTERVAL)
        ),
        context=_context(),
        trade_id="first",
    )
    second = ledger.open_position(
        intent=second_intent,
        entry=_fill(
            second_intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=2 * INTERVAL)
        ),
        context=_context(),
        trade_id="second",
    )
    held = bar(3, open_=13000.0, high=13002.0, low=12999.0, close=13001.0)

    assert manager.resolve_one(second, held.close_ts, bar=held) is None
    assert (first.bars_held, second.bars_held) == (0, 0)

    assert manager.update_open_positions(held.close_ts, bar=held) == ()
    assert (first.bars_held, second.bars_held) == (1, 1)


def test_a_bar_the_symbol_never_printed_holds_the_position_and_is_counted():
    """No bar means no price to exit at, so the position is held -- and the
    gap is counted, because an un-updated mark freezes the drawdown."""
    config = default_config()
    ledger = _ledger(config)
    volatility = VolatilityTracker(window=2)
    volatility.observe(10.0, T0)
    volatility.observe(10.0, T0)
    manager = PositionManager(
        ledger=ledger,
        execution=ExecutionModel.from_config(config, "NQ"),
        journal=Journal(run_id="run-test"),
        limits=RiskLimitManager(config.risk),
        session_policy=SessionFlattenPolicy(config.session, None, config.spec("NQ")),
        volatility=volatility,
        session_date=lambda ts: ts.date(),
        kill_switch_drawdown_pct=float(config.risk.kill_switch_drawdown_pct),
    )
    intent = _intent(T0)
    position = ledger.open_position(
        intent=intent,
        entry=_fill(intent, price=13000.0, size=2.0, ts=T0 + timedelta(seconds=INTERVAL)),
        context=_context(),
    )
    gap_ts = T0 + timedelta(seconds=4 * INTERVAL)
    assert manager.update_open_positions(gap_ts, bar=None) == ()
    assert manager.bars_without_a_bar == 1
    assert position.bars_held == 1
    ledger.mark_to_market(gap_ts, {})
    assert position.stale_mark_bars == 1
    assert ledger.diagnostics()["stale_marks"] == 1.0
    assert any("no price" in note for note in ledger.notes)

# ---------------------------------------------------------------------------
# The risk manager's output
# ---------------------------------------------------------------------------


def _long_decision(*, equity: float) -> EngineDecision:
    plan = plan_at(T0, equity=equity)
    return EngineDecision(
        signal=Signal(
            symbol="NQ",
            ts=T0,
            action=SignalAction.LONG,
            setup=SetupType.SCALP_1R,
            flow_score=71.0,
            regime=Regime.TRENDING_UP,
            data_quality=DataQuality.GOOD,
        ),
        plan=plan,
    )


def test_the_plan_s_intent_is_returned_untouched_when_the_equity_agrees():
    config = default_config()
    decision = _long_decision(equity=100_000.0)
    intent = build_intent(
        decision, config=config, spec=config.spec("NQ"), sizing_equity=100_000.0
    )
    assert intent is not None
    assert intent == decision.plan.to_trade_intent()


def test_fixed_fractional_sizing_re_sizes_the_intent_and_records_that_it_did():
    """`compound_equity=False` honoured without lying to the limit checks: the
    engine is handed real mark-to-market equity, and the SIZE is taken off the
    starting equity here, through the same pure sizing function."""
    config = default_config()
    decision = _long_decision(equity=250_000.0)  # what compounding would have used
    intent = build_intent(
        decision, config=config, spec=config.spec("NQ"), sizing_equity=100_000.0
    )
    assert intent is not None
    assert intent.equity_at_signal == pytest.approx(100_000.0)
    # 0.5% of 100k is 500 dollars against 200 dollars of risk per contract.
    assert intent.size == pytest.approx(2.0)
    assert intent.risk_dollars == pytest.approx(400.0)
    assert any("FIXED equity" in reason for reason in intent.reasons)


def test_a_wait_builds_no_intent_and_a_planless_signal_is_a_bug():
    config = default_config()
    wait = EngineDecision(signal=_wait(T0, WaitReason.WARMUP))
    assert build_intent(wait, config=config, spec=config.spec("NQ")) is None
    planless = EngineDecision(signal=_long_decision(equity=100_000.0).signal)
    with pytest.raises(BacktestError, match="no EntryPlan"):
        build_intent(planless, config=config, spec=config.spec("NQ"))


# ---------------------------------------------------------------------------
# The source's own firewall
# ---------------------------------------------------------------------------


def test_the_source_cursor_cannot_rewind_or_be_asked_about_the_future():
    store = _store_of([bar(i) for i in range(5)])
    source = StoreMarketSource(store, "NQ")
    with pytest.raises(BacktestError, match="not been advanced"):
        source.bar()
    second = bar(2).close_ts
    source.advance(int(second.timestamp() * 1_000_000_000))
    assert source.bar() is not None
    assert source.bar().close_ts == second
    with pytest.raises(BacktestError, match="backwards"):
        source.advance(int(bar(1).close_ts.timestamp() * 1_000_000_000))
    with pytest.raises(BacktestError, match="lookahead"):
        source.quote(bar(4).close_ts)

def test_a_second_signal_while_an_order_is_in_flight_is_not_submitted(stub_run):
    """`max_open_positions` counts orders in flight as well as open positions.

    The limit manager learns about a position at the FILL, so between
    submission and fill the book it checks is one that is about to grow. Two
    signals on consecutive bars would otherwise both pass and both fill, and
    the account would carry double the intended size."""
    result = stub_run("target", extra_signal=True)
    assert result.submissions_declined == 1
    assert len(result.orders) == 1
    assert len(result.trades) == 1
    directional = [row for row in result.signals if row.action is SignalAction.LONG]
    assert len(directional) == 2
    assert directional[0].order_submitted is True
    assert directional[1].intent_built is True
    assert directional[1].order_submitted is False
    assert directional[1].limit_breach == "max_open_positions_in_flight"
    assert any("NOT submitted" in note for note in result.notes)

def test_record_equity_every_bar_governs_volume_not_the_curve_s_endpoints(synthetic):
    """Turning per-bar sampling off still leaves a curve with two endpoints and
    no duplicate timestamps -- a return measured over a zero-length span is
    worse than no return at all."""
    config, calendar, store = synthetic
    sparse = config.replace(
        backtest=config.backtest.replace(record_equity_every_bar=False)
    )
    clock = store.timeline("NQ")[:300]
    result = BacktestLoop(
        sparse, "NQ", calendar=calendar, git_commit="", git_dirty=False
    ).run(StoreMarketSource.from_config(store, "NQ", sparse), clock_ns=clock)
    stamps = [point.ts for point in result.equity]
    assert len(stamps) == len(set(stamps))
    assert len(stamps) == 2  # nothing was ever realized over a zero-trade run
    assert stamps[0] < stamps[-1]
    assert result.provenance.record_equity_every_bar is False

