"""Tests for `flow_model/validation/robustness.py` (ARCHITECTURE.md 10).

The module's whole job is to stop this project fooling itself, so these tests
are written to attack it rather than to confirm it. What is checked, and why
in this order:

1. **No best cell is extractable.** Checked through the PUBLIC API --
   `dir()`, `model_fields`, `typing.get_type_hints()` -- never by grepping
   source text, because the module's own prose contains the words a text
   guard would flag (`GRID_READING_NOTE` says "No cell here is a
   recommendation"), and a guard that flags accurate prose has cost this
   project a round already. Every such check is proved non-vacuous against a
   stub that DOES expose a winner.
2. **The n=0 path first.** The committed configuration takes zero trades
   (PROVENANCE.md 2.4: 283 of 19542 bars reach the scorer, max Flow Score
   66.991221 against a 70.0 floor), so every fold and every grid cell at the
   committed config is empty. That is the live case, and `n=0` must report
   UNDEFINED rather than 0.0, while counts and sums stay measured.
3. **Every degradation only hurts** -- and where one CAN flatter, that it is
   detected and explained rather than hidden. One is found and measured
   (`wider_spread` against a quote wider than the pinned override).
4. **Seeded determinism with the global NumPy state unconsulted**, proved by
   reseeding the global state between runs and by reading the global
   Mersenne-Twister buffer before and after.
5. **The seal is untouched by any sweep**, with the runner call count proving
   nothing ran.
6. **The wait-horizon fix (f44ddb9) re-verified** on the real loop, against
   the exact trade from PROVENANCE.md 2.6: the regression reported TARGET
   +0.9869R where the truth is STOP -1.0756R.

Test doubles, and what they are not: a stub signal engine emitting one
hand-built `EntryPlan` (the pattern `tests/unit/test_backtest_loop_smoke.py`
established), and constructed `TradeRecord`s. The real engine takes no
trades, so the entry path is otherwise unreachable. Nothing here changes a
threshold, a weight or a config default -- every varied config is built in
process with `apply_overrides`, and `config/defaults.yaml` is asserted
untouched. No test reads the sealed window 2023-01-01..2025-01-01.
"""

from __future__ import annotations

import math
import re
import typing
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.analytics.metrics import (
    Comparison,
    MetricValue,
    Undefined,
    compute_trade_metrics,
)
from flow_model.backtest.fill_timing import PendingOrder, min_fill_offset
from flow_model.backtest.loop import (
    BacktestLoop,
    RunProvenance,
    RunResult,
    StoreMarketSource,
)
from flow_model.config.loader import default_config
from flow_model.core.contracts import (
    Bar,
    CostBreakdown,
    FeatureVector,
    Signal,
    TradeRecord,
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
from flow_model.core.model import FrozenModel
from flow_model.data.series import BarSeries, QuoteSeries, to_ns_array
from flow_model.data.store import DataStore, SymbolData
from flow_model.signals.engine import EngineDecision, EntryPlan
from flow_model.validation.experiment_log import ExperimentLog
from flow_model.validation.hypotheses import by_id
from flow_model.validation.robustness import (
    DEGRADATION_METRICS,
    FRAGILITY_METRIC,
    GRID_CELL_METRICS,
    GRID_READING_NOTE,
    HAIRCUT_METRICS,
    HYPOTHESIS_DRIVER_PATHS,
    NEUTRAL_SEVERITY,
    SWEEPABLE_PHASES,
    AxisMode,
    CellResult,
    DecayStep,
    DegradationKind,
    DegradationMode,
    DegradationResult,
    DegradationSpec,
    DirectionCheck,
    FragilityReport,
    FragilityVerdict,
    GridPoint,
    HypothesisReport,
    ParameterAxis,
    RobustnessError,
    RobustnessReport,
    StoreRunner,
    _degradation_notes,
    _post_hoc_degradation,
    apply_overrides,
    axis_for_driver,
    default_axes,
    default_degradations,
    fold_profit_concentration,
    fragility_report,
    hypothesis_reports,
    read_path,
    risk_pct_axis,
    robustness_report,
    run_degradations,
    run_grid,
    score_threshold_axis,
    wait_horizon_violations,
    win_rate_haircut_metrics,
)
from flow_model.validation.splits import (
    DataAccessGuard,
    DateRange,
    SealAudit,
    SealedDataAccessError,
    Split,
    SplitRegistry,
)

# The sealed window, from ARCHITECTURE.md 0.6. Named so the tests can assert
# it is never requested; no test ranges over it.
SEALED = DateRange(start=date(2023, 1, 1), end=date(2025, 1, 1))

INTERVAL = 300
T0 = datetime(2016, 1, 5, 15, 0, tzinfo=timezone.utc)
SIGNAL_INDEX = 55  # past the ATR median window, so a fill is priceable
R_DOLLARS = 400.0


# ---------------------------------------------------------------------------
# Splits: a research-window registry, well clear of the seal
# ---------------------------------------------------------------------------


def research_registry() -> SplitRegistry:
    """Two sweepable folds plus the evaluation phases a sweep must refuse."""
    return SplitRegistry(
        splits=(
            Split(
                name="train_a",
                phase=SplitPhase.TRAIN,
                range=DateRange(start=date(2015, 1, 1), end=date(2016, 1, 1)),
            ),
            Split(
                name="train_b",
                phase=SplitPhase.TRAIN,
                range=DateRange(start=date(2016, 1, 1), end=date(2017, 1, 1)),
            ),
            Split(
                name="val_a",
                phase=SplitPhase.VALIDATION,
                range=DateRange(start=date(2017, 2, 1), end=date(2018, 1, 1)),
            ),
            Split(
                name="test_a",
                phase=SplitPhase.TEST,
                range=DateRange(start=date(2018, 2, 1), end=date(2019, 1, 1)),
            ),
            Split(
                name="sealed_oos",
                phase=SplitPhase.SEALED_OOS,
                range=DateRange(start=SEALED.start, end=SEALED.end),
            ),
        )
    )


def one_fold() -> Split:
    return research_registry().get("train_b")


@pytest.fixture
def registry() -> SplitRegistry:
    return research_registry()


@pytest.fixture
def split() -> Split:
    return one_fold()


@pytest.fixture
def log(tmp_path) -> ExperimentLog:
    """An in-memory log whose repo_root is NOT a git checkout.

    Keeps the 2 `git` subprocesses per logged row out of a 48-cell sweep;
    `git_state` is tested in `tests/unit/test_experiment_log.py`.
    """
    with ExperimentLog(":memory:", repo_root=tmp_path) as handle:
        yield handle


# ---------------------------------------------------------------------------
# Axes used by most tests: small, so a grid is countable by hand
# ---------------------------------------------------------------------------


def small_axes(config) -> tuple[ParameterAxis, ...]:
    """A 3x2 grid whose values are drawn from `config.robustness`, so nothing
    here invents a sweep constant. The committed coordinate (70.0, 0.005) is
    present, which is what a fragility baseline needs."""
    return (
        ParameterAxis(
            name="flow_score_threshold",
            paths=tuple(
                f"setups.{s.value}.min_flow_score" for s in sorted(config.setups)
            ),
            values=(65.0, 70.0, 75.0),
            mode=AxisMode.PRESERVE_OFFSETS,
            unit="points",
        ),
        ParameterAxis(
            name="risk_per_trade_pct",
            paths=("risk.risk_per_trade_pct",),
            values=(0.005, 0.0075),
            unit="fraction of equity",
        ),
    )


# ---------------------------------------------------------------------------
# Constructed trades, with R controlled exactly
# ---------------------------------------------------------------------------


def r_trade(
    index: int,
    r: float,
    *,
    entry: float = 13000.0,
    stop: float = 12990.0,
    size: float = 1.0,
    point_value: float = 20.0,
    costs: CostBreakdown | None = None,
    split_label: SplitPhase | None = SplitPhase.TRAIN,
) -> TradeRecord:
    """A self-consistent record whose realized R is exactly `r`.

    `risk_dollars` is the PLANNED unit -- |planned_entry - stop| * point_value
    * size (ARCHITECTURE.md 0.2, PROVENANCE.md 4) -- and the exit price is
    solved from it, so the record's own validators accept it and `r_multiple`
    comes out at `r` rather than near it.
    """
    costs = costs if costs is not None else CostBreakdown()
    risk = abs(entry - stop) * point_value * size
    net = r * risk
    gross = net + costs.total
    exit_price = entry + gross / (point_value * size)
    signal_ts = T0 + timedelta(minutes=10 * index)
    return TradeRecord(
        trade_id=f"t{index:03d}",
        run_id="test_run",
        symbol="NQ",
        config_hash="testhash",
        signal_ts=signal_ts,
        entry_ts=signal_ts + timedelta(minutes=5),
        exit_ts=signal_ts + timedelta(minutes=30),
        bars_held=5,
        side=Side.LONG,
        setup=SetupType.SCALP_1R,
        planned_entry=entry,
        stop_price=stop,
        target_price=entry + abs(entry - stop),
        entry_price=entry,
        exit_price=exit_price,
        size=size,
        filled_size=size,
        point_value=point_value,
        exit_reason=ExitReason.TARGET if net > 0 else ExitReason.STOP,
        risk_dollars=risk,
        gross_pnl=gross,
        costs=costs,
        pnl=net,
        mae_r=-0.3 if net > 0 else -1.0,
        mfe_r=1.0 if net > 0 else 0.1,
        flow_score=75.0,
        regime=Regime.LOW_VOL,
        split_label=split_label,
    )


def trades_from(values, **kwargs) -> tuple[TradeRecord, ...]:
    return tuple(r_trade(i, r, **kwargs) for i, r in enumerate(values))


def run_result(
    trades: tuple[TradeRecord, ...] = (), *, config_hash: str = "h", seed: int = 7
) -> RunResult:
    return RunResult(
        provenance=RunProvenance(
            run_id="r",
            symbol="NQ",
            config_hash=config_hash,
            data_hash="datahash",
            seed=seed,
            split_label=SplitPhase.TRAIN,
            split_name="train_b",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            starting_equity=100_000.0,
        ),
        trades=trades,
    )


def cell(
    *,
    split_name: str = "train_b",
    score: float = 70.0,
    risk: float = 0.005,
    n: int = 0,
    expectancy: float | None = None,
    profit_factor: float | None = None,
) -> CellResult:
    """A CellResult with hand-set metrics, for the fragility arithmetic."""
    metrics = {
        "expectancy": (
            MetricValue.undefined_because("expectancy_r", Undefined.NO_TRADES, n=n)
            if expectancy is None
            else MetricValue.measured("expectancy_r", expectancy, n=n, unit="R")
        ),
        "profit_factor": (
            MetricValue.undefined_because("profit_factor", Undefined.NO_TRADES, n=n)
            if profit_factor is None
            else MetricValue.measured("profit_factor", profit_factor, n=n)
        ),
    }
    return CellResult(
        point=GridPoint(
            coordinates=(
                ("flow_score_threshold", score),
                ("risk_per_trade_pct", risk),
            )
        ),
        split_name=split_name,
        split_phase=SplitPhase.TRAIN,
        config_hash="h",
        n_trades=n,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


class CountingRunner:
    """A `Runner` that records every call. Lets a test prove nothing RAN."""

    def __init__(self, result_for=None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._result_for = result_for

    def __call__(self, config, split) -> RunResult:
        self.calls.append((split.name, {"config_hash": config.config_hash}))
        trades = () if self._result_for is None else self._result_for(config, split)
        return run_result(tuple(trades), config_hash=config.config_hash)

    @property
    def n_calls(self) -> int:
        return len(self.calls)


def threshold_keyed_trades(config, split):
    """Trades only where the score gate has been lowered below the measured
    maximum Flow Score.

    PROVENANCE.md 2.4: over a full year the highest Flow Score is 66.991221
    and the committed SCALP_1R floor is 70.0. So a cell at 65 takes trades and
    a cell at 70 or above takes none -- which is the shape the real grid has,
    reproduced here without running a year of bars. The three trades at 65 are
    the output of lowering a gate until something passed.
    """
    floor = float(read_path(config, "setups.SCALP_1R.min_flow_score"))
    if floor > 66.991221:
        return ()
    return trades_from([1.2, -1.0, 0.9])


# ---------------------------------------------------------------------------
# A stub engine over hand-built bars: the real loop, real execution model
# ---------------------------------------------------------------------------


def bar_at(
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
    """A 1R plan on NQ: size 2, a 10-point stop, so R is $400.

    Constructed, not measured. It claims no performance; it exists because the
    real engine emits no plan at the committed configuration and lowering a
    threshold to reach the entry path is the move the brief forbids.
    """
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


class StubEngine:
    """Emits one hand-built plan on one bar. Entry path only."""

    def __init__(self, config, symbol: str, signal_indices, equity: float) -> None:
        self.config = config
        self.symbol = symbol
        self.spec = config.spec(symbol)
        self.calendar = None
        self.signal_indices = tuple(signal_indices)
        self.equity = equity

    def session_date(self, ts: datetime) -> date:
        return ts.date()

    def decide(self, view, *, equity=None, limits=None) -> EngineDecision:
        ts = view.now
        features = FeatureVector(
            symbol=self.symbol,
            ts=ts,
            values={"atr": 10.0, "atr_percentile": 0.5},
            quality_by_key={
                "atr": DataQuality.GOOD,
                "atr_percentile": DataQuality.GOOD,
            },
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


def store_of(bars, *, quote_half_spread: float | None = None) -> DataStore:
    store = DataStore(latency_seconds=0.0)
    quotes = None
    if quote_half_spread is not None:
        count = len(bars)
        ts = to_ns_array(T0 + timedelta(seconds=INTERVAL * i) for i in range(count))
        quotes = QuoteSeries(
            symbol="NQ",
            ts_ns=ts,
            columns={
                "bid": np.full(count, 13000.0 - quote_half_spread),
                "ask": np.full(count, 13000.0 + quote_half_spread),
                "bid_size": np.full(count, 50.0),
                "ask_size": np.full(count, 50.0),
            },
        )
    store.add(
        SymbolData(
            symbol="NQ",
            primary_interval=INTERVAL,
            bars={INTERVAL: BarSeries.from_bars(bars)},
            quotes=quotes,
        )
    )
    return store


def bars_to_target() -> list[Bar]:
    bars = [bar_at(i) for i in range(SIGNAL_INDEX + 1)]
    bars.append(bar_at(SIGNAL_INDEX + 1, open_=13000.0, high=13002.0, low=12999.0, close=13001.0))
    bars.append(bar_at(SIGNAL_INDEX + 2, open_=13001.0, high=13003.0, low=12999.0, close=13000.0))
    bars.append(bar_at(SIGNAL_INDEX + 3, open_=13001.0, high=13015.0, low=13000.0, close=13012.0))
    bars.append(bar_at(SIGNAL_INDEX + 4, open_=13000.0))
    return bars


def bars_delayed_entry_is_cheaper() -> list[Bar]:
    """The next bar opens ABOVE the plan, the one after opens AT it.

    A LONG that waits a bar therefore pays less for the same exit. Built to
    make the delayed-entry degradation come out AHEAD, so the direction
    reporting can be tested on a case that really happens.
    """
    bars = [bar_at(i) for i in range(SIGNAL_INDEX + 1)]
    bars.append(bar_at(SIGNAL_INDEX + 1, open_=13006.0, high=13007.0, low=13000.0, close=13001.0))
    bars.append(bar_at(SIGNAL_INDEX + 2, open_=13000.0, high=13002.0, low=12999.0, close=13001.0))
    bars.append(bar_at(SIGNAL_INDEX + 3, open_=13001.0, high=13015.0, low=13000.0, close=13012.0))
    bars.append(bar_at(SIGNAL_INDEX + 4, open_=13000.0))
    return bars


def stub_runner(store, *, signal_indices=(SIGNAL_INDEX,)):
    """A `Runner` that builds a FRESH loop per call.

    `BacktestLoop` raises on a second `run()` because `RiskLimitManager`
    accumulates, so a runner that cached its loop would break the moment a
    sweep ran two cells.
    """

    def runner(config, split) -> RunResult:
        engine = StubEngine(
            config, "NQ", signal_indices, float(config.risk.starting_equity)
        )
        loop = BacktestLoop(
            config,
            "NQ",
            engine=engine,
            split_label=split.phase,
            split_name=split.name,
            git_commit="",
            git_dirty=False,
        )
        return loop.run(StoreMarketSource(store, "NQ"))

    return runner


# ===========================================================================
# 1. NO BEST CELL IS EXTRACTABLE
# ===========================================================================

#: Tokens that would name a winner. Matched as whole identifier WORDS, never
#: as substrings: `win_rate` contains "win", `max_abs_elasticity` contains
#: "max", and a substring guard flags both. Prose is never matched at all.
WINNER_TOKENS = frozenset(
    {
        "best",
        "optimal",
        "optimum",
        "optimize",
        "optimise",
        "argmax",
        "argmin",
        "recommend",
        "recommended",
        "recommendation",
        "winner",
        "winning",
        "champion",
        "chosen",
        "choose",
        "select",
        "selected",
        "selection",
        "rank",
        "ranked",
        "ranking",
        "pick",
        "picked",
        "favourite",
        "favorite",
        "fittest",
    }
)

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")


def identifier_tokens(name: str) -> set[str]:
    """snake_case and camelCase split into lowercase words."""
    return {part.lower() for part in _WORD.findall(name)}


def public_api_names(target) -> set[str]:
    """Every name a caller can reach: attributes, fields and annotations.

    Annotations come from `typing.get_type_hints`, not `__annotations__`:
    under `from __future__ import annotations` the latter holds STRINGS.
    """
    owner = target if isinstance(target, type) else type(target)
    names = {name for name in dir(target) if not name.startswith("_")}
    fields = getattr(owner, "model_fields", None)
    if isinstance(fields, dict):
        names |= set(fields)
    try:
        names |= set(typing.get_type_hints(owner))
    except Exception:  # pragma: no cover - a stub with no resolvable hints
        pass
    return names


def winner_named_in(target) -> set[str]:
    return {
        name
        for name in public_api_names(target)
        if identifier_tokens(name) & WINNER_TOKENS
    }


def ranking_refusals(rows) -> set[str]:
    """Which of max/min/sorted refuse these rows BY NAME (RobustnessError)."""
    refused = set()
    for label, call in (
        ("max", lambda: max(rows)),
        ("min", lambda: min(rows)),
        ("sorted", lambda: sorted(rows)),
    ):
        try:
            call()
        except RobustnessError:
            refused.add(label)
        except Exception:
            pass
    return refused


SWEEP_TYPES = (
    CellResult,
    DegradationResult,
    DegradationSpec,
    DirectionCheck,
    DecayStep,
    FragilityReport,
    GridPoint,
    HypothesisReport,
    ParameterAxis,
    RobustnessReport,
)


class _StubWithAWinner(FrozenModel):
    """The thing the checks above must catch. Not used by any production code."""

    best_cell: str = ""
    recommended_risk_pct: float = 0.0

    def argmax_expectancy(self) -> str:  # pragma: no cover - never called
        return self.best_cell


class _StubThatRanks(FrozenModel):
    """A result row that DOES order, so the ordering check is non-vacuous."""

    score: float = 0.0

    def __lt__(self, other: object) -> bool:
        return self.score < getattr(other, "score", 0.0)


class TestNoBestCellIsExtractable:
    def test_the_winner_check_catches_a_type_that_has_one(self):
        """Non-vacuity. Without this the next test proves nothing."""
        found = winner_named_in(_StubWithAWinner)
        assert found == {"best_cell", "recommended_risk_pct", "argmax_expectancy"}

    @pytest.mark.parametrize("target", SWEEP_TYPES, ids=lambda t: t.__name__)
    def test_no_sweep_type_names_a_winner(self, target):
        assert winner_named_in(target) == set()

    def test_the_assembled_report_names_no_winner_either(self, config):
        """The instance, not just the class: a property added at runtime
        would be invisible to a check that only read the class."""
        report = robustness_report(
            config, (cell(n=0),), small_axes(config), degradations=()
        )
        assert winner_named_in(report) == set()
        assert winner_named_in(report.fragility) == set()

    def test_the_module_exports_no_winner(self):
        import flow_model.validation.robustness as module

        for name in module.__all__:
            assert not (identifier_tokens(name) & WINNER_TOKENS), name

    def test_the_ordering_check_is_non_vacuous(self):
        """A type that ranks produces NO refusals, so the next tests bite."""
        rows = [_StubThatRanks(score=1.0), _StubThatRanks(score=2.0)]
        assert ranking_refusals(rows) == set()
        assert max(rows).score == 2.0  # it really does pick a winner

    def test_grid_cells_refuse_to_be_ranked(self):
        rows = [cell(score=65.0, n=3, expectancy=0.5), cell(score=70.0, n=0)]
        assert ranking_refusals(rows) == {"max", "min", "sorted"}

    def test_degradation_rows_refuse_to_be_ranked(self, split):
        rows = [
            DegradationResult(
                spec=DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=sev),
                mode=DegradationMode.RERUN,
                split_name=split.name,
                split_phase=split.phase,
                n_trades=0,
            )
            for sev in (1.0, 2.0)
        ]
        assert ranking_refusals(rows) == {"max", "min", "sorted"}

    @pytest.mark.parametrize("op", ["lt", "le", "gt", "ge"])
    def test_every_comparison_operator_refuses_by_name(self, op):
        import operator

        left, right = cell(score=65.0, n=3, expectancy=9.9), cell(score=70.0, n=0)
        with pytest.raises(RobustnessError, match="not orderable"):
            getattr(operator, op)(left, right)

    def test_the_refusal_says_what_to_do_instead(self):
        with pytest.raises(RobustnessError) as exc:
            max([cell(n=1, expectancy=1.0), cell(score=75.0, n=1, expectancy=2.0)])
        message = str(exc.value)
        assert "curve fit" in message
        assert "Report the whole grid" in message

    def test_equality_still_works_so_cells_can_be_compared_for_identity(self):
        """Refusing ORDER must not refuse identity: `cell.point == baseline`
        is how the fragility baseline is found."""
        assert cell(n=0) == cell(n=0)
        assert cell(n=0) != cell(score=75.0, n=0)

    def test_the_report_exposes_the_whole_grid_and_a_count_of_empty_cells(
        self, config
    ):
        cells = (cell(score=65.0, n=3, expectancy=0.4), cell(score=70.0, n=0))
        report = robustness_report(config, cells, small_axes(config))
        assert report.n_cells == 2
        assert report.cells_with_trades == 1

    def test_the_reading_note_is_on_every_report(self, config):
        report = robustness_report(config, (cell(n=0),), small_axes(config))
        assert GRID_READING_NOTE in report.notes
        assert "gate was lowered until something passed" in GRID_READING_NOTE


# ===========================================================================
# 2. THE n=0 PATH -- the live case
# ===========================================================================


class TestZeroTradeGrid:
    def test_every_cell_runs_and_is_reported_even_though_all_are_empty(
        self, config, registry, log
    ):
        runner = CountingRunner()
        cells = run_grid(
            config,
            registry,
            runner,
            log=log,
            reason="n=0 path",
            axes=small_axes(config),
        )
        # 3 sweepable splits (train_a, train_b, val_a) x 3 x 2
        assert len(cells) == 18
        assert runner.n_calls == 18
        assert all(c.n_trades == 0 for c in cells)
        assert {c.split_name for c in cells} == {"train_a", "train_b", "val_a"}

    def test_an_empty_cell_stays_in_the_grid_carrying_its_trade_count(
        self, config, split, log
    ):
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="n=0 path",
            axes=small_axes(config),
        )
        assert len(cells) == 6
        for c in cells:
            assert c.n_trades == 0
            assert "n=0" in c.describe()
            assert c.metrics["trade_count"].n == 0

    @pytest.mark.parametrize("name", ["win_rate", "expectancy", "mean_r", "profit_factor"])
    def test_a_rate_or_ratio_at_n0_is_undefined_with_a_reason_not_zero(
        self, config, split, log, name
    ):
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="n=0 path",
            axes=small_axes(config),
        )
        value = cells[0].metrics[name]
        assert value.is_defined is False
        assert value.value is None
        assert value.undefined is Undefined.NO_TRADES

    @pytest.mark.parametrize("name", ["trade_count", "total_pnl"])
    def test_a_count_or_sum_at_n0_stays_measured(self, config, split, log, name):
        """Counts and sums are defined over the empty set; rates are not.
        `analytics.metrics` draws the line in the same place."""
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="n=0 path",
            axes=small_axes(config),
        )
        value = cells[0].metrics[name]
        assert value.is_defined is True
        assert value.value == 0.0

    def test_a_metric_the_cell_never_computed_is_undefined_not_absent(self):
        empty = cell(n=0)
        missing = empty.metric("sharpe_ratio")
        assert missing.is_defined is False
        assert missing.undefined is Undefined.NO_TRADES
        assert "was not computed" in missing.note

    def test_a_metric_missing_from_a_non_empty_cell_says_insufficient_sample(self):
        small = cell(n=3, expectancy=0.4)
        missing = small.metric("sharpe_ratio")
        assert missing.undefined is Undefined.INSUFFICIENT_SAMPLE
        assert missing.n == 3

    def test_the_fragility_verdict_at_n0_is_undefined_and_says_it_is_the_live_case(
        self, config, split, log
    ):
        axes = small_axes(config)
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="n=0 path",
            axes=axes,
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.UNDEFINED
        assert report.worst_relative_drop.is_defined is False
        assert report.worst_relative_drop.value is None
        assert report.max_abs_elasticity.is_defined is False
        assert any("live case" in reason for reason in report.reasons)
        assert "0.0 would be a claim the data does not support" in (
            report.worst_relative_drop.note
        )

    def test_undefined_is_not_a_pass(self):
        """The enum says so, so a reader of the verdict cannot miss it."""
        assert FragilityVerdict.UNDEFINED.value == "UNDEFINED"
        assert set(FragilityVerdict) == {
            FragilityVerdict.FRAGILE,
            FragilityVerdict.ROBUST,
            FragilityVerdict.UNDEFINED,
        }

    def test_the_whole_report_assembles_at_n0(self, config, split, log):
        axes = small_axes(config)
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="n=0 path",
            axes=axes,
        )
        report = robustness_report(config, cells, axes, baseline=run_result())
        assert report.baseline_n_trades == 0
        assert report.cells_with_trades == 0
        assert report.fragility.verdict is FragilityVerdict.UNDEFINED
        lines = report.describe()
        assert any("6 grid cell(s), 0 with at least one trade" in line for line in lines)

    def test_degradations_all_run_at_n0_and_report_undefined_rates(
        self, config, split, log
    ):
        baseline, rows = run_degradations(
            config,
            split,
            CountingRunner(),
            "NQ",
            log=log,
            reason="n=0 degradation path",
        )
        assert baseline.trade_count == 0
        assert len(rows) == len(default_degradations(config))
        assert {row.spec.kind for row in rows} == set(DegradationKind)
        for row in rows:
            assert row.n_trades == 0
            assert row.improved_anywhere is False
            for check in row.direction:
                if check.metric in ("trade_count", "total_pnl"):
                    assert check.worse_or_equal is True
                else:
                    assert check.worse_or_equal is None
                    assert "not a measurement" in check.note

    def test_the_haircut_at_n0_keeps_the_count_and_drops_the_rates(self):
        metrics = win_rate_haircut_metrics(
            (), severity=0.10, master_seed=20240101, seed_name="t"
        )
        assert set(metrics) == set(HAIRCUT_METRICS)
        assert metrics["trade_count"].value == 0.0
        for name in ("win_rate", "expectancy", "profit_factor"):
            assert metrics[name].is_defined is False
            assert metrics[name].undefined is Undefined.NO_TRADES

    def test_the_wait_horizon_check_is_vacuous_at_n0_and_the_report_says_so(
        self, config, split, log
    ):
        _, rows = run_degradations(
            config,
            split,
            CountingRunner(),
            "NQ",
            log=log,
            reason="n=0 wait horizon",
            specs=(DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=2.0),),
        )
        report = robustness_report(config, (), small_axes(config), degradations=rows)
        assert report.wait_horizon_findings() == ()
        assert any("At n=0 the check is vacuous" in note for note in report.notes)


class TestTheRealEngineTakesNoTrades:
    """The live case through `StoreRunner`, over synthetic bars.

    This is the one test that runs the REAL signal engine. It exists because
    every other test here injects a runner, and an injected runner cannot
    establish that the configured system is empty.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def synthetic():
        from flow_model.data import SyntheticAdapter, SyntheticConfig, TradingCalendar

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

    @pytest.fixture(scope="class")
    @staticmethod
    def runner(synthetic):
        config, calendar, store = synthetic
        return StoreRunner(
            store, "NQ", calendar=calendar, git_commit="", git_dirty=False
        )

    @pytest.fixture(scope="class")
    @staticmethod
    def result(synthetic, runner):
        config, _, _ = synthetic
        return runner(config, one_fold())

    def test_the_committed_configuration_takes_zero_trades(self, result):
        assert result.trade_count == 0
        assert result.is_empty is True
        assert result.bars_evaluated > 1000

    def test_the_fold_is_inside_the_research_window_and_never_the_seal(self):
        fold = one_fold()
        assert fold.range.end <= SEALED.start
        assert fold.range.overlaps(SEALED) is False

    def test_a_second_call_builds_a_fresh_loop_and_reproduces_the_run(
        self, synthetic, runner, result
    ):
        """`BacktestLoop.run()` raises on a second call, so `StoreRunner` must
        build one loop per call -- and the same config over the same data must
        still give a bit-identical result."""
        config, _, _ = synthetic
        again = runner(config, one_fold())
        assert again.result_fingerprint() == result.result_fingerprint()

    def test_a_split_with_no_loaded_bar_is_refused_rather_than_run_empty(
        self, synthetic, runner
    ):
        config, _, _ = synthetic
        barren = Split(
            name="train_1999",
            phase=SplitPhase.TRAIN,
            range=DateRange(start=date(1999, 1, 1), end=date(1999, 2, 1)),
        )
        with pytest.raises(RobustnessError, match="no loaded bar falls inside"):
            runner(config, barren)

    def test_the_clock_is_clipped_to_the_fold(self, synthetic, runner):
        """A fold that covers half the loaded bars must evaluate half of them;
        otherwise a fold is not a fold."""
        config, _, store = synthetic
        half = Split(
            name="train_half",
            phase=SplitPhase.TRAIN,
            range=DateRange(start=date(2016, 1, 4), end=date(2016, 1, 18)),
        )
        clipped = runner(config, half)
        assert 0 < clipped.bars_evaluated < len(store.timeline("NQ"))
        assert clipped.trade_count == 0


# ===========================================================================
# 3. EVERY GRID CELL IS LOGGED, WITH A REASON
# ===========================================================================


class TestExperimentLogging:
    def test_one_row_per_cell_over_the_committed_axes(self, config, registry, log):
        axes = default_axes(config)
        expected = 3 * len(axes[0].values) * len(axes[1].values)
        assert expected == 3 * 6 * 4 == 72
        cells = run_grid(
            config, registry, CountingRunner(), log=log, reason="row count", axes=axes
        )
        assert len(cells) == expected
        assert log.count() == expected
        assert len({c.experiment_id for c in cells}) == expected

    def test_every_row_carries_a_reason_for_change_naming_its_cell(
        self, config, split, log
    ):
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="why this sweep ran",
            axes=small_axes(config),
        )
        rows = log.query()
        assert len(rows) == len(cells) == 6
        for row in rows:
            assert row.reason.startswith("why this sweep ran")
            assert "grid cell" in row.reason
            assert row.split_id == split.name
            assert row.phase is split.phase
            assert "no best cell" in row.notes

    def test_every_row_records_the_parameters_it_changed(self, config, split, log):
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="params",
            axes=small_axes(config),
        )
        by_id_map = {row.experiment_id: row for row in log.query()}
        for c in cells:
            row = by_id_map[c.experiment_id]
            assert row.params_changed == c.params_changed
            assert set(row.params_changed) == {
                "setups.SCALP_1R.min_flow_score",
                "setups.SETUP_2R.min_flow_score",
                "setups.DIRECTIONAL_3R.min_flow_score",
                "risk.risk_per_trade_pct",
            }

    def test_results_are_attached_after_the_cell_runs(self, config, split, log):
        run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(result_for=threshold_keyed_trades),
            log=log,
            reason="results",
            axes=small_axes(config),
        )
        rows = log.query()
        assert all("n_trades" in row.results for row in rows)
        counts = sorted(row.results["n_trades"] for row in rows)
        assert counts == [0, 0, 0, 0, 3, 3]
        traded = [row for row in rows if row.results["n_trades"] == 3]
        assert traded[0].results["metrics"]["win_rate"]["value"] == pytest.approx(2 / 3)
        assert traded[0].results["metrics"]["win_rate"]["n"] == 3

    def test_an_interrupted_sweep_still_leaves_its_attempt_in_the_table(
        self, config, split, log
    ):
        """Each cell is logged BEFORE it runs, so a crash cannot erase the
        fact that the data was touched."""

        def exploding(cfg, spl):
            raise KeyboardInterrupt("simulated interruption")

        with pytest.raises(KeyboardInterrupt):
            run_grid(
                config,
                SplitRegistry(splits=(split,)),
                exploding,
                log=log,
                reason="interrupted",
                axes=small_axes(config),
            )
        assert log.count() == 1
        assert log.query()[0].results == {}

    def test_a_sweep_without_a_reason_is_refused(self, config, registry, log):
        for blank in ("", "   "):
            with pytest.raises(RobustnessError, match="reason_for_change"):
                run_grid(
                    config,
                    registry,
                    CountingRunner(),
                    log=log,
                    reason=blank,
                    axes=small_axes(config),
                )
        assert log.count() == 0

    def test_a_degradation_sweep_without_a_reason_is_refused(
        self, config, split, log
    ):
        with pytest.raises(RobustnessError, match="reason_for_change"):
            run_degradations(
                config, split, CountingRunner(), "NQ", log=log, reason=" "
            )
        assert log.count() == 0

    def test_an_unknown_metric_name_is_refused_before_anything_runs(
        self, config, registry, log
    ):
        runner = CountingRunner()
        with pytest.raises(RobustnessError, match="unknown metric"):
            run_grid(
                config,
                registry,
                runner,
                log=log,
                reason="typo",
                axes=small_axes(config),
                metrics=("win_rate", "sharpe_of_my_dreams"),
            )
        assert runner.n_calls == 0
        assert log.count() == 0

    def test_degradations_log_a_baseline_and_one_row_per_spec_under_it(
        self, config, split, log
    ):
        specs = (
            DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=2.0),
            DegradationSpec(
                kind=DegradationKind.WIN_RATE_HAIRCUT, severity=0.1, seed_name="t"
            ),
        )
        _, rows = run_degradations(
            config,
            split,
            CountingRunner(),
            "NQ",
            log=log,
            reason="degradation rows",
            specs=specs,
        )
        assert log.count() == 3
        baseline_rows = [r for r in log.query() if "baseline" in r.name]
        assert len(baseline_rows) == 1
        root = baseline_rows[0].experiment_id
        for row in rows:
            record = log.get(row.experiment_id)
            assert record.parent_experiment_id == root
            assert record.reason.startswith("degradation rows")
            assert row.spec.label() in record.reason

    def test_the_grid_cells_can_be_parented_to_a_run(self, config, split, log):
        parent = log.log(
            name="parent", config_hash=config.config_hash, phase=split.phase
        )
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="lineage",
            axes=small_axes(config),
            parent_experiment_id=parent.experiment_id,
        )
        for c in cells:
            assert log.get(c.experiment_id).parent_experiment_id == parent.experiment_id

    def test_phase_touches_is_answerable_from_the_table(self, config, split, log):
        run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="touch count",
            axes=small_axes(config),
        )
        assert log.total_phase_touches(SplitPhase.TRAIN) == 6
        assert log.total_phase_touches(SplitPhase.TEST) == 0
        assert log.total_phase_touches(SplitPhase.SEALED_OOS) == 0

    def test_the_report_lists_every_experiment_id_it_produced(
        self, config, split, log
    ):
        axes = small_axes(config)
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="ids",
            axes=axes,
        )
        _, rows = run_degradations(
            config,
            split,
            CountingRunner(),
            "NQ",
            log=log,
            reason="ids",
            specs=(DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=2.0),),
        )
        report = robustness_report(config, cells, axes, degradations=rows)
        assert len(report.experiment_ids) == len(cells) + len(rows)
        for experiment_id in report.experiment_ids:
            assert log.get(experiment_id).experiment_id == experiment_id

    def test_the_embargo_is_asserted_before_a_single_cell_runs(self, config, log):
        """Section 10: the embargo is not automatic, and Phase 8 is what calls
        it. A touching fold boundary leaks a position held across it."""
        touching = SplitRegistry(
            splits=(
                Split(
                    name="train_x",
                    phase=SplitPhase.TRAIN,
                    range=DateRange(start=date(2015, 1, 1), end=date(2016, 1, 1)),
                ),
                Split(
                    name="val_x",
                    phase=SplitPhase.VALIDATION,
                    range=DateRange(start=date(2016, 1, 2), end=date(2017, 1, 1)),
                ),
            )
        )
        runner = CountingRunner()
        with pytest.raises(ValueError, match="embargo violated"):
            run_grid(
                config,
                touching,
                runner,
                log=log,
                reason="embargo",
                axes=small_axes(config),
            )
        assert runner.n_calls == 0
        assert log.count() == 0

    def test_the_embargo_length_is_the_callers_research_choice(self, config, log):
        touching = SplitRegistry(
            splits=(
                Split(
                    name="train_x",
                    phase=SplitPhase.TRAIN,
                    range=DateRange(start=date(2015, 1, 1), end=date(2016, 1, 1)),
                ),
                Split(
                    name="val_x",
                    phase=SplitPhase.VALIDATION,
                    range=DateRange(start=date(2016, 1, 2), end=date(2017, 1, 1)),
                ),
            )
        )
        cells = run_grid(
            config,
            touching,
            CountingRunner(),
            log=log,
            reason="embargo override",
            axes=small_axes(config),
            embargo_days=0,
        )
        assert len(cells) == 12


# ===========================================================================
# 4. THE SEAL IS UNTOUCHED BY ANY SWEEP
# ===========================================================================


class TestTheSealIsUntouched:
    def test_only_train_and_validation_are_sweepable(self):
        assert SWEEPABLE_PHASES == (SplitPhase.TRAIN, SplitPhase.VALIDATION)
        assert SplitPhase.TEST not in SWEEPABLE_PHASES
        assert SplitPhase.SEALED_OOS not in SWEEPABLE_PHASES

    @pytest.mark.parametrize("name", ["test_a", "sealed_oos"])
    def test_a_named_evaluation_split_is_refused_and_nothing_runs(
        self, config, registry, log, name
    ):
        runner = CountingRunner()
        with pytest.raises(RobustnessError, match="refused on an evaluation phase"):
            run_grid(
                config,
                registry,
                runner,
                log=log,
                reason="sweep an evaluation window",
                split_names=[name],
                axes=small_axes(config),
            )
        assert runner.n_calls == 0
        assert log.count() == 0

    def test_the_refusal_explains_why(self, config, registry, log):
        with pytest.raises(RobustnessError) as exc:
            run_grid(
                config,
                registry,
                CountingRunner(),
                log=log,
                reason="sweep the test set",
                split_names=["test_a"],
                axes=small_axes(config),
            )
        assert "turns that window into a training set" in str(exc.value)

    def test_the_default_split_choice_skips_evaluation_phases(
        self, config, registry, log
    ):
        cells = run_grid(
            config,
            registry,
            CountingRunner(),
            log=log,
            reason="default splits",
            axes=small_axes(config),
        )
        touched = {c.split_name for c in cells}
        assert touched == {"train_a", "train_b", "val_a"}
        assert "test_a" not in touched and "sealed_oos" not in touched
        assert all(c.split_phase in SWEEPABLE_PHASES for c in cells)

    def test_a_registry_with_no_sweepable_split_is_refused(self, config, log):
        evaluation_only = SplitRegistry(
            splits=(
                Split(
                    name="test_only",
                    phase=SplitPhase.TEST,
                    range=DateRange(start=date(2018, 1, 1), end=date(2019, 1, 1)),
                ),
            )
        )
        with pytest.raises(RobustnessError, match="no sweepable split"):
            run_grid(
                config,
                evaluation_only,
                CountingRunner(),
                log=log,
                reason="nothing to sweep",
                axes=small_axes(config),
            )

    def test_a_degradation_run_refuses_an_evaluation_split_too(
        self, config, registry, log
    ):
        runner = CountingRunner()
        with pytest.raises(RobustnessError, match="refused on an evaluation phase"):
            run_degradations(
                config,
                registry.get("sealed_oos"),
                runner,
                "NQ",
                log=log,
                reason="degrade the seal",
            )
        assert runner.n_calls == 0

    def test_a_guard_blocks_a_sweepable_split_that_reaches_into_the_seal(
        self, config, tmp_path, log
    ):
        """A TRAIN label is not a licence. If someone registers a fold that
        runs past 2023-01-01, the guard stops the sweep."""
        audit = SealAudit(tmp_path / "seal_audit.jsonl")
        guard = DataAccessGuard(SEALED, audit, max_opens=1, config_hash="h")
        overlapping = SplitRegistry(
            splits=(
                Split(
                    name="train_too_long",
                    phase=SplitPhase.TRAIN,
                    range=DateRange(start=date(2022, 1, 1), end=date(2023, 6, 1)),
                ),
            )
        )
        runner = CountingRunner()
        with pytest.raises(SealedDataAccessError, match="sealed out-of-sample window"):
            run_grid(
                config,
                overlapping,
                runner,
                log=log,
                reason="a fold that runs into the seal",
                axes=small_axes(config),
                guard=guard,
            )
        assert runner.n_calls == 0
        assert log.count() == 0
        assert audit.open_count() == 0
        assert guard.opens_remaining() == 1

    def test_a_guard_passes_a_fold_clear_of_the_seal_and_spends_no_opening(
        self, config, tmp_path, log
    ):
        audit = SealAudit(tmp_path / "seal_audit.jsonl")
        guard = DataAccessGuard(SEALED, audit, max_opens=1, config_hash="h")
        cells = run_grid(
            config,
            SplitRegistry(splits=(one_fold(),)),
            CountingRunner(),
            log=log,
            reason="a research-window sweep",
            axes=small_axes(config),
            guard=guard,
        )
        assert len(cells) == 6
        assert audit.open_count() == 0
        assert guard.opens_remaining() == 1
        assert audit.path.exists() is False

    def test_no_logged_row_claims_a_seal_opening(self, config, registry, log):
        run_grid(
            config,
            registry,
            CountingRunner(),
            log=log,
            reason="no seal",
            axes=small_axes(config),
        )
        assert all(row.seal_opened is False for row in log.query())

    def test_the_sweep_reads_only_the_research_window(self, config, registry, log):
        """The dates the runner is handed, enumerated. Nothing in
        2023-01-01..2025-01-01 appears."""
        runner = CountingRunner()
        cells = run_grid(
            config,
            registry,
            runner,
            log=log,
            reason="date audit",
            axes=small_axes(config),
        )
        swept = {registry.get(c.split_name).range for c in cells}
        for window in swept:
            assert window.overlaps(SEALED) is False
            assert window.end <= date(2023, 1, 1)


# ===========================================================================
# 5. EVERY DEGRADATION ONLY HURTS
# ===========================================================================


class TestDegradationDirectionIsStructural:
    def test_the_six_kinds_are_section_tens_six(self):
        assert {k.value for k in DegradationKind} == {
            "added_slippage",
            "wider_spread",
            "delayed_entry",
            "random_execution_delay",
            "random_missed_trades",
            "win_rate_haircut",
        }
        assert set(NEUTRAL_SEVERITY) == set(DegradationKind)

    @pytest.mark.parametrize("kind", list(DegradationKind), ids=lambda k: k.value)
    def test_a_severity_below_neutral_cannot_be_constructed(self, kind):
        neutral = NEUTRAL_SEVERITY[kind]
        seed = "t" if kind in (
            DegradationKind.RANDOM_EXECUTION_DELAY,
            DegradationKind.RANDOM_MISSED_TRADES,
            DegradationKind.WIN_RATE_HAIRCUT,
        ) else ""
        with pytest.raises(ValueError, match="below the neutral"):
            DegradationSpec(kind=kind, severity=neutral - 0.5, seed_name=seed)

    @pytest.mark.parametrize("kind", list(DegradationKind), ids=lambda k: k.value)
    def test_the_neutral_severity_reproduces_the_committed_config(self, config, kind):
        seed = "t" if kind in (
            DegradationKind.RANDOM_EXECUTION_DELAY,
            DegradationKind.RANDOM_MISSED_TRADES,
            DegradationKind.WIN_RATE_HAIRCUT,
        ) else ""
        spec = DegradationSpec(
            kind=kind, severity=NEUTRAL_SEVERITY[kind], seed_name=seed
        )
        assert spec.is_neutral is True
        assert spec.overrides(config, "NQ") == {}
        assert apply_overrides(config, spec.overrides(config, "NQ")) is config

    @pytest.mark.parametrize(
        "kind", [DegradationKind.RANDOM_MISSED_TRADES, DegradationKind.WIN_RATE_HAIRCUT]
    )
    def test_a_probability_above_one_is_refused(self, kind):
        with pytest.raises(ValueError, match="probability"):
            DegradationSpec(kind=kind, severity=1.5, seed_name="t")

    def test_a_fractional_bar_delay_is_refused(self):
        with pytest.raises(ValueError, match="whole number of bars"):
            DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=1.5)

    @pytest.mark.parametrize(
        "kind",
        [
            DegradationKind.RANDOM_EXECUTION_DELAY,
            DegradationKind.RANDOM_MISSED_TRADES,
            DegradationKind.WIN_RATE_HAIRCUT,
        ],
    )
    def test_a_stochastic_degradation_must_name_its_seed_stream(self, kind):
        with pytest.raises(ValueError, match="must name its seed"):
            DegradationSpec(kind=kind, severity=0.5, seed_name="  ")

    def test_only_the_haircut_is_post_hoc(self):
        for kind in DegradationKind:
            seed = "t" if kind in (
                DegradationKind.RANDOM_EXECUTION_DELAY,
                DegradationKind.RANDOM_MISSED_TRADES,
                DegradationKind.WIN_RATE_HAIRCUT,
            ) else ""
            spec = DegradationSpec(
                kind=kind, severity=NEUTRAL_SEVERITY[kind] + 1.0, seed_name=seed
            ) if kind is not DegradationKind.WIN_RATE_HAIRCUT else DegradationSpec(
                kind=kind, severity=0.5, seed_name="t"
            )
            expected = (
                DegradationMode.POST_HOC
                if kind is DegradationKind.WIN_RATE_HAIRCUT
                else DegradationMode.RERUN
            )
            assert spec.mode is expected

    def test_the_slippage_override_can_only_raise_the_committed_value(self, config):
        committed = float(config.execution.base_slippage_ticks)
        for severity in (1.0, 1.5, 2.0, 3.0):
            spec = DegradationSpec(
                kind=DegradationKind.ADDED_SLIPPAGE, severity=severity
            )
            overrides = spec.overrides(config, "NQ")
            applied = overrides.get("execution.base_slippage_ticks", committed)
            assert applied >= committed

    def test_the_spread_override_can_only_raise_the_typical_spread(self, config):
        typical = float(read_path(config, "instruments.NQ.typical_spread_ticks"))
        for severity in (1.0, 1.5, 2.0):
            spec = DegradationSpec(kind=DegradationKind.WIDER_SPREAD, severity=severity)
            overrides = spec.overrides(config, "NQ")
            applied = overrides.get("execution.spread_ticks_override", typical)
            assert applied >= typical

    def test_the_delay_override_only_adds_bars_and_widens_the_horizon(self, config):
        committed = int(config.execution.max_entry_wait_bars)
        for bars in (0, 1, 2, 5):
            spec = DegradationSpec(
                kind=DegradationKind.DELAYED_ENTRY, severity=float(bars)
            )
            overrides = spec.overrides(config, "NQ")
            if bars == 0:
                assert overrides == {}
                continue
            assert overrides["execution.entry_delay_bars"] == bars
            assert overrides["execution.max_entry_wait_bars"] >= committed
            assert overrides["execution.max_entry_wait_bars"] > bars

    def test_a_stochastic_override_carries_a_derived_seed_not_the_global_state(
        self, config
    ):
        from flow_model.core.determinism import derive_seed

        spec = DegradationSpec(
            kind=DegradationKind.RANDOM_MISSED_TRADES, severity=0.1, seed_name="audit"
        )
        overrides = spec.overrides(config, "NQ")
        assert overrides["backtest.seed"] == derive_seed(
            int(config.backtest.seed), "robustness", "random_missed_trades", "audit"
        )

    def test_the_default_set_covers_every_kind_at_the_committed_severities(
        self, config
    ):
        specs = default_degradations(config)
        r = config.robustness
        expected = (
            len(r.slippage_multipliers)
            + len(r.spread_multipliers)
            + len(r.entry_delay_bars)
            + len(r.slippage_multipliers)  # random execution delay, derived
            + len(r.miss_probabilities)
            + len(r.win_rate_haircuts)
        )
        assert len(specs) == expected
        assert {s.kind for s in specs} == set(DegradationKind)
        for spec in specs:
            assert spec.severity >= NEUTRAL_SEVERITY[spec.kind]
        neutral = [s for s in specs if s.is_neutral]
        assert len(neutral) == len(DegradationKind)

    def test_the_default_severities_come_from_config_not_from_this_module(self, config):
        """A severity invented in `robustness.py` would be a research choice
        made in the wrong file. Raise the config's own list and the specs
        follow."""
        wider = apply_overrides(config, {"robustness.slippage_multipliers": [1.0, 4.0]})
        severities = [
            s.severity
            for s in default_degradations(wider)
            if s.kind is DegradationKind.ADDED_SLIPPAGE
        ]
        assert severities == [1.0, 4.0]


class TestDegradationDirectionIsMeasured:
    """Measured against a real trade, through the real execution model."""

    @pytest.fixture(scope="class")
    @staticmethod
    def traded():
        config = default_config()
        store = store_of(bars_to_target())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            baseline, rows = run_degradations(
                config,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="measured degradation direction",
            )
        return config, baseline, rows

    def test_the_baseline_took_the_constructed_trade(self, traded):
        _, baseline, _ = traded
        assert baseline.trade_count == 1
        assert baseline.trades[0].r_multiple == pytest.approx(0.9494)

    def test_every_degradation_is_worse_or_equal_on_every_defined_metric(self, traded):
        """The whole point. On this trade set, with no quote stream, all six
        families come out worse or equal to the baseline."""
        _, _, rows = traded
        offenders = [
            (row.spec.label(), check.metric)
            for row in rows
            for check in row.direction
            if check.worse_or_equal is False
        ]
        assert offenders == []
        assert all(row.improved_anywhere is False for row in rows)

    @pytest.mark.parametrize(
        "kind",
        [
            DegradationKind.ADDED_SLIPPAGE,
            DegradationKind.WIDER_SPREAD,
            DegradationKind.DELAYED_ENTRY,
        ],
        ids=lambda k: k.value,
    )
    def test_a_cost_family_decays_as_its_severity_rises(self, traded, kind):
        _, _, rows = traded
        family = [row for row in rows if row.spec.kind is kind]
        family.sort(key=lambda row: row.spec.severity)
        values = [row.metrics["expectancy"].require() for row in family]
        assert values == sorted(values, reverse=True)
        assert values[0] > values[-1]  # the family really bites

    def test_added_slippage_costs_exactly_the_ticks_it_adds(self, traded):
        """A measured step, not just a direction.

        NQ is $5 a tick and the plan is size 2, so one extra tick of base
        slippage costs 2 * $5 = $10 -- on the ENTRY only. This trade exits
        through its target, which is a resting limit and is filled passively,
        so it pays no slippage. One extra tick is therefore $10 = 0.025R on
        the $400 R this plan carries, not $20.
        """
        _, _, rows = traded
        by_severity = {
            row.spec.severity: row.metrics["total_pnl"].require()
            for row in rows
            if row.spec.kind is DegradationKind.ADDED_SLIPPAGE
        }
        assert by_severity[1.0] - by_severity[1.5] == pytest.approx(5.0)
        assert by_severity[1.0] - by_severity[2.0] == pytest.approx(10.0)
        assert by_severity[1.0] - by_severity[3.0] == pytest.approx(20.0)
        expectancy = {
            row.spec.severity: row.metrics["expectancy"].require()
            for row in rows
            if row.spec.kind is DegradationKind.ADDED_SLIPPAGE
        }
        assert expectancy[1.0] - expectancy[2.0] == pytest.approx(10.0 / R_DOLLARS)

    def test_the_neutral_cell_of_each_family_equals_the_baseline(self, traded):
        _, baseline, rows = traded
        reference = compute_trade_metrics(
            baseline.trades, split_label=SplitPhase.TRAIN, names=DEGRADATION_METRICS
        )
        for row in rows:
            if not row.spec.is_neutral:
                continue
            for name in row.metrics:
                mine, theirs = row.metrics[name], reference.get(name)
                if theirs is None:
                    continue
                assert mine.or_none() == theirs.or_none(), (row.spec.label(), name)

    def test_a_certain_miss_removes_the_trade_rather_than_improving_anything(self):
        config = default_config()
        store = store_of(bars_to_target())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            baseline, rows = run_degradations(
                config,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="p=1 miss",
                specs=(
                    DegradationSpec(
                        kind=DegradationKind.RANDOM_MISSED_TRADES,
                        severity=1.0,
                        seed_name="certain",
                    ),
                ),
            )
        assert baseline.trade_count == 1
        assert rows[0].n_trades == 0
        assert rows[0].metrics["trade_count"].require() == 0.0
        assert rows[0].metrics["win_rate"].is_defined is False
        assert rows[0].improved_anywhere is False


class TestADegradationThatCanFlatter:
    """A hunt, not a happy path.

    `spread_ticks_override` outranks a real quote in `backtest/costs.py` by
    design, so a "wider spread" run pins the spread as well as widening it.
    Against a quote WIDER than the pinned value the degraded run is CHEAPER
    than the baseline. Measured below: expectancy 0.9119R -> 0.94315R on the
    same single trade. The module does not hide it -- `DirectionCheck` records
    `worse_or_equal=False` and the report names the run -- so what is asserted
    here is that it is detected, explained with the RIGHT mechanism, and never
    silently reported as a passed stress test.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def wide_quote_run():
        config = default_config()
        # 4 ticks (1.00 point) wide: four times NQ's typical_spread_ticks of 1.
        store = store_of(bars_to_target(), quote_half_spread=0.5)
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            baseline, rows = run_degradations(
                config,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="hunt a flattering degradation",
                specs=(
                    DegradationSpec(kind=DegradationKind.WIDER_SPREAD, severity=1.0),
                    DegradationSpec(kind=DegradationKind.WIDER_SPREAD, severity=1.5),
                    DegradationSpec(kind=DegradationKind.WIDER_SPREAD, severity=2.0),
                ),
            )
        return config, baseline, rows

    def test_the_pinned_spread_really_is_tighter_than_the_quote_it_replaces(
        self, wide_quote_run
    ):
        config, baseline, rows = wide_quote_run
        quoted_ticks = 1.0 / float(config.spec("NQ").tick_size)
        assert quoted_ticks == 4.0
        pinned = [
            float(row.applied["execution.spread_ticks_override"])
            for row in rows
            if row.applied
        ]
        assert pinned == [1.5, 2.0]
        assert all(value < quoted_ticks for value in pinned)

    def test_the_degraded_run_comes_out_ahead_and_that_is_recorded(
        self, wide_quote_run
    ):
        _, _, rows = wide_quote_run
        flattered = [row for row in rows if row.improved_anywhere]
        assert [row.spec.label() for row in flattered] == [
            "wider_spread@1.5",
            "wider_spread@2",
        ]
        ahead = {
            check.metric: (check.baseline.require(), check.degraded.require())
            for check in flattered[0].direction
            if check.worse_or_equal is False
        }
        assert ahead["expectancy"] == (pytest.approx(0.9119), pytest.approx(0.94315))
        assert ahead["total_pnl"] == (pytest.approx(364.76), pytest.approx(377.26))

    def test_the_trade_set_did_not_move_so_a_skipped_loser_cannot_be_the_cause(
        self, wide_quote_run
    ):
        """The reason this matters: the generic explanation for a flattering
        RERUN is "a skipped trade may have been a skipped loser". Here n is
        unchanged, so that explanation is wrong and the note must not assert
        it as the cause."""
        _, baseline, rows = wide_quote_run
        flattered = [row for row in rows if row.improved_anywhere]
        assert all(row.n_trades == baseline.trade_count == 1 for row in flattered)
        note = next(
            check.note
            for check in flattered[0].direction
            if check.worse_or_equal is False
        )
        assert "cannot say WHICH mechanism" in note
        assert "pinned spread can be tighter" in note
        assert "Compare the trade_count check" in note

    def test_the_run_carries_the_mechanism_in_its_own_notes(self, wide_quote_run):
        _, _, rows = wide_quote_run
        flattered = [row for row in rows if row.improved_anywhere]
        assert any(
            "outranks a real quote" in note for note in flattered[0].notes
        )

    def test_the_assembled_report_names_every_run_that_came_out_ahead(
        self, wide_quote_run
    ):
        config, baseline, rows = wide_quote_run
        report = robustness_report(
            config, (), small_axes(config), degradations=rows, baseline=baseline
        )
        note = next(n for n in report.notes if "came out ahead" in n)
        assert "wider_spread@1.5" in note and "wider_spread@2" in note
        assert "Not a result" in note
        assert "pinned spread can be tighter" in note

    def test_the_neutral_run_against_the_same_quote_is_unchanged(self, wide_quote_run):
        """Severity 1.0 applies NO override, so the quote is still used and
        the baseline is reproduced. That is what makes it the reference."""
        _, _, rows = wide_quote_run
        neutral = rows[0]
        assert neutral.spec.is_neutral is True
        assert neutral.applied == {}
        assert neutral.metrics["expectancy"].require() == pytest.approx(0.9119)


class TestADelayedEntryCanAlsoFlatter:
    def test_a_later_bar_that_opens_more_favourably_improves_the_run(self):
        """Honest finding, not a defect: the delayed fill is a different fill.
        What is required is that it is reported."""
        config = default_config()
        store = store_of(bars_delayed_entry_is_cheaper())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            baseline, rows = run_degradations(
                config,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="delayed entry can improve",
                specs=(
                    DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=1.0),
                ),
            )
        assert baseline.trade_count == rows[0].n_trades == 1
        assert rows[0].improved_anywhere is True
        ahead = [c for c in rows[0].direction if c.worse_or_equal is False]
        assert {c.metric for c in ahead} == {"expectancy", "total_pnl"}
        assert ahead[0].degraded.require() > ahead[0].baseline.require()


class TestVacuousDegradationsSaySo:
    """A stress test that cannot stress reads as a pass unless it says so."""

    def test_the_sub_bar_latency_jitter_cannot_move_a_fill_and_the_run_says_it(self):
        config = default_config()
        interval = int(config.data.primary_interval_seconds)
        base_ms = float(config.execution.latency_ms)
        severities = [
            s.severity
            for s in default_degradations(config)
            if s.kind is DegradationKind.RANDOM_EXECUTION_DELAY
        ]
        assert severities == [0.0, 125.0, 250.0, 500.0]
        for severity in severities:
            assert min_fill_offset(
                latency_seconds=(base_ms + severity) / 1000.0,
                interval_seconds=interval,
                entry_delay_bars=0,
            ) == min_fill_offset(
                latency_seconds=base_ms / 1000.0,
                interval_seconds=interval,
                entry_delay_bars=0,
            )

        store = store_of(bars_to_target())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            baseline, rows = run_degradations(
                config,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="sub-bar jitter",
                specs=(
                    DegradationSpec(
                        kind=DegradationKind.RANDOM_EXECUTION_DELAY,
                        severity=500.0,
                        seed_name="jitter",
                    ),
                ),
            )
        assert rows[0].metrics["expectancy"].require() == pytest.approx(
            baseline.trades[0].r_multiple
        )
        assert any("sub-bar" in note for note in rows[0].notes)
        assert any("is arithmetic, not evidence" in note for note in rows[0].notes)

    def test_a_market_order_ignores_max_entry_wait_bars(self, config):
        """Section 9 scopes the wait horizon to RESTING entries. So the
        delayed-entry run does not reach the multi-bar horizon at the
        committed `entry_order_type`, and the run says that rather than
        letting an empty findings list read as a verified fix."""
        assert config.execution.entry_order_type == "market"
        intent_source = plan_at(T0, equity=100_000.0)
        del intent_source  # the plan is not needed; the order type is
        order = PendingOrder(
            intent=_intent_for(T0),
            submitted_ts=T0,
            order_type="market",
            latency_seconds=0.25,
            max_entry_wait_bars=3,
        )
        assert order.wait_bars == 1
        resting = order.model_copy(update={"order_type": "limit"})
        assert resting.wait_bars == 3

    def test_the_delayed_entry_run_declares_that_limitation(self, config):
        store = store_of(bars_to_target())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            _, rows = run_degradations(
                config,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="delayed entry on a market order",
                specs=(
                    DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=2.0),
                ),
            )
        assert any("SINGLE-bar fill window" in note for note in rows[0].notes)
        assert any(
            "not about the fix" in note for note in rows[0].notes
        )

    def test_a_resting_order_type_does_reach_the_multi_bar_horizon(self, config):
        resting = apply_overrides(
            config,
            {
                "execution.entry_order_type": "limit",
                "execution.max_entry_wait_bars": 3,
            },
        )
        store = store_of(bars_to_target())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            _, rows = run_degradations(
                resting,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="delayed entry on a resting order",
                specs=(
                    DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=1.0),
                ),
            )
        assert not any("SINGLE-bar fill window" in note for note in rows[0].notes)


def _intent_for(ts: datetime):
    """A minimal TradeIntent, for the order-type arithmetic above."""
    from flow_model.core.contracts import TradeIntent

    return TradeIntent(
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
        equity_at_signal=100_000.0,
        flow_score=71.0,
        regime=Regime.TRENDING_UP,
    )


# ===========================================================================
# 6. THE WIN-RATE HAIRCUT: ARITHMETIC, AND MONOTONE BY ARITHMETIC
# ===========================================================================


class TestWinRateHaircut:
    SAMPLE = (1.0, 1.5, -1.0, 2.0, -1.0, 0.8, -1.2, 3.0, -0.9, 1.1)

    @pytest.fixture
    def trades(self):
        return trades_from(self.SAMPLE)

    def test_the_baseline_metrics_are_the_measured_ones(self, trades):
        metrics = win_rate_haircut_metrics(
            trades, severity=0.0, master_seed=20240101, seed_name="b"
        )
        assert metrics["trade_count"].require() == 10.0
        assert metrics["win_rate"].require() == pytest.approx(0.6)
        assert metrics["expectancy"].require() == pytest.approx(0.53)
        # gross profit 9.4 / gross loss 4.1
        assert metrics["profit_factor"].require() == pytest.approx(9.4 / 4.1)

    def test_a_haircut_re_values_ceil_of_the_winners_at_the_mean_loss(self, trades):
        """Ceil, not round: a 0.05 haircut on 6 winners must bite on one
        rather than round to zero and report an unchanged result as a
        degradation."""
        metrics = win_rate_haircut_metrics(
            trades, severity=0.05, master_seed=20240101, seed_name="b"
        )
        assert metrics["win_rate"].require() == pytest.approx(0.5)  # 6 wins -> 5
        assert metrics["trade_count"].require() == 10.0

    @pytest.mark.parametrize("severity", [0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("seed_name", ["a", "b", "c"])
    def test_every_haircut_is_worse_or_equal_to_the_baseline(
        self, trades, severity, seed_name
    ):
        """The guaranteed claim. NOT monotonicity ACROSS severities: the
        subset is drawn fresh per severity, so cutting two small winners can
        cost less than cutting one large one. Each severity is read against
        the baseline, which is what the module compares."""
        base = win_rate_haircut_metrics(
            trades, severity=0.0, master_seed=20240101, seed_name=seed_name
        )
        cut = win_rate_haircut_metrics(
            trades, severity=severity, master_seed=20240101, seed_name=seed_name
        )
        for name in HAIRCUT_METRICS:
            before, after = base[name].or_none(), cut[name].or_none()
            if before is None or after is None:
                continue
            assert after <= before + 1e-12, name

    def test_a_full_haircut_leaves_no_winner(self, trades):
        metrics = win_rate_haircut_metrics(
            trades, severity=1.0, master_seed=20240101, seed_name="b"
        )
        assert metrics["win_rate"].require() == 0.0
        assert metrics["profit_factor"].require() == 0.0
        assert metrics["expectancy"].require() < 0.0

    def test_no_record_is_altered(self, trades):
        before = [t.model_dump(mode="json") for t in trades]
        win_rate_haircut_metrics(
            trades, severity=0.5, master_seed=20240101, seed_name="b"
        )
        assert [t.model_dump(mode="json") for t in trades] == before

    def test_the_note_says_this_is_not_a_run(self, trades):
        metrics = win_rate_haircut_metrics(
            trades, severity=0.5, master_seed=20240101, seed_name="b"
        )
        for value in metrics.values():
            assert "No TradeRecord is altered and no fill is invented" in value.note
            assert "not a run and not a historical result" in value.note

    def test_a_sample_with_no_loss_has_no_magnitude_to_re_value_at(self):
        trades = trades_from([1.0, 2.0, 0.5])
        metrics = win_rate_haircut_metrics(
            trades, severity=0.5, master_seed=20240101, seed_name="b"
        )
        assert metrics["trade_count"].require() == 3.0
        for name in ("win_rate", "expectancy", "profit_factor"):
            assert metrics[name].undefined is Undefined.NO_LOSSES
        assert "no measured magnitude" in metrics["win_rate"].note

    def test_a_sample_whose_losses_average_exactly_zero_is_undefined_not_zero(self):
        """A scratch is neither a win nor a loss, so `is_loss` is false for
        it; this needs a loss whose R rounds to zero, which the record's own
        `r_multiple` rounding at 6 places produces."""
        tiny = r_trade(0, -1e-9)
        assert tiny.is_loss is True
        assert tiny.r_multiple == 0.0
        metrics = win_rate_haircut_metrics(
            (r_trade(1, 1.0), tiny), severity=0.5, master_seed=20240101, seed_name="b"
        )
        assert metrics["expectancy"].undefined is Undefined.ZERO_AVERAGE_LOSS
        assert metrics["trade_count"].require() == 2.0

    def test_a_sample_with_no_winner_is_untouched_by_a_haircut(self):
        trades = trades_from([-1.0, -0.5])
        base = win_rate_haircut_metrics(
            trades, severity=0.0, master_seed=20240101, seed_name="b"
        )
        cut = win_rate_haircut_metrics(
            trades, severity=1.0, master_seed=20240101, seed_name="b"
        )
        assert cut["expectancy"].require() == base["expectancy"].require()

    def test_profit_factor_with_no_losses_is_undefined_not_infinite(self):
        metrics = win_rate_haircut_metrics(
            trades_from([1.0, 2.0]), severity=0.0, master_seed=20240101, seed_name="b"
        )
        assert metrics["profit_factor"].undefined is Undefined.NO_LOSSES
        assert "undefined, not infinite" in metrics["profit_factor"].note

    def test_r_is_read_off_the_record_rather_than_re_derived(self):
        """ARCHITECTURE.md 0.2 / PROVENANCE.md 4: `risk_dollars` is the
        PLANNED unit and `r_multiple` is net PnL over it. A haircut that
        re-derived R from the fill would read a different number here."""
        costly = r_trade(0, 1.0, costs=CostBreakdown(commission=5.0, spread_cost=15.0))
        assert costly.risk_dollars == pytest.approx(200.0)
        assert costly.r_multiple == pytest.approx(1.0)
        metrics = win_rate_haircut_metrics(
            (costly,), severity=0.0, master_seed=20240101, seed_name="b"
        )
        assert metrics["expectancy"].require() == pytest.approx(costly.r_multiple)

    def test_an_improving_haircut_raises_rather_than_being_reported(
        self, config, split, trades, log
    ):
        """Non-vacuity for the POST_HOC guard: the haircut is monotone by
        arithmetic, so a defined improvement is a defect in the module. Fed a
        baseline that was itself already cut by MORE, the lesser cut comes out
        ahead and the guard fires."""
        baseline = run_result(trades)
        already_cut = win_rate_haircut_metrics(
            trades, severity=0.60, master_seed=int(config.backtest.seed), seed_name="x"
        )
        spec = DegradationSpec(
            kind=DegradationKind.WIN_RATE_HAIRCUT, severity=0.05, seed_name="x"
        )
        with pytest.raises(RobustnessError, match="improved on the baseline"):
            _post_hoc_degradation(
                config,
                split,
                spec,
                baseline,
                already_cut,
                log=log,
                reason="guard check",
                parent_experiment_id="",
            )

    def test_the_guard_passes_a_correctly_ordered_pair(
        self, config, split, trades, log
    ):
        baseline = run_result(trades)
        honest = win_rate_haircut_metrics(
            trades, severity=0.0, master_seed=int(config.backtest.seed), seed_name="x"
        )
        spec = DegradationSpec(
            kind=DegradationKind.WIN_RATE_HAIRCUT, severity=0.5, seed_name="x"
        )
        row = _post_hoc_degradation(
            config,
            split,
            spec,
            baseline,
            honest,
            log=log,
            reason="guard check",
            parent_experiment_id="",
        )
        assert row.mode is DegradationMode.POST_HOC
        assert row.n_trades == len(trades)
        assert row.improved_anywhere is False


# ===========================================================================
# 7. SEEDED DETERMINISM, AND THE GLOBAL NUMPY STATE IS NOT CONSULTED
# ===========================================================================


class TestDeterminism:
    SAMPLE = (1.0, 1.5, -1.0, 2.0, -1.0, 0.8, -1.2, 3.0, -0.9, 1.1)

    def test_reseeding_the_global_generator_does_not_change_the_haircut(self):
        trades = trades_from(self.SAMPLE)
        np.random.seed(1)
        first = win_rate_haircut_metrics(
            trades, severity=0.4, master_seed=20240101, seed_name="s"
        )
        np.random.seed(987654321)
        second = win_rate_haircut_metrics(
            trades, severity=0.4, master_seed=20240101, seed_name="s"
        )
        assert {k: v.model_dump() for k, v in first.items()} == {
            k: v.model_dump() for k, v in second.items()
        }

    def test_the_haircut_consumes_nothing_from_the_global_generator(self):
        """Stronger than reproducibility: the global Mersenne-Twister buffer
        and position are untouched, so this module cannot perturb anything
        else that (wrongly) reads global state."""
        trades = trades_from(self.SAMPLE)
        np.random.seed(4242)
        before = np.random.get_state()
        win_rate_haircut_metrics(
            trades, severity=0.4, master_seed=20240101, seed_name="s"
        )
        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2] == after[2]

    def test_a_full_degradation_sweep_consumes_nothing_global_either(self, split):
        config = default_config()
        store = store_of(bars_to_target())
        np.random.seed(31337)
        before = np.random.get_state()
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            run_degradations(
                config,
                split,
                stub_runner(store),
                "NQ",
                log=handle,
                reason="global state audit",
                specs=(
                    DegradationSpec(
                        kind=DegradationKind.RANDOM_EXECUTION_DELAY,
                        severity=250.0,
                        seed_name="g",
                    ),
                    DegradationSpec(
                        kind=DegradationKind.RANDOM_MISSED_TRADES,
                        severity=0.5,
                        seed_name="g",
                    ),
                    DegradationSpec(
                        kind=DegradationKind.WIN_RATE_HAIRCUT,
                        severity=0.5,
                        seed_name="g",
                    ),
                ),
            )
        after = np.random.get_state()
        assert np.array_equal(before[1], after[1])
        assert before[2] == after[2]

    def test_the_same_seed_name_reproduces_the_draw_across_global_reseeds(
        self, split
    ):
        """Same config hash and same data => bit-identical rows."""
        config = default_config()
        store = store_of(bars_to_target())

        def once(global_seed: int):
            np.random.seed(global_seed)
            with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
                baseline, rows = run_degradations(
                    config,
                    split,
                    stub_runner(store),
                    "NQ",
                    log=handle,
                    reason="determinism",
                    specs=(
                        DegradationSpec(
                            kind=DegradationKind.RANDOM_MISSED_TRADES,
                            severity=0.5,
                            seed_name="repeat",
                        ),
                        DegradationSpec(
                            kind=DegradationKind.WIN_RATE_HAIRCUT,
                            severity=0.5,
                            seed_name="repeat",
                        ),
                    ),
                )
            return baseline.result_fingerprint(), [
                {k: v.model_dump() for k, v in row.metrics.items()} for row in rows
            ]

        assert once(1) == once(2) == once(999)

    def test_a_different_seed_name_is_a_different_stream(self):
        trades = trades_from(self.SAMPLE)
        draws = {
            name: win_rate_haircut_metrics(
                trades, severity=0.3, master_seed=20240101, seed_name=name
            )["expectancy"].require()
            for name in ("a", "b", "c", "d")
        }
        assert len(set(draws.values())) > 1

    def test_a_different_master_seed_is_a_different_stream(self):
        trades = trades_from(self.SAMPLE)
        one = win_rate_haircut_metrics(
            trades, severity=0.3, master_seed=20240101, seed_name="a"
        )["expectancy"].require()
        two = win_rate_haircut_metrics(
            trades, severity=0.3, master_seed=777, seed_name="a"
        )["expectancy"].require()
        assert one != two

    def test_the_grid_is_reproducible_cell_for_cell(self, config, split):
        axes = small_axes(config)

        def once():
            with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
                cells = run_grid(
                    config,
                    SplitRegistry(splits=(split,)),
                    CountingRunner(result_for=threshold_keyed_trades),
                    log=handle,
                    reason="determinism",
                    axes=axes,
                )
            return [
                (c.point.label(), c.config_hash, c.n_trades, sorted(c.params_changed))
                for c in cells
            ]

        assert once() == once()

    def test_the_config_hash_moves_with_the_cell_and_is_stable(self, config, split):
        axes = small_axes(config)
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            cells = run_grid(
                config,
                SplitRegistry(splits=(split,)),
                CountingRunner(),
                log=handle,
                reason="hashes",
                axes=axes,
            )
        by_point = {c.point.label(): c.config_hash for c in cells}
        assert len(set(by_point.values())) == len(by_point)
        committed = [
            c for c in cells
            if c.point.as_mapping() == {
                "flow_score_threshold": 70.0,
                "risk_per_trade_pct": 0.005,
            }
        ]
        assert len(committed) == 1
        assert committed[0].config_hash == config.config_hash


# ===========================================================================
# 8. SMALL-n CELLS CANNOT READ AS RESULTS
# ===========================================================================


class TestSmallSamplesCarryTheirN:
    @pytest.fixture
    def swept(self, config, split):
        axes = small_axes(config)
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            cells = run_grid(
                config,
                SplitRegistry(splits=(split,)),
                CountingRunner(result_for=threshold_keyed_trades),
                log=handle,
                reason="lowering the gate until something passes",
                axes=axes,
            )
        return config, axes, cells

    def test_lowering_the_gate_is_what_produces_the_trades(self, swept):
        """The grid spans 65..75 and the measured maximum Flow Score over a
        full year is 66.991221 (PROVENANCE.md 2.4), so only the 65 cells
        trade. Those three trades are the output of lowering a gate."""
        _, _, cells = swept
        by_threshold = {}
        for c in cells:
            by_threshold.setdefault(
                c.point.as_mapping()["flow_score_threshold"], set()
            ).add(c.n_trades)
        assert by_threshold == {65.0: {3}, 70.0: {0}, 75.0: {0}}

    def test_every_metric_in_a_three_trade_cell_carries_n_equals_three(self, swept):
        _, _, cells = swept
        small = [c for c in cells if c.n_trades == 3]
        assert len(small) == 2
        for c in small:
            for name in GRID_CELL_METRICS:
                assert c.metrics[name].n == 3, name

    def test_the_cell_description_leads_with_its_trade_count(self, swept):
        _, _, cells = swept
        small = next(c for c in cells if c.n_trades == 3)
        text = small.describe()
        assert "n=3" in text
        assert text.index("n=3") < text.index("win_rate")

    def test_the_fragility_verdict_stays_undefined_because_the_baseline_is_empty(
        self, swept
    ):
        """The crucial one. The committed cell takes no trades, so no verdict
        can be read off the cells that only traded because the gate was
        lowered."""
        config, axes, cells = swept
        report = fragility_report(config, cells, axes)
        assert report.baseline_point.as_mapping()["flow_score_threshold"] == 70.0
        assert report.baseline.is_defined is False
        assert report.verdict is FragilityVerdict.UNDEFINED
        assert report.worst_relative_drop.is_defined is False

    def test_the_grid_still_shows_the_low_cells_and_their_n(self, swept):
        config, axes, cells = swept
        report = robustness_report(config, cells, axes)
        assert report.n_cells == 6
        assert report.cells_with_trades == 2
        lines = report.describe()
        cell_lines = [line for line in lines if line.startswith("train_b [")]
        assert len(cell_lines) == 6
        assert sum(1 for line in cell_lines if "n=3" in line) == 2
        assert sum(1 for line in cell_lines if "n=0" in line) == 4

    def test_the_report_refuses_to_read_as_a_discovery(self, swept):
        config, axes, cells = swept
        report = robustness_report(config, cells, axes)
        note = GRID_READING_NOTE
        assert note in report.notes
        assert "not evidence the system works at a lower threshold" in note
        assert "three-trade cell cannot be mistaken for evidence" in note
        assert "exposes no way to ask which one won" in note
        assert winner_named_in(report) == set()

    def test_a_worst_drop_measured_against_a_small_cell_reports_that_cells_n(
        self, config
    ):
        """A 3-trade neighbour against a 400-trade baseline is a 3-trade
        measurement. Labelling it n=400 is how a three-trade cell reads as a
        result."""
        axes = small_axes(config)
        cells = (
            cell(score=70.0, risk=0.005, n=400, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, risk=0.005, n=3, expectancy=0.5, profit_factor=1.5),
            cell(score=65.0, risk=0.005, n=0),
            cell(score=70.0, risk=0.0075, n=0),
        )
        report = fragility_report(config, cells, axes)
        assert report.worst_relative_drop.require() == pytest.approx(0.5)
        assert report.worst_relative_drop.n == 3
        assert "n=3" in report.worst_relative_drop.note
        assert "baseline n=400" in report.worst_relative_drop.note

    def test_the_verdict_text_states_both_sample_sizes(self, config):
        axes = small_axes(config)
        cells = (
            cell(score=70.0, risk=0.005, n=400, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, risk=0.005, n=3, expectancy=0.95, profit_factor=1.9),
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.ROBUST
        reason = next(r for r in report.reasons if "robust at these cutoffs" in r)
        assert "n=400" in reason and "n=3" in reason
        assert "is not evidence of an edge" in reason


# ===========================================================================
# 9. THE FRAGILITY FORMULA
# ===========================================================================


class TestFragilityFormula:
    def test_the_metric_is_the_pre_registered_one(self):
        assert FRAGILITY_METRIC == "expectancy"
        assert by_id("H-SR-CBAND-ROBUST").metric == "expectancy_relative_drop"

    def test_the_cutoffs_come_from_config_and_are_reported(self, config):
        report = fragility_report(config, (cell(n=0),), small_axes(config))
        assert report.max_relative_drop_allowed == config.robustness.fragility_max_relative_drop
        assert report.profit_factor_floor == config.robustness.fragility_profit_factor_floor
        assert f"{config.robustness.fragility_max_relative_drop:g}" in report.describe()[0]

    def test_the_relative_drop_is_measured_from_the_baseline(self, config):
        """(P(B) - P(N)) / |P(B)|. Baseline 1.0 to neighbour 0.6 is 0.4."""
        axes = small_axes(config)
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=10, expectancy=0.6, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        assert report.worst_relative_drop.require() == pytest.approx(0.4)

    def test_a_step_running_into_the_baseline_uses_the_baselines_denominator(
        self, config
    ):
        """A drop FROM the baseline, always. Reusing the step's own ratio
        would divide by the neighbour's value on a step running INTO the
        baseline -- a different quantity wearing the same name."""
        axes = small_axes(config)
        cells = (
            cell(score=65.0, n=10, expectancy=0.5, profit_factor=2.0),
            cell(score=70.0, n=10, expectancy=2.0, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        # (2.0 - 0.5) / |2.0| = 0.75, not (2.0 - 0.5) / |0.5| = 3.0
        assert report.worst_relative_drop.require() == pytest.approx(0.75)

    def test_fragile_when_a_one_step_change_exceeds_the_configured_drop(self, config):
        axes = small_axes(config)
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=10, expectancy=0.2, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        assert report.worst_relative_drop.require() == pytest.approx(0.8)
        assert report.verdict is FragilityVerdict.FRAGILE
        assert any("fragile: a one-step parameter change" in r for r in report.reasons)

    def test_robust_when_every_one_step_change_stays_inside_it(self, config):
        axes = small_axes(config)
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=10, expectancy=0.9, profit_factor=2.0),
            cell(score=65.0, n=10, expectancy=0.95, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.ROBUST
        assert report.worst_relative_drop.require() == pytest.approx(0.1)

    def test_a_neighbour_that_breaks_the_profit_factor_floor_is_fragile(self, config):
        """Even when the expectancy drop is small."""
        axes = small_axes(config)
        floor = float(config.robustness.fragility_profit_factor_floor)
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=10, expectancy=0.99, profit_factor=floor - 0.01),
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.FRAGILE
        reason = next(r for r in report.reasons if "profit factor" in r)
        assert f"{floor:g}" in reason
        assert "n=10" in reason

    def test_the_profit_factor_rule_needs_the_baseline_to_clear_the_floor(
        self, config
    ):
        """A baseline already under the floor cannot be made fragile by a
        neighbour also under it: there is nothing to fall from."""
        axes = small_axes(config)
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=0.5),
            cell(score=75.0, n=10, expectancy=0.99, profit_factor=0.4),
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.ROBUST
        assert not any("profit factor" in r for r in report.reasons)

    def test_a_zero_baseline_has_no_relative_drop_and_says_why(self, config):
        axes = small_axes(config)
        cells = (
            cell(score=70.0, n=10, expectancy=0.0, profit_factor=1.0),
            cell(score=75.0, n=10, expectancy=-0.5, profit_factor=0.8),
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.UNDEFINED
        assert report.worst_relative_drop.undefined is Undefined.INSUFFICIENT_OBSERVATIONS
        step = next(
            s for s in report.decay_profile if s.is_baseline_neighbour
        )
        assert step.relative_drop.undefined is Undefined.ZERO_DISPERSION
        assert "no denominator" in step.relative_drop.note
        assert "absolute difference is 0.5" in step.relative_drop.note

    def test_no_cell_at_the_committed_coordinate_is_undefined(self, config):
        axes = small_axes(config)
        cells = (
            cell(score=65.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=10, expectancy=0.2, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        assert report.baseline_point is None
        assert report.baseline.undefined is Undefined.INSUFFICIENT_OBSERVATIONS
        assert "no cell at the committed coordinate" in report.baseline.note
        assert report.verdict is FragilityVerdict.UNDEFINED

    def test_a_baseline_with_no_defined_neighbour_is_undefined(self, config):
        axes = small_axes(config)
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=0),
            cell(score=65.0, n=0),
            cell(score=70.0, risk=0.0075, n=0),
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.UNDEFINED
        assert report.worst_relative_drop.undefined is Undefined.INSUFFICIENT_OBSERVATIONS
        assert any("no neighbouring cell" in r for r in report.reasons)

    def test_the_elasticity_divides_the_drop_by_the_relative_parameter_step(
        self, config
    ):
        """"Per unit of parameter change", literally: a 0.4 drop across a
        70->75 move is 0.4 / (5/70) = 5.6."""
        axes = small_axes(config)
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=10, expectancy=0.6, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        step = next(
            s
            for s in report.decay_profile
            if s.axis == "flow_score_threshold" and s.from_value == 70.0
        )
        assert step.relative_drop.require() == pytest.approx(0.4)
        assert step.elasticity.require() == pytest.approx(0.4 / (5.0 / 70.0))
        assert report.max_abs_elasticity.require() == pytest.approx(5.6)

    def test_the_decay_profile_carries_every_adjacent_pair_on_every_axis(
        self, config, split
    ):
        axes = small_axes(config)
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            cells = run_grid(
                config,
                SplitRegistry(splits=(split,)),
                CountingRunner(),
                log=handle,
                reason="profile",
                axes=axes,
            )
        report = fragility_report(config, cells, axes)
        # score axis: 2 risk lines x 2 pairs = 4; risk axis: 3 score lines x 1 = 3
        assert len(report.decay_profile) == 7
        assert {s.axis for s in report.decay_profile} == {
            "flow_score_threshold",
            "risk_per_trade_pct",
        }
        for step in report.decay_profile:
            assert step.from_n == 0 and step.to_n == 0

    def test_each_decay_step_carries_both_cells_trade_counts(self, config):
        axes = small_axes(config)
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=3, expectancy=0.6, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        step = report.decay_profile[0]
        assert (step.from_n, step.to_n) == (10, 3)
        assert step.relative_drop.n == 3  # min of the two

    def test_the_verdict_covers_one_fold_and_names_the_folds_it_excludes(
        self, config
    ):
        """A grid over several folds holds a cell at the committed coordinate
        in each. Matching on the point alone made every fold's cells
        neighbours of one fold's baseline, so a defined cell in fold B could
        supply a verdict for a fold whose own neighbours were all empty."""
        axes = small_axes(config)
        cells = (
            cell(split_name="train_a", score=70.0, n=40, expectancy=1.0, profit_factor=2.0),
            cell(split_name="train_a", score=75.0, n=0),
            cell(split_name="train_a", score=65.0, n=0),
            cell(split_name="train_a", score=70.0, risk=0.0075, n=0),
            cell(split_name="train_b", score=70.0, n=3, expectancy=1.0, profit_factor=2.0),
            cell(split_name="train_b", score=75.0, n=3, expectancy=0.99, profit_factor=1.9),
            cell(split_name="train_b", score=65.0, n=3, expectancy=0.99, profit_factor=1.9),
        )
        report = fragility_report(config, cells, axes)
        assert report.baseline_split == "train_a"
        assert report.verdict is FragilityVerdict.UNDEFINED
        assert any("covers train_a only" in r for r in report.reasons)
        assert any("train_b" in r for r in report.reasons)

    def test_a_single_fold_grid_gets_a_verdict_with_no_exclusion_note(self, config):
        axes = small_axes(config)
        cells = (
            cell(split_name="train_b", score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(split_name="train_b", score=75.0, n=10, expectancy=0.9, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        assert report.baseline_split == "train_b"
        assert report.verdict is FragilityVerdict.ROBUST
        assert not any("covers" in r for r in report.reasons)

    def test_lines_never_span_folds(self, config):
        """Adjacent members of a line must differ in exactly one parameter,
        which means the same fold."""
        axes = small_axes(config)
        cells = (
            cell(split_name="train_a", score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(split_name="train_a", score=75.0, n=10, expectancy=0.9, profit_factor=2.0),
            cell(split_name="train_b", score=70.0, n=10, expectancy=5.0, profit_factor=2.0),
            cell(split_name="train_b", score=75.0, n=10, expectancy=4.0, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        assert len(report.decay_profile) == 2
        drops = sorted(s.relative_drop.require() for s in report.decay_profile)
        assert drops == [pytest.approx(0.1), pytest.approx(0.2)]

    def test_the_fragility_metric_can_be_named_explicitly(self, config):
        axes = small_axes(config)
        cells = (
            CellResult(
                point=GridPoint(
                    coordinates=(
                        ("flow_score_threshold", 70.0),
                        ("risk_per_trade_pct", 0.005),
                    )
                ),
                split_name="train_b",
                split_phase=SplitPhase.TRAIN,
                config_hash="h",
                n_trades=10,
                metrics={"win_rate": MetricValue.measured("win_rate", 0.6, n=10)},
            ),
            CellResult(
                point=GridPoint(
                    coordinates=(
                        ("flow_score_threshold", 75.0),
                        ("risk_per_trade_pct", 0.005),
                    )
                ),
                split_name="train_b",
                split_phase=SplitPhase.TRAIN,
                config_hash="h",
                n_trades=10,
                metrics={"win_rate": MetricValue.measured("win_rate", 0.3, n=10)},
            ),
        )
        report = fragility_report(config, cells, axes, metric="win_rate")
        assert report.metric == "win_rate"
        assert report.worst_relative_drop.require() == pytest.approx(0.5)


# ===========================================================================
# 10. AXES AND CONFIG OVERRIDES
# ===========================================================================


class TestAxes:
    def test_the_committed_axes_are_section_tens_two(self, config):
        axes = default_axes(config)
        assert [a.name for a in axes] == ["flow_score_threshold", "risk_per_trade_pct"]
        assert axes[0].values == config.robustness.flow_score_thresholds
        assert axes[1].values == config.robustness.risk_pcts
        assert axes[0].values[0] == 60.0 and axes[0].values[-1] == 85.0
        assert axes[1].values[0] == 0.0025 and axes[1].values[-1] == 0.01

    def test_the_score_axis_governs_every_setup(self, config):
        axis = score_threshold_axis(config)
        assert set(axis.paths) == {
            f"setups.{s.value}.min_flow_score" for s in config.setups
        }
        assert axis.mode is AxisMode.PRESERVE_OFFSETS

    def test_preserve_offsets_shifts_the_ladder_without_collapsing_it(self, config):
        """The committed ladder is 70/75/80. Setting all three to one value
        would collapse three setups into one and vary two things at once."""
        axis = score_threshold_axis(config)
        committed = {path: read_path(config, path) for path in axis.paths}
        assert sorted(committed.values()) == [70.0, 75.0, 80.0]
        shifted = axis.overrides(config, 60.0)
        assert sorted(shifted.values()) == [60.0, 65.0, 70.0]
        assert axis.committed_value(config) == 70.0

    def test_preserve_offsets_at_the_committed_value_is_the_identity(self, config):
        axis = score_threshold_axis(config)
        shifted = axis.overrides(config, axis.committed_value(config))
        for path, value in shifted.items():
            assert value == pytest.approx(float(read_path(config, path)))

    def test_absolute_sets_every_path_to_the_axis_value(self, config):
        axis = risk_pct_axis(config)
        assert axis.mode is AxisMode.ABSOLUTE
        assert axis.overrides(config, 0.0075) == {"risk.risk_per_trade_pct": 0.0075}
        assert axis.committed_value(config) == pytest.approx(
            config.risk.risk_per_trade_pct
        )

    def test_an_axis_must_govern_a_path(self):
        with pytest.raises(ValueError, match="governs no config path"):
            ParameterAxis(name="nowhere", paths=(), values=(1.0, 2.0))

    def test_an_axis_may_not_repeat_a_path(self):
        with pytest.raises(ValueError, match="repeats a config path"):
            ParameterAxis(name="dup", paths=("a.b", "a.b"), values=(1.0, 2.0))

    def test_an_axis_needs_two_points_to_be_a_measurement(self):
        with pytest.raises(ValueError, match="needs two points"):
            ParameterAxis(name="single", paths=("a.b",), values=(1.0,))

    @pytest.mark.parametrize("values", [(2.0, 1.0), (1.0, 1.0), (1.0, 3.0, 2.0)])
    def test_axis_values_must_ascend_without_repeats(self, values):
        with pytest.raises(ValueError, match="ascend without repeats"):
            ParameterAxis(name="messy", paths=("a.b",), values=values)

    def test_a_grid_needs_at_least_one_axis(self, config, registry, log):
        with pytest.raises(RobustnessError, match="needs at least one axis"):
            run_grid(
                config, registry, CountingRunner(), log=log, reason="no axes", axes=()
            )

    def test_a_grid_point_may_not_repeat_an_axis(self):
        with pytest.raises(ValueError, match="repeats an axis"):
            GridPoint(coordinates=(("a", 1.0), ("a", 2.0)))

    def test_a_grid_point_labels_itself_readably(self):
        point = GridPoint(
            coordinates=(("flow_score_threshold", 70.0), ("risk_per_trade_pct", 0.005))
        )
        assert point.label() == "flow_score_threshold=70 risk_per_trade_pct=0.005"
        assert point.as_mapping() == {
            "flow_score_threshold": 70.0,
            "risk_per_trade_pct": 0.005,
        }


class TestHypothesisDriverAxes:
    def test_a_known_driver_maps_to_its_config_path(self):
        axis = axis_for_driver("c_band", (1.0, 1.5))
        assert axis.paths == (HYPOTHESIS_DRIVER_PATHS["c_band"],)
        assert axis.values == (1.0, 1.5)
        assert "c_band" in axis.note

    def test_an_unknown_driver_is_refused_rather_than_guessed(self):
        with pytest.raises(RobustnessError, match="no config path known"):
            axis_for_driver("vibes", (1.0, 2.0))

    def test_the_driver_values_stay_the_callers_choice(self, config):
        """`RobustnessConfig` commits no grid for c_band, and a step size
        invented in robustness.py would be a research choice in the wrong
        file."""
        assert not hasattr(config.robustness, "zone_band_atrs")
        axis = axis_for_driver("c_band", (0.2, 0.3, 0.4))
        assert axis.values == (0.2, 0.3, 0.4)


class TestConfigPathsAndOverrides:
    def test_read_path_walks_attributes_and_mappings(self, config):
        assert read_path(config, "risk.risk_per_trade_pct") == pytest.approx(
            config.risk.risk_per_trade_pct
        )
        assert read_path(config, "setups.SCALP_1R.min_flow_score") == 70.0
        assert read_path(config, "instruments.NQ.typical_spread_ticks") == 1.0

    def test_read_path_names_the_missing_key_and_what_was_available(self, config):
        with pytest.raises(RobustnessError, match="no key 'NOPE'"):
            read_path(config, "setups.NOPE.min_flow_score")
        with pytest.raises(RobustnessError, match="has no field 'nope'"):
            read_path(config, "risk.nope")

    def test_an_override_produces_a_re_validated_copy_with_a_new_hash(self, config):
        derived = apply_overrides(config, {"risk.risk_per_trade_pct": 0.0075})
        assert derived is not config
        assert derived.risk.risk_per_trade_pct == pytest.approx(0.0075)
        assert derived.config_hash != config.config_hash
        assert config.risk.risk_per_trade_pct == pytest.approx(0.005)

    def test_an_out_of_range_override_is_refused_before_a_run_sees_it(self, config):
        with pytest.raises(RobustnessError, match="produced an invalid config"):
            apply_overrides(config, {"risk.risk_per_trade_pct": 5.0})

    def test_an_empty_override_path_is_refused(self, config):
        with pytest.raises(RobustnessError, match="empty override path"):
            apply_overrides(config, {"": 1.0})

    def test_no_override_returns_the_same_object(self, config):
        assert apply_overrides(config, {}) is config

    def test_the_sweep_never_writes_the_committed_defaults(self, config, split):
        """`config/defaults.yaml` is a committed artifact. A sweep that edited
        it would change the strategy rather than test it."""
        from flow_model.config.loader import DEFAULTS_PATH

        before = DEFAULTS_PATH.read_bytes()
        axes = default_axes(config)
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            run_grid(
                config,
                SplitRegistry(splits=(split,)),
                CountingRunner(),
                log=handle,
                reason="defaults untouched",
                axes=axes,
            )
        assert DEFAULTS_PATH.read_bytes() == before
        assert default_config().config_hash == config.config_hash

    def test_the_committed_config_object_is_not_mutated_by_a_sweep(
        self, config, split
    ):
        snapshot = config.to_init_dict(mode="json")
        axes = default_axes(config)
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            run_grid(
                config,
                SplitRegistry(splits=(split,)),
                CountingRunner(),
                log=handle,
                reason="no mutation",
                axes=axes,
            )
        assert config.to_init_dict(mode="json") == snapshot


# ===========================================================================
# 11. THE WAIT-HORIZON INVARIANT (f44ddb9)
# ===========================================================================


class TestWaitHorizonViolationDetection:
    def test_a_clean_trade_produces_no_finding(self):
        assert wait_horizon_violations(run_result(trades_from([1.0, -1.0]))) == ()

    def test_a_fill_on_its_own_signal_bar_is_caught(self):
        """A fill that resolves on the bar that produced the signal has used
        that bar's future."""
        trade = r_trade(0, 1.0)
        same_bar = trade.model_copy(update={"entry_ts": trade.signal_ts})
        findings = wait_horizon_violations(run_result((same_bar,)))
        assert len(findings) == 1
        assert "does not postdate its signal" in findings[0]
        assert "the fill used its own bar" in findings[0]

    def test_an_exit_before_its_entry_is_caught(self):
        """Defence in depth. `TradeRecord._check` already refuses this and so
        does `RunResult`, so both validators are bypassed to get the record in
        front of the check. It must still fire."""
        trade = r_trade(0, 1.0)
        broken = trade.model_copy(
            update={"exit_ts": trade.entry_ts - timedelta(minutes=1)}
        )
        with pytest.raises(Exception, match="exit_ts precedes entry_ts"):
            run_result((broken,))
        unvalidated = RunResult.model_construct(
            provenance=run_result().provenance, trades=(broken,)
        )
        findings = wait_horizon_violations(unvalidated)
        assert any("exit precedes entry" in f for f in findings)

    def test_an_excursion_beyond_the_stop_with_a_non_stop_exit_is_caught(self):
        """The f44ddb9 regression exactly: a bar the position held was never
        checked for its stop, so a realized loss was reported as a TARGET."""
        trade = r_trade(0, 1.0)
        assert trade.exit_reason is ExitReason.TARGET
        fabricated = trade.model_copy(update={"mae_r": -1.5})
        findings = wait_horizon_violations(run_result((fabricated,)))
        assert len(findings) == 1
        assert "beyond its own stop" in findings[0]
        assert "never checked for its stop" in findings[0]
        assert "mae_r=-1.500000" in findings[0]

    def test_the_stop_bound_is_per_record_not_a_fixed_minus_one_r(self):
        """An entry that slipped realizes MORE than 1R on a stop
        (ARCHITECTURE.md 0.2), so a hard -1.0R bound would raise a false
        positive. Here the realized entry is 2 points worse than planned
        while R stays the PLANNED $200, so the record's own stop sits at
        -1.2R and `r_multiple` reports -1.2 rather than a tidy -1.00."""
        signal_ts = T0
        slipped = TradeRecord(
            trade_id="slipped",
            run_id="test_run",
            symbol="NQ",
            config_hash="testhash",
            signal_ts=signal_ts,
            entry_ts=signal_ts + timedelta(minutes=5),
            exit_ts=signal_ts + timedelta(minutes=30),
            bars_held=5,
            side=Side.LONG,
            setup=SetupType.SCALP_1R,
            planned_entry=13000.0,
            stop_price=12990.0,
            target_price=13010.0,
            entry_price=13002.0,
            exit_price=12990.0,
            size=1.0,
            filled_size=1.0,
            point_value=20.0,
            exit_reason=ExitReason.STOP,
            risk_dollars=200.0,
            gross_pnl=-240.0,
            costs=CostBreakdown(),
            pnl=-240.0,
            mae_r=-1.2,
            mfe_r=0.0,
            flow_score=75.0,
            regime=Regime.LOW_VOL,
            split_label=SplitPhase.TRAIN,
        )
        assert slipped.r_multiple == pytest.approx(-1.2)
        stop_r = -(
            abs(slipped.entry_price - slipped.stop_price)
            * slipped.point_value
            * slipped.filled_size
        ) / slipped.risk_dollars
        assert stop_r == pytest.approx(-1.2)
        assert slipped.exit_reason is ExitReason.STOP
        assert wait_horizon_violations(run_result((slipped,))) == ()

        # An excursion PAST that same per-record stop, with a TARGET exit, is
        # the regression shape and is a finding.
        mislabelled = slipped.model_copy(
            update={"exit_reason": ExitReason.TARGET, "mae_r": -1.3}
        )
        findings = wait_horizon_violations(run_result((mislabelled,)))
        assert len(findings) == 1
        assert "-1.200000R" in findings[0]  # the bound is the record's own

    def test_a_tolerance_keeps_last_bit_noise_from_firing(self):
        trade = r_trade(0, 1.0)
        at_the_stop = trade.model_copy(update={"mae_r": -1.0 - 1e-9})
        assert wait_horizon_violations(run_result((at_the_stop,))) == ()


class TestTheWaitHorizonFixHolds:
    """The first real consumer of the multi-bar horizon path.

    PROVENANCE.md 2.6 records the regression and its true value: the bars
    between an early fill and the end of the wait horizon were uninspected, so
    a trade that had been stopped out was reported as `TARGET, +0.9869R` where
    the truth is `STOP, -1.0756R` -- flattering by 2.06R, with `mae_r` -0.5
    against a true -1.5 and `bars_held` 5 against 7.

    The scenario below reconstructs it: a resting limit entry with a 3-bar
    horizon fills on the FIRST bar, runs through its stop on the SECOND, and
    would have reached its target on the THIRD -- the bar the loop is standing
    on when the order resolves.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def resting_config():
        return apply_overrides(
            default_config(),
            {
                "execution.entry_order_type": "limit",
                "execution.max_entry_wait_bars": 3,
            },
        )

    @pytest.fixture(scope="class")
    @staticmethod
    def bars():
        out = [bar_at(i) for i in range(SIGNAL_INDEX + 1)]
        # offset 1: trades through the 13000 limit, so the entry fills HERE
        out.append(
            bar_at(SIGNAL_INDEX + 1, open_=13001.0, high=13002.0, low=12999.0, close=13001.0)
        )
        # offset 2: runs through the 12990 stop -- the bar that used to be skipped
        out.append(
            bar_at(SIGNAL_INDEX + 2, open_=13000.0, high=13001.0, low=12985.0, close=12995.0)
        )
        # offset 3: reaches the 13010 target, and is where the loop stands
        out.append(
            bar_at(SIGNAL_INDEX + 3, open_=12996.0, high=13015.0, low=12995.0, close=13012.0)
        )
        out.append(bar_at(SIGNAL_INDEX + 4, open_=13000.0))
        return out

    @pytest.fixture(scope="class")
    @staticmethod
    def result(resting_config, bars):
        return stub_runner(store_of(bars))(resting_config, one_fold())

    def test_the_horizon_really_spans_several_bars(self, resting_config, bars):
        order = PendingOrder(
            intent=_intent_for(T0),
            submitted_ts=T0,
            order_type=resting_config.execution.entry_order_type,
            latency_seconds=float(resting_config.execution.latency_ms) / 1000.0,
            max_entry_wait_bars=int(resting_config.execution.max_entry_wait_bars),
        )
        assert order.wait_bars == 3

    def test_the_fill_lands_before_the_end_of_the_horizon(self, result, bars):
        assert result.trade_count == 1
        trade = result.trades[0]
        assert trade.entry_ts == bars[SIGNAL_INDEX + 1].close_ts
        assert trade.entry_ts < bars[SIGNAL_INDEX + 3].close_ts

    def test_the_skipped_bar_is_inspected_so_the_trade_exits_at_its_stop(self, result):
        """The regression's exact numbers, from PROVENANCE.md 2.6."""
        trade = result.trades[0]
        assert trade.exit_reason is ExitReason.STOP
        assert trade.r_multiple == pytest.approx(-1.0756)
        assert trade.mae_r == pytest.approx(-1.5)
        assert trade.r_multiple < 0 < 0.9869  # the fabricated value was +0.9869R

    def test_the_invariant_check_agrees_with_the_record(self, result):
        assert wait_horizon_violations(result) == ()

    def test_a_degradation_run_over_this_path_reports_no_violation_and_says_n(
        self, resting_config, bars
    ):
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            _, rows = run_degradations(
                resting_config,
                one_fold(),
                stub_runner(store_of(bars)),
                "NQ",
                log=handle,
                reason="re-verify the wait-horizon fix under a delayed entry",
                specs=(
                    DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=0.0),
                    DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=1.0),
                ),
            )
        assert [row.n_trades for row in rows] == [1, 1]
        assert all(row.wait_horizon_findings == () for row in rows)
        report = robustness_report(
            resting_config, (), small_axes(resting_config), degradations=rows
        )
        assert report.wait_horizon_findings() == ()
        note = next(n for n in report.notes if "wait-horizon invariant holds" in n)
        assert "1 trade(s) inspected" in note

    def test_a_violation_is_escalated_onto_the_report(self, config, split):
        """Non-vacuity for the reporting path: if a finding ever appears, the
        report leads with it rather than burying it."""
        fabricated = r_trade(0, 1.0).model_copy(update={"mae_r": -1.5})
        row = DegradationResult(
            spec=DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=2.0),
            mode=DegradationMode.RERUN,
            split_name=split.name,
            split_phase=split.phase,
            n_trades=1,
            wait_horizon_findings=wait_horizon_violations(run_result((fabricated,))),
        )
        assert row.wait_horizon_findings
        report = robustness_report(config, (), small_axes(config), degradations=(row,))
        assert any("WAIT-HORIZON INVARIANT VIOLATED in 1 trade(s)" in n for n in report.notes)
        assert report.wait_horizon_findings() == row.wait_horizon_findings


# ===========================================================================
# 12. HYPOTHESES AND FOLD CONCENTRATION
# ===========================================================================


class TestFoldProfitConcentration:
    def test_no_trade_anywhere_is_undefined(self):
        value = fold_profit_concentration(
            {"f1": (), "f2": ()}, split_label=SplitPhase.TRAIN
        )
        assert value.undefined is Undefined.NO_TRADES
        assert "cannot be concentrated anywhere" in value.note

    def test_no_profit_anywhere_is_undefined_rather_than_zero(self):
        value = fold_profit_concentration(
            {"f1": trades_from([-1.0]), "f2": trades_from([-0.5])},
            split_label=SplitPhase.TRAIN,
        )
        assert value.undefined is Undefined.NO_WINS
        assert value.n == 2

    def test_the_largest_folds_share_of_gross_profit(self):
        """Gross, not net: a net denominator can be zero or negative and a
        ratio against it would be presentable nonsense."""
        folds = {
            "f1": trades_from([3.0, -1.0]),
            "f2": trades_from([1.0, -1.0]),
        }
        value = fold_profit_concentration(folds, split_label=SplitPhase.TRAIN)
        # gross profit 3.0 * $200 and 1.0 * $200 -> 600 / 800
        assert value.require() == pytest.approx(0.75)
        assert value.n == 4

    def test_an_even_split_is_a_half(self):
        folds = {"f1": trades_from([1.0]), "f2": trades_from([1.0])}
        value = fold_profit_concentration(folds, split_label=SplitPhase.TRAIN)
        assert value.require() == pytest.approx(0.5)

    def test_a_single_fold_is_one_by_arithmetic_and_says_so(self):
        """`H-CONSISTENCY-OVER-PEAK` is refuted above 0.50, so an unqualified
        1.0 from one fold would refute it every time without having compared
        anything."""
        value = fold_profit_concentration(
            {"only": trades_from([1.0, -1.0])}, split_label=SplitPhase.TRAIN
        )
        assert value.require() == pytest.approx(1.0)
        assert "1.0 by arithmetic" in value.note
        assert "not as a refutation" in value.note

    def test_several_folds_carry_no_such_caveat(self):
        value = fold_profit_concentration(
            {"a": trades_from([1.0]), "b": trades_from([1.0])},
            split_label=SplitPhase.TRAIN,
        )
        assert value.note == ""

    def test_fold_order_does_not_change_the_answer(self):
        folds = {"z": trades_from([3.0]), "a": trades_from([1.0])}
        reversed_folds = {"a": folds["a"], "z": folds["z"]}
        assert fold_profit_concentration(
            folds, split_label=SplitPhase.TRAIN
        ).require() == fold_profit_concentration(
            reversed_folds, split_label=SplitPhase.TRAIN
        ).require()


class TestHypothesisReporting:
    def test_c_band_is_not_evaluable_from_a_grid_that_did_not_sweep_it(self, config):
        report = fragility_report(config, (cell(n=0),), small_axes(config))
        c_band, _ = hypothesis_reports(report, small_axes(config))
        assert c_band.hypothesis_id == "H-SR-CBAND-ROBUST"
        assert c_band.evaluable is False
        assert c_band.refuted is None
        assert "not evaluable" in c_band.describe()
        assert "axis_for_driver('c_band', values)" in c_band.note

    def test_c_band_is_evaluable_when_the_grid_swept_that_driver(self, config):
        axis = axis_for_driver("c_band", (0.2, 0.3))
        committed = float(read_path(config, HYPOTHESIS_DRIVER_PATHS["c_band"]))
        cells = (
            CellResult(
                point=GridPoint(coordinates=(("c_band", committed),)),
                split_name="train_b",
                split_phase=SplitPhase.TRAIN,
                config_hash="h",
                n_trades=20,
                metrics={
                    "expectancy": MetricValue.measured("expectancy_r", 1.0, n=20),
                    "profit_factor": MetricValue.measured("profit_factor", 2.0, n=20),
                },
            ),
            CellResult(
                point=GridPoint(coordinates=(("c_band", committed + 0.1),)),
                split_name="train_b",
                split_phase=SplitPhase.TRAIN,
                config_hash="h",
                n_trades=20,
                metrics={
                    "expectancy": MetricValue.measured("expectancy_r", 0.5, n=20),
                    "profit_factor": MetricValue.measured("profit_factor", 2.0, n=20),
                },
            ),
        )
        axes = (
            ParameterAxis(
                name="c_band",
                paths=axis.paths,
                values=(committed, committed + 0.1),
            ),
        )
        report = fragility_report(config, cells, axes)
        c_band, _ = hypothesis_reports(report, axes)
        assert c_band.evaluable is True
        assert c_band.comparison.comparison is Comparison.AT_MOST
        assert c_band.comparison.target == by_id("H-SR-CBAND-ROBUST").threshold == 0.30
        assert c_band.comparison.measured.require() == pytest.approx(0.5)
        assert c_band.refuted is True
        assert "REFUTED" in c_band.describe()

    def test_a_drop_inside_the_registered_bound_is_not_refuted(self, config):
        committed = float(read_path(config, HYPOTHESIS_DRIVER_PATHS["c_band"]))
        axes = (
            ParameterAxis(
                name="c_band",
                paths=(HYPOTHESIS_DRIVER_PATHS["c_band"],),
                values=(committed, committed + 0.1),
            ),
        )
        cells = (
            CellResult(
                point=GridPoint(coordinates=(("c_band", committed),)),
                split_name="train_b",
                split_phase=SplitPhase.TRAIN,
                config_hash="h",
                n_trades=20,
                metrics={"expectancy": MetricValue.measured("expectancy_r", 1.0, n=20)},
            ),
            CellResult(
                point=GridPoint(coordinates=(("c_band", committed + 0.1),)),
                split_name="train_b",
                split_phase=SplitPhase.TRAIN,
                config_hash="h",
                n_trades=20,
                metrics={"expectancy": MetricValue.measured("expectancy_r", 0.9, n=20)},
            ),
        )
        report = fragility_report(config, cells, axes)
        c_band, _ = hypothesis_reports(report, axes)
        assert c_band.comparison.measured.require() == pytest.approx(0.1)
        assert c_band.refuted is False

    def test_an_undefined_measurement_refutes_nothing_and_supports_nothing(
        self, config
    ):
        committed = float(read_path(config, HYPOTHESIS_DRIVER_PATHS["c_band"]))
        axes = (
            ParameterAxis(
                name="c_band",
                paths=(HYPOTHESIS_DRIVER_PATHS["c_band"],),
                values=(committed, committed + 0.1),
            ),
        )
        cells = (
            CellResult(
                point=GridPoint(coordinates=(("c_band", committed),)),
                split_name="train_b",
                split_phase=SplitPhase.TRAIN,
                config_hash="h",
                n_trades=0,
                metrics={},
            ),
        )
        report = fragility_report(config, cells, axes)
        c_band, _ = hypothesis_reports(report, axes)
        assert c_band.evaluable is True
        assert c_band.comparison.met is None
        assert c_band.refuted is None
        assert "no measurement" in c_band.describe()

    def test_the_consistency_hypothesis_is_reported_when_a_concentration_is_given(
        self, config
    ):
        report = fragility_report(config, (cell(n=0),), small_axes(config))
        _, consistency = hypothesis_reports(
            report,
            small_axes(config),
            concentration=MetricValue.measured(
                "fold_profit_concentration", 0.8, n=40, unit="fraction"
            ),
        )
        assert consistency.hypothesis_id == "H-CONSISTENCY-OVER-PEAK"
        assert consistency.comparison.target == 0.50
        assert consistency.refuted is True
        assert "sign-disagreement half" in consistency.note

    def test_the_consistency_hypothesis_is_not_evaluable_without_one(self, config):
        report = fragility_report(config, (cell(n=0),), small_axes(config))
        _, consistency = hypothesis_reports(report, small_axes(config))
        assert consistency.evaluable is False
        assert consistency.refuted is None

    def test_the_consequence_is_carried_from_the_registry_not_restated(self, config):
        report = fragility_report(config, (cell(n=0),), small_axes(config))
        for item in hypothesis_reports(report, small_axes(config)):
            registered = by_id(item.hypothesis_id)
            assert item.consequence == registered.consequence
            assert item.metric == registered.metric
            assert item.driver == registered.driver

    def test_the_report_carries_both_hypotheses(self, config):
        report = robustness_report(config, (cell(n=0),), small_axes(config))
        assert [h.hypothesis_id for h in report.hypotheses] == [
            "H-SR-CBAND-ROBUST",
            "H-CONSISTENCY-OVER-PEAK",
        ]


# ===========================================================================
# 13. REPORT ASSEMBLY
# ===========================================================================


class TestReportAssembly:
    def test_the_description_lists_every_cell_and_every_degradation(
        self, config, split, log
    ):
        axes = small_axes(config)
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(result_for=threshold_keyed_trades),
            log=log,
            reason="describe",
            axes=axes,
        )
        _, rows = run_degradations(
            config,
            split,
            CountingRunner(),
            "NQ",
            log=log,
            reason="describe",
            specs=(DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=2.0),),
        )
        report = robustness_report(config, cells, axes, degradations=rows)
        lines = report.describe()
        for c in cells:
            assert c.describe() in lines
        for row in rows:
            assert row.describe() in lines
        assert GRID_READING_NOTE in lines

    def test_a_degradation_result_describes_its_severity_and_mode(self, split):
        row = DegradationResult(
            spec=DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=2.5),
            mode=DegradationMode.RERUN,
            split_name=split.name,
            split_phase=split.phase,
            n_trades=0,
        )
        assert row.describe().startswith("added_slippage@2.5 [rerun] n=0")

    def test_the_report_is_frozen(self, config):
        report = robustness_report(config, (cell(n=0),), small_axes(config))
        with pytest.raises(Exception):
            report.notes = ()

    def test_the_grid_cell_metrics_all_exist_in_the_analytics_registry(self):
        from flow_model.analytics.metrics import TRADE_METRICS

        for name in GRID_CELL_METRICS:
            assert name in TRADE_METRICS
        for name in DEGRADATION_METRICS:
            assert name in TRADE_METRICS
        assert GRID_CELL_METRICS[0] == "trade_count"

    def test_trade_count_comes_first_in_every_cell_rendering(self, config, split, log):
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(result_for=threshold_keyed_trades),
            log=log,
            reason="count first",
            axes=small_axes(config),
        )
        for c in cells:
            body = c.describe()
            assert body.index("trade_count") < body.index("win_rate")

    def test_the_data_hash_reaches_the_cell(self, config, split, log):
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="provenance",
            axes=small_axes(config),
        )
        assert all(c.data_hash == "datahash" for c in cells)

    def test_a_cell_cannot_hold_a_negative_trade_count(self):
        with pytest.raises(Exception):
            CellResult(
                point=GridPoint(coordinates=(("a", 1.0),)),
                split_name="s",
                split_phase=SplitPhase.TRAIN,
                config_hash="h",
                n_trades=-1,
            )

    def test_a_direction_check_with_both_sides_defined_reports_the_direction(self):
        check = DirectionCheck.build(
            "expectancy",
            MetricValue.measured("expectancy_r", 1.0, n=10),
            MetricValue.measured("expectancy_r", 0.5, n=10),
        )
        assert check.worse_or_equal is True
        assert check.note == ""

    def test_a_direction_check_tolerates_only_last_bit_noise(self):
        worse = DirectionCheck.build(
            "total_pnl",
            MetricValue.measured("total_pnl", 100.0, n=10),
            MetricValue.measured("total_pnl", 100.0 + 1e-13, n=10),
        )
        assert worse.worse_or_equal is True
        ahead = DirectionCheck.build(
            "total_pnl",
            MetricValue.measured("total_pnl", 100.0, n=10),
            MetricValue.measured("total_pnl", 100.01, n=10),
        )
        assert ahead.worse_or_equal is False

    def test_a_direction_check_with_one_side_undefined_is_not_a_measurement(self):
        check = DirectionCheck.build(
            "profit_factor",
            MetricValue.measured("profit_factor", 2.0, n=10),
            MetricValue.undefined_because("profit_factor", Undefined.NO_LOSSES, n=1),
        )
        assert check.worse_or_equal is None
        assert "not a measurement" in check.note

    def test_a_report_with_no_degradation_has_no_improvement_note(self, config):
        report = robustness_report(config, (cell(n=0),), small_axes(config))
        assert not any("came out ahead" in n for n in report.notes)

    def test_caller_notes_are_kept_alongside_the_reading_note(self, config):
        report = robustness_report(
            config, (cell(n=0),), small_axes(config), notes=("ran on synthetic bars",)
        )
        assert "ran on synthetic bars" in report.notes
        assert report.notes[0] == GRID_READING_NOTE


# ===========================================================================
# 14. MODULE-LEVEL INVARIANTS
# ===========================================================================


class TestModuleInvariants:
    def test_every_exported_name_exists(self):
        import flow_model.validation.robustness as module

        for name in module.__all__:
            assert hasattr(module, name), name

    def test_the_module_reads_no_clock_for_a_result(self, config, split, log):
        """Determinism: `_midnight` is the only datetime construction, and it
        is derived from the split's own dates."""
        from flow_model.validation.robustness import _midnight

        stamp = _midnight(date(2016, 3, 1))
        assert stamp == datetime(2016, 3, 1, tzinfo=timezone.utc)
        assert stamp.tzinfo is timezone.utc

    def test_the_seal_boundary_clock_is_utc(self):
        """`.date()` on an aware datetime gives its date in its OWN zone, so
        one instant written in two zones used to land on opposite sides of the
        seal."""
        from flow_model.validation.robustness import _midnight

        assert _midnight(SEALED.start).isoformat() == "2023-01-01T00:00:00+00:00"

    def test_the_comparison_tolerance_is_float_noise_not_a_parameter(self):
        from flow_model.validation.robustness import COMPARISON_TOLERANCE

        assert COMPARISON_TOLERANCE == 1e-12
        assert COMPARISON_TOLERANCE < 1e-9

    def test_the_error_type_is_specific(self):
        assert issubclass(RobustnessError, RuntimeError)

    def test_the_fragility_formula_in_the_docstring_matches_the_code(self):
        """A worked example against the documented formula, so the docstring
        cannot drift from the arithmetic without a test failing."""
        config = default_config()
        axes = small_axes(config)
        p_baseline, p_neighbour = 1.0, 0.55
        cells = (
            cell(score=70.0, n=10, expectancy=p_baseline, profit_factor=2.0),
            cell(score=75.0, n=10, expectancy=p_neighbour, profit_factor=2.0),
        )
        report = fragility_report(config, cells, axes)
        relative_drop = (p_baseline - p_neighbour) / abs(p_baseline)
        step = abs(75.0 - 70.0) / abs(70.0)
        assert report.worst_relative_drop.require() == pytest.approx(relative_drop)
        assert report.max_abs_elasticity.require() == pytest.approx(
            relative_drop / step
        )
        assert math.isclose(relative_drop, 0.45)


# ===========================================================================
# 15. THE SECOND PASS: six defects, each pinned by the measurement that
#     found it
# ===========================================================================
#
# The module was committed untested at 2c7ca27 and a first test pass fixed
# four defects in it whose per-defect accounting was lost (see the HEAD commit
# message). These are the ones that survived that pass. Every test below
# fails on the code as it stood before it, and each carries the wrong value
# and the true value in its assertions rather than in prose.


def coord_cell(
    coordinates: tuple[tuple[str, float], ...],
    *,
    n: int,
    expectancy: float | None = None,
    profit_factor: float | None = 2.0,
    split_name: str = "train_b",
) -> CellResult:
    """A `CellResult` on arbitrary axes. `cell()` above pins the two
    committed axes; the hypothesis driver needs its own."""
    metrics: dict[str, MetricValue] = {}
    metrics["expectancy"] = (
        MetricValue.undefined_because("expectancy_r", Undefined.NO_TRADES, n=n)
        if expectancy is None
        else MetricValue.measured("expectancy_r", expectancy, n=n, unit="R")
    )
    if profit_factor is not None:
        metrics["profit_factor"] = MetricValue.measured("profit_factor", profit_factor, n=n)
    return CellResult(
        point=GridPoint(coordinates=coordinates),
        split_name=split_name,
        split_phase=SplitPhase.TRAIN,
        config_hash="h",
        n_trades=n,
        metrics=metrics,
    )


def joint_axes(config) -> tuple[ParameterAxis, ...]:
    """A grid that sweeps the c_band driver AND the score threshold.

    This is the shape that exposed the defect: the hypothesis names one
    driver, the grid moves two, and the committed cell has a neighbour on
    each.
    """
    committed = float(read_path(config, HYPOTHESIS_DRIVER_PATHS["c_band"]))
    return (
        axis_for_driver("c_band", (committed, committed + 0.05)),
        ParameterAxis(
            name="flow_score_threshold",
            paths=tuple(f"setups.{s.value}.min_flow_score" for s in sorted(config.setups)),
            values=(70.0, 75.0),
            mode=AxisMode.PRESERVE_OFFSETS,
        ),
    )


def joint_cells(config, *, c_band_expectancy: float, score_expectancy: float):
    """Four cells: the committed one at 1.0R, one neighbour per axis."""
    band = float(read_path(config, HYPOTHESIS_DRIVER_PATHS["c_band"]))
    return (
        coord_cell((("c_band", band), ("flow_score_threshold", 70.0)), n=40, expectancy=1.0),
        coord_cell(
            (("c_band", band + 0.05), ("flow_score_threshold", 70.0)),
            n=40, expectancy=c_band_expectancy,
        ),
        coord_cell(
            (("c_band", band), ("flow_score_threshold", 75.0)),
            n=40, expectancy=score_expectancy,
        ),
        coord_cell(
            (("c_band", band + 0.05), ("flow_score_threshold", 75.0)),
            n=40, expectancy=0.5,
        ),
    )


class TestAPreRegisteredHypothesisIsScoredOnItsOwnDriver:
    """HIGH. `H-SR-CBAND-ROBUST` claims expectancy survives a one-step change
    in `c_band` within 0.30 relative. It was scored with
    `worst_relative_drop`, which is the maximum over EVERY neighbour on EVERY
    axis -- so on a joint grid a flow-score step supplied the number and the
    report published a refutation the driver never produced. Section 15's
    whole point is that the refutation condition was fixed in advance; a
    refutation attributed to the wrong parameter is worse than none.
    """

    def test_a_joint_grid_scores_c_band_on_the_c_band_step_alone(self, config):
        """Measured: REFUTED at 0.9000 (a flow_score_threshold step). True:
        not refuted, the c_band step costs 0.0500 against a 0.30 bound."""
        axes = joint_axes(config)
        cells = joint_cells(config, c_band_expectancy=0.95, score_expectancy=0.10)
        report = fragility_report(config, cells, axes)

        # the grid's worst really is the score step, and really is past 0.30
        assert report.worst_relative_drop.require() == pytest.approx(0.90)
        assert "flow_score_threshold" in report.worst_relative_drop.note
        assert 0.90 > by_id("H-SR-CBAND-ROBUST").threshold

        c_band, _ = hypothesis_reports(report, axes)
        assert c_band.comparison.measured.require() == pytest.approx(0.05)
        assert c_band.refuted is False
        assert "not refuted" in c_band.describe()

    def test_a_real_c_band_refutation_still_lands(self, config):
        """Non-vacuity: the fix must not have made the hypothesis
        unrefutable. Same joint grid, the c_band step now costs 0.60."""
        axes = joint_axes(config)
        cells = joint_cells(config, c_band_expectancy=0.40, score_expectancy=0.95)
        report = fragility_report(config, cells, axes)
        assert report.worst_relative_drop.require() == pytest.approx(0.60)

        c_band, _ = hypothesis_reports(report, axes)
        assert c_band.comparison.measured.require() == pytest.approx(0.60)
        assert c_band.refuted is True
        assert c_band.consequence == by_id("H-SR-CBAND-ROBUST").consequence

    def test_the_note_says_which_axes_were_excluded(self, config):
        axes = joint_axes(config)
        report = fragility_report(
            config, joint_cells(config, c_band_expectancy=0.95, score_expectancy=0.10), axes
        )
        c_band, _ = hypothesis_reports(report, axes)
        assert "c_band neighbours of the committed cell ALONE" in c_band.note
        assert "['flow_score_threshold']" in c_band.note

    def test_a_single_driver_grid_needs_no_exclusion_clause(self, config):
        axes = (joint_axes(config)[0],)
        band = float(read_path(config, HYPOTHESIS_DRIVER_PATHS["c_band"]))
        cells = (
            coord_cell((("c_band", band),), n=40, expectancy=1.0),
            coord_cell((("c_band", band + 0.05),), n=40, expectancy=0.5),
        )
        report = fragility_report(config, cells, axes)
        c_band, _ = hypothesis_reports(report, axes)
        assert c_band.comparison.measured.require() == pytest.approx(0.5)
        assert "the grid also swept" not in c_band.note

    def test_every_supplied_axis_gets_its_own_worst_drop(self, config):
        axes = joint_axes(config)
        report = fragility_report(
            config, joint_cells(config, c_band_expectancy=0.95, score_expectancy=0.10), axes
        )
        per_axis = report.worst_relative_drop_by_axis
        assert set(per_axis) == {"c_band", "flow_score_threshold"}
        assert per_axis["c_band"].require() == pytest.approx(0.05)
        assert per_axis["flow_score_threshold"].require() == pytest.approx(0.90)
        assert per_axis["c_band"].n == 40

    def test_at_n0_every_axis_entry_is_undefined_with_the_baselines_reason(self, config):
        """The live case. An axis with no measurement must not report 0.0,
        and the hypothesis must stay evaluable-with-no-measurement rather
        than silently becoming not-evaluable."""
        axes = joint_axes(config)
        band = float(read_path(config, HYPOTHESIS_DRIVER_PATHS["c_band"]))
        cells = (
            coord_cell(
                (("c_band", band), ("flow_score_threshold", 70.0)), n=0, profit_factor=None
            ),
            coord_cell(
                (("c_band", band + 0.05), ("flow_score_threshold", 70.0)),
                n=0, profit_factor=None,
            ),
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.UNDEFINED
        for name, value in report.worst_relative_drop_by_axis.items():
            assert value.is_defined is False, name
            assert value.or_none() is None, name
            assert value.undefined is Undefined.NO_TRADES, name
        c_band, _ = hypothesis_reports(report, axes)
        assert c_band.evaluable is True
        assert c_band.refuted is None
        assert "no measurement" in c_band.describe()

    def test_an_axis_whose_name_differs_but_whose_path_matches_is_the_driver(
        self, config
    ):
        """The driver is the config path, not the label a caller chose."""
        band = float(read_path(config, HYPOTHESIS_DRIVER_PATHS["c_band"]))
        axes = (
            ParameterAxis(
                name="zone_width",
                paths=(HYPOTHESIS_DRIVER_PATHS["c_band"],),
                values=(band, band + 0.05),
            ),
        )
        cells = (
            coord_cell((("zone_width", band),), n=40, expectancy=1.0),
            coord_cell((("zone_width", band + 0.05),), n=40, expectancy=0.6),
        )
        report = fragility_report(config, cells, axes)
        c_band, _ = hypothesis_reports(report, axes)
        assert c_band.evaluable is True
        assert c_band.comparison.measured.require() == pytest.approx(0.4)


class TestABaselineCellCannotBeNominatedOnAnUnseenAxis:
    """MEDIUM. The committed cell was matched with
    `math.isclose(value, committed.get(name, value))` -- a coordinate for an
    axis the call was not given compared against ITSELF and therefore always
    matched. Hand the report one axis of a two-axis grid and a cell at twice
    the committed risk was published as `baseline_point`, with a verdict about
    "the committed configuration" that no committed configuration produced.
    """

    def test_a_coordinate_with_no_axis_is_refused_rather_than_assumed(self, config):
        """Measured: baseline_point risk_per_trade_pct=0.01 against a
        committed 0.005, verdict FRAGILE. True: refused."""
        assert config.risk.risk_per_trade_pct == 0.005  # the committed value
        cells = (
            cell(score=70.0, risk=0.01, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, risk=0.01, n=10, expectancy=0.2, profit_factor=2.0),
        )
        score_axis_only = (small_axes(config)[0],)
        with pytest.raises(RobustnessError) as excinfo:
            fragility_report(config, cells, score_axis_only)
        message = str(excinfo.value)
        assert "risk_per_trade_pct" in message
        assert "flow_score_threshold" in message
        assert "reported as the committed baseline" in message

    def test_the_same_cells_report_normally_once_both_axes_are_given(self, config):
        """Non-vacuity: with both axes supplied, the 0.01 risk cells are
        simply NOT at the committed coordinate, so there is no baseline and
        the verdict is UNDEFINED rather than a verdict about a config the
        project does not run."""
        cells = (
            cell(score=70.0, risk=0.01, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, risk=0.01, n=10, expectancy=0.2, profit_factor=2.0),
        )
        report = fragility_report(config, cells, small_axes(config))
        assert report.baseline_point is None
        assert report.verdict is FragilityVerdict.UNDEFINED
        assert report.baseline.undefined is Undefined.INSUFFICIENT_OBSERVATIONS
        assert "no cell at the committed coordinate" in report.baseline.note
        assert any("undefined: baseline" in r for r in report.reasons)

    def test_a_grid_built_by_run_grid_always_satisfies_the_check(self, config, split, log):
        """The cells `run_grid` produces carry exactly the axes it was given,
        so the refusal cannot fire on the module's own output."""
        axes = small_axes(config)
        cells = run_grid(
            config,
            SplitRegistry(splits=(split,)),
            CountingRunner(),
            log=log,
            reason="axes agree",
            axes=axes,
        )
        report = fragility_report(config, cells, axes)
        assert report.verdict is FragilityVerdict.UNDEFINED  # n=0 everywhere


class TestElasticityIsMeasuredFromTheBaseline:
    """MEDIUM, and in the flattering direction. `elasticity(N)` is
    `relative_drop(N) / step(N)` with both measured from the committed cell.
    `max_abs_elasticity` was taken from the `DecayStep` pool, and a step
    divides by its own LOWER cell -- so for a neighbour below the baseline on
    an axis the reported decay per unit of parameter change was the smaller
    of two ratios wearing one name.
    """

    @staticmethod
    def _below_baseline_cells(config):
        """Committed cell at 0.5R, its neighbour one step BELOW at 2.0R."""
        return (
            cell(score=65.0, n=10, expectancy=2.0, profit_factor=2.0),
            cell(score=70.0, n=10, expectancy=0.5, profit_factor=2.0),
        )

    def test_the_reported_elasticity_matches_the_documented_formula(self, config):
        """Measured: 9.75. True: 42.0 = |(0.5 - 2.0) / 0.5| / (5 / 70)."""
        report = fragility_report(
            config, self._below_baseline_cells(config), small_axes(config)
        )
        drop = report.worst_relative_drop.require()
        assert drop == pytest.approx(-3.0)  # the neighbour is BETTER
        step = abs(65.0 - 70.0) / 70.0
        assert report.max_abs_elasticity.require() == pytest.approx(abs(drop) / step)
        assert report.max_abs_elasticity.require() == pytest.approx(42.0)

    def test_the_decay_step_keeps_its_own_direction(self, config):
        """The two quantities stay distinguishable: a step is still measured
        low-to-high in its own denominator, which is what makes
        `decay_profile` readable when there is no baseline at all."""
        report = fragility_report(
            config, self._below_baseline_cells(config), small_axes(config)
        )
        step = next(s for s in report.decay_profile if s.from_value == 65.0)
        assert step.is_baseline_neighbour is True
        assert step.relative_drop.require() == pytest.approx(0.75)
        assert step.elasticity.require() == pytest.approx(0.75 / (5.0 / 65.0))

    def test_a_neighbour_above_the_baseline_is_unchanged(self, config):
        """Non-vacuity in the other direction: where the baseline IS the
        lower value the two ratios coincide, and the fix must not inflate
        anything."""
        cells = (
            cell(score=70.0, n=10, expectancy=1.0, profit_factor=2.0),
            cell(score=75.0, n=10, expectancy=0.6, profit_factor=2.0),
        )
        report = fragility_report(config, cells, small_axes(config))
        assert report.max_abs_elasticity.require() == pytest.approx(0.4 / (5.0 / 70.0))

    def test_the_fix_can_only_raise_the_reported_elasticity(self, config):
        """Both pools are kept, so the maximum is over a superset."""
        report = fragility_report(
            config, self._below_baseline_cells(config), small_axes(config)
        )
        from_steps = [
            abs(s.elasticity.require())
            for s in report.decay_profile
            if s.elasticity.is_defined
        ]
        assert report.max_abs_elasticity.require() >= max(from_steps)


class TestAFlatteringRunNamesItsOwnMechanism:
    """MEDIUM. The module tells a reader of a flattering run to "read the
    run's own notes", and offered two mechanisms -- a skipped loser and a
    pinned spread. A delayed or jittered fill is a THIRD: the order is priced
    off a later bar. On the delayed-entry case below the trade set never
    moves and no spread is pinned, so both offered mechanisms are false and
    the run carried no note that applied.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def flattered():
        config = default_config()
        store = store_of(bars_delayed_entry_is_cheaper())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            baseline, rows = run_degradations(
                config,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="a flattering delayed entry must name its mechanism",
                specs=(DegradationSpec(kind=DegradationKind.DELAYED_ENTRY, severity=1.0),),
            )
        return config, baseline, rows[0]

    def test_the_run_really_does_come_out_ahead_with_its_trade_set_intact(
        self, flattered
    ):
        _, baseline, row = flattered
        assert row.improved_anywhere is True
        assert row.n_trades == baseline.trade_count == 1
        assert row.applied == {
            "execution.entry_delay_bars": 1,
            "execution.max_entry_wait_bars": 2,
        }
        assert "execution.spread_ticks_override" not in row.applied

    def test_neither_offered_mechanism_applies(self, flattered):
        """The point of the finding: the trade count is unchanged, so no
        trade was skipped, and nothing pinned a spread."""
        _, baseline, row = flattered
        assert row.metrics["trade_count"].require() == float(baseline.trade_count)
        assert not any("outranks a real quote" in note for note in row.notes)

    def test_the_run_names_the_later_bar_mechanism_in_its_own_notes(self, flattered):
        """Measured: no note on the run mentioned the fill price at all.
        True: the run says a delayed fill is priced off a later bar."""
        _, _, row = flattered
        assert any("priced off a later bar" in note for note in row.notes)
        assert any("trade set unchanged" in note for note in row.notes)

    def test_the_direction_check_offers_all_three_mechanisms(self, flattered):
        _, _, row = flattered
        note = next(c.note for c in row.direction if c.worse_or_equal is False)
        assert "cannot say WHICH mechanism" in note
        assert "skipped loser" in note
        assert "pinned spread can be tighter" in note
        assert "which bar they fill on" in note

    def test_the_assembled_report_offers_all_three_too(self, flattered):
        config, baseline, row = flattered
        report = robustness_report(
            config, (), small_axes(config), degradations=(row,), baseline=baseline
        )
        note = next(n for n in report.notes if "came out ahead" in n)
        assert "Not a result" in note
        assert "pinned spread can be tighter" in note
        assert "which bar they fill on" in note

    def test_a_supra_bar_jitter_carries_the_same_warning(self, config):
        """A random execution delay moves the fill bar for the same reason,
        so it carries the same note -- but only when the jitter can actually
        move it. The sub-bar case says it cannot instead."""
        interval = int(config.data.primary_interval_seconds)
        big = DegradationSpec(
            kind=DegradationKind.RANDOM_EXECUTION_DELAY,
            severity=float(interval) * 1000.0 * 2,
            seed_name="supra",
        )
        notes = _degradation_notes(big, config)
        assert any("priced off a later bar" in note for note in notes)
        assert not any("sub-bar" in note for note in notes)

        small = DegradationSpec(
            kind=DegradationKind.RANDOM_EXECUTION_DELAY, severity=125.0, seed_name="sub"
        )
        notes = _degradation_notes(small, config)
        assert any("sub-bar" in note for note in notes)
        assert not any("priced off a later bar" in note for note in notes)


class TestAVacuousSlippageRunSaysSo:
    """LOW. Added slippage MULTIPLIES `base_slippage_ticks`. At a config
    whose base slippage is zero, every severity re-runs the committed
    configuration -- same config hash, same fills -- while `is_neutral` is
    False, so the row read as a stress test that passed. The committed
    default is 1.0, so this is reachable rather than live.
    """

    def test_the_override_is_a_no_op_at_zero_base_slippage(self, config):
        zero = apply_overrides(config, {"execution.base_slippage_ticks": 0.0})
        spec = DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=3.0)
        assert spec.is_neutral is False
        overrides = spec.overrides(zero, "NQ")
        assert overrides == {"execution.base_slippage_ticks": 0.0}
        assert apply_overrides(zero, overrides).config_hash == zero.config_hash

    def test_the_run_declares_the_vacuity(self, config):
        """Measured: notes == (). True: the row says the run is vacuous."""
        zero = apply_overrides(config, {"execution.base_slippage_ticks": 0.0})
        notes = _degradation_notes(
            DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=3.0), zero
        )
        assert any("VACUOUS" in note for note in notes)
        assert any("base_slippage_ticks is 0" in note for note in notes)

    def test_the_committed_configuration_is_not_declared_vacuous(self, config):
        """Non-vacuity of the guard itself: at the committed 1.0 tick the
        run really does bite, and must not be labelled vacuous."""
        assert config.execution.base_slippage_ticks == 1.0
        notes = _degradation_notes(
            DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=3.0), config
        )
        assert notes == ()

    def test_the_vacuous_run_reaches_the_row_through_run_degradations(self, config):
        zero = apply_overrides(config, {"execution.base_slippage_ticks": 0.0})
        store = store_of(bars_to_target())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            baseline, rows = run_degradations(
                zero,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="a vacuous slippage run",
                specs=(
                    DegradationSpec(kind=DegradationKind.ADDED_SLIPPAGE, severity=3.0),
                ),
            )
        assert rows[0].n_trades == baseline.trade_count == 1
        assert rows[0].metrics["total_pnl"].require() == pytest.approx(
            compute_trade_metrics(
                baseline.trades, split_label=SplitPhase.TRAIN, names=("total_pnl",)
            )["total_pnl"].require()
        )
        assert any("VACUOUS" in note for note in rows[0].notes)


def bars_with_the_stop_inside_the_horizon() -> list[Bar]:
    """A resting limit that fills on the first horizon bar, is stopped on the
    second, and would have reached its target on the third -- the bar the loop
    stands on when the order resolves. PROVENANCE.md 2.6's shape."""
    out = [bar_at(i) for i in range(SIGNAL_INDEX + 1)]
    out.append(
        bar_at(SIGNAL_INDEX + 1, open_=13001.0, high=13002.0, low=12999.0, close=13001.0)
    )
    out.append(
        bar_at(SIGNAL_INDEX + 2, open_=13000.0, high=13001.0, low=12985.0, close=12995.0)
    )
    out.append(
        bar_at(SIGNAL_INDEX + 3, open_=12996.0, high=13015.0, low=12995.0, close=13012.0)
    )
    out.append(bar_at(SIGNAL_INDEX + 4, open_=13000.0))
    return out


class TestTwoMoreWaitHorizonInvariants:
    """MEDIUM, and reported as a limitation rather than a cure.

    `wait_horizon_violations` says it re-checks the f44ddb9 invariant on every
    trade. It does not: the regression understated `mae_r` BECAUSE it skipped
    the bars, so the record it emitted was internally consistent and all of
    the price/time invariants pass on it -- measured below. Two shapes that
    DO survive into a record are now caught, and the honest verification of
    the fix stays the loop-level run in `TestTheWaitHorizonFixHolds`.
    """

    @staticmethod
    def _record(**over) -> TradeRecord:
        base = dict(
            trade_id="t",
            run_id="test_run",
            symbol="NQ",
            config_hash="testhash",
            signal_ts=T0,
            entry_ts=T0 + timedelta(minutes=5),
            exit_ts=T0 + timedelta(minutes=15),
            bars_held=2,
            side=Side.LONG,
            setup=SetupType.SCALP_1R,
            planned_entry=13000.0,
            stop_price=12990.0,
            target_price=13010.0,
            entry_price=13000.0,
            size=1.0,
            filled_size=1.0,
            point_value=20.0,
            risk_dollars=200.0,
            costs=CostBreakdown(),
            flow_score=75.0,
            regime=Regime.LOW_VOL,
            split_label=SplitPhase.TRAIN,
        )
        base.update(over)
        return TradeRecord(**base)

    def _at_target(self, **over) -> TradeRecord:
        return self._record(
            exit_price=13010.0, gross_pnl=200.0, pnl=200.0,
            exit_reason=ExitReason.TARGET, mae_r=-0.2, mfe_r=1.0, **over,
        )

    def test_an_exit_on_a_later_bar_must_have_held_at_least_one_bar(self):
        """Measured: no finding. True: a position whose exit postdates its
        entry was advanced over the exit bar, so bars_held cannot be 0 --
        zero means no bar between the fill and the exit was offered to it."""
        findings = wait_horizon_violations(run_result((self._at_target(bars_held=0),)))
        assert len(findings) == 1
        assert "bars_held=0" in findings[0]
        assert "was ever inspected" in findings[0]

    def test_an_entry_and_exit_in_one_bar_is_not_a_finding(self):
        """Non-vacuity in the safe direction: both fills are stamped at their
        bar's close, so a same-bar resolution shares a timestamp and
        bars_held=0 is correct there."""
        same_bar = self._at_target(
            bars_held=0, exit_ts=T0 + timedelta(minutes=5)
        )
        assert same_bar.exit_ts == same_bar.entry_ts
        assert wait_horizon_violations(run_result((same_bar,))) == ()

    def test_a_stop_exit_whose_excursion_never_reached_the_stop_is_caught(self):
        """The mirror of the excursion check. Excursions come from each held
        bar's high and low and the stop is never moved, so a STOP exit whose
        worst excursion stopped short of the stop was filled at a price no
        bar the position held ever traded."""
        fabricated = self._record(
            exit_price=12990.0, gross_pnl=-200.0, pnl=-200.0,
            exit_reason=ExitReason.STOP, mae_r=-0.4, mfe_r=0.1,
        )
        findings = wait_horizon_violations(run_result((fabricated,)))
        assert len(findings) == 1
        assert "the exit is STOP but mae_r=-0.400000" in findings[0]
        assert "-1.000000R" in findings[0]

    def test_an_honest_stop_at_its_own_bound_is_not_a_finding(self):
        honest = self._record(
            exit_price=12990.0, gross_pnl=-200.0, pnl=-200.0,
            exit_reason=ExitReason.STOP, mae_r=-1.0, mfe_r=0.1,
        )
        assert wait_horizon_violations(run_result((honest,))) == ()

    def test_a_slipped_stop_beyond_its_bound_is_not_a_finding_either(self):
        """ARCHITECTURE.md 0.2: an entry that triggered worse than planned
        realizes MORE than 1R of loss, so mae_r sits past the bound."""
        slipped = self._record(
            entry_price=13002.0, exit_price=12990.0, gross_pnl=-240.0, pnl=-240.0,
            exit_reason=ExitReason.STOP, mae_r=-1.3, mfe_r=0.0,
        )
        assert slipped.r_multiple == pytest.approx(-1.2)
        assert wait_horizon_violations(run_result((slipped,))) == ()

    def test_the_recorded_regression_still_passes_every_record_invariant(self):
        """THE LIMITATION, pinned so nobody reads an empty result as a proof.

        PROVENANCE.md 2.6's trade: TARGET +0.9869R where the truth was STOP
        -1.0756R, with mae_r -0.5 against a true -1.5 and bars_held 5 against
        7. Its own stop sits at -1.1R, its excursion never reached it, and it
        held a positive number of bars -- so the record is self-consistent
        and this function returns nothing. What catches it is the loop-level
        run below, not a record-level invariant.
        """
        fabricated = TradeRecord(
            trade_id="f44ddb9",
            run_id="test_run",
            symbol="NQ",
            config_hash="testhash",
            signal_ts=T0,
            entry_ts=T0 + timedelta(seconds=INTERVAL),
            exit_ts=T0 + timedelta(seconds=INTERVAL * 3),
            bars_held=5,
            side=Side.LONG,
            setup=SetupType.SCALP_1R,
            planned_entry=13000.0,
            stop_price=12990.0,
            target_price=13010.0,
            entry_price=13001.0,
            exit_price=13001.0 + (0.9869 * R_DOLLARS) / (20.0 * 2.0),
            size=2.0,
            filled_size=2.0,
            point_value=20.0,
            exit_reason=ExitReason.TARGET,
            risk_dollars=R_DOLLARS,
            gross_pnl=0.9869 * R_DOLLARS,
            costs=CostBreakdown(),
            pnl=0.9869 * R_DOLLARS,
            mae_r=-0.5,
            mfe_r=0.9869,
            flow_score=71.0,
            regime=Regime.TRENDING_UP,
            split_label=SplitPhase.TRAIN,
        )
        stop_r = -(
            abs(fabricated.entry_price - fabricated.stop_price)
            * fabricated.point_value
            * fabricated.filled_size
        ) / fabricated.risk_dollars
        assert stop_r == pytest.approx(-1.1)
        assert fabricated.r_multiple == pytest.approx(0.9869)
        assert fabricated.mae_r > stop_r  # self-consistent, and wrong
        assert fabricated.bars_held >= 1
        assert wait_horizon_violations(run_result((fabricated,))) == ()

    def test_the_function_says_what_it_cannot_see(self):
        """The limitation is on the function, not only in a test name."""
        assert "WHAT A RECORD CANNOT SHOW" in wait_horizon_violations.__doc__
        assert (
            "the absence of five specific defects" in wait_horizon_violations.__doc__
        )

    def test_the_loop_level_run_is_what_catches_it(self):
        """The same scenario through the real loop: STOP -1.0756R, mae_r
        -1.5, and no violation reported -- the fix holding, measured."""
        resting = apply_overrides(
            default_config(),
            {"execution.entry_order_type": "limit", "execution.max_entry_wait_bars": 3},
        )
        result = stub_runner(store_of(bars_with_the_stop_inside_the_horizon()))(
            resting, one_fold()
        )
        assert result.trade_count == 1
        trade = result.trades[0]
        assert trade.exit_reason is ExitReason.STOP
        assert trade.r_multiple == pytest.approx(-1.0756)
        assert trade.mae_r == pytest.approx(-1.5)
        assert trade.bars_held >= 1
        assert wait_horizon_violations(result) == ()


class TestTheWiderSpreadRunOverTheMultiBarHorizon:
    """The second degradation the brief names as a consumer of the
    `max_entry_wait_bars > 1` path. A wider spread does not widen the
    horizon, so it is run against a config that already rests its entries:
    the horizon is three bars, the fill lands on the first, and the stop sits
    in a bar that only the f44ddb9 replay inspects.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def run():
        resting = apply_overrides(
            default_config(),
            {"execution.entry_order_type": "limit", "execution.max_entry_wait_bars": 3},
        )
        store = store_of(bars_with_the_stop_inside_the_horizon())
        with ExperimentLog(":memory:", repo_root="/nonexistent") as handle:
            baseline, rows = run_degradations(
                resting,
                one_fold(),
                stub_runner(store),
                "NQ",
                log=handle,
                reason="wider spreads over the multi-bar wait horizon",
                specs=(
                    DegradationSpec(kind=DegradationKind.WIDER_SPREAD, severity=1.0),
                    DegradationSpec(kind=DegradationKind.WIDER_SPREAD, severity=2.0),
                ),
            )
        return resting, baseline, rows

    def test_the_path_is_really_the_multi_bar_one(self, run):
        resting, baseline, _ = run
        assert int(resting.execution.max_entry_wait_bars) == 3
        assert str(resting.execution.entry_order_type) == "limit"
        order = PendingOrder(
            intent=_intent_for(T0),
            submitted_ts=T0,
            order_type=resting.execution.entry_order_type,
            latency_seconds=float(resting.execution.latency_ms) / 1000.0,
            max_entry_wait_bars=int(resting.execution.max_entry_wait_bars),
        )
        assert order.wait_bars == 3
        assert baseline.trades[0].exit_reason is ExitReason.STOP

    def test_every_run_reports_the_invariant_on_a_real_trade(self, run):
        _, _, rows = run
        assert [row.n_trades for row in rows] == [1, 1]
        assert all(row.wait_horizon_findings == () for row in rows)

    def test_the_wider_spread_only_costs(self, run):
        _, _, rows = run
        assert all(
            check.worse_or_equal is not False for row in rows for check in row.direction
        )
        neutral, wider = rows
        assert neutral.spec.is_neutral is True
        assert wider.applied["execution.spread_ticks_override"] == pytest.approx(2.0)
        assert wider.metrics["total_pnl"].require() < neutral.metrics[
            "total_pnl"
        ].require()

    def test_the_report_states_how_many_trades_the_invariant_saw(self, run):
        resting, _, rows = run
        report = robustness_report(resting, (), small_axes(resting), degradations=rows)
        assert report.wait_horizon_findings() == ()
