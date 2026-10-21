"""Tests for `monte_carlo/engine.py` (ARCHITECTURE.md 11).

Ordered the way the brief orders the risks. The n=0 block comes first because
n=0 is the live case: at the committed config the system takes no trades, so
the refusal path is the one that runs, and a Monte Carlo that answered it with
a flat fan at the starting balance would be this module's worst defect rather
than an edge case.

The guarantees under test are structural wherever they can be:

* a refusal and a distribution are DIFFERENT TYPES, the refusal declares no
  distributional field, and the distribution cannot be constructed at n=0;
* `STATED_LIMITATION` is a required, verbatim-validated field, checked here
  against ARCHITECTURE.md itself so the sentence cannot drift from the doc;
* `MetricValue.n` is the resample POOL size everywhere, never the simulation
  count, enforced by the distribution's own validator;
* an unobserved regime gets no state and a terminal regime gets no row, and
  `RegimeTransitions` REFUSES to hold one, so "no smoothing" is not a promise.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from flow_model.analytics.metrics import (
    MetricValue,
    MixedSplitError,
    SampleStatus,
    SplitLabelMismatchError,
    Undefined,
)
from flow_model.config.schema import AnalyticsConfig, MonteCarloConfig, RiskConfig
from flow_model.core.contracts import CostBreakdown, TradeRecord
from flow_model.core.enums import (
    ExitReason,
    MonteCarloMethod,
    Regime,
    SetupType,
    Side,
    SplitPhase,
)
from flow_model.core.determinism import rng
from flow_model.monte_carlo.engine import (
    DRAWDOWN_HISTOGRAM_EDGES,
    FAN_MAX_STEPS,
    MAX_SIMULATIONS,
    PROVENANCE_NOTE,
    STATED_LIMITATION,
    EquityFan,
    FanBand,
    MonteCarloDistribution,
    MonteCarloPlan,
    MonteCarloRefusal,
    RegimeTransitions,
    _block_indices,
    _chunk_sims,
    compare_methods,
    estimate_transitions,
    run_monte_carlo,
)

ENGINE = Path(__file__).resolve().parents[2] / "flow_model" / "monte_carlo" / "engine.py"
ARCHITECTURE = Path(__file__).resolve().parents[2] / "ARCHITECTURE.md"
METHODS = tuple(MonteCarloMethod)

#: Entry/stop geometry shared by the fixtures. R is $400 at $20/point on a
#: 20-point stop, which is the unit PROVENANCE.md 2.5 measures costs in.
ENTRY = 18000.0
STOP = 17980.0
POINT_VALUE = 20.0
RISK_DOLLARS = 400.0


@pytest.fixture
def trade():
    """A record whose realized R is exactly what the caller asks for.

    The exit price is derived from the wanted R so `TradeRecord`'s own
    validators accept it: nothing here fabricates a P&L that the price path
    does not support, and `r_multiple` is read off the record by the engine
    rather than passed in beside it.
    """

    def _make(
        i: int,
        r: float,
        regime: Regime = Regime.LOW_VOL,
        *,
        split_label: SplitPhase | None = SplitPhase.TRAIN,
        data_hash: str = "dat",
        config_hash: str = "cfg",
    ) -> TradeRecord:
        exit_price = ENTRY + (r * RISK_DOLLARS) / POINT_VALUE
        gross = (exit_price - ENTRY) * POINT_VALUE
        ts = datetime(2020, 6, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        return TradeRecord(
            trade_id=f"t{i}",
            run_id="run",
            symbol="NQ",
            config_hash=config_hash,
            data_hash=data_hash,
            signal_ts=ts,
            entry_ts=ts + timedelta(minutes=1),
            exit_ts=ts + timedelta(minutes=10),
            side=Side.LONG,
            setup=SetupType.SCALP_1R,
            planned_entry=ENTRY,
            stop_price=STOP,
            target_price=ENTRY + 20.0,
            entry_price=ENTRY,
            exit_price=exit_price,
            size=1.0,
            filled_size=1.0,
            point_value=POINT_VALUE,
            exit_reason=ExitReason.TARGET if r > 0 else ExitReason.STOP,
            risk_dollars=RISK_DOLLARS,
            gross_pnl=gross,
            costs=CostBreakdown(),
            pnl=gross,
            flow_score=75.0,
            regime=regime,
            split_label=split_label,
        )

    return _make


@pytest.fixture
def plan():
    """A cheap plan. Built in-process; no YAML default is read or changed."""
    return MonteCarloPlan.default().replace(n_simulations=400, n_curves_to_store=20)


@pytest.fixture
def pool(trade):
    """40 trades, two thirds winners, three regimes cycling."""
    regimes = (Regime.LOW_VOL, Regime.HIGH_VOL, Regime.CHOP)
    return [trade(i, 1.0 if i % 3 else -1.0, regimes[i % 3]) for i in range(40)]


# ===========================================================================
# 1. n = 0 -- THE LIVE PATH. Built and tested first.
# ===========================================================================


@pytest.mark.parametrize("method", METHODS)
def test_zero_trades_refuses_on_every_method(plan, method):
    result = run_monte_carlo([], split_label=None, plan=plan, method=method)
    assert isinstance(result, MonteCarloRefusal)
    assert result.undefined is Undefined.NO_TRADES
    assert result.n_trades == 0
    assert result.method is method


def test_the_refusal_type_declares_no_distributional_field():
    """The structural half of the n=0 guarantee: there is nowhere to put one."""
    banned = (
        "terminal_equity",
        "drawdown_percentiles",
        "p_drawdown_at_least",
        "p_loss_streak_at_least",
        "p_ruin",
        "fan",
        "sample_curves",
        "drawdown_histogram",
        "worst_drawdown",
        "n_simulations",
        "sample",
    )
    fields = set(MonteCarloRefusal.model_fields) | set(MonteCarloRefusal.model_computed_fields)
    assert not fields & set(banned), f"refusal carries distributional fields: {fields}"


@pytest.mark.parametrize(
    "attribute",
    ["terminal_equity", "fan", "p_ruin", "p_drawdown_at_least", "sample", "n_simulations"],
)
def test_reading_a_percentile_off_a_refusal_raises(plan, attribute):
    result = run_monte_carlo([], split_label=None, plan=plan)
    assert not hasattr(result, attribute)
    with pytest.raises(AttributeError):
        getattr(result, attribute)


def test_a_percentile_cannot_be_attached_to_a_refusal(plan):
    with pytest.raises(ValidationError):
        MonteCarloRefusal(
            method=plan.method,
            split_label=None,
            n_trades=0,
            plan=plan,
            limitation=STATED_LIMITATION,
            notes=(PROVENANCE_NOTE,),
            undefined=Undefined.NO_TRADES,
            terminal_equity={"p50": MetricValue.measured("x", 100_000.0, n=0)},
        )


def test_the_distribution_type_cannot_be_built_at_zero_trades(plan, pool):
    """The other structural half: `n_trades` is `ge=1` on the populated type."""
    built = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert isinstance(built, MonteCarloDistribution)
    with pytest.raises(ValidationError):
        built.replace(n_trades=0)
    assert MonteCarloDistribution.model_fields["n_trades"].metadata


def test_zero_trades_produces_no_number_resembling_a_flat_fan(plan):
    """Nothing in the dumped refusal is a percentile of the starting balance."""
    dumped = run_monte_carlo([], split_label=None, plan=plan).model_dump()
    assert "plan" in dumped
    for key in dumped:
        assert "percentile" not in key
        assert not key.startswith("p_")
        assert key not in {"terminal_equity", "fan", "sample_curves"}


def test_the_zero_trade_note_says_why_rather_than_reporting_a_failure(plan):
    result = run_monte_carlo([], split_label=None, plan=plan)
    joined = " ".join(result.notes).lower()
    assert "n=0" in joined
    assert "not a failure" in joined
    assert "empirical r distribution" in joined


def test_a_refusal_reason_and_pool_size_must_agree(plan):
    common = dict(
        method=plan.method,
        split_label=None,
        plan=plan,
        limitation=STATED_LIMITATION,
        notes=(PROVENANCE_NOTE,),
    )
    with pytest.raises(ValidationError, match="disagree"):
        MonteCarloRefusal(n_trades=3, undefined=Undefined.NO_TRADES, **common)
    with pytest.raises(ValidationError, match="disagree"):
        MonteCarloRefusal(n_trades=0, undefined=Undefined.INSUFFICIENT_SAMPLE, **common)


def test_the_refusal_describes_itself_as_carrying_nothing(plan):
    lines = run_monte_carlo([], split_label=None, plan=plan).describe()
    assert lines[0].startswith("LIMITATION: ")
    assert "no percentile, no probability, no fan" in lines[1]


# ===========================================================================
# 2. The stated limitation, as a required field
# ===========================================================================


def test_the_limitation_matches_architecture_section_11_verbatim():
    """Checked against the document, so the sentence cannot drift from it.

    Compared after stripping markdown emphasis and collapsing whitespace --
    the doc italicizes `sequence` and `model`, which the constant renders as
    capitals. Case is normalized for exactly that reason; the words are not.
    """
    section = ARCHITECTURE.read_text(encoding="utf-8").split("## 11. Monte Carlo")[1]
    section = section.split("## 12.")[0]

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("*", "")).strip().lower()

    assert normalize(STATED_LIMITATION) in normalize(section)


@pytest.mark.parametrize("method", METHODS)
def test_every_result_carries_the_limitation(plan, pool, method):
    for trades in ([], pool):
        result = run_monte_carlo(trades, split_label=SplitPhase.TRAIN if trades else None,
                                 plan=plan, method=method)
        assert result.limitation == STATED_LIMITATION
        assert result.model_dump()["limitation"] == STATED_LIMITATION
        assert result.describe()[0] == f"LIMITATION: {STATED_LIMITATION}"


@pytest.mark.parametrize("bad", ["", "MC bounds sequence risk.", STATED_LIMITATION[:-1]])
def test_a_weakened_limitation_is_refused(plan, bad):
    with pytest.raises(ValidationError, match="verbatim"):
        MonteCarloRefusal(
            method=plan.method,
            split_label=None,
            n_trades=0,
            plan=plan,
            limitation=bad,
            notes=(PROVENANCE_NOTE,),
            undefined=Undefined.NO_TRADES,
        )


def test_the_limitation_has_no_default_so_it_cannot_be_omitted():
    for model in (MonteCarloRefusal, MonteCarloDistribution):
        assert model.model_fields["limitation"].is_required()


def test_the_provenance_note_is_required_on_every_result(plan, pool):
    assert PROVENANCE_NOTE in run_monte_carlo(pool, split_label=SplitPhase.TRAIN,
                                              plan=plan).notes
    assert "synthetic.py" in PROVENANCE_NOTE
    with pytest.raises(ValidationError, match="PROVENANCE_NOTE"):
        MonteCarloRefusal(
            method=plan.method,
            split_label=None,
            n_trades=0,
            plan=plan,
            limitation=STATED_LIMITATION,
            notes=(),
            undefined=Undefined.NO_TRADES,
        )


def test_the_input_provenance_is_carried_so_a_figure_is_traceable(plan, trade):
    pool = [trade(i, 1.0, data_hash=f"d{i % 2}", config_hash="cfg") for i in range(6)]
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert result.data_hashes == ("d0", "d1")
    assert result.config_hashes == ("cfg",)


# ===========================================================================
# 3. Small n is honest about being small
# ===========================================================================


@pytest.mark.parametrize("n", [1, 2, 3, 29])
def test_a_pool_below_the_declared_minimum_is_insufficient(plan, trade, n):
    pool = [trade(i, 1.0 if i % 2 else -1.0) for i in range(n)]
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert plan.min_sample_n == AnalyticsConfig().min_sample_n == 30
    assert result.sample is SampleStatus.INSUFFICIENT
    note = " ".join(result.notes)
    plural = "" if n == 1 else "s"
    assert f"a bootstrap over {n} trade{plural} resamples the same {n}," in note
    assert "smoother, not better founded" in note


def test_an_adequate_pool_is_not_flagged(plan, pool):
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert result.sample is SampleStatus.ADEQUATE
    assert not any("below min_sample_n" in note for note in result.notes)


@pytest.mark.parametrize("method", METHODS)
def test_n_is_the_pool_size_and_never_the_simulation_count(trade, method):
    """100,000 sims over 3 trades carries three trades of information."""
    pool = [trade(i, 1.0 if i else -1.0) for i in range(3)]
    plan = MonteCarloPlan.default().replace(n_simulations=5_000, n_curves_to_store=0)
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan, method=method)
    assert result.n_simulations == 5_000
    assert {value.n for value in result.metric_values()} == {3}
    assert result.fan.n_trades == 3


def test_a_metric_value_carrying_the_simulation_count_is_refused(plan, pool):
    built = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    with pytest.raises(ValidationError, match="never the"):
        built.replace(p_ruin=MetricValue.measured("p_ruin", 0.0, n=plan.n_simulations))


def test_an_undefined_metric_cannot_live_on_a_distribution(plan, pool):
    built = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    with pytest.raises(ValidationError, match="MonteCarloRefusal's job"):
        built.replace(
            p_ruin=MetricValue.undefined_because("p_ruin", Undefined.NO_TRADES, n=40)
        )


def test_a_distribution_with_no_percentile_is_refused(plan, pool):
    built = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    with pytest.raises(ValidationError, match="is a refusal"):
        built.replace(terminal_equity={})


def test_a_single_trade_pool_is_a_point_mass_and_says_so(plan, trade):
    result = run_monte_carlo([trade(0, 1.0)], split_label=SplitPhase.TRAIN, plan=plan)
    assert result.n_trades == 1
    assert len({v.value for v in result.terminal_equity.values()}) == 1
    assert any("point mass" in note for note in result.notes)


# ===========================================================================
# 4. The three methods
# ===========================================================================


def test_iid_resamples_the_empirical_r_distribution(trade):
    """The support of a one-trade path is exactly the pool's own outcomes."""
    values = (-1.5, -1.0, 0.0, 2.0)
    pool = [trade(i, r) for i, r in enumerate(values)]
    plan = MonteCarloPlan.default().replace(
        n_simulations=4_000, trades_per_simulation=1, method=MonteCarloMethod.IID
    )
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    expected = {
        round(plan.starting_equity * (1.0 + plan.risk_fraction * r), 6) for r in values
    }
    drawn = {round(curve[-1], 6) for curve in result.sample_curves}
    assert drawn <= expected
    terminals = [v.value for v in result.terminal_equity.values()]
    assert min(terminals) >= min(expected) and max(terminals) <= max(expected)


def test_block_draws_are_contiguous_runs_of_the_observed_sequence():
    drawn = _block_indices(np.random.default_rng(1), n_pool=12, sims=200, length=12, block=4)
    for row in drawn:
        for start in range(0, 12, 4):
            block = row[start : start + 4]
            steps = (np.diff(block) - 1) % 12
            assert not steps.any(), f"block {block} is not contiguous"


def test_block_wrapping_keeps_every_outcome_equally_likely():
    """A truncating variant under-samples the tail; the circular one does not."""
    drawn = _block_indices(np.random.default_rng(3), n_pool=10, sims=20_000, length=10, block=5)
    counts = np.bincount(drawn.ravel(), minlength=10)
    assert counts.min() > 0
    assert counts.max() / counts.min() < 1.1


def test_block_preserves_the_loss_clustering_iid_destroys(trade):
    """The thing section 11 says actually kills accounts.

    The pool holds ten consecutive losses followed by thirty wins. Blocks of
    twenty reproduce that run; iid shatters it.
    """
    pool = [trade(i, -1.0 if i < 10 else 1.0) for i in range(40)]
    plan = MonteCarloPlan.default().replace(n_simulations=2_000, n_curves_to_store=0)
    block = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan,
                            method=MonteCarloMethod.BLOCK)
    iid = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan,
                          method=MonteCarloMethod.IID)
    assert block.p_loss_streak_at_least["5"].require() > iid.p_loss_streak_at_least["5"].require()
    assert block.worst_drawdown.require() < iid.worst_drawdown.require()


def test_a_block_as_long_as_the_pool_is_reported_as_rotations_only(plan, pool):
    big = plan.replace(block_length=len(pool) + 5, method=MonteCarloMethod.BLOCK)
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=big)
    assert result.effective_block_length == len(pool)
    joined = " ".join(result.notes)
    assert "clamped from 45" in joined
    assert f"distribution over {len(pool)} rotations" in joined


def test_block_declares_the_wrap_it_introduces(plan, pool):
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan,
                             method=MonteCarloMethod.BLOCK)
    assert any("a transition that never occurred" in note for note in result.notes)


def test_only_the_regime_aware_method_reports_a_chain(plan, pool):
    results = compare_methods(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert results[MonteCarloMethod.IID].transitions is None
    assert results[MonteCarloMethod.BLOCK].transitions is None
    assert results[MonteCarloMethod.REGIME_AWARE].transitions is not None
    built = results[MonteCarloMethod.IID]
    with pytest.raises(ValidationError, match="regime-aware"):
        built.replace(transitions=results[MonteCarloMethod.REGIME_AWARE].transitions)


# --- the transition matrix -------------------------------------------------


def test_transitions_are_the_raw_observed_ratios():
    sequence = [Regime.LOW_VOL, Regime.HIGH_VOL, Regime.LOW_VOL, Regime.HIGH_VOL, Regime.LOW_VOL]
    chain = estimate_transitions(sequence)
    assert chain.states == (Regime.LOW_VOL, Regime.HIGH_VOL)
    assert chain.counts == ((0, 2), (2, 0))
    assert chain.rows == {"LOW_VOL": (0.0, 1.0), "HIGH_VOL": (1.0, 0.0)}
    assert chain.n_transitions == 4
    assert chain.occupancy == (3, 2)
    assert chain.initial_distribution == (3 / 5, 2 / 5)


def test_a_regime_never_observed_gets_no_state_and_no_row():
    chain = estimate_transitions([Regime.LOW_VOL, Regime.HIGH_VOL, Regime.LOW_VOL])
    absent = {Regime.CHOP, Regime.TRENDING_UP, Regime.TRENDING_DOWN,
              Regime.DIRECTIONAL_SHIFT, Regime.UNKNOWN}
    assert set(chain.unobserved_regimes) == absent
    assert not absent & set(chain.states)
    assert not absent & {Regime(key) for key in chain.rows}
    # no uniform prior anywhere: every row is a ratio of integers it reports
    for i, state in enumerate(chain.states):
        total = sum(chain.counts[i])
        assert chain.rows[state.value] == tuple(c / total for c in chain.counts[i])


def test_a_regime_observed_only_last_is_terminal_and_has_no_row():
    chain = estimate_transitions([Regime.LOW_VOL, Regime.HIGH_VOL, Regime.CHOP])
    assert chain.terminal_states == (Regime.CHOP,)
    assert "CHOP" not in chain.rows
    assert Regime.CHOP in chain.states  # observed, so NOT dropped
    assert chain.counts[chain.states.index(Regime.CHOP)] == (0, 0, 0)


def test_the_chain_refuses_a_row_for_a_terminal_state():
    """The structural half of "no smoothing": the model will not hold one."""
    with pytest.raises(ValidationError, match="uniform prior"):
        RegimeTransitions(
            states=(Regime.LOW_VOL, Regime.CHOP),
            occupancy=(2, 1),
            counts=((1, 0), (0, 0)),
            rows={"LOW_VOL": (1.0, 0.0), "CHOP": (0.5, 0.5)},
            initial_distribution=(2 / 3, 1 / 3),
            terminal_states=(Regime.CHOP,),
            unobserved_regimes=(),
            n_transitions=1,
        )


def test_the_chain_refuses_a_state_for_an_unobserved_regime():
    with pytest.raises(ValidationError, match="listed unobserved but present"):
        RegimeTransitions(
            states=(Regime.LOW_VOL, Regime.CHOP),
            occupancy=(2, 1),
            counts=((1, 0), (1, 0)),
            rows={"LOW_VOL": (1.0, 0.0), "CHOP": (1.0, 0.0)},
            initial_distribution=(2 / 3, 1 / 3),
            terminal_states=(),
            unobserved_regimes=(Regime.CHOP,),
            n_transitions=2,
        )


@pytest.mark.parametrize(
    "rows, match",
    [
        ({"LOW_VOL": (0.4, 0.4)}, "not a probability distribution"),
        ({"LOW_VOL": (1.5, -0.5)}, "not a probability distribution"),
        ({"HIGH_VOL": (1.0, 0.0)}, "must belong to states"),
    ],
)
def test_the_chain_refuses_a_malformed_row(rows, match):
    with pytest.raises(ValidationError, match=match):
        RegimeTransitions(
            states=(Regime.LOW_VOL, Regime.CHOP),
            occupancy=(1, 1),
            counts=((0, 1), (1, 0)),
            rows=rows,
            initial_distribution=(0.5, 0.5),
            terminal_states=(),
            unobserved_regimes=(),
            n_transitions=2,
        )


def test_the_unobserved_regimes_are_named_in_the_report(plan, trade):
    pool = [trade(i, 1.0, Regime.LOW_VOL) for i in range(5)]
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan,
                             method=MonteCarloMethod.REGIME_AWARE)
    note = " ".join(result.notes)
    # Listed in `Regime` declaration order, which is why HIGH_VOL leads.
    assert "never observed, given no state: HIGH_VOL, TRENDING_UP" in note
    assert "Nothing was smoothed into existence" in note
    assert result.transitions.states == (Regime.LOW_VOL,)
    assert result.transitions.rows == {"LOW_VOL": (1.0,)}


# --- regime-aware simulation behaviour ------------------------------------


def test_regime_aware_draws_only_from_within_the_sequenced_regime(trade):
    """A deterministic alternating chain pins the whole path.

    LOW_VOL trades are all +1R and HIGH_VOL trades all -1R, and each regime
    only ever follows the other, so a correctly stratified five-step path is
    either +,-,+,-,+ or -,+,-,+,- and the longest loss run is exactly one. A
    leak across strata would produce a run of two.
    """
    pool = [
        trade(i, 1.0 if i % 2 == 0 else -1.0,
              Regime.LOW_VOL if i % 2 == 0 else Regime.HIGH_VOL)
        for i in range(20)
    ]
    plan = MonteCarloPlan.default().replace(
        n_simulations=3_000,
        trades_per_simulation=5,
        method=MonteCarloMethod.REGIME_AWARE,
        n_curves_to_store=3_000,
    )
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert result.worst_max_loss_streak.require() == 1.0
    assert result.n_truncated_paths == 0
    assert len({round(curve[-1], 6) for curve in result.sample_curves}) == 2


def test_a_path_entering_a_terminal_state_ends_rather_than_being_invented(trade):
    pool = [
        trade(i, 1.0 if i % 2 == 0 else -1.0,
              Regime.LOW_VOL if i % 2 == 0 else Regime.HIGH_VOL)
        for i in range(19)
    ]
    pool.append(trade(19, -1.0, Regime.CHOP))
    plan = MonteCarloPlan.default().replace(
        n_simulations=2_000, trades_per_simulation=6, method=MonteCarloMethod.REGIME_AWARE
    )
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert result.transitions.terminal_states == (Regime.CHOP,)
    assert 0 < result.n_truncated_paths <= plan.n_simulations
    assert result.mean_path_length.require() < result.path_length
    assert any("ended early in a terminal regime state" in n for n in result.notes)
    assert any("mixture over path lengths" in n for n in result.notes)


def test_a_truncated_step_adds_no_trade_and_no_loss(trade):
    """A one-state chain with no transition truncates every path at step one.

    The equity curve must then be flat after the first trade rather than
    accumulating invented outcomes, and the reported mean path length must
    say one rather than the nominal length.
    """
    plan = MonteCarloPlan.default().replace(
        n_simulations=200,
        trades_per_simulation=4,
        method=MonteCarloMethod.REGIME_AWARE,
        n_curves_to_store=200,
    )
    result = run_monte_carlo([trade(0, -1.0, Regime.CHOP)], split_label=SplitPhase.TRAIN,
                             plan=plan)
    assert result.transitions.terminal_states == (Regime.CHOP,)
    assert result.mean_path_length.require() == 1.0
    assert result.n_truncated_paths == 200
    for curve in result.sample_curves:
        assert len(set(curve[1:])) == 1  # one trade, then flat


# ===========================================================================
# 5. Determinism
# ===========================================================================


@pytest.mark.parametrize("method", METHODS)
def test_the_same_plan_and_pool_give_bit_identical_output(plan, pool, method):
    first = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan, method=method)
    second = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan, method=method)
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize("method", METHODS)
def test_a_different_seed_gives_a_different_draw(plan, pool, method):
    first = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan, method=method)
    other = run_monte_carlo(pool, split_label=SplitPhase.TRAIN,
                            plan=plan.replace(seed=plan.seed + 1), method=method)
    assert first.model_dump_json() != other.model_dump_json()


def test_each_method_draws_from_its_own_named_stream(plan, pool):
    """Adding a method cannot move an existing method's draws.

    Compared on the whole result rather than on a single percentile: with a
    two-valued R pool and compounding, terminal equity is a function of the
    win COUNT alone, so two methods can legitimately share a median while
    drawing completely different sequences.
    """
    streams = {m: rng(plan.seed, "monte_carlo", m.value).random(8).tobytes() for m in METHODS}
    assert len(set(streams.values())) == len(METHODS)
    results = compare_methods(pool, split_label=SplitPhase.TRAIN, plan=plan)
    dumps = {m: r.model_dump_json(exclude={"method", "notes", "plan"}) for m, r in results.items()}
    assert len(set(dumps.values())) == len(METHODS)


@pytest.mark.parametrize("method", METHODS)
def test_no_computation_touches_the_global_numpy_state(plan, pool, method):
    np.random.seed(0)
    expected = np.random.random()
    np.random.seed(0)
    run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan, method=method)
    assert np.random.random() == expected


def test_the_chunk_size_is_a_pure_function_of_the_path_length():
    assert _chunk_sims(50, 10_000) == _chunk_sims(50, 10_000)
    assert _chunk_sims(1, 10) == 10
    assert _chunk_sims(10**9, 10_000) == 1


@pytest.mark.parametrize("method", METHODS)
def test_a_run_spanning_several_chunks_still_runs_every_simulation(
    monkeypatch, plan, pool, method
):
    monkeypatch.setattr("flow_model.monte_carlo.engine.SIM_ELEMENT_BUDGET", 400)
    small = plan.replace(n_simulations=97, n_curves_to_store=97)
    assert _chunk_sims(len(pool), 97) < 97
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=small, method=method)
    assert result.n_simulations == 97
    assert result.fan.n_simulations == 97
    assert len(result.sample_curves) == 97
    assert sum(b.count for b in result.drawdown_histogram) == 97
    again = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=small, method=method)
    assert result.model_dump_json() == again.model_dump_json()


# ===========================================================================
# 6. Compounding vs fixed fractional
# ===========================================================================


@pytest.mark.parametrize("compound, expected", [(True, 1.005), (False, 1.005)])
def test_one_trade_arithmetic_is_the_declared_formula(trade, compound, expected):
    plan = MonteCarloPlan.default().replace(
        n_simulations=100, trades_per_simulation=1, compound=compound
    )
    result = run_monte_carlo([trade(0, 1.0)], split_label=SplitPhase.TRAIN, plan=plan)
    assert result.terminal_equity["p50"].require() == pytest.approx(
        plan.starting_equity * expected
    )


def test_compounding_and_fixed_differ_materially_on_drawdowns(trade):
    """Section 11 makes this configurable because it changes the answer."""
    pool = [trade(i, 2.0 if i % 2 else -1.0) for i in range(30)]
    plan = MonteCarloPlan.default().replace(
        n_simulations=1_000, risk_fraction=0.05, trades_per_simulation=60,
        n_curves_to_store=0, method=MonteCarloMethod.IID,
    )
    compounded = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    fixed = run_monte_carlo(pool, split_label=SplitPhase.TRAIN,
                            plan=plan.replace(compound=False))
    assert compounded.worst_drawdown.require() != fixed.worst_drawdown.require()
    assert compounded.terminal_equity["p95"].require() != fixed.terminal_equity["p95"].require()


def test_a_ruined_account_stays_at_zero_and_is_counted(trade):
    plan = MonteCarloPlan.default().replace(
        n_simulations=200, risk_fraction=0.5, compound=False,
        trades_per_simulation=6, n_curves_to_store=200,
    )
    result = run_monte_carlo([trade(i, -1.0) for i in range(3)],
                             split_label=SplitPhase.TRAIN, plan=plan)
    assert result.p_ruin.require() == 1.0
    assert result.worst_drawdown.require() == -1.0
    assert result.terminal_equity["p50"].require() == 0.0
    for curve in result.sample_curves:
        assert min(curve) >= 0.0
        assert curve[-1] == 0.0


def test_a_bounded_loss_sequence_cannot_ruin_a_compounding_account(trade):
    plan = MonteCarloPlan.default().replace(
        n_simulations=500, compound=True, trades_per_simulation=200, n_curves_to_store=0
    )
    result = run_monte_carlo([trade(i, -1.0) for i in range(3)],
                             split_label=SplitPhase.TRAIN, plan=plan)
    assert result.p_ruin.require() == 0.0
    assert result.worst_drawdown.require() > -1.0


def test_a_compounding_account_that_underflows_to_zero_is_counted_as_ruined(trade):
    """Ruin is read off the curve, not off the per-trade factor.

    Every factor here is 0.01 -- strictly positive -- yet 400 of them
    multiply to below float resolution, so the balance reaches exactly 0.0.
    Testing the factor alone would report a 100% drawdown beside
    `p_ruin = 0`, which is incoherent in a report.
    """
    plan = MonteCarloPlan.default().replace(
        n_simulations=50, compound=True, risk_fraction=0.99,
        trades_per_simulation=400, n_curves_to_store=50,
    )
    result = run_monte_carlo([trade(0, -1.0)], split_label=SplitPhase.TRAIN, plan=plan)
    assert result.p_ruin.require() == 1.0
    assert result.worst_drawdown.require() == -1.0
    assert result.terminal_equity["p50"].require() == 0.0
    assert min(min(curve) for curve in result.sample_curves) == 0.0


def test_a_compounding_overflow_is_undefined_rather_than_infinite(trade):
    """`MetricValue` refuses a non-finite float, so this must be a refusal."""
    plan = MonteCarloPlan.default().replace(
        n_simulations=10, compound=True, risk_fraction=0.99,
        trades_per_simulation=100_000, n_curves_to_store=0,
    )
    result = run_monte_carlo([trade(0, 500.0)], split_label=SplitPhase.TRAIN, plan=plan)
    assert isinstance(result, MonteCarloRefusal)
    assert result.undefined is Undefined.NOT_REPRESENTABLE
    assert any("overflows a float" in note for note in result.notes)


def test_the_dollar_value_of_one_r_is_declared_and_checkable(plan, trade):
    pool = [trade(i, 1.0) for i in range(5)]
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert result.r_dollars_modeled.require() == plan.starting_equity * plan.risk_fraction
    assert result.mean_risk_dollars_observed.require() == RISK_DOLLARS
    assert result.r_dollars_modeled.require() != result.mean_risk_dollars_observed.require()


def test_no_terminal_figure_is_quoted_in_r(plan, pool):
    """Under compounding the dollar value of one R moves along the path."""
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert not any(v.unit == "R" for v in result.metric_values())


# ===========================================================================
# 7. Split labels (section 0.8) and R (section 0.2)
# ===========================================================================


def test_mixed_split_phases_are_refused(plan, trade):
    pool = [trade(0, 1.0, split_label=SplitPhase.TRAIN),
            trade(1, 1.0, split_label=SplitPhase.TEST)]
    with pytest.raises(MixedSplitError):
        run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)


def test_a_mislabelled_pool_is_refused(plan, trade):
    with pytest.raises(SplitLabelMismatchError):
        run_monte_carlo([trade(0, 1.0, split_label=SplitPhase.TRAIN)],
                        split_label=SplitPhase.TEST, plan=plan)


def test_the_split_label_is_carried_onto_both_result_types(plan, trade):
    assert run_monte_carlo([], split_label=SplitPhase.TEST,
                           plan=plan).split_label is SplitPhase.TEST
    built = run_monte_carlo([trade(0, 1.0, split_label=SplitPhase.TEST)],
                            split_label=SplitPhase.TEST, plan=plan)
    assert built.split_label is SplitPhase.TEST


def test_r_is_read_off_the_record_and_never_re_derived(trade):
    """The pool is exactly `[t.r_multiple]`, the planned unit of section 0.2."""
    pool = [trade(0, -1.0756), trade(1, 0.9869)]
    plan = MonteCarloPlan.default().replace(
        n_simulations=2_000, trades_per_simulation=1,
        method=MonteCarloMethod.IID, n_curves_to_store=2_000, compound=False,
    )
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    expected = {
        round(plan.starting_equity + plan.r_dollars * t.r_multiple, 6) for t in pool
    }
    assert {round(curve[-1], 6) for curve in result.sample_curves} == expected


# ===========================================================================
# 8. The outputs section 11 names
# ===========================================================================


def test_the_committed_plan_is_the_committed_config():
    plan = MonteCarloPlan.default()
    assert plan.n_simulations == 10_000
    assert plan.method is MonteCarloMethod.BLOCK
    assert plan.block_length == 20
    assert plan.compound is True
    assert plan.percentiles == (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0)
    assert plan.drawdown_thresholds == (0.10, 0.20, 0.30)
    assert plan.streak_thresholds == (5, 10, 20)
    assert plan.starting_equity == RiskConfig().starting_equity
    assert plan.risk_fraction == RiskConfig().risk_per_trade_pct
    assert plan.seed == MonteCarloConfig().seed


@pytest.mark.parametrize("method", METHODS)
def test_every_named_output_is_present(plan, pool, method):
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan, method=method)
    assert set(result.terminal_equity) == {"p5", "p10", "p25", "p50", "p75", "p90", "p95"}
    assert set(result.p_drawdown_at_least) == {"0.1", "0.2", "0.3"}
    assert set(result.p_loss_streak_at_least) == {"5", "10", "20"}
    assert set(result.drawdown_percentiles) == set(result.terminal_equity)
    assert len(result.fan.bands) == len(plan.percentiles)
    assert len(result.drawdown_histogram) == len(DRAWDOWN_HISTOGRAM_EDGES) + 1
    assert sum(b.count for b in result.drawdown_histogram) == plan.n_simulations
    assert result.fan.steps[0] == 0 and result.fan.steps[-1] == result.path_length


@pytest.mark.parametrize("method", METHODS)
def test_the_percentiles_and_exceedances_are_monotone(plan, pool, method):
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan, method=method)
    for table in (result.terminal_equity, result.drawdown_percentiles):
        values = [table[f"p{q:g}"].require() for q in plan.percentiles]
        assert values == sorted(values)
    exceed = [result.p_drawdown_at_least[f"{d:g}"].require() for d in plan.drawdown_thresholds]
    assert exceed == sorted(exceed, reverse=True)
    streaks = [result.p_loss_streak_at_least[f"{k:d}"].require() for k in plan.streak_thresholds]
    assert streaks == sorted(streaks, reverse=True)
    assert 0.0 <= result.p_ruin.require() <= 1.0


def test_the_fan_bands_nest_and_start_at_the_starting_equity(plan, pool):
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    bands = {band.percentile: band.equity for band in result.fan.bands}
    for low, high in zip(plan.percentiles, plan.percentiles[1:]):
        assert all(a <= b for a, b in zip(bands[low], bands[high]))
    for band in result.fan.bands:
        assert band.equity[0] == plan.starting_equity


def test_the_fan_step_grid_is_bounded_on_a_long_path(trade):
    pool = [trade(i, 1.0 if i % 2 else -1.0) for i in range(30)]
    plan = MonteCarloPlan.default().replace(
        n_simulations=50, trades_per_simulation=5_000, n_curves_to_store=0
    )
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert len(result.fan.steps) <= FAN_MAX_STEPS
    assert result.fan.steps[-1] == 5_000
    assert list(result.fan.steps) == sorted(set(result.fan.steps))
    assert any("extrapolates the" in note for note in result.notes)


def test_a_fan_must_align_to_its_steps(plan):
    with pytest.raises(ValidationError, match="align to steps"):
        EquityFan(
            steps=(0, 1, 2),
            bands=(FanBand(percentile=50.0, equity=(1.0, 2.0)),),
            n_simulations=plan.n_simulations,
            n_trades=1,
        )


def test_a_fan_must_report_the_same_n_as_its_distribution(plan, pool):
    built = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    with pytest.raises(ValidationError, match="same n"):
        built.replace(fan=built.fan.replace(n_trades=len(pool) + 1))


def test_the_drawdown_histogram_bins_the_signed_depths(trade):
    plan = MonteCarloPlan.default().replace(
        n_simulations=100, risk_fraction=0.5, compound=False,
        trades_per_simulation=6, n_curves_to_store=0,
    )
    result = run_monte_carlo([trade(i, -1.0) for i in range(3)],
                             split_label=SplitPhase.TRAIN, plan=plan)
    deepest = result.drawdown_histogram[0]
    assert deepest.lower is None and deepest.upper == DRAWDOWN_HISTOGRAM_EDGES[0]
    assert deepest.count == 100
    assert deepest.label == "[-inf,-0.5)"


def test_drawdowns_are_negative_fractions_like_the_rest_of_the_system(plan, pool):
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert result.worst_drawdown.require() <= result.median_drawdown.require() <= 0.0
    assert result.worst_drawdown.unit == "fraction"


def test_describe_is_pure_formatting_and_names_the_chain(plan, pool):
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan,
                             method=MonteCarloMethod.REGIME_AWARE)
    lines = result.describe()
    assert lines[0].startswith("LIMITATION: ")
    assert any("regime chain:" in line for line in lines)
    assert any("unobserved, given no state:" in line for line in lines)
    assert PROVENANCE_NOTE in lines


def test_compare_methods_names_no_winner(plan, pool):
    results = compare_methods(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert set(results) == set(METHODS)
    assert all(isinstance(r, MonteCarloDistribution) for r in results.values())
    empty = compare_methods([], split_label=None, plan=plan)
    assert all(isinstance(r, MonteCarloRefusal) for r in empty.values())


# ===========================================================================
# 9. The plan's own refusals
# ===========================================================================


def test_more_simulations_than_the_ceiling_is_refused_by_the_plan():
    """Structural rather than checked at run time: a plan that asks for more
    than section 11 supports cannot be constructed, so no caller can hold
    one. The validator's ValueError reaches callers as a ValidationError."""
    with pytest.raises(ValidationError, match="MAX_SIMULATIONS"):
        MonteCarloPlan.default().replace(n_simulations=MAX_SIMULATIONS + 1)
    MonteCarloPlan.default().replace(n_simulations=MAX_SIMULATIONS)


@pytest.mark.slow
def test_one_hundred_thousand_simulations_are_supported(trade):
    pool = [trade(i, 1.0 if i % 3 else -1.0) for i in range(40)]
    plan = MonteCarloPlan.default().replace(
        n_simulations=MAX_SIMULATIONS, n_curves_to_store=0
    )
    result = run_monte_carlo(pool, split_label=SplitPhase.TRAIN, plan=plan)
    assert result.n_simulations == MAX_SIMULATIONS
    assert sum(b.count for b in result.drawdown_histogram) == MAX_SIMULATIONS


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"percentiles": ()}, "at least one percentile"),
        ({"percentiles": (50.0, 25.0)}, "sorted ascending"),
        ({"percentiles": (0.0, 50.0)}, "strictly between"),
        ({"drawdown_thresholds": (1.5,)}, "fractions in"),
        ({"streak_thresholds": (0,)}, "positive trade counts"),
        ({"risk_fraction": 1.0}, "less than 1"),
        ({"starting_equity": 0.0}, "greater than 0"),
        ({"n_simulations": 0}, "greater than or equal to 1"),
    ],
)
def test_the_plan_refuses_an_incoherent_setting(changes, match):
    with pytest.raises(ValidationError, match=match):
        MonteCarloPlan.default().replace(**changes)


def test_the_plan_resolves_from_the_root_config(config):
    plan = MonteCarloPlan.from_config(config)
    assert plan.method is config.monte_carlo.method
    assert plan.risk_fraction == config.risk.risk_per_trade_pct
    assert plan.min_sample_n == config.analytics.min_sample_n
    assert plan.compound is config.monte_carlo.compound


def test_the_plan_refuses_something_that_is_not_a_config():
    with pytest.raises(TypeError, match="root config"):
        MonteCarloPlan.from_config(object())


def test_the_plan_round_trips_through_its_own_dump():
    plan = MonteCarloPlan.default()
    assert MonteCarloPlan(**plan.to_init_dict()) == plan


def test_the_plan_is_echoed_on_the_result_so_a_figure_is_attributable(plan, pool):
    for trades, label in ((pool, SplitPhase.TRAIN), ([], None)):
        result = run_monte_carlo(trades, split_label=label, plan=plan)
        assert result.plan == plan


# ===========================================================================
# 10. PAPER ONLY, and the module's boundaries
# ===========================================================================

#: Identifiers that would mean this module had reached an execution path.
#: Matched on the AST, never on source text, so a docstring mentioning
#: "order" is not an offence and `_block_indices` is not a false hit.
ORDER_IDENTIFIERS = frozenset(
    {
        "place_order", "submit_order", "send_order", "cancel_order", "order",
        "orders", "broker", "buy", "sell", "execute", "fill", "fills",
        "ExecutionModel", "Ledger", "BacktestLoop",
    }
)

#: Packages a post-hoc resampler has no business importing. It reads finished
#: records; it does not decide, size, fill or route anything.
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {"backtest", "live", "signals", "risk", "features", "regime", "dashboard"}
)


def _engine_tree() -> ast.Module:
    return ast.parse(ENGINE.read_text(encoding="utf-8"), filename=str(ENGINE))


def test_the_engine_names_nothing_that_could_place_an_order():
    offenders = []
    for node in ast.walk(_engine_tree()):
        if isinstance(node, ast.Name) and node.id in ORDER_IDENTIFIERS:
            offenders.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in ORDER_IDENTIFIERS:
            offenders.append((node.lineno, node.attr))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ORDER_IDENTIFIERS:
            offenders.append((node.lineno, node.name))
    assert not offenders, f"execution-shaped identifiers in a resampler: {offenders}"


def test_the_order_guard_is_not_vacuous():
    tree = ast.parse("def f():\n    broker.place_order(1)\n")
    hits = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ORDER_IDENTIFIERS
    ]
    assert hits == ["place_order"]


def test_the_engine_imports_no_decision_or_execution_package():
    offenders = []
    for node in ast.walk(_engine_tree()):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        for module in modules:
            parts = module.split(".")
            if parts[:1] == ["flow_model"] and len(parts) > 1 and parts[1] in FORBIDDEN_IMPORT_ROOTS:
                offenders.append((node.lineno, module))
    assert not offenders, f"a resampler must not import decision code: {offenders}"


def test_undefined_has_exactly_one_representation_in_this_module():
    """`MetricValue` / `Undefined` is it; no second sentinel is introduced."""
    tree = _engine_tree()
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not {"NaN", "UNDEFINED", "MISSING", "SENTINEL"} & assigned
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert not {"Undefined", "MetricValue", "SampleStatus"} & classes


def test_the_engine_reads_no_clock():
    """Determinism: a computation that reads `now` is not reproducible."""
    banned = {"now", "utcnow", "today", "time", "monotonic", "perf_counter"}
    offenders = [
        (node.lineno, node.attr)
        for node in ast.walk(_engine_tree())
        if isinstance(node, ast.Attribute) and node.attr in banned
    ]
    assert not offenders, f"clock reads in a deterministic computation: {offenders}"
