"""How wrong the cost assumption would have to be.

The champion's verdict turns on a model, not a measurement: the recorded
gross edge is positive and the recorded net edge is negative, and the
whole difference is PAPER_FEE_PCT + PAPER_SPREAD_PCT plus the fill
model's impact and drift. These tests pin the arithmetic that says how
far that model would have to be from the truth.
"""
import pytest

from app import models
from app.analysis import cost_sensitivity as cs


def _leg(*, position_id, side, notional, cost_pct, pnl=None):
    """One filled leg. Sells carry qty/exit_price so _costed_notional
    prices them off the exit, as it does in production."""
    t = models.Trade(
        symbol="COSTCOIN", side=side, status=models.TradeStatus.FILLED.value,
        position_id=position_id, execution_cost_pct=cost_pct, pnl_usd=pnl,
        strategy_version="v-test0001",
    )
    if side == "sell":
        t.qty = 1.0
        t.exit_price = notional
        t.size_usd = notional
    else:
        t.size_usd = notional
        t.entry_price = notional
        t.qty = 1.0
    return t


def _round_trip(position_id, *, notional=100.0, cost_pct=0.005, pnl=0.0):
    """A buy and a sell, both paying cost - which is the point: a round
    trip pays twice, and an analysis that counted one leg would halve the
    apparent sensitivity."""
    return [
        _leg(position_id=position_id, side="buy", notional=notional, cost_pct=cost_pct),
        _leg(position_id=position_id, side="sell", notional=notional,
             cost_pct=cost_pct, pnl=pnl),
    ]


# ---------------------------------------------------------------------------
# both legs are counted
# ---------------------------------------------------------------------------

def test_a_round_trip_pays_cost_on_entry_and_exit():
    """$100 each way at 0.5% is $1.00 of cost, not $0.50."""
    trades = _round_trip(1, notional=100.0, cost_pct=0.005, pnl=-1.0)
    out = cs.analyse(trades)

    assert out.positions == 1
    assert out.total_notional_usd == pytest.approx(200.0)
    assert out.total_cost_usd == pytest.approx(1.0)
    assert out.net_pnl_usd == pytest.approx(-1.0)
    assert out.gross_pnl_usd == pytest.approx(0.0)


def test_the_headline_case_gross_positive_net_negative():
    """The champion's actual shape: the signal earns, execution eats it."""
    trades = []
    for i in range(10):
        # each position grosses +$0.80 but pays $1.00 to trade
        trades += _round_trip(i, notional=100.0, cost_pct=0.005, pnl=-0.20)
    out = cs.analyse(trades)

    assert out.gross_pnl_usd == pytest.approx(8.0)
    assert out.total_cost_usd == pytest.approx(10.0)
    assert out.net_pnl_usd == pytest.approx(-2.0)
    assert out.cost_exceeds_gross_edge is True
    assert out.gross_edge_per_position == pytest.approx(0.80)
    assert out.cost_per_position == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# the break-even rate
# ---------------------------------------------------------------------------

def test_breakeven_rate_is_where_the_record_would_have_made_nothing():
    """$8 of gross over $2,000 of notional breaks even at 0.4% per leg."""
    trades = []
    for i in range(10):
        trades += _round_trip(i, notional=100.0, cost_pct=0.005, pnl=-0.20)
    out = cs.analyse(trades)

    assert out.breakeven_cost_rate == pytest.approx(0.004)
    # and the scenario at that rate really does land on zero
    at_be = next(s for s in out.scenarios
                 if s.cost_rate == pytest.approx(round(0.004, 5)))
    assert at_be.net_pnl_usd == pytest.approx(0.0, abs=1e-9)


def test_a_gross_negative_record_has_no_breakeven_rate():
    """The distinction that matters: "needs free execution" and "cannot be
    fixed by execution at all" are different findings, and collapsing them
    would let a strategy with no edge look like a cost problem."""
    trades = []
    for i in range(5):
        # loses money even before any cost is charged
        trades += _round_trip(i, notional=100.0, cost_pct=0.005, pnl=-3.0)
    out = cs.analyse(trades)

    assert out.gross_pnl_usd < 0
    assert out.breakeven_cost_rate is None
    assert out.cost_exceeds_gross_edge is False   # nothing to eat


# ---------------------------------------------------------------------------
# the profit-factor target
# ---------------------------------------------------------------------------

def test_the_rate_that_would_reach_the_gates_profit_factor():
    """Solved by bisection because profit factor steps rather than glides:
    positions cross zero one at a time as the rate moves."""
    trades = []
    for i in range(6):                      # winners
        trades += _round_trip(i, notional=100.0, cost_pct=0.005, pnl=4.0)
    for i in range(6, 12):                  # losers
        trades += _round_trip(i, notional=100.0, cost_pct=0.005, pnl=-4.0)
    out = cs.analyse(trades)

    assert out.target_pf_cost_rate is not None

    # Checked through the public surface: re-running the analysis with the
    # solved rate charged as the recorded rate must clear the bar. Poking
    # at the solver's internal position objects would test the
    # implementation rather than the answer.
    rate = out.target_pf_cost_rate
    recharged = []
    for i in range(6):
        recharged += _round_trip(i, notional=100.0, cost_pct=rate,
                                 pnl=4.0 + 1.0 - 200.0 * rate)
    for i in range(6, 12):
        recharged += _round_trip(i, notional=100.0, cost_pct=rate,
                                 pnl=-4.0 + 1.0 - 200.0 * rate)
    check = cs.analyse(recharged)
    at_rate = next(s.profit_factor for s in check.scenarios
                   if s.cost_rate == pytest.approx(round(rate, 5)))
    assert at_rate >= cs.TARGET_PROFIT_FACTOR - 1e-6


def test_no_rate_reaches_the_target_when_free_execution_would_not():
    """If the strategy cannot clear 1.30 even at zero cost, say so rather
    than returning an unreachable rate."""
    trades = []
    for i in range(5):
        trades += _round_trip(i, notional=100.0, cost_pct=0.005, pnl=1.0)
    for i in range(5, 15):
        trades += _round_trip(i, notional=100.0, cost_pct=0.005, pnl=-5.0)
    out = cs.analyse(trades)
    assert out.target_pf_cost_rate is None


# ---------------------------------------------------------------------------
# unmeasurable is never zero
# ---------------------------------------------------------------------------

def test_a_leg_with_no_cost_rate_is_unmeasured_not_free():
    """Treating it as free would understate cost - the direction an error
    in a trading simulator must never point."""
    trades = _round_trip(1, notional=100.0, cost_pct=0.005, pnl=-1.0)
    trades.append(_leg(position_id=2, side="sell", notional=100.0,
                       cost_pct=None, pnl=5.0))
    out = cs.analyse(trades)

    assert out.unmeasured_legs == 1
    assert out.positions == 1                      # the costed one only
    assert out.net_pnl_usd == pytest.approx(-1.0)  # the $5 is not counted


def test_a_leg_with_no_position_id_cannot_be_attributed():
    """Costs are per position; a leg that belongs to no position cannot be
    folded into one without inventing the attribution."""
    trades = _round_trip(1, notional=100.0, cost_pct=0.005, pnl=-1.0)
    orphan = _leg(position_id=None, side="sell", notional=100.0,
                  cost_pct=0.005, pnl=2.0)
    trades.append(orphan)
    out = cs.analyse(trades)

    assert out.unmeasured_legs == 1
    assert out.positions == 1


def test_an_open_position_is_not_scored():
    """An entry with no realized exit has paid cost but has no outcome.
    Counting it would make execution look dearer than the finished record."""
    trades = _round_trip(1, notional=100.0, cost_pct=0.005, pnl=-1.0)
    trades.append(_leg(position_id=9, side="buy", notional=100.0, cost_pct=0.005))
    out = cs.analyse(trades)

    assert out.positions == 1
    assert out.total_notional_usd == pytest.approx(200.0)


def test_an_empty_record_reports_nothing_rather_than_dividing_by_zero():
    out = cs.analyse([])
    assert out.positions == 0
    assert out.measured_cost_rate is None
    assert out.breakeven_cost_rate is None
    assert out.target_pf_cost_rate is None
    assert out.scenarios == []
    assert out.gross_edge_per_position is None


def test_unfilled_legs_are_ignored_entirely():
    trades = _round_trip(1, notional=100.0, cost_pct=0.005, pnl=-1.0)
    failed = _leg(position_id=3, side="sell", notional=100.0, cost_pct=0.005, pnl=9.0)
    failed.status = models.TradeStatus.FAILED.value
    trades.append(failed)

    out = cs.analyse(trades)
    assert out.positions == 1
    assert out.unmeasured_legs == 0
