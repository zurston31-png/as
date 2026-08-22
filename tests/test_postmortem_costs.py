"""Execution-cost accounting and observation counts in the post-mortem.

Three defects lived here undetected because tests/test_postmortem.py
covers the path fields (MFE/MAE, capture, hold time) and never asserted a
cost figure or a sample count. Each of these pins one of them.

WHY THESE MATTER MORE THAN THEY LOOK

The whole point of the collection run is that the recorded numbers can be
regressed against each other afterwards. A cost column that is secretly a
function of the trade's return is worse than a missing one: it produces a
confident, wrong finding ("our winners fill better") that survives review
because the column is named `slippage_pct` and looks like slippage.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.postmortem import build_postmortem
from app.database import SessionLocal
from app.exits.manager import MAX_RECENT_PRICE_SAMPLES, record_price_tick

# The paper fill model composes total_cost = impact + spread + fee, all as
# fractions of that leg's own notional (app/execution/fill_model.py).
FEE_RATE = 0.0025
SLIPPAGE_RATE = 0.0075
TOTAL_COST = FEE_RATE + SLIPPAGE_RATE


@pytest.fixture
def db():
    session = SessionLocal()

    def wipe():
        for model in (models.Trade, models.Position):
            for row in session.query(model).all():
                if (getattr(row, "symbol", "") or "").startswith("PMC"):
                    session.delete(row)
        session.commit()

    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


def _round_trip(db, *, entry=1.0, exit_price=1.0, qty=1000.0, symbol="PMCOST",
                exit_legs=1):
    """A closed position whose legs carry exactly the fill model's costs.

    Fees are derived from each leg's own notional rather than hardcoded,
    which is the property the slippage calculation has to respect.
    """
    now = dt.datetime.now(dt.timezone.utc)
    position = models.Position(
        symbol=symbol, token_address=f"mint-{symbol}", chain="solana",
        qty=0.0, initial_qty=qty, entry_price=entry,
        stop_loss=entry * 0.8, take_profit=entry * 1.5,
        status=models.PositionStatus.CLOSED.value,
        opened_at=now - dt.timedelta(minutes=30), closed_at=now,
        close_reason="take profit",
        highest_price_since_entry=max(entry, exit_price),
        lowest_price_since_entry=min(entry, exit_price),
        realized_pnl_usd=(exit_price - entry) * qty,
    )
    db.add(position)
    db.flush()

    db.add(models.Trade(
        position_id=position.id, symbol=symbol, side="buy",
        status=models.TradeStatus.FILLED.value, size_usd=entry * qty,
        qty=qty, entry_price=entry,
        fee_usd=entry * qty * FEE_RATE, execution_cost_pct=TOTAL_COST,
        created_at=now - dt.timedelta(minutes=30),
    ))
    for i in range(exit_legs):
        leg_qty = qty / exit_legs
        db.add(models.Trade(
            position_id=position.id, symbol=symbol, side="sell",
            status=models.TradeStatus.FILLED.value, size_usd=exit_price * leg_qty,
            qty=leg_qty, exit_price=exit_price,
            fee_usd=exit_price * leg_qty * FEE_RATE, execution_cost_pct=TOTAL_COST,
            created_at=now - dt.timedelta(seconds=exit_legs - i),
        ))
    db.commit()
    return position


# ---------------------------------------------------------------------------
# slippage
# ---------------------------------------------------------------------------

def test_slippage_is_the_cost_that_is_not_the_fee(db):
    """The definition. Every leg here paid 1.00% total of which 0.25% was
    fee, so the slippage component is 0.75% - regardless of how many legs
    there were or what the trade returned."""
    pm = build_postmortem(db, _round_trip(db))
    assert pm.slippage_pct == pytest.approx(SLIPPAGE_RATE * 100)


@pytest.mark.parametrize("multiple", [1, 2, 4, 10, 50])
def test_slippage_does_not_track_the_trades_return(db, multiple):
    """The regression, and the reason this file exists.

    The old formula averaged execution cost across legs but subtracted the
    fees of ALL legs over the ENTRY notional. The fee term scales with the
    exit notional and the cost term does not, so the result was a function
    of the trade's return: a flat round trip understated slippage by the
    exit fee, a 10x winner reported -1.75%, and a 50x winner -11.75% - the
    desk paying the trader. Every leg here pays an identical 0.75%, so
    every multiple must report 0.75%.
    """
    position = _round_trip(db, exit_price=float(multiple),
                           symbol=f"PMCX{multiple}")
    pm = build_postmortem(db, position)
    assert pm.slippage_pct == pytest.approx(SLIPPAGE_RATE * 100)
    assert pm.slippage_pct > 0, "execution can cost, it cannot pay"


@pytest.mark.parametrize("exit_legs", [1, 2, 3, 5])
def test_splitting_the_exit_does_not_change_the_slippage_rate(db, exit_legs):
    """Slippage is a rate, so taking the same exit in five pieces at the
    same price pays the same rate five times, not five times the rate."""
    position = _round_trip(db, exit_legs=exit_legs, symbol=f"PMCL{exit_legs}")
    assert build_postmortem(db, position).slippage_pct == pytest.approx(
        SLIPPAGE_RATE * 100
    )


def test_cost_rates_are_weighted_by_notional_not_averaged_per_leg(db):
    """Second round on the same function.

    The first fix removed the return-dependence but still took an
    unweighted mean over legs, so a tiny expensive leg counted as much as
    a large cheap one. Here a $10 scalp-out pays 5% and a $990 exit pays
    0.5%; the position paid $0.50 + $4.95 = $5.45 on $2,000 traded
    ($1,000 in, $1,000 out), so the honest rate is 0.2725%. An unweighted
    mean over the three legs reports 2.0%, off by more than 7x.

    Every leg in the tests above was the same size and the same cost, so
    they passed either way - which is exactly why this one uses legs that
    differ.
    """
    now = dt.datetime.now(dt.timezone.utc)
    position = models.Position(
        symbol="PMCW", token_address="mint-PMCW", chain="solana",
        qty=0.0, initial_qty=1000.0, entry_price=1.0,
        stop_loss=0.8, take_profit=1.5,
        status=models.PositionStatus.CLOSED.value,
        opened_at=now - dt.timedelta(minutes=10), closed_at=now,
        close_reason="take profit",
        highest_price_since_entry=1.0, lowest_price_since_entry=1.0,
        realized_pnl_usd=0.0,
    )
    db.add(position)
    db.flush()

    # entry: $1,000 at 0% cost, so all the cost sits in the two exits
    db.add(models.Trade(
        position_id=position.id, symbol="PMCW", side="buy",
        status=models.TradeStatus.FILLED.value, size_usd=1000.0,
        qty=1000.0, entry_price=1.0, fee_usd=0.0, execution_cost_pct=0.0,
        created_at=now - dt.timedelta(minutes=10),
    ))
    # a $10 scalp at 5%, and a $990 exit at 0.5%
    db.add(models.Trade(
        position_id=position.id, symbol="PMCW", side="sell",
        status=models.TradeStatus.FILLED.value, size_usd=10.0,
        qty=10.0, exit_price=1.0, fee_usd=0.0, execution_cost_pct=0.05,
        created_at=now - dt.timedelta(minutes=5),
    ))
    db.add(models.Trade(
        position_id=position.id, symbol="PMCW", side="sell",
        status=models.TradeStatus.FILLED.value, size_usd=990.0,
        qty=990.0, exit_price=1.0, fee_usd=0.0, execution_cost_pct=0.005,
        created_at=now,
    ))
    db.commit()

    pm = build_postmortem(db, position)
    paid = 0.05 * 10.0 + 0.005 * 990.0        # $5.45
    traded = 1000.0 + 10.0 + 990.0            # $2,000
    assert pm.execution_cost_pct == pytest.approx(paid / traded * 100)

    unweighted = (0.0 + 0.05 + 0.005) / 3 * 100
    assert pm.execution_cost_pct != pytest.approx(unweighted), (
        "cost is still an unweighted mean over legs"
    )


def test_a_leg_with_no_notional_is_dropped_rather_than_called_fee_free(db):
    """A leg with no recorded price cannot have its fee turned into a
    rate. Counting it with a zero fee share would report its entire
    execution cost as slippage and drag the average up, inventing a cost
    that was never measured."""
    position = _round_trip(db, symbol="PMCNULL")
    db.add(models.Trade(
        position_id=position.id, symbol="PMCNULL", side="sell",
        status=models.TradeStatus.FILLED.value, size_usd=0.0,
        qty=0.0, exit_price=None,
        fee_usd=0.10, execution_cost_pct=TOTAL_COST,
        created_at=dt.datetime.now(dt.timezone.utc),
    ))
    db.commit()

    pm = build_postmortem(db, position)
    assert pm.slippage_pct == pytest.approx(SLIPPAGE_RATE * 100), (
        "the unpriced leg was averaged in as if it paid no fee"
    )


def test_slippage_is_none_when_no_leg_recorded_a_cost(db):
    """Unmeasurable, not zero. A live backend that never populated
    execution_cost_pct must not read as a frictionless fill."""
    position = _round_trip(db, symbol="PMCNONE")
    for leg in db.query(models.Trade).filter(
        models.Trade.position_id == position.id
    ).all():
        leg.execution_cost_pct = None
    db.commit()

    pm = build_postmortem(db, position)
    assert pm.slippage_pct is None
    assert pm.execution_cost_pct is None


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_execution_cost_is_reported_in_the_same_unit_as_every_other_pct(db):
    """`Trade.execution_cost_pct` is a FRACTION on the row - see
    app/analysis/trade_analytics.py, and app/dashboard/templates/
    performance.html which multiplies by 100 to display it. The
    post-mortem emitted it raw next to return_pct, max_gain_pct and
    slippage_pct, which are all percents, so a 1.00% execution cost
    rendered as 0.01 beside a return of 5.2."""
    pm = build_postmortem(db, _round_trip(db, exit_price=1.05, symbol="PMCUNIT"))
    assert pm.execution_cost_pct == pytest.approx(TOTAL_COST * 100)

    # The invariant behind it: cost and its slippage component are
    # comparable magnitudes, and both are on the same scale as the return.
    assert pm.slippage_pct < pm.execution_cost_pct
    assert pm.execution_cost_pct == pytest.approx(
        pm.slippage_pct + FEE_RATE * 100
    )


def test_the_dict_form_carries_the_same_units(db):
    """/api/postmortems serves as_dict() straight out, so the unit fix has
    to survive serialisation - that is where a reader actually meets it."""
    pm = build_postmortem(db, _round_trip(db, exit_price=1.05, symbol="PMCDICT"))
    row = pm.as_dict()
    assert row["execution_cost_pct"] == pytest.approx(TOTAL_COST * 100, abs=1e-4)
    assert row["slippage_pct"] == pytest.approx(SLIPPAGE_RATE * 100, abs=1e-4)


# ---------------------------------------------------------------------------
# how many observations the path was drawn from
# ---------------------------------------------------------------------------

def test_the_tick_counter_keeps_counting_past_the_buffer_cap():
    """`recent_prices` is trimmed to MAX_RECENT_PRICE_SAMPLES, but the
    high/low water marks are updated on every tick and never trimmed. So
    the buffer length is not the observation count: it saturates, and a
    position priced 30 times and one priced 300 times both left 30
    samples behind. The excursion bound is only as tight as the real
    count, so the real count is what has to be recorded."""
    position = models.Position(symbol="PMCTICK", entry_price=1.0)
    ticks = MAX_RECENT_PRICE_SAMPLES * 10
    for i in range(ticks):
        record_price_tick(position, 1.0 + i * 0.001)

    assert len(position.recent_prices) == MAX_RECENT_PRICE_SAMPLES
    assert position.price_ticks_observed == ticks


def test_an_unrecorded_tick_count_is_none_rather_than_zero(db):
    """A position closed before the counter existed did not observe zero
    prices - nobody wrote the number down. Reporting 0 would state that
    the MFE came from no observations at all, which is a stronger and
    false claim. CLAUDE.md: unmeasurable is never zero."""
    position = _round_trip(db, symbol="PMCOLD")
    assert position.price_ticks_observed is None

    pm = build_postmortem(db, position)
    assert pm.price_ticks is None
    assert pm.as_dict()["price_ticks"] is None


def test_the_buffer_depth_and_the_tick_count_are_both_reported(db):
    """They answer different questions and the post-mortem needs both:
    the buffer depth bounds what the momentum exits could see, the tick
    count bounds how tight MFE/MAE are."""
    position = _round_trip(db, symbol="PMCBOTH")
    for i in range(MAX_RECENT_PRICE_SAMPLES + 5):
        record_price_tick(position, 1.0 + i * 0.01)
    db.commit()

    pm = build_postmortem(db, position)
    assert pm.samples == MAX_RECENT_PRICE_SAMPLES
    assert pm.price_ticks == MAX_RECENT_PRICE_SAMPLES + 5
    assert pm.price_ticks > pm.samples


# ---------------------------------------------------------------------------
# the same defect in the sibling aggregators
# ---------------------------------------------------------------------------
#
# The weighting bug was found in build_postmortem, but three modules
# aggregate the same `Trade.execution_cost_pct` column and two of them had
# it too. Pinned here, next to the original, so the three cannot drift
# apart again - a reader comparing a post-mortem against the performance
# page must not get two different answers for one book.

def test_fill_audit_weights_its_mean_cost_by_notional():
    """app/analysis/fill_audit.py. A $10 fill at 5% and a $990 fill at
    0.5% is $5.45 on $1,000, i.e. 0.545% - not the 2.75% a flat mean over
    two fills reports."""
    from app.analysis.fill_audit import FillAudit, FillRecord

    audit = FillAudit(fills=[
        FillRecord(trade_id=1, symbol="A", side="buy", execution_cost_pct=0.05,
                   fee_usd=0.0, fill_delay_seconds=1.0, notional_usd=10.0),
        FillRecord(trade_id=2, symbol="A", side="sell", execution_cost_pct=0.005,
                   fee_usd=0.0, fill_delay_seconds=1.0, notional_usd=990.0),
    ])
    paid = 0.05 * 10.0 + 0.005 * 990.0
    assert audit.mean_cost_pct == pytest.approx(paid / 1000.0 * 100)
    assert audit.mean_cost_pct != pytest.approx((0.05 + 0.005) / 2 * 100)


def test_fill_audit_excludes_a_fill_with_no_notional_to_weight_by():
    """A rate with no size behind it cannot be weighted. Counting it in
    the denominator at zero weight would drag the answer toward nothing,
    which understates cost - the direction an error here must never
    point."""
    from app.analysis.fill_audit import FillAudit, FillRecord

    audit = FillAudit(fills=[
        FillRecord(trade_id=1, symbol="A", side="buy", execution_cost_pct=0.01,
                   fee_usd=0.0, fill_delay_seconds=1.0, notional_usd=1000.0),
        FillRecord(trade_id=2, symbol="A", side="sell", execution_cost_pct=0.90,
                   fee_usd=0.0, fill_delay_seconds=1.0, notional_usd=None),
    ])
    assert audit.mean_cost_pct == pytest.approx(1.0)


def test_trade_analytics_rate_agrees_with_its_own_dollar_total():
    """app/analysis/trade_analytics.py. The invariant that makes the two
    figures one measurement: the reported rate times the costed notional
    must reproduce the reported dollar cost. A flat mean breaks it."""
    from app.analysis import trade_analytics as ta

    def leg(size, cost_pct):
        return models.Trade(
            symbol="TAW", side="buy", chain="solana",
            status=models.TradeStatus.FILLED.value,
            size_usd=size, qty=size, entry_price=1.0,
            fee_usd=0.0, execution_cost_pct=cost_pct,
        )

    costs = ta.summarize_costs([leg(10.0, 0.05), leg(990.0, 0.005)])
    assert costs.avg_execution_cost_pct == pytest.approx(
        costs.total_execution_cost_usd / 1000.0
    )
    assert costs.avg_execution_cost_pct != pytest.approx((0.05 + 0.005) / 2)
