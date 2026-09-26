"""The unit convention of every field derived from `execution_cost_pct`.

Three modules aggregate that column and they DO NOT agree on units:

    Trade.execution_cost_pct              FRACTION   0.01 means 1%
    trade_analytics.avg_execution_cost_pct FRACTION  0.01 means 1%
    fill_audit.mean_cost_pct              PERCENT    1.0  means 1%
    postmortem.execution_cost_pct         PERCENT    1.0  means 1%
    postmortem.slippage_pct               PERCENT    1.0  means 1%

Two fields with the same `_pct` suffix, describing the same cost on the
same trade, differ by 100x. That is not a defect today - every render
site matches its own module's convention, checked one by one:

    performance.html:131            x100  (fraction -> percent)
    scripts/performance_report.py   x100  (fraction -> percent)
    fill_audit.summary/as_dict      raw   (already percent)
    scripts/research.py             raw   (already percent)
    /api/postmortems                raw   (already percent)

It is a trap, and it is the trap that already sprang once: the
post-mortem used to serve the raw fraction next to real percents, so
`execution_cost_pct: 0.0087` sat beside `return_pct: 5.2` on the same
JSON object.

The units are NOT unified here. Changing trade_analytics to percent
means editing two correct render sites, and rewriting working display
code for naming tidiness is how a correct number becomes a wrong one.
Instead each convention is pinned, so the mismatch is documented, locked,
and cannot drift silently - and whoever unifies them later has a test
that tells them every place to change.
"""
import pytest

from app import models
from app.analysis import trade_analytics as ta
from app.analysis.fill_audit import FillAudit, FillRecord

# One percent, expressed the way the database stores it.
ONE_PERCENT_AS_STORED = 0.01
NOTIONAL = 1000.0


def _leg(cost_pct=ONE_PERCENT_AS_STORED, notional=NOTIONAL):
    return models.Trade(
        symbol="UNIT", side="buy", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=notional, qty=notional, entry_price=1.0,
        fee_usd=0.0, execution_cost_pct=cost_pct,
    )


def _fill(cost_pct=ONE_PERCENT_AS_STORED, notional=NOTIONAL):
    return FillRecord(
        trade_id=1, symbol="UNIT", side="buy", execution_cost_pct=cost_pct,
        fee_usd=0.0, fill_delay_seconds=1.0, notional_usd=notional,
    )


def test_the_stored_column_is_a_fraction():
    """The anchor. Everything else is stated relative to this."""
    leg = _leg()
    assert leg.execution_cost_pct == 0.01
    # A 1% cost on $1,000 is $10, which is the arithmetic every consumer
    # has to agree with.
    assert leg.execution_cost_pct * NOTIONAL == pytest.approx(10.0)


def test_trade_analytics_reports_a_fraction():
    """Its two render sites both multiply by 100. If this ever becomes a
    percent, app/dashboard/templates/performance.html:131 and
    scripts/performance_report.py must change in the same commit or both
    will show 100x."""
    costs = ta.summarize_costs([_leg()])
    assert costs.avg_execution_cost_pct == pytest.approx(0.01)
    assert costs.total_execution_cost_usd == pytest.approx(10.0)
    # The invariant tying the rate to the dollars, in fraction units.
    assert costs.avg_execution_cost_pct * NOTIONAL == pytest.approx(
        costs.total_execution_cost_usd
    )


def test_fill_audit_reports_a_percent():
    """Its render sites print it raw with a % sign."""
    assert FillAudit(fills=[_fill()]).mean_cost_pct == pytest.approx(1.0)


def test_the_postmortem_reports_a_percent(tmp_path):
    """Pinned in tests/test_postmortem_costs.py against a real position;
    restated here so all four conventions sit in one place."""
    from app.analysis.postmortem import PostMortem

    fields = {f: None for f in PostMortem.__dataclass_fields__}
    fields.update(
        position_id=1, symbol="UNIT", realized_pnl_usd=0.0, fees_usd=0.0,
        samples=0, price_ticks=None,
        execution_cost_pct=1.0, slippage_pct=0.75, return_pct=5.2,
    )
    pm = PostMortem(**fields)
    row = pm.as_dict()

    # The property that failed before: every _pct on one record is one
    # unit, so they can be read side by side.
    assert row["execution_cost_pct"] == pytest.approx(1.0)
    assert row["slippage_pct"] == pytest.approx(0.75)
    assert row["return_pct"] == pytest.approx(5.2)


def test_the_two_sibling_aggregators_still_disagree_by_exactly_100x():
    """The mismatch itself, asserted rather than left implicit.

    If someone unifies the units, this test fails and points them at the
    module docstring listing every render site that has to move with it.
    A failure here is a prompt to update this file, not a defect.
    """
    fraction = ta.summarize_costs([_leg()]).avg_execution_cost_pct
    percent = FillAudit(fills=[_fill()]).mean_cost_pct
    assert percent == pytest.approx(fraction * 100), (
        "trade_analytics and fill_audit no longer differ by exactly 100x - "
        "if that was deliberate, update tests/test_cost_units.py and every "
        "render site its docstring lists"
    )
