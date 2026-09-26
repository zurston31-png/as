"""The report can ask the other Monte Carlo question, and can be told
about a backtest - but only a real one.

TWO GAPS THIS CLOSES

1. app/analysis/monte_carlo.py implements two modes. BOOTSTRAP resamples
   with replacement and answers "what range of OUTCOMES is consistent with
   this edge". SHUFFLE reorders the exact trades, holding the total fixed
   by construction, and answers "given this edge, how bad could the RIDE
   have been" - the survivability question. Only bootstrap was reachable,
   so the second question could not be asked at all.

2. The out-of-sample and walk-forward criteria had no route into the gate.
   They read "no analysis run yet" permanently, however many backtests an
   operator ran. Now a run can supply them - and a run on synthetic
   candles cannot, which is the property most worth pinning.
"""
import datetime as dt
import json

import pytest

from app import models
from app.analysis.backtest_evidence import as_payload, load
from app.analysis.report import build_performance_report
from app.database import SessionLocal

NOW = dt.datetime(2026, 9, 4, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture()
def clean_db():
    def wipe(session):
        for model in (models.Trade, models.RiskEvent, models.Signal, models.Position):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _trades(db, pnls):
    for i, pnl in enumerate(pnls):
        db.add(models.Trade(
            symbol="MCCOIN", side="sell", status=models.TradeStatus.FILLED.value,
            size_usd=100.0, pnl_usd=pnl, pnl_pct=pnl,
            opened_at=NOW - dt.timedelta(hours=1), closed_at=NOW,
            fee_usd=0.25, execution_cost_pct=0.004, position_id=1000 + i,
            strategy_version="v-mctest",
        ))
    db.commit()


def test_shuffle_mode_is_reachable_and_holds_the_total_fixed(clean_db):
    """Shuffle reorders without replacement, so every path ends at the same
    total. That invariant is what makes it a PATH-risk measure, and it is
    the cheapest way to prove the mode was actually used rather than
    silently falling back to bootstrap."""
    _trades(clean_db, [10.0, -4.0, 6.0, -2.0, 8.0, -5.0, 3.0, -1.0])

    report = build_performance_report(clean_db, monte_carlo_mode="shuffle")
    mc = report.monte_carlo
    assert mc is not None
    assert mc.mode == "shuffle"
    # sum is +15.0; every ordering must land there
    assert mc.p05_final_pnl == pytest.approx(15.0)
    assert mc.p95_final_pnl == pytest.approx(15.0)


def test_bootstrap_remains_the_default(clean_db):
    """Changing the default would silently redefine every historical
    report, so the default is pinned."""
    _trades(clean_db, [10.0, -4.0, 6.0, -2.0])
    assert build_performance_report(clean_db).monte_carlo.mode == "bootstrap"


def test_bootstrap_spreads_the_total_where_shuffle_does_not(clean_db):
    """The two modes must actually differ - a mode argument that changed
    nothing would be worse than none."""
    _trades(clean_db, [20.0, -15.0, 12.0, -9.0, 7.0, -3.0, 5.0, -11.0])
    boot = build_performance_report(clean_db, monte_carlo_mode="bootstrap").monte_carlo
    assert boot.p95_final_pnl > boot.p05_final_pnl


def test_without_evidence_the_two_criteria_stay_unmeasured(clean_db):
    """Absent a backtest they must read as not-run, never as a failure and
    never as a pass."""
    _trades(clean_db, [5.0, -2.0])
    v = build_performance_report(clean_db).validation
    by_name = {c.name: c for c in v.criteria}
    assert "no out-of-sample test run yet" in by_name["out-of-sample"].detail
    assert "no walk-forward analysis run yet" in by_name["walk-forward"].detail


def test_real_evidence_reaches_the_gate(clean_db, tmp_path):
    """The whole point: a real backtest must be able to answer them."""
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(as_payload(
        out_of_sample_trades=40, out_of_sample_profitable=True,
        walk_forward_windows=3, walk_forward_profitable_windows=3,
        data_source="csv", symbol="WIF", timeframe="15m", candles=2400,
    )))
    evidence, _ = load(p)
    _trades(clean_db, [5.0, -2.0])

    v = build_performance_report(clean_db, backtest_evidence=evidence).validation
    by_name = {c.name: c for c in v.criteria}
    assert "no out-of-sample test run yet" not in by_name["out-of-sample"].detail
    assert "no walk-forward analysis run yet" not in by_name["walk-forward"].detail


def test_synthetic_evidence_never_reaches_the_gate(clean_db, tmp_path):
    """The hazard this feature creates, pinned end to end.

    These are the real figures a default `run_backtest.py --walk-forward`
    produced on this machine: 12 profitable out-of-sample trades and 3 of 3
    profitable windows. Admitted, they would flip both criteria green on a
    market that never existed.
    """
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(as_payload(
        out_of_sample_trades=12, out_of_sample_profitable=True,
        walk_forward_windows=3, walk_forward_profitable_windows=3,
        data_source="synthetic", symbol="TESTCOIN", timeframe="15m", candles=2400,
    )))
    evidence, message = load(p)
    assert evidence is None, "synthetic evidence must not survive loading"
    assert "SYNTHETIC" in message

    _trades(clean_db, [5.0, -2.0])
    v = build_performance_report(clean_db, backtest_evidence=evidence).validation
    by_name = {c.name: c for c in v.criteria}
    assert "no out-of-sample test run yet" in by_name["out-of-sample"].detail
    assert "no walk-forward analysis run yet" in by_name["walk-forward"].detail
