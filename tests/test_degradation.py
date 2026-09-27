"""Tests for app/analysis/degradation.py.

Two failure modes matter equally. Missing a real decay wastes weeks of
paper trading on a strategy that has already stopped working. Firing on
noise is worse, because an alert that cries wolf gets muted, and then the
real one is missed too.

So there are tests for both directions, plus one asserting the module
cannot change anything - a degradation detector wired to a threshold is an
automatic weakening mechanism wearing a diagnostic label.
"""
import datetime as dt

import pytest

from app import models
from app.analysis import degradation
from app.analysis.degradation import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    MIN_BASELINE,
    MIN_RECENT,
    MetricShift,
    _drawdown,
    build_degradation,
)
from app.database import SessionLocal

NOW = dt.datetime(2026, 5, 1, 12, 0, tzinfo=dt.timezone.utc)
VERSION = "v-degrade"


@pytest.fixture
def db():
    session = SessionLocal()

    def wipe():
        for model in (models.Trade, models.Position, models.Signal):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


def trade(db, *, i, return_pct, regime="bull/normal/deep_liquidity",
          version=VERSION, slippage=0.002, mfe=None, mae=None):
    """One closed paper trade, with the path fields the analysis reads.

    Built through the real relationships rather than by setting fields
    directly: the postmortem derives return from the size-weighted exit
    legs and slippage from execution cost minus the fee share, so a
    shortcut here would test a different calculation than production uses.
    """
    at = NOW + dt.timedelta(hours=i)
    signal = models.Signal(
        received_at=at, symbol="T", token_address=f"Mint{i}", chain="solana",
        signal_type="buy", price=1.0, strategy_version=version,
    )
    db.add(signal)
    db.flush()

    entry, exit_price = 1.0, 1.0 * (1 + return_pct / 100)
    entry_leg = models.Trade(
        signal_id=signal.id, symbol="T", token_address=f"Mint{i}", chain="solana",
        side="buy", qty=100.0, entry_price=entry, size_usd=100.0, mode="paper",
        status="filled", created_at=at,
        # Fee zero so the postmortem's "cost minus fee share" reduces to
        # exactly the slippage this test is varying.
        fee_usd=0.0, execution_cost_pct=slippage,
    )
    db.add(entry_leg)
    db.flush()

    position = models.Position(
        symbol="T", token_address=f"Mint{i}", chain="solana", qty=0.0,
        initial_qty=100.0, entry_price=entry, stop_loss=0.85, take_profit=1.3,
        status=models.PositionStatus.CLOSED.value, mode="paper",
        entry_trade_id=entry_leg.id, opened_at=at,
        closed_at=at + dt.timedelta(hours=1), market_regime=regime,
        highest_price_since_entry=entry * (1 + (mfe if mfe is not None else max(return_pct, 1)) / 100),
        lowest_price_since_entry=entry * (1 + (mae if mae is not None else min(return_pct, -1)) / 100),
    )
    db.add(position)
    db.flush()

    entry_leg.position_id = position.id
    db.add(models.Trade(
        signal_id=signal.id, symbol="T", token_address=f"Mint{i}", chain="solana",
        side="sell", qty=100.0, exit_price=exit_price,
        size_usd=100.0 * (1 + return_pct / 100), mode="paper", status="filled",
        created_at=at + dt.timedelta(hours=1), position_id=position.id,
        fee_usd=0.0, execution_cost_pct=slippage,
    ))
    db.flush()
    return position


def named(report, name):
    return next(s for s in report.shifts if s.name == name)


# ---------------------------------------------------------------------------
# refusing to conclude
# ---------------------------------------------------------------------------

def test_a_short_history_produces_no_verdict(db):
    for i in range(10):
        trade(db, i=i, return_pct=5.0)
    db.commit()

    report = build_degradation(db, strategy_version=VERSION)
    assert report.comparable is False
    assert "INSUFFICIENT_DATA" in report.verdict()
    assert report.shifts == []


def test_versions_are_never_pooled(db):
    """A threshold that moved mid-run makes the two halves different
    strategies, and comparing them measures the edit rather than decay."""
    for i in range(MIN_BASELINE + MIN_RECENT + 10):
        trade(db, i=i, return_pct=5.0, version="v-other")
    db.commit()

    report = build_degradation(db, strategy_version=VERSION)
    assert report.total_trades == 0
    assert report.comparable is False


# ---------------------------------------------------------------------------
# the finding
# ---------------------------------------------------------------------------

def test_a_collapse_in_expectancy_is_detected(db):
    for i in range(MIN_BASELINE + 20):
        trade(db, i=i, return_pct=8.0 + (i % 5))
    for i in range(30):
        trade(db, i=200 + i, return_pct=-9.0 + (i % 5))
    db.commit()

    report = build_degradation(db, strategy_version=VERSION)
    assert report.comparable is True
    expectancy = named(report, "expectancy")
    assert expectancy.degraded is True
    assert expectancy.delta < 0
    assert "DEGRADATION" in report.verdict()


def test_the_path_metrics_move_before_the_result_does(db):
    """The point of watching MFE and MAE: a strategy can hold its
    expectancy for weeks while entries stop reaching as far and take more
    heat getting there. By the time the mean moves, the change is old."""
    for i in range(MIN_BASELINE + 20):
        trade(db, i=i, return_pct=4.0, mfe=30.0, mae=-3.0)
    for i in range(30):
        # Same result, much worse path.
        trade(db, i=200 + i, return_pct=4.0, mfe=6.0, mae=-25.0)
    db.commit()

    report = build_degradation(db, strategy_version=VERSION)
    assert named(report, "expectancy").degraded is False
    assert named(report, "MFE (peak reached)").degraded is True
    assert named(report, "MAE (heat taken)").degraded is True


def test_worsening_slippage_is_detected_in_the_right_direction(db):
    """Slippage is the one metric where a bigger number is worse. Getting
    the direction wrong would report improving fills as decay."""
    for i in range(MIN_BASELINE + 20):
        trade(db, i=i, return_pct=5.0, slippage=0.001)
    for i in range(30):
        trade(db, i=200 + i, return_pct=5.0, slippage=0.02)
    db.commit()

    shift = named(build_degradation(db, strategy_version=VERSION), "slippage")
    assert shift.direction == LOWER_IS_BETTER
    assert shift.degraded is True


def test_improving_slippage_is_never_called_degradation(db):
    for i in range(MIN_BASELINE + 20):
        trade(db, i=i, return_pct=5.0, slippage=0.02)
    for i in range(30):
        trade(db, i=200 + i, return_pct=5.0, slippage=0.001)
    db.commit()

    shift = named(build_degradation(db, strategy_version=VERSION), "slippage")
    assert shift.worse is False
    assert shift.degraded is False
    assert shift.p_value is None      # not even resampled - nothing to test


# ---------------------------------------------------------------------------
# not firing on noise
# ---------------------------------------------------------------------------

def test_a_steady_strategy_reports_no_degradation(db):
    """An alert that cries wolf gets muted, and then the real one is
    missed too."""
    for i in range(MIN_BASELINE + 50):
        trade(db, i=i, return_pct=5.0 + (i % 7) - 3)
    db.commit()

    report = build_degradation(db, strategy_version=VERSION)
    assert report.degraded == []
    assert "No sign of degradation" in report.verdict() or "within noise" in report.verdict()


def test_a_small_drift_is_reported_as_drift_not_degradation(db):
    """Moving the wrong way is not the same as having moved. The report
    distinguishes them so a reader is not told to act on a wobble."""
    shift = MetricShift(
        name="expectancy", direction=HIGHER_IS_BETTER,
        baseline=5.0, recent=4.0, baseline_n=40, recent_n=30, p_value=0.44,
    )
    assert shift.worse is True          # material: 1.0 against a 0.25 floor
    assert shift.degraded is False      # but resampling does not separate it


def test_an_immaterial_shift_is_not_even_tested():
    """The bug this floor closes: for a metric that barely moves, the
    bootstrap can return an emphatic p=0.000 about a meaningless amount. A
    slippage figure constant to fifteen decimal places still differs in the
    sixteenth, and resampling a constant reproduces that difference every
    single time - which was reported as DEGRADED."""
    shift = MetricShift(
        name="slippage", direction=LOWER_IS_BETTER,
        baseline=0.19999999999999993, recent=0.2000000000000001,
        baseline_n=50, recent_n=30, p_value=0.0,
    )
    assert shift.worse is False
    assert shift.degraded is False


def test_materiality_scales_with_the_metric(db):
    """Relative rather than absolute, because these metrics live on
    completely different scales - a 0.3-point move is nothing in MFE and
    enormous in slippage."""
    big = MetricShift("MFE", HIGHER_IS_BETTER, baseline=40.0, recent=39.7,
                      baseline_n=40, recent_n=30)
    small = MetricShift("slippage", LOWER_IS_BETTER, baseline=0.20, recent=0.50,
                        baseline_n=40, recent_n=30)
    assert big.worse is False           # 0.3 against a 2.0 floor
    assert small.worse is True          # 0.3 against a 0.01 floor


# ---------------------------------------------------------------------------
# by condition
# ---------------------------------------------------------------------------

def test_degradation_confined_to_one_regime_is_located(db):
    """"It stopped working" and "it stopped working in chop" call for
    different responses, and only the second is visible per axis."""
    for i in range(MIN_BASELINE + 20):
        trade(db, i=i, return_pct=8.0, regime="bull/normal/deep_liquidity")
    for i in range(15):
        trade(db, i=200 + i, return_pct=8.0, regime="bull/normal/deep_liquidity")
    for i in range(15):
        trade(db, i=300 + i, return_pct=-12.0, regime="bull/normal/thin_liquidity")
    db.commit()

    report = build_degradation(db, strategy_version=VERSION, recent_trades=30)
    thin = [g for g in report.groups if g.group == "thin_liquidity"]
    assert thin and thin[0].recent is not None and thin[0].recent < 0


def test_the_regime_label_is_split_into_axes(db):
    """The combined label fragments the sample into a dozen groups of three
    trades, and a dozen groups of three trades is a random-number
    generator."""
    assert degradation._regime_axes("bull/high_volatility/thin_liquidity") == [
        ("trend", "bull"), ("volatility", "high_volatility"), ("liquidity", "thin_liquidity"),
    ]
    assert degradation._regime_axes(None) == []


# ---------------------------------------------------------------------------
# drawdown, and the safety property
# ---------------------------------------------------------------------------

def test_drawdown_is_peak_to_trough_and_signed_negative():
    """One sign convention across the table is what stops a reader
    misjudging which way is bad."""
    assert _drawdown([10.0, -4.0, -3.0, 5.0]) == pytest.approx(-7.0)
    assert _drawdown([5.0, 5.0]) == pytest.approx(0.0)
    assert _drawdown([]) is None


def test_drawdown_carries_no_p_value(db):
    """It is a property of the SEQUENCE, so it has no per-trade sample to
    resample. Better to report it without one than to invent one."""
    for i in range(MIN_BASELINE + 40):
        trade(db, i=i, return_pct=5.0 - (i % 9))
    db.commit()

    assert named(build_degradation(db, strategy_version=VERSION), "max drawdown").p_value is None


def test_the_module_cannot_change_anything():
    """A degradation detector wired to a threshold is an automatic
    weakening mechanism wearing a diagnostic label."""
    import pathlib

    body = "\n".join(
        line for line in pathlib.Path("app/analysis/degradation.py").read_text().splitlines()
        if not line.strip().startswith("#")
    )
    for forbidden in ("db.add(", "db.commit(", "db.delete(", "setattr(settings",
                      "risk_manager", "changelog", "kill_switch"):
        assert forbidden not in body
