"""Tests for the data-integrity layer.

A corrupted row does not make an average slightly wrong. It makes it wrong
in a direction nobody checked, silently, because once summed it looks
exactly like a clean one. These tests care most about the corruptions that
manufacture an edge rather than merely add noise.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.integrity import (
    IMPLAUSIBLE_GAIN_PCT, MAX_RESOLUTION_DRIFT_MINUTES,
    check_forward_returns, check_positions, usable_forward_returns,
)

NOW = dt.datetime.now(dt.timezone.utc)


def _fr(db, *, event=1, horizon=60, ret=5.0, observed=None, due=None, filled=None,
        price=0.01, mfe=None, mae=None, regime="bull_trend/normal_volatility/deep_liquidity"):
    # When a caller pins `due`, derive `observed` backwards from it. Leaving
    # observed at its default would put it AFTER due and trip the
    # impossible-timestamp check, masking whatever the test meant to probe.
    if due is not None and observed is None:
        observed = due - dt.timedelta(minutes=horizon)
    observed = observed or NOW - dt.timedelta(hours=2)
    due = due if due is not None else observed + dt.timedelta(minutes=horizon)
    db.add(models.ForwardReturn(
        pipeline_event_id=event, token_address=f"M{event}", symbol="T",
        observed_at=observed, score=70.0, price_at_signal=price,
        horizon_minutes=horizon, due_at=due, filled_at=filled,
        return_pct=ret, max_favorable_pct=mfe, max_adverse_pct=mae,
        market_regime=regime,
    ))


# ---------------------------------------------------------------------------
# the corruption that manufactures an edge
# ---------------------------------------------------------------------------

def test_a_row_resolved_before_its_horizon_is_excluded_as_leakage(clean_db):
    """The most damaging corruption available. It does not add noise - it
    measures against a price that had not happened at decision time, which
    invents an edge rather than blurring one."""
    due = NOW - dt.timedelta(minutes=30)
    _fr(clean_db, due=due, filled=due - dt.timedelta(minutes=45))
    clean_db.commit()

    report = check_forward_returns(clean_db)
    assert report.by_code.get("future_data_leakage") == 1
    assert report.clean == 0


def test_a_timestamp_from_the_future_is_excluded(clean_db):
    _fr(clean_db, observed=NOW + dt.timedelta(hours=3))
    clean_db.commit()
    assert check_forward_returns(clean_db).by_code.get("future_timestamp") == 1


def test_a_duplicate_outcome_is_counted_once(clean_db):
    """Two rows for the same candidate and horizon double-weight that
    candidate in every average it appears in."""
    _fr(clean_db, event=7)
    _fr(clean_db, event=7)
    clean_db.commit()

    report = check_forward_returns(clean_db)
    assert report.checked == 2
    assert report.by_code.get("duplicate_outcome") == 1
    assert report.clean == 1


# ---------------------------------------------------------------------------
# the bar is "impossible", not "surprising"
# ---------------------------------------------------------------------------

def test_a_big_loss_is_kept_because_that_is_what_memecoins_do(clean_db):
    """A filter tuned to remove surprising observations removes exactly the
    tail that carries the result."""
    _fr(clean_db, ret=-85.0)
    clean_db.commit()
    assert check_forward_returns(clean_db).clean == 1


def test_an_impossible_move_is_excluded(clean_db):
    _fr(clean_db, ret=IMPLAUSIBLE_GAIN_PCT + 1)
    clean_db.commit()
    assert check_forward_returns(clean_db).by_code.get("implausible_move") == 1


def test_a_non_positive_signal_price_is_excluded(clean_db):
    _fr(clean_db, price=0.0)
    clean_db.commit()
    assert check_forward_returns(clean_db).by_code.get("invalid_price") == 1


def test_a_stale_resolution_is_excluded(clean_db):
    """Measured at the wrong moment is not the same as measured."""
    due = NOW - dt.timedelta(hours=3)
    _fr(clean_db, due=due,
        filled=due + dt.timedelta(minutes=MAX_RESOLUTION_DRIFT_MINUTES + 20))
    clean_db.commit()
    assert check_forward_returns(clean_db).by_code.get("stale_resolution") == 1


def test_a_resolution_slightly_late_is_still_usable(clean_db):
    due = NOW - dt.timedelta(hours=3)
    _fr(clean_db, due=due, filled=due + dt.timedelta(minutes=5))
    clean_db.commit()
    assert check_forward_returns(clean_db).clean == 1


# ---------------------------------------------------------------------------
# path coherence
# ---------------------------------------------------------------------------

def test_an_mfe_below_the_close_is_incoherent(clean_db):
    """The path cannot be inside its own endpoints."""
    _fr(clean_db, ret=20.0, mfe=5.0)
    clean_db.commit()
    assert check_forward_returns(clean_db).by_code.get("incoherent_path") == 1


def test_an_mae_above_the_close_is_incoherent(clean_db):
    _fr(clean_db, ret=-20.0, mae=-5.0)
    clean_db.commit()
    assert check_forward_returns(clean_db).by_code.get("incoherent_path") == 1


def test_a_coherent_path_survives(clean_db):
    _fr(clean_db, ret=5.0, mfe=30.0, mae=-12.0)
    clean_db.commit()
    assert check_forward_returns(clean_db).clean == 1


# ---------------------------------------------------------------------------
# missing regime is a warning, not an exclusion
# ---------------------------------------------------------------------------

def test_a_missing_regime_warns_without_discarding_the_row(clean_db):
    """It is still usable for overall metrics; it just cannot enter a
    per-regime comparison, which is what the promotion gate reads."""
    _fr(clean_db, regime=None)
    clean_db.commit()

    report = check_forward_returns(clean_db)
    assert report.clean == 1
    assert any("no market regime" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# the entry point everything else should use
# ---------------------------------------------------------------------------

def test_usable_rows_exclude_the_corrupted_ones(clean_db):
    """Querying ForwardReturn directly is how a duplicate ends up
    double-weighted in an expectancy nobody re-checks."""
    _fr(clean_db, event=1, ret=5.0)
    _fr(clean_db, event=1, ret=5.0)              # duplicate
    _fr(clean_db, event=2, ret=999_999.0)        # impossible
    _fr(clean_db, event=3, ret=7.0)              # clean
    clean_db.commit()

    usable = usable_forward_returns(clean_db)
    assert len(usable) == 2
    assert all(r.return_pct < 1000 for r in usable)


def test_an_empty_dataset_is_not_reported_as_corrupt(clean_db):
    report = check_forward_returns(clean_db)
    assert report.checked == 0
    assert "No observations" in report.verdict()


def test_a_heavily_corrupted_dataset_says_read_nothing_below(clean_db):
    for i in range(10):
        _fr(clean_db, event=100 + i, ret=5.0)
    for i in range(5):
        _fr(clean_db, event=200 + i, observed=NOW + dt.timedelta(hours=2))
    clean_db.commit()

    report = check_forward_returns(clean_db)
    assert report.exclusion_rate > 0.20
    assert "provisional" in report.verdict()


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------

def _position(db, *, opened, closed, entry=0.01, high=None, low=None, sell_legs=1):
    position = models.Position(
        symbol="IG", token_address="mint-IG", chain="solana",
        qty=0.0, initial_qty=100.0, entry_price=entry,
        stop_loss=entry * 0.8, take_profit=entry * 1.5,
        status=models.PositionStatus.CLOSED.value,
        opened_at=opened, closed_at=closed,
        highest_price_since_entry=high, lowest_price_since_entry=low,
        realized_pnl_usd=1.0, recent_prices=[],
    )
    db.add(position)
    db.flush()
    for _ in range(sell_legs):
        db.add(models.Trade(
            position_id=position.id, symbol="IG", side="sell",
            status=models.TradeStatus.FILLED.value, size_usd=1.0,
            qty=100.0, exit_price=entry, created_at=closed,
        ))
    return position


def test_a_position_closed_before_it_opened_is_excluded(clean_db):
    _position(clean_db, opened=NOW, closed=NOW - dt.timedelta(hours=1))
    clean_db.commit()
    assert check_positions(clean_db).by_code.get("impossible_timestamp") == 1


def test_a_closed_position_with_no_sell_leg_is_an_incomplete_postmortem(clean_db):
    """Its realised return is unknown, so averaging it against real
    numbers would silently treat missing as zero."""
    _position(clean_db, opened=NOW - dt.timedelta(hours=2), closed=NOW, sell_legs=0)
    clean_db.commit()
    assert check_positions(clean_db).by_code.get("incomplete_postmortem") == 1


def test_a_high_water_mark_below_the_low_is_incoherent(clean_db):
    _position(clean_db, opened=NOW - dt.timedelta(hours=2), closed=NOW,
              high=0.005, low=0.02)
    clean_db.commit()
    assert check_positions(clean_db).by_code.get("incoherent_path") == 1


def test_a_clean_position_survives(clean_db):
    _position(clean_db, opened=NOW - dt.timedelta(hours=2), closed=NOW,
              high=0.02, low=0.008)
    clean_db.commit()
    assert check_positions(clean_db).clean == 1
