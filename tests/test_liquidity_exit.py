"""Tests for the liquidity-drop exit.

This is the only exit that can fire while the price still looks healthy,
so the tests care about both directions of error: failing to close a
position whose pool is being drained, and closing positions because the
data feed went quiet for a tick.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.exits.manager import evaluate_liquidity, record_liquidity_tick


def _position(entry_liquidity=200_000.0):
    return models.Position(
        symbol="LQ", token_address="mint-LQ", chain="solana",
        qty=1000.0, entry_price=0.01, stop_loss=0.008, take_profit=0.015,
        status=models.PositionStatus.OPEN.value,
        opened_at=dt.datetime.now(dt.timezone.utc),
        liquidity_at_entry_usd=entry_liquidity,
    )


def test_a_drained_pool_closes_the_position():
    action = evaluate_liquidity(_position(200_000.0), 80_000.0)   # -60%
    assert action.kind == "full"
    assert "being drained" in action.reason


def test_a_softer_drop_trims_instead_of_dumping():
    """Halfway out keeps some upside if the drop was a large holder
    rotating rather than the pool being pulled."""
    action = evaluate_liquidity(_position(200_000.0), 130_000.0)  # -35%
    assert action.kind == "partial"
    assert action.fraction == 0.5


def test_normal_fluctuation_does_nothing():
    assert evaluate_liquidity(_position(200_000.0), 190_000.0).kind == "none"


def test_a_missing_reading_is_not_a_drop():
    """One quiet tick from the price feed would otherwise market-sell every
    open position at once, turning a provider hiccup into a portfolio
    event. Absence of data is not evidence of a drain."""
    assert evaluate_liquidity(_position(200_000.0), None).kind == "none"
    assert evaluate_liquidity(_position(200_000.0), 0.0).kind == "none"


def test_a_pool_under_the_hard_floor_closes_whatever_the_ratio_says():
    """A token that started thin and got thinner is dangerous even when the
    percentage drop looks survivable - there is simply not enough depth
    left to exit into."""
    action = evaluate_liquidity(_position(6_000.0), 4_000.0)      # only -33%
    assert action.kind == "full"
    assert "floor" in action.reason


def test_no_entry_baseline_means_no_ratio_judgement():
    """A position opened before depth was being recorded has nothing to
    compare against. It still gets the absolute floor check, but must not
    be closed on a ratio computed from a missing baseline."""
    position = _position(None)
    assert evaluate_liquidity(position, 150_000.0).kind == "none"
    assert evaluate_liquidity(position, 1_000.0).kind == "full", "the floor still applies"


def test_the_exit_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "LIQUIDITY_EXIT_ENABLED", False)
    assert evaluate_liquidity(_position(200_000.0), 10_000.0).kind == "none"


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

def test_the_first_reading_seeds_the_baseline():
    """Positions opened before this feature existed would otherwise stay
    permanently unassessable."""
    position = _position(None)
    record_liquidity_tick(position, 120_000.0)
    assert position.liquidity_at_entry_usd == 120_000.0
    assert position.lowest_liquidity_usd == 120_000.0


def test_the_baseline_is_not_overwritten_by_later_readings():
    """If entry depth drifted upward with the pool, a 60% drop from the
    latest reading would never look like a 60% drop from entry."""
    position = _position(200_000.0)
    for value in (180_000.0, 250_000.0, 90_000.0):
        record_liquidity_tick(position, value)
    assert position.liquidity_at_entry_usd == 200_000.0
    assert position.lowest_liquidity_usd == 90_000.0


def test_a_missing_reading_does_not_disturb_the_low_water_mark():
    position = _position(200_000.0)
    record_liquidity_tick(position, 100_000.0)
    record_liquidity_tick(position, None)
    record_liquidity_tick(position, 0.0)
    assert position.lowest_liquidity_usd == 100_000.0
