"""Unit tests for app/exits/manager.py - smart exits layered on the fixed
stop-loss/take-profit set at entry.

Positions here are plain in-memory `models.Position` instances, never added
to a session. That's deliberate: ExitManager.evaluate() only ever reads and
mutates the object passed to it, so testing it doesn't need a database - and
it exercises the same "unflushed defaults are None, not their DB default"
behavior real code sees when a freshly-queried position from a real session
happens to have never round-tripped some field.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.exits.manager import ExitManager, record_price_tick


def _position(**overrides) -> models.Position:
    defaults = dict(
        symbol="COIN",
        token_address="COINaddr",
        qty=100.0,
        initial_qty=100.0,
        entry_price=1.0,
        stop_loss=0.85,
        take_profit=1.30,
        status=models.PositionStatus.OPEN.value,
        opened_at=dt.datetime.now(dt.timezone.utc),
    )
    defaults.update(overrides)
    return models.Position(**defaults)


def _uncapped_exit_settings():
    """Disable every exit type so a test can enable exactly the one it's
    checking, avoiding cross-talk (e.g. a trailing-stop test tripping the
    unrelated momentum-loss rule at the same price point)."""
    names = [
        "TRAILING_STOP_ENABLED", "BREAK_EVEN_ENABLED", "PARTIAL_TAKE_PROFIT_ENABLED",
        "MOMENTUM_EXIT_ENABLED", "TREND_REVERSAL_EXIT_ENABLED", "TIME_BASED_EXIT_ENABLED",
    ]
    originals = {name: getattr(settings, name) for name in names}
    for name in names:
        setattr(settings, name, False)
    return originals


def _restore(originals):
    for name, value in originals.items():
        setattr(settings, name, value)


# ---------------------------------------------------------------------------
# hard stop-loss / take-profit - always checked first
# ---------------------------------------------------------------------------

def test_hard_stop_loss_triggers_full_exit():
    originals = _uncapped_exit_settings()
    try:
        rm = ExitManager()
        pos = _position(stop_loss=0.85)
        action = rm.evaluate(pos, current_price=0.80)
        assert action.kind == "full"
        assert "stop-loss" in action.reason
    finally:
        _restore(originals)


def test_hard_take_profit_triggers_full_exit():
    originals = _uncapped_exit_settings()
    try:
        rm = ExitManager()
        pos = _position(take_profit=1.30)
        action = rm.evaluate(pos, current_price=1.35)
        assert action.kind == "full"
        assert "take-profit" in action.reason
    finally:
        _restore(originals)


def test_price_between_stop_and_target_with_everything_disabled_is_a_no_op():
    originals = _uncapped_exit_settings()
    try:
        rm = ExitManager()
        pos = _position()
        action = rm.evaluate(pos, current_price=1.05)
        assert action.kind == "none"
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# break-even stop
# ---------------------------------------------------------------------------

def test_break_even_moves_stop_up_once_triggered_and_does_not_itself_exit():
    originals = _uncapped_exit_settings()
    settings.BREAK_EVEN_ENABLED = True
    settings.BREAK_EVEN_TRIGGER_PCT = 0.10
    settings.BREAK_EVEN_BUFFER_PCT = 0.01
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.85)
        action = rm.evaluate(pos, current_price=1.12)  # +12%, past the 10% trigger
        assert action.kind == "none"
        assert pos.stop_loss == pytest.approx(1.01)  # entry * 1.01
        assert pos.break_even_applied is True
    finally:
        _restore(originals)


def test_break_even_does_not_fire_below_its_trigger():
    originals = _uncapped_exit_settings()
    settings.BREAK_EVEN_ENABLED = True
    settings.BREAK_EVEN_TRIGGER_PCT = 0.10
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.85)
        rm.evaluate(pos, current_price=1.05)  # only +5%
        assert pos.stop_loss == 0.85
        assert not pos.break_even_applied
    finally:
        _restore(originals)


def test_break_even_never_lowers_a_stop_already_better_than_break_even():
    originals = _uncapped_exit_settings()
    settings.BREAK_EVEN_ENABLED = True
    settings.BREAK_EVEN_TRIGGER_PCT = 0.10
    settings.BREAK_EVEN_BUFFER_PCT = 0.01
    try:
        rm = ExitManager()
        # Stop already sitting above what break-even would set it to.
        pos = _position(entry_price=1.0, stop_loss=1.05)
        rm.evaluate(pos, current_price=1.20)
        assert pos.stop_loss == 1.05
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# trailing stop
# ---------------------------------------------------------------------------

def test_trailing_stop_does_not_activate_before_the_activation_threshold():
    originals = _uncapped_exit_settings()
    settings.TRAILING_STOP_ENABLED = True
    settings.TRAILING_STOP_ACTIVATION_PCT = 0.15
    settings.TRAILING_STOP_DISTANCE_PCT = 0.10
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.85)
        rm.evaluate(pos, current_price=1.10)  # only +10%, below the 15% activation
        assert not pos.trailing_stop_active
        assert pos.stop_loss == 0.85
    finally:
        _restore(originals)


def test_trailing_stop_activates_and_ratchets_up_behind_the_peak():
    originals = _uncapped_exit_settings()
    settings.TRAILING_STOP_ENABLED = True
    settings.TRAILING_STOP_ACTIVATION_PCT = 0.15
    settings.TRAILING_STOP_DISTANCE_PCT = 0.10
    try:
        rm = ExitManager()
        # take_profit set well above every price used below, so the hard
        # take-profit check (which always wins) never gets in the way of
        # observing the trailing stop's own ratcheting behavior.
        pos = _position(entry_price=1.0, stop_loss=0.85, take_profit=5.0)
        rm.evaluate(pos, current_price=1.20)  # +20%, past activation; peak=1.20
        assert pos.trailing_stop_active
        assert pos.stop_loss == pytest.approx(1.20 * 0.90)

        # Price rises further -> stop ratchets up with the new peak.
        rm.evaluate(pos, current_price=1.40)
        assert pos.stop_loss == pytest.approx(1.40 * 0.90)

        # Price pulls back but stays above the trailing stop -> stop must
        # NOT loosen back down with the lower price.
        rm.evaluate(pos, current_price=1.30)
        assert pos.stop_loss == pytest.approx(1.40 * 0.90)
    finally:
        _restore(originals)


def test_trailing_stop_breach_triggers_full_exit():
    originals = _uncapped_exit_settings()
    settings.TRAILING_STOP_ENABLED = True
    settings.TRAILING_STOP_ACTIVATION_PCT = 0.15
    settings.TRAILING_STOP_DISTANCE_PCT = 0.10
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.85, take_profit=5.0)
        rm.evaluate(pos, current_price=1.20)  # activates trailing, stop -> 1.08
        action = rm.evaluate(pos, current_price=1.07)  # breaches the trailing stop
        assert action.kind == "full"
        assert "trailing stop" in action.reason
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# partial profit-take
# ---------------------------------------------------------------------------

def test_partial_take_profit_fires_once_at_the_configured_trigger():
    originals = _uncapped_exit_settings()
    settings.PARTIAL_TAKE_PROFIT_ENABLED = True
    settings.PARTIAL_TAKE_PROFIT_TRIGGER_PCT = 0.20
    settings.PARTIAL_TAKE_PROFIT_SIZE_PCT = 0.5
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0)
        action = rm.evaluate(pos, current_price=1.25)  # +25%, past the 20% trigger
        assert action.kind == "partial"
        assert action.fraction == pytest.approx(0.5)
        assert pos.partial_exit_taken is True
    finally:
        _restore(originals)


def test_partial_take_profit_does_not_fire_a_second_time():
    originals = _uncapped_exit_settings()
    settings.PARTIAL_TAKE_PROFIT_ENABLED = True
    settings.PARTIAL_TAKE_PROFIT_TRIGGER_PCT = 0.20
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, partial_exit_taken=True)
        action = rm.evaluate(pos, current_price=1.30)
        assert action.kind != "partial"
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# momentum-loss exit
# ---------------------------------------------------------------------------

def test_momentum_loss_exit_triggers_on_a_sharp_drop_from_a_recent_peak():
    originals = _uncapped_exit_settings()
    settings.MOMENTUM_EXIT_ENABLED = True
    settings.MOMENTUM_EXIT_LOOKBACK_SAMPLES = 5
    settings.MOMENTUM_EXIT_DROP_PCT = 0.12
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.5, take_profit=5.0)
        for price in (1.0, 1.2, 1.5):
            rm.evaluate(pos, current_price=price)
        action = rm.evaluate(pos, current_price=1.5 * 0.85)  # 15% off the 1.5 peak
        assert action.kind == "full"
        assert "momentum loss" in action.reason
    finally:
        _restore(originals)


def test_momentum_loss_exit_does_not_trigger_on_a_mild_pullback():
    originals = _uncapped_exit_settings()
    settings.MOMENTUM_EXIT_ENABLED = True
    settings.MOMENTUM_EXIT_LOOKBACK_SAMPLES = 5
    settings.MOMENTUM_EXIT_DROP_PCT = 0.12
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.5, take_profit=5.0)
        for price in (1.0, 1.2, 1.5):
            rm.evaluate(pos, current_price=price)
        action = rm.evaluate(pos, current_price=1.45)  # ~3% off peak, well under 12%
        assert action.kind == "none"
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# trend-reversal exit
# ---------------------------------------------------------------------------

def test_trend_reversal_exit_triggers_on_two_consecutive_lower_highs_after_the_peak():
    originals = _uncapped_exit_settings()
    settings.TREND_REVERSAL_EXIT_ENABLED = True
    settings.TREND_REVERSAL_MIN_SAMPLES = 5
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.5, take_profit=5.0)
        # ramp up to a clear peak, then two strictly lower prints after it
        prices = [1.0, 1.2, 1.5, 1.35, 1.20]
        action = None
        for price in prices:
            action = rm.evaluate(pos, current_price=price)
        assert action.kind == "full"
        assert "trend reversal" in action.reason
    finally:
        _restore(originals)


def test_trend_reversal_exit_does_not_trigger_without_enough_samples():
    originals = _uncapped_exit_settings()
    settings.TREND_REVERSAL_EXIT_ENABLED = True
    settings.TREND_REVERSAL_MIN_SAMPLES = 5
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.5, take_profit=5.0)
        action = None
        for price in (1.0, 1.2, 1.1):  # only 3 samples, needs 5
            action = rm.evaluate(pos, current_price=price)
        assert action.kind == "none"
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# time-based exit
# ---------------------------------------------------------------------------

def test_time_based_exit_disabled_by_default_even_on_a_very_old_position():
    originals = _uncapped_exit_settings()
    try:
        rm = ExitManager()
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)
        pos = _position(opened_at=old)
        action = rm.evaluate(pos, current_price=1.0)
        assert action.kind == "none"
    finally:
        _restore(originals)


def test_time_based_exit_triggers_when_enabled_and_position_is_too_old():
    originals = _uncapped_exit_settings()
    settings.TIME_BASED_EXIT_ENABLED = True
    settings.MAX_POSITION_AGE_HOURS = 48.0
    try:
        rm = ExitManager()
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=72)
        pos = _position(opened_at=old)
        action = rm.evaluate(pos, current_price=1.0)
        assert action.kind == "full"
        assert "max holding time" in action.reason
    finally:
        _restore(originals)


def test_time_based_exit_does_not_trigger_before_the_max_age():
    originals = _uncapped_exit_settings()
    settings.TIME_BASED_EXIT_ENABLED = True
    settings.MAX_POSITION_AGE_HOURS = 48.0
    try:
        rm = ExitManager()
        recent = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)
        pos = _position(opened_at=recent)
        action = rm.evaluate(pos, current_price=1.0)
        assert action.kind == "none"
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# priority: hard stop/target always wins over every other rule
# ---------------------------------------------------------------------------

def test_hard_stop_loss_wins_even_if_it_coincides_with_other_enabled_rules():
    originals = _uncapped_exit_settings()
    settings.BREAK_EVEN_ENABLED = True
    settings.TRAILING_STOP_ENABLED = True
    settings.PARTIAL_TAKE_PROFIT_ENABLED = True
    try:
        rm = ExitManager()
        pos = _position(entry_price=1.0, stop_loss=0.85)
        action = rm.evaluate(pos, current_price=0.80)
        assert action.kind == "full"
        assert "stop-loss" in action.reason
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# record_price_tick
# ---------------------------------------------------------------------------

def test_record_price_tick_tracks_the_running_peak():
    pos = _position(entry_price=1.0)
    record_price_tick(pos, 1.1)
    record_price_tick(pos, 1.05)  # lower - peak must not drop
    record_price_tick(pos, 1.2)
    assert pos.highest_price_since_entry == pytest.approx(1.2)


def test_record_price_tick_caps_the_sample_buffer_length():
    from app.exits.manager import MAX_RECENT_PRICE_SAMPLES

    pos = _position(entry_price=1.0)
    for i in range(MAX_RECENT_PRICE_SAMPLES + 20):
        record_price_tick(pos, 1.0 + i * 0.001)
    assert len(pos.recent_prices) == MAX_RECENT_PRICE_SAMPLES
    # the buffer keeps the MOST RECENT samples, not the earliest ones
    assert pos.recent_prices[-1][1] == pytest.approx(1.0 + (MAX_RECENT_PRICE_SAMPLES + 19) * 0.001)


# ---------------------------------------------------------------------------
# constructor overrides (backtester dependency injection)
# ---------------------------------------------------------------------------

def test_constructor_overrides_are_used_instead_of_settings():
    rm = ExitManager(
        trailing_enabled=True, trailing_activation_pct=0.05, trailing_distance_pct=0.02,
        break_even_enabled=False, partial_enabled=False, momentum_enabled=False,
        trend_reversal_enabled=False, time_exit_enabled=True, max_position_age_hours=1.0,
    )
    assert rm.trailing_activation_pct == pytest.approx(0.05)
    assert rm.trailing_distance_pct == pytest.approx(0.02)
    assert rm.break_even_enabled is False
    assert rm.partial_enabled is False
    assert rm.momentum_enabled is False
    assert rm.trend_reversal_enabled is False
    assert rm.time_exit_enabled is True
    assert rm.max_position_age_hours == pytest.approx(1.0)


def test_omitted_constructor_kwargs_fall_back_to_settings():
    default = ExitManager()
    explicit_none = ExitManager(trailing_activation_pct=None, momentum_drop_pct=None)
    assert explicit_none.trailing_activation_pct == default.trailing_activation_pct
    assert explicit_none.momentum_drop_pct == default.momentum_drop_pct
