import datetime as dt

import pytest

from app import models
from app.config import settings
from app.risk.manager import (
    HARD_MAX_DAILY_LOSS_PCT,
    HARD_MAX_PORTFOLIO_PCT_PER_TRADE,
    HARD_MAX_TOTAL_EXPOSURE_PCT,
    HARD_MIN_CONSECUTIVE_LOSSES,
    RiskManager,
    halt_trading,
    resume_trading,
)


def _open_position(db, symbol="COIN", **overrides):
    pos = models.Position(
        symbol=symbol, token_address=f"{symbol}addr", qty=1.0, entry_price=1.0,
        stop_loss=0.8, take_profit=1.5, status=models.PositionStatus.OPEN.value,
        **overrides,
    )
    db.add(pos)
    return pos


def _filled_trade(db, symbol="COIN", side="buy", pnl_usd=None, when=None, **overrides):
    now = when or dt.datetime.now(dt.timezone.utc)
    trade = models.Trade(
        symbol=symbol, side=side, status=models.TradeStatus.FILLED.value,
        pnl_usd=pnl_usd, created_at=now, opened_at=now if side == "buy" else None,
        closed_at=now if side == "sell" else None,
        **overrides,
    )
    db.add(trade)
    return trade


# ---------------------------------------------------------------------------
# risk-based position sizing (the Stage 3 headline fix)
# ---------------------------------------------------------------------------

def _uncapped_settings():
    """Push MAX_TRADE_SIZE_USD and both exposure-pct ceilings far out of the
    way. These sizing-formula tests want to isolate the risk/stop-distance
    math itself; without this, the *default* exposure caps (10% per-token,
    60% total) silently clip the notional before the formula assertion ever
    runs, since e.g. 10% of a $10k portfolio ($1,000) is tighter than the
    formula's own output at the stop distances these tests use."""
    originals = (settings.MAX_TRADE_SIZE_USD, settings.MAX_EXPOSURE_PER_TOKEN_PCT, settings.MAX_TOTAL_EXPOSURE_PCT)
    settings.MAX_TRADE_SIZE_USD = 1_000_000
    settings.MAX_EXPOSURE_PER_TOKEN_PCT = 1.0
    settings.MAX_TOTAL_EXPOSURE_PCT = 1.0
    return originals


def _restore_settings(originals):
    settings.MAX_TRADE_SIZE_USD, settings.MAX_EXPOSURE_PER_TOKEN_PCT, settings.MAX_TOTAL_EXPOSURE_PCT = originals


def test_position_size_is_risk_over_stop_distance_not_a_flat_percent():
    """The bug this replaces: sizing used to be a flat % of the portfolio
    regardless of the stop distance, so "2% risk" was actually notional size
    - the real dollar risk floated with whatever the stop happened to be.
    Sizing must now solve for the notional that loses exactly risk_pct if
    the stop is hit: risk_amount = size * stop_pct."""
    originals = _uncapped_settings()
    try:
        rm = RiskManager()
        portfolio_value = 10_000.0
        stop_pct = 0.15
        size = rm.position_size_usd(portfolio_value, stop_loss_pct=stop_pct)

        risk_amount = portfolio_value * rm.max_pct_per_trade
        expected = risk_amount / stop_pct
        assert size == pytest.approx(expected)

        # Prove the property directly: if the stop is hit, the loss equals
        # the intended risk amount, independent of the stop distance.
        loss_if_stopped = size * stop_pct
        assert loss_if_stopped == pytest.approx(risk_amount)
    finally:
        _restore_settings(originals)


def test_wider_stop_produces_a_smaller_position():
    """A wider stop means more $ lost per unit of notional, so the notional
    must shrink to keep the dollar risk constant - the opposite of the old
    fixed-percent behavior, where a wider stop silently increased the real
    risk taken with no limit on it.

    Stop distances are chosen so neither notional crosses the per-token
    exposure hard ceiling (25% of portfolio, not configurable away) - that
    cap is a separate mechanism and would otherwise mask the formula this
    test is isolating."""
    originals = _uncapped_settings()
    try:
        rm = RiskManager()
        tight = rm.position_size_usd(10_000.0, stop_loss_pct=0.10)
        wide = rm.position_size_usd(10_000.0, stop_loss_pct=0.30)
        assert wide < tight
        # Both must still lose the same $ amount if stopped out.
        assert (tight * 0.10) == pytest.approx(wide * 0.30)
    finally:
        _restore_settings(originals)


def test_position_size_defaults_to_the_configured_stop_pct():
    originals = _uncapped_settings()
    try:
        rm = RiskManager()
        size = rm.position_size_usd(10_000.0)
        expected = (10_000.0 * rm.max_pct_per_trade) / rm.stop_loss_pct
        assert size == pytest.approx(expected)
    finally:
        _restore_settings(originals)


def test_position_size_respects_the_absolute_trade_cap():
    original_cap = settings.MAX_TRADE_SIZE_USD
    settings.MAX_TRADE_SIZE_USD = 50
    try:
        rm = RiskManager()
        size = rm.position_size_usd(portfolio_value_usd=10_000)
        assert size == 50
    finally:
        settings.MAX_TRADE_SIZE_USD = original_cap


def test_position_size_never_negative_or_absurd_on_a_near_zero_stop():
    """stop_loss_pct is clamped to the hard floor before it ever divides
    anything, so a bad ATR reading approaching 0% cannot blow the size up."""
    rm = RiskManager()
    size = rm.position_size_usd(10_000.0, stop_loss_pct=0.0001)
    assert size >= 0
    assert size <= settings.MAX_TRADE_SIZE_USD


def test_position_size_shrinks_after_a_smaller_portfolio_never_grows_to_compensate():
    """Sizing off the CURRENT portfolio value is what makes it impossible to
    scale up to recover a loss: a smaller balance can only ever produce a
    smaller or equal next trade for the same risk %."""
    originals = _uncapped_settings()
    try:
        rm = RiskManager()
        before_loss = rm.position_size_usd(10_000.0)
        after_loss = rm.position_size_usd(9_000.0)   # portfolio shrank from a loss
        assert after_loss < before_loss
    finally:
        _restore_settings(originals)


# ---------------------------------------------------------------------------
# exposure caps
# ---------------------------------------------------------------------------

def test_position_size_capped_by_total_exposure_room():
    rm = RiskManager()
    original_cap = settings.MAX_TRADE_SIZE_USD
    settings.MAX_TRADE_SIZE_USD = 1_000_000
    try:
        portfolio_value = 10_000.0
        # Already fully exposed up to the total-exposure ceiling.
        already_used = portfolio_value * rm.max_total_exposure_pct
        size = rm.position_size_usd(portfolio_value, current_total_exposure_usd=already_used)
        assert size == 0
    finally:
        settings.MAX_TRADE_SIZE_USD = original_cap


def test_position_size_capped_by_per_token_exposure_room():
    rm = RiskManager()
    original_cap = settings.MAX_TRADE_SIZE_USD
    settings.MAX_TRADE_SIZE_USD = 1_000_000
    try:
        portfolio_value = 10_000.0
        already_used = portfolio_value * rm.max_exposure_per_token_pct
        size = rm.position_size_usd(portfolio_value, current_symbol_exposure_usd=already_used)
        assert size == 0
    finally:
        settings.MAX_TRADE_SIZE_USD = original_cap


def test_position_size_is_partial_when_only_some_room_remains():
    rm = RiskManager()
    original_cap = settings.MAX_TRADE_SIZE_USD
    settings.MAX_TRADE_SIZE_USD = 1_000_000
    try:
        portfolio_value = 10_000.0
        total_cap_usd = portfolio_value * rm.max_total_exposure_pct
        used = total_cap_usd - 10.0   # only $10 of room left before the total cap
        size = rm.position_size_usd(portfolio_value, current_total_exposure_usd=used)
        assert size == pytest.approx(10.0)
    finally:
        settings.MAX_TRADE_SIZE_USD = original_cap


def test_exposure_caps_are_clamped_to_hard_ceilings(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TOTAL_EXPOSURE_PCT", 5.0)
    rm = RiskManager()
    assert rm.max_total_exposure_pct == HARD_MAX_TOTAL_EXPOSURE_PCT


# ---------------------------------------------------------------------------
# hard ceilings (existing behavior, unchanged)
# ---------------------------------------------------------------------------

def test_hard_ceiling_clamps_oversized_pct_config(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PORTFOLIO_PCT_PER_TRADE", 0.9)
    rm = RiskManager()
    assert rm.max_pct_per_trade == HARD_MAX_PORTFOLIO_PCT_PER_TRADE


def test_hard_ceiling_clamps_daily_loss_config(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_LOSS_LIMIT_PCT", 0.9)
    rm = RiskManager()
    assert rm.daily_loss_limit_pct == HARD_MAX_DAILY_LOSS_PCT


def test_consecutive_loss_floor_cannot_be_configured_away(monkeypatch):
    """A shutdown that fires on a single loss isn't a strategy filter, it's
    noise - the floor keeps someone from setting MAX_CONSECUTIVE_LOSSES=1."""
    monkeypatch.setattr(settings, "MAX_CONSECUTIVE_LOSSES", 1)
    rm = RiskManager()
    assert rm.max_consecutive_losses == HARD_MIN_CONSECUTIVE_LOSSES


def test_stop_loss_take_profit_bracket():
    rm = RiskManager()
    sl, tp = rm.stop_loss_take_profit(100.0)
    assert sl == pytest.approx(100 * (1 - rm.stop_loss_pct))
    assert tp == pytest.approx(100 * (1 + rm.take_profit_pct))
    assert sl < 100.0 < tp


# ---------------------------------------------------------------------------
# entry gate
# ---------------------------------------------------------------------------

def test_check_can_open_position_blocked_when_halted(db_session):
    rm = RiskManager()
    halt_trading(db_session, "unit test halt")
    db_session.commit()

    decision = rm.check_can_open_position(db_session)
    assert not decision.allowed
    assert "halted" in decision.reason

    resume_trading(db_session)
    db_session.commit()
    assert rm.check_can_open_position(db_session).allowed


def test_check_can_open_position_blocked_at_max_concurrent(db_session):
    rm = RiskManager()
    created = []
    for i in range(rm.max_concurrent_positions):
        created.append(_open_position(db_session, symbol=f"MAXPOSCOIN{i}"))
    db_session.commit()

    try:
        decision = rm.check_can_open_position(db_session)
        assert not decision.allowed
        assert "max concurrent positions" in decision.reason
    finally:
        for pos in created:
            db_session.delete(pos)
        db_session.commit()


def test_check_can_open_position_blocked_at_daily_trade_limit(db_session):
    rm = RiskManager()
    created = []
    for i in range(rm.max_daily_trades):
        created.append(_filled_trade(db_session, symbol=f"DAILYLIMIT{i}"))
    db_session.commit()

    try:
        decision = rm.check_can_open_position(db_session)
        assert not decision.allowed
        assert "daily trade limit" in decision.reason
    finally:
        for t in created:
            db_session.delete(t)
        db_session.commit()


def test_check_can_open_position_ignores_yesterdays_trades_for_daily_limit(db_session):
    rm = RiskManager()
    yesterday = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    created = [
        _filled_trade(db_session, symbol=f"YEST{i}", when=yesterday)
        for i in range(rm.max_daily_trades + 5)
    ]
    db_session.commit()
    try:
        decision = rm.check_can_open_position(db_session)
        assert decision.allowed
    finally:
        for t in created:
            db_session.delete(t)
        db_session.commit()


def test_cooldown_blocks_reentry_into_the_same_symbol(db_session):
    rm = RiskManager()
    trade = _filled_trade(db_session, symbol="COOLDOWNCOIN")
    db_session.commit()
    try:
        decision = rm.check_can_open_position(db_session, symbol="COOLDOWNCOIN")
        assert not decision.allowed
        assert "cooldown" in decision.reason
    finally:
        db_session.delete(trade)
        db_session.commit()


def test_cooldown_does_not_block_a_different_symbol(db_session):
    rm = RiskManager()
    trade = _filled_trade(db_session, symbol="COOLDOWNCOIN")
    db_session.commit()
    try:
        decision = rm.check_can_open_position(db_session, symbol="SOMEOTHERCOIN")
        assert decision.allowed
    finally:
        db_session.delete(trade)
        db_session.commit()


def test_cooldown_clears_after_the_configured_window(db_session, monkeypatch):
    monkeypatch.setattr(settings, "TRADE_COOLDOWN_SECONDS", 60)
    rm = RiskManager()
    old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=120)
    trade = _filled_trade(db_session, symbol="OLDCOOLDOWN", when=old)
    db_session.commit()
    try:
        decision = rm.check_can_open_position(db_session, symbol="OLDCOOLDOWN")
        assert decision.allowed
    finally:
        db_session.delete(trade)
        db_session.commit()


def test_cooldown_skipped_when_no_symbol_given(db_session):
    """check_can_open_position(db) with no symbol is a general availability
    check and must not fail due to per-symbol cooldown state."""
    rm = RiskManager()
    trade = _filled_trade(db_session, symbol="ANYCOIN")
    db_session.commit()
    try:
        assert rm.check_can_open_position(db_session).allowed
    finally:
        db_session.delete(trade)
        db_session.commit()


# ---------------------------------------------------------------------------
# daily loss halt
# ---------------------------------------------------------------------------

def test_evaluate_daily_loss_halts_past_limit(db_session):
    rm = RiskManager()
    loss_limit = settings.PORTFOLIO_STARTING_BALANCE_USD * rm.daily_loss_limit_pct
    trade = _filled_trade(db_session, side="sell", pnl_usd=-(loss_limit + 1))
    db_session.commit()
    try:
        decision = rm.evaluate_daily_loss(db_session)
        assert not decision.allowed
    finally:
        db_session.delete(trade)
        db_session.commit()


def test_evaluate_daily_loss_ignores_yesterdays_pnl(db_session):
    rm = RiskManager()
    loss_limit = settings.PORTFOLIO_STARTING_BALANCE_USD * rm.daily_loss_limit_pct
    yesterday = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    trade = models.Trade(
        symbol="COIN", side="sell", status=models.TradeStatus.FILLED.value,
        pnl_usd=-(loss_limit * 5), closed_at=yesterday,
    )
    db_session.add(trade)
    db_session.commit()
    try:
        decision = rm.evaluate_daily_loss(db_session)
        assert decision.allowed
    finally:
        db_session.delete(trade)
        db_session.commit()


# ---------------------------------------------------------------------------
# consecutive-loss shutdown
# ---------------------------------------------------------------------------

def test_consecutive_losses_halt_after_the_threshold(db_session):
    rm = RiskManager()
    created = [
        _filled_trade(db_session, symbol=f"STREAK{i}", side="sell", pnl_usd=-5.0)
        for i in range(rm.max_consecutive_losses)
    ]
    db_session.commit()
    try:
        decision = rm.evaluate_consecutive_losses(db_session)
        assert not decision.allowed
        assert "consecutive losing trades" in decision.reason
    finally:
        for t in created:
            db_session.delete(t)
        db_session.commit()


def test_a_win_breaks_the_losing_streak(db_session):
    rm = RiskManager()
    created = [
        _filled_trade(db_session, symbol=f"BREAK{i}", side="sell", pnl_usd=-5.0)
        for i in range(rm.max_consecutive_losses - 1)
    ]
    # Most recent trade is a win, so the streak within the lookback window is broken.
    created.append(_filled_trade(db_session, symbol="BREAKWIN", side="sell", pnl_usd=5.0))
    db_session.commit()
    try:
        decision = rm.evaluate_consecutive_losses(db_session)
        assert decision.allowed
    finally:
        for t in created:
            db_session.delete(t)
        db_session.commit()


def test_not_enough_closed_trades_yet_does_not_halt(db_session):
    rm = RiskManager()
    created = [
        _filled_trade(db_session, symbol=f"FEW{i}", side="sell", pnl_usd=-5.0)
        for i in range(rm.max_consecutive_losses - 1)
    ]
    db_session.commit()
    try:
        decision = rm.evaluate_consecutive_losses(db_session)
        assert decision.allowed
    finally:
        for t in created:
            db_session.delete(t)
        db_session.commit()


# ---------------------------------------------------------------------------
# constructor overrides (backtester dependency injection)
# ---------------------------------------------------------------------------

def test_constructor_overrides_are_used_instead_of_settings():
    """The backtester needs deterministic, config-driven limits independent
    of whatever .env happens to say - these kwargs are what make that
    possible while still running the exact same sizing/gating code live
    trading uses."""
    rm = RiskManager(
        max_pct_per_trade=0.03, stop_loss_pct=0.20, take_profit_pct=0.5,
        max_trade_size_usd=999_999, max_concurrent_positions=9,
        max_exposure_per_token_pct=1.0, max_total_exposure_pct=1.0,
        max_consecutive_losses=7, max_daily_trades=40, cooldown_seconds=1,
    )
    assert rm.max_pct_per_trade == pytest.approx(0.03)
    assert rm.stop_loss_pct == pytest.approx(0.20)
    assert rm.take_profit_pct == pytest.approx(0.5)
    assert rm.max_trade_size_usd == 999_999
    assert rm.max_concurrent_positions == 9
    assert rm.max_consecutive_losses == 7
    assert rm.max_daily_trades == 40
    assert rm.cooldown_seconds == 1

    # stop_loss_pct=0.20 keeps the resulting notional under the
    # HARD_MAX_EXPOSURE_PER_TOKEN_PCT ceiling (25% of portfolio) even though
    # this test set the per-token cap itself to 100% - that hard ceiling
    # isn't configurable away (see test_exposure_caps_are_clamped_to_hard_ceilings),
    # so the formula check below needs a stop wide enough to not collide
    # with it, the same lesson test_wider_stop_produces_a_smaller_position
    # already documents.
    size = rm.position_size_usd(10_000.0)
    expected = (10_000.0 * 0.03) / 0.20
    assert size == pytest.approx(expected)


def test_constructor_overrides_still_respect_hard_ceilings():
    rm = RiskManager(max_pct_per_trade=0.9, daily_loss_limit_pct=0.9, max_consecutive_losses=1)
    assert rm.max_pct_per_trade == HARD_MAX_PORTFOLIO_PCT_PER_TRADE
    assert rm.daily_loss_limit_pct == HARD_MAX_DAILY_LOSS_PCT
    assert rm.max_consecutive_losses == HARD_MIN_CONSECUTIVE_LOSSES


def test_omitted_constructor_kwargs_fall_back_to_settings():
    rm_default = RiskManager()
    rm_explicit_none = RiskManager(max_pct_per_trade=None, stop_loss_pct=None)
    assert rm_explicit_none.max_pct_per_trade == rm_default.max_pct_per_trade
    assert rm_explicit_none.stop_loss_pct == rm_default.stop_loss_pct
