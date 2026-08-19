"""Tests for app/strategy/version.py.

Two properties carry the whole feature: the label must be deterministic (or
it identifies a process, not a strategy), and it must change on behavioural
settings only (or history fragments on a log-level change and analytics
loses power for nothing).
"""
from app import models
from app.config import settings
from app.strategy.version import (
    BEHAVIORAL_SETTINGS,
    compute_label,
    current_config,
    current_label,
    register_current_version,
)


def test_the_label_is_deterministic():
    assert current_label() == current_label()
    assert compute_label(current_config()) == compute_label(current_config())


def test_the_label_looks_like_a_version():
    label = current_label()
    assert label.startswith("v-")
    assert len(label) == 10


def test_a_cosmetic_change_does_not_mint_a_new_version(monkeypatch):
    """Fragmenting trade history because someone turned up logging would
    cost analytical power and buy nothing."""
    before = current_label()
    monkeypatch.setattr(settings, "LOG_LEVEL", "DEBUG")
    assert current_label() == before


def test_a_behavioural_change_does_mint_a_new_version(monkeypatch):
    before = current_label()
    monkeypatch.setattr(settings, "MIN_SIGNAL_SCORE_TO_ENTER", settings.MIN_SIGNAL_SCORE_TO_ENTER + 5)
    after = current_label()
    assert after != before
    # ...and reverting must return to the original label, not a third one.
    monkeypatch.undo()
    assert current_label() == before


def test_every_behavioural_setting_actually_exists_on_settings():
    """A typo'd name would silently read as None and stop tracking that
    setting, which is invisible until a strategy change fails to version."""
    missing = [name for name in BEHAVIORAL_SETTINGS if not hasattr(settings, name)]
    assert missing == []


def test_the_tracked_settings_cover_the_gates_that_decide_trades():
    tracked = set(BEHAVIORAL_SETTINGS)
    for name in (
        "MIN_SIGNAL_SCORE_TO_ENTER", "MIN_MARKET_QUALITY_SCORE", "MIN_LIQUIDITY_USD",
        "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "MAX_PORTFOLIO_PCT_PER_TRADE",
        "PAPER_FEE_PCT", "SLIPPAGE_BPS",
    ):
        assert name in tracked, f"{name} changes trading behaviour but is not versioned"


def test_config_is_json_safe():
    import json

    json.dumps(current_config(), default=str)  # persisted on StrategyVersion.config


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_registering_is_an_upsert(db_session):
    first = register_current_version(db_session)
    db_session.flush()
    second = register_current_version(db_session)
    db_session.flush()

    assert first.id == second.id
    rows = db_session.query(models.StrategyVersion).filter_by(label=current_label()).count()
    assert rows == 1


def test_registering_records_the_configuration_it_versioned(db_session):
    row = register_current_version(db_session)
    db_session.flush()
    assert row.config["MIN_SIGNAL_SCORE_TO_ENTER"] == settings.MIN_SIGNAL_SCORE_TO_ENTER
    assert row.label == current_label()


def test_a_changed_strategy_gets_its_own_row(db_session, monkeypatch):
    original = register_current_version(db_session)
    db_session.flush()

    monkeypatch.setattr(settings, "STOP_LOSS_PCT", settings.STOP_LOSS_PCT + 0.03)
    changed = register_current_version(db_session)
    db_session.flush()

    assert changed.id != original.id
    assert changed.label != original.label
