"""Changing the paper account's starting balance without breaking it.

`PORTFOLIO_STARTING_BALANCE_USD` is one line in `.env`, and editing it on
a running bot used to do three wrong things quietly: the cash ledger kept
the old balance, the accounting check reported a discrepancy the exact
size of the edit (which fails the kill switch closed, so the bot stops
opening positions for a reason unrelated to its books), and the dataset
split without the strategy version changing to say so.

These tests pin all three fixes.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.safety import reconcile as reconcile_mod
from app.services import portfolio
from app.state import set_state

NOW = dt.datetime(2026, 9, 6, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture()
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


SYMBOL = "BALANCECOIN"


@pytest.fixture(autouse=True)
def clean():
    """Own only what this file creates, and put the ledger back afterwards.

    An earlier version of this fixture deleted every Trade row. The test
    database is shared and serial, so that orphaned other files' CLOSED
    positions from their sell legs - which the position-integrity check
    correctly reports as "closed but never credited to the ledger", which
    fails the kill switch, which blocked entries in twelve later webhook
    tests. Wiping shared state is not cleanup.
    """
    def snapshot():
        s = SessionLocal()
        try:
            rows = {
                r.key: r.value
                for r in s.query(models.BotState).filter(
                    models.BotState.key.in_([portfolio.CASH_KEY, portfolio.BASELINE_KEY])
                )
            }
        finally:
            s.close()
        return rows

    def restore(rows):
        s = SessionLocal()
        try:
            s.query(models.Trade).filter(models.Trade.symbol == SYMBOL).delete(
                synchronize_session=False
            )
            s.query(models.BotState).filter(
                models.BotState.key.in_([portfolio.CASH_KEY, portfolio.BASELINE_KEY])
            ).delete(synchronize_session=False)
            for key, value in rows.items():
                s.add(models.BotState(key=key, value=value))
            s.commit()
        finally:
            s.close()

    original = snapshot()
    restore({})          # start each test with no ledger state of its own
    yield
    restore(original)


def _buy(db, size_usd: float, at: dt.datetime) -> None:
    db.add(models.Trade(
        symbol=SYMBOL, side="buy", status=models.TradeStatus.FILLED.value,
        size_usd=size_usd, created_at=at,
    ))


def _sell(db, qty: float, price: float, at: dt.datetime) -> None:
    db.add(models.Trade(
        symbol=SYMBOL, side="sell", status=models.TradeStatus.FILLED.value,
        qty=qty, exit_price=price, pnl_usd=1.0, closed_at=at, created_at=at,
    ))


# --- the baseline -----------------------------------------------------------

def test_a_database_with_no_recorded_baseline_behaves_as_before(db):
    """Every database created before the baseline existed has none. The
    fallback must reproduce the old behaviour exactly - the configured
    balance, covering the whole trade record - or upgrading the code
    would itself trip the accounting check."""
    baseline = portfolio.get_ledger_baseline(db)
    assert baseline.balance_usd == settings.PORTFOLIO_STARTING_BALANCE_USD
    assert baseline.since is None


def test_the_baseline_round_trips(db):
    portfolio.set_ledger_baseline(db, 250.0, since=NOW)
    baseline = portfolio.get_ledger_baseline(db)
    assert baseline.balance_usd == 250.0
    assert baseline.since == NOW


def test_an_unreadable_baseline_falls_back_rather_than_crashing(db):
    """Reconciliation gates trading. A corrupt state row must degrade to
    the old behaviour, not take the kill switch's accounting check down
    with it - a check that cannot run counts as failed."""
    set_state(db, portfolio.BASELINE_KEY, {"balance_usd": "not a number"})
    baseline = portfolio.get_ledger_baseline(db)
    assert baseline.balance_usd == settings.PORTFOLIO_STARTING_BALANCE_USD
    assert baseline.since is None


# --- reconciliation ---------------------------------------------------------

def test_the_books_balance_against_the_recorded_baseline(db):
    portfolio.set_ledger_baseline(db, 250.0, since=NOW)
    set_state(db, portfolio.CASH_KEY, 230.0)
    _buy(db, 20.0, NOW + dt.timedelta(minutes=1))
    db.flush()

    result = reconcile_mod.reconcile(db)
    assert result.expected_cash == pytest.approx(230.0)
    assert result.balanced, result.summary()


def test_trades_from_before_a_reset_are_excluded(db):
    """They moved a ledger that no longer exists. Counting them restates
    history against a baseline that was never true for them - which is
    exactly the $750 phantom discrepancy that changing the setting used
    to produce."""
    _buy(db, 100.0, NOW - dt.timedelta(days=5))
    _sell(db, 1000.0, 0.12, NOW - dt.timedelta(days=4))
    portfolio.set_ledger_baseline(db, 250.0, since=NOW)
    set_state(db, portfolio.CASH_KEY, 250.0)
    db.flush()

    result = reconcile_mod.reconcile(db)
    assert result.filled_trades == 0
    assert result.expected_cash == pytest.approx(250.0)
    assert result.balanced, result.summary()


def test_a_genuine_drift_is_still_caught_after_a_reset(db):
    """The exclusion must not become a blanket amnesty - the check still
    has to fail when the ledger is actually wrong."""
    portfolio.set_ledger_baseline(db, 250.0, since=NOW)
    set_state(db, portfolio.CASH_KEY, 250.0)      # no buy deducted
    _buy(db, 20.0, NOW + dt.timedelta(minutes=1))
    db.flush()

    result = reconcile_mod.reconcile(db)
    assert not result.balanced
    assert result.discrepancy == pytest.approx(20.0)


# --- the strategy version ---------------------------------------------------

def test_the_collection_balance_hashes_exactly_as_before(monkeypatch):
    """Closing the versioning hole must not itself split the dataset it
    protects. At the value the current run was collected at, the label is
    unchanged - so deploy/auto_update.sh still rolls this out."""
    from app.strategy import version as v

    monkeypatch.setattr(settings, "PORTFOLIO_STARTING_BALANCE_USD", 1000.0)
    assert "PORTFOLIO_STARTING_BALANCE_USD" not in v.current_config()
    assert v.compute_label(v.current_config()) == "v-83c77cda"


def test_moving_the_balance_mints_a_new_version(monkeypatch):
    """Position size scales with the balance, so fills and P&L are not
    comparable across the change. The label has to say so - and once it
    changes, the auto-updater refuses to deploy it unattended, which is
    correct for a deliberate restart."""
    from app.strategy import version as v

    monkeypatch.setattr(settings, "PORTFOLIO_STARTING_BALANCE_USD", 1000.0)
    before = v.compute_label(v.current_config())

    monkeypatch.setattr(settings, "PORTFOLIO_STARTING_BALANCE_USD", 250.0)
    config = v.current_config()
    assert config["PORTFOLIO_STARTING_BALANCE_USD"] == 250.0
    assert v.compute_label(config) != before


def test_the_collection_value_is_a_constant_not_the_settings_default(monkeypatch):
    """A later edit to the default must not redefine what "unchanged"
    means for a dataset already in the ground: trades collected at $1,000
    stay collected at $1,000 however the default moves."""
    from app.strategy import version as v

    assert v.DEFAULTED_BEHAVIORAL_SETTINGS["PORTFOLIO_STARTING_BALANCE_USD"] == 1000.0
    monkeypatch.setattr(settings, "PORTFOLIO_STARTING_BALANCE_USD", 500.0)
    assert "PORTFOLIO_STARTING_BALANCE_USD" in v.current_config()
