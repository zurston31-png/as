"""Regression tests for the daily-loss safety limit.

Two rules are covered here, and the split matters:

  * `RiskManager.evaluate_daily_loss` - the frozen champion. Realized P&L
    only, measured against a fixed starting balance. The tests for it are
    characterisation tests: they pin the behavior that is in production
    today, INCLUDING the parts that are wrong, so that the flag being off
    is provably a no-op.

  * `app.risk.daily_loss.assess` - the challenger, behind
    RISK_EQUITY_AWARE_DAILY_LOSS. Equity drawdown against the equity the
    day actually started at.

The invariants every one of these tests exists to defend:

  I1  A profitable day never halts.
  I2  A loss strictly under the limit never halts.
  I3  A loss at or past the limit always halts. The boundary is inclusive.
  I4  Fees are counted exactly once. They are already inside `pnl_usd`
      via the fill price, and anything that charges them again is a bug.
  I5  Unrealized loss on an open position counts against the day (new rule
      only - the champion cannot see it, which is the whole point).
  I6  The limit is measured against the equity THIS DAY started at, not a
      constant, so it does not become more permissive as the account
      shrinks (new rule only).
  I7  Yesterday's losses never count against today.
  I8  A drawdown that cannot be measured - because part of the book has no
      live price - fails closed, but only when the unpriced amount is
      actually large enough to change the answer.
  I9  Restarting mid-day must not reset the day's budget.
  I10 With the flag off, the new module is not consulted at all.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.risk import daily_loss
from app.risk.manager import RiskManager
from app.services import portfolio
from app.state import set_state

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# Midday TODAY, not a fixed calendar date.
#
# This was pinned to 2026-08-21 and passed for exactly one day. The
# champion `evaluate_daily_loss` takes no `now` argument - it reads the
# real clock through `_day_bounds()` - so a fixture trade stamped with a
# hardcoded date silently falls outside "today" once the date rolls over,
# the day's realized loss sums to zero, and the halt tests assert against
# an empty window. It failed the first time the suite ran after midnight
# UTC, with no code change involved.
#
# Anchoring to the current date keeps the equity-aware tests (which DO
# pass `now=NOW` explicitly) deterministic relative to each other.
#
# That alone was still not deterministic. It replaced one hardcoded date
# with TWO clock reads - the module import that sets NOW, and the
# champion calling _day_bounds() during the test - and a run that crosses
# midnight UTC between them lands them on different days. A narrower
# window than the original bug, and the same bug.
#
# The `frozen_day` fixture below closes it: one instant governs both.
NOW = dt.datetime.now(dt.timezone.utc).replace(
    hour=12, minute=0, second=0, microsecond=0
)


@pytest.fixture(autouse=True)
def frozen_day(monkeypatch):
    """Pin the champion's day window to NOW's day.

    `evaluate_daily_loss` takes no `now` argument - it reads the clock
    itself, which is the behaviour under test and not something to change
    for a test's convenience. So the clock it reads is frozen instead, at
    the module's NOW, giving the fixture's trades and the window that
    counts them a single shared instant.

    Autouse because the hazard is the DEFAULT: any test that stamps a
    trade and lets the champion pick its own window is exposed, and
    opting in per-test is precisely the kind of thing that gets forgotten
    on the next test added at 23:59.
    """
    from app.risk import manager as risk_manager

    def fixed_bounds(now: dt.datetime | None = None):
        anchor = now or NOW
        start = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + dt.timedelta(days=1)

    monkeypatch.setattr(risk_manager, "_day_bounds", fixed_bounds)


@pytest.fixture()
def clean_db():
    """An empty book, a ledger at the starting balance, no day anchored."""
    def wipe(session):
        for model in (models.Trade, models.Position, models.RiskEvent):
            for row in session.query(model).all():
                session.delete(row)
        set_state(session, portfolio.CASH_KEY, settings.PORTFOLIO_STARTING_BALANCE_USD)
        for key in (daily_loss.DAY_START_EQUITY_KEY, daily_loss.LAST_EQUITY_KEY):
            row = session.get(models.BotState, key)
            if row is not None:
                session.delete(row)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


@pytest.fixture()
def no_live_prices(monkeypatch):
    """Every position values at cost. Used by the champion's tests, which
    never look at the book, and by the unpriced-band tests."""
    async def no_price(_addr):
        return None
    monkeypatch.setattr(portfolio.price_feed, "get_price_usd", no_price)


def _price_at(monkeypatch, price: float):
    async def fake_price(_addr):
        return price
    monkeypatch.setattr(portfolio.price_feed, "get_price_usd", fake_price)


def _closed_trade(db, pnl_usd: float, *, when: dt.datetime | None = None, fee_usd: float = 0.0):
    """A closed trade, stamped NOW by default.

    NOW rather than a fresh `datetime.now()`: with `frozen_day` active the
    champion's window is derived from NOW too, so this is the one
    timestamp guaranteed to sit inside it no matter when the suite runs.
    Taking a second clock reading here would reintroduce the straddle the
    fixture exists to remove. A test that wants a specific offset
    (yesterday, say) passes `when` explicitly.
    """
    db.add(models.Trade(
        symbol="COIN", token_address="MintCOIN", side="sell",
        status=models.TradeStatus.FILLED.value, pnl_usd=pnl_usd, fee_usd=fee_usd,
        closed_at=when or NOW,
    ))


def _open_position(db, *, qty: float, entry_price: float, symbol: str = "COIN"):
    db.add(models.Position(
        symbol=symbol, token_address=f"Mint{symbol}", qty=qty, entry_price=entry_price,
        stop_loss=entry_price * 0.85, take_profit=entry_price * 1.3,
        status=models.PositionStatus.OPEN.value,
    ))


def _set_equity(db, cash_usd: float):
    set_state(db, portfolio.CASH_KEY, cash_usd)


# ---------------------------------------------------------------------------
# the champion, characterised exactly as it behaves in production today
# ---------------------------------------------------------------------------

def test_champion_allows_a_profitable_day(clean_db):
    """I1."""
    _closed_trade(clean_db, +250.0)
    clean_db.commit()
    assert RiskManager().evaluate_daily_loss(clean_db).allowed


def test_champion_allows_a_loss_under_the_limit(clean_db):
    """I2."""
    rm = RiskManager()
    limit = settings.PORTFOLIO_STARTING_BALANCE_USD * rm.daily_loss_limit_pct
    _closed_trade(clean_db, -(limit - 0.01))
    clean_db.commit()
    assert rm.evaluate_daily_loss(clean_db).allowed


def test_champion_halts_exactly_at_the_limit(clean_db):
    """I3 - the boundary is inclusive. A limit that only fires one cent
    past itself is a different limit from the one the config names."""
    rm = RiskManager()
    limit = settings.PORTFOLIO_STARTING_BALANCE_USD * rm.daily_loss_limit_pct
    _closed_trade(clean_db, -limit)
    clean_db.commit()
    assert not rm.evaluate_daily_loss(clean_db).allowed


def test_champion_does_not_charge_fees_a_second_time(clean_db):
    """I4. `pnl_usd` is proceeds minus cost basis at FILL prices, and the
    fill model puts the fee inside the fill price
    (app/execution/fill_model.py). `Trade.fee_usd` is a record of what is
    already deducted, not an additional charge.

    This test would fail if anyone 'fixed' the check by subtracting
    fee_usd: the loss here sits one cent under the limit, and a second
    $50 of fees would push it over and halt a bot that never lost that
    money."""
    rm = RiskManager()
    limit = settings.PORTFOLIO_STARTING_BALANCE_USD * rm.daily_loss_limit_pct
    _closed_trade(clean_db, -(limit - 0.01), fee_usd=50.0)
    clean_db.commit()
    assert rm.evaluate_daily_loss(clean_db).allowed


def test_champion_ignores_yesterday(clean_db):
    """I7."""
    rm = RiskManager()
    limit = settings.PORTFOLIO_STARTING_BALANCE_USD * rm.daily_loss_limit_pct
    _closed_trade(clean_db, -limit * 5, when=NOW - dt.timedelta(days=1))
    clean_db.commit()
    assert rm.evaluate_daily_loss(clean_db).allowed


async def test_champion_cannot_see_an_unrealized_loss(clean_db, monkeypatch):
    """The defect this whole module exists to document, pinned as a test
    so it cannot be mistaken for an accident of the fixtures.

    $600 of cost basis is now worth $60 - a $540 unrealized loss against a
    $50 daily limit - and the champion reports the day as fine, because
    nothing has been sold."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 1000.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 400.0)
    _open_position(clean_db, qty=60_000, entry_price=0.01)     # $600 at cost
    clean_db.commit()
    _price_at(monkeypatch, 0.001)                              # now worth $60

    rm = RiskManager()
    assert rm.evaluate_daily_loss(clean_db).allowed, "characterising the gap, not endorsing it"

    assessment = await daily_loss.assess(
        clean_db, daily_loss_limit_pct=rm.daily_loss_limit_pct, now=NOW
    )
    assert assessment.breached, "the equity-aware rule must see what the champion cannot"


async def test_the_very_first_assessment_cannot_see_a_pre_existing_loss(clean_db, monkeypatch):
    """A limitation, pinned deliberately so nobody later mistakes it for a
    bug in the drawdown maths.

    With no prior measurement of any kind, the only reference available is
    right now, so a loss that happened before the first-ever assessment is
    outside the day it anchors. Inventing a midnight equity to catch it
    would be fabricating a measurement that was never taken. The
    assessment says which reference it used instead, and by the second day
    the anchor is real."""
    _set_equity(clean_db, 400.0)
    _open_position(clean_db, qty=60_000, entry_price=0.01)
    clean_db.commit()
    _price_at(monkeypatch, 0.001)                              # already down $540

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert not assessment.breached
    assert assessment.day_start_source == daily_loss.SOURCE_FIRST_OBSERVATION
    assert assessment.drawdown_usd == pytest.approx(0.0)
    # ...but the loss is not hidden: it is visible as unrealized.
    assert assessment.unrealized_pnl_open_usd == pytest.approx(-540.0)


# ---------------------------------------------------------------------------
# the challenger
# ---------------------------------------------------------------------------

async def test_a_flat_day_with_no_book_is_allowed(clean_db, no_live_prices):
    """I1/I2 at the trivial end: nothing happened, nothing halts, and the
    day anchors itself to the equity it opened at."""
    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert not assessment.breached
    assert assessment.drawdown_usd == pytest.approx(0.0)
    assert assessment.day_start_equity_usd == pytest.approx(
        settings.PORTFOLIO_STARTING_BALANCE_USD
    )
    assert assessment.remaining_budget_usd == pytest.approx(50.0)


async def test_a_realized_loss_under_the_limit_is_allowed(clean_db, no_live_prices):
    """I2."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 1000.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 960.0)                                # $40 of a $50 budget
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert not assessment.breached
    assert assessment.drawdown_usd == pytest.approx(40.0)
    assert assessment.remaining_budget_usd == pytest.approx(10.0)


async def test_the_boundary_is_inclusive(clean_db, no_live_prices):
    """I3. Exactly at the limit halts."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 1000.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 950.0)
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert assessment.breached
    assert assessment.remaining_budget_usd == pytest.approx(0.0)


async def test_an_unrealized_loss_counts_against_the_day(clean_db, monkeypatch):
    """I5. The headline behavior change."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 1000.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 500.0)
    _open_position(clean_db, qty=50_000, entry_price=0.01)      # $500 at cost
    clean_db.commit()
    _price_at(monkeypatch, 0.008)                               # now $400

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert assessment.breached
    assert assessment.drawdown_usd == pytest.approx(100.0)
    assert assessment.unrealized_pnl_open_usd == pytest.approx(-100.0)
    assert assessment.realized_pnl_today_usd == pytest.approx(0.0)


async def test_an_unrealized_gain_offsets_a_realized_loss(clean_db, monkeypatch):
    """Equity is the whole account, so a winner still open genuinely does
    fund a loser already closed. Both directions or neither - a rule that
    counted only unrealized losses would be a different, stricter rule
    than the one advertised."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 1000.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _closed_trade(clean_db, -60.0)                              # past the $50 limit
    _set_equity(clean_db, 440.0)                                # 1000 - 60 loss - 500 deployed
    _open_position(clean_db, qty=50_000, entry_price=0.01)      # $500 at cost
    clean_db.commit()
    _price_at(monkeypatch, 0.0104)                              # now $520, +$20

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert assessment.drawdown_usd == pytest.approx(40.0)
    assert not assessment.breached
    assert assessment.realized_pnl_today_usd == pytest.approx(-60.0)
    assert assessment.unrealized_pnl_open_usd == pytest.approx(20.0)


async def test_fees_are_not_charged_twice_by_the_new_rule_either(clean_db, no_live_prices):
    """I4 for the challenger. Equity already reflects every fee, because
    the fee is inside the fill price that moved the cash ledger. Nothing
    in this module reads `Trade.fee_usd`, and this pins that: a $200 fee
    record alongside a $40 drawdown must not produce a breach."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 1000.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _closed_trade(clean_db, -40.0, fee_usd=200.0)
    _set_equity(clean_db, 960.0)
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert not assessment.breached
    assert assessment.drawdown_usd == pytest.approx(40.0)


async def test_the_limit_rebases_as_the_account_shrinks(clean_db, no_live_prices):
    """I6, and the reason the champion's fixed reference is wrong.

    On a $600 account, 5% is $30. The champion would still be working to
    $50 - 8.3% of what is left - so the limit protects a shrinking account
    less and less, which is the opposite of what it is for."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 600.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 565.0)                                # $35 down
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert assessment.limit_usd == pytest.approx(30.0)
    assert assessment.breached

    rm = RiskManager()
    assert rm.evaluate_daily_loss(clean_db).allowed, "the champion, for contrast, sees nothing"


async def test_yesterdays_drawdown_does_not_count_against_today(clean_db, no_live_prices):
    """I7. The day-start anchor rolls, so a bad yesterday leaves today
    with a full budget - measured from where yesterday ended, not from
    where it began."""
    yesterday = (NOW - dt.timedelta(days=1))
    set_state(clean_db, daily_loss.LAST_EQUITY_KEY,
              {"equity_usd": 700.0, "observed_at": yesterday.isoformat()})
    _closed_trade(clean_db, -300.0, when=yesterday)
    _set_equity(clean_db, 700.0)
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert not assessment.breached
    assert assessment.day_start_equity_usd == pytest.approx(700.0)
    assert assessment.day_start_source == daily_loss.SOURCE_PREVIOUS_CLOSE
    assert assessment.drawdown_usd == pytest.approx(0.0)


async def test_a_new_day_anchors_to_yesterdays_close_not_to_now(clean_db, no_live_prices):
    """The rollover must not launder an overnight loss. Equity fell to
    $700 before midnight and is $650 now; today owns the $50, because the
    last real measurement before the roll is the honest reference."""
    yesterday = NOW - dt.timedelta(days=1)
    set_state(clean_db, daily_loss.LAST_EQUITY_KEY,
              {"equity_usd": 700.0, "observed_at": yesterday.isoformat()})
    _set_equity(clean_db, 650.0)
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert assessment.day_start_equity_usd == pytest.approx(700.0)
    assert assessment.drawdown_usd == pytest.approx(50.0)
    assert assessment.limit_usd == pytest.approx(35.0)
    assert assessment.breached


async def test_a_restart_mid_day_does_not_reset_the_budget(clean_db, no_live_prices):
    """I9. The anchor is persisted, so the second call - which is what a
    restarted process makes - reuses the morning's reference instead of
    re-anchoring to the depressed equity and handing out a fresh budget."""
    first = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert first.day_start_equity_usd == pytest.approx(1000.0)

    _set_equity(clean_db, 960.0)
    clean_db.commit()

    later = await daily_loss.assess(
        clean_db, daily_loss_limit_pct=0.05, now=NOW + dt.timedelta(hours=3)
    )
    assert later.day_start_equity_usd == pytest.approx(1000.0)
    assert later.drawdown_usd == pytest.approx(40.0)
    assert later.day_start_source == daily_loss.SOURCE_FIRST_OBSERVATION


async def test_a_read_only_assessment_anchors_nothing(clean_db, no_live_prices):
    """The dashboard must be able to look at the day without deciding
    what the day started at."""
    await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW, persist=False)
    assert clean_db.get(models.BotState, daily_loss.DAY_START_EQUITY_KEY) is None
    assert clean_db.get(models.BotState, daily_loss.LAST_EQUITY_KEY) is None


# ---------------------------------------------------------------------------
# I8 - what happens when the drawdown cannot be measured
# ---------------------------------------------------------------------------

async def test_a_large_unpriced_position_fails_closed(clean_db, no_live_prices):
    """A position valued at cost because its feed is dead INFLATES equity
    and therefore understates the drawdown - biased towards trading on,
    which is the dangerous direction. When the unpriced amount is big
    enough that a total loss on it would breach the limit, the honest
    answer is 'cannot tell', and cannot-tell halts."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 1000.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 800.0)
    _open_position(clean_db, qty=20_000, entry_price=0.01)      # $200, unpriceable
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert assessment.breached
    assert not assessment.measurable
    assert assessment.unpriced_positions == 1
    assert assessment.unpriced_usd == pytest.approx(200.0)
    assert "no live price" in assessment.reason


async def test_a_small_unpriced_position_does_not_halt_trading(clean_db, no_live_prices):
    """The other half of I8. Failing closed on any price hiccup at all
    would make an outage on a $5 dust position indistinguishable from a
    real breach, and an operator who sees that twice stops believing the
    halt means anything."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 1000.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 995.0)
    _open_position(clean_db, qty=500, entry_price=0.01)         # $5, unpriceable
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert not assessment.breached
    assert assessment.measurable
    assert assessment.unpriced_usd == pytest.approx(5.0)


async def test_an_unpriced_book_never_reports_the_gap_as_zero(clean_db, no_live_prices):
    """CLAUDE.md: a measurement that cannot be taken is recorded as
    unmeasurable, never as zero. The unpriced notional is reported as
    unpriced even on the allow path."""
    _set_equity(clean_db, 900.0)
    _open_position(clean_db, qty=10_000, entry_price=0.01)
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert assessment.unpriced_positions == 1
    assert assessment.unpriced_usd > 0


async def test_an_empty_account_has_no_budget(clean_db, no_live_prices):
    """A percentage of zero is zero, so without this an account that had
    lost everything would pass the daily-loss check forever."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": NOW.date().isoformat(), "equity_usd": 0.0,
               "captured_at": NOW.isoformat(), "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 0.0)
    clean_db.commit()

    assessment = await daily_loss.assess(clean_db, daily_loss_limit_pct=0.05, now=NOW)
    assert assessment.breached
    assert "no capital" in assessment.reason


# ---------------------------------------------------------------------------
# I10 - the flag
# ---------------------------------------------------------------------------

async def test_the_flag_defaults_to_off():
    """Production risk behavior must not change by merging this."""
    assert settings.RISK_EQUITY_AWARE_DAILY_LOSS is False


async def test_with_the_flag_off_the_champion_decides(clean_db, monkeypatch):
    """I10. The equity-aware rule would halt on this book; with the flag
    off it must not even be consulted."""
    monkeypatch.setattr(settings, "RISK_EQUITY_AWARE_DAILY_LOSS", False)

    def explode(*_args, **_kwargs):
        raise AssertionError("the new rule ran while the flag was off")
    monkeypatch.setattr(daily_loss, "assess", explode)

    _set_equity(clean_db, 100.0)
    _open_position(clean_db, qty=10_000, entry_price=0.01)
    clean_db.commit()
    _price_at(monkeypatch, 0.0001)

    assert (await RiskManager().assess_daily_loss(clean_db)).allowed


async def test_with_the_flag_on_the_challenger_decides(clean_db, monkeypatch):
    """The flag is wired to something real - the same book that the
    champion waves through is refused once it is on."""
    set_state(clean_db, daily_loss.DAY_START_EQUITY_KEY,
              {"date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
               "equity_usd": 1000.0, "captured_at": NOW.isoformat(),
               "source": daily_loss.SOURCE_STORED})
    _set_equity(clean_db, 400.0)
    _open_position(clean_db, qty=60_000, entry_price=0.01)      # $600 at cost
    clean_db.commit()
    _price_at(monkeypatch, 0.001)                               # now $60

    rm = RiskManager()
    monkeypatch.setattr(settings, "RISK_EQUITY_AWARE_DAILY_LOSS", False)
    assert (await rm.assess_daily_loss(clean_db)).allowed

    monkeypatch.setattr(settings, "RISK_EQUITY_AWARE_DAILY_LOSS", True)
    decision = await rm.assess_daily_loss(clean_db)
    assert not decision.allowed
    assert decision.reason


async def test_enabling_the_flag_mints_a_new_strategy_version(monkeypatch):
    """A rule change that halts on different days must not pool its
    results with the champion's. Off, the flag is absent from the digest
    so merely adding it does not split the frozen dataset; on, it is
    present and the label moves."""
    from app.strategy import version

    monkeypatch.setattr(settings, "RISK_EQUITY_AWARE_DAILY_LOSS", False)
    off_config = version.current_config()
    off_label = version.compute_label(off_config)
    assert "RISK_EQUITY_AWARE_DAILY_LOSS" not in off_config

    monkeypatch.setattr(settings, "RISK_EQUITY_AWARE_DAILY_LOSS", True)
    on_config = version.current_config()
    assert on_config["RISK_EQUITY_AWARE_DAILY_LOSS"] is True
    assert version.compute_label(on_config) != off_label
