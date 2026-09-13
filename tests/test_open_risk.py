"""Tests for worst-case open-position risk (app/risk/open_risk.py).

The invariants:

  O1  Stop risk is measured from the CURRENT MARK, not from entry, so the
      unrealized loss already inside the daily drawdown is not counted a
      second time.
  O2  A position already trading below its stop adds no further stop risk.
  O3  The stress case is strictly worse than the base case, and the extra
      comes from the same fill model that would price the real exit.
  O4  A prospective position is included in the worst case before it is
      opened - the only order in which the gate can prevent anything.
  O5  The gate refuses an entry whose worst case would overrun the day's
      remaining loss budget.
  O6  With RISK_EQUITY_AWARE_DAILY_LOSS off, the gate approves everything.
  O7  Risks that cannot be quantified are named, never modeled as zero.
"""
import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.risk import daily_loss, open_risk
from app.risk.manager import RiskManager
from app.services import portfolio
from app.state import set_state

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def clean_db():
    def wipe(session):
        for model in (models.Trade, models.Position, models.RiskEvent):
            for row in session.query(model).all():
                session.delete(row)
        set_state(session, portfolio.CASH_KEY, settings.PORTFOLIO_STARTING_BALANCE_USD)
        # The gate tests run a full daily-loss assessment, which anchors
        # the day's starting equity in bot_state. Left behind, one test's
        # book becomes the next test's day-start reference.
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


def _position(db, *, qty, entry_price, stop_loss, symbol="COIN", liquidity=None):
    pos = models.Position(
        symbol=symbol, token_address=f"Mint{symbol}", qty=qty, entry_price=entry_price,
        stop_loss=stop_loss, take_profit=entry_price * 1.3,
        status=models.PositionStatus.OPEN.value,
        liquidity_at_entry_usd=liquidity, lowest_liquidity_usd=liquidity,
    )
    db.add(pos)
    return pos


def _price_at(monkeypatch, price):
    async def fake_price(_addr):
        return price
    monkeypatch.setattr(open_risk.price_feed, "get_price_usd", fake_price)


# ---------------------------------------------------------------------------
# O1/O2 - the double-counting rule
# ---------------------------------------------------------------------------

async def test_stop_risk_is_measured_from_the_mark_not_from_entry(clean_db, monkeypatch):
    """O1. Bought at $1.00, now $0.90, stop $0.85, 100 units.

    $10 is already lost and already inside the daily equity drawdown. Only
    the $5 from here to the stop is still ahead. Measuring $15 of stop risk
    would charge that first $10 twice and halt the bot on a loss half again
    as large as the one it faces."""
    _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85, liquidity=1_000_000)
    clean_db.commit()
    _price_at(monkeypatch, 0.90)

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)
    assert risk.current_open_drawdown_usd == pytest.approx(10.0)
    assert risk.loss_if_all_stops_hit_usd == pytest.approx(5.0)
    # And the two are disjoint: together they are the entry-to-stop total.
    assert risk.current_open_drawdown_usd + risk.loss_if_all_stops_hit_usd == pytest.approx(15.0)


async def test_a_position_below_its_stop_adds_no_further_stop_risk(clean_db, monkeypatch):
    """O2. At $0.80 with a stop at $0.85 the stop loss has happened - it
    is in the drawdown, and the exit is pending rather than prospective.
    Counting it again as risk-ahead would double it."""
    _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85, liquidity=1_000_000)
    clean_db.commit()
    _price_at(monkeypatch, 0.80)

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)
    assert risk.current_open_drawdown_usd == pytest.approx(20.0)
    assert risk.loss_if_all_stops_hit_usd == pytest.approx(0.0)


async def test_a_winning_position_reports_no_drawdown_but_still_has_stop_risk(clean_db, monkeypatch):
    """Up on the trade is not the same as nothing at risk: the stop is
    still below, and the distance to it is real money."""
    _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85, liquidity=1_000_000)
    clean_db.commit()
    _price_at(monkeypatch, 1.20)

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)
    assert risk.current_open_drawdown_usd == pytest.approx(0.0)
    assert risk.loss_if_all_stops_hit_usd == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# O3 - base versus stress
# ---------------------------------------------------------------------------

async def test_the_stress_case_is_worse_than_the_base_case(clean_db, monkeypatch):
    """O3. A stop that fills exactly at its stop price is the floor, not a
    forecast - getting out costs impact, spread and fee on top."""
    _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85, liquidity=50_000)
    clean_db.commit()
    _price_at(monkeypatch, 0.90)

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)
    assert risk.stress_loss_if_all_stops_hit_usd > risk.loss_if_all_stops_hit_usd


async def test_the_stress_premium_comes_from_the_shipped_fill_model(clean_db, monkeypatch):
    """The extra is not a fudge factor - it is exactly what
    app/execution/fill_model.py charges to sell that notional into that
    pool, so this figure cannot drift away from the model that prices the
    real exit."""
    from app.execution.fill_model import price_impact_pct

    _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85, liquidity=20_000)
    clean_db.commit()
    _price_at(monkeypatch, 0.90)

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)

    exit_notional = 0.85 * 100
    expected_extra = exit_notional * (
        price_impact_pct(exit_notional, 20_000)
        + settings.PAPER_SPREAD_PCT
        + settings.PAPER_FEE_PCT
    )
    assert risk.stress_loss_if_all_stops_hit_usd == pytest.approx(
        risk.loss_if_all_stops_hit_usd + expected_extra
    )


async def test_a_thinner_pool_costs_more_to_exit(clean_db, monkeypatch):
    """Price impact is the whole reason the stress case exists: the same
    position in a shallower pool is a worse position."""
    deep = _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85,
                     symbol="DEEP", liquidity=5_000_000)
    clean_db.commit()
    _price_at(monkeypatch, 0.90)
    deep_risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)

    clean_db.delete(deep)
    _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85,
              symbol="THIN", liquidity=1_000)
    clean_db.commit()
    thin_risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)

    assert thin_risk.loss_if_all_stops_hit_usd == pytest.approx(
        deep_risk.loss_if_all_stops_hit_usd
    ), "the base case is a price distance and does not know about depth"
    assert (
        thin_risk.stress_loss_if_all_stops_hit_usd
        > deep_risk.stress_loss_if_all_stops_hit_usd
    )


async def test_the_lowest_recorded_depth_is_used_not_the_depth_at_entry(clean_db, monkeypatch):
    """A pool that has thinned since entry is the pool the exit will hit.
    Pricing the exit against entry depth would understate it precisely
    when it matters."""
    pos = _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85, liquidity=500_000)
    pos.lowest_liquidity_usd = 2_000
    clean_db.commit()
    _price_at(monkeypatch, 0.90)

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)

    from app.execution.fill_model import price_impact_pct
    exit_notional = 85.0
    assert risk.stress_loss_if_all_stops_hit_usd == pytest.approx(
        risk.loss_if_all_stops_hit_usd
        + exit_notional * (price_impact_pct(exit_notional, 2_000)
                           + settings.PAPER_SPREAD_PCT + settings.PAPER_FEE_PCT)
    )


# ---------------------------------------------------------------------------
# O4/O5 - the prospective trade and the gate
# ---------------------------------------------------------------------------

async def test_a_prospective_position_is_counted_before_it_opens(clean_db, monkeypatch):
    """O4. A trade only prevented after it opens is not prevented."""
    _price_at(monkeypatch, 1.0)

    empty = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)
    assert empty.loss_if_all_stops_hit_usd == pytest.approx(0.0)

    proposed = await open_risk.assess(
        clean_db, remaining_daily_loss_budget_usd=100.0,
        prospective_size_usd=200.0, prospective_stop_loss_pct=0.15,
        prospective_liquidity_usd=1_000_000,
    )
    assert proposed.loss_if_all_stops_hit_usd == pytest.approx(30.0)
    assert proposed.remaining_daily_loss_budget_after_open_risk_usd < 100.0


async def test_the_budget_after_open_risk_can_go_negative(clean_db, monkeypatch):
    """The field is the gate's input, so it must be allowed to report an
    overrun rather than clamping at zero and hiding it."""
    _position(clean_db, qty=1000, entry_price=1.00, stop_loss=0.85, liquidity=1_000_000)
    clean_db.commit()
    _price_at(monkeypatch, 1.00)

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=50.0)
    assert risk.loss_if_all_stops_hit_usd == pytest.approx(150.0)
    assert risk.remaining_daily_loss_budget_after_open_risk_usd < 0


async def test_the_gate_refuses_an_entry_that_would_overrun_the_day(clean_db, monkeypatch):
    """O5, the headline invariant: the bot must never approve a new trade
    when worst-case open risk would breach the daily-loss threshold."""
    monkeypatch.setattr(settings, "RISK_EQUITY_AWARE_DAILY_LOSS", True)
    _position(clean_db, qty=1000, entry_price=1.00, stop_loss=0.85, liquidity=1_000_000)
    clean_db.commit()
    _price_at(monkeypatch, 1.00)

    async def same_price(_addr):
        return 1.00
    monkeypatch.setattr(portfolio.price_feed, "get_price_usd", same_price)

    decision = await RiskManager().evaluate_open_risk(clean_db, prospective_size_usd=100.0)
    assert not decision.allowed
    assert "worst-case open risk" in decision.reason


async def test_the_gate_allows_an_entry_the_budget_can_absorb(clean_db, monkeypatch):
    """The gate has to be passable, or it is a kill switch wearing a
    different name."""
    monkeypatch.setattr(settings, "RISK_EQUITY_AWARE_DAILY_LOSS", True)
    _price_at(monkeypatch, 1.00)

    async def same_price(_addr):
        return 1.00
    monkeypatch.setattr(portfolio.price_feed, "get_price_usd", same_price)

    assert (
        await RiskManager().evaluate_open_risk(clean_db, prospective_size_usd=50.0)
    ).allowed


async def test_the_gate_is_inert_while_the_flag_is_off(clean_db, monkeypatch):
    """O6. Production risk behavior is unchanged by merging this."""
    monkeypatch.setattr(settings, "RISK_EQUITY_AWARE_DAILY_LOSS", False)

    async def explode(*_args, **_kwargs):
        raise AssertionError("open-risk assessment ran while the flag was off")
    monkeypatch.setattr(open_risk, "assess", explode)

    _position(clean_db, qty=100_000, entry_price=1.00, stop_loss=0.85)
    clean_db.commit()

    assert (
        await RiskManager().evaluate_open_risk(clean_db, prospective_size_usd=10_000.0)
    ).allowed


# ---------------------------------------------------------------------------
# O7 - measurement honesty
# ---------------------------------------------------------------------------

async def test_an_unpriced_position_is_reported_as_unmeasurable(clean_db, monkeypatch):
    """CLAUDE.md: a measurement that cannot be taken is recorded as
    unmeasurable. A drawdown computed off a stale mark is not a drawdown
    that was measured."""
    async def no_price(_addr):
        return None
    monkeypatch.setattr(open_risk.price_feed, "get_price_usd", no_price)

    _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85, liquidity=1_000_000)
    clean_db.commit()

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)
    assert risk.unpriced_positions == 1
    assert not risk.measurable


async def test_a_position_with_no_recorded_depth_is_counted_as_such(clean_db, monkeypatch):
    """The fill model falls back to an assumed pool depth. That fallback
    is an assumption, and a stress figure resting on one should say so."""
    _position(clean_db, qty=100, entry_price=1.00, stop_loss=0.85, liquidity=None)
    clean_db.commit()
    _price_at(monkeypatch, 0.90)

    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)
    assert risk.positions_without_recorded_liquidity == 1


async def test_the_unquantifiable_risks_are_named_rather_than_zeroed(clean_db, monkeypatch):
    """O7. A gap through the stop, a liquidity pull and correlated
    stop-outs are all real and none of them can be measured from what this
    bot has recorded. Listing them keeps the stress number honest about
    being a floor rather than a ceiling."""
    _price_at(monkeypatch, 1.00)
    risk = await open_risk.assess(clean_db, remaining_daily_loss_budget_usd=100.0)
    assert len(risk.unmodeled_risks) == 3
    assert any("gap" in r for r in risk.unmodeled_risks)
    assert any("liquidity pull" in r for r in risk.unmodeled_risks)
