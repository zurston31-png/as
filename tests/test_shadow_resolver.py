"""Tests for the shadow outcome resolver.

The resolver is the piece that turns recorded decisions into evidence, so
the properties worth testing hardest are the ones that decide whether that
evidence is trustworthy: no look-ahead, no invented prices, the same
numbers on a second run, and no reach into anything the paper account
owns.

Every test drives the walk with fixed candles through the injected `fetch`.
A test that needed a live feed would be testing the network.
"""
import datetime as dt
import pathlib

import pytest

from app import models
from app.config import settings
from app.data.candles import Candle, CandleSeries, Timeframe
from app.database import SessionLocal
from app.shadow import resolver
from app.shadow.exit_policy import ExitPolicy, walk

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


OPENED = dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
TF = Timeframe.M5
ENTRY = 1.00


def bar(minutes_after: int, o, h, l, c, volume=1000.0) -> Candle:
    return Candle(
        timestamp=OPENED + dt.timedelta(minutes=minutes_after),
        open=o, high=h, low=l, close=c, volume=volume,
    )


def series(candles) -> CandleSeries:
    return CandleSeries("SHDW", TF, list(candles))


def feed(candles):
    async def fetch(chain, token, symbol, timeframe, limit):
        return series(candles)
    return fetch


def no_feed():
    async def fetch(chain, token, symbol, timeframe, limit):
        return None
    return fetch


POLICY = ExitPolicy(
    stop_loss_pct=0.15, take_profit_pct=0.30,
    trailing_enabled=False, trailing_activation_pct=0.15, trailing_distance_pct=0.10,
    break_even_enabled=False, break_even_trigger_pct=0.10, break_even_buffer_pct=0.01,
    max_hold_hours=4.0,
)


@pytest.fixture
def db():
    session = SessionLocal()

    def wipe():
        for model in (models.ShadowHorizonReturn, models.ShadowPosition, models.ShadowDecision):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


@pytest.fixture(autouse=True)
def shadow_settings(monkeypatch):
    monkeypatch.setattr(settings, "SHADOW_ENABLED", True)
    monkeypatch.setattr(settings, "SHADOW_RESOLVER_ENABLED", True)
    monkeypatch.setattr(settings, "SHADOW_RESOLUTION_TIMEFRAME", "5m")
    monkeypatch.setattr(settings, "SHADOW_HORIZONS_MINUTES", "15,60")
    monkeypatch.setattr(settings, "SHADOW_UNMEASURABLE_AFTER_HOURS", 12.0)
    monkeypatch.setattr(settings, "STOP_LOSS_PCT", 0.15)
    monkeypatch.setattr(settings, "TAKE_PROFIT_PCT", 0.30)
    monkeypatch.setattr(settings, "TRAILING_STOP_ENABLED", False)
    monkeypatch.setattr(settings, "BREAK_EVEN_ENABLED", False)
    monkeypatch.setattr(settings, "MAX_POSITION_AGE_HOURS", 4.0)


def make_position(db, *, strategy_id="champion", token="ShadowMint1", entry=ENTRY,
                  opened_at=OPENED, fees=0.0025, slippage=0.0):
    decision = models.ShadowDecision(
        opportunity_id="opp-1", strategy_id=strategy_id, strategy_version="v-test",
        is_champion=strategy_id == "champion", token_address=token, symbol="SHDW",
        chain="solana", decision="BUY", reason="test", entry_price=entry,
        fill_succeeded=True, fee_pct=fees, slippage_pct=slippage, size_usd=100.0,
        decided_at=opened_at,
    )
    db.add(decision)
    db.flush()
    position = models.ShadowPosition(
        decision_id=decision.id, opportunity_id="opp-1", strategy_id=strategy_id,
        token_address=token, symbol="SHDW", opened_at=opened_at, entry_price=entry,
        size_usd=100.0, fees_pct=fees, slippage_pct=slippage,
    )
    db.add(position)
    db.flush()
    return position


# ---------------------------------------------------------------------------
# isolation - same guarantee the recorder has, and for the same reason
# ---------------------------------------------------------------------------

def test_the_resolver_cannot_reach_execution_or_champion_state():
    """A resolver that could touch the paper account would be able to move
    real balances from a hypothetical outcome. Enforced by grep because the
    edit that granted it would look harmless in review."""
    forbidden = (
        "get_execution_client", "LIVE_TRADING", "risk_manager", "adjust_cash",
        "models.Position", "models.Trade", "models.RiskEvent",
    )
    body = "\n".join(
        line for line in pathlib.Path("app/shadow/resolver.py").read_text().splitlines()
        if not line.strip().startswith("#")
    )
    assert [t for t in forbidden if t in body] == []


async def test_resolving_never_touches_a_real_position_or_trade(db):
    before = (db.query(models.Position).count(), db.query(models.Trade).count())
    make_position(db)
    await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(hours=1),
        fetch=feed([bar(0, 1.0, 1.4, 0.99, 1.35)]),
    )
    assert (db.query(models.Position).count(), db.query(models.Trade).count()) == before


# ---------------------------------------------------------------------------
# the exit walk
# ---------------------------------------------------------------------------

def test_a_bar_that_touches_both_levels_is_recorded_as_the_stop():
    """Intrabar order is unknowable. Assuming the target came first would
    manufacture winners out of ambiguity, and nothing in the resulting
    numbers would show it had happened."""
    result = walk(
        POLICY, entry_price=ENTRY, opened_at=OPENED,
        candles=[bar(0, 1.0, 1.35, 0.80, 1.20)], timeframe=TF,
    )
    assert result.exit_reason == "stop-loss"
    assert result.exit_price == pytest.approx(0.85)


def test_a_gap_through_the_stop_fills_at_the_open_not_at_the_stop():
    """The price was never available between the previous close and the
    open. Filling at the stop level would credit the trade with liquidity
    that did not exist."""
    result = walk(
        POLICY, entry_price=ENTRY, opened_at=OPENED,
        candles=[bar(0, 0.70, 0.75, 0.60, 0.65)], timeframe=TF,
    )
    assert result.exit_price == pytest.approx(0.70)


def test_the_trailing_stop_only_takes_effect_on_a_later_bar():
    """Ratcheting from a bar's own high and then testing that bar's low
    against the new level would exit on a price the trade never saw."""
    policy = ExitPolicy(**{**POLICY.__dict__, "trailing_enabled": True})
    # One bar runs up 20% (arming the trail at 15%) and falls back to +8%.
    # The trail sits at 1.20 * 0.90 = 1.08, above that bar's low of 1.05 -
    # but it was not armed while the bar was forming.
    result = walk(
        policy, entry_price=ENTRY, opened_at=OPENED,
        candles=[bar(0, 1.0, 1.20, 1.05, 1.08)], timeframe=TF,
    )
    assert result.exit_price is None

    result = walk(
        policy, entry_price=ENTRY, opened_at=OPENED,
        candles=[bar(0, 1.0, 1.20, 1.05, 1.08), bar(5, 1.08, 1.09, 1.00, 1.02)],
        timeframe=TF,
    )
    assert result.exit_reason == "trailing stop"
    assert result.exit_price == pytest.approx(1.08)


def test_the_maximum_hold_closes_at_the_last_price_before_the_deadline():
    candles = [bar(m, 1.0, 1.02, 0.98, 1.01) for m in range(0, 300, 5)]
    result = walk(POLICY, entry_price=ENTRY, opened_at=OPENED, candles=candles, timeframe=TF)
    assert result.exit_reason.startswith("max holding time")
    assert result.exit_at == OPENED + dt.timedelta(hours=4)


def test_the_envelope_comes_from_bar_extremes_not_closes():
    """A drawdown that happened inside a bar still happened. Sampling
    closes would report a stop-free path for a trade that was stopped."""
    result = walk(
        POLICY, entry_price=ENTRY, opened_at=OPENED,
        candles=[bar(0, 1.0, 1.10, 0.90, 1.00), bar(5, 1.0, 1.12, 0.95, 1.05)],
        timeframe=TF,
    )
    assert result.max_favorable_pct == pytest.approx(12.0)
    assert result.max_adverse_pct == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# look-ahead
# ---------------------------------------------------------------------------

async def test_a_bar_that_has_not_closed_yet_is_invisible(db):
    """The whole point of resolving from candles is that the simulated
    moment sees only what had happened by then. A forming bar knows the
    future relative to that moment."""
    position = make_position(db)
    # The 12:00 bar closes at 12:05 and would trip the stop. At 12:03 it is
    # still forming.
    candles = [bar(0, 1.0, 1.05, 0.50, 0.60)]
    await resolver.resolve_once(db, now=OPENED + dt.timedelta(minutes=3), fetch=feed(candles))
    assert position.closed_at is None

    await resolver.resolve_once(db, now=OPENED + dt.timedelta(minutes=6), fetch=feed(candles))
    assert position.exit_reason == "stop-loss"


async def test_the_bar_containing_the_entry_is_excluded(db):
    """Its low may have printed before the position existed. Counting it
    would report a drawdown the trade never took."""
    position = make_position(db, opened_at=OPENED + dt.timedelta(minutes=3))
    candles = [bar(0, 1.0, 1.02, 0.50, 1.00), bar(5, 1.0, 1.03, 0.99, 1.01)]
    await resolver.resolve_once(db, now=OPENED + dt.timedelta(minutes=30), fetch=feed(candles))
    assert position.closed_at is None
    assert position.max_adverse_pct == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------

async def test_the_exit_is_charged_the_same_cost_the_entry_was(db):
    """Not a fresh simulation: the entry price already embeds one leg, and
    a second leg drawn independently would make the round trip depend on
    when the resolver happened to run."""
    position = make_position(db, fees=0.003, slippage=0.007)
    await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(hours=1),
        fetch=feed([bar(0, 1.0, 1.40, 0.99, 1.35)]),
    )
    assert position.gross_return_pct == pytest.approx(30.0)
    assert position.return_pct == pytest.approx(30.0 - 1.0)


# ---------------------------------------------------------------------------
# idempotency and restart safety
# ---------------------------------------------------------------------------

async def test_running_twice_produces_identical_records(db):
    position = make_position(db)
    candles = [bar(0, 1.0, 1.10, 0.98, 1.05), bar(5, 1.05, 1.45, 1.04, 1.40)]
    fields = ("return_pct", "gross_return_pct", "exit_price", "closed_at",
              "hold_minutes", "max_favorable_pct", "max_adverse_pct", "exit_reason")

    now = OPENED + dt.timedelta(hours=2)
    await resolver.resolve_once(db, now=now, fetch=feed(candles))
    db.commit()
    first = {f: getattr(position, f) for f in fields}

    second_pass = await resolver.resolve_once(db, now=now, fetch=feed(candles))
    db.commit()
    assert {f: getattr(position, f) for f in fields} == first
    # A closed position is out of the working set entirely.
    assert second_pass["closed"] == 0


async def test_state_lives_in_the_database_so_a_restart_resumes(db):
    """No in-memory cursor. A fresh session must pick up exactly where the
    interrupted one left off."""
    position = make_position(db)
    await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(minutes=30),
        fetch=feed([bar(0, 1.0, 1.05, 0.98, 1.02)]),
    )
    db.commit()
    assert position.closed_at is None

    other = SessionLocal()
    try:
        summary = await resolver.resolve_once(
            other, now=OPENED + dt.timedelta(hours=1),
            fetch=feed([bar(0, 1.0, 1.05, 0.98, 1.02), bar(30, 1.02, 1.45, 1.01, 1.40)]),
        )
        other.commit()
        assert summary["closed"] == 1
    finally:
        other.close()
    db.expire_all()
    assert db.query(models.ShadowPosition).one().exit_reason == "take-profit"


async def test_horizon_returns_are_recorded_once_each(db):
    position = make_position(db)
    candles = [bar(m, 1.0, 1.02, 0.99, 1.0 + m / 1000) for m in range(0, 120, 5)]
    now = OPENED + dt.timedelta(hours=2)

    await resolver.resolve_once(db, now=now, fetch=feed(candles))
    db.commit()
    await resolver.resolve_once(db, now=now, fetch=feed(candles))
    db.commit()

    rows = db.query(models.ShadowHorizonReturn).filter_by(position_id=position.id).all()
    assert sorted(r.horizon_minutes for r in rows) == [15, 60]


# ---------------------------------------------------------------------------
# refusals - the things that would turn a dead token into a flat one
# ---------------------------------------------------------------------------

async def test_a_horizon_that_has_not_elapsed_is_not_recorded(db):
    make_position(db)
    await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(minutes=10),
        fetch=feed([bar(0, 1.0, 1.02, 0.99, 1.01)]),
    )
    assert db.query(models.ShadowHorizonReturn).count() == 0


# Trading stops ten minutes in, so the 15m horizon still has a quote within
# tolerance and the 60m horizon has nothing but a stale one.
QUIET_FEED = [bar(m, 1.0, 1.02, 0.99, 1.01) for m in (0, 5)]


async def test_a_horizon_with_no_quote_yet_is_left_for_a_later_pass(db):
    """Absent, not zero and not "unmeasurable". The feed may still catch
    up, and writing anything now would either fabricate a number or close
    the door on the real one."""
    make_position(db)
    await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(hours=2), fetch=feed(QUIET_FEED)
    )
    db.commit()

    rows = {r.horizon_minutes: r for r in db.query(models.ShadowHorizonReturn).all()}
    assert set(rows) == {15}
    assert rows[15].return_pct is not None


async def test_a_stale_quote_is_refused_rather_than_reused(db):
    """A close from fifty minutes before the horizon is the price from
    before the feed went quiet. Passing it off as the horizon's price is
    exactly how a dead token becomes a flat one in the dataset."""
    make_position(db)
    # Past the give-up window, so the 60m horizon is filed rather than retried.
    await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(hours=14), fetch=feed(QUIET_FEED)
    )
    db.commit()

    rows = {r.horizon_minutes: r for r in db.query(models.ShadowHorizonReturn).all()}
    assert rows[60].return_pct is None
    assert rows[60].price_at_horizon is None
    assert "no quote" in rows[60].failure_reason


async def test_a_position_with_no_candles_stays_open_before_it_is_abandoned(db):
    position = make_position(db)
    summary = await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(hours=2), fetch=no_feed()
    )
    assert summary["no_candles"] == 1
    assert position.closed_at is None
    assert position.return_pct is None


async def test_an_unmeasurable_position_is_abandoned_with_a_null_return(db):
    """Closed so it stops being retried, but never given a number. "Entered
    and unknown" and "entered and broke even" are different facts and every
    consumer already treats a NULL return as unresolved."""
    position = make_position(db)
    summary = await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(hours=48), fetch=no_feed()
    )
    assert summary["abandoned"] == 1
    assert position.closed_at is not None
    assert position.return_pct is None
    assert "unmeasurable" in position.exit_reason


async def test_the_resolver_does_nothing_when_it_is_switched_off(db, monkeypatch):
    monkeypatch.setattr(settings, "SHADOW_RESOLVER_ENABLED", False)
    position = make_position(db)
    await resolver.resolve_once(
        db, now=OPENED + dt.timedelta(hours=1),
        fetch=feed([bar(0, 1.0, 1.40, 0.99, 1.35)]),
    )
    assert position.closed_at is None


def test_the_default_candle_source_is_a_function_that_exists():
    """A regression guard with history: an earlier module imported
    `fetch_live_series`, a name that had never existed, and a broad
    `except Exception` swallowed the ImportError so the feature simply
    never ran while every test stayed green."""
    from app.data import live_provider

    assert callable(getattr(live_provider, "fetch_candles", None))


async def test_the_resolver_falls_back_to_that_source_when_none_is_injected(db, monkeypatch):
    """The injected `fetch` is a test seam, not the production path. Without
    this, the default branch could be broken indefinitely and only the
    seam would ever be exercised."""
    calls = []

    async def spy(chain, token, symbol, timeframe, limit):
        calls.append((chain, token, timeframe))
        return series([bar(0, 1.0, 1.40, 0.99, 1.35)])

    monkeypatch.setattr("app.data.live_provider.fetch_candles", spy)
    position = make_position(db)
    await resolver.resolve_once(db, now=OPENED + dt.timedelta(hours=1))

    assert calls == [("solana", "ShadowMint1", Timeframe.M5)]
    assert position.exit_reason == "take-profit"
