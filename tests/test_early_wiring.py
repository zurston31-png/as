"""Is the Early Signal Engine actually REACHED by the live buy path?

Every other early-signal test drives the engine directly. That proves the
engine works and proves nothing about whether the bot ever calls it - a
wiring mistake would leave all of them green while the watchlist stayed
permanently empty and every ForwardReturn.early_score stayed NULL. These
tests drive the real scanner entry point instead.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.rugcheck.filters import RugCheckReport
import app.services.trading_service as trading_service
from app.signals.scoring import Factor, SignalScore
from tests.conftest import make_market_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _wipe():
    def clean():
        db = SessionLocal()
        try:
            for model in (models.WatchlistEntry, models.TokenObservation,
                          models.ForwardReturn, models.PipelineEvent):
                for row in db.query(model).all():
                    db.delete(row)
            for sig in db.query(models.Signal).filter_by(source="scanner").all():
                db.delete(sig)
            db.commit()
        finally:
            db.close()
    clean()
    yield
    clean()


def _score(value: float) -> SignalScore:
    return SignalScore(
        score=value, direction="long", reliable=True,
        factors=[Factor(name="trend_direction", score=0.5, weight=1.0, reason="stub")],
    )


@pytest.fixture
def wired(monkeypatch):
    """Stub only the OUTSIDE world - every gate inside the bot runs for real."""
    from app.services import price_feed

    async def snapshot(token_address):
        return make_market_snapshot(token_address=token_address)

    monkeypatch.setattr(price_feed, "get_market_snapshot", snapshot)

    # The early engine needs candle history to score anything at all - with
    # none it fails closed and skips, which is correct behaviour but would
    # make these tests pass for the wrong reason. A generated bull series
    # stands in for the live provider; it exercises the wiring and says
    # nothing about real markets.
    from app.data.candles import Timeframe
    from app.data.providers import SyntheticCandleProvider
    import app.data.live_provider as live_provider

    series = SyntheticCandleProvider(regime="bull", seed=11).fetch(
        "EARLYCOIN", Timeframe.M5, limit=300
    )

    async def fake_candles(*args, **kwargs):
        return series

    monkeypatch.setattr(live_provider, "fetch_candles", fake_candles)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_ENABLED", True)
    monkeypatch.setattr(settings, "LIVE_SIGNAL_SCORE_ENABLED", True)
    monkeypatch.setattr(settings, "FORWARD_RETURNS_ENABLED", True)
    return monkeypatch


def _stub_gates(monkeypatch, *, security_passed: bool, technical: float):
    async def rug(*args, **kwargs):
        return RugCheckReport(
            passed=security_passed,
            reasons=[] if security_passed else ["mint authority still live"],
            liquidity_usd=150_000.0,
            rug_risk_score=20.0 if security_passed else 92.0,
        )

    async def gate(*args, **kwargs):
        return _score(technical)

    monkeypatch.setattr(trading_service, "run_rug_checks", rug)
    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", gate)


async def _discover(symbol="EARLYCOIN", address="EarlyMint111"):
    db = SessionLocal()
    try:
        await trading_service.handle_discovered_token(
            db, symbol=symbol, token_address=address, chain="solana",
            price=0.004, discovery_source="dexscreener",
        )
        db.commit()
    finally:
        db.close()


async def test_a_technical_rejection_still_reaches_the_early_engine(wired):
    """The reason the engine exists: a candidate the technical gate turns
    down should be WATCHED, not discarded."""
    _stub_gates(wired, security_passed=True,
                technical=settings.MIN_SIGNAL_SCORE_TO_ENTER - 15)

    await _discover()

    db = SessionLocal()
    try:
        entries = db.query(models.WatchlistEntry).all()
        assert len(entries) == 1, "the early engine was never reached"
        assert entries[0].token_address == "EarlyMint111"
        assert entries[0].early_score is not None
        assert entries[0].score_history, "no score point was recorded"
    finally:
        db.close()


async def test_an_observation_is_stored_so_flow_is_measurable_next_pass(wired):
    """Flow features need two snapshots. If nothing is stored on the first
    pass they are never computable on any pass."""
    _stub_gates(wired, security_passed=True, technical=60.0)

    await _discover()

    db = SessionLocal()
    try:
        rows = db.query(models.TokenObservation).all()
        assert len(rows) == 1
        assert rows[0].token_address == "EarlyMint111"
        assert rows[0].liquidity_usd is not None
    finally:
        db.close()


async def test_the_early_score_lands_on_the_forward_return_rows(wired):
    """Without this the early calibration reads an empty dataset while
    looking like it is working - the failure mode is silent."""
    _stub_gates(wired, security_passed=True, technical=60.0)

    await _discover()

    db = SessionLocal()
    try:
        rows = db.query(models.ForwardReturn).all()
        assert rows, "no forward returns were scheduled at all"
        assert all(r.early_score is not None for r in rows), (
            "forward returns were scheduled but never got the early verdict"
        )
        assert all(r.early_features for r in rows), (
            "features were not stored, so the ablation can never run"
        )
    finally:
        db.close()


async def test_a_security_failure_produces_no_watchlist_entry(wired):
    """An excellent early signal must never override a critical security
    failure. The engine is not consulted at all, so there is no early score
    that could override anything."""
    _stub_gates(wired, security_passed=False, technical=95.0)

    await _discover()

    db = SessionLocal()
    try:
        assert db.query(models.WatchlistEntry).count() == 0
        assert db.query(models.Position).filter_by(
            token_address="EarlyMint111"
        ).count() == 0
        scored = db.query(models.ForwardReturn).all()
        assert all(r.early_score is None for r in scored), (
            "a token that failed security got an early score attached"
        )
    finally:
        db.close()


async def test_the_early_engine_never_blocks_a_trade_the_strategy_wanted(wired):
    """The engine's only power is to ADD a watch, never to remove an entry.
    A crash inside it must not cost a position the strategy had approved."""
    def explode(*args, **kwargs):
        raise RuntimeError("early engine blew up")

    _stub_gates(wired, security_passed=True,
                technical=settings.MIN_SIGNAL_SCORE_TO_ENTER + 10)
    wired.setattr(trading_service, "_consider_for_watchlist", explode)

    await _discover()

    db = SessionLocal()
    try:
        assert db.query(models.Position).filter_by(
            token_address="EarlyMint111", status=models.PositionStatus.OPEN.value
        ).count() == 1, "an early-engine crash cancelled a valid entry"
    finally:
        db.close()


def test_the_candle_fetch_the_early_path_imports_actually_exists():
    """A regression guard for a bug a broad `except Exception` hid.

    Both early-signal candle fetches imported `fetch_live_series`, which
    has never existed in app/data/live_provider.py. The ImportError was
    caught by the surrounding handler, the series stayed None, and every
    candidate then failed the data-quality gate and was skipped - so the
    engine ran, logged nothing alarming, and could never score anything.
    Asserting the callable resolves is cheap; discovering this from an
    empty watchlist weeks later is not.
    """
    import inspect

    from app.data import live_provider

    assert hasattr(live_provider, "fetch_candles")
    assert inspect.iscoroutinefunction(live_provider.fetch_candles)

    for module in ("app/services/trading_service.py", "app/early/loop.py"):
        source = open(module).read()
        assert "fetch_live_series" not in source, f"{module} still imports a name that does not exist"
