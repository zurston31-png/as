"""Integration tests for the watchlist re-evaluation loop.

WATCH is only worth having if something keeps looking, and "something keeps
looking" is a claim about a background loop rather than about a scoring
function. These tests drive evaluate_once against real database rows.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.early import loop as early_loop
from app.early import watchlist as wl
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
                          models.PipelineEvent, models.ForwardReturn):
                for row in db.query(model).all():
                    db.delete(row)
            for sig in db.query(models.Signal).filter(
                models.Signal.source.in_(["scanner", "early_signal"])
            ).all():
                db.delete(sig)
            db.commit()
        finally:
            db.close()
    clean()
    yield
    clean()


@pytest.fixture
def watched(monkeypatch):
    """One WATCH entry, with the outside world stubbed."""
    from app.services import price_feed

    async def snapshot(token_address):
        return make_market_snapshot(token_address=token_address)

    monkeypatch.setattr(price_feed, "get_market_snapshot", snapshot)
    monkeypatch.setattr(settings, "EARLY_SIGNAL_ENABLED", True)

    # An entry at or above the watch threshold triggers a candle fetch and a
    # technical re-score. Left unstubbed those reach the network, which makes
    # the test slow and its result dependent on a provider being up.
    from app.data.candles import Timeframe
    from app.data.providers import SyntheticCandleProvider
    import app.data.live_provider as live_provider
    import app.signals.live_gate as live_gate

    series = SyntheticCandleProvider(regime="bull", seed=13).fetch(
        "WATCHED", Timeframe.M5, limit=300
    )

    async def fake_candles(*args, **kwargs):
        return series

    async def fake_gate(*args, **kwargs):
        return None

    monkeypatch.setattr(live_provider, "fetch_candles", fake_candles)
    monkeypatch.setattr(live_gate, "evaluate_live_entry_signal", fake_gate)

    db = SessionLocal()
    entry = models.WatchlistEntry(
        token_address="WatchMint1", symbol="WATCHED", chain="solana",
        state=wl.WATCH, first_seen_at=dt.datetime.now(dt.timezone.utc),
        early_score=60.0, best_early_score=60.0,
        score_history=[], features={},
    )
    db.add(entry)
    db.commit()
    db.close()
    return monkeypatch


async def test_a_pass_re_evaluates_every_watched_token(watched):
    summary = await early_loop.evaluate_once()
    assert summary["evaluated"] == 1

    db = SessionLocal()
    try:
        entry = wl.get(db, "WatchMint1")
        assert entry.evaluations >= 1
        assert entry.score_history, "the pass recorded no score point"
    finally:
        db.close()


async def test_a_pass_stores_an_observation_so_flow_becomes_measurable(watched):
    """Flow features need two snapshots at least a minute apart. If the loop
    did not store one on every pass they would never become computable."""
    await early_loop.evaluate_once()

    db = SessionLocal()
    try:
        assert db.query(models.TokenObservation).filter_by(
            token_address="WatchMint1"
        ).count() == 1
    finally:
        db.close()


async def test_a_missing_snapshot_does_not_fail_the_entry(watched, monkeypatch):
    """A fetch that returned nothing is not a low score.

    Failing the entry on a missing snapshot would let one bad minute from
    the price feed empty the whole watchlist, and the false-positive
    analysis would then be full of failures caused by the bot's own
    plumbing rather than by the tokens.
    """
    from app.services import price_feed

    async def nothing(token_address):
        return None

    monkeypatch.setattr(price_feed, "get_market_snapshot", nothing)

    await early_loop.evaluate_once()

    db = SessionLocal()
    try:
        entry = wl.get(db, "WatchMint1")
        assert entry.state == wl.WATCH, "a missing fetch retired a live entry"
        assert entry.failure_category is None
    finally:
        db.close()


async def test_a_stale_entry_expires_rather_than_lingering(watched):
    db = SessionLocal()
    try:
        entry = wl.get(db, "WatchMint1")
        entry.first_seen_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            hours=settings.WATCHLIST_MAX_AGE_HOURS + 1
        )
        db.commit()
    finally:
        db.close()

    summary = await early_loop.evaluate_once()
    assert summary["expired"] == 1

    db = SessionLocal()
    try:
        entry = wl.get(db, "WatchMint1")
        assert entry.state == wl.EXPIRED
        assert entry.failure_category == "expired", (
            "an expired entry must stay in the false-positive analysis"
        )
    finally:
        db.close()


async def test_the_loop_does_nothing_when_the_engine_is_disabled(watched, monkeypatch):
    monkeypatch.setattr(settings, "EARLY_SIGNAL_ENABLED", False)

    summary = await early_loop.evaluate_once()

    assert summary["evaluated"] == 0
    assert "EARLY_SIGNAL_ENABLED=false" in summary["skipped"]


async def test_a_confirmation_goes_through_the_normal_buy_path(watched, monkeypatch):
    """Never a parallel trading path.

    Every protection the bot has - kill switch, risk gate, rug check,
    market quality, exposure caps, the fill model - lives behind
    handle_discovered_token. A second entry route would eventually drift
    from it and lose one of them silently.
    """
    from app.early.engine import Decision
    import app.services.trading_service as trading_service

    calls = []

    async def fake_entry(db, **kwargs):
        calls.append(kwargs)
        return models.Signal(
            symbol=kwargs["symbol"], token_address=kwargs["token_address"],
            chain=kwargs["chain"], signal_type="buy", price=kwargs["price"],
            source="early_signal",
        )

    monkeypatch.setattr(trading_service, "handle_discovered_token", fake_entry)

    class _Confirmed:
        decision = Decision.PAPER_BUY
        early_score = 82.0
        technical_score = 71.0
        security_score = 20.0
        market_quality_score = 80.0
        late_risk = 12.0
        stage = None
        momentum = None
        features = None
        early = None
        reason = "confirmed by the test"

    async def fake_evaluate(**kwargs):
        return _Confirmed()

    monkeypatch.setattr(early_loop, "evaluate", lambda **kwargs: _Confirmed())

    await early_loop.evaluate_once()

    assert len(calls) == 1, "a confirmation did not reach the standard buy path"
    assert calls[0]["discovery_source"] == "early_signal"
    assert calls[0]["token_address"] == "WatchMint1"
