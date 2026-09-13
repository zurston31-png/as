"""Integration tests for app/scanner/loop.py.

The property that matters most: a scanner-discovered token goes through the
SAME gates a TradingView alert does. test_scanner_respects_the_rug_check and
test_scanner_respects_the_signal_score_gate prove that directly rather than
trusting that handle_discovered_token delegates correctly - if the scanner
ever grew its own bypassing trade path, those two would fail.
"""
import datetime as dt
import random

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.rugcheck.filters import RugCheckReport
import app.scanner.loop as scanner_loop
import app.services.trading_service as trading_service
from app.scanner.discovery import DiscoveredToken
from app.services import price_feed
from app.signals.scoring import Factor, SignalScore
from tests.conftest import make_market_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _token(address="ScanAddr1", symbol="SCANCOIN") -> DiscoveredToken:
    return DiscoveredToken(
        token_address=address,
        symbol=symbol,
        chain="solana",
        source="dexscreener",
        liquidity_usd=150_000.0,
        volume_24h_usd=300_000.0,
        buys_24h=400,
        sells_24h=300,
        price_usd=0.005,
        price_change_1h_pct=3.0,
        price_change_24h_pct=25.0,
        pair_created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=4),
    )


@pytest.fixture(autouse=True)
def _clean_scanner_rows():
    yield
    db = SessionLocal()
    try:
        for row in db.query(models.ScannedToken).all():
            db.delete(row)
        for sig in db.query(models.Signal).filter_by(source="scanner").all():
            db.delete(sig)
        db.commit()
    finally:
        db.close()


@pytest.fixture
def _happy_path(monkeypatch):
    """Everything downstream says yes, so a discovered token should trade."""
    async def fake_price(instrument):
        return 0.005

    async def fake_rug_check(chain, token_address):
        return RugCheckReport(passed=True, reasons=[], liquidity_usd=150_000.0, dev_wallet_pct=0.02)

    async def fake_score(chain, token_address, symbol):
        return SignalScore(
            score=88.0, direction="long", reliable=True,
            factors=[Factor(name="trend_direction", score=0.9, weight=1.0, reason="scanner test")],
        )

    async def fake_discover(chain=None):
        return [_token()]

    async def fake_snapshot(token_address):
        return make_market_snapshot(token_address=token_address)

    monkeypatch.setattr(price_feed, "get_price_usd", fake_price)
    monkeypatch.setattr(price_feed, "get_market_snapshot", fake_snapshot)
    monkeypatch.setattr(trading_service, "run_rug_checks", fake_rug_check)
    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", fake_score)
    monkeypatch.setattr(scanner_loop, "discover_tokens", fake_discover)
    monkeypatch.setattr(settings, "SCANNER_ENABLED", True)
    monkeypatch.setattr(settings, "SCANNER_RECHECK_MINUTES", 60)
    return monkeypatch


# ---------------------------------------------------------------------------
# the gate that stops auto-buying with real money by accident
# ---------------------------------------------------------------------------

def test_scanner_blocked_when_live_trading_without_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "SCANNER_ENABLED", True)
    monkeypatch.setattr(settings, "LIVE_TRADING", True)
    monkeypatch.setattr(settings, "SCANNER_ALLOW_LIVE_TRADING", False)
    reason = scanner_loop.scanner_blocked_reason()
    assert reason is not None
    assert "SCANNER_ALLOW_LIVE_TRADING" in reason


def test_scanner_allowed_in_live_when_explicitly_opted_in(monkeypatch):
    monkeypatch.setattr(settings, "SCANNER_ENABLED", True)
    monkeypatch.setattr(settings, "LIVE_TRADING", True)
    monkeypatch.setattr(settings, "SCANNER_ALLOW_LIVE_TRADING", True)
    assert scanner_loop.scanner_blocked_reason() is None


def test_scanner_blocked_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "SCANNER_ENABLED", False)
    assert scanner_loop.scanner_blocked_reason() == "SCANNER_ENABLED=false"


async def test_scan_once_does_nothing_when_blocked(monkeypatch):
    monkeypatch.setattr(settings, "SCANNER_ENABLED", False)
    summary = await scanner_loop.scan_once()
    assert summary["skipped"]
    assert summary["traded"] == 0


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

async def test_scanner_opens_a_paper_position_for_a_good_candidate(_happy_path):
    db = SessionLocal()
    try:
        summary = await scanner_loop.scan_once(db)
        assert summary["discovered"] == 1
        assert summary["evaluated"] == 1
        assert summary["traded"] == 1

        position = db.query(models.Position).filter_by(symbol="SCANCOIN", status="open").first()
        assert position is not None

        signal = db.query(models.Signal).filter_by(symbol="SCANCOIN").first()
        assert signal is not None
        assert signal.source == "scanner"
        assert signal.raw_payload["discovery_source"] == "dexscreener"
    finally:
        for row in db.query(models.Position).filter_by(symbol="SCANCOIN").all():
            db.delete(row)
        for row in db.query(models.Trade).filter_by(symbol="SCANCOIN").all():
            db.delete(row)
        db.commit()
        db.close()


async def test_scanner_records_an_audit_row_for_every_candidate(_happy_path):
    db = SessionLocal()
    try:
        await scanner_loop.scan_once(db)
        row = db.query(models.ScannedToken).filter_by(token_address="ScanAddr1").first()
        assert row is not None
        assert row.last_stage == scanner_loop.STAGE_TRADED
        assert row.evaluation_count == 1
        assert row.times_traded == 1
    finally:
        for row in db.query(models.Position).filter_by(symbol="SCANCOIN").all():
            db.delete(row)
        for row in db.query(models.Trade).filter_by(symbol="SCANCOIN").all():
            db.delete(row)
        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# the scanner must not bypass any existing protection
# ---------------------------------------------------------------------------

async def test_scanner_respects_the_rug_check(_happy_path, monkeypatch):
    async def failing_rug_check(chain, token_address):
        return RugCheckReport(passed=False, reasons=["mint authority still active"])

    monkeypatch.setattr(trading_service, "run_rug_checks", failing_rug_check)

    db = SessionLocal()
    try:
        summary = await scanner_loop.scan_once(db)
        assert summary["evaluated"] == 1
        assert summary["traded"] == 0
        assert db.query(models.Position).filter_by(symbol="SCANCOIN", status="open").first() is None
    finally:
        db.close()


async def test_scanner_respects_the_signal_score_gate(_happy_path, monkeypatch):
    async def weak_score(chain, token_address, symbol):
        return SignalScore(
            score=30.0, direction="neutral", reliable=True,
            factors=[Factor(name="trend_direction", score=0.3, weight=1.0, reason="weak")],
        )

    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", weak_score)

    db = SessionLocal()
    try:
        summary = await scanner_loop.scan_once(db)
        assert summary["traded"] == 0
        assert db.query(models.Position).filter_by(symbol="SCANCOIN", status="open").first() is None
    finally:
        db.close()


async def test_scanner_respects_a_trading_halt(_happy_path):
    from app.risk.manager import halt_trading, resume_trading

    db = SessionLocal()
    try:
        halt_trading(db, "unit test halt")
        db.commit()

        summary = await scanner_loop.scan_once(db)
        assert summary["traded"] == 0
        assert db.query(models.Position).filter_by(symbol="SCANCOIN", status="open").first() is None
    finally:
        resume_trading(db)
        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# dedupe / cost control
# ---------------------------------------------------------------------------

async def test_a_token_is_not_re_evaluated_within_the_recheck_window(_happy_path):
    db = SessionLocal()
    try:
        first = await scanner_loop.scan_once(db)
        assert first["evaluated"] == 1

        second = await scanner_loop.scan_once(db)
        assert second["skipped_recent"] == 1
        assert second["evaluated"] == 0
    finally:
        for row in db.query(models.Position).filter_by(symbol="SCANCOIN").all():
            db.delete(row)
        for row in db.query(models.Trade).filter_by(symbol="SCANCOIN").all():
            db.delete(row)
        db.commit()
        db.close()


async def test_a_token_is_re_evaluated_once_the_recheck_window_lapses(_happy_path, monkeypatch):
    db = SessionLocal()
    try:
        await scanner_loop.scan_once(db)
        monkeypatch.setattr(settings, "SCANNER_RECHECK_MINUTES", 0)
        second = await scanner_loop.scan_once(db)
        assert second["skipped_recent"] == 0
    finally:
        for row in db.query(models.Position).filter_by(symbol="SCANCOIN").all():
            db.delete(row)
        for row in db.query(models.Trade).filter_by(symbol="SCANCOIN").all():
            db.delete(row)
        db.commit()
        db.close()


async def test_prescreen_rejects_before_any_expensive_call(_happy_path, monkeypatch):
    """A token that fails the free pre-screen must never reach the rug check
    or the signal score - that ordering is the whole cost model."""
    called = {"rug": False, "score": False}

    async def tracking_rug(chain, token_address):
        called["rug"] = True
        return RugCheckReport(passed=True, reasons=[])

    async def tracking_score(chain, token_address, symbol):
        called["score"] = True
        return None

    async def thin_token(chain=None):
        return [DiscoveredToken(
            token_address="ThinAddr1", symbol="THIN", chain="solana", source="dexscreener",
            liquidity_usd=10.0, volume_24h_usd=5.0, buys_24h=1, sells_24h=1,
            price_usd=0.001, pair_created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2),
        )]

    monkeypatch.setattr(trading_service, "run_rug_checks", tracking_rug)
    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", tracking_score)
    monkeypatch.setattr(scanner_loop, "discover_tokens", thin_token)

    db = SessionLocal()
    try:
        summary = await scanner_loop.scan_once(db)
        assert summary["prescreen_rejected"] == 1
        assert summary["evaluated"] == 0
        assert not called["rug"], "rug check ran on a token that failed the free pre-screen"
        assert not called["score"], "signal score ran on a token that failed the free pre-screen"
    finally:
        db.close()


async def test_max_tokens_per_cycle_is_respected(_happy_path, monkeypatch):
    async def many_tokens(chain=None):
        return [_token(address=f"Addr{i}", symbol=f"COIN{i}") for i in range(20)]

    monkeypatch.setattr(scanner_loop, "discover_tokens", many_tokens)
    monkeypatch.setattr(settings, "SCANNER_MAX_TOKENS_PER_CYCLE", 3)

    db = SessionLocal()
    try:
        await scanner_loop.scan_once(db)
        evaluated_rows = db.query(models.ScannedToken).count()
        assert evaluated_rows <= 3
    finally:
        for row in db.query(models.Position).all():
            db.delete(row)
        for row in db.query(models.Trade).all():
            db.delete(row)
        db.commit()
        db.close()


async def test_one_bad_candidate_does_not_kill_the_whole_cycle(_happy_path, monkeypatch):
    calls = {"n": 0}

    async def flaky_score(chain, token_address, symbol):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure on the first token")
        return SignalScore(
            score=88.0, direction="long", reliable=True,
            factors=[Factor(name="t", score=0.9, weight=1.0, reason="ok")],
        )

    async def two_tokens(chain=None):
        return [_token(address="BadAddr", symbol="BADCOIN"), _token(address="GoodAddr", symbol="GOODCOIN")]

    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", flaky_score)
    monkeypatch.setattr(scanner_loop, "discover_tokens", two_tokens)

    db = SessionLocal()
    try:
        summary = await scanner_loop.scan_once(db)
        # the second token must still have been evaluated and traded
        assert summary["traded"] == 1
        assert db.query(models.Position).filter_by(symbol="GOODCOIN", status="open").first() is not None
    finally:
        for row in db.query(models.Position).all():
            db.delete(row)
        for row in db.query(models.Trade).all():
            db.delete(row)
        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# prescreen counterfactual coverage
# ---------------------------------------------------------------------------

def _rejected_token(address="ThinAddr1") -> DiscoveredToken:
    """A token the prescreen will turn down - far too illiquid."""
    return DiscoveredToken(
        token_address=address, symbol="THIN", chain="solana", source="dexscreener",
        liquidity_usd=200.0, volume_24h_usd=50.0, buys_24h=2, sells_24h=1,
        price_usd=0.004, price_change_1h_pct=0.0, price_change_24h_pct=0.0,
        pair_created_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=4),
    )


class _AlwaysSample(random.Random):
    def random(self):
        return 0.0        # below any positive rate


class _NeverSample(random.Random):
    def random(self):
        return 0.999999   # above any rate below 1.0


async def test_a_sampled_prescreen_reject_is_followed_forward(monkeypatch):
    """The prescreen rejects more candidates than every other gate
    combined, and until now none were followed - so the counterfactual
    could say nothing at all about the biggest filter in the pipeline. "No
    data" and "rejects nothing worth having" were indistinguishable."""
    monkeypatch.setattr(settings, "SCANNER_TRACK_PRESCREEN_REJECTS", True)
    monkeypatch.setattr(settings, "SCANNER_PRESCREEN_TRACKING_RATE", 0.10)

    async def one_token():
        return [_rejected_token()]

    monkeypatch.setattr(scanner_loop, "discover_tokens", one_token)

    db = SessionLocal()
    try:
        before = db.query(models.ForwardReturn).count()
        summary = await scanner_loop.scan_once(db=db, rng=_AlwaysSample())
        db.commit()

        assert summary["prescreen_rejected"] == 1
        assert summary["prescreen_tracked"] == 1
        rows = db.query(models.ForwardReturn).all()[before:]
        assert {r.horizon_minutes for r in rows} == set(scanner_loop.PRESCREEN_HORIZONS)
        # No score: these tokens never reached the scorer, and every
        # calibration query filters on a non-null score - so they serve the
        # counterfactual without entering a table describing a different
        # population.
        assert all(r.score is None for r in rows)
        assert all(r.price_at_signal == 0.004 for r in rows)
    finally:
        db.close()


async def test_an_unsampled_prescreen_reject_costs_nothing(monkeypatch):
    """Tracking every reject would multiply the bot's largest source of API
    load by the rejection rate, which is most of the flow."""
    monkeypatch.setattr(settings, "SCANNER_TRACK_PRESCREEN_REJECTS", True)
    monkeypatch.setattr(settings, "SCANNER_PRESCREEN_TRACKING_RATE", 0.10)

    async def one_token():
        return [_rejected_token("ThinAddr2")]

    monkeypatch.setattr(scanner_loop, "discover_tokens", one_token)

    db = SessionLocal()
    try:
        before = db.query(models.ForwardReturn).count()
        summary = await scanner_loop.scan_once(db=db, rng=_NeverSample())
        db.commit()

        assert summary["prescreen_rejected"] == 1
        assert summary["prescreen_tracked"] == 0
        assert db.query(models.ForwardReturn).count() == before
    finally:
        db.close()


async def test_tracking_can_be_switched_off_entirely(monkeypatch):
    monkeypatch.setattr(settings, "SCANNER_TRACK_PRESCREEN_REJECTS", False)

    async def one_token():
        return [_rejected_token("ThinAddr3")]

    monkeypatch.setattr(scanner_loop, "discover_tokens", one_token)

    db = SessionLocal()
    try:
        before = db.query(models.ForwardReturn).count()
        summary = await scanner_loop.scan_once(db=db, rng=_AlwaysSample())
        db.commit()
        assert summary["prescreen_tracked"] == 0
        assert db.query(models.ForwardReturn).count() == before
    finally:
        db.close()


async def test_a_reject_with_no_price_is_never_tracked(monkeypatch):
    """Every forward return divides by the signal price. A zero there is
    corrupt data, not a free option."""
    monkeypatch.setattr(settings, "SCANNER_TRACK_PRESCREEN_REJECTS", True)

    token = _rejected_token("ThinAddr4")
    token.price_usd = 0.0

    async def one_token():
        return [token]

    monkeypatch.setattr(scanner_loop, "discover_tokens", one_token)

    db = SessionLocal()
    try:
        before = db.query(models.ForwardReturn).count()
        await scanner_loop.scan_once(db=db, rng=_AlwaysSample())
        db.commit()
        assert db.query(models.ForwardReturn).count() == before
    finally:
        db.close()


def test_the_sample_is_random_rather_than_the_first_n():
    """The first tokens a provider happens to list are not a random draw
    from the ones it rejects - they are ordered by whatever the provider
    sorts on, which is usually correlated with exactly the properties
    being measured."""
    import inspect

    body = inspect.getsource(scanner_loop._track_prescreen_reject)
    assert "rng.random()" in body


async def test_sampling_a_reject_does_not_add_a_second_prescreen_event(monkeypatch):
    """Regression. The reject sampler used to write its own PRESCREEN row to
    anchor the forward returns, so a sampled reject produced TWO prescreen
    events for one evaluation.

    The funnel counts events, so PRESCREEN read higher than DISCOVERED -
    197 discovered against 208 prescreened in the field - and every
    prescreen pass rate was computed against an inflated denominator. The
    sampling exists to observe the population, not to change the count of
    it, so it now anchors to the event the loop already recorded.
    """
    from app.analysis.stage_funnel import build_stage_funnel

    monkeypatch.setattr(settings, "SCANNER_TRACK_PRESCREEN_REJECTS", True)
    monkeypatch.setattr(settings, "SCANNER_PRESCREEN_TRACKING_RATE", 1.0)

    async def one_token():
        return [_rejected_token()]

    monkeypatch.setattr(scanner_loop, "discover_tokens", one_token)

    db = SessionLocal()
    try:
        for row in db.query(models.PipelineEvent).all():
            db.delete(row)
        db.commit()
        # Earlier tests in this module follow the SAME mint forward and
        # leave their rows behind, so only rows created below can be
        # attributed to this scan.
        before = db.query(models.ForwardReturn).count()

        summary = await scanner_loop.scan_once(db=db, rng=_AlwaysSample())
        db.commit()
        assert summary["prescreen_tracked"] == 1, "the sampler must still have run"

        funnel = build_stage_funnel(db, window_hours=None)
        discovered = funnel.stage("DISCOVERED")
        prescreen = funnel.stage("PRESCREEN")

        assert prescreen.entered == 1, "one evaluation must record exactly one prescreen event"
        assert prescreen.entered == discovered.entered, (
            "prescreen cannot exceed discovered - every discovered token is "
            "prescreened exactly once per evaluation"
        )

        # And the forward returns are still anchored to a real prescreen row.
        anchored = db.query(models.ForwardReturn).all()[before:]
        assert anchored, "the sampled reject must still be followed forward"
        event_ids = {r.pipeline_event_id for r in anchored}
        prescreen_ids = {
            e.id for e in db.query(models.PipelineEvent).filter_by(stage="PRESCREEN").all()
        }
        assert event_ids <= prescreen_ids, "forward returns must point at the real prescreen event"
    finally:
        for row in db.query(models.PipelineEvent).all():
            db.delete(row)
        db.commit()
        db.close()
