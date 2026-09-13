"""Tests for app/analysis/funnel.py and app/analysis/token_detail.py.

Together these answer "why did the bot do (or not do) that?". The funnel
answers it in aggregate; the token detail answers it for one mint.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.funnel import STAGE_EVENTS, build_funnel
from app.analysis.token_detail import build_token_detail
from app.database import SessionLocal

MINT = "FunnelTestMint1111111111111111111111111111"
OTHER_MINT = "OtherMint22222222222222222222222222222222"


@pytest.fixture()
def db():
    """A session with every table this module reads emptied.

    Both builders scan whole tables, so rows left behind by the webhook and
    scanner integration tests would leak straight into these assertions.
    """
    models_to_wipe = (
        models.Trade, models.RiskEvent, models.RugCheckResult,
        models.Signal, models.Position, models.ScannedToken,
    )

    def wipe(session):
        for model in models_to_wipe:
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    session = SessionLocal()
    wipe(session)
    try:
        yield session
    finally:
        wipe(session)
        session.close()


def _scanned(db, address=MINT, *, stage="evaluated", reason="", traded=0):
    row = models.ScannedToken(
        token_address=address, symbol="FUNNEL", chain="solana",
        discovery_source="dexscreener",
        first_seen_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2),
        last_evaluated_at=dt.datetime.now(dt.timezone.utc),
        evaluation_count=1, times_traded=traded, last_stage=stage, last_reason=reason,
        liquidity_usd=120_000.0, volume_24h_usd=300_000.0,
    )
    db.add(row)
    return row


def _signal(db, address=MINT, **kwargs):
    signal = models.Signal(
        symbol="FUNNEL", token_address=address, chain="solana", signal_type="buy",
        price=0.004, raw_payload={}, source="scanner", **kwargs,
    )
    db.add(signal)
    db.flush()
    return signal


# ---------------------------------------------------------------------------
# funnel
# ---------------------------------------------------------------------------

def test_an_empty_database_produces_an_all_zero_funnel(db):
    funnel = build_funnel(db)
    assert funnel.tokens_seen == 0
    assert all(s.reached == 0 for s in funnel.stages)
    assert funnel.widest_drop is None


def test_stage_counts_follow_the_pipeline(db):
    # 5 discovered, 3 stopped at the free pre-screen.
    for i in range(5):
        _scanned(db, f"Mint{i}", stage="prescreen" if i < 3 else "evaluated")
    # 2 reached the shared buy path; one died on the rug check.
    signal = _signal(db)
    _signal(db, OTHER_MINT)
    db.add(models.RiskEvent(
        event_type="rug_check_rejected", details="honeypot", signal_id=signal.id
    ))
    db.flush()

    funnel = build_funnel(db)
    stages = {s.key: s for s in funnel.stages}
    assert stages["found"].reached == 5
    assert stages["found"].rejected_here == 3
    assert stages["prescreen"].reached == 2
    assert stages["evaluated"].reached == 2
    assert stages["evaluated"].rejected_here == 1
    assert stages["security"].reached == 1


def test_the_widest_drop_is_identified(db):
    signal = _signal(db)
    for _ in range(7):
        db.add(models.RiskEvent(
            event_type="signal_score_rejected", details="58/100", signal_id=signal.id
        ))
    db.add(models.RiskEvent(
        event_type="rug_check_rejected", details="honeypot", signal_id=signal.id
    ))
    db.flush()

    assert build_funnel(db).widest_drop.key == "market"   # the stage rejecting on signal score


def test_a_filled_buy_reaches_the_paper_buy_stage(db):
    signal = _signal(db)
    db.add(models.Trade(
        signal_id=signal.id, symbol="FUNNEL", token_address=MINT, side="buy",
        status=models.TradeStatus.FILLED.value, size_usd=100.0, entry_price=0.004,
    ))
    db.flush()
    stages = {s.key: s for s in build_funnel(db).stages}
    assert stages["paper_buy"].reached == 1


def test_a_failed_buy_does_not_reach_the_paper_buy_stage(db):
    signal = _signal(db)
    db.add(models.Trade(
        signal_id=signal.id, symbol="FUNNEL", token_address=MINT, side="buy",
        status=models.TradeStatus.FAILED.value, size_usd=100.0,
    ))
    db.flush()
    stages = {s.key: s for s in build_funnel(db).stages}
    assert stages["paper_buy"].reached == 0


def test_an_unrecognised_rejection_type_is_surfaced_not_dropped(db):
    """A rejection reason added later must not silently vanish from the
    funnel - the numbers have to keep adding up."""
    signal = _signal(db)
    db.add(models.RiskEvent(
        event_type="some_future_rejected", details="new gate", signal_id=signal.id
    ))
    db.flush()
    funnel = build_funnel(db)
    assert funnel.other_rejections == {"some_future_rejected": 1}


def test_non_rejection_events_are_not_counted_as_rejections(db):
    db.add(models.RiskEvent(event_type="trading_resumed", details="manually resumed"))
    db.flush()
    assert build_funnel(db).other_rejections == {}


def test_the_window_excludes_older_activity(db):
    old = _scanned(db, "OldMint")
    old.last_evaluated_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=5)
    _scanned(db, "NewMint")
    db.flush()

    assert build_funnel(db, window_hours=24).tokens_seen == 1
    assert build_funnel(db, window_hours=None).tokens_seen == 2


def test_pass_rate_is_none_when_nothing_reached_a_stage(db):
    """Zero reached is not a 0% pass rate - it is an unanswerable
    question, and rendering it as 0% would read as a failure."""
    stages = {s.key: s for s in build_funnel(db).stages}
    assert stages["found"].pass_rate is None


def test_every_declared_stage_event_is_a_rejection_name():
    for names in STAGE_EVENTS.values():
        for name in names:
            assert name.endswith(("_rejected", "_blocked", "_unavailable")), name


def test_the_funnel_serialises(db):
    import json

    _scanned(db)
    db.flush()
    json.dumps(build_funnel(db).as_dict(), allow_nan=False)


# ---------------------------------------------------------------------------
# token detail
# ---------------------------------------------------------------------------

def test_an_unknown_address_is_reported_as_never_seen(db):
    detail = build_token_detail(db, "NeverSeenMint999")
    assert detail.found is False
    assert "never seen" in detail.verdict


def test_a_prescreened_token_says_where_it_stopped(db):
    _scanned(db, stage="prescreen", reason="liquidity $4,000 below the $35,000 minimum")
    db.flush()
    detail = build_token_detail(db, MINT)
    assert detail.found is True
    assert detail.times_traded == 0
    assert "pre-screen" in detail.verdict
    assert "liquidity" in detail.verdict


def test_a_rejected_token_says_which_stage_rejected_it(db):
    signal = _signal(db, signal_score=88.0, market_quality_score=31.0)
    db.add(models.RiskEvent(
        event_type="market_quality_rejected", details="quality 31/100", signal_id=signal.id
    ))
    db.flush()
    detail = build_token_detail(db, MINT)
    assert "market_quality_rejected" in detail.verdict
    assert detail.latest_signal.signal_score == 88.0


def test_a_traded_token_reports_its_realized_pnl(db):
    signal = _signal(db)
    db.add(models.Trade(
        signal_id=signal.id, symbol="FUNNEL", token_address=MINT, side="buy",
        status=models.TradeStatus.FILLED.value, size_usd=100.0, entry_price=0.004,
    ))
    db.add(models.Trade(
        signal_id=signal.id, symbol="FUNNEL", token_address=MINT, side="sell",
        status=models.TradeStatus.FILLED.value, size_usd=100.0, exit_price=0.005,
        pnl_usd=24.50, closed_at=dt.datetime.now(dt.timezone.utc),
        close_reason="take-profit hit",
    ))
    db.flush()
    detail = build_token_detail(db, MINT)
    assert detail.times_traded == 1
    assert detail.realized_pnl_usd == pytest.approx(24.50)
    assert "traded 1x" in detail.verdict


def test_identity_is_the_mint_address_not_the_symbol(db):
    """Two different mints sharing a symbol are two different assets. If
    lookup ever keyed on the symbol, a scam clone's history would merge
    into the real token's."""
    real = _signal(db, MINT)
    clone = _signal(db, OTHER_MINT)
    real.symbol = clone.symbol = "SAMENAME"
    db.flush()

    assert len(build_token_detail(db, MINT).signals) == 1
    assert build_token_detail(db, MINT).signals[0].token_address == MINT
    assert build_token_detail(db, OTHER_MINT).signals[0].token_address == OTHER_MINT


def test_the_timeline_is_in_chronological_order(db):
    _scanned(db)
    signal = _signal(db)
    db.add(models.RiskEvent(
        event_type="rug_check_rejected", details="honeypot", signal_id=signal.id
    ))
    db.flush()

    timeline = build_token_detail(db, MINT).timeline
    assert [e.at for e in timeline] == sorted(e.at for e in timeline)
    assert [e.kind for e in timeline][0] == "discovered"


def test_a_naive_timestamp_does_not_break_the_timeline(db):
    """Sorting a mix of naive and aware datetimes raises, and SQLite hands
    back naive ones."""
    row = _scanned(db)
    row.first_seen_at = dt.datetime.now().replace(tzinfo=None)
    _signal(db)
    db.flush()
    assert len(build_token_detail(db, MINT).timeline) == 2


def test_the_detail_carries_the_security_check(db):
    signal = _signal(db)
    db.add(models.RugCheckResult(
        signal_id=signal.id, passed=True, liquidity_usd=90_000.0,
        rug_risk_score=22.0, rug_risk_level="low",
    ))
    db.flush()
    detail = build_token_detail(db, MINT)
    assert detail.latest_rug_check.rug_risk_score == 22.0
