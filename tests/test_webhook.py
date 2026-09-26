import pytest
from fastapi.testclient import TestClient

import app.services.trading_service as trading_service
from app.config import settings
from app.database import SessionLocal
from app import models
from app.main import app
from app.rugcheck.filters import RugCheckReport
from app.services import price_feed
from app.signals.scoring import Factor, SignalScore
from tests.conftest import make_market_snapshot

client = TestClient(app)


@pytest.fixture(autouse=True)
def _patch_network(monkeypatch):
    async def fake_price(instrument):
        return 0.001234

    async def fake_rug_check(chain, token_address):
        return RugCheckReport(passed=True, reasons=[], liquidity_usd=150_000.0, dev_wallet_pct=0.05)

    async def fake_signal_score(chain, token_address, symbol):
        return SignalScore(
            score=90.0, direction="long", reliable=True,
            factors=[Factor(name="trend_direction", score=0.9, weight=1.0, reason="test fixture")],
        )

    # The market-quality gate fetches a live snapshot; without one it
    # (correctly) refuses to trade, so give it a healthy market. The real
    # scorer still runs against it.
    async def fake_snapshot(token_address):
        return make_market_snapshot(token_address=token_address)

    monkeypatch.setattr(price_feed, "get_price_usd", fake_price)
    monkeypatch.setattr(price_feed, "get_market_snapshot", fake_snapshot)
    monkeypatch.setattr(trading_service, "run_rug_checks", fake_rug_check)
    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", fake_signal_score)
    yield


def _payload(symbol: str, signal: str, secret: str | None = None) -> dict:
    return {
        "secret": secret if secret is not None else settings.WEBHOOK_SECRET,
        "symbol": symbol,
        "token_address": f"{symbol}TokenAddress111",
        "chain": "solana",
        "signal": signal,
        "price": 0.001234,
        "rsi": 55,
        "ema9": 0.00125,
        "ema21": 0.00115,
        "volume": 100000,
        "volume_sma": 40000,
        "breakout_level": 0.0012,
    }


def test_webhook_rejects_invalid_secret():
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("BADSECRETCOIN", "buy", secret="wrong-secret"))
    assert resp.status_code == 401


def test_webhook_buy_opens_position_then_sell_closes_it():
    symbol = "WEBHOOKCOIN"

    resp = client.post(settings.WEBHOOK_PATH, json=_payload(symbol, "buy"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"

    db = SessionLocal()
    try:
        position = db.query(models.Position).filter_by(symbol=symbol, status="open").first()
        assert position is not None
        assert position.qty > 0
        assert position.entry_price > 0
        assert position.stop_loss < position.entry_price < position.take_profit

        trade = db.query(models.Trade).filter_by(symbol=symbol, side="buy").first()
        assert trade is not None
        assert trade.status == "filled"
        assert trade.mode == "paper"
        # Trade.position_id (Stage 8) must link the entry leg back to the
        # position it opened - without it there's no way to join an exit
        # leg back to its entry's signal/rug-check context in the journal.
        assert trade.position_id == position.id
    finally:
        db.close()

    resp2 = client.post(settings.WEBHOOK_PATH, json=_payload(symbol, "sell"))
    assert resp2.status_code == 200

    db = SessionLocal()
    try:
        position = db.query(models.Position).filter_by(symbol=symbol).order_by(models.Position.id.desc()).first()
        assert position.status == "closed"
        assert position.close_reason

        sell_trade = db.query(models.Trade).filter_by(symbol=symbol, side="sell").first()
        assert sell_trade is not None
        assert sell_trade.pnl_usd is not None
        assert sell_trade.position_id == position.id
        assert sell_trade.close_reason == position.close_reason
    finally:
        db.close()


def test_webhook_second_buy_ignored_while_position_open():
    symbol = "DUPBUYCOIN"
    client.post(settings.WEBHOOK_PATH, json=_payload(symbol, "buy"))
    client.post(settings.WEBHOOK_PATH, json=_payload(symbol, "buy"))

    db = SessionLocal()
    try:
        open_positions = db.query(models.Position).filter_by(symbol=symbol, status="open").all()
        assert len(open_positions) == 1
    finally:
        db.close()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["live_trading"] is False


# ---------------------------------------------------------------------------
# live signal-score gate (Stage 9)
# ---------------------------------------------------------------------------

def test_buy_persists_the_live_signal_score_on_the_signal_row(monkeypatch):
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("SCORECOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        signal = db.query(models.Signal).filter_by(symbol="SCORECOIN", signal_type="buy").first()
        assert signal is not None
        assert signal.signal_score == pytest.approx(90.0)
        assert signal.signal_score_reliable is True
        assert signal.signal_score_factors
    finally:
        db.close()


def test_buy_rejected_when_signal_score_is_below_the_minimum(monkeypatch):
    async def weak_score(chain, token_address, symbol):
        from app.signals.scoring import Factor, SignalScore
        return SignalScore(
            score=40.0, direction="neutral", reliable=True,
            factors=[Factor(name="trend_direction", score=0.4, weight=1.0, reason="weak setup")],
        )

    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", weak_score)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("WEAKSCORECOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        position = db.query(models.Position).filter_by(symbol="WEAKSCORECOIN", status="open").first()
        assert position is None
        event = (
            db.query(models.RiskEvent)
            .filter_by(event_type="signal_score_rejected")
            .order_by(models.RiskEvent.id.desc())
            .first()
        )
        assert event is not None
        assert "40.0" in event.details
    finally:
        db.close()


def test_buy_rejected_when_score_is_marked_unreliable_even_if_numerically_high(monkeypatch):
    """A high raw score built from mostly-missing data must not buy its way
    past the gate - reliable=False must block the entry regardless of the
    number, the same rule app/signals/scoring.py documents for itself."""
    async def unreliable_score(chain, token_address, symbol):
        from app.signals.scoring import Factor, SignalScore
        return SignalScore(
            score=95.0, direction="long", reliable=False,
            warnings=["too much missing data"],
            factors=[Factor(name="trend_direction", score=0.95, weight=1.0, reason="thin data")],
        )

    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", unreliable_score)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("UNRELIABLECOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        position = db.query(models.Position).filter_by(symbol="UNRELIABLECOIN", status="open").first()
        assert position is None
    finally:
        db.close()


def test_buy_rejected_when_live_candle_data_is_unavailable(monkeypatch):
    async def no_data(chain, token_address, symbol):
        return None

    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", no_data)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("NODATACOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        position = db.query(models.Position).filter_by(symbol="NODATACOIN", status="open").first()
        assert position is None
        event = (
            db.query(models.RiskEvent)
            .filter_by(event_type="signal_score_unavailable")
            .order_by(models.RiskEvent.id.desc())
            .first()
        )
        assert event is not None
    finally:
        db.close()


def test_live_signal_score_gate_can_be_disabled(monkeypatch):
    """LIVE_SIGNAL_SCORE_ENABLED=false must skip the gate entirely - entries
    then run on the risk gate + rug check alone, same as before the gate
    existed."""
    monkeypatch.setattr(settings, "LIVE_SIGNAL_SCORE_ENABLED", False)

    async def should_not_be_called(chain, token_address, symbol):
        raise AssertionError("evaluate_live_entry_signal must not be called when the gate is disabled")

    monkeypatch.setattr(trading_service, "evaluate_live_entry_signal", should_not_be_called)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("GATEOFFCOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        position = db.query(models.Position).filter_by(symbol="GATEOFFCOIN", status="open").first()
        assert position is not None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# market-quality gate (app/signals/market_quality.py)
# ---------------------------------------------------------------------------

@pytest.fixture()
def _roomy_risk_limits(monkeypatch):
    """Give the risk gate headroom so a test about something else isn't
    blocked by caps that earlier tests in the session have already eaten
    into - the daily-trade counter and the open-position book are shared
    across the whole run, so whether a fill happens would otherwise depend
    on test ordering. The caps themselves are covered by
    tests/test_risk_manager.py. Swapping the instance rather than the
    settings is necessary because RiskManager snapshots its limits in
    __init__.
    """
    from app.risk.manager import RiskManager

    monkeypatch.setattr(
        trading_service, "risk_manager",
        RiskManager(
            max_concurrent_positions=20, max_daily_trades=50, cooldown_seconds=0,
            max_total_exposure_pct=1.0,
        ),
    )
    return monkeypatch


def test_untradeable_market_is_rejected_even_with_a_perfect_signal(monkeypatch, _roomy_risk_limits):
    """The signal score says 90/100 and the rug check passes. The market is
    still one wash-traded burst into a pool too thin to exit, and that must
    be enough on its own to block the trade - the two existing scores have
    no way to see it.

    Needs `_roomy_risk_limits` for the same reason its siblings do: the
    daily-trade counter is shared across the run, so without headroom the
    risk gate rejects first with `buy_blocked` and the market-quality gate
    this test is about never runs. The rejection asserted here is the
    market's, not the cap's.
    """
    async def wash_traded(token_address):
        return make_market_snapshot(
            token_address=token_address,
            liquidity_usd=8_000.0,
            volume_24h_usd=4_000_000.0,
            volume_1h_usd=3_500_000.0,
            buys_24h=40,
            sells_24h=6,
        )

    monkeypatch.setattr(price_feed, "get_market_snapshot", wash_traded)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("WASHCOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        assert db.query(models.Position).filter_by(symbol="WASHCOIN", status="open").first() is None
        signal = db.query(models.Signal).filter_by(symbol="WASHCOIN").order_by(models.Signal.id.desc()).first()
        assert signal.market_quality_score is not None
        assert signal.market_quality_score < settings.MIN_MARKET_QUALITY_SCORE
        event = db.query(models.RiskEvent).filter_by(signal_id=signal.id).first()
        assert event.event_type == "market_quality_rejected"
    finally:
        db.close()


def test_stale_market_data_is_rejected_rather_than_traded_on(monkeypatch, _roomy_risk_limits):
    """A twenty-minute-old price looks authoritative and is not. Sizing a
    position and setting a stop against it is worse than skipping.

    `_roomy_risk_limits` for the same shared-counter reason as the test
    above: the assertion is that *staleness* rejected the entry, which only
    means anything if the risk cap did not reject it first.
    """
    import datetime as dt

    async def stale(token_address):
        return make_market_snapshot(
            token_address=token_address,
            observed_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20),
        )

    monkeypatch.setattr(price_feed, "get_market_snapshot", stale)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("STALECOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        assert db.query(models.Position).filter_by(symbol="STALECOIN", status="open").first() is None
        signal = db.query(models.Signal).filter_by(symbol="STALECOIN").order_by(models.Signal.id.desc()).first()
        event = db.query(models.RiskEvent).filter_by(signal_id=signal.id).first()
        assert event.event_type == "stale_data_rejected"
    finally:
        db.close()


def test_missing_market_data_fails_closed(monkeypatch):
    """No snapshot at all must reject, not fall through to a neutral score.
    This is the fail-closed rule the whole bot is built on."""
    async def nothing(token_address):
        return None

    monkeypatch.setattr(price_feed, "get_market_snapshot", nothing)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("NODATACOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        assert db.query(models.Position).filter_by(symbol="NODATACOIN", status="open").first() is None
    finally:
        db.close()


def test_a_healthy_market_records_its_quality_score_on_the_signal(_roomy_risk_limits):
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("QUALITYCOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        signal = db.query(models.Signal).filter_by(symbol="QUALITYCOIN").order_by(models.Signal.id.desc()).first()
        assert signal.market_quality_score >= settings.MIN_MARKET_QUALITY_SCORE
        # The full factor breakdown is persisted so a rejection (or an
        # entry) can be explained months later without re-fetching data
        # that no longer exists.
        assert signal.market_quality_factors
        assert {f["name"] for f in signal.market_quality_factors} >= {"liquidity_depth", "volume_concentration"}
        assert db.query(models.Position).filter_by(symbol="QUALITYCOIN", status="open").first() is not None
    finally:
        db.close()


def test_the_market_quality_gate_can_be_disabled(monkeypatch, _roomy_risk_limits):
    monkeypatch.setattr(settings, "MARKET_QUALITY_ENABLED", False)

    async def should_not_be_called(token_address):
        raise AssertionError("get_market_snapshot must not be called when the gate is disabled")

    monkeypatch.setattr(price_feed, "get_market_snapshot", should_not_be_called)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("MQOFFCOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        assert db.query(models.Position).filter_by(symbol="MQOFFCOIN", status="open").first() is not None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# provenance: every signal and trade must say which strategy produced it
# ---------------------------------------------------------------------------

def test_signals_and_trades_are_stamped_with_the_strategy_version(_roomy_risk_limits):
    """Without this, analytics silently pools results from configurations
    that never coexisted, and quotes a win rate for a strategy that never
    ran."""
    from app.strategy.version import current_label

    resp = client.post(settings.WEBHOOK_PATH, json=_payload("VERSIONCOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        signal = db.query(models.Signal).filter_by(symbol="VERSIONCOIN").order_by(models.Signal.id.desc()).first()
        trade = db.query(models.Trade).filter_by(symbol="VERSIONCOIN", side="buy").first()
        assert signal.strategy_version == current_label()
        assert trade.strategy_version == current_label()
        assert db.query(models.StrategyVersion).filter_by(label=current_label()).first() is not None
    finally:
        db.close()


def test_paper_trades_record_what_the_fill_actually_cost(_roomy_risk_limits):
    """Fees and price impact have to be on the trade row, not implied by the
    fill price, or a P&L review can't separate a bad entry from an
    expensive one."""
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("COSTCOIN", "buy"))
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        trade = db.query(models.Trade).filter_by(symbol="COSTCOIN", side="buy").first()
        assert trade.fee_usd is not None and trade.fee_usd > 0
        assert trade.execution_cost_pct is not None and trade.execution_cost_pct > 0
        assert trade.fill_delay_seconds is not None and trade.fill_delay_seconds >= 0
    finally:
        db.close()


# ---------------------------------------------------------------------------
# payload shapes TradingView actually sends
#
# Kept at the end of the module on purpose. Several tests above assert on
# the *latest* RiskEvent row or on a fill happening, and the daily-trade
# counter and open-position book are shared across the whole run - so a
# test that trades earlier in file order silently changes their outcome.
# These were originally placed mid-file and did exactly that.
# ---------------------------------------------------------------------------

def _pine_payload(symbol: str, signal: str) -> str:
    """The literal string `pine/memecoin_signal_strategy.pine` builds.

    Kept as text rather than a dict because the bug this covers was about
    JSON types: `time` is emitted unquoted, and a dict round-tripped through
    the test client would have hidden that.
    """
    return (
        '{"secret":"' + settings.WEBHOOK_SECRET + '"'
        ',"symbol":"' + symbol + '"'
        ',"token_address":"' + symbol + 'TokenAddress111"'
        ',"chain":"solana"'
        ',"signal":"' + signal + '"'
        ',"price":0.001234'
        ',"time":1766248800000'
        ',"rsi":55.12'
        ',"ema9":0.00125'
        ',"ema21":0.00115'
        ',"volume":100000.0'
        ',"volume_sma":40000.0'
        ',"breakout_level":0.0012}'
    )


def test_the_exact_pine_payload_is_accepted(_roomy_risk_limits):
    """Regression: Pine emits `"time":1766248800000` as a JSON number, but
    the field was typed `Optional[str]` and pydantic v2 does not coerce int
    to str. Every live alert was therefore rejected 422 by FastAPI before
    the route handler ran, which is why nothing appeared in the log."""
    resp = client.post(
        settings.WEBHOOK_PATH,
        content=_pine_payload("PINECOIN", "buy"),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "accepted"


def test_numeric_time_survives_as_a_parsed_datetime():
    """Coercing int -> str must not break the timestamp itself."""
    from app.schemas import TradingViewAlert

    alert = TradingViewAlert(secret="s", symbol="X", signal="buy", price=1.0, time=1766248800000)
    parsed = alert.parsed_time()
    assert parsed is not None
    assert parsed.year == 2025 and parsed.tzinfo is not None


def test_iso_time_still_parses():
    from app.schemas import TradingViewAlert

    alert = TradingViewAlert(
        secret="s", symbol="X", signal="buy", price=1.0, time="2026-08-20T20:00:00Z",
    )
    assert alert.parsed_time().year == 2026


def _current_pine_payload(symbol: str, signal: str) -> str:
    """What the current `pine/memecoin_signal_strategy.pine` emits.

    Two deliberate differences from `_pine_payload` above, which is kept as
    the regression case for the numeric form older charts still send:
      - `time` is quoted, so the payload validates against a server build
        that predates the int -> str coercion in app/schemas.py.
      - a `na` indicator is `null`, not `NaN`. Pine's str.tostring() renders
        na as "NaN", which is not valid JSON, so a single warmup bar made the
        entire body unparseable and the alert was lost rather than rejected.
    """
    return (
        '{"secret":"' + settings.WEBHOOK_SECRET + '"'
        ',"symbol":"' + symbol + '"'
        ',"token_address":"' + symbol + 'TokenAddress111"'
        ',"chain":"solana"'
        ',"signal":"' + signal + '"'
        ',"price":0.001234'
        ',"time":"1766248800000"'
        ',"rsi":55.12'
        ',"ema9":0.00125'
        ',"ema21":0.00115'
        ',"volume":null'
        ',"volume_sma":null'
        ',"breakout_level":0.0012}'
    )


def test_the_current_pine_payload_is_accepted(_roomy_risk_limits):
    resp = client.post(
        settings.WEBHOOK_PATH,
        content=_current_pine_payload("PINEQUOTEDCOIN", "buy"),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "accepted"


def test_a_quoted_pine_timestamp_parses_to_the_same_instant_as_the_bare_one():
    """The script now quotes `time`; that must not shift the recorded bar
    time, or shadow samples taken from it would be keyed to the wrong bar."""
    from app.schemas import TradingViewAlert

    quoted = TradingViewAlert(secret="s", symbol="X", signal="buy", price=1.0, time="1766248800000")
    bare = TradingViewAlert(secret="s", symbol="X", signal="buy", price=1.0, time=1766248800000)
    assert quoted.parsed_time() == bare.parsed_time()


def test_a_nan_indicator_is_recorded_as_absent_rather_than_as_a_number():
    """Charts still running the previous script send Pine's bare `NaN` for an
    indicator that has not warmed up. stdlib json accepts it, so the alert is
    kept - but it must land as absent, not as a NaN float in a numeric column,
    which would read as a measurement that was taken and came out
    unrepresentable.

    Asserted at the schema rather than through the endpoint because the point
    is the stored value, and the response body only carries the signal id.
    """
    import json as _json

    from app.schemas import TradingViewAlert

    body = _current_pine_payload("NANCOIN", "buy").replace('"volume":null', '"volume":NaN')
    alert = TradingViewAlert.model_validate(_json.loads(body))
    assert alert.volume is None


def test_a_nan_price_is_refused_because_the_stop_would_be_nan_too():
    """The indicators fail soft; the price cannot. Size, stop and take-profit
    are all derived from it, so a NaN price must reject the alert rather than
    open a position with NaN levels."""
    import json as _json

    body = _current_pine_payload("NANPRICECOIN", "buy").replace('"price":0.001234', '"price":NaN')
    resp = client.post(
        settings.WEBHOOK_PATH, content=body, headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert "price" in " ".join(resp.json()["detail"])
    assert _json.loads(body)["symbol"] == "NANPRICECOIN"


def test_malformed_json_is_reported_rather_than_swallowed():
    resp = client.post(
        settings.WEBHOOK_PATH, content="{not json", headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert "not valid JSON" in resp.json()["detail"]


def test_a_missing_required_field_names_the_field():
    resp = client.post(settings.WEBHOOK_PATH, json={"secret": settings.WEBHOOK_SECRET, "symbol": "X"})
    assert resp.status_code == 422
    detail = " ".join(resp.json()["detail"])
    assert "signal" in detail and "price" in detail


def test_a_differently_named_field_is_called_out_rather_than_ignored_silently():
    """TradingView's own default template uses `ticker`/`action`. Being told
    the field was unexpected is the difference between a two-minute fix and
    an afternoon."""
    resp = client.post(
        settings.WEBHOOK_PATH,
        json={"secret": settings.WEBHOOK_SECRET, "ticker": "WIF", "action": "buy", "price": 1.0},
    )
    assert resp.status_code == 422
    detail = " ".join(resp.json()["detail"])
    assert "unexpected field 'ticker'" in detail
    assert "unexpected field 'action'" in detail


@pytest.mark.parametrize(
    "payload",
    [
        # wrong scalar type: input is the offending value
        {"secret": "SECRET", "symbol": "X", "signal": "buy", "price": "abc"},
        # missing field: pydantic reports the WHOLE body as the input, which
        # is how the secret leaked into the error body the first time.
        {"secret": "SECRET", "symbol": "X"},
        # secret itself the wrong type
        {"secret": 12345, "symbol": "X", "signal": "buy", "price": 1.0},
    ],
    ids=["wrong-type", "missing-field", "secret-wrong-type"],
)
def test_the_secret_never_appears_in_an_error_body(payload):
    """A 422 body reaches TradingView's alert log and any screenshot of it.
    The secret must not travel with it under any validation failure."""
    payload = {**payload}
    if payload.get("secret") == "SECRET":
        payload["secret"] = settings.WEBHOOK_SECRET

    resp = client.post(settings.WEBHOOK_PATH, json=payload)
    assert resp.status_code == 422
    assert settings.WEBHOOK_SECRET not in resp.text


def test_the_secret_is_never_written_to_the_log(caplog, _roomy_risk_limits):
    """Covers the accepted path and the rejected path: the log is the whole
    point of this endpoint's rewrite, so it is the likeliest place to leak."""
    import logging

    with caplog.at_level(logging.INFO, logger="app.main"):
        client.post(settings.WEBHOOK_PATH, json=_payload("LOGCOIN", "buy"))
        client.post(settings.WEBHOOK_PATH, json={"secret": settings.WEBHOOK_SECRET, "symbol": "X"})

    assert caplog.text, "the webhook must log what it received"
    assert settings.WEBHOOK_SECRET not in caplog.text
    assert "LOGCOIN" in caplog.text


# ---------------------------------------------------------------------------
# 401: the two causes need opposite fixes
# ---------------------------------------------------------------------------

def test_an_unconfigured_server_secret_says_so_rather_than_blaming_tradingview(monkeypatch):
    """The Pine script's default Webhook Secret input is the SAME
    placeholder the server ships. A deployer who has set neither end sends
    a value that matches exactly and is still refused, because
    verify_webhook_secret refuses the placeholder outright.

    "invalid webhook secret" then points at the TradingView side, which is
    the half that is already consistent. The message has to name the .env.
    """
    from app.security import PLACEHOLDER_SECRET

    monkeypatch.setattr(settings, "WEBHOOK_SECRET", PLACEHOLDER_SECRET)
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("PLACEHOLDERCOIN", "buy",
                                                            secret=PLACEHOLDER_SECRET))
    assert resp.status_code == 401
    detail = resp.json()["detail"]
    assert "WEBHOOK_SECRET in .env" in detail
    assert "no matter what it sends" in detail


def test_an_empty_server_secret_is_treated_the_same_as_the_placeholder(monkeypatch):
    monkeypatch.setattr(settings, "WEBHOOK_SECRET", "")
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("EMPTYCOIN", "buy", secret="anything"))
    assert resp.status_code == 401
    assert "no webhook secret configured" in resp.json()["detail"]


def test_a_genuine_mismatch_points_at_the_pine_input():
    """The other direction: the server IS configured, so the fix is on the
    chart - and the message must not send the user to edit .env."""
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("MISMATCHCOIN", "buy",
                                                            secret="not-the-real-secret"))
    assert resp.status_code == 401
    detail = resp.json()["detail"]
    assert "does not match" in detail
    assert "recreate the alert" in detail
    assert "WEBHOOK_SECRET in .env is unset" not in detail


def test_the_secret_value_is_never_returned_in_a_401_body():
    """Whichever branch fires, the body goes back to TradingView and into
    its alert log. Neither message may carry the secret itself."""
    resp = client.post(settings.WEBHOOK_PATH, json=_payload("LEAKCOIN", "buy", secret="wrong"))
    assert settings.WEBHOOK_SECRET not in resp.text
    assert "wrong" not in resp.text
