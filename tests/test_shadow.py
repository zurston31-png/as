"""Tests for shadow challengers.

Two things matter most and both are about restraint: a challenger must
never touch champion state, and a duplicate must never become a second
sample. The rest is making sure the observations that do get recorded are
the ones a paired comparison needs - including the boring ones where
everybody said no.
"""
import datetime as dt
import random

import pytest

from app import models
from app.config import settings
from app.database import SessionLocal
from app.shadow import compare as shadow_compare
from app.shadow.challengers import CHAMPION_ID, Challenger, enabled
from app.shadow.opportunity import opportunity_id
from app.shadow.recorder import BUY, NO_SIGNAL, REJECT, record_opportunity

NOW = dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
REGIME = "bull_trend/normal_volatility/deep_liquidity"


@pytest.fixture
def db():
    session = SessionLocal()
    def wipe():
        for model in (models.ShadowPosition, models.ShadowDecision):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()
    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


@pytest.fixture
def one_challenger(monkeypatch):
    monkeypatch.setattr(settings, "SHADOW_ENABLED", True)
    monkeypatch.setattr(
        settings, "SHADOW_CHALLENGERS",
        '[{"strategy_id": "tighter", "min_score_to_enter": 90}]',
    )
    return "tighter"


def _record(db, *, token="ShadowMint1", price=0.01, decision=BUY, at=NOW,
            reason="test", score=70.0, liquidity=200_000.0, series=None):
    return record_opportunity(
        db, token_address=token, symbol="SHDW", chain="solana",
        reference_price=price, observed_at=at, market_regime=REGIME,
        liquidity_usd=liquidity, champion_decision=decision,
        champion_reason=reason, champion_score=score, series=series,
        rng=random.Random(7),
    )


# ---------------------------------------------------------------------------
# isolation - the guarantee that matters most
# ---------------------------------------------------------------------------

def test_the_shadow_package_cannot_reach_execution_or_champion_state():
    """Structural, not polite. A grep-level guard, because an edit that
    granted one of these paths would look harmless in review and the
    damage would surface as numbers nobody could explain."""
    import pathlib

    forbidden = (
        "get_execution_client", "LIVE_TRADING", "LIVE_EXECUTION_ACKNOWLEDGED",
        "risk_manager", "adjust_cash", "close_position", "partial_close_position",
        "models.Position", "models.Trade",
    )
    offenders = []
    for path in pathlib.Path("app/shadow").rglob("*.py"):
        body = "\n".join(
            line for line in path.read_text().splitlines()
            if not line.strip().startswith("#")
        )
        for token in forbidden:
            if token in body:
                offenders.append(f"{path.name}:{token}")
    assert offenders == [], f"shadow can reach champion or live state: {offenders}"


def test_recording_never_creates_a_real_position_or_trade(db, one_challenger):
    before_positions = db.query(models.Position).count()
    before_trades = db.query(models.Trade).count()

    _record(db, decision=BUY)
    db.commit()

    assert db.query(models.Position).count() == before_positions
    assert db.query(models.Trade).count() == before_trades
    assert db.query(models.ShadowDecision).count() >= 1


def test_a_duplicate_does_not_destroy_the_callers_uncommitted_work(db):
    """The bug this guards against was real: a bare db.rollback() on the
    duplicate path wiped a RiskEvent the champion had just written in the
    same session. The shadow system was mutating champion state - the one
    thing it must never do - and it showed up as a missing audit row
    nobody would have connected to this code.
    """
    _record(db)
    db.commit()

    # the champion writes something, then a duplicate opportunity arrives
    db.add(models.RiskEvent(event_type="champion_work", details="must survive"))
    summary = _record(db)
    db.commit()

    assert summary["skipped_duplicate"] >= 1
    survived = db.query(models.RiskEvent).filter_by(event_type="champion_work").first()
    assert survived is not None, "the shadow duplicate path destroyed champion state"


# ---------------------------------------------------------------------------
# opportunity identity and idempotency
# ---------------------------------------------------------------------------

def test_sub_second_jitter_still_pairs():
    """Execution order is not a difference in the opportunity. If jitter
    split the id, nothing would ever pair."""
    a = opportunity_id("M", NOW, 0.01)
    b = opportunity_id("M", NOW.replace(microsecond=987_654), 0.01)
    assert a == b


def test_a_later_look_is_a_new_opportunity():
    assert opportunity_id("M", NOW, 0.01) != opportunity_id("M", NOW + dt.timedelta(minutes=1), 0.01)


def test_a_different_price_is_a_new_opportunity():
    """Two evaluations in the same second at different prices are
    different opportunities - the market moved between them."""
    assert opportunity_id("M", NOW, 0.01) != opportunity_id("M", NOW, 0.02)


def test_a_replayed_webhook_does_not_create_a_second_sample(db):
    first = _record(db)
    db.commit()
    second = _record(db)
    db.commit()

    assert first["recorded"] >= 1
    assert second["recorded"] == 0
    assert second["skipped_duplicate"] >= 1
    assert db.query(models.ShadowDecision).filter_by(strategy_id=CHAMPION_ID).count() == 1


def test_restarting_and_replaying_is_idempotent(db):
    """The id comes from the opportunity, not a row counter, so a restart
    that re-processes the same candidate collapses onto the same sample."""
    _record(db)
    db.commit()
    db.close()

    fresh = SessionLocal()
    try:
        _record(fresh)
        fresh.commit()
        assert fresh.query(models.ShadowDecision).filter_by(strategy_id=CHAMPION_ID).count() == 1
    finally:
        for row in fresh.query(models.ShadowDecision).all():
            fresh.delete(row)
        fresh.commit()
        fresh.close()


# ---------------------------------------------------------------------------
# no survivorship bias
# ---------------------------------------------------------------------------

def test_a_rejection_is_recorded_not_dropped(db, one_challenger):
    """A dataset of entries only makes every strategy look identical on
    the trades it did not take, and the disagreement is the signal."""
    _record(db, decision=REJECT, reason="score too low", score=40.0)
    db.commit()

    rows = db.query(models.ShadowDecision).all()
    assert rows and all(r.decision in (REJECT, NO_SIGNAL) for r in rows)


def test_both_arms_are_recorded_even_when_both_decline(db, one_challenger):
    _record(db, decision=REJECT, score=30.0)
    db.commit()

    ids = {r.strategy_id for r in db.query(models.ShadowDecision).all()}
    assert CHAMPION_ID in ids and one_challenger in ids


def test_a_failed_fill_is_recorded_as_a_tried_and_missed(db):
    """A strategy that would have tried and missed is a different
    observation from one that never tried. Dropping the misses would
    quietly inflate every hit rate."""
    _record(db, decision=BUY, liquidity=50.0, price=0.01)   # a pool that thin fails
    db.commit()

    row = db.query(models.ShadowDecision).filter_by(strategy_id=CHAMPION_ID).one()
    assert row.decision == BUY
    if not row.fill_succeeded:
        assert row.fill_failure_reason
        assert db.query(models.ShadowPosition).count() == 0, (
            "a failed fill must not open a hypothetical position"
        )


def test_a_successful_fill_opens_a_hypothetical_position_in_its_own_table(db):
    _record(db, decision=BUY, liquidity=5_000_000.0)
    db.commit()

    row = db.query(models.ShadowDecision).filter_by(strategy_id=CHAMPION_ID).one()
    if row.fill_succeeded:
        position = db.query(models.ShadowPosition).one()
        assert position.strategy_id == CHAMPION_ID
        assert position.market_regime == REGIME
        assert position.entry_price == row.entry_price


def test_regime_is_persisted_on_every_observation(db, one_challenger):
    _record(db, decision=REJECT)
    db.commit()

    for row in db.query(models.ShadowDecision).all():
        assert row.market_regime == REGIME
        assert row.liquidity_regime == "deep_liquidity"


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def test_no_challengers_configured_is_a_valid_quiet_default(monkeypatch):
    monkeypatch.setattr(settings, "SHADOW_ENABLED", True)
    monkeypatch.setattr(settings, "SHADOW_CHALLENGERS", "")
    assert enabled() == []


def test_malformed_configuration_disables_challengers_without_crashing(monkeypatch):
    monkeypatch.setattr(settings, "SHADOW_ENABLED", True)
    monkeypatch.setattr(settings, "SHADOW_CHALLENGERS", "{not json")
    assert enabled() == []


def test_a_challenger_may_not_impersonate_the_champion(monkeypatch):
    monkeypatch.setattr(settings, "SHADOW_ENABLED", True)
    monkeypatch.setattr(
        settings, "SHADOW_CHALLENGERS", '[{"strategy_id": "champion"}]'
    )
    assert enabled() == []


def test_an_unknown_factor_override_is_rejected_at_load(monkeypatch):
    """A typo would otherwise add a weight the scorer never reads, and the
    challenger would silently be the champion wearing a different name."""
    monkeypatch.setattr(settings, "SHADOW_ENABLED", True)
    monkeypatch.setattr(
        settings, "SHADOW_CHALLENGERS",
        '[{"strategy_id": "typo", "weight_overrides": {"nonexistent": 0.5}}]',
    )
    assert enabled() == []


def test_a_challenger_inherits_champion_weights_it_does_not_override():
    """Reads as a diff, so it does not silently drift when the champion's
    weights change."""
    from app.signals.scoring import DEFAULT_WEIGHTS

    challenger = Challenger("c", weight_overrides={"rsi": 0.25})
    weights = challenger.weights()
    assert weights["rsi"] == 0.25
    assert weights["macd"] == DEFAULT_WEIGHTS["macd"]


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------

def _pair(db, oid_seed, *, champion_return=None, challenger_return=None,
          champion_in=True, challenger_in=True):
    at = NOW + dt.timedelta(minutes=oid_seed)
    oid = opportunity_id(f"M{oid_seed}", at, 0.01)
    for strategy_id, entered, ret in (
        (CHAMPION_ID, champion_in, champion_return),
        ("tighter", challenger_in, challenger_return),
    ):
        decision = models.ShadowDecision(
            opportunity_id=oid, strategy_id=strategy_id,
            is_champion=strategy_id == CHAMPION_ID,
            token_address=f"M{oid_seed}", symbol="T", chain="solana",
            decision=BUY if entered else REJECT, reason="x",
            signal_score=70.0, market_regime=REGIME,
            liquidity_regime="deep_liquidity",
            entry_price=0.01 if entered else None,
            fill_succeeded=entered if entered else None,
            decided_at=at,
        )
        db.add(decision)
        db.flush()
        if entered and ret is not None:
            db.add(models.ShadowPosition(
                decision_id=decision.id, opportunity_id=oid, strategy_id=strategy_id,
                token_address=f"M{oid_seed}", symbol="T", market_regime=REGIME,
                opened_at=at, closed_at=at + dt.timedelta(hours=1),
                entry_price=0.01, exit_price=0.01 * (1 + ret / 100),
                return_pct=ret,
            ))


def test_a_thin_pairing_reports_insufficient_data(db):
    for i in range(3):
        _pair(db, i, champion_return=5.0, challenger_return=6.0)
    db.commit()

    result = shadow_compare.compare(db, "tighter")
    assert result.conclusive is False
    assert "INSUFFICIENT_DATA" in result.verdict()


def test_only_shared_opportunities_are_paired(db):
    """An unpaired observation is not a weaker data point, it is a
    different comparison."""
    _pair(db, 1, champion_return=5.0, challenger_return=5.0)
    at = NOW + dt.timedelta(minutes=99)
    db.add(models.ShadowDecision(
        opportunity_id=opportunity_id("SOLO", at, 0.01), strategy_id=CHAMPION_ID,
        is_champion=True, token_address="SOLO", symbol="T", chain="solana",
        decision=BUY, reason="x", decided_at=at,
    ))
    db.commit()

    assert shadow_compare.compare(db, "tighter").paired == 1


def test_agreement_and_disagreement_are_counted_separately(db):
    _pair(db, 1, champion_return=5.0, challenger_return=5.0)                    # both in
    _pair(db, 2, champion_in=False, challenger_in=False)                        # both out
    _pair(db, 3, champion_return=4.0, challenger_in=False)                      # champion only
    _pair(db, 4, champion_in=False, challenger_return=8.0)                      # challenger only
    db.commit()

    result = shadow_compare.compare(db, "tighter")
    assert result.paired == 4
    assert result.both_entered == 1
    assert result.both_rejected == 1
    assert result.champion_only == 1
    assert result.challenger_only == 1


def test_declining_counts_as_a_zero_outcome_not_missing_data(db):
    """Treating a decline as missing would drop exactly the observations
    where the arms disagreed, which is the signal."""
    for i in range(shadow_compare.MIN_PAIRS + 2):
        _pair(db, i, champion_return=10.0, challenger_in=False)
    db.commit()

    result = shadow_compare.compare(db, "tighter")
    assert result.champion_expectancy == pytest.approx(10.0)
    assert result.challenger_expectancy == pytest.approx(0.0)
    assert result.difference == pytest.approx(-10.0)


def test_an_unresolved_entry_is_counted_not_guessed(db):
    """A pending outcome recorded as 0% would be a fabricated
    measurement."""
    _pair(db, 1, champion_return=None, challenger_return=None)   # entered, unresolved
    db.commit()

    result = shadow_compare.compare(db, "tighter")
    assert result.unresolved == 2
    assert result.champion_returns == []


def test_the_comparison_hands_r_multiples_to_the_gate_not_percents(db):
    """Feeding percent straight in would inflate every effect a
    hundredfold and clear the effect bar on noise."""
    for i in range(shadow_compare.MIN_PAIRS + 2):
        _pair(db, i, champion_return=10.0, challenger_return=20.0)
    db.commit()

    champion, challenger = shadow_compare.compare(db, "tighter").arms()
    assert champion.expectancy_r == pytest.approx(0.10)
    assert challenger.expectancy_r == pytest.approx(0.20)


def test_the_comparison_never_promotes_anything_itself():
    """Letting the module that generates challengers also decide which win
    would let a search mark its own homework."""
    import pathlib

    body = pathlib.Path("app/shadow/compare.py").read_text()
    assert "def promote" not in body
    assert "changelog.record" not in body


# ---------------------------------------------------------------------------
# two expectancies, because one number hides the trade-off
# ---------------------------------------------------------------------------

def test_opportunity_and_conditional_expectancy_are_reported_separately(db):
    """The worked case that makes the distinction matter.

    Over 40 paired opportunities the champion enters 8 and averages +5.5%
    on those; the challenger enters 16 and averages +2.25%. Per ENTERED
    TRADE the champion is far better. Per OPPORTUNITY the challenger wins,
    because it finds twice as many. Collapsing these into one number would
    make one of those facts disappear, and which one disappeared would
    depend on which formula happened to be chosen.
    """
    for i in range(40):
        champion_in = i < 8
        challenger_in = i < 16
        _pair(
            db, i,
            champion_return=5.5 if champion_in else None,
            challenger_return=2.25 if challenger_in else None,
            champion_in=champion_in, challenger_in=challenger_in,
        )
    db.commit()

    result = shadow_compare.compare(db, "tighter")
    assert result.paired == 40

    # Conditional: only the trades each side actually took.
    assert result.champion_trades == 8
    assert result.challenger_trades == 16
    assert result.champion_trade_expectancy == pytest.approx(5.5)
    assert result.challenger_trade_expectancy == pytest.approx(2.25)
    assert result.trade_difference == pytest.approx(-3.25)

    # Per opportunity: 8/40 * 5.5 vs 16/40 * 2.25.
    assert result.champion_expectancy == pytest.approx(1.1)
    assert result.challenger_expectancy == pytest.approx(0.9)
    assert result.difference == pytest.approx(-0.2)


def test_a_decline_is_zero_per_opportunity_and_absent_per_trade(db):
    """A trade that was never taken is not a flat trade. Averaging it in as
    one would drag every selective strategy's per-trade number toward zero
    purely for being selective."""
    for i in range(shadow_compare.MIN_PAIRS + 2):
        _pair(db, i, champion_return=10.0, challenger_in=False)
    db.commit()

    result = shadow_compare.compare(db, "tighter")
    assert result.challenger_expectancy == pytest.approx(0.0)   # per opportunity
    assert result.challenger_trades == 0
    assert result.challenger_trade_expectancy is None           # per trade: no trades
    assert result.trade_difference is None


def test_the_gate_reads_the_paired_series_not_the_self_selected_one(db):
    """Handing the gate two self-selected samples would let a challenger
    win by being choosier rather than by being better."""
    for i in range(40):
        _pair(
            db, i,
            champion_return=5.0 if i < 8 else None,
            challenger_return=2.0 if i < 16 else None,
            champion_in=i < 8, challenger_in=i < 16,
        )
    db.commit()

    result = shadow_compare.compare(db, "tighter")
    champion, challenger = result.arms()
    assert champion.expectancy_r == pytest.approx(8 / 40 * 0.05)
    assert challenger.expectancy_r == pytest.approx(16 / 40 * 0.02)


def test_an_unresolved_entry_stays_out_of_both_series(db):
    """It is not a decline and it is not a flat trade - it is a
    measurement that has not been taken yet."""
    _pair(db, 1, champion_return=None, challenger_return=None)
    db.commit()

    result = shadow_compare.compare(db, "tighter")
    assert result.champion_returns == []
    assert result.champion_trade_returns == []
    assert result.unresolved == 2


# ---------------------------------------------------------------------------
# the shipped experiment - one hypothesis per challenger
# ---------------------------------------------------------------------------

def test_the_shipped_challengers_change_exactly_one_thing_each():
    """A challenger that moved ten weights at once can be measured but not
    interpreted: "this version happened to perform better" is not a finding.
    The shipped pair brackets the champion's entry threshold and changes
    nothing else, so the collection run can answer "did the threshold cause
    this" rather than shrugging at twenty simultaneous edits.

    A future experiment that varies a weight instead is fine - what this
    guards is varying several dimensions in one arm.
    """
    from app.config import settings as live_settings
    from app.shadow.challengers import enabled as shipped

    challengers = shipped()
    assert len(challengers) == 2, "two arms: one stricter, one looser"

    for challenger in challengers:
        varied = [
            name for name, changed in (
                ("weights", bool(challenger.weight_overrides)),
                ("threshold", challenger.min_score_to_enter is not None),
                ("stop_loss", challenger.stop_loss_pct is not None),
                ("take_profit", challenger.take_profit_pct is not None),
            ) if changed
        ]
        assert varied == ["threshold"], (
            f"{challenger.strategy_id} varies {varied}; one arm must vary one dimension"
        )

    thresholds = sorted(c.threshold() for c in challengers)
    champion = live_settings.MIN_SIGNAL_SCORE_TO_ENTER
    assert thresholds[0] < champion < thresholds[1], (
        "the pair must bracket the champion - two challengers on the same side "
        "of it measure the same direction twice"
    )
