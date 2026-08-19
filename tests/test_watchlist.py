"""Tests for the WATCH state machine and the early-signal analysis.

The properties that matter are the ones that keep the engine falsifiable:
failures are recorded with a category rather than dropped, expired entries
count as false positives, and lead time is measured over every watched
token rather than only the winners.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.early_calibration import (
    MIN_BUCKET_SAMPLE,
    build_early_calibration,
    build_false_positives,
    build_lead_time,
    early_bucket,
)
from app.config import settings
from app.database import SessionLocal
from app.early import watchlist as wl
from app.early.engine import Decision, EarlyVerdict
from app.early.features import EarlyFeatures, Feature
from app.early.late_entry import LateEntryRisk, Stage
from app.early.score import EarlyScore

NOW = dt.datetime.now(dt.timezone.utc)




def verdict(
    decision=Decision.WATCH, *, early=68.0, late=15.0, stage=Stage.DEVELOPING,
    reason="promising", features=None,
) -> EarlyVerdict:
    return EarlyVerdict(
        decision=decision,
        reason=reason,
        early=EarlyScore(score=early, factors=[], reliable=True),
        late=LateEntryRisk(risk=late, flags=[], stage=stage),
        features=features if features is not None else EarlyFeatures(),
        technical_score=55.0,
    )


def features_with(**values) -> EarlyFeatures:
    f = EarlyFeatures()
    for name, value in values.items():
        f.add(Feature(name, value, True, "", "test"))
    return f


# ===========================================================================
# the state machine
# ===========================================================================

def test_a_watched_candidate_is_created_and_tracked(clean_db):
    entry = wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
                      verdict=verdict(), price=0.004)
    clean_db.commit()

    assert entry.state == wl.WATCH
    assert entry.early_score == 68.0
    assert entry.price_at_first_signal == 0.004
    assert entry.evaluations == 1
    assert len(entry.score_history) == 1


def test_a_skip_on_an_unwatched_token_creates_nothing(clean_db):
    """Almost every token is a SKIP. Creating a row for each would make the
    watchlist a log of the entire market."""
    assert wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
                     verdict=verdict(Decision.SKIP)) is None
    assert clean_db.query(models.WatchlistEntry).count() == 0


def test_score_history_accumulates_across_evaluations(clean_db):
    """The point of storing anything: it makes 'was the score improving?'
    answerable, which no single snapshot can be."""
    for score in (58.0, 64.0, 71.0):
        wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
                  verdict=verdict(early=score), price=0.004)
    clean_db.commit()

    entry = wl.get(clean_db, "M1")
    assert entry.evaluations == 3
    assert [h["early"] for h in entry.score_history] == [58.0, 64.0, 71.0]
    assert entry.best_early_score == 71.0


def test_score_history_survives_a_commit(clean_db):
    """SQLAlchemy's JSON column does not track in-place mutation, so
    appending to the list without reassigning it is silently discarded -
    which would make the whole feature store nothing."""
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana", verdict=verdict())
    clean_db.commit()
    clean_db.expire_all()

    assert len(wl.get(clean_db, "M1").score_history) == 1


def test_the_best_score_is_remembered_after_it_fades(clean_db):
    for score in (75.0, 60.0):
        wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
                  verdict=verdict(early=score))
    clean_db.commit()

    entry = wl.get(clean_db, "M1")
    assert entry.early_score == 60.0
    assert entry.best_early_score == 75.0
    assert wl.deteriorating(entry) is True


def test_a_small_wobble_is_not_deterioration(clean_db):
    for score in (70.0, 66.0):
        wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
                  verdict=verdict(early=score))
    clean_db.commit()
    assert wl.deteriorating(wl.get(clean_db, "M1")) is False


def test_confirmation_moves_the_entry_to_confirmed(clean_db):
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana", verdict=verdict())
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
              verdict=verdict(Decision.PAPER_BUY, early=78.0))
    clean_db.commit()
    assert wl.get(clean_db, "M1").state == wl.CONFIRMED


def test_a_terminal_entry_is_not_reopened_by_a_later_tick(clean_db):
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana", verdict=verdict())
    wl.mark_traded(clean_db, "M1")
    clean_db.commit()

    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
              verdict=verdict(early=30.0))
    clean_db.commit()

    entry = wl.get(clean_db, "M1")
    assert entry.state == wl.PAPER_BUY
    assert len(entry.score_history) == 2, "history still records the tick"


def test_the_watchlist_is_size_capped(clean_db, monkeypatch):
    """Unbounded, a busy market would put the bot into a rate limit and
    take the price feed down with it."""
    monkeypatch.setattr(settings, "WATCHLIST_MAX_SIZE", 3)
    for i in range(6):
        wl.record(clean_db, token_address=f"M{i}", symbol=f"S{i}", chain="solana",
                  verdict=verdict())
    clean_db.commit()
    assert clean_db.query(models.WatchlistEntry).count() == 3


# ===========================================================================
# failures are recorded, not dropped
# ===========================================================================

@pytest.mark.parametrize("features,expected", [
    (features_with(liquidity_growth=0.7), "liquidity_fell"),
    (features_with(buy_pressure_change=-0.20, buy_pressure=0.35), "buy_pressure_reversed"),
    (features_with(volume_accel_short=0.4), "volume_disappeared"),
    (EarlyFeatures(), "score_decayed"),
])
def test_a_failure_gets_a_category(clean_db, features, expected):
    """'It works 30% of the time' is not actionable. 'It fails because
    volume evaporates before confirmation' is."""
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana", verdict=verdict())
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
              verdict=verdict(Decision.SKIP, features=features))
    clean_db.commit()

    entry = wl.get(clean_db, "M1")
    assert entry.state == wl.FAILED
    assert entry.failure_category == expected
    assert wl.FAILURE_CATEGORIES[expected] in entry.reason


def test_a_token_that_ran_away_is_categorised_as_late(clean_db):
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana", verdict=verdict())
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana",
              verdict=verdict(Decision.SKIP, stage=Stage.OVEREXTENDED))
    clean_db.commit()
    assert wl.get(clean_db, "M1").failure_category == "became_late"


def test_stale_entries_expire_rather_than_lingering(clean_db, monkeypatch):
    monkeypatch.setattr(settings, "WATCHLIST_MAX_AGE_HOURS", 2)
    wl.record(clean_db, token_address="M1", symbol="AAA", chain="solana", verdict=verdict())
    entry = wl.get(clean_db, "M1")
    entry.first_seen_at = NOW - dt.timedelta(hours=5)
    clean_db.commit()

    assert wl.expire_stale(clean_db) == 1
    clean_db.commit()

    entry = wl.get(clean_db, "M1")
    assert entry.state == wl.EXPIRED
    assert entry.failure_category == "expired"


# ===========================================================================
# observations
# ===========================================================================

def test_observations_are_stored_and_returned_oldest_first(clean_db):
    from app.services.price_feed import MarketSnapshot

    for minutes in (10, 5, 0):
        snap = MarketSnapshot(
            price_usd=0.004, liquidity_usd=100_000.0, volume_24h_usd=1.0,
            buys_24h=1, sells_24h=1, price_change_1h_pct=0.0, price_change_24h_pct=0.0,
            pair_created_at=None, fdv_usd=None, token_address="M1",
            observed_at=NOW - dt.timedelta(minutes=minutes),
        )
        wl.store_observation(clean_db, "AAA", "M1", snap)
    clean_db.commit()

    rows = wl.recent_observations(clean_db, "M1")
    assert len(rows) == 3
    assert rows[0].observed_at < rows[-1].observed_at


def test_old_observations_are_pruned(clean_db, monkeypatch):
    """Their entire value is recency - a transaction count from two days
    ago says nothing about whether flow is accelerating now."""
    monkeypatch.setattr(settings, "OBSERVATION_RETENTION_HOURS", 1)
    clean_db.add(models.TokenObservation(
        token_address="M1", symbol="AAA", observed_at=NOW - dt.timedelta(hours=5)
    ))
    clean_db.add(models.TokenObservation(
        token_address="M1", symbol="AAA", observed_at=NOW
    ))
    clean_db.commit()

    assert wl.prune_observations(clean_db) == 1
    clean_db.commit()
    assert clean_db.query(models.TokenObservation).count() == 1


# ===========================================================================
# early calibration
# ===========================================================================

def _forward(db, early, return_pct, *, horizon=60, mfe=None, mae=None):
    db.add(models.ForwardReturn(
        pipeline_event_id=0, token_address=f"M{early}-{return_pct}", symbol="X",
        observed_at=NOW - dt.timedelta(hours=2), score=None, early_score=early,
        price_at_signal=1.0, horizon_minutes=horizon,
        due_at=NOW - dt.timedelta(hours=1),
        price_at_horizon=1 + return_pct / 100, return_pct=return_pct, filled_at=NOW,
        max_favorable_pct=mfe if mfe is not None else max(return_pct, 0),
        max_adverse_pct=mae if mae is not None else min(return_pct, 0),
    ))


def test_early_buckets_match_the_spec():
    assert early_bucket(42) == "<50"
    assert early_bucket(57) == "50-60"
    assert early_bucket(67) == "65-70"
    assert early_bucket(91) == "80+"


def test_a_predictive_early_score_reads_as_monotonic(clean_db):
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 45.0, -6.0)
        _forward(clean_db, 67.0, 4.0)
        _forward(clean_db, 85.0, 14.0)
    clean_db.commit()

    table = build_early_calibration(clean_db, horizon_minutes=60)
    assert table.monotonic is True
    assert "separates outcomes" in table.verdict()


def test_an_early_score_with_no_edge_after_costs_is_reported_as_no_edge(clean_db):
    """A pattern that is real and unprofitable is a different thing from an
    edge, and the verdict has to distinguish them."""
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 45.0, 0.1)
        _forward(clean_db, 85.0, 0.4)     # positive, but under the round-trip cost
    clean_db.commit()

    table = build_early_calibration(clean_db, horizon_minutes=60)
    assert "NO EDGE" in table.verdict()


def test_an_inverted_early_score_is_called_inverted(clean_db):
    """The top bucket has to be PROFITABLE for "inverted" to be the finding.

    If the best bucket also loses money after costs, the honest verdict is
    "no edge anywhere" rather than "the ranking runs backwards" - the
    ranking is beside the point when nothing in it is worth trading. So the
    data here gives the top bucket a real, positive, after-cost return that
    is simply smaller than the bottom bucket's.
    """
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 45.0, 18.0)    # net +15.7% after the round trip
        _forward(clean_db, 85.0, 4.0)     # net  +1.7% - profitable, but worse
    clean_db.commit()
    assert "INVERTED" in build_early_calibration(clean_db, horizon_minutes=60).verdict()


def test_mae_is_reported_so_a_stopped_out_winner_is_visible(clean_db):
    """A +5% horizon return that first went -30% is not a +5% trade."""
    for _ in range(MIN_BUCKET_SAMPLE):
        _forward(clean_db, 85.0, 5.0, mfe=8.0, mae=-30.0)
    clean_db.commit()

    bucket = next(
        b for b in build_early_calibration(clean_db, horizon_minutes=60).buckets
        if b.bucket == "80+"
    )
    assert bucket.mean_return_pct == pytest.approx(5.0)
    assert bucket.mean_adverse_pct == pytest.approx(-30.0)


def test_thin_buckets_produce_no_verdict(clean_db):
    _forward(clean_db, 45.0, -5.0)
    _forward(clean_db, 85.0, 12.0)
    clean_db.commit()
    table = build_early_calibration(clean_db, horizon_minutes=60)
    assert table.monotonic is None
    assert "INSUFFICIENT DATA" in table.verdict()


# ===========================================================================
# false positives and lead time
# ===========================================================================

def test_expired_entries_count_as_false_positives(clean_db):
    """A token that looked promising and went nowhere IS one. Excluding it
    is the easiest way to make a useless signal look useful."""
    for i in range(MIN_BUCKET_SAMPLE):
        clean_db.add(models.WatchlistEntry(
            token_address=f"M{i}", symbol=f"S{i}", chain="solana",
            state=wl.EXPIRED, failure_category="expired", best_early_score=72.0,
            first_seen_at=NOW, last_evaluated_at=NOW,
        ))
    clean_db.commit()

    report = build_false_positives(clean_db)
    assert report.total_resolved == MIN_BUCKET_SAMPLE
    assert report.failed == MIN_BUCKET_SAMPLE
    assert report.false_positive_rate == pytest.approx(1.0)
    assert report.conclusive


def test_a_traded_entry_counts_as_a_success(clean_db):
    clean_db.add(models.WatchlistEntry(
        token_address="M1", symbol="S1", chain="solana", state=wl.PAPER_BUY,
        best_early_score=78.0, first_seen_at=NOW, last_evaluated_at=NOW,
    ))
    clean_db.commit()
    report = build_false_positives(clean_db)
    assert report.succeeded == 1 and report.failed == 0


def test_a_dominant_residual_bucket_is_called_out_as_a_taxonomy_gap(clean_db):
    """If most failures land in 'score_decayed', the taxonomy is missing a
    category rather than the answer being 'they just faded'."""
    for i in range(MIN_BUCKET_SAMPLE):
        clean_db.add(models.WatchlistEntry(
            token_address=f"M{i}", symbol=f"S{i}", chain="solana",
            state=wl.FAILED, failure_category="score_decayed", best_early_score=70.0,
            first_seen_at=NOW, last_evaluated_at=NOW,
        ))
    clean_db.commit()
    assert "taxonomy is missing a category" in build_false_positives(clean_db).verdict()


def test_lead_time_is_inconclusive_without_enough_tracked_signals(clean_db):
    report = build_lead_time(clean_db)
    assert report.conclusive is False
    assert "INSUFFICIENT DATA" in report.verdict()


def test_lead_time_counts_every_watched_token_not_only_winners(clean_db):
    """A lead-time figure over survivors would describe a bot that only
    ever saw winners."""
    for i in range(MIN_BUCKET_SAMPLE):
        clean_db.add(models.WatchlistEntry(
            token_address=f"M{i}", symbol=f"S{i}", chain="solana", state=wl.WATCH,
            price_at_first_signal=1.0, first_signal_at=NOW - dt.timedelta(hours=2),
            first_seen_at=NOW, last_evaluated_at=NOW,
        ))
        # Only a third of them ever went anywhere.
        peak = 25.0 if i % 3 == 0 else 2.0
        clean_db.add(models.ForwardReturn(
            pipeline_event_id=0, token_address=f"M{i}", symbol=f"S{i}",
            observed_at=NOW - dt.timedelta(hours=2), price_at_signal=1.0,
            horizon_minutes=60, due_at=NOW - dt.timedelta(hours=1),
            max_favorable_pct=peak, return_pct=peak, filled_at=NOW,
        ))
    clean_db.commit()

    report = build_lead_time(clean_db)
    assert report.tracked == MIN_BUCKET_SAMPLE, "every watched token is counted"
    assert report.reached[20.0] == len([i for i in range(MIN_BUCKET_SAMPLE) if i % 3 == 0])
    assert report.reached[50.0] == 0
    assert report.conclusive


def test_the_reports_serialise(clean_db):
    import json

    json.dumps(build_false_positives(clean_db).as_dict(), allow_nan=False)
    json.dumps(build_lead_time(clean_db).as_dict(), allow_nan=False)
    json.dumps(build_early_calibration(clean_db, horizon_minutes=60).as_dict(), allow_nan=False)
