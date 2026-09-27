"""Tests for threshold replay over the bot's own recorded history.

The danger with a threshold sweep is that it always produces a winner.
These tests care about the cases where the honest output is "this proves
nothing": too few trades, too few distinct scores, no profitable level at
all, and a peak that is really a spike.
"""
import datetime as dt

import pytest

from app import models
from app.research.replay import MIN_TAKEN, replay_thresholds


def _row(db, score, ret, horizon=60, version=None):
    now = dt.datetime.now(dt.timezone.utc)
    db.add(models.ForwardReturn(
        pipeline_event_id=1, token_address="M", symbol="T",
        observed_at=now, score=score, price_at_signal=0.01,
        horizon_minutes=horizon, due_at=now, return_pct=ret,
        strategy_version=version,
    ))


def test_an_empty_history_supports_no_change(clean_db):
    report = replay_thresholds(clean_db, horizon_minutes=60)
    assert report.population == 0
    assert "INSUFFICIENT DATA" in report.verdict()
    assert "Nothing here supports changing the threshold" in report.verdict()


def test_a_thin_sample_is_not_evidence(clean_db):
    for _ in range(MIN_TAKEN - 1):
        _row(clean_db, 80.0, 10.0)
    clean_db.commit()

    report = replay_thresholds(clean_db, horizon_minutes=60)
    assert report.usable == []
    assert "INSUFFICIENT DATA" in report.verdict()


def test_a_higher_threshold_takes_fewer_trades(clean_db):
    for score in (52.0, 58.0, 63.0, 68.0, 72.0, 78.0, 83.0):
        for _ in range(MIN_TAKEN):
            _row(clean_db, score, 5.0)
    clean_db.commit()

    report = replay_thresholds(clean_db, horizon_minutes=60)
    taken = [o.taken for o in report.outcomes]
    assert taken == sorted(taken, reverse=True), "taken count must fall as the bar rises"
    assert all(o.taken + o.skipped == report.population for o in report.outcomes)


def test_a_score_that_ranks_outcomes_favours_a_higher_threshold(clean_db):
    """Machinery test on data with a known answer, not a market claim."""
    for _ in range(MIN_TAKEN + 5):
        _row(clean_db, 55.0, -6.0)
        _row(clean_db, 85.0, 18.0)
    clean_db.commit()

    report = replay_thresholds(clean_db, horizon_minutes=60)
    assert report.best is not None
    assert report.best.threshold >= 60.0
    assert report.best.expectancy_net_pct > 0


def test_a_score_with_no_edge_says_so_instead_of_naming_a_winner(clean_db):
    """The most important case. A sweep always has a top row; when every
    level loses money after costs, reporting that row as "best" would read
    as a recommendation to trade it."""
    for score in (55.0, 65.0, 75.0, 85.0):
        for _ in range(MIN_TAKEN + 2):
            _row(clean_db, score, -4.0)
    clean_db.commit()

    verdict = replay_thresholds(clean_db, horizon_minutes=60).verdict()
    assert "NO THRESHOLD WAS PROFITABLE" in verdict
    assert "trades less of the same thing" in verdict


def test_costs_are_subtracted_so_a_thin_edge_does_not_read_as_profit(clean_db):
    """A +1% gross average is a loss after a ~2.3% round trip."""
    for _ in range(MIN_TAKEN + 5):
        _row(clean_db, 85.0, 1.0)
    clean_db.commit()

    top = next(o for o in replay_thresholds(clean_db, horizon_minutes=60).outcomes
               if o.threshold == 80.0)
    assert top.mean_return_pct == pytest.approx(1.0)
    assert top.expectancy_net_pct < 0


def test_a_spike_is_called_a_spike_not_a_plateau(clean_db):
    """A threshold that only looks good because its neighbours look bad is
    the classic overfit, and it is invisible if you read only the top row.

    Note the data has to account for thresholds being CUMULATIVE: a score
    of 67 is taken at 65 and also at 60 and 55. Isolating a spike therefore
    needs the band just above the winning threshold to be good and
    everything on both sides of it to be bad - scores below drag the lower
    thresholds down, scores above drag the higher ones down.
    """
    for _ in range(MIN_TAKEN + 2):
        _row(clean_db, 62.0, -30.0)   # drags every threshold at or below 60
        _row(clean_db, 67.0, 40.0)    # the only good band
        _row(clean_db, 72.0, -30.0)   # drags every threshold at or above 70
    clean_db.commit()

    report = replay_thresholds(clean_db, horizon_minutes=60)
    note = report.plateau_note()
    assert note is not None and "spike, not a plateau" in note


def test_too_few_distinct_scores_is_flagged(clean_db):
    """If every candidate scored ~70, the thresholds are slicing one group
    and the differences between them are meaningless."""
    for _ in range(60):
        _row(clean_db, 70.0, 3.0)
    clean_db.commit()

    report = replay_thresholds(clean_db, horizon_minutes=60)
    assert any("distinct score values" in w for w in report.warnings)


def test_replay_can_be_scoped_to_one_strategy_version(clean_db):
    """Mixing versions would compare a threshold against candidates a
    different scorer produced."""
    for _ in range(MIN_TAKEN + 2):
        _row(clean_db, 85.0, 10.0, version="v-old")
        _row(clean_db, 85.0, -10.0, version="v-new")
    clean_db.commit()

    old = replay_thresholds(clean_db, horizon_minutes=60, strategy_version="v-old")
    new = replay_thresholds(clean_db, horizon_minutes=60, strategy_version="v-new")

    assert old.population == new.population == MIN_TAKEN + 2
    assert old.best.expectancy_net_pct > 0
    assert "NO THRESHOLD WAS PROFITABLE" in new.verdict()


def test_the_table_states_what_it_does_not_model(clean_db):
    """Per-trade expectancy is not an equity curve, and a reader who
    forgets that will over-trust a low threshold that concurrency limits
    would have blocked."""
    for _ in range(MIN_TAKEN + 2):
        _row(clean_db, 85.0, 5.0)
    clean_db.commit()

    table = replay_thresholds(clean_db, horizon_minutes=60).table()
    assert "not an equity curve" in table
    assert "recorded candidates" in table
