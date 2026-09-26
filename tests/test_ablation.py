"""Tests for pre-screen ablation (app/analysis/ablation.py).

The invariants:

  A1  A check's marginal contribution is the mints ONLY it rejects, which
      is a different and much smaller number than its rejection count.
  A2  Removing a check changes the pass count by exactly its unique
      rejections - no other check's recorded verdict moves, because every
      check already ran independently on every candidate.
  A3  Attribution is per mint, using the LATEST verdict, so the
      most-rescanned token does not get forty votes.
  A4  Events with no per-check detail are excluded and reported, never
      guessed at.
  A5  Shares are withheld below a sample floor.
  A6  An empty window says nothing was recorded, not that every check is
      redundant.
  A7  Nothing here changes a threshold or disables a check.
"""
import datetime as dt

import pytest

from app import models
from app.analysis.ablation import MIN_MINTS_TO_REPORT, build_ablation
from app.database import SessionLocal
from app.pipeline import PRESCREEN

NOW = dt.datetime.now(dt.timezone.utc)


@pytest.fixture()
def clean_db():
    def wipe(session):
        for row in session.query(models.PipelineEvent).all():
            session.delete(row)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _screen(db, mint, *, failed=(), minutes_ago=1.0, checks=None):
    """One recorded pre-screen verdict for one mint.

    `failed` names the checks that failed; every other check in the
    standard five is recorded as passing, mirroring what
    app/scanner/filters.py writes.
    """
    names = checks or ("liquidity", "volume", "age", "transactions", "sell_pressure")
    db.add(models.PipelineEvent(
        occurred_at=NOW - dt.timedelta(minutes=minutes_ago),
        token_address=mint, symbol=mint[:6], chain="solana",
        stage=PRESCREEN, passed=not failed, reason="", detail={
            "checks": [
                {"name": n, "passed": n not in failed, "reason": "", "value": None,
                 "threshold": None}
                for n in names
            ]
        },
    ))


def _check(report, name):
    return next(c for c in report.checks if c.check == name)


# ---------------------------------------------------------------------------
# A1/A2 - marginal contribution
# ---------------------------------------------------------------------------

def test_a_check_that_only_duplicates_others_has_no_marginal_effect(clean_db):
    """A1, the headline. `volume` rejects all ten mints here and removing
    it would change nothing, because liquidity already stopped every one
    of them. A raw rejection count would rank it joint-first."""
    for i in range(10):
        _screen(clean_db, f"Mint{i}", failed=("liquidity", "volume"))
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    volume = _check(report, "volume")
    assert volume.rejected == 10
    assert volume.uniquely_rejected == 0
    assert volume.redundant_rejections == 10
    assert report.would_pass_without("volume") == report.mints_passing_all


def test_a_check_that_catches_what_nothing_else_does_is_the_whole_gate(clean_db):
    """The other extreme. Twelve mints fail only sell_pressure; without
    it, all twelve reach the buy path."""
    for i in range(12):
        _screen(clean_db, f"Mint{i}", failed=("sell_pressure",))
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    sell = _check(report, "sell_pressure")
    assert sell.uniquely_rejected == 12
    assert sell.redundant_rejections == 0
    assert report.mints_passing_all == 0
    assert report.would_pass_without("sell_pressure") == 12


def test_rejection_count_and_marginal_contribution_can_rank_differently(clean_db):
    """The reason this module exists. `liquidity` rejects far more, but
    `age` is the only thing standing between three mints and the buy
    path."""
    for i in range(20):
        _screen(clean_db, f"Liq{i}", failed=("liquidity", "volume"))
    for i in range(3):
        _screen(clean_db, f"Age{i}", failed=("age",))
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    assert _check(report, "liquidity").rejected == 20
    assert _check(report, "liquidity").uniquely_rejected == 0
    assert _check(report, "age").rejected == 3
    assert _check(report, "age").uniquely_rejected == 3
    # ...and the ranking follows marginal contribution, not volume.
    assert report.checks[0].check == "age"


def test_removing_a_check_adds_exactly_its_unique_rejections(clean_db):
    """A2. No other verdict moves - every check already ran on every
    candidate, so this is set arithmetic rather than a re-simulation."""
    for i in range(5):
        _screen(clean_db, f"Clean{i}")                       # pass everything
    for i in range(4):
        _screen(clean_db, f"Vol{i}", failed=("volume",))     # only volume
    for i in range(2):
        _screen(clean_db, f"Both{i}", failed=("volume", "age"))
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    assert report.mints_passing_all == 5
    assert report.would_pass_without("volume") == 9          # 5 + the 4 volume-only
    assert report.would_pass_without("age") == 5             # age never acted alone


def test_removing_an_unknown_check_changes_nothing(clean_db):
    for i in range(3):
        _screen(clean_db, f"Mint{i}")
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    assert report.would_pass_without("not_a_real_check") == report.mints_passing_all


# ---------------------------------------------------------------------------
# A3 - per mint, latest verdict
# ---------------------------------------------------------------------------

def test_one_mint_screened_many_times_counts_once(clean_db):
    """A3. Otherwise the most-rescanned token decides the answer."""
    for i in range(40):
        _screen(clean_db, "Mint1", failed=("liquidity",), minutes_ago=i + 1)
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    assert report.mints_evaluated == 1
    assert _check(report, "liquidity").rejected == 1


def test_the_latest_verdict_wins_when_a_token_changes(clean_db):
    """A token whose liquidity recovered is not still a liquidity
    rejection - the current state of the funnel is what is being
    described."""
    _screen(clean_db, "Mint1", failed=("liquidity",), minutes_ago=60)
    _screen(clean_db, "Mint1", failed=(), minutes_ago=1)
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    assert report.mints_passing_all == 1
    assert _check(report, "liquidity").rejected == 0


# ---------------------------------------------------------------------------
# A4/A5/A6 - honesty about the sample
# ---------------------------------------------------------------------------

def test_events_without_per_check_detail_are_excluded_and_reported(clean_db):
    """A4. These rows predate per-check recording. Guessing which checks
    they failed would be inventing data, and silently dropping them would
    make the sample look larger than it is."""
    clean_db.add(models.PipelineEvent(
        occurred_at=NOW, token_address="Old1", symbol="OLD", stage=PRESCREEN,
        passed=False, reason="liquidity too low", detail={},
    ))
    _screen(clean_db, "New1", failed=("age",))
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    assert report.mints_evaluated == 1
    assert "no per-check detail" in report.note


def test_a_window_of_only_detail_free_events_says_so(clean_db):
    clean_db.add(models.PipelineEvent(
        occurred_at=NOW, token_address="Old1", symbol="OLD", stage=PRESCREEN,
        passed=False, reason="liquidity too low", detail={},
    ))
    clean_db.commit()

    report = build_ablation(clean_db, window_hours=24)
    assert not report.has_data
    assert "predate" in report.note


def test_a_share_is_withheld_below_the_sample_floor(clean_db):
    """A5. "100% uniquely rejected" over two tokens is noise wearing a
    percentage sign."""
    for i in range(MIN_MINTS_TO_REPORT - 1):
        _screen(clean_db, f"Mint{i}", failed=("age",))
    clean_db.commit()

    assert _check(build_ablation(clean_db, window_hours=24), "age").marginal_share is None


def test_a_share_appears_once_there_is_enough(clean_db):
    for i in range(MIN_MINTS_TO_REPORT):
        _screen(clean_db, f"Mint{i}", failed=("age",))
    clean_db.commit()

    assert _check(build_ablation(clean_db, window_hours=24), "age").marginal_share == pytest.approx(1.0)


def test_an_empty_window_is_not_a_finding_that_checks_are_redundant(clean_db):
    """A6. Zero unique rejections everywhere, because there were zero
    candidates. Reporting that as ablation output would be an argument
    for deleting every filter."""
    report = build_ablation(clean_db, window_hours=24)
    assert not report.has_data
    assert report.checks == []
    assert "absence of data" in report.note


def test_events_outside_the_window_are_excluded(clean_db):
    _screen(clean_db, "Mint1", failed=("age",), minutes_ago=60 * 24 * 30)
    clean_db.commit()

    assert not build_ablation(clean_db, window_hours=24).has_data
    assert build_ablation(clean_db, window_hours=None).has_data


# ---------------------------------------------------------------------------
# A7 - it is a report, not a tuner
# ---------------------------------------------------------------------------

def test_ablation_does_not_touch_any_threshold(clean_db):
    """A7. The mandate is to surface evidence, not to act on it. This
    pins that the module has no side effect on the settings it reports
    about."""
    from app.config import settings

    before = (
        settings.SCANNER_MIN_LIQUIDITY_USD,
        settings.SCANNER_MIN_VOLUME_24H_USD,
        settings.SCANNER_MIN_TXNS_24H,
        settings.SCANNER_MAX_SELL_SHARE,
    )
    for i in range(20):
        _screen(clean_db, f"Mint{i}", failed=("liquidity",))
    clean_db.commit()

    build_ablation(clean_db, window_hours=24)

    assert before == (
        settings.SCANNER_MIN_LIQUIDITY_USD,
        settings.SCANNER_MIN_VOLUME_24H_USD,
        settings.SCANNER_MIN_TXNS_24H,
        settings.SCANNER_MAX_SELL_SHARE,
    )


def test_the_report_is_json_safe(clean_db):
    import json

    _screen(clean_db, "Mint1", failed=("age",))
    clean_db.commit()

    payload = build_ablation(clean_db, window_hours=24).as_dict()
    json.dumps(payload)
    assert payload["would_pass_without"]["age"] == 1
