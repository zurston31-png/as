"""Tests for app/data/staleness.py.

The distinction these pin down is the one the module exists for: how long
ago the bot *observed* the data (hard reject) versus how long ago the
market last *moved* (a market-quality question, not a data-integrity one).
Conflating them either lets stale prices size positions, or rejects every
genuinely quiet token.
"""
import datetime as dt

import pytest

from app.config import settings
from app.data.staleness import check_price_sanity, check_snapshot_freshness
from tests.conftest import make_market_snapshot


# ---------------------------------------------------------------------------
# observation age
# ---------------------------------------------------------------------------

def test_a_fresh_snapshot_passes():
    verdict = check_snapshot_freshness(make_market_snapshot())
    assert verdict.fresh
    assert verdict.observation_age_seconds < 5


def test_an_old_observation_is_rejected():
    now = dt.datetime.now(dt.timezone.utc)
    stale = make_market_snapshot(observed_at=now - dt.timedelta(seconds=600))
    verdict = check_snapshot_freshness(stale, now=now)
    assert verdict.fresh is False
    assert "stale" in verdict.reason
    assert verdict.observation_age_seconds == pytest.approx(600, abs=1)


def test_no_snapshot_is_rejected_rather_than_raising():
    # Every caller must be able to handle missing and stale through one
    # code path, so None returns a verdict instead of blowing up.
    verdict = check_snapshot_freshness(None)
    assert verdict.fresh is False
    assert "no market snapshot" in verdict.reason


def test_the_freshness_limit_is_configurable(monkeypatch):
    now = dt.datetime.now(dt.timezone.utc)
    snapshot = make_market_snapshot(observed_at=now - dt.timedelta(seconds=90))

    monkeypatch.setattr(settings, "MAX_MARKET_DATA_AGE_SECONDS", 120.0)
    assert check_snapshot_freshness(snapshot, now=now).fresh

    monkeypatch.setattr(settings, "MAX_MARKET_DATA_AGE_SECONDS", 60.0)
    assert check_snapshot_freshness(snapshot, now=now).fresh is False


def test_a_naive_observed_at_is_treated_as_utc_not_crashed_on():
    """SQLite hands back naive datetimes. Subtracting one from an aware
    `now` raises, which would take out the whole buy path."""
    now = dt.datetime.now(dt.timezone.utc)
    naive = make_market_snapshot(observed_at=now.replace(tzinfo=None))
    verdict = check_snapshot_freshness(naive, now=now)
    assert verdict.fresh


def test_a_quiet_pool_is_not_treated_as_stale_data():
    """No volume in the last hour is a market-quality problem, not a
    corrupt feed. Rejecting it here would double-punish calm tokens and
    reject them for the wrong reason."""
    calm = make_market_snapshot(volume_1h_usd=0.0, buys_1h=0, sells_1h=0, price_change_1h_pct=0.0)
    assert check_snapshot_freshness(calm).fresh


# ---------------------------------------------------------------------------
# price sanity
# ---------------------------------------------------------------------------

def test_a_normal_price_passes():
    assert check_price_sanity(0.0042).fresh


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.00001])
def test_a_non_positive_price_is_always_corrupt(bad):
    verdict = check_price_sanity(bad)
    assert verdict.fresh is False
    assert "not positive" in verdict.reason


def test_a_missing_price_is_rejected():
    assert check_price_sanity(None).fresh is False


def test_an_implausible_jump_is_treated_as_a_bad_tick(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PRICE_JUMP_FACTOR", 20.0)
    verdict = check_price_sanity(1.0, previous_price=0.001)  # 1000x
    assert verdict.fresh is False
    assert "bad tick" in verdict.reason


def test_the_jump_check_is_symmetric(monkeypatch):
    """A 1000x collapse is as implausible as a 1000x spike - a decimals bug
    can produce either direction."""
    monkeypatch.setattr(settings, "MAX_PRICE_JUMP_FACTOR", 20.0)
    assert check_price_sanity(0.001, previous_price=1.0).fresh is False


def test_a_violent_but_real_memecoin_move_still_passes(monkeypatch):
    """Memecoins genuinely 5x. The limit catches broken data, not
    volatility - set it too tight and the bot exits every winner on a
    'bad tick' it invented."""
    monkeypatch.setattr(settings, "MAX_PRICE_JUMP_FACTOR", 20.0)
    assert check_price_sanity(0.005, previous_price=0.001).fresh  # 5x


def test_no_previous_price_means_no_jump_check():
    assert check_price_sanity(1_000_000.0).fresh


def test_a_zero_previous_price_does_not_divide_by_zero():
    assert check_price_sanity(0.5, previous_price=0.0).fresh
