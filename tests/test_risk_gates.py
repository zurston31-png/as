"""Tests for the two risk gates that were built and then left unwired.

Both are safety checks, so the tests care most about the two ways a safety
check fails badly: not firing when it should, and firing so often that the
sensible response is to switch it off.
"""
import datetime as dt

import pytest

from app import models
from app.config import settings
from app.data import cross_check
from app.risk import book
from app.early.watchlist import price_series as wl_price_series
from app.risk.correlation import HIGH_CORRELATION, MIN_OVERLAP


# ---------------------------------------------------------------------------
# cross-check
# ---------------------------------------------------------------------------

def test_agreeing_sources_are_trustworthy():
    check = cross_check.compare(liquidity=(100_000.0, 108_000.0), liquidity_tolerance=0.30)
    assert check.trustworthy
    assert not check.disagreements


def test_a_material_disagreement_blocks_rather_than_picking_a_side():
    """Picking the convenient number is selection bias with extra steps,
    and it always biases toward trading more."""
    check = cross_check.compare(liquidity=(250_000.0, 50_000.0), liquidity_tolerance=0.30)
    assert not check.trustworthy
    assert "DISAGREE" in check.reason
    assert "no way to tell which" in check.reason


def test_agreement_still_sizes_off_the_thinner_pool():
    """Agreement within 30% still leaves a real gap, and the thinner pool
    is the one that decides what getting out actually costs."""
    check = cross_check.compare(liquidity=(120_000.0, 100_000.0), liquidity_tolerance=0.30)
    assert check.trustworthy
    liquidity = next(a for a in check.agreements if a.field == "liquidity")
    assert liquidity.conservative(prefer="low") == 100_000.0


def test_one_silent_source_does_not_read_as_agreement():
    """"We could not check" is not "we checked and it was fine"."""
    strict = cross_check.compare(liquidity=(100_000.0, None), require_two_sources=True)
    assert not strict.trustworthy
    assert "only one source" in strict.reason

    lenient = cross_check.compare(liquidity=(100_000.0, None), require_two_sources=False)
    assert lenient.trustworthy, "a provider being down is not automatically a stop-trading event"


# ---------------------------------------------------------------------------
# correlation gate
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_book():
    from app.database import SessionLocal

    def wipe(session):
        for model in (models.Position, models.TokenObservation):
            for row in session.query(model).all():
                session.delete(row)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _position(db, symbol, mint, size_usd=1000.0, entry_price=0.001):
    """An open position worth `size_usd` at entry.

    Position carries qty and entry_price, not a notional, so the size is
    expressed as the quantity that multiplies out to it.
    """
    position = models.Position(
        symbol=symbol, token_address=mint, chain="solana",
        status=models.PositionStatus.OPEN.value,
        qty=size_usd / entry_price, entry_price=entry_price,
        stop_loss=entry_price * 0.8, take_profit=entry_price * 1.5,
        opened_at=dt.datetime.now(dt.timezone.utc),
    )
    db.add(position)
    return position


def _prices(db, symbol, mint, values):
    base = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=len(values))
    for i, value in enumerate(values):
        db.add(models.TokenObservation(
            token_address=mint, symbol=symbol, price_usd=value,
            observed_at=base + dt.timedelta(minutes=i),
        ))


# Price paths on the same scale as the fixture entry price (0.001), so a
# position marked to the last stored price is worth roughly what it cost.
# On a different scale the mark-to-market would silently rescale every
# exposure in these tests and the cap assertions would be meaningless.
ENTRY = 0.001
# A path and a copy of it. Correlating these gives rho == 1.0 exactly.
LOCKSTEP = [ENTRY * (1.0 + (i % 7) * 0.03) for i in range(MIN_OVERLAP + 8)]
# The same path reflected, so rho is strongly negative.
INVERSE = [ENTRY * 2 - v for v in LOCKSTEP]


def test_nothing_is_blocked_before_anything_has_been_measured(clean_book):
    """The most important case.

    On a fresh install there are no stored prices, so every pair is
    unmeasured. correlation.analyse treats unknown as fully correlated,
    which is right for a report and catastrophic for a gate - the bot
    would refuse to open a second position forever, never collecting the
    observations that would prove otherwise. A risk check that stops all
    trading before measuring anything is indistinguishable from a broken
    one, and the natural response is to switch it off.
    """
    _position(clean_book, "AAA", "MintA", size_usd=9_000.0)
    clean_book.commit()

    verdict = book.gate(
        clean_book, symbol="BBB", token_address="MintB",
        portfolio_value_usd=10_000.0, proposed_size_usd=1_000.0,
    )
    assert not verdict.blocked
    assert verdict.measured_pairs == 0
    assert verdict.unmeasured_pairs == 1


def test_a_measured_correlated_cluster_over_the_cap_blocks(clean_book):
    _position(clean_book, "AAA", "MintA", size_usd=4_000.0)
    _prices(clean_book, "AAA", "MintA", LOCKSTEP)
    _prices(clean_book, "BBB", "MintB", LOCKSTEP)
    clean_book.commit()

    verdict = book.gate(
        clean_book, symbol="BBB", token_address="MintB",
        portfolio_value_usd=10_000.0, proposed_size_usd=1_000.0,
    )
    assert verdict.measured_pairs == 1
    assert verdict.correlated_with and verdict.correlated_with[0][1] >= HIGH_CORRELATION

    # The held leg is marked to its last stored price, not to what it cost -
    # a position that has run up is more capital at risk, not the same
    # amount. LOCKSTEP ends 18% above entry, so 4,000 marks to 4,720.
    held = 4_000.0 * (LOCKSTEP[-1] / ENTRY)
    assert verdict.cluster_usd == pytest.approx(held + 1_000.0)
    assert verdict.cluster_usd > verdict.cap_usd == pytest.approx(3_000.0)
    assert verdict.blocked
    assert "increasing one bet, not diversifying" in verdict.reason


def test_a_measured_correlated_cluster_under_the_cap_passes(clean_book):
    _position(clean_book, "AAA", "MintA", size_usd=1_000.0)
    _prices(clean_book, "AAA", "MintA", LOCKSTEP)
    _prices(clean_book, "BBB", "MintB", LOCKSTEP)
    clean_book.commit()

    verdict = book.gate(
        clean_book, symbol="BBB", token_address="MintB",
        portfolio_value_usd=10_000.0, proposed_size_usd=1_000.0,
    )
    assert verdict.measured_pairs == 1
    assert not verdict.blocked


def test_an_uncorrelated_position_is_not_part_of_the_cluster(clean_book):
    """The gate must not degrade into a second total-exposure cap."""
    _position(clean_book, "AAA", "MintA", size_usd=9_000.0)
    _prices(clean_book, "AAA", "MintA", LOCKSTEP)
    _prices(clean_book, "BBB", "MintB", INVERSE)
    clean_book.commit()

    verdict = book.gate(
        clean_book, symbol="BBB", token_address="MintB",
        portfolio_value_usd=10_000.0, proposed_size_usd=1_000.0,
    )
    assert verdict.measured_pairs == 1
    assert verdict.correlated_with == []
    assert not verdict.blocked


def test_the_same_instrument_is_not_a_correlated_cluster(clean_book):
    """A token is perfectly correlated with itself. Counting that here
    would double-charge it - the per-token exposure cap already covers
    adding to an existing position."""
    _position(clean_book, "AAA", "MintA", size_usd=9_000.0)
    _prices(clean_book, "AAA", "MintA", LOCKSTEP)
    clean_book.commit()

    verdict = book.gate(
        clean_book, symbol="AAA", token_address="MintA",
        portfolio_value_usd=10_000.0, proposed_size_usd=1_000.0,
    )
    assert verdict.correlated_with == []
    assert not verdict.blocked


def test_too_few_overlapping_points_counts_as_unmeasured(clean_book):
    """Two points always correlate perfectly. A rho from a handful of
    observations is noise, and blocking a trade on it is worse than not
    checking."""
    _position(clean_book, "AAA", "MintA", size_usd=9_000.0)
    _prices(clean_book, "AAA", "MintA", LOCKSTEP[:5])
    _prices(clean_book, "BBB", "MintB", LOCKSTEP[:5])
    clean_book.commit()

    verdict = book.gate(
        clean_book, symbol="BBB", token_address="MintB",
        portfolio_value_usd=10_000.0, proposed_size_usd=1_000.0,
    )
    assert verdict.measured_pairs == 0
    assert verdict.unmeasured_pairs == 1
    assert not verdict.blocked


def test_the_gate_is_inert_when_disabled(clean_book, monkeypatch):
    monkeypatch.setattr(settings, "CORRELATION_RISK_ENABLED", False)
    _position(clean_book, "AAA", "MintA", size_usd=9_000.0)
    _prices(clean_book, "AAA", "MintA", LOCKSTEP)
    _prices(clean_book, "BBB", "MintB", LOCKSTEP)
    clean_book.commit()

    verdict = book.gate(
        clean_book, symbol="BBB", token_address="MintB",
        portfolio_value_usd=10_000.0, proposed_size_usd=1_000.0,
    )
    assert not verdict.blocked


# ---------------------------------------------------------------------------
# the report, which is conservative where the gate is evidential
# ---------------------------------------------------------------------------

def test_the_report_counts_an_unmeasured_pair_as_fully_correlated(clean_book):
    """Opposite default to the gate, on purpose. A book whose correlations
    are unknown must not be DESCRIBED as diversified."""
    _position(clean_book, "AAA", "MintA", size_usd=5_000.0)
    _position(clean_book, "BBB", "MintB", size_usd=5_000.0)
    clean_book.commit()

    report = book.report(clean_book)
    assert report.positions == 2
    assert report.unknown_pairs == 1
    assert report.effective_exposure_usd == pytest.approx(report.gross_exposure_usd)
    assert report.independent_bets == pytest.approx(1.0)


def test_a_measurably_uncorrelated_book_reports_more_than_one_bet(clean_book):
    _position(clean_book, "AAA", "MintA", size_usd=5_000.0)
    _position(clean_book, "BBB", "MintB", size_usd=5_000.0)
    _prices(clean_book, "AAA", "MintA", LOCKSTEP)
    _prices(clean_book, "BBB", "MintB", INVERSE)
    clean_book.commit()

    report = book.report(clean_book)
    assert report.unknown_pairs == 0
    assert report.effective_exposure_usd < report.gross_exposure_usd
    assert report.independent_bets > 1.0


def test_an_unpriced_position_is_valued_at_cost_not_dropped(clean_book):
    """Dropping it from the book would understate exposure. Valuing it at
    entry is the honest-but-imperfect option and is what the rest of the
    valuation code already does."""
    position = _position(clean_book, "AAA", "MintA", size_usd=2_500.0)
    clean_book.commit()

    assert not wl_price_series(clean_book, "MintA"), "no stored price for this token"
    assert book._exposure_usd(clean_book, position) == pytest.approx(2_500.0)


def test_a_stored_price_marks_the_position_to_market(clean_book):
    position = _position(clean_book, "AAA", "MintA", size_usd=1_000.0, entry_price=0.001)
    _prices(clean_book, "AAA", "MintA", [0.002])      # doubled since entry (0.001)
    clean_book.commit()

    assert book._exposure_usd(clean_book, position) == pytest.approx(2_000.0)
