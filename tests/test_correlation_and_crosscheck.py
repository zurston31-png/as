"""Tests for correlation risk (Phase 15) and source cross-validation (27).

Both modules exist to resist the same instinct: assuming the convenient
thing when the data does not actually say it. Correlation assumes tokens
move together unless measured otherwise; cross-check refuses to trade
rather than picking whichever source permits a bigger position.
"""
import pytest

from app.data.cross_check import Agreement, compare
from app.risk.correlation import (
    HIGH_CORRELATION,
    MIN_OVERLAP,
    analyse,
    pearson,
    returns_from_prices,
)


def _series(n=60, step=0.01, sign=1):
    """A deterministic alternating return series."""
    return [sign * step * (1 if i % 2 else -1) for i in range(n)]


# ===========================================================================
# correlation
# ===========================================================================

def test_returns_are_computed_from_consecutive_prices():
    assert returns_from_prices([100.0, 110.0, 99.0]) == pytest.approx([0.10, -0.10])


def test_a_non_positive_price_does_not_produce_an_infinity():
    assert returns_from_prices([100.0, 0.0, 50.0]) == pytest.approx([-1.0])


def test_identical_series_correlate_perfectly():
    s = _series()
    assert pearson(s, s) == pytest.approx(1.0)


def test_opposite_series_correlate_negatively():
    s = _series()
    assert pearson(s, [-x for x in s]) == pytest.approx(-1.0)


def test_too_little_overlap_is_unknown_not_zero():
    """Two points always correlate perfectly. Reporting a number from them
    would be worse than reporting nothing."""
    assert pearson([0.01, -0.01], [0.01, -0.01]) is None
    assert pearson(_series(MIN_OVERLAP - 1), _series(MIN_OVERLAP - 1)) is None
    assert pearson(_series(MIN_OVERLAP), _series(MIN_OVERLAP)) is not None


def test_a_flat_series_correlates_with_nothing():
    """A constant has no correlation. Reporting 0 would claim independence
    that was never measured."""
    assert pearson([0.0] * 40, _series(40)) is None


def test_uncorrelated_positions_diversify():
    """Four independent equal bets should be worth about half their gross."""
    import random

    rng = random.Random(11)
    series = {k: [rng.gauss(0, 0.02) for _ in range(200)] for k in "ABCD"}
    report = analyse({k: 100.0 for k in "ABCD"}, series)

    assert report.gross_exposure_usd == pytest.approx(400.0)
    assert report.effective_exposure_usd < report.gross_exposure_usd
    assert 0.3 < report.diversification_ratio < 0.8
    assert report.independent_bets > 1.5


def test_perfectly_correlated_positions_are_one_bet():
    """The whole point: five names moving together are one position at five
    times the size, and every per-position limit is satisfied throughout."""
    s = _series(80)
    series = {k: list(s) for k in "ABCDE"}
    report = analyse({k: 100.0 for k in "ABCDE"}, series)

    assert report.effective_exposure_usd == pytest.approx(report.gross_exposure_usd)
    assert report.diversification_ratio == pytest.approx(1.0)
    assert report.independent_bets == pytest.approx(1.0)
    assert len(report.high_pairs) == 10          # every pair


def test_unmeasurable_pairs_are_assumed_correlated_not_independent():
    """Defaulting to independence is how a book that is secretly one
    position passes every check."""
    report = analyse(
        {"A": 100.0, "B": 100.0},
        {"A": [0.01, -0.01], "B": [0.01, -0.01]},    # too short to measure
    )
    assert report.unknown_pairs == 1
    assert report.effective_exposure_usd == pytest.approx(report.gross_exposure_usd)
    assert "conservative direction" in report.summary()


def test_high_correlation_pairs_are_named():
    s = _series(80)
    report = analyse({"A": 100.0, "B": 100.0}, {"A": s, "B": list(s)})
    assert report.high_pairs
    assert report.high_pairs[0].correlation >= HIGH_CORRELATION


def test_a_single_position_needs_no_correlation_analysis():
    report = analyse({"A": 100.0}, {"A": _series()})
    assert report.positions == 1
    assert "not yet a question" in report.summary()


def test_an_empty_book_does_not_divide_by_zero():
    report = analyse({}, {})
    assert report.gross_exposure_usd == 0
    assert report.diversification_ratio is None
    assert report.independent_bets is None


def test_the_report_serialises():
    import json

    s = _series(80)
    json.dumps(analyse({"A": 100.0, "B": 50.0}, {"A": s, "B": list(s)}).as_dict(), allow_nan=False)


# ===========================================================================
# cross-check
# ===========================================================================

def test_close_readings_agree():
    check = compare(price=(0.00420, 0.00425))
    assert check.trustworthy
    assert check.reason == "sources agree"


def test_a_material_disagreement_blocks_trading():
    """At least one source is wrong and there is no way to tell which."""
    check = compare(price=(0.0042, 0.0100))
    assert not check.trustworthy
    assert "DISAGREE" in check.reason
    assert "no way to tell which" in check.reason


def test_the_gap_is_measured_against_the_smaller_value():
    """A source saying $10k when another says $100k is a 900% error.
    Dividing by the larger would report a forgiving 90%."""
    a = Agreement("liquidity", 10_000.0, 100_000.0, tolerance=0.30)
    assert a.relative_gap == pytest.approx(9.0)


def test_tolerances_differ_by_field_because_the_sources_do():
    """Price should match closely - both read the same pool. Liquidity and
    volume genuinely differ between providers, and one tight number for all
    three would produce constant false alarms."""
    # 20% apart: intolerable for price, fine for liquidity.
    assert not compare(price=(100.0, 120.0)).trustworthy
    assert compare(liquidity=(100_000.0, 120_000.0)).trustworthy


def test_the_conservative_value_is_chosen_never_an_average():
    """Averaging a right number with a wrong one produces a third wrong
    number and hides that they disagreed."""
    a = Agreement("liquidity", 50_000.0, 90_000.0, tolerance=1.0)
    assert a.conservative(prefer="low") == 50_000.0     # assume the thinner pool

    p = Agreement("price", 0.0040, 0.0044, tolerance=1.0)
    assert p.conservative(prefer="high") == 0.0044      # assume you pay more

    assert 70_000.0 != a.conservative(prefer="low"), "must not average"


def test_conservative_rejects_an_unknown_preference():
    a = Agreement("liquidity", 1.0, 2.0, tolerance=1.0)
    with pytest.raises(ValueError, match="prefer must be"):
        a.conservative(prefer="whichever_is_nicer")


def test_one_source_answering_is_not_a_cross_check():
    check = compare(price=(0.0042, None))
    assert check.uncrossed
    assert "no cross-check possible" in check.reason or check.trustworthy


def test_requiring_two_sources_blocks_when_only_one_answers():
    """'We could not check' is not 'we checked and it was fine'."""
    lenient = compare(price=(0.0042, None), require_two_sources=False)
    strict = compare(price=(0.0042, None), require_two_sources=True)
    assert lenient.trustworthy
    assert not strict.trustworthy


def test_neither_source_answering_is_reported_plainly():
    check = compare(price=(None, None), require_two_sources=True)
    assert not check.agreements[0].comparable
    assert "neither source" in check.agreements[0].message


def test_several_fields_are_checked_together_and_any_one_can_block():
    check = compare(
        price=(0.0042, 0.0043),          # fine
        liquidity=(50_000.0, 500_000.0),  # 10x apart
    )
    assert not check.trustworthy
    assert "liquidity" in check.reason
    assert len(check.agreements) == 2


def test_the_cross_check_serialises():
    import json

    json.dumps(compare(price=(1.0, 1.01), liquidity=(100.0, 900.0)).as_dict(), allow_nan=False)
