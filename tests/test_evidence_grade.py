"""Tests for evidence grading (app/analysis/evidence_grade.py).

The invariants:

  E1  The boundaries mean what the module says they mean: each one is the
      sample size at which the 95% confidence half-width on a proportion
      crosses 18, 10 and 5 percentage points.
  E2  A grade describes PRECISION, never quality. A precisely measured
      loss grades STRONG.
  E3  INSUFFICIENT withholds the value but never the count - "12
      observations so far" is honest where "win rate 58%" over twelve is
      not.
  E4  A set of statistics is graded by its weakest member, so three
      well-sampled numbers cannot launder one with almost no data.
  E5  A mean of a heavy-tailed distribution says its grade is optimistic.
  E6  The gap to the next grade is reported as work remaining, which is
      the actionable form during a collection run.
"""
import math

import pytest

from app.analysis.evidence_grade import (
    EARLY_AT,
    STRONG_AT,
    USABLE_AT,
    EvidenceLevel,
    Graded,
    classify,
    half_width_pp,
    shortfall_to_next,
    weakest,
)


# ---------------------------------------------------------------------------
# E1 - the boundaries are derived, not chosen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("samples,expected", [
    (0, EvidenceLevel.INSUFFICIENT),
    (1, EvidenceLevel.INSUFFICIENT),
    (EARLY_AT - 1, EvidenceLevel.INSUFFICIENT),
    (EARLY_AT, EvidenceLevel.EARLY),
    (USABLE_AT - 1, EvidenceLevel.EARLY),
    (USABLE_AT, EvidenceLevel.USABLE),
    (STRONG_AT - 1, EvidenceLevel.USABLE),
    (STRONG_AT, EvidenceLevel.STRONG),
    (10_000, EvidenceLevel.STRONG),
])
def test_the_ladder_boundaries_are_inclusive_at_the_bottom(samples, expected):
    assert classify(samples) is expected


def test_the_half_width_matches_the_standard_error_formula():
    """E1. The label is only meaningful if it tracks a real interval, so
    this checks the arithmetic rather than the constant."""
    for n in (30, 100, 385, 1000):
        assert half_width_pp(n) == pytest.approx(1.96 * math.sqrt(0.25 / n) * 100)


def test_each_boundary_is_where_the_interval_crosses_its_stated_width():
    """The docstring claims 30/100/385 are the points where the 95%
    half-width crosses 18, 10 and 5 points. If someone changes a constant
    without redoing the derivation, this is what catches it."""
    assert half_width_pp(EARLY_AT) == pytest.approx(17.9, abs=0.2)
    assert half_width_pp(USABLE_AT) == pytest.approx(9.8, abs=0.2)
    assert half_width_pp(STRONG_AT) == pytest.approx(5.0, abs=0.1)


def test_the_interval_narrows_as_the_sample_grows():
    widths = [half_width_pp(n) for n in (30, 100, 385, 1000)]
    assert widths == sorted(widths, reverse=True)


def test_a_zero_sample_has_no_interval_rather_than_an_infinite_one():
    """Undefined, not enormous. Reporting a huge number would imply the
    quantity was measured and found imprecise."""
    assert half_width_pp(0) is None


# ---------------------------------------------------------------------------
# E2 - precision is not quality
# ---------------------------------------------------------------------------

def test_a_well_measured_loss_grades_strong():
    """E2, and the misreading this module most needs to prevent. STRONG
    means the number is pinned down. If the number is -40%, STRONG means
    the bot is confidently losing."""
    losing = Graded("expectancy", -40.0, samples=500, unit="%")
    assert losing.level is EvidenceLevel.STRONG
    assert losing.reportable
    assert "says nothing about whether the number itself is good" in losing.caveat()


def test_a_spectacular_result_on_five_observations_is_insufficient():
    """The other direction: a great number from almost no data is not a
    finding, however good it looks."""
    lucky = Graded("expectancy", 250.0, samples=5, unit="%")
    assert lucky.level is EvidenceLevel.INSUFFICIENT
    assert not lucky.reportable


# ---------------------------------------------------------------------------
# E3 - what gets withheld
# ---------------------------------------------------------------------------

def test_insufficient_withholds_the_value_but_reports_the_count():
    """E3. The count is a fact worth having; the statistic is not."""
    g = Graded("win rate", 58.3, samples=12, unit="%")
    assert not g.reportable
    assert g.as_dict()["value"] is None
    assert g.as_dict()["samples"] == 12
    assert "12 observations" in g.summary()


def test_a_missing_value_is_never_reportable_at_any_sample_size():
    """A statistic that could not be computed stays uncomputed. A large n
    does not conjure a value out of None."""
    assert not Graded("MFE", None, samples=5_000).reportable


def test_a_reportable_summary_carries_the_grade_and_the_sample():
    g = Graded("win rate", 58.3, samples=200, unit="%")
    line = g.summary()
    assert "58.30%" in line
    assert "USABLE" in line
    assert "n=200" in line


# ---------------------------------------------------------------------------
# E4 - a set is only as good as its weakest member
# ---------------------------------------------------------------------------

def test_a_set_is_graded_by_its_weakest_member():
    """E4. Averaging the grades would let three well-sampled numbers carry
    a conclusion that actually rests on the fourth."""
    assert weakest([
        Graded("a", 1.0, samples=5_000),
        Graded("b", 1.0, samples=5_000),
        Graded("c", 1.0, samples=5_000),
        Graded("d", 1.0, samples=4),
    ]) is EvidenceLevel.INSUFFICIENT


def test_an_empty_set_is_insufficient_not_strong():
    """Vacuous truth is the wrong default here: "no measurements failed
    the bar" must not read as "every measurement passed"."""
    assert weakest([]) is EvidenceLevel.INSUFFICIENT


def test_levels_order_from_insufficient_up_to_strong():
    order = [
        EvidenceLevel.INSUFFICIENT, EvidenceLevel.EARLY,
        EvidenceLevel.USABLE, EvidenceLevel.STRONG,
    ]
    assert sorted(order, key=lambda level: level.rank) == order
    assert EvidenceLevel.EARLY < EvidenceLevel.STRONG


# ---------------------------------------------------------------------------
# E5 - heavy tails
# ---------------------------------------------------------------------------

def test_a_mean_warns_that_its_grade_is_optimistic():
    """E5. The ladder is derived for a proportion. A mean of memecoin
    returns keeps moving long after a proportion has settled, because one
    100x can outweigh two hundred small losses."""
    mean = Graded("expectancy", 3.0, samples=500, unit="%", is_mean=True)
    assert "optimistic ceiling" in mean.caveat()

    proportion = Graded("win rate", 55.0, samples=500, unit="%")
    assert "optimistic ceiling" not in proportion.caveat()


def test_an_insufficient_mean_does_not_bother_with_the_tail_caveat():
    """Below the floor the value is withheld anyway - qualifying a number
    nobody is being shown is just noise."""
    assert "optimistic ceiling" not in Graded("expectancy", 3.0, samples=4, is_mean=True).caveat()


# ---------------------------------------------------------------------------
# E6 - the gap to the next grade
# ---------------------------------------------------------------------------

def test_the_gap_is_expressed_as_work_remaining():
    """E6. During a collection run the useful question is always how many
    more, not how many in total."""
    assert shortfall_to_next(0) == (EvidenceLevel.EARLY, EARLY_AT)
    assert shortfall_to_next(25) == (EvidenceLevel.EARLY, 5)
    assert shortfall_to_next(30) == (EvidenceLevel.USABLE, USABLE_AT - 30)
    assert shortfall_to_next(384) == (EvidenceLevel.STRONG, 1)


def test_there_is_nothing_above_strong():
    assert shortfall_to_next(STRONG_AT) is None
    assert shortfall_to_next(1_000_000) is None


def test_the_withheld_summary_says_how_much_further_there_is_to_go():
    line = Graded("win rate", 58.3, samples=12, unit="%").summary()
    assert "18 more for EARLY" in line


# ---------------------------------------------------------------------------
# integration with the evidence report
# ---------------------------------------------------------------------------

def test_the_evidence_report_grades_itself_by_its_performance_measures(clean_db):
    """A report over an empty dataset is INSUFFICIENT, whatever else it
    happens to have counted."""
    from app.analysis.evidence import build_evidence_report

    report = build_evidence_report(clean_db)
    assert report.overall_level is EvidenceLevel.INSUFFICIENT
    assert report.as_dict()["evidence_level"] == "INSUFFICIENT"


def test_a_pile_of_rejections_does_not_grade_the_findings(clean_db):
    """"evaluated signals" counts things that happened and needs no
    confidence interval. Letting it into the grade would mean a busy
    scanner made an empty result set look well evidenced."""
    from app.analysis.evidence import Measure, EvidenceReport

    report = EvidenceReport(measures=[
        Measure("evaluated signals", 5000.0, 5000, floor=1),
        Measure("expectancy (net)", 3.0, 4, "%", is_mean=True),
        Measure("median return", 2.0, 4, "%"),
        Measure("win rate", 60.0, 4, "%"),
    ])
    assert report.overall_level is EvidenceLevel.INSUFFICIENT


def test_every_measure_reports_its_grade_in_the_dict(clean_db):
    from app.analysis.evidence import Measure

    payload = Measure("win rate", 55.0, 150, "%").as_dict()
    assert payload["level"] == "USABLE"
    assert payload["next_level"] == "STRONG"
    assert payload["samples_to_next_level"] == STRONG_AT - 150
    assert payload["half_width_pp"] == pytest.approx(8.0, abs=0.1)
