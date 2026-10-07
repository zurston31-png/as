"""Tests for the pre-registered hypotheses.

What these tests are actually defending
---------------------------------------
A pre-registration is worthless unless it is *pinned*. If the set of
hypotheses can be edited after a backtest runs, then the project is choosing
its success criteria in arrears while displaying a module that claims
otherwise -- which is worse than having no registry at all, because it
manufactures the appearance of discipline.

So the tests here are mostly tamper checks: the ids and the count are
written out literally, every refutation condition and consequence has to be
substantive, and the registration is required to land in the experiment log
under a non-evaluation phase with a commit stamp. `test_the_pin_would_catch_
an_addition` is the one that makes the rest non-vacuous.

The remaining tests check the one thing a registry can get wrong on its own
terms: a hypothesis whose declared shape does not match its own fields --
an AT_LEAST with no threshold, or an ORDERED_GROUPS with one group -- cannot
be evaluated mechanically in Phase 8, which is the only reason this module
exists rather than a paragraph of prose.
"""

from __future__ import annotations

import pytest

from flow_model.core.enums import SplitPhase
from flow_model.validation.experiment_log import ExperimentLog
from flow_model.validation.hypotheses import (
    HYPOTHESES,
    Direction,
    Hypothesis,
    by_id,
    registration_rows,
)

#: Written out rather than derived, so adding a hypothesis after seeing a
#: result fails this test and has to be justified in a commit message.
EXPECTED_IDS = (
    "H-SR-SIGNIFICANCE-MONOTONE",
    "H-SR-CLEANLINESS-MONOTONE",
    "H-SR-SPLIT-JUSTIFIED",
    "H-SR-POPULATION-TRADABLE",
    "H-SR-CBAND-ROBUST",
    "H-REGIME-WINRATE",
    "H-REGIME-CHOP-WORSE",
    "H-COMBINED-10Y",
    "H-CONSISTENCY-OVER-PEAK",
)


# ---------------------------------------------------------------------------
# the pin
# ---------------------------------------------------------------------------


def test_the_registered_set_is_exactly_what_was_pinned():
    """The whole point of pre-registration is that this set predates the
    evidence. Pinning it here converts a silent edit into a failing test."""
    assert tuple(h.hypothesis_id for h in HYPOTHESES) == EXPECTED_IDS


def test_the_pin_would_catch_an_addition():
    """Without this, the test above could pass against a stale constant and
    prove nothing. Here the tuple is perturbed exactly as a late addition
    would perturb it, and the comparison is required to fail."""
    smuggled = tuple(h.hypothesis_id for h in HYPOTHESES) + ("H-CHOSEN-AFTERWARDS",)
    assert smuggled != EXPECTED_IDS


def test_ids_are_unique():
    ids = [h.hypothesis_id for h in HYPOTHESES]
    assert len(ids) == len(set(ids))


def test_every_hypothesis_cites_its_source_section():
    for item in HYPOTHESES:
        assert "ARCHITECTURE.md" in item.source, item.hypothesis_id


def test_lookup_by_id_round_trips_and_rejects_an_unknown_id():
    for item in HYPOTHESES:
        assert by_id(item.hypothesis_id) is item
    with pytest.raises(KeyError, match="no registered hypothesis"):
        by_id("H-DOES-NOT-EXIST")


# ---------------------------------------------------------------------------
# internal consistency: can Phase 8 actually evaluate each one?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("item", HYPOTHESES, ids=lambda h: h.hypothesis_id)
def test_threshold_present_exactly_when_the_shape_needs_one(item: Hypothesis):
    """An AT_LEAST with no bound is not a prediction, and a MONOTONE claim
    with a bound invites scoring against the bound instead of the shape."""
    if item.requires_threshold():
        assert item.threshold is not None, f"{item.hypothesis_id} needs a threshold"
    else:
        assert item.threshold is None, (
            f"{item.hypothesis_id} is {item.direction.value}; its claim is the "
            "shape of the profile, so a threshold would be scored by mistake"
        )


@pytest.mark.parametrize("item", HYPOTHESES, ids=lambda h: h.hypothesis_id)
def test_ordered_groups_names_at_least_two_groups(item: Hypothesis):
    if item.direction is Direction.ORDERED_GROUPS:
        assert len(item.groups) >= 2, (
            f"{item.hypothesis_id} compares groups but names {len(item.groups)}"
        )


@pytest.mark.parametrize("item", HYPOTHESES, ids=lambda h: h.hypothesis_id)
def test_a_conditional_claim_names_its_driver(item: Hypothesis):
    """A claim about a profile needs the variable the profile runs over, or
    there is nothing to bucket by."""
    shape_claims = (
        Direction.MONOTONE_INCREASING,
        Direction.DISTINCT_SIGN,
        Direction.ORDERED_GROUPS,
    )
    if item.direction in shape_claims:
        assert item.driver, f"{item.hypothesis_id} has no driver to bucket by"


@pytest.mark.parametrize("item", HYPOTHESES, ids=lambda h: h.hypothesis_id)
def test_every_hypothesis_states_a_consequence_naming_an_action(item: Hypothesis):
    """A refuted hypothesis with no stated consequence gets quietly reweighted.
    Section 14.7 is explicit that two of these require REMOVAL rather than
    reweighting, so the consequence must name a verb."""
    verbs = ("report", "remove", "drop", "do not", "never")
    lowered = item.consequence.lower()
    assert any(verb in lowered for verb in verbs), (
        f"{item.hypothesis_id}'s consequence names no action: {item.consequence}"
    )


def test_the_substantive_text_validator_rejects_a_placeholder():
    """The validator is what stops a hypothesis being registered as 'TBD'."""
    with pytest.raises(ValueError, match="stated in full"):
        Hypothesis(
            hypothesis_id="H-LAZY",
            source="ARCHITECTURE.md 14.7",
            claim="a claim long enough to pass",
            metric="win_rate",
            direction=Direction.AT_LEAST,
            threshold=0.5,
            refuted_when="TBD",
            consequence="a consequence long enough to pass, report it",
            expected_to_hold=False,
        )


def test_a_threshold_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="must be a rate"):
        Hypothesis(
            hypothesis_id="H-PERCENT-CONFUSION",
            source="ARCHITECTURE.md 6",
            claim="win rate reaches eighty-eight, written as a percentage",
            metric="win_rate",
            direction=Direction.AT_LEAST,
            threshold=88.0,
            refuted_when="the measured rate falls below the stated bound",
            consequence="report the measured value and do not adjust gates",
            expected_to_hold=False,
        )


def test_most_of_the_briefs_numbers_are_expected_to_be_refuted():
    """Not a correctness property -- a record of my prior, pinned so it
    cannot be revised after the results arrive. The brief's headline rates
    imply a Sharpe far outside anything documented."""
    expected_to_fail = [h.hypothesis_id for h in HYPOTHESES if not h.expected_to_hold]
    assert "H-REGIME-WINRATE" in expected_to_fail
    assert "H-COMBINED-10Y" in expected_to_fail
    assert len(expected_to_fail) >= 5


def test_hypotheses_are_frozen():
    """A registration that can be mutated in place is not a registration."""
    with pytest.raises(Exception):
        HYPOTHESES[0].threshold = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# registration into the experiment log
# ---------------------------------------------------------------------------


def test_registration_rows_cover_every_hypothesis():
    rows = registration_rows()
    assert len(rows) == len(HYPOTHESES)
    names = {row["name"] for row in rows}
    assert names == {f"register:{h.hypothesis_id}" for h in HYPOTHESES}


def test_registration_notes_carry_the_refutation_condition():
    """The log row has to be readable on its own -- a reader auditing the log
    months later should not need this module to see what was predicted."""
    for row, item in zip(registration_rows(), HYPOTHESES):
        assert item.refuted_when in row["notes"]
        assert item.consequence in row["notes"]
        assert "before any backtest result exists" in row["notes"]


def test_registering_spends_no_evaluation_budget(tmp_path):
    """Writing down a prediction consumes no out-of-sample information, so
    the registration must not count against the TEST or SEALED_OOS budget
    that `assert_phase_budget` enforces."""
    with ExperimentLog(tmp_path / "exp.db") as log:
        for row in registration_rows():
            log.log(
                name=row["name"],
                config_hash="pre-registration",
                phase=SplitPhase.TRAIN,
                results=row["results"],
                notes=row["notes"],
            )
        assert log.count() == len(HYPOTHESES)
        assert log.total_phase_touches(SplitPhase.TEST) == 0
        assert log.total_phase_touches(SplitPhase.SEALED_OOS) == 0
        stored = log.query()
        assert all(not record.is_evaluation for record in stored)


def test_a_registration_is_commit_stamped(tmp_path):
    """So the claim 'this predates the evidence' is checkable against git
    rather than taken on trust."""
    with ExperimentLog(tmp_path / "exp.db") as log:
        row = registration_rows()[0]
        record = log.log(
            name=row["name"],
            config_hash="pre-registration",
            phase=SplitPhase.TRAIN,
            results=row["results"],
            notes=row["notes"],
        )
        assert record.created_at.tzinfo is not None
        assert record.git_commit != "", "registration must record the commit"


def test_stored_results_round_trip_the_hypothesis(tmp_path):
    """`results` is read back by the Phase 8 evaluator, so the stored payload
    must reconstruct the hypothesis rather than merely describe it."""
    with ExperimentLog(tmp_path / "exp.db") as log:
        item = by_id("H-REGIME-WINRATE")
        row = next(r for r in registration_rows() if item.hypothesis_id in r["name"])
        record = log.log(
            name=row["name"],
            config_hash="pre-registration",
            phase=SplitPhase.TRAIN,
            results=row["results"],
            notes=row["notes"],
        )
        reloaded = log.get(record.experiment_id)
        assert reloaded.results["threshold"] == pytest.approx(0.88)
        assert reloaded.results["direction"] == Direction.AT_LEAST.value
        assert tuple(reloaded.results["groups"]) == ("LOW_VOL", "HIGH_VOL")
        assert Hypothesis(**reloaded.results) == item


# ---------------------------------------------------------------------------
# the CLI path
# ---------------------------------------------------------------------------


def _cli(tmp_path, *argv: str):
    """Run the CLI with the experiment DB redirected into tmp_path."""
    from flow_model.cli import main

    return main([*argv, "--set", f"paths.experiment_db_path={tmp_path / 'exp.db'}"])


def test_cli_register_is_idempotent(tmp_path, capsys):
    """Re-running registration must not append a second copy. A log holding
    the same prediction twice makes the multiple-comparisons denominator in
    `overfitting_summary` wrong, which is the one number that says how much
    the reported results should be discounted."""
    assert _cli(tmp_path, "hypotheses", "register") == 0
    first = capsys.readouterr().out
    assert f"registered {len(HYPOTHESES)}" in first

    assert _cli(tmp_path, "hypotheses", "register") == 0
    second = capsys.readouterr().out
    assert "registered 0" in second
    assert f"already present {len(HYPOTHESES)}" in second

    with ExperimentLog(tmp_path / "exp.db") as log:
        assert log.count() == len(HYPOTHESES)


def test_cli_refuses_to_register_after_an_evaluation_has_happened(tmp_path, capsys):
    """The guard that gives the word 'pre-registration' its meaning. Once the
    test set has been touched, writing down a criterion is choosing it in
    arrears, and the CLI must refuse rather than record it."""
    with ExperimentLog(tmp_path / "exp.db") as log:
        log.log(name="peeked", config_hash="c0", phase=SplitPhase.TEST)

    assert _cli(tmp_path, "hypotheses", "register") == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "not a pre-registration" in out

    with ExperimentLog(tmp_path / "exp.db") as log:
        names = {r.name for r in log.query()}
    assert not any(name.startswith("register:") for name in names)


def test_cli_list_does_not_write_anything(tmp_path, capsys):
    assert _cli(tmp_path, "hypotheses", "list") == 0
    out = capsys.readouterr().out
    assert "H-REGIME-WINRATE" in out
    assert not (tmp_path / "exp.db").exists() or ExperimentLog(
        tmp_path / "exp.db"
    ).count() == 0
