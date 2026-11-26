"""The nesting test: a larger run's top K must equal the smaller run's members."""

import importlib.util
from datetime import date
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_stage_versions.py"
_spec = importlib.util.spec_from_file_location("compare_stage_versions", _PATH)
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

DAY = date(2021, 1, 29)


def row(sid, rank, member, eligible=True, reason=None):
    # (decision, security_id, eligible, reason, rank, member), as the membership table holds
    return (DAY, sid), (DAY, sid, int(eligible), reason, rank, int(member))


def rows(*items):
    return dict(row(*i) for i in items)


def test_a_larger_run_whose_top_k_are_the_old_members_is_nested():
    old = rows((1, 1, True), (2, 2, True), (3, 3, False))
    new = rows((1, 1, True), (2, 2, True), (3, 3, True), (4, 4, True))
    assert compare.nesting(old, new, 2) == {}


def test_a_security_that_changed_rank_or_eligibility_is_reported():
    old = rows((1, 1, True), (2, 2, True))
    # Security 2 became ineligible in the new run (say, a new type source), 3 moved up.
    new = rows((1, 1, True), (3, 2, True), (2, None, False, False, "not_common_stock"))
    assert compare.nesting(old, new, 2) == {DAY: ({1, 2}, {1, 3})}
