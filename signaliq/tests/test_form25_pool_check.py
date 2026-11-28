"""The Form 25 pool cross-check: parsing EDGAR's form index and classifying each filer."""

import importlib.util
from datetime import date
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "form25_pool_check.py"
_spec = importlib.util.spec_from_file_location("form25_pool_check", _PATH)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

INDEX = """Description:           Master Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    March 31, 2021

Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
10-K        ALPHA CORP                                                    1111111     2021-02-01  edgar/data/1111111/0001111111-21-000001.txt
25          BETA INC                                                      2222222     2021-02-02  edgar/data/2222222/0002222222-21-000001.txt
25-NSE      GAMMA HOLDINGS                                                3333333     2021-03-03  edgar/data/3333333/0001234567-21-000001.txt
25-NSE      DELTA LTD                                                     4444444     2021-03-04  edgar/data/4444444/0001234567-21-000002.txt
25-NSE      EPSILON CO                                                    5555555     2021-03-05  edgar/data/5555555/0001234567-21-000003.txt
25/A        ZETA CORP                                                     6666666     2021-03-06  edgar/data/6666666/0001234567-21-000004.txt
"""  # noqa: E501


def test_only_removals_in_the_period_are_read():
    removals = check.parse_form_index(INDEX)
    assert [(r.form, r.company, r.cik, r.filed) for r in removals] == [
        ("25", "BETA INC", "0002222222", date(2021, 2, 2)),
        ("25-NSE", "GAMMA HOLDINGS", "0003333333", date(2021, 3, 3)),
        ("25-NSE", "DELTA LTD", "0004444444", date(2021, 3, 4)),
        ("25-NSE", "EPSILON CO", "0005555555", date(2021, 3, 5)),
    ]


def test_every_filer_is_classified_and_an_unknown_one_is_never_counted_as_covered():
    removals = check.parse_form_index(INDEX)
    known = {"0003333333": {"GMA"}, "0004444444": {"DLT"}, "0005555555": None}
    groups = check.classify(
        removals,
        issuer_tickers={"0002222222": {"BETA"}},
        pool={"GMA"},
        tickers_of=lambda cik: known.get(cik),
    )
    assert [r.company for r, _ in groups["in_batch"]] == ["BETA INC"]
    assert [r.company for r, _ in groups["in_pool"]] == ["GAMMA HOLDINGS"]
    assert [(r.company, t) for r, t in groups["gap"]] == [("DELTA LTD", {"DLT"})]
    assert [r.company for r, _ in groups["unresolved"]] == ["EPSILON CO"]


def test_the_period_covers_every_quarter_feeding_a_decision():
    q = check.quarters()
    assert q[0] == (2019, 4)
    assert q[-1] == (2024, 4)
    assert len(q) == 21
