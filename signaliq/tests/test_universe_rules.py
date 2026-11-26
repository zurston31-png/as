"""The two proposed rules (measured, not adopted): dated types and the co-registrant choice."""

from datetime import date

from app.universe import rules


def page(day, kind, why=""):
    return rules.DatedClass(date.fromisoformat(day), "CRH", "1", "20-F", "t", kind, why, "u")


def unread(day):
    return page(day, None, rules.NOT_READ)


def test_both_rules_are_off_by_default(monkeypatch):
    monkeypatch.delenv(rules.DATED_TYPES_VARIABLE, raising=False)
    monkeypatch.delenv(rules.COREGISTRANT_VARIABLE, raising=False)
    assert rules.enabled_rules() == []


def test_a_dated_type_uses_only_cover_pages_filed_by_the_day():
    pages = [page("2020-03-06", "adr"), page("2023-10-01", "common")]
    assert rules.type_on(pages, date(2020, 1, 31)) is None  # before any page: undated type
    assert rules.type_on(pages, date(2023, 9, 29)) == "adr"  # the later page is not known yet
    assert rules.type_on(pages, date(2023, 10, 31)) == "common"


def test_unread_pages_after_the_latest_reading_make_the_type_unknown():
    pages = [page("2020-03-06", "adr"), unread("2022-03-01"), page("2024-12-30", "common")]
    kind, why = rules.dated_type(pages, date(2023, 9, 29))
    assert kind == "unknown"
    assert "not read" in why
    assert rules.type_on(pages, date(2021, 1, 29)) == "adr"


def test_only_securities_whose_conclusive_pages_disagree_count_as_changes():
    classes = {
        1: [page("2020-01-01", "common"), page("2024-01-01", "common"), unread("2022-01-01")],
        2: [page("2020-01-01", "adr"), page("2024-01-01", "common")],
        3: [page("2020-01-01", "adr"), page("2021-01-01", None, "not one recognised class")],
    }
    assert set(rules.type_changes(classes)) == {2}


def test_type_runs_group_consecutive_conclusive_pages():
    pages = [page("2019-05-09", "partnership"), unread("2019-06-01"), page("2019-07-05", "common")]
    runs = rules.type_runs([*pages, page("2024-12-23", "common")])
    assert [(k, n) for k, _, _, n in runs] == [("partnership", 1), ("common", 2)]


def test_company_names_are_compared_as_words():
    assert rules.name_tokens("PG&E Corp") == ["PG", "AND", "E"]
    assert rules.name_tokens("CHARTER COMMUNICATIONS, INC. /MO/") == ["CHARTER", "COMMUNICATIONS"]
    assert rules.name_tokens("United Rentals North America Inc") == [
        "UNITED",
        "RENTALS",
        "NORTH",
        "AMERICA",
    ]


def test_the_registrant_block_leaves_out_hidden_values_before_the_cover():
    text = (
        "false CCO HOLDINGS LLC CHARTER COMMUNICATIONS, INC. /MO/ 0001091667 FORM 8-K Date of "
        "Report (Date of earliest event reported): December 13, 2024 Charter Communications, "
        "Inc. CCO Holdings, LLC (Exact name of registrant as specified in its charter)"
    )
    assert (
        rules.registrant_block(text).strip(" (") == "Charter Communications, Inc. CCO Holdings, LLC"
    )


def test_a_row_naming_one_company_decides():
    text = (
        "Title of each class Trading Symbol(s) Name of each exchange "
        "Common Stock, $.01 par value, of United Rentals, Inc. URI New York Stock Exchange"
    )
    side, how = rules._vote(
        text, "URI", rules.name_tokens("United Rentals North America Inc"),
        rules.name_tokens("United Rentals, Inc."),
    )  # fmt: skip
    assert side == "b"
    assert "names it" in how


def test_a_name_that_starts_a_longer_name_is_not_found_inside_it():
    text = (
        "Title of each class Trading Symbol(s) Name of each exchange United Airlines Holdings, "
        "Inc. Common Stock, $0.01 par value UAL The Nasdaq Stock Market LLC"
    )
    side, _ = rules._vote(
        text, "UAL", rules.name_tokens("United Airlines Holdings, Inc."),
        rules.name_tokens("United Airlines, Inc."),
    )  # fmt: skip
    assert side == "a"


def test_named_first_decides_when_the_row_names_neither():
    text = (
        "Date of Report (Date of earliest event reported): December 13, 2024 Charter "
        "Communications, Inc. CCO Holdings, LLC (Exact name of registrant as specified) "
        "Title of each class Trading Symbol(s) Name of each exchange Class A Common Stock, "
        "$.001 Par Value CHTR NASDAQ Global Select Market"
    )
    side, how = rules._vote(
        text, "CHTR", rules.name_tokens("CCO Holdings LLC"),
        rules.name_tokens("Charter Communications, Inc. /MO/"),
    )  # fmt: skip
    assert side == "b"
    assert how == "named first among the registrants"
