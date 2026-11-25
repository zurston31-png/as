"""Evidence rules: split checks, spin-offs, security types and exit outcomes.

SYNTHETIC TEST FIXTURES, except where a test reproduces numbers seen in real
data, which is said next to it (UTX on 2020-04-03). No test calls a network.
"""

from datetime import date

import pytest

from app.universe import adjustments, outcomes, sectypes
from app.universe.outcomes import Evidence, resolve_outcome

# =============================================================================
# Split plausibility
# =============================================================================


@pytest.mark.parametrize(
    ("ratio", "before", "on", "verdict"),
    [
        (4.0, 400.0, 101.0, "consistent"),  # 4-for-1: price about a quarter
        (0.005, 0.15, 30.0, "consistent"),  # 1-for-200 reverse split
        # Real data: Alpaca's UTX "1-for-2 reverse split" on the spin-off day; the price fell.
        (0.5, 86.01, 49.93, "contradicted"),
        # Real data: LILAK 2016-06-21, a "1-for-8" the price ignored.
        (0.124769, 36.35, 35.4, "contradicted"),
        # Real data: Top Ships 2017-06-23, 1-for-15 recorded, price up 5x (collapse on the day).
        (1 / 15, 0.16, 0.8, "magnitude_mismatch"),
        # Real data: Hertz 2016-07-01, 1-for-15 recorded, price up 4x (Herc spun off that day).
        (1 / 15, 11.07, 43.87, "magnitude_mismatch"),
        (10.0, 100.0, 100.0, "contradicted"),  # a 10-for-1 split the price ignores
        (2.0, None, 50.0, "no_prices"),
        (2.0, 100.0, None, "no_prices"),
    ],
)
def test_split_check(ratio, before, on, verdict):
    assert adjustments.check_split(ratio, before, on) == verdict


def test_split_tolerance_is_a_factor_of_two_either_way():
    assert adjustments.check_split(2.0, 100.0, 25.1) == "consistent"  # expected 50, observed 25.1
    assert adjustments.check_split(2.0, 100.0, 99.0) == "consistent"
    assert adjustments.check_split(2.0, 100.0, 101.0) == "contradicted"  # moved the wrong way
    # Down 4x for a 2-for-1: right direction, too large.
    assert adjustments.check_split(2.0, 100.0, 24.9) == "magnitude_mismatch"


def test_a_small_move_in_the_right_direction_does_not_count_as_a_split():
    # A 1-for-10 reverse split with the price up only 20 %: no sign a split happened.
    assert adjustments.check_split(0.1, 10.0, 12.0) == "contradicted"


# =============================================================================
# Spin-offs
# =============================================================================


def test_utx_carrier_otis_spin_off_with_real_prices():
    # Real SIP closes: UTX 86.01 on 2020-04-02; RTX 49.93, CARR 16.92, OTIS 47.32 on 2020-04-03.
    result = adjustments.spin_off_adjustment(86.01, 49.93, [(1.0, 16.92), (0.5, 47.32)])

    assert result.status == "price_consistent"
    assert result.distribution_value == pytest.approx(40.58)
    holder = (49.93 + 40.58) / 86.01 - 1  # what a holder really had: +5.2 %
    assert result.holder_return == pytest.approx(holder)
    assert result.factor == pytest.approx((49.93 + 40.58) / 49.93)
    # The adjusted series shows exactly the holder's return over the ex-date.
    adjusted_before = 86.01 / result.factor
    assert 49.93 / adjusted_before - 1 == pytest.approx(holder)


def test_the_simple_dividend_style_factor_would_overstate_the_utx_return():
    simple_factor = 86.01 / (86.01 - 40.58)
    assert 49.93 / (86.01 / simple_factor) - 1 == pytest.approx(0.099, abs=0.001)  # not +5.2 %


def test_spin_off_records_the_prices_contradict_are_flagged():
    too_much = adjustments.spin_off_adjustment(86.01, 49.93, [(1.0, 16.92), (1.0, 47.32)])  # +33 %
    assert too_much.status == "price_inconsistent"
    worth_more_than_parent = adjustments.spin_off_adjustment(10.0, 5.0, [(1.0, 12.0)])
    assert (worth_more_than_parent.status, worth_more_than_parent.factor) == (
        "price_inconsistent",
        None,
    )


def test_spin_off_without_child_prices_is_not_applied():
    assert adjustments.spin_off_adjustment(86.01, 49.93, [(1.0, None)]).status == "prices_missing"


def test_distribution_factors_apply_before_the_ex_date_only():
    days = [date(2020, 4, 1), date(2020, 4, 2), date(2020, 4, 3), date(2020, 4, 6)]
    factors = adjustments.distribution_factors(days, [(date(2020, 4, 3), 1.9)])
    assert factors == {days[0]: 1.9, days[1]: 1.9, days[2]: 1.0, days[3]: 1.0}


# =============================================================================
# Security types
# =============================================================================


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("Common Stock", "Common Stock", "common"),  # AAPL
        ("ETP", "Mutual Fund", "etf"),  # SPY, BSJN
        ("Equity WRT", "Warrant", "warrant"),  # NOVVW
        ("Right", "Right", "right"),  # CETUR
        ("Unit", "Unit", "unit"),
        ("REIT", "Common Stock", "common"),
        ("ADR", "Depositary Receipt", "adr"),
        ("Preference", "Preferred", "preferred"),
        ("Closed-End Fund", "Closed-End Fund", "fund"),
        ("MLP", "Ltd Part", "partnership"),
        ("Something New", None, "unknown"),
    ],
)
def test_openfigi_types(first, second, expected):
    assert sectypes.figi_type(first, second) == expected


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("NOVVW", "warrant"),
        ("CETUR", "right"),
        ("BACAU", "unit"),
        ("VFIAX", "fund"),
        ("GOOGL", None),
        ("AAPL", None),
        ("META", None),
        ("BRK.B", None),
    ],
)
def test_symbol_suffix_convention_only_for_five_letter_tickers(symbol, expected):
    assert sectypes.suffix_type(symbol) == expected


def test_wikidata_types():
    # Wikidata describes the company: an operating company, whatever it listed.
    assert sectypes.wikidata_type(["public company", "multinational corporation"]) == "company"
    assert sectypes.wikidata_type(["bank holding company"]) == "company"
    # A REIT is a company, not an "investment trust" fund (Crown Castle, Simon Property).
    assert sectypes.wikidata_type(["company", "real estate investment trust"]) == "company"
    assert sectypes.wikidata_type(["closed-end fund"]) == "fund"
    assert sectypes.wikidata_type(["exchange-traded fund"]) == "etf"
    assert sectypes.wikidata_type(["human"]) is None
    assert sectypes.wikidata_type([""]) is None


def test_type_resolution_needs_independent_sources():
    vote = sectypes.TypeVote
    assert sectypes.resolve_type([vote("openfigi", "common"), vote("wikidata", "company")]) == (
        "common",
        "confirmed",
    )
    assert sectypes.resolve_type([vote("openfigi", "etf")]) == ("etf", "single_source")
    assert sectypes.resolve_type(
        [vote("openfigi", "common"), vote("symbol_suffix", "warrant")]
    ) == ("unknown", "conflict")
    assert sectypes.resolve_type([vote("declared", "common")]) == (
        "unknown",
        "unknown",
    )  # a declaration is not evidence
    assert sectypes.resolve_type([]) == ("unknown", "unknown")
    # Two records from the same source are one source.
    assert (
        sectypes.resolve_type([vote("openfigi", "common"), vote("openfigi", "common")])[1]
        == "single_source"
    )


# =============================================================================
# Exit outcomes
# =============================================================================

D = date(2023, 3, 9)


def ev(source, category, cash=None):
    return Evidence(source, category, D, f"{source} {category}", cash_per_share=cash)


def test_unknown_exits_are_never_a_total_loss():
    result = resolve_outcome([])
    assert (result.outcome, result.status, result.assumed_value) == ("unknown", "unresolved", None)
    dated_only = resolve_outcome([ev("wikidata", "delisted_no_reason")])
    assert (dated_only.status, dated_only.assumed_value) == ("unresolved", None)
    assert "listing end dated" in dated_only.note


def test_cash_acquisition_confirmed_by_two_sources():
    result = resolve_outcome([ev("alpaca", "acquired_cash", 54.2), ev("wikidata", "acquired")])
    assert (result.outcome, result.status, result.assumed_value) == (
        "acquired_cash",
        "confirmed",
        54.2,
    )


def test_acquisition_without_terms_has_no_value():
    result = resolve_outcome([ev("wikidata", "acquired")])
    assert (result.outcome, result.status, result.assumed_value) == (
        "acquired",
        "single_source",
        None,
    )


def test_stock_merger_value_is_not_guessed():
    result = resolve_outcome([ev("alpaca", "acquired_stock")])
    assert (result.outcome, result.assumed_value) == ("acquired_stock", None)


def test_confirmed_failure_uses_the_agreed_total_loss():
    result = resolve_outcome([ev("wikidata", "failure"), ev("wikidata", "bankruptcy")])
    assert (result.outcome, result.status, result.assumed_value) == (
        "failure",
        "single_source",
        0.0,
    )
    two = resolve_outcome([ev("alpaca", "bankruptcy"), ev("wikidata", "bankruptcy")])
    assert (two.outcome, two.status, two.assumed_value) == ("bankruptcy", "confirmed", 0.0)
    assert "occasionally recover" in two.note


def test_acquisition_and_failure_evidence_together_is_a_conflict():
    result = resolve_outcome([ev("alpaca", "acquired_cash", 10.0), ev("wikidata", "bankruptcy")])
    assert (result.outcome, result.status, result.assumed_value) == ("conflict", "conflict", None)


def test_a_resumed_ticker_or_a_fragment_is_unresolved_not_a_loss():
    for category in ("resumed", "fragment"):
        result = resolve_outcome([ev("alpaca_bars", category)])
        assert (result.status, result.assumed_value) == ("unresolved", None)


# =============================================================================
# Wikidata parsing
# =============================================================================


def binding(**values):
    return {key: {"value": value} for key, value in values.items()}


def test_wikidata_rows_must_be_us_listings_overlapping_the_trading_period():
    rows = [
        binding(
            item="http://www.wikidata.org/entity/Q1",
            itemLabel="Monsanto",
            exchLabel="New York Stock Exchange",
        ),
        binding(
            item="http://www.wikidata.org/entity/Q2",
            itemLabel="Monrif",
            exchLabel="Italian Stock Exchange",
        ),
        binding(
            item="http://www.wikidata.org/entity/Q3",
            itemLabel="Later company",
            exchLabel="Nasdaq",
            tStart="2025-08-29T00:00:00Z",
        ),  # out of scope
        binding(
            item="http://www.wikidata.org/entity/Q4",
            itemLabel="Earlier company",
            exchLabel="Nasdaq",
            tStart="2001-01-01T00:00:00Z",
            tEnd="2010-01-01T00:00:00Z",
        ),  # ended long before
    ]
    kept = outcomes.wikidata_matches(rows, date(2016, 1, 4), date(2018, 6, 6))
    assert [r["itemLabel"]["value"] for r in kept] == ["Monsanto"]


def test_wikidata_evidence_windows_and_the_2025_cut_off():
    last = date(2019, 7, 8)  # Red Hat's last trade
    row = binding(
        item="http://www.wikidata.org/entity/Q5",
        itemLabel="Red Hat",
        exchLabel="New York Stock Exchange",
        parents="IBM@2019-07-09T00:00:00Z|Old parent@2001-01-01T00:00:00Z",
        tEnd="2019-07-10T00:00:00Z",
        events="bankruptcy@2026-01-01T00:00:00Z",  # after the scope: ignored
    )

    found = outcomes.wikidata_evidence([row], last)

    assert sorted((e.category, e.event_date) for e in found) == [
        ("acquired", date(2019, 7, 9)),
        ("delisted_no_reason", date(2019, 7, 10)),
    ]
    assert resolve_outcome(found).outcome == "acquired"


def test_wikidata_bankruptcy_and_collapse_events():
    row = binding(
        item="http://www.wikidata.org/entity/Q6",
        itemLabel="SVB Financial Group",
        exchLabel="Nasdaq",
        events=(
            "Chapter 11 bankruptcy@2023-03-17T00:00:00Z"
            "|collapse of Silicon Valley Bank@2023-03-10T00:00:00Z"
            "|initial public offering@1988-01-01T00:00:00Z"
        ),
    )
    found = outcomes.wikidata_evidence([row], D)
    assert sorted(e.category for e in found) == ["bankruptcy", "failure"]
    assert resolve_outcome(found).outcome == "failure"


def test_an_owner_long_before_the_end_is_not_an_acquisition():
    row = binding(
        item="http://www.wikidata.org/entity/Q7",
        itemLabel="X",
        exchLabel="Nasdaq",
        owners="Founder holding@2005-01-01T00:00:00Z",
    )
    assert outcomes.wikidata_evidence([row], D) == []


def test_company_names_agree_on_a_shared_meaningful_word():
    assert sectypes.names_agree("META PLATFORMS INC-CLASS A", "Meta Platforms")
    assert sectypes.names_agree("LINKEDIN CORP - A", "LinkedIn Corporation")
    assert not sectypes.names_agree("PROSHARES S&P DYNAMIC BUFFER", "Meta Platforms")
    assert not sectypes.names_agree("GLOBAL X DEFENSE TECH ETF", "Sears Holdings Corporation")
    assert not sectypes.names_agree(
        "THE HOLDINGS CORP", "Holdings Inc"
    )  # only noise words in common


def test_undated_wikidata_rows_are_dropped_for_a_ticker_that_several_securities_used():
    undated = binding(
        item="http://www.wikidata.org/entity/Q8",
        itemLabel="Sears Holdings Corporation",
        exchLabel="Nasdaq",
    )
    assert outcomes.wikidata_matches([undated], date(2023, 9, 14), date(2024, 12, 31)) == [undated]
    assert (
        outcomes.wikidata_matches(
            [undated], date(2023, 9, 14), date(2024, 12, 31), ticker_shared=True
        )
        == []
    )


def test_wikidata_rows_are_matched_to_the_period_the_ticker_was_used():
    meta_platforms = binding(
        item="http://www.wikidata.org/entity/Q380",
        itemLabel="Meta Platforms",
        exchLabel="Nasdaq",
        tStart="2022-06-09T00:00:00Z",
    )
    # The ETF held META from 2021-06-30 to 2022-01-28: Meta Platforms' listing starts too late.
    assert outcomes.wikidata_matches([meta_platforms], date(2021, 6, 30), date(2022, 1, 28)) == []
    assert outcomes.wikidata_matches([meta_platforms], date(2022, 6, 9), date(2024, 12, 31)) == [
        meta_platforms
    ]


def test_a_cash_record_next_to_a_stock_record_of_the_same_deal_is_not_a_cash_value():
    result = resolve_outcome(
        [
            ev("alpaca", "acquired_cash", 5.65),
            ev("alpaca", "acquired_stock_and_cash"),
            ev("sec", "acquired"),
        ]
    )
    assert (result.outcome, result.status, result.assumed_value) == (
        "acquired_stock_and_cash",
        "confirmed",
        None,
    )


def test_cover_page_class_titles_give_the_listed_security_type():
    t = sectypes.cover_class_type
    assert t("Common Stock, without par value") == "common"
    assert t("Class A common stock, par value $0.0001 per share") == "common"
    assert t("Common Shares of Beneficial Interest") == "common"  # a REIT trust
    assert t("Class A ordinary shares") == "common"
    assert t("American depositary shares (each representing 20 Class A common shares") == "adr"
    assert t("Depositary Shares, each representing 1/1,000th of a share of Series A Preferred") == (
        "preferred"
    )
    assert t("Common Units Representing Limited Partner Interests") == "partnership"
    assert (
        t("Units, each consisting of one share of Class A common stock and one-half of one warrant")
        == "unit"
    )
    assert t("Redeemable warrants, each exercisable for one share of Class A common stock") == (
        "warrant"
    )
    assert t("1.431% Notes due 2024") == "debt"
    assert t("iPath S&P 500 VIX Short-Term Futures ETN") == "etn"
    assert t("SPDR Gold Shares") is None  # unclear: no vote
    assert t("") is None


def test_a_company_vote_fits_any_type_a_company_issues_but_not_a_fund():
    vote = sectypes.TypeVote
    # An ADR: Wikidata's "public company" is no contradiction (BABA, TSM, ASML).
    assert sectypes.resolve_type([vote("openfigi", "adr"), vote("wikidata", "company")]) == (
        "adr",
        "single_source",
    )
    # A REIT: OpenFIGI's REIT is common stock, Wikidata's REIT is a company (CCI, SPG).
    assert sectypes.resolve_type([vote("openfigi", "common"), vote("wikidata", "company")]) == (
        "common",
        "confirmed",
    )
    assert sectypes.resolve_type([vote("wikidata", "company")]) == ("common", "single_source")
    # A company is not a fund: that disagreement stays a conflict.
    assert sectypes.resolve_type([vote("openfigi", "etf"), vote("wikidata", "company")]) == (
        "unknown",
        "conflict",
    )


def test_only_the_issuers_own_sec_description_settles_a_disagreement():
    vote = sectypes.TypeVote
    # Today's OpenFIGI holder is an ETF, the issuer's cover pages then said common stock.
    assert sectypes.resolve_type(
        [vote("openfigi", "etf"), vote("sec_cover", "common"), vote("wikidata", "company")]
    ) == ("common", "resolved")
    # A Form 25 for another class (a preferred series) does not outvote the cover page.
    assert sectypes.resolve_type(
        [vote("sec_cover", "common"), vote("sec_form25", "preferred")]
    ) == (
        "common",
        "resolved",
    )
    # A hand-read SEC passage outranks the cover page.
    assert sectypes.resolve_type([vote("sec_cover", "common"), vote("sec_text", "fund")]) == (
        "fund",
        "resolved",
    )
    # The issuer's own descriptions disagree (an ADR that became an ordinary listing): conflict.
    assert sectypes.resolve_type([vote("sec_cover", "adr"), vote("sec_cover", "common")]) == (
        "unknown",
        "conflict",
    )
    # Without SEC text, OpenFIGI against the fifth letter is not settled by rank.
    assert sectypes.resolve_type(
        [vote("openfigi", "common"), vote("symbol_suffix", "warrant")]
    ) == (
        "unknown",
        "conflict",
    )
    assert "resolved" in sectypes.USABLE_STATUSES


def test_a_title_naming_several_classes_gives_no_vote():
    t = sectypes.cover_class_type
    # A cover table laid out by columns: titles first, then tickers (Walmart, Costco 2019).
    assert (
        t("Common Stock, par value $0.10 per share 1.900% Notes Due 2022 2.550% Notes Due 2026")
        is None
    )
    assert (
        t("Common Stock, par value $.01 per share Preferred Stock, par value $.01 per share")
        is None
    )
    # "United" is not "Units".
    assert t("United Continental Holdings, Inc. Common Stock, $0.01 par value") == "common"
