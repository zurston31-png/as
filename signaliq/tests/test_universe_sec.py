"""SEC cross-check: parsers, matching rules and how SEC evidence enters types and outcomes.

SYNTHETIC TEST FIXTURES ONLY: the feeds and notices below are shortened imitations
of EDGAR's formats with invented companies (TESTCO...), CIKs and dates. Nothing here
calls SEC.
"""

from datetime import date

import pytest

from app.database import create_session_factory, session_scope
from app.universe import evidence, outcomes, sec
from app.universe.database import open_research_database
from app.universe.models import ExitEvidence, ExternalResponse, Security, TickerHistory

# =============================================================================
# Fixtures
# =============================================================================


def entry(form, filed, href, items=""):
    items_tag = f"<items-desc>{items}</items-desc>" if items else ""
    return (
        f"<entry><content type='text/xml'><filing-date>{filed}</filing-date>"
        f"<filing-href>{href}</filing-href><filing-type>{form}</filing-type>{items_tag}"
        f"</content></entry>"
    )


def feed(cik, name, sic, entries):
    return (
        "<?xml version='1.0'?><feed><company-info>"
        f"<assigned-sic>{sic}</assigned-sic><assigned-sic-desc>TEST &amp; SIC</assigned-sic-desc>"
        f"<cik>{cik}</cik><conformed-name>{name}</conformed-name></company-info>"
        + "".join(entries)
        + "</feed>"
    )


def search_results(pairs):
    return (
        "<feed>"
        + "".join(
            f"<entry><title>{name}</title><link href='/cgi-bin/browse-edgar?action=getcompany&amp;"
            f"CIK={cik}'/></entry>"
            for name, cik in pairs
        )
        + "</feed>"
    )


def form25(cik, name, exchange, description, rule):
    return (
        "<edgarSubmission><formData><issuer><cik>"
        f"{cik}</cik><entityName>{name}</entityName></issuer><exchange><entityName>{exchange}"
        f"</entityName></exchange><descriptionClassSecurity>{description}</descriptionClassSecurity>"
        f"<ruleProvision>{rule}</ruleProvision><signatureDate>2022-05-02</signatureDate>"
        "</formData></edgarSubmission>"
    )


class FakeSec(sec.CachedSec):
    """A CachedSec that answers from a dict keyed like the real requests."""

    def __init__(self, companies, searches=None, notices=None, documents=None):
        self.companies = companies  # (cik or ticker, form) -> Company feed text
        self.searches = searches or {}
        self.notices = notices or {}  # filing href -> RemovalNotice
        self.documents = documents or {}
        self.asked: list = []

    def company(self, cik_or_ticker, before, form=""):
        self.asked.append((cik_or_ticker, before, form))
        text = self.companies.get((cik_or_ticker, form))
        return sec.parse_company_feed(text) if text else None

    def search(self, name):
        return self.searches.get(name, [])

    def removal_notice(self, filing):
        return self.notices.get(filing.href)

    def document_text(self, filing):
        return self.documents.get(filing.href, "")


# =============================================================================
# Parsers and classifications
# =============================================================================


def test_company_feed_parsing():
    text = feed(
        "0000123456",
        "TESTCO INC",
        "1311",
        [
            entry("25-NSE", "2022-05-02", "https://x/25.htm"),
            entry("8-K", "2022-04-20", "https://x/8k.htm", "items 1.03 and 9.01"),
        ],
    )
    company = sec.parse_company_feed(text)
    assert (company.cik, company.name, company.sic) == ("0000123456", "TESTCO INC", "1311")
    assert company.sic_description == "TEST & SIC"
    assert [(f.form, f.filed) for f in company.filings] == [
        ("25-NSE", date(2022, 5, 2)),
        ("8-K", date(2022, 4, 20)),
    ]
    assert company.filings[1].items == "items 1.03 and 9.01"
    assert sec.parse_company_feed("<feed>No matching company</feed>") is None


def test_company_search_lists_candidates_or_the_single_company_sec_jumped_to():
    assert sec.parse_company_search(search_results([("TESTCO A", "11"), ("TESTCO B", "22")])) == [
        ("TESTCO A", "0000000011"),
        ("TESTCO B", "0000000022"),
    ]
    jumped = feed("33", "TESTCO C", "2000", [])
    assert sec.parse_company_search(jumped) == [("TESTCO C", "0000000033")]


def test_form25_parsing():
    notice = sec.parse_form25(
        form25(
            "0000123456",
            "TESTCO INC",
            "Nasdaq Stock Market LLC",
            "Common Stock",
            "17 CFR 240.12d2-2(b)",
        ),
        "  Delisting   determination ",
    )
    assert notice.issuer_cik == "0000123456"
    assert notice.exchange == "Nasdaq Stock Market LLC"
    assert notice.class_description == "Common Stock"
    assert notice.rule == "17 CFR 240.12d2-2(b)"
    assert notice.signed == date(2022, 5, 2)
    assert notice.reason_text == "Delisting determination"
    assert sec.parse_form25("<edgarSubmission/>") is None


@pytest.mark.parametrize(
    ("description", "symbol", "expected"),
    [
        ("Common Stock", "TSTA", "common"),
        ("Common Stock, $0.01 par value per share", "TSTA", "common"),
        ("Ordinary Shares", "TSTA", "common"),
        ("Series C Common Stock", "TSTA", "common"),
        # A notice covering both the shares and the warrants: the ticker decides.
        ("Class A common stock and warrants", "TSTA", "common"),
        ("Class A common stock and warrants", "TSTAW", "warrant"),
        ("Class A Common Stock, Warrant, Unit", "TSTAU", "unit"),
        ("Class A Common Stock, Warrant, Unit", "TSTA", "common"),
        ("Warrants, each exercisable for one share", "TSTAW", "warrant"),
        ("Rights", "TSTAR", "right"),
        ("6.25% Senior Notes due 2027", "TSTA", "debt"),
        (
            "Depositary Shares, each representing 1/1000th of a share of Series A Preferred",
            "TSTA",
            "preferred",
        ),
        ("American Depositary Shares", "TSTA", "adr"),
        ("Shares of beneficial interest of the TEST Exchange-Traded Fund", "TSTA", "etf"),
        ("Warrants and Rights", "TSTA", None),  # several types, no common class: unclear
        ("", "TSTA", None),
    ],
)
def test_class_type(description, symbol, expected):
    assert sec.class_type(description, symbol) == expected


def test_notice_type_across_several_notices_of_one_issuer():
    assert sec.notice_type(["Series A Common Stock", "Series B Common Stock"], "TSTA") == "common"
    # Time Warner style: the common stock and the notes delisted together.
    assert sec.notice_type(["Common Stock", "7.70% Debentures due 2032"], "TSTA") == "common"
    assert sec.notice_type(["Common Stock", "Warrants"], "TSTAW") == "warrant"
    assert sec.notice_type(["Warrants", "Rights"], "TSTA") is None
    assert sec.notice_type([], "TSTA") is None


@pytest.mark.parametrize(
    ("rule", "category"),
    [
        ("17 CFR 240.12d2-2(a)(3)", "acquired"),
        ("17 CFR 240.12d2-2(a)(1)", "redeemed"),
        ("17 CFR 240.12d2-2(a)(2)", "redeemed"),
        ("17 CFR 240.12d2-2(a)(4)", "delisted_no_reason"),
        ("17 CFR 240.12d2-2(b)", "delisted_no_reason"),
        ("17 CFR 240.12d2-2(c)", "delisted_no_reason"),
        ("", "delisted_no_reason"),
    ],
)
def test_rule_meaning(rule, category):
    assert sec.rule_meaning(rule)[0] == category


def test_eight_k_items():
    assert sec.eight_k_category("items 1.01, 1.03, 2.04 and 9.01") == "bankruptcy"
    assert sec.eight_k_category("items 2.01, 3.03, 5.01 and 9.01") == "acquired"
    assert sec.eight_k_category("items 2.01 and 9.01") is None  # it bought something
    assert sec.eight_k_category("items 3.01 and 9.01") is None  # a deficiency notice only


def test_cash_terms_only_for_cash_only_consideration():
    cash = (
        "At the effective time, each share of common stock of TESTCO was converted into the "
        "right to receive $41.25 in cash, without interest. Other text."
    )
    assert sec.cash_terms(cash)[0] == 41.25
    mixed = (
        "Each share of common stock was converted into the right to receive $50.00 in cash and "
        "one share of common stock of TESTPARENT."
    )
    assert sec.cash_terms(mixed)[0] is None
    per_share = "Holders have the right to receive $12.10 per share in cash, without interest."
    assert sec.cash_terms(per_share)[0] == 12.10
    exchange_ratio = (
        "Each share was converted into the right to receive $5.00 in cash and 0.5 shares of "
        "TESTPARENT common stock."
    )
    assert sec.cash_terms(exchange_ratio)[0] is None
    assert sec.cash_terms("no terms here") == (None, "")


def test_the_cache_key_of_a_long_request_is_stable_between_runs(tmp_path):
    engine = open_research_database(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    factory = create_session_factory(engine)
    long_url = "https://www.sec.gov/Archives/edgar/data/" + "1" * 200
    with session_scope(factory) as session:
        session.add(
            ExternalResponse(
                source="sec",
                request_key=long_url[:96]
                + "#"
                + __import__("hashlib").sha256(long_url.encode()).hexdigest()[:24],
                response="cached text",
            )
        )
    with session_scope(factory) as session:
        client = sec.CachedSec(session, network=False)
        assert client.get(long_url) == "cached text"
        assert client.get("https://www.sec.gov/not-cached") is None  # cache only: no request
        assert client.requests == 0
    engine.dispose()


def test_fetched_answers_survive_a_run_that_is_stopped(tmp_path):
    """Answers are committed every COMMIT_EVERY requests, so a stopped run keeps them."""
    import httpx

    engine = open_research_database(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    factory = create_session_factory(engine)
    session = factory()
    client = sec.CachedSec(session, network=True, sleep=lambda _s: None)
    client._http = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=str(request.url)))
    )
    urls = [f"https://www.sec.gov/test/{i}" for i in range(sec.COMMIT_EVERY + 3)]
    for url in urls:
        assert client.get(url) == url
    session.rollback()  # the run is stopped before its session commits
    session.close()
    with session_scope(factory) as session:
        kept = {r.request_key for r in session.query(ExternalResponse)}
    assert kept == set(urls[: sec.COMMIT_EVERY])  # the last few, not yet committed, are lost
    engine.dispose()


def test_a_dropped_connection_is_retried_and_never_read_as_an_empty_answer(tmp_path):
    import httpx

    engine = open_research_database(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    factory = create_session_factory(engine)
    calls = []

    def flaky(request):
        calls.append(request.url)
        if len(calls) == 1:
            raise httpx.ConnectError("Connection reset by peer")
        if len(calls) == 2:
            return httpx.Response(503, text="File Unavailable")  # SEC's passing 503
        return httpx.Response(200, text="answer")

    def down(request):
        return httpx.Response(503, text="File Unavailable")

    with session_scope(factory) as session:
        client = sec.CachedSec(session, network=True, sleep=lambda _s: None)
        client._http = httpx.Client(transport=httpx.MockTransport(flaky))
        assert client.get("https://www.sec.gov/test/a") == "answer"
        assert len(calls) == 3
        client._http = httpx.Client(transport=httpx.MockTransport(down))
        with pytest.raises(sec.SecUnavailable):
            client.get("https://www.sec.gov/test/b")
        assert client.failures == ["https://www.sec.gov/test/b"]
    with session_scope(factory) as session:
        keys = {r.request_key for r in session.query(ExternalResponse)}
    assert keys == {"https://www.sec.gov/test/a"}  # nothing stored for the unanswered request
    engine.dispose()


# =============================================================================
# Matching a security to an SEC company
# =============================================================================

LAST = date(2022, 4, 29)


def test_a_company_is_accepted_only_with_a_timely_delisting_notice():
    companies = {
        ("TSTA", ""): feed("1", "TESTCO TODAY INC", "7372", []),  # today's holder of the ticker
        ("0000000001", "25"): feed("1", "TESTCO TODAY INC", "7372", []),
        ("0000000002", "25"): feed(
            "2", "TESTCO OLD INC", "1311", [entry("25-NSE", "2022-05-02", "h25")]
        ),
        ("0000000002", "8-K"): feed(
            "2",
            "TESTCO OLD INC",
            "1311",
            [
                entry("8-K", "2022-04-20", "h8k", "items 1.03 and 9.01"),
                entry("8-K", "2021-01-05", "hold", "items 1.03"),  # more than a year before
            ],
        ),
    }
    notice = sec.parse_form25(
        form25("2", "TESTCO OLD INC", "NYSE", "Common Stock", "17 CFR 240.12d2-2(b)")
    )
    fake = FakeSec(
        companies,
        searches={"testco old": [("TESTCO OLD INC", "0000000002")]},
        notices={"h25": notice},
    )
    found = sec.find(fake, ["TSTA"], ["testco old"], LAST)
    assert found.cik == "0000000002"
    assert found.candidates_checked == ["0000000001", "0000000002"]
    assert [n.class_description for _, n in found.notices] == ["Common Stock"]
    assert [k.href for k in found.eight_ks] == ["h8k"]


def test_a_notice_outside_the_window_or_in_2025_is_not_a_match():
    for filed in ("2022-03-01", "2022-08-15"):  # 59 days before, 108 days after
        fake = FakeSec(
            {
                ("TSTA", ""): feed("3", "T", "1", []),
                ("0000000003", "25"): feed("3", "T", "1", [entry("25", filed, "h")]),
            }
        )
        assert sec.find(fake, ["TSTA"], [], LAST).cik is None
    late = date(2024, 12, 20)
    fake = FakeSec(
        {
            ("TSTA", ""): feed("4", "T", "1", []),
            ("0000000004", "25"): feed("4", "T", "1", [entry("25-NSE", "2025-01-03", "h")]),
        }
    )
    found = sec.find(fake, ["TSTA"], [], late)
    assert found.cik is None
    assert all(before < date(2025, 1, 1) for _, before, _ in fake.asked)


# =============================================================================
# SEC evidence in outcomes and types
# =============================================================================


def finding(notices, eight_ks=(), cash=(None, ""), sic="1311"):
    return sec.SecFinding(
        cik="0000000009",
        company="TESTCO INC",
        sic=sic,
        notices=[
            (
                sec.Filing(form, filed, "", f"h{i}"),
                sec.RemovalNotice("9", "TESTCO INC", "NYSE", desc, rule, None),
            )
            for i, (form, filed, desc, rule) in enumerate(notices)
        ],
        eight_ks=[
            sec.Filing("8-K", filed, items, f"k{i}") for i, (filed, items) in enumerate(eight_ks)
        ],
        cash=cash,
    )


def test_bankruptcy_8k_with_an_exchange_delisting_is_a_bankruptcy_from_one_source():
    found = evidence.sec_evidence(
        finding(
            [("25-NSE", date(2022, 5, 2), "Common Stock", "17 CFR 240.12d2-2(b)")],
            [(date(2022, 4, 20), "items 1.03")],
        ),
        "TSTA",
    )
    assert sorted(e.category for e in found) == ["bankruptcy", "delisted_no_reason"]
    result = outcomes.resolve_outcome(found)
    assert (result.outcome, result.status, result.assumed_value) == (
        "bankruptcy",
        "single_source",
        0.0,
    )


def test_an_exchange_delisting_alone_stays_unresolved():
    found = evidence.sec_evidence(
        finding([("25-NSE", date(2022, 5, 2), "Ordinary Shares", "17 CFR 240.12d2-2(b)")]), "TSTA"
    )
    result = outcomes.resolve_outcome(found)
    assert (result.outcome, result.status, result.assumed_value) == ("unknown", "unresolved", None)


def test_a_redemption_has_no_assumed_value():
    found = evidence.sec_evidence(
        finding(
            [
                (
                    "25-NSE",
                    date(2022, 5, 2),
                    "Class A Common Stock, Warrant, Unit",
                    "17 CFR 240.12d2-2(a)(1)",
                )
            ],
            sic="6770",
        ),
        "TSTA",
    )
    result = outcomes.resolve_outcome(found)
    assert (result.outcome, result.status, result.assumed_value) == (
        "redeemed",
        "single_source",
        None,
    )


def test_redemption_and_bankruptcy_evidence_together_is_a_conflict():
    result = outcomes.resolve_outcome(
        [
            outcomes.Evidence("sec", "redeemed", None, "x"),
            outcomes.Evidence("wikidata", "bankruptcy", None, "y"),
        ]
    )
    assert (result.outcome, result.assumed_value) == ("conflict", None)


def test_a_merger_notice_and_8k_confirm_an_alpaca_merger_and_cash_terms_need_the_8k_text():
    found = evidence.sec_evidence(
        finding(
            [("25-NSE", date(2022, 5, 2), "Common Stock", "17 CFR 240.12d2-2(a)(3)")],
            [(date(2022, 5, 2), "items 2.01, 3.03 and 5.01")],
            cash=(41.25, "the right to receive $41.25 in cash"),
        ),
        "TSTA",
    )
    assert sorted(e.category for e in found) == ["acquired", "acquired_cash"]
    alpaca = outcomes.Evidence(
        "alpaca", "acquired_cash", date(2022, 5, 2), "cash_merger", None, 41.25
    )
    result = outcomes.resolve_outcome([*found, alpaca])
    assert (result.outcome, result.status, result.assumed_value) == (
        "acquired_cash",
        "confirmed",
        41.25,
    )


def test_a_notice_for_another_class_of_the_issuer_is_not_used():
    found = evidence.sec_evidence(
        finding(
            [
                ("25-NSE", date(2022, 5, 2), "Common Stock", "17 CFR 240.12d2-2(a)(3)"),
                ("25", date(2022, 5, 9), "7.70% Debentures due 2032", "17 CFR 240.12d2-2(a)(1)"),
            ]
        ),
        "TSTA",
    )
    assert [e.category for e in found] == [
        "acquired"
    ]  # the redeemed debentures are not this security


@pytest.fixture
def research(tmp_path, monkeypatch):
    engine = open_research_database(f"sqlite:///{(tmp_path / 'research.db').as_posix()}")
    factory = create_session_factory(engine)
    yield factory
    engine.dispose()


def add_ended(session, sid, symbol, first, last):
    session.add(
        Security(
            security_id=sid,
            first_session=first,
            last_trade_session=last,
            last_symbol=symbol,
            status="ended",
            exit_type="unexplained",
            declared_type="unknown",
            real_trade_sessions=200,
        )
    )
    session.add(
        TickerHistory(
            security_id=sid,
            symbol=symbol,
            valid_from=first,
            valid_to=last,
            linked_by="first listing",
        )
    )


def test_sec_type_vote_and_outcome_reach_the_security(research, monkeypatch):
    with session_scope(research) as session:
        add_ended(session, 1, "TSTA", date(2021, 1, 4), LAST)
        add_ended(session, 2, "TSTBW", date(2021, 1, 4), LAST)
    sec_found = {
        1: finding(
            [("25-NSE", date(2022, 5, 2), "Common Stock", "17 CFR 240.12d2-2(b)")],
            [(date(2022, 4, 20), "items 1.03")],
        ),
        2: finding(
            [
                (
                    "25-NSE",
                    date(2022, 5, 2),
                    "Class A common stock and warrants",
                    "17 CFR 240.12d2-2(b)",
                )
            ]
        ),
    }
    monkeypatch.setattr(evidence, "cached_sec_findings", lambda session: sec_found)
    with session_scope(research) as session:
        evidence.resolve_types(session)
        evidence.resolve_outcomes(session, [])
    with session_scope(research) as session:
        one, two = session.get(Security, 1), session.get(Security, 2)
        assert (one.security_type, one.type_status) == ("common", "single_source")
        assert (one.outcome, one.outcome_status, one.assumed_exit_value) == (
            "bankruptcy",
            "single_source",
            0.0,
        )
        # The suffix and the notice agree the warrant ticker is a warrant: two sources.
        assert (two.security_type, two.type_status) == ("warrant", "confirmed")
        assert two.outcome_status == "unresolved"
        sources = {e.source for e in session.query(ExitEvidence).filter_by(security_id=1)}
        assert sources == {"sec"}


def test_a_spac_liquidation_recorded_as_cash_merger_and_redemption_is_one_cash_exit():
    result = outcomes.resolve_outcome(
        [
            outcomes.Evidence("alpaca", "acquired_cash", None, "cash_merger", None, 10.68),
            outcomes.Evidence("sec", "redeemed", None, "25-NSE (a)(1)"),
        ]
    )
    assert (result.outcome, result.status, result.assumed_value) == ("redeemed", "confirmed", 10.68)
    # A redemption with a stock merger is still a conflict.
    result = outcomes.resolve_outcome(
        [
            outcomes.Evidence("alpaca", "acquired_stock", None, "stock_merger"),
            outcomes.Evidence("sec", "redeemed", None, "25-NSE (a)(1)"),
        ]
    )
    assert result.outcome == "conflict"


def test_a_funds_common_shares_do_not_count_as_common_stock(research, monkeypatch):
    with session_scope(research) as session:
        add_ended(session, 1, "TSTF", date(2021, 1, 4), LAST)
    fund = finding([("25-NSE", date(2022, 5, 2), "Common Shares", "17 CFR 240.12d2-2(a)(3)")])
    fund.company = "TEST MLP INCOME OPPORTUNITIES FUND"
    monkeypatch.setattr(evidence, "cached_sec_findings", lambda session: {1: fund})
    with session_scope(research) as session:
        evidence.resolve_types(session)
    with session_scope(research) as session:
        assert session.get(Security, 1).type_status == "unknown"
        assert sec.looks_like_fund("Goldman Sachs MLP Income Opportunities Fund")
        assert not sec.looks_like_fund("Fundamental Testing Inc")


def test_search_variants_match_edgar_conformed_names():
    assert sec.search_variants("LINKEDIN CORP - A") == ["LINKEDIN CORP - A", "linkedin corp"]
    assert sec.search_variants("LinkedIn Corporation") == ["LinkedIn Corporation", "linkedin corp"]
    assert sec.search_variants("Express, Inc.") == ["Express, Inc.", "express inc"]
    assert sec.search_variants("monsanto co /new/") == ["monsanto co /new/"]


# =============================================================================
# Filing lists from SEC's JSON service (company_from_submissions)
# =============================================================================


def submissions_block(rows):
    """A submissions block from (form, date, accession, items) rows. Synthetic only."""
    return {
        "form": [r[0] for r in rows],
        "filingDate": [r[1] for r in rows],
        "accessionNumber": [r[2] for r in rows],
        "items": [r[3] for r in rows],
    }


def test_json_list_matches_browse_edgar_rules_prefix_cutoff_order_and_links():
    data = {
        "cik": "123",
        "name": "TESTCO INC",
        "sic": "7372",
        "sicDescription": "Services-Prepackaged Software",
        "filings": {
            "recent": submissions_block(
                [
                    ("8-K", "2022-06-01", "0000000123-22-000003", "1.01,9.01"),  # after the cut-off
                    ("25-NSE", "2022-05-02", "0000000123-22-000002", ""),
                    ("8-K", "2022-05-02", "0000000123-22-000001", "2.01,3.03"),
                    ("25", "2022-04-01", "0000000123-22-000000", ""),
                ]
            )
        },
    }
    company = sec.company_from_submissions(data, (), date(2022, 5, 2), "25")
    assert company.cik == "0000000123"
    assert [(f.form, f.filed) for f in company.filings] == [
        ("25-NSE", date(2022, 5, 2)),  # the cut-off day is included
        ("25", date(2022, 4, 1)),
    ]
    assert company.filings[0].href == (
        "https://www.sec.gov/Archives/edgar/data/123/000000012322000002/0000000123-22-000002-index.htm"
    )
    every = sec.company_from_submissions(data, (), date(2022, 5, 2))
    assert [f.form for f in every.filings] == ["25-NSE", "8-K", "25"]  # newest first
    assert sec.eight_k_category(every.filings[1].items) == "acquired"


def test_json_list_reads_older_pages_and_keeps_only_the_newest_hundred():
    recent = [("4", "2023-01-02", f"0000000123-23-{i:06d}", "") for i in range(60)]
    older = [("4", "2015-03-02", f"0000000123-15-{i:06d}", "") for i in range(60)]
    data = {"cik": "123", "filings": {"recent": submissions_block(recent)}}
    company = sec.company_from_submissions(data, [submissions_block(older)], date(2024, 1, 1))
    assert len(company.filings) == 100
    assert sum(f.filed.year == 2015 for f in company.filings) == 40


def test_json_list_never_reads_a_missing_page_as_no_filings(tmp_path):
    import json as json_module

    import httpx

    engine = open_research_database(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    factory = create_session_factory(engine)
    main = {
        "cik": "123",
        "name": "TESTCO INC",
        "filings": {
            "recent": submissions_block([("8-K", "2023-01-02", "0000000123-23-000001", "")]),
            "files": [{"name": "CIK0000000123-submissions-001.json", "filingFrom": "2001-01-01"}],
        },
    }

    def answer(request):
        if request.url.path.endswith("CIK0000000123.json"):
            return httpx.Response(200, text=json_module.dumps(main))
        return httpx.Response(404, text="")

    with session_scope(factory) as session:
        client = sec.CachedSec(session, network=True, sleep=lambda _s: None)
        client._http = httpx.Client(transport=httpx.MockTransport(answer))
        with pytest.raises(sec.SecUnavailable):
            client.company_json("0000000123", date(2022, 1, 1), "25")
        assert client.company_json("0000000999", date(2022, 1, 1)) is None  # SEC: no such company
    engine.dispose()
