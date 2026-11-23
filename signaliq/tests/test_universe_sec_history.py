"""SEC history: full-text search, submissions, point-in-time SIC, curated quotes, deals, SPACs.

SYNTHETIC TEST FIXTURES ONLY: invented companies (TESTCO...), CIKs, dates and documents
imitating EDGAR's formats. Nothing here calls SEC or Alpaca.
"""

import json
from datetime import date

from app.universe import outcomes, sectypes
from app.universe import sec_history as sh
from app.universe.curated import (
    CURATED_DEALS,
    CURATED_FACTS,
    CURATED_RENAMES,
    SecQuote,
)
from app.universe.evidence import missing_quotes
from app.universe.membership import DailyRow, MembershipRule, decide


def fts(*hits):
    return json.dumps(
        {
            "hits": {
                "hits": [
                    {
                        "_id": f"{adsh}:{name}",
                        "_source": {
                            "ciks": [cik],
                            "display_names": [f"{company}  (TCO)  (CIK {cik})"],
                            "form": form,
                            "file_date": filed,
                            "sics": [sic],
                            "file_type": file_type,
                        },
                    }
                    for cik, company, form, filed, adsh, name, sic, file_type in hits
                ]
            }
        }
    )


def test_full_text_hits_are_parsed_and_2025_filings_dropped():
    text = fts(
        (
            "0000000011",
            "TESTCO INC",
            "8-K",
            "2020-03-02",
            "0001-20-000001",
            "ex99.htm",
            "1311",
            "EX-99.1",
        ),
        (
            "0000000011",
            "TESTCO INC",
            "8-K",
            "2025-01-03",
            "0001-25-000001",
            "ex99.htm",
            "1311",
            "EX-99.1",
        ),
    )
    hits = sh.parse_full_text(text)
    assert len(hits) == 1
    hit = hits[0]
    assert (hit.cik, hit.name, hit.form, hit.filed, hit.sic) == (
        "0000000011",
        "TESTCO INC",
        "8-K",
        date(2020, 3, 2),
        "1311",
    )
    assert hit.url.endswith("/data/11/000120000001/ex99.htm")
    assert sh.parse_full_text("not json") == []


def test_submissions_parsing_keeps_former_names_and_drops_2025_filings():
    text = json.dumps(
        {
            "cik": "22",
            "name": "TESTCO OPERATING INC",
            "sic": "3211",
            "tickers": ["TCO"],
            "formerNames": [
                {
                    "name": "TEST ACQUISITION CORP",
                    "from": "2019-01-01T00:00:00",
                    "to": "2021-03-08T00:00:00",
                }
            ],
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "S-1"],
                    "filingDate": ["2025-02-01", "2021-03-09", "2019-02-01"],
                    "accessionNumber": ["a3", "a2", "a1"],
                }
            },
        }
    )
    company = sh.parse_submissions(text)
    assert company.cik == "0000000022"
    assert company.former_names[0].name == "TEST ACQUISITION CORP"
    assert company.former_names[0].end == date(2021, 3, 8)
    assert [f[2] for f in company.filings] == ["a2", "a1"]


def test_header_sic_and_sic_on_a_day_use_only_earlier_filings():
    assert sh.header_sic("<ASSIGNED-SIC>6770\n<IRS-NUMBER>1") == "6770"
    assert sh.header_sic("STANDARD INDUSTRIAL CLASSIFICATION:	BLANK CHECKS [6770]") == "6770"
    changes = [(date(2020, 1, 10), "6770"), (date(2021, 4, 2), "3211")]
    assert sh.sic_on(changes, date(2020, 1, 9)) is None  # nothing on file yet
    assert sh.sic_on(changes, date(2021, 4, 1)) == "6770"
    assert sh.sic_on(changes, date(2021, 4, 2)) == "3211"


class FakeClient:
    """Answers submissions and filing headers from dicts, and counts header reads."""

    def __init__(self, filings, sic_by_accession):
        self.filings, self.sic_by_accession, self.header_reads = filings, sic_by_accession, 0

    def get(self, url, params=None):
        if "submissions" in url:
            recent = {
                "form": ["8-K"] * len(self.filings),
                "filingDate": [d.isoformat() for d, _ in self.filings],
                "accessionNumber": [a for _, a in self.filings],
            }
            return json.dumps({"cik": "33", "name": "T", "filings": {"recent": recent}})
        if url.endswith(".hdr.sgml"):
            self.header_reads += 1
            accession = url.rsplit("/", 1)[1].removesuffix(".hdr.sgml")
            return f"<ASSIGNED-SIC>{self.sic_by_accession[accession]}"
        return None


def test_sic_history_finds_the_change_with_few_header_reads():
    filings = [
        (date(2019, 1, 1) + (date(2019, 2, 1) - date(2019, 1, 1)) * i, f"a{i:02}")
        for i in range(40)
    ]
    sic = {f"a{i:02}": ("6770" if i < 25 else "7372") for i in range(40)}
    client = FakeClient(filings, sic)
    points = sh.sic_history(client, "33", date(2019, 1, 15), date(2022, 12, 31))
    change = filings[25][0]
    assert sh.sic_on(points, change) == "7372"
    assert sh.sic_on(points, filings[24][0]) == "6770"
    assert client.header_reads <= 12  # a binary search, not one read per filing


def test_a_company_counts_as_writer_only_with_its_own_name_before_the_ticker():
    own = "Press release. TESTCO Holdings, Inc. (NASDAQ: TCOX) announced today that it"
    assert sh.own_ticker_mention(own, "TCOX", "TESTCO HOLDINGS INC")
    bio = "Mr. Doe advised Other Retail Corp. (formerly NASDAQ: TCOX) on its restructuring."
    assert sh.own_ticker_mention(bio, "TCOX", "RMG TEST ACQUISITION CORP") is None
    other = "Mr. Doe is a director of Other Retail Corp. (NASDAQ: TCOX)."
    assert sh.own_ticker_mention(other, "TCOX", "TESTCO HOLDINGS INC") is None


class QuoteClient:
    def __init__(self, documents):
        self.documents = documents

    def get(self, url, params=None):
        return self.documents.get(url)


def test_quotes_must_appear_word_for_word_after_markup_is_removed():
    doc = "<p>each share was converted into the right to receive</p>\n<b>0.6300</b> (the &ldquo;Ratio&rdquo;)"
    client = QuoteClient({"u": doc})
    assert (
        missing_quotes(client, [SecQuote("u", "converted into the right to receive 0.6300 (the")])
        == []
    )
    wrong = SecQuote("u", "converted into the right to receive 0.6400 (the")
    assert missing_quotes(client, [wrong]) == [wrong]
    gone = SecQuote("not cached", "anything")
    assert missing_quotes(client, [gone]) == [gone]


def test_every_curated_entry_cites_sec_documents_and_exact_passages():
    for entry in (*CURATED_RENAMES, *CURATED_FACTS, *CURATED_DEALS):
        assert entry.quotes, entry
        for quote in entry.quotes:
            assert quote.url.startswith("https://www.sec.gov/Archives/edgar/data/")
            assert len(quote.text) >= 15
    for deal in CURATED_DEALS:
        assert deal.effective_date >= deal.last_trade
        assert deal.effective_date < date(2025, 1, 1)
        assert deal.shares_per_share > 0
        assert deal.cash_per_share >= 0


def test_verified_deal_terms_value_a_stock_acquisition():
    found = [
        outcomes.Evidence("alpaca", "acquired_stock", date(2021, 8, 26), "stock_merger"),
        outcomes.Evidence("sec", "acquired", date(2021, 8, 26), "25-NSE (a)(3)"),
        outcomes.Evidence(
            "sec", "deal_terms", date(2021, 8, 26), "0.63 x ADI", None, 120.5, "acquired_stock"
        ),
    ]
    result = outcomes.resolve_outcome(found)
    assert (result.outcome, result.status, result.assumed_value) == (
        "acquired_stock",
        "confirmed",
        120.5,
    )


def test_a_cash_and_stock_deal_takes_the_terms_value_not_the_cash_part():
    found = [
        outcomes.Evidence("alpaca", "acquired_cash", None, "cash", None, 5.65),
        outcomes.Evidence("alpaca", "acquired_stock_and_cash", None, "stock and cash"),
        outcomes.Evidence(
            "sec", "deal_terms", None, "5.65 + 0.2316 x NE", None, 14.2, "acquired_stock_and_cash"
        ),
    ]
    result = outcomes.resolve_outcome(found)
    assert (result.outcome, result.assumed_value) == ("acquired_stock_and_cash", 14.2)


def test_a_redemption_with_a_stated_price_takes_it():
    found = [outcomes.Evidence("sec", "redeemed", None, "warrants called", None, 0.01)]
    result = outcomes.resolve_outcome(found)
    assert (result.outcome, result.assumed_value) == ("redeemed", 0.01)


def test_a_trading_gap_is_not_an_exit_but_exit_evidence_wins():
    gap = outcomes.Evidence("alpaca_bars", "trading_gap", None, "trades again")
    assert outcomes.resolve_outcome([gap]).status == "not_an_exit"
    fail = outcomes.Evidence("sec", "bankruptcy", None, "8-K 1.03")
    assert outcomes.resolve_outcome([gap, fail]).outcome == "bankruptcy"


def test_sec_sources_count_as_one_family():
    votes = [sectypes.TypeVote("sec_form25", "common"), sectypes.TypeVote("sec_text", "common")]
    assert sectypes.resolve_type(votes) == ("common", "single_source")
    votes.append(sectypes.TypeVote("openfigi", "common"))
    assert sectypes.resolve_type(votes) == ("common", "confirmed")


def test_a_blank_check_company_is_excluded_only_while_it_is_one():
    days = [date(2021, 1, 4) + (date(2021, 1, 5) - date(2021, 1, 4)) * i for i in range(400)]
    days = [d for d in days if d.weekday() < 5]
    history = {
        1: [DailyRow(d, 20.0, 1_000_000) for d in days],
        2: [DailyRow(d, 20.0, 10) for d in days],
    }
    types = {1: "common", 2: "common"}
    rule = MembershipRule(size=1, min_history=10, window=20, min_active_sessions=10)
    changes = [(date(2020, 6, 1), "6770"), (date(2021, 6, 1), "3211")]
    excluded = {1: lambda day: "blank_check_company" if sh.sic_on(changes, day) == "6770" else None}
    before = decide(
        date(2021, 3, 31),
        history,
        types,
        rule,
        [d for d in days if d <= date(2021, 3, 31)],
        None,
        excluded,
    )
    after = decide(
        date(2021, 7, 30),
        history,
        types,
        rule,
        [d for d in days if d <= date(2021, 7, 30)],
        None,
        excluded,
    )
    reason = {d.security_id: d.reason for d in before}
    assert reason[1] == "blank_check_company"
    assert [d.security_id for d in before if d.member] == [2]
    assert [d.security_id for d in after if d.member] == [1]


def test_a_confirmed_failure_keeps_the_zero_value_labelled_as_an_assumption():
    found = [
        outcomes.Evidence("sec", "failure", date(2023, 5, 1), "FDIC appointed receiver (8-K)"),
        outcomes.Evidence("sec", "failure", date(2023, 5, 1), "FDIC receiver (second 8-K)"),
    ]
    result = outcomes.resolve_outcome(found)
    assert (result.outcome, result.assumed_value, result.value_basis) == (
        "failure",
        0.0,
        "assumed_zero",
    )
    assert "not confirmed" in result.note
    assert "modelling assumption" in result.note
    paid = outcomes.resolve_outcome(
        [outcomes.Evidence("alpaca", "acquired_cash", None, "cash", None, 95.0)]
    )
    assert paid.value_basis == "stated_cash"
    terms = outcomes.resolve_outcome(
        [
            outcomes.Evidence("alpaca", "acquired_stock", None, "stock"),
            outcomes.Evidence(
                "sec", "deal_terms", None, "0.63 x ADI", None, 120.5, "acquired_stock"
            ),
        ]
    )
    assert terms.value_basis == "deal_terms"
    assert outcomes.resolve_outcome([]).value_basis == "none"


def test_sic_history_tolerates_headers_without_a_sic_on_the_same_day():
    filings = [(date(2020, 1, 2), "b0"), (date(2020, 1, 2), "b1"), (date(2020, 6, 1), "b2")]
    client = FakeClient(filings, {"b0": "", "b1": "6770", "b2": "6770"})
    client.get_orig = client.get

    def get(url, params=None):
        text = client.get_orig(url, params)
        return "no sic here" if text == "<ASSIGNED-SIC>" else text

    client.get = get
    points = sh.sic_history(client, "33", date(2020, 1, 1), date(2020, 12, 31))
    assert len({day for day, _ in points}) == len(points)
    assert sh.sic_on(points, date(2020, 7, 1)) == "6770"
