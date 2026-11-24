"""Point-in-time issuers: which company held a ticker on a day, and blank-check tests.

SYNTHETIC TEST FIXTURES ONLY: fake tickers (TSTX...), fake CIKs and invented filings.
Nothing here calls SEC; a fake client answers from prepared texts.
"""

import json
from datetime import date

import pytest

from app.database import create_session_factory, session_scope
from app.universe import evidence
from app.universe import sec_history as sh
from app.universe.database import open_research_database
from app.universe.evidence import IssuerPoint, issuer_on
from app.universe.models import IssuerEvidence, IssuerSic, Security, TickerHistory

COVER = (
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 10-Q Title of each class "
    "Trading Symbol(s) Name of each exchange on which registered Common Stock, $0.01 par "
    "value {ticker} New York Stock Exchange"
)


# =============================================================================
# Cover pages
# =============================================================================


def test_a_cover_page_lists_the_ticker_under_trading_symbol():
    assert sh.cover_page_symbol(COVER.format(ticker="TSTX"), "TSTX")
    # The ticker must stand alone after the heading: not inside a name or another word.
    assert sh.cover_page_symbol(COVER.format(ticker="TSTXY"), "TSTX") is None
    att = "Title of each class Trading Symbol(s) Common Shares AT&T T New York Stock Exchange"
    assert sh.cover_page_symbol(att, "T")
    assert sh.cover_page_symbol(att.replace(" T New", " New"), "T") is None
    # Written before the heading (the registrant's name, a press release) does not count.
    assert sh.cover_page_symbol("TSTX announced results. Trading Symbol(s) TSTY", "TSTX") is None
    # Too far after the heading to be in the table.
    far = "Trading Symbol(s) " + "x " * 400 + "TSTX New York Stock Exchange"
    assert sh.cover_page_symbol(far, "TSTX") is None
    # Named near the table but not as a listed security (a subsidiary's 10-Q naming its
    # parent): the ticker must be followed by its exchange.
    parent = (
        "Trading Symbol Name of exchange on which registered Common Stock, par value $0.001 "
        "TSTY The Nasdaq Stock Market LLC As of May 3, 2019, TSTX owned 80% of the shares"
    )
    assert sh.cover_page_symbol(parent, "TSTY")
    assert sh.cover_page_symbol(parent, "TSTX") is None
    # Zero-width spaces from the HTML table cells do not hide the row (JOYY's 20-F).
    zw = "Trading symbol(s) American depositary shares \u200b TSTX \u200b The Nasdaq Stock Market"
    assert sh.cover_page_symbol(zw, "TSTX")


# =============================================================================
# The issuer on a day
# =============================================================================


def point(day, cik, how="cover_page", symbol="TSTX"):
    return IssuerPoint(day, cik, f"Company {cik}", how, f"https://x/{cik}/{day}", symbol)


def test_the_issuer_on_a_day_is_the_company_of_the_latest_evidence_by_then():
    points = [
        point(date(2019, 8, 1), "A"),
        point(date(2020, 5, 1), "A"),
        point(date(2020, 8, 1), "B"),  # the ticker passed to another company (IAC, 2020)
        point(date(2024, 5, 1), "B"),
    ]
    assert issuer_on(points, date(2020, 3, 31)) == (points[0], False)
    assert issuer_on(points, date(2020, 6, 30))[0].cik == "A"
    assert issuer_on(points, date(2020, 8, 31))[0].cik == "B"
    assert issuer_on(points, date(2024, 11, 29))[0].cik == "B"
    # Before any evidence: the earliest, flagged as resting on a later document.
    assert issuer_on(points, date(2019, 6, 28)) == (points[0], True)
    assert issuer_on([], date(2020, 1, 31)) == (None, False)


def test_the_delisting_notice_wins_a_tie_on_the_same_day():
    day = date(2022, 6, 30)
    points = [point(day, "A"), point(day, "B", how="form25")]
    assert issuer_on(points, day)[0].cik == "B"


# =============================================================================
# Issuer evidence from a (fake) SEC
# =============================================================================


class FakeSec:
    """Answers the URLs and searches the issuer step asks, from prepared data."""

    def __init__(self, today, filings, documents):
        self.today = today  # ticker -> (cik, title): SEC's ticker list as of today
        self.filings = filings  # cik -> [(filed, accession, filename, ticker listed)]
        self.documents = documents  # url -> text
        self.asked = []

    def get(self, url, params=None):
        self.asked.append((url, params))
        if url == evidence.COMPANY_TICKERS_URL:
            return json.dumps(
                {
                    str(i): {"cik_str": int(cik), "ticker": t, "title": title}
                    for i, (t, (cik, title)) in enumerate(self.today.items())
                }
            )
        if url == sh.FTS_URL:
            ticker = params["q"].split('"')[1]
            if "Trading Symbol" not in params["q"]:
                return json.dumps({"hits": {"total": {"value": 0}, "hits": []}})
            first = date.fromisoformat(params["startdt"])
            last = date.fromisoformat(params["enddt"])
            hits = []
            for cik, rows in self.filings.items():
                if params.get("ciks") and int(params["ciks"]) != int(cik):
                    continue
                for filed, adsh, filename, listed in rows:
                    if listed == ticker and first <= filed <= last:
                        hits.append(
                            {
                                "_id": f"{adsh}:{filename}",
                                "_source": {
                                    "file_date": filed.isoformat(),
                                    "ciks": [cik],
                                    "display_names": [f"Company {cik}  (CIK {cik})"],
                                    "form": "10-Q",
                                    "sics": [],
                                },
                            }
                        )
            return json.dumps({"hits": {"total": {"value": len(hits)}, "hits": hits}})
        return self.documents.get(url)


def filing(cik, filed, ticker, n, cover=True):
    """A filing the search finds for ``ticker``; its cover page lists ``ticker`` when
    ``cover`` is True, another ticker when it is a string, nothing when False."""
    adsh = f"{cik}-{filed.year % 100:02d}-{n:06d}"
    filename = f"doc{n}.htm"
    url = f"{sh.ARCHIVES}/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/{filename}"
    if cover is True:
        text = COVER.format(ticker=ticker)
    elif cover:
        text = COVER.format(ticker=cover) + f" Our former parent {ticker} is mentioned here."
    else:
        text = f"Press release mentioning {ticker}."
    return (filed, adsh, filename, ticker), url, text


@pytest.fixture
def research(tmp_path, monkeypatch):
    engine = open_research_database(f"sqlite:///{(tmp_path / 'research.db').as_posix()}")
    factory = create_session_factory(engine)
    yield factory
    engine.dispose()


def add_security(session, sid, tickers, status="trading"):
    session.add(
        Security(
            security_id=sid,
            first_session=tickers[0][1],
            last_trade_session=tickers[-1][2],
            last_symbol=tickers[-1][0],
            status=status,
            exit_type="none",
            declared_type="unknown",
            real_trade_sessions=100,
        )
    )
    for index, (symbol, first, last) in enumerate(tickers):
        session.add(
            TickerHistory(
                security_id=sid,
                symbol=symbol,
                valid_from=first,
                valid_to=last,
                linked_by="first listing" if index == 0 else "rename",
            )
        )
    session.flush()


def build(rows):
    filings, documents = {}, {}
    for cik, filed, ticker, n, cover in rows:
        row, url, text = filing(cik, filed, ticker, n, cover)
        filings.setdefault(cik, []).append(row)
        documents[url] = text
    return filings, documents


def test_todays_ticker_list_proposes_a_company_but_never_decides(research):
    # Today the ticker belongs to a new holding company that filed nothing back then
    # (XOM, 2026); the company that listed it on its cover pages in 2019-2024 is found.
    filings, documents = build(
        [
            ("0000000111", date(2019, 8, 1), "TSTX", 1, True),
            ("0000000111", date(2022, 5, 1), "TSTX", 2, True),
            ("0000000111", date(2024, 11, 1), "TSTX", 3, True),
        ]
    )
    client = FakeSec({"TSTX": ("0000000999", "New Holdings")}, filings, documents)
    with session_scope(research) as session:
        add_security(session, 1, [("TSTX", date(2016, 1, 4), date(2024, 12, 31))])
        points, conflicts = evidence.issuer_points(session, client, {})
    assert {p.cik for p in points[1]} == {"0000000111"}
    assert [p.filed for p in points[1]] == [date(2019, 8, 1), date(2022, 5, 1), date(2024, 11, 1)]
    assert conflicts == {}
    # The company in today's list was asked about, and rejected for want of cover pages.
    assert any(p and p.get("ciks") == "0000000999" for _, p in client.asked)


def test_a_ticker_that_passed_to_another_company_has_one_issuer_per_period(research):
    filings, documents = build(
        [
            ("0000000111", date(2019, 8, 1), "TSTX", 1, True),
            ("0000000111", date(2020, 5, 1), "TSTX", 2, True),
            ("0000000222", date(2020, 8, 1), "TSTX", 3, True),
            ("0000000222", date(2024, 5, 1), "TSTX", 4, True),
        ]
    )
    client = FakeSec({"TSTX": ("0000000222", "New Co")}, filings, documents)
    with session_scope(research) as session:
        add_security(session, 1, [("TSTX", date(2016, 1, 4), date(2024, 12, 31))])
        points, _ = evidence.issuer_points(session, client, {})
    assert issuer_on(points[1], date(2020, 3, 31))[0].cik == "0000000111"
    assert issuer_on(points[1], date(2020, 9, 30))[0].cik == "0000000222"


def test_a_companys_evidence_ends_when_its_cover_pages_stop_listing_the_ticker(research):
    # IAC/InterActiveCorp's filings list IAC until mid-2020 and MTCH after (it became Match
    # Group); the search still finds the later ones, whose text names IAC.
    rows = [("0000000111", date(2019, 5, 8), "TSTX", 1, True)]
    rows += [("0000000111", date(2019, 8 + i, 1), "TSTX", 2 + i, True) for i in range(4)]
    rows += [("0000000111", date(2020, 2 + i, 1), "TSTX", 10 + i, True) for i in range(4)]
    rows += [("0000000111", date(2020, 8, 1 + i), "TSTX", 20 + i, "TSTZ") for i in range(9)]
    rows += [("0000000111", date(2021 + i, 3, 1), "TSTX", 40 + i, "TSTZ") for i in range(4)]
    rows += [("0000000222", date(2020, 7, 6), "TSTX", 60, True)]
    rows += [("0000000222", date(2023, 7, 6), "TSTX", 61, True)]
    filings, documents = build(rows)
    client = FakeSec({}, filings, documents)
    with session_scope(research) as session:
        add_security(session, 1, [("TSTX", date(2016, 1, 4), date(2024, 12, 31))])
        points, conflicts = evidence.issuer_points(session, client, {})
    first = [pt for pt in points[1] if pt.cik == "0000000111"]
    assert first[0].filed == date(2019, 5, 8)
    assert first[-1].filed == date(2020, 5, 1)
    assert issuer_on(points[1], date(2020, 7, 31))[0].cik == "0000000222"
    assert conflicts == {}


def test_a_search_hit_without_the_ticker_on_its_cover_page_is_not_evidence(research):
    # The search matched (the ticker and "Trading Symbol" somewhere in the filing), but no
    # document lists the ticker in the cover table: no evidence, never a guess.
    filings, documents = build(
        [
            ("0000000333", date(2020, 2, 1), "TSTX", 1, False),
            ("0000000333", date(2021, 2, 1), "TSTX", 2, False),
        ]
    )
    client = FakeSec({}, filings, documents)
    with session_scope(research) as session:
        add_security(session, 1, [("TSTX", date(2016, 1, 4), date(2024, 12, 31))])
        points, _ = evidence.issuer_points(session, client, {})
    assert 1 not in points


def test_a_spac_and_the_business_it_became_share_one_company_across_the_rename(research):
    # FEAC -> SKLZ: the SPAC's cover pages list FEAC, the same CIK's later ones list SKLZ.
    filings, documents = build(
        [
            ("0000000444", date(2020, 6, 1), "TSTA", 1, True),
            ("0000000444", date(2020, 11, 1), "TSTA", 2, True),
            ("0000000444", date(2021, 3, 1), "TSTB", 3, True),
            ("0000000444", date(2024, 3, 1), "TSTB", 4, True),
        ]
    )
    client = FakeSec({}, filings, documents)
    with session_scope(research) as session:
        add_security(
            session,
            1,
            [
                ("TSTA", date(2020, 4, 27), date(2020, 12, 16)),
                ("TSTB", date(2020, 12, 17), date(2024, 12, 31)),
            ],
        )
        points, _ = evidence.issuer_points(session, client, {})
    assert {(p.symbol, p.cik) for p in points[1]} == {
        ("TSTA", "0000000444"),
        ("TSTB", "0000000444"),
    }


# =============================================================================
# Blank-check exclusions, point in time
# =============================================================================


def store(session, sid, points, sics):
    for p in points:
        session.add(
            IssuerEvidence(
                security_id=sid,
                symbol=p.symbol,
                filed=p.filed,
                cik=p.cik,
                name=p.name,
                how=p.how,
                url=p.url,
            )
        )
    for cik, changes in sics.items():
        for as_of, sic in changes:
            session.add(IssuerSic(security_id=sid, cik=cik, as_of=as_of, sic=sic))
    session.flush()


DAYS = [date(2020, m, 28) for m in range(1, 13)] + [date(2021, m, 28) for m in range(1, 13)]


def test_the_blank_check_test_follows_the_issuer_of_the_day(research):
    # The ticker's first holder was a blank-check company (SIC 6770); from August 2020 the
    # ticker belongs to an operating company. Each day is judged by that day's holder.
    with session_scope(research) as session:
        add_security(session, 1, [("TSTX", date(2019, 1, 2), date(2021, 12, 31))])
        store(
            session,
            1,
            [point(date(2019, 6, 1), "A"), point(date(2020, 8, 1), "B")],
            {"A": [(date(2019, 1, 15), "6770")], "B": [(date(2019, 1, 15), "7372")]},
        )
        test = evidence.blank_check_exclusions(session)[1]
    assert test(date(2020, 6, 30)) == "blank_check_company"
    assert test(date(2020, 8, 31)) is None


def test_blank_check_answers_use_only_what_was_on_file_by_the_day(research):
    """Look-ahead: dropping every SIC filing and every issuer document dated after a day
    never changes that day's answer (when evidence by then exists)."""
    points = [point(date(2019, 6, 1), "A"), point(date(2021, 3, 1), "B")]
    sics = {
        "A": [(date(2019, 1, 15), "6770"), (date(2020, 9, 10), "3711")],
        "B": [(date(2019, 1, 15), "6770"), (date(2021, 6, 1), "7372")],
    }
    with session_scope(research) as session:
        add_security(session, 1, [("TSTX", date(2019, 1, 2), date(2021, 12, 31))])
        store(session, 1, points, sics)
        full = evidence.blank_check_exclusions(session)[1]
        answers = {day: full(day) for day in DAYS}
    assert answers[date(2020, 6, 28)] == "blank_check_company"
    assert answers[date(2020, 10, 28)] is None
    assert answers[date(2021, 4, 28)] == "blank_check_company"
    assert answers[date(2021, 7, 28)] is None
    for day in DAYS:
        with session_scope(research) as session:
            session.execute(IssuerSic.__table__.delete())
            session.execute(IssuerEvidence.__table__.delete())
            store(
                session,
                1,
                [p for p in points if p.filed <= day],
                {c: [x for x in v if x[0] <= day] for c, v in sics.items()},
            )
            cut = evidence.blank_check_exclusions(session)[1]
            assert cut(day) == answers[day], day
