"""Evidence pipeline against a throw-away research database, with a fake network.

SYNTHETIC TEST FIXTURES ONLY: fake tickers (TESTA...), invented prices and
invented source answers. Nothing here calls OpenFIGI, Wikidata or Alpaca.
"""

from datetime import date
from typing import ClassVar

import pytest

from app.data.validation import trading_days
from app.database import create_session_factory, session_scope
from app.universe import evidence
from app.universe.database import open_research_database
from app.universe.models import ExitEvidence, Security, SecurityBar, TickerHistory, TypeEvidence

DAYS = trading_days(date(2022, 1, 3), date(2022, 12, 30))


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeHttp:
    """Stands in for httpx.Client inside app.universe.evidence."""

    calls: ClassVar[list] = []
    figi: ClassVar[dict] = {}
    wiki: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):
        FakeHttp.calls.append(("openfigi", [job["idValue"] for job in json]))
        return FakeResponse(200, [{"data": FakeHttp.figi.get(job["idValue"], [])} for job in json])

    def get(self, url, params):
        FakeHttp.calls.append(("wikidata", params["query"].count('"') // 2))
        rows = [row for rows in FakeHttp.wiki.values() for row in rows]
        return FakeResponse(200, {"results": {"bindings": rows}})


@pytest.fixture
def research(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence.httpx, "Client", FakeHttp)
    FakeHttp.calls, FakeHttp.figi, FakeHttp.wiki = [], {}, {}
    engine = open_research_database(f"sqlite:///{(tmp_path / 'research.db').as_posix()}")
    factory = create_session_factory(engine)
    yield factory
    engine.dispose()


def add_security(session, sid, symbol, first, last, *, status="ended", sessions=None, price=10.0):
    days = [d for d in DAYS if first <= d <= last]
    session.add(
        Security(
            security_id=sid,
            first_session=first,
            last_trade_session=last,
            last_symbol=symbol,
            status=status,
            exit_type="unexplained",
            declared_type="unknown",
            real_trade_sessions=sessions if sessions is not None else len(days),
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
    for d in days:
        session.add(
            SecurityBar(
                security_id=sid,
                session=d,
                symbol=symbol,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1000,
                vwap=price,
                trade_count=5,
                split_factor=1.0,
            )
        )
    session.flush()


def wiki_row(ticker, label, exchange="Nasdaq", **extra):
    row = {
        "ticker": ticker,
        "item": f"http://www.wikidata.org/entity/Q{abs(hash(label)) % 1000}",
        "itemLabel": label,
        "exchLabel": exchange,
        **extra,
    }
    return {key: {"value": value} for key, value in row.items()}


def test_openfigi_answers_are_cached_and_reused(research):
    FakeHttp.figi = {
        "TESTA": [
            {"name": "TEST A INC", "securityType": "Common Stock", "securityType2": "Common Stock"}
        ]
    }
    with session_scope(research) as session:
        first = evidence.fetch_openfigi(session, ["TESTA", "TESTB"], sleep=lambda s: None)
    with session_scope(research) as session:
        again = evidence.fetch_openfigi(session, ["TESTA", "TESTB"], sleep=lambda s: None)
        forced = evidence.fetch_openfigi(session, ["TESTA"], refresh=True, sleep=lambda s: None)

    assert (first, again, forced) == (1, 0, 1)
    assert FakeHttp.calls == [("openfigi", ["TESTA", "TESTB"]), ("openfigi", ["TESTA"])]


def test_openfigi_type_goes_only_to_the_latest_holder_of_a_reused_ticker(research):
    # TESTA was used by security 1 (ended mid-year) and later by security 2.
    # OpenFIGI describes the latest holder.
    FakeHttp.figi = {
        "TESTA": [{"name": "LATER FUND", "securityType": "ETP", "securityType2": "Mutual Fund"}]
    }
    with session_scope(research) as session:
        add_security(session, 1, "TESTA", DAYS[0], DAYS[100])
        add_security(session, 2, "TESTA", DAYS[150], DAYS[-1], status="trading")
        evidence.fetch_openfigi(session, ["TESTA"], sleep=lambda s: None)
        evidence.resolve_types(session)
    with session_scope(research) as session:
        earlier, later = session.get(Security, 1), session.get(Security, 2)
        assert (later.security_type, later.type_status) == ("etf", "single_source")
        assert (earlier.security_type, earlier.type_status) == ("unknown", "unknown")
        assert session.query(TypeEvidence).filter_by(security_id=1, source="openfigi").count() == 0


def test_two_sources_confirm_a_common_stock_and_a_suffix_conflict_is_flagged(research):
    FakeHttp.figi = {
        "TESTA": [
            {"name": "TEST A INC", "securityType": "Common Stock", "securityType2": "Common Stock"}
        ],
        "TESTW": [
            {"name": "TEST W INC", "securityType": "Common Stock", "securityType2": "Common Stock"}
        ],
    }
    FakeHttp.wiki = {"TESTA": [wiki_row("TESTA", "Test A", types="public company")]}
    with session_scope(research) as session:
        add_security(session, 1, "TESTA", DAYS[0], DAYS[-1], status="trading")
        add_security(session, 2, "TESTW", DAYS[0], DAYS[-1], status="trading")
        evidence.fetch_openfigi(session, ["TESTA", "TESTW"], sleep=lambda s: None)
        evidence.fetch_wikidata(session, ["TESTA", "TESTW"], sleep=lambda s: None)
        evidence.resolve_types(session)
    with session_scope(research) as session:
        assert (session.get(Security, 1).security_type, session.get(Security, 1).type_status) == (
            "common",
            "confirmed",
        )
        # Five letters ending in W reads as a warrant, OpenFIGI says common:
        # a conflict, not a guess.
        assert session.get(Security, 2).type_status == "conflict"


def test_outcomes_use_evidence_and_leave_unknown_exits_unresolved(research):
    FakeHttp.wiki = {
        "TESTA": [
            wiki_row(
                "TESTA",
                "Test A Bank",
                events=f"collapse of Test A Bank@{DAYS[101].isoformat()}T00:00:00Z",
            )
        ],
        "TESTB": [
            wiki_row("TESTB", "Test B", owners=f"Big Buyer@{DAYS[60].isoformat()}T00:00:00Z")
        ],
    }
    with session_scope(research) as session:
        add_security(session, 1, "TESTA", DAYS[0], DAYS[100])
        add_security(session, 2, "TESTB", DAYS[0], DAYS[90])
        add_security(session, 3, "TESTC", DAYS[0], DAYS[80])  # no evidence at all
        evidence.fetch_wikidata(session, ["TESTA", "TESTB", "TESTC"], sleep=lambda s: None)
        evidence.resolve_outcomes(session, DAYS)
    with session_scope(research) as session:
        failed, bought, unknown = (session.get(Security, i) for i in (1, 2, 3))
        assert (failed.outcome, failed.outcome_status, failed.assumed_exit_value) == (
            "failure",
            "single_source",
            0.0,
        )
        assert (bought.outcome, bought.assumed_exit_value) == ("acquired", None)
        assert (unknown.outcome, unknown.outcome_status, unknown.assumed_exit_value) == (
            "unknown",
            "unresolved",
            None,
        )
        assert session.query(ExitEvidence).filter_by(security_id=3).count() == 0


def test_evidence_dated_2025_or_later_is_ignored(research):
    FakeHttp.wiki = {
        "TESTA": [wiki_row("TESTA", "Test A", events="bankruptcy@2025-02-01T00:00:00Z")]
    }
    with session_scope(research) as session:
        add_security(session, 1, "TESTA", DAYS[0], DAYS[-1])
        evidence.fetch_wikidata(session, ["TESTA"], sleep=lambda s: None)
        evidence.resolve_outcomes(session, trading_days(date(2022, 1, 3), date(2024, 12, 31)))
    with session_scope(research) as session:
        assert session.get(Security, 1).outcome_status == "unresolved"


def test_a_ticker_that_trades_again_later_is_not_an_exit_loss(research):
    # Another price level after the pause: likely another company reusing the ticker.
    with session_scope(research) as session:
        add_security(session, 1, "TESTA", DAYS[0], DAYS[100])
        add_security(session, 2, "TESTA", DAYS[150], DAYS[-1], status="trading", price=45.0)
        evidence.resolve_outcomes(session, DAYS)
    with session_scope(research) as session:
        first = session.get(Security, 1)
        assert (first.outcome, first.outcome_status, first.assumed_exit_value) == (
            "resumed_or_fragment",
            "unresolved",
            None,
        )


def test_wikidata_server_errors_are_retried_then_reported_not_guessed(research, monkeypatch):
    attempts = []

    class Failing(FakeHttp):
        def get(self, url, params):
            attempts.append(1)
            return FakeResponse(502, {})

    monkeypatch.setattr(evidence.httpx, "Client", Failing)
    with session_scope(research) as session:
        requests, missing = evidence.fetch_wikidata(
            session, ["TESTA", "TESTB"], sleep=lambda s: None
        )
    with session_scope(research) as session:
        again, _ = evidence.fetch_wikidata(session, ["TESTA"], sleep=lambda s: None)

    assert (requests, missing) == (6, ["TESTA", "TESTB"])
    assert again == 6  # nothing was cached for the failed tickers, so they are asked again


def test_rebuilding_securities_clears_the_evidence_that_refers_to_them(research):
    from app.universe import pipeline
    from app.universe.segments import Bar, Chain, Segment

    with session_scope(research) as session:
        add_security(session, 1, "TESTA", DAYS[0], DAYS[100])
        evidence.resolve_outcomes(session, DAYS)  # writes exit evidence for security 1
        bars = tuple(Bar(d, 10, 10, 10, 10, 1000, 10, 5) for d in DAYS[:50])
        assembled = pipeline.Assembled([Chain([Segment("TESTB", bars)])], {}, {})
        pipeline.store_securities(session, assembled)  # must not fail on the foreign keys
    with session_scope(research) as session:
        assert session.query(ExitEvidence).count() == 0
        assert [s.last_symbol for s in session.query(Security)] == ["TESTB"]


def test_openfigi_answer_for_a_former_ticker_of_a_trading_security_is_name_checked(research):
    # Like FB -> META: OpenFIGI says FB is now an ETF (a later reuse); the company trades as META.
    FakeHttp.figi = {
        "TESTF": [
            {"name": "SOME BUFFER ETF", "securityType": "ETP", "securityType2": "Mutual Fund"}
        ],
        "TESTM": [
            {
                "name": "TEST MEDIA INC",
                "securityType": "Common Stock",
                "securityType2": "Common Stock",
            }
        ],
    }
    FakeHttp.wiki = {
        "TESTF": [wiki_row("TESTF", "Test Media", types="public company")],
        "TESTM": [wiki_row("TESTM", "Test Media", types="public company")],
    }
    with session_scope(research) as session:
        session.add(
            Security(
                security_id=1,
                first_session=DAYS[0],
                last_trade_session=DAYS[-1],
                last_symbol="TESTM",
                status="trading",
                exit_type="none",
                declared_type="unknown",
                real_trade_sessions=250,
            )
        )
        session.add(
            TickerHistory(
                security_id=1,
                symbol="TESTF",
                valid_from=DAYS[0],
                valid_to=DAYS[99],
                linked_by="first listing",
            )
        )
        session.add(
            TickerHistory(
                security_id=1,
                symbol="TESTM",
                valid_from=DAYS[100],
                valid_to=DAYS[-1],
                linked_by="rename",
            )
        )
        session.flush()
        evidence.fetch_openfigi(session, ["TESTF", "TESTM"], sleep=lambda s: None)
        evidence.fetch_wikidata(session, ["TESTF", "TESTM"], sleep=lambda s: None)
        evidence.resolve_types(session)
    with session_scope(research) as session:
        security = session.get(Security, 1)
        assert (security.security_type, security.type_status) == ("common", "confirmed")
        assert (
            session.query(TypeEvidence)
            .filter_by(source="openfigi_rejected", symbol="TESTF")
            .count()
            == 1
        )


def test_the_same_instrument_after_a_pause_is_a_trading_gap_not_an_exit(research):
    with session_scope(research) as session:
        add_security(session, 1, "TESTA", DAYS[0], DAYS[100])
        add_security(session, 2, "TESTA", DAYS[150], DAYS[-1], status="trading", price=11.0)
        evidence.resolve_outcomes(session, DAYS)
    with session_scope(research) as session:
        first = session.get(Security, 1)
        assert (first.outcome, first.outcome_status, first.assumed_exit_value) == (
            "trading_gap",
            "not_an_exit",
            None,
        )
        assert evidence.TRADING_GAP_MAX_SESSIONS == 250
