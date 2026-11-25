"""Trading-gap merges: one security across a pause, separate securities for a reused ticker.

SYNTHETIC TEST FIXTURES ONLY: fake tickers (TESTA...), invented prices and invented SEC
findings. Nothing here calls SEC or Alpaca.
"""

from datetime import date

import pytest

from app.data.validation import trading_days
from app.database import create_session_factory, session_scope
from app.universe import evidence, sec
from app.universe import pipeline as p
from app.universe import segments as seg
from app.universe.curated import CuratedFact, SecQuote
from app.universe.database import open_research_database
from app.universe.models import (
    CorporateAction,
    IssuerEvidence,
    Security,
    SecurityBar,
    SegmentLink,
    TickerBar,
    TickerHistory,
)

DAYS = trading_days(date(2022, 1, 3), date(2022, 12, 30))


def bars(days, price):
    return [seg.Bar(d, price, price, price, price, 1000, price, 10) for d in days]


# =============================================================================
# Chain building
# =============================================================================


def test_a_gap_link_joins_two_segments_of_one_ticker_and_nothing_else():
    first, _ = seg.split_segments("TESTA", bars(DAYS[:50], 10) + bars(DAYS[80:150], 10.5))
    assert len(first) == 2  # a 29-session silence cuts the listing in two
    segments = {"TESTA": first}
    assert len(seg.build_chains(segments, [])) == 2
    link = seg.GapLink("gap:TESTA", "TESTA", DAYS[49], DAYS[80])
    chains = seg.build_chains(segments, [], [link])
    assert len(chains) == 1
    assert [s.first_session for s in chains[0].segments] == [DAYS[0], DAYS[80]]
    assert chains[0].links == ["gap:TESTA"]
    # A link naming segments that do not exist changes nothing.
    wrong = seg.GapLink("gap:x", "TESTA", DAYS[48], DAYS[80])
    assert len(seg.build_chains(segments, [], [wrong])) == 2


# =============================================================================
# Decisions from evidence
# =============================================================================


@pytest.fixture
def research(tmp_path, monkeypatch):
    engine = open_research_database(f"sqlite:///{(tmp_path / 'research.db').as_posix()}")
    factory = create_session_factory(engine)
    monkeypatch.setattr(evidence, "cached_sec_findings", lambda session: {})
    monkeypatch.setattr(evidence, "curated_facts", lambda session: {})
    yield factory
    engine.dispose()


def add(session, sid, symbol, days, price, status="ended"):
    session.add(
        Security(
            security_id=sid,
            first_session=days[0],
            last_trade_session=days[-1],
            last_symbol=symbol,
            status=status,
            exit_type="unexplained",
            declared_type="unknown",
            real_trade_sessions=len(days),
        )
    )
    session.add(
        TickerHistory(
            security_id=sid,
            symbol=symbol,
            valid_from=days[0],
            valid_to=days[-1],
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
                trade_count=10,
                split_factor=1.0,
            )
        )
    session.flush()


def decide(factory):
    with session_scope(factory) as session:
        rows = evidence.decide_gap_links(session)
        return [(r.decision, r.reason) for r in rows]


def test_a_pause_at_a_continuous_price_is_one_security(research):
    with session_scope(research) as session:
        add(session, 1, "TESTA", DAYS[:50], 10.0)
        add(session, 2, "TESTA", DAYS[80:], 10.4, status="trading")
    [(decision, reason)] = decide(research)
    assert decision == "merged"
    assert "same security" in reason


def test_a_delisting_notice_before_trading_resumes_keeps_them_apart(research, monkeypatch):
    with session_scope(research) as session:
        add(session, 1, "TESTA", DAYS[:50], 10.0)
        add(session, 2, "TESTA", DAYS[80:], 10.0, status="trading")
    notice = (
        sec.Filing("25-NSE", DAYS[55], "", "h"),
        sec.RemovalNotice("1", "OLD CO", "NYSE", "Common Stock", "17 CFR 240.12d2-2(b)", None),
    )
    monkeypatch.setattr(
        evidence, "cached_sec_findings", lambda s: {1: sec.SecFinding(cik="1", notices=[notice])}
    )
    [(decision, reason)] = decide(research)
    assert decision == "kept_separate"
    assert "delisting notice" in reason


def test_a_price_break_keeps_a_reused_ticker_apart(research):
    with session_scope(research) as session:
        add(session, 1, "TESTA", DAYS[:50], 10.0)
        add(session, 2, "TESTA", DAYS[80:], 45.0, status="trading")
    [(decision, reason)] = decide(research)
    assert decision == "kept_separate"
    assert "price does not resume" in reason


def test_different_issuers_keep_them_apart(research):
    with session_scope(research) as session:
        add(session, 1, "TESTA", DAYS[:50], 10.0)
        add(session, 2, "TESTA", DAYS[80:], 10.0, status="trading")
        for sid, cik, day in ((1, "0000000001", DAYS[30]), (2, "0000000002", DAYS[100])):
            session.add(
                IssuerEvidence(
                    security_id=sid,
                    symbol="TESTA",
                    filed=day,
                    cik=cik,
                    name=cik,
                    how="cover_page",
                    url="u",
                )
            )
    [(decision, reason)] = decide(research)
    assert decision == "kept_separate"
    assert "different issuers" in reason


def test_the_issuers_compared_across_a_gap_are_those_of_the_day(research):
    # The later security's ticker passed to another company only after it resumed; at the
    # gap both sides belong to one company, so the issuer test does not separate them.
    with session_scope(research) as session:
        add(session, 1, "TESTA", DAYS[:50], 10.0)
        add(session, 2, "TESTA", DAYS[80:], 10.0, status="trading")
        for sid, cik, day in (
            (1, "0000000001", DAYS[30]),
            (2, "0000000001", DAYS[85]),
            (2, "0000000002", DAYS[200]),
        ):
            session.add(
                IssuerEvidence(
                    security_id=sid,
                    symbol="TESTA",
                    filed=day,
                    cik=cik,
                    name=cik,
                    how="cover_page",
                    url="u",
                )
            )
    [(decision, reason)] = decide(research)
    assert decision == "merged", reason


def test_an_exit_during_the_gap_keeps_them_apart(research):
    with session_scope(research) as session:
        add(session, 1, "TESTA", DAYS[:50], 10.0)
        add(session, 2, "TESTA", DAYS[80:], 10.0, status="trading")
        session.add(
            CorporateAction(
                id="m1",
                action_type="cash_merger",
                process_date=DAYS[52],
                acquiree_symbol="TESTA",
                rate=10.0,
                raw={},
            )
        )
    [(decision, reason)] = decide(research)
    assert decision == "kept_separate"
    assert "cash_merger" in reason


def test_a_long_gap_keeps_them_apart(research):
    days = trading_days(date(2021, 1, 4), date(2022, 12, 30))
    with session_scope(research) as session:
        add(session, 1, "TESTA", days[:20], 10.0)
        add(session, 2, "TESTA", days[300:], 10.0, status="trading")
    [(decision, reason)] = decide(research)
    assert decision == "kept_separate"
    assert "longer than" in reason


def test_sec_filings_stating_the_same_shares_resumed_override_a_price_drop(research, monkeypatch):
    with session_scope(research) as session:
        add(session, 1, "TESTA", DAYS[:50], 0.05)
        add(session, 2, "TESTA", DAYS[80:], 3.0, status="trading")  # 60x, no split recorded
    fact = CuratedFact("TESTA", DAYS[49], None, "trading_gap", DAYS[80], (SecQuote("u", "x"),), "")
    monkeypatch.setattr(evidence, "curated_facts", lambda s: {1: [(fact, [])]})
    [(decision, _)] = decide(research)
    assert decision == "merged"
    # The same fact with an unverified quote does not count.
    monkeypatch.setattr(evidence, "curated_facts", lambda s: {1: [(fact, ["missing"])]})
    with session_scope(research) as session:
        session.execute(SegmentLink.__table__.delete())
    [(decision, _)] = decide(research)
    assert decision == "kept_separate"


def test_merges_from_earlier_passes_are_kept_and_assembly_uses_them(research):
    with session_scope(research) as session:
        for d in DAYS[:50]:
            session.add(
                TickerBar(
                    symbol="TESTA",
                    session=d,
                    open=10,
                    high=10,
                    low=10,
                    close=10,
                    volume=1000,
                    vwap=10,
                    trade_count=10,
                    is_filler=False,
                )
            )
        for d in DAYS[80:]:
            session.add(
                TickerBar(
                    symbol="TESTA",
                    session=d,
                    open=10.2,
                    high=10.2,
                    low=10.2,
                    close=10.2,
                    volume=1000,
                    vwap=10.2,
                    trade_count=10,
                    is_filler=False,
                )
            )
        add(session, 1, "TESTA", DAYS[:50], 10.0)
        add(session, 2, "TESTA", DAYS[80:], 10.2, status="trading")
    assert decide(research)[0][0] == "merged"
    with session_scope(research) as session:
        chains = p.assemble(session).chains
        assert len(chains) == 1
        assert len(chains[0].segments) == 2
    # A second pass, with the securities rebuilt as one, finds nothing new and keeps the merge.
    with session_scope(research) as session:
        session.execute(TickerHistory.__table__.delete())
        session.execute(SecurityBar.__table__.delete())
        session.execute(Security.__table__.delete())
        add(session, 1, "TESTA", DAYS[:50] + DAYS[80:], 10.0, status="trading")
    assert decide(research) == []
    with session_scope(research) as session:
        kept = session.query(SegmentLink).filter_by(decision="merged").count()
        assert kept == 1


# =============================================================================
# Documented identity exceptions (a long suspension under a new ticker)
# =============================================================================

LONG = trading_days(date(2021, 1, 4), date(2024, 12, 31))


def test_an_identity_link_joins_two_tickers_across_a_long_suspension_and_nothing_else():
    old, _ = seg.split_segments("TESTA", bars(LONG[:100], 18.9))
    new, _ = seg.split_segments("TESTB", bars(LONG[800:], 20.0))
    segments = {"TESTA": old, "TESTB": new}
    rename = seg.NameChange("rn", "TESTA", "TESTB", LONG[790])
    # A rename alone does not bridge 699 sessions without a trade.
    assert len(seg.build_chains(segments, [rename])) == 2
    link = seg.IdentityLink("identity:TESTA>TESTB", "TESTA", LONG[99], "TESTB", LONG[800])
    chains = seg.build_chains(segments, [rename], [], [link])
    assert len(chains) == 1
    assert [s.symbol for s in chains[0].segments] == ["TESTA", "TESTB"]
    assert chains[0].links == ["identity:TESTA>TESTB"]
    # Off by one session on either side: the named segments do not exist, nothing is joined.
    for wrong in (
        seg.IdentityLink("x", "TESTA", LONG[98], "TESTB", LONG[800]),
        seg.IdentityLink("x", "TESTA", LONG[99], "TESTB", LONG[801]),
    ):
        assert len(seg.build_chains(segments, [], [], [wrong])) == 2


def test_an_identity_link_never_joins_overlapping_trading():
    old, _ = seg.split_segments("TESTA", bars(LONG[:100], 10))
    new, _ = seg.split_segments("TESTB", bars(LONG[50:200], 10))
    link = seg.IdentityLink("x", "TESTA", LONG[99], "TESTB", LONG[50])
    assert len(seg.build_chains({"TESTA": old, "TESTB": new}, [], [], [link])) == 2


def _ticker_bars(session, symbol, days, price):
    for d in days:
        session.add(
            TickerBar(
                symbol=symbol,
                session=d,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1000,
                vwap=price,
                trade_count=10,
                is_filler=False,
            )
        )


@pytest.mark.parametrize(("missing", "expected"), [([], 1), (["a quote"], 2)])
def test_assembly_applies_an_identity_exception_only_when_its_sec_quotes_are_found(
    research, monkeypatch, missing, expected
):
    from app.universe.curated import CuratedIdentityLink

    link = CuratedIdentityLink(
        "TESTA", LONG[99], "TESTB", LONG[800], "0000000001", (SecQuote("u", "t"),), "test"
    )
    monkeypatch.setattr(evidence, "curated_identity_links", lambda s: [(link, missing)])
    with session_scope(research) as session:
        _ticker_bars(session, "TESTA", LONG[:100], 18.9)
        _ticker_bars(session, "TESTB", LONG[800:], 20.0)
    with session_scope(research) as session:
        chains = p.assemble(session).chains
    assert len(chains) == expected
    if expected == 1:
        assert chains[0].links == [evidence.identity_link_id(link)]


def test_the_yandex_nebius_exception_names_one_registrant_and_its_exact_segments():
    from app.universe.curated import CURATED_IDENTITY_LINKS

    [link] = CURATED_IDENTITY_LINKS
    assert (link.old_symbol, link.new_symbol, link.cik) == ("YNDX", "NBIS", "0001513845")
    assert link.old_last == date(2022, 2, 25)
    assert link.new_first == date(2024, 10, 21)
    # Every quote is from the registrant's own filings, all dated before 2025.
    assert all("/edgar/data/1513845/" in q.url for q in link.quotes)
    assert all(q.url.split("/")[-2][10:12] in ("22", "24") for q in link.quotes)


# =============================================================================
# Renames confirmed by the issuer (no rename record joins them)
# =============================================================================


def _cover(session, sid, symbol, day, cik):
    session.add(
        IssuerEvidence(
            security_id=sid,
            symbol=symbol,
            filed=day,
            cik=cik,
            name=f"Co {cik}",
            how="cover_page",
            url="u",
        )
    )


def _two_tickers(session, price_after, cik_after="0000000001"):
    add(session, 1, "TESTA", DAYS[:120], 10.0)
    add(session, 2, "TESTB", DAYS[121:], price_after, status="trading")
    for sid, symbol, days in ((1, "TESTA", DAYS[:120]), (2, "TESTB", DAYS[121:])):
        _ticker_bars(session, symbol, days, 10.0 if sid == 1 else price_after)
    _cover(session, 1, "TESTA", DAYS[100], "0000000001")
    _cover(session, 2, "TESTB", DAYS[140], cik_after)


def test_one_registrant_listing_the_old_ticker_then_the_new_one_is_one_security(research):
    with session_scope(research) as session:
        _two_tickers(session, 10.3)
    with session_scope(research) as session:
        [row] = evidence.decide_issuer_renames(session)
        assert row.decision == "renamed"
        assert "CIK 0000000001 listed TESTA until the switch and TESTB" in row.reason
    with session_scope(research) as session:
        chains = p.assemble(session).chains
    assert [[s.symbol for s in c.segments] for c in chains] == [["TESTA", "TESTB"]]


def test_a_price_break_or_another_registrant_keeps_consecutive_tickers_apart(research):
    with session_scope(research) as session:
        _two_tickers(session, 31.0)
    with session_scope(research) as session:
        assert evidence.decide_issuer_renames(session) == []
        [row] = session.query(SegmentLink).all()
        assert row.decision == "rename_rejected"
        assert "price does not continue" in row.reason
    with session_scope(research) as session:
        session.execute(SegmentLink.__table__.delete())
        session.execute(IssuerEvidence.__table__.delete())
        _cover(session, 1, "TESTA", DAYS[100], "0000000001")
        _cover(session, 2, "TESTB", DAYS[140], "0000000002")
    with session_scope(research) as session:
        assert evidence.decide_issuer_renames(session) == []
        assert session.query(SegmentLink).count() == 0  # not even a candidate
    with session_scope(research) as session:
        assert len(p.assemble(session).chains) == 2
