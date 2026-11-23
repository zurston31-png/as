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
    Security,
    SecurityBar,
    SecurityIssuer,
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
        session.add(SecurityIssuer(security_id=1, cik="0000000001", name="A", how="form25"))
        session.add(SecurityIssuer(security_id=2, cik="0000000002", name="B", how="ticker_today"))
    [(decision, reason)] = decide(research)
    assert decision == "kept_separate"
    assert "different issuers" in reason


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
