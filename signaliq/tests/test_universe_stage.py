"""Staged universe: candidate pool, monthly screen, completeness bound, permanent ids.

SYNTHETIC TEST FIXTURES ONLY: fake tickers and invented prices. Nothing here calls
Alpaca, SEC or Wikidata.
"""

import json
import random
import statistics
from datetime import date

import pytest

from app.data.validation import trading_days
from app.database import create_session_factory, session_scope
from app.universe import pipeline as p
from app.universe import segments as seg
from app.universe import stage
from app.universe.database import open_research_database
from app.universe.models import Candidate, MonthlyBar, SecurityRegistry

# =============================================================================
# Sources and pool
# =============================================================================


def test_sec_list_keeps_exchange_listed_tickers_only():
    text = json.dumps(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[1, "A", "TSTA", "Nasdaq"], [2, "B", "TSTB", "OTC"], [3, "C", "tstc", "NYSE"]],
        }
    )
    assert stage.sec_tickers(text) == {"TSTA", "TSTC"}
    assert stage.sec_tickers("not json") == set()


def test_wikidata_listings_must_overlap_2016_2024():
    def row(ticker, start="", end=""):
        out = {"ticker": {"value": ticker}}
        if start:
            out["tStart"] = {"value": start + "T00:00:00Z"}
        if end:
            out["tEnd"] = {"value": end + "T00:00:00Z"}
        return out

    rows = [
        row("OLDA", end="2014-05-01"),  # gone before 2016
        row("GONE", end="2018-06-14"),  # left during the period: kept
        row("UNDT"),  # undated: kept
        row("NEWA", start="2025-03-01"),  # listed after 2024
    ]
    assert stage.wikidata_tickers(rows) == {"GONE", "UNDT"}


def test_pool_keeps_plain_tickers_and_every_source_naming_them():
    pool = stage.build_pool(
        {"sec_ticker_list": {"TSTA", "TST.B"}, "wikidata_listing": {"TSTA", "OLDB"}}
    )
    assert pool == {"TSTA": {"sec_ticker_list", "wikidata_listing"}, "OLDB": {"wikidata_listing"}}


# =============================================================================
# Screen and completeness bound
# =============================================================================


def monthly(symbol, month, value, high=10.0, trades=100):
    volume = int(value / high) if high else 0
    return MonthlyBar(
        symbol=symbol,
        month=month,
        high=high,
        close=high,
        volume=volume,
        vwap=high,
        trade_count=trades,
    )


def test_monthly_ranks_ignore_filler_months_and_keep_the_best_rank():
    rows = [
        monthly("TSTA", date(2020, 1, 1), 5e9),
        monthly("TSTB", date(2020, 1, 1), 9e9),
        monthly("TSTA", date(2020, 2, 1), 9e9),
        monthly("TSTB", date(2020, 2, 1), 1e9),
        monthly("TSTC", date(2020, 2, 1), 9e12, trades=0),  # no trade: a filler
    ]
    result = stage.monthly_ranks(rows)
    assert result.best_rank == {"TSTA": 1, "TSTB": 1}
    assert result.best_month["TSTA"] == date(2020, 2, 1)
    assert "TSTC" not in result.best_rank


@pytest.mark.parametrize("seed", range(20))
def test_the_bound_is_never_below_the_true_63_session_median(seed):
    """Random daily prices and volumes: the monthly bound must cover the real median."""
    rng = random.Random(seed)
    days = trading_days(date(2021, 1, 4), date(2021, 6, 30))
    daily = {}
    for d in days:
        low = rng.uniform(5, 50)
        close = low * rng.uniform(1.0, 1.3)
        high = close * rng.uniform(1.0, 1.2)
        volume = 0 if rng.random() < 0.1 else rng.randint(1, 10_000_000)
        daily[d] = (high, close, volume)
    bars = {}
    for d, (high, close, volume) in daily.items():
        month = d.replace(day=1)
        old = bars.get(month)
        bars[month] = MonthlyBar(
            symbol="TSTA",
            month=month,
            high=max(high, old.high) if old else high,
            close=close,
            volume=(old.volume if old else 0) + volume,
            vwap=close,
            trade_count=1,
        )
    decision = max(d for d in days if d.month == 6)
    window = [d for d in days if d <= decision][-63:]
    true_median = statistics.median(daily[d][1] * daily[d][2] for d in window)
    assert stage.median_bound(bars, decision, min_price=5.0) >= true_median


def test_a_ticker_below_the_price_filter_all_month_cannot_be_a_member():
    bars = {date(2021, 6, 1): monthly("TSTA", date(2021, 6, 1), 1e12, high=4.9)}
    assert stage.median_bound(bars, date(2021, 6, 30), min_price=5.0) == 0.0


@pytest.fixture
def research(tmp_path):
    engine = open_research_database(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    factory = create_session_factory(engine)
    yield factory
    engine.dispose()


def test_screen_records_every_candidate_and_completeness_finds_what_it_missed(research):
    months = [date(2020, m, 1) for m in range(1, 7)]
    with session_scope(research) as session:
        for i, month in enumerate(months):
            session.add(monthly("BIGA", month, 9e10))
            session.add(monthly("BIGB", month, 8e10))
            session.add(monthly("SMLA", month, 1e6))
            # Huge only in the last month, after the screen would have caught it.
            session.add(monthly("LATE", month, 1e6 if i < 5 else 7e10))
        pool = {s: {"sec_ticker_list"} for s in ("BIGA", "BIGB", "SMLA", "LATE", "NONE")}
        selected = stage.screen_candidates(session, pool, n=1, forced={})
        assert selected == ["BIGA", "BIGB", "LATE"]  # margin 3: ranks 1-3 in some month
        reasons = {c.symbol: c for c in session.query(Candidate)}
        assert reasons["NONE"].reason == "no trade in any month 2016-2024"
        assert reasons["SMLA"].selection == "not_selected"
        # Suppose the N-th member on 2020-06-30 traded 1e8 a day: SMLA could not have joined.
        gaps = stage.completeness_gaps(session, {date(2020, 6, 30): 1e8}, min_price=5.0)
        assert gaps == {}
        # With a threshold below SMLA's bound, it is reported as a possible member.
        gaps = stage.completeness_gaps(session, {date(2020, 6, 30): 1e3}, min_price=5.0)
        assert list(gaps) == ["SMLA"]


# =============================================================================
# Permanent ids
# =============================================================================


def chain(symbol, days, price=10.0):
    bars = [seg.Bar(d, price, price, price, price, 100, price, 5) for d in days]
    return seg.Chain([seg.Segment(symbol, tuple(bars))])


def test_ids_stay_the_same_across_runs_and_merges_are_recorded(research):
    days = trading_days(date(2022, 1, 3), date(2022, 12, 30))
    a, b, c = chain("TSTA", days[:50]), chain("TSTA", days[80:]), chain("TSTB", days)
    with session_scope(research) as session:
        assert p.assign_security_ids(session, [a, b, c]) == [1, 2, 3]
    with session_scope(research) as session:
        # A later run, other order, plus a new security: the old ids hold, the new one is 4.
        d = chain("TSTC", days)
        assert p.assign_security_ids(session, [c, d, a, b]) == [3, 4, 1, 2]
    with session_scope(research) as session:
        # The two TSTA segments become one security (a trading-gap merge).
        merged = seg.Chain([a.segments[0], b.segments[0]], ["gap:TSTA"])
        assert p.assign_security_ids(session, [merged, c]) == [1, 3]
        rows = {r.security_id: r for r in session.query(SecurityRegistry)}
        assert (rows[2].status, rows[2].merged_into) == ("merged", 1)
        assert rows[4].status == "absent"
        assert rows[1].status == "active"
