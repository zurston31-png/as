"""Staged universe: a survivorship-aware candidate pool, a monthly screen, a completeness proof.

The universe at a monthly decision is the top ``N`` eligible common stocks by 63-session
median dollar volume (``app.universe.membership``). Downloading daily bars for every US
ticker is not needed to find them exactly:

1. **Candidate pool.** Every ticker that any of these sources places on a US exchange at
   some point from 2016 to 2024, never today's list alone:

   * ``sec_ticker_list``: SEC's current list of exchange-listed tickers (survivors);
   * ``alpaca_corporate_actions``: tickers named by Alpaca's corporate actions 2016-2024
     (acquired, renamed, bankrupt, removed); dense from 2020 only;
   * ``wikidata_listing``: Wikidata's US listings (NYSE, Nasdaq, NYSE American, NYSE
     Arca) with dates, which cover companies that left before 2020;
   * ``validation_batch``: the named historical cases of the validation batch.

2. **Monthly screen.** Monthly SIP bars for every candidate. A candidate gets daily data
   if it ranks within ``margin x N`` by monthly traded value in any month that feeds a
   decision.

3. **Completeness proof.** For a ticker left out, an upper bound on its 63-session median
   dollar volume at each decision is computed from its monthly bars: every daily close is
   at most the month's high, so the daily dollar volumes over the four calendar months
   that contain the 63-session window sum to at most ``sum(high x volume)``; the window's
   mean is at most that sum over 63; and a median of non-negative numbers is at most twice
   their mean. If that bound reaches the dollar volume of the N-th member, the ticker
   could have been a member: it gets daily data and the stage runs again, until no ticker
   left out could have been one. A ticker whose month's high is below the minimum price
   cannot pass the price filter and is not counted.

Only screening uses the full period: which tickers get downloaded. Membership itself uses
each security's daily bars up to the decision only, so this does not leak the future.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.data.alpaca_client import AlpacaClient
from app.universe.batch import KNOWN_CASES, PLAIN_TICKER
from app.universe.models import Candidate, ExternalResponse, MonthlyBar

POOL_FIRST, POOL_LAST = date(2016, 1, 1), date(2024, 12, 31)
#: Months whose traded value can enter a decision: the 63-session window of the first
#: decision (January 2020) starts in October 2019.
SCREEN_FIRST, SCREEN_LAST = date(2019, 9, 1), date(2024, 11, 1)
#: A candidate ranking within MARGIN x N in some month gets daily bars.
MARGIN = 3
WINDOW_SESSIONS = 63
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
US_EXCHANGES = {
    "Q13677": "NYSE",
    "Q82059": "Nasdaq",
    "Q846626": "NYSE American",
    "Q2873302": "NYSE Arca",
}
_EXCHANGE_VALUES = " ".join(f"wd:{q}" for q in US_EXCHANGES)
WIKIDATA_LISTINGS = (
    "SELECT ?ticker ?item ?exch ?tStart ?tEnd WHERE { VALUES ?exch { " + _EXCHANGE_VALUES + " } "
    "?item p:P414 ?st . ?st ps:P414 ?exch ; pq:P249 ?ticker . "
    "OPTIONAL { ?st pq:P580 ?tStart } OPTIONAL { ?st pq:P582 ?tEnd } }"
)
CORPORATE_ACTION_TYPES = (
    "cash_merger,stock_merger,stock_and_cash_merger,name_change,worthless_removal,"
    "forward_split,reverse_split"
)


# =============================================================================
# Sources (cached in external_responses, so a stage can be rebuilt offline)
# =============================================================================


def _cached(session: Session, source: str, key: str):
    row = session.get(ExternalResponse, (source, key))
    return None if row is None else row.response


def _store(session: Session, source: str, key: str, response) -> None:
    session.merge(ExternalResponse(source=source, request_key=key, response=response))


def wikidata_listings(session: Session, *, network: bool, user_agent: str) -> list[dict]:
    """Wikidata statements "listed on <US exchange> with ticker X", with dates."""
    key = "us_listings_v1"
    cached = _cached(session, "wikidata_pool", key)
    if cached is None and network:
        response = httpx.get(
            WIKIDATA_URL,
            params={"query": WIKIDATA_LISTINGS, "format": "json"},
            headers={"User-Agent": user_agent, "Accept": "application/sparql-results+json"},
            timeout=300,
        )
        response.raise_for_status()
        cached = response.json()["results"]["bindings"]
        _store(session, "wikidata_pool", key, cached)
    return cached or []


def _day(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10]) if value and value[0].isdigit() else None
    except ValueError:
        return None


def wikidata_tickers(rows: Iterable[dict]) -> set[str]:
    """Tickers whose listing period may overlap 2016-2024 (undated periods are kept)."""
    out = set()
    for row in rows:
        start = _day(row.get("tStart", {}).get("value", ""))
        end = _day(row.get("tEnd", {}).get("value", ""))
        if start is not None and start > POOL_LAST:
            continue
        if end is not None and end < POOL_FIRST:
            continue
        out.add(row["ticker"]["value"].strip().upper())
    return out


def sec_tickers(text: str | None) -> set[str]:
    """Tickers on NYSE, Nasdaq or Cboe in SEC's current list (survivors only)."""
    try:
        data = json.loads(text or "")
    except ValueError:
        return set()
    fields = data.get("fields", [])
    if "ticker" not in fields or "exchange" not in fields:
        return set()
    t, x = fields.index("ticker"), fields.index("exchange")
    return {row[t].upper() for row in data.get("data", []) if row[x] in ("NYSE", "Nasdaq", "CBOE")}


def alpaca_action_tickers(
    session: Session, client: AlpacaClient | None, fetch
) -> tuple[set[str], int]:
    """Tickers named by Alpaca's market-wide corporate actions 2016-2024, by quarter."""
    tickers: set[str] = set()
    events = 0
    for year in range(POOL_FIRST.year, POOL_LAST.year + 1):
        for quarter in range(4):
            first = date(year, 3 * quarter + 1, 1)
            last = (date(year + (quarter == 3), (3 * quarter + 3) % 12 + 1, 1)) - timedelta(days=1)
            key = f"{first.isoformat()}:{last.isoformat()}"
            items = _cached(session, "alpaca_pool", key)
            if items is None and client is not None:
                items = fetch(
                    client, symbols=None, types=CORPORATE_ACTION_TYPES, first=first, last=last
                )
                _store(session, "alpaca_pool", key, items)
            for item in items or []:
                events += 1
                for field in (
                    "symbol",
                    "old_symbol",
                    "new_symbol",
                    "acquiree_symbol",
                    "acquirer_symbol",
                    "target_symbol",
                    "initiating_symbol",
                ):
                    value = item.get(field)
                    if isinstance(value, str):
                        tickers.add(value.strip().upper())
    return tickers, events


def build_pool(sources: dict[str, set[str]]) -> dict[str, set[str]]:
    """Plain tickers and the sources that name each."""
    pool: dict[str, set[str]] = defaultdict(set)
    for source, tickers in sources.items():
        for ticker in tickers:
            if PLAIN_TICKER.match(ticker):
                pool[ticker].add(source)
    return dict(pool)


def validation_tickers() -> set[str]:
    return {case.symbol for case in KNOWN_CASES}


# =============================================================================
# Monthly bars and the screen
# =============================================================================


def fetch_monthly_bars(
    session: Session, client: AlpacaClient | None, symbols: Sequence[str], utc_window
) -> int:
    """Monthly SIP raw bars, literal tickers, 2016-2024, for symbols not yet fetched.

    A symbol already fetched is marked by a cache row even when Alpaca had no bars for it.
    Returns the number of symbols fetched now.
    """
    missing = [s for s in symbols if _cached(session, "monthly_fetched", s) is None]
    if client is None or not missing:
        return 0
    start, end = utc_window(POOL_FIRST, POOL_LAST)
    for i in range(0, len(missing), 100):
        chunk = missing[i : i + 100]
        bars = client.fetch_bars(chunk, timeframe="1Month", start=start, end=end, asof="-")
        for symbol in chunk:
            for item in bars.get(symbol, []):
                month = date.fromisoformat(item["t"][:7] + "-01")
                if month > POOL_LAST:
                    continue
                session.merge(
                    MonthlyBar(
                        symbol=symbol,
                        month=month,
                        high=item["h"],
                        close=item["c"],
                        volume=int(item["v"]),
                        vwap=float(item.get("vw", item["c"])),
                        trade_count=int(item.get("n", 0)),
                    )
                )
            _store(session, "monthly_fetched", symbol, len(bars.get(symbol, [])))
        session.flush()
    return len(missing)


@dataclass(frozen=True)
class Screen:
    best_rank: dict[str, int]
    best_month: dict[str, date]
    months_with_trades: dict[str, int]


def monthly_ranks(rows: Iterable[MonthlyBar]) -> Screen:
    """Each symbol's best rank by monthly traded value (vwap x volume) in the screen months."""
    by_month: dict[date, list[tuple[float, str]]] = defaultdict(list)
    months: dict[str, int] = defaultdict(int)
    for bar in rows:
        if bar.volume <= 0 or bar.trade_count <= 0:
            continue  # a filler month: no trade
        months[bar.symbol] += 1
        if SCREEN_FIRST <= bar.month <= SCREEN_LAST:
            by_month[bar.month].append((bar.vwap * bar.volume, bar.symbol))
    best: dict[str, int] = {}
    best_month: dict[str, date] = {}
    for month in sorted(by_month):
        ranked = sorted(by_month[month], key=lambda item: (-item[0], item[1]))
        for rank, (_, symbol) in enumerate(ranked, start=1):
            if symbol not in best or rank < best[symbol]:
                best[symbol], best_month[symbol] = rank, month
    return Screen(best, best_month, dict(months))


def screen_candidates(
    session: Session, pool: dict[str, set[str]], n: int, forced: dict[str, str]
) -> list[str]:
    """Record every candidate with its screen result; return the symbols selected."""
    session.execute(delete(Candidate))
    result = monthly_ranks(session.execute(select(MonthlyBar)).scalars())
    selected = []
    for symbol in sorted(pool):
        rank = result.best_rank.get(symbol)
        months = result.months_with_trades.get(symbol, 0)
        if symbol in forced:
            selection, reason = "completeness", forced[symbol]
        elif rank is not None and rank <= MARGIN * n:
            selection = "screen"
            reason = f"ranked {rank} by monthly traded value in {result.best_month[symbol]:%Y-%m}"
        else:
            selection = "not_selected"
            reason = (
                "no trade in any month 2016-2024"
                if months == 0
                else "no trade in the screen months (2019-09 to 2024-11)"
                if rank is None
                else f"best monthly rank {rank} (outside {MARGIN * n})"
            )
        if selection != "not_selected":
            selected.append(symbol)
        session.add(
            Candidate(
                symbol=symbol,
                sources=",".join(sorted(pool[symbol])),
                months_with_trades=months,
                best_rank=rank,
                best_month=result.best_month.get(symbol),
                selection=selection,
                reason=reason,
            )
        )
    session.flush()
    return selected


def month_floor(day: date) -> date:
    return day.replace(day=1)


def months_back(month: date, count: int) -> date:
    year, index = divmod(month.year * 12 + month.month - 1 - count, 12)
    return date(year, index + 1, 1)


def median_bound(bars: dict[date, MonthlyBar], decision: date, min_price: float) -> float:
    """Upper bound on a ticker's 63-session median dollar volume at ``decision``.

    Zero when it cannot pass the price filter (its high in the decision month is below
    the minimum) or did not trade in the decision month.
    """
    month = month_floor(decision)
    now = bars.get(month)
    if now is None or now.volume <= 0 or now.high < min_price:
        return 0.0
    total = sum(
        b.high * b.volume
        for k in range(4)
        if (b := bars.get(months_back(month, k))) is not None and b.volume > 0
    )
    return 2.0 * total / WINDOW_SESSIONS


def completeness_gaps(
    session: Session, nth_member_volume: dict[date, float], min_price: float
) -> dict[str, tuple[date, float, float]]:
    """Tickers without daily data that could have been members: symbol -> (decision, bound, Nth)."""
    left_out = {
        c.symbol
        for c in session.execute(
            select(Candidate).where(Candidate.selection == "not_selected")
        ).scalars()
    }
    bars: dict[str, dict[date, MonthlyBar]] = defaultdict(dict)
    for bar in session.execute(select(MonthlyBar)).scalars():
        if bar.symbol in left_out:
            bars[bar.symbol][bar.month] = bar
    gaps = {}
    for symbol in sorted(bars):
        for decision, threshold in sorted(nth_member_volume.items()):
            bound = median_bound(bars[symbol], decision, min_price)
            if bound >= threshold:
                gaps[symbol] = (decision, bound, threshold)
                break
    return gaps


# =============================================================================
# Putting it together
# =============================================================================


def collect_pool(
    session: Session,
    client: AlpacaClient,
    *,
    fetch_actions,
    utc_window,
    sec_get,
    user_agent: str,
) -> tuple[dict[str, set[str]], dict]:
    """Build the candidate pool and fetch its monthly bars. Returns (pool, facts)."""
    wiki = wikidata_listings(session, network=True, user_agent=user_agent)
    actions, events = alpaca_action_tickers(session, client, fetch_actions)
    sources = {
        "sec_ticker_list": sec_tickers(sec_get(SEC_TICKERS_URL)),
        "alpaca_corporate_actions": actions,
        "wikidata_listing": wikidata_tickers(wiki),
        "validation_batch": validation_tickers(),
    }
    pool = build_pool(sources)
    fetched = fetch_monthly_bars(session, client, sorted(pool), utc_window)
    facts = {
        "source_sizes": {k: len(v) for k, v in sources.items()},
        "alpaca_events": events,
        "wikidata_rows": len(wiki),
        "pool_size": len(pool),
        "monthly_fetched_now": fetched,
    }
    return pool, facts


def build_stage_batch(session: Session, selected: Sequence[str]) -> int:
    """The stage's batch: every selected candidate, with its recorded reason."""
    from app.universe.models import BatchSymbol

    session.execute(delete(BatchSymbol))
    known = {case.symbol: case.declared_type for case in KNOWN_CASES}
    reasons = {c.symbol: c for c in session.execute(select(Candidate)).scalars()}
    for symbol in selected:
        candidate = reasons[symbol]
        session.add(
            BatchSymbol(
                symbol=symbol,
                category="stage",
                selection=f"stage:{candidate.selection}",
                reason=candidate.reason,
                declared_type=known.get(symbol, "unknown"),
            )
        )
    session.flush()
    return len(selected)


def nth_member_volume(session: Session, size: int) -> dict[date, float]:
    """Per decision, the 63-session median dollar volume of the N-th (last) member.

    When fewer than ``size`` securities were eligible, any eligible ticker would have been
    a member, so the threshold is zero.
    """
    from app.universe.models import Membership

    members: dict[date, list[float]] = defaultdict(list)
    for row in session.execute(select(Membership).where(Membership.member == 1)).scalars():
        members[row.decision_session].append(row.median_dollar_volume or 0.0)
    return {
        decision: (min(values) if len(values) >= size else 0.0)
        for decision, values in members.items()
    }
