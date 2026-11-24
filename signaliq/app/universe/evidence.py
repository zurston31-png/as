"""Collect independent evidence and apply it: split checks, spin-offs, types, outcomes.

Sources (all read-only):

* Alpaca corporate actions (mergers, renames, splits, spin-offs) - already stored.
* OpenFIGI (api.openfigi.com) - security type of a ticker's most recent holder.
* Wikidata (query.wikidata.org) - company items listed under a ticker: owners,
  parent organisations, bankruptcy and collapse events, listing end dates,
  "instance of" classes.
* SEC EDGAR (www.sec.gov) - the delisting notice (Form 25 / 25-NSE) filed when a
  security stopped trading: class of security, delisting rule, and 8-K filings on
  bankruptcy or completed acquisitions (``app.universe.sec``).
* Hand-entered corrections (``app.universe.curated``), each checked against prices.

Responses from OpenFIGI, Wikidata and SEC are stored in ``external_responses`` and
reused on later runs, so a report can be reproduced exactly; ``refresh=True``
asks the sources again. Any evidence dated 2025 or later is ignored.
"""

import logging
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import is_valid_symbol
from app.data.alpaca_client import AlpacaClient
from app.universe import adjustments, outcomes, sec, sectypes
from app.universe import sec_history as sh
from app.universe import segments as seg
from app.universe.batch import PLAIN_TICKER
from app.universe.curated import (
    CURATED_DEALS,
    CURATED_FACTS,
    CURATED_IDENTITY_LINKS,
    CURATED_RENAMES,
    CURATED_SPIN_OFFS,
    CuratedFact,
    CuratedIdentityLink,
    CuratedRename,
)
from app.universe.models import (
    BatchSymbol,
    CorporateAction,
    DealValue,
    ExitEvidence,
    ExternalResponse,
    IssuerEvidence,
    IssuerSic,
    Security,
    SecurityBar,
    SecurityIssuer,
    SegmentLink,
    SpinOff,
    SplitCheck,
    TickerHistory,
    TypeEvidence,
)

logger = logging.getLogger(__name__)

USER_AGENT = "SignalIQ-research/0.1"
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
OPENFIGI_PAUSE_SECONDS = 2.6  # 25 requests per minute without a key
WIKIDATA_PAUSE_SECONDS = 65.0  # Wikimedia currently allows one query per minute
WIKIDATA_GROUP = 25
MERGER_WINDOW_SESSIONS = 60
FRAGMENT_SESSIONS = 60

WIKIDATA_QUERY = """SELECT ?ticker ?item ?itemLabel ?exchLabel ?tStart ?tEnd
  (GROUP_CONCAT(DISTINCT ?typeLabel; separator="|") AS ?types)
  (GROUP_CONCAT(DISTINCT CONCAT(?ownerLabel, "@", STR(?ownStart)); separator="|") AS ?owners)
  (GROUP_CONCAT(DISTINCT CONCAT(?parentLabel, "@", STR(?parStart)); separator="|") AS ?parents)
  (SAMPLE(?dissolved) AS ?dissolvedAt)
  (GROUP_CONCAT(DISTINCT CONCAT(?eventLabel, "@", STR(?eventAt)); separator="|") AS ?events)
WHERE {
  VALUES ?ticker { %s }
  ?item p:P414 ?st . ?st ps:P414 ?exch ; pq:P249 ?ticker .
  OPTIONAL { ?st pq:P580 ?tStart } OPTIONAL { ?st pq:P582 ?tEnd }
  OPTIONAL { ?item wdt:P31 ?type . ?type rdfs:label ?typeLabel FILTER(LANG(?typeLabel) = "en") }
  OPTIONAL { ?item p:P127 ?os . ?os ps:P127 ?owner . ?owner rdfs:label ?ownerLabel FILTER(LANG(?ownerLabel) = "en") OPTIONAL { ?os pq:P580 ?ownStart } }
  OPTIONAL { ?item p:P749 ?ps . ?ps ps:P749 ?parent . ?parent rdfs:label ?parentLabel FILTER(LANG(?parentLabel) = "en") OPTIONAL { ?ps pq:P580 ?parStart } }
  OPTIONAL { ?item wdt:P576 ?dissolved }
  OPTIONAL { ?item p:P793 ?es . ?es ps:P793 ?event . ?event rdfs:label ?eventLabel FILTER(LANG(?eventLabel) = "en") OPTIONAL { ?es pq:P585 ?eventAt } }
  ?exch rdfs:label ?exchLabel FILTER(LANG(?exchLabel) = "en")
  ?item rdfs:label ?itemLabel FILTER(LANG(?itemLabel) = "en")
} GROUP BY ?ticker ?item ?itemLabel ?exchLabel ?tStart ?tEnd"""  # noqa: E501


# =============================================================================
# External sources (cached)
# =============================================================================


def _cached(session: Session, source: str, key: str):
    row = session.get(ExternalResponse, (source, key))
    return None if row is None else row.response


def _store(session: Session, source: str, key: str, response) -> None:
    session.merge(ExternalResponse(source=source, request_key=key, response=response))


def fetch_openfigi(
    session: Session, tickers: list[str], *, refresh: bool = False, sleep=time.sleep
) -> int:
    """Map tickers to OpenFIGI records (10 per request). Returns requests made."""
    wanted = [t for t in tickers if refresh or _cached(session, "openfigi", t) is None]
    requests = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as http:
        for i in range(0, len(wanted), 10):
            chunk = wanted[i : i + 10]
            jobs = [
                {
                    "idType": "TICKER",
                    "idValue": t,
                    "exchCode": "US",
                    "includeUnlistedEquities": True,
                }
                for t in chunk
            ]
            for _attempt in range(5):
                response = http.post(OPENFIGI_URL, json=jobs)
                requests += 1
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
                sleep(60)
            response.raise_for_status()
            for ticker, result in zip(chunk, response.json(), strict=True):
                _store(session, "openfigi", ticker, result.get("data") or [])
            session.commit()
            sleep(OPENFIGI_PAUSE_SECONDS)
    return requests


def fetch_wikidata(
    session: Session, tickers: list[str], *, refresh: bool = False, sleep=time.sleep
) -> tuple[int, list[str]]:
    """Look up company items listed under the tickers.

    Returns (requests made, tickers that could not be looked up). Rate limits
    (429) and server errors (5xx) are retried; a group that still fails is left
    out and reported, never guessed. Each answered group is committed at once so
    a later failure does not lose it.
    """
    wanted = [t for t in tickers if refresh or _cached(session, "wikidata", t) is None]
    requests, missing = 0, []
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    with httpx.Client(headers=headers, timeout=180) as http:
        for i in range(0, len(wanted), WIKIDATA_GROUP):
            chunk = wanted[i : i + WIKIDATA_GROUP]
            query = WIKIDATA_QUERY % " ".join(f'"{t}"' for t in chunk)
            response = None
            for _attempt in range(6):
                if requests:
                    sleep(WIKIDATA_PAUSE_SECONDS)
                try:
                    response = http.get(WIKIDATA_URL, params={"query": query, "format": "json"})
                except httpx.TransportError:
                    response = None
                requests += 1
                if response is not None and response.status_code == 200:
                    break
            if response is None or response.status_code != 200:
                logger.warning("Wikidata did not answer for %s", ", ".join(chunk))
                missing += chunk
                continue
            by_ticker: dict[str, list] = defaultdict(list)
            for binding in response.json()["results"]["bindings"]:
                by_ticker[binding["ticker"]["value"]].append(binding)
            for ticker in chunk:
                _store(session, "wikidata", ticker, by_ticker.get(ticker, []))
            session.commit()
    return requests, missing


# =============================================================================
# Helpers over stored securities
# =============================================================================


def chains_by_id(session: Session) -> dict[int, seg.Chain]:
    """Rebuild each security's chain (segments with their real-trade bars)."""
    bars: dict[int, list[SecurityBar]] = defaultdict(list)
    for bar in session.execute(
        select(SecurityBar).order_by(SecurityBar.security_id, SecurityBar.session)
    ).scalars():
        bars[bar.security_id].append(bar)
    chains = {}
    for security_id, items in bars.items():
        history = list(
            session.execute(
                select(TickerHistory)
                .where(TickerHistory.security_id == security_id)
                .order_by(TickerHistory.valid_from)
            ).scalars()
        )
        segments = []
        for h in history:
            run = tuple(
                seg.Bar(b.session, b.open, b.high, b.low, b.close, b.volume, b.vwap, b.trade_count)
                for b in items
                if h.valid_from <= b.session <= h.valid_to
            )
            segments.append(seg.Segment(h.symbol, run))
        chains[security_id] = seg.Chain(segments, [h.linked_by for h in history[1:]])
    return chains


def _close_around(chain: seg.Chain, day: date) -> tuple[float | None, float | None]:
    """Last real close before ``day`` and the close on ``day`` (None if it did not trade then)."""
    before = on = None
    for _, bar in chain.bars():
        if bar.session < day:
            before = bar.close
        elif bar.session == day:
            on = bar.close
    return before, on


# =============================================================================
# Splits and spin-offs
# =============================================================================


def apply_price_adjustments(session: Session, client: AlpacaClient, utc_window) -> None:
    """Check every owned split against prices, add spin-offs, and store the factors."""
    session.execute(delete(SplitCheck))
    session.execute(delete(SpinOff))
    chains = chains_by_id(session)
    splits = [
        seg.Split(e.symbol, e.ex_date or e.process_date, e.new_rate / e.old_rate, e.id)
        for e in session.execute(
            select(CorporateAction).where(
                CorporateAction.action_type.in_(("forward_split", "reverse_split", "unit_split"))
            )
        ).scalars()
        if e.symbol and e.new_rate and e.old_rate and (e.ex_date or e.process_date)
    ]
    accepted: dict[int, list[seg.Split]] = defaultdict(list)
    for security_id, chain in chains.items():
        for split in seg.owned_splits(chain, splits):
            before, on = _close_around(chain, split.ex_date)
            verdict = adjustments.check_split(split.ratio, before, on)
            session.add(
                SplitCheck(
                    event_id=split.event_id,
                    security_id=security_id,
                    symbol=split.symbol,
                    ex_date=split.ex_date,
                    ratio=split.ratio,
                    close_before=before,
                    close_on=on,
                    verdict=verdict,
                )
            )
            if verdict != "contradicted":
                accepted[security_id].append(split)

    # Spin-offs: Alpaca's records plus hand-entered ones.
    records = []
    for e in session.execute(
        select(CorporateAction).where(CorporateAction.action_type == "spin_off")
    ).scalars():
        ratio = (e.new_rate or 0) / (e.old_rate or 1)
        if e.symbol and e.new_symbol and ratio > 0 and (e.ex_date or e.process_date):
            records.append(
                (
                    "alpaca",
                    f"corporate action {e.id}",
                    e.symbol,
                    e.ex_date or e.process_date,
                    ((e.new_symbol, ratio),),
                )
            )
    for c in CURATED_SPIN_OFFS:
        records.append(("curated", c.reference, c.parent_symbol, c.ex_date, c.children))
    spin_factors: dict[int, list[tuple[date, float]]] = defaultdict(list)
    for source, reference, parent, ex_date, children in records:
        for security_id, chain in chains.items():
            if not seg.owned_on(chain, parent, ex_date):
                continue
            before, on = _close_around(chain, ex_date)
            child_prices = []
            for child, ratio in children:
                price = None
                if is_valid_symbol(child):
                    start, end = utc_window(ex_date, ex_date + timedelta(days=6))
                    items = client.fetch_bars(
                        [child], timeframe="1Day", start=start, end=end, asof="-"
                    ).get(child, [])
                    real = [x for x in items if x["v"] > 0 and x["t"][:10] == ex_date.isoformat()]
                    price = float(real[0]["c"]) if real else None
                child_prices.append({"symbol": child, "ratio": ratio, "close_on_ex_date": price})
            result = adjustments.spin_off_adjustment(
                before, on, [(c["ratio"], c["close_on_ex_date"]) for c in child_prices]
            )
            session.add(
                SpinOff(
                    security_id=security_id,
                    source=source,
                    reference=reference,
                    parent_symbol=parent,
                    ex_date=ex_date,
                    children=child_prices,
                    parent_close_before=before,
                    parent_close_on=on,
                    distribution_value=result.distribution_value,
                    factor=result.factor,
                    holder_return=result.holder_return,
                    status=result.status,
                )
            )
            if result.status == "price_consistent":
                spin_factors[security_id].append((ex_date, result.factor))

    for security_id, chain in chains.items():
        split_factor = seg.split_factors(chain, accepted.get(security_id, []))
        dist = adjustments.distribution_factors(split_factor, spin_factors.get(security_id, []))
        for bar in session.execute(
            select(SecurityBar).where(SecurityBar.security_id == security_id)
        ).scalars():
            bar.split_factor = split_factor[bar.session]
            bar.distribution_factor = dist[bar.session]
    session.flush()


# =============================================================================
# Types
# =============================================================================


def _wikidata_rows(session: Session, h: TickerHistory, shared: set[str]) -> list[dict]:
    """Wikidata rows for the period during which this security traded under ``h.symbol``."""
    return outcomes.wikidata_matches(
        _cached(session, "wikidata", h.symbol) or [],
        h.valid_from,
        h.valid_to,
        ticker_shared=h.symbol in shared,
    )


def _shared_tickers(session: Session) -> set[str]:
    owners = defaultdict(set)
    for h in session.execute(select(TickerHistory)).scalars():
        owners[h.symbol].add(h.security_id)
    return {symbol for symbol, ids in owners.items() if len(ids) > 1}


# =============================================================================
# SEC
# =============================================================================


def _sec_search_keys(
    session: Session, tickers: list[TickerHistory], shared: set[str]
) -> tuple[list[str], list[str]]:
    """(tickers newest first, company names) a security can be looked up by at SEC.

    Names come from Wikidata items listed under the security's tickers at the time,
    OpenFIGI records of its tickers and the batch's search names. They only propose
    candidate companies; ``sec.find`` accepts one only on a timely delisting notice.
    """
    names = set()
    for h in tickers:
        names |= {row["itemLabel"]["value"] for row in _wikidata_rows(session, h, shared)}
        names |= {
            r.get("name") for r in _cached(session, "openfigi", h.symbol) or [] if r.get("name")
        }
    extra = [n for h in tickers for n in sec.SEARCH_NAMES.get(h.symbol, ())]
    ordered = sorted(n for n in names if n.strip())
    return [h.symbol for h in reversed(tickers)], ordered + [n for n in extra if n not in names]


def sec_findings(session: Session, client: sec.CachedSec) -> dict[int, sec.SecFinding]:
    """What SEC says about every ended security: matched company, notices, 8-Ks."""
    history = defaultdict(list)
    for h in session.execute(select(TickerHistory).order_by(TickerHistory.valid_from)).scalars():
        history[h.security_id].append(h)
    shared = _shared_tickers(session)
    found = {}
    for security in session.execute(
        select(Security).where(Security.status == "ended").order_by(Security.security_id)
    ).scalars():
        tickers = history[security.security_id]
        symbols, names = _sec_search_keys(session, tickers, shared)
        text_ciks = []
        for h in reversed(tickers):
            found_text = sh.ticker_candidates(client, h.symbol, h.valid_from, h.valid_to)
            for cik in sh.ranked_candidates(found_text):
                # Only a company that names the ticker as its own is a candidate.
                if sh.confirm_writer(client, found_text[cik], h.symbol):
                    text_ciks.append(cik)
        found[security.security_id] = sec.find(
            client, symbols, names, security.last_trade_session, text_ciks
        )
    return found


def fetch_sec(
    session: Session, *, refresh: bool = False, sleep=time.sleep
) -> tuple[int, list[str]]:
    """Ask SEC about every ended security and cache the answers. Returns (requests, failures)."""
    client = sec.CachedSec(session, network=True, refresh=refresh, sleep=sleep)
    try:
        sec_findings(session, client)
    finally:
        client.close()
    return client.requests, client.failures


def cached_sec_findings(session: Session) -> dict[int, sec.SecFinding]:
    return sec_findings(session, sec.CachedSec(session, network=False))


def _sec_relevant(finding: sec.SecFinding, symbol: str, last: date | None = None):
    """(type, notices about that class) of a finding, for the security traded as ``symbol``.

    When the issuer delisted classes on different dates, only the notices filed closest
    to the security's last trade describe it (JMP Group's notes, delisted the day the
    JMPNL ticker last traded, rather than its common stock two months later).
    """
    notices = finding.notices
    if last is not None and notices:
        nearest = min(abs((f.filed - last).days) for f, _ in notices)
        notices = [(f, n) for f, n in notices if abs((f.filed - last).days) == nearest]
    kind = sec.notice_type([n.class_description for _, n in notices], symbol)
    relevant = [(f, n) for f, n in notices if sec.class_type(n.class_description, symbol) == kind]
    return kind, relevant


def _runs(points: "list[IssuerPoint]") -> "dict[tuple[str, str], list[IssuerPoint]]":
    runs: dict[tuple[str, str], list[IssuerPoint]] = defaultdict(list)
    for p in sorted(points, key=lambda p: p.filed):
        runs[(p.symbol, p.cik)].append(p)
    return dict(runs)


def resolve_types(session: Session) -> None:
    session.execute(delete(TypeEvidence))
    history = defaultdict(list)
    for h in session.execute(select(TickerHistory)).scalars():
        history[h.security_id].append(h)
    latest_holder: dict[str, tuple[date, int]] = {}
    for security_id, items in history.items():
        for h in items:
            if h.symbol not in latest_holder or h.valid_to > latest_holder[h.symbol][0]:
                latest_holder[h.symbol] = (h.valid_to, security_id)
    declared = {b.symbol: b.declared_type for b in session.execute(select(BatchSymbol)).scalars()}
    shared = _shared_tickers(session)
    sec_found = cached_sec_findings(session)
    facts = curated_facts(session)
    held = stored_issuer_points(session)
    client = sec.CachedSec(session, network=False)

    def note(security, source, symbol, kind, detail):
        session.add(
            TypeEvidence(
                security_id=security.security_id,
                source=source,
                symbol=symbol,
                security_type=kind,
                detail=detail[:500],
            )
        )

    for security in session.execute(select(Security)).scalars():
        votes: list[sectypes.TypeVote] = []
        tickers = history[security.security_id]
        wiki_rows = [(h, row) for h in tickers for row in _wikidata_rows(session, h, shared)]
        wiki_names = sorted({row["itemLabel"]["value"] for _, row in wiki_rows})
        cover = [p for p in held.get(security.security_id, []) if p.how == "cover_page"]
        # Names known for the security's own period: Wikidata's, and its issuers' on the
        # cover pages it was listed on.
        period_names = sorted(set(wiki_names) | {p.name for p in cover})
        # The issuer's own class title, read from the first and the last cover page of each
        # ticker and company (documents the issuer step confirmed).
        for (symbol, cik), run in _runs(cover).items():
            seen_titles = set()
            for p in (run[0], run[-1]):
                text = sh.plain_text(client.get(p.url))
                if not sh.cover_page_symbol(text, symbol):
                    continue
                title = sh.cover_page_class(text, symbol)
                kind = sectypes.cover_class_type(title)
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                if kind:
                    votes.append(sectypes.TypeVote("sec_cover", kind))
                note(
                    security,
                    "sec_cover" if kind else "sec_cover_unclear",
                    symbol,
                    kind or "unknown",
                    f"{p.filed} cover page of CIK {cik} ({p.name}): class '{title}' [{p.url}]"
                    + ("" if kind else "; not a recognised class title, not counted"),
                )
        for h, row in wiki_rows:
            labels = row.get("types", {}).get("value", "").split("|")
            kind = sectypes.wikidata_type(labels)
            if kind:
                votes.append(sectypes.TypeVote("wikidata", kind))
                item = row["item"]["value"].rsplit("/", 1)[-1]
                note(
                    security,
                    "wikidata",
                    h.symbol,
                    kind,
                    f"{row['itemLabel']['value']} ({item}): {', '.join(labels)[:120]}",
                )
        for h in tickers:
            if latest_holder.get(h.symbol, (None, None))[1] == security.security_id:
                for record in _cached(session, "openfigi", h.symbol) or []:
                    kind = sectypes.figi_type(
                        record.get("securityType"), record.get("securityType2")
                    )
                    name = record.get("name") or ""
                    detail = f"{name} [{record.get('securityType')}/{record.get('securityType2')}]"
                    # OpenFIGI names the ticker's holder today. A ticker this security no
                    # longer used at the end of 2024 (it stopped trading, or moved on to a
                    # new ticker like FB -> META) may have passed to another company since,
                    # so the answer is used only if its name agrees with the company
                    # Wikidata places under the ticker at the time.
                    # OpenFIGI answered after the data ends (the ticker may have passed to
                    # another company or fund since, LAZR to an ETF), so the answer is used
                    # only if its name agrees with a company known for the security's own
                    # period (Wikidata, its issuers' cover pages), when any is known.
                    if period_names and not any(
                        sectypes.names_agree(name, other) for other in period_names
                    ):
                        note(
                            security,
                            "openfigi_rejected",
                            h.symbol,
                            kind,
                            f"{detail}: names a different company than "
                            f"{'/'.join(period_names)[:200]}; not used",
                        )
                        continue
                    votes.append(sectypes.TypeVote("openfigi", kind))
                    note(security, "openfigi", h.symbol, kind, detail)
            implied = sectypes.suffix_type(h.symbol)
            if implied:
                votes.append(sectypes.TypeVote("symbol_suffix", implied))
                note(security, "symbol_suffix", h.symbol, implied, f"fifth letter {h.symbol[-1]}")
            if declared.get(h.symbol, "unknown") != "unknown":
                note(
                    security,
                    "declared",
                    h.symbol,
                    declared[h.symbol],
                    "declared in the batch definition (not counted)",
                )
        finding = sec_found.get(security.security_id)
        if finding is not None and finding.notices:
            kind, relevant = _sec_relevant(
                finding, security.last_symbol, security.last_trade_session
            )
            issuer = f"{finding.company}, CIK {finding.cik}, SIC {finding.sic or 'none'}" + (
                " (blank check)" if finding.sic == "6770" else ""
            )
            # A closed-end fund's shares are "Common Shares" in its notice (GMZ): the class
            # name alone does not make a fund's shares common stock.
            fund_issuer = kind == "common" and sec.looks_like_fund(finding.company or "")
            if kind and not fund_issuer:
                votes.append(sectypes.TypeVote("sec_form25", kind))
            for filing, notice in finding.notices:
                used = (filing, notice) in relevant and kind is not None and not fund_issuer
                note(
                    security,
                    "sec_form25" if used else "sec_form25_other",
                    security.last_symbol,
                    sec.class_type(notice.class_description, security.last_symbol) or "unknown",
                    f"{filing.form} {filing.filed}: class '{notice.class_description}' ({issuer})"
                    + (
                        ""
                        if used
                        else "; the issuer is a fund, not counted"
                        if fund_issuer
                        else "; another class or unclear, not counted"
                    ),
                )
        for fact, missing in facts.get(security.security_id, []):
            if fact.security_type is None:
                continue
            if not missing:
                votes.append(sectypes.TypeVote("sec_text", fact.security_type))
            note(
                security,
                "sec_text" if not missing else "sec_text_unverified",
                security.last_symbol,
                fact.security_type,
                f"{fact.note} [{fact.quotes[0].url}]"
                + ("" if not missing else f"; {len(missing)} quote(s) not found, not counted"),
            )
        security.security_type, security.type_status = sectypes.resolve_type(votes)
    session.flush()


# =============================================================================
# Outcomes
# =============================================================================


def resolve_outcomes(session: Session, calendar: list[date]) -> None:
    session.execute(delete(ExitEvidence))
    events = list(session.execute(select(CorporateAction)).scalars())
    history = defaultdict(list)
    for h in session.execute(select(TickerHistory).order_by(TickerHistory.valid_from)).scalars():
        history[h.security_id].append(h)
    shared = _shared_tickers(session)
    segments_by_symbol = defaultdict(list)
    for security_id, items in history.items():
        for h in items:
            segments_by_symbol[h.symbol].append((h.valid_from, h.valid_to, security_id))

    sec_found = cached_sec_findings(session)
    facts = curated_facts(session)
    deals = {
        row.security_id: row
        for row in session.execute(
            select(DealValue).where(DealValue.status == "verified")
        ).scalars()
    }
    closes = {
        (b.security_id, b.session): b.close for b in session.execute(select(SecurityBar)).scalars()
    }

    def sessions_after(day: date, count: int) -> date:
        later = [s for s in calendar if s > day]
        return later[min(count, len(later)) - 1] if later else day

    for security in session.execute(select(Security)).scalars():
        if security.status != "ended":
            security.outcome = security.outcome_status = None
            continue
        last, symbol = security.last_trade_session, security.last_symbol
        found: list[outcomes.Evidence] = []
        # Alpaca: mergers naming the last ticker, unless another listing of that
        # ticker trades in between.
        horizon = sessions_after(last, MERGER_WINDOW_SESSIONS)
        for e in events:
            if (
                e.acquiree_symbol != symbol
                or not e.process_date
                or not (last <= e.process_date <= horizon)
            ):
                continue
            if any(
                start > last and start <= e.process_date and sid != security.security_id
                for start, _, sid in segments_by_symbol[symbol]
            ):
                continue
            kind = {
                "cash_merger": "acquired_cash",
                "stock_merger": "acquired_stock",
                "stock_and_cash_merger": "acquired_stock_and_cash",
            }.get(e.action_type)
            if kind:
                found.append(
                    outcomes.Evidence(
                        "alpaca",
                        kind,
                        e.process_date,
                        f"{e.action_type} of {symbol}"
                        + (f" into {e.acquirer_symbol}" if e.acquirer_symbol else "")
                        + (f", {e.rate} cash per share" if e.rate else "")
                        + (f", {e.cash_rate} cash plus shares" if e.cash_rate else ""),
                        e.id,
                        e.rate if kind == "acquired_cash" else None,
                    )
                )
            # Alpaca: a rename to a Nasdaq "Q" ticker (bankruptcy) shortly after the last trade.
        for e in events:
            if (
                e.action_type == "name_change"
                and e.old_symbol == symbol
                and e.new_symbol
                and e.new_symbol.endswith("Q")
                and len(e.new_symbol) >= 4
                and e.process_date
                and last <= e.process_date <= sessions_after(last, 15)
            ):
                found.append(
                    outcomes.Evidence(
                        "alpaca",
                        "bankruptcy",
                        e.process_date,
                        f"renamed {symbol} -> {e.new_symbol} (Q = bankruptcy)",
                        e.id,
                    )
                )
        # Wikidata, for every ticker of the security.
        for h in history[security.security_id]:
            rows = _wikidata_rows(session, h, shared)
            found += outcomes.wikidata_evidence(rows, last)
        # SEC: the delisting notice for this class and 8-Ks on bankruptcy or a closed merger.
        finding = sec_found.get(security.security_id)
        if finding is not None and finding.cik:
            found += sec_evidence(finding, security.last_symbol, security.last_trade_session)
        # Facts read from SEC filings by a person, applied only if every quote is found.
        for fact, missing in facts.get(security.security_id, []):
            if fact.category and not missing:
                found.append(
                    outcomes.Evidence(
                        "sec",
                        fact.category,
                        fact.event_date,
                        f'{fact.note} Quoted: "{fact.quotes[-1].text[:120]}"',
                        fact.quotes[-1].url[-128:],
                        fact.cash_per_share,
                    )
                )
        if security.security_id in deals:
            found.append(deal_evidence(deals[security.security_id]))
        # The same ticker trading again later, and very short runs.
        later = sorted(
            (start, sid)
            for start, _, sid in segments_by_symbol[symbol]
            if sid != security.security_id and start > last
        )
        if later:
            found.append(
                outcomes.Evidence(
                    "alpaca_bars",
                    "resumed",
                    None,
                    f"{symbol} trades again later (halt, or the ticker was reused)",
                )
            )
            start, sid = later[0]
            gap = seg.sessions_between(last, start)
            before, after = closes.get((security.security_id, last)), closes.get((sid, start))
            ratio = gap_split_ratio(events, symbol, last, start)
            if before:
                before = before / ratio
            low, high = TRADING_GAP_PRICE_RATIO
            no_notice = finding is None or not finding.notices
            if (
                no_notice
                and before
                and after
                and gap <= TRADING_GAP_MAX_SESSIONS
                and low <= after / before <= high
            ):
                found.append(
                    outcomes.Evidence(
                        "alpaca_bars",
                        "trading_gap",
                        start,
                        f"{symbol} trades again from {start} after {gap} sessions without a "
                        f"trade, at {after} against {before:g} before the gap"
                        + (f" (after splits x{ratio:g} in the gap)" if ratio != 1.0 else "")
                        + "; no delisting notice in between",
                    )
                )
        if security.real_trade_sessions < FRAGMENT_SESSIONS:
            found.append(
                outcomes.Evidence(
                    "alpaca_bars",
                    "fragment",
                    None,
                    f"only {security.real_trade_sessions} real trading sessions",
                )
            )
        result = outcomes.resolve_outcome(found)
        for e in found:
            session.add(
                ExitEvidence(
                    security_id=security.security_id,
                    source=e.source,
                    category=e.category,
                    event_date=e.event_date,
                    detail=e.detail[:500],
                    reference=e.reference,
                )
            )
        security.outcome, security.outcome_status = result.outcome, result.status
        security.assumed_exit_value, security.exit_assumption = result.assumed_value, result.note
        security.value_basis = result.value_basis
    session.flush()


def sec_evidence(
    finding: sec.SecFinding, symbol: str, last: date | None = None
) -> list[outcomes.Evidence]:
    """Exit evidence from a matched SEC company (all filings are before 2025 by construction)."""
    found = []
    _, relevant = _sec_relevant(finding, symbol, last)
    for filing, notice in relevant:
        category, meaning = sec.rule_meaning(notice.rule)
        reason = f"; notice text: {notice.reason_text[:160]}" if notice.reason_text else ""
        found.append(
            outcomes.Evidence(
                "sec",
                category,
                filing.filed,
                f"{filing.form} filed {filing.filed} by {notice.exchange or 'unknown'}: "
                f"class '{notice.class_description}', {meaning}{reason}",
                filing.href[-128:],
            )
        )
    acquisition = next(
        (k for k in finding.eight_ks if sec.eight_k_category(k.items) == "acquired"), None
    )
    # A class that was redeemed was paid off: later events of the company (JMP Group's
    # sale two months after its notes were redeemed) are not about it.
    redeemed_on = [f.filed for f, n in relevant if sec.rule_meaning(n.rule)[0] == "redeemed"]
    eight_ks = [
        k
        for k in finding.eight_ks
        if not redeemed_on or k.filed <= max(redeemed_on) + timedelta(days=5)
    ]
    for k in eight_ks:
        kind = sec.eight_k_category(k.items)
        cash, sentence = finding.cash if k is acquisition else (None, "")
        found.append(
            outcomes.Evidence(
                "sec",
                "acquired_cash" if cash is not None else kind,
                k.filed,
                f"8-K filed {k.filed} ({finding.company}), {k.items}"
                + (
                    f"; cash per share {cash:g}; terms: {sentence[:180]}"
                    if cash is not None
                    else ""
                ),
                k.href[-128:],
                cash,
            )
        )
    return found


def plain_tickers(session: Session) -> list[str]:
    return sorted(
        {
            h.symbol
            for h in session.execute(select(TickerHistory)).scalars()
            if PLAIN_TICKER.match(h.symbol)
        }
    )


# =============================================================================
# Curated SEC facts (renames, types, exits, deal terms), re-verified every run
# =============================================================================

#: A deal's value must lie within this fraction of the target's last close.
DEAL_TOLERANCE = 0.10
#: An instrument that trades again within this many sessions at a similar price,
#: with no delisting notice in between, had a trading gap rather than an exit.
TRADING_GAP_MAX_SESSIONS = 250
TRADING_GAP_PRICE_RATIO = (0.5, 2.0)


def missing_quotes(client: sec.CachedSec, quotes) -> list:
    """Quotes not found word for word in their (cached) SEC document."""
    missing = []
    for quote in quotes:
        text = sh.plain_text(client.get(quote.url))
        if " ".join(quote.text.split()) not in text:
            missing.append(quote)
    return missing


def fetch_curated_documents(
    session: Session, *, refresh: bool = False, sleep=time.sleep
) -> tuple[int, list[str]]:
    """Fetch (or reuse from the cache) every SEC document a curated entry quotes."""
    client = sec.CachedSec(session, network=True, refresh=refresh, sleep=sleep)
    try:
        for entry in (*CURATED_RENAMES, *CURATED_IDENTITY_LINKS, *CURATED_FACTS, *CURATED_DEALS):
            for quote in entry.quotes:
                client.get(quote.url)
    finally:
        client.close()
    return client.requests, client.failures


def apply_curated_renames(session: Session) -> list[tuple[CuratedRename, list]]:
    """Add each verified curated rename as a name_change event. Returns (rename, missing quotes)."""
    client = sec.CachedSec(session, network=False)
    results = []
    for rename in CURATED_RENAMES:
        missing = missing_quotes(client, rename.quotes)
        results.append((rename, missing))
        if missing:
            continue
        session.merge(
            CorporateAction(
                id=f"curated:{rename.old_symbol}>{rename.new_symbol}",
                action_type="name_change",
                process_date=rename.date,
                old_symbol=rename.old_symbol,
                new_symbol=rename.new_symbol,
                raw={
                    "source": "curated from SEC filings",
                    "note": rename.note,
                    "documents": sorted({q.url for q in rename.quotes}),
                },
            )
        )
    session.flush()
    return results


def curated_identity_links(session: Session) -> list[tuple[CuratedIdentityLink, list]]:
    """Each documented identity exception with its missing quotes (cache only)."""
    client = sec.CachedSec(session, network=False)
    return [(link, missing_quotes(client, link.quotes)) for link in CURATED_IDENTITY_LINKS]


def identity_link_id(link: CuratedIdentityLink) -> str:
    return f"identity:{link.old_symbol}>{link.new_symbol}:{link.old_last}:{link.new_first}"


def curated_facts(session: Session) -> dict[int, list[tuple[CuratedFact, list]]]:
    """Curated facts per security (matched by last ticker and last trade), with missing quotes."""
    client = sec.CachedSec(session, network=False)
    found: dict[int, list[tuple[CuratedFact, list]]] = defaultdict(list)
    for security in session.execute(select(Security).order_by(Security.security_id)).scalars():
        for fact in CURATED_FACTS:
            if (
                fact.symbol == security.last_symbol
                and fact.last_trade == security.last_trade_session
            ):
                found[security.security_id].append((fact, missing_quotes(client, fact.quotes)))
    return dict(found)


def _acquirer_bars(
    session: Session, client: AlpacaClient, utc_window, symbol: str, effective: date
) -> list[seg.Bar]:
    """Raw SIP daily bars of the acquirer's literal ticker for ten days from ``effective``."""
    key = f"{symbol}:{effective.isoformat()}"
    cached = _cached(session, "alpaca_deal", key)
    if cached is None:
        last = min(effective + timedelta(days=10), date(2024, 12, 31))
        start, end = utc_window(effective, last)
        items = client.fetch_bars([symbol], timeframe="1Day", start=start, end=end, asof="-").get(
            symbol, []
        )
        _store(session, "alpaca_deal", key, items)
        cached = items
    from app.universe.pipeline import _to_bar  # local import: pipeline imports this module

    return sorted((_to_bar(item) for item in cached), key=lambda b: b.session)


def value_deals(session: Session, client: AlpacaClient, utc_window) -> list[DealValue]:
    """Value per share of each curated stock or cash-and-stock deal, with its checks.

    Value = cash + shares x the acquirer's raw close on the effective date (or the first
    session after it with a real trade). The terms must be quoted verbatim from the SEC
    filing, and the value must lie within ``DEAL_TOLERANCE`` of the target's last close
    (merger arbitrage keeps them close). Otherwise the value is recorded but not used.
    """
    session.execute(delete(DealValue))
    secc = sec.CachedSec(session, network=False)
    securities = {
        (x.last_symbol, x.last_trade_session): x
        for x in session.execute(select(Security).where(Security.status == "ended")).scalars()
    }
    rows = []
    for deal in CURATED_DEALS:
        security = securities.get((deal.target_symbol, deal.last_trade))
        if security is None:
            continue
        missing = missing_quotes(secc, deal.quotes)
        bars = _acquirer_bars(
            session, client, utc_window, deal.acquirer_symbol, deal.effective_date
        )
        first = next(
            (b for b in bars if b.session >= deal.effective_date and not b.is_filler), None
        )
        last_close = session.scalar(
            select(SecurityBar.close).where(
                SecurityBar.security_id == security.security_id,
                SecurityBar.session == security.last_trade_session,
            )
        )
        value = deal.cash_per_share + deal.shares_per_share * first.close if first else None
        if missing:
            status = "quote_not_found"
        elif value is None:
            status = "no_acquirer_price"
        elif last_close and abs(value / last_close - 1) > DEAL_TOLERANCE:
            status = "value_far_from_last_close"
        else:
            status = "verified"
        row = DealValue(
            security_id=security.security_id,
            acquirer_symbol=deal.acquirer_symbol,
            effective_date=deal.effective_date,
            cash_per_share=deal.cash_per_share,
            shares_per_share=deal.shares_per_share,
            acquirer_session=first.session if first else None,
            acquirer_close=first.close if first else None,
            value=value,
            target_last_close=last_close,
            status=status,
            source_url=deal.quotes[0].url,
            note=deal.note,
        )
        session.add(row)
        rows.append(row)
    session.flush()
    return rows


def deal_evidence(row: DealValue) -> outcomes.Evidence:
    if row.shares_per_share == 0:
        kind = "acquired_cash"
    elif row.cash_per_share > 0:
        kind = "acquired_stock_and_cash"
    else:
        kind = "acquired_stock"
    return outcomes.Evidence(
        "sec",
        "deal_terms",
        row.effective_date,
        f"{row.cash_per_share:g} cash + {row.shares_per_share:g} x {row.acquirer_symbol} "
        f"close {row.acquirer_close} on {row.acquirer_session} = {row.value:.4f} "
        f"(target last close {row.target_last_close}); {row.note}",
        row.source_url[-128:],
        row.value,
        kind,
    )


# =============================================================================
# Issuers and their SIC code over time (blank-check companies)
# =============================================================================

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _ticker_today(client: sec.CachedSec) -> dict[str, tuple[str, str]]:
    import json

    text = client.get(COMPANY_TICKERS_URL)
    try:
        rows = json.loads(text or "{}").values()
    except ValueError:
        return {}
    return {r["ticker"]: (str(r["cik_str"]).zfill(10), r.get("title", "")) for r in rows}


@dataclass(frozen=True)
class IssuerPoint:
    """A dated fact that company ``cik`` held ``symbol`` (see ``models.IssuerEvidence``)."""

    filed: date
    cik: str
    name: str
    how: str
    url: str
    symbol: str


#: Order of evidence kinds on the same day (the delisting notice at the last trade first).
_HOW_ORDER = {"form25": 0, "cover_page": 1, "full_text": 2}


def issuer_on(points: Sequence[IssuerPoint], day: date) -> tuple[IssuerPoint | None, bool]:
    """The issuer on ``day``: the latest evidence on or before it, and False.

    Before the first evidence, the earliest evidence and True (the identity then rests on
    a document filed later; the SIC used is still only what was on file by ``day``).
    """
    if not points:
        return None, False
    ordered = sorted(points, key=lambda p: (p.filed, -_HOW_ORDER.get(p.how, 9)))
    earlier = [p for p in ordered if p.filed <= day]
    if earlier:
        return earlier[-1], False
    return ordered[0], True


#: Candidates per ticker period proposed by a search that is not limited to one company.
OPEN_CANDIDATES = 3
#: Documents read (earliest first, then latest first) to confirm a company's cover pages.
COVER_CONFIRM_TRIES = 3


def _cover_confirmed(client: sec.CachedSec, hit, ticker: str) -> bool:
    return bool(sh.cover_page_symbol(sh.plain_text(client.get(hit.url)), ticker))


def cover_page_points(
    client: sec.CachedSec, ticker: str, first: date, last: date, cik: str
) -> list[IssuerPoint]:
    """Company ``cik``'s cover-page evidence for ``ticker`` between ``first`` and ``last``.

    Its filings found by the search, from the earliest whose document confirms the ticker
    under "Trading Symbol(s)" (of the first few) to the latest that does, found by halving
    (a company holds a ticker for one stretch of time: IAC/InterActiveCorp's filings list
    IAC until mid-2020 and MTCH after). None if no document confirms it.
    """
    # The cover table is in a filing's main document, not in its exhibits.
    mains = [
        h
        for h in sh.cover_page_hits(client, ticker, first, last, cik)
        if not h.file_type or h.file_type.upper() == h.form.upper()
    ]
    hits = sorted({(h.filed, h.adsh): h for h in mains}.values(), key=lambda h: (h.filed, h.adsh))
    head = next(
        (
            i
            for i, h in enumerate(hits[:COVER_CONFIRM_TRIES])
            if _cover_confirmed(client, h, ticker)
        ),
        None,
    )
    if head is None:
        return []

    def holds(i: int) -> bool:
        # One document may lack the table (an exhibit); its neighbour decides with it.
        return _cover_confirmed(client, hits[i], ticker) or (
            i + 1 < len(hits) and _cover_confirmed(client, hits[i + 1], ticker)
        )

    lo, hi = head, len(hits) - 1
    if _cover_confirmed(client, hits[hi], ticker):
        lo = hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if holds(mid):
            lo = mid
        else:
            hi = mid
    return [
        IssuerPoint(h.filed, h.cik, h.name, "cover_page", h.url, ticker)
        for h in hits[head : lo + 1]
    ]


def _overlap(a: list[IssuerPoint], b: list[IssuerPoint]) -> bool:
    return a[0].filed <= b[-1].filed and b[0].filed <= a[-1].filed


def issuer_points(
    session: Session,
    client: sec.CachedSec,
    sec_found: dict[int, sec.SecFinding],
    only: set[int] | None = None,
) -> tuple[dict[int, list[IssuerPoint]], dict[int, list[str]]]:
    """Dated issuer evidence for every security, and the ticker periods where two companies
    both claimed the ticker at the same time (resolved here, reported).

    Per ticker period, from 2019 on: each candidate company's own cover pages listing the
    ticker. Candidates come from the delisting notice, SEC's ticker list as of today, a
    cover-page search not limited to one company, and the companies found for the
    security's other tickers; each is accepted only on its own cover pages, so today's
    ticker list can propose a company but never decide. When two companies' cover pages
    overlap in time (a parent and a subsidiary filing together), the one with more of
    them is kept. A ticker period with no cover-page evidence (before 2019, or a filer
    without the table) falls back to a filing in which the company names the ticker as its
    own ("full_text"). The delisting notice at the last trade is evidence too ("form25").
    """
    today = _ticker_today(client)
    history = defaultdict(list)
    for h in session.execute(select(TickerHistory).order_by(TickerHistory.valid_from)).scalars():
        history[h.security_id].append(h)
    shared = _shared_tickers(session)
    known_names = {
        sid: {
            row["itemLabel"]["value"] for h in items for row in _wikidata_rows(session, h, shared)
        }
        for sid, items in history.items()
    }
    out: dict[int, list[IssuerPoint]] = {}
    conflicts: dict[int, list[str]] = defaultdict(list)
    for security in session.execute(select(Security).order_by(Security.security_id)).scalars():
        sid = security.security_id
        if only is not None and sid not in only:
            continue
        tickers = history[sid]
        points: list[IssuerPoint] = []
        finding = sec_found.get(sid)
        candidates: list[str] = []
        if finding is not None and finding.cik:
            cik = str(int(finding.cik)).zfill(10)
            candidates.append(cik)
            notice = finding.notices[-1][0] if finding.notices else None
            points.append(
                IssuerPoint(
                    security.last_trade_session,
                    cik,
                    finding.company or "",
                    "form25",
                    notice.href if notice is not None else "",
                    tickers[-1].symbol if tickers else security.last_symbol,
                )
            )
        for h in tickers:
            if h.symbol in today:
                candidates.append(today[h.symbol][0])
        open_hits = {
            h.symbol: sh.cover_page_hits(client, h.symbol, h.valid_from, h.valid_to)
            for h in tickers
        }
        for h in tickers:
            counts = Counter(hit.cik for hit in open_hits[h.symbol])
            candidates += [cik for cik, _ in counts.most_common(OPEN_CANDIDATES)]
        found_by_period: dict[int, list[list[IssuerPoint]]] = {}
        # Two passes: a company confirmed for one ticker is a candidate for the others
        # (a SPAC and the business it became file under one CIK).
        for _pass in range(2):
            ordered = list(dict.fromkeys(candidates))
            for index, h in enumerate(tickers):
                spans = [
                    span
                    for cik in ordered
                    if (span := cover_page_points(client, h.symbol, h.valid_from, h.valid_to, cik))
                ]
                found_by_period[index] = spans
            confirmed = {span[0].cik for spans in found_by_period.values() for span in spans}
            if confirmed <= set(candidates):
                break
            candidates += sorted(confirmed)
        for index, h in enumerate(tickers):
            spans = sorted(found_by_period.get(index, []), key=lambda s: (-len(s), s[0].cik))
            kept: list[list[IssuerPoint]] = []
            for span in spans:
                clash = next((k for k in kept if _overlap(k, span)), None)
                if clash is None:
                    kept.append(span)
                else:
                    conflicts[sid].append(
                        f"{h.symbol}: CIK {span[0].cik} ({len(span)} cover pages "
                        f"{span[0].filed}..{span[-1].filed}) overlaps CIK {clash[0].cik} "
                        f"({len(clash)}); kept {clash[0].cik}"
                    )
            found_text = None
            if not kept and h.valid_to >= sh.COVER_FIRST:
                # No candidate had cover pages: the companies that wrote the ticker in an
                # exchange context ("NASDAQ: YY") are tried too.
                found_text = sh.ticker_candidates(client, h.symbol, h.valid_from, h.valid_to)
                for cik in sh.ranked_candidates(found_text)[:OPEN_CANDIDATES]:
                    if cik in ordered:
                        continue
                    span = cover_page_points(client, h.symbol, h.valid_from, h.valid_to, cik)
                    if span:
                        kept = [span]
                        break
            for span in kept:
                points += span
            if kept:
                continue
            # No cover-page evidence for this ticker period: a filing naming it as its own.
            known = known_names.get(sid, set())
            if found_text is None:
                found_text = sh.ticker_candidates(client, h.symbol, h.valid_from, h.valid_to)
            for cik in sh.ranked_candidates(found_text):
                confirmed_hit = sh.confirm_writer(
                    client, found_text[cik], h.symbol, window=sh.ISSUER_WINDOW
                )
                if not confirmed_hit:
                    continue
                hit = confirmed_hit[0]
                if known:
                    company = sh.submissions(client, cik)
                    names = {hit.name} | {f.name for f in (company.former_names if company else ())}
                    if not any(sectypes.names_agree(a, b) for a in names for b in known):
                        continue
                if (
                    not h.valid_from - timedelta(days=30)
                    <= hit.filed
                    <= h.valid_to + timedelta(days=30)
                ):
                    continue
                points.append(
                    IssuerPoint(hit.filed, hit.cik, hit.name, "full_text", hit.url, h.symbol)
                )
                break
        if points:
            out[sid] = sorted(points, key=lambda p: (p.filed, _HOW_ORDER.get(p.how, 9), p.cik))
    return out, dict(conflicts)


def issuer_records(
    session: Session,
    client: sec.CachedSec,
    sec_found: dict[int, sec.SecFinding],
    *,
    commit_each: bool = False,
):
    """Per security with evidence: (id, issuer at the last trade, its evidence kind, the
    dated evidence, SIC points per company). SIC is read point in time from each company's
    filing headers, from a year before the first trade to the last."""
    points_by_sid, conflicts = issuer_points(session, client, sec_found)
    if commit_each:
        session.commit()
    out = []
    for security in session.execute(select(Security).order_by(Security.security_id)).scalars():
        points = points_by_sid.get(security.security_id)
        if not points:
            continue
        last, _ = issuer_on(points, security.last_trade_session)
        sics = {
            cik: sh.sic_history(
                client,
                cik,
                security.first_session - timedelta(days=365),
                security.last_trade_session,
            )
            for cik in sorted({p.cik for p in points})
        }
        out.append((security.security_id, last, points, sics))
        if commit_each:
            session.commit()  # keep fetched answers even if a later security fails
    return out, conflicts


def fetch_issuers(
    session: Session, *, refresh: bool = False, sleep=time.sleep
) -> tuple[int, list[str]]:
    client = sec.CachedSec(session, network=True, refresh=refresh, sleep=sleep)
    try:
        found = sec_findings(session, client)
        session.commit()
        issuer_records(session, client, found, commit_each=True)
    finally:
        client.close()
    return client.requests, client.failures


def store_issuers(session: Session) -> dict[int, list[str]]:
    """Store the dated issuer evidence, the issuer at each last trade and the SIC points.
    Returns the ticker periods where two companies' cover pages overlapped."""
    session.execute(delete(IssuerSic))
    session.execute(delete(IssuerEvidence))
    session.execute(delete(SecurityIssuer))
    client = sec.CachedSec(session, network=False)
    records, conflicts = issuer_records(session, client, sec_findings(session, client))
    for sid, last, points, sics in records:
        session.add(
            SecurityIssuer(security_id=sid, cik=last.cik, name=last.name[:160], how=last.how)
        )
        for p in points:
            session.add(
                IssuerEvidence(
                    security_id=sid,
                    symbol=p.symbol,
                    filed=p.filed,
                    cik=p.cik,
                    name=p.name[:160],
                    how=p.how,
                    url=p.url[:300],
                )
            )
        for cik, changes in sics.items():
            for as_of, sic in changes:
                session.add(IssuerSic(security_id=sid, cik=cik, as_of=as_of, sic=sic))
    session.flush()
    return conflicts


def blank_check_on(changes: list[tuple[date, str | None]], renames: list[date], day: date) -> bool:
    """Whether the issuer counts as a blank-check company on ``day``.

    It does while the SIC on file is 6770, and stops at the security's first ticker change
    after its first 6770 filing (the de-SPAC: BOWX -> WeWork), because SEC's SIC can trail
    the merger by months (WeWork kept 6770 until August 2022). Only filings made and ticker
    changes seen on or before ``day`` are used: a rename still to come is never known. So a
    SPAC whose SIC changed before the merger closed (CF Finance Acquisition II: 3211 from
    January 2021, merger in March) counts as operating from the SIC change, as SEC's
    record then said.
    """
    first = next(
        (d for d, sic in sorted(changes, key=lambda c: c[0]) if sic == sh.BLANK_CHECK_SIC), None
    )
    if first is None or first > day:
        return False
    if any(first <= r <= day for r in renames):
        return False
    return sh.sic_on(changes, day) == sh.BLANK_CHECK_SIC


def stored_issuer_points(session: Session) -> dict[int, list[IssuerPoint]]:
    out: dict[int, list[IssuerPoint]] = defaultdict(list)
    for e in session.execute(select(IssuerEvidence).order_by(IssuerEvidence.id)).scalars():
        out[e.security_id].append(IssuerPoint(e.filed, e.cik, e.name, e.how, e.url, e.symbol))
    return dict(out)


def blank_check_exclusions(session: Session):
    """Per security, a test of whether its issuer was a blank-check company on a day.

    The issuer on the day is the company of the latest dated evidence by then
    (``issuer_on``); its SIC is read only from filings made by then."""
    sics: dict[tuple[int, str], list[tuple[date, str | None]]] = defaultdict(list)
    for row in session.execute(select(IssuerSic)).scalars():
        sics[(row.security_id, row.cik)].append((row.as_of, row.sic))
    renames: dict[int, list[date]] = defaultdict(list)
    for h in session.execute(select(TickerHistory).order_by(TickerHistory.valid_from)).scalars():
        renames[h.security_id].append(h.valid_from)
    for sid in renames:
        renames[sid] = renames[sid][1:]  # the first segment's start is not a change

    def test(sid, points):
        def on(day):
            issuer, _ = issuer_on(points, day)
            changes = sorted(sics.get((sid, issuer.cik), []))
            return (
                "blank_check_company"
                if blank_check_on(changes, renames.get(sid, []), day)
                else None
            )

        return on

    return {sid: test(sid, points) for sid, points in stored_issuer_points(session).items()}


# =============================================================================
# Trading gaps: one security, not several
# =============================================================================


def gap_split_ratio(events, symbol: str, last: date, start: date) -> float:
    """Product of split ratios recorded between ``last`` and ``start`` for the ticker.

    Includes splits recorded under a temporary ticker it was renamed to in between
    (Scorpius: SCPX -> SCPXD, 1-for-200, back to SCPX).
    """
    interim = {symbol} | {
        e.new_symbol
        for e in events
        if e.action_type == "name_change"
        and e.old_symbol == symbol
        and e.process_date
        and last < e.process_date <= start
    }
    ratio = 1.0
    for e in events:
        when = e.ex_date or e.process_date
        if (
            e.action_type in ("forward_split", "reverse_split", "unit_split")
            and e.symbol in interim
            and when
            and last < when <= start
            and e.new_rate
            and e.old_rate
        ):
            ratio *= e.new_rate / e.old_rate
    return ratio


EXIT_CATEGORIES = outcomes.ACQUISITION | outcomes.FAILURE | outcomes.REDEMPTION


def decide_gap_links(session: Session) -> list[SegmentLink]:
    """Decide, with reasons, which same-ticker segments are one security across a gap.

    For each security whose last segment is followed later by another segment of the
    same ticker that starts a different security, the two are the same security only if
    all of these hold:

    1. the gap is at most ``TRADING_GAP_MAX_SESSIONS`` sessions;
    2. no delisting notice for the earlier security was filed before trading resumed;
    3. no exit is recorded for it before trading resumed (an Alpaca merger, a bankruptcy
       or merger 8-K, a curated SEC fact such as a fund liquidation);
    4. the price resumes within ``TRADING_GAP_PRICE_RATIO`` of where it stopped, after
       any split recorded in the gap, or SEC filings quoted in a curated fact say the
       same shares resumed (Scorpius);
    5. when both issuers are known, they are the same company (each the issuer of the day:
       the earlier security's at its last trade, the later one's at its first).

    Merges decided in an earlier pass are kept (their segments are now one security and
    are no longer candidates); rejected candidates are decided afresh. Returns the new
    rows: merged or kept separate, each with its reason.
    """
    session.execute(delete(SegmentLink).where(SegmentLink.decision == "kept_separate"))
    history: dict[int, list[TickerHistory]] = defaultdict(list)
    for h in session.execute(select(TickerHistory).order_by(TickerHistory.valid_from)).scalars():
        history[h.security_id].append(h)
    events = list(session.execute(select(CorporateAction)).scalars())
    closes = {
        (b.security_id, b.session): b.close for b in session.execute(select(SecurityBar)).scalars()
    }
    held = stored_issuer_points(session)

    def issuer_at(sid: int, day: date) -> str | None:
        found, _ = issuer_on(held.get(sid, []), day)
        return found.cik if found else None

    sec_found = cached_sec_findings(session)
    facts = curated_facts(session)
    by_symbol: dict[str, list[tuple[date, int, bool]]] = defaultdict(list)
    for sid, items in history.items():
        for index, h in enumerate(items):
            by_symbol[h.symbol].append((h.valid_from, sid, index == 0))
    rows = []
    for sid, items in sorted(history.items()):
        tail = items[-1]
        symbol, last = tail.symbol, tail.valid_to
        later = sorted(
            (start, other, head) for start, other, head in by_symbol[symbol] if start > last
        )
        if not later:
            continue
        start, other, head = later[0]
        gap = seg.sessions_between(last, start)
        ratio = gap_split_ratio(events, symbol, last, start)
        before = closes.get((sid, last))
        after = closes.get((other, start))
        reasons = []
        if not head:
            reasons.append("the later segment continues another security (already linked)")
        if gap > TRADING_GAP_MAX_SESSIONS:
            reasons.append(f"gap of {gap} sessions is longer than {TRADING_GAP_MAX_SESSIONS}")
        finding = sec_found.get(sid)
        if finding is not None:
            for filing, _notice in finding.notices:
                if filing.filed < start:
                    reasons.append(f"{filing.form} delisting notice filed {filing.filed}")
            for k in finding.eight_ks:
                kind = sec.eight_k_category(k.items)
                if kind and last - timedelta(days=5) <= k.filed < start:
                    reasons.append(f"8-K ({kind}) filed {k.filed}")
        for e in events:
            if (
                e.acquiree_symbol == symbol
                and e.action_type in ("cash_merger", "stock_merger", "stock_and_cash_merger")
                and e.process_date
                and last <= e.process_date < start
            ):
                reasons.append(f"Alpaca {e.action_type} {e.process_date}")
        same_by_filing = False
        for fact, missing in facts.get(sid, []):
            if missing:
                continue
            if fact.category == "trading_gap":
                same_by_filing = True
            elif fact.category in EXIT_CATEGORIES and fact.event_date and fact.event_date < start:
                reasons.append(f"SEC filing: {fact.category} {fact.event_date}")
        low, high = TRADING_GAP_PRICE_RATIO
        adjusted = before / ratio if before else None
        price_ok = bool(adjusted and after and low <= after / adjusted <= high)
        if not price_ok and not same_by_filing:
            reasons.append(
                "price does not resume where it stopped"
                + (f" ({after} against {adjusted:.6g})" if adjusted and after else " (no price)")
            )
        # The issuers on either side of the gap: the earlier security's when it stopped,
        # the later one's when it started (a ticker can pass between companies).
        a, b = issuer_at(sid, last), issuer_at(other, start)
        if a and b and a != b:
            reasons.append(f"different issuers (CIK {a} and {b})")
        merged = not reasons
        why = (
            f"same security: {gap} sessions without a trade, no delisting notice or exit "
            + (
                "before trading resumed; SEC filings say the same shares resumed"
                if same_by_filing and not price_ok
                else f"before trading resumed, price {after} against {adjusted:.6g}"
                + (f" after splits x{ratio:g}" if ratio != 1.0 else "")
            )
            if merged
            else "; ".join(reasons)
        )
        row = SegmentLink(
            link_id=f"gap:{symbol}:{last.isoformat()}:{start.isoformat()}",
            symbol=symbol,
            earlier_last=last,
            later_first=start,
            gap_sessions=gap,
            close_before=before,
            close_after=after,
            split_ratio=ratio,
            issuer_before=a,
            issuer_after=b,
            decision="merged" if merged else "kept_separate",
            reason=why[:1000],
        )
        session.add(row)
        rows.append(row)
    session.flush()
    return rows
