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
from collections import defaultdict
from datetime import date, timedelta

import httpx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import is_valid_symbol
from app.data.alpaca_client import AlpacaClient
from app.universe import adjustments, outcomes, sec, sectypes
from app.universe import segments as seg
from app.universe.batch import PLAIN_TICKER
from app.universe.curated import CURATED_SPIN_OFFS
from app.universe.models import (
    BatchSymbol,
    CorporateAction,
    ExitEvidence,
    ExternalResponse,
    Security,
    SecurityBar,
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
        symbols, names = _sec_search_keys(session, history[security.security_id], shared)
        found[security.security_id] = sec.find(client, symbols, names, security.last_trade_session)
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


def _sec_relevant(finding: sec.SecFinding, symbol: str):
    """(type, notices about that class) of a finding, for the security traded as ``symbol``."""
    kind = sec.notice_type([n.class_description for _, n in finding.notices], symbol)
    relevant = [
        (f, n) for f, n in finding.notices if sec.class_type(n.class_description, symbol) == kind
    ]
    return kind, relevant


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
                    no_longer_used = (
                        security.status == "ended" or h.valid_to < security.last_trade_session
                    )
                    if (
                        no_longer_used
                        and wiki_names
                        and not any(sectypes.names_agree(name, other) for other in wiki_names)
                    ):
                        note(
                            security,
                            "openfigi_rejected",
                            h.symbol,
                            kind,
                            f"{detail}: names a different company than "
                            f"{'/'.join(wiki_names)}; not used",
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
            kind, relevant = _sec_relevant(finding, security.last_symbol)
            issuer = f"{finding.company}, CIK {finding.cik}, SIC {finding.sic or 'none'}" + (
                " (blank check)" if finding.sic == "6770" else ""
            )
            if kind:
                votes.append(sectypes.TypeVote("sec_form25", kind))
            for filing, notice in finding.notices:
                used = (filing, notice) in relevant and kind is not None
                note(
                    security,
                    "sec_form25" if used else "sec_form25_other",
                    security.last_symbol,
                    sec.class_type(notice.class_description, security.last_symbol) or "unknown",
                    f"{filing.form} {filing.filed}: class '{notice.class_description}' ({issuer})"
                    + ("" if used else "; another class or unclear, not counted"),
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
            found += sec_evidence(finding, security.last_symbol)
        # The same ticker trading again later, and very short runs.
        if any(
            start > last
            for start, _, sid in segments_by_symbol[symbol]
            if sid != security.security_id
        ):
            found.append(
                outcomes.Evidence(
                    "alpaca_bars",
                    "resumed",
                    None,
                    f"{symbol} trades again later (halt, or the ticker was reused)",
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
    session.flush()


def sec_evidence(finding: sec.SecFinding, symbol: str) -> list[outcomes.Evidence]:
    """Exit evidence from a matched SEC company (all filings are before 2025 by construction)."""
    found = []
    _, relevant = _sec_relevant(finding, symbol)
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
    for k in finding.eight_ks:
        kind = sec.eight_k_category(k.items)
        cash, sentence = finding.cash if k is acquisition else (None, "")
        found.append(
            outcomes.Evidence(
                "sec",
                "acquired_cash" if cash is not None else kind,
                k.filed,
                f"8-K filed {k.filed} ({finding.company}), {k.items}"
                + (f"; terms: {sentence[:200]}" if cash is not None else ""),
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
