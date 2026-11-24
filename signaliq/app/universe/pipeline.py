"""Run the validation batch: fetch, assemble securities, adjust, decide membership.

Reads Alpaca's SIP daily bars and corporate actions; writes only to the SIP
research database. Bars stop at ``DATA_END`` (2024-12-31): nothing from the
reserved out-of-sample period is read.
"""

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import Settings, is_valid_symbol
from app.data.alpaca_client import AlpacaClient
from app.data.validation import NEW_YORK, trading_days
from app.universe import segments as seg
from app.universe.batch import KNOWN_CASES, PLAIN_TICKER, draw_random
from app.universe.membership import DailyRow, MembershipRule, decide, month_end_sessions
from app.universe.models import (
    BatchSymbol,
    CorporateAction,
    DealValue,
    ExitEvidence,
    IssuerSic,
    Membership,
    Security,
    SecurityBar,
    SecurityIssuer,
    SpinOff,
    SplitCheck,
    TickerBar,
    TickerHistory,
    TypeEvidence,
)
from app.universe.sectypes import USABLE_STATUSES

logger = logging.getLogger(__name__)

DATA_START = date(2016, 1, 1)
DATA_END = date(2024, 12, 31)
CA_TYPES = (
    "forward_split,reverse_split,unit_split,name_change,cash_merger,stock_merger,"
    "stock_and_cash_merger,worthless_removal,redemption,spin_off"
)
RANDOM_QUOTAS = {
    "cash_merger_acquiree": 12,
    "stock_merger_acquiree": 12,
    "name_change_old_symbol": 12,
    "split_2016_2018": 10,
}
#: A security still "trades" when its last real trade is this close to the end of the data.
STILL_TRADING_SESSIONS = 5
#: Merger events are matched to an exit when dated this close to the last trade.
EXIT_MATCH_SESSIONS = 15
MEMBERSHIP_FIRST_DECISION = date(2020, 1, 1)
#: Last decision whose month stays inside the in-sample period.
MEMBERSHIP_LAST_DECISION = date(2024, 11, 30)


def sip_client(settings: Settings) -> AlpacaClient:
    """A client for consolidated (SIP) raw daily bars. The IEX settings are not changed."""
    return AlpacaClient(
        settings.model_copy(update={"alpaca_feed": "sip", "alpaca_adjustment": "raw"})
    )


def utc_window(first: date, last: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(first, time(0, 0), tzinfo=NEW_YORK).astimezone(UTC),
        datetime.combine(last, time(23, 59, 59), tzinfo=NEW_YORK).astimezone(UTC),
    )


def _date(value) -> date | None:
    return date.fromisoformat(value[:10]) if isinstance(value, str) and value else None


def _num(value) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


# =============================================================================
# Corporate actions
# =============================================================================


def fetch_corporate_actions(
    client: AlpacaClient, *, symbols: Sequence[str] | None, types: str, first: date, last: date
) -> list[dict]:
    """All events in the window, following pagination with small pages (limit 100)."""
    events, token = [], None
    params: dict = {
        "types": types,
        "start": first.isoformat(),
        "end": last.isoformat(),
        "limit": 100,
    }
    if symbols:
        params["symbols"] = ",".join(symbols)
    for _ in range(10_000):
        page = dict(params)
        if token:
            page["page_token"] = token
        body = client.get_json("/v1/corporate-actions", page)
        for kind, items in (body.get("corporate_actions") or {}).items():
            singular = kind[:-1] if kind.endswith("s") else kind
            for item in items:
                events.append({"action_type": singular, **item})
        token = body.get("next_page_token")
        if not token:
            return events
    raise RuntimeError("corporate-action pagination did not finish")


def store_corporate_actions(session: Session, events: Iterable[dict]) -> int:
    added = 0
    for event in events:
        if not event.get("id") or session.get(CorporateAction, event["id"]) is not None:
            continue
        session.add(
            CorporateAction(
                id=event["id"],
                action_type=event["action_type"],
                process_date=_date(event.get("process_date")),
                ex_date=_date(event.get("ex_date")),
                effective_date=_date(event.get("effective_date")),
                # Spin-offs name the parent "source_symbol" and its rate "source_rate".
                symbol=event.get("symbol") or event.get("source_symbol"),
                old_symbol=event.get("old_symbol"),
                new_symbol=event.get("new_symbol"),
                acquiree_symbol=event.get("acquiree_symbol"),
                acquirer_symbol=event.get("acquirer_symbol"),
                old_rate=_num(event.get("old_rate"))
                if event.get("old_rate") is not None
                else _num(event.get("source_rate")),
                new_rate=_num(event.get("new_rate")),
                rate=_num(event.get("rate")),
                cash_rate=_num(event.get("cash_rate")),
                raw=event,
            )
        )
        added += 1
    session.flush()
    return added


# =============================================================================
# Batch
# =============================================================================


def build_batch(session: Session, client: AlpacaClient) -> list[BatchSymbol]:
    """Known cases plus the seeded random draw. Recreated from scratch each run."""
    session.execute(delete(BatchSymbol))
    rows = [
        BatchSymbol(
            symbol=k.symbol,
            category=k.category,
            selection="known_case",
            reason=k.expected,
            declared_type=k.declared_type,
        )
        for k in KNOWN_CASES
    ]
    pool_events = fetch_corporate_actions(
        client,
        symbols=None,
        types="cash_merger,stock_merger,name_change",
        first=date(2020, 1, 1),
        last=DATA_END,
    ) + fetch_corporate_actions(
        client,
        symbols=None,
        types="forward_split,reverse_split",
        first=date(2016, 1, 1),
        last=date(2018, 12, 31),
    )
    for symbol, source, reason in draw_random(
        pool_events, {r.symbol for r in rows}, quotas=RANDOM_QUOTAS
    ):
        rows.append(
            BatchSymbol(
                symbol=symbol,
                category=source,
                selection=f"random:{source}:seed",
                reason=reason,
                declared_type="unknown",
            )
        )
    session.add_all(rows)
    session.flush()
    return rows


def discover_linked_symbols(session: Session, rounds: int = 2) -> list[BatchSymbol]:
    """Add tickers that renames connect to batch tickers (old or new names)."""
    added = []
    for _ in range(rounds):
        known = {row.symbol for row in session.execute(select(BatchSymbol)).scalars()}
        new_rows = []
        for event in session.execute(
            select(CorporateAction).where(CorporateAction.action_type == "name_change")
        ).scalars():
            for here, other in (
                (event.old_symbol, event.new_symbol),
                (event.new_symbol, event.old_symbol),
            ):
                if here in known and other and PLAIN_TICKER.match(other) and other not in known:
                    known.add(other)
                    new_rows.append(
                        BatchSymbol(
                            symbol=other,
                            category="linked",
                            selection="discovered",
                            reason=f"renamed from/to {here} (event {event.process_date})",
                            declared_type="unknown",
                            discovered_from=here,
                        )
                    )
        if not new_rows:
            break
        session.add_all(new_rows)
        session.flush()
        added += new_rows
    return added


# =============================================================================
# Bars
# =============================================================================


def fetch_literal_bars(client: AlpacaClient, symbols: Sequence[str]) -> dict[str, list[dict]]:
    start, end = utc_window(DATA_START, DATA_END)
    out: dict[str, list[dict]] = {}
    valid = [s for s in symbols if is_valid_symbol(s)]
    for i in range(0, len(valid), 25):
        chunk = valid[i : i + 25]
        out.update(client.fetch_bars(chunk, timeframe="1Day", start=start, end=end, asof="-"))
    return out


def _to_bar(item: dict) -> seg.Bar:
    when = datetime.fromisoformat(item["t"].replace("Z", "+00:00")).astimezone(NEW_YORK).date()
    return seg.Bar(
        when,
        item["o"],
        item["h"],
        item["l"],
        item["c"],
        int(item["v"]),
        float(item.get("vw", item["c"])),
        int(item.get("n", 0)),
    )


def store_ticker_bars(session: Session, bars: dict[str, list[dict]]) -> None:
    session.execute(delete(TickerBar))
    rows = []
    for symbol, items in bars.items():
        for item in items:
            bar = _to_bar(item)
            rows.append(
                TickerBar(
                    symbol=symbol,
                    session=bar.session,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap=bar.vwap,
                    trade_count=bar.trade_count,
                    is_filler=bar.is_filler,
                )
            )
    session.add_all(rows)
    session.flush()


# =============================================================================
# Securities
# =============================================================================


@dataclass
class Assembled:
    chains: list[seg.Chain]
    fillers_by_symbol: dict[str, int]
    segments_by_symbol: dict[str, list[seg.Segment]]
    #: Permanent security ids, one per chain (set by ``store_securities``).
    ids: list[int] | None = None


def security_key(chain: seg.Chain) -> str:
    """A security's permanent key: the ticker and first session of its first segment."""
    first = chain.segments[0]
    return f"{first.symbol}|{first.first_session.isoformat()}"


def assign_security_ids(session: Session, chains: Sequence[seg.Chain]) -> list[int]:
    """Permanent ids for the chains, from the registry; new securities get new ids.

    An id is never reused. A registered security whose first segment is now inside
    another chain (joined by a trading-gap merge or a rename) is marked ``merged`` with
    the id it became part of; one whose ticker is not in this batch is ``absent``.
    """
    from app.universe.models import SecurityRegistry

    registry = {r.security_key: r for r in session.execute(select(SecurityRegistry)).scalars()}
    next_id = max((r.security_id for r in registry.values()), default=0) + 1
    ids = []
    in_chain: dict[str, int] = {}
    for chain in chains:
        key = security_key(chain)
        row = registry.get(key)
        if row is None:
            first = chain.segments[0]
            row = SecurityRegistry(
                security_key=key,
                security_id=next_id,
                first_symbol=first.symbol,
                first_session=first.first_session,
                status="active",
            )
            session.add(row)
            registry[key] = row
            next_id += 1
        row.status, row.merged_into = "active", None
        ids.append(row.security_id)
        for segment in chain.segments[1:]:
            in_chain[f"{segment.symbol}|{segment.first_session.isoformat()}"] = row.security_id
    active = {security_key(chain) for chain in chains}
    for key, row in registry.items():
        if key in active:
            continue
        if key in in_chain:
            row.status, row.merged_into = "merged", in_chain[key]
        else:
            row.status, row.merged_into = "absent", None
    session.flush()
    return ids


def assemble(session: Session) -> Assembled:
    by_symbol: dict[str, list[seg.Bar]] = defaultdict(list)
    for row in session.execute(
        select(TickerBar).order_by(TickerBar.symbol, TickerBar.session)
    ).scalars():
        by_symbol[row.symbol].append(
            seg.Bar(
                row.session,
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
                row.vwap,
                row.trade_count,
            )
        )
    segments, fillers = {}, {}
    for symbol, bars in by_symbol.items():
        segments[symbol], fillers[symbol] = seg.split_segments(symbol, bars)
    renames = [
        seg.NameChange(e.id, e.old_symbol, e.new_symbol, e.process_date)
        for e in session.execute(
            select(CorporateAction).where(CorporateAction.action_type == "name_change")
        ).scalars()
        if e.old_symbol and e.new_symbol and e.process_date
    ]
    from app.universe.models import SegmentLink  # derived each run, read here

    gaps = [
        seg.GapLink(row.link_id, row.symbol, row.earlier_last, row.later_first)
        for row in session.execute(
            select(SegmentLink).where(SegmentLink.decision == "merged")
        ).scalars()
    ]
    from app.universe.evidence import curated_identity_links, identity_link_id

    identities = [
        seg.IdentityLink(
            identity_link_id(link), link.old_symbol, link.old_last, link.new_symbol, link.new_first
        )
        for link, missing in curated_identity_links(session)
        if not missing
    ]
    return Assembled(seg.build_chains(segments, renames, gaps, identities), fillers, segments)


def classify_exit(chain: seg.Chain, events: list[CorporateAction], data_end_sessions: list[date]):
    """How a security that stopped trading ended, and what a holder is assumed to get."""
    last = chain.last_session
    if sum(1 for s in data_end_sessions if s > last) <= STILL_TRADING_SESSIONS:
        return "trading", "none", None, None, None, None, None
    symbol = chain.current_symbol
    near = [
        e
        for e in events
        if e.process_date
        and last <= e.process_date
        and seg.sessions_between(last, e.process_date) < EXIT_MATCH_SESSIONS
    ]
    for kind in ("cash_merger", "stock_and_cash_merger", "stock_merger"):
        for e in near:
            if e.action_type == kind and e.acquiree_symbol == symbol:
                if kind == "cash_merger":
                    return (
                        "ended",
                        kind,
                        e.id,
                        e.rate,
                        e.acquirer_symbol,
                        e.rate,
                        f"cash paid per share in the merger: {e.rate}",
                    )
                return (
                    "ended",
                    kind,
                    e.id,
                    e.cash_rate,
                    e.acquirer_symbol,
                    None,
                    f"converted into {e.acquirer_symbol or 'acquirer'} shares"
                    + (f" plus {e.cash_rate} cash" if e.cash_rate else "")
                    + "; value not computed in this batch",
                )
    for e in near:
        if (
            e.action_type == "name_change"
            and e.old_symbol == symbol
            and e.new_symbol
            and e.new_symbol.endswith("Q")
            and len(e.new_symbol) >= 4
        ):
            return (
                "ended",
                "bankruptcy_otc",
                e.id,
                None,
                None,
                0.0,
                f"renamed {e.new_symbol} (bankruptcy, OTC; no SIP data): total loss assumed",
            )
    return (
        "ended",
        "unexplained",
        None,
        None,
        None,
        0.0,
        "no corporate action explains the end: total loss assumed (the agreed default)",
    )


def store_securities(session: Session, assembled: Assembled) -> list[Security]:
    # Children first: every table that refers to a security is cleared before the securities.
    for table in (
        Membership,
        DealValue,
        IssuerSic,
        SecurityIssuer,
        ExitEvidence,
        TypeEvidence,
        SpinOff,
        SplitCheck,
        SecurityBar,
        TickerHistory,
        Security,
    ):
        session.execute(delete(table))
    calendar = trading_days(DATA_START, DATA_END)
    batch = {row.symbol: row for row in session.execute(select(BatchSymbol)).scalars()}
    events = list(session.execute(select(CorporateAction)).scalars())
    securities = []
    assembled.ids = assign_security_ids(session, assembled.chains)
    for security_id, chain in zip(assembled.ids, assembled.chains, strict=True):
        status, exit_type, event_id, cash, acquirer, value, why = classify_exit(
            chain, events, calendar
        )
        declared = next(
            (
                batch[s.symbol].declared_type
                for s in chain.segments
                if s.symbol in batch and batch[s.symbol].declared_type != "unknown"
            ),
            "unknown",
        )
        security = Security(
            security_id=security_id,
            first_session=chain.first_session,
            last_trade_session=chain.last_session,
            last_symbol=chain.current_symbol,
            status=status,
            exit_type=exit_type,
            exit_event_id=event_id,
            exit_cash_rate=cash,
            acquirer_symbol=acquirer,
            assumed_exit_value=value,
            exit_assumption=why,
            declared_type=declared,
            real_trade_sessions=sum(len(s.bars) for s in chain.segments),
        )
        session.add(security)
        for index, segment in enumerate(chain.segments):
            session.add(
                TickerHistory(
                    security_id=security_id,
                    symbol=segment.symbol,
                    valid_from=segment.first_session,
                    valid_to=segment.last_session,
                    linked_by="first listing" if index == 0 else chain.links[index - 1],
                )
            )
        securities.append(security)
    session.flush()
    return securities


def store_security_bars(session: Session, assembled: Assembled) -> None:
    splits = [
        seg.Split(e.symbol, e.ex_date or e.process_date, e.new_rate / e.old_rate)
        for e in session.execute(
            select(CorporateAction).where(
                CorporateAction.action_type.in_(("forward_split", "reverse_split", "unit_split"))
            )
        ).scalars()
        if e.symbol and e.new_rate and e.old_rate and (e.ex_date or e.process_date)
    ]
    ids = assembled.ids or assign_security_ids(session, assembled.chains)
    for security_id, chain in zip(ids, assembled.chains, strict=True):
        factors = seg.split_factors(chain, splits)
        session.add_all(
            SecurityBar(
                security_id=security_id,
                session=bar.session,
                symbol=symbol,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap,
                trade_count=bar.trade_count,
                split_factor=factors[bar.session],
            )
            for symbol, bar in chain.bars()
        )
    session.flush()


def attach_alpaca_adjusted(
    session: Session, client: AlpacaClient, settings: Settings
) -> dict[int, dict]:
    """Fetch Alpaca's own entity-mapped bars (raw and split-adjusted) to compare with ours."""
    adjusted_client = AlpacaClient(
        settings.model_copy(update={"alpaca_feed": "sip", "alpaca_adjustment": "split"})
    )
    report: dict[int, dict] = {}
    try:
        securities = list(session.execute(select(Security)).scalars())
        symbols = sorted({s.last_symbol for s in securities if is_valid_symbol(s.last_symbol)})
        start, end = utc_window(DATA_START, DATA_END)
        raw_map, adj_map = {}, {}
        for i in range(0, len(symbols), 25):
            chunk = symbols[i : i + 25]
            raw_map.update(client.fetch_bars(chunk, timeframe="1Day", start=start, end=end))
            adj_map.update(
                adjusted_client.fetch_bars(chunk, timeframe="1Day", start=start, end=end)
            )
    finally:
        adjusted_client.close()
    for security in securities:
        raw = {_to_bar(x).session: x for x in raw_map.get(security.last_symbol, [])}
        adj = {_to_bar(x).session: x for x in adj_map.get(security.last_symbol, [])}
        bars = list(
            session.execute(
                select(SecurityBar).where(SecurityBar.security_id == security.security_id)
            ).scalars()
        )
        same_raw = mismatched = missing = 0
        for bar in bars:
            ours = raw.get(bar.session)
            if ours is None:
                missing += 1
            elif math.isclose(ours["c"], bar.close, rel_tol=1e-9) and int(ours["v"]) == bar.volume:
                same_raw += 1
            else:
                mismatched += 1
            if bar.session in adj:
                bar.alpaca_adjusted_close = float(adj[bar.session]["c"])
        report[security.security_id] = {
            "bars": len(bars),
            "match": same_raw,
            "mismatch": mismatched,
            "missing": missing,
        }
    session.flush()
    return report


# =============================================================================
# Membership
# =============================================================================


def decide_membership(session: Session, rule: MembershipRule) -> list[date]:
    session.execute(delete(Membership))
    history: dict[int, list[DailyRow]] = defaultdict(list)
    for bar in session.execute(
        select(SecurityBar).order_by(SecurityBar.security_id, SecurityBar.session)
    ).scalars():
        history[bar.security_id].append(DailyRow(bar.session, bar.close, bar.volume))
    # Only a type backed by independent evidence counts; anything else is not eligible.
    types = {
        s.security_id: s.security_type if s.type_status in USABLE_STATUSES else "unknown"
        for s in session.execute(select(Security)).scalars()
    }
    calendar = trading_days(DATA_START, DATA_END)
    decisions = month_end_sessions(MEMBERSHIP_FIRST_DECISION, MEMBERSHIP_LAST_DECISION)
    # A split whose size the prices do not confirm keeps the security out until reviewed.
    held_out = {
        check.security_id: "price_adjustment_under_review"
        for check in session.execute(
            select(SplitCheck).where(SplitCheck.verdict == "magnitude_mismatch")
        ).scalars()
        if check.security_id is not None
    }
    # A blank-check company (SIC 6770 on file by the decision date) is not chosen.
    from app.universe.evidence import blank_check_exclusions  # evidence imports pipeline

    dated = blank_check_exclusions(session)
    fingerprint = rule.fingerprint()
    for when in decisions:
        for d in decide(
            when, history, types, rule, [s for s in calendar if s <= when], held_out, dated
        ):
            session.add(
                Membership(
                    decision_session=d.session,
                    security_id=d.security_id,
                    rule_fingerprint=fingerprint,
                    eligible=d.eligible,
                    reason=d.reason,
                    raw_close=d.raw_close,
                    median_dollar_volume=d.median_dollar_volume,
                    rank=d.rank,
                    member=d.member,
                )
            )
    session.flush()
    return decisions
