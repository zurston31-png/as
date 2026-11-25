"""Turn literal-ticker bar series into security identities. Pure functions, no I/O.

Three facts about Alpaca's daily bars, measured on real data, drive the rules:

1. Asked for a ticker with ``asof="-"``, Alpaca returns exactly what traded
   under that ticker on each date, whichever company it was.
2. When a ticker stops trading on an exchange (bankruptcy, delisting, a rename
   that left the ticker free), Alpaca keeps returning one bar per session with
   zero volume, zero trades and the last price repeated, until the ticker
   trades again, possibly for a different company. These are **filler bars**:
   no trade happened, so they are never treated as prices.
3. Renames are recorded as ``name_change`` corporate actions (old ticker, new
   ticker, date), densely from 2020 on.

So a security is assembled as follows:

* Each ticker's real trades (non-filler bars) are cut into **segments** wherever
  more than ``MAX_GAP_SESSIONS`` trading sessions pass without a real trade. A
  long silence ends a listing; whatever trades under the ticker later is
  treated as a different listing unless a corporate action says otherwise.
* Segments of different tickers are **linked** into one security when a
  name_change event joins them: the old ticker's segment ends shortly before
  the event and the new ticker's segment starts shortly after it.
* A ticker can pass from one company to another without a pause (Eldorado Resorts
  took the ticker CZR the session after it acquired the old Caesars). Such a run is
  **cut** at the new holder's first session (``handover_cuts``), so each company keeps
  its own segment and the rename links the new holder's.
* Every chain of linked segments is one security with one internal ID. Its
  ticker history is the list of segments.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from app.data.validation import trading_days

#: A listing ends when this many trading sessions pass without a real trade.
MAX_GAP_SESSIONS = 10

#: A rename links two segments when the old one ends at most this many sessions
#: before the event date and the new one starts at most this many after it.
LINK_WINDOW_SESSIONS = 10


@dataclass(frozen=True)
class Bar:
    session: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int

    @property
    def is_filler(self) -> bool:
        """No trade happened: zero volume and zero trades (the price is a copy)."""
        return self.volume == 0 and self.trade_count == 0


@dataclass(frozen=True)
class Segment:
    """An uninterrupted run of real trades under one ticker."""

    symbol: str
    bars: tuple[Bar, ...]

    @property
    def first_session(self) -> date:
        return self.bars[0].session

    @property
    def last_session(self) -> date:
        return self.bars[-1].session

    @property
    def key(self) -> tuple[str, date]:
        return (self.symbol, self.first_session)


@dataclass(frozen=True)
class NameChange:
    event_id: str
    old_symbol: str
    new_symbol: str
    process_date: date


@dataclass(frozen=True)
class GapLink:
    """Two segments of one ticker that are the same security across a trading gap.

    Decided from evidence (``app.universe.evidence.decide_gap_links``) and recorded with
    it; the segments themselves are kept as they are.
    """

    link_id: str
    symbol: str
    earlier_last: date
    later_first: date


@dataclass(frozen=True)
class IdentityLink:
    """Two segments of different tickers that are one security across a long suspension.

    A documented exception to the rule that a silence of more than ``MAX_GAP_SESSIONS``
    ends a listing: used only for an entry in ``app.universe.curated`` whose SEC passages
    were all found (``CURATED_IDENTITY_LINKS``). The exact segments are named by their last
    and first sessions, so nothing else is joined.
    """

    link_id: str
    old_symbol: str
    old_last: date
    new_symbol: str
    new_first: date


@dataclass(frozen=True)
class Handover:
    """A ticker that passed from one company to another with no pause in trading.

    ``cut`` is the new holder's first session under the ticker: the run of real trades is
    cut there into two segments, one per company. ``renamed_from`` is the new holder's
    previous ticker (empty for a documented exception whose earlier ticker is not in the
    data), ``link_id`` the rename event or the exception, ``exit_note`` the earlier holder's
    own exit record.
    """

    symbol: str
    cut: date
    renamed_from: str
    link_id: str
    exit_note: str


@dataclass(frozen=True)
class TickerExit:
    """A record that a ticker's holder stopped using it on ``day`` (acquired, renamed)."""

    symbol: str
    day: date
    note: str


@dataclass
class Chain:
    """One security: segments in time order, and the events that joined them."""

    segments: list[Segment]
    links: list[str] = field(default_factory=list)  # name_change event ids

    @property
    def first_session(self) -> date:
        return self.segments[0].first_session

    @property
    def last_session(self) -> date:
        return self.segments[-1].last_session

    @property
    def current_symbol(self) -> str:
        return self.segments[-1].symbol

    def bars(self) -> list[tuple[str, Bar]]:
        return [(segment.symbol, bar) for segment in self.segments for bar in segment.bars]


def sessions_between(first: date, second: date) -> int:
    """Trading sessions strictly between two dates."""
    if second <= first:
        return 0
    return max(0, len(trading_days(first, second)) - 2)


def split_segments(
    symbol: str, bars: Sequence[Bar], cuts: Iterable[date] = ()
) -> tuple[list[Segment], int]:
    """Cut one ticker's bars into segments of real trades.

    A segment also starts at each date in ``cuts`` (a handover: the first session of a new
    holder of the ticker). Returns the segments and the number of filler bars dropped.
    """
    ordered = sorted(bars, key=lambda bar: bar.session)
    real = [bar for bar in ordered if not bar.is_filler]
    fillers = len(ordered) - len(real)
    cut_days = sorted(set(cuts))
    segments: list[Segment] = []
    current: list[Bar] = []
    for bar in real:
        handed = (
            any(current[-1].session < day <= bar.session for day in cut_days) if current else False
        )
        if current and (
            handed or sessions_between(current[-1].session, bar.session) > MAX_GAP_SESSIONS
        ):
            segments.append(Segment(symbol, tuple(current)))
            current = []
        current.append(bar)
    if current:
        segments.append(Segment(symbol, tuple(current)))
    return segments, fillers


def _within(first: date, second: date, limit: int) -> bool:
    """``second`` is on or after ``first`` and at most ``limit`` sessions later."""
    return first <= second and sessions_between(first, second) < limit


def _near(day: date, event_day: date, limit: int) -> bool:
    """``day`` lies within ``limit`` sessions of ``event_day``, before or after it.

    Alpaca's rename dates can trail the actual switch by several sessions (the
    SPAC BOWX started trading as WE on 2021-10-21; the event is dated
    2021-10-27), so a link may not insist on the event coming first.
    """
    return _within(day, event_day, limit + 1) or _within(event_day, day, limit + 1)


def handover_cuts(
    segments_by_symbol: Mapping[str, Sequence[Segment]],
    name_changes: Iterable[NameChange],
    exits: Iterable[TickerExit],
) -> list[Handover]:
    """Tickers taken over by another company without a pause, from two Alpaca records.

    A rename X -> T is a handover when, at once:

    * X's segment ends within ``LINK_WINDOW_SESSIONS`` of the rename;
    * T was trading before X's last session and goes on trading after it (another company
      held T: one uninterrupted run would otherwise join two companies);
    * T's earlier holder has its own exit recorded within ``LINK_WINDOW_SESSIONS`` of the
      rename: acquired (a merger naming T as the acquired company) or renamed away from T.

    The run of T is cut at its first real trade after X's last session. Without the second
    record nothing is cut (a late rename date alone does not split a security).
    """
    exits_by_symbol: dict[str, list[TickerExit]] = {}
    for item in exits:
        exits_by_symbol.setdefault(item.symbol, []).append(item)
    found: dict[tuple[str, date], Handover] = {}
    for event in sorted(name_changes, key=lambda e: (e.process_date, e.event_id)):
        if event.old_symbol == event.new_symbol:
            continue
        old = next(
            (
                s
                for s in segments_by_symbol.get(event.old_symbol, ())
                if _near(s.last_session, event.process_date, LINK_WINDOW_SESSIONS)
            ),
            None,
        )
        if old is None:
            continue
        held = next(
            (
                s
                for s in segments_by_symbol.get(event.new_symbol, ())
                if s.first_session < old.last_session < s.last_session
            ),
            None,
        )
        if held is None:
            continue
        gone = sorted(
            (
                x
                for x in exits_by_symbol.get(event.new_symbol, ())
                if _near(x.day, event.process_date, LINK_WINDOW_SESSIONS)
            ),
            key=lambda x: (x.day, x.note),
        )
        if not gone:
            continue
        cut = next(b.session for b in held.bars if b.session > old.last_session)
        key = (event.new_symbol, cut)
        if key not in found:
            found[key] = Handover(
                event.new_symbol, cut, event.old_symbol, event.event_id, gone[0].note
            )
    return [found[k] for k in sorted(found)]


def build_chains(
    segments_by_symbol: Mapping[str, Sequence[Segment]],
    name_changes: Iterable[NameChange],
    gap_links: Iterable[GapLink] = (),
    identity_links: Iterable[IdentityLink] = (),
) -> list[Chain]:
    """Link segments by rename events and return one chain per security.

    A rename (old, new, date) links the old ticker's segment that ends within
    the window of the date to the new ticker's segment that starts within the
    window of it, either side, provided the new one starts after the old one
    ends and not so long after that the listing would count as ended. Several events for the
    same rename (Alpaca sometimes records duplicates) produce one link. A segment is linked at
    most once in each direction; a second candidate is ignored and left unlinked.
    """
    successor: dict[tuple[str, date], tuple[tuple[str, date], str]] = {}
    has_predecessor: set[tuple[str, date]] = set()
    # Documented exceptions first: each names its two segments exactly. One whose segments
    # are missing, overlap or are already linked is ignored.
    for link in sorted(identity_links, key=lambda x: (x.old_last, x.link_id)):
        old = next(
            (
                s
                for s in segments_by_symbol.get(link.old_symbol, ())
                if s.last_session == link.old_last
            ),
            None,
        )
        new = next(
            (
                s
                for s in segments_by_symbol.get(link.new_symbol, ())
                if s.first_session == link.new_first
            ),
            None,
        )
        if old is None or new is None or old.key in successor or new.key in has_predecessor:
            continue
        if new.first_session <= old.last_session:
            continue
        successor[old.key] = (new.key, link.link_id)
        has_predecessor.add(new.key)
    # Trading gaps first: the same ticker resuming is the strongest kind of link, and
    # each was decided from evidence beforehand. A link whose segments are missing or
    # already linked is ignored.
    for link in sorted(gap_links, key=lambda g: (g.symbol, g.earlier_last)):
        here = segments_by_symbol.get(link.symbol, ())
        old = next((s for s in here if s.last_session == link.earlier_last), None)
        new = next((s for s in here if s.first_session == link.later_first), None)
        if old is None or new is None or old.key in successor or new.key in has_predecessor:
            continue
        if new.first_session <= old.last_session:
            continue
        successor[old.key] = (new.key, link.link_id)
        has_predecessor.add(new.key)
    for event in sorted(name_changes, key=lambda e: (e.process_date, e.event_id)):
        if event.old_symbol == event.new_symbol:
            continue
        old = next(
            (
                s
                for s in segments_by_symbol.get(event.old_symbol, ())
                if _near(s.last_session, event.process_date, LINK_WINDOW_SESSIONS)
            ),
            None,
        )
        new = next(
            (
                s
                for s in segments_by_symbol.get(event.new_symbol, ())
                if _near(s.first_session, event.process_date, LINK_WINDOW_SESSIONS)
            ),
            None,
        )
        if old is None or new is None or old.key in successor or new.key in has_predecessor:
            continue
        if new.first_session <= old.last_session:
            continue  # overlapping trading: not one security continuing under a new ticker
        if sessions_between(old.last_session, new.first_session) > MAX_GAP_SESSIONS:
            continue  # a silence long enough to end a listing: not a continuation
        successor[old.key] = (new.key, event.event_id)
        has_predecessor.add(new.key)

    by_key = {s.key: s for items in segments_by_symbol.values() for s in items}
    chains = []
    for key in sorted(by_key, key=lambda k: (by_key[k].first_session, k[0])):
        if key in has_predecessor:
            continue
        chain = Chain([by_key[key]])
        while key in successor:
            key, event_id = successor[key]
            chain.segments.append(by_key[key])
            chain.links.append(event_id)
        chains.append(chain)
    return chains


# =============================================================================
# Split adjustment
# =============================================================================


@dataclass(frozen=True)
class Split:
    symbol: str
    ex_date: date
    #: Shares after / shares before: 4.0 for a 4-for-1 split, 0.05 for a 1-for-20 reverse split.
    ratio: float
    event_id: str | None = None


def owned_on(chain: Chain, symbol: str, day: date, handed: Iterable[tuple[str, date]] = ()) -> bool:
    """Whether an event recorded for ``symbol`` on ``day`` belongs to ``chain``.

    A ticker owns the dates after the previous ticker's last trade (so an event
    recorded under the new ticker on its first day, like XPRO on 2021-10-04,
    counts) up to the first trade under the next ticker (so an event recorded
    under the old ticker on the rename date, like UTX on 2020-04-03, counts).
    An event on the first day of the security belongs to nothing earlier.

    ``handed`` lists (ticker, first session) where a ticker passed to a new holder
    (``handover_cuts``). An event recorded under that ticker on or before that first
    session belongs to the earlier holder, never to the new one: Ingersoll-Rand plc's share
    consolidation is recorded under IR on 2020-03-02, the day it became TT and Gardner
    Denver took the ticker IR.
    """
    handed = set(handed)
    for index, segment in enumerate(chain.segments):
        later = chain.segments[index + 1] if index + 1 < len(chain.segments) else None
        earlier = chain.segments[index - 1] if index > 0 else None
        start = earlier.last_session if earlier is not None else segment.first_session
        if (segment.symbol, segment.first_session) in handed:
            start = segment.first_session
        end = later.first_session if later is not None else segment.last_session
        if symbol == segment.symbol and start < day <= end:
            return True
    return False


def owned_splits(
    chain: Chain, splits: Iterable[Split], handed: Iterable[tuple[str, date]] = ()
) -> list[Split]:
    handed = set(handed)
    return [split for split in splits if owned_on(chain, split.symbol, split.ex_date, handed)]


def split_factors(
    chain: Chain, splits: Iterable[Split], handed: Iterable[tuple[str, date]] = ()
) -> dict[date, float]:
    """Factor that turns each raw price of ``chain`` into a split-adjusted one.

    Adjusted price = raw price / factor, where the factor is the product of the
    ratios of every split of this security that takes effect after that session.
    A split belongs to the security when its ticker was the security's ticker on
    the split's date. A split recorded under the old ticker of a rename and
    dated the day the new ticker starts (Alpaca does this, for example UTX on
    2020-04-03) belongs to the security too: each ticker owns the dates up to
    the first trade under the next ticker.
    """
    owned = owned_splits(chain, splits, handed)
    factors = {}
    for _, bar in chain.bars():
        factor = 1.0
        for split in owned:
            if bar.session < split.ex_date:
                factor *= split.ratio
        factors[bar.session] = factor
    return factors
