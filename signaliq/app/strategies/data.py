"""Read stored daily bars for a simulation. Read-only: nothing is changed or filled in.

Three things are decided here, once, for every strategy:

* Only bars up to the end of the run's period are read. A run on the
  in-sample period therefore cannot see a single later bar.
* Only the unbroken part of each symbol's history is used: bars from the
  symbol's gap-free start onwards (see ``summarise_session_coverage``). An
  isolated earlier bar, such as SPY's bar of 2018-11-01, stays in the database
  and is listed as excluded, but never reaches an indicator.
* The bars read are fingerprinted, so a result can be tied to its exact input.
"""

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import SOURCE_ALPACA, Settings
from app.data.validation import is_trading_day, session_date, summarise_session_coverage
from app.models import Bar

DAILY_TIMEFRAME = "1Day"


@dataclass(frozen=True)
class DailyBar:
    """One completed daily bar."""

    session: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class DailyData:
    """The daily bars a simulation reads, per symbol, oldest first."""

    bars: dict[str, list[DailyBar]]
    #: Stored sessions that were NOT used: before the gap-free start, or on a closed day.
    excluded: dict[str, list[date]]
    source: str
    feed: str
    adjustment: str
    fingerprint: str

    @property
    def bar_count(self) -> int:
        return sum(len(items) for items in self.bars.values())

    @property
    def first_session(self) -> date | None:
        starts = [items[0].session for items in self.bars.values() if items]
        return min(starts) if starts else None

    @property
    def last_session(self) -> date | None:
        ends = [items[-1].session for items in self.bars.values() if items]
        return max(ends) if ends else None


def fingerprint_bars(bars: dict[str, list[DailyBar]]) -> str:
    """SHA-256 over every bar in a fixed order and text form."""
    digest = hashlib.sha256()
    for symbol in sorted(bars):
        for bar in bars[symbol]:
            digest.update(
                f"{symbol}|{bar.session.isoformat()}|{bar.open!r}|{bar.high!r}|{bar.low!r}|"
                f"{bar.close!r}|{bar.volume!r}\n".encode()
            )
    return digest.hexdigest()


def load_daily_bars(
    session: Session,
    settings: Settings,
    symbols: Sequence[str],
    *,
    end: date | None = None,
) -> DailyData:
    """Load the usable daily history of ``symbols`` up to ``end`` (inclusive)."""
    source, feed, adjustment = SOURCE_ALPACA, settings.alpaca_feed, settings.alpaca_adjustment
    rows = session.execute(
        select(Bar.symbol, Bar.timestamp, Bar.open, Bar.high, Bar.low, Bar.close, Bar.volume)
        .where(
            Bar.symbol.in_(list(symbols)),
            Bar.timeframe == DAILY_TIMEFRAME,
            # One feed and one adjustment only: values from different ones must never mix.
            Bar.source == source,
            Bar.feed == feed,
            Bar.adjustment == adjustment,
        )
        .order_by(Bar.symbol, Bar.timestamp)
    ).all()

    stored: dict[str, list[DailyBar]] = defaultdict(list)
    for symbol, timestamp, open_, high, low, close, volume in rows:
        day = session_date(timestamp)
        if end is None or day <= end:
            stored[symbol].append(DailyBar(day, open_, high, low, close, volume))

    bars: dict[str, list[DailyBar]] = {}
    excluded: dict[str, list[date]] = {}
    for symbol in symbols:
        items = stored.get(symbol, [])
        coverage = summarise_session_coverage(bar.session for bar in items)
        used = (
            [
                bar
                for bar in items
                if bar.session >= coverage.gap_free_from and is_trading_day(bar.session)
            ]
            if coverage is not None
            else []
        )
        kept = {bar.session for bar in used}
        bars[symbol] = used
        excluded[symbol] = [bar.session for bar in items if bar.session not in kept]

    return DailyData(
        bars=bars,
        excluded=excluded,
        source=source,
        feed=feed,
        adjustment=adjustment,
        fingerprint=fingerprint_bars(bars),
    )
