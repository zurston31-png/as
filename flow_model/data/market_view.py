"""The point-in-time firewall.

This is the single most important correctness surface in the system. A
subtle error here does not produce a wrong number in one place -- it
silently invalidates every number the system will ever produce, and the
backtest will look *better*, not worse.

The design is structural rather than procedural. A `MarketView` is built
for one `(symbol, now)` pair and, at construction, slices every series at
the visible cutoff. What it holds are read-only NumPy *views* of the
prefix `[0:visible)`. NumPy slicing is O(1), so this costs nothing, and the
consequence is that future data is not merely un-requested -- it is
physically absent from the object. Code that reaches into a private
attribute still cannot see tomorrow.

Three further rules, each enforced:

1. **No accessor takes a timestamp.** There is no `bars_between(a, b)` and
   no `value_at(ts)`. A caller can only ask for "the last n", which cannot
   be aimed at the future. `tests/unit/test_market_view.py` asserts this by
   introspection, so a future accessor that breaks the rule fails the build.

2. **No accessor returns the underlying series.** Only bounded windows.

3. **Negative latency is rejected.** A negative feed latency is literally a
   request to see the future, and it would otherwise work.

Visibility rule: a bar closing at exactly `t` is visible at `t` (you know a
bar when it closes), so the cutoff comparison is `close_ts <= now -
latency` and `searchsorted` uses `side="right"`. The backtester separately
forbids *filling* at the signal bar's close; that is an execution rule, not
a visibility rule, and the two are deliberately independent.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from flow_model.core.contracts import Bar, OptionsSnapshot, QuoteSnapshot, TickAggregate
from flow_model.data.base import Feed
from flow_model.data.series import NS_PER_SECOND, BarSeries, from_ns, to_ns

if TYPE_CHECKING:  # pragma: no cover
    from flow_model.data.store import SymbolData

_EMPTY_FLOAT = np.zeros(0, dtype=np.float64)
_EMPTY_FLOAT.flags.writeable = False


class LookaheadError(RuntimeError):
    """Raised when a view's own invariants detect future data."""


class MarketView:
    """A bounded, read-only window onto one symbol's data at one instant."""

    __slots__ = (
        "symbol",
        "now",
        "latency_seconds",
        "_now_ns",
        "_cutoff_ns",
        "_data",
        "_visible_bars",
        "_visible_ticks",
        "_visible_quotes",
        "_visible_options",
    )

    def __init__(
        self,
        data: "SymbolData",
        now: datetime,
        latency_seconds: float = 0.0,
    ) -> None:
        if latency_seconds < 0:
            raise ValueError(
                f"latency_seconds={latency_seconds} is negative, which would make "
                "data visible before it exists. Feed latency may only delay "
                "visibility, never advance it."
            )
        self.symbol = data.symbol
        self.now = now
        self.latency_seconds = float(latency_seconds)
        self._now_ns = to_ns(now)
        self._cutoff_ns = self._now_ns - int(round(latency_seconds * NS_PER_SECOND))
        self._data = data

        # Eager cutoff resolution: searchsorted is ~O(log n) and this makes
        # every later accessor a pure slice with no chance of recomputing a
        # cutoff inconsistently.
        self._visible_bars = {
            interval: series.visible_count(self._cutoff_ns)
            for interval, series in data.bars.items()
        }
        self._visible_ticks = {
            interval: series.visible_count(self._cutoff_ns)
            for interval, series in data.ticks.items()
        }
        self._visible_quotes = (
            data.quotes.visible_count(self._cutoff_ns) if data.quotes is not None else 0
        )
        self._visible_options = (
            data.options.visible_count(self._cutoff_ns) if data.options is not None else 0
        )

    # --- identity ------------------------------------------------------

    @property
    def cutoff(self) -> datetime:
        """The newest timestamp this view can see."""
        return from_ns(self._cutoff_ns)

    @property
    def primary_interval(self) -> int:
        return self._data.primary_interval

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"MarketView({self.symbol} now={self.now.isoformat()} "
            f"cutoff={self.cutoff.isoformat()} bars={self.bar_count()})"
        )

    # --- feed presence -------------------------------------------------

    def has_feed(self, feed: Feed) -> bool:
        """Whether a feed exists AND has at least one visible observation."""
        if feed is Feed.BARS:
            return self.bar_count() > 0
        if feed is Feed.QUOTES:
            return self._visible_quotes > 0
        if feed is Feed.TICK_AGGREGATE:
            return any(count > 0 for count in self._visible_ticks.values())
        if feed is Feed.OPTIONS_SNAPSHOT:
            return self._visible_options > 0
        raise ValueError(f"unknown feed: {feed!r}")

    def available_feeds(self) -> frozenset[Feed]:
        return frozenset(feed for feed in Feed if self.has_feed(feed))

    def feed_age_seconds(self, feed: Feed) -> float | None:
        """Seconds between `now` and the newest visible observation.

        None when the feed has nothing visible. Drives STALE grading.
        """
        last = self._last_feed_ns(feed)
        if last is None:
            return None
        return (self._now_ns - last) / NS_PER_SECOND

    def _last_feed_ns(self, feed: Feed) -> int | None:
        if feed is Feed.BARS:
            count = self._visible_bars.get(self.primary_interval, 0)
            series = self._data.bars.get(self.primary_interval)
            return int(series.ts_ns[count - 1]) if series is not None and count else None
        if feed is Feed.QUOTES:
            if self._data.quotes is None or not self._visible_quotes:
                return None
            return int(self._data.quotes.ts_ns[self._visible_quotes - 1])
        if feed is Feed.TICK_AGGREGATE:
            series = self._data.ticks.get(self.primary_interval)
            count = self._visible_ticks.get(self.primary_interval, 0)
            return int(series.ts_ns[count - 1]) if series is not None and count else None
        if feed is Feed.OPTIONS_SNAPSHOT:
            if self._data.options is None or not self._visible_options:
                return None
            return int(self._data.options.ts_ns[self._visible_options - 1])
        raise ValueError(f"unknown feed: {feed!r}")

    # --- bars ----------------------------------------------------------

    def _bar_series(self, interval: int | None) -> tuple[BarSeries | None, int]:
        key = self.primary_interval if interval is None else int(interval)
        series = self._data.bars.get(key)
        if series is None:
            return None, 0
        return series, self._visible_bars.get(key, 0)

    def bar_count(self, interval: int | None = None) -> int:
        """Visible bars at this instant."""
        _, count = self._bar_series(interval)
        return count

    def warmup_ok(self, bars_required: int, interval: int | None = None) -> bool:
        """Whether enough history exists for a feature with this lookback.

        The signal engine returns WAIT('warmup') on False, which prevents
        trading on a half-filled rolling window.
        """
        return self.bar_count(interval) >= bars_required

    def column(self, name: str, n: int, interval: int | None = None) -> np.ndarray:
        """Read-only window of the last `n` values of a bar column.

        Oldest-to-newest. Shorter than `n` when less history exists -- never
        padded, never wrapped. Prefer this over `bars()` in feature code:
        it avoids materializing contract objects per bar.
        """
        series, count = self._bar_series(interval)
        if series is None or count == 0 or n <= 0:
            return _EMPTY_FLOAT
        return series.window(name, count - int(n), count)

    def opens(self, n: int, interval: int | None = None) -> np.ndarray:
        return self.column("open", n, interval)

    def highs(self, n: int, interval: int | None = None) -> np.ndarray:
        return self.column("high", n, interval)

    def lows(self, n: int, interval: int | None = None) -> np.ndarray:
        return self.column("low", n, interval)

    def closes(self, n: int, interval: int | None = None) -> np.ndarray:
        return self.column("close", n, interval)

    def volumes(self, n: int, interval: int | None = None) -> np.ndarray:
        return self.column("volume", n, interval)

    def bar_timestamps(self, n: int, interval: int | None = None) -> np.ndarray:
        """Read-only window of bar close timestamps as int64 UTC ns."""
        series, count = self._bar_series(interval)
        if series is None or count == 0 or n <= 0:
            return np.zeros(0, dtype=np.int64)
        return series.ts_window(count - int(n), count)

    def bars(self, n: int, interval: int | None = None) -> tuple[Bar, ...]:
        """Last `n` bars as contract objects, oldest-to-newest."""
        series, count = self._bar_series(interval)
        if series is None or count == 0 or n <= 0:
            return ()
        start = max(0, count - int(n))
        return tuple(series.bar_at(i) for i in range(start, count))

    def last_bar(self, interval: int | None = None) -> Bar | None:
        series, count = self._bar_series(interval)
        if series is None or count == 0:
            return None
        return series.bar_at(count - 1)

    def last_price(self, interval: int | None = None) -> float | None:
        """Most recent visible close. The reference price for signals."""
        window = self.closes(1, interval)
        return float(window[0]) if window.size else None

    # --- quotes --------------------------------------------------------

    def quote(self) -> QuoteSnapshot | None:
        """Most recent visible top-of-book snapshot."""
        if self._data.quotes is None or not self._visible_quotes:
            return None
        return self._data.quotes.quote_at(self._visible_quotes - 1)

    def quote_column(self, name: str, n: int) -> np.ndarray:
        if self._data.quotes is None or not self._visible_quotes or n <= 0:
            return _EMPTY_FLOAT
        return self._data.quotes.window(
            name, self._visible_quotes - int(n), self._visible_quotes
        )

    def spreads(self, n: int) -> np.ndarray:
        """Last `n` quoted spreads. A fresh array (computed), not a view."""
        bids = self.quote_column("bid", n)
        asks = self.quote_column("ask", n)
        if bids.size == 0 or bids.size != asks.size:
            return _EMPTY_FLOAT
        return asks - bids

    # --- tick aggregates ----------------------------------------------

    def tick_column(self, name: str, n: int, interval: int | None = None) -> np.ndarray:
        key = self.primary_interval if interval is None else int(interval)
        series = self._data.ticks.get(key)
        count = self._visible_ticks.get(key, 0)
        if series is None or count == 0 or n <= 0 or not series.has(name):
            return _EMPTY_FLOAT
        return series.window(name, count - int(n), count)

    def deltas(self, n: int, interval: int | None = None) -> np.ndarray:
        """Signed volume delta per bar. Fresh array.

        Empty when no tick feed exists. Deliberately NOT estimated from bar
        volume: tick-rule approximations on bar data are a known-bad
        estimator, and returning one here would be indistinguishable from
        real order flow to every caller above.
        """
        buys = self.tick_column("buy_volume", n, interval)
        sells = self.tick_column("sell_volume", n, interval)
        if buys.size == 0 or buys.size != sells.size:
            return _EMPTY_FLOAT
        return buys - sells

    def tick_aggregates(self, n: int, interval: int | None = None) -> tuple[TickAggregate, ...]:
        key = self.primary_interval if interval is None else int(interval)
        series = self._data.ticks.get(key)
        count = self._visible_ticks.get(key, 0)
        if series is None or count == 0 or n <= 0:
            return ()
        start = max(0, count - int(n))
        return tuple(series.tick_at(i) for i in range(start, count))

    def last_tick_aggregate(self, interval: int | None = None) -> TickAggregate | None:
        aggregates = self.tick_aggregates(1, interval)
        return aggregates[-1] if aggregates else None

    # --- options -------------------------------------------------------

    def options_snapshot(self) -> OptionsSnapshot | None:
        if self._data.options is None or not self._visible_options:
            return None
        return self._data.options.snapshot_at(self._visible_options - 1)

    def options_column(self, name: str, n: int) -> np.ndarray:
        if self._data.options is None or not self._visible_options or n <= 0:
            return _EMPTY_FLOAT
        if not self._data.options.has(name):
            return _EMPTY_FLOAT
        return self._data.options.window(
            name, self._visible_options - int(n), self._visible_options
        )

    def options_are_intraday(self) -> bool:
        """False for end-of-day chain snapshots, which cannot express flow timing."""
        return self._data.options.is_intraday if self._data.options is not None else False

    # --- self-check ----------------------------------------------------

    def assert_no_future_leak(self) -> None:
        """Verify this view's cutoff arithmetic against the raw series.

        Checks both directions: nothing visible may postdate the cutoff, and
        nothing at or before the cutoff may be hidden (an off-by-one that
        hid the newest bar would be a silent accuracy loss rather than a
        lookahead, but it is still wrong).
        """
        checks: list[tuple[str, object, int]] = []
        for interval, series in self._data.bars.items():
            checks.append((f"bars[{interval}]", series, self._visible_bars.get(interval, 0)))
        for interval, series in self._data.ticks.items():
            checks.append((f"ticks[{interval}]", series, self._visible_ticks.get(interval, 0)))
        if self._data.quotes is not None:
            checks.append(("quotes", self._data.quotes, self._visible_quotes))
        if self._data.options is not None:
            checks.append(("options", self._data.options, self._visible_options))

        for name, series, count in checks:
            total = len(series)
            if count > total:
                raise LookaheadError(
                    f"{self.symbol} {name}: visible count {count} exceeds series length {total}"
                )
            if count > 0 and int(series.ts_ns[count - 1]) > self._cutoff_ns:
                raise LookaheadError(
                    f"{self.symbol} {name}: newest visible observation "
                    f"{from_ns(series.ts_ns[count - 1]).isoformat()} is after the cutoff "
                    f"{self.cutoff.isoformat()}"
                )
            if count < total and int(series.ts_ns[count]) <= self._cutoff_ns:
                raise LookaheadError(
                    f"{self.symbol} {name}: observation "
                    f"{from_ns(series.ts_ns[count]).isoformat()} is at or before the cutoff "
                    f"{self.cutoff.isoformat()} but is hidden"
                )
