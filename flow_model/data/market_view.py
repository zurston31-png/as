"""The point-in-time firewall.

This is the single most important correctness surface in the system. A
subtle error here does not produce a wrong number in one place -- it
silently invalidates every number the system will ever produce, and the
backtest will look *better*, not worse.

The design is structural rather than procedural. A `MarketView` is built for
one `(symbol, now)` pair and, at construction, slices every series at the
visible cutoff via `ColumnSeries.prefix()`. What it holds are read-only
NumPy views of the prefix `[0:visible)` and **no reference to the parent
`SymbolData`**. NumPy slicing is O(1), so this costs about five microseconds
per view, and the consequence is that future data is not merely
un-requested: it is absent from the object. Code that reaches into a private
attribute still cannot see tomorrow.

That property was not true in the first implementation, which kept the whole
`SymbolData` and re-applied a visible count inside each accessor. No
accessor leaked, but the guarantee was procedural -- a new accessor written
by someone who believed this docstring would have returned the full history,
and the self-check below would not have caught it. The docstring is
load-bearing here, so it is now made true rather than softened.

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

from datetime import datetime, timedelta

import numpy as np

from flow_model.core.contracts import Bar, OptionsSnapshot, QuoteSnapshot, TickAggregate
from flow_model.core.enums import Feed
from flow_model.data.series import (
    NS_PER_SECOND,
    BarSeries,
    ColumnSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    from_ns,
    to_ns,
)

_EMPTY_FLOAT = np.zeros(0, dtype=np.float64)
_EMPTY_FLOAT.flags.writeable = False
_EMPTY_INT = np.zeros(0, dtype=np.int64)
_EMPTY_INT.flags.writeable = False


class LookaheadError(RuntimeError):
    """Raised when a view's own invariants detect future data."""


class MarketView:
    """A bounded, read-only window onto one symbol's data at one instant."""

    __slots__ = (
        "symbol",
        "now",
        "latency_seconds",
        "primary_interval",
        "_now_ns",
        "_cutoff_ns",
        "_bars",
        "_ticks",
        "_quotes",
        "_options",
        "_hidden",
    )

    def __init__(
        self,
        data,
        now: datetime,
        latency_seconds: float = 0.0,
        now_ns: int | None = None,
    ) -> None:
        """Build a view at `now`.

        `now_ns` lets a caller that already holds an exact int64 nanosecond
        instant bypass the datetime conversion. `from_ns` truncates below the
        microsecond (datetime's resolution), so a clock that round-tripped a
        nanosecond-stamped bar close through a datetime would place the
        cutoff just before that bar and the view would not see the bar at its
        own close. `DataStore` passes the integer for exactly that reason.
        """
        if latency_seconds < 0:
            raise ValueError(
                f"latency_seconds={latency_seconds} is negative, which would make "
                "data visible before it exists. Feed latency may only delay "
                "visibility, never advance it."
            )
        self.symbol = data.symbol
        self.now = now
        self.latency_seconds = float(latency_seconds)
        self.primary_interval = int(data.primary_interval)
        self._now_ns = to_ns(now) if now_ns is None else int(now_ns)
        self._cutoff_ns = self._now_ns - int(round(latency_seconds * NS_PER_SECOND))

        # Slice every feed at the cutoff and keep only the prefixes. The
        # parent SymbolData is intentionally NOT retained: an attribute that
        # held it would be a path to tomorrow's data, which is the thing this
        # class exists to make impossible.
        self._hidden: dict[str, int | None] = {}
        self._bars: dict[int, BarSeries] = {}
        for interval, series in data.bars.items():
            self._bars[int(interval)] = self._cut(series, f"bars[{interval}]")
        self._ticks: dict[int, TickSeries] = {}
        for interval, series in data.ticks.items():
            self._ticks[int(interval)] = self._cut(series, f"ticks[{interval}]")
        self._quotes: QuoteSeries | None = (
            self._cut(data.quotes, "quotes") if data.quotes is not None else None
        )
        self._options: OptionsSeries | None = (
            self._cut(data.options, "options") if data.options is not None else None
        )

    def _cut(self, series: ColumnSeries, key: str):
        """Prefix a series at the cutoff, recording the hidden boundary.

        The boundary is kept as one integer timestamp rather than a reference
        to the parent, so `assert_no_future_leak` can still verify that
        nothing at or before the cutoff was wrongly hidden while the view
        holds no future data at all.
        """
        count = series.visible_count(self._cutoff_ns)
        self._hidden[key] = series.hidden_boundary_ns(count)
        return series.prefix(count)

    # --- identity ------------------------------------------------------

    @property
    def cutoff(self) -> datetime:
        """The newest timestamp this view can see."""
        return from_ns(self._cutoff_ns)

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
            return self._quotes is not None and len(self._quotes) > 0
        if feed is Feed.TICK_AGGREGATE:
            series = self._ticks.get(self.primary_interval)
            return series is not None and len(series) > 0
        if feed is Feed.OPTIONS_SNAPSHOT:
            return self._options is not None and len(self._options) > 0
        raise ValueError(f"unknown feed: {feed!r}")

    def available_feeds(self) -> frozenset[Feed]:
        return frozenset(feed for feed in Feed if self.has_feed(feed))

    def feed_age_seconds(self, feed: Feed) -> float | None:
        """Seconds between `now` and the newest visible observation.

        None when the feed has nothing visible. Drives STALE grading.
        """
        series = self._series_for(feed)
        if series is None or len(series) == 0:
            return None
        return (self._now_ns - int(series.ts_ns[-1])) / NS_PER_SECOND

    def _series_for(self, feed: Feed) -> ColumnSeries | None:
        """The prefix backing one feed at the PRIMARY interval.

        Tick aggregates and bars are keyed by interval; `has_feed` and
        `feed_age_seconds` both report on the primary interval so that a
        non-primary feed cannot report an age the grader would attribute to
        the primary one.
        """
        if feed is Feed.BARS:
            return self._bars.get(self.primary_interval)
        if feed is Feed.QUOTES:
            return self._quotes
        if feed is Feed.TICK_AGGREGATE:
            return self._ticks.get(self.primary_interval)
        if feed is Feed.OPTIONS_SNAPSHOT:
            return self._options
        raise ValueError(f"unknown feed: {feed!r}")

    # --- bars ----------------------------------------------------------

    def _bar_series(self, interval: int | None) -> BarSeries | None:
        key = self.primary_interval if interval is None else int(interval)
        return self._bars.get(key)

    def bar_count(self, interval: int | None = None) -> int:
        """Visible bars at this instant."""
        series = self._bar_series(interval)
        return len(series) if series is not None else 0

    def warmup_ok(self, bars_required: int, interval: int | None = None) -> bool:
        """Whether enough history exists for a feature with this lookback.

        The signal engine returns WAIT('warmup') on False, which prevents
        trading on a half-filled rolling window.
        """
        return self.bar_count(interval) >= bars_required

    def column(self, name: str, n: int, interval: int | None = None) -> np.ndarray:
        """Read-only window of the last `n` values of a bar column.

        Oldest-to-newest. Shorter than `n` when less history exists -- never
        padded, never wrapped. Prefer this over `bars()` in feature code: it
        avoids materializing contract objects per bar.

        An unknown column name raises `SchemaError` from the series rather
        than a bare `KeyError`, so a typo names itself.
        """
        series = self._bar_series(interval)
        if series is None or len(series) == 0 or n <= 0:
            return _EMPTY_FLOAT
        if not series.has(name):
            return series.col(name)  # raises SchemaError listing present columns
        total = len(series)
        return series.window(name, total - int(n), total)

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
        series = self._bar_series(interval)
        if series is None or len(series) == 0 or n <= 0:
            return _EMPTY_INT
        total = len(series)
        return series.ts_window(total - int(n), total)

    def bars(self, n: int, interval: int | None = None) -> tuple[Bar, ...]:
        """Last `n` bars as contract objects, oldest-to-newest."""
        series = self._bar_series(interval)
        if series is None or len(series) == 0 or n <= 0:
            return ()
        total = len(series)
        start = max(0, total - int(n))
        return tuple(series.bar_at(i) for i in range(start, total))

    def last_bar(self, interval: int | None = None) -> Bar | None:
        series = self._bar_series(interval)
        if series is None or len(series) == 0:
            return None
        return series.bar_at(len(series) - 1)

    def last_price(self, interval: int | None = None) -> float | None:
        """Most recent visible close. The reference price for signals."""
        window = self.closes(1, interval)
        return float(window[0]) if window.size else None

    # --- quotes --------------------------------------------------------

    def quote(self) -> QuoteSnapshot | None:
        """Most recent visible top-of-book snapshot."""
        if self._quotes is None or len(self._quotes) == 0:
            return None
        return self._quotes.quote_at(len(self._quotes) - 1)

    def quote_column(self, name: str, n: int) -> np.ndarray:
        if self._quotes is None or len(self._quotes) == 0 or n <= 0:
            return _EMPTY_FLOAT
        if not self._quotes.has(name):
            return self._quotes.col(name)  # raises SchemaError
        total = len(self._quotes)
        return self._quotes.window(name, total - int(n), total)

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
        series = self._ticks.get(key)
        if series is None or len(series) == 0 or n <= 0 or not series.has(name):
            return _EMPTY_FLOAT
        total = len(series)
        return series.window(name, total - int(n), total)

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
        series = self._ticks.get(key)
        if series is None or len(series) == 0 or n <= 0:
            return ()
        total = len(series)
        start = max(0, total - int(n))
        return tuple(series.tick_at(i) for i in range(start, total))

    def last_tick_aggregate(self, interval: int | None = None) -> TickAggregate | None:
        aggregates = self.tick_aggregates(1, interval)
        return aggregates[-1] if aggregates else None

    def tick_count(self, interval: int | None = None) -> int:
        key = self.primary_interval if interval is None else int(interval)
        series = self._ticks.get(key)
        return len(series) if series is not None else 0

    # --- options -------------------------------------------------------

    def options_snapshot(self) -> OptionsSnapshot | None:
        if self._options is None or len(self._options) == 0:
            return None
        return self._options.snapshot_at(len(self._options) - 1)

    def options_column(self, name: str, n: int) -> np.ndarray:
        if self._options is None or len(self._options) == 0 or n <= 0:
            return _EMPTY_FLOAT
        if not self._options.has(name):
            return _EMPTY_FLOAT
        total = len(self._options)
        return self._options.window(name, total - int(n), total)

    def options_count(self) -> int:
        return len(self._options) if self._options is not None else 0

    def quote_count(self) -> int:
        return len(self._quotes) if self._quotes is not None else 0

    def options_are_intraday(self) -> bool:
        """False for end-of-day chain snapshots, which cannot express flow timing."""
        return self._options.is_intraday if self._options is not None else False

    def options_cadence_seconds(self) -> float | None:
        return self.feed_cadence_seconds(Feed.OPTIONS_SNAPSHOT)

    def feed_cadence_seconds(self, feed: Feed) -> float | None:
        """Median interval between a feed's visible observations.

        The quality layer grades staleness against a feed's own cadence
        rather than one global wall-clock bound: an end-of-day options chain
        is a session old by construction and is not therefore frozen, while
        a 5-minute bar feed two sessions old is. None when fewer than two
        observations are visible, which is too little to infer a cadence.
        """
        series = self._series_for(feed)
        if series is None or len(series) < 2:
            return None
        return float(np.median(np.diff(series.ts_ns))) / NS_PER_SECOND

    # --- self-check ----------------------------------------------------

    def assert_no_future_leak(self) -> None:
        """Verify this view's own invariants.

        Checks the WHOLE visible prefix rather than only its newest row. An
        earlier version compared `ts_ns[count - 1]` alone, which certified a
        view as clean while a future row sat earlier in the prefix -- the one
        method whose entire job is to detect a leak could not detect that
        class of leak.

        Three assertions, in both directions:
          * no visible timestamp postdates the cutoff
          * the visible prefix is strictly ascending (a non-monotonic series
            makes `searchsorted` span a future row, and is reported as such
            rather than silently mis-sliced)
          * the first hidden observation genuinely postdates the cutoff, so an
            off-by-one that hid the newest bar is caught too
        """
        named: list[tuple[str, ColumnSeries]] = []
        for interval, series in self._bars.items():
            named.append((f"bars[{interval}]", series))
        for interval, series in self._ticks.items():
            named.append((f"ticks[{interval}]", series))
        if self._quotes is not None:
            named.append(("quotes", self._quotes))
        if self._options is not None:
            named.append(("options", self._options))

        for key, series in named:
            if len(series):
                newest = int(series.ts_ns.max())
                if newest > self._cutoff_ns:
                    raise LookaheadError(
                        f"{self.symbol} {key}: a visible observation at "
                        f"{from_ns(newest).isoformat()} is after the cutoff "
                        f"{self.cutoff.isoformat()}"
                    )
                if len(series) > 1 and not np.all(np.diff(series.ts_ns) > 0):
                    raise LookaheadError(
                        f"{self.symbol} {key}: the visible prefix is not strictly "
                        "ascending, so the cutoff slice cannot be trusted"
                    )
            boundary = self._hidden.get(key)
            if boundary is not None and boundary <= self._cutoff_ns:
                raise LookaheadError(
                    f"{self.symbol} {key}: observation "
                    f"{from_ns(boundary).isoformat()} is at or before the cutoff "
                    f"{self.cutoff.isoformat()} but is hidden"
                )
