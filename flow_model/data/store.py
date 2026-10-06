"""Loaded data, and the only sanctioned way to get a view of it.

`DataStore` owns the loaded series and hands out `MarketView` objects. It is
also where the sealed-out-of-sample guard is wired in: every load passes
through `DataAccessGuard.check_access`, so a sweep that happens to span the
holdout fails at load time rather than quietly producing a contaminated
result.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterator, Mapping, Sequence

import numpy as np

from flow_model.core.determinism import stable_hash
from flow_model.data.base import DatasetFingerprint, DataLayerError, DataSourceAdapter, Feed
from flow_model.data.market_view import MarketView
from flow_model.data.series import BarSeries, OptionsSeries, QuoteSeries, TickSeries, from_ns
from flow_model.utils.logging import get_logger

logger = get_logger("data.store")


class SymbolData:
    """Every series loaded for one symbol."""

    __slots__ = ("symbol", "primary_interval", "bars", "ticks", "quotes", "options", "fingerprint")

    def __init__(
        self,
        symbol: str,
        primary_interval: int,
        bars: Mapping[int, BarSeries],
        ticks: Mapping[int, TickSeries] | None = None,
        quotes: QuoteSeries | None = None,
        options: OptionsSeries | None = None,
        fingerprint: DatasetFingerprint | None = None,
    ) -> None:
        self.symbol = symbol
        self.primary_interval = int(primary_interval)
        self.bars = {int(k): v for k, v in bars.items()}
        self.ticks = {int(k): v for k, v in (ticks or {}).items()}
        self.quotes = quotes
        self.options = options
        self.fingerprint = fingerprint

        if self.primary_interval not in self.bars:
            raise DataLayerError(
                f"{symbol}: no bars at the primary interval {self.primary_interval}s "
                f"(loaded intervals: {sorted(self.bars)})"
            )
        for interval, series in self.bars.items():
            if series.symbol != symbol:
                raise DataLayerError(
                    f"bar series symbol {series.symbol!r} does not match {symbol!r}"
                )
            if series.interval_seconds != interval:
                raise DataLayerError(
                    f"{symbol}: series registered at {interval}s declares "
                    f"interval_seconds={series.interval_seconds}"
                )
        for series in (self.quotes, self.options):
            if series is not None and series.symbol != symbol:
                raise DataLayerError(
                    f"series symbol {series.symbol!r} does not match {symbol!r}"
                )
        for interval, series in self.ticks.items():
            if series.symbol != symbol:
                raise DataLayerError(
                    f"tick series symbol {series.symbol!r} does not match {symbol!r}"
                )

    # --- description ---------------------------------------------------

    @property
    def primary_bars(self) -> BarSeries:
        return self.bars[self.primary_interval]

    def available_feeds(self) -> frozenset[Feed]:
        """Feeds physically present (ignoring any point-in-time cutoff)."""
        feeds = {Feed.BARS} if len(self.primary_bars) else set()
        if self.quotes is not None and len(self.quotes):
            feeds.add(Feed.QUOTES)
        if any(len(series) for series in self.ticks.values()):
            feeds.add(Feed.TICK_AGGREGATE)
        if self.options is not None and len(self.options):
            feeds.add(Feed.OPTIONS_SNAPSHOT)
        return frozenset(feeds)

    def describe(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "primary_interval": self.primary_interval,
            "bars": {interval: len(series) for interval, series in sorted(self.bars.items())},
            "ticks": {interval: len(series) for interval, series in sorted(self.ticks.items())},
            "quotes": len(self.quotes) if self.quotes is not None else 0,
            "options": len(self.options) if self.options is not None else 0,
            "feeds": sorted(feed.value for feed in self.available_feeds()),
            "first_ts": self.primary_bars.first_ts.isoformat() if len(self.primary_bars) else None,
            "last_ts": self.primary_bars.last_ts.isoformat() if len(self.primary_bars) else None,
            "data_hash": self.fingerprint.data_hash if self.fingerprint else "",
            # `source` is the field that distinguishes a synthetic dataset from
            # a real one. Dropping it here meant the natural run-report
            # serialization of a synthetic run was indistinguishable from a
            # real one, which is how a synthetic backtest gets mistaken for
            # evidence later.
            "source": self.fingerprint.source if self.fingerprint else "unknown",
            "synthetic": bool(self.primary_bars.meta.get("synthetic", False)),
        }


class DataStore:
    """Holds loaded symbols and issues point-in-time views."""

    def __init__(self, latency_seconds: float = 0.0, guard=None) -> None:
        if latency_seconds < 0:
            raise ValueError("latency_seconds must not be negative")
        self.latency_seconds = float(latency_seconds)
        self.guard = guard
        self._symbols: dict[str, SymbolData] = {}

    # --- population ----------------------------------------------------

    def add(self, data: SymbolData) -> "DataStore":
        self._symbols[data.symbol] = data
        return self

    def load(
        self,
        adapter: DataSourceAdapter,
        symbol: str,
        start: date,
        end: date,
        primary_interval: int,
        higher_intervals: Sequence[int] = (),
        clip_to_allowed: bool = False,
    ) -> SymbolData:
        """Load one symbol through the seal guard.

        With `clip_to_allowed=True` a request overlapping the sealed window
        is trimmed to the usable range and the trim is logged, instead of
        raising. That is the right behaviour for "give me all research
        data"; the default is to raise, so an accidental full-decade sweep
        is loud.
        """
        if self.guard is not None:
            if clip_to_allowed:
                allowed = self.guard.clip_to_allowed(start, end)
                if allowed is None:
                    raise DataLayerError(
                        f"{symbol}: the requested range {start} -> {end} lies entirely "
                        "inside the sealed out-of-sample window; nothing is loadable"
                    )
                if (allowed.start, allowed.end) != (start, end):
                    logger.warning(
                        "clipped %s request %s -> %s to %s -> %s (sealed window)",
                        symbol, start, end, allowed.start, allowed.end,
                    )
                start, end = allowed.start, allowed.end
            else:
                self.guard.check_access(start, end, purpose=f"load {symbol} bars")

        feeds = adapter.available_feeds(symbol)
        if Feed.BARS not in feeds:
            raise DataLayerError(
                f"adapter {adapter.name!r} has no bar feed for {symbol!r}; "
                f"available: {sorted(f.value for f in feeds)}"
            )

        bars: dict[int, BarSeries] = {
            int(primary_interval): adapter.load_bars(symbol, primary_interval, start, end)
        }
        for interval in higher_intervals:
            try:
                bars[int(interval)] = adapter.load_bars(symbol, int(interval), start, end)
            except DataLayerError as exc:
                logger.warning("%s: no %ss bars (%s)", symbol, interval, exc)

        ticks: dict[int, TickSeries] = {}
        if Feed.TICK_AGGREGATE in feeds:
            series = adapter.load_tick_aggregates(symbol, primary_interval, start, end)
            if series is not None:
                ticks[int(primary_interval)] = series

        quotes = adapter.load_quotes(symbol, start, end) if Feed.QUOTES in feeds else None
        options = adapter.load_options(symbol, start, end) if Feed.OPTIONS_SNAPSHOT in feeds else None

        data = SymbolData(
            symbol=symbol,
            primary_interval=int(primary_interval),
            bars=bars,
            ticks=ticks,
            quotes=quotes,
            options=options,
            fingerprint=adapter.fingerprint(
                symbol, int(primary_interval), start, end, len(bars[int(primary_interval)])
            ),
        )
        self.add(data)
        logger.info(
            "loaded %s: %d bars, feeds=%s",
            symbol,
            len(data.primary_bars),
            sorted(f.value for f in data.available_feeds()),
        )
        return data

    # --- access --------------------------------------------------------

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._symbols))

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._symbols

    def __len__(self) -> int:
        return len(self._symbols)

    def get(self, symbol: str) -> SymbolData:
        try:
            return self._symbols[symbol]
        except KeyError:
            raise DataLayerError(
                f"{symbol!r} is not loaded; loaded symbols: {list(self.symbols)}"
            ) from None

    def view(self, symbol: str, now: datetime, now_ns: int | None = None) -> MarketView:
        """The only sanctioned way for code above the data layer to read data.

        `now_ns` carries an exact int64 instant when the caller has one, so a
        nanosecond-stamped observation is not pushed outside the cutoff by
        datetime's microsecond resolution.
        """
        return MarketView(
            self.get(symbol),
            now=now,
            latency_seconds=self.latency_seconds,
            now_ns=now_ns,
        )

    def timeline(self, symbol: str, interval: int | None = None) -> np.ndarray:
        """Bar close timestamps (int64 UTC ns) for the backtest clock.

        Returned read-only. The clock walks these in order; it is the single
        definition of "when the system gets to act".
        """
        data = self.get(symbol)
        key = data.primary_interval if interval is None else int(interval)
        series = data.bars.get(key)
        if series is None:
            raise DataLayerError(f"{symbol}: no bars at {key}s")
        return series.ts_window(0, len(series))

    def merged_timeline(self, symbols: Sequence[str] | None = None) -> np.ndarray:
        """Sorted union of bar close timestamps across symbols.

        Used for a multi-symbol backtest so each symbol acts on its own bar
        closes without one symbol's calendar dictating another's.
        """
        chosen = tuple(symbols) if symbols else self.symbols
        if not chosen:
            return np.zeros(0, dtype=np.int64)
        stacked = np.concatenate([np.asarray(self.timeline(s)) for s in chosen])
        return np.unique(stacked)

    def iter_views(self, symbol: str) -> Iterator[MarketView]:
        """Views at each of the symbol's bar closes, in chronological order."""
        for ts_ns in self.timeline(symbol):
            exact = int(ts_ns)
            yield self.view(symbol, from_ns(exact), now_ns=exact)

    # --- provenance ----------------------------------------------------

    def fingerprints(self) -> dict[str, DatasetFingerprint | None]:
        return {symbol: data.fingerprint for symbol, data in sorted(self._symbols.items())}

    def combined_data_hash(self) -> str:
        """One hash identifying the whole loaded dataset, for TradeRecord."""
        return stable_hash(
            {
                symbol: (data.fingerprint.data_hash if data.fingerprint else "")
                for symbol, data in sorted(self._symbols.items())
            }
        )

    def describe(self) -> dict[str, object]:
        return {
            "latency_seconds": self.latency_seconds,
            "combined_data_hash": self.combined_data_hash(),
            "symbols": {symbol: data.describe() for symbol, data in sorted(self._symbols.items())},
        }
