"""Columnar series storage.

A ten-year backtest over five symbols at five-minute bars is on the order
of a million bars per symbol, each touched once per evaluation. Holding
those as Pydantic objects would be both slow and memory-hostile, so storage
is columnar NumPy and the `Bar`/`QuoteSnapshot`/... contract objects are
materialized only at the boundary where something actually needs one.

Two invariants are enforced at construction and relied on everywhere above:

1. **Timestamps are int64 UTC nanoseconds, strictly ascending.** Not merely
   sorted -- strictly -- so duplicate timestamps cannot silently coexist and
   `searchsorted` has an unambiguous answer. An out-of-order series is an
   upstream bug; sorting it at read time would mask interleaved sessions.

2. **Every column has the same length as `ts_ns`.** A short column would
   otherwise read as zeros for the tail of the backtest.

Naive datetimes are rejected. Accepting them and assuming UTC is how a
six-hour session-boundary error gets into a backtest without anyone
noticing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np

from flow_model.core.contracts import Bar, OptionsSnapshot, QuoteSnapshot, TickAggregate
from flow_model.data.base import MonotonicityError, SchemaError

NS_PER_SECOND = 1_000_000_000


# ---------------------------------------------------------------------------
# timestamp helpers
# ---------------------------------------------------------------------------


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
NS_PER_MICROSECOND = 1_000


def to_ns(value: datetime) -> int:
    """UTC nanoseconds since the epoch. Rejects naive datetimes.

    Converted by exact integer arithmetic on a `timedelta` rather than via
    `value.timestamp()`. A float64 holds ~16 significant digits, and a
    modern epoch in seconds already consumes 10 of them, so the float route
    silently loses sub-millisecond precision -- which would mangle the
    microsecond timestamps on quote and trade data.

    Resolution note: `datetime` itself stores only microseconds, so a
    datetime round-trip is microsecond-exact, not nanosecond-exact. The
    int64 `ts_ns` arrays carry full nanosecond resolution and all
    visibility comparisons (`visible_count`) operate on those integers
    directly, so nanosecond-resolution feeds loaded as arrays keep it.
    """
    if not isinstance(value, datetime):
        raise SchemaError(f"expected datetime, got {type(value).__name__}: {value!r}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise SchemaError(
            f"naive datetime {value!r}: the data layer requires timezone-aware "
            "timestamps. Assuming UTC for a naive exchange timestamp is how "
            "session-boundary errors enter a backtest."
        )
    delta = value - EPOCH
    seconds = delta.days * 86_400 + delta.seconds
    return seconds * NS_PER_SECOND + delta.microseconds * NS_PER_MICROSECOND


def from_ns(value: int) -> datetime:
    """UTC datetime from nanoseconds, truncating below the microsecond.

    Truncation (not rounding) keeps the conversion monotonic and keeps a
    derived datetime at or before the instant it represents, so converting
    a cutoff to a datetime can never move it forward into the future.
    """
    return EPOCH + timedelta(microseconds=int(value) // NS_PER_MICROSECOND)


def to_ns_array(values: Iterable[datetime]) -> np.ndarray:
    return np.fromiter((to_ns(v) for v in values), dtype=np.int64)


def _readonly(array: np.ndarray) -> np.ndarray:
    """A read-only view, so a consumer cannot corrupt the store in place."""
    view = array.view()
    view.flags.writeable = False
    return view


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


class ColumnSeries:
    """Immutable columnar series keyed by a strictly ascending timestamp."""

    REQUIRED: tuple[str, ...] = ()
    OPTIONAL: tuple[str, ...] = ()
    NON_NEGATIVE: tuple[str, ...] = ()

    __slots__ = ("symbol", "ts_ns", "_cols", "meta")

    def __init__(
        self,
        symbol: str,
        ts_ns: np.ndarray,
        columns: Mapping[str, np.ndarray],
        meta: Mapping[str, object] | None = None,
        validate: bool = True,
    ) -> None:
        self.symbol = symbol
        # copy=True is load-bearing: np.ascontiguousarray returns the SAME
        # object for an already-contiguous array of the right dtype, so the
        # series would alias the caller's buffer and a later mutation of it
        # would silently rewrite stored market data.
        self.ts_ns = np.array(ts_ns, dtype=np.int64, copy=True, order="C")
        cols: dict[str, np.ndarray] = {}
        known = set(self.REQUIRED) | set(self.OPTIONAL)
        for name, values in columns.items():
            if name not in known:
                raise SchemaError(
                    f"{type(self).__name__}: unknown column {name!r}; "
                    f"known columns are {sorted(known)}"
                )
            cols[name] = np.array(values, copy=True, order="C")
        self._cols = cols
        self.meta = dict(meta or {})
        if validate:
            self._validate()

    # --- validation ----------------------------------------------------

    def _validate(self) -> None:
        missing = [name for name in self.REQUIRED if name not in self._cols]
        if missing:
            raise SchemaError(f"{type(self).__name__}: missing required columns {missing}")

        n = len(self.ts_ns)
        for name, values in self._cols.items():
            if len(values) != n:
                raise SchemaError(
                    f"{type(self).__name__}: column {name!r} has {len(values)} rows "
                    f"but ts_ns has {n}; a short column would read as zeros for "
                    f"the tail of the series"
                )

        if n > 1:
            diffs = np.diff(self.ts_ns)
            if np.any(diffs <= 0):
                bad = int(np.argmax(diffs <= 0))
                raise MonotonicityError(
                    f"{type(self).__name__} {self.symbol}: timestamps are not "
                    f"strictly ascending at row {bad + 1} "
                    f"({from_ns(self.ts_ns[bad]).isoformat()} -> "
                    f"{from_ns(self.ts_ns[bad + 1]).isoformat()})"
                )

        for name in self.NON_NEGATIVE:
            values = self._cols.get(name)
            if values is not None and values.size and np.nanmin(values) < 0:
                raise SchemaError(f"{type(self).__name__}: column {name!r} has negative values")

    # --- access --------------------------------------------------------

    def __len__(self) -> int:
        return int(len(self.ts_ns))

    def __bool__(self) -> bool:
        return len(self.ts_ns) > 0

    def has(self, name: str) -> bool:
        return name in self._cols

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(sorted(self._cols))

    def col(self, name: str) -> np.ndarray:
        """Read-only view of a whole column.

        Callers above the data layer must NOT use this -- it spans the entire
        history, including bars after `now`. `MarketView` is the only
        sanctioned reader, and it always slices first.
        """
        try:
            return _readonly(self._cols[name])
        except KeyError:
            raise SchemaError(
                f"{type(self).__name__} {self.symbol}: no column {name!r}; "
                f"present: {self.columns}"
            ) from None

    def window(self, name: str, start: int, end: int) -> np.ndarray:
        """Read-only slice `[start:end)` of a column, with clamped bounds.

        Clamping is essential, not defensive: a negative `start` would wrap
        to the END of the array under NumPy slicing semantics and hand the
        caller future data. `MarketView` relies on this.
        """
        n = len(self)
        start = max(0, min(int(start), n))
        end = max(start, min(int(end), n))
        return _readonly(self._cols[name][start:end])

    def ts_window(self, start: int, end: int) -> np.ndarray:
        n = len(self)
        start = max(0, min(int(start), n))
        end = max(start, min(int(end), n))
        return _readonly(self.ts_ns[start:end])

    def visible_count(self, cutoff_ns: int) -> int:
        """Rows whose timestamp is <= `cutoff_ns`.

        `side="right"` makes an observation timestamped exactly at the
        cutoff visible: at the instant a bar closes, it is known.
        """
        return int(np.searchsorted(self.ts_ns, int(cutoff_ns), side="right"))

    def index_at_or_before(self, cutoff_ns: int) -> int:
        """Index of the newest row at or before the cutoff, or -1 if none."""
        return self.visible_count(cutoff_ns) - 1

    @property
    def first_ts(self) -> datetime | None:
        return from_ns(self.ts_ns[0]) if len(self) else None

    @property
    def last_ts(self) -> datetime | None:
        return from_ns(self.ts_ns[-1]) if len(self) else None

    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(from_ns(v) for v in self.ts_ns)

    def slice_rows(self, start: int, end: int):
        """A new series containing rows `[start:end)`."""
        n = len(self)
        start = max(0, min(int(start), n))
        end = max(start, min(int(end), n))
        return type(self)(
            symbol=self.symbol,
            ts_ns=self.ts_ns[start:end],
            columns={name: values[start:end] for name, values in self._cols.items()},
            meta=self.meta,
            validate=False,
        )

    def slice_range(self, start_ns: int | None, end_ns: int | None):
        """Rows in the half-open timestamp interval [start_ns, end_ns)."""
        lo = 0 if start_ns is None else int(np.searchsorted(self.ts_ns, int(start_ns), side="left"))
        hi = len(self) if end_ns is None else int(
            np.searchsorted(self.ts_ns, int(end_ns), side="left")
        )
        return self.slice_rows(lo, hi)

    def describe(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "rows": len(self),
            "first_ts": self.first_ts.isoformat() if self.first_ts else None,
            "last_ts": self.last_ts.isoformat() if self.last_ts else None,
            "columns": self.columns,
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }


# ---------------------------------------------------------------------------
# concrete series
# ---------------------------------------------------------------------------


class BarSeries(ColumnSeries):
    """OHLCV bars. `ts_ns` is the bar CLOSE time."""

    REQUIRED = ("open", "high", "low", "close", "volume")
    OPTIONAL = ("trades", "vwap")
    NON_NEGATIVE = ("volume",)

    def __init__(self, symbol, ts_ns, columns, interval_seconds=None, meta=None, validate=True):
        combined = dict(meta or {})
        if interval_seconds is not None:
            combined["interval_seconds"] = int(interval_seconds)
        if "interval_seconds" not in combined:
            raise SchemaError("BarSeries requires interval_seconds")
        super().__init__(symbol, ts_ns, columns, meta=combined, validate=validate)

    @property
    def interval_seconds(self) -> int:
        return int(self.meta["interval_seconds"])

    def _validate(self) -> None:
        super()._validate()
        o, h, l, c = (self._cols[k] for k in ("open", "high", "low", "close"))
        if len(self) == 0:
            return
        for name, values in (("open", o), ("high", h), ("low", l), ("close", c)):
            if not np.all(np.isfinite(values)):
                raise SchemaError(
                    f"BarSeries {self.symbol}: column {name!r} contains non-finite values "
                    f"at rows {np.flatnonzero(~np.isfinite(values))[:5].tolist()}"
                )
        bad = ~((h >= l) & (l <= o) & (o <= h) & (l <= c) & (c <= h))
        if np.any(bad):
            rows = np.flatnonzero(bad)[:5].tolist()
            raise SchemaError(
                f"BarSeries {self.symbol}: OHLC ordering violated at rows {rows} "
                f"(first: o={o[rows[0]]} h={h[rows[0]]} l={l[rows[0]]} c={c[rows[0]]})"
            )

    def bar_at(self, index: int) -> Bar:
        if not 0 <= index < len(self):
            raise IndexError(f"bar index {index} out of range for {len(self)} bars")
        return Bar(
            symbol=self.symbol,
            close_ts=from_ns(self.ts_ns[index]),
            open=float(self._cols["open"][index]),
            high=float(self._cols["high"][index]),
            low=float(self._cols["low"][index]),
            close=float(self._cols["close"][index]),
            volume=float(self._cols["volume"][index]),
            interval_seconds=self.interval_seconds,
            trades=int(self._cols["trades"][index]) if self.has("trades") else None,
            vwap=float(self._cols["vwap"][index]) if self.has("vwap") else None,
        )

    @classmethod
    def from_bars(cls, bars: Sequence[Bar]) -> "BarSeries":
        if not bars:
            raise SchemaError("cannot build a BarSeries from zero bars (symbol unknown)")
        symbols = {b.symbol for b in bars}
        if len(symbols) > 1:
            raise SchemaError(f"bars span multiple symbols: {sorted(symbols)}")
        intervals = {b.interval_seconds for b in bars}
        if len(intervals) > 1:
            raise SchemaError(f"bars span multiple intervals: {sorted(intervals)}")
        return cls(
            symbol=bars[0].symbol,
            ts_ns=to_ns_array(b.close_ts for b in bars),
            columns={
                "open": np.fromiter((b.open for b in bars), dtype=np.float64, count=len(bars)),
                "high": np.fromiter((b.high for b in bars), dtype=np.float64, count=len(bars)),
                "low": np.fromiter((b.low for b in bars), dtype=np.float64, count=len(bars)),
                "close": np.fromiter((b.close for b in bars), dtype=np.float64, count=len(bars)),
                "volume": np.fromiter((b.volume for b in bars), dtype=np.float64, count=len(bars)),
            },
            interval_seconds=bars[0].interval_seconds,
        )

    def to_bars(self) -> tuple[Bar, ...]:
        return tuple(self.bar_at(i) for i in range(len(self)))

    @classmethod
    def empty(cls, symbol: str, interval_seconds: int) -> "BarSeries":
        return cls(
            symbol=symbol,
            ts_ns=np.empty(0, dtype=np.int64),
            columns={k: np.empty(0, dtype=np.float64) for k in cls.REQUIRED},
            interval_seconds=interval_seconds,
        )


class QuoteSeries(ColumnSeries):
    """Top-of-book snapshots."""

    REQUIRED = ("bid", "ask", "bid_size", "ask_size")
    NON_NEGATIVE = ("bid_size", "ask_size")

    def quote_at(self, index: int) -> QuoteSnapshot:
        if not 0 <= index < len(self):
            raise IndexError(f"quote index {index} out of range for {len(self)} quotes")
        return QuoteSnapshot(
            symbol=self.symbol,
            ts=from_ns(self.ts_ns[index]),
            bid=float(self._cols["bid"][index]),
            ask=float(self._cols["ask"][index]),
            bid_size=float(self._cols["bid_size"][index]),
            ask_size=float(self._cols["ask_size"][index]),
        )

    def crossed_count(self) -> int:
        """Quotes where ask < bid. Allowed to exist (they occur in real
        feeds) but counted, because a high rate means a broken feed."""
        if not len(self):
            return 0
        return int(np.count_nonzero(self._cols["ask"] < self._cols["bid"]))


class TickSeries(ColumnSeries):
    """Per-bar tick aggregation with aggressor classification."""

    REQUIRED = ("buy_volume", "sell_volume")
    OPTIONAL = ("unclassified_volume", "buy_trades", "sell_trades", "max_trade_size")
    NON_NEGATIVE = ("buy_volume", "sell_volume", "unclassified_volume")

    @property
    def classification_method(self) -> str:
        return str(self.meta.get("classification_method", "unknown"))

    def tick_at(self, index: int) -> TickAggregate:
        if not 0 <= index < len(self):
            raise IndexError(f"tick index {index} out of range for {len(self)} rows")

        def opt(name: str, default: float = 0.0) -> float:
            return float(self._cols[name][index]) if self.has(name) else default

        return TickAggregate(
            symbol=self.symbol,
            close_ts=from_ns(self.ts_ns[index]),
            buy_volume=float(self._cols["buy_volume"][index]),
            sell_volume=float(self._cols["sell_volume"][index]),
            unclassified_volume=opt("unclassified_volume"),
            buy_trades=int(opt("buy_trades")),
            sell_trades=int(opt("sell_trades")),
            max_trade_size=opt("max_trade_size"),
            classification_method=self.classification_method,
        )

    def coverage(self) -> float:
        """Share of volume with a known aggressor, across the series."""
        if not len(self):
            return 0.0
        classified = float(np.sum(self._cols["buy_volume"]) + np.sum(self._cols["sell_volume"]))
        unclassified = (
            float(np.sum(self._cols["unclassified_volume"])) if self.has("unclassified_volume") else 0.0
        )
        total = classified + unclassified
        return round(classified / total, 6) if total > 0 else 0.0


class OptionsSeries(ColumnSeries):
    """Chain-level options aggregates for one underlying."""

    REQUIRED = ("call_volume", "put_volume", "call_premium", "put_premium", "call_oi", "put_oi")
    OPTIONAL = (
        "call_oi_change",
        "put_oi_change",
        "delta_weighted_call_volume",
        "delta_weighted_put_volume",
        "atm_iv",
        "iv_25d_call",
        "iv_25d_put",
        "gamma_exposure_proxy",
        "underlying_price",
    )
    NON_NEGATIVE = ("call_volume", "put_volume", "call_premium", "put_premium", "call_oi", "put_oi")

    @property
    def is_intraday(self) -> bool:
        """False for end-of-day chain snapshots.

        Drives the DEGRADED grade in the quality layer: EOD data can express
        positioning but cannot express flow timing.
        """
        return bool(self.meta.get("is_intraday", False))

    @property
    def source(self) -> str:
        return str(self.meta.get("source", "unknown"))

    def snapshot_at(self, index: int) -> OptionsSnapshot:
        if not 0 <= index < len(self):
            raise IndexError(f"options index {index} out of range for {len(self)} rows")

        def opt(name: str) -> float | None:
            if not self.has(name):
                return None
            value = float(self._cols[name][index])
            return None if not np.isfinite(value) else value

        def req(name: str) -> float:
            return float(self._cols[name][index])

        return OptionsSnapshot(
            symbol=self.symbol,
            ts=from_ns(self.ts_ns[index]),
            call_volume=req("call_volume"),
            put_volume=req("put_volume"),
            call_premium=req("call_premium"),
            put_premium=req("put_premium"),
            call_oi=req("call_oi"),
            put_oi=req("put_oi"),
            call_oi_change=opt("call_oi_change") or 0.0,
            put_oi_change=opt("put_oi_change") or 0.0,
            delta_weighted_call_volume=opt("delta_weighted_call_volume") or 0.0,
            delta_weighted_put_volume=opt("delta_weighted_put_volume") or 0.0,
            atm_iv=opt("atm_iv"),
            iv_25d_call=opt("iv_25d_call"),
            iv_25d_put=opt("iv_25d_put"),
            gamma_exposure_proxy=opt("gamma_exposure_proxy"),
            underlying_price=opt("underlying_price"),
            is_intraday=self.is_intraday,
            source=self.source,
        )
