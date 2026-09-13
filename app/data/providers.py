"""Where candles come from.

Three providers, all behind one interface so the signal engine and the
backtester neither know nor care which is in use:

  CsvCandleProvider        historical data you supply, for backtesting
  CcxtCandleProvider       real OHLCV from an exchange (needs ccxt)
  SyntheticCandleProvider  reproducible generated markets, for tests

The synthetic provider is not a toy. Backtest results are only meaningful
if you can also show the engine behaves correctly on a market whose shape
you chose deliberately — a known uptrend, a known crash, a known chop — and
that requires generating one.
"""
from __future__ import annotations

import csv
import datetime as dt
import random
from abc import ABC, abstractmethod
from pathlib import Path

from app.data.candles import Candle, CandleSeries, Timeframe


class CandleProvider(ABC):
    @abstractmethod
    def fetch(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> CandleSeries: ...


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class CsvCandleProvider(CandleProvider):
    """Reads `<directory>/<symbol>_<timeframe>.csv`.

    Expected header: timestamp,open,high,low,close,volume
    `timestamp` may be unix seconds, unix milliseconds, or ISO-8601.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def path_for(self, symbol: str, timeframe: Timeframe) -> Path:
        return self.directory / f"{symbol}_{timeframe.value}.csv"

    def fetch(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> CandleSeries:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(f"no candle file for {symbol} {timeframe.value}: {path}")

        candles: list[Candle] = []
        with path.open(newline="", encoding="utf-8") as fh:
            for row_number, row in enumerate(csv.DictReader(fh), start=2):
                try:
                    candles.append(
                        Candle(
                            timestamp=_parse_timestamp(row["timestamp"]),
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row.get("volume") or 0),
                        )
                    )
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"{path}:{row_number} is not a valid candle row: {exc}") from exc

        series = CandleSeries(symbol, timeframe, candles)
        return series.tail(limit) if limit else series


def _parse_timestamp(raw: str) -> dt.datetime:
    text = str(raw).strip()
    try:
        number = float(text)
        # Anything past ~2001 in seconds is beyond 1e9; treat 1e11+ as millis.
        seconds = number / 1000 if number > 1e11 else number
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except ValueError:
        pass
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# ccxt
# ---------------------------------------------------------------------------

class CcxtCandleProvider(CandleProvider):
    """Real OHLCV from a centralized exchange.

    Only usable for symbols an exchange actually lists. On-chain memecoins
    generally are not, which is why CSV and synthetic exist.
    """

    def __init__(self, exchange_id: str = "binance", quote: str = "USDT"):
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "ccxt is not installed - run `pip install -r requirements-live.txt`"
            ) from exc
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"ccxt has no exchange named {exchange_id!r}")
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.quote = quote

    def fetch(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> CandleSeries:
        pair = symbol if "/" in symbol else f"{symbol}/{self.quote}"
        rows = self.exchange.fetch_ohlcv(pair, timeframe=timeframe.value, limit=limit)
        candles = [
            Candle(
                timestamp=dt.datetime.fromtimestamp(row[0] / 1000, tz=dt.timezone.utc),
                open=float(row[1]), high=float(row[2]), low=float(row[3]),
                close=float(row[4]), volume=float(row[5]),
            )
            for row in rows
        ]
        return CandleSeries(symbol, timeframe, candles)


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------

class SyntheticCandleProvider(CandleProvider):
    """Deterministic generated markets of a chosen shape.

    Seeded, so a given (seed, regime) always produces the same series and a
    failing test stays reproducible.

    Regimes: "bull", "bear", "sideways", "high_volatility", "low_volatility",
    "crash", "pump".
    """

    REGIMES = ("bull", "bear", "sideways", "high_volatility", "low_volatility", "crash", "pump")

    def __init__(self, regime: str = "sideways", seed: int = 42, start_price: float = 1.0):
        if regime not in self.REGIMES:
            raise ValueError(f"unknown regime {regime!r}, expected one of {self.REGIMES}")
        self.regime = regime
        self.seed = seed
        self.start_price = start_price

    def _parameters(self) -> tuple[float, float]:
        """(per-candle drift, per-candle volatility)."""
        return {
            "bull": (0.0035, 0.012),
            "bear": (-0.0030, 0.014),
            "sideways": (0.0000, 0.008),
            "high_volatility": (0.0005, 0.055),
            "low_volatility": (0.0002, 0.002),
            "crash": (-0.0150, 0.040),
            "pump": (0.0180, 0.045),
        }[self.regime]

    def fetch(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> CandleSeries:
        rng = random.Random(f"{self.seed}-{symbol}-{self.regime}-{timeframe.value}")
        drift, volatility = self._parameters()

        end = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        start = end - dt.timedelta(seconds=timeframe.seconds * limit)

        candles: list[Candle] = []
        price = self.start_price
        for i in range(limit):
            step = drift + rng.gauss(0, volatility)
            open_price = price
            close_price = max(open_price * (1 + step), 1e-12)

            # Wick beyond the body by a fraction of the move.
            spread = abs(close_price - open_price) + open_price * volatility * abs(rng.gauss(0, 0.6))
            high = max(open_price, close_price) + spread * rng.random()
            low = max(min(open_price, close_price) - spread * rng.random(), 1e-12)

            # Volume rises with the size of the move, which is what makes
            # relative-volume and spike detection meaningful on this data.
            base_volume = 100_000
            move_factor = 1 + abs(step) / max(volatility, 1e-9)
            volume = base_volume * move_factor * (0.6 + rng.random() * 0.8)

            candles.append(
                Candle(
                    timestamp=start + dt.timedelta(seconds=timeframe.seconds * i),
                    open=open_price, high=high, low=low, close=close_price,
                    volume=volume,
                )
            )
            price = close_price

        return CandleSeries(symbol, timeframe, candles)


def resample(series: CandleSeries, target: Timeframe) -> CandleSeries:
    """Aggregate a series onto a slower timeframe.

    Used for multi-timeframe confirmation, so a single fetch of fine-grained
    candles can answer "what is the higher timeframe doing?" without a second
    round trip. Buckets are aligned to epoch multiples of the target
    interval, and a trailing partial bucket is dropped — including a bar that
    has not finished forming would be look-ahead.
    """
    if target.seconds < series.timeframe.seconds:
        raise ValueError(
            f"cannot resample {series.timeframe.value} up to the faster {target.value}"
        )
    if target.seconds == series.timeframe.seconds:
        return series

    buckets: dict[int, list[Candle]] = {}
    for candle in series:
        key = int(candle.timestamp.timestamp()) // target.seconds
        buckets.setdefault(key, []).append(candle)

    expected = target.seconds // series.timeframe.seconds
    out: list[Candle] = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda c: c.timestamp)
        if len(group) < expected:
            continue  # partial bucket: either a gap, or still forming
        out.append(
            Candle(
                timestamp=dt.datetime.fromtimestamp(key * target.seconds, tz=dt.timezone.utc),
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
            )
        )
    return CandleSeries(series.symbol, target, out)
