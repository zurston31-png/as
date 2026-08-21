"""Early-signal feature extraction.

Every feature here is either computed from data the bot genuinely has, or
marked UNAVAILABLE. Nothing is estimated into existence.

WHERE EACH KIND OF FEATURE COMES FROM, AND WHY IT MATTERS

  CANDLES (1m / 5m / 15m OHLCV, GeckoTerminal)
      volume acceleration, price acceleration, compression, higher lows,
      VWAP behaviour, EMA/RSI/MACD acceleration, breakout proximity,
      relative volume.

  SUCCESSIVE SNAPSHOTS (models.TokenObservation)
      transaction rate and its change, buy-pressure change and
      persistence, liquidity growth. These have NO other source:
      DexScreener reports transaction counts only over 1h and 24h windows,
      so "transactions per minute, and is that rate rising?" is
      unanswerable from any single response. It only becomes measurable by
      differencing observations the bot stored itself.

  NOT AVAILABLE AT ALL
      unique buyers, new buyers, repeat buyers, wallet-level
      concentration, real order-book depth. These need an indexer or a
      paid wallet-level feed that this bot does not have. They are
      reported as unavailable rather than approximated, because a
      transaction count is not a participant count - one wallet can
      produce two hundred transactions, and treating that as two hundred
      participants would invert the exact signal the feature exists to
      detect.

Every feature carries `available`. A feature with no data never scores as
neutral-and-fine; the score treats missing input as missing, and enough of
it marks the whole score unreliable.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from app.data.candles import CandleSeries
from app.services.price_feed import MarketSnapshot
from app.signals import indicators as ind

logger = logging.getLogger(__name__)

# Wallet-level features the bot cannot obtain. Named explicitly so the
# reason they are missing is documented rather than inferred from silence.
UNAVAILABLE_FEATURES: dict[str, str] = {
    "unique_buyers": "needs a wallet-level indexer; a transaction count is not a participant count",
    "new_buyers": "needs a wallet-level indexer",
    "repeat_buyers": "needs a wallet-level indexer",
    "wallet_concentration": "needs holder-distribution data beyond the top-10 the rug engine reads",
    "order_book_depth": "AMMs have no order book; transaction flow is a proxy and is labelled as one",
    "social_activity": "no trustworthy provider configured",
}


@dataclass
class Feature:
    """One measured quantity, with its provenance attached."""

    name: str
    value: float | None
    available: bool
    detail: str = ""
    source: str = ""          # "candles" | "observations" | "snapshot"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 6) if self.value is not None else None,
            "available": self.available,
            "detail": self.detail,
            "source": self.source,
        }


def _missing(name: str, why: str, source: str = "") -> Feature:
    return Feature(name, None, False, why, source)


@dataclass
class EarlyFeatures:
    features: dict[str, Feature] = field(default_factory=dict)

    def add(self, feature: Feature) -> None:
        self.features[feature.name] = feature

    def get(self, name: str) -> Feature:
        return self.features.get(name) or _missing(name, "not computed")

    def value(self, name: str, default: float | None = None) -> float | None:
        feature = self.features.get(name)
        return feature.value if (feature and feature.available) else default

    @property
    def available_names(self) -> list[str]:
        return sorted(n for n, f in self.features.items() if f.available)

    @property
    def missing_names(self) -> list[str]:
        return sorted(n for n, f in self.features.items() if not f.available)

    def as_dict(self) -> dict:
        return {name: f.as_dict() for name, f in sorted(self.features.items())}

    @classmethod
    def from_dict(cls, payload: dict | None) -> "EarlyFeatures":
        """Rebuild a feature set from what was stored at signal time.

        This is what makes the early-signal ablation honest. Re-extracting
        features today would score the token on TODAY's candles and
        today's snapshots, which is look-ahead: the question is what the
        engine could have known at the moment it decided, and only the
        stored row knows that.

        `available` is read from the payload rather than inferred from the
        value being non-null. A feature can legitimately be available with
        a value of 0.0, and treating 0.0 as "missing" would silently move
        weight out of the missing-data budget and make an unreliable score
        look reliable.
        """
        features = cls()
        for name, raw in (payload or {}).items():
            if not isinstance(raw, dict):
                continue
            features.add(
                Feature(
                    name=raw.get("name", name),
                    value=raw.get("value"),
                    available=bool(raw.get("available")),
                    detail=raw.get("detail", ""),
                    source=raw.get("source", ""),
                )
            )
        return features


# ---------------------------------------------------------------------------
# candle-derived
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def volume_acceleration(series: CandleSeries) -> list[Feature]:
    """Is volume rising, and is it rising STEADILY or in one burst?

    Both are reported, because they mean opposite things. A steady climb is
    participation arriving; a single bar carrying the whole increase is one
    actor, and the price it created will not survive them leaving.
    """
    out: list[Feature] = []
    volumes = series.volumes
    if len(volumes) < 30:
        return [
            _missing("volume_accel_short", "fewer than 30 bars of history", "candles"),
            _missing("volume_accel_medium", "fewer than 30 bars of history", "candles"),
            _missing("volume_steadiness", "fewer than 30 bars of history", "candles"),
        ]

    recent_5 = sum(volumes[-5:])
    previous_5 = sum(volumes[-10:-5])
    recent_15 = sum(volumes[-15:])
    baseline_per_bar = sum(volumes[-30:]) / 30

    short = _safe_ratio(recent_5, previous_5)
    out.append(
        Feature("volume_accel_short", short, short is not None,
                f"last 5 bars are {short:.2f}x the previous 5" if short else "no previous volume",
                "candles")
    )

    medium = _safe_ratio(recent_15 / 15, baseline_per_bar)
    out.append(
        Feature("volume_accel_medium", medium, medium is not None,
                f"last 15 bars average {medium:.2f}x the 30-bar baseline" if medium else "no baseline",
                "candles")
    )

    # Steadiness: how much of the recent surge sits in its single biggest
    # bar. Low is good. 1.0 means one bar is the entire window.
    window = volumes[-5:]
    total = sum(window)
    if total > 0:
        concentration = max(window) / total
        out.append(
            Feature("volume_steadiness", 1.0 - concentration, True,
                    f"biggest of the last 5 bars is {concentration:.0%} of their volume",
                    "candles")
        )
    else:
        out.append(_missing("volume_steadiness", "no volume in the last 5 bars", "candles"))

    return out


def price_acceleration(series: CandleSeries) -> list[Feature]:
    """Recent returns over several windows, and whether the move is
    CONTROLLED or already violent."""
    out: list[Feature] = []
    closes = series.closes
    # 61, not 60. The longest lookback below reads closes[-lookback - 1]
    # with lookback=60, which is closes[-61] - and that raises IndexError on
    # a list of exactly 60. `extract` calls this outside any handler, so the
    # off-by-one took out the whole feature set for a token sitting on the
    # boundary. breakout_proximity below already gets this right with its
    # `lookback + 1` guard; the two now agree.
    if len(closes) < 61:
        return [
            _missing("return_short", "fewer than 61 bars", "candles"),
            _missing("return_medium", "fewer than 61 bars", "candles"),
            _missing("return_long", "fewer than 61 bars", "candles"),
            _missing("acceleration_smoothness", "fewer than 61 bars", "candles"),
        ]

    now = closes[-1]
    for name, lookback in (("return_short", 5), ("return_medium", 15), ("return_long", 60)):
        past = closes[-lookback - 1]
        value = (now / past - 1) * 100 if past > 0 else None
        out.append(
            Feature(name, value, value is not None,
                    f"{value:+.2f}% over {lookback} bars" if value is not None else "no valid base price",
                    "candles")
        )

    # Smoothness: the share of the last 15 bars that closed up. A clean
    # advance grinds; a pump is one or two bars and then nothing.
    window = closes[-16:]
    ups = sum(1 for a, b in zip(window, window[1:]) if b > a)
    out.append(
        Feature("acceleration_smoothness", ups / 15, True,
                f"{ups} of the last 15 bars closed up", "candles")
    )
    return out


def compression(series: CandleSeries) -> list[Feature]:
    """Is the range narrowing - a coil - and are higher lows forming?"""
    out: list[Feature] = []
    if len(series) < 40:
        return [
            _missing("range_compression", "fewer than 40 bars", "candles"),
            _missing("higher_lows", "fewer than 40 bars", "candles"),
        ]

    highs, lows, closes = series.highs, series.lows, series.closes
    recent_range = (max(highs[-10:]) - min(lows[-10:])) / closes[-1] if closes[-1] > 0 else None
    prior_range = (max(highs[-40:-10]) - min(lows[-40:-10])) / closes[-1] if closes[-1] > 0 else None

    ratio = _safe_ratio(recent_range, prior_range)
    if ratio is not None:
        # Below 1 means the recent range is tighter than the prior one.
        out.append(
            Feature("range_compression", 1.0 - min(ratio, 2.0) / 2.0, True,
                    f"recent 10-bar range is {ratio:.2f}x the prior 30-bar range", "candles")
        )
    else:
        out.append(_missing("range_compression", "no usable range", "candles"))

    swing_idx = ind.swing_lows(lows, lookback=3)
    if len(swing_idx) >= 3:
        last_three = [lows[i] for i in swing_idx[-3:]]
        rising = sum(1 for a, b in zip(last_three, last_three[1:]) if b > a)
        out.append(
            Feature("higher_lows", rising / 2, True,
                    f"{rising} of the last 2 swing-low transitions were higher", "candles")
        )
    else:
        out.append(_missing("higher_lows", "fewer than 3 swing lows to compare", "candles"))

    return out


def momentum_features(series: CandleSeries) -> list[Feature]:
    """EMA / RSI / MACD, read as ACCELERATION rather than as levels.

    Deliberately not "is RSI high". A high RSI on a token that has already
    tripled is a reason to stay away, not a reason to buy - the useful
    reading is RSI crossing UP through neutral, which is a different event
    entirely.
    """
    out: list[Feature] = []
    closes = series.closes
    if len(closes) < 60:
        return [
            _missing(n, "fewer than 60 bars", "candles")
            for n in ("ema_separation", "ema_slope", "rsi_crossing_up", "rsi_level",
                      "macd_histogram_expanding", "vwap_position")
        ]

    ema9 = ind.ema(closes, 9)
    ema21 = ind.ema(closes, 21)
    if ema9[-1] is not None and ema21[-1] is not None and ema21[-1] > 0:
        separation = (ema9[-1] / ema21[-1] - 1) * 100
        out.append(
            Feature("ema_separation", separation, True,
                    f"EMA9 is {separation:+.2f}% from EMA21", "candles")
        )
        if ema9[-6] is not None and ema9[-6] > 0:
            slope = (ema9[-1] / ema9[-6] - 1) * 100
            out.append(Feature("ema_slope", slope, True, f"EMA9 rose {slope:+.2f}% over 5 bars", "candles"))
        else:
            out.append(_missing("ema_slope", "not enough EMA history", "candles"))
    else:
        out.append(_missing("ema_separation", "EMA not yet defined", "candles"))
        out.append(_missing("ema_slope", "EMA not yet defined", "candles"))

    rsi = ind.rsi(closes, 14)
    if rsi[-1] is not None and rsi[-4] is not None:
        crossing = 1.0 if (rsi[-4] < 50 <= rsi[-1]) else 0.0
        out.append(
            Feature("rsi_crossing_up", crossing, True,
                    f"RSI {rsi[-4]:.0f} -> {rsi[-1]:.0f}"
                    + (" (crossed up through 50)" if crossing else ""), "candles")
        )
        out.append(Feature("rsi_level", rsi[-1], True, f"RSI {rsi[-1]:.0f}", "candles"))
    else:
        out.append(_missing("rsi_crossing_up", "RSI not yet defined", "candles"))
        out.append(_missing("rsi_level", "RSI not yet defined", "candles"))

    macd_line, signal_line, histogram = ind.macd(closes)
    if histogram[-1] is not None and histogram[-4] is not None:
        expanding = 1.0 if histogram[-1] > histogram[-4] and histogram[-1] > 0 else 0.0
        out.append(
            Feature("macd_histogram_expanding", expanding, True,
                    f"MACD histogram {histogram[-4]:+.6f} -> {histogram[-1]:+.6f}", "candles")
        )
    else:
        out.append(_missing("macd_histogram_expanding", "MACD not yet defined", "candles"))

    vwap = ind.vwap(series.highs, series.lows, closes, series.volumes)
    if vwap[-1] is not None and vwap[-1] > 0:
        position = (closes[-1] / vwap[-1] - 1) * 100
        out.append(
            Feature("vwap_position", position, True,
                    f"price is {position:+.2f}% versus VWAP", "candles")
        )
    else:
        out.append(_missing("vwap_position", "VWAP not defined (no volume)", "candles"))

    return out


def breakout_proximity(series: CandleSeries, lookback: int = 60) -> Feature:
    """How close price is to its recent high.

    Near it and rising is a setup. Far ABOVE it is not an opportunity, it
    is a move that already happened - so the value is signed and the score
    reads the two directions differently.
    """
    if len(series) < lookback + 1:
        return _missing("breakout_proximity", f"fewer than {lookback} bars", "candles")
    prior_high = max(series.highs[-lookback - 1:-1])
    close = series.closes[-1]
    if prior_high <= 0:
        return _missing("breakout_proximity", "no valid prior high", "candles")
    distance = (close / prior_high - 1) * 100
    return Feature(
        "breakout_proximity", distance, True,
        f"price is {distance:+.2f}% versus the {lookback}-bar high", "candles",
    )


def relative_volume_feature(series: CandleSeries) -> Feature:
    """Recent volume against the token's OWN baseline.

    Against its own, never against a fixed dollar figure: $50k in an hour
    is enormous for one token and background noise for another.
    """
    rvol = ind.relative_volume(series.volumes, period=20)
    if rvol[-1] is None:
        return _missing("relative_volume", "fewer than 20 bars of volume", "candles")
    return Feature("relative_volume", rvol[-1], True,
                   f"{rvol[-1]:.2f}x its own 20-bar average volume", "candles")


# ---------------------------------------------------------------------------
# observation-derived (successive snapshots)
# ---------------------------------------------------------------------------

def _aware(moment: dt.datetime) -> dt.datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def flow_features(observations: list) -> list[Feature]:
    """Transaction rate, buy pressure and liquidity, from stored snapshots.

    `observations` are models.TokenObservation rows, oldest first. At least
    two are needed to difference anything, and they must be far enough
    apart that the difference is signal rather than rounding - two
    snapshots ten seconds apart differ by noise.
    """
    names = ("txn_rate_change", "buy_pressure", "buy_pressure_change",
             "buy_pressure_persistence", "liquidity_growth", "liquidity_stability")
    if len(observations) < 2:
        return [
            _missing(n, "fewer than 2 stored observations - flow cannot be differenced", "observations")
            for n in names
        ]

    ordered = sorted(observations, key=lambda o: _aware(o.observed_at))
    latest, previous = ordered[-1], ordered[-2]
    elapsed_minutes = (_aware(latest.observed_at) - _aware(previous.observed_at)).total_seconds() / 60

    out: list[Feature] = []

    if elapsed_minutes < 1:
        out.append(
            _missing("txn_rate_change",
                     f"last two observations are only {elapsed_minutes * 60:.0f}s apart - "
                     "the difference would be rounding, not flow", "observations")
        )
    elif latest.buys_1h is None or previous.buys_1h is None:
        out.append(_missing("txn_rate_change", "transaction counts not reported", "observations"))
    else:
        now_total = (latest.buys_1h or 0) + (latest.sells_1h or 0)
        then_total = (previous.buys_1h or 0) + (previous.sells_1h or 0)
        change = _safe_ratio(now_total, then_total)
        out.append(
            Feature("txn_rate_change", change, change is not None,
                    f"1h transaction count {then_total} -> {now_total} over {elapsed_minutes:.0f}m",
                    "observations")
        )

    def pressure(obs) -> float | None:
        buys, sells = obs.buys_1h, obs.sells_1h
        if buys is None or sells is None:
            return None
        total = buys + sells
        if total < 20:
            return None      # too few trades for a ratio to mean anything
        return buys / total

    now_pressure = pressure(latest)
    then_pressure = pressure(previous)

    if now_pressure is None:
        out.append(_missing("buy_pressure", "too few transactions to read a ratio", "observations"))
        out.append(_missing("buy_pressure_change", "too few transactions", "observations"))
    else:
        out.append(
            Feature("buy_pressure", now_pressure, True,
                    f"{now_pressure:.0%} of 1h trades are buys", "observations")
        )
        if then_pressure is None:
            out.append(_missing("buy_pressure_change", "no comparable earlier reading", "observations"))
        else:
            delta = now_pressure - then_pressure
            out.append(
                Feature("buy_pressure_change", delta, True,
                        f"buy share moved {delta:+.1%} since the previous observation", "observations")
            )

    # Persistence: how many of the recent observations had buy-dominant
    # flow. One reading is a moment; several in a row is a condition.
    readings = [pressure(o) for o in ordered[-5:]]
    usable = [r for r in readings if r is not None]
    if len(usable) < 3:
        out.append(
            _missing("buy_pressure_persistence",
                     f"only {len(usable)} usable readings - persistence needs at least 3",
                     "observations")
        )
    else:
        dominant = sum(1 for r in usable if r > 0.5)
        out.append(
            Feature("buy_pressure_persistence", dominant / len(usable), True,
                    f"buy-dominant in {dominant} of the last {len(usable)} observations",
                    "observations")
        )

    if latest.liquidity_usd is None or previous.liquidity_usd is None:
        out.append(_missing("liquidity_growth", "liquidity not reported", "observations"))
        out.append(_missing("liquidity_stability", "liquidity not reported", "observations"))
    else:
        growth = _safe_ratio(latest.liquidity_usd, previous.liquidity_usd)
        out.append(
            Feature("liquidity_growth", growth, growth is not None,
                    f"liquidity ${previous.liquidity_usd:,.0f} -> ${latest.liquidity_usd:,.0f}",
                    "observations")
        )
        # Stability across the whole stored window. A pool whose depth
        # swings violently is not "growing", it is being manipulated.
        depths = [o.liquidity_usd for o in ordered[-6:] if o.liquidity_usd]
        if len(depths) >= 3:
            swing = (max(depths) - min(depths)) / max(depths)
            out.append(
                Feature("liquidity_stability", 1.0 - min(swing, 1.0), True,
                        f"liquidity swung {swing:.0%} across the last {len(depths)} observations",
                        "observations")
            )
        else:
            out.append(_missing("liquidity_stability", "fewer than 3 liquidity readings", "observations"))

    return out


def snapshot_features(market: MarketSnapshot | None) -> list[Feature]:
    """Structural readings available from a single snapshot."""
    if market is None:
        return [
            _missing(n, "no market snapshot", "snapshot")
            for n in ("volume_to_liquidity", "liquidity_to_marketcap", "token_age_hours")
        ]

    out: list[Feature] = []
    vtl = _safe_ratio(market.volume_24h_usd, market.liquidity_usd)
    out.append(
        Feature("volume_to_liquidity", vtl, vtl is not None,
                f"{vtl:.1f}x turnover against pool depth" if vtl else "volume or liquidity missing",
                "snapshot")
    )

    ltm = _safe_ratio(market.liquidity_usd, market.market_cap_usd)
    out.append(
        Feature("liquidity_to_marketcap", ltm, ltm is not None,
                f"pool is {ltm:.1%} of market cap" if ltm else "market cap or liquidity missing",
                "snapshot")
    )

    age = market.age_hours
    out.append(
        Feature("token_age_hours", age, age is not None,
                f"{age:.1f}h old" if age is not None else "pool creation time not reported",
                "snapshot")
    )
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def extract(
    *,
    series: CandleSeries | None,
    market: MarketSnapshot | None,
    observations: list | None = None,
) -> EarlyFeatures:
    """Every early feature the bot can honestly compute for one token."""
    result = EarlyFeatures()

    if series is not None and len(series):
        for feature in volume_acceleration(series):
            result.add(feature)
        for feature in price_acceleration(series):
            result.add(feature)
        for feature in compression(series):
            result.add(feature)
        for feature in momentum_features(series):
            result.add(feature)
        result.add(breakout_proximity(series))
        result.add(relative_volume_feature(series))
    else:
        for name in ("volume_accel_short", "volume_accel_medium", "volume_steadiness",
                     "return_short", "return_medium", "return_long", "acceleration_smoothness",
                     "range_compression", "higher_lows", "ema_separation", "ema_slope",
                     "rsi_crossing_up", "rsi_level", "macd_histogram_expanding",
                     "vwap_position", "breakout_proximity", "relative_volume"):
            result.add(_missing(name, "no candle history available", "candles"))

    for feature in flow_features(observations or []):
        result.add(feature)
    for feature in snapshot_features(market):
        result.add(feature)

    # Wallet-level features are recorded as explicitly unavailable rather
    # than omitted, so the reason is visible on the token page instead of
    # someone later assuming they were forgotten.
    for name, why in UNAVAILABLE_FEATURES.items():
        result.add(_missing(name, why, "unavailable"))

    return result
