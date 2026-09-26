"""Data quality gates.

A trading decision is only as trustworthy as the candles behind it. This
module inspects a series and reports what is wrong with it, so the caller
can skip the trade and log a specific reason rather than acting on a gap, a
duplicate, or a stale feed.

Design rule, matching the rug filter: missing or suspect data is never
treated as fine. `assess_quality` returns `tradeable=False` on anything it
cannot vouch for.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from app.data.candles import CandleSeries

# A bar whose range exceeds this multiple of the recent median is treated as
# a feed glitch rather than a real move. Memecoins genuinely do move
# violently, so this is deliberately loose — it is catching decimal-shift
# and bad-tick errors, not volatility.
EXTREME_RANGE_MULTIPLE = 20.0

# How many intervals of silence before the feed counts as stale.
STALE_INTERVALS = 3


@dataclass
class DataQualityReport:
    tradeable: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    candles_checked: int = 0
    gaps: int = 0
    duplicates: int = 0
    malformed: int = 0
    staleness_seconds: float | None = None

    @property
    def summary(self) -> str:
        if self.tradeable and not self.warnings:
            return f"ok ({self.candles_checked} candles)"
        parts = list(self.issues) + [f"(warning) {w}" for w in self.warnings]
        return "; ".join(parts) if parts else "ok"


def assess_quality(
    series: CandleSeries,
    *,
    min_candles: int = 50,
    now: dt.datetime | None = None,
    max_gap_ratio: float = 0.05,
) -> DataQualityReport:
    """Judge whether a series is fit to trade on.

    `min_candles` should be at least the longest indicator lookback you
    intend to use — an EMA200 on 60 candles is not an EMA200.
    """
    report = DataQualityReport(tradeable=True, candles_checked=len(series))

    if len(series) == 0:
        report.tradeable = False
        report.issues.append("no candles available")
        return report

    if len(series) < min_candles:
        report.tradeable = False
        report.issues.append(
            f"only {len(series)} candles, need {min_candles} for the indicators in use"
        )

    candles = series.candles

    # --- malformed bars ---
    malformed = [c for c in candles if not c.is_structurally_valid()]
    report.malformed = len(malformed)
    if malformed:
        report.tradeable = False
        first = malformed[0]
        report.issues.append(
            f"{len(malformed)} malformed candle(s), first at {first.timestamp:%Y-%m-%d %H:%M} "
            f"(o={first.open} h={first.high} l={first.low} c={first.close})"
        )

    # --- duplicates ---
    seen: set[dt.datetime] = set()
    duplicates = 0
    for c in candles:
        if c.timestamp in seen:
            duplicates += 1
        seen.add(c.timestamp)
    report.duplicates = duplicates
    if duplicates:
        report.tradeable = False
        report.issues.append(f"{duplicates} duplicate timestamp(s) - the feed repeated bars")

    # --- gaps ---
    interval = series.timeframe.seconds
    gaps = 0
    missing_bars = 0
    for prev, curr in zip(candles, candles[1:]):
        delta = (curr.timestamp - prev.timestamp).total_seconds()
        if delta > interval * 1.5:
            gaps += 1
            missing_bars += int(round(delta / interval)) - 1
    report.gaps = gaps
    if gaps:
        expected = len(candles) + missing_bars
        ratio = missing_bars / expected if expected else 0
        message = f"{gaps} gap(s) totalling ~{missing_bars} missing candle(s)"
        if ratio > max_gap_ratio:
            report.tradeable = False
            report.issues.append(f"{message} - {ratio:.1%} of the series is absent")
        else:
            report.warnings.append(message)

    # --- staleness ---
    now = now or dt.datetime.now(dt.timezone.utc)
    last = candles[-1]
    age = (now - last.timestamp).total_seconds()
    report.staleness_seconds = age
    if age > interval * STALE_INTERVALS:
        report.tradeable = False
        report.issues.append(
            f"feed is stale: newest candle is {age / 60:.0f} minutes old "
            f"on a {series.timeframe.value} timeframe"
        )

    # --- extreme bars ---
    ranges = sorted(c.range for c in candles if c.range > 0)
    if ranges:
        median_range = ranges[len(ranges) // 2]
        if median_range > 0:
            extreme = [c for c in candles if c.range > median_range * EXTREME_RANGE_MULTIPLE]
            if extreme:
                worst = max(extreme, key=lambda c: c.range)
                report.warnings.append(
                    f"{len(extreme)} candle(s) with an implausible range, worst "
                    f"{worst.range / median_range:.0f}x the median at "
                    f"{worst.timestamp:%Y-%m-%d %H:%M} - possible bad tick"
                )

    # --- flat/dead feed ---
    recent = candles[-min(20, len(candles)):]
    if len(recent) >= 5 and len({c.close for c in recent}) == 1:
        report.tradeable = False
        report.issues.append(
            f"price has not moved across the last {len(recent)} candles - feed likely frozen"
        )

    if all(c.volume == 0 for c in recent):
        report.tradeable = False
        report.issues.append("zero volume across every recent candle - no real trading")

    return report
