"""Cheap pre-screen for discovered tokens, run BEFORE any further network
call is spent on them.

Every field checked here already arrived in the discovery payload
(app/scanner/discovery.py), so rejecting on it costs nothing. That ordering
is the whole point: a scan cycle can surface hundreds of brand-new mints,
and the expensive stages downstream - the rug check (several scanner
lookups) and the signal score (pool resolution + a candle fetch) - should
only ever run on the handful that could plausibly be traded.

Fail-closed on missing data, same rule as everywhere else in this bot: a
token whose liquidity or volume simply wasn't reported is rejected, not
waved through. "The data source didn't say" is not evidence a brand-new
memecoin is safe to buy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.scanner.discovery import DiscoveredToken


@dataclass
class Check:
    """One named pre-screen test and what it decided."""

    name: str
    passed: bool
    reason: str
    value: float | int | None = None      # what was measured, for the audit trail
    threshold: float | int | None = None  # what it was measured against


@dataclass
class FilterVerdict:
    """The overall pre-screen decision, plus every individual check.

    `passed` and `reason` keep the original two-field contract. `checks`
    is the addition, and it exists because a short-circuiting filter can
    only ever report the FIRST reason a token died - which makes the funnel
    useless for tuning. "497 of 500 failed pre-screen" tells you nothing;
    "480 failed liquidity, 15 failed volume, 2 failed age" tells you which
    threshold is actually doing the work.

    Every check runs, because all of the data already arrived in the
    discovery payload - evaluating all of them costs nothing beyond a few
    comparisons.
    """

    passed: bool
    reason: str = ""
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed_names(self) -> list[str]:
        return [c.name for c in self.checks if c.passed]

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "checks": [
                {"name": c.name, "passed": c.passed, "reason": c.reason,
                 "value": c.value, "threshold": c.threshold}
                for c in self.checks
            ],
        }


def _liquidity(token: DiscoveredToken) -> Check:
    floor = settings.SCANNER_MIN_LIQUIDITY_USD
    if token.liquidity_usd is None:
        return Check("liquidity", False, "no liquidity reported by the listing source", None, floor)
    if token.liquidity_usd < floor:
        return Check(
            "liquidity", False,
            f"liquidity ${token.liquidity_usd:,.0f} below scanner minimum ${floor:,.0f}",
            token.liquidity_usd, floor,
        )
    return Check("liquidity", True, f"${token.liquidity_usd:,.0f}", token.liquidity_usd, floor)


def _volume(token: DiscoveredToken) -> Check:
    floor = settings.SCANNER_MIN_VOLUME_24H_USD
    if token.volume_24h_usd is None:
        return Check("volume", False, "no 24h volume reported by the listing source", None, floor)
    if token.volume_24h_usd < floor:
        return Check(
            "volume", False,
            f"24h volume ${token.volume_24h_usd:,.0f} below scanner minimum ${floor:,.0f}",
            token.volume_24h_usd, floor,
        )
    return Check("volume", True, f"${token.volume_24h_usd:,.0f}", token.volume_24h_usd, floor)


def _age(token: DiscoveredToken) -> Check:
    low = settings.SCANNER_MIN_TOKEN_AGE_HOURS
    high = settings.SCANNER_MAX_TOKEN_AGE_HOURS
    hours = token.age_hours
    if hours is None:
        return Check("age", False, "no pool creation time reported - token age unknown", None, low)
    if hours < low:
        return Check(
            "age", False,
            f"pool is {hours:.1f}h old, under the {low:.1f}h minimum (the highest-risk rug window)",
            hours, low,
        )
    if high > 0 and hours > high:
        return Check(
            "age", False,
            f"pool is {hours / 24:.1f}d old, past the {high / 24:.1f}d window this scanner targets",
            hours, high,
        )
    return Check("age", True, f"{hours:.1f}h old", hours, low)


def _transactions(token: DiscoveredToken) -> Check:
    floor = settings.SCANNER_MIN_TXNS_24H
    buys, sells = token.buys_24h, token.sells_24h
    if buys is None or sells is None:
        return Check("transactions", False, "no 24h buy/sell counts reported by the listing source", None, floor)
    total = buys + sells
    if total < floor:
        return Check(
            "transactions", False,
            f"only {total} trades in 24h, under the {floor} minimum", total, floor,
        )
    return Check("transactions", True, f"{total} trades in 24h", total, floor)


def _sell_pressure(token: DiscoveredToken) -> Check:
    limit = settings.SCANNER_MAX_SELL_SHARE
    buys, sells = token.buys_24h, token.sells_24h
    if buys is None or sells is None:
        return Check("sell_pressure", False, "no 24h buy/sell counts reported by the listing source", None, limit)
    total = buys + sells
    if total <= 0:
        return Check("sell_pressure", False, "no trades in 24h - no flow to read", None, limit)
    share = sells / total
    if share >= limit:
        return Check(
            "sell_pressure", False,
            f"{share * 100:.0f}% of 24h trades are sells (limit {limit * 100:.0f}%) - distribution pressure",
            share, limit,
        )
    return Check("sell_pressure", True, f"{share * 100:.0f}% sells", share, limit)


# Order matters only for which reason is reported first; every check runs.
PRESCREEN_CHECKS = (_liquidity, _volume, _age, _transactions, _sell_pressure)


def prescreen(token: DiscoveredToken) -> FilterVerdict:
    """Judge a discovered token on its listing data alone.

    Runs every check rather than stopping at the first failure, so the
    scanner funnel can report how many tokens each individual threshold
    rejects. The headline `reason` is still the first failure, which is the
    one worth showing in a log line.
    """
    checks = [check(token) for check in PRESCREEN_CHECKS]
    failures = [c for c in checks if not c.passed]

    if failures:
        return FilterVerdict(False, failures[0].reason, checks)
    return FilterVerdict(True, "passed scanner pre-screen", checks)
