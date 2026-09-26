"""Market quality score: 0 (untradeable) - 100 (clean, liquid, real).

Deliberately a THIRD score, separate from both existing ones, because they
answer different questions and conflating them loses information:

    security score   (app/rugcheck/risk_score.py)  "will this rug?"
    signal score     (app/signals/scoring.py)      "is this a good setup?"
    market quality   (here)                        "can I actually trade it?"

A token can be perfectly safe AND show a textbook breakout while being
untradeable in practice: one wash-traded volume spike with nothing behind
it, a pool too thin to exit, 90% of the day's activity in three
transactions. The signal engine reads price and volume shape and will
happily score that highly - it has no concept of whether the volume was
real. This does.

The central rule, and the reason this exists at all: HIGH VOLUME IS NOT
HIGH QUALITY. `volume_concentration` and `volume_consistency` below exist
specifically to catch volume that is large but fake - a huge 24h number
produced by a single burst, or by a handful of enormous prints, scores
WORSE here than a smaller, steadier figure.

Scoring convention matches app/signals/scoring.py: each factor scores 0.0
(bad) to 1.0 (good) with 0.5 meaning "no opinion", carries a
human-readable reason, and a factor with no data is marked unavailable
rather than being scored as if it were fine. If too much is missing the
whole score is flagged unreliable, and the caller treats that as a
rejection - the same fail-closed rule used everywhere else in this bot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.price_feed import MarketSnapshot

NEUTRAL = 0.5
MAX_UNAVAILABLE_WEIGHT = 0.35


@dataclass
class QualityFactor:
    name: str
    score: float
    weight: float
    reason: str
    available: bool = True

    @property
    def points(self) -> float:
        return self.score * self.weight * 100

    @property
    def max_points(self) -> float:
        return self.weight * 100


@dataclass
class MarketQualityScore:
    score: float                 # 0-100, higher = more tradeable
    factors: list[QualityFactor] = field(default_factory=list)
    reliable: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def unavailable(self) -> list[QualityFactor]:
        return [f for f in self.factors if not f.available]

    @property
    def concerns(self) -> list[QualityFactor]:
        return sorted(
            [f for f in self.factors if f.available and f.score < 0.45],
            key=lambda f: f.points,
        )

    def breakdown(self) -> str:
        lines = [f"Market quality {self.score:.1f}/100"]
        if not self.reliable:
            lines.append("  UNRELIABLE: " + "; ".join(self.warnings))
        for f in sorted(self.factors, key=lambda f: f.points, reverse=True):
            marker = " " if f.available else "?"
            lines.append(f"  {marker} {f.name:<24} {f.points:5.1f}/{f.max_points:4.1f}  {f.reason}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "reliable": self.reliable,
            "warnings": list(self.warnings),
            "factors": [
                {
                    "name": f.name, "score": round(f.score, 3), "weight": f.weight,
                    "points": round(f.points, 2), "reason": f.reason, "available": f.available,
                }
                for f in self.factors
            ],
        }


# Liquidity dominates: on a memecoin, everything else is academic if you
# can't get out. The two anti-wash-trading factors together carry more
# weight than raw volume, which is the whole point of the module.
DEFAULT_WEIGHTS: dict[str, float] = {
    "liquidity_depth": 0.24,
    "volume_to_liquidity": 0.16,
    "volume_concentration": 0.14,
    "volume_consistency": 0.12,
    "transaction_activity": 0.12,
    "buy_sell_balance": 0.10,
    "pool_age": 0.07,
    "price_stability": 0.05,
}


def _unavailable(name: str, weight: float, reason: str) -> QualityFactor:
    return QualityFactor(name, NEUTRAL, weight, reason, available=False)


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------

def score_liquidity_depth(market: MarketSnapshot, weight: float, min_liquidity_usd: float) -> QualityFactor:
    if market.liquidity_usd is None:
        return _unavailable("liquidity_depth", weight, "liquidity not reported")
    usd = market.liquidity_usd
    if usd >= min_liquidity_usd * 10:
        return QualityFactor("liquidity_depth", 1.0, weight, f"${usd:,.0f} - deep enough to size into and exit")
    if usd >= min_liquidity_usd * 4:
        return QualityFactor("liquidity_depth", 0.85, weight, f"${usd:,.0f} - comfortable")
    if usd >= min_liquidity_usd:
        return QualityFactor("liquidity_depth", 0.55, weight, f"${usd:,.0f} - workable but tight")
    return QualityFactor("liquidity_depth", 0.1, weight, f"${usd:,.0f} - below the minimum, exits will hurt")


def score_volume_to_liquidity(market: MarketSnapshot, weight: float) -> QualityFactor:
    """Turnover. Some is healthy; an extreme ratio is the classic wash-trade
    fingerprint (enormous volume against a pool too small to support it)."""
    if market.volume_24h_usd is None or not market.liquidity_usd:
        return _unavailable("volume_to_liquidity", weight, "volume or liquidity not reported")
    ratio = market.volume_24h_usd / market.liquidity_usd
    if ratio > 50:
        return QualityFactor("volume_to_liquidity", 0.05, weight,
                             f"{ratio:.0f}x turnover - implausible against this pool, likely wash traded")
    if ratio > 20:
        return QualityFactor("volume_to_liquidity", 0.3, weight, f"{ratio:.0f}x turnover - suspiciously high")
    if ratio >= 1.0:
        return QualityFactor("volume_to_liquidity", 1.0, weight, f"{ratio:.1f}x turnover - healthy activity")
    if ratio >= 0.2:
        return QualityFactor("volume_to_liquidity", 0.7, weight, f"{ratio:.2f}x turnover - moderate")
    return QualityFactor("volume_to_liquidity", 0.2, weight, f"{ratio:.2f}x turnover - stagnant pool")


def score_volume_concentration(market: MarketSnapshot, weight: float) -> QualityFactor:
    """Is the 24h volume a steady stream, or one burst?

    Compares the most recent hour against the 24h average hour. A token
    doing 20x its average hourly volume right now is in a spike; that
    volume may vanish the moment you try to exit into it.
    """
    if market.volume_24h_usd is None or market.volume_1h_usd is None:
        return _unavailable("volume_concentration", weight, "1h vs 24h volume not both reported")
    if market.volume_24h_usd <= 0:
        return _unavailable("volume_concentration", weight, "no 24h volume to compare against")

    average_hour = market.volume_24h_usd / 24
    if average_hour <= 0:
        return _unavailable("volume_concentration", weight, "average hourly volume is zero")
    burst = market.volume_1h_usd / average_hour

    if burst > 12:
        return QualityFactor("volume_concentration", 0.05, weight,
                             f"last hour is {burst:.0f}x the average hour - a spike, not sustained interest")
    if burst > 6:
        return QualityFactor("volume_concentration", 0.3, weight, f"last hour is {burst:.1f}x average - spiky")
    if burst >= 0.5:
        return QualityFactor("volume_concentration", 1.0, weight,
                             f"last hour is {burst:.1f}x average - steady participation")
    return QualityFactor("volume_concentration", 0.35, weight,
                         f"last hour is only {burst:.2f}x average - interest is fading")


def score_volume_consistency(market: MarketSnapshot, weight: float) -> QualityFactor:
    """Average trade size versus pool depth.

    A pool whose typical print is a large slice of its own liquidity is
    being moved by a handful of actors, not traded by a market - and those
    same actors are who you'd be exiting into.
    """
    if market.volume_24h_usd is None or market.buys_24h is None or market.sells_24h is None:
        return _unavailable("volume_consistency", weight, "volume or transaction counts not reported")
    trades = market.buys_24h + market.sells_24h
    if trades <= 0:
        return QualityFactor("volume_consistency", 0.05, weight, "no transactions in 24h")
    if not market.liquidity_usd:
        return _unavailable("volume_consistency", weight, "liquidity not reported")

    average_trade = market.volume_24h_usd / trades
    share_of_pool = average_trade / market.liquidity_usd

    if share_of_pool > 0.10:
        return QualityFactor("volume_consistency", 0.1, weight,
                             f"average trade is {share_of_pool * 100:.1f}% of the pool - a few whales, not a market")
    if share_of_pool > 0.03:
        return QualityFactor("volume_consistency", 0.45, weight,
                             f"average trade is {share_of_pool * 100:.1f}% of the pool - chunky")
    return QualityFactor("volume_consistency", 1.0, weight,
                         f"average trade is {share_of_pool * 100:.2f}% of the pool - well distributed")


def score_transaction_activity(market: MarketSnapshot, weight: float) -> QualityFactor:
    if market.buys_24h is None or market.sells_24h is None:
        return _unavailable("transaction_activity", weight, "transaction counts not reported")
    total = market.buys_24h + market.sells_24h
    if total >= 2_000:
        return QualityFactor("transaction_activity", 1.0, weight, f"{total:,} trades in 24h - busy")
    if total >= 500:
        return QualityFactor("transaction_activity", 0.8, weight, f"{total:,} trades in 24h - active")
    if total >= 100:
        return QualityFactor("transaction_activity", 0.5, weight, f"{total} trades in 24h - modest")
    return QualityFactor("transaction_activity", 0.1, weight, f"only {total} trades in 24h - illiquid in practice")


def score_buy_sell_balance(market: MarketSnapshot, weight: float) -> QualityFactor:
    if market.buys_24h is None or market.sells_24h is None:
        return _unavailable("buy_sell_balance", weight, "buy/sell counts not reported")
    total = market.buys_24h + market.sells_24h
    if total < 20:
        return _unavailable("buy_sell_balance", weight, f"only {total} trades - too few to read a ratio")
    sell_share = market.sells_24h / total
    if sell_share >= 0.75:
        return QualityFactor("buy_sell_balance", 0.15, weight,
                             f"{sell_share * 100:.0f}% sells - heavy distribution")
    if sell_share <= 0.15:
        return QualityFactor("buy_sell_balance", 0.35, weight,
                             f"only {sell_share * 100:.0f}% sells - one-sided, possibly bot accumulation")
    return QualityFactor("buy_sell_balance", 1.0, weight,
                         f"{sell_share * 100:.0f}% sells / {(1 - sell_share) * 100:.0f}% buys - two-sided market")


def score_pool_age(market: MarketSnapshot, weight: float) -> QualityFactor:
    if market.pair_created_at is None:
        return _unavailable("pool_age", weight, "pool creation time not reported")
    import datetime as dt

    hours = (dt.datetime.now(dt.timezone.utc) - market.pair_created_at).total_seconds() / 3600
    if hours >= 24 * 14:
        return QualityFactor("pool_age", 1.0, weight, f"{hours / 24:.0f} days old - established")
    if hours >= 24 * 2:
        return QualityFactor("pool_age", 0.75, weight, f"{hours / 24:.1f} days old")
    if hours >= 12:
        return QualityFactor("pool_age", 0.4, weight, f"{hours:.0f}h old - young")
    return QualityFactor("pool_age", 0.1, weight, f"{hours:.1f}h old - no track record at all")


def score_price_stability(market: MarketSnapshot, weight: float) -> QualityFactor:
    """Extreme intraday swings make stops meaningless regardless of setup."""
    if market.price_change_1h_pct is None:
        return _unavailable("price_stability", weight, "1h price change not reported")
    swing = abs(market.price_change_1h_pct)
    if swing >= 100:
        return QualityFactor("price_stability", 0.05, weight, f"{swing:.0f}% in 1h - stops are meaningless here")
    if swing >= 40:
        return QualityFactor("price_stability", 0.35, weight, f"{swing:.0f}% in 1h - violent")
    if swing >= 5:
        return QualityFactor("price_stability", 1.0, weight, f"{swing:.1f}% in 1h - tradeable volatility")
    return QualityFactor("price_stability", 0.6, weight, f"{swing:.1f}% in 1h - very quiet")


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def score_market_quality(
    market: MarketSnapshot | None,
    *,
    weights: dict[str, float] | None = None,
    min_liquidity_usd: float = 35_000.0,
) -> MarketQualityScore:
    """Score how tradeable a token actually is, 0-100.

    `market` of None yields a zero score flagged unreliable rather than a
    neutral 50: no market data at all is not a middling market, it is an
    unanswerable question, and the caller must reject on it.
    """
    weights = weights or DEFAULT_WEIGHTS

    if market is None:
        return MarketQualityScore(
            score=0.0, factors=[], reliable=False,
            warnings=["no market data available at all - cannot assess tradeability"],
        )

    factors = [
        score_liquidity_depth(market, weights["liquidity_depth"], min_liquidity_usd),
        score_volume_to_liquidity(market, weights["volume_to_liquidity"]),
        score_volume_concentration(market, weights["volume_concentration"]),
        score_volume_consistency(market, weights["volume_consistency"]),
        score_transaction_activity(market, weights["transaction_activity"]),
        score_buy_sell_balance(market, weights["buy_sell_balance"]),
        score_pool_age(market, weights["pool_age"]),
        score_price_stability(market, weights["price_stability"]),
    ]

    total_weight = sum(f.weight for f in factors)
    if total_weight <= 0:
        raise ValueError("market quality weights must sum to a positive number")

    raw = sum(f.score * f.weight for f in factors) / total_weight
    score = max(0.0, min(1.0, raw)) * 100

    unavailable_weight = sum(f.weight for f in factors if not f.available) / total_weight
    warnings: list[str] = []
    reliable = True
    if unavailable_weight > MAX_UNAVAILABLE_WEIGHT:
        reliable = False
        missing = ", ".join(f.name for f in factors if not f.available)
        warnings.append(
            f"{unavailable_weight:.0%} of the market-quality score has no data ({missing}) - "
            "treat as unassessable, not as average"
        )

    return MarketQualityScore(score=score, factors=factors, reliable=reliable, warnings=warnings)
