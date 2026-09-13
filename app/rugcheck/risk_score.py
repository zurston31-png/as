"""Rug Risk Score: 0 (safe) - 100 (critical), from many independent signals,
never one flag. Mirrors app/signals/scoring.py's design turned upside down:
each factor scores 0.0 (no risk indicated) through 0.5 (unknown) to 1.0
(maximum risk), with a plain-English reason, and the weighted average
becomes the 0-100 score.

This is layered ON TOP OF, not instead of, the existing binary checks in
app/rugcheck/filters.py (mint/freeze authority, honeypot, LP lock, holder
concentration, liquidity depth, scanner danger flags). A token that clears
every one of those can still be rejected here if the composite score is too
high; a token that fails one of those is rejected regardless of what this
score says. Nothing here weakens an existing protection - this only adds
one more.

A factor with no data scores 0.5 (neutral) and is flagged unavailable,
never 0.0 (safe) - the same rule app/signals/scoring.py enforces, applied
to the opposite failure mode: a rug-pull your scanner couldn't see because
a field was missing is not a rug-pull that didn't happen. Two factors
(`liquidity_change`, `suspicious_transfers`) are always unavailable today,
honestly, rather than faked from data the bot doesn't have: a pre-trade
screening is a single snapshot and cannot see a liquidity trend without a
prior observation, and no wired-up scanner field reports transfer patterns.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.services.price_feed import MarketSnapshot

if TYPE_CHECKING:
    # Only for type hints - filters.py imports this module, so importing
    # TokenSnapshot at runtime here would be a circular import.
    from app.rugcheck.filters import TokenSnapshot

NEUTRAL = 0.5

# How much of the total weight may come from factors with no data before the
# score stops being trustworthy - same threshold and rationale as
# app/signals/scoring.py's MAX_UNAVAILABLE_WEIGHT.
MAX_UNAVAILABLE_WEIGHT = 0.35


@dataclass
class RiskFactor:
    name: str
    score: float          # 0.0 (safe) - 1.0 (maximum risk)
    weight: float
    reason: str
    available: bool = True

    @property
    def points(self) -> float:
        """Points this factor contributes to the final 0-100 score."""
        return self.score * self.weight * 100

    @property
    def max_points(self) -> float:
        return self.weight * 100


@dataclass
class RugRiskScore:
    score: float                       # 0-100, higher = more dangerous
    level: str                         # "safe" | "caution" | "high_risk" | "critical"
    factors: list[RiskFactor] = field(default_factory=list)
    reliable: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def top_risks(self) -> list[RiskFactor]:
        return sorted(
            [f for f in self.factors if f.available and f.score > 0.55],
            key=lambda f: f.points, reverse=True,
        )

    @property
    def unavailable(self) -> list[RiskFactor]:
        return [f for f in self.factors if not f.available]

    def breakdown(self) -> str:
        lines = [f"Rug risk score {self.score:.1f}/100 ({self.level})"]
        if not self.reliable:
            lines.append("  UNRELIABLE: " + "; ".join(self.warnings))
        for f in sorted(self.factors, key=lambda f: f.points, reverse=True):
            marker = " " if f.available else "?"
            lines.append(
                f"  {marker} {f.name:<26} {f.points:5.1f}/{f.max_points:4.1f}  {f.reason}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """For persisting into RugCheckResult / the trade journal."""
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "reliable": self.reliable,
            "warnings": list(self.warnings),
            "factors": [
                {
                    "name": f.name,
                    "score": round(f.score, 3),
                    "weight": f.weight,
                    "points": round(f.points, 2),
                    "reason": f.reason,
                    "available": f.available,
                }
                for f in self.factors
            ],
        }


# Weights sum to 1.0. Authority/honeypot/liquidity-lock carry the most
# because they are the classic, hardest-to-fake rug mechanisms; the
# market-behavior factors (age, volume, buy/sell mix, price swings) carry
# less individually since any one of them alone is a weak signal, but
# together they meaningfully move the score when several agree.
DEFAULT_WEIGHTS: dict[str, float] = {
    "mint_authority": 0.14,
    "honeypot": 0.14,
    "liquidity_locked": 0.12,
    "liquidity_depth": 0.10,
    "holder_concentration": 0.10,
    "freeze_authority": 0.08,
    "dev_wallet_concentration": 0.08,
    "scanner_danger_flags": 0.08,
    "token_age": 0.05,
    "volume_liquidity_ratio": 0.04,
    "buy_sell_imbalance": 0.03,
    "price_manipulation": 0.02,
    "liquidity_change": 0.01,
    "suspicious_transfers": 0.01,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unavailable(name: str, weight: float, reason: str) -> RiskFactor:
    return RiskFactor(name=name, score=NEUTRAL, weight=weight, reason=reason, available=False)


# ---------------------------------------------------------------------------
# scanner-derived factors (from TokenSnapshot)
# ---------------------------------------------------------------------------

def score_mint_authority(snap: TokenSnapshot, weight: float) -> RiskFactor:
    if snap.mint_authority_active is None:
        return _unavailable("mint_authority", weight, "mint authority status not reported by scanner")
    if snap.mint_authority_active:
        return RiskFactor("mint_authority", 1.0, weight, "mint authority still active - supply can be inflated at will")
    return RiskFactor("mint_authority", 0.0, weight, "mint authority renounced")


def score_freeze_authority(snap: TokenSnapshot, weight: float) -> RiskFactor:
    if snap.chain.lower() != "solana":
        return _unavailable("freeze_authority", weight, "not a Solana concept on this chain")
    if snap.freeze_authority_active is None:
        return _unavailable("freeze_authority", weight, "freeze authority status not reported by scanner")
    if snap.freeze_authority_active:
        return RiskFactor("freeze_authority", 1.0, weight, "freeze authority active - issuer can block transfers/selling")
    return RiskFactor("freeze_authority", 0.0, weight, "freeze authority renounced")


def score_honeypot(snap: TokenSnapshot, weight: float) -> RiskFactor:
    if snap.honeypot is None:
        return _unavailable("honeypot", weight, "honeypot/sellability not verified by scanner")
    if snap.honeypot:
        return RiskFactor("honeypot", 1.0, weight, "flagged as a honeypot - may not be sellable")
    return RiskFactor("honeypot", 0.0, weight, "not flagged as a honeypot")


def score_liquidity_locked(snap: TokenSnapshot, weight: float) -> RiskFactor:
    if snap.lp_secured_pct is None:
        return _unavailable("liquidity_locked", weight, "LP lock/burn status not reported by scanner")
    pct = snap.lp_secured_pct
    if pct >= 0.95:
        return RiskFactor("liquidity_locked", 0.0, weight, f"{pct * 100:.0f}% of LP locked/burned")
    if pct >= 0.5:
        return RiskFactor("liquidity_locked", 0.3, weight, f"{pct * 100:.0f}% of LP locked/burned - partial")
    if pct >= 0.15:
        return RiskFactor("liquidity_locked", 0.7, weight, f"only {pct * 100:.0f}% of LP locked/burned")
    return RiskFactor("liquidity_locked", 1.0, weight, f"LP largely unsecured ({pct * 100:.0f}% locked/burned)")


def score_liquidity_depth(snap: TokenSnapshot, weight: float, min_liquidity_usd: float) -> RiskFactor:
    if snap.liquidity_usd is None:
        return _unavailable("liquidity_depth", weight, "liquidity depth not available")
    usd = snap.liquidity_usd
    if usd >= min_liquidity_usd * 4:
        return RiskFactor("liquidity_depth", 0.0, weight, f"${usd:,.0f} liquidity - deep")
    if usd >= min_liquidity_usd:
        return RiskFactor("liquidity_depth", 0.25, weight, f"${usd:,.0f} liquidity - adequate")
    if usd >= min_liquidity_usd * 0.3:
        return RiskFactor("liquidity_depth", 0.7, weight, f"${usd:,.0f} liquidity - thin")
    return RiskFactor("liquidity_depth", 1.0, weight, f"${usd:,.0f} liquidity - very thin, easy to manipulate")


def score_holder_concentration(snap: TokenSnapshot, weight: float, max_top10_pct: float) -> RiskFactor:
    if snap.top10_pct is None:
        return _unavailable("holder_concentration", weight, "top holder concentration not available")
    pct = snap.top10_pct
    if pct <= max_top10_pct * 0.5:
        return RiskFactor("holder_concentration", 0.0, weight, f"top 10 holders own {pct * 100:.1f}%")
    if pct <= max_top10_pct:
        return RiskFactor("holder_concentration", 0.4, weight, f"top 10 holders own {pct * 100:.1f}%")
    if pct <= max_top10_pct * 1.5:
        return RiskFactor("holder_concentration", 0.75, weight, f"top 10 holders own {pct * 100:.1f}% - concentrated")
    return RiskFactor("holder_concentration", 1.0, weight, f"top 10 holders own {pct * 100:.1f}% - extremely concentrated")


def score_dev_wallet_concentration(snap: TokenSnapshot, weight: float) -> RiskFactor:
    """Static snapshot of the largest identifiable dev/creator holding.

    This is the "top-wallet concentration" / "dev-wallet activity" factor at
    scan time. Actual dev-wallet SELLING after entry is tracked separately
    and continuously by app/monitor/devwallet.py, which is the mechanism
    that can actually observe activity rather than a single point in time.
    """
    if snap.dev_pct is None:
        return _unavailable("dev_wallet_concentration", weight, "dev/creator wallet share not identifiable")
    pct = snap.dev_pct
    if pct <= 0.02:
        return RiskFactor("dev_wallet_concentration", 0.1, weight, f"dev wallet holds {pct * 100:.1f}%")
    if pct <= 0.05:
        return RiskFactor("dev_wallet_concentration", 0.35, weight, f"dev wallet holds {pct * 100:.1f}%")
    if pct <= 0.10:
        return RiskFactor("dev_wallet_concentration", 0.65, weight, f"dev wallet holds {pct * 100:.1f}% - notable")
    return RiskFactor(
        "dev_wallet_concentration", 1.0, weight,
        f"dev wallet holds {pct * 100:.1f}% - large single-wallet dump risk",
    )


def score_scanner_danger_flags(snap: TokenSnapshot, weight: float) -> RiskFactor:
    if snap.rugged:
        return RiskFactor("scanner_danger_flags", 1.0, weight, "scanner has flagged this token as already rugged")
    count = len(snap.danger_flags)
    if count == 0:
        return RiskFactor("scanner_danger_flags", 0.0, weight, "no scanner-reported danger flags")
    if count == 1:
        return RiskFactor("scanner_danger_flags", 0.6, weight, f"1 scanner danger flag: {snap.danger_flags[0]}")
    return RiskFactor(
        "scanner_danger_flags", 1.0, weight,
        f"{count} scanner danger flags: " + "; ".join(snap.danger_flags[:3]),
    )


# ---------------------------------------------------------------------------
# market-behavior factors (from MarketSnapshot)
# ---------------------------------------------------------------------------

def score_token_age(market: MarketSnapshot | None, weight: float) -> RiskFactor:
    if market is None or market.pair_created_at is None:
        return _unavailable("token_age", weight, "pair creation time not available")
    age_hours = (dt.datetime.now(dt.timezone.utc) - market.pair_created_at).total_seconds() / 3600
    if age_hours >= 24 * 14:
        return RiskFactor("token_age", 0.0, weight, f"pool is {age_hours / 24:.0f} days old")
    if age_hours >= 24 * 3:
        return RiskFactor("token_age", 0.25, weight, f"pool is {age_hours / 24:.1f} days old")
    if age_hours >= 24:
        return RiskFactor("token_age", 0.55, weight, f"pool is {age_hours / 24:.1f} days old - young")
    if age_hours >= 1:
        return RiskFactor("token_age", 0.8, weight, f"pool is {age_hours:.1f} hours old - very young")
    return RiskFactor("token_age", 1.0, weight, f"pool is {age_hours * 60:.0f} minutes old - brand new, classic rug window")


def score_volume_liquidity_ratio(market: MarketSnapshot | None, weight: float) -> RiskFactor:
    if market is None or market.volume_24h_usd is None or not market.liquidity_usd:
        return _unavailable("volume_liquidity_ratio", weight, "24h volume or liquidity not available")
    ratio = market.volume_24h_usd / market.liquidity_usd
    if ratio > 20:
        return RiskFactor(
            "volume_liquidity_ratio", 0.75, weight,
            f"24h volume is {ratio:.0f}x liquidity - possible wash trading",
        )
    if ratio < 0.05:
        return RiskFactor(
            "volume_liquidity_ratio", 0.55, weight,
            f"24h volume is only {ratio:.2f}x liquidity - little real trading",
        )
    return RiskFactor("volume_liquidity_ratio", 0.1, weight, f"24h volume is {ratio:.1f}x liquidity - normal range")


def score_buy_sell_imbalance(market: MarketSnapshot | None, weight: float) -> RiskFactor:
    if market is None or market.buys_24h is None or market.sells_24h is None:
        return _unavailable("buy_sell_imbalance", weight, "24h buy/sell counts not available")
    buys, sells = market.buys_24h, market.sells_24h
    total = buys + sells
    if total < 10:
        return _unavailable("buy_sell_imbalance", weight, f"only {total} trades in 24h - too few to read")
    sell_share = sells / total
    if sell_share >= 0.75:
        return RiskFactor(
            "buy_sell_imbalance", 0.85, weight,
            f"{sell_share * 100:.0f}% of 24h trades are sells - distribution/exit pressure",
        )
    if sell_share <= 0.20:
        return RiskFactor(
            "buy_sell_imbalance", 0.4, weight,
            f"only {sell_share * 100:.0f}% of 24h trades are sells - could be one-sided bot activity",
        )
    return RiskFactor(
        "buy_sell_imbalance", 0.1, weight,
        f"{sell_share * 100:.0f}% sells / {(1 - sell_share) * 100:.0f}% buys - balanced",
    )


def score_price_manipulation(market: MarketSnapshot | None, weight: float) -> RiskFactor:
    if market is None or market.price_change_1h_pct is None or market.liquidity_usd is None:
        return _unavailable("price_manipulation", weight, "1h price change or liquidity not available")
    swing = abs(market.price_change_1h_pct)
    thin = market.liquidity_usd < 20_000
    if swing >= 200 and thin:
        return RiskFactor(
            "price_manipulation", 1.0, weight,
            f"{swing:.0f}% 1h price swing on thin (${market.liquidity_usd:,.0f}) liquidity - manipulation pattern",
        )
    if swing >= 100:
        return RiskFactor("price_manipulation", 0.6, weight, f"{swing:.0f}% 1h price swing - highly volatile")
    if swing >= 40:
        return RiskFactor("price_manipulation", 0.3, weight, f"{swing:.0f}% 1h price swing")
    return RiskFactor("price_manipulation", 0.05, weight, f"{swing:.1f}% 1h price swing - unremarkable")


def score_liquidity_change(weight: float) -> RiskFactor:
    return _unavailable(
        "liquidity_change", weight,
        "requires a prior snapshot - not available on first screening "
        "(post-entry liquidity is tracked separately by the position monitor)",
    )


def score_suspicious_transfers(weight: float) -> RiskFactor:
    return _unavailable(
        "suspicious_transfers", weight,
        "no scanner field for suspicious transfer patterns is wired up yet",
    )


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def score_rug_risk(
    snap: TokenSnapshot,
    market: MarketSnapshot | None = None,
    *,
    weights: dict[str, float] | None = None,
    min_liquidity_usd: float = 15_000.0,
    max_top10_pct: float = 0.35,
) -> RugRiskScore:
    """Score a token's rug risk from 0 (safe) to 100 (critical).

    `market` is optional so this stays synchronously testable without a
    network call - every market-behavior factor degrades to "unavailable"
    (neutral, flagged) when it's None, the same fail-safe every other
    factor here follows for missing data.
    """
    weights = weights or DEFAULT_WEIGHTS

    factors = [
        score_mint_authority(snap, weights["mint_authority"]),
        score_freeze_authority(snap, weights["freeze_authority"]),
        score_honeypot(snap, weights["honeypot"]),
        score_liquidity_locked(snap, weights["liquidity_locked"]),
        score_liquidity_depth(snap, weights["liquidity_depth"], min_liquidity_usd),
        score_holder_concentration(snap, weights["holder_concentration"], max_top10_pct),
        score_dev_wallet_concentration(snap, weights["dev_wallet_concentration"]),
        score_scanner_danger_flags(snap, weights["scanner_danger_flags"]),
        score_token_age(market, weights["token_age"]),
        score_volume_liquidity_ratio(market, weights["volume_liquidity_ratio"]),
        score_buy_sell_imbalance(market, weights["buy_sell_imbalance"]),
        score_price_manipulation(market, weights["price_manipulation"]),
        score_liquidity_change(weights["liquidity_change"]),
        score_suspicious_transfers(weights["suspicious_transfers"]),
    ]

    total_weight = sum(f.weight for f in factors)
    if total_weight <= 0:
        raise ValueError("factor weights must sum to a positive number")

    raw = sum(f.score * f.weight for f in factors) / total_weight
    score = _clamp(raw) * 100

    unavailable_weight = sum(f.weight for f in factors if not f.available) / total_weight
    warnings: list[str] = []
    reliable = True
    if unavailable_weight > MAX_UNAVAILABLE_WEIGHT:
        reliable = False
        missing = ", ".join(f.name for f in factors if not f.available)
        warnings.append(
            f"{unavailable_weight:.0%} of the score has no data ({missing}) - "
            "treat this score as uninformative, NOT as a clean bill of health"
        )

    if score < 20:
        level = "safe"
    elif score < 45:
        level = "caution"
    elif score < 70:
        level = "high_risk"
    else:
        level = "critical"

    return RugRiskScore(score=score, level=level, factors=factors, reliable=reliable, warnings=warnings)
