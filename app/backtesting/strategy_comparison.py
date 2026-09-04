"""Strategy comparison: several signal-weighting profiles run independently
through the SAME backtest engine and walk-forward split, ranked by
expectancy, profit factor, and max drawdown - never by win rate alone,
since a strategy that wins often but loses big can still be a net loser
(see app/backtesting/stats.py's profit_factor / expectancy_usd, which a
win-rate-only ranking would miss entirely).

"Strategy" here means a distinct emphasis within the existing weighted
signal score (app/signals/scoring.py's DEFAULT_WEIGHTS reshuffled toward a
different style of setup), not four independently-coded trading systems -
the engine, risk manager, and exit manager are shared across all of them on
purpose, so the comparison isolates what the SIGNAL WEIGHTING contributes
rather than mixing that together with different execution/risk assumptions
that would make the comparison meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.backtesting.types import BacktestConfig
from app.backtesting.walk_forward import WalkForwardResult, run_walk_forward
from app.data.candles import CandleSeries
from app.signals.scoring import DEFAULT_WEIGHTS

# Each profile re-normalizes to the same total weight as DEFAULT_WEIGHTS so
# no strategy gets an unfair overall-score inflation from unbalanced
# weights; only the RELATIVE emphasis differs between them.
_TOTAL_WEIGHT = sum(DEFAULT_WEIGHTS.values())


def _profile(**boosts: float) -> dict[str, float]:
    """Start from the default weights, multiply the named factors by their
    boost, then rescale everything back to the same total as the default
    profile."""
    weights = dict(DEFAULT_WEIGHTS)
    for name, multiplier in boosts.items():
        weights[name] = weights[name] * multiplier
    scale = _TOTAL_WEIGHT / sum(weights.values())
    return {name: w * scale for name, w in weights.items()}


STRATEGY_PROFILES: dict[str, dict[str, float]] = {
    # Default balance - trend/structure/volume roughly equally weighted.
    "balanced": dict(DEFAULT_WEIGHTS),
    # Momentum: rate-of-change, MACD, RSI carry more weight - chase strength
    # already in motion rather than the level or the structure around it.
    "momentum": _profile(momentum=2.5, macd=1.8, rsi=1.5, relative_volume=1.3,
                          support_resistance=0.4, bollinger=0.5),
    # Breakout: the breakout level, volume spike, and volatility expansion
    # dominate - a clean break on heavy volume, less concerned with how
    # extended price already looks.
    "breakout": _profile(breakout=2.5, volume_spike=2.5, relative_volume=1.8,
                          atr_sanity=1.5, rsi=0.4, vwap=0.5),
    # Trend-following: EMA stack, higher-timeframe agreement, and trend
    # direction dominate - stay with an established trend, discount
    # short-term oscillator noise almost entirely.
    "trend_following": _profile(ema_stack=2.2, multi_timeframe=2.2, trend_direction=2.2,
                                 rsi=0.3, bollinger=0.3, volume_spike=0.5),
    # Mean-reversion: RSI, Bollinger position, and VWAP distance dominate -
    # favor a pullback toward the mean over a breakout already underway,
    # which is the opposite read of the breakout profile above on the same
    # data (this is the only profile where an extended/overbought reading
    # is NOT automatically a bad thing the way the other three treat it).
    "mean_reversion": _profile(rsi=2.5, bollinger=2.5, vwap=2.0,
                                breakout=0.3, momentum=0.4, relative_volume=0.6),
}


@dataclass
class StrategyRanking:
    label: str
    result: WalkForwardResult
    rank_score: float


def _rank_key(result: WalkForwardResult) -> float:
    """Composite ranking score from OUT-OF-SAMPLE performance - the number
    that should decide which strategy actually gets used, since train
    performance is exactly what walk-forward selection already optimized
    and validation is only a checkpoint along the way. Combines expectancy
    and profit factor (both reward being net profitable, not just often
    right) with a max-drawdown penalty, so a strategy that wins often but
    with an ugly drawdown doesn't rank ahead of a steadier one on win rate
    alone - win rate does not appear in this formula at all.
    """
    stats = result.out_of_sample.stats
    if stats.trade_count == 0:
        return float("-inf")
    expectancy_component = stats.expectancy_r if stats.expectancy_r is not None else 0.0
    profit_factor = stats.profit_factor
    if profit_factor is None:
        pf_component = 0.0
    elif profit_factor == float("inf"):
        pf_component = 3.0  # cap - no losing trades at all is excellent, not infinitely so
    else:
        pf_component = min(profit_factor, 3.0)
    drawdown_penalty = stats.max_drawdown_pct / 100.0
    return expectancy_component + pf_component - drawdown_penalty


def compare_strategies(
    series: CandleSeries,
    profiles: dict[str, dict[str, float]] | None = None,
    *,
    base_config: BacktestConfig | None = None,
    split_fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> list[StrategyRanking]:
    """Run every strategy profile through the same walk-forward split and
    rank by out-of-sample performance, best first.

    Each profile still goes through its OWN train-only selection (there is
    only one config per profile here, so "selection" is really just
    confirming it on train before touching validation/out-of-sample) - the
    walk-forward discipline from app/backtesting/walk_forward.py applies
    per strategy, not just to the comparison as a whole.
    """
    profiles = profiles or STRATEGY_PROFILES
    base_config = base_config or BacktestConfig()

    rankings: list[StrategyRanking] = []
    for label, weights in profiles.items():
        config = BacktestConfig(**{**base_config.__dict__, "weights": weights})
        wf_result = run_walk_forward(series, config, split_fractions=split_fractions)
        rankings.append(StrategyRanking(label=label, result=wf_result, rank_score=_rank_key(wf_result)))

    rankings.sort(key=lambda r: r.rank_score, reverse=True)
    return rankings
