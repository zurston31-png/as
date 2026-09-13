"""Monte Carlo resampling of the trade sequence.

A backtest or a paper run produces ONE path. That path is the strategy's
edge plus the particular order the trades happened to arrive in, and the
order is luck. Reporting a single 18% max drawdown as "the" drawdown
therefore overstates what is known: the same set of trades, dealt in a
different order, routinely produces a much deeper one, and a live run will
deal them in a different order.

Two resampling modes, answering two different questions:

  SHUFFLE (order only, no replacement)
      Reorders exactly the trades that happened. Every path has the same
      final P&L by construction, so this isolates PATH RISK: given this
      exact edge, how bad could the ride have been? A strategy whose
      shuffled 5th-percentile drawdown would have breached the daily loss
      limit is not survivable, however good its total looks.

  BOOTSTRAP (with replacement)
      Draws N trades at random from the observed distribution, so totals
      vary too. This estimates OUTCOME RISK: what range of results is
      consistent with this edge? Its answer to "could this strategy lose
      money over the next 100 trades?" is the one that matters before
      risking anything real.

Neither invents data. Both resample only observed trade results, which is
also the honest limit of the method: if the sample is small or came from
one market regime, the resampled distribution inherits both problems.
That is why every result carries its sample size and a warning when the
sample is too small to mean anything.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

# Below this, resampling produces a confident-looking distribution built on
# nothing. The result is still returned - refusing to compute would just
# hide the sample size - but it is flagged, and no gate may pass on it.
MIN_TRADES_FOR_MONTE_CARLO = 30

DEFAULT_SIMULATIONS = 2_000


@dataclass
class MonteCarloResult:
    mode: str                          # "shuffle" | "bootstrap"
    simulations: int
    sample_size: int                   # observed trades the paths were drawn from

    median_final_pnl: float
    mean_final_pnl: float
    p05_final_pnl: float               # 5th percentile - the bad case
    p95_final_pnl: float
    probability_of_loss: float         # share of paths ending below zero

    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float        # 95th percentile drawdown - the bad ride
    worst_max_drawdown_pct: float

    longest_losing_streak_p95: int
    warnings: list[str] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.sample_size >= MIN_TRADES_FOR_MONTE_CARLO

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "simulations": self.simulations,
            "sample_size": self.sample_size,
            "reliable": self.reliable,
            "median_final_pnl": round(self.median_final_pnl, 2),
            "mean_final_pnl": round(self.mean_final_pnl, 2),
            "p05_final_pnl": round(self.p05_final_pnl, 2),
            "p95_final_pnl": round(self.p95_final_pnl, 2),
            "probability_of_loss": round(self.probability_of_loss, 4),
            "median_max_drawdown_pct": round(self.median_max_drawdown_pct, 2),
            "p95_max_drawdown_pct": round(self.p95_max_drawdown_pct, 2),
            "worst_max_drawdown_pct": round(self.worst_max_drawdown_pct, 2),
            "longest_losing_streak_p95": self.longest_losing_streak_p95,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        lines = [
            f"Monte Carlo ({self.mode}, {self.simulations:,} paths from {self.sample_size} trades)",
            f"  final P&L      median ${self.median_final_pnl:,.0f}   "
            f"5th pct ${self.p05_final_pnl:,.0f}   95th pct ${self.p95_final_pnl:,.0f}",
            f"  chance of ending down                {self.probability_of_loss:.1%}",
            f"  max drawdown   median {self.median_max_drawdown_pct:.1f}%   "
            f"95th pct {self.p95_max_drawdown_pct:.1f}%   worst {self.worst_max_drawdown_pct:.1f}%",
            f"  losing streak  95th pct {self.longest_losing_streak_p95}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _percentile(ordered: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _path_stats(pnls: list[float], starting_equity: float) -> tuple[float, float, int]:
    """(final P&L, max drawdown %, longest losing streak) for one ordering."""
    equity = starting_equity
    peak = starting_equity
    max_dd = 0.0
    streak = longest_streak = 0

    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        if pnl <= 0:
            streak += 1
            longest_streak = max(longest_streak, streak)
        else:
            streak = 0

    return equity - starting_equity, max_dd * 100, longest_streak


def run_monte_carlo(
    pnls: list[float],
    *,
    starting_equity: float,
    mode: str = "shuffle",
    simulations: int = DEFAULT_SIMULATIONS,
    path_length: int | None = None,
    rng: random.Random | None = None,
) -> MonteCarloResult:
    """Resample `pnls` into many alternative equity paths.

    `pnls` are per-trade realized dollar results, in any order - the whole
    point is that the order is not information. `path_length` defaults to
    the sample size; set it longer in bootstrap mode to ask "what might the
    next 200 trades look like?".
    """
    if mode not in ("shuffle", "bootstrap"):
        raise ValueError(f"unknown Monte Carlo mode {mode!r} - use 'shuffle' or 'bootstrap'")
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive to express drawdown as a percentage")

    rng = rng or random.Random()
    sample = [p for p in pnls if p is not None]
    warnings: list[str] = []

    if not sample:
        return MonteCarloResult(
            mode=mode, simulations=0, sample_size=0,
            median_final_pnl=0.0, mean_final_pnl=0.0, p05_final_pnl=0.0, p95_final_pnl=0.0,
            probability_of_loss=0.0, median_max_drawdown_pct=0.0, p95_max_drawdown_pct=0.0,
            worst_max_drawdown_pct=0.0, longest_losing_streak_p95=0,
            warnings=["no closed trades to resample"],
        )

    if len(sample) < MIN_TRADES_FOR_MONTE_CARLO:
        warnings.append(
            f"only {len(sample)} trades in the sample (need >={MIN_TRADES_FOR_MONTE_CARLO}) - "
            "the distribution below is arithmetic, not evidence"
        )

    length = path_length or len(sample)
    if mode == "shuffle" and path_length is not None and path_length != len(sample):
        raise ValueError(
            "shuffle mode reorders the observed trades, so path_length must equal the "
            "sample size - use bootstrap mode to project a different number of trades"
        )

    finals: list[float] = []
    drawdowns: list[float] = []
    streaks: list[int] = []

    for _ in range(simulations):
        if mode == "shuffle":
            path = sample[:]
            rng.shuffle(path)
        else:
            path = [rng.choice(sample) for _ in range(length)]

        final, dd, streak = _path_stats(path, starting_equity)
        finals.append(final)
        drawdowns.append(dd)
        streaks.append(streak)

    finals_sorted = sorted(finals)
    drawdowns_sorted = sorted(drawdowns)
    streaks_sorted = sorted(float(s) for s in streaks)

    if mode == "shuffle":
        warnings.append(
            "shuffle mode holds the trade set fixed, so every path ends at the same P&L - "
            "read the drawdown and streak figures, not the final-P&L spread"
        )

    return MonteCarloResult(
        mode=mode,
        simulations=simulations,
        sample_size=len(sample),
        median_final_pnl=statistics.median(finals),
        mean_final_pnl=sum(finals) / len(finals),
        p05_final_pnl=_percentile(finals_sorted, 0.05),
        p95_final_pnl=_percentile(finals_sorted, 0.95),
        probability_of_loss=sum(1 for f in finals if f < 0) / len(finals),
        median_max_drawdown_pct=statistics.median(drawdowns),
        p95_max_drawdown_pct=_percentile(drawdowns_sorted, 0.95),
        worst_max_drawdown_pct=max(drawdowns),
        longest_losing_streak_p95=int(round(_percentile(streaks_sorted, 0.95))),
        warnings=warnings,
    )
