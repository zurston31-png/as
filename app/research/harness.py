"""Shared machinery for running a set of strategy variants and comparing
them honestly.

Every experiment in this package has the same shape: build N variants of
the strategy, run each through the SAME chronological train/validation/
out-of-sample split, and rank them. The two things that make the answer
trustworthy both live here.

CHRONOLOGICAL SPLITS, NEVER SHUFFLED. Randomly assigning bars to train and
test leaks the future into the past: a model can learn from Tuesday
afternoon to predict Tuesday morning. Every split in this package is a
time cut, and the out-of-sample window is always the latest data.

RANKED ON OUT-OF-SAMPLE, WITH PENALTIES. Ranking by in-sample P&L selects
whichever variant fit the training noise hardest. `robust_score` below
ranks on out-of-sample expectancy and then subtracts for the things that
make a good number untrustworthy: a thin sample, a deep drawdown, and a
large gap between training and out-of-sample performance. A variant that
looks brilliant in training and mediocre afterwards is not a good variant;
it is an overfit one, and the gap penalty is what says so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.backtesting.engine import run_backtest
from app.backtesting.types import BacktestConfig, BacktestResult, BacktestStats
from app.data.candles import CandleSeries

logger = logging.getLogger(__name__)

# A variant with fewer out-of-sample trades than this has not been tested,
# whatever number it produced.
MIN_OOS_TRADES = 20

# How hard to penalise the train -> out-of-sample gap. A variant whose
# out-of-sample expectancy is far below its training expectancy has learnt
# the training window, not the market.
OVERFIT_PENALTY_WEIGHT = 0.5

# Drawdown penalty, per percentage point of out-of-sample max drawdown.
DRAWDOWN_PENALTY_WEIGHT = 0.02


@dataclass
class VariantResult:
    """One strategy variant, run through one chronological split."""

    label: str
    config: BacktestConfig
    train: BacktestResult
    oos: BacktestResult
    notes: str = ""

    @property
    def train_stats(self) -> BacktestStats | None:
        return self.train.stats

    @property
    def oos_stats(self) -> BacktestStats | None:
        return self.oos.stats

    @property
    def oos_trades(self) -> int:
        return self.oos_stats.trade_count if self.oos_stats else 0

    @property
    def tested(self) -> bool:
        """Enough out-of-sample trades to mean anything."""
        return self.oos_trades >= MIN_OOS_TRADES

    @property
    def oos_expectancy_r(self) -> float | None:
        """Average realized R per out-of-sample trade.

        R rather than dollars so variants with different position sizing
        stay comparable - a variant that simply traded bigger is not a
        better variant.
        """
        if not self.oos_stats or not self.oos_trades:
            return None
        return self.oos_stats.expectancy_r

    @property
    def overfit_gap(self) -> float | None:
        """train expectancy_r minus out-of-sample expectancy_r.

        Positive means the variant did better on the data it was chosen on
        - the signature of fitting rather than learning.
        """
        if not self.train_stats or not self.oos_stats:
            return None
        train_r, oos_r = self.train_stats.expectancy_r, self.oos_stats.expectancy_r
        if train_r is None or oos_r is None:
            return None
        return train_r - oos_r

    @property
    def robust_score(self) -> float | None:
        """Out-of-sample expectancy, penalised for what makes it fragile.

        None when untested - and None must sort LAST, never as a zero that
        happens to beat a genuinely negative result.
        """
        if not self.tested or self.oos_expectancy_r is None:
            return None
        score = self.oos_expectancy_r
        gap = self.overfit_gap
        if gap is not None and gap > 0:
            score -= OVERFIT_PENALTY_WEIGHT * gap
        if self.oos_stats:
            score -= DRAWDOWN_PENALTY_WEIGHT * max(self.oos_stats.max_drawdown_pct, 0.0)
        return score

    def as_dict(self) -> dict:
        def stats_dict(s: BacktestStats | None) -> dict | None:
            if s is None:
                return None
            return {
                "trades": s.trade_count,
                "win_rate": round(s.win_rate, 1),
                "expectancy_r": round(s.expectancy_r, 4) if s.expectancy_r is not None else None,
                "expectancy_usd": round(s.expectancy_usd, 2),
                "profit_factor": (
                    None if s.profit_factor in (None, float("inf")) else round(s.profit_factor, 3)
                ),
                "max_drawdown_pct": round(s.max_drawdown_pct, 2),
                "total_return_pct": round(s.total_return_pct, 2),
            }

        return {
            "label": self.label,
            "tested": self.tested,
            "oos_trades": self.oos_trades,
            "robust_score": round(self.robust_score, 4) if self.robust_score is not None else None,
            "overfit_gap": round(self.overfit_gap, 4) if self.overfit_gap is not None else None,
            "train": stats_dict(self.train_stats),
            "oos": stats_dict(self.oos_stats),
            "notes": self.notes,
        }


@dataclass
class Experiment:
    """A set of variants run over one dataset, ranked out-of-sample."""

    name: str
    variants: list[VariantResult] = field(default_factory=list)
    train_bars: int = 0
    oos_bars: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def tested_variants(self) -> list[VariantResult]:
        return [v for v in self.variants if v.tested]

    @property
    def ranked(self) -> list[VariantResult]:
        """Best first. Untested variants sort last, never in the middle."""
        return sorted(
            self.variants,
            key=lambda v: (v.robust_score is not None, v.robust_score or float("-inf")),
            reverse=True,
        )

    @property
    def best(self) -> VariantResult | None:
        top = self.ranked[0] if self.ranked else None
        return top if (top and top.tested) else None

    @property
    def conclusive(self) -> bool:
        """At least two variants with enough out-of-sample trades to compare."""
        return len(self.tested_variants) >= 2

    def verdict(self) -> str:
        if not self.variants:
            return f"{self.name}: no variants were run."
        if not self.conclusive:
            return (
                f"{self.name}: INSUFFICIENT DATA - only {len(self.tested_variants)} of "
                f"{len(self.variants)} variants produced {MIN_OOS_TRADES}+ out-of-sample trades. "
                "Nothing can be compared. Collect more history before drawing a conclusion."
            )
        best = self.best
        worst = self.tested_variants[-1] if self.tested_variants else None
        worst = min(self.tested_variants, key=lambda v: v.robust_score)
        spread = best.robust_score - worst.robust_score
        if spread < 0.05:
            return (
                f"{self.name}: no variant separates from the others (robust scores span only "
                f"{spread:.3f}R across {len(self.tested_variants)} tested variants). Treat them "
                "as equivalent rather than picking the nominal winner."
            )
        return (
            f"{self.name}: '{best.label}' ranks highest out-of-sample "
            f"({best.robust_score:+.3f}R adjusted, {best.oos_trades} trades) against "
            f"'{worst.label}' at {worst.robust_score:+.3f}R."
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "train_bars": self.train_bars,
            "oos_bars": self.oos_bars,
            "conclusive": self.conclusive,
            "verdict": self.verdict(),
            "best": self.best.label if self.best else None,
            "variants": [v.as_dict() for v in self.ranked],
            "warnings": list(self.warnings),
        }

    def table(self) -> str:
        lines = [
            self.verdict(),
            "",
            f"  {'variant':<28}{'OOS n':>7}{'OOS R':>9}{'train R':>9}{'gap':>8}{'DD %':>8}{'score':>9}",
        ]
        for v in self.ranked:
            oos_r = v.oos_expectancy_r
            train_r = v.train_stats.expectancy_r if v.train_stats else None
            dd = v.oos_stats.max_drawdown_pct if v.oos_stats else None
            mark = " " if v.tested else "*"
            lines.append(
                f"{mark} {v.label:<28}{v.oos_trades:>7}"
                f"{(f'{oos_r:+.3f}' if oos_r is not None else 'n/a'):>9}"
                f"{(f'{train_r:+.3f}' if train_r is not None else 'n/a'):>9}"
                f"{(f'{v.overfit_gap:+.3f}' if v.overfit_gap is not None else 'n/a'):>8}"
                f"{(f'{dd:.1f}' if dd is not None else 'n/a'):>8}"
                f"{(f'{v.robust_score:+.3f}' if v.robust_score is not None else 'n/a'):>9}"
            )
        lines.append(f"\n  * fewer than {MIN_OOS_TRADES} out-of-sample trades - not tested, not ranked")
        return "\n".join(lines)


def run_variants(
    name: str,
    series: CandleSeries,
    variants: dict[str, BacktestConfig],
    *,
    train_fraction: float = 0.6,
    symbol: str | None = None,
) -> Experiment:
    """Run every variant over the same chronological train/out-of-sample cut.

    The split is a TIME cut, always: the out-of-sample window is the latest
    `1 - train_fraction` of the series and no variant ever sees it during
    selection. Shuffling bars between the two would let a variant learn
    from Tuesday afternoon to predict Tuesday morning.
    """
    if not variants:
        raise ValueError("no variants to run")
    if not 0.1 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must leave meaningful data on both sides of the cut")

    experiment = Experiment(name=name)

    cut = int(len(series) * train_fraction)
    train_series = series.head(cut)
    oos_series = CandleSeries(
        symbol=series.symbol, timeframe=series.timeframe, candles=series.candles[cut:]
    )
    experiment.train_bars = len(train_series)
    experiment.oos_bars = len(oos_series)

    if experiment.oos_bars < 100:
        experiment.warnings.append(
            f"only {experiment.oos_bars} out-of-sample bars - too short to test anything on. "
            "Every result below is arithmetic."
        )

    for label, config in variants.items():
        try:
            train = run_backtest(train_series, config, symbol=symbol)
            oos = run_backtest(oos_series, config, symbol=symbol)
        except Exception as exc:
            logger.exception("variant %s failed", label)
            experiment.warnings.append(f"variant '{label}' raised and was skipped: {exc}")
            continue
        experiment.variants.append(VariantResult(label=label, config=config, train=train, oos=oos))

    return experiment
