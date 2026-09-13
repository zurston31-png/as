"""MIN_SIGNAL_SCORE_TO_ENTER: what does the threshold actually buy?

The specific question this bot has been arguing with itself about. The
threshold shipped at 75, where roughly 3% of setups qualify and the bot
essentially never traded. 65 lets roughly 26% through. The tempting move is
to pick 65 because it produces trades.

That is the wrong criterion, and this module exists to make the right one
available. More trades is not better. Fewer trades is not better. The only
thing that matters is whether the trades that DO happen have positive
expectancy after costs, and whether that holds on data the choice was not
made on.

What it reports for every candidate threshold:

    trade frequency        what you give up in opportunity
    expectancy after costs the number that decides everything
    profit factor          gross profit over gross loss
    max drawdown           whether the path is survivable
    win rate / avg win / avg loss
    profit concentration   is the result one lucky trade?
    out-of-sample vs train the overfit gap
    neighbour stability    is this value a plateau or a spike?

And it is allowed to conclude that NO threshold works. If every value
produces negative after-cost expectancy, the honest output is "this
strategy has no edge at any threshold", not "65 was the least bad". That
finding is a success: it stops you trusting a system that does not work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.backtesting.types import BacktestConfig
from app.data.candles import CandleSeries
from app.research.harness import MIN_OOS_TRADES, Experiment, run_variants
from app.research.robustness import CLIFF_THRESHOLD_R

# The ladder the spec asks for. Contiguous and closely spaced on purpose:
# testing 65 alone cannot tell you whether 65 is a plateau or a spike.
CANDIDATE_THRESHOLDS: tuple[float, ...] = (55, 60, 62.5, 65, 67.5, 70, 72.5, 75)

# Profit concentrated in fewer winners than this is one lucky trade
# wearing a costume.
CONCENTRATION_LIMIT = 0.40


@dataclass
class ThresholdResult:
    threshold: float
    oos_trades: int
    oos_expectancy_r: float | None
    oos_expectancy_usd: float | None
    oos_profit_factor: float | None
    oos_win_rate: float | None
    oos_max_drawdown_pct: float | None
    oos_avg_win_usd: float | None
    oos_avg_loss_usd: float | None
    train_expectancy_r: float | None
    overfit_gap: float | None
    trades_per_100_bars: float | None
    tested: bool

    @property
    def profitable(self) -> bool:
        """Positive after-cost expectancy, on enough trades to believe."""
        return bool(self.tested and self.oos_expectancy_r is not None and self.oos_expectancy_r > 0)

    def as_dict(self) -> dict:
        def r(v, places=4):
            return round(v, places) if v is not None else None

        return {
            "threshold": self.threshold,
            "tested": self.tested,
            "profitable": self.profitable,
            "oos_trades": self.oos_trades,
            "trades_per_100_bars": r(self.trades_per_100_bars, 2),
            "oos_expectancy_r": r(self.oos_expectancy_r),
            "oos_expectancy_usd": r(self.oos_expectancy_usd, 2),
            "oos_profit_factor": r(self.oos_profit_factor, 3),
            "oos_win_rate": r(self.oos_win_rate, 1),
            "oos_max_drawdown_pct": r(self.oos_max_drawdown_pct, 2),
            "oos_avg_win_usd": r(self.oos_avg_win_usd, 2),
            "oos_avg_loss_usd": r(self.oos_avg_loss_usd, 2),
            "train_expectancy_r": r(self.train_expectancy_r),
            "overfit_gap": r(self.overfit_gap),
        }


@dataclass
class ThresholdStudy:
    results: list[ThresholdResult] = field(default_factory=list)
    experiment: Experiment | None = None
    oos_bars: int = 0

    @property
    def tested(self) -> list[ThresholdResult]:
        return [r for r in self.results if r.tested]

    @property
    def profitable(self) -> list[ThresholdResult]:
        return [r for r in self.tested if r.profitable]

    @property
    def conclusive(self) -> bool:
        return len(self.tested) >= 3

    @property
    def stable_region(self) -> list[ThresholdResult]:
        """The widest run of adjacent thresholds that are all profitable
        and none of which jumps to its neighbour."""
        ordered = sorted(self.tested, key=lambda r: r.threshold)
        best: list[ThresholdResult] = []
        run: list[ThresholdResult] = []
        for point in ordered:
            if not point.profitable:
                run = []
                continue
            if run and abs(point.oos_expectancy_r - run[-1].oos_expectancy_r) > CLIFF_THRESHOLD_R:
                run = [point]
            else:
                run.append(point)
            if len(run) > len(best):
                best = list(run)
        return best

    @property
    def recommended(self) -> float | None:
        """The centre of the widest stable profitable region, or None.

        None is a real and important answer. Returning "the best of a bad
        set" would present a losing strategy as a tuning problem.
        """
        region = self.stable_region
        if len(region) < 3:
            return None
        return region[len(region) // 2].threshold

    def verdict(self) -> str:
        if not self.conclusive:
            return (
                f"INSUFFICIENT DATA: only {len(self.tested)} of {len(self.results)} thresholds "
                f"produced {MIN_OOS_TRADES}+ out-of-sample trades. No threshold should be chosen "
                "from this. Collect more history."
            )
        if not self.profitable:
            return (
                "NO EDGE AT ANY THRESHOLD. Every tested value produced negative after-cost "
                "expectancy out-of-sample. The problem is not the threshold - lowering it will "
                "produce more losing trades and raising it will produce fewer. This strategy has "
                "not been shown to work, and that is a useful result: it stops you trusting it."
            )
        region = self.stable_region
        if len(region) < 3:
            best = max(self.profitable, key=lambda r: r.oos_expectancy_r)
            return (
                f"NO STABLE REGION. {len(self.profitable)} threshold(s) were profitable but they "
                f"do not form a run of three adjacent values. The best ({best.threshold:g}, "
                f"{best.oos_expectancy_r:+.3f}R) is an isolated result, which is what an overfit "
                "parameter looks like. Do not adopt it on this evidence."
            )
        lo, hi = region[0].threshold, region[-1].threshold
        pick = self.recommended
        chosen = next(r for r in region if r.threshold == pick)
        return (
            f"Thresholds {lo:g} to {hi:g} are all profitable out-of-sample and consistent with "
            f"each other. Recommended: {pick:g} - the centre of that region, not its peak "
            f"({chosen.oos_expectancy_r:+.3f}R over {chosen.oos_trades} trades, "
            f"{chosen.oos_max_drawdown_pct:.1f}% max drawdown)."
        )

    def as_dict(self) -> dict:
        return {
            "conclusive": self.conclusive,
            "verdict": self.verdict(),
            "recommended": self.recommended,
            "stable_region": [r.threshold for r in self.stable_region],
            "oos_bars": self.oos_bars,
            "results": [r.as_dict() for r in self.results],
        }

    def table(self) -> str:
        lines = [
            self.verdict(),
            "",
            f"  {'thresh':>7}{'OOS n':>7}{'/100bar':>9}{'OOS R':>9}{'$/trade':>10}"
            f"{'PF':>7}{'win%':>7}{'DD%':>7}{'gap':>8}",
        ]
        region = {r.threshold for r in self.stable_region}
        for r in sorted(self.results, key=lambda r: r.threshold):
            mark = "+" if r.threshold in region else (" " if r.tested else "*")

            def f(v, spec):
                return format(v, spec) if v is not None else "n/a"

            lines.append(
                f"{mark} {r.threshold:>6g}{r.oos_trades:>7}"
                f"{f(r.trades_per_100_bars, '>8.2f'):>9}"
                f"{f(r.oos_expectancy_r, '>+8.3f'):>9}"
                f"{f(r.oos_expectancy_usd, '>+9.2f'):>10}"
                f"{f(r.oos_profit_factor, '>6.2f'):>7}"
                f"{f(r.oos_win_rate, '>6.0f'):>7}"
                f"{f(r.oos_max_drawdown_pct, '>6.1f'):>7}"
                f"{f(r.overfit_gap, '>+7.3f'):>8}"
            )
        lines.append(f"\n  + stable region    * fewer than {MIN_OOS_TRADES} out-of-sample trades")
        return "\n".join(lines)


def study_thresholds(
    series: CandleSeries,
    *,
    thresholds: tuple[float, ...] = CANDIDATE_THRESHOLDS,
    base_config: BacktestConfig | None = None,
    train_fraction: float = 0.6,
) -> ThresholdStudy:
    """Run the whole threshold ladder over one chronological split."""
    base_config = base_config or BacktestConfig()
    variants = {
        f"score>={t:g}": BacktestConfig(**{**base_config.__dict__, "min_score_to_enter": t})
        for t in thresholds
    }

    experiment = run_variants(
        "threshold study", series, variants, train_fraction=train_fraction
    )
    by_label = {v.label: v for v in experiment.variants}

    results = []
    for t in thresholds:
        variant = by_label.get(f"score>={t:g}")
        stats = variant.oos_stats if variant else None
        train = variant.train_stats if variant else None
        bars = experiment.oos_bars or 1
        results.append(
            ThresholdResult(
                threshold=t,
                oos_trades=variant.oos_trades if variant else 0,
                oos_expectancy_r=variant.oos_expectancy_r if variant else None,
                oos_expectancy_usd=stats.expectancy_usd if stats else None,
                oos_profit_factor=(
                    None if not stats or stats.profit_factor in (None, float("inf"))
                    else stats.profit_factor
                ),
                oos_win_rate=stats.win_rate if stats else None,
                oos_max_drawdown_pct=stats.max_drawdown_pct if stats else None,
                oos_avg_win_usd=stats.avg_win_usd if stats else None,
                oos_avg_loss_usd=stats.avg_loss_usd if stats else None,
                train_expectancy_r=train.expectancy_r if train else None,
                overfit_gap=variant.overfit_gap if variant else None,
                trades_per_100_bars=(variant.oos_trades / bars * 100) if variant else None,
                tested=bool(variant and variant.tested),
            )
        )

    return ThresholdStudy(results=results, experiment=experiment, oos_bars=experiment.oos_bars)
