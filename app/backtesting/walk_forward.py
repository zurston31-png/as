"""Walk-forward testing: train / validation / out-of-sample, with parameter
or strategy selection based ONLY on the training window.

This is the discipline that separates "I found a config that made money on
this data" from "this strategy generalizes": whichever config wins is
chosen purely from its TRAIN performance, before the validation or
out-of-sample windows are ever backtested. If a config were instead picked
by peeking at how it does on validation or out-of-sample data, that data
would no longer be independent confirmation - it would just be more
training data with an extra step, and the walk-forward result would prove
nothing about future generalization.

`CandleSeries.split()` (app/data/candles.py) already guarantees the three
windows are chronological and non-overlapping - train ends before
validation starts, validation ends before out-of-sample starts - so there
is no path for a later window's data to leak into an earlier one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.backtesting.engine import run_backtest
from app.backtesting.types import BacktestConfig, BacktestResult, BacktestStats
from app.data.candles import CandleSeries


@dataclass
class WalkForwardCandidate:
    label: str
    config: BacktestConfig
    train_result: BacktestResult
    train_score: float


@dataclass
class WalkForwardResult:
    symbol: str
    selection_metric: str
    chosen_label: str
    chosen_config: BacktestConfig
    candidates: list[WalkForwardCandidate]
    train: BacktestResult
    validation: BacktestResult
    out_of_sample: BacktestResult
    warnings: list[str] = field(default_factory=list)


def _metric_value(stats: BacktestStats, metric: str) -> float:
    value = getattr(stats, metric, None)
    if value is None:
        return float("-inf")
    return value


def run_walk_forward(
    series: CandleSeries,
    configs: dict[str, BacktestConfig] | list[BacktestConfig] | BacktestConfig,
    *,
    split_fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
    selection_metric: str = "expectancy_r",
    min_train_trades: int = 5,
    overfit_warning_ratio: float = 0.5,
) -> WalkForwardResult:
    """Split `series` into train/validation/out-of-sample, pick the best of
    `configs` by TRAIN performance alone, then report how that one choice
    holds up on the two windows it never influenced.

    `configs` may be a single BacktestConfig (the common case: "does this
    one set of parameters generalize?"), a list, or a dict of label ->
    config for a small parameter search. `selection_metric` names any
    numeric field on BacktestStats - expectancy_r (average realized R per
    trade) is the default because it is comparable across configs with
    different position sizing, unlike total_return_pct.

    A config that produces fewer than `min_train_trades` on the training
    window is not disqualified outright (an empty train result is itself
    informative), but the result carries a warning that the walk-forward
    verdict isn't statistically meaningful yet - a "good" score from 2
    trades is noise, not evidence.
    """
    if isinstance(configs, BacktestConfig):
        configs = {"default": configs}
    elif isinstance(configs, list):
        configs = {f"candidate_{i}": c for i, c in enumerate(configs)}
    if not configs:
        raise ValueError("configs must contain at least one BacktestConfig")

    train_series, val_series, oos_series = series.split(*split_fractions)

    candidates: list[WalkForwardCandidate] = []
    for label, config in configs.items():
        train_result = run_backtest(train_series, config, symbol=series.symbol)
        candidates.append(WalkForwardCandidate(
            label=label, config=config, train_result=train_result,
            train_score=_metric_value(train_result.stats, selection_metric),
        ))

    # Selection is ENTIRELY from train_score - the walk-forward guarantee.
    best = max(candidates, key=lambda c: c.train_score)

    warnings: list[str] = []
    if best.train_result.stats.trade_count < min_train_trades:
        warnings.append(
            f"even the best candidate ({best.label!r}) only produced "
            f"{best.train_result.stats.trade_count} trade(s) on the training window (wanted "
            f"≥{min_train_trades}) - this walk-forward result is not statistically meaningful yet; "
            "use a longer history or a less restrictive config before trusting it"
        )
    if math.isinf(best.train_score) and best.train_score < 0:
        warnings.append(f"no candidate produced a usable {selection_metric!r} on the training window")

    validation_result = run_backtest(val_series, best.config, symbol=series.symbol)
    out_of_sample_result = run_backtest(oos_series, best.config, symbol=series.symbol)

    val_score = _metric_value(validation_result.stats, selection_metric)
    if best.train_score > 0 and val_score < best.train_score * overfit_warning_ratio:
        warnings.append(
            f"{selection_metric} degraded from {best.train_score:.3f} on training data to "
            f"{val_score:.3f} on validation data (chosen config {best.label!r}) - possible overfitting to "
            "the training window; treat the out-of-sample result with caution rather than as confirmation "
            "the strategy works"
        )

    return WalkForwardResult(
        symbol=series.symbol,
        selection_metric=selection_metric,
        chosen_label=best.label,
        chosen_config=best.config,
        candidates=candidates,
        train=best.train_result,
        validation=validation_result,
        out_of_sample=out_of_sample_result,
        warnings=warnings,
    )
