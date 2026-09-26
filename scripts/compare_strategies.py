#!/usr/bin/env python3
"""Compare signal-weighting strategy profiles head to head via walk-forward
backtesting, ranked by out-of-sample expectancy/profit-factor/drawdown -
never by win rate alone.

    python scripts/compare_strategies.py
    python scripts/compare_strategies.py --regime pump --candles 2400
    python scripts/compare_strategies.py --csv data/ --symbol WIF --timeframe 15m

See app/backtesting/strategy_comparison.py for what a "strategy" means here
(a re-weighting of the same signal score, run through the same engine/risk/
exit logic - not four independently-coded systems).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtesting.strategy_comparison import STRATEGY_PROFILES, compare_strategies  # noqa: E402
from app.backtesting.types import BacktestConfig  # noqa: E402
from app.data.candles import Timeframe  # noqa: E402
from app.data.providers import CsvCandleProvider, SyntheticCandleProvider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="directory containing <symbol>_<timeframe>.csv (see CsvCandleProvider)")
    parser.add_argument("--symbol", default="TESTCOIN")
    parser.add_argument("--timeframe", default="15m", choices=[t.value for t in Timeframe])
    parser.add_argument("--regime", default="bull", choices=SyntheticCandleProvider.REGIMES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--candles", type=int, default=2400,
                         help="each of train/validation/out-of-sample needs enough history on its own")
    parser.add_argument("--starting-balance", type=float, default=1000.0)
    args = parser.parse_args()

    timeframe = Timeframe(args.timeframe)
    if args.csv:
        series = CsvCandleProvider(args.csv).fetch(args.symbol, timeframe, limit=0)
        print(f"Loaded {len(series)} candles for {args.symbol} {timeframe.value} from {args.csv}")
    else:
        series = SyntheticCandleProvider(regime=args.regime, seed=args.seed).fetch(args.symbol, timeframe, limit=args.candles)
        print(f"Generated {len(series)} synthetic {args.regime!r} candles (seed={args.seed})")

    base_config = BacktestConfig(starting_balance_usd=args.starting_balance)
    print(f"\nComparing {len(STRATEGY_PROFILES)} strategy profiles via walk-forward "
          f"(selected on TRAIN, ranked on OUT-OF-SAMPLE)...\n")
    rankings = compare_strategies(series, base_config=base_config)

    header = f"{'#':<3}{'Strategy':<18}{'OOS trades':>11}{'Win %':>8}{'Profit factor':>15}{'Expectancy R':>14}{'Max DD %':>10}{'Rank score':>12}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rankings, start=1):
        s = r.result.out_of_sample.stats
        pf = "inf" if s.profit_factor == float("inf") else (f"{s.profit_factor:.2f}" if s.profit_factor is not None else "-")
        exp_r = f"{s.expectancy_r:+.2f}" if s.expectancy_r is not None else "-"
        rank = "-inf" if r.rank_score == float("-inf") else f"{r.rank_score:.2f}"
        print(f"{i:<3}{r.label:<18}{s.trade_count:>11}{s.win_rate:>7.1f}%{pf:>15}{exp_r:>14}{s.max_drawdown_pct:>9.1f}%{rank:>12}")

    best = rankings[0]
    print(f"\nBest: {best.label!r}")
    for w in best.result.warnings:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
