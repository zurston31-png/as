#!/usr/bin/env python3
"""Run the strategy through the walk-forward backtester and print a report.

    python scripts/run_backtest.py                          # synthetic bull market
    python scripts/run_backtest.py --regime pump --seed 7
    python scripts/run_backtest.py --csv data/WIF_15m.csv    # your own OHLCV history

This does NOT touch the live database, the webhook, or any network call -
it is pure history-in, statistics-out, exactly what "prove the strategy
before risking real capital" means. See app/backtesting/engine.py for what
is and is not simulated (fees/slippage/spread/execution delay: yes; the
rug-pull filter: no, it needs live scanner data with nothing historical to
replay against).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtesting.engine import run_backtest  # noqa: E402
from app.backtesting.types import BacktestConfig  # noqa: E402
from app.data.candles import Timeframe  # noqa: E402
from app.data.providers import CsvCandleProvider, SyntheticCandleProvider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="directory containing <symbol>_<timeframe>.csv (see CsvCandleProvider)")
    parser.add_argument("--symbol", default="TESTCOIN")
    parser.add_argument("--timeframe", default="15m", choices=[t.value for t in Timeframe])
    parser.add_argument("--regime", default="bull", choices=SyntheticCandleProvider.REGIMES,
                         help="synthetic market shape, ignored if --csv is given")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--candles", type=int, default=600)
    parser.add_argument("--starting-balance", type=float, default=1000.0)
    parser.add_argument("--min-score", type=float, default=75.0)
    args = parser.parse_args()

    timeframe = Timeframe(args.timeframe)

    if args.csv:
        provider = CsvCandleProvider(args.csv)
        series = provider.fetch(args.symbol, timeframe, limit=0)
        print(f"Loaded {len(series)} candles for {args.symbol} {timeframe.value} from {args.csv}")
    else:
        provider = SyntheticCandleProvider(regime=args.regime, seed=args.seed)
        series = provider.fetch(args.symbol, timeframe, limit=args.candles)
        print(f"Generated {len(series)} synthetic {args.regime!r} candles (seed={args.seed})")

    config = BacktestConfig(starting_balance_usd=args.starting_balance, min_score_to_enter=args.min_score)
    result = run_backtest(series, config, symbol=args.symbol)
    s = result.stats

    print()
    print("=" * 70)
    print(f"  BACKTEST RESULT — {result.symbol} {timeframe.value}")
    print("=" * 70)
    print(f"  Starting balance   ${config.starting_balance_usd:,.2f}")
    print(f"  Final balance      ${s.final_balance_usd:,.2f}")
    print(f"  Total return       {s.total_return_pct:+.2f}%  (${s.total_return_usd:+,.2f})")
    print(f"  Max drawdown       {s.max_drawdown_pct:.2f}%")
    print()
    print(f"  Trades             {s.trade_count}  ({s.win_count} win / {s.loss_count} loss)")
    print(f"  Win rate           {s.win_rate:.1f}%")
    print(f"  Avg win / loss     ${s.avg_win_usd or 0:,.2f} / ${s.avg_loss_usd or 0:,.2f}")
    print(f"  Profit factor      {s.profit_factor}")
    print(f"  Expectancy         ${s.expectancy_usd:,.2f}/trade  ({s.expectancy_r or 0:+.2f}R)")
    print(f"  Avg R multiple     {s.avg_r_multiple}")
    print(f"  Sharpe / Sortino   {s.sharpe_ratio} / {s.sortino_ratio}")
    print(f"  Longest win streak   {s.longest_winning_streak}")
    print(f"  Longest loss streak  {s.longest_losing_streak}")

    if result.warnings:
        print()
        print("  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    if result.trades:
        print()
        print("  Last 10 trades:")
        for t in result.trades[-10:]:
            print(
                f"    {t.entry_time:%Y-%m-%d %H:%M}  score={t.signal_score:5.1f}  "
                f"{t.market_regime:<28}  P&L ${t.pnl_usd:+8.2f} ({t.r_multiple:+.2f}R)  {t.exit_reason}"
            )

    print()
    print(f"  {len(result.rejections)} entries considered and rejected — see result.rejections for reasons.")
    print("=" * 70)


if __name__ == "__main__":
    main()
