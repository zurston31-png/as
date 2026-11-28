# Strategy A (momentum breakout): in-sample results, archived

**These are simulated results on historical data.** No order was placed and no real
money was involved. They cover the in-sample (tuning) period only. The out-of-sample
period, 2025-01-01 onwards, has not been run and the parameters have not been frozen.

Nothing here is a recommendation or evidence that the strategy will make money.

## What was run

| Item | Value |
| --- | --- |
| Strategy | `momentum_breakout`, rules version `1.0`, default (agreed) parameters, no tuning |
| Period traded | 2021-06-01 to 2024-12-31, New York dates (`in_sample`) |
| History read | Daily bars from 2020-07-27 (indicator warm-up) to 2024-12-31; no later bar |
| Universe | AAPL, MSFT, NVDA, AMD, AMZN, META, TSLA, GOOGL, SPY, QQQ |
| Data | Alpaca, feed `iex`, adjustment `split`, timeframe `1Day`; 11,160 bars |
| Starting equity | 100,000 USD of virtual money per run |
| Runs | 1 `base` (5 bp slippage per side), 2 `worse` (10 bp), 3 `severe` (20 bp) |
| Run on | 2026-11-17 (UTC), code at commit `5465885` |

Parameters (`parameters_hash` 4860ad7f29baad04…, identical in all three runs):
breakout window 55, trend window 200, exit window 20, ATR window 20, stop 2 ATR, entry
skipped if the open is more than 1 ATR above the signal close, risk 0.5 % of equity per
trade, at most 20 % of equity per position, at most 5 positions.

The full rules are in [`app/strategies/momentum_breakout.py`](../../../app/strategies/momentum_breakout.py)
and the fill conventions in [`app/strategies/simulator.py`](../../../app/strategies/simulator.py).

## Headline results

| | Base (5 bp) | Worse (10 bp) | Severe (20 bp) |
| --- | --- | --- | --- |
| Total return | +38.02 % | +37.42 % | +35.55 % |
| Annualised return | +9.40 % | +9.27 % | +8.85 % |
| Trades (losing) | 62 (31) | 62 (32) | 62 (32) |
| Win rate | 50.0 % | 48.4 % | 48.4 % |
| Profit factor | 3.23 | 3.18 | 3.02 |
| Expectancy per trade | 613.23 USD, 1.14 R | 603.51 USD, 1.12 R | 573.37 USD, 1.07 R |
| Maximum drawdown | 13.2 % | 13.3 % | 13.5 % |
| Sharpe ratio (risk-free rate 0) | 1.01 | 0.99 | 0.95 |

Over the same period, buy-and-hold returned +38.7 % for SPY and +129.9 % for an equal-weight
portfolio of all ten symbols (NVDA alone +724.8 %), with drawdowns of 25.4 % to 76.7 % per
symbol. Strategy A made less than either benchmark and had a much smaller drawdown.
Roughly 41 % of its net profit came from NVDA and 63 % from its five best trades.

[`audit_report.txt`](audit_report.txt) has everything else: per-symbol results, the largest
trades, drawdown dates, holding periods, dependence on single stocks, and the integrity
checks.

## Files

| File | Contents |
| --- | --- |
| `README.md` | This file |
| `audit_report.txt` | The full audit, exactly as printed by `scripts/audit_strategy.py` |
| `audit.json` | The same audit as data, including every trade the audit read |
| `runs.json` | The three runs: parameters, cost scenario, period, data and result fingerprints, counters |
| `trades_<scenario>.csv` | All 62 trades of the run, winning and losing |
| `signals_<scenario>.csv` | All 608 signals of the run, taken or not, with the reason when not taken and the indicator values the rule compared |
| `equity_<scenario>.csv` | The virtual account at every close: 903 sessions |
| `buy_and_hold.csv` | Buy-and-hold per symbol over the same period |
| `SHA256SUMS` | Checksums of every file above except this README |

`<scenario>` is `base`, `worse` or `severe`.

### Columns

**trades**: `trade_id`, `run_id`, `signal_id` (the signal that opened it), `symbol`,
`shares`, `entry_session`, `entry_reference_price` (the bar's open), `entry_price` (after
slippage), `equity_at_signal`, `initial_stop`, `risk_amount` (planned loss at the stop),
`exit_session`, `exit_reference_price`, `exit_price` (after slippage), `exit_reason`
(`exit_signal`, `stop`, `stop_gap`, `period_end`), `sessions_held`, `gross_pnl` (at reference
prices), `slippage_cost`, `net_pnl`, `r_multiple` (net_pnl / risk_amount).

**signals**: `signal_id`, `run_id`, `symbol`, `signal_session` (the close that produced it),
`signal_type`, `close`, `rank_value` (55-session price change), `rank`, `decision` (`taken`
or `skipped`), `reason` (`already_in_position`, `no_free_slot`, `open_above_entry_limit`, ...),
`planned_shares`, `indicators` (JSON).

**equity**: `run_id`, `session`, `cash`, `positions_value`, `equity`, `open_positions`.

**buy_and_hold**: `symbol`, `first_session`, `last_session`, `entry_price` (open of the first
session), `exit_price` (close of the last), `total_return`, `total_return_after_costs` (5 bp
per side), `max_drawdown` (of the daily closes).

Dates are New York trading dates. Run timestamps in `runs.json` are UTC. Every number is
written at full precision, so it reads back exactly as it was stored.

## Assumptions

- **Fills.** A signal at a close is acted on at the next session's open. A stop inside a
  bar's range is filled at the stop; an open below the stop is filled at the open. A stop
  is assumed to come before any target within the same bar. A position can be stopped
  out on its entry day.
- **Slippage is assumed, not measured.** There is no bid/ask data. Buys fill that many basis
  points above the reference price and sells that many below. Commission is 0.
- **Sizes** are fixed at the signal close. If the cash at the next open is short, fewer
  shares are bought; the account never borrows.
- **The last session** of the period closes every open position at its close
  (`period_end`, 4 trades). These exits come from the test design, not from the rules,
  and they include gains the rules had not yet taken.
- **Sharpe** uses a risk-free rate of 0. Treasury bills paid roughly 4 to 5 % in 2023 and
  2024, so a Sharpe ratio on excess returns would be noticeably lower.
- **SPY's isolated bar of 2018-11-01** is stored but was excluded: each symbol is read only
  from the start of its unbroken history (2020-07-27).

## Limitations

- **IEX is one exchange.** Prices are IEX prices, not the consolidated tape. Breakouts, stops
  and opens are approximations of what real orders would get.
- **Ten hand-picked symbols**, mostly large technology names chosen with hindsight. The
  results say nothing about other stocks or the market as a whole.
- **A short sample.** 62 trades over three and a half years. No confidence intervals or
  significance tests were computed, and the results could differ a lot on another period.
- **A strongly rising market** for these names, with one deep decline in 2022.
- **Dividends** are not included in the strategy or in buy-and-hold. This understates
  buy-and-hold more, because it holds dividend payers the whole time.
- **Profit shares by symbol** come from one shared account. A symbol's "strategy return" is its
  net profit divided by the 100,000 starting equity, not the result of trading it alone.
  "Net without NVDA" only subtracts NVDA's trades; it does not re-simulate how that cash and
  those slots would have been used.
- **Not computed:** share of equity actually invested (exposure counts positions, not
  dollars), alpha, beta, correlation to a benchmark, Sortino ratio, turnover, and the largest
  move against or in favour of each trade before it closed (not recorded).
- **Run timestamps:** each run's `finished_at` is a few milliseconds earlier than its
  `created_at`. The runner sets `finished_at` before the database fills in `created_at`. This
  is a bookkeeping quirk and has no effect on any result.
- **Not live:** no taxes, borrowing costs, margin interest or market impact are modelled.

## Verifying and reproducing

Check that no file was changed since it was written:

```bash
cd signaliq/research/results/strategy_a && sha256sum -c SHA256SUMS
```

Every value in these files was compared with the database it came from (43,959 values,
all identical) before it was committed.

To reproduce, check out commit `5465885` or later without changes to the strategy, rebuild
the database and rerun:

```bash
python scripts/ingest.py --all
python scripts/run_strategy.py momentum_breakout --period in_sample --cost-scenario base
python scripts/run_strategy.py momentum_breakout --period in_sample --cost-scenario worse
python scripts/run_strategy.py momentum_breakout --period in_sample --cost-scenario severe
```

When the bars are the same, each run prints the same data fingerprint
`9df4889bedd03e6a190b7d2f193daee75eaa419a124b990b0e336ec867f7b3ba` and the same result
fingerprint:

| Run | Result fingerprint |
| --- | --- |
| base | `9e41aff407bf3a9c581b8f35c0cb0f894dcff4ad34c976f14c78c8b83abd982e` |
| worse | `f66257676e8921ab4ce7e226c10f1216de816bcaaf3050b984ea2247131d8350` |
| severe | `a6ec23dd54f9d8f7d898f098919140af04e24de13e6dd7f36a4f8606d574da05` |

A different data fingerprint means Alpaca served different bars, for example after a new
split re-adjusted history. The results would then differ too, and that is expected.

The export itself was made read-only with:

```bash
python scripts/export_strategy_runs.py momentum_breakout --run 1 --run 2 --run 3 --out research/results/strategy_a
```

Run ids `1`, `2` and `3` refer to the database these results came from. That database is
not in the repository.
