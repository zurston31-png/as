# Flow Model

A rules-based trading research and backtesting system for NQ, ES, GC, QQQ
and SPX.

**Status: Phase 1 of 11 complete. No backtest has been run. No performance
figures exist. No claim is made that this strategy has an edge.**

---

## What this is

A five-component scoring system (options flow, futures order flow, market
structure, volatility/regime, liquidity/momentum) that produces a 0-100 Flow
Score, gated by an explicit filter sequence, sized by hard risk limits, and
validated with walk-forward testing, Monte Carlo and parameter-sensitivity
analysis.

Every component is defined by named, measurable features. There is no
"institutional accumulation" or "smart money" anywhere in the code; see
`ARCHITECTURE.md` section 5 for what each component actually measures and
what it degrades to when its data feed is unavailable.

## Honest note on the target win rates

The research brief targets 88-92% at approximately 1R, and ~72% over ten
years. Recorded in `config.schema.ResearchTargets` as **hypotheses to be
tested, not objectives**, because:

| Target | E[R] per trade | Implied annualized Sharpe at 500 trades/yr |
|---|---|---|
| 88% at 1R | +0.76R | ~26 |
| 92% at 1R | +0.84R | ~35 |
| 72% at 1R | +0.44R | ~11 |

Top-tier HFT market-making books run Sharpe 5-10. A sustained symmetric-1R
system at 88-92% is therefore almost certainly ruled out by arithmetic.

The usual source of a reported "90% win rate" is a target much tighter than
the stop. At a 0.5R target, breakeven is already 73% after a 0.10R cost
drag. So the system defines R rigorously and measures it:

* `R = |entry - stop| x point_value x size`, fixed at entry
* realized R = net PnL / R
* `TradeRecord.r_label_is_honest()` fails a "SCALP_1R" whose planned
  reward/risk is not actually 1.0
* `SetupConfig` refuses to load a setup whose name disagrees with its ratio
* every trade stores MAE, MFE and an itemized cost breakdown

If a 90% win rate ever appears, you will be able to tell immediately whether
it is real or an artifact of R labelling.

## Install

```bash
pip install -e ".[dev]"      # or: pip install numpy pandas scipy pydantic pyyaml plotly pytest
```

## Use (what works today)

```bash
python main.py phases                       # build status
python main.py config show                  # resolved config + hash
python main.py config show --section risk
python main.py config hash                  # experiment identity
python main.py instruments                  # contract economics
python main.py splits                       # splits and sealed-OOS status
python main.py experiments summary          # overfitting summary
python main.py config validate -s risk.risk_per_trade_pct=0.0025
```

Commands for unbuilt phases exit with status 2 and name the phase. They do
not print placeholder results.

```bash
pytest                                      # 280 tests
```

## Configuration

`flow_model/config/defaults.yaml` is the single source of truth. Override
per file or per key:

```bash
python main.py config validate -c my.yaml -s monte_carlo.n_simulations=100000
```

Precedence: packaged defaults < user YAML < `-s` overrides.
Percentages are fractions (`0.005` = 0.5%); a validator rejects anything
above 0.25 to catch the 0.5-vs-0.005 error.

`config_hash` is the content hash of everything that can change a result.
Renaming a config or changing the log level deliberately does *not* change
it, so the experiment log does not fill with spurious identities.

## Anti-overfitting machinery

These are enforced at runtime, not by convention:

| Rule | Mechanism | Fails with |
|---|---|---|
| Untouched final OOS set | `DataAccessGuard` blocks any data request overlapping the sealed window | `SealedDataAccessError` |
| Seal opened at most once | budget stored in an on-disk audit log, so a rerun cannot reset it | `SealBudgetExhaustedError` |
| No repeated fitting on the test set | `ExperimentLog.assert_phase_budget` | `ExperimentBudgetError` |
| Every parameter change has a reason | `ExperimentLog.log` rejects `params_changed` without `reason` | `ValueError` |
| No fitting to the desired win rate | `tests/unit/test_no_target_leakage.py` greps decision modules for `research_targets` and for the literals 0.88/0.92/0.72 | test failure |
| No leaky split definitions | `SplitRegistry` rejects TRAIN/VALIDATION overlapping TEST/SEALED | `ValueError` |
| Multiple-comparisons denominator | `ExperimentLog.overfitting_summary()["effective_trials_on_evaluation_data"]` | reported, not remembered |

A full-decade sweep is blocked by default:

```python
guard.check_access(date(2015,1,1), date(2025,1,1))   # SealedDataAccessError
guard.clip_to_allowed(date(2015,1,1), date(2025,1,1))  # [2015-01-01 -> 2023-01-01)
```

## Data reality

Honest accounting of what the five components need, from
`ARCHITECTURE.md` section 5:

| Component | Points | Needs | Without it |
|---|---|---|---|
| Order flow | 25 | tick data with aggressor side | **disabled** -- bar-volume "delta" is a known-bad estimator and is not substituted |
| Options flow | 20 | OPRA prints (ideal) or EOD chain + OI (degraded) | disabled or capped + flagged |
| Market structure | 20 | OHLCV | full function |
| Liquidity | 15 | L1 quotes (ideal), bar volume (degraded) | degraded + flagged |
| Volatility / momentum | 20 | OHLCV | full function |

With bars only, 45 of 100 points are unavailable. `strict_component_availability`
defaults to **true**, so the engine refuses to score rather than silently
redistributing the missing weight — redistribution inflates scores and hides
the gap. Phase 2 reports exactly which feeds are reachable before any
backtest runs.

SPX ships as a **reference instrument** (`tradable: false`): a cash index
cannot be filled, so modelling costs on it would be fiction. Its options
flow is a feature source for ES/NQ. To trade it, set `tradable: true` and
`execution_proxy: ES`, and the cost model charges the proxy's real spread
and commissions.

## Layout

```
flow_model/
  config/      Pydantic schema, YAML loader, defaults.yaml
  core/        enums, data contracts, instrument specs, determinism, base model
  data/        (Phase 2) ingestion, cleaning, quality grading, PIT MarketView
  features/    (Phase 3-5) structure, volatility, liquidity, momentum, order flow, options flow
  regime/      (Phase 3) quantitative regime classifier
  signals/     (Phase 6) component scorers, Flow Score, gates, setups
  risk/        (Phase 7) sizing, limits, kill switch
  backtest/    (Phase 7) PIT clock, execution model, ledger
  validation/  splits + sealed-OOS guard + experiment log (done);
               walk-forward, robustness, consistency, lookahead audit (Phase 8)
  monte_carlo/ (Phase 9) iid / block / regime-aware resampling
  analytics/   (Phase 8) metrics and breakdowns
  database/    (Phase 7) SQLite persistence
  dashboard/   (Phase 10) Plotly report
  live/        (Phase 11) paper-trading signal mode
  utils/       logging, serialization
tests/         280 tests
reports/       generated artifacts (gitignored)
```

See `ARCHITECTURE.md` for the module interaction contract, the exact gate
order, the lookahead firewall design, and the phase gates.

## Phase status

- [x] **1** architecture, config, core contracts, logging, experiment log, OOS seal
- [ ] **2** historical data interface, cleaning, quality grading, PIT `MarketView`
- [ ] **3** structure, volatility, liquidity, momentum features
- [ ] **4** order-flow features
- [ ] **5** options-flow features
- [ ] **6** Flow Score, entry gates, setups
- [ ] **7** backtester and execution model
- [ ] **8** walk-forward validation, robustness, consistency
- [ ] **9** Monte Carlo engine
- [ ] **10** dashboard
- [ ] **11** paper-trading signal mode and research report
