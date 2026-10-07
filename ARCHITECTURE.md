# Flow Model — System Architecture

Version: 0.1 (Phase 1)
Status: architecture frozen; Phase 1 (config + core contracts) implemented.

---

## 0. Design principles

These are binding constraints, not aspirations. Every later phase is
reviewed against them.

1. **Point-in-time or it does not exist.** Feature code never receives a
   DataFrame it can index past `now`. It receives a `MarketView` that is
   physically incapable of returning future data. This is the structural
   defence against lookahead bias (section 4).
2. **R is defined once, at entry, and never revised.**
   `R = |entry_price - stop_price| * point_value * size`. Realized R is
   `pnl_after_costs / R`. A "1R target" means `|target - entry| == |entry - stop|`
   within a configured tolerance. The engine refuses to label a trade "1R"
   otherwise.
3. **No vague components.** Each of the five Flow Model components resolves
   to named scalar features with explicit formulas and explicit data
   requirements. "Institutional" and "smart money" appear nowhere in code.
4. **Missing data produces WAIT, never a guess.** Every feature carries a
   `DataQuality` status. Required-feed degradation short-circuits to WAIT.
5. **Determinism.** Same config hash + same data hash => bit-identical
   results. All randomness flows through named, seeded generators.
6. **The final out-of-sample window is sealed.** Reading it requires an
   explicit unseal with a logged reason. Accidental contamination raises.
7. **Weights and thresholds are hypotheses.** The point-scheme in the brief
   (20/25/20/15/20) is a starting prior, declared in YAML, and tested.
8. **No performance claim without a split label.** Analytics refuse to
   aggregate IS and OOS into one unlabeled number.

---

## 1. Module map

```
flow_model/
  config/        Pydantic settings, YAML loading, config hashing
  core/          enums, data contracts, instrument specs, determinism, clock types
  data/          ingestion adapters, cleaning, session calendar, quality grading
  features/      structure, volatility, liquidity, momentum, orderflow, optionsflow
  regime/        quantitative regime classifier
  signals/       component scorers -> FlowScore -> entry rules -> setups -> engine
  risk/          position sizing, risk limits, kill switch, stateful manager
  backtest/      PIT clock, execution/fill model, ledger, backtest engine
  validation/    walk-forward splits, split registry + OOS seal, experiment log,
                 robustness sweeps, consistency score, lookahead audit harness
  monte_carlo/   iid / block-bootstrap / regime-aware resampling engine
  analytics/     metrics, breakdowns, trade journal queries
  database/      SQLite schema + repositories (trades, equity, experiments)
  dashboard/     Plotly report builder (static HTML)
  live/          paper-trading signal mode
  utils/         logging, time helpers, numerics
tests/           unit + property + synthetic-dataset tests
reports/         generated HTML/JSON artifacts (gitignored)
main.py          CLI entry point
```

---

## 2. The data contracts (the spine)

Everything flows through a small number of frozen types in `core/contracts.py`.
Changing one of these is an architectural change, not a tweak.

| Type | Produced by | Consumed by | Notes |
|---|---|---|---|
| `Bar` | `data.ingest` | `data.clean`, features | OHLCV + `bar_close_ts`; immutable |
| `QuoteSnapshot` | `data.ingest` | liquidity, execution | bid/ask/sizes at a timestamp |
| `TickAggregate` | `data.ingest` | orderflow | per-bar buy/sell volume split, delta |
| `OptionsSnapshot` | `data.ingest` | optionsflow | chain-level aggregates, not raw chains |
| `FeatureVector` | `features.*` | regime, signals | `dict[str, float]` + per-key `DataQuality` |
| `RegimeState` | `regime.detector` | signals, risk, analytics | enum + confidence + diagnostics |
| `ComponentScores` | `signals.*` | `flow_score` | 5 sub-scores, each 0..max_points |
| `FlowScore` | `signals.flow_score` | entry rules | 0..100 + component breakdown |
| `Signal` | `signals.engine` | risk, backtest, live | LONG/SHORT/WAIT + reasons + quality |
| `TradeIntent` | `risk.manager` | backtest.execution | entry/stop/target/size/risk_$ |
| `TradeRecord` | `backtest.engine` | analytics, database, MC | the 24-field audit row |
| `EquityPoint` | `backtest.ledger` | analytics, dashboard | timestamped equity + open heat |

`TradeRecord` carries every field in the brief plus `split_label`,
`config_hash`, `data_hash`, `cost_breakdown`, `mae_r`, `mfe_r`,
`r_definition_check`, and `bars_held`. It is the unit of truth for all
downstream analysis: Monte Carlo, analytics, and the dashboard read only
`TradeRecord` lists, never re-derive from price data.

---

## 3. Signal pipeline — exact call order

One call per bar close, per symbol. Each stage can short-circuit to WAIT,
and the short-circuit reason is recorded even for non-trades (this is what
makes rejection analysis possible later).

```
BacktestClock.advance()
  -> MarketView(symbol, now)                      # PIT firewall
  -> DataQualityGate.grade(view)                  # GOOD/DEGRADED/STALE/MISSING
       |-- MISSING or STALE on a required feed -> WAIT("data_quality")
  -> FeatureBundle.compute(view)                  # structure/vol/liquidity/
       |                                          # momentum/orderflow/options
       |-- insufficient warmup -> WAIT("warmup")
  -> RegimeDetector.classify(features)            # -> RegimeState
       |-- regime not in config.allowed_regimes -> WAIT("regime_blocked")
  -> LiquidityGate.check(features, regime)
       |-- spread/depth/volume outside band -> WAIT("liquidity")
  -> StructureGate.check(features)                # directional bias or none
       |-- no structural bias -> WAIT("no_structure")
  -> ComponentScorers.score(features, regime)     # -> ComponentScores
  -> FlowScore.combine(scores, weights)           # -> 0..100
       |-- score < setup.min_score -> WAIT("score_below_threshold")
  -> ConfirmationGate (orderflow must agree with structure direction)
       |-- disagree -> WAIT("orderflow_conflict")
  -> ContradictionGate (options flow must not strongly oppose)
       |-- options_opposition > limit -> WAIT("options_contradiction")
  -> VolatilityGate (ATR percentile within setup band)
       |-- outside -> WAIT("volatility_band")
  -> SetupSelector.select(score, regime, features) # SCALP_1R | SETUP_2R | DIRECTIONAL_3R
  -> RiskManager.build_intent(setup, features)     # entry/stop/target/size
       |-- rr < setup.min_rr -> WAIT("rr_too_low")
       |-- any risk limit breached -> WAIT("risk_limit:<name>")
  -> TradeIntent
```

The gate order is deliberate: cheap/structural rejections run before
expensive scoring, and risk limits run last so that a trade blocked by a
daily-loss limit is still recorded as a *signal that occurred* — which
keeps the signal-level and execution-level statistics separable.

---

## 4. Lookahead firewall (how modules are prevented from cheating)

Three enforcement layers:

1. **`MarketView` is read-only and bounded.** Constructed with `now` and a
   `data_latency` offset. `view.bars(n)` returns the last `n` bars whose
   `bar_close_ts <= now - latency`. There is no method that returns a full
   series, and no method that takes a future timestamp. A feature *cannot*
   see tomorrow because the object has no accessor for it.
2. **Monotonic clock assertion.** `BacktestClock` asserts strictly
   increasing timestamps and records the highest timestamp any view has
   served. `backtest.engine` asserts every `TradeRecord.entry_ts >= signal_ts`
   and `exit_ts > entry_ts`.
3. **Automated audit harness** (`validation/lookahead.py`, Phase 7). For each
   feature: compute on the full series, then recompute bar-by-bar through
   `MarketView`, and assert equality. Any feature whose values differ is
   using future data. Also runs a *future-shuffle test*: randomize all bars
   after `t` and assert features at `t` are unchanged.

Bar-close semantics: a signal computed from the bar closing at `t` may only
execute at or after `t + latency`, and fills use prices from `t+1` onward.
The engine never fills at the signal bar's close.

---

## 5. Component definitions (what each of the five is, concretely)

Formulas are finalized in Phases 3-5. Committed now: the *data requirement*
and the *degradation path*, because that determines whether the system can
honestly run at all.

| Component | Points (prior) | Primary features | Required data | If unavailable |
|---|---|---|---|---|
| Order flow | 25 | signed delta, CVD slope, aggression ratio, absorption at level, trade-size distribution | tick-level trades w/ aggressor side, or bid/ask-classified ticks | **No proxy accepted.** Component disabled and the 25 points are reported UNAVAILABLE; the weight is **not** redistributed (see the honest-consequence paragraph below and §14.5). Bar-volume-only "delta" is a known-bad estimator and is not substituted silently. |
| Options flow | 20 | net premium (call-prem minus put-prem), delta-weighted volume, OI change, 25d skew, gamma-exposure proxy | OPRA trade prints (ideal), or EOD chain + OI (degraded) | Degrades to `DEGRADED` with daily granularity; `OptionsFlow` sub-score capped and flagged. Never synthesized. |
| Market structure | 20 | swing pivots (ATR-scaled), break-of-structure, HH/HL sequence, VWAP deviation, opening-range position, prior-day levels | OHLCV bars | Full function on bars alone. |
| Liquidity | 15 | quoted spread, spread percentile, depth imbalance, volume percentile vs time-of-day curve, participation cost estimate | L1 quotes (ideal), bar volume (degraded) | Degrades to volume-percentile only, flagged `DEGRADED`. |
| Volatility / momentum | 20 | ATR percentile, realized vol (Parkinson/Garman-Klass), vol-of-vol, ROC, efficiency ratio, momentum acceleration | OHLCV bars | Full function on bars alone. |

**Honest consequence:** with bars-only data, 25+20 = 45 of 100 points are
unavailable or degraded. The config therefore has
`strict_component_availability` (default **true**), which makes the system
refuse to produce scores from disabled components rather than quietly
reweighting. Phase 2 reports exactly which feeds are reachable before any
backtest runs.

Explicitly rejected interpretations: options flow is treated as a
*measurable order-imbalance and positioning signal*, not as evidence of
institutional intent. Large premium prints have ambiguous sign (they may be
hedges, spreads, or closing trades); the feature set therefore measures
imbalance and realized dealer-hedging pressure proxies, with no claim about
who traded or why.

---

## 6. Regime detection (mathematical, no manual labels)

`regime/detector.py` computes, per bar, from PIT data only:

- `vol_pct` = percentile rank of ATR(14) within a rolling 252-session window
- `trend_strength` = |Kendall tau of close vs time| over N bars, or ADX(14)
- `efficiency` = |close_t - close_{t-N}| / sum(|close_i - close_{i-1}|)  (Kaufman ER)
- `vol_of_vol` = stdev of rolling realized vol
- `shift_stat` = CUSUM statistic on standardized returns (detects mean breaks)

Classification (thresholds in YAML, all testable):

```
DIRECTIONAL_SHIFT  if shift_stat > cusum_threshold          (checked first)
HIGH_VOL           if vol_pct >= high_vol_pct               (default 0.80)
LOW_VOL            if vol_pct <= low_vol_pct                (default 0.20)
TRENDING_UP        if efficiency >= trend_eff and slope > 0
TRENDING_DOWN      if efficiency >= trend_eff and slope < 0
CHOP               otherwise
```

Hysteresis: a regime change requires `min_regime_bars` consecutive
qualifying bars, preventing single-bar flapping. Regimes are *not* mutually
exclusive in reality, so `RegimeState` carries the primary label plus the
full diagnostic vector, and analytics breaks results down by both.

The brief's expectation ("88-92% in low- and high-vol, worse in chop") is
registered as a **falsifiable hypothesis** in the experiment log, and
`analytics.breakdowns` tests it directly rather than assuming it.

Implemented in `validation/hypotheses.py` as `H-REGIME-WINRATE` and
`H-REGIME-CHOP-WORSE`. See section 15.

---

## 7. Flow Score

```
ComponentScores: options c_o in [0, w_o], orderflow c_f in [0, w_f],
                 structure c_s, liquidity c_l, volmom c_v
FlowScore = c_o + c_f + c_s + c_l + c_v      in [0, 100]
```

Each scorer maps its features to [0, 1] through explicit, bounded
transforms (percentile ranks or `tanh` squashes — never unbounded z-scores),
then multiplies by its weight. Weights live in
`config.flow_score.weights` and must sum to 100.

Direction is handled separately from magnitude: each component emits
`(magnitude in [0,1], direction in {-1,0,+1})`. The Flow Score is the
magnitude aggregate; direction agreement is enforced by the gates. This
avoids the common bug where a strong bearish component inflates a bullish
score.

Weight testing plan (Phase 8+): equal-weight baseline, brief's prior,
and a sweep — all evaluated on training folds only, with the sealed OOS
untouched until a single final run.

---

## 8. Risk management

`risk/sizing.py` (pure functions, fully unit-tested):

```
risk_dollars   = equity * risk_pct                       (capped by max_risk_pct)
stop_distance  = |entry - stop|                           (in price units)
risk_per_unit  = stop_distance * point_value
size           = floor(risk_dollars / risk_per_unit)      (integer contracts/shares)
actual_risk    = size * risk_per_unit
R              = actual_risk
target         = entry +/- stop_distance * rr_multiple
```

Rejects the trade if `size == 0`, if `actual_risk > equity * max_risk_pct`,
or if `stop_distance < min_stop_ticks * tick_size`.

`risk/limits.py` — stateful, checked in this order, each with its own WAIT
reason: `kill_switch` (equity drawdown from high-water mark),
`daily_loss_limit`, `max_consecutive_losses`, `cooldown_active`,
`max_open_positions`, `max_portfolio_heat` (sum of open `R` as % of equity).
All limits are hard: the manager cannot be configured to exceed
`max_risk_per_trade_pct`.

---

## 9. Backtester

Event-driven, single-threaded, deterministic.

```
for ts in clock:                      # bar closes, chronological
    ledger.mark_to_market(ts)
    manager.update_open_positions(ts) # stop/target/time exits first
    signal = signal_engine.evaluate(MarketView(symbol, ts))
    journal.record_signal(signal)      # including WAITs + reasons
    if signal.action is not WAIT:
        intent = risk_manager.build_intent(signal)
        if intent: execution.submit(intent, ts)
    execution.process_pending(ts)      # fills resolve on later bars only
```

`backtest/execution.py` models, per the brief:

- **spread**: bid/ask from quotes if present, else `typical_spread_ticks`
- **slippage**: `base_ticks` + `vol_multiplier * ATR_pct` + size-impact term
- **latency**: `latency_ms` shifts eligible fill time; stop-market orders
  additionally pay `stop_slippage_ticks`
- **partial fills**: size filled ~ min(size, participation_rate * bar_volume)
- **missed trades**: a limit entry is missed if price never trades through by
  `max_entry_wait_bars`; probabilistic miss rate also configurable for
  stress testing
- **commissions + exchange fees**: per side, per instrument
- **gap handling**: if a bar gaps past the stop, the fill is at the open, not
  the stop price (this is the single most common backtest overstatement)
- **same-bar ambiguity**: if a bar's range contains both stop and target, the
  configured `pessimistic_same_bar` (default **true**) resolves to the stop

Intrabar resolution uses finer-granularity bars when available; otherwise
the pessimistic rule is applied and the trade is tagged `ambiguous_fill=true`
so the sensitivity of results to this assumption is measurable.

---

## 10. Validation layer

- `validation/splits.py` — a `SplitRegistry` holds named date ranges with a
  `phase` (TRAIN / VALIDATION / TEST / SEALED_OOS). A `DataAccessGuard`
  wraps any data load and **raises** `SealedDataAccessError` on a sealed
  range unless given an `UnsealToken` with a logged reason. Every seal open
  is appended to an immutable audit log.

  The registry enforces four structural rules, all of which were absent in
  the first implementation and added after review:

  1. **No overlap** between a fitting phase and an evaluation phase, between
     TRAIN and VALIDATION, or between TEST and SEALED_OOS.
  2. **Chronological order.** Each earlier phase must end at or before every
     later phase begins. Non-overlap alone does not make a split honest:
     TRAIN [2020, 2021) with TEST [2010, 2011) does not overlap and is still
     a model fitted on the future. Same-phase splits are left unordered, so
     two TRAIN folds remain expressible.
  3. **A UTC seal boundary.** `_as_date` normalizes aware datetimes to UTC.
     `.date()` on an aware datetime gives its date in its own zone, so one
     instant written in two zones used to land on opposite sides of the seal.
  4. **No silent truncation.** `allowed_ranges()` returns every unsealed part
     of a request; `clip_to_allowed()` raises when a request straddles the
     seal with data on both sides, rather than returning one side and
     discarding the other.

  `assert_embargo(min_gap_days)` is available but not automatic, since the
  embargo length is a configured research choice rather than a structural
  truth. Phase 8 calls it when building folds.
- `validation/walk_forward.py` — configurable train/validate/test windows,
  rolling or anchored, with a **purge + embargo** gap between windows so a
  trade open across a boundary cannot leak. Every fold is reported
  separately; there is no aggregate-only output.
- `validation/experiment_log.py` — SQLite. Every run writes
  `experiment_id, timestamp, git_commit, config_hash, data_hash, split_id,
  phase, params_changed, reason_for_change, results_json, parent_id`.
  Querying "how many times has this config family touched the test set"
  is a first-class operation; the overfitting report in Phase 8 is generated
  from this table, not from memory.
- `validation/robustness.py` — parameter grids (score threshold 60..85,
  risk 0.25..1.0%), plus degradation runs: +slippage, wider spreads,
  N-bar delayed entry, random execution delay, random missed trades,
  and a win-rate haircut. Reports a fragility verdict based on how much
  performance decays per unit of parameter change.
- `validation/consistency.py` — the consistency score from the brief:
  dispersion-penalized stability of OOS win rate, profit factor, max
  drawdown, expectancy, across regimes and across years. Explicitly
  rewards "72% stable" over "90% in one window".

---

## 11. Monte Carlo

`monte_carlo/engine.py`, three methods:

1. **iid resample** — bootstrap trades with replacement, preserving the
   empirical R distribution.
2. **block bootstrap** — resample contiguous blocks of length L to preserve
   autocorrelation and loss clustering (the thing that actually kills
   accounts).
3. **regime-aware** — stratify by `RegimeState`, resample within regime, and
   sequence regimes from an empirical Markov transition matrix.

Default 10,000 sims, supports 100,000, vectorized with NumPy and a named
seed. Outputs: terminal-equity percentiles (5/10/25/50/75/90/95),
P(drawdown >= 10/20/30%), P(streak >= 5/10/20 losses), equity-curve fan,
drawdown distribution. Compounding vs fixed-fractional is configurable
because it changes the drawdown distribution materially.

**Stated limitation**, surfaced in every MC report: resampling historical
trades cannot produce regimes absent from the sample. MC bounds *sequence*
risk, not *model* risk.

---

## 12. Analytics, database, dashboard, live

- `analytics/metrics.py` — pure functions over `list[TradeRecord]` and the
  equity series: win/loss rate, mean/median R, expectancy, profit factor,
  Sharpe, Sortino, max/avg drawdown, streaks, recovery time, total return,
  CAGR, trade frequency, holding time, R distribution, MAE, MFE. Every
  function takes a `split_label` and refuses to mix phases silently.
- `analytics/breakdowns.py` — market, year, month, day-of-week, session,
  regime, setup, Flow Score bucket, R target. Each cell reports N, and any
  cell with N below `min_sample_n` is marked `INSUFFICIENT` rather than
  shown as a win rate (small-sample win rates are the primary way people
  fool themselves).
- `database/` — SQLite (Postgres-compatible DDL), tables: `runs`, `trades`,
  `signals`, `equity`, `experiments`, `seal_audit`.
- `dashboard/build.py` — Plotly static HTML: equity curve, drawdown curve,
  daily/cumulative PnL, Flow Score distribution with win rate overlay,
  regime performance, trade distribution, MC percentile fan, MC percentile
  table (5/25/50/75/95), walk-forward fold table, IS vs OOS panel.
- `live/paper.py` — reuses the *identical* `signal_engine` object as the
  backtest (no parallel implementation; this is enforced by a test that
  runs both paths over the same synthetic data and asserts identical
  signals). Emits LONG/SHORT/WAIT + score + confidence + entry/stop/target
  + R:R + regime + reasons + timestamp + data-quality status. Never places
  orders.

---

## 13. Phase plan and gates

| Phase | Deliverable | Gate before advancing |
|---|---|---|
| 1 | Architecture, config, core contracts, logging, experiment log, seal | All Phase-1 tests pass; config hash deterministic |
| 2 | Data interface + adapters + quality grading + feed availability report | Real data loaded and graded; session calendar verified |
| 3 | Structure, volatility, liquidity, momentum features | Lookahead audit passes for each feature |
| 4 | Order-flow features | Audit passes; no proxy substitution |
| 5 | Options-flow features | Audit passes; degradation path exercised |
| 6 | Flow Score + gates + setups | Score bounded 0..100; direction logic tested |
| 7 | Backtester + execution model | Lookahead harness green; cost model unit-tested |
| 8 | Walk-forward | Purge/embargo verified; folds reported separately |
| 9 | Monte Carlo | Distribution reproduction tests pass |
| 10 | Dashboard | Renders from real run artifacts |
| 11 | Paper trading | Backtest/live parity test passes |

No phase advances on a failing critical test.

---

## 14. Support/resistance structure: formalizing "clean price action at major levels"

Added after Phase 2 at the researcher's direction. The instruction was
"use clean price action at major support and resistance levels". Per
principle 3, that phrase cannot enter the codebase as written — "clean" and
"major" are judgments, not measurements. This section is the translation.
It defines three bounded scalars and the gates built on them.

Honest framing before the math: support/resistance is among the most widely
known and most arbitraged retail concepts in existence. Formalizing it makes
it **testable**, not profitable. This section also introduces roughly a dozen
new tunable constants, which makes it the single largest overfitting surface
in the project — every one is swept in Phase 8 against training folds only,
and the sealed window stays shut.

### 14.1 Level identification

Two sources. **Anchor levels** are objectively defined and carry zero
detection risk:

| Anchor | Definition |
|---|---|
| prior session high / low / close | from the calendar's `session_date` |
| overnight high / low | ETH range before the RTH open |
| opening range high / low | extremes of the first `or_minutes` of RTH |
| session VWAP, ±1σ | volume-weighted, anchored per `features.vwap_anchor` |
| round numbers | multiples of `round_increment` per instrument |

**Swing pivots** require detection, and this is where most retail
implementations silently acquire lookahead. Bar `i` is a confirmed swing
high *at evaluation time t* only when:

```
i <= t - k_confirm                                   (right-side confirmation)
high[i] == max(high[i-k_confirm : i+k_confirm+1])
high[i] - max(neighbours) >= m_prom * ATR[i]         (ATR-scaled prominence)
```

The `i <= t - k_confirm` condition is load-bearing. A swing high is not
*known* until `k_confirm` bars after it forms; a centred `argmax` window
evaluated at `t` reads bars after `t`. Every pivot therefore carries a
`confirmed_at_ts` and the level does not exist before it. This is checked by
the Phase 3 lookahead audit, not asserted here.

### 14.2 Clustering into zones

18245.00 and 18248.25 are one level. Pivots and anchors are merged by
single-linkage agglomeration while the price gap is below
`w = c_band * ATR_t` (default 0.25), giving a **zone** with:

```
zone_price = volume-weighted mean of members (median when no volume profile)
zone_width = max(member spread, min_width_ticks * tick_size)
```

Zone width is ATR-scaled rather than fixed in points, so the same rule works
on NQ at 50-point daily ranges and on NQ at 500-point daily ranges.

### 14.3 Significance `S ∈ [0,1]` — what "major" means

Six bounded terms, weights configurable and summing to 1:

| Term | Formula | Rationale |
|---|---|---|
| `s_touch` | `min(τ, τ_cap) / τ_cap`, `τ_cap=4` | τ = **distinct** prior touches. Distinct requires a departure of `d_sep * ATR` or `n_sep` bars between touches, so one long consolidation is not counted as twenty touches. |
| `s_reject` | `clip(median(r_j) / r_ref, 0, 1)` | `r_j` = displacement away from the zone within `h` bars of touch *j*, in ATR. A level that produced real moves matters more than one merely grazed. |
| `s_volume` | percentile rank of in-zone volume | Volume-at-price from bars: each bar's volume spread across its range. High-volume nodes are real levels. |
| `s_htf` | confirming higher intervals / available | `MarketView` already serves 15m/60m/daily. Confluence across timeframes, measured. |
| `s_age` | `exp(-a / λ)` **only when τ = 0** | An *untested* level decays with age. A level with touches is confirmed, not stale, so no decay applies. |
| `s_anchor` | 1 if the zone contains an anchor, else 0 | Anchors are referenced by many participants and are objectively defined. |

`S = Σ wᵢ sᵢ`. **"Major" is the gate `S >= s_major`** (default 0.60).

### 14.4 Cleanliness `C ∈ [0,1]` — what "clean" means

Five bounded terms, weights configurable and summing to 1:

| Term | Formula | Rationale |
|---|---|---|
| `s_eff` | `clip(E / E_ref, 0, 1)` where `E` = Kaufman efficiency ratio over `k_app` bars | A clean approach is directional. A grind into the level is not clean. |
| `s_density` | `clip(1 - ρ / ρ_max, 0, 1)` | ρ = distinct touches **within the last `k_recent` bars**. A level tested five times in thirty bars is being chewed through. |
| `s_overlap` | `clip(1 - O / O_ref, 0, 1)` | `O` = mean adjacent-bar range overlap over `k_app` bars. High overlap is churn. |
| `s_integrity` | decays with φ = failed breaks in `k_recent` | A failed break is a close beyond the zone followed by a close back inside within `h_fail` bars. Repeated pokes mean the level is not clean. |
| `s_vol` | `1 - clip(|log(ATR_t / median ATR₅₀)| / log 2, 0, 1)` | Penalizes both a panic flush and a dead tape. A clean test happens in ordinary volatility. |

`C = Σ vᵢ sᵢ`. Gate: `C >= c_min` (default 0.55).

**The deliberate tension between S and C.** Historical touches *raise*
significance; recent touches *lower* cleanliness. That is not an
inconsistency — it is the whole distinction between a well-established level
and a level under active attack, and it is why these are two scores rather
than one. A level with τ=6 spread over 400 bars and ρ=0 in the last 30 is
both major and clean. The same τ=6 packed into the last 30 bars is major and
filthy, and the system declines it.

### 14.5 Rejection confirmation `R ∈ [0,1]` — the trigger

One hard requirement plus three scored terms:

```
REQUIRED (binary gate, not scored):
    the bar's extreme entered the zone        (low <= z_hi for support)
    AND the bar closed back outside it        (close > z_hi for support)

s_close : clip((p - 0.5) / 0.5, 0, 1),  p = (close - low) / (high - low)
s_disp  : clip(|close - zone_price| / (disp_ref * ATR), 0, 1)
s_flow  : delta sign agrees with the trade direction
```

`s_flow` is **dropped, with its weight not redistributed**, when no tick feed
exists — consistent with `strict_component_availability`. `MarketView.deltas()`
returns empty rather than estimating, so this degrades honestly.

### 14.6 Wiring: gates, score, stop, and target

Gates run before scoring, in the existing pipeline position (§3,
`StructureGate`):

```
zone exists within entry_atr_window * ATR of price   else WAIT("no_structure")
S >= s_major                                          else WAIT("no_structure")
C >= c_min                                            else WAIT("no_structure")
rejection binary requirement met                      else WAIT("no_structure")
```

The STRUCTURE component's 20 points then come from
`magnitude = w_S·S + w_C·C + w_R·R`, with `direction = +1` at support and
`-1` at resistance. Gates express "must have"; the score expresses "how
good". Using a product instead would let one weak term zero an otherwise
strong setup, which is a gate's job, not a score's.

**Stop placement becomes determined rather than chosen:**

```
long at support:  stop = min(z_lo, rejection_bar.low) - b_stop * ATR
short at resistance: stop = max(z_hi, rejection_bar.high) + b_stop * ATR
```

This is the point of the whole section. R is defined as `|entry - stop|`, and
the stop now sits where the trade thesis is actually falsified — beyond the
level — rather than at an arbitrary ATR multiple.

**Target is the next opposing zone with `S >= s_major`.** This has a
consequence worth stating plainly: **setup selection becomes a measurement,
not a parameter.** The distance to the next significant level, divided by the
stop distance, *is* the achievable R:R:

```
achievable_rr = |next_opposing_zone_price - entry| / |entry - stop|

achievable_rr < min_reward_risk        -> WAIT("rr_too_low")
1.0 <= achievable_rr < 1.8             -> SCALP_1R
1.8 <= achievable_rr < 2.7             -> SETUP_2R
achievable_rr >= 2.7                   -> DIRECTIONAL_3R
```

A setup is no longer assigned by configuration; it is read off the structure.
When the next level is 0.8R away the trade is simply declined, which is the
mechanism that stops a "1R scalp" from quietly becoming a 0.4R target against
a full-width stop — the exact failure mode §0.2 and
`TradeRecord.r_label_is_honest()` exist to catch.

### 14.7 What would falsify this

Recorded now, before any result exists, so the test is not chosen after
seeing the data:

1. If win rate does not increase monotonically with `S` across its buckets,
   "major" is not measuring significance.
2. If win rate does not increase with `C`, "clean" is not measuring
   anything — the approach-quality terms are noise and should be removed
   rather than reweighted.
3. If `s_density` and `s_touch` have the same sign of association with
   outcome, the S/C split is unjustified and collapses to one score.
4. If performance requires `s_major > 0.75`, the level population is too
   small to trade and the result is a small-sample artifact.
5. If the zone-width constant `c_band` changes expectancy by more than
   `fragility_max_relative_drop` across ±1 step, the whole construction is
   fragile and should be reported as such.

---

## 15. Pre-registration: `validation/hypotheses.py`

Sections 6 and 14.7 both say their criteria are recorded before any result
exists. This section is where that stops being a promise in prose.

The failure mode being prevented is specific and extremely common: run the
backtest, look at the output, then decide which comparison counts as success.
A criterion chosen after the data is seen is not a test of anything. Since
the brief states targets *and* forbids optimizing toward them, the only way
both hold is if the targets are fixed in advance, machine-readable, and
unreadable by the code being judged.

Nine hypotheses are registered: the five from 14.7 that would falsify the
support/resistance construction, the brief's three headline numbers, and the
consistency-over-peak claim from section 10. Each carries:

| Field | Why it is there |
|---|---|
| `direction` | The *shape* of the prediction, which decides how it is refuted. A monotone claim is refuted by a non-monotone profile, not by a low level. |
| `threshold` | The bound, for `AT_LEAST` / `AT_MOST` only. Present exactly when the shape needs one, so a shape claim cannot be scored against a number by mistake. |
| `refuted_when` | The refutation condition, stated so it can be checked without judgment. |
| `consequence` | **What is done if refuted.** 14.7 (2) requires refuted cleanliness terms to be *removed*, not reweighted. A hypothesis with no stated consequence gets quietly reweighted instead, which is the overfitting loop in another costume. |
| `expected_to_hold` | My prior, pinned so it cannot be revised once results arrive. Six of nine are expected to be refuted. |

### 15.1 Why it is enforced rather than documented

- The id tuple and count are pinned literally in `test_hypotheses.py`, so a
  late addition fails a test instead of passing silently. A second test
  perturbs the tuple exactly as an addition would, so the pin is not vacuous
  against a stale constant.
- `test_no_target_leakage.py` bans `hypotheses` / `HYPOTHESES` /
  `Hypothesis` from `features`, `signals`, `regime`, `risk`, `backtest`,
  `monte_carlo` and `data`, alongside `ResearchTargets`. Same reasoning: a
  rule that can read the number it will be judged against will reproduce
  that number given enough iterations, and demonstrate nothing.
  Pre-registration has force only while the thing being tested cannot read
  the test.
- `flow-model hypotheses register` is idempotent, because a duplicate row
  would corrupt the multiple-comparisons denominator in
  `overfitting_summary()` — the one number that says how much the reported
  results should be discounted.
- It **refuses** once TEST or SEALED_OOS has been touched. At that point a
  "pre-registration" is nothing of the kind, and accepting it silently is
  exactly the self-deception the registry exists to prevent.
- Registration logs under TRAIN, so writing down a prediction spends no
  evaluation budget, and is commit-stamped, so "this predates the evidence"
  is checkable against git rather than taken on trust.

### 15.2 The expected outcome is refutation

`H-REGIME-WINRATE` (88% at 1R in favourable regimes) is registered because
it is testable, not because it is plausible. A symmetric 1R system at that
rate implies an annualized Sharpe far outside anything documented. A refuted
hypothesis recorded in advance is a real research result; a confirmed
hypothesis chosen in arrears is not a result at all.
