# Early Signal Engine — final report

**Status: BUILT, NOT VALIDATED. The bot remains paper-trading only, and
`EARLY_SIGNAL_MAY_TRADE` is `false`.**

The engine is complete, wired into the live path, and covered by tests. It
has produced **zero measured outcomes**, because it has not been run
against a live market yet — there is no database in this checkout. Items
6 through 16 below are therefore all **INSUFFICIENT DATA**, and that is
the honest answer rather than a gap to be filled with a plausible number.

---

## 1. Early features implemented

**26 measurable features**, feeding **9 weighted factors**. Each carries
its provenance (`candles`, `observations`, `snapshot`) and an `available`
flag.

| Source | Features |
|---|---|
| Candles | `volume_accel_short`, `volume_accel_medium`, `volume_steadiness`, `relative_volume`, `range_compression`, `higher_lows`, `acceleration_smoothness`, `breakout_proximity`, `ema_slope`, `ema_separation`, `rsi_level`, `rsi_crossing_up`, `macd_histogram_expanding`, `vwap_position`, `return_short`, `return_medium`, `return_long` |
| Successive snapshots (flow) | `txn_rate_change`, `buy_pressure`, `buy_pressure_change`, `buy_pressure_persistence`, `liquidity_growth`, `liquidity_stability` |
| Single snapshot | `volume_to_liquidity`, `liquidity_to_marketcap`, `token_age_hours` |

The flow features need **two stored observations at least a minute
apart**. They are unavailable on a token's first sighting by construction,
which is why `TokenObservation` is written on every pass.

## 2. Features rejected, and why

Named explicitly in `app/early/features.py::UNAVAILABLE_FEATURES` rather
than approximated:

| Wanted | Why not implemented |
|---|---|
| `unique_buyers` | Needs a wallet-level indexer. A transaction count is not a participant count. |
| `new_buyers` | Needs a wallet-level indexer. |
| `repeat_buyers` | Needs a wallet-level indexer. |
| `wallet_concentration` | Needs holder distribution beyond the top-10 the rug engine already reads. |
| `order_book_depth` | AMMs have no order book. Transaction flow is a proxy and is labelled as one. |
| `social_activity` | No trustworthy provider configured. |

None of these is estimated from something else and presented as the real
thing. A factor with no data is marked unavailable and counted against the
missing-data budget.

## 3. Early Opportunity Score formula

`app/early/score.py`. A weighted average of nine factors, each scored to
`[0, 1]`, then scaled to 0–100:

```
score = 100 × Σ(factor_score × weight) / Σ(weight)
```

| Factor | Weight |
|---|---|
| volume_acceleration | 0.18 |
| transaction_acceleration | 0.14 |
| buy_pressure | 0.14 |
| volume_quality | 0.10 |
| liquidity_quality | 0.10 |
| momentum_acceleration | 0.10 |
| price_structure | 0.09 |
| breakout_position | 0.08 |
| relative_volume | 0.07 |

**These weights are unvalidated priors** chosen from reasoning about
microstructure, not measured from outcomes.

Missing data uses `NEUTRAL = 0.5` with `available=False`; above
`MAX_UNAVAILABLE_WEIGHT = 0.40` the whole score is marked `reliable=False`
rather than reported as a number.

Two factors are deliberately **non-monotonic**, which is what makes this an
*early* score rather than a momentum score:

- **Volume acceleration** peaks near 3× and *falls* above 8× — a token
  already exploding is a late signal, not an early one.
- **Breakout position** scores 1.0 *at* the range high and 0.1 at +15%
  above it — the move already happened.

## 4. Late Entry Risk formula

`app/early/late_entry.py`. A **separate** 0–100 score that can veto an
entry and is **never averaged into** the early score. Averaging would let a
strong early reading cancel a clear "you are too late" reading, which is
precisely the trade the engine exists to avoid.

Sum of triggered flag weights, capped at 100:

| Flag | Weight |
|---|---|
| price_extended | 30 |
| above_breakout | 22 |
| rsi_stretched | 16 |
| volume_peaked | 14 |
| liquidity_deteriorating | 14 |
| buy_pressure_falling | 12 |
| far_above_vwap | 12 |
| ema_separated | 10 |
| momentum_slowing | 10 |

Stage classification reads risk **together with distance travelled**,
because the two disagree informatively — a token up 5% with several
warnings is a weak setup, not a late one:

```
not assessable          → LATE          (never enterable)
risk ≥ 70 or +150%      → OVEREXTENDED  (never enterable)
risk ≥ 45 or  +60%      → LATE
breakout > +3%          → CONFIRMED     (enterable)
+8% or volume accel>1.5 → DEVELOPING    (enterable)
otherwise               → EARLY         (enterable)
```

**A candidate that cannot be assessed is LATE**, not EARLY: "we cannot
tell" must not become an invitation to enter.

## 5. WATCH logic

```
DISCOVERED → WATCH → CONFIRMED → PAPER_BUY
          ↘ WATCH → FAILED / EXPIRED
```

"Promising" and "ready" are different facts. A two-state bot has to
collapse them — buy every promising candidate at once (chasing) or discard
it (missing the move). WATCH is the third state.

- Re-scored every `WATCHLIST_INTERVAL_SECONDS` (default 120s) by
  `app/early/loop.py`.
- Every evaluation appends to `score_history`, so "does an *improving*
  score beat a high static one?" becomes answerable later.
- Confirmation is handed to `handle_discovered_token` — **the normal buy
  path**, never a parallel one, so the kill switch, risk gate, rug check,
  market-quality gate, exposure caps and fill model all still apply.
- Failures are **recorded, not deleted**, with one of eight categories
  (`volume_disappeared`, `buy_pressure_reversed`, `liquidity_fell`,
  `security_deteriorated`, `failed_breakout`, `became_late`,
  `score_decayed`, `expired`). A watchlist that dropped its
  disappointments would be permanently unfalsifiable.
- A missing price snapshot leaves the entry alone rather than failing it —
  one bad minute from the price feed must not empty the watchlist and fill
  the false-positive table with the bot's own plumbing.

### Security ordering

Security runs **before a single early feature is computed**. On a security
failure the engine returns immediately, so `verdict.early is None` — there
is no early score that could override anything. Verified by
`test_a_security_failure_produces_no_watchlist_entry`.

## 6. Historical sample size

**ZERO.** There is no database in this checkout and the bot has not been
run against a live market. No token has been scored, no watchlist entry
created, no forward return resolved.

## 7. Score calibration results

**INSUFFICIENT DATA.** `build_early_calibration` runs and reports
`"INSUFFICIENT DATA at {h}m: fewer than two early-score buckets have 30+
measured outcomes. The Early Signal Engine has not been tested."` —
executed against an empty database and confirmed.

Buckets are `<50, 50-60, 60-65, 65-70, 70-75, 75-80, 80+`, judged on
after-cost expectancy, and the verdict distinguishes INSUFFICIENT DATA /
NO EDGE / separates outcomes / weak ranking / **INVERTED**.

## 8. Forward-return results

**INSUFFICIENT DATA.** The machinery is in place and covers the part that
usually gets skipped: forward returns are scheduled for **rejected**
candidates too, because a dataset of only the setups that cleared the bar
cannot say whether the bar is in the right place. MFE and MAE are sampled
on the path, so a +5% horizon return that first went −30% is visible as a
stopped-out loss rather than a win. An unmeasurable horizon stays NULL
with a reason and is never counted as 0%.

## 9. Lead-time analysis

**INSUFFICIENT DATA.** `build_lead_time` reports the share of tokens that
reached +10% / +20% / +50% which the engine had flagged *beforehand*, plus
median lead in minutes. Requires ≥30 tracked signals; currently 0.

## 10. False-positive rate

**INSUFFICIENT DATA.** `build_false_positives` groups resolved watchlist
entries by failure category and by early-score bucket. Requires ≥30
resolved entries; currently 0. It also warns when more than half of
failures land in the residual `score_decayed` bucket, which would mean the
taxonomy is missing a category rather than that they simply faded.

## 11. Feature-ablation results

**INSUFFICIENT DATA on real data.** The harness
(`app/research/early_ablation.py`) is built and verified on seeded data
with a known answer.

It reads features **stored at signal time** rather than replaying candles.
It has to: four of the nine factors come from differencing market
snapshots, which historical candles do not carry, so a candle backtest
would silently mark them unavailable and then report that removing them
changes nothing — a conclusion about the backtest, not the factors.

Two measurement traps were found and closed while building it:

- **Bucket separation is fooled by a pure score shift.** Removing a factor
  can push a group across a fixed bucket edge and post a 30-point drop
  while losing no information at all. The verdict comes from **Spearman
  rank correlation**, which has no edges; separation is shown for
  continuity and labelled. Locked in by
  `test_a_bucket_edge_artifact_does_not_become_a_verdict`.
- **"Changed nothing" and "had no data to remove" are the same number and
  opposite findings.** Per-factor coverage is computed from the stored
  rows, and a factor with zero coverage reads `NO DATA — never
  measurable`, not `redundant`.

Known limitation, stated on the report itself: re-scoring stored features
answers "would a different weighting have *ranked these same candidates*
better". It cannot answer "which tokens would a different weighting have
*found*" — those counterfactual tokens were never recorded.

## 12. Walk-forward results

**INSUFFICIENT DATA.** `app/research/early_walkforward.py` refits the entry
threshold on each training window and grades it on the next, over rows
sorted by observation time. Executed against an empty database; reports
`"The early threshold has NOT been walk-forward validated, and no value
for it is supported yet."`

It flags a threshold that moves more than 10 points between windows as
fitting noise **even when each individual fit looks profitable**, and
reports a stable negative result as a finding rather than a failure of the
method.

## 13. Out-of-sample expectancy after costs

**NOT MEASURED.** No out-of-sample window exists.

## 14. Profit factor

**NOT MEASURED.** Reported as `None` rather than infinity when there are no
losers yet — a small sample with no losses is a small sample, not an edge.

## 15. Maximum drawdown

**NOT MEASURED** for the early engine. No early-signal trade has been
taken, in paper or otherwise.

## 16. A vs B vs C comparison

**NOT CONCLUSIVE, and partially unmeasurable by construction.**

`app/research/strategy_modes.py` compares A (technical only), B (early
only) and C (early finds, technical confirms) on `robust_score` — OOS
expectancy in R, penalised for the train-to-OOS gap and for drawdown.
Lead time is reported alongside rather than folded into the ranking: a
mode that gets in earlier and still loses money has not won anything.

**Modes B and C are approximated.** The backtest engine walks a
`CandleSeries`, which carries no market snapshots, so B and C are stood in
for by tightening the technical threshold and are evaluated on their
candle-derived features only. Every result carries `flow_available=False`
recording that. A real comparison needs the bot to have run in paper and
stored observations.

## 17. Best validated configuration

**None. Nothing has been validated.**

`EARLY_SIGNAL_MAY_TRADE` stays `false`, which means the engine can raise a
token to WATCH and can never open a position on its own. The existing
technical strategy remains the only thing that trades.

The shipped defaults are hypotheses, not recommendations:

```
EARLY_SIGNAL_ENABLED=true
EARLY_SIGNAL_WATCH_THRESHOLD=55
EARLY_SIGNAL_CONFIRM_THRESHOLD=70
EARLY_SIGNAL_REQUIRE_TECHNICAL=true
EARLY_SIGNAL_TECHNICAL_MARGIN=25
EARLY_SIGNAL_MAY_TRADE=false
WATCHLIST_INTERVAL_SECONDS=120
WATCHLIST_MAX_AGE_HOURS=12
WATCHLIST_MAX_SIZE=200
OBSERVATION_RETENTION_HOURS=48
```

## 18. Remaining weaknesses

**Two real defects were found by writing the integration tests, and both
would have been invisible from the outside:**

1. **The technical gate returned before the early engine ran.** The engine
   would only ever have seen candidates the bot was already about to buy —
   the exact opposite of its purpose. Fixed: a technical rejection now
   carries a flag, security and market quality still run first, the engine
   gets its look, and the function returns before sizing.
2. **Both candle fetches imported `fetch_live_series`, a name that has
   never existed.** The surrounding `except Exception` swallowed the
   `ImportError`, the series stayed `None`, and every candidate then failed
   the data-quality gate and was skipped. The engine ran, logged nothing
   alarming, and could not score anything. Fixed to `fetch_candles`, the
   handler now logs at warning, and a regression guard asserts the name
   resolves.

Still open:

- **No live evidence of any kind.** Everything in items 6–16 needs the bot
  to run. Nothing here shows an early edge exists.
- **The nine weights are guesses.** Ablation cannot correct them until
  there are stored outcomes.
- **Flow features are approximations of order flow**, not order-book data,
  and are labelled as such. Real participant counts need a wallet-level
  indexer the bot does not have.
- **The A/B/C comparison cannot be completed from candles.** It needs a
  paper run with stored observations.
- **Ablation cannot see counterfactual candidates** — only re-rank the ones
  the live engine happened to record.
- **`EARLY_SIGNAL_TECHNICAL_MARGIN=25` is itself an unvalidated choice.**
  It bounds the extra API cost of looking at rejected candidates; whether
  25 points is the right window is unknown.
- Carried over from earlier work and still true: six modules bypass the
  retry/health wrapper (GoPlus, Honeypot, RugCheck.xyz, Jupiter, EVM,
  notifier); cross-check and correlation are built but not wired into the
  buy path; regime-conditional performance, shadow mode, experiment
  tracking and streaming market data are not done.

## 19. Exact files changed

38 files, +6,646 / −32, across commits `b6d1f38 … ff1d69e`.

**New — engine (`app/early/`)**
`__init__.py`, `features.py`, `score.py`, `late_entry.py`, `classifier.py`,
`engine.py`, `watchlist.py`, `loop.py`

**New — analysis and research**
`app/analysis/early_calibration.py`, `app/research/early_ablation.py`,
`app/research/early_walkforward.py`, `app/research/strategy_modes.py`

**New — dashboard**
`app/dashboard/templates/early.html`

**New — tests**
`tests/test_early_signal.py`, `tests/test_watchlist.py`,
`tests/test_early_research.py`, `tests/test_early_wiring.py`,
`tests/test_early_loop.py`

**New — docs**
`docs/EARLY_SIGNAL_REPORT.md`

**Modified**
`app/models.py` (added `TokenObservation`, `WatchlistEntry`; seven new
`ForwardReturn` columns), `app/analysis/forward_returns.py` (MFE/MAE,
outcome labels, `attach_early`), `app/analysis/calibration.py` (5m
horizon), `app/analysis/token_detail.py`, `app/services/trading_service.py`,
`app/early/loop.py`, `app/main.py`, `app/config.py`, `app/dashboard/routes.py`,
`scripts/research.py`, `.env.example`, `README.md`, `tests/conftest.py`,
`tests/test_dashboard.py`, `tests/test_score_calibration.py`, and the six
dashboard templates (nav link).

## 20. Test results

```
968 passed, 1 warning in 106.08s
```

Actually executed, not asserted. Beyond the unit tests:

- **Migration verified against a simulated pre-upgrade database**: both new
  tables created and usable through the ORM, all seven new
  `forward_returns` columns added, all eleven new indexes created, and a
  pre-existing row reads `NULL` in every new column rather than a
  fabricated zero.
- **Every new CLI command executed** against an empty database
  (`early`, `early-ablate`, `early-walkforward`): each reports
  INSUFFICIENT DATA rather than a number.
- **Ablation executed on seeded data with a known answer**: it identified
  the one factor carrying the signal, and correctly did *not* promote the
  bucket-edge artifact to a verdict.

---

## The finding

No measurable early predictive edge has been demonstrated, because none
has been measured. The engine, the labels, the calibration, the ablation
and the walk-forward all exist and all report the absence of evidence
plainly. The next step is to run the bot in paper and let the dataset
accumulate — not to tune anything.

**Paper trading only. No real funds, wallet or private key is involved
anywhere in this bot.**
