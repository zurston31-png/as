# Provenance audit — every published metric, and what produced it

**Status of this document:** Gate 1 of the audit mandate. Kept at the repository root beside `ARCHITECTURE.md` rather than in `reports/`, which is gitignored for generated run artifacts; this is a hand-written audit, not an artifact. Written 2026-10-13 at
commit `bdfa34e`, branch `flow-model-phase-1`.

**The one-line finding:** *every quantitative result this project has produced
came from a synthetic generator.* No real market bar has ever entered the
system. Nothing below is evidence about markets. It is evidence about code.

---

## 0. Why this document exists

Until now the project's measurements lived in commit messages and conversation.
That is not traceable provenance: a reader cannot tell which dataset produced a
number, whether the number was measured or derived, or whether it was ever
independently verified. Several numbers below were also *corrected* after first
publication, and a scattered record hides corrections.

Three columns matter and are given for every row: **what produced it**, **on
what data**, and **who checked it**. A number measured by an agent and never
re-run by anyone else is marked as such, because two agent-reported figures in
this project were wrong (see §5).

---

## 1. The dataset behind every number

| Property | Value |
|---|---|
| Generator | `flow_model/data/synthetic.py::SyntheticAdapter` |
| Config | `SyntheticConfig()` — all defaults, never varied |
| Seed | `20240101` (`config.seed`) |
| Instrument | `NQ` spec from `flow_model/config/defaults.yaml` (0.25 tick, $5 tick value → $20/point, CME) |
| Interval | 300 s (5-minute) primary |
| Feeds emitted | bars, quotes, tick_aggregate, options_snapshot |
| Real market bars | **zero, ever** |
| `data/` directory | does not exist |
| `reports/` artifacts | none before this file |
| Experiment-log DB | never created on disk |
| Seal audit log | never created; 0 of 1 opens spent |
| Backtest runs persisted to SQLite | zero |

**Two synthetic-specific caveats that limit what the numbers can mean**, both
already recorded in `75614dd` and repeated here because they are easy to lose:

1. The generator's **tick stream is drawn from the same innovations as its
   bars**. Order-flow features computed from ticks are therefore not
   statistically independent of the bars they are compared against. Any
   correlation between an order-flow feature and a synthetic forward return is
   an artifact of shared innovations, not a signal.
2. The options snapshot is generated, not observed. Options-flow magnitudes are
   mechanically valid and economically meaningless.

So the synthetic runs validate **mechanics** — bounds, determinism, ordering,
refusal behaviour, n=0 handling, arithmetic — and nothing else.

---

## 2. Published metrics, mapped

Legend for **Checked by**: `agent` = reported by a subagent only; `me` = I
re-ran it myself; `me (corrected)` = agent figure was wrong and I replaced it.

### 2.1 Regime detector — commit `a154e99`

Module `flow_model/regime/detector.py`. Dataset: synthetic, 9459 bars.

| Metric | Value | Checked by |
|---|---|---|
| Overall accuracy | 0.3374 (3191 / 9459 bars) | me |
| Majority baseline (always CHOP) | 0.2101 | me |
| Cohen's kappa | 0.2017 | me |
| Balanced accuracy | 0.3097 (chance 0.1667) | me |
| LOW_VOL precision / recall | 0.5039 / 0.6551 (n=1954) | me |
| HIGH_VOL | 0.4695 / 0.4037 (n=1640) | me |
| CHOP | 0.2972 / 0.4625 (n=1987) | me |
| TRENDING_UP | 0.4385 / 0.0977 (n=1750) | me |
| TRENDING_DOWN | 0.3167 / 0.0585 (n=1846) | me |
| DIRECTIONAL_SHIFT | **0.0302** / 0.1809 (n=282) | me |
| Hysteresis effect | **costs** 1.4 accuracy points | me |

**Reading.** Accuracy beats the majority baseline (0.3374 vs 0.2101) and kappa
is positive but weak. DIRECTIONAL_SHIFT's precision **equals its base rate** —
that label carries no information at all. "Ground truth" here is the generator's
own regime process, so this measures whether the detector can recover a known
generative process, which is the easiest possible version of the task.

### 2.2 Feature and component layer — commits `14b4d9d`, `95cddd4`, `75614dd`

| Metric | Value | Checked by |
|---|---|---|
| §14 cleanliness, spread touches (400 bars, rho=0) | C = 0.80, gate passed, SCALP_1R | agent |
| §14 cleanliness, packed touches (last 30, rho=6) | C = 0.342552, GATE_CLEANLINESS | agent |
| §14 significance, 1-of-1 vs 1-of-2 agreement | S = 0.825 vs 0.75 | agent |
| Order-flow on bars-only: keys emitted | 15 keys, **all zero**, quality `MISSING`, weight **not** redistributed | me |
| Options-flow cap binding | uncapped 0.514693 → capped **0.500000** | me |
| Bars-only availability | **55.0 of 100.0** points | me |
| Order-flow with ticks | points 9.87, magnitude 0.395, quality GOOD | agent |

The two rows marked `me` are the Phase 4 and Phase 5 gates; I verified both
directly rather than accepting the report.

### 2.3 Zero trades — commit `f952d3e`, superseded by `fe6c31c`

| Metric | Value | Dataset | Checked by |
|---|---|---|---|
| Phase 6 run | WAIT on **all 2500** consecutive bars | synthetic | me |
| Flow score emitted on | 80 of 2500 bars | synthetic | me |
| Max Flow Score | 65.23 vs lowest threshold 70.0 | synthetic | me |
| Scores outside [0,100] | none | synthetic | me |
| Bars-only sub-run | 627 of 627 bars WAIT | synthetic | me |

### 2.4 Gate funnel — commit `fe6c31c`

Dataset: synthetic NQ, 2016-01-04 → 2016-12-30, **19542 bars**, committed
default config. Counted twice: once from the loop's counters, once recounted
independently from the 19542 persisted signal rows. Checked by **me**.

| Stage | Reached | Blocked here |
|---|---|---|
| session | 19542 | 792 |
| market_data | 18750 | 75 |
| warmup | 18675 | 1425 |
| regime | 17250 | **5956** |
| liquidity | 11294 | 0 |
| structure (§14) | 11294 | **10550** |
| options_flow | 744 | 0 |
| order_flow | 744 | 380 |
| momentum | 364 | 81 |
| flow_score | 283 | 283 |

Structure rejections split: `no_structure` 8888, `rr_too_low` 1662.
Scores among the 283 that reach the scorer: max **66.991221**, min 27.729515,
mean 49.530303, none outside [0,100].

**This corrected an earlier published claim.** `f952d3e` and `8df4223` reported
the Flow Score threshold as the binding constraint. It binds for 1.4% of bars;
§14 structure rejects 37× more. The correction is recorded in `fe6c31c`.

**Known limitation, unresolved:** those 283 scores are a *selected subsample* —
they survived eight prior gates. They do not establish the population
distribution. Scoring every post-warmup bar is outstanding work (mandate D).

### 2.5 Execution costs — commit `8df4223`

Dataset: synthetic NQ, $400 R. Checked by **me (corrected)** — see §5.

| Metric | Value |
|---|---|
| Round turn through the STOP | $47.74 = **0.1194R** |
| Round turn through the TARGET | $21.49 = **0.0537R** |
| Gap rule cost, worst single bar | $1200 |
| Same-bar resolution swing | $2640 |

**Breakeven win rates.** `8df4223` published one figure, correctly scoped to the
1R case: `p = (1 + 0.1194) / (2 + 0.1194 − 0.0537) = 54.19%`. It is **not**
universal — it is conditional on a 1:1 payoff. The project runs three setups, so
all three are derived here from the same measured costs:

| Setup | Costless breakeven | With measured costs | Shift |
|---|---|---|---|
| SCALP_1R | 50.00% | **54.1899%** | +4.19 pts |
| SETUP_2R | 33.33% | **36.5137%** | +3.18 pts |
| DIRECTIONAL_3R | 25.00% | **27.5328%** | +2.53 pts |

These three are **derived arithmetic**, not measurements, computed from the two
measured cost figures above — which are themselves synthetic. Formula:
`p = (L + c_stop) / ((W − c_target) + (L + c_stop))` with L = 1.

### 2.6 Defects found, with measured impact — commits `ff46c86`, `f44ddb9`

| Defect | Measured wrong value | True value | Direction |
|---|---|---|---|
| Uninspected bars between early fill and horizon end (**HIGH**) | TARGET exit, **+0.9869R** | STOP, **−1.0756R** | flattering by 2.06R |
| — same trade, `mae_r` | −0.5 | −1.5 | understated risk |
| — same trade, `bars_held` | 5 | 7 | corrupts holding-time analysis |
| Stale high-water mark on final equity row | `drawdown_pct` **+0.0038**, hwm 100000.0 vs equity 100379.76 | drawdown cannot be positive | understates peak |
| In-flight blindness for heat | $800 open heat vs **$500** cap | within cap | 60% over a hard limit |
| Second `run()` reuses limit state | 1 trade, then 0 on identical data | identical | breaks determinism |
| Non-finite equity mark | `total_return` 0.039, `cagr` 0.43 across a NaN curve | undefined | publishes numbers for garbage |
| Sub-micro-R loss | `breakeven_win_rate` **0.0** | undefined | "breaks even at a 0% win rate" |

### 2.7 Test suite

| Commit | Count |
|---|---|
| `f952d3e` → `8df4223` | 2813 |
| `fe6c31c` | 2897 |
| `f44ddb9` | 3075 |
| `bdfa34e` (current) | 3075 passed, 0 failed, 0 skipped |

---

## 3. Reports that could misrepresent synthetic findings as real

Per mandate A. These are the live hazards:

1. **Commit messages state figures without the word "synthetic" in every
   instance.** `a154e99`'s subject line is *"it scores 0.337 against ground
   truth"* — true, but "ground truth" is the generator's own process. Mitigated
   by this document, not by rewriting pushed history.
2. **The 54.19% breakeven has been quoted in isolation**, including by me in
   conversation, without its 1R scoping. §2.5 fixes that.
3. **`r_label_is_honest` and the setup names imply payoff structures** that have
   never been observed, because zero trades have occurred.
4. **No run artifact exists on disk**, so no figure above can be regenerated by
   a third party without re-running the code. That is the main provenance gap
   this document cannot close by itself.

---

## 4. The R definition — RESOLVED 2026-10-17

Four sources carried **three distinct definitions** of `TradeRecord.risk_dollars`,
the denominator of every R in §2. Nothing validated the field, which is why they
coexisted undetected.

| Source | Stop distance from | Size | Verdict |
|---|---|---|---|
| `ARCHITECTURE.md` §0.2 (as written) | `entry_price` (realized) | `size` (requested) | ambiguous wording, rewritten |
| `tests/unit/test_contracts.py` | planned | requested | **correct** |
| `tests/unit/test_execution.py` | planned | requested | **correct** |
| `flow_model/backtest/ledger.py` | realized | filled | **the defect** |

**The resolution.** `TradeIntent._check_geometry` already enforced
`risk_dollars == |entry_price − stop_price| * point_value * size` on the
**planned** entry and **requested** size — at the point R is actually decided.
The ledger then silently re-derived it from the fill, which is exactly the
revision §0.2 forbids. So the ledger was the defect, not the doc.

Now:

- `risk_dollars` is **carried** from the intent, never recomputed. `TradeRecord`
  validates the identity, so a wrong value is a construction error rather than a
  plausible number in a report.
- `OpenPosition.carried_risk_dollars` (realized entry, filled size) is a
  **separate** field, and `Ledger.open_risk_dollars()` sums that for §8's
  portfolio heat. One field could not serve both: R against the realized entry
  erases execution quality, and a hard limit on planned risk understates the
  money at risk.

**Effect on the figures in §2: none.** Phase 7a already used the planned
denominator — $47.74 / $400 = 0.119350R against 0.1194 published, and
$21.49 / $400 = 0.053725R against 0.0537. The three breakeven rates in §2.5
therefore stand unchanged. Had the ledger's convention been the one adopted,
every stop would have reported exactly −1.00R and entry slippage would have
vanished from the R distribution.

**Reachability of the old defect:** it affected any trade whose fill differed
from plan. At the committed default (market entry, `max_entry_wait_bars=1`,
`size=1`) the realized entry still differs from the planned entry whenever the
bar's open is not the planned price, so this was live — but zero trades have
ever been taken, so no published figure was computed through it.

## 4a. THE POPULATION FLOW SCORE DISTRIBUTION — the zero-trade diagnosis

Measured 2026-10-19 by me, synthetic NQ, committed default config. Nothing
moved; production code untouched. Two windows: a quarter and the full year. The
full year is the one to read — the quarter understates the maximum materially,
and my first write-up of this section drew a stronger conclusion than the data
supports. Corrected below.

### Why this measurement was needed, and why it is well posed

§2.4 flagged that the 283 published scores are a selected subsample — they
survived eight gates — so they could not establish the population
distribution. The blocker looked structural: the pipeline short-circuits at the
§14 structure gate, and `StructureGate` is the only source of the trade `side`.

**That blocker is not real.** `SignalEngine.score(features)` takes *only* the
feature vector. The side is consumed by the options-flow and order-flow
*gates*, not by the scorers. So the Flow Score is well defined for every bar
that has a feature vector — every bar past the regime gate — and can be
computed through the engine's own public `score()` and `aggregate()` without
bypassing or weakening a single gate.

### The result

**Full year, 2016-01-04 → 2016-12-30, 19542 bars on the clock:**

| | published today | full population |
|---|---|---|
| n | **282** | **11239** (40× more) |
| min | 27.7295 | 16.3393 |
| median | 49.8788 | 43.2000 |
| p95 | 60.6253 | 56.8005 |
| p99 | 64.1394 | 61.2305 |
| **max** | **66.9912** | **69.6335** |
| mean / sd | 49.5303 / 6.9429 | 43.2333 / 8.1288 |
| **bars ≥ 70.0** | **0** | **0 (0.0000%)** |
| out of [0,100] | 0 | 0 |

Quarter only (2016-01-04 → 2016-03-31), for comparison: published n=76 max
59.8702; population n=1979 max 64.8123. **The quarter understates the
population maximum by 4.8 points**, which is why the year is the figure of
record.

Also, over the year: 8248 bars had no feature vector (blocked before
liquidity), and 55 had their score **refused** by the aggregate rather than
estimated — honest degradation working as §5 requires.

### What it settles, and what it does not

**Settled: the scorer is not being filtered to an unrepresentative slice.** The
opposite — the gates select *higher*-scoring bars than the population (mean
49.53 vs 43.23). The 282 published scores are representative of the top of the
distribution, not of a biased corner of it.

**Settled: the gates are not monotonic in score.** Population max 69.6335
exceeds published max 66.9912, so bars rejected at structure, order-flow or
momentum score higher than any bar that reached the scorer.

**NOT settled, and this is the correction.** My first draft of this section read
the quarter's 64.81 maximum and concluded the Flow Score "cannot reach 70".
The full year's maximum is **69.6335** — short of 70.0 by **0.37 points**, out
of 100. That is a materially weaker claim. The right statement is:

> Over 11239 scored bars in one year, zero reach 70.0, and the best misses by
> 0.37 points. The threshold sits just above the observed maximum, not far
> above it.

The distribution's right tail therefore *approaches* the threshold. Over a
longer window than one year, bars above 70.0 should be expected — rarely. How
rarely is a tail question this measurement does not answer, and the honest
reading is that the system would trade at some very low frequency rather than
never. Both readings are bad for a research programme, but they are different
findings and should not be conflated:

* "cannot reach 70" would mean the specification is internally broken.
* "reaches 69.63 in a year" means the setup is extremely rare, so a
  ten-year backtest would turn on a handful of trades — far too few for the
  statistical validation Phases 8 and 9 are built to perform, and far too few
  to establish an edge.

A tail count at 60/64/66/68/69/69.5/70 is the next measurement;
`research/population_score.py` reports it.

### Where the ceiling comes from

The highest-scoring bar in a 2016-01-04 → 2016-02-08 slice, with
`available_points = 100.0` — all five components present, no degradation:

| component | points | weight | % of weight |
|---|---|---|---|
| options_flow | 9.321 | 20 | 46.6% |
| order_flow | 12.326 | 25 | 49.3% |
| structure | 12.908 | 20 | 64.5% |
| liquidity | 14.230 | 15 | **94.9%** |
| vol_momentum | 15.115 | 20 | 75.6% |
| **total** | **63.900** | **100** | **63.9%** |

Options flow and order flow together hold **45 of the 100 points and
contribute 21.6**. That is the ceiling's source.

### The consequence for Phase 8

A score-threshold sweep over 60..85 will produce trades only in cells BELOW the
committed 70.0, because no bar in a year reaches 70.0. Those cells are the
output of lowering a gate until something passes, which the brief forbids
presenting as a result. The grid is still worth running — it measures
sensitivity — but no cell in it is a discovery.

**This is a negative result and it is reported as one.** It says nothing about
markets; the bars are synthetic. What it says is that the Flow Score's
component magnitudes and its setup thresholds were specified independently, and
the thresholds sit at or just above what the magnitudes actually produce. That
is a design fact, discoverable with no real data at all — and it is the first
thing to re-measure once real bars exist, because the component magnitudes are
a function of the data and may differ.

---

## 5. Corrections to previously published figures

Both were agent-reported figures I re-ran and found wrong. Recorded so the
pattern is visible: **an agent-reported number that nobody re-ran is not
evidence.**

1. **Cost drag.** The Phase 7a build agent reported 0.0597R / 0.0269R — exactly
   half. Its dollar figures matched to the cent; it divided by $800 where R is
   $400. The figures in §2.5 are mine.
2. **The binding constraint on zero trades.** Reported as the Flow Score
   threshold; it is §14 structure for 98.6% of bars. See §2.4.

---

## 6. What would have to change for any row above to become evidence

- Real bars, per-bar provenance (`interpolated` false, `volume > 0`, OHLC not all
  identical), with a coverage and exclusion report.
- A separate instrument spec per instrument actually used. `defaults.yaml`
  configures futures only; QQQ is a $0.01-tick, $1/point RTH-only equity, and
  NQ-calibrated tick slippage does not transfer to it.
- Phase 2's gate re-passed on real data. It currently reads "real data loaded and
  graded" and was passed on synthetic.
- Every figure in §2 re-measured on real bars, with sample sizes and uncertainty.
- The R definition resolved, and §2.5/§2.6 restated in the resolved unit.

Until then the honest summary is: **the software is extensively tested; the
strategy is untested.**
