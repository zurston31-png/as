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

## 4. The R definition — unresolved, and it scales §2.5 and §2.6

Four sources, **three distinct definitions** of `TradeRecord.risk_dollars`, the
denominator of every R above:

| Source | Stop distance from | Size |
|---|---|---|
| `ARCHITECTURE.md` §0.2 (binding) | `entry_price` (realized) | `size` (requested) |
| `tests/unit/test_contracts.py` | planned | requested |
| `tests/unit/test_execution.py` | planned | requested |
| `flow_model/backtest/ledger.py` (sole producer) | realized | **filled** |

`TradeRecord` validates `gross_pnl` against the price path and
`pnl == gross − costs`, but **does not validate `risk_dollars` at all**, which is
why three definitions coexisted undetected. A record claiming `risk_dollars=40`
on a 10-point, 2-lot, $20 trade (true risk $400) validates today and reports
`r_multiple = 10.0` for a 1R trade.

Consequence for this document: **every R figure in §2.5 and §2.6 is denominated
in a unit that is currently ambiguous.** On a 2-point entry slip the realized
denominator gives exactly −1.0000R on every stop; the planned denominator gives
−1.1000R, and the difference *is* the entry slippage. Resolution is in flight.

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
