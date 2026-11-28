# Phase 1f results: hybrid dataset (version 3), partial-loss detector, backup configuration

Scope: exactly the approved Phase 1f items. No new data was fetched (0 network requests), no
Stage 2, no expansion, no Strategy A rerun, no parameter freeze, no 2025+ data, no protected
database changed. Files here hold counts, hashes and reports only; the replacement log with
prices stays beside the database (`data/stage_1_1_hybrid/replacements.json`, not in git).

## 1. Version 2 rebuilt from the stored responses (input to the hybrid)

`data/phase1f_v2_rebuild/universe_stage.db`, rebuilt 2026-11-28 01:18-03:00 UTC by offline replay
of the Phase 1b store (proxy at 127.0.0.1:9, replay mode).

| Check | Result |
| --- | --- |
| Replay | exit 0; 2,169 answers from the store, **0 fetched live, 0 request log lines** |
| Phase 1b store after replay | sha256 unchanged (108d2a17…) |
| Tables | **21 of 21 identical** to the Phase 1b recording (`v2_rebuild_tables.txt`) |
| Stage report | **byte-identical** to the Phase 1b recording |

## 2. Hybrid dataset (version 3)

Script: `scripts/build_hybrid.py`; output: `build_hybrid_output.txt`.
Database: `data/stage_1_1_hybrid/universe_stage.db`, sha256
`d08e272e66c5aa5b7e4f7473fe7b41b93bb671af56bda345fd81dd2146158528` (508 MB), built 2026-11-28
03:02-03:10 UTC. Now in `PROTECTED_DATABASES`. Its SEC answers stay in Stage 1.1, which is its
read-only parent (as for every seeded database).

| Step | Result |
| --- | --- |
| Seed from Stage 1.1 (opened read-only) | every table equal to the parent |
| **Control**: membership recomputed on the untouched seed vs Stage 1.1's stored rows | **identical, 61,124 of 61,124 rows**, all 9 columns |
| 2021-04-19 replaced from version 2 | **505 security_bars, 504 ticker_bars** rows; row counts equal before and after; **0 without a located raw source** |
| 2021-10-25, 2022-03-08 | kept as stored; flagged; bounded (section 4) |
| Decisions recomputed | 59 |
| **B2** (each decision recomputed from data cut off at that decision) | **59 of 59 identical** |
| **B3** (every price and volume after 2022-06-30 multiplied by 10) | earlier decisions **unchanged: True** |

**Replacement provenance** (`replacements_provenance.tsv`, sha256 36e3517e…; full old/new values
in `replacements.json`, sha256 040de2ee…): every row names the Phase 1b response key, its body
sha256 and its fetch time (235 distinct responses, fetched 2026-11-27 08:38:08-09:29:04 UTC).
Columns changed: high/low/open, volume, vwap, trade_count in 1,008 rows, and SPWR's
`alpaca_adjusted_close` (empty -> 344.75) in one. **No close price changed**, so the membership
input (close x volume) changed through volume only.

**Hybrid vs the other versions** (`hybrid_vs_v1_v2_rows.txt`, `hybrid_tables*.txt`):

| Against | Bars differing | Tables differing |
| --- | --- | --- |
| Version 1 (Stage 1.1) | 2021-04-19 only (505 / 504 rows) | security_bars, ticker_bars, membership |
| Version 2 | SPY 2019-08-12 only (1 / 1 row; a provider revision outside the approved scope, left as Stage 1.1 had it) | security_bars, ticker_bars, batch_symbols, candidates |

`batch_symbols` and `candidates` differ from version 2 only in the free text of 4 `reason` rows,
which quote the 100th member's median on 2021-04-30 (615,174,483 in Stage 1.1, 617,327,746 after
the repair). Which symbols were selected is the same; the text was left as built.

**Source timestamps: revision vs what was known at the time.**

| Data | Obtained | What it represents |
| --- | --- | --- |
| Stage 1.1 bars | fetched before 2026-11-26 08:45 UTC (built 2026-11-25) | Alpaca's history as served then |
| Repaired 2021-04-19 bars | Phase 1b store, fetched 2026-11-27 08:38-09:29 UTC | Alpaca revised its history between 2026-11-26 08:45 and 2026-11-27 00:10 UTC, more than 5 years after the trading day |
| Minute evidence for 10-25 / 03-08 | Phase 1d store, fetched 2026-11-27 15:56 UTC | still defective |

The repaired bars are a **later revision of the historical record**, not information a
participant would have had in Alpaca's feed at the time. They are closer to what actually
traded (the consolidated tape), which is what the membership rule is meant to measure. The
hybrid therefore describes "the market as it traded, by the best record available on
2026-11-27", not "what a 2021 researcher would have downloaded". A point-in-time replica of a
historical data feed is not possible from this data.

## 3. Membership: all 59 decisions

| Comparison | Decisions with any difference | Member changes |
| --- | --- | --- |
| Hybrid vs Stage 1.1 (`membership_v1_vs_hybrid.txt`) | 3 (2021-04-30: 320 rows, 2021-05-28: 269, 2021-06-30: 255; columns median_dollar_volume and rank only) | **0 in all 59** |
| Hybrid vs version 2 (`membership_v2_vs_hybrid.txt`) | **0 of 59** | 0 |

## 4. Membership uncertainty from 2021-10-25 and 2022-03-08

Script: `scripts/membership_uncertainty.py`. A flagged bar lost trades, so its true volume is at
least the stored one; a median can only rise, by at most one step. For each decision whose
63-session window holds one of the days, every eligible non-member with a flagged bar is tested:
can it rise above the 100th member? The recomputed medians are asserted equal to the stored ones.
6 decisions have a window that holds one of the days.

| Decision | 100th member | Result | Who could enter, and what it would take |
| --- | --- | --- | --- |
| 2021-10-29 | PTON 574.2M | certain | |
| 2021-11-30 | FDX 605.2M | **UNCERTAIN** | DASH (rank 101) if its 2021-10-25 dollar volume was above 605.2M (stored 18.2M; neighbouring sessions 177.8M, 186.9M) |
| 2021-12-31 | TMUS 628.2M | **UNCERTAIN** | SPGI (rank 101) above 628.2M (stored 60.3M; neighbours 670.2M, 1,155.7M) |
| 2022-03-31 | CVS 751.1M | certain | |
| 2022-04-29 | ATVI 687.3M | **UNCERTAIN** | AMC (rank 101) above 687.3M (stored 11.5M; neighbours 608.9M, 403.9M) |
| 2022-05-31 | ELV 651.0M | certain | |

- Flagged set: the 955 `systemic` alerts on the two days (`membership_uncertainty_hybrid.txt`).
- **Conservative variant** (every one of the 1,845 bars on the two days treated as possibly
  short; `membership_uncertainty_hybrid_all_bars.txt`): the **same 3 decisions**; it adds PANW as
  a second candidate on 2021-12-31 (stored 619.5M, a normal-looking bar).
- Read: SPGI entering on 2021-12-31 is likely (its neighbours are above the line); AMC is
  possible; DASH needs about 3x its neighbouring sessions, so it is unlikely but cannot be
  excluded. In each case at most one member would swap (the 100th member out, the 101st in).

Why the unflagged bars are believed normal (`defect_day_ratio_distribution.txt`): on both days
the liquid securities split cleanly into below 2% of normal (455 / 491) and above 50% (442 /
405), the same shape as the neighbouring days; only 4 and 8 bars fall in between.

## 5. Partial data-loss detector (proposed, not adopted)

Script: `scripts/audit_partial_days.py` (test: `tests/test_audit_partial_days.py`). Proposal with
exact numbers: `research/plans/stage2_thresholds.md`, section G3. Stored data only; nothing
repaired or excluded.

| Category asked for | How it is handled |
| --- | --- |
| Normal low-volume days | below the 10% line only after a spike: labelled `after_volume_spike` |
| Half trading days | tagged from the exchange calendar; 0 alerts at 10% (they begin to alert at 25%) |
| News-driven volume changes | not detectable (no news data); spikes are labelled, isolated drops go to review |
| Potential missing-data events | `systemic` (many securities on one session) or `isolated`, status `potential` |
| Confirmed data-quality problems | status `confirmed` only from a reviewed evidence list (`confirmed_defects.tsv`, 503 rows: 499 on 2021-04-19 confirmed by the provider's repair, AFL and SPGI 2021-10-25, AJG and AME 2022-03-08 confirmed by minute bars) |

Results (`partial_audit_stage_1_1.txt`, `partial_audit_hybrid.txt`, alerts in the `.tsv` files):

| Dataset | Alerts | systemic confirmed / potential | after_volume_spike | isolated | half-day |
| --- | --- | --- | --- | --- | --- |
| Stage 1.1 | 1,472 | 503 / 951 (04-19: 499, 10-25: 456, 03-08: 499) | 9 | 9 | 0 |
| Hybrid | 973 | 4 / 951 (10-25: 456, 03-08: 499) | 9 | 9 | 0 |

**False positives** (`threshold_sweep_stage_1_1.txt`):
- 2021-04-19, the only day with known truth: 499 alerts, **499 repaired by the provider, 0 false
  positives**. 5 repaired bars missed (HKIB, DFNS, FERG, SPAQ, KCAC), all below the 1,000-trade
  liquidity line.
- Elsewhere at 10%: 18 alerts in 1.83 million liquid security-sessions, 0 confirmed. K
  2018-03-26 (6 trades vs a median of 20,258) looks like a real single-stock defect, not checked
  against any other source. The other 17 (CODX, DGAZ x4, ALLG, HOLO x2, SINT, TOP, GMBL x6,
  BETS) are consistent with real quiet trading after a spike or during a wind-down.
- Sweep: 5% -> 1 elsewhere, **10% -> 18**, 15% -> 88, 20% -> 306, 25% -> 899 plus 6 half days
  becoming "systemic". All 1,454 defect-day alerts are caught from 10% up.

**Scale of the problem (new):** below 10% of normal trades on 2021-10-25: **456** securities;
2022-03-08: **499**. The near-empty-bar check (G1, 5 trades or fewer) saw only 221 and 47. G1
still fails on the hybrid (`hybrid_auction_bars.txt`: 268 bars, all on the two days), as it must
until a treatment is approved.

Limits: losses milder than about 90% are invisible; volume lost with a normal trade count is
invisible; illiquid securities are not examined; a loss on every security for many days would
move the baseline; an isolated alert needs a human to judge it.

## 6. Backup configuration (for your review)

`research/plans/backup_configuration.md`: Backblaze B2 and AWS S3 alternatives with bucket
settings, Object Lock (Governance, 365 days), a key that cannot delete, the four environment
variables, and verification steps V1-V5. Nothing created, no credential requested, nothing
uploaded. The hybrid would need its own export, and it reads SEC answers from Stage 1.1, so both
must be backed up together.

## 7. Checks

| Check | Result |
| --- | --- |
| Completeness audit on the hybrid (`hybrid_completeness.txt`) | **PASS** |
| Protected data (`protected_data_check.txt`) | **PASS**: sha256 of Stage 1.1, Stage 1.0, `signaliq.db`, `sip_research.db`, both response stores unchanged; IEX 15,861 bars, e66890e79788fc78; Strategy A 13 of 13; Stage 1.1 21 of 21 tables |
| Tests | 991 passed |
| Network | 0 requests in this phase |
