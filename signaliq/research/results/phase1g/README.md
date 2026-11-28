# Phase 1g results: cleanup and review after Phase 1f

Scope: exactly the five approved follow-up items. No data fetched (0 network requests to any data
provider; one web search of Backblaze's public CLI documentation). No Stage 2, no expansion, no
Strategy A rerun, no parameter freeze, no 2025+ data. Stage 1.1, the IEX database, the Strategy A
archive and every other original database are unchanged (section 6).

What the checks below establish: reproducibility, and no leakage of future data under the rules
tested. They do not show that every historical bar is accurate. Sections 2-4 are about the bars
that are known or suspected to be wrong.

## 1. Stale free-text notes in the hybrid: fixed

Script: `scripts/fix_reason_notes.py`. Output: `fix_reason_notes_output.txt`. Old and new text:
`reason_notes_fixed.json`.

- **What changed:** the hybrid only (version 3). Stage 1.1 was not opened.
- **Guard:** the script refuses unless the hybrid still has the sha256 reviewed in Phase 1f.
- **Rows:** 4 symbols (EBON, RLX, SI, TKAT), each with a note in `batch_symbols` and in
  `candidates`, so **8 rows**. Only the `reason` column changed. Every other column of both
  tables was checked equal to version 2 before writing.
- **Content:** the quoted 100th-member median on 2021-04-30 went from 615,174,483 (computed
  from the defective bars) to 617,327,746. The script asserted that 617,327,746 is the hybrid's
  own 100th member on that date. It is the text version 2 wrote.
- **Not changed:** the other 150 notes; they already agree with version 2.
- **sha256 before:** `d08e272e66c5aa5b7e4f7473fe7b41b93bb671af56bda345fd81dd2146158528`
  (reviewed in Phase 1f).
- **sha256 after:** `7357aa5f7fb9c4efab3ff71d00f7c6714029e6a5c73a232b5980a5c4e5d2b60c`.
- **No backup copy of the pre-fix file was kept** (508 MB). The pre-fix content can be
  reconstructed: it is the fixed file plus the 8 old texts in `reason_notes_fixed.json`.

To rebuild the hybrid from scratch: `phase1f/scripts/build_hybrid.py`, then this script.

**Recheck after the fix:**

| Check | Result |
| --- | --- |
| Table fingerprints, standard and logical (`hybrid_tables.txt`, `hybrid_tables_logical.txt`) | 21 tables, standard = logical. `batch_symbols` (a351b4dd…) and `candidates` (cdaf8c7c…) now **equal version 2's** |
| Against version 2 (`hybrid_vs_v2_tables.diff`) | only `security_bars` and `ticker_bars` differ: the SPY 2019-08-12 row, left out of scope on purpose |
| Membership vs version 2 (`membership_v2_vs_hybrid.txt`) | **0 of 59** decisions differ |
| Membership vs Stage 1.1 (`membership_v1_vs_hybrid.txt`) | 3 of 59 (ranks and medians only, as in Phase 1f); **0 member changes** |
| Completeness audit (`hybrid_completeness.txt`) | **PASS**; 158 of 158 added tickers' stated values recompute |

## 2. K (Kellogg) on 2018-03-26: investigated from stored data

Full report: `k_2018-03-26.txt`. **Verdict: likely a provider data defect. Not confirmed; the
status stays `potential`.**

**Evidence for a defect:**
- **Shape:** the same auction-only shape as the four confirmed defects. Open equals the low,
  close equals the high, and there are 6 trades.
- **Only one affected:** K is the only liquid security affected that session. The next-lowest
  of 753 is at 45% of normal, and 99% are above 55%.
- **Market and peers normal:** SPY, GIS, KHC, HSY, MDLZ, CAG, SJM, PEP and KO traded 66-89% of
  their previous-day counts.
- **K's own history:** the lowest of the 20 sessions before was 13,620 trades, and the next day
  had 16,704. The day's price range (0.54) is below its 20-session minimum (0.70).
- **No corporate action** for K near that date.

**Evidence for ordinary trading:** none found.

**Why it cannot be confirmed from stored data:**
- **Monthly bars:** Alpaca's monthly bars are exact sums of the same daily bars. That holds in
  all 88,998 ticker-months checked, including the confirmed defect months, so they are not
  independent evidence.
- **Version 2:** holds the same bar.
- **IEX database:** has no K.
- **Minute bars:** none stored for K.
- **What would confirm it:** one SIP 1-minute request (not made; it needs approval).

**Membership impact: none.** The first decision is 2020-01-31, and its window starts in
October 2019.

## 3. Uncertainty reporting for the three affected decisions

Label file: `uncertain_decisions.tsv` (one row per possible entrant). The stored membership of
all 59 decisions is **kept unchanged** in every dataset. No replacement member was chosen.

| Decision | Stored 100th member (kept) | Possible entrant | Needed on the unrepaired day | Stored | Status |
| --- | --- | --- | --- | --- | --- |
| 2021-11-30 | FDX | DASH | more than 605.2M (2021-10-25) | 18.2M | **UNCERTAIN** |
| 2021-12-31 | TMUS | SPGI (or PANW, conservative variant only) | more than 628.2M (2021-10-25) | 60.3M (PANW 619.5M) | **UNCERTAIN** |
| 2022-04-29 | ATVI | AMC | more than 687.3M (2022-03-08) | 11.5M | **UNCERTAIN** |

**At most one member per decision can change.** Each possible entrant's highest possible
median stays below the 99th member (UNP 641.2M on 2021-12-31; HON 699.1M on 2022-04-29), so at
most one entrant can take the 100th place. The one exception, DASH (up to 608.0M against AAL's
607.3M on 2021-11-30), is the only entrant that decision has. Members' medians can only rise.

**Reporting rule (proposed with G-data):** any result that uses one of these three decisions
names it as uncertain and points to this file. The other 56 decisions are not affected by the
two unrepaired days.

## 4. G1 and G3 reviewed together

Proposal: `research/plans/stage2_thresholds.md`, section "G1 + G3 together". It merges them into
one gate, G-data, with two tiers. **Not adopted.**

- **G1 sits inside G3:** all 767 G1 bars on Stage 1.1 are G3 alerts, as the definitions imply
  (5 trades or fewer against a median of at least 1,000 is below 0.5%). G3 adds 705 bars: 687
  on the defect days (6-2,960 trades) and 18 elsewhere.
- **Statuses:**
  - `potential` is the default, and nothing is promoted from it automatically.
  - `confirmed` needs a reviewed row for that one bar that names recorded evidence.
  - `reviewed_normal` needs a reviewed row with a reason.
- **One code change enforces this:** `scripts/audit_partial_days.py` no longer accepts a
  session-wide `*` confirmation. Nothing used it. The hybrid's alerts are unchanged (973; 4
  confirmed), see `partial_audit_hybrid.txt`.
- **Today:** the gate fails on both datasets until review rows exist (Stage 1.1: 503
  confirmed, 969 potential; hybrid: 4 confirmed, 969 potential).

## 5. Backup setup instructions

`research/plans/backup_configuration.md` now opens with ten ordered steps. Steps 1-6 are yours:
bucket, credential, permission check, variables. Steps 7-10 follow: probe, one-dataset upload
with fresh-download F4, record, then your second copy. It also states the dependencies: the
hybrid must be backed up with Stage 1.1, and version 2 with the Phase 1b store and Stage 1.1.
The `b2 key create` syntax matches the B2 CLI v4 documentation. Nothing created, no credential
requested, nothing uploaded.

## 6. Checks

| Check | Result |
| --- | --- |
| Protected data (`protected_data_check.txt`) | **PASS**. sha256 unchanged: Stage 1.1, Stage 1.0, `signaliq.db`, `sip_research.db`, the Phase 1b and Phase 1d stores. Hybrid at its new reviewed sha256 7357aa5f…. IEX 15,861 bars, e66890e79788fc78. Strategy A 13 of 13. Stage 1.1 21 of 21 tables. Protected archive folders: 0 commits, 0 changes |
| Tests | 991 passed |
| Lint | new and changed files clean. Untouched: `tests/test_validation.py` and the archived Phase 1b scripts |

## 7. Remaining risks

- **Three decisions stay uncertain.** Resolving them needs data that isn't stored.
- **K 2018-03-26 is likely a defect but unconfirmed.** It has no effect on membership.
- **The completeness proof relies on monthly bars, which carry the same daily defects.** On
  the decisions whose windows hold 2021-04-19 (in Stage 1.1's monthly data), 2021-10-25 or
  2022-03-08, a ticker without daily data could have its upper bound understated by about one
  day's volume. Across all decisions, the closest any such ticker comes is 0.1% short (RL,
  2024-05-31, outside those windows). That margin has not been measured separately for the
  affected decisions.
- **The hybrid's pre-fix file was not kept.** It can be reconstructed from the log (section 1).
- **No dataset is backed up off this container.**
- **The data-quality gate cannot see partial losses milder than about 90%.**
