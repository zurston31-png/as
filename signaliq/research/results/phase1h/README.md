# Phase 1h results: completeness margins, uncertainty labels in reports, K request, backup status

Scope: the Phase 1h items that need no new data and no credentials. **Nothing was fetched.** The K
request waits for your explicit approval. The backup waits for your setup steps. No Stage 2, no
expansion, no Strategy A rerun, no parameter freeze, no 2025+ data. No database was written.

## 1. Completeness-proof margins at all 59 decisions

Script: `scripts/completeness_margins.py`. Output: `completeness_margins_hybrid.txt`,
`completeness_margins_stage_1_1.txt` (same monthly bars; the only difference is the 100th member
on 2021-04-30). Self-check: the overall closest case is RL on 2024-05-31 at 1.001, the same as
the completeness audit finds.

**The concern.** The proof keeps 17,492 tickers without daily data out of the universe because
an upper bound on each one's median stays below the 100th member. The bound is built from
Alpaca's monthly bars, and those are sums of the same daily bars, so a month holding 2021-04-19,
2021-10-25 or 2022-03-08 can understate a ticker by up to one day's volume. 12 decisions use such
a month in the proof's 4-month span.

**Measured with the proof's own bound** (margin in days of the ticker's average volume):

| Decisions whose span holds | Fewest days of margin |
| --- | --- |
| 2021-10-25 (4 decisions) | 8.0 (MQ, 2021-12-31) |
| 2022-03-08 (4 decisions) | 7.0 (DRIP, 2022-06-30) |
| 2021-04-19 (4 decisions) | **0.4 (TRIP, 2021-05-28)**; 1.7 (BLDP, 2021-04-30); 2.9 (TRIP, 2021-06-30) |

**How big is a missing day?** (`missing_day_size_2021-04-19.txt`) On the 503 bars that Alpaca
later repaired for 2021-04-19, the lost volume was a median of 0.64 average days (90th
percentile 1.00, maximum 2.74, BTI), and 94% lost more than 0.4 days. With the proof's own bound,
2021-05-28 is therefore **not certified**: if TRIP's 2021-04-19 bar was broken (499 of 893 liquid
stocks' were), its bound probably crosses.

**A tighter bound that is still strict closes it.** The proof sums 4 calendar months, but each
63-session window lies inside fewer months (for 2021-05-28: March-May only, not February).
Summing only the months the window touches, and adding **3 average days** of the month for
every month that holds a defective day (above the 2.74-day maximum measured):

- **0 tickers reach the 100th member at any of the 59 decisions**, on both datasets.
- TRIP on 2021-05-28 falls from 0.995 to 0.828 of the cutoff; it would need a missing day of
  more than 20 average days.
- Closest tickers per affected decision: `completeness_affected_detail.txt`.
  - The thinnest are a fund and a warrant: DRIP on 2022-06-30 (crosses only above 3.1 days)
    and DWACW in October-December 2021 (above 4.5 days). The membership rule admits only common
    stock, so neither could become a member. The proof counts them anyway, because it has no
    type information for tickers without daily data.
  - Every other ticker needs a missing day of **more than 8.0 average days** (IMPP, TIRX, DATS).

**What this rests on.** The largest loss measured was 2.74 average days, but on 2021-04-19,
an ordinary day. 2022-03-08 was the oil-price peak; a security with exceptional volume that day
could exceed it. DRIP, an inverse oil fund that cannot be a member, is the only security near
that line. The decisions without a defect month keep the proof's original margins. Several
are thin (below 1 day: VRSN 2020-01-31, TRIP 2021-05-28, CLVT 2021-09-30, CFLT 2022-02-28,
WAT 2023-08-31, RL 2024-05-31), but no known defect day lies in their spans. A defect that the
data-quality gate cannot see would still matter there.

## 2. K: the one-minute request is prepared, not sent

`requests_awaiting_approval.md`: one SIP 1-minute request for K and GIS (control) on 2018-03-26,
regular session only, about 780 bars. It runs into a new, empty, attempt-logged store, writes to
no research database, checks that only the approved URL was sent, and then protects the store.
It also states how each possible answer would be read. **It needs your explicit approval.**

## 3. The three uncertain decisions in downstream reports

- **What it prints:** the stage report (`scripts/universe_validation_batch.py`, section 6)
  prints each uncertain decision from `research/results/phase1g/uncertain_decisions.tsv`: the
  decision, the 100th member kept, the possible entrant, and what it would take. The stored
  membership is kept and no replacement is chosen.
- **Stale labels are refused.** A label is shown only while the stored 100th-member median at
  that decision equals the value in the file. Otherwise the report prints "LABEL DOES NOT MATCH
  THIS RUN ... recompute it". A missing file is reported as "uncertainty unknown".
- **Tests:** `tests/test_uncertainty_lines.py` (3 tests).
- **Rendered output:** `report_uncertainty_lines_hybrid.txt`, from the real Stage 1.1, version 2
  and hybrid databases; all 3 decisions labelled in each.
- **Archived reports:** future stage reports have these lines and the archived Stage 1.1 and
  Phase 1b reports do not. A byte comparison against an archived report now differs by exactly
  this block.
- **Strategy A** does not read the universe (it uses the IEX database), so no other report
  shows membership today.

## 4. Backup: blocked on your setup steps

The four `SIGNALIQ_ARCHIVE_*` variables are **not set** (names checked, no values read). Next:
steps 1-6 of `research/plans/backup_configuration.md` (bucket with Object Lock, Governance 365
days, a key without delete permission, the four variables in a personal environment). Then I
run the safety probe, upload one dataset at a time, download it fresh and run F4. **Until that
passes, no dataset is backed up off this container.**

## 5. Checks

| Check | Result |
| --- | --- |
| Protected data (`protected_data_check.txt`) | **PASS**: Stage 1.1, Stage 1.0, `signaliq.db`, `sip_research.db`, both response stores and the hybrid (7357aa5f…) unchanged; IEX 15,861 bars, e66890e79788fc78; Strategy A 13 of 13; Stage 1.1 21 of 21 tables |
| Tests | **994 passed** (991 + 3 new) |
| Lint | clean, except the pre-existing findings in `tests/test_validation.py` |
| Network | 0 requests |

## 6. Remaining risks

- **Completeness:** certified under a stated premise: no defective day lost more than 3 average
  days of a security's volume. That premise was measured on 2021-04-19 only.
- **Three uncertain decisions:** unresolved; they are labelled.
- **K:** unconfirmed until the request is approved.
- **No external backup.**
- **Data-quality gate:** cannot see partial losses milder than about 90%.
