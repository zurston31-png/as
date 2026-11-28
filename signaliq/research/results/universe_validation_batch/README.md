# SIP universe validation batch, report v5 (research data only)

**No strategy was run, nothing was backtested, no parameter was frozen, and no market
data or evidence dated 2025 or later was used.** The 500-stock universe has not been
started; it waits for explicit approval.

| File | What it is |
| --- | --- |
| [`validation_report.txt`](validation_report.txt) | v5, the current report (876 lines) |
| [`reproducibility.txt`](reproducibility.txt) | SHA-256 of three runs and their logs; byte comparison |
| [`merge_audit.txt`](merge_audit.txt) | Independent audit of the 21 merges and 11 rejected candidates ([`scripts/audit_merges.py`](../../../scripts/audit_merges.py)) |
| [`outcome_audit.txt`](outcome_audit.txt) | Confidence label, evidence and value basis of every ended security ([`scripts/audit_outcomes.py`](../../../scripts/audit_outcomes.py)) |
| [`curated_quotes_live_check.txt`](curated_quotes_live_check.txt) | Every curated SEC quote checked against the live SEC document |
| [`deal_values_recheck.txt`](deal_values_recheck.txt) | Every deal value recomputed from freshly fetched acquirer prices |
| `validation_report_v4.txt` ... `_v1` | Earlier reports |

## What v5 changes

### Trading-gap segments merged

A pause in trading no longer turns one security into several. Two segments of the same
ticker are one security only if all of these hold:

- the gap is at most 250 sessions;
- no delisting notice or exit (merger, bankruptcy 8-K, quoted fund liquidation) came
  before trading resumed;
- the price resumes within 0.5x-2x of where it stopped, after splits recorded in the gap,
  or SEC filings quoted in a curated fact say the same shares resumed;
- the issuers do not conflict when both are known.

Every candidate is recorded with its decision and reason (table `segment_links`). Ticker
bars and segments are not changed. The run repeats until no new merge appears (2 passes).

| | Count |
| --- | --- |
| Securities before / after | 122 / 101 |
| Merge candidates | 32 |
| Merged | 21 |
| Kept separate | 11 |

### Issuers

- **The SEC ticker list is checked for every ticker** a security used, newest first.
- **Full-text search is a fallback.** It needs the company's name within 60 characters
  before the ticker and, where Wikidata names the company for the period, a current or
  former name that agrees.
  - That rule removed two false identifications made during this work: a mortgage trust
    for First Republic, and Groupon for the 2021 SHLD ETF.

## Results

### Duplicate-security audit: passes

| Check | Result |
| --- | --- |
| Ticker-days held by more than one security | 0 |
| Same-ticker successions between two securities | 11, each with a recorded decision |
| Same issuer and same type, one after the other | 4 (CHK, CRC, DO, HTZ); each is explained (below) |
| Possible duplicates without a recorded reason | 0 |

**The 4 same-issuer pairs** are post-bankruptcy re-listings under the same SEC
registrant. The old shares were cancelled and new shares issued, so they are different
securities. Each has a delisting notice and a bankruptcy 8-K or price break before the new
listing.

### Merge-by-merge evidence: 21 of 21 pass, 11 of 11 rejections supported

The independent audit does not use the pipeline's decision code. For each merge it
rebuilds from the raw ticker bars, raw corporate actions and the issuer's SEC delisting
list:

- every bar inside the gap is a filler bar (no volume, no trades);
- the gap recounts exactly;
- the split-adjusted price ratio;
- no delisting notice from the issuer in the gap;
- one security spans both segments.

| Merged | Gaps | Price ratios | Delisting list read |
| --- | --- | --- | --- |
| NOVVW (Nova Vision warrants) ×10 | 11-31 sessions | 0.70-1.92 | yes, 0 notices |
| CETUR (Cetus rights) ×4 | 11-32 | 0.87-1.05 | yes, 0 |
| BACA (Berenson SPAC) | 11 | 1.00 | yes, 0 |
| MDGS (Medigus, later Xylo) | 11 | 0.80 | yes, 0 |
| SMMC (South Mountain, later BTRS) | 14 | 1.00 | yes, 0 |
| SCPX (Scorpius) | 32 | 0.34 after the 1-for-200 split | yes, 0 |
| HSACW (HSAC warrants, later IMVTW) | 34 | 0.89 | no SEC issuer known |
| SHLD (VictoryShares ETF) | 11 | 1.01 | no SEC issuer known |
| WDRW (Direxion ETF) | 17 | 1.17 | no SEC issuer known |

- **SCPX** is outside the price range and was merged on SEC evidence. The audit checked
  that evidence separately: both SEC passages (1-for-200 reverse split; NYSE "has
  withdrawn its delisting determination and will be lifting the trading suspension") are
  in the documents, and Alpaca records the SCPX → SCPXD rename and the split on 2024-07-18.
- **For the 3 merges without an identified SEC issuer,** the delisting-notice check could
  not be made; the other checks pass.

**Rejections:** at least one disqualifying fact was re-established from raw data for each.
Genuine ticker reuse stays separate.

| Ticker | What is kept apart | Independent facts |
| --- | --- | --- |
| CHK | Old Chesapeake vs post-bankruptcy | Notice 2020-07-20; price ratio 3.80 |
| CRC | Old California Resources vs post-bankruptcy | Notice 2020-07-31; ratio 12.7 |
| DO | Old Diamond Offshore vs post-bankruptcy | Notice 2020-05-12; 486 sessions; ratio 8.0 |
| HTZ | Old Hertz vs post-bankruptcy | Notice 2020-10-30; 258 sessions; ratio 14.7 |
| FSBC | FSB Bancorp vs Five Star Bancorp | Notice; merger records; different issuers; 254 sessions |
| MON | Monsanto vs Monument Circle | Notice; different issuers; 697 sessions |
| RATE | Bankrate vs Global X ETF | Notice; different issuers; 1,261 sessions |
| SHLD | Sears vs VictoryShares ETF | Notice 2018-11-15; 513 sessions; ratio 70 |
| SHLD | VictoryShares ETF vs Global X ETF | Quoted liquidation; 484 sessions |
| OILD | AccuShares fund vs ProShares fund | Quoted liquidation before restart |
| OILD | ProShares fund vs a later OILD | Quoted liquidation; 408 sessions |

### Look-ahead audit: passes

| Check | Result |
| --- | --- |
| Recompute every decision with data cut at that date | 59 of 59 identical |
| Multiply all prices after mid-2022 by 10 | No earlier decision changes |
| Members chosen after their last trade | 0 |
| Latest date in bars, exit evidence, SIC filings, deal dates, acquirer prices | All 2024-12-31 or earlier |
| Blank-check exclusions re-derived from facts on file by the decision date | 0 of 245 differ |

- **What membership reads:** raw bars up to the decision, the security type, split
  checks, and the SIC filed by then.
- **What it never reads:** exit outcomes, exit values, or later prices.
- **Merges** use the fact that trading resumed only to decide identity. A merged
  security's bars up to any date are unchanged, and the recompute tests hold.

### Exit-value integrity: passes

| Check | Result |
| --- | --- |
| Ended securities with a value | 55, each traced to its evidence (report section 10) |
| Stated cash (merger record, 8-K, redemption notice) | 23 |
| Verified deal terms | 11; all 12 deals recompute exactly from fresh acquirer prices |
| Assumed zero | 21: event confirmed, recovery **not** confirmed, labelled as an assumption |
| Problems found by `audit_outcomes.py` | 0 |
| Alpaca and SEC 8-K cash amounts that agree | 8 of 8 |

Of the 12 deals, 11 are used. **FSBC is held out**: +10.3% from the last close, and its
election results were not found.

The 34 values presented as payouts are stated cash or verified deal terms. Two of them
carry caveats:

- **CELG** is a lower bound: its contingent value right is not valued.
- **VMW** is the cash a holder received who did not elect.

**No value is assumed for 11:**

| Security | Status |
| --- | --- |
| FSBC | Election results not found |
| YGF, WDRW 2020 | Unresolved |
| OILD ×2, SHLD 2021 (ETFs) | Liquidated at net asset value; amount not stated |
| MON 2022 (SPAC) | Trust redemption; amount not stated |
| MONDW | Warrants redeemed; price not stated |
| NOVVW, CETUR (SPAC warrants, rights) | No terms |
| GMZ (fund) | Merged into another fund |

### Confidence of every outcome

Every one of the 66 ended securities carries a confidence label and its evidence.

| Label | Count |
| --- | --- |
| Two independent sources | 37 |
| **Lower confidence: one source** | 27 |
| Unresolved | 2 |

- **The 27 one-source outcomes stay labelled as lower-confidence evidence:** BBBY, BSJN,
  CELG, CRC, DISCA, DO 2020, ENDP, EXPR, FRC, FTR, MON 2018, PRTY, RATE, REV, SBNY, SHLD
  2018, TWX, OILD ×2, JMPNL, IMVTW, SHLD 2021, MON 2022, MONDW, NOVVW, BMAC and PYPT.
- **Passing the audits does not make them as certain as two-source outcomes.**
- **Members that later ended:** 24 of the 44 distinct members. All 24 have a value, 19 by
  two sources and 5 by one.

### Issuer identification: all eight resolved or explained

| v4 case | v5 result |
| --- | --- |
| FI → XPRO | Expro Group (CIK 1575828), full-text search |
| FISV → FI | Fiserv (CIK 798354), SEC ticker list |
| HTBX → NHWK → SCPX, and the second SCPX | Now one security; Scorpius Holdings (CIK 1476963), full-text search |
| SMMC | Merged into SMMC → BTRS; BTRS Holdings (CIK 1774155), delisting notice |
| HYW → STEC | Santech Holdings, formerly Hywin (CIK 1785680), full-text search |
| FRC | No SEC issuer: a bank registered with the FDIC, not the SEC; not a blank check |
| SBNY | No SEC issuer: a bank registered with the FDIC, not the SEC; not a blank check |

- **Overall:** 87 of 101 securities have an SEC issuer, and 76 of the 78 common stocks.
- **Securities without one:** First Republic and Signature (documented above), plus ETFs
  and one warrant chain. Those are outside the common-stock universe whatever their issuer.
- **Common stocks without an issuer or a documented reason:** 0.
- **The FRC and SBNY reasons rest on reasoning,** not on a quoted SEC passage.

### Reproducibility: passes

| Check | Result |
| --- | --- |
| Rebuild run and two cache-only runs | Byte-identical reports, SHA-256 `f7904745513b89fd9057f1416c5036f07a1158ae9e4aed4dc11b25381953e27e` |
| Run logs | Byte-identical, `59df47796eea970bc3905142394164993e356521ad2287015400227ebd522dea` |
| Network requests in the cache-only runs | 0 (SEC, OpenFIGI, Wikidata) |
| Curated SEC quotes | 42 of 42 found in 28 live SEC documents |
| IEX market data | Unchanged: 15,861 bars, fingerprint `e66890e79788fc78`, file last modified 2026-11-17 |
| Strategy A archive | All 13 checksums OK; no change since its archive commit |
| Tests | 853 passed |

## Remaining unresolved cases and limitations

1. **Unresolved outcomes (2):**
   - **YGF:** delisted by Nasdaq under rules 5250(a)(1) and 5250(c)(1) (reports not
     filed; quoted). What shareholders got is unknown, and no failure is assumed.
   - **WDRW 2020** (a Direxion ETF): why it stopped trading was not found.
2. **FSB Bancorp's payout is not used.** The election average is 10.3% from the last
   close, and the election results were not found.
3. **27 outcomes rest on one source,** mostly SEC alone.
4. **21 zero values are assumptions.** The bankruptcy or failure is confirmed, but the
   recovery is not, and old Hertz holders were in fact paid out. A sensitivity test is
   still needed.
5. **The trading-gap rule is a heuristic** (gap length, price continuity, notices).
   - Three merges had no SEC issuer to check for notices.
   - The rule cannot tell apart two different instruments that trade at the same price
     after a pause, unless a delisting notice or exit record separates them. A fund
     launching at $25 under a reused ticker (OILD 2016) was caught only by its quoted
     liquidation.
6. **Merging uses later data for identity.** It does not change any bar up to any date,
   and the recompute tests hold.
7. **Security types come from present-day sources** and apply to the whole history.
8. **SIC codes** from SEC filing headers can lag or lead a SPAC merger. The ticker-change
   rule handles the lag, but an early SIC change is followed as filed.
9. **The FRC and SBNY issuer reasons** are documented by reasoning, not quoted SEC text.
10. **Scale:** 101 securities. The 500-stock universe will bring new case types, and
    curated entries need manual research.
11. **Not modelled:** no OTC prices after an exchange delisting; dividends are excluded.
12. **Disclosure:** the IEX database holds bars to November 2026 from Milestone 1. It is
    unchanged and never read by this research.

## Reproduce

```bash
python scripts/universe_validation_batch.py --report validation_report.txt
```

```bash
python scripts/universe_validation_batch.py --report check.txt
```

```bash
cmp validation_report.txt check.txt
```

```bash
python scripts/audit_merges.py
```

```bash
python scripts/audit_outcomes.py
```

```bash
python scripts/verify_curated_quotes.py
```
