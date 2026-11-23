# SIP universe validation batch, report v4 (research data only)

**No strategy was run, nothing was backtested, no parameter was frozen, and no market
data or evidence dated 2025 or later was used.** The 500-stock universe has not been
started and waits for review.

| File | What it is |
| --- | --- |
| [`validation_report.txt`](validation_report.txt) | v4, the current report (918 lines) |
| [`curated_quotes_live_check.txt`](curated_quotes_live_check.txt) | Every curated SEC quote checked against the live SEC document |
| [`deal_values_recheck.txt`](deal_values_recheck.txt) | Every deal value recomputed outside the pipeline from freshly fetched acquirer prices |
| `validation_report_v3.txt`, `_v2`, `_v1` | Earlier reports, kept for comparison |

## The six checks

| # | Check | Result |
| --- | --- | --- |
| 1 | Classified securities | Types: 121 of 122 classified (67 by two sources, 54 by one), 1 unknown. Ended securities: 85 of 87 explained (37 by two sources, 29 by one, 19 "not an exit"), 2 unresolved, 0 conflicts. The 40 named companies with a documented outcome all match it |
| 2 | Unresolved | 2 ended securities (YGF, WDRW 2020); 1 unknown type (MDGS); 1 acquisition value held out (FSBC). Details below |
| 3 | Acquisition payouts against deal terms | 12 deals with quoted terms: all 12 recompute exactly; 11 within 10% of the target's last close and used, 1 held out (FSBC, +10.3%). 8 cash deals stated by both Alpaca and the SEC 8-K: 8 of 8 agree |
| 4 | Survivorship audit | Passes; limitations below |
| 5 | Look-ahead audit | Passes after one fix found by the audit itself (below) |
| 6 | Reproducibility | Two cache-only runs: byte-identical (SHA-256 `8126a83c4496da25…`), 0 new SEC requests |

Also verified:

- **IEX market data:** unchanged (15,861 bars, fingerprint `e66890e79788fc78`; file last modified 2026-11-17).
- **Strategy A archive:** every checksum OK.
- **Tests:** 843 passed.

## What changed since v3

### SEC sources

Three SEC sources were added, all read-only, under the same research contact address:

- **Full-text search** (efts.sec.gov) finds which company wrote a ticker as its own, for
  example "FSB Bancorp, Inc. (NASDAQ: FSBC)".
  - The company's own name must come right before the ticker, and "formerly" disqualifies
    the mention.
  - A delisting notice at the right time is still what confirms a match.
  - Effect: 59 of 90 ended securities matched in the first pass (54 in v3), including Red
    Hat, Bankrate, Mellanox, FSB Bancorp and JMP Group's notes.
- **Filing history** (data.sec.gov).
- **Filing headers.**

### Facts read from SEC filings

28 entries hold 40 exact quotes from 27 SEC documents (3 renames, 11 facts, 12 deals, plus
2 more facts added during review). Code: [`app/universe/curated.py`](../../../app/universe/curated.py).

- **Re-verified on every run:** each quote must appear word for word in its document.
  - All 40 are found in the cached copies.
  - All 40 were found again in the live SEC documents
    ([`scripts/verify_curated_quotes.py`](../../../scripts/verify_curated_quotes.py)).
- **Renames Alpaca lacks** (these now link instead of ending):
  - EnerJex → AgEagle (ENRJ → UAVS), with Alpaca's 1-for-25 reverse split; +23.5% across
    the change, split-adjusted.
  - HSAC → Immunovant warrants (HSACW → IMVTW).
  - ITHAX → Mondee warrants (ITHXW → MONDW).
- **Exits:**
  - Two commodity ETFs liquidated (OILD 2016, OILD 2020).
  - The 2021 SHLD ticker was the VictoryShares Protect America ETF, liquidated, not Sears.
  - Immunovant warrants called at $0.01.
  - JMP notes redeemed at $25.
  - Scorpius: trading suspended, 1-for-200 reverse split, then resumed (not an exit).
  - First Republic.

### First Republic

- **The bank failure is confirmed by two unrelated SEC filers:**
  - JPMorgan's 8-K: it bought the assets and deposits "from the Federal Deposit Insurance
    Corporation (FDIC)".
  - Paramount Group's 8-K: the "Federal Deposit Insurance Corporation being appointed as
    receiver".
- **The shareholder recovery is not confirmed.** Zero is kept as a conservative modelling
  assumption, labelled `assumed_zero`.

### Value basis

Every exit value now states its basis, kept apart from how well the event itself is confirmed:

| Basis | Ended securities | Meaning |
| --- | --- | --- |
| `stated_cash` | 23 | A cash amount stated in a merger record, 8-K or redemption notice |
| `deal_terms` | 11 | Terms quoted from the filing × the acquirer's raw close on the effective date |
| `assumed_zero` | 21 | Bankruptcy or failure confirmed; the zero recovery is an assumption |
| `none` | 32 | No value: 19 are not exits (trading gaps); the 13 others are listed below |

### Stock and cash-and-stock deals

| Deal | Value | Target last close |
| --- | --- | --- |
| ALXN | $60 + 2.1243 AZN = 182.72 | 182.50 |
| CELG | $50 + 1 BMY = 106.41, excluding the CVR (lower bound) | 108.24 |
| DISCA | 1 WBD = 24.78 | 24.43 |
| MXIM | 0.63 ADI = 105.44 | 103.14 |
| PXD | 2.3234 XOM = 269.51 | 269.62 |
| TWX | $53.75 + 1.437 T = 100.48 | 98.77 |
| VMW | $142.50 cash for a market holder (see below) | 142.48 |
| WLL | $6.25 + 0.5774 OAS = 69.36 | 68.03 |
| XLNX | 1.7234 AMD = 196.93 | 194.92 |
| DO 2024 | $5.65 + 0.2316 NE = 13.80 | 13.99 |
| FLAC | 1 NAMS = 10.35 | 9.87 |
| **FSBC** | **held out**: $8.90 + 0.2197 EVBN = 14.74, +10.3% from 13.36 | 13.36 |

- **VMware** was an election deal: holders chose $142.50 cash or 0.252 AVGO share.
  - The first value used the average across all holders, $193.72. The 10% check rejected it.
  - Broadcom's election-results 8-K settles it: holders who elected cash or did not elect
    "will be entitled to receive $142.50 in cash per share".
  - A holder who bought in the market after the election deadline is in that group, which
    the last close of $142.48 confirms.
- **FSB Bancorp** was also an election deal, and its election results were not found, so
  its value is not used.

### Blank-check companies (SPACs) are excluded point in time

- **SIC 6770 is read from filing headers.** For a filing that names several companies,
  only the block of the issuer in question is read.
- **The period ends at the de-SPAC ticker change.** SEC's code can lag a merger by months:
  WeWork kept 6770 until August 2022.
- **Only filings and ticker changes on or before each decision date are used.**
- **Result:** 275 decisions excluded. 82 securities have an SIC history, and 73 of the 81
  common stocks have an identified issuer.

### Trading gaps

The same instrument sometimes trades again after a pause. Three conditions identify this:

- no delisting notice in between;
- a similar price, after any split recorded during the gap;
- the pause is under 250 sessions.

Such a case is labelled "not an exit". 19 cases, mostly thinly traded SPAC warrants and
rights.

## Survivorship audit

| Check | Result |
| --- | --- |
| Batch source | Historical corporate-action events (46 random), named cases (54), renames discovered (37): not today's ticker list |
| Ended securities in the batch | 87 of 122 (57 common stocks) |
| Members that later ended | 24 of 44 distinct members |
| ... with an exit value | 24 of 24: 7 deal terms, 10 stated cash, 7 assumed zero |
| ... how confirmed | 19 by two sources, 5 by one |
| Ended securities kept out only for unknown type | 1 (MDGS, a one-session fragment), down from 11 in v3 |
| Failed companies represented | 21 bankruptcies or failures, all common stocks, all valued at assumed zero |

Limitations:

- **Recoveries:** zero recovery is an assumption, and old Hertz shareholders were in fact
  paid out. The planned sensitivity test must cover this.
- **No OTC prices:** prices after an exchange delisting (BBBY, EXPR on OTC) are not in the data.
- **Dividends** are not included.

## Look-ahead audit

| Check | Result |
| --- | --- |
| Recompute each decision with data cut at that date | 59 of 59 identical |
| Multiply all prices after mid-2022 by 10 | No earlier decision changes |
| Members chosen after their last trade | 0 |
| Latest date in bars, exit evidence, SIC filings, deal dates, acquirer prices | All 2024-12-31 or earlier |
| Blank-check exclusions re-derived from facts on file by the decision date | 0 of 275 differ |
| What membership reads | Raw bars up to the decision, the security type, split checks, and the SIC filed by then. Never exit outcomes, values or later prices |

**The audit found a leak, which was fixed.** The first version of the blank-check rule
kept a SPAC excluded after its SIC changed if the security would later change ticker,
which uses knowledge of a future rename. The audit flagged 4 of 279 exclusions. Only
renames on or before the decision date count now, and a test asserts that answers do not
depend on later renames.

**Remaining limitation:** security types come from present-day sources and apply to the
whole history. A type is not a price or an outcome, but a later reclassification would be
applied retroactively.

## Errors found and fixed during this review

1. **Full-text search false match:** "Sears Holdings Corp. (formerly NASDAQ: SHLD)" in a
   SPAC director's biography. Fixed with the own-name and "formerly" rule.
2. **JMP notes typed as common stock** from a later notice for the common shares. Fixed:
   the notice nearest the last trade is used, and events after a class's redemption no
   longer apply to it.
3. **Election deals valued at the all-holder average.** The 10% check caught it; VMware
   was fixed from the election results, and FSB Bancorp is held out.
4. **SIC read for the wrong company** in filings that name several companies. Fixed.
5. **Blank-check period:** the SIC lag kept WeWork out for 10 months. Fixed with the
   ticker-change rule, then its look-ahead removed (above).
6. **Scorpius counted as unresolved.** It was a suspension followed by a 1-for-200 reverse
   split. Fixed from SEC filings; the trading-gap check now applies splits made during the gap.
7. **Alpaca-versus-SEC cash cross-check** read the par value instead of the price. Fixed
   (8 of 8 agree).
8. **SIC history crashed on headers without a SIC** and lost fetched data. Fixed, and
   fetches are now committed per security.

## Remaining unresolved items and gaps

**Ended securities without an outcome (2):**

- **YGF** (YanGuFang, ordinary shares): removed by Nasdaq in April 2024; no reason or later
  outcome found.
- **WDRW** (Direxion leveraged ETF), March 2020: why it stopped trading was not found.

**No value, besides the 19 trading gaps (13):**

| Security | Status |
| --- | --- |
| FSBC | Election results not found |
| YGF, WDRW | Unresolved (above) |
| OILD 2016, OILD 2020, SHLD 2021 (ETFs) | Liquidated at net asset value; amount not stated |
| MON 2022, BACA (SPACs) | Trust redemptions; amount not stated (excluded from the universe anyway) |
| MONDW | Warrants redeemed; price not stated |
| CETUR, NOVVW (rights or warrants of SPACs) | Acquisitions without terms |
| GMZ (closed-end fund) | Merged into another fund |

**Unknown type (1):** MDGS, one session of Medigus (an Israeli issuer) in 2016; whether
these were ordinary shares or ADSs was not established.

**Other limitations:**

- **29 outcomes rest on one source,** mostly SEC alone. SEC is a primary source, but its
  categories come from parsing 8-K item numbers and delisting rules.
- **The trading-gap rule is a heuristic.** A new fund launching under a reused ticker at
  the standard $25 price fooled it once (OILD 2016); the confirmed liquidation took
  precedence.
  - **Recommended before backtests:** merge trading-gap segments into one security instead
    of labelling them.
- **8 common stocks have no identified issuer,** so their blank-check status is untested:
  FI→XPRO, FISV→FI, FRC, SCPX ×2, SBNY, SMMC, HYW→STEC. SMMC is a known SPAC fragment; its
  continuation (BTRS) is tested.
- **The universe scale is 122 securities.** The 500-stock universe will bring new case
  types, and curated entries take manual research that grows with it.
- **Research database size:** 1.35 GB, mostly cached SEC documents. It is git-ignored.
- **Disclosure:** the IEX database holds bars up to November 2026 from Milestone 1
  ingestion (unchanged; never read by this research).

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
python scripts/verify_curated_quotes.py
```

The first run asks SEC and Alpaca for whatever is not yet cached; later runs read the
cache and must give byte-identical reports.
