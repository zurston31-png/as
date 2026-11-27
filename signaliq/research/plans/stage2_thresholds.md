# Stage 2 thresholds, for review

This lists every exact numerical threshold in `research/plans/stage2_plan.md`. For each one:
what it measures, and what happens when it fails.

**None of these is approved.** They take effect only once you approve them one by one.
Numbers that are estimates (section D) are not thresholds and are listed only so that none
is mistaken for one.

## A. Selection rule

These parameters define the universe; they are not pass/fail checks. The ones marked
"(Stage 1)" are carried over unchanged. Changing any of them changes the rule fingerprint
and makes a new version.

| # | Threshold | What it measures | If it is not met |
| --- | --- | --- | --- |
| A1 | **N = 250** members per decision | universe size: the 250 eligible common stocks with the highest 63-session median dollar volume | not a check; fewer than 250 eligible means fewer members, and the report says so |
| A2 | **$5.00** minimum raw close on the decision day (Stage 1) | excludes low-priced stocks, using the price as traded (raw, so no later split leaks in) | the security is not eligible that month (`raw_price_below_minimum`) |
| A3 | **126** real trading sessions of history (Stage 1) | about six months of trading before a security can enter | not eligible (`history_too_short`) |
| A4 | **63**-session window (Stage 1) | the window for the median dollar volume, and for activity | not a check; defines the ranking |
| A5 | **50** of the last 63 sessions with a trade (Stage 1) | excludes thinly traded securities | not eligible (`too_few_active_sessions`) |
| A6 | Screen margin **3 × N = 750** | a ticker gets daily data if its monthly traded value ranks within 750 in any month that feeds a decision | a ticker outside 750 every month is left out, unless the completeness proof (B1) adds it |
| A7 | Completeness bound **2 × Σ(monthly high × volume over 4 months) / 63** (Stage 1 formula) | an upper limit on a left-out ticker's 63-session median, from monthly bars only | not a check; B1 uses it |

## B. Gates

A failure stops Stage 2 at the gate. Nothing is relaxed to pass.

| # | Threshold | What it measures | If it fails |
| --- | --- | --- | --- |
| B1 | **0** left-out tickers whose bound (A7) reaches the 250th member at any decision | completeness within the candidate pool | the stage adds them, fetches their data and runs again (automatic, rounds repeat); if any remains after the rounds, the gate fails and is reported ticker by ticker |
| B2 | **59 of 59** decisions identical when recomputed from data cut off at each decision | no look-ahead through later data | gate fails; the differing decisions are investigated, and no result is used until fixed |
| B3 | Every price after **2022-06-30** multiplied by **10**: **0** earlier decisions change | no look-ahead through later prices | gate fails; investigated as B2 |
| B4 | **0** members chosen after their last trade | survivorship and look-ahead | gate fails; each case is investigated |
| B5 | Latest date of every input **before 2025-01-01** | no 2025+ data | gate fails; the offending input is removed and the run repeated |
| B6 | **0** blank-check exclusions not supported by a SIC on file by the decision date; **0** members that were blank-check companies then | point-in-time SPAC exclusion | gate fails; investigated |
| B7 | Nesting: the top **100** of each Stage 2 decision equal the Stage 1.1 members, at **59 of 59** decisions; **0** unexplained differences | consistency with Stage 1.1 (same rule, larger batch) | each difference is listed with its cause; any unexplained one fails the gate. **There is no numeric tolerance.** "Explained" means a named evidence change, so this needs your judgement at the gate. |
| B8 | **0** would-be members with an unknown or conflicting type without a documented decision | that no plausible member is excluded silently for its type | each one needs one of: resolved from the issuer's SEC text, confirmed as not common, or listed as unresolved with the decisions and members it would displace. Any without a decision fails the gate |
| B9 | **0** members that later ended with neither a traced value nor an "unresolved" listing | exit values of members | gate fails until each has a value (stated cash, or deal terms re-verified against the acquirer's close) or an explicit unresolved entry |
| B10 | **0** common stocks without an issuer and without a documented reason | issuer coverage | gate fails; each one is investigated |
| B11 | **0** ticker-days held by two securities; same-ticker successions with a recorded decision **=** all same-ticker successions | duplicate identities | gate fails; investigated |
| B12 | SEC cache growth **=** answers stored (difference **0**), counted per step with retries included | that every request is accounted for | the run reports MISMATCH; the gate fails until explained |
| B13 | **0** SEC answers missing from the cache when the issuer step reads it | that a failed lookup never becomes "no issuer" | the run stops (`SecUnavailable`) instead of continuing; the fetch is repeated |
| B14 | Two full runs: byte-identical reports, **0** differing table fingerprints (all tables) | reproducibility | gate fails; the difference is investigated |
| B15 | Protected data unchanged: the **21** Stage 1.1 table fingerprints; IEX **15,861** bars with fingerprint `e66890e79788fc78`; Strategy A **13 of 13** files in `SHA256SUMS`; **0** commits touching the Stage 1.0, Stage 1.1 or Strategy A archives | that nothing protected was modified | **stop all work** and report: a protected-data failure is never worked around |

## C. Operational limits

| # | Threshold | What it measures | If it is crossed |
| --- | --- | --- | --- |
| C1 | **2 GB** minimum free disk before each new cache write | room to finish safely | the run stops cleanly (`DiskLow`); answers already stored are kept, and a rerun resumes without duplicates |
| C2 | SEC request rate: at most **10** per second (SEC's limit); the prefetch paces at one request every **0.15 s** (about **6.7** per second) | fair access to SEC | not crossed by design; an SEC refusal (HTTP 403/429) is retried with waits, and a request that never succeeds stops the run |
| C3 | New SEC answers longer than **64 KB** are stored compressed (implementation detail behind "stored compressed" in the plan) | disk use | not a check |

## D. Estimates (not thresholds)

These are projections in the plan, to be replaced by measurements at the gates. **No pass or
fail depends on them**:

- 300–600 tickers added by the completeness proof;
- 2,300–2,600 tickers with daily data;
- 2,500–2,900 securities;
- 1,400–1,800 new securities needing SEC evidence;
- 450–550 distinct members;
- 20–40 members that later end;
- 50,000–65,000 SEC requests;
- 2–8 hours of fetching;
- under 6 GB of disk;
- about 3 hours per run and 30 minutes of audits;
- 37 cached answers per security in Stage 1.1;
- 4,500 whole documents to fill the disk.

If a measured figure falls far outside its estimate (for example, distinct members outside
450–550), that is not a failure, but it will be reported and explained before the gate.

## E. Missing from the plan, for your decision

1. **The pool cross-check (plan section 4.2) has no pass threshold.** The plan says only
   that its result "is reported whatever it shows". Proposed: **0** delisted companies
   whose tickers are known but missing from the pool (`gap`); each `unresolved` filer
   (no ticker known from SEC's record) is listed and reviewed, and never counted as
   covered. Not adopted until you decide.
2. **The nesting test (B7) has no numeric tolerance,** by design: every difference must
   have a named cause.
3. **Thresholds inherited from Stage 1 that the plan relies on without restating:**
   - the 250-session trading-gap rule;
   - the 45-day-before / 90-day-after window for a delisting notice;
   - 0.5–2× price continuity for renames;
   - 10 sessions between consecutive securities for an issuer-confirmed rename;
   - 120 days of cover-page evidence around a switch;
   - the 10% tolerance between a deal-terms value and the target's last close.

   They are unchanged from Stage 1.1 and are listed here only so the list is complete. A
   change to any of them would be a new rule needing approval.

## F. Proposed after Phase 1 (raw responses, archive, co-registrant), not approved

Added after Phase 1. Like everything above, none of these is approved.

| # | Threshold | What it measures | If it fails |
| --- | --- | --- | --- |
| F1 | Every Alpaca response used by a stage run is in its raw response store: **0** responses used that were not stored (record run) | that the run's provider data is preserved exactly as received | the run is not accepted; it is repeated in record mode |
| F2 | Replay of the stored responses: report **byte-identical** to the record run, **0** differing table fingerprints (all tables), **0** network requests | reproducibility from preserved data, independent of later provider revisions | gate fails; the difference is investigated (a replay cannot legitimately differ) |
| F3 | Response store integrity: **0** bodies whose sha256 or length does not match | the stored responses are intact | stop; restore the store from the archive and verify again |
| F4 | Off-machine archive: a **fresh copy** from the private destination verifies: sha256 of **every** file, **21 of 21** Stage 1.1 table fingerprints after restore, completeness audit PASS, F3 on the restored store | durable storage that can actually be recovered | Phase 2 does not start ("required before expansion") |
| F5 | Destination visibility **private**, checked before every push | the research data is never published | nothing is pushed |
| F6 | Co-registrant measurement complete: **0** SEC answers missing when the issuer step runs with the rule on, after the approved fetch | that the rule is judged on a complete run, not on 2 of 78 cases | the rule is not adopted; the missing answers are listed for approval |

## G. Proposed after Phase 1c (data quality, attempt logging): final wording for your decision

Added after Phase 1c. **Not adopted**; this is the exact wording for you to accept, change or
reject.

| # | Threshold | What it measures | If it fails |
| --- | --- | --- | --- |
| G1 | **0** flagged auction-only bars without a reviewed explanation and an approved treatment | that defective provider bars are never used silently | gate fails; every flagged bar is listed (date, ticker, normal and flagged trade count, version-2 status, membership impact) and waits for an explanation and a treatment you approve |
| G2 | Every HTTP attempt recorded: attempts logged **=** requests sent; **0** stored responses without their logged 200 attempt; **0** attempts with neither a status nor an error; every request that ended in failure replays as the **same** failure, with **0** network requests | that the provider exchange is preserved completely, failures and retries included | the run is not accepted; it is repeated in record mode into a new, empty store |

**How G1 is measured** (`scripts/audit_auction_bars.py`, read-only, tested in
`tests/test_audit_auction_bars.py`):

1. A bar is flagged when its **security** (not its ticker) had a median of **at least 1,000**
   trades over the **20** market sessions just before it, counting sessions without a bar as 0,
   and the bar has **at most 5** trades. Counting missing sessions as 0 keeps a long suspension
   from looking liquid; using the security keeps a reused ticker from inheriting another
   company's history (both produced the 13 false alarms of the first scan).
2. Every flagged bar must have a row in a reviewed list (TSV: session, security_id,
   explanation, treatment). Treatments: *kept and flagged* (the bar stays, every report names
   it), *excluded from the median*, or *replaced from a named source*. Any treatment other than
   "kept and flagged" changes the data, so it is a new dataset version that needs your approval.
3. For each decision whose 63-session window holds a flagged bar, membership is recomputed
   with the flagged bars left out; every member change is listed.
4. Stage 1.1 today: **767 flagged bars, all unexplained: G1 FAILS** (2021-04-19 499,
   2021-10-25 221, 2022-03-08 47).

Limits: it finds only near-empty bars (5 trades or fewer). A day missing half its trades is not
caught, and the three numbers (1,000 / 20 / 5) are themselves part of what you approve.

**How G2 is measured:**

1. Each record run starts with a **new, empty** store (the driver refuses an existing file), so
   the store holds exactly that run's attempts.
2. The number of rows in `attempts` equals the number of "Alpaca request" lines in the client's
   log for the run (one line per attempt, written before the request is sent).
3. `scripts/response_store_tool.py verify` reports 0 problems: every stored body (responses and
   failed attempts) matches its sha256 and length; every stored response has its logged 200
   attempt with the same sha256; no attempt lacks both a status and an error.
4. The offline replay meets F2, and every request whose last attempt failed raises the same
   error class and status as in the record run (shown on a local fake provider in
   `research/results/phase1d/attempt_log_controlled_test.txt`, with a negative control).

Limits: it covers the Alpaca client only (SEC requests have their own cache counts, B12). Stores
recorded before attempt logging (the Phase 1b store) cannot meet G2. They still replay, and
`verify` skips the attempt checks for them.
