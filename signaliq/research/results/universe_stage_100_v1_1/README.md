# Universe Stage 1.1 (100 securities): cleanup

Stage 1.1 is a cleanup of Stage 1 before any expansion. Stage 1 (v1.0) is preserved
unchanged in `../universe_stage_100/`. Research data only: no strategy, no backtest, no
optimization, no parameter freezing, and no evidence dated 2025 or later. The IEX database
and the Strategy A archive are untouched.

**Status: complete; awaiting review. Stage 2 has not started.**

The selection rule, the membership logic and the exit-value treatment are unchanged:

- top 100 common stocks by 63-session median dollar volume;
- point-in-time decisions;
- one-source outcomes stay lower confidence;
- assumed zeros stay apart from payouts;
- unresolved outcomes stay without a value.

What changed is the evidence behind identity, issuer and type, as listed below.

RESULTS_PLACEHOLDER

## 1. YNDX / NBIS: one security, by a documented exception

Stage 1's duplicate audit failed on Yandex N.V. (YNDX) and Nebius Group (NBIS). They share
one SEC registrant (CIK 1513845), but the 665 sessions without a trade exceed the 250-session
trading-gap rule.

SEC filings by the registrant itself, quoted word for word and re-verified on every run,
establish one continuous security:

- Nasdaq suspended trading in Yandex N.V.'s Class A shares on 2022-02-28 (6-K, March 2022).
- The company renamed itself Nebius Group N.V., "(NASDAQ: YNDX, to be changed to NBIS)"
  (6-K, 2024-08-16).
- Nasdaq determined that trading "will resume" after the divestment of the Russian assets
  (6-K, 2024-10-17).
- Trading in "the Company's Class A ordinary shares" resumed on 2024-10-21 (6-K, 2024-10-21).

The 250-session rule itself is not changed. The link is a named exception
(`CURATED_IDENTITY_LINKS` in `app/universe/curated.py`):

- It joins exactly two segments: YNDX's last trade on 2022-02-25 and NBIS's first trade
  on 2024-10-21.
- It applies only while every quote is found.
- `scripts/audit_merges.py` checks it independently, from raw data:
  - both ends are real trades;
  - neither ticker traded in between;
  - both segments belong to one security;
  - that security's issuer is CIK 1513845;
  - the registrant's own filing list shows no Form 25 or Form 15 in between.

The original records are kept. NBIS's permanent id (898) is registered as "merged into"
YNDX's (616), as for every trading-gap merge. The suspension stays visible: the security has
no bars for 665 sessions, and any later use must treat it as untradeable then.

Tests in `tests/test_universe_merges.py`:

- the exception joins only the named segments;
- a rename alone does not bridge the gap;
- a missing quote leaves the two apart.

## 2. Point-in-time issuer identification

In Stage 1, a still-trading security took as its issuer the company that holds its ticker
in SEC's ticker list *today* (2026), if that company had filed during the period. Tickers
move:

- SQ is now XYZ;
- XOM now points to a new ExxonMobil holding company;
- IAC passed from IAC/InterActiveCorp to a new IAC in 2020.

Nine common stocks were left without an issuer, and a SPAC (FEAC) went undetected.

Now an issuer is dated evidence (`issuer_evidence` table, `evidence.issuer_points`):

- **cover_page**: the company's own 10-K, 10-Q, 8-K, 20-F or 40-F lists the ticker under
  "Trading Symbol(s)", followed by its exchange. SEC has required that cover table since
  2019. The document is read to confirm each company's first and last filing with the
  ticker. A company holds a ticker for one stretch of time, so the end is found by halving.
- **form25**: the delisting notice at the last trade.
- **full_text**: before 2019, or for a filer without the table, a filing in which the
  company names the ticker as its own ("(NYSE: XOM)" right after its name).

Candidate companies come from:

- the delisting notice;
- SEC's ticker list as of today;
- a cover-page search not limited to one company;
- companies that wrote the ticker in an exchange context;
- the companies found for the security's other tickers (a SPAC and the business it became
  share a CIK).

**A candidate is accepted only on its own cover pages. Today's ticker list proposes
companies but never decides.** When two companies' cover pages overlap in time (a REIT and
its operating partnership filing together), the one with more filings is kept, and the
overlap is reported.

**The issuer on a day** is the company of the latest evidence on or before that day
(`evidence.issuer_on`). The blank-check test uses that company's SIC as of that day. The SIC
is read only from filing headers dated on or before the day.

- Before a security's first evidence, the earliest evidence is used, and the report counts
  every decision where that happens.
- Trading-gap decisions compare the issuers at the gap: the earlier security's at its last
  trade, the later security's at its first.
- SEC's older submission pages are now read, so a heavy filer (Toronto-Dominion Bank) no
  longer looks as if it filed nothing in 2016–2024.

Look-ahead tests (`tests/test_universe_issuers.py`):

- dropping every SIC filing and every issuer document dated after a day never changes that
  day's blank-check answer;
- the issuer follows a ticker that passes between companies;
- today's ticker list cannot decide;
- a search hit whose cover page does not list the ticker is not evidence;
- a SPAC and its successor share one company across the rename.

## 3. Deal terms for AGN, WORK and RTN

Terms are quoted from each target's own 8-K on completion. Each value is cash plus shares
times the acquirer's raw SIP close on the effective date, and is used only if it is within
10% of the target's last close (the existing rule):

| Target | Terms per share | Effective | Acquirer price |
| --- | --- | --- | --- |
| AGN (Allergan) | $120.30 cash + 0.8660 AbbVie | 2020-05-08 | ABBV close |
| WORK (Slack) | $26.79 cash + 0.0776 Salesforce | 2021-07-21 | CRM close |
| RTN (Raytheon) | 2.3348 UTC shares (UTC renamed Raytheon Technologies, RTX) | 2020-04-03 | RTX close |

For RTN, UTC spun off Carrier and Otis to its own holders of record. Raytheon holders did
not receive them, so RTX's close on the merger day is the value.

## 4. Security classification rules

The rules are printed at the top of section 3 of the report (`app/universe/sectypes.py`):

1. **Common stock** includes REITs, tracking stock and SPAC shares. It is the only type the
   universe takes.
2. **ADRs** (depositary shares or receipts, New York registry shares) are excluded by design.
3. **ETFs, closed-end and mutual funds, and ETNs** are never common, whatever the issuer is
   called.
4. **Warrants, rights, units, preferred, partnership units and debt** come from the class
   title, OpenFIGI or Nasdaq's fifth letter.
5. **SPACs** are common by type. They are excluded by the point-in-time SIC 6770 rule, not by
   type.
6. **Wikidata describes the company, not the listed security.** "Public company", "bank" and
   "real estate investment trust" all mean *company*, which fits common stock, ADRs,
   partnership units and anything else a company issues; only a fund or a note
   contradicts it.
   - In Stage 1, "real estate investment trust" matched "investment trust" and was read as
     a fund. That is how CCI, DLR, SPG, AVB, INVH, CONE and AAIC fell into conflict.
7. **New source, `sec_cover`:** the class title the issuer gives the ticker on its own cover
   page while the security traded under it. Examples:
   - "Class A common stock" (common);
   - "American depositary shares, each representing 20 Class A common shares" (ADR);
   - "Common units representing limited partner interests" (partnership).
8. **OpenFIGI** describes today's holder of a ticker. It is used only when its company name
   agrees with a company known for the security's own period (Wikidata, cover pages).
9. **Disagreements are settled only by the issuer's own SEC description:** a hand-read SEC
   passage first, then the cover-page class title. Such a type is `resolved`, and the
   outvoted evidence is kept and listed in the report. Otherwise it stays in `conflict`
   (for example, an ADR that became an ordinary listing). Nothing is guessed.

`confirmed`, `single_source` and `resolved` common stocks may be members. Each type keeps
its sources (`type_evidence`).
