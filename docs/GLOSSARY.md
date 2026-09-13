# Glossary

Memecoin and market-data terms, each mapped to where it actually lives in
this codebase. Written for someone picking the project up cold — the
owner after a break, a second assistant, or a reviewer trying to work out
whether a number on the dashboard means what they think it means.

Two things this document is deliberately **not**:

- **Not a strategy.** Nothing here is a threshold, a rule, or an edge.
  Where a constant is quoted it is quoted as documentation of what the
  code currently does, not as a recommendation. Every behavioural
  constant is covered by the strategy version hash
  (`app/strategy/version.py`) and is frozen for the collection run.
- **Not a source of truth.** The code is. If this file and the code
  disagree, the code is right and this file is stale — fix it.

---

## Ecosystem terms

**Blockchain** — a shared ledger every transaction is recorded on. This
bot reads Solana and EVM chains; it never writes to either, because it
is paper-only.

**Token** — an asset issued on an existing chain rather than having a
chain of its own. Every candidate this bot screens is a token.

**Memecoin** — a token whose value rests on attention and community
rather than on a product. The whole design of this bot follows from
that: the security screen matters more than the indicators, because the
dominant risk is not "the price falls", it is "the token was never real".

**Wallet** — a keypair that holds tokens. **This project has none.**
`LIVE_TRADING` and `LIVE_EXECUTION_ACKNOWLEDGED` are both `false` and
there is no private key anywhere in the repository or its deployment.

**DEX (decentralised exchange)** — venue where tokens swap directly
against a liquidity pool rather than through an order book. The fill
model in `app/execution/fill_model.py` simulates a constant-product AMM
because that is what a DEX swap actually executes against.

**Liquidity pool** — the paired reserves a swap trades against. Pool
depth is the single biggest driver of what a trade costs; see *price
impact* below.

---

## Trader slang

These appear in community chatter and in token descriptions. None of
them is an input to any decision the bot makes — they are here so the
vocabulary in a Telegram alert or a Discord post is legible.

**Shill** — promoting a token to attract buyers. **FOMO** — buying
because a price is running. **FUD** — talk that pushes a price down.
**Aping** — buying with no research. **DYOR** — "do your own research".
**Degen** — a trader who knowingly takes very high-risk positions.

**HODL / diamond hands** — holding through drawdowns. **Paper hands** —
selling at the first sign of pressure. Worth knowing because the bot's
exit rules are, in these terms, firmly paper-handed: the median position
in the current record is held about four minutes.

**Whale** — a holder large enough to move the price alone. Measured here
as `top10_holder_pct` and `dev_wallet_pct` on `RugCheckReport`
(`app/rugcheck/filters.py`). Concentration is a *risk input*, and one of
the exit rules fires when a dev or top wallet dumps.

**Bagholder** — someone left holding a token after it collapses.

**Moon** — a very large upward move.

**Pump and dump** — a price inflated deliberately and then sold into.
**Rug pull** — the developers take the liquidity and abandon the
project. Guarding against these is the entire job of
`app/rugcheck/filters.py`, which is **fail-closed**: a token whose
security data cannot be verified is rejected, not approved. That module
is not a tuning candidate.

---

## Core metrics

**Market cap** — `price × circulating supply`. Recorded as
`market_cap_usd` on `MarketSnapshot` (`app/services/price_feed.py`) and
available as a reporting dimension via
`breakdown_by_market_cap()` in `app/analysis/trade_analytics.py`.

**FDV (fully diluted valuation)** — `price × total supply`. Recorded as
`fdv_usd`. The gap between FDV and market cap is how much supply is not
yet circulating — a large gap means tokens are waiting to be unlocked.

**Circulating supply** — coins in the market now. Not stored as its own
field; it is implied by `market_cap_usd ÷ price_usd`.

**Total supply** — every coin that exists, including locked and
reserved. Implied by `fdv_usd ÷ price_usd`.

**Max supply** — the hard cap, if any. **Not recorded.** DexScreener,
the market-data source, does not publish it.

**Burn rate / inflation rate** — the pace at which supply is destroyed
or created. **Not recorded**, same reason. Both would need a chain
indexer, not a price API.

**Distribution** — how supply is spread across holders. Recorded as
`top10_holder_pct` and `dev_wallet_pct`. Concentrated supply is the
precondition for most rug pulls, which is why it is a screen input
rather than a curiosity.

---

## Activity and market quality

**Volume (24h)** — value traded in a day, `volume_24h_usd`. The snapshot
also carries **5m, 1h and 6h** windows, and buy/sell transaction counts
for each. That is deliberate: quality depends on the *shape* of
activity, not the total. A day's volume arriving in one five-minute
burst and the same volume spread evenly are very different markets, and
comparing the windows is what separates them.

**Liquidity** — how much can be traded without moving the price, held as
`liquidity_usd`. The regime classifier labels a pool `THIN` below
$25,000 and `DEEP` above $250,000
(`THIN_LIQUIDITY_USD` / `DEEP_LIQUIDITY_USD` in
`app/signals/market_regime.py`). *Quoted as documentation — these are
frozen.*

**Volatility** — how violently the price moves. Carried as
`price_change_1h_pct` and `price_change_24h_pct`, and classified by ATR
band (`HIGH_VOLATILITY_ATR = 0.030`, `LOW_VOLATILITY_ATR = 0.008`).
Volatility also feeds the fill model: the price keeps moving between
signing a swap and it landing, and that drift is charged against the
fill.

**Price action** — the price path over time. Indicators are computed in
pure Python over candles; no numpy or pandas anywhere, on purpose.

**ATH / ATL (all-time high / low)** — the lifetime extremes. **Not
recorded globally.** What *is* recorded is per-position:
`highest_price_since_entry` and `lowest_price_since_entry`
(`app/exits/manager.py`), which is what the trailing and reversal exits
actually need.

**Price-to-market-cap ratio** — a token's price against its valuation.
The comparable figure computed here is the **liquidity-to-market-cap
ratio** (`app/early/features.py`), which is the more useful one for this
purpose: it says how much of a token's notional value can actually be
sold.

---

## Terms specific to this project

**Paper trading** — every fill in the record is simulated against live
market prices. No real funds, no wallet, no order ever reaches a chain
or an exchange. This is enforced in code and in CI, not by convention.

**Fill model** — `app/execution/fill_model.py`. A simulated swap is
charged `impact + spread + fee + drift`, where impact comes from actual
pool depth via constant-product AMM maths, and **drift is signed** — it
can help or hurt. A swap can also *fail* outright when impact plus
adverse drift exceeds the slippage tolerance, exactly as an on-chain
swap with slippage protection reverts.

**Execution cost** — the total of those four components, stored per leg
as `execution_cost_pct`. Note the unit trap: **it is a fraction**
(`0.004` = 0.4%), despite the name. Everything named `*_pct` in
`fill_model.py` is a fraction; the report multiplies by 100 at the point
of display.

**Spread + fee floor** — the part of a fill's cost that is unavoidable
before impact or drift. A mean fill cost far below this floor is an
arithmetic impossibility rather than good luck, which is what the fill
audit's `BELOW FLOOR` verdict exists to catch
(`app/analysis/fill_audit.py`).

**Strategy version** — a hash over every behavioural setting, scoring
weight, and regime/liquidity constant (`app/strategy/version.py`).
Changing any of them mints a new version and **splits the dataset**,
which is why the configuration is frozen for the collection run.
Reporting code in `app/analysis/` is deliberately *not* covered by the
hash, so fixing how a number is displayed cannot invalidate the sample.

**Champion / challenger** — the champion is the frozen live
configuration. A strategy idea worth testing becomes a challenger and
runs alongside it (`app/shadow/`); it never becomes an edit to the
champion. This is the mechanism that lets the project improve without
destroying its own evidence.

**Unmeasurable** — a value that could not be observed. Recorded as
`NULL`, never as `0`. A token with unavailable volume is not a token
with zero volume, and every score downstream treats those two cases
differently.
