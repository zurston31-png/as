# Memecoin Trading Bot

A fully autonomous memecoin trading bot: TradingView Pine Script alerts feed
a FastAPI webhook server that screens every buy for rug-pull/scam risk,
sizes and executes the trade under hard-coded risk limits, manages
stop-loss/take-profit/dev-wallet-exit monitoring in the background, and
reports everything to Telegram/Discord and a small dashboard. **Starts in
paper trading mode and stays there until you explicitly flip a flag.**

> **This is not financial advice, and memecoins are extremely high risk.**
> See [Disclaimer](#disclaimer) before you fund a real wallet.

> 🪟 **New to this / on Windows?** Start with
> **[GETTING_STARTED_WINDOWS.md](GETTING_STARTED_WINDOWS.md)** — a
> no-experience-required walkthrough. Double-click `START_HERE.bat` and it
> sets everything up for you. The rest of this README assumes some
> command-line familiarity.

## Architecture

```
TradingView (Pine Script alert)
        │  webhook POST (JSON, shared secret)
        ▼
FastAPI webhook listener  ──▶  Signal persisted (SQLite/Postgres)
        │
        ▼ (buy signal)
Risk gate (halted? max positions?) ──reject──▶ notify + log
        │ ok
        ▼
Rug-check filter (GoPlus + honeypot.is) ──fail──▶ notify + log, no trade
        │ pass
        ▼
Position sizing (% of portfolio, hard-capped)
        │
        ▼
Execution backend (Jupiter / CEX via ccxt / experimental EVM / paper)
        │
        ▼
Position opened with SL/TP  ──▶  Telegram/Discord notification

Background loop (independent of webhooks):
  - polls open positions' price every N seconds → stop-loss / take-profit exit
  - polls holder distribution → dev-wallet-sell auto-exit
  - daily P&L summary job

Dashboard (HTTP Basic Auth): open positions, trade history, stats, halt/resume
```

Everything is persisted to the database (`Signal`, `RugCheckResult`, `Trade`,
`Position`, `RiskEvent`, `DailyPnL`) for a full audit trail of what the bot
saw and why it did or didn't act.

## Project layout

```
app/
  main.py                FastAPI app: webhook, startup/shutdown, scheduler
  config.py               all tunable settings (env-driven)
  models.py / database.py / state.py / schemas.py / security.py
  rugcheck/                GoPlus + honeypot.is + aggregated filter logic
  execution/                paper / jupiter (Solana) / cex (ccxt) / evm (experimental)
  risk/manager.py          position sizing, daily loss halt, SL/TP, hard ceilings
  services/                trading_service (orchestrator), portfolio ledger,
                            price feed (DexScreener), daily reporting
  monitor/                  background SL/TP + dev-wallet-sell loop
  notifications/            Telegram + Discord
  dashboard/                server-rendered status page + halt/resume controls
pine/memecoin_signal_strategy.pine   TradingView indicator + alert() payload
deploy/                    systemd unit + VPS setup guide
scripts/init_db.py
tests/                      pytest suite (risk manager, rug filter, webhook)
```

## Quickstart (paper trading, local)

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# at minimum, set WEBHOOK_SECRET to a long random string:
python -c "import secrets; print(secrets.token_urlsafe(32))"

python scripts/init_db.py
uvicorn app.main:app --reload --port 8000
```

In another terminal, expose it publicly for TradingView (dev only):

```bash
ngrok http 8000
```

Open `http://localhost:8000/` (HTTP Basic Auth: `DASHBOARD_USERNAME` /
`DASHBOARD_PASSWORD` from `.env`) to see the dashboard. `LIVE_TRADING=false`
by default, so every fill is simulated — no real funds or keys are touched.

## Automatic token scanner

The bot finds its own tokens — you don't have to paste a contract address or
build a TradingView chart per memecoin. Every 60s
(`SCANNER_INTERVAL_SECONDS`):

```
DexScreener + Birdeye new listings
   ↓  discover           app/scanner/discovery.py
   ↓  free pre-screen    app/scanner/filters.py   (liquidity / volume / age / txns / sell pressure)
   ↓  signal score       app/signals/live_gate.py (0-100, must clear MIN_SIGNAL_SCORE_TO_ENTER)
   ↓  rug check          app/rugcheck/            (binary gates + 0-100 rug risk score)
   ↓  risk + sizing      app/risk/manager.py      (exposure caps, cooldowns, halts)
   ↓  paper position     monitored by the usual stop-loss / take-profit / smart exits
```

**DexScreener** needs no API key and lists any token once it has a pool and
a trade. **Birdeye** is optional (`BIRDEYE_API_KEY`) and adds its dedicated
new-listing feed, including launchpad/meme-platform tokens. Either source
failing is logged and skipped, not fatal — a discovery outage means "no
candidates this cycle", never a crash.

The scanner is a **source of signals, not a second trading path**: it
records a `Signal` with `source="scanner"` and calls the exact same
`_handle_buy_signal` a webhook alert does. Everything already built — the
risk gate, the score gate, the rug filter, exposure-aware sizing, the smart
exits, the journal, the dashboard — applies unchanged, because there is no
separate implementation that could drift from it. Tests assert this
directly: a scanner candidate is rejected by a failing rug check, a weak
signal score, and an active trading halt.

Cost ordering is load-bearing. A cycle can surface hundreds of new mints;
the free pre-screen rejects most of them using data already in the listing
payload, so the expensive stages (several rug-check lookups, a pool
resolution plus candle fetch) only ever run on the handful worth it. A
`ScannedToken` row per candidate records how far it got and why it stopped,
visible in the dashboard's **Auto Scanner** panel — so "found 300 tokens,
traded none" is answerable rather than mysterious.

```bash
python scripts/scan_once.py           # discover + pre-screen only, no trades
python scripts/scan_once.py --trade   # one full cycle (opens PAPER positions)
```

Run the first form on your real server before leaving the scanner
unattended: the DexScreener/Birdeye response shapes are the one part that
couldn't be verified from the development sandbox, and that script shows
you exactly what comes back plus how your `SCANNER_MIN_*` thresholds are
filtering it.

**Safety**: with `LIVE_TRADING=true` the scanner refuses to run unless
`SCANNER_ALLOW_LIVE_TRADING=true` is also set. Auto-discovering tokens
nobody has vetted and buying them unattended with real money is a much
bigger step than either autonomy or live trading alone, so it takes its own
deliberate opt-in rather than arriving as a side effect of an upgrade.

## Setting up TradingView alerts

1. Open `pine/memecoin_signal_strategy.pine` in TradingView's Pine Editor,
   add it to each chart in your watchlist.
2. In the indicator's settings, fill in:
   - **Webhook Secret** — must exactly match `WEBHOOK_SECRET` in `.env`.
   - **Token/Contract Address** — the on-chain mint/contract for that
     chart's symbol. Required for the rug-check filter and DEX execution;
     without it, buy signals are rejected outright (fail-closed by design).
   - **Chain** — `solana`, `ethereum`, `bsc`, `base`, etc.
3. Tune the strategy inputs (RSI/EMA/volume/breakout lengths — see
   [Tuning the strategy](#tuning-the-strategy) below).
4. Right-click the chart → **Add Alert** → Condition: this indicator →
   **"Any alert() function call"** → Webhook URL:
   `https://<your-domain-or-ngrok>/webhook/tradingview` → Create.
   The alert message is generated by the script itself (JSON with symbol,
   signal, price, timestamp, and indicator values), so the dialog's
   Message box is ignored.
5. Repeat per symbol. `SYMBOLS_WATCHLIST` in `.env` is informational only
   (shown on the dashboard) — the real watchlist is whatever charts you've
   added alerts to in TradingView.

## Tuning the strategy

All in the Pine script's input panel, per chart:

| Input | Default | Effect |
|---|---|---|
| RSI Length / Oversold / Overbought | 14 / 30 / 70 | Momentum filter — buys are blocked once RSI ≥ Overbought |
| EMA Fast / Slow Length | 9 / 21 | Crossover trigger for buy/sell |
| Volume SMA Lookback / Spike Multiplier | 20 / 2.0× | Buys require volume ≥ multiplier × its own SMA |
| Breakout Lookback | 20 | "N-period high" for the momentum breakout filter |
| Require close above breakout high | true | Turn off to trade the EMA/RSI/volume combo without the breakout confirmation |

Buy fires on: bullish EMA cross **and** volume spike **and** RSI not
overbought **and** (optionally) a new N-period high. Sell fires on: bearish
EMA cross, **or** RSI overbought, **or** RSI oversold while price is below
the slow EMA (momentum breakdown). Edit the `buyCondition`/`sellCondition`
expressions directly in the script for anything more custom.

## Risk management

Configured in `.env`, enforced in `app/risk/manager.py` on **every** trade —
there is no code path that bypasses `RiskManager`:

### Position sizing is risk-based, not a flat percentage

`MAX_PORTFOLIO_PCT_PER_TRADE` is the fraction of the portfolio you're willing
to **lose** on a trade if its stop is hit — not the position size itself.
The bot solves for the notional that makes that true:

```
risk_amount = portfolio_value × MAX_PORTFOLIO_PCT_PER_TRADE
position_size_usd = risk_amount / stop_loss_pct
```

A wider stop therefore produces a *smaller* position and a tighter stop a
*larger* one, so the dollar amount actually at risk stays constant no
matter where the stop sits. The earlier version sized every trade at a
flat percent of the portfolio regardless of stop distance, so "2% risk"
was really just the notional size — the real risk floated with the stop
with no cap on it at all.

Because sizing reads the *current* portfolio value on every trade, a
losing streak shrinks the next position automatically — there is no code
path that can scale a position up to recover a previous loss.

| Setting | Default | Meaning |
|---|---|---|
| `MAX_PORTFOLIO_PCT_PER_TRADE` | 2% | Fraction of the portfolio at risk per trade (see formula above), capped further by `MAX_TRADE_SIZE_USD` |
| `MAX_TRADE_SIZE_USD` | $200 | Absolute per-trade cap |
| `DAILY_LOSS_LIMIT_PCT` | 5% | If realized losses today exceed this % of `PORTFOLIO_STARTING_BALANCE_USD`, the bot halts and alerts you |
| `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT` | 15% / 30% | Set on every position at entry, enforced by the background monitor loop |
| `MAX_CONCURRENT_POSITIONS` | 5 | New buys are rejected once this many positions are open |
| `MAX_EXPOSURE_PER_TOKEN_PCT` | 10% | Cap on notional in a single token, as a fraction of portfolio value; sizing is clipped to whatever room remains |
| `MAX_TOTAL_EXPOSURE_PCT` | 60% | Same cap, portfolio-wide across all open positions |
| `MAX_CONSECUTIVE_LOSSES` | 4 | Auto-halts after this many losing trades in a row, independent of the daily $ loss limit — a losing streak can stay under that limit while still signaling the strategy has stopped working |
| `MAX_DAILY_TRADES` | 8 | Hard cap on new positions opened per calendar day (UTC) |
| `TRADE_COOLDOWN_SECONDS` | 900 | Minimum time between trades on the same symbol, so the bot can't immediately re-enter right after being stopped out |

A signal can pass every other check and still be sized to **$0** (and
rejected as `exposure_cap_rejected`) if the per-token or total exposure cap
is already used up — see `RiskManager.position_size_usd` in
`app/risk/manager.py`.

**Hard ceilings** in `app/risk/manager.py` (`HARD_MAX_*` / `HARD_MIN_*`
constants) clamp all of the above regardless of what `.env` says — e.g.
`MAX_PORTFOLIO_PCT_PER_TRADE` can never exceed 10%, `DAILY_LOSS_LIMIT_PCT`
can never exceed 25%, `MAX_EXPOSURE_PER_TOKEN_PCT` can never exceed 25%,
and `MAX_CONSECUTIVE_LOSSES` can never be set below 2 (so a single loss
can't be configured into a full shutdown) — no matter how `.env` is
misconfigured. Edit the constants themselves if you deliberately want a
wider band.

When the daily loss limit **or** the consecutive-loss limit trips, the bot
sets a persisted `trading_halted` flag, sends a Telegram/Discord alert, and
**stops opening new positions** until you resume it from the dashboard
("Resume trading" button) or via `POST /api/halt` / `POST /api/resume`
(Basic Auth).

## Smart exits

The background monitor loop (`app/monitor/position_monitor.py`) checks
every open position on each tick against `app/exits/manager.py`, layered on
top of the fixed stop-loss/take-profit set at entry:

- **Break-even stop** — once up `BREAK_EVEN_TRIGGER_PCT`, the stop moves to
  entry + `BREAK_EVEN_BUFFER_PCT`, so the trade can no longer turn into a
  loser from there even if it fully reverses.
- **Trailing stop** — once up `TRAILING_STOP_ACTIVATION_PCT`, the stop
  starts trailing `TRAILING_STOP_DISTANCE_PCT` behind the peak price and
  ratchets up with it; it never loosens on a pullback.
- **Partial profit-take** — once, at `PARTIAL_TAKE_PROFIT_TRIGGER_PCT` gain,
  sells `PARTIAL_TAKE_PROFIT_SIZE_PCT` of the position and lets the rest run
  under the remaining rules.
- **Momentum-loss exit** — exits if price falls `MOMENTUM_EXIT_DROP_PCT` off
  its peak within the last `MOMENTUM_EXIT_LOOKBACK_SAMPLES` monitor ticks,
  catching a sharp reversal faster than the trailing stop's own distance
  would.
- **Trend-reversal exit** — exits on two consecutive lower highs after the
  position's peak, read from the same recent-tick samples.
- **Time-based exit** — off by default; when enabled
  (`TIME_BASED_EXIT_ENABLED=true`), force-exits a position after
  `MAX_POSITION_AGE_HOURS` regardless of price.

All of these are built from prices this specific position actually traded
through since entry, not a live indicator feed — the bot doesn't have live
OHLCV wired in for on-chain memecoins yet (see `app/data/`, which today only
feeds the backtester). The fixed stop-loss and take-profit from the risk
manager always take priority and fire even if every rule above is disabled.
Every close records the exact reason (`Position.close_reason` /
`Trade.pnl_usd`) — e.g. `"trailing stop hit at $0.00123 (peak $0.00145)"` —
so the trade journal can show why each trade actually ended, not just that
it did.

## Rug-pull / scam filter

Runs before every buy (`app/rugcheck/filters.py`), fail-closed (missing
data blocks the trade, it does not default to "pass"):

- **Ownership / mint authority** — rejects if the mint/owner can still
  inflate supply (via GoPlus Security API).
- **Honeypot** — GoPlus's `is_honeypot` flag on all chains, plus a
  honeypot.is simulation on EVM chains (Solana isn't supported by
  honeypot.is, so it relies on GoPlus there).
- **Liquidity locked** — rejects unless ≥50% of LP tokens are tagged
  locked/burned.
- **Holder concentration** — rejects if the top 10 holders own more than
  `MAX_TOP10_HOLDER_PCT` (default 35%) of supply.
- **Liquidity depth** — rejects if liquidity is under `MIN_LIQUIDITY_USD`
  (default $15,000), and separately rejects if the sized position would
  exceed `MAX_PRICE_IMPACT_PCT` of that liquidity.
- **Dev wallet monitoring** — the largest non-LP holder at entry (preferring
  GoPlus's `creator_address` when present) is treated as the "dev wallet."
  The background monitor re-checks it every `DEV_WALLET_POLL_INTERVAL_SECONDS`
  and auto-exits the position if that wallet's share of supply drops by more
  than `DEV_WALLET_SELL_ALERT_PCT` (default 10 points) from entry — a
  classic pre-rug signal. This is a heuristic, not ground truth; treat it as
  an early warning.

Set `RUGCHECK_ENABLED=false` only if you fully understand the risk — the
bot will log a loud warning and trade without screening.

### Rug Risk Score (0-100)

On top of the binary checks above, every screened token also gets a
composite **Rug Risk Score** from `app/rugcheck/risk_score.py` — 0 (safe) to
100 (critical) — from 14 independent factors: mint/freeze authority,
honeypot status, liquidity lock, liquidity depth, holder concentration, dev
wallet concentration, scanner danger flags, token/pool age, the 24h
volume-to-liquidity ratio, buy/sell imbalance, and 1h price-swing-vs-liquidity
(a manipulation heuristic). A token scoring `REJECT_RUG_SCORE_ABOVE` (default
65, `"high_risk"`/`"critical"`) or above is rejected **even if it passed
every binary check** — this is an additional gate layered on top, never a
replacement for them.

Two factors — liquidity change and suspicious transfer patterns — are
always reported unavailable rather than faked: a pre-trade screen is a
single snapshot with no prior observation to diff against, and no scanner
field for transfer patterns is wired up yet. Like the signal score, a
factor with no data scores neutral (50/100 on that factor) and is flagged,
never scored as if it were safe — and if too much of the score has no data
behind it, the whole score is marked `reliable: False` rather than quietly
presented as trustworthy. The score, level, and full per-factor breakdown
are persisted on every screened signal (`RugCheckResult.rug_risk_score` /
`rug_risk_level` / `rug_risk_factors`) and shown in the dashboard's Recent
Signals panel, so a rejection or a pass can both be read back with the
reasoning behind it.

## Execution backends

Set via `CHAIN` + `EXECUTION_BACKEND` in `.env`. **Whenever `LIVE_TRADING=false`,
the paper engine is used no matter what `EXECUTION_BACKEND` says** — this is
enforced in `app/execution/__init__.py`, not just a default.

- **`jupiter`** (Solana, default) — quote, sign, and submit are all
  implemented (`app/execution/jupiter.py`). A past version of this file
  computed entry price from raw base units, mixing USDC's 6 decimals with
  the token's own — correct only for a 6-decimal token, and Solana
  memecoins commonly use 9, so the stop-loss/take-profit derived from it
  would be wrong by 1000x. Fixed: every conversion between raw base units
  and whole tokens/USD now goes through the mint's actual on-chain decimals
  (`get_mint_decimals`, via Solana RPC `getAccountInfo`, cached per mint).
  That math is unit-tested against mocked RPC responses
  (`tests/test_jupiter_decimals.py`) but the sign-and-submit path itself has
  never been exercised against a funded wallet or mainnet — see
  `LIVE_EXECUTION_ACKNOWLEDGED` below before using this for real money.
- **`cex`** — Binance/Coinbase/Kraken/etc. via `ccxt`. Use an API key with
  **trade-only** permission (never enable withdrawals on it). The only
  backend built on a mature, widely-used execution library rather than
  code written for this project — `LIVE_EXECUTION_ACKNOWLEDGED` is not
  required for it.
- **`evm_1inch`** — quote, sign, and submit are implemented
  (`app/execution/evm.py`) using standard EIP-1559 gas pricing and
  `eth_getTransactionCount` nonce management. Deliberately does **not**
  implement mempool resubmission/replacement or MEV-aware submission —
  building those properly is its own project, and the file says so rather
  than presenting a bare-minimum signer as production-grade. Same
  never-tested-against-real-funds caveat as Jupiter applies; see
  `LIVE_EXECUTION_ACKNOWLEDGED` below.
- **`paper`** — force paper fills even with `LIVE_TRADING=true`, useful for
  extended live-data paper testing.

### The second live-execution gate

`LIVE_EXECUTION_ACKNOWLEDGED=true` is required in **addition** to
`LIVE_TRADING=true` before `jupiter` or `evm_1inch` will submit anything —
not because they're known to be broken (unlike the old Jupiter decimals
bug, which returned an explanatory error instead of trading), but because
their sign-and-submit code has only ever been run against mocked responses
in this repo's tests, never a funded wallet. Requiring a second, separate
flag makes arming one of them a deliberate decision rather than a side
effect of flipping `LIVE_TRADING` alone. Read the relevant file yourself
first, and start with a small trade size. `cex` doesn't need this flag —
it's built on `ccxt`, a mature library this project didn't write.

Live-only dependencies (`solders`, `solana`, `ccxt`, `web3`) live in
`requirements-live.txt` so the base install for paper trading stays light:

```bash
pip install -r requirements.txt -r requirements-live.txt
```

## Monitoring & notifications

Set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` and/or `DISCORD_WEBHOOK_URL`
in `.env` (either, both, or neither — each is independently optional). You'll
get a message for: trade executed, rug-check rejection, risk-halt triggered,
errors, and a daily P&L summary at `DAILY_SUMMARY_HOUR_UTC`.

The dashboard (`/`, HTTP Basic Auth) shows an equity curve (built from
closed-trade P&L, rendered as a dependency-free inline SVG — no JS charting
library or CDN involved), portfolio stats (win rate, profit factor,
expectancy, max drawdown, current win/loss streak, realized-today and
unrealized P&L, current exposure in $ and % of portfolio), open positions
with live current price, unrealized P&L, age, and the signal/rug scores
recorded at entry, recent trades, recent signals with their rug score and
verdict, recent risk/rejection events, and current mode (PAPER/LIVE) with a
halt/resume control. It auto-refreshes every 20s.

### Trade journal

`/journal` (same auth) shows one record per position — a full round trip,
or still open — assembling what's otherwise spread across the dashboard's
separate tables: entry price/size/stop/target, the TradingView indicator
payload that triggered it (RSI, EMA9/21, volume vs its SMA, breakout
level), the live 0-100 signal score computed at entry (see "Live signal
score" below), the rug check's score and the scanner that produced it, and
every exit leg (full or partial) with its own reason and P&L —
`Trade.close_reason` is recorded per leg, not just once on the position,
since a position can close in more than one step (a partial profit-take,
then later a full exit for a different reason). This is "study why the bot
won or lost" made concrete.

### Performance and validation (`/performance`)

The page that answers the question the headline numbers cannot: **is this
record strong enough to believe?** It leads with a validation verdict
rather than a return, because "+34%" and "on 12 trades" mean very
different things and the second is the part people skip.

Every strategy starts **EXPERIMENTAL** and stays there until it clears
every criterion in `app/analysis/validation.py`: at least 100 closed
trades, positive expectancy *after costs*, profit factor ≥ 1.30, realized
drawdown ≤ 25%, no single trade contributing more than 40% of gross
profit, a resampled 95th-percentile drawdown ≤ 35%, a profitable
out-of-sample run, and a majority of walk-forward windows profitable.

Three statuses, and the distinction between the first two is the point:

| Status | Meaning |
| --- | --- |
| `EXPERIMENTAL` | not enough evidence yet — whatever the numbers say |
| `FAILING` | enough evidence, and the answer is no |
| `VALIDATED` | cleared every criterion |

Insufficient evidence is never treated as a pass, and never as a failure
either. Too few trades, an infinite profit factor from having no losing
trades yet, and a concentration ratio computed from under ten winners are
all "we don't know", because branding a young strategy a failure for doing
arithmetic would be as wrong as calling it proven. The thresholds are
module constants rather than settings — a bar that can be lowered from
`.env` when the strategy fails to clear it is not a bar. And
`cleared_for_real_money` is `False` by construction even for a VALIDATED
strategy: every fill on that page was simulated, and simulated evidence is
not evidence about real execution.

Below the verdict: where the money actually went (fees, slippage, total
execution cost, and what share of legs have cost data recorded at all),
holding-time distribution including whether losers are being held longer
than winners, largest win/loss, and per-bucket breakdowns by signal score,
market quality, entry liquidity, holding time and exit reason — each
bucket carrying its own trade count and flagged `THIN` below 20, so a
suggestive-looking row cannot be mistaken for evidence.

**Monte Carlo** (`app/analysis/monte_carlo.py`) resamples the trade
sequence, because a run produces one path and the order of that path is
luck. Shuffle mode reorders the same trades to isolate *path risk* (same
total, different ride); bootstrap mode draws with replacement to estimate
*outcome risk* (what range of results is consistent with this edge, and
what the chance of ending down over the next N trades actually is). The
95th-percentile resampled drawdown, not the realized one, is the number to
size positions against.

Also available as `GET /api/performance` (JSON) and
`python scripts/performance_report.py` (terminal, with `--version`,
`--list-versions` and `--json`).

**Strategy versioning** underpins all of it: every Signal and Trade is
stamped with a deterministic hash of the 46 settings that materially
change trading behaviour (`app/strategy/version.py`). Cosmetic changes —
log level, poll intervals, dashboard password — deliberately do not mint a
new version, since fragmenting history on those costs analytical power and
buys nothing. Reporting across more than one version produces a loud
warning, because a win rate spanning a threshold change describes a
strategy that never existed.

### Scanner pipeline and system health (`/pipeline`)

The funnel, stage by stage:

```
FOUND -> PRE-SCREEN -> EVALUATED -> SECURITY -> MARKET QUALITY -> SIGNAL -> PAPER BUY
```

with how many reached each stage and how many were rejected there. **A
narrow funnel is the design, not a fault** — this bot exists to reject
weak setups, and rejecting the overwhelming majority of brand-new listings
is the expected shape. The only legitimate conclusion to draw from a
narrow funnel is about the quality of what *discovery* is surfacing, never
that a filter should be lowered to let more through. The page says so
directly, because a funnel chart that narrows hard invites exactly the
wrong reaction.

Alongside it, an **upstream health panel** (`app/services/api_health.py`)
showing per-service last success, failure count, consecutive failures and
the last error. This exists because every gate in this bot fails closed,
which means a dead API and a genuinely quiet market produce *identical*
output: no trades, and logs that say "rejected: no data". The health table
is the only thing that tells them apart. A degraded service never relaxes
a gate — it is reported, not compensated for.

Clicking any mint opens **`/token/<mint>`**: everything the bot knows about
that token on one timeline — when it was discovered, every signal with its
scores, every rejection with its reason, every fill, and a one-line verdict
saying what the bot ultimately did and why. Keyed on the mint address,
never the symbol: symbols are not unique and are trivially spoofed, so
looking up by symbol could merge a scam clone's history into the real
token's.

### Live signal score

The 0-100 composite signal score (`app/signals/scoring.py`) gates every
live buy, not just backtests. `app/data/live_provider.py` fetches live
OHLCV candles from [GeckoTerminal](https://www.geckoterminal.com/dex-api)
(free, no API key — chosen over paid alternatives like Birdeye for that
reason), resolving the token's highest-liquidity pool first since
GeckoTerminal's OHLCV endpoint is keyed by pool, not by token address.
`app/signals/live_gate.py` scores the result and is **fail-closed**: no
trustworthy live candle data (fetch failure, an unmapped chain, too few
candles — `SIGNAL_SCORE_MIN_CANDLES`) rejects the trade outright, the same
rule the rug-check filter already follows for missing data. A score below
`MIN_SIGNAL_SCORE_TO_ENTER` (default 75) or marked unreliable (too much
missing data within the score itself) is rejected too, before the rug
check ever runs. Set `LIVE_SIGNAL_SCORE_ENABLED=false` to skip this gate
entirely (loud warning logged) and fall back to the pre-Stage-9 behavior:
entries on the TradingView alert + rug check alone.

**Honesty note**: `app/data/live_provider.py` is written from
documented/trained knowledge of GeckoTerminal's public API shape, not
verified against a live response from this development environment
(outbound HTTP to public APIs is proxied/restricted there in a way it
won't be on your deployment server). Every parse is defensive — a shape
mismatch fails closed (rejects trades) rather than admitting bad ones —
but run `python scripts/diagnose_token.py` against a real token address on
your actual deployment (step 4/4) before trusting this gate's verdicts
unattended; it fetches the exact same pool-resolution and OHLCV endpoints
this module uses and tells you plainly if the shape doesn't match.

## Realistic paper execution

Paper fills are simulated by `app/execution/fill_model.py` against the
**actual pool** the trade would hit, not a flat assumption. Five costs:

- **Price impact** — derived, not configured. For a constant-product AMM,
  buying with `d` against quote-side reserve `R` gives
  `effective/spot = 1 + d/R`, and DexScreener's `liquidity.usd` is the total
  across both sides, so impact = `2 × trade_usd / liquidity_usd`. A $20 and
  a $20,000 trade against the same thin pool now cost wildly different
  amounts — the flat 0.5% this replaces charged them **identically**, which
  is the main way a memecoin paper simulator flatters itself.
- **Spread** (`PAPER_SPREAD_PCT`)
- **Confirmation delay drift** — a swap isn't instant; the price keeps
  moving between signing and inclusion, scaled from the token's own 1h
  volatility by `sqrt(t)`. Signed, not always adverse.
- **Fees** (`PAPER_FEE_PCT`), per side
- **Failed fills** — when impact + adverse drift exceed `SLIPPAGE_BPS` the
  swap **reverts**, exactly as on-chain. `slippage_bps` was previously
  accepted by the paper engine and silently ignored, so paper trading never
  experienced a failed fill at all.

What this changes in practice — the same $2,000 buy:

| Pool liquidity | Price impact | Outcome |
|---|---|---|
| $20,000 | 20.0% | reverts (over tolerance) |
| $100,000 | 4.0% | reverts |
| $500,000 | 0.8% | fills at 1.18% total cost |
| $5,000,000 | 0.08% | fills at 0.46% total cost |

`app/startup_checks.py` warns if `MAX_TRADE_SIZE_USD` against the thinnest
pool the scanner accepts (`SCANNER_MIN_LIQUIDITY_USD`) implies impact above
your slippage tolerance — those fills would silently revert forever.

**On `scripts/run_demo.py`:** it prints a ~+31% trade. That price move is
*scripted* ($1.00 → $1.33) to demonstrate the entry→monitor→exit cycle. It
is **not** a trading result and says nothing about whether the strategy
works. The demo now says so in its own output.

## "It never trades" — how to diagnose

```bash
python scripts/why_no_trades.py
```

Reads the bot's own database and walks the funnel — config sanity, halt
state, what was discovered, where candidates died, what the risk gates
rejected — and names the most likely cause. Read-only; it never places or
cancels anything.

This exists because "the bot never trades" and "the market had no good
setups" produce *identical* logs, and the shipped defaults used to be
silently incapable of trading at all. Two of them contradicted each other:

- `MIN_SIGNAL_SCORE_TO_ENTER` was **75**, which measurement showed is
  roughly the **97th percentile** of what the scoring engine actually
  produces. The score is a weighted average of 14 factors, so it regresses
  toward 50 by construction and its practical ceiling is nowhere near 100.
  Only ~8% of setups reached 75 *before* the rug check took its cut.
- `SCANNER_MIN_TOKEN_AGE_HOURS` was **6**, but the score gate needs
  `SIGNAL_SCORE_MIN_CANDLES × SIGNAL_SCORE_TIMEFRAME` = 60 × 15m = **15
  hours** of history. Every token between those two ages passed the
  pre-screen and was then guaranteed to fail the score gate, forever.

Both are fixed (65 and 16h), and `app/startup_checks.py` now validates
config coherence at boot — logging loud warnings and surfacing them on the
dashboard — so this class of mistake is loud instead of silent. The checks
only ever warn; they never override a value you set deliberately.

Measured threshold behavior, if you want to tune it:

| Threshold | Setups qualifying | Of those, genuinely trending |
|---|---|---|
| 60 | ~53% | 89% |
| **65 (default)** | **~40%** | **89%** |
| 70 | ~21% | 93% |
| 75 | ~8% | 100% |

Raise it toward 70 if paper trading shows too many marginal entries; lower
it toward 60 for more activity. A regression test
(`test_shipped_defaults_can_actually_trade`) fails if a future change
pushes the defaults back into never-trading territory.

## Operational notes

**Upgrading an existing deployment.** `init_db()` runs an additive schema
migration on startup (`app/migrations.py`): SQLAlchemy's `create_all()`
creates new tables but never adds a column to a table that already exists,
so a database from an earlier build would otherwise come back up missing
every column added since and die on the first query. The migration adds
missing columns only — it never renames, drops, or retypes anything, and it
never does a computed backfill. Columns with a simple scalar default take
that default on existing rows (a signal written before the scanner existed
really was `source="tradingview"`); everything else lands NULL, which
honestly means "not recorded at the time". If this project ever needs a
destructive or data-transforming change, that's the point to adopt Alembic
— this deliberately refuses to do the dangerous half.

**Rate limits.** Every data source here is a free or cheap public tier, and
three subsystems hit them at once: the scanner (a batch every 60s), the
position monitor (a price per open position every 30s), and the trade path
(rug-check lookups plus a candle fetch per candidate). `app/services/http.py`
retries 429s and 5xx with jittered exponential backoff, honoring
`Retry-After` when the server sends one, and does *not* retry other 4xx
(a bad address fails the same way however often you ask). Jitter matters
because the scanner fires a batch on a fixed tick — a fixed backoff would
have them all retry in lockstep and re-trigger the same limit. If you're
getting throttled, the log says so explicitly rather than looking like
"no good setups".

**Paper fees.** Paper mode charges `PAPER_FEE_PCT` (default 0.25%) per side
on top of its slippage buffer, matching `BacktestConfig.fee_pct`. Without
it, paper trading was systematically cheaper than the backtest meant to
validate it — a strategy could pass on paper purely because paper was
free, which is the wrong direction for an error to lean.

## Deployment

See [`deploy/vps_setup.md`](deploy/vps_setup.md) for full step-by-step
instructions (Docker and systemd paths, ngrok for dev, Caddy/nginx for TLS
in production, firewall). Summary:

```bash
cp .env.example .env && nano .env
docker compose up -d --build
docker compose logs -f bot
```

`restart: unless-stopped` (Docker) or `Restart=always` (the provided
`deploy/systemd/memecoin-bot.service`) means the bot comes back automatically
after a crash or reboot — no manual restart needed.

## Going-live checklist

1. **Paper trade for at least 1–2 weeks.** Review every simulated trade,
   rejection, and halt in the dashboard/DB. Confirm the strategy and risk
   limits behave the way you expect.
2. Fund a **dedicated** wallet/exchange sub-account with only the capital
   you're fully prepared to lose. Memecoins carry substantial risk of total
   loss, including from tokens that pass the rug-check filter — the filter
   reduces but does not eliminate scam risk.
3. Set `PORTFOLIO_STARTING_BALANCE_USD` to match what you actually funded —
   in live mode the bot tracks P&L against this figure via an internal
   ledger, it does not poll a wallet/exchange balance automatically.
4. Double-check `MAX_PORTFOLIO_PCT_PER_TRADE`, `DAILY_LOSS_LIMIT_PCT`,
   `MAX_CONCURRENT_POSITIONS`, `MAX_TRADE_SIZE_USD`,
   `MAX_EXPOSURE_PER_TOKEN_PCT`, `MAX_TOTAL_EXPOSURE_PCT`,
   `MAX_CONSECUTIVE_LOSSES`, `MAX_DAILY_TRADES`, and `TRADE_COOLDOWN_SECONDS`
   for your real risk tolerance.
5. Set `SOLANA_PRIVATE_KEY` (or CEX API keys), `pip install -r
   requirements-live.txt`, then set `LIVE_TRADING=true` and restart.
6. Watch the first several live trades closely and confirm Telegram/Discord
   alerts are arriving.

## Backtesting

```bash
python scripts/run_backtest.py                          # synthetic bull market
python scripts/run_backtest.py --regime pump --seed 7    # synthetic pump
python scripts/run_backtest.py --csv data/ --symbol WIF --timeframe 15m   # your own OHLCV history
```

`app/backtesting/engine.py` walks a `CandleSeries` one bar at a time using
only `series.head(i+1)` at bar i — the same look-ahead guarantee
`CandleSeries.up_to()` gives — and reuses the exact same `RiskManager`
(position sizing) and `ExitManager` (trailing/break-even/partial/momentum/
trend/time exits) that live trading runs, both now dependency-injectable so
a backtest is deterministic and independent of `.env`. A signal confirmed
at bar i fills at a LATER bar's open (`execution_delay_bars`, default 1),
run through realistic per-side fees, slippage, and spread
(`BacktestConfig.fee_pct` / `slippage_pct` / `spread_pct`) — never the price
that produced the signal.

Entry requires the signal score ≥ `min_score_to_enter` (default 75) **and**
a tradeable, allowed market regime (`app/signals/market_regime.py`) **and**
higher-timeframe/volume/momentum each individually confirming, not just
outweighed in the composite **and** a minimum reward:risk
(`min_reward_risk`) computed from the nearest real resistance level above
price — a setup with a wall right overhead is rejected outright, not given
a fabricated target that always clears the bar. The stop is ATR-based by
default (`use_atr_stop`, `atr_multiple`), falling back to a fixed percent
only when ATR is unavailable.

Two things are intentionally NOT simulated: the rug-pull filter (no
historical scanner data to replay against — every backtested trade
implicitly assumes it would have passed screening), and cross-symbol
portfolio-level position sizing (the engine backtests one symbol's series
at a time; loop it over tokens yourself for "across tokens", and a single
long-enough series naturally spans multiple regimes for "across regimes" —
`BacktestTrade.market_regime` records which regime each trade actually
happened in).

`BacktestResult.stats` reports total return, trade count, win rate, avg
win/loss, profit factor, expectancy (in $ and R), max drawdown, Sharpe and
Sortino (annualized by observed trade frequency — a defensible
approximation for an irregular trade-return series, not a literal
fixed-interval Sharpe), avg R multiple, and longest winning/losing streak.
`BacktestResult.rejections` records every entry considered and why it
didn't qualify — the same "show why it didn't trade" transparency the
signal score and rug score give for live signals.

### Walk-forward testing

```bash
python scripts/run_backtest.py --walk-forward
python scripts/run_backtest.py --walk-forward --regime pump
```

A single backtest proves a config made money on one slice of history —
walk-forward testing is what checks it wasn't just tuned to that slice.
`app/backtesting/walk_forward.py` splits the series into train / validation
/ out-of-sample windows (`CandleSeries.split()`, chronological and
non-overlapping) and picks a config **using train performance only** —
validation and out-of-sample are backtested *after* that choice is locked
in, so neither can influence which config wins. `run_walk_forward()` takes
either a single `BacktestConfig` (the common case: "does this one set of
parameters generalize?") or a `dict`/`list` of candidates for a small
parameter search, selected by `selection_metric` (default `expectancy_r` —
average realized R per trade, comparable across configs regardless of
position sizing).

If validation performance falls to less than half of what train showed,
the result carries an explicit overfitting warning rather than silently
reporting the (misleading) train number as if it were confirmed. A
candidate with too few training trades to be statistically meaningful gets
its own warning too, rather than a lucky small sample being reported as if
it were a real edge.

### Strategy comparison

```bash
python scripts/compare_strategies.py
python scripts/compare_strategies.py --regime pump --candles 3000
```

`app/backtesting/strategy_comparison.py` runs five signal-weighting
profiles — `balanced` (the default), `momentum`, `breakout`,
`trend_following`, and `mean_reversion` — through the identical engine,
risk manager, and exit manager, each via its own walk-forward split, and
ranks them by **out-of-sample** expectancy, profit factor, and max
drawdown. Win rate never enters the ranking formula at all: a strategy that
wins often but with an ugly drawdown or a poor profit factor should not
outrank a steadier one just because it's "right" more often, which is
exactly the failure mode a win-rate-only comparison would produce. A
"strategy" here is a re-weighting of the same composite signal score
(`app/signals/scoring.py`'s factor weights reshuffled toward a different
style of setup — e.g. `mean_reversion` favors RSI/Bollinger/VWAP over
breakout structure), not four independently-coded trading systems — keeping
everything else identical is what makes the comparison isolate what the
signal weighting itself contributes.

## Strategy research — is this thing any good?

Everything above builds a bot that trades. This part answers whether it
*should*, and it is deliberately capable of saying no.

Open **`/research`** in the dashboard, or run:

```bash
python scripts/research.py report          # the validation verdict
python scripts/research.py distribution    # what the scorer actually produces
python scripts/research.py calibration     # does a higher score predict better?
python scripts/research.py funnel          # where discovered tokens die
```

### The validation report

Every question gets one of exactly three grades — `PASS`, `FAIL`, or
`INSUFFICIENT DATA` — and the third does most of the work early on. That is
the honest output: a report that resolved everything into pass or fail
would be inventing confidence, and the confidence it invents is always
"this looks fine", because a small sample of a random strategy usually
does.

**FAIL is a successful outcome.** Learning a strategy has no edge is worth
more than suspecting it might — it stops you spending months tuning
something that cannot work.

### Score calibration — the measurement that can falsify the engine

The bot records what **every scored candidate** did over the next 15m / 30m
/ 1h / 2h / 4h / 8h / 24h, **including the ones it rejected**, and groups
those outcomes by score bucket.

Following the rejects is the whole point. The bot only trades what it
already scored highly, so judging the score from trades alone asks "did the
setups we liked do well?" — a question a completely random score would also
pass. Letting the rejected 55s disagree is what turns the score from an
assertion into a measurement.

If 75-rated setups do not out-perform 65-rated ones, the report says so.

Unmeasurable outcomes are never zero-filled: a token whose feed went quiet
is recorded as unmeasurable with a reason, because calling it a 0% return
would turn a dead token into evidence.

### Threshold, ablation and robustness research

```bash
python scripts/research.py thresholds <mint>   # the 55 → 75 ladder
python scripts/research.py ablate     <mint>   # does each factor earn its weight?
python scripts/research.py sweep      <mint> min_score_to_enter 60,62.5,65,67.5,70
```

All three judge on **out-of-sample** performance over a chronological
split, penalised for the train-to-out-of-sample gap and for drawdown.
Ranking on in-sample P&L would just select whichever variant fitted the
training noise hardest.

- **thresholds** reports trade frequency, after-cost expectancy, profit
  factor, drawdown and the overfit gap for each value — and can conclude
  *"NO EDGE AT ANY THRESHOLD"*. More trades is never treated as better.
- **ablate** removes one scoring factor at a time. A factor can come back
  as *helping*, *no measurable effect*, or **HURTS**. "Trading bots
  commonly use it" is not evidence.
- **sweep** looks for a **plateau**, not a peak, and recommends the centre
  of the widest stable region. An isolated high value is reported as
  *"NO STABLE REGION — what an overfit parameter looks like."*

### Safety: the entry kill switch

Before every entry the bot asks whether its own state can be trusted:
does the cash ledger reconcile against the trade record, is the open book
structurally sound, has the price feed responded recently, and can enough
of the book be priced to size a trade off it.

It **fails closed** — a check that cannot run counts as a failure — and it
**never closes existing positions**. Halting entries is safe; liquidating
a book because a feed hiccuped turns a data problem into a real loss.

## Early Signal Engine — catching demand as it arrives

The technical score reads a chart that *already looks good*. By the time an
RSI cross, a MACD expansion and a volume spike all agree, the move is
usually well underway, and the entry that follows is a chase. The Early
Signal Engine (`app/early/`) is a **fourth, separate 0-100 score** that
asks a different question: is demand *arriving right now*?

It is separate on purpose. Blending it into the technical score would make
both unreadable and would let a strong early reading paper over a weak
setup. The four scores stay independent:

| Score | Question | Module |
|---|---|---|
| Technical | Does this setup look good? | `app/signals/scoring.py` |
| Security | Will this rug? | `app/rugcheck/risk_score.py` |
| Market quality | Can this actually be traded? | `app/signals/market_quality.py` |
| **Early opportunity** | **Is demand arriving?** | **`app/early/score.py`** |

### The nine early factors

Volume acceleration (0.18), transaction acceleration (0.14), buy pressure
(0.14), volume quality (0.10), liquidity quality (0.10), momentum
acceleration (0.10), price structure (0.09), breakout position (0.08),
relative volume (0.07).

**These weights are unvalidated priors.** They were chosen by reasoning
about microstructure, not measured from outcomes. Until
`python scripts/research.py early` says otherwise, they are a hypothesis.

Several are deliberately **non-monotonic**. Volume acceleration peaks
around 3x and *falls* above 8x, because a token already exploding is not an
early signal, it is a late one. Breakout position scores highest *at* the
range high and worst 15% above it — the move already happened.

### Anti-chase: Late Entry Risk

A second, separate 0-100 score (`app/early/late_entry.py`) that can **veto**
an entry but is never averaged into the early score. Averaging would let a
strong early reading cancel out a clear "you are too late" reading, which
is exactly the trade the whole system exists to avoid.

It classifies a candidate into a stage — EARLY, DEVELOPING, CONFIRMED,
LATE, OVEREXTENDED — and only the first three are enterable. **A candidate
that cannot be assessed is classified LATE**, because "we cannot tell"
must not become an invitation to enter.

### The WATCH state machine

```
DISCOVERED → WATCH → CONFIRMED → PAPER_BUY
          ↘ WATCH → FAILED / EXPIRED
```

"Promising" and "ready" are different facts, and a bot with only two states
has to collapse them — buy every promising candidate immediately (chasing)
or throw it away (missing the move). WATCH is the third state: the token
stays under observation, is re-scored every `WATCHLIST_INTERVAL_SECONDS`,
and enters only on confirmation that arrives *before* it becomes
overextended.

Entries that do not work out are **recorded, not deleted** — FAILED with
one of eight categories, or EXPIRED. A watchlist that quietly dropped its
disappointments would make the engine permanently unfalsifiable.

### Where it sits in the pipeline

Security runs **first**, before a single early feature is computed. An
excellent early signal can never override a failed security check — on a
security failure the engine returns before scoring anything at all, so
there is no early score to override with.

The engine's only power is to put a candidate the **technical** gate turned
down onto the WATCH list instead of discarding it. It never opens a
position directly: a CONFIRMED token is handed to the normal buy path,
which re-runs every existing risk, exposure and liquidity check.

### The switch that matters

```
EARLY_SIGNAL_MAY_TRADE=false     # default
```

False means the early engine can raise a token to WATCH and can **never**
open a position on its own. Turning it on before calibration shows that
higher early scores actually precede better outcomes is trading on a guess.

### Measuring whether any of it works

```bash
python scripts/research.py early              # calibration, lead time, false positives
python scripts/research.py early-ablate       # which early factors earn their weight
python scripts/research.py early-walkforward  # is the threshold stable out-of-sample?
python scripts/research.py modes  <mint>      # A: technical / B: early / C: both
```

or open **`/early`** in the dashboard.

- **Calibration** buckets every scored candidate — *including the rejected
  ones* — by early score and reports what actually happened next, with MFE
  and MAE so a +5% horizon return that first went −30% is visibly a
  stopped-out loss rather than a win.
- **Lead time** asks the question the engine exists for: of the tokens that
  went on to move +10%, +20%, +50%, what share did the engine flag
  *beforehand*?
- **False positives** groups every failed watchlist entry by cause. If more
  than half land in the residual `score_decayed` bucket, the taxonomy is
  missing a category — that is reported, not smoothed over.
- **Ablation** re-scores the features **stored at signal time** with one
  factor's weight zeroed. It cannot be done as a candle backtest: four of
  the nine factors come from differencing market snapshots, which candles
  do not carry, so a backtest would silently mark them unavailable and
  then report that removing them changes nothing. The verdict comes from
  rank correlation, not bucket separation — bucket edges move when a score
  merely shifts.
- **Walk-forward** refits the entry threshold on each training window and
  grades it on the next, so a threshold chosen on noise is visible as a
  threshold that will not sit still.

Every one of these reports **INSUFFICIENT DATA** until it has enough
measured outcomes, and that is the state a fresh install is in. An absence
of evidence is printed as an absence of evidence.

### Current status

**Built, not validated.** The engine runs and is covered by tests, but it
has produced no measured outcomes yet, so nothing here shows that an early
edge exists. The full accounting is in
[`docs/EARLY_SIGNAL_REPORT.md`](docs/EARLY_SIGNAL_REPORT.md) — including
the items that are honestly INSUFFICIENT DATA and the two defects that
made the engine unreachable until integration tests caught them.

### Collecting the data — the actual next step

Nothing in this section is validated, and none of it can be until the bot
has run. The way to change that is to leave it running in paper mode and
let the dataset build. **Do not tune anything first** — every threshold in
here is a prior, and adjusting a prior to make early results look better
is how a system stops being testable.

```bash
python scripts/init_db.py
uvicorn app.main:app --port 8000          # leave it running
```

Requires outbound access to `api.dexscreener.com` (discovery) and
`api.geckoterminal.com` (candles). Both are read-only public endpoints,
and no key, wallet or private key is involved.

Then check progress whenever you like:

```bash
python scripts/research.py readiness
```

```
  early calibration      [##########..............]     13/30  ~26h at the current rate
  lead time              [######..................]      8/30  ~44h at the current rate
  early ablation         [#####...................]      9/40  ~3.4 days at the current rate
```

Each row turns to `READY` exactly when the tool beside it stops saying
INSUFFICIENT DATA — the report counts what the tool itself will accept, so
it cannot promise an answer that then fails to appear. The ETAs
extrapolate the rate observed so far; a quiet weekend halves it. They are
for deciding when to look again, not dates.

The sample floors are not adjustable from the report, deliberately. Below
them the answers are noise, and a shortcut past them would be the most
damaging thing in this repository.

### What the engine genuinely cannot see

Named explicitly in `app/early/features.py` rather than approximated:

| Wanted | Why it is unavailable |
|---|---|
| Unique / new / repeat buyers | Needs a wallet-level indexer. A transaction count is not a participant count. |
| Wallet concentration | Needs holder distribution beyond the top-10 the rug engine reads. |
| Order-book depth | AMMs have no order book. Transaction flow is a proxy and is labelled as one. |
| Social activity | No trustworthy provider configured. |

A factor with no data is marked unavailable and counted against the
missing-data budget; above `MAX_UNAVAILABLE_WEIGHT` (40%) the whole score
is marked unreliable. It is never silently replaced with a zero or an
average.

## Two more things between a signal and a position

Both of these were built, tested, and then left disconnected — the modules
passed their unit tests while protecting nothing in the running bot.

### Data cross-check — when two sources disagree, don't trade

The DexScreener snapshot and the security scanner both report a pool's
liquidity, and both readings are already in hand by the time the buy path
needs one. When they disagree beyond tolerance, **at least one is wrong and
there is no way to tell which.** The dangerous instinct is to pick the
convenient number — the higher liquidity permits a bigger position — which
is selection bias with extra steps, and it biases every result toward
trading more. So the trade is skipped and the disagreement logged with both
values.

When they agree, sizing still uses the **more conservative** figure, never
an average. Agreement within 30% leaves a real gap, and the thinner pool is
what decides the cost of getting out.

Liquidity only, deliberately. The only second "price" available is the one
the alert fired at, which is not another provider's reading of the same
instant — a memecoin routinely moves more than any sane tolerance between
the alert and the evaluation, so comparing them would reject constantly on
normal movement. That is a staleness check, and staleness has its own gate.

### Correlation risk — five memecoins are not five positions

The per-token and total exposure caps are both satisfied by a book that is
really *one bet at five times the intended size*. This caps the bet.

The gate and the report deliberately disagree about unmeasured pairs:

| | unmeasured pair | why |
|---|---|---|
| **Report** (dashboard) | counted as fully correlated | a book whose correlations are unknown must not be *described* as diversified |
| **Gate** (blocks a trade) | ignored | on a fresh install every pair is unmeasured, so blocking would refuse a second position forever — and never collect the data proving otherwise |

That is not the check being softened. Describing risk should assume the
worst; refusing a trade needs a reason. A risk check that halts all trading
before measuring anything is indistinguishable from a broken bot, and the
sensible-looking response is to switch it off — which is worse than not
having it.

Return history comes from the position monitor, which already fetches a
price for every open position on every pass and used to discard it. It is
the only price history the bot has for tokens it *holds*: the early engine
watches candidates, and a candidate stops being watched the moment it
becomes a position.

```
CORRELATION_RISK_ENABLED=true
MAX_CORRELATED_CLUSTER_PCT=0.30
CROSS_CHECK_ENABLED=true
CROSS_CHECK_LIQUIDITY_TOLERANCE=0.30
```

## Every outbound call goes through one wrapper

`app/services/http.py` owns retry, backoff, `Retry-After`, and API-health
recording. Nine modules use it and nothing else opens its own client —
enforced by a test, because a raw client gets no backoff (so a rate limit
on a *security* scanner becomes an immediate failure) and never registers
with the health tracker (so the kill switch cannot tell a dead provider
from a quiet market). Both are invisible until the day they matter.

POST is not simply GET with a body. A GET can always be repeated; a POST
that times out may already have been processed, and repeating it can submit
the same thing twice:

| | retries on |
|---|---|
| `get_json` | any transient status, any network error |
| `post_json(idempotent=True)` | same — for endpoints that only *build* or *read* (a quote, an `eth_call`, an unsigned transaction) |
| `post_json()` **default** | **429 only** — the one response that says the server definitely did not process the request |

A failed security lookup raises `LookupFailed` rather than returning empty.
"We checked and found nothing" and "we could not check" are different facts
about a security screen, and collapsing them would let a provider outage
read as a clean bill of health.

## Configuration reference

Every variable is documented inline in [`.env.example`](.env.example) —
copy it to `.env` and adjust. Highlights not covered above:

- `SLIPPAGE_BPS` — max slippage tolerance passed to the execution backend.
- `MAX_GAS_PRICE_GWEI` — reserved for the EVM backend's gas guard once
  execution is completed.
- `PRICE_POLL_INTERVAL_SECONDS` — how often the background loop checks
  open positions for SL/TP.
- `DATABASE_URL` — swap the default SQLite for Postgres by setting e.g.
  `postgresql+psycopg2://bot:password@localhost:5432/memecoin_bot` (and
  `pip install psycopg2-binary`).

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Covers risk-manager sizing/clamping/halting, the rug-check filter's pure
evaluation logic against crafted GoPlus-shaped fixtures, and an end-to-end
webhook → paper trade → close round trip with all outbound network calls
mocked.

## Disclaimer

This software is provided for educational and research purposes. It is not
financial advice. Memecoin trading is extremely high risk — tokens can lose
all value rapidly, liquidity can vanish, and even tokens that pass every
automated check in this project (ownership, mint authority, liquidity lock,
holder concentration, honeypot simulation) can still be scams, exploited, or
simply collapse in price. The rug-check filter reduces exposure to the most
common scam patterns; it cannot guarantee safety. You are solely responsible
for any funds you configure this bot to trade with, for reviewing and
understanding the code before running it live, and for complying with the
laws and regulations that apply to you. Start in paper trading mode, review
results thoroughly, and never trade with money you cannot afford to lose.
