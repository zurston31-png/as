# Trading signal bot

A rules-first intraday signal bot with an AI confirmation layer, a risk engine,
and a local dashboard that tells you when your conditions line up — with a live
chat box you can ask "why did you skip that one?".

```
TradingView / broker data  ──►  strategy engine  ──►  AI confirmation  ──►  risk engine  ──►  LONG / SHORT / NO TRADE
                                (deterministic)        (can only veto)      (sizes + limits)      └─► screen alert, sound, paper fill
        screen capture ─────────────────────────────►  (backup: corroborate or contradict)
```

The important design decision is the one in the arrows: **the model never reads
the chart to find a trade.** Structured data already carries the exact numbers
that a screenshot only approximates, so the rules run on numbers, and the AI
layer is a second opinion on a setup the rules already found. It can say *wait*
or *reject*; it cannot invent a trade. Screen vision sits off to the side as a
backup — it exists to notice that your feed has gone stale or that the chart on
screen isn't the symbol the bot thinks it's trading.

## Quick start

```bash
pip install -r requirements.txt
python -m tradebot doctor          # what's installed, what's configured
python -m tradebot serve --open    # dashboard on http://127.0.0.1:8787
```

The first run uses a synthetic feed, so you get a moving chart, real signals and
paper fills without any data setup. It is random data — it is there to exercise
the pipeline, not to be traded.

To use the AI layers:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or: ant auth login
```

Without a key the bot runs rules-only and says so in the dashboard header.

## The five pieces

### 1. Market data

Four sources, set with `feed.source`:

| source | what it is | when to use it |
|---|---|---|
| `webhook` | TradingView alerts POSTed to `/webhook/tradingview` | **the recommended live path** |
| `poll` | any REST endpoint returning a price or OHLCV rows | broker/exchange APIs |
| `replay` | a CSV, streamed bar by bar | dry runs, debugging rules |
| `synthetic` | a random walk | first run, UI work |

For TradingView, set the alert message to JSON and point its webhook at the bot:

```json
{
  "secret": "your-shared-secret",
  "symbol": "{{ticker}}", "timeframe": "{{interval}}", "time": "{{timenow}}",
  "open": {{open}}, "high": {{high}}, "low": {{low}},
  "close": {{close}}, "volume": {{volume}}, "closed": true
}
```

Set the same string in `feed.webhook_secret` (or `TRADEBOT_WEBHOOK_SECRET`).
Alerts without a valid secret are rejected. TradingView needs to reach the port,
so for live use put it behind a tunnel rather than binding `0.0.0.0`.

Alerts that carry only a price are aggregated into bars locally, so a plain
`{"secret": "...", "close": {{close}}}` alert also works.

### 2. Strategy engine

`config/config.yaml` → `strategy`. The shipped preset is the stack from the
brief:

| rule | what it checks |
|---|---|
| `ema_stack` | 9 EMA on the correct side of the 21 |
| `ema_cross_fresh` | the cross happened within the last N bars |
| `vwap_position` | price on the correct side of session VWAP |
| `rsi_window` | momentum confirms but isn't exhausted (50–72 long) |
| `market_structure` | swing structure isn't against the trade |
| `liquidity_sweep` | stops run and rejected on the far side *(optional)* |
| `volume_confirmation` | bar volume beats its 20-bar average |
| `candle_direction` | the trigger bar itself agrees |

Rules in `strategy.optional_rules` contribute to the score but never block on
their own; everything else is a hard requirement, and the weighted score must
also clear `strategy.min_score`. Two other presets ship: `trend_pullback`
(no fresh cross required, fires more often) and `sweep_reversal`.

Writing your own rule is a function and a dict entry — see
`tradebot/strategy/rules.py`.

**Levels.** Entry is the close of the trigger bar. The stop goes behind the
protecting swing padded by a fraction of ATR, falling back to a pure ATR stop
when there's no usable swing. TP1 and TP2 are multiples of that risk
(2R and 4R by default).

### 3. Trade signal

```
LONG  MNQ1!  5m   [14:35:00]
  Entry : 25420
  Stop  : 25360   (60 away)
  TP1   : 25540   (2R)
  TP2   : 25680   (4R)
  Size  : 2   risking 0.50% ($250)
  AI    : confirm 0.78 - fresh cross holding above VWAP, volume 1.4x
```

### 4. Computer integration

`python -m tradebot serve` runs a local web app: live candles with EMA/VWAP
overlays and the signal's levels drawn on the chart, the full rule checklist
(so you can see *which* condition failed), risk counters, open positions, and
the chat box. New signals arrive over a websocket and fire a desktop
notification and a sound.

The chat box has the bot's live state attached to every message, so it can
answer from the actual numbers:

- *"Why was that skipped?"* → names the rules that failed and what they'd need to be
- *"How much risk budget is left today?"*
- *"What would need to change for a long here?"*

It has no execution tools and cannot place or modify an order.

### 5. Risk engine

Every limit from the brief, enforced before anything is marked actionable:

| setting | default | |
|---|---|---|
| `risk_per_trade_pct` | 0.5% | drives position sizing |
| `max_trades_per_day` | 5 | |
| `max_daily_loss_pct` | 2% | trips the kill switch |
| `max_open_positions` | 1 | |
| `max_consecutive_losses` | 2 | then a cooldown |
| `cooldown_minutes_after_loss` | 30 | |
| `require_stop` | true | a signal without a stop is refused |
| `min_rr` | 1.5 | refuses setups whose TP1 isn't worth the risk |
| `kill_switch` | false | the big red button, also in the dashboard |

Sizing is `(equity × risk%) ÷ (stop distance × point value)`, rounded **down** to
your contract step. If that rounds to zero the trade is refused rather than
quietly upsized — which is exactly what happens if you try to risk 0.5% of
$25k on a full-size NQ contract. That's a real constraint, not a bug; it's why
the shipped config uses MNQ at `point_value: 2.0`. **Set `market.point_value`
and `qty_step` to match your actual instrument** — every risk number depends on
them.

State lives in `data/risk_state.json` and survives restarts, so a kill switch
you flipped stays flipped. Daily counters reset on the session date; an
automatic halt clears with the new day, one you flipped by hand does not.

## The staged rollout

Do these in order. Each stage is a real gate — if a stage doesn't look good,
the next one won't fix it.

**1. Historical backtest.**

```bash
python -m tradebot gen-data --out data/demo.csv --bars 4000   # or bring your own CSV
python -m tradebot backtest --csv data/demo.csv
```

Reports win rate, expectancy in R, profit factor, max drawdown, worst losing
streak, and a breakdown of why setups were blocked. It runs the same strategy,
sizing and risk code the live loop runs. Fills are pessimistic on purpose: entry
takes slippage against you, and any bar containing both your stop and your
target is scored as a stop.

The AI layer is off here by default — a backtest that calls a model on every bar
is slow and expensive, and you're trying to measure the *rules*. `--ai` turns it
on for a short window when you want to sanity check the confirmation layer.

On the synthetic demo data you should expect roughly break-even-to-negative
expectancy. A random walk has no edge, and seeing that is the point: if your
real data doesn't look meaningfully better than the random data, you don't have
a strategy yet.

**2. Live market paper trading.** Point `feed.source` at your real data and
leave `execution.mode: paper`. Let it run for weeks, not days. Compare the live
paper results to the backtest — a big gap means your backtest is lying to you.

**3. Alerts only.** `execution.mode: alerts_only` — the bot signals and counts
against your risk limits, and you place the trades. This is where you find out
whether you can actually follow the thing.

**4. Broker integration.** Deliberately not implemented.
`tradebot/execution/base.py:LiveBroker` raises with an explanation. Filling it
in means order placement, fill reconciliation *and* position sync against the
broker's own records — not just "send an order". Don't shortcut it by forwarding
to the paper broker.

## Screen vision (the backup layer)

Off by default. `pip install mss pillow`, then:

```yaml
vision:
  enabled: true
  mode: backup                                        # or "always"
  region: {left: 0, top: 0, width: 1920, height: 1080}
```

In `backup` mode a screenshot is only read when the structured feed has gone
quiet for `stale_feed_seconds`. The reading is structured (symbol, timeframe,
last price, visible levels, patterns) and is used to *contradict* the feed: if
the screen shows a different symbol, or a price more than
`price_tolerance_pct` away from the feed's, the signal is blocked. It is never
used to originate a trade.

## Configuration

`config/config.yaml`, with a few environment overrides (`TRADEBOT_SYMBOL`,
`TRADEBOT_TIMEFRAME`, `TRADEBOT_PORT`, `TRADEBOT_WEBHOOK_SECRET`,
`TRADEBOT_AI_DISABLED`, `TRADEBOT_KILL_SWITCH`). Every field and its default is
in `tradebot/config.py`.

AI settings worth knowing:

```yaml
ai:
  model: "claude-opus-5-5"
  effort: "medium"              # low | medium | high | xhigh | max
  required: false               # true => an unavailable AI layer means NO TRADE
  confirm_min_confidence: 0.55  # a "confirm" below this is treated as a wait
```

`required: false` fails open to rules-only when the API is unreachable;
`required: true` fails closed. Neither one lets the model create a trade.

## Layout

```
tradebot/
  models.py          Candle, Series, TradeSignal, Position, verdicts
  config.py          typed config + YAML loading
  indicators.py      EMA, RSI, ATR, session VWAP (no numpy)
  structure.py       swings, trend, BOS/CHoCH, liquidity sweeps
  strategy/          context builder, rules, presets, the engine
  ai/                confirmation layer, chat session, chart vision
  risk/              limits, sizing, kill switch, persisted state
  execution/         paper broker, live broker stub
  data/              webhook, poll, replay, synthetic sources
  backtest/          runner + metrics
  app/               FastAPI server and the dashboard
  orchestrator.py    the pipeline that wires it together
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

110 tests covering the indicator maths, structure detection, every risk limit,
paper-broker fill logic, the data parsers, the HTTP and websocket surface, and
the AI layers with the network stubbed — including that every way a model call
can fail degrades to "no trade" or "rules only" rather than an exception in the
trading loop.

## What this is not

It is not a strategy that is known to make money — the rules are a starting
point and the defaults are examples. It doesn't place orders. And nothing here
is financial advice; you own every trade it suggests.
