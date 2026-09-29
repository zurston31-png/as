# Trading signal bot

A rules-first intraday signal bot with an AI confirmation layer, a risk engine,
and a local dashboard that tells you when your conditions line up — with a live
chat box you can ask "why did you skip that one?".

```
market data
   -> freshness check         stale feed => no decision at all
   -> indicator context       closed bars only, as of decision_ts
   -> strategy rules          deterministic; is there a setup?
   -> duplicate gate          one setup, one signal
   -> AI confirmation         can only veto; failure counts as WAIT
   -> position reconciliation book vs. broker
   -> risk engine             limits, then explicit sizing
   -> LONG / SHORT / NO TRADE
   -> audit record, screen alert, sound, paper fill

screen capture ──────────────► backup: corroborate or contradict
```

The important design decision is in the arrows: **the model never reads the
chart to find a trade.** Structured data already carries the exact numbers a
screenshot only approximates, so the rules run on numbers, and the AI layer is
a second opinion on a setup the rules already found. It can say *wait* or
*reject*; it cannot invent a trade. Screen vision sits off to the side as a
backup that notices a stale feed or the wrong symbol on screen.

## Quick start

```bash
pip install -r requirements.txt
python -m tradebot doctor          # what's installed, what's configured, how the strategy parses
python -m tradebot serve --open    # dashboard on http://127.0.0.1:8787
```

The first run uses a synthetic feed, so you get a moving chart, real signals and
paper fills without any data setup. It is random data — it exercises the
pipeline, it is not a market.

For the AI layers: `export ANTHROPIC_API_KEY=sk-ant-...` (or `ant auth login`).
Without a key the bot runs rules-only and says so in the dashboard header.

## The strategy is configuration, not code

`config/config.yaml` holds the whole strategy: which rules run, whether each is
required or advisory, every period and threshold, and the target geometry.
Changing RSI to 45–65, making volume advisory, or moving TP2 to 3R is a config
edit.

```yaml
strategy:
  name: "ema_vwap_rsi"
  min_score: 0.75            # weighted share of rules that must pass
  rules:
    - {rule: ema_stack,           mode: required, params: {fast: 9, slow: 21}}
    - {rule: ema_cross_fresh,     mode: required, params: {fast: 9, slow: 21, max_bars: 5}}
    - {rule: vwap_position,       mode: required}
    - {rule: rsi_window,          mode: required, params: {period: 14, long_min: 45, long_max: 65}}
    - {rule: market_structure,    mode: required, params: {width: 2, lookback: 60}}
    - {rule: liquidity_sweep,     mode: required, params: {min_wick_ratio: 0.5}}
    - {rule: volume_confirmation, mode: advisory, params: {multiple: 1.1}}
  entry:
    stop: {method: swing_or_atr, swing_buffer_atr: 0.25, atr_period: 14}
    tp1_r: 2.0
    tp2_r: 4.0
```

`mode` is `required` (must pass), `advisory` (only moves the score) or
`disabled`. A bad config fails at startup with a specific message, never
mid-session. `python -m tradebot doctor` prints the parsed result.

Rules available out of the box: `ema_stack`, `ema_cross_fresh`,
`vwap_position`, `rsi_window`, `market_structure`, `structure_break`,
`liquidity_sweep`, `volume_confirmation`, `candle_direction`, `atr_range`.
Adding your own is a function in `tradebot/strategy/rules.py` plus a dict
entry; it is then usable from YAML by name.

### Why isn't it firing?

Strict conjunctions are much stricter than they look. `tradebot rules` shows
each rule's own pass rate and then the funnel:

```
$ python -m tradebot rules --csv data/demo.csv --ignore-session

  rule                    mode         passes    rate
  ema_cross_fresh         required        562    7.1%
  liquidity_sweep         required        487    6.2%
  ...
  Funnel - bars surviving each required rule, in order
  ema_stack                    3936  50.00%   (kept 50% of the previous step)
  ema_cross_fresh               525   6.67%   (kept 13% of the previous step)
  vwap_position                 237   3.01%   (kept 45% of the previous step)
  rsi_window                    171   2.17%   (kept 72% of the previous step)
  market_structure              135   1.71%   (kept 79% of the previous step)
  liquidity_sweep                 5   0.06%   (kept  4% of the previous step)
```

Requiring both a fresh EMA cross *and* a liquidity sweep takes this strategy
from 135 setups to 5. That is a choice, not a bug — but it should be a choice
you made on purpose.

### Is that rule earning its strictness?

Don't loosen a rule just because you got one signal. `tradebot ablate-rules`
runs the stack with each required rule removed in turn, over identical data:

```bash
python -m tradebot ablate-rules --csv data/demo.csv --mode leave-one-out
```

```
  variant                     qualif  ambig signals  trades   win%   expectancy     PF   maxDD%  TP1%  TP2%
  full stack                     ...    ...     ...     ...    ...          ...    ...      ...   ...   ...
  - liquidity_sweep              ...    ...     ...     ...    ...          ...    ...      ...   ...   ...
```

Three setup columns, and the differences matter:

- **`qualif`** — side-evaluations that passed, before anything else looked at
  them. Relaxing a rule can only raise this, so it's the clean measure of
  strictness.
- **`ambig`** — bars where *both* directions qualified at once. The engine
  treats those as no-setup, so a rule set loose enough to be ambiguous produces
  *fewer* usable setups, not more. If dropping a rule spikes this column, that
  rule was supplying the directional decision (`ema_stack` usually is).
- **`signals`** — what survived the duplicate and position gates. Those couple
  decisions across time: a looser rule set enters earlier, holds the book
  longer, and can admit *fewer* signals than a stricter one.

Judge strictness on `qualif`, direction on `ambig`, results on `trades`. Both
non-monotonic effects are real and each has a test pinning it down — they're
why the naive "fewer rules means more setups" reading is wrong.

`--mode incremental` instead stacks the rules up one at a time in config order,
which is the view for "what did adding volume actually do?".

The report ends with a plain-language reading per rule, and explicitly flags
when a comparison rests on too few trades to mean anything:

```
  liquidity_sweep    removes   130 qualifying bars   expectancy +0.31R with / +0.08R without
                     keeping it improves expectancy; dropping it adds 42 trades
                     (too few trades to trust - treat as a hint, not a result)
```

## Market data

| `feed.source` | what it is | when to use it |
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
Alerts without a valid secret are rejected. Alerts carrying only a price are
aggregated into bars locally.

**Freshness is checked continuously**, not only when the screen backup wakes up.
Two clocks are tracked, because they fail differently: the age of the newest
bar (a feed that stopped producing) and time since anything arrived at all (a
socket that is open but dead). Past `feed.max_stale_bars` intervals, every
signal is blocked and the dashboard header turns red. Outside session hours the
check is suspended, since both numbers are expected to be large.

## Signals

```
LONG  MNQ1!  5m   [14:35:00]
  Entry : 25420
  Stop  : 25360   (60 away)
  TP1   : 25540   (2R)
  TP2   : 25680   (4R)
  Size  : 2   risking 0.50% ($250)
  AI    : confirm 0.78 - fresh cross holding above VWAP, volume 1.4x
```

**One setup produces exactly one signal.** The conditions behind a setup stay
true for several bars, so a side that has fired is disarmed until the setup
stops qualifying (`strategy.signal.rearm_on_invalidation`) or a bar count
passes (`rearm_bars`). `require_flat` keeps a new signal from firing while a
position is open, and reversals are off unless you turn them on.

## Risk engine

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

Sizing is explicit, and every intermediate is written to the audit log so a
position size can always be re-derived:

```
risk_dollars           = equity × risk_percent
stop_risk_per_contract = |entry − stop| × point_value
contracts              = floor(risk_dollars / stop_risk_per_contract)
```

If `contracts` reaches zero the result is a rejection with
`reason_code = "position_size_below_minimum"`. **Risk is never increased to make
a position fit.** That is exactly what happens if you try to risk 0.5% of $25k
on a full-size NQ contract: $125 of budget against $1,200 of per-contract risk.
It's why the shipped config uses MNQ at `point_value: 2.0`. **Set
`market.point_value` and `qty_step` to your actual instrument** — every risk
number depends on them.

Every refusal carries a stable `reason_code` (`kill_switch_active`,
`cooldown_after_losses`, `reward_to_risk_below_minimum`, …) so rejections can be
counted and compared across runs rather than string-matched.

**Position state is reconciled before every sizing decision**, even in paper
mode. The book records what the risk engine believes is open; the broker holds
what actually is. If they disagree the kill switch trips and signalling stops —
sizing against a fiction is worse than not trading. The same check is what you'd
run against a live broker's own position records.

## Audit log

Every decision, taken or blocked, gets one append-only JSONL record carrying
the whole chain: the trigger candle, the decision timestamp, every rule's
output, the AI verdict, the full sizing arithmetic, the risk decision, the
position state at the time, and the config fingerprint.

"Immutable" here means tamper-evident, not write-protected: each record carries
a sequence number and the SHA-256 of the previous one, so editing or deleting
anything mid-file breaks the chain from that point and `verify` says where.

```bash
python -m tradebot audit verify    # walk the chain
python -m tradebot audit tail      # last 20 decisions, one line each
```

## Proving the strategy before adding more AI

Do these in order. Each is a real gate.

### 1. Backtest

```bash
python -m tradebot gen-data --out data/demo.csv --bars 20000
python -m tradebot backtest --csv data/demo.csv
```

The backtest **drives the live orchestrator**, bar by bar, rather than
reimplementing the pipeline — a backtest that runs different code from the live
loop measures something you will never trade. Every gate applies. There's a
test asserting the two produce identical trades over the same bars.

**No lookahead is structural**, not checked after the fact. A decision's
context is built from closed bars at or before that bar's timestamp; a forming
bar is never visible to a rule; out-of-order bars raise rather than being
silently used; and the broker marks existing positions against a bar *before*
any new entry is opened on it, so a position never fills against the bar that
created it. Four tests pin this down, including one that proves an indicator at
bar N doesn't change when bar N+1 arrives.

**Costs are pessimistic on purpose.** Entry takes slippage against you; stops
take their own, larger slippage, because a stop is a market order into the move
that triggered it; targets fill at their limit; commission is charged **per
side**, so every round turn pays twice; and a bar containing both your stop and
your target is scored as a stop.

### 2. Walk-forward, with a sealed holdout

```bash
python -m tradebot walkforward --csv data/demo.csv
```

Consecutive folds, each reported separately — a strategy whose edge lives in
one fold is fitted to that fold, and a single aggregate number hides exactly
that. The most recent 20% is held back and **not run** unless you pass
`--holdout`, and every such run is recorded with the config fingerprint at the
time:

```
  4/4 folds profitable · mean +0.240R · sd 0.236 · worst +0.016R

  Holdout (sealed)
  ! This holdout has now been run 2 times across 2 different configurations.
    It is no longer a clean out-of-sample test - each look leaks information
    into your choices. Treat the number below as optimistic and get fresh data.
```

That ledger is the honest part. Nothing can stop you rerunning the holdout, but
after the third look with three different configs it isn't a holdout any more,
and the report says so.

Worth seeing on the demo data: 4/4 folds profitable at +0.24R, then **−0.22R on
the holdout**. That is the failure mode this whole section exists to catch.

### 3. Does the AI layer earn its place?

```bash
python -m tradebot ablate --csv data/demo.csv --ai
```

Four arms over identical data:

| arm | what it isolates |
|---|---|
| `rules` | the rules alone, sizing only, no daily limits |
| `rules+ai` | the same, plus the AI veto |
| `rules+ai+risk` | the full production stack |
| `random#1..N` | arm 3 with coin flips vetoing at the *same rate* the model did |

The random control is the arm that matters. A veto layer reduces trade count,
and reducing trade count changes drawdown and expectancy on its own — so "the
AI improved the numbers" proves nothing until you show that vetoing the same
proportion *at random* doesn't improve them equally. If `rules+ai` doesn't beat
the control, the model is contributing no information; it's just trading less,
and a lower `max_trades_per_day` would do that for free.

Verdicts are cached on disk by decision identity, so every arm sees identical
model output (a difference between arms is the arm, not model variance) and
re-running is free. Without `--ai` and with an empty cache the report refuses to
draw a conclusion rather than claiming the AI doesn't help. `--veto-rate 0.3`
asks "what would vetoing 30% at random do?" without spending anything.

### 4. Live market paper trading

Point `feed.source` at your real data, leave `execution.mode: paper`. Weeks,
not days. Compare live paper to the backtest — a big gap means your backtest is
lying to you.

### 5. Alerts only

`execution.mode: alerts_only` — the bot signals and counts against your risk
limits, you place the trades. This is where you find out whether you can
actually follow the thing.

### 6. Broker integration

Deliberately not implemented. `tradebot/execution/base.py:LiveBroker` raises
with an explanation. Filling it in means order placement, fill reconciliation,
position sync against the broker's own records, and broker-side stops — not
just "send an order". Don't shortcut it by forwarding to the paper broker.

Keep live execution disabled until verified fills, position sync, duplicate
prevention and broker-side stops are all in place.

## Computer integration

`python -m tradebot serve` runs a local web app: live candles with overlays
derived from your configured rules, the signal's levels drawn on the chart, the
full rule checklist (so you see *which* condition failed), a system-health panel
(feed age, armed sides, duplicates blocked, reconciliation, audit count), risk
counters, open positions, and the chat box. Signals arrive over a websocket and
fire a desktop notification and a sound.

The chat box has the bot's live state attached to every message:

- *"Why was that skipped?"* → names the rules that failed and what they'd need to be
- *"Walk me through the sizing on the last signal."*
- *"How much risk budget is left today?"*

It has no execution tools and cannot place or modify an order.

## Screen vision (the backup layer)

Off by default. `pip install mss pillow`, then:

```yaml
vision:
  enabled: true
  mode: backup                                        # or "always"
  region: {left: 0, top: 0, width: 1920, height: 1080}
```

In `backup` mode a screenshot is only read when the feed is lagging. The
reading is structured (symbol, timeframe, last price, visible levels, patterns)
and is used to *contradict* the feed: a different symbol on screen, or a price
more than `price_tolerance_pct` away, blocks the signal. It never originates a
trade.

## Commands

```
tradebot doctor                      dependencies, config, parsed strategy, audit chain
tradebot serve                       dashboard + live loop
tradebot rules        --csv data.csv per-rule pass rates and the funnel
tradebot ablate-rules --csv data.csv which rule removes the setups, and is it earning it?
tradebot backtest    --csv data.csv  one period
tradebot walkforward --csv data.csv  folds + a sealed holdout
tradebot ablate      --csv data.csv  is the AI layer worth it?
tradebot audit verify|tail|dump      the decision log
tradebot gen-data                    synthetic OHLCV for testing
```

## Layout

```
tradebot/
  models.py          Candle, Series, TradeSignal, Position, verdicts
  config.py          typed config + YAML loading
  indicators.py      EMA, RSI, ATR, session VWAP (no numpy)
  structure.py       swings, trend, BOS/CHoCH, liquidity sweeps
  monitor.py         continuous feed freshness
  audit.py           append-only, hash-chained decision log
  strategy/
    spec.py          the strategy as data, validated at startup
    context.py       decision-time view; lazy features, no lookahead
    rules.py         the conditions, parameterised from config
    engine.py        evaluation, duplicate gate, entry geometry
  ai/                confirmation layer, chat session, chart vision
  risk/              limits, explicit sizing, kill switch, persisted state
  execution/         paper broker, position book + reconciliation, live stub
  data/              webhook, poll, replay, synthetic sources
  backtest/          runner, metrics, walk-forward, rule + AI ablation, rule stats
  app/               FastAPI server and the dashboard
  orchestrator.py    the pipeline that wires it together
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

242 tests covering the indicator maths, structure detection, strategy-config
validation, the duplicate gate, no-lookahead (four angles), every risk limit
and reject code, the sizing formula, cost modelling, position reconciliation,
the audit chain under tampering, feed freshness, the data parsers,
walk-forward, both ablation harnesses, the HTTP and websocket surface, and the
AI layers with the network stubbed — including that every way a model call can
fail degrades to "no trade" rather than an exception in the trading loop.

Two of them pin down counter-intuitive behaviour that a naive reading gets
wrong: dropping a rule can *reduce* admitted signals (the gates couple
decisions across time), and it can reduce usable setups outright (a looser
stack can qualify in both directions at once). Both are asserted rather than
assumed, so a future refactor can't quietly "fix" them into being monotonic.

## What this is not

It is not a strategy that is known to make money — the rules are a starting
point and the defaults are examples. It doesn't place orders. And nothing here
is financial advice; you own every trade it suggests.
