"""Central configuration, loaded from environment variables / .env.

Everything a deployer needs to tune (risk limits, rug-check thresholds,
polling intervals, API endpoints) lives here so the rest of the codebase
never reads os.environ directly.
"""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Core ---
    APP_ENV: str = "development"
    LIVE_TRADING: bool = False
    # A second, explicit gate on top of LIVE_TRADING for execution backends
    # whose sign-and-submit path has never been exercised against a funded
    # wallet from this codebase's own tests (Jupiter, the EVM/1inch backend)
    # - the math around them is unit-tested, the actual on-chain submission
    # cannot be without moving real money. Requiring a second flag means
    # arming one of those paths is a deliberate second decision, not a side
    # effect of flipping LIVE_TRADING alone. Not required for EXECUTION_BACKEND=cex
    # (ccxt is a mature, widely-used library, not code written for this project).
    LIVE_EXECUTION_ACKNOWLEDGED: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./data/memecoin_bot.db"

    # --- database snapshots (app/backup.py) ---
    # The research dataset is the only thing here that cannot be rebuilt.
    # Code is in git; a month of observed forward returns is not.
    BACKUP_ENABLED: bool = True
    # WHERE THIS POINTS IS THE WHOLE POINT. A backup written next to the
    # database dies with it on any host that replaces the filesystem on
    # deploy. Point it at a mounted volume; the app warns at startup when
    # it looks like it is on the same disk it is protecting against.
    BACKUP_DIR: str = "./backups"
    BACKUP_INTERVAL_MINUTES: int = 60
    BACKUP_KEEP: int = 24
    # Restore automatically when the database is missing or has no tables.
    # Never over a database that already holds rows - see app/backup.py.
    BACKUP_RESTORE_ON_EMPTY: bool = True
    # Silences the startup warning about snapshots sharing a disk with the
    # database. Set it only if that disk genuinely survives a redeploy - a
    # VPS, a bare-metal box, a mounted volume. The warning is deliberately
    # noisy because a false alarm costs a log line and a miss costs the
    # whole dataset.
    BACKUP_DIR_IS_PERSISTENT: bool = False
    LOG_LEVEL: str = "INFO"

    # --- Webhook ---
    WEBHOOK_SECRET: str = "changeme-generate-a-long-random-string"
    WEBHOOK_PATH: str = "/webhook/tradingview"
    SYMBOLS_WATCHLIST: str = "WIF,BONK,POPCAT,PEPE,DOGE"

    # --- Chain / execution ---
    CHAIN: str = "solana"                  # solana | evm
    EXECUTION_BACKEND: str = "jupiter"     # jupiter | cex | evm_1inch | paper
    QUOTE_MINT: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC (Solana)

    SOLANA_RPC_URL: str = "https://api.mainnet-beta.solana.com"
    SOLANA_PRIVATE_KEY: Optional[str] = None
    JUPITER_API_BASE: str = "https://quote-api.jup.ag/v6"
    JUPITER_PRICE_API_BASE: str = "https://price.jup.ag/v6"

    EVM_RPC_URL: Optional[str] = None
    EVM_PRIVATE_KEY: Optional[str] = None
    EVM_CHAIN_ID: int = 1
    ONEINCH_API_KEY: Optional[str] = None
    ONEINCH_API_BASE: str = "https://api.1inch.dev/swap/v6.0"
    # An ERC20 stablecoin address (e.g. USDC) on your target chain. Required
    # for live EVM execution - trades are sized in USD and this is what gets
    # swapped against; native-currency-denominated buying isn't supported
    # (see app/execution/evm.py for why).
    EVM_QUOTE_TOKEN_ADDRESS: Optional[str] = None

    CEX_EXCHANGE: str = "binance"
    CEX_API_KEY: Optional[str] = None
    CEX_API_SECRET: Optional[str] = None

    # Trading fee charged per side in PAPER mode. Defaults to the same
    # 0.25% the backtester uses (BacktestConfig.fee_pct) so paper results
    # and backtest results are comparable - a paper engine that charged no
    # fee would look better than the backtest meant to validate it, which
    # is the wrong direction for an error to point.
    PAPER_FEE_PCT: float = 0.0025

    # --- Realistic paper fill model (app/execution/fill_model.py) ---
    # Price impact is DERIVED from pool liquidity (constant-product AMM),
    # not configured - a big trade against a thin pool costs more, which is
    # the single most important thing a memecoin simulator must get right.
    # These cover the rest.
    PAPER_SPREAD_PCT: float = 0.0015
    # Confirmation window. A swap isn't instant, and on a volatile token the
    # price drift during it can dwarf the spread. Solana lands in ~0.4-2s;
    # widen these for a slower chain.
    PAPER_MIN_CONFIRM_SECONDS: float = 0.4
    PAPER_MAX_CONFIRM_SECONDS: float = 2.5
    # When false, paper fills always succeed (the old behavior). Leaving it
    # true means a fill can REVERT when impact + drift exceed SLIPPAGE_BPS,
    # exactly as an on-chain swap with slippage protection does - which is
    # itself a real cost a simulator should charge.
    PAPER_ALLOW_FAILED_FILLS: bool = True

    SLIPPAGE_BPS: int = 150
    MAX_GAS_PRICE_GWEI: float = 50.0
    MAX_TRADE_SIZE_USD: float = 200.0

    # --- Risk management (hard-coded, non-negotiable limits) ---
    MAX_PORTFOLIO_PCT_PER_TRADE: float = 0.02
    DAILY_LOSS_LIMIT_PCT: float = 0.05
    STOP_LOSS_PCT: float = 0.15
    TAKE_PROFIT_PCT: float = 0.30
    MAX_CONCURRENT_POSITIONS: int = 5
    PORTFOLIO_STARTING_BALANCE_USD: float = 1000.0
    MAX_EXPOSURE_PER_TOKEN_PCT: float = 0.10
    MAX_TOTAL_EXPOSURE_PCT: float = 0.60
    # --- correlation risk (app/risk/book.py) ---
    # Five memecoins are not five independent positions. The per-token and
    # total exposure caps above are both satisfied by a book that is really
    # one bet at five times the intended size; this caps the bet itself.
    CORRELATION_RISK_ENABLED: bool = True
    # Largest share of the portfolio allowed in one MEASURABLY correlated
    # cluster. Only pairs with enough overlapping observations and a
    # correlation at or above 0.70 count - an unmeasured pair is reported
    # but never blocks, because on a fresh install every pair is unmeasured
    # and blocking on that would stop the bot opening a second position
    # before it could ever collect the data proving otherwise.
    MAX_CORRELATED_CLUSTER_PCT: float = 0.30
    # --- data cross-check (app/data/cross_check.py) ---
    # Two providers reporting different liquidity for the same token means
    # one of them is wrong, and there is no way to tell which. Sizing off
    # the wrong one costs the position, so the trade is skipped instead.
    CROSS_CHECK_ENABLED: bool = True
    # Require a second source to have answered at all. Off by default: a
    # provider being down is a reason to be careful, not automatically a
    # reason to stop trading, and the disagreement check still applies.
    CROSS_CHECK_REQUIRE_TWO_SOURCES: bool = False
    CROSS_CHECK_LIQUIDITY_TOLERANCE: float = 0.30
    MAX_CONSECUTIVE_LOSSES: int = 4
    MAX_DAILY_TRADES: int = 8
    TRADE_COOLDOWN_SECONDS: int = 900

    # --- Smart exits (Stage 4) ---
    TRAILING_STOP_ENABLED: bool = True
    TRAILING_STOP_ACTIVATION_PCT: float = 0.15   # start trailing once up this much from entry
    TRAILING_STOP_DISTANCE_PCT: float = 0.10     # trail this far behind the peak price

    BREAK_EVEN_ENABLED: bool = True
    BREAK_EVEN_TRIGGER_PCT: float = 0.10         # move stop to break-even once up this much
    BREAK_EVEN_BUFFER_PCT: float = 0.01          # lock in a hair of profit, not exactly $0

    PARTIAL_TAKE_PROFIT_ENABLED: bool = True
    PARTIAL_TAKE_PROFIT_TRIGGER_PCT: float = 0.20  # take partial profit once up this much
    PARTIAL_TAKE_PROFIT_SIZE_PCT: float = 0.50     # fraction of the position sold

    MOMENTUM_EXIT_ENABLED: bool = True
    MOMENTUM_EXIT_LOOKBACK_SAMPLES: int = 6        # recent monitor-tick samples considered
    MOMENTUM_EXIT_DROP_PCT: float = 0.12           # exit if price fell this much off the recent peak

    TREND_REVERSAL_EXIT_ENABLED: bool = True
    TREND_REVERSAL_MIN_SAMPLES: int = 5            # samples needed before checking for reversal

    TIME_BASED_EXIT_ENABLED: bool = False
    MAX_POSITION_AGE_HOURS: float = 48.0
    # --- autopilot (app/autopilot/) ---
    # The self-improvement loop. Diagnoses recorded data, logs proposals,
    # and judges registered challengers at the promotion gate. It cannot
    # modify source and cannot reach the live-trading gates.
    AUTOPILOT_ENABLED: bool = True
    # Deliberately slow. Running the search more often does not improve
    # faster - it resamples the same weeks and calls the noise a signal.
    # The gate counts attempts across cycles, so a busier loop makes
    # promotion HARDER, which is the correct incentive.
    AUTOPILOT_INTERVAL_HOURS: float = 6.0
    # --- shadow challengers (app/shadow/) ---
    # Challengers evaluate the same opportunities as the champion and
    # record what they WOULD have done. They cannot open, close or size a
    # real paper position - separate tables, no execution client.
    SHADOW_ENABLED: bool = True
    # JSON list of parameter overrides. The shipped pair brackets the
    # champion's entry threshold and changes NOTHING else, so the answer
    # to "did the challenger do better" is an answer about the threshold
    # rather than about twenty parameters that all moved at once. Weight
    # experiments come after this one concludes, one factor at a time.
    SHADOW_CHALLENGERS: str = (
        '[{"strategy_id": "strict-70", "min_score_to_enter": 70,'
        ' "description": "entry threshold +5 vs champion; nothing else changed"},'
        ' {"strategy_id": "loose-60", "min_score_to_enter": 60,'
        ' "description": "entry threshold -5 vs champion; nothing else changed"}]'
    )
    # Notional used for hypothetical fills. Fixed rather than taken from
    # the risk manager, because sizing depends on live exposure and a
    # challenger must never read - let alone move - that state.
    SHADOW_POSITION_USD: float = 100.0
    # --- shadow outcome resolver (app/shadow/resolver.py) ---
    # Without this running, every hypothetical entry stays open forever and
    # the shadow tables hold decisions with no outcomes - which is a
    # dataset that cannot answer the question it was collected for.
    SHADOW_RESOLVER_ENABLED: bool = True
    # Candle granularity the exit rule is walked on. Finer sees a stop
    # breach that a coarser bar averages away; too fine and the provider's
    # 1000-candle ceiling stops covering the maximum hold.
    SHADOW_RESOLUTION_TIMEFRAME: str = "5m"
    SHADOW_RESOLVE_BATCH: int = 50
    SHADOW_RESOLVE_INTERVAL_SECONDS: int = 300
    # Fixed horizons recorded regardless of the exit rule, so a good entry
    # cut short by a stop can be told apart from a bad entry.
    SHADOW_HORIZONS_MINUTES: str = "15,60,240,1440"
    # How long past its due point an outcome is chased before being filed
    # as unmeasurable. A token whose feed went quiet half a day ago is not
    # going to answer, and retrying forever is a permanent background load.
    SHADOW_UNMEASURABLE_AFTER_HOURS: float = 12.0
    # --- liquidity-drop exit (app/exits/manager.py) ---
    # A pool being drained while the position is open is the one failure the
    # price-based exits cannot catch in time: the quote looks normal until
    # the depth is gone, and by then the stop-loss fills into nothing.
    LIQUIDITY_EXIT_ENABLED: bool = True
    # Fraction of entry liquidity below which the position is closed outright.
    LIQUIDITY_EXIT_DROP_PCT: float = 0.50
    # A softer drop takes the partial-exit route instead of a full close.
    LIQUIDITY_WARN_DROP_PCT: float = 0.30
    # Floor in dollars, independent of the entry level. A pool that started
    # thin and got thinner is dangerous even if the ratio looks survivable.
    LIQUIDITY_EXIT_FLOOR_USD: float = 5_000.0

    # --- Live signal score (Stage 2's scoring engine, wired into live entries) ---
    # Master switch. False skips the score gate entirely (loud warning
    # logged, same pattern as RUGCHECK_ENABLED=false) - entries then run on
    # the TradingView alert + rug check alone, same as before this existed.
    LIVE_SIGNAL_SCORE_ENABLED: bool = True
    # Measured, not guessed. The score is a weighted average of 14 factors,
    # so it regresses toward 50 by construction and its practical ceiling is
    # nowhere near 100. Across 120 synthetic runs spanning every regime the
    # median was 58 and the 95th percentile 74, giving these qualifying
    # rates: 60 -> ~43%, 65 -> ~26%, 70 -> ~10%, 75 -> ~3%, 80 -> ~1%.
    #
    # This defaulted to 75 (from the original spec's "only enter if score
    # >= 75"), which turns out to be the ~97th percentile of this engine's
    # own output - so once the rug check took its cut too, the bot traded
    # essentially never. 65 still rejects three out of four setups, which is
    # the "reject weak setups" intent, while letting it actually trade.
    # Raise it if paper trading shows too many marginal entries.
    MIN_SIGNAL_SCORE_TO_ENTER: float = 65.0
    SIGNAL_SCORE_TIMEFRAME: str = "15m"
    SIGNAL_SCORE_CANDLE_LIMIT: int = 300
    # Fewer live candles than this and the score is treated the same as no
    # data at all (rejected, not scored on a thin sample) - matches the
    # backtester's own warmup_bars default or reasoning by margin, though it
    # doesn't have to match exactly since a live gate only needs ONE
    # trustworthy reading, not a full walk-forward history.
    SIGNAL_SCORE_MIN_CANDLES: int = 60
    GECKOTERMINAL_API_BASE: str = "https://api.geckoterminal.com/api/v2"

    # --- Market quality score (app/signals/market_quality.py) ---
    # Deliberately separate from BOTH the security score (is this a scam?)
    # and the signal score (is this a good setup?). A token can be perfectly
    # safe and show a textbook breakout while being untradeable in practice:
    # one wash-traded volume spike, a pool nobody can exit, activity
    # concentrated in three transactions. This scores tradeability itself.
    MARKET_QUALITY_ENABLED: bool = True
    MIN_MARKET_QUALITY_SCORE: float = 50.0

    # --- Data freshness / sanity (app/data/staleness.py) ---
    # Stale data is worse than missing data because it looks authoritative:
    # a price from 20 minutes ago passes every check and then sizes a
    # position against a market that has moved on.
    MAX_MARKET_DATA_AGE_SECONDS: float = 120.0
    # A move beyond this factor versus the previous observation is treated
    # as a bad tick rather than a real move. Deliberately generous -
    # memecoins genuinely do move violently, so this catches broken data
    # (decimals bugs, thin-pool prints, feed glitches), not volatility.
    MAX_PRICE_JUMP_FACTOR: float = 20.0

    # --- score calibration research (app/analysis/forward_returns.py) ---
    # Follow the price of every SCORED candidate - traded or rejected - so
    # the bot can eventually answer whether a higher score actually precedes
    # a better outcome. Judging the score from trades alone cannot answer
    # that: the bot only trades what it already liked, so the rejected 55s
    # never get to disagree.
    # Costs one price lookup per distinct mint per resolution pass, which is
    # why it is a switch. Turning it off stops the bot ever being able to
    # validate or falsify its own scoring engine.
    FORWARD_RETURNS_ENABLED: bool = True
    FORWARD_RETURN_RESOLVE_INTERVAL_SECONDS: int = 300
    FORWARD_RETURN_BATCH_LIMIT: int = 200

    # --- global entry kill switch (app/safety/killswitch.py) ---
    # Stops NEW positions when the bot cannot trust its own state. It never
    # closes existing ones: halting entries is safe, force-liquidating a
    # book because a price feed hiccuped turns a data problem into a
    # realised loss.
    # The two failures this exists for are the silent ones - trading on a
    # feed that stopped updating, and sizing positions off a cash balance
    # that is wrong. Both look completely normal in the logs.
    KILL_SWITCH_ENABLED: bool = True
    # A critical data source that has not succeeded in this long is treated
    # as down, not merely slow.
    KILL_SWITCH_MAX_DATA_AGE_SECONDS: float = 900.0
    KILL_SWITCH_MAX_CONSECUTIVE_FAILURES: int = 5
    # Position size is a fraction of portfolio value, and portfolio value
    # includes open positions. Once this share of the book is being valued
    # at cost (because no live price came back), that fraction is computed
    # from a number that no longer means anything.
    KILL_SWITCH_MAX_STALE_VALUATION_SHARE: float = 0.50

    # --- Early Signal Engine (app/early/) ---
    # Tries to detect demand ARRIVING, as opposed to the technical score's
    # reading of whether the chart already looks good.
    EARLY_SIGNAL_ENABLED: bool = True
    # Below this, a candidate is not interesting enough to keep looking at.
    EARLY_SIGNAL_WATCH_THRESHOLD: float = 55.0
    # At or above this - plus a healthy pattern and an enterable stage - the
    # candidate is CONFIRMED.
    EARLY_SIGNAL_CONFIRM_THRESHOLD: float = 70.0
    # Require the existing technical score to agree before entering. This is
    # combination strategy "C": early signal finds the candidate, the
    # existing strategy confirms the entry.
    EARLY_SIGNAL_REQUIRE_TECHNICAL: bool = True
    # How far BELOW the trading threshold a technical score may sit and still
    # be shown to the early engine. The engine exists to watch charts that do
    # not look good yet, so stopping at the technical gate would leave it
    # seeing only candidates the bot was already going to buy. Continuing
    # costs a security lookup and a candle fetch per candidate, so the window
    # is bounded rather than open: with the default threshold of 65 this
    # means a technical score of 40 or better still gets a look.
    EARLY_SIGNAL_TECHNICAL_MARGIN: float = 25.0
    #
    # THE SWITCH THAT MATTERS. False means the early engine can raise a token
    # to WATCH and can NEVER open a position on its own - the existing
    # technical strategy remains the only thing that trades.
    #
    # The weights in app/early/score.py are unvalidated priors, chosen from
    # reasoning about microstructure rather than from measured outcomes.
    # Enabling this before app/analysis/early_calibration.py shows higher
    # scores actually precede better outcomes means trading on a guess.
    EARLY_SIGNAL_MAY_TRADE: bool = False
    # How often WATCH tokens are re-scored, and how long they stay on the
    # list without confirming.
    WATCHLIST_INTERVAL_SECONDS: int = 120
    WATCHLIST_MAX_AGE_HOURS: float = 12.0
    WATCHLIST_MAX_SIZE: int = 200
    # Observation history feeds the flow features (transaction rate, buy
    # pressure change) which have no other source. Pruned by age since their
    # whole value is recency.
    OBSERVATION_RETENTION_HOURS: float = 48.0

    # --- Automatic token scanner (Stage 9) ---
    # Finds newly listed tokens itself instead of waiting for a TradingView
    # alert to name one. Discovered tokens go through the SAME pipeline a
    # webhook alert does (risk gate -> signal score -> rug check -> sizing),
    # so nothing here bypasses an existing protection.
    SCANNER_ENABLED: bool = True
    SCANNER_INTERVAL_SECONDS: int = 60
    # Auto-discovery + auto-buy + real money is a much bigger step than any
    # one of those alone: the scanner will happily find a token nobody has
    # ever looked at and open a position in it unattended. In paper mode
    # that's the point. With LIVE_TRADING=true the scanner refuses to run
    # unless this is ALSO set, so upgrading a live deployment can't silently
    # start auto-buying brand-new memecoins.
    SCANNER_ALLOW_LIVE_TRADING: bool = False

    # Cheap pre-screen, applied to the listing payload before any rug check
    # or candle fetch is spent on a candidate (app/scanner/filters.py).
    SCANNER_MAX_TOKENS_PER_CYCLE: int = 30
    # Raised from 25k so the worst realistic case stays fillable: price
    # impact is trade/(liquidity/2), so MAX_TRADE_SIZE_USD=200 against a
    # 25k pool implied 1.6% impact against a 1.5% SLIPPAGE_BPS tolerance -
    # those fills revert. 35k puts the worst case at ~1.1%, comfortably
    # inside tolerance. app/startup_checks.py re-checks this relationship
    # if you change any of the three.
    SCANNER_MIN_LIQUIDITY_USD: float = 35_000.0
    SCANNER_MIN_VOLUME_24H_USD: float = 50_000.0
    SCANNER_MIN_TXNS_24H: int = 100
    SCANNER_MAX_SELL_SHARE: float = 0.70
    # Skip the first hours of a pool's life - the highest-risk rug window,
    # and too little history for the signal engine to read anyway.
    #
    # MUST be at least SIGNAL_SCORE_MIN_CANDLES x SIGNAL_SCORE_TIMEFRAME
    # (currently 60 x 15m = 15h) or the two gates contradict each other:
    # younger tokens clear the pre-screen and are then guaranteed to fail
    # the score gate for lack of history, forever, while the logs look
    # completely normal. app/startup_checks.py warns loudly if this drifts
    # out of sync again.
    SCANNER_MIN_TOKEN_AGE_HOURS: float = 16.0
    SCANNER_MAX_TOKEN_AGE_HOURS: float = 720.0   # 30 days; 0 disables the ceiling
    # How long before a rejected token is worth re-evaluating. Without this
    # the scanner re-analyses the same few hundred tokens every cycle.
    SCANNER_RECHECK_MINUTES: int = 60

    BIRDEYE_API_KEY: Optional[str] = None
    BIRDEYE_API_BASE: str = "https://public-api.birdeye.so"

    # --- Rug-pull / scam filter ---
    RUGCHECK_ENABLED: bool = True
    MAX_TOP10_HOLDER_PCT: float = 0.35
    MIN_LIQUIDITY_USD: float = 15000.0
    MAX_PRICE_IMPACT_PCT: float = 0.05
    DEV_WALLET_SELL_ALERT_PCT: float = 0.10
    GOPLUS_API_KEY: Optional[str] = None
    GOPLUS_API_BASE: str = "https://api.gopluslabs.io/api/v1"
    HONEYPOT_API_BASE: str = "https://api.honeypot.is/v2"
    RUGCHECK_API_BASE: str = "https://api.rugcheck.xyz/v1"
    # Reject when the Solana specialist reports any risk it classes as
    # danger. Set false to downgrade those to warnings (NOT recommended -
    # "large amount of LP unlocked" arrives as exactly such a risk).
    REJECT_ON_DANGER_RISKS: bool = True

    # Composite Rug Risk Score (0-100, app/rugcheck/risk_score.py) threshold.
    # This is an ADDITIONAL gate on top of the binary checks above, not a
    # replacement - a token can fail this even after passing every one of
    # them, if enough moderate risk factors stack up together.
    REJECT_RUG_SCORE_ABOVE: float = 65.0

    # --- Monitoring loop ---
    PRICE_POLL_INTERVAL_SECONDS: int = 30
    DEV_WALLET_POLL_INTERVAL_SECONDS: int = 300
    DAILY_SUMMARY_HOUR_UTC: int = 23

    # --- Notifications ---
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    DISCORD_WEBHOOK_URL: Optional[str] = None

    # --- Dashboard ---
    DASHBOARD_USERNAME: str = "admin"
    DASHBOARD_PASSWORD: str = "changeme"


settings = Settings()
