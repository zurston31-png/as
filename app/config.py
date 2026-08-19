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
