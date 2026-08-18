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

    CEX_EXCHANGE: str = "binance"
    CEX_API_KEY: Optional[str] = None
    CEX_API_SECRET: Optional[str] = None

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

    # --- Rug-pull / scam filter ---
    RUGCHECK_ENABLED: bool = True
    MAX_TOP10_HOLDER_PCT: float = 0.35
    MIN_LIQUIDITY_USD: float = 15000.0
    MAX_PRICE_IMPACT_PCT: float = 0.05
    DEV_WALLET_SELL_ALERT_PCT: float = 0.10
    GOPLUS_API_KEY: Optional[str] = None
    GOPLUS_API_BASE: str = "https://api.gopluslabs.io/api/v1"
    HONEYPOT_API_BASE: str = "https://api.honeypot.is/v2"

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
