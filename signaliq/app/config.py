"""Application configuration for SignalIQ.

Every setting is read from environment variables, optionally supplied through a
local ``.env`` file (see ``.env.example``). API credentials are never
hard-coded, are held as ``SecretStr`` values and are never written to logs.
"""

import logging
import os
import re
import time
from collections.abc import Mapping
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'data' / 'signaliq.db').as_posix()}"

#: The curated Milestone 1 starter universe. It is NOT the whole US stock market.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "AMZN",
    "META",
    "TSLA",
    "GOOGL",
    "SPY",
    "QQQ",
)

#: Value stored in ``bars.source`` for data downloaded from Alpaca.
SOURCE_ALPACA = "alpaca"

#: Alpaca's official historical stock bars endpoint (multi-symbol).
ALPACA_BARS_PATH = "/v2/stocks/bars"

#: Largest page size the bars endpoint accepts.
ALPACA_MAX_PAGE_LIMIT = 10_000

#: Feed names documented by Alpaca for historical stock bars.
KNOWN_FEEDS = frozenset({"iex", "sip", "boats", "otc"})

#: Corporate-action adjustment names documented by Alpaca.
KNOWN_ADJUSTMENTS = frozenset({"raw", "split", "dividend", "spin-off", "all"})

#: Timeframes whose ingestion is implemented and tested in this milestone.
#: The database and API are timeframe-generic so intraday can be added later.
SUPPORTED_INGESTION_TIMEFRAMES: tuple[str, ...] = ("1Day",)

#: Ticker symbols such as AAPL or BRK.B (upper case).
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{0,9}(?:[.\-][A-Z0-9]{1,5})?$")

#: Canonical timeframe names. Only the long forms are accepted so that the same
#: timeframe is never stored under two different spellings (e.g. 1Day and 1D).
TIMEFRAME_PATTERN = re.compile(
    r"^(?:(?:[1-9]|[1-5][0-9])Min|(?:[1-9]|1[0-9]|2[0-3])Hour|1Day|1Week|(?:1|2|3|4|6|12)Month)$"
)

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

#: ``.env.example`` placeholders start with this text and are treated as unset.
_PLACEHOLDER_PREFIX = "REPLACE_WITH"

#: Environment variable name -> ``Settings`` field name.
_ENV_FIELDS: dict[str, str] = {
    "APCA_API_KEY_ID": "alpaca_api_key_id",
    "APCA_API_SECRET_KEY": "alpaca_api_secret_key",
    "SIGNALIQ_ALPACA_DATA_BASE_URL": "alpaca_data_base_url",
    "SIGNALIQ_ALPACA_FEED": "alpaca_feed",
    "SIGNALIQ_ALPACA_ADJUSTMENT": "alpaca_adjustment",
    "SIGNALIQ_HISTORY_START": "history_start",
    "SIGNALIQ_SYMBOLS": "symbols",
    "SIGNALIQ_DATABASE_URL": "database_url",
    "SIGNALIQ_HTTP_TIMEOUT_SECONDS": "http_timeout_seconds",
    "SIGNALIQ_HTTP_MAX_RETRIES": "http_max_retries",
    "SIGNALIQ_LOG_LEVEL": "log_level",
}

_CREDENTIAL_ENV_NAMES = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def is_valid_symbol(value: str) -> bool:
    """Return True when ``value`` looks like an upper-case ticker symbol."""
    return bool(SYMBOL_PATTERN.fullmatch(value))


def is_valid_timeframe(value: str) -> bool:
    """Return True when ``value`` is a canonical timeframe such as ``1Day``."""
    return bool(TIMEFRAME_PATTERN.fullmatch(value))


class Settings(BaseModel):
    """Validated, immutable application settings."""

    model_config = ConfigDict(frozen=True)

    # Alpaca credentials. Optional here so that the API server and the
    # inspection tools work without them; ingestion requires them.
    alpaca_api_key_id: SecretStr | None = None
    alpaca_api_secret_key: SecretStr | None = None

    # Market data request configuration.
    alpaca_data_base_url: str = "https://data.alpaca.markets"
    alpaca_feed: str = "iex"
    alpaca_adjustment: str = "split"
    alpaca_currency: str = "USD"
    alpaca_page_limit: int = Field(default=ALPACA_MAX_PAGE_LIMIT, ge=1, le=ALPACA_MAX_PAGE_LIMIT)
    default_timeframe: str = "1Day"
    history_start: date = date(2016, 1, 1)
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS

    # Storage.
    database_url: str = DEFAULT_DATABASE_URL

    # HTTP behaviour.
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_retries: int = Field(default=5, ge=0, le=20)
    http_backoff_base_seconds: float = Field(default=1.0, ge=0)
    http_backoff_max_seconds: float = Field(default=60.0, ge=0)

    log_level: str = "INFO"

    @field_validator("alpaca_data_base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith("https://"):
            raise ValueError(
                "must start with https:// (credentials are never sent over plain HTTP)"
            )
        return value

    @field_validator("alpaca_feed")
    @classmethod
    def _validate_feed(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in KNOWN_FEEDS:
            raise ValueError(f"unknown feed {value!r}; expected one of {sorted(KNOWN_FEEDS)}")
        return value

    @field_validator("alpaca_adjustment")
    @classmethod
    def _validate_adjustment(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in KNOWN_ADJUSTMENTS:
            raise ValueError(
                f"unknown adjustment {value!r}; expected one of {sorted(KNOWN_ADJUSTMENTS)}"
            )
        return value

    @field_validator("alpaca_currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        value = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("must be a three-letter ISO 4217 currency code such as USD")
        return value

    @field_validator("default_timeframe")
    @classmethod
    def _validate_timeframe(cls, value: str) -> str:
        value = value.strip()
        if not is_valid_timeframe(value):
            raise ValueError(f"invalid timeframe {value!r}; examples: 1Min, 15Min, 1Hour, 1Day")
        return value

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        symbols = tuple(symbol.strip().upper() for symbol in value)
        if not symbols:
            raise ValueError("at least one symbol is required")
        invalid = [symbol for symbol in symbols if not is_valid_symbol(symbol)]
        if invalid:
            raise ValueError(f"invalid symbol(s): {invalid}")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        return symbols

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith("sqlite:///"):
            raise ValueError("only SQLite is supported in Milestone 1 (sqlite:///path/to/file.db)")
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in _LOG_LEVELS:
            raise ValueError(f"unknown log level {value!r}; expected one of {sorted(_LOG_LEVELS)}")
        return value

    @property
    def has_alpaca_credentials(self) -> bool:
        """True when both Alpaca credential values are configured."""
        return bool(
            self.alpaca_api_key_id
            and self.alpaca_api_key_id.get_secret_value()
            and self.alpaca_api_secret_key
            and self.alpaca_api_secret_key.get_secret_value()
        )

    def require_alpaca_credentials(self) -> tuple[str, str]:
        """Return ``(key_id, secret_key)`` or raise a helpful error.

        The returned values are secrets: pass them to the HTTP client only and
        never log or print them.
        """
        key_id = self.alpaca_api_key_id.get_secret_value() if self.alpaca_api_key_id else ""
        secret = self.alpaca_api_secret_key.get_secret_value() if self.alpaca_api_secret_key else ""
        if not key_id or not secret:
            raise ConfigurationError(
                "Alpaca API credentials are not configured. Copy .env.example to .env and set "
                "APCA_API_KEY_ID and APCA_API_SECRET_KEY to your own keys."
            )
        return key_id, secret


def _resolve_sqlite_url(url: str) -> str:
    """Make relative SQLite file paths absolute, relative to the project folder.

    This keeps the database location stable no matter which directory a command
    is started from.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw_path = url[len(prefix) :]
    if not raw_path or raw_path == ":memory:" or raw_path.startswith("file:"):
        return url
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return f"{prefix}{path.as_posix()}"


def load_settings(
    env_file: Path | str | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Build :class:`Settings` from a ``.env`` file and environment variables.

    Real environment variables take precedence over values in the ``.env``
    file. Empty values and untouched ``.env.example`` placeholders count as
    "not set".
    """
    raw: dict[str, str] = {}
    if env_file is not None and Path(env_file).is_file():
        raw.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    raw.update(os.environ if environ is None else environ)

    values: dict[str, object] = {}
    for env_name, field_name in _ENV_FIELDS.items():
        value = raw.get(env_name, "").strip()
        if not value:
            continue
        if env_name in _CREDENTIAL_ENV_NAMES and value.upper().startswith(_PLACEHOLDER_PREFIX):
            continue
        values[field_name] = value

    if "database_url" in values:
        values["database_url"] = _resolve_sqlite_url(str(values["database_url"]))

    try:
        return Settings(**values)
    except ValidationError as exc:
        # Report field names and reasons only: never echo raw input values.
        field_to_env = {field: env for env, field in _ENV_FIELDS.items()}
        problems = "; ".join(
            f"{field_to_env.get(str(error['loc'][0]), error['loc'][0])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"Invalid configuration: {problems}") from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (loaded once from the environment)."""
    return load_settings()


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging with UTC timestamps.

    The ``httpx`` loggers are quietened because the Alpaca client writes its
    own request log lines (which never include credentials).
    """
    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
