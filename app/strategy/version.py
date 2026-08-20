"""Strategy versioning.

Every Signal and Trade records which strategy configuration produced it,
so analytics can never silently pool results from materially different
strategies. If the entry threshold moved from 75 to 65 halfway through a
paper-trading run, a combined win rate describes a strategy that never
existed - and that is exactly the number someone would otherwise quote.

The version label is a short, deterministic hash of the settings that
actually change trading behavior. Two rules make it useful rather than
noisy:

  BEHAVIORAL SETTINGS ONLY. Changing LOG_LEVEL, a dashboard password, or a
  poll interval does not mint a new version, because fragmenting history
  on a cosmetic change costs analytical power and buys nothing. The list
  below is explicit rather than "everything in Settings" precisely so that
  adding an unrelated setting later doesn't silently invalidate history.

  DETERMINISTIC. The same configuration always produces the same label,
  across restarts and machines, so a version genuinely identifies a
  strategy rather than a process lifetime.

CODE IS PART OF THE CONFIGURATION

Some things that decide trades are not settings at all. The scoring weight
map and the regime boundaries live in code, and an edit to either changes
every score and every regime label while leaving a settings-only hash
untouched - so history would pool observations from two different
strategies under one version and nothing would say so. Those constants are
therefore digested into the label alongside the settings. Editing a weight
mints a new version, which is the whole point.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging

from sqlalchemy.orm import Session

from app import models
from app.config import settings

logger = logging.getLogger(__name__)

# Settings that materially change which trades happen or how big they are.
# Anything absent from this list is treated as cosmetic for versioning
# purposes - see the module docstring for why the list is explicit.
BEHAVIORAL_SETTINGS = (
    # entry gating
    "LIVE_SIGNAL_SCORE_ENABLED",
    "MIN_SIGNAL_SCORE_TO_ENTER",
    "SIGNAL_SCORE_TIMEFRAME",
    "SIGNAL_SCORE_MIN_CANDLES",
    "MIN_MARKET_QUALITY_SCORE",
    # security gating
    "RUGCHECK_ENABLED",
    "MAX_TOP10_HOLDER_PCT",
    "MIN_LIQUIDITY_USD",
    "MAX_PRICE_IMPACT_PCT",
    "REJECT_RUG_SCORE_ABOVE",
    # sizing and risk
    "MAX_PORTFOLIO_PCT_PER_TRADE",
    "MAX_TRADE_SIZE_USD",
    "DAILY_LOSS_LIMIT_PCT",
    "STOP_LOSS_PCT",
    "TAKE_PROFIT_PCT",
    "MAX_CONCURRENT_POSITIONS",
    "MAX_EXPOSURE_PER_TOKEN_PCT",
    "MAX_TOTAL_EXPOSURE_PCT",
    "MAX_CONSECUTIVE_LOSSES",
    "MAX_DAILY_TRADES",
    "TRADE_COOLDOWN_SECONDS",
    # exits
    "TRAILING_STOP_ENABLED",
    "TRAILING_STOP_ACTIVATION_PCT",
    "TRAILING_STOP_DISTANCE_PCT",
    "BREAK_EVEN_ENABLED",
    "BREAK_EVEN_TRIGGER_PCT",
    "BREAK_EVEN_BUFFER_PCT",
    "PARTIAL_TAKE_PROFIT_ENABLED",
    "PARTIAL_TAKE_PROFIT_TRIGGER_PCT",
    "PARTIAL_TAKE_PROFIT_SIZE_PCT",
    "MOMENTUM_EXIT_ENABLED",
    "MOMENTUM_EXIT_DROP_PCT",
    "TREND_REVERSAL_EXIT_ENABLED",
    "TIME_BASED_EXIT_ENABLED",
    "MAX_POSITION_AGE_HOURS",
    # what the scanner will even consider
    "SCANNER_ENABLED",
    "SCANNER_MIN_LIQUIDITY_USD",
    "SCANNER_MIN_VOLUME_24H_USD",
    "SCANNER_MIN_TXNS_24H",
    "SCANNER_MAX_SELL_SHARE",
    "SCANNER_MIN_TOKEN_AGE_HOURS",
    "SCANNER_MAX_TOKEN_AGE_HOURS",
    # execution costs - these change simulated P&L directly
    "PAPER_FEE_PCT",
    "PAPER_SPREAD_PCT",
    "PAPER_ALLOW_FAILED_FILLS",
    "SLIPPAGE_BPS",
    # how a shadow observation is measured. Not entry logic, but a change
    # to any of these changes the recorded outcome for identical trading:
    # a coarser candle hides a stop breach, a different horizon set
    # answers a different question, and a different notional moves the
    # modeled price impact and therefore the fill.
    "SHADOW_POSITION_USD",
    "SHADOW_RESOLUTION_TIMEFRAME",
    "SHADOW_HORIZONS_MINUTES",
)


def _code_constants() -> dict:
    """Behavioral constants that live in code rather than in settings.

    Imported lazily: app.signals imports settings, and pulling it in at
    module scope here would close an import cycle.
    """
    from app.signals.market_regime import (
        DEEP_LIQUIDITY_USD, HIGH_VOLATILITY_ATR, LOW_VOLATILITY_ATR,
        THIN_LIQUIDITY_USD, TREND_SEPARATION,
    )
    from app.signals.scoring import DEFAULT_WEIGHTS

    return {
        "scoring_weights": dict(sorted(DEFAULT_WEIGHTS.items())),
        "regime_trend_separation": TREND_SEPARATION,
        "regime_high_volatility_atr": HIGH_VOLATILITY_ATR,
        "regime_low_volatility_atr": LOW_VOLATILITY_ATR,
        "liquidity_thin_usd": THIN_LIQUIDITY_USD,
        "liquidity_deep_usd": DEEP_LIQUIDITY_USD,
    }


_cached_label: str | None = None
_cached_config: dict | None = None


def current_config() -> dict:
    """The behavioral settings and constants, as a plain sorted dict."""
    config = {name: getattr(settings, name, None) for name in sorted(BEHAVIORAL_SETTINGS)}
    config.update(_code_constants())
    return config


def compute_label(config: dict | None = None) -> str:
    """Short deterministic label for a configuration, e.g. "v-3f9a2c1b"."""
    config = current_config() if config is None else config
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return "v-" + hashlib.sha256(encoded).hexdigest()[:8]


def current_label() -> str:
    """The active strategy version label, cached per configuration.

    Recomputed when the config changes rather than once per process, so a
    settings override applied at runtime (as the tests do) is reflected
    immediately instead of being stamped with a stale label.
    """
    global _cached_label, _cached_config
    config = current_config()
    if _cached_config != config:
        _cached_config = config
        _cached_label = compute_label(config)
    return _cached_label


def register_current_version(db: Session, notes: str | None = None) -> models.StrategyVersion:
    """Upsert the active version, returning the row.

    Called at startup and before recording a signal, so a version always
    exists to point at even on a database that predates this feature.
    """
    label = current_label()
    row = db.query(models.StrategyVersion).filter_by(label=label).first()
    now = dt.datetime.now(dt.timezone.utc)

    if row is None:
        row = models.StrategyVersion(
            label=label, created_at=now, last_seen_at=now,
            config=current_config(), notes=notes,
        )
        db.add(row)
        existing = db.query(models.StrategyVersion).count()
        if existing:
            logger.warning(
                "strategy configuration changed - now running %s. Results from earlier versions "
                "are kept separate; do not pool them when judging performance.", label,
            )
        else:
            logger.info("registered initial strategy version %s", label)
    else:
        row.last_seen_at = now

    return row
