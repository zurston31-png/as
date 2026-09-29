"""Typed configuration, loaded from YAML with environment-variable overrides.

Every knob the user asked for lives here: strategy thresholds, the AI layer,
and the whole risk engine. Nothing in the pipeline reads os.environ directly
apart from secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dependency at runtime
    yaml = None  # type: ignore

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


@dataclass
class MarketConfig:
    symbol: str = "NQ1!"
    timeframe: str = "5m"
    point_value: float = 1.0          # account currency per 1.0 of price move, per unit
    tick_size: float = 0.25
    qty_step: float = 1.0             # 1.0 = whole contracts, 0.0001 = fractional crypto
    min_qty: float = 1.0
    session_tz: str = "America/New_York"
    session_start: str = "09:30"
    session_end: str = "16:00"
    trade_session_only: bool = True


DEFAULT_STRATEGY: dict[str, Any] = {"preset": "ema_vwap_rsi"}


@dataclass
class AIConfig:
    enabled: bool = True
    model: str = "claude-opus-5-5"
    effort: str = "medium"            # low | medium | high | xhigh | max
    max_tokens: int = 4000
    # What an unavailable / malformed / timed-out verdict means. "wait" is the
    # safe default: a confirmation layer you cannot reach has not confirmed
    # anything. "rules_only" trades the rules alone when the API is down.
    on_failure: str = "wait"          # wait | rules_only
    confirm_min_confidence: float = 0.55
    candles_in_prompt: int = 40
    timeout_seconds: float = 45.0
    chat_model: str = "claude-opus-5-5"
    chat_effort: str = "low"
    chat_max_tokens: int = 8000
    chat_history_turns: int = 24


@dataclass
class VisionConfig:
    """Screen capture, used as a *backup* corroboration layer."""

    enabled: bool = False
    mode: str = "backup"              # backup | always | off
    region: Optional[dict[str, int]] = None   # {"left":0,"top":0,"width":1920,"height":1080}
    monitor: int = 1
    max_width: int = 1400             # downscale before sending to the model
    model: str = "claude-opus-5-5"
    max_tokens: int = 2000
    min_interval_seconds: float = 20.0
    save_captures: bool = False
    capture_dir: str = "data/captures"
    # When the structured feed goes quiet for this long, vision takes over.
    stale_feed_seconds: float = 120.0
    # Reject the structured signal if the screen disagrees by more than this %.
    price_tolerance_pct: float = 0.5


@dataclass
class RiskConfig:
    starting_equity: float = 25_000.0
    risk_per_trade_pct: float = 0.5           # percent of equity, matches the brief
    max_trades_per_day: int = 5
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 1
    max_consecutive_losses: int = 2
    cooldown_minutes_after_loss: int = 30
    require_stop: bool = True
    max_risk_per_trade_pct: float = 2.0       # hard ceiling regardless of sizing
    min_rr: float = 1.5
    kill_switch: bool = False
    state_path: str = "data/risk_state.json"
    audit_path: str = "data/audit.jsonl"
    # Flatten everything and stop for the day when equity drops this far.
    daily_drawdown_kill_pct: float = 3.0


@dataclass
class ExecutionConfig:
    mode: str = "paper"               # paper | alerts_only | live (live is a stub)
    slippage_ticks: float = 1.0        # entry, against you
    stop_slippage_ticks: float = 1.0   # stops slip further than limits do
    target_slippage_ticks: float = 0.0 # a resting limit usually fills at its price
    commission_per_unit: float = 0.75  # per contract PER SIDE - charged twice per trade
    partial_at_tp1: float = 0.5       # scale out half at TP1
    move_stop_to_breakeven_after_tp1: bool = True
    trades_path: str = "data/trades.jsonl"


@dataclass
class FeedConfig:
    source: str = "synthetic"         # synthetic | replay | webhook | poll
    csv_path: str = ""
    replay_speed: float = 60.0        # bars per second when replaying
    poll_url: str = ""
    poll_interval_seconds: float = 15.0
    webhook_secret: str = ""
    warmup_csv: str = ""              # history preloaded before live bars arrive
    # Continuous freshness: a decision bar older than this many *bar intervals*
    # blocks every signal, long before the screen-capture backup gets involved.
    max_stale_bars: float = 2.5
    max_stale_seconds: float = 0.0    # absolute override; 0 = derive from the timeframe


@dataclass
class NotifyConfig:
    sound: bool = True
    desktop: bool = True
    console: bool = True
    only_actionable: bool = True      # don't beep for NO TRADE


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    open_browser: bool = False


@dataclass
class Config:
    market: MarketConfig = field(default_factory=MarketConfig)
    # Free-form: validated into a StrategySpec at startup. See strategy/spec.py.
    strategy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_STRATEGY))
    ai: AIConfig = field(default_factory=AIConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        data: dict[str, Any] = {}
        if path.exists():
            if yaml is None:
                raise RuntimeError("pyyaml is required to read a config file")
            data = yaml.safe_load(path.read_text()) or {}
        cfg = _build(cls, data)
        cfg.apply_env_overrides()
        return cfg

    def apply_env_overrides(self) -> None:
        """A handful of settings are more convenient as env vars."""
        env = os.environ
        if "TRADEBOT_SYMBOL" in env:
            self.market.symbol = env["TRADEBOT_SYMBOL"]
        if "TRADEBOT_TIMEFRAME" in env:
            self.market.timeframe = env["TRADEBOT_TIMEFRAME"]
        if "TRADEBOT_WEBHOOK_SECRET" in env:
            self.feed.webhook_secret = env["TRADEBOT_WEBHOOK_SECRET"]
        if "TRADEBOT_PORT" in env:
            self.server.port = int(env["TRADEBOT_PORT"])
        if env.get("TRADEBOT_AI_DISABLED", "").lower() in ("1", "true", "yes"):
            self.ai.enabled = False
        if env.get("TRADEBOT_KILL_SWITCH", "").lower() in ("1", "true", "yes"):
            self.risk.kill_switch = True

    def to_dict(self) -> dict[str, Any]:
        return _unbuild(self)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a plain dict, ignoring unknown keys."""
    nested = _nested(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name in nested and isinstance(value, dict):
            kwargs[f.name] = _build(nested[f.name], value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _nested(cls: type) -> dict[str, type]:
    """Map field name -> dataclass type for fields that are themselves configs.

    `from __future__ import annotations` turns every annotation into a string,
    so the types are read off a default instance rather than off `f.type`.
    """
    out: dict[str, type] = {}
    probe = cls()
    for f in fields(cls):
        val = getattr(probe, f.name)
        if is_dataclass(val):
            out[f.name] = type(val)
    return out


def _unbuild(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _unbuild(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_unbuild(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _unbuild(v) for k, v in obj.items()}
    return obj
