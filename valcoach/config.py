"""Configuration: file, then environment, then CLI flags (last wins)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "claude-opus-5"


def valcoach_home() -> str:
    return os.environ.get("VALCOACH_HOME") or os.path.join(
        os.path.expanduser("~"), ".valcoach"
    )


def config_path() -> str:
    return os.path.join(valcoach_home(), "config.json")


def assets_path() -> str:
    return os.path.join(valcoach_home(), "assets.json")


@dataclass
class Config:
    # identity
    riot_id: str = ""
    puuid: str = ""
    # data source
    provider: str = "henrik"
    region: str = "na"
    platform: str = "pc"
    shard: str = ""
    henrik_api_key: str = ""
    riot_api_key: str = ""
    file_paths: List[str] = field(default_factory=list)
    # analysis
    queue: str = ""                 # e.g. "competitive" to ignore deathmatch noise
    trade_window_ms: int = 4000     # a death is "traded" if avenged within this
    isolation_units: float = 1800.0 # game units; ~a long corridor apart
    # coaching
    model: str = DEFAULT_MODEL
    effort: str = "high"
    anthropic_api_key: str = ""     # prefer ANTHROPIC_API_KEY in the environment
    # runtime
    db_path: str = ""
    watch_interval: int = 180

    # ---- persistence ---------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        cfg = cls()
        target = path or config_path()
        if os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                cfg.update(data)
            except (OSError, json.JSONDecodeError):
                pass
        cfg.update(cls._from_env())
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v not in (None, "")})
        if not cfg.db_path:
            from .store import default_db_path

            cfg.db_path = default_db_path()
        if not cfg.shard:
            cfg.shard = cfg.region
        return cfg

    @staticmethod
    def _from_env() -> Dict[str, Any]:
        env_map = {
            "riot_id": "VALCOACH_RIOT_ID",
            "puuid": "VALCOACH_PUUID",
            "provider": "VALCOACH_PROVIDER",
            "region": "VALCOACH_REGION",
            "platform": "VALCOACH_PLATFORM",
            "shard": "VALCOACH_SHARD",
            "henrik_api_key": "HENRIK_API_KEY",
            "riot_api_key": "RIOT_API_KEY",
            "model": "VALCOACH_MODEL",
            "db_path": "VALCOACH_DB",
            "queue": "VALCOACH_QUEUE",
        }
        out: Dict[str, Any] = {}
        for key, env in env_map.items():
            value = os.environ.get(env)
            if value:
                out[key] = value
        # The Anthropic SDK reads ANTHROPIC_API_KEY itself; only mirror it so
        # `valcoach status` can report whether coaching is available.
        if os.environ.get("ANTHROPIC_API_KEY"):
            out["anthropic_api_key"] = os.environ["ANTHROPIC_API_KEY"]
        return out

    def update(self, data: Dict[str, Any]) -> None:
        known = {f for f in asdict(self)}
        for key, value in (data or {}).items():
            if key in known and value is not None:
                current = getattr(self, key)
                if isinstance(current, list) and isinstance(value, str):
                    value = [v for v in value.split(",") if v]
                elif isinstance(current, bool):
                    value = str(value).lower() not in ("0", "false", "no", "")
                elif isinstance(current, int) and not isinstance(current, bool):
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        continue
                elif isinstance(current, float):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                setattr(self, key, value)

    def save(self, path: Optional[str] = None) -> str:
        target = path or config_path()
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        data = asdict(self)
        # Secrets that live in the environment do not belong in the file.
        if os.environ.get("ANTHROPIC_API_KEY"):
            data["anthropic_api_key"] = ""
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        return target

    # ---- assets --------------------------------------------------------
    def load_assets(self) -> Dict[str, Dict[str, str]]:
        """UUID→name tables + map callouts, refreshed by ``valcoach assets``."""
        path = assets_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

    def redacted(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("henrik_api_key", "riot_api_key", "anthropic_api_key"):
            if data.get(key):
                data[key] = f"set ({len(str(data[key]))} chars)"
        return data
