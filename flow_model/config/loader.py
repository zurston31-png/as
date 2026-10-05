"""Configuration loading, merging and override parsing.

Precedence, lowest to highest:

    packaged defaults.yaml  <  user YAML file(s)  <  explicit --set overrides

Merging is a deep merge for mappings. Sequences are *replaced*, not
concatenated: appending to a list of weights or thresholds is almost never
what a user means, and silently growing a list is hard to notice.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from flow_model.config.schema import FlowModelConfig

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


class ConfigError(ValueError):
    """Raised for malformed config files or override expressions."""


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level of a config file must be a mapping")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`. Sequences are replaced."""
    out = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce_scalar(text: str) -> Any:
    """Parse an override value using YAML scalar rules.

    YAML gives us ints, floats, booleans, null, dates and inline
    lists/mappings for free, so `--set risk.risk_per_trade_pct=0.0025` and
    `--set backtest.symbols=[NQ,ES]` both work.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    try:
        value = yaml.safe_load(stripped)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse override value {text!r}: {exc}") from exc
    return value


def parse_override(expression: str) -> tuple[list[str], Any]:
    """Parse `a.b.c=value` into (['a','b','c'], parsed_value)."""
    if "=" not in expression:
        raise ConfigError(
            f"override {expression!r} must be of the form dotted.key=value"
        )
    key_text, _, value_text = expression.partition("=")
    path = [part for part in key_text.strip().split(".") if part]
    if not path:
        raise ConfigError(f"override {expression!r} has an empty key")
    return path, _coerce_scalar(value_text)


def apply_override(data: dict[str, Any], path: list[str], value: Any) -> dict[str, Any]:
    """Set a nested key, creating intermediate mappings as needed.

    Refuses to overwrite a scalar with a mapping implicitly, which would
    otherwise silently discard config.
    """
    node: Any = data
    for part in path[:-1]:
        existing = node.get(part)
        if existing is None:
            node[part] = {}
        elif not isinstance(existing, dict):
            raise ConfigError(
                f"cannot apply override at {'.'.join(path)}: "
                f"{part!r} is a {type(existing).__name__}, not a mapping"
            )
        node = node[part]
    node[path[-1]] = value
    return data


def load_raw(
    paths: Iterable[str | Path] | None = None,
    overrides: Iterable[str] | None = None,
    include_defaults: bool = True,
) -> dict[str, Any]:
    """Resolve the merged raw mapping without validating it."""
    data: dict[str, Any] = _read_yaml(DEFAULTS_PATH) if include_defaults else {}
    for path in paths or ():
        data = deep_merge(data, _read_yaml(path))
    for expression in overrides or ():
        key_path, value = parse_override(expression)
        data = apply_override(data, key_path, value)
    return data


def load_config(
    paths: Iterable[str | Path] | None = None,
    overrides: Iterable[str] | None = None,
    include_defaults: bool = True,
) -> FlowModelConfig:
    """Load, merge, validate.

    Validation errors are re-raised as `ConfigError` with the offending
    field paths intact, so a bad YAML value names itself.
    """
    raw = load_raw(paths=paths, overrides=overrides, include_defaults=include_defaults)
    try:
        return FlowModelConfig(**raw)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"invalid configuration:\n{exc}") from exc


def default_config() -> FlowModelConfig:
    """The packaged baseline."""
    return load_config()


def save_config(config: FlowModelConfig, path: str | Path) -> Path:
    """Write a resolved config back to YAML.

    Writes constructor inputs only, so the file can be loaded again
    (a dump including computed fields would not round-trip).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_init_dict(mode="json")
    with p.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Resolved Flow Model configuration\n"
            f"# config_hash: {config.config_hash}\n"
        )
        yaml.safe_dump(payload, fh, sort_keys=True, default_flow_style=False)
    return p
