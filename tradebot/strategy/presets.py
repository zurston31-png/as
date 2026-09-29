"""Named starting points.

A preset is the default value of `strategy.rules`. Pick one with
`strategy.preset:` and override individual rules by listing `strategy.rules:`
yourself - the config always wins.
"""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, list[dict[str, Any]]] = {
    # The stack from the brief.
    "ema_vwap_rsi": [
        {"rule": "ema_stack", "mode": "required", "params": {"fast": 9, "slow": 21}},
        {"rule": "ema_cross_fresh", "mode": "required",
         "params": {"fast": 9, "slow": 21, "max_bars": 5}},
        {"rule": "vwap_position", "mode": "required"},
        {"rule": "rsi_window", "mode": "required",
         "params": {"period": 14, "long_min": 50, "long_max": 72,
                    "short_min": 28, "short_max": 50}},
        {"rule": "market_structure", "mode": "required",
         "params": {"width": 2, "lookback": 60}},
        {"rule": "liquidity_sweep", "mode": "advisory",
         "params": {"width": 2, "lookback": 5, "min_wick_ratio": 0.5}},
        {"rule": "volume_confirmation", "mode": "required",
         "params": {"period": 20, "multiple": 1.1}},
        {"rule": "candle_direction", "mode": "required", "weight": 0.5},
    ],
    # Trend continuation without demanding a fresh cross - fires more often.
    "trend_pullback": [
        {"rule": "ema_stack", "mode": "required", "params": {"fast": 9, "slow": 21}},
        {"rule": "vwap_position", "mode": "required"},
        {"rule": "rsi_window", "mode": "required",
         "params": {"period": 14, "long_min": 45, "long_max": 68,
                    "short_min": 32, "short_max": 55}},
        {"rule": "market_structure", "mode": "required"},
        {"rule": "volume_confirmation", "mode": "advisory"},
    ],
    # Mean reversion off a swept level; ignores the EMA stack entirely.
    "sweep_reversal": [
        {"rule": "liquidity_sweep", "mode": "required",
         "params": {"width": 2, "lookback": 3, "min_wick_ratio": 0.55}},
        {"rule": "structure_break", "mode": "advisory"},
        {"rule": "rsi_window", "mode": "required",
         "params": {"period": 14, "long_min": 25, "long_max": 55,
                    "short_min": 45, "short_max": 75}},
        {"rule": "candle_direction", "mode": "required"},
        {"rule": "atr_range", "mode": "advisory", "params": {"min_pct": 0.02}},
    ],
}


def rules_for(name: str) -> list[dict[str, Any]]:
    if name not in PRESETS:
        raise KeyError(f"unknown strategy preset {name!r}; have {sorted(PRESETS)}")
    return [dict(r) for r in PRESETS[name]]
