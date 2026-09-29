"""Named rule stacks.

A preset is just an ordered list of rule names. `optional_rules` in the config
decides which of them are allowed to fail without killing the setup.
"""

from __future__ import annotations

PRESETS: dict[str, list[str]] = {
    # The stack from the brief: 9/21 EMA cross, VWAP side, RSI window, market
    # structure, liquidity sweep, volume confirmation.
    "ema_vwap_rsi": [
        "ema_stack",
        "ema_cross_fresh",
        "vwap_position",
        "rsi_window",
        "market_structure",
        "liquidity_sweep",
        "volume_confirmation",
        "candle_direction",
    ],
    # Trend continuation without demanding a fresh cross - fires more often.
    "trend_pullback": [
        "ema_stack",
        "vwap_position",
        "rsi_window",
        "market_structure",
        "volume_confirmation",
    ],
    # Mean-reversion off a swept level; ignores the EMA stack entirely.
    "sweep_reversal": [
        "liquidity_sweep",
        "structure_break",
        "rsi_window",
        "candle_direction",
    ],
}


def rules_for(name: str) -> list[str]:
    if name not in PRESETS:
        raise KeyError(f"unknown strategy preset {name!r}; have {sorted(PRESETS)}")
    return list(PRESETS[name])
