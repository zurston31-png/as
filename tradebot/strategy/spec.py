"""The strategy, as data.

A strategy is a list of rule specs plus entry geometry. It comes out of YAML,
so changing "RSI 45-65", making volume advisory, or moving TP2 to 3R is a
config edit, not a code edit:

    strategy:
      preset: ema_vwap_rsi        # optional starting point
      rules:
        - rule: rsi_window
          mode: required
          params: {period: 14, long_min: 45, long_max: 65}
        - rule: volume_confirmation
          mode: advisory

`mode` is one of required / advisory / disabled. Required rules must all pass.
Advisory rules only move the weighted score, which must clear `min_score`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MODES = ("required", "advisory", "disabled")


class StrategySpecError(ValueError):
    """A strategy config that cannot be honoured. Raised at startup, not mid-session."""


@dataclass(slots=True)
class RuleSpec:
    rule: str
    mode: str = "required"
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def required(self) -> bool:
        return self.mode == "required"

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "mode": self.mode, "weight": self.weight,
                "params": dict(self.params)}


@dataclass(slots=True)
class StopSpec:
    method: str = "swing_or_atr"      # swing_or_atr | atr | fixed_ticks
    atr_period: int = 14
    atr_multiple: float = 1.2
    swing_buffer_atr: float = 0.25
    swing_width: int = 2
    fixed_ticks: float = 40.0
    min_ticks: float = 4.0

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__slots__}


@dataclass(slots=True)
class EntrySpec:
    stop: StopSpec = field(default_factory=StopSpec)
    tp1_r: float = 2.0
    tp2_r: float = 4.0
    min_rr: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return {"stop": self.stop.to_dict(), "tp1_r": self.tp1_r,
                "tp2_r": self.tp2_r, "min_rr": self.min_rr}


@dataclass(slots=True)
class SignalSpec:
    """Duplicate-signal protection.

    The same conditions stay true for several bars after a cross, so without
    this one setup emits a signal on every bar until it decays. Once a side has
    fired, it stays disarmed until the setup stops qualifying (or `rearm_bars`
    pass, whichever the config allows).
    """

    rearm_on_invalidation: bool = True   # re-arm when the setup stops qualifying
    rearm_bars: int = 0                  # 0 = only invalidation re-arms
    require_flat: bool = True            # no new signal while a position is open
    allow_reversal: bool = False         # a flip may signal against an open position

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__slots__}


@dataclass
class StrategySpec:
    name: str = "ema_vwap_rsi"
    rules: list[RuleSpec] = field(default_factory=list)
    entry: EntrySpec = field(default_factory=EntrySpec)
    signal: SignalSpec = field(default_factory=SignalSpec)
    min_score: float = 0.75
    structure_window: int = 60
    warmup_bars: int = 0                 # 0 = derive it from the rule params

    # ------------------------------------------------------------- building

    @classmethod
    def from_config(cls, data: dict[str, Any], presets: dict[str, list[dict]]) -> "StrategySpec":
        from .rules import RULES

        data = dict(data or {})
        name = data.get("name") or data.get("preset") or "ema_vwap_rsi"
        raw_rules = data.get("rules")
        if raw_rules is None:
            preset = data.get("preset") or name
            if preset not in presets:
                raise StrategySpecError(
                    f"strategy.preset {preset!r} is unknown; available: {sorted(presets)}. "
                    f"Alternatively list rules explicitly under strategy.rules."
                )
            raw_rules = presets[preset]
        if not isinstance(raw_rules, list) or not raw_rules:
            raise StrategySpecError("strategy.rules must be a non-empty list")

        rules: list[RuleSpec] = []
        seen: set[str] = set()
        for index, entry in enumerate(raw_rules):
            if isinstance(entry, str):
                entry = {"rule": entry}
            if not isinstance(entry, dict):
                raise StrategySpecError(f"strategy.rules[{index}] must be a name or a mapping")
            rule = entry.get("rule") or entry.get("name")
            if not rule:
                raise StrategySpecError(f"strategy.rules[{index}] is missing a 'rule' key")
            if rule not in RULES:
                raise StrategySpecError(
                    f"unknown rule {rule!r}; available: {sorted(RULES)}"
                )
            if rule in seen:
                raise StrategySpecError(f"rule {rule!r} is listed twice")
            seen.add(rule)
            mode = str(entry.get("mode", "required")).lower()
            if mode not in MODES:
                raise StrategySpecError(
                    f"rule {rule!r} has mode {mode!r}; expected one of {MODES}"
                )
            rules.append(RuleSpec(
                rule=rule,
                mode=mode,
                weight=float(entry.get("weight", 1.0)),
                params=dict(entry.get("params") or {}),
            ))

        if not any(r.required for r in rules if r.enabled):
            raise StrategySpecError(
                "at least one rule must be required - a strategy where every rule "
                "is advisory will fire on almost anything"
            )

        spec = cls(
            name=name,
            rules=rules,
            entry=_entry_from(data.get("entry") or {}),
            signal=_signal_from(data.get("signal") or {}),
            min_score=float(data.get("min_score", 0.75)),
            structure_window=int(data.get("structure_window", 60)),
            warmup_bars=int(data.get("warmup_bars", 0)),
        )
        if not 0.0 <= spec.min_score <= 1.0:
            raise StrategySpecError("strategy.min_score must be between 0 and 1")
        if spec.entry.tp1_r <= 0 or spec.entry.tp2_r <= 0:
            raise StrategySpecError("strategy.entry.tp1_r and tp2_r must be positive")
        return spec

    # -------------------------------------------------------------- queries

    def active(self) -> list[RuleSpec]:
        return [r for r in self.rules if r.enabled]

    def min_bars(self) -> int:
        """How much history the longest-lookback rule needs before it can speak."""
        if self.warmup_bars:
            return self.warmup_bars
        need = self.structure_window
        for r in self.active():
            for key in ("period", "slow", "fast", "lookback", "atr_period"):
                value = r.params.get(key)
                if isinstance(value, (int, float)):
                    need = max(need, int(value))
        need = max(need, self.entry.stop.atr_period)
        return need + 5

    def overlays(self) -> list[dict[str, Any]]:
        """Chart overlays the dashboard should draw, derived from the rules."""
        out: list[dict[str, Any]] = []
        for r in self.active():
            if r.rule in ("ema_stack", "ema_cross_fresh"):
                for key in ("fast", "slow"):
                    period = int(r.params.get(key, 9 if key == "fast" else 21))
                    if not any(o.get("period") == period for o in out):
                        out.append({"type": "ema", "period": period,
                                    "label": f"EMA{period}"})
            elif r.rule == "vwap_position":
                out.append({"type": "vwap", "label": "VWAP"})
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_score": self.min_score,
            "structure_window": self.structure_window,
            "warmup_bars": self.min_bars(),
            "rules": [r.to_dict() for r in self.rules],
            "entry": self.entry.to_dict(),
            "signal": self.signal.to_dict(),
        }


def _entry_from(data: dict[str, Any]) -> EntrySpec:
    stop_data = data.get("stop") or {}
    stop = StopSpec(**{k: v for k, v in stop_data.items() if k in StopSpec.__slots__})
    if stop.method not in ("swing_or_atr", "atr", "fixed_ticks"):
        raise StrategySpecError(f"unknown stop method {stop.method!r}")
    return EntrySpec(
        stop=stop,
        tp1_r=float(data.get("tp1_r", 2.0)),
        tp2_r=float(data.get("tp2_r", 4.0)),
        min_rr=float(data.get("min_rr", 1.5)),
    )


def _signal_from(data: dict[str, Any]) -> SignalSpec:
    return SignalSpec(**{k: v for k, v in data.items() if k in SignalSpec.__slots__})
