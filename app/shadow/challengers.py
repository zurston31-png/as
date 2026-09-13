"""The registry of strategies running beside the champion.

A challenger is a set of parameter overrides, not a fork of the code. That
constraint is what keeps the comparison fair: both arms go through the
same scoring function, the same fill model and the same regime
classifier, so a measured difference is attributable to the parameters
rather than to two code paths that drifted apart.

Challengers are declared in configuration and are inert by default. An
empty registry means the shadow system records champion decisions only,
which is still useful - it is the baseline every later comparison needs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.signals.scoring import DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)

CHAMPION_ID = "champion"


@dataclass(frozen=True)
class Challenger:
    """One parameter variant evaluated alongside the champion."""
    strategy_id: str
    description: str = ""
    # Only the factors being changed need listing; the rest come from the
    # champion's map, so a challenger reads as a diff rather than as a
    # full copy that silently drifts when the champion's weights change.
    weight_overrides: dict[str, float] = field(default_factory=dict)
    min_score_to_enter: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    def weights(self) -> dict[str, float]:
        """The full weight map this challenger scores with.

        Renormalisation is left to the scorer, which divides by the sum of
        whatever it is given - so an override that changes the total does
        not silently rescale every score against the champion's.
        """
        merged = dict(DEFAULT_WEIGHTS)
        for name, value in self.weight_overrides.items():
            if name not in merged:
                raise KeyError(
                    f"{self.strategy_id} overrides unknown factor {name!r} - a typo here "
                    "would otherwise add a weight the scorer never reads"
                )
            merged[name] = value
        return merged

    def threshold(self) -> float:
        return (
            self.min_score_to_enter
            if self.min_score_to_enter is not None
            else settings.MIN_SIGNAL_SCORE_TO_ENTER
        )

    def as_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "description": self.description,
            "weight_overrides": dict(self.weight_overrides),
            "min_score_to_enter": self.threshold(),
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
        }


def _parse(raw: str) -> list[Challenger]:
    """Build challengers from the JSON in SHADOW_CHALLENGERS.

    A malformed entry disables that challenger and logs why, rather than
    taking the bot down. But it is never silently skipped: a challenger
    that quietly failed to load would leave a comparison looking complete
    while missing an arm.
    """
    try:
        entries = json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.error("SHADOW_CHALLENGERS is not valid JSON (%s) - no challengers loaded", exc)
        return []

    if not isinstance(entries, list):
        logger.error("SHADOW_CHALLENGERS must be a JSON list - no challengers loaded")
        return []

    loaded: list[Challenger] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("strategy_id"):
            logger.error("skipping a challenger with no strategy_id: %r", entry)
            continue
        strategy_id = str(entry["strategy_id"])
        if strategy_id == CHAMPION_ID:
            logger.error("a challenger may not be called %r - skipping", CHAMPION_ID)
            continue
        if strategy_id in seen:
            logger.error("duplicate challenger id %r - skipping the second", strategy_id)
            continue
        try:
            challenger = Challenger(
                strategy_id=strategy_id,
                description=str(entry.get("description", "")),
                weight_overrides={
                    str(k): float(v) for k, v in (entry.get("weight_overrides") or {}).items()
                },
                min_score_to_enter=(
                    float(entry["min_score_to_enter"])
                    if entry.get("min_score_to_enter") is not None else None
                ),
                stop_loss_pct=(
                    float(entry["stop_loss_pct"])
                    if entry.get("stop_loss_pct") is not None else None
                ),
                take_profit_pct=(
                    float(entry["take_profit_pct"])
                    if entry.get("take_profit_pct") is not None else None
                ),
            )
            challenger.weights()       # fail now, not on the first opportunity
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("challenger %s is misconfigured (%s) - skipping", strategy_id, exc)
            continue
        loaded.append(challenger)
        seen.add(strategy_id)
    return loaded


def enabled() -> list[Challenger]:
    """Challengers currently configured. Empty is a valid, quiet default."""
    if not settings.SHADOW_ENABLED:
        return []
    raw = (settings.SHADOW_CHALLENGERS or "").strip()
    return _parse(raw) if raw else []
