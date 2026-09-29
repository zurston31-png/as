"""Stand-ins for the AI confirmation layer, used when measuring its value.

Three of them:

* `NullConfirmer` - the layer is absent. Every setup passes through.
* `CachedConfirmer` - wraps the real layer and memoises verdicts on disk, keyed
  by the decision itself. Re-running an arm costs nothing and, more importantly,
  every arm sees *identical* AI verdicts, so a difference between arms is the
  arm and not model variance.
* `RandomConfirmer` - vetoes at a given rate with no information. This is the
  control: an AI layer that only improves results as much as random vetoing
  does is not adding information, it is just trading less.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ..models import AIVerdict, Decision


def signal_key(signal) -> str:
    """A stable identity for one decision, so verdicts can be cached and replayed."""
    parts = [
        signal.symbol, signal.timeframe, signal.strategy, signal.side.value,
        (signal.decision_ts or signal.ts).isoformat(),
        f"{signal.entry:.6f}", f"{signal.stop:.6f}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


class NullConfirmer:
    """No AI layer at all."""

    name = "none"
    last_error = ""

    def __init__(self, *_args, **_kwargs) -> None:
        self.calls = 0

    async def confirm(self, *_args, **_kwargs) -> AIVerdict:
        self.calls += 1
        return AIVerdict(Decision.SKIPPED, 1.0, "AI layer not in this arm", source="none")

    def accepts(self, verdict: AIVerdict) -> tuple[bool, str]:
        return True, "no AI layer"


class RandomConfirmer:
    """Vetoes at `veto_rate` with no reference to the market. The control arm."""

    name = "random"
    last_error = ""

    def __init__(self, veto_rate: float, seed: int = 0) -> None:
        self.veto_rate = max(0.0, min(1.0, veto_rate))
        self.rng = random.Random(seed)
        self.calls = 0
        self.vetoes = 0

    async def confirm(self, *_args, **_kwargs) -> AIVerdict:
        self.calls += 1
        if self.rng.random() < self.veto_rate:
            self.vetoes += 1
            return AIVerdict(Decision.REJECT, 0.0, "randomised veto (control arm)",
                             source="random")
        return AIVerdict(Decision.CONFIRM, 1.0, "randomised pass (control arm)",
                         source="random")

    def accepts(self, verdict: AIVerdict) -> tuple[bool, str]:
        if verdict.decision is Decision.REJECT:
            return False, "randomised veto"
        return True, "randomised pass"


class CachedConfirmer:
    """The real layer, with verdicts memoised on disk by decision identity."""

    name = "cached"

    def __init__(self, inner, cache_path: str | Path) -> None:
        self.inner = inner
        self.path = Path(cache_path)
        self.cache: dict[str, dict[str, Any]] = self._load()
        self.hits = 0
        self.misses = 0

    @property
    def last_error(self) -> str:
        return getattr(self.inner, "last_error", "")

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, indent=2, sort_keys=True))
        except OSError:
            pass

    async def confirm(self, signal, ctx, candles, vision_note=None) -> AIVerdict:
        key = signal_key(signal)
        cached = self.cache.get(key)
        if cached:
            self.hits += 1
            return AIVerdict(
                decision=Decision(cached["decision"]),
                confidence=float(cached.get("confidence", 0.0)),
                rationale=cached.get("rationale", ""),
                risk_flags=list(cached.get("risk_flags", [])),
                source="cached",
                model=cached.get("model", ""),
            )
        self.misses += 1
        verdict = await self.inner.confirm(signal, ctx, candles, vision_note)
        # Never cache a failure - it would freeze a transient outage into the
        # dataset and every later arm would inherit it.
        if verdict.decision is not Decision.UNAVAILABLE:
            self.cache[key] = verdict.to_dict()
            self.save()
        return verdict

    def accepts(self, verdict: AIVerdict) -> tuple[bool, str]:
        return self.inner.accepts(verdict)

    def stats(self) -> dict[str, int]:
        return {"cached": len(self.cache), "hits": self.hits, "misses": self.misses}
