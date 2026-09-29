"""The AI confirmation layer.

Its job is narrow on purpose. The rules engine has already decided there *is* a
setup; the model only gets to say "confirm", "wait" or "reject", and it is fed
structured numbers rather than a picture of a chart. That keeps the model doing
what it is good at - weighing context that is awkward to encode as a rule, like
a setup that is technically valid but sitting under a session high with three
bars of stalling volume - and keeps it away from reading pixel values.

The layer can only ever *subtract* trades. A "confirm" on a setup the rules
rejected is discarded before it gets here.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from ..config import Config
from ..models import AIVerdict, Candle, Decision, StrategyVerdict, TradeSignal
from ..strategy.context import MarketContext
from .client import describe_error, get_client

log = logging.getLogger(__name__)

SYSTEM = """You are the confirmation layer of a rules-based intraday trading bot.

A deterministic strategy engine has already found a setup that satisfies every \
one of its hard rules. You are the second opinion before a human sees an alert. \
You cannot create trades and you cannot change the levels - you can only confirm \
the setup, tell it to wait, or reject it.

You receive structured market data, never an image. Trust the numbers you are \
given; do not invent levels, prices, or indicator values that are not present.

Decide as follows:
- "confirm": the context genuinely supports the setup. The rules agree and \
nothing in the wider picture contradicts them.
- "wait": the setup is plausible but something is unresolved - price is extended \
into a level, the trigger bar is weak, volume is thin, or the move needs one \
more bar of confirmation.
- "reject": the context actively contradicts the setup - it is fighting the \
higher-level structure, the risk:reward is illusory because a major level sits \
between entry and target, or the data looks stale or inconsistent.

Be strict. A missed trade costs nothing; a bad trade costs real money. When the \
evidence is ambiguous, prefer "wait" over "confirm".

Set confidence to your genuine probability that this setup is worth taking, from \
0.0 to 1.0. Do not anchor on 0.5. List concrete risk_flags (short snake_case \
tags such as "extended_from_vwap", "thin_volume", "into_prior_day_high", \
"stale_data"); use an empty list when there are none. Keep the rationale under \
60 words and specific to the numbers you were given."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["confirm", "wait", "reject"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "confidence", "rationale", "risk_flags"],
    "additionalProperties": False,
}


class ConfirmationLayer:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.last_error: str = ""

    @property
    def enabled(self) -> bool:
        return self.cfg.ai.enabled

    async def confirm(
        self,
        signal: TradeSignal,
        ctx: MarketContext,
        recent: list[Candle],
        vision_note: Optional[str] = None,
    ) -> AIVerdict:
        if not self.cfg.ai.enabled:
            return AIVerdict(Decision.SKIPPED, 1.0, "AI layer disabled in config", source="none")

        client = get_client(self.cfg.ai.timeout_seconds)
        if client is None:
            self.last_error = "anthropic SDK not installed"
            return AIVerdict(Decision.UNAVAILABLE, 0.0, self.last_error, source="none")

        prompt = self._prompt(signal, ctx, recent, vision_note)
        started = time.monotonic()
        try:
            response = await client.messages.create(
                model=self.cfg.ai.model,
                max_tokens=self.cfg.ai.max_tokens,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": self.cfg.ai.effort,
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
            )
        except Exception as exc:  # noqa: BLE001 - never let this break the loop
            self.last_error = describe_error(exc)
            log.warning("AI confirmation failed: %s", exc)
            return AIVerdict(Decision.UNAVAILABLE, 0.0, f"confirmation call failed: {self.last_error}",
                             source="none", model=self.cfg.ai.model)

        latency = int((time.monotonic() - started) * 1000)

        if getattr(response, "stop_reason", None) == "refusal":
            self.last_error = "model declined to answer"
            return AIVerdict(Decision.UNAVAILABLE, 0.0, self.last_error, source="none",
                             latency_ms=latency, model=self.cfg.ai.model)

        try:
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
        except (StopIteration, json.JSONDecodeError, AttributeError) as exc:
            self.last_error = f"unparseable verdict: {type(exc).__name__}"
            return AIVerdict(Decision.UNAVAILABLE, 0.0, self.last_error, source="none",
                             latency_ms=latency, model=self.cfg.ai.model)

        self.last_error = ""
        return AIVerdict(
            decision=Decision(data["decision"]),
            confidence=float(data.get("confidence", 0.0)),
            rationale=str(data.get("rationale", "")).strip(),
            risk_flags=[str(f) for f in data.get("risk_flags", [])],
            source="structured",
            latency_ms=latency,
            model=self.cfg.ai.model,
        )

    def accepts(self, verdict: AIVerdict) -> tuple[bool, str]:
        """Apply the configured policy to a verdict. Returns (allow, reason)."""
        ai = self.cfg.ai
        if verdict.decision is Decision.SKIPPED:
            return True, "AI layer disabled"
        if verdict.decision is Decision.UNAVAILABLE:
            if ai.required:
                return False, f"AI confirmation required but unavailable ({verdict.rationale})"
            return True, "AI unavailable - falling back to rules only"
        if verdict.decision is Decision.REJECT:
            return False, f"AI rejected: {verdict.rationale}"
        if verdict.decision is Decision.WAIT:
            return False, f"AI says wait: {verdict.rationale}"
        if verdict.confidence < ai.confirm_min_confidence:
            return False, (
                f"AI confidence {verdict.confidence:.2f} below the "
                f"{ai.confirm_min_confidence:.2f} threshold"
            )
        return True, f"AI confirmed ({verdict.confidence:.2f})"

    # -------------------------------------------------------------- prompt

    def _prompt(
        self,
        signal: TradeSignal,
        ctx: MarketContext,
        recent: list[Candle],
        vision_note: Optional[str],
    ) -> str:
        n = self.cfg.ai.candles_in_prompt
        bars = [
            {
                "t": c.ts.isoformat(),
                "o": round(c.open, 4),
                "h": round(c.high, 4),
                "l": round(c.low, 4),
                "c": round(c.close, 4),
                "v": round(c.volume, 2),
            }
            for c in recent[-n:]
        ]
        verdict: StrategyVerdict | None = signal.strategy_verdict
        payload = {
            "proposed_trade": {
                "side": signal.side.value,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "entry": signal.entry,
                "stop": signal.stop,
                "tp1": signal.tp1,
                "tp2": signal.tp2,
                "stop_distance": round(signal.stop_distance, 4),
                "rr_to_tp1": signal.rr1,
                "rr_to_tp2": signal.rr2,
            },
            "indicators": ctx.to_dict(),
            "strategy": {
                "preset": signal.strategy,
                "score": round(verdict.score, 3) if verdict else None,
                "checks": [
                    {"rule": c.name, "passed": c.passed, "detail": c.detail, "required": c.required}
                    for c in (verdict.checks if verdict else [])
                ],
            },
            "recent_candles_oldest_first": bars,
        }
        text = (
            "Evaluate this setup.\n\n```json\n"
            + json.dumps(payload, indent=2, default=str)
            + "\n```"
        )
        if vision_note:
            text += (
                "\n\nA screen-capture reading of the user's chart is attached below as a "
                "secondary, lower-confidence source. Use it only to catch a contradiction "
                "with the structured data (for example a stale feed or a different symbol on "
                "screen). Where the two disagree, say so in risk_flags.\n\n"
                f"{vision_note}"
            )
        return text
