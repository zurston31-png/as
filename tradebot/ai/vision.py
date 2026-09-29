"""Reading the chart off the screen - the backup layer.

What this returns is explicitly lower-trust than the structured feed. It exists
to answer questions the feed can't when the feed is broken: is the chart still
printing, is it even the right symbol, roughly where is price. The orchestrator
uses it to *contradict* structured data, never to originate a trade.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from ..config import Config
from ..vision.capture import ScreenCapture
from .client import describe_error, get_client

log = logging.getLogger(__name__)

SYSTEM = """You read trading charts from screenshots for a trading bot's backup \
data path. The bot already has precise numeric market data; your reading is a \
sanity check against it, not a replacement.

Report only what is legibly on screen. If the image is not a price chart, is too \
small to read, or is ambiguous, set readable to false and say why in notes - \
that is a useful answer, not a failure. Never estimate a price you cannot \
actually read off an axis or a price label; use null instead.

Keep key_levels to at most six horizontal levels that are visibly significant \
(prior highs/lows, a clear range boundary, an obvious round number being \
respected). Keep patterns to short tags you can actually see, such as \
"higher_highs", "range_bound", "long_upper_wicks", "gap_up"."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "readable": {"type": "boolean"},
        "symbol": {"type": ["string", "null"]},
        "timeframe": {"type": ["string", "null"]},
        "last_price": {"type": ["number", "null"]},
        "trend": {"type": "string", "enum": ["bullish", "bearish", "range", "unclear"]},
        "key_levels": {"type": "array", "items": {"type": "number"}},
        "patterns": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["readable", "symbol", "timeframe", "last_price", "trend",
                 "key_levels", "patterns", "notes"],
    "additionalProperties": False,
}


class ChartVision:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.capture = ScreenCapture(config.vision)
        self.last_reading: Optional[dict[str, Any]] = None
        self.last_error: str = ""
        self._last_call: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.cfg.vision.enabled and self.cfg.vision.mode != "off"

    def throttled(self) -> bool:
        return (time.monotonic() - self._last_call) < self.cfg.vision.min_interval_seconds

    async def read(self, force: bool = False) -> Optional[dict[str, Any]]:
        """Grab the screen and return a structured reading, or None."""
        if not self.enabled:
            return None
        if self.throttled() and not force:
            return self.last_reading
        client = get_client(self.cfg.ai.timeout_seconds)
        if client is None:
            self.last_error = "anthropic SDK not installed"
            return None
        cap = self.capture.grab()
        if cap is None:
            self.last_error = "screen capture unavailable"
            return None

        self._last_call = time.monotonic()
        try:
            response = await client.messages.create(
                model=self.cfg.vision.model,
                max_tokens=self.cfg.vision.max_tokens,
                system=SYSTEM,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": cap.base64},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Read this chart. For reference the bot believes it is trading "
                                f"{self.cfg.market.symbol} on the {self.cfg.market.timeframe} "
                                f"timeframe - report what you actually see, which may differ."
                            ),
                        },
                    ],
                }],
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = describe_error(exc)
            log.warning("vision read failed: %s", exc)
            return None

        if getattr(response, "stop_reason", None) == "refusal":
            self.last_error = "model declined the image"
            return None

        try:
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
        except (StopIteration, json.JSONDecodeError, AttributeError):
            self.last_error = "unparseable vision reading"
            return None

        data["captured_at"] = cap.ts.isoformat()
        data["capture_size"] = [cap.width, cap.height]
        self.last_error = ""
        self.last_reading = data
        return data

    # ----------------------------------------------------------- reconcile

    def disagreement(self, reading: Optional[dict[str, Any]], price: float) -> Optional[str]:
        """Does the screen contradict the structured feed? Returns a reason or None."""
        if not reading or not reading.get("readable"):
            return None
        tolerance = self.cfg.vision.price_tolerance_pct
        screen_price = reading.get("last_price")
        if isinstance(screen_price, (int, float)) and price:
            drift = abs(screen_price - price) / price * 100.0
            if drift > tolerance:
                return (
                    f"screen shows {screen_price:g} but the feed says {price:g} "
                    f"({drift:.2f}% apart, tolerance {tolerance:g}%)"
                )
        screen_symbol = (reading.get("symbol") or "").strip().upper()
        want = self.cfg.market.symbol.strip().upper()
        if screen_symbol and want and not _symbols_match(screen_symbol, want):
            return f"chart on screen is {screen_symbol}, the bot is trading {want}"
        return None

    def note_for_prompt(self, reading: Optional[dict[str, Any]]) -> Optional[str]:
        if not reading:
            return None
        return "```json\n" + json.dumps(reading, indent=2, default=str) + "\n```"

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.cfg.vision.mode,
            "capture": ScreenCapture.availability(),
            "last_error": self.last_error,
            "last_reading": self.last_reading,
            "throttled": self.throttled(),
        }


def _symbols_match(a: str, b: str) -> bool:
    """Tolerate the ways the same instrument gets written across platforms."""
    def norm(s: str) -> str:
        s = s.upper()
        for ch in " -_/!1":
            s = s.replace(ch, "")
        return s.split(":")[-1]
    na, nb = norm(a), norm(b)
    return na in nb or nb in na
