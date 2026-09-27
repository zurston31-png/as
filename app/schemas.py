"""Pydantic schema for the inbound TradingView webhook payload.

Matches the JSON built by `pine/memecoin_signal_strategy.pine`'s
`buildPayload()` function — keep the two in sync if you change one.
"""
import datetime as dt
import math
from typing import Optional

from pydantic import BaseModel, field_validator


class TradingViewAlert(BaseModel):
    secret: str
    symbol: str
    token_address: Optional[str] = None
    chain: str = "solana"
    signal: str                       # "buy" | "sell" (case-insensitive)
    price: float
    time: Optional[str] = None        # unix ms (as sent by the Pine script) or ISO-8601
    rsi: Optional[float] = None
    ema9: Optional[float] = None
    ema21: Optional[float] = None
    volume: Optional[float] = None
    volume_sma: Optional[float] = None
    breakout_level: Optional[float] = None

    @field_validator("signal")
    @classmethod
    def normalize_signal(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator(
        "rsi", "ema9", "ema21", "volume", "volume_sma", "breakout_level", mode="before",
    )
    @classmethod
    def non_finite_is_absent(cls, v):
        """`NaN` and `Infinity` mean "not measured", so record them as absent.

        Pine's `str.tostring()` renders an `na` series value as a bare `NaN`,
        which stdlib json accepts (see app/webhook_debug.parse_body - the
        alert is deliberately not thrown away over an indicator that has yet
        to warm up). Letting that through as a float would put a NaN in a
        numeric column, where it reads as a measurement that was taken and
        came out unrepresentable, rather than one that was never available.
        """
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    @field_validator("price")
    @classmethod
    def price_must_be_finite(cls, v: float) -> float:
        """Unlike the indicators, an unusable price fails the alert closed.

        Everything downstream - position size, the stop, the take-profit - is
        computed from it, so a NaN price would silently produce NaN levels
        instead of no trade.
        """
        if not math.isfinite(v):
            raise ValueError("price must be a finite number")
        return v

    @field_validator("time", mode="before")
    @classmethod
    def accept_numeric_time(cls, v):
        """Pine sends `"time":1766248800000` - a JSON number, not a string.

        Pydantic v2 does not coerce int to str, so an unquoted timestamp
        failed validation and FastAPI answered 422 before the handler ran.
        Coercing here keeps `parsed_time()` working on either form.
        """
        if isinstance(v, bool) or v is None:
            return None if v is None else str(v)
        if isinstance(v, (int, float)):
            return repr(v) if isinstance(v, float) else str(v)
        return v

    def parsed_time(self) -> Optional[dt.datetime]:
        if self.time is None:
            return None
        try:
            ms = float(self.time)
            return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc)
        except (TypeError, ValueError):
            pass
        try:
            return dt.datetime.fromisoformat(str(self.time).replace("Z", "+00:00"))
        except ValueError:
            return None
