"""Pydantic schema for the inbound TradingView webhook payload.

Matches the JSON built by `pine/memecoin_signal_strategy.pine`'s
`buildPayload()` function — keep the two in sync if you change one.
"""
import datetime as dt
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
