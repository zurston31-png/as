"""Dev/team wallet monitoring: auto-exit if the dev wallet's estimated share
of supply drops sharply after entry (a classic pre-rug signal).

"Dev wallet" is a heuristic — see estimate_dev_holder_pct() in
app/rugcheck/filters.py for how it's identified. This module just re-polls
the same security scanner periodically and compares against the percentage
captured at entry (Position.dev_wallet_pct_at_entry).
"""
import datetime as dt
import logging

from app.config import settings
from app.rugcheck.filters import estimate_dev_holder_pct
from app.rugcheck.goplus import fetch_token_security

logger = logging.getLogger(__name__)

_last_checked: dict[int, dt.datetime] = {}


async def check_dev_wallet_exit(position) -> str | None:
    if position.dev_wallet_pct_at_entry is None or not position.token_address:
        return None

    now = dt.datetime.now(dt.timezone.utc)
    last = _last_checked.get(position.id)
    if last and (now - last).total_seconds() < settings.DEV_WALLET_POLL_INTERVAL_SECONDS:
        return None
    _last_checked[position.id] = now

    try:
        data = await fetch_token_security(position.chain, position.token_address)
    except Exception:
        logger.warning("dev wallet check failed for %s", position.symbol, exc_info=True)
        return None

    current_pct = estimate_dev_holder_pct(data)
    if current_pct is None:
        return None

    drop = position.dev_wallet_pct_at_entry - current_pct
    if drop >= settings.DEV_WALLET_SELL_ALERT_PCT:
        return (
            f"dev/top wallet sold ~{drop * 100:.1f}% of supply "
            f"({position.dev_wallet_pct_at_entry * 100:.1f}% -> {current_pct * 100:.1f}%)"
        )
    return None
