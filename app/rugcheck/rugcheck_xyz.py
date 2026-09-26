"""RugCheck.xyz client — Solana token safety reports.

Used in preference to GoPlus on Solana for two reasons: it indexes new
launches (GoPlus frequently has no record of them, and new launches are
exactly what a memecoin bot trades), and it publishes its own risk
assessment rather than only raw fields, so the bot can defer to a
Solana specialist instead of re-deriving LP-lock and insider analysis.

API: https://api.rugcheck.xyz/v1/tokens/{mint}/report
"""
import logging


from app.config import settings
from app.services import http

logger = logging.getLogger(__name__)

# Sentinel for "the API answered, and it has never heard of this token".
# A plain {} could not be told apart from a transport failure.
_NOT_INDEXED = object()


# Send an explicit User-Agent. Some public APIs sit behind a WAF that
# throttles or blocks default library agents, and the diagnostic script that
# successfully reached this endpoint identified itself, so match it rather
# than leaving httpx's default as an unexamined difference.
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "memecoin-trading-bot/1.0",
}


async def fetch_token_report(mint: str) -> dict:
    """Return RugCheck's report, or {} when it has no usable record.

    A 404 means the token is not indexed and is reported as "no record"
    rather than raised, so callers can distinguish that from a transport or
    server failure.
    """
    url = f"{settings.RUGCHECK_API_BASE}/tokens/{mint}/report"
    data = await http.get_json(
        url, headers=HEADERS, timeout=20,
        label=f"rugcheck.xyz {mint}", service="rugcheck_xyz",
        on_status={404: _NOT_INDEXED},
    )
    if data is _NOT_INDEXED:
        logger.info("RugCheck has no record of %s", mint)
        return {}
    if data is None:
        # Distinct from the 404 above: that was an answer, this is silence.
        raise http.LookupFailed(f"RugCheck did not answer for {mint}")

    if not isinstance(data, dict):
        logger.warning("RugCheck returned a %s for %s, expected an object", type(data).__name__, mint)
        return {}
    # Error-shaped bodies are returned with a 200 by some endpoints.
    if data.get("error") or data.get("message") and len(data) <= 2:
        logger.info("RugCheck reported no usable data for %s: %s", mint, data)
        return {}
    return data
