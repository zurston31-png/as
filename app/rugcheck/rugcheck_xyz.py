"""RugCheck.xyz client — Solana token safety reports.

Used in preference to GoPlus on Solana for two reasons: it indexes new
launches (GoPlus frequently has no record of them, and new launches are
exactly what a memecoin bot trades), and it publishes its own risk
assessment rather than only raw fields, so the bot can defer to a
Solana specialist instead of re-deriving LP-lock and insider analysis.

API: https://api.rugcheck.xyz/v1/tokens/{mint}/report
"""
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def fetch_token_report(mint: str) -> dict:
    url = f"{settings.RUGCHECK_API_BASE}/tokens/{mint}/report"
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    return data if isinstance(data, dict) else {}
