"""honeypot.is client — EVM-only honeypot simulation
(https://honeypot.is / api.honeypot.is). Solana honeypot detection is
covered instead via GoPlus's `is_honeypot` / transfer-fee fields in
goplus.py, since honeypot.is does not support Solana.
"""
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def check_honeypot_evm(token_address: str, chain_id: int | None = None) -> dict:
    url = f"{settings.HONEYPOT_API_BASE}/IsHoneypot"
    params = {"address": token_address}
    if chain_id:
        params["chainID"] = chain_id
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
