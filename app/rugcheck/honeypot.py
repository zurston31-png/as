"""honeypot.is client — EVM-only honeypot simulation
(https://honeypot.is / api.honeypot.is). Solana honeypot detection is
covered instead via GoPlus's `is_honeypot` / transfer-fee fields in
goplus.py, since honeypot.is does not support Solana.
"""
import logging


from app.config import settings
from app.services import http

logger = logging.getLogger(__name__)


async def check_honeypot_evm(token_address: str, chain_id: int | None = None) -> dict:
    url = f"{settings.HONEYPOT_API_BASE}/IsHoneypot"
    params = {"address": token_address}
    if chain_id:
        params["chainID"] = chain_id
    data = await http.get_json(
        url, params=params, label=f"honeypot.is {token_address}", service="honeypot",
    )
    if data is None:
        # Raised, not returned as {}. An empty result would read as "not a
        # honeypot" to the caller, which is the most dangerous possible
        # interpretation of a failed honeypot check.
        raise http.LookupFailed(f"honeypot.is did not answer for {token_address}")
    return data
