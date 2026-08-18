"""GoPlus Security API client (https://docs.gopluslabs.io).

Covers both EVM chains (token_security/{chain_id}) and Solana
(solana/token_security). Response field names come from GoPlus's public
docs as of this build; GoPlus does evolve its schema occasionally, so if
`run_rug_checks` starts rejecting everything, check the raw response saved
in `RugCheckResult` / logs against the current GoPlus docs first.
"""
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

EVM_CHAIN_IDS = {
    "ethereum": "1",
    "eth": "1",
    "bsc": "56",
    "polygon": "137",
    "arbitrum": "42161",
    "base": "8453",
    "avalanche": "43114",
    "optimism": "10",
}


def _headers() -> dict:
    headers = {}
    if settings.GOPLUS_API_KEY:
        headers["Authorization"] = settings.GOPLUS_API_KEY
    return headers


async def fetch_evm_token_security(chain: str, token_address: str) -> dict:
    chain_id = EVM_CHAIN_IDS.get(chain.lower(), str(settings.EVM_CHAIN_ID))
    url = f"{settings.GOPLUS_API_BASE}/token_security/{chain_id}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params={"contract_addresses": token_address}, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
    return (data.get("result") or {}).get(token_address.lower(), {})


async def fetch_solana_token_security(token_address: str) -> dict:
    url = f"{settings.GOPLUS_API_BASE}/solana/token_security"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params={"contract_addresses": token_address}, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
    return (data.get("result") or {}).get(token_address, {})


async def fetch_token_security(chain: str, token_address: str) -> dict:
    if chain.lower() == "solana":
        return await fetch_solana_token_security(token_address)
    return await fetch_evm_token_security(chain, token_address)
