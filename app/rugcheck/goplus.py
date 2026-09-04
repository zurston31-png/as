"""GoPlus Security API client (https://docs.gopluslabs.io).

Covers both EVM chains (token_security/{chain_id}) and Solana
(solana/token_security). Response field names come from GoPlus's public
docs as of this build; GoPlus does evolve its schema occasionally, so if
`run_rug_checks` starts rejecting everything, check the raw response saved
in `RugCheckResult` / logs against the current GoPlus docs first.
"""
import logging


from app.config import settings
from app.services import http

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


async def _security(url: str, token_address: str, key: str) -> dict:
    """One GoPlus lookup through the shared retry/health wrapper.

    Raises LookupFailed rather than returning {} when the API cannot be
    reached. The caller distinguishes "no record" from "could not check"
    in the audit trail, and collapsing the two would let a provider outage
    read as a clean bill of health for the token.
    """
    data = await http.get_json(
        url,
        params={"contract_addresses": token_address},
        headers=_headers(),
        label=f"goplus {token_address}",
        service="goplus",
    )
    if data is None:
        raise http.LookupFailed(f"GoPlus did not answer for {token_address}")
    return (data.get("result") or {}).get(key, {})


async def fetch_evm_token_security(chain: str, token_address: str) -> dict:
    chain_id = EVM_CHAIN_IDS.get(chain.lower(), str(settings.EVM_CHAIN_ID))
    return await _security(
        f"{settings.GOPLUS_API_BASE}/token_security/{chain_id}",
        token_address, token_address.lower(),
    )


async def fetch_solana_token_security(token_address: str) -> dict:
    return await _security(
        f"{settings.GOPLUS_API_BASE}/solana/token_security",
        token_address, token_address,
    )


async def fetch_token_security(chain: str, token_address: str) -> dict:
    if chain.lower() == "solana":
        return await fetch_solana_token_security(token_address)
    return await fetch_evm_token_security(chain, token_address)
