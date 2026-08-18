"""Experimental EVM execution backend (1inch aggregator).

This is intentionally a scaffold, not a finished implementation. Building a
production-grade EVM signer — nonce management, EIP-1559 gas pricing,
mempool resubmission/replacement, MEV-aware submission — is a project in
its own right, and getting it wrong on-chain is expensive and irreversible.

Price lookups (get_price) work today via DexScreener and are safe to use
for paper trading / monitoring on EVM chains. `buy`/`sell` deliberately
refuse to execute until you implement transaction signing for your target
chain(s). Review app/execution/jupiter.py for the equivalent Solana pattern
if you want to complete this for EVM before enabling LIVE_TRADING with
CHAIN=evm.
"""
from app.execution.base import ExecutionClient, SwapResult
from app.services import price_feed

NOT_IMPLEMENTED_MSG = (
    "EVM live execution is not implemented in this build. Wire up transaction "
    "signing in app/execution/evm.py before enabling LIVE_TRADING with "
    "CHAIN=evm / EXECUTION_BACKEND=evm_1inch. Until then, use CHAIN=solana "
    "(EXECUTION_BACKEND=jupiter) or EXECUTION_BACKEND=cex for live trading."
)


class EvmExecutionClient(ExecutionClient):
    async def get_price(self, instrument: str) -> float | None:
        return await price_feed.get_price_usd(instrument)

    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult:
        return SwapResult(success=False, error=NOT_IMPLEMENTED_MSG)

    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult:
        return SwapResult(success=False, error=NOT_IMPLEMENTED_MSG)
