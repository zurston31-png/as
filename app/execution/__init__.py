"""Execution backend factory.

Safety invariant: whenever LIVE_TRADING is false, PaperExecutionClient is
returned NO MATTER WHAT EXECUTION_BACKEND is set to. Live backends are only
ever constructed when the deployer has explicitly flipped LIVE_TRADING=true.
"""
from app.config import settings
from app.execution.base import ExecutionClient
from app.execution.paper import PaperExecutionClient


def get_execution_client() -> ExecutionClient:
    if not settings.LIVE_TRADING or settings.EXECUTION_BACKEND == "paper":
        return PaperExecutionClient()

    backend = settings.EXECUTION_BACKEND
    if backend == "jupiter":
        from app.execution.jupiter import JupiterExecutionClient

        return JupiterExecutionClient()
    if backend == "cex":
        from app.execution.cex import CexExecutionClient

        return CexExecutionClient()
    if backend == "evm_1inch":
        from app.execution.evm import EvmExecutionClient

        return EvmExecutionClient()

    raise ValueError(f"unknown EXECUTION_BACKEND: {backend!r}")
