"""Common interface every execution backend implements."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SwapResult:
    success: bool
    filled_qty: float = 0.0
    avg_price: float = 0.0
    tx_hash: str | None = None
    error: str | None = None


class ExecutionClient(ABC):
    @abstractmethod
    async def get_price(self, instrument: str) -> float | None: ...

    @abstractmethod
    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult: ...

    @abstractmethod
    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult: ...
