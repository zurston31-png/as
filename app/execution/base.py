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
    # --- execution-cost accounting ---
    # What the fill cost beyond the mid price. Populated by the paper
    # engine from app/execution/fill_model.py; live backends leave these
    # None rather than guessing, since a real fill's cost breakdown isn't
    # recoverable from a swap receipt without extra work. Recorded on the
    # Trade row so "total fees paid" and "average slippage" are answerable
    # instead of being invisible inside the P&L.
    fee_usd: float | None = None
    execution_cost_pct: float | None = None
    fill_delay_seconds: float | None = None


class ExecutionClient(ABC):
    @abstractmethod
    async def get_price(self, instrument: str) -> float | None: ...

    @abstractmethod
    async def buy(self, instrument: str, usd_amount: float, slippage_bps: int) -> SwapResult: ...

    @abstractmethod
    async def sell(self, instrument: str, qty: float, slippage_bps: int) -> SwapResult: ...
