"""Execution-layer safety invariants.

These guard the two ways the bot could move real money by accident: the
paper engine not being forced when live trading is off, and a live backend
whose accounting is known to be wrong staying reachable.
"""
import pytest

from app.config import settings
from app.execution import get_execution_client
from app.execution.paper import PaperExecutionClient

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _restore_settings():
    live, backend = settings.LIVE_TRADING, settings.EXECUTION_BACKEND
    yield
    settings.LIVE_TRADING, settings.EXECUTION_BACKEND = live, backend


@pytest.mark.parametrize("backend", ["jupiter", "cex", "evm_1inch", "paper"])
def test_paper_engine_forced_whenever_live_trading_is_off(backend):
    """The safety switch must win over EXECUTION_BACKEND, always."""
    settings.LIVE_TRADING = False
    settings.EXECUTION_BACKEND = backend
    assert isinstance(get_execution_client(), PaperExecutionClient)


async def test_jupiter_refuses_to_submit_while_the_decimals_bug_stands():
    """avg_price mixes USDC base units with the token's, so entry price — and
    therefore the stop-loss and take-profit derived from it — is wrong for any
    token that does not have exactly 6 decimals. Submission stays closed."""
    from app.execution.jupiter import DECIMALS_BUG_MSG, JupiterExecutionClient

    settings.LIVE_TRADING = True
    client = JupiterExecutionClient()
    result = await client._execute_swap({"inAmount": "20000000", "outAmount": "20000000000"})

    assert not result.success
    assert "disabled" in result.error
    assert result.error == DECIMALS_BUG_MSG


async def test_jupiter_refuses_even_with_a_private_key_configured(monkeypatch):
    """The guard must not be bypassable by supplying credentials."""
    from app.execution.jupiter import JupiterExecutionClient

    settings.LIVE_TRADING = True
    monkeypatch.setattr(settings, "SOLANA_PRIVATE_KEY", "a" * 88)
    result = await JupiterExecutionClient()._execute_swap({"inAmount": "1", "outAmount": "1"})
    assert not result.success


async def test_evm_backend_refuses_live_execution():
    from app.execution.evm import EvmExecutionClient

    settings.LIVE_TRADING = True
    client = EvmExecutionClient()
    buy = await client.buy("0x" + "1" * 40, 20.0, 150)
    sell = await client.sell("0x" + "1" * 40, 1.0, 150)
    assert not buy.success and "not implemented" in buy.error
    assert not sell.success


async def test_paper_engine_prices_and_sizes_consistently(monkeypatch):
    """qty * fill_price must equal the USD sized by the risk manager, or the
    cash ledger and the position value drift apart."""
    from app.execution import paper

    async def fake_price(_addr):
        return 0.001
    monkeypatch.setattr(paper.price_feed, "get_price_usd", fake_price)

    client = PaperExecutionClient()
    result = await client.buy("SomeToken", 20.0, 150)
    assert result.success
    assert result.filled_qty * result.avg_price == pytest.approx(20.0)
    # Fills are deliberately pessimistic, never better than the mid price.
    assert result.avg_price > 0.001


async def test_paper_sell_fills_below_mid(monkeypatch):
    from app.execution import paper

    async def fake_price(_addr):
        return 0.001
    monkeypatch.setattr(paper.price_feed, "get_price_usd", fake_price)

    result = await PaperExecutionClient().sell("SomeToken", 1000.0, 150)
    assert result.success
    assert result.avg_price < 0.001


async def test_paper_engine_fails_closed_without_a_price(monkeypatch):
    from app.execution import paper

    async def no_price(_addr):
        return None
    monkeypatch.setattr(paper.price_feed, "get_price_usd", no_price)

    result = await PaperExecutionClient().buy("SomeToken", 20.0, 150)
    assert not result.success
    assert result.filled_qty == 0
