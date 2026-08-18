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
    live, backend, acked = settings.LIVE_TRADING, settings.EXECUTION_BACKEND, settings.LIVE_EXECUTION_ACKNOWLEDGED
    yield
    settings.LIVE_TRADING, settings.EXECUTION_BACKEND = live, backend
    settings.LIVE_EXECUTION_ACKNOWLEDGED = acked


@pytest.mark.parametrize("backend", ["jupiter", "cex", "evm_1inch", "paper"])
def test_paper_engine_forced_whenever_live_trading_is_off(backend):
    """The safety switch must win over EXECUTION_BACKEND, always."""
    settings.LIVE_TRADING = False
    settings.EXECUTION_BACKEND = backend
    assert isinstance(get_execution_client(), PaperExecutionClient)


async def test_jupiter_refuses_to_submit_without_the_second_acknowledgement_flag():
    """LIVE_TRADING alone must not be enough to arm Jupiter submission - the
    sign-and-submit path has never been exercised against a funded wallet
    from this codebase's own tests, and LIVE_EXECUTION_ACKNOWLEDGED is the
    deliberate second decision that's required on top."""
    from app.execution.jupiter import LIVE_EXECUTION_UNACKNOWLEDGED_MSG, JupiterExecutionClient

    settings.LIVE_TRADING = True
    settings.LIVE_EXECUTION_ACKNOWLEDGED = False
    client = JupiterExecutionClient()
    result = await client._execute_swap(
        {"inAmount": "20000000", "outAmount": "20000000000"}, input_decimals=6, output_decimals=9,
    )

    assert not result.success
    assert result.error == LIVE_EXECUTION_UNACKNOWLEDGED_MSG


async def test_jupiter_refuses_even_with_a_private_key_configured(monkeypatch):
    """The guard must not be bypassable by supplying credentials alone."""
    from app.execution.jupiter import JupiterExecutionClient

    settings.LIVE_TRADING = True
    settings.LIVE_EXECUTION_ACKNOWLEDGED = False
    monkeypatch.setattr(settings, "SOLANA_PRIVATE_KEY", "a" * 88)
    result = await JupiterExecutionClient()._execute_swap(
        {"inAmount": "1", "outAmount": "1"}, input_decimals=6, output_decimals=6,
    )
    assert not result.success


async def test_jupiter_fails_closed_when_live_deps_are_missing(monkeypatch):
    """Even with both flags set and a key configured, a missing
    solana/solders install must fail with a clear, actionable error rather
    than an unhandled ImportError bubbling out of the trade path."""
    from app.execution.jupiter import JupiterExecutionClient

    settings.LIVE_TRADING = True
    settings.LIVE_EXECUTION_ACKNOWLEDGED = True
    monkeypatch.setattr(settings, "SOLANA_PRIVATE_KEY", "a" * 88)
    result = await JupiterExecutionClient()._execute_swap(
        {"inAmount": "1", "outAmount": "1"}, input_decimals=6, output_decimals=6,
    )
    assert not result.success
    assert "deps missing" in result.error


async def test_evm_backend_refuses_without_a_quote_token_configured(monkeypatch):
    """Without EVM_QUOTE_TOKEN_ADDRESS there is nothing to size a USD trade
    against - this must fail closed before any network call, live or not."""
    from app.execution.evm import EvmExecutionClient

    monkeypatch.setattr(settings, "EVM_QUOTE_TOKEN_ADDRESS", None)
    settings.LIVE_TRADING = True
    client = EvmExecutionClient()
    buy = await client.buy("0x" + "1" * 40, 20.0, 150)
    sell = await client.sell("0x" + "1" * 40, 1.0, 150)
    assert not buy.success and "EVM_QUOTE_TOKEN_ADDRESS" in buy.error
    assert not sell.success and "EVM_QUOTE_TOKEN_ADDRESS" in sell.error


async def test_evm_backend_refuses_to_submit_without_the_second_acknowledgement_flag(monkeypatch):
    from app.execution.evm import LIVE_EXECUTION_UNACKNOWLEDGED_MSG, EvmExecutionClient

    monkeypatch.setattr(settings, "EVM_QUOTE_TOKEN_ADDRESS", "0x" + "2" * 40)
    settings.LIVE_TRADING = True
    settings.LIVE_EXECUTION_ACKNOWLEDGED = False
    result = await EvmExecutionClient()._execute_swap("0x" + "1" * 40, "0x" + "2" * 40, 1, 150)
    assert not result.success
    assert result.error == LIVE_EXECUTION_UNACKNOWLEDGED_MSG


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
