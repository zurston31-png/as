"""A rules-first trading signal bot with an AI confirmation layer.

    data -> strategy engine -> AI confirmation -> risk engine -> signal

See README.md for the staged rollout: backtest, then live paper, then alerts,
then (only if the numbers hold up) broker integration.
"""

__version__ = "0.1.0"

from .config import Config
from .models import Side, TradeSignal

__all__ = ["Config", "Side", "TradeSignal", "__version__"]
