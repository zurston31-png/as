"""Market data: candles, quality gates, and the providers that supply them.

Everything downstream — indicators, signal scoring, market regime,
backtesting — reads from here, so this layer is responsible for making sure
bad data never reaches a trading decision.
"""
from app.data.candles import Candle, CandleSeries, Timeframe
from app.data.quality import DataQualityReport, assess_quality

__all__ = ["Candle", "CandleSeries", "Timeframe", "DataQualityReport", "assess_quality"]
