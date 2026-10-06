"""Data layer: ingestion, cleaning, quality grading and point-in-time access.

The one rule that matters here: code above this layer reads market data
through `MarketView` and nothing else. A `MarketView` is built for a single
(symbol, instant) and physically cannot hold data later than its cutoff, so
lookahead is prevented by the shape of the object rather than by discipline.

Typical wiring:

    store = DataStore(latency_seconds=cfg.data.data_latency_seconds,
                      guard=guard_from_config(cfg))
    store.load(adapter, "NQ", start, end, primary_interval=300)
    grader = QualityGrader(cfg.data, cfg.flow_score)
    print("\\n".join(grader.availability(store.get("NQ")).summary_lines()))

    for view in store.iter_views("NQ"):
        report = grader.grade(view)
        if not report.is_tradable(cfg.data.min_quality_to_trade):
            continue        # -> WAIT("data_quality")
"""

from flow_model.core.enums import Feed
from flow_model.data.adapters import (
    ColumnMap,
    CsvAdapter,
    CsvAdapterConfig,
    InMemoryAdapter,
    ParquetAdapter,
)
from flow_model.data.base import (
    CleanReport,
    DatasetFingerprint,
    DataLayerError,
    DataSourceAdapter,
    FeedStatus,
    MonotonicityError,
    SchemaError,
    SessionCalendarProtocol,
)
from flow_model.data.calendar import (
    TRADABLE_REASONS,
    TradingCalendar,
    good_friday,
    us_half_days,
    us_market_holidays,
)
from flow_model.data.clean import BarCleaner, detect_gaps
from flow_model.data.market_view import LookaheadError, MarketView
from flow_model.data.quality import (
    AvailabilityReport,
    ComponentAvailability,
    QualityGrader,
    QualityReport,
    QualityThresholds,
)
from flow_model.data.series import (
    BarSeries,
    ColumnSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    from_ns,
    to_ns,
    to_ns_array,
)
from flow_model.data.store import DataStore, SymbolData
from flow_model.data.synthetic import (
    RegimeParams,
    SyntheticAdapter,
    SyntheticConfig,
    SyntheticDataset,
    SyntheticMarketGenerator,
)

__all__ = [
    "AvailabilityReport",
    "BarCleaner",
    "BarSeries",
    "CleanReport",
    "ColumnMap",
    "ColumnSeries",
    "ComponentAvailability",
    "CsvAdapter",
    "CsvAdapterConfig",
    "DataLayerError",
    "DataSourceAdapter",
    "DataStore",
    "DatasetFingerprint",
    "Feed",
    "FeedStatus",
    "InMemoryAdapter",
    "LookaheadError",
    "MarketView",
    "MonotonicityError",
    "OptionsSeries",
    "ParquetAdapter",
    "QualityGrader",
    "QualityReport",
    "QualityThresholds",
    "QuoteSeries",
    "RegimeParams",
    "SchemaError",
    "SessionCalendarProtocol",
    "SymbolData",
    "SyntheticAdapter",
    "SyntheticConfig",
    "SyntheticDataset",
    "SyntheticMarketGenerator",
    "TRADABLE_REASONS",
    "TickSeries",
    "TradingCalendar",
    "detect_gaps",
    "from_ns",
    "good_friday",
    "to_ns",
    "to_ns_array",
    "us_half_days",
    "us_market_holidays",
]
