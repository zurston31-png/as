"""Analysis pipeline: context → metrics → detectors → report."""

from .context import (
    DeathContext,
    KillContext,
    MatchContext,
    RoundContext,
    build_contexts,
    build_match_context,
)
from .detectors import BENCHMARKS, DetectorInput, Finding, run_detectors
from .metrics import Metrics, SplitMetrics, add_shot_totals, compute_metrics
from .report import MatchSummary, ProgressLine, Report, build_report

__all__ = [
    "BENCHMARKS", "DeathContext", "DetectorInput", "Finding", "KillContext",
    "MatchContext", "MatchSummary", "Metrics", "ProgressLine", "Report",
    "RoundContext", "SplitMetrics", "add_shot_totals", "build_contexts",
    "build_match_context", "build_report", "compute_metrics", "run_detectors",
]
