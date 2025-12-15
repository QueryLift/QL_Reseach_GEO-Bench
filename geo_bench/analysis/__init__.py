"""
Analysis Module
===============
引用分析とメトリクス計算

Usage:
    from geo_bench.analysis import CitationAnalyzer, calc_primary_source_rate
"""

from .citation import CitationAnalyzer
from .metrics import (
    PrimarySourceRate,
    SourceScores,
    SourceScoresStats,
    Stats,
    aggregate_source_scores,
    calc_primary_source_rate,
    calc_source_scores,
    calc_stats,
    is_primary_source,
    primary_source_rate_to_dict,
    source_scores_stats_to_dict,
    stats_to_dict,
)

__all__ = [
    "CitationAnalyzer",
    "Stats",
    "calc_stats",
    "stats_to_dict",
    "is_primary_source",
    "SourceScores",
    "SourceScoresStats",
    "calc_source_scores",
    "aggregate_source_scores",
    "source_scores_stats_to_dict",
    "PrimarySourceRate",
    "calc_primary_source_rate",
    "primary_source_rate_to_dict",
]
