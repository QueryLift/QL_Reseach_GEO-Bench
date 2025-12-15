"""
Metrics Calculation
===================
GEO論文に基づく引用メトリクスの計算と一次情報源の分析

Reference: Aggarwal et al., "GEO: Generative Engine Optimization", KDD 2024
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..types import CitationMetrics, SourceContent


# =============================================================================
# Statistics
# =============================================================================

@dataclass
class Stats:
    """統計値"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    values: list[float]


def calc_stats(values: list[float]) -> Stats:
    """数値リストから統計値を計算"""
    if not values:
        return Stats(mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0, values=[])
    return Stats(
        mean=statistics.mean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
        median=statistics.median(values),
        values=values,
    )


def stats_to_dict(stats: Stats, decimals: int = 2) -> dict:
    """Stats オブジェクトを辞書に変換"""
    return {
        "mean": round(stats.mean, decimals),
        "std": round(stats.std, decimals),
        "min": round(stats.min, decimals),
        "max": round(stats.max, decimals),
        "median": round(stats.median, decimals),
        "values": [round(v, decimals) for v in stats.values],
    }


# =============================================================================
# Primary Source Detection
# =============================================================================

def is_primary_source(
    url: str,
    primary_domains: list[str],
    media_type: str | None = None,
) -> bool:
    """URLまたはメディアタイプが一次情報源かどうかを判定"""
    # TARGET は常に一次情報源
    if media_type == "TARGET":
        return True
    # ドメイン部分一致で判定
    url_lower = url.lower()
    for domain in primary_domains:
        if domain.lower() in url_lower:
            return True
    return False


# =============================================================================
# Source Scores
# =============================================================================

@dataclass
class SourceScores:
    """ソース別スコア（一次情報源/非一次情報源）"""
    primary_imp_wc_values: list[float]
    primary_imp_pwc_values: list[float]
    primary_frequency_values: list[int]
    non_primary_imp_wc_values: list[float]
    non_primary_imp_pwc_values: list[float]
    non_primary_frequency_values: list[int]


@dataclass
class SourceScoresStats:
    """ソース別スコア統計"""
    primary_imp_wc: Stats
    primary_imp_pwc: Stats
    primary_frequency: Stats
    non_primary_imp_wc: Stats
    non_primary_imp_pwc: Stats
    non_primary_frequency: Stats


def calc_source_scores(
    metrics: dict[int, CitationMetrics],
    sources: list[SourceContent],
    primary_domains: list[str],
    target_indices: set[int] | None = None,
) -> SourceScores:
    """一次情報源/非一次情報源のスコアを計算"""
    target_indices = target_indices or set()

    primary_imp_wc: list[float] = []
    primary_imp_pwc: list[float] = []
    primary_frequency: list[int] = []
    non_primary_imp_wc: list[float] = []
    non_primary_imp_pwc: list[float] = []
    non_primary_frequency: list[int] = []

    for idx, m in metrics.items():
        # ターゲットインデックスの場合は一次情報源として扱う
        if idx in target_indices:
            primary_imp_wc.append(m.imp_wc)
            primary_imp_pwc.append(m.imp_pwc)
            primary_frequency.append(m.citation_frequency)
            continue

        # Webソースの場合
        source_idx = idx - 1  # 0-indexed
        if 0 <= source_idx < len(sources):
            source = sources[source_idx]
            if is_primary_source(source["url"], primary_domains, source.get("media_type")):
                primary_imp_wc.append(m.imp_wc)
                primary_imp_pwc.append(m.imp_pwc)
                primary_frequency.append(m.citation_frequency)
            else:
                non_primary_imp_wc.append(m.imp_wc)
                non_primary_imp_pwc.append(m.imp_pwc)
                non_primary_frequency.append(m.citation_frequency)

    return SourceScores(
        primary_imp_wc_values=primary_imp_wc,
        primary_imp_pwc_values=primary_imp_pwc,
        primary_frequency_values=primary_frequency,
        non_primary_imp_wc_values=non_primary_imp_wc,
        non_primary_imp_pwc_values=non_primary_imp_pwc,
        non_primary_frequency_values=non_primary_frequency,
    )


def aggregate_source_scores(scores_list: list[SourceScores]) -> SourceScoresStats:
    """複数のSourceScoresを集計してSourceScoresStatsに変換"""
    primary_wc_all: list[float] = []
    primary_pwc_all: list[float] = []
    primary_frequency_all: list[float] = []
    non_primary_wc_all: list[float] = []
    non_primary_pwc_all: list[float] = []
    non_primary_frequency_all: list[float] = []

    for scores in scores_list:
        primary_wc_all.extend(scores.primary_imp_wc_values)
        primary_pwc_all.extend(scores.primary_imp_pwc_values)
        primary_frequency_all.extend([float(f) for f in scores.primary_frequency_values])
        non_primary_wc_all.extend(scores.non_primary_imp_wc_values)
        non_primary_pwc_all.extend(scores.non_primary_imp_pwc_values)
        non_primary_frequency_all.extend([float(f) for f in scores.non_primary_frequency_values])

    return SourceScoresStats(
        primary_imp_wc=calc_stats(primary_wc_all),
        primary_imp_pwc=calc_stats(primary_pwc_all),
        primary_frequency=calc_stats(primary_frequency_all),
        non_primary_imp_wc=calc_stats(non_primary_wc_all),
        non_primary_imp_pwc=calc_stats(non_primary_pwc_all),
        non_primary_frequency=calc_stats(non_primary_frequency_all),
    )


def source_scores_stats_to_dict(stats: SourceScoresStats) -> dict:
    """SourceScoresStats を辞書に変換"""
    return {
        "primary": {
            "imp_wc": stats_to_dict(stats.primary_imp_wc),
            "imp_pwc": stats_to_dict(stats.primary_imp_pwc),
            "citation_frequency": stats_to_dict(stats.primary_frequency),
        },
        "non_primary": {
            "imp_wc": stats_to_dict(stats.non_primary_imp_wc),
            "imp_pwc": stats_to_dict(stats.non_primary_imp_pwc),
            "citation_frequency": stats_to_dict(stats.non_primary_frequency),
        },
    }


# =============================================================================
# Primary Source Rate Calculations
# =============================================================================

@dataclass
class PrimarySourceRate:
    """一次情報源の引用率"""
    by_imp_wc: float
    by_frequency: float


def calc_primary_source_rate(
    metrics: dict[int, CitationMetrics],
    sources: list[SourceContent],
    primary_domains: list[str],
    target_indices: set[int] | None = None,
) -> PrimarySourceRate:
    """一次情報源の引用率を計算（imp_wcとcitation_frequency両方）"""
    if not metrics:
        return PrimarySourceRate(by_imp_wc=0.0, by_frequency=0.0)

    target_indices = target_indices or set()

    total_imp_wc = 0.0
    primary_imp_wc = 0.0
    total_freq = 0
    primary_freq = 0

    for idx, m in metrics.items():
        total_imp_wc += m.imp_wc
        total_freq += m.citation_frequency

        # ターゲットインデックスの場合は一次情報源
        if idx in target_indices:
            primary_imp_wc += m.imp_wc
            primary_freq += m.citation_frequency
            continue

        # Webソースの場合
        source_idx = idx - 1  # 0-indexed
        if 0 <= source_idx < len(sources):
            source = sources[source_idx]
            if is_primary_source(source["url"], primary_domains, source.get("media_type")):
                primary_imp_wc += m.imp_wc
                primary_freq += m.citation_frequency

    return PrimarySourceRate(
        by_imp_wc=(primary_imp_wc / total_imp_wc) * 100 if total_imp_wc > 0 else 0.0,
        by_frequency=(primary_freq / total_freq) * 100 if total_freq > 0 else 0.0,
    )


def primary_source_rate_to_dict(rate: PrimarySourceRate, decimals: int = 2) -> dict:
    """PrimarySourceRate を辞書に変換"""
    return {
        "by_imp_wc": round(rate.by_imp_wc, decimals),
        "by_frequency": round(rate.by_frequency, decimals),
    }
