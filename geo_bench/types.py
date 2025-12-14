"""
GEO-bench Type Definitions
==========================
共通の型定義（TypedDict, dataclass）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class SourceContent(TypedDict):
    """ソースコンテンツの型"""
    url: str
    content: str
    media_type: str  # "HTML", "PDF", "TARGET"


class TargetConfig(TypedDict):
    """ターゲット設定の型"""
    id: str
    domain: str
    title: str
    file: str
    content: str


class GeneratedQuestions(TypedDict):
    """生成された質問の型"""
    vague: str
    experiment: str
    aligned: str


@dataclass
class Citation:
    """個別の引用情報"""
    sentences: list[str] = field(default_factory=list)
    word_count: int = 0
    position_sum: float = 0.0
    first_pos: int = -1


@dataclass
class CitationMetrics:
    """引用メトリクス"""
    imp_wc: float = 0.0       # Impression Word Count (%)
    imp_pwc: float = 0.0      # Position Weighted Word Count (%)
    citation_frequency: int = 0
    first_citation_position: int = -1


@dataclass
class RunResult:
    """1回の実行結果"""
    run_number: int
    answer_without: str
    answer_with: str
    metrics_without: dict[int, CitationMetrics]
    metrics_with: dict[int, CitationMetrics]
    target_cited: bool
    target_imp_wc: float
    target_imp_pwc: float
    target_citation_frequency: int
    target_first_position: int
    primary_source_rate_without: float
    primary_source_rate_with: float
    source_scores_without: dict[str, dict]
    source_scores_with: dict[str, dict]
    has_non_primary_sources: bool


@dataclass
class TargetSummary:
    """ターゲットごとのサマリー"""
    target_id: str
    target_title: str
    question_type: str
    question: str
    num_runs: int
    target_citation_rate: float
    avg_target_imp_wc: float
    avg_target_imp_pwc: float
    avg_target_citation_frequency: float
    avg_target_first_position: float
    avg_primary_source_rate_without: float
    avg_primary_source_rate_with: float
    primary_source_rate_delta: float
    avg_source_scores_without: dict[str, dict]
    avg_source_scores_with: dict[str, dict]
    non_primary_source_count: int
    runs: list[RunResult] = field(default_factory=list)


@dataclass
class QuestionTypeSummary:
    """質問タイプごとのサマリー"""
    question_type: str
    num_targets: int
    avg_target_citation_rate: float
    avg_target_imp_wc: float
    avg_target_imp_pwc: float
    avg_target_citation_frequency: float
    avg_target_first_position: float
    avg_primary_source_rate_without: float
    avg_primary_source_rate_with: float
    avg_primary_source_rate_delta: float
    valid_targets_for_primary_rate: int
    targets: list[TargetSummary] = field(default_factory=list)


@dataclass
class DomainSummary:
    """ドメインごとのサマリー"""
    domain: str
    provider: str
    question_type_summaries: dict[str, QuestionTypeSummary] = field(default_factory=dict)
