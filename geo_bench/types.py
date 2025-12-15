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
    url: str
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
