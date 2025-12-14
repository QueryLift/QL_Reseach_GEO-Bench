"""
GEO-bench
=========
GEO (Generative Engine Optimization) 論文に基づくWeb検索エミュレータ

Reference: Aggarwal et al., "GEO: Generative Engine Optimization", KDD 2024

モジュール構成:
- analysis: 引用分析とメトリクス計算
- content: Webコンテンツ取得とMarkdown処理
- llm: LLMクライアント（GPT, Claude, Gemini）
- output: ファイル出力ユーティリティ
- questions: 質問生成
- targets: ターゲット検出

Usage:
    from geo_bench import ExperimentRunner
    from geo_bench.config import load_config

    config = load_config("config.json")
    runner = ExperimentRunner(...)
    await runner.run_experiment(targets, questions)
"""

# 主要なクラスをトップレベルでエクスポート
from .analysis import CitationAnalyzer
from .config import load_config, validate_config
from .content import WebContentFetcher, strip_markdown
from .llm import LLMClient, create_llm_client
from .runner import ExperimentRunner
from .targets import discover_targets, get_target_domains, load_target_file
from .types import CitationMetrics, SourceContent, TargetConfig

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "ExperimentRunner",
    "CitationAnalyzer",
    "WebContentFetcher",
    "LLMClient",
    # Types
    "CitationMetrics",
    "SourceContent",
    "TargetConfig",
    # Functions
    "create_llm_client",
    "load_config",
    "validate_config",
    "discover_targets",
    "get_target_domains",
    "load_target_file",
    "strip_markdown",
]
