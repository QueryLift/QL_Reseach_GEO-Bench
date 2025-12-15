"""
File Output Functions
=====================
実験結果をJSON形式でファイルに保存する機能を提供する。

主要な機能:
- 実験設定の保存
- 個別実行結果の保存（回答、メトリクス）
- ターゲットサマリーの保存
- ドメインサマリーの保存
- ルートサマリーの保存

全ての出力はJSON形式で、階層的な構造を持つ。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..analysis import (
    PrimarySourceRate,
    SourceScoresStats,
    Stats,
    primary_source_rate_to_dict,
    source_scores_stats_to_dict,
    stats_to_dict,
)
from ..types import CitationMetrics

if TYPE_CHECKING:
    from ..types import SourceContent


# =============================================================================
# JSON Serialization Helpers
# =============================================================================

def save_json(path: Path, data: dict | list) -> None:
    """
    データをJSONファイルとして保存

    Args:
        path: 保存先のファイルパス
        data: 保存するデータ（辞書またはリスト）

    Note:
        日本語文字はそのまま保存（ensure_ascii=False）
        日付などの特殊な型はstrに変換
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# =============================================================================
# Run Result Output
# =============================================================================

def save_run_metrics(
    output_dir: Path,
    run_index: int,
    metrics: dict[int, CitationMetrics],
    sources: list[SourceContent],
    include_target: bool,
    target_index: int | None = None,
) -> None:
    """
    1回分の引用メトリクスをJSONで保存

    Args:
        output_dir: 出力ディレクトリ
        run_index: 実行インデックス（1-indexed）
        metrics: インデックス -> CitationMetrics のマッピング
        sources: ソースリスト
        include_target: ターゲットを含むかどうか
        target_index: ターゲットのインデックス（1-indexed、include_target=Trueの場合に必須）

    Output file format:
        {
            "run_index": 1,
            "include_target": true,
            "target_index": 6,  // include_target=trueの場合のみ
            "citations": [
                {
                    "index": 1,
                    "url": "https://...",
                    "is_target": false,
                    "media_type": "HTML",
                    "imp_wc": 12.34,
                    "imp_pwc": 10.21,
                    "citation_frequency": 3,
                    "first_position": 0
                },
                ...
            ]
        }
    """
    suffix = "with" if include_target else "without"
    filename = f"{run_index}_metrics_{suffix}.json"

    citations = []
    for idx in sorted(metrics.keys()):
        m = metrics[idx]
        source_idx = idx - 1  # 0-indexed
        is_target = include_target and idx == target_index

        if 0 <= source_idx < len(sources):
            source = sources[source_idx]
            citations.append({
                "index": idx,
                "url": source["url"],
                "is_target": is_target,
                "media_type": source.get("media_type", "UNKNOWN"),
                "imp_wc": round(m.imp_wc, 2),
                "imp_pwc": round(m.imp_pwc, 2),
                "citation_frequency": m.citation_frequency,
                "first_position": m.first_citation_position,
            })

    data = {
        "run_index": run_index,
        "include_target": include_target,
        "citations": citations,
    }
    if include_target and target_index is not None:
        data["target_index"] = target_index

    save_json(output_dir / filename, data)


def save_run_answer(
    output_dir: Path,
    run_index: int,
    answer: str,
    include_target: bool,
    target_index: int | None = None,
) -> None:
    """
    1回分の回答をJSONで保存

    Args:
        output_dir: 出力ディレクトリ
        run_index: 実行インデックス（1-indexed）
        answer: 生成された回答テキスト
        include_target: ターゲットを含むかどうか
        target_index: ターゲットのインデックス（1-indexed、include_target=Trueの場合のみ）

    Output file format:
        {
            "run_index": 1,
            "include_target": true,
            "target_index": 6,  // include_target=trueの場合のみ
            "answer": "..."
        }
    """
    suffix = "with" if include_target else "without"
    filename = f"{run_index}_answer_{suffix}.json"

    data = {
        "run_index": run_index,
        "include_target": include_target,
        "answer": answer,
    }
    if include_target and target_index is not None:
        data["target_index"] = target_index

    save_json(output_dir / filename, data)


def save_run_result(
    output_dir: Path,
    run_index: int,
    answer_without: str,
    answer_with: str,
    metrics_without: dict[int, CitationMetrics],
    metrics_with: dict[int, CitationMetrics],
    sources: list[SourceContent],
    sources_with_target: list[SourceContent],
    target_index: int,
) -> None:
    """
    1回分の実験結果を全て保存（回答 + メトリクス）

    without/with両方の回答とメトリクスを4つのJSONファイルとして保存する。

    Args:
        output_dir: 出力ディレクトリ
        run_index: 実行インデックス（1-indexed）
        answer_without: ターゲットなしの回答
        answer_with: ターゲットありの回答
        metrics_without: ターゲットなしのメトリクス
        metrics_with: ターゲットありのメトリクス
        sources: Webソースリスト（ターゲットなし）
        sources_with_target: ターゲットを含むソースリスト
        target_index: ターゲットのインデックス（1-indexed）

    Output files:
        - {run_index}_answer_without.json
        - {run_index}_answer_with.json
        - {run_index}_metrics_without.json
        - {run_index}_metrics_with.json
    """
    # 回答を保存
    save_run_answer(output_dir, run_index, answer_without, include_target=False)
    save_run_answer(output_dir, run_index, answer_with, include_target=True, target_index=target_index)

    # メトリクスを保存
    save_run_metrics(output_dir, run_index, metrics_without, sources, include_target=False)
    save_run_metrics(output_dir, run_index, metrics_with, sources_with_target, include_target=True, target_index=target_index)


# =============================================================================
# Sources Output
# =============================================================================

def save_sources(output_dir: Path, sources: list[SourceContent]) -> None:
    """
    検索で取得したソース一覧をJSONで保存

    Args:
        output_dir: 出力ディレクトリ
        sources: ソースリスト

    Output file format:
        [
            {
                "index": 1,
                "url": "https://...",
                "media_type": "HTML",
                "content": "..."
            },
            ...
        ]
    """
    data = [
        {
            "index": i + 1,
            "url": s["url"],
            "media_type": s.get("media_type", "UNKNOWN"),
            "content": s["content"],
        }
        for i, s in enumerate(sources)
    ]
    save_json(output_dir / "sources.json", data)


# =============================================================================
# Target Summary Output
# =============================================================================

def save_target_summary(
    output_dir: Path,
    target_id: str,
    title: str,
    domain: str,
    provider: str,
    question_type: str,
    question: str,
    num_runs: int,
    citation_rate: float,
    imp_wc: Stats,
    imp_pwc: Stats,
    primary_rate_without: PrimarySourceRate | Stats,
    primary_rate_with: PrimarySourceRate | Stats,
    primary_rate_diff: Stats | None = None,
    source_scores_without: SourceScoresStats | None = None,
    source_scores_with: SourceScoresStats | None = None,
    num_runs_with_non_primary: int | None = None,
) -> None:
    """
    ターゲットの集計結果をJSONで保存

    Args:
        output_dir: 出力ディレクトリ
        target_id: ターゲットID
        title: ターゲットタイトル（H1タグの内容）
        domain: ドメイン名
        provider: LLMプロバイダー名
        question_type: 質問タイプ（vague, experiment, aligned）
        question: 使用した質問文
        num_runs: 実行回数
        citation_rate: ターゲットが引用された割合
        imp_wc: imp_wcの統計値
        imp_pwc: imp_pwcの統計値
        primary_rate_without: ターゲットなしの一次情報源率
        primary_rate_with: ターゲットありの一次情報源率
        primary_rate_diff: 一次情報源率の変化（Stats形式、省略可能）
        source_scores_without: ターゲットなしのソース別スコア統計
        source_scores_with: ターゲットありのソース別スコア統計
        num_runs_with_non_primary: 非一次情報源があった実行回数
            （一次情報源率の統計計算対象となるrun数）

    Output file: summary.json
    """
    # primary_rate の変換（Stats or PrimarySourceRate）
    def convert_primary_rate(rate):
        if isinstance(rate, Stats):
            return stats_to_dict(rate)
        else:
            # PrimarySourceRate の場合
            return primary_source_rate_to_dict(rate)

    data = {
        "target_id": target_id,
        "title": title,
        "domain": domain,
        "provider": provider,
        "question_type": question_type,
        "question": question,
        "num_runs": num_runs,
        "citation_rate": round(citation_rate, 3),
        "imp_wc": stats_to_dict(imp_wc),
        "imp_pwc": stats_to_dict(imp_pwc),
        "primary_source_rate": {
            "without": convert_primary_rate(primary_rate_without),
            "with": convert_primary_rate(primary_rate_with),
        },
    }

    # 非一次情報源があった実行回数を追加
    if num_runs_with_non_primary is not None:
        data["num_runs_with_non_primary"] = num_runs_with_non_primary

    # オプショナルなフィールド
    if primary_rate_diff is not None:
        data["primary_source_rate"]["diff"] = stats_to_dict(primary_rate_diff)

    if source_scores_without is not None:
        data["source_scores"] = {
            "without": source_scores_stats_to_dict(source_scores_without),
        }
        if source_scores_with is not None:
            data["source_scores"]["with"] = source_scores_stats_to_dict(source_scores_with)

    save_json(output_dir / "summary.json", data)


# =============================================================================
# Question Type Summary Output
# =============================================================================

def question_type_summary_to_dict(
    question_type: str,
    num_targets: int,
    num_runs_per_target: int,
    citation_rate_avg: float,
    citation_rate_median: float,
    imp_wc_avg: float,
    imp_wc_median: float,
    imp_pwc_avg: float,
    imp_pwc_median: float,
    primary_rate_without_avg: float,
    primary_rate_without_median: float,
    primary_rate_with_avg: float,
    primary_rate_with_median: float,
    primary_rate_diff_avg: float,
    primary_rate_diff_median: float,
    # 引用回数ベースの一次情報源率
    primary_rate_by_freq_without_avg: float = 0.0,
    primary_rate_by_freq_without_median: float = 0.0,
    primary_rate_by_freq_with_avg: float = 0.0,
    primary_rate_by_freq_with_median: float = 0.0,
    primary_rate_by_freq_diff_avg: float = 0.0,
    primary_rate_by_freq_diff_median: float = 0.0,
    source_scores_without: SourceScoresStats | None = None,
    source_scores_with: SourceScoresStats | None = None,
    target_details: list[dict] | None = None,
) -> dict:
    """
    質問タイプの集計結果を辞書に変換

    Args:
        question_type: 質問タイプ（vague, experiment, aligned, all）
        num_targets: ターゲット数
        num_runs_per_target: ターゲットあたりの実行回数
        citation_rate_avg: 引用率の平均
        citation_rate_median: 引用率の中央値
        imp_wc_avg: imp_wcの平均
        imp_wc_median: imp_wcの中央値
        imp_pwc_avg: imp_pwcの平均
        imp_pwc_median: imp_pwcの中央値
        primary_rate_without_avg: ターゲットなしの一次情報源率（imp_wc）の平均
        primary_rate_without_median: ターゲットなしの一次情報源率（imp_wc）の中央値
        primary_rate_with_avg: ターゲットありの一次情報源率（imp_wc）の平均
        primary_rate_with_median: ターゲットありの一次情報源率（imp_wc）の中央値
        primary_rate_diff_avg: 一次情報源率（imp_wc）の変化の平均
        primary_rate_diff_median: 一次情報源率（imp_wc）の変化の中央値
        primary_rate_by_freq_*: 引用回数ベースの一次情報源率（各種統計）
        source_scores_without: ターゲットなしのソース別スコア統計
        source_scores_with: ターゲットありのソース別スコア統計
        target_details: ターゲットごとの詳細情報リスト

    Returns:
        dict: JSON出力用の辞書形式
    """
    result = {
        "question_type": question_type,
        "num_targets": num_targets,
        "num_runs_per_target": num_runs_per_target,
        "citation_rate": {
            "avg": round(citation_rate_avg, 3),
            "median": round(citation_rate_median, 3),
        },
        "imp_wc": {
            "avg": round(imp_wc_avg, 2),
            "median": round(imp_wc_median, 2),
        },
        "imp_pwc": {
            "avg": round(imp_pwc_avg, 2),
            "median": round(imp_pwc_median, 2),
        },
        "primary_source_rate": {
            "by_imp_wc": {
                "without": {
                    "avg": round(primary_rate_without_avg, 2),
                    "median": round(primary_rate_without_median, 2),
                },
                "with": {
                    "avg": round(primary_rate_with_avg, 2),
                    "median": round(primary_rate_with_median, 2),
                },
                "diff": {
                    "avg": round(primary_rate_diff_avg, 2),
                    "median": round(primary_rate_diff_median, 2),
                },
            },
            "by_frequency": {
                "without": {
                    "avg": round(primary_rate_by_freq_without_avg, 2),
                    "median": round(primary_rate_by_freq_without_median, 2),
                },
                "with": {
                    "avg": round(primary_rate_by_freq_with_avg, 2),
                    "median": round(primary_rate_by_freq_with_median, 2),
                },
                "diff": {
                    "avg": round(primary_rate_by_freq_diff_avg, 2),
                    "median": round(primary_rate_by_freq_diff_median, 2),
                },
            },
        },
    }

    if source_scores_without is not None:
        result["source_scores"] = {
            "without": source_scores_stats_to_dict(source_scores_without),
        }
        if source_scores_with is not None:
            result["source_scores"]["with"] = source_scores_stats_to_dict(source_scores_with)

    if target_details is not None:
        result["targets"] = target_details

    return result


# =============================================================================
# Domain Summary Output
# =============================================================================

def save_domain_summary(
    output_dir: Path,
    provider: str,
    domain: str,
    num_runs_per_target: int,
    question_type_summaries: dict[str, dict],
    all_summary: dict,
) -> None:
    """
    ドメインの集計結果をJSONで保存

    Args:
        output_dir: 出力ディレクトリ
        provider: LLMプロバイダー名
        domain: ドメイン名
        num_runs_per_target: ターゲットあたりの実行回数
        question_type_summaries: 質問タイプ別サマリー（key: question_type）
        all_summary: 全質問タイプの集計サマリー

    Output file: {provider}_summary.json
    """
    data = {
        "domain": domain,
        "provider": provider,
        "num_runs_per_target": num_runs_per_target,
        "question_types": question_type_summaries,
        "all": all_summary,
    }
    save_json(output_dir / f"{provider}_summary.json", data)


# =============================================================================
# Root Summary Output
# =============================================================================

def save_root_summary(
    output_dir: Path,
    providers: list[str],
    num_runs: int,
    question_types: list[str],
    primary_domains: list[str],
    domain_summaries: dict[str, dict[str, dict]],
    overall_summary: dict | None = None,
) -> None:
    """
    実験全体のルートサマリーをJSONで保存

    Args:
        output_dir: 出力ディレクトリ
        providers: 使用したLLMプロバイダーのリスト
        num_runs: 実行回数
        question_types: 質問タイプのリスト
        primary_domains: 一次情報源のドメインリスト
        domain_summaries: ドメイン -> プロバイダー -> サマリー のマッピング
        overall_summary: 全体の集計サマリー（オプション）

    Output file: summary.json
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "providers": providers,
        "num_runs": num_runs,
        "question_types": question_types,
        "primary_domains": primary_domains,
        "domains": domain_summaries,
    }

    if overall_summary is not None:
        data["overall"] = overall_summary

    save_json(output_dir / "summary.json", data)


# =============================================================================
# Experiment Config Output
# =============================================================================

def save_experiment_config(
    output_dir: Path,
    providers: list[str],
    num_runs: int,
    max_sources: int,
    primary_domains: list[str],
    question_types: list[str],
    targets: list[dict],
    questions: dict[str, dict],
) -> None:
    """
    実験設定をJSONで保存

    Args:
        output_dir: 出力ディレクトリ
        providers: 使用するLLMプロバイダーのリスト
        num_runs: 実行回数
        max_sources: Web検索で取得する最大ソース数
        primary_domains: 一次情報源のドメインリスト
        question_types: 質問タイプのリスト
        targets: ターゲット情報のリスト（id, domain, titleを含む）
        questions: ターゲットID -> 質問 のマッピング

    Output file: experiment_config.json
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "providers": providers,
        "num_runs": num_runs,
        "max_sources": max_sources,
        "primary_domains": primary_domains,
        "question_types": question_types,
        "targets": [
            {"id": t["id"], "domain": t["domain"], "title": t["title"]}
            for t in targets
        ],
        "questions": questions,
    }
    save_json(output_dir / "experiment_config.json", data)
