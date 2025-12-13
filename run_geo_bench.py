#!/usr/bin/env python3
"""
GEO-bench 実行スクリプト

このスクリプトは config.json の設定に基づいて GEO-bench を実行します。

使用方法:
    python run_geo_bench.py
"""

import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# .envファイルから環境変数をロード
from dotenv import load_dotenv
load_dotenv()

from llm_clients import create_llm_client
from ql_geo_bench import (
    GEOBench,
    TargetContent,
    GEOBenchResult,
    strip_markdown,
)


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込む"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"エラー: 設定ファイルのJSON形式が不正です: {e}")
        sys.exit(1)


def load_target_file(file_path: str) -> str:
    """ターゲットファイルを読み込み、Markdownを除去"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return strip_markdown(content)
    except FileNotFoundError:
        print(f"エラー: ターゲットファイルが見つかりません: {file_path}")
        sys.exit(1)


def setup_output_dir(base_dir: str) -> Path:
    """出力ディレクトリを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(base_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_json(output_dir: Path, filename: str, data: dict):
    """JSONファイルを保存"""
    file_path = output_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_csv(output_dir: Path, filename: str, headers: list, rows: list):
    """CSVファイルを保存"""
    file_path = output_dir / filename
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def build_target_contents(targets_config: list[dict]) -> list[TargetContent]:
    """config からターゲットコンテンツを構築"""
    target_contents: list[TargetContent] = []

    for target_info in targets_config:
        target_contents.append({
            "id": target_info["id"],
            "url": target_info["url"],
            "content": load_target_file(target_info["file"]),
        })

    return target_contents


def save_results(
    result: GEOBenchResult,
    target_contents: list[TargetContent],
    output_dir: Path,
    question: str,
):
    """結果をファイルに保存"""

    # ルックアップ用辞書を構築
    target_by_index = {ti.target_index: ti for ti in result.target_infos}
    content_by_id = {tc["id"]: tc["content"] for tc in target_contents}

    # === ソースファイル（統合）===
    # Webソース + ターゲット（cited_without/cited_with両方）
    sources_data = []
    # Webソース部分
    for i, source in enumerate(result.sources):
        sources_data.append({
            "index": i + 1,
            "url": source["url"],
            "content": source["content"],
            "media_type": source["media_type"],
            "is_target": False,
            "cited_without": (i + 1) in result.citations_without_targets,
            "cited_with": (i + 1) in result.citations_with_targets,
        })
    # ターゲット部分
    for ti in result.target_infos:
        sources_data.append({
            "index": ti.target_index,
            "url": ti.target_url,
            "content": content_by_id[ti.target_id],
            "media_type": "TARGET",
            "is_target": True,
            "target_id": ti.target_id,
            "cited_with": ti.target_index in result.citations_with_targets,
        })
    save_json(output_dir, "sources.json", sources_data)

    # 回答（ターゲットなし）
    answer_without_data = {
        "question": question,
        "include_target": False,
        "answer": result.answer_without_targets,
    }
    save_json(output_dir, "answer_without.json", answer_without_data)

    # 引用メトリクス CSV（ターゲットなし）
    metrics_headers = [
        "index", "url", "is_target",
        "imp_wc", "imp_pwc",
        "citation_frequency", "first_position"
    ]
    metrics_without_rows = []
    for i in sorted(result.citations_without_targets.keys()):
        m = result.citations_without_targets[i]
        source_idx = i - 1
        url = result.sources[source_idx]["url"]
        metrics_without_rows.append([
            i,
            url,
            "No",
            round(m.imp_wc, 2),
            round(m.imp_pwc, 2),
            m.citation_frequency,
            m.first_citation_position if m.first_citation_position is not None else "N/A"
        ])
    save_csv(output_dir, "metrics_without.csv", metrics_headers, metrics_without_rows)

    # ターゲット情報をまとめる
    targets_info = [
        {"id": ti.target_id, "url": ti.target_url, "index": ti.target_index}
        for ti in result.target_infos
    ]

    # 回答（ターゲットあり）
    answer_with_data = {
        "question": question,
        "include_target": True,
        "targets": targets_info,
        "answer": result.answer_with_targets,
    }
    save_json(output_dir, "answer_with.json", answer_with_data)

    # 引用メトリクス CSV（ターゲットあり）
    metrics_with_rows = []
    for i in sorted(result.citations_with_targets.keys()):
        m = result.citations_with_targets[i]
        ti = target_by_index.get(i)
        if ti:
            url = ti.target_url
            is_target = "Yes"
            target_id = ti.target_id
        else:
            source_idx = i - 1
            url = result.sources[source_idx]["url"]
            is_target = "No"
            target_id = ""
        metrics_with_rows.append([
            i,
            url,
            is_target,
            target_id,
            round(m.imp_wc, 2),
            round(m.imp_pwc, 2),
            m.citation_frequency,
            m.first_citation_position if m.first_citation_position is not None else "N/A"
        ])

    metrics_with_headers = [
        "index", "url", "is_target", "target_id",
        "imp_wc", "imp_pwc",
        "citation_frequency", "first_position"
    ]
    save_csv(output_dir, "metrics_with.csv", metrics_with_headers, metrics_with_rows)

    # サマリー用データを収集
    summary_results = []
    for ti in result.target_infos:
        metrics = result.citations_with_targets.get(ti.target_index)
        summary_results.append({
            "target_id": ti.target_id,
            "target_url": ti.target_url,
            "target_index": ti.target_index,
            "included": metrics is not None,
            "imp_wc": metrics.imp_wc if metrics else 0,
            "imp_pwc": metrics.imp_pwc if metrics else 0,
        })

    return summary_results


def print_summary(summary_results: list[dict], output_dir: Path):
    """結果のサマリーをコンソールに出力"""
    print(f"\n完了: {output_dir}")
    for r in summary_results:
        status = f"Imp_wc: {r['imp_wc']:.1f}%" if r['included'] else "引用なし"
        print(f"  {r['target_id']}: {status}")


async def run_benchmark(
    question: str,
    target_contents: list[TargetContent],
    max_sources: int,
    provider: str = "gpt",
) -> GEOBenchResult:
    """GEO-bench を実行"""
    llm = create_llm_client(provider)
    print(f"[LLM] {llm.name}")
    bench = GEOBench(llm=llm, target_contents=target_contents, max_sources=max_sources)

    try:
        result = await bench.run(question)
        return result
    finally:
        await bench.close()


async def main():
    # 設定を読み込み
    config = load_config("config.json")

    question = config.get("question")
    output_base_dir = config.get("output_dir", "outputs")
    max_sources = config.get("max_sources", 5)
    provider = config.get("provider", "gpt")  # "gpt", "claude", "gemini"
    targets_config = config.get("targets", [])

    # バリデーション
    if not question:
        print("エラー: 質問文が指定されていません。config.json で指定してください。")
        sys.exit(1)

    if not targets_config:
        print("エラー: ターゲットが指定されていません。config.json で指定してください。")
        sys.exit(1)

    # 出力ディレクトリを作成
    output_dir = setup_output_dir(output_base_dir)

    # 実行設定を保存
    save_json(output_dir, "config.json", {
        "question": question,
        "provider": provider,
        "targets": targets_config,
        "max_sources": max_sources,
        "timestamp": datetime.now().isoformat()
    })

    # ターゲットコンテンツを構築
    target_contents = build_target_contents(targets_config)

    # GEO-bench を実行
    print(f"実行中... (ターゲット数: {len(target_contents)})")

    result = await run_benchmark(
        question=question,
        target_contents=target_contents,
        max_sources=max_sources,
        provider=provider,
    )

    # 結果を保存
    summary_results = save_results(
        result=result,
        target_contents=target_contents,
        output_dir=output_dir,
        question=question,
    )

    # サマリーを表示
    print_summary(summary_results, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
