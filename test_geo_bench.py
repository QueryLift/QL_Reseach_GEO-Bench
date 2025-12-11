"""
GEO-bench テストスクリプト
複数ターゲット対応版（効率化版）
"""

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path

# .envファイルから環境変数をロード（必ずimportの前に実行）
from dotenv import load_dotenv
load_dotenv()

from ql_geo_bench import LLMClient, GEOBench, TargetContent, strip_markdown

CONFIG_FILE = "config.json"


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_target_content(file_path: str) -> str:
    """ターゲットファイルを読み込む（Markdown記号を除去してPlain textに変換）"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return strip_markdown(content)


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
    """config.jsonのtargetsからTargetContentリストを構築"""
    target_contents: list[TargetContent] = []

    for target_info in targets_config:
        target_contents.append({
            "id": target_info["id"],
            "url": target_info["url"],
            "content": load_target_content(target_info["file"]),
        })

    return target_contents


async def main():
    # 設定を読み込み
    config = load_config(CONFIG_FILE)

    question = config["question"]
    targets_config = config["targets"]
    output_base_dir = config["output_dir"]
    max_sources = config.get("max_sources", 5)

    # 出力ディレクトリを作成
    output_dir = setup_output_dir(output_base_dir)

    # 実行設定を保存
    save_json(output_dir, "config.json", {
        "question": question,
        "targets": targets_config,
        "max_sources": max_sources,
        "timestamp": datetime.now().isoformat()
    })

    # ターゲットコンテンツを構築
    target_contents = build_target_contents(targets_config)

    # LLMクライアントを初期化
    llm = LLMClient()

    # GEOBench を初期化（複数ターゲット対応）
    bench = GEOBench(llm=llm, target_contents=target_contents, max_sources=max_sources)

    print(f"実行中... (ターゲット数: {len(target_contents)})")

    try:
        # 一括実行（Web検索とターゲットなし回答は1回のみ、全ターゲット一括）
        result = await bench.run(question)

        # 回答（ターゲットなし）
        references_without = []
        for i in sorted(result.citations_without_targets.keys()):
            source_idx = i - 1  # sources は 0-indexed
            if 0 <= source_idx < len(result.sources):
                source = result.sources[source_idx]
                references_without.append({
                    "index": i,
                    "url": source.get("url", f"web_source_{i}"),
                    "content": source.get("content", ""),
                    "is_target": False
                })
            else:
                references_without.append({
                    "index": i,
                    "url": f"web_source_{i}",
                    "content": "",
                    "is_target": False
                })

        answer_without_data = {
            "question": question,
            "include_target": False,
            "answer": result.answer_without_targets,
            "references": references_without
        }
        save_json(output_dir, "answer_without.json", answer_without_data)

        # 引用メトリクス CSV（ターゲットなし）
        metrics_headers = [
            "index", "url", "is_target",
            "word_count_pct", "position_adjusted_pct",
            "citation_frequency", "first_position"
        ]
        metrics_without_rows = []
        for i in sorted(result.citations_without_targets.keys()):
            m = result.citations_without_targets[i]
            source_idx = i - 1
            url = result.sources[source_idx].get("url", f"web_source_{i}") if 0 <= source_idx < len(result.sources) else f"web_source_{i}"
            metrics_without_rows.append([
                i,
                url,
                "No",
                round(m.word_count, 2),
                round(m.position_adjusted, 2),
                m.citation_frequency,
                m.first_citation_position if m.first_citation_position is not None else "N/A"
            ])
        save_csv(output_dir, "metrics_without.csv", metrics_headers, metrics_without_rows)

        # === ターゲットありの出力（全ターゲット一括） ===
        references_with = []
        for i in sorted(result.citations_with_targets.keys()):
            target_info = next((ti for ti in result.target_infos if ti.target_index == i), None)
            if target_info:
                target_content = next(
                    (tc["content"] for tc in target_contents if tc["id"] == target_info.target_id),
                    ""
                )
                references_with.append({
                    "index": i,
                    "url": target_info.target_url,
                    "content": target_content,
                    "is_target": True,
                    "target_id": target_info.target_id
                })
            else:
                source_idx = i - 1
                if 0 <= source_idx < len(result.sources):
                    source = result.sources[source_idx]
                    references_with.append({
                        "index": i,
                        "url": source.get("url", f"web_source_{i}"),
                        "content": source.get("content", ""),
                        "is_target": False
                    })
                else:
                    references_with.append({
                        "index": i,
                        "url": f"web_source_{i}",
                        "content": "",
                        "is_target": False
                    })

        targets_info = [
            {"id": ti.target_id, "url": ti.target_url, "index": ti.target_index}
            for ti in result.target_infos
        ]

        answer_with_data = {
            "question": question,
            "include_target": True,
            "targets": targets_info,
            "answer": result.answer_with_targets,
            "references": references_with
        }
        save_json(output_dir, "answer_with.json", answer_with_data)

        # 引用メトリクス CSV（ターゲットあり）
        metrics_with_headers = [
            "index", "url", "is_target", "target_id",
            "word_count_pct", "position_adjusted_pct",
            "citation_frequency", "first_position"
        ]
        metrics_with_rows = []
        for i in sorted(result.citations_with_targets.keys()):
            m = result.citations_with_targets[i]
            target_info = next((ti for ti in result.target_infos if ti.target_index == i), None)
            if target_info:
                url = target_info.target_url
                is_target = "Yes"
                target_id = target_info.target_id
            else:
                source_idx = i - 1
                url = result.sources[source_idx].get("url", f"web_source_{i}") if 0 <= source_idx < len(result.sources) else f"web_source_{i}"
                is_target = "No"
                target_id = ""
            metrics_with_rows.append([
                i,
                url,
                is_target,
                target_id,
                round(m.word_count, 2),
                round(m.position_adjusted, 2),
                m.citation_frequency,
                m.first_citation_position if m.first_citation_position is not None else "N/A"
            ])
        save_csv(output_dir, "metrics_with.csv", metrics_with_headers, metrics_with_rows)

        # サマリー出力
        summary_results = []
        for ti in result.target_infos:
            visibility = ti.get_visibility(result.citations_with_targets)
            summary_results.append({
                "target_id": ti.target_id,
                "target_url": ti.target_url,
                "target_index": ti.target_index,
                "visibility": visibility,
            })

        print(f"\n完了: {output_dir}")
        for r in summary_results:
            v = r["visibility"]
            status = f"Word Count: {v.get('word_count', 0):.1f}%" if v.get('included') else "引用なし"
            print(f"  {r['target_id']}: {status}")

    finally:
        await bench.close()


if __name__ == "__main__":
    asyncio.run(main())
