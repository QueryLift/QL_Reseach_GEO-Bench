#!/usr/bin/env python3
"""
GEO-bench 実験スクリプト

複数ターゲット・複数プロバイダーでの実験を実行し、
結果を階層的に集計・出力する。

特徴:
- Web検索は各ターゲット×プロバイダーにつき1回のみ
- 回答生成は各ターゲット×プロバイダーにつきN回繰り返し
- 結果はターゲット→ドメイン→ルートの階層で集計
"""

import asyncio
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
load_dotenv()

from llm_clients import create_llm_client, LLMClient
from geo_bench import (
    CitationAnalyzer,
    CitationMetrics,
    SourceContent,
    TargetContent,
    TargetInfo,
    WebContentFetcher,
    strip_markdown,
)


# =============================================================================
# Configuration
# =============================================================================

# ドメインごとのプロンプト
DOMAIN_PROMPTS = {
    "edu": "自民党の教育政策について教えて",
    "local": "自民党の地域創生の政策について教えて",
    "econ": "自民党の経済政策について教えて",
}

# 実験設定
EXPERIMENT_CONFIG = {
    # "providers": ["gemini", "gpt"],  # 使用するプロバイダー
    "providers": ["gpt"],  # 使用するプロバイダー
    "num_runs": 5,                   # 各ターゲットの繰り返し回数
    "max_sources": 5,                # Web検索で取得するソース数
    "targets_dir": "targets",        # ターゲットファイルのディレクトリ
    "output_dir": "outputs",         # 出力ディレクトリ
    # 一次情報源と判定するドメイン（部分一致）
    "primary_domains": [
        "jimin.jp",
    ],
}

# GEO論文のプロンプトテンプレート
PROMPT_TEMPLATE = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results.
The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Francais, Espamol, Deutsch, or Japanese should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {question}

Search Results:
{sources}"""


# =============================================================================
# Data Types
# =============================================================================

class TargetConfig(TypedDict):
    """ターゲット設定"""
    id: str
    file: str
    url: str
    domain: str
    title: str  # MDファイルの最初のH1タグの内容


@dataclass
class RunResult:
    """1回の実行結果"""
    run_index: int
    answer_without: str
    answer_with: str
    metrics_without: dict[int, CitationMetrics]
    metrics_with: dict[int, CitationMetrics]
    target_index: int
    target_cited: bool
    target_imp_wc: float
    target_imp_pwc: float
    # 一次情報源の引用率
    primary_source_rate_without: float
    primary_source_rate_with: float
    primary_source_rate_diff: float  # with - without


# Note: Stats dataclass is defined later in the file after utility functions


@dataclass
class TargetSummary:
    """ターゲットの集計結果"""
    target_id: str
    title: str  # H1タグの内容
    domain: str
    provider: str
    num_runs: int
    # ターゲットが引用された割合
    citation_rate: float
    # Imp_wc の統計
    imp_wc: "Stats"
    # Imp_pwc の統計
    imp_pwc: "Stats"
    # 一次情報源の引用率（without）
    primary_rate_without: "Stats"
    # 一次情報源の引用率（with）
    primary_rate_with: "Stats"
    # 一次情報源の引用率の変化（with - without）
    primary_rate_diff: "Stats"


@dataclass
class DomainSummary:
    """ドメインの集計結果"""
    domain: str
    provider: str
    num_runs_per_target: int
    # 全ターゲット平均
    avg_citation_rate: float
    avg_imp_wc: float
    avg_imp_pwc: float
    # 一次情報源の引用率
    avg_primary_rate_without: float
    avg_primary_rate_with: float
    avg_primary_rate_diff: float
    # ターゲットごとの詳細
    target_summaries: list[TargetSummary]


# =============================================================================
# Utility Functions
# =============================================================================

def load_target_file(file_path: str) -> str:
    """ターゲットファイルを読み込み、Markdownを除去"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return strip_markdown(content)


def extract_h1_title(file_path: str) -> str:
    """MDファイルから最初のH1タグの内容を抽出"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # H1タグ（# で始まり ## ではない）を検出
            match = re.match(r'^#\s+(.+)$', line)
            if match:
                return match.group(1).strip()
    # H1が見つからない場合はファイル名を使用
    return Path(file_path).stem


def discover_targets(targets_dir: str, domains: list[str]) -> list[TargetConfig]:
    """指定ドメインのターゲットファイルを検出"""
    targets = []
    base_path = Path(targets_dir)

    for domain in domains:
        domain_path = base_path / domain
        if not domain_path.exists():
            print(f"警告: ドメインディレクトリが見つかりません: {domain_path}")
            continue

        for md_file in sorted(domain_path.glob("*.md")):
            target_id = md_file.stem
            title = extract_h1_title(str(md_file))
            targets.append({
                "id": target_id,
                "file": str(md_file),
                "url": f"https://jimin.jp/TARGET",
                "domain": domain,
                "title": title,
            })

    return targets


def sanitize_folder_name(name: str) -> str:
    """フォルダ名に使用できない文字を置換"""
    # ファイルシステムで問題になる文字を置換
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '_')
    # 長すぎる場合は切り詰め
    if len(name) > 50:
        name = name[:50]
    return name


def format_sources(sources: list[SourceContent]) -> str:
    """ソースをプロンプト用にフォーマット"""
    lines = []
    for i, source in enumerate(sources, 1):
        url = source["url"]
        content = source["content"]
        lines.append(f"[{i}] URL: {url}")
        lines.append(f"    Content: {content}")
        lines.append("")
    return "\n".join(lines)


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
    """統計値を計算"""
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


def is_primary_source(url: str, primary_domains: list[str]) -> bool:
    """URLが一次情報源かどうかを判定（ドメイン部分一致）"""
    url_lower = url.lower()
    for domain in primary_domains:
        if domain.lower() in url_lower:
            return True
    return False


def calc_primary_source_rate(
    metrics: dict[int, CitationMetrics],
    sources: list[SourceContent],
    primary_domains: list[str],
    exclude_indices: set[int] | None = None,
) -> float:
    """
    一次情報源の引用率を計算（引用されたソースのうち一次情報源の割合）

    Args:
        metrics: 引用メトリクス
        sources: ソースリスト
        primary_domains: 一次情報源のドメインリスト
        exclude_indices: 計算から除外するインデックス（1-indexed、ターゲット用）
    """
    if not metrics:
        return 0.0

    exclude_indices = exclude_indices or set()

    # ターゲットを除外した合計を計算
    total_imp_wc = 0.0
    primary_imp_wc = 0.0

    for idx, m in metrics.items():
        if idx in exclude_indices:
            continue  # ターゲットは除外

        source_idx = idx - 1  # 0-indexed
        if 0 <= source_idx < len(sources):
            total_imp_wc += m.imp_wc
            url = sources[source_idx]["url"]
            if is_primary_source(url, primary_domains):
                primary_imp_wc += m.imp_wc

    return (primary_imp_wc / total_imp_wc) * 100 if total_imp_wc > 0 else 0.0


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """実験実行クラス"""

    def __init__(
        self,
        providers: list[str],
        num_runs: int,
        max_sources: int,
        output_base_dir: str,
        primary_domains: list[str],
    ):
        self.providers = providers
        self.num_runs = num_runs
        self.max_sources = max_sources
        self.output_base_dir = Path(output_base_dir)
        self.primary_domains = primary_domains
        self.fetcher = WebContentFetcher()
        self.analyzer = CitationAnalyzer()
        self.llm_clients: dict[str, LLMClient] = {}

    def _get_llm(self, provider: str) -> LLMClient:
        """LLMクライアントを取得（キャッシュ）"""
        if provider not in self.llm_clients:
            self.llm_clients[provider] = create_llm_client(provider)
        return self.llm_clients[provider]

    async def run_experiment(
        self,
        targets: list[TargetConfig],
        domain_prompts: dict[str, str],
    ) -> Path:
        """実験を実行"""
        # 出力ディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # 実験設定を保存
        self._save_json(output_dir / "experiment_config.json", {
            "providers": self.providers,
            "num_runs": self.num_runs,
            "max_sources": self.max_sources,
            "primary_domains": self.primary_domains,
            "domain_prompts": domain_prompts,
            "targets": [{"id": t["id"], "domain": t["domain"]} for t in targets],
            "timestamp": datetime.now().isoformat(),
        })

        # ドメインごとに集計
        all_domain_summaries: dict[str, dict[str, DomainSummary]] = {}

        # ターゲットをドメインごとにグループ化
        targets_by_domain: dict[str, list[TargetConfig]] = {}
        for target in targets:
            domain = target["domain"]
            if domain not in targets_by_domain:
                targets_by_domain[domain] = []
            targets_by_domain[domain].append(target)

        # 各ドメイン×プロバイダーで実行
        for domain, domain_targets in targets_by_domain.items():
            question = domain_prompts.get(domain)
            if not question:
                print(f"警告: ドメイン {domain} のプロンプトが定義されていません")
                continue

            domain_dir = output_dir / domain
            domain_dir.mkdir(exist_ok=True)

            all_domain_summaries[domain] = {}

            for provider in self.providers:
                print(f"\n{'='*60}")
                print(f"ドメイン: {domain} | プロバイダー: {provider}")
                print(f"ターゲット数: {len(domain_targets)}件を並列実行")
                print(f"{'='*60}")

                llm = self._get_llm(provider)

                # 全ターゲットを並列で実行
                target_tasks = [
                    self._run_target_experiment(
                        llm=llm,
                        provider=provider,
                        target=target,
                        question=question,
                        output_dir=domain_dir,
                    )
                    for target in domain_targets
                ]
                target_summaries = await asyncio.gather(*target_tasks)

                # ドメイン集計
                domain_summary = self._aggregate_domain(
                    domain=domain,
                    provider=provider,
                    target_summaries=list(target_summaries),
                )
                all_domain_summaries[domain][provider] = domain_summary

                # ドメインサマリーを保存
                self._save_domain_summary(domain_dir, provider, domain_summary)

        # ルートサマリーを保存
        self._save_root_summary(output_dir, all_domain_summaries)

        # リソースをクリーンアップ
        await self.fetcher.close()

        print(f"\n完了: {output_dir}")
        return output_dir

    async def _run_target_experiment(
        self,
        llm: LLMClient,
        provider: str,
        target: TargetConfig,
        question: str,
        output_dir: Path,
    ) -> TargetSummary:
        """1ターゲットの実験を実行"""
        target_title = target['title']
        folder_name = f"{provider}_{sanitize_folder_name(target_title)}"
        target_dir = output_dir / folder_name
        target_dir.mkdir(exist_ok=True)

        # 1. Web検索（1回のみ）
        print(f"  [{provider}][{target_title}] Web検索開始")
        source_urls = await self._search_web(llm, question)

        # 2. ソースを取得
        sources = await self._fetch_sources(source_urls)

        # ソースを保存
        self._save_json(target_dir / "sources.json", [
            {"index": i + 1, "url": s["url"], "content": s["content"], "media_type": s["media_type"]}
            for i, s in enumerate(sources)
        ])

        # 3. ターゲットコンテンツを準備
        target_content = load_target_file(target["file"])
        target_source: SourceContent = {
            "url": target["url"],
            "content": target_content,
            "media_type": "TARGET",
        }

        # sources_with_targets: [Webソース1, ..., WebソースN, ターゲット]
        sources_with_targets = sources.copy()
        sources_with_targets.append(target_source)
        target_index = len(sources_with_targets)  # 1-indexed

        # 4. 回答生成をN回並列で実行
        print(f"  [{provider}][{target_title}] 回答生成 {self.num_runs}回を並列実行中...")

        # 全run分のwithout/with回答を一括で並列生成
        # タスクリスト: [without_1, with_1, without_2, with_2, ...]
        answer_tasks = []
        for _ in range(self.num_runs):
            answer_tasks.append(self._generate_answer(llm, question, sources))
            answer_tasks.append(self._generate_answer(llm, question, sources_with_targets))

        all_answers = await asyncio.gather(*answer_tasks)

        # 結果を処理
        run_results: list[RunResult] = []
        for run_idx in range(1, self.num_runs + 1):
            # all_answers: [without_1, with_1, without_2, with_2, ...]
            answer_without = all_answers[(run_idx - 1) * 2]
            answer_with = all_answers[(run_idx - 1) * 2 + 1]

            # メトリクス計算
            metrics_without = self.analyzer.analyze(answer_without, sources)
            metrics_with = self.analyzer.analyze(answer_with, sources_with_targets)

            # ターゲットの引用情報
            target_metrics = metrics_with.get(target_index)
            target_cited = target_metrics is not None
            target_imp_wc = target_metrics.imp_wc if target_metrics else 0.0
            target_imp_pwc = target_metrics.imp_pwc if target_metrics else 0.0

            # 一次情報源の引用率を計算
            # without: 全ソースを対象
            primary_rate_without = calc_primary_source_rate(
                metrics_without, sources, self.primary_domains
            )
            # with: ターゲットを除外して計算
            primary_rate_with = calc_primary_source_rate(
                metrics_with, sources, self.primary_domains
            )
            primary_rate_diff = primary_rate_with - primary_rate_without

            run_result = RunResult(
                run_index=run_idx,
                answer_without=answer_without,
                answer_with=answer_with,
                metrics_without=metrics_without,
                metrics_with=metrics_with,
                target_index=target_index,
                target_cited=target_cited,
                target_imp_wc=target_imp_wc,
                target_imp_pwc=target_imp_pwc,
                primary_source_rate_without=primary_rate_without,
                primary_source_rate_with=primary_rate_with,
                primary_source_rate_diff=primary_rate_diff,
            )
            run_results.append(run_result)

            # 個別結果を保存
            self._save_run_result(target_dir, run_idx, run_result, sources, sources_with_targets)

        # ターゲット集計
        target_summary = self._aggregate_target(
            target_id=target["id"],
            title=target["title"],
            domain=target["domain"],
            provider=provider,
            run_results=run_results,
        )

        # ターゲットサマリーを保存
        self._save_target_summary(target_dir, target_summary)

        print(f"  [{provider}][{target_title}] 完了 (citation_rate: {target_summary.citation_rate:.1%})")
        return target_summary

    async def _search_web(self, llm: LLMClient, query: str) -> list[str]:
        """Web検索を実行"""
        results = await llm.search_web(query, max_results=self.max_sources)
        return [r["url"] for r in results if "url" in r]

    async def _fetch_sources(self, urls: list[str]) -> list[SourceContent]:
        """複数URLからコンテンツを並列取得"""
        tasks = [self.fetcher.fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)

        sources: list[SourceContent] = []
        for url, (content, media_type) in zip(urls, results):
            sources.append({
                "url": url,
                "content": content,
                "media_type": media_type,
            })
        return sources

    async def _generate_answer(
        self,
        llm: LLMClient,
        question: str,
        sources: list[SourceContent],
    ) -> str:
        """回答を生成"""
        sources_text = format_sources(sources)
        prompt = PROMPT_TEMPLATE.format(question=question, sources=sources_text)
        return await llm.acall_standard(prompt)

    def _aggregate_target(
        self,
        target_id: str,
        title: str,
        domain: str,
        provider: str,
        run_results: list[RunResult],
    ) -> TargetSummary:
        """ターゲットの結果を集計"""
        cited_count = sum(1 for r in run_results if r.target_cited)

        return TargetSummary(
            target_id=target_id,
            title=title,
            domain=domain,
            provider=provider,
            num_runs=len(run_results),
            citation_rate=cited_count / len(run_results) if run_results else 0.0,
            imp_wc=calc_stats([r.target_imp_wc for r in run_results]),
            imp_pwc=calc_stats([r.target_imp_pwc for r in run_results]),
            primary_rate_without=calc_stats([r.primary_source_rate_without for r in run_results]),
            primary_rate_with=calc_stats([r.primary_source_rate_with for r in run_results]),
            primary_rate_diff=calc_stats([r.primary_source_rate_diff for r in run_results]),
        )

    def _aggregate_domain(
        self,
        domain: str,
        provider: str,
        target_summaries: list[TargetSummary],
    ) -> DomainSummary:
        """ドメインの結果を集計"""
        if not target_summaries:
            return DomainSummary(
                domain=domain,
                provider=provider,
                num_runs_per_target=0,
                avg_citation_rate=0.0,
                avg_imp_wc=0.0,
                avg_imp_pwc=0.0,
                avg_primary_rate_without=0.0,
                avg_primary_rate_with=0.0,
                avg_primary_rate_diff=0.0,
                target_summaries=[],
            )

        avg_citation_rate = statistics.mean([t.citation_rate for t in target_summaries])
        avg_imp_wc = statistics.mean([t.imp_wc.mean for t in target_summaries])
        avg_imp_pwc = statistics.mean([t.imp_pwc.mean for t in target_summaries])
        avg_primary_without = statistics.mean([t.primary_rate_without.mean for t in target_summaries])
        avg_primary_with = statistics.mean([t.primary_rate_with.mean for t in target_summaries])
        avg_primary_diff = statistics.mean([t.primary_rate_diff.mean for t in target_summaries])

        return DomainSummary(
            domain=domain,
            provider=provider,
            num_runs_per_target=target_summaries[0].num_runs,
            avg_citation_rate=avg_citation_rate,
            avg_imp_wc=avg_imp_wc,
            avg_imp_pwc=avg_imp_pwc,
            avg_primary_rate_without=avg_primary_without,
            avg_primary_rate_with=avg_primary_with,
            avg_primary_rate_diff=avg_primary_diff,
            target_summaries=target_summaries,
        )

    # =========================================================================
    # File Output
    # =========================================================================

    def _save_json(self, path: Path, data: dict | list):
        """JSONを保存"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _save_csv(self, path: Path, headers: list, rows: list):
        """CSVを保存"""
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def _save_run_result(
        self,
        target_dir: Path,
        run_idx: int,
        result: RunResult,
        sources: list[SourceContent],
        sources_with_targets: list[SourceContent],
    ):
        """1回分の結果を保存"""
        prefix = f"{run_idx}_"

        # 回答（without）
        self._save_json(target_dir / f"{prefix}answer_without.json", {
            "run_index": run_idx,
            "include_target": False,
            "answer": result.answer_without,
        })

        # 回答（with）
        self._save_json(target_dir / f"{prefix}answer_with.json", {
            "run_index": run_idx,
            "include_target": True,
            "target_index": result.target_index,
            "answer": result.answer_with,
        })

        # メトリクス（without）
        metrics_without_rows = []
        for i in sorted(result.metrics_without.keys()):
            m = result.metrics_without[i]
            metrics_without_rows.append([
                i,
                sources[i - 1]["url"],
                round(m.imp_wc, 2),
                round(m.imp_pwc, 2),
                m.citation_frequency,
                m.first_citation_position if m.first_citation_position is not None else "N/A",
            ])
        self._save_csv(
            target_dir / f"{prefix}metrics_without.csv",
            ["index", "url", "imp_wc", "imp_pwc", "citation_frequency", "first_position"],
            metrics_without_rows,
        )

        # メトリクス（with）
        metrics_with_rows = []
        num_sources = len(sources)
        for i in sorted(result.metrics_with.keys()):
            m = result.metrics_with[i]
            is_target = i > num_sources
            if is_target:
                url = sources_with_targets[i - 1]["url"]
            else:
                url = sources[i - 1]["url"]
            metrics_with_rows.append([
                i,
                url,
                "Yes" if is_target else "No",
                round(m.imp_wc, 2),
                round(m.imp_pwc, 2),
                m.citation_frequency,
                m.first_citation_position if m.first_citation_position is not None else "N/A",
            ])
        self._save_csv(
            target_dir / f"{prefix}metrics_with.csv",
            ["index", "url", "is_target", "imp_wc", "imp_pwc", "citation_frequency", "first_position"],
            metrics_with_rows,
        )

    @staticmethod
    def _stats_to_dict(stats: Stats, decimals: int = 2) -> dict:
        """Stats を辞書に変換"""
        return {
            "mean": round(stats.mean, decimals),
            "std": round(stats.std, decimals),
            "min": round(stats.min, decimals),
            "max": round(stats.max, decimals),
            "median": round(stats.median, decimals),
            "values": [round(v, decimals) for v in stats.values],
        }

    def _save_target_summary(self, target_dir: Path, summary: TargetSummary):
        """ターゲットサマリーを保存"""
        self._save_json(target_dir / "summary.json", {
            "target_id": summary.target_id,
            "title": summary.title,
            "domain": summary.domain,
            "provider": summary.provider,
            "num_runs": summary.num_runs,
            "citation_rate": round(summary.citation_rate, 3),
            "imp_wc": self._stats_to_dict(summary.imp_wc),
            "imp_pwc": self._stats_to_dict(summary.imp_pwc),
            "primary_source_rate": {
                "without": self._stats_to_dict(summary.primary_rate_without),
                "with": self._stats_to_dict(summary.primary_rate_with),
                "diff": self._stats_to_dict(summary.primary_rate_diff),
            },
        })

    def _save_domain_summary(self, domain_dir: Path, provider: str, summary: DomainSummary):
        """ドメインサマリーを保存"""
        self._save_json(domain_dir / f"{provider}_summary.json", {
            "domain": summary.domain,
            "provider": summary.provider,
            "num_runs_per_target": summary.num_runs_per_target,
            "avg_citation_rate": round(summary.avg_citation_rate, 3),
            "avg_imp_wc": round(summary.avg_imp_wc, 2),
            "avg_imp_pwc": round(summary.avg_imp_pwc, 2),
            "primary_source_rate": {
                "avg_without": round(summary.avg_primary_rate_without, 2),
                "avg_with": round(summary.avg_primary_rate_with, 2),
                "avg_diff": round(summary.avg_primary_rate_diff, 2),
            },
            "targets": [
                {
                    "target_id": t.target_id,
                    "title": t.title,
                    "citation_rate": round(t.citation_rate, 3),
                    "imp_wc": self._stats_to_dict(t.imp_wc),
                    "imp_pwc": self._stats_to_dict(t.imp_pwc),
                    "primary_rate_without": self._stats_to_dict(t.primary_rate_without),
                    "primary_rate_with": self._stats_to_dict(t.primary_rate_with),
                    "primary_rate_diff": self._stats_to_dict(t.primary_rate_diff),
                }
                for t in summary.target_summaries
            ],
        })

    def _save_root_summary(
        self,
        output_dir: Path,
        all_summaries: dict[str, dict[str, DomainSummary]],
    ):
        """ルートサマリーを保存"""
        root_data = {
            "timestamp": datetime.now().isoformat(),
            "providers": self.providers,
            "num_runs": self.num_runs,
            "primary_domains": self.primary_domains,
            "domains": {},
        }

        for domain, provider_summaries in all_summaries.items():
            root_data["domains"][domain] = {}
            for provider, summary in provider_summaries.items():
                root_data["domains"][domain][provider] = {
                    "avg_citation_rate": round(summary.avg_citation_rate, 3),
                    "avg_imp_wc": round(summary.avg_imp_wc, 2),
                    "avg_imp_pwc": round(summary.avg_imp_pwc, 2),
                    "primary_source_rate": {
                        "avg_without": round(summary.avg_primary_rate_without, 2),
                        "avg_with": round(summary.avg_primary_rate_with, 2),
                        "avg_diff": round(summary.avg_primary_rate_diff, 2),
                    },
                }

        # 全体の平均も計算
        all_citation_rates = []
        all_imp_wc = []
        all_imp_pwc = []
        all_primary_without = []
        all_primary_with = []
        all_primary_diff = []

        for provider_summaries in all_summaries.values():
            for summary in provider_summaries.values():
                if summary.target_summaries:  # ターゲットが存在する場合
                    all_citation_rates.append(summary.avg_citation_rate)
                    all_imp_wc.append(summary.avg_imp_wc)
                    all_imp_pwc.append(summary.avg_imp_pwc)
                    all_primary_without.append(summary.avg_primary_rate_without)
                    all_primary_with.append(summary.avg_primary_rate_with)
                    all_primary_diff.append(summary.avg_primary_rate_diff)

        if all_citation_rates:
            root_data["overall"] = {
                "avg_citation_rate": round(statistics.mean(all_citation_rates), 3),
                "avg_imp_wc": round(statistics.mean(all_imp_wc), 2),
                "avg_imp_pwc": round(statistics.mean(all_imp_pwc), 2),
                "primary_source_rate": {
                    "avg_without": round(statistics.mean(all_primary_without), 2),
                    "avg_with": round(statistics.mean(all_primary_with), 2),
                    "avg_diff": round(statistics.mean(all_primary_diff), 2),
                },
            }

        self._save_json(output_dir / "summary.json", root_data)


# =============================================================================
# Main
# =============================================================================

async def main():
    config = EXPERIMENT_CONFIG

    # ターゲットを検出
    domains = list(DOMAIN_PROMPTS.keys())
    targets = discover_targets(config["targets_dir"], domains)

    if not targets:
        print("エラー: ターゲットが見つかりません")
        sys.exit(1)

    print(f"検出されたターゲット: {len(targets)}件")
    for domain in domains:
        count = sum(1 for t in targets if t["domain"] == domain)
        print(f"  {domain}: {count}件")

    # 実験を実行
    runner = ExperimentRunner(
        providers=config["providers"],
        num_runs=config["num_runs"],
        max_sources=config["max_sources"],
        output_base_dir=config["output_dir"],
        primary_domains=config.get("primary_domains", []),
    )

    await runner.run_experiment(targets, DOMAIN_PROMPTS)


if __name__ == "__main__":
    asyncio.run(main())
