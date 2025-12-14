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

import argparse
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

# 質問タイプ
QUESTION_TYPES = ["vague", "experiment", "aligned"]

# 実験設定
EXPERIMENT_CONFIG = {
    # "providers": ["gemini", "gpt"],  # 使用するプロバイダー
    "providers": ["gpt"],  # 使用するプロバイダー
    "num_runs": 5,                   # 各ターゲットの繰り返し回数
    "max_sources": 5,                # Web検索で取得するソース数
    "targets_dir": "targets",        # ターゲットファイルのディレクトリ
    "output_dir": "outputs",         # 出力ディレクトリ
    "questions_cache_file": "questions_cache.json",  # 質問キャッシュファイル
    # 一次情報源と判定するドメイン（部分一致）
    "primary_domains": [
        "jimin.jp",
    ],
}

# 質問生成プロンプト
QUESTION_GENERATION_PROMPT = """あなたは政策に関する質問を生成するアシスタントです。
以下のターゲット記事のタイトルと内容に基づいて、3種類の質問を日本語で生成してください。

【ターゲット記事】
タイトル: {title}
内容: {content}

【生成する質問の種類】

1. vague（抽象的な質問）:
ターゲットのタイトルを一段抽象化したカテゴリーについて、総合的に知りたいという質問。
例: タイトルが「物価・インフレ対策」なら「自民党の経済政策の柱になっているものを総合的に知りたいです。」

2. experiment（中間的な質問）:
ターゲットのタイトルを一段抽象化した形の質問。タイトルの具体的なテーマについて聞く。
例: タイトルが「物価・インフレ対策」なら「自民党の物価上昇についてどう考えていますか？」

3. aligned（具体的な質問）:
ターゲットの記事内容に完全に沿った、具体的で詳細な質問。記事の核心的な内容について深く聞く。
例: 「自民党は物価・インフレ対策や賃上げ・雇用・労働市場の政策で、物価上昇に負けない賃金アップを全国に広げるため、どのような仕組みを企業や労働者と一緒につくろうとしているのですか。」

【出力形式】
以下のJSON形式で出力してください。余計な説明は不要です。
{{
  "vague": "抽象的な質問文",
  "experiment": "中間的な質問文",
  "aligned": "具体的な質問文"
}}"""

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
    content: str  # ファイル内容（質問生成用）


class GeneratedQuestions(TypedDict):
    """生成された質問"""
    vague: str
    experiment: str
    aligned: str


@dataclass
class SourceScores:
    """ソース別スコア（一次情報源/非一次情報源）"""
    # 一次情報源のスコア
    primary_imp_wc_values: list[float]
    primary_imp_pwc_values: list[float]
    # 非一次情報源のスコア
    non_primary_imp_wc_values: list[float]
    non_primary_imp_pwc_values: list[float]


@dataclass
class RunResult:
    """1回の実行結果"""
    run_index: int
    question_type: str  # vague, experiment, aligned
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
    # ソース別スコア
    source_scores_without: SourceScores
    source_scores_with: SourceScores


# Note: Stats dataclass is defined later in the file after utility functions


@dataclass
class SourceScoreStats:
    """ソース別スコア統計"""
    # 一次情報源
    primary_imp_wc: "Stats"
    primary_imp_pwc: "Stats"
    # 非一次情報源
    non_primary_imp_wc: "Stats"
    non_primary_imp_pwc: "Stats"


@dataclass
class TargetSummary:
    """ターゲットの集計結果"""
    target_id: str
    title: str  # H1タグの内容
    domain: str
    provider: str
    question_type: str  # vague, experiment, aligned
    question: str  # 使用した質問
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
    # ソース別スコア（without/with）
    source_scores_without: SourceScoreStats
    source_scores_with: SourceScoreStats


@dataclass
class QuestionTypeSummary:
    """質問タイプ別の集計結果"""
    question_type: str
    provider: str
    num_targets: int
    num_runs_per_target: int
    # 集計値（avg と median）
    citation_rate_avg: float
    citation_rate_median: float
    imp_wc_avg: float
    imp_wc_median: float
    imp_pwc_avg: float
    imp_pwc_median: float
    # 一次情報源率
    primary_rate_without_avg: float
    primary_rate_without_median: float
    primary_rate_with_avg: float
    primary_rate_with_median: float
    primary_rate_diff_avg: float
    primary_rate_diff_median: float
    # ソース別スコア
    source_scores_without: SourceScoreStats
    source_scores_with: SourceScoreStats
    # ターゲットごとの詳細
    target_summaries: list[TargetSummary]


@dataclass
class DomainSummary:
    """ドメインの集計結果"""
    domain: str
    provider: str
    num_runs_per_target: int
    # 質問タイプ別の集計
    question_type_summaries: dict[str, QuestionTypeSummary]  # key: question_type
    # 全体集計（全質問タイプ合算）
    all_summary: QuestionTypeSummary


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
            content = load_target_file(str(md_file))
            targets.append({
                "id": target_id,
                "file": str(md_file),
                "url": f"https://jimin.jp/TARGET",
                "domain": domain,
                "title": title,
                "content": content,
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


def calc_source_scores(
    metrics: dict[int, CitationMetrics],
    sources: list[SourceContent],
    primary_domains: list[str],
    exclude_indices: set[int] | None = None,
) -> SourceScores:
    """
    一次情報源/非一次情報源のスコアを計算

    Args:
        metrics: 引用メトリクス
        sources: ソースリスト
        primary_domains: 一次情報源のドメインリスト
        exclude_indices: 計算から除外するインデックス（1-indexed、ターゲット用）
    """
    exclude_indices = exclude_indices or set()

    primary_imp_wc = []
    primary_imp_pwc = []
    non_primary_imp_wc = []
    non_primary_imp_pwc = []

    for idx, m in metrics.items():
        if idx in exclude_indices:
            continue

        source_idx = idx - 1  # 0-indexed
        if 0 <= source_idx < len(sources):
            url = sources[source_idx]["url"]
            if is_primary_source(url, primary_domains):
                primary_imp_wc.append(m.imp_wc)
                primary_imp_pwc.append(m.imp_pwc)
            else:
                non_primary_imp_wc.append(m.imp_wc)
                non_primary_imp_pwc.append(m.imp_pwc)

    return SourceScores(
        primary_imp_wc_values=primary_imp_wc,
        primary_imp_pwc_values=primary_imp_pwc,
        non_primary_imp_wc_values=non_primary_imp_wc,
        non_primary_imp_pwc_values=non_primary_imp_pwc,
    )


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
# Question Generator
# =============================================================================

class QuestionGenerator:
    """質問生成クラス"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def generate(self, title: str, content: str) -> GeneratedQuestions:
        """
        ターゲットのタイトルと内容から3種類の質問を生成

        Args:
            title: ターゲットのタイトル
            content: ターゲットの内容

        Returns:
            GeneratedQuestions: vague, experiment, aligned の3種類の質問
        """
        prompt = QUESTION_GENERATION_PROMPT.format(title=title, content=content)
        response = await self.llm.acall_standard(prompt)

        # JSONをパース
        try:
            # JSON部分を抽出（```json ... ``` で囲まれている場合に対応）
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                questions = json.loads(response)

            return GeneratedQuestions(
                vague=questions.get("vague", ""),
                experiment=questions.get("experiment", ""),
                aligned=questions.get("aligned", ""),
            )
        except json.JSONDecodeError as e:
            print(f"警告: 質問生成のJSONパースに失敗: {e}")
            print(f"レスポンス: {response[:200]}...")
            # フォールバック: デフォルトの質問を返す
            return GeneratedQuestions(
                vague=f"{title}について教えてください。",
                experiment=f"{title}についての政策を教えてください。",
                aligned=f"{title}の具体的な内容を詳しく教えてください。",
            )


def load_questions_cache(cache_file: str) -> dict[str, GeneratedQuestions]:
    """質問キャッシュを読み込む"""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # dict[target_id, GeneratedQuestions]
        return {
            target_id: GeneratedQuestions(
                vague=q["vague"],
                experiment=q["experiment"],
                aligned=q["aligned"],
            )
            for target_id, q in data.items()
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"警告: キャッシュファイルの読み込みに失敗: {e}")
        return {}


def save_questions_cache(cache_file: str, cache: dict[str, GeneratedQuestions]):
    """質問キャッシュを保存"""
    data = {
        target_id: {
            "vague": q["vague"],
            "experiment": q["experiment"],
            "aligned": q["aligned"],
        }
        for target_id, q in cache.items()
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def generate_all_questions(
    targets: list[TargetConfig],
    llm: LLMClient,
    cache_file: str,
) -> dict[str, GeneratedQuestions]:
    """
    全ターゲットの質問を生成（キャッシュ利用）

    Args:
        targets: ターゲットリスト
        llm: LLMクライアント
        cache_file: キャッシュファイルパス

    Returns:
        dict[target_id, GeneratedQuestions]
    """
    # キャッシュを読み込む
    cache = load_questions_cache(cache_file)

    # 未生成のターゲットを抽出
    targets_to_generate = [t for t in targets if t["id"] not in cache]

    if not targets_to_generate:
        print(f"質問キャッシュから{len(targets)}件の質問を読み込みました")
        return cache

    print(f"質問生成: {len(targets_to_generate)}件（キャッシュ済み: {len(cache)}件）")

    generator = QuestionGenerator(llm)

    # 並列で質問を生成
    async def generate_for_target(target: TargetConfig) -> tuple[str, GeneratedQuestions]:
        print(f"  質問生成中: {target['title']}")
        questions = await generator.generate(target["title"], target["content"])
        return target["id"], questions

    tasks = [generate_for_target(t) for t in targets_to_generate]
    results = await asyncio.gather(*tasks)

    # キャッシュに追加
    for target_id, questions in results:
        cache[target_id] = questions

    # キャッシュを保存
    save_questions_cache(cache_file, cache)
    print(f"質問キャッシュを保存しました: {cache_file}")

    return cache


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
        questions: dict[str, GeneratedQuestions],
    ) -> Path:
        """
        実験を実行

        Args:
            targets: ターゲットリスト
            questions: ターゲットIDごとの質問 dict[target_id, GeneratedQuestions]
        """
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
            "question_types": QUESTION_TYPES,
            "targets": [{"id": t["id"], "domain": t["domain"], "title": t["title"]} for t in targets],
            "questions": {tid: dict(q) for tid, q in questions.items()},
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
            domain_dir = output_dir / domain
            domain_dir.mkdir(exist_ok=True)

            all_domain_summaries[domain] = {}

            for provider in self.providers:
                print(f"\n{'='*60}")
                print(f"ドメイン: {domain} | プロバイダー: {provider}")
                print(f"ターゲット数: {len(domain_targets)} × 質問タイプ: {len(QUESTION_TYPES)}")
                print(f"{'='*60}")

                llm = self._get_llm(provider)

                # 全ターゲット×全質問タイプを並列で実行
                target_tasks = []
                for target in domain_targets:
                    target_questions = questions.get(target["id"])
                    if not target_questions:
                        print(f"警告: ターゲット {target['id']} の質問が見つかりません")
                        continue

                    for q_type in QUESTION_TYPES:
                        question = target_questions[q_type]
                        target_tasks.append(
                            self._run_target_experiment(
                                llm=llm,
                                provider=provider,
                                target=target,
                                question=question,
                                question_type=q_type,
                                output_dir=domain_dir,
                            )
                        )

                all_target_summaries = await asyncio.gather(*target_tasks)

                # ドメイン集計（質問タイプ別に自動分類）
                domain_summary = self._aggregate_domain(
                    domain=domain,
                    provider=provider,
                    all_target_summaries=list(all_target_summaries),
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
        question_type: str,
        output_dir: Path,
    ) -> TargetSummary:
        """1ターゲット×1質問タイプの実験を実行"""
        target_title = target['title']
        folder_name = f"{provider}_{question_type}_{sanitize_folder_name(target_title)}"
        target_dir = output_dir / folder_name
        target_dir.mkdir(exist_ok=True)

        # 1. Web検索（1回のみ）
        print(f"  [{provider}][{question_type}][{target_title}] Web検索開始")
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
        print(f"  [{provider}][{question_type}][{target_title}] 回答生成 {self.num_runs}回を並列実行中...")

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

            # ソース別スコアを計算
            source_scores_without = calc_source_scores(
                metrics_without, sources, self.primary_domains
            )
            source_scores_with = calc_source_scores(
                metrics_with, sources, self.primary_domains, exclude_indices={target_index}
            )

            run_result = RunResult(
                run_index=run_idx,
                question_type=question_type,
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
                source_scores_without=source_scores_without,
                source_scores_with=source_scores_with,
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
            question_type=question_type,
            question=question,
            run_results=run_results,
        )

        # ターゲットサマリーを保存
        self._save_target_summary(target_dir, target_summary)

        print(f"  [{provider}][{question_type}][{target_title}] 完了 (citation_rate: {target_summary.citation_rate:.1%})")
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
        question_type: str,
        question: str,
        run_results: list[RunResult],
    ) -> TargetSummary:
        """ターゲットの結果を集計"""
        cited_count = sum(1 for r in run_results if r.target_cited)

        # ソース別スコアを集計
        def aggregate_source_scores(
            scores_list: list[SourceScores],
        ) -> SourceScoreStats:
            """複数RunのSourceScoresを集計してSourceScoreStatsに"""
            primary_wc_all = []
            primary_pwc_all = []
            non_primary_wc_all = []
            non_primary_pwc_all = []

            for scores in scores_list:
                primary_wc_all.extend(scores.primary_imp_wc_values)
                primary_pwc_all.extend(scores.primary_imp_pwc_values)
                non_primary_wc_all.extend(scores.non_primary_imp_wc_values)
                non_primary_pwc_all.extend(scores.non_primary_imp_pwc_values)

            return SourceScoreStats(
                primary_imp_wc=calc_stats(primary_wc_all),
                primary_imp_pwc=calc_stats(primary_pwc_all),
                non_primary_imp_wc=calc_stats(non_primary_wc_all),
                non_primary_imp_pwc=calc_stats(non_primary_pwc_all),
            )

        source_scores_without = aggregate_source_scores(
            [r.source_scores_without for r in run_results]
        )
        source_scores_with = aggregate_source_scores(
            [r.source_scores_with for r in run_results]
        )

        return TargetSummary(
            target_id=target_id,
            title=title,
            domain=domain,
            provider=provider,
            question_type=question_type,
            question=question,
            num_runs=len(run_results),
            citation_rate=cited_count / len(run_results) if run_results else 0.0,
            imp_wc=calc_stats([r.target_imp_wc for r in run_results]),
            imp_pwc=calc_stats([r.target_imp_pwc for r in run_results]),
            primary_rate_without=calc_stats([r.primary_source_rate_without for r in run_results]),
            primary_rate_with=calc_stats([r.primary_source_rate_with for r in run_results]),
            primary_rate_diff=calc_stats([r.primary_source_rate_diff for r in run_results]),
            source_scores_without=source_scores_without,
            source_scores_with=source_scores_with,
        )

    def _aggregate_question_type(
        self,
        question_type: str,
        provider: str,
        target_summaries: list[TargetSummary],
    ) -> QuestionTypeSummary:
        """質問タイプの結果を集計"""
        if not target_summaries:
            empty_stats = calc_stats([])
            empty_source_stats = SourceScoreStats(
                primary_imp_wc=empty_stats,
                primary_imp_pwc=empty_stats,
                non_primary_imp_wc=empty_stats,
                non_primary_imp_pwc=empty_stats,
            )
            return QuestionTypeSummary(
                question_type=question_type,
                provider=provider,
                num_targets=0,
                num_runs_per_target=0,
                citation_rate_avg=0.0,
                citation_rate_median=0.0,
                imp_wc_avg=0.0,
                imp_wc_median=0.0,
                imp_pwc_avg=0.0,
                imp_pwc_median=0.0,
                primary_rate_without_avg=0.0,
                primary_rate_without_median=0.0,
                primary_rate_with_avg=0.0,
                primary_rate_with_median=0.0,
                primary_rate_diff_avg=0.0,
                primary_rate_diff_median=0.0,
                source_scores_without=empty_source_stats,
                source_scores_with=empty_source_stats,
                target_summaries=[],
            )

        citation_rates = [t.citation_rate for t in target_summaries]
        imp_wc_means = [t.imp_wc.mean for t in target_summaries]
        imp_pwc_means = [t.imp_pwc.mean for t in target_summaries]
        primary_without_means = [t.primary_rate_without.mean for t in target_summaries]
        primary_with_means = [t.primary_rate_with.mean for t in target_summaries]
        primary_diff_means = [t.primary_rate_diff.mean for t in target_summaries]

        # ソース別スコアを集計（全ターゲットの全値をフラットに）
        def aggregate_all_source_scores(
            get_scores: callable,
        ) -> SourceScoreStats:
            primary_wc_all = []
            primary_pwc_all = []
            non_primary_wc_all = []
            non_primary_pwc_all = []

            for t in target_summaries:
                scores = get_scores(t)
                primary_wc_all.extend(scores.primary_imp_wc.values)
                primary_pwc_all.extend(scores.primary_imp_pwc.values)
                non_primary_wc_all.extend(scores.non_primary_imp_wc.values)
                non_primary_pwc_all.extend(scores.non_primary_imp_pwc.values)

            return SourceScoreStats(
                primary_imp_wc=calc_stats(primary_wc_all),
                primary_imp_pwc=calc_stats(primary_pwc_all),
                non_primary_imp_wc=calc_stats(non_primary_wc_all),
                non_primary_imp_pwc=calc_stats(non_primary_pwc_all),
            )

        return QuestionTypeSummary(
            question_type=question_type,
            provider=provider,
            num_targets=len(target_summaries),
            num_runs_per_target=target_summaries[0].num_runs,
            citation_rate_avg=statistics.mean(citation_rates),
            citation_rate_median=statistics.median(citation_rates),
            imp_wc_avg=statistics.mean(imp_wc_means),
            imp_wc_median=statistics.median(imp_wc_means),
            imp_pwc_avg=statistics.mean(imp_pwc_means),
            imp_pwc_median=statistics.median(imp_pwc_means),
            primary_rate_without_avg=statistics.mean(primary_without_means),
            primary_rate_without_median=statistics.median(primary_without_means),
            primary_rate_with_avg=statistics.mean(primary_with_means),
            primary_rate_with_median=statistics.median(primary_with_means),
            primary_rate_diff_avg=statistics.mean(primary_diff_means),
            primary_rate_diff_median=statistics.median(primary_diff_means),
            source_scores_without=aggregate_all_source_scores(lambda t: t.source_scores_without),
            source_scores_with=aggregate_all_source_scores(lambda t: t.source_scores_with),
            target_summaries=target_summaries,
        )

    def _aggregate_domain(
        self,
        domain: str,
        provider: str,
        all_target_summaries: list[TargetSummary],
    ) -> DomainSummary:
        """ドメインの結果を集計（質問タイプ別）"""
        # 質問タイプ別にグループ化
        summaries_by_type: dict[str, list[TargetSummary]] = {}
        for t in all_target_summaries:
            if t.question_type not in summaries_by_type:
                summaries_by_type[t.question_type] = []
            summaries_by_type[t.question_type].append(t)

        # 質問タイプ別の集計
        question_type_summaries: dict[str, QuestionTypeSummary] = {}
        for q_type, summaries in summaries_by_type.items():
            question_type_summaries[q_type] = self._aggregate_question_type(
                q_type, provider, summaries
            )

        # 全体集計
        all_summary = self._aggregate_question_type(
            "all", provider, all_target_summaries
        )

        return DomainSummary(
            domain=domain,
            provider=provider,
            num_runs_per_target=all_target_summaries[0].num_runs if all_target_summaries else 0,
            question_type_summaries=question_type_summaries,
            all_summary=all_summary,
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

    def _source_score_stats_to_dict(self, stats: SourceScoreStats) -> dict:
        """SourceScoreStats を辞書に変換"""
        return {
            "primary": {
                "imp_wc": self._stats_to_dict(stats.primary_imp_wc),
                "imp_pwc": self._stats_to_dict(stats.primary_imp_pwc),
            },
            "non_primary": {
                "imp_wc": self._stats_to_dict(stats.non_primary_imp_wc),
                "imp_pwc": self._stats_to_dict(stats.non_primary_imp_pwc),
            },
        }

    def _save_target_summary(self, target_dir: Path, summary: TargetSummary):
        """ターゲットサマリーを保存"""
        self._save_json(target_dir / "summary.json", {
            "target_id": summary.target_id,
            "title": summary.title,
            "domain": summary.domain,
            "provider": summary.provider,
            "question_type": summary.question_type,
            "question": summary.question,
            "num_runs": summary.num_runs,
            "citation_rate": round(summary.citation_rate, 3),
            "imp_wc": self._stats_to_dict(summary.imp_wc),
            "imp_pwc": self._stats_to_dict(summary.imp_pwc),
            "primary_source_rate": {
                "without": self._stats_to_dict(summary.primary_rate_without),
                "with": self._stats_to_dict(summary.primary_rate_with),
                "diff": self._stats_to_dict(summary.primary_rate_diff),
            },
            "source_scores": {
                "without": self._source_score_stats_to_dict(summary.source_scores_without),
                "with": self._source_score_stats_to_dict(summary.source_scores_with),
            },
        })

    def _question_type_summary_to_dict(self, summary: QuestionTypeSummary) -> dict:
        """QuestionTypeSummary を辞書に変換"""
        return {
            "question_type": summary.question_type,
            "num_targets": summary.num_targets,
            "num_runs_per_target": summary.num_runs_per_target,
            "citation_rate": {
                "avg": round(summary.citation_rate_avg, 3),
                "median": round(summary.citation_rate_median, 3),
            },
            "imp_wc": {
                "avg": round(summary.imp_wc_avg, 2),
                "median": round(summary.imp_wc_median, 2),
            },
            "imp_pwc": {
                "avg": round(summary.imp_pwc_avg, 2),
                "median": round(summary.imp_pwc_median, 2),
            },
            "primary_source_rate": {
                "without": {
                    "avg": round(summary.primary_rate_without_avg, 2),
                    "median": round(summary.primary_rate_without_median, 2),
                },
                "with": {
                    "avg": round(summary.primary_rate_with_avg, 2),
                    "median": round(summary.primary_rate_with_median, 2),
                },
                "diff": {
                    "avg": round(summary.primary_rate_diff_avg, 2),
                    "median": round(summary.primary_rate_diff_median, 2),
                },
            },
            "source_scores": {
                "without": self._source_score_stats_to_dict(summary.source_scores_without),
                "with": self._source_score_stats_to_dict(summary.source_scores_with),
            },
            "targets": [
                {
                    "target_id": t.target_id,
                    "title": t.title,
                    "question": t.question,
                    "citation_rate": round(t.citation_rate, 3),
                    "imp_wc": self._stats_to_dict(t.imp_wc),
                    "imp_pwc": self._stats_to_dict(t.imp_pwc),
                    "primary_rate_without": self._stats_to_dict(t.primary_rate_without),
                    "primary_rate_with": self._stats_to_dict(t.primary_rate_with),
                    "primary_rate_diff": self._stats_to_dict(t.primary_rate_diff),
                }
                for t in summary.target_summaries
            ],
        }

    def _save_domain_summary(self, domain_dir: Path, provider: str, summary: DomainSummary):
        """ドメインサマリーを保存"""
        # 質問タイプ別のサマリー
        question_types_data = {
            q_type: self._question_type_summary_to_dict(q_summary)
            for q_type, q_summary in summary.question_type_summaries.items()
        }

        self._save_json(domain_dir / f"{provider}_summary.json", {
            "domain": summary.domain,
            "provider": summary.provider,
            "num_runs_per_target": summary.num_runs_per_target,
            "question_types": question_types_data,
            "all": self._question_type_summary_to_dict(summary.all_summary),
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
            "question_types": QUESTION_TYPES,
            "primary_domains": self.primary_domains,
            "domains": {},
        }

        for domain, provider_summaries in all_summaries.items():
            root_data["domains"][domain] = {}
            for provider, summary in provider_summaries.items():
                # 質問タイプ別のサマリー
                question_types_summary = {}
                for q_type, q_summary in summary.question_type_summaries.items():
                    question_types_summary[q_type] = {
                        "citation_rate": {
                            "avg": round(q_summary.citation_rate_avg, 3),
                            "median": round(q_summary.citation_rate_median, 3),
                        },
                        "imp_wc": {
                            "avg": round(q_summary.imp_wc_avg, 2),
                            "median": round(q_summary.imp_wc_median, 2),
                        },
                        "imp_pwc": {
                            "avg": round(q_summary.imp_pwc_avg, 2),
                            "median": round(q_summary.imp_pwc_median, 2),
                        },
                        "primary_source_rate": {
                            "without": {
                                "avg": round(q_summary.primary_rate_without_avg, 2),
                                "median": round(q_summary.primary_rate_without_median, 2),
                            },
                            "with": {
                                "avg": round(q_summary.primary_rate_with_avg, 2),
                                "median": round(q_summary.primary_rate_with_median, 2),
                            },
                            "diff": {
                                "avg": round(q_summary.primary_rate_diff_avg, 2),
                                "median": round(q_summary.primary_rate_diff_median, 2),
                            },
                        },
                    }

                # 全体集計
                all_s = summary.all_summary
                root_data["domains"][domain][provider] = {
                    "question_types": question_types_summary,
                    "all": {
                        "citation_rate": {
                            "avg": round(all_s.citation_rate_avg, 3),
                            "median": round(all_s.citation_rate_median, 3),
                        },
                        "imp_wc": {
                            "avg": round(all_s.imp_wc_avg, 2),
                            "median": round(all_s.imp_wc_median, 2),
                        },
                        "imp_pwc": {
                            "avg": round(all_s.imp_pwc_avg, 2),
                            "median": round(all_s.imp_pwc_median, 2),
                        },
                        "primary_source_rate": {
                            "without": {
                                "avg": round(all_s.primary_rate_without_avg, 2),
                                "median": round(all_s.primary_rate_without_median, 2),
                            },
                            "with": {
                                "avg": round(all_s.primary_rate_with_avg, 2),
                                "median": round(all_s.primary_rate_with_median, 2),
                            },
                            "diff": {
                                "avg": round(all_s.primary_rate_diff_avg, 2),
                                "median": round(all_s.primary_rate_diff_median, 2),
                            },
                        },
                    },
                }

        # 全体の集計（全ドメイン×全プロバイダー）
        all_summaries_flat: list[QuestionTypeSummary] = []
        for provider_summaries in all_summaries.values():
            for domain_summary in provider_summaries.values():
                all_summaries_flat.append(domain_summary.all_summary)

        if all_summaries_flat:
            citation_rates = [s.citation_rate_avg for s in all_summaries_flat]
            imp_wc_avgs = [s.imp_wc_avg for s in all_summaries_flat]
            imp_pwc_avgs = [s.imp_pwc_avg for s in all_summaries_flat]
            primary_without_avgs = [s.primary_rate_without_avg for s in all_summaries_flat]
            primary_with_avgs = [s.primary_rate_with_avg for s in all_summaries_flat]
            primary_diff_avgs = [s.primary_rate_diff_avg for s in all_summaries_flat]

            root_data["overall"] = {
                "citation_rate": {
                    "avg": round(statistics.mean(citation_rates), 3),
                    "median": round(statistics.median(citation_rates), 3),
                },
                "imp_wc": {
                    "avg": round(statistics.mean(imp_wc_avgs), 2),
                    "median": round(statistics.median(imp_wc_avgs), 2),
                },
                "imp_pwc": {
                    "avg": round(statistics.mean(imp_pwc_avgs), 2),
                    "median": round(statistics.median(imp_pwc_avgs), 2),
                },
                "primary_source_rate": {
                    "without": {
                        "avg": round(statistics.mean(primary_without_avgs), 2),
                        "median": round(statistics.median(primary_without_avgs), 2),
                    },
                    "with": {
                        "avg": round(statistics.mean(primary_with_avgs), 2),
                        "median": round(statistics.median(primary_with_avgs), 2),
                    },
                    "diff": {
                        "avg": round(statistics.mean(primary_diff_avgs), 2),
                        "median": round(statistics.median(primary_diff_avgs), 2),
                    },
                },
            }

        self._save_json(output_dir / "summary.json", root_data)


# =============================================================================
# Main
# =============================================================================

# ターゲットを検出するドメインリスト
TARGET_DOMAINS = ["econ"]


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="GEO-bench 実験スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q", "--generate-questions-only",
        action="store_true",
        help="質問生成のみを実行（実験は実行しない）",
    )
    parser.add_argument(
        "--show-questions",
        action="store_true",
        help="生成された質問を表示",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    config = EXPERIMENT_CONFIG

    # ターゲットを検出
    targets = discover_targets(config["targets_dir"], TARGET_DOMAINS)

    if not targets:
        print("エラー: ターゲットが見つかりません")
        sys.exit(1)

    print(f"検出されたターゲット: {len(targets)}件")
    for domain in TARGET_DOMAINS:
        count = sum(1 for t in targets if t["domain"] == domain)
        print(f"  {domain}: {count}件")

    # 質問生成用のLLMを取得（最初のプロバイダーを使用）
    question_llm = create_llm_client(config["providers"][0])
    print(f"\n質問生成用LLM: {question_llm.name}")

    # 質問を生成（キャッシュ利用）
    questions = await generate_all_questions(
        targets=targets,
        llm=question_llm,
        cache_file=config["questions_cache_file"],
    )

    # 質問表示オプション
    if args.show_questions:
        print("\n" + "="*60)
        print("生成された質問")
        print("="*60)
        for target in targets:
            q = questions.get(target["id"])
            if q:
                print(f"\n[{target['title']}]")
                print(f"  vague: {q['vague']}")
                print(f"  experiment: {q['experiment']}")
                print(f"  aligned: {q['aligned']}")

    # 質問生成のみの場合はここで終了
    if args.generate_questions_only:
        print(f"\n質問生成完了: {config['questions_cache_file']}")
        return

    # 実験を実行
    runner = ExperimentRunner(
        providers=config["providers"],
        num_runs=config["num_runs"],
        max_sources=config["max_sources"],
        output_base_dir=config["output_dir"],
        primary_domains=config.get("primary_domains", []),
    )

    await runner.run_experiment(targets, questions)


if __name__ == "__main__":
    asyncio.run(main())
