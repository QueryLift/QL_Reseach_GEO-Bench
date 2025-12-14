#!/usr/bin/env python3
"""
GEO-bench 実験スクリプト

複数ターゲット・複数プロバイダーでの実験を実行し、
結果を階層的に集計・出力する。

特徴:
- Web検索は各ターゲット×プロバイダーにつき1回のみ
- 回答生成は各ターゲット×プロバイダーにつきN回繰り返し
- 結果はターゲット→ドメイン→ルートの階層で集計
- TARGETは常に一次情報源としてカウント
- 一次情報源率はimp_wcベースとcitation_frequencyベースの両方を計算
"""

import argparse
import asyncio
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
    WebContentFetcher,
    strip_markdown,
)
from question_generator import (
    QUESTION_TYPES,
    GeneratedQuestions,
    generate_all_questions,
)
from metrics import (
    PrimarySourceRate,
    SourceScores,
    SourceScoreStats,
    Stats,
    aggregate_source_scores,
    calc_primary_source_rate,
    calc_source_scores,
    calc_stats,
    stats_to_dict,
)
from file_output import (
    question_type_summary_to_dict,
    save_domain_summary,
    save_experiment_config,
    save_json,
    save_root_summary,
    save_run_result,
    save_sources,
    save_target_summary,
)


# =============================================================================
# Configuration
# =============================================================================

# 実験設定
EXPERIMENT_CONFIG = {
    "providers": ["gpt"],                # 使用するプロバイダー
    "num_runs": 2,                        # 各ターゲットの繰り返し回数
    "max_sources": 5,                     # Web検索で取得するソース数
    "targets_dir": "targets",             # ターゲットファイルのディレクトリ
    "output_dir": "outputs",              # 出力ディレクトリ
    "questions_cache_file": "questions_cache.json",  # 質問キャッシュファイル
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
    """
    ターゲット設定

    Attributes:
        id: ターゲットID（ファイル名から拡張子を除いたもの）
        file: ファイルパス
        url: ターゲットURL（ダミー）
        domain: ドメイン名
        title: タイトル（H1タグの内容）
        content: ファイル内容（質問生成用）
    """
    id: str
    file: str
    url: str
    domain: str
    title: str
    content: str


@dataclass
class RunResult:
    """
    1回の実行結果

    Attributes:
        run_index: 実行インデックス（1-indexed）
        question_type: 質問タイプ（vague, experiment, aligned）
        answer_without: ターゲットなしの回答
        answer_with: ターゲットありの回答
        metrics_without: ターゲットなしのメトリクス
        metrics_with: ターゲットありのメトリクス
        target_index: ターゲットのインデックス（1-indexed）
        target_cited: ターゲットが引用されたかどうか
        target_imp_wc: ターゲットのimp_wc
        target_imp_pwc: ターゲットのimp_pwc
        primary_rate_without: ターゲットなしの一次情報源率
        primary_rate_with: ターゲットありの一次情報源率
        source_scores_without: ターゲットなしのソース別スコア
        source_scores_with: ターゲットありのソース別スコア
        has_non_primary_without: withoutに非一次情報源が含まれているか
            （一次情報源率の統計計算に含めるかどうかの判定用）
    """
    run_index: int
    question_type: str
    answer_without: str
    answer_with: str
    metrics_without: dict[int, CitationMetrics]
    metrics_with: dict[int, CitationMetrics]
    target_index: int
    target_cited: bool
    target_imp_wc: float
    target_imp_pwc: float
    # 一次情報源率（by_imp_wc と by_frequency）
    primary_rate_without: PrimarySourceRate
    primary_rate_with: PrimarySourceRate
    # ソース別スコア
    source_scores_without: SourceScores
    source_scores_with: SourceScores
    # withoutに非一次情報源が含まれているか（一次情報源率が100%未満）
    has_non_primary_without: bool = False


@dataclass
class TargetSummary:
    """
    ターゲットの集計結果

    Attributes:
        target_id: ターゲットID
        title: タイトル（H1タグの内容）
        domain: ドメイン名
        provider: LLMプロバイダー名
        question_type: 質問タイプ
        question: 使用した質問文
        num_runs: 実行回数
        num_runs_with_non_primary: 非一次情報源があった実行回数
            （一次情報源率の統計計算対象となるrun数）
        citation_rate: ターゲットが引用された割合
        imp_wc: imp_wcの統計
        imp_pwc: imp_pwcの統計
        primary_rate_without_by_wc: imp_wcベースの一次情報源率（without）
        primary_rate_with_by_wc: imp_wcベースの一次情報源率（with）
        primary_rate_without_by_freq: frequencyベースの一次情報源率（without）
        primary_rate_with_by_freq: frequencyベースの一次情報源率（with）
        source_scores_without: ソース別スコア統計（without）
        source_scores_with: ソース別スコア統計（with）
    """
    target_id: str
    title: str
    domain: str
    provider: str
    question_type: str
    question: str
    num_runs: int
    num_runs_with_non_primary: int  # 非一次情報源があった実行回数
    citation_rate: float
    imp_wc: Stats
    imp_pwc: Stats
    # imp_wcベースの一次情報源率（withoutに非一次情報源があったrunのみ）
    primary_rate_without_by_wc: Stats
    primary_rate_with_by_wc: Stats
    primary_rate_diff_by_wc: Stats
    # frequencyベースの一次情報源率（withoutに非一次情報源があったrunのみ）
    primary_rate_without_by_freq: Stats
    primary_rate_with_by_freq: Stats
    primary_rate_diff_by_freq: Stats
    # ソース別スコア統計
    source_scores_without: SourceScoreStats
    source_scores_with: SourceScoreStats


@dataclass
class QuestionTypeSummary:
    """
    質問タイプ別の集計結果

    Attributes:
        question_type: 質問タイプ
        provider: LLMプロバイダー名
        num_targets: ターゲット数
        num_runs_per_target: ターゲットあたりの実行回数
        citation_rate_avg/median: 引用率の統計
        imp_wc_avg/median: imp_wcの統計
        imp_pwc_avg/median: imp_pwcの統計
        primary_rate_*: 各種一次情報源率の統計
        source_scores_*: ソース別スコア統計
        target_summaries: ターゲットごとの詳細
    """
    question_type: str
    provider: str
    num_targets: int
    num_runs_per_target: int
    citation_rate_avg: float
    citation_rate_median: float
    imp_wc_avg: float
    imp_wc_median: float
    imp_pwc_avg: float
    imp_pwc_median: float
    # imp_wcベースの一次情報源率
    primary_rate_without_avg_by_wc: float
    primary_rate_without_median_by_wc: float
    primary_rate_with_avg_by_wc: float
    primary_rate_with_median_by_wc: float
    primary_rate_diff_avg_by_wc: float
    primary_rate_diff_median_by_wc: float
    # frequencyベースの一次情報源率
    primary_rate_without_avg_by_freq: float
    primary_rate_without_median_by_freq: float
    primary_rate_with_avg_by_freq: float
    primary_rate_with_median_by_freq: float
    primary_rate_diff_avg_by_freq: float
    primary_rate_diff_median_by_freq: float
    # ソース別スコア
    source_scores_without: SourceScoreStats
    source_scores_with: SourceScoreStats
    target_summaries: list[TargetSummary]


@dataclass
class DomainSummary:
    """
    ドメインの集計結果

    Attributes:
        domain: ドメイン名
        provider: LLMプロバイダー名
        num_runs_per_target: ターゲットあたりの実行回数
        question_type_summaries: 質問タイプ別サマリー
        all_summary: 全質問タイプの集計サマリー
    """
    domain: str
    provider: str
    num_runs_per_target: int
    question_type_summaries: dict[str, QuestionTypeSummary]
    all_summary: QuestionTypeSummary


# =============================================================================
# Utility Functions
# =============================================================================

def load_target_file(file_path: str) -> str:
    """
    ターゲットファイルを読み込み、Markdownを除去

    Args:
        file_path: ファイルパス

    Returns:
        str: Markdown記法を除去したプレーンテキスト
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return strip_markdown(content)


def extract_h1_title(file_path: str) -> str:
    """
    MDファイルから最初のH1タグの内容を抽出

    Args:
        file_path: ファイルパス

    Returns:
        str: H1タグの内容、見つからない場合はファイル名
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = re.match(r'^#\s+(.+)$', line)
            if match:
                return match.group(1).strip()
    return Path(file_path).stem


def discover_targets(targets_dir: str, domains: list[str]) -> list[TargetConfig]:
    """
    指定ドメインのターゲットファイルを検出

    Args:
        targets_dir: ターゲットディレクトリのパス
        domains: 検索対象のドメインリスト

    Returns:
        list[TargetConfig]: 検出されたターゲットの設定リスト
    """
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
    """
    フォルダ名に使用できない文字を置換

    Args:
        name: 元の名前

    Returns:
        str: サニタイズされた名前
    """
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '_')
    if len(name) > 50:
        name = name[:50]
    return name


def format_sources(sources: list[SourceContent]) -> str:
    """
    ソースをプロンプト用にフォーマット

    Args:
        sources: ソースリスト

    Returns:
        str: プロンプトに挿入するフォーマット済み文字列
    """
    lines = []
    for i, source in enumerate(sources, 1):
        url = source["url"]
        content = source["content"]
        lines.append(f"[{i}] URL: {url}")
        lines.append(f"    Content: {content}")
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """
    実験実行クラス

    複数ターゲット・複数プロバイダーでの実験を実行し、
    結果を階層的に集計・出力する。

    Attributes:
        providers: 使用するLLMプロバイダーのリスト
        num_runs: 各ターゲットの繰り返し回数
        max_sources: Web検索で取得するソース数
        output_base_dir: 出力ディレクトリのベースパス
        primary_domains: 一次情報源と判定するドメインリスト
        fetcher: Webコンテンツ取得クラス
        analyzer: 引用分析クラス
        llm_clients: LLMクライアントのキャッシュ
    """

    def __init__(
        self,
        providers: list[str],
        num_runs: int,
        max_sources: int,
        output_base_dir: str,
        primary_domains: list[str],
    ):
        """
        ExperimentRunnerを初期化

        Args:
            providers: 使用するLLMプロバイダーのリスト
            num_runs: 各ターゲットの繰り返し回数
            max_sources: Web検索で取得するソース数
            output_base_dir: 出力ディレクトリのベースパス
            primary_domains: 一次情報源と判定するドメインリスト
        """
        self.providers = providers
        self.num_runs = num_runs
        self.max_sources = max_sources
        self.output_base_dir = Path(output_base_dir)
        self.primary_domains = primary_domains
        self.fetcher = WebContentFetcher()
        self.analyzer = CitationAnalyzer()
        self.llm_clients: dict[str, LLMClient] = {}

    def _get_llm(self, provider: str) -> LLMClient:
        """
        LLMクライアントを取得（キャッシュ使用）

        Args:
            provider: プロバイダー名

        Returns:
            LLMClient: LLMクライアントインスタンス
        """
        if provider not in self.llm_clients:
            self.llm_clients[provider] = create_llm_client(provider)
        return self.llm_clients[provider]

    async def run_experiment(
        self,
        targets: list[TargetConfig],
        questions: dict[str, GeneratedQuestions],
        output_name: str | None = None,
    ) -> Path:
        """
        実験を実行

        Args:
            targets: ターゲットリスト
            questions: ターゲットIDごとの質問マッピング
            output_name: 出力フォルダ名（Noneの場合はタイムスタンプ）

        Returns:
            Path: 出力ディレクトリのパス
        """
        # 出力ディレクトリを作成
        folder_name = output_name if output_name else datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_base_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 実験設定を保存
        save_experiment_config(
            output_dir=output_dir,
            providers=self.providers,
            num_runs=self.num_runs,
            max_sources=self.max_sources,
            primary_domains=self.primary_domains,
            question_types=QUESTION_TYPES,
            targets=targets,
            questions={tid: dict(q) for tid, q in questions.items()},
        )

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

                # 全ターゲットを並列で実行
                target_tasks = []
                for target in domain_targets:
                    target_questions = questions.get(target["id"])
                    if not target_questions:
                        print(f"警告: ターゲット {target['id']} の質問が見つかりません")
                        continue

                    target_tasks.append(
                        self._run_target_all_questions(
                            llm=llm,
                            provider=provider,
                            target=target,
                            target_questions=target_questions,
                            output_dir=domain_dir,
                        )
                    )

                nested_summaries = await asyncio.gather(*target_tasks)
                all_target_summaries = [s for summaries in nested_summaries for s in summaries]

                # ドメイン集計
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

    async def _run_target_all_questions(
        self,
        llm: LLMClient,
        provider: str,
        target: TargetConfig,
        target_questions: GeneratedQuestions,
        output_dir: Path,
    ) -> list[TargetSummary]:
        """
        1ターゲットの全質問タイプを同じソースで実験

        ポイント:
        - Web検索は1回のみ（experiment質問で検索）
        - 同じソースを全質問タイプで使い回す
        - 質問の違いによる効果のみを測定
        - TARGETは常に一次情報源としてカウント

        Args:
            llm: LLMクライアント
            provider: プロバイダー名
            target: ターゲット設定
            target_questions: 3種類の質問
            output_dir: 出力ディレクトリ

        Returns:
            list[TargetSummary]: 各質問タイプのサマリーリスト
        """
        target_title = target['title']

        # 1. Web検索（experiment質問で1回のみ）
        search_query = target_questions["experiment"]
        print(f"  [{provider}][{target_title}] Web検索開始（全質問タイプ共通）")
        source_urls = await self._search_web(llm, search_query)

        # 2. ソースを取得（1回のみ）
        sources = await self._fetch_sources(source_urls)

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

        # 4. 各質問タイプごとに回答生成（同じソースを使用）
        all_summaries: list[TargetSummary] = []

        for question_type in QUESTION_TYPES:
            question = target_questions[question_type]
            folder_name = f"{provider}_{question_type}_{sanitize_folder_name(target_title)}"
            target_dir = output_dir / folder_name
            target_dir.mkdir(exist_ok=True)

            # ソースを保存
            save_sources(target_dir, sources)

            # 回答生成をN回並列で実行
            print(f"  [{provider}][{question_type}][{target_title}] 回答生成 {self.num_runs}回を並列実行中...")

            # 全run分のwithout/with回答を一括で並列生成
            answer_tasks = []
            for _ in range(self.num_runs):
                answer_tasks.append(self._generate_answer(llm, question, sources))
                answer_tasks.append(self._generate_answer(llm, question, sources_with_targets))

            all_answers = await asyncio.gather(*answer_tasks)

            # 結果を処理
            run_results: list[RunResult] = []
            for run_idx in range(1, self.num_runs + 1):
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

                # 一次情報源の引用率を計算（TARGETを一次情報源としてカウント）
                # without: TARGETなし
                primary_rate_without = calc_primary_source_rate(
                    metrics_without, sources, self.primary_domains
                )
                # with: TARGETを一次情報源としてカウント
                primary_rate_with = calc_primary_source_rate(
                    metrics_with, sources, self.primary_domains,
                    target_indices={target_index}
                )

                # ソース別スコアを計算（TARGETを一次情報源としてカウント）
                source_scores_without = calc_source_scores(
                    metrics_without, sources, self.primary_domains
                )
                source_scores_with = calc_source_scores(
                    metrics_with, sources, self.primary_domains,
                    target_indices={target_index}
                )

                # withoutに非一次情報源があるかどうか（一次情報源率が100%未満）
                has_non_primary_without = (
                    primary_rate_without.by_imp_wc < 100.0 or
                    primary_rate_without.by_frequency < 100.0
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
                    primary_rate_without=primary_rate_without,
                    primary_rate_with=primary_rate_with,
                    source_scores_without=source_scores_without,
                    source_scores_with=source_scores_with,
                    has_non_primary_without=has_non_primary_without,
                )
                run_results.append(run_result)

                # 個別結果を保存
                save_run_result(
                    output_dir=target_dir,
                    run_index=run_idx,
                    answer_without=answer_without,
                    answer_with=answer_with,
                    metrics_without=metrics_without,
                    metrics_with=metrics_with,
                    sources=sources,
                    sources_with_target=sources_with_targets,
                    target_index=target_index,
                )

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
            save_target_summary(
                output_dir=target_dir,
                target_id=target_summary.target_id,
                title=target_summary.title,
                domain=target_summary.domain,
                provider=target_summary.provider,
                question_type=target_summary.question_type,
                question=target_summary.question,
                num_runs=target_summary.num_runs,
                citation_rate=target_summary.citation_rate,
                imp_wc=target_summary.imp_wc,
                imp_pwc=target_summary.imp_pwc,
                primary_rate_without=target_summary.primary_rate_without_by_wc,
                primary_rate_with=target_summary.primary_rate_with_by_wc,
                primary_rate_diff=target_summary.primary_rate_diff_by_wc,
                source_scores_without=target_summary.source_scores_without,
                source_scores_with=target_summary.source_scores_with,
                num_runs_with_non_primary=target_summary.num_runs_with_non_primary,
            )

            print(f"  [{provider}][{question_type}][{target_title}] 完了 (citation_rate: {target_summary.citation_rate:.1%})")
            all_summaries.append(target_summary)

        return all_summaries

    async def _search_web(self, llm: LLMClient, query: str) -> list[str]:
        """
        Web検索を実行

        Args:
            llm: LLMクライアント
            query: 検索クエリ

        Returns:
            list[str]: 検索結果のURLリスト
        """
        results = await llm.search_web(query, max_results=self.max_sources)
        return [r["url"] for r in results if "url" in r]

    async def _fetch_sources(self, urls: list[str]) -> list[SourceContent]:
        """
        複数URLからコンテンツを並列取得

        リダイレクトがある場合は最終URLをソースに保存する。
        （Geminiの検索結果はGoogleリダイレクトURLが含まれるため）

        Args:
            urls: URLリスト

        Returns:
            list[SourceContent]: ソースコンテンツのリスト
        """
        tasks = [self.fetcher.fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)

        sources: list[SourceContent] = []
        for content, media_type, final_url in results:
            sources.append({
                "url": final_url,  # リダイレクト後の最終URLを使用
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
        """
        回答を生成

        Args:
            llm: LLMクライアント
            question: 質問文
            sources: ソースリスト

        Returns:
            str: 生成された回答
        """
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
        """
        ターゲットの結果を集計

        一次情報源率の統計は、withoutに非一次情報源が含まれているrunのみを対象とする。
        これは、withoutが100%一次情報源の場合、with/withoutの比較が意味を持たないため。

        Args:
            target_id: ターゲットID
            title: タイトル
            domain: ドメイン
            provider: プロバイダー
            question_type: 質問タイプ
            question: 質問文
            run_results: 実行結果リスト

        Returns:
            TargetSummary: 集計結果
        """
        cited_count = sum(1 for r in run_results if r.target_cited)

        # ソース別スコアを集計
        source_scores_without = aggregate_source_scores(
            [r.source_scores_without for r in run_results]
        )
        source_scores_with = aggregate_source_scores(
            [r.source_scores_with for r in run_results]
        )

        # 一次情報源率の統計は、withoutに非一次情報源があったrunのみを対象とする
        valid_runs = [r for r in run_results if r.has_non_primary_without]
        num_runs_with_non_primary = len(valid_runs)

        # 一次情報源率の統計（imp_wcベース）- フィルタリング済みrunから計算
        without_by_wc = [r.primary_rate_without.by_imp_wc for r in valid_runs]
        with_by_wc = [r.primary_rate_with.by_imp_wc for r in valid_runs]
        diff_by_wc = [r.primary_rate_with.by_imp_wc - r.primary_rate_without.by_imp_wc for r in valid_runs]

        # 一次情報源率の統計（frequencyベース）- フィルタリング済みrunから計算
        without_by_freq = [r.primary_rate_without.by_frequency for r in valid_runs]
        with_by_freq = [r.primary_rate_with.by_frequency for r in valid_runs]
        diff_by_freq = [r.primary_rate_with.by_frequency - r.primary_rate_without.by_frequency for r in valid_runs]

        return TargetSummary(
            target_id=target_id,
            title=title,
            domain=domain,
            provider=provider,
            question_type=question_type,
            question=question,
            num_runs=len(run_results),
            num_runs_with_non_primary=num_runs_with_non_primary,
            citation_rate=cited_count / len(run_results) if run_results else 0.0,
            imp_wc=calc_stats([r.target_imp_wc for r in run_results]),
            imp_pwc=calc_stats([r.target_imp_pwc for r in run_results]),
            primary_rate_without_by_wc=calc_stats(without_by_wc),
            primary_rate_with_by_wc=calc_stats(with_by_wc),
            primary_rate_diff_by_wc=calc_stats(diff_by_wc),
            primary_rate_without_by_freq=calc_stats(without_by_freq),
            primary_rate_with_by_freq=calc_stats(with_by_freq),
            primary_rate_diff_by_freq=calc_stats(diff_by_freq),
            source_scores_without=source_scores_without,
            source_scores_with=source_scores_with,
        )

    def _aggregate_question_type(
        self,
        question_type: str,
        provider: str,
        target_summaries: list[TargetSummary],
    ) -> QuestionTypeSummary:
        """
        質問タイプの結果を集計

        一次情報源率の統計は、非一次情報源があったターゲットのみを対象とする。
        これにより、withoutが100%一次情報源のケースを除外し、
        with/withoutの比較が意味のあるターゲットのみで統計を取る。

        Args:
            question_type: 質問タイプ
            provider: プロバイダー
            target_summaries: ターゲットサマリーリスト

        Returns:
            QuestionTypeSummary: 集計結果
        """
        if not target_summaries:
            empty_stats = calc_stats([])
            empty_source_stats = aggregate_source_scores([])
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
                primary_rate_without_avg_by_wc=0.0,
                primary_rate_without_median_by_wc=0.0,
                primary_rate_with_avg_by_wc=0.0,
                primary_rate_with_median_by_wc=0.0,
                primary_rate_diff_avg_by_wc=0.0,
                primary_rate_diff_median_by_wc=0.0,
                primary_rate_without_avg_by_freq=0.0,
                primary_rate_without_median_by_freq=0.0,
                primary_rate_with_avg_by_freq=0.0,
                primary_rate_with_median_by_freq=0.0,
                primary_rate_diff_avg_by_freq=0.0,
                primary_rate_diff_median_by_freq=0.0,
                source_scores_without=empty_source_stats,
                source_scores_with=empty_source_stats,
                target_summaries=[],
            )

        # 全ターゲットから基本統計を計算
        citation_rates = [t.citation_rate for t in target_summaries]
        imp_wc_means = [t.imp_wc.mean for t in target_summaries]
        imp_pwc_means = [t.imp_pwc.mean for t in target_summaries]

        # 一次情報源率の統計は、非一次情報源があったターゲットのみを対象
        valid_targets = [t for t in target_summaries if t.num_runs_with_non_primary > 0]

        if valid_targets:
            # imp_wcベースの一次情報源率（有効なターゲットのみ）
            primary_without_means_by_wc = [t.primary_rate_without_by_wc.mean for t in valid_targets]
            primary_with_means_by_wc = [t.primary_rate_with_by_wc.mean for t in valid_targets]
            primary_diff_means_by_wc = [t.primary_rate_diff_by_wc.mean for t in valid_targets]

            # frequencyベースの一次情報源率（有効なターゲットのみ）
            primary_without_means_by_freq = [t.primary_rate_without_by_freq.mean for t in valid_targets]
            primary_with_means_by_freq = [t.primary_rate_with_by_freq.mean for t in valid_targets]
            primary_diff_means_by_freq = [t.primary_rate_diff_by_freq.mean for t in valid_targets]

            primary_without_avg_by_wc = statistics.mean(primary_without_means_by_wc)
            primary_without_median_by_wc = statistics.median(primary_without_means_by_wc)
            primary_with_avg_by_wc = statistics.mean(primary_with_means_by_wc)
            primary_with_median_by_wc = statistics.median(primary_with_means_by_wc)
            primary_diff_avg_by_wc = statistics.mean(primary_diff_means_by_wc)
            primary_diff_median_by_wc = statistics.median(primary_diff_means_by_wc)

            primary_without_avg_by_freq = statistics.mean(primary_without_means_by_freq)
            primary_without_median_by_freq = statistics.median(primary_without_means_by_freq)
            primary_with_avg_by_freq = statistics.mean(primary_with_means_by_freq)
            primary_with_median_by_freq = statistics.median(primary_with_means_by_freq)
            primary_diff_avg_by_freq = statistics.mean(primary_diff_means_by_freq)
            primary_diff_median_by_freq = statistics.median(primary_diff_means_by_freq)
        else:
            # 有効なターゲットがない場合は0.0
            primary_without_avg_by_wc = 0.0
            primary_without_median_by_wc = 0.0
            primary_with_avg_by_wc = 0.0
            primary_with_median_by_wc = 0.0
            primary_diff_avg_by_wc = 0.0
            primary_diff_median_by_wc = 0.0
            primary_without_avg_by_freq = 0.0
            primary_without_median_by_freq = 0.0
            primary_with_avg_by_freq = 0.0
            primary_with_median_by_freq = 0.0
            primary_diff_avg_by_freq = 0.0
            primary_diff_median_by_freq = 0.0

        # ソース別スコアを集計（全ターゲット）
        def aggregate_all_source_scores(get_scores) -> SourceScoreStats:
            all_scores = [get_scores(t) for t in target_summaries]
            # SourceScoreStatsの各Statsからvaluesを取り出して再集計
            from metrics import SourceScores as SS
            combined = []
            for stats in all_scores:
                combined.append(SS(
                    primary_imp_wc_values=stats.primary_imp_wc.values,
                    primary_imp_pwc_values=stats.primary_imp_pwc.values,
                    primary_freq_values=[int(v) for v in stats.primary_freq.values],
                    non_primary_imp_wc_values=stats.non_primary_imp_wc.values,
                    non_primary_imp_pwc_values=stats.non_primary_imp_pwc.values,
                    non_primary_freq_values=[int(v) for v in stats.non_primary_freq.values],
                ))
            return aggregate_source_scores(combined)

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
            primary_rate_without_avg_by_wc=primary_without_avg_by_wc,
            primary_rate_without_median_by_wc=primary_without_median_by_wc,
            primary_rate_with_avg_by_wc=primary_with_avg_by_wc,
            primary_rate_with_median_by_wc=primary_with_median_by_wc,
            primary_rate_diff_avg_by_wc=primary_diff_avg_by_wc,
            primary_rate_diff_median_by_wc=primary_diff_median_by_wc,
            primary_rate_without_avg_by_freq=primary_without_avg_by_freq,
            primary_rate_without_median_by_freq=primary_without_median_by_freq,
            primary_rate_with_avg_by_freq=primary_with_avg_by_freq,
            primary_rate_with_median_by_freq=primary_with_median_by_freq,
            primary_rate_diff_avg_by_freq=primary_diff_avg_by_freq,
            primary_rate_diff_median_by_freq=primary_diff_median_by_freq,
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
        """
        ドメインの結果を集計（質問タイプ別）

        Args:
            domain: ドメイン名
            provider: プロバイダー
            all_target_summaries: 全ターゲットのサマリーリスト

        Returns:
            DomainSummary: 集計結果
        """
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
    # File Output Helpers
    # =========================================================================

    def _question_type_summary_to_dict(self, summary: QuestionTypeSummary) -> dict:
        """
        QuestionTypeSummaryを辞書に変換

        Args:
            summary: 質問タイプサマリー

        Returns:
            dict: JSON出力用の辞書
        """
        from file_output import source_score_stats_to_dict

        target_details = [
            {
                "target_id": t.target_id,
                "title": t.title,
                "question": t.question,
                "citation_rate": round(t.citation_rate, 3),
                "imp_wc": stats_to_dict(t.imp_wc),
                "imp_pwc": stats_to_dict(t.imp_pwc),
                "primary_rate_by_wc": {
                    "without": stats_to_dict(t.primary_rate_without_by_wc),
                    "with": stats_to_dict(t.primary_rate_with_by_wc),
                    "diff": stats_to_dict(t.primary_rate_diff_by_wc),
                },
                "primary_rate_by_freq": {
                    "without": stats_to_dict(t.primary_rate_without_by_freq),
                    "with": stats_to_dict(t.primary_rate_with_by_freq),
                    "diff": stats_to_dict(t.primary_rate_diff_by_freq),
                },
            }
            for t in summary.target_summaries
        ]

        return question_type_summary_to_dict(
            question_type=summary.question_type,
            num_targets=summary.num_targets,
            num_runs_per_target=summary.num_runs_per_target,
            citation_rate_avg=summary.citation_rate_avg,
            citation_rate_median=summary.citation_rate_median,
            imp_wc_avg=summary.imp_wc_avg,
            imp_wc_median=summary.imp_wc_median,
            imp_pwc_avg=summary.imp_pwc_avg,
            imp_pwc_median=summary.imp_pwc_median,
            primary_rate_without_avg=summary.primary_rate_without_avg_by_wc,
            primary_rate_without_median=summary.primary_rate_without_median_by_wc,
            primary_rate_with_avg=summary.primary_rate_with_avg_by_wc,
            primary_rate_with_median=summary.primary_rate_with_median_by_wc,
            primary_rate_diff_avg=summary.primary_rate_diff_avg_by_wc,
            primary_rate_diff_median=summary.primary_rate_diff_median_by_wc,
            primary_rate_by_freq_without_avg=summary.primary_rate_without_avg_by_freq,
            primary_rate_by_freq_without_median=summary.primary_rate_without_median_by_freq,
            primary_rate_by_freq_with_avg=summary.primary_rate_with_avg_by_freq,
            primary_rate_by_freq_with_median=summary.primary_rate_with_median_by_freq,
            primary_rate_by_freq_diff_avg=summary.primary_rate_diff_avg_by_freq,
            primary_rate_by_freq_diff_median=summary.primary_rate_diff_median_by_freq,
            source_scores_without=summary.source_scores_without,
            source_scores_with=summary.source_scores_with,
            target_details=target_details,
        )

    def _save_domain_summary(self, domain_dir: Path, provider: str, summary: DomainSummary):
        """
        ドメインサマリーを保存

        Args:
            domain_dir: ドメインディレクトリ
            provider: プロバイダー
            summary: ドメインサマリー
        """
        question_types_data = {
            q_type: self._question_type_summary_to_dict(q_summary)
            for q_type, q_summary in summary.question_type_summaries.items()
        }

        save_domain_summary(
            output_dir=domain_dir,
            provider=provider,
            domain=summary.domain,
            num_runs_per_target=summary.num_runs_per_target,
            question_type_summaries=question_types_data,
            all_summary=self._question_type_summary_to_dict(summary.all_summary),
        )

    def _save_root_summary(
        self,
        output_dir: Path,
        all_summaries: dict[str, dict[str, DomainSummary]],
    ):
        """
        ルートサマリーを保存

        Args:
            output_dir: 出力ディレクトリ
            all_summaries: 全ドメイン×プロバイダーのサマリー
        """
        domain_data = {}

        for domain, provider_summaries in all_summaries.items():
            domain_data[domain] = {}
            for provider, summary in provider_summaries.items():
                # 質問タイプ別サマリーを簡略化
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
                            "by_imp_wc": {
                                "without": {
                                    "avg": round(q_summary.primary_rate_without_avg_by_wc, 2),
                                    "median": round(q_summary.primary_rate_without_median_by_wc, 2),
                                },
                                "with": {
                                    "avg": round(q_summary.primary_rate_with_avg_by_wc, 2),
                                    "median": round(q_summary.primary_rate_with_median_by_wc, 2),
                                },
                                "diff": {
                                    "avg": round(q_summary.primary_rate_diff_avg_by_wc, 2),
                                    "median": round(q_summary.primary_rate_diff_median_by_wc, 2),
                                },
                            },
                            "by_frequency": {
                                "without": {
                                    "avg": round(q_summary.primary_rate_without_avg_by_freq, 2),
                                    "median": round(q_summary.primary_rate_without_median_by_freq, 2),
                                },
                                "with": {
                                    "avg": round(q_summary.primary_rate_with_avg_by_freq, 2),
                                    "median": round(q_summary.primary_rate_with_median_by_freq, 2),
                                },
                                "diff": {
                                    "avg": round(q_summary.primary_rate_diff_avg_by_freq, 2),
                                    "median": round(q_summary.primary_rate_diff_median_by_freq, 2),
                                },
                            },
                        },
                    }

                # 全体集計
                all_s = summary.all_summary
                domain_data[domain][provider] = {
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
                            "by_imp_wc": {
                                "without": {
                                    "avg": round(all_s.primary_rate_without_avg_by_wc, 2),
                                    "median": round(all_s.primary_rate_without_median_by_wc, 2),
                                },
                                "with": {
                                    "avg": round(all_s.primary_rate_with_avg_by_wc, 2),
                                    "median": round(all_s.primary_rate_with_median_by_wc, 2),
                                },
                                "diff": {
                                    "avg": round(all_s.primary_rate_diff_avg_by_wc, 2),
                                    "median": round(all_s.primary_rate_diff_median_by_wc, 2),
                                },
                            },
                            "by_frequency": {
                                "without": {
                                    "avg": round(all_s.primary_rate_without_avg_by_freq, 2),
                                    "median": round(all_s.primary_rate_without_median_by_freq, 2),
                                },
                                "with": {
                                    "avg": round(all_s.primary_rate_with_avg_by_freq, 2),
                                    "median": round(all_s.primary_rate_with_median_by_freq, 2),
                                },
                                "diff": {
                                    "avg": round(all_s.primary_rate_diff_avg_by_freq, 2),
                                    "median": round(all_s.primary_rate_diff_median_by_freq, 2),
                                },
                            },
                        },
                    },
                }

        # 全体の集計（全ドメイン×全プロバイダー）
        all_summaries_flat: list[QuestionTypeSummary] = []
        for provider_summaries in all_summaries.values():
            for domain_summary in provider_summaries.values():
                all_summaries_flat.append(domain_summary.all_summary)

        overall_summary = None
        if all_summaries_flat:
            citation_rates = [s.citation_rate_avg for s in all_summaries_flat]
            imp_wc_avgs = [s.imp_wc_avg for s in all_summaries_flat]
            imp_pwc_avgs = [s.imp_pwc_avg for s in all_summaries_flat]

            # 一次情報源率は、有効なデータがあるサマリーのみを対象
            # （withoutに非一次情報源があったケースが1つ以上あるサマリー）
            def has_valid_primary_data(s: QuestionTypeSummary) -> bool:
                # 有効なターゲットがあればprimary_rate値が設定されている
                # 全て0.0の場合は有効なデータがないと判断
                return (
                    s.primary_rate_without_avg_by_wc != 0.0 or
                    s.primary_rate_with_avg_by_wc != 0.0 or
                    s.primary_rate_without_avg_by_freq != 0.0 or
                    s.primary_rate_with_avg_by_freq != 0.0
                )

            valid_summaries = [s for s in all_summaries_flat if has_valid_primary_data(s)]

            if valid_summaries:
                # imp_wcベース（有効なサマリーのみ）
                primary_without_avgs_by_wc = [s.primary_rate_without_avg_by_wc for s in valid_summaries]
                primary_with_avgs_by_wc = [s.primary_rate_with_avg_by_wc for s in valid_summaries]
                primary_diff_avgs_by_wc = [s.primary_rate_diff_avg_by_wc for s in valid_summaries]

                # frequencyベース（有効なサマリーのみ）
                primary_without_avgs_by_freq = [s.primary_rate_without_avg_by_freq for s in valid_summaries]
                primary_with_avgs_by_freq = [s.primary_rate_with_avg_by_freq for s in valid_summaries]
                primary_diff_avgs_by_freq = [s.primary_rate_diff_avg_by_freq for s in valid_summaries]

                primary_rate_data = {
                    "by_imp_wc": {
                        "without": {
                            "avg": round(statistics.mean(primary_without_avgs_by_wc), 2),
                            "median": round(statistics.median(primary_without_avgs_by_wc), 2),
                        },
                        "with": {
                            "avg": round(statistics.mean(primary_with_avgs_by_wc), 2),
                            "median": round(statistics.median(primary_with_avgs_by_wc), 2),
                        },
                        "diff": {
                            "avg": round(statistics.mean(primary_diff_avgs_by_wc), 2),
                            "median": round(statistics.median(primary_diff_avgs_by_wc), 2),
                        },
                    },
                    "by_frequency": {
                        "without": {
                            "avg": round(statistics.mean(primary_without_avgs_by_freq), 2),
                            "median": round(statistics.median(primary_without_avgs_by_freq), 2),
                        },
                        "with": {
                            "avg": round(statistics.mean(primary_with_avgs_by_freq), 2),
                            "median": round(statistics.median(primary_with_avgs_by_freq), 2),
                        },
                        "diff": {
                            "avg": round(statistics.mean(primary_diff_avgs_by_freq), 2),
                            "median": round(statistics.median(primary_diff_avgs_by_freq), 2),
                        },
                    },
                    "num_valid_summaries": len(valid_summaries),
                }
            else:
                primary_rate_data = {
                    "by_imp_wc": {"without": {"avg": 0.0, "median": 0.0}, "with": {"avg": 0.0, "median": 0.0}, "diff": {"avg": 0.0, "median": 0.0}},
                    "by_frequency": {"without": {"avg": 0.0, "median": 0.0}, "with": {"avg": 0.0, "median": 0.0}, "diff": {"avg": 0.0, "median": 0.0}},
                    "num_valid_summaries": 0,
                }

            overall_summary = {
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
                "primary_source_rate": primary_rate_data,
            }

        save_root_summary(
            output_dir=output_dir,
            providers=self.providers,
            num_runs=self.num_runs,
            question_types=QUESTION_TYPES,
            primary_domains=self.primary_domains,
            domain_summaries=domain_data,
            overall_summary=overall_summary,
        )


# =============================================================================
# Main
# =============================================================================

# ターゲットを検出するドメインリスト
TARGET_DOMAINS = ["econ"]


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数をパース

    Returns:
        argparse.Namespace: パース結果
    """
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
    parser.add_argument(
        "-o", "--output-name",
        type=str,
        default=None,
        help="出力フォルダ名（指定しない場合はタイムスタンプ）",
    )
    return parser.parse_args()


async def main():
    """メイン関数"""
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
        targets=[{"id": t["id"], "title": t["title"], "content": t["content"]} for t in targets],
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

    await runner.run_experiment(targets, questions, output_name=args.output_name)


if __name__ == "__main__":
    asyncio.run(main())
