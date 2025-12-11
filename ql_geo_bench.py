"""
GEO-bench Emulator
==================
GEO (Generative Engine Optimization) 論文に基づく Web 検索エミュレータ。
target_contents の有無による引用パターンの差分をトレースする。

Reference: Aggarwal et al., "GEO: Generative Engine Optimization", KDD 2024
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import TypedDict

import httpx
import openai
from bs4 import BeautifulSoup


# =============================================================================
# Type Definitions
# =============================================================================

class SourceContent(TypedDict):
    """検索ソースの型定義"""
    url: str
    content: str


@dataclass
class Citation:
    """引用情報"""
    index: int
    url: str
    sentences: list[str] = field(default_factory=list)
    word_count: int = 0
    position_sum: float = 0.0  # Position-adjusted計算用


class TargetContent(TypedDict):
    """ターゲットコンテンツの型定義（ID付き）"""
    id: str
    url: str
    content: str


@dataclass
class CitationMetrics:
    """
    引用メトリクス (GEO論文 Section 2.2.1)

    Attributes:
        word_count: 式(2) Imp_wc - 正規化ワードカウント (%)
            Imp_wc(c_i, r) = Σ_{s∈S_{c_i}} |s| / Σ_{s∈S_r} |s|
            引用に関連する文のワード数の割合

        position_adjusted: 式(3) Imp_pwc - 位置調整済みワードカウント (%)
            Imp_pwc(c_i, r) = Σ_{s∈S_{c_i}} |s| · e^{-pos(s)/|S|} / Σ_{s∈S_r} |s|
            回答の前半に引用されるほど高い値になる

        citation_frequency: 引用された回数
        first_citation_position: 最初に引用された文の位置 (0-indexed)
    """
    # 式(2) Imp_wc: Word Count
    word_count: float = 0.0
    # 式(3) Imp_pwc: Position-Adjusted Word Count
    position_adjusted: float = 0.0

    # 追加メトリクス
    citation_frequency: int = 0
    first_citation_position: int | None = None

    def __repr__(self) -> str:
        return (
            f"CitationMetrics(word_count={self.word_count:.2f}, "
            f"position_adjusted={self.position_adjusted:.2f}, "
            f"frequency={self.citation_frequency}, "
            f"first_pos={self.first_citation_position})"
        )


@dataclass
class TargetInfo:
    """ターゲット情報"""
    target_id: str
    target_url: str
    target_index: int  # sources_with_targets 内でのインデックス（1-indexed）

    def get_visibility(self, citations: dict[int, CitationMetrics]) -> dict:
        """ターゲットの visibility を計算"""
        metrics = citations.get(self.target_index)

        if metrics is None:
            return {
                "included": False,
                "word_count": 0,
                "position_adjusted": 0,
                "citation_frequency": 0,
            }

        return {
            "included": True,
            "word_count": metrics.word_count,
            "position_adjusted": metrics.position_adjusted,
            "citation_frequency": metrics.citation_frequency,
            "first_citation_position": metrics.first_citation_position,
        }


@dataclass
class GEOBenchResult:
    """GEO-bench 実行結果"""
    question: str

    # ターゲットなしの結果
    answer_without_targets: str
    citations_without_targets: dict[int, CitationMetrics]

    # 全ターゲットを含む結果（1回の回答生成）
    answer_with_targets: str
    citations_with_targets: dict[int, CitationMetrics]

    # ターゲット情報（どのインデックスがどのターゲットか）
    target_infos: list[TargetInfo] = field(default_factory=list)

    # Web検索で取得したソース（ターゲットを含まない）
    sources: list[SourceContent] = field(default_factory=list)



# =============================================================================
# Citation Analyzer
# =============================================================================

class CitationAnalyzer:
    """
    引用を分析するクラス (GEO論文 Section 2.2.1)

    メトリクス:
    - Word Count: 引用に関連する文の正規化ワードカウント
    - Position-Adjusted Word Count: 位置による重み付けワードカウント
    """

    # 引用パターン: [1], [2], [1][2][3] など
    CITATION_PATTERN = re.compile(r'\[(\d+)\]')

    def analyze(self, response: str, sources: list[SourceContent]) -> dict[int, CitationMetrics]:
        """
        レスポンスから引用メトリクスを計算

        Args:
            response: LLMのレスポンステキスト
            sources: ソースのリスト（インデックスはURLと対応）

        Returns:
            インデックス -> CitationMetrics のマッピング
        """
        # 文に分割
        sentences = self._split_into_sentences(response)
        total_word_count = sum(len(s.split()) for s in sentences)

        if total_word_count == 0:
            return {}

        # 各ソースの引用情報を収集
        citations: dict[int, Citation] = {}
        for idx in range(len(sources)):
            citations[idx + 1] = Citation(
                index=idx + 1,
                url=sources[idx].get("url", ""),
            )

        # 各文を解析
        # 式(2), (3) の分子部分を計算
        for pos, sentence in enumerate(sentences):
            cited_indices = self._extract_citations(sentence)
            word_count = len(sentence.split())  # |s|

            # 複数引用の場合はワードカウントを分割（論文の仕様）
            share = word_count / len(cited_indices) if cited_indices else 0

            for idx in cited_indices:
                if idx in citations:
                    citations[idx].sentences.append(sentence)
                    # 式(2) Imp_wc: Σ|s| の累積
                    citations[idx].word_count += share
                    # 式(3) Imp_pwc: Σ(|s| · e^{-pos(s)/|S|}) の累積
                    # pos(s) = 文の位置, |S| = 総文数
                    position_weight = pow(2.718281828, -pos / len(sentences)) if sentences else 1
                    citations[idx].position_sum += share * position_weight

        # メトリクスを計算
        # 式(2), (3) の分母（Σ_{s∈S_r} |s|）で正規化
        metrics: dict[int, CitationMetrics] = {}
        for idx, cit in citations.items():
            if cit.word_count > 0 or cit.sentences:
                # 式(2) Imp_wc = Σ|s| / Σ|s_r| (正規化)
                normalized_wc = cit.word_count / total_word_count if total_word_count > 0 else 0
                # 式(3) Imp_pwc = Σ(|s|·e^{-pos/|S|}) / Σ|s_r| (正規化)
                normalized_pwc = cit.position_sum / total_word_count if total_word_count > 0 else 0

                # 最初の引用位置を見つける
                first_pos = None
                for pos, sentence in enumerate(sentences):
                    if idx in self._extract_citations(sentence):
                        first_pos = pos
                        break

                metrics[idx] = CitationMetrics(
                    word_count=normalized_wc * 100,  # パーセンテージ
                    position_adjusted=normalized_pwc * 100,
                    citation_frequency=len(cit.sentences),
                    first_citation_position=first_pos,
                )

        return metrics

    def _split_into_sentences(self, text: str) -> list[str]:
        """テキストを文に分割"""
        # 簡易的な文分割（。.!? で分割）
        sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_citations(self, sentence: str) -> list[int]:
        """文から引用インデックスを抽出"""
        matches = self.CITATION_PATTERN.findall(sentence)
        return [int(m) for m in matches]


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """OpenAI API クライアント"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY が設定されていません")

        self.async_client = openai.AsyncOpenAI(api_key=api_key)

        # レートリミット設定
        self.rate_limit_interval = float(os.getenv("LLM_RATE_LIMIT_INTERVAL", "2.0"))
        self._rate_limit_lock = asyncio.Lock()
        self._last_call_time = 0.0

    @staticmethod
    def _extract_text(response) -> str:
        """Responses API からテキストを抽出"""
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError("LLM からのレスポンスにテキストが含まれていません")
        return text

    async def _wait_for_rate_limit(self):
        """レートリミットを待機"""
        async with self._rate_limit_lock:
            now = time.time()
            wait_time = self.rate_limit_interval - (now - self._last_call_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call_time = time.time()

    async def acall_standard(self, prompt: str) -> str:
        """非同期で gpt-5 を呼び出し"""
        await self._wait_for_rate_limit()
        response = await self.async_client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            input=prompt,
        )
        return self._extract_text(response)

    async def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Web検索を実行してソースURLを取得

        OpenAI の web_search_preview ツールを使用
        """
        await self._wait_for_rate_limit()

        response = await self.async_client.responses.create(
            model="gpt-5",
            tools=[
                { "type": "web_search" },
            ],
            input=f"Search for: {query}",
        )

        # 検索結果からURLを抽出
        results = []
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'web_search_call':
                    # web_search_call の結果を解析
                    pass

        # output_text からURLを抽出（フォールバック）
        text = response.output_text or ""
        urls = re.findall(r'https?://[^\s\)]+', text)

        for url in urls[:max_results]:
            results.append({"url": url.rstrip('.,;:')})

        return results


# =============================================================================
# Web Content Fetcher
# =============================================================================

class WebContentFetcher:
    """Webページのコンテンツを取得"""

    DEFAULT_TIMEOUT = 10.0
    MAX_CONTENT_LENGTH = 10000  # 最大文字数

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.DEFAULT_TIMEOUT,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; GEO-bench/1.0)"
                }
            )
        return self._client

    async def fetch(self, url: str) -> str:
        """URLからテキストコンテンツを取得"""
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            # HTMLをパース
            soup = BeautifulSoup(response.text, 'html.parser')

            # 不要な要素を削除
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()

            # テキストを抽出
            text = soup.get_text(separator=' ', strip=True)

            # 正規化
            text = re.sub(r'\s+', ' ', text)

            # 長さ制限
            if len(text) > self.MAX_CONTENT_LENGTH:
                text = text[:self.MAX_CONTENT_LENGTH] + "..."

            return text

        except Exception as e:
            return f"[Error fetching {url}: {e}]"

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# GEO Bench
# =============================================================================

class GEOBench:
    """
    GEO-bench エミュレータ

    GEO論文に基づき、Generative Engine の動作をエミュレートし、
    target_contents の有無による引用パターンの差分を分析する。
    複数ターゲットに対応。
    """

    # GEO論文 Listing 1 のプロンプト
    PROMPT_TEMPLATE = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results.
The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Francais, Espamol, Deutsch, or Japanese should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {question}

Search Results:
{sources}"""

    def __init__(
        self,
        llm: LLMClient,
        target_contents: list[TargetContent],
        max_sources: int = 5,
    ):
        """
        Args:
            llm: LLMクライアント
            target_contents: 比較対象のコンテンツリスト（各要素は id, url, content を含む辞書）
            max_sources: 取得するソースの最大数
        """
        self.llm = llm
        self.target_contents = target_contents
        self.max_sources = max_sources
        self.fetcher = WebContentFetcher()
        self.analyzer = CitationAnalyzer()

    async def run(self, question: str) -> GEOBenchResult:
        """
        GEO-bench を実行

        Args:
            question: ユーザーの質問

        Returns:
            GEOBenchResult: 比較結果（全ターゲット一括）
        """
        # 1. Web検索でソースを取得
        source_urls = await self._search_web(question)

        # 2. 各ソースのコンテンツを取得
        sources = await self._fetch_sources(source_urls)

        # 3. ターゲットなしで回答生成
        answer_without = await self._generate_answer(question, sources)
        metrics_without = self.analyzer.analyze(answer_without, sources)

        # 4. 全ターゲットを一括でソースリストに追加
        sources_with_targets = sources.copy()
        target_infos: list[TargetInfo] = []

        for target in self.target_contents:
            target_source: SourceContent = {
                "url": target.get("url", ""),
                "content": target.get("content", ""),
            }
            sources_with_targets.append(target_source)
            target_index = len(sources_with_targets)  # 1-indexed（追加後の長さ）

            target_infos.append(TargetInfo(
                target_id=target.get("id", ""),
                target_url=target.get("url", ""),
                target_index=target_index,
            ))

        # 5. 全ターゲットを含むソースで1回だけ回答生成
        answer_with = await self._generate_answer(question, sources_with_targets)
        metrics_with = self.analyzer.analyze(answer_with, sources_with_targets)

        return GEOBenchResult(
            question=question,
            answer_without_targets=answer_without,
            citations_without_targets=metrics_without,
            answer_with_targets=answer_with,
            citations_with_targets=metrics_with,
            target_infos=target_infos,
            sources=sources,
        )

    async def _search_web(self, query: str) -> list[str]:
        """Web検索を実行してURLリストを取得"""
        results = await self.llm.search_web(query, max_results=self.max_sources)
        return [r["url"] for r in results if "url" in r]

    async def _fetch_sources(self, urls: list[str]) -> list[SourceContent]:
        """複数のURLからコンテンツを並列取得"""
        tasks = [self.fetcher.fetch(url) for url in urls]
        contents = await asyncio.gather(*tasks)

        sources: list[SourceContent] = []
        for url, content in zip(urls, contents):
            sources.append({
                "url": url,
                "content": content,
            })

        return sources

    async def _generate_answer(
        self,
        question: str,
        sources: list[SourceContent],
    ) -> str:
        """ソースを使って回答を生成"""
        prompt = self._build_prompt(question, sources)
        return await self.llm.acall_standard(prompt)

    def _build_prompt(
        self,
        question: str,
        sources: list[SourceContent],
    ) -> str:
        """GEO論文のプロンプトを構築"""
        sources_text = self._format_sources(sources)
        return self.PROMPT_TEMPLATE.format(
            question=question,
            sources=sources_text,
        )

    def _format_sources(self, sources: list[SourceContent]) -> str:
        """ソースをプロンプト用にフォーマット"""
        lines = []
        for i, source in enumerate(sources, 1):
            url = source.get("url", "unknown")
            content = source.get("content", "")
            lines.append(f"[{i}] URL: {url}")
            lines.append(f"    Content: {content}")
            lines.append("")
        return "\n".join(lines)

    async def close(self):
        """リソースをクリーンアップ"""
        await self.fetcher.close()


# =============================================================================
# Utility Functions
# =============================================================================

def strip_markdown(content: str) -> str:
    """
    Markdownテキストをプレーンテキストに変換

    Args:
        content: Markdown形式のテキスト

    Returns:
        プレーンテキスト
    """
    # 見出し（# ## ### など）を除去
    content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
    # 太字・斜体（** __ * _）を除去
    content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
    content = re.sub(r'__(.+?)__', r'\1', content)
    content = re.sub(r'\*(.+?)\*', r'\1', content)
    content = re.sub(r'_(.+?)_', r'\1', content)
    # リンク [text](url) を text に変換
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    # インラインコード ` を除去
    content = re.sub(r'`([^`]+)`', r'\1', content)
    # 水平線 --- *** ___ を除去
    content = re.sub(r'^[-*_]{3,}\s*$', '', content, flags=re.MULTILINE)
    # リスト記号（- * + 1. など）を除去
    content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
    # 引用 > を除去
    content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
    # 連続する空行を1つに
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()

