"""
GEO-bench Emulator
==================
GEO (Generative Engine Optimization) 論文に基づく Web 検索エミュレータ。
target_contents の有無による引用パターンの差分をトレースする。

Reference: Aggarwal et al., "GEO: Generative Engine Optimization", KDD 2024
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import time
from dataclasses import dataclass, field
from typing import TypedDict

import httpx
import openai
import pdfplumber
from bs4 import BeautifulSoup


# =============================================================================
# Type Definitions
# =============================================================================

class SourceContent(TypedDict):
    """検索ソースの型定義"""
    url: str
    content: str
    media_type: str  # "PDF", "HTML", "TARGET", "ERROR"


@dataclass
class Citation:
    """引用情報"""
    index: int
    url: str
    sentences: list[str] = field(default_factory=list)
    word_count: int = 0
    position_sum: float = 0.0  # Position-adjusted計算用
    first_pos: int | None = None  # 最初に引用された位置


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
        imp_wc: 式(2) Imp_wc - 正規化ワードカウント (%)
            Imp_wc(c_i, r) = Σ_{s∈S_{c_i}} |s| / Σ_{s∈S_r} |s|

        imp_pwc: 式(3) Imp_pwc - 位置調整済みワードカウント (%)
            Imp_pwc(c_i, r) = Σ_{s∈S_{c_i}} |s| · e^{-pos(s)/|S|} / Σ_{s∈S_r} |s|

        citation_frequency: 引用された回数
        first_citation_position: 最初に引用された文の位置 (0-indexed)
    """
    imp_wc: float = 0.0
    imp_pwc: float = 0.0
    citation_frequency: int = 0
    first_citation_position: int | None = None


@dataclass
class TargetInfo:
    """ターゲット情報"""
    target_id: str
    target_url: str
    target_index: int  # sources_with_targets 内でのインデックス（1-indexed）


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

        # 引用された Citation のみオンデマンドで作成
        citations: dict[int, Citation] = {}
        num_sources = len(sources)

        # 各文を解析し、引用ごとに分子を累積
        # - Imp_wc 分子: Σ|s| （引用された文のワード数合計）
        # - Imp_pwc 分子: Σ|s|·e^{-pos/|S|} （位置重み付きワード数合計）
        num_sentences = len(sentences)
        for pos, sentence in enumerate(sentences):
            cited_indices = self._extract_citations(sentence)
            if not cited_indices:
                continue

            word_count = len(sentence.split())  # |s|
            share = word_count / len(cited_indices)  # 複数引用時は均等分割
            weight = pow(2.718281828, -pos / num_sentences)  # e^{-pos/|S|}

            for idx in cited_indices:
                if idx < 1 or idx > num_sources:
                    continue  # 無効なインデックスはスキップ
                if idx not in citations:
                    citations[idx] = Citation(index=idx, url=sources[idx - 1]["url"])
                cit = citations[idx]
                if cit.first_pos is None:
                    cit.first_pos = pos
                cit.sentences.append(sentence)
                cit.word_count += share
                cit.position_sum += share * weight

        # 分母で正規化してメトリクスを生成
        # - Imp_wc = 分子 / Σ|s_r| * 100 (%)
        # - Imp_pwc = 分子 / Σ|s_r| * 100 (%)
        return {
            cit.index: CitationMetrics(
                imp_wc=cit.word_count / total_word_count * 100,
                imp_pwc=cit.position_sum / total_word_count * 100,
                citation_frequency=len(cit.sentences),
                first_citation_position=cit.first_pos,
            )
            for cit in citations.values()
        }

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
        self.rate_limit_interval = float(os.getenv("LLM_RATE_LIMIT_INTERVAL", "0.5"))
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

    async def fetch(self, url: str) -> tuple[str, str]:
        """
        URLからテキストコンテンツを取得

        Returns:
            (content, media_type): コンテンツとメディアタイプ（"PDF" or "HTML"）
        """
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            # PDFの場合
            if url.lower().endswith('.pdf'):
                text = self._extract_pdf(response.content)
                media_type = "PDF"
            else:
                # UTF-8でデコード
                response.encoding = 'utf-8'
                text = self._extract_html(response.text)
                media_type = "HTML"

            # 正規化
            text = re.sub(r'\s+', ' ', text)

            # 長さ制限
            if len(text) > self.MAX_CONTENT_LENGTH:
                text = text[:self.MAX_CONTENT_LENGTH] + "..."

            print(f"[Web] {media_type} {len(text)}文字 <- {url}")
            return text, media_type

        except Exception as e:
            print(f"[Web] エラー <- {url}: {e}")
            return f"[Error fetching {url}: {e}]", "ERROR"

    def _extract_html(self, html: str) -> str:
        """HTMLからテキストを抽出"""
        soup = BeautifulSoup(html, 'html.parser')

        # 不要な要素を削除
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        return soup.get_text(separator=' ', strip=True)

    def _extract_pdf(self, content: bytes) -> str:
        """PDFからテキストを抽出"""
        text_parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return ' '.join(text_parts)

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
        print("[API] search_reference: Web検索を実行中...")
        source_urls = await self._search_web(question)

        # 2. 各ソースのコンテンツを取得
        sources = await self._fetch_sources(source_urls)

        # 3. ターゲットありのソースリストを事前構築
        sources_with_targets = sources.copy()
        target_infos: list[TargetInfo] = []

        for target in self.target_contents:
            target_source: SourceContent = {
                "url": target["url"],
                "content": target["content"],
                "media_type": "TARGET",
            }
            sources_with_targets.append(target_source)
            target_index = len(sources_with_targets)  # 1-indexed（追加後の長さ）

            target_infos.append(TargetInfo(
                target_id=target["id"],
                target_url=target["url"],
                target_index=target_index,
            ))

        # 4. without/with を並列で回答生成
        print("[API] without/with: 回答生成中（並列）...")
        answer_without, answer_with = await asyncio.gather(
            self._generate_answer(question, sources),
            self._generate_answer(question, sources_with_targets),
        )

        # 5. メトリクス計算
        metrics_without = self.analyzer.analyze(answer_without, sources)
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
            url = source["url"]
            content = source["content"]
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

