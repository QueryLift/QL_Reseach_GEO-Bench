"""
GEO-bench Core Components
=========================
GEO (Generative Engine Optimization) 論文に基づく引用分析とWebコンテンツ取得。

提供するコンポーネント:
- CitationAnalyzer: LLM回答の引用パターンを分析
- WebContentFetcher: WebページのコンテンツをHTML/PDFから取得
- strip_markdown: Markdownをプレーンテキストに変換

Reference: Aggarwal et al., "GEO: Generative Engine Optimization", KDD 2024
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import TypedDict

import httpx
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

    def _is_pdf(self, response: httpx.Response, url: str) -> bool:
        """Content-TypeヘッダーまたはURLからPDFかどうかを判定"""
        # Content-Typeヘッダーを優先（リダイレクト後のヘッダー）
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" in content_type:
            return True
        # フォールバック: URL末尾で判定
        if url.lower().endswith('.pdf'):
            return True
        return False

    async def fetch(self, url: str) -> tuple[str, str, str]:
        """
        URLからテキストコンテンツを取得

        Returns:
            (content, media_type, final_url): コンテンツ、メディアタイプ、リダイレクト後の最終URL
        """
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            # リダイレクト後の最終URL
            final_url = str(response.url)

            # PDFの場合（Content-TypeヘッダーまたはURL末尾で判定）
            if self._is_pdf(response, final_url):
                text = self._extract_pdf(response.content)
                media_type = "PDF"
            else:
                # UTF-8でデコード
                response.encoding = 'utf-8'
                text = self._extract_html(response.text)
                media_type = "HTML"

            # 正規化（改行は保持）
            text = re.sub(r'[^\S\n]+', ' ', text)  # 改行以外の空白を正規化
            text = re.sub(r' *\n *', '\n', text)   # 改行前後の空白を除去
            text = re.sub(r'\n+', '\n', text)      # 連続する改行を1つに

            # 長さ制限
            if len(text) > self.MAX_CONTENT_LENGTH:
                text = text[:self.MAX_CONTENT_LENGTH] + "..."

            # リダイレクトがあった場合はログに表示
            if final_url != url:
                print(f"[Web] {media_type} {len(text)}文字 <- {url} -> {final_url}")
            else:
                print(f"[Web] {media_type} {len(text)}文字 <- {url}")
            return text, media_type, final_url

        except Exception as e:
            print(f"[Web] エラー <- {url}: {e}")
            return f"[Error fetching {url}: {e}]", "ERROR", url

    def _extract_html(self, html: str) -> str:
        """HTMLからテキストを抽出"""
        soup = BeautifulSoup(html, 'html.parser')

        # 不要な要素を削除
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        # 見出しタグの前後に改行を挿入
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.insert_before('\n')
            tag.insert_after('\n')

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
    # 見出し（# ## ### など）の記号のみ除去（改行は保持）
    content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)
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

