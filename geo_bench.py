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

import asyncio
import atexit
import glob
import io
import os
import random
import re
import shutil
import signal
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypedDict
from urllib.parse import urlparse

import httpx
import pdfplumber
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


# =============================================================================
# Chrome Temp Directory Management
# =============================================================================

# グローバルに作成された一時ディレクトリを追跡（クリーンアップ用）
_chrome_temp_dirs: set[str] = set()


def _cleanup_old_chrome_dirs(max_age_hours: float = 24) -> None:
    """古いChromeユーザーデータディレクトリを自動削除"""
    possible_dirs = [
        tempfile.gettempdir(),
        '/tmp',
        os.path.join(os.getcwd(), 'chrome_temp'),
    ]

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for base_dir in possible_dirs:
        if not os.path.exists(base_dir):
            continue
        try:
            pattern = os.path.join(base_dir, 'chrome_user_data_*')
            for dir_path in glob.glob(pattern):
                if not os.path.isdir(dir_path):
                    continue
                try:
                    dir_mtime = os.path.getmtime(dir_path)
                    if current_time - dir_mtime > max_age_seconds:
                        shutil.rmtree(dir_path, ignore_errors=True)
                except (OSError, IOError):
                    continue
        except (OSError, PermissionError):
            continue


def _create_chrome_user_data_dir() -> str:
    """Chrome用のユーザーデータディレクトリを作成"""
    _cleanup_old_chrome_dirs(max_age_hours=24)

    possible_dirs = [
        tempfile.gettempdir(),
        '/tmp',
        os.path.join(os.getcwd(), 'chrome_temp'),
    ]

    for base_dir in possible_dirs:
        try:
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
            if not os.access(base_dir, os.W_OK):
                continue

            unique_name = f'chrome_user_data_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000)}'
            user_data_dir = os.path.join(base_dir, unique_name)
            os.makedirs(user_data_dir, mode=0o700, exist_ok=False)

            # 書き込みテスト
            test_file = os.path.join(user_data_dir, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)

            _chrome_temp_dirs.add(user_data_dir)
            return user_data_dir
        except (OSError, PermissionError, IOError):
            continue

    # フォールバック
    user_data_dir = tempfile.mkdtemp(prefix='chrome_user_data_')
    _chrome_temp_dirs.add(user_data_dir)
    return user_data_dir


def _cleanup_chrome_temp_dir(user_data_dir: str | None) -> None:
    """一時ディレクトリをクリーンアップ"""
    if not user_data_dir:
        return
    try:
        if os.path.exists(user_data_dir):
            shutil.rmtree(user_data_dir, ignore_errors=True)
        _chrome_temp_dirs.discard(user_data_dir)
    except Exception:
        pass


def _cleanup_all_chrome_dirs() -> None:
    """プロセス終了時にすべての一時ディレクトリをクリーンアップ"""
    for temp_dir in list(_chrome_temp_dirs):
        _cleanup_chrome_temp_dir(temp_dir)
    _chrome_temp_dirs.clear()


# プロセス終了時のクリーンアップを登録
atexit.register(_cleanup_all_chrome_dirs)


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
# Web Content Fetcher (Selenium-based)
# =============================================================================

class WebContentFetcher:
    """
    Seleniumを使用してWebページのコンテンツを取得

    JavaScriptでレンダリングされるページに対応。
    PDFはhttpxで直接ダウンロードして処理。
    webdriver_managerで自動的にChromeDriverを管理。
    """

    PAGE_LOAD_TIMEOUT = 30  # ページ読み込みタイムアウト（秒）
    IMPLICIT_WAIT = 10  # 暗黙的待機（秒）
    WAIT_TIME = 2  # 動的コンテンツ読み込み待機（秒）
    SLEEP_TIME = 30  # レートリミット時の待機（秒）
    MAX_RETRIES = 6  # 最大リトライ回数
    MAX_CONTENT_LENGTH = 8000  # 最大文字数

    # リアルなブラウザヘッダー（URL非依存）
    BROWSER_HEADERS = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9,ja;q=0.8",
        "cache-control": "max-age=0",
        "dnt": "1",
        "priority": "u=0, i",
        "sec-ch-ua": '"Chromium";v="143", "Not A(Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    }

    # 並列フェッチ数
    MAX_WORKERS = 20

    # 同一ドメインへのアクセス間隔（秒）
    DOMAIN_RATE_LIMIT_INTERVAL = float(os.getenv("WEB_RATE_LIMIT_INTERVAL", "0.3"))

    def __init__(self):
        self._http_client: httpx.AsyncClient | None = None
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        # ドメインごとのレートリミット管理
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._domain_last_access: dict[str, float] = {}
        self._global_lock = asyncio.Lock()  # ドメインロック取得用

    def _create_driver(self) -> webdriver.Chrome:
        """
        Selenium WebDriverのインスタンスを作成

        毎回新しいドライバーを作成し、使用後にクリーンアップする方式。
        webdriver_managerでChromeDriverを自動管理。
        """
        options = Options()
        options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        options.add_argument(f'user-agent={self.BROWSER_HEADERS["user-agent"]}')
        options.add_argument('--log-level=3')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # ユーザーデータディレクトリを作成
        user_data_dir = _create_chrome_user_data_dir()
        options.add_argument(f'--user-data-dir={user_data_dir}')

        # webdriver_managerでChromeDriverを自動取得
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # タイムアウト設定
        driver.set_page_load_timeout(self.PAGE_LOAD_TIMEOUT)
        driver.implicitly_wait(self.IMPLICIT_WAIT)

        # クリーンアップ用にパスを保存
        driver.user_data_dir = user_data_dir  # type: ignore

        return driver

    async def _get_http_client(self) -> httpx.AsyncClient:
        """PDF取得用のhttpxクライアントを取得"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self.PAGE_LOAD_TIMEOUT,
                follow_redirects=True,
                headers=self.BROWSER_HEADERS,
            )
        return self._http_client

    def _is_pdf_url(self, url: str) -> bool:
        """URLがPDFかどうかを判定"""
        return url.lower().endswith('.pdf')

    def _fetch_with_selenium(self, url: str) -> tuple[str, str, str]:
        """
        Seleniumでページを取得（同期処理）

        毎回新しいドライバーを作成し、使用後にクリーンアップする。
        レートリミット検出時はリトライを行う。

        Returns:
            (content, media_type, final_url)
        """
        for retry in range(self.MAX_RETRIES):
            driver = None
            user_data_dir = None

            try:
                driver = self._create_driver()
                user_data_dir = getattr(driver, 'user_data_dir', None)

                driver.get(url)

                # ページ読み込み完了を待機（body要素の存在を確認）
                WebDriverWait(driver, self.PAGE_LOAD_TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                # 動的コンテンツの読み込みを待つ
                # Note: time.sleep()はスレッドプール内で実行されるため、asyncioイベントループをブロックしない
                time.sleep(self.WAIT_TIME)

                # レートリミット検出（Cloudflare等の"Just a moment"ページ）
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text().strip().lower() if title else ""

                # レートリミットされている場合はリトライ
                if "just a moment" in title_text or "satisfied" in title_text:
                    print(f"[Web] レートリミット検出、{self.SLEEP_TIME}秒待機してリトライ ({retry + 1}/{self.MAX_RETRIES}): {url}")
                    time.sleep(self.SLEEP_TIME + random.randint(1, 60))
                    continue

                # 最終URL（リダイレクト後）
                final_url = driver.current_url

                # テキストを抽出
                text = self._extract_html(html)

                return text, "HTML", final_url

            except TimeoutException:
                if retry < self.MAX_RETRIES - 1:
                    print(f"[Web] タイムアウト、リトライ ({retry + 1}/{self.MAX_RETRIES}): {url}")
                    continue
                raise Exception(f"ページ読み込みタイムアウト ({self.PAGE_LOAD_TIMEOUT}秒)")
            except WebDriverException as e:
                if retry < self.MAX_RETRIES - 1:
                    print(f"[Web] Seleniumエラー、リトライ ({retry + 1}/{self.MAX_RETRIES}): {url} - {e}")
                    continue
                raise Exception(f"Seleniumエラー: {e}")
            finally:
                # ドライバーを終了
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass
                # 一時ディレクトリをクリーンアップ
                _cleanup_chrome_temp_dir(user_data_dir)

        # リトライ回数超過
        raise Exception(f"最大リトライ回数 ({self.MAX_RETRIES}) を超過")

    def _extract_domain(self, url: str) -> str:
        """URLからドメインを抽出"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return "unknown"

    async def _wait_for_domain_rate_limit(self, domain: str) -> None:
        """ドメインごとのレートリミットを待機"""
        # ドメイン用のロックを取得（なければ作成）
        async with self._global_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = asyncio.Lock()
            domain_lock = self._domain_locks[domain]

        # ドメインごとにレートリミットを適用
        async with domain_lock:
            now = time.time()
            last_access = self._domain_last_access.get(domain, 0.0)
            wait_time = self.DOMAIN_RATE_LIMIT_INTERVAL - (now - last_access)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._domain_last_access[domain] = time.time()

    async def fetch(self, url: str) -> tuple[str, str, str]:
        """
        URLからテキストコンテンツを取得

        Returns:
            (content, media_type, final_url): コンテンツ、メディアタイプ、リダイレクト後の最終URL
        """
        # ドメインごとのレートリミット
        domain = self._extract_domain(url)
        await self._wait_for_domain_rate_limit(domain)

        try:
            # PDFの場合はhttpxで直接取得
            if self._is_pdf_url(url):
                return await self._fetch_pdf(url)

            # HTMLページはSeleniumで取得（スレッドプールで実行）
            loop = asyncio.get_event_loop()
            text, media_type, final_url = await loop.run_in_executor(
                self._executor,
                self._fetch_with_selenium,
                url
            )

            # 正規化
            text = self._normalize_text(text)

            # 長さ制限
            if len(text) > self.MAX_CONTENT_LENGTH:
                text = text[:self.MAX_CONTENT_LENGTH] + "..."

            # ログ出力
            if final_url != url:
                print(f"[Web] {media_type} {len(text)}文字 <- {final_url} (リダイレクト最終URL)")
            else:
                print(f"[Web] {media_type} {len(text)}文字 <- {url}")

            return text, media_type, final_url

        except Exception as e:
            print(f"[Web] エラー <- {url}: {e}")
            return f"[Error fetching {url}: {e}]", "ERROR", url

    async def _fetch_pdf(self, url: str) -> tuple[str, str, str]:
        """PDFをhttpxで取得してテキスト抽出"""
        client = await self._get_http_client()
        response = await client.get(url)
        response.raise_for_status()

        final_url = str(response.url)
        text = self._extract_pdf(response.content)
        text = self._normalize_text(text)

        if len(text) > self.MAX_CONTENT_LENGTH:
            text = text[:self.MAX_CONTENT_LENGTH] + "..."

        if final_url != url:
            print(f"[Web] PDF {len(text)}文字 <- {final_url} (リダイレクト最終URL)")
        else:
            print(f"[Web] PDF {len(text)}文字 <- {url}")

        return text, "PDF", final_url

    def _normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        text = re.sub(r'[^\S\n]+', ' ', text)  # 改行以外の空白を正規化
        text = re.sub(r' *\n *', '\n', text)   # 改行前後の空白を除去
        text = re.sub(r'\n+', '\n', text)      # 連続する改行を1つに
        return text.strip()

    def _extract_html(self, html: str) -> str:
        """HTMLからテキストを抽出"""
        soup = BeautifulSoup(html, 'html.parser')

        # 不要な要素を削除
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
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
        """リソースを解放"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._executor.shutdown(wait=False)
        # 残っている一時ディレクトリをクリーンアップ
        _cleanup_all_chrome_dirs()


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

