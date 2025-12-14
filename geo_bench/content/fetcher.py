"""
Web Content Fetcher
===================
Seleniumを使用してWebページのコンテンツを取得
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
# Web Content Fetcher
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

    # リアルなブラウザヘッダー
    BROWSER_HEADERS = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
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

    MAX_WORKERS = 20
    DOMAIN_RATE_LIMIT_INTERVAL = float(os.getenv("WEB_RATE_LIMIT_INTERVAL", "0.3"))

    def __init__(self):
        self._http_client: httpx.AsyncClient | None = None
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._domain_last_access: dict[str, float] = {}
        self._global_lock = asyncio.Lock()

    def _create_driver(self) -> webdriver.Chrome:
        """Selenium WebDriverのインスタンスを作成"""
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

        user_data_dir = _create_chrome_user_data_dir()
        options.add_argument(f'--user-data-dir={user_data_dir}')

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        driver.set_page_load_timeout(self.PAGE_LOAD_TIMEOUT)
        driver.implicitly_wait(self.IMPLICIT_WAIT)
        driver.user_data_dir = user_data_dir  # type: ignore

        return driver

    def _quit_driver_safely(self, driver: webdriver.Chrome) -> None:
        """ドライバーを安全に終了"""
        if driver is None:
            return

        try:
            service_pid = None
            if hasattr(driver, 'service') and driver.service and driver.service.process:
                service_pid = driver.service.process.pid

            try:
                import psutil
                if service_pid:
                    parent = psutil.Process(service_pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except Exception:
                            pass
            except ImportError:
                pass
            except Exception:
                pass

            if service_pid:
                try:
                    os.kill(service_pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass

            try:
                driver.quit()
            except Exception:
                pass
        except Exception:
            pass

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
        """Seleniumでページを取得（同期処理）"""
        for retry in range(self.MAX_RETRIES):
            driver = None
            user_data_dir = None

            try:
                driver = self._create_driver()
                user_data_dir = getattr(driver, 'user_data_dir', None)

                driver.get(url)

                WebDriverWait(driver, self.PAGE_LOAD_TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                time.sleep(self.WAIT_TIME)

                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text().strip().lower() if title else ""

                if "just a moment" in title_text or "satisfied" in title_text:
                    print(f"[Web] レートリミット検出、{self.SLEEP_TIME}秒待機してリトライ ({retry + 1}/{self.MAX_RETRIES}): {url}")
                    time.sleep(self.SLEEP_TIME + random.randint(1, 60))
                    continue

                final_url = driver.current_url
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
                self._quit_driver_safely(driver)
                _cleanup_chrome_temp_dir(user_data_dir)

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
        async with self._global_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = asyncio.Lock()
            domain_lock = self._domain_locks[domain]

        async with domain_lock:
            now = time.time()
            last_access = self._domain_last_access.get(domain, 0.0)
            wait_time = self.DOMAIN_RATE_LIMIT_INTERVAL - (now - last_access)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._domain_last_access[domain] = time.time()

    async def fetch(self, url: str) -> tuple[str, str, str]:
        """URLからテキストコンテンツを取得"""
        domain = self._extract_domain(url)
        await self._wait_for_domain_rate_limit(domain)

        try:
            if self._is_pdf_url(url):
                return await self._fetch_pdf(url)

            loop = asyncio.get_event_loop()
            text, media_type, final_url = await loop.run_in_executor(
                self._executor,
                self._fetch_with_selenium,
                url
            )

            text = self._normalize_text(text)

            if len(text) > self.MAX_CONTENT_LENGTH:
                text = text[:self.MAX_CONTENT_LENGTH] + "..."

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
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def _extract_html(self, html: str) -> str:
        """HTMLからテキストを抽出"""
        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
            tag.decompose()

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
        self._executor.shutdown(wait=True, cancel_futures=True)
        _cleanup_all_chrome_dirs()
