"""
Gemini Client (Google)
======================
Google Gemini クライアント実装
"""

from __future__ import annotations

import asyncio
import os
import random
import re

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import LLMClient


class GeminiClient(LLMClient):
    """Google Gemini クライアント（検索グラウンディング使用）"""

    MODEL = "gemini-2.5-flash"
    DEFAULT_RATE_LIMIT_INTERVAL = 3.3
    MAX_RETRIES = 5  # 429エラー時の最大リトライ回数

    def __init__(self):
        super().__init__()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY が設定されていません")
        self.client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return f"Gemini ({self.MODEL})"

    async def _retry_on_error(self, func, *args, **kwargs):
        """429エラーまたは接続エラー時にリトライ"""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                result = await func(*args, **kwargs)
                print(f"[Gemini] 成功")
                return result
            except genai_errors.ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 20 + random.randint(0, 60)
                    print(f"[Gemini] 429 RESOURCE_EXHAUSTED - {wait_time}秒待機後リトライ ({attempt + 1}/{self.MAX_RETRIES})")
                    await asyncio.sleep(wait_time)
                    last_error = e
                else:
                    raise
            except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
                # ネットワーク接続エラー（DNS解決失敗、タイムアウト等）
                wait_time = 10 + random.randint(0, 20)
                print(f"[Gemini] 接続エラー - {wait_time}秒待機後リトライ ({attempt + 1}/{self.MAX_RETRIES}): {e}")
                await asyncio.sleep(wait_time)
                last_error = e
        raise RuntimeError(f"Gemini API: {self.MAX_RETRIES}回リトライしましたが失敗しました: {last_error}")

    async def acall_standard(self, prompt: str) -> str:
        """非同期で Gemini を呼び出し（グラウンディングなし）"""
        await self._wait_for_rate_limit()

        async def _call():
            # google-genai は同期APIのため、スレッドプールで実行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.MODEL,
                    contents=prompt,
                )
            )
            text = response.text.strip() if response.text else ""
            if not text:
                raise RuntimeError("LLM からのレスポンスにテキストが含まれていません")
            return text

        return await self._retry_on_error(_call)

    async def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """Google検索グラウンディングを使用してソースURLを取得"""
        await self._wait_for_rate_limit()

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        async def _call():
            # google-genai は同期APIのため、スレッドプールで実行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.MODEL,
                    contents=query,
                    config=config,
                )
            )
            return response

        response = await self._retry_on_error(_call)

        # groundingMetadata からURLを抽出
        results = []
        seen_urls: set[str] = set()

        # candidates から grounding_metadata を取得
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                # grounding_chunks からURLを抽出
                if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                            url = chunk.web.uri
                            if url not in seen_urls:
                                seen_urls.add(url)
                                results.append({"url": url})

        # フォールバック: テキストからURLを抽出
        if not results and response.text:
            urls = re.findall(r'https?://[^\s\)]+', response.text)
            for url in urls[:max_results]:
                clean_url = url.rstrip('.,;:')
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    results.append({"url": clean_url})

        return results[:max_results]
