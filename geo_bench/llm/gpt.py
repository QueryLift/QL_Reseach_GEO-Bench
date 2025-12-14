"""
GPT Client (OpenAI)
===================
OpenAI GPT-5 クライアント実装
"""

from __future__ import annotations

import os
import re
from typing import Any

import openai

from .base import LLMClient


class GPTClient(LLMClient):
    """OpenAI GPT-5 クライアント"""

    MODEL = "gpt-5"
    DEFAULT_RATE_LIMIT_INTERVAL = 0.5

    def __init__(self):
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY が設定されていません")
        self.async_client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return f"GPT ({self.MODEL})"

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Responses API からテキストを抽出"""
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError("LLM からのレスポンスにテキストが含まれていません")
        return text

    async def acall_standard(self, prompt: str) -> str:
        """非同期で gpt-5 を呼び出し"""
        await self._wait_for_rate_limit()
        response = await self.async_client.responses.create(
            model=self.MODEL,
            reasoning={"effort": "low"},
            input=prompt,
        )
        return self._extract_text(response)

    async def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """Web検索を実行してソースURLを取得"""
        await self._wait_for_rate_limit()

        response = await self.async_client.responses.create(
            model=self.MODEL,
            tools=[{"type": "web_search"}],
            input=query,
        )

        # output_text からURLを抽出
        text = response.output_text or ""
        urls = re.findall(r'https?://[^\s\)]+', text)

        results = []
        for url in urls[:max_results]:
            results.append({"url": url.rstrip('.,;:')})

        return results
