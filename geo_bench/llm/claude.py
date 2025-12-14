"""
Claude Client (Anthropic)
=========================
Anthropic Claude クライアント実装
"""

from __future__ import annotations

import os
import re
from typing import Any

import anthropic

from .base import LLMClient


class ClaudeClient(LLMClient):
    """Anthropic Claude クライアント（Web検索ツール使用）"""

    MODEL = "claude-sonnet-4-5-20250929"
    DEFAULT_RATE_LIMIT_INTERVAL = 2.0

    def __init__(self):
        super().__init__()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY が設定されていません")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return f"Claude ({self.MODEL})"

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Messages API からテキストを抽出"""
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text.strip()
        raise RuntimeError("LLM からのレスポンスにテキストが含まれていません")

    async def acall_standard(self, prompt: str) -> str:
        """非同期で Claude を呼び出し（Web検索なし）"""
        await self._wait_for_rate_limit()
        response = await self.async_client.messages.create(
            model=self.MODEL,
            max_tokens=20000,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_text(response)

    async def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """Web検索ツールを使用してソースURLを取得"""
        await self._wait_for_rate_limit()

        response = await self.async_client.messages.create(
            model=self.MODEL,
            max_tokens=20000,
            messages=[{"role": "user", "content": query}],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_results,
            }],
        )

        # レスポンスからURLを抽出
        results = []
        seen_urls: set[str] = set()

        for block in response.content:
            # web_search_tool_result からURLを抽出
            if hasattr(block, 'type') and block.type == 'web_search_tool_result':
                if hasattr(block, 'content') and block.content:
                    for result in block.content:
                        if hasattr(result, 'url') and result.url not in seen_urls:
                            seen_urls.add(result.url)
                            results.append({"url": result.url})

            # テキストブロックの引用からもURLを抽出
            if hasattr(block, 'citations') and block.citations:
                for citation in block.citations:
                    if hasattr(citation, 'url') and citation.url not in seen_urls:
                        seen_urls.add(citation.url)
                        results.append({"url": citation.url})

        # フォールバック: テキストからURLを抽出
        if not results:
            text = self._extract_text(response)
            urls = re.findall(r'https?://[^\s\)]+', text)
            for url in urls[:max_results]:
                clean_url = url.rstrip('.,;:')
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    results.append({"url": clean_url})

        return results[:max_results]
