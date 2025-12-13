"""
LLM Clients for GEO Benchmark

Abstract base class and implementations for GPT, Claude, and Gemini.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import anthropic
import openai
from google import genai
from google.genai import types


# =============================================================================
# Abstract Base Class
# =============================================================================

class LLMClient(ABC):
    """LLMクライアントの抽象基底クラス"""

    def __init__(self):
        # レートリミット設定
        self.rate_limit_interval = float(os.getenv("LLM_RATE_LIMIT_INTERVAL", "0.5"))
        self._rate_limit_lock = asyncio.Lock()
        self._last_call_time = 0.0

    async def _wait_for_rate_limit(self):
        """レートリミットを待機"""
        async with self._rate_limit_lock:
            now = time.time()
            wait_time = self.rate_limit_interval - (now - self._last_call_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call_time = time.time()

    @abstractmethod
    async def acall_standard(self, prompt: str) -> str:
        """標準的なLLM呼び出し（Web検索なし）"""
        pass

    @abstractmethod
    async def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """Web検索を実行してソースURLを取得"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """クライアント名"""
        pass


# =============================================================================
# GPT Client (OpenAI)
# =============================================================================

class GPTClient(LLMClient):
    """OpenAI GPT-5 クライアント"""

    MODEL = "gpt-5"

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
            input=f"Search for: {query}",
        )

        # output_text からURLを抽出
        text = response.output_text or ""
        urls = re.findall(r'https?://[^\s\)]+', text)

        results = []
        for url in urls[:max_results]:
            results.append({"url": url.rstrip('.,;:')})

        return results


# =============================================================================
# Claude Client (Anthropic)
# =============================================================================

class ClaudeClient(LLMClient):
    """Anthropic Claude クライアント（Web検索ツール使用）"""

    MODEL = "claude-sonnet-4-5-20250929"

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
            messages=[{"role": "user", "content": f"Search for: {query}"}],
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


# =============================================================================
# Gemini Client (Google)
# =============================================================================

class GeminiClient(LLMClient):
    """Google Gemini クライアント（検索グラウンディング使用）"""

    MODEL = "gemini-2.5-flash"

    def __init__(self):
        super().__init__()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY が設定されていません")
        self.client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return f"Gemini ({self.MODEL})"

    async def acall_standard(self, prompt: str) -> str:
        """非同期で Gemini を呼び出し（グラウンディングなし）"""
        await self._wait_for_rate_limit()

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

    async def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """Google検索グラウンディングを使用してソースURLを取得"""
        await self._wait_for_rate_limit()

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        # google-genai は同期APIのため、スレッドプールで実行
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.MODEL,
                contents=f"Search for: {query}",
                config=config,
            )
        )

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


# =============================================================================
# Factory
# =============================================================================

def create_llm_client(provider: str = "gpt") -> LLMClient:
    """
    LLMクライアントを作成するファクトリ関数

    Args:
        provider: "gpt", "claude", "gemini" のいずれか

    Returns:
        LLMClient インスタンス
    """
    provider = provider.lower()
    if provider == "gpt":
        return GPTClient()
    elif provider == "claude":
        return ClaudeClient()
    elif provider == "gemini":
        return GeminiClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gpt', 'claude', or 'gemini'")
