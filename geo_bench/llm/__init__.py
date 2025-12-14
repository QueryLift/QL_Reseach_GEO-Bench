"""
LLM Clients
===========
LLMクライアント（GPT, Claude, Gemini）

Usage:
    from geo_bench.llm import create_llm_client, LLMClient

    client = create_llm_client("gemini")
    response = await client.acall_standard("Hello")
"""

from .base import LLMClient
from .claude import ClaudeClient
from .gemini import GeminiClient
from .gpt import GPTClient


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


__all__ = [
    "LLMClient",
    "GPTClient",
    "ClaudeClient",
    "GeminiClient",
    "create_llm_client",
]
