"""
LLM Base Client
===============
LLMクライアントの抽象基底クラス
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime


class LLMClient(ABC):
    """LLMクライアントの抽象基底クラス"""

    # サブクラスでオーバーライドするデフォルトのレートリミット間隔
    DEFAULT_RATE_LIMIT_INTERVAL = 3.3

    def __init__(self):
        # レートリミット設定（環境変数で上書き可能）
        env_interval = os.getenv("LLM_RATE_LIMIT_INTERVAL")
        if env_interval:
            self.rate_limit_interval = float(env_interval)
        else:
            self.rate_limit_interval = self.DEFAULT_RATE_LIMIT_INTERVAL
        print(f"[{self.__class__.__name__}] rate_limit_interval: {self.rate_limit_interval}s")
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
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] LLM API呼び出し (interval: {self.rate_limit_interval}s)")

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
