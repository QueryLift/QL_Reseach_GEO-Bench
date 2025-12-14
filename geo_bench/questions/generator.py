"""
Question Generator
==================
LLMを使用した質問生成
"""

from __future__ import annotations

import asyncio
import json
import re

from ..llm.base import LLMClient
from ..types import GeneratedQuestions
from .cache import load_questions_cache, save_questions_cache


class QuestionGenerator:
    """LLMを使用して質問を生成するクラス"""

    def __init__(self, llm: LLMClient, prompt_template: str):
        """
        Args:
            llm: 質問生成に使用するLLMクライアント
            prompt_template: 使用するプロンプトテンプレート
        """
        self.llm = llm
        self.prompt_template = prompt_template

    async def generate(self, title: str, content: str) -> GeneratedQuestions:
        """ターゲットのタイトルと内容から3種類の質問を生成"""
        prompt = self.prompt_template.format(title=title, content=content)
        response = await self.llm.acall_standard(prompt)

        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                questions = json.loads(response)

            return GeneratedQuestions(
                vague=questions.get("vague", ""),
                experiment=questions.get("experiment", ""),
                aligned=questions.get("aligned", ""),
            )
        except json.JSONDecodeError as e:
            print(f"警告: 質問生成のJSONパースに失敗: {e}")
            print(f"レスポンス: {response[:200]}...")
            return GeneratedQuestions(
                vague="OpenAIのAI技術について教えてください。",
                experiment=f"OpenAIの{title}について教えてください。",
                aligned=f"OpenAIの{title}の具体的な内容を詳しく教えてください。",
            )


async def generate_all_questions(
    targets: list[dict],
    llm: LLMClient,
    cache_file: str,
    prompt_template: str,
) -> dict[str, GeneratedQuestions]:
    """
    全ターゲットの質問を生成（キャッシュ利用）

    Args:
        targets: ターゲット情報のリスト（id, title, content を含む）
        llm: LLMクライアント
        cache_file: キャッシュファイルパス
        prompt_template: プロンプトテンプレート

    Returns:
        target_id -> GeneratedQuestions のマッピング
    """
    cache = load_questions_cache(cache_file)
    targets_to_generate = [t for t in targets if t["id"] not in cache]

    if not targets_to_generate:
        print(f"質問キャッシュから{len(targets)}件の質問を読み込みました")
        return cache

    print(f"質問生成: {len(targets_to_generate)}件（キャッシュ済み: {len(cache)}件）")

    generator = QuestionGenerator(llm, prompt_template)

    async def generate_for_target(target: dict) -> tuple[str, GeneratedQuestions]:
        print(f"  質問生成中: {target['title']}")
        questions = await generator.generate(target["title"], target["content"])
        return target["id"], questions

    tasks = [generate_for_target(t) for t in targets_to_generate]
    results = await asyncio.gather(*tasks)

    for target_id, questions in results:
        cache[target_id] = questions

    save_questions_cache(cache_file, cache)
    print(f"質問キャッシュを保存しました: {cache_file}")

    return cache
