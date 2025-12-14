"""
質問生成モジュール

ターゲット記事のタイトルと内容から3種類の質問（vague, experiment, aligned）を
LLMを使用して自動生成する。生成結果はJSONファイルにキャッシュされる。

質問タイプ:
- vague: 最も抽象的（二段階抽象化）
- experiment: 中間的（タイトルのテーマで質問）
- aligned: 最も具体的（記事内容に沿った詳細な質問）
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from llm_clients import LLMClient


# =============================================================================
# Data Types
# =============================================================================

class GeneratedQuestions(TypedDict):
    """
    生成された3種類の質問を格納する型

    Attributes:
        vague: 最も抽象的な質問（二段階抽象化）
        experiment: 中間的な質問（タイトルのテーマで質問）
        aligned: 最も具体的な質問（記事内容に沿った詳細な質問）
    """
    vague: str
    experiment: str
    aligned: str


# =============================================================================
# Constants
# =============================================================================

# 質問タイプのリスト
QUESTION_TYPES = ["vague", "experiment", "aligned"]

# 質問生成プロンプト（OpenAI用）
QUESTION_GENERATION_PROMPT_FOR_OPENAI = """あなたはOpenAIの技術記事に関する質問を生成するアシスタントです。
以下のターゲット記事のタイトルと内容に基づいて、3種類の質問を日本語で生成してください。

【重要】すべての質問は「OpenAIがどのような情報を発信しているか」を聞く形式にしてください。

【ターゲット記事】
タイトル: {title}
内容: {content}

【生成する質問の種類】

1. vague（最も抽象的な質問）:
ターゲットのタイトルを「二段階」抽象化した、広いカテゴリーについて総合的に知りたいという質問。
具体的なテーマ名は出さず、上位概念で聞く。
例:
- タイトル「GPT-4 Turbo」→「OpenAIのAI技術について教えてください。」
- タイトル「Function Calling」→「OpenAIのAPI機能について教えてください。」
- タイトル「Whisper API」→「OpenAIの音声技術について教えてください。」
- タイトル「DALL-E 3」→「OpenAIの画像生成技術について教えてください。」

2. experiment（中間的な質問）:
ターゲットのタイトルのテーマについて、全体像や基本的な特徴を聞く質問。
タイトルのテーマ名は使うが、記事の詳細には踏み込まない。
例:
- タイトル「GPT-4 Turbo」→「OpenAIのGPT-4 Turboについて教えてください。」
- タイトル「Function Calling」→「OpenAIのFunction Callingとは何ですか？」

3. aligned（最も具体的な質問）:
ターゲットの記事内容に完全に沿った、具体的で詳細な質問。
記事に登場する具体的な機能、数値、技術用語などを含めて深く聞く。
例: 「OpenAIのGPT-4 TurboのJSON modeはどのように動作し、どのようなユースケースで活用できますか？また、従来のGPT-4と比較してコンテキスト長やレスポンス速度はどの程度改善されていますか？」

【出力形式】
以下のJSON形式で出力してください。余計な説明は不要です。
{{
  "vague": "最も抽象的な質問文",
  "experiment": "中間的な質問文",
  "aligned": "最も具体的な質問文"
}}"""

# 質問生成プロンプト（自民党政策用）
QUESTION_GENERATION_PROMPT_FOR_JIMIN = """あなたは政策に関する質問を生成するアシスタントです。
以下のターゲット記事のタイトルと内容に基づいて、3種類の質問を日本語で生成してください。

【重要】すべての質問は「自民党がどのような政策を発信しているか」を聞く形式にしてください。

【ターゲット記事】
タイトル: {title}
内容: {content}

【生成する質問の種類】

1. vague（最も抽象的な質問）:
ターゲットのタイトルを「二段階」抽象化した、広いカテゴリーについて総合的に知りたいという質問。
具体的なテーマ名は出さず、上位概念で聞く。
例:
- タイトル「子ども・子育て支援金」→「自民党の社会保障政策について教えてください。」
- タイトル「防衛費増額」→「自民党の安全保障政策について教えてください。」
- タイトル「デジタル田園都市国家構想」→「自民党の地方創生政策について教えてください。」
- タイトル「GX推進法」→「自民党の環境・エネルギー政策について教えてください。」

2. experiment（中間的な質問）:
ターゲットのタイトルのテーマについて、全体像や基本的な特徴を聞く質問。
タイトルのテーマ名は使うが、記事の詳細には踏み込まない。
例:
- タイトル「子ども・子育て支援金」→「自民党の子ども・子育て支援金について教えてください。」
- タイトル「防衛費増額」→「自民党の防衛費増額とは何ですか？」

3. aligned（最も具体的な質問）:
ターゲットの記事内容に完全に沿った、具体的で詳細な質問。
記事に登場する具体的な政策、数値、制度などを含めて深く聞く。
例: 「自民党の子ども・子育て支援金の財源はどのように確保され、具体的にどのような給付が予定されていますか？また、従来の子育て支援策と比較してどのような点が拡充されていますか？」

【出力形式】
以下のJSON形式で出力してください。余計な説明は不要です。
{{
  "vague": "最も抽象的な質問文",
  "experiment": "中間的な質問文",
  "aligned": "最も具体的な質問文"
}}"""

# =============================================================================
# QuestionGenerator Class
# =============================================================================

class QuestionGenerator:
    """
    LLMを使用して質問を生成するクラス

    ターゲット記事のタイトルと内容から、3種類の抽象度レベルの質問を生成する。

    Attributes:
        llm: LLMクライアントインスタンス
        prompt_template: 使用するプロンプトテンプレート
    """

    def __init__(self, llm: LLMClient, prompt_template: str):
        """
        QuestionGeneratorを初期化

        Args:
            llm: 質問生成に使用するLLMクライアント
            prompt_template: 使用するプロンプトテンプレート（必須）
        """
        self.llm = llm
        self.prompt_template = prompt_template

    async def generate(self, title: str, content: str) -> GeneratedQuestions:
        """
        ターゲットのタイトルと内容から3種類の質問を生成

        Args:
            title: ターゲット記事のタイトル（H1タグの内容）
            content: ターゲット記事の本文内容

        Returns:
            GeneratedQuestions: vague, experiment, aligned の3種類の質問を含む辞書

        Note:
            LLM呼び出しが失敗した場合やJSONパースに失敗した場合は、
            デフォルトの質問文を返す。
        """
        prompt = self.prompt_template.format(title=title, content=content)
        response = await self.llm.acall_standard(prompt)

        # JSONをパース
        try:
            # JSON部分を抽出（```json ... ``` で囲まれている場合に対応）
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
            # フォールバック: デフォルトの質問を返す
            return GeneratedQuestions(
                vague="OpenAIのAI技術について教えてください。",
                experiment=f"OpenAIの{title}について教えてください。",
                aligned=f"OpenAIの{title}の具体的な内容を詳しく教えてください。",
            )


# =============================================================================
# Cache Functions
# =============================================================================

def load_questions_cache(cache_file: str) -> dict[str, GeneratedQuestions]:
    """
    質問キャッシュファイルを読み込む

    Args:
        cache_file: キャッシュファイルのパス

    Returns:
        target_id -> GeneratedQuestions のマッピング辞書
        ファイルが存在しないか読み込みに失敗した場合は空辞書を返す
    """
    cache_path = Path(cache_file)
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # dict[target_id, GeneratedQuestions]
        return {
            target_id: GeneratedQuestions(
                vague=q["vague"],
                experiment=q["experiment"],
                aligned=q["aligned"],
            )
            for target_id, q in data.items()
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"警告: キャッシュファイルの読み込みに失敗: {e}")
        return {}


def save_questions_cache(cache_file: str, cache: dict[str, GeneratedQuestions]):
    """
    質問キャッシュをファイルに保存

    Args:
        cache_file: 保存先ファイルパス
        cache: target_id -> GeneratedQuestions のマッピング辞書
    """
    data = {
        target_id: {
            "vague": q["vague"],
            "experiment": q["experiment"],
            "aligned": q["aligned"],
        }
        for target_id, q in cache.items()
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# Batch Generation Function
# =============================================================================

async def generate_all_questions(
    targets: list[dict],
    llm: LLMClient,
    cache_file: str,
    prompt_template: str,
) -> dict[str, GeneratedQuestions]:
    """
    全ターゲットの質問を生成（キャッシュ利用）

    既にキャッシュに存在するターゲットはスキップし、
    新規ターゲットのみ質問を生成する。

    Args:
        targets: ターゲット情報のリスト。各要素は以下のキーを含む:
            - id: ターゲットID
            - title: タイトル
            - content: 本文内容
        llm: LLMクライアント
        cache_file: キャッシュファイルパス
        prompt_template: 使用するプロンプトテンプレート（必須）

    Returns:
        target_id -> GeneratedQuestions のマッピング辞書
    """
    import asyncio

    # キャッシュを読み込む
    cache = load_questions_cache(cache_file)

    # 未生成のターゲットを抽出
    targets_to_generate = [t for t in targets if t["id"] not in cache]

    if not targets_to_generate:
        print(f"質問キャッシュから{len(targets)}件の質問を読み込みました")
        return cache

    print(f"質問生成: {len(targets_to_generate)}件（キャッシュ済み: {len(cache)}件）")

    generator = QuestionGenerator(llm, prompt_template)

    # 並列で質問を生成
    async def generate_for_target(target: dict) -> tuple[str, GeneratedQuestions]:
        print(f"  質問生成中: {target['title']}")
        questions = await generator.generate(target["title"], target["content"])
        return target["id"], questions

    tasks = [generate_for_target(t) for t in targets_to_generate]
    results = await asyncio.gather(*tasks)

    # キャッシュに追加
    for target_id, questions in results:
        cache[target_id] = questions

    # キャッシュを保存
    save_questions_cache(cache_file, cache)
    print(f"質問キャッシュを保存しました: {cache_file}")

    return cache
