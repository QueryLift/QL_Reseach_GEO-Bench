"""
Question Cache
==============
質問キャッシュの読み書き
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import GeneratedQuestions


def load_questions_cache(cache_file: str) -> dict[str, GeneratedQuestions]:
    """質問キャッシュファイルを読み込む"""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
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
    """質問キャッシュをファイルに保存"""
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
