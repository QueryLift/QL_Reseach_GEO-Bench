"""
Configuration
=============
実験設定の読み込み・検証
"""

import json
from pathlib import Path
from typing import Any


# 必須フィールド
REQUIRED_FIELDS = [
    "providers",
    "num_runs",
    "max_sources",
    "targets_dir",
    "output_dir",
    "questions_cache_file",
    "primary_domains",
    "prompt_type",
]


def load_config(config_path: str) -> dict[str, Any]:
    """
    設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        dict: 設定辞書

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        ValueError: 必須フィールドが不足している場合
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 必須フィールドの検証
    for field in REQUIRED_FIELDS:
        if field not in config:
            raise ValueError(f"設定ファイルに必須フィールドがありません: {field}")

    return config


def validate_config(config: dict[str, Any]) -> list[str]:
    """
    設定の妥当性を検証

    Args:
        config: 設定辞書

    Returns:
        list[str]: エラーメッセージのリスト（空ならOK）
    """
    errors = []

    # providers の検証
    providers = config.get("providers", [])
    if not providers:
        errors.append("providers は1つ以上必要です")
    for p in providers:
        if p not in ["gpt", "claude", "gemini"]:
            errors.append(f"不明なプロバイダー: {p}")

    # num_runs の検証
    num_runs = config.get("num_runs", 0)
    if num_runs < 1:
        errors.append("num_runs は1以上必要です")

    # max_sources の検証
    max_sources = config.get("max_sources", 0)
    if max_sources < 1:
        errors.append("max_sources は1以上必要です")

    # targets_dir の検証
    targets_dir = config.get("targets_dir", "")
    if targets_dir and not Path(targets_dir).exists():
        errors.append(f"targets_dir が存在しません: {targets_dir}")

    # prompt_type の検証
    prompt_type = config.get("prompt_type", "")
    if prompt_type not in ["openai", "jimin"]:
        errors.append(f"不明な prompt_type: {prompt_type}")

    return errors
