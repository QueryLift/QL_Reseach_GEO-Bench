"""
Markdown Processing
===================
Markdownをプレーンテキストに変換
"""

from __future__ import annotations

import re


def _citation_to_letter(match: re.Match) -> str:
    """[1] → [[a]], [2] → [[b]] などに変換"""
    num = int(match.group(1))
    if num < 1:
        return match.group(0)  # 0以下はそのまま
    # 1→a, 2→b, ..., 26→z, 27→aa, 28→ab, ...
    result = ""
    while num > 0:
        num -= 1
        result = chr(ord('a') + (num % 26)) + result
        num //= 26
    return f"[[{result}]]"


def strip_markdown(content: str) -> str:
    """
    Markdownテキストをプレーンテキストに変換

    Args:
        content: Markdown形式のテキスト

    Returns:
        プレーンテキスト
    """
    # 引用番号 [1], [2] などを [[a]], [[b]] に変換（Search Results:の番号との混同を防ぐ）
    content = re.sub(r'\[(\d+)\]', _citation_to_letter, content)
    # 見出し（# ## ### など）の記号のみ除去（改行は保持）
    content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)
    # 太字・斜体（** __ * _）を除去
    content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
    content = re.sub(r'__(.+?)__', r'\1', content)
    content = re.sub(r'\*(.+?)\*', r'\1', content)
    content = re.sub(r'_(.+?)_', r'\1', content)
    # リンク [text](url) を text に変換
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    # インラインコード ` を除去
    content = re.sub(r'`([^`]+)`', r'\1', content)
    # 水平線 --- *** ___ を除去
    content = re.sub(r'^[-*_]{3,}\s*$', '', content, flags=re.MULTILINE)
    # リスト記号（- * + 1. など）を除去
    content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
    # 引用 > を除去
    content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
    # 連続する空行を1つに
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()
