"""
Target Discovery
================
ターゲットファイルの検出と読み込み
"""

import re
from pathlib import Path

from ..content import strip_markdown
from ..types import TargetConfig


# ターゲットコンテンツの最大文字数（Webコンテンツと同じ制限）
MAX_TARGET_CONTENT_LENGTH = 8000


def load_target_file(file_path: str) -> str:
    """
    ターゲットファイルを読み込み、Markdownを除去

    Args:
        file_path: ファイルパス

    Returns:
        str: Markdown記法を除去したプレーンテキスト（最大8000文字）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = strip_markdown(content)
    if len(content) > MAX_TARGET_CONTENT_LENGTH:
        content = content[:MAX_TARGET_CONTENT_LENGTH] + "..."
    return content


def extract_h1_title(file_path: str) -> str:
    """
    MDファイルから最初のH1タグの内容を抽出

    Args:
        file_path: ファイルパス

    Returns:
        str: H1タグの内容、見つからない場合はファイル名
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = re.match(r'^#\s+(.+)$', line)
            if match:
                return match.group(1).strip()
    return Path(file_path).stem


def discover_targets(targets_dir: str, domains: list[str]) -> list[TargetConfig]:
    """
    指定ドメインのターゲットファイルを検出

    Args:
        targets_dir: ターゲットディレクトリのパス
        domains: 検索対象のドメインリスト

    Returns:
        list[TargetConfig]: 検出されたターゲットの設定リスト
    """
    targets = []
    base_path = Path(targets_dir)

    for domain in domains:
        domain_path = base_path / domain
        if not domain_path.exists():
            print(f"警告: ドメインディレクトリが見つかりません: {domain_path}")
            continue

        for md_file in sorted(domain_path.glob("*.md")):
            target_id = md_file.stem
            title = extract_h1_title(str(md_file))
            content = load_target_file(str(md_file))
            targets.append({
                "id": target_id,
                "file": str(md_file),
                "url": f"https://TARGET",
                "domain": domain,
                "title": title,
                "content": content,
            })

    return targets


def get_target_domains(targets_dir: str) -> list[str]:
    """
    ターゲットディレクトリ内の全ドメイン（サブディレクトリ）を検出

    Args:
        targets_dir: ターゲットディレクトリのパス

    Returns:
        list[str]: ドメイン名のリスト
    """
    base_path = Path(targets_dir)
    if not base_path.exists():
        return []
    return [d.name for d in base_path.iterdir() if d.is_dir() and not d.name.startswith(".")]


def sanitize_folder_name(name: str) -> str:
    """
    フォルダ名に使用できない文字を置換

    Args:
        name: 元の名前

    Returns:
        str: サニタイズされた名前
    """
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '_')
    if len(name) > 50:
        name = name[:50]
    return name
