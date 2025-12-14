"""
Targets Module
==============
ターゲットファイルの検出と読み込み

Usage:
    from geo_bench.targets import discover_targets, load_target_file
"""

from .discovery import (
    MAX_TARGET_CONTENT_LENGTH,
    discover_targets,
    extract_h1_title,
    get_target_domains,
    load_target_file,
    sanitize_folder_name,
)

__all__ = [
    "discover_targets",
    "load_target_file",
    "extract_h1_title",
    "get_target_domains",
    "sanitize_folder_name",
    "MAX_TARGET_CONTENT_LENGTH",
]
