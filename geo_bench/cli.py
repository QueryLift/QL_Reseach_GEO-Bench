"""
CLI Module
==========
コマンドラインインターフェース
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from .config import load_config
from .llm import create_llm_client
from .questions import (
    QUESTION_GENERATION_PROMPT_FOR_JIMIN,
    QUESTION_GENERATION_PROMPT_FOR_OPENAI,
    generate_all_questions,
)
from .runner import ExperimentRunner
from .targets import discover_targets, get_target_domains


# プロンプトタイプのマッピング
PROMPT_TYPES = {
    "openai": QUESTION_GENERATION_PROMPT_FOR_OPENAI,
    "jimin": QUESTION_GENERATION_PROMPT_FOR_JIMIN,
}


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数をパース

    Returns:
        argparse.Namespace: パース結果
    """
    parser = argparse.ArgumentParser(
        description="GEO-bench 実験スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="設定ファイルのパス（必須）",
    )
    parser.add_argument(
        "-q", "--generate-questions-only",
        action="store_true",
        help="質問生成のみを実行（実験は実行しない）",
    )
    parser.add_argument(
        "--show-questions",
        action="store_true",
        help="生成された質問を表示",
    )
    parser.add_argument(
        "-o", "--output-name",
        type=str,
        default=None,
        help="出力フォルダ名（指定しない場合はタイムスタンプ）",
    )
    return parser.parse_args()


async def main():
    """メイン関数"""
    # 環境変数を読み込む
    load_dotenv()

    args = parse_args()

    # 設定ファイルを読み込む
    print(f"設定ファイル: {args.config}")
    config = load_config(args.config)

    # ドメインを自動検出
    target_domains = get_target_domains(config["targets_dir"])
    if not target_domains:
        print(f"エラー: ターゲットディレクトリが空です: {config['targets_dir']}")
        sys.exit(1)

    print(f"検出されたドメイン: {target_domains}")

    # ターゲットを検出
    targets = discover_targets(config["targets_dir"], target_domains)

    if not targets:
        print("エラー: ターゲットが見つかりません")
        sys.exit(1)

    print(f"検出されたターゲット: {len(targets)}件")
    for domain in target_domains:
        count = sum(1 for t in targets if t["domain"] == domain)
        print(f"  {domain}: {count}件")

    # 質問生成用のLLMを取得（最初のプロバイダーを使用）
    question_llm = create_llm_client(config["providers"][0])
    print(f"\n質問生成用LLM: {question_llm.name}")

    # プロンプトタイプを取得
    prompt_type = config.get("prompt_type", "openai")
    prompt_template = PROMPT_TYPES.get(prompt_type)
    if prompt_template is None:
        print(f"エラー: 不明なプロンプトタイプ: {prompt_type}")
        print(f"利用可能なタイプ: {list(PROMPT_TYPES.keys())}")
        sys.exit(1)
    print(f"質問生成プロンプト: {prompt_type}")

    # 質問を生成（キャッシュ利用）
    questions = await generate_all_questions(
        targets=[{"id": t["id"], "title": t["title"], "content": t["content"]} for t in targets],
        llm=question_llm,
        cache_file=config["questions_cache_file"],
        prompt_template=prompt_template,
    )

    # 質問表示オプション
    if args.show_questions:
        print("\n" + "="*60)
        print("生成された質問")
        print("="*60)
        for target in targets:
            q = questions.get(target["id"])
            if q:
                print(f"\n[{target['title']}]")
                print(f"  vague: {q['vague']}")
                print(f"  experiment: {q['experiment']}")
                print(f"  aligned: {q['aligned']}")

    # 質問生成のみの場合はここで終了
    if args.generate_questions_only:
        print(f"\n質問生成完了: {config['questions_cache_file']}")
        return

    # 実験を実行
    runner = ExperimentRunner(
        providers=config["providers"],
        num_runs=config["num_runs"],
        max_sources=config["max_sources"],
        output_base_dir=config["output_dir"],
        primary_domains=config.get("primary_domains", []),
    )

    await runner.run_experiment(
        targets,
        questions,
        output_name=args.output_name,
        questions_cache_file=config["questions_cache_file"],
    )


def run():
    """エントリーポイント"""
    asyncio.run(main())


if __name__ == "__main__":
    run()
