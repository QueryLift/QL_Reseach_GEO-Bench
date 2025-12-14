#!/usr/bin/env python3
"""
GEO-bench 実験スクリプト

複数ターゲット・複数プロバイダーでの実験を実行し、
結果を階層的に集計・出力する。

Usage:
    python run_experiment.py -c configs/jimin-gemini-config.json
    python run_experiment.py -c configs/jimin-gemini-config.json -o my_experiment
    python run_experiment.py -c configs/jimin-gemini-config.json -q
    python run_experiment.py -c configs/jimin-gemini-config.json --show-questions
"""

from geo_bench.cli import run

if __name__ == "__main__":
    run()
