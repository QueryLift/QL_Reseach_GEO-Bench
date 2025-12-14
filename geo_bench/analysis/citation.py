"""
Citation Analyzer
=================
引用分析（GEO論文 Section 2.2.1）
"""

from __future__ import annotations

import re

from ..types import Citation, CitationMetrics, SourceContent


class CitationAnalyzer:
    """
    引用を分析するクラス (GEO論文 Section 2.2.1)

    メトリクス:
    - Word Count: 引用に関連する文の正規化ワードカウント
    - Position-Adjusted Word Count: 位置による重み付けワードカウント
    """

    # 引用パターン: [1], [2], [1][2][3] など
    CITATION_PATTERN = re.compile(r'\[(\d+)\]')

    def analyze(self, response: str, sources: list[SourceContent]) -> dict[int, CitationMetrics]:
        """
        レスポンスから引用メトリクスを計算

        Args:
            response: LLMのレスポンステキスト
            sources: ソースのリスト（インデックスはURLと対応）

        Returns:
            インデックス -> CitationMetrics のマッピング
        """
        # 文に分割
        sentences = self._split_into_sentences(response)
        total_word_count = sum(len(s.split()) for s in sentences)

        if total_word_count == 0:
            return {}

        # 引用された Citation のみオンデマンドで作成
        citations: dict[int, Citation] = {}
        num_sources = len(sources)

        # 各文を解析し、引用ごとに分子を累積
        # - Imp_wc 分子: Σ|s| （引用された文のワード数合計）
        # - Imp_pwc 分子: Σ|s|·e^{-pos/|S|} （位置重み付きワード数合計）
        num_sentences = len(sentences)
        for pos, sentence in enumerate(sentences):
            cited_indices = self._extract_citations(sentence)
            if not cited_indices:
                continue

            word_count = len(sentence.split())  # |s|
            share = word_count / len(cited_indices)  # 複数引用時は均等分割
            weight = pow(2.718281828, -pos / num_sentences)  # e^{-pos/|S|}

            for idx in cited_indices:
                if idx < 1 or idx > num_sources:
                    continue  # 無効なインデックスはスキップ
                if idx not in citations:
                    citations[idx] = Citation()
                cit = citations[idx]
                if cit.first_pos == -1:
                    cit.first_pos = pos
                cit.sentences.append(sentence)
                cit.word_count += share
                cit.position_sum += share * weight

        # 分母で正規化してメトリクスを生成
        # - Imp_wc = 分子 / Σ|s_r| * 100 (%)
        # - Imp_pwc = 分子 / Σ|s_r| * 100 (%)
        return {
            idx: CitationMetrics(
                imp_wc=cit.word_count / total_word_count * 100,
                imp_pwc=cit.position_sum / total_word_count * 100,
                citation_frequency=len(cit.sentences),
                first_citation_position=cit.first_pos,
            )
            for idx, cit in citations.items()
        }

    def _split_into_sentences(self, text: str) -> list[str]:
        """テキストを文に分割"""
        # 簡易的な文分割（。.!? で分割）
        sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_citations(self, sentence: str) -> list[int]:
        """文から引用インデックスを抽出"""
        matches = self.CITATION_PATTERN.findall(sentence)
        return [int(m) for m in matches]
