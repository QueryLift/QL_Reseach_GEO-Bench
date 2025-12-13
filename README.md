# GEO-bench Emulator

GEO (Generative Engine Optimization) 論文に基づくWeb検索エミュレータ。

**Reference**: Aggarwal et al., "GEO: Generative Engine Optimization", KDD 2024

## セットアップ

```bash
# 仮想環境を作成・有効化
python3 -m venv venv
source venv/bin/activate

# 依存パッケージをインストール
pip install openai anthropic google-genai httpx beautifulsoup4 python-dotenv pdfplumber
```

`.env` ファイルを作成（使用するプロバイダーに応じて設定）：

```
# GPT (OpenAI) を使用する場合
OPENAI_API_KEY=sk-your-api-key-here

# Claude (Anthropic) を使用する場合
ANTHROPIC_API_KEY=sk-ant-your-api-key-here

# Gemini (Google) を使用する場合
GOOGLE_API_KEY=your-google-api-key-here

# レートリミット（オプション）
LLM_RATE_LIMIT_INTERVAL=2.0
```

## 設定

`config.json` を作成：

```json
{
  "question": "質問文",
  "provider": "gpt",
  "targets": [
    {
      "id": "target_1",
      "file": "target1.md",
      "url": "https://example.com/page1"
    }
  ],
  "output_dir": "outputs",
  "max_sources": 5
}
```

### プロバイダー

| provider | モデル | Web検索方式 |
|----------|--------|-------------|
| `gpt` | gpt-5 | web_search ツール |
| `claude` | claude-sonnet-4-5-20250929 | web_search_20250305 |
| `gemini` | gemini-2.5-flash | Google検索グラウンディング |

## 実行

```bash
python run_geo_bench.py
```

## 処理フロー

1. **Web検索** - 選択したプロバイダーのWeb検索機能でソースURLを取得
2. **コンテンツ取得** - 各URLからHTML/PDFを取得しテキスト抽出（並列処理）
3. **回答生成** - without/with を `asyncio.gather` で並列実行

**API呼び出し**: 計3回（検索1回 + 回答生成2回並列）

## 出力ファイル

```
outputs/YYYYMMDD_HHMMSS/
├── config.json          # 実行設定
├── sources.json         # 全ソース+ターゲット（cited_without/cited_with両方）
├── answer_without.json  # 回答のみ
├── answer_with.json     # 回答のみ（ターゲット情報含む）
├── metrics_without.csv  # 引用メトリクス（citedのみ）
└── metrics_with.csv     # 引用メトリクス（citedのみ）
```

## メトリクス

GEO論文 Section 2.2.1 に基づく：

| メトリクス | 論文 | 説明 |
|-----------|------|------|
| imp_wc | 式(2) Imp_wc | 回答に占めるワード数の割合 (%) |
| imp_pwc | 式(3) Imp_pwc | 位置重み付きワードカウント (%) |
| citation_frequency | - | 引用回数 |
| first_position | - | 最初に引用された位置 |

## 参考文献

- [GEO論文 arXiv](https://arxiv.org/abs/2311.09735)
