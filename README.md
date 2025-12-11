# GEO-bench Emulator

GEO (Generative Engine Optimization) 論文に基づくWeb検索エミュレータ。

**Reference**: Aggarwal et al., "GEO: Generative Engine Optimization", KDD 2024

## セットアップ

```bash
# 仮想環境を作成・有効化
python3 -m venv venv
source venv/bin/activate

# 依存パッケージをインストール
pip install openai httpx beautifulsoup4 python-dotenv
```

`.env` ファイルを作成：

```
OPENAI_API_KEY=sk-your-api-key-here
LLM_RATE_LIMIT_INTERVAL=2.0
```

## 設定

`config.json` を作成：

```json
{
  "question": "質問文",
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

## 実行

```bash
python run_geo_bench.py
```

## 処理フロー

1. **Web検索** - `gpt-5` + `web_search` ツールでソースURLを取得
2. **コンテンツ取得** - 各URLからHTMLを取得しテキスト抽出（並列処理）
3. **回答生成（ターゲットなし）** - `gpt-5` で回答生成、引用メトリクス計算
4. **回答生成（ターゲットあり）** - ソースリスト末尾にターゲットを追加して `gpt-5` で回答生成

**API呼び出し**: 計3回（検索1回 + 回答生成2回）

## 出力ファイル

```
outputs/YYYYMMDD_HHMMSS/
├── config.json          # 実行設定
├── answer_without.json  # 回答（ターゲットなし）
├── answer_with.json     # 回答（全ターゲットあり）
├── metrics_without.csv  # 引用メトリクス（ターゲットなし）
└── metrics_with.csv     # 引用メトリクス（全ターゲットあり）
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
