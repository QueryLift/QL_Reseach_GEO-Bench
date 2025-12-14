# GEO-bench Emulator

GEO (Generative Engine Optimization) 論文に基づくWeb検索エミュレータ。

**Reference**: Aggarwal et al., "GEO: Generative Engine Optimization", KDD 2024

## セットアップ

### 1. Chrome / Chromium のインストール

Seleniumを使用したWebコンテンツ取得にChromeが必要です。

**macOS:**
```bash
# Homebrew を使用
brew install --cask google-chrome

# または Chromium
brew install --cask chromium
```

**Ubuntu / Debian:**
```bash
# Google Chrome
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt update
sudo apt install -y google-chrome-stable

# または Chromium（軽量）
sudo apt install -y chromium-browser
```

### 2. Python環境のセットアップ

```bash
# 仮想環境を作成・有効化
python3 -m venv venv
source venv/bin/activate

# 依存パッケージをインストール
pip install openai anthropic google-genai httpx beautifulsoup4 python-dotenv pdfplumber selenium webdriver-manager
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
# 基本実行（出力フォルダ名はタイムスタンプ）
python run_experiment.py

# 出力フォルダ名を指定
python run_experiment.py -o my_experiment

# 質問生成のみ（実験は実行しない）
python run_experiment.py --generate-only

# 生成された質問を表示
python run_experiment.py --show-questions
```

### コマンドライン引数

| 引数 | 説明 |
|------|------|
| `-o, --output-name` | 出力フォルダ名（省略時はタイムスタンプ `YYYYMMDD_HHMMSS`） |
| `--generate-only` | 質問生成のみを実行（実験は実行しない） |
| `--show-questions` | 生成された質問を表示 |

## 処理フロー

各ターゲット × 各質問タイプごとに：

1. **Web検索** - 質問タイプごとに独自のWeb検索を実行してソースURLを取得
2. **コンテンツ取得** - 各URLからHTML/PDFを取得しテキスト抽出（Selenium使用）
3. **回答生成** - without/with を `asyncio.gather` で並列実行

**API呼び出し**: 各ターゲットにつき計9回（検索3回 + 回答生成6回）

## 出力ファイル

```
outputs/{output_name}/
├── config.json                      # 実験設定
├── summary.json                     # 全体サマリー
├── {domain}/
│   ├── summary.json                 # ドメインサマリー
│   └── {provider}/
│       └── {question_type}/
│           └── {target_id}/
│               ├── summary.json     # ターゲットサマリー
│               ├── sources.json     # 取得ソース一覧
│               └── run_{n}/
│                   ├── answer_without.json
│                   ├── answer_with.json
│                   ├── metrics_without.json
│                   └── metrics_with.json
```

### 質問タイプ

| タイプ | 説明 |
|--------|------|
| `vague` | 最も抽象的（二段階抽象化した広いカテゴリー） |
| `experiment` | 中間的（タイトルのテーマで質問） |
| `aligned` | 最も具体的（記事内容に沿った詳細な質問） |

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
