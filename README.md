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
pip install openai anthropic google-genai httpx beautifulsoup4 python-dotenv pdfplumber selenium webdriver-manager psutil
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

設定ファイル（JSON）を作成します。`configs/` フォルダに以下のサンプルが用意されています：

| ファイル | ターゲット | プロバイダー |
|----------|-----------|-------------|
| `configs/openai-gpt-config.json` | OpenAI | GPT |
| `configs/openai-gemini-config.json` | OpenAI | Gemini |
| `configs/jimin-gpt-config.json` | Jimin | GPT |
| `configs/jimin-gemini-config.json` | Jimin | Gemini |

```json
{
  "providers": ["gemini"],
  "num_runs": 2,
  "max_sources": 5,
  "targets_dir": "jimin_targets",
  "output_dir": "outputs",
  "questions_cache_file": "jimin-questions.json",
  "primary_domains": ["jimin.jp"],
  "prompt_type": "jimin"
}
```

### 設定項目（すべて必須）

| 項目 | 説明 |
|------|------|
| `providers` | 使用するプロバイダーのリスト |
| `num_runs` | 各ターゲットの繰り返し回数 |
| `max_sources` | Web検索で取得するソース数 |
| `targets_dir` | ターゲットファイルのディレクトリ |
| `output_dir` | 出力ディレクトリ |
| `questions_cache_file` | 質問キャッシュファイル |
| `primary_domains` | 一次情報源と判定するドメイン |
| `prompt_type` | 質問生成プロンプトタイプ（`"openai"` or `"jimin"`） |

### プロバイダー

| provider | モデル | Web検索方式 |
|----------|--------|-------------|
| `gpt` | gpt-5 | web_search ツール |
| `claude` | claude-sonnet-4-5-20250929 | web_search_20250305 |
| `gemini` | gemini-2.5-flash | Google検索グラウンディング |

## 実行

```bash
# 基本的な実行（モジュールとして実行）
python -m geo_bench.cli -c configs/test-config.json

# Jimin × Gemini で実行
python -m geo_bench.cli -c configs/jimin-gemini-config.json

# Jimin × GPT で実行
python -m geo_bench.cli -c configs/jimin-gpt-config.json

# OpenAI × Gemini で実行
python -m geo_bench.cli -c configs/openai-gemini-config.json

# 出力フォルダ名を指定
python -m geo_bench.cli -c configs/jimin-gemini-config.json -o my_experiment

# 質問生成のみ（実験は実行しない）
python -m geo_bench.cli -c configs/jimin-gemini-config.json -q

# 生成された質問を表示
python -m geo_bench.cli -c configs/jimin-gemini-config.json --show-questions
```

**注意**: 必ず `python -m geo_bench.cli` の形式で実行してください。直接 `python geo_bench/cli.py` で実行すると相対インポートエラーが発生します。

### コマンドライン引数

| 引数 | 説明 |
|------|------|
| `-c, --config` | 設定ファイルのパス（必須） |
| `-o, --output-name` | 出力フォルダ名（省略時はタイムスタンプ `YYYYMMDD_HHMMSS`） |
| `-q, --generate-questions-only` | 質問生成のみを実行（実験は実行しない） |
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

## 実行結果データ

実験の実行結果データは以下のGoogle Driveで公開しています：

- [実行結果データ (Google Drive)](https://drive.google.com/drive/folders/10aZiDYnIZuNKdPqDujumbsonPiPiORVm?usp=sharing)

## 参考文献

- [GEO論文 arXiv](https://arxiv.org/abs/2311.09735)
