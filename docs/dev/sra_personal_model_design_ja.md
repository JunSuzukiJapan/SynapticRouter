# 個人環境(RTX 3080 / RAM 16GB)向け SRA-LLM 自作・実験設計

本ドキュメントでは、個人のローカル環境（VRAM 8GB / RAM 16GB）において、Synaptic Routing Architecture (SRA) を用いた小規模なLLMを自作し、SRAのモジュール性（Hot-Swap等）を検証するための具体的なモデルサイズとタスク設計をまとめる。

---

## 1. マシンスペックに合わせたモデルサイズの決定

VRAM 8GBの環境でフルスクラッチ学習（AdamWオプティマイザ等を使用）を行う場合、メモリの大部分は「モデルパラメータ」「勾配」「オプティマイザのステート」「アクティベーション（計算の途中経過）」の4つで消費される。

これを考慮すると、安定して学習（バッチサイズも確保）できる現実的なモデルサイズは **約1億〜1.5億（100M〜150M）パラメータ**（GPT-2 Smallと同等規模）となる。

### 推奨するSRAアーキテクチャ構成案
* **`dim` (隠れ層の次元)**: 512
* **`layers` (Transformerの層数)**: 6
* **`num_synapses` (シナプス/エキスパートの数)**: 16
* **`k` (同時に発火するシナプス数)**: 2
* **`syn_hidden` (シナプス内部の次元数)**: 1024〜1536
* **`max_seq_len` (最大系列長)**: 512 〜 1024

**選定の理由**: 
SRAの特長である「総パラメータ数に対して推論時のアクティブパラメータが少ない（スパース計算）」という恩恵により、モデル容量（知識の器）を100M以上に保ちながら、1ステップあたりの計算負荷とメモリ使用量を低く抑えることができる。勾配チェックポイント（Gradient Checkpointing）や FP16/BF16 の混合精度学習を併用すれば、VRAM 8GBでもバッチサイズを確保して効率的な学習が可能である。

---

## 2. ホットスワップ実験に最適なタスク（5ドメイン）

SRAの醍醐味である「特定のシナプスだけが専門化する」現象や、「後から追加・削除（Hot-Swap）しても破局的忘却が起きない」ことを実証するには、**トークンの分布や文法ルールが全く異なる（遠い）ドメイン**を選ぶのが最適である。

以下の5つのタスクを提案する。

### タスク1: 日本語の日常対話（Casual Japanese Chat）
* **内容**: 「こんにちは」「明日の天気は？」といった一般的な日本語の会話。
* **ドメインの特徴**: ひらがな・漢字の一般的なトークン分布。口語表現が中心。モデルのベースとなる「人格」や「基本言語能力」の確認に使う。

### タスク2: Pythonコードスニペットの生成（Python Code Generation）
* **内容**: 与えられた指示に従って数行のPython関数を出力するタスク。
* **ドメインの特徴**: 英語ベースの予約語（`def`, `import`）、厳密なインデント、`_` や `:` などの記号が頻出する。タスク1（日常日本語）とは構造が完全に異なるため、全く別のシナプスが割り当てられる（発火する）ことが期待できる。

### タスク3: 数式（LaTeX）の生成・変換（Math & LaTeX）
* **内容**: 「ピタゴラスの定理を書いて」という入力に対して、`a^2 + b^2 = c^2` といったLaTeX記法の数式を出力するタスク。
* **ドメインの特徴**: `\frac`, `\sum`, `_`, `{}` など、特殊な記号の組み合わせが連続する。自然言語ともコードとも異なる、純粋な「論理・数式表現」のドメインである。

### タスク4: 構造化データ（JSON / XML）のフォーマット（Data Structuring）
* **内容**: テキストから人名や年齢を抽出し、正しいJSON形式で出力するタスク。
* **ドメインの特徴**: `{ "name": "...", "age": ... }` のようなキーとバリューの厳密な構造。これも日常会話からは非常に遠く、特定のシナプスが「JSONのカッコを閉じる能力」として専門化しやすい。

### タスク5: 英語の感情分析・分類（English Sentiment Analysis）
* **内容**: 英語の短いレビュー文を読んで、`Positive` または `Negative` に分類するタスク。
* **ドメインの特徴**: 日本語ドメインとは完全に異なる「英語トークン」を使用し、かつ生成ではなく「分類（出力が固定の単語）」というタスク形式である。

---

## 3. このタスク構成での実験（Hot-Swap）の進め方

この5つのタスクをSRAで学習させると、以下のような興味深い実験結果が得られる。

1. **ベース学習とシナプスの分化確認**
   まずは5つのタスクを混ぜて学習させる。学習後、ルーターのヒートマップを確認すると、「コードを書く時はシナプスAとBが発火」「日本語を話すときはシナプスCとDが発火」というように、**明確にシナプスの使用先が分かれている**ことが観察できる。
2. **削除（Unplug）の実験**
   例えば、Pythonコード専用に発火しているシナプスを `pop_synapses()` で強制的に削除する。すると、**日本語の会話能力は全く無傷のまま、コードを書く能力だけが完全に消滅**（文法が崩壊）する。「特定の能力だけを手術のように取り除ける」ことが確認できる。
3. **追加（Plug-in）の実験**
   後から新しいドメイン（例：フランス語翻訳など）を学習させる際、既存のシナプスを「凍結（Freeze）」して新しいシナプスを追加する。すると、過去の5タスクの能力を一切忘れることなく（破局的忘却ゼロ）、新しい能力を獲得できる。

---

## 4. 下準備（環境構築とデータ準備）

### 4.1 Ubuntu環境のセットアップ
Ubuntu (標準的な 22.04 LTS 等) には、深層学習に必要なGPUドライバやライブラリが標準では入っていない場合がある。

1. **NVIDIA ドライバと CUDA Toolkit のインストール**
   RTX 3080 を使用するため、NVIDIAのプロプライエタリドライバを導入する。
   ```bash
   sudo apt update
   sudo apt install -y nvidia-driver-535 nvidia-utils-535
   # インストール後、OSを再起動し、以下のコマンドでGPUが認識されているか確認する
   nvidia-smi
   ```

2. **Python 環境 (Miniconda) の構築**
   OS標準のPython環境を汚さないために、仮想環境を利用する。
   ```bash
   # Minicondaのダウンロードとインストール
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # ターミナルを再起動後、SRA用の環境を作成
   conda create -n sra_env python=3.10 -y
   conda activate sra_env
   ```

3. **PyTorch と必須ライブラリのインストール**
   CUDA環境に対応した PyTorch をインストールする。
   ```bash
   # CUDA 11.8 用のPyTorchインストール例（利用環境のCUDAバージョンに合わせる）
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # SRAの実装・学習データ取得に必要なパッケージ
   pip install datasets sentencepiece transformers matplotlib seaborn
   ```

### 4.2 学習データの用意の仕方
提案した5つのタスクに対応するデータセットは、Hugging Face の `datasets` ライブラリを使って動的に取得するのが最も効率的である（事前の巨大なダウンロードが不要になる）。

```python
from datasets import load_dataset, interleave_datasets

# タスク1: 日本語対話 (例: OASST2)
# ※実際にはフィルタリング関数で日本語のみに絞り込む
ds_chat = load_dataset("OpenAssistant/oasst2", split="train")

# タスク2: Pythonコード (例: Code Alpaca)
ds_code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

# タスク3: 数式/LaTeX (例: MathInstruct)
ds_math = load_dataset("TIGER-Lab/MathInstruct", split="train")

# タスク4: 構造化データ (JSON)
# ※データセット内にJSONタスクが含まれるものを利用するか、合成データセットを使う
ds_json = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

# タスク5: 英語感情分類 (例: IMDBレビュー)
ds_sentiment = load_dataset("imdb", split="train")
```

**データの混ぜ方（マルチタスク学習）**:
学習時にこれら5つの異なるドメインのデータセットからランダムにバッチを抽出して混合する（Interleaving）。
これにより、モデルは常に多種多様なドメインに触れ続け、シナプスの専門化（ルーティングの分岐）が促される。

```python
# 5つのデータセットを均等な確率 (各20%) で混ぜ合わせる
mixed_dataset = interleave_datasets(
    [ds_chat, ds_code, ds_math, ds_json, ds_sentiment],
    probabilities=[0.2, 0.2, 0.2, 0.2, 0.2]
)

# 実際の学習ループでは、この mixed_dataset からバッチを取り出す
for batch in mixed_dataset:
    # 学習処理...
    pass
```
