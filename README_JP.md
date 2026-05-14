[English](README.md) | [日本語](README_JP.md) | [中文](./docs/zh/README.md) | [한국어](./docs/ko/README.md) | [Español](./docs/es/README.md) | [Français](./docs/fr/README.md) | [Deutsch](./docs/de/README.md) | [Português](./docs/pt/README.md) | [Русский](./docs/ru/README.md) | [Italiano](./docs/it/README.md) | [العربية](./docs/ar/README.md) | [हिन्दी](./docs/hi/README.md)

# Synaptic Routing Architecture (SRA)
Synaptic Routing Architecture (SRA) は、生物の脳（シナプス）の仕組みから着想を得た、新しい動的・スパース（疎）なモジュール型ニューラルネットワーク・アーキテクチャです。
巨大で静的なTransformerに代わり、入力を適切な「シナプス（極小モジュール）」へ動的にルーティングすることで、より効率的な学習と構造的な知能の実現を目指しています。

## 🎮 インタラクティブ・デモ (Jupyter Notebooks)

SRAの「タスクに応じた脳の使い分け」や「堅牢性」を、ブラウザ上で今すぐ体験できるJupyter Notebookを用意しています。Google Colabで数秒で実行できますので、ぜひお試しください。

| # | デモ | 説明 | Colab |
|---|------|------|-------|
| 🟢 1 | [SRA クイックスタート](./notebooks/01_sra_quickstart.ipynb) | SRAの構造とルーティングの可視化 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart.ipynb) |
| 🔵 2 | [学習とルーティング](./notebooks/02_learning_and_routing_demo.ipynb) | シングルタスクの学習とルーティングの特化 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo.ipynb) |
| 🔴 3 | [マルチタスク・ルーティング](./notebooks/03_multitask_routing_demo.ipynb) ✨ | マルチタスク学習とタスクごとのシナプス使い分け | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo.ipynb) |
| 🕹️ 4 | [Decision Transformer](./notebooks/04_decision_transformer_routing_demo.ipynb) | 強化学習における知覚と行動の分離 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo.ipynb) |
| 🧠 5 | [シナプス破壊実験](./notebooks/05_lesion_experiment_demo.ipynb) ✨ | 特定シナプス破壊が他タスクに影響しないことを証明 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo.ipynb) |
| 🔌 6 | [ホットスワップ実験](./notebooks/06_hotswap_experiment_demo.ipynb) | シナプスの動的ホットスワップとRouterの学習限界 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo.ipynb) |
| 👑 7 | [Super Router (Gumbel)](./notebooks/07_super_router_gumbel_demo.ipynb) | Gumbel-Softmaxによるモデル統合とHard Routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](./notebooks/08_sra_llm_demo_shakespeare.ipynb) | SRAによるTiny LLMの構築と学習 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare.ipynb) |
| 📚 9 | [マルチドメインLLM](./notebooks/09_sra_llm_demo_multidomain.ipynb) | 複数ドメイン（Code/Math/Text）の同時学習 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain.ipynb) |
| 💻 10 | [プラグイン・ホットスワップ](./notebooks/10_hotswap_plugins_demo.ipynb) | Zero-Shot Hot-Swap（破局的忘却ゼロ） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb) |
| 🗑️ 11 | [シナプス削除](./notebooks/11_synapse_deletion_demo.ipynb) | シナプスの動的削除（pop & clear） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb) |

> 📓 各ノートブックの詳細な説明は [`notebooks/README_JP.md`](./notebooks/README_JP.md) をご覧ください。

## 🎯 背景と目的 (Motivation)

近年、AIモデルの巨大化が進む一方で、単一の巨大なネットワーク（モノリシックなモデル）では「計算リソースの増大」や「複数タスク学習時の干渉・破局的忘却（Catastrophic Forgetting）」といった課題が顕在化しています。
SRAはこれらの課題に対し、**「入力に応じて必要な極小モジュール（シナプス）だけを動的に呼び出して組み合わせる」** という疎（Sparse）なアプローチで解決を試みています。これにより、性質の異なる複数のタスクを同一ネットワーク内で干渉させずに学習させ、スケーラビリティと高い学習効率を実現することを目指しています。

## 💡 基本的なアイデア

通常のAIモデル（Transformerなど）は、巨大な１つの「頭脳」ですべての処理を行おうとします。しかしこの方式では、モデルを賢く・大きくするたびに計算の負担が重くなりすぎてしまいます。そこでSRAでは、**「小さな専門家の脳（SRAではこれを『シナプス』と呼びます）」をたくさん用意し、その時々の問題に合わせて必要な専門家だけを呼び出して使う**という仕組みを採用しました。

ここでポイントになるのが、「どの専門家を呼び出すか」を決める仕組みです。SRAには「ルーター（案内役）」が存在し、入力されたデータを見て一番得意そうな専門家を瞬時に選び出します。このルーターは、各専門家が賢くなっていく（学習する）のと同時に「誰を選ぶのが正解か」を自ら学び、自動で最適な振り分けができるように成長していきます。

### 🧠 複数「仮想ニューロン（Cell Assembly）」の並行創発
SRAの巨大なネットワーク空間では、タスクごとに異なるシナプスの組み合わせが動的に選択されます。これは、算術を解くときの「シナプスA+B+Cの組み合わせ」と、翻訳をするときの「シナプスD+E+Fの組み合わせ」が、それぞれ全く異なる**「独立した仮想ニューロン」**としてネットワーク内に同時に複数並立して存在している状態を作り出します。人間の脳が視覚野と運動野のニューロン群（セルアセンブリ）を並行して機能させているのと同じ構造を、AIモデル内で自律的に再現します。

## 🧠 アーキテクチャの概要 (Architecture)

SRAは主に以下の要素で構成されています：

1. **Synapse（シナプスモジュール）**
   - 独立した極小の計算ユニット（小型TransformerやMLP）。
   - 学習を通じて特定の機能やパターン処理に特化・専門化していきます。
2. **Router（ルーター）**
   - 入力トークンに応じて、全シナプスの中から最適なシナプスを `Top-k` 個だけ動的に選択し、スパースな計算を実現します。
3. **Synapse Space（シナプス空間）**
   - 各シナプスはEmbedding空間上に配置され、シナプス間の距離が「機能の類似性」を表すように自己組織化されます。
4. **Local Learning Rule（局所学習ルール）**
   - 誤差逆伝播（バックプロパゲーション）だけでなく、`trace × routing × reward` のような3因子ルール（Hebbian, STDP, 報酬）を用いた局所的な学習を組み合わせ、負荷分散や専門化を促進します。

## 📁 プロジェクト構成 (Directory Structure)

本リポジトリの主な構成は以下の通りです：

- `src/` : SRAモデルのコア実装および各学習・評価スクリプト
  - `sra_gpu_models.py` / `sra_language_models.py` : MoE技術を応用したSRAモデル実装
  - `train_mtl_algo.py` / `train_mtl_lang.py` : マルチタスク学習（アルゴリズム推論・言語モデル）用実行スクリプト
- `docs/` : アーキテクチャの設計判断（ADR）や各種実験・ルーティング分析のレポート
- `data/` : 学習・検証に使用するデータセット（Code, Math, Textなど）
- `tests/` : 各種コンポーネントのテストコード

## 🚀 使い方とコード例 (Usage)

### 必須要件
```bash
pip install torch
```

### 基本的な実行方法
`copy`, `reverse`, `paren`, `addmod` などの構造的なアルゴリズムタスクを用いて、モデルの学習と推論をテストできます。

単一のタスクを実行する場合：
```bash
python src/sra_experiment.py --task reverse --steps 2000
```

すべてのタスク（タスクスイート）を連続して実行する場合：
```bash
python src/sra_experiment.py --task-suite
```

### 他のアーキテクチャとの比較実行
SRAと標準的な `Transformer` や `MLP` との比較を行うためのスクリプトが用意されています。

```bash
# copyタスクで各アーキテクチャを比較実行する
python src/compare_architectures.py --task copy --steps 500
```

## 📊 他アーキテクチャとの比較 (Comparison)

> **📝 追記予定 (To be updated):**
> 現在、標準的なTransformerモデルとの定量的な比較検証を進めています。  
> 初期の検証（少ステップでのコピー・タスク）では、SRAがTransformerよりも早くValidation Lossを下げ、高い精度に到達する（学習効率・Sample Efficiencyが高い）兆候が確認されています。
>
> 今後、ステップごとの速度（スループット）やVRAM効率の改善を含む、詳細なベンチマーク結果をここに追記する予定です。

## 🧪 実験・分析レポート (Experiments & Analysis)

- [アルゴリズム的推論におけるマルチタスク学習とルーティング分析](./docs/routing_analysis_algorithmic.md)
  - SRAが複数のアルゴリズムタスクを干渉なく同時学習し、タスクの性質に応じて自律的にエキスパート（シナプス）を分離・モジュール化できることを検証したレポートです。
- [異種ドメイン言語モデリングにおけるルーティング分析 (Code / Math / Text)](./docs/routing_analysis_language.md)
  - SRAを用いて文法・語彙の異なるドメイン（コード、数式、自然言語）を同時学習させ、各ドメインごとにシナプスが機能分化（専門化）して推論するメカニズムを検証したレポートです。
- [多言語機械翻訳におけるルーティング分析（英・仏・日）とゼロショット汎化](./docs/dev/multilingual_translation_routing_analysis.md)
  - 言語の構文構造（SVOやSOV）に応じてモデルが自律的に翻訳モジュールを切り替える現象や、未学習の言語ペアを翻訳する際に「英語」を中継地点（ピボット言語）として無意識に利用する驚くべき汎化能力について解説しています。
- [Decision Transformer（強化学習）における知覚と方策の完全分離](./docs/dev/decision_transformer_routing_analysis.md)
  - SRAにゲームをプレイさせた結果、環境を見るための「知覚（視覚）」モジュールは全タスクで共有しつつ、どう動くかを決める「方策（脳）」モジュールはタスク（宝探し or 逃亡）ごとに完全に使い分けるという、生命のようなモジュール構造を自律的に獲得したことを示す興味深いレポートです。
- [SRA Encoder-Decoder による実用レベル多言語翻訳の検証](./docs/dev/sra_seq2seq_translation_analysis.md)
  - SRAを Encoder-Decoder 型に拡張し、実コーパス（opus100）を用いた 30,000 ステップの学習で「Merci beaucoup.」「Good morning.」などの実用表現を BLEU=1.0 で翻訳できることを実証したレポートです。Cross-Attention の導入により Decoder-only（BLEU=0）から全体平均 BLEU=0.27 へと飛躍し、FR→EN 方向では BLEU=0.56 という実用に迫る精度を達成しました。


## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: 動的モジュール経路による破局的忘却の克服](docs/ja/paper_draft.md)
- [Hotswappable LLM: SRAによるZero-Shotモジュール合成と外科的知識削除](docs/ja/paper_hotswap.md)

## 🤝 コントリビュートとライセンス (Contributing & License)

本プロジェクトは初期段階の実験的アーキテクチャです。バグ報告や機能に関する議論、パフォーマンス改善のプルリクエストなどは、IssueやPRにて歓迎いたします！

- **ライセンス**: 本リポジトリは [MIT License](./LICENSE) のもとで公開されています。詳細については [`LICENSE`](./LICENSE) ファイルをご参照ください。
