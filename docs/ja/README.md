[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)
Synaptic Routing Architecture (SRA) は、生物の脳（シナプス）の仕組みから着想を得た、新しい動的・スパース（疎）なモジュール型ニューラルネットワーク・アーキテクチャです。
巨大で静的なTransformerに代わり、入力を適切な「シナプス（極小モジュール）」へ動的にルーティングすることで、より効率的な学習と構造的な知能の実現を目指しています。

## 🎮 インタラクティブ・デモ (Jupyter Notebooks)

SRAの「タスクに応じた脳の使い分け」や「堅牢性」を、ブラウザ上で今すぐ体験できるJupyter Notebookを用意しています。Google Colabで数秒で実行できますので、ぜひお試しください。

| # | デモ | 説明 | Colab |
|---|------|------|-------|
| 🟢 1 | [SRA クイックスタート](../../notebooks/01_sra_quickstart.ipynb) | SRAの構造とルーティングの可視化 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart.ipynb) |
| 🔵 2 | [学習とルーティング](../../notebooks/02_learning_and_routing_demo.ipynb) | シングルタスクの学習とルーティングの特化 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo.ipynb) |
| 🔴 3 | [マルチタスク・ルーティング](../../notebooks/03_multitask_routing_demo.ipynb) ✨ | マルチタスク学習とタスクごとのシナプス使い分け | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo.ipynb) |
| 🕹️ 4 | [Decision Transformer](../../notebooks/04_decision_transformer_routing_demo.ipynb) | 強化学習における知覚と行動の分離 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo.ipynb) |
| 🧠 5 | [シナプス破壊実験](../../notebooks/05_lesion_experiment_demo.ipynb) ✨ | 特定シナプス破壊が他タスクに影響しないことを証明 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo.ipynb) |
| 🔌 6 | [ホットスワップ実験](../../notebooks/06_hotswap_experiment_demo.ipynb) | シナプスの動的ホットスワップとRouterの学習限界 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo.ipynb) |
| 👑 7 | [Super Router (Gumbel)](../../notebooks/07_super_router_gumbel_demo.ipynb) | Gumbel-Softmaxによるモデル統合とHard Routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](../../notebooks/08_sra_llm_demo_shakespeare.ipynb) | SRAによるTiny LLMの構築と学習 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare.ipynb) |
| 📚 9 | [マルチドメインLLM](../../notebooks/09_sra_llm_demo_multidomain.ipynb) | 複数ドメイン（Code/Math/Text）の同時学習 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain.ipynb) |
| 💻 10 | [プラグイン・ホットスワップ](../../notebooks/10_hotswap_plugins_demo.ipynb) | Zero-Shot Hot-Swap（破局的忘却ゼロ） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb) |
| 🗑️ 11 | [シナプス削除](../../notebooks/11_synapse_deletion_demo.ipynb) | シナプスの動的削除（pop & clear） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb) |

## 🎯 背景と目的 (Motivation)

近年、AIモデルの巨大化が進む一方で、単一の巨大なネットワーク（モノリシックなモデル）では「計算リソースの増大」や「複数タスク学習時の干渉・破局的忘却（Catastrophic Forgetting）」といった課題が顕在化しています。
SRAはこれらの課題に対し、**「入力に応じて必要な極小モジュール（シナプス）だけを動的に呼び出して組み合わせる」** という疎（Sparse）なアプローチで解決を試みています。これにより、性質の異なる複数のタスクを同一ネットワーク内で干渉させずに学習させ、スケーラビリティと高い学習効率を実現することを目指しています。

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

- [アルゴリズム的推論におけるマルチタスク学習とルーティング分析](./routing_analysis_algorithmic.md)
  - SRAが複数のアルゴリズムタスクを干渉なく同時学習し、タスクの性質に応じて自律的にエキスパート（シナプス）を分離・モジュール化できることを検証したレポートです。
- [異種ドメイン言語モデリングにおけるルーティング分析 (Code / Math / Text)](./routing_analysis_language.md)
  - SRAを用いて文法・語彙の異なるドメイン（コード、数式、自然言語）を同時学習させ、各ドメインごとにシナプスが機能分化（専門化）して推論するメカニズムを検証したレポートです。


## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways](./paper_draft.md)
- [Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion](./paper_hotswap.md)

## 🤝 コントリビュートとライセンス (Contributing & License)

本プロジェクトは初期段階の実験的アーキテクチャです。バグ報告や機能に関する議論、パフォーマンス改善のプルリクエストなどは、IssueやPRにて歓迎いたします！

- **ライセンス**: 本リポジトリは [MIT License](../../LICENSE) のもとで公開されています。詳細については [`LICENSE`](../../LICENSE) ファイルをご参照ください。
