# Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways

**Jun Suzuki**, Independent Researcher

## Abstract
人間の脳は、歩く、話す、計算するといった全く異なる性質のタスクを、互いに干渉することなく同一の脳内で学習し、実行することができます。これは、脳の神経回路（シナプス）がタスクごとに動的にルーティングされ、空間的に隔離された「機能局在」を持っているためです。一方で、人工神経網（ANN）においては、単一の巨大なネットワークで複数のタスクを学習させると過去の記憶が破壊される「破局的忘却（Catastrophic Forgetting）」が生じます。

本稿では、この生物学的な「シナプスの動的形成と空間的隔離」のメカニズムに着想を得た継続学習モデル「Synaptic Routing Architecture (SRA)」を提案します。SRAは、単層の極めてシンプルな「ルーター（Router）」と、複数の独立した極小モジュール（シナプス）で構成されます。私の実験により、SRAは推論時に外部からタスクIDを与えられなくとも、入力からタスクの性質を自律的に見極め、経路選択（ルーティング）とタスクの表現学習を**完全にEnd-to-Endで同時に学習する**ことを証明しました。人為的な重みの凍結（Freeze）や複雑な進化アルゴリズムなしに、モデル内部に自律的な機能局在が創発し、破局的忘却を完全に回避できることを示します。

## 1. Introduction
深層学習の分野において、モデルが継続的に新しい知識を獲得していく「継続学習（Continual Learning）」は、真の汎用人工知能（AGI）を実現するための最大の壁の一つです。現在の巨大なTransformerモデルのような「モノリシック（単一で密な）ネットワーク」は、新しいドメインのデータで微調整を行うと、以前に学習した知識を忘却してしまいます。

この問題に対し、本研究では「脳の機能局在（Functional Localization）」に着目します。脳の言語野や運動野のように、異なる機能は物理的に異なる回路を使用することで干渉を防ぎます。SRAは、この生物学的アプローチを計算機上に再現するため、ルーターと呼ばれる動的なルーティング機構を通じて、極小の独立したネットワーク（シナプス）を動的にオン/オフ（プラグイン/アンプラグ）できるアーキテクチャとして設計されました。

## 2. Related Work & SRAの新規性
破局的忘却を防ぐための既存アプローチとして、EWC (Elastic Weight Consolidation) のように「過去のタスクで重要だった重みの更新にペナルティを与える」正則化手法が存在します。しかし、これらの手法はモデルの容量（キャパシティ）に上限があり、タスクが増えればいずれ限界を迎えます。

よりSRAに近い「構造的・モジュール的」なアプローチとして、Google DeepMindによる **PathNet (2017)** が挙げられます。PathNetは多数のモジュールを用意し、遺伝的アルゴリズムを用いてタスクごとの「パス（経路）」を探索し、学習後はその重みを凍結（Freeze）することで忘却を防ぎます。

### SRAの圧倒的な優位性（同時学習の力）
PathNetのような従来のアプローチに対し、SRAの持つ根本的な新規性は**「経路の探索（ルーティング）とモジュールの学習を、微分可能な形で同時に（End-to-Endで）行う」**という点にあります。

1. **ルーティングの自律学習（Task-Agnostic）:** PathNetは、推論時に「今どのタスクを実行しているか（タスクID）」を外部から教える必要があります。しかしSRAは、たった1層の線形ルーターが、入力特徴量のコサイン類似度から「これは数学タスクだ」「これは言語タスクだ」と**推論時に自律的に判断し、適切なシナプスへルーティング**します。
2. **重み凍結や進化アルゴリズムの排除:** 遺伝的アルゴリズムのような巨大な計算コストを必要とせず、通常の誤差逆伝播法（Backpropagation）だけでルーターとシナプスが協調して学習します。
3. **動的な機能局在の創発:** ルーターが「類似したタスクは同じシナプスに」「異なるタスクは別のシナプスに」という経路を自発的に学習するため、人為的に重みを凍結せずとも、スパースな活性化による空間的隔離が自然と出来上がります。

## 3. Architecture (Neuro-inspired Design)
SRAは、生物の脳のシナプス形成を模倣した動的でスパース（疎）なアーキテクチャです。

### 3.1 The Router (Dynamic Synaptic Formation)
SRAの心臓部となるルーターは、入力された情報に対して「どの神経回路を発火させるべきか」を決定します。ルーターは単なる1層の線形層（Linear層）であり、入力特徴と各シナプスが持つ「埋め込みベクトル」とのコサイン類似度を計算し、最も合致する（発火する）Top-k 個のシナプスを決定します。

### 3.2 Tiny Synapses (Functional Modules)
シナプスは、独立した非常に小さなMulti-Head AttentionとMLPで構成されます。ルーターによって「発火」を指示されたシナプスのみが計算を実行し、それ以外のシナプスのパラメータは干渉を受けません。これにより、脳の空間的隔離と同様の忘却耐性を持ちます。

### 3.3 Architecture Diagram
以下の図は、入力がルーターによって評価され、最適なシナプス（神経回路）が発火する流れを示しています。

```mermaid
graph TD
    X[Input Token] --> Base[Residual Base]
    X --> Norm[LayerNorm]
    
    Norm --> Router["Router (Synaptic Routing)"]
    Norm --> SynapseSpace
    
    subgraph Synapse Space (Functional Modules)
        SynapseSpace((Select Top-k))
        S1["Synapse 0<br/>(Mini-Transformer)"]
        S2["Synapse 1<br/>(Mini-Transformer)"]
        S3["Synapse ..."]
        Sn["Synapse 15<br/>(Mini-Transformer)"]
    end
    
    Router -- "Routing Weights" --> SynapseSpace
    SynapseSpace --> S1
    SynapseSpace --> S2
    SynapseSpace -.-> Sn
    
    S1 --> Combine((Weighted Sum))
    S2 --> Combine
    Sn -.-> Combine
    
    Base --> Combine
    Combine --> Out[Output Representation]
```

## 4. Experiment 1: Algorithmic Reasoning
ルーターとモジュールを同時に学習させることで、タスクの性質を見抜けるかを検証するため、全く異なる4つのアルゴリズム的推論タスク（`copy`, `reverse`, `paren`, `addmod`）をタスクIDなしで同時に学習させました。

### 結果と機能局在の創発
10,000ステップの同時学習の結果、すべてのタスクでAccuracy 100%を達成しました。さらにルーターの経路分布を分析すると、驚くべき結果が得られました。

**ルーターによるタスクのクラスタリング:**
- **系列操作野**: `COPY` と `REVERSE` （類似度 0.969 で同じシナプスを共有）
- **計算・論理野**: `PAREN` と `ADDMOD` （類似度 0.858 で同じシナプスを共有）
- 上記2グループ間の類似度は 0.029 〜 0.336 と非常に低く、明確に機能が分離。

人間が一切の指示（タスクID）を与えなくとも、ルーターの同時学習によって「順番を入れ替えるタスク」と「ロジックを要するタスク」が自律的にクラスタリングされ、**脳の機能局在のようなモジュールの棲み分けが創発**しました。

## 5. Experiment 2: Cross-Domain Language Modeling
次に、さらに難易度の高い「異種ドメイン言語モデリング」を実施しました。文法や語彙が全く異なる `Code` (Python)、`Math` (LaTeX)、`Text` (自然言語) の3ドメインを同時に学習させました。

### 結果
わずか1000ステップの学習にもかかわらず、ルーターは以下のような「ドメインごとの専門回路」の形成を完了させました。
- `Code` 野: **シナプス 8** が支配的
- `Math` 野: **シナプス 10 と 13** が担当
- `Text` 野: **シナプス 0 と 15** が担当

モノリシックなモデルであれば破局的忘却が起きてしまうような状況でも、ルーターがドメインごとに専門のシナプス（独立したパラメータ空間）を割り当てたことで、相互の干渉を最小限に抑えることに成功しました。

## 6. Experiment 3: Multilingual Machine Translation
構文構造の異なる3言語（英語:SVO、フランス語:SVO、日本語:SOV）を用いた多言語機械翻訳を行いました。

### 結果と考察
シナプス使用率を分析したところ、英仏間（SVO同士）の翻訳時に高頻度で活性化する「SVO共有シナプス」と、日本語（SOV）への翻訳時にのみ使用率が跳ね上がる「SOV特化シナプス」が自律的に形成されていました。これは、ルーターが「言語」というラベルではなく、水面下で「語順・構文ルール」という本質的な言語の構造を獲得し、それに基づいて神経回路を動的に切り替えていることを示しています。

## 7. Experiment 4: Decision Transformer (Offline RL)
最後に、自然言語以外のドメインへの適応性を示すため、強化学習（RL）の軌跡データを学習させる Decision Transformer としての検証を行いました。ルールが全く異なる2つの環境（「Treasure」タスクと「Escape」タスク）のプレイログを同時に学習させました。

### 結果と考察
1トークンごとのルーティングを可視化した結果、**「知覚（Perception）」と「方策（Policy）」の完全な機能分離**という驚異的な現象が確認されました。
- **知覚トークン（状態の把握）**: 現在地などの状態トークンに対しては、タスクの種類にかかわらず**例外なく共通のシナプス（Expert 1）**にルーティングされました。これは共通の「空間知覚野」が形成されたことを意味します。
- **行動トークン（方策の決定）**: 一方で、次の行動を生成するトークンに対しては、Treasure用とEscape用で完全に別のシナプスへと経路が分岐しました。

「同じ目で環境を知覚し、異なる脳で判断を下す」という理想的なモジュール構造が、ルーティングの同時学習によって人為的設計なしに獲得されました。

## 8. Conclusion
本稿では、Synaptic Routing Architecture (SRA) を通じて、すべてのタスクで全パラメータを共有する「従来の静的なニューラルネットワーク」から、「生物学的な空間隔離と動的ルーティングを備えたモジュラー・ネットワーク」へのパラダイムシフトの可能性を示しました。

最大のブレイクスルーは、**ルーターの経路選択とモジュールの表現学習をEnd-to-Endで同時に行うことで、タスクIDに依存しない（Task-Agnosticな）継続学習が可能になったこと**です。PathNetのような複雑な進化アルゴリズムを必要とせず、ただシンプルなルーターを最適化するだけで、モデル内部に自律的な「機能局在」が創発しました。
SRAは、破局的忘却を克服し、無限に新しいタスク（シナプス）をプラグインできる、スケーラブルな汎用人工知能（AGI）へ向けた重要なステップです。

## Appendix: Interactive Demos

本稿で解説したSRAのアーキテクチャや実験結果を、ブラウザ上で実際に動かして体験できるJupyter Notebookデモを用意しています。以下のバッジからGoogle Colabを開き、気楽にお試しください。

- **1. 基礎構造とルーティングの確認**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart.ipynb)
- **2. シングルタスクの学習とルーティングの特化**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo.ipynb)
- **3. マルチタスク学習とタスクによる使い分け**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo.ipynb)
- **4. Decision Transformerと知覚・行動の分離**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo.ipynb)
- **5. 【必見】シナプス破壊（Lesion）実験**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo.ipynb)

## Appendix: Detailed Technical Reports

本稿の各実験に関する、より詳細な生データ、ログ、およびアーキテクチャ設計の過程については、リポジトリ内の以下のテクニカルレポート（Markdown）をご参照ください。

- **[SRA GPU Optimization & Benchmarking Report](./dev/SRA_GPU_Optimization_Report.md)**
- **[Multilingual Translation Routing Analysis](./dev/multilingual_translation_routing_analysis.md)**
- **[Decision Transformer Routing Analysis](./dev/decision_transformer_routing_analysis.md)**
