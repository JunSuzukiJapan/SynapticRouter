# All You Need Is Router: Dynamic Sparse Modularity in Neural Networks

**Jun Suzuki**, Independent Researcher

## Abstract
近年、深層学習モデルは巨大化の一途を辿り、学習に必要な計算リソースは爆発的に増加しています。また、単一の巨大なネットワーク（モノリシック・モデル）で性質の異なる複数のタスクを学習させると「破局的忘却（Catastrophic Forgetting）」が生じやすいという問題もあります。本稿では、この問題に対する解として「Synaptic Routing Architecture (SRA)」を提案し、Attentionを持たない極めてシンプルな「単層のルーター（Router）」が、自律的に複数の極小モデル（シナプス）へタスクを振り分け、破局的忘却を完全に回避できることを実験的に証明します。結論として、複雑なタスクを同時に学習する上で本当に必要だったのは、巨大で密なTransformerではなく、入力に応じて適切なモジュールを選択する「ルーター」でした。

## 1. Introduction
「Attention Is All You Need」の登場以降、Transformerアーキテクチャは自然言語処理から画像認識、強化学習に至るまであらゆるドメインを席巻しました。しかし、パラメータを密（Dense）に活性化させる従来のアプローチでは、スケールアップに伴う計算コストが指数関数的に増大します。
近年、Mixtral などに代表される MoE (Mixture of Experts) が注目を集めていますが、SRA はこの MoE の概念をさらに推し進め、「極小の計算ユニット（シナプス）」と「それらを動的に組み合わせる軽量なルーター」によって構成されるネットワークを設計しました。本稿では、「Routerこそがマルチタスク学習におけるモデルの頭脳である」という仮説を検証します。

## 2. Architecture (SRA)
SRAは生物の脳を模倣した動的でスパース（疎）なアーキテクチャです。巨大なTransformerの代わりに、非常に軽量なコンポーネントの組み合わせで構築されています。

### 2.1 The Router (All You Need Is Router)
SRAの心臓部であり、「肝」となるのがルーターです。ルーター自体はAttentionなどの複雑な機構を一切持たず、実体は**単なる1層の線形層（Linear層）**です。
ルーターは入力されたデータの隠れ状態と、各シナプスが持つ「特徴ベクトル（埋め込み）」との内積（コサイン類似度）を計算し、最もスコアが高い（合致する） Top-k 個のシナプスを素早く判定します。

### 2.2 Tiny Synapses
各シナプスは、小型のMulti-Head AttentionとMLPからなる独立した極小モジュールです。ルーターによって選ばれたシナプスのみが計算を実行するため、非常に高い計算効率を誇ります。

### 2.3 Architecture Diagram
以下の図は、入力がルーターによって評価され、最適なシナプスにルーティングされる流れを示しています。

```mermaid
graph TD
    X[Input Token] --> Base[Residual Base]
    X --> Norm[LayerNorm]
    
    Norm --> Router["Router (Linear Layer)"]
    Norm --> SynapseSpace
    
    subgraph Synapse Space
        SynapseSpace((Select Top-k))
        S1["Synapse 0<br/>(Mini-Transformer)"]
        S2["Synapse 1<br/>(Mini-Transformer)"]
        S3["Synapse ..."]
        Sn["Synapse 15<br/>(Mini-Transformer)"]
    end
    
    Router -- "Output Routing Weights" --> SynapseSpace
    SynapseSpace --> S1
    SynapseSpace --> S2
    SynapseSpace -.-> Sn
    
    S1 --> Combine((Weighted Sum))
    S2 --> Combine
    Sn -.-> Combine
    
    Base --> Combine
    Combine --> Out[Output Representation]
```

## 3. Experiment 1: Algorithmic Reasoning
ルーターが異なるタスクを自律的に見分けられるかを検証するため、性質の全く異なる4つのアルゴリズム的推論タスク（`copy`, `reverse`, `paren`, `addmod`）を1つのSRAモデルに同時に学習させました。

### 結果
10,000ステップの同時学習の結果、すべてのタスクにおいて**Accuracy 100%（完全な推論）**を達成しました。
さらに、ルーターがどのタスクでどのシナプスを使ったのか（ルーティング分布）を抽出し、タスク間のコサイン類似度を分析したところ、驚くべき結果が得られました。

**ルーターによるタスクのクラスタリング（深いレイヤー）:**
- **系列操作グループ**: `COPY` と `REVERSE` （類似度 0.969）
- **計算/論理グループ**: `PAREN` と `ADDMOD` （類似度 0.858）
- 上記2グループ間の類似度は 0.029 〜 0.336 と明確に分離。

人間が一切の指示を与えなくとも、ルーターは「系列の順番を入れ替えるタスク」と「ロジックや計算を要するタスク」を**自律的に見抜き、似たタスクではシナプスを共有し、異なるタスクでは明確に別のシナプスを使うようにモジュールを分離**していました。

## 4. Experiment 2: Cross-Domain Language Modeling
次に、さらに難易度の高い「異種ドメイン言語モデリング」を実施しました。文法や語彙が全く異なる `Code` (Python)、`Math` (LaTeX)、`Text` (自然言語) の3ドメインを同時に学習させました。

### 結果
わずか1000ステップの学習にもかかわらず、Pythonのインデント、LaTeXの特殊記法、自然言語の文脈を完璧に推論・生成することができました。

**シナプスの使用頻度の推移と専門化:**
学習初期（Warmup時）には全シナプスが均等に使われていましたが、学習終盤においてルーターは以下のような「ドメインごとの棲み分け」を完了させました。
- `Code` の処理: **シナプス 8** が支配的
- `Math` の処理: **シナプス 10 と 13** が担当
- `Text` の処理: **シナプス 0 と 15** が担当

モノリシックなモデルであれば破局的忘却が起きてしまうような状況でも、ルーターがドメインごとに専門のシナプス（独立したパラメータ空間）を割り当てたことで、相互の干渉を最小限に抑えることに成功しました。

## 5. Conclusion
本稿では、Synaptic Routing Architecture (SRA) を通じて、「巨大なモデルの一括計算」から「極小モジュールの動的選択」へのパラダイムシフトの可能性を示しました。
アルゴリズム的推論および異種ドメイン言語モデリングの実験結果が示す通り、複数タスクの干渉を防ぎ、高い汎化性能と学習効率を両立するために本当に必要なのは、複雑なAttention機構の巨大化ではなく、シンプルで賢い「ルーター」の存在でした。まさに、**"All You Need Is Router"** なのです。
