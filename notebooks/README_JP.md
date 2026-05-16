# Synaptic Routing Architecture (SRA) Interactive Demos

このディレクトリには、SRA（Synaptic Routing Architecture）の仕組みをブラウザ上で直感的に理解し、気楽に試していただくための Jupyter Notebook が用意されています。

これらのノートブックは、GPU環境をお持ちでない方でも **Google Colab** を使って数分で実行・体験できるように設計されています。

## 📓 ノートブック一覧

### 🟢 1. 基礎構造とルーティングの確認
**ファイル:** [`01_sra_quickstart.ipynb`](./01_sra_quickstart.ipynb)

SRAの基本的なモデル構造を初期化し、ランダムなデータを入力した際の「ルーティング（どの専門家が選ばれるか）」をヒートマップで可視化します。「とりあえずSRAを動かしてみたい」という方向けの最もシンプルな入門用デモです。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart.ipynb)

---

### 🔵 2. シングルタスクの学習とルーティングの特化
**ファイル:** [`02_learning_and_routing_demo.ipynb`](./02_learning_and_routing_demo.ipynb)

SRAに「Copyタスク（入力をそのまま出力する）」を実際に学習させます。数秒の学習の前後でルーティングのヒートマップを描画し、**学習が進むにつれてルーターが特定の専門家（シナプス）を意図的に選ぶように変化する（特化する）様子**を視覚的に体験できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo.ipynb)

---

### 🔴 3. マルチタスク学習とタスクによる使い分け
**ファイル:** [`03_multitask_routing_demo.ipynb`](./03_multitask_routing_demo.ipynb)

SRA最大の強みである**「マルチタスク学習における専門家（シナプス）の使い分け」**をデモします。
1つのモデルに対して `copy` と `reverse` という相反するタスクを同時に学習させます。学習後、入力されたタスクに応じてルーターが**全く異なるシナプスの経路を選択する様子**を並べて比較します。SRAのダイナミックな挙動を実感したい方に最適です。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo.ipynb)

---

### 🕹️ 4. Decision Transformerと知覚・行動の分離
**ファイル:** [`04_decision_transformer_routing_demo.ipynb`](./04_decision_transformer_routing_demo.ipynb)

SRAを強化学習のエージェント（Decision Transformer）として使い、GridWorld（迷路）の「宝探し」と「逃走」タスクを解かせます。
**「タスク（目的）による脳の使い分け」**に加えて、入力されるトークンが「状態（State）」「報酬（Reward）」「行動（Action）」のどれであるかによって**「知覚シナプス」と「行動シナプス」が自動的に分離して形成される様子**をヒートマップで確認できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo.ipynb)

---

### 🧠 5. 【必見】シナプス破壊（Lesion）実験
**ファイル:** [`05_lesion_experiment_demo.ipynb`](./05_lesion_experiment_demo.ipynb)

SRAのネットワークが「機能ごとに完全にモジュール化されている」ことを証明するハッカー的な実験デモです。
マルチタスク学習済みのモデルから、**特定のタスク（Reverse）で使われている専門家（シナプス）の重みを意図的にゼロに破壊**します。すると、Reverseタスクだけが解けなくなり、**別のシナプスを使っているCopyタスクは100%の正解率を維持し続ける（無傷である）**という驚異的な堅牢性をインタラクティブに体験できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo.ipynb)
---

### 🔌 6. シナプスの動的ホットスワップ実験とRouterの学習限界
**ファイル:** [`06_hotswap_experiment_demo.ipynb`](./06_hotswap_experiment_demo.ipynb)

SRAの醍醐味である「プラグインとしてのシナプスの動的な追加・削除（Hot-Swap）」を実証します。
稼働中のフランス語・ドイツ語翻訳モデルに対し、後からスペイン語専用のシナプスをマージする実験を行います。
本ノートブックでは、ホットスワップを成立させるための**ベースモデルの知識空間（Embedding/Attention層など）の共有・凍結の重要性**を学ぶと同時に、標準的なハードルーティング（Top-k）では追加シナプスのルーティングを後から学習（微分）できないという**SRA最大の壁（勾配消失問題）**に直面します。この限界が、次項の「Gumbel-Softmax（Super Router）」へと続く重要な伏線となります。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo.ipynb)

---

### 👑 7. 上位ルーターによるモデル統合とGumbel-Softmax
**ファイル:** [`07_super_router_gumbel_demo.ipynb`](./07_super_router_gumbel_demo.ipynb)

複数の特化型モデル（仏・独用モデルと西用モデル）を束ね、入力に応じて動的に処理を振り分ける「上位ルーター（Super Router）」を構築します。
単純なSoft Routingが抱える「ルーターの怠け（Lazy Routing）」問題を実証し、Gumbel-Softmax を用いることで不要なモデルの計算を100%カットする**完璧なHard Routing**を達成する過程を体験できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo.ipynb)

---

### 📖 8. SRAによるLLMの構築と学習 (TinyShakespeare)
**ファイル:** [`08_sra_llm_demo_shakespeare.ipynb`](./08_sra_llm_demo_shakespeare.ipynb)

SRA（Synaptic Routing Architecture）を用いて小規模な言語モデル（Tiny LLM）を一から学習させる手順を体験します。データセットにはシェイクスピアのテキストを使用し、学習後にSRAがどのようにテキストを生成し、文章のパターンに応じてどのようなルーティング（シナプスの使い分け）を行うのかを可視化します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare.ipynb)

---

### 📚 9. SRAによるマルチドメインLLMの学習と専門化
**ファイル:** [`09_sra_llm_demo_multidomain.ipynb`](./09_sra_llm_demo_multidomain.ipynb)

SRAの得意とする「複数ドメイン（Code, Math, Text）の同時学習」を小規模なLLMで体験します。ドメインごとにSRAが自動的にシナプスを分業（専門化）させる様子をヒートマップで確認でき、SRAの効率的なマルチタスク学習の仕組みを実証します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain.ipynb)

---

### 💻 10. 実践的プラグイン・ホットスワップ（Zero-Shot Hot-Swap）
**ファイル:** [`10_hotswap_plugins_demo.ipynb`](./10_hotswap_plugins_demo.ipynb)

複数の開発チームが「コード用プラグイン」「数学用プラグイン」を完全に独立して並行学習させ、それを本番環境のベースモデルに「事後的に物理合体（Hot-Swap）」させる一連のワークフローを実証します。合体後もすべてのドメインのLossが独立学習時と全く同一（Zero Forgetting）であることを証明しています。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)

---

### 🗑️ 11. シナプスの動的削除（Hot-Swapの取り消し・特定ドメインのパージ）
**ファイル:** [`11_synapse_deletion_demo.ipynb`](./11_synapse_deletion_demo.ipynb)

SRAの機能である「シナプスの削除」を実証します。後から追加したシナプスを末尾から物理的に削除し追加前の状態に復元する「プラグインの取り外し（pop_synapses）」と、複数ドメインを学習したベースモデルから特定の機能（例：Math）だけを抽出し、他と共用していないシナプスを安全にゼロクリアして無効化する「特定ドメインのパージ（clear_synapses）」の両方を体験できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

---

### 🧠 12. 仮想ニューロンの階層的創発（Virtual Neuron Emergence）
**ファイル:** [`12_virtual_neuron_experiment.ipynb`](./12_virtual_neuron_experiment.ipynb)

SRAモデルに対して「5つの異なるドメイン（NLP, Code, Math, DNA, CSV）× 5タスク = 全25タスク」の文字列処理を同時に学習させ、ルーターの重みを解析します。タスク間の「シナプスの重なり具合（Jaccard類似度）」を基にクラスタリングを行うことで、ネットワーク内にドメインごとの**独立した複数の仮想ニューロン（Cell Assembly）が階層構造を持って自律的に創発する様子**を視覚的に観察できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/12_virtual_neuron_experiment.ipynb)

---

### 🧬 13. 仮想ニューロン単位での安全なホットスワップ（アンラーニング）
**ファイル:** [`13_virtual_neuron_hotswap.ipynb`](./13_virtual_neuron_hotswap.ipynb)

ノートブック12で証明された「仮想ニューロン（Cell Assembly）」を単位として用いる実験です。単一のシナプスを無作為に削除すると発生する「破滅的忘却」を防ぐため、特定の機能（例：DNA処理）だけが専用で使っている「特化シナプス」を特定し、それらを物理的に破壊します。これにより、他のドメインの機能を一切損なわずに特定の知識だけを完璧にアンラーニングし、さらにバックアップからホットスワップ（復元）できることを実証します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/13_virtual_neuron_hotswap.ipynb)

---

### 🔬 14. 削除単位（シナプス量）の精度影響比較実験
**ファイル:** [`14_compare_deletion_units.ipynb`](./14_compare_deletion_units.ipynb)

本実験では、特定のタスクの知識をモデルから安全に削除する際、「専用シナプスを1個だけ削除する」場合と、「専用シナプスをすべて（仮想ニューロン単位で）削除する」場合での精度の低下度合い（Graceful Degradation）を比較します。他タスクとの共有シナプスは決して破壊しないため、どちらの場合も他タスクの精度は100%維持される「安全な削除」が成立することをグラフで確認できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/14_compare_deletion_units.ipynb)

---

### 📊 15. 仮説検証：シナプス数（キャパシティ）と安全なアンラーニングの閾値
**ファイル:** [`15_capacity_hypothesis_experiment.ipynb`](./15_capacity_hypothesis_experiment.ipynb)

本実験では、「モデルのシナプス容量（キャパシティ）を増やせば、各シナプスの専門性が細かく分離され、より低い（厳しい）閾値でも他タスクを破壊することなく安全に特定の知識を削除（アンラーニング）できるのではないか？」という仮説を検証します。シナプス数を変化させた複数のモデルを訓練し、アンラーニング時の保護閾値とタスク精度の関係をグラフで可視化・分析します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment.ipynb)

---

### 🚀 16. Lazy Routingの改善（Capacity LimitとEntropy制御）
**ファイル:** [`16_lazy_routing_prevention_experiment.ipynb`](./16_lazy_routing_prevention_experiment.ipynb)

SRAなどのMoEモデルで発生しがちな「一部のシナプスしか使われない（Lazy Routing）」問題を改善するための2つのアプローチを比較検証します。1つのシナプスに処理上限を設ける「Capacity Limit（ハード制約）」と、損失関数を操作して均等な使用を促す「Entropy Control（ソフト制約）」を導入し、ルーティング分布のエントロピーや死んだシナプスの割合がどのように改善されるかを可視化します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/16_lazy_routing_prevention_experiment.ipynb)

---

### 🔄 17. Routing Fallback（再割り当て）の実装と検証
**ファイル:** [`17_routing_fallback_experiment.ipynb`](./17_routing_fallback_experiment.ipynb)

SRAのルーターにおいて、容量上限（Capacity Limit）に達したシナプスからあぶれたトークンを次の候補シナプスへ強制的に転送する「Routing Fallback（再割り当て）」ロジックを実装します。単なる切り捨てによる情報欠落を防ぎ、Dead Synapse Ratio を完全に 0% に近づけつつ、モデル全体の精度（CE Loss）を向上させる手法とその効果を検証します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/17_routing_fallback_experiment.ipynb)

## 🚀 実行方法

1. **Google Colabで実行する場合（推奨）**:
   各項目の下にある `Open In Colab` バッジをクリックしてください。ブラウザ上で実行環境が立ち上がり、上から順番にセルを実行するだけで体験できます。

2. **ローカル環境で実行する場合**:
   ```bash
   git clone https://github.com/JunSuzukiJapan/SynapticRouter.git
   cd SynapticRouter
   pip install -r requirements.txt
   jupyter lab
   ```
   として、ブラウザから `notebooks/` 以下のファイルを開いて実行してください。
