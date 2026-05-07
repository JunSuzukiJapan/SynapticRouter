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

### 🔌 6. シナプスの動的ホットスワップ実験
**ファイル:** [`06_hotswap_experiment_demo.ipynb`](./06_hotswap_experiment_demo.ipynb)

SRAの醍醐味である「プラグインとしてのシナプスの追加・置換」を実証します。
独立して学習させたスペイン語翻訳モデル（Model 2）の重みを、フランス語・ドイツ語翻訳モデル（Model 1）へ稼働中にマージする実験を行います。基礎となる特徴表現（Attention/Embedding）の共有がなぜ重要なのか、アーキテクチャのモジュール性に関する深い洞察を得られます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo.ipynb)

---

### 👑 7. 上位ルーターによるモデル統合とGumbel-Softmax
**ファイル:** [`07_super_router_gumbel_demo.ipynb`](./07_super_router_gumbel_demo.ipynb)

複数の特化型モデル（仏・独用モデルと西用モデル）を束ね、入力に応じて動的に処理を振り分ける「上位ルーター（Super Router）」を構築します。
単純なSoft Routingが抱える「ルーターの怠け（Lazy Routing）」問題を実証し、Gumbel-Softmax を用いることで不要なモデルの計算を100%カットする**完璧なHard Routing**を達成する過程を体験できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo.ipynb)

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
