# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) は、生物の脳（シナプス）の仕組みから着想を得た、新しい動的・スパース（疎）なモジュール型ニューラルネットワーク・アーキテクチャです。
巨大で静的なTransformerに代わり、入力を適切な「シナプス（極小モジュール）」へ動的にルーティングすることで、より効率的な学習と構造的な知能の実現を目指しています。

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
