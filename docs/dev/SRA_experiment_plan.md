# SRA 最小実験セット

## 目的
Synaptic Routing Architecture（SRA）の最小検証。
自然言語や翻訳ではなく、構造差が出やすい人工タスクから始める。

## タスク

### 1. copy
入力列をそのまま出力する。
- 目的: 記憶・系列保持の最低限チェック

### 2. reverse
入力列を逆順に出力する。
- 目的: 位置・順序操作を見る

### 3. paren
括弧列が balanced かを Y/N で出力する。
- 目的: スタック的・状態的な処理を見る

### 4. addmod
1桁同士の加算を mod 10 で出力する。
- 目的: 小さな演算・規則学習を見る

## 最初に見る指標

- validation loss
- sequence accuracy
- synapse usage distribution
- k=1,2,4 の差
- load_balance を入れた場合と外した場合の差

## 推奨実行順

```bash
python sra_experiment.py --task copy --steps 1000
python sra_experiment.py --task reverse --steps 2000
python sra_experiment.py --task paren --steps 3000
python sra_experiment.py --task addmod --steps 1000
```

## 比較実験

```bash
python sra_experiment.py --task reverse --k 1 --steps 2000
python sra_experiment.py --task reverse --k 2 --steps 2000
python sra_experiment.py --task reverse --k 4 --steps 2000
```

## 次に追加する機能

1. シナプスembeddingによる距離ベースrouting
2. 近傍シナプスの混合
3. short_state / long_state
4. usage penalty
5. Router freeze後のSynapse specialization
