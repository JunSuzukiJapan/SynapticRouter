# 小規模LLM 事前学習レイヤ比較メモ

更新日: 2026-06-08

## 目的

`TinyStories 90% + WikiText-2 10%` の軽量事前学習条件で、

- `layers=1`
- `layers=3`
- `layers=5`

を短時間で比較し、長時間学習の本命深さを決める。

## 比較条件

共通条件:

- script: `scripts/run_small_llm_pretrain_layer_sweep.sh`
- sources: `tinystories,wikitext2`
- weight: `tinystories=0.90`, `wikitext2=0.10`
- preset: `8gb_infer`
- `dim=448`
- `synapses=32`
- `syn_hidden=896`
- `seq_len=256`
- `batch_size=3`
- `grad_accum=6`
- `steps=2000`
- `warmup=200`
- `joint=1400`
- `stabilize=300`
- `specialize=100`

比較したのは `layers` のみ。

出力先:

- 1層: `checkpoints/mixed_sra_pretrain_en_1l_smoke.pt`
- 3層: `checkpoints/mixed_sra_pretrain_en_3l_smoke.pt`
- 5層: `checkpoints/mixed_sra_pretrain_en_5l_smoke.pt`

ログ:

- `logs/mixed_sra_pretrain_en_1l_smoke.log`
- `logs/mixed_sra_pretrain_en_3l_smoke.log`
- `logs/mixed_sra_pretrain_en_5l_smoke.log`

## 結果サマリ

| layers | final val_loss | best val_loss | best step | TinyStories final | WikiText-2 final | training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.8778 | 3.8424 | 1950 | 3.5768 | 6.5869 | 4441.3s |
| 3 | 3.5015 | 3.3306 | 1900 | 3.1768 | 6.4239 | 10937.6s |
| 5 | 3.4678 | 3.2977 | 1800 | 3.1253 | 6.5503 | 17436.3s |

学習時間の目安:

- 1層: 約 1.2 時間
- 3層: 約 3.0 時間
- 5層: 約 4.8 時間

## 数値の読み

### 1層 -> 3層

改善幅は大きい。

- final `3.8778 -> 3.5015`
- best `3.8424 -> 3.3306`

1層は明らかに浅く、深さを増やす価値がある。

### 3層 -> 5層

改善はあるが差は小さい。

- final `3.5015 -> 3.4678`
- best `3.3306 -> 3.2977`

5層は最良だが、3層との差はかなり小さい。

### データセット別

`TinyStories` では 5層が最良。

- 1層: `3.5768`
- 3層: `3.1768`
- 5層: `3.1253`

`WikiText-2` では 3層が最良。

- 1層: `6.5869`
- 3層: `6.4239`
- 5層: `6.5503`

そのため、全体では 5層がわずかに勝つが、汎化バランスでは 3層も強い。

## 生成サンプルの所感

### 1層

- 文の接続がかなり弱い
- 反復が多い
- 話の飛び方が大きい

### 3層

- 1層より局所的な自然さは明確に改善
- ただし意味の継続性はまだ弱い
- 物語断片の継ぎ足し感が残る

### 5層

- 3層より文としての滑らかさは少し改善
- ただし意味内容の改善は限定的
- 深くしたぶんの改善はあるが、計算コストに対しては小さい

## 学習ダイナミクス上の気づき

- 1層は早いが、早い段階で頭打ちになりやすい
- 3層と5層は後半まで改善が続く
- 3層も5層も `specialize` に入った終盤で少し悪化している

今回の短い比較では、最良値は final step より少し前に出ている。

- 1層 best: `step=1950`
- 3層 best: `step=1900`
- 5層 best: `step=1800`

短時間スモークでは `specialize` フェーズが最終値を少し悪化させる可能性がある。

## 判断

### 性能重視

5層が最良。

ただし差は小さい。

### 時間効率重視

3層が有力。

- 1層よりかなり良い
- 5層との差は小さい
- 学習時間は 5層の約 6 割

## 推奨

長時間学習の本命は **3層** にする。

理由:

- 改善幅と計算コストのバランスが最も良い
- 5層は少し良いが、差が小さい
- まず 3層で `18000 step` を回してから、必要なら 5層長時間版を比較すれば十分

運用方針:

- 本命: `scripts/train_small_llm_pretrain_en.sh`
  - `layers=3`
  - `steps=18000`
- 比較用: `scripts/train_small_llm_pretrain_en_5l_long.sh`
  - `layers=5`
  - `steps=18000`

## 補足

今回の比較は、深さだけを変えた短時間スモークであり、最終性能の確定ではない。

ただし、少なくとも次のことは言える。

- 1層は浅すぎる
- 3層で大きく改善する
- 5層は最良だが、改善幅は限定的

したがって、現時点では **3層を主軸にして十分妥当**。
