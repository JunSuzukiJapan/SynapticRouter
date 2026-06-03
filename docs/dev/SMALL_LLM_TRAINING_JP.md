# 小規模LLM学習メモ

`src/train_small_llm.py` は、公開データセットを Hugging Face `datasets` 経由で取得し、SRA の小規模言語モデルを学習するためのスクリプトです。

## 事前準備

```bash
python3 -m pip install datasets
```

## まずは TinyStories で試す

```bash
python3 src/train_small_llm.py \
  --dataset wikitext2 \
  --steps 1000 \
  --batch-size 16 \
  --seq-len 128 \
  --checkpoint-path checkpoints/wikitext2_sra.pt
```

- データは `data/hf_cache/` にキャッシュされます。
- デフォルトでは `wikitext / wikitext-2-raw-v1` の `train` / `validation` を使います。
- トークナイザは依存を増やさないため、UTF-8 バイト列ベースです。
- Apple Silicon で Metal を使いたい場合は `--device mps` を指定します。

## 中断後に再開する

```bash
python3 src/train_small_llm.py \
  --dataset wikitext2 \
  --steps 2000 \
  --resume checkpoints/wikitext2_sra.pt \
  --checkpoint-path checkpoints/wikitext2_sra.pt
```

- `--resume` に既存チェックポイントを渡すと、`model` / `optimizer` / `step` / RNG state を復元します。
- `--steps` は「最終的にどこまで進めるか」です。たとえば 1000 step まで終わっているチェックポイントに対して `--steps 2000` を指定すると、1001 step から再開します。

## 別の公開データセットを使う

プリセット以外も指定できます。

```bash
python3 src/train_small_llm.py \
  --dataset-path wikitext \
  --dataset-name wikitext-2-raw-v1 \
  --train-split train \
  --valid-split validation \
  --text-column text \
  --checkpoint-path checkpoints/wikitext2_sra.pt
```

TinyStories を使う場合:

```bash
python3 src/train_small_llm.py \
  --dataset tinystories \
  --checkpoint-path checkpoints/tinystories_sra.pt
```

## 学習済みモデルをチャット風に使う

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/wikitext2_sra.pt \
  --device mps \
  --chat
```

1回だけ生成を見る場合:

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/wikitext2_sra.pt \
  --device mps \
  --prompt "Alexander was featured in the 1985 documentary The Search for Meng"
```

検証 loss を見る場合:

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/wikitext2_sra.pt \
  --device mps \
  --eval
```

注意:

- `wikitext2` での学習は事前学習のみなので、これは「指示に従うアシスタント」ではなく「Wikipedia 風テキストを継続生成する小型LM」です。
- そのためチャット REPL は動きますが、会話品質を上げるには別途 SFT 用の対話データで追加学習が必要です。

## 8GB 前提の最大構成メモ

MPS 上で 1 step の forward/backward を確認した構成:

- `dim=320, layers=6, synapses=32, syn_hidden=640` : 約 `113.75M` params
- `dim=384, layers=6, synapses=32, syn_hidden=768` : 約 `155.96M` params
- `dim=512, layers=8, synapses=32, syn_hidden=1024` : 約 `329.21M` params

このリポジトリでは、8GB 級で「最大を狙う」候補は `dim=512 / layers=8 / synapses=32 / syn_hidden=1024`。

ただし注意:

- これは「1 step 通る」確認であり、長時間学習の安定性や実効スループットは別
- 実運用では `dim=384` か `dim=512` を、`batch-size` と `grad-accum` を落として回すのが現実的

## 会話に使う公開データ

事前学習向け:

- `roneneldan/TinyStories`
- `HuggingFaceFW/fineweb-edu` の小規模サンプル
- `wikitext / wikitext-2-raw-v1`

会話SFT向け:

- `HuggingFaceH4/no_robots`
- `OpenAssistant/oasst2`

参考:

- TinyStories: <https://huggingface.co/datasets/roneneldan/TinyStories>
- FineWeb-Edu: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- no_robots: <https://huggingface.co/datasets/HuggingFaceH4/no_robots>
- OASST2: <https://huggingface.co/datasets/OpenAssistant/oasst2>

## 追加した会話学習スクリプト

```bash
./.venv/bin/python src/train_sft_small_llm.py \
  --base-model checkpoints/wikitext2_sra.pt \
  --save checkpoints/wikitext2_sra_chat.pt \
  --device mps
```

短い確認用:

```bash
PYTHONPATH=src ./.venv/bin/python -u src/train_sft_small_llm.py \
  --base-model checkpoints/wikitext2_sra.pt \
  --save checkpoints/wikitext2_sra_chat.pt \
  --device mps \
  --steps 10 \
  --batch-size 4 \
  --grad-accum 1 \
  --log-every 1 \
  --max-train-no-robots 1000 \
  --max-train-oasst2 0
```

## 混合事前学習スクリプト

`src/train_pretrain_mixed_small_llm.py` を追加した。公開データを混ぜて、8GB 級の大きめ SRA を事前学習するための入口。

デフォルトの混合:

- `tinystories`
- `fineweb_edu_sample`
- `wikitext2`

8GB 上限寄りの推奨起動例:

```bash
./.venv/bin/python src/train_pretrain_mixed_small_llm.py \
  --model-preset 8gb_max \
  --device mps \
  --checkpoint-path checkpoints/mixed_sra_pretrain.pt
```

より安全寄り:

```bash
./.venv/bin/python src/train_pretrain_mixed_small_llm.py \
  --model-preset 8gb_safe \
  --device mps \
  --checkpoint-path checkpoints/mixed_sra_pretrain.pt
```

スモークテスト用の最小例:

```bash
./.venv/bin/python src/train_pretrain_mixed_small_llm.py \
  --sources wikitext2 \
  --tokenizer-type byte \
  --device mps \
  --steps 1 \
  --dim 128 \
  --layers 2 \
  --synapses 8 \
  --syn-hidden 256 \
  --seq-len 64 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --grad-accum 1
```

補足:

- `tiktoken` は初回に語彙ファイル取得が必要
- `chat_small_llm.py` と `train_sft_small_llm.py` は checkpoint 内の tokenizer 設定を読むように更新済み

## よく触る引数

- `--max-train-examples`: 学習に読む件数上限
- `--max-valid-examples`: 検証に読む件数上限
- `--save-every`: 何 step ごとにチェックポイント保存するか
- `--dim`, `--layers`, `--synapses`, `--k`: モデル規模とSRAルーティング設定
- `--cpu`: GPU/MPS を使わず CPU 強制

## MPS / Metal の現状メモ

2026-06-03 時点で、このマシンでは以下を確認済みです。

- macOS: `26.4.1`
- ハードウェア: Apple Silicon (`M4 Pro`)
- `torch.backends.mps.is_built() == True`
- それでも `torch.backends.mps.is_available() == False`
- `torch.ones(1, device="mps")` は `The MPS backend is supported on macOS 14.0+` で失敗

確認した組み合わせ:

- `/usr/bin/python3` + `torch 2.8.0`
- `uv` 管理 Python `3.12.12` + `torch 2.12.0`
- `uv` 管理 Python `3.12.12` + `torch 2.13.0.dev20260602` nightly

上記すべてで MPS は利用不可でした。現時点ではローカルコードの問題ではなく、`macOS 26.x` 系と PyTorch MPS の相性問題として扱うのが妥当です。

したがって当面の推奨は:

- 学習は `CPU` で進める
- MPS 復旧は環境タスクとして別管理する
