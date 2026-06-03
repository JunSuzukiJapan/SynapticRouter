# 次回やること: 小規模LLM / SRA 会話化

## 現在地

混合事前学習は完了済み。

- checkpoint: `checkpoints/mixed_sra_pretrain.pt`
- 完了 step: `4000 / 4000`
- model preset: `8gb_safe`
- tokenizer: `tiktoken / cl100k_base`
- sources: `tinystories,fineweb_edu_sample,wikitext2`

主な学習設定:

- `dim=384`
- `layers=6`
- `synapses=32`
- `syn_hidden=768`
- `seq_len=256`
- `batch_size=4`
- `grad_accum=4`

## 次回の目的

1. この事前学習済み checkpoint をベースに会話 SFT を回す
2. 生成品質を確認する
3. 必要なら追加で SFT を継続する

## 最初にやること

事前学習 checkpoint の内容確認:

```bash
./.venv/bin/python - <<'PY'
import torch
ckpt=torch.load('checkpoints/mixed_sra_pretrain.pt', map_location='cpu')
print('step=', ckpt.get('step'))
print('model_preset=', ckpt['args'].get('model_preset'))
print('sources=', ckpt['args'].get('sources'))
print('tokenizer=', ckpt['args'].get('tokenizer_type'), ckpt['args'].get('tokenizer_name'))
PY
```

## 次回の本命コマンド

まずは `no_robots` だけで SFT:

```bash
./.venv/bin/python src/train_sft_small_llm.py \
  --base-model checkpoints/mixed_sra_pretrain.pt \
  --save checkpoints/mixed_sra_chat.pt \
  --device mps \
  --steps 300 \
  --batch-size 8 \
  --grad-accum 2 \
  --log-every 25 \
  --save-every 100 \
  --max-train-no-robots 8000 \
  --max-train-oasst2 0 \
  --max-valid-no-robots 500 \
  --max-valid-oasst2 0
```

余裕があれば `oasst2` も追加:

```bash
./.venv/bin/python src/train_sft_small_llm.py \
  --base-model checkpoints/mixed_sra_pretrain.pt \
  --save checkpoints/mixed_sra_chat.pt \
  --device mps \
  --steps 500 \
  --batch-size 8 \
  --grad-accum 2 \
  --log-every 25 \
  --save-every 100 \
  --max-train-no-robots 8000 \
  --max-train-oasst2 12000 \
  --max-valid-no-robots 500 \
  --max-valid-oasst2 1000
```

## 確認方法

SFT 後の評価:

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/mixed_sra_chat.pt \
  --device mps \
  --eval
```

単発生成:

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/mixed_sra_chat.pt \
  --device mps \
  --prompt "System: You are a helpful assistant.\n\nUser: Tell me about Albert Einstein.\nAssistant:"
```

REPL:

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/mixed_sra_chat.pt \
  --device mps \
  --chat
```

## 期待値

この時点で目指すのは:

- `wikitext2_sra.pt` ベースよりも会話らしい返答になること
- `User:` / `Assistant:` 形式を維持できること
- 完全な高品質チャットではなくても、意味の通る短い応答が増えること

## 注意

- `chat_small_llm.py` は checkpoint 内の tokenizer 設定を読む
- `mixed_sra_pretrain.pt` は `tiktoken` 前提なので、`.venv` 側で `tiktoken` が使えること
- MPS は Codex の通常サンドボックスでは見えないことがあるので、必要なら通常ターミナルで回す
