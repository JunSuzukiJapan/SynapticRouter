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

補足:

- 出力先 checkpoint の現物は `checkpoints/mixed_sra_pretrain.pt`
- SFT 出力先の `checkpoints/mixed_sra_chat.pt` はまだ未作成
- そのため、現時点の生成確認は `checkpoints/mixed_sra_pretrain.pt` を対象にする
- `src/train_sft_small_llm.py` は `--local-datasets-only` がデフォルト有効なので、`no_robots` / `oasst2` がローカルに未キャッシュだとそのまま失敗する
- `docs/dev/SMALL_LLM_TRAINING_JP.md` の 2026-06-03 メモでは、このマシンの PyTorch は `mps` build 済みでも runtime では `is_available() == False`

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

実行デバイス確認:

```bash
./.venv/bin/python - <<'PY'
import torch
mps = getattr(torch.backends, "mps", None)
print("mps_built=", bool(mps and mps.is_built()))
print("mps_available=", bool(mps and mps.is_available()))
print("cuda_available=", torch.cuda.is_available())
PY
```

データセットキャッシュ確認:

```bash
./.venv/bin/python - <<'PY'
from datasets import DownloadConfig, load_dataset
for name in ["HuggingFaceH4/no_robots", "OpenAssistant/oasst2"]:
    try:
        load_dataset(name, split="train", download_config=DownloadConfig(local_files_only=True))
        print(name, "cached=YES")
    except Exception as exc:
        print(name, "cached=NO", type(exc).__name__)
PY
```

この確認で `cached=NO` が出たら、SFT に入る前に通常ターミナル側で一度データ取得が必要。

## 次回の本命コマンド

まずは `no_robots` だけで SFT。

現状の安全側は `--device auto`。`mps_available=True` を再確認できた場合だけ `--device mps` に変える。

```bash
./.venv/bin/python src/train_sft_small_llm.py \
  --base-model checkpoints/mixed_sra_pretrain.pt \
  --save checkpoints/mixed_sra_chat.pt \
  --device auto \
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
  --device auto \
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

SFT 前の現状確認は `mixed_sra_pretrain.pt` を使う。

評価:

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/mixed_sra_pretrain.pt \
  --device auto \
  --eval
```

単発生成:

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/mixed_sra_pretrain.pt \
  --device auto \
  --prompt "System: You are a helpful assistant.\n\nUser: Tell me about Albert Einstein.\nAssistant:"
```

REPL:

```bash
./.venv/bin/python src/chat_small_llm.py \
  --model checkpoints/mixed_sra_pretrain.pt \
  --device auto \
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
- `train_sft_small_llm.py` はデフォルトで `local_files_only=True` なので、HF キャッシュ未取得なら先に通常ターミナルでデータを落とす
- 2026-06-03 時点のメモでは、このマシンは通常ターミナルでも `mps_available=False`。まずは `--device auto` で CPU fallback を許容し、MPS は再確認してから使う
