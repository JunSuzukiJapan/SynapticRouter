#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f checkpoints/mixed_sra_pretrain_en.pt ]]; then
  echo "missing checkpoint: checkpoints/mixed_sra_pretrain_en.pt" >&2
  exit 1
fi

avail_kb=$(df -k . | awk 'NR==2 {print $4}')
if [[ "${avail_kb}" -lt 2097152 ]]; then
  echo "not enough free disk space for JA SFT. need ~2GB free, have $((avail_kb / 1024))MB" >&2
  exit 1
fi

./.venv/bin/python src/train_sft_small_llm.py \
  --base-model checkpoints/mixed_sra_pretrain_en.pt \
  --save checkpoints/mixed_sra_chat_ja.pt \
  --device auto \
  --steps 1500 \
  --batch-size 4 \
  --grad-accum 4 \
  --log-every 50 \
  --save-every 250 \
  --sft-sources llm_jp_oasst2_33k_ja \
  --max-train-llm_jp_oasst2_33k_ja 8000 \
  --max-valid-llm_jp_oasst2_33k_ja 500
