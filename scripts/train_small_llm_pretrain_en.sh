#!/usr/bin/env bash
set -euo pipefail

./.venv/bin/python src/train_pretrain_mixed_small_llm.py \
  --sources tinystories,wikitext2 \
  --weight-tinystories 0.90 \
  --weight-wikitext2 0.10 \
  --model-preset 8gb_safe \
  --device auto \
  --checkpoint-path checkpoints/mixed_sra_pretrain_en.pt
