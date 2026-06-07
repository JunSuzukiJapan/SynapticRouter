#!/usr/bin/env bash
set -euo pipefail

./.venv/bin/python src/train_pretrain_mixed_small_llm.py \
  --sources tinystories,wikitext2 \
  --weight-tinystories 0.90 \
  --weight-wikitext2 0.10 \
  --model-preset 8gb_infer \
  --layers 5 \
  --steps 18000 \
  --warmup-steps 1800 \
  --joint-steps 12600 \
  --stabilize-steps 2700 \
  --log-every 100 \
  --save-every 400 \
  --device auto \
  --checkpoint-path checkpoints/mixed_sra_pretrain_en_5l_long.pt
