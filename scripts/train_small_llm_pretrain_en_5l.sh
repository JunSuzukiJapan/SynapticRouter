#!/usr/bin/env bash
set -euo pipefail

./.venv/bin/python src/train_pretrain_mixed_small_llm.py \
  --sources tinystories,wikitext2 \
  --weight-tinystories 0.90 \
  --weight-wikitext2 0.10 \
  --model-preset 8gb_infer \
  --layers 5 \
  --steps 2000 \
  --warmup-steps 200 \
  --joint-steps 1400 \
  --stabilize-steps 300 \
  --log-every 50 \
  --save-every 200 \
  --device auto \
  --checkpoint-path checkpoints/mixed_sra_pretrain_en_5l_smoke.pt
