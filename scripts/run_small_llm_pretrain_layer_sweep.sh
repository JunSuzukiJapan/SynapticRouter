#!/usr/bin/env bash
set -euo pipefail

STEPS=${STEPS:-2000}
WARMUP_STEPS=${WARMUP_STEPS:-200}
JOINT_STEPS=${JOINT_STEPS:-1400}
STABILIZE_STEPS=${STABILIZE_STEPS:-300}
LOG_EVERY=${LOG_EVERY:-50}
SAVE_EVERY=${SAVE_EVERY:-200}
DEVICE=${DEVICE:-auto}

mkdir -p checkpoints logs

for layers in 1 3 5; do
  checkpoint_path="checkpoints/mixed_sra_pretrain_en_${layers}l_smoke.pt"
  log_path="logs/mixed_sra_pretrain_en_${layers}l_smoke.log"

  echo "=== layers=${layers} steps=${STEPS} checkpoint=${checkpoint_path} ==="
  ./.venv/bin/python src/train_pretrain_mixed_small_llm.py \
    --sources tinystories,wikitext2 \
    --weight-tinystories 0.90 \
    --weight-wikitext2 0.10 \
    --model-preset 8gb_infer \
    --layers "${layers}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --joint-steps "${JOINT_STEPS}" \
    --stabilize-steps "${STABILIZE_STEPS}" \
    --log-every "${LOG_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --device "${DEVICE}" \
    --checkpoint-path "${checkpoint_path}" | tee "${log_path}"
done
