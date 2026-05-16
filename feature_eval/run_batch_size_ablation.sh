#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Activate the intended Python environment first,
# for example: conda activate pytorch_env

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-./datasets_local}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-100}"
LINEAR_BATCH_SIZE="${LINEAR_BATCH_SIZE:-256}"
WORKERS="${WORKERS:-0}"
LINEAR_LR="${LINEAR_LR:-0.001}"
LINEAR_WEIGHT_DECAY="${LINEAR_WEIGHT_DECAY:-0}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
RESULT_DIR="${RESULT_DIR:-feature_eval/results/batch_size_ablation_128_256_512_1024}"
REPORT_DIR="${REPORT_DIR:-feature_eval/report/batch_size_ablation_128_256_512_1024}"

mkdir -p "${RESULT_DIR}"

"${PYTHON_BIN}" feature_eval/linear_eval.py \
  --checkpoint runs/May04_21-52-10_DESKTOP-UB2KPK6/checkpoint_0100.pth.tar \
  --name batch_size_128_may04 \
  --data "${DATA_DIR}" \
  --epochs "${LINEAR_EPOCHS}" \
  --batch-size "${LINEAR_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LINEAR_LR}" \
  --weight-decay "${LINEAR_WEIGHT_DECAY}" \
  --label-fraction 1.0 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output-dir "${RESULT_DIR}"

"${PYTHON_BIN}" feature_eval/linear_eval.py \
  --checkpoint runs/20260503-161329_cifar10_resnet18_aug-strong_proj-on_bs-256_chengbo_e100_j32/checkpoint_0100.pth.tar \
  --name batch_size_256_20260503 \
  --data "${DATA_DIR}" \
  --epochs "${LINEAR_EPOCHS}" \
  --batch-size "${LINEAR_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LINEAR_LR}" \
  --weight-decay "${LINEAR_WEIGHT_DECAY}" \
  --label-fraction 1.0 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output-dir "${RESULT_DIR}"

"${PYTHON_BIN}" feature_eval/linear_eval.py \
  --checkpoint runs/20260506-192532_cifar10_resnet18_aug-strong_proj-on_bs-512_chengbo_bs512_e100/checkpoint_0100.pth.tar \
  --name batch_size_512_20260506 \
  --data "${DATA_DIR}" \
  --epochs "${LINEAR_EPOCHS}" \
  --batch-size "${LINEAR_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LINEAR_LR}" \
  --weight-decay "${LINEAR_WEIGHT_DECAY}" \
  --label-fraction 1.0 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output-dir "${RESULT_DIR}"

"${PYTHON_BIN}" feature_eval/linear_eval.py \
  --checkpoint runs/20260516-104713_cifar10_resnet18_aug-strong_proj-on_bs-1024_chengbo_bs1024_e100/checkpoint_0100.pth.tar \
  --name batch_size_1024_20260516 \
  --data "${DATA_DIR}" \
  --epochs "${LINEAR_EPOCHS}" \
  --batch-size "${LINEAR_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LINEAR_LR}" \
  --weight-decay "${LINEAR_WEIGHT_DECAY}" \
  --label-fraction 1.0 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output-dir "${RESULT_DIR}"

"${PYTHON_BIN}" feature_eval/summarize_batch_size.py \
  --result 128="${RESULT_DIR}/batch_size_128_may04.json" \
  --result 256="${RESULT_DIR}/batch_size_256_20260503.json" \
  --result 512="${RESULT_DIR}/batch_size_512_20260506.json" \
  --result 1024="${RESULT_DIR}/batch_size_1024_20260516.json" \
  --n-views 2 \
  --baseline-batch-size 256 \
  --output-dir "${REPORT_DIR}"
