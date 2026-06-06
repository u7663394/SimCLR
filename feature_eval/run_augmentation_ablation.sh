#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Activate the intended Python environment first,
# for example: conda activate pytorch_env

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-./datasets_local}"
RUN_ROOT="${RUN_ROOT:-runs/augmentation_ablation_e100}"
RESULT_DIR="${RESULT_DIR:-feature_eval/results/augmentation_ablation_e100}"
REPORT_DIR="${REPORT_DIR:-feature_eval/report/augmentation_ablation_e100}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-256}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-100}"
LINEAR_BATCH_SIZE="${LINEAR_BATCH_SIZE:-256}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
CHECKPOINT_NAME="$(printf 'checkpoint_%04d.pth.tar' "${PRETRAIN_EPOCHS}")"

mkdir -p "${RUN_ROOT}" "${RESULT_DIR}" "${REPORT_DIR}"

run_pretrain() {
  local augmentation="$1"
  local checkpoint_path="${RUN_ROOT}/${augmentation}/${CHECKPOINT_NAME}"

  if [[ -f "${checkpoint_path}" ]]; then
    echo "Skip pretraining ${augmentation}: ${checkpoint_path} exists"
    return
  fi

  "${PYTHON_BIN}" run.py \
    -data "${DATA_DIR}" \
    -dataset-name cifar10 \
    --arch resnet18 \
    --out_dim 128 \
    --epochs "${PRETRAIN_EPOCHS}" \
    --batch-size "${PRETRAIN_BATCH_SIZE}" \
    --temperature 0.07 \
    --seed "${SEED}" \
    --n-views 2 \
    --lr 0.0003 \
    --weight-decay 1e-4 \
    --workers "${WORKERS}" \
    --augmentation "${augmentation}" \
    --experiment-name "${augmentation}" \
    --output-dir "${RUN_ROOT}"
}

run_linear_eval() {
  local augmentation="$1"
  local checkpoint_path="${RUN_ROOT}/${augmentation}/${CHECKPOINT_NAME}"

  "${PYTHON_BIN}" feature_eval/linear_eval.py \
    --checkpoint "${checkpoint_path}" \
    --name "${augmentation}" \
    --data "${DATA_DIR}" \
    --epochs "${LINEAR_EPOCHS}" \
    --batch-size "${LINEAR_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --lr 0.001 \
    --weight-decay 0 \
    --label-fraction 1.0 \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --output-dir "${RESULT_DIR}"
}

for augmentation in baseline no_blur no_color_jitter no_grayscale; do
  run_pretrain "${augmentation}"
  run_linear_eval "${augmentation}"
done

"${PYTHON_BIN}" feature_eval/summarize_augmentation.py \
  --train-summary baseline="${RUN_ROOT}/baseline/summary.json" \
  --train-summary no_blur="${RUN_ROOT}/no_blur/summary.json" \
  --train-summary no_color_jitter="${RUN_ROOT}/no_color_jitter/summary.json" \
  --train-summary no_grayscale="${RUN_ROOT}/no_grayscale/summary.json" \
  --linear-result baseline="${RESULT_DIR}/baseline.json" \
  --linear-result no_blur="${RESULT_DIR}/no_blur.json" \
  --linear-result no_color_jitter="${RESULT_DIR}/no_color_jitter.json" \
  --linear-result no_grayscale="${RESULT_DIR}/no_grayscale.json" \
  --output-dir "${REPORT_DIR}"
