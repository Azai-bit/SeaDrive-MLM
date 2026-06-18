#!/usr/bin/env bash
set -euo pipefail

# Run on the training server after copying the generated dataset directory.
# Example:
#   DATASET_DIR=/home/ubuntu/ws_myboat2.0/tmp/hf_success_dataset \
#   bash /home/ubuntu/ws_myboat2.0/xtf/LLM/server_finetune_qwen3_vl_colreg.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/tze/LLModels/Qwen/Qwen3-VL-2B-Instruct}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-/home/ubuntu/tze/LLModels/Qwen/Qwen3-VL-2B-COLREG}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/tmp/hf_success_vl_dataset}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/tmp/qwen3_vl_colreg_sft_work}"
ARTIFACT_DIR="${ARTIFACT_DIR:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
MERGE_LORA="${MERGE_LORA:-1}"
MERGE_LORA_DEVICE="${MERGE_LORA_DEVICE:-cpu-reload}"
MIN_LABEL_TOKENS="${MIN_LABEL_TOKENS:-16}"
LABEL_STATS_LOG_STEPS="${LABEL_STATS_LOG_STEPS:-25}"
LABEL_PREFLIGHT_SAMPLES="${LABEL_PREFLIGHT_SAMPLES:-32}"
IMAGE_MAX_SIDE="${IMAGE_MAX_SIDE:-384}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-3136}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
EVAL_GENERATE_SAMPLES="${EVAL_GENERATE_SAMPLES:-0}"
EVAL_GENERATE_MAX_NEW_TOKENS="${EVAL_GENERATE_MAX_NEW_TOKENS:-180}"
REASONING_MAX_WORDS="${REASONING_MAX_WORDS:-24}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_DIR}/train.jsonl" ]]; then
  echo "Dataset train.jsonl does not exist: ${DATASET_DIR}/train.jsonl" >&2
  echo "Generate it first with xtf/LLM/make_success_hf_dataset.py, then copy it to the server." >&2
  exit 1
fi

if [[ -e "$OUTPUT_MODEL_PATH" ]]; then
  echo "Output path already exists: ${OUTPUT_MODEL_PATH}" >&2
  echo "Move or remove it first, or set OUTPUT_MODEL_PATH to a new directory." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python is not available: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [
    name for name in ("torch", "transformers", "datasets", "peft", "accelerate", "PIL")
    if importlib.util.find_spec(name) is None
]
if missing:
    print("Missing Python packages: " + ", ".join(missing))
    raise SystemExit(1)
PY
then
  echo "Install the missing packages in the active training environment before running." >&2
  echo "Typical packages: torch transformers datasets peft accelerate pillow sentencepiece protobuf" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"

echo "Starting Qwen3-VL COLREG fine-tuning"
echo "  base model:   ${MODEL_PATH}"
echo "  dataset dir:  ${DATASET_DIR}"
echo "  output model: ${OUTPUT_MODEL_PATH}"
echo "  work dir:     ${WORK_DIR}"
echo "  artifacts:    ${ARTIFACT_DIR:-${WORK_DIR}/training_results}"
echo "  cuda devices: ${CUDA_VISIBLE_DEVICES}"
echo "  cuda alloc:   ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  epochs:       ${NUM_TRAIN_EPOCHS}"
echo "  max seq len:  ${MAX_SEQ_LENGTH}"
echo "  batch/accum:  ${PER_DEVICE_TRAIN_BATCH_SIZE}/${GRADIENT_ACCUMULATION_STEPS}"
echo "  lr:           ${LEARNING_RATE}"
echo "  warmup steps: ${WARMUP_STEPS}"
echo "  image max:    ${IMAGE_MAX_SIDE}px"
echo "  image pixels: ${IMAGE_MIN_PIXELS}-${IMAGE_MAX_PIXELS}"
echo "  attention:    ${ATTN_IMPLEMENTATION}"
echo "  preflight:    ${LABEL_PREFLIGHT_SAMPLES} samples"
echo "  LoRA:         r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "  merge LoRA:   ${MERGE_LORA}"
echo "  merge device: ${MERGE_LORA_DEVICE}"
echo "  eval samples: ${EVAL_GENERATE_SAMPLES}"
echo "  reasoning max:${REASONING_MAX_WORDS} words"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU memory snapshot before training:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv || true
fi

merge_flag=()
if [[ "$MERGE_LORA" == "0" ]]; then
  merge_flag=(--no-merge-lora)
fi

artifact_flag=()
if [[ -n "$ARTIFACT_DIR" ]]; then
  artifact_flag=(--artifact-dir "$ARTIFACT_DIR")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$PYTHON_BIN" "${SCRIPT_DIR}/finetune_qwen3_vl_colreg.py" \
  --model-path "$MODEL_PATH" \
  --dataset-dir "$DATASET_DIR" \
  --output-model-path "$OUTPUT_MODEL_PATH" \
  --work-dir "$WORK_DIR" \
  "${artifact_flag[@]}" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-steps "$WARMUP_STEPS" \
  --weight-decay "$WEIGHT_DECAY" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --merge-lora-device "$MERGE_LORA_DEVICE" \
  --min-label-tokens "$MIN_LABEL_TOKENS" \
  --label-stats-log-steps "$LABEL_STATS_LOG_STEPS" \
  --label-preflight-samples "$LABEL_PREFLIGHT_SAMPLES" \
  --image-max-side "$IMAGE_MAX_SIDE" \
  --image-min-pixels "$IMAGE_MIN_PIXELS" \
  --image-max-pixels "$IMAGE_MAX_PIXELS" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  --eval-generate-samples "$EVAL_GENERATE_SAMPLES" \
  --eval-generate-max-new-tokens "$EVAL_GENERATE_MAX_NEW_TOKENS" \
  --reasoning-max-words "$REASONING_MAX_WORDS" \
  "${merge_flag[@]}"

echo "Fine-tuning finished."
echo "Model saved at: ${OUTPUT_MODEL_PATH}"
