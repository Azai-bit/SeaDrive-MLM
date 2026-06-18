#!/usr/bin/env bash
set -euo pipefail

# Run this script on the server, for example:
#   bash /home/ubuntu/xtf/LLM/server_qwen35_api_test.sh
#
# It starts an OpenAI-compatible vLLM API server for the local Qwen model.
# Send test requests from local_test_qwen35_api.py on the client machine.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/tze/LLModels/Qwen/Qwen3-VL-2B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-2b-instruct-local}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen35_2b_vllm_api.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/qwen35_2b_vllm_api.pid}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-180}"
WAIT_LOG_INTERVAL_SECONDS="${WAIT_LOG_INTERVAL_SECONDS:-10}"

API_BASE="http://127.0.0.1:${PORT}"

cleanup() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping API server pid=${pid}"
      kill "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available in PATH" >&2
  exit 1
fi

if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("vllm") else 1)
PY
then
  echo "Python package 'vllm' is not installed in the current environment." >&2
  echo "Activate the environment that contains vLLM, or install it before running this script." >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use. Set PORT=8001 or stop the existing service." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "Starting vLLM OpenAI-compatible API server"
echo "  model: ${MODEL_PATH}"
echo "  host:  ${HOST}"
echo "  port:  ${PORT}"
echo "  name:  ${SERVED_MODEL_NAME}"
echo "  log:   ${LOG_FILE}"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES}"
echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  flashinfer sampler: ${VLLM_USE_FLASHINFER_SAMPLER}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  >"$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
SERVER_PID="$(cat "$PID_FILE")"

echo "Waiting for API server readiness..."
deadline=$((SECONDS + MAX_WAIT_SECONDS))
last_wait_log=$SECONDS
until curl -fsS "${API_BASE}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "API server process exited before becoming ready." >&2
    echo "Last 120 log lines:" >&2
    tail -n 120 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "API server did not become ready within ${MAX_WAIT_SECONDS}s." >&2
    echo "Last 80 log lines:" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if (( SECONDS - last_wait_log >= WAIT_LOG_INTERVAL_SECONDS )); then
    echo "Still waiting for API server readiness... elapsed=$((SECONDS - (deadline - MAX_WAIT_SECONDS)))s"
    echo "Recent log lines:"
    tail -n 8 "$LOG_FILE" || true
    last_wait_log=$SECONDS
  fi
  sleep 2
done

echo "Server is ready."
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Local API base: ${API_BASE}/v1"
echo "Remote API base: http://SERVER_IP:${PORT}/v1"
echo "Keep this script running while testing from the local machine."

wait "$SERVER_PID"
