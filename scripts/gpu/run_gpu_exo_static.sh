#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../config/env.sh
source "${PROJECT_ROOT}/config/env.sh"

BASE="${EDGEVISOR_ENGINE_DIR}"
MODEL="${EDGEVISOR_EXO_MODEL:-${EDGEVISOR_MODEL3}}"
TOK="${EDGEVISOR_EXO_TOKENIZER:-${EDGEVISOR_TOKENIZER}}"
GPU_LIST="${EDGEVISOR_EXO_GPUS:-0,1,2}"
TOTAL_LAYERS="${EDGEVISOR_EXO_TOTAL_LAYERS:-28}"
MEMORY_FIELD="${EDGEVISOR_EXO_MEMORY_FIELD:-total}"
PROMPT="${EDGEVISOR_EXO_PROMPT:-Hi}"
STEPS="${EDGEVISOR_EXO_STEPS:-16}"
MAX_SEQ_LEN="${EDGEVISOR_EXO_MAX_SEQ_LEN:-512}"
BASE_PORT="${EDGEVISOR_EXO_BASE_PORT:-19301}"

IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "EDGEVISOR_EXO_GPUS must contain at least one GPU index" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EDGEVISOR_LOG_ROOT}/gpu/gpu_exo_static_${STAMP}"
mkdir -p "${LOG_DIR}"

PLAN_JSON="${LOG_DIR}/exo_plan.json"
RATIOS="$(
  python3 "${BASE}/src/edgevisor_exo_plan.py" \
    --gpu-indices "${GPU_LIST}" \
    --total-layers "${TOTAL_LAYERS}" \
    --memory-field "${MEMORY_FIELD}"
)"
python3 "${BASE}/src/edgevisor_exo_plan.py" \
  --gpu-indices "${GPU_LIST}" \
  --total-layers "${TOTAL_LAYERS}" \
  --memory-field "${MEMORY_FIELD}" \
  --json >"${PLAN_JSON}"

echo "EDGEVISOR_EXO_GPUS=${GPU_LIST}"
echo "EDGEVISOR_EXO_RATIOS=${RATIOS}"
echo "EDGEVISOR_EXO_PLAN=${PLAN_JSON}"

cd "${BASE}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

WORKER_ARGS=()
ROOT_GPU="${GPUS[0]}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

for ((i = 1; i < ${#GPUS[@]}; i++)); do
  gpu="${GPUS[$i]}"
  port=$((BASE_PORT + i - 1))
  ./dllama worker --port "${port}" --nthreads 1 --gpu-index "${gpu}" >"${LOG_DIR}/worker_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
  WORKER_ARGS+=("127.0.0.1:${port}")
done

sleep 4

CMD=(
  ./dllama inference
  --prompt "${PROMPT}"
  --steps "${STEPS}"
  --model "${MODEL}"
  --tokenizer "${TOK}"
  --buffer-float-type q80
  --nthreads 1
  --max-seq-len "${MAX_SEQ_LEN}"
  --gpu-index "${ROOT_GPU}"
  --ratios "${RATIOS}"
)

if [[ "${#WORKER_ARGS[@]}" -gt 0 ]]; then
  CMD+=(--workers "${WORKER_ARGS[@]}")
fi

"${CMD[@]}" >"${LOG_DIR}/root_gpu${ROOT_GPU}.log" 2>&1
RC=$?

echo "LOG_DIR=${LOG_DIR}"
echo "RC=${RC}"
echo "--- plan ---"
cat "${PLAN_JSON}" || true
echo "--- root tail ---"
tail -n 120 "${LOG_DIR}/root_gpu${ROOT_GPU}.log" || true
for ((i = 1; i < ${#GPUS[@]}; i++)); do
  gpu="${GPUS[$i]}"
  echo "--- worker gpu${gpu} tail ---"
  tail -n 80 "${LOG_DIR}/worker_gpu${gpu}.log" || true
done

exit "${RC}"
