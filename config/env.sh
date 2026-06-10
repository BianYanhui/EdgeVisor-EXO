#!/usr/bin/env bash
# Shared runtime configuration for EdgeVisor-EXO scripts.
# Override any value from the environment before invoking a script.

if [[ -z "${EDGEVISOR_PROJECT_ROOT:-}" ]]; then
  EDGEVISOR_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

_edgevisor_default_model3="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
_edgevisor_default_tokenizer="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
if [[ ! -f "${_edgevisor_default_model3}" ]]; then
  _edgevisor_default_model3="/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
fi
if [[ ! -f "${_edgevisor_default_tokenizer}" ]]; then
  _edgevisor_default_tokenizer="/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
fi

export EDGEVISOR_PROJECT_ROOT
export EDGEVISOR_ENGINE_DIR="${EDGEVISOR_ENGINE_DIR:-${EDGEVISOR_PROJECT_ROOT}/EdgeVisor}"
export EDGEVISOR_MODEL3="${EDGEVISOR_MODEL3:-${_edgevisor_default_model3}}"
export EDGEVISOR_TOKENIZER="${EDGEVISOR_TOKENIZER:-${_edgevisor_default_tokenizer}}"
export EDGEVISOR_LOG_ROOT="${EDGEVISOR_LOG_ROOT:-${EDGEVISOR_PROJECT_ROOT}/runtime_logs}"

# EdgeVisor-EXO defaults: pipeline-only, memory-proportional layer split.
export EDGEVISOR_EXO_GPUS="${EDGEVISOR_EXO_GPUS:-0,1,2}"
export EDGEVISOR_EXO_TOTAL_LAYERS="${EDGEVISOR_EXO_TOTAL_LAYERS:-28}"
export EDGEVISOR_EXO_MEMORY_FIELD="${EDGEVISOR_EXO_MEMORY_FIELD:-total}"

export CPATH="${EDGEVISOR_PROJECT_ROOT}/tools/vulkan_deps/root/usr/include${CPATH:+:${CPATH}}"
export PATH="${EDGEVISOR_PROJECT_ROOT}/tools/vulkan_deps/root/usr/bin:${PATH}"
export LIBRARY_PATH="${EDGEVISOR_PROJECT_ROOT}/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${EDGEVISOR_PROJECT_ROOT}/tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
