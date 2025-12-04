#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT}/jetson/llama.cpp"
BIN="${LLAMA_DIR}/build/bin/llama-cli"
PREFERRED_MODELS=(
  "${ROOT}/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
  "${ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B-Q2_K_L.gguf"
  "${ROOT}/models/qwen2.5-0.5b-instruct-q3_k_m.gguf"
  "${ROOT}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)
MODEL="${MODEL:-}"
NGL="${NGL:-999}"
CTX="${CTX:-2048}"
PROMPT="${PROMPT:-Say one short sentence proving this Jetson run is using CUDA.}"

export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_DIR}/build/bin/libllama.so}"

choose_default_model() {
  local candidate
  for candidate in "${PREFERRED_MODELS[@]}"; do
    if [ -f "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  local search_paths=()
  [ -d "${ROOT}/models" ] && search_paths+=("${ROOT}/models")
  [ -d "${ROOT}/jetson/models" ] && search_paths+=("${ROOT}/jetson/models")
  if [ "${#search_paths[@]}" -eq 0 ]; then
    return 1
  fi
  candidate=$(find "${search_paths[@]}" -maxdepth 2 -type f -name '*.gguf' 2>/dev/null | LC_ALL=C sort | head -n 1)
  if [ -n "${candidate}" ]; then
    echo "${candidate}"
    return 0
  fi
  return 1
}

DEFAULT_MODEL="$(choose_default_model || true)"
MODEL="${MODEL:-${DEFAULT_MODEL}}"

if [ -z "${MODEL}" ]; then
  echo "MODEL is required. Place one of the preferred GGUFs (Llama-3.2-1B-Q8_0, DeepSeek-R1-Distill-Qwen-1.5B-Q2_K_L, qwen2.5-0.5b-q3_k_m, TinyLlama Q4_K_M) in models/ or set MODEL to an existing .gguf path." >&2
  exit 1
fi

if [[ "${MODEL}" != *.gguf ]]; then
  echo "MODEL must point to a .gguf file (got: ${MODEL})." >&2
  exit 1
fi

if [ ! -x "${BIN}" ]; then
  echo "llama-cli not found at ${BIN}. Run jetson/jetson_cuda_setup.sh first." >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "Model not found at ${MODEL}. Place a GGUF in models/ (e.g., ${PREFERRED_MODELS[*]}) or set MODEL to an existing .gguf path." >&2
  exit 1
fi

echo "Using LLAMA_CPP_LIB=${LLAMA_CPP_LIB}"
echo "Running inference with -ngl ${NGL}, -c ${CTX}"
set -x
"${BIN}" \
  -m "${MODEL}" \
  -ngl "${NGL}" \
  -c "${CTX}" \
  -p "${PROMPT}"
