#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT}/jetson/llama.cpp"
BIN="${LLAMA_DIR}/build/bin/llama-cli"
MODEL="${MODEL:-}"
NGL="${NGL:-999}"
CTX="${CTX:-2048}"
PROMPT="${PROMPT:-Say one short sentence proving this Jetson run is using CUDA.}"

export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_DIR}/build/bin/libllama.so}"

if [ -z "${MODEL}" ]; then
  echo "MODEL is required. Set MODEL to a .gguf file path or use jetson/jetson_cuda_menu.sh option 2." >&2
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
  echo "Model not found at ${MODEL}. Adjust MODEL env var or run setup." >&2
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
