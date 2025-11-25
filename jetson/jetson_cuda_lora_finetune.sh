#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Quick finetune template using llama.cpp's finetune binary (produces a full finetuned GGUF).
# Adjust paths and hyperparams as needed for your dataset/VRAM.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT}/jetson/llama.cpp"
FINETUNE_BIN="${LLAMA_DIR}/build/bin/llama-finetune"
BASE_MODEL="${BASE_MODEL:-${ROOT}/jetson/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT}/jetson/data/alpaca_tiny.jsonl}"
OUT_MODEL="${OUT_MODEL:-${ROOT}/jetson/output/finetuned.gguf}"
THREADS="${THREADS:-6}"
NGL="${NGL:-999}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-16}"
CTX="${CTX:-2048}"
LR="${LR:-2e-4}"

export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_DIR}/build/bin/libllama.so}"

if [ ! -x "${FINETUNE_BIN}" ]; then
  echo "finetune binary not found at ${FINETUNE_BIN}. Run jetson/jetson_cuda_setup.sh first." >&2
  exit 1
fi

if [ ! -f "${BASE_MODEL}" ]; then
  echo "Base model not found at ${BASE_MODEL}. Set BASE_MODEL or run setup." >&2
  exit 1
fi

if [ ! -f "${TRAIN_DATA}" ]; then
  echo "Training data not found at ${TRAIN_DATA}. Set TRAIN_DATA to your JSONL." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_MODEL}")"

echo "Using LLAMA_CPP_LIB=${LLAMA_CPP_LIB}"
echo "Finetuning model to ${OUT_MODEL}"
set -x
"${FINETUNE_BIN}" \
  --model "${BASE_MODEL}" \
  --file "${TRAIN_DATA}" \
  --output "${OUT_MODEL}" \
  -t "${THREADS}" \
  -b "${BATCH}" \
  -c "${CTX}" \
  -ngl "${NGL}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LR}"
