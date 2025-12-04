#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find_gguf_models() {
  local search_paths=()
  [ -d "${ROOT}/models" ] && search_paths+=("${ROOT}/models")
  [ -d "${ROOT}/jetson/models" ] && search_paths+=("${ROOT}/jetson/models")
  if [ "${#search_paths[@]}" -eq 0 ]; then
    return 0
  fi
  find "${search_paths[@]}" -maxdepth 2 -type f -name '*.gguf' 2>/dev/null | LC_ALL=C sort -u
}

setup() {
  bash "${ROOT}/jetson/jetson_cuda_setup.sh"
}

infer() {
  mapfile -t GGUF_MODELS < <(find_gguf_models)
  local MODEL=""

  if [ "${#GGUF_MODELS[@]}" -gt 0 ]; then
    echo "Available GGUF models:"
    local idx
    for idx in "${!GGUF_MODELS[@]}"; do
      printf "  %d) %s\n" $((idx + 1)) "${GGUF_MODELS[$idx]}"
    done
    read -r -p "Select model number (1-${#GGUF_MODELS[@]}) or enter a .gguf path: " selection
    if [ -z "${selection}" ]; then
      echo "Model selection is required." >&2
      return
    fi
    if [[ "${selection}" =~ ^[0-9]+$ ]]; then
      local sel_idx=$((selection))
      if [ "${sel_idx}" -lt 1 ] || [ "${sel_idx}" -gt "${#GGUF_MODELS[@]}" ]; then
        echo "Selection out of range." >&2
        return
      fi
      MODEL="${GGUF_MODELS[$((sel_idx - 1))]}"
    else
      MODEL="${selection}"
    fi
  else
    echo "No .gguf models found under models/ or jetson/models/."
    read -r -p "Enter a path to a .gguf model: " MODEL
  fi

  if [ -z "${MODEL}" ]; then
    echo "Please provide a .gguf model path." >&2
    return
  fi

  if [[ "${MODEL}" != *.gguf ]]; then
    echo "Model must be a .gguf file (got: ${MODEL})." >&2
    return
  fi

  if [ ! -f "${MODEL}" ]; then
    echo "Model not found at ${MODEL}." >&2
    return
  fi

  read -r -p "Prompt [Say one short sentence proving CUDA is working on Jetson.]: " PROMPT
  read -r -p "n_gpu_layers (-ngl) [999]: " NGL
  read -r -p "ctx (-c) [2048]: " CTX
  PROMPT=${PROMPT:-"Say one short sentence proving CUDA is working on Jetson."}
  NGL=${NGL:-999}
  CTX=${CTX:-2048}

  PROMPT="$PROMPT" MODEL="$MODEL" NGL="$NGL" CTX="$CTX" \
    bash "${ROOT}/jetson/jetson_cuda_infer.sh"
}

finetune() {
  mapfile -t GGUF_MODELS < <(find_gguf_models)
  local BASE_MODEL=""

  if [ "${#GGUF_MODELS[@]}" -gt 0 ]; then
    echo "Available GGUF models:"
    local idx
    for idx in "${!GGUF_MODELS[@]}"; do
      printf "  %d) %s\n" $((idx + 1)) "${GGUF_MODELS[$idx]}"
    done
    read -r -p "Select base model number (1-${#GGUF_MODELS[@]}) or enter a .gguf path: " selection
    if [ -z "${selection}" ]; then
      echo "Base model selection is required." >&2
      return
    fi
    if [[ "${selection}" =~ ^[0-9]+$ ]]; then
      local sel_idx=$((selection))
      if [ "${sel_idx}" -lt 1 ] || [ "${sel_idx}" -gt "${#GGUF_MODELS[@]}" ]; then
        echo "Selection out of range." >&2
        return
      fi
      BASE_MODEL="${GGUF_MODELS[$((sel_idx - 1))]}"
    else
      BASE_MODEL="${selection}"
    fi
  else
    echo "No .gguf models found under models/ or jetson/models/."
    read -r -p "Enter a path to a .gguf base model: " BASE_MODEL
  fi

  read -r -p "Training data JSONL [${ROOT}/jetson/data/alpaca_tiny.jsonl]: " TRAIN_DATA
  read -r -p "Output model path [${ROOT}/jetson/output/finetuned.gguf]: " OUT_MODEL
  read -r -p "Epochs [1]: " EPOCHS
  read -r -p "Batch size [16]: " BATCH
  read -r -p "n_gpu_layers (-ngl) [999]: " NGL
  read -r -p "ctx (-ctx) [2048]: " CTX
  TRAIN_DATA=${TRAIN_DATA:-"${ROOT}/jetson/data/alpaca_tiny.jsonl"}
  OUT_MODEL=${OUT_MODEL:-"${ROOT}/jetson/output/finetuned.gguf"}
  EPOCHS=${EPOCHS:-1}
  BATCH=${BATCH:-16}
  NGL=${NGL:-999}
  CTX=${CTX:-2048}

  if [ -z "${BASE_MODEL}" ]; then
    echo "Please provide a .gguf base model path." >&2
    return
  fi
  if [[ "${BASE_MODEL}" != *.gguf ]]; then
    echo "Base model must be a .gguf file (got: ${BASE_MODEL})." >&2
    return
  fi
  if [ ! -f "${BASE_MODEL}" ]; then
    echo "Base model not found at ${BASE_MODEL}." >&2
    return
  fi

  TRAIN_DATA="$TRAIN_DATA" BASE_MODEL="$BASE_MODEL" OUT_MODEL="$OUT_MODEL" \
  EPOCHS="$EPOCHS" BATCH="$BATCH" NGL="$NGL" CTX="$CTX" \
    bash "${ROOT}/jetson/jetson_cuda_lora_finetune.sh"
}

while true; do
  echo
  echo "Jetson CUDA menu:"
  echo "1) Setup (build llama.cpp + download model)"
  echo "2) Run CUDA inference"
  echo "3) Run finetune (GGUF output)"
  echo "q) Quit"
  read -r -p "Choose an option: " choice
  case "$choice" in
    1) setup ;;
    2) infer ;;
    3) finetune ;;
    q|Q) exit 0 ;;
    *) echo "Invalid choice" ;;
  esac
done
