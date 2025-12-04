#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFERRED_MODELS=(
  "${ROOT}/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
  "${ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B-Q2_K_L.gguf"
  "${ROOT}/models/qwen2.5-0.5b-instruct-q3_k_m.gguf"
  "${ROOT}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

find_gguf_models() {
  local search_paths=()
  [ -d "${ROOT}/models" ] && search_paths+=("${ROOT}/models")
  [ -d "${ROOT}/jetson/models" ] && search_paths+=("${ROOT}/jetson/models")
  if [ "${#search_paths[@]}" -eq 0 ]; then
    return 0
  fi
  local results=()
  local add_unique
  add_unique() {
    local candidate="$1"
    local existing
    for existing in "${results[@]}"; do
      if [ "${existing}" = "${candidate}" ]; then
        return
      fi
    done
    results+=("${candidate}")
  }
  local preferred
  for preferred in "${PREFERRED_MODELS[@]}"; do
    if [ -f "${preferred}" ]; then
      add_unique "${preferred}"
    fi
  done
  while IFS= read -r path; do
    [ -n "${path}" ] && add_unique "${path}"
  done < <(find "${search_paths[@]}" -maxdepth 2 -type f -name '*.gguf' 2>/dev/null | LC_ALL=C sort -u)
  printf "%s\n" "${results[@]}"
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
    echo "No .gguf models found under models/ or jetson/models/ (expected one of: ${PREFERRED_MODELS[*]})."
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

while true; do
  echo
  echo "Jetson CUDA menu:"
  echo "1) Setup (build llama.cpp + download model)"
  echo "2) Run CUDA inference"
  echo "q) Quit"
  read -r -p "Choose an option: " choice
  case "$choice" in
    1) setup ;;
    2) infer ;;
    q|Q) exit 0 ;;
    *) echo "Invalid choice" ;;
  esac
done
