#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

setup() {
  bash "${ROOT}/jetson/jetson_cuda_setup.sh"
}

infer() {
  read -r -p "Prompt [Say one short sentence proving CUDA is working on Jetson.]: " PROMPT
  read -r -p "Model path [${ROOT}/jetson/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]: " MODEL
  read -r -p "n_gpu_layers (-ngl) [999]: " NGL
  read -r -p "ctx (-c) [2048]: " CTX
  PROMPT=${PROMPT:-"Say one short sentence proving CUDA is working on Jetson."}
  MODEL=${MODEL:-"${ROOT}/jetson/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"}
  NGL=${NGL:-999}
  CTX=${CTX:-2048}
  PROMPT="$PROMPT" MODEL="$MODEL" NGL="$NGL" CTX="$CTX" \
    bash "${ROOT}/jetson/jetson_cuda_infer.sh"
}

finetune() {
  read -r -p "Training data JSONL [${ROOT}/jetson/data/alpaca_tiny.jsonl]: " TRAIN_DATA
  read -r -p "Base model [${ROOT}/jetson/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf]: " BASE_MODEL
  read -r -p "Output model path [${ROOT}/jetson/output/finetuned.gguf]: " OUT_MODEL
  read -r -p "Epochs [1]: " EPOCHS
  read -r -p "Batch size [16]: " BATCH
  read -r -p "n_gpu_layers (-ngl) [999]: " NGL
  read -r -p "ctx (-ctx) [2048]: " CTX
  TRAIN_DATA=${TRAIN_DATA:-"${ROOT}/jetson/data/alpaca_tiny.jsonl"}
  BASE_MODEL=${BASE_MODEL:-"${ROOT}/jetson/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"}
  OUT_MODEL=${OUT_MODEL:-"${ROOT}/jetson/output/finetuned.gguf"}
  EPOCHS=${EPOCHS:-1}
  BATCH=${BATCH:-16}
  NGL=${NGL:-999}
  CTX=${CTX:-2048}
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
