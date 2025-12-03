#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT}/jetson/llama.cpp"
MODEL_DIR="${ROOT}/jetson/models"
MODEL_FILE="${MODEL_DIR}/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
CUDA_ARCH="${CUDA_ARCH:-87}"  # Orin = 87

export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "== Jetson CUDA setup =="
echo "Repo root: ${ROOT}"
echo "llama.cpp dir: ${LLAMA_DIR}"
echo "Model will be stored at: ${MODEL_FILE}"

mkdir -p "${MODEL_DIR}"

if [ -d "${LLAMA_DIR}/.git" ]; then
  echo "-- Updating existing llama.cpp clone"
  git -C "${LLAMA_DIR}" pull --ff-only
else
  echo "-- Cloning llama.cpp"
  git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
fi

echo "-- Configuring llama.cpp with CUDA (arch=${CUDA_ARCH})"
cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

echo "-- Building llama.cpp"
cmake --build "${LLAMA_DIR}/build" -j"$(nproc)"

if [ ! -f "${MODEL_FILE}" ]; then
  echo "-- Downloading TinyLlama Q4_K_M"
  wget -O "${MODEL_FILE}" "${MODEL_URL}"
else
  echo "-- Model already present at ${MODEL_FILE}"
fi

if [ -e "${ROOT}/models" ]; then
  echo "-- models/ already exists in repo; not touching it."
else
  echo "-- Linking jetson/models -> repo models/"
  ln -s "${MODEL_DIR}" "${ROOT}/models"
fi

echo "Done. Test inference with: ${ROOT}/jetson/jetson_cuda_infer.sh"
echo "If you want the app to reuse this build, set:"
echo "  export LLAMA_CPP_LIB=${LLAMA_DIR}/build/bin/libllama.so"
