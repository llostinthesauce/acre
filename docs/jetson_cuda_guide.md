# Jetson CUDA sidecar (CLI-first)

This keeps the main ACRE desktop app CPU-first/cross-platform, and adds a Jetson-only CUDA toolchain in `jetson/` for people who want full GPU acceleration (inference + LoRA finetune) via llama.cpp.

## What’s included
- `jetson/jetson_cuda_setup.sh` — clone & build llama.cpp with CUDA (SM87), download TinyLlama GGUF, stage everything under `jetson/`.
- `jetson/jetson_cuda_infer.sh` — one-line CUDA inference using the built llama-cli.
- `jetson/jetson_cuda_lora_finetune.sh` — LoRA finetune template using llama.cpp’s finetune binary (default data: `jetson/data/alpaca_tiny.jsonl`).

## Prereqs on Jetson
- JetPack / CUDA toolkit installed (so `/usr/local/cuda/bin` and `/usr/local/cuda/lib64` exist).
- Packages: `git`, `cmake`, build essentials, `wget`.
- (Optional but recommended) Performance mode: `sudo nvpmodel -m 0` and `sudo jetson_clocks`.

## Setup (build + download model)
```bash
cd /home/acre/ACRE_Capstone/acre
bash jetson/jetson_cuda_setup.sh
```
This clones llama.cpp into `jetson/llama.cpp`, builds with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`, and downloads TinyLlama Q4_K_M into `jetson/models/`.
If `models/` doesn’t exist in the repo, the script symlinks `jetson/models` → `models`.

## CUDA inference (CLI)
```bash
cd /home/acre/ACRE_Capstone/acre
PROMPT="Say one short sentence proving CUDA is working on Jetson." \
NGL=999 \
bash jetson/jetson_cuda_infer.sh
# In another shell: sudo tegrastats   # watch GR3D%
# Or use the interactive menu: bash jetson/jetson_cuda_menu.sh (option 2)
```
Env vars you can override: `MODEL`, `PROMPT`, `NGL`, `CTX`, `LLAMA_CPP_LIB`.

## Finetune (template, full GGUF output)
```bash
cd /home/acre/ACRE_Capstone/acre
# edit jetson/data/alpaca_tiny.jsonl or point TRAIN_DATA to your JSONL
TRAIN_DATA=/path/to/your.jsonl \
OUT_MODEL=/home/acre/jetson/output/finetuned.gguf \
bash jetson/jetson_cuda_lora_finetune.sh
# Or use the interactive menu: bash jetson/jetson_cuda_menu.sh (option 3)
```
Adjust hyperparams via env vars: `EPOCHS`, `BATCH`, `CTX`, `NGL`, `LORA_R`, `LORA_ALPHA`, `LR`, `THREADS`, `BASE_MODEL`.

## Interactive menu
Run a simple menu with the common actions:
```bash
cd /home/acre/ACRE_Capstone/acre
bash jetson/jetson_cuda_menu.sh
# 1) setup, 2) CUDA inference, 3) LoRA finetune
```

## Using the CUDA lib with the GUI (optional)
- If you still want the desktop app to use this CUDA build, set:
  ```bash
  export LLAMA_CPP_LIB=/home/acre/ACRE_Capstone/acre/jetson/llama.cpp/build/bin/libllama.so
  ```
- And make sure your GGUFs are visible to the app (either symlink `jetson/models` to `models` or copy the files).
If you prefer to keep the GUI CPU-only, leave it as-is; the sidecar tooling is independent.
