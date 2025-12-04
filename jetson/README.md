# Jetson Guide (CUDA sidecar + training notes)

This folder keeps Jetson-specific docs and scripts so the main ACRE app stays cross-platform/CPU-first.

## CUDA sidecar (CLI-first)
- `jetson_cuda_setup.sh` — clone/build llama.cpp with CUDA (SM87), download TinyLlama GGUF into `jetson/models/`.
- `jetson_cuda_infer.sh` — one-line CUDA inference via `llama-cli` (expects `models/Llama-3.2-1B-Instruct-Q8_0.gguf` unless `MODEL` is set).
- `jetson_cuda_menu.sh` — interactive menu wrapping setup + inference.

Prereqs: JetPack/CUDA installed (`/usr/local/cuda/bin/lib64`), `git`, `cmake`, `wget`. Optional perf mode: `sudo nvpmodel -m 0` and `sudo jetson_clocks`.

Setup + model download:
```bash
cd /home/acre/ACRE_Capstone/acre
bash jetson/jetson_cuda_setup.sh
```

CUDA inference:
```bash
cd /home/acre/ACRE_Capstone/acre
PROMPT="Say one short sentence proving CUDA is working on Jetson." NGL=999 \
bash jetson/jetson_cuda_infer.sh
# or: bash jetson/jetson_cuda_menu.sh (option 2)
# watch GPU: sudo tegrastats
```
Env overrides: `MODEL`, `PROMPT`, `NGL`, `CTX`, `LLAMA_CPP_LIB`.

Interactive menu:
```bash
cd /home/acre/ACRE_Capstone/acre
bash jetson/jetson_cuda_menu.sh   # 1) setup, 2) inference
```

## GUI on Jetson (CPU-only)
The desktop GUI is forced to CPU on Jetson (no CUDA). Do not export `LLAMA_CPP_LIB` for the GUI; keep GGUFs under `models/` for CPU inference.

## Jetson training profile (app)
- Jetson defaults: `config/jetson_training.json` + dataset `example_datasets/jetson_training.json`, conservative batch/epochs, gradient accumulation, checkpointing, swap dir `outputs/jetson_training_swap`.
- When Jetson is detected, the training dialog loads that profile and shows this doc path.
- PyTorch on Jetson: install NVIDIA’s JetPack-specific wheels (not PyPI). Use `scripts/jetson_torch_install.py` to print the correct `pip install` line. Re-run after JetPack upgrades.
- CuSPARSELt (optional): `bash install_cusparselt.sh` if you want those kernels.
- Traces/snapshots land under `models/<output-name>` when training finishes. Edit `config/jetson_training.json` to tweak defaults; the dialog reloads on next open.

## Defaults and quick recommendations
- App text defaults: temperature = 0.7, max_tokens = 512 (see `model_manager/GenerationConfig` and `config/settings.json` prefs). Llama.cpp context defaults to 4096 in the manager; on Jetson, keep context smaller (1024–2048) to save memory.
- Llama.cpp CLI: use `-ngl 999` to offload all layers; lower `-c` if you hit OOM; batch stays modest by default in the scripts.

Recommended GGUFs by device size (Jetson-friendly):
- 8 GB Jetson (Orin Nano/low-end): TinyLlama 1.1B Chat Q4_K_M (`~0.6 GB`), context 1024–2048, `-ngl 999`.
- 8–16 GB Jetson (Orin NX/AGX trimmed): Phi-2 2.7B Q4_K_M (`~1.6 GB`), context 1024–2048, `-ngl 999`.
- 16 GB+ (AGX Xavier/Orin with headroom): Llama 3.1/3.2 3B Instruct Q4_K_M (`~2.2 GB`), context 2048 (maybe 3072 if memory allows), `-ngl 999`.

If in doubt, start with TinyLlama Q4_K_M, context 1024–2048, and measure GR3D%/memory with `sudo tegrastats`. Adjust temperature (0.6–0.8) and max_tokens (256–512) to keep latency predictable.

Note: On Jetson, the app only supports GGUF/GGML models (llama.cpp). Transformers/HF checkpoints (e.g., SmolLM, Qwen) are filtered out in the GUI; convert them to GGUF if needed.
