# Jetson Guide (CUDA sidecar + training notes)

This folder keeps Jetson-specific docs and scripts so the main ACRE app stays cross-platform/CPU-first.

## CUDA sidecar (CLI-first)
- `jetson_cuda_setup.sh` — clone/build llama.cpp with CUDA (SM87), download TinyLlama GGUF into `jetson/models/`.
- `jetson_cuda_infer.sh` — one-line CUDA inference via `llama-cli`.
- `jetson_cuda_lora_finetune.sh` — finetune template (full GGUF output; heavy on Jetson).
- `jetson_cuda_menu.sh` — interactive menu wrapping the above.

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

Finetune template (full GGUF, expect to hit memory limits on small Jetsons):
```bash
cd /home/acre/ACRE_Capstone/acre
TRAIN_DATA=/path/to/your.jsonl \
OUT_MODEL=/home/acre/jetson/output/finetuned.gguf \
bash jetson/jetson_cuda_lora_finetune.sh
# or menu option 3
```
Use tiny batch/ctx, expect limited success on 8 GB devices.

Interactive menu:
```bash
cd /home/acre/ACRE_Capstone/acre
bash jetson/jetson_cuda_menu.sh   # 1) setup, 2) inference, 3) finetune
```

## Using the CUDA lib with the GUI (optional)
If you want the desktop app to reuse this CUDA build:
```bash
export LLAMA_CPP_LIB=/home/acre/ACRE_Capstone/acre/jetson/llama.cpp/build/bin/libllama.so
```
Ensure GGUFs are visible to the app (`jetson/models` symlinked to `models/` or copy files). Otherwise keep the GUI CPU-first.

## Jetson training profile (app)
- Jetson defaults: `config/jetson_training.json` + dataset `example_datasets/jetson_training.json`, conservative batch/epochs, gradient accumulation, checkpointing, swap dir `outputs/jetson_training_swap`.
- When Jetson is detected, the training dialog loads that profile and shows this doc path.
- PyTorch on Jetson: install NVIDIA’s JetPack-specific wheels (not PyPI). Use `scripts/jetson_torch_install.py` to print the correct `pip install` line. Re-run after JetPack upgrades.
- CuSPARSELt (optional): `bash install_cusparselt.sh` if you want those kernels.
- Traces/snapshots land under `models/<output-name>` when training finishes. Edit `config/jetson_training.json` to tweak defaults; the dialog reloads on next open.
