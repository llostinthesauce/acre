# Dependency & Vendor Plan

This project must run fully offline. We ship a frozen vendor tree so the GUI never hits PyPI at runtime.

## Scope (what we vendor)
- Core/UI: customtkinter, darkdetect, Pillow, send2trash, soundfile, pymupdf, cryptography, protobuf<6, numpy<2.
- Text (llama.cpp): llama-cpp-python (CPU wheels per platform).
- Transformers stack: torch (CPU/CUDA per platform except Jetson), torchvision, transformers, tokenizers, safetensors, huggingface_hub.
- Diffusers stack: diffusers, accelerate.
- MLX (Apple-only): mlx-lm.
- LoRA/peft/datasets are intentionally **not** shipped in the GUI (training lives in jetson scripts only).
- Jetson note: torch must come from NVIDIA’s wheels, not PyPI.

## Layout we ship
```
vendor/
  README.md
  macosx_arm64/      # unpacked wheels or wheels; offline
  macosx_x86_64/
  manylinux2014_x86_64/
  win_amd64/
  jetson/README.md   # explains how to install NVIDIA torch wheels
```

## Bootstrap expectations
- Detect platform/arch → pick the matching vendor subdir.
- Append that subdir to `sys.path` first; never attempt network installs.
- If a required package is missing from vendor, print a clear “missing from vendor for <platform>” message and exit instead of hitting PyPI.

## Building the vendor tree (done on clean hosts)
1) Start from a locked file (future: `requirements.lock`) with exact versions above.
2) For each platform, run (example for mac arm64):
```
pip download -r requirements.lock \
  --only-binary=:all: \
  --platform macosx_11_0_arm64 \
  --dest build/macosx_arm64
```
3) Option A: commit wheels directly. Option B: unzip wheels into `vendor/<platform>/` so import works without pip.
4) Repeat for macosx_11_0_x86_64, manylinux2014_x86_64, win_amd64.
5) Jetson: do **not** vendor torch; add `jetson/README.md` pointing to NVIDIA’s wheel URL/command.

## Regeneration script (add later)
- `scripts/build_vendor.sh` should: clear `vendor/<platform>`, run `pip download`, unzip wheels, and write a short manifest (package==version, platform).
- `scripts/verify_vendor.py` should import-check the vendored modules for the current platform to catch missing deps early.

## Repo hygiene
- Remove `vendor/` from `.gitignore`; add `.gitattributes` if needed to keep LF endings.
- Keep `config/settings.json` ignored (per-user), but commit vendor contents.

## Docs to surface
- Add a short `vendor/README.md` explaining platform folders and Jetson torch instructions.
- Note in main README: “Offline-only; all Python deps are bundled under vendor/. Jetson users must install NVIDIA’s torch wheels separately.”
