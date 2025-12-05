# Cleanup Plan

- Choose Python 3.11 or 3.12 and rebuild vendor per `deps.md` for macOS (arm64/x86), manylinux_x86_64, and win_amd64. Use `pip download --only-binary=:all:` with a lock file, unzip wheels into `vendor/<platform>/`, and write a manifest (package==version, platform).
- Add the lock file and vendor manifest to the repo; include `platform_utils.py` and the `vendor/` tree in build artifacts.
- Expand and lock `requirements.txt` to cover all imports used by the app (customtkinter, darkdetect, pillow, send2trash, soundfile, pymupdf, protobuf<6, cryptography, numpy<2, llama-cpp-python, torch/torchvision/transformers/tokenizers/safetensors/huggingface_hub/accelerate/diffusers/datasets/auto-gptq/peft, psutil, pynvml, sentencepiece, etc.). Remove runtime pip installs from the app.
- Add the missing Jetson torch helper script (or update docs) so Jetson users can install NVIDIA’s wheels offline.
- Rewrite the Dockerfile for offline use: base on a preloaded image, install tk/libsndfile (and other needed OS libs), copy `platform_utils.py` and the vendor tree, and avoid any PyPI/apt/network calls at build or runtime.
