# Jetson Training Guide

## 1. Prepare Jetson dependencies

1. JetPack ships with custom PyTorch wheels that are not served through PyPI, so install the matching build for your device before training. Run `python scripts/jetson_torch_install.py` to detect your JetPack version and copy the suggested `pip install` command (the script points at the NVIDIA redist index for JetPack 5.1+ by default).  
2. Once PyTorch, torchvision, and torchaudio are installed, run `bash install_cusparselt.sh` if you plan to use CuSPARSELt-accelerated kernels—the script already packaged in this repo deploys the correct libs for the card.
3. If you ever upgrade JetPack, rerun the script or visit the official NVIDIA Jetson download page to pick the wheel that matches your new CUDA/JetPack combination:  
   `https://developer.download.nvidia.com/compute/redist/jp/`

## 2. Jetson training profile

- The Jetson-focused configuration lives in `config/jetson_training.json`. It wires the provided dataset (`example_datasets/jetson_training.json`), conservative defaults (1 epoch, batch size 1, 2.5e-4 learning rate), and `training_arguments` that enable gradient accumulation (`4` steps), checkpointing, and single-worker loading so the GPU memory stays under control.
- A `swap_directory` (`outputs/jetson_training_swap`) is declared so you can monitor temporary state if you want to extend the profile. The application creates the directory on demand.
- The profile also exposes a short `notes` string that surfaces in the UI to remind you what the Jetson defaults prioritize.

## 3. Launching model training

1. Start the app and click **Train Model** in the sidebar. When a Jetson device is detected, the dialog:
   - Autofills the dataset with `example_datasets/jetson_training.json`.
   - Sets epochs, learning rate, and batch size to the profile defaults.
   - Shows the profile note and the `docs/jetson_training.md` path so you can reference this guide while training.
2. Choose a base Hugging Face model (must already be downloaded under `models/`), adjust the output name, and hit **Start Training**.
3. The Jetson profile routes the Hugging Face `Trainer` through gradient accumulation/checkpointing and ensures the CPU vs. GPU decision respects the configured device preference (`auto`, `cuda`, etc.). Out-of-memory risks become less likely because the profile keeps GPU usage minimal.

## 4. Tracking progress and next steps

- Updated model snapshots land in `models/<output-name>`. The UI refreshes the model list when training completes.
- If you need to tweak the defaults, edit `config/jetson_training.json` (epoch/batch defaults or additional `training_arguments`). The dialog reloads the new values the next time it opens.
- Keep this guide handy if you rebuild Jetson drivers or swap JetPack versions—rerun `scripts/jetson_torch_install.py` whenever you need a fresh wheel recommendation.
