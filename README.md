# Offline LLM Switcher

Tkinter desktop app for running and swapping between large language models entirely offline.  
Models are executed locally via [ctransformers](https://github.com/marella/ctransformers) (llama.cpp under the hood), so no network calls or hosted APIs are required.

## Features
- Dark, glassy desktop UI tuned for standalone/offline use.
- Browse and load local `.gguf`, `.bin`, or `.safetensors` models.
- Maintain an offline model registry (`models/registry.json`) for friendly names and metadata.
- Register new models directly from the GUI (copies weights into `models/` and prompts for the architecture).
- Choose between the lightweight ctransformers backend (GGUF/bin) or a transformers+torch backend for native safetensors.
- Password-gated startup with PBKDF2 + Fernet encryption for chat transcripts.
- Per-model conversation history resumes automatically; clear/reset with one click.
- Threaded text generation so the UI stays responsive during inference.

## Backends at a Glance
| Backend          | Weight format        | Dependencies              | Best for                          |
|------------------|----------------------|---------------------------|-----------------------------------|
| `ctransformers`  | `.gguf`, `.bin`      | `ctransformers` only      | Quantized llama.cpp models (fast) |
| `transformers`   | `.safetensors` (+cfg)| `torch`, `transformers`   | Native HF repositories            |

Pick the backend per model when you register it. Both options run fully offline once the wheels and weights are on disk.

## Quick Start
1. **Create a Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   - If you plan to run raw `.safetensors` weights, pre-download the matching `torch` and `transformers` wheels on an online machine and install them alongside the base requirements.

2. **Add model weights**
   - Download the desired GGUF/quantized files on a machine with internet access.
   - Copy them onto the target offline system (USB, external drive, etc.).
   - Place them in the `models/` directory. You can drop them there manually or use the **Add Model** button inside the app to copy from another location.
   - The registry is updated automatically, but you can also edit `models/registry.json` manually if you prefer.
   - For safetensors, grab the entire Hugging Face folder (config, tokenizer, weights) so it can run without hitting the network.

3. **Launch the GUI**
   ```bash
   python app.py
   ```

4. **Authenticate, switch models, and chat**
   - On first launch you'll be prompted to set the master password (no recovery—store it safely).
   - Select a model in the list, click **Load Model**, then type prompts in the chat window.
   - Responses stream back in the main pane; the status panel shows which backend and architecture are active.
   - Conversations auto-save per model (encrypted in `history/`) once the workspace is unlocked.

## Architecture Hints
When registering a model you must provide the ctransformers `model_type` (architecture).  
Here are common options for the models you mentioned:

| Model family        | Recommended `model_type` |
|---------------------|--------------------------|
| Llama 3.x / Llama 2 | `llama`                  |
| "ChatGPT OSS" (Qwen)| `qwen2`                  |
| DeepSeek Coder      | `deepseek` (or `starcoder` for some builds) |
| Mistral / Mixtral   | `mistral`                |

If you leave the field blank, the app tries to infer the type from the filename (falls back to `llama`).

## Security & Storage
- Master password hashes live in `config/settings.json` (PBKDF2-SHA256, 480k iterations, random salt).
- Chat transcripts are stored per model in `history/*.chat` as Fernet-encrypted blobs keyed off the master password.
- Clearing chat removes the encrypted file for that model; without the password, transcripts remain unreadable.
- To rotate the password, delete `config/settings.json` and the `history/` folder (you will lose access to existing chats).
- To verify persistence, load a model, send a message, close the window, then relaunch and load the same model—your last conversation will be restored from the encrypted history file.

## Adding More Models
- Inside the app, click **Add Model** and pick the local `.gguf`, `.bin`, or `.safetensors` asset you previously downloaded.
- Give it a friendly display name when prompted, then provide the architecture (e.g., `llama`, `qwen2`, `deepseek`); leave blank to let the app guess from the filename.
- Pick the execution backend: `ctransformers` for quantized GGUF/bin (fast, low-memory) or `transformers` for full safetensors (requires `torch` + `transformers`).
- GGUF/bin assets are copied in directly; safetensors models copy their entire Hugging Face directory (config, tokenizer, weights) into `models/` so they remain fully offline.
- The registry updates automatically, and the new model appears in the list ready to load.
- Prefer to manage things manually? Drop files into `models/` yourself and edit `models/registry.json` to add or tweak entries.

## Offline Registry
- The app stores metadata in `models/registry.json`.  
- Removing an entry from the file (or using the GUI in future updates) will hide a model without deleting the weight file.  
- Deleting a model file requires manual cleanup—models are intentionally never removed automatically.

## Roadmap
- Package the full environment into a Docker/Podman container for offline deployment on Unix systems.
- Add model deletion/rename tools and richer metadata editing in the GUI.
- Surface local hardware stats (RAM/VRAM) to help decide which model quantization to load.

## Notes
- Keep large model binaries out of version control—`.gitignore` already excludes the `models/` directory except for the registry.
- ctransformers must be installed with CPU/GPU support relevant to the hardware you plan to run on. Pre-compiled wheels exist for most platforms; offline installation requires fetching the wheel ahead of time.
- For safetensors, also stage `torch` and `transformers` wheels that match your platform/architecture before you go offline.
- For offline installs, download wheels for `ctransformers`, `cryptography`, optional `torch`, optional `transformers`, and transfer them alongside your model weights.
