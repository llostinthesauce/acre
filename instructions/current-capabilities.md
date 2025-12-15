# ACRE — Current capabilities

This document describes what the ACRE app does today (as implemented on the `main` branch). It’s intended to be a practical, “what exists right now” reference.

## What ACRE is

ACRE is a local, offline-first desktop GUI for switching between multiple on-device AI workflows:

- **Chat / text generation** (local models)
- **Document analysis** (PDF/text → summary)
- **Image generation** (text → image via diffusion models)
- **OCR** (image → text)
- **ASR** (audio → text transcription)
- **TTS** (text → audio `.wav`)

The “switchboard” idea is: you drop models into `models/`, ACRE detects what type they are, and you can load them from the sidebar without changing the UI or workflow.

## How to run it

The canonical entrypoint is `app.py`, which calls:

- `acre_app.bootstrap.setup_environment()` to configure “offline mode” + dependencies
- `acre_app.ui.run_app()` to start the GUI

Quick start (mirrors `README.md`):

1. Create/activate a virtual environment (recommended).
2. Install Python deps: `pip install -r requirements.txt`
3. Start the app: `python app.py`

## Offline-first behavior (what it currently does)

### “Offline mode” flags

At startup, ACRE sets environment variables to force Hugging Face tooling into offline mode:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_ALLOW_CODE_DOWNLOAD=1`

And model-loading code uses **`local_files_only=True`** for:

- Transformers tokenizers/models
- Diffusers pipelines
- Transformers pipelines (OCR/ASR/TTS)

This means: **models must already be present locally** under `models/` (or referenced locally), and ACRE should not attempt to pull model weights from the network during normal inference.

### Important nuance: dependency installation is still dynamic

`acre_app/bootstrap.py` has a “missing imports” path that can run:

- `python -m pip install -t vendor ...`

If the requested Python packages aren’t importable. This is convenient during development, but it can conflict with a strict “never touch the network” offline guarantee unless:

- the `vendor/` directory already contains everything needed, or
- pip is configured to only install from an offline wheel cache.

`notes/deps.md` and `notes/cleanup.md` describe a stricter, fully-offline vendoring plan.

## Repository layout (what’s used by the app)

- `app.py` — entrypoint
- `acre_app/` — GUI + app logic
- `model_manager/` — model loading, backend selection, inference, history persistence
- `models/` — local model folders/files you can load in the app
- `history/` — per-user, per-model chat histories (optionally encrypted)
- `outputs/` — generated images, TTS audio, benchmarks logs, etc.
- `config/` — `settings.json`, `guardrail_list.json`, Jetson training profile
- `vendor/` — optional “vendored” Python deps used to run offline
- `notes/` — internal docs about dependency/vendoring strategy

## App flow (what you see when you launch)

### 1) Gate screens: account, login, disclaimer

On launch, the GUI builds a “gate” before you get to the main app:

- **First run:** you create an account (username/password).
- **Later runs:** you log in as an existing user (user dropdown + password).
- **Disclaimer:** you must acknowledge a disclaimer before proceeding.
- **Remember me:** optional “remember me on this device for 30 days”.

Where this lives:

- Gate UI: `acre_app/ui.py` (`build_gate_ui`)
- Settings storage: `acre_app/settings.py`
- History encryption key derivation: `acre_app/crypto.py`

What’s stored locally:

- Passwords are stored as a PBKDF2 hash (with per-user salt and iterations) in `config/settings.json`.
- Each user also has an **encryption salt + iterations** used to derive an encryption key for chat history encryption.
- If “Remember me” is enabled, a per-user `remember_key` and `remember_expires` is stored in `config/settings.json`.

### 2) Main workspace UI: models sidebar + tabs

After passing the gate, ACRE builds the main UI:

**Left sidebar: “Models”**

- A listbox of detected models in `models/`
- Buttons:
  - **Refresh** — re-scan and re-render the list
  - **Load Model** — load the selected model into the active backend
  - **Add Model** — import a model file/folder into `models/`
  - **Clear History** — clear history for the currently-loaded model (chat only)
  - **Switch User** — logs out and returns to the gate
- Right-click / context menu:
  - **Rename…** — set a per-user display alias for a model name
  - **Reveal in Finder** — open the model folder/file in the OS file explorer

**Right side: tabs**

- **Chat** — local chat + attachments row
- **Gallery** — browse generated images in `outputs/<user>/`
- **Settings** — configure generation + UI + tools

## Models: adding, listing, and loading

### Where models live

ACRE looks for models under `models/`.

The app supports both:

- **Single GGUF/GGML files** (e.g., `tinyllama-...Q4_K_M.gguf`)
- **Model folders** (Transformers/Diffusers/MLX layouts, GGUF bundles, etc.)

### Importing models into `models/`

Using **Add Model**:

- If you choose “folder import”, ACRE copies the selected folder into `models/`.
- If you choose “single file import”, ACRE copies the selected file into `models/`.

### Loading models (automatic backend selection)

When you load a model, ACRE chooses a backend automatically based on file patterns in the selected file/folder. This logic lives in `model_manager/manager.py`.

Supported backend categories:

#### Text generation

- **llama.cpp** (`llama_cpp-python`) for `.gguf` / `.ggml`
- **Transformers** (“HFBackend”) for local `config.json` models
- **AutoGPTQ** for GPTQ-quantized folders (`quantize_config.json`)
- **MLX-LM** for Apple Silicon MLX model folders (various heuristics)

#### Image generation

- **Diffusers text-to-image** when a folder contains `model_index.json`

#### “Voice” and other pipelines

- **OCR** when a folder looks like vision/image-to-text (e.g., TrOCR)
- **ASR** when the model type indicates Whisper
- **TTS** when model type indicates text-to-speech / speech

#### Vision models

- **Vision/chat** routing exists (e.g., Qwen VL patterns → `PhiVisionBackend`)
- A vision backend supports image+prompt analysis, but the current GUI does not expose this yet (see “Limitations”).

#### Jetson guardrails

If the platform is detected as Jetson (`platform_utils.is_jetson()`):

- The app restricts model loading to **GGUF/GGML (llama.cpp)**.
- Device selection defaults to CPU-only behavior in multiple places.

## Chat: how text generation works

### Sending prompts

In the Chat tab:

- Type in the prompt box and press **Enter** to send.
- **Shift+Enter** inserts a newline (it does not send).

What happens when you send:

- A safety filter checks your message for blocked terms (`config/guardrail_list.json`).
- If a compatible model is loaded, ACRE runs generation on a background thread.
- The chat panel updates when generation completes.

### Safety filter (guardrails)

ACRE loads a list of guardrail terms from:

- `config/guardrail_list.json`

If a prompt includes any blocked terms, the request is rejected and a safety notice is shown in the chat.

### History (per-model, per-user)

If “history enabled” is on:

- Chat is stored per user under: `history/<user>/`
- Each loaded model gets its own history file (filename is “sanitized” from the model name).
- History is appended as `{role: "user"|"assistant", content: ...}` entries.

If history is off:

- Generation runs as a single-turn prompt without adding to stored history.

### Encrypted history at rest

When logged in, ACRE derives a per-user key and uses Fernet encryption for history at rest:

- If encryption is active, history files are stored as encrypted blobs (not readable JSON).
- If an old plaintext history file is detected, ACRE attempts to migrate it to encrypted storage.

### Prompt formatting (chat templates)

For backends that expose a `chat_template` (Transformers/MLX), ACRE tries to apply the model’s chat template via `apply_chat_template(...)` before generation.

It also supports prompt “tags”:

- `/think` and `/no_think` (recognized in template formatting and stripped from the displayed user message)

## Attachments row (multimodal entry points)

The “attachments” row under the chat history changes based on the loaded backend:

- **Text models:** “Analyze Document…” (PDF/text → summary)
- **Diffusion (image generation):** “Diffusion: type prompt → image” (no attachment picker)
- **OCR models:** “Choose Image…” → OCR result appears in chat
- **ASR models:** “Choose Audio…” → transcription appears in chat
- **TTS models:** “TTS: type text → audio (.wav)” (TTS runs when you send a prompt)
- **Vision models:** “Analyze Document…” (image analysis is currently disabled in the GUI)

## Document analysis (PDF/text summarization)

Supported document types:

- `.pdf`
- `.txt`
- `.md` / `.markdown`

How it works:

- PDF text extraction uses PyMuPDF (`fitz`).
- Long documents are chunked by paragraphs (roughly max ~3200 chars per chunk).
- Each chunk gets a short summary, then the chunk summaries get merged into a final summary.
- The app adds a single “Analyzing document: …” user entry and a single assistant summary entry to history (instead of storing all chunk prompts).

## Image generation (diffusion)

If you load a Diffusers text-to-image model folder:

- Sending a prompt runs `generate_image(...)` and saves a `.png` under `outputs/<user>/`.
- The Gallery tab shows the output images with thumbnails.

Image generation settings (defaults) live in the Settings tab:

- width / height
- inference steps
- guidance scale
- seed (blank means “random each run”)

## Gallery (generated images)

The Gallery tab:

- Displays images found in `outputs/<user>/` (png/jpg/jpeg).
- Generates and caches thumbnails in `outputs/<user>/.thumbnails/`.
- Provides actions per image:
  - Open
  - Reveal in folder
  - Delete (send to trash when possible; otherwise asks for permanent delete)

## Voice/audio workflows (current)

What exists today is “file-based voice”:

- **ASR:** pick an audio file → transcription appears in chat.
- **TTS:** type text and send → a `.wav` is generated in `outputs/<user>/` and the saved path is shown in chat.

What does *not* exist yet in the GUI:

- microphone recording / live dictation
- speaker playback controls inside the app
- push-to-talk / VAD / streaming voice chat

## Settings tab (what can be configured)

Settings are persisted to `config/settings.json` (under `"prefs"`), and include:

### Appearance

- Theme selection (multiple built-in color themes)
- App text size
- Chat text size

### Text responses

- Temperature slider
- Max new tokens

### Image generation

- Width / height
- Steps
- Guidance
- Seed

### Interface & performance

- Device preference (`auto`, `mps`, `cuda`, `cpu`; Jetson forces CPU)
- UI scale
- Enable/disable history
- Run a tiered performance test for TinyLlama (prints detailed metrics to the terminal)

### Shortcuts

- Open outputs folder
- Open models folder
- Clear all histories for the current user
- Free VRAM / clear caches (tries backend cleanup + Torch cache clears + GC)

### Diagnostics

Shows local versions / availability for key libraries (and offline env flags), e.g.:

- Python
- torch, transformers, diffusers, llama_cpp, soundfile, protobuf
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`

### Model training (local fine-tuning)

Opens a training dialog that can:

- Load a dataset (JSON/JSONL) or use a built-in demo dataset.
- Fine-tune a compatible Transformers base model using `transformers.Trainer`.
- Save the trained output as a new model folder under `models/`.

## Local benchmarking output

For text generation runs, ACRE writes a lightweight benchmark log:

- `outputs/<user>/benchmarks.txt`

It records model/backend/device, duration, rough token estimate, and TPS estimate.

## Limitations / known gaps (today)

This list is intentionally “current state”, not a roadmap:

- **Strict offline guarantee is not enforced end-to-end** because dependency installation may occur via pip at runtime if modules are missing.
- **Vision image Q&A exists at the backend layer**, but the GUI currently disables image analysis (only document analysis is exposed).
- **Voice is file-based** (ASR via file picker; TTS outputs files). There’s no microphone input or in-app playback controls.
- **No streaming token UI** (generation updates after completion); there is also no “stop/cancel generation” control.
- **Model download is manual** (the app provides instructions but doesn’t manage downloads inside the GUI).
- **Training docs mention Jetson files/scripts that may not exist on this branch** (the UI references `jetson/README.md` and a torch install script).

