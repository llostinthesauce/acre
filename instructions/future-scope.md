# ACRE — Next steps & future scope (privacy-first offline multimodal)

This document is a forward-looking roadmap for turning ACRE into a **multi-platform, privacy-first, offline local AI GUI that “just works”**—including multimodal workflows (voice/vision/docs) and an optional **local OpenAI-compatible API server**.

It is intentionally thorough and grouped by workstream so you can pick a path and execute in phases.

---

## North star (product definition)

ACRE becomes a local-first AI desktop app with:

- **Zero cloud dependency by default**: no telemetry, no background downloads, no network calls unless explicitly requested.
- **Multi-platform**: macOS (Apple Silicon + Intel), Windows (x64), Linux (x64) as first-class targets; Jetson as a supported “edge” target with clear constraints.
- **Multi-modal**: text chat, doc upload & Q/A, image generation, OCR, ASR, TTS, and vision (image Q/A).
- **Reliability over novelty**: predictable installs, clear errors, good defaults, graceful fallbacks, simple mental model.
- **Interoperability**: optional **local OpenAI-compatible endpoints** so other tools can use ACRE as a backend.

---

## Design principles (privacy-first “just works”)

1. **Offline means offline**: if a user disconnects Wi‑Fi, everything still works (except optional explicit download flows).
2. **Explicit network consent**: any network access is an explicit user action with visible status and a cancel button.
3. **Data stays on disk, encrypted by default**:
   - user data separation
   - at-rest encryption for histories and libraries
   - no logs containing prompts unless the user opts in
4. **Deterministic builds**: the exact same inputs create the exact same shipped artifact (and dependency set).
5. **Predictable performance**: device preference is respected; GPU/CPU usage is visible; OOM is handled gracefully.
6. **Safe-by-default model execution**: models that require custom code (`trust_remote_code`) are treated as untrusted and surfaced with clear warnings/controls.

---

## Phase plan (recommended sequencing)

### Phase 0 — Stabilize + make offline real (foundation)

Goal: run on a clean machine with **no network** and no “pip at runtime”.

- Remove runtime `pip install` behavior; replace with a hard failure + “missing dependency” guidance.
- Ship a complete vendored dependency set per platform (or packaged executable) so imports never touch PyPI.
- Add a smoke-test harness that verifies:
  - app starts
  - each backend can load a tiny local model and run one inference
  - no outbound connections attempted (when network is disabled)

### Phase 1 — UX + multimodal polish (daily-driver)

Goal: smooth, responsive UI with strong multimodal workflows that feel “native”.

- Streaming generation + stop/cancel + progress indicators.
- Proper attachment UX for docs/images/audio (drag-drop, paste, “recent”, per-message attachments).
- Playback/recording in-app for voice features (not just file pickers).

### Phase 2 — Local OpenAI-compatible server mode (interop)

Goal: expose `http://localhost:.../v1/...` endpoints that other tools can hit.

- Start/stop server from the GUI.
- Implement core endpoints for chat/completions + streaming.
- Expand to embeddings, image generation, audio transcribe, TTS.

### Phase 3 — “Offline knowledge base” (docs that scale)

Goal: local document library + search + citations.

- Local ingestion + chunking + OCR + metadata.
- Local embeddings + vector search.
- “Ask my docs” mode with references.

### Phase 4 — Hardening + packaging for broad distribution

Goal: installers + auto-update (offline-capable) + strong security posture.

---

## Workstreams & detailed backlog

### 1) Offline dependency strategy (must-have)

Current state: the app is offline-oriented for model loading, but Python dependencies may still be installed dynamically.

Next steps:

- **Decide on a shipping strategy**
  - Option A (Python app): ship per-platform `vendor/<platform>/` and ensure `sys.path` prioritizes it.
  - Option B (packaged executable): ship via PyInstaller/Nuitka/Briefcase with all deps embedded.
- **Lock dependencies**
  - introduce a `requirements.lock` (fully pinned versions)
  - generate per-platform wheels ahead of time
- **Vendor verification**
  - add a script that validates “all imports used by the app resolve from the shipped set”
  - add a manifest (package==version, hashes)
- **Network kill switch**
  - optional: add a runtime “no network” guard (e.g., disable HF hub, block sockets, or detect and warn)
- **OS-level deps**
  - document native dependencies (e.g., Tk, audio libs, GPU runtimes) per platform

### 2) Model lifecycle & compatibility (must-have)

Goal: model loading is obvious, reliable, and self-documenting.

- **Model registry UI**
  - show model type (text/diffusion/ocr/asr/tts/vision)
  - show required backend + dependencies
  - show approximate RAM/VRAM requirements (heuristics ok)
  - show recommended device preference
- **Import UX**
  - drag-drop model folders/files into the sidebar
  - detect duplicates and handle upgrades
  - “Verify model” action (checks required files exist; runs a one-token test)
- **Safe handling of `trust_remote_code`**
  - display a clear warning when a model requires custom code
  - provide a “never run model code” global toggle
  - ideally run those models in an isolated subprocess
- **Conversion helpers (optional)**
  - GGUF conversion guidance or one-click tooling (where feasible)
  - quantization recommendations and compatibility notes

### 3) Core chat UX (must-have)

Goal: ACRE feels like a polished chat app even before multimodal expansion.

- **Streaming tokens**
  - show tokens as they generate
  - add “Stop” and “Regenerate” buttons
  - add “Copy”, “Save”, “Export chat”
- **Session management**
  - conversation list (not just per-model history)
  - per-conversation system prompt / presets
  - import/export conversations
- **Reliability**
  - generation queue (avoid overlapping requests)
  - consistent error formatting (OOM, missing deps, invalid model)

### 4) Multimodal: documents (must-have)

Goal: doc upload becomes a first-class feature (not a “summary button”).

- **Document ingestion**
  - drag-drop PDFs, text, markdown
  - directory import (“add a folder of docs”)
  - track metadata (filename, source, timestamps)
- **Scanned PDFs**
  - automatic OCR fallback when PDF text extraction is empty
  - allow choosing OCR model and quality/speed presets
- **Ask-the-doc**
  - answer questions with citations (chunk references)
  - optional summary-first + drill-down pattern
- **Local knowledge base**
  - embeddings generation (local model)
  - vector index (SQLite + FAISS-like, or a simple local vector store)
  - per-user encrypted storage

### 5) Multimodal: voice (must-have)

Goal: “voice in, voice out” is smooth and offline.

**Voice input**

- microphone capture (push-to-talk + optional VAD)
- live waveform / recording indicator
- ASR with local Whisper-style models
- automatic punctuation and optional diarization (later)

**Voice output**

- in-app playback controls (play/pause/seek)
- voice selection (per TTS model) + speed/pitch controls if supported
- caching TTS outputs per message

**Voice chat mode**

- “hands-free” conversation loop (ASR → LLM → TTS)
- safe interruption handling (barge-in)

### 6) Multimodal: vision (must-have)

Goal: load a vision model and ask questions about images locally.

- **Attachment support**
  - per-message image attachments (not just a global “analyze” button)
  - drag-drop, paste-from-clipboard, and “open camera” capture
- **Vision Q/A UI**
  - ask: “What’s in this image?” or “Extract table text”
  - show the image thumbnail inline in chat
  - allow multiple images per turn (later)
- **Model compatibility**
  - explicitly support common local vision chat families (Qwen-VL, LLaVA variants, Phi-vision, etc.)
  - unify prompt formatting and image preprocessing

### 7) Multimodal: image generation (should-have)

Goal: diffusion feels like a product feature, not a demo.

- progress indicator and cancel
- prompt history and reuse
- seed management + “variations”
- optional img2img/inpainting if supported by the selected pipeline
- gallery improvements (tags, search, batch export)

### 8) Local OpenAI-compatible API server mode (must-have per your request)

Goal: ACRE can run a local service so other apps can talk to it like OpenAI.

**Core design**

- Server is **off by default**; user explicitly enables it.
- Binds to `127.0.0.1` by default; optional LAN binding with warnings.
- Has basic auth/token support (even on localhost) to prevent casual cross-app abuse.

**Endpoints to implement**

- `GET /v1/models` — list loaded/available models and their capabilities
- `POST /v1/chat/completions` — main chat endpoint
  - support streaming (`stream=true`) via SSE
  - support system/user/assistant roles
  - support tool/function calling in the response format (later)
- `POST /v1/completions` — legacy completions compatibility (optional)
- `POST /v1/embeddings` — required for RAG and interop
- `POST /v1/images/generations` — text-to-image
- `POST /v1/audio/transcriptions` — ASR
- `POST /v1/audio/speech` — TTS

**Mapping to ACRE internals**

- Create a stable internal “capabilities API” so the server can:
  - select an appropriate model (or require a model id)
  - route to the correct backend (text/image/audio/vision)
  - enforce concurrency limits and cancellation
- Support per-user separation:
  - separate histories and storage
  - optional per-user API tokens

**Operational UX**

- Settings panel: server port, bind address, auth token, “start on launch”
- Status indicator: running, requests in flight, last error
- Logs: local-only, redact prompts by default

### 9) Privacy & security (must-have)

Goal: the app is credibly “privacy-first”.

- **At-rest encryption everywhere it matters**
  - histories (already present) + document library + embeddings index
  - optional encryption for generated outputs
- **Secrets handling**
  - store “remember me” keys in OS keychain/credential vault instead of plaintext JSON
  - provide “lock app” / “require password on open”
- **Network transparency**
  - “Network activity” panel showing any attempted connections
  - explicit “Download models…” flow that shows URLs and bytes
- **Model code risk**
  - warnings and allowlists for `trust_remote_code`
  - sandboxed execution (subprocess + restricted environment) where feasible

### 10) Performance & hardware support (must-have)

Goal: the same UI scales from laptop CPU to GPU rigs to edge devices.

- **Better device selection**
  - clear “Auto / CPU / GPU” explanation per platform
  - show active device per loaded model
  - graceful GPU fallback on OOM (auto-switch to CPU where possible)
- **Streaming + cancellation**
  - prevents UI lockups
  - allows long multimodal tasks without frustration
- **Profiling and benchmarking**
  - keep the existing perf test
  - add “quick benchmark” per loaded model and store results locally

### 11) Packaging & distribution (must-have for multi-platform)

Goal: a non-developer can install and run ACRE offline.

- **macOS**
  - signed app bundle (DMG)
  - MLX support for Apple Silicon, MPS fallback where relevant
- **Windows**
  - installer (MSI/EXE) with bundled deps
  - GPU detection and clear CUDA requirements
- **Linux**
  - AppImage or distro packages
  - handle Tk + audio libs (document clearly)
- **Jetson**
  - clear supported path (likely GGUF/llama.cpp first)
  - explicit torch install path if Transformers are supported later

### 12) Testing & CI (should-have, becomes must-have for release)

- unit tests for:
  - model detection
  - history encryption/migration
  - document extraction/chunking
  - safety guardrail matching
- integration “smoke tests” per backend (tiny models)
- CI matrix for mac/windows/linux (build artifact + run smoke test)

---

## Definition of “done” (success criteria)

ACRE meets the “privacy-first offline GUI that just works” bar when:

- A clean install can run **fully offline** with no surprise installs/downloads.
- A user can:
  - load a text model and chat (streaming + cancel)
  - upload a doc and ask questions with citations
  - attach an image and ask vision questions (with a supported local model)
  - record audio and get transcription; get spoken responses with playback
  - generate images and manage them in-gallery
- Optional server mode exposes a functional **OpenAI-compatible** API locally for:
  - chat/completions (including streaming)
  - embeddings
  - images
  - audio transcribe + speech
- No data leaves the machine unless the user explicitly enables a network action.

