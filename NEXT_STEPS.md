# Next Steps

## Jetson Fine-Tuning
- Document and automate torch installation via NVIDIA’s Jetson wheels; include scripts that detect Jetson and point to the correct wheel.
- Add a dedicated training pipeline tailored for Jetson-class GPUs (lower batch sizes, gradient accumulation defaults, swap support).
- Provide sample configs/recipes for fine-tuning GGUF or HF models directly on-device or offload heavy steps to x86 with export utilities.

## Encryption Hardening
- Audit the current remember-me implementation and extend encryption to cover history, attachments, and cached model outputs.
- Expose settings to rotate keys and to enforce per-user passphrases for saved models.
- Add an optional secure enclave integration (macOS Keychain / Linux keyctl) so secrets never touch disk in plaintext.

## Document Upload & Awareness
- Expand the upload pipeline so models can ingest PDFs/Office docs with chunking and metadata tracking.
- Persist embeddings or summaries per document and allow users to scope chats to selected files.
- Surface document provenance in the UI so responses can cite the source.

## Model Weights & Defaults
- Ship curated default weights for common tasks (chat, coding, TTS) and expose presets in the UI.
- Let users pin a model as the default per task type (text/image/audio) and store that preference in `config/settings.json`.
- Include validation that warns when a preset points at missing weights or incompatible hardware.

## Bundled Dependencies
- Pre-build the `vendor/` directory (or wheels) and ship it with releases so the app works fully offline after the first install.
- Provide platform-specific bundles (Intel, Apple Silicon, Jetson) to avoid unsupported wheels being installed automatically.
- Add checksum/manifest verification so updates can skip re-downloading identical packages.
