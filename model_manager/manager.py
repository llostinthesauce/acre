from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .backends import (
    ASRBackend,
    AutoGPTQBackend,
    BaseBackend,
    DiffusersT2IBackend,
    HFBackend,
    LlamaCppBackend,
    MLXBackend,
    OCRBackend,
    PhiVisionBackend,
    TTSBackend,
)
from .config import GenerationConfig


class ModelManager:
    SUPPORTED_EXTENSIONS = (".gguf", ".ggml")

    def __init__(self, models_dir: str = "models", history_dir: str = "history", device_pref: str = "auto"):
        self._models_dir = Path(models_dir)
        self._history_dir = Path(history_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._impl: Optional[BaseBackend] = None
        self._backend: Optional[str] = None
        self._current_model_name: Optional[str] = None
        self._history_file: Optional[Path] = None
        self._history: List[dict] = []
        self._config = GenerationConfig()
        self._history_enabled = True
        self._device_pref = device_pref
        self._llama_threads: Optional[int] = None
        self._llama_ctx: Optional[int] = 4096
        self._llama_gpu_layers: Optional[int] = self._suggest_llama_gpu_layers(device_pref)
        self._kind: str = "text"
        self._encryptor: Optional[Any] = None

    def set_encryptor(self, encryptor: Optional[Any]) -> None:
        self._encryptor = encryptor

    def set_history_enabled(self, enabled: bool) -> None:
        self._history_enabled = bool(enabled)

    def set_text_config(self, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> None:
        if max_tokens is not None:
            self._config.max_tokens = int(max_tokens)
        if temperature is not None:
            self._config.temperature = float(temperature)

    def list_models(self) -> List[str]:
        output: List[str] = []
        for item in sorted(self._models_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                output.append(item.name)
            elif item.is_dir():
                if any(
                    child.is_file() and child.suffix.lower() in self.SUPPORTED_EXTENSIONS
                    for child in item.iterdir()
                ):
                    output.append(item.name)
                elif (item / "model_index.json").exists():
                    output.append(item.name)
                elif (item / "quantize_config.json").exists():
                    output.append(item.name)
                elif (item / "config.json").exists():
                    output.append(item.name)
        return output

    def _suggest_llama_gpu_layers(self, pref: str) -> Optional[int]:
        pref = (pref or "auto").lower()
        if pref == "cuda":
            return -1
        if pref == "mps":
            return -1
        if pref != "auto":
            return 0
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return -1
            if torch.cuda.is_available():
                return -1
        except Exception:
            return 0
        return 0

    def _detect_backend(self, candidate: Path) -> Tuple[str, Path, str]:
        if candidate.is_file() and candidate.suffix.lower() in {".gguf", ".ggml"}:
            name = candidate.name.lower()
            if "flux" in name:
                raise ValueError("This model file looks like a FLUX diffusion build; use the Diffusers folder variant.")
            return ("llama_cpp", candidate, "text")
        if candidate.is_dir():
            gguf = sorted(
                child
                for child in candidate.iterdir()
                if child.is_file() and child.suffix.lower() in {".gguf", ".ggml"}
            )
            if gguf:
                if any("flux" in child.name.lower() for child in gguf):
                    raise ValueError("Folder contains FLUX diffusion builds; use a Diffusers model folder instead.")
                return ("llama_cpp", gguf[0], "text")
            if "phi" in candidate.name.lower() and "vision" in candidate.name.lower():
                return ("phi_vision", candidate, "vision")
            if (candidate / "model_index.json").exists():
                return ("diffusers_t2i", candidate, "image")
            if (candidate / "quantize_config.json").exists():
                return ("auto_gptq", candidate, "text")
            if (candidate / "config.json").exists():
                try:
                    config = json.loads((candidate / "config.json").read_text())
                except Exception:
                    config = {}
                architectures = {entry.lower() for entry in config.get("architectures", [])}
                model_type = str(config.get("model_type", "")).lower()
                name_lower = candidate.name.lower()
                quant_info = config.get("quantization") or {}
                quant_conf = config.get("quantization_config") or {}
                if "mlx" in name_lower:
                    return ("mlx_lm", candidate, "text")
                quant_str = f"{quant_info} {quant_conf}".lower()
                if "mlx" in quant_str:
                    return ("mlx_lm", candidate, "text")
                name_lower = candidate.name.lower()
                if "phi" in name_lower and "vision" in name_lower:
                    return ("phi_vision", candidate, "vision")
                if "visionencoderdecodermodel" in architectures or model_type in {"vision-encoder-decoder", "trocr"}:
                    return ("ocr_trocr", candidate, "ocr")
                if ("whisper" in model_type) or any("whisper" in entry for entry in architectures) or model_type in {"speech_to_text", "audio-encoder-decoder"}:
                    return ("asr_whisper", candidate, "asr")
                if model_type in {"text-to-speech", "speecht5", "vits", "parler-tts"} or any("texttospeech" in entry for entry in architectures):
                    return ("tts_transformers", candidate, "tts")
                return ("transformers", candidate, "text")
        raise ValueError(f"Unsupported model location: {candidate}")

    def load_model(self, name: str, *, device_pref: Optional[str] = None) -> Tuple[bool, str]:
        candidate = self._models_dir / name
        if not candidate.exists():
            return (False, f"model not found: {candidate}")
        try:
            backend_type, model_path, kind = self._detect_backend(candidate)
        except ValueError as exc:
            return (False, str(exc))
        self._reset_session()
        target_device = device_pref or self._device_pref
        try:
            if backend_type == "llama_cpp":
                self._impl = LlamaCppBackend(
                    model_path,
                    n_threads=self._llama_threads,
                    n_ctx=self._llama_ctx,
                    n_gpu_layers=self._llama_gpu_layers,
                )
            elif backend_type == "diffusers_t2i":
                self._impl = DiffusersT2IBackend(model_path, target_device)
            elif backend_type == "auto_gptq":
                self._impl = AutoGPTQBackend(model_path, target_device)
            elif backend_type == "transformers":
                self._impl = HFBackend(model_path, target_device)
            elif backend_type == "ocr_trocr":
                self._impl = OCRBackend(model_path, target_device)
            elif backend_type == "asr_whisper":
                self._impl = ASRBackend(model_path, target_device)
            elif backend_type == "tts_transformers":
                self._impl = TTSBackend(model_path, target_device)
            elif backend_type == "mlx_lm":
                self._impl = MLXBackend(model_path, target_device)
            elif backend_type == "phi_vision":
                self._impl = PhiVisionBackend(model_path, target_device or self._device_pref)
            else:
                raise RuntimeError("unknown backend")
        except Exception as exc:
            self._reset_session()
            if self._looks_like_oom(exc):
                self.cleanup_memory()
                message = (
                    "Out of memory while loading this model. Switch the device preference to CPU or pick a smaller build, "
                    "then try again. Caches were cleared."
                )
            else:
                message = str(exc)
                if "VisionEncoderDecoder" in message or "image-to-text" in message:
                    message = "This model is an OCR (image→text) model; use the OCR pipeline."
            return (False, message)
        self._backend = self._impl.name
        self._current_model_name = name
        self._kind = kind
        self._history_file = self._history_dir / f"{self._safe_filename(name)}.json"
        self._history = self._load_history() if (self._kind == "text" and self._history_enabled) else []
        return (True, str(model_path))

    def unload(self) -> None:
        if self._impl:
            try:
                self._impl.unload()
            except Exception:
                pass
        self._reset_session()
        self.cleanup_memory()

    def is_loaded(self) -> bool:
        return self._impl is not None

    def cleanup_memory(self) -> None:
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    @property
    def backend(self) -> str:
        return self._backend or "unloaded"

    @property
    def current_model_name(self) -> Optional[str]:
        return self._current_model_name

    def is_image_backend(self) -> bool:
        return self._kind == "image"

    def is_ocr_backend(self) -> bool:
        return self._kind == "ocr"

    def is_asr_backend(self) -> bool:
        return self._kind == "asr"

    def is_tts_backend(self) -> bool:
        return self._kind == "tts"

    def is_vision_backend(self) -> bool:
        return self._kind == "vision"

    def generate(self, user_prompt: str) -> str:
        if not self._impl:
            raise RuntimeError("No model loaded.")
        if self._kind != "text":
            raise RuntimeError("Loaded model is not a text generator.")
        snapshot = list(self._history)
        self._history.append({"role": "user", "content": user_prompt})
        use_template = hasattr(self._impl, "format_chat")
        stop_backup: Optional[Tuple[str, ...]] = None
        prompt: str
        if use_template:
            prompt = self._format_with_template()
            if prompt:
                stop_backup = self._config.stop
                self._config.stop = ()
            else:
                use_template = False
        if not use_template:
            prompt = self._build_prompt_plain() if self._history_enabled else user_prompt
        try:
            text = self._impl.generate(prompt, self._config)
            text = self._postprocess_response(text)
            if self._history_enabled:
                self._history.append({"role": "assistant", "content": text})
                self._save_history()
            else:
                self._history = snapshot
            return text
        except Exception as exc:
            self._history = snapshot
            if self._looks_like_oom(exc):
                self.cleanup_memory()
                raise RuntimeError(
                    "Generation failed: out of memory. Lower max tokens, disable chat history, or run Free VRAM from Settings."
                ) from exc
            raise RuntimeError(f"Generation failed: {exc}") from exc
        finally:
            if stop_backup is not None:
                self._config.stop = stop_backup

    def run_ocr(self, image_path: str) -> str:
        if not self._impl or not self.is_ocr_backend():
            raise RuntimeError("No OCR model loaded.")
        runner = getattr(self._impl, "run", None)
        if not callable(runner):
            raise RuntimeError("Loaded model does not support OCR.")
        return runner(image_path)

    def run_asr(self, audio_path: str) -> str:
        if not self._impl or not self.is_asr_backend():
            raise RuntimeError("No ASR model loaded.")
        runner = getattr(self._impl, "run", None)
        if not callable(runner):
            raise RuntimeError("Loaded model does not support ASR.")
        return runner(audio_path)

    def run_tts(self, text: str, outdir: Path) -> str:
        if not self._impl or not self.is_tts_backend():
            raise RuntimeError("No TTS model loaded.")
        runner = getattr(self._impl, "run", None)
        if not callable(runner):
            raise RuntimeError("Loaded model does not support TTS.")
        return runner(text, outdir)

    def generate_image(self, prompt: str, **kwargs) -> str:
        if not self._impl:
            raise RuntimeError("No model loaded.")
        if not hasattr(self._impl, "generate_image"):
            raise RuntimeError("Loaded model does not support image generation.")
        return self._impl.generate_image(prompt, **kwargs)

    def analyze_image(self, image_path: str, question: str) -> str:
        if not self._impl or not self.is_vision_backend():
            raise RuntimeError("No vision model loaded.")
        analyzer = getattr(self._impl, "analyze_image", None)
        if not callable(analyzer):
            raise RuntimeError("Loaded model does not support image analysis.")
        return analyzer(Path(image_path), question, self._config)

    def get_history(self) -> List[dict]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history = []
        if self._history_file and self._history_file.exists():
            try:
                self._history_file.unlink()
            except OSError:
                pass

    def add_history_entry(self, role: str, content: str) -> None:
        if self._kind != "text":
            return
        entry = {"role": role, "content": content}
        self._history.append(entry)
        if self._history_enabled:
            self._save_history()

    def _reset_session(self) -> None:
        self._impl = None
        self._backend = None
        self._current_model_name = None
        self._history_file = None
        self._history = []
        self._kind = "text"

    def describe_session(self) -> str:
        if not self._impl or not self._current_model_name:
            return "No model loaded."
        details = [
            f"Model: {self._current_model_name}",
            f"Backend: {self._backend or 'unknown'}",
        ]
        inspector = getattr(self._impl, "runtime_info", None)
        runtime: dict[str, Any] = {}
        if callable(inspector):
            try:
                runtime = inspector() or {}
            except Exception:
                runtime = {}
        device = runtime.get("device") or self._device_pref or "cpu"
        details.append(f"Device: {device}")
        dtype = runtime.get("dtype")
        if dtype:
            details.append(f"Precision: {dtype}")
        context = runtime.get("max_position_embeddings") or runtime.get("tokenizer_max_length")
        if context:
            details.append(f"Context limit: {context} tokens")
        details.append(f"Max tokens: {self._config.max_tokens}")
        details.append(f"Temperature: {self._config.temperature}")
        details.append(f"History: {'on' if self._history_enabled else 'off'}")
        return "\n".join(details)

    def _looks_like_oom(self, exc: Exception) -> bool:
        try:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        message = str(exc).lower()
        if not message:
            return False
        for token in ("out of memory", "mps backend", "unable to allocate", "cuda error 2"):
            if token in message:
                return True
        return False

    def _load_history(self) -> List[dict]:
        if not self._history_file or not self._history_file.exists():
            return []
        try:
            blob = self._history_file.read_bytes()
        except Exception:
            return []
        if not blob:
            return []
        if self._encryptor:
            try:
                plaintext = self._encryptor.decrypt(blob)
                data = json.loads(plaintext.decode("utf-8"))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            try:
                fallback_text = blob.decode("utf-8")
                data = json.loads(fallback_text)
                if isinstance(data, list):
                    try:
                        payload = self._encryptor.encrypt(fallback_text.encode("utf-8"))
                        self._history_file.write_bytes(payload)
                    except Exception:
                        pass
                    return data
            except Exception:
                return []
            return []
        try:
            text = blob.decode("utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            return []
        return []

    def _save_history(self) -> None:
        if not self._history_file:
            return
        try:
            payload = json.dumps(self._history, ensure_ascii=False, indent=2).encode("utf-8")
            if self._encryptor:
                payload = self._encryptor.encrypt(payload)
                self._history_file.write_bytes(payload)
            else:
                self._history_file.write_text(payload.decode("utf-8"), encoding="utf-8")
        except Exception:
            pass

    def _build_prompt_plain(self) -> str:
        parts = []
        for message in self._history:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _safe_filename(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", name)

    def _format_with_template(self) -> str:
        formatter = getattr(self._impl, "format_chat", None)
        if not callable(formatter):
            return ""
        try:
            formatted = formatter(self._history)
        except Exception:
            return ""
        return formatted or ""

    def _postprocess_response(self, text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"<\s*(?:start|end)\s*[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<\|[^>]*\|>", "", cleaned)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        filtered: List[str] = []
        last_norm = ""
        repeats = 0
        for sentence in sentences:
            candidate = sentence.strip()
            if not candidate:
                continue
            normalized = candidate.lower()
            if normalized == last_norm:
                repeats += 1
                if repeats >= 1:
                    continue
            else:
                repeats = 0
            filtered.append(candidate)
            last_norm = normalized
        filtered_text = " ".join(filtered).strip()
        lines = [line.strip() for line in filtered_text.splitlines() if line.strip()]
        return "\n".join(lines)
